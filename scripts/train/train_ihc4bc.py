#!/usr/bin/env python3
"""
Train a unified multi-stain BioFieldStain on IHC4BC stains (HER2, Ki67, ER, PR).

A single model conditioned on a stain-type embedding. The class embedding
(nn.Embedding(5, 256)) is repurposed as a stain embedding:
    0 = HER2, 1 = Ki67, 2 = ER, 3 = PR, 4 = null (CFG dropout)

Usage:
    python scripts/train/train_ihc4bc.py --data_dir /path/to/IHC4BC_Compressed \
        --split_manifest /path/to/ihc4bc_pgvms_splits.json
"""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import WandbLogger

from biofield_stain.models.trainer import BioFieldStainTrainer
from biofield_stain.data.ihc4bc_dataset import IHC4BCMultiStainCropDataModule


class ValMetricsJSONLCallback(pl.Callback):
    """Append aggregated validation metrics to a JSONL file after each val epoch."""

    def __init__(self, output_path: str):
        self.output_path = Path(output_path)
        self.output_path.parent.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _to_number(value):
        if value is None:
            return None
        if hasattr(value, "detach"):
            value = value.detach()
        if hasattr(value, "item"):
            try:
                return float(value.item())
            except (TypeError, ValueError):
                pass
        try:
            return float(value)
        except (TypeError, ValueError):
            return None

    def on_validation_epoch_end(self, trainer, pl_module):
        if trainer.sanity_checking:
            return

        metrics = trainer.callback_metrics
        record = {
            "epoch": int(trainer.current_epoch),
            "epoch_1based": int(trainer.current_epoch) + 1,
            "global_step": int(trainer.global_step),
        }

        for key, value in metrics.items():
            if isinstance(key, str) and key.startswith("val/"):
                numeric = self._to_number(value)
                if numeric is not None:
                    record[key] = numeric

        with self.output_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print(f"[val-jsonl] appended metrics to {self.output_path}")


def main():
    parser = argparse.ArgumentParser(description='Train BioFieldStain on IHC4BC')
    parser.add_argument('--data_dir', type=str, required=True,
                        help='Path to IHC4BC_Compressed root directory')
    parser.add_argument('--split_manifest', type=str, required=True,
                        help='Explicit IHC4BC train/val/test manifest. Use the paired 1,000-pair test split.')
    parser.add_argument('--stains', nargs='+', default=['HER2', 'Ki67', 'ER', 'PR'],
                        help='Stains to include (default: all 4)')
    parser.add_argument('--ckpt_dir', type=str, default='checkpoints/mist_multistain',
                        help='Checkpoint save directory')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size (16 for 80GB A100, 8 for 40GB)')
    parser.add_argument('--max_epochs', type=int, default=100,
                        help='Max epochs')
    parser.add_argument('--wandb_name', type=str, default='mist_multistain',
                        help='Wandb run name')
    parser.add_argument('--resume_from', type=str, default=None,
                        help='Resume from checkpoint path')
    parser.add_argument('--precision', type=str, default='16-mixed',
                        help='Lightning precision string, e.g. 16-mixed or bf16-mixed')
    parser.add_argument('--check_val_every_n_epoch', type=int, default=1,
                        help='Run full validation every N epochs (default: 1)')
    parser.add_argument('--val_metrics_jsonl', type=str, default=None,
                        help='Append validation summaries to this JSONL file (default: <ckpt_dir>/val_metrics.jsonl)')
    parser.add_argument('--early_stop_patience', type=int, default=12,
                        help='Early stopping patience measured in validation checks (default: 12)')
    parser.add_argument('--early_stop_monitor', type=str, default='val/lpips',
                        help='Metric monitored by checkpointing and early stopping (default: val/lpips)')
    parser.add_argument('--disable_early_stop', action='store_true',
                        help='Disable early stopping while keeping checkpointing enabled')
    parser.add_argument('--innovation_mode',
                        choices=['baseline', 'biofield', 'disentangle', 'atlas', 'full'],
                        default='baseline',
                        help='Configuration switch: baseline=BioFieldStain; biofield=explicit DAB expression field; disentangle=morphology-expression regularizer; atlas=in-batch pathology atlas prototype; full=all innovations')
    parser.add_argument('--expr_field_weight', type=float, default=None,
                        help='BioField L1 weight for predicted DAB expression map; default depends on innovation_mode')
    parser.add_argument('--expr_pearson_weight', type=float, default=None,
                        help='BioField Pearson loss weight; default depends on innovation_mode')
    parser.add_argument('--expr_iod_weight', type=float, default=None,
                        help='BioField integrated-expression loss weight; default depends on innovation_mode')
    parser.add_argument('--disentangle_weight', type=float, default=None,
                        help='Morphology-expression disentanglement weight; default depends on innovation_mode')
    parser.add_argument('--atlas_weight', type=float, default=None,
                        help='In-batch pathology atlas prototype loss weight; default depends on innovation_mode')
    parser.add_argument('--atlas_topk', type=int, default=3,
                        help='Number of same-stain nearest in-batch prototypes for atlas loss')
    parser.add_argument('--innovation_warmup_steps', type=int, default=2000,
                        help='Linearly ramp innovation losses over this many optimizer steps; baseline losses are unchanged')
    parser.add_argument('--biofield_adapter', action='store_true',
                        help='Enable BioField-gated marker-specific residual stain adapter')
    parser.add_argument('--biofield_adapter_hidden', type=int, default=32,
                        help='Hidden channels for the BioField gated adapter')
    parser.add_argument('--biofield_adapter_scale', type=float, default=0.15,
                        help='Maximum residual scale for the BioField gated adapter')
    parser.add_argument('--biofield_adapter_reg_weight', type=float, default=None,
                        help='L1 penalty on adapter residual magnitude; default is 0.02 when adapter is enabled')
    parser.add_argument('--biofield_adapter_freeze_backbone', action='store_true',
                        help='Freeze the main generator and train only expression head / BioField adapter')
    parser.add_argument('--stain_experts', action='store_true',
                        help='Enable marker-specific decoder expert routing inside the generator')
    parser.add_argument('--stain_expert_hidden', type=int, default=32,
                        help='Hidden channels for each marker-specific decoder expert')
    parser.add_argument('--stain_expert_scale', type=float, default=0.10,
                        help='Residual scale for marker-specific decoder experts')
    parser.add_argument('--stain_expert_layers', type=str, default='4,3,2',
                        help='Comma-separated decoder expert layers, choose from 5,4,3,2,1')
    parser.add_argument('--stain_experts_freeze_backbone', action='store_true',
                        help='Freeze the shared generator backbone and train only marker-specific experts')
    parser.add_argument('--init_from_checkpoint', type=str, default=None,
                        help='Initialize weights from a checkpoint with strict=False, without resuming optimizer/scheduler state')
    args = parser.parse_args()

    if args.check_val_every_n_epoch < 1:
        raise ValueError('--check_val_every_n_epoch must be >= 1')
    if args.resume_from and args.init_from_checkpoint:
        raise ValueError('Use either --resume_from for exact training resume or --init_from_checkpoint for non-strict initialization, not both.')

    biofield_enabled = args.innovation_mode in ('biofield', 'full')
    disentangle_enabled = args.innovation_mode in ('disentangle', 'full')
    atlas_enabled = args.innovation_mode in ('atlas', 'full')
    # Conservative defaults: keep the verified BioFieldStain objective dominant.
    expr_field_weight = args.expr_field_weight if args.expr_field_weight is not None else (0.05 if biofield_enabled else 0.0)
    expr_pearson_weight = args.expr_pearson_weight if args.expr_pearson_weight is not None else (0.01 if biofield_enabled else 0.0)
    expr_iod_weight = args.expr_iod_weight if args.expr_iod_weight is not None else (0.01 if biofield_enabled else 0.0)
    # The config already uses he_edge_weight=0.5; keep extra disentangle off unless explicitly ablated.
    disentangle_weight = args.disentangle_weight if args.disentangle_weight is not None else 0.0
    atlas_weight = args.atlas_weight if args.atlas_weight is not None else (0.02 if atlas_enabled else 0.0)
    biofield_adapter_reg_weight = (
        args.biofield_adapter_reg_weight
        if args.biofield_adapter_reg_weight is not None
        else (0.02 if args.biofield_adapter else 0.0)
    )
    stain_expert_layers = tuple(
        int(x.strip()) for x in args.stain_expert_layers.split(',') if x.strip()
    )
    invalid_layers = sorted(set(stain_expert_layers) - {5, 4, 3, 2, 1})
    if invalid_layers:
        raise ValueError(f'Unsupported --stain_expert_layers values: {invalid_layers}')

    val_metrics_jsonl = args.val_metrics_jsonl or str(Path(args.ckpt_dir) / 'val_metrics.jsonl')

    print("=" * 70)
    print("TRAINING: BioField-Stain on IHC4BC")
    print(f"  Stains: {args.stains}")
    print(f"  IHC4BC split manifest: {args.split_manifest}")
    print(f"  Validation every: {args.check_val_every_n_epoch} epoch(s)")
    print(f"  Validation JSONL: {val_metrics_jsonl}")
    print(f"  Checkpoint/EarlyStop monitor: {args.early_stop_monitor}")
    print(f"  Innovation mode: {args.innovation_mode}")
    print(f"  Innovation weights: expr_field={expr_field_weight}, expr_pearson={expr_pearson_weight}, expr_iod={expr_iod_weight}, disentangle={disentangle_weight}, atlas={atlas_weight}, atlas_topk={args.atlas_topk}, warmup_steps={args.innovation_warmup_steps}")
    print(f"  BioField gated adapter: enabled={args.biofield_adapter}, hidden={args.biofield_adapter_hidden}, scale={args.biofield_adapter_scale}, reg={biofield_adapter_reg_weight}, freeze_backbone={args.biofield_adapter_freeze_backbone}")
    print(f"  Marker-specific decoder experts: enabled={args.stain_experts}, hidden={args.stain_expert_hidden}, scale={args.stain_expert_scale}, layers={stain_expert_layers}, freeze_backbone={args.stain_experts_freeze_backbone}")
    if args.init_from_checkpoint:
        print(f"  Initialize from checkpoint: {args.init_from_checkpoint}")
    if args.disable_early_stop:
        print("  Early stopping: disabled")
    else:
        print(f"  Early stopping patience: {args.early_stop_patience} validation check(s)")
    print("=" * 70)

    # Paper hyperparameters
    model = BioFieldStainTrainer(
        # Architecture (identical to BCI except no class conditioning)
        num_classes=5,          # 4 stains + null
        null_class=4,
        class_dim=64,
        uni_dim=1024,
        ndf=64,
        input_skip=True,
        edge_encoder='v2',
        edge_base_ch=32,
        uni_spatial_size=32,    # 32x32 patch tokens from UNI
        label_names=['HER2', 'Ki67', 'ER', 'PR'],
        # Optimizer
        gen_lr=1e-4,
        disc_lr=4e-4,
        warmup_steps=1000,
        # Loss weights (paper configuration)
        lpips_weight=1.0,
        lpips_256_weight=0.5,
        lpips_512_weight=0.0,
        he_edge_weight=0.5,
        l1_lowres_weight=1.0,
        adversarial_weight=0.0,
        uncond_disc_weight=1.0,
        dab_intensity_weight=0.2,
        dab_contrast_weight=0.0,    # No class ordering across stains
        dab_sharpness_weight=0.0,
        gram_style_weight=0.0,
        edge_weight=0.0,
        crop_disc_weight=0.0,
        feat_match_weight=10.0,
        patchnce_weight=0.0,
        bg_white_weight=0.0,
        # GAN training
        r1_weight=10.0,
        r1_every=16,
        adversarial_start_step=2000,
        # CFG dropout
        cfg_drop_class_prob=0.10,
        cfg_drop_uni_prob=0.10,
        cfg_drop_both_prob=0.05,
        # EMA
        ema_decay=0.999,
        # On-the-fly UNI extraction
        extract_uni_on_the_fly=True,
        uni_spatial_pool_size=32,
        # Paper innovation / ablation controls
        innovation_mode=args.innovation_mode,
        expr_field_weight=expr_field_weight,
        expr_pearson_weight=expr_pearson_weight,
        expr_iod_weight=expr_iod_weight,
        disentangle_weight=disentangle_weight,
        atlas_weight=atlas_weight,
        atlas_topk=args.atlas_topk,
        innovation_warmup_steps=args.innovation_warmup_steps,
        biofield_adapter=args.biofield_adapter,
        biofield_adapter_hidden=args.biofield_adapter_hidden,
        biofield_adapter_scale=args.biofield_adapter_scale,
        biofield_adapter_reg_weight=biofield_adapter_reg_weight,
        biofield_adapter_freeze_backbone=args.biofield_adapter_freeze_backbone,
        stain_experts=args.stain_experts,
        stain_expert_hidden=args.stain_expert_hidden,
        stain_expert_scale=args.stain_expert_scale,
        stain_expert_layers=stain_expert_layers,
        stain_experts_freeze_backbone=args.stain_experts_freeze_backbone,
    )

    if args.init_from_checkpoint:
        ckpt = torch.load(args.init_from_checkpoint, map_location='cpu')
        state_dict = ckpt.get('state_dict', ckpt)
        state_dict = {k: v for k, v in state_dict.items() if not k.startswith('_uni_model.')}
        missing, unexpected = model.load_state_dict(state_dict, strict=False)
        print(f"[init_from_checkpoint] loaded {args.init_from_checkpoint}")
        print(f"[init_from_checkpoint] missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print(f"[init_from_checkpoint] first missing keys: {missing[:8]}")
        if unexpected:
            print(f"[init_from_checkpoint] first unexpected keys: {unexpected[:8]}")

    dm = IHC4BCMultiStainCropDataModule(
        base_dir=args.data_dir,
        split_manifest=args.split_manifest,
        stains=args.stains,
        batch_size=args.batch_size,
        num_workers=4,
        image_size=(512, 512),
        crop_size=512,
        null_class=4,
    )

    ckpt_callback = ModelCheckpoint(
        dirpath=args.ckpt_dir,
        filename='best-{epoch:03d}-{step:06d}',
        save_top_k=3,
        monitor=args.early_stop_monitor,
        mode='min',
        save_last=True,
    )

    lr_monitor = LearningRateMonitor(logging_interval='step')
    val_jsonl_callback = ValMetricsJSONLCallback(val_metrics_jsonl)
    callbacks = [ckpt_callback, lr_monitor, val_jsonl_callback]
    if not args.disable_early_stop:
        callbacks.append(EarlyStopping(
            monitor=args.early_stop_monitor,
            mode='min',
            patience=args.early_stop_patience,
            verbose=True,
        ))

    wandb_logger = WandbLogger(
        project='biofieldstain',
        name=args.wandb_name,
        save_dir='wandb',
    )

    trainer = pl.Trainer(
        max_epochs=args.max_epochs,
        accelerator='gpu',
        devices=1,
        precision=args.precision,
        callbacks=callbacks,
        logger=wandb_logger,
        log_every_n_steps=10,
        check_val_every_n_epoch=args.check_val_every_n_epoch,
    )

    trainer.fit(model, dm, ckpt_path=args.resume_from)
    print("Training complete!")


if __name__ == "__main__":
    main()
