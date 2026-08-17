"""IHC4BC adapter with the same batch contract as the MIST data module.

IHC4BC stores aligned H&E and IHC patches under separate marker directories.
This adapter only converts its layout into the existing CropPairedDataset
interface; model architecture and losses remain unchanged.

Split membership is read from an explicit
JSON manifest. Do not generate a random test set and label it as the published
IHC4BC 1,000-pair benchmark.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import List, Optional, Tuple

import pytorch_lightning as pl
from PIL import Image
from torch.utils.data import DataLoader

from biofield_stain.data.bci_dataset import CropPairedDataset


STAIN_TO_LABEL = {'HER2': 0, 'Ki67': 1, 'ER': 2, 'PR': 3}
IHC4BC_DIRNAME = {'HER2': 'Her2', 'Ki67': 'Ki67', 'ER': 'ER', 'PR': 'PR'}


def _load_manifest(path: str | Path) -> dict:
    manifest_path = Path(path)
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f'IHC4BC split manifest not found: {manifest_path}. '
            'Use the benchmark split; do not create a random test set.'
        )
    with manifest_path.open('r', encoding='utf-8') as handle:
        manifest = json.load(handle)
    if not isinstance(manifest.get('splits'), dict):
        raise ValueError(f'Invalid IHC4BC manifest: missing object key "splits" in {manifest_path}')
    return manifest


class IHC4BCMultiStainCropDataset(CropPairedDataset):
    """Multi-stain IHC4BC dataset using a fixed split manifest.

    Manifest paths are relative to each marker folder, for example:
    ``Patient_106_107011/Subregion_0/106_107011_0_0_0.jpg``.
    """

    def __init__(
        self,
        base_dir: str,
        split_manifest: str,
        stains: List[str],
        split: str,
        image_size: Tuple[int, int] = (512, 512),
        crop_size: int = 512,
        augment: bool = False,
        null_class: int = 4,
    ):
        super().__init__(
            he_dir='.', ihc_dir='.', image_size=image_size, crop_size=crop_size,
            augment=augment, null_class=null_class,
        )
        self.base_dir = Path(base_dir)
        self.samples = []  # (he_path, ihc_path, stain_label, relative_id)
        manifest = _load_manifest(split_manifest)
        split_entries = manifest['splits'].get(split)
        if not isinstance(split_entries, dict):
            raise ValueError(f'IHC4BC manifest has no "{split}" split')

        for stain in stains:
            if stain not in STAIN_TO_LABEL:
                raise ValueError(f'Unknown stain: {stain}. Expected one of {list(STAIN_TO_LABEL)}')
            rel_paths = split_entries.get(stain)
            if not isinstance(rel_paths, list):
                raise ValueError(f'Manifest split "{split}" has no list for stain "{stain}"')

            marker_dir = IHC4BC_DIRNAME[stain]
            he_root = self.base_dir / 'Images' / 'HandE' / marker_dir
            ihc_root = self.base_dir / 'Images' / 'IHC' / marker_dir
            if not he_root.is_dir() or not ihc_root.is_dir():
                raise FileNotFoundError(f'Missing IHC4BC image directories for {stain}: {he_root}, {ihc_root}')

            missing = []
            for rel_path in rel_paths:
                rel = Path(rel_path)
                if rel.is_absolute() or '..' in rel.parts:
                    raise ValueError(f'Manifest path must be relative to marker directory: {rel_path}')
                he_path = he_root / rel
                ihc_path = ihc_root / rel
                if he_path.is_file() and ihc_path.is_file():
                    self.samples.append((he_path, ihc_path, STAIN_TO_LABEL[stain], rel.as_posix()))
                else:
                    missing.append(rel.as_posix())
            if missing:
                preview = ', '.join(missing[:3])
                raise FileNotFoundError(f'{stain}/{split}: {len(missing)} paired files missing, e.g. {preview}')
            print(f'  {stain} ({split}): {len(rel_paths)} manifest pairs')

        counts = Counter(label for _, _, label, _ in self.samples)
        print(f'IHC4BC Crop Dataset ({split}): {len(self.samples)} total | {dict(sorted(counts.items()))}')

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        he_path, ihc_path, stain_label, relative_id = self.samples[index]
        he_img = Image.open(he_path).convert('RGB')
        ihc_img = Image.open(ihc_path).convert('RGB')
        return self._process_pair(he_img, ihc_img, stain_label, relative_id)


class IHC4BCMultiStainCropDataModule(pl.LightningDataModule):
    """DataModule with unchanged output tensors for the trainer."""

    def __init__(
        self,
        base_dir: str,
        split_manifest: str,
        stains: Optional[List[str]] = None,
        batch_size: int = 4,
        num_workers: int = 4,
        image_size: Tuple[int, int] = (512, 512),
        crop_size: int = 512,
        null_class: int = 4,
    ):
        super().__init__()
        self.base_dir = base_dir
        self.split_manifest = split_manifest
        self.stains = stains or ['HER2', 'Ki67', 'ER', 'PR']
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.image_size = image_size
        self.crop_size = crop_size
        self.null_class = null_class

    def _dataset(self, split: str, augment: bool):
        return IHC4BCMultiStainCropDataset(
            base_dir=self.base_dir,
            split_manifest=self.split_manifest,
            stains=self.stains,
            split=split,
            image_size=self.image_size,
            crop_size=self.crop_size,
            augment=augment,
            null_class=self.null_class,
        )

    def setup(self, stage=None):
        if stage in ('fit', None):
            self.train_dataset = self._dataset('train', augment=True)
            self.val_dataset = self._dataset('val', augment=False)
        if stage in ('test', 'predict'):
            self.test_dataset = self._dataset('test', augment=False)

    def _loader(self, dataset, shuffle: bool):
        return DataLoader(
            dataset, batch_size=self.batch_size, shuffle=shuffle,
            num_workers=self.num_workers, pin_memory=True,
            persistent_workers=self.num_workers > 0,
        )

    def train_dataloader(self):
        return self._loader(self.train_dataset, shuffle=True)

    def val_dataloader(self):
        return self._loader(self.val_dataset, shuffle=False)

    def test_dataloader(self):
        return self._loader(self.test_dataset, shuffle=False)
