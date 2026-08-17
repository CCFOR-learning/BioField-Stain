# BioField-Stain

BioField-Stain is a unified virtual immunohistochemistry (IHC) staining framework for translating hematoxylin and eosin (H&E) images into marker-specific IHC images. The current release focuses on the MIST breast cancer benchmark and supports four biomarkers: HER2, Ki67, ER, and PR.

The model combines frozen UNI pathology foundation-model tokens, a SPADE-guided U-Net generator, target-marker embeddings, BioField-guided decoder modulation, and protein-aware supervision in DAB optical-density space.

## Highlights

- Unified multi-marker H&E-to-IHC generation for HER2, Ki67, ER, and PR.
- Dense morphology guidance from frozen UNI spatial tokens.
- BioField supervision for marker-aware expression consistency.
- DAB-aware evaluation metrics for virtual IHC quality assessment.
- Training, inference, and evaluation scripts for the MIST dataset.

## Installation

```bash
git clone https://github.com/CCFOR-learning/BioField-Stain.git
cd BioField-Stain

pip install -r requirements.txt
pip install -e .
```

## UNI Weights

BioField-Stain uses the UNI ViT-L/16 pathology foundation model as a frozen morphology encoder. UNI weights can be loaded from Hugging Face through `timm`, or placed locally as:

```text
weights/
+-- uni-vit-l16/
    +-- pytorch_model.bin
```

Please follow the license and access requirements of the original UNI model.

## Dataset

This release is organized for the MIST dataset. The expected folder structure is:

```text
MIST/
+-- HER2/
|   +-- TrainValAB/
|       +-- trainA/    # H&E training images
|       +-- trainB/    # IHC training images
|       +-- valA/      # H&E validation images
|       +-- valB/      # IHC validation images
+-- Ki67/
|   +-- TrainValAB/
|       +-- trainA/
|       +-- trainB/
|       +-- valA/
|       +-- valB/
+-- ER/
|   +-- TrainValAB/
+-- PR/
    +-- TrainValAB/
```

Images in `trainA`/`trainB` and `valA`/`valB` should be paired by filename.

## Training

Train the unified four-marker model on MIST:

```bash
python scripts/train/train_mist.py \
  --data_dir /path/to/MIST \
  --ckpt_dir checkpoints/mist \
  --batch_size 16 \
  --max_epochs 75 \
  --precision 16-mixed
```

To train on selected markers only:

```bash
python scripts/train/train_mist.py \
  --data_dir /path/to/MIST \
  --stains HER2 Ki67 \
  --ckpt_dir checkpoints/mist_subset
```

## Evaluation

Evaluate a trained checkpoint on the MIST validation set:

```bash
python scripts/eval/eval_mist.py \
  --checkpoint checkpoints/mist/last.ckpt \
  --data_dir /path/to/MIST \
  --output_dir eval_output/mist
```

For 1024-resolution evaluation:

```bash
python scripts/eval/eval_mist_1024.py \
  --checkpoint checkpoints/mist/last.ckpt \
  --data_dir /path/to/MIST \
  --output_dir eval_output/mist_1024
```

## Inference

Generate virtual IHC images for a folder of H&E inputs:

```bash
python scripts/infer_mist.py \
  --checkpoint checkpoints/mist/last.ckpt \
  --input /path/to/he_images \
  --stain HER2 \
  --output outputs/her2
```

Supported stain names are `HER2`, `Ki67`, `ER`, and `PR`.

## Project Structure

```text
biofield_stain/
+-- data/      # MIST data loaders
+-- models/    # generator, discriminator, BioField trainer, and loss modules
+-- utils/     # DAB extraction and evaluation metrics

scripts/
+-- train/     # training entry points
+-- eval/      # evaluation entry points
+-- infer_mist.py
```

## Notes

This repository contains source code only. Dataset files, trained checkpoints, generated images, and experiment outputs are not included.

## Citation

If you use this code, please cite the BioField-Stain manuscript once it becomes available.
