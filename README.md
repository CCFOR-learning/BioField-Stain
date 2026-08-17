# BioField-Stain

BioField-Stain is a multi-stain virtual IHC synthesis model for translating H&E inputs into marker-specific IHC images. The model uses frozen UNI spatial tokens together with a SPADE U-Net generator, stain embeddings, and BioField supervision for pathology-aware synthesis.

## Features

- Multi-stain generation for HER2, Ki67, ER, and PR.
- Frozen UNI feature guidance with dense spatial tokens.
- BioField supervision for marker-aware appearance and expression consistency.
- Training and evaluation scripts for MIST and BCI-style paired datasets.

## Installation

```bash
pip install -r requirements.txt
pip install -e .
```

## Data Layout

### MIST

```text
MIST/
├── HER2/
│   └── TrainValAB/
│       ├── trainA/
│       ├── trainB/
│       ├── valA/
│       └── valB/
├── Ki67/
├── ER/
└── PR/
```

### BCI

```text
BCI/
├── HE/
│   ├── train/
│   └── test/
└── IHC/
    ├── train/
    └── test/
```

## Training

### MIST

```bash
python scripts/train/train_mist.py   --data_dir /path/to/MIST   --ckpt_dir checkpoints/mist   --batch_size 16   --max_epochs 75
```

### IHC4BC

```bash
python scripts/train/train_ihc4bc.py   --data_dir /path/to/IHC4BC   --split_manifest /path/to/split.json   --ckpt_dir checkpoints/ihc4bc
```

## Evaluation

### MIST

```bash
python scripts/eval/eval_mist.py   --checkpoint checkpoints/mist/last.ckpt   --data_dir /path/to/MIST
```

### MIST 1024

```bash
python scripts/eval/eval_mist_1024.py   --checkpoint checkpoints/mist/last.ckpt   --data_dir /path/to/MIST
```

## Inference

```bash
python scripts/infer_mist.py   --checkpoint checkpoints/mist/last.ckpt   --input /path/to/he_images   --stain HER2   --output outputs/her2
```

## Project Structure

```text
biofield_stain/
├── data/
├── models/
└── utils/

scripts/
├── train/
├── eval/
└── infer_mist.py
```
