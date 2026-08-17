#!/usr/bin/env python3
from __future__ import annotations

import random
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from typing import Iterable, Sequence

import torch
from PIL import Image
from torch.cuda.amp import autocast
from torch.utils.data import Dataset
from torchvision import transforms
import torchvision.transforms.functional as TF

from biofield_stain.models.trainer import BioFieldStainTrainer

VALID_EXTS = {'.png', '.jpg', '.jpeg', '.tif', '.tiff'}
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]
M11_MEAN = [0.5, 0.5, 0.5]
M11_STD = [0.5, 0.5, 0.5]
STAIN_TO_LABEL = {'HER2': 0, 'Ki67': 1, 'ER': 2, 'PR': 3}


def make_power_2(img: Image.Image, base: int = 4, method=Image.BICUBIC) -> Image.Image:
    ow, oh = img.size
    w = int(round(ow / base) * base)
    h = int(round(oh / base) * base)
    if w == ow and h == oh:
        return img
    return img.resize((w, h), method)


def _crop_box(w: int, h: int, crop_size: int, x: int, y: int):
    tw = th = crop_size
    if w > tw or h > th:
        return x, y, x + tw, y + th
    return 0, 0, w, h


class PairedTransform:
    def __init__(
        self,
        load_size: int = 1024,
        crop_size: int = 512,
        train: bool = True,
        color_jitter_he: bool = True,
        eval_crop: str = 'center',
        jitter_brightness: float = 0.3,
        jitter_contrast: float = 0.3,
        jitter_saturation: float = 0.2,
    ):
        self.load_size = load_size
        self.crop_size = crop_size
        self.train = train
        self.color_jitter_he = color_jitter_he and train
        self.eval_crop = eval_crop
        self.he_jitter = transforms.ColorJitter(
            brightness=jitter_brightness,
            contrast=jitter_contrast,
            saturation=jitter_saturation,
        )

    def _sample_crop_flip(self, w: int, h: int):
        if self.train:
            x = random.randint(0, max(0, w - self.crop_size))
            y = random.randint(0, max(0, h - self.crop_size))
            flip = random.random() > 0.5
            return x, y, flip
        if self.eval_crop == 'full':
            return 0, 0, False
        if self.eval_crop == 'center':
            x = max(0, (w - self.crop_size) // 2)
            y = max(0, (h - self.crop_size) // 2)
            return x, y, False
        x = random.randint(0, max(0, w - self.crop_size))
        y = random.randint(0, max(0, h - self.crop_size))
        return x, y, False

    def __call__(self, he_img: Image.Image, ihc_img: Image.Image):
        he = he_img.resize((self.load_size, self.load_size), Image.BICUBIC)
        ihc = ihc_img.resize((self.load_size, self.load_size), Image.BICUBIC)
        eval_crop_size = self.crop_size
        if not self.train and self.eval_crop == 'full':
            eval_crop_size = min(he.size)
        w, h = he.size
        x, y, flip = self._sample_crop_flip(w, h)
        box = _crop_box(w, h, eval_crop_size, x, y)
        he = he.crop(box)
        ihc = ihc.crop(box)
        he = make_power_2(he)
        ihc = make_power_2(ihc)
        if flip:
            he = he.transpose(Image.FLIP_LEFT_RIGHT)
            ihc = ihc.transpose(Image.FLIP_LEFT_RIGHT)
        if self.color_jitter_he:
            he = self.he_jitter(he)
        return he, ihc


class MISTEvalDataset(Dataset):
    def __init__(
        self,
        data_dir: str,
        stains: Sequence[str],
        split: str = 'val',
        load_size: int = 1024,
        crop_size: int = 512,
    ):
        self.base_dir = Path(data_dir)
        self.stains = list(stains)
        self.split = 'val' if split == 'test' else split
        self.transform = PairedTransform(
            load_size=load_size,
            crop_size=crop_size,
            train=False,
            color_jitter_he=False,
            eval_crop='center',
        )
        self.uni_crop_transform = transforms.Compose(
            [
                transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
        )
        split_he = 'trainA' if self.split == 'train' else 'valA'
        split_ihc = 'trainB' if self.split == 'train' else 'valB'
        self.samples = []
        for stain in self.stains:
            if stain not in STAIN_TO_LABEL:
                raise ValueError(f'Unknown stain: {stain}')
            stain_label = STAIN_TO_LABEL[stain]
            he_dir = self.base_dir / stain / 'TrainValAB' / split_he
            ihc_dir = self.base_dir / stain / 'TrainValAB' / split_ihc
            if not he_dir.exists() or not ihc_dir.exists():
                raise FileNotFoundError(f'Missing pair dirs: {he_dir} | {ihc_dir}')
            he_stems = {p.stem: p for p in sorted(he_dir.iterdir()) if p.suffix.lower() in VALID_EXTS}
            ihc_stems = {p.stem: p for p in sorted(ihc_dir.iterdir()) if p.suffix.lower() in VALID_EXTS}
            for stem in sorted(set(he_stems) & set(ihc_stems)):
                self.samples.append((he_stems[stem], ihc_stems[stem], stain, stain_label))

    def __len__(self):
        return len(self.samples)

    def _prepare_uni_sub_crops(self, he_pil: Image.Image) -> torch.Tensor:
        w, h = he_pil.size
        num_crops = 4
        cw = w // num_crops
        ch = h // num_crops
        sub_crops = []
        for i in range(num_crops):
            for j in range(num_crops):
                left = j * cw
                top = i * ch
                sub = he_pil.crop((left, top, left + cw, top + ch))
                sub_crops.append(self.uni_crop_transform(sub))
        return torch.stack(sub_crops)

    def __getitem__(self, idx: int):
        he_path, ihc_path, stain, stain_label = self.samples[idx]
        he_img = Image.open(he_path).convert('RGB')
        ihc_img = Image.open(ihc_path).convert('RGB')
        he_crop, ihc_crop = self.transform(he_img, ihc_img)
        he_tensor = TF.normalize(TF.to_tensor(he_crop), M11_MEAN, M11_STD)
        ihc_tensor = TF.normalize(TF.to_tensor(ihc_crop), M11_MEAN, M11_STD)
        uni_sub_crops = self._prepare_uni_sub_crops(he_crop)
        return {
            'he': he_tensor,
            'ihc': ihc_tensor,
            'uni_sub_crops': uni_sub_crops,
            'label': stain_label,
            'stain': stain,
            'filename': he_path.name,
            'he_path': str(he_path),
            'ihc_path': str(ihc_path),
        }


def collate_eval_batch(batch):
    return {
        'he': torch.stack([item['he'] for item in batch], dim=0),
        'ihc': torch.stack([item['ihc'] for item in batch], dim=0),
        'uni_sub_crops': torch.stack([item['uni_sub_crops'] for item in batch], dim=0),
        'label': torch.tensor([item['label'] for item in batch], dtype=torch.long),
        'stain': [item['stain'] for item in batch],
        'filename': [item['filename'] for item in batch],
        'he_path': [item['he_path'] for item in batch],
        'ihc_path': [item['ihc_path'] for item in batch],
    }


def load_model(checkpoint: str, device: torch.device) -> BioFieldStainTrainer:
    model = BioFieldStainTrainer.load_from_checkpoint(checkpoint, map_location=device, strict=False)
    model = model.to(device)
    model.eval()
    return model


def run_generate(model, he, uni_sub_crops, labels, *, device, use_amp=True, guidance_scale=1.0):
    he = he.to(device, non_blocking=True)
    labels = labels.to(device, non_blocking=True)
    uni_sub_crops = uni_sub_crops.to(device, non_blocking=True)
    with torch.no_grad():
        with autocast(enabled=(use_amp and device.type == 'cuda'), dtype=torch.float16):
            uni = model._extract_uni_from_sub_crops(uni_sub_crops)
            pred = model.generate(he, uni, labels, guidance_scale=guidance_scale)
    return torch.nan_to_num(pred, nan=0.0, posinf=1.0, neginf=-1.0).clamp(-1.0, 1.0)


def prepare_single_input(image_path: str | Path, *, load_size: int = 1024, crop_size: int = 512):
    image_path = Path(image_path)
    image = Image.open(image_path).convert('RGB')
    transform = PairedTransform(load_size=load_size, crop_size=crop_size, train=False, color_jitter_he=False, eval_crop='center')
    he_crop, _ = transform(image, image)
    he_tensor = TF.normalize(TF.to_tensor(he_crop), M11_MEAN, M11_STD).unsqueeze(0)
    uni_crop_transform = transforms.Compose(
        [
            transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
        ]
    )
    w, h = he_crop.size
    num_crops = 4
    cw = w // num_crops
    ch = h // num_crops
    sub_crops = []
    for i in range(num_crops):
        for j in range(num_crops):
            left = j * cw
            top = i * ch
            sub = he_crop.crop((left, top, left + cw, top + ch))
            sub_crops.append(uni_crop_transform(sub))
    uni_sub_crops = torch.stack(sub_crops).unsqueeze(0)
    return he_crop, he_tensor, uni_sub_crops


def save_tensor_image(tensor: torch.Tensor, path: str | Path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    x = tensor.detach().cpu().float()
    if x.ndim == 4:
        x = x[0]
    x = ((x + 1.0) * 0.5).clamp(0, 1)
    arr = (x.permute(1, 2, 0).numpy() * 255.0).round().astype('uint8')
    Image.fromarray(arr).save(path)


def iter_input_images(path: str | Path) -> Iterable[Path]:
    path = Path(path)
    if path.is_file():
        yield path
        return
    for child in sorted(path.iterdir()):
        if child.suffix.lower() in VALID_EXTS:
            yield child
