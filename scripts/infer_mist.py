#!/usr/bin/env python3
from __future__ import annotations

import argparse
import time
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


import torch
from tqdm import tqdm

from scripts.mist_eval_utils import (
    STAIN_TO_LABEL,
    iter_input_images,
    load_model,
    prepare_single_input,
    run_generate,
    save_tensor_image,
)


def main():
    parser = argparse.ArgumentParser(description='Inference with BioField-Stain on MIST-style inputs')
    parser.add_argument('--checkpoint', required=True)
    parser.add_argument('--input', required=True, help='Single image or directory')
    parser.add_argument('--stain', required=True, choices=sorted(STAIN_TO_LABEL.keys()))
    parser.add_argument('--output', required=True, help='Output directory')
    parser.add_argument('--guidance_scale', type=float, default=1.0)
    parser.add_argument('--load_size', type=int, default=1024)
    parser.add_argument('--crop_size', type=int, default=512)
    parser.add_argument('--no_amp', action='store_true')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = load_model(args.checkpoint, device)
    label = torch.tensor([STAIN_TO_LABEL[args.stain]], dtype=torch.long)
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    paths = list(iter_input_images(args.input))
    if not paths:
        raise FileNotFoundError(f'No images found under: {args.input}')

    print(f'Inference: {len(paths)} image(s) | stain={args.stain} | device={device} | out={out_dir}')
    total_start = time.perf_counter()
    image_times = []
    pbar = tqdm(paths, desc='Inference', unit='img')
    for path in pbar:
        _, he_tensor, uni_sub_crops = prepare_single_input(
            path,
            load_size=args.load_size,
            crop_size=args.crop_size,
        )
        t0 = time.perf_counter()
        pred = run_generate(
            model,
            he_tensor,
            uni_sub_crops,
            label,
            device=device,
            use_amp=(not args.no_amp),
            guidance_scale=args.guidance_scale,
        )
        if device.type == 'cuda':
            torch.cuda.synchronize()
        elapsed = time.perf_counter() - t0
        image_times.append(elapsed)
        out_path = out_dir / f'{Path(path).stem}_{args.stain}_virtual.png'
        save_tensor_image(pred, out_path)
        pbar.set_postfix(last=f'{elapsed:.2f}s', avg=f'{sum(image_times)/len(image_times):.2f}s/img', refresh=False)

    total_elapsed = time.perf_counter() - total_start
    avg = sum(image_times) / len(image_times)
    print(f'\nDone. {len(paths)} image(s) in {total_elapsed:.1f}s (avg {avg:.3f}s/img).')
    print(f'Outputs: {out_dir}')


if __name__ == '__main__':
    main()
