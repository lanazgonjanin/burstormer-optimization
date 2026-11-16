"""Render Burstormer outputs for every reference frame, for visual comparison.

Reads a reference_frame_sweep.py result (per_burst.csv), picks the bursts whose PSNR varies most across reference
frames, and for each one saves:
    gt.jpg            ground truth (Canon HR image)
    ref_XX.jpg        super-resolved output with frame XX as reference, warped + colour-matched to the ground truth
                      as the aligned metrics see it, then one per-burst display gain (from the frame-0 result) so the
                      brightness matches the GT; the same gain is used for every reference frame of a burst.
                      Black regions are parts of the GT that frame XX never saw (the GT crop is framed on frame 0).
    input_XX.jpg      the low-res raw frame XX (simple demosaic, nearest-neighbour upscale)
plus manifest.json with all per-burst metrics from the CSV.

Example:
    python visualize_reference_frames.py --sweep_dir ./Results/Real/sweep_scenes --n 4
"""

import argparse
import csv
import json
import os
from collections import defaultdict
from typing import Dict, List

import cv2
import numpy as np
import torch
import torch.nn.functional as F

from datasets.burstsr_dataset import BurstSRDataset
from pwcnet.pwcnet import PWCNet
from reference_frame_sweep import NUM_FRAMES, load_model, reorder
from utils.metrics import AlignedPred
from utils.postprocessing_functions import process_burstsr_image_rgb


def to_srgb(im: torch.Tensor, meta: Dict, norm: float) -> np.ndarray:
    """Linear camera RGB [3, H, W] -> uint8 BGR for cv2, using the repo's BurstSR post-processing."""
    rgb = process_burstsr_image_rgb(im.cpu(), meta, external_norm_factor=norm)
    return cv2.cvtColor((rgb.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8), cv2.COLOR_RGB2BGR)


def load_rows(path: str) -> List[Dict]:
    with open(path, newline='') as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r['ref'] = int(r['ref'])
        for k in ('psnr', 'ssim', 'lpips', 'sharpness'):
            r[k] = float(r[k])
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description='Render outputs for each reference frame')
    parser.add_argument('--weights', default='./Trained_models/Real/burstormer_real.pth', type=str)
    parser.add_argument('--data_root', default='./burstsr_dataset', type=str)
    parser.add_argument('--pwcnet', default='./pwcnet/pwcnet-network-default.pth', type=str)
    parser.add_argument('--sweep_dir', default='./Results/Real/sweep_scenes', type=str)
    parser.add_argument('--out_dir', default=None, type=str, help='Default: <sweep_dir>/visuals')
    parser.add_argument('--n', default=4, type=int, help='Number of bursts to render (largest PSNR spread first)')
    parser.add_argument('--bursts', default=None, nargs='+', help='Explicit burst names instead of --n')
    args = parser.parse_args()
    out_dir = args.out_dir or os.path.join(args.sweep_dir, 'visuals')

    rows = load_rows(os.path.join(args.sweep_dir, 'per_burst.csv'))
    by_burst = defaultdict(dict)
    for r in rows:
        by_burst[r['burst']][r['ref']] = r
    spread = {b: max(d[k]['psnr'] for k in d) - min(d[k]['psnr'] for k in d) for b, d in by_burst.items()}
    chosen = args.bursts or sorted(spread, key=spread.get, reverse=True)[:args.n]

    alignment_net = PWCNet(load_pretrained=True, weights_path=args.pwcnet).cuda()
    aligner = AlignedPred(alignment_net=alignment_net)
    model = load_model(args.weights)
    dataset = BurstSRDataset(root=args.data_root, split='val', burst_size=NUM_FRAMES, crop_sz=80,
                             random_flip=False, ref_id=0, shuffle_frames=False)

    manifest = {'rendered': [], 'all_rows': rows}
    for name in chosen:
        burst, gt, meta_burst, meta_gt, _ = dataset[dataset.burst_list.index(name)]
        burst, gt = burst[None].cuda(), gt[None].cuda()
        bdir = os.path.join(out_dir, name)
        os.makedirs(bdir, exist_ok=True)

        # One brightness normalisation per burst so all images of the burst are directly comparable
        gt_lin = gt[0] * meta_gt['norm_factor']
        norm = float(gt_lin.max())
        cv2.imwrite(os.path.join(bdir, 'gt.jpg'), to_srgb(gt[0], meta_gt, norm), [cv2.IMWRITE_JPEG_QUALITY, 92])

        in_norm = None
        display_gain = None
        for ref in range(NUM_FRAMES):
            burst_ref = reorder(burst, ref)
            with torch.no_grad():
                output = model(burst_ref).clamp(0.0, 1.0)
                pred_m, _, valid = aligner(output, gt, burst_ref)
            if display_gain is None:
                # Display-only: the colour-matched output is brighter than the GT with this checkpoint, so compute one
                # per-channel gain from the frame-0 result and apply it to every reference of this burst. Differences
                # between reference frames are preserved; metrics are untouched.
                v = valid[0].float()
                display_gain = (gt[0] * v).sum((1, 2)) / ((pred_m[0] * v).sum((1, 2)) + 1e-8)
            pred_disp = pred_m[0] * display_gain.view(3, 1, 1)
            cv2.imwrite(os.path.join(bdir, f'ref_{ref:02d}.jpg'), to_srgb(pred_disp, meta_gt, norm),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

            # Raw RGGB frame -> RGB (average the two greens), shown at the output size with nearest-neighbour
            raw = burst[0, ref]
            rgb = torch.stack([raw[0], (raw[1] + raw[2]) / 2, raw[3]])
            rgb_lin = rgb * meta_burst['norm_factor']
            in_norm = in_norm or float(rgb_lin.max())
            rgb = F.interpolate(rgb[None], scale_factor=8, mode='nearest')[0]
            cv2.imwrite(os.path.join(bdir, f'input_{ref:02d}.jpg'), to_srgb(rgb, meta_burst, in_norm),
                        [cv2.IMWRITE_JPEG_QUALITY, 92])

        manifest['rendered'].append({'burst': name, 'psnr_spread': spread[name],
                                     'display_gain': [round(float(x), 3) for x in display_gain]})
        print(f'Rendered {name} (PSNR spread across refs {spread[name]:.2f} dB)')

    with open(os.path.join(out_dir, 'manifest.json'), 'w') as f:
        json.dump(manifest, f)
    print(f'Wrote images and manifest.json to {out_dir}')


if __name__ == '__main__':
    main()
