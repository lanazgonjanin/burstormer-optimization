"""Contact sheet of robustness_sweep.py images: one column per degradation level, one row per model.

Needs sweeps that were run with --save_images. Rows: the degraded input reference frame, then the output of each run
(for example the calibrated model and the noise-augmented model); the last column is the ground truth.

Example:
    python robustness_panel.py --runs calibrated=./Results/Real/robustness_noise \
        noise_aug=./Results/Real/robustness_noise_aug --out ./Results/Real/noise_panel
"""

import argparse
import os
import re
from typing import Dict, List

import cv2
import numpy as np

LABEL_H, LABEL_W, PAD = 28, 150, 4


def text(img: np.ndarray, s: str, x: int, y: int) -> None:
    cv2.putText(img, s, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (20, 20, 20), 1, cv2.LINE_AA)


def main() -> None:
    parser = argparse.ArgumentParser(description='Build contact sheets from robustness sweep images')
    parser.add_argument('--runs', nargs='+', required=True, help='label=sweep_dir pairs, shown top to bottom')
    parser.add_argument('--out', default='./Results/Real/robustness_panel', type=str)
    parser.add_argument('--crop', default=256, type=int, help='Centre crop (pixels of the 640x640 output) per cell')
    args = parser.parse_args()

    runs: Dict[str, str] = dict(r.split('=', 1) for r in args.runs)
    first_dir = os.path.join(next(iter(runs.values())), 'images')
    os.makedirs(args.out, exist_ok=True)

    for burst in sorted(os.listdir(first_dir)):
        files = sorted(f for f in os.listdir(os.path.join(first_dir, burst)) if f.endswith('_model.jpg'))
        tags = [f[:-len('_model.jpg')] for f in files]  # e.g. noise_03_0.001
        levels = [re.match(r'.+_\d+_(.+)$', t).group(1) for t in tags]

        def cell(path: str) -> np.ndarray:
            img = cv2.imread(path)
            a = (img.shape[0] - args.crop) // 2
            return img[a:a + args.crop, a:a + args.crop]

        rows: List[List[np.ndarray]] = [[cell(os.path.join(first_dir, burst, f'{t}_input.jpg')) for t in tags]]
        row_labels = ['input (ref frame)']
        gt = cell(os.path.join(first_dir, burst, 'gt.jpg'))
        for label, sweep_dir in runs.items():
            rows.append([cell(os.path.join(sweep_dir, 'images', burst, f'{t}_model.jpg')) for t in tags])
            row_labels.append(label)

        c = args.crop
        n_cols = len(tags) + 1
        sheet = np.full((LABEL_H + len(rows) * (c + PAD), LABEL_W + n_cols * (c + PAD), 3), 245, np.uint8)
        for j, level in enumerate(levels + ['ground truth']):
            text(sheet, level if j == len(levels) else f'level {level}', LABEL_W + j * (c + PAD) + 6, 19)
        for i, (label, row) in enumerate(zip(row_labels, rows)):
            y = LABEL_H + i * (c + PAD)
            text(sheet, label, 6, y + c // 2)
            for j, img in enumerate(row + [gt]):
                x = LABEL_W + j * (c + PAD)
                sheet[y:y + c, x:x + c] = img
        out_path = os.path.join(args.out, f'{burst}.jpg')
        cv2.imwrite(out_path, sheet, [cv2.IMWRITE_JPEG_QUALITY, 90])
        print(f'Wrote {out_path}')


if __name__ == '__main__':
    main()
