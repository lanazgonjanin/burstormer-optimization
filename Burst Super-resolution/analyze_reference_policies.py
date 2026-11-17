"""Compare reference-frame selection policies from a reference_frame_sweep.py result (no GPU needed).

Reads <sweep_dir>/per_burst.csv and reports, for each policy, the PSNR gain over always using frame 0, measured on
the region common to all references (so the comparison is not biased by the ground truth being framed on frame 0),
with a bootstrap 95% confidence interval.

Policies
    fixed_ref_k         always frame k
    sharpest_frame      highest variance of the Laplacian (rewards noise)
    sharpest_smoothed   highest gradient energy after a Gaussian blur
    most_central        smallest mean displacement to the other frames
    central_sharp       sharpest_smoothed among the 4 most central frames
    guarded             frame 2 unless its sharpness is abnormal next to the other first five frames (ref_selection.py);
                        a table shows how the result changes with the margin
    oracle              best frame per burst using the ground truth (upper bound, optimistic)
    oracle_split_half   frame chosen on half of the image patches, scored on the other half. If a frame is better
                        for real it is better on both halves; the gap to `oracle` is selection noise.

Example:
    python analyze_reference_policies.py --sweep_dir ./Results/Real/sweep_dev
"""

import argparse
import csv
import math
import os
from collections import defaultdict
from typing import Callable, Dict, List

import numpy as np
from scipy.stats import spearmanr

from ref_selection import GUARD_DEFAULT, GUARD_MARGIN, GUARD_WINDOW, guarded_choice

HALF_A, HALF_B = [0, 2, 4, 6, 8], [1, 3, 5, 7]  # checkerboard over the 3x3 patch grid


def load(path: str) -> Dict[str, Dict[int, Dict]]:
    by_burst = defaultdict(dict)
    with open(path, newline='') as f:
        for r in csv.DictReader(f):
            row = {k: (v if k in ('burst', 'patch_psnr_common') else float(v)) for k, v in r.items()}
            row['ref'] = int(row['ref'])
            row['patches'] = np.array([float(x) for x in r['patch_psnr_common'].split(';')])
            by_burst[r['burst']][row['ref']] = row
    return by_burst


def bootstrap_ci(values: np.ndarray, groups: List[str], n_boot: int = 2000) -> List[float]:
    """95% interval of the mean. Resamples scenes when there are at least 8 of them, bursts otherwise."""
    rng = np.random.default_rng(0)
    scenes = sorted(set(groups))
    if len(scenes) >= 8:
        by_scene = [values[[g == s for g in groups]] for s in scenes]
        stats = [np.concatenate([by_scene[i] for i in rng.integers(0, len(scenes), len(scenes))]).mean()
                 for _ in range(n_boot)]
    else:
        stats = [values[rng.integers(0, len(values), len(values))].mean() for _ in range(n_boot)]
    return [float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))]


def main() -> None:
    parser = argparse.ArgumentParser(description='Analyse reference-frame selection policies')
    parser.add_argument('--sweep_dir', default='./Results/Real/sweep_dev', type=str)
    parser.add_argument('--margin', default=GUARD_MARGIN, type=float,
                        help='Guarded policy: how far the sharpness of frame 2 may be from its neighbours before it is replaced')
    args = parser.parse_args()

    by_burst = load(os.path.join(args.sweep_dir, 'per_burst.csv'))
    refs = sorted(next(iter(by_burst.values())))
    bursts = sorted(b for b, d in by_burst.items()
                    if all(r in d for r in refs) and not any(math.isnan(d[r]['psnr_common']) for r in refs))
    dropped = len(by_burst) - len(bursts)
    scenes = [b.split('_')[0] for b in bursts]
    n_scenes = len(set(scenes))
    print(f'{len(bursts)} bursts from {n_scenes} scenes, {len(refs)} candidate references'
          + (f' ({dropped} bursts dropped: no common region)' if dropped else ''))
    print(f'Confidence intervals resample {"scenes" if n_scenes >= 8 else "bursts (fewer than 8 scenes: optimistic)"}')
    print(f'Mean share of the image common to all references: '
          f'{np.mean([by_burst[b][refs[0]]["common_frac"] for b in bursts]):.0%}\n')

    def central_sharp(d: Dict[int, Dict]) -> int:
        central = sorted(d, key=lambda r: d[r]['centrality'])[:4]
        return max(central, key=lambda r: d[r]['sharpness_smooth'])

    policies: Dict[str, Callable[[Dict[int, Dict]], int]] = {f'fixed_ref_{k}': (lambda d, k=k: k) for k in refs}
    policies.update({
        'sharpest_frame': lambda d: max(d, key=lambda r: d[r]['sharpness']),
        'sharpest_smoothed': lambda d: max(d, key=lambda r: d[r]['sharpness_smooth']),
        'most_central': lambda d: min(d, key=lambda r: d[r]['centrality']),
        'central_sharp': central_sharp,
        # Restricted to the first few frames, as HDR+ does (it picks the sharpest of the first 3)
        'sharp_first3': lambda d: max(refs[:3], key=lambda r: d[r]['sharpness_smooth']),
        'sharp_first5': lambda d: max(refs[:5], key=lambda r: d[r]['sharpness_smooth']),
        'central_first5': lambda d: min(refs[:5], key=lambda r: d[r]['centrality']),
        'oracle_first5': lambda d: max(refs[:5], key=lambda r: d[r]['psnr_common']),
        'oracle': lambda d: max(d, key=lambda r: d[r]['psnr_common']),
    })

    def guarded(margin: float) -> Callable[[Dict[int, Dict]], int]:
        return lambda d: guarded_choice({r: d[r]['sharpness_smooth'] for r in d}, margin=margin)

    has_guard = GUARD_DEFAULT in refs
    if has_guard:
        # Frame 2 unless its sharpness is abnormal next to the other early frames (ref_selection.py)
        policies['guarded'] = guarded(args.margin)
    for d in by_burst.values():
        for r in d:
            # Distance of this frame from frame 0, which the ground truth is registered to
            d[r]['dist_from_frame0'] = math.hypot(d[r]['shift_x'], d[r]['shift_y'])

    base = refs[0]
    out_rows = []
    print(f'{"policy":<20}{"PSNR":>8}{"PSNR_common":>13}{"gain vs frame " + str(base):>18}{"95% CI":>18}{"wins":>7}')
    for name, pick in policies.items():
        chosen = [pick(by_burst[b]) for b in bursts]
        psnr = np.array([by_burst[b][c]['psnr'] for b, c in zip(bursts, chosen)])
        common = np.array([by_burst[b][c]['psnr_common'] for b, c in zip(bursts, chosen)])
        gain = common - np.array([by_burst[b][base]['psnr_common'] for b in bursts])
        lo, hi = bootstrap_ci(gain, scenes)
        wins = float(np.mean(gain > 0))
        out_rows.append({'policy': name, 'psnr': psnr.mean(), 'psnr_common': common.mean(), 'gain': gain.mean(),
                         'ci_low': lo, 'ci_high': hi, 'win_rate': wins})
        print(f'{name:<20}{psnr.mean():>8.2f}{common.mean():>13.2f}{gain.mean():>+18.3f}'
              f'{f"[{lo:+.3f}, {hi:+.3f}]":>18}{wins:>7.0%}')

    if has_guard:
        # How the guarded rule behaves as the margin changes. Choose the margin on dev, report it on test.
        base_common = np.array([by_burst[b][base]['psnr_common'] for b in bursts])
        fixed = np.array([by_burst[b][GUARD_DEFAULT]['psnr_common'] for b in bursts])
        print(f'\nGuarded rule by margin (default frame {GUARD_DEFAULT}, first {GUARD_WINDOW} frames are candidates):')
        print(f'{"margin":>8}{"gain vs frame " + str(base):>18}{"gain vs frame " + str(GUARD_DEFAULT):>18}'
              f'{"95% CI":>18}{"switched":>10}')
        for margin in (0.0, 0.05, 0.1, 0.2, 0.3, 0.5, 1.0):
            chosen = [guarded(margin)(by_burst[b]) for b in bursts]
            common = np.array([by_burst[b][c]['psnr_common'] for b, c in zip(bursts, chosen)])
            lo, hi = bootstrap_ci(common - fixed, scenes)
            print(f'{margin:>8g}{(common - base_common).mean():>+18.3f}{(common - fixed).mean():>+18.3f}'
                  f'{f"[{lo:+.3f}, {hi:+.3f}]":>18}{np.mean([c != GUARD_DEFAULT for c in chosen]):>10.0%}')

    # Split-half oracle: choose on one half of the patches, score on the other half, and the reverse
    gains_in, gains_out = [], []
    for b in bursts:
        d = by_burst[b]
        for pick_half, score_half in ((HALF_A, HALF_B), (HALF_B, HALF_A)):
            ok = [r for r in refs if not np.isnan(d[r]['patches'][pick_half]).all()
                  and not np.isnan(d[r]['patches'][score_half]).all()]
            if base not in ok:
                continue
            best = max(ok, key=lambda r: np.nanmean(d[r]['patches'][pick_half]))
            gains_in.append(np.nanmean(d[best]['patches'][pick_half]) - np.nanmean(d[base]['patches'][pick_half]))
            gains_out.append(np.nanmean(d[best]['patches'][score_half]) - np.nanmean(d[base]['patches'][score_half]))
    print(f'\nOracle headroom check (gain over frame {base}, dB):')
    print(f'  chosen and scored on the same patches:   {np.mean(gains_in):+.3f}')
    print(f'  chosen on one half, scored on the other: {np.mean(gains_out):+.3f}   <- realistic upper bound')
    out_rows.append({'policy': 'oracle_split_half', 'psnr': '', 'psnr_common': '', 'gain': float(np.mean(gains_out)),
                     'ci_low': '', 'ci_high': '', 'win_rate': float(np.mean(np.array(gains_out) > 0))})

    # Which input features track the quality of a reference? Rank correlation within each burst, then averaged.
    print('\nMean within-burst rank correlation with PSNR_common (positive = higher feature, better reference):')
    for feat in ('ref', 'dist_from_frame0', 'sharpness', 'sharpness_smooth', 'centrality'):
        rhos = []
        for b in bursts:
            x = [by_burst[b][r][feat] for r in refs]
            y = [by_burst[b][r]['psnr_common'] for r in refs]
            if len(set(x)) > 1 and len(set(y)) > 1:
                rhos.append(spearmanr(x, y)[0])
        label = 'frame index' if feat == 'ref' else feat
        print(f'  {label:<18}{np.mean(rhos):+.2f}')

    out_path = os.path.join(args.sweep_dir, 'policy_analysis.csv')
    with open(out_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(out_rows[0].keys()))
        writer.writeheader()
        writer.writerows(out_rows)
    print(f'\nWrote {out_path}')


if __name__ == '__main__':
    main()
