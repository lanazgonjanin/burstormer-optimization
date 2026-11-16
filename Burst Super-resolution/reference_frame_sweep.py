"""Reference-frame sweep for Burstormer on the real BurstSR validation set.

The original evaluation always uses raw frame 0 as the reference. This script evaluates every burst once per
candidate reference frame (the chosen frame is placed first; the remaining frames stay in capture order) and
records aligned PSNR / SSIM / LPIPS, plus a cheap no-reference sharpness score for each reference frame.

Outputs (in --out_dir):
    per_burst.csv  one row per (burst, reference frame)
    summary.csv    mean metrics per reference frame, plus selection policies (oracle, sharpest frame)

Example:
    python reference_frame_sweep.py --weights ./Trained_models/Real/Burstormer.ckpt
    python reference_frame_sweep.py --weights ... --limit 20 --refs 0 3 7     # quick check
"""

import argparse
import csv
import math
import os
import random
import time
from collections import defaultdict
from typing import Dict, List, Tuple

import cv2
import lpips
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data.dataloader import DataLoader

from Network_Real_burstSR import Base_Model
from datasets.burstsr_dataset import BurstSRDataset
from pwcnet.pwcnet import PWCNet
from utils.metrics import AlignedPred, make_patches

NUM_FRAMES = 14


class FastAlignedMetrics:
    """Same numbers as utils.metrics.AlignedPSNR / AlignedSSIM / AlignedLPIPS, computed ~10x faster.

    The originals each split the output into patches and each re-run PWC-Net alignment + colour matching per patch
    (3x redundant), then compute SSIM with scipy and LPIPS on the CPU. Here every patch is aligned once and all three
    metrics are computed from that on the GPU. Equivalence is checked by `--verify_metrics`.
    """

    def __init__(self, alignment_net: torch.nn.Module) -> None:
        self.aligner = AlignedPred(alignment_net=alignment_net)
        self.lpips_net = lpips.LPIPS(net='alex', verbose=False).cuda().eval()
        gauss = cv2.getGaussianKernel(11, 1.5)
        self.ssim_window = torch.from_numpy(gauss * gauss.T).cuda().view(1, 1, 11, 11)  # float64, as in utils/ssim.py

    def _ssim_map(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """GPU port of utils.ssim.cal_ssim (zero-padded 'same' convolution, float64)."""
        c1, c2 = (0.01 * 255) ** 2, (0.03 * 255) ** 2
        img1, img2 = img1.double()[None, None], img2.double()[None, None]
        conv = lambda x: F.conv2d(x, self.ssim_window, padding=5)
        mu1, mu2 = conv(img1), conv(img2)
        s1 = conv(img1 * img1) - mu1 * mu1
        s2 = conv(img2 * img2) - mu2 * mu2
        s12 = conv(img1 * img2) - mu1 * mu2
        return (((2 * mu1 * mu2 + c1) * (2 * s12 + c2)) / ((mu1 * mu1 + mu2 * mu2 + c1) * (s1 + s2 + c2)))[0, 0]

    @torch.no_grad()
    def __call__(self, output: torch.Tensor, labels: torch.Tensor, burst: torch.Tensor) -> Tuple[float, float, float]:
        psnrs, ssims, lpipss = [], [], []
        for p, g, b in zip(*make_patches(output, labels, burst)):
            pred_m, gt, valid = self.aligner(p[None], g[None], b[None])
            validf = valid.float()

            # AlignedL2 -> AlignedPSNR
            mse = F.mse_loss(pred_m, gt, reduction='none')
            mse = (mse * validf).sum() / (validf.sum() * (mse.numel() / valid.numel()) + 1e-12)
            psnrs.append(-10.0 * math.log10(mse.item() + 1e-12))

            # AlignedSSIM (first colour channel only, as in the original)
            ssim_map = self._ssim_map(pred_m[0, 0] * 255, gt[0, 0] * 255).float()
            v = validf.squeeze()
            ssims.append(((ssim_map * v).sum() / (v.sum() * (ssim_map.numel() / v.numel()) + 1e-12)).item())

            # AlignedLPIPS
            lpipss.append(self.lpips_net(2 * pred_m - 1, 2 * gt - 1).item())
        return float(np.mean(psnrs)), float(np.mean(ssims)), float(np.mean(lpipss))


def set_seed(seed: int) -> None:
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    np.random.seed(seed)
    random.seed(seed)


def load_model(weights_path: str) -> Base_Model:
    """Load Burstormer weights from either the original .pth format or a PyTorch Lightning .ckpt."""
    checkpoint = torch.load(weights_path, map_location='cpu', weights_only=False)
    if isinstance(checkpoint, dict):
        for key in ('model_state_dict', 'state_dict'):
            if key in checkpoint:
                checkpoint = checkpoint[key]
                break
    state_dict = {k[len('model.'):] if k.startswith('model.') else k: v for k, v in checkpoint.items()}

    model = Base_Model()
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    # The model's own loss/metric modules carry no weights we care about; anything else missing is a real problem
    if missing or unexpected:
        raise RuntimeError(f'Checkpoint does not match Base_Model.\nMissing: {missing}\nUnexpected: {unexpected}')
    return model.eval().cuda()


def frame_sharpness(burst: torch.Tensor) -> torch.Tensor:
    """Variance of the Laplacian of the green channel for each frame. burst: [1, N, 4, H, W] packed RGGB."""
    green = burst[0, :, 1:3].mean(dim=1, keepdim=True)  # [N, 1, H, W]
    kernel = torch.tensor([[0., 1., 0.], [1., -4., 1.], [0., 1., 0.]], device=burst.device).view(1, 1, 3, 3)
    lap = F.conv2d(green, kernel)
    return lap.flatten(1).var(dim=1)  # [N]


def reorder(burst: torch.Tensor, ref: int) -> torch.Tensor:
    """Put frame `ref` first and keep the remaining frames in capture order."""
    perm = [ref] + [i for i in range(burst.shape[1]) if i != ref]
    return burst[:, perm]


def summarise(rows: List[Dict], refs: List[int]) -> List[Dict]:
    by_burst = defaultdict(dict)
    for r in rows:
        by_burst[r['burst']][r['ref']] = r
    complete = [b for b, d in by_burst.items() if all(ref in d for ref in refs)]

    summary = []
    for ref in refs:
        sel = [by_burst[b][ref] for b in complete]
        summary.append({'policy': f'fixed_ref_{ref}', 'psnr': np.mean([s['psnr'] for s in sel]),
                        'ssim': np.mean([s['ssim'] for s in sel]), 'lpips': np.mean([s['lpips'] for s in sel]),
                        'times_best_psnr': sum(max(by_burst[b].values(), key=lambda x: x['psnr'])['ref'] == ref
                                               for b in complete)})

    policies = {
        'oracle_best_psnr': lambda d: max(d.values(), key=lambda x: x['psnr']),
        'sharpest_frame': lambda d: max(d.values(), key=lambda x: x['sharpness']),
    }
    for name, pick in policies.items():
        sel = [pick(by_burst[b]) for b in complete]
        summary.append({'policy': name, 'psnr': np.mean([s['psnr'] for s in sel]),
                        'ssim': np.mean([s['ssim'] for s in sel]), 'lpips': np.mean([s['lpips'] for s in sel]),
                        'times_best_psnr': ''})
    for s in summary:
        s['n_bursts'] = len(complete)
    return summary


def verify_metrics(model: Base_Model, metrics: FastAlignedMetrics, alignment_net: torch.nn.Module,
                   args: argparse.Namespace) -> None:
    """Check FastAlignedMetrics reproduces the original AlignedPSNR/SSIM/LPIPS on a few bursts and refs."""
    from utils.metrics import AlignedPSNR, AlignedSSIM, AlignedLPIPS
    psnr_fn, ssim_fn = AlignedPSNR(alignment_net=alignment_net), AlignedSSIM(alignment_net=alignment_net)
    lpips_fn = AlignedLPIPS(alignment_net=alignment_net)

    dataset = BurstSRDataset(root=args.data_root, split='val', burst_size=NUM_FRAMES, crop_sz=80,
                             random_flip=False, ref_id=0, shuffle_frames=False)
    worst = np.zeros(3)
    for idx in range(args.verify_metrics):
        burst, labels, _, _, name = dataset[idx * (len(dataset) // args.verify_metrics)]
        burst, labels = burst[None].cuda(), labels[None].cuda()
        for ref in (0, 7):
            burst_ref = reorder(burst, ref)
            with torch.no_grad():
                output = model(burst_ref).clamp(0.0, 1.0)
            t0 = time.time()
            orig = np.array([float(psnr_fn(output, labels, burst_ref)), float(ssim_fn(output, labels, burst_ref)),
                             float(lpips_fn(output, labels, burst_ref))])
            t1 = time.time()
            fast = np.array(metrics(output, labels, burst_ref))
            t2 = time.time()
            worst = np.maximum(worst, np.abs(orig - fast))
            print(f'{name} ref={ref:2d}  original PSNR/SSIM/LPIPS = {orig.round(5)} ({t1 - t0:.1f}s)   '
                  f'fast = {fast.round(5)} ({t2 - t1:.1f}s)')
    print(f'Max abs difference  PSNR {worst[0]:.2e}  SSIM {worst[1]:.2e}  LPIPS {worst[2]:.2e}')


def main() -> None:
    parser = argparse.ArgumentParser(description='Sweep the reference frame for real BurstSR evaluation')
    parser.add_argument('--weights', default='./Trained_models/Real/Burstormer.ckpt', type=str)
    parser.add_argument('--data_root', default='./burstsr_dataset', type=str)
    parser.add_argument('--pwcnet', default='./pwcnet/pwcnet-network-default.pth', type=str)
    parser.add_argument('--out_dir', default='./Results/Real/ref_frame_sweep', type=str)
    parser.add_argument('--refs', default=list(range(NUM_FRAMES)), type=int, nargs='+',
                        help='Reference frame indices to evaluate (default: all 14)')
    parser.add_argument('--limit', default=None, type=int, help='Only evaluate the first N bursts')
    parser.add_argument('--random_weights', action='store_true',
                        help='Skip loading weights (pipeline smoke test only; metrics are meaningless)')
    parser.add_argument('--verify_metrics', default=0, type=int,
                        help='Compare the fast metrics against the original utils.metrics classes on N bursts, then exit')
    args = parser.parse_args()

    set_seed(50)
    os.makedirs(args.out_dir, exist_ok=True)

    alignment_net = PWCNet(load_pretrained=True, weights_path=args.pwcnet).cuda()
    metrics = FastAlignedMetrics(alignment_net)

    model = Base_Model().eval().cuda() if args.random_weights else load_model(args.weights)

    if args.verify_metrics:
        verify_metrics(model, metrics, alignment_net, args)
        return

    # Load in capture order (frame 0 first, no shuffling); reference reordering happens on the GPU tensor
    dataset = BurstSRDataset(root=args.data_root, split='val', burst_size=NUM_FRAMES, crop_sz=80,
                             random_flip=False, ref_id=0, shuffle_frames=False)
    if args.limit is not None:
        dataset = torch.utils.data.Subset(dataset, range(min(args.limit, len(dataset))))
    loader = DataLoader(dataset, batch_size=1, num_workers=2)

    per_burst_path = os.path.join(args.out_dir, 'per_burst.csv')
    fields = ['burst', 'ref', 'psnr', 'ssim', 'lpips', 'sharpness']
    rows: List[Dict] = []
    start = time.time()

    with open(per_burst_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for i, (burst, labels, _, _, burst_name) in enumerate(loader):
            burst, labels = burst.cuda(), labels.cuda()
            sharpness = frame_sharpness(burst).cpu().numpy()
            for ref in args.refs:
                burst_ref = reorder(burst, ref)
                with torch.no_grad():
                    output = model(burst_ref).clamp(0.0, 1.0)
                # The metrics colour-match using burst_ref[:, 0], i.e. the frame the output is aligned to
                psnr, ssim, lpips_val = metrics(output, labels, burst_ref)
                row = {'burst': burst_name[0], 'ref': ref, 'psnr': psnr, 'ssim': ssim, 'lpips': lpips_val,
                       'sharpness': float(sharpness[ref])}
                rows.append(row)
                writer.writerow(row)
            f.flush()
            if (i + 1) % 10 == 0 or i == 0:
                elapsed = time.time() - start
                print(f'[{i + 1}/{len(dataset)}] {elapsed / (i + 1):.2f}s/burst, '
                      f'ETA {elapsed / (i + 1) * (len(dataset) - i - 1) / 60:.1f} min', flush=True)

    summary = summarise(rows, args.refs)
    summary_path = os.path.join(args.out_dir, 'summary.csv')
    with open(summary_path, 'w', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=['policy', 'n_bursts', 'psnr', 'ssim', 'lpips', 'times_best_psnr'])
        writer.writeheader()
        writer.writerows(summary)

    print(f'\n{"policy":<20}{"PSNR":>9}{"SSIM":>8}{"LPIPS":>8}{"#best":>7}')
    for s in summary:
        print(f'{s["policy"]:<20}{s["psnr"]:>9.3f}{s["ssim"]:>8.4f}{s["lpips"]:>8.4f}{str(s["times_best_psnr"]):>7}')
    print(f'\nWrote {per_burst_path} and {summary_path}')


if __name__ == '__main__':
    main()
