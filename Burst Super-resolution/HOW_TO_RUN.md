# How to run the pipeline manually

These steps run the full pipeline on real BurstSR, ending with the guarded reference-frame selector
(`ref_selection.py`). Run every command from this folder (`Burst Super-resolution/`).

> **Requires an NVIDIA GPU with CUDA.** The scripts call `.cuda()` directly (e.g. `reference_frame_sweep.py`), and
> PWC-Net's correlation layer uses CuPy. They will not run on macOS. Use Linux or Windows with an NVIDIA card; the
> original runs used an RTX 4070.

## 1. Environment
```bash
python -m venv .venv && source .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-ref-frame.txt
```

## 2. Data and weights
- **BurstSR val set** (~1.5 GB, 882 bursts):
  `gdown 1Qjx1jCMhdabkCedoJP704cGyM3wo8QiH -O burstsr_dataset/val.zip`, then unzip so you have
  `burstsr_dataset/val/<burst>/...`
- **PWC-Net weights:** download `http://content.sniklaus.com/github/pytorch-pwc/network-default.pytorch` and save it
  as `pwcnet/pwcnet-network-default.pth`.
- **Burstormer real-data weights:** download from
  https://drive.google.com/file/d/1ME1WuRaL3Bv7tBlbcO5hY-pOZ5439E0C/view (in a browser if gdown hits the quota limit)
  and save as `Trained_models/Real/burstormer_real.pth`. Create the folder first.

## 3. Calibrate the model (do this first, ~10 min)
The downloaded weights produce colours in a different space from the input, so every later step uses the fine-tuned
model. Downloads the 528 MB VGG weights for LPIPS on first use.
```bash
python finetune_real.py --init ./Trained_models/Real/burstormer_real.pth \
  --out ./Trained_models/Real/burstormer_real_calibrated.pth --steps 3000
```

## 4. Score each frame as the reference
```bash
python reference_frame_sweep.py --weights ./Trained_models/Real/burstormer_real_calibrated.pth \
  --bursts_file ./splits/dev.txt --limit 40 --out_dir ./Results/Real/sweep_dev
```
Writes `per_burst.csv` (one row per burst × reference frame) and `summary.csv` to `--out_dir`.

## 5. Compare selection policies (no GPU needed)
```bash
python analyze_reference_policies.py --sweep_dir ./Results/Real/sweep_dev
```
Compares frame 0, frame 2, sharpest, guarded and other policies, with confidence intervals.

## 6. Check that the guarded selector avoids a damaged frame
```bash
python robustness_sweep.py --degradation blur --degrade_frames 2 --ref_policy frame2 \
  --bursts_file ./splits/dev.txt --limit 12 --out_dir ./Results/Real/rescue_blur_f2_frame2
python robustness_sweep.py --degradation blur --degrade_frames 2 --ref_policy guarded \
  --bursts_file ./splits/dev.txt --limit 12 --out_dir ./Results/Real/rescue_blur_f2_guarded
```
Compare the two runs. The guarded run should keep its PSNR when frame 2 is blurred.

## Optional
- **Quick check** (one burst per scene × all 14 reference frames, ~2–3 min):
  `python reference_frame_sweep.py --weights <pth> --limit 23 --out_dir ./Results/Real/sweep_scenes`
- **Images** (ground truth and the output for each reference frame):
  `python visualize_reference_frames.py --sweep_dir <dir> --n 4`
- **Noise limits:**
  `python robustness_sweep.py --weights <pth> --degradation noise --bursts_file ./splits/dev.txt --limit 24`
- **Selector alone:** `guarded_choice({frame_idx: sharpness, ...})` in `ref_selection.py` is pure Python and needs no
  GPU.

More detail and the full list of experiments: [Readme.md](Readme.md), [../docs/PLAN.md](../docs/PLAN.md),
[../docs/TICKETS.md](../docs/TICKETS.md).
