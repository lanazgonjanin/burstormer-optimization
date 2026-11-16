## NTIRE 2021 Burst Super-Resolution Challenge - Track 1 Synthetic
### Training
- Download [Zurich RAW to RGB dataset](http://people.ee.ethz.ch/~ihnatova/pynet.html#dataset).
```
python Burstormer_Track_1_training.py
```
### Developement Phase:
- Download [syn_burst_val](https://data.vision.ee.ethz.ch/bhatg/syn_burst_val.zip) and extract it in root directory.
- Download [Trained model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/akshay_dudhane_mbzuai_ac_ae/ER8mPnjoSIZAnaKA8YyCeE8BA_uQr_73b5qRZx9sh9Rzvw?e=Sc4HFJ) and place it in './Trained_models/Synthetic/Burstormer.ckpt'.
        
```
python Track_1_evaluation.py
```
- Results: stored in './Results/Synthetic/Developement Phase'


## NTIRE 2021 Burst Super-Resolution Challenge - Track 2 Real-world
### Training
- Download [BurstSR train and validation set](https://github.com/goutamgmb/NTIRE21_BURSTSR/blob/master/burstsr_links.md).
- Download [Pretrained Burstormer on synthetic burst SR dataset](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/akshay_dudhane_mbzuai_ac_ae/ER8mPnjoSIZAnaKA8YyCeE8BA_uQr_73b5qRZx9sh9Rzvw?e=Sc4HFJ) and place it in './logs/Track_2/saved_model/Burstormer.ckpt'.
```
python Burstormer_Track_2_training.py
```
### Developement Phase:
- Download [burstsr_dataset](https://data.vision.ee.ethz.ch/bhatg/BurstSRChallenge/val.zip) and extract it in root directory.
- Download [Trained model](https://mbzuaiac-my.sharepoint.com/:u:/g/personal/akshay_dudhane_mbzuai_ac_ae/EQK37mvF2axEgMYfJ2-IZUgBHkMxvw1qwqLarlBcRnKLNQ?e=bVAf4F) and place it in './Trained_models/Real/Burstormer.pth'.

```
python Track_2_evaluation.py
```
- Results: stored in './Results/Real/Developement Phase'


## Reference-frame selection experiment (real BurstSR)
The original evaluation always uses raw frame 0 as the reference. `reference_frame_sweep.py` evaluates each burst
with every candidate reference frame (chosen frame first, remaining frames in capture order) and reports aligned
PSNR / SSIM / LPIPS per frame, plus an oracle (best frame per burst) and a sharpest-frame policy.

### Setup
```
python -m venv .venv
.venv\Scripts\activate            # Linux/macOS: source .venv/bin/activate
pip install torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements-ref-frame.txt
```
Data and weights (run from this folder). The original ETH and SharePoint links in the sections above are dead.
- **BurstSR val set** (~1.5 GB, Google Drive mirror from [deep-burst-sr](https://github.com/goutamgmb/deep-burst-sr)):
  `gdown 1Qjx1jCMhdabkCedoJP704cGyM3wo8QiH -O burstsr_dataset/val.zip`, then unzip so you have `burstsr_dataset/val/<burst>/...`
- **PWC-Net weights**: download `http://content.sniklaus.com/github/pytorch-pwc/network-default.pytorch`
  and save as `pwcnet/pwcnet-network-default.pth`
- **Burstormer real-data weights** (mirror linked from [BSRD](https://github.com/placerkyo/BSRD)):
  https://drive.google.com/file/d/1ME1WuRaL3Bv7tBlbcO5hY-pOZ5439E0C/view — download in a browser if gdown hits the
  quota limit, and save as `Trained_models/Real/burstormer_real.pth`

### Run
```
# Small test: one burst from each of the 23 scenes x all 14 reference frames (~2-3 min on an RTX 4070)
python reference_frame_sweep.py --weights ./Trained_models/Real/burstormer_real.pth --limit 23 --out_dir ./Results/Real/sweep_scenes
# Render images (GT, output per reference frame, raw input frames) for the 4 bursts that vary most
python visualize_reference_frames.py --sweep_dir ./Results/Real/sweep_scenes --n 4
# Specific frames only
python reference_frame_sweep.py --weights ./Trained_models/Real/burstormer_real.pth --limit 20 --refs 0 4 6 9
# Check the fast metrics against the original utils.metrics implementation
python reference_frame_sweep.py --weights ./Trained_models/Real/burstormer_real.pth --verify_metrics 4
```
Outputs `per_burst.csv` (one row per burst x reference frame) and `summary.csv` in `--out_dir`.


## Optimization experiments (plan: [../docs/PLAN.md](../docs/PLAN.md), tickets: [../docs/TICKETS.md](../docs/TICKETS.md))
All commands run from this folder with the setup above. `splits/{train,dev,test}.txt` split the 882 val bursts by
scene (15 / 4 / 4 scenes), because crops of one scene are near-duplicates.

### 1. Calibrated model (needed first)
The downloaded `burstormer_real.pth` produces output in a different colour space from its input and scores below
bilinear upsampling under the aligned metrics. A short fine-tune with the Track 2 loss on the train scenes fixes
that (about 10 min on an RTX 4070; downloads the 528 MB VGG weights for LPIPS on first use):
```
python finetune_real.py --init ./Trained_models/Real/burstormer_real.pth --out ./Trained_models/Real/burstormer_real_calibrated.pth --steps 3000
```
It prints PSNR on held-out dev-scene bursts next to the bilinear baseline. Numbers from this model are comparable
with each other, not with the paper (it is trained on part of the val set).

### 2. Reference-frame selection
```
# Every frame as reference on dev-scene bursts. Also scores each reference on the image region common to all of
# them (PSNR_common), since the ground-truth crop is framed on frame 0.
python reference_frame_sweep.py --weights ./Trained_models/Real/burstormer_real_calibrated.pth --bursts_file ./splits/dev.txt --limit 40 --out_dir ./Results/Real/sweep_dev
# Selection policies (sharpest, most central, ...) vs frame 0 with confidence intervals; no GPU needed
python analyze_reference_policies.py --sweep_dir ./Results/Real/sweep_dev
```

### 3. Limits: how much degradation the model tolerates
```
# Noise sweep (shot + read noise added to the raw burst). Other degradations: frames, shift, blur, outlier
python robustness_sweep.py --weights ./Trained_models/Real/burstormer_real_calibrated.pth --degradation noise --bursts_file ./splits/dev.txt --limit 24 --save_images 2
```
Prints model / bilinear / burst-mean PSNR per level and the level at which the model breaks. Results go to
`./Results/Real/robustness_<degradation>` unless `--out_dir` is given.

### 4. Mitigation: noise-augmented fine-tuning
```
python finetune_real.py --init ./Trained_models/Real/burstormer_real.pth --out ./Trained_models/Real/burstormer_real_noise_aug.pth --steps 3000 --noise_max 0.1
python robustness_sweep.py --weights ./Trained_models/Real/burstormer_real_noise_aug.pth --degradation noise --bursts_file ./splits/dev.txt --limit 24 --save_images 2 --out_dir ./Results/Real/robustness_noise_aug
# Side-by-side images: input / calibrated / noise-augmented at every noise level, plus the ground truth
python robustness_panel.py --runs calibrated=./Results/Real/robustness_noise noise_aug=./Results/Real/robustness_noise_aug --out ./Results/Real/noise_panel
```
