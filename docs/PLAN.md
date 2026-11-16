# Burstormer optimization plan

Last updated: 2026-11-16. Scope: real-world burst super-resolution (BurstSR, Track 2).

## Goals

1. **Reference-frame selection (Track A).** Burstormer always treats frame 0 of the burst as the reference. Find
   out whether a method that picks the reference per burst gives better results, and build it if so.
2. **Limits and mitigation (Track B).** Degrade the input (noise first, then other factors) until Burstormer stops
   producing a usable result, find out why it fails there, and add a method that moves that limit.

Both tracks depend on a baseline we can trust, so that comes first (Phase 0).

## Where we are

| Item | State |
|---|---|
| Environment | Working on Python 3.14, torch 2.11 + cu128, CuPy 14 (see `Burst Super-resolution/Readme.md`) |
| Data | BurstSR **val** set only (882 bursts, 23 scenes, 14 frames each). Train set not downloaded |
| Evaluation | `reference_frame_sweep.py`: aligned PSNR / SSIM / LPIPS, verified against the original metric code |
| Reference-frame test | 23 bursts (one per scene) x 14 reference frames, results below |
| Checkpoint | `burstormer_real.pth` (epoch 12, mirror linked from the BSRD repo). **Does not reproduce the paper** |
| Working model | `burstormer_real_calibrated.pth`, our fine-tune of that checkpoint (see "Results so far") |

Findings 1 and 2 were measured on the downloaded checkpoint, before calibration.

### Finding 1: the reference frame matters, but frame 0 is a reasonable default

23 bursts, one per scene, every frame tried as the reference.

| Choice of reference | PSNR (dB) | Versus frame 0 |
|---|---|---|
| Frame 0 (original behaviour) | 30.43 | |
| Best single fixed frame (frame 2) | 30.58 | +0.15 |
| Sharpest frame (variance of Laplacian) | 29.81 | -0.62 |
| Best frame per burst, chosen in hindsight | 31.13 | +0.70 |

Two things make these numbers provisional. The ground-truth crop is framed on frame 0, so other references lose
field of view and are penalised for geometry rather than quality (ticket T04). The hindsight number picks the
maximum of 14 noisy scores, so it overstates what any real selector can reach (ticket T05).

### Finding 2: the available checkpoint scores below bilinear upsampling

8 bursts spread over scenes, standard aligned PSNR.

| Output | PSNR (dB) |
|---|---|
| Burstormer, this checkpoint | 29.59 |
| Burstormer, colour transform fitted on its own output instead of on the input frame | 39.82 |
| Bilinear upsampling of input frame 0 (no network) | 40.78 |
| Burstormer, as reported in the paper (full val set) | 48.82 |

The metric fits a colour transform from input frame 0 to the ground truth and applies it to the output, so it
assumes the output is in the input's colour space. This checkpoint's output is not: relative to its input, red
is about 1.5-1.9x, green 1.2-1.6x and blue 1.6-1.9x.

Ruled out as causes: output clipping (affects 1 of 8 bursts), input crop size, frame order, input scale, and
white-balanced input (25.57 dB, worse). The upstream repository has an unanswered issue reporting the same
reproduction problem (akshaydudhane16/Burstormer#5), and its released `Network_Real_burstSR.py` crashed on an
undefined variable, so the released code is not exactly what produced the paper numbers.

**Consequence.** Results from this checkpoint show how *this* model behaves, not how a correctly trained
Burstormer behaves. Every experiment below reports the bilinear baseline next to the model, and conclusions are
re-checked once Phase 0 delivers a working model.

## Results so far (2026-11-16)

All numbers below use the calibrated model unless stated, on scenes it was not trained on. Small tests: 12-40
bursts. Commands are in `Burst Super-resolution/Readme.md`.

### Phase 0: calibrated model (T02, done)

3,000 fine-tuning steps (about 10 minutes) with the Track 2 loss on the 15 train scenes.

| Model | PSNR on 12 dev-scene bursts (dB) |
|---|---|
| Downloaded checkpoint | 30.74 |
| Bilinear upsampling of frame 0 | 41.19 |
| Calibrated checkpoint | 45.90 |

On 24 test-scene bursts the calibrated model gets 48.44 dB against 43.47 dB for bilinear. These numbers are not
comparable with the paper's 48.82 dB: the model is trained on part of the val set and scored on a subset.

### Track A: reference frame (T04-T06, first pass on 40 dev bursts)

| Reference policy | PSNR, original metric | PSNR, common region | Gain over frame 0 (common region) |
|---|---|---|---|
| Frame 0 | 45.34 | 45.44 | |
| Frame 2 (best fixed frame) | 45.52 | 45.77 | +0.34 [+0.07, +0.64] |
| Frame 13 | 38.42 | 43.24 | -2.20 [-2.64, -1.75] |
| Sharpest frame | 41.97 | 44.59 | -0.85 [-1.29, -0.42] |
| Sharpest frame after smoothing | 42.75 | 44.90 | -0.54 [-0.95, -0.15] |
| Most central frame | 42.27 | 44.56 | -0.88 [-1.19, -0.59] |
| Sharpest of the first 3 frames | 45.48 | 45.65 | +0.22 [+0.02, +0.45] |
| Best frame per burst (oracle) | 44.93 | 46.03 | +0.59 [+0.33, +0.90] |
| Oracle, chosen on half the patches and scored on the rest | | | +0.54 |

Intervals are 95% bootstrap over bursts (4 scenes, so they are optimistic).

- **Most of the penalty for late frames was framing.** Frame 13 loses 6.9 dB on the original metric and 2.2 dB
  when every reference is scored on the same pixels.
- **The headroom is real, about +0.5 dB.** The split-half check keeps +0.54 of the +0.59 dB oracle gain, so it is
  not selection noise. T05 decision: go.
- **The best reference is always one of the first five frames**, and quality falls steadily with frame index
  (within-burst rank correlation -0.86).
- **Sharpness and motion centrality over all 14 frames lose to frame 0.** They pick late frames. Restricted to
  the first 3 frames, sharpness gains a little.
- **Open confound.** Frame index and distance from frame 0 are equally correlated with quality (-0.86 each), and
  the ground truth is registered to frame 0. We cannot yet tell "later frames are worse references" from "the
  metric favours outputs near frame 0". Synthetic bursts, where ground truth can be rendered for any reference,
  can separate the two (added to T04).

### Track B: limits (T09-T10, first pass)

Noise is added to the raw burst as shot + read noise; the level is the shot-noise coefficient. Synthetic
Burstormer training covered levels 0.0001 to 0.012.

| Degradation (12 dev bursts, noise 24) | Clean | Where the model breaks |
|---|---|---|
| Noise | 45.56 dB | 3 dB down at level 0.0048; below clean bilinear at 0.0077; worse than plain frame averaging at 0.042 |
| Blur (Gaussian, every frame) | 45.90 dB | 3 dB down at sigma 0.65; no better than bilinear from sigma 1.5 |
| Fewer distinct frames | 45.90 dB | Degrades steadily; 3 dB down at 4 frames; equals bilinear at 1 frame |
| Extra misalignment (up to 16 px) | 45.90 dB | Not reached: 45.34 dB at 16 px |
| Outlier frames | 45.90 dB | Not reached: 43.89 dB with all 13 non-reference frames misaligned |

The model is robust to misalignment and outlier frames, and fragile to noise and blur. At noise level 0.1 the
output is structured artefacts and scores 6 dB below a plain average of the frames.

### Track B: mitigation (T13, first pass)

Same 3,000-step fine-tune with noise augmentation (half of the training bursts get noise at a level drawn
log-uniformly from 0.0001 to 0.1).

| Noise level | Calibrated (dB) | Noise-augmented (dB) | Change |
|---|---|---|---|
| 0 (clean) | 48.44 | 48.32 | -0.12 |
| 0.003 | 46.18 | 46.85 | +0.67 |
| 0.01 | 42.56 | 45.39 | +2.83 |
| 0.03 | 36.25 | 42.03 | +5.78 |
| 0.1 | 23.22 | 35.68 | +12.46 |

24 test-scene bursts. The 3 dB breaking level moves from 0.0038 to 0.0102 (2.7x) and the level at which the
model falls below clean bilinear from 0.0074 to 0.0187 (2.5x). The dev split shows the same (0.0048 to 0.0095).
Clean-input cost: -0.12 dB PSNR on test (+0.15 dB on dev), LPIPS slightly worse (0.047 to 0.049). The clean cost
is at the 0.1 dB budget, so it needs a larger run before we call it met.

Image panels: `Results/Real/noise_panel/` after running the Readme commands.

## Phases

Tickets are in [TICKETS.md](TICKETS.md); IDs below refer to them.

### Phase 0: a baseline we can trust (blocks final conclusions in both tracks)

- **T01** Resolve the checkpoint problem: find the authors' real checkpoint, or establish that it cannot be had.
- **T02** Calibration fine-tune: short fine-tune with the repo's own loss so the output returns to the input's
  colour space. First on a scene-level split of the val set (no download needed); on the BurstSR train set if we
  want numbers comparable to the paper.
- **T03** Evaluation protocol: fixed dev and test burst lists split by scene, paired comparisons with bootstrap
  confidence intervals, bilinear baseline in every table.

Exit criterion: a model that beats bilinear upsampling on held-out scenes under the standard metric.

### Track A: reference-frame selection

- **T04** Fair comparison across references (remove the ground-truth framing bias).
- **T05** Oracle headroom, corrected for selection noise. Decision point: if the corrected headroom is under
  about 0.2 dB, stop Track A and record the negative result.
- **T06** Training-free selectors: motion-central frame, noise-robust sharpness, maximum overlap, fixed index.
- **T07** Learned selector (stretch): small regressor from per-frame features to expected gain.
- **T08** Reference-agnostic fine-tuning (stretch): train with a random reference so the model is not tuned to
  frame 0.

Exit criterion: a selector that beats frame 0 on the held-out test split with a confidence interval that
excludes zero, or a documented negative result.

### Track B: limits and mitigation

- **T09** Degradation harness: noise (shot + read), low light, fewer usable frames, extra misalignment, blur,
  one corrupted frame.
- **T10** Locate the breaking points with stated failure criteria; curves and image panels.
- **T11** Failure analysis: which stage breaks (alignment, fusion or upsampling).
- **T12** Training-free mitigation: pre-denoising, variance stabilisation, outlier-frame rejection.
- **T13** Mitigation by training: noise-augmented fine-tuning and noise-level conditioning.
- **T14** Final report: before/after curves on the test split, full val run, write-up.

Exit criterion: the breaking point moves by a stated margin (target: the model tolerates at least 2x the noise
level at the same failure threshold) with no more than 0.1 dB loss on clean input.

## Order of work

```
T01 ─┬─> T02 ──> T03 ─┬─> T04 ──> T05 ──> T06 ──> (T07, T08)
     │                └─> T10 ──> T11 ──> T12 ──> T13 ──> T14
     └─> T09 (harness does not depend on the checkpoint)
```

T09 and T04 can be built and smoke-tested against the current checkpoint while Phase 0 is in progress. Their
numbers are provisional until re-run on the Phase 0 model.

## Evaluation protocol

- **Metrics.** Aligned PSNR, SSIM and LPIPS from `reference_frame_sweep.FastAlignedMetrics` (same numbers as the
  original `utils/metrics.py`, about 10x faster).
- **Baselines in every table.** Bilinear upsampling of the reference frame, and frame-0 Burstormer.
- **Splits.** By scene, never by burst: crops of one scene are near-duplicates. Dev split for exploring, test
  split touched only for reported numbers.
- **Statistics.** Paired differences per burst, bootstrap 95% interval resampled over scenes.
- **Degraded input.** The degraded burst goes to the model. The metric's colour matching uses the clean burst,
  so the metric itself is not degraded along with the input.
- **Size of runs.** Small tests first (one or two bursts per scene, minutes on an RTX 4070). Full 882-burst runs
  only for final numbers.

## Risks

| Risk | Effect | Response |
|---|---|---|
| No correct checkpoint can be found | Conclusions rest on a model below the paper's quality | T02 fine-tune; report relative changes |
| Val set is small and scenes are few (23) | Wide confidence intervals | Scene-level bootstrap; download the train set if needed |
| Reference-frame headroom is mostly selection noise | Track A yields no real gain | T05 decision point before building selectors |
| Mitigation helps noisy input but hurts clean input | Net loss in normal use | Clean-input regression check is part of T12/T13 acceptance |
| Fine-tuning on val scenes leaks into evaluation | Inflated numbers | Scene-level split; test scenes never used in training |

## Log

- 2026-11-15: environment ported, reference-frame sweep and visualisation added, 23-burst test run.
- 2026-11-16: checkpoint diagnosed (Finding 2), plan and tickets written.
- 2026-11-16: calibration fine-tune (T02), scene splits (T03, partly), common-region scoring and selection
  policies (T04-T06 first pass), degradation harness and breaking points (T09-T10 first pass), noise-augmented
  fine-tune (T13 first pass). See "Results so far".

## Next steps

1. T01: ask upstream for the correct checkpoint; decide whether to download the BurstSR train set.
2. T04: resolve the frame-0 registration confound on synthetic bursts before building a learned selector.
3. T06: test "sharpest of the first 3" and fixed frame 2 on the test scenes with more bursts.
4. T11: find out which stage fails under noise and under blur.
5. T13: longer noise-augmented run, noise-level conditioning, and a blur counterpart; full val-set numbers.
