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
