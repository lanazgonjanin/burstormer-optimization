# Tickets

Source for the GitHub issues. `docs/create_github_issues.ps1` turns each `## ` section into one issue (title,
labels and milestone from the first lines, the rest as the body). Context for all of them is in
[PLAN.md](PLAN.md).

## T01: Real-data checkpoint scores below bilinear upsampling; find the cause or a correct checkpoint
Labels: baseline, bug
Milestone: M0 Trustworthy baseline

**Why.** On 8 val bursts spread over scenes, `burstormer_real.pth` (epoch 12, from the BSRD mirror) gets 29.59 dB
aligned PSNR. Bilinear upsampling of input frame 0 gets 40.78 dB. The paper reports 48.82 dB.

**Known so far.**
- The output is in a different colour space from the input: red about 1.5-1.9x, green 1.2-1.6x, blue 1.6-1.9x.
  The aligned metric fits its colour transform on the input frame, so it penalises this heavily.
- Fitting the colour transform on the output itself gives 39.82 dB, still below bilinear.
- Ruled out: output clipping, crop size (24 / 48 / 80), frame order, input scale, white-balanced input (25.57 dB).
- Upstream has an unanswered report of the same problem: akshaydudhane16/Burstormer#5. The released
  `Network_Real_burstSR.py` crashed on an undefined variable (`adjust_sharpness` line), so the released code is
  not exactly what produced the paper numbers.

**Tasks.**
- [ ] Ask upstream for the real-data checkpoint and the exact evaluation code (comment on upstream issue #5).
- [ ] Look for other mirrors of the authors' `Burstormer.pth` and compare file hashes with ours.
- [ ] Check the synthetic checkpoint (BSRD mirror, Drive id `1E_6NdmzNjaEpxlrN06SQ9JAyOvGG5wA4`) on the synthetic
      val set to see whether that one reproduces the paper (41.5 dB range).
- [ ] Record the outcome in `docs/PLAN.md`.

**Done when.** We either have a checkpoint that beats bilinear under the standard metric, or a written
conclusion that none is available and T02 is the route.

## T02: Calibration fine-tune to get a working real-data model
Labels: baseline, training
Milestone: M0 Trustworthy baseline
Depends on: T01

**Why.** The checkpoint has learned structure but its colour mapping is inconsistent with the evaluation
pipeline. A short fine-tune with the repo's own loss (aligned L1 + LPIPS, colour-matched on the input frame)
should bring the output back to the input's colour space. The same script is needed later for T08 and T13.

**Tasks.**
- [ ] `finetune_real.py`: fine-tune `Base_Model` from a checkpoint with the Track 2 loss, random crops and flips.
- [ ] Scene-level split of the val set (train scenes / held-out scenes), written to a file and reused everywhere.
- [ ] Fine-tune on the train scenes; evaluate on the held-out scenes against bilinear and the original checkpoint.
- [ ] If the val-only model is good enough to work with, decide whether to download the BurstSR train set
      (tens of GB) for a model comparable to the paper.

**Done when.** A checkpoint beats bilinear upsampling on held-out scenes under the standard metric, and the
command that produced it is in the Readme.

## T03: Evaluation protocol: scene-level splits, paired statistics, baselines in every table
Labels: infra
Milestone: M0 Trustworthy baseline

**Why.** Burst names are `<scene>_<crop>`; crops of one scene are near-duplicates, so results must be split and
resampled by scene. Single-number comparisons on 23 bursts cannot separate a 0.15 dB effect from noise.

**Tasks.**
- [ ] `splits/dev.txt` and `splits/test.txt` (scene-level, fixed), plus the T02 train/held-out split.
- [ ] Shared helper: paired per-burst differences with a bootstrap 95% interval resampled over scenes.
- [ ] Bilinear-reference-frame baseline reported by every evaluation script.
- [ ] `--bursts_file` option on the sweep scripts.

**Done when.** Every results table has a baseline row, an interval on each difference, and names the split used.

## T04: Fair comparison across reference frames (remove the ground-truth framing bias)
Labels: track-a-ref-frame
Milestone: M1 Reference-frame selection
Depends on: T03

**Why.** The Canon ground-truth crop is framed on frame 0. With another frame as reference the output covers a
shifted area, part of the ground truth is never seen, and the metric penalises geometry rather than quality.
Later frames drift further, so they look worse than they are.

**Tasks.**
- [ ] Per burst, compute the region that is valid for every candidate reference (intersection of the metric's
      valid masks) and score all candidates on that region only.
- [ ] Report both the original and the common-region numbers.
- [ ] Re-run the 23-burst test and see whether the downward trend for late frames survives.

**Done when.** We can say how much of the reference-frame effect is framing and how much is real.

## T05: Oracle headroom for reference selection, corrected for selection noise
Labels: track-a-ref-frame
Milestone: M1 Reference-frame selection
Depends on: T04

**Why.** "Best frame per burst in hindsight" gained +0.70 dB, but the maximum of 14 noisy scores is biased
upward. We need the headroom a real selector could reach before investing in one.

**Tasks.**
- [ ] Estimate metric noise: repeat each (burst, reference) with the other frames in shuffled order.
- [ ] Split-half check: choose the best reference on half of the patches, score it on the other half.
- [ ] Report corrected headroom with an interval, on the dev split.

**Done when.** A go / no-go decision for Track A is recorded: go if corrected headroom is at least about 0.2 dB.

## T06: Training-free reference selectors
Labels: track-a-ref-frame
Milestone: M1 Reference-frame selection
Depends on: T05

**Why.** Cheapest route to a gain; no retraining. HDR+ picks its reference by sharpness among the first frames;
plain variance-of-Laplacian sharpness already failed here (-0.62 dB), probably because it rewards noise.

**Tasks.**
- [ ] Motion-central frame: the frame with the smallest mean displacement to all others (phase correlation or
      PWC-Net on demosaiced frames).
- [ ] Noise-robust sharpness: gradient energy after smoothing, or after subtracting the estimated noise floor.
- [ ] Maximum overlap with frame 0's field of view (control for the framing effect).
- [ ] Combined score (sharpness within the k most central frames).
- [ ] `--ref_policy` option in `reference_frame_sweep.py` and `Track_2_evaluation.py`.
- [ ] Table against frame 0, best fixed frame and the oracle, on dev then test.

**Done when.** One policy beats frame 0 on the test split with an interval excluding zero, or none does and that
is written down.

## T07: Learned reference selector (stretch)
Labels: track-a-ref-frame, training, stretch
Milestone: M1 Reference-frame selection
Depends on: T06

**Why.** If hand-built scores leave most of the oracle headroom unused, a small model can learn which frame the
network prefers.

**Tasks.**
- [ ] Features per frame: sharpness, displacement to the other frames, overlap, brightness, index.
- [ ] Regressor (gradient-boosted trees or a small MLP) predicting PSNR gain over frame 0, trained on sweep
      results from train scenes.
- [ ] Evaluate on held-out scenes; compare with T06.

**Done when.** Held-out gain is reported next to the best T06 policy.

## T08: Reference-agnostic fine-tuning (stretch)
Labels: track-a-ref-frame, training, stretch
Milestone: M1 Reference-frame selection
Depends on: T02, T06

**Why.** The model was trained with frame 0 as reference every time. Part of the cost of other references may be
the model, not the frames. Training with a random reference removes that.

**Tasks.**
- [ ] Add random `ref_id` per sample to the training dataset.
- [ ] Fine-tune with `finetune_real.py`; re-run the reference sweep and the T06 policies on the new model.

**Done when.** We know whether selector gains grow when the model is not tied to frame 0.

## T09: Degradation harness for stress-testing the model
Labels: track-b-robustness
Milestone: M2 Limits
Depends on: T03

**Why.** To find where Burstormer breaks we need controlled, repeatable degradations of real bursts.

**Tasks.**
- [ ] `robustness_sweep.py` with degradations applied to the raw burst before the model:
  - noise: signal-dependent shot noise + read noise (`data_processing/camera_pipeline.add_noise` model);
  - low light: scale exposure down, add noise, apply gain back;
  - fewer usable frames: replace k frames with copies of the reference;
  - extra misalignment: integer shifts of the non-reference frames;
  - blur: Gaussian blur on a subset of frames;
  - one corrupted frame: strong noise or a large shift on a single frame.
- [ ] Clean burst used for the metric's colour matching; degraded burst only goes to the model.
- [ ] Fixed seeds so every level is reproducible; per-burst CSV and summary per level.
- [ ] Bilinear baseline on the same degraded reference frame at every level.

**Done when.** One command produces a metric-versus-level table for any degradation on any burst list.

## T10: Locate the breaking points
Labels: track-b-robustness
Milestone: M2 Limits
Depends on: T09, T02

**Why.** "Cannot produce a clear result" needs a definition before we can say we moved it.

**Tasks.**
- [ ] Failure criteria, reported side by side: (a) model falls below bilinear upsampling of the same degraded
      reference frame, (b) PSNR drops 3 dB below the clean result, (c) LPIPS exceeds a fixed threshold.
- [ ] Sweep each degradation on the dev split; interpolate the level where each criterion is crossed.
- [ ] Image panels at clean, just before, at, and past the breaking point.
- [ ] Relate synthetic noise levels to real ones (noise profile in the Samsung EXIF data, ISO).

**Done when.** A table lists the breaking level per degradation and criterion, with images that show it.

## T11: Failure analysis: which stage breaks
Labels: track-b-robustness
Milestone: M2 Limits
Depends on: T10

**Why.** The mitigation depends on the mechanism. Noise can break alignment (offset prediction), fusion, or just
pass through upsampling.

**Tasks.**
- [ ] Noise on the reference only versus on the non-reference frames only.
- [ ] Alignment check: predicted offsets on noisy versus clean input for the same burst.
- [ ] Burst benefit: 14 distinct frames versus the reference repeated 14 times, at each noise level. Where the
      two meet, the burst no longer helps.
- [ ] Feature statistics through `conv1 -> EDA -> pooling -> back-projection` on noisy versus clean input.

**Done when.** A short write-up names the stage that fails first, with evidence.

## T12: Training-free mitigation
Labels: track-b-robustness, mitigation
Milestone: M3 Mitigation
Depends on: T11

**Why.** Quick to try, no retraining, and tells us how much of the limit is an input-conditioning problem.

**Tasks.**
- [ ] Pre-denoise each raw frame (Gaussian / bilateral / non-local means after variance stabilisation).
- [ ] Temporal pre-merge: align and average groups of frames before the network.
- [ ] Outlier-frame rejection: drop or down-weight frames that disagree with the reference.
- [ ] Re-run T10 for each; check clean-input regression.

**Done when.** Breaking levels with and without each method are in one table, including the clean-input cost.

## T13: Mitigation by training: noise-augmented fine-tuning and noise conditioning
Labels: track-b-robustness, mitigation, training
Milestone: M3 Mitigation
Depends on: T02, T11

**Why.** The likely strongest fix. The real-data model has only seen the noise levels in BurstSR.

**Tasks.**
- [ ] Noise augmentation in `finetune_real.py` (random shot/read levels from the T09 model).
- [ ] Noise-level conditioning: an extra input channel carrying the per-pixel noise standard deviation.
- [ ] Depending on T11: a noise-aware weighting in fusion (down-weight frames by estimated noise and
      misalignment), in the spirit of Wiener merging in HDR+.
- [ ] Re-run T10; check clean-input regression (target: at most 0.1 dB loss).

**Done when.** The breaking noise level at least doubles at the same failure threshold, within the clean-input
budget, on held-out scenes.

## T14: Final report
Labels: infra
Milestone: M3 Mitigation
Depends on: T06, T13

**Tasks.**
- [ ] Full val-set runs for the final reference policy and the final mitigation.
- [ ] Before/after curves and image panels; tables with intervals.
- [ ] Readme section with the commands to reproduce every number.

**Done when.** Someone outside the project can reproduce the headline numbers from the Readme.
