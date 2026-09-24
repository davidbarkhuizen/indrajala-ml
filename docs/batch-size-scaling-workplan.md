# Workplan: batch-size scaling on full MNIST

**Status: stages 1 and 2 done (2026-09-24); stage 3 next.** Results are in [Results](#results) at the end.

A demo and a measured study: does the linear learning-rate scaling rule (Goyal et al. 2017:
multiply the rate by the factor the batch grows, with warmup) hold for this codebase's dense
MNIST network from batch 32 up to batch 1024? It is also the first demo whose hot path runs
products with a long `k`, which is what decides whether optimization candidate 6 in
[optimizations.md](optimizations.md) matters.

## Why

- **It answers an open question from earlier work.** The learning-rate schedule work (#231-#233)
  found that scaling the rate linearly to batch 128 (`learning_rate=64.0`) diverged to coin-flip
  accuracy, and that `linear_warmup` fixed it, best with `momentum=0.9` at 25 or more warmup
  steps (90.25% ± 2.00% at 50). That was on a 320-example binary digit-3 proxy with only 3
  batches per epoch, 15 steps in all. At 50 warmup steps the ramp never reached its target rate
  (it ended at 19.2), so the result could just as well be "a low rate throughout". Its own
  caveat said a retest at real MNIST scale would have to define warmup in batches again. That
  work also found that `momentum=0.0` recovered much less cleanly (81.38% ± 13.42%), which was
  flagged but not followed up. The written-up findings went with the docs tree in #300. The
  numbers above are from the #233 commit message and diff.
- **No demo trains at a large batch.** Every demo trains single-example or at batch 32, so no
  product with `k` = 512 ever runs outside the batch-512 epochs of the measurement recipe.
  Optimization candidate 6 (dense `accumulate_gradient_batch` at long `k`, 1.6-3.4x numpy on
  one thread) is parked for that reason. This demo would give it a real workload. Whether that
  workload makes candidate 6 worth doing is for stage 3 to measure.
- **It fits the project's direction.** It tests a published claim, records the answer either
  way, and adds no capability: the trainer, the schedule, the momentum networks and the sweep
  runner all exist already.

## The existing pieces

- **Trainer:** `train.train_backprop_network_mini_batch` takes `batch_size`, reshuffles every
  epoch, and accepts `learning_rate` as a float or a schedule function indexed by batch step.
  `learn_batch` averages the gradient over the batch, so the linear rule is `lr_B = lr_32 * B /
  32`.
- **Schedule:** `lr_schedule.linear_warmup(target_rate, warmup_steps)`, which counts steps in
  batches.
- **Networks:** `RustArrayMultiClassBackpropClassifierNetwork` and its momentum sibling
  `MomentumRustArrayMultiClassBackpropClassifierNetwork.randomized(..., momentum)`, plus the numpy
  (`Vectorized...`) versions of both.
- **Data:** `load_mnist_dataset` on the full 60000-row training set and the 10000-row test set,
  as `demo_rust_vs_vectorized_mnist_recognition` uses them. The 2000-row subset and UCI digits
  (1797 rows) are too small: at batch 512 an epoch would be 4 steps.
- **Sweeps:** `benchmark_sweep.run_parameter_sweep`, with `report_progress=True`.
- **Profiling:** `rust_op_breakdown` in `demo_conv_rust_vs_vectorized_digit_recognition` (cProfile
  of Rust time by op), which could be generalized to the dense network.

Architecture: dense 784 -> 30 -> 10, the architecture the existing MNIST demos use. Conv is an
optional extension (stage 5).

## Measurement pitfall to design around

The trainer's pocket snapshot calls `_training_accuracy` before the first epoch and after every
epoch. That is one single-example `classify_state` for each of the 60000 training rows, whatever
the batch size. It is a fixed cost per epoch that doesn't depend on the batch, and at full MNIST
it is probably a large share of each epoch (not measured yet; in the 2000-row conv epoch it was
about a quarter). Epoch wall clock across batch sizes would mostly measure this pass. Stage 3
must time the training steps apart from the accuracy passes.

## Stages

Each stage is one PR. Accuracy runs use the Rust backend only. Timing uses numpy and Rust
only, each in its own process, never interleaved. Pure-Python networks are not run.

### Stage 1: the batch-32 baseline

Find the batch-32 rate that the scaling rule multiplies up. Sweep `learning_rate` over about 5
values around the demos' 0.5, for 5 epochs, 5 seeds, `momentum` 0.0 and 0.9. Report test
accuracy per epoch, mean ± sd. Pick `lr_32` as the best rate whose seeds are all stable (no
seed below chance plus a margin). Record every rate tried, not just the one chosen. This
stage also gives the "batch-32 band" that stage 2 compares against.

As run: the grid was widened to a doubling ladder from 0.0625 to 16, not 5 values around 0.5.
This is Nielsen's 784-30-10 sigmoid/MSE network with an averaged gradient, where rates near 3
are known to work, and momentum 0.9 multiplies the effective rate by about 10, so one ladder
has to bracket both. `lr_32` is picked per momentum, and each momentum gets its own band.
"Stable" is fixed as every seed's final test accuracy at 20% or more (chance plus 10 points).

### Stage 2: the scaling sweep

Fix every choice below before running anything:

- **Batch sizes:** 32, 128, 512, 1024.
- **Rate:** `lr_32 * B / 32` (scaled), and `lr_32` unscaled as a control, so that "larger batch"
  and "scaled rate" can be told apart.
- **Warmup:** none, and `linear_warmup` over a fixed fraction of training, set in epochs and
  converted to steps (`ceil(E_w * 60000 / B)`). Use E_w = 0.25 and 1 epoch. This is the
  batch-count definition #233's caveat asked for.
- **Momentum:** 0.0 and 0.9, to retest the #233 asymmetry at scale.
- **Budget:** 5 epochs at every batch size (equal epochs, so larger batches get fewer steps;
  report steps alongside), 5 seeds.

Report test accuracy per epoch (mean ± sd), and the final result against the batch-32 band.

**Hypotheses and pass criteria**, fixed in advance:

1. Without warmup, the scaled rate diverges from batch 128 up. This is the proxy result
   reproduced at scale.
2. With warmup, the scaled rate ends within the batch-32 band (the mean is within the
   batch-32 seeds' spread) up to some largest batch size. Record which one. Goyal et al. saw
   the rule hold to large batches on ImageNet, but a 5-epoch budget on a small network may
   break it much earlier. Either result gets recorded.
3. The unscaled control falls further behind batch 32 as B grows, because it takes fewer steps
   at the same rate.
4. The #233 momentum asymmetry either survives at scale or it doesn't. No prediction is made.

Estimated run time: 4 batch sizes x 3 warmups x 2 rates x 2 momenta x 5 seeds is 240 runs of 5
epochs, at about 5 s per Rust epoch. That is about 100 minutes of CPU time, about 25-35 minutes
of wall clock on 4 workers. This is an estimate. Get a measured ETA from the sweep's progress
output, and run it in the background. If it is too long, drop the unscaled control at
momentum 0.9 first.

### Stage 3: timing and the op profile

For each batch size, from identical initial weights and with `random.seed` fixed before each
epoch, run one epoch per process for numpy and Rust, median of 5. Time three things
separately: the whole epoch, the step loop (the `learn_batch` calls), and the accuracy
passes. Then profile Rust time by op at B = 512 and 1024 (a dense version of
`rust_op_breakdown`).

**Decision rule for the optimization candidates**, fixed in advance. Estimate each candidate's
stake as time saved per epoch at the measured rates:

- **Candidate 6** (long-`k` accumulate): from the op table, about 0.3-0.6 ms a call over 118
  calls, roughly 35-70 ms, about 1% of an epoch. Promote it only if the profile shows at least
  about 5% of the step loop.
- **Candidate 7** (the dataset as one backend array): per-row tuple-to-array conversion, in
  both the steps and the accuracy pass. If stage 3 already measures conversion at 10% or more of
  the epoch, that meets candidate 7's own go/no-go, so its stage 0 is done.
- **The accuracy pass itself:** if it dominates the epoch, a batched accuracy pass is worth
  considering. That is a trainer change, not a kernel change, and it changes no training
  result.

Record the numbers in optimizations.md against the candidates they bear on, whatever they show.

### Stage 4: the demo

Add `indrajala_ml/demos/demo_batch_size_scaling.py`, registered in `demos/registry.py`. It runs a
reduced version of stage 2 that finishes in a few minutes: batch sizes 32, 128, 512 and 1024,
the scaled rate with and without warmup, `momentum` 0.9, 3 seeds, 3 epochs. It prints test
accuracy by batch size and warmup, plus Rust step-loop time per epoch. It is headless, like the
other timing demos. Add a smoke test on a tiny subset (a few hundred rows, a small batch
ladder) that checks the demo runs and the scaled rate is `lr_32 * B / 32`. Accuracy is not
pinned. Update the README demo list if it lists demos.

### Stage 5 (optional): conv

Repeat stage 2, reduced, for the MNIST conv network (`ConvSpec(3, 8)`, dense 32) on full MNIST.
It is about 20 s per Rust epoch, estimated from the 2000-row epoch, so the sweep needs
cutting down to fit. Go ahead only if stage 2's dense result is interesting enough to test on a
second architecture.

## Out of scope

- New optimizers or schedules beyond `linear_warmup` (no cosine decay, no LARS/LAMB).
- Changing the trainer's pocket snapshot or its accuracy pass as part of this work. Stage 3 may
  recommend a change; that would be its own PR.
- Any kernel change. This plan only measures. Kernel work follows the rules in
  optimizations.md.
- Timing inside the parallel accuracy sweeps. Sweep workers share cores, so their timings mean
  nothing.

## Results

The study's code: `indrajala_ml/batch_size_scaling.py` (the epoch loop, rate scaling, warmup
steps, identical initial weights per seed) and `scripts/batch_size_scaling_sweep.py` (the
sweeps). The epoch loop is the trainer's without the pocket snapshot and its training-accuracy
passes. A test pins it to the trainer: from the same weights and shuffle seed, one epoch leaves
identical weights. Initial weights are drawn by numpy from the seed and restored into the Rust
network, so every cell with the same seed starts from the same weights.

`load_mnist_dataset` now shares one float object per pixel value (256 in all) instead of
boxing 47 million floats. That took a full training-set load from 1.9 GB to 0.45 GB (and 5.1 s
to 2.4 s), with identical values. Without it, 4 sweep workers don't fit in this machine's 5 GB.

### Stage 1: the batch-32 baseline

`python scripts/batch_size_scaling_sweep.py baseline`: batch 32, no warmup, 5 epochs, seeds
0-4, Rust. Test accuracy on the full 10000-row test set, mean ± sd over seeds. 90 runs in about
6 minutes on 4 workers.

**momentum 0.0**

| rate | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 | worst seed | stable |
|---|---|---|---|---|---|---|---|
| 0.0625 | 46.79% ± 2.93% | 72.16% ± 1.60% | 82.99% ± 0.68% | 87.03% ± 0.39% | 88.60% ± 0.34% | 88.22% | yes |
| 0.125 | 72.04% ± 3.47% | 87.10% ± 0.51% | 89.48% ± 0.21% | 90.30% ± 0.13% | 90.90% ± 0.16% | 90.66% | yes |
| 0.25 | 86.84% ± 0.61% | 90.29% ± 0.14% | 91.39% ± 0.12% | 92.02% ± 0.11% | 92.48% ± 0.07% | 92.40% | yes |
| 0.5 | 90.11% ± 0.17% | 91.95% ± 0.12% | 92.69% ± 0.06% | 93.28% ± 0.10% | 93.61% ± 0.14% | 93.41% | yes |
| 1 | 91.77% ± 0.13% | 93.10% ± 0.14% | 93.77% ± 0.13% | 94.33% ± 0.19% | 94.56% ± 0.22% | 94.19% | yes |
| 2 | 92.95% ± 0.19% | 94.01% ± 0.33% | 94.61% ± 0.24% | 95.10% ± 0.17% | 95.16% ± 0.25% | 94.74% | yes |
| **4** | 93.64% ± 0.32% | 94.64% ± 0.34% | 94.93% ± 0.12% | 95.32% ± 0.13% | 95.44% ± 0.44% | 94.70% | yes |
| 8 | 93.34% ± 0.53% | 93.92% ± 0.60% | 94.45% ± 0.48% | 94.67% ± 0.51% | 94.69% ± 0.53% | 94.18% | yes |
| 16 | 11.53% ± 2.90% | 14.08% ± 5.28% | 18.03% ± 10.68% | 19.23% ± 10.07% | 18.58% ± 11.01% | 10.10% | no |

**momentum 0.9**

| rate | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 | worst seed | stable |
|---|---|---|---|---|---|---|---|
| 0.0625 | 90.62% ± 0.16% | 92.29% ± 0.16% | 93.08% ± 0.07% | 93.67% ± 0.10% | 94.03% ± 0.12% | 93.94% | yes |
| 0.125 | 91.93% ± 0.20% | 93.46% ± 0.26% | 94.07% ± 0.13% | 94.64% ± 0.13% | 94.88% ± 0.18% | 94.65% | yes |
| **0.25** | 90.75% ± 3.90% | 93.87% ± 0.46% | 94.50% ± 0.38% | 94.99% ± 0.35% | 95.13% ± 0.35% | 94.76% | yes |
| 0.5 | 63.39% ± 13.09% | 72.73% ± 13.49% | 74.79% ± 13.45% | 80.55% ± 13.86% | 80.55% ± 13.75% | 67.80% | yes |
| 1 | 25.36% ± 10.18% | 41.08% ± 16.63% | 47.05% ± 19.88% | 50.66% ± 17.75% | 52.71% ± 16.79% | 30.90% | yes |
| 2 | 17.35% ± 7.36% | 23.53% ± 9.89% | 27.05% ± 7.83% | 33.51% ± 10.77% | 35.30% ± 14.13% | 20.94% | yes |
| 4 | 10.20% ± 0.68% | 11.78% ± 3.38% | 11.54% ± 2.80% | 11.85% ± 3.18% | 12.78% ± 3.67% | 9.74% | no |
| 8 | 11.54% ± 3.65% | 11.73% ± 4.08% | 11.46% ± 3.47% | 11.59% ± 3.76% | 11.65% ± 3.89% | 8.92% | no |
| 16 | 10.02% ± 0.89% | 10.02% ± 0.89% | 10.02% ± 0.89% | 9.98% ± 0.88% | 9.98% ± 0.88% | 8.92% | no |

**Chosen:**

- momentum 0.0: `lr_32 = 4`. Batch-32 band 95.44% ± 0.44%, seeds 94.70% to 95.88%.
- momentum 0.9: `lr_32 = 0.25`. Batch-32 band 95.13% ± 0.35%, seeds 94.76% to 95.69%.

**Observations:**

- The demos' 0.5 is well below the best batch-32 rate without momentum. It reaches 93.61% here, against 95.44% at 4.
- Both chosen rates sit near the stability edge. Without momentum, 8 still trains and 16
  diverges. With momentum 0.9, 0.5 already loses 15 points and 4 diverges. The 16x momentum ratio (4 against 0.25) is more than the 10x
  that 1/(1 - momentum) predicts, but the grid only resolves factors of 2. Because the chosen
  rates are this close to the edge, stage 2's scaled rates at B >= 128 start well past the
  largest stable batch-32 rate. That is the regime the warmup is meant to handle.
- The 20% stability bar is loose: at momentum 0.9 it admits rates 0.5-2, which are badly
  degraded. It didn't affect the choice, which takes the best mean among stable rates.
- At momentum 0.9 and rate 0.25, epoch 1 has one slow seed (sd 3.90%). It has recovered by
  epoch 2.

### Stage 2: the scaling sweep

`python scripts/batch_size_scaling_sweep.py scaling --lr32 0.0=4 --lr32 0.9=0.25`: every cell
from the plan, 5 epochs, seeds 0-4, Rust. At B = 32 the scaled and unscaled rates are the same,
so that cell ran once. That makes 210 runs, which took 16 minutes on 4 workers. The band is the
batch-32, no-warmup seeds' final accuracy, from lowest to highest. "In band" means the cell's
final mean lies inside it, and "above" would mean over the top of it (no cell was).

**momentum 0.0, `lr_32` = 4** (band 94.70% to 95.88%)

| B | rate | value | warmup epochs (steps) | total steps | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 | in band |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | scaled | 4 | 0 (0) | 9375 | 93.64% ± 0.32% | 94.64% ± 0.34% | 94.93% ± 0.12% | 95.32% ± 0.13% | 95.44% ± 0.44% | yes |
| 32 | scaled | 4 | 0.25 (469) | 9375 | 93.35% ± 0.26% | 94.47% ± 0.27% | 95.13% ± 0.23% | 95.36% ± 0.18% | 95.40% ± 0.12% | yes |
| 32 | scaled | 4 | 1 (1875) | 9375 | 92.35% ± 0.32% | 94.10% ± 0.37% | 94.91% ± 0.18% | 95.40% ± 0.11% | 95.41% ± 0.19% | yes |
| 128 | scaled | 16 | 0 (0) | 2345 | 10.20% ± 0.68% | 11.29% ± 1.28% | 16.39% ± 7.72% | 23.35% ± 14.82% | 24.28% ± 16.13% | no |
| 128 | unscaled | 4 | 0 (0) | 2345 | 91.56% ± 0.21% | 92.94% ± 0.16% | 93.72% ± 0.15% | 94.37% ± 0.16% | 94.52% ± 0.25% | no |
| 128 | scaled | 16 | 0.25 (118) | 2345 | 92.98% ± 0.52% | 94.21% ± 0.41% | 94.87% ± 0.14% | 95.02% ± 0.14% | 95.39% ± 0.15% | yes |
| 128 | unscaled | 4 | 0.25 (118) | 2345 | 91.28% ± 0.16% | 92.89% ± 0.20% | 93.68% ± 0.19% | 94.30% ± 0.20% | 94.51% ± 0.21% | no |
| 128 | scaled | 16 | 1 (469) | 2345 | 91.52% ± 1.25% | 93.76% ± 0.33% | 94.64% ± 0.20% | 95.11% ± 0.17% | 95.28% ± 0.31% | yes |
| 128 | unscaled | 4 | 1 (469) | 2345 | 89.72% ± 0.32% | 92.44% ± 0.25% | 93.48% ± 0.20% | 94.16% ± 0.27% | 94.41% ± 0.30% | no |
| 512 | scaled | 64 | 0 (0) | 590 | 11.47% ± 2.49% | 11.47% ± 2.49% | 11.47% ± 2.49% | 11.47% ± 2.49% | 11.47% ± 2.49% | no |
| 512 | unscaled | 4 | 0 (0) | 590 | 84.28% ± 1.70% | 89.74% ± 0.34% | 91.16% ± 0.21% | 91.75% ± 0.14% | 92.14% ± 0.30% | no |
| 512 | scaled | 64 | 0.25 (30) | 590 | 10.06% ± 0.19% | 9.92% ± 0.24% | 10.19% ± 1.27% | 11.99% ± 3.52% | 11.90% ± 3.58% | no |
| 512 | unscaled | 4 | 0.25 (30) | 590 | 83.29% ± 2.47% | 89.66% ± 0.49% | 91.18% ± 0.21% | 91.80% ± 0.19% | 92.17% ± 0.33% | no |
| 512 | scaled | 64 | 1 (118) | 590 | 11.35% ± 2.14% | 11.48% ± 3.20% | 9.79% ± 1.00% | 11.92% ± 4.97% | 11.91% ± 4.96% | no |
| 512 | unscaled | 4 | 1 (118) | 590 | 69.48% ± 3.11% | 88.83% ± 0.69% | 90.85% ± 0.25% | 91.54% ± 0.23% | 92.02% ± 0.35% | no |
| 1024 | scaled | 128 | 0 (0) | 295 | 10.33% ± 0.60% | 10.33% ± 0.60% | 10.33% ± 0.60% | 10.33% ± 0.60% | 10.33% ± 0.60% | no |
| 1024 | unscaled | 4 | 0 (0) | 295 | 66.62% ± 5.49% | 86.15% ± 0.54% | 89.16% ± 0.15% | 90.18% ± 0.07% | 90.78% ± 0.09% | no |
| 1024 | scaled | 128 | 0.25 (15) | 295 | 10.19% ± 0.25% | 10.19% ± 0.25% | 9.54% ± 0.61% | 9.71% ± 0.50% | 9.65% ± 0.43% | no |
| 1024 | unscaled | 4 | 0.25 (15) | 295 | 68.33% ± 4.09% | 86.42% ± 0.51% | 89.25% ± 0.30% | 90.16% ± 0.08% | 90.77% ± 0.14% | no |
| 1024 | scaled | 128 | 1 (59) | 295 | 10.16% ± 0.86% | 9.87% ± 0.58% | 9.91% ± 0.60% | 9.87% ± 0.58% | 9.87% ± 0.58% | no |
| 1024 | unscaled | 4 | 1 (59) | 295 | 46.53% ± 4.52% | 83.38% ± 0.58% | 88.54% ± 0.39% | 89.87% ± 0.11% | 90.56% ± 0.12% | no |

**momentum 0.9, `lr_32` = 0.25** (band 94.76% to 95.69%)

| B | rate | value | warmup epochs (steps) | total steps | epoch 1 | epoch 2 | epoch 3 | epoch 4 | epoch 5 | in band |
|---|---|---|---|---|---|---|---|---|---|---|
| 32 | scaled | 0.25 | 0 (0) | 9375 | 90.75% ± 3.90% | 93.87% ± 0.46% | 94.50% ± 0.38% | 94.99% ± 0.35% | 95.13% ± 0.35% | yes |
| 32 | scaled | 0.25 | 0.25 (469) | 9375 | 92.84% ± 0.21% | 94.16% ± 0.26% | 94.72% ± 0.06% | 95.13% ± 0.15% | 95.39% ± 0.15% | yes |
| 32 | scaled | 0.25 | 1 (1875) | 9375 | 91.64% ± 0.21% | 93.84% ± 0.27% | 94.62% ± 0.11% | 95.15% ± 0.10% | 95.38% ± 0.16% | yes |
| 128 | scaled | 1 | 0 (0) | 2345 | 14.01% ± 4.76% | 15.89% ± 4.77% | 15.93% ± 4.80% | 15.96% ± 4.83% | 18.08% ± 7.60% | no |
| 128 | unscaled | 0.25 | 0 (0) | 2345 | 80.36% ± 11.62% | 88.95% ± 4.36% | 90.84% ± 4.11% | 91.43% ± 4.12% | 93.51% ± 0.19% | no |
| 128 | scaled | 1 | 0.25 (118) | 2345 | 92.85% ± 0.15% | 94.30% ± 0.14% | 94.72% ± 0.11% | 95.18% ± 0.12% | 95.39% ± 0.11% | yes |
| 128 | unscaled | 0.25 | 0.25 (118) | 2345 | 90.17% ± 0.23% | 92.22% ± 0.14% | 93.00% ± 0.12% | 93.62% ± 0.14% | 93.91% ± 0.17% | no |
| 128 | scaled | 1 | 1 (469) | 2345 | 91.79% ± 0.14% | 93.90% ± 0.20% | 94.64% ± 0.21% | 95.16% ± 0.06% | 95.28% ± 0.05% | yes |
| 128 | unscaled | 0.25 | 1 (469) | 2345 | 87.79% ± 0.61% | 91.77% ± 0.10% | 92.77% ± 0.13% | 93.42% ± 0.15% | 93.78% ± 0.12% | no |
| 512 | scaled | 4 | 0 (0) | 590 | 11.02% ± 1.53% | 11.02% ± 1.53% | 11.02% ± 1.53% | 11.02% ± 1.53% | 11.02% ± 1.53% | no |
| 512 | unscaled | 0.25 | 0 (0) | 590 | 30.48% ± 6.76% | 61.67% ± 8.80% | 74.99% ± 11.11% | 81.17% ± 10.72% | 85.05% ± 7.75% | no |
| 512 | scaled | 4 | 0.25 (30) | 590 | 82.88% ± 12.04% | 89.82% ± 5.11% | 90.56% ± 4.58% | 92.76% ± 3.77% | 93.08% ± 3.93% | no |
| 512 | unscaled | 0.25 | 0.25 (30) | 590 | 62.20% ± 4.78% | 86.57% ± 0.41% | 89.60% ± 0.23% | 90.60% ± 0.12% | 91.23% ± 0.08% | no |
| 512 | scaled | 4 | 1 (118) | 590 | 90.62% ± 0.25% | 93.57% ± 0.20% | 94.24% ± 0.19% | 94.91% ± 0.18% | 95.09% ± 0.09% | yes |
| 512 | unscaled | 0.25 | 1 (118) | 590 | 44.28% ± 2.96% | 84.33% ± 0.78% | 89.23% ± 0.25% | 90.36% ± 0.19% | 91.10% ± 0.14% | no |
| 1024 | scaled | 8 | 0 (0) | 295 | 10.33% ± 0.60% | 10.33% ± 0.60% | 10.33% ± 0.60% | 10.33% ± 0.60% | 10.33% ± 0.60% | no |
| 1024 | unscaled | 0.25 | 0 (0) | 295 | 13.31% ± 4.04% | 30.78% ± 5.47% | 44.85% ± 4.93% | 62.30% ± 8.46% | 69.84% ± 9.62% | no |
| 1024 | scaled | 8 | 0.25 (15) | 295 | 36.90% ± 10.54% | 44.73% ± 18.12% | 46.06% ± 17.95% | 47.44% ± 18.02% | 47.50% ± 18.06% | no |
| 1024 | unscaled | 0.25 | 0.25 (15) | 295 | 36.22% ± 5.71% | 57.84% ± 5.32% | 77.29% ± 2.89% | 85.90% ± 0.60% | 88.41% ± 0.12% | no |
| 1024 | scaled | 8 | 1 (59) | 295 | 71.19% ± 10.87% | 86.79% ± 8.39% | 90.07% ± 4.77% | 90.73% ± 4.83% | 92.81% ± 3.82% | no |
| 1024 | unscaled | 0.25 | 1 (59) | 295 | 30.67% ± 8.84% | 60.35% ± 1.88% | 78.56% ± 1.47% | 86.25% ± 0.36% | 88.54% ± 0.33% | no |

**Against the hypotheses:**

1. **Without warmup, the scaled rate diverges from batch 128 up: confirmed** at both momenta.
   At B = 128 it ends at 24.28% ± 16.13% (momentum 0.0) and 18.08% ± 7.60% (0.9). At 512
   and 1024 it stays at chance from the first epoch. This is the #233 proxy result at full
   scale.
2. **With warmup, the rule holds to:**
   - **B = 128 without momentum**, with either warmup (95.39% at 0.25 epochs, 95.28% at 1).
     At 512 and 1024 (rates 64 and 128) it fails completely, even with a 1-epoch warmup:
     every run stays near chance.
   - **B = 512 with momentum 0.9**, with the 1-epoch warmup only (95.09% ± 0.09%, 16x
     fewer steps than batch 32). With the 0.25-epoch warmup, B = 512 misses the band (93.08%
     ± 3.93%: one seed lags). At 1024 the 1-epoch warmup reaches 92.81% ± 3.82%, below the
     band.

   So a 5-epoch budget on this network breaks the rule at 4x to 16x the base batch, not at
   the thousands Goyal et al. reached. A likely cause, not tested here: the stable rate has
   a ceiling set by the loss surface's curvature, not by gradient noise. Without momentum,
   the largest rate that trained at batch 32 was 8, and 16 at B = 128 still trained after
   warmup, but 64 at B = 512 did not, with any warmup. A larger batch only removes gradient
   noise, and the rule cannot push the rate past that ceiling. Longer warmup doesn't help
   without momentum: 1 epoch is no better than 0.25 at any size.
3. **The unscaled control falls further behind as B grows: confirmed.** Without momentum it
   ends at 94.52%, 92.14% and 90.78% at B = 128, 512 and 1024 (no warmup). With momentum 0.9
   it ends at 93.51%, 85.05% and 69.84%. Where the scaled rate diverges (B >= 512, momentum
   0.0), the unscaled control is 80 points better. Where the rule is past its limit, not
   scaling the rate is the better choice.
4. **The #233 momentum asymmetry survives in the same direction**: momentum 0.9 with warmup
   holds the rule to a 4x larger batch than momentum 0.0. But the comparison is confounded.
   Momentum 0.9's effective batch-32 rate, 0.25 / (1 - 0.9) = 2.5, is lower than momentum
   0.0's 4, so at every B its effective scaled rate is lower too (40 against 64 at B = 512).
   The stage 1 grid picked each momentum's best rate. Separating momentum from the effective
   rate would need a matched-effective-rate run (for example momentum 0.9 at `lr_32` = 0.4).
   That is not done here.

**Other findings:**

- **Warmup costs nothing at batch 32** (95.40% and 95.41% against 95.44%, momentum 0.0). At
  momentum 0.9 it helps (95.39% and 95.38% against 95.13%), because it removes the one slow
  seed in epoch 1.
- **Momentum 0.9 has an early instability that doesn't depend on scaling.** The unscaled
  control, at the batch-32 rate, has a large spread in its early epochs without warmup: at
  B = 128, 80.36% ± 11.62% after epoch 1, and at 1024, 69.84% ± 9.62% at the end. A 1-epoch
  warmup makes it tight (±0.33% at 1024). The same momentum 0.9 rate is stable at batch 32
  from epoch 2, so the larger batch adds instability at the same rate. This was not
  predicted, and it is not explained here.
