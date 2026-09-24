# Workplan: batch-size scaling on full MNIST

**Status: planned, not started (2026-09-24).**

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
