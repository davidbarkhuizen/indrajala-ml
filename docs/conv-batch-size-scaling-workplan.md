# Workplan: batch-size scaling for the conv network

**Status: not started.**

A measured study and a demo: does the linear learning-rate scaling rule (Goyal et al. 2017:
multiply the rate by the factor the batch grows, with warmup) hold for the conv network on full
MNIST, from batch 32 up to batch 512? The dense study (`indrajala_ml/batch_size_scaling.py`)
answered this for 784 -> 30 -> 10. No conv network has been tested.

It would also be the first demo to train conv at N = 512. That is the workload optimization
candidate 1 needs ([candidates.md](optimizations/candidates.md), "Conv accumulate with a large
`cols`"). The demo has to be worth running for its own question. It must not exist only to give
the optimization a workload.

## Why

- **It tests a published claim on a second architecture, and records the answer either way.**
  The dense result is specific: without warmup, the scaled rate diverged from B = 128 up. With a
  one-epoch warmup, the rule held to B = 128 at momentum 0.0 and to B = 512 at momentum 0.9.
  Whether conv behaves the same is open. Conv has fewer, shared weights and a different
  curvature, so the dense result doesn't carry over by assumption.
- **No demo trains conv at a large batch.** Every conv demo trains single-example or at batch 32
  on 2000 rows. At N = 32 the conv accumulate is 1.1-1.2x its single calls, and about 1-1.5% of
  an epoch. So candidate 1 is ranked with a small stake.
- **At B = 512, large batches don't pay off per example yet.** From a scratch probe on the
  current build (below), a B = 512 conv step costs about what B = 32 does per example. For
  conv-pool-conv it costs 28% more. A study that times the step loop at each batch size records
  that, and gives any fix a before and after measured on a real training run.

## What is measured so far

A scratch probe on the current build (2026-09-25): Rust, full MNIST, default threading (the
8M-flop threshold), `learn_batch_rows` on a prepared dataset. The first 8192 shuffled examples
are timed plainly, then profiled with cProfile. It was one run per cell in its own process, so
read the numbers as rough:

| architecture (dense 32) | µs per example, B = 32 | µs per example, B = 512 | `conv_accumulate_gradient_batch` share, B = 32 → 512 |
| --- | --- | --- | --- |
| `[ConvSpec(3, 8)]` | 172 | 164 | 21% → 34% |
| `[ConvSpec(3, 8), PoolSpec(2), ConvSpec(3, 8)]` | 239 | 305 | 30% → 31% |

- **The threaded stake is smaller than the one-thread op figure.** candidates.md quotes 57-58 ms
  per B = 512 accumulate call on one thread. In a threaded training step the accumulate takes
  about 28 ms (34% of 164 µs × 512). If the fix brought it back to its B = 32 cost per example,
  it would save about 7-12% of a B = 512 step. The candidate's own stage 0 must re-measure this.
- **For conv-pool-conv, the accumulate is only part of the B = 512 slowdown.** It accounts for
  about 22 of the 66 extra µs per example. Downstream (about 20) and forward also grow.
- Estimated step loop per full epoch: about 10 s for conv and 14-18 s for conv-pool-conv, at
  either batch size. The test-set pass comes on top.

## The existing pieces

- **Study code:** `batch_size_scaling.py` has `scaled_learning_rate`, `warmup_steps`,
  `learning_rate_schedule`, `train_epoch` (the trainer's loop without the pocket snapshot, pinned
  to the trainer by a test) and `train_and_evaluate`. `initial_network` is dense-only. It draws
  the weights with numpy from the seed and restores them into the Rust network.
  `scripts/batch_size_scaling_sweep.py` runs the sweeps through `run_parameter_sweep`.
  `scripts/batch_size_timing.py` does the timing.
- **Networks:** `ConvRustArrayMultiClassBackpropClassifierNetwork` and its numpy sibling, both
  `ArrayConvShape` over each backend's plain multiclass network. **Neither has momentum.**
  `MomentumRustArrayLayer` updates through `pa.layer_momentum_apply_accumulated_gradient`, which
  takes a `(W, b)` pair of any shape. A conv `W` is `(channel_count, fan_in)`, so a momentum conv
  layer may need no crate change. That is unverified.
- **Architecture:** the conv demo's `[ConvSpec(3, 8)]` with dense `[32]`. conv-pool-conv only for
  timing (see Stages).
- **Profiling:** `scripts/epoch_op_profile.py` profiles the conv demo's 2000-row subset only.

## Pitfalls to design around

- **The 1-ULP control.** Conv training is chaotically sensitive: a 1-ULP change flips
  end-of-run results. Compare accuracy across seeds (mean ± sd), never one run's final accuracy
  against another's.
- **The test pass is per row for Rust conv.** Batched Rust conv inference measured slower.
  10000 test rows are a fixed cost per epoch, so time the step loop apart from it, as the dense
  study does.
- **Tuple batches versus prepared rows.** `train_epoch` calls `learn_batch` on tuple lists. For
  conv at 60000 rows, the conversion may be a visible share of a step. Before switching the loop
  to `prepare_dataset` and `learn_batch_rows`, measure the conversion. If the loop changes,
  re-pin it to the trainer.
- **Timing rules.** numpy and Rust each in their own process, never interleaved. No timing
  inside the parallel accuracy sweeps, whose workers share cores. Pure Python is never run.
- **Memory.** Each sweep worker loads full MNIST itself (about 0.5 GB), which caps the worker
  count.

## Stages

Each stage is one PR. The accuracy runs use the Rust backend.

### Stage 1: conv in the study code, and the batch-32 baseline

Give `batch_size_scaling.py` an architecture parameter (dense or conv), keeping the dense path
bit-identical. The conv path uses `initial_network`'s scheme: numpy draws from the seed, and the
weights are restored into the Rust network. Then sweep the batch-32 rate for conv at momentum 0.0
over a doubling ladder, about 0.125 to 8, for 3 seeds and 2 epochs. Pick `lr_32` as the dense
study did: the best rate whose seeds are all stable (every final test accuracy 20% or more).
Record every rate tried.

### Stage 2: does B = 512 train at momentum 0.0?

The go/no-go question. At B = 128 and 512, run the scaled rate (`lr_32 * B / 32`) with no warmup
and with a one-epoch warmup, plus the unscaled rate as the control. 3 seeds, 3 epochs.

- **If the scaled rate with warmup reaches the batch-32 band at B = 512:** go to stage 4.
- **If it diverges, or holds only to B = 128 as dense did at momentum 0.0:** record that and go to
  stage 3.

### Stage 3 (only if stage 2 fails): momentum conv

A momentum sibling of each conv network, numpy and Rust: a conv layer subclass whose
`apply_accumulated_gradient` keeps previous-delta state, as `MomentumArrayLayer` and
`MomentumRustArrayLayer` do. Build it only if it's worth having in its own right, since it adds a
network family. It needs parity tests (numpy against Rust step by step, within the 1-ULP control)
and a save/load round trip. Then rerun stage 1 at momentum 0.9 and stage 2 with it.

If momentum conv isn't worth building, stop at stage 2. Record the null in the study's findings,
and record in candidates.md that no trained configuration runs conv at N = 512.

### Stage 4: the scaling sweep and the timing

- **Accuracy:** B = 32, 128, 512, and 1024 if 512 held; scaled and unscaled rate; warmup 0, 0.25
  and 1 epoch; 5 seeds; 3 epochs; at each momentum that trained in stage 2 or 3.
- **Timing:** the step loop's µs per example at each batch size, numpy and Rust, for conv and
  conv-pool-conv, each in its own process. A profiled run gives each crate op's share. Commit the
  scratch probe as a script (generalizing `epoch_op_profile.py` to full MNIST and a batch size
  would also do). Update [current-baseline.md](optimizations/current-baseline.md) with the
  numbers, and candidate 1's stake in [candidates.md](optimizations/candidates.md) with the
  measured B = 512 share.

Write the findings into `batch_size_scaling.py`'s docstring, next to the dense ones.

### Stage 5: the demo

`demo_conv_batch_size_scaling`, a reduced sweep as `demo_batch_size_scaling` is for dense: the
batch sizes that held, the stage 2 rate, warmup 0 and 1 epoch, 3 seeds, a few epochs. It reports
test accuracy (mean, min, max) and the median step-loop seconds per epoch. Add it to
`demos/registry.py` with a run time estimate. It must fit in about 15 minutes. At about 10 s per
epoch plus the test pass, 3 batch sizes × 2 warmups × 3 seeds × 3 epochs is about 10-12 minutes
serial.

## After this plan

Candidate 1 follows the rules in [optimizations.md](optimizations.md), with its own stage 0.
This demo's step loop is its end-to-end measurement at B = 512. For conv-pool-conv, the
downstream and forward growth at B = 512 are separate leads.

## Out of scope

- New optimizers or schedules beyond `linear_warmup` and momentum.
- Batched Rust conv inference for the test pass (a lead in candidates.md, measured slower so far).
- Any kernel change. This plan only measures.
- Architectures beyond the conv demo's. conv-pool-conv is timed but not swept for accuracy.
