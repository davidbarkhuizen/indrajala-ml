# Workplan: batch-size scaling for the conv network

**Status: stages 1 and 2 done. At momentum 0.0 the rule holds to B = 128 with warmup and fails
at B = 512, where no rate reaches the band. Stage 3 (momentum conv, planned in detail below) is
next, starting with 3a.**

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

### Stage 1 (done): conv in the study code, and the batch-32 baseline

`batch_size_scaling.py` takes `architecture="dense"` or `"conv"`. The dense path is unchanged.
The conv path draws the weights with numpy from the seed and restores them into the Rust network.
The sweep script takes `--architecture conv`, which sweeps momentum 0.0 only:

    python scripts/batch_size_scaling_sweep.py baseline --architecture conv --epochs 2 --seeds 3

Result: full MNIST, Rust, momentum 0.0, 3 seeds, 2 epochs, final test accuracy:

| rate | epoch 1 | epoch 2 | worst seed | stable |
| --- | --- | --- | --- | --- |
| 0.125 | 89.92% ± 0.48% | 91.71% ± 0.39% | 91.33% | yes |
| 0.25 | 91.46% ± 0.35% | 92.56% ± 0.38% | 92.20% | yes |
| 0.5 | 92.10% ± 0.38% | 93.26% ± 0.18% | 93.14% | yes |
| 1 | 92.64% ± 0.38% | 95.62% ± 0.39% | 95.23% | yes |
| **2** | 95.45% ± 0.28% | **96.82% ± 0.39%** | 96.39% | yes |
| 4 | 95.55% ± 0.29% | 96.17% ± 0.61% | 95.73% | yes |
| 8 | 83.97% ± 8.41% | 89.56% ± 4.08% | 86.37% | yes |
| 16 | 10.08% ± 0.25% | 10.25% ± 0.13% | 10.10% | no |

- **`lr_32` = 2** (band 96.39% - 97.14%). It is the best rate whose seeds all finish at 20% or
  more. The conv demo's 0.5 reaches only 93.26%.
- **The stability edge is between 8 and 16, as it is for dense at momentum 0.0.** 8 is erratic:
  one seed fell between epochs, from 91.0% to 88.2%. 16 stays at chance on every seed. So stage
  2's scaled rates (8 at B = 128, 32 at B = 512) reach and pass the rate that diverges at B = 32.
  Whether warmup or the larger batch lifts that ceiling is the stage 2 question.
- **The tuple conversion is about 6% of the step loop.** `train_epoch` still calls `learn_batch`
  on tuples. The conversion can't change accuracy, so the loop is unchanged. Stage 4's timing
  should switch to prepared rows, or report the conversion separately (a scratch probe:
  0.10 s of 1.58 s for 8192 rows at B = 32).

### Stage 2 (done): does B = 512 train at momentum 0.0? No.

The go/no-go question. At B = 128 and 512, the scaled rate (`lr_32 * B / 32`) with no warmup and
with a one-epoch warmup, plus the unscaled rate as the control. 3 seeds, 3 epochs:

    python scripts/batch_size_scaling_sweep.py scaling --architecture conv --lr32 0.0=2 \
        --batch-sizes 32 128 512 --warmups 0 1 --epochs 3 --seeds 3

Result: full MNIST, Rust, momentum 0.0, `lr_32` = 2. The batch-32 band (no warmup, epoch 3) is
97.16% ± 0.61%, seeds 96.45% - 97.53%.

| B | rate | value | warmup epochs (steps) | total steps | epoch 1 | epoch 2 | epoch 3 | in band |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | scaled | 2 | 0 (0) | 5625 | 95.45% ± 0.28% | 96.82% ± 0.39% | 97.16% ± 0.61% | yes |
| 32 | scaled | 2 | 1 (1875) | 5625 | 91.72% ± 1.52% | 96.47% ± 0.26% | 96.98% ± 0.33% | yes |
| 128 | scaled | 8 | 0 (0) | 1407 | 87.67% ± 7.94% | 94.90% ± 0.73% | 96.23% ± 0.33% | no |
| 128 | unscaled | 2 | 0 (0) | 1407 | 91.46% ± 0.66% | 92.89% ± 0.99% | 94.80% ± 0.34% | no |
| **128** | **scaled** | **8** | **1 (469)** | 1407 | 92.21% ± 0.39% | 96.13% ± 0.16% | **96.63% ± 0.72%** | **yes** |
| 128 | unscaled | 2 | 1 (469) | 1407 | 88.86% ± 1.06% | 92.11% ± 0.61% | 93.65% ± 0.62% | no |
| 512 | scaled | 32 | 0 (0) | 354 | 10.00% ± 0.38% | 10.00% ± 0.38% | 10.00% ± 0.38% | no |
| 512 | unscaled | 2 | 0 (0) | 354 | 80.06% ± 5.57% | 86.88% ± 2.15% | 90.26% ± 2.64% | no |
| 512 | scaled | 32 | 1 (118) | 354 | 11.08% ± 1.14% | 10.64% ± 0.63% | 13.94% ± 6.19% | no |
| 512 | unscaled | 2 | 1 (118) | 354 | 69.59% ± 1.83% | 85.25% ± 2.30% | 90.69% ± 0.72% | no |

- **The rule holds to B = 128, with a one-epoch warmup only.** Without warmup the scaled rate 8
  reaches 96.23% but misses the band, and one seed fell to 78.6% in epoch 1. This matches dense
  at momentum 0.0.
- **At B = 512 the scaled rate 32 stays at chance, with or without warmup.** It is 2-4x the rate
  that already diverges at B = 32 (16). As for dense, the larger batch did not lift that ceiling.
  One warmup seed reached 21.1% in epoch 3, so the rate is not fully dead after the ramp, but it
  is far from training.
- **The unscaled control at B = 512 reaches about 90.5%, 6.5 points below the band.** It is still
  climbing, and it has 16x fewer steps than B = 32.
- **A capped rate at B = 512 trains, but it doesn't reach the band in 3 epochs.** A scratch probe
  (the sweep's `run_config`, B = 512, one-epoch warmup, seeds 0-2, 3 epochs) tested rates below
  the scaled 32:

  | rate | epoch 1 | epoch 2 | epoch 3 | worst seed |
  | --- | --- | --- | --- | --- |
  | 4 | 77.93% ± 1.52% | 87.00% ± 2.28% | 91.18% ± 1.04% | 90.22% |
  | 8 | 84.61% ± 3.45% | 87.41% ± 4.06% | 92.29% ± 1.68% | 90.55% |
  | 16 | 75.40% ± 9.19% | 92.73% ± 0.55% | 94.63% ± 1.02% | 93.46% |

  Rate 16 is stable at B = 512 with warmup, though it stays at chance at B = 32 without warmup.
  So the stable rate rises by at least 2x, less than the 16x of the linear rule (32 stays at
  chance). The best capped rate finishes 2.5 points below the band and is still
  climbing.
- **The stable rate at B = 512 tops out between 16 and 24, and the larger batch does raise it.**
  Two more probe cells (same setup): rate 24 at B = 512 is erratic (29.12% ± 32.73%; one seed
  reached 66.9%, two stayed at chance). Rate 16 at B = 32 with warmup trains badly (67.15% ±
  3.07%, falling from 79.90% after the warmup epoch), where at B = 512 it reaches 94.63%. So the
  batch, not the warmup, carries rate 16, and the stable rate grows about 2x for a 16x batch.
- Warmup costs little at B = 32 (96.98% against 97.16%, within the spread).

So the plan goes to stage 3.

### Stage 3: momentum conv

At momentum 0.0 the conv network plateaus below the band at B = 512 (stage 2), so momentum is the
remaining lever. Dense needed momentum 0.9 to reach B = 512. Momentum conv is also worth having in
its own right: the README lists `Momentum` and `Conv` as features of all three implementations,
and no implementation combines them yet.

Each sub-stage is one PR (3a is a crate PR first, then the parent PR that moves `rust/`).

#### 3a: numpy and Rust apply the update in the same order

The Rust update ops group their arithmetic differently from numpy and pure Python, which are the
references (`fused.rs`):

| update | numpy and pure Python | Rust today | same bits |
| --- | --- | --- | --- |
| `layer_apply_accumulated_gradient` (dense and conv) | `(lr * g) / B` | `(lr / B) * g` | only for B a power of two |
| `layer_momentum_apply_accumulated_gradient` | `(lr * g) / B + m * prev` | `(lr / B) * g + m * prev` | only for B a power of two |
| `layer_l2_apply_accumulated_gradient` (W) | `lr * (g / B + λ * w)` | `w - (lr / B) * g - lr * λ * w` | no |
| Adam, and `layer_sgd_step` (B = 1) | | | yes |

Dividing by a power of two is exact, so the study's B = 32, 128 and 512 agree. But the last,
partial batch of an epoch doesn't (60000 / 128 and 60000 / 512 both leave 96 rows), nor do the
tests' batch sizes such as 6. So a numpy against Rust difference can come from the update's
grouping rather than from BLAS, and conv training's chaotic sensitivity turns that into different
end-of-run results.

- **Change:** the three Rust ops compute each element exactly as the numpy expression does, same
  operations in the same order. numpy is canonical: the README names it as the reference for Rust,
  pure Python computes the same `learning_rate * accum / batch_size`, and the ops' doc comments
  already quote the numpy expressions. Neither grouping is more accurate (both round twice).
- **Tests (crate and parent):** the fused-op tests compare against the numpy layers with exact
  equality (`==` on the bits), not a tolerance, at batch sizes 1, 6 and 96 as well as powers of
  two. Mutation check: each new exact test must fail against the current ops.
- **Golden run:** this is a numerics change, not a refactoring. The golden run's batches are 4
  rows, so the prediction is bit-identical checkpoints for every network except the L2 ones.
  Check the prediction, then re-record.
- **Timing:** the apply ops now divide per element instead of multiplying. Time the step loop
  before and after (`scripts/prepared_dataset_timing.py time`, both builds committed first,
  separate processes). Expect noise: the apply is one pass over the parameters per batch.
- **Out of scope:** reduction order in matmuls and gradient accumulation. OpenBLAS's blocking
  isn't reproducible, and that is what the 1-ULP control covers.

#### 3b: the conv networks honor hyperparameters (structural, bit-identical)

The conv networks can't host a hyperparameter-bearing layer today:

- `build_conv_array_network_layers` calls `dense_cls(size, previous_size)` directly, bypassing
  `ArrayNetworkBase._new_layer`, which passes `layer_cls.hyperparameters` from the network. Change
  it to take a layer factory, `network._new_layer`, for the dense and conv layers alike (conv
  layer classes get `hyperparameters = ()`).
- The conv save envelope (`save_conv_model_json` / `load_conv_model_json`) has no `extra`. Add it
  as `save_array_model_json` has it: `ArrayConvShape.save` passes `_extra_state()`, and `load`
  passes `_extra_init_kwargs(state)` to the constructor. Files without extra keys load unchanged.
- The pure-Python `ConvMultiClassBackpropClassifierNetwork` hard-codes `BackpropLayer` for its
  dense layers and `ConvKernel` inside `ConvLayer`. Add class-level hooks (`dense_layer_cls`, and
  a kernel class on `ConvLayer`) defaulting to today's classes.

Follows the README's Refactoring rules: golden run bit-identical, no hot-path change (only
construction and save/load are touched, so no timing), public names and saved files unchanged.

#### 3c: the pure-Python momentum conv reference

- `MomentumConvKernel`: `ConvKernel` with the previous deltas, zero-initialized, and the update of
  `make_momentum_node_cls`: `Δw = lr * accum / B + m * prev`, positions summed and examples
  averaged as `ConvKernel` does. A factory, as `make_momentum_node_cls`, because momentum has no
  default.
- `MomentumConvMultiClassBackpropClassifierNetwork(..., momentum)`: momentum kernels in the conv
  layers and `make_momentum_layer_cls(momentum)` for the dense and output layers. Pool layers are
  unchanged: they have no weights.
- **Tests:** at momentum 0.0, bit-identical to `ConvMultiClassBackpropClassifierNetwork` through
  `learn` and `learn_batch` (`x + 0.0 * prev` is `x`). A hand-computed two-step kernel update. A
  save/load round trip that keeps `momentum`. No new gradient check: momentum changes only the
  update, and the gradients are covered by the existing conv checks.
- The previous deltas are not saved, as for every momentum network (`snapshot()` covers W and b).

#### 3d: numpy and Rust momentum conv

- **Layers:** `MomentumConvArrayLayer(ConvArrayLayer)` and
  `MomentumConvRustArrayLayer(ConvRustArrayLayer)`, with `hyperparameters = ("momentum",)` and
  previous-delta state shaped as `W` `(channel_count, fan_in)` and `b` `(channel_count,)`. numpy
  uses `MomentumArrayLayer`'s expression. Rust calls
  `pa.layer_momentum_apply_accumulated_gradient`, which only checks that shapes match, so it takes
  the conv shapes with no crate change.
- **Networks:** `MomentumConvVectorizedMultiClassBackpropClassifierNetwork` and
  `MomentumConvRustArrayMultiClassBackpropClassifierNetwork`: the conv networks with the momentum
  conv layer and `MomentumArrayLayer` / `MomentumRustArrayLayer` for the dense tail, and
  `hyperparameters = ("momentum",)`.
- **Tests,** parametrized over both backends as the conv network tests are:
  - parity with 3c's reference after every `learn` step and every `learn_batch` batch, over the
    conv tests' `ARCHITECTURES` (pooling, stride and multi-channel), at a non-power-of-two batch
    size, within the conv tests' `WEIGHT_ATOL` (1e-13);
  - at momentum 0.0, bit-identical to the plain conv networks on the same backend;
  - numpy against Rust after every batch. After 3a, any difference comes only from BLAS reduction
    order, so the tolerance can be the conv tests' own;
  - the layer against its formula on hand-set gradients (as `test_momentum_array_layer.py`), and
    the fused op on conv shapes;
  - save/load round trips, including a model saved by either backend loading into the other, with
    `momentum` in the envelope.

#### 3e: momentum in the study, and the reruns

- `_initial_conv_network` builds the momentum conv network for `momentum > 0`, and conv's
  `MOMENTA` becomes `[0.0, 0.9]`.
- **Stage 1 at momentum 0.9:** `baseline --architecture conv --epochs 2 --seeds 3`, with conv
  rates extended down to 0.03125 (dense's best at 0.9 was 0.25, far below its 4 at 0.0). About
  15 minutes.
- **Stage 2 at momentum 0.9:** `scaling --architecture conv --lr32 0.9=<rate>`, B = 32, 128 and
  512, warmup 0 and 1, 3 seeds, 3 epochs. About 20 minutes.
- **Gate:** if the scaled rate with warmup reaches the batch-32 band at B = 512, go to stage 4 at
  both momenta. If not, record the null in the study's findings and candidates.md, and stop.

Momentum and rate are confounded, as for dense: a pass at 0.9 shows that momentum makes B = 512
work, not why.

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
