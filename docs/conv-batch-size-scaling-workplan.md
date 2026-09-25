# Workplan: batch-size scaling for the conv network

**Status: stages 1 and 2 done. At momentum 0.0 the rule holds to B = 128 with warmup and fails
at B = 512, where no rate reaches the band. Stage 3 (momentum conv, planned in detail below) is
under way: 3a and 3b (the update rules in the literature's form), 3c (the conv networks honor
hyperparameters) and 3d (the pure-Python momentum conv reference) are done, 3e (numpy and Rust
momentum conv) is next.**

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
and only the pure-Python implementation (3d) combines them so far.

Each sub-stage is one PR; 3a and 3b are each a crate PR first, then the parent PR that moves `rust/`.

#### The update rules follow the literature

3a and 3b come first because momentum conv must be built on update rules that match the published
forms, in all three implementations. Otherwise results aren't comparable with the literature, or
between our own backends. The reference is Goyal et al. 2017, the paper this study tests (section
2 and section 3):

| rule | the paper's form | pure Python and numpy | Rust (`fused.rs`) |
| --- | --- | --- | --- |
| SGD, eq. (2) | `w - lr * (g / B)` | as the paper (3a) | as the paper (3a) |
| weight decay, eq. (8), "λw added to the aggregated gradients" | `w - lr * (g / B + λ * w)` | as the paper (3a) | as the paper (3a) |
| momentum, eq. (9), the "reference implementation" | `u = m * u + g / B; w - lr * u` | as the paper (3b) | as the paper (3b) |

`g` is the gradient summed over the batch, so `g / B` is the paper's mean gradient. Adam already
follows Kingma & Ba's Algorithm 1 (`g / B` first) in all three, and `layer_sgd_step` is B = 1,
where the groupings agree.

- **The grouping differences are about an ULP each,** and vanish when B is a power of two
  (dividing by it is exact), except for subnormal gradients. But the last, partial batch of an epoch isn't one (60000 / 128 and
  60000 / 512 both leave 96 rows), nor are the tests' batch sizes such as 6. Conv training's
  chaotic sensitivity turns an ULP into different end-of-run results.
- **The momentum difference is a different rule, not a rounding.** Eq. (10) folds the rate into
  the velocity. The paper: for a fixed rate the two are equivalent, but when the rate changes (as in
  warmup) eq. (10) needs a "momentum correction" of `lr_{t+1} / lr_t`, which our code didn't
  apply. So the dense study's momentum 0.9 cells with warmup (including the finding that B = 512
  holds) were not the paper's algorithm, and are pending a rerun on eq. (9).

The README's "Update rules" section records the forms and their sources, as the rule for any
future optimizer.

#### 3a: SGD and weight decay in the paper's grouping (done)

Done (crate #30, parent PR below). `tests/test_update_rule_forms.py` checks all nine SGD and weight
decay implementations against the paper's formula bit for bit, and each fails on its old form.
The golden run's prediction held: only the Rust L2 network changed. The step loop's timing is
within noise: over two before/after rounds, Rust's B = 32 epochs moved +1.7% (dense) and -2%
(conv), while the same unchanged build varied up to 10% between rounds.


- **Change:** `w - lr * (g / B)` in `BackpropNode`, `ConvKernel`, `ArrayLayer`, `ConvArrayLayer`
  and `layer_apply_accumulated_gradient`. The Rust L2 op takes numpy's `lr * (g / B + λ * w)`.
  Crate PR first, then the parent PR that moves `rust/`.
- **Tests:** the fused-op tests compare Rust with the numpy layers bit for bit (not within a
  tolerance), at batch sizes 1, 6 and 96 as well as powers of two. Mutation check: each exact
  test must fail on today's ops.
- **Golden run:** a numerics change, not a refactoring. Its batches are 4 rows, so the prediction
  is bit-identical checkpoints for every network except the Rust L2 ones. Check that, then
  re-record.
- **Timing:** the apply ops gain a division per element. Time the step loop before and after
  (`scripts/prepared_dataset_timing.py time`, both builds committed first, separate processes).
  Expect noise, since the apply is one pass over the parameters per batch.
- **Out of scope:** the reduction order of matmuls and gradient accumulation. OpenBLAS's blocking
  isn't reproducible, and the 1-ULP control covers it.

#### 3b: momentum as the paper's eq. (9) (done)

Done (crate #31, parent PR below). `tests/test_update_rule_forms.py` checks all three momentum
implementations against eq. (9) bit for bit, over three steps with a changing rate, at momenta 0.0
and 0.9. On the old form, every case at 0.9 fails, and at 0.0 the cases at batch sizes 6 and 96 fail
(there eq. (10) at momentum 0 is SGD in the old grouping). A hand-computed test doubles the rate on
step 2, where the two forms give 0.404 and 0.422. The golden run's prediction held: only the numpy
and Rust momentum networks changed. The fused op costs 2-6% more per call (a division and a
multiply per element), once per layer per batch. The dense study's momentum findings with warmup
are marked pending a rerun, in place.

- **Change:** every momentum layer keeps a velocity `u` (was the previous delta), zero-initialized:
  `u = m * u + g / B; w = w - lr * u`, the same for `b`. That covers `make_momentum_node_cls`,
  `MomentumArrayLayer`, `MomentumRustArrayLayer` and `layer_momentum_apply_accumulated_gradient`,
  whose arguments are renamed (`prev_delta_*` to `velocity_*`). Docstrings cite eq. (9) and say
  it replaces Rumelhart et al.'s form (eq. (10)).
- **Tests:** the hand-computed momentum tests are redone for eq. (9), with a changing rate in at
  least one (where eq. (9) and eq. (10) differ). The fused op is compared with numpy bit for bit.
  The golden run changes for every momentum network. Check that nothing else changes, then
  re-record.
- **Results to retest (later, a separate task):** the dense study's momentum 0.9 findings in
  `batch_size_scaling.py`'s docstring (the batch-32 rate, B = 512 holding with a one-epoch
  warmup) and `demo_batch_size_scaling`. At a constant rate the two forms are mathematically
  equivalent, so the study's no-warmup momentum cells change only by rounding. Mark the findings
  as pending a rerun, in place, until then.

#### 3c: the conv networks honor hyperparameters (done)

Done (parent PR below, no crate change). The golden run is bit-identical and a plain conv
network's save file has the same keys. New tests build an array sibling whose conv, dense and
output layer classes take a hyperparameter, and a pure-Python network with every hook overridden.
Each fails on the old wiring.

- **Array networks:** `build_conv_array_network_layers` builds the conv, dense and output layers
  through a layer factory, `network._new_layer`, which now takes any layer class and its
  arguments (`HyperparameterLayerClass` in `array_protocols.py`). The conv layer classes have
  `hyperparameters = ()`. The output layer is `output_layer_cls`'s. It was `hidden_layer_cls`'s,
  and for the plain conv networks the two are the same class.
- **Save envelope:** `save_conv_model_json` takes `extra` as `save_array_model_json` does, and
  `load_conv_model_json` takes `extra_init_kwargs`. `ArrayConvShape` passes `_extra_state()` and
  `_extra_init_kwargs`.
- **Pure Python:** `ConvMultiClassBackpropClassifierNetwork` builds its layers from
  `conv_layer_cls` and the inherited `hidden_layer_cls` / `output_layer_cls` (not a new
  `dense_layer_cls`), and `ConvLayer` its kernels from `_kernel_cls`, as `BackpropLayer` does
  with `_node_cls`.

#### 3d: the pure-Python momentum conv reference (done)

Done (parent PR below, no crate change). `make_momentum_kernel_cls` and
`make_momentum_conv_layer_cls` are in `momentum_conv_layer.py`. `MomentumConvKernel` joins the
bit-for-bit eq. (9) check in `tests/test_update_rule_forms.py` on the conv shape. The golden run is
bit-identical, since no existing class changed. The new tests catch each of these mutations: the
velocity not carried (weights or bias), the conv or output hook left unset, `momentum` not saved or
not loaded, and the rate folded into the velocity.

- `MomentumConvKernel`: `ConvKernel` with a velocity, zero-initialized, and 3b's eq. (9) update:
  `u = m * u + accum / B; w = w - lr * u`, positions summed and examples averaged as `ConvKernel`
  does. A factory, as `make_momentum_node_cls`, because momentum has no default.
- `MomentumConvMultiClassBackpropClassifierNetwork(..., momentum)`: sets 3c's hooks in `__init__`,
  as `MomentumBackpropClassifierNetwork` does: `conv_layer_cls` a `ConvLayer` whose `_kernel_cls`
  is the momentum kernel, and `make_momentum_layer_cls(momentum)` for `hidden_layer_cls` and
  `output_layer_cls`. Pool layers are unchanged: they have no weights. `save` passes
  `extra={"momentum": ...}` and `load` its `extra_init_kwargs`.
- **Tests:** at momentum 0.0, bit-identical to `ConvMultiClassBackpropClassifierNetwork` through
  `learn` and `learn_batch` (at `m` = 0, eq. (9) is exactly 3a's `w - lr * (g / B)`). A hand-computed two-step kernel update. A
  save/load round trip that keeps `momentum`. No new gradient check: momentum changes only the
  update, and the gradients are covered by the existing conv checks.
- The velocity is not saved, as for every momentum network (`snapshot()` covers W and b).

#### 3e: numpy and Rust momentum conv

- **Layers:** `MomentumConvArrayLayer(ConvArrayLayer)` and
  `MomentumConvRustArrayLayer(ConvRustArrayLayer)`, with `hyperparameters = ("momentum",)` and a
  velocity shaped as `W` `(channel_count, fan_in)` and `b` `(channel_count,)`. numpy
  uses `MomentumArrayLayer`'s expression. Rust calls
  `pa.layer_momentum_apply_accumulated_gradient`, which only checks that shapes match, so it takes
  the conv shapes with no crate change.
- **Networks:** `MomentumConvVectorizedMultiClassBackpropClassifierNetwork` and
  `MomentumConvRustArrayMultiClassBackpropClassifierNetwork`: the conv networks with the momentum
  conv layer and `MomentumArrayLayer` / `MomentumRustArrayLayer` for the dense tail, and
  `hyperparameters = ("momentum",)`.
- **Tests,** parametrized over both backends as the conv network tests are:
  - parity with 3d's reference after every `learn` step and every `learn_batch` batch, over the
    conv tests' `ARCHITECTURES` (pooling, stride and multi-channel), at a non-power-of-two batch
    size, within the conv tests' `WEIGHT_ATOL` (1e-13);
  - at momentum 0.0, bit-identical to the plain conv networks on the same backend;
  - numpy against Rust after every batch. After 3a and 3b, any difference comes only from BLAS
    reduction order, so the tolerance can be the conv tests' own;
  - the layer against its formula on hand-set gradients (as `test_momentum_array_layer.py`), and
    the fused op on conv shapes;
  - save/load round trips, including a model saved by either backend loading into the other, with
    `momentum` in the envelope.

#### 3f: momentum in the study, and the reruns

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
work, not why. With 3b, momentum under warmup is the paper's eq. (9), so no momentum correction
is needed.

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
