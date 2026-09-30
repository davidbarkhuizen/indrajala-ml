# Workplan: batch normalization

**Status: done (2026-09-30), stages 0-6. Decisions D1-D8 settled by the owner (2026-09-30), and the remainder group (Pitfalls) with stage 6.**

This is step 2 of [primitives-roadmap.md](primitives-roadmap.md). It adds one new layer kind,
batch normalization (Ioffe & Szegedy 2015), for dense and conv networks, under every update rule,
in all three implementations. It builds on the composable layers of step 1 (done; its open
items are in [next-steps.md](next-steps.md)): a batch-norm network is a
Sequential network with a `BatchNorm` spec in its list. No existing network, preset or save file
changes. This is a feature, not refactoring: the golden run must stay bit-identical, because
nothing it covers changes.

## Why

- **The open question.** The conv batch-size study hit a ceiling on the stable rate: the linear
  scaling rule fails at B = 512 at momentum 0.0 and 0.9 (`batch_size_scaling.py`). Batch norm is
  the literature's standard way to raise it, and it is part of Goyal et al. 2017's setup
  (ResNet-50, statistics over 32 examples per worker, their § 2.3). Without it, the study can't be
  compared with the paper's full setup.
- **The parity rules.** It is the first layer whose forward pass depends on the rest of the batch,
  the first that behaves differently in training (batch statistics) and inference (running
  averages), and its backward pass is a known source of errors. It gets hand-computed tests and
  gradient checks in all three implementations.

## Where things are now

Counts are from `main` at 4fc6be9.

- **Specs.** `layer_specs.py` has `Dense(size, activation, dropout, output, loss)`, `Conv` and
  `Pool`. `validate_layer_specs` is the only gate, and it accepts only what all three
  implementations build. Conv and pool come before every dense layer.
- **Activations are fused into their layer.** A dense layer is sigmoid, ReLU or dropout-sigmoid,
  and a conv layer is always ReLU (`conv_forward_batch` applies it as it writes `A`). No layer is
  linear. Separate activation layers are out of scope ([next-steps.md](next-steps.md), From composable layers).
- **Hidden deltas.**
  - numpy: every hidden delta is already split. It calls `next_layer.downstream*()`, then applies
    its own derivative.
  - Rust: a dense hidden delta is fused. `layer_hidden_delta*`, `layer_relu_hidden_delta*` and
    `layer_dropout_hidden_delta*` read `next_layer.W` and `next_layer.delta` themselves. A conv or
    pool layer calls `next_layer.downstream*()`, then `array_relu_mask`.
  - Pure Python: a node reads the next layer's nodes (`downstream_sum`).
- **The optimizer** (`optimizers.py`, `python_optimizer.py`) steps each layer's `(W, b)` from
  `(grad_W, grad_b)`, with state keyed by layer index. The Rust ops it calls are shape-agnostic:
  each checks W against its gradient and state, and b likewise.
- **Training mode.** `_set_training_mode` switches every `TrainingModeLayer` (dropout today) for
  the forward pass of a `learn*` call. A layer records `_was_training` for its backward pass.
- **Batches.**
  - numpy and Rust have a single-example path (`learn`, `learn_row`: `forward`, then
    `step_single`) and a batch path (`learn_batch*`: `forward_batch` over the whole batch).
  - **Pure Python has no batch forward pass.** `_learn_batch` runs forward, backward and
    accumulate one example at a time, then applies once. A node holds one value, not one per
    example.
  - `train.py` keeps a final short batch (`_chunk_into_batches`), so a batch of one can occur
    whenever the training set size modulo the batch size is 1.
- **Inference** is `classify_state` (single-example `forward`) and `classify_rows` (`forward_batch`
  in chunks of `CLASSIFY_CHUNK_ROWS`), both outside training mode.
- **Snapshots and checkpoints.** `snapshot()` is `(W, b)` per layer, `()` for pool. `train.py`'s
  pocket restores a checkpoint, the snapshot plus the optimizer's state. Format 2 saves both.
- **Gradient checks.** No test compares a backward pass with finite differences today. The
  existing tests are hand-computed examples and parity between implementations.
- **Summation order.** numpy sums along a contiguous axis pairwise, and along a strided axis row
  by row. The crate sums as its own code says (`sum_axis0`, the conv `grad_b` sum over `N * P`).
  The two agree only where someone has checked.

## Decisions (settled)

The owner settled D1-D8 on 2026-09-30, each as recommended in the draft.

- **D1. Batch norm is a layer that carries the activation, after a linear layer.**
  `Dense(30, activation="linear"), BatchNorm(activation="sigmoid")`, and for conv
  `Conv(3, 8, activation="linear"), BatchNorm(activation="relu")`. This is the paper's form (§ 3.2:
  normalize `Wu + b`, before the nonlinearity), and Goyal et al.'s ResNet form (conv, BN, ReLU).
  The activation stays fused, now into the norm layer, and batch norm is one new layer kind. The
  linear layer's hidden delta is the norm layer's downstream, since the derivative is 1. So neither
  a split sigmoid delta nor a sigmoid-derivative crate op is needed, whatever the composable-layers
  workplan's After this plan expected. The norm layer's own delta is
  `downstream * activation'(A)`, which the existing fused ops `layer_hidden_delta*` and
  `layer_relu_hidden_delta*` compute from the next layer's `W` and delta, so they are reused as
  they are.
  - Rejected: a flag on the layer (`Dense(30, batch_norm=True)`), which needs a batch-norm variant
    of every dense and conv class and fused op; and a `BatchNorm()` layer after the activation,
    which isn't the paper's form and would need the split hidden delta.
- **D2. A linear layer has no bias,** as the paper says (§ 3.2: `b` "can be ignored", since the
  mean subtraction cancels it, and `β` takes its role). Its gradient would be zero only in exact
  arithmetic. In floats it is rounding noise, which Adam would normalize into steps of about `lr`.
- **D3. Pure Python gets a layer-major batch path.** A pure-Python network that contains a
  `BatchNorm` layer trains a batch layer by layer: each layer holds one value per node per
  example, forward for the whole batch, then backward for the whole batch. Networks without batch
  norm keep today's example-major loop, so their bits don't change. Every accepted spec still
  builds in all three implementations, and the roadmap's gradient checks run in all three.
- **D4. A network with batch norm refuses a one-example training step.** A batch of one normalizes
  every value to 0, so the layer would output `β` and pass no gradient down. `learn`, `learn_row`,
  and `learn_batch*` with one example each raise an error that names the batch-norm layer.
  `train.py` drops a final batch of one for such a network only, and says so in its docstring.
  Other networks keep the short batch, so their runs don't change.
- **D5. Moving averages, with PyTorch's constants.**
  - Training normalizes with the batch mean and the biased variance (paper, Algorithm 1).
  - Inference uses running averages, updated in each training forward pass as
    `running = (1 - rate) * running + rate * batch`. The running variance takes the unbiased batch
    variance, `B / (B - 1)` times the biased one (Algorithm 2's correction). The moving average is
    the frameworks' convention, and it needs no extra pass, unlike Algorithm 2's average over the
    training set, which every evaluation (each epoch's accuracy pass, the pocket) would need first.
    The README cites the choice.
  - Spec fields: `BatchNorm(activation, epsilon=1e-5, running_rate=0.1)`, PyTorch's defaults. The
    name avoids `momentum`, which is already an update rule.
  - Initialization: `γ = 1`, `β = 0`, running mean 0, running variance 1. Nothing is drawn, so the
    draw order of every other layer is unchanged, as with pool layers.
- **D6. Ghost batches are in this plan, as stage 6.** Goyal et al. compute statistics over 32
  examples per worker, whatever the total batch (their § 2.3), as the ghost batch norm of Hoffer
  et al. 2017 does. `BatchNorm(..., group_size=None)` splits a batch into groups of `group_size`
  rows, each normalized with its own statistics. The study needs it to compare with the paper.
- **D7. Weight decay applies to neither `γ` nor `β`.** Under `WeightDecay` both step with plain SGD.
  Goyal et al. § 5.1 state that they don't apply weight decay to the BN coefficients (stage 0
  quotes the text into the README). This also matches the rule's treatment of biases (eq. (8)).
  The linear layer's `W` is decayed as usual.
- **D8. No presets.** Batch norm is reached through the Sequential networks only. A preset is added
  when a demo or the study needs one, as its own PR.

## The design

```python
network = SequentialArrayNetwork(
    input_shape=(28, 28, 1),
    layers=[
        Conv(3, 8, activation="linear"),
        BatchNorm(activation="relu"),
        Dense(32, activation="linear"),
        BatchNorm(activation="sigmoid"),
        Dense(10, output=True),
    ],
    update_rule=Momentum(0.9),
    shape="multiclass",
    backend=RUST,
)
```

- **Specs.**
  - `Dense.activation` and `Conv` gain `"linear"` (`Conv` gains an `activation` field, default
    `"relu"`).
  - `BatchNorm(activation, epsilon, running_rate, group_size)`.
  - `validate_layer_specs` accepts a linear layer only when a `BatchNorm` directly follows it, and a
    `BatchNorm` only directly after a linear layer. A `BatchNorm` after a linear conv takes
    `"relu"`, and after a linear dense layer `"sigmoid"` or `"relu"`. It refuses dropout on either
    layer of the pair, and batch norm on the output layer.
  - The existing lists all stay valid, with the same meaning.
- **Layer shapes.** A dense norm layer normalizes each of its `size` features over the batch. A
  conv norm layer normalizes each channel over the batch and every position (paper § 3.2: the
  effective batch is `B * p * q`). The flat layout is channel-major, as today, so a channel is a
  contiguous run of `P` values in each row.
- **Parameters and state.**
  - `γ` and `β` are trained parameters, one per feature or channel. The optimizer steps them from
    `grad_gamma` and `grad_beta`, with state keyed by layer index as for `(W, b)`.
  - `running_mean` and `running_var` are layer state, not optimizer state. `snapshot()` carries
    them with the weights, so the pocket, `restore`, the ensembles' worker boundary and format 2
    all keep them.
  - A norm layer's snapshot entry is `(gamma, beta, running_mean, running_var)`. A linear layer's
    entry is `(W,)` (D2).
- **The optimizer.** A weighted layer is stepped through one accessor for its parameter pair,
  `(W, b)` or `(γ, β)`, so the rules' formulas aren't duplicated. A linear layer without a bias
  (D2) steps `W` only. On Rust, stage 1 checks that the fused apply ops take 1D `γ` in W's place
  (they shape-check W only against its own gradient and state). If they don't, stage 3's crate PR
  covers it. The fused single-example SGD step never applies, because a network with batch norm
  has no single-example step (D4).
- **Forward.** In training, a norm layer computes the batch statistics, normalizes, applies `γ`,
  `β` and the activation, updates the running averages, and caches `x̂` and `1 / sqrt(var + ε)`
  for the backward pass. In inference it uses the running averages. It is a
  `TrainingModeLayer`, so `_set_training_mode` switches it with the dropout layers.
- **Backward.**
  - The norm layer's delta is `dL/dy`, from the next layer through the existing fused ops (D1).
  - `accumulate_gradient_batch` adds `Σ δ·x̂` to `grad_gamma` and `Σ δ` to `grad_beta`.
  - `downstream_batch()` returns `dL/dz`, the gradient into the linear layer, by the paper's
    § 3 chain rule. The linear layer's `compute_hidden_delta_batch` is that downstream, unchanged.
- **Pure Python (D3).** `BatchNormLayer` holds per-node lists of `z`, `x̂` and the statistics over
  the batch. The layer-major path runs the same formulas as the array layers, per scalar, in the
  paper's order. The example-major loop stays for every network without batch norm.
- **Format 2.** A `batch_norm` layer entry records its spec fields. The weights list holds
  `[gamma, beta, running_mean, running_var]` for it, and `[W]` for a bias-free linear layer. The
  optimizer state is `[v_gamma, v_beta]` (momentum) or the four Adam moments. Older code refuses
  these files, as format 2 already accepts.

## Pitfalls to design around

- **One published form, in one order, in all three implementations.**
  - The forward pass follows Algorithm 1: `μ = Σx / m`, `σ² = Σ(x - μ)² / m` (two passes, not
    `E[x²] - μ²`), `x̂ = (x - μ) / sqrt(σ² + ε)`, `y = γ·x̂ + β`.
  - The backward pass follows the paper's § 3 chain rule, term by term, not a compact rearranged
    form: they round differently.
  - Write the exact expressions into the README's layer section and hold every implementation to
    them, as the Update rules table does.
- **numpy's summation order.** numpy sums a contiguous axis pairwise, and the conv statistics sum
  over each channel's contiguous positions. The numpy layers must sum in the order the crate uses.
  Before any numpy code lands, stage 0 pins that order with a test against an explicit sequential
  loop, at the shapes the tests use. Where numpy's own order differs, the numpy layer sums
  explicitly in the crate's order. A numpy-Rust difference is explained, never accepted within a
  tolerance.
- **The running averages must not move in inference.** `classify_rows` runs `forward_batch`
  outside training mode. A norm layer updates its averages only when training is on, and a test
  checks that classifying leaves them bit-identical.
- **Training mode reaches the backward pass.** `_set_training_mode(False)` runs before the
  backward pass, so the layer records `_was_training`, as dropout does.
- **Ghost groups (D6)** must divide the batch or handle a remainder group. Settled by the owner
  (2026-09-30): the last group is the remainder, refused if it has one row (D4's reason).
- **The golden run and the timing.** No existing network changes, so the golden run stays
  bit-identical at every stage. The optimizer's parameter accessor changes `apply` for every
  network, so the stage that adds it is timed within noise
  ([measurement.md](measurement.md)). The new layers have no baseline
  to be timed against, so their timings are recorded, not compared.
- **Registry walks.** `test_rust_array_layer_forward_batch.py` and
  `test_rust_array_layer_sgd_step.py` walk `RustArrayLayer` subclasses, and `test_prepared_dataset.py`
  and `test_seeded_init_parity.py` walk the networks. A new layer class goes into their expected
  sets, or is shown not to belong, in the PR that adds it.
- **Gradient checks** need a loss that is a function of the whole batch. The check perturbs one
  weight, reruns the batch forward pass in training mode from the same state, and compares with
  the accumulated gradient. The running averages must be restored between perturbations, or the
  check reads a moving target.

## Stages

Each stage is one PR in indrajala-ml, except where a crate PR comes first (then a "Bump rust/"
PR). Every stage passes `./cli test`, `./cli lint`, the golden run bit-identical, and the legacy
and format-2 fixtures. Until a stage builds batch norm in an implementation, that implementation's
builder refuses a `BatchNorm` spec with an error naming the stage, as `sequential_save_not_yet`
did.

### Stage 0: the gates

1. `tests/gradient_check.py`: a finite-difference check over a whole batch's loss, for any
   Sequential network in any implementation. It is run against today's layers first (dense
   sigmoid, ReLU, softmax and cross-entropy, conv, pool) to show that the check itself works.
2. The summation-order test (Pitfalls): numpy's sums at the batch-norm shapes against a sequential
   loop and against the crate's `sum_axis`.
3. Quote Goyal et al. § 5.1 (D7) and the paper's Algorithms 1 and 2 (D5) into the README's
   new batch-norm section, with the exact expressions.

### Stage 1: dense batch norm in numpy

1. Specs and validation (D1, D2). The optimizer's parameter accessor, timed within noise.
2. `LinearArrayLayer` and `BatchNormArrayLayer`, dense only.
3. The snapshot, restore and checkpoint entries, and the pocket through `train.py`. D4's refusal
   and `train.py`'s final batch of one.
4. Tests: hand-computed forward and backward for a 2-feature, 3-example batch; gradient checks
   under every rule; inference with running averages; classifying doesn't move the averages.

### Stage 2: dense batch norm in pure Python (D3)

1. The layer-major batch path, used only by networks with a `BatchNorm` layer.
2. `LinearLayer` and `BatchNormLayer`, with scalar formulas.
3. Tests: the same hand-computed cases, and gradient checks. Parity with numpy: the gap is
   measured and explained, as for the existing pure-Python parity tests.

### Stage 3: dense batch norm in Rust

1. Crate PR (indrajala-math-rust):
   - a linear forward op, the crate's `linear_preactivation(_batch)` without the bias (D2);
   - a batch-norm forward op fused with its activation, in training and inference forms, single
     example (inference only) and batch;
   - a batch-norm downstream op;
   - a `γ`/`β` accumulate op.
   Each op mirrors the numpy formula. Crate tests check them against their reference formulas.
2. Bump `rust/`, then add `LinearRustArrayLayer` and `BatchNormRustArrayLayer`.
3. Tests: bit-identity with numpy for every op and for whole training runs under every rule.
   - Done: every batch-norm op is numpy's by bits given the same inputs, except for the sigmoid's
     `exp`, whose last bit numpy's `np.exp` computes differently on some CPUs (a CI runner showed
     it, #497). The tests give the numpy layer the crate's `exp`. Whole runs are not, and
     can't be: the linear layer's `X @ W.T` is BLAS in numpy and the crate's FMA chains in Rust,
     as for every dense layer, and Adam's `1 - beta**t` rounds differently (`powi`). Measured
     after 50 steps: at most 4.4e-12 relative, and the same networks without batch norm 1.2e-12.
   - The optimizer steps a pair of parameters per fused call: a linear layer's W takes an empty
     array in b's place, so no crate op was needed for it.

### Stage 4: conv batch norm (per channel) in all three implementations

1. The conv linear layer (`Conv(..., activation="linear")`) and per-channel statistics, in numpy,
   pure Python and Rust (a crate PR and a bump first).
2. The same tests as stages 1-3: hand-computed, gradient checks, and numpy-Rust bit-identity.

Split it by implementation if the diff passes about 1,500 lines.

Split as 4a numpy, 4b pure Python, 4c Rust (a crate PR, then the bump), each one PR, as stages 1-3.

- 4a, done: `ConvSpec.activation` (`"relu"`, the default, or `"linear"`); `LinearConvArrayLayer`,
  the conv layer's products without the bias and the ReLU; and `BatchNormArrayLayer(...,
  positions)`, which computes the dense layer's expressions on the `(N * P, C)` view of the conv
  layer's channel-major `(N, C * P)`, so its rows are the README's order and `m = N * P`. Format 2
  leaves `activation` out of a ReLU conv entry, so saved files don't change.
- 4b, done: `LinearConvLayer` (`LinearConvKernel` without a bias, `LinearConvUnit` without the
  bias and the ReLU), and `BatchNormLayer(..., positions)`: a `BatchNormNode` per channel holds its
  parameters and its lists in the README's order, and a `BatchNormPosition` per value is the layer's
  node. The layer-major path needed no change. Given the same inputs the layer is numpy's by bits;
  whole networks agree within the pure-Python parity tolerance (at most 1.1e-12 relative, 20 steps).
- 4c, done: the crate's `batch_norm_*` ops take `positions` and index the channel-major layout in
  place, in the `(N * P, C)` view's row order, so they are numpy's by bits given the same inputs;
  `conv_linear_forward_batch` and `conv_linear_accumulate_gradient_batch` are the conv ops without
  the bias (and the ReLU). `LinearConvRustArrayLayer` shares its geometry, W and downstream with
  `ConvRustArrayLayer`. A ReLU batch-norm layer's hidden delta masks the next layer's downstream,
  which may be a pool or conv layer's. Whole networks after 50 steps: at most 3.5e-12 relative
  from numpy, and the same conv networks without batch norm 3.4e-13; subtracting the mean turns
  the products' rounding difference into a larger relative one, as for dense batch norm.

### Stage 5: format 2, load_network and the docs

1. Format 2's `batch_norm` and bias-free linear entries. Resume tests: train N steps, save, load,
   train M more, equal to N + M steps by bits, per rule and implementation. The running averages
   resume too.
2. Format-2 fixtures for a batch-norm network per implementation.
3. README: the batch-norm section, the preset table's note, and the composable-layers list in
   [next-steps.md](next-steps.md).

Done: `batch_norm` entries record the spec's fields, and a linear conv entry its `"activation":
"linear"` (a ReLU conv entry is unchanged). Weights are as above; in pure Python a channel is
`[[gamma], beta, running_mean, running_var]` and a linear node or kernel `[weights]`. The array
optimizer state is keyed by parameter (`m_gamma`, `v_beta`, `velocity_W` alone for a linear layer),
and pure Python's has no bias entries for a linear node. `tests/test_format2.py` resumes dense and
conv batch-norm networks by bits under every rule in all three implementations, through the class
and `load_network`, and loads numpy and Rust files into each other. The fixtures are
`BatchNorm<class>` in `tests/fixtures/saved_models/`, a conv and a dense pair under Adam.

### Stage 6: ghost batches (D6)

1. `group_size`, in all three implementations: statistics per group of rows, and the running
   averages updated once per group, in row order.
2. Tests: one group equals plain batch norm by bits; hand-computed groups of 2 in a batch of 4;
   the remainder group.
3. README: ghost groups in the batch-norm section. Mark roadmap step 2 done.

Done: `BatchNorm(group_size=None)`, 2 or more (`validate_layer_specs`). `layer_specs.ghost_groups`
splits a batch into `(first, end)` example ranges and refuses a last group of one. The numpy and
pure-Python layers normalize each group as a batch of their own, and the crate's
`batch_norm_forward_batch` and `batch_norm_downstream_batch` take `group_size` (crate #45), their
`var` and `std` per group per channel. The gradients of `γ` and `β` stay one fold over the batch.
A network refuses a batch that leaves a group of one before its forward pass, naming the layer,
and `train_backprop_network_mini_batch` refuses such a batch size before training. Format 2 writes
`group_size` only when set, so earlier files and fixtures are unchanged.
`tests/test_batch_norm_ghost_groups.py` checks the three workplan cases, each group against a plain
layer run on it alone, the pure-Python and Rust layers against numpy's by bits, gradient checks in
every implementation, and the refusals; `tests/test_format2.py` resumes a ghost-group network by
bits.

## After this plan

- **The study rerun.** Rerun the conv batch-size cells that failed (B = 512, momentum 0.0 and
  0.9) with batch norm, and with ghost groups of 32. That is an experiment with its own plan.
- **Batch norm under dropout or after pool**, if a use appears.
- **Folding batch norm into the preceding weights** for inference: a speed change, measured.

## Out of scope

- Layer norm, group norm and instance norm. They normalize within an example, a different layer.
- Synchronized statistics across workers or processes.
- Separate activation layers (still out of scope, [next-steps.md](next-steps.md)).
- Any change to an existing network's numerics, or to the crate's existing ops.
