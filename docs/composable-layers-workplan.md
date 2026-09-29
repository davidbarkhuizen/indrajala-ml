# Workplan: composable layers and optimizers

**Status: decisions D1-D5 settled by the owner (2026-09-29). Stages 0-2 done; stage 3 next.**

This is step 1 of [primitives-roadmap.md](primitives-roadmap.md). Stages 1-4 are structural
refactoring: they change structure only, never numerics, under the README's Refactoring rules. At
the end, a network is built from a list of layer specs and one optimizer, in all three
implementations. Batch normalization (step 2) is then one new layer kind, not one new class per
combination. Stage 5 adds one feature on top, the format-2 save file with optimizer state, so a
loaded network resumes training.

## Why

Two things are fixed per class today, and each combination of them is a class:

- **The update rule.** SGD, momentum, Adam and L2 weight decay are layer subclasses that override
  `apply_accumulated_gradient`: `MomentumArrayLayer`, `AdamRustArrayLayer`, `L2ArrayLayer`, and so
  on, for each backend. Conv repeats it: `MomentumConvArrayLayer` and `MomentumConvRustArrayLayer`
  hold the same velocity code as the dense momentum layers, over a different base class. Adam and
  L2 for conv don't exist, only because nobody has written the four more classes.
- **The architecture.** A network class fixes `hidden_layer_cls` and `output_layer_cls`, and every
  hidden layer has the same class. The conv networks build their layers on a separate path
  (`ArrayConvShape.__init__`, `build_conv_array_network_layers`) and override randomize,
  snapshot, restore, save and load, because the base assumes every layer is dense.

So momentum with conv took three PRs (#448, #451, #452) and six new classes. Batch norm crossed
with {dense, conv} × {SGD, momentum, Adam, L2} × three implementations would take dozens more, and
it can't be expressed at all without layers of different kinds in one network.

## Where things are now

Counts are from `main` at d68fc5c.

- **39 backprop network classes** in `indrajala_ml/model/` (40 `*ClassifierNetwork`s less the
  perceptron), over three bases: `BackpropNetworkBase` (pure Python), and `ArrayNetworkBase` with
  `NumpyArrayNetworkBase` and `RustArrayNetworkBase`. The shape mixins in
  `array_network_shapes.py` (multiclass, single-output, conv) hold targets, classification and
  save/load.
- **Array sibling networks** are two-line classes: they set `hidden_layer_cls`,
  `output_layer_cls` and `hyperparameters`, and store each hyperparameter as an attribute before
  `super().__init__()`. `_new_layer` passes the hyperparameters to the layer's constructor by
  name, and `_extra_state` writes them as flat keys in the save file (`{"momentum": 0.9}`).
- **Array update rules** live in `apply_accumulated_gradient` of 8 layer classes (momentum, Adam
  and L2 for each backend, and momentum conv for each backend). numpy's momentum is a shared
  function (`momentum_update`). The Rust layers call the crate's fused ops
  `layer_{,momentum_,adam_,l2_}apply_accumulated_gradient`. **Every one of these ops is
  shape-agnostic**: it shape-checks W against its gradient and state, and b likewise. So an
  optimizer can drive conv weights with no crate change. Momentum conv already does this.
- **Optimizer state** lives in the layers (numpy and Rust) or in each node and kernel (pure
  Python): velocities, Adam's m and v, and a step count `t` per layer or per node. The counts all
  step once per `learn*` call, so today they always equal one global count.
- **The fused single-example step.** `RustArrayLayer.sgd_step` fuses accumulate and apply into
  `layer_sgd_step`, which is valid only for plain SGD. Each non-SGD Rust layer overrides it back
  to `unfused_sgd_step`. `tests/test_rust_array_layer_sgd_step.py` checks that every subclass
  that overrides the update also overrides `sgd_step`.
- **Activations are layer kinds, and fused.** ReLU, softmax, cross-entropy and dropout are layer
  subclasses. On Rust, each forward is one call (matmul, bias, activation), and each hidden delta
  is one call that reads the next layer's `W` and `delta`. Dropout's op is fused with the sigmoid,
  so no op exists for dropout after a ReLU. In `fused.rs`, `layer_hidden_delta` is
  `layer_downstream`, then the elementwise derivative in the same order. So a split path (the
  next layer's `downstream()`, then the derivative) would give the same bits. It would cost one
  more boundary crossing, and for sigmoid it would need a new crate op (only `array_relu_mask`
  exists).
- **Pure Python** builds its update rules as node-class factories (`make_momentum_node_cls`,
  `make_adam_node_cls`, `make_l2_node_cls`), plus `make_momentum_kernel_cls` for conv, and dropout
  as `make_dropout_node_cls`. A network sets its layer classes as instance attributes in
  `__init__`. The per-node form follows the literature's per-weight formulas and is the parity
  reference for the array networks.
- **Save files** hold the constructor's shape arguments, the snapshot and the flat
  hyperparameters. They don't record the class: `cls.load(path)` needs the right class. There are
  five envelopes: pure-Python (`save_model_json`, with `input_bounds`), array multiclass, array
  single-output, conv (shared by all three implementations), and the two ensemble envelopes.
  Optimizer state isn't saved by any of them.
- **`snapshot()` / `restore()`** carry weights only. `train.py` uses them to restore the
  best-epoch weights at the end of training. `ensemble_train.py` uses them to move trained
  sub-networks across the worker boundary as plain lists. The scripts and demos use them to start
  numpy and Rust from the same weights.
- **Registry walks in the tests**: `test_seeded_init_parity.py` asserts it covers every subclass
  of `ArrayNetworkBase`. `test_prepared_dataset.py` walks them too, and
  `test_rust_array_layer_forward_batch.py` and `test_rust_array_layer_sgd_step.py` walk every
  subclass of `RustArrayLayer`. `tests/array_network_contract.py` generates the per-network parity
  tests from an `ArrayNetworkSpec` with positional hyperparameters.
- **The golden run** (`scripts/golden_training_run.py`) covers the 16 dense multiclass array
  networks, the numpy and Rust conv networks (`ConvSpec(3, 2), PoolSpec(2)`), 4 single-output
  networks and 2 ensembles. It **doesn't cover momentum conv**, which the batch-size study trains,
  or any pure-Python network.

## Decisions (settled)

- **D1. Public network names stay, as presets, and the update-only layer classes go.** All 39
  network classes keep their names and constructor signatures. Each becomes a preset that builds
  a spec and an optimizer. Demos, `demos/registry.py`, `ensemble_train.py`, the scripts and the
  tests construct them by name. The 8 array layer subclasses and the 4 pure-Python factories
  (`make_momentum_node_cls`, `make_adam_node_cls`, `make_l2_node_cls`,
  `make_momentum_kernel_cls`) exist only to carry an update rule. Only tests import them, and
  stage 6 deletes them.
- **D2. An update rule is a frozen dataclass**: `SGD()`, `Momentum(momentum)`,
  `Adam(beta1, beta2, epsilon)` with Kingma & Ba's defaults, and `WeightDecay(l2_lambda)`. They are
  typed under strict pyright, hashable and picklable, and save as
  `{"rule": "momentum", "momentum": 0.9}`. `WeightDecay` is its own rule, SGD with Goyal et al.'s
  eq. (8). Weight decay combined with momentum or Adam is a different published rule each time
  (Goyal eq. (10)'s form, AdamW against Adam with L2), so it isn't a flag on the other rules (see
  After this plan).
- **D3. Pure Python takes the same specs and rules** (stages 2 and 4). Every spec the array
  networks accept then has a pure-Python parity reference, which batch norm needs too. The
  per-weight formulas stay per weight. Only where they live and how they are dispatched changes.
- **D4. Format 2 everywhere, with optimizer state.** Every network, the presets included, saves a
  format-2 envelope: the layer specs, the update rule, the weights and the optimizer's state. A
  loaded network resumes training where it stopped. Loading reads format 2 and every legacy
  envelope, with the legacy files pinned by committed fixtures. Legacy files load with fresh
  optimizer state, as they do today. Files saved after stage 5 don't load on older code, which is
  accepted.
- **D5. One optimizer per network, holding all the state.** The network owns
  `self.optimizer = Optimizer(rule, backend)`. The optimizer keeps each parameter's state in a
  table keyed by the parameter's position (layer index, then `"W"` or `"b"`, or the node index in
  pure Python), never by `id()`. It keeps one step count `t` for the network. The layers keep
  their parameters and gradient accumulators, and lose every update formula.

## The design

```python
# a spec is backend-free data; the builder maps it to a backend's layer classes
network = SequentialArrayNetwork(
    input_shape=(28, 28, 1),
    layers=[Conv(kernel_size=5, channel_count=8), Pool(2), Dense(30), Dense(10, output=True)],
    update_rule=Momentum(0.9),
    shape="multiclass",
    backend=RUST,
)


# a preset is the same network under its existing name and signature
class MomentumConvRustArrayMultiClassBackpropClassifierNetwork(...):
    def __init__(self, input_height, input_width, conv_specs, dense_layer_sizes, class_count, momentum): ...
```

- **Layer specs** (`indrajala_ml/model/layer_specs.py`): `Dense(size, activation="sigmoid" |
  "relu", dropout=None)`, `Conv(kernel_size, channel_count, stride)` (today's `ConvSpec`),
  `Pool(pool_size, stride)` (`PoolSpec`), and the output layer
  `Dense(size, output=True, activation="sigmoid" | "softmax", loss="squared" | "cross_entropy")`.
  Activations stay fused into the layer, not separate layers. That keeps every Rust op as it is,
  with no extra crossings.
- **The builder** maps a spec to each implementation's existing layer class (`ReLURustArrayLayer`,
  `DropoutArrayLayer`, `ConvLayer`, ...). It rejects a spec that some implementation can't build:
  dropout after a ReLU, a softmax hidden layer, a dense layer followed by a conv or pool layer
  (the fused dense hidden delta reads the next layer's `W`). Only combinations built in all three
  implementations are accepted, so every accepted spec is parity-testable.
- **Update rules** (`indrajala_ml/model/update_rules.py`): D2's dataclasses. They hold only
  hyperparameters.
- **The optimizer** (`indrajala_ml/model/optimizers.py`), one per network and implementation:
  - `begin_step()` increments `t` once. `apply(index, layer, learning_rate, batch_size)` applies
    the rule to one weighted layer's `(W, b, grad_W, grad_b)` and resets its accumulators. The
    network's loop stays as it is: it calls `apply` exactly where it calls
    `layer.apply_accumulated_gradient` today, interleaved with each layer's accumulate, in
    forward order. The formulas are moved verbatim from today's `apply_accumulated_gradient`: the
    same expressions and the same in-place `-=` on numpy, and the same fused op per layer on Rust.
    That op takes W and b together, so there is still one crossing per layer.
  - `step_single(layer, input_activation, learning_rate)` is the single-example path. For `SGD`
    on Rust it calls the fused `layer_sgd_step`, as `RustArrayLayer.sgd_step` does today.
    Otherwise it accumulates, then applies at `batch_size=1`.
  - `state()` / `load_state(state)` give the table as nested lists, and take it back through the
    backend's `owned`. This is what stage 5 saves.
  - The pure-Python optimizer holds the same table per node or kernel, as lists of floats, with
    the scalar formulas from the node factories.
- **The layer protocol loses the update.** `ArrayNetworkLayer` drops `apply_accumulated_gradient`
  and `sgd_step`. A weighted layer exposes `W`, `b`, `grad_W`, `grad_b` and
  `reset_gradient_accum()`. `ArrayNetworkBase._learn_input` and `_learn_batch_input` call the
  optimizer after the accumulate pass. Activation subclasses are unaffected.
- **One network build path.** `ArrayNetworkBase` builds `self.layers` from specs, so
  `ArrayConvShape`'s own `__init__` goes. randomize, snapshot and restore walk the layers through
  one protocol: `WeightedArrayLayer` gains `fan_in`, so dense and conv randomize alike, and a
  weightless layer (pool) snapshots as `()`. `_set_training_mode` calls every layer that has
  `set_training_mode`, so the dropout networks' overrides go. The shape mixins keep targets and
  classification.
- **Checkpoints.** `snapshot()` / `restore()` stay weights only, since every current caller wants
  exactly that. `checkpoint()` / `restore_checkpoint()` add the optimizer state. `train.py`'s
  best-epoch restore moves to checkpoints, so a network saved after training carries the state
  that matches its weights. Training output doesn't change: the restore is the last thing
  training does.

## Format 2

```json
{
  "format": 2,
  "implementation": "numpy",
  "shape": "multiclass",
  "input": {"dimension": 64},
  "layers": [{"kind": "dense", "size": 30, "activation": "sigmoid"}, {"kind": "dense", "size": 10, "output": true}],
  "update_rule": {"rule": "adam", "beta1": 0.9, "beta2": 0.999, "epsilon": 1e-8},
  "weights": [[W, b], [W, b]],
  "optimizer_state": {"t": 120, "layers": [{"m_W": ..., "v_W": ..., "m_b": ..., "v_b": ...}, ...]}
}
```

- `input` holds `dimension`, or `height`/`width` for conv, and `input_bounds` for pure Python.
  `class_count` comes from the output spec.
- The numpy and Rust files load into each other, as the conv envelope's already do. Pure-Python
  files hold per-node weights and state, and load into pure Python only. That is the same rule
  as today.
- A preset's `load` reads format 2 or its legacy envelope. For format 2 it checks that the file's
  spec and rule are its own, and fails with the difference otherwise. `load_network(path)` builds
  whatever a format-2 file describes.
- The ensembles' envelopes hold format-2 sub-networks.
- Resuming is exact for the optimizer: train N steps, save, load and train M more equals N + M
  steps without the save, by bits. Dropout's masks and the epoch shuffle come from global RNG
  state that isn't the network's (docs/rng-audit.md, Three global states), so the resume tests
  use rules without dropout, or reseed. Saving RNG state is out of scope.

## Pitfalls to design around

- **Bit-identity comes from moving code, not rewriting it.** Each formula moves with its operand
  order and grouping unchanged (README, Update rules: `g / B`, never `lr / B`). numpy's in-place
  `W -= ...` moves as it is, not rewritten into a rebinding form.
- **One global `t` replaces the per-layer and per-node counts.** They are equal today because
  every layer steps once per `learn*` call. The optimizer increments `t` once in `begin_step`,
  before any layer applies, and each layer used to increment its own at the start of its apply,
  so every layer sees the same `t` as before. The golden run's Adam entries check
  it.
- **Update order.** Today the final loop accumulates and applies layer by layer, in forward
  order, after every delta has been computed. The loop keeps that order and only swaps the apply
  call. Accumulating reads deltas, activations and conv's cached `cols`, never a weight, so the
  order couldn't change a value anyway. Keeping it means nobody has to argue that.
- **The draw order is part of the contract.** randomize draws W then b per layer, in forward
  order, and pool layers draw nothing. The dropout masks are drawn in forward order during
  training only. A generic walk has to keep both orders, or seeded runs change
  (`test_seeded_init_parity.py`, and the golden run's dropout entries).
- **Per-call overhead.** The single-example Rust ops at small shapes are dominated by the Python
  boundary. The optimizer adds a method call and a state-table lookup per layer per step. Each
  stage that touches `learn*` is timed with `scripts/prepared_dataset_timing.py time`, numpy and
  Rust in separate processes, both builds committed and alternated
  ([optimizations/measurement.md](optimizations/measurement.md)). It must be within run-to-run
  noise. If it isn't, the state is held as a list indexed like `layers` instead of a dict, and the
  SGD path skips the table.
- **The ensembles pickle** classes and snapshots across `multiprocessing.Pool`
  (`ensemble_train.py`). Rules are dataclasses and pickle. The optimizer's state crosses as nested
  lists, as snapshots already do. The pure-Python factory classes are closures and don't pickle,
  and deleting them in stage 6 removes that limit.
- **Registry walks** assert completeness over `all_subclasses(...)`. Deleting a class (stage 6)
  or adding a generic one (stage 3) changes what they walk, so update their expected sets in the
  same PR, never loosen them.
- **Golden files are per machine** (BLAS). Record on `main` just before stage 1, on the Ryzen
  machine, and check every later stage on it. The golden run's save/load round trip reads
  whatever format `save` writes, so it keeps working across stage 5.

## Stages

Each stage is one PR in indrajala-ml. None changes the crate. Every stage passes `./cli test`,
`./cli lint`, the golden run bit-identical, and the legacy save fixtures. Stages that touch
`learn*` or `classify_rows` are also timed within noise.

### Stage 0: the gates

1. **Extend the golden run** to cover what the plan touches: numpy and Rust momentum conv; a
   two-conv, strided configuration (`ConvSpec(3, 2, stride=2), ConvSpec(2, 2)`, no pool), so a
   conv layer follows a conv layer; and the pure-Python networks at small shapes (plain, momentum,
   Adam, L2, ReLU, softmax, cross-entropy, dropout, conv, momentum conv, and the ensemble). The
   pure-Python networks are slow, so keep them to a few steps each. They are compared by bits like
   the rest.
2. **Legacy save fixtures**: `tests/fixtures/saved_models/`, one small file per saveable network
   class, written by today's `main` with fixed weights and hyperparameters. Add a test that loads
   each file with its class and compares the predictions stored beside it by bits. Every legacy
   envelope must keep loading after stage 5.
3. **Baseline timings**, recorded in the PR: `prepared_dataset_timing.py time` for dense and conv,
   single-example and mini-batch 32, numpy and Rust.

Done when the extended golden run is recorded on `main` and the fixtures test is green.

### Stage 1: the optimizer for numpy and Rust dense networks

1. Add `update_rules.py` (D2) and `optimizers.py` with the numpy and Rust optimizers. They hold
   the code moved from the 6 dense update subclasses and `momentum_update`, and the global `t`.
2. `ArrayNetworkBase` owns the optimizer and calls it after the accumulate pass. `ArrayLayer` and
   `RustArrayLayer` lose `apply_accumulated_gradient` and `sgd_step`, and gain
   `reset_gradient_accum()`. The fused SGD step moves to `step_single`.
3. The Momentum, Adam and L2 dense networks build their rule from their constructor arguments.
   `hyperparameters` and `_extra_state` still write the same flat keys, so the files are unchanged
   until stage 5. The update subclasses remain as plain `ArrayLayer`s under their old names until
   stage 6, so their imports keep working.
4. Conv layers keep their own update for now. The optimizer calls a layer's
   `apply_accumulated_gradient` where the layer still has one, which lasts until stage 3.
5. `test_rust_array_layer_sgd_step.py` checks the new invariant: the fused step is used iff the
   rule is `SGD`. `test_update_rule_forms.py` runs its forms against the optimizers.

Done when the golden run is bit-identical and dense learn timing is within noise.

### Stage 2: the optimizer for pure Python

1. The pure-Python optimizer holds the per-weight formulas moved from the node factories, and one
   state table per node, keyed by (layer index, node index). The operation order is unchanged.
   `test_update_rule_forms.py` already checks each form bit for bit.
2. `BackpropNetworkBase` owns the optimizer and calls it where it calls `apply_gradients` and
   `apply_accumulated_gradients` today. `BackpropNode` and `ConvKernel` keep their accumulators
   and lose the update.
3. The pure-Python conv kernels share the same optimizer through the kernel's weight list.
   `make_dropout_node_cls` is an activation, not a rule, so it stays as it is.

Done when the pure-Python golden entries are bit-identical.

### Stage 3: conv under the optimizer, and one array build path

1. The optimizer steps conv layers too. `ConvArrayLayer` and `ConvRustArrayLayer` lose their
   update, and the momentum conv layers become plain conv layers under their old names. No new
   combination is exposed yet.
2. `layer_specs.py`, and the builder for numpy and Rust, with the validity rules from The design.
3. `ArrayNetworkBase` builds from specs. `ArrayConvShape.__init__` and
   `build_conv_array_network_layers` fold into the builder, and randomize, snapshot, restore and
   `_set_training_mode` become the generic walks. The dropout networks lose their overrides.
4. Every array preset builds its spec from its existing constructor arguments. Add
   `SequentialArrayNetwork`, multiclass and single-output.
5. `tests/array_network_contract.py` gains spec-built cases next to the presets: at least one
   spec per preset, built generically, which must match the preset by bits.

This is the largest stage. Split it at 3.1 | 3.2-3.3 | 3.4-3.5 if the diff passes about 1,500
lines. Done when the golden run is bit-identical, conv and dense timing is within noise, and the
fixtures load.

### Stage 4: pure-Python networks from the same specs

1. The builder maps specs to pure-Python layers (`BackpropLayer` and its siblings, `ConvLayer`,
   `MaxPoolLayer`). `BackpropNetworkBase` and the conv network share the spec build, as in
   stage 3.
2. The pure-Python presets keep their names and signatures.
3. The parity contract builds its reference network from the same spec as the array network under
   test, instead of a hand-paired class (`matching_*_array_backprop_networks`).

Done when the pure-Python golden entries are bit-identical and the fixtures load.

### Stage 5: format 2, with optimizer state (a feature)

This stage changes what files hold, not how anything trains.

1. `optimizer.state()` / `load_state()` for all three implementations, and `checkpoint()` /
   `restore_checkpoint()` on every network. `train.py`'s best-epoch restore uses checkpoints.
2. The format-2 writer and reader. Every `save` writes format 2, and every `load` reads format 2
   and its legacy envelopes. Add `load_network(path)`. The ensembles' envelopes nest format-2
   sub-networks.
3. Tests:
   - a resume test per rule and per implementation: train N, save, load, train M, which must
     equal N + M by bits;
   - numpy-to-Rust and Rust-to-numpy loads of format-2 files;
   - a preset refuses a file whose spec or rule isn't its own;
   - the legacy fixtures still load, with fresh optimizer state.
4. README: the save format, and that files from this version don't load on older code.

Done when the golden run is bit-identical, the resume tests pass by bits, and the fixtures load.

### Stage 6: remove the scaffolding and update the docs

1. Delete the update-only layer subclasses (D1). Move their tests onto the optimizers, and update
   the registry walks' expected sets. The four pure-Python rule factories already went in stage
   2, with their tests.
2. README: the Models section describes specs, rules, the optimizer and presets, and the
   class-prefix table becomes a preset table. Mark step 1 done in `primitives-roadmap.md`.
3. List the new combinations the plan made possible but didn't enable (below).

Done when no class exists only to carry an update rule.

## After this plan

- **New combinations** are behaviour changes, not refactoring, so each is its own PR with
  hand-computed tests and parity tests in all three implementations: Adam and L2 for conv
  (numerics that exist, applied to conv weights), and ReLU hidden layers under every rule.
- **Weight decay with momentum or Adam** needs a published form chosen and cited first (README,
  Update rules).
- **Batch normalization** (roadmap step 2) gets its own workplan. It will need the split hidden
  delta, because a dense layer followed by a norm layer can't read `W` from the next layer. That
  means a sigmoid-derivative crate op, bit-identical by the argument above, and a check that the
  extra crossing is within noise. Its running averages are layer state, not optimizer state, and
  format 2 saves them with the weights.
- **Saving RNG state** so a dropout run resumes exactly. This waits on explicit generator objects
  (docs/rng-audit.md, Open work).

## Out of scope

- Any change to training numerics, the crate or the kernels.
- Separate activation layers, which would add a crossing per layer on Rust. Activations stay fused.
- Parameter groups, per-layer learning rates, and schedulers beyond today's `lr_schedule.py`.
- Deleting or renaming any network class (D1).
