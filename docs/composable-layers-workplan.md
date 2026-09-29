# Workplan: composable layers and update rules

**Status: planned. Decisions D1-D5 are the owner's. Each has a recommendation, and stage 0 can
start before they are settled; stage 1 needs D1, D2 and D5.**

This is step 1 of [primitives-roadmap.md](primitives-roadmap.md). It is a structural refactoring:
it changes structure only, never numerics, under the README's Refactoring rules. At the end, a
network is built from a list of layer specs and one update rule, in all three implementations.
Batch normalization (step 2) is then one new layer kind, not one new class per combination.

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

- **39 backprop network classes** in `indrajala_ml/model/` (40 `*ClassifierNetwork`s less the perceptron), over three bases: `BackpropNetworkBase`
  (pure Python), `ArrayNetworkBase` with `NumpyArrayNetworkBase` and `RustArrayNetworkBase`.
  The shape mixins in `array_network_shapes.py` (multiclass, single-output, conv) hold targets,
  classification and save/load.
- **Array sibling networks** are two-line classes: they set `hidden_layer_cls`,
  `output_layer_cls` and `hyperparameters`, and store each hyperparameter as an attribute before
  `super().__init__()`. `_new_layer` passes the hyperparameters to the layer's constructor by
  name, and `_extra_state` writes them as flat keys in the save file (`{"momentum": 0.9}`).
- **Array update rules** live in `apply_accumulated_gradient` of 8 layer classes (momentum, Adam
  and L2 for each backend, and momentum conv for each backend). numpy's momentum is a shared
  function (`momentum_update`). The Rust layers call the crate's fused ops
  `layer_{,momentum_,adam_,l2_}apply_accumulated_gradient`. **Every one of these ops is
  shape-agnostic**: it shape-checks W against its gradient and state, and b likewise. So a rule
  object can drive conv weights with no crate change. Momentum conv already does this.
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
  `make_adam_node_cls`, `make_l2_node_cls`, `make_dropout_node_cls`), plus
  `make_momentum_kernel_cls` for conv. A network sets its layer classes as instance attributes in
  `__init__`. The per-node form follows the literature's per-weight formulas and is the parity
  reference for the array networks.
- **Save files** hold the constructor's shape arguments, the snapshot and the flat
  hyperparameters. They don't record the class: `cls.load(path)` needs the right class. Optimizer
  state (velocities, Adam's m, v and t) is not saved, for any network.
- **Registry walks in the tests**: `test_seeded_init_parity.py` asserts it covers every subclass
  of `ArrayNetworkBase`. `test_prepared_dataset.py` walks them too, and
  `test_rust_array_layer_forward_batch.py` and `test_rust_array_layer_sgd_step.py` walk every
  subclass of `RustArrayLayer`. `tests/array_network_contract.py` generates the per-network parity
  tests from an `ArrayNetworkSpec` with positional hyperparameters.
- **The golden run** (`scripts/golden_training_run.py`) covers the 16 dense multiclass array
  networks, the numpy and Rust conv networks (`ConvSpec(3, 2), PoolSpec(2)`), 4 single-output
  networks and 2 ensembles. It **doesn't cover momentum conv**, which the batch-size study trains,
  or any pure-Python network.

## Decisions (owner)

- **D1. Public names stay, as presets.** Recommended: every network class keeps its name,
  constructor signature and save format. Each becomes a preset that builds a spec and a rule. This
  is what the README's Refactoring rule requires: demos, `demos/registry.py`,
  `ensemble_train.py`, the scripts and the tests construct them by name, and saved files must
  load. The layer subclasses that exist only for an update rule (the 8 array ones and
  `make_momentum_kernel_cls`) are imported only by tests. Recommended: delete them in stage 5,
  and move their tests onto the rule objects. The alternative, keeping them as aliases, keeps
  names nobody constructs.
- **D2. The update rule is a frozen dataclass**, passed as one object: `SGD()`,
  `Momentum(momentum=0.9)`, `Adam(beta1, beta2, epsilon)` with the published defaults,
  `WeightDecay(l2_lambda)`. Recommended over strings plus kwargs: typed under strict pyright,
  hashable, picklable across the ensemble's worker boundary, and it saves as
  `{"rule": "momentum", "momentum": 0.9}`. `WeightDecay` is its own rule, SGD with eq. (8).
  Weight decay combined with momentum or Adam is a different published rule each time (Goyal
  eq. (10)'s form, or AdamW against Adam with L2), so it isn't a flag on the other rules. It is
  out of scope here (see After this plan).
- **D3. Pure Python takes the same specs and rules.** Recommended. It is the parity reference for
  every combination, so a combination the array networks can build without a pure-Python
  counterpart can't be parity-tested. Batch norm needs a pure-Python reference too. The node-level
  formulas stay per weight, as they are, and only the dispatch changes. The alternative is to leave
  the pure-Python factories as they are, which saves stages 2 and 4. Then every new combination
  needs a hand-made pure-Python class, the problem this plan removes.
- **D4. The save format.** Recommended: the presets keep writing exactly today's envelopes,
  checked against committed fixture files (stage 0). A network built from specs directly writes a
  new envelope, `"format": 2`, with `"layers"` (the specs) and `"update_rule"` (D2's dict). It is
  loaded by a generic `load_network(path)`, which dispatches on the backend recorded in the file.
  Old envelopes are never rewritten.
- **D5. Optimizer state lives in the layer.** Recommended: the rule creates a per-layer state
  object (`rule.new_state(backend, W, b)`) that holds velocities or m, v and t, and steps W and b.
  This is where the state lives today. Snapshot, pickling and the per-layer `t` stay as they are,
  and nothing is keyed by `id(layer)`. A central PyTorch-style optimizer holding a dict per
  parameter buys nothing here: there are no parameter groups, and every layer has one `(W, b)`
  pair.

## The design

```python
# a spec is backend-free data; the builder maps it to a backend's layer class
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
- **Update rules** (`indrajala_ml/model/update_rules.py`): D2's dataclasses. Each backend has a
  state class per rule whose `apply(W, b, grad_W, grad_b, learning_rate, batch_size)` holds the
  code moved verbatim from today's `apply_accumulated_gradient`: the same expressions and the same
  in-place `-=` on numpy, the same fused op on Rust. `rule.fuses_sgd_step` is true only for `SGD`,
  so `RustArrayLayer.sgd_step` fuses exactly where it does today.
- **Layers take a rule, not a subclass.** `ArrayLayer`, `RustArrayLayer`, `ConvArrayLayer` and
  `ConvRustArrayLayer` get an `update_rule` constructor keyword (default `SGD()`), and
  `apply_accumulated_gradient` delegates to its state. Activation subclasses are unaffected. A
  ReLU layer with Adam is `ReLUArrayLayer(..., update_rule=Adam())`.
- **One network build path.** `ArrayNetworkBase` builds `self.layers` from specs, so
  `ArrayConvShape`'s own `__init__` goes. randomize, snapshot and restore walk the layers through
  one protocol: `WeightedArrayLayer` gains `fan_in`, so dense and conv randomize alike, and a
  weightless layer (pool) snapshots as `()`. `_set_training_mode` calls every layer that has
  `set_training_mode`, so the dropout networks' overrides go. The shape mixins keep targets,
  classification and the envelopes.

## Pitfalls to design around

- **Bit-identity comes from moving code, not rewriting it.** Each formula moves with its operand
  order and grouping unchanged (README, Update rules: `g / B`, never `lr / B`). numpy's in-place
  `W -= ...` stays in place: a rebinding form gives the same values, but it would change aliasing
  that `snapshot` and `restore` rely on.
- **The draw order is part of the contract.** randomize draws W then b per layer, in forward
  order, and pool layers draw nothing. The dropout masks are drawn in forward order during
  training only. A generic walk has to keep both orders, or seeded runs change
  (`test_seeded_init_parity.py`, and the golden run's dropout entries).
- **Per-call overhead.** The single-example Rust ops at small shapes are dominated by the Python
  boundary. One more attribute lookup and method call per layer per step can show up. Each stage
  that touches `learn*` is timed with `scripts/prepared_dataset_timing.py time`, numpy and Rust in
  separate processes, both builds committed and alternated
  ([optimizations/measurement.md](optimizations/measurement.md)). It must be within run-to-run
  noise, or the stage keeps a fast path, as `sgd_step` already is one.
- **The ensembles pickle classes and snapshots** across `multiprocessing.Pool`
  (`ensemble_train.py`). Specs and rules must pickle. The pure-Python factory classes are closures
  and don't pickle, which is why no pure-Python sibling is an ensemble classifier today. Stage 4
  shouldn't make this worse.
- **Registry walks** assert completeness over `all_subclasses(...)`. Deleting a class (stage 5)
  or adding a generic one (stage 3) changes what they walk, so update their expected sets in the
  same PR, never loosen them.
- **Golden files are per machine** (BLAS). Record on `main` just before stage 1, on the Ryzen
  machine, and check every later stage on it.

## Stages

Each stage is one PR in indrajala-ml. None changes the crate. Every stage passes `./cli test`,
`./cli lint`, the golden run bit-identical, and the save fixtures. Stages that touch `learn*` or
`classify_rows` are also timed within noise.

### Stage 0: the gates

1. **Extend the golden run** to cover what the plan touches: numpy and Rust momentum conv; a
   two-conv, strided configuration (`ConvSpec(3, 2, stride=2), ConvSpec(2, 2)`, no pool), so a
   conv layer follows a conv layer; and the pure-Python networks at small shapes (plain, momentum,
   Adam, L2, ReLU, softmax, cross-entropy, dropout, conv, momentum conv, and the ensemble). The
   pure-Python networks are slow, so keep them to a few steps each. They are compared by bits like
   the rest.
2. **Save fixtures**: `tests/fixtures/saved_models/`, one small file per saveable network class,
   written by today's `main` with fixed weights and hyperparameters. Add a test that loads each
   file with its class and compares the predictions stored beside it by bits. This pins D4's
   promise that old files keep loading.
3. **Baseline timings**, recorded in the PR: `prepared_dataset_timing.py time` for dense and conv,
   single-example and mini-batch 32, numpy and Rust.

Done when the extended golden run is recorded on `main` and the fixtures test is green.

### Stage 1: update rules as objects, numpy and Rust, dense layers

1. Add `update_rules.py`: `SGD`, `Momentum`, `Adam` and `WeightDecay`, with a numpy and a Rust
   state per rule, holding code moved from the 6 dense update subclasses.
2. `ArrayLayer` and `RustArrayLayer` take `update_rule`. `sgd_step` fuses iff
   `update_rule.fuses_sgd_step`.
3. The Momentum, Adam and L2 array networks set `update_rule` instead of their layer classes.
   `hyperparameters` and `_extra_state` still write the same flat keys, so the files are
   unchanged. The update subclasses stay for now, as thin classes that pass a rule. They go in
   stage 5.
4. `test_rust_array_layer_sgd_step.py` checks the new invariant (fused iff the rule is `SGD`), and
   `test_update_rule_forms.py` runs its forms against the rule states as well as the layers.

Done when the golden run is bit-identical and dense learn timing is within noise.

### Stage 2: update rules for pure Python

1. The per-weight formulas move from the node factories into scalar forms of the same rule
   objects (`rule.new_node_state(fan_in)` with `apply` over a node's weight list and its bias). The
   operation order is unchanged, and `test_update_rule_forms.py` checks each form bit for bit
   already.
2. `BackpropNode`, and `ConvKernel` for conv, take a rule. The four factories return the same
   classes as thin wrappers until stage 5. `make_dropout_node_cls` is an activation, not a rule,
   so it stays as it is.

Done when the pure-Python golden entries are bit-identical.

### Stage 3: conv layers take the rule, and one array build path

1. `ConvArrayLayer` and `ConvRustArrayLayer` take `update_rule`, and the momentum conv layers
   become thin classes like stage 1's. No new combination is exposed yet.
2. `layer_specs.py`, and the builder for numpy and Rust, with the validity rules from The design.
3. `ArrayNetworkBase` builds from specs. `ArrayConvShape.__init__` and
   `build_conv_array_network_layers` fold into the builder, and randomize, snapshot, restore and
   `_set_training_mode` become the generic walks. The dropout networks lose their overrides.
4. Every array preset builds its spec from its existing constructor arguments. Add
   `SequentialArrayNetwork` (multiclass and single-output), and D4's format-2 envelope with
   `load_network`.
5. `tests/array_network_contract.py` gains spec-built cases next to the presets: at least one
   spec per preset, built generically, which must match the preset by bits.

This is the largest stage. Split it at 3.1 | 3.2-3.3 | 3.4-3.5 if the diff passes about 1,500
lines. Done when the golden run is bit-identical, conv and dense timing is within noise, and the
fixtures load.

### Stage 4: pure-Python networks from the same specs

1. The builder maps specs to pure-Python layers (`BackpropLayer` and its siblings, `ConvLayer`,
   `MaxPoolLayer`). `BackpropNetworkBase` and the conv network share the spec build, as in
   stage 3.
2. The pure-Python presets keep their names, signatures and `save_model_json` envelopes.
3. The parity contract builds its reference network from the same spec as the array network under
   test, instead of a hand-paired class (`matching_*_array_backprop_networks`).

Done when the pure-Python golden entries are bit-identical and the fixtures load.

### Stage 5: remove the scaffolding and update the docs

1. Delete the update-only layer subclasses and the rule factories (D1). Move their tests onto the
   rules, and update the registry walks' expected sets.
2. README: the Models section describes specs, rules and presets, and the class-prefix table
   becomes a preset table. Mark step 1 done in `primitives-roadmap.md`.
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
  extra crossing is within noise.
- **Saving optimizer state** to resume training: format 2 has room for it. It isn't needed yet.

## Out of scope

- Any change to the numerics, the crate or the kernels.
- Separate activation layers, which would add a crossing per layer on Rust. Activations stay fused.
- A central optimizer object, parameter groups, per-layer learning rates and schedulers beyond
  today's `lr_schedule.py`.
- Deleting or renaming any network class (D1).
