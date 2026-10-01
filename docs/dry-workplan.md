# Workplan: remove duplicated code (DRY)

**Status: decisions D1-D6 settled by the owner (2026-10-01); stages 1-8 are planned.**

An audit of indrajala-ml on 2026-10-01 (`main` at 8dc0a07) found code that is written twice where
one copy would do. This plan removes it, one PR per stage. It is refactoring: no network, preset,
save file or numeric result changes, so the golden run must stay bit-identical at every stage.

## Why

- **Copies drift.** A fix made in one copy and not its twin is how the implementations stop
  agreeing, and parity between pure Python, numpy and Rust is the project's core invariant. The
  copies found here are glue (persistence, ensembles, optimizer bookkeeping, shape walking,
  argument checks), not numerics, so sharing them costs nothing in parity.
- **New primitives multiply them.** Residual connections and attention (roadmap steps 3-5) each
  add a layer kind to both builders, both network bases and the twin layer classes. Every copy
  left in place is a place the next layer has to be written twice.

## How the audit was done

- `symilar` (pylint's duplicate finder; imports, docstrings and signatures ignored) over
  `indrajala_ml/`, `scripts/` and `cli`: 0.29% duplicated at 8 lines or more, and 24 blocks at 5
  lines or more. Over `tests/`: 0.91% at 10 lines or more.
- Function names defined in more than one module, and test helpers defined in more than one file.
- Reading each pair it found, and the twin classes (numpy and Rust layers, the three optimizers,
  the two builders, the two network bases) side by side, for copies `symilar` misses because a
  name or a type differs.

Literal duplication is low: most of the repo already shares what it can (`WeightAndBias`,
`validate_layer_specs`, `run_json_worker`, `interleaved_runs`, the array backends). What is left
is below.

## The three implementations are not duplication

Pure Python, numpy and Rust compute the same things three times on purpose: each is the others'
reference. Their numerics stay separate, and so do the numpy and Rust layer twins' method bodies
(a numpy expression against one fused Rust call each, by design). Pure Python stays free of numpy
and Rust imports (`python_optimizer.py`'s docstring). This plan shares only code that is the same
in every implementation that has it, and puts anything pure Python needs in a numpy-free module.

## Findings

Line numbers are from 8dc0a07.

### In the package

1. **The two ensemble bases.** `ArrayEnsembleBase` (`array_ensemble_base.py:29-54`) and
   `EnsembleBackpropClassifierNetwork` (`ensemble_backprop_classifier_network.py:25-52`) have the
   same `__init__`, `predict_probabilities`, `classify_state`, `snapshot`, `restore`, `checkpoint`
   and `restore_checkpoint`, about 25 lines each. Only `save`/`load` differ.
2. **Format-2 persistence on the two network bases.** `ArrayNetworkBase`
   (`array_network_base.py:293-328`) and `BackpropNetworkBase` (`backprop_network_base.py:223-257`)
   have the same `save`, `load`, `from_format2`, `_from_file`, `_load_legacy`, `checkpoint` and
   `restore_checkpoint`. They differ in one value: the implementation name `check_kind` is given
   (`cls.backend.name`, or `PYTHON`).
3. **The pure-Python `learn`.** `BackpropClassifierNetwork.learn` (`backprop_classifier_network.py:46-55`)
   and `MultiClassBackpropClassifierNetwork.learn` (`multiclass_backprop_classifier_network.py:50-62`)
   are the same body with a different label type. The array networks already have one `learn` on
   their base (`array_network_base.py:167`).
4. **The three optimizers' bookkeeping.** `NumpyOptimizer`, `RustOptimizer` (`optimizers.py:60-114`,
   `187-248`) and `PythonOptimizer` (`python_optimizer.py:43-61`) each repeat the rule dispatch
   (`match rule` onto `_apply_*`), `t` and `begin_step`. The numpy and Rust ones also repeat
   `state`, `load_state`, the pool-layer skip in `apply` and `_zeros`, which differ only in the
   backend's copy and zeros, already in `array_backend.py` (`owned`, `zeros`).
5. **Walking the specs' shapes.** `build_array_layers` (`array_layer_builder.py:100-133`) and
   `build_python_layers` (`python_layer_builder.py:41-93`) each work out every layer's input shape,
   a conv or pool layer's output shape, and a batch-norm layer's `positions`. Each new layer kind
   adds the same shape rule to both.
6. **Conv and pool argument checks and output size.** `validate_conv_arguments` and
   `validate_pool_arguments` live in the numpy modules (`conv_array_layer.py:17`,
   `max_pool_array_layer.py:14`), so pure Python can't import them: `ConvLayer`
   (`conv_layer.py:60-82`) and `MaxPoolLayer` (`max_pool_layer.py:84-104`) repeat the same asserts
   and messages. The output size `(input - k) // stride + 1` is written four times (both pure-Python
   layers, `ConvGeometryArrayLayer`, `MaxPoolArrayLayer`). The Rust layers take it from the crate's
   `pa.ConvGeometry`, which stays.
7. **Batch norm's accessors on its numpy and Rust twins.** `BatchNormArrayLayer` and
   `BatchNormRustArrayLayer` repeat `parameters`, `gradients`, `set_parameters`, `running_state`,
   `set_running_state`, `set_training_mode`, `decayed` and the two output-delta refusals. These
   are what `WeightAndBias` (`array_parameters.py`) already shares for (W, b) layers.

### In the tests

8. **Test helpers.** `_array_layer_like` (a numpy or Rust layer with a pure-Python layer's weights)
   is in six files (the three optimizer tests, `test_relu_array_layer.py`,
   `test_softmax_array_layer.py`, `test_cross_entropy_array_layer.py`), with a `layer_cls` fixture
   in nine. `snapshot_bits` is identical in `tests/array_network_contract.py` and
   `tests/test_seeded_init_parity.py`.
9. **The three array-optimizer test files.** `test_array_optimizer_{momentum,adam,weight_decay}.py`
   (about 140 lines each) are the same two parity tests (batch size one, then whole batches,
   against the pure-Python optimizer at every step) and the same reset test, with a different rule.
   Each also has one test of its own rule.
10. **The numpy and Rust batch-norm network tests.** `test_batch_norm_array_network.py` and
    `test_batch_norm_rust_network.py`, and their conv pair, each repeat the network-level tests
    (running averages move only in training, classifying uses them, per-parameter optimizer state,
    weight decay skips gamma and beta, a one-example step is refused, snapshot and checkpoint
    round trips) with only the backend changed. `_network` already takes the backend. Their
    layer-level and parity tests are backend-specific and stay where they are.

## Decisions (settled)

The owner settled D1-D6 on 2026-10-01, each as recommended in the draft.

- **D1. Scope: indrajala-ml only.** The crate (`rust/`, indrajala-math-rust) is a separate repo
  with its own conventions, and none of the findings is in it. It gets its own audit if wanted.
- **D2. The per-rule presets stay as they are, for now.** The 12 preset classes that set a rule
  (Adam, Momentum, conv Momentum and L2, each in pure Python, numpy and Rust) each repeat a
  hyperparameter `__init__` and a two-line `_update_rule`. A mixin per rule could hold
  `_update_rule` only: each `__init__` keeps its shape's explicit, typed signature, which strict
  pyright and `preset_init_kwargs` need. More presets are coming, and the better form will be
  clearer then, so this is revisited when they are added ([next-steps.md](next-steps.md) once
  this plan retires).
- **D3. The numpy and Rust layer twins share their identical methods only (finding 7).** Their
  method bodies stay separate (see "The three implementations are not duplication"). Rejected: a
  backend-generic layer whose array operations go through the backend, which would put an
  indirection on every hot call and undo the "one fused Rust call per method" design.
- **D4. The tests are in scope (stages 7 and 8).** Merging test files changes which file a test
  lives in, not what it checks: each stage's PR lists the tests collected before and after per
  merged file, and every old test maps to a new one.
- **D5. Scripts and demos are out of scope.** The timing scripts share a few lines of median and
  table printing, and two demos share a three-line `_backend_array`. The scripts are the
  measurement tools `scripts/ab.py` runs, and the copies are short: touching them buys little and
  risks a measurement.
- **D6. Only stage 4 (the optimizers) gets an A/B.** The optimizer's `apply` runs inside every
  numpy and Rust `learn*` step. No other stage changes a numpy or Rust `learn*` or `classify_rows`
  path (stages 1-2 and 5-6 are construction, persistence and accessors; stage 3 is pure Python,
  which is never benchmarked; stages 7-8 are tests), so they need none: "a PR that changes no
  `learn*` or `classify_rows` path needs no A/B" (docs/measurement.md). Each such PR says so.

## Stages

Every stage is one PR and passes `./cli lint`, `./cli test` and
`.venv/bin/python scripts/golden_training_run.py check data/refactoring/golden_run.json`
(bit-identical). No public class, module or function a demo, test or saved file uses is renamed or
deleted.

### Stage 1: one ensemble base (finding 1)

A numpy-free `EnsembleBase[ClassifierT]` holds the shared methods; `ArrayEnsembleBase` and
`EnsembleBackpropClassifierNetwork` keep their own `save`/`load` and typing.

### Stage 2: one format-2 persistence (finding 2)

A numpy-free mixin holds `save`, `load`, `from_format2`, `_from_file`, `_load_legacy`,
`checkpoint` and `restore_checkpoint`, with the implementation name as a class-level hook.
`tests/test_format2.py` and the legacy-load tests cover it unchanged.

### Stage 3: one pure-Python `learn` (finding 3)

`learn` moves onto `BackpropNetworkBase`; the two classifiers keep their `learn_batch` signatures.

### Stage 4: one optimizer skeleton (finding 4)

- A numpy-free base holds `rule`, `t`, `begin_step` and the dispatch, for all three optimizers.
- An array base holds `state`, `load_state`, the pool-layer skip and `_zeros` through the
  backend's `owned` and `zeros`. `RustOptimizer` keeps its fused SGD `step_single` and its
  rebinding `apply`.
- Pitfall: `_rust_zeros` makes zeros shaped like a parameter, including the zero-length array in a
  linear layer's `b` place. The shared `_zeros` must keep that.
- A/B (D6): `scripts/ab.py run --bench prepared_dataset_timing`, report in the PR.

### Stage 5: one shape walk and one set of conv and pool checks (findings 5 and 6)

- `validate_conv_arguments`, `validate_pool_arguments` and the output size move to a numpy-free
  module; the pure-Python, numpy and Rust layers call them. Pure Python keeps its extra
  input-node-count check.
- One function gives each spec's input shape, output shape and batch-norm `positions`; both
  builders loop over it and keep only the choice of class.

### Stage 6: batch norm's accessors (finding 7)

A `GammaAndBeta[A]` beside `WeightAndBias` in `array_parameters.py`, used by both batch-norm array
layers.

### Stage 7: test helpers and the optimizer tests (findings 8 and 9)

- `array_layer_like` and the `layer_cls` fixture move to `tests/helpers.py`; `snapshot_bits` is
  kept once, in `tests/array_network_contract.py`.
- `test_array_optimizer.py` holds the shared tests parametrized by rule (Momentum, Adam,
  WeightDecay) and each rule's own test. The three files are deleted.

### Stage 8: the batch-norm network tests (finding 10)

The network-level tests that differ only in backend move into backend-parametrized tests, one file
for dense and one for conv. The layer-level and Rust-against-numpy parity tests stay in the
backend files.

## After this plan

- Retire this workplan once every stage is merged: anything left open moves to
  [next-steps.md](next-steps.md).
- The per-rule presets (D2), revisited when more presets are added.
- A rerun of the audit after each new primitive, so new copies are caught while they are small.

## Out of scope

- The crate (D1), the per-rule presets until more exist (D2), the layer twins' numerics (D3),
  scripts and demos (D5).
- Renaming or deleting public classes (the composable-layers workplan's D1).
- Any change to a numeric result, a saved file or the golden run.
