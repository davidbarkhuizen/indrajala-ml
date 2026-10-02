# Workplan: remove duplicated code, second pass (DRY rerun)

**Status: in progress; decisions D1-D9 settled by the owner (2026-10-02). Stages 1-5 done.**

The first DRY workplan (#506-#515, `git show e066333:docs/dry-workplan.md`) asked for the audit
to be rerun after each new primitive ([next-steps.md](next-steps.md), "From the DRY audit").
Residual connections (#532-#539) and layer norm and attention (#542-#549) have merged since, so
the audit was rerun on 2026-10-02 (`main` at ebfcc47). This plan removes what it found, one PR per
stage. It is refactoring: no network, preset, save file or numeric result changes, so the golden
run must stay bit-identical at every stage.

## Why

Literal duplication is lower than last time, but the copies that are left came in with the two
new primitives, which were written per backend without using the shared homes the first pass set
up. Each copy is a place the next layer kind (multi-head attention, other norms) has to be
written again, and a place where two copies can drift apart.

## How the audit was done

The method from the first pass ([next-steps.md](next-steps.md)):

- `symilar` (pylint's duplicate finder, `-i --ignore-docstrings --ignore-imports
  --ignore-signatures`). Package: 0.20% duplicated at 8 lines or more (0.29% on 2026-10-01).
  Tests: 0.16% at 10 lines or more (0.91%). Scripts, demos and `cli`: 0.00% at 10 lines or more.
  Rerun at 5 lines (package) and 6 lines (tests) to group the smaller copies by file pair.
- Function names defined in more than one module, and test helpers defined in more than one file.
- Reading the twin classes side by side: the numpy and Rust attention, layer-norm, residual and
  token layers, the pure-Python parameter rows, and the two builders.

## Findings

Line numbers are from ebfcc47.

### In the package

1. **Four `_ParameterFree` classes.** `residual_array_layer.py:49`,
   `residual_rust_array_layer.py:22`, `token_rust_array_layer.py:21` and
   `token_array_layer.py:65`. The two residual ones are the same apart from type hints; the token
   Rust one adds the two "delta is the next layer's downstream" methods. The numpy token one is
   different (its single-example methods go through a batch of one, `BatchShaped`) and stays.
2. **The hidden-layer methods.** The two `compute_output_delta*` refusals and the two
   `compute_hidden_delta*` methods that take the next layer's downstream as the delta are written
   out in `AffineArrayLayer` (`residual_array_layer.py:36-46`), `AffineRustArrayLayer`
   (`affine_rust_array_layer.py:27-37`), `LayerNormRustArrayLayer`
   (`layer_norm_rust_array_layer.py:49-60`), both max-pool array layers
   (`max_pool_array_layer.py:68-79`, `max_pool_rust_array_layer.py:54-65`) and both Add layers.
   `AttentionRustArrayLayer` and the token-wise dense Rust layer have the refusals only
   (`attention_rust_array_layer.py:73-77`, `token_rust_array_layer.py:175-179`). The refusals'
   messages vary.
3. **The Add twins.** `AddArrayLayer` and `AddRustArrayLayer` have the same method bodies (`+`).
   Left as they are (D2).
4. **The attention layer's parameter methods.** `decayed`, `parameters`, `gradients` and
   `set_parameters` over Wq, bq, Wk, bk, Wv, bv, Wo, bo, and `projection_shapes`, are the same in
   `attention_array_layer.py:44-77` and `attention_rust_array_layer.py:31-63`. The order is also the
   draw, save and step order.
5. **The pure-Python parameter rows.** `WeightRow` and `PositionRow` (`token_layer.py:37`, `:153`)
   have the same `__init__`, `weights`, `set_weights` and `reset_gradient_accum`; they differ in
   their two class flags and `WeightRow.randomize`. `LayerNormFeature` (`layer_norm_layer.py:25`)
   and `BatchNormNode` (`batch_norm_layer.py:38`) have the same gamma-and-beta WeightSet surface
   (`weights` as `[gamma]`, beta as the bias, the flags, `reset_gradient_accum`), but store beta
   differently: a `bias` attribute against `beta` behind a `bias` property.
6. **The two builders.** `_tokens` is the same in `python_layer_builder.py:64` and
   `array_layer_builder.py:169`. The two `_dense_layer` functions test dropout and linear in the
   opposite order (harmless: dropout is validated to a sigmoid layer, `layer_specs.py:356`). The
   fork, add and body-first wiring loop is also the same in both; it stays (D6).

### In the tests

7. **Nine bit-comparison helpers.** `_bits` in `test_batch_norm_ghost_groups.py`,
   `test_batch_norm_rust_network.py`, `test_batch_norm_python_network.py`, `test_checkpoint.py`,
   `test_run_checkpoint.py`, `test_rust_array_layer_sgd_step.py` and `test_update_rule_forms.py`,
   and `bits` in `saved_model_fixtures.py` and `test_residual_array_network.py`. Some compare bytes
   (`struct.pack`, `.tobytes()`), some `float.hex` strings; `test_checkpoint.py`'s and
   `test_run_checkpoint.py`'s are the same function.
8. **Smaller helpers.** `_split` is the same in the three `test_attention_*_network.py` files;
   `_numpy` is in three Rust test files beside `helpers.to_numpy`; `_max_relative_gap` has three
   variants (`test_batch_norm_rust_network.py`, `test_batch_norm_conv_rust_network.py`,
   `test_residual_rust_network.py`); the `math_exp` and `crate_exp` fixtures are each written twice,
   patching a different module's exp.
9. **The new network tests, split by backend.** `test_attention_{array,rust,python}_network.py`,
   `test_residual_{array,python}_network.py` and `test_layer_norm_{array,python}_network.py` repeat
   the same network-level scenarios with only the backend changed: the finite-difference gradient
   check, the README model building its layers wired together, and learn agreeing with a learn
   batch of one example.

### Not duplication

- The per-rule presets (D1, the first pass's D2).
- The numpy and Rust twins' numerics, such as the two layer norms' `forward_batch` (D1, the first
  pass's D3).
- `paint_brush_stroke` in `digit_capture.py` and `mnist_capture.py` (each checks its own grid
  size around the shared `capture_common.stamp_brush`), `sigmoid` and `_transpose` (pure Python
  against numpy), and the demos' `main` functions.

## Decisions (settled)

The owner settled D1-D9 on 2026-10-02, one at a time.

- **D1. The first pass's scope stands.** indrajala-ml only (the crate is out); the 12 per-rule
  presets stay until more are added; the numpy and Rust twins share their identical methods only;
  the tests are in scope; scripts and demos are out (they measure 0.00%); only a change to the
  optimizers gets an A/B. No stage here touches the optimizers, so none gets one: stages 1-4 move
  methods with identical bodies, and stages 5-7 are tests. See D9 for stage 1.
- **D2. The Add twins stay as they are (finding 3).** About 25 lines; each backend keeps its own
  named, typed class. Rejected: one generic Add for both backends (loses the per-backend class the
  tests check), and a shared base under two named subclasses. They take their base from stage 1's
  module like the other parameter-free layers, but keep their own methods.
- **D3. One numpy-free module of hidden-layer mixins (findings 1 and 2).** `Hidden` (the two
  output-delta refusals, a generic message from the class name), `DeltaIsDownstream` (the two
  hidden-delta methods) and `ParameterFree` (the two no-op gradient accumulations). They replace
  the three interchangeable `_ParameterFree` classes and the copies in the classes listed in
  finding 2. The hand-written refusal messages are lost; no test matches them. Rejected: merging
  only the `_ParameterFree` classes, which leaves the methods to be copied by the next layer kind.
- **D4. An `AttentionProjections[A]` mixin in `array_parameters.py` (finding 4),** beside
  `WeightAndBias` and `GammaAndBeta`, so the eight parameters' order is written once.
- **D5. Both pure-Python pairs share (finding 5).** `PositionRow` becomes the base `WeightRow`
  extends with its bias flag and `randomize`; a `GammaAsWeights` mixin holds the gamma-and-beta
  surface, and `LayerNormFeature` stores `beta` behind a `bias` property as `BatchNormNode` does.
  Pure Python is parity-only and never benchmarked, so the property costs nothing measured.
- **D6. The builders share `_tokens` only (finding 6).** It moves to `layer_specs.py` beside
  `image_shape`, and the two `_dense_layer` functions test in the same order. The wiring loop stays
  in each builder: sharing it needs callbacks for the fork and the add, more indirection than its
  12 lines are worth. Revisit if another layer kind that links layers arrives.
- **D7. One `bits()` in `tests/helpers.py`, as `float.hex` (findings 7 and 8),** recursing through
  arrays of either backend, lists, tuples, dicts and floats: readable pytest diffs, 0.0 and -0.0
  differ, NaNs match (as the batch-norm tests already intend). The smaller helpers move to
  `helpers.py` too, the exp fixtures as one factory that patches a given module.
- **D8. The shared network scenarios are parametrized by backend (finding 9),** as the first
  pass's stage 8 did for batch norm. Layer-level tests (each backend's own API) and the parity
  tests against numpy stay in the per-backend files. Each PR lists the tests collected before and
  after, and every old test maps to a new one.

- **D9. Stage 1 gets no A/B either.** It moves `compute_hidden_delta*`, which runs in every
  backward step, onto mixins, so it changes a `learn*` path in CLAUDE.md's sense. The bodies are
  identical and only the class defining them changes; CPython caches method lookup per type, so
  the defining class costs nothing measurable. The first pass's stage 6 (`GammaAndBeta`, read by
  the optimizer every step) set the precedent. The PR says so.

## Stages

Every stage is one PR and passes `./cli lint`, `./cli test` and
`.venv/bin/python scripts/golden_training_run.py check data/refactoring/golden_run.json`
(bit-identical). No public class, module or function a demo, test or saved file uses is renamed or
deleted. No stage gets an A/B (D1, D9); each PR says so.

### Stage 1: the hidden-layer mixins (findings 1 and 2, D3)

- A numpy-free module (`model/hidden_layers.py`) holds `Hidden`, `DeltaIsDownstream` and
  `ParameterFree`.
- The residual and token Rust `_ParameterFree` classes go; their layers, and the classes in
  finding 2, take the mixins. The numpy token `_ParameterFree(BatchShaped)` stays.
- Pitfall: the Rust layers pick their fused hidden-delta ops by the next layer's concrete class
  (`rust_array_layer.py:22`, `:89`, `:102`), so the mixins must not replace a concrete class in an
  `isinstance` check; they only add bases.
- Pitfall: a method a layer overrides (the fork's lazy Rust downstream, attention's `_backward` in
  its hidden-delta methods) stays on the layer, ahead of the mixin in the MRO.

### Stage 2: the attention parameters (finding 4, D4)

`AttentionProjections[A]` in `array_parameters.py` holds `decayed`, `parameters`, `gradients` and
`set_parameters`; both attention array layers use it. The seeded-draw parity and format-2 tests
cover the order unchanged.

### Stage 3: the pure-Python parameter rows (finding 5, D5)

- `PositionRow` is the base of `WeightRow`.
- `GammaAsWeights` holds the gamma-and-beta surface for `LayerNormFeature` and `BatchNormNode`;
  `LayerNormFeature`'s `channel.bias` reads and writes (forward, gradients, snapshot, restore) go
  through the property or use `beta`.
- The golden run has no layer norm, so the pure-Python layer-norm parity tests against numpy are
  the bit-identical check for this stage.

### Stage 4: the builders (finding 6, D6)

`_tokens` moves to `layer_specs.py` (public, beside `image_shape`, as `token_shape`: the builders
already name a local `tokens`); both builders import it. The pure-Python `_dense_layer` tests linear
before dropout, as the array one does.

### Stage 5: the test helpers (findings 7 and 8, D7)

`bits`, `split`, `max_relative_gap` and an exp-patching fixture factory in `tests/helpers.py`; the
nine bit helpers and the smaller copies go, and `_numpy` becomes `helpers.to_numpy`. Every changed
test still passes, and the PR lists the files changed.

### Stage 6: the attention network tests (finding 9, D8)

`test_attention_network.py` holds the finite-difference check, the README model's wiring (its
expected classes per backend) and learn against a batch of one, parametrized by backend: numpy
within rounding, Rust and pure Python by bits, Rust's dropout frozen. The per-backend files keep
their layer-level and parity tests.

### Stage 7: the residual and layer-norm network tests (finding 9, D8)

`test_residual_network.py` and `test_layer_norm_network.py`, the same way.
`test_residual_rust_network.py` is all parity tests and stays as it is.

## After this plan

- Retire this workplan once every stage is merged: anything left open moves to
  [next-steps.md](next-steps.md).
- The wiring loop in the builders (D6), if another layer kind that links layers arrives.
- The per-rule presets, when more are added (the first pass's D2).
- Rerun the audit after the next primitive.

## Out of scope

- The crate, scripts and demos, the per-rule presets, the twins' numerics (D1).
- The Add twins (D2) and the builders' wiring loop (D6).
- Renaming or deleting public classes (the composable-layers workplan's D1).
- Any change to a numeric result, a saved file or the golden run.
