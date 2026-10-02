# Workplan: the source layout

**Status: decisions settled (D1-D5); stages 0-5 done. Before stage 6, pytest collects 9768 tests.**

`indrajala_ml/` has grown flat. `indrajala_ml/model/` holds 190 modules in one directory: 61
one-change presets, 50-odd layers across three implementations, the network bases, optimizers,
protocols, specs and persistence. The package root mixes datasets, trainers, capture, the
benchmark tooling and two utilities. A handful of modules run past 300 lines with several
unrelated parts.

This plan groups the modules into subpackages by what they are, and splits the large modules
whose parts stand alone. It moves code only. No behavior, op, rule or crate changes; a moved
function keeps its name, body and signature. Every stage passes the golden run bit-identical and
`./cli test` / `./cli lint`, and changes no `learn*` or `classify_rows` path's code, so it needs no
A/B (docs/measurement.md, §1).

## Constraints the layout must respect

- **Saved files name classes, not modules.** Format 2 stores `type(network).__name__` and the
  legacy files are read by their class's `load()`, so moving a class breaks no file. Renaming a
  class would; this plan renames no class.
- **The tests discover classes by walking `indrajala_ml.model`.** Four test modules and
  `tests/saved_model_fixtures.py` use `pkgutil.iter_modules(indrajala_ml.model.__path__)`, which
  doesn't recurse: once the modules move into subpackages they would find nothing and pass
  vacuously. Stage 1 replaces this with one recursive helper and a test that it finds the classes
  it found before.
- **`scripts/ab.py` imports `indrajala_ml.train` on both sides of an A/B** (its provenance probe),
  and `--script-from new` runs the new tree's driver against the old tree. An A/B across the move
  would fail on import (D4).
- **Pure-Python networks mustn't import numpy or the crate** (`load_network.py`'s docstring). A
  subpackage's modules keep their own imports; no `__init__.py` re-exports a subpackage's names, so
  importing one module never imports its siblings. The packages stay namespace packages, as now.
- **Imports stay absolute and name the module** (`from indrajala_ml.model.layers.rust.conv_rust_array_layer
  import ConvRustArrayLayer`). Module file names keep their backend suffixes, so a name stays
  unique across the tree and grep finds it.

## Proposed layout

`model/` by role, then by implementation (D1); presets beside their bases (D2):

```
indrajala_ml/
  model/
    protocols/      classifier_protocols, layer_protocols, array_protocols, classification
    specs/          layer_specs (split, below), hidden_layers, update_rules, window_geometry,
                    bounds, array_network_shapes
    layers/
      python/       base_node, state_node, state_layer, association_node, association_layer,
                    backprop_node, backprop_layer, relu_layer, dropout_layer, batch_norm_layer,
                    layer_norm_layer, attention_layer, token_layer, residual_layer, conv_layer,
                    conv_kernel, conv_unit, conv_front_end, linear_conv_layer, linear_layer,
                    max_pool_layer, cross_entropy_output_layer, softmax_output_layer,
                    fan_in_aware_init, layer_major, python_layer_builder
      array/        array_backend, array_parameters, array_layer_builder (shared by numpy and Rust)
      numpy/        array_layer, relu_array_layer, dropout_array_layer, batch_norm_array_layer,
                    layer_norm_array_layer, attention_array_layer, token_array_layer,
                    residual_array_layer, conv_array_layer, max_pool_array_layer,
                    linear_array_layer, softmax_array_layer, cross_entropy_array_layer
      rust/         rust_array_layer and its 12 *_rust_array_layer siblings, affine_rust_array_layer
    optimizers/     optimizer_base, python_optimizer, optimizers (split, below)
    networks/
      python/       backprop_network_base, backprop_classifier_network,
                    multiclass_backprop_classifier_network, conv_multiclass_backprop_classifier_network,
                    sequential_backprop_network, linear_classifier_network, + its 21 presets
      numpy/        numpy_array_network_base, array_backprop_classifier_network,
                    vectorized_multiclass_backprop_classifier_network,
                    conv_vectorized_multiclass_backprop_classifier_network, + its 20 presets
      rust/         rust_array_network_base, rust_array_backprop_classifier_network,
                    rust_array_multiclass_backprop_classifier_network,
                    conv_rust_array_multiclass_backprop_classifier_network, + its 20 presets
      array_network_base, sequential_array_network (generic over both array backends)
    ensembles/      ensemble_base, array_ensemble_base, the three ensemble networks
    persistence/    format2 (split, below), format2_persistence, load_network, model_io, checkpoint
  data/             mnist_data, digits_data, iris_data, dataset_utils, prepared_dataset, targets
  training/         train (split, below), ensemble_train (split, below), evaluate,
                    multiclass_evaluate, lr_schedule, run_checkpoint
  capture/          capture_common, digit_capture, mnist_capture
  measurement/      machine_profile (split, below) + its schema, benchmark_sweep, benchmark_data
  studies/          batch_size_scaling
  graphics/         chart (split, below)
  demos/            by topic, in the menu's order (D3): linear/, backprop/, uci_digits/,
                    mnist/, conv/, timing/; menu, registry, capture_app and timing.py stay
  pcg64.py, geometry.py   stay at the root: used everywhere, one module each
```

Every module above is placed; the stage that moves it checks that `find indrajala_ml -name '*.py'`
lists the same module names before and after, apart from the splits.

## Tests and demos (D3)

`tests/` mirrors the source tree: a module's tests sit at its path (`indrajala_ml/model/layers/rust/`
→ `tests/model/layers/rust/`), and the scripts' tests in `tests/scripts/`. The rules for the tests
that don't map to one module:

- **A test over several implementations goes to their common parent.** Most array tests take the
  `backend` fixture and run on numpy and Rust together (`test_relu_array_layer.py` tests both
  ReLU layers), so they go to `tests/model/layers/` or `tests/model/networks/`, not under `numpy/`
  or `rust/`. A pure-Python-only test goes under `python/`.
- **A test over every network or every layer** (`test_seeded_init_parity.py`,
  `test_fused_layer_ops.py` and its per-rule siblings, `test_summation_order.py`,
  `test_numerical_parity.py`) goes to the lowest folder that holds everything it walks.
- **A pipeline test** (`test_*_training_pipeline.py`, `test_train*.py`) goes with the trainer it
  drives, in `tests/training/`.
- **The shared modules stay at the root**: `conftest.py`, `helpers.py`, `array_network_contract.py`,
  `saved_model_fixtures.py`, `gradient_check.py` and `fixtures/`. Their imports don't change.
- **Test file names stay unique across the tree.** pytest's default import mode needs unique
  basenames without `__init__.py` files; a stage checks it (`find tests -name 'test_*.py'` has no
  duplicate basename), and pytest collects the same number of tests before and after each move.

Each test moves in the stage that moves its module, so a stage leaves the two trees in step.

`demos/` follows the menu's sections; a demo that fits two topics goes where the menu lists it
(`demo_rust_vs_vectorized_mnist_recognition` is in the MNIST section, so `mnist/`). The registry's
`module` strings and `tests/demos/` move with the demos; the menu numbers don't change.

## Splits (D5)

A module is split when it is past about 300 lines **and** its parts stand alone (a reader of one
needn't read the other). One long class is not split. Proposed:

| Module | Lines | Into |
| --- | --- | --- |
| `model/layer_specs.py` | 531 | `layer_specs` (the spec dataclasses, `expand_specs`, `spec_paths`), `spec_shapes` (`SpecShape`, `spec_shapes`, `image_shape`, `token_shape`), `spec_validation` (`validate_layer_specs` and its checks), `single_example` (the refusals, `ghost_groups`, `batch_norm_index`) |
| `ensemble_train.py` | 521 | `balanced_dataset` (index selection, building the balanced binary sets), `worker_sizing` (memory, bytes per example, worker count), `ensemble_train` (training one classifier, the parallel and serial trainers) |
| `model/format2.py` | 518 | `format2_json` (layer, rule, input, optimizer-state and RNG codecs), `format2` (network and checkpoint to JSON, `NetworkFile`, the kind checks, restore, ensembles) |
| `machine_profile.py` | 466 | `machine_profile_capture` (the parsers and `capture`), `machine_profile` (schema, validate, compare, `main`) |
| `train.py` | 389 | `training_data` (the generated training sets), `training_diagnostics` (`TrainingDiagnostic`, `ConvergenceSeries`), `train` (the trainers) |
| `model/optimizers.py` | 313 | `array_optimizer_base` (`momentum_update`, `ArrayOptimizerBase`), `numpy_optimizer`, `rust_optimizer` |
| `graphics/chart.py` | 301 | `classifier_plots` (the network, training-data and heatmap plots), `evaluation_plots` (confusion matrix, sample predictions), `chart` (figures, axes, legends, series, window placement) |

A split module's test file splits along the same seams where it has matching sections
(`test_layer_specs.py`, `test_format2.py`, `test_machine_profile.py`, `test_chart.py`); the new
test files take the new modules' names.

Done in stage 2: `format2_json`'s codecs that `format2` calls lost their leading underscore
(`lists`, `input_to_json`, `optimizer_state_to_json`, `rng_to_json` and their readers), the one
rename a split needs. `test_format2.py`'s two layer-entry tests became `test_format2_json.py`.
`test_layer_specs.py` stays whole: its sections are by feature (batch norm, residual blocks, patch
models), each mixing validation, the shape walk and the builders, so none matches one new module.

Done in stage 5: `worker_sizing`'s three functions that `ensemble_train` calls lost their leading
underscore (`available_memory_bytes`, `estimate_bytes_per_example`, `select_worker_count`).
`test_train.py`, `test_ensemble_train.py` and `test_machine_profile.py` split along the same seams
into `test_training_data.py`, `test_balanced_dataset.py`, `test_worker_sizing.py` and
`test_machine_profile_capture.py` (the collectors' tests, with their fixtures). `SCHEMA_VERSION`
went with `capture`, which writes it; the schema file moved beside `machine_profile`.

Not split: `array_network_base.py` (330, one class), `batch_norm_layer.py` (307, one layer's
nodes and layer), `demos/registry.py` (321, one table), `demos/demo_layer_op_timing.py` (338, one
demo), `batch_size_scaling.py` (308, one study).

## Decisions

- **D1. How `model/` is grouped.** Proposed: by role (layers, networks, optimizers, ...), then by
  implementation. Alternatives: by implementation first (`model/python/`, `model/numpy/`,
  `model/rust/`, each with layers and networks), or by layer kind (`layers/conv/` with all three
  implementations of a kind). **Decided: by role, then by implementation** (owner, 2026-10-02):
  one rule for all of `model/`; the pure-Python boundary stays visible, and the code generic over
  both array backends has a home. The matching file names keep a layer kind's three
  implementations easy to find.
- **D2. Where the 61 presets go.** Proposed: beside their base, in `networks/<implementation>/`.
  Alternative: their own `presets/<implementation>/` (or `presets/<change>/`). **Decided: beside
  their base** (owner, 2026-10-02): a preset is its base plus one change and is read with it; the
  `<change>_` prefix sets the presets apart from the bases in a listing.
- **D3. `demos/` and `tests/`.** Proposed: leave both flat. Alternatives: subfolders for `demos/`
  only, or for both with `tests/` mirroring the source tree. **Decided: both, `tests/` mirroring
  the source** (owner, 2026-10-02). See Tests and demos for the rules.
- **D4. Compatibility across the move.** Proposed: no shims at the old paths; `ab.py`'s probe
  accepts `indrajala_ml.train` or `indrajala_ml.training.train`, and an A/B across the move runs
  each side's own driver (`--script-from` left at its default). Alternatives: shim modules at the
  old paths, or no `ab.py` change. **Decided: no shims, `ab.py` accepts both layouts** (owner,
  2026-10-02): nothing outside this repository imports `indrajala_ml`, and the A/B harness is the
  only reader of the old paths that matters. `--script-from new` against a tree from before the
  move is not supported.
- **D5. Which modules are split.** Proposed: the seven in Splits. Alternatives: only the four
  past 450 lines, or a 200-line bar (about 20 more). **Decided: the seven** (owner, 2026-10-02):
  each has a clean seam, and the lower bar would split single classes.

## Stages

Each stage is one PR: `git mv` (so history follows), the moved modules' tests (D3), every import,
`pyproject.toml`'s paths (per-file ignores, package data), and the README's and docs' file
references updated together. pytest collects the same number of tests before and after.

0. This workplan.
1. **Discovery and the harness.** A recursive `model_modules()` helper in `tests/helpers.py`
   replaces the five `iter_modules` walks, with a test that it finds the same classes; `ab.py`'s
   probe made path-agnostic (D4). Nothing moves.
2. **`model/`: protocols, specs, optimizers, ensembles, persistence**, with the `layer_specs`,
   `format2` and `optimizers` splits.
3. **`model/layers/`.** Done: the layer tests went to `tests/model/layers/` (numpy and Rust
   together, the fused-op tests and `test_summation_order.py` among them), `python/` and `rust/`;
   `test_rust_array_layer_sgd_step.py`, which steps layers and networks, to `tests/model/`.
   `test_batch_norm_ghost_groups.py` waits for stage 4 with the other batch-norm network tests.
   The crate's comments that cite the layer modules follow in a crate PR and a "Bump rust/".
4. **`model/networks/`** and the presets. Done: the network tests that run on one implementation
   only, all pure Python (`test_model.py` and `test_learn_batch.py` among them), went to
   `tests/model/networks/python/`; the rest, which take the `backend` fixture or check pure Python
   against numpy (the `*_python_network` tests), to `tests/model/networks/`, with
   `test_batch_norm_ghost_groups.py` and `test_seeded_init_parity.py`. No network test runs on one
   array backend alone, so `tests/model/networks/` has no `numpy/` or `rust/`. The crate cites no
   network module, so no crate PR. `test_numerical_parity.py` and `test_numpy_rng_streams.py` test
   the crate and numpy's streams, not a module here; they stay at the root until stage 6 places them.
5. **The package root:** `data/`, `training/`, `capture/`, `measurement/`, `studies/`, with the
   `train`, `ensemble_train` and `machine_profile` splits. Done: each module's tests went to its
   folder under `tests/` (the pipeline tests to `tests/training/`). The paths built from
   `__file__` (`machine_profile_capture`'s repository root, two tests') go up one more level.
   `test_patch_attention_study.py` and `test_residual_depth_study.py` test scripts, not
   `studies/`, so they wait for `tests/scripts/` in stage 6.
   `ab.py`'s probe now picks the trainer by which file the side's tree has: trying
   `indrajala_ml.training.train` first found the editable install's once `training/` existed,
   since `indrajala_ml` is a namespace package.
6. **`graphics/`** (the `chart` split) and **`demos/`** with their tests; `tests/scripts/` for the
   scripts' tests; then retire this plan.

## After this plan

Nothing planned.
