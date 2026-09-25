# Refactoring: work plans

Structural duplication still in the code, with a staged plan for each item, largest payoff
first. Each plan changes structure only, never numerics: every stage must keep every parity test
passing and leave training bit-identical. When an item is done it is removed from this page, and
its measurements go in its PR.

Rules for every stage:

- **Bit-identical training.** Before the first stage of an item, record a golden run on `main`
  with `python scripts/golden_training_run.py record data/refactoring/golden_run.json` (ignored
  by git, since numpy's BLAS makes the file valid only on the machine that recorded it). It
  trains every array network, numpy and Rust, from fixed injected weights (never `randomize()`:
  `pa.uniform` can't be seeded) through `learn`, `learn_row`, `learn_batch` and
  `learn_batch_rows`, and records every snapshot, prediction and save/load round trip as
  `float.hex`. Each stage must pass `... check data/refactoring/golden_run.json`: the same bits,
  not within a tolerance. It catches a 1-ULP change to the learning rate.
- **No hot-path slowdown.** A stage that touches `learn*` or `classify_rows` is timed before and
  after, numpy and Rust in separate processes (`scripts/prepared_dataset_timing.py time`), with
  both builds committed first. It must be within run-to-run noise.
- **Public names stay.** Demos, `demos/registry.py`, `ensemble_train.py` and the tests construct
  the concrete classes by name, and saved model files must still load.

## 1. One array network base for both backends

**Where it stands.** The base and the two shapes are shared. `ArrayNetworkBase` calls the array
operations that differ between numpy and Rust through `self.backend`
(`indrajala_ml/model/array_backend.py`, `NUMPY` and `RUST`), and `RustArrayNetworkBase` is a
subclass that sets `backend = RUST` and the Rust layer classes. The single-example step always
calls `layer.sgd_step`: numpy's layers make the unfused `accumulate_gradient` and
`apply_accumulated_gradient` calls, and Rust's plain dense layer fuses them. The multiclass and
single-output shapes are mixins (`array_network_shapes.py`), so each concrete name is the shape
over one backend's base: `VectorizedMultiClassBackpropClassifierNetwork(ArrayMultiClassShape,
ArrayNetworkBase)`, `RustArrayMultiClassBackpropClassifierNetwork(ArrayMultiClassShape,
RustArrayNetworkBase)`, and likewise `ArraySingleOutputShape` for the single-output pair. The
conv pair is `ArrayConvShape` over each backend's plain multiclass network, differing only in
its conv and pool layer classes and Rust's row-by-row `classify_rows` (batched Rust conv
inference was measured slower, `docs/optimizations/rejected.md`). The ensemble pair is
`ArrayEnsembleBase` (`array_ensemble_base.py`), differing only in `classifier_cls`.

**Stages** (one PR each):

4. Update the docstrings that still describe the numpy and Rust networks as separate.

## 2. Hyperparameters declared once

**The duplication.** Every network with a hyperparameter spells each name out several times:

- in `__init__`, which stores it and also closes over it in a layer-class lambda;
- in a `randomized` override that exists only to pass it through (19 files define `randomized`);
- in `_extra_state` and `_extra_init_kwargs`, which round-trip it through `save` and `load`.

This covers momentum, L2, Adam and dropout, in the numpy, Rust and per-node families.

**Target shape.**

- `randomized` is defined once per base, as `cls(*args, **kwargs)` followed by `randomize()`.
  That covers the conv networks and the per-node `BackpropNetworkBase` family too.
- A class attribute `hyperparameters = ("beta1", "beta2", "epsilon")` lets the base class
  generate `_extra_state` and `_extra_init_kwargs`.
- A class method `layer_cls_for(**hyperparameters)` replaces each lambda assignment.
- Each sibling is left with its `__init__` signature (its defaults and required arguments are
  documented API) and its docstring.

**Stages:**

1. The shared `randomized`, deleting every override whose body is only a pass-through. Callers
   pass hyperparameters positionally in places, so the shared version forwards `*args` and
   `**kwargs` unchanged.
2. `hyperparameters` generating `_extra_state` and `_extra_init_kwargs`, on the array families.
   A saved-file round-trip test per sibling already exists and gates this stage.

Doing item 1 first halves this item's file count if the sibling pairs are also merged; the two
items don't otherwise depend on each other.

## 3. Test files parameterized by backend

**The duplication.** 21 numpy/Rust pairs of test files (`test_*_array_layer.py` /
`test_*_rust_array_layer.py`, `test_*_vectorized_multiclass_backprop_model.py` /
`test_*_rust_array_multiclass_backprop_model.py`, and so on) differ in:

- the class under test;
- `np.array` versus `pa.Array`;
- `np.allclose` versus `pytest.approx`;
- a few genuinely backend-specific tests, such as the Rust fused-op checks and the numpy
  overflow sweep.

The pairs total about 6,500 lines.

**Target shape.**

- One file per feature, with a `backend` fixture parameterized over `numpy` and `rust`.
- The fixture supplies the network or layer class, `wrap`, and an `assert_close(actual,
  expected, rtol, atol)` that converts through `.tolist()`.
- Backend-specific tests stay in the merged file, marked by backend.
- Test IDs change, from `test_x` to `test_x[numpy]` / `test_x[rust]`; the test count must not
  drop.

**Stages:** one PR per two to four features, starting with the plain dense pair
(`test_vectorized_multiclass_backprop_model.py` / `test_rust_array_multiclass_backprop_model.py`)
as the template. Each PR compares the collected test list before and after
(`pytest --collect-only -q`): every old test maps to a parameterized one.

Doing item 1 first makes the network-level merges simpler: one class parameterized by backend
means the fixture supplies a backend, not a class.
