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

## 1. Hyperparameters declared once

**The duplication.** Every network with a hyperparameter spells each name out several times:

- in `__init__`, which stores it and also closes over it in a layer-class lambda;
- in `_extra_state` and `_extra_init_kwargs`, which round-trip it through `save` and `load`.

This covers momentum, L2, Adam and dropout, in the numpy, Rust and per-node families.

`randomized` is already shared: `ArrayNetworkBase` and `BackpropNetworkBase` each define it once,
forwarding `*args` and `**kwargs` to the constructor.

**Target shape.**

- A class attribute `hyperparameters = ("beta1", "beta2", "epsilon")` lets the base class
  generate `_extra_state` and `_extra_init_kwargs`.
- A class method `layer_cls_for(**hyperparameters)` replaces each lambda assignment.
- Each sibling is left with its `__init__` signature (its defaults and required arguments are
  documented API) and its docstring.

**Stages:**

1. `hyperparameters` generating `_extra_state` and `_extra_init_kwargs`, on the array families.
   A saved-file round-trip test per sibling already exists and gates this stage.
2. `layer_cls_for` replacing the lambda assignments in `__init__`.

## 2. Test files parameterized by backend

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

The numpy and Rust networks already share `ArrayNetworkBase`, so the fixture's `wrap` can be the
backend object's `vector`/`matrix` (`indrajala_ml/model/array_backend.py`).
