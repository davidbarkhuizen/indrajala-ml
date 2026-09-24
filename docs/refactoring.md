# Refactoring: work plans

Structural duplication still in the code, with a staged plan for each item, largest payoff
first. Each plan changes structure only, never numerics: every stage must keep every parity test
passing and leave training bit-identical. When an item is done it is removed from this page, and
its measurements go in its PR.

Rules for every stage:

- **Bit-identical training.** Before the first stage of an item, record a golden run on `main`
  with a scratch probe: every affected network, built from fixed injected weights (never
  `randomize()`: `pa.uniform` can't be seeded), trained for a few `learn`, `learn_row`,
  `learn_batch` and `learn_batch_rows` steps, with the snapshot and `classify_rows` output saved.
  Each stage must reproduce that run exactly, not within a tolerance.
- **No hot-path slowdown.** A stage that touches `learn*` or `classify_rows` is timed before and
  after, numpy and Rust in separate processes (`scripts/prepared_dataset_timing.py time`), with
  both builds committed first. It must be within run-to-run noise.
- **Public names stay.** Demos, `demos/registry.py`, `ensemble_train.py` and the tests construct
  the concrete classes by name, and saved model files must still load.

## 1. One array network base for both backends

**The duplication.** `ArrayNetworkBase` and `RustArrayNetworkBase` are the same class body:
`__init__`, `_forward_input`, `_learn_batch_input`, the abstract hooks, `snapshot` and almost all
of `_learn_input` match line for line. The same holds for the classes one level up:

- the multiclass pair `VectorizedMultiClassBackpropClassifierNetwork` /
  `RustArrayMultiClassBackpropClassifierNetwork`;
- the single-output pair `ArrayBackpropClassifierNetwork` / `RustArrayBackpropClassifierNetwork`;
- the ensemble pair `EnsembleArrayBackpropClassifierNetwork` /
  `EnsembleRustArrayBackpropClassifierNetwork`;
- the conv pair's `randomize`, `snapshot` and `restore`.

`rust_array_network_base.py` says they aren't shared because "the two backends' array APIs
differ throughout". That is out of date: the differences come down to these operations.

| operation | numpy | Rust |
|---|---|---|
| tuple to vector | `np.array(state, dtype=np.float64)` | `pa.Array(list(state))` |
| batch of tuples to matrix | `np.array([...], dtype=np.float64)` | `pa.Array([list(s) ...])` |
| one row of a prepared matrix | `states[i]` | `states.row(i)` |
| rows by index / by range | `states[list(idx)]`, a slice | `states.take_rows(list(...))` |
| dataset backend name | `"numpy"` | `"rust"` |
| fan-in-aware draw | `fan_in_aware_random_layer` | `fan_in_aware_random_rust_layer` |
| list or array to owned array | `np.array(x, dtype=np.float64).copy()` | `x.copy()` or `pa.Array(x)` |
| zeros, argmax, argmax per row | `np.zeros`, `np.argmax`, `axis=1` | `pa.Array.zeros`, `pa.argmax`, per-row `index(max)` |
| output to list | `float(x[0])`, `.tolist()` | `.tolist()` |

There is one behavioral difference. For single examples, Rust calls the fused `layer.sgd_step`,
while numpy calls `accumulate_gradient` and then `apply_accumulated_gradient`.

**Target shape.** A small backend object (`indrajala_ml/model/array_backend.py`: `NUMPY` and
`RUST`, each a namespace of the operations above) and one `ArrayNetworkBase` that calls it through
`self.backend`. Numpy's `ArrayLayer` gains an `sgd_step` that makes the same two calls, which
keeps it bit-identical, so the shared `_learn_input` always calls `sgd_step`. The shape classes
become one multiclass class and one single-output class, parameterized by backend. The existing
concrete names stay as two-line subclasses that set `backend` and the default layer classes.

**Stages** (one PR each):

0. The golden-run probe and the timing baseline, both recorded on `main`.
1. `array_backend.py` and `ArrayLayer.sgd_step`. `ArrayNetworkBase` takes over
   `RustArrayNetworkBase`'s body through `self.backend`, and `RustArrayNetworkBase` becomes a
   subclass that sets `backend = RUST`. Rust's single-output `restore`, which also accepts plain
   lists, becomes the shared behavior through the backend's "to owned array". Measure the
   hot-path timing.
2. The multiclass and single-output shape pairs, with `save` and `load` converting through the
   backend.
3. The ensemble pair, and the conv pair's `randomize`, `snapshot` and `restore`. Only the conv
   pair's `classify_rows` stays per backend, because batched Rust conv inference was measured
   slower (`docs/optimizations/rejected.md`).
4. Update the docstrings that describe the two bases as separate, starting with
   `RustArrayNetworkBase`'s.

**Risk.** The hot path gains one attribute lookup per call (`self.backend.row`, and so on). If
stage 1 measures a slowdown, bind the backend's functions as class attributes when the class is
created instead.

## 2. Hyperparameters declared once

**The duplication.** Every network with a hyperparameter spells each name out several times:

- in `__init__`, which stores it and also closes over it in a layer-class lambda;
- in a `randomized` override that exists only to pass it through (22 files define `randomized`);
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
