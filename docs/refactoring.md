# Refactoring: work plans

Structural duplication still in the code, with a staged plan for each item. Each plan changes structure only, never numerics: every stage must keep every parity test
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

## 1. Test files parameterized by backend

**The duplication.** 7 numpy/Rust pairs of test files (`test_*_array_layer.py` /
`test_*_rust_array_layer.py`, `test_*_vectorized_multiclass_backprop_model.py` /
`test_*_rust_array_multiclass_backprop_model.py`, and so on) differ in:

- the class under test;
- `np.array` versus `pa.Array`;
- `np.allclose` versus `pytest.approx`;
- a few genuinely backend-specific tests, such as the Rust fused-op checks and the numpy
  overflow sweep.

**Target shape.** The template is `tests/test_array_multiclass_backprop_model.py`, which
replaced the plain dense pair:

- One file per feature, named without the backend (`test_array_*` or `test_<feature>_array_*`).
- The `backend` fixture (`tests/conftest.py`) runs a test once per backend, as `test_x[numpy]`
  and `test_x[rust]`. It is the production backend object (`array_backend.py`): `backend.name`
  picks the class from the file's `{"numpy": ..., "rust": ...}` map, and `backend.owned` wraps
  nested lists in that backend's arrays.
- Where the pair's tolerances differ, the merged test keeps the stricter one for both backends:
  `pytest.approx(expected, rel, abs)` (on `.tolist()` for arrays) bounds the error by the larger
  of the two tolerances, where `np.allclose` allows their sum.
- Backend-specific tests stay in the merged file, run on their backend only.
- Comments and docstrings are trimmed as the files are merged.

**Stages:** one PR per two to four features. Each PR compares the collected test list before and
after (`pytest --collect-only -q`): every old test maps to a parameterized one, and the count
doesn't drop.
