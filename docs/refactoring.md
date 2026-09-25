# Refactoring: work plans

Structural duplication still in the code, with a staged plan for each item. Each plan changes
structure only, never numerics: every stage must keep every parity test passing and leave
training bit-identical. When an item is done it is removed from this page, and its measurements
go in its PR.

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

**No open items.**
