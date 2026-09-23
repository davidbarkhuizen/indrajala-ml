# Workplan: optimization 5, build the training data as one backend array

Order: last (see `optimizations.md`). Lowest value, and the only item that changes the trainer
interface. It should be re-justified by measurement after items 1–4 have landed, because the
fixed per-example cost it removes becomes a larger share only once the maths is faster.

## Context

Both backends convert Python tuples to a backend array on every call:

- `RustArrayNetworkBase._forward`/`learn`: `pa.Array(list(state))`;
- `RustArrayNetworkBase.learn_batch`: `pa.Array([list(state) for state, _ in batch])`;
- the numpy `ArrayNetworkBase` equivalents with `np.array`.

One 784-pixel MNIST row costs 49 µs in Rust and 55 µs in numpy. Both pay it, so the ratio
doesn't change, but neither backend's maths can remove it. The trainers (`train.py`:
`train_linear_classifier_network`, `train_backprop_network_mini_batch`, `_training_accuracy`)
take `list[tuple[state, label]]`, shuffle it in Python, and pass tuples through. The same
contract is used by the pure-Python networks, the ensembles, and the sweeps.

## Stage 0: decide (measurement, no code PR)

Before any interface work, measure on the current `main`, after items 1–4:

1. The conversion's share of one epoch, single-example and mini-batch (32), for the dense
   production network (784 -> 30 -> 10) and the conv networks, in both backends. Use cProfile
   own time of `Array.__new__`/`np.array`, plus a direct A/B: the same epoch with pre-converted
   inputs passed through a test-only path.
2. Where the per-epoch accuracy passes (`_training_accuracy`) spend time. They convert every
   training row again on each epoch.

**Go/no-go:** proceed only if the conversion is at least about 10% of epoch time for some
production configuration. Otherwise record the numbers in `recommended-optimizations.md` item
5 and close. The criterion is fixed here, before measuring, so the decision isn't argued after
the fact.

## Design, if it goes ahead

The goal is to keep the existing interface working and add a fast path, not to replace it.

- **A `PreparedDataset` built once per training run** (new module, for example
  `indrajala_ml/prepared_dataset.py`). It holds the states as one backend matrix `(rows,
  dimension)`, the labels as a Python list, and the backend it belongs to. A network exposes
  `prepare_dataset(rows) -> PreparedDataset` and knows its own backend. The pure-Python
  networks don't implement it.
- **Row access without conversion:**
  - numpy: a row slice is a view, so it's free.
  - Rust: `RustArray` slicing copies (a `(slice, slice)` index returns a new array). That's still
    a memcpy of 784 doubles, far cheaper than parsing 784 Python floats, but it needs measuring. A
    `gather_rows(matrix, indices) -> matrix` crate op would build a shuffled mini-batch in one
    call. It's the only crate change this plan expects.
- **Network entry points:** `learn_row(lr, dataset, i)` and `learn_batch_rows(lr, dataset,
  indices)` on the two array network bases. They share their bodies with `learn`/`learn_batch`,
  which become thin wrappers that convert and then call the shared body. So the two paths can't
  drift, and `learn`/`learn_batch` stay bit-identical.
- **Trainers:** `train_linear_classifier_network` and `train_backprop_network_mini_batch` detect
  a student with `prepare_dataset` and use it. They shuffle an index list in place of the tuple
  list. **The shuffle must consume the RNG exactly as it does now.** `shuffle(indices)` on a
  list of the same length makes the same permutation as `shuffle(rows)`, so the seeded
  end-to-end pins (0.9875 / epoch 10 / 0.925 and the rest) stay exact. A test asserts that
  directly: the same seed gives the same visiting order under both paths.
  `_training_accuracy` uses the prepared matrix via a batched `predict` where one exists.
- **Unchanged:** the pure-Python networks, the `(state, label)` list contract everywhere it's
  used outside the two trainers, the ensembles, and the sweeps.

## Stages, if it goes ahead

1. **Crate PR (only if the Rust row copy measures material):** `gather_rows`, tested for exact
   equality against a Python loop of row slices.
2. **PR here:** `PreparedDataset`, `prepare_dataset`, and the shared `learn`/`learn_row` bodies
   for both array bases (dense and conv networks inherit them). Tests: `learn_row` on a
   prepared dataset produces exactly the same weights as `learn` on the tuple, step by step,
   for every numpy and Rust network class. This is enumerated from a registry or parametrized
   list, so a new class can't be silently missed.
3. **PR here:** the trainers' fast path and the RNG-equivalence test. All existing end-to-end
   pins must pass unchanged, with no tolerance change. Anything else means the visiting order
   changed.
4. Measure the end-to-end demos (conv and dense), both backends, and record the before/after in
   the PR.

## Out of scope

- Changing `load_*` dataset functions to return arrays directly. `PreparedDataset` is built from
  their current output, so the data modules don't change.
- Any change to what the trainers compute (early stopping, LR schedules, accuracy definitions).
