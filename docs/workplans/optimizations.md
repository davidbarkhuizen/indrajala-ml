# Workplans: recommended optimizations

`docs/recommended-optimizations.md` lists five optimizations found by the Rust CNN stage 4
timing. Each has its own workplan:

| # | workplan | value | numerics |
| --- | --- | --- | --- |
| 1 | [`optimization-1-dense-transposes.md`](optimization-1-dense-transposes.md) | high | stage A bit-identical; B and C change bits |
| 2 | [`optimization-2-dense-accumulate-gradient.md`](optimization-2-dense-accumulate-gradient.md) | high | bit-identical |
| 3 | [`optimization-3-conv-forward.md`](optimization-3-conv-forward.md) | measure first | A and B bit-identical; C to be decided by measurement |
| 4 | [`optimization-4-single-example-conv-path.md`](optimization-4-single-example-conv-path.md) | low to medium | bit-identical |
| 5 | [`optimization-5-dataset-as-array.md`](optimization-5-dataset-as-array.md) | low | bit-identical |

## Implementation order

0. **Shared stage 0: a committed per-op benchmark** (below). The per-op table in
   `recommended-optimizations.md` came from ad hoc scripts that were never committed. Every plan
   here needs before/after numbers for the same ops, so the harness comes first.
1. **Optimization 2** (dense accumulate_gradient). It is the most expensive Rust op in the
   measured case (639 µs), and it can be done bit-identically, so nothing downstream moves.
   Doing a bit-identical change first also means the later bit-changing changes are measured
   against a baseline that has only one variable left.
2. **Optimization 1** (dense transposes). Stage A (`delta_batch.T @ X`) is bit-identical and
   goes first. Stages B (`layer_downstream`) and C (`X @ W.T`) change summation order, so each
   gets its own PR with the parity protocol below.
3. **Optimization 3** (conv forward), stages A and B. Dropping `Z` and adding a forward-only
   path are both bit-identical.
4. **Optimization 4** (single-example conv path). It comes after 3A because dropping `Z`
   removes one of the reshapes 4 would otherwise have to handle.
5. **Optimization 3**, stage C (the `O`-small formulation). It is a measurement question, and
   it touches the same function as 3A/3B, so it is done last among the Rust changes, on a
   settled `conv_forward_batch`.
6. **Optimization 5** (dataset as one array). It is the only one that changes the trainer
   interface, both backends pay it equally, and it is the lowest-value item. Do it last or not
   at all.

Items 1 and 2 also speed up the dense production networks, not just the conv networks. Item 2
before item 1 is a risk ordering, not a value ordering: both are high value.

## Shared stage 0: per-op benchmark harness

A PR here, no crate change.

- `indrajala_ml/demos/bench_layer_ops.py` (or a `benchmarks/` script if the demos registry is the
  wrong home; decide from how `demos/timing.py` is used). It times each single-example and
  batch layer method, for both backends, at the shapes in `recommended-optimizations.md`: a
  `ConvSpec(3, 8)` layer on 28x28 input, the 32 x 5408 dense layer after it, and the dense
  production shape (784 -> 30 -> 10). It also covers the 8x8 UCI conv shape used in item 4.
- Method: 300 calls per loop, median of 5 loops, µs per call, numpy and Rust interleaved. That
  is the method the existing table used, so the first run should reproduce it. If it doesn't,
  that's a finding to record before any optimization starts.
- Output: a plain table, one row per op, numpy and Rust columns, and the Rust/numpy ratio.
- Numpy and Rust only. Pure Python is never timed.
- A smoke test that runs it with a tiny loop count, as `test_demo_conv_rust_vs_vectorized_digit_
  recognition.py` does for the conv demo.

Record the baseline table in the stage 0 PR. Every later PR quotes the before/after rows for
the ops it touches, plus the end-to-end ratios from
`python -m indrajala_ml.demos.demo_conv_rust_vs_vectorized_digit_recognition`. It also quotes
`demo_rust_vs_vectorized_mnist_recognition` when a dense op changes. Runs longer than a couple
of minutes go in the background, with an ETA.

## Shared rules

- **Two repos.** Crate changes land in `indrajala-math-rust` (`rust/`) first, with its own
  numpy-only tests, then a PR here that bumps the submodule and runs the full suite
  (`./cli build-rust && ./cli test`) before either is merged.
- **Bit-identical claims are tested, not assumed.** A stage labelled bit-identical gets a test
  that compares the old and new op with exact equality (`==` on `tolist()`, not `approx`) over
  random inputs, including the edge cases called out in its plan. If exact equality fails,
  the stage is reclassified as bit-changing and follows the protocol below.
- **Bit-changing protocol.** For a stage that changes summation order:
  1. The existing `rtol` parity tests (`tests/test_*fused_layer_ops.py`, the step-by-step
     network parity tests) must pass unchanged. They are the correctness gate.
  2. Run the full suite. If an end-to-end pinned result moves (for example the Rust conv
     network's 0.9875 / epoch 10 / 0.925), don't loosen it silently. Measure it against a
     1-ULP control: nudge one initial weight by 1 ULP on the *old* code and see whether the
     pinned number moves by a similar amount. If it does, the change is rounding sensitivity
     (see "Training chaotic sensitivity" in `recommended-optimizations.md`); update the pin
     and record the control in the PR. If it doesn't, treat it as a bug.
  3. Record in the PR how far the new op's results are from the old one's (max abs and max
     ULP difference) at the benchmark shapes.
- **Each stage is its own PR, merged before the next starts.** A stage that measures no gain
  is closed with its numbers recorded, not merged.
