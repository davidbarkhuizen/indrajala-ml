# RNG audit

The random number generators this project uses: how the Rust crate's generator
(`rust/src/random.rs`) reproduces numpy's legacy `np.random` bit for bit, how to seed a run, and
what is still open. The measurements and checks come from these harnesses:

- `rust/tests/test_random_numpy_parity.py`: the crate against `np.random`, bit for bit. It covers
  every draw function, the fused dropout masks, every seeding path and numpy's rejections.
- `tests/test_seeded_init_parity.py`: seeded `randomized()` gives bit-identical numpy and Rust
  networks, for every array network class.
- `tests/test_dropout_fused_layer_ops.py` and `tests/test_dropout_array_multiclass_backprop_model.py`:
  seeded training-mode dropout, with identical masks at the layer and the network level.
- `tests/test_numpy_rng_streams.py`: a pure-Python MT19937 that reproduces `np.random`'s stream and
  seeding. It is the algorithm the crate ports, checked against numpy without the crate. It also
  pins how the stdlib `random`'s stream relates to numpy's.
- `scripts/rng_audit.py quality | time`: statistical checks, and per-draw timing with one process
  per backend.

Measured on an AMD Ryzen 7 3700U, numpy 2.2.6. The crate's CI runs the parity tests against the
latest numpy.

## Where randomness comes from

| Use | Code | Generator | Seeded by |
|---|---|---|---|
| Pure-Python weight init | `fan_in_aware_weights_and_bias`, `randomize()` of the node networks | stdlib `random`, global MT19937 | `random.seed(s)` |
| Pure-Python dropout | `DropoutNode.forward` (`random.random() >= p`) | stdlib `random`, global | `random.seed(s)` |
| Epoch shuffle, all backends | `train.py`, `epoch_order` (`random.shuffle`) | stdlib `random`, global | `random.seed(s)` |
| Data splits, sampling, ensemble jobs | `dataset_utils`, `benchmark_data`, `ensemble_train` | `random.Random(seed)` instances | an explicit seed argument |
| numpy weight init | `fan_in_aware_random_layer` (`np.random.uniform`) | legacy `np.random`, global MT19937 | `np.random.seed(s)`, `NUMPY.seed(s)` |
| numpy dropout | `DropoutArrayLayer` (`np.random.random(shape) >= p`) | legacy `np.random`, global | as above |
| Rust weight init | `fan_in_aware_random_rust_layer` (`pa.uniform`) | the crate's MT19937, a global separate from numpy's | `pa.seed(s)`, `RUST.seed(s)` |
| Rust dropout | `layer_dropout_forward*` and `bernoulli_mask` | the crate's, as above | as above |

`seed_everything(s)` (`indrajala_ml/seeding.py`) seeds all three global states alike. The
ensemble workers call it, because a forked worker inherits all three states from its parent.
`backend.seed(s)` seeds only the state that backend's `random_layer` and dropout draw from.

After the same seed, numpy and Rust draw the same weights and masks, so a seeded Rust run
reproduces a seeded numpy run. They agree to the backends' matmul differences, which are about an
ULP. Python's `random` is the same MT19937 with the same two-draw double, but `random.seed(s)`
runs `init_by_array` over `|s|`'s 32-bit words, low word first. So `random.seed(s)` gives the
stream of `np.random.seed(words)` and `pa.seed(words)`, not of `np.random.seed(s)`
(`test_stdlib_random_is_np_random_seeded_with_the_seeds_words`).

## The crate's generator

| | crate (`random.rs`) | numpy legacy `np.random` | numpy `default_rng()` |
|---|---|---|---|
| Algorithm | MT19937 | MT19937 | PCG64 (128-bit LCG, XSL-RR output) |
| State, period | 19937 bits, 2^19937 - 1 | the same | 128 bits + increment, 2^128 |
| Seeding | numpy's, below | `init_genrand`, `init_by_array` or entropy | `SeedSequence` hash of the seed or OS entropy |
| Unit float | `((a >> 5) * 2^26 + (b >> 6)) * 2^-53`, two 32-bit draws | the same | `(u64 >> 11) * 2^-53` |
| `uniform` | `low + (high - low) * u`, `OverflowError` on a non-finite range | the same | the same |
| Dropout mask | `u >= p` | `np.random.random(shape) >= p`, the same | the same |
| Stream stability | numpy's | frozen (NEP 19 compatibility guarantee) | may change between numpy versions |

The crate is a copy of `np.random`, not a view of numpy's state. The two states are separate:
`pa.seed(s)` never touches numpy's, and one seed gives the same stream in each. Borrowing numpy's
state through `get_state`/`set_state` costs about 115 µs per round trip, against about 12 µs for a
batch-32 dropout mask, and it would make the crate depend on numpy's internals.

`random(shape)`, `uniform(low, high, shape)` and `bernoulli_mask(p, shape)` fill in C order. The
fused `layer_dropout_forward*` draw one flat `batch * size` mask in row-major order, only when
`training` is true, as `DropoutArrayLayer` does. The order of draws is part of the contract. Every
draw happens in one call on the calling thread, and never in the kernels' scoped worker threads.
The state is a `Mutex`, which is never contended because the crate never releases the GIL.

### Seeding

`pa.seed(seed)` matches `np.random.seed` for every input: the same stream for every seed numpy
accepts, and the same exception type for every seed it rejects. numpy's three paths:

1. **Anything `operator.index` accepts runs `init_genrand(s)`.** That covers Python ints, bools and
   numpy integer scalars. numpy first calls `.squeeze()` when the seed has one, so `np.array([5])`
   and `np.array([[5]])` seed exactly like `5`. `np.array([5], dtype=np.uint64)` also squeezes to
   an index. A Python list `[5]` has no `squeeze`, so it takes path 2 and gives a different stream
   from `5`. An int outside `[0, 2^32 - 1]` raises `ValueError("Seed must be between 0 and 2**32 - 1")`.
2. **Any other sequence runs `init_by_array(key)`.** That covers lists, tuples, ranges, 1-D
   arrays and buffers such as `array.array`, `bytearray` and `memoryview`. numpy takes
   `np.asarray(seed)`, then checks it in this order: non-empty (`ValueError("Seed must be
   non-empty")`), cast to int64 with `casting='safe'` (`TypeError`), 1-D (`ValueError("Seed array
   must be 1-d")`), then every word in range (`ValueError`, as path 1). The cast fails for:
   - float, complex, string and object elements;
   - `uint64` arrays and buffers;
   - a list holding a value of 2^63 or more, or below -2^63, since numpy picks `uint64` or object
     for the whole list;
   - timedelta arrays.

   An inhomogeneous nested list fails in `asarray`, with numpy's `ValueError`.
3. **`None` draws from OS entropy.** Key word 0 is `0x80000000`, so the state is never all zero,
   and the rest is entropy. The position is kept, not reset, so until the next twist the draws are
   tempered entropy words.

The unseeded state follows numpy's global `RandomState`. It is built at import: key word 0 is
`0x80000000`, the rest is entropy, and the position is 623. Processes forked from an importer
therefore inherit one stream, as they inherit numpy's. The crate's entropy comes from std's
`RandomState`, keyed from the OS once per thread, with the pid and the clock mixed in, so it needs
no dependency.

The crate doesn't link numpy, so it duck-types these rules. It calls `squeeze` if present, then
tries `operator.index`. Otherwise it walks the seed the way `np.asarray` discovers a shape and a
dtype. numpy values are read through `.dtype.kind` and `.itemsize`: kinds `b`, `i` and `u` are
accepted, except 8-byte `u`. Buffers are read through `memoryview.format`. The three `ValueError`
messages match numpy's word for word. numpy's cast `TypeError` messages aren't reproduced.

numpy versions differ on one seed. numpy 2.2 still takes `np.bool_` as an index, with a
deprecation warning, while later versions reject it: its `squeeze` gives a 0-d array, which fails
as "Seed array must be 1-d". `pa.seed` follows `operator.index`, so it matches whichever numpy is
installed.

The position after `seed(None)` isn't observable without `get_state`, which the crate doesn't
provide. A probe build that exposed `(pos, key[0])` confirmed each case against numpy's
`get_state()[2]`: 623 at import, kept by `seed(None)`, and 624 after an int or sequence seed.

### Statistical quality

`scripts/rng_audit.py quality --repeats 20`, 10M draws per generator:

| Generator | chi-square z (4096 bins) | KS p | lag-1 z | call-to-call z |
|---|---:|---:|---:|---:|
| crate | +0.18 | 0.246 | +0.39 | -0.90 |
| numpy legacy | +0.45 | 0.505 | -0.26 | +0.31 |
| numpy PCG64 | +1.13 | 0.559 | +0.08 | +0.81 |

`bernoulli_mask`'s keep rate over 2M draws: z = -1.07, +0.78 and +0.72 at drop probabilities 0.1,
0.5 and 0.9. Over 20 repeats on 2M draws, the crate's KS p-values spread over [0.09, 1.00], and its
lag-1 z has mean -0.33 and sd 1.20, as independent draws should. Every statistic is within chance.
The crate and numpy legacy are the same algorithm, so their rows differ only by their entropy
seeds.

### Speed

`scripts/rng_audit.py time --repeats 5`, nanoseconds per draw, median over processes:

| Case | crate | numpy legacy | numpy PCG64 |
|---|---:|---:|---:|
| uniform (128, 64) | 5.80 | 6.42 | 4.12 |
| uniform (784, 128) | 5.75 | 6.13 | 3.84 |
| uniform (1000, 1000) | 5.82 | 6.27 | 4.24 |
| mask (1, 128) | 7.86 | 27.12 | 25.86 |
| mask (32, 128) | 5.33 | 6.51 | 4.87 |
| mask (512, 128) | 5.21 | 5.88 | 4.12 |

The crate runs MT19937 slightly faster than numpy's legacy path, and avoids numpy's per-call
overhead at batch 1. The RNG is not a hot path. A 784 x 128 init happens once per network, and
the dropout mask is a small part of a training step. One Rust dropout epoch (784-128-10, batch 32,
p = 0.5, 8192 rows) takes a median of 206 ms. The xorshift128+ generator the crate used before
drew at about 3 ns, and the same epoch took 200 ms, within that build's 193-226 ms spread.

## Open findings

### Three global states (medium)

`random`, `np.random` and the crate's RNG are all global. `seed_everything` seeds them together,
but with global state every draw shifts every later one. For example, a dropout mask drawn during
training changes the weights the next `randomize()` gets unless the code reseeds in between. The
scripts still seed per call site: `batch_size_scaling`, `accuracy_pass_timing` and the timing
scripts draw weights once with numpy and restore them into both backends, which is correct and
simple. numpy's own guidance (NEP 19) is to pass explicit `Generator` objects. That is the larger,
cleaner version: generator objects in the crate and on the Python side, passed to layers.

### FMA contraction on other platforms (low, latent)

Rust never fuses `low + range * u` into an FMA. numpy's C might, depending on the compiler and
flags. GCC in ISO C mode doesn't contract, but clang contracts within an expression by default,
and arm64 has FMA in its baseline. Linux x86_64 wheels target a baseline without FMA, so today's
CI can't show a difference. The PyPI plan's multi-platform CI
([pypi-release-workplan.md](pypi-release-workplan.md), stage 3) runs the parity tests on arm64
and macOS and will catch it. If a numpy build contracts, record it and decide then. Don't
pre-emptively add `mul_add`.

### `[low, high)` is not strictly guaranteed (informational)

`low + (high - low) * u` can round up to `high` for some asymmetric ranges. This is true of numpy
too, and numpy's docs say so. The crate's `test_random_uniform.py` asserts a strict upper bound
for `(-2, 5)`, which holds for its draws. For the symmetric `[-limit, limit)` that init uses, no
fan-in from 1 to 4999 can produce `limit`.

## Open work

- Explicit generator objects (numpy's `Generator` style) instead of global state, as above.
- PCG64 and `default_rng` parity. The float formula differs from the legacy stream's, and a port
  needs a `SeedSequence` port too. It gives up the frozen-stream guarantee that makes the legacy
  stream a stable target.
- Matching the pure-Python networks with the array networks from one seed. The streams already
  match when seeded through the words, as above. What's left is draw order: the per-node networks
  draw weights node by node, and their dropout draws one `random.random()` per node. Until that's
  checked, the per-node dropout reference is compared with the array networks only at eval.
- `get_state`/`set_state`, and broadcast `low`/`high`, aren't provided. The repo doesn't use them.
