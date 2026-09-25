# RNG audit

What random number generators this project uses, how the Rust crate's generator
(`rust/src/random.rs`) compares with numpy's, what is wrong with the current arrangement, and what to change. The
measurements come from two committed harnesses:

- `tests/test_numpy_rng_streams.py`: a pure-Python MT19937 that reproduces numpy's legacy
  `np.random` stream bit for bit, and a check that numpy's PCG64 `Generator` uses the crate's float
  formula.
- `scripts/rng_audit.py quality | time`: statistical checks and per-draw timing for the crate,
  legacy `np.random` and `default_rng()`, timed with one process per backend.

Measured on an AMD Ryzen 7 3700U, numpy 2.2.6 (the parity test also passes on numpy 1.21.5).

## Where randomness comes from

| Use | Code | Generator | How it is seeded |
|---|---|---|---|
| Pure-Python weight init | `fan_in_aware_weights_and_bias`, `randomize()` of the node networks | stdlib `random`, global MT19937 | `random.seed(s)` at the call site |
| Pure-Python dropout | `DropoutNode.forward` (`random.random() >= p`) | stdlib `random`, global | `random.seed(s)` |
| Epoch shuffle, all backends | `train.py`, `epoch_order` (`random.shuffle`) | stdlib `random`, global | `random.seed(s)` |
| Data splits, sampling, ensemble jobs | `dataset_utils`, `benchmark_data`, `ensemble_train` | `random.Random(seed)` instances | explicit seed argument |
| numpy weight init | `fan_in_aware_random_layer` (`np.random.uniform`) | legacy `np.random`, global MT19937 | `np.random.seed(s)` |
| numpy dropout | `DropoutArrayLayer` (`np.random.random(shape) >= p`) | legacy `np.random`, global | `np.random.seed(s)` |
| Rust weight init | `fan_in_aware_random_rust_layer` (`pa.uniform`) | crate xorshift128+ | none: a new seed from the wall clock on every call |
| Rust dropout | `layer_dropout_forward*` and `bernoulli_mask` (`draw_bernoulli_mask`) | crate xorshift128+ | none, as above |

So a seeded run needs up to three generators seeded in step, and on the Rust backend one of them
can't be seeded at all.

## The crate's generator compared with numpy's

| | crate (`random.rs`) | numpy legacy `np.random` | numpy `default_rng()` |
|---|---|---|---|
| Algorithm | xorshift128+ | MT19937 | PCG64 (128-bit LCG, XSL-RR output) |
| State, period | 128 bits, 2^128 - 1 | 19937 bits, 2^19937 - 1 | 128 bits + increment, 2^128 |
| Seeding | splitmix64 of `nanos ^ counter * φ`, new per call | `init_genrand(seed)` for an int seed | `SeedSequence` hash of the seed or OS entropy |
| Seedable | no | yes | yes |
| Unit float | `(u64 >> 11) * 2^-53` | `((a >> 5) * 2^26 + (b >> 6)) * 2^-53`, two 32-bit draws | `(u64 >> 11) * 2^-53` |
| `uniform` | `low + (high - low) * u` | the same | the same |
| Dropout mask | `u >= p` | `np.random.random(shape) >= p`, the same | the same |
| Stream stability | n/a | frozen (NEP 19 compatibility guarantee) | may change between numpy versions |

The crate's formulas match numpy's: 53-bit floats, the same `uniform` transform and the same
mask comparison. The distributions are therefore identical, and only the underlying bit streams
differ. `test_pcg64_generator_floats_are_the_crates_top_53_bits_formula` confirms that `Generator`
builds its floats exactly the way the crate does.

### Statistical quality

`scripts/rng_audit.py quality`, 10M draws per generator:

| Generator | chi-square z (4096 bins) | KS p | lag-1 z | call-to-call z |
|---|---:|---:|---:|---:|
| crate | -0.93 | 0.551 | -0.49 | -0.28 |
| numpy legacy | -0.66 | 0.305 | -0.84 | +0.82 |
| numpy PCG64 | +1.72 | 0.411 | -0.16 | -0.15 |

`bernoulli_mask` keep rate over 2M draws: z = +0.97, +0.60 and -0.54 at drop probabilities 0.1, 0.5
and 0.9.

Every statistic is within chance for all three generators. A single run can land on a borderline
value: an exploratory run gave the crate KS p = 0.045 and lag-1 z = +2.03. The harness's 20
repeats on 2M draws settle it: the KS p-values spread over [0.02, 0.88], and the lag-1 z has mean
+0.21 and sd 0.94, as independent draws should. The call-to-call z is the check that matters for
the crate's reseed-per-call design: the first draws of consecutive calls are uncorrelated.

### Speed

`scripts/rng_audit.py time --repeats 5`, nanoseconds per draw, median over processes:

| Case | crate | numpy legacy | numpy PCG64 |
|---|---:|---:|---:|
| uniform (128, 64) | 2.99 | 6.37 | 4.16 |
| uniform (784, 128) | 2.93 | 6.03 | 3.90 |
| uniform (1000, 1000) | 3.20 | 6.28 | 4.03 |
| mask (1, 128) | 5.79 | 27.07 | 25.81 |
| mask (32, 128) | 3.10 | 6.51 | 4.75 |
| mask (512, 128) | 2.98 | 5.87 | 4.04 |

The crate is the fastest of the three, about 2x faster than legacy `np.random` and 1.3x faster than
PCG64 per draw. At batch 1, numpy's per-call overhead dominates. The RNG is not a hot path: a
784 x 128 init happens once per network, and a batch-32 dropout mask over 128 units takes
about 12 µs of a training step.

## Findings

### 1. The premise that numpy's stream can't be reproduced is false (high)

`random.rs`, `rust/README.md`, `rust_array_layer.py`, `array_network_shapes.py`, the crate's
`test_random_uniform.py`/`test_dropout_rng.py` and `scripts/golden_training_run.py` all say that a
hand-rolled generator can never reproduce numpy's Mersenne Twister stream, so Rust/numpy parity for
random ops can only be statistical. That is wrong. MT19937, its `init_genrand` seeding and numpy's
53-bit `random_double` are short, public, deterministic algorithms, and numpy freezes the legacy
stream. `test_mersenne_twister_reproduces_the_legacy_stream_randomize_and_dropout_draw` reproduces
`np.random.seed(s)`, then `uniform` for W, `uniform` for b, then a `random() >= p` dropout mask,
bit for bit, at seeds 0, 1, 42 and 2^32 - 1, in about 60 lines of Python. The stdlib's
`random.seed(0)` stream differs from numpy's even though both are MT19937, because the stdlib
seeds through `init_by_array`. A port would match `np.random` specifically.

This premise is the root of findings 2 and 5. Once it goes, "same seed, same weights and masks on
both backends" is achievable, which would make Rust/numpy parity exact for randomized networks,
the one place it is statistical today.

### 2. The Rust backend is not reproducible (high)

`uniform` and `bernoulli_mask` take no seed and reseed from the wall clock on every call. The
consequences are already visible in the repo:

- `golden_training_run.py` injects weights from `random.Random`, because `randomize()` can't be
  seeded on Rust. It trains the Rust dropout network at `drop_probability=0.0`, so the bit-identical
  refactoring gate never covers a real Rust dropout mask.
- `batch_size_scaling.initial_network` draws Rust networks' weights with numpy so that seeds mean
  anything.
- Any Rust run with `randomize()` or dropout can't be repeated. A regression can't be bisected from
  a seed, and paired-seed comparisons that include Rust dropout aren't paired. Training is
  chaotically sensitive (1-ULP differences flip end-of-run conv results), so run-to-run variance
  isn't a small effect either.
- `DropoutRustArrayLayer` is tested for parity with `DropoutArrayLayer` only at `training=False`,
  and only statistically at `training=True`.

### 3. Unseeded draws seed from the clock and a per-process counter (low)

`fresh_seed()` is `nanos ^ counter * 0x9E3779B97F4A7C15`, expanded by splitmix64. Within one
process the counter keeps calls apart. After `fork`, which is how `multiprocessing.Pool` starts
sweep workers on Linux, every child inherits the same counter value. Two workers that call
`uniform` in the same nanosecond therefore draw identical streams. This hasn't been observed and is
unlikely at nanosecond resolution, but it's the textbook failure of time-based seeding. numpy's
unseeded `default_rng()` draws from OS entropy instead. The fix needs no dependency: std's
`std::collections::hash_map::RandomState` is keyed from OS randomness once per process and
per-thread incremented, so hashing the counter with it gives OS-derived seeds. Reading
`/dev/urandom` also works on Linux and macOS.

### 4. xorshift128+ is dated, but its known weakness doesn't reach the floats (low)

xorshift128+ fails TestU01's linearity tests (MatrixRank, LinearComp) on its lowest bits (Vigna,
*Further scramblings of Marsaglia's xorshift generators*, 2017; Lemire and O'Neill, 2019). The crate
discards the low 11 bits when it makes a float, which is how Vigna recommends using the `+`
scramblers, and the quality checks above find nothing. The shift triple 23/17/26 is the one from
the paper's preprint, which V8 shipped. The published version uses 23/18/5. Vigna's current
recommendation for floats is xoshiro256+. None of this matters if the crate adopts MT19937
(recommendation 1). If it keeps its own generator instead, move to xoshiro256+ or ++.

### 5. Three generators, two global states (medium)

The Python side uses the stdlib `random` global (shuffles, pure-Python init and dropout) and the
legacy `np.random` global (numpy init and dropout). A reproducible numpy run needs both seeded, and
the scripts do this pairwise at each call site (`batch_size_scaling`, `accuracy_pass_timing`,
`golden_training_run`, the tests). With global state, every draw shifts every later one. For
example, a dropout mask drawn during training changes the weights the next `randomize()` gets
unless the code reseeds in between. numpy's own guidance (NEP 19) is to pass explicit `Generator`
objects. The tests already use `default_rng` in places, while the production code uses the legacy
global.

### 6. The Rust init limit uses `** 0.5`, the other backends use `sqrt` (low, latent)

`fan_in_aware_random_rust_layer` computes `1.0 / (previous_size**0.5)`. The numpy layer uses
`np.sqrt` and the pure-Python code uses `math.sqrt`, which agree with each other. `x ** 0.5` isn't
correctly rounded: over fan-ins 1 to 99,999 it differs from `sqrt` in 82 cases (e.g. 2921, 5579),
and 71 of those survive the `1.0 /` as a 1-ULP different limit. None of the fan-ins in use hit
this, and it can't show today because Rust init is unseeded. It would break bit-identical seeded
parity for those fan-ins once recommendation 1 lands.

### 7. `[low, high)` is not strictly guaranteed (informational)

`low + (high - low) * u` can round up to `high` for some asymmetric ranges. This is true of numpy
too, and numpy's docs say so. The `random.rs` docstring and `test_random_uniform.py` claim a strict
upper bound. For the symmetric `[-limit, limit)` that init uses, no fan-in from 1 to 4999 can
produce `limit`.

## Recommendations

In order:

1. **Make the crate's RNG a seedable MT19937 that reproduces `np.random` bit for bit.** Port
   `init_genrand`, the twist and tempering, and numpy's `random_double` into `random.rs` (about 60
   lines, no dependency, in keeping with the crate's hand-built posture). Expose a module-level
   `seed(int)` that mirrors `np.random.seed`, which the repo's code already calls, and have
   `uniform`, `bernoulli_mask` and the fused dropout ops draw from that state. Then
   `np.random.seed(s)` on numpy and `pa.seed(s)` on Rust give identical weights and masks, provided
   the draws happen in the same order, which `randomize()` already guarantees. Gate it on a crate
   test that compares against `np.random` directly, with `tests/test_numpy_rng_streams.py` as the
   pure-Python oracle. This fixes finding 2, retires the workarounds in `golden_training_run.py` and
   `batch_size_scaling.py`, and makes the dropout parity tests exact. Cost: MT runs at about legacy
   numpy's speed per draw, roughly 2x the current crate; measure it with `rng_audit.py time`. One
   open design choice: global state like `np.random`, which is simple and matches today's call
   sites, or a `Generator`-style object passed to layers, which is explicit and thread-safe but
   touches every constructor. A Rust global needs a `Mutex` or `thread_local`, but it won't be
   contended, since the crate never releases the GIL and threads only inside kernels. Parity also
   requires drawing in numpy's C order on one thread: the draws can't move into the kernels'
   scoped worker threads.

   Matching PCG64 and `default_rng` instead is also possible, since the float formula already
   matches. It needs a `SeedSequence` port, moving the Python side to `Generator` objects first,
   and it gives up the frozen-stream guarantee. Legacy MT matches the code as it stands.

2. **Seed unseeded draws from OS entropy** (finding 3), whether or not 1 lands.

3. **Use `math.sqrt` in `fan_in_aware_random_rust_layer`** (finding 6). It's a one-line change and a
   precondition for exact seeded parity.

4. **Add one seeding entry point on the Python side**, e.g. `seed_everything(s)` that seeds `random`,
   `np.random` and (after 1) the crate, instead of pairwise seeding at each call site (finding 5).
   Moving production code to explicit `Generator`/`random.Random` objects is the larger, cleaner
   version.

5. **Correct the docs as the code changes:** the "can never reproduce" wording listed in finding 1,
   the `[low, high)` claim (finding 7), and the `rust/README.md` row for `random.rs`.
