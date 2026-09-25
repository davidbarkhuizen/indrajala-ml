# Workplan: a Rust RNG identical to numpy's `np.random`

**Status: planned. Nothing is built yet.**

The goal is to replace the crate's xorshift128+ (`rust/src/random.rs`) with a drop-in equivalent of
numpy's legacy global `np.random`. After `pa.seed(s)`, the crate's `uniform`, `random` and dropout
masks must return the same bits that `np.random.uniform`, `np.random.random` and
`np.random.random(shape) >= p` return after `np.random.seed(s)`. A numpy network and a Rust network
seeded alike then start from identical weights and draw identical dropout masks.

This plan matches numpy exactly and nothing more. Improving on numpy (a faster or newer generator,
explicit generator objects) is for later. See [rng-audit.md](rng-audit.md), findings 4 and 5.

## Why

- **The Rust backend can't be reproduced today.** `randomize()` and dropout reseed from the clock
  on every call ([rng-audit.md](rng-audit.md), finding 2).
- **Parity becomes exact everywhere.** Randomized init and training-mode dropout are the only
  places where Rust/numpy parity is statistical today.
- **The workarounds go away.** The golden run trains Rust dropout only at `p = 0`, and the scaling
  study and the conv demo draw Rust weights with numpy.
- **It's known to be achievable.** `tests/test_numpy_rng_streams.py` already reproduces
  `np.random`'s stream bit for bit in pure Python.

## What "identical" means

numpy's legacy stream is specified by three pieces, all short and public. numpy freezes this stream
(NEP 19):

- **Seeding.** `np.random.seed(s)` has three paths (below). The int and sequence paths set the
  position to 624, so the first draw twists.
- **Generator.** The MT19937 twist and tempering (Matsumoto & Nishimura 1998).
- **Doubles.** `random_double` takes two 32-bit draws: `((a >> 5) * 67108864.0 + (b >> 6)) / 9007199254740992.0`.

Built on those:

- `uniform(low, high, size)` computes `range = high - low` once and fills C order with
  `low + range * u`. It raises `OverflowError` if `range` isn't finite.
- `random(shape)` fills C order with `u`.
- A dropout mask is `random(shape) >= p`, cast to float. `DropoutArrayLayer` draws it only in
  training mode, `(size,)` per example or `(batch, size)` per batch. The crate's fused
  `layer_dropout_forward*` already draws only when `training` is true, one flat `batch * size` draw
  in row-major order. The draw count and order already match.

### Seeding, exactly

`seed()` must match numpy for every input: the same stream for every seed numpy accepts, and the
same exception type for every seed it rejects. The paths below were measured against numpy 2.2.6.
`tests/test_numpy_rng_streams.py` pins the stream for each accepting path.

1. **Anything `operator.index` accepts runs `init_genrand(s)`.** That covers Python ints, bools and
   numpy integer scalars. numpy first calls `.squeeze()` when the seed has one, so `np.array([5])`
   and `np.array([[5]])` seed exactly like `5`. A Python list `[5]` has no `squeeze`, so it takes
   path 2 and gives a different stream from `5`. An int outside `[0, 2^32 - 1]` raises
   `ValueError("Seed must be between 0 and 2**32 - 1")`.
2. **Any other sequence runs `init_by_array(key)`.** That covers lists, tuples, ranges and 1-D
   arrays. The key is the sequence cast to int64 with `casting='safe'`, range-checked, then taken
   as uint32 words. The seeding differs from path 1 even for one word. The rejections:

   | Seed | numpy raises |
   |---|---|
   | empty | `ValueError("Seed must be non-empty")` |
   | not 1-D | `ValueError("Seed array must be 1-d")` |
   | a word outside `[0, 2^32 - 1]` | `ValueError`, as path 1 |
   | float or string elements | `TypeError`, the cast fails |
   | a `uint64` array, even with small values | `TypeError`, uint64 to int64 isn't a safe cast |

   int8 to int64 and uint32 arrays and `[True, 2]` are accepted.
3. **`None` draws from OS entropy.** It sets key word 0 to `0x80000000` and fills the rest from
   `SeedSequence().generate_state`. It does **not** reset the position; it keeps the previous one.
   So until the next twist, draws are tempered entropy words, not twisted output. The values are
   random by definition, so only the procedure can match. The crate follows it: OS entropy, the
   word-0 rule (a non-zero state), and the position left alone.

numpy also clears its cached Gaussian (`has_gauss`) on seeding. The crate has no normal draws, so
that has no counterpart until one is added.

The crate doesn't link numpy, so it duck-types these rules. It calls `squeeze` if present, tries
`__index__`, and otherwise iterates. Array dtypes are read through `.dtype.kind` and `.itemsize`:
kinds `b`, `i` and `u` are accepted, except 8-byte `u`. For a Python list, numpy first picks a
dtype for the whole list, so a list holding a value of 2^63 or more fails as `uint64` or object
(`TypeError`), not as out of range. Every rejection matches numpy's exception type. The two
`ValueError` messages that name the seed's shape or range also match word for word. The
`TypeError` messages are numpy's cast messages, which aren't reproduced.

The rest of the scope is only what the repo uses: scalar `low`/`high` and 1-D and 2-D shapes.
Broadcasting and `get_state`/`set_state` are out of scope.

## Decisions

- **numpy keeps its own RNG; only the crate's Rust code uses the crate's RNG.** The two states are
  separate. `pa.seed(s)` mirrors `np.random.seed(s)` and never touches numpy's state, and the
  numpy backend's draws are unchanged. Same seed means same stream in each, not one shared stream.
  So the crate is a drop-in copy of `np.random`, not a view of numpy's own state. The alternative
  was to draw from numpy's live state through `np.random.get_state`/`set_state`, which would need
  no new seed call. It costs 115 µs per round trip, measured, against about 12 µs for a batch-32
  dropout mask. It would also make the crate depend on numpy's internals. Borrowing numpy's MT19937 through its
  `bitgen_t` capsule would be exact by construction, but it isn't a Rust RNG, and it would make
  numpy a runtime dependency of the crate.
- **One seed call per backend, through the backend object.** `ArrayBackend` gains
  `seed = staticmethod(np.random.seed)` and the Rust backend gets `staticmethod(pa.seed)`. Code
  written against the backend (`self.backend.seed(s)`) then seeds whichever RNG `random_layer`
  draws from.
- **Unseeded state follows numpy's procedure, not its values.** Before any `pa.seed`, or after
  `pa.seed(None)`, the state comes from OS entropy with numpy's word-0 rule, as numpy's does. The
  entropy is built from `std::collections::hash_map::RandomState`, so there is no dependency. This
  also fixes [rng-audit.md](rng-audit.md) finding 3 (clock-based seeding).
- **After `fork`, children inherit the state, the same as numpy's global.** Pool workers that want
  distinct streams seed themselves, as `benchmark_sweep` workers already do.
- **xorshift128+ is removed.** There is no generator switch: the crate has one RNG, and it is
  numpy's.

## Pitfalls to design around

- **FMA contraction.** Rust never fuses `low + range * u` into an FMA. numpy's C might, depending
  on the compiler and flags: GCC in ISO C mode doesn't contract, but clang contracts within an
  expression by default, and arm64 has FMA in its baseline. Linux x86_64 wheels target a baseline
  without FMA, so today's CI can't show a difference. The PyPI plan's multi-platform CI
  ([pypi-release-workplan.md](pypi-release-workplan.md), stage 3) runs the parity tests on arm64
  and macOS and will catch it. If a numpy build contracts, record it and decide then. Don't
  pre-emptively add `mul_add`.
- **Seed validation.** pyo3 extracting a `u32` raises `OverflowError`, where numpy raises
  `ValueError`. Take the argument as a Python object, follow the three paths in "Seeding, exactly"
  in numpy's order (`squeeze`, then `__index__`, then iteration), and raise numpy's exception
  types. Take the edge-case table from a numpy run, not from reading numpy's source: the uint64 and
  list-dtype cases only show up when you run them.
- **The init limit.** `fan_in_aware_random_rust_layer` computes `1 / previous_size ** 0.5`, which is
  1 ULP off `np.sqrt` for 71 of the fan-ins 1 to 99,999 ([rng-audit.md](rng-audit.md), finding 6).
  It must become `math.sqrt`, or seeded weights differ at those fan-ins.
- **Draw order is part of the contract.** One call must draw its values sequentially in C order on
  one thread. The RNG must never move into the kernels' scoped worker threads. The global state is
  a `Mutex`, which is never contended because the crate never releases the GIL.
- **The golden run stays bit-identical through stages 1 to 3.** It injects its weights, and the
  Rust dropout network runs at `p = 0`, where every mask entry is 1 whatever the generator. A
  golden difference before stage 4 means something besides the RNG changed.
- **Speed.** MT19937 is roughly half as fast as xorshift128+ per draw (legacy numpy measured about
  6 ns against the crate's 3). The RNG isn't a hot path, but measure it with
  `scripts/rng_audit.py time`, committing each build first
  ([optimizations/measurement.md](optimizations/measurement.md)). Also time one dropout training
  epoch before and after, and record both.

## Stages

Each crate stage is a PR in the crate repo followed by a submodule bump PR here, with
`./cli build-rust && ./cli test` green and both CIs passing, as in
[pypi-release-workplan.md](pypi-release-workplan.md).

### Stage 1: MT19937 and `pa.seed`, checked against numpy (crate)

1. In `random.rs`, replace `Xorshift128Plus` with `Mt19937`: `key: [u32; 624]`, `pos`,
   `init_genrand`, `init_by_array`, the twist, the tempering, `next_u32` and numpy's
   `next_double`. Replace `fresh_seed` with an OS-entropy seed that follows numpy's `None`
   procedure.
2. Add one global `static STATE: Mutex<Option<Mt19937>>`, seeded lazily from entropy on first
   use. `uniform`, `draw_bernoulli_mask` (and so `bernoulli_mask` and the fused dropout ops) draw
   from it instead of constructing a generator per call.
3. Add the Python API: `seed(seed: int | None = None)` and `random(shape)`, the analogue of
   `np.random.random`. Add numpy's `OverflowError` for a non-finite range to `uniform`. Update the
   stub and the stubtest allowlist if needed.
4. Replace the statistical-only docstrings in `random.rs`, `test_random_uniform.py` and
   `test_dropout_rng.py`. Add `rust/tests/test_random_numpy_parity.py`, bit-identical against
   `np.random`:
   - `seed` then `random` and `uniform` at many seeds, including 0, 1 and 2^32 - 1, and at 1-D and
     2-D shapes, with lengths that cross the 312-double twist boundary;
   - interleaved sequences (`uniform`, `uniform`, `random`, `bernoulli_mask`, ...), so the
     position carries across calls and kinds;
   - `bernoulli_mask(p, shape)` against `(np.random.random(shape) >= p).astype(float)`, and the fused
     `layer_dropout_forward*` masks against the same;
   - reseeding restarts the stream; `training=False` draws nothing (the next draw still matches
     numpy);
   - each seeding path. Streams must be identical for ints, bools, numpy ints, squeezable arrays,
     and lists, tuples, ranges and 1-D arrays of every accepted dtype, including keys longer than
     624 words.
   - a table of rejected seeds, run through both `np.random.seed` and `pa.seed`, raising the same
     exception type. The table covers the "Seeding, exactly" cases, including uint64 arrays and a
     list holding 2^63.
   - `seed(None)` and unseeded calls differ between processes. `seed(None)` keeps the position, as
     numpy's does.
5. Keep the statistical tests. They still hold, and `scripts/rng_audit.py quality` reruns them.
6. Update `rust/README.md`'s `random.rs` row.

Done when the crate CI passes with the parity tests, and `rng_audit.py time` numbers for old and
new builds are in the PR.

### Stage 2: seed through the backend, and exact init parity (indrajala-ml)

1. Bump `rust/`. The golden run check must be bit-identical.
2. Change `fan_in_aware_random_rust_layer` to use `math.sqrt`.
3. Add `seed` to both backend objects in `array_backend.py`.
4. Add a parity test over every array network class, dense and conv, including the dropout,
   momentum and Adam variants. After `backend.seed(s)`, `randomized(...)` must give bit-identical
   numpy and Rust snapshots, at several seeds and at fan-ins including 2921 and 5579 (the `** 0.5`
   cases).

Done when the parity test passes and the golden check is bit-identical.

### Stage 3: exact dropout training parity (indrajala-ml)

1. Make the training-mode dropout parity tests exact: seed both backends, then compare `forward`,
   `forward_batch` and a full `learn_batch` step of `DropoutArrayLayer` against
   `DropoutRustArrayLayer`. Use the same bar as the `training=False` tests. Replace
   `DropoutRustArrayLayer`'s "only statistical at training=True" docstring.
2. Add an end-to-end check: a seeded dropout network trained a few epochs on both backends gives
   the same snapshots, to the bar the non-dropout end-to-end parity tests already use.
   Training is chaotically sensitive, so if matmul results already differ by 1 ULP between the
   backends, judge it step by step against a no-dropout control run.

Done when both pass.

### Stage 4: retire the workarounds (indrajala-ml)

1. `golden_training_run.py`: seed with `pa.seed` too, train the Rust dropout network at the same
   `drop_probability` as numpy's, and drop the docstring paragraph about the workaround.
   Re-record the golden file on main after merge. This is the one intended golden change, and the
   PR explains it.
2. `batch_size_scaling.initial_network`, the conv demo's `initial_snapshot`, and the timing
   scripts that draw Rust weights with numpy can stay as they are. Drawing once and restoring into
   both backends is still correct. Remove their "the RNGs aren't comparable" comments. Switch to
   `backend.seed` + `randomized` only where it makes the code simpler.
3. Add a `seed_everything(s)` entry point that seeds `random`, `np.random` and `pa.seed` for the
   call sites that seed pairwise today ([rng-audit.md](rng-audit.md), recommendation 4).

Done when the golden run covers real Rust dropout and no comment claims the RNGs can't match.

### Stage 5: docs (indrajala-ml)

Update [rng-audit.md](rng-audit.md) in place to the new state: findings 1, 2, 3 and 6 resolved, and
the tables with the MT19937 crate numbers. Move this workplan's still-relevant content, such as
the FMA note, into that doc. Then delete this workplan.

## After this plan

- Explicit generator objects (numpy's `Generator` style) instead of global state.
- A faster generator or PCG64 parity (`default_rng`), weighed against the frozen-stream guarantee
  that makes the legacy stream a stable target.
- The pure-Python backend's stdlib `random`. It is also MT19937, but seeded through
  `init_by_array`, so matching it is a separate piece of work.

## Out of scope

- `get_state`/`set_state`, broadcasting `low`/`high`, and any other `np.random` function the repo
  doesn't call.
- The wording of numpy's cast `TypeError` messages. The exception types match.
