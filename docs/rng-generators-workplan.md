# Workplan: explicit generator objects

**Status: stages 1-6 done (crate #46 and #522: the crate's PCG64 `Generator`; #523: the pure-Python
port; #524: the numpy and Rust networks own their generators; #525: the pure-Python networks
do; #526: the trainers' shuffles and the sampling helpers take a `random.Random`; #527: the
generator's state in checkpoints and format-2 files; #528: run checkpoints); stage 7b, the globals
removed, in progress.**

Every random draw the package makes comes from one of three global states: Python's `random`,
numpy's legacy `np.random` and the crate's MT19937 ([rng-audit.md](rng-audit.md), "Three global
states"). This plan moves every draw in a training run onto generator objects: numpy's
`Generator(PCG64)`, a bit-identical PCG64 in the crate and in pure Python, and a `random.Random`
for the trainers' shuffles. Each network owns its generator, and its state is saved with the
network, so a saved dropout run resumes exactly (next-steps.md, "Saving RNG state").

## Why

- **Global state couples unrelated draws.** A dropout mask drawn in training shifts the weights the
  next `randomize()` gets, unless the code reseeds in between. A second network trained in the
  same process shifts the first one's masks.
- **A run can't be resumed exactly.** A checkpoint and a format-2 file hold the weights and the
  optimizer's state, but not the position of the stream the next dropout mask comes from, nor the
  next epoch's batch order.
- **Forked workers inherit one stream.** `ensemble_train` calls `seed_everything` in each worker
  for this reason. A generator per network removes the problem rather than working around it.
- **numpy's own guidance** (NEP 19) is explicit `Generator` objects, not the legacy global.

## Decisions (settled 2026-10-01)

- **D1. Scope: all three implementations and the shuffles.** numpy and Rust init and dropout, the
  pure-Python networks' init and dropout, and the trainers' epoch shuffles.
- **D2. numpy's `Generator(PCG64)`, not the legacy MT19937.** The crate gets a PCG64 and a
  `SeedSequence` that reproduce `np.random.default_rng` bit for bit. Seeded dropout results move,
  and the golden run's dropout entries are re-recorded (below).
- **D3. The generator's state is saved.** `checkpoint()` holds it, and `restore_checkpoint()`
  rewinds it. A format-2 file carries it in an optional field.
- **D4. The package stops using global state.** Nothing in `indrajala_ml` seeds or draws from
  `random`, `np.random` or the crate's module functions. `backend.seed` and `seed_everything` are
  deleted once the tests and scripts are migrated. The crate keeps its module functions (D7).
- **D5. The pure-Python networks get a pure-Python PCG64.** It reproduces `default_rng`'s `random`
  and `uniform`, as `tests/test_numpy_rng_streams.py` reproduces MT19937, so all three
  implementations draw from one stream family and share one state layout.
- **D6. The shuffles use a `random.Random` passed to the trainers.** The algorithm is today's
  `random.shuffle`, so a seed gives today's batch order and the golden run's shuffles don't move.
  It is backend-free and serves all three implementations alike.
- **D7. The crate's module functions stay numpy's legacy mirror.** `pa.seed`, `pa.random`,
  `pa.uniform` and `pa.bernoulli_mask` keep drawing from the global MT19937, matching `np.random`.
  The PCG64 is a separate `pa.Generator`. That is numpy's own split, and the existing parity tests
  keep passing.
- **D8. The network owns its generator.** `randomized(..., seed=s)` or `randomized(..., rng=g)`
  sets `network.rng`, and so does assigning it. `randomize()` and the dropout layers draw from it.
  The `learn*` signatures and the protocols don't change. Init and dropout share one stream, as
  they share the global today.
- **D9. No seed means OS entropy.** A network built or randomized without `seed` or `rng` gets a
  fresh entropy-seeded generator, as `default_rng()` does. Its state is still saved, so even an
  unseeded run resumes exactly; only its first run can't be replayed from scratch.
- **D10. The trainer's run state goes into a run checkpoint and a self-contained run file.** The
  pocket restores the best epoch at the end of a run, so the model file holds the best epoch's
  network, not the last's, and a bit-identical resume needs both (owner, 2026-10-01). The trainer
  returns `result.run_checkpoint`, a `RunCheckpoint` taken before the pocket restore: the last
  epoch's network checkpoint, the best epoch's, the best accuracy and its epoch index, the
  per-epoch accuracies, the convergence series, the shuffle generator's `random.Random` state, and
  the epoch and batch counters (the learning-rate schedule and warmup read the batch count).
  `train_backprop_network_mini_batch` takes `resume_from=`. `save_run` writes one JSON file that
  embeds both networks as format-2 network entries. The model file stays the pocketed model. Runs
  resume at epoch boundaries only, and a run file holds one network, never an ensemble.

## What "identical" means

numpy's `default_rng(seed)` is `Generator(PCG64(SeedSequence(seed)))`. Each piece is short and
public:

- **`SeedSequence(entropy, spawn_key=())`** hashes the entropy (an int of any size, a sequence of
  ints, or 128 bits of OS entropy for `None`) and the spawn key into a pool of four 32-bit words.
  `generate_state(4, uint64)` expands the pool into 256 bits. `spawn(n)` gives children with spawn
  keys `(0,)` to `(n - 1,)`.
- **PCG64** (O'Neill 2014, XSL-RR) seeds `state` and `inc` from those 256 bits, steps a 128-bit
  LCG and outputs 64 bits per step.
- **Doubles.** `Generator.random` is `(next_uint64 >> 11) * 2^-53`, one draw per double.
- **`uniform(low, high, size)`** is `low + (high - low) * u`, filling C order.
- **A dropout mask** is `rng.random(shape) >= p`, cast to float, as today.
- **The state** is numpy's `bit_generator.state`: `{"bit_generator": "PCG64", "state": {"state":
  int, "inc": int}, "has_uint32": int, "uinteger": int}`. The two last fields buffer half of a
  64-bit draw for 32-bit draws, which nothing here makes, but they round-trip unchanged.

Checked against numpy 2.5.3: `default_rng(7).random(3)` equals `(raw >> 11) * 2^-53` over
`PCG64(SeedSequence(7)).random_raw(3)`, and `uniform(-0.5, 0.5, 3)` equals `-0.5 + 1.0 * random(3)`.

The three implementations hold the same state layout. A network saved from numpy loads into Rust
and draws the same next mask, and a pure-Python network's state means the same thing.

## Design

- **The backend makes the generator.** `NumpyBackend.default_rng = np.random.default_rng`,
  `RustBackend.default_rng = pa.default_rng`, and the pure-Python networks use the pure-Python
  port's. `random_layer(rng, size, previous_size)` and `random_weights(rng, ...)` draw from `rng`.
- **`network.rng` is a property.** Its setter hands the generator to every layer that draws, as
  `_set_training_mode` reaches the training-mode layers. The dropout layers draw from that
  reference: `rng.random(...)` on numpy, and the fused `pa.layer_dropout_forward*` take it as a new
  `rng` argument (D7: without one they draw from the global, as today).
- **Ensembles spawn.** An ensemble's sub-network `i` gets `SeedSequence(seed).spawn(n)[i]`, so the
  sub-networks' streams are independent and one ensemble seed reproduces them all.
  `ensemble_train`'s workers seed their network and their shuffle and stop calling
  `seed_everything`.
- **The checkpoint gains the generator's state.** `Checkpoint` gets an `rng` field (the state
  dict), and `restore_checkpoint` sets it. The pocket checkpoint restores at the end of training,
  so this changes only what a later draw gets, never a result recorded today.
- **Format 2 gains `"rng"`.** `{"bit_generator": "PCG64", "state": "0x...", "inc": "0x...",
  "has_uint32": 0, "uinteger": 0}`, with the 128-bit integers as hex strings, since JSON readers
  outside Python lose precision on large integers. A file without it loads with an entropy
  generator (D9). An ensemble's file nests one per sub-network.

## Pitfalls to design around

- **The golden run's dropout entries move.** It injects its weights, so init draws don't show, and
  its shuffles stay on the same stream (D6). Only the numpy, Rust and pure-Python dropout entries
  change. They are re-recorded in the stages that switch them, with the PR saying which entries
  moved and why. Every other entry must stay bit-identical.
- **Recorded study results were drawn from the legacy stream.** The batch-size studies'
  docstring numbers and `docs/optimizations` baselines came from `np.random` weights. A rerun
  differs by seed noise. Don't rerun them for this plan; say so where they are quoted.
- **numpy may change `Generator`'s methods between versions.** The PCG64 stream itself is fixed,
  but NEP 19 lets `Generator`'s algorithms change. The crate's CI runs the parity tests against the
  latest numpy and will catch it.
- **FMA contraction**, as for the legacy stream ([rng-audit.md](rng-audit.md)): `uniform` must not
  fuse. Don't pre-emptively add `mul_add`.
- **Draw order stays part of the contract.** One call draws in C order on the calling thread. The
  generator never moves into the kernels' worker threads.
- **128-bit arithmetic.** Rust has `u128`. The pure-Python port uses Python ints masked to 128 bits.
  The LCG's multiplier and the XSL-RR rotation are the places a port goes wrong; test with seeds
  whose state crosses the top bit.
- **The pure-Python networks still draw in another order.** They draw weights node by node, and
  their dropout draws one value per node. Same stream family is not the same values; matching them
  stays in [rng-audit.md](rng-audit.md)'s open work.

## Stages

A crate stage is a PR in the crate repo, then a "Bump rust/" PR here, as in earlier workplans.
Every stage passes `./cli test`, `./cli lint` and the golden check, except the entries listed as
moving.

### Stage 1: PCG64, SeedSequence and `pa.Generator` (crate)

1. In `random.rs`, add `SeedSequence` (entropy as int, sequence or `None`; `spawn_key`;
   `generate_state`; `spawn`) and `Pcg64`.
2. Add `pa.SeedSequence`, `pa.Generator` (`random(shape)`, `uniform(low, high, shape)`,
   `bernoulli_mask(p, shape)`, a `state` property with a setter) and `pa.default_rng(seed=None)`,
   which accepts an int, a sequence, `None` or a `SeedSequence`. Raise numpy's exception types for
   seeds numpy rejects (a negative int, for one).
3. Give the fused `layer_dropout_forward*` an optional `rng: Generator`. Without one they draw from
   the global MT19937, as now.
4. `rust/tests/test_random_pcg64_parity.py`, bit-identical against numpy: many seeds (0, 1, 2^32,
   2^64 + 1, 2^127, sequences, spawned children); 1-D and 2-D shapes; interleaved `random`,
   `uniform` and masks; the fused masks with `rng`; the state read from numpy and set in the
   crate, and back, continuing identically; `None` differing between processes. The existing MT19937
   parity tests stay as they are.
5. `scripts/rng_audit.py quality | time` gain a crate PCG64 row. The PR quotes both against numpy's
   PCG64.

Done when the crate CI passes with the parity tests. Then bump `rust/`; the golden run is
bit-identical.

### Stage 2: the pure-Python PCG64 (indrajala-ml)

1. `indrajala_ml/pcg64.py`: `SeedSequence`, `Pcg64Generator` with scalar `random()` and
   `uniform(low, high)`, a `state` property in numpy's layout, and `default_rng`. Backend-free; no
   numpy import.
2. `tests/test_pcg64.py` against numpy, as stage 1's parity tests.

Pure Python only, so no A/B. Nothing uses it yet; the golden run is bit-identical.

### Stage 3: numpy and Rust networks own their generators

1. `default_rng` on both backends; `random_layer` and `random_weights` take `rng`.
2. `network.rng`, set by `randomized(..., seed=, rng=)`, by assignment, or from entropy (D9). Its
   setter reaches the dropout layers. `DropoutArrayLayer` draws from `rng.random`;
   `DropoutRustArrayLayer` passes `rng` to the fused ops.
3. Ensembles spawn their sub-networks' generators.
4. A parity test over every array network class: the same seed gives bit-identical numpy and Rust
   init, masks and `learn_batch` steps.
5. The golden run sets each network's generator from `SEED` instead of `seed_everything`. Re-record
   the numpy and Rust dropout entries only.
6. A/B: `prepared_dataset_timing` has no dropout network, so add a dropout arm or a probe in this
   PR (CLAUDE.md), then A/B both backends.

`backend.seed` stays until stage 7, for the tests not yet migrated.

### Stage 4: the pure-Python networks own their generators

`randomize()`, `fan_in_aware_init` and `DropoutNode` draw from the network's `Pcg64Generator`,
set as in stage 3. Re-record the pure-Python dropout entry only. Pure Python only, so no A/B.

### Stage 5: the trainers' shuffles

The trainers, `dataset_utils`, `batch_size_scaling`, `benchmark_sweep` and `ensemble_train` take a
`random.Random` (D6), entropy-seeded when none is passed (D9). The golden run passes
`random.Random(SEED)` and stays bit-identical. A change to the trainers' training path, so A/B
with `prepared_dataset_timing`.

### Stage 6: the generator's state in checkpoints and files

1. `Checkpoint.rng`; `restore_checkpoint` sets it.
2. Format 2's optional `"rng"` field, with a cross-backend test: save on numpy, load on Rust, and
   the next masks are identical; pure Python round-trips its own.
3. The README's Saving and loading section and `format2.py`'s docstring.

The golden run's `loaded` entries compare snapshots, which don't include the generator, so they
stay bit-identical.

### Stage 7: run checkpoints, and the globals removed

Two PRs: 7a is items 1 and 2, 7b items 3 and 4.

1. `RunCheckpoint` (D10), `resume_from=` on `train_backprop_network_mini_batch`, and its JSON
   file.
2. A resume test on all three implementations, with dropout and a warmup schedule: train k
   epochs, save both files, load them in a new process, finish, and match an uninterrupted run bit
   for bit.
3. Migrate the remaining tests and scripts (about 200 seeding calls) and delete `backend.seed` and
   `seed_everything`. A grep in the PR shows that `indrajala_ml` no longer calls `np.random.*`,
   the module-level `random.*` draws or `pa.seed`/`pa.uniform`/`pa.random`/`pa.bernoulli_mask`.
   This step may split into more PRs by test area.
4. [rng-audit.md](rng-audit.md): close "Three global states" and the open-work items this plan
   finished; [next-steps.md](next-steps.md): drop "Saving RNG state".

Then retire this workplan.

## Out of scope

- Matching the pure-Python networks' values with the array networks' from one seed (draw order).
- Normal draws, integer draws and broadcast `low`/`high`. Nothing here uses them.
- Other bit generators (Philox, SFC64, PCG64DXSM).
- Changing the crate's legacy module functions (D7).
