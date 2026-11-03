# Workplan: a module tree for the crate

**Status: drafted 2026-11-03 at the owner's direction; no code yet.** The owner: "the rust crates
namespace is flat - give the crate some proper structure". The owner delegates the decisions (each
below with its options' pros and cons).

## Where it stands

The core (`indrajala-math`, `rust/core/src/`) has 20 top-level modules, one per file, side by side:
`activation`, `attention`, `batch_norm`, `conv`, `cr_exp`, `erf`, `factored`, `fused`, `generator`,
`layer_norm`, `linalg`, `mnist`, `ops`, `program`, `random`, `recurrent`, `scalar`, `tokens`,
`ufuncs`, plus the root's `Array`, `Shape` and `Error`. The bindings (`rust/src/`) mirror them file
for file, and register about 150 functions and 6 classes into one flat Python module.

What the flat list hides, and what it costs:

- **No layers.** Scalar rules (`scalar`, `cr_exp`, `erf`), array ops (`ops`, `ufuncs`, `linalg`),
  generators (`random`, `generator`), layer passes (`fused`, `attention`, `conv`, ...) and the
  window program sit at one level; nothing says which may use which.
- **The dependencies run both ways.** `ufuncs` imports from `fused` (`activation_mask`,
  `dropout_hidden_delta_from_downstream`) while `fused` imports from `ufuncs` (`array_softmax`,
  `sum_axis0`); `activation` imports `tanh_in_place` from `recurrent`; `attention` and `recurrent`
  build on `fused`'s dense passes. Shared kernels live in whichever layer needed them first.
- **Names repeat their module:** `attention::attention_forward_batch`,
  `layer_norm::layer_norm_forward`, `conv::conv_downstream_batch`.
- **Mixed modules:** `fused` (1,003 lines) holds the dense layer's passes, the activations' and
  dropout's, softmax, the gated layer and four optimizers' updates; `batch_norm` holds the
  bias-free linear layer; `ufuncs` and `ops` split elementwise work between them; `random` and
  `generator` are numpy's two generator families under unrelated names.
- **No home for the next work:** the multithreading plan (D3, D5) adds a pool and `CpuInfo`; each
  new layer adds a 21st, 22nd top-level module.

Both crates are published at 0.1.0 (2026-10-31), with no promise of stability before 1.0; PyPI's
next release already carries breaking changes (`rust/pypi-pre-release-notes.md`).

## The IDs, in time order

| ID | When | Short label | What it is |
|---|---|---|---|
| D1 | 2026-11-03 | layers, downward only | the core's tree has four layers, each using only those below it: `num` < `array ops` (`ops`, `linalg`, `random`) < `nn` < `program` |
| D2 | 2026-11-03 | names without stutter | a function is named within its module: `nn::attention::forward_batch`, not `attention::attention_forward_batch` |
| D3 | 2026-11-03 | Python stays flat | the Python module keeps its flat names; only the bindings' Rust source mirrors the tree |
| D4 | 2026-11-03 | no old paths | no re-exports or deprecated aliases at the old paths; the core's next release is 0.2.0 |
| D5 | 2026-11-03 | moves, not changes | no arithmetic, signature (beyond the path) or kernel changes in this plan; the golden run and a tier 1 A/B check it |
| D6 | 2026-11-03 | before multithreading | this plan runs before the multithreading plan, which then builds on the tree |

## The tree (D1)

```
indrajala_math
├── Array, Shape, Error          root re-exports (array, error stay private modules)
├── num                          f64 scalar rules, no Array
│   ├── scalar                   relu, max, powi (today's scalar)
│   ├── exp                      correctly rounded exp, sigmoid, tanh (today's cr_exp; tanh_in_place from recurrent)
│   ├── erf
│   └── activation               the Activation enum (it works on slices), gelu, gelu_tanh, their derivatives
├── ops                          elementwise over Arrays
│   ├── arith                    same-shape, scalar and broadcast-row arithmetic (today's ops)
│   ├── unary                    exp, erf, softmax, sum_axis0, argmax (from ufuncs)
│   └── mask                     the activation, sigmoid and dropout masks (from ufuncs and fused)
├── linalg                       matmul family, outer, threading and kernel overrides (unchanged)
├── random
│   ├── legacy                   numpy's np.random MT19937 global (today's random)
│   └── generator                numpy's default_rng: PCG64, SeedSequence (today's generator)
├── nn                           one function per layer pass
│   ├── dense                    sigmoid and activation layers: forwards, deltas, gradients, skip variants;
│   │                            affine; linear (from batch_norm)
│   ├── dropout                  the dropout layer's passes (from fused)
│   ├── softmax                  the softmax output layer's passes (from fused)
│   ├── gated                    the gated layer (from fused)
│   ├── optim                    sgd, momentum, l2, adam updates (from fused)
│   ├── norm                     batch_norm, layer_norm
│   ├── conv                     convolution, max pooling
│   ├── attention                self (attention.rs's self-attention), cross
│   ├── tokens                   patches, token mean, embedding, token dropout
│   └── recurrent                state input, FiLM, pooling, GRU, codes, soft codes, factored, the recurrence step
├── program                      the window program (composes nn)
└── data
    └── mnist
```

A kernel two layers share moves to the lowest layer that uses it, so every `use crate::` points
down the list or within a module. A small test (`core/tests/layers.rs`) greps the sources and fails
on an upward import, so the rule outlives this plan. The multithreading plan's pool and `CpuInfo`
(its D3, D5) go in a fifth, bottom layer, `sys`, beside `num`.

`batch_norm` keeps its own private activation enum (ReLU and sigmoid by name), a duplicate of
`num::activation`'s in part; merging them would be a change, not a move (D5), so it goes to
next-steps.md.

Large single-concern modules (`linalg`, 1,207 lines; `program`, 1,326) stay one file each: their
size is one concern's, and splitting them is not this plan's. `attention` (1,562) splits in two,
self and cross, because it holds two layers.

## Decisions

**D1, layers, downward only.**
- *(a) The layered tree above (chosen).* Pros: says what each module may use, so a new kernel or
  layer has one obvious place; ends the `fused`/`ufuncs` cycle; gives the thread pool a home; the
  test keeps it so. Cons: every file moves, and history needs `git log --follow`; shared kernels
  move away from the layer that first used them.
- *(b) Group by family only (`nn::`, `math::`), no layering rule.* Pros: fewer moves. Cons: the
  cycles stay, and so does the question of where a shared kernel goes.
- *(c) Keep flat, only split `fused`.* Pros: least churn. Cons: not what was asked; the list keeps
  growing with each layer.

**D2, names without stutter.**
- *(a) Drop the module's name from its functions (chosen).* Pros: Rust's convention (clippy's
  pedantic `module_name_repetitions` lint); paths read as structure
  (`nn::norm::layer_norm::forward`); shorter. Cons: a Rust name no longer greps to its Python name;
  the bindings map each one, by hand, once.
- *(b) Keep today's names under the new paths.* Pros: one name across Rust, Python and the docs.
  Cons: `nn::attention::attention_forward_batch` repeats itself, and a structured tree with flat
  names gets half the benefit.

**D3, Python stays flat.**
- *(a) Keep `indrajala_math_rust`'s flat, prefixed names (chosen).* Pros: indrajala-ml's 115 files,
  the studies and the demos are untouched; the `.pyi` stub and stubtest stay simple; the names
  already carry their family (`layer_norm_forward`); PyO3 submodules aren't importable as
  `import indrajala_math_rust.nn` without registering them in `sys.modules` by hand. Cons: the
  structure is Rust-only; Python's ~150 names stay in one list.
- *(b) Mirror the tree as Python submodules.* Pros: one structure in both languages. Cons: a
  breaking PyPI change across every call site, the `sys.modules` workaround, and stubtest's
  allowlist grows; the Python side's structure already lives in indrajala-ml's layer classes.

The bindings' Rust (`rust/src/`) mirrors the core's tree, and each area registers its own functions
(`nn::attention::register(m)`), so `lib.rs` stops being one list of 150 `add_function` lines.

**D4, no old paths.**
- *(a) Move, with no aliases; the core's next release is 0.2.0 (chosen).* Pros: one path per item
  (CLAUDE.md, structure first); nothing to retire later; the crates.io crate has had 0.1.0 for
  three days and no known dependents. Cons: anyone on 0.1.0 who updates must change their imports.
- *(b) Deprecated re-exports at the old paths for one release.* Pros: a soft landing. Cons: a
  duplicate path for every item, docs.rs shows both, and a retirement step to remember.

**D5, moves, not changes.**
- *(a) Only paths and names change (chosen).* Pros: a mechanical diff, reviewable as moves; any bit
  or speed change is a mistake, so the checks have one expected answer. Cons: kernels that would
  benefit from a rewrite while they move (the multithreading plan's stage −1 items) wait.
- *(b) Rewrite as we go.* Cons: mixes a refactor with changes the golden run and A/B must judge
  separately.

Checks: `codegen-units = 1` already makes inlining independent of how rustc splits the crate
(Cargo.toml), so no timing should move; but this changes every timed path, so it is tier 1
(measurement.md §1): one `epoch_op_profile` A/B on jebel at `--order ONNO`, plus the golden run's
check (both machines' `golden_run.json` and `golden_run_rust.json`), and the crate's own `==` tests.

**D6, before multithreading.**
- *(a) Now, before the multithreading plan (chosen).* Pros: that plan touches every kernel and adds
  modules; it lands once, in the tree. Cons: delays it, though it is planned last anyway (its O1).
- *(b) After it.* Cons: the threading work moves twice.

## Stages

Each is its own PR, merged before the next. Stages 2 and 3 change the crate, so they land in
`indrajala-math-rust` first; stage 4 is the "Bump rust/" PR here.

1. **This workplan** (indrajala-ml).
2. **The core's tree** (crate PR): `git mv` each file to its place, split `fused`, `ufuncs`,
   `attention` and `batch_norm` as the tree says, move the shared kernels down, drop the stutter
   (D2), add the layering test, update the core's README, doc examples, tests and the bindings'
   `use` paths so it builds. `core/README.md` and `pypi-pre-release-notes.md` gain a crates.io
   section listing the moves (D4). `cargo doc`, clippy, the crate's tests.
3. **The bindings' tree** (crate PR): `rust/src/` mirrors the core, each area with its `register`;
   `lib.rs` calls them in order. The Python names are unchanged (D3); stubtest passes untouched.
4. **Bump rust/** (indrajala-ml): `./cli build-rust && ./cli test`, the golden run's check on
   pyramidon and jebel, the tier 1 A/B (D5). Then this workplan retires.

## Not in this plan

- Splitting `linalg` or `program` internally.
- Any kernel change (D5); the multithreading plan's stage −1 has those.
- Python submodules (D3 (b)).
