# Workplan: the window loop in the crate (the window program)

**Status: written 2026-10-16 at the owner's request ("what network infrastructure is still missing
in rust/numpy? we are probably now wasting time using python-only infra for long runs ... plan the
work in detail"). D1-D10 settled under the owner's delegation, each with its options' pros and
cons. Stage 0 not started.**

## Why

The studies' networks are `WindowRecurrentRustArrayNetwork`s: each layer's arithmetic is one crate
call, but the loop over windows, each window's walk through its layers, the training step's
backward pass over the windows (with its recomputed forward pass, W20), the examples, the labels
and reading the answers back are Python. At `k` 1 a 1,024-column input is 1,024 Python iterations
of about 7-10 crate calls each, on small arrays.

A first profile (2026-10-16, `pyramidon`, loaded by two studies, cProfile, so the Python share is
overstated; not a timing claim):

| path | network | crate share | the largest Python items |
| --- | --- | --- | --- |
| long test, U1 at `n` 128 and 1,024, 300 cases | fork probe, `p` 7 family (attention window) | 52-56% | `addition_data.digits` (about 8% at 1,024; `x // 3**t` per digit, quadratic in `n`), `Encoding.example` (about 9%), `argmax_token_rows` via `tolist()` (about 5%), the window loop |
| training batch, 64 examples at 64 columns | E2b' seed 1, step 7 (linear window) | about 70% | the window loop and the per-layer calls (about 25%) |

The long tests take about 2 h a network at `p` 7 (447,000 cases; 4,500-6,100 s of it at 1,024) and
about 100 min at `p` 5; they are now the slowest step between a study's end and its findings. The
`p` 11 ladder and the approximation claim will add dozens more.

## Goal

The Rust networks' window loop in the crate, as one call per pass over any window the specs
accept, factored as the recurrence is (each factor its own pooling, cell and codes), **bit for bit
the per-layer path's results**, so no study, golden entry or saved network moves; and the
Python-side data path of the long tests rid of its quadratic and per-element work. Measured before
and after with `ab.py`.

## Decisions (settled under the owner's delegation)

**D1. What the crate runs.**
- (a) **A window program: an interpreter of the window's layer list** (a sequence of op
  descriptors built from the specs, run per window, then the factored recurrence step), covering
  every layer kind a window may hold. Pros: one path for every architecture a study builds, now
  and later (a new layer kind adds one op to the program, which a test enforces); reuses the
  crate's existing per-layer functions, so the bits follow (D3). Cons: the most work up front; an
  interpreter's dispatch per op (negligible against the ops).
- (b) A hand-written kernel per architecture (the linear window, the attention window). Pros:
  fastest per architecture; least Rust. Cons: every new window needs new Rust; two kernels to keep
  equal to the per-layer path; blocks the next studies' windows.
- (c) Only the recurrence chain in the crate (pooling, cell, codes over all windows), the layers
  per call as now. Pros: little work. Cons: the layers are most of the calls; little gain.

**Chosen: (a).** The project's priority is an extensible framework (CLAUDE.md, Structure first).

**D2. How the program reads the parameters.**
- (a) **Passed in each call as one flat list, in the network's parameter order** (snapshot's).
  Pros: never stale (the optimizer replaces arrays, it doesn't mutate them); the program holds no
  Python references; about 40 references a call, negligible. Cons: a list built per call in Python.
- (b) The program holds `Py<RustArray>` references to the layers' arrays. Pros: no list per call.
  Cons: stale after every optimizer step unless refreshed, a bug class.
- (c) The crate owns the whole network, optimizer included. Pros: fastest training. Cons: a second
  network implementation; format 2, snapshot, the optimizer's per-layer apply all duplicated.

**Chosen: (a).**

**D3. Bits.**
- (a) **Bit-identical to today's per-layer Rust path**: the program calls the same crate functions
  in the same order, and the Python-side compositions it absorbs (the residual's add, Position's
  add, the poolings' gradient added to the downstream, the StateInputs' state deltas summed into
  `dC`, the output delta times `1 / W`) in the same order of operands. Pros: the golden run, every
  saved network's certificate and every study's reruns stay exactly reproducible; parity tests are
  exact equality. Cons: forgoes reassociation or fused arithmetic that would change the last bits.
- (b) Allow last-bit changes for speed; re-record the golden run. Pros: room for faster kernels.
  Cons: needs the owner's approval (measurement.md §8); breaks byte-identical reruns of past studies.

**Chosen: (a).** A faster kernel that changes bits is a later, separate decision.

**D4. Order of work.**
- (a) **Inference first** (forward over every window: long tests, screens, `drift`, `loss_batch`,
  `state_codes`, the certificate's `window_step`), then the training step. Pros: inference is the
  larger saving (the long tests) and the simpler half; its program is reused by the training step.
  Cons: two crate PRs and two bumps.
- (b) Both at once. Pros: one bump. Cons: one large PR; the long tests wait for the harder half.

**Chosen: (a).**

**D5. The training step's backward pass (W20 recomputes each window's forward pass).**
- (a) **Keep every window's forward results in the program and drop the recompute.** The
  recomputed pass is deterministic (its generator restored), so the kept values are the same bits.
  Pros: saves one of the two forward passes per window (the recurrence step alone is about a third
  of a batch). Cons: memory of every window's activations: at training lengths (up to `m + q - 1`,
  18-64 columns) and batch 32, of the order of tens of MB at the widest window used (computed
  exactly in stage 4 and written beside it); a bound the program checks (it refuses above a set
  budget, with the message naming it).
- (b) Recompute in Rust as Python does now. Pros: memory flat in `W`. Cons: the extra pass.
- (c) Keep at most every `r`-th window (checkpointing). Pros: memory and time between. Cons: a
  setting to tune; not needed at today's lengths.

**Chosen: (a)**, with the bound; (c) only if a study trains long windows.

**D6. Dropout's masks in the program.** With D5 (a), each window's masks are drawn once in its
forward pass from the network's crate `Generator`, in today's order, and kept; the generator's
state at the end is the state today's path restores (`after`). Nothing to choose beyond keeping
that order; a test checks the generator's end state and every mask against the per-layer path.

**D7. The numpy backend.**
- (a) **Unchanged: its per-window Python loop stays the reference** (and the snap wall stays
  numpy's, the snap-wall workplan's D8). Pros: one simple reference the program is tested
  against; nothing studies use runs on numpy. Cons: numpy stays slow (unused for long runs).
- (b) Vectorise numpy's loop too. Pros: none for the studies. Cons: work and risk for no use.

**Chosen: (a).** The per-layer Rust path also stays, as the program's parity reference in the
tests (not a runtime fallback).

**D8. Threads.**
- (a) **One thread in the program.** Pros: bits trivially kept; the machines already run several
  processes (studies, long-test workers). Cons: a single long test uses one core.
- (b) Split the examples (rows) across threads inside the program. Pros: rows are independent, so
  bits are kept and one long test could use several cores. Cons: competes with the processes
  already running; the crate's per-call thread spawn (i7-9700k notes, §4) is unmeasured.

**Chosen: (a)**; (b) is a candidate in next-steps, measured first.

**D9. The long tests' Python data path** (independent of the crate):
- (a) **Fix it in Python, in its own PR, first:** base-3 digits by repeated `divmod` (or a
  divide-and-conquer conversion) instead of `x // 3**t`; `Encoding.example` and `column_tokens`
  building tokens and labels in one pass without re-deriving digits; answers read with numpy over
  the crate array's values (one `tolist()` a chunk) until the crate's `argmax_token_rows` (stage 2)
  replaces it. Pros: quick, independent of the crate, every task gains (training data too).
  Cons: the training data's generation is touched, so its examples must be checked identical.
- (b) Leave it to the crate work. Pros: none. Cons: the crate doesn't build examples.

**Chosen: (a).** Its check: old and new examples equal over every task's catalogue at several
lengths, and the golden run.

**D10. What decides "worth it" between stages.**
- (a) **A measured gate after stage 0**: the crate stages (2-5) go ahead only if stage 0's
  breakdown, on the quiet machine, puts the window loop's Python and per-call overhead at 20% or
  more of a long test or of a training step. Pros: no large Rust work on a guess (memory:
  existence-first-compute). Cons: one more step before the Rust starts.
- (b) Go straight to the crate work. Pros: starts sooner. Cons: may build a large program for a
  small saving.

**Chosen: (a).** If the gate fails, stage 1 alone ships and stages 2-5 move to next-steps with
the measurement.

## Measurement

- **The probe:** `scripts/window_recurrent_timing.py` (today: `learn_batch` only) gains inference
  cases and the studies' cells, each a row of the probe contract (`ab.py --bench cmd`):
  - `forward` at the long-test shapes: the `p` 7 family window (attention) at `n` 1,024, rows
    `CHUNK_TOKENS // ((n + 1) * width)`; the E2b' linear window (`k` 1 `s` 16 `m` 16) at 1,024;
  - `learn` at E2b''s cell (`W` up to 18) and the division family's (`p` 7, the step's longest);
  - `long-test` end to end: 2,000 cases of U1 at 1,024 through `QueryModel.predict` and
    `task.evaluate` (examples, network, labels, checking), so stage 1's gain shows;
  - the control: the fixed crate matmul it times today.
- **Where:** timing on `jebel` (`ab.py remote --host jebel run ...`), development on `pyramidon`.
  `jebel` is held by the chain until the approximation claim's share ends (about 2026-10-17
  midday); A/Bs queue behind its lock. Stage 0's breakdown is a profile, not an A/B, and runs on
  `pyramidon` between studies.
- **Tiers:** stage 1 claims a speedup: tier 2 (6 passes) with the `long-test` row. Stages 3 and 5
  change a timed path and claim a speedup: tier 2 with the `forward` and `learn` rows. Each tier 2
  run archived after its PR merges (§10).
- **Correctness:** the golden run (it has each implementation's window-recurrent and continuous
  entries, dropout included) passes unchanged at every stage that touches the package; the parity
  tests below are exact equality.

## Stages

0. **Measure first** (indrajala-ml, one PR, tier 0). The probe's new cases (above), and a
   per-component breakdown of a long test and a training step on the quiet machine (crate calls,
   the window loop's Python, examples, labels, checking), recorded here. Settles D10's gate.
1. **The long tests' Python data path** (indrajala-ml, one PR, tier 2; D9). `addition_data.digits`,
   `column_tokens`, `Encoding.example`, the answers' read-back. Tests: old against new examples and
   labels equal on every task (addition, modular, the family's encodings) at lengths 1-64 and 1,024;
   golden run; A/B. A note in working-notes on the trap (`x // BASE**t` at long lengths).
2. **The window program, inference** (indrajala-math-rust, one PR). A `WindowProgram` built from op
   descriptors (one per layer kind a window may hold: Embedding, Position, StateInput, Dense with
   its activations, LayerNorm, Residual's body, Attention with its causal, dropout and rotary
   settings, Dropout, TokenMean, and whatever else `validate_window_specs` accepts by then),
   then the factored recurrence (`recurrence_step_forward`'s code, unchanged) per window. One call:
   `window_program_forward(program, X, parameters, C0, keep)` with `keep` choosing what comes back
   (the outputs; also each window's states, cells' outputs and code indices for `window_states`
   and `state_codes`). Also `argmax_token_rows` as a crate op (ties to the first, as today).
   Rust tests: the program against the per-layer functions called in sequence, exact equality,
   one test per op kind and the composed windows the studies use.
3. **Bump rust/ and use it for inference** (indrajala-ml, one PR, tier 2).
   `WindowRecurrentRustArrayNetwork` builds its program from its specs at construction (and on
   load); `forward_windows`, `window_states`, `state_codes`, `window_step`, `loss_batch` and
   `forward_rows` call it; numpy's network unchanged (D7). Tests: for every layer kind and the
   factored, cross and continuous recurrences, the program's outputs, states and indices equal the
   per-layer path's exactly (the per-layer path kept as a test helper); a test enumerating
   `layer_specs`' kinds fails when a window kind has no op. A saved study network (E2b', a division
   family network) gives the same certificate and the same long-test answers on 2,000 cases.
   Golden run; A/B.
4. **The window program, training** (indrajala-math-rust, one PR). `window_program_learn(program,
   X, Y, parameters, gradients, C0, rng, weight)`: forward over every window keeping its results
   and masks (D5, D6), then backward from the last window, the gradients accumulated into the
   arrays passed and returned; the memory bound refused above its budget. Rust tests: against the
   per-layer forward and backward functions in sequence (with W20's recompute), exact equality of
   every gradient and of the generator's end state.
5. **Bump rust/ and use it for training** (indrajala-ml, one PR, tier 2). `_accumulate_gradients`
   on Rust calls the program; the optimizer's per-layer apply unchanged; the snap wall's networks
   stay on numpy (D7). Tests: one training step's every gradient equal to the per-layer path's; a
   3-step study run byte-identical before and after (as #779 and #790 checked); golden run; A/B.
6. **Close** (indrajala-ml, one PR). A real long test and a short study run timed before and after
   on `jebel` (`ab.py`), the result here; working-notes (building a program, the memory bound);
   leftovers (D8 (b), D5 (c), a bits-changing kernel) to next-steps; this plan retired.

## Known limits

- The gain on training is bounded by the crate's share of a step (about 70% today): the program
  removes the Python and the extra forward pass, not the arithmetic.
- The certificate's breadth-first search stays Python (seconds a network); it gains only through
  `window_step`.
- Long tests remain one core each (D8); more workers are bounded by `pyramidon`'s 5 GB.
