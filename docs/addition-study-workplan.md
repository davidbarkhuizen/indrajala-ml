# Workplan: the smallest network that adds (a study)

**Status: D1, D2 and D4 settled by the owner; D3 and D5-D11 settled by the agent as recommended
(2026-10-10). Stages 1 (this plan), 2 (the catalogue) and 3 (the candidates and the harness)
done.**

Which is the smallest network, built from indrajala-ml's existing layers and networks, that
reliably internalises addition: that adds two or three base-3 numbers it never saw, through every
carry structure, with the algebraic properties addition has? Every viable candidate is trained over
a size ladder, and each is judged against one catalogue of properties, by an automated harness that
runs unattended and reports at the end.

## Why

Addition is the smallest task with a known algorithm whose hard part is exactly what a network
must discover: the carry, which can depend on every column below. Its input space is too large to
memorise past a few digits, its carry structures are few enough to test every one, and its
algorithm's parts are known, so a failure can be named (which property, which carry pattern), not
only counted. It measures what each of the framework's primitives contributes (attention against a
per-token FFN, the causal mask, the token encoding) at the smallest sizes, where they matter most.

## The task (D1, D2)

- **Base 3, three operand slots.** Operands `a, b, c` of `n` base-3 digits each, `0 <= a, b, c <
  3^n`; the sum `< 3^(n+1)`, so `n + 1` digits. Two-operand addition is the case with a zero in one
  slot. One network must have every property (D2).
- **The column view.** Column `t`'s digit sum `s_t = a_t + b_t + c_t` is in `0..6`; with the carry
  in `k_t` (in `0..2`), the sum digit is `(s_t + k_t) mod 3` and the carry out
  `floor((s_t + k_t) / 3)`, again in `0..2`. As a map from carry in to carry out:

  | `s_t` | carry 0 | carry 1 | carry 2 | kind |
  | --- | --- | --- | --- | --- |
  | 0 | 0 | 0 | 0 | reset to 0 |
  | 1 | 0 | 0 | 1 | depends |
  | 2 | 0 | 1 | 1 | depends |
  | 3 | 1 | 1 | 1 | reset to 1 |
  | 4 | 1 | 1 | 2 | depends |
  | 5 | 1 | 2 | 2 | depends |
  | 6 | 2 | 2 | 2 | reset to 2 |

  The decimal two-operand adder's kill, generate and propagate become three resets and four
  dependent columns. **The carry into column `t` is fixed by the nearest reset column below `t`,
  composed through the dependent columns between them**; their count is the carry's **distance**
  `L` (0 when the column below is a reset). An `n`-digit addition's carry structure is its pattern
  of column sums, one of `7^n`.
- **The target algorithm.** A ripple-carry adder computes the carries one after another, in depth
  `n`; a network of fixed depth can't. The algorithm a transformer can express is the parallel
  (carry-lookahead) one: classify each column by its sum (local), find the nearest reset below each
  column (attention: the most recent token of a kind), and compose the maps between (more layers, or
  a wider head, as `L` grows). Its logical components, each a property below (M1-M4): the column
  sum mod 3, the carry out of a reset, the carry over a distance `L`, and the final carry. The
  study's prediction, to confirm or refute: one causal layer reaches some distance `L*`, and a
  second extends it to `n - 1`.

## The property catalogue (D2)

A property is the contract the final network must meet. Each has five parts: a **statement**; a
**generator** of cases consistent with it (seeded, apart from the training data's seeds); a
**verifier**, an oracle (Python's `+`) or a relation among the network's own answers; a
**coverage** (exhaustive, or sampled with a count); and a **pass bar** (D9). On a failure the
verifier **shrinks** to a minimal failing case: the shortest carry pattern, then the smallest
digits. `f(a, b, c)` is the candidate's decoded answer, `n + 1` digits.

| id | statement | generator | verifier | coverage |
| --- | --- | --- | --- | --- |
| **B1** adds three | `f(a, b, c) = a + b + c` | uniform triples | oracle | exhaustive at `n = 4` (531,441); 100,000 sampled at `n = 6` |
| **B2** adds two | `f` with a zero in any one slot is the two-operand sum | uniform pairs, the zero in a uniform slot | oracle | as B1, over pairs |
| **B3** held out | B1 on the hashed partition training never draws (D8) | uniform triples in the partition | oracle | 100,000 sampled |
| **A1** commutativity | all 6 orders of `(a, b, c)` give one answer | triples, distinct and with repeats | relation | 10,000 triples × 6 |
| **A2** associativity | `f(f(a, b, 0), c, 0) = f(a, f(b, c, 0), 0) = f(a, b, c)` | triples of `n - 1` digits (each partial sum fits `n` digits) | relation; a partial sum whose top digit isn't 0 fails | 10,000 triples |
| **A3** identity | `f(a, 0, 0) = f(0, a, 0) = f(0, 0, a) = a` | uniform `a` | oracle | exhaustive at `n <= 6` (729) |
| **M1** column sum | with every `s_t <= 2` (no carries), each digit is `s_t` | uniform patterns over `{0, 1, 2}^n`, digits drawn per column sum | oracle | every pattern × 4 |
| **M2** carry from a reset | the carry out of a reset (0, 1 or 2) reaches the next column | a reset at every column, every value; only the next column's digit checked | oracle | every column × value × 16 |
| **M3** carry over distance `L` | the carry into a column at distance `L` | every `L` in `1..n` (`L = n`: down to column 0's carry in), every carry value, every dependent map in between; only the digit at distance `L` checked | oracle | 1,000 per `L`; reports `L*`, the largest `L` with every case at `1..L` right |
| **M4** final carry | sum digit `n`, carry out of the top column, 0, 1 or 2 | top columns' patterns giving each | oracle | 1,000 per value |
| **M5** every carry pattern | every one of the `7^n` column-sum patterns | concrete digits per pattern | oracle | exhaustive over patterns: `7^4 × 4`, `7^6 × 1` (117,649) |

B are the definition of success, A the algebra (implied by full correctness, so they catch
structured errors cheaply, with no oracle), M the algorithm's components (targeted strata of B, so
a failure names its component). The catalogue is code (Stage 2); a property added later is a new
entry, run against every saved result by `report`.

## The candidates (D4)

A candidate is an **encoder** (`(a, b, c)` to its input), a **network spec** with its size knobs,
and a **decoder** (its output to the sum's `n + 1` digits). The catalogue only sees `f(a, b, c)`,
so every candidate meets every property, and a later candidate is one more entry. Every one is
built from today's specs, on the Rust backend.

| # | candidate | shape | encoding | size knobs | it tests |
| --- | --- | --- | --- | --- | --- |
| 1 | **Column-aligned causal transformer** | `sequence` | `T = n + 1` tokens, least significant column first, a token per column, `9 a_t + 3 b_t + c_t` (vocabulary 27; the last column `(0, 0, 0)`); label `t` the sum's digit `t` | layers, `d`, heads, FFN width | the parallel carry: the main contender |
| 2 | … **unordered tokens** | `sequence` | a token per column's multiset `{a_t, b_t, c_t}` (vocabulary 10) | as 1 | commutativity in the encoding: a smaller embedding |
| 3 | … **non-causal** | `sequence` | as 1, `Attention(causal=False)` | as 1 | whether the mask's structure helps |
| 4 | **FFN-only per token** (control) | `sequence` | as 1, no attention | `d`, FFN width, blocks | the column sums alone: it must fail every carry |
| 5 | **String-format causal transformer** | `sequence` | `a+b+c=` then the sum, every number least significant digit first, one character a token (vocabulary 5); every token in the loss; scored on the answer's `n + 1` positions | as 1 | the GPT-style setting, the alignment not given |
| 6 | **Dense, one network per sum digit** | `multiclass` × `(n + 1)` | the `3n` digits as values `/ 2`; digit `i`'s network a softmax over 3 | width, depth, activation | the threshold-circuit construction; its size the sum of its networks' |
| 7 | **Dense regression, digits in** | `single_output` | the `3n` digits; the target `(a + b + c) / (3^(n+1) - 1)`, decoded by rounding | width, depth | how far precision reaches |
| 8 | **Dense regression, scalars in** (control) | `single_output` | `a, b, c` scaled to `[0, 1]`; as 7 | width, depth | the trivial linear case, and its precision |

Scoring 5 on its answer positions with the true digits before each (teacher forcing) is exact for
greedy decoding: if every position's argmax is right given the right prefix, greedy decoding
produces the right answer. Its operand tokens are random, so their loss doesn't go to 0; it is
reported apart.

Excluded on their merits: the whole sum as one class (`multiclass` over `3^(n+1)` classes), a
lookup table that can't answer an unseen sum. Excluded for missing infrastructure: see the last
section.

## The harness (D10)

`scripts/addition_study.py`, built so that a sweep runs from one command to its report with nobody
watching:

- **`run CONFIG`** takes a JSON config (candidates, ladders, `n`, seeds, budget, workers) and runs
  it in worker processes. **Each run** (candidate, size, `n`, seed) **writes its own result file**,
  atomically (write, then rename), with its property results, `L*`, parameter count, epochs and wall
  time, the commit, the crate's `.so` hash and the machine. A rerun skips finished runs, so a
  sweep that dies (a reboot, a dropped session) **resumes** where it stopped; a run that raises is
  recorded with its traceback and retried once.
- **The search** (D6) chooses the next runs from the results so far: no hand-picking between
  batches.
- **Early stopping**: a run evaluates a fast screen of the catalogue each epoch (a spread of each
  property's cases); a passing screen triggers the whole catalogue, and the run stops when that
  passes too (a screen alone missed a rare failure in testing), at the budget, or after a plateau
  (D7). The whole catalogue runs once more at the end, its failures shrunk.
- **A rung is decided as soon as its seeds settle it** (4 passing, or 2 failing of 5): its
  remaining seeds aren't run.
- **Every run's trained model is kept:** its network(s) in format 2 under `models/<run>/` with a
  manifest (candidate, `n`, size), passing or not; `load_model` rebuilds it as an adder, and
  `check MODEL` runs the whole catalogue against it again. A failing model is kept for later probes
  (Excluded for missing infrastructure).
- **`status`** prints finished, running and failed runs; **`candidates`** lists the eight and their
  ladders; **`report`** builds the tables (Findings)
  from the result files at any time, mid-sweep included, as Markdown for the PR.
- **One command on `jebel`:** `nohup ... run CONFIG; ... report` in the background (measurement.md
  §4's pattern); its exit is the signal, its last lines the report.
- **The harness is tested without training:** oracle candidates (a correct adder; one that drops
  carries; one whose carry reaches `L` columns; a non-commutative one) stand in for networks, so
  the catalogue's verifiers, shrinking, the search and resuming are tested by `./cli test`; each
  broken adder must fail exactly its properties.

## Decisions

Each lists the options considered, with pros and cons, and the choice.

- **D1. Base and operands. Settled by the owner: (c).**
  - (a) Base 2. Pros: the smallest vocabulary (8 with three slots), the circuit literature's base.
    Cons: a smaller carry structure (four column sums, `0..3`).
  - (b) Base 10. Pros: the familiar case. Cons: a vocabulary of 1,000 with three slots: the
    embedding is most of every small model.
  - (c) *Chosen.* Base 3. Pros: carries of 0, 1 and 2, seven column maps (three resets, four
    dependent), a vocabulary of 27. Cons: more expensive than base 2.
- **D2. One network, three slots, properties as contracts. Settled by the owner: (a).**
  - (a) *Chosen.* One network with three operand slots, two-operand addition as a zero slot; the
    property catalogue, each property a generator, a verifier, a coverage and a pass bar. Pros: the
    final network has every property; associativity and commutativity are tested on it. Cons: a
    bigger model than a two-operand adder.
  - (b) A two-operand network, "adds three" as `f(f(a, b), c)`. Pros: smaller. Cons: adding three
    is the procedure's property, not the network's; widths must be padded.
- **D3. Widths. Settled: (a).**
  - (a) *Chosen.* `n = 4` and `n = 6`. `n = 4` is exhaustive (every triple, 531,441, so B1 there
    claims the whole space); `n = 6` is exhaustive over carry patterns (117,649) and too large to
    memorise (`3.9 × 10^8` triples), so it is the headline. Pros: one exhaustive claim each way;
    the size threshold's growth with `n` shows from two points. Cons: two points only.
  - (b) `n = 6` only. Pros: half the runs. Cons: no exhaustive check of the whole space.
  - (c) `n = 4, 6, 8`. Pros: a curve of threshold against `n`. Cons: `7^8` patterns (5.8 M) sampled,
    not exhaustive, and the most expensive runs; a later sweep if 4 and 6 differ.
- **D4. The candidates. Settled by the owner:** all eight above.
- **D5. Size ladders.** Settled:
  - 1-3 and 5: layers `{1, 2, 3}`, heads `{1, 2}`, `d` in `{4, 6, 8, 12, 16, 24, 32}` (`d`
    divisible by the heads), FFN width `2d`. Pros: the region where the threshold should sit
    (the sequence study's `d = 64` is far above it), every rung a real point. Cons: no FFN-width
    knob; added if the threshold sits at the ladder's edge.
  - 4: one and two blocks, `d` in `{8, 16, 32, 64}`, FFN `2d` (a control: two sizes per depth would
    do; four show it fails at every size).
  - 6: depth `{1, 2}` hidden layers, ReLU, width `{2, 4, 8, 16, 32, 64}` per digit network.
  - 7 and 8: depth `{1, 2}`, ReLU, width `{8, 16, 32, 64, 128}`.
  - **Size is the trainable parameter count**, every table included (embedding, position, layer
    norms, biases), counted from the built network.
- **D6. The search. Settled: (a).**
  - (a) *Chosen.* Per candidate and per fixed shape (layers and heads, or depth), bisect the
    ladder on success (D9), then run the rungs either side of the threshold to confirm it. Pros:
    `log` of the ladder's runs; the boundary itself measured. Cons: assumes success grows with
    size; small networks' optimisation can break that; the confirming rungs and the report's
    per-rung rates show where it does.
  - (b) The full grid. Pros: the whole success curve. Cons: 2-3 times the runs, most far from the
    threshold.
- **D7. Training. Settled: (a).**
  - (a) *Chosen.* Adam, mini-batches of 64, **a fresh random training set each epoch** (`2^16`
    examples, a new seed per epoch: one epoch a call, the network's optimizer state carried
    across), drawn as a third uniform triples, a third uniform over carry patterns (so long
    distances aren't rare), a third two-operand cases; the hashed held-out partition never drawn
    (D8). The rate per candidate from a short probe (`{0.001, 0.003, 0.01}`, Stage 4); the epoch
    budget and the plateau rule from Stage 4's timing, written back here. Pros: unlimited data,
    so the only limit is capacity, the question's; the pattern third fixes uniform sampling's
    rarity of long carries. Cons: a fixed budget can call a network too small that is only slow;
    the confirming rungs (D6) and the report's epochs-to-pass show how near the budget each pass
    was.
  - (b) One fixed training set. Pros: a sample-efficiency curve. Cons: mixes overfitting into
    the size question; a later study.
- **D8. Held out. Settled: (a).** A triple is held out when a hash of `(a, b, c)` (the ordered
  triple, and so each of its orders: sorted first) falls in one tenth. Pros: deterministic, nothing
  stored, commutativity can't leak a held-out triple through its reorder. Cons: B1's exhaustive
  check at `n = 4` includes trained triples, so it claims correctness on the space, B3 alone the
  generalisation. At `n = 2` the tenth removes whole contexts (every order of a triple is one
  column-1 multiset over one column-0 carry), and a network that fits the rest fails exactly those
  (found in Stage 3's tests): the study's widths start at 4.
- **D9. The pass bar. Settled: (a).**
  - (a) *Chosen.* A network passes a property at 100% of its cases; a size **succeeds** when 4 of 5
    seeds pass every property. Pros: "reliably" made exact; a seed-lucky size doesn't count.
    Cons: one wrong case fails a run; the report gives each property's accuracy too.
  - (b) 99.9%. Pros: tolerant of a rare slip. Cons: a missed long carry is exactly the rare slip.
- **D10. The harness.** Settled as above. Pros: unattended, resumable (the attention dropout
  sweep had no resume and wrote its report only at the end), extensible by catalogue entries and
  candidate entries. Cons: more code than a single script; its tests carry it.
- **D11. Where it runs. Settled.** Development and the calibration on `pyramidon`, the sweep on
  `jebel` (nothing else running), Rust backend, `OPENBLAS_NUM_THREADS=1`, 6 workers. Tier 0: no
  timed path changes.

## Stages

1. **This workplan.**
2. **The catalogue** (done; `indrajala_ml/data/addition_data.py`): digits, column sums and carry maps,
   the hashed partition, every property's generator, verifier and shrinking, the training mixture;
   tests: each generator's cases satisfy its statement, the oracle adders pass and fail exactly as
   they should.
3. **The candidates and the harness** (done; `indrajala_ml/studies/addition_study.py`,
   `scripts/addition_study.py`): the eight encoders, specs and decoders (candidate 5 decoded
   greedily, a digit at a time); `run`, `status`, `report`, `check`, `candidates`; resuming,
   retries, the search, early stopping, the models kept; tests with stand-in candidates (the
   search's choices, resuming, a run that raises, the report) and every real candidate trained,
   saved and reloaded.
4. **Calibration** on `pyramidon`: every candidate at one middle size at `n = 4`, the rate probe,
   timing; the budget and plateau rule written into D7.
5. **The sweep** on `jebel`, unattended; the findings into the script's docstring (findings, the
   tables, the protocol, as `sequence_study.py`'s); the PR with the report.
6. **Docs**: a README section, next-steps (this plan's leftovers and the section below), the
   workplan retired.

## Findings (Stage 5)

Per candidate: the smallest succeeding size (parameters, and its shape), the success rate per rung,
`L*` per rung, each property's pass rate, epochs to pass; across candidates, one table of the
thresholds. To be filled.

## Excluded for missing infrastructure

Options left out because the framework lacks a piece today, what the piece is, and whether a
planned step provides it:

| excluded option | why it matters | missing piece | planned? |
| --- | --- | --- | --- |
| **Recurrent adder** (one column a step, the carry as state) | the minimal algorithm (ripple carry); any length by construction | a recurrent layer with backpropagation through time, in all three implementations and the crate | no |
| **Weight-tied (looped) transformer** | depth that grows with `n`: ripple or prefix steps by iterating one block | blocks sharing weights, and a variable iteration count | no |
| **Length generalisation** (train at `n`, test at `n + k`) | separates an algorithm from a fixed-width circuit | positions not tied to `T` (rotary, step 8; or no `Position()`, which validation allows today) **and** loading trained weights into a network of another `T` (format 2 records the input shape) | positions: step 8; the reload: no |
| **Loss on answer tokens only** (candidate 5 cleanly) | the operand tokens' unpredictable loss is noise in 5's training and its reported loss | per-token counted targets | step 11 (segments, D6: uncounted targets) |
| **Variable-length answers**, decoded autoregressively | numbers of mixed widths without padding to `n + 1` | generation (step 9), and step 11's counted targets | steps 9, 11 |
| **One dense network for all sum digits** (6 as one network, shared hidden layers) | a dense candidate whose carry computation is shared, so its size is comparable to 1's | an output of several independent softmaxes (or sigmoids) on a flat network | no |
| **Supervised carries** (auxiliary outputs for each column's carry or kind) | trains, and tests directly, the algorithm's components | several output heads and losses on one network | no |
| **Probes of the components** (a linear probe for carry in, per column, on hidden activations) | whether a passing network computes the carries as the target algorithm does, not only its outputs | an accessor for a layer's activations on a batch (not checked; may be small); the probes themselves are `multiclass` networks | no |
| **1-D convolution along the tokens** (a local, ripple-like candidate with per-token outputs) | carries over a receptive field: a different inductive bias from attention | conv over a token sequence in the `sequence` shape | no |
| **Rate schedules beyond the current callable, AdamW, clipping** | small networks may train only with them; a "too small" verdict could be an optimiser's | step 10's training recipe | step 10 |

Each becomes a next-steps entry when this plan retires; a study rerun with a candidate it enables
is a new catalogue run, not new harness code.
