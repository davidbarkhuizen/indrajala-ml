# Workplan: adders for numbers of any length (high level)

**Status: a high-level plan (2026-10-11), at the owner's request. It names the missing pieces,
their order and the decisions that frame them. Each piece gets its own detailed workplan before
any code, and this plan is revisited then. It isn't on the primitives roadmap yet; where its
pieces meet roadmap steps, those steps are named.**

**Revised by the owner (2026-10-12): milestone A is now a window-recurrent adder
(`docs/window-recurrent-adder-workplan.md`), recurrence along the columns, H3's option (b); its
first subplan, `docs/length-generalisation-workplan.md` (P1, P4, P5 and a
fixed-depth existence study), was terminated after its reading (next-steps, From length
generalisation). P1 is parked; P2, P3 and P7 are deferred; composing a fixed-length adder outside
the network is parked (Parked, below).**

The addition study (`docs/addition-study-workplan.md`) asks for the smallest network that adds
base-3 numbers of a fixed `n` digits. This plan asks what the framework lacks for an adder trained
on some lengths to add numbers of **any** length, including lengths it never saw and operands of
unequal lengths, while meeting the same property catalogue.

## What "any length" means (H1)

Two claims, of different strength:

- **Length generalisation.** Trained on lengths up to `n_train`, the network passes the catalogue
  at every tested length up to a multiple of it (say 4-10×). A fixed-depth network can do this,
  but only up to some length, which the study measures.
- **Arbitrary length.** The computation grows with the input (iterations, or a recurrence), so no
  test length is out of reach in principle; precision and cost are the only limits.

## Why the fixed-length adders don't extend

**The task needs unbounded carry distances.** From the addition workplan's table, restricted to
the carries a column can receive:
- a column with sum `s = 2` maps carry `0 -> 0` and `1 -> 1`, passing a 0-or-1 carry through
  unchanged;
- columns with `s = 4` or `s = 5` pass a 1-or-2 carry through unchanged.

A run of such columns of any length carries the value from below it unchanged, as decimal
addition's propagate does. So M3's distance `L` is unbounded, and the network must find the
nearest deciding column below each column at any distance. Softmax attention spreads its weight
as `T` grows unless its scores grow with `T` (Chiang & Cholak 2022, "Overcoming a theoretical
limitation of self-attention"). So a fixed-depth model that reaches `L* = n - 1` at the trained
length can still fail past it.

**The framework ties a network to one length `T`.** With the column encoding, `T = n + 1`:

| what | where | consequence |
| --- | --- | --- |
| attention layers are built for one `tokens`: its causal mask, its rotary table and its reshapes | `attention_array_layer.py` (`_set_up`), the Python and Rust layers likewise | a network runs at one `T` only |
| `Position` is a learned table, a row per token | the token layers | no row for a token past `T`; untrained rows if the table is enlarged |
| format 2 refuses a file whose input shape differs from the network's | `persistence/format2.py` | a model trained at one `T` can't be loaded at another |
| one `T` per network, so one per training batch | `SequentialArrayNetwork` | no training across lengths |
| the harness builds and judges one `n` per run; the catalogue's exhaustive coverage (B1 at `n = 4`, M5's `7^n`) doesn't scale | `addition_study.py`, `addition_data.py` | no property spans lengths |
| attention's cost and memory are `O(T²)` | every attention implementation | long test lengths are slow; step 12 (context length) tiles it |

**What already doesn't depend on `T`:**
- The crate's attention ops read `T` from their arrays and take the rotary table as an argument.
- Every weight except `Position`'s is per feature, not per token.
- The column encoding gives each column its own token, so digits are aligned by construction.
- Rotary positions (roadmap step 8) are relative and have no table to train.
- Candidates 6-8 (dense) take a fixed-size input and can't be extended. A dense column cell run
  as a recurrence is piece P7.

**What the literature reports** (to read in the subplans): learned absolute positions don't
extrapolate. Methods that extend addition past the trained length include:
- randomised positions (Ruoss et al. 2023, "Randomized Positional Encodings Boost Length
  Generalization of Transformers");
- FIRE positions with randomised positions and index hints (Zhou et al. 2024, "Transformers Can
  Achieve Length Generalization But Not Robustly"), where rotary positions do poorly;
- per-digit "Abacus" embeddings with looped blocks (McLeish et al. 2024, "Transformers Can Do
  Arithmetic with the Right Embeddings");
- position coupling (Cho et al. 2024);
- looped transformers with a step count that grows with the input (Fan et al. 2024, "Looped
  Transformers for Length Generalization");
- for arbitrary length, recurrent convolution (Kaiser & Sutskever 2016, "Neural GPUs Learn
  Algorithms").

## The missing pieces

Each is a subplan of its own. "Step N" is the primitives roadmap's.

- **P1. Length-free token networks.**
  - Attention and the other token layers take `T` from the batch. They build the causal mask
    and the rotary table per `T` they meet, caching them, and a longer rotary table keeps the
    shorter one's rows.
  - Applies in all three implementations and the Rust layers. The crate needs no change beyond
    a check that its ops have no hidden length assumption.
  - A network's input shape gains a variable token count. `Position` keeps a fixed one, which
    validation enforces.
  - Format 2 records the variable count, and a file trained at one `T` loads at any `T` when the
    model has no `Position`.
  - At a fixed `T` the arithmetic is unchanged, so the golden run should stay bit-identical;
    that has to be checked, not assumed.
  - Everything else depends on this piece.
- **P2. Positions that extrapolate.**
  - The step 8 rotary scheme is the baseline, and no positions at all (already allowed) is a
    control.
  - New `Attention(positions=...)` schemes, using the field the rotary workplan's D2 left open:
    ALiBi (Press et al. 2022, "Train Short, Test Long"), and FIRE if the subplan's reading
    favours it.
  - Optionally, attention scores scaled with `log T` against the dilution above.
  - Each scheme is a forward and backward in three implementations and the crate. A study picks
    among them.
- **P3. Positions as data.**
  - An explicit integer position per token, supplied by the encoder, instead of `0..T-1`.
    `Position` and rotary would read it.
  - It enables randomised position offsets in training, Abacus-style digit indices, and position
    coupling, which aligns digits across operands in the string format.
  - It is a per-token input beside the token ids, the same mechanism as step 11's segment ids
    (its "forward context"), so it is built on step 11.
- **P4. Training across lengths.**
  - Bucketed batches, one length per batch, need only P1.
  - Mixed lengths in one batch need padding and a loss over counted tokens: step 11's segments.
  - The training mixture spans `n_train`'s range, possibly as a curriculum.
  - The bucketed form comes first.
- **P5. The catalogue and harness across lengths.**
  - A generalisation group of properties: B1, M3 and M5, sampled at every test length above
    `n_train`, plus unequal-length operands (zero-padded columns).
  - Sampled strata at large `n`: M3's distances up to `n`, M5's patterns sampled.
  - Configs that separate train lengths from test lengths.
  - A report of accuracy and `L*` against length, and the largest length that passes.
  - The harness's runs, resume and model store carry over unchanged.
- **P6. Long test lengths.**
  - Test lengths in the thousands of columns need step 12's tiled attention for memory.
  - Below about 1,000 columns, P1 suffices. Planned with step 12, not here.
- **P7. Depth that grows with length (the arbitrary-length claim).**
  - A looped (weight-tied) block: one block's weights applied `k` times, `k` chosen per input
    (from `n`, or adaptively), optionally re-reading the input each step (McLeish et al.'s input
    injection).
  - Needs parameters shared between several uses: the gradient summed over uses, in all three
    implementations and the crate. Today each layer owns its parameters.
  - Needs a variable iteration count at inference, and memory for `k` steps' activations in
    training.
  - The alternative is a recurrent cell over the columns with backpropagation through time:
    the window-recurrent adder, milestone A now (the owner, 2026-10-12).
- **P8. The string format at any length (candidate 5).**
  - Answers of varying length decoded autoregressively need step 9 (generation).
  - Its loss on answer tokens only needs step 11's counted targets.
  - Its digit alignment needs P3 (position coupling or Abacus indices).
  - Last, because it needs the most.

## Decisions (high level)

Settled under the owner's standing delegation (production target, built for extensibility); each
subplan revisits its own.

- **H1. The claim. Settled: (c).**
  - (a) Length generalisation only. Pros: P1, P2, P4 and P5 suffice; no new parameter sharing.
    Cons: a fixed-depth result always has a longest length, and the study only measures where.
  - (b) Arbitrary length only, through P7 from the start. Cons: the largest piece first, with
    nothing to compare it against.
  - (c) *Chosen.* (a) first as milestone A, then (b) as milestone B, with milestone A's best model
    as its baseline. Pros: each milestone answers a question alone, and B's gain is measured
    against A. Cons: two studies.
- **H2. The first pieces. Settled: (a).**
  - (a) *Chosen.* P1, P4 (bucketed) and P5 first, with step 8's rotary and no positions as the
    schemes: the first measurement of how far the column transformers extend, with no new
    position scheme. Pros: the smallest path to a number, from pieces every later option needs.
    Cons: rotary may extrapolate poorly; a negative result is still the baseline P2 is judged
    against.
  - (b) P2's new schemes first. Cons: chooses a scheme before seeing the failure it must fix.
- **H3. The growing-depth mechanism. Settled: (a), at milestone B; superseded by the owner
  (2026-10-12):** recurrence along the columns, windows of `k` with a discrete state between them,
  is milestone A (`docs/window-recurrent-adder-workplan.md`). The looped block stays an option
  for after it.
  - (a) *Chosen.* A looped transformer block (P7). Pros: reuses the blocks the framework has;
    the published route for addition (McLeish et al., Fan et al.); weight sharing is useful
    beyond this (tied embeddings, for one). Cons: shared parameters touch every implementation's
    parameter handling.
  - (b) A recurrent cell. Pros: the ripple-carry algorithm itself, linear cost. Cons: a new
    family; nothing else here uses it yet.
- **H4. The encoding. Settled: (a).**
  - (a) *Chosen.* The column encoding (candidates 1-3) carries the milestones. It aligns digits
    by construction, so length is the only new variable.
  - (b) The string format (5) as well, from the start. Cons: needs steps 9 and 11 and P3 before
    its first run. It follows as P8.

## Order and dependencies

1. **Milestone A, now the window-recurrent adder** (`docs/window-recurrent-adder-workplan.md`).
   As first planned (terminated, kept for the record): length generalisation, its subplan P1, P4
   bucketed and P5 with an existence study at `d` 16 and 16 layers:
   - P1, then P4 (bucketed) and P5, then a study: the column transformers trained at
     `n_train` (say 4-8) and tested to 10× it, with rotary or no positions.
   - Then P2's schemes against that baseline, and P3 once step 11 lands (randomised offsets, digit
     indices).
   - Needs step 8 merged; P3 needs step 11.
2. **Milestone B, arbitrary length:** P7, then a study with the looped block against milestone
   A's best model, tested at lengths where milestone A failed.
3. **Later:**
   - P6 with step 12, when test lengths outgrow untiled attention.
   - P8 after steps 9 and 11.

## Timing and checks

- P1 changes timed paths (every attention layer's set-up): tier 1, one A/B of the attention case.
  The golden run must stay bit-identical at fixed `T`.
- P2 and P7 add new paths, so tier 0 for the existing ones, with their own parity tests in all
  three implementations.
- The studies run on `jebel` under the study-runs workplan's lock.

## Parked

- **Composing a fixed-length adder outside the network** (the owner, 2026-10-12, for later
  consideration): a fixed-`n` column adder outputs `n + 1` digits, the top one a carry; for two
  operands, add chunk `k` as `(a_k, b_k, c = carry_{k-1})` and hand the top digit on, exact at any
  length in `T / n` steps with the sweep's adders (three operands need a fourth slot or a second
  pass). The network doesn't learn length; code composes it.

## Not in this plan

- Other operations (multiplication, subtraction): each its own question once P1-P5 exist.
- Bases other than 3, and more operands: parameters of the catalogue, not infrastructure.
- Dense candidates 6-8 at other lengths: fixed input by construction; a recurrent cell over the
  columns (P7's alternative, milestone A's design) covers a dense column cell.
