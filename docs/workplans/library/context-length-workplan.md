# Workplan: context length (roadmap step 12)

**Status: decisions D1-D10 settled (2026-10-10) by building for extensibility, speed and real-world
production (the owner's direction, 2026-10-10); D9's golden re-record approved by the owner
(2026-10-10) under measurement.md §8. Stage 1 (this plan) done. Revised 2026-11-01 against what
steps 10 (the training recipe) and 11 (segments) built since: refinements under D3-D8, and D11-D13
added. D11, the end-of-paragraph token, is the owner's decision of 2026-11-01.**

Roadmap step 12 ([primitives-roadmap.md](../../primitives-roadmap.md)): sequence models over windows of
256 and 1,024 characters, not 64, with the attention that makes that affordable (tiled, with an
online softmax, never holding an `(N, h, T, T)` array), training windows that change every epoch,
and a study of what longer context buys on the long-form corpora, with and without step 11's
document masks.

## Why

The long-form corpora are the targets (the owner, 2026-10-10). A paragraph's median length is 729
characters on Herodotus, 1,130 on Euclid and 1,266 on the *Muqaddimah*; a 64-character window sees
under a tenth of the paragraph it is in, so a model can't use the argument it is predicting
within: who is speaking, which figure Euclid is naming, the clause the sentence began. No mask
or schedule fixes that; only a longer window does. Every production language model's first
scaling axis after width and depth is its context.

What stops it today is attention's cost. The attention layers keep `P`, `(N, h, T, T)`, for the
backward pass: quadratic in `T` per window. At `T = 1024` and 4 heads that is 33 MB per window per
layer in float64: 1.07 GB per layer at a batch of 32, 67 MB at D5's token budget (2 windows), and
every doubling of `T` or the batch doubles it again. The work per token grows with `T` too, and a
causal model computes the half of `S` it then masks. Production attention (FlashAttention, Dao et al. 2022) tiles
the keys, keeps a running max and sum per query (the online softmax, Milakov & Gimelshein 2018),
never stores `P`, and recomputes each tile in the backward pass: memory linear in `T`, and fast
because each tile stays in cache.

## Decisions (settled 2026-10-10; refined 2026-11-01)

Each lists the options considered, with pros and cons, and the choice.

- **D1. Scope. Settled: (a).**
  - (a) *Chosen.* Tiled attention with an online softmax in all three implementations and the
    crate (D3); windows re-cut every epoch (D4); a token budget per step (D5); a context-length
    study on the long-form corpora with document-mask arms (D8); the benchmarks and memory
    measurements (D7). Pros: what long context needs in production, measured. Cons: attention's
    third rewrite (after steps 4 and 5), with every implementation's parity redone.
  - (b) Only raise `T`. Cons: 1 GB of `P` per layer at 1,024; nothing past that.
  - (c) Keep `P`, but in float32. Cons: halves memory, still quadratic; every implementation is
    float64, and mixed precision is a step of its own.
- **D2. The lengths. Settled.** `T` in {64, 256, 1024} characters: 1,024 is about a paragraph on
  the long-form corpora. Longer, once the study shows the curve still falling at 1,024.
- **D3. Tiled attention. Settled: (a).**
  - (a) *Chosen.* Per head and query block, the keys in blocks of `b` (64 by default; a layer
    option, not a spec field, since it changes the grouping of sums, not the model): `m` the
    running row max, `l` the running sum, `H` the running unnormalized output; for each key block,
    `m' = max(m, max_u S_tu)`, `e = exp(S - m')`, `l = l * exp(m - m') + sum(e)`, `H = H * exp(m -
    m') + e V`, then `H / l`. Masked scores (steps 6, 11) are `-inf` in their block, and a block
    wholly masked (above the causal diagonal, or across segments) is skipped (D9 of step 11's
    "varlen" gain). The forward keeps `m` and `l` per query (`(N, h, T)`), not `P`; the backward
    recomputes each block's `P` from them (the structure of Dao et al. 2022's backward pass). The same
    expressions, block order and folds in all three implementations, so parity is by bits where
    it held. Pros: memory linear in `T`, so the batch or `T` can grow; causal attention does
    about half the work by skipping blocks above the diagonal; the production algorithm. Cons: one path for every `T`, so its arithmetic differs
    from today's at `T = 64` (the final division comes after `e V`, not before), and every
    attention golden entry changes (D9).
  - (b) Tiled above a threshold, today's path below it. Pros: today's bits kept at `T <= b`.
    Cons: two attention paths in every implementation, a special case the project's
    structure-first rule refuses (CLAUDE.md).
  - (c) Today's path with `P` recomputed in the backward (checkpointing). Cons: memory still
    quadratic within the forward; no block skipping.

  **Refinements (2026-11-01), against steps 10 and 11 as built:**

  - *The masks.* A tile's mask is the causal one (`u > t`) and step 11's segment one (another
    segment, or a pad's key, never the diagonal; numpy's `attention_blocked`, the crate's
    `segments=` on `attention_forward(_batch)`), built per tile from the segment ids, never as an
    `(N, T, T)` array. A tile is skipped when it is wholly masked. Skipping is exact where the
    sums are sequential folds: a wholly masked tile gives `m' = m`, `exp(m - m') = 1` and `e = 0`,
    so `l` and `H` keep their bits, and the crate and pure Python, which loop over examples, may
    skip per example by bits. numpy's batched form skips a tile only when it is wholly masked for
    every example of the batch, and computes each key block over the range of rows that reach it;
    its products are BLAS's, whose order of summation may follow a product's shape, so there
    skipping is within rounding (stage 4 found it; numpy's products were never the others' bits).
    Options for the granularity:
    (i) per example everywhere: pros, the most skipped; cons, numpy loses its batched products
    over `(N, h)`. (ii) per batch in numpy, per example elsewhere (*chosen*): pros, each
    implementation skips what its loop shape allows at no cost to parity; cons, numpy skips less
    when a batch's rows disagree (rare for causal skipping, which is the same for every row).
    (iii) only causal skipping: cons, gives up the segments workplan's D9 gain. Within a tile
    computed, a row may still be wholly masked (a query whose segment starts after the block):
    its `m'` is `-inf`, so the shift uses `0` in its place (`e = exp(-inf) = 0`, `l` and `H` stay
    0), as FlashAttention's kernels do; every row ends with its diagonal, so no row's `l` is 0.
  - *Folds.* Each tile's row sums (of `e`, and of `dP * P` in the backward) are left folds in key
    order (the summation-order rule, `tests/model/layers/test_summation_order.py`), and the
    tiles combine in key-block order. The backward's row term is FlashAttention-2's
    `D = rowsum(dH * H)`, one fold over `d_k`, in place of today's `sum(dP * P)` over `T`: the same
    quantity, linear in `T`, and computed once per query rather than per tile.
  - *Decoding.* Prefill (`prefill` in `attention_array_layer.py`, `attention_layer.py`,
    `attention_rust_array_layer.py`) is the forward pass on `L <= T` tokens and moves to the tiled
    path with it: one forward path, so a prompt's prefill and the training forward stay one code.
    The step attends one query row to a cache (numpy's batched `(N, c)` cache masked by the keys'
    segments; Rust and pure Python one row at a time through `attention_step`). Its `P` is
    `(N, h, 1, c)`, linear in `T` already, so tiling saves no memory; but the full pass's
    arithmetic changes, so a step kept in today's form would no longer match it by bits. Options:
    (i) the step runs the same tiled attend, one query row against the cache's keys in blocks of
    `b` from the cache's first column (*chosen*): pros, one attention arithmetic for the forward,
    prefill and step, so a decode stays its implementation's full pass by bits on Rust and pure
    Python (within rounding on numpy, as today) while the cache lies in the window, the strongest
    test the decoder has; cons, a loop of at most `T / b` blocks for one row (16 at 1,024), a small
    overhead on an overhead-bound numpy step. (ii) The step as is: pros, no change to a working
    path; cons, two softmax forms, and decode-against-full-pass parity drops from bits to rounding.
    (iii) Postpone: cons, prefill would keep the old forward, two forward paths.
    `tests/generation/` stays green.
- **D4. Training windows. Settled: (a).**
  - (a) *Chosen.* Each epoch, the paragraphs (step 11's units) are shuffled by the run's generator
    and packed into windows again, so the windows' cuts change every epoch while every paragraph
    is seen once. The trainer gets an epoch-dependent dataset protocol; the generator's state is
    already in run files, so a resumed run re-cuts identically. Pros: every position becomes a
    target at different contexts across epochs (at `T = 1024` a fixed cut wastes most positions'
    long contexts); boundaries stay whole; production's shuffled-documents-then-pack order. Cons:
    the prepared dataset is rebuilt each epoch (a pass over the text; small beside training).
  - (b) Random offsets into the text (the sequence task's D4 (b)). Cons: windows overlap and
    straddle boundaries arbitrarily; some text seen twice an epoch, some not at all.
  - (c) Fixed cuts. Cons: at 1,024 each paragraph's opening characters never see long context.

  **Refinement (2026-11-01).** It builds on step 11's packed loader
  (`text_data.load_packed_text_dataset`: paragraphs each followed by the end-of-paragraph token,
  cut into windows). `train_backprop_network_mini_batch` takes, in place of a list of examples,
  an object with `epoch(rng) -> list[Example]` (a `PackedEpochs` holding the training
  paragraphs, the vocabulary and `T`): each epoch the trainer's shuffle generator (`random.Random`,
  already saved in run files as `shuffle_state`) shuffles the paragraphs, they are packed and cut,
  and the windows are shuffled into batches as today. A list stays a fixed dataset, by bits as
  before. A resumed run re-cuts identically because the generator's state at the epoch boundary is
  what the run file holds; the held-out windows stay fixed (`load_packed_text_dataset`'s, so
  evaluations compare across epochs). The window count is the same every epoch (the tokens are,
  and the remainder shorter than a window is dropped); which tokens fall in it changes. Options
  for the hook: (i) the epoch object (*chosen*): pros, one protocol any later dataset (random
  crops, curricula) implements; the trainer stays dataset-agnostic. (ii) A callback building the
  list: cons, the same with no type to check against. (iii) Pre-built epochs: cons, `E` copies
  of the corpus in memory and nothing for a resume to re-derive. A run file still doesn't record
  `clip_norm` (next-steps, From the training recipe); the study's runs don't resume, so that
  stays in next-steps.
- **D5. Batch size. Settled.** A token budget per step, `B * T = 2,048` (today's 32 x 64): batch 8
  at 256, 2 at 1,024. Steps per epoch and tokens per step are then equal across `T`, so the arms
  differ in context alone. Gradient accumulation over micro-batches (step 10's split step, D10
  there) lets a larger budget run in the same memory, and the study's rate tune checks small
  batches stay stable. *Refinement (2026-11-01):* step 10's split step (every layer's gradient
  accumulated, then one `step()` applying it, clipping by the global norm between) is the
  micro-batch mechanism already; at the study's budget no micro-batches are needed (D3 makes the
  memory linear), so none is built here.
- **D6. The model. Settled.** The 2-layer, `d = 64`, 4-head model with rotary positions (step 8),
  step 7's dropout and step 10's recipe: the context is the only change. Scaling width and depth
  with context is a study of its own, once this one says how much context helps. *Refinement
  (2026-11-01):* step 10's recipe as its study left it: warmup, then cosine to a tenth, AdamW 0.1,
  clipping at 1.0 (`scripts/studies/lr_schedule_study.py`, C1 and C2); `Embedding(...,
  segments=True)` on the packed corpora; the end-of-paragraph token's target counted (D11).
- **D7. Measurements. Settled.** A benchmark of attention forward and backward at `T` in {64, 256,
  1024}, tiled against today's (at the last commit before the switch), time and peak memory, on
  `jebel` through `ab.py` (a probe added in the stage that needs it). *Refinement (2026-11-01):*
  the crate has no allocation counter, and one would be a crate change before anything is
  measured. Options for memory: (i) a counting global allocator in the crate: pros, exact bytes;
  cons, a crate PR and bump first, and it says nothing about numpy. (ii) Each case in a fresh
  process, its peak resident set (`getrusage`'s `ru_maxrss`) less the same process's before the
  case runs (*chosen*): pros, one measure for numpy and the crate alike, transient arrays
  included, no crate change; cons, page-granular and including the allocator's slack, fine for a
  quadratic-against-linear verdict. (iii) `tracemalloc` alone: cons, blind to the crate.
  `tracemalloc`'s peak is reported beside (ii) for numpy as a cross-check. Time goes through
  `ab.py` (an adapter for the script, measurement.md §4); memory is the script's own report, not
  an `ab.py` comparison, since it is deterministic and needs no passes.
- **D8. The study. Settled.** `scripts/context_length_study.py`, numpy (Rust if the benchmark shows
  numpy too slow at 1,024), on Herodotus, Euclid and the *Muqaddimah*, Tiny Shakespeare as a
  reference: `T` in {64, 256, 1024}, and at 256 and 1,024 the document mask on and off (step 11);
  5 seeds; epochs and the grid sized from a timing first, as the sequence study did. Reported:
  held-out bits per character overall and by position in the window (whether later positions,
  with more context, predict better), on both held-out splits. *Refinement (2026-11-01):* the
  study follows the research repo's `docs/protocols/experiment-design.md` and its feasibility
  checklist (`information-flow.md`): its claims and their refutations are written into this
  workplan, in stage 9's first PR, before any of its data, and no setting changes after.
  It lives at `scripts/studies/context_length_study.py`, with the other primitives' studies. Its
  unmasked arm needs a packed batch without the mask: `Embedding(..., segments=True)` with a
  network option that keeps the positions and counted targets but lets attention see across
  segments; built with the study, as the arm's one switch.
- **D9. Golden run and timing.** Tiled attention changes every attention entry's bits (D3 (a)): a
  re-record under measurement.md §8 as fundamentally better structured (one path, memory linear in
  `T`), **approved by the owner on 2026-10-10**, recorded in stage 4; every other entry
  bit-identical. A speedup is claimed at long `T`: tier 2, the full protocol, archived; at `T = 64`
  the A/B must show no slowdown, or the PR says how much and why.
- **D10. Out of scope.** Mixed precision (float32 or bfloat16 compute), sequence parallelism, and
  sparse or sliding-window attention patterns: each its own step when a context past 1,024 or a
  larger model needs it. Recorded in next-steps.
- **D11. A paragraph's first character (added 2026-11-01). Settled by the owner: an
  end-of-paragraph token, (a).** Step 11 counts no target whose input is the separator, so after
  a separator a decode samples a paragraph's first character from a row no loss shaped (next-steps,
  From generation); at 1,024 characters a window holds about a paragraph, so generating several is
  where this matters.
  - (a) *Chosen (the owner).* The separator is the end-of-paragraph token it already is (id V,
    `<|sep|>`, the last token of the paragraph it ends, in that paragraph's segment), and its
    target counts: from the paragraph just ended, the model predicts the next one's first
    character. A target counts unless its input is a pad. Pros: one special token, as GPT's
    `<|endoftext|>`; every character of the text, and every paragraph's end, is a counted target;
    the decoder already keeps the token in its paragraph's segment, so decoding needs no change;
    no token or file format changes, so saved models and tokenizers load as they are. Cons: under
    the document mask the first character sees the paragraph before it, through the token's row
    (within one book that is real context, not leakage between unrelated documents); the packed
    golden entries move (the owner's decision is the approval: counting a target can't keep their
    bits).
  - (b) A begin-of-paragraph token (a third special id) opening each paragraph's segment, its
    target counted. Pros: the first character is predicted inside the paragraph's own mask.
    Cons: an extra token a paragraph, a vocabulary and file-format change, and the first
    character's prediction sees nothing but the token, a unigram of openings.
  - (c) The separator moved into the next paragraph's segment (opening it), its target counted:
    (b)'s semantics with one token. Cons: the token then ends a paragraph from outside it, so
    the last character's prediction of it crosses the mask; the decoder's segment rule changes.

  Naming: the code keeps `SEPARATOR`, `separator_id` and the `<|sep|>` string (a rename is churn
  across three implementations and every saved tokenizer, for no change in behaviour); the docs
  call it the end-of-paragraph token.
- **D12. Dropout under tiling (added 2026-11-01). Settled: (a).** Step 7's dropout draws one
  `(N, h, T, T)` mask per forward batch, every entry drawn in row-major order, and keeps it for the
  backward pass: quadratic memory again, which D3 doesn't address.
  - (a) *Chosen.* The forward saves the dropout generator's state before its draws and draws the
    mask a key block at a time; the backward restores a copy of that state and draws the same
    blocks again in the same order. Every entry is still drawn (the masked ones included, as
    today), so the draw count, and the generator's state after the batch, are today's. The order
    changes to key-block-major, `(key block, example, head, query, key in block)`, so numpy
    draws an `(N, h, T, b)` block in one call, the shape the key-block loop of the forward and
    backward passes reads (refined in stage 4 from query-block-major, which no pass's loop
    matches); memory `N h T b`, linear in `T`. Pros: memory
    linear; no new RNG; the network's generator advances as before, so nothing after the
    attention's draws shifts. Cons: the mask's bits change (the order), which D9's re-record covers;
    the draws are made twice (once in the backward), a cost linear in the work already done.
  - (b) A counter-based generator (Philox, as FlashAttention's kernels) drawing each tile's mask
    from its coordinates. Pros: tiles drawn in any order, in parallel. Cons: a new generator in
    numpy, the crate and pure Python, bit-identical, for parallelism the CPU kernels don't use.
  - (c) Keep the stored mask, as booleans. Cons: still quadratic (8 MB per layer at the study's
    1,024), and the crate's mask is float64 today.
- **D13. The parked length-free shapes (added 2026-11-01). Settled: (a), stay parked.** The local
  branch `length-free-shapes` (`bd13b16`; next-steps, "Length-free token networks (its P1) are
  parked") makes token networks' shapes length-free.
  - (a) *Chosen.* Leave it parked. Pros: the study builds a network per `T` from the same specs,
    `T` a spec field as today; nothing here needs one network at several lengths. Cons: a model
    trained at 1,024 can't be loaded at 256 to compare on the same weights.
  - (b) Revive it as a stage. Pros: one network at any length, which generation past `T` would
    use. Cons: a format change (format 2's `{"dimension": null}`) in a step about attention's
    cost; its value comes with length generalisation, not this study.

## Stages

1. **This workplan** (and its revision of 2026-11-01).
2. **The end-of-paragraph token** (D11): the separator's target counted in numpy's and pure
   Python's segment builders (the crate's networks read numpy's) and in the evaluation; the
   packed golden entries re-recorded, on both pyramidon and jebel, and the committed
   `data/refactoring/golden_run_rust.json`; every other entry bit-identical.
3. **The benchmark** (D7): attention at `T` in {64, 256, 1024}, today's path, time and memory,
   recorded before anything changes.
4. **Tiled attention in numpy** (D3, D12): the expressions, tests by hand (one block equals the
   softmax up to the final division's grouping; masked blocks skipped give the same bits as
   unskipped; memory linear in `T`); the golden re-record (D9, approved).
5. **The crate**: tiled forward and backward, numpy's bits; "Bump rust/" with the tier 2 A/B.
6. **Rust layers**; 7. **Pure Python**; parity.
8. **Epoch-dependent windows** (D4) and the token budget (D5) in the trainer.
9. **The study** (D8): its claims registered first.
10. **Docs**; the workplan retired.
