# Workplan: context length (roadmap step 12)

**Status: decisions D1-D10 settled (2026-10-10) by building for extensibility, speed and real-world
production (the owner's direction, 2026-10-10), except D9's golden re-record, which needs the
owner's approval under measurement.md §8. Stage 1 (this plan) done.**

Roadmap step 12 ([primitives-roadmap.md](primitives-roadmap.md)): sequence models over windows of
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

## Decisions (settled 2026-10-10)

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
- **D5. Batch size. Settled.** A token budget per step, `B * T = 2,048` (today's 32 x 64): batch 8
  at 256, 2 at 1,024. Steps per epoch and tokens per step are then equal across `T`, so the arms
  differ in context alone. Gradient accumulation over micro-batches (step 10's split step, D10
  there) lets a larger budget run in the same memory, and the study's rate tune checks small
  batches stay stable.
- **D6. The model. Settled.** The 2-layer, `d = 64`, 4-head model with rotary positions (step 8),
  step 7's dropout and step 10's recipe: the context is the only change. Scaling width and depth
  with context is a study of its own, once this one says how much context helps.
- **D7. Measurements. Settled.** A benchmark of attention forward and backward at `T` in {64, 256,
  1024}, tiled against today's (at the last commit before the switch), time and peak memory
  (`tracemalloc` for numpy; the crate's allocation counter), on `jebel` through `ab.py` (a probe
  added in the stage that needs it).
- **D8. The study. Settled.** `scripts/context_length_study.py`, numpy (Rust if the benchmark shows
  numpy too slow at 1,024), on Herodotus, Euclid and the *Muqaddimah*, Tiny Shakespeare as a
  reference: `T` in {64, 256, 1024}, and at 256 and 1,024 the document mask on and off (step 11);
  5 seeds; epochs and the grid sized from a timing first, as the sequence study did. Reported:
  held-out bits per character overall and by position in the window (whether later positions,
  with more context, predict better), on both held-out splits.
- **D9. Golden run and timing.** Tiled attention changes every attention entry's bits (D3 (a)): a
  re-record under measurement.md §8 as fundamentally better structured (one path, memory linear in
  `T`), which needs **the owner's approval** before stage 3 merges; every other entry
  bit-identical. A speedup is claimed at long `T`: tier 2, the full protocol, archived; at `T = 64`
  the A/B must show no slowdown, or the PR says how much and why.
- **D10. Out of scope.** Mixed precision (float32 or bfloat16 compute), sequence parallelism, and
  sparse or sliding-window attention patterns: each its own step when a context past 1,024 or a
  larger model needs it. Recorded in next-steps.

## Stages

1. **This workplan.**
2. **The benchmark** (D7): attention at `T` in {64, 256, 1024}, today's path, time and memory,
   recorded before anything changes.
3. **Tiled attention in numpy** (D3): the expressions, tests by hand (one block equals the
   softmax up to the final division's grouping; masked blocks skipped give the same bits as
   unskipped; memory linear in `T`); the golden re-record (D9, with approval).
4. **The crate**: tiled forward and backward, numpy's bits; "Bump rust/" with the tier 2 A/B.
5. **Rust layers**; 6. **Pure Python**; parity.
7. **Epoch-dependent windows** (D4) and the token budget (D5) in the trainer and run files.
8. **The study** (D8).
9. **Docs**; the workplan retired.
