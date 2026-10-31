# Workplan: segments: packing, document masks and padding (roadmap step 11)

**Status: decisions D1-D11 settled (2026-10-10) by building for extensibility, speed and real-world
production (the owner's direction, 2026-10-10: the long-form corpora are the targets, a toy corpus
never decides the design). Stage 1 (this plan) done.**

Roadmap step 11 ([primitives-roadmap.md](../../primitives-roadmap.md)): one segment id per token, from
which attention, the loss and the evaluation know which tokens belong together and which are
padding; text packed into full windows with explicit boundaries for training; padded batches of
variable-length inputs for inference. All three implementations and the crate.

## Why

Every sequence example so far is a fixed window cut from a corpus with no regard to its
structure. Production systems handle variable-length text two ways, and both need this step:

- **Training packs.** Documents (here, paragraphs) are joined with a separator token and cut into
  full windows, so no compute is spent on padding (GPT-3 packs documents delimited by an
  end-of-text token, unmasked; Brown et al. 2020, appendix B). A window then spans boundaries,
  and a document mask can stop a token attending across one (Llama 3 masks attention between
  documents in one sequence; Dubey et al. 2024, §3.2). The long-form
  corpora are where this matters: a paragraph's median length is 729 characters on Herodotus,
  1,266 on the *Muqaddimah* and 1,130 on Euclid, so step 12's longer windows will span several.
- **Inference pads.** Batched generation (step 9's decoder, several prompts at once), encoder
  tasks and sequence-to-sequence (step 15, sources of different lengths) batch inputs of
  different lengths, padded to one width, and the padding must change nothing.

Both are one mechanism: a segment id per token, `0` for padding, and attention allowed from `t` to
`u` only if `u <= t` (causal), `segment(u) == segment(t)` and `segment(u) != 0`. T5X and the
variable-length attention kernels production uses take this form. The causal mask (step 6) was
the first mask; this is the general one.

## What exists, and what changes

| piece | today | this plan |
| --- | --- | --- |
| tokens | the corpus's characters, ids `0` to `V - 1` | plus two special tokens, a separator and a pad (D2) |
| examples | fixed windows of `T + 1` characters from the text's start | packed windows of paragraphs joined by the separator (D7); padded examples for inference (D8) |
| masks | causal, in attend | causal and same-segment and not padding (D4) |
| per-batch information | none: a layer sees its input array only | a forward context, carrying segment ids now (D3) |
| positions | per window | per segment (D5) |
| the loss | the mean over `T` tokens per example | the mean over the batch's counted tokens (D6) |

## Decisions (settled 2026-10-10)

Each lists the options considered, with pros and cons, and the choice.

- **D1. Scope. Settled: (a).**
  - (a) *Chosen.* Special tokens; segment ids; the mask in attention, the loss and the evaluation;
    packed training windows on all four corpora; padded batches in step 9's decoder; the forward
    context that carries them; all three implementations and the crate; fixtures, checkpoints and
    golden entries. The study of document masks moves to step 12, where windows span several
    paragraphs (D11). Pros: the production form of both uses at once, one mechanism. Cons: the
    largest refactor since composable layers (D3).
  - (b) Padding only. Cons: what production training does (packing, boundaries) left out; step
    12's long windows then cross paragraphs blindly.
  - (c) Packing only. Cons: batched inference and step 15 still lack padding.
- **D2. Special tokens. Settled: (a).**
  - (a) *Chosen.* The tokenizer (step 9's file, `kind: "characters"`) gains `special` tokens after
    the symbols: a separator (`<|sep|>`, id `V`) ending each paragraph, and a pad (`<|pad|>`, id
    `V + 1`). The corpus's blank-line paragraph breaks become the separator. `Embedding`'s table
    has `V + 2` rows; the pad's row is zero, never updated, and the output layer has `V + 1`
    classes (a model predicts the separator, never the pad). Pros: the boundary is a token the
    model learns to predict (a paragraph's end) and generation can emit; the pad is outside every
    class; the tokenizer's `special` list admits BOS, EOS or role tokens later, and BPE keeps the
    same scheme. Cons: the vocabulary grows by two and the corpora's models differ from step 6's
    by two rows.
  - (b) Keep `\n\n` as the boundary, no special tokens. Cons: a boundary spelled as two ordinary
    characters can't be told from text; a pad still needs an id.
  - (c) Lengths beside the state, no pad token. Cons: a second field in every example and dataset
    path; packing still needs a boundary marker.
- **D3. Carrying per-batch information to the layers. Settled: (a).**
  - (a) *Chosen.* An explicit forward context: every layer's forward, backward and gradient
    passes take a `ForwardContext` (`segments` now; step 9's decode positions, step 15's encoder
    output later), built once per batch by the network from the input ids. Layers that don't need
    it ignore it. A first stage changes every layer's signature in all three implementations with
    an empty context and nothing else, the golden run bit-identical. Pros: what a layer reads is in
    its signature; one channel for every later per-batch input; no state left over between
    batches. Cons: every layer class and its tests change, a refactor before the feature.
  - (b) A per-batch setter, as `set_training_mode`. Cons: hidden state that a forgotten call leaves
    stale; a new setter for each later input.
  - (c) The ids carried alongside the activations. Cons: every layer passes them through, and the
    `(N, T, d)` layout stops being the tokens alone (the `(N * T, d)` view without a copy).
- **D4. The mask in attention. Settled.** For query `t` and key `u`: masked (`-inf` before the max
  shift) unless `u <= t` (when causal), `segment(u) == segment(t)` and `segment(u) != 0`. A
  padded query's row keeps its own key (the diagonal is never masked, so no row is all `-inf`; its
  output is discarded by D6). The backward pass is unchanged, as for the causal mask (a masked
  weight is exactly 0). The crate's attention forward ops take the segment ids, an `(N, T)`
  integer array. One choice; the mask is the definition.
- **D5. Positions. Settled: (a).**
  - (a) *Chosen.* Positions count from 0 at each segment's first token. With step 8's rotary
    attention, scores depend on `t - u` within a segment only, so this changes nothing there; a
    learned `Position` reads row `t - start(segment)`. Pros: a paragraph sees the positions it
    would at inference, starting at 0. Cons: learned positions need the segment starts (from the
    context).
  - (b) Positions per window. Cons: a paragraph's first token sits at an arbitrary position, which
    inference never reproduces.
- **D6. The loss. Settled: (a).**
  - (a) *Chosen.* The mean over the batch's counted tokens: a target counts unless it is a pad or
    lies in another segment than its input (the token after a separator, the next paragraph's
    first, is not predictable from the masked context). Per example, `delta = (P - Y) * B / C`, `C`
    the batch's count, zero at uncounted positions; with every target counted, `C = B * T` and this
    is today's `(P - Y) / T`, bit for bit. Pros: PyTorch's `ignore_index` mean, the form
    production uses; the training loss is the per-token loss the evaluation reports; correct under
    gradient accumulation (each token weighs the same however the batch is split). Cons: the
    output delta reads `C`, from the context (D3).
  - (b) The mean per example over its counted tokens. Pros: a per-example delta. Cons: a short
    example's tokens weigh more; across micro-batches the loss isn't the token mean (the error
    Hugging Face's trainers fixed in 2024).
- **D7. Packed training windows. Settled.** `text_data` gains a packed loader: paragraphs (blank-line
  separated) in order, each followed by the separator, concatenated, cut into non-overlapping
  windows of `T + 1` tokens; the held-out split (the last 10%, and step 7's spread split) cut
  at a paragraph boundary, so no paragraph is in both. No padding in training. The fixed-window
  loader stays for the earlier studies' reproducibility.
- **D8. Padded inference batches. Settled.** Step 9's decoder takes a batch of prompts of
  different lengths, left-padded to the longest (so every row generates at the same column, the
  form batched decoders use), the pads segment 0 and masked, each row's positions counted from its
  first real token (D5). The array networks' `forward_rows` and `sequence_evaluate` accept padded
  examples too.
  - **D8's form, settled at stage 8 (2026-10-31).** Each choice's options, pros and cons:
    - *One decoder, a batch.* (a) *Chosen.* The Decoder decodes a batch of N prompts
      (`prefill_batch`, `step_batch`); `prefill(ids)` and `step(id)` are a batch of one. Every
      decoding layer's `prefill(X, cache, context)` and `step(x, cache, context)` take the batch's
      rows (N * L, then N) and a forward context with N (`examples`), the pass's first column
      (`column`, the rotary angles') and, when the batch has segments, each token's segment and
      position. Pros: one path; a single prompt with a separator gets its paragraph mask and
      positions too (step 9's decoder had neither, so it decoded such a prompt unlike training);
      a batch of one with no special ids is today's computation, bit for bit. Cons: every decoding
      layer's signature changes in all three implementations. (b) A second, batched decoder beside
      step 9's. Cons: two paths, and the single one stays wrong on separators.
    - *Padding.* (a) *Chosen.* Left-padded to the longest prompt, as D8 says. Prompts of one length
      need no pad, so a model without segments decodes them (it refuses prompts of different
      lengths). (b) Padded to the window T, so the prefill is the forward pass's own shape. Cons:
      every prefill and step pays for T keys; a model without a pad can't decode a short prompt.
    - *The step's mask.* The decoder tracks each row's segment (1 + the separators before the
      token; a pad's 0) and position (from the segment's first token, D5), by segments.py's
      rules. (a) *Chosen for numpy:* the cache keeps each key's segment and the step masks keys of
      another segment or a pad's (its own key never), one batched product per head; with no
      segments, no mask, today's path. (b) *Chosen for Rust and pure Python:* each row's cache
      holds its current segment's keys only, emptied when a row starts a segment, so the step
      attends to the whole cache unmasked: the crate's `attention_step` per row, unchanged, and
      pure Python's per-row folds, the masked zeros left off as now. Equal to (a), since a masked
      weight is exactly 0. Pros of the split: numpy batches now; the crate needs no new op yet.
      Cons: two cache forms until a batched crate step (with the keys' segments) lets Rust take
      (a), which next-steps.md records.
    - *A learned Position past the window* (step 9's D3, per row). A row's cached decode is exact
      while its current segment lies in the window (position < T): its query reads only that
      segment's keys, all decoded with training's positions. Once any real row's position reaches T,
      every step from there is a full pass over each row's last T tokens (a padded batch through
      the network's forward pass, its own segment context), said once, as step 9's decoder did
      past T.
    - *A paragraph's first token.* D6 counts no target whose input is a separator, so the
      distribution a decode draws after a separator is an untrained one; recorded in
      next-steps.md (a begin-of-paragraph token, or counting that target, is a later choice).
- **D9. Speed. Settled.** The mask is computed per head from the segment ids inside the crate's
  attend, never materialized as an `(N, T, T)` array in the crate. A batch whose segments are all
  1 (no packing, no padding) takes today's causal path, bit for bit and at today's speed. Skipping
  whole masked blocks (the varlen kernels' gain) waits for step 12's tiled attention, which is
  where it pays.
- **D10. Golden run and timing. Settled.** Stage 2 (the context, D3) keeps every bit; its tier 1
  A/Bs cover the dense, conv and attention cases, which must not move. New golden entries: a packed
  sequence model with segment masks in each implementation, re-recorded on both machines, archived
  as `new-functionality`. The crate's attention ops change: tier 1 A/B with no segments.
  - **Stage 2's A/Bs (2026-11-01, `jebel`).** Attention (`sequence_training_timing.py`, 4 passes):
    every row within noise. Dense and conv (`prepared_dataset_timing`, extended to 6 passes): 15
    rows within noise and one consistent and small, Rust's dense B=32 epoch, +1.1% (0.763 to
    0.771 s); its numpy control and every other row stayed. The likely cause is the keyword
    argument every pass now takes (D3), seen where a layer's call does the least work. Options:
    (a) *Chosen.* Accept it, recorded. Pros: D3's explicit context is the structure every later
    per-batch input uses (structure first); 8 ms an epoch on the most call-bound row. Cons: D10's
    "must not move" isn't met on that row. (b) Pass the context positionally. Cons: the
    keyword-only default, which lets a layer run alone without one, goes in every layer, for an
    unmeasured gain. (c) Skip the context for layers that don't read it. Cons: two call paths, which
    D3 (a) rejected.
- **D11. The study. Settled.** None in this step: at `T = 64` a window crosses a paragraph boundary
  in about 9% of windows on Herodotus (64 of 729 characters), too few for a document mask to show.
  Step 12's study, at windows that span several paragraphs, has the masked and unmasked arms. This
  step's claims are tests: a packed batch's every segment gives the bits it gives alone, padded or
  not; a padded row's real tokens match the unpadded row.

## Stages

1. **This workplan.**
2. **The forward context** (D3): every layer's passes take it, empty; golden run bit-identical;
   tier 1 A/Bs (D10).
3. **Tokens and data** (D2, D7): special tokens in the tokenizer, the packed loader, the
   separator in the corpora's text; tests.
4. **numpy** (D4-D6): segment ids into the context, the mask, positions per segment, the loss and
   the evaluation; the equivalence tests (D11).
5. **The crate** (D4, D9): segment ids through the attention ops; then a "Bump rust/" PR with its
   A/B.
6. **Rust layers**; 7. **Pure Python**; parity as every step.
8. **Padded inference** (D8): the decoder's padded batches (after step 9), `forward_rows`.
9. **Fixtures, checkpoints and golden entries** (D10).
10. **Docs**; the workplan retired.
