# Workplan: dropout in attention (roadmap step 7)

**Status: decisions D1-D10 settled by the owner (2026-10-10), each as recommended. Stages 1 (this
plan) and 2 (the specs) done. The RNG draw-order workplan, which it waited for, is retired (2026-10-10,
[next-steps.md](next-steps.md)), so its masks are tested in all three implementations by bits from
the first stage.**

Roadmap step 7 ([primitives-roadmap.md](primitives-roadmap.md)): dropout inside the transformer,
on the attention weights and on the blocks' outputs, in all three implementations and the crate,
and a study of whether it closes the sequence study's train/held-out gaps.

## Why

The sequence study (`scripts/sequence_study.py`, roadmap step 6) found training loss below
held-out loss at 2 layers: by 0.3 bits per character on Tiny Shakespeare and the *Muqaddimah*, and
by 0.49 on Euclid (1.01 against 1.50), the gap growing with the model (ffn 0.17, 1-layer 0.44 on
Euclid); on Herodotus none. The MNIST patch models never overfit at 5 epochs, so dropout had
nothing to show there (step 5's case). Now it has a measured question: how much of each gap is
overfitting that dropout removes, and how much is the held-out text being different (Euclid's
held-out part is its last 10%, from Book XII's similar pyramids through Book XIII; even the
counted bigram is 0.16 bits worse there).

Where transformers drop out:

- Vaswani et al. 2017, §5.4 ("Residual Dropout"): "We apply dropout to the output of each
  sub-layer, before it is added to the sub-layer input and normalized. In addition, we apply
  dropout to the sums of the embeddings and the positional encodings", at `P_drop = 0.1`.
- GPT-2's and nanoGPT's three: `embd_pdrop` (after the embeddings and positions), `resid_pdrop`
  (each block's output, before the add) and `attn_pdrop` (the attention weights `P`, before
  `P V`). The last isn't in the paper; it is in its reference code (tensor2tensor) and in BERT's
  and GPT's.

Step 5 named both places (next-steps.md, From multi-head attention, the extension points): an
inverted mask on `P[i]` in attend, kept for the backward pass, drawn by the layer from the
network's generator; and a token-wise dropout after combine, outside attention.

## What exists, and what changes

| piece | today | this plan |
| --- | --- | --- |
| dropout | `Dense(dropout=p)`, fused with a sigmoid hidden layer only, in all three implementations and the crate (`layer_dropout_*`) | dropout among the tokens (D1, D2) |
| the training switch | `TrainingModeLayer.set_training_mode`, which `learn*` switches on for the forward pass (dropout, batch norm) | the attention layer and the token dropout join it |
| masks | the network's generator, saved in format 2 and run checkpoints, so masks resume by bits; numpy and Rust by bits, pure Python after the RNG workplan | an `(N, h, T, T)` mask on `P` (D4), a token mask (D2) |
| attention | project, attend, combine; causal option | attend drops `P` in training (D3) |
| validation | a token block's body ends in `Attention` or an affine `Dense`; no dropout among tokens | D2's placement |
| the sequence study | a contiguous split, last 10% held out | a held-out set from across the text as well (D8) |

## Decisions (settled 2026-10-10)

Each lists the options considered, with pros and cons, and the choice.

- **D1. Scope. Settled: (a).**
  - (a) *Chosen.* Both of the literature's mechanisms: dropout on the attention weights
    (`Attention(dropout=p)`) and a token-wise dropout for the blocks' outputs and the embedding
    (D2); all three implementations and the crate; fixtures, checkpoints and golden entries; a
    study (D8). Pros: GPT's three dropouts, each where its literature puts it; the study can
    separate what each does. Cons: two mechanisms, two sets of crate ops.
  - (b) The attention weights only. Pros: the smallest: one named place, one option on one layer.
    Cons: the paper's own dropout is the residual one; a study of `attn_pdrop` alone measures the
    weaker regularizer and can't tell whether dropout as such closes the gaps.
  - (c) Residual dropout only. Pros: the paper's form. Cons: leaves step 5's other named place;
    GPT's `attn_pdrop` is the one setting a reader would look for next.
- **D2. The form of residual and embedding dropout. Settled: (a).**
  - (a) *Chosen.* A token-wise `Dropout(p)` spec: a layer with no weights that drops each
    token's features, inverted, in training. It stands after `Position` (the embedding dropout)
    and ends a block's body after `Attention` or the affine `Dense` (the residual dropout), the
    only places validation allows. Pros: one layer for both of the paper's dropouts; step 5's
    named place ("a token-wise dropout layer, outside attention"); attention and `Dense` keep one
    meaning each. Cons: on Rust one more crossing per use (a mask draw and a product, small beside
    attention's); "separate activation layers" stay out of scope (next-steps.md, From composable
    layers), so the reason this is a layer and they aren't has to be written down: a dropout has
    no activation to fuse with among tokens (the token-wise `Dense` is ReLU or linear).
  - (b) Dropout fields on the layers that end a body: `Attention(output_dropout=p)` and
    `Dense(dropout=p)` on a token-wise linear `Dense`, and `Position(dropout=p)` for the
    embedding. Pros: fused, no crossing; `Dense` already has a `dropout` field. Cons: three
    fields on three specs for one operation; a linear `Dense`'s dropout means something its
    sigmoid one doesn't (the op is fused with the sigmoid today), so the crate needs a linear
    variant of every `layer_dropout_*` op; `Position` gains a training mode.
- **D3. Dropout on `P`. Settled: (a).** In attend, in training: `M[i]` a 0/1 mask, `P~[i] = P[i] *
  M[i] / keep`, `H[i] = P~[i] V[i]`; the backward pass `dV[i] = P~[i]^T dH[i]`, `dP[i] = (dH[i]
  V[i]^T) * M[i] / keep`, then the softmax's backward as now, with the undropped `P[i]`.
  - (a) *Chosen.* Inverted dropout as above, rows of `P~` no longer summing to 1 in training;
    at inference `P~ = P`. Pros: the reference implementations' form (PyTorch's
    `F.scaled_dot_product_attention(dropout_p=...)`, nanoGPT); the dense layers' dropout is
    inverted too. Cons: the backward pass keeps both `P` and `M` (or `P~`): one more `(N, h, T, T)`
    cache.
  - (b) Renormalize each row after dropping. Pros: rows stay distributions. Cons: no reference
    uses it; a new backward; changes the expectation dropout is meant to keep.
- **D4. The mask's draws. Settled: (a).**
  - (a) *Chosen.* One `(N, h, T, T)` mask per forward batch, row-major (example, head, query,
    key), `u >= p` as the dense masks, every entry drawn, the causally masked ones included (their
    `P` is 0 either way). Pros: a fixed count and order, the same in all three implementations; no
    dependence on `causal`. Cons: about half the draws wasted under a causal mask (`T(T-1)/2` of
    `T²` per head).
  - (b) Draw only the unmasked entries. Pros: half the draws when causal. Cons: the order then
    depends on `causal`, and every implementation must walk the triangle the same way; the draws
    are a small part of a step either way.
- **D5. The training switch on Rust. Settled: (a).** The crate's attention forward ops take
  `dropout` and `training` keywords and the network's `Generator`, returning the mask with the
  caches; the backward op takes the mask (D3 (a)).
  - (a) *Chosen.* As stated, keyword-only with defaults (`dropout = 0.0`), so a call without
    them computes today's bits. Pros: the causal option's pattern (D7 of the sequence task); an
    unchanged call keeps its bits, checked by the A/B (D10). Cons: a crate PR first, then a bump.
  - (b) A separate set of dropout attention ops. Pros: the existing ops untouched. Cons: two
    copies of the three blocks in the crate, against the multi-head workplan's single structure.
- **D6. Pure Python. Settled: (a).** After the RNG workplan, a pure-Python network with a dropout
  layer trains layer-major (its D3), so its masks are numpy's order by construction.
  - (a) *Chosen.* Both mechanisms in pure Python, parity with numpy by bits for the masks and
    within the dense layers' rounding for training, as every step. Pros: the project's rule; the
    RNG workplan makes the masks checkable by bits. Cons: the attention layer's per-example fields
    gain the mask.
  - (b) numpy and Rust only. Cons: breaks the rule every step has kept.
- **D7. Format 2 and checkpoints. Settled.** An `"attention"` entry writes `"dropout"` only when
  set, as `"causal"`; a `"dropout"` entry for the token layer, with no weights and no optimizer
  state. Masks resume by bits through the generator's saved state, as the dense dropout's do. One
  choice only; recorded for completeness.
- **D8. The study. Settled: (a).** `scripts/attention_dropout_study.py`, numpy, Adam, the sequence
  study's 2-layer model and protocol (`indrajala_ml/studies/sequence_study.py`), timed first and
  the grid sized from it.
  - (a) *Chosen.* Arms at the 2-layer model: no dropout (the control), `attn_pdrop` 0.1,
    residual and embedding dropout 0.1, both 0.1, both 0.2. Corpora: Euclid (the widest gap),
    Tiny Shakespeare (0.3) and Herodotus (no gap: the control, where dropout should only cost).
    Held out: the contiguous split as before, and a second split from across the text (the text
    cut into 100 equal blocks, each block's last tenth held out), so that a gap present on the
    contiguous split and absent on the spread one is the text's shift, not overfitting. Pros:
    each arm answers one question (which mechanism, how much); the spread split answers Euclid's
    open question (next-steps.md, From the sequence task). Cons: a loader option (`split`) in
    `text_data`; 5 arms x 3 corpora x 2 splits at the 2-layer cost (72 to 132 s an epoch per job
    at 6 workers in the sequence study) is about 7 hours on `jebel` at 10 epochs and 5 seeds,
    before tuning and dropout's own cost; the timing may cut it to one split for the arms and
    both for the control.
  - (b) All four corpora, one split. Pros: the sequence study's breadth. Cons: can't separate
    overfitting from the shift, the gap's open question.
  - (c) Tiny Shakespeare only, `p` swept 0 to 0.3. Pros: cheap; the standard corpus. Cons: its
    gap is the middling one; says nothing about Euclid.
- **D9. Golden run. Settled.** New entries: the sequence model with both dropouts, `p = 0.1`, in
  each implementation, as #612 and #632 did (the earlier entries bit-identical, the file
  re-recorded on both machines and archived as `new-functionality`). One choice only.
- **D10. Timing. Settled.** The crate's attention ops change: tier 1, one A/B of the attention case
  (`--order ONNO`) with dropout off, whose path must not move. The token dropout is new and untimed
  until a benchmark uses it.

## Stages

1. **This workplan** (D1-D10 settled).
2. **Specs** (done): `Attention(dropout)`, `Dropout(p)` (D2), validation, `spec_shapes`, format 2's
   entries (D7); builders refuse both with "not yet" until their stages
   (`refuse_dropout_specs_until`; an `Attention(dropout=0.0)` builds as before, a `Dropout(0.0)`
   is refused with the rest).
3. **numpy**: the mask on `P` (D3, D4) and the token dropout, the training switch, parity tests by
   hand; tier 1 A/B of numpy's attention case with dropout off.
4. **The crate** (`indrajala-math-rust`): the attention ops' `dropout`, `training`, `rng` keywords
   and the mask through backward (D5); a token dropout forward and backward; numpy's bits; then a
   "Bump rust/" PR here with the tier 1 A/B (D10).
5. **Rust layers**: through the new ops; masks by bits against numpy from one seed.
6. **Pure Python** (D6): masks by bits against numpy from one seed, training within the dense
   layers' rounding.
7. **Fixtures, checkpoints and golden entries** (D7, D9).
8. **The study** (D8): the loader's spread split first, its own PR.
9. **Docs**: README, roadmap, next-steps; the workplan retired.
