# Workplan: rotary position embedding (roadmap step 8)

**Status: decisions D1-D9 settled by the owner (2026-10-10). Stage 1 (this plan) done. Added to
the roadmap by the owner on 2026-10-10, before generation (now step 9), so that generation's
key/value cache stays valid as its window slides.**

Rotary position embedding (RoPE; Su et al. 2021, "RoFormer: Enhanced Transformer with Rotary
Position Embedding", arXiv 2104.09864): positions applied inside attention, by rotating each
query and key, so that the score between tokens `t` and `u` depends on their content and on
`t - u` only. All three implementations and the crate.

## Why

The sequence models add a learned `Position` row to each token before the first block (the
layer-norm and attention workplan, D7). Every key and value then depends on the absolute
position its token sat at. Generation (step 9, the owner's choice of a key/value cache) keeps
each earlier token's keys and values; once its 64-character window slides by one, every token's
position changes and every cached key and value is stale, so the cache would pay only for the
first 64 characters. Under RoPE a key is rotated by its token's position once, and a query's
rotation makes the score depend on the difference alone: cached keys stay valid as the window
slides, and the oldest is dropped. RoPE is the position scheme of most decoders since 2021
(GPT-NeoX, PaLM, LLaMA), and it is parameter-free.

## The form

For a head of `d_k` features (even), pair feature `j` with feature `j + d_k / 2`, `j < d_k / 2`
(the "rotate half" layout, D3), with frequency `theta_j = base^(-2j / d_k)`, `base = 10000`. At
position `t`, with `c = cos(t theta_j)` and `s = sin(t theta_j)`:

```text
rotate, each head's q (and k) at position t, j < d_k / 2
  q'_j          = q_j * c - q_{j+d_k/2} * s
  q'_{j+d_k/2}  = q_j * s + q_{j+d_k/2} * c
backward, the inverse rotation (the transpose), on dq' (and dk')
  dq_j          =  dq'_j * c + dq'_{j+d_k/2} * s
  dq_{j+d_k/2}  = -dq'_j * s + dq'_{j+d_k/2} * c
```

Then `q'_t . k'_u` depends on `t - u` and the contents only (RoFormer, §3.2). It sits at the end
of attention's project block, after the bias, on `Q` and `K` only (values are not rotated); attend
and combine are unchanged.

## Decisions

Settled by building for extensibility and speed (the owner, 2026-10-10), with the alternatives
and why they lose:

- **D1. Scope. Settled: (a).** (a) RoPE as an attention option in all three implementations and
  the crate; fixtures, checkpoints, golden entries; a study (D8). (b) numpy and Rust only: breaks
  the project's rule. (c) RoPE plus ALiBi or T5's relative biases: no use yet; D2's field admits
  them later.
- **D2. The spec. Settled: (a).** (a) `Attention(positions="rotary")`, a `positions` field
  defaulting to `"none"`, written to format 2 only when set, with `rotary_base: float = 10000.0`.
  Pros: the place where positions act; a string field admits `"alibi"` or `"relative"` later
  without a new spec; a model without `Position()` and with rotary attention is the usual decoder.
  Cons: a second meaning of "position" beside the `Position` spec, which the README explains.
  (b) A `Rotary()` spec. Cons: it acts inside attention, between project and attend, which a
  separate layer can't reach. (c) A boolean `rotary=True`. Cons: a second scheme needs a second
  boolean and a rule that they exclude each other.
- **D3. The pairing. Settled: (a).** (a) Rotate half: feature `j` with `j + d_k / 2`, as GPT-NeoX
  and LLaMA. Pros: each half a contiguous slice, so the rotation is four products over whole
  slices in numpy and the crate (speed). Cons: not the paper's own layout. (b) Interleaved,
  `2j` with `2j + 1`, RoFormer's. Cons: strided access. The two are the same model up to a fixed
  permutation of `Wq`'s and `Wk`'s rows; nothing here loads outside weights.
- **D4. The angles. Settled: (a).** `cos` and `sin` aren't correctly rounded, and numpy's,
  Rust's and Python's `math` may differ in the last bit (as `exp` does, which the tests swap in).
  (a) One table of `c` and `s` per layer, `(positions, d_k / 2)`, computed once in Python with
  `math.cos` and `math.sin` from `t * theta_j` (`theta_j` from `base ** (-2j / d_k)`, one
  expression for all three), held by the layer, not saved (it derives from the spec), and passed
  to the crate as arrays. Pros: the same bits in all three implementations by construction; no
  trigonometry in the hot path (speed); generation extends the table past `T`. Cons: a
  `(T, d_k / 2)` table per layer, two arrays per crate call. (b) Each implementation computes its
  own. Cons: parity by bits lost to the last-bit differences.
- **D5. The crate. Settled: (a).** (a) The four attention ops take `cos` and `sin` keyword
  arrays (default none, so an unrotated call keeps its bits); project rotates `Q` and `K` after
  the bias, project's backward rotates `dQ` and `dK` back before the products. (b) Separate
  rotation ops around the attention ops. Cons: two more crossings per pass, and the caches split
  between calls (speed).
- **D6. Validation. Settled: (a).** `positions="rotary"` needs an even key size, and refuses a
  `Position()` in the same token part (one position scheme per model; mixing them is a later
  option if a use appears). A causal and a non-causal attention may both rotate.
- **D7. Golden run and timing. Settled.** New entries: the sequence model with rotary attention in
  each implementation (earlier entries bit-identical; re-recorded on both machines, archived as
  `new-functionality`). The crate's attention ops change: tier 1, one A/B of the attention case
  with rotation off, whose path must not move; the rotation's own cost is measured in the study.

Settled by the owner (2026-10-10), as proposed:

- **D8. The study. Settled: (a).** (a) The sequence study's 1- and 2-layer models with rotary attention
  and no `Position`, against the learned-position models' recorded results, on all four corpora at
  the same protocol (tuned rate, 10 epochs, 5 seeds), numpy; and the time per epoch with and
  without rotation. Pros: a like-for-like answer on every corpus against baselines recorded on
  `jebel` at the same protocol; about 2.5 hours, as the sequence sweep. (b) The 2-layer model on
  Tiny Shakespeare and Euclid only. Cons: no answer on Herodotus or the *Muqaddimah*. (c) Both
  arms rerun side by side. Pros: the same code and machine state for both. Cons: twice the cost.
- **D9. Default for new sequence models. Settled: (a).** (a) The README's sequence example and
  step 9's `train` use rotary attention, without `Position`, if the study shows it at least
  matching learned positions (within the seeds' spread); else they keep `Position`. Pros:
  generation's cache works on default models; the data decides. (b) Rotary regardless. Cons:
  could ship a worse default on these corpora. (c) Keep `Position`. Cons: generation's cache then
  pays only for the first 64 characters on default models.

## Stages

1. **This workplan.**
2. **Specs**: `Attention(positions, rotary_base)`, validation (D6), format 2's fields; builders
   refuse rotation with "not yet" until their stages.
3. **numpy**: the table (D4), the rotation and its backward in project; tests by hand (a rotation
   preserves norms; the score depends on `t - u` only; one token at position 0 is the unrotated
   layer by bits); tier 1 A/B of numpy's attention case with rotation off.
4. **The crate**: `cos` and `sin` through the four ops (D5), numpy's bits; then a "Bump rust/" PR
   with the tier 1 A/B.
5. **Rust layers**: through the ops; parity with numpy.
6. **Pure Python**: the rotation in its project block; parity.
7. **Fixtures, checkpoints and golden entries** (D7).
8. **The study** (D8).
9. **Docs**: README, roadmap, next-steps; the workplan retired.
