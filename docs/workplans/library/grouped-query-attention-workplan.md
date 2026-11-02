# Workplan: grouped- and multi-query attention (roadmap step 14)

**Status: decisions D1-D8 settled 2026-10-10; revised 2026-11-02 against the library as steps 9 to
13 left it (D1, D4, D6 and D7 corrected, D2, D3 and D5 made exact, D9 to D12 new), settled by
building for extensibility, speed and real-world production (the owner's direction, 2026-10-10).
Started 2026-11-02 at the owner's word, while step 13's study runs; the multithreading workplan
moved to the end (its O1). Stage 1 (this plan) done.**

Roadmap step 14 ([primitives-roadmap.md](../../primitives-roadmap.md)): attention whose `h` query heads
share `g` key/value heads (Ainslie et al. 2023, "GQA"; `g = 1` is multi-query attention, Shazeer
2019), in all three implementations and the crate, measured where it pays: the key/value cache of
step 9's decoder.

## Why

A decoder's key/value cache holds `2 * h * d_k` values per token per layer. At long context and
large batch, the cache, not the weights, bounds how many sequences a server decodes at once and how
fast: each step reads the whole cache. Sharing key/value heads divides the cache by `h / g`, at a
small cost in quality. Llama 2's 70B model and Llama 3 at every size use GQA with `g = 8`. Step 5
named its place (next-steps.md, From multi-head attention): `Wk` and `Wv` in `g` row blocks, with
head `i` reading block `i // (h / g)`. Before step 9 there was no cache to shrink, which is why the
roadmap first judged it unmeasurable. With step 9's cache and step 12's windows, it has a production
question to answer.

## What the revision found (2026-11-02)

The plan was written before generation (step 9), the training recipe (step 10), segments (step 11),
tiled attention and the context study (step 12), and the FFN activations (step 13). Read against
today's code:

- **Attend is tiled in every implementation** (step 12, D3): the keys in blocks of 16, per query row
  a running max `m`, sum `l` and output `H`, and `P` never held. A head's work is per (example, key
  block, head) in the crate (`core/src/attention.rs`, `tiled_forward` and `tiled_backward`), on
  `(N, h, T, d_k)` views in numpy (`tiled_forward`, `tiled_backward`), and per (example, head) in
  pure Python (`AttentionLayer._tiled`). The crate addresses `K` and `V` by the query head's columns
  (`tile(v, ..., i * d_k, d_k)`; `scores` likewise). That is the one index GQA changes in the
  forward: head `i` reads group `i / (h / g)`.
- **The backward writes `dK` and `dV` per head and key block** (`dK[i] = dS^T Q[i]`, assigned per
  block). Under GQA a group's gradient is its heads' sum (D3), in a fixed order, so the three
  implementations keep agreeing by bits where they do today.
- **The decoder's caches are `h * d_k` wide:** numpy's `AttentionCache(width)`, the Rust layer's
  `AttentionRustCache(width)` and the crate's `attention_step(k_cache, v_cache)` (shape `(c, width)`).
  Pure Python's `AttentionCache` holds rows. Each must become `g * d_k`, and the cache's size is the
  measurement (D7).
- **Rotary rotates `Q` and `K` per head** (`Rotation` in the crate; `_rotated` in numpy and pure
  Python, with `self.heads` heads of `d_k`). `K` then has `g` heads to rotate, not `h`.
- **The dropout masks are per query head** (`(N, h, T, b)` per key block, every entry drawn). They
  stay as they are: a weight is a query head's, so GQA changes no draw.
- **The window program** (`core/src/program.rs`, `Op::Attention { heads, ... }`) runs the research
  networks' windows through the same crate attention. GQA there is out of scope (D10), as GELU was
  for step 13.
- **Format 2 writes an `Attention` field only when it isn't the default** (`format2_json.py`,
  `layer_to_json`). So a `key_value_heads` defaulting to `None` (meaning `heads`) leaves every
  existing file and fixture unchanged.
- **The parameters' shapes come from one place:** `AttentionProjections.projection_shapes` gives
  randomization, `_zeros`, the optimizers and snapshots their shapes. So `Wk` and `Wv` at
  `(g * d_k, d)` reach all of them, and at `g = h` the draws are today's, bit for bit.
- **The generation benchmark** (`scripts/measurement/generation_timing.py`) times the 2-layer model
  at 4 heads and `T` 64, batch 1. It has no options for heads, `g`, window or batch, which D7's
  measurement needs (D11).
- **The golden sequence model has 2 heads** (`SEQUENCE_SPECS`: `heads=2, key_size=3`), so the old
  plan's "`g = 2` and `g = 1`" entries need a model with more heads (D6).
- **Step 12 chose windows of 256 and step 13 chooses the FFN** (its study runs now). D7's "step 12's
  longest context" is now 256, and the study's model takes step 13's FFN.

## Decisions (settled 2026-10-10; revised 2026-11-02)

Each lists the options considered, with pros and cons, and the choice.

- **D1. Scope. Settled: (a) (made exact 2026-11-02).**
  - (a) *Chosen.* `Attention(key_value_heads=g)`, with `g` dividing `heads` and the default `None`
    meaning `heads` (today's attention, bit for bit). It covers all three implementations and the
    crate, the decoder's caches sized by `g`, converting a trained model to grouped heads (D5),
    fixtures, golden entries, and a study (D7). Pros: the production form with its conversion
    path. Cons: the attention ops' key/value indexing changes in every implementation.
  - (b) Multi-query only (`g = 1`). Cons: the quality loss Ainslie et al. measured; production
    settled on groups.
- **D2. The layout. Settled (made exact 2026-11-02).** As step 5 named it:
  - `Wq` stays `(h * d_k, d)`. `Wk` and `Wv` are `(g * d_k, d)` in `g` row blocks, with `bk` and
    `bv` `(g * d_k,)`. `Wo` stays `(d, h * d_k)`.
  - Query head `i` attends with key/value block `i // (h / g)`, so a group is `h / g` consecutive
    query heads.
  - Packed caches: `K`, `V` and their gradients are `(N * T, g * d_k)`; `Q`, `H` and theirs stay
    `(N * T, h * d_k)`.
  - `AttentionProjections` gains `key_value_heads` and `kv_width = g * d_k`, and
    `projection_shapes` gives `Wk` and `Wv` that width, so randomization, the optimizers and the
    snapshots follow it with no change of their own.
  - Format 2 writes `"key_value_heads"` only when it isn't `None` (its default).
- **D3. The backward. Settled (made exact 2026-11-02).** A shared key/value head's gradient, `dK`
  and `dV` in each key block, is the sum over its group's query heads: a left fold in head order
  from the first head's value (not from zero, so a `-0.0` survives as numpy's fold keeps it), in
  every implementation.
  - The fold comes before `dK`'s division by the scale: a group's `dK` is `(fold of its heads'
    dS^T Q) / s`, one division, as the crate divides once at the end (made exact at stage 2).
  - numpy's `tiled_forward` and `tiled_backward` take `K` and `V` at `g` heads and expand them to
    `h` (`expand`); the backward computes each query head's `dK` and `dV` as today, then folds each
    group over its heads (`fold`: `np.cumsum` along the head axis, the last entry), then divides.
  - The crate accumulates per key block: the group's first head assigns, each later head adds.
  - Pure Python folds the same way. Every element's sum is then the same sequence of additions, so
    parity holds by bits where it does today.
  - `dQ` is unchanged: a query head's own.
- **D4. The crate. Settled: (a) (made exact 2026-11-02).**
  - (a) *Chosen.* The four attention ops (`attention_forward`, `_batch`, `attention_step`,
    `attention_downstream_batch`; `attention_accumulate_gradient_batch` reads shapes) take
    `key_value_heads` as a keyword defaulting to `heads`, so every caller today is unchanged.
    `AttentionOptions` gains it; `scores` and `tiled_forward` read head `i`'s keys and values from
    block `i / (h / g)`; `tiled_backward` folds per D3. Within a key block, the heads of a group
    run consecutively, so the group's `K` and `V` tiles are cut once and reused (the "loaded once
    per group" of the old plan, in the tiled form). The rotation takes `K`'s head count (`g`).
    Pros: no copy of `K` or `V`, which is the memory GQA exists to save; one code path, with
    `g = h` today's.
  - (b) Expanding `K` and `V` to `h` heads before attend. Pros: attend untouched. Cons: it copies
    the memory GQA saves, and in decode it copies the whole cache every step.
  - numpy may expand (it is the reference, not the production path), as D3 says. Its cache still
    holds `g` heads, expanded per call.
- **D5. Converting trained models. Settled (made exact 2026-11-02).**
  - `to_grouped(document, g)` on a format-2 document (the network's `network_to_json`), so one
    function serves all three implementations. Each `Attention` entry's spec gains
    `key_value_heads=g`, and each group's `Wk` and `Wv` rows and `bk` and `bv` entries become the
    mean of its heads' blocks (Ainslie et al. 2023, §2.1): a left fold in head order, then divided
    by `h / g`.
  - The converted document carries no optimizer state, so uptraining starts its optimizer afresh,
    under the recipe at a fraction of the original epochs (D7).
  - Pros: production's path to GQA from an existing model, and one function across backends.
    Cons: a function to keep in step with format 2's attention entry.
- **D6. Golden run and timing. Settled (revised 2026-11-02).**
  - **Golden entries:** a 4-head variant of the golden sequence model (`heads=4, key_size=3`) at
    `g = 2` and `g = 1`, in each implementation and in the Rust reference (both architectures'
    CI). Earlier entries stay bit-identical (`g` defaults to `heads`).
  - **Tier 1 for the crate's changed ops:** `attention_timing` at `g = heads` with `--order ONNO`
    on `jebel` (now on solar: daylight only). Its rows must hold, since `g = h` is the old path.
  - **No tier 2:** no speedup over the old code is claimed. GQA's decode gain is between
    configurations of one tree, which is the study's measurement (D7, D11), not an A/B.
- **D7. The study. Settled (revised 2026-11-02).**
  - **The model:** the context-length study's 256 model (2 layers, `d = 64`, rotary, the
    document mask, the recipe, the token budget, 30 epochs on Rust), with step 13's chosen FFN, at
    8 query heads of `d_k = 8`, so `g` has room: 8, 4, 2, 1.
  - **The arms:** trained from scratch at each `g`, and uptrained from the `g = 8` model converted
    to each `g < 8` (D5; Ainslie et al.'s 5% of the original steps is 1.5 epochs, so 2 epochs).
  - **Data:** Herodotus, Euclid and the *Muqaddimah*, both splits, 5 seeds.
  - **Measures:** held-out bits per character; the cache's bytes per token,
    `2 * g * d_k * 8 * layers`; and decode characters per second at batch 1 and batch 16 at
    windows of 256 (D11), on `jebel` by daylight.
  - **Registration:** claims and a budget from a probe, registered in this plan before the data
    (experiment-design §2), at stage 8. The from-scratch arms alone are 120 runs, so the
    registration sizes them against `stone`'s or `jebel`'s time when it is written.
- **D8. Out of scope.** Multi-head latent attention (DeepSeek-V2's compressed cache) and cache
  quantization: further cache reductions, each its own step if the study shows the cache binding.
- **D9. Validation. Settled (new 2026-11-02).** `key_value_heads` is `None` or at least 1, and it
  divides `heads`. Refused: in a window-recurrent network's window (D10), and a `key_value_heads`
  equal to `heads` written out (the spec says `None`, so a model has one spelling). Pros: one
  spelling per model in specs and files. Cons: a user who writes `g = h` gets a refusal, with a
  message naming `None`.
- **D10. The window program. Settled: (a) (new 2026-11-02).**
  - (a) *Chosen.* Unchanged: `Op::Attention` passes `key_value_heads = heads`, and
    `validate_window_specs` refuses a grouped Attention, as it refuses GELU (step 13's D5). Pros: the
    research networks' path untouched; their attention is tiny, and its cache is the recurrent
    state, not a decoder's. Cons: none for today's research.
  - (b) GQA in the program too. Cons: no research network asks for it; when one does, it's a field
    on the op (step 13's D8 precedent).
- **D11. The measurement tool. Settled (new 2026-11-02).** `generation_timing.py` gains `--heads`,
  `--key-value-heads`, `--context` and `--batch` (prompts decoded together), with today's
  defaults, and reports the cache's bytes per token beside the rate. It is used in-tree by the
  study (D7), so it needs no `ab.py` adapter. Its options land with stage 6. Pros: the study's decode
  numbers come from the existing tool. Cons: none.
- **D12. Order of the stages. Settled (new 2026-11-02).** As step 12's plan found, a crate change
  that keeps its old signatures needs no layer change in its bump. With `key_value_heads`
  defaulting to `heads`, the crate's stage bumps alone, tier 1 at `g = h`, and the Rust layers
  follow. Stages 2 (numpy) and 3 (the crate) can run in either order; numpy first fixes the
  reference the crate's tests check against.

## Stages

1. **This workplan** (revised 2026-11-02).
2. **Specs, validation and numpy** (D2, D3, D9, D10):
   - `Attention.key_value_heads` (format 2 by default; the shape walk unchanged); validation.
   - `AttentionProjections` (`key_value_heads`, `kv_width`, `projection_shapes`).
   - numpy's attend expanding `K` and `V` to `h` heads, its backward folding per group, and its
     rotary on `K` by `g` heads.
   - numpy's `AttentionCache` at `g * d_k`.
   - Tests: `g = h` bit-identical (golden run, unchanged); gradient checks at `g` in {1, 2, h} on the
     shared spec tables (`MULTI_HEAD`, `SEQUENCE`, `TOKENS`); the fold order (a test that pins D3's
     order against a different one); decoding at each `g` against the full pass; refusals.
3. **The crate** (D4): the ops' `key_value_heads`, block `i / (h / g)` in `scores` and attend, the
   group's tiles cut once per block, the backward's fold, `K`'s rotation by `g` heads,
   `attention_step`'s cache at `g * d_k`. Its tests: by bits against numpy's expressions at
   `g` in {1, 2, h}, and new `bits.txt` families. Then a "Bump rust/" PR, with tier 1 on `jebel`
   (`attention_timing`, `g = h`, `--order ONNO`).
4. **The Rust layers:** `key_value_heads` through `AttentionRustArrayLayer` and
   `AttentionRustCache`; parity with numpy (crate `exp`).
5. **Pure Python:** `AttentionLayer` (its `_tiled`, backward fold, rotary and cache); parity.
6. **The conversion and the decode tool** (D5, D11): `to_grouped`, tested on each implementation
   (a converted `g = h` model is the original, bit for bit; a converted model's `Wk` rows are its
   heads' means); `generation_timing.py`'s options.
7. **Fixtures and golden entries** (D6): `SequenceGrouped<class>` fixtures (`g = 2` of 4 heads);
   the 4-head golden entries at `g = 2` and `g = 1`, and the Rust reference.
8. **The study** (D7): registered first (claims, cells, feasibility, budget from a probe), then
   the sweep, then the findings.
9. **Docs**: README (Layer norm and attention gains GQA); the workplan retired.
