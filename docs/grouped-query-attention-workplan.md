# Workplan: grouped- and multi-query attention (roadmap step 14)

**Status: decisions D1-D8 settled (2026-10-10) by building for extensibility, speed and real-world
production (the owner's direction, 2026-10-10). Stage 1 (this plan) done.**

Roadmap step 14 ([primitives-roadmap.md](primitives-roadmap.md)): attention whose `h` query heads
share `g` key/value heads (Ainslie et al. 2023, "GQA"; `g = 1` is multi-query attention, Shazeer
2019), in all three implementations and the crate, measured where it pays: the key/value cache of
step 9's decoder at step 12's context lengths.

## Why

A decoder's key/value cache holds `2 * h * d_k` values per token per layer; at long context and
large batch it, not the weights, bounds how many sequences a server decodes at once and how fast
(each step reads the whole cache). Sharing key/value heads divides it by `h / g` at a small cost
in quality; Llama 2's 70B model and Llama 3 at every size use GQA with `g = 8`. Step 5 named its
place (next-steps.md, From multi-head attention: `Wk`, `Wv` with `g` row blocks; head `i` reads
block `i // (h / g)`). Before step 9 there was no cache to shrink, which is why the roadmap first
judged it unmeasurable; with step 9's cache and step 12's long windows it has a production
question to answer.

## Decisions (settled 2026-10-10)

Each lists the options considered, with pros and cons, and the choice.

- **D1. Scope. Settled: (a).**
  - (a) *Chosen.* `Attention(key_value_heads=g)`, `g` dividing `heads`, default `heads` (today's
    attention, bit for bit); all three implementations and the crate; the decoder's cache sized
    by `g`; converting a trained multi-head model to grouped heads (D5); fixtures, checkpoints,
    golden entries; a study (D7). Pros: the production form with its conversion path. Cons: the
    attention ops' key/value indexing changes in every implementation.
  - (b) Multi-query only (`g = 1`). Cons: the quality loss Ainslie et al. measured; production
    settled on groups.
- **D2. The layout. Settled.** As step 5 named it: `Wk` and `Wv` are `(g * d_k, d)` with `g` row
  blocks, `Wq` stays `(h * d_k, d)`, and query head `i` attends with key/value block
  `i // (h / g)`. Format 2 writes `"key_value_heads"` only when it isn't `heads`.
- **D3. The backward. Settled.** A shared key/value head's gradient is the sum over its group's
  query heads, a left fold in head order, in every implementation (so parity holds by bits where
  it did).
- **D4. The crate. Settled: (a).** (a) The attention ops take `key_value_heads` (keyword, default
  `heads`), indexing the shared blocks in attend and summing in the backward; with step 12's tiled
  attention, a key/value block is loaded once per group. (b) Expanding `K` and `V` to `h` heads
  before attend. Cons: copies the memory GQA exists to save.
- **D5. Converting trained models. Settled.** `to_grouped(network, g)`: each group's key/value
  projection rows mean-pooled from its heads' (Ainslie et al. 2023, §2.1, then a short
  "uptraining" at a fraction of the original steps). Pros: production's path to GQA from an
  existing model. Cons: a function to keep in step with format 2.
- **D6. Golden run and timing. Settled.** New entries with `g = 2` and `g = 1` in each
  implementation; earlier entries bit-identical. The attention ops change: tier 1 A/B at
  `g = heads`. The decode speedup is claimed: tier 2 on the generation benchmark (step 9).
- **D7. The study. Settled.** At step 12's longest context, a model with 8 query heads (`d_k = 8`
  at `d = 64`, so `g` has room: 8, 4, 2, 1), trained from scratch at each `g` and uptrained from
  the `g = 8` model (D5), on Herodotus, Euclid and the *Muqaddimah*; held-out bits per character,
  the cache's bytes per token, and decode tokens per second at batch 1 and batch 16 on `jebel`.
- **D8. Out of scope.** Multi-head latent attention (DeepSeek-V2's compressed cache) and cache
  quantization: further cache reductions, each its own step if the study shows the cache binding.

## Stages

1. **This workplan.**
2. **Specs and numpy** (D2, D3); 3. **The crate** (D4) and a "Bump rust/" PR with its A/B;
4. **Rust layers**; 5. **Pure Python**; parity.
6. **The decoder's cache by `g`** (after step 9) and the conversion (D5).
7. **Fixtures, checkpoints and golden entries** (D6).
8. **The study** (D7).
9. **Docs**; the workplan retired.
