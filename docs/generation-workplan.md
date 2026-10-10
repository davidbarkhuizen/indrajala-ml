# Workplan: generation with a key/value cache (roadmap step 9)

**Status: decisions D1-D9 settled by the owner (2026-10-10), building for extensibility and
speed. Stage 1 (this plan) done. Starts after step 8 (rotary positions,
[rotary-positions-workplan.md](rotary-positions-workplan.md)), whose rotated keys this plan's
cache keeps valid as its window slides.**

Roadmap step 9 ([primitives-roadmap.md](primitives-roadmap.md)): sampling text from a trained
sequence model, character by character, through an incremental decoding path: each new character
costs one token's pass, with every earlier token's keys and values cached.

## Why

The sequence task measured its models by held-out loss alone (its D11 left generation out). A
loss says how well a model predicts; a sample shows what it predicts: whether it spells, keeps
Euclid's "I say that" and its letter names, closes Shakespeare's speeches. It will show what
steps 7 (dropout), 8 (rotary positions) and 10 (longer training) buy.

Measured on `pyramidon` (one thread, the sequence study's 2-layer model, vocabulary 65), one full
pass over a 64-token window takes 2.5 ms in numpy and 1.5 ms in Rust: 1.5 to 2.5 seconds per
1,000 characters without a cache. A cache makes a character one token's pass through the
projections and FFNs and one row of attention scores, about `T` times less work. Incremental
decoding is also framework structure every later sequence feature uses (longer contexts,
cross-attention's decoder, step 14).

## What exists, and what changes

| piece | today | this plan |
| --- | --- | --- |
| a model's output | `predict_probabilities(state)`: all `T` tokens' probabilities, one full pass | `prefill(ids)` then `step(id)`, each the last token's probabilities (D2) |
| the token layers | a batch of whole windows | each also decodes: a prompt's tokens, then one token at a time (D2) |
| attention | `(T, T)` scores per head per pass | a cache of rotated keys and values, one row of scores per step (D3) |
| sampling | none | a chain of filters, then one seeded draw (D4) |
| the vocabulary | rebuilt from the corpus; not saved with a model | a tokenizer file beside the model (D5) |
| trained models | the studies discard them | `scripts/sample_text.py train` saves one (D6) |

## Decisions (settled 2026-10-10)

Each lists the options considered, with pros and cons, and the choice. D1 was the owner's pick;
D2-D9 follow from building for extensibility and speed (the owner, 2026-10-10), and the position
scheme behind D3 was settled by adding step 8.

- **D1. Scope. Settled: (b).**
  - (a) A sampling module and script over `predict_probabilities`, a full pass per character.
    Pros: no network, crate or format change. Cons: `T` times the work per character; no decoding
    path for later steps to build on.
  - (b) *Chosen.* (a) plus incremental decoding with a key/value cache (D2, D3), in all three
    implementations and the crate. Pros: the way decoders run; one token's pass per character;
    the structure later sequence work reuses. Cons: a second forward path through every token
    layer, with parity against the full pass.
  - (c) (a) plus a `./cli demo` entry. Cons: a demo must train (about 7 minutes) or ship a model,
    and its claim must stay true; nothing for extensibility or speed.
- **D2. The decoding protocol. Settled: (a).**
  - (a) *Chosen.* A network's `decoder()` returns a `Decoder` holding one cache per layer:
    `prefill(ids)` runs the prompt's `L` tokens (positions `0` to `L - 1`) and returns the last
    token's probabilities, `step(id)` appends one token and returns its probabilities. Each token
    layer implements `prefill` and `step` on its own cache: attention keeps keys and values;
    embedding, layer norm, token-wise dense, the residual fork and add, dropout (identity at
    inference) and the token-wise output keep nothing. A layer that can't decode (`TokenMean`,
    `Patches`, a non-causal attention, whose earlier outputs would depend on later tokens) refuses
    at `decoder()`. Pros: each layer owns its decode, as it owns its forward; a new token layer adds
    two methods; the prompt is the prefill, so a short prompt needs no fill or padding. Cons:
    every token layer in three implementations gains two methods.
  - (b) A network-level decode loop that recomputes the full pass for the window. Cons: that is
    D1 (a).
- **D3. The cache and the sliding window. Settled: (a).**
  - (a) *Chosen.* Attention's cache is a ring of `T` rotated keys and values per head. With
    rotary attention (step 8) the window slides exactly: past `T` tokens, the oldest key and value
    are dropped and the new token's position keeps counting (its rotation from step 8's table,
    extended past `T`), so scores depend on distances up to `T - 1`, as in training. A model with a
    learned `Position` decodes from its cache until the window fills, then falls back to a full
    pass over the last `T` tokens per character, exact but uncached, and says so once. Pros:
    exact for both position schemes; fast for rotary models, which step 8's D9 makes the default if
    they match. Cons: learned-position models get no speedup past `T`.
  - (b) For learned positions, re-prefill the last `T / 2` tokens when the window fills. Pros:
    one full pass per `T / 2` characters. Cons: the context drops to `T / 2` at each refill: a
    model that sees less than it could, silently.
- **D4. Sampling. Settled: (a).**
  - (a) *Chosen.* A chain of filters on one position's probabilities, then one draw: temperature
    (a softmax of `log p / temperature`; 0 is the argmax), top-k (the `k` most likely, ties by id)
    and top-p (the smallest set, in descending probability, whose mass reaches `p`; Holtzman et
    al. 2019), each a filter, renormalized after; then one uniform draw `u` from a generator seeded
    by `--seed` and the first id whose cumulative probability (a left fold in id order) exceeds
    `u`. Pros: the standard knobs; a new sampler is a new filter; deterministic and testable by
    hand. Cons: a cumulative sum's rounding decides a draw at a boundary, so numpy and Rust models
    can sample different text from one seed (D7).
  - (b) Temperature and top-k only. Cons: a fixed sampler, not a chain.
  - (c) Greedy only. Cons: tends to repeat itself; shows the mode, not the distribution.
- **D5. The vocabulary. Settled: (a).**
  - (a) *Chosen.* A tokenizer file beside the model, `<model>.tokenizer.json`:
    `{"kind": "characters", "corpus": ..., "sha256": ..., "symbols": ...}`, the corpus's name and
    its pinned checksum (`scripts/fetch_datasets.py`), the symbols in id order; `sample` refuses one
    whose size isn't the model's output size. Pros: `kind` admits a BPE or word-level tokenizer
    later without touching format 2; the usual model and tokenizer split. Cons: two files to keep
    together.
  - (b) A `"vocabulary"` field in format 2. Cons: a property of the data stored as the network's,
    bound to character tokens.
  - (c) Rebuild from `--corpus`. Cons: a model and a corpus can be paired wrongly.
- **D6. Where models come from. Settled: (a).**
  - (a) *Chosen.* `scripts/sample_text.py train --corpus C --epochs N --seed S --out M`: the
    sequence study's 2-layer arm (`sequence_study.arm_specs`) at its tuned rate, with step 8's
    default positions (its D9) and optionally step 7's dropout, numpy or Rust, saving the model
    and its tokenizer; `sample --model M --prompt ... --length N --temperature --top-k --top-p
    --seed`. Pros: one command to a model, at the studies' settings. Cons: the script depends on
    the study's module, whose arms are the reference models.
  - (b) The study scripts gain `--save`. Cons: a sweep saves dozens of models to get one.
  - (c) Commit a trained model. Cons: megabytes of JSON, stale as the models improve.
- **D7. Parity. Settled.** The decoder against its own implementation's full pass: pure Python and
  the crate by bits (pure Python's products are left folds, a row at a time; the crate's, if each
  output row is computed alone, which stage 4 checks before claiming it); numpy within rounding (a
  one-row product and a whole-window product may take different BLAS kernels), never as a
  tolerance on anything but the products. Samples: the same seed and model file give the same
  text; no claim that numpy and Rust sample the same text (D4).
- **D8. Golden run and timing. Settled.** No training path changes: no golden entries. The crate
  gains decode ops (a one-token attention step on the cache, with the rotation); if its existing
  attention ops are refactored to share blocks with them, a tier 1 A/B of the attention case,
  whose path must not move. The speedup is measured within one tree: a generation benchmark,
  characters per second cached against uncached, numpy and Rust, on `jebel`, in the PR.
- **D9. The README's samples. Settled: (a).** (a) A Generation section with one sample each from
  Tiny Shakespeare and Euclid, quoted with the command, seed, sampling settings and the model's
  held-out bits per character; replaced when step 10's longer training gives better ones. (b) The
  command only. Cons: shows nothing generation is for.

## Stages

1. **This workplan.**
2. **The sampler** (D4): the filter chain and the draw, pure functions, tested by hand.
3. **numpy decoding** (D2, D3): `decoder()`, every token layer's `prefill` and `step`, attention's
   ring cache; parity with the full pass (D7); the refusals.
4. **The crate**: the decode ops (D8), numpy's results; then a "Bump rust/" PR, with the tier 1 A/B
   if the existing ops changed.
5. **Rust layers**: decoding through the ops; parity by bits with Rust's full pass.
6. **Pure Python**: decoding; parity by bits with its full pass.
7. **The script and the tokenizer file** (D5, D6), and the generation benchmark (D8).
8. **Docs** (D9): README, roadmap, next-steps; the workplan retired.
