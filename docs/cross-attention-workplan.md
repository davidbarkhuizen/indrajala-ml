# Workplan: cross-attention and an encoder-decoder model (roadmap step 15)

**Status: decisions D1-D10 settled (2026-10-10) by building for extensibility, speed and real-world
production (the owner's direction, 2026-10-10). The dataset repository (D4) is the owner's to
create, as the corpora's were. Stage 1 (this plan) done.**

Roadmap step 15 ([primitives-roadmap.md](primitives-roadmap.md)): attention from one sequence to
another, and the encoder-decoder transformer it builds (Vaswani et al. 2017, §3.1, §3.2.3), on a
real translation task: Greek to English, Herodotus and Euclid, aligned passage by passage from the
same Perseus sources the corpora came from. All three implementations and the crate.

## Why

Every model so far reads one sequence. Production tasks often condition on a second: translation,
summarization, speech or image to text, retrieval-augmented generation. Cross-attention is how a
transformer reads it: queries from the sequence being generated, keys and values from an encoder's
output. It is the last of step 5's named extension points (next-steps.md, From multi-head
attention), and it needs what the steps before it built: padding masks for sources of different
lengths (step 11), a context to carry the encoder's output to the decoder's layers (step 11's
forward context), long windows (step 12), and the decoder with its cache (step 9).

## What exists, and what changes

| piece | today | this plan |
| --- | --- | --- |
| a network | a list of layer specs, one input | an encoder stack and a decoder stack (D2) |
| attention | self-attention, causal or not | plus `CrossAttention`, queries from the decoder, keys and values from the encoder's output (D3) |
| per-batch information | step 11's forward context: segments | plus the encoder's output and its segments (`memory`) |
| data | one text per corpus | aligned Greek-English pairs (D4) |
| tokenizers | one per model (step 9's file) | a source and a target tokenizer (D5) |
| evaluation | bits per character, accuracy | plus chrF on generated translations (D6) |

## Decisions (settled 2026-10-10)

Each lists the options considered, with pros and cons, and the choice.

- **D1. Scope. Settled: (a).**
  - (a) *Chosen.* `CrossAttention`; an encoder-decoder network; the aligned dataset; source and
    target tokenizers; decoding with the encoder's keys and values computed once; chrF; all three
    implementations and the crate; fixtures, checkpoints, golden entries; a study (D9). Pros: the
    production form, on real data. Cons: the largest step on the roadmap.
  - (b) Cross-attention on a synthetic task (copying, reversal). Cons: a toy that decides nothing
    (the owner's direction, 2026-10-10); says nothing about real sources.
- **D2. The network. Settled: (a).**
  - (a) *Chosen.* `EncoderDecoderNetwork(encoder=[specs], decoder=[specs])`: two Sequential stacks;
    the encoder's output and its segments go to the decoder through the forward context
    (`context.memory`); the decoder's `CrossAttention` layers read it; the backward sums the
    memory's gradient over the cross-attention layers (a left fold in layer order) and runs it back
    through the encoder. Format 2 saves both stacks. Pros: reuses the Sequential machinery and
    step 11's context; a decoder-only model is the same network without an encoder. Cons: a
    second network shape beside Sequential.
  - (b) A general graph of layers. Pros: any topology. Cons: a framework rewrite no use yet needs;
    the two-stack network covers every encoder-decoder in production (T5, BART, Whisper).
- **D3. `CrossAttention`. Settled.** `CrossAttention(heads, key_size, key_value_heads)` (step 14's
  grouping included): `Q` from the decoder's tokens, `K` and `V` from `memory`, masked by the
  memory's padding (step 11's segment 0), never causal, no rotary positions (the two sequences'
  positions are unrelated; T5 and BART add none in cross-attention). It sits in a decoder block
  between causal self-attention and the FFN, pre-LN, as `Residual((LayerNorm(), CrossAttention()))`.
  The crate's attention blocks take a separate key/value input; step 12's tiling applies as is.
- **D4. The data. Settled: (a).**
  - (a) *Chosen.* A new repository, `indrajala-datasets-perseus-parallel` (the owner's to create),
    whose `clean.py` builds aligned pairs from `PerseusDL/canonical-greekLit` at the commit the
    Euclid corpus pins: Herodotus's Greek (`tlg0016.tlg001.perseus-grc2`) with Perseus's English
    (`perseus-eng2`, A. D. Godley's translation, 1920-1925) by book and chapter, and Euclid's Greek
    (`tlg1799.tlg001.perseus-grc2`) with Heath's English (`perseus-eng2`) by book and proposition
    (and definition, postulate, common notion), cleaned as the corpora are (Heath's notes and
    figures dropped). Perseus's encodings are CC BY-SA 4.0; the translations are public domain in
    the United States. The pair counts and the alignment's exceptions are recorded with the
    repository, and the split holds out whole chapters and propositions. Pros: real parallel text
    in two styles (narrative, formal proof), from sources already pinned; citation alignment, no
    statistical aligner. Cons: about two thousand pairs, small for translation, so the study
    measures what attention to a source buys, not a usable translator; long pairs (a chapter is
    hundreds of characters) need step 12's windows.
  - (b) A standard translation corpus (WMT, Europarl). Pros: production scale. Cons: millions of
    pairs, beyond what this framework trains in float64 on one CPU; licences vary.
  - (c) Herodotus's Greek with Rawlinson's English (the corpus we hold). Cons: Rawlinson's
    paragraphs don't follow the chapter numbering (1,856 paragraphs against about 1,500 chapters),
    so alignment needs an aligner.
- **D5. Tokenizers. Settled.** A source tokenizer (Greek characters, accents kept: they carry
  meaning in the source) and a target tokenizer (step 9's English characters, step 11's special
  tokens), each a file beside the model with its `kind`.
- **D6. Evaluation. Settled.** Target bits per character under teacher forcing, and chrF
  (Popović 2015, character n-gram F-score) of greedy and sampled translations of the held-out
  sources: the character-level metric production MT reports beside BLEU, and the right one for
  character models.
- **D7. Decoding. Settled.** Step 9's decoder with the source encoded once; each `CrossAttention`
  layer's keys and values computed once per source and kept for every step (the cross-attention
  cache); batched with step 11's padding.
- **D8. Golden run and timing. Settled.** New entries: a small encoder-decoder in each
  implementation; earlier entries bit-identical. The crate's attention gains a key/value input:
  tier 1 A/B of self-attention, which must not move.
- **D9. The study. Settled.** On the parallel data at step 12's context, an encoder-decoder against
  a decoder-only model that reads the Greek as a prefix before a separator and then writes the
  English (the other production form, with loss on the English only), at equal parameters; and
  each against a decoder-only English model with no source (what the source buys). Bits per
  character and chrF, 5 seeds.
- **D10. Out of scope.** Pretraining the encoder on Greek alone, multilingual tokenizers (BPE over
  both scripts), beam search: each its own step once the study shows where the gains are.

## Stages

1. **This workplan.**
2. **The dataset repository** (D4), the owner's to create; its `clean.py`, alignment report and
   tag; `scripts/fetch_datasets.py`'s entry; a loader of pairs.
3. **Specs**: `CrossAttention`, `EncoderDecoderNetwork`, validation, format 2.
4. **numpy**: the network (D2), `CrossAttention` (D3); tests by hand (with one memory token, every
   weight is 1; padding in the memory changes nothing).
5. **The crate**: the key/value input through the attention ops; "Bump rust/" with its A/B.
6. **Rust layers**; 7. **Pure Python**; parity.
8. **Decoding** (D7) and chrF (D6).
9. **Fixtures, checkpoints and golden entries** (D8).
10. **The study** (D9).
11. **Docs**; the workplan retired.
