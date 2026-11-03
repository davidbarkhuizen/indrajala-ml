# Workplan: cross-attention and an encoder-decoder model (roadmap step 15)

**Status: decisions D1-D10 settled (2026-10-10) by building for extensibility, speed and real-world
production (the owner's direction, 2026-10-10). Revised 2026-11-03 at the owner's direction (windows
of 1,024, more texts): D4 rewritten on a measurement of Perseus's parallel texts, D11 new (packed
pairs), D3 and D9 amended to match; D12 (the model's width and head size) left to the study's
registration. Stage 1 (this plan) done. Stage 2 (2026-11-03): the dataset repository, its fetch
entry and the loader of packed pairs here (D13 new: a packed pair's layout). The same day, at the
owner's direction, D4's second corpus became Plutarch's *Lives* in place of the New Testament, and
Pausanias was dropped: the repository at `v2026-11-03.2` (13,122 pairs of the histories, 9,794 of
the *Lives*; its `report.md` the alignment report).**

Roadmap step 15 ([primitives-roadmap.md](../../primitives-roadmap.md)): attention from one sequence to
another, and the encoder-decoder transformer it builds (Vaswani et al. 2017, §3.1, §3.2.3), on a
real translation task: Greek to English, the Greek historians and Plutarch's *Lives* aligned
section by section from Perseus's parallel texts (D4). All three implementations
and the crate.

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
| data | one text per corpus | aligned Greek-English pairs, packed into windows of 1,024 (D4, D11) |
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
  memory's padding (step 11's segment 0) and, in a packed window, by segment: a target token reads
  only its own pair's source (D11). Never causal, no rotary positions (the two sequences'
  positions are unrelated; T5 and BART add none in cross-attention). It sits in a decoder block
  between causal self-attention and the FFN, pre-LN, as `Residual((LayerNorm(), CrossAttention()))`.
  The crate's attention blocks take a separate key/value input; step 12's tiling applies as is.
- **D4. The data. Settled: (a) (revised 2026-11-03; the first choice is now (b)).**

  **The measurement (2026-11-03).** At the commit the Euclid corpus pins
  (`70427323b61267cdcd0256b9b02009c23392de07`), 592 of `PerseusDL/canonical-greekLit`'s works have
  both a Greek text and an English translation. Many prose works carry the same book.chapter.section
  numbering in both. The probe paired each work's deepest numbered divisions by citation, dropping
  notes, headings and figures, and measured characters per side. The dataset repository's alignment
  report supersedes these numbers.

  | work (English) | pairs | Greek / English chars | median pair | fit 512 | fit 1,024 |
  | --- | --- | --- | --- | --- | --- |
  | Herodotus (Godley, 1920-25) | 4,326 sections | 1.16M / 1.47M | 262 / 320 | 90% | 100% (longest 1,321) |
  | Thucydides (measured on Hobbes; Jowett and Crawley also held) | 3,575 | 0.96M / 1.21M | 245 / 311 | 87% | 100% |
  | Pausanias (W. H. S. Jones) | 3,170 | 1.35M / 1.51M | 418 / 464 | 64% | 100% |
  | Xenophon, *Anabasis* (Brownson) | 1,457 | 0.35M / 0.48M | 230 / 317 | 93% | 100% |
  | Xenophon, *Hellenica* (Brownson) | 1,146 | 0.43M / 0.57M | 363 / 486 | 56% | 99% |
  | Xenophon, *Cyropaedia* (Miller) | 1,408 | 0.48M / 0.62M | 318 / 417 | 71% | 98% |
  | Diogenes Laertius (Hicks) | 1,211 | 0.70M / 0.89M | 575 / 731 | 4% | 98% |
  | New Testament (Westcott-Hort; the World English Bible), Matthew and Luke | ~1,070 and 1,150 verses | 0.11M / 0.12M each | ~100 / 105 | 100% | 100% |
  | Plutarch's *Lives* (Perrin), *Alexander* and *Pompey* | ~380 a Life | ~0.13M a Life | ~350 / 450 | 71% | 100% |
  | Demosthenes, *False Embassy*; Aeschines, *Against Ctesiphon* | 338; 260 | 0.15M; 0.12M | ~420 / 520 | 48% | 99-100% |
  | Euclid (Heath), by proposition | 471 | 0.56M / 0.61M | 1,016 / 1,103 | - | 47% |

  These need a statistical aligner: Plato and Aristotle (Stephanus or Bekker pages of about 2,000
  characters, or levels that differ between the two files); Polybius, Josephus and Epictetus
  (numbering that differs between the Greek and the English); Homer (Greek by line, English by
  passage); and Strabo, Appian, Athenaeus, Galen and Apollodorus (most units above 1,024).

  - (a) *Chosen.* A new repository, `indrajala-datasets-perseus-parallel` (the owner's to create).
    Its `clean.py` builds pairs from the pinned commit at the deepest citation both files share,
    cleaned as the corpora are. Two corpora (revised 2026-11-03, the owner: the New Testament
    dropped, Pausanias dropped rather than kept in a private repository, Plutarch's *Lives* in
    Perrin's translation as the second corpus):
    - **The histories:** Herodotus (Godley), Thucydides (the translation the cleaner measures as
      aligning best), Xenophon's *Anabasis*, *Hellenica* and *Cyropaedia*, and Diogenes Laertius.
    - **The *Lives*:** Plutarch's *Parallel Lives* with their comparisons, 66 works, Perrin's Greek
      and English (Loeb, 1914-26).

    The orators (63 Demosthenes speeches have both texts) are a reserve for scale, not built.

    **Built (stage 2, 2026-11-03),** `indrajala-datasets-perseus-parallel` at `v2026-11-03.2`; its
    `report.md` is the alignment report and its `LICENSE-TEXT.md` the licence check per volume.
    - **The histories:** 13,122 pairs, 4.08M Greek and 5.00M English characters (train 11,753,
      validation 737, test 632).
    - **The *Lives*:** 9,794 pairs, 3.30M Greek and 4.22M English characters (train 8,841,
      validation 507, test 446). Every unit aligns after renames in *Solon*, *Camillus* and
      *Demetrius*.
    - **Thucydides:** Jowett's translation isn't at the pinned commit. Of the three that are
      (Crawley, C. F. Smith, Dale), Crawley aligns best.
    - **Splits** hold out whole books of the histories, a different book of each work, and two
      pairs of *Lives* whole with their comparisons (*Pericles* and *Fabius Maximus*, *Timoleon*
      and *Aemilius Paulus*). Each held-out book or Life is cut at its middle pair into
      validation, then test. The report finds no near-duplicates across the split.
    - **Dropped from the first build (`v2026-11-03`):**
      - **The New Testament:** the WEB translates the Majority Text, not Westcott and Hort's.
      - **Pausanias:** his translation is in copyright under life-plus-70 terms until 2034 and 2035.

    **Licences:** Perseus's encodings are CC BY-SA 4.0. Each translation's copyright was checked per
    volume and recorded in the repository. Every one here is in the public domain in the United
    States and under terms of life plus 70 years: Godley, Crawley, Brownson, Miller, Hicks and
    Perrin.

    Pros:
    - Real parallel text, aligned by citation, with no statistical aligner.
    - About eleven times the first plan's pairs, nearly every pair fitting a 1,024 window.
    - A size that supports a model of a few hundred thousand to a million parameters, which the
      ~1M-character corpora don't.
    - Two backdrops in two Greeks, so a result is checked on two (our rule for a general
      architecture): classical Attic and Ionic history, and the Lives' later Greek of the imperial
      age.
    - Every text in the public domain under both terms.

    Cons:
    - The two backdrops are both narrative prose, so they differ less than the New Testament did.
    - Pairs of 250-450 characters mostly pad a window of 1,024 unless packed (D11).
  - (b) The first plan's choice: Herodotus by book and chapter, and Euclid by proposition. Pros:
    two styles, narrative and formal proof. Cons, measured: about 2,000 pairs; chapters of several
    hundred to over 1,000 characters; Euclid's propositions fit 1,024 in only 47% of cases. Too
    small and too long for the windows this framework trains.
  - (c) The New Testament alone. Pros: exact verse alignment, short pairs. Cons: one register and one
    backdrop; a religious text as the only one; its paired English (the WEB) translates a
    different Greek text.
  - (d) A standard translation corpus (WMT, Europarl). Pros: production scale. Cons: millions of
    pairs, beyond what this framework trains in float64 on one CPU (about a million parameters a day
    on `jebel`); licences vary; not the classical texts the corpora come from.
  - (e) Herodotus's Greek with Rawlinson's English (the corpus we hold). Cons: Rawlinson's
    paragraphs don't follow the numbering, so alignment needs an aligner; Godley is aligned already.
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
- **D9. The study. Settled (amended 2026-11-03: D4's two corpora, the histories and the *Lives*;
  D11's windows).** On each of D4's two corpora, in D11's packed windows of 1,024, an encoder-decoder against
  a decoder-only model that reads the Greek as a prefix before a separator and then writes the
  English (the other production form, with loss on the English only), at equal parameters; and
  each against a decoder-only English model with no source (what the source buys). Bits per
  character and chrF, 5 seeds.
- **D10. Out of scope.** Pretraining the encoder on Greek alone, multilingual tokenizers (BPE over
  both scripts), beam search: each its own step once the study shows where the gains are.
- **D11. Windows and packing. Settled: (a) (new 2026-11-03).**
  - (a) *Chosen.* Windows of 1,024 on both sides (step 12's longest), several pairs packed into a
    window as step 11 packs paragraphs. A source window and a target window hold the same pairs in
    the same order, each pair one segment on both sides. `CrossAttention` masks by segment: a
    target token reads only the source tokens of its own segment (D3), and the decoder's
    self-attention keeps step 11's document mask. A pair longer than a window (about 70 in the
    histories by the measurement: Cyropaedia's, Hellenica's and Diogenes' longest; recorded by the
    alignment report) is dropped, never cut. Decoding (D7) encodes one
    source at a time, so its mask is padding only.
    Pros:
    - Every measured pair fits, so no data is lost to the window.
    - Packing keeps the padding low at medians of 250-450 characters.
    - The segment mask is step 11's machinery, reused, with the query's and the key's segments
      compared.
    Cons:
    - A packed pair's cross-attention mask is per segment, not just padding: a mask input to attend
      in every implementation and the crate.
    - Windows of 1,024 cost about 1.3 times 256's per token in step 12's study.
  - (b) Windows of 512, one pair a window, padded. Pros: no segment mask in cross-attention; about
    half the attention cost of 1,024. Cons: drops 7-44% of the histories' pairs (all but 4% of
    Diogenes'), and pads the rest.
  - (c) Windows of 1,024, one pair a window. Pros: no segment mask. Cons: windows mostly padding
    (medians of 250-450 characters), several times the compute for the same pairs.

- **D12. The model's width and head size. Open until the study's registration (new 2026-11-03).**
  Today's sequence models are `d = 64` with 4 heads of 16 (`d_k = 16`); production heads are 64 or
  128 wide. A standalone `d_k = 64` probe on today's model was declined (the owner, 2026-11-03): at
  `d = 64` a head of 64 doubles a model already at the ~1M-character corpora's data limit. D4's
  corpora (about 6M characters a side) support `d = 128` to `256`, where heads of 32 to 64 come
  from the width. So the encoder's and decoder's `d`, heads and `key_size` are set when the study
  (D9) is registered, sized by that data and by `jebel`'s time, with `d_k` of at least 32.

- **D13. A packed pair's layout. Settled: (a) (new 2026-11-03, at stage 2).** Step 11's rule
  makes a separator end its paragraph: it belongs to the paragraph before it, and its target, the
  next paragraph's first character, counts. In a target window that would predict each pair's
  first English character from a position of the previous pair's segment, so its cross-attention
  would read the previous pair's Greek.
  - (a) *Chosen.* The source window packs each pair's Greek followed by the separator, as step
    11 packs paragraphs, so the separator ends its pair. The target row is the separator, then
    each pair's English followed by the separator: the decoder's input is the row less its last
    token, its labels the row less its first (step 9's shift), so a separator is both the end of
    one pair and the start of the next. The decoder's segments take the opposite rule: a target
    separator opens the pair after it. Pair k is segment k on both sides, and every English
    character, the first included, is predicted from its own pair's Greek; the last label of each
    pair is the separator, which ends a generated translation (D7). Pads go before a row's first
    token (step 11's D8), and a window holds whole pairs only.
    Pros:
    - Every target counted is predicted from its own pair's source.
    - No new special token: the separator is the start and the end token, as `</s>` is in T5's and
      BART's decoders.
    - Decoding (D7) starts each translation from one separator and stops at the next.
    Cons:
    - A second segment rule (a separator opens its segment) beside step 11's, for the decoder's
      side of an encoder-decoder network: a flag to segments in every implementation and the crate.
  - (b) A third special token, a start token: each target pair is the start token, the English and
    the separator, with step 11's rule on both sides. Pros: one segment rule. Cons: a token more a
    pair; the start token's position is the previous pair's last label, a target that must not
    count, so a new rule for the loss instead of for segments.
  - (c) Step 11's layout as it is, the separator ending each target pair. Pros: no change. Cons:
    each pair's first English character is predicted from the previous pair's segment and source,
    and the window's first pair's first character not at all.

- **D14. The encoder-decoder's specs. Settled: (a) (new 2026-11-03, at stage 3).**
  - (a) *Chosen.* `EncoderDecoder(encoder, decoder)`, two lists of layer specs, beside the
    Sequential list.
    - **The encoder:** an `Embedding` with segments (step 11's rule: a separator ends its pair),
      token layers whose attention is never causal, and a final `LayerNorm`, so the memory is
      normalised as a pre-LN stack's output is. No output layer.
    - **The decoder:** a sequence model.
      - An `Embedding` with segments and `separator_opens=True` (D13).
      - Causal self-attention only.
      - At least one `CrossAttention`, which ends a residual block's body as `Attention` does,
        `Residual((LayerNorm(), CrossAttention(...)))`.
      - A token-wise softmax output layer.
    - **`CrossAttention(heads, key_size, key_value_heads, dropout)`:** never causal, never rotary
      (D3). Its keys and values are projected from the memory's width, which may differ from the
      decoder's, and the shape walk records that width for the builders. Its parameters are named
      as an `Attention`'s: Wq, bq, Wk, bk, Wv, bv, Wo, bo.
    - **Anywhere else** (a Sequential network, a window), a `CrossAttention` is refused: there is
      no memory to read.
    - **Format 2:**
      - Shape `"encoder_decoder"`: `"encoder"` holds the encoder's layer entries, `"layers"` the
        decoder's.
      - `"input"` is `{"source": {"dimension": S}, "target": {"dimension": T}}`.
      - One index space for weights, optimizer state and snapshots: `expand_specs(encoder +
        decoder)`, the encoder's layers first.
      - An entry has `"kind": "cross_attention"`, with its fields only when not the default.
        `"separator_opens": true` appears on an `Embedding` only when set, so earlier files don't
        change.
    Pros:
    - Each stack is validated and shape-walked as today's lists are, so their rules carry over.
    - One index space keeps the optimizers', checkpoints' and snapshots' code as it is.
    - A decoder-only model is the decoder list alone, with no `CrossAttention`.
    Cons:
    - A second network spec beside the list.
    - Validation and the shape walk need an encoder-decoder entry point.
  - (b) One flat list with a marker spec between the stacks. Pros: one list. Cons: the decoder's
    `Embedding` stands mid-list (today the first layer starts the token part), and the shape walk
    would cross from the memory back to the decoder's ids.
  - (c) A general graph of layers. Cons: D2's (b).

## Stages

1. **This workplan.**
2. **The dataset repository** (D4), the owner's to create; its `clean.py`, the licence check per
   volume, the alignment report (pairs, lengths, the pairs dropped by D11, the cross-split
   near-duplicates) and tag; `scripts/data/fetch_datasets.py`'s entry; a loader of packed pairs
   (D11, D13). Done 2026-11-03.
3. **Specs**: `CrossAttention`, `EncoderDecoderNetwork`, validation, format 2; the decoder's
   segment rule (D13: a target separator opens its pair).
4. **numpy**: the network (D2), `CrossAttention` (D3); tests by hand (with one memory token, every
   weight is 1; padding in the memory changes nothing).
5. **The crate**: the key/value input through the attention ops; "Bump rust/" with its A/B.
6. **Rust layers**; 7. **Pure Python**; parity.
8. **Decoding** (D7) and chrF (D6).
9. **Fixtures, checkpoints and golden entries** (D8).
10. **The study** (D9).
11. **Docs**; the workplan retired.
