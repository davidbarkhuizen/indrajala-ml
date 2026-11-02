# Workplan: grouped- and multi-query attention (roadmap step 14)

**Status: decisions D1-D8 settled 2026-10-10; revised 2026-11-02 against the library as steps 9 to
13 left it (D1, D4, D6 and D7 corrected, D2, D3 and D5 made exact, D9 to D12 new; D13, the study's
FFN, from the literature), settled by
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
    document mask, the recipe, the token budget, 30 epochs on Rust), with SwiGLU in every FFN
    (`GatedDense(170, "silu")`, step 13's equal-parameter width; D13), at 8 query heads of
    `d_k = 8`, so `g` has room: 8, 4, 2, 1.
  - **The arms:** trained from scratch at each `g`, and uptrained from the `g = 8` model converted
    to each `g < 8` (D5; Ainslie et al.'s 5% of the original steps is 1.5 epochs, so 2 epochs).
  - **Data:** Herodotus, Euclid and the *Muqaddimah*, both splits, 5 seeds.
  - **Measures:** held-out bits per character; the cache's bytes per token,
    `2 * g * d_k * 8 * layers`; and decode characters per second at batch 1 and batch 16 at
    windows of 256 (D11), on `jebel` by daylight.
  - **Registration:** claims and a budget from a probe, registered in this plan before the data
    (experiment-design §2), at stage 8. The from-scratch arms alone are 120 runs, so the
    registration sizes them against `stone`'s or `jebel`'s time when it is written.
- **D13. The study's FFN. Settled: (a) (new 2026-11-02, from the literature, at the owner's
  word).** D7 first said "step 13's chosen FFN", but step 13's registration has claims (F1-F3) and
  no adoption rule. The FFN is a background factor here: the same in every arm, the claims
  comparing `g` with `g`. What matters is that the trade-off is measured where the literature
  measured it, so it can be compared:
  - Every published GQA or MQA result is on a gated FFN. Ainslie et al. 2023 (§3.1) ran on T5.1.1,
    whose FFN is GEGLU (T5's released-checkpoints notes: "GEGLU activation in feed-forward hidden
    layer, rather than ReLU"). Llama 2's GQA models use SwiGLU (Touvron et al. 2023, §2.2), and its
    attention ablation (A.2.1) compensates for the smaller attention in the FFN's width. PaLM pairs
    SwiGLU with multi-query attention (Chowdhery et al. 2022, §2). None used ReLU.
  - Narang et al. 2021 found that "most modifications do not meaningfully improve performance",
    but SwiGLU and GEGLU did, across pre-training, fine-tuning and supervised training (§3.1), and
    the two are indistinguishable (Table 1: final loss 1.789 and 1.792, ReLU 1.838).
  - Options:
    - (a) *Chosen.* SwiGLU (`GatedDense(170, "silu")`), whatever step 13 finds. Pros: production's
      decoder-only form with GQA (Llama 2, PaLM), so the measured quality-against-cache trade-off
      is comparable with theirs; built and tested (step 13). Cons: its quality edge on our corpora
      is what step 13 tests and may not show, which doesn't bear on comparing `g` with `g`.
    - (b) Step 13's winner. Pros: the best FFN on our data. Cons: ties the design to a study whose
      claims may come out undecided at 5 seeds; a GELU win would put GQA on an FFN no GQA result
      used.
    - (c) GEGLU. Pros: the GQA paper's own FFN. Cons: an encoder-decoder model's; Narang et al.
      find it indistinguishable from SwiGLU, so it buys nothing over production's choice.
    - (d) ReLU. Pros: our baseline. Cons: no GQA or MQA result uses it; and no existing runs serve
      as the `g = 8` arm anyway (D7's 8 heads, not the context-length study's 4).
  - Sources (indrajala-literature): `ainslie-2023-gqa-grouped-query-attention`,
    `touvron-2023-llama-2-open-foundation-chat`, `chowdhery-2022-palm-scaling-pathways`,
    `narang-2021-transformer-modifications-transfer`, `shazeer-2020-glu-variants-improve-transformer`,
    `shazeer-2019-one-write-head-fast-decoding`.
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

## The study's registration (stage 8, written before any of its data)

Under the research repo's `docs/protocols/experiment-design.md`. The code is
`indrajala_ml/studies/gqa_study.py` (its docstring is the protocol; the training is
`context_length_study.train_packed`, which gained a start document and a save path), and
`tests/studies/test_gqa_study.py` checks its configuration against this section.

**Cells.** The FFN study's SwiGLU model (D13): the context-length study's `256` model (the 2-layer
rotary model, `d = 64`, no dropout, the document mask on, windows of 256 on the packed corpora)
with `GatedDense(170)` FFNs, under the recipe (warmup over the first epoch, then cosine to a tenth
of the corpus's tuned rate, AdamW 0.1, clipping at 1.0) at the token budget (batches of 8 windows),
on Rust. Its attention has 8 query heads of `d_k = 8` (D7), sharing `g` key/value heads:

| arm | `g` | training | cache bytes a token | runs |
| --- | --- | --- | --- | --- |
| `g8` | 8 (multi-head) | from scratch, 30 epochs; its final network saved | 2,048 | 30 |
| `g4` | 4 | from scratch, 30 epochs | 1,024 | 30 |
| `g2` | 2 | from scratch, 30 epochs | 512 | 30 |
| `g1` | 1 (multi-query) | from scratch, 30 epochs | 256 | 30 |
| `up4` | 4 | the same seed's `g8` network converted (D5), then 2 epochs | 1,024 | 30 |
| `up2` | 2 | likewise | 512 | 30 |
| `up1` | 1 | likewise | 256 | 30 |

The cache is both layers' keys and values, `2 x 2 x g x 8` float64s a token. The three long-form
corpora (Herodotus, Euclid, the *Muqaddimah*; Tiny Shakespeare is outside this study), both
held-out splits, 5 seeds (a screen), the same seeds in every arm: 30 runs an arm, 210 in all.

- **Uptraining** (D5, D7): `to_grouped` mean-pools each group's key and value projections from
  the `g8` network of the same corpus, split and seed (Ainslie et al. 2023, §2.1), with a fresh
  optimizer state, then trains 2 epochs under the same recipe, its schedule over those 2 epochs
  (warmup over the first, cosine over the second, from the corpus's tuned peak rate). Two epochs
  are 7% of the 30, near Ainslie et al.'s 5% (1.5 epochs), at the epoch's grain.
- **Parameters are not matched.** At `g` below 8 the arms have fewer key and value parameters:
  each layer's `Wk`, `bk`, `Wv` and `bv` are `g x 8` rows, not 64, so `g1` has 14,560 fewer
  parameters than `g8` (two layers of 7,280). This is the production comparison: the same
  architecture with key/value heads dropped, as Ainslie et al.'s conversions do. Llama 2's ablation
  instead widened the FFN to compensate (its A.2.1); that is a different question, not asked here.
  A loss at small `g` may therefore include the parameters' share.

**Feasibility** (experiment-design §1, information-flow's register). Every arm is the FFN study's
SwiGLU model with its key/value heads shared, each query head still attending over every key with
its own query, so each can represent the character model `g8` does up to its smaller key/value
width. The exclusion bound and code counts are a finite machine's and don't apply to a character
model. The learning rates are the ReLU model's tuned ones (`rotary_study.RATES`), as the FFN study
used, not retuned per arm. An uptraining run starts from the trained `g8` network, which exists
for every (corpus, split, seed) once the scratch stage ends (the code refuses to start one early).

**Claims** are on the final-epoch held-out bits per character over the counted targets, paired by
(corpus, split, seed): 30 pairs a comparison, a sign test on the count of pairs lower, and Holm over
the four at 0.05. The literature measured these at billions of parameters and tokens; this tests
them at 0.1 million parameters on character corpora.

- **G1, multi-head below multi-query (`g8` < `g1`), one-sided.** The direction is the literature's:
  Shazeer 2019 reports multi-query attention's "only minor quality degradation"; Ainslie et al. 2023's Table 1
  has MQA-XXL at 46.6 against MHA-XXL's 47.2 (average dev score, uptrained); Llama 2's Table 18 has
  MHA ahead of MQA on 9 of 11 tasks at 30B. G1 holds when the Holm-adjusted one-sided p < 0.05, and
  is refuted otherwise (no loss shown at this scale).
- **G2, GQA at `g` 2 against multi-head (`g2` vs `g8`), two-sided.** No direction is registered:
  the literature reports GQA close to multi-head (Ainslie et al.: 47.1 against 47.2; Llama 2:
  "comparably to the MHA baseline on most evaluation tasks"). G2 differs when the Holm-adjusted
  two-sided p < 0.05, and is undecided otherwise. Undecided is not shown equal.
- **G3, GQA at `g` 2 below multi-query (`g2` < `g1`), one-sided.** The direction is the
  literature's: Ainslie et al. 47.1 against 46.6; Llama 2's GQA "is better than the MQA variant on
  average". G3 holds when the Holm-adjusted one-sided p < 0.05, and is refuted otherwise.
- **U1, uptrained against from scratch at `g` 2 (`up2` vs `g2`), two-sided.** No direction is
  registered: Ainslie et al. didn't compare an uptrained model with one trained from scratch (their
  Limitations), and the uptrained run has seen 32 epochs against 30. U1 differs when the
  Holm-adjusted two-sided p < 0.05, and is undecided otherwise.
- **Descriptive, not tested:** per cell, each arm's final bits (mean ± sd), its last three epochs'
  fall (scratch arms), its training bits, the held-out bits by position (16 buckets) and its
  seconds per epoch (each machine's, labelled); `g4`, `up4`, `up1` and the other pairs (`up1` vs
  `g1`, `up4` vs `g4`, the trend over `g`), post hoc where read as comparisons; and decode
  characters per second at batch 1 and 16 for each `g` (`generation_timing.py --heads 8
  --key-value-heads g --context 256 --batch B`, on `jebel` by daylight, after the training).

**What a result does.**
- If G2 doesn't show `g2` above `g8` (undecided, or `g2` lower), GQA at `h / g = 4` is recommended
  in the README as the decoding models' default, at a quarter of the cache; otherwise multi-head
  stays the default and GQA is documented with its measured cost.
- If U1 doesn't show `up2` above `g2`, `to_grouped` and a short uptraining are documented as the
  path from an existing multi-head model; otherwise training from scratch is.
- G1 and G3 say whether the literature's ordering (MHA, then GQA, then MQA) holds at this scale.

No setting changes after the data. A flaw found mid-study is recorded and fixed in a new registered
experiment.

**Budget, backend and hosts** (measured before launch: `gqa_study.py time`, one epoch of
Herodotus's contiguous split on Rust, one job at a time on `jebel`, 2026-11-02 about 15:35).
- **Seconds a training epoch:** `g8` 110, `g1` 106 (`jebel`, one job). Key/value heads barely move
  training at this size; their saving is in decoding.
- **Memory:** a peak of 0.47 GB a run.
- **The work:** 1,200 scratch epochs a corpus (4 arms x 2 splits x 5 seeds x 30) and 60 uptraining
  epochs. At `jebel`'s single-job speed, per corpus's training windows (Herodotus 5,233, the
  *Muqaddimah* 3,556, Euclid 2,896): about 36, 25 and 20 hours of one job's scratch work, and 2, 1
  and 1 of uptraining.
- **The run** (the owner, 2026-11-02: every machine but `litshe`, from 2026-11-03 05:00): one
  corpus a machine, the scratch stage (`sweep --epochs 30`), then on the same machine the
  uptraining stage (`uptrain --epochs 30`): Herodotus on `stone` (10 workers), the *Muqaddimah*
  on `jebel` (8 workers, PL1 65 W), Euclid on `klip` (6 workers, 5.3 GB; its thermal guard at
  85 C). Expected about 4 to 7 hours of wall clock; `klip` is the slowest and may run longer.
- **Resuming:** each run's result is written as it ends, so a stopped stage resumes by whole runs.

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
   *Done 2026-11-02* (#1190). What the code settled:
   - **D3's order, made exact:** each group's `dK` is folded, then divided by the scale once, as the
     crate divides at the end; numpy's tiled passes take `K` and `V` at `g` heads (`expand`, `fold`).
   - **The context-length study** rebuilt `Attention` field by field, dropping any new field; it now
     uses `dataclasses.replace` (D7's model is that study's).
   - **The tier 1 A/B** moved two numpy forward rows (+7.4% at T=256 causal, -5.1% at T=64 packed)
     though the numpy ops at `g = h` are the old ones. An A/A was quiet (1.3%); the old commit with
     `expand` and `fold` added unused moved the same rows by the same amounts. A loaded-code
     effect, recorded in measurement.md §7 (the bump moved them again with only the `.so` changed).
3. **The crate** (D4): the ops' `key_value_heads`, block `i / (h / g)` in `scores` and attend, the
   group's tiles cut once per block, the backward's fold, `K`'s rotation by `g` heads,
   `attention_step`'s cache at `g * d_k`. Its tests: by bits against numpy's expressions at
   `g` in {1, 2, h}, and new `bits.txt` families. Then a "Bump rust/" PR, with tier 1 on `jebel`
   (`attention_timing`, `g = h`, `--order ONNO`). *Done 2026-11-02* (crate #93, bump #1191): the
   crate's tests check each op by bits against its tiled composition at multi-query and groups of 2
   and 3; `bits.txt` gains attention's forward and downstream at 4 heads, sharing 2 and sharing 1;
   the golden run held. Tier 1: no slower Rust row (every consistent one faster, -2.9% to -6.8%,
   no speedup claimed).
4. **The Rust layers:** `key_value_heads` through `AttentionRustArrayLayer` and
   `AttentionRustCache`; parity with numpy (crate `exp`). *Done 2026-11-02* (#1192; tier 1 within
   1.2%).
5. **Pure Python:** `AttentionLayer` (its `_tiled`, backward fold, rotary and cache); parity.
   *Done 2026-11-02* (#1193): the layer-major path is the example-major loop by bits;
   `python_array_snapshot.projections` splits `Wk` and `Wv` at `kv_width`.
6. **The conversion and the decode tool** (D5, D11): `to_grouped`, tested on each implementation
   (a converted `g = h` model is the original, bit for bit; a converted model's `Wk` rows are its
   heads' means); `generation_timing.py`'s options. *Done 2026-11-02* (#1194):
   `persistence/grouping.py`; a converted document's optimizer state is fresh (`t` 0, each layer's
   made on its first step); a batch above 1 decodes cached only (the uncached path is one row's
   full pass).
7. **Fixtures and golden entries** (D6): `SequenceGrouped<class>` fixtures (`g = 2` of 4 heads);
   the 4-head golden entries at `g = 2` and `g = 1`, and the Rust reference. *Done 2026-11-02*:
   "numpy/rust/python grouped model" (`g = 2`) and "... multi-query model" (`g = 1`), after the FFN
   entries (143 locally, 47 in the Rust reference); no earlier entry moved (checked entry by entry).
   `SequenceGrouped<class>` fixtures for each implementation's sequence class.
8. **The study** (D7, D13): registered first (claims, cells, feasibility, budget from a probe), then
   the sweep, then the findings. *Registered 2026-11-02* (above).
9. **Docs**: README (Layer norm and attention gains GQA); the workplan retired.
