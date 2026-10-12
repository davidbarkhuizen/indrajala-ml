# Workplan: length generalisation for the column adders (milestone A)

**Status: decisions D1-D12 settled (2026-10-12): D1, D2, D4 and D6 by the owner, the rest under
the owner's standing delegation. Stages 1-2 done (the plan; the reading). The first detailed subplan of
`docs/arbitrary-length-addition-workplan.md`: its milestone A, pieces P1, P4 (bucketed) and P5,
with the existence study. Needs the rotary-positions workplan (roadmap step 8) merged through
stage 7 (done).**

## The question

Can a column adder trained on numbers of up to 7 digits add numbers it has never seen at any
length, with no further training, and how far? The addition study found the smallest networks that
add at one fixed `n` (`scripts/addition_study.py`'s findings: 3 layers at `d` 6 at `n = 6`); none
of them can even be run at another length, since each has a learned `Position` of `n + 1` rows.
This plan builds what a length-free adder needs and runs one study: **existence first**. A
generously sized network trained by a curriculum over lengths 1-7, checked at lengths up to 64.
If one works, a later stage whittles it down; if none does, the result is milestone B's baseline
(the looped block, P7).

## Why it might work, and why it might not

- For: every weight of a column transformer without `Position` is per feature, not per token, so
  nothing in the network is tied to `T`. Rotary positions (step 8) make attention's scores depend
  on token distance only, with no table to train. A causal network with no positions at all can
  still tell its columns apart through the mask, given a first token to count from (D12).
- Against: a carry can pass through any number of columns (the parent plan's "unbounded carry
  distances"), so a fixed-depth network must find the nearest deciding column below each column at
  any distance. The addition sweep suggests the learned carry is depth-bound: one layer carried over
  2-3 columns at best (`L*` at `n = 6`), up to `d` 32. Softmax attention spreads its weight as `T`
  grows. The literature finds plain rotary failing at or just past its trained length, and no
  adder computing every column in one pass at fixed depth that extends (below).

So a fixed-depth network can't add numbers of every length, but may reach a fixed multiple of its
training length. The study measures that reach.

## What the literature found (stage 2)

Seven papers (arXiv ids given). Every adder in them is decimal and adds two operands, except Cho's
three-operand appendix and Fan's binary.

| paper | network and output | positions | trained to | furthest reached |
| --- | --- | --- | --- | --- |
| Zhou et al. 2024 (2402.09371) | decoder, 6 layers, `d` 512, 25M; one digit at a time | FIRE, randomised, with index hints | 40 digits | 100 at above 98%, the best of 10 seeds (2.5×); to 30: 45; to 20: 25; to 10: none |
| McLeish et al. 2024 (2405.17399) | decoder, 16 layers, `d` 1024; one digit at a time | Abacus (one learned row per digit significance, offset at random in training) | 20 digits | 97.9% from 21 to 100 (16 layers, input injection); 120 looped (8 layers × 2, 6×). FIRE and none: 2.9% and 3.2% from 21 to 100 |
| Cho et al. 2024 (2405.20671) | decoder, 1 layer, 4 heads; one digit at a time | coupled: the digits of one significance share a position id | 30 digits | 200 at above 95%, the median of 8 runs (6.67×); trained to 10: 70. Three operands, trained to 40: 90-100 |
| Fan et al. 2024 (2409.15647) | decoder block of 3 layers, `d` 256, **looped** `n + 1` times; every output at once | none | 19 binary digits, by a curriculum | near-perfect at 30, where next-token networks of 20× its depth fail 10 past the longest trained |
| Kazemnejad et al. 2023 (2305.19466) | decoder, about 107M; one token at a time | none, T5 bias, ALiBi, rotary, sinusoidal | 8 digits (addition) | over 10 tasks, no positions ranks first (mean reciprocal rank 0.69), rotary next to last (0.33, sinusoidal 0.22); addition reaches 1.125× (Zhou's reading) |
| Ruoss et al. 2023 (2305.16843) | encoder, 5 layers, `d` 64; every output at once | none and five schemes, the five also randomised | 40 bits | binary addition near chance at 41-500 for every scheme (per-bit accuracy 49.8-64.5%, chance 50%; the best of 10 seeds) |
| Allen-Zhu & Li 2024 (2404.05405) | GPT-2, knowledge storage | | | 2 bits of knowledge a parameter at 1,000 exposures a fact; 1 bit at 100; int8 keeps it, int4 0.7 |

What it means here:

- **Every adder that extends is autoregressive or looped.** Zhou, McLeish and Cho write one digit
  per pass and read the carry from the digit just written (Cho's one-layer construction attends to
  it), so a long carry chain unrolls over passes. Fan's loops `n + 1` times. Ours writes every
  column at once at a fixed depth. The two networks in these papers with that output don't
  extend on addition: Ruoss's encoder stays near chance past 40, and Fan's fixed-depth baseline
  fits worse even within its lengths and falls behind the looped one. Fan notes that no fixed-depth program for addition is known. The likely result is a
  break at 8 or 10 in both arms; the study measures it, and milestone B (D11) is the remedy every
  paper points to.
- **The column encoding already solves what these schemes fix.** Index hints, Abacus and coupling
  are three ways to give digits of one significance one position; Zhou names finding the operands,
  not the carry, as the hard part. A column token holds its three digits, so this study isolates
  the carry over distance, the property the addition sweep found failing (M3, M5).
- **Plain rotary fails at or just past the trained length** in every paper that tries it
  (Kazemnejad, Zhou, Cho, McLeish, Ruoss). Randomised positions (Ruoss) help it, a little on
  addition; that and FIRE are P2's schemes, waiting on this study's failure (D2, the parent plan's
  H2).
- **No positions needs an anchor.** Kazemnejad's proof that a causal network with no positions
  recovers each token's position anchors on a start token: uniform attention over the prefix
  weighs it `1 / t`. A column sequence has no such token: its first column is a column like the
  others. D12 adds one. Cho proves one layer with no positions can't add at all (a permuted prefix
  gives the same output); 16 layers aren't bound by that.
- **Seeds vary widely.** Zhou's 10 seeds all fit 1-40 and differ from near perfect at 100 digits
  to 0% at 90 (McLeish's summary); the data order matters too. The report gives each seed's break length (as planned)
  and claims only what every seed shows.
- **Training lengths are short.** Reach grows with the longest length trained (Zhou: 10 gives
  nothing, 40 gives 2.5×; Cho: 10 gives 70, 30 gives 200). Seven is shorter than any of them; a
  failure at 7 doesn't say a longer curriculum fails. D1 stands; stage 10 can lengthen it.
- **Depth.** McLeish trains 16 layers with DeepNorm initialisation; Cho finds 6 layers generalise
  worse than 1-5 (median, not best seed); Zhou finds size matters little. The calibration (stage 8)
  checks the curriculum's first step trains at 16 layers (Known limits, depth and residual scale).
- **The curriculum** (D1) is Fan's: the longest length raised step by step. The others sample every
  length at once from the start, which is D3's control.
- **Memorisation.** Allen-Zhu & Li's 2 bits a parameter is for facts seen 1,000 times; the argument
  for `n = 3` uses it, and `n = 4` needs only the trivial bound (below).

## The study (D1-D8)

| choice | settled |
| --- | --- |
| network | column-causal (candidate 1's encoding: a token per column, `9 a + 3 b + c`, vocabulary 27; a start token before the first column, D12), **no `Position`**, `d` 16, **16 layers**, 2 heads (head size 8), FFN width 32: about 36,000 parameters |
| arms | `Attention(positions="rotary")` and `positions="none"` (the control) |
| training | **the curriculum**: lengths 1-4 together until every property passes at every one of them; then add 5, then 6, then 7, each step training on every length so far (cumulative), warm-started from the previous step |
| within a step | each length an equal share of every epoch's batches; a batch is one length; Adam at the sweep's rate 0.003, started fresh at each step; at most 200 epochs, patience 30 (D7 of the addition study) |
| seeds | 5 |
| checks | after every step (lengths 1-4, 1-5, 1-6, 1-7): the sampled catalogue at 8, 10, 12, 16, 24, 32, 48, 64 |
| verdict | per seed and step, the **break length**: the largest test length at which it and every shorter one pass. A model **works** if it passes at 16 or beyond (more than 2× its longest training length) |

Why `n = 4` is the curriculum's first real step: at `n = 1` and `2` every input fits in a lookup
table (27 and 729 triples); `n = 3` (19,683 triples, about 125 kbit of answers) is beyond what a
36,000-parameter network stores at about 2 bits a parameter (Allen-Zhu & Li 2024, for facts seen
1,000 times), and `n = 4` (531,441, about 4.2 Mbit) is beyond it even at 64 bits a parameter (the
weights' float64: 2.3 Mbit). Lengths
1-3 are stepping stones; held-out inputs at the training lengths (B3, from `n = 3`) and every test
length past 7 are what no network can have memorised.

## Stages

1. **This workplan.**
2. **Reading** (done: What the literature found). Ruoss et al. 2023 (randomised positions), Zhou et al. 2024 (FIRE, index hints),
   McLeish et al. 2024 (Abacus, looped blocks), Cho et al. 2024 (position coupling), Fan et al.
   2024 (looped transformers), Kazemnejad et al. 2023 ("The Impact of Positional Encoding on Length
   Generalization in Transformers"), and Allen-Zhu & Li 2024 (knowledge capacity). For each: the
   task, the encoding, the position scheme, the training lengths, the furthest length reached. A
   "What the literature found" section here, written before any code, and any decision it moves
   revisited in that PR (docs only).
3. **P1a, specs and shapes** (done). Token ids of any count are the input shape `(None,)`
   (`AnyInputShape`), which only an `Embedding` reads; the shape walk carries `None` tokens through
   and refuses a `Position` over them (it needs a fixed count). Format 2 records `{"dimension":
   null}`. A network still takes a fixed `InputShape`, and loading a length-free file is refused,
   naming the stage, until each implementation's token layers take `T` from the batch (stages
   4-6). Every existing network keeps its fixed shape, so no saved file or fixture changes.
4. **P1b, numpy.** The numpy network builds over `(None,)` and loads a length-free file.
   `AttentionArrayLayer` takes `T` from each batch, not from construction: the
   causal mask and the rotary table are built for each `T` it meets and cached; a longer rotary
   table keeps the shorter one's rows (`rotary_table` already does). Every other token layer
   (`Embedding`, `LayerNorm`, the per-token `Dense`, `Residual`, the sequence output) audited for
   a hidden `T`. Tests: one network run at several `T`, and at each `T` bit-identical to a network
   built for that `T`; gradient checks at two lengths. Golden run bit-identical (D9); tier 1 A/B of
   the attention case.
5. **P1c, the crate and the Rust layers.** The crate's attention ops already read `T` from their
   arrays; tests that each op agrees with numpy at several `T` with one set of weights. The Rust
   layer as numpy's, and the Rust network over `(None,)`. Tier 1 A/B of the attention case.
6. **P1d, pure Python.** The same, the network over `(None,)`, parity with numpy at several `T`.
   Tier 0.
7. **P5 and P4, the harness across lengths.**
   - The catalogue at any `n` (D8): exhaustive where it is small, sampled where it isn't.
   - The column encoding with its start token (D12).
   - A study config names its training lengths and its test lengths; a curriculum is a list of
     steps, each a set of training lengths, warm-started from the previous step's model.
   - Bucketed batches (D5): each epoch draws an equal share of examples per length, batches each
     of one length, the batch order shuffled by the run's seeded generator.
   - `generalise MODEL --n N...`: loads a saved model at each test length and runs the catalogue.
   - The report: per arm, step and seed, the break length, `L*` and exact-match accuracy at each
     test length, and the examples seen (D10).
   - Fix: a result's `crate` field hashes `indrajala_math_rust.__file__`, the package's
     `__init__.py`, so it never identified the build (every sweep result shows `639a07d4bc7d`); hash
     the extension module instead.
   - Tier 0 (studies code).
8. **Calibration.** On `pyramidon`: one seed of each arm through the curriculum's first step with
   a short budget, timing an epoch at each length and the `generalise` pass at 64, so the study's
   cost is measured, not guessed. A note here with the figures.
9. **The existence study** on `jebel`, under the machine lock: 2 arms × 5 seeds through the
   curriculum, checked after every step. Findings in the script's docstring.
10. **What follows from it (D11).** Decided on the findings, each as its own PR:
    - if a model works: whittle it down, depth then width, by the addition study's bisection, with
      the all-at-once control (D3) and the unordered encoding beside it;
    - if none works: the findings are milestone B's baseline, and the parent plan's P7 (the looped
      block) is next.
11. **Docs**: README, next-steps, the parent plan's status; this workplan retired.

## Decisions

Settled by the owner (2026-10-12):

- **D1. The regime. Settled: the curriculum, cumulative.**
  - (a) *Chosen.* Lengths 1-4 until every property passes at each, then add 5, 6 and 7 one at a
    time, each step training on every length so far. Pros: the network keeps what it learned (no
    forgetting); each step's stopping rule, "passes at every length so far", stays meaningful.
    Cons: a step's epoch grows with `k`.
  - (b) Each step on its new length only. Pros: the purest test of extending an algorithm. Cons: a
    network can pass 5 by unlearning 3, found only at the end.
  - (c) As (b), each step checked against the earlier lengths. Cons: a forgetting network stalls
    the step, and needs a rule for that.
- **D2. Position schemes. Settled: rotary and none.**
  - (a) *Chosen.* Rotary (step 8) and no positions, as a control. Pros: if rotary breaks, the
    control says whether rotary caused it (distances it never saw) or any network trained so does;
    no new code for the control. Cons: twice the runs.
  - (b) Rotary only. Cons: a failure can't be laid to the scheme.
  - Learned `Position` can't run at another length, and P2's new schemes (ALiBi, FIRE, scores
    scaled by `log T`) wait for this study's failures (the parent plan's H2).
- **D4. Size. Settled: existence first, at `d` 16 and 16 layers.**
  - (a) *Chosen.* One generous network; whittle it down only once one works (stage 10). Pros: the
    question "is it possible" is answered before any cost goes to "how small"; a failure can't be
    blamed on size. Cons: a large first network (about 4× a 4-layer one per epoch).
  - (b) A depth ladder (2, 4, 8, 16) from the start. Pros: the reach's slope against depth at once,
    milestone B's motivation if the carry is depth-bound. Cons: 7.5× the cost before knowing
    anything works.
  - (c) Narrower than `d` 16 (`d` 12 with 2 heads, `d` 8 with 1). Cons: a token must carry its
    digits, its carry state and its sense of position; too narrow and a failure confounds
    "impossible" with "too small". Width is whittled in stage 10 instead.
- **D6. What "works" means. Settled: (c).**
  - (a) Every property at every length up to 2× the longest trained (14). Cons: may be too high a
    bar for a first pass.
  - (b) Any length past 7. Cons: passing 8 may be a near-miss, not an algorithm.
  - (c) *Chosen.* Each model's break length reported; it works if it passes at 16 or beyond. Pros:
    one number per model; an algorithm is unlikely to stop at its boundary. The whole curve is
    reported regardless.

Settled under the owner's standing delegation:

- **D3. Controls. Settled: the position control now; the all-at-once control in stage 10.**
  - (a) *Chosen.* The existence study runs the curriculum only; the same network trained on 1-7
    together from the start, with the same number of examples, joins at stage 10. Pros: the
    existence run stays at 10 runs. Cons: if a model works, whether the curriculum's order mattered
    is unknown until then.
  - (b) Both controls now. Cons: twice the cost, for a question that matters only if something
    works.
- **D5. Lengths within a step. Settled: equal shares.**
  - (a) *Chosen.* Each length gets an equal share of each epoch's batches, so the newest gets
    `1 / k`. Pros: one rule, no knob. Cons: at the first step half the epoch is on `n = 1, 2`,
    trivial and slightly short-range; harmless for correctness, slower.
  - (b) The newest length weighted up. Cons: a second knob with no evidence it's needed; added if
    a step stalls.
- **D7. Warm starts. Settled: weights carried, Adam fresh.**
  - (a) *Chosen.* Each step loads the previous step's saved model and starts a new Adam. Pros: no
    optimizer state in the model file; the weights carry what was learned. Cons: a brief loss
    spike at each step, which the 30-epoch patience absorbs.
  - (b) Adam's moments carried. Cons: needs them saved; format 2 holds weights only.
- **D8. The catalogue at long lengths. Settled: sampled, with targeted strata.**
  - Random sums rarely hold long carry chains (their chance falls geometrically with the chain), so
    their accuracy overstates a model at long `n`. The verdict rests on the strata:
    - M3, the carry over distance `L`: every `L` from 1 to `n`, a fixed number of cases each
      (calibrated in stage 8; 1,000 per `L` as now if affordable);
    - M5, every carry pattern: sampled past `n = 6` (`7^n` patterns);
    - B1 sampled; A1-A3, M1, M2, M4 as now, per stratum;
    - B3, the held-out tenth, at the training lengths only (every test length is unseen whole).
  - `L*` at each test length shows how a model breaks: a carry failing past some distance, or
    everything at once.
- **D9. Bit-identity. Settled: P1 keeps the golden run.** At a fixed `T`, P1 builds the same mask
  and table and does the same arithmetic, so the golden run must stay bit-identical; a change that
  can't is reported in its PR and needs the owner (docs/measurement.md, §8).
- **D10. Compute is recorded.** Every run records the examples it saw and its epochs per step, so
  stage 10's all-at-once control can be given the curriculum's budget, and the comparison isn't
  more training against less.
- **D12. A start token. Settled (stage 2): (a).**
  - (a) *Chosen.* Every column sequence begins with a start token (vocabulary 28, `T` = `n + 2`) in
    both arms; its label is a fixed 0, ignored when decoding the sum. Pros: the control gets the
    anchor Kazemnejad's construction needs, so a failure in it is about length, not about a network
    that can't locate its columns; both arms see the same input, so the arms differ in one thing.
    Cons: a departure from candidate 1's encoding (the addition sweep's thresholds don't carry
    over exactly); one more token to every sequence.
  - (b) No start token. Pros: candidate 1 unchanged. Cons: the control may fail at the training
    lengths themselves, and then it says nothing about rotary's break.
  - (c) The start token in the control only. Cons: the arms then differ in two things.
- **D11. Milestone A's verdict. Settled: measure, then milestone B.** (a) *Chosen.* Whatever the
  existence study finds, milestone B (the looped block) follows; stage 10 whittles a working model
  first, or records the failure as B's baseline. Pros: a fixed scope; B is the mechanism the
  literature credits with real extrapolation on addition. Cons: if everything fails early, the
  answer is only "it fails". (b) One round of fixes from P2 or P3 first, if the best arm doesn't
  reach 2×. Cons: open-ended; P3 waits on step 11.

## Known limits

- **One encoding.** Column-unordered (vocabulary 10) joins at stage 10; the addition sweep found the
  encodings differ little once deep enough.
- **Five seeds may not locate a break length.** The sweep's thresholds were already non-monotone at
  the margin. The report gives each seed's break length, not a mean, and claims only what every
  seed shows.
- **Depth and residual scale.** The blocks are pre-norm (`LayerNorm` inside each `Residual`), so
  gradients have a clean path through 16 layers, but nothing scales a block's output by depth (as
  GPT-2's `1 / sqrt(2 * layers)`), so the residual stream grows with depth. If seeds stall early,
  that is the first suspect; a fix touches initialisation and the golden run, and is its own PR.
- **Likely negative.** If the literature holds, plain rotary and no positions extend little; the
  study's value is then the per-step curve and milestone B's baseline.
- **Cost before data.** Stages 3-7 are about six PRs before the first number. The structure is
  right (every network length-free, in every implementation), and generation (step 9) and segments
  (step 11) need the same.

## Timing and checks

- P1 changes timed paths (every attention layer's set-up and call): stages 4 and 5 are tier 1, one
  A/B of the attention case each. Stage 6 (pure Python) and stage 7 (studies) are tier 0.
- Every stage that touches the package passes the golden run, bit-identical (D9).
- The study runs on `jebel` under the machine lock; its results are written with provenance under
  `data/studies/`.
