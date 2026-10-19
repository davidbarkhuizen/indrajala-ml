# Workplan: a path through the window for each factor and for the answers (window branches)

**Status: drafted 2026-10-19 at the owner's request ("draft the plan in detail ... i think we should
enhance our architecture regardless"), from the readout probe after the division ladder's L3. The
owner's conjecture: "we need one attention head per codebook/factor, and ... one for the answer".
D1-D7 settled under the owner's delegation, each with its options' pros and cons. W1 and B1-B3
registered here, before their data. The build goes ahead whatever W1 and the separability check
find (the owner); they decide what the branches' first claim is read against.** **Amended
2026-10-19, after the architecture's novelty pass (the literature review's "Extension (2026-10-19)")
and with W1 certifying in-run (5 of 6 by 06:49, its reading in progress): W2, a control matched to
B1 in size, registered before its data (D8).**

## Why

A finite-state transducer (a Mealy machine) is two functions of the state `q` and the input `a`:
the transition `delta(q, a)`, the next state, and the output `lambda(q, a)`, the answer. Our
window computes both from one representation:

- The window (W11: `D, BLOCKS, HEADS, FFN = 16, 3, 2, 32`) is three transformer blocks over the
  window's tokens, each block after a `StateInput` adding the whole state `P c`, writing one
  residual stream **16 wide** per token.
- Each factor's pooling (one learned query, one head) reads the tokens the output layer reads
  (`AttentionPooling(OUTPUT)`) and feeds its cell and codes: that is `delta`, per factor.
- The answers are one `Dense(classes)` over the same tokens: `lambda`.

So 16 numbers a token carry what every factor needs for its next state and what a linear map needs
for every answer. **The readout probe** (the probe script's findings, "the readout probe"; #862):
L3's seed 5 (`p` 13, factors 32 wide) had learned division's 169-state machine exactly (0 wrong
states in both verdicts), but 15-22 of its 1,521 (state, query) answers were wrong, confidently
(logit leads to 4.3), on the states' own codes. Training the output layer alone for 30 epochs fixed
none of the 15 that remained. The poolings read the output layer's input, so no earlier layer can
be trained without moving the state.

**The separability check (2026-10-19, before this plan; the probe script's `separability`, built
in stage 2):** on the same network, the output layer's input at the quotient's
position for every reachable code and window (225 codes x 9 = 2,025 pairs, 16 features each), an
unregularised softmax readout fitted to convergence on all of them (full batch, 60,000 Adam
steps) left **6 wrong**, flat from step 20,000; a multiclass perceptron, which reaches 0 errors in
finitely many passes on linearly separable data, never got below 6 in 20,000 passes. For each of the
6, the nearest feature vector with a different answer is closer than the nearest with the same
answer (0.32-0.50 against 0.33-0.88). So the window's 16 features don't linearly separate the
answers: no linear readout of them certifies, however trained (strong evidence, not a proof: a
perceptron's non-convergence isn't one). It also shows the probe's sampled training was part of
its 15 (fitted on every pair, 6 remain). Script and output in
data/window-recurrent/division-ladder/separability/l3-seed5/, manifest `separability.sha256`.

The general algebras' round 2 points the same way: S5 formed an exact core yet reached 385-748
codes, and division certifies at 121 states where S5 and move-to-front 5 fail at 120 in the same
cell. A shared, narrow window serving several consumers is one explanation common to both; this
plan tests it.

## The design

**Window branches:** the window's first blocks are a shared trunk; its last block is replaced by
one branch per consumer, each branch a block of its own (`StateInput`, a residual single-head
attention, a residual FFN, a LayerNorm) over the trunk's tokens. Factor `g`'s pooling reads branch
`g`'s tokens; the output layer reads the answer branch's. With `G` factors, `G + 1` branches.

- The state no longer reads the answers' features: `lambda`'s branch can be trained alone after the
  machine is right (`readout-tune` on the answer branch, sound by construction), and its loss no
  longer shapes what the poolings read except through the trunk.
- Each factor gets its own head and FFN before its pooling: the owner's "one attention head per
  codebook", and the answers theirs.
- Every path keeps W11's depth (3 blocks), so the comparison with today's window changes the
  separation, not the depth (D2).

## Decisions

- **D1. Where the paths split: (b) the last block, one branch per consumer.**
  - (a) Only the answers split off: an answer head (LayerNorm, `Dense(FFN, relu)`, `Dense(classes)`)
    after the trunk, the factors reading the trunk as now. Pros: smallest change; makes `lambda`
    nonlinear and trainable alone. Cons: the factors still share one stream; not the owner's
    conjecture (a head per codebook).
  - (b) *Chosen.* The last block per consumer. Pros: the owner's conjecture; each consumer gets
    its own attention and FFN; the trunk still shares the work every consumer needs (reading the
    columns and the state); depth unchanged. Cons: the crate's program needs branching (D5); about
    `G + 1` blocks' cost in the last layer instead of one.
  - (c) Branches from the first block (a window per consumer). Pros: full separation. Cons: about
    `G + 1` times the window's cost; nothing shared, so each consumer relearns reading the columns.
  - (d) A wider stream with a slice per consumer, heads assigned to slices. Pros: one program, no
    branching op. Cons: the residual adds mix slices anyway (no separation without masking);
    hard to reason about.
- **D2. The trunk and the branches' size: the trunk W11's first two blocks unchanged (2 heads), each
  branch one block of one head, `D` and `FFN` as W11's.** Pros: every path 3 blocks deep, as now;
  one head a branch is the owner's conjecture as stated; parameters grow by about one block a
  consumer beyond the third block's. Cons: a branch's single head may be too few; more parameters
  than today (an unmatched comparison in size; matched in depth and width).
  - Alternatives: two heads a branch (Cons: not the conjecture; more cost); the trunk at three
    blocks plus branches (Cons: depth changes too, two changes at once).
- **D3. The width `D`: 16 in the branches' claims (B1-B3), as their controls; 32 probed apart
  (W1).** Pros: one change between each claim and its control; W1 answers the "crowded stream"
  question with no build. Cons: if W1 alone works, the branches' first claim is read at a width
  already known to suffice; a branches-at-32 cell is then registered after B1 (stage 7), not here.
  (2026-10-19: W1 works in-run at 5 of 6, so that cell, if run, can read only "no harm" against
  it (D7). The question it was for, separation against size, is W2's (D8).)
- **D4. Gradients: the answer branch's loss flows into the trunk as today; no stop-gradient.** Pros:
  the trunk must learn what the answers need from the columns; one change from today. Cons: the
  trunk is still shaped by the answers; a stop-gradient variant stays a follow-up.
- **D5. The crate's branching: (a) a `branch` op over the existing fork stack.**
  - (a) *Chosen.* `fork` saves the trunk's activation (as a residual's does); each branch's ops
    follow; a new `branch` op ends one branch and makes the saved trunk activation current again,
    the branch's result kept for its reader; the last branch is the answers', ending at the output
    op. `reads` becomes one index per factor (each factor's branch end). Backward: each branch's
    downstream into the fork summed, as `add` sums the residual's. Pros: reuses the fork stack's
    bookkeeping and tests; a straight op list stays a straight list; today's programs unchanged
    (no `branch` op: bit-identical). Cons: a new op in forward, backward and the parity tests;
    `reads` changes type (one index, or one a factor).
  - (b) A sub-program per branch, the program a tree. Pros: general. Cons: a rewrite of the
    program's walk; far more change for the same networks.
  - (c) Branches as separate networks in Python. Cons: loses the window program's one-call pass
    (the window program workplan's speed), and the numpy reference splits from the crate's.
- **D6. The package's spec: `Branches(trunk_end, (branch specs ...))` in the layer specs, validated
  with the recurrence (one branch a factor plus one ending in the output layer), saved in format 2
  as the program's op list with `branch` ops; the per-layer path (numpy and Rust array networks) and
  the window program both run it, with parity tests.** Pros: the spec states the design; format 2
  keeps old networks loading unchanged. Cons: format 2's window gains an op kind (a reader that
  predates it refuses a branched network, as it should).
- **D7. Six seeds a cell, 600 epochs, the fork, stop at certified, as L1-L3 and the rounds.** Pros:
  the controls already exist at these settings; 0 of 6 against 4 of 6 is Fisher one-sided p 0.030.
  Cons: a gain from a nonzero control (`p` 11's 4 of 6) can't be shown at 6 seeds: those cells
  read as "no harm" (B3).
- **D8. Size against separation (2026-10-19): (b) a window without branches matched to B1 in
  parameters, W2.** W1 works in-run, so a wider stream suffices at 13, and the branches add
  parameters (Known limits). If B1 works, it can't tell separation from size alone. The counts
  (`wr.parameter_count`, L3's cell, crate afa7204 on `branches-stage5`): L3 22,142; B1 28,702;
  W1 47,726; branches at 32 69,038; without branches at width 20, 27,386; at 22, 30,296. Widths
  must be even (2 heads divide the stream).
  - (a) No control, the size limit stated. Pros: no runs. Cons: a B1 success reads "branches or
    size", what the literature already predicts for hard parameter sharing (Vandenhende et al.).
  - (b) *Chosen.* W2: width 22, no branches, 5.6% more parameters than B1. Pros: one flag
    (`--window-width 22`, built), no build; it errs against the branches (more parameters, one path),
    so B1 working where W2 fails is separation, not size. Cons: 6 more runs; width and size change
    together (the size can't be matched at width 16 without another change: depth or heads).
  - (c) The branches at 32 against W1 (D3's stage-7 cell, now). Cons: W1 at 5 of 6 leaves no room
    for a gain at 6 seeds (D7): only "no harm" can be read.
  - (d) A fourth block at width 16 (depth instead of width). Cons: changes depth, which D2 kept
    fixed.

## The claims (registered before the data)

Certified means both verdicts (the default snap ratio and 1.0), recertified on `pyramidon`. Each
cell **works** if at least one of seeds 0-5 certifies; **helps** if its control failed 0 of 6 and
it certifies 4 or more (Fisher one-sided p 0.030); 1-3 of 6 against 0 is "works, gain undecided".

- **W1, the width (no build beyond a flag):** L3's cell (division `p` 13, factors 32 wide, 38
  codes a factor) with the window's `D` 32 instead of 16 (`FFN` 64, twice, as W11's ratio; heads
  2), seeds 0-5. Its control is L3 (0 of 6), if the flag's default repeats L3's first epochs by
  bits on the flag's commit; else L3's cell is rerun beside it. **Works:** a crowded 16-wide stream
  limited division at 13. **0 of 6:** width alone isn't enough; the branches' claims read as
  planned.
  **Result (2026-10-19): works, 5 of 6** (seeds 0, 1, 2, 4, 5 at steps 8-11, each recertified in
  both verdicts; Fisher one-sided p 0.008 against L3), on jebel at 85a9e44: findings in the probe
  script's docstring (W1).
- **W2, the size control (D8; registered 2026-10-19, before its data):** L3's cell with the
  window 22 wide (`--window-width 22`: `FFN` 44, heads 2), no branches, seeds 0-5: 30,296
  parameters against B1's 28,702. Its control is L3, by W1's by-bits rule (the flag's default
  repeats L3 on 85a9e44). Read with B1, by D7's counts: **B1 helps over size** if B1 certifies 4
  or more and W2 0 of 6 (p 0.030): the separation, not the parameters, cures 13. **Both work:**
  size at B1's count suffices; the separation's own part undecided. **B1 0 of 6:** W2 says
  whether the parameters B1 adds would have sufficed without the branches. Otherwise descriptive.
- **B1, the branches on division:** L3's cell with window branches (D1-D3, `D` 16), seeds 0-5,
  against L3 (0 of 6; the same by-bits rule). **Helps or works:** separate paths for `delta` and
  `lambda` cure division's failure at 13 where width (W1) did or didn't. **0 of 6:** the shared
  window isn't what limits division at 13; read with W1 and the readout check (the branches'
  answer branch tuned alone on each run's best network, as the readout probe).
- **B2, the branches on the general algebras' `n` 5:** round 2's cells (word-problem-s5,
  move-to-front-5; two factors, `--codes-ratio 32/11`, reading the output, the fork) with window
  branches, seeds 0-5 each, against round 2 (0 of 6 each; the same by-bits rule). Per cell, works
  and helps as above. Read with D9's probes (`breaks`, `by-length`) on every failure's best network,
  for whether the extra-codes mechanism (S5) and failure A (move-to-front 5) change.
- **B3, no harm:** division `p` 11 (the p11 ladder's cell, 4 of 6) and the general algebras' S4
  and move-to-front 4 (round 1's cell, 4 of 6 each), with window branches, seeds 0-5. **Harm** on a
  cell is 0 of 6 against its 4 of 6 (p 0.030); otherwise reported descriptively.
- **Adoption (D7's reading, the owner's "regardless"):** the branches are built whatever the claims
  find. They become the default window for new studies if B1 or B2 works on a cell whose control
  failed and B3 shows no harm; else they stay an option, and the findings say what they changed.

## Feasibility (experiment-design.md, section 1)

The branches change the window, not the codes or the cap: every cell keeps its control's bounds.
`m` is the joint codes, the cap the exclusion bound `m + q - 1`.

| claim | cell | states `q` | codes a factor | `m` | `m / q` | cap |
| --- | --- | --- | --- | --- | --- | --- |
| W1, W2, B1 | division `p` 13, factors 32 wide | 169 | 38 | 1,444 | 8.5 | 1,612 |
| B2 | word-problem-s5, move-to-front-5 | 120 | 32 | 1,024 | 8.5 | 1,143 |
| B3 | division `p` 11 | 121 | 32 | 1,024 | 8.5 | 1,144 |
| B3 | word-problem-s4, move-to-front-4 | 24 | 15 | 225 | 9.4 | 248 |

## Known limits

- One branch design (D1 (b), D2): a failure is this design's, not every separation's.
- The branches add parameters (28,702 against L3's 22,142 on L3's cell). W2 (D8) is B1's size
  control. B2 and B3 have none: a B2 success reads "branches or size".
- Six seeds a cell: existence and clear gains only (D7).

## Stages

1. **This plan.** The separability check ran before it (above).
2. **The probe tools and W1 (tier 0):** `separability` in the probe script (the check above as a
   repo tool, with a test on a small cell); `transformer_window(tokens, classes, width=16)` and
   `run --window-width D` (`FFN` twice `D`), its default today's specs (tested; no golden run:
   the training path is unchanged); a few epochs of W1's cell on `stone` for its epoch time, and
   the by-bits check of the default against L3. W1 ran on `jebel` (05:28-07:20), from 85a9e44 (#864's
   merge). Its findings in the probe script's docstring, a PR of their own. *Done.*
3. **The crate (indrajala-math-rust):** the `branch` op (D5 (a)): forward, backward, `reads` one
   index a factor; tests against hand-built programs and the per-layer reference values; a program
   without `branch` bit-identical (the crate's tests and, after the bump, the golden run). Its own
   PR, then "Bump rust/" here (tier 1: an A/B of the window program's benchmark, `--order ONNO`).
4. **The package (tier 1):** D6's `Branches` spec and validation; the per-layer path in the numpy
   and Rust array networks; format 2; the Rust network's program built with `branch`; parity tests
   (numpy against Rust, branched windows, training and inference); the golden run bit-identical
   (no network in it branches). One PR, or two if the persistence is large.
5. **The probe's flag (tier 0):** `run --branches` (D1-D3), its feasibility line unchanged,
   `readout-tune` training the answer branch alone when the network has branches (tested: every
   factor branch, the trunk and the recurrence unchanged by bits); a few epochs a cell for the
   epoch time, the round's estimate.
6. **B1, B2, B3 and W2** (48 runs: 6 for B1, 12 for B2, 18 for B3, 6 for W2), on `jebel` and
   `stone`, whole runs a machine, the B claims in that order and W2 beside B1 (from `main`, no
   build: the flag is #864's); manifests, recertified in both verdicts on `pyramidon`.
7. **Findings** in the probe script's docstring; adoption by the rule above; a branches-at-32 cell
   registered if W1 works and B1 doesn't (D3); this plan retired, its leftovers (D4's stop-gradient,
   D2's two heads a branch, the trunk's depth) to next-steps.md.

**The design in the literature** (the review's "Extension (2026-10-19)"): the branches are hard
parameter sharing (a shared trunk, a branch a task: Vandenhende et al. 2020; Guo et al. 2020), and
the hypothesis is negative transfer between `delta`'s and `lambda`'s consumers. Separate paths for
the two in a recurrent Mealy network are Tiňo & Šajda's (1995: one second-order path each, an
extra layer on `lambda`), at 5 states, extracted by clustering. The factor's own pooling, cell and
codes are RIMs-like (Goyal et al.). What these don't have is the diagnosis that motivates the
branches here (an exact `delta`, a `lambda` its features can't separate, both proved) and a check
of the result. The findings say so.

Related: the general algebras' stages 5-7 (the fork's trigger, the guard's softenings) replay the
`n` 5 cells: they wait for B2, so they replay the window that is adopted.
