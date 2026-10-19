# Workplan: one code, one state (opening the three bottlenecks)

**Status: drafted 2026-10-19 at the owner's request ("permanently document this analysis ... write
up a workplan for this that we can start in earnest this evening or tomorrow"), from the
pairs-per-code reading of W1, B2, B1 (mid-run) and round 2 (below). D1-D7 settled under the owner's
delegation, each with its options' pros and cons. C1 and C2 registered here, before their data;
C3 is the commitment-control plan's. The goal is one general architecture and training method for
every machine: each task is a backdrop for one bottleneck, never a cell to fit
([bottlenecks.md](bottlenecks.md)).**

## Why: what every certified network has, and how the failures miss it

The machine-reading probe `breaks` (`indrajala_ml/studies/machine_reading.py`) walks a network's
snapped machine beside the target's from the start, breadth first, and collects the **pairs**
(network code, target state) reached. A code is **right** for its state when the network's outputs
from it agree with the target's on every word of up to `d` tokens. Two numbers summarise it:

- **pairs per code:** how many target states each code stands for on average. **1.0 means each
  code is one state**; much more means codes have become ambiguous.
- **right pairs:** how many pairs give the target's outputs.

Read on 2026-10-19, at each run's best step (the worst answer at its best) or its certified step,
about a second and 54-64 MB a network on `pyramidon` (data in
`data/window-recurrent/window-branches/{b2-early,peek}/`, B2's in `b2/`, manifest `b2.sha256`
a2b39a33...):

| run | codes | pairs per code | right pairs | first codes broken | reading |
| --- | --- | --- | --- | --- | --- |
| **certified:** W1 seed 2 (division 13), B2 S5 seed 4, B2 move-to-front-5 seed 5 | 253, 331, 219 | **1.0** | all | 0 | one code, one state |
| L3 seed 5 (division 13, window 16, best step) | 225 | **1.0** | 119 | 41% | bottleneck 3: an exact state, wrong answers (the readout probe) |
| W1 seed 3 (division 13, window 32) | 392 | 17 | 45 | 87% | bottleneck 2 |
| round 2 S5, seeds 0-5 (window 16, no branches) | 568-846 | 98-120 | 69-375 | 2-51% | bottleneck 2: an exact core in some, every code ambiguous on longer words |
| B2 S5 seeds 0, 1, 3, 5 (window 16, branches) | 493-774 | 36-120 | 55-277 | 3-45% | bottleneck 2 |
| B2 S5 seed 2, step 18 | 744 | **1.65** | 729 | 0.6% | nearly one code, one state; lost by step 26 |
| B2 S5 seed 4, step 12 (5 steps before its certificate) | 409 | 24 | 338 | 0% | became 1.0 by step 17 |
| B2 move-to-front-5 seeds 0, 1 | 4-6 | - | 0 | - | bottleneck 1: no state |
| B2 move-to-front-5 seed 3 | 34 | 22 | 0 | - | bottleneck 1, partly formed |
| B2 move-to-front-5 seed 5, step 20 (3 steps before its certificate) | 265 | 2.2 | 144 | 21% | became 1.0 by step 23 |
| round 2 move-to-front-5 seeds 0, 2 | 336-339 | 1.7-3.3 | 25-61 | 43-45% | states formed, transitions wrong |
| B1 seeds 0-5 at steps 10-14 (division 13, window 16, branches; **mid-run**) | 141-415 | 11-81 | 1-4 | 89-100% | bottleneck 2, where L3 had an exact state |

What it shows:

1. **One code, one state is what every certified network has, and the three bottlenecks are three
   ways of missing it:** no state (1), codes that stand for several states (2), and an exact state
   with wrong answers (3). Extra codes as such aren't the fault: S5 seed 4 certified with 211 codes
   beyond its 120 states, every one right.
2. **The branches at width 16 move division from bottleneck 3 to 2.** Without branches (L3) the
   state was exact and the answers wrong; with a single-head branch a factor (B1, mid-run) every
   seed's codes stand for 11-81 states. Separating the answers helped them; splitting the factors'
   path starved their transitions. W1's width cured division without separating anything.
3. **Closeness at one step doesn't decide success; what training does next does.** Move-to-front-5
   seed 5 certified from 21% of its first codes broken; S5 seed 2 was at 0.6% and lost it
   (retention, across every bottleneck).
4. **Pairs per code tracks the machine, not the screen**, at a second a network: the signal a
   training control needs. P2's restore-best kept the best screen, which was never a machine.

## The remedies

- **R1, the answers' own branch, the factors on the shared window** (bottleneck 3 without causing
  2): the window's first two blocks a trunk; then two branches, each one block: the **state
  branch** (W11's third block unchanged: two heads, read by every factor's pooling, as in L3) and
  the **answer branch** (one head, its own FFN and LayerNorm, ending in the output layer). Every
  path 3 blocks deep. A general window: nothing in it depends on the task.
- **R2, pairs per code logged** at every step's end (a durable measurement; D1).
- **R3, keep the best machine:** within a step, the network with the fewest pairs per code is kept
  and restored at a plateau's end, in place of the best screen (D5).
- **R4, counterexample training:** the walk's ambiguous pairs name the codes that stand for
  several states and the shortest words reaching them; those words, with their answers, become a
  share of each epoch's training (D6). Inputs and answers only: the proof chooses which inputs.
- **R5, the usage thermostat** for bottleneck 1: the commitment-control workplan's stage 2, its
  design there (its diagnostic, #871: the commitment's pull is failure A's cause). This plan
  schedules it; it isn't redesigned here.

## Decisions

- **D1. R2's measure: at every step's end, `d` the task's (the probe's default), logged as
  `machine: {codes, pairs, right}`; `trace` prints pairs per code.**
  - (a) *Chosen.* Every step's end. Pros: about a second at 120-169 states against a step's
    minutes; every run from now on readable this way. Cons: a step's inside is unseen.
  - (b) Every diagnosis too. Pros: R3 needs it within a step. Cons: 4-10 times the cost; R3 adds it
    only where it runs (D5).
  - (c) Offline only (as today). Cons: needs every step's network kept and read after; no
    training control can use it.
- **D2. R1's shape: (a) a shared state branch and an answer branch from W11's two-block trunk.**
  - (a) *Chosen.* Pros: the factors' path is exactly L3's (the third block, two heads, width
    16), so a change in the state is the answer branch's doing; the answers get their own head
    (the owner's "one for the answer"); depth 3 everywhere. Cons: a build (`Branches` gains a
    branch read by several factors: `reads` one index a factor, two factors the same index).
  - (b) An answer head after the whole window (LayerNorm, `Dense(FFN, relu)`, `Dense`; the window
    branches plan's D1 (a)). Pros: no branching. Cons: no attention of its own; the answers' depth
    4.
  - (c) An answer block on top of the window. Pros: no branching. Cons: the factors would read
    a stream the answers no longer shape only if gradients were stopped; depth changes.
- **D3. R1's width: 16 first (one change against L3 and against B1), then the adopted width.**
  Pros: C1 reads R1 against L3 (window 16, no branches) and B1 (16, a branch a factor) with one
  change each. Cons: if only width works (W1, W2), R1 at 16 may fail for that reason; then R1 at 32
  is probed (existence first), labelled exploratory.
- **D4. The base window for R3 and R4: the one the window branches plan adopts (its stage 7);
  until then, W11 16 wide.** Pros: R3/R4 are training methods, read on the best architecture;
  their control is that window's own registered result (B2 or round 2), repeated by bits. Cons:
  C2's cells wait for the adoption.
- **D5. R3's rule: at each diagnosis within a step, pairs per code read; the network with the
  fewest kept (ties: more right pairs); a plateau's end restores it, a row logging it.**
  - (a) *Chosen.* Pros: P2's restore-best mechanism, with a signal that tracks machines. Cons: a
    diagnosis's cost rises by a second; keeping isn't curing (P2): R3 is run with R4.
  - (b) Keep across steps (the run's best). Cons: a later step's longer words make it no longer
    comparable.
- **D6. R4's sampling: at each diagnosis, the shortest words to the pairs whose code is wrong for
  their state within `d` (every code standing for several states is wrong for at least one, and a
  code with wrong outputs is wrong; up to 256, the walk's breadth-first order), each followed by `d`
  uniform tokens and the final zero column so its answers tell its states apart, with its answers
  from the target; a quarter of each epoch's examples drawn from them.** (Worded so at the build,
  before any data: the words go to the wrong pairs, which include the ambiguous ones.)
  - (a) *Chosen.* Pros: aims training at the codes that stand for several states; three quarters
    of the data unchanged, so the curriculum still covers every length. Cons: a build (words to
    examples); the share is a guess, probed first.
  - (b) All of an epoch from the counterexamples. Cons: overfitting a few words.
  - (c) Weighting their loss instead. Cons: the windows are drawn from fixed lengths, so the same
    words are rarely drawn at all.
- **D7. Probes first (2 seeds a backdrop, a stated expectation each), claims at 6 seeds, 600
  epochs, the fork, stop at certified; Fisher one-sided against the control, as the window branches
  plan's D7.**

## Amendment (2026-10-19, afternoon, before C0's and the 32-wide probe's data): width 32 as the base

**Why:** the window's width is a feasibility bound, not a setting to hold fixed for a clean
comparison (experiment-design.md, section 1; the owner: "we have made the mistake of designing an
experiment that can't succeed because it falls outside of the feasible region - again"). At width 16
the window is too narrow for division at 13 (L3 0 of 6; W2 at 22, 3 of 6; W1 at 32, 5 of 6), and
splitting it into a single-head branch per consumer starved the factors' transitions (B1 1 of 6; B3's
S4 0 of 5 so far against round 1's 4 of 6, its failures at 2-23 pairs per code). So:

- **D4' (replacing D4): the base window is W11 at width 32 (`--window-width 32`), on every
  backdrop.** Pros: the width where division works; general (nothing per task); separation and the
  training remedies are then tested where the base has capacity. Cons: about twice the epoch time;
  the algebras at 32 are unmeasured, hence C0.
- **D3' (R1's width): R1 is probed at 32 too** (seeds 0 and 1 on the three backdrops, after its
  16-wide probe; exploratory, read against C0's same seeds), and C1 at 16 stays registered as
  written: R1 adds a block for the answers without taking any capacity from the factors' path.
- **C0, the base at width 32, registered:** round 1's cells (word-problem-s4, move-to-front-4) and
  round 2's (word-problem-s5, move-to-front-5), two factors, `--codes-ratio 32/11`, reading the
  output, the fork, 600 epochs, `--window-width 32`, seeds 0-5 (24 runs), against the same cells at
  width 16 by the by-bits rule (the default width repeats them): round 1 S4 4 of 6, move-to-front-4
  4 of 6; round 2 S5 0 of 6, move-to-front-5 0 of 6.
  - **S5 and move-to-front-5:** works at 1 of 6 or more, helps at 4 or more (p 0.030).
  - **S4 and move-to-front-4:** harm at 0 of 6 (p 0.030); otherwise descriptive.
  - **Read every failure's bottleneck** by its pairs per code (R2's step-end rows) and state's
    share: 1 (no state), 2 (codes standing for several states), or 3 (an exact state, wrong
    answers).
  - **No harm on S4 and move-to-front-4** confirms width 32 as the base (D4'). Harm on either
    means width isn't free, and the base is chosen per its evidence in this plan's findings.
- **Feasibility:** the cells' codes and caps are their rounds' (the table below); the window at 32
  is W1's, sufficient at division 13.
- **Where:** `jebel` once B3 ends (6 at once, beside R3/R4's probe), from main at this plan's merge.

## The claims (registered before the data)

Certified means both verdicts, recertified on `pyramidon`. A cell **works** if at least one of
seeds 0-5 certifies; **helps** if its control certified 0 of 6 and it certifies 4 or more (p 0.030).

- **C1, R1 at width 16** (after its probe): division `p` 13 (L3's cell), against L3 (0 of 6) and,
  descriptively, B1; S5 and move-to-front-5 (round 2's cell), against round 2 (0 of 6 each) and,
  descriptively, B2 (1 of 6 each). Each control by the by-bits rule (the flag's absence repeats the
  control's first epochs). **Helps on division:** separating only the answers cures bottleneck 3
  without causing 2. **0 of 6 on division:** with C1's pairs per code read: at 1.0 with wrong
  answers, the answer branch isn't enough; above 1.0, it disturbed the state too.
- **C2, R3 and R4 together on the base window (D4)**, after their probe: S5 and move-to-front-5,
  against the base window's registered result there. **Helps:** counterexamples on a kept machine
  cure bottleneck 2. Read with each failure's pairs per code over the run (R2).
- **C3, the thermostat** (bottleneck 1): registered in the commitment-control workplan's stage 2,
  its cells failure A's (S3, move-to-front 4 and 5).

## Feasibility (experiment-design.md, section 1)

The remedies change the window or the training data, not the codes or the cap.

| cell | states `q` | codes a factor | `m` | `m / q` | cap |
| --- | --- | --- | --- | --- | --- |
| division `p` 13, factors 32 wide | 169 | 38 | 1,444 | 8.5 | 1,612 |
| word-problem-s5, move-to-front-5 | 120 | 32 | 1,024 | 8.5 | 1,143 |

## Known limits

- B1's reading in "Why" is mid-run (steps 10-14 of 600 epochs); C1's comparison with B1 uses B1's
  registered result when it ends.
- One measure of a machine; `d` the task's distinguishing length plus one.
- Six seeds a cell: existence and clear gains only.

## Stages

1. **This plan**, and the analysis in [bottlenecks.md](bottlenecks.md) (docs).
2. **R2 (tier 0):** pairs per code in every step's end row, `trace` printing it, a test on a small
   cell.
3. **R1 (tier 0, or 1 if the crate changes):** the shared branch in `Branches` (several factors
   reading one branch), `run --answer-branch`, parity tests (numpy against Rust), the golden run
   unchanged. Then its probe: 2 seeds each on division 13, S5 and move-to-front-5, on the first free
   machine, its expectation written first.
4. **C1** on `jebel` and `stone` once the current batch ends (B3's cells).
5. **R3 and R4:** the build, tests, a probe (2 seeds on S5), then C2 on the base window.
6. **R5:** the commitment-control plan's stage 2 (its amendment, build, C3).
7. **Findings** in the probe script's docstring and [bottlenecks.md](bottlenecks.md); the
   architecture and training adopted; this plan retired, its leftovers to next-steps.md.

Related: [window-branches-workplan.md](window-branches-workplan.md) (stage 6 running: B1, W2, B3;
its adoption decides D4), [commitment-control-workplan.md](commitment-control-workplan.md) (R5),
[code-consolidation-workplan.md](code-consolidation-workplan.md) (its remedies for bottleneck 2
spent; R3 replaces its P2's signal).
