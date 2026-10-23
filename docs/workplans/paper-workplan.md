# Workplan: one focused paper (learning finite-state transducers exactly, with proof)

**Status: written 2026-10-23, at the owner's direction, after the programme audit: "since I am an
unknown independent researcher, our goal should be to identify what of the work (findings/results)
so far would be of genuine value to the greater research community value as a single focussed
paper, then identify what further tests need to be done to flesh out the rigour or answer any
important remaining questions for that paper. we should then pause our current experimental
programme, and focus on doing the work to finalize the paper." D1-D9 settled under the owner's
delegation, each with its options' pros and cons; the owner may reopen any. Supersedes P1 and P2 as
separate submissions (both stay as source drafts until the paper replaces them).**

## Why one paper, and what bar it has to clear

The audit's assessment (2026-10-23, in chat; the findings in indrajala-research
`review-2026-10-23-programme-findings.md`): P1 alone would likely be rejected (its core idea is Zeng
et al. 1993's, its machines have 3-5 states, its headline findings are partly known), and P2 alone is
more interesting but rests on screens and a control that can't carry its "because". An unknown
author's paper has to be of evident value on its evidence alone: registered comparisons at 20 seeds,
baselines, every number re-derivable from the data, and a result nobody else has.

The result nobody else has: **finite-state transducers of 100-169 states learned by gradient descent
from input-output examples alone and proved correct at every length, and proved decompositions of
them, read from the network.** The extracted-machine literature stops at a few states and checks by
sampling or queries.

## The paper (D1)

**Working title:** *Learning finite-state transducers exactly by gradient descent, with proof.*

**The question:** can a network trained by gradient descent on input-output examples alone (never on
states or traces) come to compute a finite-state transducer exactly at every length, how large, and
how would one know?

**Its claims** (K-numbers, so as not to collide with the claims plans' C-numbers):

| # | claim | evidence now | status | test needed |
| --- | --- | --- | --- | --- |
| K1 | A snapped window-recurrent network is a Mealy machine, and a complete certificate (every window from every reachable code, outputs and states, near ties branched) proves it equal to the target at every length or gives a counterexample | P1's certificate; run on 724 networks; every certified network passes every long-test case (millions) | strong | T6 (target-free form), T7 |
| K2 | Sampled tests overstate: networks passing every sampled test, one passing 328,000 long-test cases, are provably wrong; more codes buy test-passing networks, not correct ones | E1; E3 (registered, Holm) | strong (the phenomenon is Weiss et al.'s; the complete check and C3 are ours) | none |
| K3 | The snap is what extrapolates | E2 (0 of 20 against 8 of 20), E2b' (0 of 20 against 20 of 20) | registered at two arithmetic cells (C2b's continuous arm run before its registration, F20) | T3 (at scale, on division) |
| K4 | Training on the target's approximations makes product machines learnable: division at 25 states 13-15 of 20 against 1 of 20 | P2's registered comparisons | strong at `p` 5 | T2 (rates at 49, 121, 169) |
| K5 | One approximation suffices (the sum, 10 of 20), where an equally distinguishing non-approximation doesn't (the product, 0 of 20) | A2, A3 registered | the "because" underdetermined (F3) | **T1** |
| K6 | Scale: certified 49-, 121- and 169-state transducers from answers alone | screens (5 of 8, 4 of 6) and W1 (5 of 6, registered) | screens | **T2** |
| K7 | Beyond arithmetic: groups and reset machines (S3, S4, move-to-front) certified with the same cell; random machines only to about 12 states | general algebras round 1 (6 seeds) | screens | **T5** |
| K8 | What the network learns: its factors read exactly as the target's approximations (congruence quotients), sometimes a cover; on S3 through a point's stabiliser, never the sign | 8 S3 and 15 move-to-front networks, read after the hypothesis formed; not registered | the most original finding; exploratory | **T4** |
| K9 | The proof needs no target in the loop: an answers-only stop under a state bound, then Chow's test set with queries, proves the learned machine equal to an unknown target of at most the bound's states | the slack reading and the Chow tool (2026-10-23, offline) | new, offline | **T6** |
| K10 | Training past a proof can destroy it; the loss begins as 1-3 flipped transitions the loss doesn't see, then Adam's amplification | P1's replays; P2's collapses | exploratory mechanism | none (a short section, stated as exploratory) |

**Out of the paper (D2):** the state-formation strand's crystallisation method (annealing, the
crucible, the continuous profile, the margin stop, the freeze, restarts); the certificate guard as a
remedy (one sentence: it held replays and harmed training); the enforced factorisation (one
sentence); window branches; the factorisation machine; the batch runner. They stay in the research
notes; the crystallisation strand can be its own later paper once it works beyond two backdrops.

## The tests (each registered in its own decision here before its data; 20 fresh seeds a comparison)

| test | question | design | runs | cost (estimate) |
| --- | --- | --- | --- | --- |
| **T1** | Does the sum help because it is an approximation, or because it is an extra output the network can learn by itself? (F3, K5) | division `p` 5, the quotient plus one extra output, three arms on the same 20 fresh seeds: the sum (an approximation; A2's arm repeated), a **suffix output** (the sum of the last two columns' digits mod `p`: learnable, state-carrying, not a function of the target's state, so no approximation), a **token output** (the column's own digits' sum mod `p`: learnable, carrying no state); the approximation claim's settings otherwise | 60 | 10-55 min a run; one night on two machines |
| **T2** | The rates at scale (K6), and the control there (K4) | the cover at `p` 7, 11 and 13 at the base window (32 wide), 20 fresh seeds each; the quotient alone at `p` 11, 20 seeds; every certified network long-tested | 80 | `p` 13 about 30-40 min a success, failures to their epoch budget (about 2 h); about a day on `jebel` and `stone` |
| **T3** | Baselines at scale (K3): does the snap matter, and does the transformer, on division? | the cover at `p` 5: (a) the state continuous (no codes), (b) the window without its transformer (the linear window, E6), 20 seeds each, against T1's sum arm and A1's cover | 40 | short; judged by long tests where no certificate applies (a continuous state) |
| **T4** | Are the learned factors the target's approximations (K8), on networks that didn't shape the hypothesis? | offline: `approximations` on every certified network of P2, round 1, T2 and T5 not read before; expectations registered before the reading (the share of factors with a nontrivial exact approximation, the covers, the sign on S3 and S4, which congruences on division) | 0 (offline) | hours |
| **T5** | Breadth rates (K7) | round 1's cell at the base window (32 wide), 20 fresh seeds each: word-problem-s3, word-problem-s4, move-to-front-4, random-q12-0, random-q16-0 | 100 | small machines; S4 slow (up to 600 epochs); one or two nights on every machine |
| **T6** | The target-free proof (K9) | offline: on every run of T1, T2, T3 and T5, the answers-only stop at the exact bound and 1.1 times it (`answers-stop --logged` or the logged answers), then Chow's test set on each stop's network; registered expectations: no false discovery at the exact bound, the queries each proof needs | 0 (offline) | hours |
| **T7** | Reproducibility | a script that regenerates every table and figure from the result files; a sample of runs re-run on another machine (`jebel`, `litshe`) bit for bit; every statistic recomputed; the datapack's manifest | 0-10 | a day |
| **T8** | The literature, current | a sweep for 2025-2026 work on neural automaton learning, extraction and verification, state tracking and length generalisation; the related work rewritten | 0 | a day |

**Optional, decided after T1 (D8):** T10, an approximation's output beside a machine the method
fails on (random-q16-0, or S4's harder seeds), existence first: if an approximation lets a random
machine form, K4 generalises beyond arithmetic, the paper's strongest possible addition.

Feasibility (experiment-design.md, information-flow.md) is written beside each cell at its
registration: codes at least the target's states (3-10 times), the exclusion bound as the cap, the
window 32 wide (row 3), the crucible row not applying (no crucible).

## Decisions

- **D1. The paper's centre: learned transducers at scale, with proof, and what they contain.**
  - (a) *Chosen.* K1-K9 as above: the certificate is the instrument, scale through approximations the
    headline, the decomposition reading the second finding. Pros: the one result nobody else has
    (proved 100-169-state machines), and the most original observation (learned factors are the
    target's approximations); every part already has evidence; it joins P1 and P2 into one story.
    Cons: arithmetic carries the scale; breadth rests on T5.
  - (b) A methods paper (P1's content, the certificate and the measurements). Cons: the audit's
    assessment: likely rejected (Zeng et al. 1993; 3-5 states).
  - (c) The crystallisation paper. Cons: two backdrops, about 3 in 5, 0 of 10 beyond; not ready.
- **D2. What is out:** the crystallisation strand, the guard as a remedy, window branches, the
  factorisation work, restarts. Pros: focus; one method throughout (division `p` 11's cell, at the
  base window). Cons: S3's round-1 rate is low (1 of 6) without the crucible; T5 measures it as it is.
- **D3. One method throughout: the division and round-1 cell** (two factors, codes scaled with
  `sqrt(q)`, reading the output, the fork, the commitment as built, stop at the first certified step),
  at the base window (32 wide). Pros: the cell behind every scale result; simple to state; the paper
  makes no claim about state formation's remedies. Cons: the certificate is in the loop (the stop);
  K9 answers that offline (T6).
- **D4. Where the tests run: the batch runner** (revised by the owner, 2026-10-23: "should runs not
  rather be on the batch runner - it will be faster"). Measure mode, fixed registered seeds, no
  cutoff (each run to its epoch budget or its first certificate).
  - (a) *Chosen:* the runner, extended (stage 3) with the paper's backdrops (division at `p` 5-13 with
    its arms, run by `--p P --seed S`; the machines by `--runs TASK:SEED`) and cells (the division and
    round-1 cell at the 32-wide window, its stop the first certified step, no `--state-bound`); T1's
    short batch first, as the runner's stage-4 live check (`jebel` by day, `stone`'s bundle, a window
    closing), before T2 and T5's long batches. Pros: every slot of every machine filled within its
    hours, no idle gaps between hand launches; a closed window's attempt re-queued and re-run
    identically elsewhere; one ledger a batch. Cons: a build on the runner; its live checks unproven
    until T1's batch.
  - (b) The probe scripts by hand (the first choice). Pros: proven. Cons: manual placement; machines
    idle between launches.
- **D5. Statistics:** every comparison at 20 fresh seeds an arm, Fisher's exact test (two-sided by
  default, one-sided only as registered with its direction, both reported where they differ: the
  audit's F9), Holm across the paper's registered family, Wilson intervals; screens labelled screens.
- **D6. The paper's form: one manuscript for TMLR** (double-blind, judged on claims supported by
  evidence and interest to the community, no page limit), a short version for LearnAut after.
  Pros: the venue that fits an evidence-first paper by an unknown author; the workshop for the
  automata audience. Cons: TMLR's review takes months. The owner's call on the venue (reopenable).
- **D7. The manuscript:** a new file, `indrajala-research/papers/learning-transducers-with-proof.tex`,
  built from P1 and P2; both kept until it supersedes them, then archived (their history kept).
- **D8. T10 decided after T1's reading:** run only if T1 leaves the approximation's role standing.
- **D9. The pause (the owner's direction):** the experimental programme stops planning and launching
  new work. **Running registered probes run to their registered ends** (the rule: the deadline's
  completion (D67) and the reheat against restarts' (D63) moved runs, all ending today); their
  outcomes and integrity (`same-prefix`) are recorded, their full readings deferred. **Paused:** the
  two-phase cool (D66), the batch runner's state-formation batch (stage 5; the runner itself serves the paper, D4), the audit's set-2 follow-ups that serve the
  state-formation strand (F5's gate, F10's sensitivity, F11's relaxation times, F17's large-tier
  profile, F18's approximation beside S3, which T10 partly takes up). **Kept:** the audit's protocol
  amendments (F9, F24) and the fresh-seed rate's verdict (F1: D64 stays 7 of 12, D67 reported on its
  own), done in stage 1.

- **D10. T4 registered: the decomposition reading** (2026-10-23, before any of its networks is read
  by `approximations`). The tool: `approximations NETWORK... --task T --arm A --vocabulary ...` (the
  run's own arm and vocabulary), which names division's congruences too since stage 2: its p + 1 lines
  `A + kB` (the sum `A + B`, the difference `A - B`, `A`, `B`), all of its nontrivial congruences
  (window_tasks.linear_labels, proved and tested). Read on the snapped certified network of each run.
  - **The networks** (fixed now; two factors each; one-factor networks left out, as a lone factor
    holds the whole state once certified):
    - *division, P2's and W1's* (58): `p` 5: `division-claim/` family 13, quotient 1;
      `approximation-claim/` family 15, sum 10, quotient 1 (`--vocabulary sum difference product`);
      `division-probe-2a/` seeds 4, 5, 7 (the cover; the networks of P2's "not factored by residue"
      reading, so reported apart); `p` 7: `division-probe-4/` seed 2, `fork-probe/` fork 0, 2, 3 and
      nofork 0, 2; `p` 11: `p11-ladder/` 1, 2, 4, 5; `p` 13: `window-branches/w1/` 0, 1, 2, 4, 5 (W1,
      K6's registered 5 of 6). None read by `approximations`; their logs' factor shares (`A`, `B`,
      `S`, `D`, a continuous proxy) were seen during training.
    - *the algebras, round 1's unread* (11): word-problem-s3 seed 2, word-problem-s4 seeds 1-4,
      move-to-front-3 seeds 0-5 (`general-algebras/round1/`). Round 1's four move-to-front-4
      networks were read in shaping K8 and are left out.
    - *T2's and T5's certified networks*, when they exist, by the same rules, reported as fresh.
  - **Expectations** (a factor's approximation nontrivial: 1 < blocks < the target's states):
    - **T4a, the algebras:** at least half the factors hold a nontrivial approximation (the 14
      move-to-front-4 networks read before: 25 of 28 factors). *Refuted* below a quarter.
    - **T4b, the word problems:** no factor's approximation is the sign (`parity`, A_n's cosets)
      alone, and every named congruence a factor determines on S3 and S4 short of the whole state is a
      stabiliser's (`places`): the chain through a stabiliser, not Krohn and Rhodes's through the sign. *Refuted*
      by any factor holding `parity` without a finer `places`.
    - **T4c, division (expected against K8):** P2's reading (each factor holding 0.37-0.71 of every
      direction, no factor a function of one residue) predicts that fewer than a quarter of division's
      factors determine a line, so the state is held jointly, with the cover's arm (trained on the sum
      and difference) no different. *K8 extends to division* if at least half determine a line (in
      the family and sum arms, the trained `A + B` or `A - B` most of all); in between, reported as
      mixed. Reported by arm and `p`, `division-probe-2a` apart.
    - **Reported, no expectation:** covers (the meet separating every state), cascades, each factor's
      autonomy.
  - **The paper's use:** T4a and T4b standing, K8 is stated for the algebras on networks that didn't
    shape it; T4c decides whether K8 claims division (if refuted, the paper says division's learned
    state is held jointly, P2's section confirmed exactly).
  - Options: (a) *chosen*, every unread two-factor network with expectations from the readings so
    far. Pros: networks not read before; the counter-prior (P2's) stated rather than ignored;
    division's reading exact now. Cons: the algebras' unread set is small (11) until T5. (b) Only
    T2's and T5's fresh networks. Pros: wholly fresh. Cons: waits on the long batches; leaves 58
    division networks unread. (c) Re-read every network, the 21 that shaped K8 included. Cons: the
    hypothesis's own data can't test it.
- **D11. T6 registered: the target-free proof** (2026-10-23, before any of its runs exist). The tool:
  `answers-proof LOG... --task T --arm A --slack 1 1.1 --limit 20000000 --out DIR`, on every run of
  T1, T2, T3's coded arms (a continuous state has no machine to read) and T5: at each bound (the
  target's states times 1.0 and 1.1, rounded down), the answers-only stop from the log (the logged
  answers all right at the step's lengths and 24 and 48, the minimised machine within the bound),
  then Chow's test set on that step's saved network, the target a black box. The paper's cell saves
  every step's network, so every stop has one.
  - **Reach** (the test set's words, the target's own machine): division `p` 5 904 at the bound,
    82,300 at 1.1; `p` 7 1,770 and 1.3e7; `p` 11 4,360 and 1.4e15; `p` 13 6,090 and 1.3e19; S3 50
    and 50; S4 363 and 11,253; move-to-front-4 1,331 and 41,261; random-q12-0 305 and 1,830;
    random-q16-0 405 and 2,430. So at 1.1 `p` 11 and 13 are out of reach: their stops are judged by
    the certificate alone, reported as such.
  - **An integrity check, not an expectation:** at the exact bound Chow's theorem makes its verdict
    the machine's equality with the target, so on every stop it agrees with the strict verdict
    (snap ratio 1.0); any disagreement is a bug, diagnosed before the reading goes on.
  - **Expectations:**
    - **T6a:** no false discovery and no false stop at the exact bound (the slack reading: none in
      90 runs). *Refuted* by any one.
    - **T6b:** at 1.1, no wrong machine where Chow reaches (the slack reading: none). *Refuted* by any
      stop Chow refutes; reported with the bound and the counterexample's length.
    - **T6c:** every certified run's answers-only stop lands at or before its certified step (kept or
      strict-only), so the proof needs no certificate in the loop; "missed" runs (certified before the
      logged answers all passed) reported, expected under a tenth.
    - **Reported:** the words each proof queried, per target.
  - Options: (a) *chosen*, the exact bound and 1.1, Chow where it reaches. Pros: K9 measured on
    every paper run, its cost stated; the slack bound shows a discoverer's margin. Cons: no proof at
    1.1 for `p` 11 and 13. (b) The exact bound only. Pros: proof everywhere. Cons: says nothing of a
    discoverer who knows the size only roughly. (c) 1.5 as well. Cons: out of reach for every target
    but S3 (the slack reading's table).
- **D12. The paper's cell, exactly: W1's** (2026-10-23, stage 3, before any run of it). D3's
  "division and round-1 cell" left the factors' width open: round 1 ran `--codes-ratio 32/11` at
  width 16, W1 (the only registered result at 169 states) two factors 32 wide of 38 codes, which is
  `--codes-ratio 32/11` at width 32 (38 = ceil(32/11 * 13); the `p` 11 ladder's 32 codes too). The
  runner's `paper` cell: `--factors 2 --codes-ratio 32/11 --factor-width 32 --reads output --fork
  --window-width 32 --budget-epochs 600`, stopping at the first certified step, no `--state-bound`
  (its run record W1's but the budget, checked); 600 epochs as W1 and round 1; T2's and T5's cell.
  T1 and T3 keep their own design's cell, the approximation claim's (`approximation`: two factors of
  16, reading the output, the default window, 400 epochs, no fork), since T1 repeats A2's arm and T3
  compares with T1 and A1. Backdrops (the runner's `BACKDROPS`): T1's `d5-sum`, `d5-suffix`, `d5-token`
  on one vocabulary (`sum suffix token`); T2's `d7-family`, `d11-family`, `d13-family`,
  `d11-quotient`; T5's `s3`, `s4`, `mtf4`, `rq12`, `rq16`.
  - (a) *Chosen:* width 32 for every backdrop of T2 and T5. Pros: one method throughout (D3), and it is the cell
    behind the scale result (K6), so T2 measures W1's own cell at 20 seeds. Cons: T5's algebras run
    at a width round 1 didn't (its six seeds a screen of another cell); more parameters per run.
  - (b) Width 16 throughout (round 1's). Pros: round 1's cell exactly. Cons: at 169 states it is not
    the cell that certified (W1's L3 baseline at width 32, the window's width the change); T2 would
    test a cell with no evidence at scale.
  - (c) Width by backdrop (16 for the algebras, 32 for division). Cons: two methods, against D3.

## Findings: T4, the decomposition reading (2026-10-23, read by D10's rules)

`data/window-recurrent/paper/t4/read.sh` at 533ec4d (D10 merged first), on the 69 networks D10
names, each with its run's arm and vocabulary; a JSON a network, `tally.py` → `tally.txt`, the
manifest `paper/t4.sha256` (85 files). A factor's approximation nontrivial: 1 < blocks < states.

| set | networks | factors nontrivial | networks holding a line (both factors) | covers | determines |
| --- | --- | --- | --- | --- | --- |
| `p` 5 division claim, family | 13 | 8/26 | 5 (3) | 3 | `A + B` 4, `A - B` 4 |
| `p` 5 division claim, quotient | 1 | 0/2 | 0 | 0 | - |
| `p` 5 approximation claim, family | 15 | 13/30 | 8 (5) | 5 | `A + B` 7, `A - B` 6 |
| `p` 5 approximation claim, sum | 10 | 7/20 | 7 (0) | 0 | `A + B` 7 |
| `p` 5 approximation claim, quotient | 1 | 1/2 | 1 (0) | 0 | `B` 1 |
| `p` 5 probe 2a (apart) | 3 | 3/6 | 2 (1) | 1 | `A + B` 2, `A - B` 1 |
| `p` 7 probe 4 and fork probe | 6 | 3/12 | 2 (1) | 1 | `A + B` 1, `A - B` 2 |
| `p` 11 ladder | 4 | 6/8 | 4 (2) | 2 | `A + B` 3, `A - B` 3 |
| `p` 13 W1 | 5 | 2/10 | 2 (0) | 0 | `A + B` 1, `A - B` 1 |
| word-problem-s3 | 1 | 2/2 | - | 1 | `places {1}`, `places {2}` |
| word-problem-s4 | 4 | 3/8 | - | 0 | `places {1}` 2, `places {2}` 1 |
| move-to-front-3 | 6 | 9/12 | - | 1 | `order {0, 1}` 4, `{0, 2}` 5, `{1, 2}` 1 |

- **T4a met:** the algebras' factors 14 of 22 nontrivial (0.64; the bar a half, the refutation a
  quarter). S4 alone 3 of 8.
- **T4b met:** no factor holds `parity`; every congruence named on S3 and S4 is a `places` one (a
  point's or a pair's stabiliser): S3 seed 2 a cover by the places of items 1 and 2, as 4a''''s seed
  7 was.
- **T4c mixed, by its registered rule:** division's factors 43 of 116 nontrivial (0.37; 40 of 110,
  0.36, without probe 2a), between the expectation (below a quarter) and K8's extension (at least a
  half). By set it ranges from 0.20 (W1) to 0.75 (`p` 11).
- **What the mixed count hides, read beside it (not registered):** every one of the 43 nontrivial
  division factors holds *exactly one line* (its approximation `p` blocks of `p`, naming one of
  the p + 1 lines), and the line is almost always an output the arm trained on: the sum arm's 7
  are all `A + B`, never `A - B` (7 of its 10 networks hold the sum in a factor of its own); the
  cover's arms hold `A + B` 18 times and `A - B` 17; of the 116 factors none holds another line
  but the quotient-alone network's one `B`. 12 of the 58 division networks are covers, their two
  factors the sum and the difference apart. So where a division factor decomposes, it holds the
  approximation it was given; about a third do, the rest hold the state jointly.
- **A check for P2 before the manuscript:** P2's "the learned state is not factored by residue"
  says each factor of "the cover's first networks" at a solved step holds 0.37-0.71 of every
  direction `xA + yB`. If those are probe 2a's networks (reported apart here), two of the three hold
  `A + B` or `A - B` exactly in a factor at their certified step; the section's networks and step
  need confirming, and its wording correcting if they are these.
- **For the paper (D10's use):** K8 stands for the algebras on networks that didn't shape it (T4a,
  T4b). On division it holds in a narrower form than K8 states: a third of the factors, each
  holding exactly a trained approximation; stated so, with the count as registered (mixed). The
  sum arm's factors (only ever the sum) are the sharpest observation, exploratory (not registered),
  and T1's arms (the suffix and token outputs) can test it: a factor holding a non-approximation
  output exactly would say the factor holds what it is asked, not what is an approximation.

## Stages

1. **This plan;** the audit's standing decisions applied (F1 recorded in the state-formation
   workplan, the protocol amended for F9 and F24, F3's reword in contribution.md); the Chow tool
   merged (K9's instrument).
2. **Offline first:** T4 and T6's tools and expectations registered; T8's sweep; T7's regeneration
   script begun. **Done 2026-10-23:** T4 and T6 registered (D10, D11), their tools here (division's
   lines named in `approximations`, the readings' `--vocabulary`, `answers-proof`); T8's sweep in
   indrajala-research's window-recurrent-literature.md (its 2026-10-23 extension: no 2025-2026
   work learns an automaton by gradient descent and proves it at every length), the papers in
   indrajala-literature; T7's `scripts/paper_tables.py` (P2's approximation table regenerated
   identically), its further tables added with the manuscript (stage 6). T4's reading of the 69
   networks named in D10 is offline and may run now; T6 waits on its runs.
3. **Builds:** T1's two control outputs (suffix and token) as answer kinds of the division encoding;
   T3's continuous and linear windows in the division probe if not already there; each tested, the
   golden run unchanged.
4. **Registrations and runs:** T1 and T3 (short, `p` 5) first; then T2 and T5 (the long ones) across
   every machine; long tests on every certified network. Once the running probes end (D9).
5. **Readings:** T1-T6 by their registered rules, into the probe's findings and here.
6. **The manuscript** (D7), every number from the data by T7's script.
7. **Critique:** the audit's checks rerun on the manuscript; figures re-derived; a cold read (the
   owner's to arrange; a fresh-context reviewer as a first pass, if the owner agrees).
8. **The datapack, the preprint and submission:** the owner's decisions (arXiv's endorsement for a
   first submission; Zenodo; TMLR).

## Machines

`jebel` by day (8 at once, off overnight), `stone` weekday nights (12), `klip` (all cores, <= 85 C),
`pyramidon` (3 small runs beside development). Memory per run measured at each registration (division
`p` 13 the largest).
