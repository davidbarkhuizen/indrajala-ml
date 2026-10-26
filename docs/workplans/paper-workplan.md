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
| **T10r** | Does T10b replicate on machines that didn't produce it? (D24) | P0 against P2 on fresh draws, 25 machines an arm, 2 seeds each, read alone (never pooled with T10) | 100 | about 3 h a run; about a day on the fleet after T11 |
| **T11** | K4 beyond arithmetic: do the algebras' approximations, given as outputs, make them learnable? (D20) | T5's backdrops `s3`, `s4`, `mtf4` under the `paper` cell, each with its coarsest pair of approximations that determine the state as two extra outputs, 20 fresh seeds each, against T5's rates | 60 | short but S4 (3-4 h a spent run); the probe sets it |

**Optional, decided after T1 (D8):** T10, an approximation's output beside a machine the method
fails on, existence first: if an approximation lets a random machine form, K4 generalises beyond
arithmetic, the paper's strongest possible addition. Redesigned in D19: random-q16-0 is primitive
(no approximation to give), so planted machines against primitive ones of the same shape.

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
- **D6. The paper's form: an open release, its authorship stated as it is** (revised by the owner,
  2026-10-23; first TMLR). The owner: "this is a proper collaboration - my contribution is conceptual
  and high level, your contribution is real and extensive", and the owner can't defend the paper
  line by line, which TMLR, the ML conferences and arXiv require of a human author (none admits an
  AI author; arXiv bans a year for unchecked AI content since 2026-05). Chosen: the manuscript and
  its datapack published openly, Zenodo with a DOI and the repositories public, with a full
  contribution statement (the owner's direction and decisions; Claude's design, code, runs, analysis
  and drafting); verifiability in place of an author's defence: every number regenerated by T7's
  script, every certificate re-checkable from the networks, the data public. Later, the owner's
  intent: approach Anthropic with the published work for review and, ideally, co-authorship (Claude
  and a human Anthropic delegates); a long shot, the owner's to pursue.
  - (a) *Chosen:* the open release. Pros: honest about who did what; fast; no gatekeeper; anyone
    can verify. Cons: no peer review; its reach rests on links and citations.
  - (b) A human co-author in automata learning, then TMLR, LearnAut or ICGI. Pros: peer review,
    honestly. Cons: someone to find, their checking's time. **Taken up beside (a)** (the owner,
    2026-10-23): the owner writes to about five researchers whose work the paper builds on or
    contrasts with (automaton extraction, proofs for learned networks, conformance testing,
    expressivity, Mealy machines in recurrent networks), asking them to review it and consider
    co-authorship; their names are kept out of this repository.
  - (c) An AI-authorship venue (Agents4Science-like). Cons: no 2026 edition confirmed; 1 in 5
    accepted in 2025; outside the automata audience.
- **D7. The manuscript:** a new file, `indrajala-research/papers/learning-transducers-with-proof.tex`,
  built from P1 and P2; both kept until it supersedes them, then archived (their history kept). It cites neither (D22).
- **D8. T10 decided after T1's reading:** run only if T1 leaves the approximation's role standing. **Decided 2026-10-24: runs** (T1a and T1b hold; "Findings: T1"). Its design: D19.
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
  - **Amended 2026-10-25 (the owner), before any of T10's batch runs ended:** the scope extended to
    T10's 90 batch runs (`t10/batch/`, `paper_reading.py proof`), read by the same rules and reported
    apart from T6's 280 (T6 itself already read). At q 16 the bounds are 16 and 17, both within
    reach (`random-q16-0` 405 and 2,430 words before repeats; every stop gets Chow's verdict). T6a
    and T6c's expectations stand for them as registered; T6b, refuted as registered on T6's runs, is
    read on T10's as fresh, its count of stops Chow refutes reported. The tool was checked first on
    T10's probe (already read by its own rule, D19): its one certified run, `t10-probe-p2-401`, kept
    at both bounds and proved (405 and 2,030 words); the other 7 silent. Reported with the probe,
    not counted.
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
- **D13. T1 registered: is it the approximation, or any extra output?** (2026-10-23, before any of
  its runs.) Division at `p` 5, the quotient plus one extra output, three arms on one vocabulary
  (`sum suffix token`) and the same 20 fresh seeds, **300-319**: `d5-sum` (an approximation; A2's
  arm on T1's vocabulary), `d5-suffix` (the last two columns' digit sum mod `p`: learnable,
  state-carrying, no function of division's state), `d5-token` (the column's own digit sum mod `p`:
  learnable, no state). The runner's `approximation` cell (D12: the approximation claim's settings,
  two factors of 16, reading the output, the default window, 400 epochs, stop at certified, no
  fork), measure mode, no cutoff.
  - **Feasibility** (experiment-design.md section 1, information-flow.md): the certified answer's
    target `q` 25 in every arm, and the machine all the outputs need: 25 states (sum, a function of
    the state; token, of the column alone) or 125 (suffix: the state and the last column's digit
    sum, 5 values). Joint codes 16 x 16 = 256: 10.2, 10.2 and 2.0 times those (suffix below the
    3-5 times surplus; stated, kept: it is the control's own cost). The cap 280 = m + 25 - 1, the
    exclusion bound for the quotient (the network's machine projected on it has at most m states).
    The window 16 wide: this cell certified at `p` 5 (A1, A2: rows 3, 5). Every state reachable and
    told apart at one column (row 14). Probe (2026-10-23, `pyramidon`, seed 300, 6 epochs): 68-72 MB a
    run; the approximation claim's history for the budget: a success 10-30 min, a failure (400 epochs)
    1-2 h on `pyramidon`; about 100 run-hours in all if the controls mostly fail.
  - **Success:** certified at the stop and recertified in both verdicts (`readout`'s rule), every
    certified network long-tested (U1, S1 at 128, 256, 1,024 columns).
  - **Tests (one-sided, registered with their direction: K5 claims the approximation helps beyond
    any learnable extra output, so the sum is expected to certify more often):** T1a `d5-sum` against
    `d5-suffix`, T1b `d5-sum` against `d5-token`, Fisher's exact, Holm over the two at its reading
    (and over the paper's family at the end, D5), both sides reported where they differ.
  - **Expectations and refutations:** the sum about 10 of 20 (A2). **K5's "because" stands** if both
    T1a and T1b hold (Holm 0.05). **It is refuted** if either control certifies at least as often as
    the sum (then an extra learnable output helps as much: the sum's advantage over the product was
    not its being an approximation). Otherwise (lower but not significant) undecided, reported so.
  - **Exploratory, not registered:** `approximations` on every certified network (do the controls'
    outputs sit in a factor of their own, as the sum did in T4?); the T1 networks' epochs to certify.
  - **Then D8:** T10 is decided on T1's reading.
  - **Where:** the runner, its stage-4 live check: the controller on `pyramidon`, workers on `stone`
    (12 slots, its windows) and `klip` (8 cores, 5 GB: 6 slots at 70 MB, at <= 85 C), `jebel` by day
    when free; findings into this workplan, data `data/window-recurrent/paper/t1/` with a manifest.
  - Options: (a) *chosen*, three arms on one new vocabulary and fresh seeds. Pros: one change between
    arms (the output asked); the sum rerun beside the controls, not taken from A2 (another
    vocabulary). Cons: the sum's rate measured again (20 runs). (b) The controls against A2's sum.
    Cons: a different vocabulary, and data already read. (c) Only the suffix control. Cons: the token
    asks whether a stateless output helps, a separate question at 20 more runs.
- **D14. T3 registered: does the snap matter, and the transformer, on division?** (2026-10-23, before
  any of its runs.) The cover at `p` 5 on the approximation claim's vocabulary (`sum difference
  product`, A1's arm: `d5-family-wide`), the same 20 fresh seeds 300-319, three arms that differ in
  one thing each from the first: `approximation` (the transformer window and the snapped state: the
  control, run fresh), `approximation-linear` (E6's window, no blocks), `approximation-continuous`
  (E2's continuous state, no codes). 400 epochs, measure mode.
  - **Feasibility:** the control and linear arms as D13's (25 states, 256 codes, cap 280); the linear
    window keeps the GRU cell's nonlinearity and none in the window (E6 certified the carry automaton
    so, 2 of 6); the continuous state has no codes, so rows 1, 2 and 14 don't apply: it can't be
    certified, only long-tested. Probe: 63-70 MB a run; the linear window's epochs about 5 times
    faster; the continuous arm trains all 400 epochs (no certificate stops it), about 2 h a run.
  - **Success:** the control and linear arms as D13 (certified, both verdicts, long-tested); the
    continuous arm's network at its budget's end passing every long-test case (U1, S1 at 128, 256,
    1,024 columns). The asymmetry favours the continuous arm (a long test passes networks a
    certificate refutes: K2), so its refutation of K3 would be conservative.
  - **Tests (one-sided, direction registered: K3 claims each of the snap and the transformer is
    needed, so the control is expected to succeed more often):** T3a control against continuous, T3b
    control against linear, Fisher's exact, Holm over the two (and the paper's family, D5).
  - **Expectations and refutations:** the control about 15 of 20 (A1); continuous about 0 (E2: 0 of
    20 at 1,024 columns; C2, C2b). **K3 (the snap) stands** if T3a holds; refuted if the continuous
    arm succeeds as often. **The transformer's role** stands if T3b holds; if the linear arm certifies
    as often, the paper says the window needs no blocks at `p` 5 (stated as found). Undecided
    otherwise.
  - **Where:** as D13, after T1 in the runner's queue; data `data/window-recurrent/paper/t3/`.
  - Options: (a) *chosen*, the control run fresh beside the two baselines. Pros: same code, seeds and
    machines, one change an arm. Cons: 20 runs more than (b). (b) Against A1's existing runs. Cons:
    data already read, other seeds, another commit. (c) Against T1's sum arm (the plan's first
    wording). Cons: another vocabulary and outputs, two changes at once.
- **D15. T2 registered: the rates at scale, and the control there** (2026-10-23, before any of its
  runs). The runner's `paper` cell (D12), measure mode, no cutoff, fresh seeds **300-319** (unused
  on all four hosts) on four backdrops: `d7-family`, `d11-family`, `d13-family` (the cover, trained
  on the quotient, sum and difference) and `d11-quotient` (the quotient alone): 80 runs.
  - **Feasibility** (experiment-design.md section 1, information-flow.md; tested against the config,
    `test_t2_and_t5s_cells_meet_their_registered_bounds`): two factors 32 wide, `ceil(32 p / 11)`
    codes each, the cap the exclusion bound `m + q - 1`; the window 32 wide (row 3: the width W1
    certified at 169 states); every state reachable and told apart.

    | backdrop | q | codes a factor | m (x q) | cap |
    | --- | --- | --- | --- | --- |
    | d7-family | 49 | 21 | 441 (9.0) | 489 |
    | d11-family, d11-quotient | 121 | 32 | 1,024 (8.5) | 1,144 |
    | d13-family | 169 | 38 | 1,444 (8.5) | 1,612 |

  - **Cost** (the budget probe, `jebel`, 2026-10-23, seed 390, outside every range, 15 minutes each,
    8 at once, crate `crate-647f23c`; cost alone was to be read, but the runs' last rows, accuracy
    included, were printed while their epochs were extracted: seen, set aside, no expectation below
    uses them): peak RSS 82-93 MB a run, so memory never binds and cores set the slots; 8.5-9.1
    seconds an epoch on average over the first 100-174 epochs, 9.3-10.1 at the last step (the quotient
    alone 5.2 and 6.3). W1's `p` 13 spent run took 600 epochs in 113 minutes, its successes 209-255
    epochs (29-37 minutes). So a spent run is about 2 hours, and T2 at most about 150 run-hours.
  - **Success:** as D13: certified at the stop and recertified in both verdicts, every certified
    network long-tested (U1, S1 at 128, 256, 1,024 columns: 140-174 minutes a network at `p` 11).
  - **T2a, K6's rates (registered bar: a quarter).** K6 claims the cell learns these machines
    reliably, not by a lucky seed. At each `p` it **stands** if the Wilson 95% lower bound is at
    least 0.25 (9 of 20 or more), is **refuted** at that `p` if the upper bound is below 0.25 (1 of
    20 or none), and is **undecided** between, reported with its rate. Expected about 10-15 of 20 at
    each `p`: W1 5 of 6 at `p` 13 in this exact cell, and screens of another cell (width 16, the
    default window): the `p` 11 ladder's 4 of 6 and the fork probe's 3 of 4 at `p` 7. No test across
    `p`: the trend is reported with the slowest success's epochs against the 600 budget.
  - **T2b, K4 at scale:** `d11-family` against `d11-quotient`, Fisher's exact, one-sided (registered
    direction: K4 claims the approximations' outputs make the product machine learnable, so the
    cover certifies more often), in the paper's Holm family (D5), both sides reported where they
    differ. Expected about 0 of 20 for the quotient alone (`p` 5: 1 of 20 registered, 0 of 6 in a
    screen; no run at `p` 11 before). **K4 at 121 states stands** if T2b holds at Holm 0.05;
    **refuted** if the quotient alone certifies at least as often; undecided otherwise.
  - **Then:** T4 on every certified two-factor network (D10's rules, reported as fresh); T6 on every
    run (D11).
  - Options: (a) *chosen*, the cover at three `p` with the quarter bar, the control at 11. Pros: W1's
    own cell at 20 seeds; the bar can refute K6 at a `p`; the control at the scale P2 never ran.
    Cons: the bar is a choice (a quarter: at that rate, 20 seeds find about 5 networks); K4 is
    tested at one large `p` only. (b) The control at every `p` (`p` 7 and 13 too). Pros: K4 at each
    scale. Cons: 40 more runs, mostly spent (about 80 run-hours); `p` 5 and 11 bracket it. (c) Rates
    reported with no bar. Pros: no threshold to choose. Cons: nothing could refute K6, against the
    protocol (claims and refutations registered first).
- **D16. T5 registered: breadth rates** (2026-10-23, before any of its runs). The `paper` cell
  (D12, width 32, where round 1 ran 16: its rates are a screen of another cell), measure mode, no
  cutoff, the same fresh seeds 300-319, on `s3`, `s4`, `mtf4`, `rq12`, `rq16`: 100 runs.
  - **Feasibility** (as D15, tested by the same test): `ceil(32 sqrt(q) / 11)` codes a factor.

    | backdrop | q | codes a factor | m (x q) | cap |
    | --- | --- | --- | --- | --- |
    | s3 | 6 | 8 | 64 (10.7) | 69 |
    | s4, mtf4 | 24 | 15 | 225 (9.4) | 248 |
    | rq12 | 12 | 11 | 121 (10.1) | 132 |
    | rq16 | 16 | 12 | 144 (9.0) | 159 |

  - **Cost** (the probe, as D15's): the training process 69 MB at `s3` (the probe's `time -v` caught
    only the parent of `--runs`' worker, 51 MB; noted for the runner's own measure), so memory never
    binds; 6.1-6.6 seconds an epoch on average, 7.4-8.8 at the last step (`s3` about 3 alone). A
    spent run about 1.5 hours; T5 at most about 150 run-hours. Round 1's S4 success took 576 of 600
    epochs: the budget crowds there, and the slowest success is reported against it.
  - **Success:** as D15, the long tests the machines' uniform words at 128, 256 and 1,024 (round 2's).
  - **T5a, K7 for the algebras:** per backdrop (`s3`, `s4`, `mtf4`), D15's bar: **stands** at a
    Wilson lower bound of 0.25 or more (9 of 20), **refuted** at 1 of 20 or none, undecided
    between. Expected: `s4` and `mtf4` about 13 of 20 each (round 1: 4 of 6), `s3` about 3 of 20
    (round 1: 1 of 6, a state that never forms), so `s3` is expected undecided or refuted, and K7 is
    then stated without S3's rate claim (D2's con, measured as it is).
  - **T5b, the random machines' limit** ("random machines only to about 12 states"): **stands** if
    `rq12` meets the bar and `rq16` certifies 1 of 20 or none; **the limit moves** if `rq16` meets
    the bar too (reported as found, the stronger claim); otherwise reported with both rates.
    Expected: `rq12` about half (2 of 3 before), `rq16` none (0 of 9 at round 1's cell).
  - **No significance test:** T5 measures rates against a bar, so it adds nothing to the Holm family.
  - **Then:** T4 and T6 as D15.
  - Options: (a) *chosen*, the five backdrops under the paper cell with D15's bar. Pros: one method
    and one bar throughout; rates on networks that shaped nothing. Cons: S3 is expected to fail the
    bar; width 32 untried on the algebras at scale. (b) Round 1's width-16 cell. Cons: two methods,
    against D12. (c) A larger budget for the algebras (S4's crowding). Pros: fewer censored runs.
    Cons: a second cell, against D3 and D12; the crowding is reported instead.
- **D17. Where T2 and T5 run: one batch, longest first** (2026-10-23). One ledger
  `data/window-recurrent/paper/t2-t5/` under the `paper` cell, attempts in the order `d13-family`,
  `d11-family`, `d11-quotient`, `d7-family`, `s4`, `mtf4`, `rq16`, `rq12`, `s3` (the longest first,
  so the batch's tail is short runs), on its own port beside T1's. It starts on `jebel` now (7 slots,
  24/7 since 2026-10-23), the hosts T1 and T3 use joining after T3 (a restart; the ledger resumes),
  so no host runs two fleets.
  - (a) *Chosen.* Pros: one controller fills every slot across both tests; `jebel` works now rather
    than idling until T3 ends. Cons: the two tests share a ledger, split by backdrop at the reading.
  - (b) Two batches, T2 then T5. Pros: a ledger a test. Cons: a gap between them; T5's short runs
    can't fill T2's tail.
- **D18. Where T3 runs: one batch of three cells, the longest first** (2026-10-23, before any of its
  runs). One ledger `data/window-recurrent/paper/t3/`, measure mode, the runner taught several cells a
  batch (each attempt names its cell; ids `<cell>.d5-family-wide-<seed>`), the arms in the order
  `approximation-continuous` (all 400 epochs, about 2 h a run), `approximation`,
  `approximation-linear` (about 5 times faster), on T1's hosts (`pyramidon` 3, `klip` 3, `stone` 11,
  open all weekend: the owner) after T1. D14's registration is unchanged: the same three arms, seeds
  and budget; only where they run.
  - (a) *Chosen* (the owner, 2026-10-23: speeds things up in the long run). Pros: 60 attempts packed
    on 17 slots, about 5.5 h; later tests with several arms reuse it. Cons: a runner build and PR
    before T3 starts; the ledger's attempts change shape (an older ledger still reads).
  - (b) Three batches, one after another. Pros: no build. Cons: 20 attempts on 17 slots leave a
    second wave of 3 each time; about 9-10 h.
  - (c) Three batches at once, the hosts split. Pros: no build. Cons: unbalanced (11 against 3 and 3);
    one daemon a host, so no host serves two.

- **D19. T10's design: does an approximation let a random machine form?** (drafted 2026-10-24,
  before any of its runs; **registered 2026-10-24, after T5's batch ended**: its size and budget
  filled by their rules below, nothing else changed.) D8 runs T10, but
  its first named backdrop can't take it: **`random-q16-0` and `random-q12-0` are primitive** (no
  nontrivial congruence, Rystsov and Szykuła 2023's term; the `congruences` command, #980), as is
  every random machine in our runs and 54 of 55 random machines checked (`random-q16` seeds 0-39
  among them; the feasibility register's row 17), so they have no approximation to be given or to find
  (for random permutation machines a theorem: Corollary 2 of indrajala-research's
  `primitive-machines-note.md`, from Dixon 1969's Theorem 2, primitive with probability at least
  1 - 1/q + O(q^-2); for our random transformation machines, `random-q` among them, an observation). The arithmetic and the algebras
  have many (division its p + 1 lines; S3 4, S4 16, move-to-front-4 264), and T4 found the algebras'
  networks holding them unasked (14 of 22 factors). **The hypothesis (H), stated as one about scale:**
  past the sizes where primitive machines still certify (`random-q12-0`: 2 of 3, 1 of 4), a target
  with an approximation is learnable where a primitive one of the same shape isn't, found by the
  network (as on the algebras) or given as an output (as on division, K4). **Rivals H must beat:** the
  state-formation work's reading of random machines' failures (the most crowded melting cascade;
  `order`, the answers' predictability against random machines, which agrees with primitivity on all
  five backdrops read, so the existing data can't tell them apart); and a cover by components that
  are no quotient of the target (Krohn-Rhodes), which a congruence-free target still admits.
  - **Arms (three, the same shape: q states, the identity and 4 random maps, 4 classes):** P0, primitive
    random machines (`random-q<q>-<m>`); P1, planted machines (`random-planted-q<q>-b<b>-<m>`, #981:
    random-q's machine with its states in b blocks a random b-state machine moves between), the target
    alone; P2, P1's machines with the planted quotient given as an extra output (a build: a query token
    asking the block, as division's encoding asks its approximations; tested, the golden run kept).
  - **The machine is the unit:** 15 machines an arm, 2 seeds each (90 runs; 10 before the power
    amendment below). P1 and P2 share their 15 machines (paired). Draws fixed in advance: P0 the first 10 machine seeds whose `congruences` says
    primitive; P1 the first 10 whose only nontrivial principal congruence is the planted one (so the
    planting is the one designed difference; draws that gain others are skipped and listed). Planting
    also constrains the transitions (each token's target block fixed), so other properties can shift:
    the ones measured below are compared between P0 and P1 before any run, and stated. At q 16, b 4
    (offline, 2026-10-24): every `random-q16` seed 0-39 is primitive (P0: seeds 0-9); 47 of 60 planted
    draws carry the planted congruence alone (P1, P2: seeds 0, 4-11, 13, 16, 18-20, 22; the existence
    probe's: 15; P0 seeds 0-14, all primitive).
    Why 10 by 2, not 5 by 4: with P0 at 0, T10a at 5 against 5 machines needs 4 of 5 planted machines succeeding (p
    0.024; 3 of 5 gives 0.083), at 10 against 10 it needs 5 of 10 (p 0.016; 4 of 10, 0.043, misses
    Holm's 0.025): the claim is about machines, so machines buy power where seeds don't.
    The planted draws (an owner's question, 2026-10-24): (a) *chosen*, skip a draw that gains any other
    congruence. Pros: the planting is the one structural difference from P0; the skipped draws carry
    more structure, so if anything the skip works against H (conservative). Cons: a selection, so the
    claim is about machines with exactly one approximation. (b) Every draw kept, its extra congruences
    reported. Pros: no selection. Cons: the number of congruences varies, so a difference could be
    more structure, not an approximation's existence, and 10 machines can't adjust for it. (c) Every
    draw kept, read by its number of congruences. Pros: asks whether more approximations help more.
    Cons: a second question, without the power to answer it.
  - **Measured beside each machine, before any run:** `congruences`, `order` (rms z against its
    ensemble; offline 2026-10-24 the planted q16-b4 draws read 0.32, 1.01 and 2.32, the b2 one 0.81,
    against 0.86 and 1.07 for `random-q16-0` and `-1` and 6.13 for move-to-front-4: planting a
    congruence doesn't make a machine ordered, so H and `order` come apart here), Chow's `k`, the
    crucible depth (row 15), each token's rank (its image's size) and whether the machine
    synchronises (a reset word). The melting cascade (the other rival) is read from P0's and P1's runs
    afterwards, exploratorily.
  - **Size and budget (registered 2026-10-24, after T5's batch ended; filled by the rules above,
    nothing else changed):** T5's `rq16` certified **0 of 20** (every run spent at 600 epochs; the
    certificates, so the long tests can't raise it), so **q 16, b 4 (blocks of 4)**, the draws as
    drafted: P0 `random-q16` seeds 0-14; P1 and P2 planted seeds 0, 4-11, 13, 16, 18-20, 22 (1-3, 12,
    14, 17 and 21 skipped: other congruences); the probe's machine seed 15. **Run seeds 400 and 401**
    on every machine (fresh: T5 ran `random-q16-0` at 300-319). **The epoch budget: T5's, the `paper`
    cell's 600.** T5's `rq16` runs took 113-267 minutes (median 186) to spend it on the hosts of its
    batch, so the batch is about 90 x 3 h, half a day on every machine. **The bars** (`rank-bars
    --machines 15`, exact, 15 against 15, Holm's 0.025 for the smaller p): P0 at 0 machines
    succeeding, 5 machines (p 0.0211); at 1, 7 (0.0176); at 2, 9 (0.0105); at 3, 10 (0.0127).
    **Feasibility (every cell, as D16's table for `rq16`):** row 1, 12 codes a factor, 144 joint (9.0
    x q); row 2, the cap 159 (the exclusion bound at m 144, q 16); row 14, every state reachable and
    every pair told apart within the check (Chow's `k` 1-2, depth 4-6, below); row 15, crucible depth
    4-6, within its cap of 8 (descriptive: the `paper` cell has no crucible); row 17, P0 primitive, P1
    and P2 the planted congruence alone (4 blocks of 4), checked by
    `test_t10s_cells_meet_their_registered_bounds_and_draws`. **P0 against P1, measured before any
    run** (`data/window-recurrent/paper/t10/measures-q16.txt`, 31 draws): crucible depth mean 4.8
    both (4-6); Chow's `k` 1-2 both; every machine synchronising; `order` mean 0.65 (0.15-1.07)
    against 0.83 (0.18-2.07: P1 seeds 18 at 2.07 and 5 at 1.56 above P0's maximum); P1's non-identity
    tokens rank lower, 9.37 against 10.25 (the planting fixes each token's target block, narrowing its
    image). The probe's machine (seed 15): `order` 1.84, token ranks 16, 10, 5, 9, 6. **The backdrops**
    in `indrajala_ml/batch.py`: `t10-p0-<m>`, `t10-p1-<m>`, `t10-p2-<m>` (`--arm planted`),
    `t10-probe-p1`, `t10-probe-p2`. **The existence probe:** `t10-probe-p1` and `t10-probe-p2`, run
    seeds 400-403 (8 runs), the `paper` cell at its **600 epochs** (amended 2026-10-24 before any of the probe's data: the
    registered 400-epoch cut isn't expressible, the runner's `--cutoff-epochs` being for continuous
    runs only; its first launch, `t10/probe-void/`, refused at the argument check, nothing trained;
    the paper cell certified T5's machines at 52-394 epochs, median about 183); it passes if either
    arm certifies once. If neither
    does, T10 drops to q 12 (b 4, blocks of 3: P0's own rate the control's measure; draws by the same
    rules) or is reported as not run, with the probe. Where: `klip` and `pyramidon` (3 slots each) while T2
    and T5's long tests hold `jebel` and `stone`; the batch on the runner once they end.
  - **Success:** as D13's (certified at the stop, both verdicts, every certified network long-tested).
  - **Tests (one-sided, machine-level, Holm over the two), each against the primitive control:** T10a
    P1 more often than P0 (found: the target alone), T10b P2 more often than P0 (given, as K4 on
    division); each an exact Wilcoxon rank-sum over the 10 against 10 machines' success counts. Holm
    over the two: the smaller p must reach 0.025. P2 against P1 (does giving add to having; paired,
    a signed-rank) is reported, exploratory. The counts (0-2 a machine) tie heavily, so both tests are exact permutation tests
    with ties (midranks), their attainable p-values computed and stated at registration; the pooled
    Fisher's over runs is reported beside them as descriptive, never as the test. **Power, stated:**
    with P0 at 0, a test passes when 5 of 10 machines succeed at least once (p 0.016); at a per-run
    rate r a machine does with probability 1 - (1 - r)^2 (0.51 at r 0.3), so a rate of about 0.3 is
    near the limit this design detects; a smaller effect reads as undecided, not as a refutation.
    **The power amendment (2026-10-24, before T5's `rq16` was read; the bars above are at 10 machines,
    as drafted):** the power above held only with P0 at 0, but the size rule allowed P0 2 of 20, and
    one P0 success moves the bar (exact, 10 against 10: 5 of 10 machines p 0.070, 6 of 10 0.029,
    7 of 10 0.010). Simulated power of T10a (p at most 0.025), P0's per-run rate r0 against P1's r1:
    10 machines, r0 0: 0.28, 0.65, 0.90 at r1 0.2, 0.3, 0.4; r0 0.1: 0.07, 0.20, 0.45. 15 machines,
    r0 0: 0.69, 0.95, 1.00; r0 0.05: 0.28, 0.63, 0.88. 20 machines, r0 0: 0.89, 0.99, 1.00; r0
    0.05: 0.40, 0.78, 0.97. Options: (a) unchanged, the conditional bars stated. Pros: no change, 60
    runs. Cons: at P0 2 of 20, power 0.20 at r1 0.3, so T10 reads undecided even where H holds. (b)
    The size rule tightened, q 16 only if `rq16` certifies 0 of 20, else q 24. Pros: H's own
    statement (past the sizes where primitive machines still certify); P0 near 0. Cons: power still
    0.65 at r1 0.3; q 24 may not form (the existence probe guards it). (c) *Chosen:* (b) and 15
    machines an arm (90 runs). Pros: power 0.95 at r1 0.3 with P0 at 0, 0.63 if P0 leaks at 0.05;
    the draws exist (P0 seeds 0-14 primitive; P1 the next five carrying the planted congruence
    alone). Cons: 1.5 times the runs. (d) 20 machines. Pros: the most power. Cons: twice the runs,
    for less gain over (c). The attainable p-values at 15 against 15 are computed at registration,
    as before.
  - **Expectations and refutations:** P0 about the T5 rate (0 of 20 at q 16 by the size's choice, the power amendment).
    **H stands at q** if T10a holds (the network finds a planted approximation it isn't given); its
    weaker form (given, as on division) if T10b holds alone; **H is refuted at q** if P1 and P2 both
    succeed no more often than P0 (the approximation present and given, and no help); between,
    undecided. Within
    P1, success against `order` is read exploratorily (H predicts no dependence on `order`).
  - **Existence first:** before the registered batch, one planted machine at q (not one of the 10), 4
    seeds, P1 and P2,
    on a short budget: does the cell form it at all? If neither arm ever forms, T10 isn't feasible at q
    (row 1 holds, but no cell is shown able to succeed): drop to the next smaller q, or report T10 as
    not run, with the probe.
  - **Where:** the runner, when T2/T5's batch and the long tests it needs have the machines; data
    `data/window-recurrent/paper/t10/`.
  - Options: (a) *chosen*, planted against primitive machines of one shape, several machines an arm.
    Pros: only the planted congruence differs; the machine is the unit, so the claim is about machines
    with approximations, not two machines; separates H from `order`. Cons: a build (P2's output); 2
    seeds a machine read each machine's own rate coarsely (the test doesn't need it). (b) The original
    T10 on S4's harder seeds. Pros: no build, S4 has approximations. Cons: S4 runs take 3-4 h; it asks
    whether an approximation rescues hard seeds of one machine, not whether having one matters, and
    can't separate H from `order` (S4 is ordered). (c) One planted and one primitive machine, 20 seeds
    each. Pros: the paper's usual 20 seeds. Cons: two machines, so a difference is the machines',
    not H's.

- **D20. T11 registered: the groups with their approximations' outputs** (the owner, 2026-10-25:
  "whether that helps IS most of the point of P1"; registered before any of its runs, after T5's
  reading). K4 claims the target's approximations given as outputs make product machines learnable,
  tested on division only (K4; T2b: `d11-family` 16 of 20 against the quotient alone 0 of 20). T5's
  algebras trained on their answers alone and certified 1-4 of 20 (T5a), although their certified
  networks held approximations unasked (T4: 14 of 22 factors). T11 gives the algebras theirs, as
  division's encoding gives its sum and difference. **One change from T5:** the outputs; the `paper`
  cell (the fork, the commitment as built, 600 epochs, stopped at the first certified step), the
  32-wide window and T5's backdrops otherwise. The rule is now the protocol's (indrajala-research
  `experiment-design.md` section 1, #60).
  - **The rule (the owner's): two approximations that together determine the state, the coarsest
    such pair** (division's cover in two factors, matching `--factors 2`): among every congruence of
    the machine (the principal ones, `Mealy.principal_congruences`, closed under joins), the pairs
    whose meet is the identity; the pair with the fewest blocks in total; ties to the simplest
    quotients (the smallest transition monoids, summed); then the first in the machine's state
    order. Computed offline 2026-10-25 (`congruences`; the closure and the search a script, made a
    tool in T11's build):

    | backdrop | congruences | the pair (blocks; monoid) | reading | tied after blocks, after monoids |
    | --- | --- | --- | --- | --- |
    | `s3` | 4 | 2 (2) and 3 (6) | the sign, and a point's place | 3, 3 (the three points) |
    | `s4` | 28 | 6 (6) and 4 (24) | S3 (the Klein four-group's cosets), and a point's place | 16, 4 (the four points) |
    | `mtf4` | 264 atoms | 6 (10) and 8 (15) | the order of items 0, 1 and 3 (`order {0, 1, 3}`), and the set of items more recent than item 2 | 4, 4 |

    Each pair's states are the target's exactly (6 = 2 x 3; 24 = 6 x 4; move-to-front's 6 x 8 = 48
    label pairs, 24 of them reached). The remaining ties are relabellings (the points, or the item
    left out), so the state order's choice changes no structure. S3's pair holds the sign, which
    T4 never found a network holding unasked (K8): whether it is used once given is read (T4's tool).
  - **Arm and backdrops:** `--arm cover` (a build: `window_tasks.query_partitions` names the pair
    for `word-problem-s3`, `word-problem-s4` and `move-to-front-4`, queries `cover-a` and `cover-b`,
    the coarser first; the encoding asks them as T10's P2 asks its planted block; tested, the golden
    run kept). The runner's backdrops `t11-s3`, `t11-s4`, `t11-mtf4`.
  - **Seeds: 20 fresh a backdrop, 700-719.** Not T5's 300-319 paired: the extra outputs change the
    output layer and the drawn examples, so a seed's run isn't T5's twin; the comparison is two
    rates (D5).
  - **Feasibility** (T5's, D16: the codes and caps unchanged, the queries adding outputs, not
    states): `s3` 8 codes a factor, 64 joint (10.7 x q), cap 69; `s4` and `mtf4` 15, 225 (9.4), 248;
    the window 32 wide (row 3); every state reachable and told apart (row 14); row 17, each pair a
    congruence pair meeting in the identity, checked by `test_t11s_cells_meet_their_registered_bounds`.
  - **Cost** (the probe, 2026-10-25, `pyramidon` beside T10's three runs, seed 699, 30 epochs, claims
    nothing; `t11/probe/`): S4 714 s alone (about 24 s an epoch), move-to-front-4 884 s and S3 1,068 s
    in one pool (29 and 36 s), against T5's 6.1-6.6 s an epoch: a column three positions (the token
    and two queries), on a loaded host. Peak RSS 54-63 MB (`time -v` sees only `--runs`' parent, as
    T5's probe; T5 measured 69 MB a run), so memory never binds and cores set the slots. A spent run
    about 2-6 h; with T5's failure rates (16-19 of 20 spent) the batch is about 150-250 run-hours,
    about a day on the runner. Where: the runner once T10's batch frees its slots.
  - **Success:** as D13 (certified at the stop, both verdicts, every certified network long-tested,
    D16's lengths).
  - **T11a, K4 beyond arithmetic:** per backdrop, the cover's certified rate against T5's (`s3` 1,
    `s4` 4, `mtf4` 4 of 20), Fisher's exact, one-sided (K4 names the direction), Holm over the three
    in the paper's family (D5). **Stands** on a backdrop if significant at its Holm level; the exact
    bars, one-sided: against 1 of 20, 8 at 0.0167, 7 at 0.025, 6 at 0.05; against 4 of 20, 12, 11
    and 10. **Refuted** on a backdrop if the cover certifies no more often than T5; undecided
    between. **K4 generalises to the algebras** if T11a stands on at least one, reported per
    backdrop.
  - **T11b, the reliability bar:** per backdrop, D15's bar (a Wilson lower bound of 0.25, 9 of 20),
    reported: if met, K7 is restated as "learned reliably given their approximations".
  - **Expected:** higher on all three, as division's 0 to 16; `s4` and `mtf4` about 10-14 of 20,
    `s3` the least certain (its failures never formed a state, T5a; the sign never held unasked).
  - **Then:** T4 on every certified network (do its factors hold the given pair, or others?); T6 on
    every run (D11, its rules), reported apart.
  - Options: (a) *chosen*, the cover pair by the rule, the `paper` cell, T5 the control. Pros: one
    change from T5, its runs the control; the rule is division's, so K4 is tested as stated. Cons: a
    build; T5's control ran at another commit (bit-identity of the `paper` cell's training checked by
    the golden run, the control not rerun). (b) Every maximal approximation as an output. Cons: many
    outputs (move-to-front's 264 atoms); not division's form. (c) One approximation only. Cons: it
    doesn't determine the state, so not K4's cover.

- **D21. T7's reruns registered: the paper's runs bit for bit on another machine** (2026-10-25,
  before any rerun). **The sample, by a rule fixed now:** the shortest certified run (by minutes) of
  each of T1, T3, T2 and T5, rerun on `litshe` (aarch64, the crate's NEON path; every original ran on
  x86-64): T1 `d5-sum-319` (4.6 min on its host), T3 `approximation.d5-family-wide-319` (6.3), T5
  `s3-302` (7.4), T2 `d7-family-309` (11.8), in that order, one at a time (<= 70 C). Each by the
  runner's own command (`batch.BACKDROPS` and `CELLS`: the backdrop's arguments and its batch's cell),
  at main (7dc0915) with the crate 647f23c (the batches': T2/T5's and T3's recorded, T1's attempts on
  `pyramidon`'s; T1's `stone` attempts ran 45d6a37, `same-prefix` identical to 647f23c, T1's
  findings).
  - **Expectation:** each rerun bit for bit the original: `same-prefix RERUN ORIGINAL` every row the
    same but `seconds` (`differs` None), and its certified network's sha256 the original's.
    **Refuted** by any difference, reported with the first differing step and diagnosed (an x86-64
    against aarch64 float difference, or code drift since the batch's commit) before T7 claims
    reproducibility; the paper then states what reproduces (the same machine, or the certificate's
    verdict) rather than bit identity.
  - Options: (a) *chosen*, one shortest certified run a test on `litshe`. Pros: every test's code
    path checked (both cells, division and an algebra), across architectures and from the current
    main, in a few Pi hours; a certified run checks the certified network too. Cons: no spent run
    (a failure's longer course); 4 runs, so a rare divergence could pass unseen. (b) A random sample
    of 10. Pros: no selection. Cons: spent runs take days on a Pi. (c) `jebel` (x86-64) as well.
    Pros: the same architecture, a cleaner reading if `litshe` differs. Cons: `jebel` is T10's;
    deferred to stage 5's `jebel` reruns, as the handover plans.

- **D22. Paper 1 is self-contained: it cites no unpublished work of ours** (the owner, 2026-10-25:
  "P1 has to be self-contained with references only to external authors"; the internal notes, P1
  and P2 here, may cite each other, and a later paper may cite paper 1). Amends D7: the manuscript
  is built from P1 and P2 but cites neither. The results it relies on become its own, reported with
  their data: addition's tests against the certificate, its code-count comparison, its replays and
  the certificate guard (K2, K10) in the manuscript's Appendix A; K4's 25-state comparison from the
  manuscript's own approximation table (15 against 1 of 20, in place of P2's 13 against 1); K1's
  count is the 385 networks reported there (in place of P1's 724). So the datapack (stage 8) adds
  addition's studies (`matrix`, `control`, `certificates`, `long-tests`, `e3-e4`, `stable-replay`,
  `replay-guard`) and `approximation-claim`, with their manifests. Two misattributions fixed on
  the way: K2's counterexamples and K10's replays are on addition, not modular sums.
  - Options: (a) *chosen*, fold the relied-on results in as the paper's own (an appendix). Pros:
    every claim keeps its evidence; the data exist with manifests; the paper supersedes P1 and P2
    (D1, D7) anyway. Cons: a longer paper; the appendix's numbers are prose from P1's findings, not
    regenerated by `paper_tables.py` (no table). (b) Drop every claim resting on P1 or P2. Pros:
    shorter. Cons: K2 loses its counterexamples and K10 its mechanism, the paper's two clearest
    illustrations. (c) Publish P1 and P2 first and cite them. Pros: the citations stand. Cons: two
    releases before this one, against the schedule; the owner's rule names paper 1 as the one
    published.

- **D23. The prose's numbers regenerated, as the tables are** (2026-10-25; the owner: "we need some
  tool to auto-compile the appendixes numbers, even if there is no table"). Each number in the prose
  is marked `\pn{key}{value}` (the manuscript defines `\pn` to print its value); `paper_tables.py`'s
  `NUMBER_SOURCES` compute every key from the result files, `check` fails on any difference or a key
  with no source, and `fill` writes the regenerated values in. **Stage 1 (#1012):** Appendix A's,
  K2's and K10's addition numbers (71 marks). **Stage 2 (2026-10-25):** every other measured number
  in the results' prose (K1, K3-K9 from T1-T6's data): 131 marks in all, every one the same; the
  numbers left unmarked are settings (lengths, bounds, seeds). p values written to two significant
  figures throughout (0.0101 became 0.010, 6.44e-6 became 6.4e-6).
  - **Found by stage 2, corrected:** K1's "every network a long test failed is one the certificate
    rejected" held only vacuously (the 20 that failed are T3's continuous networks, with no
    certificate; no coded network failed a case): restated so. K8's "each into exactly one of its
    trained lines" omitted the quotient-alone network's factor holding `B` (T4's findings name it):
    restated, a network trained on no approximation finding one of its own.
  - **Found by stage 1, corrected in the manuscript** (the notes' own recorded values, now
    regenerated): the first counterexample network (1,559 + 6,329 + 5,231) answers 19,680 read at
    12 columns, not 8 (at 8 it is right: it fails only after a leading zero window); the first
    flips' update norms are 0.015-0.046, not 0.016-0.046 (replay 3's 0.01545); "94-164 pairs wrong
    at the worst" is over all 10 replays, not the 9 that amplified (94-160); the code count's
    Holm-adjusted p is 0.043 (3 times 0.01418), not 0.042 (3 times the rounded 0.014).
  - Options: (a) *chosen*, inline checked marks `\pn{key}{value}`. Pros: the source still reads as
    prose; every number tied to its computation; one `check` for tables and prose; `fill` updates
    in place. Cons: markup in the source; a value's wording (e.g. "none", "every") must come from
    its source. (b) A generated macros file (`\newcommand` per key, `\input`). Pros: compiled in
    without a fill step. Cons: the source no longer shows its numbers; a generated file to keep in
    sync across the two repositories. (c) A sidecar list of expected numbers matched in the text.
    Pros: no markup. Cons: fragile matching; a number edited in the text slips past.
- **D24. T10r drafted: T10b replicated on fresh machines** (drafted 2026-10-26, before any of its
  runs and before T10's long tests end; registered when the owner approves the draft). **Why:**
  read provisionally on its recertified networks (the long tests still running), T10b, the planted
  approximation given as an output (P2) against primitive machines (P0), passes its registered bar
  with no margin: P0 2 of 15 machines, P2 8 of 15 (seed 16 twice), p 0.0206 against Holm's 0.025; with
  seed 16 succeeding once, 0.0251. T10a (P1, the approximation present, not given) reads 3 of 15, p
  0.50. A replication tests T10b on machines that didn't produce it; **it is read alone, never
  pooled with T10** (adding machines after T10's p is seen would be optional stopping), and T10's
  findings are written by D19 whatever T10r finds.
  - **The question:** given its planted approximation as an output, does a random 16-state machine
    form more often than a primitive one of the same shape? (T10b's, K4 beyond arithmetic.) P1
    isn't repeated: T10a read flat, and the claim at stake is T10b's.
  - **Arms (D19's, unchanged):** P0 primitive `random-q16-<m>`; P2 `random-planted-q16-b4-<m>`,
    `--arm planted` (the block as an extra output). The `paper` cell, its 600 epochs, run seeds 400
    and 401 on every machine (the machines are fresh draws, so the runs are; the same seeds keep
    the cell exactly T10's).
  - **The draws (D19's rules, the next seeds after T10's, measured 2026-10-26 before any run,
    `data/window-recurrent/paper/t10r/measures-q16.txt`):** P0 `random-q16` seeds 15-39, all
    primitive (as D19 found for 0-39). P2 the first 25 planted seeds from 23 whose only nontrivial
    principal congruence is the planted one: 23, 24, 27-41, 43, 44, 46, 48-52; skipped 25, 26, 42,
    45, 47 (other congruences). **P0 against P2, before any run:** `order` mean 0.77 (0.09-1.48)
    against 1.11 (0.29-2.44); Chow's `k` 1-2 both; crucible depth mean 4.7 against 4.9 (4-7 both);
    P2's non-identity tokens rank lower, 8.80 against 10.43; every machine synchronising (the same
    shifts T10's draws showed).
  - **Size: 25 machines an arm, 100 runs.** Simulated power (one-sided rank-sum at 0.05, 2 seeds a
    machine; P0's per-run rate r0 against P2's r2): at T10's provisional rates (r0 0.067, r2 0.3),
    0.68 at 15 machines, 0.82 at 20, **0.91 at 25**, 0.96 at 30; at r0 0, r2 0.2: 0.84, 0.91,
    0.97, 0.99.
  - **The test:** one, so no Holm: an exact one-sided Wilcoxon rank-sum (midranks) of P2's machines'
    success counts (0-2) against P0's, **p at most 0.05**. The bars (`rank-bars --machines 25
    --control 0 1 2 3 4 5`, every success a machine's only one): P0 at 0 machines succeeding, P2
    needs 5 (p 0.025); at 1, 6 (0.049); at 2, 8 (0.037); at 3, 9 (0.048); at 4, 11 (0.031); at 5, 12
    (0.036). The pooled Fisher's over runs reported beside it, descriptive.
  - **Success:** as D13's (certified at the stop, both verdicts, every certified network
    long-tested). T6's proof (D11 amended) on its runs, reported apart, as T10's.
  - **Expectations and refutations:** at T10's rates, P0 about 3 of 25 machines and P2 about 13.
    **T10b replicates** if the test passes. **Not replicated** if it doesn't: then T10b is reported
    with T10r beside it, as one passing test not confirmed, and the paper's claim falls back to
    existence (random machines certified given their approximation), not a rate's difference.
  - **Feasibility (every cell, as D19's):** row 1, 12 codes a factor, 144 joint (9.0 x q); row 2,
    the cap 159 (the exclusion bound at m 144, q 16); row 14, every state reachable, every pair told
    apart within the check (Chow's `k` 1-2); row 15, crucible depth 4-7, within its cap of 8
    (descriptive); row 17, P0 primitive, P2 the planted congruence alone. Tested by a backdrop test
    beside `test_t10s_cells_meet_their_registered_bounds_and_draws`.
  - **Where and when:** the runner, one batch longest first (P2 then P0, as T10's ledger's times
    rank them), after T11's batch ends (about Tue 2026-10-27 morning): backdrops `t10r-p0-<m>` and
    `t10r-p2-<m>` in `indrajala_ml/batch.py`, data `data/window-recurrent/paper/t10r/`. Cost from
    T10's ledger: median 2.9-3.0 h a run (longest 12.0 h, a P2), so about 300 run-hours, roughly
    12-14 h on the fleet (`stone` nights only): the release moves about a day.
  - Options: (a) *chosen*, 25 machines an arm, P0 and P2 only, read alone at 0.05. Pros: power 0.91
    at T10's rates; machines are the unit, so power comes from machines, not seeds; fresh draws
    test the claim, not T10's machines. Cons: 100 runs, about a day of the fleet; the release moves.
    (b) 15 machines an arm (T10's size). Pros: 60 runs, about 8 h. Cons: power 0.68 at T10's rates,
    so a true effect fails to replicate a third of the time. (c) 20 machines. Pros: 80 runs. Cons:
    power 0.82. (d) 30 machines. Pros: power 0.96. Cons: 120 runs, needs `random-q16` seeds past
    39 (unmeasured), little gain over (a). (e) More seeds on T10's machines, merged. Pros: no new
    draws. Cons: optional stopping; seeds don't add machines, the test's unit. (f) Keep Holm's
    0.025. Pros: T10's bar. Cons: one test, no multiplicity to correct; power falls (0.81 at 25).

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
- **A check for P2 before the manuscript (done 2026-10-23, exploratory, not T4):** P2's "the learned
  state is not factored by residue" reads the modular-family probe's first division networks
  (`modular-family-probe/family-output-seed0`, `seed1`; `scripts/modular_family_probe.py`'s
  docstring), not probe 2a's. Its numbers are seed 1's **step 6**, which is not a solved step: 98.7%
  right on samples, it fails the certificate (1,025 wrong outputs). The two networks that certified
  (seed 1 step 11, seed 0 step 12), read by `approximations` now: **no factor of either determines
  any nontrivial approximation** (each factor's finest one block; 14 and 10 codes, 14 and 12), nor
  does step 6's. So the section's claim holds for its networks, its "solved step" is corrected in P2
  (read at an uncertified step; the certified ones agree), and it is not a general claim: in T4 a
  third of later division factors hold a line. Outputs and manifest:
  `data/window-recurrent/paper/p2-check/`, `paper/p2-check.sha256` (4 files).
- **For the paper (D10's use):** K8 stands for the algebras on networks that didn't shape it (T4a,
  T4b). On division it holds in a narrower form than K8 states: a third of the factors, each
  holding exactly a trained approximation; stated so, with the count as registered (mixed). The
  sum arm's factors (only ever the sum) are the sharpest observation, exploratory (not registered),
  and T1's arms (the suffix and token outputs) can test it: a factor holding a non-approximation
  output exactly would say the factor holds what it is asked, not what is an approximation.

## Findings: T1, the approximation against other extra outputs (2026-10-24, read by D13's rules)

The runner, one batch (`data/window-recurrent/paper/t1/`), 60 attempts at 51c4167, 2026-10-23
to 19:05: controller on `pyramidon`; workers on `pyramidon` (crate 647f23c), `klip` and `stone`
(crate 45d6a37: T1 started before the crate check, #969; `same-prefix` on `d5-sum-304` against a
647f23c rerun, 31 rows identical). Three attempts (`d5-sum-300`, `d5-sum-302`, `d5-suffix-302`)
were lost on `pyramidon` at a reconnect and re-run in their own directories, appending to their logs
(the re-run is deterministic: the same seed, the same rows); the runner (#974) and `readout` (#976)
now read a log's last run, so each counts once, at 400 epochs. Read on `pyramidon`: `t1/read.sh`
(recertified in both verdicts, `t1/certify/`), the 7 long tests one at a time (2,719-3,702 s each,
2026-10-23 19:36 to 2026-10-24 02:05; the long-test code is unchanged from 51c4167 to 028cb02), then
`readout t1/runs --certify t1/certify --long-tests t1/long-tests --pairs sum:suffix sum:token`
(#978) into `t1/readout-long.txt`, `long-summary` (#977) into `t1/long-tests-summary.txt`.

| arm | succeeded (certified, both verdicts, long-tested) | Wilson 95% | failures near a solution | failures collapsed |
| --- | --- | --- | --- | --- |
| `d5-sum` (an approximation) | 6 of 20 | 0.15-0.52 | 11 of 14 | 10 |
| `d5-suffix` (state-carrying, no approximation) | 0 of 20 | 0.00-0.16 | 0 of 20 | 2 |
| `d5-token` (stateless) | 1 of 20 | 0.01-0.24 | 0 of 19 | 7 |

- **T1a holds:** sum more often than suffix, Fisher's exact one-sided p 0.0101, Holm over the two
  0.0202. **T1b holds:** sum more often than token, p 0.0457, Holm 0.0457. **K5's "because"
  stands** (D13: both under 0.05): an extra output the network can learn does not help as the
  approximation does; the state-carrying one never certified.
- **The sum's rate is below its expectation:** 6 of 20 against about 10 (A2's arm, on its own
  vocabulary `sum difference product`). Its failures mostly
  came near (11 of 14 had every answer at >= 0.95 at some diagnosis) and lost it; the controls'
  never came near. Stated so in the paper.
- **Every certified network certified in both verdicts and passed every long-test case:** 7
  networks (6 sum, 1 token), each minimised to 25 states, 0 wrong outputs and states, codes reached
  59-105 (59-103 strict), snap ratio >= 1.001, logit lead >= 0.247; 375,000 cases each (U1 100,000,
  S1 25,000, at 128, 256 and 1,024 columns), 2,625,000 in all, none failing. The sum's 6 took 55-161
  epochs (4.6-15.8 min), the token's one 186 (19.8 min).
- **Exploratory, not registered (D13): `approximations`** (`t1/approximations/`,
  `approximations-sum.txt`, `approximations-token.txt`): 3 of the 6 sum networks hold `A + B`
  exactly in one factor (5 blocks; the other factor none), as T4's sum arm did (7 of 10); no factor
  holds another line, and the token network holds none. No suffix network certified, so T4's open
  question (does a factor hold a trained output that is no approximation?) stays open.
- **D8 decided:** T1 leaves the approximation's role standing, so T10 runs. It is registered in
  its own decision before any of its runs, placed when T2/T5's batch frees the machines.
- Data `data/window-recurrent/paper/t1/`, manifest `paper/t1.sha256` 9a8026fa...e6d745508f (1,472
  files, run from `paper/t1`).

## Findings: T3, the snap and the transformer at scale (2026-10-24, read by D14's rules)

The runner, one batch (`data/window-recurrent/paper/t3/`), 60 attempts, ended before T2/T5's
launch; recertified in both verdicts (`t3/certify/`). Long tests on 33 networks (the control's 13
certified, the continuous arm's 20 `network.json`; a continuous state has no certificate, so the
long tests are its verdict): `t3/long-worker.sh`, each network claimed by an atomic `mkdir`, on
`pyramidon` (one at a time, memory) and `jebel` (`~/code/wt-batch` at 9ef332c, its long-test code
identical to main's; results rsynced to `t3/long-tests/`), 2026-10-24 02:06 to 16:43. Then
`readout t3/runs --certify t3/certify --long-tests t3/long-tests --pairs
approximation:approximation-continuous approximation:approximation-linear` into
`t3/readout-long.txt`, `long-summary` into `t3/long-tests-summary.txt`.

| arm | succeeded (certified, both verdicts, long-tested) | Wilson 95% | failures near a solution | failures collapsed |
| --- | --- | --- | --- | --- |
| `approximation` (the control: snapped codes, the transformer window) | 13 of 20 | 0.43-0.82 | 6 of 7 | 5 |
| `approximation-continuous` (no codes) | 0 of 20 | 0.00-0.16 | 20 of 20 | 0 |
| `approximation-linear` (no blocks) | 0 of 20 | 0.00-0.16 | 0 of 20 | 8 |

- **T3a holds:** the control more often than the continuous arm, Fisher's exact one-sided p
  6.44e-6, Holm over the two 1.29e-5. **K3 (the snap) stands.** Every continuous network came near
  (every answer at >= 0.95 at some diagnosis) and every one failed every long-test case: 0 of
  375,000 each, 7,500,000 in all, from 128 columns. A continuous state fits the training lengths
  and drifts beyond them; the snap is what makes the learned machine hold at every length.
- **T3b holds:** the control more often than the linear arm, p 6.44e-6, Holm 1.29e-5. **The
  transformer's role stands:** without the blocks no run came near.
- **Every certified control network passed every long-test case:** 13 networks, 375,000 cases each
  (U1 100,000, S1 25,000, at 128, 256 and 1,024 columns), 4,875,000 in all, none failing; certified
  at 50-339 epochs (6.3-67.7 min). 27 large drops in 15 control runs, 23 recovered.
- Data `data/window-recurrent/paper/t3/`, manifest `paper/t3.sha256` b98acaa7...599130b0 (1,696
  files, run from `paper/t3`, the claims directory left out).

## Findings: T4 on T2's and T5's fresh networks (2026-10-25, read by D10's rules)

D10's third set: `approximations` on the snapped certified network of each of T2's and T5's 60
certified runs (recertified in both verdicts, `t2-t5/certify/`), each with its run's task, arm and
vocabulary, 2026-10-24 (`t4-t6-reading.out`); text lines in `t2-t5/approximations/<backdrop>.txt`,
`tally.py` (T4's rules, from the text: no JSON was written) → `tally.txt`. Read before T2/T5's long
tests end; a network that fails one leaves this set, and the counts are redone.

| set | networks | factors nontrivial | networks naming one | covers | determines |
| --- | --- | --- | --- | --- | --- |
| `d7-family` (`p` 7) | 17 | 9/34 | 8 | 1 | `A + B` 2, `A - B` 7 |
| `d11-family` (`p` 11) | 16 | 9/32 | 8 | 1 | `A + B` 3, `A - B` 6 |
| `d13-family` (`p` 13) | 15 | 6/30 | 5 | 1 | `A + B` 2, `A - B` 4 |
| `s3` | 1 | 0/2 | - | 0 | - |
| `s4` | 4 | 1/8 | 1 | 0 | `places {3}` 1 |
| `mtf4` | 4 | 8/8 | 4 | 0 | `order {1, 3}` 3, `{0, 3}` 2, `{0, 1}` 1, `{1, 2}` 1, `{0, 1, 3}` 1 |
| `rq12` | 3 | 0/6 | - | 0 | - (primitive: no nontrivial congruence exists) |

- **T4a holds, exactly at its bar:** the algebras' factors 9 of 18 nontrivial (0.50; the bar a
  half, the refutation a quarter). MTF4 carries it (8 of 8, 5 naming an `order` congruence; the
  other 3 a two-block partition that is a congruence but no named one); S4 1 of 8, S3 0 of 2. Few
  fresh algebra networks (9), as D10's cons said.
- **T4b holds:** no factor holds `parity`; S4's one named congruence is a stabiliser's (`places
  {3}`).
- **T4c mixed, by its registered rule:** division's factors 24 of 96 nontrivial (0.25, exactly at the
  quarter: not "fewer than a quarter", so not the expectation; far below K8's half). By `p`: 7 0.26,
  11 0.28, 13 0.20. As in the first reading, every nontrivial division factor holds *exactly one
  line*, always a trained one (`A - B` 17, `A + B` 7; no `A`, `B` or other line); one cover in each
  `p`. The first reading's 0.37 on older networks and this 0.25 agree in kind: a minority of
  division's factors decompose, each into a trained approximation; the rest hold the state jointly.
- **For the paper:** unchanged from T4's first reading. K8 for the algebras now rests on two sets of
  networks that didn't shape it (both T4a and T4b holding twice, the second just); on division, the
  narrower form (a minority of factors, each a trained line), with both counts mixed as registered.
- `rq12`'s machine is primitive, so no factor can hold a nontrivial approximation: reported, outside
  T4a (a random machine is no algebra).
- Manifest `paper/t4-fresh.sha256` a68ad079...1b9ea2bc (9 files, run from
  `paper/t2-t5/approximations`).

## Findings: T6, the target-free proof (2026-10-25, read by D11's rules)

`answers-proof LOG... --task T --arm A --slack 1 1.1 --limit 20000000 --out DIR` on every coded run
of T1, T2, T3 and T5, 2026-10-24 (`t4-t6-reading.out`): 280 runs, in `t1/proof`, `t3/proof` (the
control and the linear arm; the continuous arm has no machine to read) and `t2-t5/proof`, a
`<group>.txt` and a JSON directory each; `t6-tally.py` → `t6-tally.txt`. A stop is *kept* when the
answers-only stop lands on the certified step, a *false stop* when it lands earlier.

| outcome | exact bound | 1.1 times the bound |
| --- | --- | --- |
| no stop (the run never certified) | 200 | 200 |
| kept, Chow proved | 80 | 43 |
| kept, out of Chow's reach (`p` 11, 13) | 0 | 17 |
| false stop, Chow proved the machine equal to the target | 0 | 5 |
| false stop, Chow refuted | 0 | **4** |
| false stop, out of Chow's reach (`p` 11, 13) | 0 | 11 |

- **The integrity check passes:** at the exact bound every one of the 80 stops is a certified run's,
  at its certified step, and Chow proved each (the strict verdict's equality, as the theorem says).
- **T6a holds:** no false discovery and no false stop at the exact bound; the 200 runs that never
  certified never stopped.
- **T6c holds:** every certified run's answers-only stop is its certified step at the exact bound;
  no "missed" run (0 of 80, the expectation under a tenth).
- **T6b is refuted as registered** ("at 1.1, no wrong machine where Chow reaches"): 4 stops landed on
  wrong machines, each refuted by Chow within its first batch of 4,096 queries, the counterexample
  4-7 tokens:

  | run | bound | stop, certified | counterexample |
  | --- | --- | --- | --- |
  | `d7-family-300` | 53 | step 7, step 11 | 5 tokens |
  | `d11-family-300` | 133 | step 17, step 22 | 7 tokens |
  | `d11-family-310` | 133 | step 13, step 32 | 5 tokens |
  | `d13-family-316` | 185 | step 10, step 11 | 4 tokens |

  Chow refuted each, so nothing false was proved: the failure is the answers-only stop rule's under
  slack, not the proof's. The slack reading (no false stop in 90 runs) was too small a sample to
  show it. The 5 proved false stops are the near-tie case: the network's machine already equal to
  the target, its margin short of certification.
- **What 1.1 costs:** the proof stays sound (it never accepted a wrong machine) but is incomplete:
  at `p` 11 and 13 the test set exceeds the limit (5.5e9 to 1.3e19 words), so 28 stops (17 kept, 11
  false) have the certificate's verdict only; 4 of those false stops were refuted early, the other
  11 are undecided by Chow.
- **Words each proof queried** (registered, reported; repeats dropped, so below D11's reach figures,
  which count the test set before repeats, `chow.test_set_size`): at the exact bound division `p` 5
  904, `p` 7 1,768, `p` 11 4,360, `p` 13 6,088; S3 50, S4 363, MTF4 1,187, `rq12` 305; at 1.1 `p` 5
  73,264, `p` 7 11,603,128, S3 50, S4 9,093, MTF4 29,705, `rq12` 1,530. `rq16` and `d11-quotient`
  never certified.
- **For the paper:** K9 stands at the exact bound on every paper run (80 proofs, no false stop). The
  slack claim is stated honestly: with a 10% looser bound the stop rule can land on a wrong machine
  (4 of 80), which Chow then catches where it reaches; a discoverer who knows the size only roughly
  needs the proof in the loop, not the stop alone.
- Manifest `paper/t6.sha256` 25d3145d...51a94fb5 (297 files, run from `paper`: the three proof
  directories, the tally and the reading's log).

## Findings: T2, the rates at scale (2026-10-25, read by D15's rules)

`readout t2-t5/runs --arms ... --certify t2-t5/certify --long-tests t2-t5/long-tests`
(`t2-t5/readout-all.txt`; the pair alone, `--pairs d11-family:d11-quotient`, in `readout.txt`). Every
run recertified in both verdicts; every certified network long-tested (`paper_reading.py long`, 18
shares: `jebel` 0-6 and three extras on `pyramidon`, `stone` 7-17; `long-tests-summary.txt`): **60
networks, 34,614,000 cases, every case passed**, so no rate below changes and T4's fresh counts
stand.

| backdrop | q | certified | Wilson 95% | epochs to certify (min-max, median) | failures near a solution |
| --- | --- | --- | --- | --- | --- |
| `d7-family` | 49 | **17 of 20** | 0.64-0.95 | 81-474, 195 | 3 of 3 |
| `d11-family` | 121 | **16 of 20** | 0.58-0.92 | 146-430, 261 | 2 of 4 |
| `d13-family` | 169 | **15 of 20** | 0.53-0.89 | 187-534, 313 | 4 of 5 |
| `d11-quotient` | 121 | **0 of 20** | 0.00-0.16 | - | 0 of 20 |

- **T2a holds at every `p`** (the bar a Wilson lower bound of 0.25): 0.64, 0.58 and 0.53. K6 stands:
  the cell learns these machines reliably at 49, 121 and 169 states, above the 10-15 of 20 expected.
  The trend (registered as reported, not tested): the rate falls by one network a step and the
  median epochs to certify rise (195, 261, 313); the slowest success, `d13-family` at 534 epochs, is
  within the 600 budget, so the budget crowds at `p` 13. Near a solution (the worst answer at least
  0.95) are 9 of the 12 failures.
- **T2b holds:** `d11-family` 16 of 20 against `d11-quotient` 0 of 20, Fisher's exact one-sided p
  7.7e-8, below Holm's smallest threshold in the paper's family (D5). **K4 stands at 121 states:**
  without its approximations' outputs the quotient alone never came near (its worst answer at best
  0.436-0.485 in every run, all 20 spent).
- Data `data/window-recurrent/paper/t2-t5/`, manifest `paper/t2-t5.sha256` 0eee086b...4b22893fa
  (6,960 files, run from `paper/t2-t5`), shared with T5's.

## Findings: T5, breadth rates (2026-10-25, read by D16's rules)

The same readout and long tests as T2's (one batch, D17).

| backdrop | q | certified | Wilson 95% | epochs to certify | failures near a solution |
| --- | --- | --- | --- | --- | --- |
| `s3` | 6 | **1 of 20** | 0.01-0.24 | 52 | 0 of 19 |
| `s4` | 24 | **4 of 20** | 0.08-0.42 | 157-394, 191 | 6 of 16 |
| `mtf4` | 24 | **4 of 20** | 0.08-0.42 | 152-309, 195.5 | 1 of 16 |
| `rq12` | 12 | **3 of 20** | 0.05-0.36 | 95-230, 120 | 8 of 17 |
| `rq16` | 16 | **0 of 20** | 0.00-0.16 | - | 11 of 20 |

- **Medians corrected by T7 (2026-10-25):** `s4`'s and `mtf4`'s were first written 200 and 207, the
  upper of the two middle values of four (157, 182, 200, 394; 152, 184, 207, 309); the median is
  191 and 195.5, as `paper_tables.py show tab:breadth` regenerates. Nothing else changes.
- **T5a:** `s3` is **refuted** (1 of 20, the bar's refutation at 1 or none), as expected ("undecided
  or refuted": a state that never forms; no failure came near). `s4` and `mtf4` are **undecided** (4
  of 20 each, lower bounds 0.08), well below the 13 of 20 expected from round 1's 4 of 6 at width 16.
  So **K7 is restated as existence at low rates:** the same cell certifies groups and a reset machine,
  each certificate proved and long-tested, at 1-4 of 20; no algebra meets the reliability bar.
- **T5b is reported with both rates:** `rq12` 3 of 20 doesn't meet the bar (lower bound 0.05), and
  `rq16` 0 of 20, so "random machines only to about 12 states" neither stands nor moves: existence
  at 12 states (3 of 20), none at 16. Both are primitive (D19), so neither has an approximation to
  give; T10 asks whether a planted one changes that.
- Collapses are common where the cell struggles (`rq16` 16 of 20 failures, `rq12` 12, `s4` 6, `s3`
  and `mtf4` 5), rare on division (at most 1 a backdrop): recorded for the failure kinds, not tested.
- **Then:** the groups' approximations given as outputs, T11 (registered in D20, before any of its
  runs).

## Findings: T7's reruns, bit for bit on aarch64 (2026-10-25, read by D21's rules)

D21's sample, each rerun on `litshe` (aarch64, the crate's NEON path) at 7dc0915 with the crate
647f23c, one at a time, peak 48-56 C; every original ran on x86-64.

| test | run | log rows (`same-prefix`) | certified network | sha256 (both) |
| --- | --- | --- | --- | --- |
| T1 | `d5-sum-319` | 74 identical, `differs` None | step 9 | 45c72add3cc9d641... |
| T3 | `approximation.d5-family-wide-319` | 65 identical, `differs` None | step 7 | aba8c525105e0fbe... |
| T5 | `s3-302` | 65 identical, `differs` None | step 6 | abe61dcbc72de473... |
| T2 | `d7-family-309` | 94 identical, `differs` None | step 6 | 0fb6ec7b16a44ced... |

- **D21's expectation holds:** all four are bit for bit their originals, every logged row the same
  but `seconds` and the certified network's sha256 the original's (`d7-family-309`'s six step
  networks too). So the paper claims bit identity across x86-64 and aarch64, from the current main,
  for the four tests' code paths (both cells, division and an algebra).
- `d7-family-309`'s first try died at 09:34 when `litshe` lost power or crashed (52 C, not
  heat); restarted from scratch at 14:35, its partial run kept as `d7-family-309.crashed-0934`.
- Data `data/window-recurrent/paper/t7-reruns/`, manifest `paper/t7-reruns.sha256`
  81e5f4d8...d5c2647be6 (71 files).

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
   identically), its further tables added with the manuscript (stage 6). **T7, 2026-10-25:** T1's, T3's, T2's and T5's tables regenerated from their batches (`tab:extra-outputs`, `tab:baselines`, `tab:scale`, `tab:breadth`), every number as their findings but T5's two medians (corrected there); T4's (`tab:decomposition`, `tab:decomposition-fresh`) and T6's (`tab:proof`, `tab:refuted`) the same day, every number as their findings. **Reruns (D21), 2026-10-25:** all four bit for bit on `litshe` ("Findings: T7's reruns"). T4's reading of the 69
   networks named in D10 is offline and may run now; T6 waits on its runs.
3. **Builds:** T1's two control outputs (suffix and token) as answer kinds of the division encoding;
   T3's continuous and linear windows in the division probe if not already there; each tested, the
   golden run unchanged.
4. **Registrations and runs:** T1 and T3 (short, `p` 5) first; then T2 and T5 (the long ones) across
   every machine; T10 and T11 (D19, D20) once registered; long tests on every certified network. Once the running probes end (D9).
5. **Readings:** T1-T6 by their registered rules, into the probe's findings and here.
6. **The manuscript** (D7), every number from the data by T7's script.
7. **Critique:** the audit's checks rerun on the manuscript; figures re-derived; a cold read (the
   owner's to arrange; a fresh-context reviewer as a first pass, if the owner agrees).
8. **The release (D6):** the datapack on Zenodo with a DOI, the repositories public (the literature's
   third-party PDFs excepted), the manuscript with its contribution statement and a page on how to
   verify it; then the owner's approach to Anthropic.

## Machines

The runner's hosts (`indrajala_ml.batch.HOSTS`): `jebel` around the clock since 2026-10-23 (7),
`stone` weekday nights and weekends (11; opened outside them by the owner's grant), `klip` (3, <= 85
C), `pyramidon` (3 beside development). Memory per run measured at each registration: under 100 MB
for every backdrop of T1, T2, T3 and T5 (D13-D16), so cores set the slots.
