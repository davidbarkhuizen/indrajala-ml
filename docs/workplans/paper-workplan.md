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

## Stages

1. **This plan;** the audit's standing decisions applied (F1 recorded in the state-formation
   workplan, the protocol amended for F9 and F24, F3's reword in contribution.md); the Chow tool
   merged (K9's instrument).
2. **Offline first:** T4 and T6's tools and expectations registered; T8's sweep; T7's regeneration
   script begun.
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
