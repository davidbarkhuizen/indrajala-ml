# Workplan: quasi-static crystallisation with incremental reheat

**Status: drafted 2026-10-27 at the owner's direction; nothing built or run.** The owner, on the
crucible probe's held cools: "stopping the cooling doesn't really help, only a reheat could allow it
to climb out of the energy well"; on D63's reheat, which may run back to the cool's start: "that is
bound to fail, or just be equivalent to starting again. we need a reheat where we heat it to just
below the previous critical temperature, so that we soften without losing the fundamental
structure"; "we should 'backtrack' to just above the temp of last transition, so that we undo only
the previous stage of incremental crystallization, not all the way back to scratch, if that doesn't
work after waiting a further time there, we can undo again"; "quasi-static I mean"; the name:
"quasi-static crystallization with incremental reheat". D1-D9 below are settled under the owner's
delegation, each with its options' pros and cons, unless marked for the owner.

It extends the collapse's remedy (D63 (a), the reheat, in
[state-formation-workplan.md](state-formation-workplan.md)) and feeds crystallisation's A/B (part
X of [post-paper-retests-workplan.md](post-paper-retests-workplan.md)), whose arm rule (its D5) it
amends (D8 here). The goal is [machine-discovery.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/research-notes/machine-discovery.md)'s
stage A, and the first item of the failure kinds' ranking
([failure-kinds.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/research-notes/failure-kinds.md),
2026-10-27): formation, a problem of training dynamics, not capacity.

## The IDs, in time order

| ID | When | Short label | What it is |
|---|---|---|---|
| D1 | 2026-10-27 | the stages | a transition is a new running maximum of a factor's code groups in the cool; each run keeps their stack |
| D2 | 2026-10-27 | the trigger | a hold that lasts one wait without recovering starts the reheat |
| D3 | 2026-10-27 | the ceiling | the clock runs back to the epoch before the latest transition, no further |
| D4 | 2026-10-27 | the pace | quasi-static: the cool's own path retraced, at most one epoch of clock an epoch |
| D5 | 2026-10-27 | the wait | the cool's e-folding time, `halving / ln 2` epochs (43 at the cell's 30) |
| D6 | 2026-10-27 | undoing again | not recovered after the wait at a ceiling: pop it, the next ceiling the transition before |
| D7 | 2026-10-27 | the empty stack | back to the first transition and not recovered: the run ends, spent, for a restart |
| D8 | 2026-10-27 | in X | the incremental reheat as X's second arm in place of D63's, if stages 0 and 2 support it |
| D9 | 2026-10-27 | the gate | stage 0 (offline) now; the build and every run after paper 1 (the owner may bring the build forward) |

## Why

**The held cool loses what it formed.** The crucible probe (2026-10-27, `continuous`, S4 seeds
300-303, with and without the recombining window; claims nothing) formed far more state than the
paper cell on the same seeds (state share up to 0.74 against 0.09-0.30), then every one of its eight
runs held its cool once the screen dipped, and stayed held: no temperature moved for over an hour.
At the held temperature the state drifted away or hardened onto a few codes. Base seed 300's trace:
the code groups split in stages as it cooled (2, 5, 7, 8, 10, 11, 13, then 15 at T 0.71, epoch
111), the state share reached 1.0 at epochs 105-109, the screen fell to 0.67, the clock held at T
0.64/0.69, and at epoch 184 both factors hardened onto 1-2 codes; a single step of crystallisation
back (T 0.64 to just above 0.71) is where the whole state existed. The best run (recombining 301)
fell from a state share of 0.74 to 0.30 while held.

**D63's reheat runs back too far, or not at all.** D63 (a) extends the hold's rate below zero, so
the clock runs back while the screen stays low, "never past the cool's start": a collapse that
doesn't recover in a few epochs re-melts the run (twice the critical temperature), losing every
stage, at the cost of the epochs it took to form them. Nothing ties how far it runs back to what
formed. The incremental reheat undoes one stage of crystallisation at a time, quasi-statically, and
only as far as needed.

**What is measured already.** Each epoch of the cool logs each soft factor's code groups
(`code_groups`: codes closer than `T max(1/100, 10 kappa)` squared, D20's), the temperatures, the
clock, the rate and the held temperatures (`floors`, D51). The stages of crystallisation are in
every log; nothing new is needed to see them.

## The design

In the cool (D48's continuous crystallisation, `--crucible --continuous`), with `--incremental-reheat`
(exclusive with `--reheat`):

1. **Record the stages (D1).** Each factor keeps the running maximum of its groups. An epoch at
   which any factor's groups pass its maximum is a transition; the run pushes the clock of the epoch
   before it (where the temperatures were just above the split) and every factor's maximum then.
2. **Hold as now (D49).** The screen below the hold's level stops the clock; recovered, it runs on.
3. **Trigger (D2).** A hold that has lasted the wait (D5) with the screen still below the level
   starts a reheat.
4. **Reheat to the ceiling (D3, D4).** The clock runs back along the cool's own temperature profile,
   one epoch of clock an epoch at most, to the latest transition's clock, no further. A held
   temperature (D51) the clock passes back over is released (D63's rule); growth only forward.
5. **Wait there (D5).** The clock holds at the ceiling for the wait. If the screen recovers, the cool
   resumes forward from there, the stage popped and the factors' maxima reset to its record (so the
   split is recorded again when it re-forms).
6. **Undo again (D6).** Not recovered after the wait: pop the stage; the next ceiling is the
   transition before, and the reheat runs on to it (4).
7. **The empty stack (D7).** Back to the first transition and still not recovered: the run ends,
   censored (`spent`), for the batch's restart (D63 (e)'s `t*`).

Logged in each cool epoch's `anneal`: the stack's depth, the ceiling's clock and temperatures, the
wait's epochs, and `reheats` (D66's count). The stack, the ceiling and the wait are in the run's
state, so a checkpoint resumes them (`--resume`).

## Decisions

- **D1. The stages: new running maxima of each factor's code groups.**
  - (a) *Chosen.* Pros: already logged every epoch; monotone, so the flicker between 14 and 15 at a
    near-tie isn't a transition; no constant. Cons: a split's epoch is read at `code_groups`'
    threshold, which moves with T; a stage can be several splits in one epoch (it is then one stage).
  - (b) The melt profile's transitions (`transitions`, the warm-up's states cooled offline). Cons:
    computed once, from the warm-up's states, not the network's own crystallisation in the cool.
  - (c) The hardening events (a factor's temperature held, D51). Cons: two a run at most (one a
    factor), too coarse to undo one stage.
- **D2. The trigger: a hold lasting one wait.**
  - (a) *Chosen.* Pros: the hold keeps its chance first (it recovered base seed 300's collapse
    once, after about 90 epochs); a reheat only for a hold that isn't working. Cons: epochs spent
    held before the reheat (43 at the cell's halving).
  - (b) D63's: reheat as soon as the screen falls below the line's extension. Cons: reheats on every
    dip, including those the hold recovers from.
  - (c) A collapse measure (codes in use falling). Cons: a second signal; the screen is D49's one.
- **D3. The ceiling: the epoch before the latest transition.**
  - (a) *Chosen.* Pros: just above the split's temperature, so the latest stage alone is undone;
    from the run's own record. Cons: if the stage that broke is older than the latest, one wait is
    spent first (then D6 undoes the next).
  - (b) The melting point (the first transition). Cons: undoes every stage at once: a restart in
    all but name.
- **D4. The pace: quasi-static, the cool's path retraced.**
  - (a) *Chosen.* The clock runs back at most one epoch of clock an epoch, the cool's own forward
    pace, so the temperature retraces the profile (exponential, D48) in reverse. Pros: the owner's
    "quasi-static"; the network meets each temperature for as long as it did cooling. Cons: a deep
    undo takes as long as the cool did over that stretch.
  - (b) A jump to the ceiling. Cons: not quasi-static; the codes see a step in T.
  - (c) The rate from the screen, as D63's (faster the further the screen falls). Cons: couples the
    pace to the depth of the fall, which is what made D63's run-back unbounded in practice.
- **D5. The wait: the cool's e-folding time, `halving / ln 2`.**
  - (a) *Chosen.* Pros: the cool's own time scale, the window D61 (c) and D63 (e) already use; no new
    constant (43 epochs at the cell's 30). Cons: may be short for a slow recovery (base 300 took
    about 90 held epochs).
  - (b) D63 (e)'s `t*` per backdrop. Cons: read from completed runs per backdrop, so not available
    for a new machine (discovery's case).
  - (c) Two e-folding times. Cons: a constant chosen by eye; stage 0 may motivate it (then decided
    there, before any run).
- **D6. Undo again: one stage at a time.**
  - (a) *Chosen.* Pros: the owner's "undo again"; only as deep as needed. Cons: a run whose break is
    many stages back spends a wait at each.
  - (b) Undo two stages each time (geometric). Cons: undoes stages that were fine.
- **D7. The empty stack: end the run for a restart.**
  - (a) *Chosen.* Pros: below the first transition nothing formed is left to keep; a fresh seed is
    the better use of the budget (D63 (e), path dependence). Cons: a run that would have recovered by
    re-melting is ended (reported: how many runs reach the empty stack).
  - (b) Re-melt (D63's limit, the cool's start). Cons: the restart in all but name, at the cost of
    the warm-up.
- **D8. In crystallisation's A/B (X): the incremental reheat as the second arm.** X's D5 runs
  `continuous-reheat` only if D63's paired verdict favours it.
  - (a) *Chosen.* X's arms become `continuous` against `continuous-ireheat`, if stage 0 shows held
    or collapsed runs with stages to return to and stage 2's probe shows the mechanism working
    (recoveries after a reheat); D63's verdict read and reported beside it (its arm not run, or run
    only if its verdict is clearly favourable and this plan's stage 2 is not). Pros: one remedy, the
    one designed against the failure seen; X's compute unchanged. Cons: D63's reheat gets no fresh
    batch unless its verdict earns one.
  - (b) Three arms. Cons: 50% more of X's runs.
  - (c) Its own batch after X. Cons: X adopts a default without the remedy for its commonest end.
- **D9. The gate: stage 0 now, the build after paper 1.**
  - (a) *Chosen.* Pros: stage 0 is offline (reading logs on disk), exempt from finishing the planned
    work first, and shapes the build; paper 1 keeps priority (post-paper D2). Cons: about a day's
    delay to X when it starts (the build and its tests).
  - (b) Build now on a branch, run nothing (as the recombining window was). Pros: ready when X
    starts. Cons: competes with paper 1's last week. **The owner's to choose; (a) unless the owner
    brings it forward.**

## Stages

0. **The offline reading** (now; on `pyramidon`, from logs on disk; a durable tool,
   `modular_family_probe.py transitions LOG ...`). For every run of D63's probe (22 seeds,
   `--reheat`; `~/code/data-d63/probe-d63/`, archived as `option1/probe-d63/`), its D64 twins
   (`option1/probe-d67-d64/`) and the crucible probe's eight (`klip:~/code/crucible-probe/`,
   `stone:~/labour/crucible-probe/`): each transition (clock, temperatures, groups), the state share
   and screen at each; each hold (its start, length, the temperature, whether it recovered); each of
   D63's reheats (how far back it ran: how many transitions it crossed, whether it passed the first,
   and whether the run then recovered or certified). **Read for:** how often a hold or collapse
   follows a stage at which the state share was high (the case the reheat can rescue); the gap
   between the held temperature and the latest transition's (one stage, or many); and whether D63's
   uncapped reheats that crossed the first transition ever recovered (the owner's "bound to fail").
   The reading may revise D5's wait (D5 (c)) and D3's ceiling, before any build. Into this plan's
   findings, and failure-kinds.md's log.
1. **The build** (after paper 1, D9): `--incremental-reheat` in `ContinuousCrystallisation` (the
   stack, the trigger, the ceiling, the pace, the wait, the pop, the end), its state in
   `ContinuousState` and the checkpoint, its log fields; the cell `continuous-ireheat` in
   `batch.CELLS`. Tests: the rule on synthetic screens and groups (each of D1-D7's cases: a hold that
   recovers in the wait, one reheat and recovery, two pops, the empty stack, the flicker not a
   transition, a held temperature released, resume mid-reheat); `continuous` and `continuous-reheat`
   unchanged (their tests); the golden run bit-identical. One PR (tier 1: the flag off changes no
   timed path; tier 0 if the A/B's rule says so).
2. **The probe** (claims nothing): the crucible probe's four S4 seeds (300-303) and D63's
   collapses (S4 0 and 2, D57's), `continuous-ireheat`, 1,600 epochs, on `stone` or `klip`. Read:
   each reheat's depth, whether the state re-formed after it, any certificate. The mechanism works if
   runs recover after a reheat that the held cool lost; it fails if every reheat ends in another
   collapse or the empty stack.
3. **In X (D8):** X's registration names the arm, its claims, seeds and test by X's D16-D21, with the
   other arm's.

## Feasibility

The cell is `continuous`'s with one flag: the same codes, window, exclusion bound and state bound
(the feasibility register's rows unchanged; [information-flow.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/protocols/information-flow.md)).
One new failure it can't escape by the mechanism: a run whose screen falls before its first
transition (S3 24's, D63) has no stage to return to; D7 ends it.

## The size, roughly

Stage 0: one to two hours on `pyramidon` (reading only). Stage 1: about half a day, its tests on
`jebel` in a scratch copy. Stage 2: six runs at 1,600 epochs, about 5-9 hours on one machine
overnight. X: no more runs than its D5 already allows (D8 (a)).
