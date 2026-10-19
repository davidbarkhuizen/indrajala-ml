# The bottlenecks: what stops a network learning a machine (a living map)

**Written 2026-10-19 at the owner's request, after W1 and B2's first runs: "document this
3-bottleneck view ... so we can keep it in mind going forward - and work to open up each
bottleneck".** Every failure seen since the general algebras' round 1 is one of three kinds, each
with its own cause and its own fix. A fix for one isn't expected to move another, so a study is
read against the bottleneck its cells hit. This file maps them to their evidence and plans; the
evidence lives in the probe script's findings (`scripts/modular_family_probe.py`) and the plans
named below. Update a row when a result is read, and date it in the log at the end.

A transducer is two functions of the state: the transition `delta`, the next state, and the output
`lambda`, the answer. The three bottlenecks are the state never forming, the state's transitions
going wrong on codes beyond an exact core, and the answers not being separable on a right state.

| | 1. State formation (failure A) | 2. Extra codes | 3. Answer separation |
| --- | --- | --- | --- |
| **what fails** | no state forms: 2-6 codes a factor, the state's share 0.06-0.35, the machine minimising to 3-6 | a state forms with an exact core (each state's first code breaks 0-4% of its transitions), but longer words reach many more codes, whose transitions break | the machine is exact, but some answers on it are wrong |
| **where seen** | S3 (round 1, 5 of 6 failures); move-to-front 4 (round 1, P1: seeds 0, 3, 4); move-to-front 5 (round 2, 4 of 6; B2's first three seeds) | S4's near miss (round 1); random-4-0, random-q16-0 (2b, A, B, P2); S5 (round 2: 385-748 codes for 120 states, minimised to 175-842; B2's first three seeds: 425-737) | division `p` 13 (L3: seed 5's 169-state machine exact, 15-22 of 1,521 answers wrong) |
| **cause, as far as shown** | the commitment's pull: with nothing yet distinguishing the states, a constant meets it most cheaply. Diagnostic (2026-10-19, registered): with the commitment off for steps 1-5, 3 of 6 failure-A runs formed a state (0.87, 0.97, 1.0) and S3 seed 5 certified; the states eroded when it returned at step 6 | not duplicates: an extra code's outputs are wrong within `d` tokens, so the network's own equivalence can't merge them (A's reading). Not rarity (random-4-0). Where they first arise is unshown | a crowded window: the answers and every factor's transition read one 16-wide stream; its features don't linearly separate the answers (the separability check) |
| **tried** | the commitment off early (the diagnostic: forms states, doesn't keep them) | A, consolidation (0 of 6, pulled little); B, the long tail (0 of 6, a longer exact prefix in 2 seeds); P2, restore the step's best (0 of 6: the best was never a machine); P1, the guard (fewer certified) | readout-tune (15 left: not an under-trained readout); W1, the window 32 wide: **5 of 6 against 0 of 6, cured** |
| **plan** | [commitment-control-workplan.md](commitment-control-workplan.md) (deferred, re-aimed at failure A) | [code-consolidation-workplan.md](code-consolidation-workplan.md) (paused, its remedies spent) | [window-branches-workplan.md](window-branches-workplan.md) (stage 6 running) |
| **next** | write the diagnostic's findings; build the usage thermostat (D1 (a): the commitment off while a factor's code use is low, back on once a state has formed), its probes on S3, move-to-front 4 and 5 | a diagnosis before another remedy: on B2's failures, where each extra code first appears (the step, the transition, the code it leaves), and whether the codes training visits are the ones that break | B1 (branches) against W2 (size) tonight; then division `p` 17 with the window that wins, registered before its data, for whether the cure holds as the machine grows |

## The common measure: one code, one state (2026-10-19)

Every certified network has each of its codes standing for exactly one target state (pairs per
code 1.0, `breaks`); each bottleneck is a way of missing that: no state (1), codes standing for
several states (2), an exact state with wrong answers (3). Read across W1, L3, B1 (mid-run), B2 and
round 2 in [one-code-one-state-workplan.md](one-code-one-state-workplan.md) ("Why"), which plans
the remedies: the answers' own branch with the factors on the shared window (3), pairs per code
logged and used to keep the best machine, counterexample training (2), the thermostat (1).

- **B2 (2026-10-19): the branches moved 1 and 2 in one seed each** (S5 1 of 6, move-to-front-5
  1 of 6, against 0 of 6 each in round 2): works, gain undecided.
- **The branches at width 16 move division from 3 to 2** (B1 mid-run: codes standing for 11-81
  states where L3's stood for one): splitting the factors' path starved their transitions.

## How they combine

- **One cell can hit more than one.** Move-to-front 5's seeds split between 1 and 2 (round 2: four
  failure A, two with states but no exact core). Division `p` 13 at width 16 hit 3 with its state
  exact. So a cell's count says little until each failure is read by D9's probes (`breaks`,
  `by-length`) and assigned.
- **A fix is read on its own bottleneck's cells.** The branches target 3; B2 (S5, move-to-front 5)
  hits 1 and 2, so a 0 of 6 there doesn't count against the branches on division, and isn't
  evidence the branches generalise either.
- **Across all three: retention.** Near solutions are found and lost when trained on (every
  random-q16-0 run; L2's collapses). Stopping at the first certified step keeps a solution once
  reached; it doesn't help a run that only came near. The guard as built freezes near solutions
  (P1); its first softening, arming only on a step that passes both verdicts, is in
  [next-steps.md](next-steps.md).

## Log

- 2026-10-19: written. 3 is cured on division at 13 by width (W1); the branches' answer (B1,
  W2) tonight. 1 has a cause shown (the commitment) and a fix to build. 2 has no remedy that
  works and no shown cause.
- 2026-10-19: the common measure (pairs per code) and its workplan; B2 read (1 of 6 on each
  cell).
