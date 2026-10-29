# Workplan: the general algebras (beyond arithmetic), and the fork's trigger and the guard tuned across them

**Status: written 2026-10-17, at the owner's request: wrap up division, run the ladder, then the
general (non-arithmetic) algebras, and tune the fork's trigger and the certificate guard there,
across genuinely divergent tasks ("fine tune when we can use genuinely divergent tasks"). Takes up
[next-steps.md](../../next-steps.md)'s "Beyond arithmetic" item 1 and the two items deferred to it.
D1-D9 settled under the owner's delegation, each with its options' pros and cons; D2 the owner's
choice. E1 registered
here, before its data; T1 and G2 are registered at their stages, before theirs. Paused since
2026-10-20 after round 2 (stage 4): the owner set all planned work aside for
[state-formation-workplan.md](state-formation-workplan.md), so stages 5-8 (the trigger, the
guard's softenings, G2) wait for a decision to resume or retire (the programme audit, 2026-10-23,
F29).**

## Why

Every target so far is an arithmetic transducer (the carry automaton, modular sums, division), the
weakness `~/code/indrajala-research/contribution.md` names under "Breadth". The question is whether
the method needs that structure. Krohn-Rhodes splits every finite machine into group (permutation)
parts and aperiodic (reset) parts, so the test is a pair of machines at those two extremes with
everything else equal, and a random machine as the typical case between them. The groups include
S5, the non-solvable case that Liu et al. 2023 ("Transformers learn shortcuts to automata") and
Merrill's circuit-complexity results make hard for fixed-depth transformers. A recurrence certified
exact on S5 at every length makes a sharp contrast with that.

The fork's trigger (the p11-trigger workplan's stage 4) and the certificate guard (the guard
workplan's G1, never run) were each deferred until there were tasks unlike division. This plan's
existence runs supply those tasks and their logs.

## Decisions

- **D1. The machines: three classes over one state space, at three sizes.** At size `n` (3, 4, 5)
  the states are the `n!` orderings of `n` items, and each class has `n + 1` tokens, token 0 the
  identity:
  - **group:** the word problem of S_n. The tokens are the identity, the star transpositions
    `(1 i)` for `i = 2..n`, and the `n`-cycle. S3 and S4 are solvable; S5 is not.
  - **aperiodic:** move-to-front over `n` items (an LRU cache's recency order). Token 0 is no
    access; token `i` moves item `i` to the front. Its monoid is a left regular band: every token
    idempotent, no group part.
  - **random:** `n` random maps of the `n!` states plus the identity, drawn from a seeded generator
    and redrawn until every state is reachable from the start. Its monoid is typically mixed (its
    group part reported).

  Checked (scratchpad, 2026-10-17): every state is reachable in every class at every size, within
  2 tokens at `n` 3 and 4-5 at `n` 5; with D2's outputs every machine is minimal (partition
  refinement: 6, 24 and 120 states).
  - (a) *Chosen.* Pros: the group and aperiodic machines share states, tokens and outputs (orderings
    of `n`), so the class is the only change; the sizes 6, 24 and 120 run from solvable to
    non-solvable; 120 states sits beside division's `p` 11 at 121, so the top rung is the
    ladder's best-known cell with only the task changed; LRU is a real-world machine; the identity
    token keeps window_tasks' convention (token 0 leaves the start fixed), so the harness can run
    them too. Cons: one aperiodic family (a band) stands for all aperiodic machines; the identity
    token is one easy token in `n + 1`.
  - (b) Random Mealy machines only, split afterwards by their algebra. Pros: the plain generality
    test. Cons: random machines are almost never pure groups or pure aperiodic, so the split isn't
    controlled; sizes and alphabets don't match across the classes.
  - (c) S3, A5 and S5 against flip-flop registers (`b` set and reset bits, `2^b` states). Pros:
    textbook extremes. Cons: a register is a direct product of 1-bit factors that the network's
    factors can copy, a structural advantage the groups don't have; 128 isn't 120.
- **D2. An output that hides most of the state, `n` classes, the label of the state after the
  token:** for the group, the item at position 1 of the ordering (the running product's image of
  position 1); for move-to-front, the last item, the one an LRU eviction takes; for the random
  machine, a random label in `n` classes per state, redrawn until the machine is minimal.
  - (a) *Chosen (the owner, 2026-10-17).* Pros: the state must be inferred from a few classes, as
    in division (its quotient, 12 classes for 121 states), so the training signal scores the answer,
    never the state (learn the method, not imitate); every machine still minimal at its full size
    (checked). Cons: harder than (b), so a weaker first test of existence; not the literature's
    exact task (Liu et al. output the whole state), so their numbers aren't directly comparable.
  - (b) The output is the state (the running product; the recency order). Pros: the literature's
    word problem; the densest signal. Cons: the state given as the label, close to supervising it;
    easier than division; `n!` classes. After this plan.
  - (c) The group's output: where item 1 sits. Cons: rejected by the check: the machine minimises
    to `n` states (an approximation of the group's action on points), not `n!`.
- **D3. The cell: division `p` 11's, with only the task changed and the codes scaled with the
  states:** the family probe's training, two factors of width 16, reading the output, the fork with
  today's trigger, the guard off, 600 epochs a run, stopping at certified, `k` 1. Each factor gets
  `ceil(32 sqrt(q) / 11)` codes, the ladder's D2 rule (each factor's codes scaled with `sqrt(q)`,
  which is `p` for division).
  - (a) *Chosen.* Pros: the cell that certified 121 states (4 of 6), unchanged at 120; the same rule
    as the ladder, so the two studies read together. Cons: tuned on division; a class that fails may
    fail on the settings (Known limits).
  - (b) One factor of `m` codes. Pros: no factor structure assumed for machines that have no
    product decomposition. Cons: against the factored-state results (factored-not-flat); a new cell
    whose failure couldn't be told apart from the cell's.
- **D4. Six seeds a cell (0-5). A cell "works" if at least one network certifies in both verdicts
  and "fails" at 0 of 6**, as the ladder's D3. Pros: 0 of 6 against `p` 11's 4 of 6 is Fisher
  one-sided p 0.030, the fewest runs for a significant failure. Cons: rates between classes aren't
  told apart at 6 (4 of 6 against 1 of 6 is p 0.12): the classes are compared as a screen, and the
  rates come only after existence.
- **D5. The rungs in order, smallest first:** round 1 is `n` 3 and 4 together (36 runs, minutes
  each), round 2 is `n` 5 (18 runs). Round 2 runs only once round 1 has a working cell in every
  class; a class failing at `n` 3 or 4 gets its failures read (D9) before anything at `n` 5.
  **Owner, 2026-10-18: round 2 is deferred, not dropped, until the random class passes round 1**
  (a working cell at `n` 3 and 4): its failures are read first (D9), and a fix is registered as a
  new experiment (round 1's settings stay as registered). Stages 5-7 replay round 2's `n` 5 cells,
  so they wait with it. The random class's failures and their fix: `git show df1dcac:docs/code-consolidation-workplan.md`.
  **Owner, 2026-10-18, registered here before round 2's data: the random class is set aside, and
  round 2 goes on without it.** Neither remedy rescued a random machine at 16 states (stage 2b of
  the consolidation plan: the control, B and A each 0 of 6 on random-q16-0, every run's near
  solution lost after its best step). The recorded limit: at round 1's cell the method certifies
  random machines to about 12 states (random-q8-1 3 of 3, random-q12-0 2 of 3); near solutions
  aren't kept above that, and it isn't capacity. So this condition lifts. Round 2 is the group
  and aperiodic classes at `n` 5 (word-problem-s5, move-to-front-5; 12 runs), round 1's cell
  unchanged (no remedy added), and E1 is read over those two classes (below). Random machines
  return only by a later decision of the owner's, for instance if the consolidation plan's P2
  (the step's best restored) certifies random-q16-0.
  - (a) *Chosen.* Pros: experiment-design.md section 2, a small case working before stepping up;
    the small rungs cost little. Cons: one round more than running all three sizes at once.
  - (b) Every size in one round. Pros: one launch. Cons: 120-state runs spent on a class that may not
    work at 6 states.
- **D6. One generic probe, not a second one:** the family probe's run, screen, diagnosis, fork and
  guard taken over an encoding protocol (width, tokens, classes, example, answer kinds,
  coordinates). Division's `Encoding` becomes one implementation and `MachineEncoding` (any
  `window_tasks.Mealy`, `k` 1) another; `window_tasks` gains a `MachineTask` for the certificate,
  named `word-problem-s<n>`, `move-to-front-<n>` and `random-<n>-<seed>`.
  A machine's training words end with the identity token, as division's end with a zero column, so
  even one token trains the state (the build's probe: without it 5 of 6 runs kept one code).
  - (a) *Chosen.* Pros: the fork, the trigger, `drops`, the guard and `guard-epoch` run unchanged
    on every class, which is what tuning them across classes needs; division's runs stay
    reproducible (its encoding's examples unchanged, tested); new classes later (Tomita, UTF-8,
    the affine map) cost one machine each. Cons: a refactor of `modular_family.py` (842 lines)
    before the first run.
  - (b) A separate probe for machines. Pros: division untouched. Cons: two copies of the fork and
    guard loops, tuned apart, which defeats the point of tuning them across classes.
- **D7. Answer kinds by input token:** a screen's kinds (the trigger's mean and worst signals) are
  each token's share of right answers. Pros: generic (any machine has tokens); `n + 1` kinds, few
  enough to be stable; a collapse on one generator shows. Cons: not division's kinds (its quotient
  classes and queries), so division keeps its own and the trigger's signals mean slightly different
  things on the two (T1 scores both).
- **D8. Where:** round 1 on `pyramidon` (small, while `jebel` finishes the ladder), round 2 on
  `jebel` under the machine lock; recertification on `pyramidon`. Pros: no waiting on the ladder.
  Cons: round 1's epoch times aren't comparable to `jebel`'s (never claimed).
- **D9. Diagnosis per class before any change:** a failing cell is read before the next step
  (experiment-design.md section 4): where the runs break, the certificates' counterexamples, the
  minimised states, the codes reached, and the factors' readouts against each machine's
  coordinates. The coordinates are each position's item for every class, and the sign for the
  groups.

## The claim (registered before the data)

**E1, per cell (class x size):** **works** if at least one of seeds 0-5 certifies (both verdicts:
the default snap ratio and 1.0) against its machine; **fails** at 0 of 6 (Fisher one-sided p 0.030
against `p` 11's 4 of 6). A working cell with 1-3 of 6 is also labelled **a decline, undecided**.

The method **generalises past arithmetic at 120 states** if all three classes work at `n` 5. It's
**refuted for a class** if that class fails at a size where the other two work. Then the class's
Krohn-Rhodes type, group or aperiodic, is the finding, read with D9's diagnosis.

**With the random class set aside (D5, the owner, 2026-10-18; registered before round 2's data):**
round 1's reading stands (E1 refuted for the random class at `n` 3 and 4). At `n` 5 the method
**generalises past arithmetic at 120 states for both Krohn-Rhodes extremes** if word-problem-s5 and
move-to-front-5 both work. It's **refuted for a class** if that class fails where the other works,
and the class's type is then the finding, read with D9. If both fail, the finding is a failure at
120 states for both extremes, not for a class, read with D9 against division `p` 11's 4 of 6 at
121.

Reported per run, as the ladder reports it: certified or not, step, epochs, forks, minimised states,
codes reached per factor, each factor's readout. Per cell: the count with its Wilson interval, and
the slowest success's epochs against the 600-epoch budget.

## Feasibility (experiment-design.md, section 1)

Two factors of width 16, each `ceil(32 sqrt(q) / 11)` codes; the cap is the exclusion bound
`m + q - 1`, `m` the joint codes, `q` the minimal states. Every machine's states are reachable and
minimal under D2's outputs (checked; tested in the build against the config).

| `n` | minimal states `q` | tokens | classes | codes a factor | joint codes `m` | `m / q` | exclusion bound = cap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 6 | 4 | 3 | 8 | 64 | 10.7 | 69 |
| 4 | 24 | 5 | 4 | 15 | 225 | 9.4 | 248 |
| 5 | 120 | 6 | 5 | 32 | 1,024 | 8.5 | 1,143 |
| division `p` 11 (for comparison) | 121 | 9 | 12 | 32 | 1,024 | 8.5 | 1,144 |

Every success is certified, so it's exact at every length. No run reaches its cap in 600 epochs
(`p` 11 certified at steps 10-31).

## Known limits

- One cell, tuned on division: a failure is this cell's within 600 epochs, and D9 reads it before
  anything is re-tuned.
- One aperiodic family (move-to-front) and one random machine a size.
- One output a machine (D2): the state as the label (D2b) and other outputs come after.

## Stages

1. **This plan.**
2. **The build**, one PR, tier 0 (the probe is never timed; no golden run, since the package's
   training path is unchanged): D6's encoding protocol and `MachineEncoding`, the three machines
   and `MachineTask`, D7's kinds, `run --task NAME` beside `--p` (several tasks in one pool, into
   `DIR/<task>/seed<s>`), and the readout over tasks. Tests: division's examples, screens and
   kinds unchanged; each machine's states, reachability, minimality and tokens; the feasibility
   table against the config. A probe of a few epochs per size on `pyramidon` measures the epoch
   time and memory for round 2's budget.
3. **Round 1** (`n` 3 and 4, three classes, 36 runs) on `pyramidon`; certified in both verdicts;
   E1's readout per cell.
4. **Round 2** (`n` 5) on `jebel`, from a worktree pinned at `main`: 12 runs, word-problem-s5 and
   move-to-front-5, seeds 0-5 (D5, the owner, 2026-10-18: the random class set aside). Budget from
   `p` 11's runs (35-120 minutes each at 6 at once) and stage 2's probe: 12 runs at 6 workers is
   two waves, about 1.5-4 h. Long tests (uniform words at 128, 256 and 1,024) on one certified network a
   cell (the ladder's D6). Findings in the probe's docstring; the networks to indrajala-models.
   **Round 2's result (2026-10-19; the runs in the probe's docstring): 0 of 12, so both extremes
   fail at 120 states (E1's third answer), not one class.** word-problem-s5 0 of 6, move-to-front-5
   0 of 6 (Wilson 95%: 0.00-0.39 each), against division `p` 11's 4 of 6 at 121 states, the same
   cell. Every run spent its 600 epochs (43-92 minutes). There are no long tests, since no network
   certified. **D9 found two mechanisms, both round 1's:**
   - **Extra codes (S4's near miss, at scale):** all six S5 runs and move-to-front-5's seeds 0 and
     2 formed a state (share 0.88-1.0) but minimise to 175-842 states against 120. S5 has an exact
     core at `d` 2: the 120 states' first codes break 0-4%. Its extra codes break 1-17% from right
     codes, and 83-100% from every code reached (385-748 reached). Seed 5's first codes break 0 of
     720, with exact accuracy 1.0 at 24 and 0.999 at 48. Move-to-front-5 has no exact core at `d` 4:
     only 40-69 of the reached codes are right, and the first codes break 39-49%.
   - **Failure A:** move-to-front-5's seeds 1, 3, 4 and 5 never formed a state (2-3 codes a factor,
     share 0.21-0.24, minimised to 3).
   So the state count alone isn't the limit: division certifies at 121 in the same cell. (Confounded,
   the programme audit's F16: division trains on its two approximations' outputs beside the
   quotient, the algebras on one output that hides most of the state.)

5. **The fork's trigger, tuned across the classes (T1, registered in this stage's PR before its
   replays):** `drops` over every log from rounds 1-2 and division's (the fork probe, `p` 11, the
   ladder), with candidate triggers chosen by false alarms and drops caught across all of them
   together, not per class. The test, as the p11-trigger workplan set it: paired replays (the same
   seeds, only the trigger changed) of the `n` 5 cells and division `p` 11. **Better** if it
   certifies more seeds, or as many in fewer epochs, losing none that today's trigger certified.
   Better becomes the default; otherwise today's stays and the finding is recorded.
6. **The guard's cause probe and softenings (probes, no claim):** on certified networks from every
   class, whether the flips the guard undoes land on rarely visited codes; then `guard-epoch`
   comparing the softenings next-steps lists (a flip judged only after N batches; a flipping update
   scaled down instead of undone; the last right machine restored once an epoch; only updates that
   add wrong pairs undone; only visited states guarded; the optimizer calmed once right). Measured:
   whether the machine stays right, and the share of updates the guard touches.
7. **G2, registered in stage 6's findings PR before its data:** the fork alone against the fork plus
   the best softening from stage 6, paired, at the `n` 5 cells and division `p` 11. Its claim and
   refutation are written then, from stage 6's numbers.
8. **Retire:** findings in the probe's docstring and P2's notes, the leftovers (D2b's whole-state outputs;
   the Tomita grammars, parity and Dyck; real-world transducers; A5; holonomy-coordinate queries
   such as S5's sign, the cover claim for groups) to next-steps.md; this plan deleted.
