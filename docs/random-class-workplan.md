# Workplan: the random class's failures, A's fix (E1b) and B's origins

**Status: written 2026-10-18 at the owner's requests, after round 1 of the general algebras
(docs/general-algebras-workplan.md, E1) failed the random class at `n` 3 and 4 and the owner
deferred round 2 until it passes (that plan's D5): "keep two factors; register the fix
experiment", then "probe more for B's origins before we attempt to test a fix, once we are done
with A". D1-D6 settled under the owner's delegation, each with its options' pros and cons. E1b
registered here, before its data; B's fix is registered only after its probes.**

## Why

Round 1 (E1, the general-algebras plan's cell D3, seeds 0-5): move-to-front 6 of 6 and 4 of 6, the
word problem 1 of 6 (S3) and 4 of 6 (S4), random-3-0 0 of 5 ended (seed 4 still running, stuck as
the rest), random-4-0 0 of 6. D9's reading (a scratch reader of every run's best-share step network
and its last network; it becomes code-consolidation stage 2's `breaks`) found three failures:

- **A. The state never forms (`n` 3):** random-3-0's six runs hold one code from the first epoch,
  every accuracy frozen to three decimals for 600 epochs; word-problem-s3's five failures peak at a
  state share of 0.06-0.32, at most 6 codes on 2-6 of 6 states, and end on one code;
  move-to-front-4 seed 3 the same at `n` 4. One code answers every length-1 word from the start
  state, so step 1 (lengths 1) passes without a state, in every class.
- **B. A wide state that never closes (random at `n` 4):** random-4-0's best networks reach all 24
  states with 54-67 codes (minimising to 49-64), right through depth 1-2 and breaking from depth
  2-3, where every state has been reached and the transitions must return to codes already there;
  4 of 6 then erode to 2-3 codes. The breaks don't follow the rarely visited states (their share
  of wrong transitions about their share of codes) or the merging transitions (less often than
  their share). No run passes a step after 3-4, against the designed machines' failures, which
  break at depth 2-5 at a state share of 0.90-1.00.
- **C. Late breaks on the designed machines (`n` 4):** word-problem-s4 seeds 0 and 5,
  move-to-front-4 seed 5: division's kind, for the fork and guard (the general-algebras plan's
  stages 5-7).

Random-3-0 is also an unfair draw: the best stateless guess (each token's majority label over the
stationary states) is right 0.720 of the time against chance 0.333, 18th of the first 20 draws at
`n` 3 (median 0.589; draw 4 is 1.000: no state needed at all). The generator checks reachability
and minimality, not that the state matters. Random-4-0 is typical (0.422, 13th of 20).

No factor of any network, the certified included, is a function of the target state, and no
partition of a network that reaches every state is named by a coordinate form. The readout names
trivial partitions (one block, or a few states reached) as coordinates: a measurement bug, fixed in
code-consolidation stage 2.

## Decisions

- **D1. A, then B (the owner, 2026-10-18):** A's fix is registered now (E1b, at `n` 3, where A
  is); B is probed for its origins first, and its fix registered from what they find.
  - (a) *Chosen.* Pros: A has a diagnosed mechanism (a first step passable without a state) and a
    direct fix; B has none, and a fix tested before its cause is known likely spends a study on a
    guess (experiment-design.md, section 4, diagnose before redesigning). Cons: the random class
    can't pass round 1 (it needs `n` 4) until B's fix, so round 2 waits longer.
  - (b) A and B fixed together, from probes of candidates. Cons: B's candidates are guesses; the
    owner's call.
- **D2. A's fix: the curriculum from length 2:** step `j` trains lengths `1..j+1`, so the first step
  can't be passed without a state. The cap is unchanged (the longest length still the exclusion
  bound).
  - (a) *Chosen.* Pros: aimed at A's mechanism; one flag; the answers alone train it (no state
    given). Cons: the run's every step longer by one, so slightly slower epochs.
  - (b) Larger codebooks at `n` 3. Cons: A isn't capacity (64 joint codes for 6 states; one code
    used); move-to-front-3 certifies 6 of 6 with the same codebooks.
  - (c) Holding a step until it passes. Cons: every certified run of round 1 plateaued at some
    steps and moved on before certifying (move-to-front-4 seed 4 at 17 steps).
  - (d) A third factor. Cons: the owner kept two (2026-10-18); not capacity, as (b).
- **D3. E1b's machines: the word problem and move-to-front as in E1, and two new random draws,
  each one whose stateless guess is at most halfway from chance to 1:** at most
  `1/n + (1 - 1/n) / 2` (0.667 at `n` 3), the first two seeds from 1 that pass: `random-3-1`
  (0.568) and `random-3-2` (0.545). The generator is unchanged (round 1's names keep their
  machines); `window_tasks` gains the stateless guess's accuracy, and a test checks the draws
  against the bound.
  - (a) *Chosen.* Pros: the state must carry at least half of what a stateless guess leaves, so the
    class is tested on machines that need memory; two draws tell the class from a draw (E1's "one
    random machine a size" limit). Cons: the machines differ from E1's, so a random success joins
    the change and the draws (D5).
  - (b) Random-3-0 again. Pros: the change is the only difference from E1. Cons: it needs little
    state, a test of the generator, not the method.
  - (c) A filtered generator in place of today's. Cons: renames round 1's machines.
- **D4. E1b at `n` 3 only:** the four machines, seeds 0-5, 24 runs. Pros: A is at `n` 3 (one of
  `n` 4's 18 runs); short. Cons: `n` 4 at the changed cell (move-to-front-4 seed 3's A among them)
  waits for B's fix, which runs at that cell.
- **D5. No control arm (E1's cell on the new draws), but the designed machines are E1's own:**
  their change is the only difference from E1. Pros: half the random runs. Cons: a random success
  isn't split between the change and the draws.
- **D6. Where:** E1b on `jebel` (idle, 6 workers), recertified on `pyramidon`; B's probes on
  `jebel` after it. Pros: `pyramidon` swaps at 6 workers; `stone` keeps L2. Cons: none known.

## The claim (registered before the data)

**E1b, per cell (machine), the curriculum from length 2 at `n` 3:** **works** if at least one of
seeds 0-5 certifies (both verdicts); **fails** at 0 of 6, as E1. A working cell with 1-3 of 6 is
also labelled **a decline, undecided**.

- **A is fixed at `n` 3** if both random draws work and no run of the four cells holds one code
  through its first five steps (A's signature: one code per factor at every diagnosis of steps
  1-5, in a run not certified by then). It's **refuted** if a random draw fails with that
  signature in at least half its runs.
- **The change keeps the designed classes** if move-to-front-3 works. **It helps the word problem**
  if word-problem-s3 certifies more than E1's 1 of 6: Fisher one-sided against 1 of 6, a screen
  (4 of 6 against 1 of 6 is p 0.12, the general-algebras plan's D4), reported as the count.

Reported per run as E1 (certified or not, step, epochs, forks, minimised states, codes per factor,
each factor's readout); per cell the count with its Wilson interval and the slowest success's
epochs against 600.

## Feasibility (experiment-design.md, section 1)

Two factors of width 16, each `ceil(32 sqrt(q) / 11)` codes; the cap is the exclusion bound
`m + q - 1`. Every machine is reachable and minimal (the generator's checks and round 1's build
test); D3's draws are within the stateless bound (tested).

| `n` | minimal states `q` | tokens | classes | codes a factor | joint codes `m` | `m / q` | exclusion bound = cap |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 3 | 6 | 4 | 3 | 8 | 64 | 10.7 | 69 |

The curriculum from length 2 ends a step early: step `j` trains to `j + 1`, its last step the cap
minus 1, so the longest length is still the bound.

## B's origins (probes, no claim; after E1b)

Short probes on `jebel`, each written with what would answer it before its data, chosen from each
other's results, kept as probe subcommands with their data's manifests. The first questions:

- **Structure or merging?** Random permutations (each token a bijection: a random group action on
  24 states) against random maps: B on both, or only on maps that merge states. Merging was
  ruled out as where the breaks land, not as what the class needs.
- **Where in size?** Random machines between 6 and 24 states (8, 12, 16): the smallest `q` where
  B appears, against A's machines at 6.
- **The breaks' anatomy** (stage 2's `breaks`, extended): at depth 2-3, does a wrong transition go
  to a code of a state reached by a similar word (the tree read as a trie), or anywhere?
- **The step's own reading:** what the window learns of a random map with the state held fixed in
  the probe (a capacity read of the window, never a training method), against the designed maps.
- **The rate and the width** (half the rate; factor width 32) as readings of B's erosion and of the
  window's room, not as fixes.

B's fix is registered (claim, refutation, cells) from what these find, then the `n` 4 cells at the
changed cell (round 1's `n` 4 again, the random class on D3's draws at `n` 4: `random-4-1` 0.431,
`random-4-2` 0.379, under 0.625).

## Known limits

- A random success is the change's and the draws' together (D5).
- Two random draws, filtered by D3's bound: the class's typical machine, not its hardest.
- `n` 3 only: whether the change keeps `n` 4's designed cells is read when B's fix runs there.

## Stages

1. **This plan.**
2. **The build**, one PR, tier 0 (the probe is never timed, the package's training path unchanged):
   `run --first-length L` (step `j` trains `1..j+L-1`, default 1); `window_tasks.stateless_accuracy`;
   tests: the default trains as today (a machine's and division's first epoch unchanged), D3's
   draws against the bound, E1b's feasibility table against the config.
3. **E1b** on `jebel`, 24 runs at 6 workers, from a worktree pinned at the build's merge, holding
   the machine lock: `--task word-problem-s3 move-to-front-3 random-3-1 random-3-2 --first-length 2`,
   E1's settings otherwise. Budget from round 1's `n` 3 runs on `jebel` (successes minutes,
   failures 80 min at 600 epochs): 4 waves, at most about 5 h, about 1-2 h if A is fixed.
   Recertified on `pyramidon` in both verdicts, readout, manifest, findings in the probe's
   docstring.
4. **B's probes** (above), then B's fix registered here (its claim, cells and feasibility).
5. **B's fix and `n` 4**, then the general-algebras plan's D5 updated (round 2's cell); findings in
   `modular_family_probe.py`'s docstring; this plan retired.
