# Workplan: one code a state (consolidating codes, and longer training words)

**Status: drafted 2026-10-18 at the owner's request ("we need to test both options"), from the D9
diagnosis of general-algebras round 1 (below). D1-D8 settled under the owner's delegation, each with
its options' pros and cons; the owner confirmed D2 and made a search of B's settings an optional,
later follow-up (D4; 2026-10-18). C1-C3 registered here, before their data.**

## Why

D9 on round 1's failures ([general-algebras-workplan.md](general-algebras-workplan.md), the
handover of 2026-10-18; the probes `by-length` and `breaks`, stage 2) found one mechanism in two
cells:

- **S4, seed 0 (the near miss), step 13:** an exact 24-code core. Each state's first code (the one
  its shortest word reaches) breaks 0 of its 120 transitions. But longer words land on 58 more codes
  (104 reached, the network's machine minimising to 97), whose transitions break 22%. Exact to
  length 6, then about 0.3% a token: 0.90 at 24, 0.71 at 60. Training past step 13 eroded it.
- **random-4-0, seeds 0-3, best steps 9-11:** the same, worse. First codes break 7-15%, the extra
  codes 35-44%. Exact to length 3, then 0.6 at 11 and 0.44 at 40. Not rarity: rare and common states
  break alike. Every pair of states differs at one token, so the answers already say all there is
  about the state, and coordinate queries add nothing.

Extra codes aren't the fault: certified networks reach 38-71 codes and minimise to 24. The fault is
that a state reached by a long word lands on a code whose transitions were never made right. Two
remedies follow:

- **A, consolidate:** codes the network itself can't tell apart are pulled onto one code.
- **B, longer words:** extra codes get visited, and corrected, more often. Round 1 also trained far
  below its exclusion bound: the cap at S4 is 248, and runs stopped at steps 9-30. Below the bound,
  a wrong machine can fit every training word (experiment-design.md, section 1). This is E8's
  lesson, met again through the curriculum's pace rather than its cap.

A third finding goes into every arm: **random-3-0's step 1 passes with no state.** Every one-token
word answers label 2, so a one-code network scores 1.0, and every run stays on one code (4 of 4,
frozen for 56 steps). The final identity token can't help when the one-token answers agree (D3).

A second reading (2026-10-18, every run of round 1 including `jebel`'s seed 5; a scratch reader of
each run's best-share step network and its last network) adds three findings:

- **S3's failures are a state that never forms, and step 1 isn't why:** word-problem-s3's five
  failures peak at a state share of 0.06-0.32, at most 6 codes on 2-6 of 6 states, and end on one
  code; move-to-front-4 seed 3 the same. Step 1 needs a state on every round-1 machine but
  random-3-0 (the final identity's answer depends on the token before it), so D3 doesn't touch
  them: undiagnosed, read again in stage 2.
- **A random draw can need little or no state:** the best stateless guess (each token's majority
  label over the states' limiting distribution) is right 0.720 of the time on random-3-0 against
  chance 0.333, 18th of the first 20 draws at `n` 3 (median 0.589; draw 4 is 1.000). Random-4-0 is
  typical (0.422, 13th of 20). The generator checks reachability and minimality only; D3 handles
  the first step's trap, not a draw that needs little state overall.
- **The readout names trivial partitions as coordinates:** one block is named "position-1 + ... +
  position-n" (a permutation's positions always sum to the same value), and partitions over a few
  reached states get names too. Fixed in stage 2.

## Decisions

- **D1. A's signal comes from the network's own behaviour, never the target's states.** At each
  diagnosis (every 5 epochs) the network's machine is walked as the certificate walks it, but with no
  target: its reachable joint codes, breadth first, each with its outputs. Codes are grouped by
  `d`-equivalence: no word of `d` tokens or fewer tells them apart by the network's outputs (Moore's
  `k`-equivalence). A group's representative is its code reached by the shortest word. The commitment
  term then pulls a duplicate's cell output `h` to its representative's code instead of to its own
  nearest code: `beta ||h - sg[e_rep]||^2`, per factor, on the representative's factor codes. The
  forward snap is unchanged (nearest code), so codes merge only through training.
  - (a) *Chosen: the commitment's target re-pointed.* Pros: no new loss weight (`beta` as it is,
    0.25); the network's architecture and the certificate unchanged, so a certified network is a
    trained network, nothing edited; an empty table is today's training exactly (golden run). Cons:
    a crate change (the commitment target per window), the parity tests in the Python reference
    network too; a group built from wrong outputs pulls toward a wrong code (the fork catches the
    collapse that follows).
  - (b) A hard alias at the snap (duplicates snapped to their representative). Pros: immediate;
    exact once the groups are right. Cons: the alias table becomes part of the network, so the
    certificate must apply it (a second machine semantics); a wrong merge is instantly wrong
    everywhere; close to "rerouting by hand", which the diagnosis ruled out as evidence.
  - (c) A separate consolidation loss with its own weight. Pros: tunable. Cons: one more setting to
    tune, and tuning across tasks is what this project avoids before existence.
  - (d) Groups from the target's answers (words whose continuations agree share a state). Rejected:
    that tells the network the target's states (learn the method, not imitate).
- **D2. `d` = the target's distinguishing length + 1** (Chow's `k`-distinguishability, computed by
  `Mealy.distinguishing_length`; 1 for S4 and random-4-0, so `d` 2; 2 for random-3-0, so `d` 3).
  Pros: just enough to split every pair of the target's states when the network's outputs are right,
  plus a margin against a one-token coincidence; a property of the task, like its window, not of
  any state. Cons: uses one number from the target. *(a), the per-task `d`, chosen; confirmed by the
  owner (2026-10-18).*
  - (b) *Rejected:* a fixed `d` (2) for every task. Cons: too small for targets that need longer
    words to tell states apart (division's), so it silently merges their states.
- **D3. Every arm starts the curriculum where a stateless network can't pass.** Step 1 trains
  lengths `1..s0`, where `s0` is the least length at which the answers aren't a function of the token
  alone (computed from the task's answers, exhaustively at these sizes). The other steps follow as
  before. `s0` is 1 for every machine in round 1 except random-3-0 (2), so this changes only
  random-3-0's runs (tested). Pros: generic; fixes the flaw for any machine without redrawing it.
  Cons: a change from round 1's cell, so random-3-0 is compared only within this plan. (b) Redraw
  random-3-0 until its one-token answers differ. Cons: changes the machine, and leaves the trap for
  the next one.
- **D4. B: half of each epoch's examples at lengths `1..j` as now, half uniform over `j+1..4j`**
  (capped at the exclusion bound), with the same examples an epoch. Pros: the curriculum's short
  words stay (the early steps that built the cores); the long half visits the extra codes; one
  change (the length distribution). Cons: an epoch costs about twice the tokens (mean length 1.25
  `j` against 0.5 `j`), so wall-clock per epoch doubles and the minutes budget is set from stage
  4's probe; the factor 4 and the half are untested choices.
  - (b) A faster ramp: step `j` trains `1..4j`. Pros: one number. Cons: drops the short-word
    phase whose cores were right; step 1 already at 4 tokens.
  - (c) The same tokens an epoch (fewer examples). Pros: equal compute. Cons: two things change
    (lengths and examples), and fewer short words.

  **B's settings space afterwards, optional (the owner, 2026-10-18):** the factor (4), the long
  share (a half) and the schedule ((a)-(c)) are searched only after this plan's round, and only if
  it's worth it: a later, separately registered experiment, not a stage here.
- **D5. The arms: control, A, B, each against the same control rerun on this plan's commit.** Pros:
  one change between each arm and the control (with D3 in all three), on the same code. Cons:
  re-running a control costs a third of the runs. A+B only if both help (D8).
- **D6. The cells: S4 and random-4-0 (the diagnosed mechanism) and random-3-0 (its class, with D3).
  Division `p` 17 and S3 join by a rule registered here:** each joins only if stage 2's D9 finds the
  same signature (its best network's first codes break at most half as often as its extra codes,
  and exact accuracy falls with length after a perfect start). Pros: no cell without a diagnosis
  (experiment-design.md, section 4). Cons: a conditional join means the plan's size is known only
  after stage 2.
- **D7. Six seeds a cell (0-5), as E1 and the ladder.** Pros: 0 of 6 against 4 of 6 is Fisher
  one-sided p 0.030, the fewest runs for existence and a clear gain. Cons: S4's control rate (3 of 4
  finished in round 1) leaves no room to show a gain there at 6 seeds; S4 reads as "no harm" plus
  the mechanism (C3).
- **D8. Where and when:** `stone` (weekdays 18:00-07:00, weekends; studies only, the sandbox rules)
  and `jebel`, an arm a machine, each run whole on one machine (experiment-design.md, section 5).
  The round must fit `stone`'s window or be split by whole runs. If A and B both help (C1 on any
  cell), a follow-up A+B is registered then, as its own experiment.

## The claims (registered before the data)

- **C1, per cell and arm (A, B):** the arm **works** on a cell if at least one of seeds 0-5
  certifies (both verdicts). It **helps** where the control fails (0 of 6) and the arm certifies 4
  or more of 6 (Fisher one-sided p 0.030); 1-3 of 6 against 0 is "works, gain undecided". An arm
  **fails** on a cell at 0 of 6.
- **C2, the mechanism (each arm's failures and its certified networks, by `by-length` and
  `breaks`):** A should leave fewer extra codes, or extra codes breaking less often, than the
  control at the same step. B should lengthen the exact prefix (the longest `L` with exact accuracy
  1.0). A run where an arm certifies but C2 doesn't move is reported as such: helped, not through the
  mechanism.
- **C3, no harm:** an arm is **harmful** on S4 if it certifies 0 of 6 where the control certifies 3
  or more of 6 (Fisher one-sided p 0.091, a screen; reported as a decline, undecided, unless 0 of 6
  against 4 or more).

Reported per run, as round 1: certified or not, step, epochs, forks, minimised states, codes reached
per factor, worst answer at its best (three decimals); for A, the groups' count and the duplicates
pulled at each diagnosis. Per cell: the count with its Wilson interval and the slowest success's
epochs.

## Feasibility (experiment-design.md, section 1)

Round 1's cell (two factors of width 16, `ceil(32 sqrt(q) / 11)` codes each) with D3, D4 and A
added; the cap is the exclusion bound `m + q - 1`. B's long half never passes the cap.

| cell | minimal states `q` | codes a factor | joint codes `m` | `m / q` | exclusion bound = cap | `s0` (D3) | `d` (D2) |
| --- | --- | --- | --- | --- | --- | --- | --- |
| random-3-0 | 6 | 8 | 64 | 10.7 | 69 | 2 | 3 |
| word-problem-s4 | 24 | 15 | 225 | 9.4 | 248 | 1 | 2 |
| random-4-0 | 24 | 15 | 225 | 9.4 | 248 | 1 | 2 |

Every success is certified, so it's exact at every length. A run reaching step 62 trains B's long
half to the cap (4 x 62 = 248); round 1's runs stopped by step 30.

## Known limits

- One cell, tuned on division: a failure is this cell's within its budget, read by D9 before any
  change.
- A rests on the network's outputs being right at the representatives. Where they aren't, A pulls
  toward a wrong code: C2 shows it.
- B's factor 4 and its half are untested choices; a B that fails isn't evidence against longer
  words in general, only against this schedule. Its settings space is the optional follow-up
  (D4).

## Stages

1. **This plan.**
2. **The D9 probes as tools, and D6's diagnosis:** `by-length` (exact accuracy over all words of
   each length, and how often the code names the state) and `breaks` (which transitions break, from
   right codes, by first and extra codes, by depth) as `modular_family_probe.py` subcommands. They
   generalise "a right code" from one token's outputs to D2's `d`-token outputs, so division
   qualifies. The readout's naming fixed (no name for one block or a few states). One PR, tier 0
   (#829, #830). Then `p` 17's and S3's best networks read. D6's rule decides whether each joins,
   recorded in this plan before stage 3.

   **Stage 2's result (2026-10-18; each failing run's best step network, the worst answer's
   peak):** neither joins. S3's five failures have no right code at D2's `d` 2 at all (2-10 codes,
   exact accuracy 0.81 at length 2, 0.1-0.4 at 4): failure A, a state that never forms, not this
   plan's mechanism. `p` 17's six (523-1,121 codes) break from right first codes 72-88% of the time
   (`d` 1), exact to length 2, 0.85-0.99 at 3, 0.36-0.63 at 4: no exact core. **And random-4-0's
   core depends on `d`:** at `d` 1 its first codes break 4-14% and its extra codes 34-38% (the
   diagnosis above), but at D2's `d` 2 only 15-23 of its 24 first codes are right, and those break
   17-31%, about as often as the extra codes: no exact core at the plan's own `d`, unlike S4 (1 of
   120 at `d` 2). Random-4-0 stays a cell (D6 names it), read with this.
2b. **B's origins, probes before the round (the owner, 2026-10-18: "probe more for B's origins
   before we attempt to test a fix").** Probes claim nothing; each question's answer is written
   before its data. Round 1's cell (D3 of the general-algebras plan, today's curriculum, 600 epochs),
   seeds 0-2, on `jebel` (12 runs, 6 at once, about 1.5-2 h), from a worktree pinned at the
   machines' merge; read with `breaks` and `by-length` at `d` 1 and 2. Baseline: random-4-0, 0 of 6
   in round 1, B's signature (every state reached, breaks from depth 2-3, no exact core at `d` 2).
   - **Merging or structure?** `random-perm-4-0`: random-4's machine with every token a bijection
     (24 states, 5 tokens, 4 classes; stateless guess 0.333). A certified seed, or failures with an
     exact core at `d` 2 (S4's kind), say B needs maps that merge states; B's signature in all
     three says it doesn't.
   - **Where in size?** `random-q8-1`, `random-q12-0`, `random-q16-0`: random-4's tokens and
     classes at 8, 12 and 16 states, each the first draw whose stateless guess is at most halfway
     from chance to 1 (0.399, 0.520, 0.544; at most 0.625). The smallest size failing 0 of 3 with
     B's signature is where B appears; if all three work, B appears between 16 and 24.

   | probe | states `q` | codes a factor | joint codes `m` | `m / q` | exclusion bound = cap |
   | --- | --- | --- | --- | --- | --- |
   | random-q8-1 | 8 | 9 | 81 | 10.1 | 88 |
   | random-q12-0 | 12 | 11 | 121 | 10.1 | 132 |
   | random-q16-0 | 16 | 12 | 144 | 9.0 | 159 |
   | random-perm-4-0 | 24 | 15 | 225 | 9.4 | 248 |

   Findings in the probe's docstring with the data's manifest; the next probe chosen from them.
3. **The build.** The crate first (indrajala-math-rust): the commitment's target per window,
   defaulting to the nearest code, with parity tests against the Python reference. The golden run is
   bit-identical (an empty table is today's training); tier 1 A/B on the training path. Then the
   "Bump rust/" PR and the probe's PR (tier 0): D3's `s0` (and a test that it's 1 for every round-1
   machine but random-3-0), D4's lengths (`--long-tail 4`), A's groups and pull (`--consolidate`),
   the per-diagnosis log of groups and pulls, the feasibility table against the config.

   **`run --resume` (the owner, 2026-10-18), its own PR in this stage, tier 0:** a run that is
   stopped goes on from its last epoch, so `stone`'s runs can pause at 07:00 and go on at 18:00.
   Each epoch's end leaves a checkpoint beside the run: the network with its optimizer's state
   (Adam's moments and step count) and its generator's, the loop's place and counters (step,
   epoch, epochs spent, the step's best and patience, the fork's rate, count, peak and snapshot,
   the guard's state, A's groups) and the minutes trained, so the wall-clock budget counts
   training only. `run --resume DIR` reads it and trains on. The test: a run stopped at an epoch
   and resumed against the same run uninterrupted, every network and log row but the seconds the
   same by bits, on a small cell in the suite and once on round 1's cell (`random-4-0`, a few
   steps) before the budget probe.
4. **The budget probe:** one seed an arm a cell, 30 epochs, on `stone`, for each arm's epoch time
   and memory; the round's estimate as a range, and its split across `stone` and `jebel` (D8).
5. **The round:** 54 runs (3 arms x 3 cells x 6 seeds), plus 36 for each cell D6 adds; recertified
   in both verdicts on `pyramidon`, from manifests (rsync from `stone`, never an authed service
   there). Readout per cell and arm; C2's probes on every failure's best network and on every
   certified one.
6. **Findings** in the probe's docstring; certified networks to indrajala-models; if A and B both
   help, A+B registered as its own experiment (D8); this plan retired, its leftovers to
   next-steps.md, among them the optional search of B's settings (D4).

Related: [commitment-control-workplan.md](commitment-control-workplan.md) (deferred) controls the
commitment's *weight*; A changes only its *target*, with the weight unchanged.
