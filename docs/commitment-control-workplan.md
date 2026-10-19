# Workplan: controlling the codes' commitment (a factor that collapses)

**Status: written 2026-10-15 at the owner's request; DEFERRED behind option 2
(`git show e45944e:docs/modular-family-workplan.md`): it starts only if option 2's probe
fails, or if a later probe shows factors collapsing. Decisions D1-D5 settled (the owner delegates
a workplan's decisions; each with its options' pros and cons). Probe-driven, as option 2.**

**Since (2026-10-15):** option 2's probe certified division (2 of 2, the poolings reading the
output), so this plan stays deferred. Collapse was seen again, but **after** a solution: every run
that solved or nearly solved collapsed when trained on (codes to 2-8, the answer to chance), none
while learning. If a later probe needs a run to keep training past a solution, the thermostat
(D1 (a)) is the first thing to try; otherwise stopping at the first certified step suffices.

**Later (2026-10-15 evening): traced, the division runs' collapse is abrupt** (within 5 epochs,
mostly mid-step, no slow fall of the codes in use before it; `scripts/modular_family_probe.py`'s
findings, "the collapse, traced"). That fits an optimiser instability better than the commitment's
pull, so this plan stays deferred: the fix for the division runs is planned apart (the owner: not
the commitment, unless it is shown to be the cause), its candidates in
[next-steps.md](next-steps.md), "From the division work".

**The diagnostic's answer (2026-10-19; findings in `scripts/modular_family_probe.py`): 3 of 6
formed a state with the commitment off for steps 1-5, so by its registered answer the commitment's
pull is failure A's cause and stage 2, the thermostat, is next** (failure A is bottleneck 1 of
[bottlenecks.md](bottlenecks.md)). Its readings shape the build: the commitment's return at full
weight eroded the states found (D2's ramp and hysteresis are needed), and one state formed and was
lost with the commitment off (it should come back once a state has formed: the restore level).
Remedy A's interaction (below) is moot for now: A isn't kept (the consolidation plan's results).

**Revisited (2026-10-18, at the owner's request: is it still relevant, would it still be
useful?). As written, no; re-aimed at failure A, possibly; it stays deferred, re-aimed, its first
step a diagnostic probe rather than a build.**

- **Both of its reasons are gone.** It was written for C9'''s staggered factors, each collapsing
  to one code in its turn; the factored-state plan is retired (`git show
  e45944e:docs/factored-state-workplan.md`), and the factors that train today (the modular family,
  the general algebras) spread over their codes from the start. Collapse **after** a solution is
  an optimiser cascade, not the commitment: the snap-wall plan's machine-diff probe found every
  collapse beginning as 1-3 (state, window) pairs flipped by an ordinary update, amplified by Adam
  within 1-13 batches, and its certificate guard kept the proof in 10 of 10 replays against 1 of
  10 without; the fork restores the network after one. D5's probe cells (the C9'' networks, a C9'
  cell) no longer exist.
- **Where it still fits: failure A, a state that never forms.** General-algebras round 1:
  word-problem-s3 failed 5 of 6 and move-to-front-4 seed 3 the same way, peaking at a state share of
  0.06-0.32 on at most 6 codes for 2-6 of 6 states, **and ending on one code**
  (docs/code-consolidation-workplan.md, "Why", the second reading). That is this plan's mechanism:
  with nothing yet distinguishing the states, the commitment's pull is met most cheaply by a
  constant; and the restart probe kept 14-16 codes with the commitment off against 2-8 with it on.
  It is the one open failure the plan fits, in the algebras the work returns to. Unshown: the runs'
  codes in use over time haven't been read against this.
- **A new constraint: remedy A works through the commitment.** The consolidation plan's A
  (`--consolidate`) re-points the commitment's target from a code to its group's representative,
  with `beta` unchanged; a thermostat setting `beta` to 0 switches A off with it. A control of the
  commitment's weight and A's control of its target are designed together, or not both used.
- **So the next step is a diagnostic, not stage 2's build:** replay word-problem-s3's failed
  seeds with the commitment off for their first steps (D1 (b), a phase schedule: a probe's
  setting, nothing to build beyond a flag) and read whether a state forms (codes in use, the
  state's share, the exact prefix) against the same seeds with it on. A state forming where it
  didn't makes the thermostat (D1 (a), stage 2) worth building, its probes on failure A's cells
  instead of D5's; no change retires this plan, its idea to next-steps.md. Registered with its
  probe, before its data, when the general algebras take it up.
- **The quantiser (D1 (d))** stays behind the thermostat: a crate change across three
  implementations, for a failure not yet shown to be the commitment's.

**The diagnostic, registered 2026-10-18 before its data (a probe; it claims nothing).** Built in
this PR: `run --commitment-from J` sets every factor's commitment to 0 for the steps before J and to
its specs' 0.25 from J. The value is set before each epoch's training, so a fork's restored network
has it too. Without the flag a run is unchanged, by bits against `main` and with the golden run.

- **The runs:** round 1's failure-A runs, word-problem-s3 seeds 0, 1, 3, 4 and 5 and move-to-front-4
  seed 3. Round 1's cell (two factors, `--codes-ratio 32/11`, reading the output, the fork, 600
  epochs) with **one change, `--commitment-from 6`:** steps 1-5 without the commitment. Five,
  because round 1's certified S3 seed formed its state by then (seed 2's state share 0.96 at step
  4's end, 1.0 at step 5's), while the six failures spent 47-86 epochs in steps 1-5 and never
  passed a share of 0.06-0.38.
- **The control:** the same seeds' round-1 runs, if the first epochs of one of them (word-problem-s3
  seed 0) repeat by bits on the build's commit; else they are rerun beside the probe.
- **A run forms a state** if its state share at step 5's end is at least 0.5, above every failure's
  steps 1-5 and below the certified seed's 0.96 at step 4.
- **Answers, written before the data:**
  - **3 or more of 6 form a state (or any certifies):** the commitment's pull is failure A's cause.
    The thermostat (D1 (a), stage 2) is worth building, its probes on failure A's cells instead of
    D5's.
  - **1 or 2 of 6:** suggestive. Read the codes in use and whether the state survives the
    commitment's return at step 6, and choose one more probe from that.
  - **0 of 6:** the commitment isn't failure A's cause. This plan retires and its idea moves to
    next-steps.md.
  - Read in every case: whether a state formed in steps 1-5 survives step 6 (the state share and
    codes in use at steps 6-10, against step 5's).
- **Feasibility:** round 1's (S3: 6 states, 8 codes a factor, `m` 64, cap 69; move-to-front 4: 24,
  15, 225, 248; `s0` 1 for both).
- **Where:** `jebel` after the general algebras' round 2, the six runs at once in one pool (`run
  --runs word-problem-s3:0 word-problem-s3:1 word-problem-s3:3 word-problem-s3:4 word-problem-s3:5
  move-to-front-4:3 ... --commitment-from 6`). **Corrected before the data (2026-10-18):** up to
  about 4 h, not an hour. Round 1's six failing runs took 80-250 minutes each (6 at once on
  `pyramidon`), since a failure spends its 600 epochs.

## Why

Each code layer's loss, per window, is `weight * (||sg[h] - e||^2 + beta ||h - sg[e]||^2)`
(`CodesArrayLayer`, W4, after van den Oord et al. 2017; `beta` the commitment, 0.25): the codebook
term pulls a code to its outputs, the commitment term pulls an output `h` to its code.

- **C9''s collapse.** A staggered factor's cell came out of its turn giving one output for every
  input (seed 6's factor 1: a feature's spread across inputs 0.018 at initialisation, under 0.001
  after; its update gate's bias 1.0 to 2.5), on one code. A state of one code passes nothing on;
  the commitment pull is the one consistent force left, and the cheapest way to meet it is a
  constant.
- **The restart probe** (`data/window-recurrent/probe-factor-restart/`, manifest
  `probe-factor-restart.sha256` 488174f1...1cedbb32, both machines; its script beside the data):
  a re-drawn factor with the commitment on fell from 11-12 codes to 2-8 within 5 epochs; with it
  off for those 5 epochs, it kept 14-16. Two networks, so suggestive, not shown.
- **What the commitment buys.** With the state snapped in the forward pass whatever `beta` is, the
  commitment shapes learning, not the machine: it keeps `h` near its code, so the straight-through
  gradient (computed at the code, applied to `h`) is accurate and **the snap margins are wide:
  the certificate's snap ratio** (E8c's certified counters 1.02-7.91; a margin under 1.05 is a
  near tie, branched). Off for good: outputs spread to their regions' edges, near ties multiply,
  codes flip between batches.

So the commitment should be off while a factor is starting or collapsing, and on whenever it is
working and before it is certified.

## Decisions

- **D1. What switches it: (a) a usage thermostat, and (d) a soft-to-hard quantiser as its
  alternative, both built and probed** (the owner, 2026-10-15: "both a and d").
  - (a) *Chosen.* **The usage thermostat.** Per factor, the entropy of its codes' use over an
    epoch's training windows (the network's own statistic, never the target): below a floor,
    `beta` 0; above a higher restore level, `beta` ramped back to 0.25 (D2). Pros: reads only the
    network; catches a collapse at any time, a restart included; per factor; harness-level (a
    wrapper as `FactorRates`), the crate already takes a commitment per factor. Cons: two levels
    to set; without the gap between them it oscillates.
  - (d) *Chosen as the alternative.* **A soft-to-hard quantiser:** in training, a factor's next
    state the softmax-weighted mix of its codes (weights from `-||h - e||^2 / tau`), `tau`
    annealed to a hard snap (D3). Pros: every code gets a gradient, so no dead codes and no pull
    to one; the principled fix (soft-to-hard vector quantisation, Agustsson et al. 2017). Cons: a
    change to the codes' forward and backward in all three implementations and the crate (the
    crate first, its own PR); the state is a blend, not a machine, until the anneal ends, so a
    step is certified only after it.
  - (b) A phase schedule (off for a turn's first epochs). Pros: the probe's setting. Cons: blind to
    a collapse later; a special case of (a).
  - (c) A per-window gate (the commitment only where `h` is already near its code). Cons: a crate
    change for a second margin; a variation of (a) if (a) half works.
  - (e) A learned `beta`. Cons: its gradient drives it to 0; never.
- **D2. The thermostat's levels.** Off below 1.58 bits (fewer than about 3 codes in effective
  use, `log2 3`), restored above 2.32 bits (`log2 5`, a counter's states at `p` 5), ramped from 0
  to 0.25 over 5 epochs. Pros: the probe's survivors used 14-16 codes, collapsed factors 1-2: the
  levels sit between; the ramp avoids a jump in the loss. Cons: chosen from one probe.
- **D3. The quantiser's anneal.** `tau` from 1.0, halved every 5 epochs, a hard snap once a
  window's largest weight is above 0.99 on 99% of an epoch's windows; then the commitment on.
  Pros: ends hard by a measured test, not a fixed epoch. Cons: a schedule to set; a probe first.
- **D4. Certification only with the commitment on and the state hard:** a step is certified only
  after 5 epochs with `beta` at 0.25 (and, for (d), hard). Pros: the margins the certificate
  needs. Cons: a certified network may come a few epochs later.
- **D5. Probes, not runs.** Each mechanism probed on (i) the collapsed C9'' networks (seeds 0 and
  6, the restart probe's arms as controls) and (ii) a fresh C9' cell, 2 seeds, about 1 h on
  `jebel`; success registered with each probe (a factor that keeps at least 3 codes through 40
  epochs and gains share of a residue from its post-restart baseline).

## Stage 2, re-aimed at failure A (2026-10-19, before its data)

The owner, 2026-10-19: the thermostat is worth pursuing now, the diagnostic's evidence driving it
(bottleneck 1 of [bottlenecks.md](bottlenecks.md); R5 of
[one-code-one-state-workplan.md](one-code-one-state-workplan.md)). D2, D4 and D5 were set for
C9''s collapsed factors, which no longer exist; failure A's evidence re-sets them. D1 (a) stands.

- **The signal (D1 (a), made exact): each factor's effective codes,** `2^H` of its codes' use
  over a fixed sample of the step's examples at each epoch's end (every window's snapped code;
  the sample drawn from its own generator, so training's draws don't change). Round 1's failure-A
  runs sat at 1-5 codes a factor, mostly 1-2; the certified S3 seed used 5-7; the diagnostic's
  runs kept 5-8 with the commitment off.
- **D2' (replacing D2). The levels: off below 3 effective codes; restored once a factor's effective
  codes reach half its codebook, ramped from 0 to its specs' commitment over 20 epochs; off again
  only below 3.**
  - (a) D2 as written (off below 3, restore above 5, a 5-epoch ramp). Pros: settled. Cons: at
    move-to-front-5 (32 codes) a formed state uses 14-32, so 5 restores long before a state forms;
    and the diagnostic's full return eroded formed states within a step (S3 seed 0: 0.97 to 0.56).
  - (b) *Chosen.* Half the codebook to restore, 20 epochs of ramp. Pros: scales with the cell (S3
    4 of 8, move-to-front-5 16 of 32), every certified network here past it (S3 7-8 of 8, W1 22-32
    of 38, move-to-front 4 12-14 of 15); the ramp four times D2's against the erosion seen; the
    gap from 3 to half the codebook is the hysteresis. Cons: one more level; a state formed on
    fewer than half its codes waits off (S3 seed 3 lost a state formed with it off: the restore
    level exists for that case, and may need lowering).
  - (c) Restore when the effective codes stop growing (a plateau). Cons: a run could plateau low
    for long; another setting (its patience).
- **D4' (replacing D4): a step is certified as now, whatever the commitment.** The certificate's
  two verdicts (snap ratio 1.0 and 1.05) already test the margins D4 waited for. Pros: no special
  case; stop at certified unchanged. Cons: a network certified with the commitment low may have
  thinner margins: the verdicts say whether they suffice.
- **D5' (replacing D5): the probe, then C3.** The probe: the diagnostic's six runs (S3 seeds 0, 1,
  3, 4, 5; move-to-front 4 seed 3, round 1's cell) with `--thermostat`, its expectation written
  in its script before the data; its control the diagnostic and round 1. **C3, registered here:**
  round 2's move-to-front-5 cell (two factors, `--codes-ratio 32/11`, reading the output, the
  fork, 600 epochs, seeds 0-5) with `--thermostat`, against round 2 (0 of 6, four of them failure
  A) by the by-bits rule. **Helps:** 4 of 6 or more (Fisher one-sided p 0.030). **Works:** 1-3.
  Read every failure: failure A (a state's share below 0.5 at its best) or another bottleneck. A
  0 of 6 with fewer failure-A runs than round 2's four still shows the thermostat moving
  bottleneck 1. Feasibility: round 2's (`q` 120, 32 codes a factor, `m` 1,024, cap 1,143).

**The probe's answer (2026-10-19; findings in `scripts/modular_family_probe.py`): code use is the
wrong restore signal, so C3 doesn't launch as registered.** 0 of 6 certified, 2 formed a state (one
lost); effective codes reach half the codebook within 2-3 steps while the state's share is about
0.2, so the commitment returns before a state forms. The diagnostic's fixed schedule (off for steps
1-5) did better: 3 of 6 formed, one certified, all at the step where the answers came right.

- **D2'' (2026-10-19, before its probe's data): off below 3 effective codes, as D2'; from off,
  restored once the step's best screen reaches `restore_screen` (0.95), in place of half the
  codebook; then ramped back over 20 epochs as D2' (`run --thermostat --thermostat-screen 0.95`).** The screen is the answers' accuracy on the
  step's words (the network answering its training lengths), an inputs-and-answers signal like the
  fork's trigger, never the target's states. With the final zero window, whose answer needs the
  state, a screen near 1 needs a state.
  - (a) *Chosen.* The step's screen at 0.95. Pros: it is when the diagnostic's states formed;
    general (every task has a screen); still per step, so a later step's new lengths can switch it
    off again (below the floor) and back. Cons: a step whose screen never reaches 0.95 trains
    without the commitment throughout (near ties: the certificate's verdicts still judge them).
  - (b) Restore at a fixed step (the diagnostic's schedule). Cons: blind to the run; 5 suited S3 by
    chance.
  - (c) The network's own machine (its `d`-equivalence groups) stabilising. Pros: no answers used.
    Cons: groups stabilise on wrong machines too (consolidation's reading).
- **Its probe:** the same six runs with D2'', its expectation written before its data; then C3 as
  registered, with D2'' in place of D2'.

## Stages (deferred)

1. **This plan.**
2. **The thermostat** in the harness, with tests (the levels, the hysteresis, the ramp, a frozen
   `beta` bit for bit the network as now); the restart probe turned into a repo tool beside it;
   its probes.
3. **The quantiser:** the crate's codes forward and backward with `tau` (`indrajala-math-rust`
   first), then the package's three implementations, parity tests; its probes.
   **(2026-10-19) This is level 1's annealing** (the owner: a temperature that slowly cools, the
   state crystallising): it follows the thermostat's second probe (D2''), whatever that finds, if
   either it or the population plan ([population-workplan.md](population-workplan.md), level 2)
   shows the commitment's timing matters; its `tau` is then the population's natural temperature.
4. **Retire this plan;** leftovers to next-steps.md.

Stage 2 is tier 0 (harness, pure Python). Stage 3 changes a timed path (the crate's recurrence):
tier 1, with the golden run.
