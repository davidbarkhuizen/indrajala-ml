"""
The modular family's probe (docs/modular-family-workplan.md, stage 2): one network answering the
running quotient, sum and difference of two numbers mod p at every window (query tokens), against
the quotient alone; a factored state of C9''s cell, trained by the curriculum within a wall-clock
budget, diagnosed as it trains (indrajala_ml/studies/modular_family.py).

    OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/modular_family_probe.py run \\
        --arm family --seed 0 --budget-minutes 60 --out data/window-recurrent/modular-family-probe/family-seed0
    OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/modular_family_probe.py run \\
        --arm family --reads output --first-seed 2 --seeds 6 --budget-minutes 60 --out DIR
    OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/modular_family_probe.py run \\
        --arm family --reads output --factor 32 128 --first-seed 0 --seeds 4 --budget-minutes 60 --out DIR
    OPENBLAS_NUM_THREADS=1 .venv/bin/python scripts/modular_family_probe.py run \\
        --arm family --reads output --first-seed 100 --seeds 14 --workers 3 --budget-epochs 400 \\
        --budget-minutes 180 --out DIR
    .venv/bin/python scripts/modular_family_probe.py long-test NETWORK.json [...] --task modular-division-5 --out DIR
    .venv/bin/python scripts/modular_family_probe.py trace DIR/seed<s>/log.jsonl [...]
    .venv/bin/python scripts/modular_family_probe.py readout DIR --pairs family:quotient sum:quotient sum:product

`--fork` turns on snapshot and fork on collapse (the collapse-fork workplan, retired:
`git show 89cf3a3:docs/collapse-fork-workplan.md`): a step's epoch
whose screen falls below 0.6 of the step's best goes back to that best network, halves the step's
rate and takes new data, at most 4 times a step; each fork a line in the log, each step's end its
count.
`--first-seed F --seeds N` runs seeds F..F+N-1, a process each, `--workers` at once (default
all), into DIR/seed<s>/ (and its output and errors in DIR/seed<s>.log); `--budget-epochs` stops a
run when that many epochs are trained in all (or `--budget-minutes` pass, whichever first).
`--trigger-signal`, `--trigger-rule`, `--trigger-threshold` and `--trigger-floor` set the fork's
trigger (indrajala_ml/studies/collapse.py; the default is the fork's first: pooled, below 0.6 of
the peak, no floor). `drops LOG... [--trigger-...]` scores a trigger against runs' logs, offline
(docs/p11-trigger-workplan.md, D6): each run's large drops (the worst answer kind from 0.9 down
by more than 0.3), how many the trigger catches, of those never recovered how many, and its false
alarms; at each epoch where the log has every answer kind's share (from the p11-trigger build on),
else at each diagnosis (a proxy).
`trace` prints runs' logs a line per diagnosis (each factor's codes in use, the state's share, each
answer's accuracy), marking a collapse: the codes in use below half the diagnosis before.
`long-test` runs the long tests (docs/division-workplan.md, D4: U1 and S1 at 128, 256 and 1,024
columns, the harness's sizes) on saved family networks, one slot (the task's), into
DIR/<network>.long-test.json (skipped when there), a line each of its cases and failures.

Each run writes `log.jsonl` (a line per diagnosis: each operation's accuracy, the quotient by
class, each factor's codes used and its share of A, B, the sum S, the difference D and the zero
tests; at each step's end, the quotient's accuracy at 24 and 48 columns) and `network.json` (the
network at its last step's end).

Findings (stage 2 of docs/modular-family-workplan.md), on jebel (i7-9700K), one process a run,
OPENBLAS_NUM_THREADS=1, 2026-10-15; p 5, two factors of 16 codes, s 16 each, k 1 column (D3), the
probe's settings (D4), a 60-minute budget a run. Data in data/window-recurrent/modular-family-probe/
(a directory a run: log.jsonl, network.json; manifest `modular-family-probe.sha256`,
4d465007...a9847c25, 41 files, run from data/window-recurrent), the D10 check's in
data/window-recurrent/probe-d10/ (`probe-d10.sha256`, 4236a059...557ae9c3, 81 files), the certified networks and their certificates in
modular-family-probe/certify/ (in the first manifest); both machines.

- **Registered arms (the poolings before the state, D10, as C9's cell), indrajala-ml 9b4be51,
  06:08-07:09: none works.** family seeds 0 and 1 and quotient seeds 0 and 1 all end their hour
  with the nonzero quotients at 0.33-0.37 (a guess 0.25), the state's share of the pair 0.22-0.27,
  24-30% right at 24 columns; every factor's share of A, B, S or D at most 0.18, the shares frozen
  to three decimals from step to step. The family made no difference.
- **The cause is D10 (the D10 check: the harness, E8c's own cell and task, `modular-sum-5`, seeds 0
  and 1, 30 minutes each, the same commit):** one factor whose pooling reads before the state passes
  steps 1-2, then plateaus at every step from 3 (31-91 epochs each, break 0): 0 of 2 certified; the
  same factor reading the window's output certifies as E8c did, step for step (seed 0 at step 4,
  seed 1 at step 5): 2 of 2. Reading before the state, a factor's cell gets only the embedded column
  and must compute the whole transition itself; reading the output, the window's three blocks do
  most of it (a reading, not tested apart). Every factored division run (C9, C9', C9'') read before
  the state.
- **Reading the output (indrajala-ml 600b19b, 06:53-07:54), the family certified 2 of 2; the
  quotient alone 0 of 2.** family seed 1 was right on every answer at 3-12 columns and at 24 and 48
  from step 7, seed 0 from step 6. Their networks saved at steps 11 (seed 1, sha256 718b382d...4bd595be)
  and 12 (seed 0, 77b558a0...838bc3ef) **certify against division's machine, both verdicts** (certify,
  2da6d0b, pyramidon): seed 1 branched 33 states reached, minimised to 25, 0 wrong, 2 near ties,
  logit lead >= 2.03; strict 32 reached, minimised to 25, snap ratio >= 1.018, lead >= 3.81; seed 0
  branched 53 reached, minimised to 25, 6 near ties, lead >= 2.21; strict 52, minimised to 25, snap
  ratio >= 1.025, lead >= 2.21. Their sums certify against the modular sum too (minimised to 5). The
  first certified networks for division. seed 1's step-6 network, 98.7% right at 48 columns on
  samples, fails the certificate (1,025 wrong outputs): sampled tests overstate "works". The
  quotient alone reached nonzero quotients of at most 0.64 and at most 36% at 24 columns; seed 1
  collapsed at step 9 (codes 2 and 5).
- **The state isn't factored by residue.** At seed 1's step 6 each factor uses 14 codes and holds
  0.37-0.71 of every direction xA + yB (no factor a function of one), the pair of codes 63 states
  for the 25; the certificate's readout agrees (neither factor's code a function of the target's
  state). So the registered success's factor clause (a factor at 0.9 of A, B, S or D) isn't met;
  its quotient clause is: an outcome the plan didn't name, division solved with its state spread
  over both codebooks.
- **Training past a solution destroyed it, in every run that got far:** family seed 1 collapsed at
  step 12 (codes 4 and 4), seed 0 at step 14, both quotient runs by the hour's end (all four ended
  at nonzero quotients 0.13-0.31, 20-30% at 24 columns). The certified networks had been saved first.
  The probe now certifies each step's quotient and stops at the first certified (38f787c), as the
  harness's `stop_at_certified` does (E8b).
- Two counts at 2 seeds an arm are a screen: the family's part (2 of 2 against 0 of 2) is
  suggestive, not shown. Before launch the probe's examples lacked the harness's final zero window
  and the state never split in either arm (the step at one column passed on the column alone); found
  in local checks, fixed in 9b4be51 before any registered run.

Findings (stages 2 and 3 of docs/division-workplan.md), 2026-10-15; p 5, reading the output, k 1
column, the probe's settings, 60 minutes a run, stop at certified; the runs on jebel (i7-9700K), a
process each, OPENBLAS_NUM_THREADS=1. Data in data/window-recurrent/division-probe-<probe>/ (a
directory a seed: log.jsonl, network.json, certified-step-<s>.network.json; seed<s>.log,
probe.out, launch.txt; certificates in certify/, long tests in long-tests/), one manifest each,
run from data/window-recurrent, both machines: `division-probe-2a.sha256` 159740f5...529d032d
(29 files), `division-probe-2b.sha256` 785ecb08...4586bbd2 (14), `division-probe-3a.sha256`
63fcbd13...6f70fff1 (35), `division-probe-3b.sha256` 89a1e28e...32efd627 (14),
`division-probe-3c.sha256` 1e2dc84f...e42c0be5 (19). Certificates (both verdicts: snap ratio 1.05
and 1.0) on pyramidon.

- **Probe 2a (family, two factors of 16, seeds 2-7, c5378bc, 08:29-09:29): 3 of 6 certified;
  with the first probe's two, 5 of 8** (Wilson 0.31-0.86, a screen). seed 5 at step 7 (6.3
  minutes, sha256 141c25db...40610515), seed 7 at step 8 (19.7, 6e31fcec...e308dc9a), seed 4 at step
  10 (24.1, 4529c5dc...b7ab6d66), each against division's 25-state machine in both verdicts, 0 wrong
  outputs, 0 wrong states: branched 65, 57 and 88 codes reached, minimised to 25, 19, 13 and 35 near
  ties, logit lead >= 1.41, 0.598 and 1.07 (seed 4's snap ratio >= 1.001); strict 63, 57 and 88
  reached. Seeds 2, 3 and 6 ran out the hour (steps 17, 17, 21), each **collapsed**: the state
  minimised to 7, 6 and 11, 23-25% at 24 columns. seed 3 was right on every sampled answer at steps
  6-9 (100% at 24 columns, 99.99% at 48, minimised to 30 at step 9) without certifying, and
  collapsed at step 10 (minimised to 2 by step 12): sampled tests overstate "works" again, and
  training past a near-solution destroys it. Every success came within 25 minutes and every failure
  is a collapse, so a longer budget wouldn't raise the rate; the collapse would (the
  commitment-control workplan's thermostat).
- **Probe 2b (the long tests, D4: U1 100,000 and S1 25,000 cases at each of 128, 256 and 1,024
  columns): all five certified networks pass every one of their 375,000 cases**, read by
  `long_test_failures` (the first probe's two on pyramidon, 4,276 and 4,284 s; 2a's three on jebel,
  not pyramidon as D8 says, since jebel was free, 4,412-4,423 s, three at once). The certificate
  and the long tests agree.
- **Probe 3b (the quotient alone, two factors of 16, seeds 2-5, c5378bc, 09:30-10:30): 0 of 4; with
  the first probe's two, 0 of 6.** Each ran out the hour (steps 20-34) at nonzero quotients
  0.19-0.28, 21-27% at 24 columns; their best nonzero quotients 0.61-0.77, at 24 columns at most
  46%; seed 3 collapsed to one code a factor. The family against the quotient alone: 5 of 8 against
  0 of 6, Fisher's exact (two-sided) p 0.031, a screen across two probes (one per arm, not
  interleaved seeds), so the family looks like a real ingredient, the question for stage 5.
- **Probe 3a (family, one factor s 32 m 128, D5, seeds 0-3, 92b98da from a worktree, 10:30-10:48):
  4 of 4 certified**, so the second codebook isn't needed at p 5 (D9's question). seed 0 at step 5
  (8.2 minutes, 92677677...4385246b), seed 1 at step 6 (10.7, d50aad5a...1fb1452b), seed 3 at step 6
  (14.5, 25ca3d32...443529da), seed 2 at step 8 (18.0, ed3f9f3a...233867b8); both verdicts, 0 wrong
  outputs, 0 wrong states, 61-77 codes reached, minimised to 25, logit leads >= 2.27, 0.62, 2.41 and
  0.103 (seed 3's thin, but both verdicts hold). Long tests: every one of each network's 375,000
  cases passes (jebel, 92b98da, four at once, 4,336-4,349 s). Against two factors of 16 (5 of 8):
  p 0.49, no difference shown; every 3a run finished within 18 minutes. The working
  configuration stays factored (D9 (a)): the product state's sizes multiply in one codebook as `p`
  grows, so stage 4 keeps D7's two factors.
- **Probe 3c (the quotient and the sum, `--arm sum`, 3a's factor, seeds 0-1, ae2a09f from a
  worktree, 11:05-11:18): 2 of 2 certified, so one dense counter is enough.** seed 1 at step 10
  (9.4 minutes, 913daa16...ec78b606), seed 0 at step 9 (12.7, 4e9e5dd4...56c3a367); both verdicts, 0
  wrong outputs, 0 wrong states, 41 and 51 codes reached, minimised to 25, snap ratio >= 1.006,
  logit lead >= 4.7 and 4.14 (the widest margins yet). seed 1 minimised to 25 at steps 8 and 9 and
  certified only at 10. Long tests: every one of each network's 375,000 cases passes (jebel,
  ae2a09f, 3,534 and 3,539 s).
- **Eleven certified networks for division at p 5, all certified in both verdicts and passing
  every long-test case** (archived in indrajala-models, window-recurrent/2026-10-15-division).
- The feasibility table's cap of 152 for the one-factor cells isn't enforced: `run` passes no cap,
  so the curriculum's default 280 applied; no run passed step 34. (Since 8364a83 the cap defaults
  to the cell's exclusion bound, `modular_family.exclusion_bound`.)

Findings (stage 4 of docs/division-workplan.md, probe 4), 2026-10-15; p 7, the family, two factors
of 16 (256 joint states for 49; the exclusion bound 304, the cap), reading the output, k 1 column,
the probe's settings, 60 minutes a run, stop at certified, seeds 0-3; jebel (i7-9700K), the main
checkout at 8364a83, a process each, OPENBLAS_NUM_THREADS=1, 12:41-13:42. Data in
data/window-recurrent/division-probe-4/ (as stages 2-3's; the certificates on pyramidon, the long
test on jebel), manifest `division-probe-4.sha256` 5de66ce4...2e131958 (20 files, run from
data/window-recurrent), both machines.

- **1 of 4 certified: division at p 7, 49 states, for the first time.** seed 2 at step 7 (11.2
  minutes, 139 epochs, sha256 972d4699...fd929c4d), against division's 49-state machine in both
  verdicts, 0 wrong outputs, 0 wrong states: branched 88 joint codes reached, minimised to 49, 34
  near ties, logit lead >= 2.05; strict 87 reached, snap ratio >= 1.004, lead >= 2.33. It was right
  on every sampled answer at 24 columns from step 5 (minimised to 72, then 50 at step 6) and
  certified at step 7. Every p 7 attempt before (E7, E7': the quotient alone, one codebook) failed:
  0 of 80.
- **Its long test: every one of its 447,000 cases passes** (U1 and S1 at 128, 256 and 1,024
  columns, `--task modular-division-7`; jebel, 8364a83, 5,054 s; `long_test_failures`: none).
- **Seeds 0, 1 and 3 ran out the hour** (steps 19, 19, 18; 428-441 epochs), each **collapsed**:
  peaks of 31%, 33% and 30% at 24 columns (steps 10, 7, 13; minimised to 81-117), ending minimised to
  17, 5 and 20 at 19-21% at 24 columns. None was rising at the hour's end, so the registered
  conditional rule calls no 120-minute rerun; the next probe for p 7 would be D7 (b), two factors
  of 32, not yet decided against stage 5's needs.

Findings (the collapse, traced with `trace`; 2026-10-15, from the logs of every run above, no new
runs):

- **A collapse is abrupt.** Within one diagnosis interval (5 epochs) the codes in use fall to half
  or less and every answer to about chance: 15 such events in 11 runs (the codes in use halving from
  at least 8), 11 mid-step and 4 at a step's first diagnosis. p 7 seed 1, step 10, epochs 15 to 20:
  codes [15, 14] to [1, 1], the difference 0.91 to 0.20; 2a seed 3: right on every answer at step
  9, at chance at step 10's first diagnosis (codes 29 to 20 there, 3 by epoch 15). Runs partly
  recover and collapse again (p 7 seed 0, steps 11-13), and a collapse isn't always the end: 3c
  seed 0 collapsed at step 5 (codes 60 to 24) and certified at step 9.
- **No slow drift precedes it:** the codes in use hold at 15-16 a factor up to the diagnosis before.
  That fits an optimiser instability (a loss spike; Thilak et al. 2022's late instabilities of
  Adam) better than the commitment loss's gradual pull, the deferred commitment-control plan's
  premise. Training has no gradient clipping and a constant rate (Adam, 0.003), and the
  back-propagated sequences lengthen with every curriculum step.
- **At p 7 the codebooks are saturated:** both factors at 16 of 16 codes from step 2 to the
  collapse, one counter learned (the difference or the sum at 0.95-1.00), the other near 0.5. That
  argues for 32 codes a factor (division workplan, D7 (b)) apart from any fix.
- Not measured: the loss and the gradient's norm per batch around a collapse (the diagnosis is
  every 5 epochs); the next probe should log them.

Findings (stage 5 of docs/division-workplan.md, the registered claim, D11-D14), 2026-10-15; p 5,
two factors of 16, reading the output, the probe's settings, 400 epochs trained in all (180
minutes only a guard; no run reached it), stop at certified, seeds 100-119 in each arm; ce4015d,
OPENBLAS_NUM_THREADS=1; jebel seeds 100-113 (each arm 3 at once), pyramidon 114-119 (each arm 2 at
once), 14:00-18:10. Every certified network recertified on pyramidon (f22a9f0, the same code) in
both verdicts. Data in data/window-recurrent/division-claim/, manifest `division-claim.sha256`
3b17c4ca...518ed4b49 (168 files, run from data/window-recurrent; the long tests apart), both
machines. The readout is `readout data/window-recurrent/division-claim`.

- **Works: the family helps.** The family 13 of 20 certified (Wilson 95%: 0.43-0.82), the quotient
  alone 1 of 20 (0.01-0.24); Fisher's exact test, one-sided, p 6.9e-5 (registered: works at p <
  0.05). The screens' difference (5 of 8 against 0 of 6) holds at 20 fresh seeds.
- **The family's successes:** steps 6-12 (median 8), 99-257 epochs (median 177), 10-30 minutes on
  jebel and 18-55 on pyramidon (the minutes are the machine's and its load's; the epochs are the
  measure, D13). The quotient alone's one: seed 100, step 10, 202 epochs, 15.0 minutes.
- **Every recertification agrees:** 14 of 14 certified in both verdicts against division's
  25-state machine, 0 wrong outputs, 0 wrong states, each minimised to 25; 35-74 joint codes
  reached; snap ratio >= 1.0002 (220 near ties in the branched verdict, none strict), logit lead
  >= 0.10.
- **Every failure spent its 400 epochs:** the family's 7 at steps 14-20, the quotient's 19 at
  steps 13-25 (no run ended on the guard).
- **The family's failures mostly lost what they had; the quotient's mostly never had it**
  (`readout`'s course of each run). The worst answer at its best: 4 of the family's 7 failures
  reached >= 0.95 (seeds 101, 111, 113, 119 at 0.96-1.00, steps 6-8; 115 and 116 at 0.87 and 0.90),
  against 2 of the quotient's 19 (their median 0.79). That is the sampled answers, not the states:
  runs were right on every sampled answer, 48 columns included, with 160-177 states minimised (seed
  104, steps 6-8) and certified a step or more later, the minimised count falling to 25 in one
  step; seed 101, right on every sampled answer at step 6 with 67, never certified.
- **Large drops are common and mostly recovered** (from the diagnoses, every 5 epochs: the worst
  answer from >= 0.9 down by more than 0.3 within one interval): 16 in 10 of the family's runs (6
  of the 13 that certified, 4 of the 7 failures) and 8 in 4 of the quotient's; 7 at a step's first
  diagnosis, 17 mid-step. 17 of the 24 came back to >= 0.9 within 1-6 diagnoses. Whether the codes
  collapsed didn't decide it: seed 100 fell to [7, 6] codes and certified, the quotient's 105 and
  115 kept all 16 a factor and never came back. So a failure is a drop not recovered from, and the
  family's successes survived theirs.
- **The curriculum moves on while broken:** a step ends on a plateau after 10 epochs (`patience`)
  without beating its best screen score, and a drop leaves that best at the peak before it, so a
  run has 10 epochs to beat its own peak or takes on the next length broken. 13 of the family's 16
  drops ended their step on a plateau; of all the steps ending on one, 119 of the family's 133 and
  261 of the quotient's 262 ended with the worst answer below 0.9.
- **seed 107 never came near:** the sum 0.98-1.00 at 9 of steps 8-17 (0.58 at 12), the difference
  near chance (0.45-0.56) over steps 7-17 with 9-16 codes a factor in use: one counter learned, the
  other stuck, as at p 7 (stage 4).
- **For the fork (the collapse-fork workplan):** its snapshot is the step's start (the step
  before's network) and its best since, so it reaches drops at a boundary and mid-step alike; but it
  fires only below 0.6 of the step's peak screen score (the share of every answer right, not the
  worst answer). Of the 24 drops, from the logged screen scores (a diagnosis's; the fork sees every
  epoch, so these are the fewest it would fire on): 6 below 0.6 of the step's logged peak (100,
  111, 119, the quotient's 113 and 118 twice); 4 at a step's start below 0.6 of any start above
  0.97 (102, 106, 113, the quotient's 118); 11 not, 8 above 0.6 of the logged peak (110 twice, 112,
  113, 117, the quotient's 105 twice and 115) and 3 at a step's start above 0.6 of any start up to
  1.0 (101 at 0.76, 119 at 0.90, the quotient's 118 at 0.61); 3 unknown (no screen logged). So
  seed 101's loss, the clearest, is one it would miss. A restore doesn't reset the plateau count. Read the fork probe with this in
  mind; a drop threshold on the worst answer, or a step that ends only when its peak is recovered,
  are the alternatives the logs suggest.
- **Measured from here on** (after stage 5, the training unchanged: a 3-step run's network and
  diagnoses identical before and after): every epoch's screen score and the step's start (the step
  before's network on the new lengths), each step's network (`step-<j>.network.json`), and in
  `readout` each run's large drops (`large_drops`) with how many recovered. Stage 5's logs have
  the screen score at diagnoses only, and only the last network.
- **The long tests (D4) pass: every one of the 14 networks passes every case**, 375,000 each (U1
  100,000 and S1 25,000 at each of 128, 256 and 1,024 columns), 5,250,000 in all, no failure listed
  (`long_test_failures`: none). 11 on pyramidon (f22a9f0, one at a time, 4,232-6,143 s each), 3 on
  jebel (ce4015d, the same code, 2,321-3,430 s), 2026-10-15 19:00 to 2026-10-16 09:12, each worker
  claiming its next network by a mkdir on pyramidon (`~/code/stage5-long-tests.sh`; two of jebel's
  workers claimed the quotient's seed 100 in the same second, so it was tested twice, both passing).
  Results in `division-claim/long-tests/<arm>-<seed>/`, the per-network summary
  `division-claim/long-tests-summary.txt`, logs `long-tests-pyramidon.txt` and
  `long-tests-jebel.txt`; manifest `division-claim-long-tests.sha256` ae437c1d...c499329 (14 files,
  run from data/window-recurrent).

Findings (the fork probe, stage 3 of the collapse-fork workplan, D6, registered before its data;
the workplan retired: `git show 89cf3a3:docs/collapse-fork-workplan.md`), 2026-10-16; p 7, the family, two factors of 32 codes (`s` 16), reading the output, seeds
0-3, 400 epochs a run (`--budget-epochs`), stop at certified; two arms on the same seeds, with
`--fork` (drop 0.4, cool 0.5, at most 4 forks a step) and without; jebel, ~/code/wt-fork at a55ef4c,
8 runs at once, 06:35-07:38. Every certified network recertified on pyramidon in both verdicts
(`certify/`, `certify.log`). Data in data/window-recurrent/fork-probe/, manifest
`fork-probe.sha256` 7cdcacc0...75324e162 (47 files, run from data/window-recurrent). The readout
is `readout data/window-recurrent/fork-probe --arms fork nofork`.

| seed | fork | without |
| --- | --- | --- |
| 0 | certified, step 9, 149 epochs, 1 fork | certified, step 12, 208 epochs |
| 1 | spent at step 15, 18 forks, the worst answer at best 0.85, 358 states minimised | spent at step 14, collapsed, at best 0.70 |
| 2 | certified, step 11, 382 epochs, 6 forks | certified, step 12, 260 epochs |
| 3 | certified, step 15, 328 epochs, 23 forks | spent at step 17: a drop at step 12 (1.00 to 0.19) never recovered |

- **Works, as registered:** the fork arm certifies 3 of 4 (Wilson 95%: 0.30-0.95) against 2 of 4
  (0.15-0.85), and no fork run ends collapsed (seed 1's last step minimised to 358 states, not
  below half of 49). A screen (D6): one-sided Fisher p 0.5; the runs are paired, not a rate.
- **Five certified 49-state networks** (stage 4 had one in 4 runs at 16 codes a factor): each
  robustly certified against division's machine mod 7, 0 wrong outputs and states, minimised to
  49 from 82-242 codes reached (79-225 strict); snap ratio >= 1.000, logit lead >= 0.028 (fork
  seed 0's branched verdict; 2.63 strict). **Every one passes all 447,000 long-test cases**
  (U1 and S1 at 128, 256 and 1,024; `pyramidon`, the main checkout at 07d4cbd, one at a time
  2-2.5 h each, `~/code/fork-long-tests.sh`): `fork-probe/long-tests/`, summary
  `fork-probe/long-tests-summary.txt`, log `fork-probe/long-tests.log`; manifest
  `fork-probe-long-tests.sha256` 4fc08d26...7e84ab (5 files, run from data/window-recurrent).
  Archived in indrajala-models (window-recurrent/2026-10-15-division, `family-f2x32-p7-*`).
- **The arms are the same run until the first fork** (the same seed, the fork off is unchanged):
  seed 2's two arms are identical to step 6, epoch 53. So each seed is a paired comparison. Seed 3
  is the fork's case: the run without lost everything at step 12 and spent its budget; with the
  fork, 4 forks at step 12 (worst answer 0.81 after the first three, 1.00 after the fourth) and
  certified at step 15. Seed 0 forked once, at step 9's first epoch, and certified there, 3 steps
  and 59 epochs before the run without (which fell 1.00 to 0.25 at step 9 and recovered). Seed 2
  certified in both, the fork a step sooner but 122 epochs later.
- **Many forks are at a step's start:** 28 of the 48 at a step's epochs 1-3, the new length
  dropping the screen below 0.6 of the step's peak (its start included); the fork restores the
  start at half the rate. Seed 1 forked 3 times in each of steps 10-15 (at the start of 11-15) and
  stayed near its best (the worst answer mostly 0.82-0.85) instead of collapsing as without; its last machine
  minimises to 358 states, too many rather than too few.
- **Drops the fork didn't catch:** 4 large drops (the worst answer from >= 0.9 down > 0.3 between
  diagnoses) in the fork arm, all recovered: 2 forked in the next epochs (seed 2 at step 6, seed 3
  at step 8), 2 never forked (seed 2, steps 6 and 7: the screen, every answer's share, stayed
  above 0.6 of its peak while the worst answer fell). Without the fork, 8 drops in 3 runs (`readout`), 7
  recovered; the lost one is seed 3's.
- **Next, as registered:** the `p` 11 ladder, the fork on (two factors of 32, then 64; 4-6 seeds).
  The alternatives the logs suggest (a trigger on the worst answer; a step that ends only when
  its peak is recovered) stay in next-steps.md.

Findings (the approximation claim, A1-A3 and D1-D6 registered before its data; the workplan
retired: `git show c201622:docs/approximation-claim-workplan.md`), 2026-10-16/17; p 5, two factors of 16, reading the output, one vocabulary for every
arm (`--vocabulary sum difference product`), stage 5's settings otherwise (400 epochs, 180 minutes
only a guard, stop at certified, no fork), seeds 200-219 in each of four arms: `quotient` alone,
`sum` (the quotient and the sum, one approximation), `family` (the cover) and `product` (the
quotient and `A B mod p`, an extra output that is not an approximation). Code 7618dfe,
OPENBLAS_NUM_THREADS=1: pyramidon seeds 214-219 (`~/code/wt-approx`, each arm 3 at once in turn),
2026-10-16 15:13 to 00:46; jebel seeds 200-213 (`~/code/wt-p11`, two arms at once, 3 workers
each), 23:38 to 06:21. Every certified network recertified in both verdicts (`certify/`,
`post-jebel.txt`; on jebel at 7618dfe: pyramidon had no memory to spare beside its long tests, and
the certificate is deterministic). Data in data/window-recurrent/approximation-claim/, manifest
`approximation-claim.sha256` 9517dc2b...cf6f2772 (1,501 files). The readout is `readout data/window-recurrent/approximation-claim --pairs
family:quotient sum:quotient sum:product` (`readout.txt`).

| arm | certified (Wilson 95%) | successes: steps, epochs (median) | failures near a solution |
| --- | --- | --- | --- |
| family | 15/20 (0.53-0.89) | 6-12, 78-235 (136) | 4 of 5 |
| sum | 10/20 (0.30-0.70) | 7-11, 64-160 (116) | 5 of 10 |
| quotient | 1/20 (0.01-0.24) | 12, 279 | 4 of 19 |
| product | 0/20 (0.00-0.16) | - | 0 of 20 |

- **A1, A2 and A3 hold.** Fisher's exact, one-sided, Holm over the three (D5): A1 family more often
  than quotient p 5.0e-6 (adjusted 1.5e-5); A3 sum more often than product p 2.2e-4 (4.4e-4); A2
  sum more often than quotient p 0.0017 (0.0017). By the plan's reading: one approximation helps,
  and because it is an approximation, not because it is one more output.
- **Stage 5 replicates with the wider vocabulary:** the family 15 of 20 and the quotient 1 of 20
  here against 13 and 1 there (A1), so the product's token, one more embedding row, changes neither.
- **The cover does more than one approximation:** 15 of 20 against 10 (not a registered test).
- **Every recertification agrees:** 26 of 26 certified in both verdicts against division's 25-state
  machine, 0 wrong outputs, 0 wrong states, each minimised to 25; 36-137 joint codes reached (36-131
  strict); snap ratio >= 1.000 (653 near ties branched, none strict), logit lead >= 0.32.
- **Every failure spent its 400 epochs** (steps 14-22; no run ended on the guard).
- **The product neither helps nor is learned** (exploratory, after the data: each run's step end
  with the best worst answer). The quotient is the worst answer in all 20 product runs, at a
  median 0.59 right against the quotient alone's 0.66, and the product itself 0.71; no product run
  came near a solution (the quotient alone's 4 did), 11 of 20 collapsed. The sum and the cover's
  runs are right on their extra outputs at a median 1.00. Read digit by digit, the sum's next value
  is a function of the sum and the column, so it can be learned on its own; the product's needs
  both residues, as the quotient's does: the extra output that doesn't help is one as hard as the
  target. The family's worst answer at its best is the difference in 18 of 20.
- **Large drops** (`readout`): the family 34 in 14 runs (29 recovered), the sum 14 in 7 (10), the
  quotient 11 in 5 (9), the product none.
- **Every certified network passes every long-test case:** 26 of 26, all 375,000 cases each
  (division's U1 and S1 at 128, 256 and 1,024 columns; 9,750,000 in all), on jebel at 7618dfe,
  6 at a time, 2026-10-17 06:37 to 11:32, 46-68 minutes a network (`long-tests/`,
  `post-jebel.txt`; copied to pyramidon, manifest `approximation-claim-long-tests.sha256`
  a1c41b20...4ebd4c05, 53 files). Archived in indrajala-models
  (window-recurrent/2026-10-15-division, `*-approx-*`).

Findings (the `p` 11 ladder: docs/p11-trigger-workplan.md, stage 3, registered before its data:
works if at least one network certifies against division's 121-state machine in both verdicts),
2026-10-16; p 11, the family, two factors of 32 codes (`s` 16), reading the output, the fork on
with today's trigger (pooled screen below 0.6 of the step's peak), 600 epochs (900 minutes only a
guard), stop at certified, seeds 0-5. Code 7618dfe, OPENBLAS_NUM_THREADS=1, jebel, 6 at once,
21:38 to 23:38 (`~/code/jebel-chain.sh`, after E2b'). Data in data/window-recurrent/p11-ladder/,
manifest `p11-ladder.sha256` ef3281cd...f460b7dc (155 files, run from data/window-recurrent). The
readout is `readout data/window-recurrent/p11-ladder --arms family`.

| seed | end | step | epochs | minutes | forks | worst answer at best | large drops |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 0 | spent | 17 | 600 | 97.5 | 12 | 0.75 | 0 |
| 1 | certified | 10 | 293 | 34.5 | 1 | 1.00 | 0 |
| 2 | certified | 16 | 402 | 52.1 | 16 | 1.00 | 2 |
| 3 | spent | 27 | 600 | 119.9 | 0 | 0.41 | 0 |
| 4 | certified | 31 | 527 | 91.7 | 72 | 1.00 | 1 |
| 5 | certified | 16 | 543 | 80.1 | 11 | 1.00 | 1 |

- **Works: 4 of 6 certified** (Wilson 95%: 0.30-0.90) at the ladder's first rung, two factors of
  32, so the second rung (64 a factor) doesn't run. The largest machine learned so far: 121 states
  against `p` 7's 49.
- **Every one robustly certified:** both verdicts against division's machine mod 11, 0 wrong
  outputs, 0 wrong states, each minimised to 121; 210-428 joint codes reached strict (snap ratio
  1.0), 212-448 at 1.05 (821 near ties branched, none strict); logit lead >= 0.26 (seed 4 at 1.05;
  0.37 strict). `certify/`, `certify.log`, on pyramidon.
- **Every one passes every long-test case:** all 663,000 each (division's U1 and S1 at 128, 256 and
  1,024 columns), on pyramidon at c4a91e7 (`~/code/p11-long-tests.sh`), one at a time, 2026-10-17
  04:02 to 14:11, 140-174 minutes a network (`long-tests/`, `long-tests.log`), manifest
  `p11-ladder-long-tests.sha256` d0ab5c4e...6d0133250c49 (5 files). Archived in indrajala-models
  (window-recurrent/2026-10-15-division, `family-f2x32-p11-*`).
- **No run collapsed, and every large drop recovered:** 4 large drops in 3 runs (`readout`), all
  certified runs. The two failures never came near a solution: seed 0's worst answer at best 0.75,
  seed 3's 0.41 (at step 9, then 18 more steps without the fork firing once). Neither failure is the
  collapse the trigger targets.
- **The fork fires most where the run succeeds slowest:** 72 forks in seed 4 (certified at step 31,
  the latest), 1 in seed 1 (the earliest, step 10). Exploratory: whether those forks rescued seed 4
  or held it back isn't separable without the alternate trigger's paired replays (stage 4).
- **Next:** stage 4, the alternate trigger (D7), is the plan's next step; the certificate guard
  (`replay --guard`, the snap-wall workplan's stage 2d) joins the fork rather than superseding
  it (the plan's status, 2026-10-17).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import contextlib
import functools
import json
import multiprocessing
import sys
from pathlib import Path

from indrajala_ml.studies import modular_family as mf
from indrajala_ml.studies import window_recurrent_certificate as wc
from indrajala_ml.studies import window_recurrent_study as wr
from indrajala_ml.studies.collapse import RULES, SIGNALS, Trigger

ARMS = {"family": mf.FAMILY, "quotient": mf.QUOTIENT, "sum": mf.SUM, "product": mf.PRODUCT}


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run", help="one probe run")
    run.add_argument("--arm", choices=sorted(ARMS), required=True)
    seeds = run.add_mutually_exclusive_group(required=True)
    seeds.add_argument("--seed", type=int, help="one run, into --out")
    seeds.add_argument("--first-seed", type=int, help="--seeds runs from this seed, at once, into --out/seed<s>")
    run.add_argument("--seeds", type=int, default=1)
    run.add_argument("--p", type=int, default=5)
    run.add_argument("--factors", type=int, default=2, help="factors of 16 codes, s 16 each")
    run.add_argument(
        "--factor",
        type=int,
        nargs=2,
        action="append",
        metavar=("S", "M"),
        help="a factor of width S, M codes (for --factors)",
    )
    run.add_argument(
        "--reads", choices=["before_state", "output"], default="before_state", help="the poolings' tokens (D10)"
    )
    run.add_argument("--budget-minutes", type=float, default=60.0)
    run.add_argument("--budget-epochs", type=int, help="epochs trained in all, whichever budget is spent first")
    run.add_argument("--workers", type=int, help="runs at once (default: every seed at once)")
    run.add_argument(
        "--fork", action="store_true", help="snapshot and fork on collapse (the collapse-fork workplan, retired)"
    )
    _trigger_arguments(run)
    run.add_argument(
        "--vocabulary",
        nargs="+",
        choices=list(mf.OPERATIONS),
        default=list(mf.VOCABULARY),
        help="the operations with a token (the approximation claim's: sum difference product)",
    )
    run.add_argument("--out", type=Path, required=True)
    certify = commands.add_parser("certify", help="a saved family network's certificate, one slot")
    certify.add_argument("network", type=Path)
    certify.add_argument("--task", required=True, help="modular-division-p (the quotient) or modular-sum-p (the sum)")
    certify.add_argument("--arm", choices=sorted(ARMS), default="family")
    certify.add_argument("--snap-ratio", type=float, default=wc.SNAP_RATIO)
    certify.add_argument(
        "--vocabulary",
        nargs="+",
        choices=list(mf.OPERATIONS),
        default=list(mf.VOCABULARY),
        help="the operations with a token (the approximation claim's: sum difference product)",
    )
    certify.add_argument("--out", type=Path, required=True)
    long = commands.add_parser("long-test", help="saved family networks' long tests, one slot")
    long.add_argument("networks", type=Path, nargs="+")
    long.add_argument("--task", required=True, help="modular-division-p (the quotient) or modular-sum-p (the sum)")
    long.add_argument("--arm", choices=sorted(ARMS), default="family")
    long.add_argument("--n", type=int, nargs="+", default=list(wr.LONG_LENGTHS))
    long.add_argument("--properties", nargs="+", default=list(mf.LONG_PROPERTIES))
    long.add_argument(
        "--vocabulary",
        nargs="+",
        choices=list(mf.OPERATIONS),
        default=list(mf.VOCABULARY),
        help="the operations with a token (the approximation claim's: sum difference product)",
    )
    long.add_argument("--out", type=Path, required=True)
    trace = commands.add_parser("trace", help="runs' logs, a line per diagnosis, collapses marked")
    trace.add_argument("logs", type=Path, nargs="+", help="each run's log.jsonl")
    drops = commands.add_parser("drops", help="a collapse trigger scored against runs' logs, offline")
    drops.add_argument("logs", type=Path, nargs="+", help="each run's log.jsonl (runs made without the fork)")
    _trigger_arguments(drops)
    readout = commands.add_parser("readout", help="a registered comparison: each arm's runs, counts, Fisher's test")
    readout.add_argument("root", type=Path, help="the study's directory, a <arm>/seed<s> per run")
    readout.add_argument("--arms", nargs="+", default=["family", "quotient"], help="the first is tested as better")
    readout.add_argument("--certify", type=Path, help="the recertifications (default: root/certify)")
    readout.add_argument(
        "--pairs",
        nargs="+",
        metavar="BETTER:OTHER",
        help="the registered tests, each arm against another (Holm over them); the arms are theirs, in order",
    )
    args = parser.parse_args(argv)
    if args.command == "drops":
        trigger = _trigger(args)
        print(f"trigger: {trigger.describe()}")
        scores = [
            mf.score_trigger([json.loads(line) for line in path.read_text().splitlines()], trigger)
            for path in args.logs
        ]
        for path, score in zip(args.logs, scores, strict=True):
            print(
                f"{path}: {score.points} points, {score.drops} large drops, {score.caught} caught, "
                f"{score.unrecovered_caught}/{score.unrecovered} unrecovered caught, {score.false_alarms} false alarms, "
                f"{score.early} early"
            )
        print(
            f"{len(scores)} runs: {sum(s.drops for s in scores)} large drops, {sum(s.caught for s in scores)} caught; "
            f"{sum(s.unrecovered_caught for s in scores)} of {sum(s.unrecovered for s in scores)} never recovered "
            f"caught; {sum(s.false_alarms for s in scores)} false alarms near a solution, "
            f"{sum(s.early for s in scores)} firings while still learning"
        )
        return 0
    if args.command == "readout":
        pairs = [(better, other) for better, other in (pair.split(":") for pair in args.pairs)] if args.pairs else None
        arms = list(dict.fromkeys(arm for pair in pairs for arm in pair)) if pairs else args.arms
        print("\n".join(mf.readout(args.root, arms, args.certify or args.root / "certify", pairs)))
        return 0
    if args.command == "trace":
        for path in args.logs:
            print(f"== {path}")
            print("\n".join(mf.trace([json.loads(line) for line in path.read_text().splitlines()])))
        return 0
    if args.command == "long-test":
        args.out.mkdir(parents=True, exist_ok=True)
        p = int(args.task.rsplit("-", 1)[1])
        for path in args.networks:
            print(long_test_file(path, mf.Encoding(p, ARMS[args.arm], tuple(args.vocabulary)), args), flush=True)
        return 0
    if args.command == "certify":
        p = int(args.task.rsplit("-", 1)[1])
        encoding = mf.Encoding(p, ARMS[args.arm], tuple(args.vocabulary))
        result = mf.certify_file(args.network, encoding, args.task, args.snap_ratio)
        args.out.write_text(json.dumps(result, indent=1))
        print(wc.summary(result))
        return 0
    encoding = mf.Encoding(args.p, ARMS[args.arm], tuple(args.vocabulary))
    factors = [tuple(f) for f in args.factor] if args.factor else [(16, 16)] * args.factors
    probe = functools.partial(
        mf.run,
        encoding,
        factors,
        budget_seconds=args.budget_minutes * 60,
        reads=args.reads,
        budget_epochs=args.budget_epochs,
        fork=mf.Fork(trigger=_trigger(args)) if args.fork else None,
    )
    if args.seed is not None:
        probe(seed=args.seed, out=args.out)
        return 0
    args.out.mkdir(parents=True, exist_ok=True)
    seeds = range(args.first_seed, args.first_seed + args.seeds)
    context = multiprocessing.get_context("forkserver")
    with concurrent.futures.ProcessPoolExecutor(
        min(args.workers or len(seeds), len(seeds)), mp_context=context
    ) as pool:
        runs = {pool.submit(logged_run, probe, s, args.out): s for s in seeds}
        for done in concurrent.futures.as_completed(runs):
            print(f"seed {runs[done]}: {done.result()}", flush=True)
    return 0


def _trigger_arguments(command: argparse.ArgumentParser) -> None:
    default = Trigger()
    command.add_argument("--trigger-signal", choices=SIGNALS, default=default.signal, help="the trigger's signal")
    command.add_argument(
        "--trigger-rule",
        choices=RULES,
        default=default.rule,
        help="ratio: below threshold x peak; fall: threshold below it",
    )
    command.add_argument("--trigger-threshold", type=float, default=default.threshold)
    command.add_argument(
        "--trigger-floor", type=float, default=default.floor, help="fire only once the step's peak is this"
    )


def _trigger(args: argparse.Namespace) -> Trigger:
    return Trigger(args.trigger_signal, args.trigger_rule, args.trigger_threshold, args.trigger_floor)


def logged_run(probe: functools.partial[None], seed: int, out: Path) -> str:
    """One of several runs, into out/seed<s>/, its output and errors to out/seed<s>.log; how it ended."""
    with (out / f"seed{seed}.log").open("w") as log, contextlib.redirect_stdout(log), contextlib.redirect_stderr(log):
        probe(seed=seed, out=out / f"seed{seed}")
    rows = [json.loads(line) for line in (out / f"seed{seed}" / "log.jsonl").read_text().splitlines()]
    last = next(r for r in reversed(rows) if "end" in r)
    verdict = "certified" if last["certified"] else last["end"]
    return f"step {last['step']}, {verdict}, {last['seconds'] / 60:.1f} min"


def long_test_file(path: Path, encoding: mf.Encoding, args: argparse.Namespace) -> str:
    """One network's long tests into args.out/<network>.long-test.json, unless there; a line of its cases and failures."""
    target = args.out / f"{path.stem}.long-test.json"
    if target.exists():
        return f"{path.stem}: done before"
    long = mf.long_test(path, encoding, args.task, args.n, args.properties)
    target.write_text(json.dumps(long, indent=1))
    cases, failures = wr.long_test_failures(long)
    seconds = sum(length["seconds"] for length in long["lengths"].values())
    return f"{path.stem}: {cases:,} cases, " + (", ".join(failures) or "every case passed") + f"; {seconds:.0f} s"


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
