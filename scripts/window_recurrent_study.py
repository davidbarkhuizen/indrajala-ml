"""
The window-recurrent adder (docs/window-recurrent-adder-workplan.md): whether a network that reads a
window of k columns and hands a snapped state from window to window learns, through a curriculum of
window counts, to add base-3 numbers of any length, under indrajala_ml/studies/window_recurrent_study.py's
harness and indrajala_ml/data/addition_data.py's catalogue.

    python scripts/window_recurrent_study.py run CONFIG.json [--workers W] [--arm ARM]
    python scripts/window_recurrent_study.py status CONFIG.json
    python scripts/window_recurrent_study.py report CONFIG.json [--md FILE]
    python scripts/window_recurrent_study.py check NETWORK.json --n N [N ...] [--task T] [--properties ID ...]
        [--sampled S] [--per-stratum P] [--doubling] [--json FILE]
    python scripts/window_recurrent_study.py certify NETWORK.json [...] [--task T] [--out DIR] [--workers W]
        [--snap-ratio R]
    python scripts/window_recurrent_study.py explore CONFIG.json [--n N ...] [--json FILE] [--workers W]
    python scripts/window_recurrent_study.py compare ORIGINAL_DIR RERUN_DIR
    python scripts/window_recurrent_study.py long-test NETWORK.json [...] [--task T] [--n N ...] [--properties ID ...]
        --out DIR [--workers W]
    python scripts/window_recurrent_study.py networks CONFIG.json
    python scripts/window_recurrent_study.py verdicts CONFIG.json --certificates DIR --strict DIR [--long-tests DIR]
    python scripts/window_recurrent_study.py probe CONFIG.json [--steps J ...] [--epochs E] [--json FILE]
    python scripts/window_recurrent_study.py diagnose CONFIG.json [--run KEY ...] [--n N ...] [--cases C]
        [--follow] [--interval S]

A config names the study's directory (results/, one JSON file per run, and models/, each run's steps
and their networks in format 2), its workers, its settings and its matrix (load_config's
docstring), its task among them (addition by default, or modular-division-5 or -7:
indrajala_ml/studies/window_tasks.py). `run` resumes from what is on disk, a run from its last saved step, and prints a line
per finished run, then the report; it holds the machine lock, as every study. `--arm` runs one arm's
runs only (a study's arms split between machines; keys and directory unchanged). `check` loads a saved
step's network and runs its task's catalogue (without the held-out property, B3 or U2) at each
length: its properties, the samples (addition's B1-B3, A3, M1 and M5; division's U1 and U2), the
cases per stratum (M3, M4; S1, S2) and M3 or S2 at 1, 2, 4, ..., n only (the long tests of
docs/window-recurrent-claims-workplan.md, D9) as given. `long-test` is those long tests over many
networks and W processes (at 128, 256 and 1,024 by default, the stratified distances doubling), each
network's results as DIR/<run>-step-<j>.json (`check --json`'s), a network already there skipped,
a line each of its cases and failures. `certify` writes each saved network's
certificate against its task's machine (indrajala_ml/studies/window_recurrent_certificate.py) as
DIR/<run>-<step>.json and prints a line for each, over W processes; --snap-ratio 1 certifies the
network's own snaps only, branching no near tie (strict, D3 as first written). `compare` is C0's check
(the claims workplan): every run in both study directories, its result without timings and
provenance and each step's saved network, exactly; a line per run, exit 1 if any differ. `explore`
is E5: for every finished snapped run of a study, the codes its last network reaches at its check
lengths and at each N (8, 16 and 64 by default), the codes new past the check lengths, and how
many failing and working runs whose check passed reach a new code (Fisher's exact test). `networks`
prints each finished run's last step's network, the one its result's works describes (to pass to
`certify` and `check`). `verdicts` is the findings' tables: per cell, the runs working (W10),
certified (the strict certificates' directory) and robustly certified (the branched one's), each with
its Wilson interval (D4); the runs by works and class; and with the long tests' directory (each
network's `check --json` as <run>-step-<j>.json), per class the networks tested and every failure.
`probe` is a study's budget before its launch (wr.probe): its first run's network trained for
--epochs at each of --steps (default 8 and the last), each epoch's training and screen seconds, the
whole check at the last step's longest length, a certificate for a snapped state, and the process's
peak memory after each (measurement.md, section 2: size workers from it and from results' peak_mb).
`diagnose` is what each saved step's network carries (indrajala_ml/studies/window_diagnostics.py):
its accuracy per answer class and each factor's share of each of the target's coordinates, their
zero tests and the answer, written as step-<j>.diagnosis.json beside the network and printed a line
each, a step already diagnosed skipped; --follow keeps diagnosing each new step as it is saved
until the machine's lock is free (the study ended), for deciding early whether a run is going
anywhere. It takes no lock: run it beside the study, one core.

OPENBLAS_NUM_THREADS=1, as every study.

Findings (stage 9), on jebel (i7-9700K), indrajala-ml 6b94933, crate 8c20e510c9a9,
`scripts/window_recurrent_matrix.json`: the curriculum arm, `k` 1, 2, 4 x `s` 4, 16 x `m` 4, 8,
16, 5 seeds, 3 steps (1 window, then 1-2, then 1-3), at most 200 epochs a step, patience 30, the
catalogue's default sizes, each step's test pass up to its first failing length (W12), 6 workers.
90 runs, none raised. Wall time 2026-10-12 21:43 to 2026-10-13 05:14 (7.5 h; 43.4 h of runs, 24.6
of them training and 18.8 testing); a run took 2-67 minutes (median 27). The results are on jebel
and pyramidon in `data/window-recurrent/matrix/results/` (90 files, listed with their sha256 in
`data/window-recurrent/matrix/results.sha256`, itself sha256 407d6ec3...7941cab9), every step's
network beside them in `models/` (156 MB), and the log in `data/window-recurrent-matrix.log`
(sha256 b47626f0...68a7dd59).

- Half the runs work (W10: a break of 16 columns or more): 45 of 90. 43 of them pass every
  property at every test length up to 64 columns, after training on at most 3 windows: 3, 6 and 12
  columns at `k` 1, 2 and 4, window ratios of 21.3, 10.7 and 5.3. The other two break at 24 and 32
  (`k` 4, one B1 case in 100,000 wrong at the next length). In every working run the carry reaches the
  whole length (`L*` = n at every test length).
- It takes the recurrence: no run works after step 1, which trains one window and so never hands
  a state on; 9 work from step 2 (2 windows; at `k` 1, two columns of training add 64), 36 more
  from step 3, and none that worked at step 2 stopped working at step 3. A working run's step 3
  passed its check in a median of 2 epochs.
- Failure is mostly all or nothing: 43 of the 45 failing runs fail at 8 columns already, most
  across B1, A2, M2-M5 (their step 3 best screens: median 0.924). A few are near misses: two `k` 4
  runs break at 12 (their trained length) with 1-2 cases in 100,000 wrong at 16, and one `k` 2 run
  misses 4 of B1's 100,000 at 8 (and a few of A1's, M3's and M5's).
- Passing the check isn't working at `k` 1: 5 runs passed their step 3 check (every triple at 1-3
  columns) and still got 0.2-8.3% of B1 wrong at 8.
- By axis: `k` 1, 2, 4 work in 16, 11 and 18 of 30 runs; `s` 4 and 16 in 24 and 21 of 45 (the
  state's width barely matters); `m` 4, 8 and 16 in 10, 16 and 19 of 30 (more codes, more working
  runs). No cell works on every seed: the best, 4 of 5, are `k` 1 `s` 16 `m` 8, `k` 4 `s` 4 `m` 16
  and `k` 4 `s` 16 `m` 16; the worst, `k` 2 `s` 16 `m` 4, on none. Within a cell the seed decides,
  so what fails is the optimisation, not the capacity: every cell has a state that can hold a
  carry (3 values), and 6 working runs use exactly 3 codes.
- The codes used by working runs: 3-11 (median 5) over the lengths tested, so a working network
  uses more codes than the carry's 3 values; the probes (stage 10) say what the others hold.
- Cost: a step 3 epoch trains in a median 6.8 s; a working run's step 3 test pass (every length)
  took a median 17.6 minutes, a failing one's (length 8 alone) about a minute.

Per cell, its seeds working and each seed's break (in columns, seeds 0-4) after steps 2 and 3, and
the codes the working seeds used:

| k | s | m | parameters | seeds working | breaks after step 2 | breaks after step 3 | codes used (working) |
|---|---|---|---|---|---|---|---|
| 1 | 4 | 4 | 7,683 | 2/5 | 0/64/64/0/0 | 0/64/64/0/0 | 4,4 |
| 1 | 4 | 8 | 7,699 | 3/5 | 0/0/0/0/0 | 0/64/64/0/64 | 8,6,7 |
| 1 | 4 | 16 | 7,731 | 2/5 | 0/0/0/0/0 | 0/64/64/0/0 | 11,11 |
| 1 | 16 | 4 | 9,651 | 2/5 | 64/0/0/64/0 | 64/0/0/64/0 | 4,4 |
| 1 | 16 | 8 | 9,715 | 4/5 | 0/0/64/0/0 | 64/64/64/0/64 | 6,4,4,7 |
| 1 | 16 | 16 | 9,843 | 3/5 | 0/0/0/0/0 | 64/64/64/0/0 | 7,7,6 |
| 2 | 4 | 4 | 7,699 | 1/5 | 0/0/0/0/0 | 0/0/64/0/0 | 3 |
| 2 | 4 | 8 | 7,715 | 3/5 | 0/64/0/0/0 | 0/64/64/0/64 | 4,6,4 |
| 2 | 4 | 16 | 7,747 | 3/5 | 0/0/0/0/0 | 0/64/0/64/64 | 8,7,5 |
| 2 | 16 | 4 | 9,667 | 0/5 | 0/0/0/0/0 | 0/0/0/0/0 | - |
| 2 | 16 | 8 | 9,731 | 1/5 | 0/0/0/0/0 | 64/0/0/0/0 | 6 |
| 2 | 16 | 16 | 9,859 | 3/5 | 0/0/0/0/0 | 0/0/64/64/64 | 9,4,8 |
| 4 | 4 | 4 | 7,731 | 3/5 | 8/0/64/0/64 | 64/0/64/0/64 | 4,3,3 |
| 4 | 4 | 8 | 7,747 | 3/5 | 0/0/12/0/0 | 0/64/64/12/64 | 4,5,5 |
| 4 | 4 | 16 | 7,779 | 4/5 | 0/0/0/0/8 | 64/64/12/32/64 | 7,5,7,7 |
| 4 | 16 | 4 | 9,699 | 2/5 | 0/0/0/0/8 | 0/64/0/0/64 | 3,3 |
| 4 | 16 | 8 | 9,763 | 2/5 | 64/0/0/0/0 | 64/0/0/64/0 | 4,3 |
| 4 | 16 | 16 | 9,891 | 4/5 | 8/0/0/0/0 | 64/24/64/64/0 | 5,6,4,6 |

Findings (the control, W16 and W30), on jebel, indrajala-ml 1dd22f9, crate 8c20e510c9a9,
`scripts/window_recurrent_control.json`: the all-at-once arm (one step on window counts 1-3
together, at most 200 epochs) at the matrix's three best cells, seeds 0-4, the matrix's settings
otherwise. 15 runs, none raised. Wall time 2026-10-13 05:28 to about 06:29 (5.3 h of runs; a run
took 1-33 minutes, median 23). The results are on jebel and pyramidon in
`data/window-recurrent/control/results/` (15 files, listed with their sha256 in
`data/window-recurrent/control/results.sha256`, itself sha256 6eed795f...a6facf9b), the networks
beside them in `models/` (9.2 MB), and the log in `data/window-recurrent-control.log` (sha256
913e996f...c52662dd).

- The curriculum isn't needed here: the control works as often, 12 of 15 runs against the
  curriculum's 12 of 15 at the same cells and seeds. At `k` 4 it works on all 10 (curriculum 8 of
  10; Fisher's exact test p 0.47); at `k` 1 `s` 16 `m` 8 on 2 of 5 (curriculum 4 of 5; p 0.52).
  At 5 seeds neither difference is decided; the claims workplan's E4 measures it at 20.
- It needs far fewer examples where it works: at `k` 4 it passes its check within 4-16 epochs,
  0.26-1.05 M examples (median 0.39 M), against the curriculum's 4.8-12.3 M over its three steps
  for the runs that work (median 6.4 M), about 16 times fewer. The curriculum spends 78-99% of
  its examples on its first two steps (one window, then one or two; 61 of the matrix's 90 runs
  stop step 1 on the plateau), and once it trains on 1-3 windows it passes in a few epochs, as the
  control does from the start.
- Which seeds work isn't shared between the arms: at `k` 4 the curriculum's two failing seeds
  (seed 2 of `s` 4, seed 4 of `s` 16) work all at once, and at `k` 1 the arms fail on different
  seeds (all at once: 0, 2, 4; curriculum: 3).
- Exact on short sums still isn't generalising: `k` 1's seed 2 passed its check (every sum of 1-3
  columns) in 3 epochs and failed at 8, as five of the matrix's `k` 1 runs did.

| cell | all at once: working, breaks (seeds 0-4) | epochs | curriculum: working, breaks after step 3 |
|---|---|---|---|
| `k` 1 `s` 16 `m` 8 | 2/5, 0/64/0/64/0 | 42, 1, 3, 3, 46 | 4/5, 64/64/64/0/64 |
| `k` 4 `s` 4 `m` 16 | 5/5, 64/64/64/48/64 | 13, 6, 7, 5, 6 | 4/5, 64/64/12/32/64 |
| `k` 4 `s` 16 `m` 16 | 5/5, 64/64/64/64/64 | 5, 4, 16, 6, 9 | 4/5, 64/24/64/64/0 |

Findings (E1 of docs/window-recurrent-claims-workplan.md: the certificate and the long tests), on
pyramidon: `certify` over every network the matrix and the control tested (the matrix's step 3, the
control's step 1: 105), 4-7 workers. Two verdicts (the claims workplan's D3): **robustly
certified**, every near tie (snap ratio under 1.05) branched both ways, indrajala-ml 094a7b1
(#703), 2026-10-13 about 07:50 to 08:26, in `data/window-recurrent/certificates/` (105 files,
listed with their sha256 in `data/window-recurrent/certificates.sha256`, itself sha256
67a4e6b7...9550f05d), log `data/window-recurrent-certificates.log` (sha256 f8c38080...56ae98a1); and
**certified**, the network's own snaps only (`--snap-ratio 1`), indrajala-ml 35175ea (#704; the
certificate unchanged but for the option), 08:30 to 09:22 in two parts, in
`data/window-recurrent/certificates-strict/` (`certificates-strict.sha256`, sha256
299675b8...a924fc18), log `data/window-recurrent-certificates-strict.all.log` (the two parts
joined; sha256 e1b8e612...3dcce757). 105.9 M window evaluations each way; 3.4 h of worker time
branched, 2.9 h strict; a `k` 4 network 115-689 s branched, 74-759 s strict.

- No failing network is certified, either way: of the 48 networks that break before 16 columns
  (W10's failing), none: here the certificate passes nothing the tests fail.
- Of the 57 working networks (W10), 52 are certified and 48 robustly certified (matrix 42 and 38
  of 45, control 10 and 10 of 12). Every working `k` 1 network is certified both ways; by `k` 1, 2
  and 4, working 18, 11 and 28, certified 18, 11 and 23, robustly 18, 8 and 22.
- Every certified network's machine minimises to 3 states, the carry automaton, while reaching 3-11
  codes (median 5; exactly 3 in 9 certified networks, 8 robustly): the extra codes are equivalent
  copies of a carry value. The smallest digit lead of a certified network
  is 0.153 (a logit difference).
- 16 robust certificates rest on near ties, up to 116,944 a network, at snap ratios down to
  1.0000002: real, not duplicate codes (each codebook's closest pair is 0.02-0.3 apart). The 4
  networks certified but not robustly (matrix `k2-s16-m16-seed3` and `seed4`, `k2-s4-m8-seed2`,
  `k4-s16-m4-seed1`) are correct as they compute and wrong only when a near tie (ratio
  1.006-1.03) snaps the other way.
- **The sampled tests overstate working (C1 refuted for W10's "works", D13).** 5 working networks
  aren't certified: 3 break before 64 columns, as their tests said (matrix `k4-s16-m16-seed1` at
  24, `k4-s4-m16-seed3` at 32, control `k4-s4-m16-seed3` at 48), and **2 passed every test to 64
  columns and are wrong**: control `all-at-once-k4-s16-m16-seed2` (1559 + 6329 + 5231 = 13,119
  at 8 columns; the network says 19,680) and matrix `curriculum-k4-s4-m8-seed2` (514107 + 167355
  + 27153 = 708,615 at 12; it says 531,468), each confirmed by running the network on
  `wr.example(triple, n - k, k, 0)`, `n` the certificate's length (a carry counterexample shows only at alignment 0, which the
  harness's hash picks for few cases).
- The failing networks' counterexamples (strict): 42 a wrong carry, 11 a wrong digit, after 1-4
  windows (22, 18, 6 and 7 networks), so every failure shows within 4 windows. Their machines
  minimise to 2-9 states.

The long tests (D9), on jebel: `check --n 128 256 1024 --properties B1 M3 --doubling` on the 59
networks certified or near a miss (a break of 12 or more): the 52 certified, the 2 counterexample
networks and 5 more not certified. B1's 100,000 cases at each length and M3's 1,000 at each
distance 1, 2, 4, ... `n`: 328,000 cases a network, 19.35 M in all. indrajala-ml e1b5be3 (#704's
branch; its package as 35175ea's), 7 workers, 2026-10-13 09:38 to 14:00, in
`data/window-recurrent/long-tests/` (a `.json` and a `.log` a network, listed with their sha256 in
`data/window-recurrent/long-tests.sha256`, itself sha256 4f3542f4...9666ec77), log
`data/window-recurrent-long-tests.log` (sha256 c5a5869d...35b49106; its "exit" figures are
always 0, a shell slip: read each network's `.log` and `.json`).

- **Every certified network passes every case**: the 48 robustly certified and the 4 certified
  only strictly, 17.06 M cases, none wrong. The certificate is borne out where it can be tested.
- Every network that fails a long test is one the certificate rejects: the 5 not certified fail
  B1 at every length (1-453 of 100,000 at 1,024), but for matrix `k4-s4-m16-seed3` at 128,
  and one of them an M3 case too (matrix `k4-s16-m16-seed1` at 256).
- **The long tests miss what the certificate finds.** Of the 2 counterexample networks, control
  `all-at-once-k4-s16-m16-seed2` fails 1 case of 328,000 (B1 at 1,024) and matrix
  `curriculum-k4-s4-m8-seed2` passes all 328,000, though it is wrong at 12 columns: a sampled test,
  however long, doesn't establish working.

Per cell, its networks working (W10), certified and robustly certified, of 5:

| arm | k | s | m | working | certified | robustly certified |
|---|---|---|---|---|---|---|
| curriculum | 1 | 4 | 4 | 2 | 2 | 2 |
| curriculum | 1 | 4 | 8 | 3 | 3 | 3 |
| curriculum | 1 | 4 | 16 | 2 | 2 | 2 |
| curriculum | 1 | 16 | 4 | 2 | 2 | 2 |
| curriculum | 1 | 16 | 8 | 4 | 4 | 4 |
| curriculum | 1 | 16 | 16 | 3 | 3 | 3 |
| curriculum | 2 | 4 | 4 | 1 | 1 | 1 |
| curriculum | 2 | 4 | 8 | 3 | 3 | 2 |
| curriculum | 2 | 4 | 16 | 3 | 3 | 3 |
| curriculum | 2 | 16 | 4 | 0 | 0 | 0 |
| curriculum | 2 | 16 | 8 | 1 | 1 | 1 |
| curriculum | 2 | 16 | 16 | 3 | 3 | 1 |
| curriculum | 4 | 4 | 4 | 3 | 3 | 3 |
| curriculum | 4 | 4 | 8 | 3 | 2 | 2 |
| curriculum | 4 | 4 | 16 | 4 | 3 | 3 |
| curriculum | 4 | 16 | 4 | 2 | 2 | 1 |
| curriculum | 4 | 16 | 8 | 2 | 2 | 2 |
| curriculum | 4 | 16 | 16 | 4 | 3 | 3 |
| all at once | 1 | 16 | 8 | 2 | 2 | 2 |
| all at once | 4 | 4 | 16 | 5 | 4 | 4 |
| all at once | 4 | 16 | 16 | 5 | 4 | 4 |

Findings (E7 of docs/window-recurrent-claims-workplan.md: modular division, C7), trained on jebel
(i7-9700K), indrajala-ml 67fc12f, crate 8c20e510c9a9, 6 workers, 2026-10-13 14:58 to 20:42
(`window_recurrent_e7_p5.json` then `_p7.json`: 2 `p` x 2 arms x `k` 1 and 4 x `m` 32 and 64 x 5
seeds, `s` 16, the matrix's settings). Results and models in `data/window-recurrent/e7-p5/` and
`e7-p7/` (listed with their sha256 in `data/window-recurrent/e7-p5.sha256`, itself sha256
6ca9ee8c...a992317f, and `e7-p7.sha256`, 10d0b5bd...0f5a2b7a), copied to pyramidon and checked;
logs `data/window-recurrent-e7-p5.log` (196cd344...0776527a) and `-p7.log` (2c06d807...fbbc1349).
Certified on pyramidon at indrajala-ml 9c50fe6, 3 workers, both verdicts (D13), in each
directory's `certificates/` and `certificates-strict/` (`certificates.sha256`: p 5 6d0b3e11...46d4cb7d,
p 7 4cae99c0...5426ada6; `certificates-strict.sha256`: p 5 43c9d4af...88c26033, p 7
dc612ff5...a5d4eff5); about 140 s a pass of 40 networks. No long tests: none certified (D22).

- **C7 is refuted as registered**: no run works and no network certifies, either way, at either
  `p`: 0 of 40 at `p` 5 and 0 of 40 at `p` 7 (Wilson 95%: 0.00-0.09 each), every cell 0 of 5.
- **E7's design was flawed** (found during its p 7 runs; the claims workplan's E7, D23): the
  automata minimise to exactly 25 and 49 states, so `p` 7 at `m` 32 (20 runs) can't be the
  automaton; they ran as registered, a negative control, and fail as they must. The other cells
  had 1.3 and 2.6 times the states' codes, the low end of what worked on addition.
- **At `k` 4 no run fits its training lengths**: best screens 0.11-0.34 at every `m` and `p`;
  the certificates find a wrong next state in the first window of every network.
- **At `k` 1 every run is a near miss at length 3**, the first length that uses every transition
  (every state is reachable within 2 columns): the curriculum passed lengths 1 and 2 every time
  and plateaued (never the budget: D26) on length 3, whose U1 is all 729 pairs; at `p` 5 the
  networks got 10-342 of them wrong, median 172 at `m` 32 and 85 at `m` 64 (doubling the codes
  about halved the errors; best, all-at-once `m` 64 seed 0, 10 wrong, a best screen of 0.996); at
  `p` 7, 67-500, median about 136 at `m` 64. Every certificate's counterexample is a wrong next
  state after 2 windows (one `p` 7 network's after 1).
- The `k` 1 `m` 64 networks reach codes for most of the automaton's states (the certificate's
  `phi` covers 23-25 of `p` 5's 25 and 31-46 of `p` 7's 49) from 37-60 codes, but their
  transitions between them don't close: the codes are there, the dynamics aren't. Every run
  used almost all its codes (a median of 31 of 32 and 62-63 of 64 per cell), as addition's
  working runs did (14 of 16).
- E7' (the claims workplan) is the test E7 meant to be: `k` 1 at 5 and 10 times the states'
  codes, 10 seeds a cell.

Findings (E7' of docs/window-recurrent-claims-workplan.md: modular division with enough codes,
C7'), trained on jebel, indrajala-ml 9c50fe6, crate 8c20e510c9a9, 6 workers, 2026-10-13 20:45 to
23:56 (`window_recurrent_e7_prime_p5.json` then `_p7.json`: 2 `p` x 2 arms x 2 `m` x 10 seeds,
`k` 1, `s` 16, E7's settings; `m` 128 and 256 at `p` 5, 256 and 512 at `p` 7, about 5 and 10
times the states). Results and models in `data/window-recurrent/e7-prime-p5/` and `-p7/`
(`e7-prime-p5.sha256`, sha256 67f56ab0...05dc44d8; `e7-prime-p7.sha256`, 69c4efe2...4fd8c79e),
copied to pyramidon and checked; logs `data/window-recurrent-e7-prime-p5.log` (eab06819...c7a1b8ac)
and `-p7.log` (a258d10a...8da00eea). Certified on pyramidon at indrajala-ml cf6ac86, both verdicts
(`certificates.sha256`: p 5 de4a138c...10c88e98, p 7 1453332d...d9b5cc27;
`certificates-strict.sha256`: p 5 eb3530ac...3a9703aa, p 7 cd76b51e...aca8370f). No long tests:
none certified.

- **C7' is refuted as registered**: no run works and no network certifies, either way: 0 of 40
  at `p` 5 and 0 of 40 at `p` 7 (Wilson 95%: 0.00-0.09 each), every cell 0 of 10.
- **The codes are no longer the limit.** The networks now hold codes for (nearly) every state
  of the automaton: the certificate's `phi` covers all 25 of `p` 5's states in every all-at-once
  `m` 256 network, and 47-49 of `p` 7's 49 at all-at-once `m` 256, from 30-154 codes reached. Yet
  79 of the 80 counterexamples are a wrong next state (76 after 2 windows, 2 after 1, 1 after 3),
  and one a wrong output: the states are represented, the transitions between them aren't exact.
- More codes cut the errors at length 3, the first to use every transition (U1, all 729 pairs),
  but none reach zero. Median and best pairs wrong, `k` 1, against E7's `m` 64:

| `p` | arm | E7 `m` 64 | E7' 5 times | E7' 10 times |
|---|---|---|---|---|
| 5 | all-at-once | 88 (best 10) | 25 (8) | 28 (4) |
| 5 | curriculum | 79 (39) | 93 (43) | 38 (23) |
| 7 | all-at-once | 195 (113) | 60 (24) | 51.5 (28) |
| 7 | curriculum | 113 (67) | 106 (41) | 70 (27) |

- The best network, `p` 5 all-at-once `m` 256, is 4 pairs of 729 from exact at length 3; from 5
  to 10 times the codes the all-at-once medians stop improving (25 to 28, 60 to 51.5).
- **6 of E7''s 160 steps stopped on the budget while still improving** (best screen within the
  last 21 epochs of 200: five at `p` 7, four of them curriculum step 3 at `m` 256-512, best
  screens 0.94-0.99). D26 kept E7's budget (no `k` 1 budget stop in E7); E7' shows a few runs it
  cut short, none near exact (best screens under 0.99).

Findings (E8 of docs/window-recurrent-claims-workplan.md: the modular-sum ladder's first rungs,
C8), trained on pyramidon (Ryzen 7 3700U), indrajala-ml b886001, crate 8c20e510c9a9, 3 workers,
2026-10-14 05:50 to 06:19 (`window_recurrent_e8_q4.json` then `_q5.json`: `q` 4 and 5 x 2 arms x
10 seeds, `k` 1, `s` 16, `m` 32, E7''s settings). Results and models in
`data/window-recurrent/e8-q4/` and `-q5/` (`e8-q4.sha256`, sha256 011ee2de...596754fd;
`e8-q5.sha256`, 8ec6a2ae...ebb5ca80), logs `data/window-recurrent-e8-q4.log` (51504a22...646b3d13)
and `-q5.log` (36923d55...d9b94fce). Certified on pyramidon, both verdicts
(`certificates.sha256`: q 4 184bc355...1590b51e, q 5 0fc8b159...d3f8c535;
`certificates-strict.sha256`: q 4 852c0b1b...90a1eafb, q 5 9ced897f...bd3031ff); the certified
network long-tested (U1 and S1 at 128, 256 and 1,024; `e8-q4/long-tests.sha256`,
a4ca4818...ee347172).

- **C8 holds at `q` 4, weakly; not at `q` 5.** One network certifies (both verdicts), q 4
  all-at-once seed 3: its 8 codes minimise to the 4-state automaton, and it passes all 312,000
  long-test cases. 1 of 20 at `q` 4 (Wilson 0.01-0.24), 0 of 20 at `q` 5 (0.00-0.16).
- **The failures are memorising, not fitting** (the opposite of E7's): 37 of the 40 runs passed
  every check at lengths 1-3, most in 2-20 epochs, and failed at the first test length (8). Their
  codes minimise to machines mostly larger than the task's: at `q` 4, 17 of the 19 failing
  networks' machines have 6-15 states (one 2, one 4), at `q` 5 all 20 have 7-22, against 4 and 5;
  the counterexamples come mostly after 3-4 windows, just past the training lengths.
- **Why the design allowed it** (D23's exclusion bound, added after E8): a network of `m` codes
  exactly right on every input up to length `L` is the `q`-state machine only if
  `L >= m + q - 1` (two machines of `n` and `n'` states that agree on every input up to `n + n' - 1`
  are equivalent). E8 trained to 3 columns with 32 codes, where 35-36 were needed; so did every
  earlier experiment (addition's `k` 1 needed 6-18, its `k` 4 trained to 12). E8b is E8 trained
  past the bound.

Findings (E3 and E4 of docs/window-recurrent-claims-workplan.md, its stage 7: the code count and the
curriculum against all at once, C3-C5; C0's reruns), trained on jebel (i7-9700K), indrajala-ml
cf6ac86, crate 8c20e510c9a9, 6 workers, finished 2026-10-14 12:41 (`window_recurrent_e3_e4_trimmed.json`,
D34: `k` 4 `s` 4, the curriculum at `m` 3, 4, 6, 8 and 16 and all at once at `m` 16, seeds 0-19,
120 runs). Results and models in `data/window-recurrent/e3-e4/` (`e3-e4.sha256`, sha256
448dd136...652dda8a), log `data/window-recurrent-e3-e4.log` (52f0b1df...133d5f0a). Every run's
last network certified on pyramidon at 9dce80e, both verdicts (`e3-e4/certificates.sha256`,
37c1c694...42a38caa; `certificates-strict.sha256`, a68ed4f2...3b74f7a3; logs `certify-branched.log`
60f328ea...b0d93bc8, `certify-strict.log` f8ffc9b7...7dc7a7f3). The registered tests are D4's on
the working rate (W10, exact to 16 columns and past); the certified rate beside them is a
measurement. Holm's correction over C2-C4 and C6 waits for C2 and C6 (stage 8); the p values here
are raw.

| arm | `m` | working | certified (strict) | robustly certified | working, not certified |
| --- | --- | --- | --- | --- | --- |
| curriculum | 3 | 3/20 (0.05-0.36) | 3/20 (0.05-0.36) | 2/20 | 0 |
| curriculum | 4 | 11/20 (0.34-0.74) | 10/20 (0.30-0.70) | 10/20 | 1 |
| curriculum | 6 | 8/20 (0.22-0.61) | 6/20 (0.15-0.52) | 6/20 | 2 |
| curriculum | 8 | 11/20 (0.34-0.74) | 7/20 (0.18-0.57) | 6/20 | 4 |
| curriculum | 16 | 13/20 (0.43-0.82) | 8/20 (0.22-0.61) | 8/20 | 5 |
| all at once | 16 | 19/20 (0.76-0.99) | 12/20 (0.39-0.78) | 11/20 | 7 |

- **C0 holds: 15 of 15 reruns reproduced exactly** (the matrix's `k` 4 `s` 4 runs, 6b94933,
  against their reruns here: results up to timings, and every step's weights, optimizer state and
  generator, `compare`).
- **C3, as registered, holds: the working rate rises with `m`** (Cochran-Armitage over `m` 3-16,
  scores `m`: z 2.45, p 0.014). **But the certified rate doesn't** (z 0.79, p 0.43): from `m` 4 on
  it stays near 0.3-0.5, and the extra codes buy working networks that aren't the automaton
  (working but not certified: 0, 1, 2, 4, 5 of 20 from `m` 3 to 16). Under C1's restatement
  (certified is what works at every length), extra codes don't make training find the automaton
  more often; `m` 3, exactly the automaton's size, is the one low cell (3 of 20).
- **C4, as registered, is refuted: all at once works more often** (19 of 20 against the
  curriculum's 13 of 20 at `m` 16, Fisher's exact p 0.044), on far fewer examples (working runs'
  medians 393,216 against 7,798,784, Mann-Whitney p 2e-6, every all-at-once run below every
  curriculum one). Certified, the arms don't differ (12 against 8 of 20, p 0.34): all at once's
  extra working runs are mostly ones the certificate fails (7 of its 19).
- **Every certified network, in every cell, minimises to the 3-state carry automaton**, from 3
  reachable codes (at `m` 3) to 10 (all at once). Of the 19 working networks not certified, 14 give
  a wrong digit and 5 a wrong carry on some reachable code and window that the sampled tests to 64
  columns missed; none fails only on a label's margin.
- **C5 (a measurement):** the rates above with their Wilson intervals; at 20 seeds a cell they
  span about 0.4 each.
- **The steps:** the curriculum's step 1 (one window) plateaus in all 100 runs; every working run
  passes step 3 (lengths 1-3), 19 of the 46 at step 2 already. The failures differ with `m`: at `m`
  3-6 they never fit (all three steps plateau, 36 of the 38 failures there), at `m` 8-16 they mostly
  fit (12 of 16 pass step 3) and break at 8-12 columns: memorising, as the exclusion bound (D23)
  predicts for training to 3 windows (12 columns) with 8-16 codes.

Findings (E6 of docs/window-recurrent-claims-workplan.md, re-registered by D37: the window without
its transformer, as an existence test), trained on pyramidon (Ryzen 7 3700U), indrajala-ml 9114f21,
crate abcba3d7a30e, 3 workers (memory: E8c's long tests ran beside it), 2026-10-14 17:06 to 17:26
(`window_recurrent_e6_linear.json`: the linear window, the embedding and output layer only, at `k`
1 `s` 16 `m` 16 on addition, the curriculum capped at 18 columns, the bound `m + q - 1`, each step
certified, the study stopped at its first certified run, 6 seeds). Results, models and per-step
certificates in `data/window-recurrent/e6-linear/` (56 files, `e6-linear.sha256`, sha256
716761f3...62f2e5da, run from `data/window-recurrent`), log `data/window-recurrent/e6-linear.log`
(9a6c848b...cba8648d; its run ended in the report's crash on runs stopped before their first epoch,
fixed with these findings, after every result was written). Strict certificates of the two
certified networks in `e6-linear/certificates-strict/` (listed in the same manifest).

- **The window's transformer isn't needed at `k` 1: at least 2 of 6 runs certify.** Seed 1 at step
  3 (trained to 3 columns) and seed 2 at step 4 (4 columns), both verdicts (the harness's per-step
  certificate branches near ties; the strict one, `--snap-ratio 1`, agrees), each minimised to the
  3-state carry automaton from 5 and 8 codes reached, 0 wrong outputs and states, no near ties, snap
  ratios at least 2.99 and 1.37, logit leads at least 5.01 and 2.29. Both pass every test length to
  64 columns. "At least": seeds 1 and 2 certified in parallel before the stop reached them.
- **A network of 2,963 parameters** (the transformer window's `k` 1 networks: 9,651-9,843 at `s`
  16) certifies, so at one column a window the GRU does the whole step; the FFN window (D37's
  fallback) wasn't run.
- The others: seed 0 fitted steps 1-3 and plateaued from step 4 until stopped at step 7; seeds 3-5
  were stopped before their first epoch (3 workers). No rate is claimed (D37).
"""

from __future__ import annotations

import argparse
import concurrent.futures
import dataclasses
import functools
import json
import multiprocessing
import sys
import time
from pathlib import Path

from indrajala_ml.data import addition_data as ad
from indrajala_ml.measurement import machine_lock
from indrajala_ml.model.persistence.load_network import load_network
from indrajala_ml.studies import window_diagnostics as wd
from indrajala_ml.studies import window_recurrent_certificate as wc
from indrajala_ml.studies import window_recurrent_study as wr
from indrajala_ml.studies import window_tasks as wt


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    for name in ("run", "status", "report", "networks"):
        command = commands.add_parser(name)
        command.add_argument("config", type=Path)
        if name == "report":
            command.add_argument("--md", type=Path, help="write the report to this file too")
        if name == "run":
            command.add_argument(
                "--workers", type=int, help="the config's worker count, overridden (a study moved between machines)"
            )
            command.add_argument(
                "--arm", choices=wr.ARMS, help="this arm's runs only (a study's arms split between machines)"
            )
    check = commands.add_parser("check")
    check.add_argument("network", type=Path)
    check.add_argument("--n", type=int, nargs="+", required=True, help="the lengths to check")
    check.add_argument("--task", default=wt.ADDITION.name, help="the network's task (window_tasks.task_named)")
    check.add_argument("--properties", nargs="+", help="these properties only (default: all but the held-out one)")
    check.add_argument("--sampled", type=int, default=ad.SAMPLED, help="the sampled properties' sample")
    check.add_argument("--per-stratum", type=int, default=ad.PER_STRATUM, help="the stratified properties' per stratum")
    check.add_argument("--doubling", action="store_true", help="M3 (S2) at 1, 2, 4, ..., n only")
    check.add_argument("--json", type=Path, help="write each length's results and seconds to this file")
    long = commands.add_parser("long-test")
    long.add_argument("networks", type=Path, nargs="+")
    long.add_argument("--task", default=wt.ADDITION.name, help="the networks' task (window_tasks.task_named)")
    long.add_argument("--n", type=int, nargs="+", default=list(wr.LONG_LENGTHS), help="the lengths")
    long.add_argument("--properties", nargs="+", help="these properties only (default: all but the held-out one)")
    long.add_argument("--out", type=Path, required=True, help="write each network's results here")
    long.add_argument("--workers", type=int, default=1)
    certify = commands.add_parser("certify")
    certify.add_argument("networks", type=Path, nargs="+")
    certify.add_argument("--task", default=wt.ADDITION.name, help="the networks' task (window_tasks.task_named)")
    certify.add_argument("--out", type=Path, help="write each certificate here, as <run>-<step>.json")
    certify.add_argument("--workers", type=int, default=1)
    certify.add_argument(
        "--snap-ratio", type=float, default=wc.SNAP_RATIO, help="near ties branch under this ratio (1: strict)"
    )
    compare = commands.add_parser("compare")
    compare.add_argument("original", type=Path, help="a study's directory (results/ and models/)")
    compare.add_argument("rerun", type=Path, help="another study's directory, rerunning some of its runs")
    explore = commands.add_parser("explore")
    explore.add_argument("config", type=Path)
    explore.add_argument(
        "--n", type=int, nargs="+", default=list(wr.EXPLORED_LENGTHS), help="the lengths past training"
    )
    explore.add_argument("--json", type=Path, help="write every run's codes to this file")
    explore.add_argument("--workers", type=int, default=1)
    probe = commands.add_parser("probe")
    probe.add_argument("config", type=Path)
    probe.add_argument("--steps", type=int, nargs="+", help="curriculum steps to time (default: 8 and the last)")
    probe.add_argument("--epochs", type=int, default=2, help="training epochs at each step")
    probe.add_argument("--json", type=Path, help="write the probe's figures to this file")
    diagnose = commands.add_parser("diagnose")
    diagnose.add_argument("config", type=Path)
    diagnose.add_argument("--run", nargs="+", help="these runs' keys only (default: every run)")
    diagnose.add_argument("--n", type=int, nargs="+", default=list(wd.LENGTHS), help="the lengths drawn")
    diagnose.add_argument("--cases", type=int, default=wd.CASES, help="cases drawn at each length")
    diagnose.add_argument("--follow", action="store_true", help="diagnose each new step until the lock is free")
    diagnose.add_argument("--interval", type=float, default=30.0, help="seconds between looks when following")
    verdicts = commands.add_parser("verdicts")
    verdicts.add_argument("config", type=Path)
    verdicts.add_argument("--certificates", type=Path, required=True, help="certify's branched certificates")
    verdicts.add_argument("--strict", type=Path, required=True, help="certify --snap-ratio 1's certificates")
    verdicts.add_argument("--long-tests", type=Path, help="each network's check --json, as <run>-step-<j>.json")
    args = parser.parse_args(argv)

    if args.command == "check":
        long = wr.long_test(
            args.network,
            wt.task_named(args.task),
            args.n,
            args.properties,
            args.sampled,
            args.per_stratum,
            args.doubling,
            log=lambda line: print(line, flush=True),
        )
        if args.json is not None:
            args.json.write_text(json.dumps(long, indent=1))
        return 0
    if args.command == "long-test":
        args.out.mkdir(parents=True, exist_ok=True)
        context = multiprocessing.get_context("forkserver")
        with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=context) as pool:
            test = functools.partial(
                wr.long_test_file, out=args.out, task_name=args.task, lengths=args.n, properties=args.properties
            )
            for line in pool.map(test, args.networks):
                print(line, flush=True)
        return 0
    if args.command == "certify":
        if args.out is not None:
            args.out.mkdir(parents=True, exist_ok=True)
        context = multiprocessing.get_context("forkserver")
        with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=context) as pool:
            for path, result in zip(
                args.networks,
                pool.map(
                    functools.partial(wc.certify_file, task_name=args.task, snap_ratio=args.snap_ratio), args.networks
                ),
                strict=True,
            ):
                name = wr.network_name(path)
                print(f"{name}: {wc.summary(result)}", flush=True)
                if args.out is not None:
                    (args.out / f"{name}.json").write_text(json.dumps(result, indent=1))
        return 0
    if args.command == "compare":
        compared = wr.compare_runs(args.original, args.rerun)
        for key, found in compared.items():
            print(f"{key}: " + ("reproduced" if not found else f"{len(found)} differ, e.g. {', '.join(found[:5])}"))
        differ = sum(bool(found) for found in compared.values())
        print(f"{len(compared)} compared, {differ} differ")
        return 1 if differ else 0
    study = wr.load_config(args.config)
    if args.command == "probe":
        steps = args.steps or sorted({min(8, study.settings.steps), study.settings.steps})
        figures = wr.probe(study.runs[0], study.settings, steps, args.epochs)
        for row in figures["epochs"]:
            print(
                f"step {row['step']} epoch {row['epoch']}: train {row['train_seconds']:.1f} s, "
                f"screen {row['screen_seconds']:.1f} s ({row['screen']:.3f}), peak {row['peak_mb']:.0f} MB",
                flush=True,
            )
        check = figures["check"]
        print(f"whole check at {check['n']}: {check['seconds']:.1f} s, peak {check['peak_mb']:.0f} MB")
        if "certificate" in figures:
            certificate = figures["certificate"]
            print(
                f"certificate: {certificate['seconds']:.1f} s, {certificate['states']} states, "
                f"peak {certificate['peak_mb']:.0f} MB"
            )
        if args.json is not None:
            args.json.write_text(json.dumps(figures, indent=1))
        return 0
    if args.command == "diagnose":
        runs = [spec for spec in study.runs if args.run is None or spec.key in args.run]
        while True:
            ending = args.follow and machine_lock.holder() is None  # looked at before the pass: none missed
            for spec in runs:
                if (study.models / spec.key).is_dir():
                    task = wt.task_named(spec.task)
                    for text in wd.diagnose_new(study.models / spec.key, task, load_network, args.n, args.cases):
                        print(text, flush=True)
            if not args.follow or ending:
                return 0
            time.sleep(args.interval)
    if args.command == "explore":
        context = multiprocessing.get_context("forkserver")
        with concurrent.futures.ProcessPoolExecutor(args.workers, mp_context=context) as pool:
            explored = [
                run
                for run in pool.map(functools.partial(wr.explore_run, study, lengths=args.n), study.runs)
                if run is not None
            ]
        for run in explored:
            new = ", ".join(f"{n}: {run['new'][str(n)]}" for n in args.n)
            print(f"{run['key']}: works {run['works']}, check passes {run['passes']}, new codes {new}", flush=True)
        print(wr.explore_summary(explored, args.n))
        if args.json is not None:
            args.json.write_text(json.dumps(explored, indent=1))
        return 0
    if args.command == "networks":
        for path in wr.last_networks(study).values():
            print(path)
        return 0
    if args.command == "verdicts":
        print(wr.verdicts(study, args.certificates, args.strict, args.long_tests))
        return 0
    if args.command == "run":
        if args.workers is not None:
            study = dataclasses.replace(study, workers=args.workers)
        if args.arm is not None:
            study = dataclasses.replace(study, runs=[spec for spec in study.runs if spec.arm == args.arm])
        with machine_lock.study():
            wr.run(study, log=lambda line: print(line, flush=True))
        print(wr.report(study))
    elif args.command == "status":
        print(wr.status(study))
    else:
        text = wr.report(study)
        print(text)
        if args.md is not None:
            args.md.write_text(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
