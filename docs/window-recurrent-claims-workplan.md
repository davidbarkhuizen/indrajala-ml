# Workplan: substantiating the window-recurrent adder's findings

**Status: written 2026-10-13, at the owner's request ("the work we would need to do to substantiate
our claims/findings"), after the matrix (stage 9 of
[window-recurrent-adder-workplan.md](window-recurrent-adder-workplan.md)) and while its control
ran. Decisions D1-D20 settled (D13 and D14 by the owner, the rest under the owner's standing
delegation), each with its options' pros and cons. Stages 1-5 done (stage 2 refined D3's snap margin: a near tie branches). E7' (C7'; D23-D26)
added 2026-10-13 at the owner's request, to run before stage 7, after E7's design flaw (p 7 at `m` 32 has fewer codes
than states), with D23 making a feasibility check a rule for every cell. E8 (C8; D27-D30), the
modular-sum ladder, added 2026-10-14 at the owner's request after E7' was refuted.** The literature it answers to is in
[window-recurrent-literature.md](window-recurrent-literature.md).

This plan is the window-recurrent workplan's stage 10 ("what follows, decided on the findings"),
written out as its own plan because it is larger than a stage: a certificate, one package change,
four experiments and a report. When it retires, the parent's stage 10 is done.

## Why

The matrix found that a transformer over a window of columns, handing a snapped GRU state from
window to window, learns base-3 addition that holds to 64 columns from at most 12 columns of
training. The literature review found that the headline (a learned recurrent carry
extrapolating) is long known, and the study is weaker than the literature where it overlaps:

- **Length:** tested to 64 columns; the Neural GPU (Kaiser & Sutskever 2016) to 2,000 bits,
  Rational Transductors (Mohri 2026) to 1,000 digits.
- **Seeds:** 5 per cell; Price et al. 2016 used about 15 per configuration.
- **No ablation** of the one ingredient that may be new: the snapping.

What may be new is narrower: the design (not found in the literature); that more codes make
training succeed more often although 3 suffice; and, because the state is snapped, the chance to
**certify** a learned adder correct at every length, not only test it. This plan turns each into
a claim with the evidence it needs, or drops it.

## The claims

Each claim is stated as it would be written if its experiment comes out as predicted, with the
experiment that tests it and what would refute it. The analysis rules (D4) are fixed here, before
any run.

| id | claim | experiment | refuted if |
| --- | --- | --- | --- |
| **C1** | A working network is a correct adder at **every** length: its snapped state makes it a finite transducer, and that transducer is the carry automaton | E1, the certificate | a working network fails the certificate (then C1 is "to 64 columns", as now) |
| **C2** | The snapping is what makes the recurrence extrapolate: the same network with a continuous state works far less often past its training lengths | E2, the ablation | the continuous arm's working rate at 1,024 columns is within D4's test of the snapped arm's |
| **C3** | More codes than the automaton needs make training find it more often | E3, the code count | no monotone trend over `m` (D4's trend test) |
| **C4** | The curriculum isn't needed: training on 1-3 windows at once works as often (the control: 12 of 15 against 12 of 15, on about 16 times fewer examples at `k` 4), so the recurrence, not the schedule, generalises | E4, the arms | the arms' working rates differ by D4's test (then the claim is the difference, whichever way) |
| **C5** | The seed decides: at a fixed configuration the working rate is a stable fraction, measured with its interval | E3 and E4's 20 seeds a cell | (a measurement, not a hypothesis: reported with intervals) |
| **C7** | The method isn't addition's: the same design learns modular division, `(a / b) mod p` of two long base-3 numbers, and its learned machine certifies exactly against the modular automaton (`p^2` states), at every length | E7, the second task (D14-D17) | no network certifies at either `p` |
| **C7'** | With enough codes the method learns modular division: at `k` 1 and about 5 and 10 times as many codes as the minimal automaton's states, some networks certify against it at every length, at both `p` | E7', modular division with enough codes (D23-D26) | no network certifies at either `p` |
| **C8** | The method learns a machine of a few states beyond addition's 3: the modular sum `(a + b) mod q` of two long base-3 numbers at `q` 4 and 5, certified exactly against its `q`-state automaton at every length; then, rung by rung, larger `q` | E8, the modular-sum ladder (D27-D30) | no network certifies at either `q` |
| **C6** | Exact on short sums isn't generalising: failing runs reach state codes at long lengths that training never produced (Buitrago Ruiz & Gu's unexplored states) | E5, from saved networks | failing runs' codes at 8 columns are all codes seen at the check lengths |

C0, before all of them: **the runs reproduce.** The harness is deterministic (seeded generators,
W26's order, one thread), so a rerun of a matrix run gives the same result file up to timings;
E3's first five seeds at the matrix's cells are exact reruns and are compared (D4).

## The experiments

### E1. The certificate (C1), and long tests

**Why it can be done.** A window's work is a function of the state it receives and its `k`
column tokens: the tokens get a learned `Position` over `k`, the state enters through the
`StateInput`s, and the window gives its `k` digits and the next state. With the state snapped,
the state received is exactly one of `m` codes, so the network is a finite-state transducer:
states the codes, input symbols the `27^k` windows (531,441 at `k` 4), outputs `k` digits. The
first window's state is the snapped learned start (W8). The whole machine can be enumerated.

**The certificate.** For a network: from the start code, breadth first over every window, collect
the reachable codes and, for each reachable code `q` and window `w`, the digits `D(q, w)` and the
next code `G(q, w)`. It passes when there is a map `phi` from reachable codes to carries {0, 1, 2}
with `phi(start) = 0` such that for every reachable `q` and every `w`:

- `D(q, w)` are the `k` sum digits of `w`'s columns with carry in `phi(q)`, and
- `phi(G(q, w))` is the carry out of `w`'s columns.

`phi` is found as the breadth-first search goes (each code's carry is fixed by the first window
that reaches it, then checked by every other); a conflict fails the certificate and names the
smallest counterexample. A pass proves the network adds correctly at every length and alignment:
an addition is a start state, a word of windows (the zero columns of W6-i and W9 are windows like
any other), and the outputs the automaton's. It also gives the machine itself: its minimised form
should be the 3-state carry automaton, with the codes mapping onto carries many to one.

**Numerical margin.** The certificate evaluates the windows in batches; the network in use may
evaluate them in other batches, and rounding can differ between them (BLAS order). So the
certificate also records, for every `(q, w)`, the snapping margin (the distance to the second
nearest code over the distance to the nearest) and each digit's logit margin (D3). A digit's
near tie fails the certificate. A snap's near tie branches (stage 2): every code within the ratio
is a next code too, explored and checked as the snapped one is, so a pass holds whichever way
rounding snaps the tie; the near ties are counted apart. No certificate rests on a tie.

**Cost** (measured on `pyramidon`, one core, 2026-10-13): about 31,000 window evaluations a
second (B1's 100,000 cases at 128 columns in 106 s, 33 windows each). A `k` 4 network is at most
16 x 531,441 = 8.5 M evaluations, about 5 minutes; `k` 1 and 2 take seconds. All 105 networks
(the matrix's 90 and the control's 15, each at its last step), about 3 hours on one core, under
an hour on six. Measured in stage 2: a `k` 4 network reaching 7 codes in 100 s, one reaching 9 in
127 s (about 37,000 evaluations a second); a `k` 1 network in under a second.

**Long tests, as an independent check.** On every certified network and every near miss (a break
of 12 or more), B1 (100,000 cases) at 128, 256 and 1,024 columns, and M3 at distances 1, 2, 4,
..., up to `n` (1,000 cases each, a subsampling of the strata, D9). If the certificate is sound,
every certified network passes these; a failure is a bug in the certificate or the margins.

**Analysis.** Per network: certified or not, its reachable codes, its minimised machine's size,
the smallest margin. Per cell: certified networks. Cross-tabulated against "works" (W10) from
the matrix: the prediction is that every working network is certified and no failing one is.

### E2. The snapping ablation (C2)

**Design.** The window-recurrent network with `codes=None`: the GRU's output is the next state,
unsnapped, and the codebook and commitment losses are dropped (D3, the package change of stage
4). Everything else the same: the cells of D5, the curriculum, the settings, seeds 0-19 (D4).
Its snapped counterpart is E3's runs at the same cells, `m` 16 and 8, the same seeds.

**Measures.** Per run: works (W10's break of 16 or more), the break, and B1 at 256 and 1,024
columns for every run whose break is 64. The prediction (C2): continuous states drift, so even
where they pass 64 they fail at 1,024 (Robinson 2020; Brandoit et al. 2026). The certificate
doesn't apply to a continuous state; the long tests are the evidence.

**Analysis.** Per cell, the working rate at 64 and at 1,024 columns of each arm, Fisher's exact
test between arms (D4).

### E3. The code count (C3, C5)

**Design.** The curriculum arm at the cells of D5, `m` in {3, 4, 6, 8, 16, 32} at `k` 4 `s` 4,
and {3, 4, 8, 16} at `k` 1 `s` 16 (D6), 20 seeds each (D4).

**Measures.** Per run: works; certified (E1's tool, run on every run's last network); codes used
and codes reachable; the epochs to pass.

**Analysis.** The working rate per `m` with its Wilson interval; the Cochran-Armitage trend test
over `m` per cell (D4). Whether the trend is optimisation, not capacity: at `m` 3 the codebook is
exactly the automaton's size, so a lower rate there with the same certified machines when it does
work says the extra codes help the search, not the solution. The codes used by working runs
against `m`: if working runs use more codes than 3 and the minimised machines are all the 3-state
automaton, the extra codes are redundant copies of carries.

**Trimmed (D34):** `k` 4 `s` 4 at `m` 3, 4, 6, 8 and 16 only; `m` 32 and the `k` 1 cells dropped.

### E4. Curriculum against all at once (C4, C5)

**Design.** The all-at-once arm (W16) at D5's cells, at the matrix's best `m` (16 at `k` 4 `s` 4,
8 at `k` 1 `s` 16), seeds 0-19. Its curriculum counterpart is E3's runs at the same `m`.

**Measures.** Works; certified; examples to pass (the curriculum's over its three steps, the
control's in its one, D7).

**Analysis.** Working rates and Fisher's exact test per cell; examples to pass compared by the
Mann-Whitney test over the working runs (D4). C4 holds if the rates don't differ; whatever the
result, the claim is written as the data show.

**Trimmed (D34):** `k` 4 `s` 4 `m` 16 only; the `k` 1 `s` 16 `m` 8 cell dropped.

### E5. Unexplored states (C6)

**Design.** No new training. For every failing run of the matrix and of E3 with a passing check
(exact at the training lengths), and as a comparison every working run: the codes its states use
over B1's first 1,000 cases at each check length and at 8, 16 and 64 columns (`state_codes`, as
the harness's codes used, W28), from the saved networks.

**Analysis.** The share of failing runs whose codes at 8 columns include a code never used at the
check lengths, against the same share for working runs (Fisher's exact test, D4). If C6 holds,
the intervention Buitrago Ruiz & Gu found (training from other sequences' final states) becomes
a candidate fix, an experiment for a later plan.

### E6. A baseline without the window's transformer (D8)

**Design.** The recurrence alone: `k` 1, the window's transformer replaced by its embedding and
output layer only (no attention blocks), so the GRU does the whole step, snapped, at `m` 4, 8
and 16, 20 seeds each, the curriculum. Asks whether the transformer inside the window does
anything at `k` 1, where it sees one column. As built (D21): the linear window at `m` 4, 8 and
16, and the FFN window (the blocks without attention) at `m` 8, 80 runs.

**Analysis.** Working rates against E3's `k` 1 runs at the same `m` (Fisher's exact test). The
windows of `k` 2 and 4 need the transformer to add inside the window, so the baseline is at `k` 1
only.

### E7. Modular division, the second task (C7; D14-D20)

**Design.** Two base-3 numbers `a` and `b`, read most significant digit first a column at a time
(a column's token is `3 a_i + b_i`, 9 values), and at every column the running answer
`(A * B^-1) mod p` of the prefixes read so far, or "undefined" while `B mod p` is 0: `p + 1`
classes. The last column's output is the answer. A finite transducer of `p^2` states
(`(A mod p, B mod p)`), with an output per column, the adder's shape, so the network is the
adder's with a 9-token embedding and `p + 1` output classes. `p` in {5, 7} (25 and 49 states);
the curriculum and the all-at-once arms; `k` 1 and 4; `m` 32 and 64 (more codes than states,
C3); 5 seeds a cell to start, as the matrix.

**Measures.** As the matrix's (works, the break, codes used), and E1's certificate against the
modular automaton, both verdicts (D13), and the long tests on every certified network.

**Analysis.** Per cell, the working rate and the certified rate; every certified machine's
minimised size against `p^2`; the codes used against the states needed.

**A flaw in E7's design (found 2026-10-13, during its p 7 runs).** "`m` 32 and 64 (more codes
than states)" is false at `p` 7, `m` 32: the modular automaton minimises to exactly `p^2` states
(25 at `p` 5, 49 at `p` 7; checked by partition refinement on `window_tasks.modular_machine`),
so a 32-code network can't be it, and those 20 runs can't certify. D20's "room for 49 states and
32-64 codes" missed it too. E7 runs to the end as registered (stopping it would have saved about
as much as it lost); its findings report the cells as a negative control. The other cells have
1.3 and 2.6 times the states' codes, the low end: on addition the matrix's pooled curriculum runs
worked 10 of 30 at 1.3 times (`m` 4), 16 of 30 at 2.7 and 19 of 30 at 5.3 (post hoc, not C3's
test: Cochran-Armitage p 0.028). D23 makes the check a rule; E7' is the test E7 meant to be.

### E7'. Modular division with enough codes (C7'; D23-D26)

**Design.** E7's task, harness, catalogue (D18), certificate (D17) and settings, changing only
the codes and `k`: `p` 5 and 7; `k` 1; the curriculum and all-at-once arms; `s` 16 (D20); `m` at
about 5 and 10 times the minimal automaton's states (D24); 10 seeds a cell: 2 x 2 x 2 x 10 = 80
runs.

**Feasibility (D23), stated per cell:**

| `p` | minimal states | reachable within | told apart within | `m` (codes / states) |
| --- | --- | --- | --- | --- |
| 5 | 25 | 2 columns | 1 column | 128 (5.1), 256 (10.2) |
| 7 | 49 | 2 columns | 1 column | 256 (5.2), 512 (10.4) |

Every state is reachable and every pair of states gives different outputs within the first check
length, so the check can see the whole automaton. A snap over 512 codes in 16 units has room
(codes are free vectors; addition's working runs used 14 of 16). The budget: E7's epoch time
doesn't depend on `m` (5.1 s at `m` 32 and 5.2 s at 64, `k` 1 all-at-once, `p` 5; 9.9 and 10.0 s
at `k` 4 curriculum), so more codes cost epochs, not time per epoch.

**Why `k` 1 only.** From E7's step records (2026-10-13 19:40: p 5 complete, p 7 part-way), every
`k` 4 run failed to fit even its training lengths, at both `m`
and both `p` (best screens 0.11-0.34), while `k` 1 nearly worked: the curriculum passed steps 1
and 2 every time and plateaued at step 3 with best screens 0.83-1.0; all-at-once reached
0.80-0.996. More codes can't help a window that can't fit four columns' transitions, so `k` 4 is
a question about the window's capacity, not the codes' (D25).

**Measures and analysis.** As E7's: works, the break, codes used; both certificates (D13);
the long tests U1, S1 and S2 on every certified network (D22); per cell the working and
certified rates with Wilson intervals; every certified machine's minimised size against `p^2`.
C7' is refuted if no network certifies at either `p`. The 5 against 10 times comparison per `p`
and arm is Fisher's exact test (D4), reported as a measurement on the second task (C3's
question), not a registered claim.

### E8. The modular-sum ladder (C8; D27-D30)

**Why.** E7 and E7' refuted C7 and C7': with codes for every state, the window still didn't learn
a 25- or 49-state transition table exactly. Addition's machine has 3 states. E8 asks where between
3 and 25 the current design stops, with a task of the same shape whose size is one parameter (the
owner, 2026-10-14: "run that and get it to work, then step up").

**The task.** The modular sum: E7's pairs, tokens (`3 a_i + b_i`, 9), reading order (most
significant first), held-out tenth and training mixture; at every column the running
`(A + B) mod q` of the prefixes, `q` classes. Its automaton has exactly `q` states (state
`(A + B) mod q`; minimised by partition refinement, `test_window_tasks`), so it is the additive
half of division, whose state is the pair. Catalogue: U1, U2 and S1 as division's, verified against
the running sums (no undefined class, so no S2, and no reach).

**Design (the first rungs).** `q` 4 and 5; both arms; `k` 1; `s` 16; W11's window; `m` 32;
E7''s settings; 10 seeds a cell: 2 x 2 x 10 = 40 runs.

**Feasibility (D23):**

| `q` | minimal states | reachable within | told apart within | `m` (codes / states) |
| --- | --- | --- | --- | --- |
| 4 | 4 | 1 column | 1 column | 32 (8) |
| 5 | 5 | 1 column | 1 column | 32 (6.4) |

Every residue is a one-column answer and outputs equal states, so every state is reached and
every pair told apart at the first check length. The budget: E7's `k` 1 runs took 7-15 minutes on
`jebel`, the network here the same but for its classes; on `pyramidon`'s 3 workers about 2.5-5.5 h
(measured once running).

**Measures and analysis.** As E7's: works, both certificates (D13), long tests U1 and S1 on every
certified network (D22's lengths), per cell the working and certified rates with Wilson intervals,
every certified machine's minimised size against `q`. C8 is refuted if no network certifies at
either `q`.

**Outcome (2026-10-14).** C8 held weakly at `q` 4 (1 of 20 certified, the 4-state automaton, every
long test passed) and not at `q` 5 (0 of 20); 37 of 40 runs fitted their training lengths
exactly and then failed at 8 columns: they had memorised (D23's exclusion bound; findings in the
study script's docstring). E8b trains past the bound.

### E8b. The modular sum, trained past the exclusion bound (C8; D31)

**Design.** E8's task, rungs and settings but two: `m` 16 at both rungs (4 and 3.2 times the
states, a surplus kept for learning: the owner, 2026-10-14) and 20 steps, so training reaches 20
columns at `k` 1, the bound `m + q - 1` (19 at `q` 4, 20 at `q` 5); both arms, 10 seeds: 2 x 2 x 10
= 40 runs. No wrong machine of at most 16 states then agrees with the automaton on every input up
to 20 columns, so memorising can't pass training exactly; training and its checks see a sample of
those inputs (exhaustive only to 6 columns), so the certificate still decides.

**Feasibility (D23):** `m` 16 against 4 and 5 states; every state reached and told apart within 1
column; longest training length 20 against the exclusion bound's 19 and 20; the budget from a
probe on `pyramidon` before launch. **The probe** (2026-10-14, `pyramidon`: `q` 5, `m` 5, 12
columns, all-at-once, one run): 31 epochs at about 12 s each, a 187 s check, 652 s in all, stopped
on a plateau without fitting (a codebook at the bound trained badly: D31's surplus). Scaled to 20
columns: about 20-40 minutes an all-at-once run, 1-3 h a curriculum run (20 steps, each warm-started
and checked), about 50 worker-hours in all.

**Launch (the owner, 2026-10-14).** The arms split between machines, each config run once per arm
(`run --arm`; keys and directories unchanged): the all-at-once arm on `pyramidon` at once
(`--workers 3`, 20 runs: about 2.5-5 h), the curriculum arm on `jebel` after stage 7 ends (6 workers,
20 runs: about 4-12 h). Before the analysis, `jebel`'s `results/` and `models/` are copied into
`pyramidon`'s directories (the arms' keys are disjoint), with their manifests (D8).

**The all-at-once arm stopped (the owner, 2026-10-14, D33).** Launched on `pyramidon` at 07:27
(`6febef4`), stopped at 08:30 with 6 of `q` 4's 10 runs finished, `q` 5 not started. All 6 failed
alike: `plateau after 31 epochs`, break 0, an identical screen (0.026) every epoch, and every state
on one code on real inputs, from the initial state through every window. A probe on fresh networks
(seeds 0 and 1, 2 epochs, E8b's settings, only the training lengths changed; codes per state at 8
columns): lengths 1-3 and 1-6 split the state (3-7 codes), 1-10 and 1-20 don't (1-2 codes). A fresh
network's state starts on 1-2 codes; short lengths give the gradient that splits it, and at 1-20
in equal shares they are a tenth of training. So the arm could produce no working network, and it
was stopped (D33). The 6 runs stay in `data/window-recurrent/e8b-q4/`, recorded in the findings;
the curriculum arm, which starts at 1 and 2 windows, runs as registered.

**Measures and analysis.** As E8's, and C8 refuted at a rung if no network certifies there.

**The curriculum arm stopped (the owner, 2026-10-14, D35).** Launched on `jebel` at 12:44
(`9dce80e`), stopped at 13:58 in `q` 4's first 6 runs (12-15 of 20 steps each), `q` 5 not started.
Exploratory, not E8b's registered measure (which reads each run's final network): seeds 1, 3 and
5's every step network strictly certified on `pyramidon` (`data/window-recurrent/e8b-q4-steps/`,
36 networks, manifest `e8b-q4-steps.sha256`). **All three certify from step 4** (training to 4
columns, far below the bound of 19): the 4-state automaton, 0 wrong outputs, 0 wrong states.
**Training on destroyed two:** seed 3 certified at steps 4-8 and fails from 9, seed 1 at 4-11 and
fails at 12, each after a plateau, its state down to 3 codes; seed 5 certified at 4-13. Seeds 0, 2
and 4 fitted steps 1-3 and plateaued from step 4 or 5. A run's final network undercounts what the
curriculum finds, so the arm was stopped and its question moved to E8c.

### E8c. The modular sum as an existence test (C8; D35)

**Claim.** C8 as an existence claim at each rung: some network at `q` 4, and some at `q` 5, trained
by the curriculum, is certified against the `q`-state automaton, near ties branched (the
harness's per-step verdict; it implies the strict one, the actual snapping being one branch). None
found at a rung is "not found within the budget", not a refutation (the factored-state workplan's
C9 reads the same way). A certified network is the proof whatever its training length, so the
bound is the cap, not the claim's support.

**Design.** E8b's curriculum cell (`k` 1, `s` 16, `m` 16) and settings, at `q` 4 and 5, 20 steps
(the bound `m + q - 1`, 19 and 20, is the cap), with the existence settings the factored-state
workplan's stage 7 built: **`certify_steps`** (each step's network certified, a run ended at its
first certified step) and **`stop_at_certified`** (each rung's study stopped at its first certified
run). 6 seeds a rung (0-5, `jebel`'s 6 workers); the all-at-once arm left out (D33: its state
collapses at 1-20). `scripts/window_recurrent_e8c_q4.json` and `_q5.json` (a test checks them
against E8b's), run on `jebel` from `main` (results in the tuple layout), `q` 4 then `q` 5.

**Feasibility (D23):** `m` 16 against 4 and 5 states; the cap 20 against the bound's 19 and 20.
Seeds 0-5 at `q` 4 are E8b's runs again (C0: the harness is deterministic), so `q` 4 is expected to
stop at step 4 (seed 1, 3 or 5), about 30 minutes; it checks the protocol before `q` 5.

**Budget.** E8b's pace on `jebel`: 12-15 steps in 74 minutes a run (about 5-6 minutes a step, the
step's certificate at `k` 1 a few seconds), so a run to the cap is about 2 h, and `q` 5's 6 seeds
in parallel at most about 2-3 h. Within a day.

**Measures and analysis.** Per rung: certified or "not found within the budget"; the certified
network's step, codes reached and minimised; its long tests (U1, S1). E8's analysis otherwise.

## Decisions

Settled under the owner's standing delegation, with each option's pros and cons:

- **D1. What the plan claims. Settled: (a).**
  - (a) *Chosen.* C0-C6 with E1-E6: the certificate first, then the ablation and the measured
    rates. Pros: each finding becomes a claim with a test that could refute it; the certificate
    is the strongest evidence available to a snapped state and costs hours, not days. Cons:
    about 30 hours of `jebel` and one package change.
  - (b) The empirical claims only (long tests, more seeds), no certificate. Pros: no new tool.
    Cons: still "tested to `n`", the literature's level, where a proof for every length is
    within reach.
  - (c) Write up now with the caveats. Pros: no more compute. Cons: the review shows every
    headline already known; what is new is untested.
- **D2. Arbitrary length: how. Settled: (a).**
  - (a) *Chosen.* The certificate by enumeration (E1), and long tests (128-1,024) as an
    independent check. Pros: a proof, checked against data. Cons: two tools.
  - (b) Long tests only, to 1,000 or 2,000 columns. Pros: comparable to the literature. Cons:
    still finite; a failure past the last length stays possible.
  - (c) The certificate only. Pros: cheapest. Cons: a bug in the certificate would go unseen.
- **D3. The ablation and the margins. Settled: (a) and a tolerance of 1.05.**
  - (a) *Chosen.* `WindowRecurrence(codes=None)`: the GRU's output passed on unsnapped, the
    codebook losses gone; everything else unchanged. Pros: one variable changes; no crate change
    (the snapping is the crate's `codes_forward`/`codes_backward`, which are skipped). Cons: a
    package change in three implementations, with parity tests and a golden entry.
  - (b) Snap at test time only (train continuous, snap after). Pros: no training change. Cons:
    a network never trained to its codes; not the design's counterfactual.
  - (c) Robinson's binning. Pros: a published discretiser. Cons: a second discretiser, not an
    ablation of ours.
  - The margins: a snap holds when the second-nearest code is at least 1.05 times as far as the
    nearest, a digit when its logit leads the next by at least 1e-6 (both far above rounding
    error). Pros: no certificate rests on a tie. Cons: a network can fail on margin
    alone; reported apart from a wrong output.
  - **Refined in stage 2: a snap's near tie branches** rather than failing. Options:
    - (i) *Chosen.* Branch: every code within 1.05 of the snapped one is a next code too, explored
      and checked. Pros: a proof for either snap, so still no certificate rests on a tie; a network
      whose ties fall between copies of one carry is certified, as it should be (a working network
      of the matrix, `k` 1 `s` 16 `m` 16 seed 0, has 3 snaps at a ratio of 1.001, between codes of
      one carry). Cons: a few more codes to explore; the count of near ties is reported, not a
      verdict.
    - (ii) Fail every near tie (D3 as first written). Pros: simplest. Cons: fails networks that
      are correct under every rounding.
    - A digit's near tie can't branch (one of its two digits is wrong): it fails.
  - **Settled again in stage 3, by the owner, on E1's data: (iii), two named verdicts.** 105
    networks certified both ways (`certify --snap-ratio 1` is strict): 52 strictly, 48 with
    branching; the 4 between are correct as they compute and wrong only when a near tie (ratio
    1.006-1.03) snaps the other way. Options:
    - (i) Branching decides, strict reported beside. Pros: a pass holds whatever the rounding (another
      batch, another BLAS). Cons: fails networks correct as they run.
    - (ii) Strict decides, branching reported beside. Pros: the network as it runs. Cons: a proof
      resting on snaps at ratios down to 1.0000002, which another rounding could flip.
    - (iii) *Chosen.* Both, named: **certified** (strict: the network's own snaps) and **robustly
      certified** (branched: either snap of every near tie). Pros: nothing hidden; each claim says
      which it rests on. Cons: two figures where one would do.
- **D4. Seeds and statistics, fixed before any run. Settled: (a).**
  - (a) *Chosen.* 20 seeds a cell and arm (seeds 0-19; 0-4 rerun the matrix's, which is C0's
    check). Rates with Wilson 95% intervals; two arms by Fisher's exact test, two-sided; a trend
    over `m` by the Cochran-Armitage test; examples to pass by Mann-Whitney. The tests of C2-C4
    and C6 corrected together by Holm's method at 0.05. Pros: 20 seeds tell 0.8 from 0.4 (Fisher,
    p 0.023); fixed rules, no choice after the data. Cons: 0.75 against 0.5 stays undecided at
    20 seeds (p 0.19).
  - **Implemented** in `indrajala_ml/studies/inference.py`, by hand (no scipy): `wilson`,
    `fisher_exact` (exact, in integers), `cochran_armitage` (the normal approximation),
    `mann_whitney` (exact without ties; with ties, normal with tie and continuity corrections) and
    `holm`.
  - (b) 10 seeds. Pros: half the cost. Cons: tells only differences near 0.6.
  - (c) 50 seeds. Pros: differences near 0.25. Cons: 2.5 times the cost; worth it only for a
    result that 20 leaves open, as a follow-up.
- **D5. The cells. Settled: (a).**
  - (a) *Chosen.* `k` 4 `s` 4 (the smallest of the matrix's best cells, 7,779 parameters at `m`
    16) and `k` 1 `s` 16 (the longest extrapolation, a window ratio of 21.3; the purest
    recurrence). Pros: the two ends of `k`; both among the best; `k` 2, the weakest, left out.
    Cons: two cells, not the matrix.
  - (b) All three of W30's cells. Pros: as the control. Cons: `k` 4 `s` 16 adds 50% for a cell
    whose `s` barely mattered.
  - (c) The whole matrix at 20 seeds. Cons: 360 runs, about 27 hours for the curriculum arm
    alone.
- **D6. The code counts. Settled: (a).**
  - (a) *Chosen.* {3, 4, 6, 8, 16, 32} at `k` 4 `s` 4 and {3, 4, 8, 16} at `k` 1 `s` 16. Pros: 3,
    the automaton's size, is the test of "more than needed"; 32 asks whether the trend saturates;
    the `k` 1 cell replicates the trend more cheaply. Cons: 200 runs.
  - (b) The matrix's {4, 8, 16}. Pros: cheaper. Cons: no `m` 3, so no test of C3's "more than
    needed".
- **D7. Comparing the arms' cost. Settled: (a).**
  - (a) *Chosen.* Each arm at its own budget (the curriculum's 3 steps of up to 200 epochs, the
    control's one), compared by working rate and by examples to pass. Pros: each arm as designed;
    the examples are recorded per step (D10 of the terminated plan). Cons: the curriculum may see
    more examples; the examples column shows how many.
  - (b) Matched examples (the control's budget set to the curriculum's). Pros: equal data. Cons:
    a new setting, and the control already passes in a median of a few epochs.
- **D8. A baseline. Settled: (a).**
  - (a) *Chosen.* E6, the recurrence without the window's transformer, at `k` 1. Pros: asks
    whether the design's novel part (the transformer inside the step) matters; cheap at `k` 1.
    Cons: one more spec variant (a window with no attention blocks, which validation must allow).
  - (b) No baseline. Pros: cheaper. Cons: "transformer + recurrent state" with no evidence the
    transformer contributes.
  - (c) A reimplementation of a published model (Neural GPU, Rational Transductor). Pros: a
    direct comparison. Cons: a new architecture in three implementations; the literature's own
    figures are the comparison, with the caveat that tasks differ.
- **D9. The long tests. Settled: (a).**
  - (a) *Chosen.* B1 at 100,000 cases at 128, 256 and 1,024; M3 at distances 1, 2, 4, ... `n`, 1,000
    each. Pros: B1 at 1,024 is about 15 minutes a network (measured: 106 s at 128, 216 s at 256,
    linear in `n`); M3 at every distance would be `n` strata, about a million cases at 1,024 (156 s
    for its 128,000 at 128). Cons: M3 subsampled; the
    certificate covers every distance.
  - (b) The whole catalogue at 1,024. Cons: hours a network.
- **D10. Where it runs. Settled.** E1 and E5 on `pyramidon` (no training; nothing is timed);
  E2-E4 and E6 on `jebel` under the machine lock, 6 workers, `OPENBLAS_NUM_THREADS=1`, each a
  config under `scripts/`, results recorded by path and sha256 (as the matrix's).
- **D11. Where the findings go. Settled: (a).**
  - (a) *Chosen.* Each experiment's findings in `scripts/window_recurrent_study.py`'s docstring
    (as the matrix's), the certificate's in its script's, and a report at the end:
    `docs/window-recurrent-report.md`, each claim with its evidence and the literature's
    comparison, then the README. Pros: the project's convention, and one place that states the
    claims. Cons: a long docstring; the report summarises it.
  - (b) A paper-style write-up only. Cons: the evidence wouldn't sit beside the code.
- **D12. Archiving the networks. Settled: (a).**
  - (a) *Chosen.* The certified networks archived in `davidbarkhuizen/indrajala-models` (as the
    addition sweep's threshold models), with their certificates. Pros: a claim of correctness
    can be rechecked by anyone; small (about 0.5 MB a network). Cons: a PR to the archive per
    experiment.
  - (b) On the two machines only. Cons: a certified adder lost with a disk.
  - **Which networks, settled in stage 3 by the owner: (ii).** (i) The 48 robustly certified.
    Pros: smallest; every one a rounding-proof adder. Cons: loses the evidence of the other
    claims. (ii) *Chosen.* The 52 certified and the 2 networks whose tests passed to 64 columns
    with a counterexample at 12, each with both certificates. Pros: every E1 claim can be
    rechecked from the archive. Cons: a few MB more; the archive's README explains the classes.
    (iii) All 105. Pros: E1 rerunnable whole. Cons: about 50 MB, mostly failing networks, against
    the archive's convention.
- **D13. C1 against E1's result. Settled in stage 3 by the owner: (c).** C1's prediction ("every
  working network certified") fails: 9 of W10's 57 working networks aren't robustly certified
  (3 break before 64 columns, 4 only on a near tie, and 2 whose tests passed to 64 columns are
  wrong at 12). Options:
  - (a) Report C1 refuted for W10's "works" and restate it: a certified network adds at every
    length, and the sampled tests overstate working. Pros: the pre-registered rule kept. Cons: no
    exact criterion for the later experiments.
  - (b) Redefine "works" as certified from E3 on. Pros: exact. Cons: a definition changed after
    the data; E3/E4's rates no longer comparable with the matrix's.
  - (c) *Chosen.* (a) for E1, and E3/E4 report the sampled "works" and the certificates' verdicts
    side by side. Pros: honest and comparable. Cons: a column more; every E3/E4 network certified
    (already planned).
- **D14. A second task. Settled by the owner (2026-10-13): modular division, in this plan's one
  paper.** Options put: (a) multiplication by a constant (the adder's shape; the carry's states
  as a dial; cheapest); (b) modular division (finite-state where general multiplication isn't; a
  non-trivial function, modular inverses; its own state size); (c) none (the adder alone); and a
  second paper on several tasks. The owner chose one paper, the adder and modular division only.
  Pros: the method shown on a second automaton, of 25-49 states against the carry's 3, with an
  inverse to learn. Cons: a harness generalisation and a second matrix on `jebel`.
- **D15. How the task answers. Settled: (b).**
  - (a) One answer, after the last column. Pros: the textbook statement. Cons: a new output path
    in all three implementations and a new certificate mode; one label a sequence, a weak signal.
  - (b) *Chosen.* The running answer at every column, the last column's the answer. Pros: a
    finite transducer with an output per column, the adder's shape: no package change (the
    network takes any embedding size and class count); a label a column; the certificate checks
    every output. Cons: an "undefined" class while `B mod p` is 0; a less usual statement.
  - (c) A readout from the final state. Pros: explicit. Cons: (a)'s costs and a new part.
- **D16. The reading order and `p`. Settled: most significant first; `p` 5 and 7.**
  Most significant first, the state is `(A mod p, B mod p)`, `p^2` states; least significant
  first it needs `3^i mod p` too (100 states at `p` 5, 294 at 7). `p` 3 is trivial (the last
  digit decides) and 2 a digit-sum parity; 5 and 7 give 25 and 49 states. Pros: small enough to
  certify (`9^k` windows, 6,561 at `k` 4, times the reachable codes), large enough to test C3's
  codes against states. Cons: the opposite order to the adder's, so the harness's reading order
  becomes the task's.
- **D17. The certificate. Settled: (a).**
  - (a) *Chosen.* `phi` maps codes to any target Mealy machine's states (its start, transition
    and output functions over the task's column tokens), the carry automaton one instance and the
    modular automaton another; the margins, the branching, the minimisation and the
    counterexample unchanged. Pros: one certificate, the stage 2 tests carried over; a third
    task would be a new target only. Cons: the carry's specialisation (digits and carries as
    arrays) becomes a table per machine.
  - (b) A second certificate for the modular task. Pros: no change to E1's. Cons: two copies of
    the enumeration and the margins.
- **D18. Modular division's catalogue and training mixture (stage 5). Settled: (a).**
  - (a) *Chosen.* Four properties, each checking every column's answer (the running quotient or
    undefined): **U1**, every pair at `n` <= 6 (`9^n` <= 600,000), else 100,000 sampled; **U2**,
    the held-out tenth (by hash, D8's), checked from `n` 3 and never at a test length (B3's
    rules); **S1**, every final state `(a mod p, b mod p)` reachable at `n`, 1,000 each; **S2**,
    `b`'s first `L` digits zero then a nonzero digit, 1,000 per `L` in 1..`n` (at 1, 2, 4, ...,
    `n` in the long tests, D9), its `L*` the test pass's reach (M3's place). The training mixture,
    in turn: a uniform pair, a zero prefix of `b`, a zero `a`; a held-out pair drawn again. Pros:
    every target state is checked at the end of a case (S1), and the undefined class over runs of
    any length (S2), the two ways the task differs from addition; the harness's checks, tests and
    long tests keep their shape. Cons: no relation properties (addition's A1-A3).
  - (b) Analogues of addition's eleven. Pros: comparable tables. Cons: most have none (the column
    sum, the carry patterns), and relations such as `(3a, 3b)` giving `(a, b)`'s answer are
    implied by the target machine, which the certificate checks whole.
  - (c) U1 alone. Pros: simplest. Cons: a final state or a long undefined run a sample misses
    goes unchecked by the tests (the certificate still catches it, but E7 compares the two).
- **D19. The certificate's JSON over any target (stage 5). Settled: (a).**
  - (a) *Chosen.* The target's terms for every task: `wrong_outputs` and `wrong_states` (stage
    2's `wrong_digits`, `wrong_carries`), the counterexample's kinds `outputs` and `state`
    (`digits`, `carry`), its `state_in` (`carry_in`) and `case` (`addition`), its `after` (the
    window after: a word telling the two states apart, padded, or zeros), and the `task`. E1's
    files keep stage 2's names; the certificate's docstring maps them. Pros: one schema for every
    task; nothing called a carry that isn't one. Cons: E1's files and later ones differ in names.
  - (b) Stage 2's names for every task. Pros: one schema with E1's files. Cons: a modular
    certificate's "carries" are states of 25 or 49.
  - (c) Stage 2's names for addition, the target's for others. Cons: two schemas, a special case.
- **D20. E7's state width `s` (stage 6). Settled: (a), the owner agreeing ("let's not run both
  widths if we can avoid it").** E7's design fixes `p`, the arms, `k` and
  `m`, not `s`. On addition `s` made no clear difference (the matrix: 24 of 45 runs working at
  `s` 4, 21 at 16; robustly certified 21 and 17).
  - (a) *Chosen.* `s` 16 only: 80 runs. Pros: a state of 16 units has room for 49 states and 32-64
    codes kept well apart (the snap's margins), so a failure says more about the design than about
    a cramped cell; the cost as planned (about 8 h). Cons: `s`'s effect on a 25-49-state task
    unmeasured.
  - (b) `s` 4 only. Pros: the matrix's slightly better width on addition; smaller networks. Cons:
    4 units carrying up to 49 states, 64 codes crowded in 4 dimensions: a failure would be
    ambiguous.
  - (c) Both: 160 runs. Pros: measures `s` on the second task. Cons: about 16 h of `jebel`, twice
    the plan, for a factor no claim rests on (C7 asks whether the design learns the task).
- **D21. E6's window (stage 8). Settled: (c), under the owner's delegation.** D8's "embedding and
  output layer only" can't stand as written: validation puts a StateInput right before a residual
  block (W18), and at `k` 1 a window's attention reads one token, so it is a linear map. A probe
  on `pyramidon` (2026-10-13; `k` 1 `s` 16 `m` 8, seeds 0 and 1, the matrix's settings) trained
  each window: the full window's step 1 reproduced the matrix's bit for bit (seed 0), and the
  matrix's runs work on both seeds (break 64); the linear window (2,835 parameters) passed steps
  1-2 on seed 1 and reached a best screen of 0.95-0.98 at step 3, working on neither; the FFN
  window (6,355) passed step 1 on both and failed steps 2-3 (best 0.98-0.99), working on neither.
  Both train; neither fails for a trivial reason.
  - (a) The linear window: the embedding, the state through one StateInput and one linear
    residual (no nonlinearity but the LayerNorms), the output layer. Pros: D8's question as meant
    (does the window's network do anything, or does the recurrence do the step?); no package
    change. Cons: it removes the attention and the FFNs together, so a difference can't say which.
  - (b) The FFN window: W11's blocks without their attention. Pros: removes exactly the
    transformer's token mixing. Cons: at `k` 1 the attention is a linear map of one token, so (b)
    alone barely tests the design's novel part.
  - (c) *Chosen.* Both: (a) at `m` 4, 8 and 16 (D8's cells) and (b) at `m` 8 (the matrix's best at
    `k` 1 `s` 16), 20 seeds each: 80 runs. Pros: (a) answers D8, (b) splits a difference between
    the attention and the FFNs, at a quarter more runs; baseline runs that fail stop early on
    plateaus. Cons: about 2 h more of `jebel` than (a) alone.
- **D22. E7's long tests (stage 6). Settled: (a), under the owner's delegation.** D9 names
  addition's properties (B1, M3); division's catalogue (D18) has U1 (B1's place), S2 (M3's) and
  S1, and U2 is never checked at a test length. Lengths and sizes as D9: 128, 256 and 1,024,
  100,000 sampled, 1,000 per stratum, S2 at 1, 2, 4, ..., `n`.
  - (a) *Chosen.* U1, S1 and S2: every property a test pass reads. Pros: every reachable final
    state checked at 1,024 (25 or 49 strata), the one thing division's tests check that addition's
    long tests had no counterpart for; the long tests then test what the certificate certifies, state
    by state. Cons: 136,000-160,000 cases a network at each length against E1's 111,000, about 1.2-1.5
    times E1's cost a network.
  - (b) U1 and S2, D9's two in division's terms. Pros: E1's cost; the same shape as E1's table.
    Cons: a final state a sample of 100,000 rarely ends in goes untested at long lengths.
  - (c) U1 alone. Pros: cheapest. Cons: neither the long undefined runs nor the final states
    tested past the check lengths.
- **D23. Every cell must be able to succeed (the owner's rule, 2026-10-13: "always check that
  our conditions could at least result in success... unless there is some other sensible
  reason"). Settled: (a).**
  - (a) *Chosen.* Before a cell is configured, its necessary conditions are computed and written
    beside it: the target's minimal state count against `m`; that every state is reachable and
    every pair told apart within the check lengths; the window and state width against the
    columns a window must carry; the budget from measured epoch times. A cell below a bound is
    dropped unless kept for a stated reason (a negative control, a threshold probe), recorded
    with it. Pros: no compute spent on outcomes known in advance; the bound makes a failure
    interpretable. Cons: a minimisation and a few probes per new task.
  - (b) Check by eye at review. Pros: no procedure. Cons: what E7 did; it missed `m` 32 < 49.
  - **The exclusion bound (added 2026-10-14, after E8, with the owner).** The conditions above make
    the target checkable, not wrong machines excluded. Two machines of `n` and `n'` states that
    agree on every input up to length `n + n' - 1` are equivalent, so a network of `m` codes exactly
    right on every training length up to `L` is the target's `q`-state machine when
    `L >= m + q - 1`; below it, a wrong machine of up to `m` states can fit every training case,
    and E8's networks found such machines (memorising lengths 1-3 with 6-22 states). So a cell
    also states its longest training length (steps x `k` columns) against `m + q - 1`, and meets
    it unless the plan says why not. Every config before E8b falls short (addition's `k` 1 trained
    to 3 columns against 6-18 needed; E7 and E7' to 3 against 56-560), recorded in its test's list.
- **D24. E7''s code counts. Settled: (a).**
  - (a) *Chosen.* About 5 and 10 times the minimal states (128 and 256 at `p` 5, 256 and 512 at
    `p` 7). Pros: 5 times is where addition worked best (5.3 at `m` 16), 10 times asks whether
    more still helps; both far above E7's 1.3-2.6, so a failure no longer reads as too few codes.
    Cons: 512 codes is a large codebook (training's codes losses over more vectors).
  - (b) 5 times only (40 runs). Pros: half the runs. Cons: a failure at 5 times would leave
    open whether more codes help.
  - (c) A ladder from 1.3 to 10 times. Pros: the whole curve on the second task. Cons: two to
    three times the runs for a curve C7' doesn't need.
- **D25. E7''s window: `k` 1 only. Settled: (a).**
  - (a) *Chosen.* `k` 1, both arms. Pros: the `k` E7 shows can fit (D23's budget condition);
    E7's runs at `k` 1 are the comparison; about 3-8 h of `jebel`. Cons: says nothing about
    longer windows on this task.
  - (b) Also `k` 4 at E7''s codes. Pros: comparable with E7 cell for cell. Cons: E7's `k` 4 runs
    never fit their training lengths at any `m` (best screens 0.11-0.34): D23 drops them, since
    more codes can't fix a window that can't fit.
  - (c) Also `k` 4 with a larger window. Pros: asks the capacity question directly. Cons: a
    different network, so a different claim; it belongs to its own experiment, after E7''s.
- **D26. E7''s recipe and seeds. Settled: (a).**
  - (a) *Chosen.* The matrix's and E7's settings unchanged (`max_epochs` 200, patience 30, rate
    0.003, 3 steps), 10 seeds a cell. Pros: one change from E7 (the codes), so a difference is
    the codes'; 10 seeds give intervals half as wide as E7's 5 at little cost (`k` 1 runs took
    7-15 minutes in E7). Cons: if training needs more epochs at 49 states, E7' won't show it.
  - (b) A longer budget (`max_epochs` 400, patience 60). Pros: rules out an early stop. Cons:
    two changes at once; E7's `k` 1 runs stopped on plateaus, not the budget (two `budget` stops in
    E7's step records, both at `k` 4), so the evidence doesn't ask for it.
  - Checked before E7' starts, from E7's step records (E7' runs before E7's certificates, stage
    7a): if E7's `k` 1 steps stopped on the budget or with screens still rising at the stop, (b) is
    adopted and recorded here first, never after.
    **Checked (2026-10-13, E7's 40 `k` 1 runs, all finished): (a) stands.** 80 steps: 40 passed,
    40 stopped on a plateau, none on the budget, and every plateau's best screen came at least
    the patience's 30 epochs before its stop.
- **D27. E8's task. Settled: (a), the owner choosing the modular sum (2026-10-14).**
  - (a) *Chosen.* The modular sum `(A + B) mod q`, `q` states. Pros: division's shape exactly
    (pairs, tokens, order, an output a column), so a working rung carries straight over to
    division; one parameter sets the size; division's additive half. Cons: no undefined class,
    so one of division's difficulties (S2) isn't on the ladder.
  - (b) One number's residue `A mod q`, `q` states. Pros: the textbook divisibility automaton.
    Cons: 3 tokens, not 9: a different embedding, so a result transfers less directly.
  - (c) Division at `p` 2 (4 states). Pros: division itself. Cons: trivial in base 3 (3 = 1 mod 2:
    a digit-sum parity), and one size only.
- **D28. E8's first rungs and cells. Settled: (a).**
  - (a) *Chosen.* `q` 4 and 5 (the owner: start there), both arms, `m` 32 (at least 5 times the
    states, where E7' saw no gain from more), 10 seeds. Pros: the likeliest to work, so the ladder
    starts from a working rung; both arms, because which one carries over is itself unknown
    here. Cons: two rungs only before the next decision.
  - (b) `q` 5, 7, 11, 13 at once. Pros: the whole lower ladder in one sitting. Cons: if `q` 5
    fails, the rest is spent knowing nothing more.
- **D29. Where E8 runs. Settled: (a), the owner's (2026-10-14).**
  - (a) *Chosen.* Started on `pyramidon` (3 workers) while `jebel` runs stage 7, moved to `jebel`
    when stage 7 ends: `run --workers W` overrides the config's count, the harness resumes from
    the results on disk, and a run's result doesn't depend on the machine (the E6 probe reproduced
    the matrix's runs bit for bit across the two). A run is moved only finished (an unfinished
    all-at-once run restarts from its seed; a curriculum run's resume from a middle step isn't
    checked for bits, so none is moved mid-run). Pros: the ladder starts at once. Cons:
    `pyramidon` is slower and shared with development.
  - (b) Wait for `jebel`. Pros: one machine. Cons: about 18 h of waiting.
- **D30. Stepping up. Settled: (a).**
  - (a) *Chosen.* A rung "works" when some network certifies (C8's test). After a working rung,
    the next: `q` 7, then 11, 13 and 16, then division's 25, each a registered config with its
    bound (D23). At the first rung that doesn't work, the window's capacity is tested there
    (wider windows, `d` 32 and 64, kept on a branch until then) before going higher. Pros: each
    step answers one question; the capacity change is tested where it's needed. Cons: a rung at
    a time.
  - (b) Widen the window now. Pros: possibly faster to division. Cons: two changes at once, and
    no rung to say whether width was the limit.
- **D31. E8b's codes and training length (the exclusion bound). Settled: (a), with the owner.**
  - (a) *Chosen.* `m` 16 at `q` 4 and 5 (a surplus of 3-4 times the states), trained to 20 columns.
    Pros: the owner's surplus for flexible learning (addition worked best with spare codes; E8's
    certified network used 8 of 32); past the bound, so memorising can't fit. Cons: 20 steps at
    `k` 1, the longest training yet (cost from the probe).
  - (b) `m` at the states (`q`), trained to `2q - 1` columns (7 and 9). Pros: the cheapest bound.
    Cons: no surplus: the owner prefers one, and a codebook at the bound may train badly.
  - (c) E8's 32 codes, trained to 36 columns. Pros: E8's codes unchanged. Cons: nearly twice (a)'s
    training length for a surplus (a) already gives.
- **D32. Modular division waits for a factored state (the owner, 2026-10-14).** E7 and E7' failed
  at fitting: one codebook must learn division's 25- or 49-state table flat, while its state is two
  independent residues `(A mod p, B mod p)`, each a modular sum's counter. Division resumes once
  the recurrence has a factored state (a codebook per residue, product quantisation), a package
  change with its own plan ([factored-state-workplan.md](factored-state-workplan.md)); until then
  the ladder is the modular sum.
- **D33. Stop a run that can produce no working network (the owner, 2026-10-14).** The studies aim
  for existence first, a working network, and then search for its bound; compute is limited, so a
  long run that the evidence says can't produce one is stopped and recorded, not finished.
  - (a) *Chosen.* Stop E8b's all-at-once arm (above). Pros: about 4.5 h of `pyramidon` saved; 6
    seeds and the probe already show the mechanism. Cons: the arm's rate is 0 of 6 at `q` 4, not
    of 10 at both rungs, as registered.
  - (b) Let it finish. Pros: E8b exactly as registered. Cons: hours for a result the probe
    predicts, against the owner's existence-first strategy.
- **D34. Stage 7 trimmed to the cells that answer a question (the owner, 2026-10-14).** Under D33's
  existence-first strategy, the 161 runs left were weighed by the question each answers.
  - (a) *Chosen.* Keep `k` 4 `s` 4: the curriculum at `m` 3, 4, 6, 8, 16 (C3's trend, from exactly
    the automaton's 3 states to 5 times them) and the all-at-once arm at `m` 16 (C4 where both arms
    are known to work, and E8b's question: the curriculum is needed at 20 windows; is it at 3?).
    Drop `m` 32 (past the curve's informative range; `m` 16 is 5 times the states) and every `k` 1
    cell (curriculum `m` 3-16, all-at-once `m` 8): training there reaches 3 columns, far short of the
    exclusion bound (5-18), so their rates measure how often memorising lands on the machine. C0's
    reruns at `k` 4 (20 runs) stay; those at `k` 1 go. Pros: about 41 runs (4-5 h) left instead of
    161 (17 h or more); `jebel` free for E8b's curriculum arm a day sooner. Cons: C3's trend and C5's
    rates at `k` 4 only; C4 at one cell; C0 at `k` 4 only.
  - (b) Finish as registered. Pros: every claim at both cells. Cons: about 12 more hours of `jebel`,
    mostly on `k` 1 rates the exclusion bound makes hard to read.
  - (c) Stop stage 7 now. Pros: `jebel` free at once. Cons: C3 without `m` 16 and C4 without its
    all-at-once arm: two registered questions left unanswered for 4-5 h.
- **D35. E8b's curriculum arm stopped; C8 re-registered as an existence test, E8c (the owner,
  2026-10-14).** E8b's per-step certificates (above) showed certified networks at step 4 that later
  steps destroyed, so its registered measure, the final network, can't see what training finds.
  The flaw is recorded here and fixed in a new registered experiment (experiment-design.md).
  - (a) Leave E8b as registered; certify every step as an exploratory measure. Pros: no setting
    changed after data; every step's network is saved, so nothing is lost. Cons: the registered
    verdict undercounts; `jebel` spends hours on steps after certification.
  - (b) *Chosen.* **Stop E8b now; register E8c** with `certify_steps` and `stop_at_certified`, both
    rungs. Pros: existence first (D33, the owner's strategy); the protocol matches C9's; `jebel`
    free for C9 hours sooner. Cons: E8b's curriculum arm stays incomplete (recorded), and `q` 4 is
    run again, briefly.
  - (c) Finish `q` 4 as registered, then `q` 5 as E8c. Pros: `q` 4's registered result complete.
    Cons: two protocols across the rungs; the verdicts don't compare.

  (A different "E8c", E8b's cells trained short with 100 seeds, was proposed on 2026-10-14 and
  declined, never registered; the name is this experiment's.)

## Stages

1. **This workplan and the literature review.** Docs only.
2. **Done.** **The certificate and the long tests** (tier 0): `indrajala_ml/studies/window_recurrent_certificate.py`
   (the enumeration, `phi`, the margins, the minimised machine) and a `certify` command and
   `check --properties`/`--sampled` options in `scripts/window_recurrent_study.py`. Tests: the
   harness's adding fake certifies at `k` 1, 2 and 4; a fake that drops a carry at a window
   boundary, one that carries only inside a window, and one with a near-tie each fail with the
   right counterexample; a certified fake passes B1 at 1,024.
3. **Done.** **E1 on `pyramidon`:** every matrix and control network certified, long-tested; findings in
   the certificate script's docstring; the certified networks archived (D12).
4. **Done.** **The continuous state** (tier 0: no timed path changed): `codes: Codes | None`, the
   three implementations skipping the snap, format 2 writing `"codes": null`, validation; parity
   tests, a fixture and a golden entry for each implementation (the golden run 121 entries to 124,
   the 121 bit-identical: checked).
5. **Done.** **The second task's harness and certificate** (D14-D19; tier 0, no timed path): a task in the
   harness (its column tokens, labels, reading order, catalogue of tests and target machine), the
   adder and modular division as instances; the certificate over a target machine (D17), E1's
   certificates reproduced bit for bit from the adder's instance. Tests: a modular fake
   certifies at `p` 5 and 7, a faulty one fails with the right counterexample. **Done:** E1's
   105 networks recertified by the new code (D18, D19), branched and strict, every field but the
   timings equal under D19's names: 105 compared and 0 differ against `certificates/`, 105 and 0
   against `certificates-strict/` (on `pyramidon`, 3 workers, 2026-10-13).
6. **Done (2026-10-13): C7 refuted as registered, 0 of 80 working or certified; findings in the
   study script's docstring, nothing to archive (D12: certified networks only).** **E7 on `jebel`:** the modular-division matrix (`scripts/window_recurrent_e7_p5.json` and
   `_p7.json`, `s` 16, D20), certified and long-tested: findings. **The findings' tables in**
   (for stages 6-8): `networks CONFIG` (each finished run's last network, for `certify` and
   `check`) and `verdicts CONFIG --certificates DIR --strict DIR [--long-tests DIR]` (per cell the
   runs working, certified and robustly certified with Wilson intervals; the runs by works and
   class; per class the networks long-tested and every failure). E1's tables and long-test
   figures come out of it unchanged.
   **The long tests' tool in:** `long-test NETWORK ... --task T [--properties ...] --out DIR
   --workers W` (D9's lengths and sizes, a network done before skipped), `check`'s results file
   for each; for E7, `--properties U1 S1 S2` (D22). It reproduces E1's results at 128 exactly (two
   networks, on `pyramidon`).
7. **E3 and E4 on `jebel`** (one config, the curriculum arm at every `m` and the all-at-once arm,
   about 240 runs; C0's reruns compared first): findings; E1's certificate run on every network.
   **Config in:** `scripts/window_recurrent_e3_e4.json`, 240 runs (the curriculum arm at `k` 4
   `s` 4, `m` 3, 4, 6, 8, 16, 32, and at `k` 1 `s` 16, `m` 3, 4, 8, 16; the all-at-once arm at
   `m` 16 and 8 of those cells; seeds 0-19), the matrix's settings otherwise; 6 workers. Its
   seeds 0-4 at the matrix's cells (30 runs) and the control's (10) are those runs again, same
   keys and settings (a test checks it), so C0 compares their result files.
   **Trimmed (D34, 2026-10-14):** at 79 of 240 runs, stopped and resumed at `cf6ac86` with
   `scripts/window_recurrent_e3_e4_trimmed.json` (the same directory and settings; a test checks it
   is a subset keeping C0's `k` 4 reruns): the `k` 4 `s` 4 cells only, the curriculum at `m` 3, 4, 6,
   8 and 16 and the all-at-once arm at `m` 16, 120 runs; `m` 32 and every `k` 1 cell dropped.
7a. **Done (2026-10-13): C7' refuted as registered, 0 of 80 working or certified; findings in the
   study script's docstring.** **E7' on `jebel`** (C7'; D23-D26), **before stage 7** (the owner, 2026-10-13: the second
   task's answer first; E3/E4 wait about 5 h), as soon as E7 ends and D26's check is read from
   E7's step records. **Configs in:** `scripts/window_recurrent_e7_prime_p5.json` and `_p7.json`,
   80 runs at `k` 1 (40 a `p`), E7's settings with D24's codes and 10 seeds (a test checks them
   against E7's). **D23's test in:** every committed study config's snapped cells have at least
   their task's minimal state count of codes (each target machine is minimal with every state
   reachable, `test_window_tasks`), E7's `p` 7 `m` 32 the one recorded exception. Then E7's
   analysis: certified and long-tested, findings.
7b. **E8 on `pyramidon`, then `jebel`** (C8; D27-D30), from 2026-10-14, beside stage 7.
   **In:** the modular sum (`modular-sum-q`: `window_tasks.ModularSum`, `modular_data.sum_catalogue`;
   the data module renamed from `modular_division_data`, division's cases unchanged),
   `scripts/window_recurrent_e8_q4.json` and `_q5.json` (a test checks them), `run --workers`.
   Then E7's analysis, and the next rung (D30).
7c. **E8b on `pyramidon`** (C8; D23's exclusion bound, D31): `scripts/window_recurrent_e8b_q4.json`
   and `_q5.json` (a test checks them, and every config after E8 is tested against the bound), the
   arms split by `run --arm` (all-at-once on `pyramidon`, curriculum on `jebel` after stage 7: E8b's
   launch; the all-at-once arm stopped after 6 runs, D33), then E8's analysis. If a rung works, the next (D30) is registered at the bound; division waits
   for the factored state (D32). **The curriculum arm stopped too (D35, 2026-10-14)**, in `q` 4's
   first 6 runs: its question moved to E8c.
7d. **E8c on `jebel`** (C8 as an existence test; D35): `scripts/window_recurrent_e8c_q4.json` and
   `_q5.json` (a test checks them against E8b's), from `main`, `q` 4 then `q` 5, each stopped at its
   first certified run; then the certified networks' long tests and findings.
8. **E2 and E6 on `jebel`** (the continuous arm and the baseline, about 100 runs): findings.
   **E2's config in:** `scripts/window_recurrent_e2.json`, 40 runs (the curriculum arm at `k` 4
   `s` 4 and `k` 1 `s` 16, a continuous state, seeds 0-19, E3's settings; its keys say
   `continuous` where E3's say `m<m>`). The harness takes `m` null as a continuous state
   (`WindowRecurrence(s, None)`) and records no codes for it (`codes_used` null). **E6's config
   in:** `scripts/window_recurrent_e6.json` (D21), 80 runs at `k` 1 `s` 16, E3's settings; a cell's
   `"window"` names its window (`WINDOWS`: `transformer`, the default; `linear`; `ffn`), and the
   run's key carries it (`curriculum-linear-k1-s16-m8-seed0`).
9. **E5 on `pyramidon`:** findings. **The tool in:** `scripts/window_recurrent_study.py explore
   CONFIG` (`explore_run`, `explore_summary`): every finished snapped run's last network, its codes
   over the codes property's first 1,000 cases at its check lengths and at 8, 16 and 64, the new
   ones, and failing against working runs whose check passed (Fisher's exact test). At `k` 4 the
   check lengths are 4, 8 and 12, so 8 is no length past training there: read 16 for it. About
   4.5 s and 145 MB a run on `pyramidon`.
10. **The report** (D11), the README section, the parent's stage 10 marked done; this workplan
   retired.

## Cost

From the matrix (a run's median 27 minutes on `jebel`'s 6 workers, 2-67), and the measured
evaluation rate:

| experiment | runs or networks | where | estimate |
| --- | --- | --- | --- |
| E1 | 105 networks: certificate and long tests | `pyramidon` | 3-6 h on one core; under 2 h on six |
| E3 + E4 | 100 curriculum + 20 all-at-once runs (trimmed from 240, D34) | `jebel` | about 13 h (79 runs before the trim, about 4-5 h after) |
| E2 + E6 | 40 continuous + 60 baseline runs | `jebel` | about 8 h (4-15) |
| E5 | saved networks only | `pyramidon` | under 1 h |
| E7 | 2 `p` x 2 `k` x 2 `m` x 2 arms x 5 seeds = 80 runs, certified | `jebel` | about 8 h (4-15), longer if 49 states train slowly |
| E7' | 2 `p` x 2 `m` x 2 arms x 10 seeds at `k` 1 = 80 runs, certified | `jebel` | about 5 h (3-8): E7's `k` 1 runs took 7-15 minutes; more codes, more epochs |
| E8 | 2 `q` x 2 arms x 10 seeds at `k` 1 = 40 runs a pair of rungs, certified | `pyramidon`, then `jebel` | about 2.5-5.5 h a pair on `pyramidon`, about half that on `jebel` |
| E8b | 2 `q` x 2 arms x 10 seeds at `k` 1, 20 steps = 40 runs, certified | all-at-once on `pyramidon`, curriculum on `jebel` | about 2.5-5 h on `pyramidon` (3 workers), 4-12 h on `jebel` (6) |
| E8c | 2 `q` x 6 seeds at `k` 1, curriculum, 20 steps, each step certified, stopped at the first certified run | `jebel` | `q` 4 about 30 min (expected at step 4), `q` 5 at most about 2-3 h |

About 34 hours of `jebel` in three sittings, the package change of stage 4 and the harness
change of stage 5.

## Known limits

- **One task, one base.** Base-3 addition of three operands; the claims are about this task.
  Base 10 or two operands would be a later replication (a new tokenisation, the same harness).
- **The certificate is for snapped networks.** It proves a given network correct, not that
  training finds one; the rates (C3-C5) are the evidence for that, with their intervals.
- **Compared to the literature by its figures,** not by reimplementation (D8 (c)): the Neural
  GPU's and Rational Transductors' tasks differ (binary or decimal, two operands).
- **20 seeds tell large differences only** (D4: 0.8 from 0.4, not 0.75 from 0.5); an undecided
  comparison is reported as such.

## Timing and checks

Tier 0 throughout unless stage 4 changes a timed path (it adds a path; the existing ones keep
their bits, checked by the golden run, which every package change passes).
