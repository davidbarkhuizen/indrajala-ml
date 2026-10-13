# Workplan: substantiating the window-recurrent adder's findings

**Status: written 2026-10-13, at the owner's request ("the work we would need to do to substantiate
our claims/findings"), after the matrix (stage 9 of
[window-recurrent-adder-workplan.md](window-recurrent-adder-workplan.md)) and while its control
ran. Decisions D1-D12 settled under the owner's standing delegation, each with its options' pros
and cons. Stages 1-4 done (stage 2 refined D3's snap margin: a near tie branches).** The literature it answers to is in
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

### E4. Curriculum against all at once (C4, C5)

**Design.** The all-at-once arm (W16) at D5's cells, at the matrix's best `m` (16 at `k` 4 `s` 4,
8 at `k` 1 `s` 16), seeds 0-19. Its curriculum counterpart is E3's runs at the same `m`.

**Measures.** Works; certified; examples to pass (the curriculum's over its three steps, the
control's in its one, D7).

**Analysis.** Working rates and Fisher's exact test per cell; examples to pass compared by the
Mann-Whitney test over the working runs (D4). C4 holds if the rates don't differ; whatever the
result, the claim is written as the data show.

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
anything at `k` 1, where it sees one column.

**Analysis.** Working rates against E3's `k` 1 runs at the same `m` (Fisher's exact test). The
windows of `k` 2 and 4 need the transformer to add inside the window, so the baseline is at `k` 1
only.

### E7. Modular division, the second task (C7; D14-D17)

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
5. **The second task's harness and certificate** (D14-D17; tier 0, no timed path): a task in the
   harness (its column tokens, labels, reading order, catalogue of tests and target machine), the
   adder and modular division as instances; the certificate over a target machine (D17), E1's
   certificates reproduced bit for bit from the adder's instance. Tests: a modular fake
   certifies at `p` 5 and 7, a faulty one fails with the right counterexample.
6. **E7 on `jebel`:** the modular-division matrix, certified and long-tested: findings.
7. **E3 and E4 on `jebel`** (one config, the curriculum arm at every `m` and the all-at-once arm,
   about 240 runs; C0's reruns compared first): findings; E1's certificate run on every network.
8. **E2 and E6 on `jebel`** (the continuous arm and the baseline, about 100 runs): findings.
9. **E5 on `pyramidon`:** findings.
10. **The report** (D11), the README section, the parent's stage 10 marked done; this workplan
   retired.

## Cost

From the matrix (a run's median 27 minutes on `jebel`'s 6 workers, 2-67), and the measured
evaluation rate:

| experiment | runs or networks | where | estimate |
| --- | --- | --- | --- |
| E1 | 105 networks: certificate and long tests | `pyramidon` | 3-6 h on one core; under 2 h on six |
| E3 + E4 | 200 curriculum + 40 all-at-once runs | `jebel` | about 18 h (10-30) |
| E2 + E6 | 40 continuous + 60 baseline runs | `jebel` | about 8 h (4-15) |
| E5 | saved networks only | `pyramidon` | under 1 h |
| E7 | 2 `p` x 2 `k` x 2 `m` x 2 arms x 5 seeds = 80 runs, certified | `jebel` | about 8 h (4-15), longer if 49 states train slowly |

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
