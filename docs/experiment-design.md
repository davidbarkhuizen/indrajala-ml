# Experiment design

How a learning study in this project is designed, run and judged: a study that trains networks
and makes claims from them (the addition and sequence studies, the window-recurrent matrix, its
claims plan's experiments). [measurement.md](measurement.md) is the other half: how a change is
*timed*. Read this before designing a study or writing a study workplan.

Each rule names where it was decided and what taught it, so it can be argued with from the
evidence. "The claims plan" is the window-recurrent claims workplan, retired; its report, with the
`git show` for the plan, is [window-recurrent-report.md](window-recurrent-report.md).

## 1. Every condition must be able to succeed

Before a cell (one setting of a study's matrix) is configured, its necessary conditions are
computed and written beside it in the workplan, and a test checks the committed config against
them. A cell below a bound is dropped unless it is kept for a stated reason (a negative control, a
threshold probe), recorded with it. (The owner's rule, 2026-10-13; the claims plan's D23.)

- **The model can represent a solution.** For a snapped recurrent network: the code count `m` at
  least the target machine's *minimal* state count, computed (partition refinement), not assumed.
  Likewise the state width, window, vocabulary and lengths. *E7 ran modular division mod 7 with 32
  codes against a 49-state automaton: 20 runs that could never work, missed by the plan and its
  review.*
- **The per-step channel has the capacity too** (2026-10-19): the window's residual stream carries
  everything the next state and the answers need, so its width is a capacity bound like the codes'. A
  cell's window is at least the width shown sufficient on a comparable cell, and an architectural
  change is tested at a width where its base works. Splitting a too-narrow stream divides its
  capacity. *The window branches plan tested a single-head branch per consumer at width 16, which
  L3 had shown too narrow (division 13: 0 of 6 at 16, 3 at 22, 5 at 32); the factors' transitions
  starved (B1 1 of 6; B3's S4 0 of 5 against round 1's 4 of 6), a second infeasible design.*
- **The check can see the target:** every target state reachable, and every pair of states told
  apart, within the check lengths.
- **Training excludes wrong solutions: the exclusion bound.** Two finite machines of `n` and `n'`
  states that agree on every input up to length `n + n' - 1` are equivalent. So a network of `m`
  codes exactly right on every input up to length `L` is the target's `q`-state machine when
  **`L >= m + q - 1`**; below it, a wrong machine of up to `m` states can fit every training case,
  and training finds one. Design the training set so it can't be memorised and generalisation is
  forced. *E8 (the modular sum) trained to 3 columns with 32 codes: 37 of 40 runs fitted
  their training lengths exactly and failed at 8 columns, their machines minimising to 6-22 states
  against 4-5. Every earlier experiment, addition's included, fell short of the bound.* A test
  holds every config registered after E8 to it.
- **Surplus capacity is kept, and paid for in training length** (the owner, 2026-10-14). Spare
  codes help training (addition worked best at about 5 times its states; E8's one certified
  network used 8 of its 32 codes; in E8b's one probe run, a codebook at exactly the state count
  didn't fit its training lengths; E8c's certified counters at `m` 16 reached 7-13 codes for 5
  states before minimising: learning used about twice the states), so `m` stays at about 3-5 times
  the states and `L` grows to meet the bound, rather than `m` shrinking to it.
- **An existence test takes its cell from what worked, and a wall-clock budget** (the owner,
  2026-10-14: "existence first, then refine"; "when this architecture works, it tends to be
  clear-cut and early"). Reuse the settings of the nearest certified networks (their codes used,
  not only their `m`); shrinking toward the minimum is the refinement after a network exists. The
  cap stays at the bound (feasibility), and `budget_hours` stops the study at a deadline a few
  times past the step where the evidence says it certifies (the factored-state workplan's D13).
- **The budget is measured before launch:** epoch time and memory from a probe run (one seed, one
  cell), then the study's estimate as a range. No study long enough to matter starts on a guess.

## 2. Claims are registered before the data

- A study's claims are written in its workplan before it runs, each with the experiment that tests
  it and **what would refute it** (the claims plan's table). The analysis (tests, intervals,
  corrections) is fixed with them (D4).
- **No settings change after seeing results.** A flaw found mid-study is recorded and the study
  runs as registered (E7's impossible cells ran on as a negative control); the fix is a new,
  separately registered experiment (E7', E8b). A conditional rule registered beforehand is
  allowed ("if the step records show runs cut off while improving, use the longer budget": D26).
- **One change between an experiment and its follow-up** (D26), so a difference has one cause.
- **Get a small case working before stepping up** (the owner, 2026-10-14): a ladder of task sizes
  (the modular sum at `q` 4, 5, ... before division's 25 states), each rung registered with its
  bounds, the next registered only after a rung works (D30).
- **Existence first; spend compute where a working network can come out** (the owner,
  2026-10-14; the claims plan's D33). A study first finds a configuration that yields a working
  (certified) network, then searches for its bound (a binary search along the axis in question);
  rates and wide matrices come after. A long run that the evidence says can't yield a working
  network (a cheap probe of a few epochs and seeds is enough) isn't launched, or is stopped and
  recorded. E8b's all-at-once arm was stopped after 6 runs: a probe showed its state stuck on one
  code at training lengths 1-20.
- **Probes first; long runs only with a stated reason** (the owner, 2026-10-15, after C9'': its
  4 h on two machines told less than a 10-minute probe of its saved networks). Work by short
  probes (minutes to about an hour, a few seeds, one question, success and failure written before
  the data), the next chosen from each result, extended or varied as needed. A probe claims
  nothing; a claim still needs its registered test and a certificate. Probes are kept as repo
  tools and their results with manifests (the modular-family and commitment-control workplans).
- **Short, focused experiments** (the owner, 2026-10-14): a matrix only where a question needs
  one. After a lone working network, a short follow-up with more seeds of its configuration,
  sized from the observed rate and labelled exploratory.

## 3. Statistics

Fixed before any run (the claims plan's D4), implemented by hand in
`indrajala_ml/studies/inference.py` (no scipy):

- Rates with **Wilson 95% intervals**; two arms by **Fisher's exact test**, two-sided; a trend over
  an ordered factor by **Cochran-Armitage**; a continuous comparison by **Mann-Whitney**; a family
  of tests corrected by **Holm** at 0.05.
- **Seeds:** 20 a cell tells 0.8 from 0.4 (Fisher p 0.023) and not 0.75 from 0.5 (p 0.19); 5 a cell
  is a screen, and its rates are reported as such. An undecided comparison is reported undecided.
- A look the plan didn't register is labelled **post hoc**, and never stands in for the
  registered test (the addition code-count trend before E3: "post hoc, not C3's test").

## 4. What counts as evidence

- **Sampled tests overstate "works".** Of 57 networks whose tests passed (W10), 2 passed every test
  to 64 columns and were wrong at 12, and a long test of 328,000 cases missed one of them (the
  claims plan's E1, D13). Where a model admits proof, **certify** it (the window certificate:
  every window from every reachable state, against the target machine), and report the sampled
  verdict and the certificate side by side (D13).
- **Test at full catalogue sizes.** Reduced samples passed lengths the full sizes fail on a
  near-working network: a rare failure is what separates working from a near miss (W12).
- **Long tests are an independent check** of the certificate, at lengths far past training (D9:
  128, 256 and 1,024 columns); they confirm, they don't establish.
- **A near tie is not a pass:** a decision within a margin (a snap within 1.05 of the next code)
  is branched both ways or reported apart (D3).
- **Diagnose before redesigning:** read the failures (where runs break, what the certificates'
  counterexamples are, how many states the learned machines minimise to) before choosing the next
  experiment. E7's said *fitting* (transitions wrong at training lengths); E8's said *memorising*
  (exact at training lengths, larger machines): different fixes.

## 5. Reproducibility and records

- **Deterministic runs:** seeded generators, a fixed draw order, one BLAS thread
  (`OPENBLAS_NUM_THREADS=1`), so a rerun reproduces its result file up to timings (C0; the claims
  plan). The E6 probe reproduced matrix runs bit for bit across `jebel` and `pyramidon`.
- **A study may move between machines** (D29: `run --workers W`; the harness resumes from disk),
  only between whole runs (a curriculum run's resume from a middle step isn't checked for bits),
  on one commit.
- **Records:** result files are gitignored; each study's findings go in its script's docstring
  with every file's path, host and sha256 (the study-runs workplan's D8), copied to both machines;
  certified networks are archived in `davidbarkhuizen/indrajala-models` (D12).
- **Long jobs:** never let one swap (size workers as measurement.md section 2 says);
  `jebel` at or below 65 W; studies hold `jebel`'s machine lock with `ab.py` (measurement.md).

## 6. Writing a study's workplan

A study workplan states, before any run: its claims and refutations; its cells with each one's
feasibility bounds (section 1) in a table; its seeds and statistics; its budget from a probe; where
it runs; where its findings go. Its decisions give every option's pros and cons. Its configs are
committed with tests that check them against the plan (cells, settings, bounds).
