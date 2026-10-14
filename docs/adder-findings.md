# What the adder studies have found

A summary of the findings so far (2026-10-14) along the line of work that started from the
addition study. Each section names where its findings are recorded in full, with their data,
provenance and checksums; this document adds no new results and repeats no figure that isn't
there.

The task throughout is base-3 addition of three numbers (two-operand addition is a zero in one
slot). The carry into a column is 0, 1 or 2, so a correct adder of any length is a 3-state machine
run along the columns.

## 1. Fixed-length adders (the addition study)

*Recorded in `scripts/addition_study.py`'s docstring and the README (The addition study); the
retired workplan is `git show 8990f74:docs/addition-study-workplan.md`.*

- **Only the column transformers (a token per column) passed calibration.** The FFN-only control,
  the dense networks and the string format didn't, so the sweep (619 runs on `jebel`) searched
  those three encodings, at `n` 4 and 6, with 1-3 layers and 1-2 heads.
- **The smallest reliable adders are small:** 2 layers at `d` 6 at `n = 4` (831 parameters for the
  unordered encoding, 933 for the others), and 3 layers at `d` 6 at `n = 6` (1,197 and 1,299). `d` 4
  fails everywhere; one layer passes at `n = 4` only wider and after many more epochs, and never at
  `n = 6` up to `d` 32.
- **What fails is the carry over distance** (M3, M5). Failing runs mostly still pass the column
  sum, identity and the carry out of a reset column whole.
- **The networks sit on a sharp edge:** near the threshold one size up flips a run from failing to
  passing, and some ladders are non-monotone.

## 2. Why a fixed-depth network can't add at any length

*Recorded in [arbitrary-length-addition-workplan.md](arbitrary-length-addition-workplan.md) and
[next-steps.md](next-steps.md) (From length generalisation); the terminated plan's literature
section is `git show c3d619b:docs/length-generalisation-workplan.md`.*

- A carry can pass unchanged through a run of columns of any length, so an adder must find a
  column's deciding column at any distance below it.
- A fixed-depth network that finds it in one long hop needs position codes that stay
  distinguishable at any distance, which bounded codes can't give: rotary positions lose
  resolution, and no positions dilute attention as the length grows.
- The owner terminated the length-generalisation plan (2026-10-12) for the **window-recurrent
  adder** ([window-recurrent-adder-workplan.md](window-recurrent-adder-workplan.md)): a network
  reads a window of `k` columns and hands a state of width `s`, snapped to a codebook of `m` codes,
  from window to window. The recurrence, not depth, carries the carry.

## 3. The window-recurrent adder works about half the time

*Recorded in `scripts/window_recurrent_study.py`'s docstring (Findings (stage 9) and the control).*

- **45 of the matrix's 90 runs work** (a first failing length, the break, of 16 columns or more),
  after training on at most 3 windows (3, 6 and 12 columns at `k` 1, 2 and 4). 43 of them pass every
  property at every test length up to 64 columns.
- **It takes the recurrence:** no run works after training on one window, which never hands a state
  on; 9 work from 2 windows, 36 more from 3.
- **The seed decides, not the cell.** No cell works on every seed, and every cell has a state that
  can hold the carry: what fails is the optimisation, not the capacity. More codes help (`m` 4, 8
  and 16 work in 10, 16 and 19 of 30 runs); the state's width barely matters (`s` 4 and 16: 24 and
  21 of 45).
- **Failure is mostly all or nothing:** 43 of the 45 failing runs fail at 8 columns already.
- **Exact on short sums isn't generalising:** 5 runs at `k` 1 passed every sum of 1-3 columns and
  still got 0.2-8.3% of B1 wrong at 8.
- **The curriculum isn't needed here** (the control, 15 runs): training on 1-3 windows at once
  worked as often as the curriculum at the same cells and seeds (12 of 15 each), from about 16 times
  fewer examples where it worked. At 5 seeds the difference is undecided; E4 measures it at 20.

## 4. A proof beats testing (E1, the certificate)

*Recorded in the same docstring (Findings (E1)); the claims are in
[window-recurrent-claims-workplan.md](window-recurrent-claims-workplan.md).*

- **A snapped network can be proved correct at every length:** enumerating its codes and windows
  gives the machine it computes, and minimising that machine compares it with the task's. The
  certificate ran over every network the matrix and the control tested, 105.
- **Every certified network's machine minimises to the 3-state carry automaton**, while reaching
  3-11 codes (median 5): the extra codes are equivalent copies of a carry value.
- **The sampled tests overstate working (C1 refuted as registered).** Of the 57 networks that work
  by the tests, 52 are certified and 48 robustly certified (near ties branched both ways). **2
  networks that passed every test up to 64 columns are provably wrong**, each confirmed by running
  it on the certificate's counterexample (at 8 and 12 columns). "Works" now means certified.
- **The certificate is borne out where it can be tested:** every certified network passes every
  long-test case (17.06 M cases at 128, 256 and 1,024 columns), no failing network certifies, and
  every network failing a long test is one the certificate rejects. The long tests missed one of
  the two counterexample networks entirely (all 328,000 of its cases passed).

## 5. A harder task: modular division (E7, E7')

*Recorded in the same docstring (Findings (E7) and (E7')).*

- **E7 (C7): 0 of 80.** Its `p` 7 `m` 32 cells couldn't succeed: the automaton has 49 states and the
  codebook 32 codes (a negative control, run as registered). At `k` 4 no run fits its training
  lengths; at `k` 1 every run is a near miss at length 3, the first length to use every transition.
- **E7' (C7', enough codes): 0 of 80** at 5 and 10 times the states' codes. The networks now hold
  codes for nearly every state, yet 79 of the 80 counterexamples are a wrong next state: **the
  states are represented, the transitions between them aren't exact.** Fitting, not codes, is the
  limit. More codes cut the errors at length 3 but none reach zero (the best, 4 of 729 pairs wrong).
- **D32 (the owner):** division waits for a **factored state**, a codebook per residue, since its
  state is two independent counters `(A mod p, B mod p)`, each a modular sum's.

## 6. The modular-sum ladder and memorising (E8)

*Recorded in the same docstring (Findings (E8)); the bound is the claims workplan's D23.*

- **C8 holds at `q` 4, weakly; not at `q` 5.** One network certifies at `q` 4 (1 of 20): its 8
  codes minimise to the 4-state automaton, and it passes all 312,000 long-test cases. None of 20 at
  `q` 5.
- **The failures are memorising, the opposite of E7's:** 37 of the 40 runs passed every check at
  lengths 1-3 and failed at 8. Their codes minimise to machines larger than the task's (6-22
  states against 4 and 5).
- **The exclusion bound:** a network of `m` codes exact on every input up to length `L` is the
  `q`-state machine only if `L >= m + q - 1` (two machines of `n` and `n'` states that agree on
  every input up to `n + n' - 1` are equivalent). E8 trained to 3 columns with 32 codes, where 35-36
  were needed, and every earlier experiment, addition's included, trained short of its bound:
  memorising was never excluded.

## The rules these findings set

*Recorded in [experiment-design.md](experiment-design.md) and CLAUDE.md (Studies).*

1. **Every cell must be able to succeed:** at least as many codes as the target's minimal states.
2. **Training must exclude memorising:** the longest training length at least `m + q - 1`.
3. **Keep a codebook surplus** (3-5 times the states) and pay for it in training length: a codebook
   at the bound trains badly (E8b's probe, D31).
4. **Register claims before the data; certify where a model admits proof.** Sampled tests overstate
   working.

## Open

- **Stage 7 (E3 and E4, 240 runs on `jebel`):** addition's rates at 20 seeds a cell (C2-C5) and the
  matrix's reruns (C0), running since 2026-10-13; findings when it ends.
- **E8b (C8 past the bound; D31):** the modular sum at `q` 4 and 5, `m` 16, trained to 20 columns,
  the first experiment built to exclude memorising. Its all-at-once arm runs on `pyramidon`, its
  curriculum arm on `jebel` after stage 7.
- **Then:** the ladder's next rung if E8b climbs it (D30), the factored state for division (D32),
  and stages 8 and 9 (E2's continuous state, E6's baseline windows, E5's code probes).
