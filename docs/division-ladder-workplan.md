# Workplan: where division fails in `p` (a quick existence ladder)

**Status: written 2026-10-17 at the owner's request: "what is the fastest existence test ladder
we can do to see where our arch fails ito modular division? ... quick and dirty, but still
statistically significant ... the goal being to see where we fail ito p", before moving to the
general algebras. D1-D6 settled under the owner's delegation, each with its options' pros and
cons. L1 registered here, before its data.**

## Why

Division mod `p` is a `p^2`-state transducer. Certified so far: `p` 5 (13 of 20, registered), `p` 7
(5 of 8, screens) and `p` 11 (4 of 6, stage 3 of the p11-trigger workplan, retired: `git show
28894cf:docs/p11-trigger-workplan.md`). Nothing above 121 states has been tried. One quick sweep up
the primes finds the first `p` where the architecture stops yielding a certified network, as a
point for the papers and a calibration before the general algebras.

## Decisions

- **D1. The cell: `p` 11's, with only `p` changed and the capacity scaled with it (D2):** the
  family, two factors of width 16, reading the output, the fork with today's trigger, the guard
  off, 600 epochs a run, stop at certified.
  - (a) *Chosen.* Pros: the cell that certified 121 states; the rung differs from it in `p` alone
    (and the codes D2 ties to `p`). Cons: tuned at `p` 11 (rate, epochs); a failure higher up may be
    the settings', not the architecture's (Known limits).
  - (b) Re-tune per rung. Cons: slow, and a failure point then mixes the tuning in.
- **D2. Codes: each factor `ceil(32 p / 11)`**, `p` 11's ratio (32 codes for an 11-counter; joint
  codes 8.5 times the states) at every rung.
  - (a) *Chosen.* Pros: the surplus rule (experiment-design.md, section 1) held constant, so
    running out of codes isn't what fails. Cons: two things grow with `p` (the task and the
    codebook), so the failure point is the pair's.
  - (b) 2 x 32 at every rung. Pros: one variable. Cons: at `p` 23 the joint codes are 1.9 times the
    states, below the surplus kept; at `p` 37 below the states (infeasible); a failure would be
    capacity's.
  - (c) Powers of two. Cons: the surplus jumps by 2 between rungs.
- **D3. Six seeds a rung (0-5); "works" if at least one network certifies in both verdicts;
  "fails" at 0 of 6.** Pros: 0 of 6 against `p` 11's 4 of 6 is Fisher one-sided p 0.030, Wilson
  upper bound 0.39: significant at the 0.05 level with the fewest runs; 6 a rung fills half of
  `jebel`'s 12-run rounds. Cons: 1-3 of 6 is no significant fall against 4 of 6 (1 of 6: p 0.12),
  so it's read as a decline, undecided, not a failure; `p` 11's 4 of 6 was on the crate before its
  `exp` changed (aarch64 stage 2), the best baseline there is.
- **D4. A binary search over primes, two rungs a round.** Round 1: `p` 17 and 23 together.
  Round 2 by rule: both work, 31 and 41; 17 works and 23 fails, 19; 17 fails, 13. Each later round
  bisects the primes left between the highest working rung and the lowest failing one (the two
  thirds' points when two remain). The search stops when those are adjacent primes, or at 41
  working.
  - (a) *Chosen.* Pros: the bracket in two rounds (about two nights' halves), against four or
    five rung by rung. Cons: skips the gentle ladder (experiment-design.md, section 2, one rung
    after another); each skipped rung is filled only where the bracket needs it.
  - (b) Rung by rung (13, 17, 19, 23, ...). Pros: the protocol's ladder. Cons: twice the rounds.
- **D5. One process for a round:** `modular_family_probe.py run` takes several `--p` and
  `--codes-ratio R` (each factor `ceil(R p)` codes), every rung's seeds in one pool, into
  `DIR/p<p>/seed<s>`, under the machine lock it holds. Pros: both rungs under one lock, interleaved
  so a round ends together. Cons: a small change to the probe.
- **D6. Long tests on one certified network a rung** (the first by seed), U1 and S1 at 128, 256
  and 1,024 on `pyramidon`; the others only if one fails. Pros: the certificate establishes; the
  long tests confirm (experiment-design.md, section 4), at about 2 h a network. Cons: not every
  network long-tested as the earlier studies did.

## The claim (registered before the data)

**L1, per rung:** **works** if at least one of seeds 0-5 certifies (both verdicts) against
division's `p^2`-state machine; **fails** if none does (0 of 6: below `p` 11's 4 of 6, Fisher
one-sided p 0.030). A working rung with 1-3 of 6 is also labelled **a decline, undecided** (no
significant fall at 6 seeds); for D4's search it works.
**The failure point** is the smallest failing rung above a working one, bracketed by D4's rounds.

Reported per run: certified or not, step, epochs, forks, minimised states, joint codes reached, the
worst answer at its best; per rung the count with its Wilson interval. Whether successes crowd the
600-epoch budget (the slowest success's epochs) is read beside each rung, since a failure under a
crowded budget is the budget's as much as the architecture's.

## Feasibility (experiment-design.md, section 1)

Two factors of width 16, each `ceil(32 p / 11)` codes (a counter needs `p`); the cap is the
exclusion bound `m + q - 1` with `m` the joint codes.

| `p` | target states `q` | codes a factor | joint codes `m` | `m / q` | exclusion bound = cap |
| --- | --- | --- | --- | --- | --- |
| 11 (stage 3) | 121 | 32 | 1,024 | 8.5 | 1,144 |
| 13 | 169 | 38 | 1,444 | 8.5 | 1,612 |
| 17 | 289 | 50 | 2,500 | 8.7 | 2,788 |
| 19 | 361 | 56 | 3,136 | 8.7 | 3,496 |
| 23 | 529 | 67 | 4,489 | 8.5 | 5,017 |
| 29 | 841 | 85 | 7,225 | 8.6 | 8,065 |
| 31 | 961 | 91 | 8,281 | 8.6 | 9,241 |
| 37 | 1,369 | 108 | 11,664 | 8.5 | 13,032 |
| 41 | 1,681 | 120 | 14,400 | 8.6 | 16,080 |

No run reaches its cap in 600 epochs (`p` 11's certified at steps 10-31); every success is
certified, which is exact at every length.

## Known limits

- One cell's settings, tuned at `p` 11: the failure point is this cell's, within 600 epochs.
- A screen: 6 seeds a rung finds the point; it doesn't measure the rates around it.

## Stages

1. **This plan.**
2. **The build**, one PR, tier 0: the ladder's primes in `modular_data.PRIMES` (the certificate's
   and data's parametrised tests run on them), D5 in the probe, a test of the feasibility table
   against the config.
3. **Round 1** (`p` 17 and 23) on `jebel`, 12 runs, 8 at once; recertified on `pyramidon`. From a
   worktree pinned at the build's merge, holding the machine lock: `modular_family_probe.py run
   --p 17 23 --arm family --factors 2 --codes-ratio 32/11 --reads output --fork --budget-epochs 600
   --budget-minutes 900 --first-seed 0 --seeds 6 --workers 8 --out
   data/window-recurrent/division-ladder/round1`. Its budget, from `p` 11's runs (35-120 min at 6
   at once, the network the same size): 3-5 h; a few epochs at each rung on `jebel` first check
   the epoch time and memory.
4. **Round 2** by D4's rule.
5. **Findings** in `modular_family_probe.py`'s docstring, P2's notes, the networks to
   indrajala-models; this plan retired.
