# Workplan: after paper 1, sound re-tests of crystallisation and branching, then depth

**Status: drafted 2026-10-24 at the owner's direction: "we will pick this up again once: 1. we are
done with paper 1; 2. we have conducted rigourous (statistically sound) tests of all the other work
we have covered since then - which was mainly annealing / crystallization, and maybe also branching
might be worth re-testing properly; at that point, we should [test] if increasing network depth
helps with cases where we are otherwise struggling", isolated ("agreed on properly isolating it");
and on branching: "is it possible to have an architecture where we branch as we are (on
factors/code-books) - and then recombine again before output?", "let's incorporate it into the
branching study". D1-D14 settled under the owner's delegation, each with its options' pros and
cons. Nothing here runs or is built before paper 1 is out (D2). Items marked *pending* are filled at
registration (stage 1) by the rules stated here, before any of this plan's data.**

The goal is [machine-discovery.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/research-notes/machine-discovery.md)'s
stage A: one general architecture and training method that forms a state, keeps it and certifies on
every backdrop. The order in [next-steps.md](../next-steps.md), "After paper 1", is this plan's.

## The IDs, in time order

| ID | When | Short label | What it is |
|---|---|---|---|
| D1 | 2026-10-24 | one plan, three parts | crystallisation (A), branching (B), depth (C), in that order |
| D2 | 2026-10-24 | the gate | nothing built or run before paper 1's release |
| D3 | 2026-10-24 | the common control | the paper cell; T2's and T5's runs reused, with a reproduction check |
| D4 | 2026-10-24 | the backdrops | A on D64's five; B and C on the struggling set, by a rule fixed now, and a no-harm backdrop |
| D5 | 2026-10-24 | crystallisation's arms | `continuous`; `continuous-reheat` only if the collapse's (D63's) paired verdict favours it |
| D6 | 2026-10-24 | crystallisation's claims | A1 against the paper cell, A2 on the holdouts |
| D7 | 2026-10-24 | the recombining window | branch a factor, sum the branches' ends, one shared block, the output |
| D8 | 2026-10-24 | branching's arms | `branches-32` (the retired design at the base width, two heads a branch) and `recombine`, both 3 blocks deep |
| D9 | 2026-10-24 | branching's claims | B1, B2 against the paper cell; B1 against B2 reported apart |
| D10 | 2026-10-24 | adoption | helps where struggling, harms nowhere: then the default for part C |
| D11 | 2026-10-24 | depth's arm | 6 blocks against 3, the default's other settings unchanged |
| D12 | 2026-10-24 | depth's claim | C1 against the default |
| D13 | 2026-10-24 | seeds and statistics | 20 fresh seeds a cell; registered one-sided "helps", both sides reported; Holm a part |
| D14 | 2026-10-24 | budgets and machines | each cell's registered epochs; time and memory from a probe; the batch runner |
| stages 0-7 | 2026-10-24 | the plan's stages | the gate, registration, builds, probes, A and B, their reading, C, findings |

From earlier plans: the collapse (D63), the fresh-seed rate (D64), the deadline's completion (D67)
and the method's constants (F10) in [state-formation-workplan.md](state-formation-workplan.md); the
window branches' W1, W2, B1-B3 (`git show e7f1ff6:docs/window-branches-workplan.md`); the paper's
T2 (D15) and T5 (D16) in [paper-workplan.md](paper-workplan.md); the feasibility register's rows in
[information-flow.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/protocols/information-flow.md).

## Why

**Crystallisation (part A).** The state-formation strand's method, the continuous crystallisation
(D54's cell, the margin stop under D60: the batch runner's `continuous` cell), is out of paper 1
(the paper's D2) because it rests on two backdrops: D64 certified S3 and move-to-front-4 7 of 12
(between its bars) and S4, random-q12-0 and random-q16-0 0 of 10 (with D56-D58, S4 1 of 6,
random-q12-0 1 of 8, random-q16-0 0 of 7). Every failure ended frozen. Its design seeds were S3's and
move-to-front-4's; it was never set against the paper cell on the same backdrops, and the collapse's
remedies (D63: reheat and restarts) have no paired verdict yet (D67 completed their runs). Its
constants were never read for sensitivity (F10; not in this plan: a per-constant reading belongs
after a rate exists to read it against).

**Branching (part B).** The retired window branches (`--branches`: W11's first two blocks as a
trunk, then a single-head block a factor and one for the answers) ran at width 16, which division's
ladder had shown too narrow (L3: 0 of 6 at 16, 5 of 6 at 32), and with one head a branch: B1 and B2
1 of 6 a cell, harm on S4 (B3, 0 of 6 against 4 of 6, p 0.030 one-sided). The feasibility register
names both as design faults (rows 3 and 4), so the branches were never tested where they could
succeed. In that design nothing recombines: each factor's pooling reads its branch's end, and the
answers read their own branch, which reads only the trunk.

**The recombining window (the owner's question).** A Mealy machine's output `lambda(q, a)` depends
on the whole state; when the state is factored, `q = (q_1, ..., q_G)`, the transition can be computed
a factor at a time (a cover: each factor's next state from its own and the others'), but the answer
generally needs every factor at once. Branching per factor gives each factor's transition its own
path; recombining before the output gives the answers a stream that sees every factor's path, rather
than the trunk alone. It keeps the state factored (one codebook a factor; memory `factored-not-flat`)
and adds a merge only on the output's side.

**Depth (part C).** The window has been three blocks deep (W11's `BLOCKS` 3) through every study
since W11; depth was varied only in the primitives' early studies (plain networks got worse with
depth, residual ones didn't; two transformer layers beat one). Where the default still fails, depth
is the untested axis of the window's capacity beside width (row 3), which W1 showed matters.

## The design

### The recombining window (D7)

With `G` factors and the base width `D` 32:

1. **The trunk:** the Embedding, the Position and W11's first block (a `StateInput`, attention of
   two heads, the FFN).
2. **The branches:** one a factor, each a block of its own (a `StateInput`, attention of two heads,
   the FFN, a LayerNorm) over the trunk's tokens. Factor `g`'s pooling reads branch `g`'s end, as in
   `--branches`.
3. **The merge:** the branches' ends summed, token by token (each `D` wide, so the sum is too).
4. **The recombining block:** a `StateInput`, attention of two heads, the FFN, a LayerNorm, over the
   merged tokens; the output layer reads its end.

Every path is three blocks deep, as W11's and `branches-32`'s (D8), so part B changes the paths'
separation and not depth, and depth stays part C's alone. The answers' path passes through every
factor's branch; the factors' paths don't pass through the answers' block, so the answers' loss
shapes the branches but no branch reads the answers' features (`--branches`'s separation, kept).

**The build:** a `merge` op in the crate's window program (`indrajala-math-rust` first, then a "Bump
rust/" PR), ending the branches and summing their ends into the current activation; backward copies
the gradient to each branch's end, as `add` does for a residual. `Branches(..., merge=True)` in the
layer specs (format 2's op list gains `merge`; a reader that predates it refuses such a network);
the numpy path and the window program both run it, with parity tests. Today's programs have no
`merge` op: the golden run is unchanged.

### Depth (D11)

`--window-blocks N`: `transformer_window`'s block count (W11's 3 by default), every block W11's. The
golden run unchanged at 3.

## Decisions

- **D1. One plan, three parts: (a).**
  - (a) *Chosen.* One plan, parts A, B, C in the agreed order, sharing a control, a backdrop rule and
    one registration. Pros: the order and the shared control are stated once; A and B run as one
    batch (both are step 2 of the order), the machines filled; C's base follows from A's and B's
    adoption by a rule written now. Cons: a longer plan; the state-formation plan's remaining test
    items (the panel, 4b; D63's paired reading) move here, so that plan needs a note pointing here
    (stage 1).
  - (b) Three plans. Pros: each smaller. Cons: the control, the backdrop rule and the adoption rule
    repeated three times; C's dependence on A and B split across plans.
  - (c) Extend the state-formation plan with A, and a plan each for B and C. Pros: A sits with its
    history (memory `read-open-workplans-first`: extend, don't duplicate). Cons: that plan is 2,285
    lines of a probe-driven strand; a registered re-test against the paper cell is a different kind
    of work, and B and C have nothing to do with it.
- **D2. The gate: (a) nothing built or run until paper 1's release (its stage 8).**
  - (a) *Chosen.* Pros: the owner's order; the paper's machines and attention undivided; the paper's
    T2 and T5 read first, which this plan's backdrop rule needs (D4). Cons: the builds (D7, D11) could
    be done offline earlier.
  - (b) The builds now, the runs after. Pros: ready sooner. Cons: against the owner's "don't worry
    about this now"; a crate change in the middle of the paper's batches is a risk to them.
  - (c) After the paper's runs, before its manuscript. Pros: the machines are idle while it is
    written. Cons: not the owner's order ("done with paper 1"); the manuscript may need reruns.
- **D3. The common control: (b) the paper cell, its T2 and T5 runs reused, two seeds a backdrop
  re-run to check they reproduce.**
  - (a) A fresh control in the same batch, 20 seeds a backdrop. Pros: concurrent, the same commit.
    Cons: about 100-160 more runs repeating ones already on disk.
  - (b) *Chosen.* Pros: the paper cell's 20-seed rates on every backdrop exist (seeds 300-319);
    runs are deterministic and bit-identical across machines (experiment-design.md, section 5), so a
    rerun of two seeds a backdrop on this plan's commit with `same-prefix` shows the control is the
    same cell; the golden run checks training is unchanged. Cons: a control from an earlier commit;
    if a re-run differs, the control is run fresh (the rule: any difference, then (a) for that
    backdrop).
  - (c) No control: the arms' rates alone. Cons: no comparison; nothing could be adopted.
- **D4. The backdrops.**
  - **Part A: D64's five**: word-problem-s3, move-to-front-4, word-problem-s4, random-q12-0,
    random-q16-0, all in T5 with the paper cell. S3 and move-to-front-4 are the method's design
    backdrops (fresh seeds here), the other three its holdouts.
  - **Parts B and C: the struggling set, by a rule fixed now, before T2's and T5's data:** every
    backdrop of T2 and T5 whose paper cell certified **10 or fewer of 20** (both verdicts, as the
    paper's), and if none did, the two lowest (ties to the larger machine). **Plus one no-harm
    backdrop**: the backdrop with the highest paper rate among S3, S4 and move-to-front-4 (ties to
    S3), to read harm on a machine where the cell works. *pending: the set, from T2's and T5's
    findings.*
  - Options for B and C: (a) every T2 and T5 backdrop. Pros: harm read everywhere. Cons: 9
    backdrops a cell (180 runs an arm), most where the cell already works and nothing can be gained.
    (b) *Chosen:* the struggling set and one no-harm backdrop. Pros: compute where a gain can show
    (the owner: "helps with cases where we are otherwise struggling"); harm still read once. Cons:
    harm on the other working backdrops unread until adoption's check (D10). (c) The holdouts only.
    Cons: division's and the larger machines' failures, if any, left out.
- **D5. Crystallisation's arms: `continuous` always; `continuous-reheat` (D63's reheat) only if the
  collapse's paired verdict, read offline at stage 1 by D63's registered rule, favours reheat.**
  - (a) `continuous` alone. Pros: half the compute; the method as D64 read it. Cons: D63's remedy
    for the freeze, every failure's end, never gets a sound test.
  - (b) Both always. Pros: both tested. Cons: about 100 more long runs (1,600 epochs) for an arm
    the existing data may already rule out.
  - (c) *Chosen:* the second arm conditional on the existing data's paired reading. Pros: the data
    already on disk decides whether the arm is worth its compute; the rule is fixed before that
    reading. Cons: D63's pairs are few (its 22 seeds); a weak reading may drop a remedy that would
    have helped (reported so).
  - Restarts (D63's other arm) are not an arm: they are a cost reading on the same runs
    (`restart-cost`, in epochs), reported for every cell.
- **D6. Crystallisation's claims (registered; refutations stated):**
  - **A1. Against the paper cell, per backdrop:** `continuous`'s rate against the paper cell's,
    Fisher two-sided (no direction: the method may help on the holdouts and lose elsewhere), Holm over
    the five. Refuted as a general improvement if it is significantly better on none.
  - **A2. Beyond its design backdrops:** works on a holdout if it certifies **at least 8 of 20**
    there (0.40, Wilson 0.22-0.61); exists there at 1 of 20. Refuted ("works beyond two backdrops",
    the paper's D2 condition for its own paper) if no holdout reaches 8; the crystallisation paper
    stays shelved then.
  - The same two for `continuous-reheat` if it runs (D5), and the two arms against each other,
    Fisher two-sided, reported as a third family.
- **D7. The recombining window: (a) the sum, then one shared block.**
  - (a) *Chosen.* The branches' ends summed, a full block over the sum, the output. Pros: the width
    stays `D`, so row 3 holds as at the base; the merge is a residual-like op (the crate's `add`
    already sums); the recombining block can mix the factors nonlinearly before the linear output
    (row 5); smallest build. Cons: a sum can let branches cancel or crowd each other in the same
    `D` numbers (but the recombining block's attention and FFN can separate them, and at 32 the
    stream already carries both consumers in the base).
  - (b) Concatenate the ends (`G D` wide), project back to `D` with a Dense, then the block. Pros:
    nothing mixes before a learned projection. Cons: a width-changing op and a projection in the
    crate's program; more parameters; the projection is linear, so it can only learn what the sum
    with a learned scale a branch already gives, apart from the scale.
  - (c) The output attends over the branches' ends (each token reading every branch's token). Pros:
    the most general. Cons: a new attention over a branch axis; the largest build; harder to read.
  - (d) The answers read the snapped next state (the codes) and the window's tokens, not the
    branches. Pros: `lambda` then depends on the state only through its codes, the machine's own
    variables. Cons: changes the output's timing and inputs (the answer reads a state the window
    has just made); a different question from the owner's (an output map, not a recombination);
    worth its own probe later.
- **D8. Branching's arms: `branches-32` and `recombine`, both at the base width 32, two heads a
  branch, every path 3 blocks deep; on the paper cell's other settings.**
  - `branches-32`: the retired design (`--branches`) at width 32 with two heads a branch: the test it
    never had (rows 3 and 4 met).
  - `recombine`: D7's window (trunk one block, a branch a factor, the merge, the recombining block).
  - Options: (a) one head a branch (the owner's original conjecture). Cons: row 4: at 16 a single head
    starved the transitions on S4; at 32 one head per consumer still has half the base's heads.
    (b) *Chosen:* two heads, the base's. Pros: no consumer below the base's attention; one change
    from the base (the separation). Cons: about `G` blocks more cost than the base. (c) A trunk of
    two blocks before `recombine`'s branches (4 deep). Cons: depth changes too (part C's question).
  - So `branches-32` and `recombine` each differ from the paper cell in the window's paths alone,
    at equal depth; they differ from each other in the trunk's length (2 against 1) and the merge: two
    changes, so their comparison is reported, not claimed (D9).
- **D9. Branching's claims (registered):**
  - **B1. `recombine` helps where the paper cell struggles:** on each struggling backdrop, Fisher
    one-sided (`recombine` greater; the owner's conjecture names the direction), two-sided reported
    beside it; Holm over the struggling set. Refuted on a backdrop if not significant there.
  - **B2. `branches-32` helps there:** the same test.
  - **Harm, both arms:** on the no-harm backdrop, Fisher two-sided; harm if significantly lower.
  - `recombine` against `branches-32`: rates and Fisher two-sided, reported (two changes, D8).
  - Each arm's failures read by kind (failure-kinds.md: formation, conversion, extra codes,
    readout), and the readout's separability check on each failure that has an exact state: does
    the recombination fix readout failures (row 5's) in particular.
- **D10. Adoption: an arm becomes the default if it meets its claim on at least one struggling
  backdrop (B1, B2 or A1 there) and shows harm nowhere, harm then read on every T2 and T5 backdrop
  not yet in the batch (a 6-seed screen a backdrop, two-sided Fisher against the paper cell's
  first 6 seeds of 300-305), any significant drop blocking it.**
  - Several adopted: the one with the larger total of certified runs over the struggling set; a
    tie to the cheaper in epochs a certificate. None: the paper cell stays the default.
  - Options: (a) adopt per backdrop (a cell a machine). Cons: against one architecture for every
    machine (memory `general-architecture-tasks-as-backdrops`). (b) *Chosen:* one default, checked
    on every backdrop. Pros: the programme's rule; part C has one base. Cons: an arm that helps
    one machine and harms another isn't adopted (reported so, its gain recorded).
- **D11. Depth's arm: 6 blocks against the default's 3, nothing else changed.**
  - (a) 4 blocks. Pros: cheapest. Cons: a small step may show no effect at 20 seeds where a larger
    one would; "no effect at 4" says little about depth.
  - (b) 2, 4 and 6 (a matrix). Cons: three arms where the question is whether depth helps at all
    (memory `short-focused-experiments`).
  - (c) *Chosen:* 6, twice the base. Pros: existence first (memory `existence-first-compute`): a
    clear step either shows depth can help or makes a smaller step unlikely to; a bound by halving
    (4, then 5) follows only if 6 helps, registered then. Cons: about twice the epoch time and the
    window's activation memory (measured at stage 3); more depth may train worse, though the
    residual, pre-LayerNorm blocks are the case He et al. 2016 and our depth study found stable.
  - Applied to the whole window: for `recombine` or `branches-32` as default, the extra blocks go in
    the trunk (the paths' separation unchanged); for the plain window, in the stack. *pending, by
    D10's result.*
- **D12. Depth's claim: C1, 6 blocks help where the default struggles:** on each backdrop of the
  struggling set (re-derived from the default's rates if the default changed: the same rule, D4, on
  the default's 20 seeds), Fisher one-sided (6 greater), two-sided beside it, Holm over the set; harm
  on the no-harm backdrop, two-sided. Refuted on a backdrop if not significant. The default's rates
  are its own control (D3's rule if it is the paper cell; part B's runs if it is an arm of B).
- **D13. Seeds and statistics: (a) 20 fresh seeds a cell** (A: 400-419, B: 500-519, C: 600-619);
  Wilson 95% intervals; Fisher as each claim says; Holm at 0.05 within each part's family; one-sided
  only where registered above, both sides reported where they disagree; certificate at both
  verdicts on the stop's network (D41); long tests on every certified network (the paper's D15
  rule), which confirm and don't establish. Options: (b) 6-seed screens first. Pros: cheaper. Cons:
  a screen claims nothing, and the owner asked for sound tests; 20 tells 0.8 from 0.4.
- **D14. Budgets and machines:** each cell's registered epochs: the paper cell 600, `continuous` and
  `continuous-reheat` 1,600 (D64's), the arms of B and C the same as their control's. No wall-clock
  budget (runaway guard only). Time and memory a run from stage 3's probes, then the batch's estimate
  as a range; the batch runner on every machine, memory planned per host before launch (memory
  `machine-load-limits`). Options: (b) a shorter budget for B and C. Cons: an arm judged on less
  training than its control is not one change.

## Feasibility (stage 1 fills the table; a test in `tests/` checks the cells against it)

Every cell, every backdrop: rows 1 (joint codes from `--codes-ratio 32/11` at least `q`), 2 (the
exclusion bound against the cell's longest length), 3 (width 32, the base's: met), 4 (`branches-32`
and `recombine`: every consumer's path two heads at the full width, so no consumer below the base;
the merge narrows nothing), 5 (the output layer reads a block's end at width 32), 14 (the check
sees every state), 15 (`continuous` only: depth plus Chow's `k` within the crucible's cap of 8; S4
and move-to-front-4 read in D16's table), 17 (each backdrop's congruences). *pending: the table,
one row a cell and backdrop, from `congruences`, `measures` and the configs.*

## Stages

0. **The gate (D2):** paper 1 released (the paper workplan's stage 8).
1. **Registration (one PR):** D63's paired verdict read offline by its rule (D5); T2's and T5's
   rates from the paper's findings fix the struggling set (D4); every *pending* item filled by its
   rule and nothing else changed; the feasibility table; the batch runner's cells (`continuous`,
   `continuous-reheat` if run, `branches-32`, `recombine`) and backdrops in `indrajala_ml/batch.py`,
   with a test that they meet their bounds; a note in the state-formation plan pointing its open
   test items here.
2. **The builds (one PR each):** `--window-blocks` (pure Python, the golden run unchanged); the
   crate's `merge` op (`indrajala-math-rust`, parity tests; tier 1 A/B of the window program's
   benchmark, `--order ONNO`), then "Bump rust/", then `Branches(..., merge=True)` and `--recombine`
   (numpy and crate paths, parity tests, the golden run unchanged); `--branch-heads` for `--branches`
   (default 1, the retired design's; 2 here).
3. **Probes (claim nothing):** `recombine` and `branches-32` on the struggling set's hardest backdrop,
   4 seeds, a short epoch budget: do they form a state at all; if neither ever forms, B's batch is
   reported as not run, with the probe. A depth-6 run of the paper cell, one seed, for its epoch time
   and memory. The batch's estimate from these.
4. **Parts A and B, one batch** on the runner (D1), and D3's reproduction runs; registered runs to
   their end (memory `no-deviations-from-early-peeks`).
5. **The reading:** A1, A2, B1, B2 and harm by their rules, into "Findings: A" and "Findings: B"
   here, with manifests; D10's screens if an arm qualifies; the default named.
6. **Part C:** the struggling set re-derived for the default if it changed (D12), its pending items
   filled, registered (one PR), the batch, the reading into "Findings: C".
7. **Findings** into [failure-kinds.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/research-notes/failure-kinds.md)
   and the feasibility register (rows 4 and 3 re-read, a row for depth); this plan retired, its
   leftovers (depth's bound, D7's options (b)-(d), the constants' sensitivity, F10) to
   [next-steps.md](../next-steps.md).

## The size, roughly (stage 3 replaces it)

- **A:** 5 backdrops x 20 x 1 or 2 arms = 100-200 runs of up to 1,600 epochs (D64's spent runs about
  1,600 epochs, 10 h at 0.38 minutes an epoch on `stone`): the largest part, days on every machine.
- **B:** 2 arms x (the struggling set + 1) x 20, of 600 epochs (S4 3.2 h median in T5), plus 10
  reproduction runs: for a set of 3, 160 runs.
- **C:** (the struggling set + 1) x 20 of 600 epochs at about twice the epoch time.

## Machines

The batch runner on `jebel`, `stone` (when the owner opens it), `pyramidon` and `klip`, memory per
host from stage 3; `litshe` parity only.
