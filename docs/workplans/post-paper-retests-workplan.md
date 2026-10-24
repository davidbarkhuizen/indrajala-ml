# Workplan: after paper 1, branching and depth, then a verdict on crystallisation

**Status: drafted 2026-10-24 at the owner's direction: "we will pick this up again once: 1. we are
done with paper 1; 2. we have conducted rigourous (statistically sound) tests of all the other work
we have covered since then - which was mainly annealing / crystallization, and maybe also branching
might be worth re-testing properly; at that point, we should [test] if increasing network depth
helps with cases where we are otherwise struggling", isolated ("agreed on properly isolating it");
on branching: "is it possible to have an architecture where we branch as we are (on
factors/code-books) - and then recombine again before output?", "let's incorporate it into the
branching study"; on crystallisation: "it's probably worth making a proper verdict about continous
quasi-static crystallization - because it still seems to me to be the right general approach - and
specifically with helping us to not lose solutions that we otherwise do lose at the
snap-to-finite-state - so its something we would then want to use by default - athough by itself it
won't help us to solve everything - i think that's where we need the depth and split-then-merge
arch"; and, after the critique of that view, the order: "we should rather look at depth and
branching first, then return to crystallization at that point". D1-D15 settled under the owner's
delegation, each with its options' pros and cons. Nothing here is built or run before paper 1 is out
(D2). Items marked *pending* are filled at each part's registration by the rules stated here, before
that part's data.**

The goal is [machine-discovery.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/research-notes/machine-discovery.md)'s
stage A: one general architecture and training method that forms a state, keeps it and certifies on
every backdrop. Parts: **B, branching** (claims B1, B2), then **L, depth** (claim L1), then **X,
crystallisation** (claims X1-X3), each on the default the part before it leaves.

## The IDs, in time order

| ID | When | Short label | What it is |
|---|---|---|---|
| D1 | 2026-10-24 | three parts in sequence | branching (B), then depth (L) on B's default, then crystallisation (X) on L's |
| D2 | 2026-10-24 | the gate | nothing built or run before paper 1's release |
| D3 | 2026-10-24 | the controls | each part's default's own runs; the paper cell's T2 and T5 runs reused, with a reproduction check |
| D4 | 2026-10-24 | the backdrops | B and L on the struggling set, by a rule fixed now, and a no-harm backdrop; X on D64's five and the set |
| D5 | 2026-10-24 | crystallisation's arms | `continuous`; `continuous-reheat` only if the collapse's (D63's) paired verdict favours it |
| D6 | 2026-10-24 | crystallisation's claims | X1 keeps formed solutions, X2 no harm, X3 the holdouts |
| D7 | 2026-10-24 | the recombining window | branch a factor, sum the branches' ends, one shared block, the output |
| D8 | 2026-10-24 | branching's arms | `branches-32` (the retired design at the base width, two heads a branch) and `recombine`, both 3 blocks deep |
| D9 | 2026-10-24 | branching's claims | B1, B2 against the default; B1 against B2 reported apart |
| D10 | 2026-10-24 | adoption | B, L: helps where struggling, harms nowhere; X: keeps solutions, harms nowhere |
| D11 | 2026-10-24 | depth's arm | 6 blocks against 3, the default's other settings unchanged |
| D12 | 2026-10-24 | depth's claim | L1 against the default |
| D13 | 2026-10-24 | seeds and statistics | 20 fresh seeds a cell; registered one-sided "helps", both sides reported; Holm a part |
| D14 | 2026-10-24 | budgets and machines | each cell's registered epochs; time and memory from a probe; the batch runner |
| D15 | 2026-10-24 | formed and kept | formed at state share 0.95; kept if certified; the stratified exact test |
| stages 0-6 | 2026-10-24 | the plan's stages | the gate, builds, B, L, X, findings |

From earlier plans: the collapse (D63), the fresh-seed rate (D64), the deadline's completion (D67)
and the method's constants (F10) in [state-formation-workplan.md](state-formation-workplan.md); the
window branches' W1, W2, B1-B3 (`git show e7f1ff6:docs/window-branches-workplan.md`; their B1-B3 are
written "the window branches' B1" here, to keep them apart from this plan's B1, B2); the paper's T2
(D15) and T5 (D16) in [paper-workplan.md](paper-workplan.md); the feasibility register's rows in
[information-flow.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/protocols/information-flow.md).

## Why

**Branching (part B).** The retired window branches (`--branches`: W11's first two blocks as a
trunk, then a single-head block a factor and one for the answers) ran at width 16, which division's
ladder had shown too narrow (L3: 0 of 6 at 16, 5 of 6 at 32), and with one head a branch: the window
branches' B1 and B2 1 of 6 a cell, harm on S4 (their B3, 0 of 6 against 4 of 6, p 0.030 one-sided).
The feasibility register names both as design faults (rows 3 and 4), so the branches were never
tested where they could succeed. In that design nothing recombines: each factor's pooling reads its
branch's end, and the answers read their own branch, which reads only the trunk.

**The recombining window (the owner's question).** A Mealy machine's output `lambda(q, a)` depends
on the whole state; when the state is factored, `q = (q_1, ..., q_G)`, the transition can be computed
a factor at a time (a cover: each factor's next state from its own and the others'), but the answer
generally needs every factor at once. Branching per factor gives each factor's transition its own
path; recombining before the output gives the answers a stream that sees every factor's path, rather
than the trunk alone. It keeps the state factored (one codebook a factor; memory `factored-not-flat`)
and adds a merge only on the output's side.

**Depth (part L).** The window has been three blocks deep (W11's `BLOCKS` 3) through every study
since W11; depth was varied only in the primitives' early studies (plain networks got worse with
depth, residual ones didn't; two transformer layers beat one). Where the default still fails, depth
is the untested axis of the window's capacity beside width (row 3), which W1 showed matters.

**Why B and L first (the owner's order, D1).** Most failures never form a state (D64: 10 of its 15),
and no method of keeping a solution helps those; depth and the split-then-merge window are the
candidates for forming one. They run on the paper cell (600 epochs a run, its 20-seed rates already
on disk), so they are cheap and soon. And crystallisation's own question is easier to answer on a
stronger architecture: the more runs form a state, the more formed runs X1 has to read.

**Crystallisation (part X).** The state-formation strand's method, the continuous crystallisation
(D54's cell, the margin stop under D60: the batch runner's `continuous` cell), is out of paper 1
(the paper's D2) because it rests on two backdrops: D64 certified S3 and move-to-front-4 7 of 12
(between its bars) and S4, random-q12-0 and random-q16-0 0 of 10 (with D56-D58, S4 1 of 6,
random-q12-0 1 of 8, random-q16-0 0 of 7). Every failure ended frozen. The owner's case for it is a
mechanism, not a rate: a network can learn the machine in its continuous state and lose it when the
state is snapped to finitely many codes, and the quasi-static cool's gradual handover from the
continuous state to the codes (D25, D29, D50) is meant to keep what formed. So its verdict is
whether it keeps formed solutions (X1) without harm (X2), and its adoption doesn't ask it to solve
the cases where nothing forms.

**The critique, recorded (2026-10-24).** For the view: the principle is deterministic annealing's
(Rose), the established remedy for the poor minima of an abrupt quantisation; the loss is real in
our data (S4's joint separation fell from 0.98 to 0.25 at an abrupt snap). Against, or unshown: (1)
the paper cell trains with codes from its start, so it has no continuous phase to lose a solution
from; the snap losses seen were inside crystallisation's own anneal, so how much the default loses
at a snap is unmeasured (X1's control reading, D15, measures it); (2) crystallisation still loses
formed solutions (D64: 5 of its 15 failures formed and lost a state; every failure froze): it
reduces the loss at best; (3) it costs more (1,600 epochs a run against 600; not faster in epochs
to a certificate), a cost every later study would pay as the default; (4) its evidence is its two
design backdrops. None refutes the view; X tests it.

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
separation and not depth, and depth stays part L's alone. The answers' path passes through every
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

- **D1. Three parts in sequence: branching (B), then depth (L) on B's default, then crystallisation
  (X) on L's: (c).**
  - (a) Crystallisation first, then B and L on its default (the first draft). Pros: B and L tested
    on the method the owner expects to use. Cons: B and L wait for X's long runs and then run at
    1,600 epochs (about 2.5 times the compute a run); the larger failure class (never formed) waits.
  - (b) X and B as one batch against the paper cell, then L. Pros: the machines filled sooner.
    Cons: X's keeping reading on the weaker architecture, with fewer formed runs to read.
  - (c) *Chosen (the owner).* B, L, X. Pros: B and L at 600 epochs against controls on disk, soon;
    they target never-formed runs; X then reads keeping on the strongest architecture, with more
    formed runs (more power for X1). Cons: B's and L's verdicts are on the paper cell's training; if
    X's method is adopted after them, their gains are checked again only by X2's harm reading and
    D10's screens, not re-registered.
  - The state-formation plan's remaining test items (the panel, 4b; D63's paired reading) move to
    part X; that plan gets a note pointing here (X's registration).
- **D2. The gate: (a) nothing built or run until paper 1's release (its stage 8).**
  - (a) *Chosen.* Pros: the owner's order; the paper's machines and attention undivided; the paper's
    T2 and T5 read first, which the backdrop rule needs (D4). Cons: the builds (D7, D11) could be
    done offline earlier.
  - (b) The builds now, the runs after. Pros: ready sooner. Cons: against the owner's "don't worry
    about this now"; a crate change in the middle of the paper's batches is a risk to them.
  - (c) After the paper's runs, before its manuscript. Pros: the machines are idle while it is
    written. Cons: not the owner's order ("done with paper 1"); the manuscript may need reruns.
- **D3. The controls: each part's default's own 20 seeds a backdrop; the paper cell's (the default
  until a part adopts an arm) are T2's and T5's runs, reused, with two seeds a backdrop re-run to
  check they reproduce: (b).**
  - (a) A fresh control in each batch. Pros: concurrent, the same commit. Cons: 20 runs a backdrop
    repeating ones already on disk.
  - (b) *Chosen.* Pros: runs are deterministic and bit-identical across machines (experiment-design.md,
    section 5), so a rerun of two seeds a backdrop on the part's commit with `same-prefix` shows the
    control is the same cell; the golden run checks training is unchanged; an adopted arm's batch
    runs are the next part's control. Cons: a control from an earlier commit; any difference in a
    re-run, and that backdrop's control is run fresh ((a)).
  - (c) No control: the arms' rates alone. Cons: no comparison; nothing could be adopted.
- **D4. The backdrops.**
  - **B and L: the struggling set, by a rule fixed now, before T2's and T5's data:** every backdrop
    of T2 and T5 whose default certified **10 or fewer of 20** (both verdicts, as the paper's), and if
    none did, the two lowest (ties to the larger machine). **Plus one no-harm backdrop**: the highest
    default rate among S3, S4 and move-to-front-4 (ties to S3), to read harm where the cell works.
    L re-derives the set from B's default's rates if B adopted an arm. *pending: the set, from T2's
    and T5's findings.*
  - **X: D64's five** (word-problem-s3, move-to-front-4, word-problem-s4, random-q12-0,
    random-q16-0; S3 and move-to-front-4 its design backdrops, fresh seeds here, the others its
    holdouts) **and any backdrop of L's struggling set not among them** (division's, if there).
  - Options for B and L: (a) every T2 and T5 backdrop. Cons: 9 backdrops a cell (180 runs an arm),
    most where the cell already works. (b) *Chosen:* the struggling set and one no-harm backdrop.
    Pros: compute where a gain can show (the owner: "helps with cases where we are otherwise
    struggling"); harm still read once. Cons: harm on the other working backdrops unread until
    adoption's screens (D10). (c) The holdouts only. Cons: division's and the larger machines'
    failures, if any, left out.
- **D5. Crystallisation's arms: `continuous` always; `continuous-reheat` (D63's reheat) only if the
  collapse's paired verdict, read offline at X's registration by D63's registered rule, favours
  reheat: (c).**
  - (a) `continuous` alone. Pros: half the compute. Cons: D63's remedy for the freeze, every
    failure's end, never gets a sound test.
  - (b) Both always. Cons: about 100 more long runs for an arm the existing data may rule out.
  - (c) *Chosen.* Pros: the data on disk decides whether the arm earns its compute; the rule is
    fixed before that reading. Cons: D63's pairs are few; a weak reading may drop a remedy that
    would have helped (reported so).
  - Restarts (D63's other arm) are not an arm: a cost reading on the same runs (`restart-cost`, in
    epochs), reported for every cell.
  - Both arms carry L's default's window (its depth and paths); the crystallisation is the one
    change.
- **D6. Crystallisation's claims (registered; refutations stated):**
  - **X1. It keeps solutions the snap otherwise loses (the primary claim):** among the runs that
    formed a solution (D15), the share kept, `continuous` greater than the default, one-sided (the
    owner's claim names the direction), by the stratified exact test over X's backdrops (D15);
    two-sided beside it. Per backdrop, Fisher one-sided, reported. Refuted if the stratified test is
    not significant at 0.05. X1 also reads the losses: for each run formed and not kept, where it
    was lost (at the handover, by the coupling's rise; later, by a collapse; or never converted), in
    the failure kinds' terms; and the default's own losses, which measure the critique's point (1).
  - **X2. No harm:** the certified rate against the default, Fisher two-sided per backdrop, Holm over
    X's backdrops; harm if any is significantly lower. Twenty seeds can't show two rates equal, so
    each difference is reported with its interval (Newcombe's, from the Wilson intervals): no harm is
    "no significant drop", not equivalence.
  - **X3. Beyond its design backdrops** (descriptive, not a gate for adoption): works on a holdout at
    **at least 8 of 20** (0.40, Wilson 0.22-0.61), exists there at 1 of 20. Refuted ("works beyond
    two backdrops", the paper's D2 condition for a crystallisation paper) if no holdout reaches 8.
  - The same three for `continuous-reheat` if it runs (D5), and the two arms against each other,
    Fisher two-sided, a further family.
  - Options for the primary claim: (a) the certified rate against the default. Pros: the plainest
    outcome. Cons: mixes forming and keeping; where nothing forms, a method that keeps every formed
    solution shows no gain. (b) *Chosen:* kept among formed. Pros: the owner's mechanism exactly;
    the never-formed cases are B's and L's. Cons: conditions on a reading the arms may affect
    differently (a method that forms less but keeps what forms looks good), so X2's unconditioned
    rate is read beside it and harm blocks adoption (D10). (c) The gradual handover against an
    abrupt snap within crystallisation. Pros: isolates the handover. Cons: tests a component, not
    the method against the default it would replace; a further build.
- **D7. The recombining window: (a) the sum, then one shared block.**
  - (a) *Chosen.* The branches' ends summed, a full block over the sum, the output. Pros: the width
    stays `D`, so row 3 holds as at the base; the merge is a residual-like op (the crate's `add`
    already sums); the recombining block can mix the factors nonlinearly before the linear output
    (row 5); smallest build. Cons: branches can crowd each other in the same `D` numbers (the
    recombining block's attention and FFN can separate them; at 32 the stream already carries both
    consumers in the base).
  - (b) Concatenate the ends (`G D` wide), project back to `D`, then the block. Pros: nothing mixes
    before a learned projection. Cons: a width-changing op and a projection in the crate's program;
    a linear projection adds little over the sum but a scale a branch.
  - (c) The output attends over the branches' ends. Pros: the most general. Cons: a new attention
    over a branch axis; the largest build; harder to read.
  - (d) The answers read the snapped next state (the codes) and the window's tokens. Pros: `lambda`
    then depends on the state only through its codes. Cons: changes the output's timing and inputs;
    an output map, not a recombination; worth its own probe later.
- **D8. Branching's arms: `branches-32` and `recombine`, at the base width 32, two heads a branch,
  every path 3 blocks deep, on the paper cell's other settings: (b).**
  - `branches-32`: the retired design (`--branches`) at width 32 with two heads a branch: the test it
    never had (rows 3 and 4 met). `recombine`: D7's window.
  - (a) One head a branch (the owner's original conjecture). Cons: row 4: at 16 a single head
    starved the transitions on S4; at 32 one head per consumer still has half the base's heads.
    (b) *Chosen:* two heads, the base's. Pros: no consumer below the base's attention; one change
    from the base (the separation). Cons: about `G` blocks more cost than the base. (c) A trunk of
    two blocks before `recombine`'s branches (4 deep). Cons: depth changes too (part L's question).
  - Each arm differs from the paper cell in the window's paths alone, at equal depth; they differ
    from each other in the trunk's length (2 against 1) and the merge: two changes, so their
    comparison is reported, not claimed (D9).
- **D9. Branching's claims (registered):**
  - **B1. `recombine` helps where the default struggles:** on each struggling backdrop, Fisher
    one-sided (`recombine` greater; the owner's conjecture names the direction), two-sided beside it;
    Holm over the set. Refuted on a backdrop if not significant there.
  - **B2. `branches-32` helps there:** the same test.
  - **Harm, both arms:** on the no-harm backdrop, Fisher two-sided; harm if significantly lower.
  - `recombine` against `branches-32`: rates and Fisher two-sided, reported (two changes, D8).
  - Each arm's failures read by kind (failure-kinds.md: formation, conversion, extra codes,
    readout), and the separability check on each failure with an exact state: does the
    recombination fix readout failures (row 5's) in particular.
- **D10. Adoption: an arm of B or L becomes the default if it meets its claim on at least one
  struggling backdrop and shows harm nowhere; X's `continuous` (or `continuous-reheat`) if X1 holds
  and X2 shows no harm. Either way harm is then read on every T2 and T5 backdrop not yet in its
  batch (a 6-seed screen a backdrop, two-sided Fisher against the control's first 6 seeds), any
  significant drop blocking it: (b).**
  - Several arms of B adopted: the one with more certified runs over the struggling set; a tie to
    the cheaper in epochs a certificate. None: the default stays.
  - (a) Adopt per backdrop (a cell a machine). Cons: against one architecture for every machine
    (memory `general-architecture-tasks-as-backdrops`). (b) *Chosen:* one default, checked on every
    backdrop. Pros: the programme's rule; each part has one base. Cons: an arm that helps one machine
    and harms another isn't adopted (reported so, its gain recorded).
  - Crystallisation is adopted for keeping solutions, not for solving the struggling cases (the
    owner: "by itself it won't help us to solve everything").
- **D11. Depth's arm: 6 blocks against the default's 3, nothing else changed: (c).**
  - (a) 4 blocks. Pros: cheapest. Cons: a small step may show nothing at 20 seeds where a larger
    one would.
  - (b) 2, 4 and 6 (a matrix). Cons: three arms where the question is whether depth helps at all
    (memory `short-focused-experiments`).
  - (c) *Chosen:* 6, twice the base. Pros: existence first (memory `existence-first-compute`); a
    bound by halving (4, then 5) follows only if 6 helps, registered then. Cons: about twice the
    epoch time and the window's activation memory (measured by L's probe); more depth may train
    worse, though residual, pre-LayerNorm blocks are the case He et al. 2016 and our depth study
    found stable.
  - For `recombine` or `branches-32` as default, the extra blocks go in the trunk (the paths'
    separation unchanged); for the plain window, in the stack. *pending, by B's result.*
- **D12. Depth's claim: L1, 6 blocks help where the default struggles:** on each backdrop of L's
  struggling set (D4), Fisher one-sided (6 greater), two-sided beside it, Holm over the set; harm on
  the no-harm backdrop, two-sided. Refuted on a backdrop if not significant.
- **D13. Seeds and statistics: (a) 20 fresh seeds a cell** (B: 400-419, L: 500-519, X: 600-619);
  Wilson 95% intervals; Fisher as each claim says; Holm at 0.05 within each part's family; one-sided
  only where registered above, both sides reported where they disagree; certificate at both
  verdicts on the stop's network (D41); long tests on every certified network (the paper's D15
  rule), which confirm and don't establish. (b) 6-seed screens first. Pros: cheaper. Cons: a screen
  claims nothing, and the owner asked for sound tests; 20 tells 0.8 from 0.4.
- **D14. Budgets and machines:** each cell's registered epochs: the paper cell 600 (B's and L's arms
  the same), `continuous` and `continuous-reheat` 1,600 (D64's). No wall-clock budget (runaway guard
  only). Time and memory a run from each part's probe, then the batch's estimate as a range; the
  batch runner on every machine, memory planned per host before launch (memory
  `machine-load-limits`). (b) A shorter budget for an arm. Cons: an arm judged on less training than
  its control is not one change.
- **D15. Formed and kept:**
  - **Formed:** at some step the run's state separation share (the state-separation check: the
    share of the target's states with a code of their own, as D64's findings read "formed") reached
    **0.95** (D64's "formed then lost" at 0.99; 4a's at 0.95-0.97). For the default (codes from the
    start), read on its logged steps if its logs carry the share, else offline on its step networks;
    *pending, at X's registration: which; if neither exists for the control's runs, X's control is
    run fresh with the reading logged (D3's (a)).*
  - **Kept:** certified at both verdicts on the stop's network (D41).
  - **The test:** a stratified exact test of kept among formed, conditioning on each backdrop's
    formed counts (the exact Cochran-Mantel-Haenszel test, its statistic the total kept by
    `continuous`, counted exactly in integers as `rank_sum_greater` is), built in
    `indrajala_ml/studies/inference.py` with a brute-enumeration check. (a) Pooled Fisher. Cons:
    biased when the backdrops' rates differ, as they will. (b) *Chosen:* the stratified exact test.
    Pros: right for strata of different rates; exact. Cons: a small build. (c) Per backdrop only.
    Cons: at a handful of formed runs a backdrop, no single test has power; the owner asked for a
    verdict on the method, across backdrops.

## Feasibility (each part's registration fills its rows; a test in `tests/` checks its cells)

Every cell, every backdrop: rows 1 (joint codes from `--codes-ratio 32/11` at least `q`), 2 (the
exclusion bound against the cell's longest length), 3 (width 32, the base's: met), 4 (`branches-32`
and `recombine`: every consumer's path two heads at the full width, so no consumer below the base;
the merge narrows nothing), 5 (the output layer reads a block's end at width 32), 14 (the check sees
every state), 15 (`continuous` only: depth plus Chow's `k` within the crucible's cap of 8), 17 (each
backdrop's congruences). *pending: one row a cell and backdrop, from `congruences`, `measures` and
the configs.*

## Stages

0. **The gate (D2):** paper 1 released (the paper workplan's stage 8).
1. **B's builds (one PR each):** the crate's `merge` op (`indrajala-math-rust`, parity tests; tier 1
   A/B of the window program's benchmark, `--order ONNO`), then "Bump rust/", then
   `Branches(..., merge=True)` and `--recombine` (numpy and crate paths, parity tests, the golden run
   unchanged); `--branch-heads` for `--branches` (default 1, the retired design's; 2 here).
2. **Part B:** the struggling set from T2's and T5's findings (D4); the cells, feasibility rows and
   claims registered (one PR); probes first (claim nothing): both arms on the set's hardest backdrop,
   4 seeds, a short epoch budget (if neither ever forms a state, B is reported as not run, with the
   probe); D3's reproduction runs; the batch on the runner, run to its end (memory
   `no-deviations-from-early-peeks`); the reading (B1, B2, harm, D10's screens) into "Findings: B".
3. **Part L:** `--window-blocks` (pure Python, the golden run unchanged); a depth-6 probe of the
   default, one seed, for its epoch time and memory; the set re-derived if B adopted an arm (D4);
   registered (one PR); the batch; the reading (L1, harm, D10) into "Findings: L".
4. **Part X:** D63's paired verdict read offline (D5); D15's formation reading located for the
   control; D15's stratified exact test (`inference.py`); the cells on L's default and their
   feasibility rows registered (one PR); a note in the state-formation plan pointing its open test
   items here; the batch; the reading (X1-X3, D10) into "Findings: X".
5. **Findings** into [failure-kinds.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/research-notes/failure-kinds.md)
   and the feasibility register (rows 3 and 4 re-read, a row for depth).
6. **Retire** this plan; its leftovers (depth's bound, D7's options (b)-(d), the constants'
   sensitivity, F10) to [next-steps.md](../next-steps.md).

## The size, roughly (each part's probe replaces it)

- **B:** 2 arms x (the struggling set + 1) x 20, of 600 epochs (S4 3.2 h median in T5), plus the
  reproduction runs: for a set of 3, 160 runs; a day or two on every machine.
- **L:** (the set + 1) x 20 of 600 epochs at about twice the epoch time.
- **X:** (5 or more backdrops) x 20 x 1 or 2 arms of 1,600 epochs (D64's spent runs about 10 h on
  `stone`): the largest part, days on every machine.

## Machines

The batch runner on `jebel`, `stone` (when the owner opens it), `pyramidon` and `klip`, memory per
host from each part's probe; `litshe` parity only.
