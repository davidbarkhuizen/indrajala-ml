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
branching first, then return to crystallization at that point"; and, on T2's and T5's early reading
(2026-10-24, the paper cell certifying S3 1 of 20 and move-to-front-4 4 of 20, against D64's
crystallisation at 5 of 8 and 2 of 4): "that is what i was saying earlier about starting with
crystallization and making it the default - we definitely found it to improve success": the order
back to crystallisation first (D1 (a)), its control at its own epoch budget (D3). D1-D15 (and D19-D21) settled under the owner's
delegation, each with its options' pros and cons. Nothing here is built or run before paper 1 is out
(D2). Items marked *pending* are filled at each part's registration by the rules stated here, before
that part's data.**

**Revised 2026-10-25 (the owner): X is an A/B of the two methods on paper 1's task matrix (D16-D21),
replacing its holdouts-first design.** The owner: "it qualitatively seems like we regressed in terms
of speed and number of solutions found when we returned to finalizing P1, i.e. dropped
crystallization ... I still want to properly assess the effect of the crystallization method"; "the
best way to do this would be to have a matrix of tasks, with our best setup ... then run the matrix
of tasks for a few different seeds, across both methods", for "a proper view of the effect of the
crystallization method across tasks (and thus machine types)" and "very good analysis data for
determining the failure modes of the crystallization method ... and if/when it is harmful". X's parts
of D3, D4, D6, D10, D13, D14 and the whole of D15 are superseded (marked there); D1's order, D2's gate
and D5's arm rule stand.

The goal is [machine-discovery.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/research-notes/machine-discovery.md)'s
stage A: one general architecture and training method that forms a state, keeps it and certifies on
every backdrop. Parts: **X, crystallisation** (claims X1-X3), then **B, branching** (claims B1,
B2), then **L, depth** (claim L1), each on the default the part before it leaves.

## The IDs, in time order

| ID | When | Short label | What it is |
|---|---|---|---|
| D1 | 2026-10-24, revised twice | three parts in sequence | crystallisation (X), then branching (B) on its default, then depth (L) on B's |
| D2 | 2026-10-24 | the gate | nothing built or run before paper 1's release |
| D3 | 2026-10-24, revised | the controls | X's: the paper cell at 1,600 epochs, fresh; B's and L's: the default's own runs |
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
| D16 | 2026-10-25 | X as an A/B | the paper's method against crystallisation, aligned, on paper 1's task matrix (the owner) |
| D17 | 2026-10-25 | the aligned settings | no fork, width 32, 1,600 epochs, the answers-only stop with the margin wait, both arms; the commitment each its own (the owner) |
| D18 | 2026-10-25 | approximations in both arms | each task its target's approximations' outputs, the groups' built for X (the owner) |
| D19 | 2026-10-25 | X's tasks | paper 1's nine backdrops |
| D20 | 2026-10-25 | X's seeds | 10 a cell, the same seeds in both arms, read in pairs |
| D21 | 2026-10-25 | X's readings | the paired effect and harm per task, epochs to a certificate, every run's failure kind and the stop's losses |
| stages 0-6 | 2026-10-24, revised | the plan's stages | the gate, X, B's builds, B, L, findings |

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

**Why crystallisation first (the owner's order, D1).** T2's and T5's early reading (certificates,
before the long tests) put the paper cell at S3 1 of 20 and move-to-front-4 4 of 20, where D64's
crystallisation certified 5 of 8 and 2 of 4 (post hoc: other seeds, another batch). If it is the
default, B and L tested on the paper cell would test a base about to be replaced. Two cautions,
which X's design answers: the comparison is confounded by the budget (crystallisation trains 1,600
epochs, the paper cell 600), so X's control is the paper cell at 1,600 (D3); and on the holdouts the
two are alike and low (crystallisation S4 1 of 6, random-q12-0 1 of 8 in D56-D58; the paper cell
S4 4 of 20, random-q12-0 3 of 20), where B and L are the candidates (most failures never form a
state: D64, 10 of its 15).

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

- **D1. Three parts in sequence: crystallisation (X), then branching (B) on its default, then depth
  (L) on B's: (a), chosen again (the owner, on T2's and T5's early reading) after (c).**
  - (a) *Chosen.* Crystallisation first, then B and L on its default. Pros: B and L tested on the
    method the owner expects to use; the early reading's gap (S3 1 of 20 against 5 of 8) says the
    default is likely to change. Cons: B and L wait for X's long runs and, if X is adopted, run at
    1,600 epochs (about 2.5 times the compute a run); the larger failure class (never formed) waits.
  - (b) X and B as one batch against the paper cell, then L. Pros: the machines filled sooner.
    Cons: X's keeping reading on the weaker architecture, with fewer formed runs to read.
  - (c) *Chosen first, replaced.* B, L, X. Pros: B and L at 600 epochs against controls on disk, soon;
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
- **D3. The controls** (*X's part superseded 2026-10-25 by D16: both arms run fresh*)**: X's, the paper cell at crystallisation's budget (1,600 epochs), fresh, 20
  seeds a backdrop, T5's 600-epoch runs reported beside it (so a gain is the method's, not the
  budget's: one change); B's and L's, the default's own 20 seeds a backdrop (X's runs if
  crystallisation is adopted); where the paper cell at 600 epochs is the control, T2's and T5's runs
  reused, with two seeds a backdrop re-run to check they reproduce: (b).**
  - (a) A fresh control in each batch. Pros: concurrent, the same commit. Cons: 20 runs a backdrop
    repeating ones already on disk.
  - (b) *Chosen.* Pros: runs are deterministic and bit-identical across machines (experiment-design.md,
    section 5), so a rerun of two seeds a backdrop on the part's commit with `same-prefix` shows the
    control is the same cell; the golden run checks training is unchanged; an adopted arm's batch
    runs are the next part's control. Cons: a control from an earlier commit; any difference in a
    re-run, and that backdrop's control is run fresh ((a)).
  - (c) No control: the arms' rates alone. Cons: no comparison; nothing could be adopted.
- **D4. The backdrops** (*X's part superseded 2026-10-25 by D19*)**.**
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
- **D6. Crystallisation's claims** (*superseded 2026-10-25 by D21; kept for the record*)**:**
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
- **D10. Adoption** (*X's part amended 2026-10-25: D21's adoption rule*)**: an arm of B or L becomes the default if it meets its claim on at least one
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
- **D13. Seeds and statistics: (a) 20 fresh seeds a cell** (B: 400-419, L: 500-519; X's 10 a cell by
  D20);
  Wilson 95% intervals; Fisher as each claim says; Holm at 0.05 within each part's family; one-sided
  only where registered above, both sides reported where they disagree; certificate at both
  verdicts on the stop's network (D41); long tests on every certified network (the paper's D15
  rule), which confirm and don't establish. (b) 6-seed screens first. Pros: cheaper. Cons: a screen
  claims nothing, and the owner asked for sound tests; 20 tells 0.8 from 0.4.
- **D14. Budgets and machines:** each cell's registered epochs: the paper cell 600 (B's and L's arms
  the same), `continuous` and `continuous-reheat` 1,600 (D64's); X's arms both 1,600 (D17). No wall-clock budget (runaway guard
  only). Time and memory a run from each part's probe, then the batch's estimate as a range; the
  batch runner on every machine, memory planned per host before launch (memory
  `machine-load-limits`). (b) A shorter budget for an arm. Cons: an arm judged on less training than
  its control is not one change.
- **D15. Formed and kept** (*superseded 2026-10-25 by D21: formation is read on every run, as a failure
  kind, not as a claim's condition*)**:**
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

- **D16. X as an A/B: the paper's method against crystallisation, aligned (D17), on paper 1's task
  matrix (D19), both arms with each task's approximations (D18) and run fresh (the owner,
  2026-10-25): (a).**
  - (a) *Chosen.* Pros: the effect across machine types, not on two design backdrops and their
    holdouts; every run's failure is analysis data for improving the method and for seeing where it
    harms; one comparison of the two methods as we would use them. Cons: the arms differ in more
    than one setting (the cool, the crucible, the commitment, D17), so a difference is the method's,
    not a component's; both arms fresh (about 180 runs, below).
  - (b) The plan's design (D3, D4, D6): D64's five, kept among formed, a fresh control at 1,600.
    Cons: its primary claim conditions on formation, hard to define for codes from the start; the
    holdouts are where the method is expected to fail, read first.
  - (c) T5's runs reused as the control at 600 epochs, only the new arm run. Pros: half the runs.
    Cons: T5's cell has the fork and stops at the first certificate, which D17 aligns away; its
    groups had no approximations (D18); the budgets differ.
- **D17. The aligned settings (the owner, 2026-10-25):** both arms without the fork, factors 32 wide,
  1,600 epochs, and the answers-only stop under the exact state bound with the margin wait and its
  deadline (D35, D59, D60 in the state-formation plan); the commitment each arm's own (the paper
  arm's as built; `continuous`'s `--anneal-never-commit`).
  - **The fork:** off in both. The owner: it doesn't fit the crystallisation analogy; whether the
    paper cell is better without it is its own test, after X. (a) Each its own. Cons: a difference
    of the method's that isn't the crystallisation.
  - **The width:** 32 in both (the paper's; `--factor-width`'s default is 16, and `continuous` was
    developed at it). Division's ladder: 0 of 6 at 16, 5 of 6 at 32 (L3). Cons: crystallisation
    untested at 32; memory a run measured by X's probe.
  - **The budget:** 1,600 in both, the longer (D64's). The paper arm stops at its stop, so the
    longer budget costs only its failures.
  - **The stop:** the answers-only stop with the margin wait. Why the two diverged: the paper's
    (D3) stops at the first certified step, W1's, with the target-free proof read offline (T6);
    crystallisation moved the target out of the loop for discovery (D35), then waited out near ties
    (D59) because its soft, slowly sharpening state passed the answers on networks the default
    verdict failed (12 of D54's and D55's runs), the wait bounded at 200 steps (D60). On the paper's
    runs the answers-only stop landed on the certified step at the exact bound 80 of 80 times (T6c);
    D59's certified networks had no split tie. The owner: the answers-only stop for "consistency and
    alignment with the long-term goal", with the solutions it loses read afterwards (D21); with the
    margin wait. (a) *Chosen.* Pros: target-free, one rule; likely the paper's stop on its own data.
    Cons: the wait can train past an exact machine and lose it (P1's 3b), read as a loss. (b) The
    plain answers-only stop. Cons: crystallisation's near-tie stops, a known handicap to one arm.
    (c) The first certificate. Cons: the target in the loop. (d) Each its own. Cons: the stop
    confounded with the method.
  - **The commitment:** each its own (the owner: "the approaches fundamentally diverge here"): the
    paper cell needs it (D37: S4, move-to-front-4 seed 0 and division `p` 13 certified only with
    it); crystallisation's anneal was designed without it.
  - *pending, at registration: both cells' full arguments, each difference from the paper's and
    `continuous`'s registered cells listed.*
- **D18. Approximations in both arms: each task trains with its target's approximations' outputs
  (the owner, 2026-10-25: "groups get approximations in the A/B").**
  - Division's families carry theirs as in paper 1; the groups (S3, S4) and move-to-front-4 get
    theirs, a build (P2's output, which T10 built for its planted machines, extended to a target's
    congruence quotients); a task whose target has no proper congruence trains on its answers alone,
    so stated.
  - Why the groups: their certified networks learned approximations unprompted (K8, T4), but those
    are 1-4 of 20 a backdrop; division's networks did too, yet giving the approximations took
    `d11` from 0 of 20 (the quotient alone) to 16 of 20.
  - Crystallisation has never trained beside an approximation (D39's (i) was never run); the owner:
    "no problem", the A/B tests the paper's learning with and without crystallisation.
  - *pending, at registration: each task's approximations by one rule, from `congruences` (row 17),
    and the build's tests.*
- **D19. X's tasks: paper 1's nine backdrops (settled under the owner's delegation):** `d7-family`,
  `d11-family`, `d13-family`, `d11-quotient`, word-problem-s3, word-problem-s4, move-to-front-4,
  random-q12-0, random-q16-0.
  - (a) *Chosen.* Pros: every machine type paper 1 measured (division's covers, the quotient alone,
    groups, a reset machine, random machines); harm readable where the paper cell works (division's
    15-17 of 20) and gain where it struggles. Cons: division's runs are the longest (about 2 h a
    spent run at 600 epochs).
  - (b) Without `d13-family`. Pros: about a ninth fewer runs, the longest. Cons: the trend over
    division's sizes read on two points.
  - (c) T10's planted machines added. Cons: T10's question, not a machine type; wait for its reading.
- **D20. X's seeds: 10 a cell, the same seeds in both arms (600-609), read in pairs (settled under the
  owner's delegation):**
  - (a) *Chosen.* Pros: the owner's "a few different seeds", enough to classify failures by kind
    across types; pairing, if a seed draws the same initial weights in both arms (*pending: checked
    at registration*), gives the comparison more power than two rates. Cons: per task, 10 pairs show
    only large effects; the across-task test carries the verdict.
  - (b) 20 (D13). Cons: twice the compute, for a reading whose main use is the failure analysis.
  - (c) 4-6. Cons: no per-task reading at all.
- **D21. X's readings (registered before its data; settled under the owner's delegation):**
  - **The effect:** per task, certified (both verdicts on the stop's network, D41) in each arm, the
    pairs' discordant counts by McNemar's exact test, two-sided; across tasks, a stratified exact
    test of the discordant pairs (the sign of each task's difference pooled exactly), two-sided,
    one-sided for crystallisation beside it (the owner's impression names the direction).
  - **Harm:** a task where crystallisation is significantly lower (Holm over the nine).
  - **Speed:** epochs to a certificate in each arm, per task, and the pairs' difference where both
    certified (reported, with the cost in epochs per certified machine).
  - **Failures:** every run's kind (failure-kinds.md: never formed, formed then lost (where: the
    handover, a collapse, the snap), froze, extra codes, readout), and formation (the state
    separation share reaching 0.95, D15's reading) read on every run of both arms.
  - **The stop's losses:** every run certified at a step the stop didn't end on (missed, or lost in
    the wait), and every false stop, counted per arm (the certificate logged every step as a judge).
  - **Adoption (D10's X part, amended):** crystallisation becomes the default if the across-task
    test favours it and no task shows harm; otherwise the paper's method stays, the failures read
    for a remedy.

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
1. **Part X:** D63's paired verdict read offline (D5); the groups' approximations built (D18, one PR);
   D21's exact tests (`inference.py`); the two aligned cells (D17) and their feasibility rows
   registered (one PR); a probe of each arm at width 32 for epoch time and memory; a note in the state-formation plan pointing its open test
   items here; the batch, run to its end (memory `no-deviations-from-early-peeks`); the reading
   (X1-X3, D10) into "Findings: X".
2. **B's builds (one PR each):** the crate's `merge` op (`indrajala-math-rust`, parity tests; tier 1
   A/B of the window program's benchmark, `--order ONNO`), then "Bump rust/", then
   `Branches(..., merge=True)` and `--recombine` (numpy and crate paths, parity tests, the golden run
   unchanged); `--branch-heads` for `--branches` (default 1, the retired design's; 2 here). These may
   be built while X runs.
3. **Part B:** the struggling set from the default's rates (D4); the cells on the default, feasibility
   rows and claims registered (one PR); probes first (claim nothing): both arms on the set's hardest
   backdrop, 4 seeds, a short epoch budget (if neither ever forms a state, B is reported as not run,
   with the probe); the batch; the reading (B1, B2, harm, D10) into "Findings: B".
4. **Part L:** `--window-blocks` (pure Python, the golden run unchanged); a depth-6 probe of the
   default, one seed, for its epoch time and memory; the set re-derived if B adopted an arm (D4);
   registered (one PR); the batch; the reading (L1, harm, D10) into "Findings: L".
5. **Findings** into [failure-kinds.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/research-notes/failure-kinds.md)
   and the feasibility register (rows 3 and 4 re-read, a row for depth).
6. **Retire** this plan; its leftovers (depth's bound, D7's options (b)-(d), the constants'
   sensitivity, F10) to [next-steps.md](../next-steps.md).

## The size, roughly (each part's probe replaces it)

- **B:** 2 arms x (the struggling set + 1) x 20, of 600 epochs (S4 3.2 h median in T5), plus the
  reproduction runs: for a set of 3, 160 runs; a day or two on every machine.
- **L:** (the set + 1) x 20 of 600 epochs at about twice the epoch time.
- **X:** 9 tasks x 10 seeds x 2 arms = 180 runs of up to 1,600 epochs (D64's spent runs about 10 h
  on `stone`; a run that stops early costs less): the largest part, days on every machine.

## Machines

The batch runner on `jebel`, `stone` (when the owner opens it), `pyramidon` and `klip`, memory per
host from each part's probe; `litshe` parity only.
