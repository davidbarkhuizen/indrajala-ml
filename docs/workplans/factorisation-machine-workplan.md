# Workplan: a network-machine that factors integers

**Status: written 2026-10-13, at the owner's request ("a detailed high-level workplan covering the
missing infrastructure that would be required to allow the network-machine to truly perform a
prime-factor decomposition"). Decisions D1-D12 settled under the owner's standing delegation, each
with its options' pros and cons. Revised 2026-10-14 against
[experiment-design.md](../experiment-design.md) and the work since (the claims workplan's E7-E8b and
D30-D34, the factored-state workplan): D1, D3, D4, D6, D9, D10, D11 and D12 revised, D13-D14
added, F1 restated as an existence claim, the stages and the cost redone. D13 then set by the
owner: **the controllers' state is factored, one factor per register**, R1 the cross-term's
first test. The owner granted the
study one multi-day run, the concluding one (D14). D11 revised by the owner (2026-10-14): F1 keeps
the calls given; F2's second arm learns them, factored per call level, in place of the inlined
flat controller (which multiplies control states as a single codebook multiplied division's). F1 reversed by the owner (2026-10-14, evening; D15-D17): no traces and no given decomposition; the network learns from inputs and outputs only and finds its own decomposition ("we can't hand the network the decomposition, it should learn it itself"), trained with the exact machine forward and straight-through gradients backward (D16), and claimed in tiers, the main route a modular certificate against the specifications of the subroutines it learned (D17). Items superseded by the reversal are marked so where they stand. Stages 4-7 and the cost then restated for D15-D17, with D18 (the learned call structure's shape) and D19 (the training executor, batched) settled under delegation, and a capacity row added to each rung's feasibility (2026-10-14). No stage started.** It follows
[factorisation-note.md](../factorisation-note.md), which states why the window-recurrent network
can't factor and what a machine that can would need (P1-P5); this plan builds that machine. It
reuses the window-recurrent network, its target-machine certificate (the claims workplan's
stage 5, D17), the factored state's tuple certificate and harness
(`git show e45944e:docs/factored-state-workplan.md`, stages 6-7), and the study rules of
experiment-design.md.

## Why

The window-recurrent network, its state snapped, is a finite-state transducer, and E1 showed that
its machine can be read off and proved correct at every length. Finite-state is also its limit:
it can't factor (the note's counting argument). The note's conclusion is that a **snapped
controller driving an exact, unbounded tape for as many steps as it chooses** has everything
factoring needs, and that the controller, being finite, could still be certified. The result would
be a learned program, proved equal to a reference algorithm on every input, that factors any
integer. Its cost would be that algorithm's.

That is the claim this plan works towards, registered here before any data:

> **F1 (reversed, D15).** A network with a factored state (a factor per register and per call level,
> bounded depth), run as the controller of a tape machine with an exact call stack and trained on
> **inputs and outputs only** (`N` on the input tape; its prime factors on the output tape; no
> traces, no reference states, no given subroutines), learns a program that factors every integer
> correctly, shown in the tiers of D17: **certified** (a proof at every length, by mapping onto a
> known program or by the modular certificate), **partially certified** (every learned subroutine
> certified, the loop verified to a bound), or **verified to a bound** (never called certified).
>
> *As first registered (superseded):* networks trained on execution traces of a factoring program,
> one per subroutine, each certified equal to that subroutine's controller.

F1 is an **existence** claim (experiment-design.md, section 2): one certified network per
controller proves it, however found; no rates, arms or controls. Running out of a rung's budget
records "not found within the budget", not a refutation, and stops the ladder there (D14).

And, as the research step beyond it:

> **F2 (superseded by D15).** Its two arms, inputs and outputs only, and the call structure learned
> by one factored network, are both now part of F1.

*Before D15:* F1 was engineering plus E1's method, *if* the method reached controllers of the rungs' sizes:
today it has certified machines of 3 states (addition, E1) and 4 (the modular sum, E8, 1 of 20),
not 5 (E8 `q` 5, 0 of 20; E8b's curriculum arm pending), and not division's 25 (E7, E7'). The
ladder (D14) is built to find where it stops. F2 is open (Déletang et al. 2022 found no
architecture that learned even binary multiplication from inputs and outputs).

**What this is not.** Not fast factoring: the machine runs trial division (exponential in the
digits), then Pollard's rho (about `N^(1/4)`). Factoring fast would be a result in number theory,
not in learning. The demonstration range is set by the step budget (the cost table below):
numbers to about 24 base-3 digits (38 bits) by trial division, about 40 (63 bits) by rho.

## The machine

A **tape machine** (D1): three tapes, each a row of cells holding one symbol, with one head each,
and an exact **call stack** of return points.

- **Tape 0, the input:** `N`'s digits, most significant first, between end markers; read-only.
- **Tape 1, the work tape:** the candidate divisor, the running remainder, the quotient, scratch.
- **Tape 2, the output:** the prime factors written as they are found, separated.
- **The call stack:** which controller runs, and where each caller resumes. Its depth is fixed by
  the program (no recursion: R7 calls R6 calls R5 calls R3, at most 4 deep), so it is a finite,
  exact part of the machine, as the tapes are.

Symbols (D2): the base-3 digits 0-2, a blank, a separator and two end markers: 7 symbols.

A **step**: the running controller reads the three symbols under the heads and its state; it
writes a symbol under each head, moves each head left, right or not, and either moves to its next
state, **calls** a subroutine (its controller starts in its start state; the caller's next state is
pushed), or **returns** (the caller resumes at the pushed state; a return from the top controller
halts). The tapes, heads and stack are exact by construction: plain memory, not learned (P1, P3,
P4). The machine runs until it halts or a step budget runs out (P2, P5).

**Each controller is the window-recurrent network as it is** (D4): a window of `k = 3` columns,
one per head, each column's token the symbol under that head (7 tokens), the state **factored**
(D13): one factor per register of the reference program (its program counter, a carry or borrow,
a comparison's result), each its own cell and codebook, the cross-term on, the state a tuple of
codes; and each column's output that head's action, a write symbol and a move (7 x 3 = 21 classes),
column 2's also the control action (continue, call a named subroutine, return; D10). What changes
is where the next window comes from: from the tapes after the actions, not from a fixed row of
tokens. So each controller is a finite machine of `m = m_1 ... m_G` states over `7^3 = 343`
windows, **and the target-machine certificate applies to it**, over the factored state's tuples:
enumerate every reachable state against every window the reference can see, map states to the
reference controller's (`phi`), check every action and next state, and read which register each
factor holds (the factored-state workplan's D9). A certified controller **is** its reference controller; the machine around
the controllers is exact; so if every controller is certified, the network-machine computes what
the reference program computes, on every input.

What it then factors correctly depends only on the reference program, which is ordinary code,
tested and argued as any program is (D7).

**Why one controller per subroutine (D3, D13).** Trial division compiled flat is an estimated
100-300 states. Under the exclusion bound a flat controller with a surplus of codes (3-5 times)
trains on segments of `m + q - 1`, 400-1,800 steps, a machine 25-75 times larger than any the
method has certified, and E7' showed one codebook failing to fit a 25-state table. Split at its
subroutines, each controller is a few to a few tens of states, each trained, certified and run
alone, the smallest first: the ladder of experiment-design.md's "small case first". The call
structure is then given, not learned; learning it is F2's learned-calls arm (D11). Within a controller, the
factored state gives each register its own codebook (D13), the structure E7' lacked: a program
counter and a carry learned as two small counters, not their product as one table.

## The missing infrastructure

Eight pieces, each a stage or part of one. None exists in the package now.

### I1. The tape machine and its executors

- **A tape model:** tapes growing on demand, heads, symbols, the call stack, a step function taking
  actions to the next configuration; configurations hashable (for traces and tests).
- **Three executors** of a set of controllers (D5):
  - the **reference executor**, running the reference programs' tables (I2), in Python;
  - the **network executor**, running networks as the controllers step by step (`window_step`
    with the window built from the heads' symbols), on either backend: slow (about 31,000 window
    evaluations a second, E1's rate), for validation;
  - the **compiled executor**, running certified networks' extracted tables (codes, actions and
    next codes, from their certificates) and the stack in the crate, Rust: about 10^8 steps a
    second, the executor for the long runs.
- **The step budget and halting:** a budget a run, a function of `n` given by the reference
  program's bound; a run out of budget is a failure, recorded with its configuration.

### I2. Reference programs: the algorithm ladder

A small language for tape-machine controllers (D3): **a state of named registers** (the program
counter and each flag or digit a subroutine carries, each with its own range: these become the
network's factors, D13), and for each state and the symbols under the heads, the writes, moves, and next state, call or return. It compiles each
program two ways: **a table per subroutine** (the units trained and certified, called through the
stack), and **one flat table** with every subroutine inlined (the whole program as one machine:
the composition's check). Each program gets its own tests: exhaustive at short
lengths against Python's integers, sampled at long ones.

**Each table's size is computed, not estimated** (experiment-design.md, section 1): its minimal
state count by partition refinement over the table's domain (the windows it can see; a window
outside it given a sink output, so the count is an upper bound on a partial machine's minimum),
its states' reachability and distinguishability within the trace lengths, and each register's
range (its factor's minimal codes). These fill the
feasibility table (below) when each rung is written.

The ladder (D6), each rung a program, each later one calling earlier ones:

| rung | program | tapes | why it's on the ladder |
| --- | --- | --- | --- |
| R1 | copy, compare (`a < b`, `a = b`) | 2 | head movement, the basics of every later rung |
| R2 | increment, decrement, add | 2 | the carry: E1's automaton on a tape |
| R3 | subtract with borrow | 2 | the remainder's step |
| R4 | multiply (shift and add) | 3 | superlinear steps (Déletang et al.'s failure case); needed by rho |
| R5 | divide with remainder (schoolbook long division by repeated subtraction) | 3 | the trial's work |
| R6 | divisibility test `d \| N` | 3 | R5 with the quotient discarded |
| R7 | **trial division factorisation**: `d` from 2 while `d^2 <= N`; on `d \| N`, write `d`, set `N = N / d`, repeat; write the last `N > 1` | 3 | **F1's target** |
| R8 | gcd (Euclid by subtraction or by R5) | 3 | for rho |
| R9 | **Pollard's rho** with Floyd's cycle finding, `x -> x^2 + c mod N`, a restart with the next `c` on failure, primality by trial division of the factors found up to a bound (or Miller-Rabin, R10) | 3+ | the second target: sub-exponential steps |
| R10 | Miller-Rabin with fixed bases | 3+ | primality of large factors, deterministic below 3.3 * 10^24 with the first 13 primes as bases |

Rungs R1-R7 are the plan's core; R8-R10 its stretch (stage 9). R7's own controller (the loop over
`d`, its calls and writes) is a rung of its own: F1 holds when it and every controller it calls
are certified.

### I3. Number-theory data and the factorisation catalogue

- An **oracle** in Python: factorisation by trial division and Pollard's rho, Miller-Rabin; tested
  against each other and exhaustively to `3^12`. No new dependency (D8).
- **A catalogue of properties** for factoring, as addition's and division's catalogues: every `N`
  at short lengths; sampled at long ones; and strata that stress the algorithm: primes (the
  longest trial), prime powers (repeated division), semiprimes `pq` with balanced factors (the
  hardest for trial division and rho), smooth numbers (many small factors), squares, Carmichael
  numbers (for R10), and the edge cases 0, 1, 2 and 3. Each case's result checked whole: the
  factors written, in order, their product `N`, each prime. Checked at full catalogue sizes
  (experiment-design.md, section 4).
- **Per-rung catalogues** for R1-R6 and R8 (each rung's inputs and its answer), so each rung is
  trained and certified alone before it's composed.

### I4. Traces and the training data

- **Trace generation:** a reference program run on an input gives, per controller, the trace of
  each call: for every step, the symbols under the heads (the window), the reference state, and
  the actions. A call's trace is exactly a window-recurrent example: windows in, an action per
  column out, the state carried. **Every call starts in its controller's start state**, so a
  controller's examples are whole calls from their start, with no teacher-forced mid-run states
  (D9); a long call is cut only past the training cap.
- **The training length is the exclusion bound** (experiment-design.md, section 1; D14): a
  controller of `m` codes against a reference table of `q` states trains on calls of up to
  `m + q - 1` steps, the inputs chosen long enough to give them. A curriculum over call length
  from 1-2 steps (E8b: a fresh network's state starts on 1-2 codes, and only short lengths split
  it), the network and its optimiser carried (W25), the bound its cap.
- **Action labels per head** with the control action on column 2; a loss weighting so a call's
  one return isn't swamped (D10).

### I5. The controller certificate

The target-machine certificate (the claims workplan's D17), with the factored-state workplan's
tuple states, per-factor readout and `window_tasks.Coordinates` (its stage 6), already does nearly
everything: the controller's windows are the heads' symbols, its outputs the actions, its states
tuples of codes, and the reference's registers its coordinates, so the readout names the register
each factor holds (or a mixture of them). What's added:

- **Windows that can't occur are left out:** the input tape's head never reads a work symbol, so
  the enumeration is over the windows the reference can see from each state (its table's domain),
  not all 343. A window outside it is "don't care" in the reference and must be shown unreachable
  from the start (a reachability pass over configurations' symbol triples, as the certificate's
  breadth-first search over codes).
- **Call and return** as output classes, the reference's mapped.
- **The composition:** with every controller certified, the composed machine equals the flat
  table by construction (the inlining is the composition); a check runs the composed certified
  tables against the flat table's executor on every `N < 3^12`.
- **The certified tables' export**, for the compiled executor (I1): codes, windows, actions, next
  codes; and a check that the compiled executor and the network executor agree step for step on
  sampled inputs.

### I6. Correctness of the reference programs

The certificate reduces the network's correctness to the reference program's. That needs its own
evidence (D7): exhaustive runs against the oracle for every `N < 3^12` (531,441 numbers), sampled
runs to 40 digits per stratum, and for R7 a written argument (the loop invariant: the factors
written times the remaining `N` is the input, every factor written is prime, `d` only grows); a
mechanised proof (Lean) is a later option, not this plan.

### I7. Measurement and the harness

- **The window study's harness, with a task per controller**: a controller's calls as its
  examples, its reference table as the target machine; runs of rung, `m` and seed; each step saved
  and resumed; the factored-state workplan's `certify_steps` (a certificate after every curriculum
  step) and `stop_at_certified` (the pool stopped at the first certified run); status and report.
  A new harness only for what a controller's task can't express.
- **Measures:** certified or not, and at which step; steps per input against the reference's
  (equal if certified); wall time per executor; the largest `N` factored in a budget; codes used
  against the reference's states; on a failure, the certificates' counterexamples, read before
  anything is changed (experiment-design.md, section 4: fitting or memorising).
- On `jebel`, workers sized by a measured peak RSS, PL1 65 W, under the machine lock, as every
  study.

### I8. Package and persistence

- The tape machine and its executors in `indrajala_ml/machines/` (new): the Python reference and
  network executors; the compiled executor in the crate (`indrajala-math-rust` first, then a
  "Bump rust/" PR); parity tests between them.
- The controller network: the window-recurrent network, unchanged (format 2, its golden entries),
  one factor per register, the cross-term on (D13). A new class only if D4's reuse fails (per-head classes the window
  network can't express).
- Certified tables saved beside the network (format 2's JSON or a sidecar), with the
  certificate's sha256; certified networks archived in `davidbarkhuizen/indrajala-models`.

## Feasibility

Written per rung when its table is compiled (stages 2-3), and a test checks each committed config
against it (experiment-design.md, sections 1 and 6). For a controller of reference size `q`
(minimal states, computed) and registers of ranges `q_1 ... q_G`, factor `g` gets `m_g` codes,
at least `q_g + 1` and about `2 q_g` (a surplus per factor costs its product, `r^G`, so it is
smaller than a flat codebook's 3-5 times), and the network's states are `m = m_1 ... m_G`:

| rung | controller | `q` | registers `q_g` | reached / told apart within | codes `m_g` | `m` | cap `m + q - 1` steps | inputs giving calls that long | probe: estimate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R1 | copy, compare | computed | computed | computed | | | | | |
| ... | | | | | | | | | |
| R7 | the loop over `d` | computed | computed | computed | | | | | |

**Representable only with the cross-term.** A program counter's next value reads the carry (and
the carry's, the counter), so the registers don't update independently: with the cross-term off
(C9's cell) a factored controller can't represent its table, and the cell would fail D23's rule.
Every controller runs with it on; the poolings read the output layer's tokens (the default, D10 of
the factored-state workplan), the independence `BEFORE_STATE` enforces not being wanted here.

**The bound on a tape.** A controller's inputs are its windows, but on a tape they aren't free
words: the tapes answer the controller's own moves, so training sees the words the reference's
calls produce. The bound then makes memorising unlikely rather than impossible; the certificate,
over every window of the domain from every reachable code, is the proof (as the factored-state
workplan's D7).

**The capacity row (D15-D17, D18).** Learned from inputs and outputs, a rung's whole machine has
no exclusion bound (D17), so each rung's row adds: the shape (`J`, `L`, each factor's codes, `S`
their product), at least the reference's (D18); the table's bits, `S x 343 x` the bits of an
entry's actions; and the curriculum's cap, the smallest input size at which some `T` inputs'
answers hold more bits than the table, with `T` written (the cost's table for R7). A row whose
inputs all together hold fewer bits than the table at every size its budget allows is an
impossible cell, not launched. A test checks each config against its row.

## Decisions

Settled under the owner's standing delegation, with each option's pros and cons; revisions of
2026-10-14 marked.

- **D1. The machine model. Settled: (a), with an exact call stack (revised 2026-10-14).**
  - (a) *Chosen.* A multi-tape Turing machine, three tapes, one head each, **and an exact call
    stack of bounded depth** (D3). Pros: each controller is a finite table over (state, symbols
    under heads), exactly what the certificate checks; the tapes and the stack are trivially
    exact; well understood (time bounds, simulation). Cons: head shuttling costs steps (a long
    division is `O(n^2)` moves at best); the stack is more machine to build than tapes alone.
  - (b) A register machine (registers of unbounded integers, increment, decrement, test for zero).
    Pros: few steps per operation. Cons: the controller's view is "zero or not" per register, so
    the arithmetic lives in the machine, not the network; nothing learned.
  - (c) A hierarchical program machine with a learned call structure (Neural
    Programmer-Interpreters, Reed & de Freitas 2016). Pros: compositional, recursion. Cons: an
    unbounded stack needs its certificate abstracted (Cai et al. 2017 verify recursion on a
    finite set of configurations). The revision takes its compositional half with the calls
    given and the depth bounded, which needs no abstraction.
  - (d) A soft, differentiable memory (Neural Turing Machines). Cons: inexact; drift (C2): the
    opposite of what the certificate needs.
- **D2. The base and the alphabet. Settled: (a).**
  - (a) *Chosen.* Base 3, as the studies: digits 0-2, blank, separator, two end markers (7
    symbols). Pros: E1's and E8's automata reusable as subroutines and their certificates
    comparable. Cons: binary would need fewer windows (`6^3` against `7^3`).
  - (b) Base 2. Pros: the literature's (Kaiser & Sutskever, Déletang et al.); smaller tables.
    Cons: a break with the studies' tasks.
- **D3. How reference programs are written and what is learned. Settled: (d) (revised
  2026-10-14; was (a)).**
  - (a) A small Python DSL compiled to one flat table, subroutines inlined, one controller learned.
    Pros: the table is one target machine; one network. Cons: inlining multiplies states (100-300
    for R7); the bound puts its training segments at 400-1,800 steps and its size 25-75 times past
    the method's certified frontier (4 states): against "small case first", and likely past a
    day a run.
  - (b) Hand-written tables. Cons: unreadable past R2.
  - (c) Subroutine calls kept (a call stack), only the per-subroutine tables. Pros: small
    controllers. Cons: no flat table for the composition check or F2's flat arm.
  - (d) *Chosen.* **The DSL compiles each program both ways**: a table per subroutine, each
    learned and certified as its own controller, composed by the exact stack (D1); and the flat
    inlined table, the composition's check (I5). Pros: each controller
    near the method's frontier, so the ladder starts where networks are known to certify and
    steps up one controller at a time; short bounds, so runs fit a day; R3's certified controller
    serves R5, R6 and R7 alike. Cons: the call structure is given, not learned, so F1 claims less
    than a learned call structure would (F2 asks for that, D11); more compiler (two outputs).
- **D4. The controller. Settled: (a) (revised 2026-10-14: one per subroutine).**
  - (a) *Chosen.* The window-recurrent network, `k` = 3 heads, a column per head, unchanged, its
    snapped state the controller's state; **one network per subroutine** (D3). Pros: no package
    change to start; the certificate, the harness, the factored state and the training carried
    over. Cons: a column's output is one head's action, so the control action goes on one column,
    and per-column classes are the product write x move. **The curriculum is the training arm**:
    the claims workplan's C4 ("the curriculum isn't needed") is untested past 3 windows, and
    E8b's all-at-once arm at 20 windows left the state on one code (D33).
  - (b) A new controller class (one output vector, a head for each part of the action). Pros:
    the natural shape. Cons: a new network in three implementations, format 2, fixtures and golden
    entries before anything is learned.
- **D5. Executors. Settled: (a).**
  - (a) *Chosen.* Reference (Python), network (either backend, validation) and compiled (Rust, the
    certified tables and the stack). Pros: long runs at 10^8 steps a second, possible only because
    the certificate makes the table the network's exact behaviour; the network executor checks
    the compiled one. Cons: three executors kept in parity.
  - (b) The network executor only. Cons: 31,000 steps a second: R7 at 20 digits takes hours a
    number.
- **D6. The algorithms and their long runs. Settled: (a) (revised 2026-10-14: the long runs
  sized).**
  - (a) *Chosen.* Trial division (R7) as F1's target, Pollard's rho with Miller-Rabin (R9, R10) as
    the stretch. R7's long runs to 24 digits (30 s a number compiled), rho's to 40 (30 s a number);
    R7 at 40 digits (7 days a number) isn't run. Each long-run batch sized to a day. Pros: R7 is
    the simplest correct factoriser; rho reaches 63 bits in the step budget. Cons: neither is
    fast; rho's random restarts need a fixed sequence of `c` (deterministic, so certifiable).
  - (b) Trial division only. Cons: tops out near 38 bits.
  - (c) The quadratic sieve or the number field sieve. Pros: the fast algorithms. Cons: linear
    algebra over GF(2) and sieving on tapes: far more states and tapes; a later plan if (a)
    succeeds.
- **D7. Evidence for the reference programs. Settled: (a).**
  - (a) *Chosen.* Exhaustive runs to `3^12`, sampled runs per stratum to 40 digits, a written
    invariant per program. Pros: cheap and strong. Cons: not a proof for every `N`.
  - (b) A mechanised proof (Lean). Pros: a proof. Cons: a project of its own; later.
- **D8. Number-theory code. Settled: (a).** (a) *Chosen.* Our own oracle (trial division, rho,
  Miller-Rabin) with tests. Pros: no dependency, the same code reviewed. Cons: code to write.
  (b) sympy. Pros: trusted. Cons: a large dependency for three functions; added later if a
  check needs it.
- **D9. Training examples. Settled: (d) (revised 2026-10-14; was (b)).**
  - (a) Whole traces of the whole program, short inputs only. Cons: one flat controller (D3 (a));
    R7's traces on inputs long enough to exercise every branch are 10^5-10^6 steps.
  - (b) Segments of the flat trace, starting from mid-run states given as codes once a partial
    certificate maps them. Cons: the partial certificate first; tied to the flat controller.
  - (c) Per-step examples, the state given as the reference state's one-hot. Cons: the network
    doesn't learn its own state; nothing recurrent learned (a lookup table fit).
  - (d) *Chosen.* **Each controller's calls, whole, from its start state**, by a curriculum over
    call length capped at the exclusion bound (I4). Pros: every example starts where the
    controller starts, so no teacher-forced states; bounded by the controller's own bound, not
    the program's trace; the window study's curriculum as it is. Cons: a call long enough for the
    bound needs inputs chosen for it (written in the feasibility table).
- **D10. The control action and the loss. Settled: (a) (revised 2026-10-14: calls and returns).**
  (a) *Chosen.* Continue, call (one class per callee) and return on column 2, the loss weighted
  so a call's control actions count as much as its other steps together. Pros: the network
  decides its calls and when to stop (P5). Cons: a weight to choose. (b) The executor calls and
  returns on reference states the certificate names. Cons: the network doesn't decide: not P5.
- **D11. F2's arms. Settled: (b), by the owner (revised 2026-10-14; was (a)).** After F1, two
  existence tests, each registered with its feasibility row and probe, each "found" or "not found
  within the budget", as F1. The first is the same in every option: **inputs and outputs only** per
  controller (no trace). The second asks for the call structure learned:
  - (a) **One flat controller** for R7 on full traces, every subroutine inlined. Pros: no
    decomposition given at all. Cons: inlining multiplies control states (subroutine x position x
    return point: 100-300 for R7, 25-75 times past the certified frontier), the same blow-up as
    one codebook for division's A x B (the claims workplan's D32); its cap likely over a day.
  - (b) *Chosen.* **One network that learns its calls, factored**: a state factor per register (as
    D13) and a factor per call level up to the stack's bounded depth (D1), the network deciding
    call and return itself (D10's classes); certified as a product machine (the factored-state
    certificate). Pros: the structure learned without inlining's product: state sizes add across
    levels instead of multiplying, each factor near the frontier; the strongest claim within
    reach. Cons: new design (a D entry for the level factors and the trace's call/return labels);
    the factors must coordinate through the cross-term, which C9 and R1 test first.
  - (c) Inputs and outputs only; no second arm. Pros: smallest. Cons: the call structure stays
    given everywhere; F2 asks less.
  The owner first chose F1 itself as the flat controller, then, on the distinction between a flat
  program and a flat state, chose (b) here with F1 unchanged: "surely the controller needs to be
  multi-factor and not flat".
- **D12. Where the plan sits. Settled: (b) (revised 2026-10-14; was (a)).**
  - (a) Everything after the claims workplan's report. Cons: the infrastructure, which needs no
    `jebel` and answers nothing about the method, waits for nothing.
  - (b) *Chosen.* **The infrastructure (stages 2-5) may start any time**, development on
    `pyramidon`, tier 0. **The experiments (stage 6 on) start after the planned experimental work
    ends** (the claims workplan's stages 7-9 and E8b, the factored-state workplan's C9) **and the
    owner's strategy review that follows it** (memory: finish what is planned, then reconsider).
    E8b and C9 say where the method stands (5 states; a product of two counters learned), which
    sets the first rungs' codes. Pros: no `jebel` time
    taken from registered work; the ladder registered on the newest evidence. Cons: infrastructure
    built before it is known whether the method reaches R1's size.
  - (c) Experiments too, beside the claims work. Cons: competes for `jebel`; against finishing
    what's planned first.
- **D13. The controller's state. Settled: (b), by the owner (2026-10-14; first settled (a)
  under delegation). A cross-term existence test before R1 (a known task, the carry automaton split
  into a counter and a carry, 6 seeds, a day) was put and declined: R1 is its first test.**
  - (a) One factor (today's network) first; a factored state as a rung's follow-up when its
    failures read as fitting. Pros: one change at a time. Cons: misses the study's point; a
    controller's state is registers, and E7' showed one codebook failing to fit a product table.
  - (b) *Chosen.* **Factored from the start: one factor per register the DSL declares, the
    cross-term on** (the factored-state workplan's D1 (d) and D2 (b), which named this plan's
    registers among G factors' uses). Pros: each register its own small codebook, the shape of a
    controller's state; the readout names what each factor holds; the certificate and harness
    already take tuples. Cons: the cross-term on is untested in any study (C9 runs it off), so
    R1 is its first test; the exclusion bound counts the product of the factors' codes (the
    factored-state workplan's D7), so it doesn't shorten training and a surplus is kept small per
    factor (Feasibility).
  - (c) Factored, a fixed G (the program counter and one data factor) for every controller. Pros:
    one network shape. Cons: arbitrary; a controller with two flags tangles them in one factor.
  - (d) Factored, the cross-term off. Cons: the registers interact, so it can't represent a
    controller (Feasibility): every cell impossible.
- **D14. The experiments' design: an existence ladder (added 2026-10-14).**
  - (a) *Chosen.* **Each controller a rung, one configuration** (`k` 3, `m` about 3-5 times its
    `q`, the curriculum capped at `m + q - 1`), **6 seeds in parallel on `jebel`**, each certified
    after every curriculum step and **the pool stopped at the first certified network**
    (experiment-design.md, section 2; the factored-state workplan's D7-D8). A rung registered with
    its feasibility row before it runs; the next registered only once a rung works (the claims
    workplan's D30); a 2-epoch probe first, for the budget and for a state stuck on one code
    (D33), and a rung the probe shows can't succeed isn't launched. **Every rung within a day**,
    its fallback (fewer codes, the surplus kept at least 2 times `q`) registered with it; **except
    one: the owner allows the study a single multi-day run to conclude it (2026-10-14), spent on
    E-F1** (R7's own controller), registered with its probe's estimate and a stated end, not
    extended after the data. If E-F1 fits a day, the exception goes unused; it isn't moved to
    another rung without the owner. Pros: compute only where a certified controller can come
    out; each rung answers one question; the study ends. Cons: no rates (a lone success gets a
    small exploratory seed follow-up, labelled so); a rung not found stops the ladder.
  - (b) A matrix per rung (codes, arms, 20 seeds). Pros: rates with intervals. Cons: against the
    owner's short-focused rule; days of `jebel` for questions F1 doesn't ask.

- **D15. What F1 learns from. Settled: (a), by the owner (2026-10-14, evening).** The owner: "we
  need to reverse F1 - we can't hand the network the decomposition - it should learn it
  unsupervised itself".
  - (a) *Chosen.* **Inputs and outputs only; the decomposition learned.** The network is trained on
    `N` and its factors; the trace, the reference states, the per-subroutine controllers and the
    call structure are not given. "Unsupervised" is as to the method: the answer stays the
    target, or nothing defines the task. One factored network (a factor per register and per call
    level, as D11 (b)); every ladder rung (R1 copy and compare, R2 add, ...) learned from scratch
    from its own inputs and outputs, no trained subroutine carried into a later rung (that would
    hand the decomposition over again). Pros: the claim is discovery, not imitation; it subsumes
    F2. Cons: the hardest form; inputs-and-outputs learning has stalled well short of this before
    (RL-NTM solved copy, duplicate, reverse and repeat-copy, nothing superlinear, and only by a
    curriculum: Zaremba & Sutskever 2015; NRAM generalised on its easier tasks: Kurach et al.
    2016); the certificate loses its reference (D17).
  - (b) Traces, as first registered. Cons: hands the network the decomposition (the owner's
    objection); the method is NPI's (Reed & de Freitas 2016), and with recursion given, its proof
    is Cai et al.'s (2017).
  Superseded by it: D3's per-subroutine controllers as the learned objects (the DSL still compiles
  the reference programs, now the oracle, the long runs' check and the tier-1 reference), D9's
  examples (I4's traces become the reference executor's diagnostics, not training data), D11's
  arms (both now F1), and D14's "each controller a rung" (now each task a rung).
- **D16. How it learns. Settled: hard forward, soft backward, by the owner (2026-10-14, evening).**
  - (a) *Chosen.* **The exact machine forward; straight-through gradients backward.** Every action
    (a write, a move, a call, a return) is a hard choice in the forward pass, so the tapes, heads
    and stack stay exact in training as at test, and the machine trained is the machine certified;
    the backward pass is relaxed: straight-through Gumbel-softmax for the actions (Jang et al. 2017;
    Bengio et al. 2013), soft attention over tape cells for the reads' gradients, as the snapped
    state already trains (VQ's straight-through). With it: **a dense, verifiable loss on the answer,
    not the method** (each factor written scored as it is written: divides `N`, is prime, the
    product so far divides `N`); **expert iteration on its own verified runs** (Anthony et al.
    2017; Zelikman et al. 2022: runs whose output verifies become its training traces, its own,
    not ours); **a curriculum over input size** (small `N`, short runs, where a near-random
    controller sometimes succeeds, which starts the iteration; RL-NTM needed one). More layers are
    acceptable (the owner). Pros: a gradient through the whole run with no train/test mismatch;
    reuses the straight-through machinery. Cons: straight-through over thousands of discrete steps
    is biased and may not train; new layers (soft-read gradients, action heads).
  - (b) Soft in training, hard at test (NTM, NRAM: Graves et al. 2014; Kurach et al. 2016). Cons:
    the soft machine drifts and the hard one differs from what trained (NRAM's discretisation was
    exact only on some tasks).
  - (c) Reinforcement learning on the exact machine. Cons: sparse reward over thousands of steps;
    RL-NTM's ceiling.
  - (d) Search over snapped controllers (evolutionary or MDL, Lan et al. 2022). Cons: not gradient
    learning; scales badly with 343 windows times states.
- **D17. What F1 claims, and how it is proved. Settled: (b) as the main route, by the owner
  (2026-10-14, evening).** A network that finds its own decomposition may compute no known
  machine, so the certificate's `phi` onto a reference has nothing to map to, and correctness of
  an arbitrary tape program on every input is undecidable in general. But a subroutine that sweeps
  the heads monotonically over aligned digits (copy, compare, add, subtract, a division's step)
  is a finite transducer over columns: the object the certificate already proves at every length.
  - (a) `phi` onto a known program's flat table, the certificate as built. Pros: automatic.
    Cons: only if the network learns the reference state for state.
  - (b) *Chosen.* **The modular certificate against identified specifications.** Read the learned
    call structure off the snapped network (its calls and returns are discrete); for each learned
    subroutine, record its input-output behaviour, identify its function, and certify it as a
    transducer against that specification at every length; prove the outer loop from the
    certified specifications by an invariant proof, mechanised in Lean (D7 (b)) or written and
    checked. Pros: the decomposition stays the network's, only named by us; every piece proved;
    a stronger claim than with traces (a program it invented, proved); graceful (each certified
    subroutine is a result). Cons: identifying a specification is manual per subroutine; a
    non-monotone subroutine needs another certificate; a loop proof per algorithm. Cai et al.
    (2017) prove generalisation from base cases and reductions given the recursion; here the
    structure is the network's.
  - (c) A mechanised proof of the whole extracted machine. Cons: a project per algorithm.
  - (d) Verified to a bound only. Cons: what E1 showed overstates success; a fallback tier only.
  **The tiers**, registered before any data: *certified* by (a) or (b); *partially certified*,
  every learned subroutine certified and the loop verified to a bound; *verified to a bound*,
  exhaustive to `N < 3^12` and sampled to 40 digits. **Memorising:** the exclusion bound applies
  per certified sweep subroutine; the whole machine has unbounded memory and no such bound, so at
  the top level the defence is capacity (a training set of `N` holding far more answers than a
  controller's table of `S` states x 343 windows can store), written beside each rung.
- **D18. The learned call structure's shape. Settled: (a), under the owner's delegation
  (2026-10-14).** D15 leaves the decomposition to the network, but the network needs a shape to
  hold one: call slots, a depth, and codes per factor.
  - (a) *Chosen.* **One network for every level, `J` call slots, depth `L`, sized from the rung's
    reference program with one spare of each:** `J` at least the reference's subroutines plus
    one, `L` its call depth plus one, each slot its own learned start state (so the slot is in the
    state), the stack holding the suspended levels' state tuples (D11 (b)'s level factors), the
    state `G` factors with codes at least the largest register range plus one (Feasibility).
    Column 2's classes: continue, return and one call per slot. Pros: every cell able to succeed
    (the reference's decomposition fits, with room for another); one network, so its table and the
    capacity row stay small; the reference gives sizes only, never which slot does what. Cons: the
    sizes are a hint from our program; a decomposition needing more slots than the reference's
    can't be found.
  - (b) A large fixed shape for every rung (8 slots, depth 6). Pros: no hint at all. Cons: more
    control classes and codes, so a larger table, a larger `T` (the cost table) and more to search;
    unused slots cost training.
  - (c) One level, no calls (factored, inlined). Pros: simplest. Cons: inlining's product (D11
    (a)): R7's 100-300 states, far past the frontier.
  - (d) A network per slot. Pros: each subroutine its own weights. Cons: the slots' networks
    multiply the parameters and the table; which network a slot gets is itself a given
    decomposition.
- **D19. The training executor. Settled: (a), under the owner's delegation (2026-10-14).**
  - (a) *Chosen.* **The window network on the Rust array backend, `B` inputs stepped in lockstep**
    (one `window_step` of `B` rows a machine step), the tapes, heads and stack in Python (an array
    of symbols per row), inputs of a batch drawn at one size so they halt near together, halted
    rows masked. Pros: the network and its backward reused as the studies train them; batching
    amortises Python's per-step cost; no crate change to start. Cons: Python's loop per step
    remains; rows halting early idle their slots.
  - (b) Per input, unbatched (the pure-Python lanes). Pros: simplest. Cons: about `10^4` steps a
    second: E-F1 at 12 days a pass (the cost table). Kept for the parity test only.
  - (c) The training executor in the crate (forward and backward through the machine). Pros: the
    fastest. Cons: a crate change and its parity with (a) before anything learns; taken only if
    stage 5's probe makes E-F1 infeasible with (a) (the cost).

## Stages

1. **This workplan and a literature review** (docs only): neural program induction and
   verification (Reed & de Freitas 2016; Cai et al. 2017; Graves et al. 2014; Kaiser & Sutskever
   2016; Déletang et al. 2022; Graves 2016) and, for D15-D16, learning from inputs and outputs and
   its signal (Zaremba & Sutskever 2015; Kurach et al. 2016; Anthony et al. 2017; Zelikman et al.
   2022; Ellis et al. 2021; Mankowitz et al. 2023; Jang et al. 2017; Bengio et al. 2013), each
   paper kept in the literature repository (`davidbarkhuizen/indrajala-literature`, cloned as
   `~/code/indrajala-literature/`): **all fetched and read (2026-10-14)**; their bearing is in
   D15-D17. The workplan is done (#708, revised; reversed by D15-D17; stages 4-7 and the cost
   restated for D15-D17, with D18-D19, on 2026-10-14).
2. **I1's tape model, the call stack and the reference executor; I2's DSL compiling both ways;
   rungs R1-R3** (tier 0): programs tested exhaustively; each table's minimal states computed and
   R1-R3's feasibility rows written.
3. **I3's oracle and catalogues; rungs R4-R7** (tier 0): R7 checked against the oracle to `3^12`
   (I6), and the flat table against the composed tables on the same; R4-R7's feasibility rows.
Stages 4-7 restated for D15-D17 (2026-10-14). *As first written (superseded):* 4, I5's controller
certificate against a given reference table; 5, I4's traces and a network trained on R1's; 6-7,
each rung's controller trained on its traces, R7 calling R1-R6's certified controllers.

4. **I5': the modular certificate (D17 (b))** (tier 0), built and tested on stand-ins before any
   network learns:
   - **The call structure read off** a snapped network: run on a rung's catalogue by the network
     executor, the call slots it uses (D18), each slot's call sites, and each call's tape contents
     before and after (its observed input-output behaviour).
   - **A sweep's certificate:** a slot whose heads each move one way only (a sweep) is a finite
     transducer over columns; its machine, extracted from the slot's start code over every window
     of its domain, is checked against a **specification**, a DSL program (I2) named by us after
     reading its behaviour, by the target-machine certificate (states mapped by a breadth-first
     search of the product, `phi` many-to-one, actions and next states checked; a mismatch gives
     a counterexample input). A slot that isn't a sweep is reported as such: it falls to a lower
     tier, it doesn't fail the certificate.
   - **The loop:** with every slot certified, the outer program (the top level's machine over the
     slots' specifications) is checked to a bound (every `N < 3^12` on the compiled tables) and
     its invariant written (D7).
   - **The export** of each certified slot's table for the compiled executor (I1), as before.
   - Tests: stand-in networks whose machines equal R1's and R2's tables certify against their own
     programs, and one composed of R2's and R3's tables certifies slot by slot; faulty stand-ins (a
     wrong write, a wrong move, a missed return, a wrong call) fail with a counterexample the
     executor runs wrong; a non-sweep stand-in reports its tier.
5. **I9: learning through the machine (D16) and the network executor** (tier 0), on `pyramidon`:
   - **The executor, batched** (D19): `B` inputs run in lockstep, one `window_step` of `B` rows a
     machine step, each row its own tapes, heads and stack; a halted row masked. It records each
     step's window, state and sampled actions: the hard forward.
   - **The backward:** the recorded windows replayed through the window network's existing
     backward (W20's re-run), plus the new gradient routed through the tapes: an action's
     straight-through Gumbel-softmax (Jang et al. 2017; Bengio et al. 2013); **a read's gradient to
     the write that put that symbol in the cell** (the cell's last writer, a hard pointer, so
     nothing soft in the forward); a move's through the three cells it could have brought under
     the head, weighted by the move's probabilities. Needs the window's input gradient per token
     (the embedding's backward, exposed).
   - **The loss on the answer only:** each output-tape cell's cross-entropy against the answer's
     symbol there (dense, on the answer, not the method); for R7, D16's verifier terms (each factor
     scored as written: divides `N`, is prime, the product so far divides `N`); a run out of its
     budget scores its unwritten cells wrong.
   - **Expert iteration** (D16): `K` runs sampled per input; a run whose whole output verifies
     becomes a trace of the network's own actions, trained on with each step's cross-entropy
     alongside the answer's loss.
   - **The curriculum over input size** (D16): from the smallest inputs, the network and its
     optimiser carried (W25), capped at the rung's capacity row (Feasibility).
   - **The call structure** (D18): the level factors, the call slots' classes on column 2, the
     exact stack holding the suspended levels' codes.
   - **The harness's machine task**: a run of rung, shape (D18) and seed; steps saved and resumed;
     `certify_steps` running stage 4's certificate after each curriculum step and
     `stop_at_certified`; status and report.
   - Tests: the batched executor equals a one-input executor step for step; the routed gradients
     against finite differences of a relaxed run on a tiny network; and **R1's copy learned from
     inputs and outputs** by a tiny network on `pyramidon` and certified by stage 4: the method's
     smoke test, not a registered claim.
6. **E-R: rungs R1-R6 learned and certified** on `jebel`, one at a time (D14), **each from scratch
   on its own inputs and outputs** (D15: no trained subroutine carried into a later rung), each
   registered with its feasibility row (D18's capacity, the exclusion bound per expected sweep,
   the capacity row) and a 2-epoch probe for its budget; each rung reported in D17's tiers, its
   call structure as learned (the slots it uses against the reference program's subroutines).
   Waits for D12's gate.
7. **E-F1: R7 learned from `N` and its factors and certified in D17's tiers:** F1's experiment, the
   study's one multi-day run if its probe needs it (D14; the cost below says it likely will). The
   certified machine checked against the oracle to `3^12`. Findings.
8. **I1's compiled executor in the crate** (`indrajala-math-rust`, then "Bump rust/"); parity with
   the network executor; the certified R7 machine run to its budget: every `N` to `3^12`, strata
   to 24 digits, each batch within a day.
9. **Stretch: R8-R10**, rho with Miller-Rabin, each task a rung (D14, D15), run to 40 digits.
10. *Superseded by D15:* E-F2's arms are F1 itself (inputs and outputs only, the call structure
    learned).
11. **The report**; this workplan retired.

## Cost

Steps per number on the tape machine, estimated as about `10 n^2` steps per division (head
shuttling included), `n` the input's base-3 digits:

| `n` (bits) | trial division (R7), steps | rho (R9), steps | R7 on the network executor | R7 compiled |
| --- | --- | --- | --- | --- |
| 12 (19) | about 10^6 | about 10^5 | 30 s | 0.01 s |
| 20 (32) | about 2 x 10^8 | about 3 x 10^6 | 2 h | 2 s |
| 24 (38) | about 3 x 10^9 | about 10^7 | 1 day | 30 s |
| 40 (63) | about 6 x 10^13 | about 3 x 10^9 | - | not run (7 days a number); rho 30 s |

**Training (restated for D15-D16, 2026-10-14).** With inputs and outputs only, a training example
is a whole run of the machine on one input, forward and backward through every step, and `K`
more sampled runs for expert iteration: a pass costs `T (1 + K)` runs, `T` the inputs it trains
on. `T` is set by the capacity defence (D17; Feasibility's capacity row): the answers trained on
must hold more bits than a table of the network's `S` states (all its factors' codes, every level)
over 343 windows can store, at about 15.8 bits an entry (3 heads x 21 actions, x 6 control classes
with 4 call slots, D18). The rate: E1's 31,000 window evaluations a second forward, about `10^4`
steps a second with the backward on one worker unbatched; batching (D19) raises it by a factor
the probe measures. R7's average steps a number below `3^n` are computed (trial division's
divisibility tests times `10 n^2`); for `K` 8:

| R7's `S` | table, bits | `n` | inputs `N < 3^n` | `T` needed | average steps a run | a pass at `10^4` steps/s | at `10^5` steps/s |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 16 | 86,710 | 8 | 6,559 | 6,860: **more than exist** | 1.4 x 10^4 | - | - |
| 16 | 86,710 | 10 | 59,047 | 5,488 | 5.3 x 10^4 | 3.0 days | 7.3 h |
| 64 | 346,842 | 10 | 59,047 | 21,952 | 5.3 x 10^4 | 12 days | 29 h |

So **R7's training must reach inputs of 10 digits** (below that every `N` together holds fewer
bits than a 16-state table), **a pass there takes days unbatched**, and E-F1 fits the one
multi-day run only if batching reaches about `10^5` steps a second and a few passes suffice;
fewer states (`S`) shorten it in proportion. Stage 5's probe measures the batched rate before E-R
is registered; if it is far below `10^5`, the training executor's forward moves into the crate
(D19 (c)) before E-F1. The lower rungs are cheap by comparison (R1-R3's runs are `O(n)` steps,
R5-R6's `O(n^2)`), each within a day (D14), their rows computed in stages 2-3. Certification of a
slot of tens of codes over at most 343 windows is seconds, so it runs every step. The dominant
costs: engineering (stages 2-5, stage 5's learning layers the largest) and E-F1's training.

## Known limits

- **Not fast factoring.** Trial division is exponential in `n`, rho about `3^(n/4)`. The claim
  is correctness at every length, not speed.
- **The method's frontier is small.** Certified machines so far have 3 and 4 states; R1's
  controllers are near that, R5-R7's likely past it. The ladder may stop early; it then says
  where, and why (its counterexamples).
- **The cross-term is untested.** C9 learns factors with it off; every controller here needs it
  on (Feasibility), so R1 is its first test.
- *Superseded by D15:* "the call structure is given" and "with full traces a controller is close
  to a fitted table": the network now learns from inputs and outputs only.
- **Inputs-and-outputs learning has stalled short of this before** (Zaremba & Sutskever 2015;
  Kurach et al. 2016; Déletang et al. 2022): R1 may be where the ladder stops, and that is a
  finding.
- **Straight-through over thousands of steps is biased** (D16); expert iteration needs a
  near-random controller to succeed sometimes on the smallest inputs to start.
- **A learned subroutine may not be a sweep**, or may compute no function we can name: it then
  falls to a lower tier (D17), never to "certified".
- **The reference program's correctness is tested and argued, not proved** (D7).
- **A step budget bounds every run**; a correct program on a large input may run out of it.

## Timing and checks

Tier 0 for stages 1-7 and 9-10 (new code, no timed path changed); stage 8 changes the crate: its
own A/B is not needed (a new function, no existing timed path), but its parity tests are. If
stage 5's input gradient changes the window network's `learn` path rather than adding beside it,
that PR is tier 1; D19 (c), if taken, is a crate change with its parity tests. Every
stage that changes the package passes the golden run. Studies on `jebel` under the machine lock,
workers sized by a measured peak RSS, PL1 at 65 W.
