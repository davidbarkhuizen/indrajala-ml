# Workplan: a network-machine that factors integers

**Status: written 2026-10-13, at the owner's request ("a detailed high-level workplan covering the
missing infrastructure that would be required to allow the network-machine to truly perform a
prime-factor decomposition"). Decisions D1-D12 settled under the owner's standing delegation, each
with its options' pros and cons. Revised 2026-10-14 against
[experiment-design.md](experiment-design.md) and the work since (the claims workplan's E7-E8b and
D30-D34, the factored-state workplan): D1, D3, D4, D6, D9, D10, D11 and D12 revised, D13-D14
added, F1 restated as an existence claim, the stages and the cost redone. D13 then set by the
owner: **the controllers' state is factored, one factor per register**, R1 the cross-term's
first test. The owner granted the
study one multi-day run, the concluding one (D14). D11 revised by the owner (2026-10-14): F1 keeps
the calls given; F2's second arm learns them, factored per call level, in place of the inlined
flat controller (which multiplies control states as a single codebook multiplied division's). No stage started.** It follows
[factorisation-note.md](factorisation-note.md), which states why the window-recurrent network
can't factor and what a machine that can would need (P1-P5); this plan builds that machine. It
reuses the window-recurrent network, its target-machine certificate (the claims workplan's
stage 5, D17), the factored state's tuple certificate and harness
([factored-state-workplan.md](factored-state-workplan.md), stages 6-7), and the study rules of
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

> **F1.** Networks with a factored state, one factor per register of the program, trained on
> execution traces of a factoring program, one network per subroutine, each run as its
> subroutine's controller on a tape machine with an exact call stack, are each certified equal to
> that subroutine's controller; so the machine factors every integer
> correctly, at any length, in the program's number of steps.

F1 is an **existence** claim (experiment-design.md, section 2): one certified network per
controller proves it, however found; no rates, arms or controls. Running out of a rung's budget
records "not found within the budget", not a refutation, and stops the ladder there (D14).

And, as the research step beyond it:

> **F2.** The same holds with less than full traces per subroutine (inputs and outputs only), or
> for one network that learns the call structure itself, its state factored per register and per
> call level (bounded depth), never inlined (D11).

F1 is engineering plus E1's method, *if* the method reaches controllers of the rungs' sizes:
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

## Stages

1. **This workplan and a literature review** (docs only): neural program induction and
   verification (Reed & de Freitas 2016; Cai et al. 2017; Graves et al. 2014; Kaiser & Sutskever
   2016; Déletang et al. 2022; Graves 2016), each paper kept in the literature repository (`davidbarkhuizen/indrajala-literature`, cloned as `~/code/indrajala-literature/`) (Déletang 2022
   and Kaiser & Sutskever are there; the rest to fetch). The workplan is done (#708, revised).
2. **I1's tape model, the call stack and the reference executor; I2's DSL compiling both ways;
   rungs R1-R3** (tier 0): programs tested exhaustively; each table's minimal states computed and
   R1-R3's feasibility rows written.
3. **I3's oracle and catalogues; rungs R4-R7** (tier 0): R7 checked against the oracle to `3^12`
   (I6), and the flat table against the composed tables on the same; R4-R7's feasibility rows.
4. **I5: the controller certificate** over a reference table: windows restricted to the table's
   domain, reachability, calls and returns, the tables' export. Tests: a stand-in controller equal
   to R2's table certifies; faulty ones (a wrong write, a wrong move, a missed return, a wrong
   call) fail with a counterexample input the executor runs wrong.
5. **I1's network executor, I4's traces and the harness's controller task** (tier 0): a tiny
   network trained on R1 end to end on `pyramidon`.
6. **E-R: rungs R1-R6 trained and certified** on `jebel`, one at a time (D14), each registered
   with its feasibility row and probe; each certified controller run by the network executor on
   long inputs. Waits for D12's gate.
7. **E-F1: R7's controller trained and certified**, with R1-R6's certified controllers called:
   F1's experiment, the study's one multi-day run if its probe needs it (D14). The composed
   machine checked against the flat table to `3^12`. Findings.
8. **I1's compiled executor in the crate** (`indrajala-math-rust`, then "Bump rust/"); parity with
   the network executor; the certified R7 machine run to its budget: every `N` to `3^12`, strata
   to 24 digits, each batch within a day.
9. **Stretch: R8-R10**, rho with Miller-Rabin, each controller a rung (D14), run to 40 digits.
10. **E-F2: the arms** (D11). Findings.
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

Training: a rung per controller, each within a day of `jebel` (D14), set by its probe. For scale,
E8b's curriculum runs to 20 windows were estimated at 1-3 h each at `k` 1; a controller's cap of
`m + q - 1` steps at `k` 3 grows with its `q`, which is why D3 keeps controllers small. E-F1 may
take the study's one multi-day run. Certification of a controller of tens of codes over at most
343 windows is seconds, so it runs every step. The dominant costs are engineering (I1-I5, stages
2-5) and the long runs of stage 8.

## Known limits

- **Not fast factoring.** Trial division is exponential in `n`, rho about `3^(n/4)`. The claim
  is correctness at every length, not speed.
- **The method's frontier is small.** Certified machines so far have 3 and 4 states; R1's
  controllers are near that, R5-R7's likely past it. The ladder may stop early; it then says
  where, and why (its counterexamples).
- **The cross-term is untested.** C9 learns factors with it off; every controller here needs it
  on (Feasibility), so R1 is its first test.
- **The call structure is given** (D3): F1 learns each controller, not the program's
  decomposition; F2's learned-calls arm asks for that.
- **With full traces a controller is close to a fitted table.** What's learned beyond the table is
  the state: codes for each controller's reference states, found from its calls.
- **The reference program's correctness is tested and argued, not proved** (D7).
- **A step budget bounds every run**; a correct program on a large input may run out of it.

## Timing and checks

Tier 0 for stages 1-7 and 9-10 (new code, no timed path changed); stage 8 changes the crate: its
own A/B is not needed (a new function, no existing timed path), but its parity tests are. Every
stage that changes the package passes the golden run. Studies on `jebel` under the machine lock,
workers sized by a measured peak RSS, PL1 at 65 W.
