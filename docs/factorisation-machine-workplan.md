# Workplan: a network-machine that factors integers

**Status: written 2026-10-13, at the owner's request ("a detailed high-level workplan covering the
missing infrastructure that would be required to allow the network-machine to truly perform a
prime-factor decomposition"). Decisions D1-D12 settled under the owner's standing delegation, each
with its options' pros and cons. No stage started. It follows
[factorisation-note.md](factorisation-note.md), which states why the window-recurrent network
can't factor and what a machine that can would need (P1-P5); this plan builds that machine.** It
starts after the window-recurrent claims workplan
([window-recurrent-claims-workplan.md](window-recurrent-claims-workplan.md)) and reuses its
network, its certificate and its tasks (stage 5).

## Why

The window-recurrent network, its state snapped, is a finite-state transducer, and E1 showed that
its machine can be read off and proved correct at every length. Finite-state is also its limit:
it can't factor (the note's counting argument). The note's conclusion is that a **snapped
controller driving an exact, unbounded tape for as many steps as it chooses** has everything
factoring needs, and that the controller, being finite, could still be certified. The result would
be a learned program, proved equal to a reference algorithm on every input, that factors any
integer. Its cost would be that algorithm's.

That is the claim this plan works towards:

> **F1.** A network trained on execution traces of a factoring algorithm, run as the controller
> of a tape machine, is certified equal to that algorithm's controller; so it factors every
> integer correctly, at any length, in the algorithm's number of steps.

and, as the research step beyond it:

> **F2.** The same holds for a controller trained on less than full traces (the call structure
> only, or inputs and outputs only).

F1 is engineering plus E1's method; F2 is open (Déletang et al. 2022 found no architecture that
learned even binary multiplication from inputs and outputs).

**What this is not.** Not fast factoring: the machine runs trial division (exponential in the
digits), then Pollard's rho (about `N^(1/4)`). Factoring fast would be a result in number theory,
not in learning. The demonstration range is set by the step budget (the cost table below):
numbers to about 24 base-3 digits (38 bits) by trial division, about 40 (63 bits) by rho.

## The machine

A **tape machine** (D1): three tapes, each a row of cells holding one symbol, with one head each.

- **Tape 0, the input:** `N`'s digits, most significant first, between end markers; read-only.
- **Tape 1, the work tape:** the candidate divisor, the running remainder, the quotient, scratch.
- **Tape 2, the output:** the prime factors written as they are found, separated.

Symbols (D2): the base-3 digits 0-2, a blank, a separator and two end markers: 7 symbols.

A **step**: the controller reads the three symbols under the heads and its state; it writes a
symbol under each head, moves each head left, right or not, and moves to its next state, or halts.
The tapes and heads are exact by construction: plain memory, not learned (P1, P3, P4). The
machine runs until the controller halts or a step budget runs out (P2, P5).

**The controller is the window-recurrent network as it is** (D4): a window of `k = 3` columns, one
per head, each column's token the symbol under that head (7 tokens), the state a snapped code, and
each column's output that head's action, a write symbol and a move (7 x 3 = 21 classes, plus a
halt class on column 2). The window network already gives `k` outputs from `k` tokens and a state,
and hands the snapped state on; what changes is where the next window comes from: from the tapes
after the actions, not from a fixed row of tokens. So the controller is a finite machine of `m`
states over `7^3 = 343` windows, **and E1's certificate applies to it unchanged**: enumerate every
reachable code against every window, map codes to the reference controller's states (`phi`), and
check every action and next state. A certified controller **is** the reference controller; the
tape machine around it is exact; so the network-machine computes what the reference program
computes, on every input.

What it then factors correctly depends only on the reference program, which is ordinary code,
tested and argued as any program is (D7).

## The missing infrastructure

Eight pieces, each a stage or part of one. None exists in the package now.

### I1. The tape machine and its executors

- **A tape model:** tapes growing on demand, heads, symbols, a step function taking actions to the
  next configuration; configurations hashable (for traces and tests).
- **Three executors** of a controller (D5):
  - the **reference executor**, running a reference program's transition table (I2), in Python;
  - the **network executor**, running a network as the controller step by step (`window_step`
    with the window built from the heads' symbols), on either backend: slow (about 31,000 window
    evaluations a second, E1's rate), for validation;
  - the **compiled executor**, running a certified network's extracted table (its codes, actions
    and next codes, from the certificate) in the crate, Rust: about 10^8 steps a second, the
    executor for the long runs.
- **The step budget and halting:** a budget a run, a function of `n` given by the reference
  program's bound; a run out of budget is a failure, recorded with its configuration.

### I2. Reference programs: the algorithm ladder

A small language for tape-machine controllers (D3): named states, and for each state and the
symbols under the heads, the writes, moves and next state; subroutines by composition (a
subroutine's states copied in with its return state bound), so the result is one flat table, the
target machine the certificate checks against. Each program gets its own tests: exhaustive at
short lengths against Python's integers, sampled at long ones.

The ladder (D6), each rung a program, each later one built from earlier ones:

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

Rungs R1-R7 are the plan's core; R8-R10 its stretch (stage 9).

### I3. Number-theory data and the factorisation catalogue

- An **oracle** in Python: factorisation by trial division and Pollard's rho, Miller-Rabin; tested
  against each other and exhaustively to `3^12`. No new dependency (D8).
- **A catalogue of properties** for factoring, as addition's and division's catalogues: every `N`
  at short lengths; sampled at long ones; and strata that stress the algorithm: primes (the
  longest trial), prime powers (repeated division), semiprimes `pq` with balanced factors (the
  hardest for trial division and rho), smooth numbers (many small factors), squares, Carmichael
  numbers (for R10), and the edge cases 0, 1, 2 and 3. Each case's result checked whole: the
  factors written, in order, their product `N`, each prime.
- **Per-rung catalogues** for R1-R6 and R8 (each rung's inputs and its answer), so each rung is
  trained and certified alone before it's composed.

### I4. Traces and the training data

- **Trace generation:** a reference program run on an input gives its trace: for every step, the
  symbols under the heads (the window), the reference state, and the actions. A trace is exactly a
  window-recurrent example: windows in, an action per column out, the state carried. Traces of
  millions of steps are cut into **segments** that start from a recorded reference state; the
  network's state at a segment's start is the code the certificate maps to that state
  (teacher-forced states), which needs the network's codes to be addressed by reference state:
  either the segment starts from the start state only (short inputs) or from a state the network
  has been trained to reach (D9).
- **A curriculum over input length**, as the adder's over window counts: rung by rung, short
  inputs first; the network and its optimiser carried (W25).
- **Action labels per head** with a halt class; a loss weighting so the halt (one step a run)
  isn't swamped (D10).

### I5. The controller certificate

E1's certificate over a target machine (stage 5's, D17) already does nearly everything: the
controller's windows are the heads' symbols and its outputs the actions. What's added:

- **Windows that can't occur are left out:** the input tape's head never reads a work symbol, so
  the enumeration is over the windows the reference program can see from each state (its table's
  domain), not all 343. A window outside it is "don't care" in the reference and must be shown
  unreachable from the start (a reachability pass over configurations' symbol triples, as the
  certificate's breadth-first search over codes).
- **The halt** as an output class and the reference's halting states mapped.
- **The certified table's export**, for the compiled executor (I1): codes, windows, actions, next
  codes; and a check that the compiled executor and the network executor agree step for step on
  sampled inputs.

### I6. Correctness of the reference programs

The certificate reduces the network's correctness to the reference program's. That needs its own
evidence (D7): exhaustive runs against the oracle for every `N < 3^12` (531,441 numbers), sampled
runs to 40 digits per stratum, and for R7 a written argument (the loop invariant: the factors
written times the remaining `N` is the input, every factor written is prime, `d` only grows); a
mechanised proof (Lean) is a later option, not this plan.

### I7. Measurement and the harness

- A **study harness** for the tape machine, as the window study's: runs of rung, `m`, seed and
  arm (trace supervision level, F2); each step saved and resumed; checks, tests and the
  certificate per step; status and report.
- **Measures:** certified or not; steps per input against the reference's (equal if certified);
  wall time per executor; the largest `N` factored in a budget; codes used against the reference's
  states.
- On `jebel`, sized by memory, PL1 65 W, as every study.

### I8. Package and persistence

- The tape machine and its executors in `indrajala_ml/machines/` (new): numpy and Python
  reference executors; the compiled executor in the crate (`indrajala-math-rust` first, then a
  "Bump rust/" PR); parity tests between them.
- The controller network: the window-recurrent network, unchanged (format 2, its golden entries).
  A new class only if D4's reuse fails (a halt head or per-head classes the window network can't
  express).
- Certified tables saved beside the network (format 2's JSON or a sidecar), with the
  certificate's sha256.

## Decisions

Settled under the owner's standing delegation, with each option's pros and cons:

- **D1. The machine model. Settled: (a).**
  - (a) *Chosen.* A multi-tape Turing machine, three tapes, one head each. Pros: the controller
    is a finite table over (state, symbols under heads), exactly what E1's certificate checks;
    the tapes are trivially exact; well understood (time bounds, simulation). Cons: head
    shuttling costs steps (a long division is `O(n^2)` moves at best).
  - (b) A register machine (registers of unbounded integers, increment, decrement, test for zero).
    Pros: few steps per operation. Cons: the controller's view is "zero or not" per register, so
    the arithmetic lives in the machine, not the network; nothing learned.
  - (c) A hierarchical program machine (Neural Programmer-Interpreters, Reed & de Freitas 2016):
    the controller calls subprograms with a stack. Pros: compositional, recursion. Cons: a call
    stack is more infrastructure, and its certificate needs the stack's contents bounded or
    abstracted (Cai et al. 2017 verify recursion on a finite set of configurations): a later
    plan.
  - (d) A soft, differentiable memory (Neural Turing Machines). Cons: inexact; drift (C2): the
    opposite of what the certificate needs.
- **D2. The base and the alphabet. Settled: (a).**
  - (a) *Chosen.* Base 3, as the studies: digits 0-2, blank, separator, two end markers (7
    symbols). Pros: E1's and E7's automata reusable as subroutines and their certificates
    comparable. Cons: binary would need fewer windows (`6^3` against `7^3`).
  - (b) Base 2. Pros: the literature's (Kaiser & Sutskever, Déletang et al.); smaller tables.
    Cons: a break with the studies' tasks.
- **D3. How reference programs are written. Settled: (a).**
  - (a) *Chosen.* A small Python DSL compiled to a flat table, subroutines inlined. Pros: the
    table is the target machine as stage 5's certificate takes it; readable programs; testable
    alone. Cons: inlining multiplies states (a subroutine used twice is copied twice).
  - (b) Hand-written tables. Cons: unreadable past R2.
  - (c) Subroutine calls kept (a call stack). Pros: fewer states. Cons: D1 (c)'s stack.
- **D4. The controller. Settled: (a).**
  - (a) *Chosen.* The window-recurrent network, `k` = 3 heads, a column per head, unchanged, its
    snapped state the controller's state. Pros: no package change to start; the certificate,
    the harness and the training all carried over; the claims workplan's findings (more codes
    help, the curriculum isn't needed) tested on a new task for free. Cons: a column's output is
    one head's action, so a decision coupling the heads (e.g. "halt") goes on one column, and
    per-column classes are the product write x move.
  - (b) A new controller class (one output vector, a head for each part of the action). Pros:
    the natural shape. Cons: a new network in three implementations, format 2, fixtures and golden
    entries before anything is learned.
- **D5. Executors. Settled: (a).**
  - (a) *Chosen.* Reference (Python), network (either backend, validation) and compiled (Rust, the
    certified table). Pros: long runs at 10^8 steps a second, possible only because the
    certificate makes the table the network's exact behaviour; the network executor checks the
    compiled one. Cons: three executors kept in parity.
  - (b) The network executor only. Cons: 31,000 steps a second: R7 at 20 digits takes hours a
    number.
- **D6. The algorithms. Settled: (a).**
  - (a) *Chosen.* Trial division (R7) as F1's target, Pollard's rho with Miller-Rabin (R9, R10) as
    the stretch. Pros: R7 is the simplest correct factoriser, its controller small (an estimated
    100-300 states after inlining); rho reaches 63 bits in the step budget. Cons: neither is fast;
    rho's random restarts need a fixed sequence of `c` (deterministic, so certifiable).
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
- **D9. Training on long traces. Settled: (b).**
  - (a) Whole traces, short inputs only. Pros: exact state from the start. Cons: R7's traces on
    inputs long enough to exercise every branch are 10^5-10^6 steps.
  - (b) *Chosen.* Segments: traces cut into segments of a few hundred steps, each starting from the
    start state on a short input, or, once a network's codes map to reference states (a partial
    certificate), from a mid-run state given as that code. Pros: bounded examples; every branch
    reached. Cons: needs the partial certificate's map to seed states, so the curriculum starts
    from whole short traces.
  - (c) Per-step examples, the state given as the reference state's one-hot. Pros: simplest. Cons:
    the network doesn't learn its own state; nothing recurrent learned (it's a lookup table fit).
- **D10. The halt and the loss. Settled: (a).** (a) *Chosen.* A halt class on column 2, the loss
  weighted so each run's halt counts as much as its other steps together. Pros: the network
  learns to stop. Cons: a weight to choose. (b) The executor halts on a reference state the
  certificate names. Cons: the network doesn't decide when to stop: not P5.
- **D11. F2's supervision levels. Settled: (a).** (a) *Chosen.* Three arms, after F1: full traces;
  the call structure only (which subroutine at each step, not its actions); inputs and outputs
  only. Pros: measures how much supervision the method needs. Cons: the last arm is likely to fail
  (Déletang et al.); reported as it comes out.
- **D12. Where the plan sits. Settled: (a).** (a) *Chosen.* After the claims workplan's report
  (its stage 10); this plan's stage 1 can start earlier, being independent. Pros: the claims
  paper isn't delayed; stage 5's target-machine certificate is in place. Cons: none.

## Stages

1. **This workplan and a literature review** (docs only): neural program induction and
   verification (Reed & de Freitas 2016; Cai et al. 2017; Graves et al. 2014; Kaiser & Sutskever
   2016; Déletang et al. 2022; Graves 2016), each paper kept in `~/code/papers/`.
2. **I1's tape model and reference executor; I2's DSL and rungs R1-R3** (tier 0): programs
   tested exhaustively.
3. **I3's oracle and catalogues; rungs R4-R7** (tier 0): R7 checked against the oracle to `3^12`
   (I6).
4. **I5: the controller certificate** over a reference table: windows restricted to the table's
   domain, reachability, the halt, the table's export. Tests: a stand-in controller equal to R2's
   table certifies; faulty ones (a wrong write, a wrong move, a missed halt) fail with a
   counterexample input the executor runs wrong.
5. **I1's network executor and I4's traces and segments; I7's harness** (tier 0): a tiny
   network trained on R2 end to end.
6. **E-R: rungs R1-R6 trained and certified** on `jebel` (`m` and seeds as E3's design), each
   certified rung's network run by the network executor on long inputs.
7. **E-F1: R7 trained and certified**: F1's experiment. Findings.
8. **I1's compiled executor in the crate** (`indrajala-math-rust`, then "Bump rust/"); parity with
   the network executor; the certified R7 run to its budget: every `N` to `3^12`, strata to 24
   digits.
9. **Stretch: R8-R10**, rho with Miller-Rabin, certified and run to 40 digits.
10. **E-F2: the supervision arms** (D11). Findings.
11. **The report**; this workplan retired.

## Cost

Steps per number on the tape machine, estimated as about `10 n^2` steps per division (head
shuttling included), `n` the input's base-3 digits:

| `n` (bits) | trial division (R7), steps | rho (R9), steps | R7 on the network executor | R7 compiled |
| --- | --- | --- | --- | --- |
| 12 (19) | about 10^6 | about 10^5 | 30 s | 0.01 s |
| 20 (32) | about 2 x 10^8 | about 3 x 10^6 | 2 h | 2 s |
| 24 (38) | about 3 x 10^9 | about 10^7 | 1 day | 30 s |
| 40 (63) | about 6 x 10^13 | about 3 x 10^9 | - | 7 days; rho 30 s |

Training: rung by rung, each about a matrix's cost (hours on `jebel`); certification of a
controller of up to a few hundred codes over at most 343 windows is seconds. The dominant costs
are engineering (I1-I5, stages 2-5) and the long runs of stage 8.

## Known limits

- **Not fast factoring.** Trial division is exponential in `n`, rho about `3^(n/4)`. The claim
  is correctness at every length, not speed.
- **F1 with full traces is close to fitting a table.** What's learned beyond the table is the
  state: the network must find codes for 100-300 reference states from traces. The interest is in
  that, in the certificate, and in F2.
- **The reference program's correctness is tested and argued, not proved** (D7).
- **A step budget bounds every run**; a correct program on a large input may run out of it.

## Timing and checks

Tier 0 for stages 1-7 and 9-10 (new code, no timed path changed); stage 8 changes the crate: its
own A/B is not needed (a new function, no existing timed path), but its parity tests are. Every
stage that changes the package passes the golden run. Studies on `jebel` under the machine lock,
workers sized by memory, PL1 at 65 W.
