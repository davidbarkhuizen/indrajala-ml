# Note: what a network-machine would need to factor integers

**Status: a note, 2026-10-13, written at the owner's question: "what properties would we require
from a network-machine in order for it to be able to perform prime factorization, in theory?".
Nothing here is planned work.** It places the window-recurrent study's networks
([window-recurrent-report.md](window-recurrent-report.md)) against a task far
outside what they can compute, so that the class they *can* compute is stated precisely.

## The short answer

The snapped window-recurrent network can't factor at any size, and no change to its training
alters that. Its snapped state makes it a finite-state transducer. E1's certificate rests on that
property, and it is also the limit: a finite-state machine computes only regular (rational,
one-pass) functions, and factoring isn't one. A machine that could factor in theory needs five
properties the snapped network lacks: memory that grows with the input, computation time that
grows faster than the input, exact operations on that memory, repeated access to the input and
intermediate results, and output of varying length with a halt. With all five it is a Turing
machine. Even then, its cost is bounded below by the best factoring algorithm it can carry out,
and the best known classical algorithms are super-polynomial in the number of digits.

## Why the snapped network can't

**What it is.** A window's work is a function of the state it receives and its `k` tokens. A
snapped state is one of `m` codes. So the network, at every length, is a Mealy machine of at most
`m` states over `T^k` window symbols (`T` the task's tokens: 27 for addition, 9 for modular
division). Whatever it computes, a machine of `m` states computes.

**A counting argument.** Say a one-pass machine reads `N`'s `n` digits and gives `N`'s
factorisation when the input ends. That output can depend only on the final state and the last
window: at most `m * T^k` different outputs. But the `n`-digit integers have about `3^n` different
factorisations (each prime of `n` digits is its own, and there are about `3^n / (n ln 3)` of
them). So `m * T^k >= 3^n / (n ln 3)`, so `log2 m` must grow like `n log2 3`. **No fixed `m`
suffices: the state must grow linearly with the input.** Outputs given before the input ends don't
help either. Read most significant first, `N`'s factorisation depends on its last digit (`N` and
`N + 1` share no factor). Read least significant first, it depends on the top digit.

**The same from the language side.** A finite automaton can't even recognise the primes. The base-`b` representations of
the primes are not a regular language in any base (Hartmanis & Shank 1968). Factoring is harder
than deciding primality: a factoriser decides primality by checking whether one factor comes back.

**What it can do that looks like factoring.** "Does the fixed prime `p` divide `N`?" is
`N mod p`, a `p`-state automaton read most significant first (E7's modular division with `b`
= 1 gives it at the last column). So a snapped network can learn, and E1's certificate can prove at every length,
**trial division by a fixed, finite set of primes**: it finds the small factors of any `N`. That
isn't factoring: the set of primes is fixed when the machine is built, so a large prime factor
goes unseen. Even *verifying* a claimed factorisation `N = p * q` is out of reach for one pass,
because long multiplication isn't finite-state (its carries grow with the length; Kaiser &
Sutskever 2016, after Blumensath & Grädel 2000).

## The properties a factoring machine needs

Each property is necessary, for the reason given; together they are sufficient in principle.

1. **Memory that grows with the input (P1).** By the counting argument, at least `n log2 3` bits,
   roughly the input's own size. Any working algorithm needs more for its intermediates: a
   candidate divisor, a remainder, a quotient, each up to `n` digits. A fixed-width snapped state
   fails P1. A continuous state meets it only in principle: rational-weight RNNs with unbounded
   precision are Turing complete (Siegelmann & Sontag 1995), but they hold `n` digits in the
   precision of a few numbers, which training and floating point don't preserve.
2. **Computation time that grows faster than the input (P2).** A network that spends a fixed
   amount of work per token does `O(n)` work in all. A linear-time classical factoring algorithm
   is not known; the best known, the general number field sieve, is sub-exponential in `n`
   (about `exp(c n^(1/3) (log n)^(2/3))`), and RSA's security rests on there being nothing much
   better. So the machine must choose how many steps to take, a number that grows with `n`:
   adaptive computation time (Graves 2016), or a fixed but super-linear number of steps such as
   the Neural GPU's (Kaiser & Sutskever 2016). Déletang et al. 2022 make the same point for
   multiplication: memory-augmented networks perform one step per input token, so superlinear
   tasks also need more steps per token. In their tests, no architecture they tried learned
   binary multiplication, not even an RNN with a tape.
3. **Exact operations on that memory (P3).** Division by a candidate must be exact for every digit
   at every length. A single carry wrong at column 900 gives a wrong remainder, and a wrong
   remainder gives a wrong factor. This is where the snap matters. A discrete controller is exact
   by construction, and can be certified exact the way E1 certifies the adder; a continuous
   controller drifts (C2's prediction, E2's measurement).
4. **Repeated, addressable access to the input and intermediates (P4).** Trial division reads
   `N` once per candidate divisor; the sieve methods read and write large tables. A one-pass
   reader, the window network as it stands, reads each digit once. The machine needs a tape or
   random-access memory with read and write heads (Neural Turing Machines, Graves et al. 2014;
   the Tape-RNN of Déletang et al. 2022), and the input must be on it.
5. **Output of varying length, with a halt (P5).** `N` has between 1 and about `n log2 3`
   prime factors (counting repeats), so the machine must write a variable-length answer and
   signal that it has finished.

**Sufficient, in principle:** a finite controller (P3) driving an unbounded tape (P1, P4) for as
many steps as it chooses (P2), writing its answer and halting (P5). That is a Turing machine,
and a Turing machine can run any factoring algorithm. The smallest such algorithm, trial division
by every `d` up to `sqrt(N)`, needs a controller of a few dozen states (compare, subtract, shift,
increment, halt) and `O(3^(n/2) n^2)` steps: exponential in the number of digits.

**What it costs, whatever the design.** P1-P5 make factoring *possible*; they don't make it
*fast*. A learned machine runs at the cost of the algorithm it has learned. If it learns trial
division, the cost is exponential. If it learns the number field sieve, it is sub-exponential. If
it learned something polynomial, that would be a new classical factoring algorithm, a result in
complexity theory rather than in machine learning. On a quantum computer Shor's algorithm (1994)
factors in polynomial time, a different machine model altogether.

## What this says about the study's networks

- **They compute exactly the regular functions.** Given the snap, the class is finite-state
  transductions: addition, divisibility by a fixed `p`, modular division, any function a
  finite-state reader computes in one pass. That is the class C7 claims and E1's certificate
  covers.
- **The certificate's idea extends to the controller of a machine with P1-P5.** If the controller
  is a snapped network and the tape is exact by construction, the controller is a finite machine,
  and a certificate in E1's manner can prove its transition table equal to a given Turing
  machine's (for each state and symbol read: the symbol written, the move, the next state). That
  would be a certified learned algorithm whose correctness doesn't depend on the input's length.
  Learning such a controller by gradient descent is the open part. Déletang et al. 2022 found
  that even tape-augmented networks fail at binary multiplication.
- **A feasible step towards factoring, within the class:** a certified small-prime sieve (the
  divisibility automata for 2, 3, 5, 7, ... as one product machine, `2 * 3 * 5 * 7 = 210`
  states, read once). It finds every small factor of any `N`, at any length, with a proof.

## References

- J. Hartmanis and H. Shank, 1968. On the recognition of primes by automata. *Journal of the ACM*
  15(3).
- H. T. Siegelmann and E. D. Sontag, 1995. On the computational power of neural nets. *Journal of
  Computer and System Sciences* 50(1).
- A. Graves, G. Wayne and I. Danihelka, 2014. Neural Turing machines. arXiv:1410.5401.
- A. Graves, 2016. Adaptive computation time for recurrent neural networks. arXiv:1603.08983.
- Ł. Kaiser and I. Sutskever, 2016. Neural GPUs learn algorithms. ICLR (`indrajala-literature/kaiser-2015-neural-gpu`).
- A. Blumensath and E. Grädel, 2000. Automatic structures. LICS (cited by Kaiser & Sutskever for
  long multiplication's representations not being regular).
- G. Déletang et al., 2022. Neural networks and the Chomsky hierarchy. arXiv:2207.02098
  (`indrajala-literature/deletang-2022-neural-networks-chomsky-hierarchy`).
- P. W. Shor, 1994. Algorithms for quantum computation: discrete logarithms and factoring. FOCS.
