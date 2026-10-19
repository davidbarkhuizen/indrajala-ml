# The goal: discovering finite-state machines from behaviour and known structure

**Written 2026-10-19 at the owner's request: "it has been the ultimate goal of this work from the
inception".** This is the destination, not a workplan. Every study and workplan here serves it,
and each should be able to say how. Its stages (below) become workplans when the work reaches
them. The terms follow the papers' (the literature review's "Terms";
`~/code/indrajala-papers/contribution.md`):
- the **target machine** is the machine to be found;
- an **approximation** of it is a homomorphic image (a surjective morphism onto a smaller machine);
- a **cover** is a machine that emulates it (an injective morphism into, for example, a product of
  its approximations);
- **model** always means the network, never a machine.

## The goal, stated

A system behaves as a finite-state transducer (a Mealy machine), the **target machine** `T`, which
is unknown. What is known:

1. **Behaviour:** observed input words and the outputs `T` gives, sampled, and possibly queries:
   the outputs `T` gives on words we choose.
2. **Structure:** some of `T`'s properties, as machines or bounds:
   - known **approximations** `A_1, ..., A_k` (machines `T` maps onto, the map itself unknown);
   - a **bound** `m` on `T`'s states;
   - possibly symmetries or the input alphabet's meaning.

**Find a machine `M` that provably accounts for all of it:**
- `M` reproduces every observed output;
- each `A_i` is an approximation of `M`;
- `M` is minimal among the machines found;
- given queries and the bound `m`, `M` is **proved equivalent to `T`** (Chow's theorem, below).

The machine is found by training a network on the behaviour and the approximations' outputs, and
reading the machine off the network exactly. **A network that reliably learns a target machine
from its approximations is then a tool for discovering machines nobody wrote down.** It finds a
novel machine that models a system's behaviour, consistent with its known structure, checked
exactly rather than estimated.

## What the work so far established towards it

Every result below is in the papers or the probe script's findings. Each is a piece the goal needs.

| piece | what we have | where |
| --- | --- | --- |
| reading a machine off a network exactly | the snapped state makes the trained network a finite machine; the certificate walks it from every reachable code, outputs and states, near ties branched | P1; `window_recurrent_certificate` |
| extrapolation | the snap is what extrapolates (0/20 continuous against 8/20 and 20/20 snapped at 1,024) | P1 (2c) |
| no memorising | training to the exclusion bound `m + q - 1`; short of it, networks learn larger, wrong machines | P1 (2d); experiment-design.md |
| keeping a solution | stop at the first certified step; training past it destroys it | P1 (3a, 3b) |
| **learning from approximations** | training on the outputs of the target's approximations beside its own made product machines learnable (division: 1/20 to 13-15/20); **one approximation suffices** (the sum alone, 10/20, where an equally informative non-approximation, the product, gives 0/20) | P2 (5, 5b) |
| scale | certified transducers of 25, 49, 121 and 169 states (division `p` 5-13), 120 states (S5, move-to-front 5), from answers alone | P2; W1; B2 |
| breadth | permutation groups (S3-S5), move-to-front and random machines beside arithmetic | the general algebras |
| what stops it | three bottlenecks, one shared measure (one code, one state), remedies in progress | bottlenecks.md; one-code-one-state-workplan.md |

Result 5b is the seed of the goal. An approximation's outputs carry structure the network can use
even though they don't determine the target's state. That is exactly the kind of known property a
discovery problem supplies.

## How the method carries over

- **Training signal:** the target's observed outputs (one answer slot) and each known
  approximation's outputs (one more slot each), as the modular family trains division beside its
  sum and difference. Inputs and answers only: no states or traces, as now.
- **The exclusion bound** needs the target's state count, unknown here. The known bound `m`
  stands in for it: train to `m_codes + m - 1`, which excludes memorising for any target within
  the bound. **A state bound is therefore a required known property**, not an optional one.
- **The architecture** is the general one this work is converging on (the window the
  one-code-one-state plan adopts, width 32 pending C0; bottlenecks.md). Nothing in it is chosen per
  task, which is what discovery needs: the target is unknown, so the architecture can't be fitted
  to it.

## Verifying without the target: what can be proved

Today's certificate checks the learned machine against a known target. In discovery the target is
unknown, but four exact checks remain, each built from what the certificate already does:

1. **Consistency with the behaviour.** The learned machine `M`, read off and minimised, reproduces
   every observed word's outputs. A walk over the observations' prefix tree, exact on every
   observed prefix.
2. **Each known approximation is an approximation of `M`.** There is a surjective map from `M`'s
   states onto `A_i`'s that commutes with every transition and output `A_i` has. This is decidable
   by walking `M` beside `A_i` from both starts, as `machine_reading.product` walks a network
   beside its target. The map exists iff no state of `M` is paired with two states of `A_i`
   (`A_i` minimal). That is the same "one code, one state" property the bottleneck work measures,
   here between `M` and an approximation.
3. **Minimality.** `M` minimised is exact; its state count is a fact, not an estimate.
4. **Equivalence to `T`, given queries and the bound (Chow 1978).** A minimal `M` with `n` states
   is equivalent to a target with at most `m` states **if and only if** they agree on the test set
   `P · Z`:
   - `P` is a transition cover of `M`;
   - `Z = W ∪ X W ∪ ... ∪ X^(m-n) W`, with `X` the input alphabet;
   - `W` is a characterization set of `M`.

   The theorem is in Chow's appendix, under its assumptions: the same input alphabet, both
   deterministic and completely specified. The test set comes from `M` alone; only its outputs need
   `T`. **So with query access to the system and a state bound, a discovered machine can be proved
   equal to the unknown target**: from behaviour alone, not estimated from a sample. Its size grows
   as `|X|^(m-n)`, so the bound must be tight for large alphabets.

**What can't be proved without queries:** from a fixed sample alone, many machines fit every
observation. The claim is then "consistent with every observation and every known approximation,
and minimal among those found". That is the claim any learner from a finite sample makes. Gold's
identification in the limit and RPNI's characteristic samples say when a sample suffices; minimal
consistent identification is NP-hard in general (Gold 1978).

## Identifiability: what the known structure can and can't do

- **Approximations alone don't determine `T`.** Many machines share the same approximations. 5b's
  sum is an approximation whose outputs don't determine division's state. Behaviour of `T` itself
  is always needed.
- **If the known approximations together cover `T`** (their product emulates it), `T`'s state is
  a function of theirs. What remains is `T`'s output function on that state, a far smaller
  problem. Division's sum and difference counters cover it (odd `p`), which is why training on
  the full set of approximations so closely resembles supervising the state (Dan et al. 2022, the
  closest prior work).
- **Between the two**, the approximations narrow the search to machines that map onto them. The
  network uses that as structure in its training signal, as 5b shows it can.

## Why a network, when classical learners exist

Classical learners find finite machines from symbolic data far more efficiently: L\* (Angluin 1987)
with queries, RPNI and OSTIA (Oncina, García & Vidal 1992) from samples, SAT-based minimal
identification. The papers say so ("Why a network"). The network's place in discovery is where
those learners can't go:
- **inputs that aren't clean symbols** (signals, images, noisy logs): the network learns the map
  from raw input to symbols jointly with the machine, and the certificate still applies to the
  machine it reads off;
- **known structure as a training signal**: approximations supplied as extra outputs, which
  classical learners don't take;
- **noise in the observed outputs**, which breaks exact consistency-based learners.

On clean symbolic data the classical learners are the baseline any discovery claim must be
compared against.

## What stands between here and there

1. **Reliability** (now): the three bottlenecks. Discovery needs a network that finds the machine
   in most seeds, not one in six (bottlenecks.md; one-code-one-state-workplan.md;
   commitment-control-workplan.md).
2. **Scale:** certified at 120-169 states today. Real systems may need more, through more factors
   and their coordinates (the factored state), not a flat codebook.
3. **Verification tools without the target:** the four checks above as repo tools: the
   observation walk, the approximation check, minimisation, and Chow's test set, generated from
   `M` and run against a black box.
4. **Learning from approximations without the target's full outputs:** partial or sampled target
   outputs beside complete approximation outputs.
5. **Raw inputs:** a front end mapping non-symbolic input to the window's tokens, trained jointly.
6. **Noise:** observed outputs with errors; the certificate's claim becomes "consistent except on
   the observations it rejects", stated exactly.

## The route, in stages

Each stage is registered as its own workplan when the work reaches it, with claims before data.

- **A. Reliability** (in progress): the bottleneck remedies, until a general architecture
  certifies most seeds on every backdrop (division, the permutation groups, move-to-front, random
  machines).
- **B. The verification tools:** the four checks above, each tested on known targets (where the
  ordinary certificate gives the answer to compare).
- **C. Hidden targets:** known machines, hidden from the method. Train on partial behaviour and
  chosen approximations, extract `M`, and only then compare it with the hidden target. Measure
  recovery against the amount of behaviour, which approximations are given (none, one, a full
  cover), and the state bound's slack. The honest test of discovery, since the truth is known
  afterwards.
- **D. Queries:** the method proposes words (Chow's test set from its current `M`, or words where
  its candidates disagree) and the hidden target answers them. Recovery against query count,
  compared with L\*.
- **E. Raw inputs and noise:** stage C and D's targets with their inputs encoded non-symbolically
  and their outputs perturbed.
- **F. A real system:** a system whose machine is unknown but whose properties are (a protocol, a
  controller, a parser, a log), with its known approximations and a state bound. The discovered
  machine is reported with its consistency, approximation and minimality checks and, given
  queries, Chow's equivalence.

## Relation to the papers

P1 (the method) and P2 (product machines from approximations) are this goal's first two steps:
reading a machine off a network exactly, and learning one from its target's approximations. A
discovery paper would follow stage C: recovering hidden machines from behaviour and known
structure, each recovery proved.

Related: [bottlenecks.md](bottlenecks.md) (where the work stands),
[one-code-one-state-workplan.md](one-code-one-state-workplan.md),
[factorisation-machine-workplan.md](factorisation-machine-workplan.md) (a machine that discovers its
own decomposition), [window-recurrent-literature.md](window-recurrent-literature.md) (the terms,
Dan et al. 2022, the classical learners).
