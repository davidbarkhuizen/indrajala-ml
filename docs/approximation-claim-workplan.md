# Workplan: the approximation claim (does one approximation help, or any extra output?)

**Status: written and registered 2026-10-16, before any data, at the owner's request ("a - run it
(when we can)"; the contribution review of 2026-10-15 in `~/code/indrajala-papers`). Decisions
D1-D6 settled under the owner's delegation, each with its options' pros and cons. Stage 1 (the
build) in the same PR as this plan; stage 2 (the runs) when the machines are free.**

## Why

Division's stage 5 (docs/division-workplan.md) showed that training on the cover of division's
target machine (the quotient with the sum and the difference) certifies far more often than the
quotient alone: 13 of 20 against 1 of 20, Fisher one-sided p 6.9e-5. P2 also reports, from 2
seeds and no control, that **one approximation without a cover** (the quotient with the sum
alone) suffices. The contribution review calls this potentially the most original finding in
either paper, on the weakest evidence. Stage 5 can't separate two readings of it:

- **the approximation helps:** the sum is a homomorphic image of the target machine (a
  congruence: its state is a function of the target's, and its transitions respect that), so
  training on it shapes the state toward the target's structure;
- **any extra output helps:** a second answer per window, whatever its relation to the state,
  gives the state more gradient.

The product `AB` mod `p` separates them. It is an extra output read off the same state, and
together with the quotient it distinguishes every state at one column, as the sum does. But it
is **not** an approximation: the product isn't a congruence of division's machine
(`test_division_readouts_approximations_and_the_cover`). If the sum helps and the product
doesn't, the approximation is the reason.

## Design

**The cell is stage 5's exactly:** `p` 5, two factors of 16 codes (`s` 16), the poolings reading
the window's output, the probe's settings, 400 epochs trained in all (`--budget-epochs 400`; 180
minutes only a guard), stop at certified, no fork. Only the arm changes, and every arm shares one
vocabulary (D2).

| arm | queries per window | an approximation? | covers the state? |
| --- | --- | --- | --- |
| `quotient` | the quotient alone | - | no |
| `sum` | + the sum | yes | no (the sum alone isn't 1-distinguishing) |
| `family` | + the sum and the difference (the cover) | both | yes |
| `product` | + the product `AB` | **no** | no |

**Seeds 200-219 in every arm** (D3). The certificate is against division's 25-state machine, in
both verdicts; every certified network is recertified on `pyramidon` in both, then long-tested
(division's U1 and S1 at 128, 256 and 1,024 columns, D4 of the division workplan).

## Feasibility (experiment-design.md, section 1)

| cell | target states | the model's states | per factor | exclusion bound `m + q - 1` | cap |
| --- | --- | --- | --- | --- | --- |
| every arm, `p` 5, two factors of 16 | 25 | 256 joint | 16 (a counter 5) | 280 | 280 |

Every arm reads the same state and can represent the target (stage 5's family certified 13 of 20
in this cell, its quotient arm 1 of 20). No run reaches its cap in 400 epochs; every success is
certified, which is exact at every length.

## The claims (registered 2026-10-16, before the data)

Each is Fisher's exact test, one-sided, on the counts certified in both verdicts. **Holm's
correction over the three, family-wise 0.05.**

- **A1, the cover helps (stage 5 replicated with the wider vocabulary):** `family` certifies more
  often than `quotient`. **Holds** if its Holm-adjusted p < 0.05; **refuted** otherwise.
- **A2, one approximation helps:** `sum` certifies more often than `quotient`. **Holds** if its
  Holm-adjusted p < 0.05; **refuted** otherwise.
- **A3, the approximation, not any extra output:** `sum` certifies more often than `product`.
  **Holds** if its Holm-adjusted p < 0.05; **refuted** otherwise.

**Readings, stated before the data.**
- A2 and A3 hold: one approximation helps, and because it is an approximation.
- A2 holds and A3 doesn't: an extra output helps, approximation or not (`product` about as good
  as `sum`).
- A2 refuted: P2's "one approximation suffices" isn't supported at 20 seeds, and the paper says
  so.
- A1 refuted with the wider vocabulary: the vocabulary matters, and the comparison with stage 5
  is reported as such.

**Reports.** Each arm's count certified with its Wilson 95% interval; each success's step, epochs
and minutes; each arm's runs by outcome (`readout`: near a solution and lost it, or never near);
the drops' scorer (`drops`, its default trigger) on every run, descriptive.

## Decisions

- **D1. The arms: the quotient alone, + the sum, + the cover, + the product. Settled: (a).**
  - (a) *Chosen.* **These four.** Pros: A1-A3 each a pair of arms; the product is the control the
    review asked for. Cons: four arms is 80 runs.
  - (b) Without `family` (stage 5 has it). Pros: 60 runs. Cons: stage 5's family used a narrower
    vocabulary (D2), so without the arm no comparison inside this study shows the cover's effect.
  - (c) Add `difference` alone (the other approximation). Pros: symmetry. Cons: 100 runs; the sum
    and difference are the same kind of image (both congruences), so it adds little.
- **D2. One vocabulary for every arm: sum, difference and product. Settled: (a).**
  - (a) *Chosen.* **Every arm embeds all three query tokens,** using only its own. Pros: the
    arms' networks differ only in what they're asked; the same seed draws the same initial weights
    in every arm. Cons: not stage 5's networks (one more embedding row), so stage 5's counts are
    context, not this study's control.
  - (b) Each arm its own vocabulary. Cons: the arms' networks then differ in size and initial
    draw, a confound.
- **D3. Seeds 200-219.** Pros: fresh (stage 5 used 100-119, the probes 0-7); 20 a cell, as stage
  5. Cons: none known.
- **D4. No fork. Settled: (a).**
  - (a) *Chosen.* **Stage 5's settings: no fork.** Pros: the fork could equalise the arms by
    rescuing collapses, which would confound the arm's effect; stage 5's numbers stay comparable.
    Cons: fewer successes overall.
  - (b) The fork on. Pros: more successes. Cons: the confound; a second change from stage 5.
- **D5. Holm over A1-A3. Settled: (a).**
  - (a) *Chosen.* **Holm, family-wise 0.05.** Pros: three registered tests of one study; Holm is
    uniformly more powerful than Bonferroni. Cons: none at three tests.
  - (b) Uncorrected. Cons: three chances at 0.05.
- **D6. Where: `jebel` (6 workers) after the `p` 11 ladder (docs/p11-trigger-workplan.md, stage 3),
  and `pyramidon` (3 workers) from launch, the seeds split. Settled: (a).**
  - (a) *Chosen.* **`pyramidon` seeds 214-219 (24 runs), `jebel` seeds 200-213 (56 runs),** each
    machine taking every arm at once. Pros: about 9 h on `jebel` from its start and 10-11 h on
    `pyramidon` (stage 5: about 55 minutes a run on `jebel`, 80 on `pyramidon`); runs are
    deterministic, so neither the machine nor the worker count changes a result. Cons: `pyramidon`
    busy during development; 3 workers, not 4, beside the fork probe's long tests (3.3 GB free, no
    swapping).
  - (b) `jebel` alone. Cons: about 13 h after the `p` 11 ladder.

## Stages

1. **The build** (with this plan): `product` in `modular_family.OPERATIONS`, the `product` arm,
   `Encoding.vocabulary` (default today's two operations, so every earlier study's tokens and
   networks are unchanged: a 3-step run at `p` 5 gives a byte-identical network), `--vocabulary` on
   `run`, `certify` and `long-test`.
2. **The runs** (D6): `run --p 5 --arm <arm> --factors 2 --reads output --vocabulary sum
   difference product --budget-epochs 400 --budget-minutes 180 --first-seed S --seeds N --workers W
   --out data/window-recurrent/approximation-claim/<arm>`; recertify every certified network on
   `pyramidon` in both verdicts; `readout ... --arms sum quotient` and the others; long tests.
3. **Findings:** A1-A3 as registered, in the probe script's docstring and here; P2's
   "one approximation" section rewritten on the result; the manifest; the networks archived in
   indrajala-models. Then this plan retired.

Tier 0: the study module and its script, pure Python; no timed path.
