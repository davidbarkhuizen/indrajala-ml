# Literature review: learned recurrent adders and length generalisation

**Status: written 2026-10-13, after the window-recurrent matrix (stage 9) and while its control ran,
to judge which findings are new. It backs
[window-recurrent-claims-workplan.md](window-recurrent-claims-workplan.md), which plans the work to
substantiate them.** Every paper read is kept in `~/code/papers/` as PDF and text, under the file
name given; arXiv ids are given for each.

## Scope and method

The question: which of the window-recurrent adder's findings (stage 9 of
[window-recurrent-adder-workplan.md](window-recurrent-adder-workplan.md); the findings in
`scripts/window_recurrent_study.py`'s docstring) are already in the literature? The searches
(2026-10-13) covered recurrent and Neural-GPU-style length generalisation on arithmetic,
discretised or quantised recurrent states, segment- and block-recurrent transformers on
arithmetic, curricula for algorithmic learning, and recent (2025-2026) work on length
generalisation in recurrent models. Eleven papers were read in the parts that bear on the
findings; the window-recurrent workplan's own reading (twelve papers) is summarised beside them.

A search can't prove absence: preprints from the last weeks may not be indexed, and work under
other names (neural automata, finite-state controllers, chunked state passing) can be missed. The
gaps below are "not found", not "not done".

## What our study found (the claims under review)

From 90 runs (`k` 1, 2, 4 columns a window x `s` 4, 16 state width x `m` 4, 8, 16 codes x 5
seeds), each a causal transformer over a window of base-3 column tokens whose GRU state is snapped
to the nearest of `m` learned codes and handed to the next window, trained end to end from the
digits alone, by a curriculum of 1, then 1-2, then 1-3 windows:

1. 45 of 90 runs pass every catalogue property at every test length to 64 columns (two of them
   to 24 and 32) after training on at most 3 windows (3, 6, 12 columns): window ratios 5.3-21.3.
2. No run works after one window of training; the recurrence is what generalises.
3. More codes give more working runs (10, 16, 19 of 30 at `m` 4, 8, 16), although 3 suffice;
   the state's width barely matters (24 and 21 of 45 at `s` 4, 16).
4. The seed decides within a cell (no cell works on every seed).
5. Five `k` 1 runs were exact on every sum of 1-3 columns and still failed at 8.
6. The control (all-at-once training, W16, W30) works as often as the curriculum: 12 of 15 runs
   against 12 of 15 at the same cells and seeds (10 of 10 at `k` 4), on about 16 times fewer
   examples at `k` 4. The curriculum isn't needed here.

## The papers

### Recurrent and recurrent-like networks that learn addition and extrapolate

| paper | what it does | result | bearing |
| --- | --- | --- | --- |
| Kaiser & Sutskever 2016, Neural GPUs Learn Algorithms (1511.08228; `kaiser-2015-neural-gpu`) | a convolutional GRU applied as many times as the input is long | binary addition and multiplication: trained on up to 20 bits, no error on any input tested up to 2,000 bits; generalising models are rare among seeds, found with curriculum, noise and parameter sharing relaxation | the headline (a learned carry extrapolating 100x) is long established |
| Price, Zaremba & Sutskever 2016, Extensions and Limitations of the Neural GPU (1611.00736; `price-2016-neural-gpu-extensions-limitations`) | the Neural GPU on decimal operations, a curriculum over bases, larger models | very sensitive to the seed: about 15 seeds a configuration; of 1,121 decimal-addition seeds, 94% fail by a carry of 7-9 digits, 5.5% reach 33 or more, the best 112; failures on unusual inputs (long runs of zeros) | finding 4 is known in kind; our rate (half the runs) is far higher, but the architecture differs |
| Robinson 2020, Progress Extrapolating Algorithmic Learning to Arbitrary Sequence Lengths (2003.08494; `robinson-2020-extrapolating-algorithmic-learning`) | bins activations to discretise the state, adds a "digital" loss pulling them to the bins, and a localised memory | no extrapolation error detected within memory limits on copy, parentheses and binary addition; counts the distinct computational states a network uses, constant with length where it extrapolates and growing where it doesn't | the closest prior use of a discretised state for extrapolation, and of counting the states used (our "codes used") |
| Papazov, D'Angelo & Flammarion 2025, Exact Learning of Arithmetic with Differentiable Agents (2511.22751; `papazov-2025-exact-learning-arithmetic-differentiable-agents`) | differentiable finite-state transducers trained on policy trajectories of an expert agent (intermediate steps supervised) | binary and decimal addition and multiplication with no error on inputs thousands of times longer than training | exact generalisation, but supervised on the algorithm's steps; ours sees only the sum's digits |
| Mohri 2026, Rational Transductors (2602.07599; ICML 2026 oral; `mohri-2026-rational-transductors`) | a transformer beside a matrix-valued recurrence from weighted automata; for addition, column-stochastic transitions and no positional encodings | two-number decimal addition, trained on 10-40 digits (length sampled uniformly), 100% exact at 1,000 digits; 5 seeds, negligible variance; a standard transformer is 0% at 100 | the nearest recent result: a transformer with a recurrent state that learns the carry automaton; one digit a step, a continuous constrained state, no seed sensitivity reported |
| Patriota 2024, Arbitrary-Length Generalization for Addition in a Tiny Transformer (2406.00075; `patriota-2024-arbitrary-length-addition-tiny-transformer`) | a tiny transformer trained on two-digit additions, run right to left, a digit and carry at a time | generalises to unseen lengths | recurrence by the decoding procedure, outside the network; the parked "compose outside the network" option of our parent plan |

### Discrete and finite-state recurrent states

| paper | what it does | result | bearing |
| --- | --- | --- | --- |
| Koul, Greydanus & Fern 2019, Learning Finite State Representations of Recurrent Policy Networks (1811.12530; `koul-2019-finite-state-recurrent-policies`) | inserts quantised bottlenecks (straight-through) into a trained RNN policy's memory and observations, then extracts a Moore machine | small machines, e.g. 3 memory states for Pong | snapping a recurrent state with a straight-through gradient is known; there it is inserted after training, for interpretation |
| van den Oord et al. 2017, VQ-VAE (1711.00937; `van-den-oord-2017-vq-vae-neural-discrete-representation`) | learned codebook, nearest-code snapping, straight-through, codebook and commitment losses | | our discretiser (W4, W14) |
| Brandoit et al. 2026, Length-Independent State Tracking under a Parallel Scan (2609.32447; `brandoit-2026-length-independent-state-tracking`) | formalises finite-state tracking at finite precision; proves tracking correct at every length needs contraction (against perturbation) and separation (between states); affine recurrences realise only definite automata | a non-affine scan-compatible RNN learns exact transition tables, which certifies correctness beyond the tested lengths | the theory for why snapping helps (it contracts every state onto a code and keeps codes apart), and the model for a certificate: an extracted exact transition table |

### Length generalisation in recurrent models

| paper | what it does | result | bearing |
| --- | --- | --- | --- |
| Buitrago Ruiz & Gu 2025, Understanding and Improving Length Generalization in Recurrent Models (2507.02782; `buitrago-2025-length-generalization-recurrent-models`) | the "unexplored states hypothesis": recurrent models fail past their training length when training visits only part of the states the recurrence reaches | interventions (noise or another sequence's final state as the initial state) give 2k to 128k with 500 post-training steps | a candidate explanation of finding 5, testable on our codes (do failing runs reach codes at 8 columns that training never produced?) |
| Jiang 2026, Learning Length-Extrapolatable Recurrent Models (2609.09157; `jiang-2026-length-extrapolatable-recurrent-models`) | rescales the backward "state credit" through time (CST) | gains to 128x the training length | an alternative fix if the seed failures are credit, not capacity |
| Ge et al. 2026, Register Tokens for Bounded-State Reasoning in Diffusion Language Models (2609.16372; `ge-2026-register-tokens-bounded-state-reasoning`) | a diffusion LM decodes fixed-size chunks and carries continuous register vectors between them | on LongArithmetic, beyond the training horizon (5-8 operations) registers solve 5 of 64 at 8 operations and none beyond | the nearest chunk-and-carried-state design; its continuous state doesn't extrapolate, ours (discrete) does; tasks and scales differ, so it motivates the snapping ablation rather than settling it |

### Read for the window-recurrent workplan (summarised there)

Bulatov et al. 2022, Recurrent Memory Transformer (2207.06881): memory tokens between segments,
trained and tested at the same segment counts. Hutchins et al. 2022, Block-Recurrent Transformers
(2203.07852): a gated recurrent cell over blocks; the model learns to ignore the state unless its
gates are biased to remember (our W13). Cho et al. 2014 (1406.1078): the GRU. Press et al. 2022,
ALiBi (2108.12409); Sun et al. 2022, xPos (2212.10554): distance biases for fixed-depth
extrapolation. And the terminated length-generalisation plan's reading (`git show
c3d619b:docs/length-generalisation-workplan.md`): Zhou et al. 2024, McLeish et al. 2024 (Abacus
embeddings, 20 to 120 digits), Cho et al. 2024 (position coupling, 30 to 200 digits), Fan et al.
2024 (looped transformers, a loop count raised by a curriculum), Kazemnejad et al. 2023, Ruoss et
al. 2023, Allen-Zhu & Li 2024. These extend fixed-depth transformers by position schemes or loops;
none hands a state between windows.

### Surfaced, not read

Recurrent Looped Transformer (2026 project page): recurrence over depth, addition to 32 digits no
better than GPT. Looped transformers for length generalisation (2409.15647). What Algorithms can
Transformers Learn? (2310.16028). Zaremba & Sutskever 2014, Learning to Execute (curricula for
program execution).

## What is known, and what isn't (as far as found)

**Known.**

- A learned recurrent carry extrapolates far: 100x (Neural GPU), 25x and exact (Rational
  Transductors), thousands of times with supervised steps (DFSTs). Our 5-21x, tested to 64
  columns, is weaker on length than all three.
- Discretising a recurrent state to stop drift, and counting the states used (Robinson 2020);
  snapping with a straight-through gradient (Koul et al. 2019; VQ-VAE); the theory that exact
  tracking at every length needs contraction and separation (Brandoit et al. 2026).
- Seed sensitivity of learned algorithms (Kaiser & Sutskever; Price et al.: 5.5% of seeds).
- Recurrent models failing past training when training leaves states unexplored (Buitrago Ruiz &
  Gu).
- Curricula helping algorithmic learning (Price et al.; Fan et al.); Rational Transductors'
  "curriculum" is uniform length sampling, close to our all-at-once arm.

**Not found.**

1. **The design:** a transformer over a window of columns with a VQ-snapped GRU state handed
   between windows, trained end to end from input-output pairs, for exact arithmetic. The nearest
   are Rational Transductors (a recurrence per token, continuous and constrained) and register
   tokens (chunks with a continuous carried state, which doesn't extrapolate).
2. **An effect of the codebook's size on whether training finds the solution** when a codebook of
   the solution's size (3) would do: more codes, more working runs.
3. **A certificate for a learned window-recurrent adder:** because the state is snapped, the
   next state is an exact function of (code, window contents), and both sets are finite (`m`
   codes; `27^k` windows, 531,441 at `k` 4), so the learned machine can be enumerated and checked
   against the carry automaton for every length at once. Brandoit et al. certify their NFSM's
   transition tables likewise; no certificate for a learned adder with a transformer inside the
   step was found.
4. **A controlled curriculum comparison** for this design: the control found none at 5 seeds
   (12 of 15 each way), so the finding is that this design needs no curriculum, against the
   curricula of the Neural GPU (Price et al.) and looped transformers (Fan et al.).

**Weaker than the literature, to fix before any claim:** test lengths (64 against 1,000-2,000),
seeds per cell (5 against Price's 15), and no ablation of the snapping itself.

## References (arXiv ids; local copies in `~/code/papers/`)

1511.08228, 1611.00736, 2003.08494, 2511.22751, 2602.07599, 2406.00075, 1811.12530, 1711.00937,
2609.32447, 2507.02782, 2609.09157, 2609.16372; and, from the window-recurrent workplan,
2207.06881, 2203.07852, 1406.1078, 2108.12409, 2212.10554.
