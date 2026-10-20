# Literature review: learned recurrent adders and length generalisation

**Status: written 2026-10-13, after the window-recurrent matrix (stage 9) and while its control ran,
to judge which findings are new; extended the same day, after the certificate (E1), to the older
literature on discretised recurrent states and automaton extraction, which the first searches
missed. It backed the claims workplan, which planned the work to substantiate them
(retired; its report, [window-recurrent-report.md](window-recurrent-report.md)). Extended
2026-10-15 for the division paper and 2026-10-19 for the architecture's novelty (the sections at
the end).** Every paper read is kept in the literature repository (`davidbarkhuizen/indrajala-literature`, cloned as `~/code/indrajala-literature/`) as PDF and text, under the file
name given; arXiv ids are given for each.

## Scope and method

The question: which of the window-recurrent adder's findings (stage 9 of
`git show e45944e:docs/window-recurrent-adder-workplan.md`; the findings in
`scripts/window_recurrent_study.py`'s docstring) are already in the literature? The searches
(2026-10-13) covered recurrent and Neural-GPU-style length generalisation on arithmetic,
discretised or quantised recurrent states, segment- and block-recurrent transformers on
arithmetic, curricula for algorithmic learning, and recent (2025-2026) work on length
generalisation in recurrent models. Eleven papers were read in the parts that bear on the
findings; the window-recurrent workplan's own reading (twelve papers) is summarised beside them.
The extension (after E1) searched grammatical inference with recurrent networks, discretised or
clustered states trained in the loop, automaton extraction, proofs for learned arithmetic and
verification of recurrent networks, and read twelve more.

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
7. The certificate (E1, claims workplan; every network at its last step): of the 105 networks
   (the matrix's 90 and the control's 15), 52 are certified (the network's own snaps) and 48
   robustly (either snap of every near tie), every one a working run that passed every length to
   64 columns, and every one minimising to the 3-state carry automaton while using 3-11 codes. No
   failing network is certified. Two networks that passed every test length to 64 columns
   (100,000 cases each) are wrong at 12, found by the certificate and confirmed on the networks.

## The papers

### Recurrent and recurrent-like networks that learn addition and extrapolate

| paper | what it does | result | bearing |
| --- | --- | --- | --- |
| Kaiser & Sutskever 2016, Neural GPUs Learn Algorithms (1511.08228; `kaiser-2015-neural-gpu`) | a convolutional GRU applied as many times as the input is long | binary addition and multiplication: trained on up to 20 bits, no error on any input tested up to 2,000 bits; generalising models are rare among seeds, found with curriculum, noise and parameter sharing relaxation | the headline (a learned carry extrapolating 100x) is long established |
| Price, Zaremba & Sutskever 2016, Extensions and Limitations of the Neural GPU (1611.00736; `price-2016-neural-gpu-extensions-limitations`) | the Neural GPU on decimal operations, a curriculum over bases, larger models | very sensitive to the seed: about 15 seeds a configuration; of 1,121 decimal-addition seeds, 94% fail by a carry of 7-9 digits, 5.5% reach 33 or more, the best 112; failures on unusual inputs (long runs of zeros) | finding 4 is known in kind; our rate (half the runs) is far higher, but the architecture differs |
| Robinson 2020, Progress Extrapolating Algorithmic Learning to Arbitrary Sequence Lengths (2003.08494; `robinson-2020-extrapolating-algorithmic-learning`) | bins activations to discretise the state, adds a "digital" loss pulling them to the bins, and a localised memory | no extrapolation error detected within memory limits on copy, parentheses and binary addition; counts the distinct computational states a network uses, constant with length where it extrapolates and growing where it doesn't | a later use of a discretised state for extrapolation (Zeng et al. 1993 is the first found, below), and of counting the states used (our "codes used") |
| Papazov, D'Angelo & Flammarion 2025, Exact Learning of Arithmetic with Differentiable Agents (2511.22751; `papazov-2025-exact-learning-arithmetic-differentiable-agents`) | differentiable finite-state transducers trained on policy trajectories of an expert agent (intermediate steps supervised) | binary and decimal addition and multiplication with no error on inputs thousands of times longer than training | exact generalisation, but supervised on the algorithm's steps; ours sees only the sum's digits |
| Mohri 2026, Rational Transductors (2602.07599; ICML 2026 oral; `mohri-2026-rational-transductors`) | a transformer beside a matrix-valued recurrence from weighted automata; for addition, column-stochastic transitions and no positional encodings | two-number decimal addition, trained on 10-40 digits (length sampled uniformly), 100% exact at 1,000 digits; 5 seeds, negligible variance; a standard transformer is 0% at 100 | the nearest recent result: a transformer with a recurrent state that learns the carry automaton; one digit a step, a continuous constrained state, no seed sensitivity reported |
| Patriota 2024, Arbitrary-Length Generalization for Addition in a Tiny Transformer (2406.00075; `patriota-2024-arbitrary-length-addition-tiny-transformer`) | a tiny transformer trained on two-digit additions, run right to left, a digit and carry at a time | generalises to unseen lengths | recurrence by the decoding procedure, outside the network; the parked "compose outside the network" option of our parent plan |

### Discrete and finite-state recurrent states

| paper | what it does | result | bearing |
| --- | --- | --- | --- |
| Koul, Greydanus & Fern 2019, Learning Finite State Representations of Recurrent Policy Networks (1811.12530; `koul-2019-finite-state-recurrent-policies`) | inserts quantised bottlenecks (straight-through) into a trained RNN policy's memory and observations, then extracts a Moore machine | small machines, e.g. 3 memory states for Pong | snapping a recurrent state with a straight-through gradient is known; there it is inserted after training, for interpretation (during training: Zeng et al. 1993, below) |
| van den Oord et al. 2017, VQ-VAE (1711.00937; `van-den-oord-2017-vq-vae-neural-discrete-representation`) | learned codebook, nearest-code snapping, straight-through, codebook and commitment losses | | our discretiser (W4, W14) |
| Brandoit et al. 2026, Length-Independent State Tracking under a Parallel Scan (2609.32447; `brandoit-2026-length-independent-state-tracking`) | formalises finite-state tracking at finite precision; proves tracking correct at every length needs contraction (against perturbation) and separation (between states); affine recurrences realise only definite automata | a non-affine scan-compatible RNN learns exact transition tables, which certifies correctness beyond the tested lengths | the theory for why snapping helps (it contracts every state onto a code and keeps codes apart), and the model for a certificate: an extracted exact transition table |

### Discretised states during training, and automaton extraction (the extension)

The first searches looked for length generalisation and discrete states in recent work; this line,
from grammatical inference with recurrent networks (1990s onward), is the closest prior work to the
snap and the certificate.

| paper | what it does | result | bearing |
| --- | --- | --- | --- |
| Zeng, Goodman & Smyth 1993, Learning Finite State Machines with Self-Clustering Recurrent Networks (Neural Computation 5:976; `zeng-1993-self-clustering-recurrent-networks`) | a second-order RNN's state hard-thresholded **during training**, trained by a pseudo-gradient (the sigmoid's gradient through the threshold: straight-through in all but name) | the analog network's clusters drift and merge on long strings (14 of 15 networks, some at 1.3x the training length); the discretised network's states are isolated points, so it "behaves exactly like a state machine": the machine read off it, Moore-minimised, is the target (Tomita 4: 6 points to 4 states), and it classifies correctly "independent of the lengths" | **the snap, the drift it cures, and reading the exact machine off and minimising it are all here**, for binary-input acceptors of a few states |
| Das & Mozer 1993, A Unified Gradient-Descent/Clustering Architecture for Finite State Machine Induction (NeurIPS 6; `das-1993-unified-gradient-clustering-fsm-induction`) | DOLCE: an RNN with an adaptive clustering of its state space learned with the weights; the state recovered as the most probable cluster | better generalisation than unclustered and rigidly quantised nets on the Tomita languages | **a learned codebook in the loop**, VQ's ancestor for recurrent states |
| Kolen 1993, Fool's Gold: Extracting Finite State Machines from Recurrent Network Dynamics (NeurIPS 6; `kolen-1993-fools-gold-extracting-fsm`) | shows that extraction by partitioning a continuous state space can return illusory machines (sensitivity to initial conditions, the measurement's granularity) | | the critique a snapped state escapes: our machine isn't inferred from a partition, it is the network |
| Omlin & Giles 1996, Extraction of Rules from Discrete-Time Recurrent Neural Networks (Neural Networks 9:41; `omlin-1996-extraction-rules-discrete-time-rnn`) | DFAs extracted by quantising a trained second-order RNN's state space, then minimised | the smallest consistent extracted DFA generalises better than the network it came from | certifying the extracted machine, not sampling the network, is the old lesson too |
| Weiss, Goldberg & Yahav 2018, Extracting Automata from Recurrent Neural Networks Using Queries and Counterexamples (ICML; 1711.09576; `weiss-2018-extracting-automata-queries-counterexamples`) | Angluin's L* with the RNN as teacher, an abstraction of its state space refined on conflict | on networks at 100% train and test accuracy, extraction finds words the network misclassifies, outside both sets | **"high test accuracy, still wrong" is known in kind**; our certificate's two networks passing to 64 columns and wrong at 12 are a case of it |
| Wang & Niepert 2019, State-Regularized Recurrent Neural Networks (ICML; 1901.08817; `wang-2019-state-regularized-rnns`) | a GRU/LSTM whose state is a softmax-weighted mix of `k` learnable centroids, nearest-centroid as the temperature goes to 0, trained end to end | the minimal correct DFA recovered for every Tomita grammar; with `k` 50 available the network uses only the minimal number; better extrapolation on balanced parentheses; mentions Zeng's drift | **the modern learned-centroid state**, the nearest to our codes; it reports the minimal machine used, where ours use 3-11 codes and minimise to 3 |
| Oliva & Lago-Fernández 2020, Separation of Memory and Processing in Dual Recurrent Neural Networks (2005.13971; `oliva-2020-dual-recurrent-memory-processing`) | noise in the recurrent units' activation drives them to a binary regime; regular languages and **addition in bases 2 and 10** | the minimal Moore machine extracted for base-2 addition is the carry automaton; tested on strings of 200,000 digits | **a gradient-trained, effectively discrete RNN that learns the carry automaton, extracted**: tested, not certified; one digit a step, no transformer |
| Lan, Geyer, Chemla & Katzir 2022, Minimum Description Length Recurrent Neural Networks (TACL 10; 2111.00600; `lan-2022-minimum-description-length-rnns`) | networks found by evolutionary search under an MDL objective (no gradient), with floor and other units | a binary adder from 400 pairs, **proved correct for every input by hand** (induction on the carry); "no other RNN has been proven to hold a carry-over ... for an unbounded number of digits" | **a proof for all lengths of a learned adder exists**: by hand, of one network found by search |
| Michalenko et al. 2019, Representing Formal Languages: A Comparison between Finite Automata and Recurrent Neural Networks (ICLR; 1902.10297; `michalenko-2019-representing-formal-languages-fa-rnn`) | whether a simple decoding maps a trained RNN's states to the minimal DFA's | one exists, but onto an abstraction of the minimal DFA (its states clustered into "superstates"), not its states | background: a continuous state needn't hold the minimal machine's states apart, which a snapped one is trained to |
| Delétang et al. 2023, Neural Networks and the Chomsky Hierarchy (ICLR; 2207.02098; `deletang-2022-neural-networks-chomsky-hierarchy`) | 20,910 models over 15 tasks, the Chomsky hierarchy as the predictor of length generalisation | its binary addition feeds the operands one after the other (context-sensitive): only a tape-augmented RNN generalises | not our task: aligned columns make addition a finite transducer, which is why a finite state can do it |
| Liu et al. 2023, Transformers Learn Shortcuts to Automata (ICLR; 2210.10749; `liu-2022-transformers-shortcuts-automata`) | shallow transformers simulate automata by parallel "shortcuts" | standard training finds such shortcuts; the paper studies their brittleness and mitigations | why a transformer alone (fixed depth) doesn't give every length, and a recurrence does |
| Ahuja & Mansouri 2024, On Provable Length and Compositional Generalization (2402.04875; `ahuja-2024-provable-length-compositional-generalization`) | sufficient conditions for length generalisation of limited-capacity sequence models | provable, under assumptions on the hypothesis class and the training distribution | a theory of when, not a check of a given network |

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
  columns, is weaker on length than all three as a test; for the certified networks the
  certificate covers every length (E1).
- Discretising a recurrent state to stop drift (Zeng et al. 1993, during training; Robinson 2020),
  and counting the states used (Robinson 2020); snapping with a straight-through gradient (Zeng et
  al.'s pseudo-gradient; Koul et al. 2019; VQ-VAE); the theory that exact tracking at every length
  needs contraction and separation (Brandoit et al. 2026).
- Seed sensitivity of learned algorithms (Kaiser & Sutskever; Price et al.: 5.5% of seeds).
- Recurrent models failing past training when training leaves states unexplored (Buitrago Ruiz &
  Gu).
- Curricula helping algorithmic learning (Price et al.; Fan et al.); Rational Transductors'
  "curriculum" is uniform length sampling, close to our all-at-once arm.
- **From the extension:** discretising the recurrent state during training with a straight-through
  style gradient, so that it stops drifting and *is* a finite machine, read off exactly and
  minimised to the target (Zeng et al. 1993); a learned clustering or codebook of the state
  trained with the weights (Das & Mozer 1993; Wang & Niepert 2019, who recover the minimal DFA);
  extracting the carry automaton from an effectively discrete gradient-trained adder (Oliva &
  Lago-Fernández 2020, tested to 200,000 digits); a hand proof that one learned (evolved) adder
  is correct for every input (Lan et al. 2022); networks at 100% test accuracy that extraction
  shows wrong elsewhere (Weiss et al. 2018). Extraction from a continuous state can mislead
  (Kolen 1993), and an extracted machine can be better than its network (Omlin & Giles 1996).

**Not found** (revised after the extension: the snap's idea and the extraction are 1993's; what
stays new is narrower and stated so).

1. **The design:** a transformer over a window of columns with a VQ-snapped GRU state handed
   between windows, trained end to end from input-output pairs, for exact arithmetic. The nearest
   are Rational Transductors (a recurrence per token, continuous and constrained) and register
   tokens (chunks with a continuous carried state, which doesn't extrapolate); the discretised
   RNNs above (Zeng; Das & Mozer; Wang & Niepert; Oliva) read one symbol a step with a small cell
   and no transformer, on acceptors or binary addition.
2. **An effect of the codebook's size on whether training finds the solution** when a codebook of
   the solution's size (3) would do: more codes, more working runs. Wang & Niepert give their
   networks spare centroids (`k` 50) and find the minimal number used; they report no success
   rate against `k`. Ours differ in both: the working networks use 3-11 codes (redundant copies
   of a carry, minimised away), and the rate of working runs rises with `m`.
3. **A certificate for a learned window-recurrent adder:** because the state is snapped, the
   next state is an exact function of (code, window contents), and both sets are finite (`m`
   codes; `27^k` windows, 531,441 at `k` 4), so the learned machine can be enumerated and checked
   against the carry automaton for every length at once. Brandoit et al. certify their NFSM's
   transition tables likewise; no certificate for a learned adder with a transformer inside the
   step was found. **Revised:** reading the exact machine off a discretised network and
   minimising it is Zeng et al.'s (1993), and a learned adder proved for every input is Lan et
   al.'s (2022, by hand, one evolved network). What stays new: the check is automatic and
   complete over an input alphabet far too large to read off training data (every one of `27^k`
   windows from every reachable code), it carries numerical margins (a near tie branches, so a
   pass holds for either rounding), it compares against a transducer's outputs (digits and
   carries) rather than an acceptor's verdict, and it ran on 105 gradient-trained networks (48
   certified robustly, 52 strictly, every one minimising to the 3-state carry automaton). That it
   catches networks that pass 100,000-case tests at every length to 64 columns (two, wrong at
   12) is Weiss et al.'s phenomenon, shown here with a complete check rather than a search.
4. **A controlled curriculum comparison** for this design: the control found none at 5 seeds
   (12 of 15 each way), so the finding is that this design needs no curriculum, against the
   curricula of the Neural GPU (Price et al.) and looped transformers (Fan et al.).

**Positioning, after the extension.** The claim to make is not "a discretised state makes a
network a finite machine that can be read off" (1993), but: a transformer-in-the-step,
window-recurrent network with a VQ-snapped state learns base-3 addition from examples, and its
learned machine can be certified exactly and automatically over every window, at every length
and alignment, with margins; and certification shows sampled long tests overstate working.
E2 (the continuous ablation) and E6 (the cell without the transformer) say which part matters.

**Weaker than the literature, to fix before any claim:** seeds per cell (5 against Price's 15:
E3 and E4 run 20), no ablation of the snapping itself (E2), no baseline without the transformer
in the step (E6), and one task (base-3 addition). Test lengths (64 against 1,000-2,000) matter
now only for networks that aren't certified: the certificate covers every length, and the long
tests (D9, to 1,024 columns) check it.

## Extension (2026-10-15): for the division paper

Searched after division certified (the modular family's probe, division stages 2-4): automaton
decomposition terms, auxiliary and multi-task training, factored codebooks, codebook collapse, and
loss of a learned solution. 22 papers read in the parts that bear on the findings, each in the
literature repository; what they changed:

- **Terms** (Egri-Nagy & Nehaniv 2024, `egri-nagy-2024-covering-lemma-transformation-semigroups`,
  2404.11923; 2015, `egri-nagy-2015-computational-holonomy-decomposition`, 1508.06345; Chow 1978,
  `chow-1978-testing-software-design-finite-state-machines`): an injective morphism is an
  *emulation* or *covering* (the covering machine computes at least as much), a surjective one's
  target an *approximation*; cascade components are *coordinates*; a machine is
  *k-distinguishable* if every pair of states is told apart by some input of length k (Chow's
  footnote), and his *characterization set* W tells every pair apart. So division's machine is
  covered by the sum and difference counters (odd `p`), each an approximation; checked in
  `test_division_readouts_approximations_and_the_cover`.
- **Auxiliary outputs.** Caruana 1997 (`caruana-1997-multitask-learning`): multitask learning's
  "eavesdropping", a feature easy to learn for one task and hard for another, our reading of the
  cover. Dan et al. 2022 (`dan-2022-robust-generalization-regular-languages`, 2202.09717): supervising
  an RNN's states with the target DFA's improves robust generalisation on modulo languages a lot
  (e.g. 66 to 99%); their label-free count task helps 5-8% and records more than the state, where our
  approximations record less: **the closest prior work**. The CLRS line (Velickovic et al. 2020,
  2022; Ibarz et al. 2022) supervises every intermediate step ("hints") and finds positive transfer
  between related algorithms; Wies et al. 2023 prove intermediate supervision can make a composite
  learnable; Jaderberg et al. 2017, auxiliary tasks in RL. None trains on homomorphic images of a
  target machine; none shows one approximation without a cover sufficing (not found).
- **Modular arithmetic.** Power et al. 2022 include `x / y mod p` among grokking tasks (a symbol
  table, not a transducer); Nanda et al. 2023 reverse-engineer modular addition; He et al. 2024
  (`he-2024-learning-to-grok-modular-arithmetic`, 2406.02550) train on many `a x + b y mod p` (our sum
  and difference are two) and find deeper models' out-of-distribution generalisation **transient,
  needing early stopping**: a parallel to the collapse past a solution. Allchin 2026 (2607.11953):
  RL agents rarely learn permutation automata's states; division's machine is one (so is the
  modular sum, which we learn). Lee et al. 2023 (data formats for arithmetic): read, not cited.
- **Codes.** Product quantisation (Jegou et al. 2011) is our factored state's codebook. Index
  collapse (Kaiser et al. 2018, whose decomposed VQ partly answers it; Lancucki et al. 2020 and
  Dhariwal et al. 2020, restarts; Zheng & Vedaldi 2023, dead codes; Huh et al. 2023, the embeddings'
  drift from the codebook and the commitment loss's asymmetry). Our collapse is index collapse, but
  late and abrupt, after the codebook was in full use.
- **Late instability.** Thilak et al. 2022 (`thilak-2022-slingshot-mechanism-adaptive-optimizers`,
  2206.04817): with Adam, after a network fits its data, the last layer's norm grows until training
  examples are misclassified, in cycles: the best-fitting published account of the abrupt collapse
  (the probe script's findings, "the collapse, traced").

## Extension (2026-10-19): the architecture's novelty

Asked by the owner: how novel is the current architecture (W11's window, the factored snapped
state with a pooling, a cell and codes a factor, and the window branches of
the window branches workplan, retired 2026-10-19: `git show e7f1ff6:docs/window-branches-workplan.md`)? Searched modular recurrent networks,
discretised states and communication, factored categorical latents, branched multi-task
networks, Mealy machines learned by networks, and chunked recurrent transformers (2024-2026).
Seven papers were read in the parts that bear on the design: method, experiments and related work
in full. Each is in the literature repository.

| paper | what it does | bearing |
| --- | --- | --- |
| Tiňo & Šajda 1995, Learning and Extracting Initial Mealy Automata with a Modular Neural Network Model (Neural Computation 7:822; `tino-1995-learning-extracting-initial-mealy-automata`) | a recurrent network for Mealy machines with **one second-order path for the transition (state units) and one for the output** (a hidden layer second-order in state and input, then an extra first-order layer); trained by gradient on input/output strings; the machine extracted afterwards by clustering (a star-topology SOM), then minimised | **separate `delta` and `lambda` paths are 1995's**, at 4 machines of up to 5 states; the extra output layer was "theoretically unnecessary" and converged 3 of 10 runs against 5 of 10 for the plain Neural Mealy Machine (Chen et al. 1992), one task: no evidence either way |
| Goyal et al. 2021, Recurrent Independent Mechanisms (ICLR; 1909.10893; `goyal-2019-recurrent-independent-mechanisms`) | `k` recurrent modules (GRU or LSTM each, own parameters), each reading the input through its own attention, its query from its own state; top-`k` competition; attention between modules | **the per-factor pooling and cell are RIMs-like**; RIMs' states are continuous, their queries state-dependent (ours are learned constants), their modules compete; no machine is read off |
| Liu et al. 2021, Discrete-Valued Neural Communication (2107.02367; `liu-2021-discrete-valued-neural-communication`) | VQ (straight-through) on messages between components, a vector split into `G` segments quantised against **one shared codebook**; in GNNs, transformers and RIMs | a factored discretisation, but of communication, shared across segments; their ablation finds discretising RIMs' state updates **worse** than discretising communication (the adding task), the opposite of our choice; no finite machine |
| Hafner et al. 2021, Mastering Atari with Discrete World Models (DreamerV2; ICLR; 2010.02193; `hafner-2020-dreamerv2-discrete-world-models`) | a recurrent world model whose stochastic state is **32 categorical variables of 32 classes**, straight-through; categorical beat Gaussian latents on 42 of 55 games | a factored discrete recurrent state trained straight-through is established at scale; it sits beside a continuous GRU state, so the model isn't a finite machine |
| Vandenhende et al. 2020, Branched Multi-Task Networks (BMVC; 1904.02920; `vandenhende-2019-branched-multi-task-networks`) | hard parameter sharing: shared shallow layers, task-specific deeper branches, where to branch chosen by task affinity | **the window branches are this design** (one trunk, a branch a consumer); its term for what the branches avoid is negative transfer |
| Guo et al. 2020, Learning to Branch for Multi-Task Learning (ICML; 2006.01895; `guo-2020-learning-to-branch-multi-task`) | the branching learned (Gumbel-softmax over a tree of layers) | the same; our branch point is fixed (D1, D2) |
| Zhang et al. 2026, Recurrent Looped Transformer (2610.07591; `zhang-2026-recurrent-looped-transformer`) | a parallel causal encoder and a recurrent transformer decoder whose final output feeds back, gated, into the next token's input; continuous state; one readout for the answer and the state; 8 layers, width 512, 3 seeds | **the nearest recent design**: its chunked variant (RLT-2, the state handed on once per chunk of 4 and gated into every token) is our window's shape with a continuous state. Chunking keeps 64-bit parity (99%) but drops swap-based S5 at 64 operations from 100% to 20%; standard S5 (all 120 permutations) stays near chance in every variant; addition fails beyond 8 digits |

Surfaced, not read: Soulos et al. 2024, Recurrent Transformers Trade-Off Parallelism for Length
Generalization on Regular Languages (NeurIPS 2024 workshop; OpenReview 6PjZA4Jvge; no arXiv
version, and OpenReview refuses scripted downloads): from the abstract, chunked recurrent
transformers' generalisation on regular languages falls quickly beyond chunks of 2. Chen et al.
1992's Neural Mealy Machine (cited by Tiňo & Šajda).

**What the architecture's parts are, as far as found.**

- Each part has precedents: the snapped state (Zeng et al. 1993, above); factored discrete
  latents trained straight-through (DreamerV2; product quantisation; DVNC's segments); a pooling,
  cell and state a module (RIMs); the state added to a transformer over a chunk (Block-Recurrent
  Transformers, RMT, RLT-2); a shared trunk with a branch a consumer (branched multi-task networks);
  separate transition and output paths for a Mealy machine (Tiňo & Šajda 1995).
- **Not found:** their combination, in which the factored codes make the trained network a finite
  product machine, checked exactly. Of the modular and factored designs above, none is finite or
  read off; of the Mealy networks, none is checked beyond its training and test strings or exceeds
  a few states.
- **So the network is not the contribution** (the contribution document's "Where it is weakest").
  The branches' novelty is in the diagnosis that motivates them, not the design. The diagnosis is
  the readout probe and the separability check: a transition proved exact, an output its features
  don't linearly separate. If they help, the finding is that this negative transfer can be shown
  and cured on a proved machine.
- **A candidate finding, not a claim:** chunked recurrence with a continuous state loses
  permutation tracking (RLT-2; Soulos et al.'s abstract). Our snapped state certifies with the
  state handed on once per 4 input symbols (addition at `k` 4); division and the algebras hand it
  on once a symbol (a column, a token), as RLT-1 does. A registered contrast
  (the same window, continuous against snapped, on swap-based S5) would test whether the snap is
  what lets a chunked step track state (P1's 2c, at a chunk). Standard S5 failing in RLT at 120
  states, as ours does (round 2), says that cell is hard beyond our design.

## Extension (2026-10-20): state formation, the commitment, and lost states

Asked by the owner: look further for known understanding of our problem, and known fixes. The
problem, as the night of 2026-10-19 left it ([bottlenecks.md](bottlenecks.md); findings to be
written into the probe script's docstring):

- **Bottleneck 1 (no state forms) is mostly the commitment's.** With the commitment never on
  (`--commitment-from 1000`), 4 of round 1's 6 failure-A runs certified (S3 seeds 0, 3, 5;
  move-to-front-4 seed 3), against at most 1 of 6 for the diagnostic and both thermostat probes.
  Two seeds (S3 1, 4) collapsed to 1-4 codes **with the commitment off and the snap on**.
- **States formed are lost when the commitment returns** (the diagnostic, both thermostat probes:
  every formed state eroded within 1-2 steps of the return); with it off, S3 seed 3 lost a state
  and re-formed it.
- **Annealing (`--anneal`, D6-D11) merged the codes:** at width 32, 23-27 of a factor's 32 codes
  became identical (squared distance under 1e-6) by step 2-3 at `tau` 1.0, the codes starting
  about 0.03 apart; hardness stalled near 0.30 until `tau` was 1/64. Neither arm formed a state.
- **Bottleneck 2 (codes standing for several states)** moved with R3/R4 (keep the best machine,
  counterexample training): C2 at width 32 certified S5 2 of 6 and move-to-front-5 3 of 6, against
  0 of 6 each; with the commitment off, R3/R4 certified neither of move-to-front-5's seeds 0, 1.
- **Certified networks have thin margins** (near ties under the strict snap ratio of 1.05 in each
  commitment-off certificate, and in two each of C0's and C2's).

Searched vector-quantisation collapse and its fixes (2017-2026), soft-to-hard and annealed
quantisation, deterministic annealing, and finite automata learned by gradient with discrete or
noisy states. Eleven papers added to the literature repository, read in the parts that bear on
the above (method, mechanism and the experiments behind each claim); two already there re-read.

| paper | what it shows | bearing |
| --- | --- | --- |
| Rose 1998, Deterministic Annealing for Clustering, Compression, Classification, Regression, and Related Optimization Problems (Proc. IEEE 86:2210; `rose-1998-deterministic-annealing-clustering`) | with squared-error distortion and soft assignments `p ~ exp(-d/T)`, every codevector sits at the data's mean above a **critical temperature `T_c = 2 lambda_max`** (twice the largest eigenvalue of the data's covariance; Theorem 1); below it a cluster splits along its principal axis; coincident codevectors split only under a perturbation | **the exact mechanism of the annealing probe's merge:** our soft codes are DA with `T = tau`, started far above `T_c` for codes 0.03 apart; and identical codes stay identical under a deterministic gradient, so they never split. Annealing has to start below `T_c` (measured from the outputs `h`), or perturb coincident codes at each transition, as DA does |
| Mavridis & Baras 2021, Online Deterministic Annealing for Classification and Clustering (2102.05836; `mavridis-2021-online-deterministic-annealing`) | DA as an online (stochastic-approximation) algorithm; codevectors duplicated with a small perturbation and merged back when they coincide; the number in use grows by bifurcation as `T` falls | an online recipe for the same: duplicate-and-perturb instead of a fixed codebook all at one point |
| Agustsson et al. 2017, Soft-to-Hard Vector Quantization (NeurIPS; 1704.00648; `agustsson-2017-soft-to-hard-vector-quantization`) | soft assignments annealed to hard (D1 (d)'s source); **the codes initialised by clustering the network's own pre-quantisation outputs, and `sigma_0` from them**; the schedule tied to the gap between the soft and hard losses, or the iteration; "too slow" lets the network invert the soft assignments (growing weights), "too fast" vanishes the gradients | D8 and D9 set `tau` by epoch and in absolute units; the source sets both from the network's outputs. A cheap fix within the built annealing: codes from the sample's `h` (k-means), `tau_0` below `2 lambda_max` |
| Takida et al. 2022, SQ-VAE (ICML; 2205.07547; `takida-2022-sq-vae-self-annealed-stochastic-quantization`) | stochastic quantisation whose variance is **learned**; it falls during training ("self-annealing"), so the quantiser becomes deterministic without a schedule; codebook use without restarts or the commitment's heuristics | the temperature as a parameter, not a schedule: no `tau_0` or halving to set |
| Zhao et al. 2026, Continuous First, Discrete Later: VQ-VAEs Without Dimensional Collapse (2605.06870; `zhao-2026-continuous-first-discrete-later-vq-vae`) | **quantising before the encoder has learned a direction deactivates it for good:** directions below a "water level" stop evolving through the commitment term's interaction with the straight-through gradient; the latent's effective dimension collapses (to 2-5 of 512 at scale) and a loss floor follows that no codebook fix passes. Fix: **an unquantised warm-up**, its length predicted by their theory (an adaptive switch) | the closest account of failure A found: the state's directions are frozen before they mean anything, the commitment the agent, never recovered. It predicts what we saw (the commitment off cures most; the snap alone still collapses two seeds: our snap is on from step 1) and names the next remedy: the state continuous for the first steps, then snapped (Lancucki et al. 2020 also delay quantisation) |
| Tang et al. 2026, StableVQ (2609.26774; `tang-2026-stablevq-stable-vq-tokenizer-training`) | collapse and instability from the **coupling** of encoder and codebook; standard training relies on encoder oscillations to bring codes into use; fixes: a corrected straight-through gradient, a codebook loss over regions, separate learning-rate schedules | the same coupling as the commitment's pull, in tokenizers; separate schedules for codes and network is an idea we haven't tried |
| Mentzer et al. 2024, Finite Scalar Quantization (ICLR; 2309.15505; `mentzer-2023-finite-scalar-quantization`) | each of a few bounded dimensions rounded to a fixed grid; the codebook their product; **no codebook to learn, no commitment loss, no collapse** by construction, competitive with VQ | a factored quantiser with no commitment at all (one axis a dimension); a new kind of component, so the owner's; its grid would fix the codes' spread, which annealing showed matters |
| Zhu et al. 2025, SimVQ (ICCV; 2411.02038; `zhu-2024-simvq-representation-collapse-one-linear-layer`) | collapse from **disjoint optimisation**: only the nearest code gets a gradient; codes as a frozen random basis times one trained linear map, so every code moves | why unused codes never join (our dead codes); a reparameterisation of the codes, a build |
| Fifty et al. 2025, the rotation trick (ICLR; 2410.06424; `fifty-2024-rotation-trick-vector-quantization`) | replaces the straight-through gradient with a rotation and rescaling from the output to its code; unlike straight-through, it **can push points within one code's region apart** | bottleneck 2's form: windows of different target states on one code get one straight-through update, so nothing separates them; this gradient can. A crate change |
| Oliva & Lago-Fernández 2020, Stability of Internal States in RNNs Trained on Regular Languages (2006.10828; `oliva-2020-stability-internal-states-rnn-regular-languages`) | **noise in the recurrent pre-activation** drives units to saturation, into clusters that act as a DFA's states; transitions deterministic, stable on arbitrarily long strings, and **perturbed states return to their clusters**; noise ramped up from 0 works too | a margin-maker other than the commitment (the near ties) and a cure for drift; their companion paper (2005.13971, read 2026-10-15) learned the carry automaton this way |
| Umili & Capobianco 2024, DeepDFA (ECAI; 2408.08622; `umili-2024-deepdfa-automata-learning-probabilistic-relaxations`) | a DFA's probabilistic relaxation as a recurrent network over a bounded set of states, transitions sharpened to one-hot by temperature annealing; robust to label noise | another parameterisation, the state bound built in; its annealing is of the transitions, not of codes in a learned space |
| *re-read:* Wang & Niepert 2019, State-Regularized RNNs (`wang-2019-state-regularized-rnns`) | the soft centroid state at a **fixed `tau` 1**, its centroids **initialised uniformly in [-0.5, 0.5]** | spread wide against `tau`, so below Rose's `T_c`: why their soft centroids don't coincide and ours did |
| *re-read:* Huh et al. 2023, Straightening Out the Straight-Through Estimator (`huh-2023-straightening-straight-through-estimator`) | index collapse traced to the divergence between the commitment's target and the codes; an affine reparameterisation and an alternating ("synchronised") commitment update | already in the 2026-10-15 extension; it names the commitment as a collapse source, as Zhao et al. do |

**What is known, mapped to our findings.**

- **The commitment freezing what isn't learned yet is known** (Zhao et al. 2026's mechanism; Huh et
  al. 2023 and StableVQ's coupling), as is the fix family: quantise later (an unquantised warm-up),
  or drop the commitment (FSQ). Our result, that the commitment off for good certifies 4 of 6, is
  that family's half-step: the snap stays on from the start. Zhao et al. predict the two seeds that
  collapse anyway; their warm-up is the remedy to probe for them.
- **The annealing merge is deterministic annealing above its critical temperature** (Rose 1998,
  Theorem 1), exact and old. The fix is equally known: `tau` below `2 lambda_max` of the outputs'
  covariance at the start, the codes initialised from the outputs (Agustsson et al.; Wang and
  Niepert's wide spread), duplicates perturbed (DA), or `tau` learned (SQ-VAE). D9's choice of an
  absolute `tau` is what failed, as its own con half-said.
- **States lost and thin margins:** noise in the recurrence (Oliva and Lago-Fernández) makes
  states attracting and margins wide without the commitment; the drift it cures is Zeng et al.'s
  (1993). Not found: a study of a state lost **while training continues** on longer words, our
  retention problem; the closest is grokking's "lost solutions" (the 2026-10-15 extension).
- **Codes standing for several states:** the straight-through gradient can't separate windows on
  one code (the rotation trick's motivation); disjoint optimisation keeps unused codes away
  (SimVQ). **Counterexample training (R4) has no gradient-training precedent found:** L*-style
  counterexamples are used to extract automata from trained networks (Weiss et al. 2018; Moore
  machines from transformers, 2410.06045, surfaced, not read), not as training data.

**Candidate remedies, in the order this review would probe them** (each a probe first, its
expectation written before its data; builds are the owner's in the frozen phase):

1. **Annealing with `tau_0` from the outputs' scale** (`tau_0` below `2 lambda_max` of the sample's
   `h`, or the codes initialised from it, and the commitment kept off after hardening, as the never
   probe found). A change to the built annealing (D9 revisited), not a new component.
2. **A continuous warm-up:** the state unsnapped (identity codes, as the continuous network's test
   uses) for the first steps, then snapped with the codes initialised from the states seen. For S3
   seeds 1 and 4. A small harness build.
3. **Noise in the cell's pre-activation** (Oliva and Lago-Fernández) as the margin-maker in the
   commitment's place. A layer change.
4. **The rotation trick** for bottleneck 2, beside R3/R4. A crate change.
5. **FSQ-style fixed-grid factors.** A new kind of component: the owner's call.

Not found: any study of a factored discrete recurrent state learning a finite machine to proof that
reports the commitment's effect; the VQ results are on autoencoders and tokenizers, where the
quantiser's input is a single forward pass, not a state fed back.

## References (arXiv ids; local copies in the literature repository (`davidbarkhuizen/indrajala-literature`, cloned as `~/code/indrajala-literature/`))

1511.08228, 1611.00736, 2003.08494, 2511.22751, 2602.07599, 2406.00075, 1811.12530, 1711.00937,
2609.32447, 2507.02782, 2609.09157, 2609.16372; and, from the window-recurrent workplan,
2207.06881, 2203.07852, 1406.1078, 2108.12409, 2212.10554. The extension: 1711.09576, 1901.08817,
2005.13971, 2111.00600, 1902.10297, 2207.02098, 2210.10749, 2402.04875; Zeng, Goodman & Smyth,
Neural Computation 5(6):976-990, 1993 (doi 10.1162/neco.1993.5.6.976); Das & Mozer, NeurIPS 6,
19-26, 1993; Kolen, NeurIPS 6, 501-508, 1993; Omlin & Giles, Neural Networks 9(1):41-52, 1996.
The architecture's extension: 1909.10893, 2107.02367, 2010.02193, 1904.02920, 2006.01895,
2610.07591; Tiňo & Šajda, Neural Computation 7(4):822-844, 1995 (doi 10.1162/neco.1995.7.4.822).
The state formation extension: 2309.15505, 1704.00648, 2205.07547, 2410.06424, 2411.02038,
2605.06870, 2609.26774, 2006.10828, 2408.08622, 2102.05836; Rose, Proceedings of the IEEE
86(11):2210-2239, 1998 (doi 10.1109/5.726788).
