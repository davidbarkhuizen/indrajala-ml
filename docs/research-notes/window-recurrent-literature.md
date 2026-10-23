# Literature review: learned recurrent adders and length generalisation

**Status: written 2026-10-13, after the window-recurrent matrix (stage 9) and while its control ran,
to judge which findings are new; extended the same day, after the certificate (E1), to the older
literature on discretised recurrent states and automaton extraction, which the first searches
missed. It backed the claims workplan, which planned the work to substantiate them
(retired; its report, [window-recurrent-report.md](window-recurrent-report.md)). Extended
2026-10-15 for the division paper, 2026-10-19 for the architecture's novelty, 2026-10-20 for
state formation and 2026-10-22 for the freeze, restarts and random machines (the sections at the
end).** Every paper read is kept in the literature repository (`davidbarkhuizen/indrajala-literature`, cloned as `~/code/indrajala-literature/`) as PDF and text, under the file
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
| Giles, Miller, Chen, Chen, Sun & Lee 1992, Learning and Extracting Finite State Automata with Second-Order Recurrent Neural Networks (Neural Computation 4:393; `giles-1992-learning-extracting-fsa-second-order-rnn`, read 2026-10-22, supplied by the owner) | a second-order RNN trained on Tomita grammars **partitions its state space into well-separated clusters during training**; a DFA is extracted **after training** by quantising each state unit into `q` intervals, then minimised (Hopcroft and Ullman); different initial weights give different extracted machines that minimise to the same one; the extracted machine can classify long strings better than the network | the origin of extraction by quantising a trained network's state: the clusters form unforced and are read only after; our snap quantises during training and certifies the machine it trains with |
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
problem, as the night of 2026-10-19 left it ([failure-kinds.md](failure-kinds.md); findings to be
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

## Extension (2026-10-22): the freeze, restarts, collapses and random machines

Asked by the owner: review the concepts and findings that arose since the 2026-10-20 extension. The
problem, as the continuous crystallisation (D48-D66, [state-formation-workplan.md](../workplans/state-formation-workplan.md))
left it:

- **The method works on ordered machines, as a batch:** with the margin stop, S3 11 of 16 and
  move-to-front-4 7 of 8 on D55's fresh seeds (D59); the method is statistical (a clear majority
  certifying, not every seed).
- **Every failure ends frozen:** the quasi-static clock (D49) holds at 0 for 738-1,472 epochs; the
  random machines freeze before the state forms, near the factors' melting point (0.54-1.16 of it);
  the coupling, read from the codes' sharpness (D50), keeps quenching while the clock is held; a hold
  in the window between melt and freeze (two runs) did not form. One run thawed after 445 epochs and
  certified (S3 seed 30).
- **The larger machine fails at keeping:** S4 formed, then collapsed at a steady coupling (the screen
  0.99 to 0.42 in one or two epochs), then froze. **Coupling jumps** (moves >= 0.1 an epoch): 58-201
  in every spent failure, at most 11 in certified runs.
- **Success is forming times keeping; size is not the variable** (the scaling conjecture): the random
  machines fail at forming, S4 at keeping; `order` reads a target's order against random machines.
- **The remedies planned:** the reheat (D63 (a), the clock run back) against kill-and-resample (D63
  (e)) on cost per certified machine; the slowed re-cool after each reheat (D66, the forward rate
  times `2^-k`); the paced coupling (D61, a trust region or a Savitzky-Golay line); the order-biased
  cool (D65) set aside; constants derived, never fixed.

Searched: freezing and re-annealing in simulated annealing, constant-thermodynamic-speed schedules,
restart strategies and heavy-tailed run times, warm restarts in SGD, loss of plasticity and
re-initialisation, glassy training dynamics, loss spikes, replica exchange, seed variance in
benchmarks, and the learnability of random automata by statistical queries and gradient descent.
Eighteen papers added to the literature repository (its PR #8; Salamon & Berry 1983, Nulton & Salamon 1988,
Salamon et al. 1988, Schafer 2011, Savitzky & Golay 1964 and Andresen & Gordon 1994 later, #10-#14 and
#18, supplied by the owner), read in the parts that bear on the
above; one re-read.

| paper | what it shows | bearing |
| --- | --- | --- |
| Kirkpatrick, Gelatt & Vecchi 1983, Optimization by Simulated Annealing (Science 220:671; `kirkpatrick-1983-optimization-by-simulated-annealing`) | melt, then cool in slow stages until the system **"freezes" and no further changes occur**; a rapid cool leaves it **frozen into a metastable state**, and the best of several rapid quenches is reported beside the anneal; **specific heat maxima mark the onset of freezing**, two peaks for two kinds of ordering | the vocabulary's source. SA's freeze is the configuration's; ours is the schedule's (the clock held by the gate while the coupling quenches): see "the freeze" below. Peaks per kind of ordering are our cascades (D21) |
| Andresen & Gordon 1993, Analytic Constant Thermodynamic Speed-Cooling Strategies (Open Syst. Inf. Dyn. 2:1; `andresen-1993-constant-thermodynamic-speed-cooling`) | the schedule of constant thermodynamic speed, **`dT/dt = -v T / (eps sqrt(C))`**, with the heat capacity `C` and the **relaxation time `eps`** both estimated from the run's past; the most efficient general schedule (minimal entropy production) | the derived bias of the note on random machines (the halving time from splits per octave times the relaxation time `tau`) is this schedule; D33's adaptive cool (`T exp(-pace / sqrt(C))`) had `C` without `eps`. As `eps` diverges at a kinetic arrest, the schedule stops: such a schedule cannot pass a glass on its own. Their 1994 journal version (PRE 50:4346; `andresen-1994-constant-thermodynamic-speed-entropy-production`, supplied by the owner) derives it from first principles: the optimum keeps entropy production fixed **in the system's own time** (time in relaxation times), constant speed its leading term, for **uniform systems** whose intensive quantities are uniquely defined, which a glass is not |
| Salamon & Berry 1983, Thermodynamic Length and Dissipated Availability (PRL 51:1127; `salamon-1983-thermodynamic-length-dissipated-availability`, supplied by the owner) | for a process near equilibrium, the availability dissipated is at least **`L^2 eps / tau`**: the path's thermodynamic length squared (Weinhold's metric), times a **mean relaxation time**, over the total time; equality at **constant thermodynamic speed** | the derivation behind Andresen and Gordon's schedule and the random-machine note's bias: for a fixed budget, time is spent where the path is long (the octaves where codes split) in proportion to the relaxation time there. As `eps` grows without bound (a glass) the bound does too: no budget passes the arrest by schedule alone |
| Nulton & Salamon 1988, Statistical Mechanics of Combinatorial Optimization (PRA 37:1351; `nulton-1988-statistical-mechanics-combinatorial-optimization`, supplied by the owner) | annealing as a **system following a target** (the equilibrium distribution at the current `T`); the schedule keeps their **discrepancy constant** (statistical distinguishability), which gives **`dT/dt = -v T / (eps sqrt(C))`** (their 3.9): heat capacity and relaxation time from one criterion; optimality offered as a conjecture, supported by early experiments | the closest account of our quasi-static clock (D49): it too reads a lag (the screen against the hold's level) but **stops** at a set lag, where theirs **moves at** a set lag; both stop as `eps` diverges. A constant-lag form of D49 would be their schedule |
| Salamon, Nulton, Harland, Pedersen, Ruppeiner & Liao 1988, Simulated Annealing with Constant Thermodynamic Speed (Comput. Phys. Commun. 49:423; `salamon-1988-simulated-annealing-constant-thermodynamic-speed`, supplied by the owner) | the schedule run on an **ensemble of n parallel copies sharing one `T`**, the speed `v = (E - E_eq) / sigma(E)` read across them (a single run is not ergodic on annealing's time scales); the ensemble **stops representing equilibrium once phase space breaks into more fragments than copies**, expected for glassy problems; on problems without local minima a **direct quench can beat** any equilibrium-following schedule; consistent results on an exactly solvable graph partition | our parallel batch read as one ensemble (the owner's lean: a full melt for parallel runs) would measure the lag across seeds, not within a run; their fragmentation caveat is the freeze seen from the ensemble's side; the quench remark is the order-biased cool's (D65) fast side for ordered machines |
| Ingber 1989, Very Fast Simulated Re-Annealing (Math. Comput. Modelling 12:967; `ingber-1989-very-fast-simulated-reannealing`, the author's draft) | an exponential schedule per parameter, **each with its own annealing time**; **re-annealing**: every hundred or so acceptances, each parameter's annealing time is rescaled from its current sensitivity (`s_i`, the cost's derivative at the best point), so the insensitive parameters are made hotter again | the reheat's precedent, with the run-back derived from the run (no constant) and **per axis**: D63 (a) runs one clock back for all factors; Ingber's form would run each factor's back by its own reading (its sharpness or its share) |
| Luby, Sinclair & Zuckerman 1993, Optimal Speedup of Las Vegas Algorithms (IPL 47:173; `luby-1993-optimal-speedup-las-vegas-algorithms`) | for an algorithm whose answer is always correct when it stops, with known run-time distribution `q`, the optimal strategy is a **fixed cutoff `t*` repeated**, minimising **`l(t) = (t - sum_{t'<t} q(t')) / q(t)`**, the expected cost per success; it stays optimal when runs may be suspended and resumed (Theorem 4); unknown `q`: the universal sequence (1, 1, 2, 1, 1, 2, 4, ...) within `O(log)` of optimal | **a certified run is a Las Vegas run** (when it stops, the machine is proved). D63 (e)'s window is `t*`, derivable from the runs' own `q` per backdrop (D64's and D54-D59's logs): no constant. And D63's comparison is exactly `l` of two algorithms: the reheat is a different algorithm with its own `q` |
| Gomes, Selman, Crato & Kautz 2000, Heavy-Tailed Phenomena in SAT and CSP (J. Autom. Reasoning 24:67; `gomes-2000-heavy-tailed-phenomena-satisfiability-csp`) | randomised search's run times are **heavy-tailed** (Pareto-Levy, possibly no finite mean), with a **left tail** of short successful runs; **rapid restarts remove the heavy tail** and exploit the short runs (up to 100x); censored runs (stopped at a cutoff) handled by tail estimators | our spent runs are censored at the budget; S3 seed 30's thaw is the tail's other side. Read `q`'s tail before choosing `t*` |
| Loshchilov & Hutter 2017, SGDR (ICLR; 1608.03983; `loshchilov-2017-sgdr-warm-restarts`) | **warm restarts**: the learning rate reset high, the weights kept, then annealed again; **periods doubled after each restart (`T_mult` 2)** | D66's `2^-k` is this schedule, triggered by the freeze instead of a fixed period; both keep the weights (a warm, not a cold, restart) |
| Earl & Deem 2005, Parallel Tempering (PCCP 7:3910; physics/0508111; `earl-2005-parallel-tempering-theory-applications`) | replicas at a ladder of temperatures, adjacent ones **swapping configurations** by a Metropolis test; the hottest never trapped; the ladder set so swaps are accepted at a steady rate | for a parallel batch, the alternative to independent seeds: a frozen run's configuration exchanged with a hotter one's. Its exactness (detailed balance) does not carry to SGD, which is no Metropolis chain: a heuristic here, and a batch-level component (the owner's call) |
| Dohare et al. 2023, Maintaining Plasticity in Deep Continual Learning (2306.13812; Nature 2024; `dohare-2023-maintaining-plasticity-deep-continual-learning`) | over a long sequence of tasks, gradient descent **loses plasticity** until it learns no better than a shallow net; correlates: **dead units, growing weights, falling effective rank**; remedy: **continual backpropagation** (low-utility units re-initialised), or L2 with shrink-and-perturb; gradient descent alone is not enough, "a random, non-gradient component" is needed | the frozen run as lost plasticity: the curriculum's growth (D52) is a non-stationary task. Its correlates are measurable offline on our frozen runs (codes in use, `d_eff`, weight norms); the reheat's thermal noise (D17) and a fresh seed are the non-gradient component |
| Ash & Adams 2020, On Warm-Starting Neural Network Training (NeurIPS; 1910.08475; `ash-2020-warm-starting-neural-network-training`) | data arriving in stages: warm-started networks generalise worse than fresh ones; **shrink and perturb** (`theta <- lambda theta + noise`) closes the gap; shrinking **keeps the hypothesis (the argmax) while raising the output's entropy**; shrinking the last layer alone does not suffice | a reheat in weight space: it softens without discarding the machine, as D63 (a) intends with the clock. Our curriculum is their staged data |
| Baity-Jesi et al. 2018, Comparing Dynamics: DNNs versus Glassy Systems (ICML; 1803.06969; `baity-jesi-2018-dynamics-dnn-versus-glassy-systems`) | over-parametrised networks slow down through flat directions, without barrier crossing, unlike glasses; **under-parametrised networks show typical glassy dynamics** (ageing, stuck after a quench) | their variable is parameters against training examples; our words are generated afresh, effectively unbounded against a small network, which is the under-parametrised side: a hypothesis for why runs freeze, untested here |
| Cohen et al. 2021, Gradient Descent ... at the Edge of Stability (ICLR; 2103.00065; `cohen-2021-edge-of-stability-gradient-descent`) | the Hessian's top eigenvalue (their **"sharpness"**) rises to `2/eta`, then hovers there while the loss falls **non-monotonically**, with spikes | a candidate mechanism for coupling jumps and collapses; **their sharpness is not our `sharpness`** (the codes' mean largest soft weight, D50): a name clash to keep apart in writing. We train with Adam, where the slingshot (below) is the closer account |
| *re-read:* Thilak et al. 2022, the Slingshot Mechanism (2206.04817; `thilak-2022-slingshot-mechanism-adaptive-optimizers`) | with adaptive optimisers, after the data is fitted, the **last layer's norm grows, then a loss spike**, in cycles | S4's collapse fits its timing (formed and strict-exact at steps 130-134, then the fall at a steady coupling): untested, because the readout's weight norm is not logged |
| Savitzky & Golay 1964, Smoothing and Differentiation of Data by Simplified Least Squares Procedures (Anal. Chem. 36:1627; `savitzky-1964-smoothing-differentiation-least-squares`) and Schafer 2011, What Is a Savitzky-Golay Filter? (IEEE SPM 28(4):111; `schafer-2011-what-is-savitzky-golay-filter`), both supplied by the owner | a degree-`N` local least-squares fit over `2M + 1` samples, **read at the window's central point**, is a fixed convolution (tabulated integers), and so are its derivatives there; Schafer: the filters for `N` and `N + 1` coincide for even `N`, so **degree 1 centred is a moving average** (P2); the 3 dB cutoff `f_c ~ (N + 1) / (3.2 M - 4.6)` (M >= 25) | D61 (c) reads the degree-1 line **at the newest epoch** (one-sided), which neither paper treats: its offline lag advantage over the moving average (-0.02 to 0.05 against 0.16-0.53) comes from the one-sided reading alone, since centred the two are one filter. The cutoff formula gives a derived window for a chosen smoothness |
| Li et al. 2018, Hyperband (JMLR 18; 1603.06560; `li-2018-hyperband-bandit-hyperparameter-optimization`) | **successive halving**: run many configurations on a small budget, keep the best half, double the budget, repeat; Hyperband hedges over the starting count | an early-stopping rule needs an early signal that predicts success; ours would be `formation` (the state share) or the snapped machine's long answers (answers-only); the freeze-based kill is the other kind (a stall, not a rank) |
| Bouthillier et al. 2021, Accounting for Variance in ML Benchmarks (MLSys; 2103.03098; `bouthillier-2021-accounting-variance-ml-benchmarks`) | the seed, the data's order and the hyperparameters' search each move results markedly; **randomise every source**, compare by `P(A > B)` with percentile-bootstrap intervals, not by means | the statistical method's practice: paired fresh seeds (D64's pair the arms), rates with intervals (Wilson's), never chosen seeds; a further source to randomise would be the data's order |
| Rubin, Seroussi & Ringel 2024, Grokking as a First Order Phase Transition (ICLR; 2310.03789; `rubin-2024-grokking-first-order-phase-transition`) | grokking maps to a **first-order** transition (an equilibrium theory): past it, a **mixed phase** of "droplets" (some units specialised to the teacher, the rest not) | a first-order transition is what can be **supercooled**: cooled past the melting point without nucleating. Our random runs near their melting point without forming are that picture, loosely (no free energy is defined for our training) |
| Angluin, Eisenstat, Kontorovich & Reyzin 2010, Lower Bounds on Learning Random Structures with Statistical Queries (ALT; `angluin-2010-lower-bounds-random-structures-sq`, the Yale report) | random DFAs cannot be weakly learned with polynomially many statistical queries **under an arbitrary (adversarial) input distribution** | does not apply as such: our words are uniform, our answers per step |
| Angluin & Chen 2015, Learning a Random DFA from Uniform Strings and State Information (ALT; `angluin-2015-learning-random-dfa-uniform-strings-state`) | a random DFA **is** learnable in the SQ model from uniform strings labelled with the state reached; a random DFA's diameter is `O(log n)` (Trakhtenbrot and Barzdin) and its random walk mixes fast | richer labels make random machines easy for SQ: our per-step answers sit between accept/reject and the state |
| Giapitzakis, Fountoulakis, Nichani & Lee 2025, SQ Complexity of Learning Semiautomata (2510.04115; `giapitzakis-2025-sq-complexity-learning-semiautomata`) | the first **SQ hardness for semiautomata under the uniform distribution**, from the transition structure alone: distinct random permutation semiautomata's final states decorrelate as the random walk on `S_N x S_N` mixes, needing words of length **`Omega(N^2 ln N)`** | for N = 16 that is about 700 symbols, with final-state labels only; our lengths are under 20 with an answer at every step, so the theorem does not explain random-q16-0's failures; it says what a long-word curriculum would meet |
| Abbe et al. 2021, On the Power of Differentiable Learning versus PAC and SQ Learning (NeurIPS; 2108.04190; `abbe-2021-power-differentiable-learning-pac-sq`) | SGD can always simulate SQ learning; beyond SQ only with fine enough gradient precision against the batch size; coarse precision or large batches make it **no stronger than SQ** | the bridge that makes SQ lower bounds bear on gradient training at all; with the precision caveat, they bound, not predict |

**What is known, mapped to our findings.**

- **The freeze is the known failure of an equilibrium-gated schedule at a kinetic arrest.** Adaptive
  and constant-speed schedules (Huang et al. 1986, Romeo 1989, Andresen and Gordon 1993) advance only
  as fast as the system relaxes; where the relaxation time diverges (a glass) they stop, and our gate
  (D29, D49) holds the clock exactly so. Kirkpatrick et al.'s remedy for a quench frozen into a
  metastable state is a new quench (the best of several), and the field's later answers are three:
  re-anneal (Ingber), restart (Luby, Gomes), or exchange with a hotter copy (parallel tempering). All
  three are D63's options or near them; nothing found suggests a slower cool passes the arrest, which
  matches D65's stage 0. The coupling quenching at a held clock (a glass forming at fixed `T`) has no
  counterpart found: it is a property of our coupling's reading (D50), not of annealing.
- **Restart theory gives D63 (e) its constant from the data.** A certified run is a Las Vegas run, so
  Luby et al.'s `l(t)` on each backdrop's run-time distribution gives the cutoff `t*` (and, for a
  heavy tail, the gain); the reheat changes the distribution itself, and the comparison is `l` against
  `l`. The owner's caution (the reheat may reach solutions that resampling discards) is outside
  Luby's model, which only counts time to the first success: read the reheat's certificates apart,
  as D63 already plans.
- **Warm restarts with doubling periods are standard** (SGDR), the slowed re-cool's (D66) schedule;
  **per-axis re-annealing from current sensitivities** (Ingber) was considered and dropped (item 4
  below: our factors fail together).
- **Lost plasticity is a known end of long gradient training on changing data**, with measurable
  correlates and a non-gradient remedy (re-initialisation, shrink and perturb); whether our frozen
  runs show its correlates is an offline reading on existing networks.
- **Collapses after a fit, under Adam, are the slingshot's signature** (re-read), the edge of
  stability its full-batch cousin; neither is tested here, for want of the readout norm in the logs.
  D61 smooths the coupling's reading; if a collapse is an optimiser instability, D61 cannot cure it.
- **Random machines:** theory proves random automata hard for SQ (and so for coarse gradient
  methods) only with adversarial distributions or very long words and final-state labels; with state
  labels they are easy. Our failures are at short lengths with dense answers, where no result found
  applies: the difficulty of random-q16-0 is not explained by the theory found, and the scaling
  conjecture's reading (forming fails, the cascade crowded) stands as our own.

**Measurements and options this review points to** (no new experiment before the planned ones end;
the offline items may run any time):

1. **Luby's `l(t)` per backdrop, offline**, on the completed runs' step ends (D54-D59, D64 once
   read), the spent runs as censored: `t*` for D63 (e) and the curve D63 compares against. A durable
   `population` subcommand.
2. **The readout's weight norm, logged each epoch** (with Adam's state if cheap), so a collapse can be
   tested for the slingshot; built into the next change that touches the per-epoch row.
3. **Plasticity's correlates on the frozen runs, offline:** codes in use, `d_eff` (stage 2's
   `state-spread`), weight norms, on the saved networks where they exist.
4. ~~Ingber's per-factor run-back~~ **dropped (the owner, 2026-10-22):** read offline the same day
   on nine frozen runs (random-q16-0 0-3, random-q12-0 1-3, S3 31) and S4's two collapses, no run
   formed one factor while another stayed stuck: the factors' shares fall together and stay equally
   low; only their live readings diverge (one factor quenched hard while another is liquid). The
   evidence does not support a per-factor reheat.
5. **Parallel tempering for a batch:** a new component, the owner's call.

Surfaced, not read: Nikishin et
al. 2022, the primacy bias and resets (2205.07802); Nourani & Andresen 1998
(constant speed against other schedules; since added to the literature repository, its #9, unread); Mitarchuk et al. 2024 (gradient descent's limits for
finite-precision recurrent networks, TMLR). Kontorovich 2009 (random DFAs PAC-learnable, 0907.0453)
was withdrawn for an error.

Not found: a study of a cooling schedule gated on equilibrium stalling at a kinetic arrest in neural
training; restart theory applied to training runs that end in a proof; any SQ or gradient result for
transducers with an answer at every step at short lengths.

## References (arXiv ids; local copies in the literature repository (`davidbarkhuizen/indrajala-literature`, cloned as `~/code/indrajala-literature/`))

1511.08228, 1611.00736, 2003.08494, 2511.22751, 2602.07599, 2406.00075, 1811.12530, 1711.00937,
2609.32447, 2507.02782, 2609.09157, 2609.16372; and, from the window-recurrent workplan,
2207.06881, 2203.07852, 1406.1078, 2108.12409, 2212.10554. The extension: 1711.09576, 1901.08817,
2005.13971, 2111.00600, 1902.10297, 2207.02098, 2210.10749, 2402.04875; Giles et al., Neural Computation 4(3):393-405, 1992; Zeng, Goodman & Smyth,
Neural Computation 5(6):976-990, 1993 (doi 10.1162/neco.1993.5.6.976); Das & Mozer, NeurIPS 6,
19-26, 1993; Kolen, NeurIPS 6, 501-508, 1993; Omlin & Giles, Neural Networks 9(1):41-52, 1996.
The architecture's extension: 1909.10893, 2107.02367, 2010.02193, 1904.02920, 2006.01895,
2610.07591; Tiňo & Šajda, Neural Computation 7(4):822-844, 1995 (doi 10.1162/neco.1995.7.4.822).
The state formation extension: 2309.15505, 1704.00648, 2205.07547, 2410.06424, 2411.02038,
2605.06870, 2609.26774, 2006.10828, 2408.08622, 2102.05836; Rose, Proceedings of the IEEE
86(11):2210-2239, 1998 (doi 10.1109/5.726788).
The freeze and restarts extension: 1608.03983, 1603.06560, 1803.06969, 1910.08475, 2103.00065,
2103.03098, 2108.04190, 2306.13812, 2310.03789, 2510.04115, physics/0508111; Kirkpatrick, Gelatt &
Vecchi, Science 220(4598):671-680, 1983; Ingber, Mathematical and Computer Modelling 12(8):967-973,
1989; Salamon & Berry, Physical Review Letters 51(13):1127-1130, 1983;
Nulton & Salamon, Physical Review A 37(4):1351-1356, 1988; Schafer, IEEE Signal Processing Magazine
28(4):111-117, 2011; Savitzky & Golay, Analytical Chemistry 36(8):1627-1639, 1964; Salamon, Nulton, Harland, Pedersen, Ruppeiner &
Liao, Computer Physics Communications 49:423-428, 1988;
Andresen & Gordon, Open Systems & Information Dynamics 2(1):1-12, 1993, and Physical Review E
50(6):4346-4351, 1994; Luby, Sinclair &
Zuckerman, Information Processing Letters 47(4):173-180, 1993; Gomes, Selman, Crato & Kautz, Journal
of Automated Reasoning 24:67-100, 2000; Angluin, Eisenstat, Kontorovich & Reyzin, ALT 2010 (LNCS
6331:194-208); Angluin & Chen, ALT 2015:119-133.
