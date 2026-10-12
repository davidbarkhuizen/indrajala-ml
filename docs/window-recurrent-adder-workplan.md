# Workplan: a window-recurrent adder (milestone A, revised)

**Status: decisions W1-W10 settled by the owner (2026-10-12), in dialogue; W11-W20 settled under
the owner's standing delegation. Stage 1 (this plan) merged (#685); stage 2 (specs) merged (#686);
stage 3 (numpy layers) merged (#687); stages 4-7 (the networks, fixtures and golden entries) done;
W21-W29 settled at their stages; stage 8's harness done, its calibration next. It replaces
`docs/length-generalisation-workplan.md`, which the owner terminated before its existence study
(next-steps, From length generalisation).**

## The question

Can a network that reads a fixed-width window of columns and hands a small state from window to
window learn, from sums of up to a few windows, to add numbers of any length? The length lives in
the number of steps the same window-and-cell run, never in a position a network must tell apart.
Composing a fixed-length adder outside the network, the carry handed over in code, would add any
length today (the addition sweep's `n = 6` adder, its third operand as the carry in); it is
parked (the parent plan's Parked), because the question is whether the network learns it.

## Why this design

From the owner's discussion of the terminated plan (2026-10-12):

- **The trap of position codes.** A network with codes of fixed dimension and bounded magnitude,
  at finite precision, can't give an unbounded set of positions codes that a fixed comparison tells
  apart by a margin that doesn't shrink: an infinite sequence in a compact set has points
  arbitrarily close (Bolzano-Weierstrass). A scheme either wraps (rotary's periodic angles alias
  far distances) or converges (no positions recovers position as about `1 / t`). And softmax
  dilutes a lookup's weight as distractors multiply, unless far tokens are pushed toward `-inf`
  (ALiBi's additive bias, Press et al. 2022), not merely scaled toward 0 (xPos's multiplicative
  decay, Sun et al. 2022). A fixed-depth network that finds the deciding column in one long hop
  meets both.
- **The task's structure.** Every column applies the same rule (translation invariance), and the
  carry has three values: each column sum `s` in 0..6 maps carry-in to carry-out by one of seven
  order-preserving maps of {0, 1, 2} (sums 0, 3 and 6 constant; 1, 2, 4 and 5 not), so addition is
  a 3-state automaton run over the columns. Cut the columns anywhere: all the upper part needs from
  everything below is one value of three.
- **So: a window and a state.** Inside a window, positions run only to `k`, a learned `Position`
  never extrapolates; across windows the state needs three distinguishable values. Both are finite,
  so neither meets the trap. What replaces it is the state's stability over more steps than
  training showed, which a discrete state (W4) makes exact.

## What the literature found

Read for this plan (local copies kept outside the repo; arXiv ids given):

| paper | what it does | what it means here |
| --- | --- | --- |
| Bulatov et al. 2022, Recurrent Memory Transformer (2207.06881) | memory tokens read at a segment's start and written at its end, passed to the next segment; backpropagation through 0-4 segments | the token-based state (W1's option (a), not chosen); solves copy and reverse to 9 segments, trained and tested at the same segment counts: the sections read report no extrapolation in segment count |
| Hutchins et al. 2022, Block-Recurrent Transformers (2203.07852) | a recurrent cell over blocks: state vectors updated with LSTM-style gates, cross-attending to the block's tokens | the gated cell (W1b-iii); **a known failure: the model learns to ignore the state and doesn't recover**, avoided by gate biases initialised to "remember" (input gate -1, forget gate +1): W13 |
| van den Oord et al. 2017, VQ-VAE (1711.00937) | a vector snapped to the nearest of `K` learned codes, a straight-through gradient; codes learned by `‖sg[z] - e‖²` (or moving averages, γ 0.99), the encoder held to them by a commitment loss `β ‖z - sg[e]‖²`, β 0.25, robust from 0.1 to 2 | the discretiser (W4): W14 |
| Cho et al. 2014, the GRU (1406.1078) | reset and update gates, `r = σ(W_r x + U_r h)`, `z = σ(W_z x + U_z h)`, `h̃ = tanh(W x + U (r ⊙ h))`, `h' = z ⊙ h + (1 - z) ⊙ h̃`; "not able to get meaningful results" with a plain tanh unit | the cell (W1b-iii); `tanh` is new to the package: W15 |
| Press et al. 2022, ALiBi (2108.12409); Sun et al. 2022, xPos (2212.10554) | a linear additive distance bias, trained at 1,024 tokens, run at 2,048; rotary with an exponential decay, blockwise attention | the escape from the position trap for fixed-depth networks (above); not needed inside a window |

The terminated plan's reading (Zhou, McLeish, Cho 2024, Fan, Kazemnejad, Ruoss, Allen-Zhu & Li;
`git show c3d619b:docs/length-generalisation-workplan.md`, What the literature found) stands: the
adders that extend write one digit per pass or loop a block; Fan's looped transformer raises its
loop count by a curriculum, as W6 raises the window count.

## The network

Per window `w` of `k` columns, with state `c_w` (width `s`):

1. The window's column tokens (`Embedding`, then a learned `Position` over `k`, W7).
2. A causal transformer of `L` blocks (W11); before each block, `P_l c_w` is added to every token
   (W1b-ii, one projection `P_l` of `(d, s)` per block).
3. The token-wise softmax output: the window's `k` sum digits.
4. A learned attention pooling over the last block's tokens gives the window's summary `u_w`
   (W1b-i: one learned query, one head).
5. A GRU cell: `h = GRU(c_w, u_w)` (W1b-iii); then `c_{w+1}` = the nearest of `m` learned codes to
   `h`, straight-through (W4).

`c_0` is a learned vector snapped like any state (W8). The weights are shared across windows; the
gradient flows back through every window (W5). An `n`-column sum runs `ceil((n + a) / k) + 1`
windows: `a` random zero columns below (W6-i), and one window of zero columns whose bottom digit is
the final carry (W9).

## Decisions

Settled by the owner (2026-10-12):

- **W0. Window size `k`. Settled: a dimension of the study's matrix**, not a choice made here.
- **W1. How the state is carried. Settled: (b).**
  - (a) State tokens in the window, written and read through attention (RMT). Pros: no new layer.
    Cons: the state rides every block's residual stream, the most exposed to drift; it needs a
    position of its own.
  - (b) *Chosen.* A separate recurrent cell. Pros: a short, explicit path for the state, whose
    stability can be controlled and inspected. Cons: new layers, and two interfaces to design
    (W1b-i, W1b-ii).
  - (c) Input injection alone. Cons: says how the state enters, not how it is made.
- **W1b-i. The window's summary for the cell. Settled: (c), a learned attention pooling.**
  - (a) The top column's output token. Pros: no new layer; in a causal window it has seen every
    column. Cons: one token does two jobs, the top digit and the summary.
  - (b) The mean over the window's tokens. Cons: dilutes the top column by `1 / k`; order lost.
  - (c) *Chosen.* One learned query attends over the window's output tokens. Pros: the network
    chooses what to read, at any `k`; frees the top token. Cons: a new layer in three
    implementations and the crate.
- **W1b-ii. How the state enters the window. Settled: (c), every block's input.**
  - (a) Added to every token's embedding. Pros: one projection. Cons: depth can wash it out.
  - (b) The bottom column's embedding only. Cons: upper columns fetch it through attention.
  - (c) *Chosen.* A projection of the state added to every block's input. Pros: the strongest
    signal, never washed out; each block reads it its own way. Cons: one projection per block,
    the most new wiring.
- **W1b-iii. The cell. Settled: (b), a GRU.**
  - (a) A small MLP. Pros: existing layers. Cons: no gate to hold a value.
  - (b) *Chosen.* A gated cell (GRU). Pros: the standard defence against drift and vanishing
    gradients; it can hold a state through windows that only propagate. Cons: a new layer and a
    new activation (`tanh`, W15) in three implementations and the crate.
  - (c) A plain `tanh` RNN cell. Cons: vanishing and exploding gradients; Cho et al. got no
    meaningful results without gates.
- **W2. The state's width `s`. Settled: a dimension of the matrix.**
  - (a) `d`. Cons: far more capacity than three values; room to drift. (b) Small and fixed. Cons:
    a guess. (c) *Chosen.* Varied. Pros: shows whether capacity matters. Cons: the matrix grows.
- **W3. Supervising the state. Settled: (a), unsupervised**: only the sum digits are targets.
  - Pros: the network must discover what to carry, the real test. Cons: slower, less reliable
    training; the state's meaning found only by probing. (b), an auxiliary loss on the true carry,
    and (c), annealed supervision, were declined: they hand over the decomposition.
- **W4. The state's stability. Settled: (c), discretised** to the nearest of `m` learned codes,
  straight-through.
  - Pros: exact over any number of steps; drift is impossible. Cons: new machinery; a gradient
    approximation; `m` a knob; it imposes a finite-state answer (W4-i keeps `m` free).
  - (a) The gate alone, (b) a `LayerNorm` on the state, (d) (a) then (c) if runs drift: declined.
- **W4-i. The number of codes `m`. Settled: a dimension of the matrix.** Pros: shows how much
  the finite-state hint matters, and whether the codes used converge to three. Cons: the matrix
  grows. The report counts the codes each model uses.
- **W5. Training through the windows. Settled: (a), full backpropagation through all windows.**
  Pros: exact; a few windows at these lengths. Cons: cost linear in windows. (b) Truncated, and
  (c) none, were declined: (c) gives an unsupervised state no signal at all.
- **W6. The regime. Settled: (b), a curriculum by window count**: 1 window, then 2, then 3, ...,
  each step training on every count so far (cumulative), warm-started.
  - Pros: exercises the recurrence step by step at every `k` (Fan et al.'s loop curriculum).
    Cons: tied to `k`, so digit lengths differ across the matrix; new code beside the addition
    study's lengths.
  - (a) The terminated plan's curriculum by digits: barely varies the window count at large `k`.
    (c) All at once: no curriculum; kept as the control (W16).
- **W6-i. Random alignment. Settled: (a), yes**: each example starts `a` columns into its first
  window, `a` uniform in 0..`k - 1`, the columns below zero. Pros: a carry crosses a window
  boundary at every column of the number. Cons: padding; a step's digit range varies.
- **W7. Positions inside a window. Settled: (a), a learned `Position` over `k` columns.** Pros:
  a fixed table never extrapolated; the existing layer. Cons: `k` rows, so models of different
  `k` share no weights. (b) Rotary and (c) none: no benefit at a small fixed `T`; (c) meets Cho's
  one-layer impossibility.
- **W8. The initial state. Settled: (a), a learned vector, snapped like any state.** Pros: the
  network chooses what "no carry yet" is. Cons: early flips between codes through the
  straight-through gradient. (b) A designated code and (c) zeros were declined.
- **W9. The final carry. Settled: (b), one more window of zero columns**, its bottom digit the
  final carry. Pros: every window the same shape and job. Cons: one extra step. (a) An extra top
  digit in the last window and (c) a readout from the final state were declined.
- **W10. What "works" means. Settled: (c), both**: the break length in columns, the largest
  test length at which it and every shorter one pass, works at 16 or beyond (the terminated
  plan's D6); and beside it the ratio of windows run at the break to the most windows trained.

Settled under the owner's standing delegation:

- **W11. The window's transformer. Settled: (a).**
  - (a) *Chosen.* `d` 16, 3 blocks (attention and FFN), 2 heads, FFN width 32, causal, fixed.
    Pros: the addition sweep's 3 layers carried the whole distance at `n = 6` at `d` 6-8, so 3
    blocks cover any in-window carry up to `k` 6; `d` 16 leaves room for the state's projection;
    one size keeps the matrix to `k`, `s` and `m`. Cons: at `k` 1 or 2 most of it is idle.
  - (b) A fourth matrix dimension. Cons: the matrix's cost, before anything is known to work.
- **W12. The matrix's values and seeds. Settled at stage 8, from the calibration's cost.**
  Provisional: `k` in {1, 2, 4}, `s` in {4, 16}, `m` in {4, 8, 16}, 5 seeds: 90 runs, and the
  control (W16). Pros: settles the grid on measured cost, not a guess. Cons: one more decision
  after the code.
- **W13. Gate initialisation. Settled: (a).**
  - (a) *Chosen.* The GRU's update gate biased toward keeping the old state at initialisation
    (`b_z` +1), its weights drawn small as a dense layer's. Pros: Hutchins et al.'s fix for the
    cell-ignored local optimum, in the GRU's terms. Cons: a constant to justify.
  - (b) Biases at zero. Cons: the failure Hutchins et al. report, from which training doesn't
    recover.
- **W14. The discretiser's training. Settled: (a).**
  - (a) *Chosen.* VQ-VAE's losses: the codes by `‖sg[h] - e‖²`, the cell held to them by
    `β ‖h - sg[e]‖²`, β 0.25, both averaged over the windows and added to the digits' loss. Pros:
    the published form, robust to β. Cons: two extra loss terms, and a code no state lands near
    gets no gradient (a dead code); the report counts codes used, and a reset of dead codes is
    the first fix if `m` above 3 leaves most unused.
  - (b) Moving-average codes (γ 0.99). Pros: no code loss. Cons: a second update rule outside
    the optimizers, and state the optimizer doesn't own.
- **W15. `tanh`. Settled: (a).**
  - (a) *Chosen.* `tanh(x) = 1 - 2 / (exp(2x) + 1)`, through each implementation's `exp`, as
    `sigmoid` is. Pros: no new transcendental: the three implementations agree where `exp` does,
    and tests swap `exp` as they do now. Cons: not the library's `tanh`; the form loses relative
    precision near 0, harmless for a gate's candidate.
  - (b) Each library's `tanh`. Cons: not correctly rounded, so the three implementations differ in
    the last bit (rotary's lesson, its shared table).
- **W16. Controls. Settled: (a).**
  - (a) *Chosen.* One control arm: the same network trained on every window count at once (W6's
    (c)), at the matrix's best cell, with the curriculum's examples (the terminated plan's D10:
    every run records its examples and epochs per step). Pros: whether the curriculum mattered,
    for one more set of seeds. Cons: only at one cell.
  - (b) The control at every cell. Cons: doubles the matrix.
- **W17. The catalogue at long lengths. Settled: the terminated plan's D8**: sampled, with the
  carry strata (M3 every distance `L` from 1 to `n`, a fixed count each; M5 sampled past `n = 6`;
  B3 at the training lengths only), and test lengths 8, 10, 12, 16, 24, 32, 48, 64.
- **W18. Where the state enters, as data. Settled: (a)** (stage 2).
  - (a) *Chosen.* A layer spec, `StateInput`, standing among the window's tokens right before a
    residual block, its weights `[P]`, `(d, s)`. W1b-ii's (c) is one before each block that reads
    the state; W11's window has one before each of its 3 attention blocks, the `P_l` of The
    network. Pros: where the state enters is explicit, validated, and saved as a layer like any
    other, with no wiring special to the network; W1b-ii's (a) is the same spec, once. Cons: a spec
    kind only a window may hold, so the token part's validation takes a window flag.
  - (b) A field of the recurrence ("every block"), the network inserting the projections. Pros: the
    window's specs stay a plain sequence model's. Cons: the placement is hidden in the network's
    code; another placement is a new field value and new code; `P` lives outside the layers'
    weights and optimizer state.
- **W19. The pooling's form. Settled: (a)** (stage 3).
  - (a) *Chosen.* `s_t = q · x_t`, `p = softmax(s)`, `u = Σ_t p_t x_t`, over the tokens the output
    layer reads; `q` starts at zero, so the pooling starts as the mean, and draws nothing. Pros: no
    parameter without a function: a key projection only reparametrises `q` (`q · (W_k x + b_k)`,
    whose `b_k` term cancels in the softmax), and a value projection is absorbed by the GRU's `W`.
    Cons: no `1 / sqrt(d)` scale, harmless with a learned `q`; one query reads one mixture.
  - (b) A full attention head (`q`, `W_k`, `b_k`, `W_v`, `b_v`). Pros: the familiar form. Cons: the
    same functions with more parameters, and more code in three implementations and the crate.
- **W20. Backpropagation through the windows (W5's how). Settled: (a)** (stage 3, used at stage 4).
  - (a) *Chosen.* Recompute: the forward pass keeps only each window's state and generator state;
    the backward pass runs from the last window, re-running each window's forward pass (its
    generator restored, so its dropout masks repeat) before its backward pass. Pros: every
    existing layer runs unchanged in all three implementations, each keeping one pass's values;
    memory of one window's activations. Cons: a training step runs each window forward twice.
  - (b) A copy of the window's layers per window, sharing parameters. Pros: no recompute. Cons:
    parameters shared across layer objects (the optimizer, snapshots, Rust's rebinding), and the
    copies' count changes with the curriculum's window count.
  - (c) Layers that hand back their saved values. Cons: every layer rewritten in three
    implementations.
- **W21. The Rust parts. Settled: (a)** (stage 5).
  - (a) *Chosen.* New crate ops, one call per pass of each part (`state_input_*`,
    `attention_pooling_*`, `gru_*`, `codes_*`), composed of the crate's products, `array_softmax`
    and numpy's elementwise grouping, and `concatenate_columns`. Pros: one boundary crossing per
    pass, no per-example Python loops; attention's pattern; the codes' ops numpy's by bits. Cons:
    a crate PR, and more Rust to test.
  - (b) The parts composed in Python from `pa.Array`'s existing ops. Pros: no crate change. Cons:
    `Array` is 2-D with no concatenation, no reduction but `sum_axis0` and no broadcast over
    tokens, so the pooling and the StateInput would loop over the examples in Python, slower than
    numpy.
  - (c) Crate ops for the token-shaped parts only, the GRU composed. Cons: the GRU's gradient
    blocks need concatenation or slice writes, which `Array` lacks: a crate change anyway.
- **W22. How pure Python runs a window. Settled: (a)** (stage 6).
  - (a) *Chosen.* Every window through the layer-major batch path (`layer_major.py`), given two
    small hooks: `begin_example`, so a StateInput adds its example's state, and a gradient beside
    the output layer's, the pooling's. Pros: one path, with or without dropout; its masks are
    numpy's (N, size) order, so seeded training matches numpy's draws; the batch is at hand for
    the parts, which work on batches as numpy's do. Cons: the lanes' copying makes it slower than
    the example-major loop (pure Python isn't timed).
  - (b) The example-major loop, layer-major only with dropout, as the other pure-Python networks.
    Pros: the faster path without dropout. Cons: two paths through each window to keep in step,
    and the parts would still need the batch's rows gathered from the per-example passes.
  - (c) A loop of the network's own over the window's layers. Cons: a second copy of the
    layer-major path's residual and batch-norm handling.
- **W23. The alignment in a check. Settled: (a)** (stage 8).
  - (a) *Chosen.* Each case's own, from the high half of a hash of the triple and `n` (the low
    half decides D8's tenth), the same for every model and every order of the triple. Pros: across
    a property's cases a carry crosses a window boundary at every column; deterministic, so every
    model meets the same inputs; A1 still tests only the order. Cons: each case is seen at one
    alignment only.
  - (b) Every alignment for every case. Pros: no case escapes a bad alignment. Cons: `k` times the
    check's cost, for what (a) covers across the cases.
  - (c) `a` = 0. Cons: the boundaries fall at the same columns in every case: a carry across a
    boundary is tested only there.
- **W24. A step's check. Settled: (a)** (stage 8).
  - (a) *Chosen.* The catalogue at each data window count's longest length, `n = w k` for `w` in
    the step's counts; held-out triples (D8, B3) only from `n = 3`. Pros: one length per window
    count, so `k` 4's third step checks 3 lengths, not 12; the test pass covers the lengths in
    between. Below 3 a held-out tenth can't pass at all: at `k` 1 the first step trains on `n` 1
    alone, and a held-out triple's column token would never be seen. Cons: a length between the
    counts' longest is checked only through the test pass's sampled cases.
  - (b) Every training length 1 to `j k`. Cons: at `k` 4 four times the check's cost, mostly on
    lengths the longest's alignment already covers.
  - (c) The longest length only. Cons: a step could pass while forgetting a shorter count.
- **W25. Adam across the curriculum's steps. Settled: (a)** (stage 8).
  - (a) *Chosen.* Carried: the network and its optimizer continue; each step's saved file holds
    Adam's moments, so a resumed step continues the same. Pros: no loss spike at a step; no extra
    code. Cons: the moments were estimated on fewer windows.
  - (b) Fresh at each step (the terminated plan's D7, chosen when format 2 held no optimizer
    state). Cons: a spike the patience must absorb; a choice made for a limit since removed.
- **W26. An epoch's examples and batches. Settled: (a)** (stage 8).
  - (a) *Chosen.* An equal share of the epoch per data window count (the terminated plan's D5);
    within a count, `a` uniform, `n` uniform over the lengths that fill that many windows at `a`,
    and D7's three kinds in turn; batches each of one window count (the network runs one count
    per batch), their order shuffled. Pros: each count trained as much as the others; no padding.
    Cons: within a count, short lengths are as likely as long ones.
  - (b) Lengths drawn uniformly over 1 to `j k`. Cons: a count's share grows with `k`'s lengths in
    it, unevenly at small counts.
- **W27. A step's stop. Settled: (a)** (stage 8).
  - (a) *Chosen.* The addition study's rule, per step: a screen each epoch (every k-th case of
    each check), and on a passing screen the whole check; else a plateau (`patience` epochs
    without more screen cases passed) or the budget (`max_epochs`). Pros: proven in the addition
    sweep; a step that can't pass stops. Cons: a step that plateaus moves on unlearnt (recorded:
    each step's `stop` and `passes`).
  - (b) A fixed epoch count per step. Cons: wastes epochs after passing, or stops short.
- **W28. The codes used. Settled: (a)** (stage 8).
  - (a) *Chosen.* The codes every state snaps to over B1's first 1,000 cases at each check and
    test length (`state_codes`, a forward pass of its own). Pros: every length's states, at a
    small fixed cost. Cons: a code used only by rarer inputs could be missed.
  - (b) Over every case of the test pass. Cons: a second forward pass over every case, doubling
    the test pass.
- **W29. The catalogue past `n` = 6. Settled: (a)** (stage 8).
  - (a) *Chosen.* Every property whose space passes the exhaustive limit is sampled: A3 and M1
    too (their spaces are `3 * 3^n` and `4 * 3^n`; W17 kept them "as now", which `n` 64 can't
    afford), and M5 past 6; the samples' sizes are settings (B1-B3, A3, M1 and M5's count, A1 and
    A2's, M3 and M4's per stratum), set by the calibration. Up to `n` 6 with the default sizes every
    case is the addition study's (checked). Pros: any length affordable; the strata W17 names
    unchanged. Cons: A3 and M1 sampled from `n` 12 and 11 (exhaustive until then, the larger
    share of a check at 10 and 11).
  - (b) A3 and M1 exhaustive. Cons: impossible past `n` about 12.

## Stages

1. **This workplan**, with the length-generalisation workplan terminated (next-steps), the parent
   plan revised and the roadmap's exclusion of recurrent networks removed. Docs only.
2. **Specs.** *Done.* A window-recurrent network's spec: the window's token specs (as today), `k`,
   the state width, the cell, the pooling, `m` and the code losses' β; validation; format 2's
   entry (a shape "window_recurrent"); every builder refuses it "not yet" until its stage. Tier 0.
   Built as: the window's specs with `StateInput`s (W18), over the input `(k,)`, beside a
   `WindowRecurrence(state_size, Codes(count, commitment), GRU(update_bias), AttentionPooling())`
   (`model/specs/window_recurrence.py`), checked by `validate_window_recurrent`; format 2 adds
   `"recurrence"` to a "window_recurrent" file; the builders and `load_network` refuse it.
3. **numpy, the new layers**: *Done.* `tanh` (W15), the GRU cell, the attention pooling, the
   discretiser with its straight-through gradient and losses, the per-block state projection; each
   with a gradient check (the discretiser's against its straight-through definition). Tier 0: no
   timed path changes. Built as `model/layers/numpy/recurrent_array_layer.py`: the
   `StateInputArrayLayer`, which the numpy builder builds given a window's `state_size`, and the
   network's own parts (`AttentionPoolingArrayLayer`, `GRUArrayLayer`, `CodesArrayLayer`), each
   with `forward_batch`, `backward_batch` and `draw`, keeping one pass's values (W20).
4. **numpy, the network**: *Done.* Windows with random alignment, the state through the windows,
   the final window, backpropagation through all windows, `learn_batch`, save and load; a gradient
   check through two and three windows; with the state's projections at zero and one window, the
   network's digits are the plain sequence network's by bits, so the wiring is checked apart from
   training. Tier 0. Built as `WindowRecurrentArrayNetwork`
   (`model/networks/window_recurrent_array_network.py`): any window count per batch, the
   same for every example; the loss per example the mean of every column's cross-entropy plus 1 /
   W of each snap's code losses (the initial state's and each window's); W20's recompute, the
   generator restored per window. The gradient check, through 1, 2 and 3 windows, makes the codes
   the identity (the straight-through surrogate); the codes' terms are checked through the network
   apart. Format 2 names the network's own parts after the window's layers
   (`RECURRENCE_PARAMETERS`); `load_network` loads a numpy file. Aligning the columns (W6-i) and
   the final zero window (W9) are the caller's: the harness's, at stage 8.
5. **The crate, then the Rust layers and network**: *Done.* Any op the layers need (the GRU's and
   the pooling's batch ops), indrajala-math-rust first, then a "Bump rust/" PR; parity with numpy.
   Tier 0 for existing paths: their ops unchanged (an A/B only if a shared op changes). Built as
   the crate's `src/recurrent.rs` (indrajala-math-rust #56, W21): one call per pass of each part,
   and `concatenate_columns`; then `model/layers/rust/recurrent_rust_array_layer.py`, and the
   network as a shape mixin over either backend, `WindowRecurrentShape`
   (`model/networks/window_recurrent_array_network.py`), whose parts are the backend's:
   `WindowRecurrentArrayNetwork` on numpy and `WindowRecurrentRustArrayNetwork` on Rust. The two
   draw the same weights by bits, and training keeps numpy's weights within 5.8e-13 of each
   entry's scale after 20 steps (the products' rounding, the crate's against BLAS).
6. **Pure Python**: *Done.* The same, parity with numpy. Tier 0. Built as
   `model/layers/python/recurrent_layer.py`: the `StateInputLayer`, which the pure-Python builder
   builds given a window's `state_size`, and the network's own parts (`AttentionPoolingLayer`,
   `GRULayer`, `CodesLayer`, `InitialStateLayer`) on batches of rows, their weights weight sets
   the optimizer steps as a layer's; then `WindowRecurrentBackpropNetwork`
   (`model/networks/python/window_recurrent_backprop_network.py`), which runs every window through
   the layer-major batch path (`layer_major.py`, which now tells a layer each example's index and
   takes the pooling's gradient beside the output layer's), so its dropout masks are numpy's. It
   draws numpy's weights by bits; training keeps numpy's weights within 4.4e-13 of each entry's
   scale after 20 steps, and a step's gradients within 3.9e-15 (pure Python's folds against
   BLAS); the codes and the initial state are numpy's by bits. `load_network` loads all three
   implementations' files, and no builder refuses a window any more.
7. **Fixtures, checkpoints and golden entries**: *Done.* The new network in each implementation;
   the 118 earlier golden entries bit-identical. Built as a saved-model fixture per implementation
   (`tests/saved_model_fixtures.py`'s `WINDOW_MODEL`, a dropout and two StateInputs over 3 columns
   beside a recurrence with every field off its default, its states two windows), which loads,
   resumes, re-saves byte-identically and loads across numpy and Rust
   (`tests/model/persistence/test_window_recurrent_format2.py`, `test_format2.py`); checkpoints
   that resume training by bits under every rule, also as nested lists across a worker boundary,
   and a run file that round trips and refuses another recurrence; and three golden entries
   (121 in all), `WINDOW_SPECS` over 2 columns on the sequence rows read as 3 windows, with a
   dropout whose masks repeat in each window's recomputed pass.
8. **The harness**: the window-count curriculum, random alignment, the catalogue across lengths
   (W17), the report (break length in columns and the window ratio, `L*` per test length, codes
   used, examples and epochs per step); the result's `crate` field hashing the extension module,
   not `__init__.py` (the terminated plan's stage 7 fix). Then a **calibration on `pyramidon`**:
   one seed per `k` through two curriculum steps, timing an epoch and the test pass at 64, and W12
   settled from it. Tier 0. The harness *done* (W23-W29), built as
   `indrajala_ml/studies/window_recurrent_study.py` with `scripts/window_recurrent_study.py`
   (`run`, `status`, `report`, `check`): a run is an arm, `k`, `s`, `m` and a seed on the Rust
   network; each step is saved with its network as it finishes, so a stopped run resumes at its
   next step; the report gives per cell the seeds working and per step the seeds passing and the
   break lengths, per run and step the stop, epochs, examples, break, window ratio, `L*` at each
   test length and codes used, and the cost (seconds per epoch training and screening, the check,
   each test length). The catalogue's sizes are settings (`addition_data.catalogue`, W29);
   `WindowRecurrentShape.state_codes` gives each state's code; `studies.common.crate_hash` hashes
   the extension module, for both studies. The calibration's config is
   `scripts/window_recurrent_calibration.json` (`k` 1, 2, 4 at the provisional grid's largest `s`
   and `m`, 16 and 16, 1 seed, 2 steps); its figures and W12 follow in their own PR.
9. **The matrix study on `jebel`** under the machine lock. Findings in the script's docstring.
10. **What follows**, decided on the findings, each its own PR: if a cell works, whittle it down
    and run the control (W16); if none does, the probes say whether the codes, the cell or the
    window failed.
11. **Docs**: README, roadmap, next-steps, the parent plan; this workplan retired.

## Known limits

- **One encoding.** The column-causal tokens (candidate 1, vocabulary 27); the start token of the
  terminated plan's D12 isn't needed (the window has a `Position`).
- **The state's meaning is found by probing**, not supervised (W3): a model that passes may carry
  more than the carry, and the report says only what its codes do.
- **The discrete state imposes finiteness.** A model that works shows a finite-state solution can
  be learned, which is weaker than discovering that one exists; `m` free (W4-i) and the codes
  used are the measure of how much was imposed.
- **No new parameter sharing across a network's layers** is needed: the window's weights are one
  network applied per window, as a batch is. The parent plan's P7 (a looped block) stays deferred.
- **Cost before data.** Stages 2-8 are about eight PRs before the first number, and the network
  kind is new in every implementation and the crate.

## Timing and checks

- No existing timed path changes: the new layers and the network are new paths, tier 0, each
  with gradient checks and parity in all three implementations. If stage 5 changes a shared crate
  op, that PR is tier 1.
- Every stage that touches the package passes the golden run, bit-identical (118 entries), until
  stage 7 adds the new network's entries under owner approval of the new functionality
  (measurement.md, §8); from stage 7, 121 entries.
- The study runs on `jebel` under the machine lock; results are written with provenance under
  `data/studies/`.
