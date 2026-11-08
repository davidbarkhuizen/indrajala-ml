# Workplan: a window-recurrent network with an array window, learning from a given gradient

**Status (2026-11-08): approved; S1 done, S2 next. Stage M1 of indrajala-research's
`smb-machine-policy-workplan.md` (the finite-state machine policy for Super Mario Bros 1-1, D24-D31
there). The decisions below (D1-D9, this plan's own) are settled under the owner's delegation; the
owner reviewed them and said go (2026-11-08), and chose D7 (the Rust path): the crate stage lands
before M2.**

In chat this plan's decisions are named "the array window plan's Dn", so they don't collide with the
NES plans' D1-D31.

## The question

The machine-policy plan needs a policy that is a finite-state (Mealy) machine. It reads a screen of
tile planes each decision and holds a snapped state between decisions. PPO trains it from a loss PPO
computes itself. That needs three things the library doesn't have:

1. **A window of array layers.** The window-recurrent network runs a window of layers once per step and
   carries a snapped state from step to step. Today the window must be a token model: an Embedding of k
   token ids, then token layers, then a token-wise softmax. The policy's window is today's conv body
   (`Conv(1,16), Conv(3,16), Pool(2), Conv(3,32), Dense(256,"relu")`) over one step's planes.
2. **A loss given at every step.** The network learns only from labels today (`learn_batch`, one
   digit per column). PPO's loss belongs to the caller: the library must take the gradient at every
   step's outputs, as `learn_batch_output_gradient` does for a feed-forward network.
3. **Sequences that don't start at the initial state.** Recurrent PPO (the machine-policy plan's
   D29) trains on 32-decision pieces of longer rollouts. Each piece starts from the state stored when
   it was played, and an episode may end and restart inside a piece.

M1 is done (the machine-policy plan's table) when: the token window is unchanged (the golden run
bit-identical); gradients match finite differences through the sequence and the snap; numpy agrees
with Rust; and a saved network round-trips.

## What exists (read 2026-11-08, `main` at `33ebebd`)

- **The network** (`window_recurrent_array_network.py`, `WindowRecurrentShape`). For every window it
  sets the StateInputs' state, runs the window's layers, and then runs the recurrence. The recurrence
  works per factor: attention pooling over the tokens the output layer reads, a GRU cell, and the
  codes' snap. Training (`per_layer_gradients`) runs the windows forward and keeps each window's input
  state and generator state. It then goes backward from the last window, re-running each window's
  forward pass before its backward pass (W20). Each window's output delta is weighted by 1/W, and so
  is each snap's codes' losses (W14).
- **The recurrence's tokens are `self.columns`** (the k columns). For a token window that is the
  token count. An array window has one token (below), so the count must come from the shape walk
  instead. For every token window both give k, so no bits move.
- **A conv stack's output is already one token.** `spec_shapes.token_shape` reads an image or flat
  shape as `(1, prod(shape))`, and the builder builds a StateInput from `token_shape`. A StateInput
  after `Conv(3,32)` on the 13 × 16 tile grid would be `(1, 480)`, and `P` would be `(480, s)`. Two
  things refuse it today: the shape walk asserts a StateInput reads a 2-D token shape, and the
  validator refuses a token spec (StateInput is one) outside a token model.
- **Attention pooling over one token is exactly that token.** `p = e / e = 1` and `u = 1 · x`; `q`'s
  gradient is exactly 0. No special case is needed.
- **The given loss.** `Dense(n, "linear", output=True, loss="given")` builds an affine output layer
  whose delta is the caller's gradient. The base network's `learn_batch_output_gradient(lr, X,
  gradient, clip_norm)` calls `gradient(Y)` on the step's own outputs, so one forward pass serves PPO's
  loss and the update. The window network doesn't implement it.
- **Rust's dense layers fuse their hidden delta with the next layer's `W`**
  (`pa.layer_activation_hidden_delta_batch(next_layer.W, next_layer.delta_batch, ...)`). They read
  `downstream_batch()` only before a LayerNorm (`before_layer_norm`). Token layers and conv layers
  always read `downstream_batch()`. The window's backward pass adds the pooling's gradient through a
  wrapper (`_WithPooling`) that has no `W`. So a Rust `Dense(256,"relu")` standing before the output
  layer would fail there today. numpy's dense layers read `downstream_batch()` and are fine.
- **The Rust network builds a window program** (one crate call per pass) from its layers at
  construction. `_program_op` raises for conv, pool and flat dense layers. The per-layer path
  (`per_layer_windows`, `per_layer_gradients`, run with Rust layers) is the program's reference, and
  it is tested (`per_layer_path`).
- **The pure-Python implementation** (`WindowRecurrentBackpropNetwork`) is token-only. It has no conv
  front end.
- **Format 2** saves the input shape, layer specs and recurrence generically. Loading picks the class
  by shape (`window_recurrent`). The same class serves both window kinds, so its file loads unchanged.
- **The backends have no row select.** A reset inside a sequence needs one (D5).

## The design

**One class, two window kinds.** The kind comes from the first layer. An Embedding makes a **token
window**, exactly as today. A Conv or Dense makes an **array window**: each step is one observation's
flat array of `prod(input_shape)` floats, and the network's input is `(N, S · prod(input_shape))` for
S steps. The policy's window would be:

```python
window = [
    Conv(1, 16),
    Conv(3, 16),
    Pool(2),
    Conv(3, 32),
    StateInput(),
    Dense(256, "relu"),
    Dense(12, "linear", output=True, loss="given"),
]
recurrence = WindowRecurrence((Factor(32, Codes(8)), Factor(32, Codes(8))))  # the machine-policy plan's D27
policy = WindowRecurrentRustArrayNetwork((13, 16, C), window, recurrence, Adam())
```

**The new API**, on both backends:

- `run_steps(X, *, state=None, starts=None, machine=False) -> Steps(outputs, states, codes, cells)`
  - `X` is `(N, S · step)`. `state` is `(N, state_size)`, the state before the first step; None means
    the learned initial state, snapped. `starts` is `(N, S)` booleans: a true entry means that row's
    state is the initial state at that step (an episode began there), whatever came before.
  - `machine=False` reads the network exactly as its training step does: soft codes at a temperature,
    couplings, continuous factors (D6). PPO's rollout uses this reading, so the policy that acted is
    the policy that trains.
  - `machine=True` is the inference reading: every factor hard-snapped. Each step's code indices and
    pre-snap cell outputs are kept, for the reader (M3) and its near-tie margins. Greedy evaluation
    (the machine-policy plan's D24) uses it.
  - `states` holds each step's next state, S of them; rollouts store them.
- `learn_batch_output_gradient(lr, X, gradient, clip_norm=None, *, state=None, starts=None) -> Y`
  - It has the base network's contract: `gradient` maps all S steps' outputs `Y`, `(N, S · outputs)`,
    to the loss's gradient at them, with row i being `d loss_i / d Y_i`. The step follows the mean
    over rows.
  - It runs `run_steps`' training reading forward and calls `gradient` once on all steps' outputs. It
    then goes backward window by window, as `per_layer_gradients` does (generator restored, forward
    re-run).

## Decisions (settled under the owner's delegation; options with pros and cons)

- **D1. The window kind.**
  - (a) *Chosen.* **One class; the first layer decides** (Embedding: tokens; Conv or Dense: array).
    Pros: one recurrence, one format-2 shape, one set of registry entries; everything the token window
    learned (couplings, temperatures, consolidation, the reader's `state_codes`) applies unchanged.
    Cons: the class's docstrings and `columns` name now cover two kinds (`columns` becomes the step's
    width, k for a token window).
  - (b) A new class beside it. Pros: no risk to the token window. Cons: a duplicate path for the
    recurrence, saving and training, which the structure rule (CLAUDE.md) refuses; a new format-2
    shape and registry entries.
- **D2. Where the state enters an array window.**
  - (a) *Chosen.* **Anywhere after the first layer and before the output layer, as one token of the
    shape there.** The policy puts it after the conv stack and before `Dense(256,"relu")`, so the relu
    mixes state and screen. Pros: today's body is unchanged, with no added layer; `P` is small
    (480 × 32); the state and screen interact before the action, so "this state and this screen" can
    choose differently from "that state and this screen". Cons: the conv stack never sees the state;
    whether that matters is for M5's probes.
  - (b) Only right before the output layer. Pros: simplest. Cons: the logits are then the screen's
    plus a per-state bias, with no interaction; the press-release fits that, but the trap needn't.
  - (c) As extra planes into the first conv layer. Pros: every layer sees the state. Cons: a new
    StateInput kind (broadcast over cells), 208 cells × s extra inputs to every conv.
- **D3. The array window's output layer.**
  - (a) *Chosen.* **The given loss only** (`Dense(n, "linear", output=True, loss="given")`), one row of
    n outputs per step. `classify_state` gives each step's argmax (the greedy action). Pros: what PPO
    needs; no new target layout. Cons: a supervised per-step task on an array window waits; it goes to
    next-steps.md as "not planned yet".
  - (b) A softmax output too, a label per step. Pros: general. Cons: a target layout and loss
    weighting no task here needs yet.
- **D4. The loss's weights.**
  - (a) *Chosen.* **The caller's gradient enters as given; each snap's codes' losses are weighted
    1/S, as the token window weights them 1/W.** The convention is documented: row i's loss is
    normally the mean over its steps, the token window's convention, so the codes' share matches it.
    Pros: the caller owns its loss's scale, as with the feed-forward network; the codes' losses keep
    their tuned relation to it. Cons: a caller that sums over steps (rather than averaging) gives the
    codes S times less weight, so the docstring must say so.
  - (b) A weight argument for the codes' losses. Pros: explicit. Cons: a knob with no use yet;
    `set_commitments` already scales them.
- **D5. Stored states and resets.**
  - (a) *Chosen.* **`state` and `starts`, as above.** A reset row's gradient goes to the initial
    state, summed over its resets, through one snap of the initial state per row
    (`first_state_backward`). The gradient to a given `state` is dropped: it is stale, R2D2's stored
    state. The select is a new backend op, `select_rows(mask, A, B)`: `np.where` on numpy; on Rust a
    crate op if one exists, else built through lists (`(N, 64)` arrays, a cost to watch in M2's budget).
    Pros: exactly what the machine-policy plan's D29 needs; an episode boundary inside a piece is
    exact, with no state leaking across it. Cons: a new backend op; one more path through the
    backward pass.
  - (b) Cut sequences at episode ends and pad. Pros: no select. Cons: padding and masks in every loss;
    the pieces' lengths vary.
- **D6. The rollout's reading.**
  - (a) *Chosen.* **The rollout reads the network as its training step does** (`run_steps`, the
    default reading). For one batch of rows, learn's forward outputs equal `run_steps`' bit for bit, so
    PPO's first ratio is exactly 1 on the same rows. Across different batch compositions (a rollout's
    environments against a minibatch's sequences) BLAS may round differently, so they agree only to
    rounding; the tests say which. Dropout layers run in inference mode in `run_steps`, and a window
    with dropout carries no ratio guarantee (the policy has none). Pros: PPO's importance ratio means
    what it says; the soft-to-hard handover (the machine-policy plan's D28) happens inside the policy
    that acts. Cons: while the codes are soft or coupled, the acting policy isn't yet a machine; only
    `machine=True` is.
  - (b) Roll out with the hard snap. Pros: the acting policy is always a machine. Cons: the ratio
    compares two different functions until the coupling reaches 1, biasing PPO.
- **D7. The Rust path. The owner's choice (2026-11-08): (b), "land it before M2 - it might be useful elsewhere - and it will def speed things up".**
  - **Common to both: the per-layer path, with Rust layers (S3).** It is the reference for the
    window program, as it is for token windows (`per_layer_path`, the window program's parity tests).
    So it is built either way. Rust's activation layers read `downstream_batch()` whenever the next
    layer has no `W` to fuse with (the pooling wrapper). `before_layer_norm` widens to that rule,
    which changes no existing path, since each had a `W` or a LayerNorm.
  - **What the crate stage would be (S4).** The window program (`core/src/program.rs`) is built op by
    op, and each op has four pieces: a forward, a downstream gradient, an activation mask and a
    gradient accumulation. The crate already has the batch kernels (`nn/conv.rs`:
    `forward_batch`, `downstream_batch`, `accumulate_gradient_batch`, `max_pool_forward_batch`,
    `max_pool_downstream_batch`). The work:
    - two new ops: `Conv { geometry, relu }` and `MaxPool { geometry }`;
    - a flat dense layer as `TokenDense { tokens: 1 }`, if its bits equal the flat layer's (checked
      first; a `Dense` op if not);
    - a `GivenOutput` op: affine, its delta the given gradient;
    - in `window_program_learn`:
      - an optional given gradient, `(N, S · outputs)`, in place of the targets. Python first runs the
        program forward to get the outputs for `gradient(Y)`, so the forward runs twice, once in the
        crate's learn, as the per-layer path does;
      - the resets mask and the initial state's gradient summed over the resets (D5);
      - a stored first state per row;
    - the bindings, the stubs and the crate's tests; a crate release; then a "Bump rust/" PR here;
    - parity: the program by bits the per-layer path's, outputs, states, codes and every gradient,
      through resets.

    It changes the crate's timed `window_program_learn` signature, so token windows' learn is tier 1:
    an A/B on klip and stone as a guard (`jebel` is parked).
  - (a) **Per-layer path only for now; the crate stage (S4) only if M2's first probe shows the path is
    the bottleneck** (decisions a second against the conv probe's 215 on klip). Pros: M2 starts after
    S3, days sooner; no crate release before knowing the network learns at all (existence first); S4
    keeps its full design here, ready. Cons: a Python loop over 7 layers × 32 steps per minibatch; if
    the path is slow, M2's first probe measures a cost that S4 then changes, and the budget is set
    twice.
  - (b) *Chosen (the owner).* **Both now: S3, then S4, before M2.** Pros: M2's budget is measured once, on the path the
    probes will use; one crate call per pass from the first probe; the program is the production path
    for every window kind, not only tokens (structure first). Cons: a crate stage (a crate PR, a
    release, a bump PR and their A/Bs), likely 1-2 days, before 1-1 is tried; if M4 or M5 change the
    window (D2's placement, a different body), new ops may follow.
- **D8. The pure-Python implementation.**
  - (a) *Chosen.* **It refuses an array window, by name and reason.** Pros: the Python
    implementation has no conv front end; the class's builder refuses it with a clear message. Cons:
    the three-implementation symmetry breaks for this kind; recorded in next-steps.md.
  - (b) A Python conv front end. Cons: large; no user.
- **D9. Stages and timing tier.**
  - (a) *Chosen.* **S1 this plan; S2 numpy (specs, network, tests); S3 Rust (the delta rule, the
    per-layer path, `select_rows`, parity); S4 the crate's array ops, before M2 (D7); the plan retires
    with S4.** S2 touches numpy's
    `per_layer_gradients`, a timed `learn` path, with no speedup claimed: tier 1. `jebel` is out of
    service, so the tier-1 A/B (`window_recurrent_timing`, `--order ONNO`) runs on klip, the parked
    plan's stand-in, and is recorded as a guard, not a claim. The golden run is checked on a runner
    (stone or klip). Pros: each PR is reviewable alone; numpy is the reference Rust is checked
    against. Cons: two PRs before M2 can use the Rust backend.
  - (b) One PR. Pros: faster. Cons: about 1,500 lines with tests, on both backends at once.

## Feasibility (the register's checks that bear on a library stage)

| check | here |
|---|---|
| the window can represent what worked | the policy's window is today's body, which cleared 1-1 89-93 of 100 greedy; D2 adds `P` (480 × 32) and nothing else |
| the gradient reaches the state | through the StateInput's `P`, the relu, the pooling (one token: exact) and the cell; the FD tests check it over 1-3 steps and through resets |
| the memory span | sequences of 32 (the machine-policy plan's D29), backpropagated whole |
| cost | per minibatch: S forward passes, then S re-run forward and S backward per-layer passes, about 3 conv passes per decision against PPO's feed-forward 2 (forward, backward); measured at M2 |

## Tests (each stage's; numpy in S2, Rust in S3)

1. **Token windows unchanged:** the full suite passes, and the golden run is bit-identical
   (`golden_training_run.py check data/refactoring/golden_run.json`, on a runner).
2. **Finite differences through the sequence:** an array window, 1-3 steps, a random fixed `G` as the
   loss `sum(G ⊙ Y) / N`. Three cases:
   - the codes as the identity, as the token window's test does;
   - soft codes at a temperature with a coupling of 0.5, the snap's gradient included (the objective
     adds each snap's codes' losses, `CodesArrayLayer.loss_batch`);
   - a continuous factor.
   Every parameter is checked, the initial state's included.
3. **Stored states and resets:**
   - a sequence from a given state equals the same steps run after the steps that led to it, bit for
     bit;
   - a reset at step t equals a fresh sequence from t, bit for bit;
   - FD through a reset, with the initial state's gradient summed over resets and nothing crossing a
     boundary;
   - the given state receives no gradient.
4. **The rollout's reading:**
   - `run_steps` one step at a time, carrying `states`, equals one S-step call bit for bit;
   - learn's forward outputs (seen by `gradient`) equal `run_steps`' on the same batch bit for bit;
   - `machine=True`'s codes equal `state_codes`'.
5. **Learning:** a few steps of a given loss lower it on its batch. A 2-step memory toy (a cue at step
   1, answer at step 2, through a given cross-entropy) is learned to 100% with every factor snapped:
   the library's own check, before M4's control.
6. **Refusals:**
   - learn with labels on a given-loss window;
   - `loss_batch` on one;
   - a token spec other than StateInput in an array window;
   - an array window in the Python implementation;
   - a `starts` or `state` of the wrong shape.
7. **Saving:** an array-window network saves, loads and resumes training by bits (format 2, both
   backends).
8. **Rust (S3):**
   - numpy against Rust from one seed: outputs, states, codes and every gradient, to the products'
     rounding;
   - the Rust per-layer path's array window against numpy through resets;
   - the token windows' program path unchanged (its parity tests pass).

## Stages

| # | What | Done when |
|---|---|---|
| S1 | This plan | merged after the owner's review |
| S2 | numpy: the shape walk and validator accept an array window (D1, D2, D3); the recurrence's tokens from the shape walk; `run_steps`, `learn_batch_output_gradient`, `state`/`starts` (D4-D6); the Python refusal (D8); tests 1-7 on numpy | CI green; golden run bit-identical; tier-1 guard on klip read |
| S3 | Rust: the widened delta rule, the per-layer path for array windows, `select_rows` (D5, D7); test 8 and tests 2-7 on Rust. | CI green; golden run bit-identical; M2 can start in indrajala-rl |
| S4 | The window program's array ops in the crate (D7's list), released; a "Bump rust/" PR here routes array windows through the program; parity with S3's path by bits; the tier-1 guards; next-steps.md takes the leftovers; this plan retired | crate CI and release; CI green here; parity by bits; golden run bit-identical |

## Risks

- **The crate stage's cost (D7).** S4 is a crate PR, a release and a bump PR before M2; if M4 or M5
  change the window (D2's placement, another body), new ops may follow.
- **`select_rows` on Rust through lists** may cost at small N. If it shows, it gets a crate op.
- **PPO's noisier gradient through the snap** is the machine-policy plan's D28 risk, tested at M4,
  not here.
