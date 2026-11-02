# indrajala-ml: how the neural networks are implemented

An audit of the code base on its own terms: what the networks compute, how training is wired, and
which design decisions the code embodies. It was written from the source alone (`indrajala_ml/`,
`rust/`), without the repository's docs, workplans or the literature. Sections 1-15 are that audit;
§14 lists the decisions and §15 the points to check.

§16 is the second pass: each decision checked against the papers themselves (the local library,
`indrajala-literature` and `indrajala-literature-private`, with 19 papers fetched to fill its gaps),
and a verdict per decision. Sections 1-15 report each source as the code cites it; §16 is where the
citations were read.

Contents:

1. [The shape of the code base](#1-the-shape-of-the-code-base)
2. [Describing a network: layer specs](#2-describing-a-network-layer-specs)
3. [Backpropagation without autograd](#3-backpropagation-without-autograd)
4. [Layers](#4-layers)
5. [Output layers and losses](#5-output-layers-and-losses)
6. [Initialisation](#6-initialisation)
7. [Optimisers](#7-optimisers)
8. [The training loop](#8-the-training-loop)
9. [Network families](#9-network-families)
10. [The window-recurrent network](#10-the-window-recurrent-network)
11. [Decoding and generation](#11-decoding-and-generation)
12. [Numerics, determinism and the three implementations](#12-numerics-determinism-and-the-three-implementations)
13. [Persistence](#13-persistence)
14. [Design decisions, summarised](#14-design-decisions-summarised)
15. [Points to check against the literature, and findings](#15-points-to-check-against-the-literature-and-findings)
16. [The design against the literature](#16-the-design-against-the-literature)

---

## 1. The shape of the code base

The same networks are implemented three times:

| Implementation | Where | Unit of computation |
|---|---|---|
| Pure Python | `indrajala_ml/model/layers/python/`, `networks/python/` | one **node** (neuron) object per unit; scalars, lists, left folds |
| numpy | `layers/numpy/`, `networks/numpy/` | one layer, arrays; batch rows `(N, features)` |
| Rust | `rust/core` (the `indrajala-math` crate), `rust/src` (pyo3 bindings), `layers/rust/`, `networks/rust/` | one **fused op** per layer method, one crate call per pass |

The pure-Python implementation is the readable reference, numpy the vectorised form, and Rust the
fast one. The three are held to each other by parity: the same seed gives the same initial weights
in all three (one random-number stream family, §12), and the same arithmetic, in the same grouping,
gives the same bits wherever the transcendental functions agree.

The layering, from the bottom:

```
specs (backend-free data)        layer_specs.py, update_rules.py, window_recurrence.py
  validation + shape walk        spec_validation.py, spec_shapes.py
builders                         array_layer_builder.py (numpy, Rust), python_layer_builder.py
layers                           forward / delta / downstream / accumulate, per backend
networks                         ArrayNetworkBase, BackpropNetworkBase, + "shape" mixins
optimisers                       OptimizerBase -> Python / Numpy / Rust optimizer
training                         train.py (mini-batch loop, pocket, schedules, resume)
```

A network is a list of **layer specs** plus an **update rule**. Everything else (classes per
backend, the loss, the output contract) follows from those two and from a "shape": single output,
multiclass, sequence, or window-recurrent.

There are also many *preset* classes (e.g. `momentum_conv_rust_array_multiclass_backprop_classifier_network.py`),
each a few lines: a preset differs from its parent only in the spec hooks (`_hidden_spec`,
`_output_spec`) or the rule (`_update_rule`), set by mixins (`spec_hooks.py`). They are the
historical surface; `SequentialArrayNetwork(input_shape, specs, rule, shape=, backend=)` is the
generic one.

## 2. Describing a network: layer specs

`model/specs/layer_specs.py` defines frozen dataclasses:

| Spec | Meaning |
|---|---|
| `Dense(size, activation, dropout, output, loss, bias)` | fully connected; hidden sigmoid / ReLU / GELU / GELU-tanh / linear; output sigmoid or softmax; `bias=True` with `linear` is an *affine* layer (ends a residual body) |
| `BatchNorm(activation, epsilon, running_rate, group_size)` | normalises the **linear** layer before it, then applies sigmoid or ReLU; ghost batch norm with `group_size` |
| `Conv(kernel_size, channel_count, stride, activation)`, `Pool(pool_size, stride)` | valid-padding conv (ReLU or linear-before-BN), max pooling |
| `Residual(body)` | `x + body(x)`; body ends in an affine layer or an attention (optionally then a Dropout) |
| `Patches(p)` | image to `(H/p · W/p, p·p·C)` tokens |
| `Embedding(vocabulary, size, segments)` | token ids to rows of a learned table; optional paragraph segments with separator and pad ids |
| `Position()` | learned `(T, d)` table added to the tokens |
| `LayerNorm(epsilon)` | per-token normalisation |
| `Attention(heads, key_size, causal, dropout, positions, rotary_base, document_mask, key_value_heads)` | multi-head self-attention, optional causal mask, attention dropout, RoPE, document mask, GQA/MQA |
| `GatedDense(size, activation)` | GLU family: SwiGLU (SiLU), GEGLU (GELU) |
| `Dropout(p)` | inverted dropout over tokens |
| `TokenMean()` | mean over tokens, `(T, d)` to `(d,)` |
| `StateInput()` | the window-recurrent network's state entering its window (§10) |
| `Branches(branches, reads, merge)` | the window-recurrent network's per-factor paths (§10) |

**Expansion.** `expand_specs` flattens `Residual(body)` into `Fork, *body, Add` and `Branches` into
`Split, branch₀, Branch, branch₁, …, [Merge]`. Every layer index (snapshots, optimiser state,
checkpoints) is an index into the expanded list, so the runtime network is always a flat chain of
layers; skip connections are references between layers in that chain (§3).

**Validation.** `spec_validation.validate_layer_specs` accepts only combinations that every
implementation builds, so every accepted network is parity-testable. Among the rules it enforces:

- dropout on a dense layer only with a sigmoid activation (it is fused with the sigmoid);
- a `BatchNorm` only right after a linear (bias-free, activation-free) Dense or Conv; ReLU only after
  a conv;
- among tokens: no `BatchNorm` (a `LayerNorm` instead), at most one `Position`, and learned
  positions or rotary attention, never both;
- a token block's body ends in `Attention` or an affine `Dense`, optionally followed by `Dropout`; a
  `GatedDense` stands only right before that affine layer;
- the output layer is the last, sigmoid or softmax; softmax requires the cross-entropy loss; a
  token-wise (per-token) output layer must be softmax;
- residual bodies hold no nested residual block.

**Shape walk.** `spec_shapes` computes each expanded spec's input and output shape: flat `(n,)`,
image `(H, W, C)` (channel-major when flat), or tokens `(T, d)` (token-major when flat, index
`t·d + j`). A `Dense` over tokens acts on each token with shared weights. A flat layer seen by a
`LayerNorm` or `Attention` is one token.

## 3. Backpropagation without autograd

There is no computational graph and no automatic differentiation. Each layer implements its own
backward pass by hand, with a small fixed protocol (`protocols/array_protocols.py`,
`protocols/layer_protocols.py`):

```
forward(x) / forward_batch(X)                -> activation, caches what the backward pass needs
compute_output_delta(_batch)(target)         -> output layer only: delta from the loss
compute_hidden_delta(_batch)(next_layer)     -> delta from next_layer.downstream()
downstream(_batch)()                          -> the gradient this layer sends to its input
accumulate_gradient(_batch)(input_activation) -> adds this batch's parameter gradients
```

Key conventions:

- **`delta` is the gradient at the layer's pre-activation** (`dL/dz`), as in the classic
  presentation of backprop. A sigmoid hidden layer computes `delta = downstream ⊙ a(1−a)`, ReLU
  `downstream ⊙ [a > 0]`, GELU `downstream ⊙ gelu'(z)`.
- **The layer that owns the weights computes the gradient to its input** (`downstream = δ W`, or
  `Wᵀδ` for one example). So a layer reads only its successor's `downstream()`, which lets conv,
  pool, norm, attention and token layers sit on either side of anything.
- **Activations are fused into the layer** (a `Dense` carries its activation); there are no separate
  activation layers. Batch norm carries the activation of the linear layer before it.
- **The loss is fused into the output layer**: the output layer's delta *is* the loss gradient at
  its pre-activation (§5). There is no separate loss object in the training path.
- **Gradients are sums.** `accumulate_gradient_batch` adds `δᵀ X` (and `Σ δ` for the bias); the
  optimiser divides by the batch size B. A conv layer sums over output positions and examples alike.
- **Parameter-free layers** (pool, fork, add, patches, token mean, dropout) have no-op accumulation;
  layers whose delta is just the downstream use the `DeltaIsDownstream` mixin (`specs/hidden_layers.py`).

**One example and a batch.** Every layer has single-example and batch methods. In the numpy token
layers the single-example methods are literally the batch ones on a batch of one (`BatchShaped`).
`learn(x, y)` is a step on one example; `learn_batch` runs forward, deltas, then accumulation for
every layer, then **one** optimiser step over all layers (`ArrayNetworkBase._learn_batch_input`).
Doing every layer's accumulation before any update is what makes global-norm clipping possible.

**Skip connections in a chain.** A residual block is `Fork … body … Add`. The builder wires the
`Fork` to its `Add` and to the body's first layer. Forward: the fork keeps `x`, the add returns
`body(x) + x`. Backward (in reverse order, so the add and body are done first): the add's delta is
its downstream, passed unchanged to both paths; the fork's delta is
`body_first.downstream() + add.delta`. `Branches` work the same way (`array/branch_layers.py`):
the `Split` keeps its input and, when asked for its downstream, sums its branches' first layers'
downstreams in branch order; a `Merge` sums the branch ends.

**Training mode.** Dropout and batch norm behave differently in training; the network switches
training mode on for the **forward pass only** of a learn call. The backward pass reads
`_was_training` (or the saved mask) so it differentiates what the forward pass actually did.

**The pure-Python path** works node by node: each `BackpropNode` holds its inputs, weights, bias,
activation and delta. A batch normally runs example-major (forward, backward, accumulate per
example, then one update). A network with batch norm or a drawing dropout layer runs **layer-major**
(`layers/python/layer_major.py`): all examples through layer 1, then layer 2, …, keeping each
example's per-node state in "lanes". Batch norm needs this because its output depends on the whole
batch; dropout takes it so that masks are drawn in numpy's `(batch, size)` row-major order, layer by
layer, giving the same random stream as numpy.

## 4. Layers

Notation: `X` is `(N, in)`, `W` is `(out, in)`, `b` is `(out,)`, `R` the `(N·T, d)` token rows of
a token batch. Every sum other than a matrix product is a **left fold in a fixed order** (numpy
does it through `np.cumsum(...)[-1]`, never `.sum()`, whose pairwise summation would change the
bits).

### 4.1 Dense family (`numpy/array_layer.py` and siblings)

| Class | Forward | Delta |
|---|---|---|
| `ArrayLayer` (sigmoid) | `A = σ(X Wᵀ + b)` | hidden `D·A(1−A)`; output: squared loss (§5) |
| `ReLUArrayLayer` | `relu(X Wᵀ + b)`; `−0.0` normalised to `+0.0` | `D·[A > 0]` (derivative 0 at 0) |
| `GELUArrayLayer` | exact GELU `z·Φ(z)` through erf, or the tanh form | `D·gelu'(Z)` |
| `DropoutArrayLayer` | `σ(XWᵀ+b) ⊙ M / keep`, `M = u ≥ p` per example and unit | `D·σ'·M/keep` |
| `LinearArrayLayer` | `X Wᵀ`, no bias, no activation (before BN) | its downstream |
| `AffineArrayLayer` | `X Wᵀ + b`, no activation (ends a residual body) | its downstream |
| `SoftmaxArrayLayer` | row-wise max-shifted softmax | `A − Y` |
| `CrossEntropyArrayLayer` | sigmoid | `A − Y` (binary cross-entropy per unit) |

Sigmoid is `1/(1+exp(−z))` directly (overflow of `exp(−z)` yields exactly 0). Dropout exists only
fused with a sigmoid hidden layer; ReLU/GELU dense layers have none.

### 4.2 Batch normalisation (`batch_norm_array_layer.py`)

Batch norm is a *pair*: `Dense(n, "linear")` (no bias, since the mean subtraction would cancel it
and `β` takes its role) then `BatchNorm(activation)`.

- Training: per feature (per channel over batch × positions after a conv), `μ`, biased `var`,
  `x̂ = (x−μ)/√(var+ε)`, `y = act(γ x̂ + β)`. Running averages move at `running_rate` (0.1), the
  running variance from the **unbiased** `ss/(m−1)`.
- Inference: the running averages.
- Backward: the paper's term-by-term chain rule (`dvar`, `dmu`, then `dx`), not the compact form.
- **Ghost batch norm** (`group_size`): the batch is split into consecutive groups, each normalised
  on its own statistics and each moving the running averages in turn; `γ, β` gradients sum over the
  whole batch.
- Refusals: a one-example training step, or a batch whose last ghost group has one example (it would
  normalise to 0 and pass no gradient). The trainer drops a final batch of one for such networks.
- `γ`, `β` start at 1 and 0 and are never weight-decayed.

### 4.3 Layer normalisation (`layer_norm_array_layer.py`)

Per token (a flat layer is one token): `μ`, biased variance, `ε = 1e−5`, `y = γ x̂ + β`, `γ = 1`,
`β = 0` at start, no activation, same in training and inference. Backward in the compact form
`dx = ((dx̂ − mean(dx̂)) − x̂·mean(dx̂·x̂)) / std` with `dx̂ = δ·γ`.

### 4.4 Convolution and pooling (`conv_array_layer.py`, `max_pool_array_layer.py`)

- Valid padding, configurable stride, square kernels; `W` is `(channels_out, C·k·k)`, the kernel in
  `(channel, row, col)` order. Forward is **im2col** (`sliding_window_view`) then a product; the
  downstream is col2im as one strided slice-add per kernel offset.
- Activations are flat and channel-major `(N, C·H·W)`, so a dense layer after the front end sees
  the same index order in all three implementations.
- A conv layer is ReLU, or linear (no bias) before a batch norm.
- Max pooling: non-overlapping by default (stride = pool size), per channel, ties to the first slot
  in row-major order; the gradient goes to the winner only.
- The conv presets read a single-channel image (`(H, W, 1)`).

### 4.5 Token layers (`token_array_layer.py`)

- **Patches**: a fixed permutation of a channel-major image into `T = (H/p)(W/p)` tokens of
  `p·p·C` features, patch-row-major; its backward is the inverse permutation. No class token, no
  patch projection: the first token-wise `Dense` is the projection.
- **TokenMean**: mean over tokens; the patch model's pooling, in place of a class token.
- **Embedding**: `E[ids]`; gradient is a scatter-add (`np.add.at`) in example-then-token order. With
  segments, the pad id's row is zero at initialisation and its gradient is zeroed every step. `E` is
  not weight-decayed.
- **Position**: learned `(T, d)` table, **zero-initialised**, added to the tokens, not decayed. With
  segments, each token reads the row of its position *within its paragraph*.
- **TokenDense**: a dense layer with weights shared across tokens (ReLU, GELU, or linear+bias).
- **GatedDense**: one packed `(2·size, d)` weight (gate rows then value rows) and one bias; output
  `act(a) ⊙ v`. One parameter pair for the optimiser, weight decayed, bias not.
- **TokenDropout**: inverted dropout over every feature of every token; masks drawn row-major.
- **TokenSoftmax**: the token-wise output layer (§5).

### 4.6 Attention (`attention_array_layer.py`, `rust/core/src/attention.rs`, `python/attention_layer.py`)

The most elaborate layer. Per example over `T` tokens of width `d`, `h` query heads of `d_k`
(default `d/h`), `g` key/value heads (default `h`):

```
project   Q = R Wqᵀ + bq   (N·T, h·d_k);   K = R Wkᵀ + bk, V = R Wvᵀ + bv   (N·T, g·d_k)
          [rotary: rotate Q and K per head and token, after the biases]
attend    per head i, with key/value head j = i // (h/g):
          S = Q_i K_jᵀ / √d_k  (masked: −inf),  P = softmax_rows(S),  H_i = P V_j
          [dropout: P ⊙ M / keep before the product]
combine   out = [H_0 … H_{h−1}] Woᵀ + bo
```

Design points:

- **Biases on all four projections.** The scale is computed once as `√d_k` and divided by, never
  multiplied by its reciprocal.
- **Tiled, online softmax** (FlashAttention-2 form), in numpy, pure Python and Rust alike. Keys come
  in blocks of 16 (`BLOCK`) with block boundaries at absolute multiples of 16 columns. Per query row
  it keeps a running max `m`, sum `l` and unnormalised output `H`:
  `m' = max(m, rowmax(S)); e = exp(S − m'); l = l·e^{m−m'} + Σe; H = H·e^{m−m'} + e V`, then
  `H / l`. Only `m` and `l` are kept (memory linear in `T`); the backward recomputes each block's
  `P = exp(S − m)/l`, uses `D = rowsum(dH ⊙ H)`, `dS = P ⊙ (dP − D)`. Blocks no query can reach are
  skipped, and a wholly masked row shifts by 0 so it contributes nothing; since every sum is a
  sequential fold, skipping is exact.
- **Masks** are built per block: causal (`u ≤ t`), document/segment (another paragraph's key or a
  pad is masked, never the query's own column), and a band (a sliding-window reference used by
  the decode tests). With `document_mask=False`, only pads are masked across paragraphs.
- **Attention dropout** drops the normalised weights (`e ⊙ M / keep` inside the online softmax). The
  mask is drawn for every `(example, head, query, key)` entry, masked ones included, block by block.
  It is **not stored**: the generator's state before the draws is saved and the backward pass
  re-draws the identical masks from a copy.
- **Rotary positions (RoPE)**: rotate-half pairing (feature `j` with `j + d_k/2`), angle
  `t·base^{−2j/d_k}`, base 10000, applied to Q and K after their biases; V is not rotated. Backward
  rotates `dQ`, `dK` back. The `cos`/`sin` table is computed once in pure Python (`rotary_table.py`)
  and read by all three implementations, so no library's `cos` can differ.
- **GQA / MQA**: K and V projected to `g` heads; numpy expands them by `np.repeat`; the backward
  folds `dK`, `dV` over each group in head order. `persistence/grouping.py` converts a trained
  multi-head model to grouped heads by **mean-pooling** each group's K and V projections (the
  method the code attributes to Ainslie et al.), with a fresh optimiser state.
- Single-example methods are the batch ones on one example; there is no `(T, T)` matrix anywhere
  except the inspection helpers `weights()` and `dropout_mask()`.

### 4.7 Activations (`numpy/activations.py`, `python/activations.py`, `core/src/activation.rs`)

`tanh(z) = 1 − 2/(exp(2z) + 1)`, GELU exact `(0.5z)(1 + erf(z/√2))` with erf a port of fdlibm's
`s_erf.c`, GELU-tanh (GPT-2's), SiLU `z·σ(z)`. Each is written once as an expression with a fixed
grouping, and the three implementations use the same expression, so they agree by bits wherever
their `exp` agrees. Every transcendental goes through one module-level `exp` that tests can
replace.

## 5. Output layers and losses

The loss is implied by the output layer's delta:

| Output | Delta | Implied loss per example |
|---|---|---|
| sigmoid, `loss="squared"` (the default) | `(a − y)·a(1−a)` | `½ Σ (a − y)²` |
| sigmoid, `loss="cross_entropy"` | `a − y` | binary cross-entropy per unit (one-vs-rest if several) |
| softmax (requires cross-entropy) | `a − y` | categorical cross-entropy |
| token-wise softmax | `(P − Y)/T` | mean over the T tokens of the categorical CE |
| token-wise softmax with segments | `(P − Y)·N/count`, 0 where not counted | mean over the batch's counted targets |

The optimiser then divides the summed gradient by B, so the effective objective is the batch
mean of the per-example loss. With segments the delta's `N/count` factor cancels that `1/B`,
giving a mean over every counted token in the batch. A target is counted unless its input is a
pad.

Targets are one-hot rows built per shape (`_target_batch_array`). Classification is argmax (first
on a tie) or a 0.5 threshold for one output.

## 6. Initialisation

- **Default ("fan-in aware")**: every weight matrix and bias uniform in `[−1/√fan_in, 1/√fan_in]`,
  W then b, layer by layer in forward order (`ArrayNetworkBase.randomize`). This applies to ReLU
  and GELU layers too; there is no He/Kaiming-normal scaling. `fan_in` is `C·k·k` for a conv.
- Exceptions:
  - LayerNorm/BatchNorm: `γ = 1`, `β = 0`; running mean 0, running variance 1.
  - Position table: zeros.
  - Embedding `E` `(vocab, d)`: uniform `±1/√d` (drawn "as a weight matrix of fan-in size"); pad
    row zeroed.
  - Attention: each projection drawn as a dense layer, `Wq, bq, Wk, bk, Wv, bv, Wo, bo` in that
    order.
  - Window-recurrent parts (§10): `StateInput.P` `±1/√s`; GRU gates each drawn with fan-in `d + s`,
    then the update-gate bias set to `update_bias` (1.0); codebook `±1/√s`; pooling query and
    initial state zeros.
- **Legacy low-dimensional nets** (`BackpropClassifierNetwork`): first-layer weights
  `±2/half_width(bounds)` per input dimension, later layers `±1`, biases `±1`, tuned for 1–2-D
  geometric toy problems; the code's own comments say this saturates at high dimension, which is
  why fan-in-aware exists.
- Pure-Python draws the same values in the same order as numpy (`fan_in_aware_init.py`), so the
  three implementations start from identical weights.

## 7. Optimisers

Update rules are hyperparameter-only dataclasses (`specs/update_rules.py`); the state and the
arithmetic live in one optimiser object per network (`optimizers/`), keyed by layer index, with
**one step count `t` for the whole network**. With `g` the gradient summed over a batch of B:

| Rule | Update |
|---|---|
| `SGD` | `w ← w − lr·(g/B)` |
| `Momentum(m)` | `u ← m·u + g/B;  w ← w − lr·u` (no dampening; the rate outside the velocity) |
| `WeightDecay(λ)` | decayed: `w ← w − lr·(g/B + λw)` (coupled L2); others plain SGD |
| `Adam(β1, β2, ε, weight_decay)` | `m, v` on `g/B`; `m̂ = m/(1−β1ᵗ)`, `v̂ = v/(1−β2ᵗ)`; decayed and `wd > 0`: `w ← w·(1 − lr·wd)` first (AdamW, decoupled); then `w ← w − lr·m̂/(√v̂ + ε)` |

Defaults: Adam `0.9, 0.999, 1e−8`, `weight_decay 0`. `βᵗ` uses a square-and-multiply `powi` shared
with the crate, rather than `**`, because `pow` rounds differently.

**What is decayed** is a per-layer `decayed` tuple aligned with `parameters()`: weight matrices
(dense, conv, attention projections, `GatedDense`, `StateInput.P`, GRU `W` and `U`) are; biases,
`γ`/`β`, embeddings, positions, codebooks, pooling queries and initial states are not.

**Gradient clipping** (`optimizer_base._clip_scale`): the global L2 norm over every layer's
accumulated gradient, divided by B (the norm of the *mean* gradient), left-folded over layers in
order. If it exceeds `clip_norm`, every gradient is read times `clip_norm/‖g‖`.

**Rust optimiser** (`rust_optimizer.py`): each rule is one fused crate call per **pair** of a layer's
parameters, `(W, b)`-shaped: `(W, b)`, `(γ, β)`, attention's four `(W, b)` pairs, or a lone
parameter paired with an empty array. The fused ops treat the pair's second member as a bias. For
plain SGD on one example, accumulate-and-apply is fused into a single `layer_sgd_step`.

## 8. The training loop

`training/train.py:train_backprop_network_mini_batch`:

- Each epoch: shuffle (Python's `random.Random`, seedable), chunk into batches (a final short batch
  kept; a final batch of one dropped for batch-norm networks), `learn_batch` per batch, where the
  rate is a float or a schedule read at the batch counter.
- **Pocket**: after each epoch, the training accuracy; the best epoch's checkpoint is kept and
  **restored at the end** (the pocket algorithm, applied to gradient training). There is no
  validation set inside the trainer; the pocket selects on training accuracy.
- **Schedules** (`lr_schedule.py`): `Constant`, `Warmup(steps, then)` (linear from `then(0)/steps`),
  `CosineDecay(rate, steps, floor)`, `LinearDecay`; the "production recipe" is
  `Warmup(…, CosineDecay(rate, total − warmup, rate/10))`.
- **Resumable runs**: a `RunCheckpoint` holds the network and optimiser state, the shuffle
  generator state, counters, the pocket and the schedule; resuming continues bit for bit and refuses
  a different schedule.
- **Per-epoch data** (`EpochData`): the examples can be redrawn each epoch from the run's generator.
- **Prepared datasets**: array networks train from one backend matrix and row indices, so an epoch
  doesn't convert Python tuples each step.

`train_linear_classifier_network` is the one-example loop for the hard-threshold network (§9).
`ensemble_train.py` trains one-vs-rest ensembles in worker processes.

## 9. Network families

- **LinearClassifierNetwork** (`networks/python/linear_classifier_network.py`): hard-threshold
  `AssociationNode`s (`1 if w·x + θ > 0`), a fixed output node that fires when at least `k` hidden
  nodes fire (AND, OR or k-of-n, a MADALINE-style committee), trained by a minimum-disturbance
  rule: on an error, the wrong-side hidden node with the smallest `|z|` takes a perceptron step
  `w += lr·d·x`, `θ += lr·d`. Not gradient-based.
- **BackpropClassifierNetwork**: sigmoid MLP, one sigmoid output, squared loss by default; presets
  add ReLU, dropout, cross-entropy, momentum, Adam, L2, fan-in-aware init.
- **Multiclass**: argmax over `class_count` outputs (sigmoid+squared, sigmoid+BCE, or softmax+CE).
- **Conv**: conv/pool front end, dense hidden layers, multiclass output, on a single-channel image.
- **Ensembles** (`model/ensembles/`): one binary sub-network per class, the class with the highest
  probability wins.
- **Sequential**: any validated spec list and rule, for the multiclass, single-output or
  **sequence** shape. The patch model (ViT-like: `Patches`, token Dense projection, `Position`,
  pre-LN residual blocks, `TokenMean`, dense head) and the character-level causal language model
  (`Embedding`, `Position` or RoPE, pre-LN blocks of `Residual((LayerNorm(), Attention(causal=True)))`
  and `Residual((LayerNorm(), Dense(ff), Dense(d, linear, bias)))`, a final `LayerNorm`, token-wise
  softmax) are both Sequential networks.
- **Window-recurrent**: §10.

The transformer blocks are **pre-LN** (the `LayerNorm` is the first layer of each residual body),
with a final `LayerNorm` before the output layer. The FFN is two layers (ReLU/GELU) or a GLU
followed by an affine layer.

## 10. The window-recurrent network

`networks/window_recurrent_array_network.py`, `layers/numpy/recurrent_array_layer.py`,
`specs/window_recurrence.py`, `specs/window_presets.py`, and in Rust `recurrent.rs`, `factored.rs`,
`program.rs`. This is the network the code is built around, and its design is the least
conventional.

### 10.1 What it computes

An input is `W` windows of `k` token ids (columns), and the label is a class (a digit) per column.
A **window** is a causal sequence model over its `k` tokens (an Embedding, a Position, residual
blocks, a token-wise softmax output). A **state** `c` carries information from window to window:

```
c_0          = snap(c_init)                         learned initial state, snapped
for w = 1..W:
  tokens     = window(x_w ; c_{w-1})                state enters through StateInputs
  y_w        = softmax per column                   the window's digits
  for each factor g:
    u_g      = AttentionPool_g(tokens)              one learned query
    h_g      = GRU_g(c_{w-1,g}, u_g [, c_{w-1,−g}])
    c_{w,g}  = snap_g(h_g)                          nearest code, straight-through
```

- **StateInput**: `x_t ← x_t + P c` for every token, `P` `(d, s)`, one `StateInput` before each
  residual block that should read the state (each with its own `P`). The state conditions the
  window additively, broadcast over all tokens, like a learned bias.
- **Attention pooling**: one learned query `q` (zero at start, so initially a mean), scores
  `q·x_t` (**no** `1/√d` scaling, no key or value projections), softmax over tokens,
  `u = Σ p_t x_t`. By default it reads the tokens the output layer reads (after the final
  LayerNorm); `BEFORE_STATE` reads the tokens before the first `StateInput`, so that without the
  cross-term each factor's next state depends only on its own state and the window's columns.
- **GRU cell** (gates in order r, z, h̃):
  `r = σ(W_r u + U_r c + b_r)`, `z = σ(W_z u + U_z c + b_z)`,
  `h̃ = tanh(W_h u + U_h (r ⊙ c) + b_h)`, `h = z ⊙ c + (1 − z) ⊙ h̃`. Here `z` is the
  *keep* gate, and its bias starts at `update_bias = 1.0`, biasing the cell toward keeping its state.
- **Codes** (vector quantisation): `m` learned codes of width `s`; `h` snaps to the nearest by
  squared distance (lowest index on a tie). The gradient passes straight through to `h`. Losses per
  snapped state, weighted by `1/W`:
  `‖sg[h] − e‖²` (codebook) + `β‖h − sg[e]‖²` (commitment, `β = 0.25`). The codebook learns by
  gradient (no EMA updates, no dead-code restarts).
- **Factors**: the state is `G` factors side by side, each with its own pooling, GRU, codes and
  initial state. With `cross=True`, each factor's GRU input is `[u_g, c_{−g}]` (the other factors'
  states). A factor with `codes=None` is continuous (no snap, no code losses).
- **Branches**: the window can split, after a shared trunk, into one branch per factor (each
  factor's pooling reads its branch's end) plus an "answers" branch feeding the output layer; or,
  with `merge`, every branch belongs to a factor and the branch ends are summed into recombining
  layers before the output. `reads` lets several factors share one branch.

**A machine view.** With every factor snapped, the state is a tuple of code indices.
`state_codes`, `first_step` and `window_step` read the network as a finite-state transducer: state
codes per window, and the transition from a state and a window's columns to the next state and the
window's outputs.

### 10.2 The loss and the backward pass

Per example: the mean of every column's cross-entropy over all `W·k` columns (each window's
token-softmax delta times `1/W`), plus `1/W` times the code losses of every snapped state, the
initial state's and each window's (`W + 1` terms).

Training is **backpropagation through time over windows with recomputation**
(`per_layer_gradients`):

1. Forward through all windows in training mode, keeping each window's input state and the
   generator state before it (not the activations).
2. Backward from the last window: restore the generator (so the dropout masks repeat), **re-run
   the window's forward pass**, then backward: the recurrence's backward (codes, GRU, pooling)
   gives the pooled tokens' gradient and the previous state's; the window's layers then
   backpropagate, with the pooling gradient injected where the poolings read (the `_WithPooling`,
   `_Pooled`, `_Merged` adapters add it to the reading layer's downstream); each `StateInput`'s
   state gradient `(Σ_t δ_t) P` is added to `dc`.
3. Finally the initial state's gradient through its snap.

This is per-window activation checkpointing: memory holds one window's activations at a time. The
Rust network instead runs the whole recurrence in one crate call (the "window program",
`core/src/program.rs`): the window's layers compiled to a list of ops at construction, every window
forward and back in one call, each window's results and masks kept (no recomputation), and the same
bits as the per-layer path.

### 10.3 Training-schedule controls (training state, not saved with the model)

- `set_commitments`: change `β` per factor during training.
- `set_temperatures`: **soft quantisation**: `q = Σ_k w_k e_k`, `w = softmax(−d/τ)`; loss
  `β‖h − sg[q]‖² + Σ_k sg[w_k]‖sg[h] − e_k‖²`; gradient to `h` through the soft weights, or with
  `set_straight(True)` straight through. Only the training step's windows are soft; every
  inference pass and the initial state stay hard.
- `set_continuous`: bypass the snap for a factor (warm-up); codes unread.
- `set_couplings`: next state `(1 − l)·h + l·q`, a gradual hand-over from continuous to snapped;
  the snap path takes `l·dq`, the cell `(1−l)·dq`.
- `consolidate`: map duplicate joint codes to a representative; the commitment then pulls `h`
  toward the representative's codes.
- `CodesArrayLayer.wall`: a "snap wall" margin loss on the ratio of the nearest and next-nearest
  code distances, gated to examples whose every digit is right. numpy only, and marked retired in
  the code.

### 10.4 The reference window (`specs/window_presets.py`)

`transformer_window`: width `D = 16`, 3 blocks, 2 heads (`d_k = 8`), FFN width 32 (2×D), ReLU,
causal attention, a learned Position; each block is `StateInput, Residual(LN, Attention),
Residual(LN, Dense(32, relu), Dense(16, linear, bias))`; then `LayerNorm` and a token-wise softmax.
Variants: `branched_window`, `recombined_window`, `answer_branch_window`, `linear_window` (no
nonlinearity but the LayerNorms, so the recurrence must do the work) and `ffn_window` (no
attention). A window is restricted to ReLU, ungrouped attention (`validate_window_specs`), because
the Rust window program implements only those ops.

## 11. Decoding and generation

`generation/decoder.py` and the decoding protocol (`protocols/decoding_protocols.py`): each token
layer has `start_decoding`, `prefill(X)` and `step(x)`. Token-wise layers decode row by row with no
cache. A causal attention keeps a **KV cache** of rotated keys and values (`(N, c, g·d_k)`), drops
the oldest past `T` (a sliding window), and attends through the same tiled kernel, its key blocks
cut at the same absolute columns as a full pass so the online softmax matches it block for block.
A non-causal attention refuses to decode. With learned positions, a token past `T` has no row and
the decoder falls back to full passes over the last `T` tokens. Batches of prompts are left-padded
with the pad id when the model has segments.

`generation/sampling.py`: filters (temperature as a softmax of `log p / t`, top-k, top-p), each
renormalising by a left fold, then a draw by cumulative left fold against one uniform.
`tokenizer.py` is a character tokenizer.

## 12. Numerics, determinism and the three implementations

Determinism is a first-class requirement, and it shapes the arithmetic everywhere:

- **Fixed summation order.** Every non-product sum is a left fold in a stated order (numpy through
  `cumsum`, never pairwise `.sum()`); shared sums across factors start from the first term, not
  0.0, so that a `−0.0` survives.
- **Rust products** (`core/src/linalg.rs`): every output is one FMA chain in a fixed grouping. The
  dot product uses 4 interleaved lanes combined `(l0+l1)+(l2+l3)`, identically in the scalar,
  AVX2/FMA and NEON paths, so a product has the same bits on every machine. Threading (above 8 M
  flops, up to 8 threads) splits whole output rows, so it can't change a value. A batched forward
  row equals the single-example forward exactly. `codegen-units = 1` keeps inlining stable.
- **numpy products are BLAS's**, so numpy and Rust agree only to within rounding (the code says so
  and tests to a tolerance), while pure Python and numpy share expressions and grouping.
- **Correctly rounded `exp` in Rust** (`core/src/cr_exp.rs`, a port of CORE-MATH's `cr_exp`): its
  bits are defined by mathematics, so every machine agrees. numpy uses `np.exp`, which isn't
  correctly rounded; tests swap the module-level `exp` to compare implementations bit for bit.
- **erf** is a port of fdlibm's in all three; the **rotary table** is computed once in Python;
  `powi` replaces `pow`; ReLU normalises `−0.0`.
- **Random numbers**: `pcg64.py` (pure Python) and `core/src/generator.rs` port numpy's
  `default_rng` (SeedSequence → PCG64 → doubles) bit for bit and share numpy's state layout. A
  network owns one generator for its initial weights and every dropout mask, so a seed fixes the
  run. The crate also ports the legacy MT19937 global stream.
- **Fused Rust ops** (`core/src/fused.rs`): one call per layer method. The code's own reason: the
  Python/Rust call count, not per-call cost, dominates at these sizes.

## 13. Persistence

"Format 2" (`persistence/format2*.py`) is one JSON document per network: the implementation, the
shape, the input shape (and bounds), the layer specs, the recurrence, the update rule, every
parameter as nested lists, the optimiser state (`t` and per-layer moments, by parameter name), and
the generator state. A checkpoint restores training by bits; a model saved by one array backend
loads into the other. Presets also record their constructor arguments; legacy envelopes still
load.

## 14. Design decisions, summarised

| Area | Decision in the code | The code's stated reason or evident purpose |
|---|---|---|
| Differentiation | hand-written per-layer backward; no autograd, no graph | three implementations must do identical arithmetic in identical order |
| Graph | a flat chain; residual/branch skips are layer references | the forward and backward loops stay single loops over the list |
| Activations | fused into layers | every Rust op is fused; fewer boundary crossings |
| Loss | fused into the output layer's delta | the classic `a − y` simplifications; no loss object |
| Gradients | summed; the optimiser divides by B | one convention for single-example and batch steps |
| Step order | every layer's gradient accumulated, then every update | global-norm clipping needs the full gradient |
| Specs | backend-free dataclasses, validated to what every backend builds | parity-testable by construction; one description, three implementations |
| Init | uniform `±1/√fan_in` everywhere (incl. ReLU), positions and pooling query zero | from the same seed, numpy, Rust and Python draw the same weights |
| Norms | BN as a separate linear + norm pair, no bias before BN; LN pre-block | the bias is redundant under BN; the transformer blocks are pre-LN |
| Weight decay | weights only; biases, norms, tables, codes, queries, initial states exempt | the standard AdamW parameter grouping |
| Attention | tiled online softmax in all implementations; masks per block; no `(T, T)` storage | memory linear in `T`; skipping blocks is exact under sequential folds |
| Attention dropout | mask regenerated from a saved generator state | no quadratic mask storage |
| Positions | learned table or RoPE, not both; RoPE table shared | identical `cos`/`sin` bits across implementations |
| Determinism | fixed folds, fixed FMA grouping, correctly rounded exp in Rust, ported RNGs | results reproducible across machines and implementations |
| Training | pocket on training accuracy; resumable bit for bit; schedules as data | runs are records; a resumed run equals an uninterrupted one |
| Recurrence | windows of a transformer, a GRU per state factor, VQ snap with straight-through, attention pooling | a discrete state the network can be read as a finite-state machine from |
| Recurrence training | BPTT over windows with per-window recomputation (numpy), or one crate call (Rust) | memory of one window; speed in Rust |
| Soft/hard state controls | temperature, continuous, coupling, straight, consolidation: training state only | state-formation schedules without changing the model |

## 15. Points to check against the literature, and findings

### 15.1 Points to check (design choices whose standing the literature pass should settle)

1. **Initialisation**: uniform `±1/√fan_in` for ReLU/GELU layers rather than He initialisation; the
   embedding table drawn uniform `±1/√d` (not `N(0, 1)` or `N(0, 0.02²)`); zero-initialised learned
   positions.
2. **Default loss**: sigmoid output with squared error is the default `Dense(output=True)`; check
   which presets and studies still use it.
3. **BN running variance** uses the unbiased estimate while normalisation uses the biased one (the
   code says PyTorch's defaults); ghost BN moves the running averages once per group.
4. **Pocket on training accuracy**: the trainer restores the best-training-accuracy epoch; there is
   no held-out selection in the trainer.
5. **Attention dropout** drops after normalisation (`P ⊙ M / keep`) and draws masks for masked
   entries too (only the stream order depends on it).
6. **RoPE** uses rotate-half pairing (`j`, `j + d_k/2`), applied after the projection biases.
7. **GQA conversion** by mean-pooling K/V heads, then uptraining from a fresh optimiser state.
8. **Window FFN width** is 2× the stream width (32 at `D = 16`), not the usual 4×.
9. **Attention pooling** has no `1/√d` scaling and no key/value projections; its query starts at
   zero.
10. **GRU convention**: `h = z ⊙ c + (1 − z) ⊙ h̃` (`z` keeps), reset applied before `U_h`
    (`U_h(r ⊙ c)`, the original form, not the cuDNN `r ⊙ (U_h c)` variant); update bias 1.0.
11. **VQ**: codebook learned by gradient (no EMA, no restarts), `β = 0.25`, nearest by squared
    distance, codebook initialised uniform `±1/√s`; soft quantisation by `softmax(−d/τ)` and the
    coupling hand-over are this code's own mechanisms; check what the literature has.
12. **Code-loss weighting**: `1/W` per snapped state for `W + 1` states, against a CE that is a mean
    over all columns.
13. **State injection**: an additive `P c` to every token before each block (a learned bias per
    block); compare with prefix tokens, cross-attention, or FiLM-style conditioning.
14. **Gradient clipping** on the norm of the mean gradient (`‖Σg‖/B`), which matches clipping
    PyTorch's mean-loss gradients.

### 15.2 Findings from the audit

1. **The Rust optimiser doesn't weight-decay the GRU's recurrent matrix `U`; numpy does.**
   `GRURustArrayLayer.parameters()` is `(W, U, b)` with `decayed = (True, True, False)`
   (`layers/rust/recurrent_rust_array_layer.py:138`). `RustOptimizer._pairs`
   (`optimizers/rust_optimizer.py:50`) groups parameters two at a time, so `(W, U)` becomes one
   fused-op pair with `U` in the bias slot, and `b` is paired with an empty array. The fused ops
   never decay their second member: `layer_l2_apply_accumulated_gradient` steps it with plain SGD
   (`rust/core/src/fused.rs:425`), and the Adam op applies `weight_decay` "on the pair's weight" only
   (`rust_optimizer.py:186`). `NumpyOptimizer` decays each parameter by its own flag, so it decays
   `U`. Under `Adam(weight_decay > 0)` or `WeightDecay`, a window-recurrent network trains
   differently on Rust and numpy. It's latent: no study in this repo pairs weight decay with a
   window-recurrent network, and no test covers the combination. **Fixed in #1199:** a parameter
   takes the next as its pair's second only when the next isn't decayed, so the GRU steps `W`
   alone, then `(U, b)`; `tests/model/optimizers/test_weight_decay_pairs.py` holds both backends to
   it.
2. **A stale crate comment**: `rust/core/src/recurrent.rs:26-27` says the sigmoid and tanh use
   "Rust's `f64::exp`, as `exp` (`ufuncs.rs`) is", but the module imports
   `cr_exp::{exp_in_place, sigmoid_in_place}` and `ufuncs::exp` is the correctly rounded one.
3. **The parity claims are scoped**: numpy against Rust is "within rounding" (BLAS products,
   `np.exp`); bit-identity holds across pure Python, numpy and Rust only where `exp` is the same
   implementation, which the tests arrange by swapping `exp`. A reader of results across backends
   should expect last-bit differences between a numpy run and a Rust run.
4. **The snap wall is numpy-only** and refused by the Rust window program; the code marks it
   retired. It remains a training-path branch in `CodesArrayLayer`.

## 16. The design against the literature

Each decision in §14 and point in §15.1, checked against the paper it rests on, read in full where
it bears on the decision. Sources are cited by their key in `indrajala-literature/library.bib`.
The 19 papers fetched for this pass are held in `indrajala-literature-private` (none is openly
licensed), with their entries and rows in the public repository. The pocket algorithm's own paper
(`gallant-1990-perceptron-based-learning-algorithms`, doi:10.1109/72.80230), which the owner
obtained, is held there too; it moved the pocket's verdict from adverse to consistent (16.3.8).

Verdicts:

- **Consistent:** the code does what the source says.
- **Equivalent:** it differs in form but not in what it computes or learns.
- **Divergent (benign):** it differs from the source, but the source gives no evidence against the
  difference at this code's scale or use.
- **Adverse:** the source reports evidence against the code's choice.

### 16.1 Consistent or equivalent

| Decision | Source | Verdict |
|---|---|---|
| Adam's moments, bias corrections and `ε` outside the square root; defaults `0.9, 0.999, 1e−8` | `kingma-2015-adam`, Algorithm 1 | consistent |
| Momentum `u ← m u + g/B; w ← w − lr u`; coupled L2 `w − lr(g/B + λw)` | `goyal-2017-accurate-large-minibatch-sgd`, eqs. (8), (9) | consistent |
| No weight decay on BN's `γ, β` | Goyal et al. ("we do not apply weight decay on the learnable BN coefficients") | consistent |
| Linear warmup | Goyal et al. §2.2 | consistent |
| AdamW: decoupled decay on the weights | `loshchilov-2019-decoupled-weight-decay`, Algorithm 2 | equivalent: the paper decays by `η_t λ θ`, the schedule multiplier without `α`; the code (PyTorch's grouping) by `lr·λ·θ`, so its `λ` is the paper's divided by `α`. Same family, different units for `λ` |
| Clipping the gradient's global norm | `pascanu-2013-difficulty-training-rnn`, Algorithm 1 | consistent (on the mean gradient, matching a mean loss) |
| BN: bias dropped before BN, BN before the nonlinearity, biased variance in training, unbiased `m/(m−1)` for inference | `ioffe-2015-batch-normalization`, §3.2 and Algorithm 2 | consistent |
| Ghost BN: each ghost group normalised alone, the running averages moved group by group | `hoffer-2017-train-longer-generalize-better`, Algorithm 1 and the text after it (the paper reports that frameworks' equal-weight update worsens generalisation; the code doesn't do that) | consistent (the paper tracks a running σ, the code a running variance: equivalent) |
| Layer norm per token, `γ = 1, β = 0` | `ba-2016-layer-normalization` | consistent |
| Pre-LN blocks with a final LayerNorm | `xiong-2020-layer-normalization-transformer` (pre-LN's gradients are well behaved at initialisation; warmup becomes optional) | consistent; the recipe's warmup is harmless, not required |
| Tiled attention: online softmax, `m` and `l` kept, `D = rowsum(dO ∘ O)` in the backward, dropout masks regenerated | `dao-2023-flashattention-2`, Algorithms 1 and 2 | consistent |
| RoPE, base 10000 | `su-2021-roformer-rotary-position-embedding` | equivalent: RoFormer pairs features `(2i, 2i+1)`; the code pairs `(j, j + d_k/2)` (rotate half). A fixed permutation of the query and key features, absorbed by the learned projections |
| GELU, exact and the tanh form (`0.044715`) | `hendrycks-2016-gelu` | consistent |
| GLU FFN (SwiGLU, GEGLU), `size ≈ 8d/3` at equal parameters | `shazeer-2020-glu-variants-improve-transformer` | divergent (benign): Shazeer omits the biases (T5's convention), and the code keeps them |
| GQA: query head `i` reads key/value head `i // (h/g)`; conversion by mean-pooling K and V heads, then uptraining | `ainslie-2023-gqa-grouped-query-attention`, §2.1 | consistent |
| GRU equations, reset inside `U_h(r ⊙ c)`, `h = z c + (1−z) h̃` | `cho-2014-rnn-encoder-decoder-gru`, eqs. (5)-(8) | consistent |
| GRU update-gate bias 1.0, toward keeping the state | `jozefowicz-2015-empirical-exploration-recurrent-architectures` (bias 1 on the LSTM's forget gate) | consistent by analogy: `z` is the GRU's keep gate |
| VQ: nearest code, straight-through, codebook `‖sg[h] − e‖²` and commitment `β‖h − sg[e]‖²`, `β = 0.25` | `van-den-oord-2017-vq-vae-neural-discrete-representation`, eq. (3) | consistent |
| Soft codes `softmax(−d/τ)` annealed toward hard | `agustsson-2017-soft-to-hard-vector-quantization`, eq. (7) (`σ = 1/τ`) | consistent |
| Straight-through estimator | `bengio-2013-straight-through-estimator` | consistent |
| BPTT over windows with per-window recomputation | `chen-2016-training-deep-nets-sublinear-memory` | consistent |
| Inverted dropout (scale in training, not at test) | `srivastava-2014-dropout` (scales the weights at test time) | equivalent |
| Cosine decay to a floor | `loshchilov-2017-sgdr-warm-restarts` | consistent |
| MRI: the wrong-side unit nearest zero adapts, its weights moved along its input | `widrow-1990-30-years-adaptive-neural-networks`, §V | consistent in kind; see 16.3.5 |
| The pocket: the best epoch's network kept by its training accuracy, replaced only when strictly better, and restored at the end | `gallant-1990-perceptron-based-learning-algorithms`, §III-A and Fig. 1 (the pocket algorithm with ratchet) | consistent; see 16.3.8 |

### 16.2 Adverse

1. **The default output loss is the squared error on a sigmoid.** `Dense(..., output=True)` defaults
   to `loss="squared"`, and the sigmoid presets train with it. Glorot & Bengio
   (`glorot-2010-understanding-difficulty-deep-feedforward`, Fig. 5 and the text beside it) find that
   cross-entropy "worked much better (for classification problems) than the quadratic cost", which
   shows "more severe plateaus". They note the observation goes back to Solla et al. 1988. The
   `a(1−a)` factor in the squared-error delta vanishes on a saturated wrong output, which is the
   plateau. *Change:* make the output layer's loss an explicit argument with no default, and have
   the legacy presets name `"squared"`, so no existing bit moves and no new network gets the
   squared error by omission. A classifier should use sigmoid with BCE, or softmax with CE.
   *Done* (the audit-remedies workplan, stage 2): `Dense.loss` is `None` on a hidden layer and
   named on the output layer, which validation requires.
2. **The codebook starts at a scale and place unrelated to the states it quantises, learns only by
   its selected codes' gradients, and never revives an unused code.** Codes are drawn uniformly in
   `±1/√s`, independent of where the GRU's outputs `h` lie, and are stepped only by the codebook
   loss of the codes chosen.
   - `lancucki-2020-robust-training-vq-bottleneck` finds that codebook learning "can suffer from
     poor initialization", sensitive to the codebook's scale relative to the encoder outputs (its
     Fig. 1). It fixes this with data-dependent initialisation, periodic re-initialisation from the
     encoder outputs, and a higher codebook learning rate.
   - `dhariwal-2020-jukebox` (§3.1) restarts dead codes near random encoder outputs.
   - `zheng-2023-online-clustered-codebook` re-anchors unused codes to encoder features.
   - `huh-2023-straightening-straight-through-estimator` traces instability to codebook gradient
     sparsity and the commitment loss's asymmetry.
   - `zhu-2024-simvq-representation-collapse-one-linear-layer` traces collapse to "disjoint
     codebook optimization, where only a few code vectors are updated".

   For this code the stakes differ from the papers': it wants a *small* set of used codes (a
   minimal machine), so an unused code isn't by itself a failure. But two parts of the findings
   carry over directly: a codebook that starts away from the states, and a used code that drifts
   from the states it serves. *Change, as candidates to test, not defaults:* initialise each
   factor's codebook from its first states (k-means++ or random distinct samples of `h` over a
   first batch), and add an option to re-seed a code unused for a window of steps, from the
   states. Both are training-state options, like the temperatures and couplings already there,
   and are judged by their fresh-seed success rates.

   *Corrected 2026-11-02 (the remedies workplan):* the library's default is the random draw, as
   above, but the research harness already initialises codes from the states. The state-formation
   workplan's k-means codes (its D6), in indrajala-studies (`modular_family.py`, `kmeans` and
   `codes_from_states`), run k-means++ and Lloyd's iterations over each factor's cells' outputs at
   the warm-up's end. The audit, written from this repository alone, didn't see it. What neither
   repository has is re-seeding dead codes, a codebook learning rate and SimVQ's
   reparameterisation. The owner reports codebook collapse as a recurring failure. The plan,
   [audit-remedies-workplan.md](workplans/library/audit-remedies-workplan.md), promotes the
   k-means codes into the library and builds the other three, as options a registered study judges.

### 16.3 Divergent (benign), and optional changes

1. **Initialisation.** Uniform `±1/√fan_in` has variance `1/(3·fan_in)`, PyTorch's `nn.Linear`
   default. `he-2015-delving-deep-rectifiers` derives `2/fan_in` for ReLU and reports that
   Xavier-scaled initialisation fails to converge at 30 layers. `chowdhery-2022-palm-scaling-pathways`
   uses `N(0, 1/fan_in)` for kernels and `N(0, 1)` for embeddings, "since layer normalization is
   not applied to the embeddings". Here the deep networks are pre-LN residual transformers, whose
   normalisation and identity paths make the initial scale matter much less, and the plain ReLU MLPs
   and conv nets are a few layers deep. So nothing here is shown to suffer. *Optional:* a He
   initialiser for ReLU and GELU layers and conv kernels, as a spec option, would let a deep plain
   network be built correctly.
2. **State injection is additive** (`x_t + P c`, FiLM with `γ := 1`). In
   `perez-2018-film` (Table 2) restricting FiLM to `γ := 1` drops CLEVR accuracy to 95.9 against
   97.4-97.7 for the full model, and the authors find that FiLM "largely conditions through `γ`".
   The task and the conditioning signal differ, so this is evidence, not proof, for the window. *Optional:* a
   multiplicative `StateInput` (`x_t ⊙ (1 + G c) + P c`), tested against the additive one on
   fresh seeds.
3. **Attention pooling** is one learned query with no key or value projections and no `1/√d`
   scale. The Set Transformer's pooling (`lee-2019-set-transformer`, PMA) is multi-head attention
   from learned seeds with projections, followed by a feed-forward layer. The code's pooling is a
   deliberately minimal summary feeding a GRU, which has its own input projection, and no source
   shows the simpler form failing. No change suggested.
4. **The window's FFN is 2× the stream width.** `vaswani-2017-attention` uses `d_ff = 4·d_model`
   (2048 at 512). A capacity choice for a 16-wide window, not a correctness issue.
5. **MRI adapts one unit per error.** Widrow & Lehr's MRI adapts every unit needed to correct the
   output (for an OR output wrongly on, every unit giving +1), choosing the ones nearest zero. The
   code adapts the single nearest unit per presentation, so an error needing several flips takes
   several presentations. A legacy, non-gradient demo network; correct it if the network is used
   again.
6. **Embedding initialisation** `±1/√d` against PaLM's `N(0, 1)`. Under pre-LN blocks the embedding's
   scale meets a LayerNorm before every sublayer. Benign.
7. **AdamW's `λ`** is in PyTorch's units (decay `lr·λ`), not the paper's (`η_t·λ`). Record the
   convention with every run that reports `λ`; no change.
8. **The pocket selects on training accuracy.** This is the pocket as Gallant defines it
   (`gallant-1990-perceptron-based-learning-algorithms`, §III-A, Fig. 1).
   - The pocket keeps the weights that correctly classify the most training examples, which he
     calls *optimal*.
   - The ratchet replaces them only when the new count is strictly higher (step 3bb).
   - Theorem 1 is about the pocket weights becoming optimal on the training set, with probability
     approaching 1.
   - Gallant measures generalisation separately: on "new test data" with learning turned off, and
     by the distribution-free bounds of his §VI. He doesn't select on it.

   The trainer does the same per epoch, the ratchet over whole-training-set accuracy, with a
   strictly-better test. It differs in two respects, neither adverse:
   - Gallant first nominates a candidate by its run of consecutive correct classifications, then
     checks the full count. The trainer checks the full count every epoch.
   - Gallant's motivation is perceptron learning on nonseparable data, which "can go from an
     optimal set of weights to a worst-possible set in one iteration". Gradient training is
     smoother, so the pocket matters less there but costs nothing.

   An earlier draft of this section counted the pocket as adverse, from
   `prechelt-1998-early-stopping-but-when`. Prechelt's validation-based early stopping answers a
   different question: when to stop so that the *held-out* error is lowest. For the tasks that
   certify, fitting the training set is the goal, which is Gallant's criterion. For a held-out
   claim made from the returned network, Prechelt applies, and the network should be selected on
   a validation set. *Optional:* a selection-set argument for the trainer, recording which set
   selected, for runs whose claims are held-out ones.

### 16.4 What this changes

| Item | Kind | Recommendation |
|---|---|---|
| The output layer's default loss (16.2.1) | adverse | make it explicit; presets name `"squared"`; no bits move |
| The codebook's start and its unused codes (16.2.2) | adverse, for this code's aim in part | data-dependent initialisation and optional re-seeding, as training options, judged on fresh seeds |
| The trainer's pocket on training accuracy (16.3.8) | consistent with Gallant; optional for held-out claims | a selection-set argument for runs whose claims are held-out ones |
| He initialisation (16.3.1) | optional | a spec option |
| A multiplicative state input (16.3.2) | optional | a variant to test |
| MRI (16.3.5) | legacy | fix if used again |
