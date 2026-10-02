# Workplan: layer norm and single-head attention

**Status: in progress; decisions D1-D12 settled (2026-10-02). Stages 0, 1 and 2 done.**

Roadmap step 4 ([primitives-roadmap.md](primitives-roadmap.md)). A small vision transformer on
MNIST (Dosovitskiy et al. 2020, "An Image is Worth 16x16 Words", arXiv 2010.11929): the image cut
into 7x7 patches, 16 tokens, each embedded, given a position, passed through a pre-LN attention
block, `x + Attention(LN(x))` (Xiong et al. 2020, arXiv 2002.04745), averaged over the tokens and
classified. This plan adds token sequences, layer norm and single-head self-attention to the layer
specs and builds them in all three implementations (pure Python, numpy, Rust), with format-2
files, a gradient check, parity tests and a study against the conv results already measured.

## Why

- **Attention is the roadmap's next primitive**, and steps 2 and 3 (batch norm, residual blocks)
  were its prerequisites. Multi-head attention and the full transformer block are step 5; this plan
  builds the pieces they reuse: token sequences, layer norm, the attention layer's forward and
  backward passes, and residual blocks over tokens.
- **Layer norm is batch norm over the other axis.** It normalizes each example's (each token's)
  features, so it needs no batch statistics, no running averages and no single-example refusal:
  it trains on one example as it does on a batch.
- **Every current op multiplies weights by activations.** Attention multiplies two activation
  tensors (`Q K^T`, then `P V`) and takes a softmax over the sequence, inside a hidden layer. Both
  are new in all three implementations and in the crate.
- **MNIST already has measured baselines.** The conv presets and the dense networks give a
  patch model something to be compared against without new data loading.

## Decisions (settled 2026-10-02)

Each lists the options considered and the owner's choice.

- **D1. Scope: the attention block alone, or the token-wise feed-forward block too. Settled
  2026-10-02: (b).**
  - (a) The attention block only: `Residual((LayerNorm(), Attention()))`. The token-wise dense
    layers exist (for the patch embedding, D4), but validation refuses them inside a block until
    step 5.
  - (b) *Chosen.* Both blocks of a transformer layer: the attention block and
    `Residual((LayerNorm(), Dense(64, "relu"), Dense(32, "linear", bias=True)))` over tokens. With
    D4's token-wise `Dense` and token residual blocks (needed by the attention block anyway), the
    FFN block is a composition, not new arithmetic: it costs tests and a study arm. A patch model
    without it is weak, which would blur the study. Step 5 is then multi-head attention and what
    a deeper transformer needs (masking, dropout in attention).
- **D2. A token sequence's shape and layout. Settled 2026-10-02: (a).**
  - (a) *Chosen.* A new input shape, `(tokens, features)`, beside `(dimension,)` and
    `(height, width, channels)`, flat and token-major: index `t * features + j`, so each token is
    contiguous and a batch reshapes to `(N, T, d)` and `(N * T, d)` without a copy.
  - (b) Channel-major, as conv activations are (`c * P + p`), with batch norm's `_rows` /
    `_flat` transposes in every token layer. Every token op would pay a transpose.
- **D3. How an image becomes tokens. Settled 2026-10-02: (a).**
  - (a) *Chosen.* `Patches(7)`, a parameter-free layer that cuts an `(H, W, C)` image into
    `(H/7 * W/7, 7 * 7 * C)` tokens, patches in row-major order, each patch's values in the conv
    kernel's (channel, row, col) order; then a token-wise `Dense(d, "linear", bias=True)` (D4) is the
    embedding. Together they are a stride-7 convolution with a bias, as the ViT paper notes. A
    patch size must divide the image; `Patches` is the first layer, with no conv front end.
  - (b) One `PatchEmbed(patch_size=7, size=32)` spec, the reshape and the embedding fused.
    Fewer layers, but the embedding's arithmetic is a dense layer's, written a second time.
  - (c) `Conv(7, 32, stride=7)` then a layer transposing channel-major to token-major. Reuses
    conv, but conv has no linear-with-bias form (linear conv is bias-free, before a `BatchNorm`).
- **D4. `Dense` over a token sequence. Settled 2026-10-02: (a).**
  - (a) *Chosen.* The same `Dense` spec acts on each token, with weights shared over the
    tokens (PyTorch's `nn.Linear` on the last axis): `(T, d_in)` to `(T, size)`. The residual
    body rule (ends in `Dense(n, "linear", bias=True)`) holds unchanged. A token-wise dense layer's
    activation is ReLU or linear-with-bias (D1's FFN, D3's embedding), with no dropout; sigmoid and
    `BatchNorm` over tokens are refused. The output `Dense` reads a flat input, so a token
    sequence ends in `TokenMean` (D8) before it. The cost, accepted: `Dense` means "the whole
    input" after a conv front end (it flattens the image) and "each token" after `Patches`, so
    a spec's meaning depends on its input shape, and its checks on where it stands.
  - (b) A separate `TokenDense` spec. Explicit, but a second spelling of the same layer, and
    `Residual`'s body rule needs a second form.
- **D5. Layer norm's form and where it may stand. Settled 2026-10-02: (b).**
  `LayerNorm(epsilon=1e-5)`: each token's (or a flat layer's) features normalized by their mean
  and (biased) variance, then `gamma * xhat + beta` per feature,
  gamma starting at 1 and beta at 0, nothing drawn; no activation after it (pre-LN). Same in
  training and inference.
  - (a) On token sequences, and on the flat vector after `TokenMean` (ViT's final
    norm). No Rust dense layer ever comes right before one, so the fused hidden delta that reads the
    next layer's `W` (residual-connections D8) needs no new case, and no existing network's
    arithmetic moves.
  - (b) *Chosen.* Also in flat dense networks, wherever a dense hidden layer may stand (in a
    residual body too), as an alternative to batch norm: a general primitive, so layer norm
    against batch norm can be compared on the existing networks. A flat `LayerNorm` normalizes
    each example's features. The cost, accepted: a Rust sigmoid, ReLU or dropout layer right
    before a `LayerNorm`, or before a fork whose body starts with one, has no `W` to read for its
    fused hidden delta (residual-connections D8). It takes `next_layer.downstream_batch()` and a
    mask op instead: `array_relu_mask` exists, and the crate gains `array_sigmoid_mask` and
    `array_dropout_mask`, single and batch (the unfused path residual-connections D8 (a)
    described). Those layers' hidden-delta methods gain a branch, so stage 4 has an A/B. In the
    pure-Python per-node networks, a flat `LayerNorm` couples every node of its layer, as the
    softmax output layer does, so it is a layer-level class there too.
- **D6. The attention layer's form. Settled 2026-10-02: (a).** `Attention()`, single head, keys
  as wide as the tokens (`d_k = d`), no mask (an image's tokens all see each other):
  `Q = X Wq^T + bq`, `K`, `V` likewise, `P = softmax_rows(Q K^T / sqrt(d))`, `out = (P V) Wo^T + bo`.
  - (a) *Chosen.* Biases on all four projections (PyTorch's `nn.MultiheadAttention` default,
    and ViT's `qkv_bias`), so the parameter shapes match the reference models. The output
    projection makes the layer end affine, so it can end a residual body as
    `Dense(n, "linear", bias=True)` does. Known and accepted: `bk` has no effect on `P` (it adds
    `q_i . bk` to every score in row i, and softmax ignores a constant per row), so its gradient is
    exactly 0 in exact arithmetic and rounding noise (~1e-17) in floating point, which differs
    between numpy and Rust; and `bv` duplicates `bo` (each row of `P` sums to 1, so `P (V + bv) =
    P V + bv`). Parity tests explain `bk`'s noise gradient rather than compare it to a tolerance.
  - (b) Biases on the output projection only: no dead or redundant parameter, five arrays instead
    of eight, but it departs from the reference models and drops `bq`, the one input bias with an
    effect (`bq . k_j`, a key-dependent term).
  - (c) `bq` and `bo` only: every bias with an effect, but no reference model's shape.
  - Either way, a key size other than `d` (`Attention(key_size=...)`) and masking go to step 5.
- **D7. Positions. Settled 2026-10-02: (a).**
  - (a) *Chosen.* `Position()`, a learned `(T, d)` table added to the tokens (the ViT paper's
    kind), starting at zero. It draws nothing, so adding it never shifts a later layer's draws
    (the fork's rule), and a no-positions study arm differs from the full model by that layer
    alone. Not decayed. It learns from the first step: each position's gradient is its token's
    delta summed over the batch, so positions part after one step; only step 0 is blind to patch
    order. The cost, accepted: a new parameter kind in all three implementations, one added to
    activations with no inputs (optimizer state, snapshot, format 2, checkpoint, gradient check,
    and in pure Python a node with weights and no inputs).
  - (b) Learned, drawn uniformly at initialization, ±0.02·√3 to match ViT's truncated normal's
    standard deviation (normal draws don't exist here, rng-audit.md). The same new parameter kind
    as (a), plus 512 draws that shift every later layer's initialization, for little: the
    embedding's outputs are ~0.3 at initialization and the `LayerNorm` before attention rescales
    them, so a table of ±0.035 barely registers at step 0.
  - (c) Fixed 2D sin-cos (Beyer et al. 2022, "Better plain ViT baselines", arXiv 2205.01580). No
    parameters, and bit-identical by construction if the table is computed once with `math.sin`
    and `math.cos` and handed to numpy and Rust. But `Position` needs the patch grid, not only
    `T`, `d` must be divisible by 4, and positions can't adapt.
- **D8. The readout. Settled 2026-10-02: (a).**
  - (a) *Chosen.* `TokenMean()`, the mean over the tokens, `(T, d)` to `(d,)`, parameter-free,
    nothing drawn (global average pooling; Beyer et al. 2022 find it as good as a class token).
    With `T = 16`, `1/16` is exact, which keeps the uniform-attention test exact by bits. The
    readout attends to nothing, so the only attention maps are token to token.
  - (b) A learned class token prepended to the sequence, its final value read out (the ViT
    paper's). Attention does the pooling, but it is one more parameter kind (concatenated onto
    the sequence), the sequence becomes `T + 1 = 17` (so `1/17` is inexact and the
    uniform-attention test loses its exactness), and Dosovitskiy et al. needed another learning
    rate for it to match average pooling.
- **D9. Rust: fused ops per layer, or general primitives. Settled 2026-10-02: (a).**
  - (a) *Chosen.* One fused crate call per pass per layer, as batch norm has
    (`batch_norm_forward_batch`, `batch_norm_downstream_batch`): `layer_norm_forward(_batch)`,
    `layer_norm_downstream_batch`, `layer_norm_accumulate_gradient_batch`,
    `attention_forward(_batch)`, `attention_downstream_batch`, `attention_accumulate_gradient_batch`.
    One crossing per pass, and the crate controls every sum's order (the README's folds and
    attention's three-term `dX`). The token-wise dense layer reuses the existing dense ops on the
    `(N * T, d)` view (D2). The building blocks the bit tests compare against (batched matrix
    products, a row softmax and its backward) are crate-internal functions, not exposed. The cost,
    accepted: attention's backward op is long, and step 5's heads extend or rewrite it; the
    signatures leave room for a head count.
  - (b) General primitives (batched `matmul` over `(N, T, d)` stacks, a row softmax and its
    backward) composed in Python: small, reviewable, reusable by step 5, and the Rust layer reads
    as the numpy one, but six to ten crossings and intermediate arrays per pass where (a) has
    one, and attention is the one Rust layer that isn't a fused call.
  - (c) (b) first, then fusion as a measured optimization through `ab.py` if the crossings show.
- **D10. Pure Python. Settled 2026-10-02: (a).**
  - (a) *Chosen.* All three implementations, as every primitive so far. `Patches`, `Position`,
    `LayerNorm` (token and flat, D5), `Attention`, `TokenMean` and the token-wise dense layer are
    layer-level classes in the way `ConvLayer` is: plain-Python loops over tokens and features,
    the README's expressions written out, with `.nodes` the later layers read. The golden run and
    the format-2 fixtures get pure-Python patch-model entries. The cost, accepted: stage 3 is
    likely the largest, and the layer-major batch path needs a per-example home for attention's
    layer-level caches (stage 3 settles it). Pure Python is parity-only, never timed.
  - (b) numpy and Rust only. Saves stage 3, but breaks the three-implementation invariant for the
    first time (the builders would need a "not in pure Python" refusal), leaves the pure-Python
    golden entry and fixture out, and makes D5's flat `LayerNorm` the one normalization pure
    Python lacks.
- **D11. The study. Settled 2026-10-02: (a).**
  - (a) *Chosen.* `scripts/patch_attention_study.py`, numpy, MNIST, as
    `residual_depth_study.py`: d = 32, 16 tokens; arms (1) patches, embedding, positions, FFN
    block, mean, norm (no attention: tokens never see each other until the mean); (2) the
    attention block alone; (3) attention then FFN (D1); and the controls, the conv preset and a
    dense network of about the same parameter count, rerun under the same protocol. Adam, one
    learning rate per arm from a short sweep, 3 seeds, 5 epochs. Recorded per cell: test
    accuracy, parameter count, seconds per epoch. One attention epoch is timed first and the grid
    sized from it; the PR says so. The expected result: arm 3 beats arm 1, and the conv network
    beats them all at this data size (ViTs need more data; Dosovitskiy et al. 2020). If not, the
    study says so.
  - (b) (a), plus the residual depth study's batch-norm cells rerun with `LayerNorm` in place
    (D5's flat layer norm against batch norm). Next-steps under (a).
  - (c) No study in this plan; a demo network only, and the study to next-steps.
- **D12. Sequential only, no preset class. Settled 2026-10-02: (a).**
  - (a) *Chosen.* Patch models are built with `SequentialArrayNetwork` and the pure-Python
    `Sequential*` networks; no named class, no new legacy loader or fixture class (as
    residual-connections D10). The study builds its specs with a helper in its own script. A
    preset can follow if the study finds a configuration worth naming, under next-steps' rule for
    presets.
  - (b) A named preset per implementation: discoverable, but three classes with loaders and
    fixtures, the duplication next-steps holds off on, and hyperparameters frozen before the
    study.

## The design

Under the settled decisions.

### Specs

```python
@dataclass(frozen=True)
class Patches:
    """An (H, W, C) image as (H/p * W/p, p * p * C) tokens: row-major patches, each in the conv
    kernel's (channel, row, col) order. Parameter-free, the first layer."""

    patch_size: int


@dataclass(frozen=True)
class Position:
    """A learned (T, d) table added to the tokens, starting at zero."""


@dataclass(frozen=True)
class LayerNorm:
    """Each token's (or the flat vector's) features normalized, then gamma * xhat + beta."""

    epsilon: float = 1e-5


@dataclass(frozen=True)
class Attention:
    """Single-head self-attention over the tokens, ending in an affine output projection."""


@dataclass(frozen=True)
class TokenMean:
    """The mean over the tokens: (T, d) to (d,)."""
```

`LayerSpec` gains the five. The patch model over MNIST:

```python
[
    Patches(7),  # (28, 28, 1) -> (16, 49)
    Dense(32, activation="linear", bias=True),  # the embedding, per token
    Position(),
    Residual((LayerNorm(), Attention())),  # x + Attention(LN(x))
    Residual((LayerNorm(), Dense(64, activation="relu"), Dense(32, activation="linear", bias=True))),
    TokenMean(),  # (16, 32) -> (32,)
    LayerNorm(),
    Dense(10, activation="softmax", output=True, loss="cross_entropy"),
]
```

`validate_layer_specs` gains a token part, between `Patches` and `TokenMean`, beside the conv
front end and the dense part: a network has a conv front end or a token part, not both. In the
token part it accepts `Dense` (ReLU, or linear with `bias=True` (D4)), `Position` (once, before any
block), `LayerNorm` and `Residual`; a token block's body ends in `Dense(n, "linear", bias=True)` or
`Attention`. After `TokenMean` the dense part follows today's rules, in which `LayerNorm` may
stand wherever a dense hidden layer may (D5).
`spec_shapes` walks `(T, d)` shapes, and the residual block's flat-input check becomes "flat or
tokens".

### Layers

New per implementation: `PatchesLayer`, `PositionLayer`, `LayerNormLayer`, `AttentionLayer`,
`TokenMeanLayer`, and a token-wise dense layer (ReLU and affine). `Fork` and `Add` are elementwise
and take any shape. The network's forward and backward loops don't change: each layer reads only
the next layer's `downstream`, which every new layer provides. Snapshots and optimizer state:
`Position` (`P`), `LayerNorm` (`gamma`, `beta`, `GammaAndBeta`), `Attention`
(`Wq, bq, Wk, bk, Wv, bv, Wo, bo`, in that order, which is also the draw order). Weights are
decayed, biases, `gamma`, `beta` and `P` not.

### Arithmetic

Per example, `X` the `(T, d)` tokens, every sum a left fold in index order unless a matrix
product (BLAS on numpy, the crate's FMA products on Rust, as today):

```text
layer norm, each token t (a flat layer is one token)
  mu = sum_j x_tj / d;  v = sum_j (x_tj - mu)^2 / d;  xhat = (x - mu) / sqrt(v + eps)
  y = gamma * xhat + beta
  backward: dxhat = delta * gamma
            dx = (dxhat - sum_j dxhat_j / d - xhat * sum_j (dxhat_j * xhat_j) / d) / sqrt(v + eps)
  gradients: gamma += sum over examples and tokens of delta * xhat; beta += sum of delta

attention
  Q = X Wq^T + bq;  K = X Wk^T + bk;  V = X Wv^T + bv
  S = (Q K^T) / sqrt(d);  P = softmax over each row of S (max-shifted, as SoftmaxArrayLayer)
  H = P V;  out = H Wo^T + bo
  backward: dH = delta Wo;  dP = dH V^T;  dV = P^T dH
            dS = P * (dP - rowsum(dP * P));  dQ = dS K / sqrt(d);  dK = dS^T Q / sqrt(d)
            dX = dQ Wq + dK Wk + dV Wv          (three terms: a fixed order, stated in the README)
  gradients: Wo += delta^T H, bo += colsum(delta); Wq += dQ^T X, bq += colsum(dQ); K, V likewise

position   out = X + P;  dX = delta;  P += sum over examples of delta
token mean out = sum_t x_t / T;  dX_t = delta / T for every t
patches    a fixed permutation; dX the inverse permutation of delta
```

The README gains a "Layer norm and attention" section with these expressions in their exact
grouping, as batch norm's and residual connections' have. The README's forms bind the crate;
the left folds are numpy's `np.cumsum` along the axis (tests/test_summation_order.py).

### Parity standard

The batch-norm and residual workplans': crate ops equal their numpy expressions by bits where no
matrix product is involved, and the crate's own composition of its existing ops where one is (op
tests); whole networks are compared after 50 steps against the measured no-attention control,
and any gap is explained (BLAS against the crate's FMA products, Adam's steep step at |g| ≈ ε),
never accepted as a tolerance.

Exact tests that need no tolerance:

- **one token**: with `T = 1`, `P = [[1]]`, so attention is `(X Wv^T + bv) Wo^T + bo` by bits;
- **uniform attention**: with `Wq = Wk = 0` and `bq = bk = 0`, every score is 0 and every
  weight is exactly `1/16` (`T = 16`), so `H` is each column's mean of `V`, by bits against a
  `TokenMean` of `V` in the same fold order;
- **identity blocks**: as residual connections', with `Wo` and `bo` zero the attention block
  leaves every other layer's outputs and gradients unchanged by bits;
- **layer norm** against hand-computed values; `learn` and `learn_batch` of one example agree.

## Pitfalls to design around

- **Fused hidden deltas read the next layer's `W`** on Rust (residual-connections D8). Under D5
  (b) a Rust sigmoid, ReLU or dropout layer may come right before a `LayerNorm`, or before a fork
  whose body starts with one. Such a layer takes the downstream and a mask op (D5); every layer not
  in that position keeps its fused call, so existing networks' arithmetic doesn't change (the
  golden run checks), and the type check per layer per step is what the A/B measures.
- **`bk`'s gradient is rounding noise** (D6): numpy and Rust disagree in it by construction, and
  under Adam a noise gradient still takes steps. The parity tests say so and compare `bk` apart.
- **A fork's `body_first` may have no `W`.** Token blocks start with `LayerNorm`; the Rust fork
  already computes its delta from `body_first.downstream_batch()` when asked, and the layer
  before a token fork (`Position`, a token-wise dense layer, an `Add`) asks for it, never reads
  `body_first.W`.
- **Three-term gradient sums.** `dX` of attention sums three products; its order is fixed in the
  README and every implementation follows it. Residual connections had none.
- **Matrix products between activations.** `Q K^T` and `P V` per example: numpy's `np.matmul` on
  `(N, T, d)` stacks against the crate's dot products. They don't match by bits; the gap is
  explained as the dense layers' is.
- **Softmax over a hidden row.** Max-shifted as the output layer's, so exponentials can't
  overflow; numpy-vs-X comparisons use the `crate_exp` / `math_exp` fixtures.
- **`LayerMajorBatch`** (pure Python) keeps per-example state in node lanes (`example_fields`).
  Attention's per-example caches (`Q`, `K`, `V`, `P`) are layer-level; they need lanes too, or the
  batch path keeps them per example in another way. Decided in stage 3, by reading `layer_major.py`.
- **Indices.** "layer i" in messages, snapshots and optimizer state stay indices into
  `expand_specs`; the new layers are one expanded spec each.
- **No in-place writes on a shared input**, as residual connections pinned: `Position` and the
  fork return new arrays, never write `X`.
- **ab.py runs the new tree's script on both sides**, so the study's timing helper must work on an
  old tree.

## Stages

One PR per stage; a crate stage is a crate PR, then a "Bump rust/" PR here. Every stage passes
`./cli lint`, `./cli test` and the golden check bit-identical.

### Stage 0: the plan and the README section

This workplan (D1-D12 settled), then the README's "Layer norm and attention" section with the exact
forward and backward expressions. Docs only.

### Stage 1: specs

1. `Patches`, `Position`, `LayerNorm`, `Attention`, `TokenMean`; the `(tokens, features)` shape in
   `spec_shapes`; the token part in `validate_layer_specs`; residual blocks over tokens.
2. Both builders and `format2.layer_to_json` refuse the new specs with "not yet (the layer-norm and
   attention workplan, stage N)", as earlier workplans' stages did.
3. `tests/test_layer_specs.py`: accepted and refused cases (a patch size that doesn't divide the
   image, `Patches` after a conv layer, two `Position`s, a sigmoid or dropout token layer, a
   `BatchNorm` among tokens, a token body not ending affine or in `Attention`, a `TokenMean`
   without `Patches` before it; a flat `LayerNorm` before and after each hidden-layer kind and as a
   residual body's first layer, accepted).

No `learn*` path changes: no A/B.

### Stage 2: numpy

1. The six layer classes; the builder builds them.
2. `tests/test_attention_array_network.py`: the gradient check (`tests/gradient_check.py`) on patch
   models, with and without each block, single example and batch; the exact tests above.
   `tests/test_layer_norm_array_network.py`: the gradient check on dense networks with a flat
   `LayerNorm` (after sigmoid, ReLU and dropout layers, and first in a residual body).
3. A short MNIST smoke run in the test suite's time budget: the patch model learns (loss falls).

No existing layer changes: no A/B; the PR says so. If the PR is too large to review, it splits into
layer norm and tokens first, attention second.

### Stage 3: pure Python

1. The layer-level classes and their `.nodes`, the flat `LayerNorm` in the per-node dense
   networks included; `LayerMajorBatch` keeps their per-example state.
2. Parity with numpy after 50 steps at the standard above; the gradient check and exact tests on
   the pure-Python networks.

Pure Python is parity-only: no A/B.

### Stage 4: Rust

1. **Crate PR.** The layer-norm and attention ops (D9), and `array_sigmoid_mask` and
   `array_dropout_mask` (D5), single and batch, each tested by bits
   against the crate's own unfused composition (matrix products, row softmax, sums) and against
   numpy's expression where no product is involved; the patch permutation and position sum
   reuse existing ops if they can. The type stub; `cargo fmt` / `clippy`.
2. **Bump rust/.** The Rust layer classes; the token-wise dense layer over the `(N * T, d)` view
   with the existing dense ops. Parity with numpy after 50 steps; the gradient check and exact
   tests on Rust.
3. **A/B**: the Rust sigmoid, ReLU and dropout layers' hidden-delta methods gain the
   before-`LayerNorm` branch (D5), so `ab.py run --bench prepared_dataset_timing`, both `.so`
   hashes differing in the header. Existing networks never meet a `LayerNorm`, so every row should
   be within noise.

### Stage 5: format 2, checkpoints and the golden run

1. `layer_to_json` / `layer_from_json` for the five specs and token-wise `Dense`; `load_network`
   builds them. Existing files don't change.
2. Snapshots and optimizer state per expanded layer; checkpoint round trips resume by bits.
3. One format-2 fixture per implementation (a Sequential patch model after two steps, and a
   dense network with a flat `LayerNorm`); the
   existing fixtures regenerate byte-identically.
4. The golden run gains patch-model entries (numpy, Rust, pure Python); existing entries don't move.

### Stage 6: the study (D11)

As D11 settles it. Findings in the script's docstring; raw outputs under `data/attention/`,
untracked. Sweeps run with `OPENBLAS_NUM_THREADS=1`.

### Stage 7: docs and retirement

README Models mentions the new specs; the roadmap marks step 4 done and sets step 5's scope by what
this plan left. Only once every stage and decision is done: the workplan is deleted and its
leftovers move to next-steps.md.

## Out of scope

- Multi-head attention, a key size other than the token size, masking (causal or padding), dropout
  in attention: roadmap step 5.
- Group norm and instance norm.
- Layer norm against batch norm on the dense networks (D11 (b)): the residual depth study's
  batch-norm cells rerun with `LayerNorm`.
- A class token (D8 (b)), fixed sin-cos or drawn positions (D7 (b), (c)).
- Sequence data (text) and its loading; conv-then-tokens hybrids.
