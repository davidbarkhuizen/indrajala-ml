# Workplan: residual connections

**Status: in progress; decisions D1-D10 settled (2026-10-01). Stages 0-1 done.**

Roadmap step 3 ([primitives-roadmap.md](primitives-roadmap.md)). A residual block adds its input to
its body's output, `out = x + F(x)` (He et al. 2016, "Identity Mappings in Deep Residual
Networks", arXiv 1603.05027). This plan adds dense residual blocks to the layer specs and builds
them in all three implementations (pure Python, numpy, Rust), with format-2 files, a gradient check,
parity tests, and a small depth study that shows what they buy.

## Why

- **Attention needs them.** A transformer block is two residual blocks: `x + Attention(LN(x))`,
  then `x + FFN(LN(x))`. The FFN block, `x + W2 relu(W1 x + b1) + b2`, is exactly the dense block
  this plan builds (roadmap step 4).
- **Depth.** Sigmoid networks here stop improving after two or three hidden layers: the gradient
  shrinks by the activation's derivative at every layer. The identity path carries the gradient
  past every block unchanged, so depth stops costing gradient.
- **Batch norm matters more in deep networks.** The batch-norm results so far are on networks two
  or three layers deep. Blocks with `Dense(linear), BatchNorm(relu)` inside them give it networks
  where it should matter.

## Decisions (settled 2026-10-01)

Each lists the options considered and the owner's choice.

- **D1. How a block is spelled in the specs. Settled 2026-10-01: (a).**
  - (a) *Chosen.* A nested spec, `Residual(body=(Dense(64, "relu"), Dense(32, "linear",
    bias=True)))`. A block can't be malformed, and it reads as PyTorch's and Keras's do. The
    builders flatten it into layers (The design, below), so specs and layers are no longer one to
    one: the flattening is one function, `expand_specs`, and messages name the spec path.
  - (b) Flat marker specs, `Fork()` ... `Add()`, one layer each. Specs and layers stay one to one,
    which every builder, `spec_shapes` and every "layer i" message assume today. Brackets can be
    unbalanced, so validation checks them.
- **D2. Scope: dense blocks only. Settled 2026-10-01: (a).**
  - (a) *Chosen.* Dense blocks only. Validation refuses a block among the conv and pool layers.
  - (b) Conv blocks too. A conv block must keep its shape, and conv here is 'valid' padding only
    (`conv_layer.py`), so it needs 'same' padding (new geometry, im2col and crate ops in all three
    implementations) and a conv layer with a bias and no activation. That is a workplan of its own,
    as large as this one; it goes to next-steps.md under (a).
- **D3. The block's form: the identity after the add. Settled 2026-10-01: (a).**
  - (a) *Chosen.* `out = x + F(x)`, nothing after the add (He et al. 2016's pre-activation form,
    and the transformer's). The body ends in an affine layer (D4), so `F` can be negative and the
    sum isn't pushed one way.
  - (b) `out = relu(x + F(x))` (He et al. 2015, the original ResNet). Activations are fused into
    layers here, so the ReLU after the add needs an add-and-ReLU layer, and the identity path is no
    longer an identity for the gradient.
  - (c) A body ending in any hidden layer (sigmoid or ReLU), `x + relu(W x + b)`. No new layer
    class, but ReLU's output is never negative, so every block can only increase the activations.
- **D4. The affine layer at the end of a body: `Dense(n, activation="linear", bias=True)`.
  Settled 2026-10-01: (a).**
  - (a) *Chosen.* A `bias` field on `Dense`, default `False`. Today's `Dense(n, "linear")` is
    bias-free and comes before a `BatchNorm`, and stays so: saved files and validation don't change.
    `bias=True` is accepted only on a linear layer, and only as a body's last layer.
  - (b) A new activation literal, `"identity"` (affine), beside `"linear"` (bias-free).
  - In both, the affine layer is hidden only. It is also the output projection attention needs.
- **D5. Identity shortcuts only. Settled 2026-10-01: (a).**
  - (a) *Chosen.* The block's output size must equal its input size; validation refuses a
    mismatch. Every block in a transformer is like this.
  - (b) Projection shortcuts too (`x W_s + F(x)` when the sizes differ, He et al. 2015 option B):
    a trained shortcut layer, a second gradient sum, and its own parity tests. Next-steps under (a).
- **D6. No nested blocks. Settled 2026-10-01: (a).**
  - (a) *Chosen.* A body holds no `Residual`. Nothing on the roadmap needs nesting, and the
    refusal is easy to lift later: the design below nests without change.
  - (b) Allow nesting from the start, with tests for it.
- **D7. Initialization. Settled 2026-10-01: (a).**
  - (a) *Chosen.* Fan-in-aware for every weighted layer, as today, W then b in forward order. The
    fork and add layers draw nothing, so adding a block never shifts a later layer's draws (the pool
    layer's rule).
  - (b) The body's last layer starts at zero, so each block starts as the identity (Goyal et al.
    2017's zero-γ; Zhang et al. 2019, Fixup). Easy to add later as an option; next-steps under (a).
- **D8. Rust: a dense layer's hidden delta when the next layer is a fork.** Today a Rust dense
  layer's hidden delta is one fused call that reads `next_layer.W` and `next_layer.delta`
  (`rust_array_layer.py`, `relu_rust_array_layer.py`, `dropout_rust_array_layer.py`, the sigmoid
  branch of `batch_norm_rust_array_layer.py`). A fork has no `W`. **Settled 2026-10-01: (b).**
  - (a) Unfused ops that take a downstream array: `array_relu_mask` exists (batch norm added it),
    and the crate gains `array_sigmoid_mask(downstream, a)` and `array_dropout_mask(downstream,
    base_activation, mask, keep_probability, was_training)`. Before a fork, the layer calls
    `next_layer.downstream_batch()` and the mask op. One more crossing per block per step.
  - (b) *Chosen.* Fused ops with the skip term: `layer_hidden_delta_skip(W, delta, skip, a)`,
    `layer_relu_hidden_delta_skip` and `layer_dropout_hidden_delta_skip`, each single and batch
    (six ops), computing `(delta @ W + skip) * f'(a)` in one call. Before a fork, the layer reads
    the body's first layer's `W` and `delta` and the add's delta through the fork (`fork.body_first`,
    `fork.add`): a body's first layer always has a `W` and a delta (a dense, linear or affine
    layer). Every layer not before a fork keeps its fused call, so existing networks' arithmetic
    doesn't change.
- **D9. A depth study. Settled 2026-10-01: (a), with Stage 6's grid as proposed.**
  - (a) *Chosen.* A stage that trains plain and residual networks at increasing depth on MNIST
    and records the results in a script's docstring, as `batch_size_scaling.py` does. Its grid is
    Stage 6's.
  - (b) No study in this plan; next-steps.
- **D10. Sequential only, no preset class. Settled 2026-10-01: (a).**
  - (a) *Chosen.* Residual networks are built with `SequentialArrayNetwork` and the pure-Python
    `Sequential*` networks; no named class, so no new legacy loader or fixture class. The study
    builds its specs with a helper in its own script.
  - (b) A named preset per implementation.

## The design

Under the settled decisions.

### Specs

```python
@dataclass(frozen=True)
class Residual:
    """out = x + body(x): the body's layers, then the block's input added to their output."""

    body: tuple[LayerSpec, ...]
```

`LayerSpec` gains `Residual`, and `Dense` gains `bias: bool = False` (D4). A network over MNIST:

```python
[
    Dense(64, activation="relu"),
    Residual((Dense(128, activation="relu"), Dense(64, activation="linear", bias=True))),
    Residual((Dense(128, activation="linear"), BatchNorm("relu"), Dense(64, activation="linear", bias=True))),
    Dense(10, activation="softmax", output=True, loss="cross_entropy"),
]
```

`validate_layer_specs` accepts a block where a dense hidden layer may stand, and checks:

- the body isn't empty, holds no `Residual` (D6) and no conv or pool layer (D2), and is a valid
  hidden dense sequence by today's rules (linear and `BatchNorm` pairs, dropout only on sigmoid);
- the body's last layer is `Dense(n, "linear", bias=True)`, and `n` is the block's input size (D5);
- `bias=True` appears nowhere else.

A block counts as a dense layer when finding the front end (today it is "every layer before the
first `Dense`"), so a block after a conv front end is refused by D2's rule, not mistaken for front
end.

### Layers

`expand_specs(specs)` is the one place blocks are flattened. It turns each `Residual(body)` into
`Fork, *body, Add`, where `Fork` and `Add` are internal specs that no user writes. `spec_shapes`,
both builders and `batch_norm_index` / `refuse_single_example_groups` walk the expanded list;
`layer_specs` on a network keeps the specs as written, for format 2. Layer indices (snapshot
entries, optimizer state, checkpoints) are indices into the expanded list.

Three new layers per implementation, none with a `W` attribute, so the optimizer skips the fork
and add as it skips a pool layer, and their snapshot entries are `()`:

- **Fork** (parameter-free). Forward: keeps its input, `x` (and `X` for a batch), and returns it
  unchanged. Backward: `delta = body_first.downstream() + add.delta`. Downstream: its delta.
- **Add** (parameter-free, holds its fork). Forward: `y + fork.x`. Backward: `delta =
  next_layer.downstream()`. Downstream: its delta, the identity's.
- **Affine** (`W`, `b`). Forward `W x + b`; hidden delta `next_layer.downstream()`; downstream and
  gradient accumulation are the sigmoid layer's. numpy subclasses `ArrayLayer`, Rust
  `RustArrayLayer` (so `step_single`'s fused SGD applies), pure Python `BackpropLayer` with an
  identity node.

The network's forward and backward loops don't change: each layer still reads only the next
layer in the backward pass, and the fork reaches its add through the reference it is built with.
The backward order (reversed) computes the add's delta before the body's and the body's before the
fork's, which is what the fork needs.

### Arithmetic

- The add, `y + x`, and the fork's sum, `downstream + skip`, are one IEEE addition each, which is
  commutative, so their operand order can't move bits. No sum here has three terms.
- `delta = downstream * a * (1 - a)` before a fork, in numpy's order (`(downstream * a) * (1 - a)`),
  which the new crate op must match bit for bit.
- The README gains a "Residual connections" section with these expressions, as batch norm's has.

### Parity standard

The batch-norm workplan's: crate ops equal their numpy expressions by bits (op tests); whole
networks are compared after 50 steps against the measured no-residual control, and any gap is
explained (BLAS against the crate's FMA products, Adam's `beta**t`), never accepted as a tolerance.
Numpy-vs-X sigmoid comparisons use the `crate_exp` / `math_exp` fixtures.

## Pitfalls to design around

- **Fused hidden deltas read the next layer's weights.** Rust: every dense hidden delta (D8).
  Pure Python: `BackpropNode.compute_hidden_delta` sums `node.delta * node.input_node_weights[i]`
  over the next layer's nodes. Stage 3 changes it to take `next_layer.downstream_sum(i)`, which
  `BackpropLayer.downstream_sum` already computes in the same order, so the golden run must stay
  bit-identical.
- **`LayerMajorBatch` loads only the previous layer's lane** (`_select(index - 1, ...)`). An add
  reads its fork's input too, and a fork's backward reads its add's delta. A layer must declare the
  layers it reads, and the batch path loads all of their lanes. New pure-Python node classes list
  their per-example state in `example_fields`.
- **Single example and batch.** A fork keeps `x` and `X` apart; `classify_rows`' chunked
  `forward_batch` and `learn`'s single-example path both pass through it.
- **No in-place writes on the shared input.** The fork returns the same array it was given, which
  the network's `activations` list also holds. Nothing writes activations in place today (`+=` is
  used on gradients only); a test pins it.
- **Indices.** "layer i" in messages, `batch_norm_index`, snapshots and optimizer state must agree
  on the expanded index. A message names the spec path too (`layer 2, block body 1`).
- **The Rust optimizer's fused step** is taken for `isinstance(layer, RustArrayLayer)`; the affine
  layer must be one, and the fork and add must not be.
- **ab.py runs the new tree's script on both sides**, so a timing script added here must work on
  an old tree.

## Stages

One PR per stage; a crate stage is a crate PR, then a "Bump rust/" PR here. Every stage passes
`./cli lint`, `./cli test` and the golden check bit-identical.

### Stage 0: the plan and the README section

This workplan (D1-D10 settled), then the README's "Residual connections" section with the exact
forward and backward expressions. Docs only.

### Stage 1: specs

1. `Residual`, `Dense.bias`, `expand_specs`, the internal `Fork` / `Add` specs; `validate_layer_specs`
   and `spec_shapes` over the expanded list; `batch_norm_index` and `refuse_single_example_groups`
   find a `BatchNorm` inside a body.
2. Both builders and `format2.layer_to_json` refuse `Residual` and `bias=True` with "not yet (the
   residual-connections workplan, stage N)", as batch norm's stages did.
3. `tests/test_layer_specs.py`: accepted and refused cases (empty body, size mismatch, nesting, a
   block among conv layers, `bias=True` elsewhere, a body not ending affine, a `BatchNorm` pair
   inside a body).

No `learn*` path changes: no A/B.

### Stage 2: numpy

1. `AffineArrayLayer`, `ForkArrayLayer`, `AddArrayLayer`; the builder builds them.
2. `tests/test_residual_array_network.py`:
   - the gradient check (`tests/gradient_check.py`) on residual networks, single block and two in a
     row, with and without a `BatchNorm` pair, sigmoid and ReLU bodies, both shapes;
   - **identity blocks**: with the affine layer's `W` and `b` zero, a network with a block gives
     the same outputs and the same gradients for every other layer, by bits, as the network without
     it (`x + 0 = x`, and the fork's delta is `0 + skip`);
   - `learn` and `learn_batch` of one example agree, as the existing contract tests check.

numpy's dense layers already read `next_layer.downstream()`, so no existing numpy layer changes and
no `learn*` path of an existing network moves: no A/B; the PR says so.

### Stage 3: pure Python

1. `BackpropNode.compute_hidden_delta(downstream)`, with `BackpropLayer.compute_hidden_deltas`
   passing `next_layer.downstream_sum(i)` (bit-identical by construction; the golden run checks).
   Its own commit, so a golden failure points at it.
2. `AffineLayer`, `ForkLayer`, `AddLayer` with their nodes and `example_fields`; `LayerMajorBatch`
   loads every layer a layer reads.
3. Parity with numpy after 50 steps at the standard above; the gradient check and identity-block
   tests on the pure-Python networks.

Pure Python is parity-only: no A/B.

### Stage 4: Rust

1. **Crate PR.** `affine_forward(w, x, b)` and `affine_forward_batch`; the six skip ops (D8),
   `layer_hidden_delta_skip`, `layer_relu_hidden_delta_skip` and `layer_dropout_hidden_delta_skip`,
   single and batch, sharing the existing fused ops' downstream loop with the skip added before the
   activation's derivative, and tested as those ops are (by bits against the crate's own unfused
   `layer_downstream*` plus the add and mask). The type stub; `cargo fmt` / `clippy`.
2. **Bump rust/.** `AffineRustArrayLayer` (a `RustArrayLayer`), `ForkRustArrayLayer`,
   `AddRustArrayLayer` (the add through `Array.__add__`); the sigmoid, ReLU and dropout layers and
   batch norm's sigmoid branch call the skip ops when the next layer is a fork. The Rust fork
   computes its summed delta only when `downstream*()` is asked for it (by an add before it, or
   batch norm's ReLU branch, which already takes a downstream), so a fused predecessor pays no
   extra crossing.
3. Parity with numpy after 50 steps; the gradient check and identity-block tests on Rust.
4. **A/B**: the Rust dense hidden-delta methods change, so `ab.py run --bench
   prepared_dataset_timing`, both `.so` hashes differing in the header. Existing networks never meet
   a fork, so every row should be within noise; an added cost is the type check per layer per step.

### Stage 5: format 2, checkpoints and the golden run

1. `layer_to_json` writes `{"kind": "residual", "body": [...]}` and `{"bias": true}` only when set,
   so existing files don't change; `layer_from_json` reads them; `load_network` builds them.
2. Snapshots and optimizer state per expanded layer; `checkpoint()` / `restore_checkpoint()` and
   `run_checkpoint` round trips resume by bits.
3. One format-2 fixture per implementation (a Sequential residual network after two steps); the 14
   existing fixtures regenerate byte-identically.
4. The golden run gains residual entries (numpy, Rust, pure Python). Existing entries don't move;
   the PR says which entries were added.

### Stage 6: the depth study (D9)

The grid (settled with D9): MNIST, ReLU width 64, depths 2, 4, 8 and 16 hidden layers, plain
against residual (each block `Dense(64, relu), Dense(64, linear, bias=True)`), with and without
batch norm in the blocks; SGD with momentum 0.9 at one learning rate tuned on the shallowest plain
network; 3 seeds, 3 epochs; numpy. Recorded per cell: test accuracy, and the gradient's norm at the
first layer at initialization. Findings in the study script's docstring, as
`batch_size_scaling.py`'s. The expected result (He et al. 2016) is that plain networks get worse with
depth and residual ones don't; if not, the study says so.

### Stage 7: docs and retirement

README Models and the preset table mention `Residual`; the roadmap marks step 3 done. Only once
every stage and decision is done: the workplan is deleted and its leftovers move to next-steps.md.

## Out of scope

- Conv residual blocks and 'same' padding (D2).
- Projection shortcuts (D5), nesting (D6), zero-initialized block ends (D7).
- A standalone activation layer or post-add activations (D3).
- Layer norm and attention: roadmap steps 4 and 5.
