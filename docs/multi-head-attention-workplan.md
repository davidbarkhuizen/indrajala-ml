# Workplan: multi-head attention

**Status: decisions D1-D10 settled (2026-10-09). Stage 0 done: this plan (#602) and the README's
attention arithmetic in its three blocks (#603). Stage 1 done: the attention benchmark case (#604).
Stage 2 done: the spec (#605). Stage 3 done: numpy in three blocks, the golden run
bit-identical (#606). Stage 4 done: pure Python in the same blocks (#607). Stage 5 done: the crate
in blocks with a `heads` argument (indrajala-math-rust #51) and the Rust layer passing it, the
golden run bit-identical. Stage 6 done: two-head patch-model fixtures, multi-head checkpoint round
trips and a multi-head golden entry per implementation (#612). Stages 7-8 not started.**

Roadmap step 5 ([primitives-roadmap.md](primitives-roadmap.md)): multi-head attention and a full
transformer block. Step 4 (the layer-norm and attention workplan, retired:
`git show 7d3bd0a:docs/layer-norm-attention-workplan.md`) built token sequences, layer norm, a
single-head `Attention()` with keys as wide as the tokens, learned positions, token-wise dense
layers and residual blocks over tokens, in all three implementations, with format-2 files, golden
entries and a study. This plan gives `Attention` heads and a key size, restructures the attention
layer in all three implementations and the crate into building blocks with named extension points
for the work that follows (masking, dropout, cross-attention, key/value head counts), and trains
deeper multi-head patch models against step 4's numbers.

**The priority is a well-structured, extensible framework** (CLAUDE.md, Structure first). Where
keeping step 4's bits would need a special case, a second code path or a design that blocks the
extension points below, this plan takes the better structure and re-records the golden run, under
measurement.md §8's structure clause: the stage's PR names the entries that moved and the
structural gain, and the new bits pass the parity standard.

## Why

- **Heads are the transformer's attention** (Vaswani et al. 2017, "Attention Is All You Need",
  arXiv 1706.03762, §3.2.2): `h` heads each attend in their own `d_k`-wide subspace, and their
  outputs are concatenated before the output projection, so one layer can attend to several
  tokens for several reasons at once. A single head averages them into one distribution per token.
- **Step 4's attention block was weak on its own** (92.4%, under the dense control's 93.9%), and
  attention then FFN beat FFN alone by 0.4 points, less than a standard deviation over 3 seeds
  (`scripts/patch_attention_study.py`'s docstring). More heads and a second layer are the two
  levers the literature points to; the study (D8) measures both against those numbers.
- **Step 4's attention is monolithic.** Each crate op is one long function (`rust/src/attention.rs`),
  and the numpy and pure-Python layers do the projections, the softmax and the head-free core in
  one method. Masking, dropout on the weights, cross-attention and grouped key/value heads would
  each be another rewrite of those functions. Splitting them now, while adding heads, makes each
  later feature an addition at a named place.
- **Key size apart from the head count.** With `d_k = d / h`, the per-head score matrix `Q_h K_h^T`
  has rank at most `d / h`, a bottleneck Bhojanapalli et al. 2020 ("Low-Rank Bottleneck in
  Multi-head Attention Models", arXiv 2002.07028) show costs accuracy as `h` grows at fixed `d`.
  A `key_size` that defaults to `d / h` but can be set (D3) lets the study separate "more heads"
  from "narrower heads".

## What exists, and what changes

| piece | today (step 4) | this plan |
| --- | --- | --- |
| spec | `Attention()` | `Attention(heads=1, key_size=None)` (D2, D3) |
| parameters | `Wq, Wk, Wv, Wo` each `(d, d)`; `bq, bk, bv, bo` each `(d,)` | `Wq, Wk, Wv` `(h·d_k, d)`, `bq, bk, bv` `(h·d_k,)`, `Wo` `(d, h·d_k)`, `bo` `(d,)`: the same eight arrays, heads as row blocks (D2) |
| scale | `sqrt(d)` | `sqrt(d_k)` (equal at one head) |
| structure | one method or function per pass | three building blocks per pass in every implementation: project, attend (per head, with options), combine (D4) |
| Rust | 4 long fused ops | the same 4 exported ops, each a thin wrapper over crate-internal blocks and an `AttentionOptions` struct (D4) |
| format 2 | `{"kind": "attention"}` | `heads` and `key_size` written only when not the default |
| validation | a token block's body ends in `Attention()` | unchanged; `heads >= 1`, `key_size >= 1`, `heads` divides `d` when `key_size` is `None` |
| draws | `Wq, bq, Wk, bk, Wv, bv, Wo, bo`, fan-in `d` | the same order; `Wo`'s fan-in is `h·d_k` (equal to `d` at the defaults) |

`spec_shapes` doesn't change: attention maps `(T, d)` to `(T, d)` whatever its heads.

## Decisions (settled 2026-10-09)

The owner set the criterion, a well-structured, extensible framework, and the golden-run clause
that goes with it (measurement.md §8), and delegated the choices under it. Each lists the options
considered and the choice.

- **D1. Scope. Settled: (a).**
  - (a) *Chosen.* Heads and a key size (D2, D3) in all three implementations; the attention layer
    restructured into building blocks with extension points (D4); the study over heads and depth
    (D8); the docs. Dropout in attention (D6) and masking (D7) are not built, but their places in
    the structure are named (Extension points), so their workplans add, not rewrite.
  - (b) (a) plus dropout in attention. (c) (a) plus a causal mask and a sequence task.
    Both rejected under D6 and D7.
- **D2. The parameters' layout. Settled: (a).**
  - (a) *Chosen.* Packed: each of `Wq, Wk, Wv` is one `(h·d_k, d)` matrix whose rows
    `[i·d_k, (i+1)·d_k)` are head `i`'s, its bias `(h·d_k,)` likewise; `Wo` is `(d, h·d_k)`, its
    columns in the same head blocks. Pros: a fixed list of eight named arrays, which is the
    framework's parameter model (`AttentionProjections`, optimizer state, snapshots, checkpoints,
    format 2's names don't change); `Wq`, `Wk` and `Wv` stay separate, so each can later take its own
    width (grouped-query attention: `Wk`, `Wv` with fewer row blocks) or its own input
    (cross-attention); at one head the shapes and draw order are step 4's. PyTorch's layout. Cons:
    a head's parameters are slices; per-head inspection slices.
  - (b) Per head: `Wq_0, ..., Wq_{h-1}` and so on, `3h + 1` weights. Pros: each head's arrays its
    own. Cons: the parameter list's length depends on `h`, which breaks the fixed-list model every
    consumer relies on; `3h` products instead of 3.
  - (c) One fused `Wqkv` `(3·h·d_k, d)` (timm's `qkv`). Pros: one product. Cons: forces one width
    and one input for Q, K and V, which blocks cross-attention and grouped key/value heads; rejected
    for structure, not for its golden cost.
- **D3. Key size. Settled: (a).**
  - (a) *Chosen.* `Attention(heads=1, key_size=None)`: `key_size` is each head's width `d_k`
    (Keras `MultiHeadAttention`'s `key_dim`; not PyTorch's `kdim`, which is the width of the key
    *input* for cross-attention); `None` means `d / heads`, which must then divide. Values are as
    wide as keys (`d_v = d_k`). Pros: the literature's meaning; at the default the parameter count
    doesn't depend on `h` (`4d² + 4d`), so a heads sweep is fair; a set `key_size` tests the
    bottleneck; a later `value_size=None` ("as `key_size`") or `key_value_heads=None` ("as
    `heads`") adds without breaking anything, format 2 omitting defaults. Cons: a set `key_size`
    makes `h·d_k` differ from `d`, a new place for width mix-ups (tests cover it).
  - (b) `key_size` as the total width `h·d_k`. Cons: neither library's meaning; makes fewer
    key/value heads awkward to express later.
  - (c) (a) plus `value_size` now. Cons: nothing uses it; it can be added later as above.
- **D4. Structure: building blocks in every implementation, thin exported crate ops.
  Settled: (a).**
  - (a) *Chosen.* Each implementation's attention layer, and the crate, split into the same three
    blocks per pass:
    1. **project**: `Q`, `K`, `V` from their input rows, one affine product each (today's dense
       ops on the `(N·T, d)` rows);
    2. **attend**: per example and head, scores, the softmax and the weighted sum, and in the
       backward pass their gradients, taking an options value (today only the head count and
       `d_k`; Extension points);
    3. **combine**: the heads side by side, then the output projection; backward, its gradient
       and `dH`.

    On Rust the four exported ops (`attention_forward`, `attention_forward_batch`,
    `attention_downstream_batch`, `attention_accumulate_gradient_batch`) keep one crossing per pass
    (step 4's D9) but become thin wrappers that validate shapes, build an `AttentionOptions` and
    call crate-internal `project`, `attend_forward`, `attend_backward` and `combine` functions;
    `heads` is a new argument. Each block is tested on its own against numpy's expression by bits
    where no product is involved, and against the crate's own composition where one is. numpy and
    pure Python get the same three methods, so the implementations share seams as well as
    expressions. Pros: masking, dropout on `P`, cross-attention and grouped heads each change one
    block; smaller tested units; the three implementations read alike. Cons: the largest stages
    (3 to 5) grow; a signature change to the crate's ops, landed in step with the Rust layer.
  - (b) `heads` added to the four ops as they are. Smaller now, but every later feature rewrites
    long functions.
  - (c) New multi-head ops beside the untouched single-head ones. Two copies of one arithmetic, and
    a branch on `h` in the Rust layer: the special case the structure clause exists to avoid.
  - (d) General primitives exposed and composed in Python. Rejected in step 4 (D9 (b)): 6-10
    crossings per pass, times `h`, and Python owning every sum's order.
- **D5. A full transformer block. Settled: (a).**
  - (a) *Chosen.* No new spec: a transformer layer is the two pre-LN blocks step 4 already
    composes, `Residual((LayerNorm(), Attention(heads=h)))` then
    `Residual((LayerNorm(), Dense(f, "relu"), Dense(d, "linear", bias=True)))`, stacked by writing
    them again. ReLU in the FFN (Vaswani et al.'s). The study builds them with a helper in its
    script. Pros: the framework is built on composable layers (roadmap step 1); a macro spec
    would freeze choices that should vary independently (pre-LN against post-LN, the activation,
    where dropout goes). Cons: long spec lists for deep models, which a helper hides.
  - (b) A `TransformerLayer(heads, ffn_size)` spec flattened by `expand_specs`. A second spelling
    of a composition, with its own format-2 kind and validation, freezing the block's shape.
  - (c) (a) with GELU in the FFN (ViT's). GELU is an activation for `Dense`, not part of attention:
    it needs `erf` (not in stable Rust; the `tanh` form is a different function) in all three
    implementations and the crate. Its own workplan; next-steps.
- **D6. Dropout in attention. Settled: (a).**
  - (a) *Chosen.* Not built; its place named (Extension points: a dropout mask on `P` is an
    attend option). Dropout today is fused with the sigmoid and refused among tokens; building it
    means a mask draw from the network's generator in all three implementations (the pure-Python
    networks draw in another order, rng-audit.md) and a training/inference switch in a token layer,
    a workplan's worth on its own. At 5 epochs on MNIST step 4's patch models were still improving,
    not overfitting, so the study doesn't need it.
  - (b) Dropout on `P` (Vaswani et al.'s and timm's `attn_drop`). (c) Dropout after the output
    projection (`proj_drop`), a token-wise dropout the FFN could share. Each its own later plan.
- **D7. Masking. Settled: (a).**
  - (a) *Chosen.* Not built; its place named (Extension points: an additive mask on the scores
    before the max shift is an attend option). A causal mask is useful only with a per-token loss
    (a language model), which needs a per-token output layer and a sequence dataset, none of which
    exist; a padding mask needs sequences of different lengths. Its own workplan, with the
    sequence task.
  - (b) A causal flag now, `Attention(causal=True)`. Cheap and exactly testable (`exp(-inf)` is
    exactly 0), but nothing would use it, and its first real use should decide its form.
- **D8. The study. Settled: (a).**
  - (a) *Chosen.* `scripts/multi_head_attention_study.py`, numpy, MNIST, step 4's protocol (patch
    7, `d = 32`, `T = 16`, Adam at batch 32, a rate per arm from a short `tune`, 5 epochs,
    `OPENBLAS_NUM_THREADS=1`), with 5 seeds instead of 3 to resolve margins of step 4's size.
    Arms, all attention-then-FFN (FFN width 64):
    1. `h = 1`, one layer: step 4's `attention-ffn`, rerun as the anchor;
    2. `h = 2`, one layer (`d_k = 16`);
    3. `h = 4`, one layer (`d_k = 8`);
    4. `h = 4, d_k = 32`, one layer: the bottleneck test (D3), 16,800 attention parameters
       against the defaults' 4,224;
    5. `h = 1`, two layers;
    6. `h = 4`, two layers.

    And step 4's `ffn` arm rerun at 5 seeds, the no-attention control. The conv and dense
    controls are cited from step 4, not rerun: their accuracies don't depend on the machine (the
    golden run is byte-identical on both), and seconds per epoch compare only within one study.
    Recorded per cell as step 4's. One epoch of the slowest arm (6) is timed first and the grid
    sized from it; the PR says so. What's expected: arms 2-3 at or above arm 1 at equal parameters
    (Vaswani et al.'s Table 3, rows (A)), arm 4 at or above arm 3 (Bhojanapalli et al.), and depth
    helping more than heads at this size. If not, the study says so.
  - (b) (a) plus `d = 64` arms; (c) (a) plus patch size 4 (`T = 49`). Both to next-steps: they
    change what step 4's numbers control for.
- **D9. Pure Python. Settled: (a).**
  - (a) *Chosen.* All three implementations, with the same three blocks (D4). Pure Python is the
    reference the others are checked against, and the three-implementation invariant is the
    framework's.
  - (b) numpy and Rust only: breaks that invariant for the first time.
- **D10. Timing. Settled: (a).**
  - (a) *Chosen.* No benchmark times attention today, so stage 1 adds a one-head attention case to
    `focused_benchmark.py` (forward and backward of a batch of patch tokens, numpy and Rust),
    written against `Attention()` so it runs on the old tree too (`ab.py` runs the new tree's
    script on both sides). Stages 3 and 5 A/B it at one head, tier 1 (`--order ONNO`). The
    restructure is chosen for structure, not speed; a slower row is reported with its cause
    (an extra copy per block, say), and fixed if cheap, else listed in next-steps' crate tuning.
  - (b) No attention benchmark: the restructure's cost would go unmeasured.

## Extension points

Where the work this plan leaves plugs in, so its workplans add rather than restructure:

| later work | where it goes | what it adds |
| --- | --- | --- |
| masking, causal or padding (D7) | attend, forward: an additive mask on `S[i]` before the max shift | a mask field in the options; backward unchanged (`P_ij = 0` zeroes `dS_ij`) |
| dropout on the weights (D6 (b)) | attend: an inverted mask on `P[i]` before `H[i] = P[i] V[i]`, kept for backward | a mask field in the options, drawn by the layer from the network's generator |
| dropout after the projection (D6 (c)) | after combine | a token-wise dropout layer, outside attention |
| a separate value width | project and combine: `Wv` `(h·d_v, d)`, `Wo` `(d, h·d_v)` | a `value_size` field defaulting to `key_size` |
| grouped- or multi-query attention | project: `Wk`, `Wv` with `g` row blocks; attend: head `i` reads key/value block `i // (h / g)` | a `key_value_heads` field defaulting to `heads` |
| cross-attention | project: `K`, `V` from a second input | a second input to the layer, which `Sequential` doesn't have: its own design |

None of the fields is added in this plan; each is added, with its default, by the work that uses it.

## The design

### Spec

```python
@dataclass(frozen=True)
class Attention:
    """
    Self-attention over the tokens in heads heads of key_size features each (d / heads when None,
    which heads must then divide), values as wide as keys, no mask, biases on all four
    projections, ending in the affine output projection. It ends a token block's body.
    """

    heads: int = 1
    key_size: int | None = None
```

`validate_layer_specs` refuses `heads < 1` and `key_size < 1`; `spec_shapes`, where `d` is known,
refuses `heads` not dividing `d` when `key_size` is `None`, with the spec and the token width in the
message. `Attention.head_size(d)` resolves `d_k` for the builders. Format 2 writes
`heads` and `key_size` only when not the default, so existing files load unchanged;
`layer_from_json` passes them through `_TOKEN_SPECS` as today.

### Arithmetic

Per example, `X` the `(T, d)` tokens, `h` heads of `d_k` features, `s = sqrt(d_k)`, `[i]` head
`i`'s block (rows `i·d_k` to `(i+1)·d_k - 1` of `Wq`, `Wk`, `Wv` and their biases, the same columns
of `Q`, `K`, `V`, `H` and of `Wo`). Every sum a left fold in index order unless a matrix product,
as step 4's:

```text
project
  Q = X Wq^T + bq;  K = X Wk^T + bk;  V = X Wv^T + bv     (T, h·d_k) each: the product, then the bias
attend, each head i
  S[i] = (Q[i] K[i]^T) / s                               (T, T)
  m_t  = max_u(S[i]_tu)
  e_tu = exp(S[i]_tu - m_t)
  P[i]_tu = e_tu / sum_u(e_tu)
  H[i] = P[i] V[i]                                       (T, d_k)
combine
  H = [H[0] ... H[h-1]]                                  (T, h·d_k): the heads side by side
  out = H Wo^T + bo                                      one product over all h·d_k columns
backward
  combine:  dH = delta Wo                                (T, h·d_k)
            grad_Wo += delta^T H;  grad_bo += sum(delta) over the batch's rows
  attend, each head i
            dP[i] = dH[i] V[i]^T
            dV[i] = P[i]^T dH[i]
            r_t   = sum_u(dP[i]_tu * P[i]_tu)
            dS[i]_tu = P[i]_tu * (dP[i]_tu - r_t)
            dQ[i] = (dS[i] K[i]) / s
            dK[i] = (dS[i]^T Q[i]) / s
  project:  dX = (dQ Wq + dK Wk) + dV Wv                 three products over all heads, in this order
            grad_Wq += dQ^T X;  grad_bq += sum(dQ)       and Wk, bk from dK; Wv, bv from dV
```

At one head with `d_k = d` every expression is step 4's. The heads meet only in the products that
span them, `H Wo^T` and `dQ Wq` (and its two siblings): each one product over all `h·d_k` columns,
never a sum of per-head products. The README's "Layer norm and attention" section is rewritten to
this form, in these three blocks, in stage 0.

What stays true per head: `bk[i]` is inert (it adds `q_t · bk[i]` to row `t` of `S[i]`), so its
gradient is rounding noise and parity tests compare it apart as today; `bv` still duplicates `bo`
(through `Wo bv`).

### Layers

- **numpy** (`attention_array_layer.py`): `_project`, `_attend`, `_combine` and their backward
  counterparts. `_project` is today's products on the `(N·T, d)` rows, giving `(N·T, h·d_k)`;
  `_attend` works on the `(N, h, T, d_k)` view (`reshape(N, T, h, d_k).transpose(0, 2, 1, 3)`), its
  products `np.matmul` over the `(N, h)` leading axes, the softmax and `r` through `np.cumsum` along
  the last axis; `_combine` transposes back, reshapes (a copy when `h > 1`) and projects. One code
  path for every `h`.
- **pure Python** (`attention_layer.py`): the same three methods; `_attend` loops over heads with
  the same left folds; the `example_fields` caches hold the packed `(T, h·d_k)` matrices and `P` as
  one `(T, h·T)`, as the crate's.
- **Rust** (`attention_rust_array_layer.py`): passes `heads` to each crate op; nothing else changes.
- **crate** (`rust/src/attention.rs`, or an `attention/` module if it outgrows one file):
  `AttentionOptions { heads, key_size }`, `project`, `attend_forward`, `attend_backward`,
  `combine` and `combine_backward`, crate-internal and each tested; the four `#[pyfunction]`s
  validate, build the options and compose them. `p` is cached as `(N·T, h·T)`.
- **`AttentionProjections`**: `projection_shapes` returns `(h·d_k, d)` three times and
  `(d, h·d_k)`; `features` stays `d`; new attributes `heads` and `key_size`. The builders pass the
  spec's `heads` and resolved `key_size` to every implementation's constructor.

### Parity standard

Step 4's: crate ops equal their numpy expressions by bits where no product is involved and the
crate's own composition where one is; whole networks compared after 50 steps (step by step under
`Adam`, as step 4 left it in next-steps); any gap explained, never accepted as a tolerance.

Exact tests that need no tolerance, each in every implementation:

- **one token**: `P[i] = [[1]]` for every head, so attention is `(X Wv^T + bv) Wo^T + bo` by bits,
  at any `h`;
- **uniform attention**: with `Wq = Wk = 0` and `bq = bk = 0`, every head's weights are exactly
  `1/16` at `T = 16`;
- **identical heads**: with each head's blocks of `Wq`, `Wk`, `Wv` and their biases equal, every
  head's `P[i]` is equal by bits, and so is every `H[i]`;
- **a silent head**: with head `i`'s columns of `Wo` zero, `dH[i]` is exactly zero, so head `i`'s
  blocks of `grad_Wq`, `grad_Wk`, `grad_Wv` and their biases are exactly zero;
- **identity blocks** as today: with `Wo` and `bo` zero the attention block leaves every other
  layer unchanged;
- **blocks compose**: each implementation's `project`, `attend`, `combine` called in turn equal the
  layer's pass by bits (the layer adds nothing between them).

And the gradient check (`tests/gradient_check.py`) at `h ∈ {1, 2, 4}`, at a `key_size` other than
`d / h`, single example and batch.

**Against step 4.** At one head, each stage compares the restructured layer with step 4's outputs
on the golden run's attention entries. Equal bits are expected (the expressions and their grouping
don't change), but not required: if a stage's single code path or block split moves them, the PR
re-records the golden run under §8's structure clause, listing the entries that moved, the
measured difference, and which structural gain (one path for every `h`, separable blocks) moved
them. Parity between the three implementations must hold either way.

## Pitfalls to design around

- **`d_k` isn't `d`.** Every place that reads `features` as the projections' width (the crate's
  shape checks, `examples_and_tokens`' divisor, `reset_gradient_accum`, `projection_shapes`, the
  scale) must read `d` or `h·d_k` deliberately. Tests at `key_size ≠ d / h` (and `h·d_k ≠ d`) catch
  a mix-up that the defaults hide.
- **numpy's 4D stacks.** `np.matmul` on the `(N, h, T, d_k)` view may take another BLAS path than
  step 4's 3D stack. One path for every `h` is the design; if it moves one-head bits, that's the
  structure clause's case above, not a reason for a branch.
- **`Wo`'s fan-in** is `h·d_k`, so its draw limit is `1/sqrt(h·d_k)`, equal to today's at the
  default. A study arm with `d_k = 32, h = 4` draws `Wo` narrower than `Wq`.
- **The crate's `p` cache** becomes `(N·T, h·T)`; the shape checks and the one-example path
  (`attention_forward`, 1D) both change. The batch-of-one-equals-one-example test covers both.
- **Pure-Python `LayerMajorBatch` lanes** hold the caches as lists; the blocks must assign new
  lists, never write into a cached one (step 4's rule, `attention_layer.py`'s docstring).
- **Scale per head**: `sqrt(d_k)` is inexact for `d_k = 8`; it is computed once (`math.sqrt` /
  `f64::sqrt`, both correctly rounded) and divided by, never multiplied by its reciprocal.
- **Format 2 forward compatibility**: an older checkout refuses a file with `heads`; files written
  at the defaults stay loadable by older code. Nothing promises more.
- **`ab.py` runs the new tree's script on both sides** (D10): the attention benchmark case uses
  only `Attention()`.

## Stages

One PR per stage; a crate stage is a crate PR, then a "Bump rust/" PR here. Every stage passes
`./cli lint`, `./cli test`, and, where it could reach training results, the golden check
bit-identical or re-recorded under §8 as above. The golden-run structure clause (CLAUDE.md,
measurement.md §8) landed first, as #601.

### Stage 0: the plan and the README

This workplan, then the README's "Layer norm and attention" section rewritten to the per-head
expressions in their three blocks (describing one head as what exists). Docs only, tier 0.

### Stage 1: an attention benchmark case (D10)

`focused_benchmark.py` gains an attention case: a batch of 32 patch-token examples (`T = 16,
d = 32`) through `forward_batch`, the hidden delta and `accumulate_gradient_batch` of a one-head
`Attention()` layer, numpy and Rust. New measurement, no timed path changed: tier 0. One `ab.py`
A/A on `jebel` checks the case's noise is within the profile's rules before stages 3 and 5 rely on
it.

### Stage 2: the spec

1. `Attention(heads=1, key_size=None)`; validation and its refusals; `layer_to_json` omits the
   defaults.
2. Both builders refuse `heads > 1` or a `key_size` with "not yet (the multi-head attention
   workplan, stage N)", as earlier stages did.
3. `tests/model/specs/`: accepted and refused cases (`heads=0`, `heads=3` at `d = 32`,
   `key_size=0`, `heads=3, key_size=8` accepted); format 2 round trips at and off the defaults;
   every existing fixture re-saves byte-identically.

No layer changes: tier 0.

### Stage 3: numpy

1. `AttentionArrayLayer(tokens, features, heads, key_size)` in its three blocks;
   `projection_shapes`; the builder passes them.
2. The golden run: bit-identical, or re-recorded under §8's structure clause (Parity standard,
   Against step 4).
3. `tests/model/networks/test_attention_array_network.py`: the gradient check at `h ∈ {1, 2, 4}`
   and at `key_size ≠ d / h`; the exact tests above, the blocks-compose test included.
4. A/B (D10): the stage-1 case, numpy rows, tier 1, `--order ONNO`.

### Stage 4: pure Python

1. The three blocks in `attention_layer.py`; the builder passes `heads` and `key_size`.
2. Parity with numpy after 50 steps under `SGD`, `Momentum` and `WeightDecay`, step by step under
   `Adam`, at `h = 1`, `2` and `4`; the gradient check and exact tests.

Pure Python is never timed: tier 0.

### Stage 5: Rust

1. **Crate PR** (`indrajala-math-rust`): `AttentionOptions` and the internal blocks (D4); the four
   ops as wrappers with `heads`; `p` as `(N·T, h·T)`. Each block tested by bits against numpy's
   expression where no product is involved and the crate's composition where one is, at shapes
   reaching `h ∈ {1, 2, 4}`, `d_k ≠ d / h`, one example and a batch; at one head, against step 4's
   ops' outputs recorded before the change. The type stub; `cargo fmt` / `clippy`.
2. **Bump rust/.** The Rust layer passes `heads`; parity with numpy, the gradient check and the
   exact tests on Rust; the golden run as in stage 3.
3. **A/B** (D10): the stage-1 case, Rust rows, tier 1, `--order ONNO`, the header's two `.so`
   hashes differing.

### Stage 6: format 2, checkpoints and the golden run

1. One format-2 fixture per implementation of a two-head patch model after two steps; the
   existing fixtures regenerate byte-identically (or as re-recorded in stages 3-5, with the reason).
2. Checkpoint round trips of a multi-head model resume by bits.
3. The golden run gains a multi-head entry per implementation (`h = 4`, one layer, under `Adam`).
   A new golden version, archived (`golden_training_run.py archive`, measurement.md §10), its
   reason `new-functionality` if no entry moved in stages 3-5, else `material`.

### Stage 7: the study (D8)

`scripts/multi_head_attention_study.py` as D8 settles it, reusing `patch_attention_study.py`'s
protocol helpers rather than copying them (moved to a shared module if both scripts need them;
next-steps, From the DRY audit). Findings in the script's docstring; raw outputs under
`data/attention/`, untracked. Run on `jebel` with nothing else running, since seconds per epoch are
recorded.

### Stage 8: docs and retirement

The README's Models and "Layer norm and attention" mention heads, key size, the three blocks and
the study's findings; the roadmap marks step 5 done. Only once every stage and decision is done:
this workplan is deleted, listed under next-steps' retired workplans, and its leftovers move to
next-steps.md, replacing that file's "From layer norm and attention" paragraph on step 5: the
Extension points table (dropout, masking, value width, grouped heads, cross-attention), GELU
(D5 (c)), and the study's `d = 64` and patch-4 arms (D8 (b), (c)).

## Out of scope

- Everything in the Extension points table: named, not built.
- GELU (D5 (c)), a `TransformerLayer` spec (D5 (b)), a class token (step 4's D8 (b)), relative
  position biases.
- Head pruning or per-head analysis tools (Michel et al. 2019, "Are Sixteen Heads Really Better
  than One?", arXiv 1905.10650), beyond what the study records.
- Speed beyond D10: a faster multi-head kernel is a crate-tuning item for next-steps, with its own
  A/B.
