# Workplan: multi-layer convolutional networks (pure-Python implementation)

## Context

Convolution exists only in the pure-Python, one-object-per-node implementation:

- `ConvKernel` (`indrajala_ml/model/conv_kernel.py`) - one channel's shared weights + bias;
  already accepts `in_channels` (weights sized `k² · in_channels`).
- `ConvUnit` (`conv_unit.py`) - one output spatial position, ReLU, reads the shared kernel.
- `ConvLayer` (`conv_layer.py`) - input must be a plain `StateLayer`, single input channel,
  'valid' padding, stride supported, no stacking.
- `ConvMultiClassBackpropClassifierNetwork` - exactly one conv layer, then dense hidden layers,
  then a one-vs-rest output layer. Pinned UCI digits result: 0.9875 best training accuracy,
  0.925 test accuracy (`tests/test_conv_multiclass_backprop_model.py`).

The numpy (`Array…`) and Rust (`RustArray…`) implementations have no convolution at all.

## The core gap: backprop *through* a conv layer

`BackpropNetworkBase._backward_hidden_layers` calls
`node.compute_hidden_delta(next_layer.nodes, own_index)`, and `relu_hidden_delta` computes
`sum(node.delta * node.input_node_weights[own_index] for node in next_layer_nodes)`. That
assumes every downstream node is fully connected to every upstream node and owns a weight
indexed by upstream position. True when a dense layer follows the conv layer (why one conv layer
works today); false for conv -> conv: a `ConvUnit` has no `input_node_weights`, and an upstream
unit feeds only the downstream units whose receptive field covers it, each at a different kernel
weight index.

Fix: at construction, `ConvLayer` builds a reverse map `input index -> [(downstream unit,
kernel weight index)]`. The upstream delta's downstream sum is then
`Σ unit.delta · unit.kernel.weights[w]` over that list - the "full convolution with a flipped
kernel" written out as explicit indices. A dense O(N_down · N_up) scan would be ~29M operations
per example at MNIST scale, so the sparse map is required.

Hook: a layer-level `downstream_sum(own_index)` method - on `BackpropLayer` a pure extraction
of the existing dense sum, on `ConvLayer` the sparse reverse-map sum - used by the conv
network's own `_backward_hidden_layers` override. No change to `compute_hidden_delta`'s
signature across the sigmoid/ReLU/dropout node siblings.

## Measured cost (pure Python, one `learn()` call, before any of this work)

| Configuration | ms / example |
| --- | --- |
| 8×8 digits, 3×3×8 conv -> dense [32] | 6.9 |
| 28×28 MNIST, 3×3×8 conv -> dense [64] | 223 |

MNIST with one conv layer is ~3.7 h/epoch, so pure-Python multi-layer CNNs are realistically a
UCI-digits experiment. On 8×8 inputs two valid 3×3 convs go 8 -> 6 -> 4 and the first layer
already sees much of the image, so whether a second conv layer helps is an open question - a
null result is a real possibility and gets recorded honestly either way.

## Stage 1 - backprop through a conv layer

- `ConvLayer` builds the reverse map over its input layer's nodes and exposes
  `downstream_sum(own_index)`; `BackpropLayer.downstream_sum` is the extracted dense sum.
- `ConvUnit` gains a delta-from-downstream-sum path.
- The conv network overrides `_backward_hidden_layers` to use `downstream_sum`.
- Tests: `downstream_sum` matches a brute-force per-pair computation; a finite-difference
  gradient check whose loss passes through two stacked conv layers, checking every kernel
  weight and bias of both. The existing single-layer digits result must be unchanged.

## Stage 2 - multi-channel input to `ConvLayer`

- Accept any input layer whose nodes are channel-major, plus `in_channels`/height/width;
  receptive field is `in_channels × k × k` at flat index `c·H·W + r·W + col`, kernel weight
  order (channel, row, col) documented and tested.
- Tests: multi-channel single-hot-pixel wiring, hand-computed 2-channel forward example.

## Stage 3 - network takes a list of conv layer specs

- A `ConvSpec(kernel_size, channel_count, stride)` dataclass; the constructor takes
  `conv_specs: list[ConvSpec]`, chaining each layer's output height/width/channels into the
  next.
- `randomize()`: per-conv-layer kernel fan-in (`k² · in_ch`); dense tail starts from the last
  conv layer's node count.
- Save/load: JSON envelope gains a `conv_layers` list (no committed conv model file exists, so
  no compatibility cost).
- Generalize the existing class rather than add a sibling: a one-spec network must reproduce
  the pinned 0.9875 / 0.925 digits result exactly (same RNG draw order), a stronger regression
  proof than a new class would get.

## Stage 4 - measured comparison: 1 vs 2 conv layers on UCI digits

- Multi-seed comparison of one vs two conv layers (matched budget, same dense tail) on the
  bundled UCI digits split; record mean/spread of training and test accuracy. No wall time: the
  pure-Python implementation is never used for timing (README, Models section).
- Result recorded as-is, including a null or negative result.

Status: the comparison exists as `demo_conv_depth_uci_digits_comparison.py` (dense baseline,
conv1 3x3x8, conv2 3x3x8 -> 3x3x8, and a parameter-matched conv2-wide 3x3x8 -> 3x3x16; 8
paired seeds, 20 epochs). **The measurement was not run to completion**: the first full sweep
was stopped after ~46 minutes on 8 logical cores (4 physical) with no results, and the
comparison was deliberately left unmeasured rather than rerun. Open question as of this
workplan: whether a second conv layer helps on 8x8 digits.

## Stage 5 - `MaxPoolLayer`

- Weight-free layer: forward with a cached argmax per window, backward routes each delta only
  to its argmax input (via the same `downstream_sum` hook), no-op gradient hooks,
  `snapshot_state() -> []`. Sits in `trainable_layers`, since both forward and snapshot iterate
  over it.
- Tests: hand-computed forward/argmax, gradient check through conv -> pool -> conv, then the
  same measured digits comparison as stage 4 with pooling in place of / in addition to stride.

Status: `MaxPoolLayer`/`PoolSpec` (`max_pool_layer.py`) are in, and `conv_specs` may mix
`ConvSpec` and `PoolSpec`. Correctness is covered by layer-level and network-level
finite-difference gradient checks through conv -> pool -> conv. The stage-4 demo gains
`conv1-pool` (3x3 conv then 2x2 max pool) and `conv1-stride2` (3x3 conv at stride 2), both ending
at 3x3x8 with identical parameter counts, isolating pooling vs. strided downsampling. **Not run**,
for the same reason as stage 4 - an open question alongside it.

## Out of scope

A numpy `ConvArrayLayer` (im2col/col2im) and its Rust counterpart - the route to MNIST scale.
It needs the hard-coded `next_layer.W.T @ next_layer.delta` (18 sites across 7 array/Rust-array
layer files) refactored into a per-layer `backward_to_input(delta)` first, plus im2col/col2im
kernels in the `indrajala-math-rust` submodule. A separate project, justified only by a concrete
need for MNIST-scale CNNs.
