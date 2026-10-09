# Roadmap: the next ML primitives

**Status: a proposed order, not a workplan. Each step gets its own workplan before any code.**

Which primitive to add next, and in what order: normalization, residual connections, attention,
transformers.

## What exists

Dense backprop, the perceptron and MADALINE, ReLU, softmax with cross-entropy, conv and max
pooling, momentum, Adam, L2, dropout, ensembles, linear warmup, batch norm with ghost groups,
residual blocks, layer norm and single-head self-attention over patch tokens (README, Models,
Batch normalization, Residual connections and Layer norm and attention), built from composable
layer specs and update rules. Each is in all three implementations (pure Python, numpy, Rust).
There is no multi-head attention, no masking and no recurrence.

## The order

1. **Composable layers and optimizers** (a refactor, before the next primitive): done
   (2026-09-30); its open items are in [next-steps.md](next-steps.md).
2. **Batch normalization**: done (2026-09-30), and the conv batch-size study rerun with it
   (2026-10-01).
3. **Residual connections**: done (2026-10-01), dense blocks with identity shortcuts.
4. **Layer normalization and single-head self-attention**: done (2026-10-02), as a patch model
   on MNIST.
5. **Multi-head attention and a full transformer block.**

## 1 to 4. Done

Composable layers and optimizers, then batch normalization, both done (2026-09-30); their retired
workplans and the work they left open are in [next-steps.md](next-steps.md). The case made for
each is in this file's history: `git show 058087a:docs/primitives-roadmap.md`.

The conv batch-size study was rerun with batch norm, plain and with ghost groups of 32 (Goyal et
al. 2017's full setup), on 2026-10-01. Batch norm adds about a point of accuracy, but the linear
rule still fails at B = 512 at both momenta (`batch_size_scaling.py`'s docstring; open questions
in next-steps.md).

Residual connections, dense blocks `out = x + F(x)` with identity shortcuts, were done on
2026-10-01; the retired workplan and its open work (conv blocks, projection shortcuts) are in
next-steps.md. The depth study (`scripts/residual_depth_study.py`) found what He et al. 2016
predict: plain networks get worse with depth and residual ones don't.

Layer norm (over tokens and in flat dense networks) and single-head self-attention were done on
2026-10-02, as a small vision transformer on MNIST: `Patches`, a token-wise `Dense` embedding,
learned `Position`s, pre-LN attention and FFN blocks, and `TokenMean`. The retired workplan and its
open work are in next-steps.md. The patch-attention study (`scripts/patch_attention_study.py`)
found the expected result, weakly: attention then FFN beats FFN alone by less than a standard
deviation, and the conv network beats every patch model at this data size. Step 5 is next: its
workplan is [multi-head-attention-workplan.md](multi-head-attention-workplan.md).

## 5. Multi-head attention and a full transformer block

Step 4 built the pieces a transformer layer composes: token sequences, layer norm, single-head
attention, learned positions, token-wise dense layers and residual blocks over tokens. The specs
already stack attention and FFN blocks into a deeper encoder; no study has trained one. What step
4 left for this step (its workplan's "Out of scope"):

- **Multi-head attention**: `h` heads of `d / h` features each, concatenated before the output
  projection. The crate's attention ops are single-head, one fused call per pass (step 4's D9);
  heads extend or rewrite them.
- **A key size other than the token size** (`Attention(key_size=...)`).
- **Masking**, causal or padding. An image's tokens all see each other, so a mask needs a sequence
  task, and with it new data loading (no sequence dataset exists): the step's workplan decides
  whether masking waits for one.
- **Dropout in attention**, on the attention weights `P` or after the output projection.

Step 4's study left the patch models 2.2 points under the conv network at MNIST's size, and
attention's gain over the FFN block alone within a standard deviation; a deeper or multi-head
model is measured against those numbers. The case made for steps 4 and 5 is in this file's
history: `git show 7d3bd0a:docs/primitives-roadmap.md`.

## Out of scope

- Recurrent networks (RNN, LSTM): no sequence dataset, and attention covers the sequence case
  in the literature this project follows.
- GPU backends.
