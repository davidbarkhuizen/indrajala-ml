# Roadmap: the next ML primitives

**Status: a proposed order, not a workplan. Each step gets its own workplan before any code.**

Which primitive to add next, and in what order: normalization, residual connections, attention,
transformers.

## What exists

Dense backprop, the perceptron and MADALINE, ReLU, softmax with cross-entropy, conv and max
pooling, momentum, Adam, L2, dropout, ensembles, linear warmup, batch norm with ghost groups,
residual blocks, layer norm and multi-head self-attention over patch tokens, stacked into transformer
layers (README, Models, Batch normalization, Residual connections and Layer norm and attention),
built from composable layer specs and update rules. Each is in all three implementations (pure
Python, numpy, Rust). There is no masking, no dropout in attention and no recurrence.

## The order

1. **Composable layers and optimizers** (a refactor, before the next primitive): done
   (2026-09-30); its open items are in [next-steps.md](next-steps.md).
2. **Batch normalization**: done (2026-09-30), and the conv batch-size study rerun with it
   (2026-10-01).
3. **Residual connections**: done (2026-10-01), dense blocks with identity shortcuts.
4. **Layer normalization and single-head self-attention**: done (2026-10-02), as a patch model
   on MNIST.
5. **Multi-head attention and a full transformer block**: done (2026-10-09), as deeper multi-head
   patch models on MNIST.

No step 6 is proposed yet. The candidates are the extension points step 5 named (masking with a
sequence task, dropout in attention, grouped key/value heads, cross-attention) and GELU, in
[next-steps.md](next-steps.md), From multi-head attention.

## 1 to 5. Done

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
deviation, and the conv network beats every patch model at this data size.

Multi-head attention was done on 2026-10-09: `Attention(heads, key_size)` in all three
implementations and the crate, each restructured into the same three blocks (project, attend,
combine) with named places for masking, dropout and grouped heads; a transformer layer is the
attention and FFN blocks step 4 built, written again for depth. The retired workplan and its open
work are in next-steps.md. The multi-head attention study (`scripts/multi_head_attention_study.py`)
found that more heads change nothing at this size (16 tokens of 32), and depth helps: two layers of
4 heads reach 97.0%, a point under the conv network. The case made for step 5 is in this file's
history: `git show d70cf0f:docs/primitives-roadmap.md`.

## Out of scope

- Recurrent networks (RNN, LSTM): no sequence dataset, and attention covers the sequence case
  in the literature this project follows.
- GPU backends.
