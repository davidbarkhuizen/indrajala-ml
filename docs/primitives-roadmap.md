# Roadmap: the next ML primitives

**Status: a proposed order, not a workplan. Each step gets its own workplan before any code.**

Which primitive to add next, and in what order: normalization, residual connections, attention,
transformers.

## What exists

Dense backprop, the perceptron and MADALINE, ReLU, softmax with cross-entropy, conv and max
pooling, momentum, Adam, L2, dropout, ensembles, linear warmup, batch norm with ghost groups,
residual blocks, layer norm and multi-head self-attention over patch tokens, stacked into transformer
layers, and a causal transformer over token ids for next-token prediction, with a per-token output
and loss (README, Models, Batch normalization, Residual connections, Layer norm and attention and
The sequence task), built from composable layer specs and update rules. Each is in all three
implementations (pure Python, numpy, Rust). There is no padding mask, no dropout in attention, no
text generation and no recurrence.

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
6. **A sequence task with causal masking**: done (2026-10-10), as next-character prediction on
   four text corpora.

No step 7 is proposed yet. The candidates are the extension points step 5 named and step 6 left
(dropout in attention, grouped key/value heads, cross-attention, padding masks), generation, and
GELU, in [next-steps.md](next-steps.md), From multi-head attention and From the sequence task.

## 1 to 6. Done

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

The sequence task was done on 2026-10-10: `Embedding(vocabulary, size)`, a token part ending in
a token-wise softmax output with the mean per-token cross-entropy (the `"sequence"` network shape),
and `Attention(causal=True)`, the mask in attend before the max shift, in all three
implementations and the crate; four character-level corpora (Tiny Shakespeare, Herodotus, the
*Muqaddimah*, Euclid's *Elements*), each in its own dataset repository. The retired workplan and
its open work are in next-steps.md. The sequence study (`scripts/sequence_study.py`) found what
the literature predicts: a per-token model only reaches the bigram floor, one causal attention
layer takes 0.8 to 1.5 bits per character under it, a second layer 0.09 to 0.17 more (2-layer:
Euclid 1.50, Herodotus 1.95, Tiny Shakespeare 2.48, the *Muqaddimah* 2.79), and without the mask
the model copies the next input, 0.04 to 0.06 bits per character on held-out text. The case made
for step 6 is in this file's history: `git show f6e843b:docs/primitives-roadmap.md`.

## Out of scope

- Recurrent networks (RNN, LSTM): attention covers the sequence case in the literature this
  project follows.
- GPU backends.
