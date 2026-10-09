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
6. **A sequence task with causal masking**: planned (2026-10-09,
   [sequence-task-workplan.md](sequence-task-workplan.md)). Next-token prediction on a small text
   dataset: the network's first per-token output and loss, and attention's first mask.

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

## 6. A sequence task with causal masking

Every model so far reads one example and gives one label: a patch model's tokens are pooled by
`TokenMean` before a single output. Attention's defining use, a model of sequences that predicts
each token from the ones before it (Vaswani et al. 2017; Radford et al. 2018, "Improving Language
Understanding by Generative Pre-Training"), needs three things the framework lacks: a sequence
dataset, an output and a loss at every token, and a causal mask so that token `t` attends only to
tokens `<= t`. Step 5 named the mask's place (attend, before the max shift: an additive mask on
the scores, the backward pass unchanged because a masked weight is exactly zero), and left it for
this step because without the other two it can't be trained or tested on a real task.

The candidates for step 6, each one of step 5's extension points (next-steps.md, From multi-head
attention) or GELU:

| candidate | pros | cons |
| --- | --- | --- |
| **A sequence task with causal masking** (proposed) | opens a new task family, language modelling, rather than refining the image one; the mask is the smallest attention change, already placed; per-token outputs and losses are framework structure every later sequence feature needs (padding masks, generation, cross-attention); a text dataset tests the framework off MNIST | the largest step: a dataset and loader, a new network shape, a per-token output and loss, and new evaluation (per-token accuracy, bits per character), in all three implementations and the crate |
| Dropout in attention | regularization the MNIST patch models might use; its place is named | needs a training and inference switch in a token layer and mask draws in three implementations' orders; the patch models don't overfit at 5 epochs, so a study would likely show nothing |
| GELU | ViT's and GPT's FFN activation; small | needs `erf`, which stable Rust lacks; an activation, not a primitive; at this size unlikely to move accuracy |
| Grouped- or multi-query attention | named place; fewer key/value parameters | an inference-memory optimization: nothing here is memory-bound, so there is nothing to measure |
| Cross-attention | named place | needs a second input to a layer, which `Sequential` doesn't have, and an encoder-decoder task: its own design, after a sequence task exists |

A sequence task comes first because the rest either need it (cross-attention, padding masks) or
have nothing to show on MNIST. Its workplan settles, with each option's pros and cons:

- **the dataset**: a small character-level corpus (Tiny Shakespeare, about 1.1 M characters and 65
  symbols, Karpathy 2015's char-rnn), packaged and checksum-pinned as MNIST is
  (`scripts/fetch_datasets.py`), or a synthetic task generated from a seed, or both;
- **the input**: one-hot tokens through the existing token-wise `Dense` (no new layer), or an
  `Embedding(vocabulary, d)` spec reading token ids;
- **the output and loss**: a token-wise softmax output at every position and cross-entropy averaged
  over the tokens, a new network shape beside `multiclass` and `single_output`, with per-token
  targets through the trainer, the evaluation and the save format;
- **the mask**: `Attention(causal=True)`, a field with a default (step 5's D7), in attend in all
  three implementations and the crate, with a test that a future token can't change an earlier
  output;
- **the study**: what a causal transformer reaches against a unigram and a bigram baseline and an
  FFN-only model, in bits per character, and that removing the mask lets the loss fall toward
  zero, on held-out text too, since every position can read the token it predicts (the leak a
  mask prevents).

Generation (sampling text from the model), padding masks and variable-length sequences stay out
unless the workplan finds them needed.

## Out of scope

- Recurrent networks (RNN, LSTM): attention covers the sequence case in the literature this
  project follows.
- GPU backends.
