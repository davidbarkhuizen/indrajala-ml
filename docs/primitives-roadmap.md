# Roadmap: the next ML primitives

**Status: a proposed order, not a workplan. Each step gets its own workplan before any code.**

Which primitive to add next, and in what order: normalization, residual connections, attention,
transformers.

## What exists

Dense backprop, the perceptron and MADALINE, ReLU, softmax with cross-entropy, conv and max
pooling, momentum, Adam, L2, dropout, ensembles, linear warmup and batch norm with ghost groups
(README, Models and Batch normalization), built from composable layer specs and update rules. Each
is in all three implementations (pure Python, numpy, Rust). There is no residual connection, no
layer norm, no recurrence and no attention.

## The order

1. **Composable layers and optimizers** (a refactor, before the next primitive): done
   (2026-09-30); its open items are in [next-steps.md](next-steps.md).
2. **Batch normalization**: done (2026-09-30), and the conv batch-size study rerun with it
   (2026-10-01).
3. **Residual connections.**
4. **Layer normalization and single-head self-attention**, as a patch model on MNIST.
5. **Multi-head attention and a full transformer block.**

## 1 and 2. Done

Composable layers and optimizers, then batch normalization, both done (2026-09-30); their retired
workplans and the work they left open are in [next-steps.md](next-steps.md). The case made for
each is in this file's history: `git show 058087a:docs/primitives-roadmap.md`.

The conv batch-size study was rerun with batch norm, plain and with ghost groups of 32 (Goyal et
al. 2017's full setup), on 2026-10-01. Batch norm adds about a point of accuracy, but the linear
rule still fails at B = 512 at both momenta (`batch_size_scaling.py`'s docstring; open questions
in next-steps.md). Residual connections are next.

## 3. Residual connections

A small change once layers compose: the skip adds the block's input to its output. They are
needed by the attention block (step 4) and by any network deep enough for batch norm to matter.

## 4 and 5. Attention after that, not first

- **A transformer is a composition, not one primitive.** It needs token embeddings, layer norm,
  residual connections, a matrix product between two activation tensors (every current op
  multiplies weights by activations), a softmax over the sequence, masking and positional
  encodings. Steps 2 and 3 are prerequisites, and layer norm is batch norm over another axis.
- **There is no sequence dataset.** The first attention model should be a small vision
  transformer on MNIST (Dosovitskiy et al. 2020): 7x7 patches as 16 tokens, one attention block.
  It can be compared against the conv results already measured. Text or other sequence tasks
  would need new data loading.
- **The cost multiplies by three.** Each primitive is written in pure Python, numpy and Rust,
  with crate ops, a type stub and parity tests. Attention's backward pass is several times the
  work of batch norm's.

## Out of scope

- Recurrent networks (RNN, LSTM): no sequence dataset, and attention covers the sequence case
  in the literature this project follows.
- GPU backends.
