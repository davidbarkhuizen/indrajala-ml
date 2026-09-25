# Roadmap: the next ML primitives

**Status: a proposed order, not a workplan. Each step gets its own workplan before any code.**

Which primitive to add next, and in what order: normalization, residual connections, attention,
transformers.

## What exists

Dense backprop, the perceptron and MADALINE, ReLU, softmax with cross-entropy, conv and max
pooling, momentum, Adam, L2, dropout, ensembles and linear warmup (README, Models). Each is in
all three implementations (pure Python, numpy, Rust). There is no normalization, no residual
connection, no recurrence and no attention.

## The order

1. **Composable layers and optimizers** (a refactor, before the next primitive).
2. **Batch normalization**, then the conv batch-size study with it.
3. **Residual connections.**
4. **Layer normalization and single-head self-attention**, as a patch model on MNIST.
5. **Multi-head attention and a full transformer block.**

## 1. Composable layers and optimizers first

Every feature so far is a sibling class per backend: `Momentum…`, `Conv…`, `MomentumConv…`.
Combining two features costs planned stages: momentum with conv took three PRs (#448, #451,
#452). Batch norm crossed with {dense, conv} × {SGD, momentum, Adam} × three implementations
would multiply the class count again.

A planned step should make the optimizer and the normalization composable (for example, a
network built from a layer spec with an optimizer object) instead of adding subclasses. This is
a design decision for the owner, and it needs its own workplan: the save format, the registry
walks in the tests and the golden run all depend on the class structure.

## 2. Batch normalization next

- **It serves the open question.** The conv batch-size study hit a ceiling on the stable rate:
  the linear rule fails at B = 512 at momentum 0.0 and 0.9 (findings in `batch_size_scaling.py`).
  Batch normalization is the literature's standard way to raise it (Ioffe & Szegedy 2015). It is
  also part of the setup the study tests: Goyal et al. 2017 train ResNet-50, which uses it, and
  their section 2.3 fixes its statistics at 32 examples per worker whatever the total batch.
  Without it, the study is not comparable with the paper's full setup.
- **It tests the parity rules.** It is the first layer whose forward pass depends on the rest of
  the batch. It behaves differently in training (batch statistics) and inference (running
  averages), and its backward pass is a known source of errors. Hand-computed tests and gradient
  checks in all three implementations should catch those errors.
- **It fits the current design.** It is a per-layer primitive like dropout or ReLU, with numpy
  and Rust ops, plus the per-channel variant for conv.

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
