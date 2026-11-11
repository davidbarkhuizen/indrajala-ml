# Roadmap: the next ML primitives

**Status: no step chosen.** This file lists the primitives that could come next. Each one gets its
own workplan (in `docs/workplans/library/`) before any code, and the owner sets the order.

Steps 1-15 are all built. They ran from composable layers to cross-attention, and the steps were
numbered in this file. The case made for each step and its place in the order is in this file's
history (`git show 989819a:docs/primitives-roadmap.md`). The retired workplans, named by step
("roadmap step 9"), are indexed in [workplans/retired.md](workplans/retired.md). What the library
has now is in the README (Models and the sections after it).

## How the next step is chosen

Since step 15, a library step has come from a need in a research plan, not from a list. The array
window, for example, was the machine-policy plan's M1 (indrajala-research). The goal set on
2026-10-10 still holds for any step: build for real-world production, with the long-form corpora
as targets, and never let a toy reference decide a design. When a step's study question is one
the literature answers, the owner has preferred the literature over a library study (steps 13-15,
2026-11-03). Offer that route before building a study.

## Candidates

Each candidate is described where its leftover work is, in [next-steps.md](next-steps.md). They
are grouped here by area, in no order.

- **Structure of a network** (next-steps, From residual connections, From batch norm, From
  multi-head attention, From composable layers):
  - conv residual blocks, which need 'same' padding;
  - projection shortcuts, and zero-initialised block ends;
  - group and instance norm;
  - a `TransformerLayer` spec;
  - separate activation layers;
  - a separate value width in attention, and padding masks beyond segments.
- **Attention and positions** (From rotary positions, From grouped-query attention, From context
  length):
  - ALiBi or T5's relative biases;
  - multi-head latent attention, and cache quantization;
  - sparse or sliding-window attention;
  - per-segment (varlen) tiling;
  - a class token.
- **Sequences and generation** (From generation, From cross-attention, From the sequence task):
  - a BPE tokenizer;
  - beam search, repetition penalties and stop sequences;
  - a band-masked prefill for prompts longer than `T`;
  - a decoder-only translation model;
  - word-level corpora.
- **Training** (From composable layers, From the RNG generators workplan, From context length):
  - parameter groups and per-layer learning rates;
  - mixed precision;
  - sequence parallelism;
  - run checkpoints that resume mid-epoch and resume ensembles.
- **Recurrence and the array window** (From the array window):
  - a supervised per-step output;
  - the program's ops for the other array layers;
  - a pure-Python array window.
- **Other architectures** (From the addition study):
  - a weight-tied (looped) transformer;
  - a 1-D convolution along the tokens;
  - an accessor for a layer's activations on a batch, which the probes need.

## Out of scope for now

- GPU backends, for the moment (the owner, 2026-10-12). Nothing else is excluded: work with no
  step yet is listed in next-steps as "not planned yet".
