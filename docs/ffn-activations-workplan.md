# Workplan: FFN activations: GELU and SwiGLU (roadmap step 13)

**Status: decisions D1-D8 settled (2026-10-10) by building for extensibility, speed and real-world
production (the owner's direction, 2026-10-10). Stage 1 (this plan) done.**

Roadmap step 13 ([primitives-roadmap.md](primitives-roadmap.md)): the activations production
transformers use in their feed-forward blocks: GELU (BERT, GPT-2, ViT) and the gated SwiGLU
(PaLM, LLaMA and most decoders since 2022), in all three implementations and the crate.

## Why

The transformer blocks' FFNs are ReLU today. Production models moved off ReLU: GELU (Hendrycks &
Gimpel 2016, "Gaussian Error Linear Units") in BERT, GPT-2 and ViT, then gated linear units
(Shazeer 2020, "GLU Variants Improve Transformer"), SwiGLU above all, in PaLM (Chowdhery et al.
2022) and LLaMA (Touvron et al. 2023). The roadmap's original step proposed GELU alone; a
production FFN today is SwiGLU, so this step brings both, measured against ReLU on the long-form
corpora.

## Decisions (settled 2026-10-10)

Each lists the options considered, with pros and cons, and the choice.

- **D1. Scope. Settled: (a).**
  - (a) *Chosen.* `"gelu"` (the exact, `erf` form) and `"gelu_tanh"` (GPT-2's approximation) as
    `Dense` activations; a gated FFN for SwiGLU (D3); `erf` implemented once (D2); all three
    implementations and the crate; fixtures, checkpoints, golden entries; a study (D7). Pros: the
    FFNs every production family uses. Cons: a new FFN shape (gating) beside a new activation.
  - (b) GELU only, the original step. Cons: the activation production decoders have already moved
    past.
  - (c) SwiGLU only. Cons: GELU is still the encoders' (BERT, ViT) and GPT-2's, and the patch
    models (step 4) are ViT-shaped.
- **D2. `erf`. Settled: (a).** Stable Rust's standard library has no `erf`, numpy has none (only
  SciPy's), and Python's `math.erf` is the platform's libm, which isn't correctly rounded, so three
  implementations would disagree in the last bit.
  - (a) *Chosen.* One `erf`, a port of fdlibm's `s_erf.c` (Sun, 1993; the algorithm behind most
    libms' `erf`), written once in pure Python (`indrajala_ml/math/erf.py`), elementwise in numpy
    in the same operations and grouping, and in the crate; tested against `math.erf` within an
    ulp or two and against each other by bits. Pros: one function, the same bits in all three; no
    SciPy dependency. Cons: a numerical routine to own (its tests pin it).
  - (b) `math.erf` and a Rust crate's `erf` (`libm`). Cons: parity by bits lost; a dependency in
    the crate.
  - (c) Only the `tanh` approximation. Cons: a different function from BERT's and ViT's GELU; the
    tests would call it GELU wrongly.
- **D3. The gated FFN. Settled: (a).**
  - (a) *Chosen.* A spec `GatedDense(size, activation="silu")`: two token-wise projections of the
    input, `a = x W_g^T + b_g` and `v = x W_v^T + b_v`, out `silu(a) * v` (Shazeer 2020's
    `FFN_SwiGLU`), followed by the existing affine `Dense` back to `d`. SiLU is `x * sigmoid(x)`
    (Elfwing et al. 2018), and `"gelu"` gives GEGLU. The two projections are one `(2 * size, d)`
    weight for one product (speed), split after. Pros: one spec for the GLU family, its
    activation a field; one product. Cons: a new layer kind in three implementations and the
    crate, with a backward through the product.
  - (b) A `gated=True` field on `Dense`. Cons: `Dense` gains a second weight and a second meaning.
  - (c) Separate `Multiply` and activation layers. Cons: activations stay fused (next-steps.md,
    From composable layers), and a crossing per piece on Rust.
- **D4. Parameter parity. Settled.** A gated FFN has three matrices where ReLU's has two; the study
  compares at equal parameters, SwiGLU's `size` at two thirds of ReLU's (PaLM's and LLaMA's
  `8d/3` against `4d`).
- **D5. Where they may stand. Settled.** GELU wherever ReLU may (dense and token-wise hidden
  layers); `GatedDense` in a token block's FFN body, before the affine `Dense`. Both with dropout
  after them only through step 7's `Dropout`. Validation refuses the rest.
- **D6. Golden run and timing. Settled.** New entries: a sequence model with a SwiGLU FFN and one
  with GELU, in each implementation; earlier entries bit-identical. New ops in the crate, existing
  ones untouched: tier 0 unless a shared helper moves, then tier 1 on the dense case.
- **D7. The study. Settled.** The best sequence model from steps 7 to 12 with its FFN as ReLU, GELU
  and SwiGLU at equal parameters, on Herodotus, Euclid and the *Muqaddimah* (Tiny Shakespeare as a
  reference), at step 12's chosen context, 5 seeds, sized from a timing.
- **D8. Out of scope.** Other GLU variants beyond `GatedDense`'s activation field (ReGLU, Bilinear)
  until a use appears; activations as separate layers.

## Stages

1. **This workplan.**
2. **`erf`** (D2): pure Python, numpy, the crate; tests.
3. **GELU** (D1, D5): the activation in specs and all three implementations and the crate; parity.
4. **`GatedDense`** (D3): specs, numpy, the crate and a "Bump rust/" PR, Rust, pure Python; parity.
5. **Fixtures, checkpoints and golden entries** (D6).
6. **The study** (D7).
7. **Docs**; the workplan retired.
