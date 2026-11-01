# Workplan: FFN activations: GELU and SwiGLU (roadmap step 13)

**Status: decisions D1-D8 settled 2026-10-10; revised 2026-11-01 against the library as steps 9 to
12 left it (D2 and D7 corrected, D3 to D6 extended, D9 and D10 new), settled by building for
extensibility, speed and real-world production (the owner's direction, 2026-10-10). Stage 1 (this
plan) done.**

Roadmap step 13 ([primitives-roadmap.md](../../primitives-roadmap.md)): the activations production
transformers use in their feed-forward blocks: GELU (BERT, GPT-2, ViT) and the gated SwiGLU
(PaLM, LLaMA and most decoders since 2022), in all three implementations and the crate.

## Why

The transformer blocks' FFNs are ReLU today. Production models moved off ReLU: GELU (Hendrycks &
Gimpel 2016, "Gaussian Error Linear Units") in BERT, GPT-2 and ViT, then gated linear units
(Shazeer 2020, "GLU Variants Improve Transformer"), SwiGLU above all, in PaLM (Chowdhery et al.
2022) and LLaMA (Touvron et al. 2023). The roadmap's original step proposed GELU alone; a
production FFN today is SwiGLU, so this step brings both, measured against ReLU on the long-form
corpora.

## What the revision found (2026-11-01)

The plan was written before generation (step 9), the training recipe (step 10), segments (step 11)
and context length (step 12), and before the crate's correctly rounded `exp` settled how the three
implementations share transcendental functions. Read against today's code:

- **One `erf` can't give the same bits in all three by itself.** fdlibm's `erf` calls `exp` twice
  for `|x| >= 1.25`. The library's rule for `exp` is settled (README, Batch normalization's and
  attention's parity paragraphs): each implementation uses its own (Python's `math.exp`, numpy's
  `np.exp`, the crate's correctly rounded port of CORE-MATH's `cr_exp`, the aarch64 workplan's
  D4), and the parity tests give the numpy layers the other implementation's `exp`
  (`tests/helpers.py`'s `exp_by_math`, `exp_by_crate`). So `erf` follows the same rule: the same
  operations everywhere, the bits equal where `exp`'s are (D2).
- **`tanh` was unplanned.** `gelu_tanh` needs it; `math.tanh`, `np.tanh` and Rust's `f64::tanh`
  (the platform's libm, which breaks the crate's x86_64/aarch64 parity) differ. The window-recurrent
  cell already has the answer, `tanh(z) = 1 - 2 / (exp(2z) + 1)` through `exp` (its W15; numpy's
  `recurrent_array_layer.tanh`, the crate's `recurrent.rs` `tanh_in_place`) (D2).
- **The crate's ReLU is fused into its dense ops** (`fused.rs`: `layer_relu_forward`, `_batch`,
  `layer_relu_hidden_delta`, `_batch`, `_skip`, `_skip_batch`, and the window program's call), and
  ReLU's backward reads its output (`a > 0`) where GELU's needs its input `z` (D9).
- **Generation decodes row by row** (step 9; `RowDecoding`), so a new FFN layer must decode one
  token a row; **AdamW decays by parameter pair** (step 10; a layer's `decayed` flags; README,
  Update rules), so a new layer's weights must say they are decayed and its biases not (D3).
- **Studies now run on Rust under the recipe** (warmup-cosine, AdamW 0.1, clipping at 1.0) for 30
  epochs, and step 12's study is choosing the context and the token budget, so D7's study waits for
  its report.
- Segments (step 11) and tiled attention (step 12) don't touch the FFN: it is token-wise, and a pad
  token's FFN output is never read by a counted loss.

## Decisions (settled 2026-10-10; revised 2026-11-01)

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
- **D2. `erf`, `tanh` and SiLU: the transcendental functions. Settled: (a) (revised 2026-11-01).**
  Stable Rust's standard library has no `erf`, numpy has none (only SciPy's), and Python's
  `math.erf` is the platform's libm.
  - (a) *Chosen.* A port of fdlibm's `s_erf.c` (Sun, 1993, its permission notice kept beside the
    code; the algorithm behind most libms' `erf`) in pure Python, elementwise in numpy (each range's
    branch under a mask, in the same operations and grouping) and in the crate, each calling **its
    own implementation's `exp`**: the module-level `exp` the parity tests already patch in numpy,
    `math.exp` in pure Python, `cr_exp` in the crate. Its other operations are `+ - * /`, `fabs`
    and one exact truncation (the low 32 bits of `|x|` cleared, a bit mask in all three), so the
    three agree by bits wherever `exp` does, and the crate stays bit-identical across architectures
    (only `cr_exp` and `+ - * /`, no FMA contraction). `tanh` (for `gelu_tanh`) is W15's
    `1 - 2 / (exp(2z) + 1)` and SiLU (D3) `z * sigmoid(z)` with the existing
    `1 / (1 + exp(-z))`, both through `exp` in the same way. Tests: `erf` within 1 ulp of a
    high-precision reference (a `decimal` series in the test, no new dependency) on a grid and on
    each of fdlibm's range boundaries, and the three implementations by bits with `exp` patched.
    Pros: the library's existing rule, no new parity mechanism; small (about 100 lines a side);
    the crate stays bit-identical on x86_64 and aarch64. Cons: not correctly rounded (fdlibm's
    error is under 1 ulp); numpy's production bits follow `np.exp`'s, as today's softmax and
    sigmoid do.
  - (b) A port of CORE-MATH's correctly rounded `cr_erf` in all three, `cr_exp`'s companion. Pros:
    bits defined by mathematics. Cons: double-double arithmetic with fused multiply-adds, which
    numpy lacks, so numpy's port would be slow and long; and numpy's `exp` elsewhere still isn't
    correctly rounded, so the parity tests' patching stays anyway.
  - (c) `math.erf` and a Rust crate's `erf` (`libm`). Cons: parity by bits lost; a dependency in
    the crate; nothing in numpy.
  - (d) Only the `tanh` approximation. Cons: a different function from BERT's and ViT's GELU; the
    tests would call it GELU wrongly.
- **D3. The gated FFN. Settled: (a) (extended 2026-11-01).**
  - (a) *Chosen.* A spec `GatedDense(size, activation="silu")`: two token-wise projections of the
    input, `a = x W_g^T + b_g` and `v = x W_v^T + b_v`, out `silu(a) * v` (Shazeer 2020's
    `FFN_SwiGLU`), followed by the existing affine `Dense` back to `d`. SiLU is `x * sigmoid(x)`
    (Elfwing et al. 2018), and `"gelu"` gives GEGLU. The two projections are one `(2 * size, d)`
    weight and one `(2 * size,)` bias, one product (speed), split after; so the layer is one
    parameter pair to the optimizers (the Rust optimizer's `_Pair`), its weight decayed under AdamW
    and its bias not, as `Dense`'s. Biases as `Dense`'s ReLU layer has them (PaLM and LLaMA drop
    every bias; a bias-free model is its own change, across every layer, not this step's). It
    decodes one token a row (`RowDecoding`, step 9) as the token-wise dense layer does. Pros: one
    spec for the GLU family, its activation a field; one product; one optimizer pair. Cons: a new
    layer kind in three implementations and the crate, with a backward through the product.
  - (b) A `gated=True` field on `Dense`. Cons: `Dense` gains a second weight and a second meaning.
  - (c) Separate `Multiply` and activation layers. Cons: activations stay fused (next-steps.md,
    From composable layers), and a crossing per piece on Rust.
- **D4. Parameter parity. Settled (made exact 2026-11-01).** A gated FFN has three matrices where
  ReLU's has two; the study compares at equal parameters, SwiGLU's `size` near two thirds of
  ReLU's (PaLM's and LLaMA's `8d/3` against `4d`). At the sequence models' `d = 64`, ReLU's FFN
  (`size` 256) has `2 * 64 * 256 + 256 + 64 = 33,088` parameters and a gated FFN of `size` `g`
  has `3 * 64 * g + 2g + 64 = 194g + 64`: `g = 170` gives 33,044 (0.13% fewer), the nearest. The
  study writes both counts beside its arms.
- **D5. Where they may stand. Settled.** GELU wherever ReLU may (dense and token-wise hidden
  layers); `GatedDense` in a token block's FFN body, before the affine `Dense`. Both with dropout
  after them only through step 7's `Dropout`. Validation refuses the rest. The window program's ops
  (the crate's `program.rs`, the research networks') stay ReLU until a research network asks for
  another (D8).
- **D6. Golden run and timing. Settled (revised 2026-11-01).** New entries: a sequence model with a
  SwiGLU FFN and one with GELU, in each implementation and in the Rust reference
  (`golden_run_rust.json`, checked on both architectures' CI runners); earlier entries
  bit-identical. Timing by tier: `erf` and `GatedDense` are new ops, existing ones untouched: tier
  0. D9's activation parameter changes the timed ReLU ops: tier 1, `focused_benchmark` on the dense
  case, `--order ONNO`, on `jebel` (so after step 12's study frees it).
- **D7. The study. Settled (revised 2026-11-01).** The best sequence model from steps 7 to 12 with
  its FFN as ReLU, GELU and SwiGLU at equal parameters (D4), on Herodotus, Euclid and the
  *Muqaddimah* (Tiny Shakespeare as a reference), both held-out splits, 5 seeds, on Rust under the
  recipe for 30 epochs, at the context and token budget step 12's findings choose; sized from a
  timing. Its claims registered before the data in their own PR (experiment-design §2), as step
  12's were (#1175). So stage 6 starts only after step 12's report.
- **D8. Deferred.** Other GLU variants beyond `GatedDense`'s activation field (ReGLU, Bilinear);
  activations as separate layers; GELU and `GatedDense` in the window program (D5). Each when a use
  appears.
- **D9. GELU in the crate's fused dense ops. Settled: (a) (new 2026-11-01).**
  - (a) *Chosen.* The fused ReLU family becomes one family with an activation argument (ReLU,
    GELU, GELU-tanh), the forward keeping `z` where the activation's derivative needs it (GELU's)
    and the hidden deltas reading `z` or `a` as the activation says; the ReLU arm's arithmetic
    unchanged, so every ReLU entry stays bit-identical (the golden runs check it) and a tier 1 A/B
    shows its speed held. Pros: one family for every later activation (SiLU's derivative for
    `GatedDense` reuses it); no duplicated kernels (the DRY passes' rule). Cons: a timed path
    touched, so tier 1, which waits for `jebel`.
  - (b) A parallel `layer_gelu_*` family. Pros: tier 0, nothing timed touched. Cons: six ops
    duplicated per activation, and a third for the next activation.
  - (c) GELU as an elementwise op after the affine product (no fusion). Pros: simplest. Cons: an
    extra pass and crossing per layer on Rust; the activations stay fused (next-steps.md, From
    composable layers).
- **D10. Order of the stages, while step 12's study holds `jebel`. Settled (new 2026-11-01).**
  Stage 2 (`erf`, tier 0) and stage 4's specs, numpy and pure-Python work need no timing and go
  first; stage 3's crate change (D9, tier 1) waits for `jebel`, its A/B run once the study ends.

## Stages

1. **This workplan** (revised 2026-11-01).
2. **`erf`** (D2): pure Python, numpy, the crate (a crate PR, then its bump); the high-precision
   reference test and the parity by bits with `exp` patched.
3. **GELU** (D1, D5, D9): `"gelu"` and `"gelu_tanh"` in specs and validation, the numpy and
   pure-Python dense and token-wise layers, the crate's activation-parameterised fused family
   (crate PR, bump, tier 1 A/B on `jebel`); persistence; parity.
4. **`GatedDense`** (D3): specs, numpy, the crate and a "Bump rust/" PR, Rust, pure Python; row
   decoding; the optimizer pair and its decay flag; persistence; parity.
5. **Fixtures, checkpoints and golden entries** (D6), the Rust reference on both architectures.
6. **The study** (D7): after step 12's report; registration first, then the sweep.
7. **Docs**; the workplan retired.
