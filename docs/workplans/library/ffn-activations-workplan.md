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

- **D11. The study's ReLU arm. Settled: (a) (new 2026-11-02).**
  - (a) *Chosen.* The context-length study's `256` runs (step 12's sweep, 40 runs). That arm is this
    study's ReLU model exactly: the 2-layer rotary model at windows of 256 with the document mask,
    the recipe, the token budget, 30 epochs on Rust, the same seeds. ReLU's arithmetic is
    bit-identical since then (the golden runs through stages 3 to 5). The runs are read from that
    study's two result files, checked by sha256 against its findings, and a probe reruns one run's
    first epoch on this commit and checks its held-out cross-entropy by bits. Pros: a third of the
    study's compute saved (80 runs, not 120). The control was run once, before any new arm, so
    nothing about the new arms shaped it; the precedent is the rotary study's `read_control`.
    Cons: the control and the new arms ran on different commits (the bit check covers that) and on
    different machines (Rust's bits are machine-independent).
  - (b) Rerun ReLU beside the new arms. Pros: one commit, one sweep. Cons: 40 more runs, about half
    a day of `stone` for numbers already in hand, bit for bit.

## The study's registration (stage 6, written before any of its data)

Under the research repo's `docs/protocols/experiment-design.md`. The code is
`indrajala_ml/studies/ffn_activation_study.py` (its docstring is the protocol; the training is
`context_length_study.train_packed`), and `tests/studies/test_ffn_activation_study.py` checks its
configuration against this section.

**Cells.** The context-length study's `256` model (step 12's choice: windows of 256 on the long-form
corpora, the document mask on): the 2-layer rotary model, `d = 64`, 4 heads, no dropout, under the
recipe (warmup over the first epoch, then cosine to a tenth of the corpus's tuned rate, AdamW 0.1,
clipping at 1.0) at the token budget (batches of 8 windows), each epoch's training windows cut anew
and the held-out windows fixed, 30 epochs on Rust. Three arms differ only in each block's FFN, at
equal parameters (D4):

| arm | FFN hidden layer | FFN parameters a block | runs |
| --- | --- | --- | --- |
| `relu` | `Dense(256, "relu")` | 33,088 | the context-length study's (D11) |
| `gelu` | `Dense(256, "gelu")` | 33,088 | new |
| `swiglu` | `GatedDense(170)` | 33,044 | new |

Every corpus (Herodotus, Euclid and the *Muqaddimah*, plus Tiny Shakespeare as a reference outside
the claims), both held-out splits, 5 seeds (a screen), the same seeds in every arm. That is 4 x 2 x 5
= 40 runs an arm, and 80 new.

**Feasibility** (experiment-design §1, information-flow's register). Every arm is the model the
context-length study trained on every corpus at 256, with one FFN swapped for another of equal
parameters, so each can represent the character model the ReLU arm does. The exclusion bound and
code counts are a finite machine's and don't apply to a character model. The learning rates are
the ReLU model's tuned ones (`rotary_study.RATES`), not retuned per arm. That favours ReLU if
anything, so it can only make F1 and F2 harder to hold. The budget is measured before launch
(below).

**Claims** are on the final-epoch held-out bits per character over the counted targets, paired by
(corpus, split, seed) over the long-form corpora: 30 pairs a comparison, a sign test on the count of
pairs lower, and Holm over the three at 0.05.

- **F1, SwiGLU below ReLU, one-sided.** The direction is the literature's: Shazeer 2020's Table 1
  (held-out log-perplexity at equal parameters and computation) has FFN_SwiGLU below FFN_ReLU at
  both training lengths, 1.944 against 1.997 at 65,536 steps and 1.636 against 1.677 at 524,288,
  and PaLM and LLaMA adopted it. F1 holds when the Holm-adjusted one-sided p < 0.05, and is refuted
  otherwise. Refuted means SwiGLU isn't shown to help at this model, corpus scale and budget, not
  that it can't.
- **F2, GELU against ReLU, two-sided.** No direction is registered. Hendrycks & Gimpel 2016
  report GELU ahead, but Shazeer 2020's Table 1 has FFN_GELU below FFN_ReLU only at 65,536 steps
  (1.983 against 1.997) and above it at 524,288 (1.679 against 1.677). F2 differs when the
  Holm-adjusted two-sided p < 0.05, and is undecided otherwise.
- **F3, SwiGLU against GELU, two-sided.** No direction is registered: Shazeer's table favours
  SwiGLU at both lengths (1.944 against 1.983, 1.636 against 1.679), but either result would change
  the default, so both sides count. F3
  differs when the Holm-adjusted two-sided p < 0.05, and is undecided otherwise.
- **Descriptive, not tested:** per cell, each arm's final bits (mean ± sd), its last three epochs'
  fall (convergence), its training bits, the held-out bits by position (16 buckets), and its
  seconds per epoch (each arm's cost; the ReLU arm's seconds were measured on `jebel` and `stone`
  in step 12, so the cost comparison is labelled across machines).

**What a result does.**
- If F1 holds, SwiGLU becomes the FFN of later sequence work, unless F3 differs with GELU lower,
  in which case GELU does.
- If F1 is refuted and F2 differs with GELU lower, GELU does.
- Otherwise ReLU stays.

No setting changes after the data. A flaw found mid-study is recorded and fixed in a new registered
experiment.

**Budget, backend and hosts** (measured before launch: `ffn_activation_study.py time`, one epoch
of Herodotus's contiguous split an arm on Rust, one job at a time on `stone`, 2026-11-02 13:11 to
13:15).
- **Seconds a training epoch:** ReLU 66, GELU 76, SwiGLU 75. GELU's `erf` and SwiGLU's gate cost
  about 15% over ReLU at this size.
- **Memory:** a peak of 0.47 GB a run.
- **The control:** the ReLU probe reproduced the context-length study's first epoch by bits
  (held-out cross-entropy 1.7314670724730785 both ways), as D11 requires.
- **The run:** the study runs on **Rust**, **30 epochs**, on **`stone`**, 8 workers (the owner's
  grant, 2026-11-02, for this study at any hour; 8 workers run 1.34 times one job's time, the
  working notes). Herodotus is the largest corpus, so its runs take about 50 minutes and the 80
  runs about 7 to 10 hours.
- **Resuming:** each run's result is written as it ends, so a stopped sweep resumes by whole runs.

## Stages

1. **This workplan** (revised 2026-11-01).
2. **`erf`** (D2): pure Python, numpy, the crate (a crate PR, then its bump); the high-precision
   reference test and the parity by bits with `exp` patched.
3. **GELU** (D1, D5, D9): `"gelu"` and `"gelu_tanh"` in specs and validation, the numpy and
   pure-Python dense and token-wise layers, the crate's activation-parameterised fused family
   (crate PR, bump, tier 1 A/B on `jebel`); persistence; parity. *Done 2026-11-02* (crate #90;
   ml #1182, the benchmark's ReLU cases, and the bump, #1183). What the code settled:
   - **The expressions and their grouping** are the README's (Activations), the same in
     `python/activations.py`, `numpy/activations.py` and the crate's `core/src/activation.rs`;
     the three agree by bits on the crate's `exp` (`test_gelu_implementations.py`), and the crate is
     within a few ulps of a 60-digit reference (its `tests/test_activation_ops.py`).
   - **`tanh` lives in `activations.py`** (numpy and pure Python), moved from the recurrent
     modules, which import it: one `tanh` per implementation, the cell's and GELU-tanh's. Tests
     that give the cell another `exp` now patch `activations.exp` too.
   - **The crate's family replaces the ReLU names** (`layer_relu_*`, `array_relu_mask`):
     `layer_activation_forward*` (returning `(a, z)`, `z` only for the GELUs),
     `layer_activation_hidden_delta*`, their `_skip` variants and `array_activation_mask`, the
     activation by name. Conv and batch norm call the mask's ReLU arm. A breaking change for the
     crate's next PyPI release (0.1.0 now).
   - **Layers:** `GELUArrayLayer` (numpy), `GELULayer`/`GELUTanhLayer` (pure Python, nodes keeping
     `z` per example), `ActivationRustArrayLayer` (Rust; `ReLURustArrayLayer` its ReLU arm); the
     token-wise layers take `"gelu"`/`"gelu_tanh"` and keep `Z`. Format 2 needs no change: the
     activation is the `Dense` entry's field.
   - **The benchmark had no ReLU case:** `focused_benchmark`'s dense case timed the sigmoid layer
     only, so #1182 added ReLU's forward and hidden delta (`--shape relu`) before the A/B.
   - **The tier 1 A/B** (`focused_benchmark --shape relu --shape "conv 28x28"`, 6 passes;
     #1183's comment): every batched ReLU op and every conv row within noise; one row consistently
     slower, the single-example ReLU `hidden_delta` at 30 x 784 on Rust, +6.5% (about 0.2 us a
     call), accepted as D9's cost on the legacy single-example path. A first run had it and the
     256 x 64 row slower by more; the crate's second commit dropped the mask's repeated shape check,
     which settled 256 x 64.
4. **`GatedDense`** (D3): specs, numpy, the crate and a "Bump rust/" PR, Rust, pure Python; row
   decoding; the optimizer pair and its decay flag; persistence; parity. *Done 2026-11-02* (crate #92 and its bump).
   What the code settled:
   - **The layout:** `W` is `(2 * size, d)`, the gate's rows then the value's, and `b`
     `(2 * size,)`. Randomization draws it as one dense layer's pair, so the three implementations
     draw the same weights. Its delta is the pre-activation's, `(rows, 2 * size)`: each row's gate
     part `(d * v) * act'(a)`, then its value part `d * act(a)`. The downstream and the gradients
     are the dense ops' on that delta, so the crate gained only `gated_forward_batch` and
     `gated_delta_batch`. SiLU joins the activation family (`"silu"`), through the dense layers'
     sigmoid.
   - **Where it stands:** a token block's body, right before its affine layer (D5). Validation
     refuses it elsewhere: outside a body, ending one, before attention or a LayerNorm, after the
     mean, flat, and in a window.
   - **Layers:** `GatedDenseArrayLayer` (numpy), `GatedDenseRustArrayLayer`, and `GatedDenseLayer`
     (pure Python, with nodes keeping the gate's and the value's pre-activations and deltas per
     example). Format 2's kind is `"gated_dense"`.
   - **Tests:**
     - SiLU by bits across the three implementations (`test_activation_implementations.py`).
     - SwiGLU and GEGLU cases in the shared spec tables (gradient checks, parity and format 2 on
       every implementation, a dropped SwiGLU FFN included).
     - Decoding by bits on Rust and pure Python.
     - D4's counts on all three implementations (`test_gated_dense.py`).
     - Tier 0: new ops only.
5. **Fixtures, checkpoints and golden entries** (D6), the Rust reference on both architectures. *Done
   2026-11-02.*
   - **Golden entries:** `numpy/rust/python gelu model` (GELU_SPECS, the sequence model with the
     exact GELU in its FFN) and `... swiglu model` (SWIGLU_SPECS, `GatedDense(3)`), after the
     packed ones. The local file has 6 more entries (137), and the committed Rust reference
     (`golden_run_rust.json`, checked on x86_64 and aarch64 in CI) has 2 more (45). No earlier
     entry moved.
   - **Fixtures:** `SequenceGelu<class name>` and `SequenceSwiGLU<class name>` for each
     implementation's sequence class (12 files), each a format-2 file pinning a dense entry's
     `"gelu"` and a `"gated_dense"` entry. Checkpoints are format 2's: the fixtures hold the
     optimizer state after two steps.
6. **The study** (D7): after step 12's report; registration first, then the sweep. *Registered 2026-11-02* (the section above, D11).
7. **Docs**; the workplan retired.
