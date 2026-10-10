# Next steps: work left over from completed workplans

A workplan is deleted once its last stage merges, and never before: only when every stage and
decision is resolved and only future work is left. Whatever it left open (its "After this plan"
list, and the parts of its "Out of scope" that still bind later work) moves here. A workplan still
in progress keeps its own list: [pypi-release-workplan.md](pypi-release-workplan.md),
[attention-dropout-workplan.md](attention-dropout-workplan.md),
[rotary-positions-workplan.md](rotary-positions-workplan.md),
[generation-workplan.md](generation-workplan.md),
[training-recipe-workplan.md](training-recipe-workplan.md),
[segments-workplan.md](segments-workplan.md),
[context-length-workplan.md](context-length-workplan.md),
[ffn-activations-workplan.md](ffn-activations-workplan.md),
[grouped-query-attention-workplan.md](grouped-query-attention-workplan.md),
[cross-attention-workplan.md](cross-attention-workplan.md). The order
of the next ML
primitives is in [primitives-roadmap.md](primitives-roadmap.md).

## Retired workplans

Each one is still in git history, with its decisions (D1, D2, ...), design and stages. Code and
docs cite them by section:

| Workplan | Plan PR | Stages | Read it with |
| --- | --- | --- | --- |
| Composable layers and optimizers (roadmap step 1) | #475 | #476-#485 | `git show 3a5d179:docs/composable-layers-workplan.md` |
| The A/B harness and a stand-alone measurement guide | #487 | #488-#491 | `git show 3a5d179:docs/ab-harness-workplan.md` |
| Batch normalization (roadmap step 2) | #486 | #493-#503 | `git show 189921c:docs/batch-norm-workplan.md` |
| Removing duplicated code (DRY) | #506 | #507-#514 | `git show e066333:docs/dry-workplan.md` |
| The conv batch-size study with batch norm | #516 | #517-#519 | `git show acbc49c:docs/conv-batch-norm-scaling-workplan.md` |
| Explicit generator objects (RNG generators) | #521 | crate #46, #522-#529 | `git show aaf8bb2:docs/rng-generators-workplan.md` |
| Residual connections (roadmap step 3) | #531 | crate #47, #532-#538 | `git show 365a173:docs/residual-connections-workplan.md` |
| Layer norm and single-head attention (roadmap step 4) | #540 | crate #48, #542-#548 | `git show 7d3bd0a:docs/layer-norm-attention-workplan.md` |
| Removing duplicated code, second pass (DRY rerun) | #550 | #551-#557 | `git show 88e1815:docs/dry-rerun-workplan.md` |
| Presets for the one-change combinations | #559 | #560-#564 | `git show 2a799bf:docs/presets-workplan.md` |
| The source layout | #566 | #567-#573 | `git show b362621:docs/source-layout-workplan.md` |
| A baseline on the new benchmark machine (i7-9700K) | #575 | #576-#590, #592 | `git show 480164d:docs/benchmark-machine-workplan.md` |
| A benchmark archive, tiered benchmarking, and a remote benchmark machine | #584 | #593-#598, archive #1-#11 | `git show 056b808:docs/benchmark-archive-workplan.md` |
| Multi-head attention and a transformer block (roadmap step 5) | #602 | crate #51-#52, #603-#613, archive #12-#13 | `git show d70cf0f:docs/multi-head-attention-workplan.md` |
| A sequence task with causal masking (roadmap step 6) | #620 | crate #53, #621-#634, archive #14-#17 | `git show f6e843b:docs/sequence-task-workplan.md` |
| Pure Python's random draws in numpy's order (RNG draw order) | #637 | #649-#652, archive #18-#19 | `git show 7a80858:docs/rng-draw-order-workplan.md` |

The optimization docs (the Rust-against-numpy baseline, and the implemented, rejected and
candidate optimizations) were retired the same way: `git show 0a04977:docs/optimizations.md` and
`git show 0a04977:docs/optimizations/<name>.md`.

What they built is documented in the README (Models, Saving and loading, Update rules, Batch
normalization, Residual connections, Layer norm and attention, The sequence task), [measurement.md](measurement.md)
and [rng-audit.md](rng-audit.md). The batch-size studies' findings are in
`indrajala_ml/studies/batch_size_scaling.py`'s docstring, the depth study's in
`scripts/residual_depth_study.py`'s, the patch-attention study's in
`scripts/patch_attention_study.py`'s, the multi-head attention study's in
`scripts/multi_head_attention_study.py`'s, the sequence study's in `scripts/sequence_study.py`'s.

## From composable layers

- **New combinations.** The Sequential networks build every spec list `validate_layer_specs`
  accepts under every rule, in all three implementations. Every one-change combination has a
  preset in all three (the presets workplan; README, Presets). These have none:
  - two or more changes at once: ReLU, dropout, cross-entropy or softmax under `Momentum`, `Adam`
    or `WeightDecay`, ReLU with dropout, dropout with a softmax output, and so on, dense and conv
    (the presets workplan rejected one preset per combination: the class count grows
    multiplicatively, and the Sequential networks are the API for them);
  - batch norm, dense and conv, under every rule (README, Batch normalization).

  One of them, conv then pool, ReLU, dropout and a softmax output under `Adam`, is trained against
  its pure-Python reference (`tests/model/networks/test_sequential_array_network.py`). Giving any other one a
  preset, or using it in a demo, is a behaviour change, not refactoring: its own PR, with
  hand-computed tests and parity tests in all three implementations.
- **Still refused by the specs:** dropout on a ReLU layer (the dropout op is fused with the
  sigmoid), a conv or pool layer after a dense one (the fused hidden delta reads the next layer's
  `W`), and one rule per layer.
- **Weight decay with momentum or Adam** needs a published form chosen and cited first (README,
  Update rules). For Adam it is chosen: AdamW (Loshchilov & Hutter 2019), roadmap step 10.
- **Ensembles over the new one-output presets** (the presets workplan). The ensembles fix their
  sub-network class (`classifier_cls`); a choice of sub-network is its own change.
- **Demos for the new conv presets** (the presets workplan's D4), each with its own tuned,
  measured PR: a demo makes an accuracy claim.

Still out of scope, for later workplans too (batch norm's included):

- Separate activation layers, which would add a crossing per layer on Rust. Activations stay fused.
- Parameter groups and per-layer learning rates. (Schedulers beyond today's `lr_schedule.py` were
  lifted out of scope on 2026-10-10: roadmap step 10.)
- Deleting or renaming any network class (the workplan's D1).

## From batch norm

- **Batch norm under dropout or after pool**, if a use appears. Neither passes
  `validate_layer_specs` today.
- **Folding batch norm into the preceding weights** for inference: a speed change, measured.

Still out of scope:

- Group norm and instance norm. They normalize within an example, a different layer (layer
  norm, the third, is built: README, Layer norm and attention).
- Synchronized statistics across workers or processes.

## From the conv batch-size study with batch norm

The rule still fails at B = 512 with batch norm, plain or in ghost groups of 32
(`batch_size_scaling.py`'s docstring). Open questions it leaves, each a scratch probe or a short
sweep with the existing script:

- **A gentler base rate.** At momentum 0.0 every batch-32 rate from 0.5 to 4 is within the seeds'
  spread, and the study's rule picked 2. Scaled from 0.5, B = 512 runs at 8, inside the stable
  range, where scaled from 2 it runs at 32, past the rate that fails at B = 32.
- **Capped rates at B = 512,** as the old study probed without batch norm (4, 8, 16, 24 with a
  1-epoch warmup): how far batch norm raises the stable rate at B = 512.
- **Longer runs at B = 512.** The unscaled control at momentum 0.9 ends 0.3 points below the band
  after 3 epochs (354 steps) and is still climbing; the scaled rate 32 at 0.0 reaches the band's
  neighbourhood in epoch 2 and then falls. More epochs would show whether either settles.
- **The dense momentum 0.9 rerun.** The dense study's momentum 0.9 cells with warmup, including
  its finding that the rule holds to B = 512, were measured with eq. (10) momentum without the
  momentum correction, and are pending a rerun on eq. (9) (`batch_size_scaling.py`'s docstring).
  The sweep script runs it as it is: `scaling --lr32 0.9=0.25` at the dense study's grid.

The old conv plan's timing and demo stages were for a rule that holds, and stay unplanned.

## From the A/B harness

- **Count the savings.** After the first two real A/Bs through `scripts/ab.py`, count the agent
  turns and characters read per A/B from the session transcripts, against the numbers before the
  harness: 20-30 checks and 40-130K characters of raw output per A/B. Record the result here.
- **Waiting on CI** (`gh pr checks --watch` loops) is worth a similar treatment.
- **A `perf_region.py` adapter.** Its counters need `sudo` and a driver marking regions, so the
  harness has none yet.

Still out of scope:

- Non-timing gates. The golden run, tests and lint are each already one command.
- Significance tests (Mann-Whitney and the like). The protocol judges by ranges and agreement
  between passes, and the report adds no p-values.
- Other machines. The profile check refuses them; comparing machines is a different question.

## From the DRY audit

- **The per-rule presets** (the workplan's D2). The preset classes that set a rule each repeat a
  hyperparameter `__init__` and a two-line `_update_rule`. There were 12 at the audit (Adam,
  Momentum, conv Momentum and L2, each in pure Python, numpy and Rust); the presets workplan
  brought them to 27 (`Momentum`, `Adam` and `WeightDecay`, each one-output, multiclass and conv,
  in all three implementations). D2 waited for more presets before choosing a shared form; that
  point has come, and the form is the next DRY pass's to choose.
- **Rerun the audit after each new primitive**, so new copies are caught while they are small:
  `symilar` (pylint's duplicate finder, with `-i --ignore-docstrings --ignore-imports
  --ignore-signatures`) over `indrajala_ml/`, `scripts/` and `tests/`, a search for function names
  defined in more than one module, and the twin classes read side by side. The 2026-10-01 audit
  found 0.29% duplicated at 8 lines or more in the package, and 0.91% at 10 lines or more in the
  tests.
  The rerun of 2026-10-02 (the DRY rerun workplan, #550-#557) found 0.20% at 8 lines or more in
  the package and 0.16% at 10 lines or more in the tests, the copies left having come in with
  residual connections, layer norm and attention.
  The rerun of 2026-10-09, after multi-head attention, found 0.16% at 8 lines or more in the
  package, 0.14% at 10 lines or more in the tests and 0.00% in the scripts, and removed what it
  found in one PR: `AttentionProjections` took the attention layers' shared setup (`_set_up`), the
  gradients' order (`_set_gradients`) and `reset_gradient_accum`, and `tests/helpers.py` took
  `batches` and `learn_in_step` (the 50-step parity loop, 11 copies), `assert_snapshots_close` (5)
  and `assert_learn_and_a_batch_of_one_agree` (3), `test_checkpoint` the restored-checkpoint
  scenario (3) and `test_layer_specs` the README's patch model (2).
- **Shared homes for the next layer kind.** The conv and pool argument checks and output size are
  in `model/specs/window_geometry.py`, one shape walk (`spec_shapes.spec_shapes`) feeds both builders,
  the optimizers share `OptimizerBase` and `ArrayOptimizerBase`, and a layer's optimizer accessors
  are `WeightAndBias`, `GammaAndBeta` or `AttentionProjections` (`array_parameters.py`). A hidden
  layer takes its output-delta refusals, its downstream-as-delta methods and its no-op gradient
  accumulation from `model/specs/hidden_layers.py` (`Hidden`, `DeltaIsDownstream`, `ParameterFree`); a
  pure-Python gamma-and-beta row from `GammaAsWeights`; a token shape from
  `spec_shapes.token_shape`. Tests share `bits`, `split`, `max_relative_gap`, `patching`,
  `randomized`, `learn_in_step` and the `assert_*` scenarios (`tests/helpers.py`), and a scenario
  every implementation runs takes conftest's `implementation` fixture (`tests/model/networks/test_{attention,residual,layer_norm}_network.py`). A new layer
  kind extends these rather than adding a copy. A study takes its cached MNIST
  loading, mean ± sd cells and Markdown tables from `indrajala_ml/studies/common.py`.
- **The linear, conv and batch-norm layers' own output-delta refusals** (the rerun's stage 1).
  They are hand-written, with their own messages, rather than taken from `Hidden`; they were not
  in the rerun's findings. Move them onto the mixin when one of those layers is next touched.
- **The builders' wiring loop** (the rerun's D6). The fork, add and body-first loop is the same in
  `python_layer_builder.py` and `array_layer_builder.py`; sharing it needs callbacks for the fork
  and the add, more indirection than its 12 lines are worth. Revisit if another layer kind that
  links layers arrives.

Still out of scope:

- The crate (the workplan's D1), and scripts and demos (D5): they get their own audit if wanted.
- The numpy and Rust layer twins' numerics (D3): they share identical methods only; their method
  bodies (a numpy expression against one fused Rust call) stay separate by design.
- The Add twins (the rerun's D2): `AddArrayLayer` and `AddRustArrayLayer` have the same method
  bodies, but each backend keeps its own named, typed class, which the tests check.

## From residual connections

- **Conv residual blocks** (D2). A conv block must keep its shape, and conv is 'valid' padding
  only (`conv_layer.py`), so it needs 'same' padding (new geometry, im2col and crate ops in all
  three implementations) and a conv layer with a bias and no activation: a workplan of its own.
- **Projection shortcuts** (D5): `x W_s + F(x)` when a block changes size (He et al. 2015's
  option B), a trained shortcut layer with a second gradient sum and its own parity tests.
- **Zero-initialized block ends** (D7): the body's last layer starting at zero, so each block
  starts as the identity (Goyal et al. 2017's zero-γ; Zhang et al. 2019, Fixup), as an option.
- **The depth study's open question**: with batch norm, the plain network's first-layer gradient
  at initialization grows tenfold from 2 to 16 hidden layers (the residual one's 1.8x), and at 16
  its seeds' standard deviation is 3 points in the first two epochs. Unexplained.

Still out of scope:

- Nested blocks (D6). The design nests without change; only validation refuses it.
- An activation after the add, `relu(x + F(x))` (D3), and a standalone activation layer.

## From the RNG generators workplan

- **Run checkpoints beyond one network's mini-batch run** (`indrajala_ml/training/run_checkpoint.py`):
  resuming mid-epoch, resuming `train_linear_classifier_network` (it returns no run checkpoint),
  and resuming an ensemble's run, whose sub-networks train as separate jobs.

Still out of scope:

- Normal draws, integer draws and broadcast `low`/`high`. Nothing uses them.
- Other bit generators (Philox, SFC64, PCG64DXSM).
- Changing the crate's legacy MT19937 module functions, which mirror `np.random` (the workplan's
  D7).

## From the RNG draw order

From one seed, every pure-Python network with an array twin now builds numpy's weights and draws
numpy's dropout masks by bits ([rng-audit.md](rng-audit.md)), and the parity tests seed both
sides. Left over:

- **The bounds-width networks' draws** (D1 (c)): `BackpropClassifierNetwork`'s presets and
  `LinearClassifierNetwork` scale each node's weights by the input bounds, which no array network
  draws, so there is no order to match. Out of scope unless they gain an array twin.
- **A step-by-step gradient check of the batch-norm conv networks** would need the attention
  tests' `_scales` to give a linear conv layer before batch norm batch norm's scale, as it gives a
  `LinearLayer` (found in stage 2, #649). Batch norm's own gamma and beta gradients can also cancel
  to about 1e-6 there ("conv and dense pairs", seed 19). Nothing checks those networks step by step
  today.
- **Docstring figures measured with the old draw order**, such as `randomize_fan_in_aware`'s
  99.5% training and 96.9% test accuracy on UCI digits, are historical: re-measure one only when it
  is cited as current.

## From layer norm and attention

Multi-head attention and a key size were roadmap step 5, now done; masking and dropout in attention
moved to From multi-head attention. Besides those:

- **Layer norm against batch norm on the dense networks** (the workplan's D11 (b)): the residual
  depth study's batch-norm cells rerun with a flat `LayerNorm` in place of `BatchNorm`.
- **Parity under Adam is per step.** Under `SGD`, `Momentum` and `WeightDecay` the pure-Python
  and Rust patch models are compared with numpy after 50 steps; under `Adam` they are compared
  step by step, each step's gradients from the same parameters
  (`tests/model/networks/test_attention_python_network.py`, `tests/model/networks/test_attention_rust_network.py`). Adam's
  step is steepest where `|g|` is at or under its `ε`, which turns `bk`'s rounding-noise gradient
  into whole steps, so the trajectory itself is sensitive: numpy against numpy with one weight
  nudged by one ulp drifts 2e-8 to 3e-5 in 50 steps, as much as the implementations differ, and
  with `bk`'s gradient zeroed on both sides the gap the bias alone explains falls to 2.4e-12. The
  owner hasn't settled this standard; a 50-step Adam comparison with `bk` held out is the
  alternative.
- **A flat `LayerNorm` after a conv front end** is accepted and normalizes the whole `(h, w, c)`
  output as one token, keeping its shape. Per position (each pixel's channels, as a transformer's
  token) or per channel (group norm) are the other readings; none is used yet.
- **Inert parameters.** `bk` (no effect on `P`) and `bv` (the same as `bo`) are kept to match the
  reference models (D6). A flat `LayerNorm`'s `beta` right before a linear layer and its
  `BatchNorm` is inert too: the batch norm cancels the constant shift, so its gradient is noise.
- **A preset** (D12), if a patch model configuration is worth naming, under From composable
  layers' rule for presets.

Still out of scope:

- A class token (D8 (b)), fixed sin-cos or drawn positions (D7 (b), (c)).
- Conv-then-tokens hybrids.

## From multi-head attention

The workplan settled `Attention(heads, key_size)`, the parameters packed with heads as row blocks
(D2), and every implementation and the crate in three blocks per pass: project, attend, combine
(D4). The patch study's open margin is answered: at 5 seeds attention then FFN beats FFN alone by
0.40 +- 0.30 points, on 4 of 5 seeds (`scripts/multi_head_attention_study.py`).

- **The extension points** (the workplan's D6, D7), each a field with a default, added by the work
  that uses it:

  | later work | where it goes | what it adds |
  | --- | --- | --- |
  | padding masks (causal is done: `Attention(causal=True)`, the sequence task) | attend, forward: an additive mask on `S[i]` before the max shift, as the causal mask's `-inf` | a mask input; backward unchanged (`P_ij = 0` zeroes `dS_ij`) |
  | dropout on the weights | attend: an inverted mask on `P[i]` before `H[i] = P[i] V[i]`, kept for backward | a mask field in the options, drawn by the layer from the network's generator |
  | dropout after the projection | after combine | a token-wise dropout layer, outside attention |
  | a separate value width | project and combine: `Wv` `(h·d_v, d)`, `Wo` `(d, h·d_v)` | a `value_size` field defaulting to `key_size` |
  | grouped- or multi-query attention | project: `Wk`, `Wv` with `g` row blocks; attend: head `i` reads key/value block `i // (h / g)` | a `key_value_heads` field defaulting to `heads` |
  | cross-attention | project: `K`, `V` from a second input | a second input to the layer, which `Sequential` doesn't have: its own design |

  Dropout in attention needs a training and inference switch in a token layer and mask draws in
  all three implementations' orders.
- **GELU** (D5 (c)), ViT's FFN activation, an activation for `Dense`: it needs `erf`, which stable
  Rust lacks (the `tanh` form is a different function), in all three implementations and the crate.
  Planned with SwiGLU as roadmap step 13 ([ffn-activations-workplan.md](ffn-activations-workplan.md)).
- **The study's other arms** (D8 (b), (c)): run on 2026-10-09 (`scripts/patch_geometry_study.py`,
  findings in its docstring). Patch 4 (`T = 49`) costs every arm 0.4 to 1.2 points at 5 epochs;
  the low-rank bound still doesn't measurably bind (+0.42 +- 0.83 for `key_size=32`); heads start
  to help with more or wider tokens (+0.6 over one head at `T = 49` and at `d = 64`); and `d = 64`
  adds 0.13 points to the best model at 3.7 times the parameters. Patch 7 and `d = 32` stay the
  MNIST patch models' trade; the gap to conv (0.9 points) is for the sequence task to revisit, not
  more MNIST geometry.
- **The numpy attention backward's 10% from stage 3.** #606's tier 1 A/B reported the numpy rows
  within noise, but before #610 both sides imported the new tree's package. Rerun on 2026-10-09
  with the fixed script (`ab.py run --script-from`, #616; report on #616), stage 3 made
  `hidden_delta_batch` 10.2% slower (243 to 268 us per call, the stage-1 attention case at batch
  32, consistent over 6 passes); `forward_batch` and `accumulate_gradient_batch` stayed within
  noise. Main's numpy backward is still stage 3's. The cost came with the three-block structure
  (D10), likely in the backward's (N, h, T, d_k) views or its stacked `np.matmul`s. The sequence
  task's mask touched the numpy layer's forward only and left this as it was. Still worth a look:
  a fix must keep the blocks and their bits.

Still out of scope:

- A `TransformerLayer` spec (D5 (b)): a transformer layer is the attention and FFN blocks written
  again; a spec would freeze pre-LN against post-LN, the activation and where dropout goes.
- A class token, relative position biases, and head pruning or per-head analysis (Michel et al.
  2019, "Are Sixteen Heads Really Better than One?").
- A faster multi-head kernel: a crate-tuning item (From the benchmark machine baseline), with its
  own A/B.

## From the sequence task

The workplan settled `Embedding(vocabulary, size)` over token ids (D5), a token-wise softmax output
with the mean of the `T` cross-entropies as the `"sequence"` network shape (D6), and
`Attention(causal=True)`, `-inf` on the future scores before the max shift (D7), in all three
implementations and the crate; four corpora, each in its own dataset repository (D2, D3); fixed
non-overlapping windows of 65 characters, the last 10% held out (D4). The study
(`scripts/sequence_study.py`, D9) found every arm where D9 expected it (README, The sequence task).

- **Generation** (D11 (b)): `scripts/sample_text.py`, a sliding window of the last `T` characters,
  sampling from the output at the last position, `T` forward passes per window's worth of text. The
  fixed-`T` network has no shorter prefix; a sliding window needs none.
- **Longer training.** Neither transformer had converged at 10 epochs: their held-out losses
  still fell by 0.01 to 0.05 bits per character over the last three. More epochs, with a decaying
  rate once the curve flattens (`lr_schedule.py` has linear warmup only), would show where the
  2-layer models stop; the 5-seed sweep took about 2.5 h on `jebel` at 6 workers.
- **Euclid's train/held-out gap.** On Euclid the 2-layer model is 0.49 bits per character lower on
  training windows than held out (1.01 against 1.50), the widest of the four corpora (Herodotus:
  none), and the gap grows with the model (ffn 0.17, 1-layer 0.44). The held-out part is the
  text's last 10%, from Book XII's similar pyramids (XII.8) through Book XIII, and even the
  counted bigram is 0.16 bits worse there, so the gap mixes overfitting with a shift in the text. A held-out set of
  windows drawn from across the text would separate the two.
- **Windows at random offsets** (D4 (b)): every position of every character a target, the usual
  language-model sampler. It needs a sampler in the trainer (an epoch no longer a permutation of
  fixed rows) and its state in run checkpoints. Overlapping fixed windows at stride `T / 2`
  (D4 (c)) are the cheaper middle step, with correlated examples.
- **Padding masks and variable-length windows** (D1 (b)): nothing pads while windows are fixed;
  the mask's place is From multi-head attention's table.
- **A synthetic sequence task** (D2 (b)), copying or reversal from a seed: a known optimal loss,
  and a task only attention over earlier tokens solves. The study's leak arm and the mask's own
  test (a future token can't change an earlier output) already cover what it would test.

Still out of scope:

- A summed per-token loss (D6 (b)): with the mean, a step's size doesn't depend on `T`.
- A separate `TokenOutput` spec (D6 (c)): inside a token part `Dense` already means "per token".
- A network-level causal flag (D7 (b)): the mask is per layer, so an encoder and a causal block
  can mix later.
- Word-level corpora (D2 (d)): a vocabulary of 10,000 or more makes the output layer the whole cost.

## From the benchmark machine baseline

- **Crate tuning for the gaps it found.** Where Rust lags numpy on the i7, or lost ground from
  the laptop, and what to investigate for each, ordered by payoff:
  [i7-9700k-rust-optimization.md](i7-9700k-rust-optimization.md). Each item gets its own crate
  workplan with an A/B on both machines (measurement.md §9).
- **Code placement on the i7** (multi-head attention, stage 5, #609). The multi-head crate's
  attention ops ran up to 14% slower per call on `jebel` with the same instructions per call
  (within 0.3%): in the new build, forward's µops from the legacy decoder (`idq.mite_uops`) tripled
  as uop-cache delivery fell, a placement effect on Coffee Lake (the JCC erratum's 32-byte
  windows), and ops whose code didn't change moved by up to 13% between builds. `pyramidon` showed
  no change. To investigate: `perf record` the hot loop's addresses in both builds, and whether
  `-C llvm-args=-x86-branches-within-32B-boundaries` (or function and loop alignment) makes crate
  A/Bs on the i7 stable; until then, a crate A/B's few-percent rows there may be placement.
- **Rerunning the baseline** after a change that could move every number (kernel, BIOS,
  numpy/OpenBLAS, rustc, the frequency policy): rerun the workplan's stages 4-5 at the same
  tags, then at `main`, and replace [machine_profiles/i7-9700k.md](machine_profiles/i7-9700k.md)'s
  tables, with the reason, in one PR.
- **The laptop's columns** need rerunning only after a software change on the laptop (numpy,
  OpenBLAS, rustc, its kernel). The two machines run the same stack today, and their golden runs
  are byte-identical (D5). A crate change for the i7 is A/B'd on the laptop too, preferably on one
  thread: its default-threading A/As are noisy (a 32% per-pass spread at the median row).

Still out of scope:

- Claiming a cross-machine speedup from the laptop columns: they're context, and only their
  Rust/numpy ratios compare.
- GPU work, BIOS settings beyond cooling and power limits, and kernel boot parameters
  (`isolcpus`, `nohz_full`): the i7's noise didn't need them.

## From the benchmark archive

- **Results from other people's machines,** if the project is used widely: submissions to the
  archive by PR, with a recorded profile, a clean tree and a named commit. The record format
  allows it; reviewing untrusted submissions needs its own plan.
- **A run queue on `jebel`** (several A/Bs queued from `pyramidon`, run back to back), if one run
  at a time under the run lock turns out to be the bottleneck.
- **Crate tuning on CPU attributes** ([i7-9700k-rust-optimization.md](i7-9700k-rust-optimization.md))
  uses the archive's profiles and runs from both machines.

Still out of scope:

- Changing the timing protocol (the verdict rule, the noise rules per machine) except from data,
  as tier 1's pass count was.
- Hosting results anywhere but the archive repository: no dashboards, no release assets.
- Benchmarks in CI: GitHub's runners are shared and noisy, so no timing is ever taken there.
