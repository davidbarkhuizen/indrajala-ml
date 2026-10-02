# Next steps: work left over from completed workplans

A workplan is deleted once its last stage merges, and never before: only when every stage and
decision is resolved and only future work is left. Whatever it left open (its "After this plan"
list, and the parts of its "Out of scope" that still bind later work) moves here. A workplan still
in progress keeps its own list: [pypi-release-workplan.md](pypi-release-workplan.md). The order of
the next ML primitives is in [primitives-roadmap.md](primitives-roadmap.md).

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

The optimization docs (the Rust-against-numpy baseline, and the implemented, rejected and
candidate optimizations) were retired the same way: `git show 0a04977:docs/optimizations.md` and
`git show 0a04977:docs/optimizations/<name>.md`.

What they built is documented in the README (Models, Saving and loading, Update rules, Batch
normalization, Residual connections, Layer norm and attention), [measurement.md](measurement.md)
and [rng-audit.md](rng-audit.md). The batch-size studies' findings are in
`indrajala_ml/batch_size_scaling.py`'s docstring, the depth study's in
`scripts/residual_depth_study.py`'s, the patch-attention study's in
`scripts/patch_attention_study.py`'s.

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
  its pure-Python reference (`tests/test_sequential_array_network.py`). Giving any other one a
  preset, or using it in a demo, is a behaviour change, not refactoring: its own PR, with
  hand-computed tests and parity tests in all three implementations.
- **Still refused by the specs:** dropout on a ReLU layer (the dropout op is fused with the
  sigmoid), a conv or pool layer after a dense one (the fused hidden delta reads the next layer's
  `W`), and one rule per layer.
- **Weight decay with momentum or Adam** needs a published form chosen and cited first (README,
  Update rules).
- **Ensembles over the new one-output presets** (the presets workplan). The ensembles fix their
  sub-network class (`classifier_cls`); a choice of sub-network is its own change.
- **Demos for the new conv presets** (the presets workplan's D4), each with its own tuned,
  measured PR: a demo makes an accuracy claim.

Still out of scope, for later workplans too (batch norm's included):

- Separate activation layers, which would add a crossing per layer on Rust. Activations stay fused.
- Parameter groups, per-layer learning rates, and schedulers beyond today's `lr_schedule.py`.
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
- **Shared homes for the next layer kind.** The conv and pool argument checks and output size are
  in `model/window_geometry.py`, one shape walk (`layer_specs.spec_shapes`) feeds both builders,
  the optimizers share `OptimizerBase` and `ArrayOptimizerBase`, and a layer's optimizer accessors
  are `WeightAndBias`, `GammaAndBeta` or `AttentionProjections` (`array_parameters.py`). A hidden
  layer takes its output-delta refusals, its downstream-as-delta methods and its no-op gradient
  accumulation from `model/hidden_layers.py` (`Hidden`, `DeltaIsDownstream`, `ParameterFree`); a
  pure-Python gamma-and-beta row from `GammaAsWeights`; a token shape from
  `layer_specs.token_shape`. Tests share `bits`, `split`, `max_relative_gap`, `patching` and
  `randomized` (`tests/helpers.py`), and a scenario every implementation runs takes conftest's
  `implementation` fixture (`tests/test_{attention,residual,layer_norm}_network.py`). A new layer
  kind extends these rather than adding a copy.
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

- **Matching the pure-Python networks with the array networks from one seed.** All three draw
  from one PCG64 stream family, but the per-node networks draw weights node by node and one
  dropout draw per node, so the same seed gives other values. Until the draw order matches, the
  per-node dropout reference is compared with the array networks only at eval
  ([rng-audit.md](rng-audit.md), Open work).
- **Run checkpoints beyond one network's mini-batch run** (`indrajala_ml/run_checkpoint.py`):
  resuming mid-epoch, resuming `train_linear_classifier_network` (it returns no run checkpoint),
  and resuming an ensemble's run, whose sub-networks train as separate jobs.

Still out of scope:

- Normal draws, integer draws and broadcast `low`/`high`. Nothing uses them.
- Other bit generators (Philox, SFC64, PCG64DXSM).
- Changing the crate's legacy MT19937 module functions, which mirror `np.random` (the workplan's
  D7).

## From layer norm and attention

Multi-head attention, a key size other than the token size, masking and dropout in attention are
roadmap step 5 ([primitives-roadmap.md](primitives-roadmap.md)). Besides those:

- **Layer norm against batch norm on the dense networks** (the workplan's D11 (b)): the residual
  depth study's batch-norm cells rerun with a flat `LayerNorm` in place of `BatchNorm`.
- **The patch study's open margin.** Attention then FFN beat FFN alone by 0.4 points, less than
  the FFN arm's standard deviation over 3 seeds, and the FFN arm led until epoch 3. More seeds or
  epochs would show whether attention's gain is real at this size.
- **Parity under Adam is per step.** Under `SGD`, `Momentum` and `WeightDecay` the pure-Python
  and Rust patch models are compared with numpy after 50 steps; under `Adam` they are compared
  step by step, each step's gradients from the same parameters
  (`tests/test_attention_python_network.py`, `tests/test_attention_rust_network.py`). Adam's
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
- Sequence data (text) and its loading; conv-then-tokens hybrids.
