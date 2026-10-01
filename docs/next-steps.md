# Next steps: work left over from completed workplans

A workplan is deleted once its last stage merges. Whatever it left open (its "After this plan"
list, and the parts of its "Out of scope" that still bind later work) moves here. A workplan still
in progress keeps its own list: [pypi-release-workplan.md](pypi-release-workplan.md) and
[rng-generators-workplan.md](rng-generators-workplan.md). The order
of the next ML primitives is in [primitives-roadmap.md](primitives-roadmap.md).

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

What they built is documented in the README (Models, Saving and loading, Update rules, Batch
normalization) and [measurement.md](measurement.md). The batch-size studies' findings are in
`indrajala_ml/batch_size_scaling.py`'s docstring.

## From composable layers

- **New combinations.** The Sequential networks build every spec list `validate_layer_specs`
  accepts under every rule, in all three implementations. These combinations have no preset:
  - `Adam` and `WeightDecay` with conv layers (numerics that exist, applied to conv weights);
  - ReLU hidden layers under `Momentum`, `Adam` or `WeightDecay`, and in a conv network;
  - dropout under a rule other than `SGD`, and in a conv network;
  - softmax or cross-entropy output layers after conv, or under a rule other than `SGD`;
  - on pure Python, the multiclass dense presets that exist only as numpy and Rust (cross-entropy,
    ReLU, dropout, momentum, Adam, weight decay), and on numpy and Rust, the one-output presets
    that exist only in pure Python (ReLU, dropout, momentum, Adam, weight decay);
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

Still out of scope, for later workplans too (batch norm's included):

- Separate activation layers, which would add a crossing per layer on Rust. Activations stay fused.
- Parameter groups, per-layer learning rates, and schedulers beyond today's `lr_schedule.py`.
- Deleting or renaming any network class (the workplan's D1).

## From batch norm

- **Batch norm under dropout or after pool**, if a use appears. Neither passes
  `validate_layer_specs` today.
- **Folding batch norm into the preceding weights** for inference: a speed change, measured.

Still out of scope:

- Layer norm, group norm and instance norm. They normalize within an example, a different layer.
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

- **The per-rule presets** (the workplan's D2). The 12 preset classes that set a rule (Adam,
  Momentum, conv Momentum and L2, each in pure Python, numpy and Rust) each repeat a
  hyperparameter `__init__` and a two-line `_update_rule`. They stay as they are until more
  presets are added, when the better shared form should be clearer.
- **Rerun the audit after each new primitive**, so new copies are caught while they are small:
  `symilar` (pylint's duplicate finder, with `-i --ignore-docstrings --ignore-imports
  --ignore-signatures`) over `indrajala_ml/`, `scripts/` and `tests/`, a search for function names
  defined in more than one module, and the twin classes read side by side. The 2026-10-01 audit
  found 0.29% duplicated at 8 lines or more in the package, and 0.91% at 10 lines or more in the
  tests.
- **Shared homes for the next layer kind.** The conv and pool argument checks and output size are
  in `model/window_geometry.py`, one shape walk (`layer_specs.spec_shapes`) feeds both builders,
  the optimizers share `OptimizerBase` and `ArrayOptimizerBase`, and a layer's optimizer accessors
  are `WeightAndBias` or `GammaAndBeta` (`array_parameters.py`). A new layer kind extends these
  rather than adding a copy.

Still out of scope:

- The crate (the workplan's D1), and scripts and demos (D5): they get their own audit if wanted.
- The numpy and Rust layer twins' numerics (D3): they share identical methods only; their method
  bodies (a numpy expression against one fused Rust call) stay separate by design.
