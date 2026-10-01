# Next steps: work left over from completed workplans

A workplan is deleted once its last stage merges. Whatever it left open (its "After this plan"
list, and the parts of its "Out of scope" that still bind later work) moves here. A workplan still
in progress keeps its own list: [pypi-release-workplan.md](pypi-release-workplan.md). The order
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

What they built is documented in the README (Models, Saving and loading, Update rules, Batch
normalization) and [measurement.md](measurement.md).

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
- **Saving RNG state** so a dropout run resumes exactly. This waits on explicit generator objects
  ([rng-audit.md](rng-audit.md), Open work).

Still out of scope, for later workplans too (batch norm's included):

- Separate activation layers, which would add a crossing per layer on Rust. Activations stay fused.
- Parameter groups, per-layer learning rates, and schedulers beyond today's `lr_schedule.py`.
- Deleting or renaming any network class (the workplan's D1).

## From batch norm

- **The study rerun.** Rerun the conv batch-size cells that failed (B = 512, momentum 0.0 and
  0.9) with batch norm, and with ghost groups of 32. That is an experiment with its own plan, and
  the next step in [primitives-roadmap.md](primitives-roadmap.md).
- **Batch norm under dropout or after pool**, if a use appears. Neither passes
  `validate_layer_specs` today.
- **Folding batch norm into the preceding weights** for inference: a speed change, measured.

Still out of scope:

- Layer norm, group norm and instance norm. They normalize within an example, a different layer.
- Synchronized statistics across workers or processes.

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
