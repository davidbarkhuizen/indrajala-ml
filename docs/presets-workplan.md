# Workplan: closing the preset gaps

**Status: decisions settled (D1-D4); stages 0-3 done.**

The composable-layers workplan made the Sequential networks build every accepted spec list under
every rule, in all three implementations, but left the preset table (README, Presets) with empty
cells and listed the combinations that have no preset ([next-steps.md](next-steps.md), From
composable layers). This plan gives those combinations presets: named classes, with their own
constructor arguments, that the demos, `ensemble_train.py` and saved files can use by name.

It adds classes only. No layer, op, rule or crate changes: every combination below already trains
through the Sequential networks, on Rust too. Each new preset equals its Sequential network by bits.

## The gaps

A preset is one base network with **one** thing changed: its update rule (`_update_rule`), its
hidden layers (`_hidden_spec`) or its output layer (`_output_spec`). Read that way, the table has
three kinds of hole.

**A. Pure-Python multiclass dense** (6). numpy and Rust have them, pure Python doesn't:

| Change | Proposed name |
| --- | --- |
| cross-entropy loss | `CrossEntropyMultiClassBackpropClassifierNetwork` |
| ReLU hidden layers | `ReLUMultiClassBackpropClassifierNetwork` |
| dropout | `DropoutMultiClassBackpropClassifierNetwork` |
| `Momentum` | `MomentumMultiClassBackpropClassifierNetwork` |
| `Adam` | `AdamMultiClassBackpropClassifierNetwork` |
| `WeightDecay` | `L2RegularizedMultiClassBackpropClassifierNetwork` |

**B. numpy and Rust one-output** (10). Pure Python has them, numpy and Rust don't:

| Change | numpy | Rust |
| --- | --- | --- |
| ReLU hidden layers | `ReLUArrayBackpropClassifierNetwork` | `ReLURustArrayBackpropClassifierNetwork` |
| dropout | `DropoutArrayBackpropClassifierNetwork` | `DropoutRustArrayBackpropClassifierNetwork` |
| `Momentum` | `MomentumArrayBackpropClassifierNetwork` | `MomentumRustArrayBackpropClassifierNetwork` |
| `Adam` | `AdamArrayBackpropClassifierNetwork` | `AdamRustArrayBackpropClassifierNetwork` |
| `WeightDecay` | `L2ArrayBackpropClassifierNetwork` | `L2RustArrayBackpropClassifierNetwork` |

`FanInAwareBackpropClassifierNetwork` is not a gap: the array one-output networks already
initialize fan-in-aware (`ArrayBackpropClassifierNetwork`'s docstring), so the plain array preset
is its counterpart.

**C. Conv** (18). Only `SGD` and `Momentum` have conv presets. The other one-change presets, in all
three implementations (`<Impl>` is empty, `Vectorized` or `RustArray`):

| Change | Proposed name |
| --- | --- |
| `Adam` | `AdamConv<Impl>MultiClassBackpropClassifierNetwork` |
| `WeightDecay` | `L2Conv<Impl>...` (pure Python: `L2RegularizedConv...`, D3) |
| ReLU dense hidden layers | `ReLUConv<Impl>...` |
| dropout on the dense hidden layers | `DropoutConv<Impl>...` |
| cross-entropy loss | `CrossEntropyConv<Impl>...` |
| softmax output, cross-entropy loss | `SoftmaxConv<Impl>...` |

The conv layers themselves are already ReLU (or linear before a `BatchNorm`); "ReLU" here means
the dense hidden layers after the front end, as in the dense presets.

**Not gaps, by design:**

- **Combinations of two or more changes** (ReLU under `Adam`, dropout with softmax, ...). One
  preset per combination grows the class count multiplicatively; `SequentialArrayNetwork` and the
  pure-Python Sequential networks are the API for those.
- **Batch norm, residual blocks, layer norm and patch models.** Their workplans chose to have no
  presets (README, Presets); their spec lists are too varied for a fixed constructor.
- **Ensembles of the new one-output presets.** The ensembles fix their sub-network class
  (`classifier_cls`); a choice of sub-network is its own change.

Total: 34 classes (D1: all three groups).

## What each new preset needs

A preset is about 30 lines: a subclass of its base that sets its hyperparameters and overrides one
hook. The cost is in what registers it:

1. The class, in `indrajala_ml/model/`, with a docstring naming its siblings.
2. A contract entry: `tests/array_network_contract.py`'s `ArrayNetworkSpec` (numpy and Rust, with
   its `equivalent` specs and rule), or the pure-Python counterpart in
   `tests/test_sequential_backprop_network.py`, so it equals its Sequential network by bits.
3. A format-2 fixture in `tests/saved_model_fixtures.py`, written with
   `python -m tests.saved_model_fixtures` (`test_every_saveable_class_has_a_fixture` requires one).
   New presets have no legacy envelope; only format 2.
4. The class-coverage tables: `tests/test_seeded_init_parity.py` and
   `tests/test_prepared_dataset.py` (both fail on an array class they don't list).
5. A golden-run entry in `scripts/golden_training_run.py` (D2).
6. The README's preset table, and the next-steps entry it closes.

Checks to watch:

- **Dropout in a conv network.** The dense dropout presets assert every hidden layer drops out
  (`self.hidden_layers`); a conv network's hidden layers include the front end, so the conv
  dropout preset selects the dense ones only.
- **One-output pure-Python presets initialize by bounds width; the array ones fan-in-aware.** The
  group B presets follow their array base (fan-in-aware, `(layer_sizes, dimension)`), so their
  pure-Python counterparts differ at initialization; parity compares them from the same weights,
  never from the same seed. Their hyperparameters are keyword-only, after the `input_bounds` the
  array one-output networks accept and ignore (`ensemble_train.py` passes it positionally), so a
  required one can follow its default.
- **Pure-Python multiclass presets** take `MultiClassBackpropClassifierNetwork`'s arguments
  (`layer_sizes, dimension, input_bounds, class_count`) plus their hyperparameters.

No stage touches a `learn*` or `classify_rows` path, so none needs an A/B (docs/measurement.md,
§1); group A is pure Python, never timed anyway. Every stage passes the golden run's existing
entries unchanged, then adds its own (D2).

## Decisions

- **D1. Scope: all three groups, 34 classes** (owner, 2026-10-02). The per-class cost is small
  and mechanical, and a full matrix needs no footnotes on empty cells. Rejected: A and B only (16),
  which leaves conv with only `SGD` and `Momentum`, and B and C only (28), which leaves the dense
  rows uneven across implementations.
- **D2. The golden run pins the new presets** (owner, 2026-10-02). Each stage adds its presets to
  `scripts/golden_training_run.py`, one line each. The contract tests compare a preset with its
  Sequential network, which runs the same code, so they can't see a drift both share; today no
  golden entry covers these combinations. A stage first runs `check` against the old file, proving
  the existing entries unchanged, then `record`s the file with its new entries, and its PR says
  both happened. No existing bits move, so this is not a re-recording under docs/measurement.md
  §8. Rejected: leaving the golden run as is.
- **D3. Weight-decay presets keep their implementation's prefix** (owner, 2026-10-02): pure
  Python's `L2Regularized` (`L2RegularizedMultiClassBackpropClassifierNetwork`,
  `L2RegularizedConvMultiClassBackpropClassifierNetwork`), numpy's and Rust's `L2`
  (`L2ArrayBackpropClassifierNetwork`, `L2ConvVectorizedMultiClassBackpropClassifierNetwork`, ...).
  Each class matches its siblings in the same implementation; the existing names can't change
  (composable-layers D1). Rejected: `L2` everywhere (pure Python would then disagree with itself)
  and `WeightDecay` for the new classes (a third spelling).
- **D4. No demos** (owner, 2026-10-02). The demos and `demos/registry.py` stay as they are. No
  demo builds one of these combinations through a Sequential network today (batch-size scaling
  uses batch norm, the residual study residual blocks, and the patch study's conv control changes
  two things, `Adam` and a softmax output), so there is nothing to switch; a new demo makes an
  accuracy claim that needs its own tuning and measurement. Rejected: adding registry demos for
  the conv presets.

## Stages

Each stage is one PR, merged before the next; the golden run, `./cli test` and `./cli lint` pass at
every stage.

0. **This workplan**, with D1-D4 settled.
1. **Group A**: the six pure-Python multiclass dense presets.
2. **Group B**: the ten numpy and Rust one-output presets.
3. **Group C, the update rules**: `Adam` and `WeightDecay` conv presets, all three implementations
   (6).
4. **Group C, the layers**: ReLU, dropout, cross-entropy and softmax conv presets, all three
   implementations (12). Split at dropout if the diff passes about 1,500 lines.
5. **Docs and retirement**: README's preset table and Models count, next-steps' "New
   combinations" list cut to what's left (the multi-change combinations), this workplan retired
   into next-steps.md.

Stages 1-4 are independent of each other.

## After this plan

- Ensembles over the new one-output presets.
- Demos for the new conv presets (D4), each with its own tuned, measured PR.
- Weight decay with momentum or Adam still needs a published form chosen first (next-steps.md).
