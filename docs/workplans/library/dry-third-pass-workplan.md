# Workplan: remove duplicated code, third pass (DRY, 2026-10-30)

**Status: in progress; decisions D1-D10 settled (2026-10-30, delegated to Claude by the owner's
standing rule). Stages 1-5 done; stages 6-8 open.**

The first two passes (`git show e066333:docs/dry-workplan.md`, #506-#515; `git show
88e1815:docs/dry-rerun-workplan.md`, #550-#557) and the rerun after multi-head attention
([next-steps.md](../../next-steps.md), "From the DRY audit") asked for the audit to be rerun after
each new primitive. Rotary positions, attention dropout, sequences, the window-recurrent network
and its studies have merged since, and the per-rule presets' deferred form (the first pass's D2)
has come due. This plan removes what the audit of 2026-10-30 (`main` at d38f8b6) found, the tests
and test helpers included, one PR per stage. It is refactoring: no network, preset, save file or
numeric result changes, so the golden run stays bit-identical at every stage.

## Why

Each copy is a place the next layer kind, rule or study option has to be written again, and a
place where two copies drift apart. Two of the earlier passes' homes have already been bypassed:
`tests/helpers.bits` and `to_numpy` exist, but the crate's tests carry 11 `_bits` and 7 `_numpy`
of their own, and the hidden-layer mixins (`model/specs/hidden_layers.py`) exist, but the four
linear layers still write their refusals by hand.

## How the audit was done

- **A clone scan.** `symilar`, the earlier passes' tool, is not installed in the venv, so the scan
  was a window hash over normalised lines (strings and numbers masked; blank, comment, import and
  bracket-only lines dropped) at 8 and 10 lines, over `indrajala_ml/`, `scripts/`, `tests/`,
  `rust/tests/` and `cli`, now `scripts/measurement/duplicates.py` (stage 1, which also skips a
  window of one line repeated). Lines inside a reported clone, either side counted, at 8 lines:
  package 3.4% (1147 of 34085), scripts 1.8%, tests 4.2%, crate tests 2.0%; at 10 lines: 2.2%,
  1.4%, 2.7%, 1.2%. These are not comparable with the earlier `symilar` percentages (a different
  count).
- **Function names defined in more than one module** (`^def name`), and test helpers defined in
  more than one file.
- **The twin classes read side by side**: the preset classes across the three implementations,
  the two network bases, the shape mixins, the numpy and Rust recurrence layers, the linear
  layers, and the study modules' repeated loops.

## Findings

Line numbers are from d38f8b6.

### In the package

1. **The per-rule presets' `_update_rule` and `hyperparameters`.** 27 classes (Momentum, Adam and
   WeightDecay; one-output, multiclass and conv; pure Python, numpy and Rust) each repeat a
   hyperparameter `hyperparameters` tuple, the attribute assignments and a two-line
   `_update_rule` that rebuilds the rule from them, e.g.
   `networks/numpy/adam_conv_vectorized_multiclass_backprop_classifier_network.py:21-40` against
   its Rust and pure-Python twins. The hyperparameter names are already written once, as the
   rule's dataclass fields (`model/specs/update_rules.py`). The `__init__` signatures differ by
   implementation (the numpy one-output presets take `input_bounds` optional and the
   hyperparameter keyword-only; the pure-Python ones take `input_bounds` required).
2. **The network bases' spec hooks.** `ArrayNetworkBase` (`networks/array_network_base.py:96-122`)
   and `BackpropNetworkBase` (`networks/python/backprop_network_base.py:115-141`) define the same
   `rng` setter, `_window_state_size`, `_hidden_spec`, `_output_spec`, `_dense_specs` and default
   `_update_rule`.
3. **The token-wise output hooks.** `predict_probabilities`, `classify_state`, `_classify_output`
   and `_classify_output_batch` are the same in `SequentialSequenceShape`
   (`specs/array_network_shapes.py:318-331`) and `WindowRecurrentShape`
   (`networks/window_recurrent_array_network.py:405-418`).
4. **The conv networks' argument checks.** The pure-Python conv network's `__init__`
   (`networks/python/conv_multiclass_backprop_classifier_network.py:51-64`) and `ArrayConvShape`'s
   (`specs/array_network_shapes.py:177-190`) validate the class count, the dense sizes and that
   the specs hold a conv layer, then set the same five attributes.
5. **The recurrence layers' schedule setters.** `set_temperatures`, `set_continuous`,
   `set_couplings` and the checking half of `set_commitments` are the same in
   `RecurrenceArrayLayer` (`layers/numpy/recurrent_array_layer.py:531-560`) and
   `RecurrenceRustArrayLayer` (`layers/rust/recurrent_rust_array_layer.py:284-314`); their
   `__init__`s' first lines are the pure-Python `RecurrenceLayer`'s too
   (`layers/python/recurrent_layer.py:417-431`).
6. **The linear layers' hand-written hidden-layer methods** (the rerun's leftover). The numpy and
   Rust linear and linear-conv layers (`layers/numpy/linear_array_layer.py:54-69`,
   `layers/numpy/conv_array_layer.py:211-226`, `layers/rust/linear_rust_array_layer.py:47-60`,
   `layers/rust/conv_rust_array_layer.py:156-169`) each write the two output-delta refusals, the
   single-example refusals and the downstream-as-delta batch method.
7. **The ensemble trainer's job tuples.** `training/ensemble_train.py` passes each Pool job as a
   10- or 12-field positional tuple, unpacked and re-passed field by field in both workers
   (`:98-120`, `:146-178`) and built in four places (`:229-238`, `:276-305`, `:335-350`,
   `:383-395`).
8. **`_git`** is the same subprocess call in `measurement/benchmark_archive.py:35` and
   `scripts/measurement/ab.py:422`, differing only in the error raised.
9. **The study's soft-assignment loop.** `hardness`, `sharpness` and `thermals`
   (`studies/modular_family.py:1965-2050`) each walk the rows' windows, slice each soft factor's
   cell output and compute its distances to the codes and the softmax at its temperature, then
   differ in the one statistic they sum.
10. **The run's identity re-expanded.** `RunConfig.identity` calls `_run_identity`
    (`studies/modular_family.py:2998-3080`), a 30-parameter function that takes back, one by one,
    the fields the frozen `RunConfig` already holds, so each new run option is written in three
    signatures.

### In the tests

11. **The crate tests' own helpers.** `_bits` in 11 files of `rust/tests/` (some `struct.pack`,
    some `.tobytes()`), `_numpy` in 7 and `_array` in several, beside `tests/helpers.py`'s `bits`
    and `to_numpy`, which the crate's suite can't import (it is the crate repository's, and runs
    standalone).
12. **`CarryTable`,** a test double of the carry machine, is the same class in
    `tests/studies/test_certificate_guard.py:17-42` and `tests/studies/test_window_replay.py:72-97`.
13. **The small study settings.** The same 11-field `wr.Settings(...)` is written in six tests of
    `tests/studies/test_window_recurrent_study.py` (`:276`, `:314`, `:364`, `:391`, `:512`,
    `:849`) and in `tests/studies/test_window_stagger.py:145`, some with one field changed.
14. **The program-against-per-layer gradient check.** The same 12-line comparison ends four tests
    of `tests/model/networks/test_window_recurrent_rust_network.py` (`:234`, `:255`, `:276`,
    `:318`).
15. **The prepared-dataset constructors.** `tests/data/test_prepared_dataset.py:82-147` lists each
    numpy class's constructor and then each Rust twin's, the same lambdas twice under the other
    backend's names.
16. **The python and Rust network tests' shared scenarios.** `test_learn_and_a_batch_of_one_agree`
    and the saved-network round trip are the same in `test_rotary_{python,rust}_network.py`
    (`:115-133`, `:125-143`), `test_attention_dropout_{python,rust}_network.py` (`:123-142`,
    `:108-126`) and `test_sequence_{python,rust}_network.py` (`:225-239`, `:172-186`), only the
    implementation changed (the rerun's finding 9, again).
17. **The batch-norm dense and conv network tests** share their finite-difference scenario
    (`test_batch_norm_conv_network.py:101-119`, `test_batch_norm_dense_network.py:111-129`) and
    their set-up (`:28-42` in both).

### Not duplication

- **The tests' `_equivalent` functions** (21 files): each is the test's own statement of what its
  preset is, the oracle the preset is checked against. Deriving it from the class would test the
  class against itself.
- **The test functions named alike across files** (`test_snapshot_and_restore_round_trip` and the
  like): most already run one shared scenario (`tests/array_network_contract.py`,
  `tests/helpers.py`); the names are the per-file entry points.
- **`modular_family_probe.py`'s flag blocks** (`:1105-1295`): the scan's largest hit, but
  different flags of the same shape.
- **`ab.py`'s probe adapters** (`:210-222`, `:311-323`): each adapter's own script and arguments,
  in the protocol's adapter form.
- **The study scripts' `--limit`/`--out` arguments** (`patch_study.py:210`,
  `residual_depth_study.py:232`): two lines each.
- **`tests/test_pcg64.py` against `rust/tests/test_random_pcg64_parity.py`**: two suites' own
  reference checks; each runs standalone.
- **The scripts' `_bits`**: `golden_training_run.py:609` records float.hex strings for its JSON,
  `parity_probe.py:39` the uint64 patterns its ULP distances need; different jobs.
- **The numpy and Rust twins' numerics** and **the Add twins** (the earlier passes' D3 and D2).

## Decisions (settled)

Settled 2026-10-30 by Claude, under the owner's standing delegation of workplan decisions; each
lists the options weighed.

- **D1. Scope: the package, the scripts and every test suite, the crate's tests included.** The
  crate's Rust (`rust/src`) stays out (the first pass's D1). Pros: the user asked for the tests and
  helpers; the crate's tests are where the helpers multiplied. Cons: the crate's stage costs a
  crate PR and a "Bump rust/" PR. Rejected: the package alone (leaves findings 11-17).
- **D2. The golden run stays bit-identical; no stage gets an A/B.** Every stage moves identical
  bodies or replaces a tuple by a named one; the precedent is the rerun's D9 (a method's defining
  class costs nothing measurable). Each PR says so. Rejected: tier 1 A/Bs for stages 2 and 3
  (hours of `jebel` for a method lookup).
- **D3. The presets share a `RulePreset` mixin (finding 1, the first pass's D2).** It holds
  `rule: ClassVar[type]`, derives `hyperparameters` from the rule's dataclass fields, and defines
  one `_update_rule`, `self.rule(**{name: getattr(self, name) ...})`. Each preset keeps its own
  typed `__init__`. Pros: the hyperparameters are named once, in the rule; a new rule needs no
  `_update_rule`; signatures, class names and save files unchanged. Cons: the 27 `__init__`s stay.
  Rejected: (a) one `__init__` per rule and topology shared by the three implementations, which
  would change public signatures (finding 1's differences); (b) presets generated by a factory,
  which keeps the names but loses pyright's view of each class; (c) deleting the presets for
  `Sequential*` networks, which breaks saved files' and demos' class names, an API decision
  beyond a DRY pass.
- **D4. A numpy-free `SpecHooks` mixin in `model/specs/` holds the shared hooks (finding 2)**,
  listed by both bases. Rejected: one base inheriting the other (the pure-Python base has no
  backend).
- **D5. A `TokenWiseOutput[A]` mixin in `specs/array_network_shapes.py` (finding 3)**, listed by
  both shapes. Rejected: `WindowRecurrentShape` extending `SequentialSequenceShape` (different
  `__init__` and format-2 shape).
- **D6. One `conv_arguments` check in `specs/array_network_shapes.py` (finding 4)**, called by
  both `__init__`s, returning nothing and setting the attributes on the network. The pure-Python
  conv network sets `dimension` later, as now.
- **D7. A `RecurrenceSchedule` mixin in `specs/window_recurrence.py` (finding 5)** with the three
  setters; `set_commitments` and `set_straight` stay per backend (each adds its own work after the
  shared check). The pure-Python layer has no setters and stays as it is.
- **D8. The linear layers take `Hidden` and a new `BatchOnly` mixin (finding 6)**:
  `compute_hidden_delta` and `downstream` refuse a single example,
  `compute_hidden_delta_batch` takes the next layer's downstream. The hand-written refusal
  messages give way to `Hidden`'s generic one (no test matches them, as the rerun's D3 found).
- **D9. Named jobs and one `_git` (findings 7 and 8).** A frozen `_Job` NamedTuple (it pickles
  for the Pool) replaces the positional tuples; `benchmark_archive._git` takes the error to raise
  and `ab.py` imports it. Findings 9 and 10: one `_soft_assignments` generator (each factor's
  distances and weights per window) feeds the three readings, and `RunConfig.identity` builds the
  identity from its own fields, `_run_identity` removed; the identity dict's keys and values stay
  exactly as before, which a test checks against a fixed config.
- **D10. The tests share through the homes the earlier passes set up (findings 11-17).** The
  crate's tests get one `rust/tests/helpers.py` (`bits` as `.tobytes()` of float64, so 0.0 and
  -0.0 differ, as every copy intends; `to_numpy`, `to_array`); `CarryTable` moves to `tests/studies/doubles.py`; the study tests take a `small_settings(**changes)`
  helper; the gradient check becomes `assert_program_is_per_layer`; the prepared-dataset
  constructors are written once per topology and mapped to both backends' names; the python and
  Rust scenarios of finding 16 run once, parametrized by implementation, in
  `test_{rotary,attention_dropout,sequence}_network.py` (the rerun's D8); the batch-norm dense and
  conv tests share their scenario through `tests/helpers.py`. Each PR lists the tests collected
  before and after, and every old test maps to a new one.

## Stages

Every stage is one PR and passes `./cli lint`, `./cli test` and
`.venv/bin/python scripts/measurement/golden_training_run.py check data/refactoring/golden_run.json`
(bit-identical). No public class, module or function a demo, test or saved file uses is renamed or
deleted. No stage gets an A/B (D2).

### Stage 1: the audit as a repo tool (done)

- `scripts/measurement/duplicates.py`: the scan above (stdlib only), reporting clone pairs and the
  share of lines in clones per tree, with tests. next-steps.md's "From the DRY audit" names it
  as the rerun's method, with this audit's numbers as the baseline.

### Stage 2: the shared hooks and mixins (findings 2-6, D4-D8) (done)

- `model/specs/spec_hooks.py`'s `SpecHooks`; `TokenWiseOutput` and `ConvArguments` in
  `specs/array_network_shapes.py` (`ConvArguments` a typed mixin rather than D6's function, so
  pyright sees the attributes it sets); `RecurrenceSchedule` in `specs/window_recurrence.py`;
  `BatchOnly` in `specs/hidden_layers.py`, taken by the batch-norm layers too (next-steps.md's
  leftover named them with the linear ones), their refusals now `Hidden`'s message.

### Stage 3: the per-rule presets (finding 1, D3) (done)

- `RulePreset` in `model/specs/spec_hooks.py`, on the 27 presets. Found while doing it, and done
  here: the Sequential and window-recurrent networks' `_update_rule`, `return self.update_rule`,
  written in 8 classes (3 shapes and 3 pure-Python Sequential networks, both window-recurrent
  ones), now one `GivenRule` mixin beside it.

### Stage 4: named jobs, `_git` and the study's loops (findings 7-10, D9) (done)

- The ensemble trainer's jobs are frozen dataclasses (`_Training`, shared, and `_DecodedJob` or
  `_IndexedJob`); `RunConfig.identity` builds the identity itself, pinned by
  `test_a_runs_identity_names_its_defaults_alone_and_every_other_setting`, captured from the old
  code before the change; `_soft_assignments` feeds the three readings.
- **`_git` (finding 8) stays as it is,** an amendment to D9 found doing it: `ab.py`'s raises
  `AbError` with stderr's lines, which its callers and tests read, against the archive's
  `ArchiveError` with a message. Sharing it would need an error factory for four lines.

### Stage 5: the test helpers (findings 12-15, 17, D10) (done)

- `tests/studies/helpers.py` (D10's `doubles.py`, named as `tests/helpers.py`): `CarryTable` and
  `TINY`, the study tests' smallest settings, which test_window_recurrent_study.py already had at
  its foot; six of the seven copies were it, the seventh `TINY` at two epochs.
- The four schedules' program-against-per-layer tests are one test over `SCHEDULES`; the
  prepared-dataset constructors are written for the numpy classes and the Rust twins derived by
  name (`_rust_twin`); `assert_a_checkpoint_resumes_training_by_bits` in `tests/helpers.py` serves
  four batch-norm files (the pure-Python dense and conv ones too, not only finding 17's pair).
  Tests collected in the changed files before and after: 185 (studies and window-recurrent), 390
  (prepared dataset), 371 (batch norm).

### Stage 6: the per-implementation network scenarios (finding 16, D10)

### Stage 7: the crate's test helpers (finding 11, D10)

- In `indrajala-math-rust` first, then a "Bump rust/" PR here.

### Stage 8: rerun the audit; retire this plan

- Run stage 1's tool, record the numbers in next-steps.md, move any leftovers there, and delete
  this workplan.
