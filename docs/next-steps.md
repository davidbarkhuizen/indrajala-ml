# Next steps: work to be done

**The goal all of it serves:** [machine-discovery.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/research-notes/machine-discovery.md), discovering finite-state
machines from behaviour and known structure, each discovery proved; its route in stages A-F.

This file lists work that no open workplan holds, most of it left by retired workplans, with enough
of the reasoning to take it up. When an item is done, it is deleted here; the record of what was
built is the README, the study scripts' docstrings, and the retired plans
([workplans/retired.md](workplans/retired.md) says how to read each from git history). Nothing
listed here is excluded except GPU backends, for the moment (the owner, 2026-10-12): "not planned"
means only that no workplan has taken it up. An item with more than one stage gets a workplan
before any code.

The open workplans keep their own lists: this repository's are indexed in
[workplans/README.md](workplans/README.md) (the parked `jebel` plan holds the crate's remaining speed
work), the research plans in indrajala-research's
[workplans/README.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/workplans/README.md).
Of the research plans, the general algebras, the factorisation machine and the adders of any length
have been paused since 2026-10-20; each waits for a decision to resume or retire (the programme
audit's F29).

Two parts: **1. Infrastructure**, the common library code (indrajala-ml's package, the crate, the
measurement and study tooling, releases), in an order of implementation; **2. Topic work**, the
studies and research questions, grouped by topic.

## 1. Infrastructure: the common library code, in order

### How the order was made

The items below form a dependency graph: an edge where one item needs another built first (the
"needs" on each item), or where a guard must exist before the change it guards. The numbered list
is a topological sort of that graph. Most items have no edge between them; those ties are broken,
in order, by: guards and owner decisions that gate other work first; then what unblocks the most
later items; then what a current research plan is likely to need; then cost, cheapest first. The
graph's edges:

- a cross-attention benchmark (2) before batched cross-attention steps (33);
- the harness's worker shutdown (3) and the sweep's resume (4) before draining a study (5);
- 0.2.0 (8) before `cargo semver-checks` (9); the presets' signatures (20) before 1.0 (52), and 1.0
  before the demos depend on the package (53);
- a standalone activation layer (10) before an activation after the add (11) and dropout on ReLU (12);
- 'same' padding and a bias-only conv layer (13) before conv residual blocks (14);
- the new layers and per-layer update rules (10-17) before presets for combinations (19);
- a separate value width and padding masks (23, 24), and the activation layer (10), before a
  `TransformerLayer` spec (25);
- the batched crate decode step (32) before batched cross-attention steps (33);
- into topic work: the parked length-free shapes (27) before running an adder at another length,
  ALiBi (28) for a length study, a BPE tokenizer (35) before word-level corpora and the encoder's
  pretraining, an activation accessor (49) before the adders' probes;
- from research: M2's recipe before array-window golden entries (39), M5's probes before the
  state's other placements (38) (indrajala-research's machine-policy plan).

### Guards and owner decisions that gate later work

2. **A benchmark for cross-attention.** `attention_timing` times self-attention only; a change to
   the `cross_attention_*` ops has nothing to A/B against until one is added (measurement.md).
3. **Stopping a study stops its workers.** A study's forkserver workers don't hold the lock
   (`machine_lock`'s docstring): a SIGTERM to the study's process alone frees the lock while each
   worker trains on to the end of the run it holds, and only then fails to report it. Today the
   rule is to kill the whole process group. A fix: the harness traps SIGTERM and terminates its
   pool before it exits (or each worker asks for `PR_SET_PDEATHSIG`), with a test that the lock
   and the workers end together.
4. **A resume for `run_parameter_sweep`** (the patch, sequence and other `*_study.py` sweeps); the
   addition and window-recurrent harnesses resume already.
5. **Draining a study for an A/B** (the study-runs plan's D4 (b) and D5; needs 3 and 4): `ab.py run
   --drain` stops a resumable study at its next gap, runs the A/B and leaves the study to be
   resumed. Today an A/B is refused and waits for the study to end (D4 (a)). A variant has `ab.py`
   resume the study itself, from commands read from the lock's record.
6. **The owner's standard for parity under Adam.** Under `SGD`, `Momentum` and `WeightDecay` the
   pure-Python and Rust patch models are compared with numpy after 50 steps; under `Adam` step by
   step, each step's gradients from the same parameters
   (`tests/model/networks/test_attention_{python,rust}_network.py`). Adam's step is steepest where
   `|g|` is at or under its `ε`, which turns `bk`'s rounding-noise gradient into whole steps: numpy
   against numpy with one weight nudged by one ulp drifts 2e-8 to 3e-5 in 50 steps, as much as the
   implementations differ, and with `bk`'s gradient zeroed on both sides the gap falls to 2.4e-12.
   The alternative: a 50-step Adam comparison with `bk` held out. (`bk`, with no effect on `P`, and
   `bv`, the same as `bo`, are kept to match the reference models; a flat `LayerNorm`'s `beta`
   right before a linear layer and its `BatchNorm` is inert too.)
7. **`Dropout(0.0)`: refuse, skip or keep drawing.** The token layer draws its mask at 0, as the
   dense dropout layer does; `Attention(dropout=0.0)` draws nothing, keeping a model without
   dropout on its old bits. Refusing it in validation, or skipping the draw at 0 (which changes its
   draws; no golden entry has one at 0 today), is the owner's call.

### The release

8. **0.2.0, the next crates.io and PyPI release** (the crate README's Releasing section): a breaking
   one, with the crate structure's renames and no aliases (its D4; the map is in the crate's
   `pypi-pre-release-notes.md`), the ReLU ops gone into the activation family (`layer_relu_*`,
   `array_relu_mask`), and the crate performance work. It first exercises crates.io's trusted
   publishing. A failed rehearsal is re-run when its fix is outside the workflow; a fix to
   `release.yml` itself needs a new tag, and the verify job accepts only `testpypi-v<version>`
   (accept a suffix, `testpypi-v0.2.0-2`, if that recurs). A rehearsal tag is never moved.
9. **`cargo semver-checks` in CI** (needs 8: a second release to check against).

### Layers and network structure

10. **A standalone activation layer** (composable layers; residual connections' D3): activations
    are fused into `Dense` and `Conv` today, and a separate layer adds a crossing per layer on Rust,
    so it is an option beside the fused ones, not a replacement.
11. **An activation after the add,** `relu(x + F(x))` (residual connections' D3; needs 10).
12. **Dropout on a ReLU layer** (needs 10, or a ReLU variant of the fused dropout op): refused
    today because the dropout op is fused with the sigmoid.
13. **'Same' padding for conv, and a conv layer with a bias and no activation:** new geometry,
    im2col and crate ops in all three implementations. Conv is 'valid' only (`conv_layer.py`).
14. **Conv residual blocks** (residual connections' D2; needs 13): a conv block must keep its
    shape. A workplan of its own.
15. **Projection shortcuts** (residual D5): `x W_s + F(x)` when a block changes size (He et al.
    2015's option B), a trained shortcut layer with a second gradient sum and its own parity tests.
    Beside it, two options: **zero-initialised block ends** (D7; Goyal et al. 2017's zero-γ, Zhang
    et al. 2019's Fixup), and **nested blocks** (D6; the design nests without change, only
    validation refuses it).
16. **The other refused layer orders:** a conv or pool layer after a dense one (the fused hidden
    delta reads the next layer's `W`); batch norm under dropout or after pool, if a use appears.
17. **Per-layer update rules and parameter groups:** one rule a network today; the optimizer's
    `learning_rate_scales` covers the codebook's rate only.
18. **Weight decay with momentum,** once a published form is chosen and cited (README, Update
    rules); for Adam it is AdamW.
19. **Presets for combinations** (after 10-17, so the set is whole). The Sequential networks build
    every spec list `validate_layer_specs` accepts, and every one-change combination has a preset;
    two or more changes at once (ReLU, dropout, cross-entropy or softmax under `Momentum`, `Adam` or
    `WeightDecay`, ...) and batch norm under every rule have none: the presets workplan rejected one
    class per combination, and the Sequential networks are the API for them. Giving one a preset is
    a behaviour change: its own PR, with hand-computed and parity tests in all three. One, conv then
    pool, ReLU, dropout and a softmax output under `Adam`, is trained against its pure-Python
    reference already. With it: **a choice of sub-network for ensembles** (they fix
    `classifier_cls`), and **a patch model preset** (layer norm's D12) if a configuration is worth
    naming.
20. **The presets' typed `__init__`s** (the DRY third pass's D3): each preset keeps its own
    signature, which differs by implementation (numpy's one-output presets take `input_bounds`
    optional and their hyperparameter keyword-only; pure Python's `input_bounds` required). One
    `__init__` per rule and topology changes public signatures: an API decision, for 1.0 (52).
21. **Folding batch norm into the preceding weights** for inference: a speed change, measured.
22. **Normalisation within an example:** group norm and instance norm, and a flat `LayerNorm` after
    a conv front end read per position (each pixel's channels) or per channel; today it normalises
    the whole `(h, w, c)` output as one token. Not planned: synchronised batch statistics across
    workers.
23. **A separate value width in attention:** `Wv` `(h·d_v, d)`, `Wo` `(d, h·d_v)`, a `value_size`
    field defaulting to `key_size` (multi-head attention's D6, D7).
24. **Padding masks beyond segments:** an additive mask on `S[i]` in attend before the max shift,
    as the causal mask's `-inf`; backward unchanged (`P_ij = 0` zeroes `dS_ij`). Packed and padded
    segments are `segments` already.
25. **A `TransformerLayer` spec** (multi-head D5 (b); after 10, 23, 24): today a transformer layer is
    the attention and FFN blocks written out; a spec freezes pre-LN against post-LN, the activation
    and where dropout goes. Not planned, beside it: a class token, fixed sin-cos or drawn positions,
    conv-then-tokens hybrids, relative position biases, head pruning.

### Training and run state

26. **Run state:** record `clip_norm` in the run file (version 3), so a resumed run under another
    clip is refused as one under another schedule is; run checkpoints that resume mid-epoch,
    resume `train_linear_classifier_network` (which returns no run checkpoint), and resume an
    ensemble's run, whose sub-networks train as separate jobs.

### Attention, positions and context

27. **Length-free token networks** (length generalisation's P1, parked): stage 3 (P1a: `(None,)`
    token ids in the shape walk, `AnyInputShape`, format 2's `{"dimension": null}`) was built and
    passed but never opened as a PR: the local branch `length-free-shapes` (`bd13b16`). Loading
    trained weights into a network of another `T` needs it.
28. **ALiBi or T5's relative biases,** beside rotary (rotary's D1 (c)): `positions` is a string, so
    either joins without a new spec. An additive distance bias is the scheme that doesn't fail at
    long range by construction (length generalisation's P2). Not planned: a `Position()` and rotary
    in one token part, partial or interleaved rotation.
29. **The numpy attention backward's 10%** (multi-head stage 3): `hidden_delta_batch` 243 to 268 us
    a call at batch 32, consistent over 6 passes (#616), likely the backward's (N, h, T, d_k) views
    or stacked `np.matmul`s. A fix must keep the three blocks and their bits.
30. **Long context:** per-segment (varlen) tiling, if bitwise position independence is needed
    (context length's D3 refinement); sparse or sliding-window attention, mixed precision and
    sequence parallelism, each when a context past 1,024 or a larger model needs it (its D10).
31. **A smaller key/value cache:** multi-head latent attention (DeepSeek-V2: above multi-head at
    14% and 4% of its cache, its Table 9) and cache quantization, each its own step if a cache ever
    binds (grouped-query attention's D8). Beside them, not planned: drawing only the unmasked
    attention-dropout entries under a causal mask (about half the draws; changes every dropout
    golden entry, worth it only if the draws show in a profile).

### Decoding and tokens

32. **A batched crate decode step** (K13): Rust's batched decode runs `attention_step` once a row,
    each row's cache its current paragraph's keys (the segments workplan, D8 (b)); a crate step
    over an `(N, c)` cache with the keys' segments, numpy's form (D8 (a)), would batch it and give
    the two one cache form. No epoch reaches it: on `jebel` (2026-11-04) Rust reads 20,173
    characters a second at batch 8 and 20,806 at 32, flat, against numpy's 7,770 and 13,652.
33. **Batched cross-attention decoding** (needs 2 and 32): each step attends from the cached keys
    and values in one call a pass (`cross_attention_attend_batch`), but the decoder's
    self-attention steps row by row on Rust and in pure Python.
34. **Prefill and positions past `T`:** a band-masked prefill in chunks of `T` for a prompt longer
    than `T` (one exact pass per chunk; today it prefills `T` and then steps); a learned `Position`
    past `T` falls back to a full pass per token (exact, uncached; rotary models slide). And
    **numpy's cached step** is overhead-bound (5.1x uncached against Rust's 12.1x): a fused step,
    if numpy's decode ever needs to be fast.
35. **A BPE tokenizer** (the tokenizer file's `kind`; over both scripts for translation, the
    cross-attention plan's D10).
36. **Decoding options:** beam search (on `prefill_batch` and `step_batch`), repetition penalties,
    stop sequences.

### The array window

37. **A supervised per-step output on an array window** (array window D3 (b)): a softmax output
    with a label per step; only the given loss is accepted now. A task that needs labels gets the
    target layout and its loss weighting then. And the **pure-Python array window** (D8): refused
    by name; a Python conv front end has no user.
38. **The state's other placements** (D2 (b): only before the output layer; (c): extra planes into
    the first conv), after M5's probes (indrajala-research's machine-policy plan).
39. **Golden entries for an array window,** when its bits must be watched across refactors (after
    M2's recipe settles); today its tests pin it (central differences on numpy, numpy against Rust
    within rounding, the program against the per-layer path by bits).
40. **The program's ops for the other array layers** (D7): a residual block, a LayerNorm,
    a sigmoid dense layer, a dense layer with dropout or a FiLM StateInput runs the per-layer path
    (correct, slower). Each gets an op, checked by bits against the per-layer path, when a window
    needs it.

### The crate's speed (the parked plan and its leftovers)

41. **The [parked `jebel` plan](workplans/infrastructure/jebel-parked-workplan.md)** when `jebel`
    is healthy: the threshold re-sweep, the column split, new threaded sites (the window program
    among them, P4), placement (P8: loops aligned to 32 bytes on x86, `litshe`'s cheaper
    pin) and the closing baseline.
42. **A transpose for x86 that doesn't lose to 4K aliasing:** the AVX2 kernel lost up to 52% where
    the output's rows lie a multiple of 4 KB apart (512 rows). One that gathers a block into a
    contiguous buffer before writing it out is the candidate; the measured kernels are on the
    crate's `avx2-transpose` branch.
43. **K19, musl's golden run:** CI builds and tests the crate on musl, not the committed reference;
    a check to add if musl becomes a target. **K20,** numpy's expressions (the reference path's own
    speed).
44. **K22, checkpointing the window's forward** (the window program's D5 (c)): keep every
    `r`-th window's forward and recompute the rest, if a study trains windows long enough to pass
    the memory budget (2.9 MiB at E2b''s cell, 21.7 MiB at the fork probe's, against 2 GiB).
45. **The attention window's kernels:** the program gains least on it (14% forward, 31% training),
    as its crate share is larger; the next gain is in the attention kernels, not the loop.
46. **K18, a bits-changing kernel** (fused or reassociated arithmetic, in the window program too):
    the owner's approval and a re-recorded golden run (measurement.md, §8).
47. **K17, SVE or AVX-512 kernels** (`exp`, the transpose, the products) in the canonical order
    (aarch64's D5), when a machine with them is in hand.

### Measurement and study tooling

48. **The harnesses' timing:** an end-time estimate in `status` (the rate over the last hour and a
    range, for the addition and window-recurrent harnesses; a sweep's average rate overstates its
    slower later ladders); and the addition harness's split timing fields (`fit_seconds`,
    `screen_seconds`, `check_seconds`: its `train_seconds` includes the whole catalogue on a
    passing run, so seconds an epoch read 4-8 times the true 2.5-3 s).
49. **An accessor for a layer's activations on a batch** (the addition study's Excluded; it may be
    small).
50. **A `perf_region.py` adapter for `ab.py`:** its counters need `sudo` and a driver marking
    regions.
51. **Off-machine results and queues, when needed:** archiving studies' raw results in
    `indrajala-benchmarks` (widening the archive past timing: the owner's call; today each file has
    one copy, recorded by path, host and sha256); a run queue on `jebel` if one run at a time
    under the lock becomes the bottleneck; submissions from other people's machines (its own plan,
    if the project is used widely). The runner's hosts keep ml's venvs; a studies-only dependency
    would need them rebuilt from indrajala-studies.

### 1.0 and distribution

52. **indrajala-ml on PyPI, then 1.0** (the release plan's D1; needs 20). Beside it, each when asked
    for: free-threaded wheels (3.14t: `gil_used = false` and extra wheels, after a thread-safety
    review of `Array`'s interior mutability), ndarray interop in the core (`From` conversions
    behind a feature), a benchmark wheel matrix on each platform's runner.
53. **indrajala-demos depends on the published package** (the demo split's D2; needs 52): until
    then a "Bump indrajala-ml/" PR there carries each library change.

### Kept up as the code changes

- **The DRY audit after each new primitive**, so copies are caught small:
  `.venv/bin/python scripts/measurement/duplicates.py` (the clone scan, at 8 lines and `--lines
  10`), a search for function names defined in more than one module, and the twin classes read
  side by side. The last pass (the third, 2026-10-30) left, at 8 lines: package 2.2%, scripts
  1.8%, tests 3.1%, crate tests 2.0%; at 10: 1.3%, 1.4%, 1.6%, 1.2%: the baseline a rerun compares
  against. Its leftovers, each when its file is next touched: helpers inside single test files
  (`rust/tests/test_window_program.py`'s two 22-line set-ups, `rust/tests/test_attention_ops.py`,
  `tests/array_network_contract.py`, `tests/model/persistence/test_format2.py`,
  `tests/measurement/test_machine_profile_capture.py`, the sequence files' embedding tests); the
  pure-Python recurrence layer's set-up, shared when it gains the schedule's setters; the
  builders' wiring loop (`python_layer_builder.py` and `array_layer_builder.py`, 12 lines), if
  another layer kind that links layers arrives. Not copies: `tests/test_pcg64.py` against
  `rust/tests/test_random_pcg64_parity.py`, and `ab.py`'s and the archive's `_git`. Not planned:
  an audit of the crate, the scripts or the demos; sharing the numpy and Rust twins' method bodies
  or the `Add` twins' classes (separate by design).
- **Shared homes for the next layer kind.** A new layer kind extends these rather than adding a
  copy. The conv and pool argument checks and output size are in `model/specs/window_geometry.py`,
  one shape walk (`spec_shapes.spec_shapes`) feeds both builders, the optimizers share
  `OptimizerBase` and `ArrayOptimizerBase`, and a layer's optimizer accessors are `WeightAndBias`,
  `GammaAndBeta` or `AttentionProjections` (`array_parameters.py`). A hidden layer takes its
  output-delta refusals, its downstream-as-delta methods and its no-op gradient accumulation from
  `model/specs/hidden_layers.py` (`Hidden`, `DeltaIsDownstream`, `ParameterFree`, and `BatchOnly`
  for a layer that trains on batches only); a network its spec and rule hooks from
  `model/specs/spec_hooks.py` (`SpecHooks`, `RulePreset`, `GivenRule`); a token-wise output its
  predictions from `TokenWiseOutput` and a conv preset its argument checks from `ConvArguments`
  (`specs/array_network_shapes.py`); a recurrence layer its schedule's setters from
  `RecurrenceSchedule` (`specs/window_recurrence.py`); a pure-Python gamma-and-beta row from
  `GammaAsWeights`; a token shape from `spec_shapes.token_shape`. Tests share `bits`, `split`,
  `max_relative_gap`, `patching`, `randomized`, `learn_in_step` and the `assert_*` scenarios
  (`tests/helpers.py`), the crate's op tests `bits`, `to_numpy` and `to_array`
  (`rust/tests/array_helpers.py`), and a scenario every implementation runs takes conftest's
  `implementation` fixture. A study takes its cached MNIST loading, mean ± sd cells and Markdown
  tables from `indrajala_ml/studies/common.py`.
- **Not planned:** normal and integer draws, broadcast `low`/`high`, other bit generators (Philox,
  SFC64, PCG64DXSM), and changing the crate's legacy MT19937 functions (the RNG generators plan's
  D7); the bounds-width networks' draws (`BackpropClassifierNetwork`'s presets and
  `LinearClassifierNetwork`; no array twin, so no order to match); renaming or deleting a network
  class (composable layers' D1); a summed per-token loss, a separate `TokenOutput` spec, a
  network-level causal flag (the sequence task's D6, D7); dropout fields on the layers that end a
  body, renormalising `P` after dropping (attention dropout's D2 (b), D3 (b)); non-timing gates
  and significance tests in `ab.py`.

## 2. Topic work, by topic

### Machine discovery: the window-recurrent research

The research workplans are in indrajala-research; these are the leftovers no open plan there holds.

**The collapse and the fork** (the division work; the collapse-fork workplan, `git show
89cf3a3:docs/collapse-fork-workplan.md`, whose probe worked as registered):

- **The fork in the harness** (`window_recurrent_study`, in indrajala-studies; D5 built it in the
  probe tool only), and **the cause** of a collapse. Candidates, cheapest first: log the loss and
  the gradient's norm per batch around a collapse (a probe, to confirm the cause); clip the
  gradient's global norm; a rate that warms up at each curriculum step and decays; rollback and
  cool (restore the step's best network and halve the rate on a sharp fall of the training
  answers); an exponential moving average of the weights, certified in place of the raw weights;
  then the codebook's own remedies (code restarts, an EMA codebook) and, last, the commitment
  controls (`git show df1dcac:docs/commitment-control-workplan.md`, deferred for a probe that must
  train past a solution or shows factors collapsing). Each probed on replays of `p` 7 seeds 0, 1
  and 3. **Division stage 5's runs** sharpen it: large drops are common (24 in 14 of 40 runs, in
  successes too) and 17 of 24 recovered, so a failure is a drop not recovered from; the curriculum
  moves on after a plateau whatever the accuracy, so an unrecovered drop meets longer lengths
  broken; and the fork's trigger (the mean screen score below 0.6 of the peak) misses about half
  the drops, seed 101's among them. Two candidates from this: **a step ends only when its peak is
  recovered** (or the step's best network is restored before moving on), and **a drop threshold
  on the worst answer**. Most of the room above the 65% at `p` 5 looks like retention, not search.
- **The fork with the certificate guard:** the guard joins the fork, tested as the fork alone
  against the fork plus the guard (the guard, below).

**The guard and the trigger across machine classes** (taken up by
[general-algebras-workplan.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/workplans/general-algebras-workplan.md),
its stages 5-7, **paused since 2026-10-20**; the detail stays here until it resumes):

- **The certificate guard, softened** (the guard workplan, `git show
  ed0e378:docs/guard-training-workplan.md`; deferred by the owner with the trigger, "fine tune when
  we can use genuinely divergent tasks"). Built and kept: `certificate_guard.py` (the read by
  levels, `Guard`), `train_epoch(..., guard)`, the family's `--guard off|on|both` with its stop
  rule for a frozen run, and `guard-epoch` (one epoch from a saved network, unguarded and
  guarded). Its registered test, G1 (division at `p` 11, the fork alone against the fork plus the
  guard), was never run: from 7 right `p` 11 networks one further epoch lost the right machine in 7
  of 7, and the guard held each only by undoing 66-100% of the updates, freezing learning. Before a
  test, softenings to compare with `guard-epoch`: judge a flip only after N batches; scale a
  flipping update down instead of undoing it; restore the last right machine once an epoch (the
  fork with a correctness trigger); undo only updates that add wrong pairs; guard only the states
  training visits; calm the optimizer once right (cap the update, reset Adam's moments); shrink the
  codebook to the minimised machine once right. First the cause probe: whether the undone flips
  land on rarely visited codes (`p` 11 reaches 210-456 codes for 121 states). **Seen again in the
  consolidation plan's P1 (2026-10-19):** armed on a perfect screen, the guard froze two
  move-to-front 4 networks with 6-8 near ties at 1.05, undoing 96-100% of the updates, where
  unguarded they certified within 1-3 steps. The first softening to try is arming only on a step
  that passes both verdicts.
- **The fork's trigger, tuned across machine classes** (the p11-trigger workplan's stage 4, `git
  show 28894cf:docs/p11-trigger-workplan.md`; the owner, 2026-10-17: the trigger "would need to be
  either target machine agnostic, or we need a way to tune it in general"). The tools exist:
  `collapse.Trigger`, every screen's per-kind shares in the logs, and `modular_family_probe.py
  drops`, which scores a trigger offline against any logs. The candidate from division's logs: the
  mean over answer kinds more than 0.2 below the step's peak, floor 0.8. The test: a trigger chosen
  with `drops` over every class's logs, registered, then paired replays (the same seeds, only the
  trigger changed); **better** if it certifies more seeds, or as many with fewer epochs, losing
  none today's trigger certified.

**Beyond arithmetic: other finite-state targets** (the owner and Claude, 2026-10-17). Every target
so far is an arithmetic transducer, the weakness indrajala-research's `contribution.md` names under
"Breadth". Every target below is an explicit machine, so the certificate applies unchanged. Each
cell keeps the feasibility rules (codes at least the minimal states; training to the exclusion
bound `m + q - 1`), which for random machines with many states means long training lengths. In the
order recommended:

1. **Machines with no arithmetic reading, split by their algebra** (the general algebras plan):
   random Mealy machines; pure permutation (group) machines, the word problems of S3, A5 and S5,
   against aperiodic (reset, flip-flop) machines. The general claim: **training on the coordinates
   of a machine's holonomy decomposition helps**, not only for division. Liu et al. 2023 and
   Merrill's results make S5 the hard case for transformers, so a certified recurrence exact on S5
   at every length is a sharp contrast.
2. **The standard regular-language benchmarks:** the Tomita grammars, parity, bounded-depth Dyck
   (Weiss et al. 2018; Wang et al. 2019): directly comparable, but they fill tables.
3. **Real-world transducers:** a UTF-8 decoder or validator (about 9 states, Hoehrmann's DFA);
   finite-state phonology and morphology (Kaplan and Kay's rewrite rules, a fragment of an HFST
   analyser); protocol and controller machines; sequential circuits (Gray and BCD converters, LFSRs
   and CRCs).
4. **Later, beyond finite state:** counters and pushdown languages (unbounded Dyck); the
   certificate as it stands wouldn't apply.

**A measurement gap: the components' kind (2026-10-21).** `approximations` reads a learned
machine's decomposition as a parallel cover or a cascade, but not whether the components are
irreducible (Krohn and Rhodes: primes, the simple groups, and units): each component's own
transformation monoid, and for the lower one its action within each block above (its holonomy).
So it tells S3's network cascade from Krohn and Rhodes's only by the named congruence on top, not by
the parts' algebra. Wanted before a discovery study reports a decomposition: a reading of each
component's monoid, its groups and its prime loops.

**Composite tasks and approximations** (the division, modular-family and factored-state work):

- **The running affine map mod `p`** (`x' = a x + b`, `a != 0`; AGL(1, p), `p (p - 1)` states):
  a cascade, not a direct product, so no cover by approximations exists: the test of whether one
  approximation helps where no cover does. Later, `a = 0` adds resets.
- **The approximation claim's open questions** (A1-A3 hold: `git show
  c201622:docs/approximation-claim-workplan.md`): whether the sum helps *because* it is an
  approximation needs a learnable non-approximation control (the programme audit's F3); whether
  any approximation helps (one control, the product, at one machine), and at larger `p`.
- **Factored readouts:** every certified division network spreads its state over both codebooks.
  To make factors carry residues again, the factored-state plan's D10 reopens: (a) poolings over
  the output layer's tokens, or (c) one window pass per factor, its `StateInput`s reading that
  factor alone ((b) blocked learning even one counter).
- **Staggered factors** (C9'', 0 certified): start a factor at its turn re-drawn and re-seeded, its
  commitment off until it splits, a turn kept only if the answer improved; or train jointly until
  each factor splits, then stagger. And the factored state's untried variations: the cross-term
  on, more factors; a state loss only if the owner chooses it.
- **"Characteristic sample"** is used only with a named algorithm (de la Higuera 1997, Definition
  1); Oncina & García 1992 (RPNI) is still to read.

**Smaller open runs** (the commitment, one-code-one-state, consolidation, population, window
branches and window-recurrent claims plans; none planned):

- C2 at 6 seeds with R3/R4 on every backdrop, and R3/R4 against R4 alone.
- The population plan's level 2 (selection among runs by their machines).
- Division `p` 17 at width 32, for whether width's cure holds as the machine grows.
- The window branches: a stop-gradient between the answer branch and the trunk (D4); two heads a
  branch and a deeper trunk (D2); the branches at the base width on the algebras' `n` 4 cells.
- E3's dropped cells (D34): `m` 32 and every `k` 1 cell; the curriculum against all at once at
  more cells.
- The transformer against the linear window at `k` 1 `s` 16 `m` 16, trained to the bound with
  every step certified: a candidate, not registered.
- The snap wall, paused (P1 refuted C1): the candidate read after the data (the collapsing seed
  snaps its 3 states to 8 codes, the others to 3-5; spare codes could swap or merge), for a
  registered probe on networks with many codes in use.
- The audit remedies' codebook primitives (k-means codes, re-seeding, the codebook's rate, SimVQ,
  the selection set, He initialisation, FiLM, MRI): candidates in the state-formation workplan
  (research #139), judged only by a registered study there.

**The batch runner's limits** (indrajala-studies; working-notes, The batch runner):

- **Find mode's verifier** (the programme audit's F7): in discovery the stop is answers-only under
  a state bound, which gave false discoveries at a slack bound (1.1 times the target's size: 25
  false stops, 1 false discovery); find mode needs an exact answers-only check (Chow's test set
  under the bound, with queries) before Luby's cost model applies.
- Restarts assume attempts independent and identically distributed: true of fresh seeds, not of
  attempts on different backdrops. A cut attempt loses a certificate that would have come after the
  cutoff, by design. A ledger from before stage 5 holds summaries without `epochs`.

**The programme audit's open calls** (2026-10-23; the reviews in indrajala-research's
`docs/reviews/review-2026-10-23-*.md`), the owner's: F1 (the fresh-seed rate's verdict against the
deadline's completion); F2, F3, F20, F22, F23, F27 (the papers' wording, and whether it waits for
stage A); F5 (retire the panel's gate or run its affordable form); F6, F32 (the slack bound read
offline; Chow's test set built); F7 (above); F9, F10, F24 (the protocol's sidedness and mid-run
moves; a table of the method's constants); F13-F18 (contribution.md brought up to the fresh-seed
evidence, the class line, the 120-against-121 confound, the method's scope, an approximation beside
S3). The reviews' lesson: when a gate is dropped, re-home every recommendation that waited on it.

### Sequence models and language

- **An encoder-decoder that reads its source.** The translation probe (#1242;
  `scripts/studies/translation_probe.py`'s docstring) was abandoned by the owner: English bits per
  character fell from 3.33 to 1.81 in 23 epochs, but the source ablation's gap reached only 0.027
  bits and the greedy outputs ignore the Greek (chrF 20.6). No translation claim rests on the model
  until a run learns to use its source. Beside it: **a decoder-only translation model** (the Greek
  as a prefix, the loss on the English only; cross-attention's D18 (c)) when a question needs the
  comparison, and **pretraining the encoder on Greek alone** (D10; with the BPE tokenizer, infra 35).
- **Corpora to add** (the owner, 2026-10-31: both in the long run): Newton's *Principia* (Motte's
  1729 translation), long uniform prose like Euclid's; Whitehead and Russell's *Principia
  Mathematica*, which needs a source with its symbols first. Check each edition's licence (the
  literature rule). **Word-level corpora** need the BPE tokenizer (infra 35); a vocabulary of
  10,000 or more makes the output layer the whole cost.
- **A synthetic sequence task** (the sequence task's D2 (b)): copying or reversal from a seed, with
  a known optimal loss. The study's leak arm and the mask's own test already cover what it would
  test.
- **The recipe:** the cosine's peak stayed the constant rate's tuned one (D8 (a)); a decay usually
  tolerates a higher peak, untested. Not planned: which half of the recipe gains (the owner took
  it from the literature: most likely AdamW's; both halves stay).
- **The studies' open questions, not planned:** attention dropout's (a rate tuned per arm; longer
  runs; a larger model; whether its gain across Euclid's shift is real); rotary's (`rotary_base`
  untuned; its own tuned rate; convergence past 10 epochs); context length's (the owner,
  2026-11-02: no follow-ups; C3 holds only at this model, the 1,024 arm trains on 2 windows a step,
  Euclid's contiguous split overfits by epoch 30).
- **Grouped-query attention, not measured here:** its quality at this library's sizes, an uptrained
  model against one trained from scratch (the study's U1), and the decode rate at each `g`
  (`generation_timing.py`, minutes). The registered study's code (`gqa_study.py`) stays, never run.
  Reinvesting the saved parameters (MobileLLM widened the embedding) is a model-design choice.
- **The FFN activations study** (abandoned after 24 of 80 runs, so ReLU stays), if a later model
  needs the FFN decided by measurement. Its 24 partial runs are in pyramidon's
  `data/studies/ffn-activations/sweep/` (with a manifest), not analysed. Size its budget on the
  machine's throughput at the worker count it will run (working-notes, `stone`'s throughput).

### Image models: MNIST, batch size and depth

- **The conv batch-size study with batch norm** (the rule still fails at B = 512;
  `batch_size_scaling.py`'s docstring), each a scratch probe or a short sweep: a gentler base rate
  (from 0.5, B = 512 runs at 8, inside the stable range); capped rates at B = 512 (4, 8, 16, 24 with
  a 1-epoch warmup); longer runs at B = 512; and the dense momentum 0.9 rerun on eq. (9) (`scaling
  --lr32 0.9=0.25`), its earlier cells measured without the momentum correction.
- **The depth study's open question:** with batch norm, the plain network's first-layer gradient at
  initialisation grows tenfold from 2 to 16 hidden layers (the residual one's 1.8x), and at 16 its
  seeds' standard deviation is 3 points in the first two epochs. Unexplained.
- **Layer norm against batch norm on the dense networks** (layer norm's D11 (b)): the depth study's
  batch-norm cells rerun with a flat `LayerNorm`.
- **Demos for the new conv presets** (the presets plan's D4), each a tuned, measured PR: a demo
  makes an accuracy claim.
- **Tests and figures:** a step-by-step gradient check of the batch-norm conv networks (the
  attention tests' `_scales` would need to give a linear conv layer before batch norm its scale;
  nothing checks them step by step today); docstring figures measured with the old draw order
  (such as `randomize_fan_in_aware`'s 99.5% and 96.9% on UCI digits) are re-measured only when
  cited as current.

### The adders and length

The addition study's leftovers (`git show 8990f74:docs/addition-study-workplan.md`; the findings in
indrajala-studies' `scripts/studies/addition_study.py`'s docstring):

- **The adders' threshold as a check for each new primitive:** at `n = 4`, d 4 fails at every
  candidate and shape, d 6 passes from 2 layers; whether a change moves that edge is one harness
  config per question. Not yet run for rotary and no positions against the learned `Position`; the
  recipe (including whether it smooths the non-monotone ladders, which suggests the "too small"
  verdicts near the edge are partly the optimiser's); the FFN activations and grouped heads.
- **Milestone A's baseline** ([arbitrary-length-addition-workplan.md](https://github.com/davidbarkhuizen/indrajala-research/blob/main/docs/workplans/arbitrary-length-addition-workplan.md),
  H2, paused): retrain the column transformers with rotary or no positions at the sweep's
  threshold sizes, rates and budget; running a model at another length needs infra 27.
- **Probes of the passing adders** (needs infra 49): the smallest passing 2-layer networks against
  near-misses one width below, to locate the carry lookahead (which layer finds the deciding column,
  which head propagates the carry); retrained on the code of the day.
- **More of the catalogue:** thresholds for candidates 4-8 (each wants a reason to expect a pass
  first: the recipe, the answer-only loss for 5, a larger budget for 7 and 8, whose best screens
  reached 0.999 and 0.998); a third width, `n = 8` (sampled carry patterns, the costliest runs);
  an FFN-width knob, if a threshold turns out to be the FFN's; sample efficiency, one fixed
  training set (its own study); and candidates now buildable but never run (the string format's
  answer-only loss, variable-length answers decoded autoregressively, the training recipe).
- **Candidates needing new infrastructure, not planned:** a weight-tied (looped) transformer
  (blocks sharing weights, a variable iteration count); one dense network for every sum digit
  (several softmaxes on one flat network); supervised carries (auxiliary heads and losses); a 1-D
  convolution along the tokens, a local, ripple-like bias against attention's.
- **For any later fixed-depth length study** (length generalisation, terminated 2026-10-12; its
  literature, `git show c3d619b:docs/length-generalisation-workplan.md`): rotary and no positions
  fail at long range by construction (resolution and dilution); an additive distance bias (ALiBi,
  infra 28) doesn't, and D12's start token is what a no-positions control needs.

### Performance readings and measurement practice

- **Unexplained readings for the next baseline:** a candidate +1.2% in the window program's
  inference forward on token windows (klip, `forward family p7 f32 n1024`, 5.761 -> 5.832 s
  through ONNONO; no mechanism found; re-check on `jebel`: an A/B of crate f73d652 against
  2c4415b, then `kernel_code_diff.py` on `program::forward` if it holds); after crate #54, Rust
  `hidden_delta_batch` at batch 32 3.5% slower with its products unchanged and
  `accumulate_gradient_batch` 7.6% faster; #678's crate bump 7-18% faster on five Rust rows. Read
  as codegen or layout shifts (x86 placement: infra 41); none claimed.
- **Rerunning the baseline** after a change that could move every number (kernel, BIOS,
  numpy/OpenBLAS, rustc, the frequency policy): the benchmark-machine plan's stages 4-5 at the same
  tags, then at `main`, replacing [machine_profiles/i7-9700k.md](machine_profiles/i7-9700k.md)'s
  tables in one PR; the parked plan's closing baseline is one. The laptop's columns need rerunning
  only after a software change on it; a crate change for the i7 is A/B'd on the laptop too,
  preferably on one thread (its default-threading A/As spread 32% per pass at the median row).
- **The agent's cost of an A/B:** count the turns and characters read per A/B from the session
  transcripts, against the numbers before the harness (20-30 checks and 40-130K characters of raw
  output per A/B), and record it here; waiting on CI (`gh pr checks --watch` loops) is worth the
  same treatment.
- **Not planned:** changing the timing protocol except from data; hosting results anywhere but the
  archive repository; benchmarks in CI; claiming a cross-machine speedup from the laptop columns;
  BIOS settings beyond cooling and power, and kernel boot parameters; numpy's cross-machine
  behaviour (aarch64's D1); checking other aarch64 machines beyond the committed reference.
