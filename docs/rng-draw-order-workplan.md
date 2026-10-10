# Workplan: pure Python's random draws in numpy's order

**Status: decisions D1-D7 settled by the owner (2026-10-10), each as recommended. Stages 1 (this
plan), 2 (weights), 3 (dropout masks) and 4 (the parity tests) done; stage 5 not started. Done before roadmap step 7
(the owner, 2026-10-10).**

From one seed, numpy and Rust build the same network and draw the same dropout masks, by bits
(`tests/model/networks/test_seeded_init_parity.py`). Pure Python draws from the same PCG64 stream
(`indrajala_ml/pcg64.py`, bit for bit with numpy's `default_rng`: `tests/test_pcg64.py`), but not
in the same order, so the same seed gives it other weights and other masks. This plan makes pure
Python draw in numpy's order wherever the two have the same network, so that all three
implementations start, and drop out, alike from one seed.

## Why

- **The parity tests work around it.** The pure-Python parity tests (attention's, layer norm's,
  batch norm's, the sequence task's) can't seed both sides; they build pure Python, then restore
  its snapshot into numpy (`as_array_snapshot`, `_as_array_snapshot`), and pin pure Python's own
  draw order separately. Each new layer kind repeats the workaround.
- **Dropout is compared only at eval.** The per-node dropout reference never meets the array
  networks in training ([rng-audit.md](rng-audit.md), Open work), so dropout's training path has
  no three-way parity test. Roadmap step 7 (dropout in attention) adds masks in a third place; done
  first, this plan lets that step test its masks in all three implementations by bits from the start.
- **It is the open item of a retired workplan** (next-steps.md, From the RNG generators workplan;
  rng-audit.md, Open work), recorded again by the sequence task (#631).

## What differs today

| draw | numpy and Rust | pure Python |
| --- | --- | --- |
| a layer with a bias (dense, conv, token-wise dense) | the whole `W` (rows, fan-in) row-major, then the whole `b` | per row (node, kernel, unit): its weights, then its bias (`fan_in_aware_weights_and_bias`) |
| an attention layer | per projection, in the order `q, k, v, o`: `W`, then `b` | per row of every projection in turn: weights, then bias |
| a layer without a bias (linear, linear conv, embedding) | the whole `W` | per row, weights only: **already numpy's order** |
| dropout masks in a training batch | per layer in forward order, one `(batch, size)` mask, row-major | example-major (`BackpropNetworkBase._learn_batch`): per example, each dropout layer in turn. **The same as numpy's with one dropout layer**; with two or more the layers' draws interleave |
| dropout masks, layer-major batch (a network with batch norm) | as above | per layer, then per example and node: **already numpy's order** |
| `BackpropClassifierNetwork`'s and `LinearClassifierNetwork`'s bounds-width init | no array twin | per node, scaled by the input bounds |

The conv kernel's weights are already in numpy's column order (in channel, kernel row, kernel
column), and every draw is the same function of the stream (`uniform`, `random`), so the order is
the only difference.

## Decisions (settled 2026-10-10)

Each lists the options considered, with pros and cons, and the choice.

- **D1. Scope. Settled: (a).**
  - (a) *Chosen.* The weights of every pure-Python layer with an array twin (D2), and the
    dropout masks of a training batch (D3). Pros: closes both halves of the open item; after it,
    every pure-Python network with an array twin is seeded like numpy. Cons: re-records every
    affected pure-Python golden entry (D5).
  - (b) Weights only. Pros: one stage, one re-record. Cons: leaves the masks, which step 7's
    attention dropout needs in the same order; the dropout networks re-recorded twice if (b) is
    followed later.
  - (c) (a) plus the bounds-width init of `BackpropClassifierNetwork` and
    `LinearClassifierNetwork`. Pros: none for parity: those have no array twin, so there is no
    order to match. Cons: changes their golden entries for nothing. Out of scope.
- **D2. How pure Python draws a layer's weights. Settled: (a).**
  - (a) *Chosen.* Each layer's `randomize_fan_in_aware` draws all its rows' weights in row
    order, then all its biases, through one shared helper in `fan_in_aware_init.py` (rows, fan-in
    -> weight rows and biases), replacing `fan_in_aware_weights_and_bias`; an attention layer
    draws each projection that way, `q, k, v, o` in turn. Pros: each layer keeps owning its draw;
    the helper states numpy's order once; no network-level change. Cons: touches every pure-Python
    layer with a bias (dense, conv, token-wise dense, attention).
  - (b) The network draws numpy-shaped arrays per layer from the layer's parameter shapes and
    hands them over through `restore_state`. Pros: numpy's order by construction, one place.
    Cons: pure Python's `randomize` then depends on the array layers' shapes; `restore_state`
    becomes part of initialization; a layer can't own its draw.
- **D3. Dropout masks in a pure-Python training batch. Settled: (a).**
  - (a) *Chosen.* A pure-Python network with a dropout layer trains layer-major, as one with
    batch norm does (`layer_major.py`): the forward pass runs each layer over the batch before the
    next, so each dropout layer draws its examples' masks, node by node, in numpy's row-major
    order. Pros: reuses the existing path; no new mechanism; attention dropout in step 7 draws in
    order on it too; the path already accumulates gradients in example order, so the dropout
    networks' gradients are unchanged apart from the masks. Cons: the layer-major path's lanes are
    slower (pure Python is never timed); dropout networks join a path written for batch norm, so
    its docstring and its selection (`batch_norm_index`) become "a layer that needs the batch's
    order".
  - (b) Each dropout layer draws its `(batch, size)` mask up front, layers in forward order, at
    the start of a training batch; the example-major loop then reads row `n` for example `n`.
    Pros: keeps the example-major loop. Cons: a mask buffer and a "draw now, read later" protocol
    on each dropout layer, a second mechanism beside the layer-major path.
  - (c) Leave it. Pros: the order already matches whenever a network has one dropout layer; no
    change. Cons: a network with two (the specs allow dropout on every sigmoid hidden layer), or
    step 7's attention dropout beside another, draws out of order, and nothing tests it.
- **D4. The tests. Settled: (a).**
  - (a) *Chosen.* `test_seeded_init_parity.py` gains pure Python: every pure-Python Sequential
    network and preset with an array twin, from one seed, against numpy by bits (weights and, after
    a seeded training batch, the generator's state and the masks). The parity tests that restore
    numpy from pure Python's snapshot seed both instead, and the tests that pin pure Python's own
    draw order go. Pros: removes the workaround the plan exists for. Cons: many test files touched
    (attention, layer norm, batch norm, residual, sequence, conv parity tests).
  - (b) Add the seeded tests, keep the restoring ones. Pros: smaller. Cons: keeps two ways of
    building a parity pair, and the next layer kind copies the old one.
- **D5. The golden run. Settled: (a).** The draws change every pure-Python entry whose network
  draws fan-in-aware weights with a bias, and the dropout entries; numpy's and Rust's entries
  can't move. A re-record under §8 (measurement.md), as "more correct": all three implementations
  seeded alike. *Found in stage 2:* the golden run injects every network's weights from its own
  `random.Random` and draws only dropout masks from the network's generator, so stage 2's weight
  order can't reach it: it stayed bit-identical (112 networks), with no re-record. Stage 3's masks
  re-record once.
  - (a) *Chosen.* Each stage that changes draws re-records once (stage 2 the weights, stage 3
    the masks), on both machines, archived as `material`, with the list of entries that changed and
    the rest checked bit-identical first. Pros: each re-record has one cause, and its list of
    changed entries can be checked against that cause. Cons: two re-records.
  - (b) One re-record at the end. Pros: one archive record. Cons: a stage in between fails the
    golden check, which every stage must pass.
- **D6. Saved models. Settled: (a).** The committed pure-Python fixtures
  (`tests/fixtures/saved_models/`) hold their weights, so loading them doesn't draw.
  - (a) *Chosen.* Keep them; stage 2 checks that `test_legacy_saved_models.py` passes
    unchanged. Pros: saved files stay what they were. Cons: none known.
  - (b) Regenerate them from the new draws. Cons: rewrites committed fixtures for no reason.
- **D7. Timing. Settled.** Pure Python only, never timed: tier 0, no A/B (measurement.md, §1).

## Stages

1. **This workplan** (D1-D7 settled), with the roadmap: done before step 7.
2. **Weights** (D2), done: the shared helper (`fan_in_aware_weights_and_biases`, and
   `fan_in_aware_weights` for the layers without a bias); every pure-Python layer with a bias
   draws numpy's order, a conv layer its kernels and an attention layer each projection as one
   matrix; seeded init parity against numpy by bits, weights and generator state, for every
   pure-Python network with an array twin (D4), and a check that every pure-Python class is
   either covered or bounds-width. The golden run stayed bit-identical (D5): no re-record. The
   tests that pinned pure Python's own order now pin numpy's (`tests/helpers.py`'s `numpy_draws`);
   the two digit-pipeline tests' measured accuracies were re-measured. New starting weights
   exposed two parity checks tuned to one seed, both on gradients that are rounding noise (a bias
   that batch norm or softmax cancels): attention's bk bound measured over 12 seeds (now 1e-11 of
   the layer's scale), and the batch-norm conv network "after a relu conv" under Adam, whose
   trajectory Adam's epsilon amplifies past 1e-9 at half of 20 seeds, compared step by step
   instead.
3. **Dropout masks** (D3), done: a pure-Python network with a dropout layer trains layer-major
   (`BackpropNetworkBase._layer_major`: batch norm or dropout; `batch_norm_index` keeps only the
   one-example refusal). A seeded training batch draws numpy's masks by bits and leaves the
   generator in numpy's state, with one, two and three dropout layers, and behind conv and pool
   layers (`test_seeded_init_parity.py`); the old example-major loop fails it with two or more.
   The golden re-record (D5) moved exactly the two pure-Python entries with two dropout layers
   (`python dropout` and `python multiclass dropout`, `LAYER_SIZES = [4, 3]`); `python dropout
   conv`, with one, and every other entry stayed bit-identical, archived as `material`.
4. **The parity tests** (D4), done: every pure-Python parity test (attention's, multi-head
   attention's, layer norm's, batch norm's, batch norm's conv networks', the residual blocks', the
   sequence task's) seeds both sides from one seed and checks them alike by bits, weights and
   generator state, before training (`tests/python_array_snapshot.py`: `seeded_like`,
   `assert_seeded_alike`, and the one `as_array_snapshot` that replaced the attention and
   batch-norm tests' two). The tests that pinned the network draws against a reconstruction of
   numpy's order (`numpy_draws`) now check against the numpy network itself, and `numpy_draws`
   went. Layer norm's parity test no longer reseeds both generators before training: seeded alike,
   they draw the same masks. The trajectories start from the same bits as before, so every
   tolerance stands unchanged. Kept by design: the step-by-step Adam checks still restore numpy
   from pure Python at each step, since they compare gradients from the same weights, not
   trajectories; and restores between two networks of one implementation (pure Python's learn
   against learn_batch, numpy against Rust) aren't this plan's workaround.
5. **Docs**: rng-audit.md's Open work, next-steps.md's entry, the README's notes on parity; the
   workplan retired.
