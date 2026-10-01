# Workplan: the conv batch-size study with batch norm

**Status: stages 1 (#517), 2 (#518) and 3 done. The rule fails at B = 512 for both arms at both
momenta, so stage 4 is skipped (D6). Stage 5 is next.**

The conv batch-size study (`indrajala_ml/batch_size_scaling.py`, findings in its docstring) found
that the linear learning-rate scaling rule (Goyal et al. 2017) fails for the conv network at
B = 512 at momentum 0.0 and 0.9, with or without warmup. That network had no batch norm. Goyal et
al.'s setup has batch norm after every conv, with its statistics over groups of 32 examples. This
plan reruns the study with batch norm, in plain form and with ghost groups of 32, to find whether
the rule then holds at B = 512. It is the next step in
[primitives-roadmap.md](primitives-roadmap.md) and the first item of
[next-steps.md](next-steps.md), From batch norm.

## Why

- **The failed cells were tested without half of the paper's setup.** Batch norm is part of the
  ResNet-50 that Goyal et al. scale to 8k. Their § 2.3 fixes its sample size at 32 per worker
  whatever the total batch size, and argues that this keeps the loss function unchanged as the
  batch grows. So the old conv result doesn't yet say whether the rule fails for conv or only for
  conv without batch norm.
- **Batch norm is new (#493-#503) and hasn't trained anything real.** Its tests are parity and
  hand-computed checks on small networks. This is its first use on full MNIST, in a setting where
  the literature predicts it should matter.
- **Either answer gets recorded.** If B = 512 still fails, the study's curvature-ceiling reading
  gains weight. If it holds, B = 1024 is next (D6).

## What the old study found

Conv `[ConvSpec(3, 8)]` (ReLU), dense `[32]` (sigmoid), 10 sigmoid outputs, squared loss. Full
MNIST, Rust, 3 seeds, 3 epochs:

| momentum | `lr_32` | batch-32 band | B = 128, scaled, 1-epoch warmup | B = 512, scaled |
| --- | --- | --- | --- | --- |
| 0.0 | 2 | 97.16% ± 0.61% | 96.63% ± 0.72%, in band | chance, either warmup |
| 0.9 | 0.25 | 96.99% ± 0.70% | 96.63% ± 0.53%, in band | chance, either warmup |

At B = 512 the best momentum 0.0 cell was a capped rate of 16 (94.63% ± 1.02%), and at momentum
0.9 the unscaled 0.25 (87.12% ± 4.70%). The tables are in PR #454 and
`git show 4e2eda7:docs/conv-batch-size-scaling-workplan.md`.

## Decisions

Settled with the owner on 2026-10-01, all as recommended.

- **D1: the batch-32 rate is re-tuned for the batch-norm network.** Batch norm changes the loss
  surface and is known to tolerate higher rates, so the old `lr_32` (2 and 0.25) may not be the
  new network's best. Stage 2 reruns the batch-32 baseline at both momenta, and stage 3 scales
  from the rates it finds, as Goyal et al. tune at the base batch size and then scale.
- **D2: batch norm goes on every hidden layer.** The network is
  `Conv(3, 8, activation="linear"), BatchNorm("relu"), Dense(32, activation="linear"),
  BatchNorm("sigmoid"), Dense(10, output=True)`. Ioffe & Szegedy put batch norm before every
  hidden nonlinearity. A dense layer left without it would keep its own limit on the stable rate,
  and a failure couldn't then be attributed.
- **D3: both arms, plain batch norm and ghost groups of 32.** At B = 32 the two are the same bits
  (a group at least the batch's size is plain batch norm), so the baseline is shared. At B = 128
  and 512, plain batch norm takes its statistics over the whole batch, and ghost groups keep them
  at B = 32's sample size. This shows whether the rule needs the fixed group size or only batch
  norm.
- **D4: the old study's grid.** B = 32, 128 and 512; warmup 0 and 1 epoch; the scaled rate and
  the unscaled control; 3 seeds; 3 epochs. Every cell is directly comparable with the old tables.
- **D5: the dense momentum 0.9 rerun stays out of this plan.** It's a different question (the old
  dense network on eq. (9)) and is listed in [next-steps.md](next-steps.md).
- **D6: a gated B = 1024 stage.** If B = 512 reaches the batch-32 band for any momentum and arm,
  stage 4 reruns those at B = 1024 on the same grid. If no B = 512 cell does, the null is
  recorded and the plan stops. Either way, the old plan's timing and demo stages go to
  next-steps.md, not here.

## Pitfalls to design around

- **The 1-ULP control.** Conv training is chaotically sensitive. Compare accuracy across seeds
  (mean ± sd, and the batch-32 band's seed range), never one run against another.
- **The same starting weights across cells.** As in the old study, a seed's weights are drawn
  once with numpy and restored into the Rust network, so every rate, batch size, momentum and arm
  starts from the same weights. The linear layers draw W only (no bias), and batch norm draws
  nothing (`gamma` 1, `beta` 0, running mean 0, running variance 1), so the draws differ from the
  old network's: the batch-norm and old networks don't share starting weights, and aren't meant
  to.
- **No batch of one.** Batch norm refuses a training batch of one example, and ghost groups
  refuse a last group of one. 60000 rows leave a final batch of 96 at B = 128, 512 and 1024 (none
  at 32), which is three full groups of 32, so no cell is refused. Stage 1's tests pin this.
- **The test pass is inference.** `accuracy` classifies one row at a time, which batch norm does
  with the running averages. Plain batch norm at B = 512 moves them once per batch (118 times an
  epoch), and ghost groups 16 times per batch. At `running_rate` 0.1 both converge well within an
  epoch, but the test accuracy measures the running averages as well as the weights.
- **The rate ladder may be too short.** The conv ladder tops out at 16, which was at chance for
  the old network at B = 32. If batch norm's best stable rate is the ladder's top, the ladder is
  extended until a rate fails before `lr_32` is chosen.
- **Run time is unmeasured.** Batch norm adds per-step work, and the Rust conv batch-norm ops
  haven't been timed on full MNIST. Stage 1 times one epoch at B = 32 and 512 so the sweeps'
  estimates are real.
- **Memory.** Each sweep worker loads full MNIST itself (about 0.5 GB), which caps the worker
  count at 4, as before.

## Stages

Each stage is one PR, on the Rust backend, with the standing gates (`./cli lint`, `./cli test`,
the golden run).

### Stage 1: the batch-norm conv network in the study code

- `batch_size_scaling.py` takes `architecture="conv-bn"` and a `group_size` (`None` for plain
  batch norm). `initial_network` builds D2's network through `SequentialArrayNetwork` with
  `SGD()` at momentum 0.0 and `Momentum(m)` otherwise, draws its weights with numpy from the seed
  (`SequentialArrayNetwork(..., backend=NUMPY).randomized()`), and restores the snapshot into the
  Rust network. The `dense` and `conv` paths are unchanged.
- The sweep script takes `--architecture conv-bn` and `--group-size N`. One invocation runs one
  arm. A `conv-bn` rate ladder adds 32 and 64 to conv's.
- **Tests:** the numpy and Rust starting networks are identical for a seed, at both momenta and
  both arms; the starting weights don't depend on the momentum, the arm or the batch size; the
  built specs are D2's, with the group size in both batch-norm layers; at B = 32 the two arms
  give bit-identical training (a group as large as the batch is plain batch norm); a smoke run
  of `train_and_evaluate` on a few hundred rows at B = 128 and 512 with both arms, covering the
  final batch of 96.
- **Measure:** one full-MNIST epoch's step loop at B = 32 and 512, both arms, against the old
  conv network, one process each, quoted in the PR to estimate stages 2 to 4.

### Stage 2: the batch-32 baseline (D1)

    python scripts/batch_size_scaling_sweep.py baseline --architecture conv-bn --momenta 0.0 0.9 \
        --epochs 2 --seeds 3 --out baseline.json

At B = 32 the arms are the same bits, so this runs once, without `--group-size`. `lr_32` per
momentum is the best rate whose seeds all finish at 20% or more, as before. Record the table
here and the chosen rates in the PR.

Result: full MNIST, Rust, 3 seeds, 2 epochs, B = 32 (both arms), final test accuracy:

| rate | momentum 0.0, epoch 2 | worst seed | momentum 0.9, epoch 2 | worst seed |
| --- | --- | --- | --- | --- |
| 0.03125 | 96.25% ± 0.30% | 95.93% | 97.82% ± 0.10% | 97.73% |
| 0.0625 | 97.11% ± 0.09% | 97.01% | 97.88% ± 0.11% | 97.81% |
| 0.125 | 97.45% ± 0.13% | 97.30% | **98.00% ± 0.24%** | 97.79% |
| 0.25 | 97.72% ± 0.11% | 97.59% | 94.66% ± 5.78% | 87.98% |
| 0.5 | 97.85% ± 0.16% | 97.71% | 75.46% ± 5.79% | 68.78% |
| 1 | 97.88% ± 0.08% | 97.82% | 52.61% ± 15.63% | 39.35% |
| **2** | **97.92% ± 0.17%** | 97.82% | 25.44% ± 6.40% | 18.08% |
| 4 | 97.85% ± 0.19% | 97.63% | 10.56% ± 0.90% | 9.58% |
| 8 | 94.46% ± 5.70% | 87.88% | 10.65% ± 0.94% | 9.58% |
| 16 | 34.64% ± 4.88% | 29.09% | 10.07% ± 1.12% | 9.28% |
| 32 | 12.06% ± 2.90% | 9.58% | 9.95% ± 1.26% | 8.92% |
| 64 | 11.38% ± 1.82% | 9.58% | 9.95% ± 1.26% | 8.92% |

- **`lr_32` = 2 at momentum 0.0 (band 97.82% - 98.12%) and 0.125 at 0.9 (band 97.79% -
  98.26%),** by the rule above. The ladder's top two rates fail at both momenta, so it was long
  enough.
- **Batch norm raises accuracy, not the stability edge.** At their `lr_32` the 2-epoch means are 1.1
  and 1.6 points above the old network's (96.82% at momentum 0.0, 96.43% at 0.9 in #454). The edge is where it
  was: at momentum 0.0, 8 is erratic and 16 fails (the old network: 8 erratic, 16 at chance). At
  0.9 the best rate is half the old one (0.125 against 0.25), and 0.25 is erratic (one seed
  87.98%). So stage 3's scaled rates at B = 512 (32 and 2) are 2x and 8x past rates that already
  fail at B = 32, as in the old study.
- **At momentum 0.0 the rate barely matters from 0.5 to 4** (97.85% - 97.92%, within the
  seeds' spread). The rule picks 2, the best mean. Scaling from 0.5 instead would put B = 512 at
  8, inside the stable range: a different, gentler test of the rule, left to the owner after
  stage 3.
- The sweep took 53 minutes, twice stage 1's estimate: 4 workers on 4 cores gained little over
  serial.

### Stage 3: the scaling sweep at B = 32, 128 and 512 (D3, D4)

Once per arm:

    python scripts/batch_size_scaling_sweep.py scaling --architecture conv-bn [--group-size 32] \
        --lr32 0.0=<rate> --lr32 0.9=<rate> --batch-sizes 32 128 512 --warmups 0 1 \
        --epochs 3 --seeds 3 --out scaling-<arm>.json

The B = 32 cells run in both invocations, and must give identical accuracies: a free check of
stage 1's bit-identity at full scale. Record both arms' tables here, against the old study's.

Result: full MNIST, Rust, 3 seeds, 3 epochs. The batch-32 bands (no warmup, epoch 3) are
98.05% ± 0.08% (seeds 97.96% - 98.12%) at momentum 0.0 and 98.10% ± 0.17% (97.91% - 98.21%) at
0.9. Epoch 1 is left out here; it is in the raw output in the PR.

Momentum 0.0:

| B | rate | value | warmup | epoch 2, plain | epoch 3, plain | epoch 2, ghost 32 | epoch 3, ghost 32 | in band (plain, ghost) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | scaled | 2 | 0 | 97.92% ± 0.17% | 98.05% ± 0.08% | 97.92% ± 0.17% | 98.05% ± 0.08% | yes, yes |
| 32 | scaled | 2 | 1 | 97.81% ± 0.01% | 97.96% ± 0.11% | 97.81% ± 0.01% | 97.96% ± 0.11% | no, no |
| 128 | scaled | 8 | 0 | 78.36% ± 9.85% | 87.95% ± 9.46% | 84.70% ± 5.86% | 88.01% ± 9.64% | no, no |
| 128 | unscaled | 2 | 0 | 97.88% ± 0.01% | 97.72% ± 0.43% | 97.83% ± 0.19% | 97.59% ± 0.20% | no, no |
| 128 | scaled | 8 | 1 | 97.79% ± 0.02% | 97.46% ± 0.51% | 97.74% ± 0.23% | 97.37% ± 0.63% | no, no |
| 128 | unscaled | 2 | 1 | 97.71% ± 0.09% | 97.92% ± 0.18% | 97.69% ± 0.13% | 97.78% ± 0.06% | no, no |
| 512 | scaled | 32 | 0 | 9.99% ± 0.36% | 9.99% ± 0.36% | 9.99% ± 0.36% | 9.99% ± 0.36% | no, no |
| 512 | unscaled | 2 | 0 | 96.70% ± 0.69% | 97.01% ± 0.45% | 96.66% ± 0.47% | 96.94% ± 0.33% | no, no |
| 512 | scaled | 32 | 1 | 96.62% ± 0.91% | 91.16% ± 5.83% | 97.07% ± 0.29% | 92.75% ± 6.62% | no, no |
| 512 | unscaled | 2 | 1 | 96.05% ± 1.15% | 96.51% ± 0.66% | 96.27% ± 0.97% | 96.45% ± 0.62% | no, no |

Momentum 0.9:

| B | rate | value | warmup | epoch 2, plain | epoch 3, plain | epoch 2, ghost 32 | epoch 3, ghost 32 | in band (plain, ghost) |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| 32 | scaled | 0.125 | 0 | 98.00% ± 0.24% | 98.10% ± 0.17% | 98.00% ± 0.24% | 98.10% ± 0.17% | yes, yes |
| 32 | scaled | 0.125 | 1 | 97.83% ± 0.24% | 98.12% ± 0.07% | 97.83% ± 0.24% | 98.12% ± 0.07% | yes, yes |
| 128 | scaled | 0.5 | 0 | 62.51% ± 15.47% | 65.44% ± 15.67% | 64.61% ± 15.50% | 65.42% ± 15.60% | no, no |
| 128 | unscaled | 0.125 | 0 | 97.91% ± 0.25% | 98.06% ± 0.05% | 97.78% ± 0.14% | 98.09% ± 0.12% | yes, yes |
| 128 | scaled | 0.5 | 1 | 97.75% ± 0.33% | 98.17% ± 0.10% | 97.71% ± 0.33% | 98.13% ± 0.08% | yes, yes |
| 128 | unscaled | 0.125 | 1 | 97.84% ± 0.12% | 97.90% ± 0.06% | 97.76% ± 0.06% | 97.99% ± 0.10% | no, yes |
| 512 | scaled | 2 | 0 | 14.21% ± 6.86% | 14.40% ± 6.36% | 14.14% ± 7.09% | 14.08% ± 6.68% | no, no |
| 512 | unscaled | 0.125 | 0 | 96.72% ± 0.25% | 97.56% ± 0.09% | 96.63% ± 0.23% | 97.52% ± 0.07% | no, no |
| 512 | scaled | 2 | 1 | 36.72% ± 4.78% | 41.35% ± 9.01% | 37.12% ± 4.95% | 43.35% ± 9.54% | no, no |
| 512 | unscaled | 0.125 | 1 | 96.69% ± 0.13% | 97.61% ± 0.11% | 96.69% ± 0.15% | 97.57% ± 0.17% | no, no |

- **No B = 512 scaled cell reaches the band, in either arm or at either momentum.** So stage 4
  doesn't run (D6).
- **At momentum 0.0, batch norm takes the scaled rate 32 at B = 512 from chance to training, but
  not stably.** With the 1-epoch warmup every seed reaches 95.9% - 97.6% in epoch 2, then two of
  three fall in epoch 3 (plain: 89.8% and 86.1%; ghost: 85.2%). The old network stayed at chance
  (13.94%). Without warmup it is still at chance.
- **At momentum 0.9 the scaled rate 2 (an effective rate of 20) fails as before:** 14% without
  warmup, 41-43% with it.
- **B = 128 holds at momentum 0.9 with warmup (98.17% and 98.13%), and narrowly misses at 0.0**
  (97.46% and 97.37%, about 0.6 points below a band only 0.16 points wide). The old network held
  at both, against bands 1.1 and 1.2 points wide. The band criterion is strict when the seeds
  agree this closely: the B = 32 warmup cell at 0.0 misses at 97.96%, the band's own lower edge.
- **The unscaled control is now the best B = 512 cell, and it is close.** At momentum 0.9, 0.125
  reaches 97.61% (plain) and 97.57% (ghost) with warmup, about 0.3 below the band's lower edge
  and still climbing. The old network's best B = 512 cell was 94.63% (a capped 16 at 0.0).
- **Plain batch norm and ghost groups of 32 can't be told apart.** Every cell agrees within the
  seeds' spread. For this network, the statistics' sample size doesn't decide whether the rule
  holds.
- **The arms' B = 32 cells are identical to the last digit,** the full-scale check of stage 1's
  bit-identity.
- The two sweeps took 1 h 59 min together (11:04 - 13:03).

### Stage 4: B = 1024, gated (D6)

Only for the momentum and arm combinations whose B = 512 scaled cell is in band (or above) in
stage 3. The same grid with `--batch-sizes 32 1024`, which reruns the B = 32 band. If no
combination passed stage 3, skip this stage.

### Stage 5: the findings, and the plan retired

- Write the findings into `batch_size_scaling.py`'s docstring, next to the old conv ones, with
  this plan's tables cited by `git show`.
- Move the leftovers to [next-steps.md](next-steps.md): the capped-rate probes if B = 512 failed;
  the full grid (0.25-epoch warmup, 5 seeds), timing and a demo if it held.
- Delete this workplan and update [primitives-roadmap.md](primitives-roadmap.md).

## Out of scope

- The dense momentum 0.9 rerun (D5).
- Other architectures, layer norm, and batch norm's inference fold.
- Timing the study for optimization work, and any kernel change.
- Capped-rate probes at B = 512, unless the owner asks after stage 3.
