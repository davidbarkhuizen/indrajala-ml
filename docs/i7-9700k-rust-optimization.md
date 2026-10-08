# Rust against numpy on the i7: what to investigate

An outline, not a workplan. It covers the places where the crate's Rust lags numpy on the
i7-9700K (`jebel`), or where its ratio to numpy got markedly worse from the Ryzen laptop to the
i7. It also has one opportunity that is likely significant although Rust doesn't lag there
(item 3). Small wins are left out: the aim is the large gaps, not diminishing returns. For each, it says which kernel or parameter change to investigate, ordered by
expected payoff. Each item becomes its own workplan when it is taken up.

The numbers are from the benchmark machine workplan's stages 5 and 6, in
[machine_profiles/i7-9700k.md](machine_profiles/i7-9700k.md): the per-op table, the conv demo
ratio table and the epoch shares. Ratios are Rust / numpy time, so above 1 means Rust lags.
Absolute times aren't compared across machines, only ratios.

## The gaps

Every per-op row that lags numpy on the i7 or got markedly worse than on the laptop, with the
item that covers it. Ratios are given at default threading / one thread each.

| shape | op | batch | i7 | laptop | item |
| --- | --- | --- | --- | --- | --- |
| 32 x 5408 (conv tail) | `accumulate_gradient_batch` | 512 | 1.25 / **3.93** | 0.59 / 2.81 | 1 |
| 30 x 784 | `accumulate_gradient_batch` | 512 | 1.44 / 1.49 | 1.04 / 1.31 | 1 |
| 32 x 5408 | `forward_batch` | 512 | 1.55 / 1.55 | 1.31 / 0.99 | 2 |
| 32 x 5408 | `forward_batch` | 32 | **2.56** / 1.34 | 1.50 / 0.92 | 2, 4 |
| 30 x 784 | `forward_batch` | 512 | 1.23 / 1.21 | 0.94 / 1.05 | 2 |
| 32 x 5408 | `downstream_batch` | 32 | **3.09** / 0.91 | 1.53 / 1.01 | 4 |
| 32 x 5408 | `accumulate_gradient_batch` | 32 | 1.64 / 0.93 | 1.42 / 1.10 | 4 |
| 30 x 784 | `downstream_batch` | 512 | 1.85 / 0.94 | 1.41 / 1.21 | 4 |
| 30 x 784 | `downstream_batch` | 32 | 1.23 / 0.74 | 1.25 / 0.99 | 4 |
| 30 x 784 | `forward_batch` | 32 | 1.15 / 0.80 | - | 4 |

There is also the MNIST-subset conv demo (item 5). Rust still wins every network there, but
every ratio moved towards numpy, most for the single-example `conv` network (0.20-0.21 to
0.41). The UCI digits ratios held.

Rust doesn't lag in whole epochs (conv mini-batch at 0.81x and 0.94x, dense at 0.45x), the
conv tail's `downstream_batch` at batch 512, or `accumulate_gradient_batch` at 30 x 784 batch
32. These are left out unless one of the items below moves them.

Every change below has to keep the golden run bit-identical. Each output stays one FMA chain in
its fixed order, so the changes are limited to blocking, tiling, threading and dispatch, which
the crate already does without moving a bit.

## 1. Block `matmul_2d` over `K` for few-row, long-`K` products

**Rows:** `accumulate_gradient_batch` at batch 512. The conv tail takes 15.4 ms against numpy's
3.9 ms on one thread, the largest gap in the table (the laptop's was 2.8x). 30 x 784 is at about
1.45x both ways.

**The likely cause:** `delta_batch.T @ X` goes through `matmul_add` and `matmul_2d`. That kernel
sizes its row blocks so one block of `a` is 16 KB: 4 rows at `K` = 512. Each block streams all
of `b`, and the 32 rows read it 8 times. On the conv tail `b` is 512 x 5408, 22 MB, past the
12 MB L3, so that's about 177 MB from memory per call, which accounts for the 15 ms. At 30 x 784,
`b` is 3.2 MB, past the L2, and is read 8 times from L3. Why the gap grew from the laptop's
isn't established: numpy's gain from the i7's 8 cores may explain part of it.

**To investigate:** run these shapes like `matmul_long_k` does, in `K` slabs that `RESUME` each
chain (exact), with every row and column tile served from cache before the next slab, so `b`
is read about once. Pick the route by shape in the dispatch (few rows, `b` past the cache).
Threaded batch 512 needs the same check: threading by rows makes every thread read all of `b`,
and `matmul_long_k` threads by column chunks for that reason.

**Payoff:** high, with a known mechanism and a pattern the crate already has. One read of `b`
could bring the conv tail to numpy's time or better.

## 2. Tile `matmul_nt` for a `W` past the L2

**Rows:** `forward_batch` (`X @ W.T`, through `matmul_nt`). The conv tail is at 1.55x at batch
512 and 1.34x at batch 32 on one thread. Both were about level on the laptop, and batch 32 is
the only batch-32 op where one-thread Rust trails. 30 x 784 is at 1.2x at batch 512.

**The likely cause:** the conv tail's `W` is 32 x 5408, 1.4 MB, past the 256 KB L2. Each 4-row
block of `X` runs the 4 x 2 register tile across all of `W`, so `W` streams from L3 once per 4
rows of `X`: 128 times at batch 512. At 30 x 784, `W` (188 KB) nearly fills the i7's 256 KB
L2 by itself, so it competes with `X`'s rows. The laptop's L2 is 512 KB per core, which fits
both shapes doing better there.

**To investigate:** block over `K` so a slab of `W` stays in L2 across many row blocks of `X`.
The 4-lane accumulators are stored and resumed per slab (exact), and the lanes are combined only
after the last slab, keeping `dot_product`'s grouping. Retest the 4 x 2 tile under the new
blocking.

**Payoff:** medium to high. The gap is smaller than item 1's, but `forward_batch` runs at every
batch size and in the accuracy passes.

## 3. Spread `conv_forward_batch` over examples

**Why, although Rust doesn't lag:** `conv_forward_batch` is the largest single op in a Rust conv
epoch: 36% of a mini-batch-32 epoch (the laptop's 29%) and 23% single-example. It runs on one
thread at every batch size, im2col included, on a machine with 8 idle cores. `matmul_narrow`'s
row threading of the product alone gained nothing at N = 512.

**To investigate:** split the batch's examples across threads. Each example's `cols` slab,
product and `A` row are independent and written in order, so no output changes value. Threading
the whole example (im2col, product, and the bias and ReLU pass) parallelizes the memory-bound
part too. Find the batch size where it starts to pay: the threshold work found that in training,
threaded calls start on cold, clocked-down cores.

**Payoff:** likely significant at batch 512, given the op's share. Less certain at batch 32.

## 4. Threaded scaling against OpenBLAS

**Rows:** the default-threading rows that are level or better on one thread: conv-tail batch 32
(numpy 2.6-3.1x faster, against about 1.5x on the laptop), and 30 x 784's `downstream_batch` at
batch 512 (1.85x threaded, 0.94x on one thread). OpenBLAS uses the i7's 8 real cores, and on the
laptop it had 4 cores with SMT. The crate gained less from the extra cores.

**The two parts:**

- **Below the threshold** (conv tail, batch 32, 5.5M flops): the crate stays on one thread under
  its 8M threshold. In epochs, every threshold from 2M to 12M timed equal (stage 6), so lowering
  it alone doesn't help.
- **Above it** (30 x 784 `downstream_batch` at batch 512, 12M flops): the crate does thread, but
  it gains far less than OpenBLAS does.

**To investigate:**

1. Measure the per-call spawn and join cost. The crate spawns OS threads on every call
   (`std::thread::scope` in `for_each_row_range` and `matmul_long_k`), while OpenBLAS keeps a
   pool.
2. Measure how row splitting scales on the threaded shapes.
3. If spawning matters, try a persistent pool (or `rayon`) and re-sweep the threshold.

A pool must not spin while numpy runs, or it causes the slowdown OpenBLAS's spinning causes
Rust now (1.56x).

**Payoff:** uncertain. It's large on isolated ops if spawning is the cost, but the threshold
result suggests cold cores in training limit it. Investigate before committing to it.

## 5. The conv layer on 28x28 inputs

**The gap:** the MNIST-subset conv demo ratios moved towards numpy on every network. The 8x8
UCI digits ratios held. `conv_forward_batch`'s share of a mini-batch-32 Rust epoch grew from
29% to 36%, and it's the largest single op in both the single-example and mini-batch epochs.

**To investigate:** find which ops lost ground at 28x28. Time the conv ops against numpy's, per
op, at batch 1, 32 and 512:

- `conv_forward_batch`
- the downstream, through `matmul_narrow`
- the accumulate, through `matmul_long_k`

The single-example `conv` network lost the most (0.20 to 0.41), and item 3 doesn't reach batch 1.

**Payoff:** medium. Rust is still ahead on these networks, so the target is the lost ground, not
a lag. The per-op timing decides whether there is a kernel to fix.

## How each item proceeds

- **Wait for the laptop's numbers first:** its ratios above are still from its retired baseline
  (older commits). Stage 5's laptop runs (D11) replace them and may change which rows count as
  "markedly worse".
- **Workflow:** each item gets its own workplan, with the crate change landing in
  `indrajala-math-rust` first and then a "Bump rust/" PR here.
- **Stop at the bar:** an item ends when its next step can't clear the 5% noise bar in an
  epoch. Don't polish an item past that.
- **No new fixed constants:** block and slab sizes that items 1 and 2 introduce come from the
  machine's cache sizes, read at run time, with an override hook, as the owner directed. The two
  machines' L2s differ by 2x (256 KB against 512 KB per core).
- **Measuring:** each change is timed with `ab.py` as a crate A/B (the two `.so` hashes must
  differ) on both machines, and passes the golden run bit-identical. A change for the i7 must not
  cost the laptop.
