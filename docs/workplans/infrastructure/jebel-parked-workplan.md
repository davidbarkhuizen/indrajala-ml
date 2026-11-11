# Workplan: the crate performance work that waits for a claim machine (parked)

**Parked by the owner, 2026-11-05.** `jebel`, the machine that makes speedup claims, resets itself under
one benchmark, so nothing here can be claimed until it, or a replacement, is healthy. The crate performance
workplan is retired and this holds what's left of it. Nothing here is running, and nothing else depends on it.

The retired plan, with its decisions, findings and every stage's readings:
`git show 4176d10:docs/workplans/infrastructure/crate-performance-workplan.md`. Its stage 12
close-out readings are in indrajala-benchmarks' `readings/2026-11-05-crate-threads/`
([retired.md](../retired.md)); its leftovers outside this plan in [next-steps.md](../../next-steps.md),
"From the crate performance workplan".

## To resume

1. **A healthy claim machine** (below). Run the check; don't assume.
2. **Its profile:** re-record it if anything changed. `jebel` is recorded at PL1 and PL2 65 W.
3. **A clean A/A** of `scripts/measurement/crate_perf/ab_all.sh`'s benchmarks, or the noise rules don't hold.
4. **The items in order,** one PR each, tier 2 with guards, as the retired plan's stages did. D6 stands: epochs decide.

## `jebel`: where it stands

Nine unclean resets from 2026-11-04 14:33 to 2026-11-05 05:28, all during
`sequence_training_timing --windows 1024`, the working-notes' entry has them. No machine check, firmware record or
crash dump, with `kernel.panic` 0.

| ruled out | by |
|---|---|
| kernel | reset on -38 as on -42; klip ran the benchmark clean on -42 |
| power limits | reset at PL1 and PL2 65 W |
| heat | 57-69 C at the last ones |
| solar, UPS, other load | reset on the grid, with klip and litshe off and stone moved. Whether the grid circuit bypasses the UPS is to confirm |
| the pool, the crate | the old side's crate does it; klip runs it clean |
| attack | audit of 2026-11-05: every ssh login since Nov 1 from pyramidon, no failures, one key, kernel untainted |

So hardware, by elimination; nothing logged shows it directly. **The check, in order:** `memtest86+` from the
boot menu, each stick alone; XMP off and voltages, in the BIOS; reseat RAM and power cables; then the benchmark at
`--rust-threads 1`, then at a 4 GHz cap (`scaling_max_freq`); then another power supply. **The owner's say before each
run of that benchmark:** each reset is an unclean reset of their machine. **Healthy means** three clean A/Bs of it in a
row: it has run clean between resets before, once as late as 2026-11-05 05:13.

**2026-11-09, read without load** (the owner rebooted it; it stays off-line while the owner diagnoses it on the side):
- **Each unplanned reset is an instant cut:** in seven of the boots ending 2026-11-04 to 11-05 the journal stops
  mid-routine (an ssh login from `pyramidon`), with no shutdown, panic, oops or machine check, and the machine is back
  in one to two minutes. The other boot endings (2026-11-05 04:53, 05:24, 05:29) were orderly shutdowns. That points at
  power or the board (the supply's power-good dropping, the CPU regulators' over-current or over-temperature trip, or
  a CPU internal error), not software or a plain RAM fault, which usually leave a trace.
- **The board:** ASUS PRIME H310M-K R2.0, BIOS 1005 (2019-08-07), legacy boot, i7-9700K, 30 GB. A budget board whose CPU
  regulators have no heatsink, under an 8-core 95 W part: a suspect, though 65 W should be within it.
- **Missing:** lm-sensors (no board voltages), stress-ng, memtester; no passwordless sudo. memtest86+ 8.00 is installed,
  but GRUB's menu is hidden (timeout 0): hold Shift at boot.
- **The raw runs are copied off** to `pyramidon:~/data/jebel-rescue-2026-11-09/` (`ab-runs/`: the `2026-11-04-thr-*` and
  `2026-11-05-pool-seq*` runs; `k24-profile/`). No `budget-*` or `pool-sequence*` runs were on its disk.

**The diagnosis (agreed with the owner, 2026-11-09):**
1. memtest86+, each stick alone (the owner, at the machine).
2. Instruments: `sudo apt install lm-sensors stress-ng`, `sudo sensors-detect --auto` (the owner).
3. A logger that outlives a reset: `pyramidon` reads `jebel`'s sensors, clocks and temperatures once a second over ssh,
   kept on `pyramidon`, so the seconds before a reset are kept.
4. A load ladder, the owner's OK before each step (each reset is unclean), up to 30 minutes a step or until it resets:

   | step | load | a reset here points at |
   |---|---|---|
   | a | 1 core, integer | the supply or the board, badly |
   | b | all 8 cores, integer | sustained power: supply or regulators |
   | c | all 8 cores, AVX2 (stress-ng matrix) | vector current: regulators or CPU voltage |
   | d | stress-ng `--vm`, 75% of RAM | RAM or the memory controller |
   | e | the benchmark at `--rust-threads 1` | its own load at one thread |
   | f | the benchmark at a 4 GHz cap | the top clocks |

   Only (c) or (e): cap frequency or voltage. Everything from (b): another supply.
5. BIOS: a newer version than 1005, memory at defaults (XMP off).

**Or replace it.** D12 then names the new machine. A new machine is tier 3: a full baseline, with its own noise rules.
klip is the nearest stand-in, quiet (1.1% median spread at one thread) but 4 cores, 5.3 GB and thermally guarded: a claim
there says less about a production machine, so it wasn't taken.

## The parked items

| # | item | what's known | why it waits |
|---|---|---|---|
| P1 | **The threshold re-swept per machine, on epochs** (K15). `thread_cost_probe.py epochs --threshold T` on each backdrop | The pool made threaded calls cheap: b512 rows -6% to -38% on `jebel`, few of which reach an epoch at today's 8M flops. Machine-specific (D5): 32 x 128 x 1352 gains 1.9x on `jebel`, 1.3x on stone, 1.16x on klip; on `litshe` 32 x 30 x 784 gains 3x, where it loses on x86. A per-machine rule needs building. The largest gain left | A claim |
| P2 | **The column split with packing for long products** (K15) | 512 x 2048 x 2048 is 3.07x behind OpenBLAS at 8 threads on `jebel`, where 8 threads run slower than 4, and OpenBLAS reaches 4.5x there, so the bus isn't what stops it | A claim |
| P3 | **More chunks than threads** (D3) | Today chunks = threads. Should even out stone's fast and slow cores | stone guards it; `jebel` claims |
| P4 | **New threaded sites, by measured share** (K14, K16): conv's forward over examples, the tiled attend over (example, head) and query blocks, the window program over examples, row-wise ops only where a profile asks | Studies' own workload: a window network at 4 processes gains x3.5 on klip, x2.4 on stone, x1.3 on `litshe`, so a thread can't assume it pays | A claim |
| P5 | **`litshe`'s default budget and threshold for aarch64** | Its best `d` 512 epoch is at 2 threads (47.2 s, against 50.0 at 4): large products contend for its bus. Goes with P1's rule | With P1 |
| P6 | **#115's sequence epoch on `jebel`,** still provisional: -7.6%, from `2026-11-05-pool-seq-solar`, at PL2 65 W with a temperature logger, on a failing machine. Re-read, then archive | klip's guard read -12.7% | The machine |
| P7 | **The closing baseline (K5, tier 3):** `jebel`'s per-op table and epochs, replacing `machine_profiles/i7-9700k.md`'s tables, with the Ryzen's columns. Those tables are from before the kernels, at PL2 120 W | Also settles kernel -42, which is unbaselined, and PL2 | The machine |
| P8 | **Placement (K2, K3, K4):** loops aligned to 32 bytes, as a full baseline; `litshe`'s cheaper pin; `learn linear k1 s16 m16 B64 W19` +4.2%, #1174's loader +2.3%, #1175's numpy causal forward +2.5%; K30's `prepare / rust` +5-7% on `jebel`, no epoch moved | All under the bar, or small. When a read needs them | A full baseline |
| P9 | **PL2 at 95 W** once the supply and machine are settled: the owner's next setting | 65 W may clip threaded bursts: unmeasured | The machine |

Stage 12's threshold for aarch64 is P5; the rest of stages 12 and 13 are done. K13, K17-K20 and K22 stay in next-steps; K21 (the forward context's 1.1%) was accepted.

## Decisions carried over

D2 (bits never depend on the thread count), D3 (the pool, built), D4 (one budget, built), D5 (the threshold measured
per machine), D6 (epochs decide), D12 (one claim machine, laptops guard), D13 (backdrops at `d` 512). Their options are in the
retired plan. **None is reopened by parking.**

## Where things are

- **Tools:** `scripts/measurement/thread_cost_probe.py` (`calls`, `scaling`, `epochs`), and `scripts/measurement/crate_perf/`:
  `ab_all.sh` (one change's A/Bs on all three machines), `thread_readings.sh`, `green.sh`, `golden_at.sh`. Read one before running it: they name these machines.
- **Readings:** indrajala-benchmarks' `readings/2026-11-05-crate-threads/`: every machine's pooled tables from stages 8 and 12, `litshe`'s curve, the window processes run, the temperature logs. Tier 2 runs through #115: its `runs/`.
- **On `jebel`'s disk only, while it's off:** raw runs not archived: `2026-11-04-thr-*`, `budget-*`, `pool-sequence*`, `2026-11-05-pool-seq*`; `~/data/indrajala-ml/runs/k24-profile/`. Their pooled tables are in the readings above, except the `pool-seq*` runs, whose results are P6's. **Copy them off if it comes back only briefly.**
