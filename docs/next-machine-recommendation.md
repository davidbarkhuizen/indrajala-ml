# The next machine: a recommendation

A recommendation for the next development and benchmark machine (CPU, motherboard, RAM and GPU),
written 2026-10-08. It's a buying guide, not a plan: nothing in the code changes for this
machine before it arrives (see [i7-9700k-rust-optimization.md](i7-9700k-rust-optimization.md),
"Kernels and parameters from the machine").

## The brief

- **Budget:** $2000 for CPU, motherboard, RAM and GPU, bought in South Africa in rand: about
  R33,000-34,000 at R16.4-17 to the dollar.
- **AVX-512** for the crate's next kernel generation.
- **A GPU** modern and strong enough for later GPU work.
- **A useful third data point** next to the i7-9700K (`jebel`: 8 cores, no SMT, 256 KB L2,
  12 MB L3) and the Ryzen 7 3700U laptop (`pyramidon`: 4 cores with SMT, 512 KB L2, 4 MB L3)
  for the `CpuInfo` formulas.

## Recommendation: AMD AM5, Ryzen 7 9700X (Zen 5)

Prices are South African retail in rand, 2026, VAT included. RAM prices are volatile and the
listings disagree, so check them before buying.

| part | choice | approx. price |
| --- | --- | --- |
| CPU | Ryzen 7 9700X (8 cores, 16 threads) | R6,500 |
| motherboard | B850, e.g. ASUS TUF Gaming B850M-Plus WiFi | R3,300-4,200 |
| RAM | 64 GB DDR5-6000 CL30 (2 x 32 GB); 32 GB saves about R3,500 | R6,900 (32 GB: R2,400-3,800) |
| GPU | RTX 5060 Ti 16 GB | R12,500-14,000 |
| **total** | | **about R29,000-31,500** |

**Why:**

- **Full-width AVX-512.** Zen 5 runs 512-bit operations natively; Zen 4 double-pumps them on
  256-bit units. Intel's current desktop parts (Core Ultra 200S) have no AVX-512, so they would
  add nothing the i7 doesn't already cover.
- **8 cores, like the i7.** A clean comparison against `jebel`: the same core count, but with
  SMT, a 1 MB L2 per core and a 32 MB L3. It tests the `CpuInfo` formulas on a third cache
  hierarchy.
- **One chiplet (CCD).** The 16-core parts span two CCDs, and the latency between them is a
  noise source for threading measurements. The 9700X has one.
- **65 W by default.** Thermals are easy to hold steady, unlike the i7, which needed PL1 capped
  at 65 W to stop throttling.
- **The AM5 upgrade path.** AMD has said AM5 carries on, so a later Zen 6 part should drop into
  the same board. Zen 6's timing isn't confirmed.
- **NVIDIA for the GPU.** CUDA has the most mature tooling, and 16 GB of VRAM is the useful
  minimum. Every consumer GPU runs f64 at a small fraction of its f32 rate, so a GPU backend
  means f32 and a tolerance contract whichever card is bought.

## Alternatives

| option | parts | approx. total | when to choose it |
| --- | --- | --- | --- |
| more GPU | 9700X, B850, 32 GB, RTX 5070 Ti 16 GB (R19,500-21,000) | R32,500-34,000 | GPU work becomes the main focus soon. At the edge of the budget, and the VRAM is still 16 GB. |
| more cores | Ryzen 9 9950X (about R11,200), B850, 32 GB, RTX 5060 Ti 16 GB | about R30,500 | Large CPU training workloads matter more than clean kernel benchmarks: 16 cores, but two CCDs. |

## Not in the budget

- a PSU: about 650 W with a 5060 Ti, 750 W with a 5070 Ti
- a CPU cooler: the 9700X ships without one
- a case
- an NVMe drive

## When it arrives

- **Its own campaign:** a machine profile with its own `noise_rules`, an A/A noise table and the
  per-op table, as the benchmark machine workplan did for the i7.
- **AVX-512 kernels** go behind the crate's existing ISA dispatch. They keep `dot_product`'s
  4-lane grouping (two 4-lane chains per register), so the golden run stays bit-identical with
  the other machines.
- **Whether the 512-bit path pays** is measured on this machine, not assumed from the feature
  flag.

## Sources

- [Evetech: Ryzen 9 9950X](https://www.evetech.co.za/amd-ryzen-9-9950x-processor/best-deal/20990)
- [tech.co.za: Ryzen 9 9950X](https://tech.co.za/product/amd-ryzen-9-9950x-16-core-4-3ghz-am5-cpu/)
- [Computersonly: Ryzen 7 9700X](https://computersonly.co.za/product/amd-ryzen-7-9700x-8-core-5-5ghz/)
- [Evetech: B850 motherboards](https://www.evetech.co.za/amd-b850-motherboards/x/1696)
- [Evetech: RTX 5070 Ti prices](https://evezone.evetech.co.za/build-lab/rtx-5070-ti-price-south-africa)
- [Evetech: RTX 5060 Ti](https://www.evetech.co.za/nvidia-geforce-rtx-5060-ti/x/1729)
- [Wootware: RTX 5060 Ti](https://www.wootware.co.za/computer-hardware/video-cards-video-devices/shopby/geforce_rtx_5060_ti)
- [Evetech: RAM prices in 2026](https://www.evetech.co.za/ram-prices-rising-2026-sa-buyers/e/4021)
- [Evetech: 32 GB DDR5](https://www.evetech.co.za/32gb-ddr5-ram-south-africa/x/2264)
- [Long Forecast: USD/ZAR](https://longforecast.com/dollar-to-rand-forecast-2017-2018-2019-2020-2021-usd-to-zar)
- [ACEMAGIC: Ryzen 5 9600X against Core Ultra 5 250K Plus](https://acemagic.com/blogs/product-reviews/ryzen-5-9600x-vs-intel-core-ultra-5-250k-plus)
