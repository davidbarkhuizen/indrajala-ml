# Probe launches

The shell scripts that launched the probes and studies run by hand, before the batch runner
(`scripts/batch.py`) took over launching, placing and moving runs. They are kept **verbatim**, as
they ran from `~/code/` on `pyramidon` (copied 2026-10-29, timestamps kept). Their paths
(`~/code/wt-*`, `~/labour/`, scratchpad sites, process ids) are the ones of that time and no longer
resolve. They are records, not tools: don't run them.

Many headers are a probe's **registration**: its population, cell, feasibility and expectations,
written before its data. The workplans cite them by name, and D63's and D67's by sha256
(`probe-d63.sh` b32f5c2995709114..., `probe-d67.sh` 411f31578d966b10...), which these copies keep.
Retired workplans are in `docs/next-steps.md`'s "Retired workplans" and in git history.

| scripts | what | workplan |
| --- | --- | --- |
| `probe-4a.sh`, `probe-4a2.sh`, `probe-4a3.sh`, `probe-4a3-resume.sh` | stages 4a, 4a', 4a'' (D14-D25) | state-formation |
| `probe-d26.sh`, `probe-d29.sh` | stages 4a''' and 4a'''' (D26-D30) | state-formation |
| `panel-pyramidon.sh`, `panel-cost.sh` | option 1's panel (stage 4, D11), the panel's cost (D36) | state-formation |
| `probe-d32.sh`, `probe-d33.sh`, `probe-d33-rest.sh`, `probe-d34.sh`, `probe-d38.sh` | D32-D34, D38 | state-formation |
| `probe-4b-core.sh` | the core matrix (4b-core) | state-formation |
| `probe-d54.sh` ... `probe-d59.sh`, `probe-d61.sh`, `probe-d63.sh`, `probe-d64.sh` | D54-D64 | state-formation |
| `probe-d67.sh`, `move-d67-mtf4-when-free.sh`, `move-d67-mtf4-to-jebel.sh` | the deadline's completion (D67), two runs moved to `jebel` | state-formation |
| `snap-wall-p1.sh`, `replay-guard.sh` | P1, the certificate guard | snap-wall |
| `algebras-round1.sh` | round 1 (E1) | general-algebras |
| `thermostat-probe.sh`, `thermostat-probe-2.sh`, `anneal-probe.sh`, `anneal-probe-stone.sh`, `never-probe-2.sh`, `never-harm.sh`, `never-scorecard.sh`, `never-scorecard-division.sh` | the thermostat, annealing and the commitment off | commitment-control (retired) |
| `approx-pyramidon.sh` | the approximation claim, `pyramidon`'s share | approximation-claim (retired) |
| `stage5-post-launch.sh`, `stage5-long-tests.sh`, `stage5-long-tests-finish.sh`, `fork-long-tests.sh`, `p11-long-tests.sh` | division's stage 5 and its long tests (D4, D8); the fork probe's and the p 11 ladder's long tests | division (retired) |
| `e2b-snapped-launch.sh`, `e2b-long-tests.sh`, `e2b-prime-long-tests.sh`, `e2b-long-nets*.txt`, `e2b-prime-long-nets-20.txt` | E2b and E2b', their long tests and the networks tested | the claims workplan (retired) |

Not kept: `~/code/merge-779.sh` and `stone-slots0.sh` (one-off operations) and
`collapse-drops-analysis.py` (the `drops` reading replaced it). `e2-launch.sh`, which the working
notes name, was already gone.
