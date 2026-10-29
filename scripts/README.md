# Scripts

Command-line tools, grouped by purpose. Run each from the repository root
(`.venv/bin/python scripts/<group>/<script>.py ...`). A group's scripts may import each other (their
directory is on `sys.path` when run); no script imports another group's.

| group | what |
| --- | --- |
| `measurement/` | timing and its protocol ([docs/measurement.md](../docs/measurement.md)): `ab.py`, the benchmarks it runs, the machine profile and setup, parity, thermal and thread-cost probes, the builds' machine code compared (`kernel_code_diff.py`), the golden run |
| `runner/` | the batch runner (`batch.py`) and the thermal guard it wraps `klip`'s attempts in |
| `studies/` | one script per study, the modular family's probe and its readings; `configs/` holds the studies' run configurations |
| `paper/` | paper 1's tables, its batch readings, its datapack and the Zenodo deposit |
| `data/` | datasets fetched and checked, and the migration of files written before the factored state |

Grouped 2026-10-29. A commit before then has every script flat in `scripts/`; `ab.py` runs either
layout's scripts, so an A/B against an older commit still works. Paper 1's tags (`paper1-v1`,
`paper1-v1.1`) and its datapack's how-to-verify page use the flat layout.
