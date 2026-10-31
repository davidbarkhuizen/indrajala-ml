# Scripts

Command-line tools, grouped by purpose. Run each from the repository root
(`.venv/bin/python scripts/<group>/<script>.py ...`). A group's scripts may import each other (their
directory is on `sys.path` when run); no script imports another group's.

| group | what |
| --- | --- |
| `measurement/` | timing and its protocol ([docs/measurement.md](../docs/measurement.md)): `ab.py`, the benchmarks it runs, the machine profile and setup, parity, thermal and thread-cost probes, the builds' machine code compared (`kernel_code_diff.py`), the golden run, the DRY audit's clone scan (`duplicates.py`) |
| `runner/` | the batch runner (`batch.py`) and the thermal guard it wraps `klip`'s attempts in |
| `studies/` | one script per study and the modular family's probe; `configs/` holds the window-recurrent run configurations |
| `data/` | datasets fetched and checked |

Grouped 2026-10-29. A commit before then has every script flat in `scripts/`; `ab.py` runs either
layout's scripts, so an A/B against an older commit still works. Paper 1's tags (`paper1-v1`,
`paper1-v1.1`) and its datapack's how-to-verify page use the flat layout.

Moved to [indrajala-studies](https://github.com/davidbarkhuizen/indrajala-studies) on 2026-10-31 (the studies-repo workplan): paper 1's scripts (`paper/`), the
addition study's and the window-recurrent study's scripts, `window_program_check.py`,
`profile_page.py`, the addition configs and `migrate_factored_state.py`. It runs the
window-recurrent configs here from its `ml/` submodule.
