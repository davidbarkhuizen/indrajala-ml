# Workplan: machine profile, a JSON record of the benchmark machine, with a schema and a compare tool

Status: planned 2026-09-24, not started. Delete this workplan once it is implemented (as with
earlier workplans), moving anything worth keeping into the docs it touches.

## Context

Every timing in `docs/optimizations.md` comes from one machine, described only in prose ("This
machine": Ryzen 7 3700U laptop, 4 cores / 8 threads, 512 KB L2 per core, 4 MB L3, `schedutil`,
1.1-1.5 GHz idle to 3.8 GHz boost, `perf_event_paranoid` 4). Nothing checks that a new run happens
on the same machine and setup as the documented numbers. The user wants a script that captures
the machine's profile (CPU, memory, GPU, and so on) as a JSON file, a JSON schema describing that
file, and a way to confirm the profile hasn't changed between a new run and the docs.

Decisions (user, 2026-09-24):
- Split the profile into **identity** (compared: hardware, OS, cpufreq policy, software stack)
  and **state** (recorded only: clocks now, load, free memory, power, commits).
- Include the **full software stack**: Python, numpy and its BLAS build, thread env vars,
  rustc, the crate's build profile.
- A **standalone tool** (`profile`, `compare`). Nothing is wired into the timing scripts or
  made a test gate on the machine itself.
- Validate with the **`jsonschema`** package, added as a dependency.

Sources confirmed on this machine, all without root: `/proc/cpuinfo`, `/proc/meminfo`,
`/sys/devices/system/cpu/cpu*/cache/index*`, `/sys/devices/system/cpu/cpu0/cpufreq/*`
(`acpi-cpufreq`, `schedutil`, min 1.4 GHz / max 2.3 GHz base) and `/sys/devices/system/cpu/cpufreq/boost`
(1), `lspci -mm` (Radeon Vega, integrated), `/sys/class/power_supply` (AC0, BAT0),
`/proc/sys/kernel/perf_event_paranoid`, `/etc/os-release`, `numpy.show_config(mode="dicts")`
(numpy 2.2.6, scipy-openblas 0.3.29, `DYNAMIC_ARCH ... Haswell MAX_THREADS=64`), `rustc --version`
(1.75.0), and `rust/Cargo.toml` (no `[profile.release]`, so default). There is no `nvidia-smi`.
CI (GitHub `ubuntu-latest`) may lack cpufreq, lspci and power supplies, so every one of these
must be optional: `null` or an empty list, never an error.

## Files

- `indrajala_ml/machine_profile.py`: the logic. The collectors are small functions that take the
  file text or command output as an argument, so tests can feed them fixtures, plus
  `capture() -> dict`, `validate(profile)`, `compare(reference, current) -> list[Difference]`
  and `main()`.
- `indrajala_ml/machine_profile.schema.json`: JSON Schema Draft 2020-12. `additionalProperties:
  false` throughout, `schema_version` const 1, and nullable types where a source can be missing.
- `scripts/machine_profile.py`: a thin CLI, alongside the other measurement tools.
  - `python scripts/machine_profile.py profile [--out FILE]` captures and validates the profile,
    then writes JSON to the file or stdout.
  - `python scripts/machine_profile.py compare REFERENCE [CURRENT]` validates both profiles
    (capturing `CURRENT` now if it's omitted), prints each identity difference as `path:
    reference -> current`, prints the state fields side by side for context, and exits 0 if the
    identities match and 1 if they don't.
- `docs/machine_profiles/ryzen7-3700u.json`: the reference profile for the documented numbers,
  captured with the machine idle and on AC power.
- `docs/optimizations.md`: the "This machine" paragraph points to the reference profile and the
  `compare` command, and "How to measure" says to run `compare` before quoting new numbers.
- `pyproject.toml`: add `jsonschema` to dependencies. `requirements.txt`: pin `jsonschema` and
  its new transitive dependencies (attrs, jsonschema-specifications, referencing, rpds-py) at the
  installed versions, in pip-compile's format, leaving the existing pins alone.

## Profile shape

```
{
  "schema_version": 1,
  "identity": {
    "cpu": {"vendor", "model_name", "architecture", "sockets", "physical_cores", "logical_cpus",
            "threads_per_core",
            "caches": [{"level", "type", "size_kib", "shared_by_logical_cpus"}],
            "isa": {"sse4_2", "avx", "avx2", "fma", "avx512f"},   // booleans from cpuinfo flags
            "frequency": {"driver", "governor", "min_mhz", "max_mhz", "boost_enabled"}},  // nullable
    "memory": {"total_mib", "swap_total_mib"},
    "gpus": [{"class", "vendor", "device"}],                     // from lspci -mm; [] if unavailable
    "os": {"system", "kernel_release", "distribution", "perf_event_paranoid"},
    "software": {"python": {"implementation", "version"},
                 "numpy": {"version", "blas": {"name", "version", "configuration"}},
                 "rustc": "..." | null,
                 "crate_release_profile": {...} | "default",
                 "thread_env": {"OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS",
                                "RAYON_NUM_THREADS"}}                // null when unset
  },
  "state": {
    "captured_at": "ISO 8601 UTC",
    "hostname": "...",
    "load_average": [1, 5, 15],
    "memory_available_mib": ...,
    "cpu_mhz_now": {"min", "max"},
    "power": {"on_ac": bool | null, "battery_percent": int | null},
    "commits": {"repo": sha | null, "repo_dirty": bool | null, "rust": sha | null}
  }
}
```

Caches are read from sysfs and de-duplicated by (level, type, `shared_cpu_list`): the L2 is
per core and the L3 is shared. Kernel release is identity, because a kernel update can change
scheduler and cpufreq behaviour. If that proves too noisy it can move to state later.

## Tests (`tests/test_machine_profile.py`)

- `capture()` on whatever machine runs the tests (CI included) validates against the schema.
- The committed reference profile validates.
- The parsers, fed fixture text, give the expected fields: a cpuinfo excerpt with flags, sysfs
  cache entries (per-core L2 against a shared L3), an `lspci -mm` line, a meminfo excerpt,
  missing cpufreq, and no power supply.
- `compare`: identical identities give no differences. Changing a nested identity field (the
  governor, an ISA flag, the numpy BLAS version) reports exactly that path. Changing any state
  field reports nothing. A list difference (an extra GPU or cache) is reported.
- The schema rejects a missing required key, an unknown key, and a wrong `schema_version`.
- CLI: `profile --out` then `compare ref out` exits 0 against itself, and exits 1 with the path
  printed after an identity field is edited.
- There is deliberately no test that fails when run on a different machine (the user chose a
  standalone tool).

## Verification

- `./cli test` passes locally, and in CI, where capture runs on a machine very different from
  this one.
- Run `profile` here and read the output against the prose description (4 cores / 8 threads,
  512 KB L2 x4, 4 MB L3, `schedutil`, boost on). Then run `compare
  docs/machine_profiles/ryzen7-3700u.json`, which should exit 0.
- Set `OPENBLAS_NUM_THREADS=1` and run `compare` again; it should exit 1 and name
  `identity.software.thread_env.OPENBLAS_NUM_THREADS`.

One PR: squash-merge after CI, then sync `main`.
