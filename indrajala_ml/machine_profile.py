"""
The benchmark machine's profile: a JSON record of the hardware, OS and software stack the timings
in docs/optimizations.md were measured on, a schema for it, and a comparison of two profiles.

The profile has two parts:
- identity: what a timing depends on and should not change between a documented number and a
  new run (CPU, caches, ISA, cpufreq policy, memory, GPUs, OS, the Python/numpy/BLAS/Rust stack
  and the thread env vars). compare() reports every difference here.
- state: what is expected to change from run to run (clocks now, load, free memory, power,
  commits). It is recorded for context and never compared.

Every source is optional (CI runners may have no cpufreq, lspci or power supply): a missing one
gives null or an empty list, never an error. The collectors below take the file text or command
output as an argument so they can be tested from fixtures; capture() reads the real sources.

    python scripts/machine_profile.py profile [--out FILE]
    python scripts/machine_profile.py compare REFERENCE [CURRENT]
"""

import argparse
import datetime
import json
import os
import platform
import shlex
import socket
import subprocess
import sys
from collections.abc import Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, cast

if sys.version_info >= (3, 11):
    import tomllib
else:
    import tomli as tomllib

SCHEMA_VERSION = 1
SCHEMA_PATH = Path(__file__).with_name("machine_profile.schema.json")
REPO_ROOT = Path(__file__).resolve().parent.parent
RUST_ROOT = REPO_ROOT / "rust"

ISA_FLAGS = ["sse4_2", "avx", "avx2", "fma", "avx512f"]
THREAD_ENV_VARS = ["OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "RAYON_NUM_THREADS"]
GPU_CLASSES = {"VGA compatible controller", "3D controller", "Display controller"}

# a JSON object: the profile, and the sections the collectors return
JSONObject = dict[str, Any]


def read_text(path: str | Path) -> str | None:
    """The file's text, or None if it can't be read."""
    try:
        return Path(path).read_text()
    except OSError:
        return None


def run_command(args: Sequence[str], cwd: str | Path | None = None) -> str | None:
    """The command's stripped stdout, or None if it can't be run or fails."""
    try:
        result = subprocess.run(args, cwd=cwd, capture_output=True, text=True, timeout=10, check=False)
    except (OSError, subprocess.SubprocessError):
        return None
    if result.returncode != 0:
        return None
    return result.stdout.strip()


# --- collectors (text in, fields out) ---


def parse_cpuinfo(text: str) -> JSONObject:
    """vendor, model_name, sockets, physical_cores, logical_cpus and ISA flags from
    /proc/cpuinfo. Fields the text lacks are None."""
    processors: list[dict[str, str]] = []
    current: dict[str, str] = {}
    for line in text.splitlines():
        if not line.strip():
            if current:
                processors.append(current)
                current = {}
            continue
        key, _, value = line.partition(":")
        current[key.strip()] = value.strip()
    if current:
        processors.append(current)
    processors = [p for p in processors if "processor" in p]

    first: dict[str, str] = processors[0] if processors else {}
    flags = set(first.get("flags", "").split())
    packages = {p.get("physical id") for p in processors if "physical id" in p}
    cores = {(p.get("physical id"), p.get("core id")) for p in processors if "core id" in p}
    return {
        "vendor": first.get("vendor_id"),
        "model_name": first.get("model name"),
        "sockets": len(packages) or None,
        "physical_cores": len(cores) or None,
        "logical_cpus": len(processors) or None,
        "isa": {flag: flag in flags for flag in ISA_FLAGS},
    }


def parse_cpu_list(text: str) -> int:
    """The number of CPUs in a sysfs CPU list such as "0-3,8"."""
    count = 0
    for part in text.strip().split(","):
        if not part:
            continue
        low, _, high = part.partition("-")
        count += int(high or low) - int(low) + 1
    return count


def parse_size_kib(text: str) -> int:
    """A sysfs cache size ("512K", "4096K", "8M") in KiB."""
    text = text.strip()
    units = {"K": 1, "M": 1024, "G": 1024 * 1024}
    if text[-1] in units:
        return int(text[:-1]) * units[text[-1]]
    return int(text) // 1024


def summarize_caches(entries: Iterable[Mapping[str, str]]) -> list[JSONObject]:
    """One entry per distinct cache, from every CPU's sysfs index* entries (dicts of level,
    type, size and shared_cpu_list text). A cache shared by several CPUs appears once per CPU in
    sysfs, so entries are first de-duplicated by (level, type, shared_cpu_list); identical caches
    (the per-core L2s) are then counted as instances of one entry."""
    distinct: dict[tuple[int, str, str], Mapping[str, str]] = {}
    for entry in entries:
        key = (int(entry["level"]), entry["type"].strip(), entry["shared_cpu_list"].strip())
        distinct[key] = entry
    summary: dict[tuple[int, str, int, int], int] = {}
    for (level, cache_type, shared), entry in distinct.items():
        described = (level, cache_type, parse_size_kib(entry["size"]), parse_cpu_list(shared))
        summary[described] = summary.get(described, 0) + 1
    return [
        {"level": level, "type": cache_type, "size_kib": size, "shared_by_logical_cpus": shared, "instances": instances}
        for (level, cache_type, size, shared), instances in sorted(summary.items())
    ]


def parse_frequency(files: Mapping[str, str | None]) -> JSONObject | None:
    """The cpufreq policy from cpu0's cpufreq files (a dict of file name -> text, None when
    missing) and the global boost file (key "boost"). None if there is no cpufreq at all."""
    if files.get("scaling_driver") is None and files.get("scaling_governor") is None:
        return None

    def mhz(name: str) -> int | None:
        text = files.get(name)
        return int(text.strip()) // 1000 if text is not None else None

    boost = files.get("boost")
    if boost is None:
        boost = files.get("cpb")
    return {
        "driver": driver.strip() if (driver := files.get("scaling_driver")) else None,
        "governor": governor.strip() if (governor := files.get("scaling_governor")) else None,
        "min_mhz": mhz("cpuinfo_min_freq"),
        "max_mhz": mhz("cpuinfo_max_freq"),
        "boost_enabled": boost.strip() == "1" if boost is not None else None,
    }


def parse_meminfo(text: str) -> dict[str, int]:
    """/proc/meminfo's fields in MiB (the file gives kB)."""
    fields: dict[str, int] = {}
    for line in text.splitlines():
        key, _, value = line.partition(":")
        parts = value.split()
        if parts and parts[0].isdigit():
            fields[key.strip()] = int(parts[0]) // 1024
    return fields


def parse_lspci(text: str) -> list[dict[str, str]]:
    """The display controllers in `lspci -mm` output. Each line is: slot "class" "vendor"
    "device", optional -rXX / -pXX flags, then the subsystem vendor and device."""
    gpus: list[dict[str, str]] = []
    for line in text.splitlines():
        fields = [f for f in shlex.split(line) if not f.startswith("-")]
        if len(fields) >= 4 and fields[1] in GPU_CLASSES:
            gpus.append({"class": fields[1], "vendor": fields[2], "device": fields[3]})
    return gpus


def parse_os_release(text: str) -> str | None:
    """PRETTY_NAME from /etc/os-release."""
    for line in text.splitlines():
        key, _, value = line.partition("=")
        if key == "PRETTY_NAME":
            return value.strip().strip('"')
    return None


def parse_power(supplies: Mapping[str, Mapping[str, str | None]]) -> JSONObject:
    """on_ac and battery_percent from /sys/class/power_supply (a dict of supply name -> dict of
    file name -> text). Either is None when there is no such supply."""
    on_ac: bool | None = None
    battery_percent: int | None = None
    for files in supplies.values():
        supply_type = (files.get("type") or "").strip()
        if supply_type == "Mains" and (online := files.get("online")) is not None:
            on_ac = (on_ac or False) or online.strip() == "1"
        elif supply_type == "Battery" and (capacity := files.get("capacity")) is not None:
            battery_percent = int(capacity.strip())
    return {"on_ac": on_ac, "battery_percent": battery_percent}


def parse_blas(config: Mapping[str, Any] | None) -> JSONObject | None:
    """The BLAS numpy was built against, from numpy.show_config(mode="dicts")."""
    blas: Mapping[str, Any] | None = (config or {}).get("Build Dependencies", {}).get("blas")
    if not blas:
        return None
    return {
        "name": blas.get("name"),
        "version": blas.get("version"),
        "configuration": blas.get("openblas configuration"),
    }


def parse_release_profile(cargo_toml: str) -> JSONObject | str:
    """The crate's [profile.release] table, or "default" if it sets none."""
    profile: JSONObject | None = tomllib.loads(cargo_toml).get("profile", {}).get("release")
    return profile if profile else "default"


# --- reading the real sources ---


def _cpu_dirs() -> list[Path]:
    root = Path("/sys/devices/system/cpu")
    return sorted(p for p in root.glob("cpu[0-9]*") if p.name[3:].isdigit())


def _cache_entries() -> list[dict[str, str]]:
    entries: list[dict[str, str]] = []
    for cpu in _cpu_dirs():
        for index in sorted(cpu.glob("cache/index*")):
            files = {name: read_text(index / name) for name in ("level", "type", "size", "shared_cpu_list")}
            present = {name: text for name, text in files.items() if text is not None}
            if len(present) == len(files):
                entries.append(present)
    return entries


def _frequency_files() -> dict[str, str | None]:
    cpufreq = Path("/sys/devices/system/cpu/cpu0/cpufreq")
    names = ["scaling_driver", "scaling_governor", "cpuinfo_min_freq", "cpuinfo_max_freq", "cpb"]
    files = {name: read_text(cpufreq / name) for name in names}
    files["boost"] = read_text("/sys/devices/system/cpu/cpufreq/boost")
    return files


def _cpu_mhz_now() -> dict[str, int] | None:
    values: list[int] = []
    for cpu in _cpu_dirs():
        text = read_text(cpu / "cpufreq" / "scaling_cur_freq")
        if text is not None:
            values.append(int(text.strip()) // 1000)
    if not values:
        return None
    return {"min": min(values), "max": max(values)}


def _power_supplies() -> dict[str, dict[str, str | None]]:
    supplies: dict[str, dict[str, str | None]] = {}
    for supply in sorted(Path("/sys/class/power_supply").glob("*")):
        supplies[supply.name] = {name: read_text(supply / name) for name in ("type", "online", "capacity")}
    return supplies


def _numpy_software() -> JSONObject | None:
    try:
        import numpy
    except ImportError:
        return None
    try:
        config: Mapping[str, Any] | None = numpy.show_config(mode="dicts")
    except TypeError:  # numpy < 1.26 has no mode argument
        config = None
    return {"version": numpy.__version__, "blas": parse_blas(config)}


def _git_commit(path: Path) -> str | None:
    return run_command(["git", "rev-parse", "HEAD"], cwd=path)


def _git_dirty(path: Path) -> bool | None:
    status = run_command(["git", "status", "--porcelain"], cwd=path)
    return None if status is None else bool(status)


def _perf_event_paranoid() -> int | None:
    text = read_text("/proc/sys/kernel/perf_event_paranoid")
    return int(text.strip()) if text is not None else None


def capture() -> JSONObject:
    """The profile of the machine this runs on."""
    # clocks first: the rest (numpy's import, git, rustc, lspci) boosts the cores within ms
    cpu_mhz_now = _cpu_mhz_now()
    cpuinfo = parse_cpuinfo(read_text("/proc/cpuinfo") or "")
    logical: int | None = cpuinfo["logical_cpus"] or os.cpu_count()
    physical: int | None = cpuinfo["physical_cores"]
    meminfo = parse_meminfo(read_text("/proc/meminfo") or "")
    lspci = run_command(["lspci", "-mm"])
    os_release = read_text("/etc/os-release")
    cargo_toml = read_text(RUST_ROOT / "Cargo.toml")
    load = os.getloadavg() if hasattr(os, "getloadavg") else None

    return {
        "schema_version": SCHEMA_VERSION,
        "identity": {
            "cpu": {
                "vendor": cpuinfo["vendor"],
                "model_name": cpuinfo["model_name"],
                "architecture": platform.machine(),
                "sockets": cpuinfo["sockets"],
                "physical_cores": physical,
                "logical_cpus": logical,
                "threads_per_core": logical // physical if logical and physical else None,
                "caches": summarize_caches(_cache_entries()),
                "isa": cpuinfo["isa"],
                "frequency": parse_frequency(_frequency_files()),
            },
            "memory": {
                "total_mib": meminfo.get("MemTotal"),
                "swap_total_mib": meminfo.get("SwapTotal"),
            },
            "gpus": parse_lspci(lspci) if lspci is not None else [],
            "os": {
                "system": platform.system(),
                "kernel_release": platform.release(),
                "distribution": parse_os_release(os_release) if os_release else None,
                "perf_event_paranoid": _perf_event_paranoid(),
            },
            "software": {
                "python": {
                    "implementation": platform.python_implementation(),
                    "version": platform.python_version(),
                },
                "numpy": _numpy_software(),
                # run in the crate, so rustup reports rust-toolchain.toml's pinned toolchain
                "rustc": run_command(["rustc", "--version"], cwd=RUST_ROOT),
                "crate_release_profile": parse_release_profile(cargo_toml) if cargo_toml else None,
                "thread_env": {name: os.environ.get(name) for name in THREAD_ENV_VARS},
            },
        },
        "state": {
            "captured_at": datetime.datetime.now(datetime.timezone.utc)
            .replace(microsecond=0)
            .isoformat()
            .replace("+00:00", "Z"),
            "hostname": socket.gethostname(),
            "load_average": [round(value, 2) for value in load] if load else None,
            "memory_available_mib": meminfo.get("MemAvailable"),
            "cpu_mhz_now": cpu_mhz_now,
            "power": parse_power(_power_supplies()),
            "commits": {
                "repo": _git_commit(REPO_ROOT),
                "repo_dirty": _git_dirty(REPO_ROOT),
                "rust": _git_commit(RUST_ROOT),
            },
        },
    }


# --- schema and comparison ---


def load_schema() -> JSONObject:
    return json.loads(SCHEMA_PATH.read_text())


def validate(profile: JSONObject) -> None:
    """Raise jsonschema.ValidationError if the profile doesn't match the schema."""
    import jsonschema

    jsonschema.validate(profile, load_schema(), cls=jsonschema.Draft202012Validator)


@dataclass(frozen=True)
class Difference:
    path: str
    reference: object
    current: object

    def __str__(self) -> str:
        return f"{self.path}: {json.dumps(self.reference)} -> {json.dumps(self.current)}"


def _as_json_object(value: object) -> dict[str, object] | None:
    # a parsed JSON value as an object (string keys), or None if it isn't one
    return cast("dict[str, object]", value) if isinstance(value, dict) else None


def _as_json_list(value: object) -> list[object] | None:
    return cast("list[object]", value) if isinstance(value, list) else None


def _differences(path: str, reference: object, current: object) -> Iterator[Difference]:
    reference_object, current_object = _as_json_object(reference), _as_json_object(current)
    reference_list, current_list = _as_json_list(reference), _as_json_list(current)
    if reference_object is not None and current_object is not None:
        for key in sorted(set(reference_object) | set(current_object)):
            yield from _differences(f"{path}.{key}", reference_object.get(key), current_object.get(key))
    elif reference_list is not None and current_list is not None and len(reference_list) == len(current_list):
        for index, (ref_item, cur_item) in enumerate(zip(reference_list, current_list)):
            yield from _differences(f"{path}[{index}]", ref_item, cur_item)
    elif reference != current:
        yield Difference(path, reference, current)


def compare(reference: JSONObject, current: JSONObject) -> list[Difference]:
    """Every leaf of the identity that differs, as dotted paths. Lists of different lengths
    (an extra GPU or cache) are reported whole. State is never compared."""
    return list(_differences("identity", reference["identity"], current["identity"]))


def _flatten(prefix: str, value: object) -> Iterator[tuple[str, object]]:
    value_object = _as_json_object(value)
    if value_object is not None:
        for key, item in value_object.items():
            yield from _flatten(f"{prefix}.{key}", item)
    else:
        yield prefix, value


# --- CLI ---


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").split("\n\n")[0])  # None under -OO
    commands = parser.add_subparsers(dest="command", required=True)
    profile_parser = commands.add_parser("profile", help="capture this machine's profile")
    profile_parser.add_argument("--out", help="write the JSON here instead of stdout")
    compare_parser = commands.add_parser("compare", help="compare two profiles' identities; exit 1 if they differ")
    compare_parser.add_argument("reference", help="the reference profile (JSON)")
    compare_parser.add_argument("current", nargs="?", help="the profile to check (default: capture one now)")
    args = parser.parse_args(argv)

    if args.command == "profile":
        profile = capture()
        validate(profile)
        text = json.dumps(profile, indent=2) + "\n"
        if args.out:
            Path(args.out).write_text(text)
        else:
            sys.stdout.write(text)
        return 0

    reference = json.loads(Path(args.reference).read_text())
    validate(reference)
    current = json.loads(Path(args.current).read_text()) if args.current else capture()
    validate(current)

    differences = compare(reference, current)
    current_state = dict(_flatten("state", current["state"]))
    print("state (recorded, not compared): reference | current")
    for path, value in _flatten("state", reference["state"]):
        print(f"  {path}: {json.dumps(value)} | {json.dumps(current_state.get(path))}")
    if differences:
        print(f"identity differs ({len(differences)}):")
        for difference in differences:
            print(f"  {difference}")
        return 1
    print("identity matches")
    return 0
