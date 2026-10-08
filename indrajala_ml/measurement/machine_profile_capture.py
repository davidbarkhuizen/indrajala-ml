"""
Capturing the benchmark machine's profile (machine_profile.py has its schema and comparison).

Every source is optional (CI runners may have no cpufreq, lspci or power supply): a missing one
gives null or an empty list, never an error. The collectors below take the file text or command
output as an argument so they can be tested from fixtures; capture() reads the real sources.
"""

import datetime
import os
import platform
import shlex
import socket
import subprocess
import tomllib
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

SCHEMA_VERSION = 4
REPO_ROOT = Path(__file__).resolve().parents[2]
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
    except OSError, subprocess.SubprocessError:
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
    missing), the global boost file (key "boost") and intel_pstate's status and no_turbo files
    (keys "intel_pstate_status", "no_turbo"). None if there is no cpufreq at all.

    Turbo is AMD's boost or cpb file, or under intel_pstate (which has neither) no_turbo inverted.
    The energy-performance preference (EPP) decides how fast a hardware-managed core clocks up,
    as the governor does elsewhere."""
    if files.get("scaling_driver") is None and files.get("scaling_governor") is None:
        return None

    def mhz(name: str) -> int | None:
        text = files.get(name)
        return int(text.strip()) // 1000 if text is not None else None

    def stripped(name: str) -> str | None:
        text = files.get(name)
        return text.strip() if text is not None else None

    boost = files.get("boost")
    if boost is None:
        boost = files.get("cpb")
    if boost is not None:
        boost_enabled: bool | None = boost.strip() == "1"
    elif (no_turbo := files.get("no_turbo")) is not None:
        boost_enabled = no_turbo.strip() == "0"
    else:
        boost_enabled = None
    return {
        "driver": driver.strip() if (driver := files.get("scaling_driver")) else None,
        "governor": governor.strip() if (governor := files.get("scaling_governor")) else None,
        "min_mhz": mhz("cpuinfo_min_freq"),
        "max_mhz": mhz("cpuinfo_max_freq"),
        "boost_enabled": boost_enabled,
        "energy_performance_preference": stripped("energy_performance_preference"),
        "intel_pstate_status": stripped("intel_pstate_status"),
    }


def parse_power_limits(constraints: Mapping[str, Mapping[str, str | None]]) -> JSONObject | None:
    """The package power limits in watts from RAPL's package-0 zone (a dict of constraint index
    -> dict of "name" and "power_limit_uw" text): PL1 (long_term) and PL2 (short_term). None
    without a RAPL package zone. The limit, not the clock policy, decides how fast a long
    all-core load runs."""
    watts: dict[str, int] = {}
    for files in constraints.values():
        name, limit = files.get("name"), files.get("power_limit_uw")
        if name is not None and limit is not None:
            watts[name.strip()] = int(limit.strip()) // 1_000_000
    if not watts:
        return None
    return {"long_term_w": watts.get("long_term"), "short_term_w": watts.get("short_term")}


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
    names = [
        "scaling_driver",
        "scaling_governor",
        "cpuinfo_min_freq",
        "cpuinfo_max_freq",
        "cpb",
        "energy_performance_preference",
    ]
    files = {name: read_text(cpufreq / name) for name in names}
    files["boost"] = read_text("/sys/devices/system/cpu/cpufreq/boost")
    intel_pstate = Path("/sys/devices/system/cpu/intel_pstate")
    files["intel_pstate_status"] = read_text(intel_pstate / "status")
    files["no_turbo"] = read_text(intel_pstate / "no_turbo")
    return files


def _power_limit_constraints() -> dict[str, dict[str, str | None]]:
    zone = Path("/sys/class/powercap/intel-rapl:0")
    if (read_text(zone / "name") or "").strip() != "package-0":
        return {}
    constraints: dict[str, dict[str, str | None]] = {}
    for index in range(2):
        files = {name: read_text(zone / f"constraint_{index}_{name}") for name in ("name", "power_limit_uw")}
        constraints[str(index)] = files
    return constraints


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


def power_policy() -> JSONObject:
    """The frequency policy and package power limits as they are now: the part of
    identity.cpu that software can change while the machine is up (thermald rewrites RAPL's
    PL1), so ab.py re-reads it after every pass."""
    return {
        "frequency": parse_frequency(_frequency_files()),
        "power_limits": parse_power_limits(_power_limit_constraints()),
    }


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
                **power_policy(),
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
            "captured_at": datetime.datetime.now(datetime.UTC)
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
