import copy
import json
import subprocess
import sys
from pathlib import Path

import jsonschema
import pytest

from indrajala_ml import machine_profile as mp

REFERENCE_PATH = Path(__file__).resolve().parent.parent / "docs/machine_profiles/ryzen7-3700u.json"
SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts/machine_profile.py"

CPUINFO = """\
processor\t: 0
vendor_id\t: AuthenticAMD
model name\t: AMD Ryzen 7 3700U with Radeon Vega Mobile Gfx
physical id\t: 0
core id\t\t: 0
flags\t\t: fpu sse4_2 avx fma avx2

processor\t: 1
vendor_id\t: AuthenticAMD
model name\t: AMD Ryzen 7 3700U with Radeon Vega Mobile Gfx
physical id\t: 0
core id\t\t: 0
flags\t\t: fpu sse4_2 avx fma avx2

processor\t: 2
vendor_id\t: AuthenticAMD
model name\t: AMD Ryzen 7 3700U with Radeon Vega Mobile Gfx
physical id\t: 0
core id\t\t: 1
flags\t\t: fpu sse4_2 avx fma avx2

processor\t: 3
vendor_id\t: AuthenticAMD
model name\t: AMD Ryzen 7 3700U with Radeon Vega Mobile Gfx
physical id\t: 0
core id\t\t: 1
flags\t\t: fpu sse4_2 avx fma avx2
"""

MEMINFO = """\
MemTotal:        6012300 kB
MemFree:         1000000 kB
MemAvailable:    4419584 kB
SwapTotal:       2097148 kB
HugePages_Total:       0
"""

LSPCI = (
    '00:00.0 "Host bridge" "Advanced Micro Devices, Inc. [AMD]" "Raven/Raven2 Root Complex" '
    '"Advanced Micro Devices, Inc. [AMD]" "Raven/Raven2 Root Complex"\n'
    '03:00.0 "VGA compatible controller" "Advanced Micro Devices, Inc. [AMD/ATI]" '
    '"Picasso/Raven 2 [Radeon Vega Series / Radeon Vega Mobile Series]" -rc1 '
    '"ASUSTeK Computer Inc." "Picasso"\n'
    '01:00.0 "3D controller" "NVIDIA Corporation" "GA107M" -ra1 "Dell" "Device 0a61"\n'
)


def cache_entry(level: int, cache_type: str, size: str, shared: str) -> dict[str, str]:
    return {"level": f"{level}\n", "type": f"{cache_type}\n", "size": f"{size}\n", "shared_cpu_list": f"{shared}\n"}


@pytest.fixture(scope="module")
def reference() -> mp.JSONObject:
    return json.loads(REFERENCE_PATH.read_text())


# --- capture and schema ---


def test_capture_validates_on_this_machine():

    mp.validate(mp.capture())


def test_reference_profile_validates(reference: mp.JSONObject):

    mp.validate(reference)


def test_schema_is_a_valid_draft_2020_12_schema():

    jsonschema.Draft202012Validator.check_schema(mp.load_schema())


def test_schema_rejects_a_missing_required_key(reference: mp.JSONObject):

    profile = copy.deepcopy(reference)
    del profile["identity"]["cpu"]["isa"]
    with pytest.raises(jsonschema.ValidationError):
        mp.validate(profile)


def test_schema_rejects_an_unknown_key(reference: mp.JSONObject):

    profile = copy.deepcopy(reference)
    profile["identity"]["os"]["colour"] = "blue"
    with pytest.raises(jsonschema.ValidationError):
        mp.validate(profile)


def test_schema_rejects_a_wrong_schema_version(reference: mp.JSONObject):

    profile = copy.deepcopy(reference)
    profile["schema_version"] = 2
    with pytest.raises(jsonschema.ValidationError):
        mp.validate(profile)


# --- collectors ---


def test_parse_cpuinfo():

    cpu = mp.parse_cpuinfo(CPUINFO)

    assert cpu["vendor"] == "AuthenticAMD"
    assert cpu["model_name"] == "AMD Ryzen 7 3700U with Radeon Vega Mobile Gfx"
    assert cpu["sockets"] == 1
    assert cpu["physical_cores"] == 2
    assert cpu["logical_cpus"] == 4
    assert cpu["isa"] == {"sse4_2": True, "avx": True, "avx2": True, "fma": True, "avx512f": False}


def test_parse_cpuinfo_of_nothing_gives_nulls():

    cpu = mp.parse_cpuinfo("")

    assert cpu["vendor"] is None
    assert cpu["physical_cores"] is None
    assert cpu["logical_cpus"] is None
    assert not any(cpu["isa"].values())


def test_parse_cpu_list():

    assert mp.parse_cpu_list("0-1\n") == 2
    assert mp.parse_cpu_list("0-7") == 8
    assert mp.parse_cpu_list("0,4") == 2
    assert mp.parse_cpu_list("0-3,8,10-11") == 7


def test_parse_size_kib():

    assert mp.parse_size_kib("512K\n") == 512
    assert mp.parse_size_kib("8M") == 8192


def test_summarize_caches_per_core_l2_against_shared_l3():

    # four logical CPUs, two cores: each CPU lists its core's L2 and the one shared L3
    entries: list[dict[str, str]] = []
    for cpu in range(4):
        core_cpus = "0-1" if cpu < 2 else "2-3"
        entries.append(cache_entry(2, "Unified", "512K", core_cpus))
        entries.append(cache_entry(3, "Unified", "4096K", "0-3"))

    assert mp.summarize_caches(entries) == [
        {"level": 2, "type": "Unified", "size_kib": 512, "shared_by_logical_cpus": 2, "instances": 2},
        {"level": 3, "type": "Unified", "size_kib": 4096, "shared_by_logical_cpus": 4, "instances": 1},
    ]


def test_parse_frequency():

    files = {
        "scaling_driver": "acpi-cpufreq\n",
        "scaling_governor": "schedutil\n",
        "cpuinfo_min_freq": "1400000\n",
        "cpuinfo_max_freq": "2300000\n",
        "cpb": None,
        "boost": "1\n",
    }

    assert mp.parse_frequency(files) == {
        "driver": "acpi-cpufreq",
        "governor": "schedutil",
        "min_mhz": 1400,
        "max_mhz": 2300,
        "boost_enabled": True,
    }


def test_parse_frequency_without_cpufreq_is_null():

    files = dict.fromkeys(
        ["scaling_driver", "scaling_governor", "cpuinfo_min_freq", "cpuinfo_max_freq", "cpb", "boost"]
    )

    assert mp.parse_frequency(files) is None


def test_parse_meminfo():

    meminfo = mp.parse_meminfo(MEMINFO)

    assert meminfo["MemTotal"] == 5871
    assert meminfo["MemAvailable"] == 4316
    assert meminfo["SwapTotal"] == 2047


def test_parse_lspci_keeps_display_controllers_and_skips_revision_flags():

    assert mp.parse_lspci(LSPCI) == [
        {
            "class": "VGA compatible controller",
            "vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
            "device": "Picasso/Raven 2 [Radeon Vega Series / Radeon Vega Mobile Series]",
        },
        {"class": "3D controller", "vendor": "NVIDIA Corporation", "device": "GA107M"},
    ]


def test_parse_os_release():

    assert mp.parse_os_release('NAME="Ubuntu"\nPRETTY_NAME="Ubuntu 22.04.5 LTS"\n') == "Ubuntu 22.04.5 LTS"
    assert mp.parse_os_release("NAME=Ubuntu\n") is None


def test_parse_power():

    supplies = {
        "AC0": {"type": "Mains\n", "online": "1\n", "capacity": None},
        "BAT0": {"type": "Battery\n", "online": None, "capacity": "87\n"},
    }

    assert mp.parse_power(supplies) == {"on_ac": True, "battery_percent": 87}


def test_parse_power_with_no_power_supply_is_null():

    assert mp.parse_power({}) == {"on_ac": None, "battery_percent": None}


def test_parse_blas():

    config = {
        "Build Dependencies": {
            "blas": {
                "name": "scipy-openblas",
                "version": "0.3.29",
                "found": True,
                "openblas configuration": "OpenBLAS 0.3.29  DYNAMIC_ARCH Haswell MAX_THREADS=64",
            }
        }
    }

    assert mp.parse_blas(config) == {
        "name": "scipy-openblas",
        "version": "0.3.29",
        "configuration": "OpenBLAS 0.3.29  DYNAMIC_ARCH Haswell MAX_THREADS=64",
    }
    assert mp.parse_blas(None) is None


def test_parse_release_profile():

    assert mp.parse_release_profile('[package]\nname = "x"\n') == "default"
    assert mp.parse_release_profile("[profile.release]\nlto = true\ncodegen-units = 1\n") == {
        "lto": True,
        "codegen-units": 1,
    }


# --- compare ---


def test_identical_identities_give_no_differences(reference: mp.JSONObject):

    assert mp.compare(reference, copy.deepcopy(reference)) == []


@pytest.mark.parametrize(
    "path, value",
    [
        (["cpu", "frequency", "governor"], "performance"),
        (["cpu", "isa", "avx512f"], True),
        (["software", "numpy", "blas", "version"], "0.3.30"),
        (["software", "thread_env", "OPENBLAS_NUM_THREADS"], "1"),
    ],
)
def test_a_changed_identity_field_reports_exactly_its_path(reference: mp.JSONObject, path: list[str], value: object):

    current = copy.deepcopy(reference)
    parent = current["identity"]
    for key in path[:-1]:
        parent = parent[key]
    old = parent[path[-1]]
    parent[path[-1]] = value

    assert mp.compare(reference, current) == [mp.Difference("identity." + ".".join(path), old, value)]


def test_a_changed_cache_inside_the_list_reports_its_index(reference: mp.JSONObject):

    current = copy.deepcopy(reference)
    current["identity"]["cpu"]["caches"][2]["size_kib"] = 1024

    assert [d.path for d in mp.compare(reference, current)] == ["identity.cpu.caches[2].size_kib"]


def test_an_extra_gpu_is_reported(reference: mp.JSONObject):

    current = copy.deepcopy(reference)
    current["identity"]["gpus"].append({"class": "3D controller", "vendor": "NVIDIA Corporation", "device": "GA107M"})

    differences = mp.compare(reference, current)

    assert [d.path for d in differences] == ["identity.gpus"]
    assert differences[0].current == current["identity"]["gpus"]


def test_state_changes_are_never_reported(reference: mp.JSONObject):

    current = copy.deepcopy(reference)
    current["state"] = {
        "captured_at": "2030-01-01T00:00:00Z",
        "hostname": "elsewhere",
        "load_average": [7.0, 7.0, 7.0],
        "memory_available_mib": 1,
        "cpu_mhz_now": None,
        "power": {"on_ac": False, "battery_percent": 3},
        "commits": {"repo": None, "repo_dirty": None, "rust": None},
    }

    assert mp.compare(reference, current) == []


# --- CLI ---


def run_script(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run([sys.executable, str(SCRIPT_PATH), *args], capture_output=True, text=True, check=False)


def test_cli_profile_then_compare_against_itself_and_an_edited_copy(tmp_path: Path):

    out = tmp_path / "profile.json"
    assert run_script("profile", "--out", str(out)).returncode == 0

    same = run_script("compare", str(out), str(out))
    assert same.returncode == 0, same.stdout + same.stderr
    assert "identity matches" in same.stdout

    edited = json.loads(out.read_text())
    edited["identity"]["os"]["kernel_release"] = "0.0.0-edited"
    edited_path = tmp_path / "edited.json"
    edited_path.write_text(json.dumps(edited))

    differs = run_script("compare", str(out), str(edited_path))
    assert differs.returncode == 1
    assert "identity.os.kernel_release: " in differs.stdout
    assert '-> "0.0.0-edited"' in differs.stdout
