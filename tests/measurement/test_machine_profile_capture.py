import pytest

from indrajala_ml.measurement import machine_profile_capture as mpc

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


# --- collectors ---


def test_parse_cpuinfo():

    cpu = mpc.parse_cpuinfo(CPUINFO)

    assert cpu["vendor"] == "AuthenticAMD"
    assert cpu["model_name"] == "AMD Ryzen 7 3700U with Radeon Vega Mobile Gfx"
    assert cpu["sockets"] == 1
    assert cpu["physical_cores"] == 2
    assert cpu["logical_cpus"] == 4
    assert cpu["isa"] == {"sse4_2": True, "avx": True, "avx2": True, "fma": True, "avx512f": False}


def test_parse_cpuinfo_of_nothing_gives_nulls():

    cpu = mpc.parse_cpuinfo("")

    assert cpu["vendor"] is None
    assert cpu["physical_cores"] is None
    assert cpu["logical_cpus"] is None
    assert not any(cpu["isa"].values())


def test_parse_cpu_list():

    assert mpc.parse_cpu_list("0-1\n") == 2
    assert mpc.parse_cpu_list("0-7") == 8
    assert mpc.parse_cpu_list("0,4") == 2
    assert mpc.parse_cpu_list("0-3,8,10-11") == 7


def test_parse_size_kib():

    assert mpc.parse_size_kib("512K\n") == 512
    assert mpc.parse_size_kib("8M") == 8192


def test_summarize_caches_per_core_l2_against_shared_l3():

    # four logical CPUs, two cores: each CPU lists its core's L2 and the one shared L3
    entries: list[dict[str, str]] = []
    for cpu in range(4):
        core_cpus = "0-1" if cpu < 2 else "2-3"
        entries.append(cache_entry(2, "Unified", "512K", core_cpus))
        entries.append(cache_entry(3, "Unified", "4096K", "0-3"))

    assert mpc.summarize_caches(entries) == [
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
        "energy_performance_preference": None,
        "boost": "1\n",
        "intel_pstate_status": None,
        "no_turbo": None,
    }

    assert mpc.parse_frequency(files) == {
        "driver": "acpi-cpufreq",
        "governor": "schedutil",
        "min_mhz": 1400,
        "max_mhz": 2300,
        "boost_enabled": True,
        "energy_performance_preference": None,
        "intel_pstate_status": None,
    }


@pytest.mark.parametrize("no_turbo, boost_enabled", [("0\n", True), ("1\n", False)])
def test_parse_frequency_under_intel_pstate_reads_turbo_from_no_turbo(no_turbo: str, boost_enabled: bool):

    # the i7-9700K's sysfs: intel_pstate has no boost or cpb file
    files = {
        "scaling_driver": "intel_pstate\n",
        "scaling_governor": "powersave\n",
        "cpuinfo_min_freq": "800000\n",
        "cpuinfo_max_freq": "4900000\n",
        "cpb": None,
        "energy_performance_preference": "balance_performance\n",
        "boost": None,
        "intel_pstate_status": "active\n",
        "no_turbo": no_turbo,
    }

    assert mpc.parse_frequency(files) == {
        "driver": "intel_pstate",
        "governor": "powersave",
        "min_mhz": 800,
        "max_mhz": 4900,
        "boost_enabled": boost_enabled,
        "energy_performance_preference": "balance_performance",
        "intel_pstate_status": "active",
    }


def test_parse_frequency_without_cpufreq_is_null():

    files = dict.fromkeys(
        [
            "scaling_driver",
            "scaling_governor",
            "cpuinfo_min_freq",
            "cpuinfo_max_freq",
            "cpb",
            "energy_performance_preference",
            "boost",
            "intel_pstate_status",
            "no_turbo",
        ]
    )

    assert mpc.parse_frequency(files) is None


def test_parse_meminfo():

    meminfo = mpc.parse_meminfo(MEMINFO)

    assert meminfo["MemTotal"] == 5871
    assert meminfo["MemAvailable"] == 4316
    assert meminfo["SwapTotal"] == 2047


def test_parse_lspci_keeps_display_controllers_and_skips_revision_flags():

    assert mpc.parse_lspci(LSPCI) == [
        {
            "class": "VGA compatible controller",
            "vendor": "Advanced Micro Devices, Inc. [AMD/ATI]",
            "device": "Picasso/Raven 2 [Radeon Vega Series / Radeon Vega Mobile Series]",
        },
        {"class": "3D controller", "vendor": "NVIDIA Corporation", "device": "GA107M"},
    ]


def test_parse_os_release():

    assert mpc.parse_os_release('NAME="Ubuntu"\nPRETTY_NAME="Ubuntu 22.04.5 LTS"\n') == "Ubuntu 22.04.5 LTS"
    assert mpc.parse_os_release("NAME=Ubuntu\n") is None


def test_parse_power():

    supplies = {
        "AC0": {"type": "Mains\n", "online": "1\n", "capacity": None},
        "BAT0": {"type": "Battery\n", "online": None, "capacity": "87\n"},
    }

    assert mpc.parse_power(supplies) == {"on_ac": True, "battery_percent": 87}


def test_parse_power_with_no_power_supply_is_null():

    assert mpc.parse_power({}) == {"on_ac": None, "battery_percent": None}


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

    assert mpc.parse_blas(config) == {
        "name": "scipy-openblas",
        "version": "0.3.29",
        "configuration": "OpenBLAS 0.3.29  DYNAMIC_ARCH Haswell MAX_THREADS=64",
    }
    assert mpc.parse_blas(None) is None


def test_parse_release_profile():

    assert mpc.parse_release_profile('[package]\nname = "x"\n') == "default"
    assert mpc.parse_release_profile("[profile.release]\nlto = true\ncodegen-units = 1\n") == {
        "lto": True,
        "codegen-units": 1,
    }


def test_parse_power_limits():

    # the i7-9700K's RAPL package-0 zone, with PL1 lowered by the setup script
    constraints = {
        "0": {"name": "long_term\n", "power_limit_uw": "65000000\n"},
        "1": {"name": "short_term\n", "power_limit_uw": "120000000\n"},
    }

    assert mpc.parse_power_limits(constraints) == {"long_term_w": 65, "short_term_w": 120}


def test_parse_power_limits_without_rapl_is_null():

    assert mpc.parse_power_limits({}) is None
    assert mpc.parse_power_limits({"0": {"name": None, "power_limit_uw": None}}) is None
