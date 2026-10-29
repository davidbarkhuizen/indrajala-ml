#!/usr/bin/env bash
#
# Prepare a benchmark machine for timing, once per boot:
#
#     sudo scripts/measurement/benchmark_machine_setup.sh
#
# Reads which machine it is on and sets the frequency policy that machine's profile records
# (docs/machine_profiles/), perf_event_paranoid 2 (for scripts/measurement/perf_region.py) and holds snap
# refreshes, then prints the state it set:
#
# - the i7-9700K (jebel, intel_pstate): the governor, EPP, turbo and the package power limit PL1,
#   stopping thermald, which resets the limit;
# - a Raspberry Pi 5 (litshe, cpufreq-dt): the performance governor at a fixed maximum clock, and
#   apt's daily timers stopped (the aarch64 workplan's D8).
#
# None of it survives a reboot except the snap hold, which expires, and ab.py's machine check
# refuses a run when the policy differs from the machine's profile. It changes nothing that isn't
# on this list (the benchmark machine workplan's D2 and D3: git show
# 480164d:docs/benchmark-machine-workplan.md).

set -euo pipefail

perf_event_paranoid="2"
# Longer than a day's measurement session; `snap refresh --unhold` ends it early.
snap_hold="24h"

cpu_root="/sys/devices/system/cpu"
intel_pstate="$cpu_root/intel_pstate"
rapl_package="/sys/class/powercap/intel-rapl:0"

setup_i7_9700k() {
    # The frequency policy timed under (workplan D1: the defaults, chosen by stage 2's A/As).
    local governor="powersave"
    local energy_performance_preference="balance_performance"
    local no_turbo="0"
    # PL1 (RAPL long_term), lowered from the stock 95 W: at 95 W a 10-minute all-core load reached
    # 100 C and throttled, and the cooling can't be improved (workplan stage 2). One busy core
    # draws about 37 W, so single-threaded clocks are unaffected. PL2 (short_term) stays at 120 W.
    local pl1_watts="65"

    # the governor first: under intel_pstate, switching it to performance resets the EPP
    for policy in "$cpu_root"/cpufreq/policy*; do
        echo "$governor" > "$policy/scaling_governor"
        echo "$energy_performance_preference" > "$policy/energy_performance_preference"
    done
    echo "$no_turbo" > "$intel_pstate/no_turbo"

    if [[ "$(cat "$rapl_package/name" 2>/dev/null)" != "package-0" \
        || "$(cat "$rapl_package/constraint_0_name")" != "long_term" ]]; then
        echo "expected RAPL's package-0 zone with long_term as constraint 0; PL1 not set" >&2
        exit 1
    fi
    echo "$((pl1_watts * 1000000))" > "$rapl_package/constraint_0_power_limit_uw"
    # thermald (--adaptive) cools through RAPL first, and puts PL1 back to the package's 95 W
    # (RAPL's max_power_uw) within about 15-40 minutes, restarted after this script or not.
    # Stopped, PL1 held at 65 W for 109 minutes of A/As (workplan stage 4). It starts again at
    # boot. The CPU still throttles itself at 100 C, and 65 W keeps an all-core load near 78 C
    # (stage 2). ab.py re-reads PL1 after every pass.
    systemctl stop thermald

    common_setup
    echo "governor:        $(sort -u "$cpu_root"/cpufreq/policy*/scaling_governor | paste -sd,)"
    echo "EPP:             $(sort -u "$cpu_root"/cpufreq/policy*/energy_performance_preference | paste -sd,)"
    echo "no_turbo:        $(cat "$intel_pstate/no_turbo")"
    echo "PL1, PL2:        $(($(cat "$rapl_package/constraint_0_power_limit_uw") / 1000000)) W, $(($(cat "$rapl_package/constraint_1_power_limit_uw") / 1000000)) W"
    echo "thermald:        $(systemctl is-active thermald || true)"
    common_report
}

setup_raspberry_pi_5() {
    # A fixed clock: the performance governor at max_khz (the aarch64 workplan's D8). The Pi has
    # no RAPL and no turbo; its firmware throttles when hot (no active cooler, D10), which ab.py
    # reads from `vcgencmd get_throttled` before a run and after every pass.
    local governor="performance"
    local max_khz="2400000"

    for policy in "$cpu_root"/cpufreq/policy*; do
        echo "$governor" > "$policy/scaling_governor"
        echo "$max_khz" > "$policy/scaling_max_freq"
    done
    # apt's daily runs (the package lists, unattended-upgrades) until the next boot
    systemctl stop apt-daily.timer apt-daily-upgrade.timer

    common_setup
    echo "governor:        $(sort -u "$cpu_root"/cpufreq/policy*/scaling_governor | paste -sd,)"
    echo "max clock:       $(($(sort -u "$cpu_root"/cpufreq/policy*/scaling_max_freq | head -1) / 1000)) MHz"
    echo "apt timers:      $(systemctl is-active apt-daily.timer apt-daily-upgrade.timer | paste -sd,)"
    echo "firmware:        $(vcgencmd get_throttled 2>/dev/null || echo 'throttled: not readable')"
    common_report
}

common_setup() {
    sysctl --quiet kernel.perf_event_paranoid="$perf_event_paranoid"
    snap refresh --hold="$snap_hold" > /dev/null
}

common_report() {
    echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)"
    echo "snap refreshes:  held for $snap_hold ($(snap refresh --time | grep '^hold:' || echo 'hold: not shown'))"
}

if [[ $EUID -ne 0 ]]; then
    echo "run with sudo: sudo $0" >&2
    exit 1
fi

if [[ "$(cat "$intel_pstate/status" 2>/dev/null)" == "active" ]]; then
    setup_i7_9700k
elif [[ "$(tr -d '\0' < /proc/device-tree/model 2>/dev/null)" == "Raspberry Pi 5"* ]]; then
    setup_raspberry_pi_5
else
    echo "not a known benchmark machine (intel_pstate active, or a Raspberry Pi 5); not changing anything" >&2
    exit 1
fi
