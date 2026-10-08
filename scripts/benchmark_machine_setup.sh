#!/usr/bin/env bash
#
# Prepare the benchmark machine (the i7-9700K) for timing, once per boot:
#
#     sudo scripts/benchmark_machine_setup.sh
#
# Sets the frequency policy the machine profile records, perf_event_paranoid 2 (for
# scripts/perf_region.py) and holds snap refreshes, then prints the state it set. None of it
# survives a reboot except the snap hold, which expires, and ab.py's machine check refuses a run
# when the frequency policy or perf_event_paranoid differ from docs/machine_profiles/i7-9700k.json.
# It changes nothing that isn't on this list (docs/benchmark-machine-workplan.md, D2 and D3).

set -euo pipefail

# The frequency policy timed under (workplan D1). The defaults until stage 2 settles D1.
governor="powersave"
energy_performance_preference="balance_performance"
no_turbo="0"
perf_event_paranoid="2"
# Longer than a day's measurement session; `snap refresh --unhold` ends it early.
snap_hold="24h"

cpu_root="/sys/devices/system/cpu"
intel_pstate="$cpu_root/intel_pstate"

if [[ $EUID -ne 0 ]]; then
    echo "run with sudo: sudo $0" >&2
    exit 1
fi

if [[ "$(cat "$intel_pstate/status" 2>/dev/null)" != "active" ]]; then
    echo "expected intel_pstate in active mode (the i7-9700K's driver); not changing anything" >&2
    exit 1
fi

# the governor first: under intel_pstate, switching it to performance resets the EPP
for policy in "$cpu_root"/cpufreq/policy*; do
    echo "$governor" > "$policy/scaling_governor"
    echo "$energy_performance_preference" > "$policy/energy_performance_preference"
done
echo "$no_turbo" > "$intel_pstate/no_turbo"

sysctl --quiet kernel.perf_event_paranoid="$perf_event_paranoid"

snap refresh --hold="$snap_hold" > /dev/null

echo "governor:        $(sort -u "$cpu_root"/cpufreq/policy*/scaling_governor | paste -sd,)"
echo "EPP:             $(sort -u "$cpu_root"/cpufreq/policy*/energy_performance_preference | paste -sd,)"
echo "no_turbo:        $(cat "$intel_pstate/no_turbo")"
echo "perf_event_paranoid: $(cat /proc/sys/kernel/perf_event_paranoid)"
echo "snap refreshes:  held for $snap_hold ($(snap refresh --time | grep '^hold:' || echo 'hold: not shown'))"
