#!/usr/bin/env bash
#
# Install the benchmark machine's setup script so the dev host can re-apply it over ssh after a
# reboot (the benchmark archive workplan's D9). Run once on the benchmark machine, and again
# whenever scripts/benchmark_machine_setup.sh changes:
#
#     sudo scripts/install_benchmark_setup.sh
#
# It copies the setup script to a root-owned /usr/local/sbin/indrajala-benchmark-setup and allows
# the invoking user to run exactly that path as root without a password:
#
#     sudo /usr/local/sbin/indrajala-benchmark-setup
#
# The repository's copy is never what sudo runs: anyone who can edit the checkout could otherwise
# run anything as root. `ab.py remote` refuses to re-apply an installed copy that differs from the
# checkout's, and names this command.

set -euo pipefail

target="/usr/local/sbin/indrajala-benchmark-setup"
sudoers="/etc/sudoers.d/indrajala-benchmark-setup"
source="$(dirname "$(readlink -f "$0")")/benchmark_machine_setup.sh"

if [[ $EUID -ne 0 || -z "${SUDO_USER:-}" || "$SUDO_USER" == "root" ]]; then
    echo "run with sudo from the account that runs benchmarks: sudo $0" >&2
    exit 1
fi

install -o root -g root -m 0755 "$source" "$target"

# written to a temporary file and checked by visudo first: a broken sudoers file can lock sudo out
entry="$(mktemp)"
trap 'rm -f "$entry"' EXIT
echo "$SUDO_USER ALL=(root) NOPASSWD: $target" > "$entry"
visudo --check --quiet --file "$entry"
install -o root -g root -m 0440 "$entry" "$sudoers"

echo "installed $target ($(sha256sum "$target" | cut -c1-12)) and $sudoers for $SUDO_USER"
echo "check: sudo -n $target"
