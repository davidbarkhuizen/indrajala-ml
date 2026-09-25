"""
Hardware counters for one region of a Python driver, each run in its own process: `perf stat`
starts with its counters disabled, and the driver switches them on and off around the code it
measures (`with counted():`), through perf's control FIFO. Imports, setup and warm-up stay out.

The driver says what one unit of work is (`report(units=...)`, say windows or examples), and may
report other values (`report(us_per_call=...)`); the table shows every event per unit, IPC, and
those values, one row per process, so a process-level mode (a run that is fast or slow as a whole)
shows up as rows that differ.

    python scripts/perf_region.py --processes 8 -- path/to/driver.py --batch 32
    python scripts/perf_region.py --events cycles,instructions,ls_stlf -- driver.py

A driver:

    sys.path.insert(0, "<repo>/scripts")
    from perf_region import counted, report
    ...                      # setup, warm-up
    with counted():
        for _ in range(reps):
            op()
    report(units=reps * windows, us_per_call=...)

Run standalone (not under this tool), `counted()` does nothing. Needs
`kernel.perf_event_paranoid` <= 2 for user-space counts (1 or lower to count the kernel too). Zen
has 6 general counters: more events are multiplexed, and the `run%` column says by how much.
"""

import argparse
import contextlib
import json
import os
import subprocess
import sys
import tempfile

DEFAULT_EVENTS = "cycles,instructions,ls_dc_accesses,l2_cache_accesses_from_dc_misses,ls_stlf,ls_misal_accesses"
CTL_ENV, ACK_ENV = "PERF_REGION_CTL", "PERF_REGION_ACK"
REPORT_PREFIX = "perf_region:"


@contextlib.contextmanager
def counted():
    """Counts the enclosed code when run under perf_region.py; a no-op otherwise."""
    ctl_path, ack_path = os.environ.get(CTL_ENV), os.environ.get(ACK_ENV)
    if not ctl_path:
        yield
        return
    assert ack_path, f"{ACK_ENV} must be set with {CTL_ENV}"
    with open(ctl_path, "w") as ctl, open(ack_path) as ack:

        def command(word: str) -> None:
            ctl.write(word + "\n")
            ctl.flush()
            ack.readline()

        command("enable")
        try:
            yield
        finally:
            command("disable")


def report(**values: float) -> None:
    """Values for this process's row; `units` (the work counted) divides every event."""
    print(REPORT_PREFIX, json.dumps(values), flush=True)


def run_process(events: str, driver: list[str], openblas_threads: str) -> dict[str, dict[str, float]]:
    with tempfile.TemporaryDirectory() as tmp:
        ctl, ack, out = (os.path.join(tmp, name) for name in ("ctl", "ack", "out"))
        os.mkfifo(ctl)
        os.mkfifo(ack)
        command = ["perf", "stat", "-x,", "-o", out, "-D", "-1", "--control", f"fifo:{ctl},{ack}", "-e", events]
        command += ["--", sys.executable, *driver]
        env = dict(os.environ, **{CTL_ENV: ctl, ACK_ENV: ack, "OPENBLAS_NUM_THREADS": openblas_threads})
        result = subprocess.run(command, env=env, capture_output=True, text=True, check=False)
        if result.returncode != 0:
            sys.exit(f"driver failed ({result.returncode}):\n{result.stdout}\n{result.stderr}")
        values: dict[str, float] = {}
        for line in result.stdout.splitlines():
            if line.startswith(REPORT_PREFIX):
                values.update(json.loads(line[len(REPORT_PREFIX) :]))
        counts: dict[str, float] = {}
        running: dict[str, float] = {}
        with open(out) as f:
            for line in f:
                fields = line.strip().split(",")
                if len(fields) < 5 or line.startswith("#"):
                    continue
                value, _, name, _, percent = fields[:5]
                counts[name] = float(value) if value.replace(".", "").isdigit() else float("nan")
                running[name] = float(percent) if percent else float("nan")
    return {"values": values, "counts": counts, "running": running}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--processes", type=int, default=6, help="separate processes to run the driver in")
    parser.add_argument("--events", default=DEFAULT_EVENTS, help="comma-separated perf events")
    parser.add_argument(
        "--openblas-threads", default="1", help="OPENBLAS_NUM_THREADS for the driver (spinning workers pollute counts)"
    )
    parser.add_argument("--json", help="also write every process's raw counts here")
    parser.add_argument("driver", nargs=argparse.REMAINDER, help="-- driver.py [driver args]")
    args = parser.parse_args()
    driver = args.driver[1:] if args.driver[:1] == ["--"] else args.driver
    if not driver:
        parser.error("give a driver script after --")

    rows = [run_process(args.events, driver, args.openblas_threads) for _ in range(args.processes)]
    events = list(rows[0]["counts"])
    value_keys = [k for k in rows[0]["values"] if k != "units"]
    per = "per unit" if "units" in rows[0]["values"] else "total"
    print(f"events {per}; run% is the lowest share of the region any event was counted")
    header = [f"{k:>12}" for k in value_keys] + [f"{e[:22]:>22}" for e in events] + [f"{'IPC':>5}", f"{'run%':>5}"]
    print(" ".join(header))
    for row in rows:
        units = row["values"].get("units", 1)
        cells = [f"{row['values'][k]:12.4g}" for k in value_keys]
        cells += [f"{row['counts'][e] / units:22.4g}" for e in events]
        c = row["counts"]
        ipc = c["instructions"] / c["cycles"] if c.get("cycles") and "instructions" in c else float("nan")
        cells += [f"{ipc:5.2f}", f"{min(row['running'].values()):5.0f}"]
        print(" ".join(cells))
    if args.json:
        with open(args.json, "w") as f:
            json.dump(rows, f, indent=1)


if __name__ == "__main__":
    main()
