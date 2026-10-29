"""
A thermal probe: `--cores` single-threaded workers running the crate's matrix product (its NEON
kernel on aarch64, the hottest steady load the machine runs), for `--minutes`, the CPU's
temperature sampled every `--every` seconds; every worker is stopped at once if it goes over
`--stop-above` (inclusive: exactly at it runs on). It answers how many busy cores a machine without
an active cooler holds within its limits (the aarch64 workplan's litshe: the owner's 70 C hard
limit, 65 C preferred; one core holds 51-60 C, four pass 65 C in minutes).

It holds the machine lock (ab.py's and the studies'), so it never runs beside an A/B or a study;
with `--wait` it waits for the lock, else it refuses. It starts once the CPU is below
`--start-below`. The result, written to `--out` as JSON: the samples (seconds, C), the peak,
whether it was stopped, and the Raspberry Pi firmware's throttle flags before and after (None on
other machines).

    python scripts/measurement/thermal_probe.py --cores 2 --minutes 20 --stop-above 65 --wait --out FILE
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

import ab  # scripts/ isn't a package: this script's directory is on sys.path

from indrajala_ml.measurement import machine_lock
from indrajala_ml.measurement.machine_profile_capture import parse_throttled

# one worker: a 256 x 256 product on one thread, forever, until stopped
WORKLOAD = (
    "import random, indrajala_math_rust as pa\n"
    "pa.set_matmul_threading(1, 0)\n"
    "g = random.Random(0)\n"
    "a = pa.Array([[g.uniform(-1, 1) for _ in range(256)] for _ in range(256)])\n"
    "while True:\n"
    "    a @ a\n"
)
COOL_WAIT_S = 1800
LOCK_POLL_S = 60


def worker_command() -> list[str]:
    """One worker's command line."""
    return [sys.executable, "-c", WORKLOAD]


def throttled() -> int | None:
    """The Pi firmware's throttle flags now, or None where there is no `vcgencmd`."""
    try:
        done = subprocess.run(["vcgencmd", "get_throttled"], capture_output=True, text=True, check=False)
    except OSError:
        return None
    return parse_throttled(done.stdout.strip() or None)


def wait_until_below(start_below: float, poll_s: float) -> float | None:
    """The temperature once below start_below (None where it can't be read); refuses after COOL_WAIT_S."""
    deadline = time.monotonic() + COOL_WAIT_S
    while (temp := ab.cpu_temp_c()) is not None and temp >= start_below:
        if time.monotonic() > deadline:
            raise SystemExit(f"not started: the CPU is at {temp:.1f} C, not below {start_below:g} C")
        time.sleep(poll_s)
    return temp


def probe(cores: int, seconds: float, stop_above: float, every: float) -> dict[str, Any]:
    """Run cores workers for seconds, sampling the temperature; stop them all over stop_above."""
    before = throttled()
    workers = [subprocess.Popen(worker_command()) for _ in range(cores)]
    start = time.monotonic()
    samples: list[tuple[float, float]] = []
    stopped = False
    try:
        while (elapsed := time.monotonic() - start) < seconds:
            if (temp := ab.cpu_temp_c()) is not None:
                samples.append((round(elapsed, 1), temp))
                if temp > stop_above:
                    stopped = True
                    break
            time.sleep(every)
    finally:
        for worker in workers:
            worker.terminate()
        for worker in workers:
            worker.wait()
    peak = max((temp for _, temp in samples), default=None)
    return {
        "cores": cores,
        "seconds": seconds,
        "ran_s": round(time.monotonic() - start, 1),
        "stop_above_c": stop_above,
        "stopped": stopped,
        "peak_c": peak,
        "throttled_before": before,
        "throttled_after": throttled(),
        "samples": samples,
    }


def main(argv: list[str]) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--cores", type=int, required=True)
    parser.add_argument("--minutes", type=float, default=20.0)
    parser.add_argument("--stop-above", type=float, default=65.0, help="C: every worker stopped over it")
    parser.add_argument("--start-below", type=float, default=50.0, help="C: wait for the CPU to cool below it")
    parser.add_argument("--every", type=float, default=5.0, help="seconds between samples")
    parser.add_argument("--wait", action="store_true", help="wait for the machine lock instead of refusing")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    what = " ".join([Path(sys.argv[0]).name, *argv])
    while True:
        try:
            with machine_lock.hold(what, "study"):
                started_at = wait_until_below(args.start_below, args.every)
                result = {
                    "started_at_c": started_at,
                    **probe(args.cores, args.minutes * 60, args.stop_above, args.every),
                }
                break
        except machine_lock.MachineBusy as busy:
            if not args.wait:
                raise SystemExit(f"not started: {busy}") from None
            time.sleep(LOCK_POLL_S)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1))
    verdict = "STOPPED over" if result["stopped"] else "held within"
    print(
        f"{args.cores} cores, {result['ran_s']:g} s: peak {result['peak_c']} C, {verdict} {args.stop_above:g} C; "
        f"throttle flags {result['throttled_before']} -> {result['throttled_after']}"
    )
    return 1 if result["stopped"] else 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
