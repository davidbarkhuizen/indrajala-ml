import asyncio
import json
import os
import signal
import socket
import sys
import time
from pathlib import Path

import pytest

from indrajala_ml import batch as b

# A stub run (fast, deterministic): seed s even certifies at both verdicts at step s, an odd seed never;
# the budget is 6 steps ("spent"); --cutoff-steps T cuts an attempt not certified by step T ("cut").
STUB = """
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
seed = int(args[args.index("--runs") + 1].split(":")[1])
out = Path(args[args.index("--out") + 1]) / f"seed{seed}"
cutoff = int(args[args.index("--cutoff-steps") + 1]) if "--cutoff-steps" in args else None
sleep = float(os.environ.get("STUB_SLEEP", "0.01"))
out.mkdir(parents=True, exist_ok=True)
with (out / "log.jsonl").open("w") as log:
    for step in range(1, 7):
        time.sleep(sleep)
        won = seed % 2 == 0 and step == seed
        end = "passed" if won else "spent" if step == 6 else "cut" if cutoff is not None and step >= cutoff else "epoch"
        row = {"step": step, "end": end, "state_share": 1.0, "seconds": float(step), "certified": won,
               "certificates": {"strict": {"certified": won}}}
        log.write(json.dumps(row) + "\\n"); log.flush()
        if end != "epoch":
            break
print(f"seed {seed}: step {step}, {end}")
"""


def free_port() -> int:
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def worker(tmp_path: Path, name: str, slots: int, env: dict[str, str] | None = None) -> b.Worker:
    stub = tmp_path / "stub.py"
    stub.write_text(STUB)
    return b.Worker(tmp_path / name, slots, sys.executable, tmp_path, script=str(stub), host=name, env=env, poll=0.05)


def run_batch(tmp_path: Path, batch: b.Batch, *workers: b.Worker) -> b.State:
    directory = tmp_path / "batch"
    b.register(directory, batch)
    port = free_port()

    async def go() -> None:
        controller = b.Controller(directory)
        serving = asyncio.create_task(controller.serve(port))
        await asyncio.sleep(0.1)
        await asyncio.wait_for(asyncio.gather(*(w.serve(port, retry=0.1) for w in workers)), timeout=60)
        await asyncio.wait_for(serving, timeout=10)

    asyncio.run(go())
    return b.State.read(directory)


def test_attempts_read_a_seed_spec_with_each_backdrops_cutoff():
    made = b.attempts("s3:40-42,mtf4:7", {"s3": 137})
    assert [(a.id, a.seed, a.cutoff) for a in made] == [
        ("s3-40", 40, 137),
        ("s3-41", 41, 137),
        ("s3-42", 42, 137),
        ("mtf4-7", 7, None),
    ]
    with pytest.raises(AssertionError, match="registered cell"):
        b.Batch("x", "measure", "nope", made)
    with pytest.raises(AssertionError, match="find needs k"):
        b.Batch("x", "find", "continuous", made)


def test_a_measure_batch_runs_every_attempt_across_two_workers_and_logs_each_end(tmp_path: Path):
    batch = b.Batch("m", "measure", "continuous", b.attempts("s3:1-6"))
    state = run_batch(tmp_path, batch, worker(tmp_path, "w1", 2), worker(tmp_path, "w2", 1))
    assert state.done and not state.assigned and set(state.ended) == {f"s3-{s}" for s in range(1, 7)}
    assert {i: s["success"] for i, s in state.ended.items()} == {
        "s3-1": None,
        "s3-2": 2,
        "s3-3": None,
        "s3-4": 4,
        "s3-5": None,
        "s3-6": 6,
    }
    assert all(s["end"] == "spent" for i, s in state.ended.items() if int(i.split("-")[1]) % 2)
    hosts = [json.loads(line)["host"] for line in (tmp_path / "batch" / b.LEDGER).read_text().splitlines()[1:-1]]
    assert {"w1", "w2"} <= set(hosts)  # both workers ran attempts
    assert b.report(tmp_path / "batch")[1].startswith("  s3: 6 of 6 ended, 3 certified (0.50,")
    assert b.runs_of(list(state.ended.values())).count((None, 6.0, True)) == 3


def test_a_find_batch_stops_starting_attempts_once_k_have_succeeded(tmp_path: Path):
    batch = b.Batch("f", "find", "continuous", b.attempts("s3:1-5"), k=1)
    state = run_batch(tmp_path, batch, worker(tmp_path, "w", 1))
    assert set(state.ended) == {"s3-1", "s3-2"} and state.ended["s3-2"]["success"] == 2  # 3-5 never start


def test_a_cut_attempt_ends_cut_and_unsuccessful(tmp_path: Path):
    batch = b.Batch("c", "measure", "continuous", b.attempts("s3:2,s3:4", {"s3": 3}))
    state = run_batch(tmp_path, batch, worker(tmp_path, "w", 2))
    assert state.ended["s3-2"]["success"] == 2 and state.ended["s3-4"] == {
        "success": None,
        "step": 3,
        "end": "cut",
        "seconds": 3.0,
    }


def test_the_controller_rebuilds_from_its_ledger_and_requeues_a_hosts_lost_attempt(tmp_path: Path):
    directory = tmp_path / "batch"
    b.register(directory, b.Batch("r", "measure", "continuous", b.attempts("s3:1-3")))
    controller = b.Controller(directory)
    controller.log({"type": "assigned", "id": "s3-1", "host": "h"})
    controller.log({"type": "ended", "id": "s3-1", "host": "h", "summary": b.summary(tmp_path / "none")})
    controller.log({"type": "assigned", "id": "s3-2", "host": "h"})
    rebuilt = b.Controller(directory).state  # a restarted controller
    assert set(rebuilt.ended) == {"s3-1"} and rebuilt.assigned == {"s3-2": "h"}
    assert rebuilt.next_attempt() == b.Attempt("s3-3", "s3", 3)

    async def hello() -> None:  # h comes back without s3-2: lost, so it is queued again
        port = free_port()
        again = b.Controller(directory)
        serving = asyncio.create_task(again.serve(port))
        await asyncio.sleep(0.1)
        _, writer = await asyncio.open_connection("127.0.0.1", port)
        await b.send(writer, {"type": "hello", "host": "h", "slots": 1, "running": []})
        await asyncio.sleep(0.2)
        writer.close()
        serving.cancel()
        assert again.state.next_attempt() == b.Attempt("s3-2", "s3", 2)

    asyncio.run(hello())
    assert '"type": "lost"' in (directory / b.LEDGER).read_text()


def test_a_restarted_worker_re_adopts_its_running_attempts_from_their_pid_files(tmp_path: Path):
    first = worker(tmp_path, "w", 1, env={"STUB_SLEEP": "2"})
    running = first.launch(b.Attempt("s3-1", "s3", 1), "continuous")
    assert running.pid > 0
    again = worker(tmp_path, "w", 1)  # the daemon restarted while its attempt runs
    assert list(again.running) == ["s3-1"] and again.running["s3-1"].pid == running.pid
    assert first.command(running.attempt, "continuous")[3:5] == ["--runs", "word-problem-s3:1"]
    os.killpg(running.pid, signal.SIGTERM)  # its own session
    deadline = time.monotonic() + 5
    while b.alive(running.pid) and time.monotonic() < deadline:
        time.sleep(0.05)
    assert not b.alive(running.pid) and not worker(tmp_path, "w", 1).running  # ended: nothing to adopt
