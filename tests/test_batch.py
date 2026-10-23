import asyncio
import datetime
import json
import os
import shutil
import signal
import socket
import sys
import time
from pathlib import Path
from typing import Any

import pytest

from indrajala_ml import batch as b

# A stub run (fast, deterministic): seed s even certifies at both verdicts at step s, an odd seed never;
# an epoch a step, the budget 6 epochs ("spent", its row logging the untrained epoch too, as a run's
# does); --cutoff-epochs T cuts an attempt not certified by T epochs trained ("cut").
STUB = """
import json, os, sys, time
from pathlib import Path
args = sys.argv[1:]
seed = int(args[args.index("--runs") + 1].split(":")[1])
out = Path(args[args.index("--out") + 1]) / f"seed{seed}"
cutoff = int(args[args.index("--cutoff-epochs") + 1]) if "--cutoff-epochs" in args else None
sleep = float(os.environ.get("STUB_SLEEP", "0.01"))
out.mkdir(parents=True, exist_ok=True)
with (out / "log.jsonl").open("w") as log:
    for step in range(1, 7):
        time.sleep(sleep)
        won = seed % 2 == 0 and step == seed
        end = "passed" if won else "spent" if step == 6 else "cut" if cutoff is not None and step >= cutoff else "epoch"
        row = {"step": step, "end": end, "epochs": 2 if end == "spent" else 1, "state_share": 1.0,
               "seconds": float(step), "certified": won, "certificates": {"strict": {"certified": won}}}
        log.write(json.dumps(row) + "\\n"); log.flush()
        if end != "epoch":
            break
print(f"seed {seed}: step {step}, {end}")
"""


SAST = datetime.timezone(datetime.timedelta(hours=2))  # the hosts' local hours


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
        "epochs": 3,
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


def test_a_hosts_window_reads_its_hours_left_and_is_none_outside_them():
    thursday_night = datetime.datetime(2026, 10, 22, 20, 0, tzinfo=SAST)  # a Thursday
    assert b.window_left(b.HOSTS["stone"], thursday_night) == 11 * 3600  # weekday nights to 07:00
    assert (
        b.window_left(b.HOSTS["stone"], datetime.datetime(2026, 10, 22, 12, 0, tzinfo=SAST)) is None
    )  # the owner's day
    assert b.window_left(b.HOSTS["stone"], datetime.datetime(2026, 10, 24, 10, 0, tzinfo=SAST)) == 14 * 3600  # Saturday
    assert (
        b.window_left(b.HOSTS["jebel"], datetime.datetime(2026, 10, 22, 12, 0, tzinfo=SAST)) == 6 * 3600
    )  # by day (solar)
    assert b.window_left(b.HOSTS["jebel"], thursday_night) is None
    assert b.window_left(b.HOSTS["klip"], thursday_night) == float("inf")


def test_a_worker_on_another_commit_is_refused_and_given_nothing(tmp_path: Path):
    directory = tmp_path / "batch"
    b.register(directory, b.Batch("p", "measure", "continuous", b.attempts("s3:1-2"), commit="abc123"))
    port = free_port()

    async def go() -> None:
        controller = b.Controller(directory)
        serving = asyncio.create_task(controller.serve(port))
        await asyncio.sleep(0.1)
        await asyncio.wait_for(worker(tmp_path, "w", 1).serve(port, retry=0.1), timeout=10)  # returns: refused
        serving.cancel()

    asyncio.run(go())
    state = b.State.read(directory)
    assert not state.assigned and not state.ended
    assert '"type": "refused"' in (directory / b.LEDGER).read_text()


def test_a_windows_end_stops_a_hosts_attempts_and_queues_them_again(tmp_path: Path):
    directory = tmp_path / "batch"
    b.register(directory, b.Batch("w", "measure", "continuous", b.attempts("s3:1-2")))
    clock = [datetime.datetime(2026, 10, 22, 12, 0, tzinfo=SAST)]
    host = b.Host("w", 2, str(tmp_path), sys.executable, str(tmp_path / "w"), windows=(b.Window(b.ALL_DAYS, 12, 1),))
    port = free_port()

    async def go() -> b.Controller:
        controller = b.Controller(directory, {"w": host}, now=lambda: clock[0], check=0.1)
        serving = asyncio.create_task(controller.serve(port))
        await asyncio.sleep(0.1)
        running = asyncio.create_task(worker(tmp_path, "w", 2, env={"STUB_SLEEP": "5"}).serve(port, retry=0.1))
        await asyncio.sleep(1.0)  # both attempts assigned and running
        clock[0] = datetime.datetime(2026, 10, 22, 13, 30, tzinfo=SAST)  # the window has closed
        await asyncio.sleep(1.5)
        running.cancel()
        serving.cancel()
        return controller

    controller = asyncio.run(go())
    ledger = (directory / b.LEDGER).read_text()
    assert ledger.count('"type": "assigned"') == 2 and ledger.count('"why": "window"') == 2
    assert not controller.state.assigned and controller.state.next_attempt() == b.Attempt("s3-1", "s3", 1)
    assert not controller.fits("w", b.Attempt("s3-1", "s3", 1))  # nothing given outside its hours


def test_a_remote_attempt_is_pulled_and_checked_against_its_manifest(tmp_path: Path):
    source = tmp_path / "source"
    (source / "seed1").mkdir(parents=True)
    (source / "seed1" / "log.jsonl").write_text('{"step": 1}\n')
    (source / "pid").write_text("{}")
    files = b.manifest(source)
    assert list(files) == ["seed1/log.jsonl"]  # the pid file left out
    copy = tmp_path / "copy"
    shutil.copytree(source, copy)
    assert b.verify(copy, files)
    (copy / "seed1" / "log.jsonl").write_text("changed\n")
    assert not b.verify(copy, files)


def test_the_fleets_commands_follow_each_hosts_rules():
    stone = b.worker_line(b.HOSTS["stone"], 47111)
    assert stone.startswith("source ~/labour/env.sh; ") and "PYTHONPATH=/home/david/labour/site-straight:" in stone
    assert "--slots 11" in stone and "setsid nohup" in stone and stone.endswith("< /dev/null &")
    klip = b.worker_line(b.HOSTS["klip"], 47111)
    assert "--wrapper=/home/david/code/thermal-guard.sh" in klip and "--wrapper=--" in klip
    steps = b.prepare(b.HOSTS["stone"], "abc123", Path("/repo"), Path("/tmp"))
    assert steps[1][3:5] == ["bundle", "create"] and "--no-recurse-submodules" in steps[3][2]
    assert b.prepare(b.HOSTS["klip"], "abc123", Path("/repo"), Path("/tmp"))[0][2].startswith("cd ~/code/indrajala-ml")


def test_a_batch_with_a_remote_host_ends_once_its_attempts_are_pulled(tmp_path: Path):
    directory = tmp_path / "batch"
    b.register(directory, b.Batch("r", "measure", "continuous", b.attempts("s3:1-2")))
    host = b.Host("w", 2, str(tmp_path), sys.executable, str(tmp_path / "w"))  # remote: its attempts pulled
    port = free_port()

    def pull(h: b.Host, attempt: str, destination: Path) -> bool:  # a local copy stands in for rsync over ssh
        shutil.copytree(Path(h.root) / attempt, destination, dirs_exist_ok=True)
        return True

    async def go() -> None:
        controller = b.Controller(directory, {"w": host}, pull=pull, check=0.1)
        serving = asyncio.create_task(controller.serve(port))
        await asyncio.sleep(0.1)
        await asyncio.wait_for(worker(tmp_path, "w", 2).serve(port, retry=0.1), timeout=30)
        await asyncio.wait_for(serving, timeout=10)  # done: every attempt ended and pulled

    asyncio.run(go())
    ledger = (directory / b.LEDGER).read_text()
    assert ledger.count('"ok": true') == 2 and ledger.rstrip().endswith("}") and '"type": "done"' in ledger
    assert (directory / "attempts" / "s3-2" / "seed2" / "log.jsonl").exists()


def test_a_division_backdrop_runs_by_p_seed_and_arm_its_log_at_the_runs_own_path(tmp_path: Path):
    w = worker(tmp_path, "w", 1)
    attempt = b.Attempt("d5-suffix-3", "d5-suffix", 3)
    command = w.command(attempt, "paper")
    run = command[command.index("run") + 1 :]
    assert run[:6] == ["--p", "5", "--seed", "3", "--arm", "suffix"]
    assert run[6:10] == ["--vocabulary", "sum", "suffix", "token"]
    assert "--state-bound" not in command and "--runs" not in command  # the paper's cell stops on its certificate
    assert run[run.index("--factor-width") + 1] == "32" and run[run.index("--budget-epochs") + 1] == "600"
    running = b.Running(attempt, tmp_path / "w" / attempt.id, 0)
    assert w.log_of(running) == running.directory / "log.jsonl"
    machine = b.Running(b.Attempt("s3-1", "s3", 1), tmp_path / "w" / "s3-1", 0)
    assert w.log_of(machine) == machine.directory / "seed1" / "log.jsonl"
    bounded = w.command(machine.attempt, "continuous")
    assert bounded[bounded.index("--state-bound") + 1] == "6"
    assert b.BACKDROPS["d13-family"].args(0)[:2] == ["--p", "13"] and b.BACKDROPS["d13-family"].bound == 169
    assert "--vocabulary" not in b.BACKDROPS["d11-quotient"].args(0)  # T2's: the default vocabulary


def test_t1s_cell_is_the_approximation_claims():
    command = worker(Path("/tmp"), "w", 1).command(b.Attempt("d5-sum-0", "d5-sum", 0), "approximation")
    assert command[command.index("--budget-epochs") + 1] == "400"
    assert not {"--fork", "--codes-ratio", "--window-width", "--state-bound"} & set(command)


def test_a_paper_cells_run_succeeds_at_its_first_certificate_and_a_continuous_one_never(tmp_path: Path):
    paper = tmp_path / "paper.jsonl"
    rows: list[dict[str, Any]] = [
        {"step": 1, "epoch": 0, "accuracy": {}},  # an epoch's screen: no end
        {"step": 1, "end": "passed", "epochs": 30, "certified": False, "minimised": 40, "seconds": 9.0},
        {"step": 2, "end": "passed", "epochs": 25, "certified": True, "minimised": 25, "seconds": 20.0},
    ]
    paper.write_text("".join(json.dumps(r) + "\n" for r in rows))
    read = b.summary(paper)
    assert read["success"] == 55 and read["end"] == "passed" and read["step"] == 2
    continuous = tmp_path / "continuous.jsonl"
    rows = [{"step": 1, "end": "spent", "epochs": 400, "certified": False, "long": {"24": 0.2}, "seconds": 5.0}]
    continuous.write_text("".join(json.dumps(r) + "\n" for r in rows))
    read = b.summary(continuous)
    assert read["success"] is None and read["end"] == "spent" and read["epochs"] == 399


def test_t3s_cells_change_only_the_window_or_the_state():
    w = worker(Path("/tmp"), "w", 1)
    base = w.command(b.Attempt("d5-family-wide-0", "d5-family-wide", 0), "approximation")
    for cell, baseline in (("approximation-linear", "linear"), ("approximation-continuous", "continuous")):
        command = w.command(b.Attempt("d5-family-wide-0", "d5-family-wide", 0), cell)
        assert [a for a in command if a not in ("--baseline", baseline)] == base
    assert base[base.index("--vocabulary") + 1 : base.index("--vocabulary") + 4] == ["sum", "difference", "product"]


def test_t1_and_t3s_cells_meet_their_registered_bounds():
    # paper-workplan.md D13, D14: two factors of 16 codes (256 joint), the cap the quotient's exclusion bound,
    # the suffix arm's machine (division's state and the last column's digit sum) within the codes
    from indrajala_ml.studies import modular_family as mf

    cell = b.CELLS["approximation"].args
    assert cell[cell.index("--factors") + 1] == "2" and "--codes-ratio" not in cell  # the default 16 x 16
    factors = [(16, 16), (16, 16)]
    joint = 16 * 16
    assert mf.exclusion_bound(25, factors) == joint + 25 - 1 == 280
    assert joint >= 25 * 5  # the suffix arm's 125 states
    assert {b.BACKDROPS[name].arm for name in ("d5-sum", "d5-suffix", "d5-token")} == {"sum", "suffix", "token"}
    assert all(b.BACKDROPS[n].vocabulary == b.T1_VOCABULARY for n in ("d5-sum", "d5-suffix", "d5-token"))


def test_an_opened_host_takes_work_outside_its_hours():
    thursday_noon = datetime.datetime(2026, 10, 22, 12, 0, tzinfo=SAST)  # stone's owner's day
    assert b.window_left(b.HOSTS["stone"], thursday_noon) is None
    opened = b.opened({"stone": b.HOSTS["stone"], "klip": b.HOSTS["klip"]}, ["stone"])
    assert b.window_left(opened["stone"], thursday_noon) is not None and opened["klip"] == b.HOSTS["klip"]
    assert b.HOSTS["stone"].windows  # the table itself unchanged
