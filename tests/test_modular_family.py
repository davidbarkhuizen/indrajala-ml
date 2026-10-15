"""The modular family's encoding and probe (docs/modular-family-workplan.md, stage 2)."""

import hashlib
import json
import random
from pathlib import Path

import numpy as np

from indrajala_ml.data import modular_data as md
from indrajala_ml.studies import modular_family as mf


def test_the_column_answers_division_and_the_queries_their_operations():
    rng = random.Random(0)
    for p in (5, 7):
        encoding = mf.Encoding(p, mf.FAMILY)
        for _ in range(50):
            n = rng.randrange(1, 9)
            pair = (rng.randrange(3**n), rng.randrange(3**n))
            columns = md.column_tokens(pair, n)
            tokens, labels = encoding.example(columns)
            assert len(tokens) == len(labels) == 3 * n
            assert labels[0::3] == md.answers(pair, n, p)
            assert labels[1::3] == md.sums(pair, n, p)
            a, b = md.digits(pair[0], n), md.digits(pair[1], n)
            x = y = 0  # the prefixes mod p
            for t in reversed(range(n)):
                x, y = (3 * x + a[t]) % p, (3 * y + b[t]) % p
            assert labels[-1] == (x - y) % p
            assert tokens[0::3] == tuple(float(c) for c in columns)
            assert set(tokens[1::3]) == {9.0} and set(tokens[2::3]) == {10.0}


def test_the_quotient_alone_is_division_one_token_a_window():
    encoding = mf.Encoding(5, mf.QUOTIENT)
    pair, n = (17, 40), 4
    tokens, labels = encoding.example(md.column_tokens(pair, n))
    assert encoding.width == 1 and len(tokens) == n
    assert labels == md.answers(pair, n, 5)
    assert encoding.tokens == 11 and encoding.classes == 6


def test_the_sum_arm_asks_the_sum_alone_with_the_familys_token():
    encoding = mf.Encoding(5, mf.SUM)
    pair, n = (17, 40), 4
    tokens, labels = encoding.example(md.column_tokens(pair, n))
    assert encoding.width == 2 and len(tokens) == 2 * n
    assert labels[0::2] == md.answers(pair, n, 5) and labels[1::2] == md.sums(pair, n, 5)
    assert set(tokens[1::2]) == {9.0} and encoding.tokens == 11


def test_the_cap_is_the_cells_exclusion_bound():
    assert mf.exclusion_bound(5, [(16, 16), (16, 16)]) == 280  # the division workplan's table
    assert mf.exclusion_bound(5, [(32, 128)]) == 152
    assert mf.exclusion_bound(7, [(16, 16), (16, 16)]) == 304


def test_a_probe_run_trains_diagnoses_and_saves(tmp_path: Path):
    settings = mf.Settings(
        epoch_examples=64, max_epochs=2, diagnose_every=1, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    mf.run(mf.Encoding(5, mf.FAMILY), [(4, 4), (4, 4)], 0, 1e9, tmp_path, settings, cap=2)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    ends = [r for r in rows if "end" in r]
    assert [r["step"] for r in ends] == [1, 2]
    assert set(rows[0]["accuracy"]) >= {"sum", "difference"}
    assert len(rows[0]["factors"]) == 2 and set(rows[0]["factors"][0]["shares"]) == {"A", "B", "S", "D", "A=0", "B=0"}
    assert set(ends[-1]["long"]) == {"4"}
    assert all(isinstance(r["certified"], bool) and r["minimised"] >= 1 for r in ends)  # each step's certificate
    assert (tmp_path / "network.json").exists()


def test_a_probe_run_stops_when_its_epochs_are_spent(tmp_path: Path):
    settings = mf.Settings(
        epoch_examples=64, max_epochs=2, diagnose_every=10, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    mf.run(mf.Encoding(5, mf.QUOTIENT), [(4, 4)], 0, 1e9, tmp_path, settings, budget_epochs=3)
    ends = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines() if '"end"' in line]
    assert ends[-1]["end"] == "spent" and len(ends) <= 3


def test_a_fork_restores_the_steps_best_network_cools_the_rate_and_stops_at_its_limit(tmp_path: Path):
    """drop -1 calls every epoch that doesn't raise the peak a collapse (below twice it), so steps
    fork, each fork halving the step's rate, at most `limit` times a step."""
    settings = mf.Settings(
        epoch_examples=64, max_epochs=8, patience=10, diagnose_every=10, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    fork = mf.Fork(drop=-1.0, cool=0.5, limit=2)
    mf.run(mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, 1e9, tmp_path, settings, cap=2, fork=fork)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    forks = [r for r in rows if "fork" in r]
    ends = {r["step"]: r["forks"] for r in rows if "end" in r}
    assert forks and set(ends) == {1, 2}
    for step, count in ends.items():
        mine = [r for r in forks if r["step"] == step]
        assert len(mine) == count <= fork.limit
        assert [(r["fork"], r["rate"]) for r in mine] == [(f, 0.003 * 0.5**f) for f in range(1, count + 1)]
    assert all(r["screen"] < 2 * r["peak"] and r["restored"] == r["peak"] for r in forks)  # the peak's network back
    assert (tmp_path / "snapshot.network.json").exists()


def test_without_a_fork_a_run_logs_no_forks(tmp_path: Path):
    settings = mf.Settings(
        epoch_examples=64, max_epochs=2, diagnose_every=10, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    mf.run(mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, 1e9, tmp_path, settings, cap=1)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    assert not any("fork" in r or "forks" in r for r in rows)
    assert not (tmp_path / "snapshot.network.json").exists()


def test_the_query_machine_runs_each_column_with_its_queries_and_reads_one_slot():
    from indrajala_ml.studies import window_recurrent_certificate as wc

    encoding = mf.Encoding(5, mf.FAMILY)
    net = mf.network(encoding, [(4, 4), (4, 4)], 0, reads="output")
    state, _ = wc.NetworkMachine(net).start()
    columns = np.arange(9, dtype=np.int64)[:, np.newaxis]
    full = np.concatenate([columns, np.full((9, 1), 9), np.full((9, 1), 10)], axis=1)
    whole = wc.NetworkMachine(net).step(state, full)
    for slot in range(3):
        steps = mf.QueryMachine(net, encoding, slot).step(state, columns)
        assert steps.digits.shape == (9, 1)
        assert (steps.digits[:, 0] == whole.digits[:, slot]).all()
        assert (steps.states == whole.states).all()


class Oracle:
    """A stand-in network answering every window right: its rows' own labels."""

    def __init__(self, columns: int, shift: int = 0) -> None:
        self.columns, self.shift = columns, shift

    def prepare_dataset(self, rows: list[mf.Example]) -> list[mf.Example]:
        return rows

    def classify_rows(self, rows: list[mf.Example]) -> list[tuple[int, ...]]:
        return [labels[self.shift :] + labels[: self.shift] for _, labels in rows]


def test_the_query_model_reads_its_slot_so_a_right_network_passes_the_long_tests():
    from indrajala_ml.studies import window_tasks as wt

    encoding = mf.Encoding(5, mf.FAMILY)
    for name, slot in (("modular-division-5", 0), ("modular-sum-5", 1)):
        task = wt.task_named(name)
        tested = [p for p in task.catalogue(50, 50, 4) if p.id in mf.LONG_PROPERTIES]
        model = mf.QueryModel(Oracle(3), encoding, task, slot)
        for n in (2, 9):
            results = task.evaluate(model.predict(n), n, tested)
            assert [r.id for r in results] == ["U1", "S1"] and all(r.holds for r in results)
        # the wrong slot's labels fail
        wrong = mf.QueryModel(Oracle(3, shift=1), encoding, task, slot)
        assert not all(r.holds for r in task.evaluate(wrong.predict(9), 9, tested))


def test_a_saved_networks_long_test_is_the_harness_json(tmp_path: Path):
    from indrajala_ml.studies import window_recurrent_study as wr

    encoding = mf.Encoding(5, mf.FAMILY)
    path = tmp_path / "network.json"
    mf.network(encoding, [(4, 4), (4, 4)], 0, reads="output").save(str(path))
    long = mf.long_test(path, encoding, "modular-division-5", lengths=(2, 3))
    assert set(long["lengths"]) == {"2", "3"} and long["slot"] == 0 and len(long["sha256"]) == 64
    assert set(long["lengths"]["3"]["properties"]) == {"U1", "S1"}
    cases, _ = wr.long_test_failures(long)
    assert cases == 3**4 + 3**6 + 2 * 25 * 1000  # U1 every pair at 2 columns and at 3; S1 1,000 a state


def test_trace_marks_a_collapse_where_the_codes_in_use_halve():
    def row(step: int, codes: list[int], **extra: object) -> dict[str, object]:
        return {"step": step, "epoch": 5, "accuracy": {"sum": 0.5}, "factors": [{"codes": c} for c in codes], **extra}

    lines = mf.trace([row(1, [16, 15]), row(1, [14, 14]), row(2, [3, 4]), row(2, [4, 4], end="plateau", minimised=7)])
    assert ["collapse" in line for line in lines] == [False, False, True, False]
    assert lines[0].startswith("step   1 epoch   5 codes [16, 15] share 0.00 sum 0.50")
    assert lines[-1].endswith("END plateau min 7")


def test_outcome_stops_at_the_first_certified_step_and_counts_its_epochs():
    def row(step: int, codes: int, worst: float, **extra: object) -> dict[str, object]:
        accuracy = {"sum": 1.0, "quotient-0": worst}
        return {"step": step, "accuracy": accuracy, "factors": [{"codes": codes}], **extra}

    rows = [
        row(0, 1, 0.1, epoch=0, seconds=0.2),
        row(1, 8, 0.6, end="passed", epochs=3, certified=False, seconds=30.0),
        row(2, 16, 0.97, epoch=5, seconds=200.0),
        row(2, 7, 0.3, end="plateau", epochs=40, certified=False, seconds=300.0),  # a collapse
        row(3, 9, 1.0, end="passed", epochs=2, certified=True, seconds=330.0),
    ]
    assert mf.outcome(rows) == mf.Outcome(3, "certified", 45, 5.5, 1.0, 3, 1)
    assert mf.outcome(rows[:4]) == mf.Outcome(2, "plateau", 43, 5.0, 0.97, 2, 1)
    assert mf.outcome(rows[:1]).end == "running"
    spent = row(3, 7, 0.4, end="spent", epochs=6, certified=False, seconds=330.0)  # 5 trained
    assert mf.outcome([*rows[:4], spent]).epochs == 48


def test_readout_counts_a_run_only_when_recertified_in_both_verdicts(tmp_path: Path):
    def run(arm: str, seed: int, certified: bool, network: bytes = b"") -> None:
        out = tmp_path / arm / f"seed{seed}"
        out.mkdir(parents=True)
        end = {
            "step": 4,
            "end": "passed" if certified else "spent",
            "epochs": 7,
            "certified": certified,
            "seconds": 60.0,
            "accuracy": {"sum": 1.0 if certified or seed == 10 else 0.5},  # seed 10 came near, then failed
            "factors": [{"codes": 9}],
        }
        (out / "log.jsonl").write_text(json.dumps(end) + "\n")
        if certified:
            (out / "certified-step-4.network.json").write_bytes(network)

    def certificate(name: str, network: bytes, ratio: float) -> None:
        sha = hashlib.sha256(network).hexdigest()
        result = {"sha256": sha, "snap_ratio": ratio, "certified": True}
        (tmp_path / "certify" / f"{name}.certificate.json").write_text(json.dumps(result))

    (tmp_path / "certify").mkdir()
    run("family", 1, True, b"one")
    run("family", 2, True, b"two")  # only one verdict: not counted
    run("family", 10, False)
    run("quotient", 1, False)
    run("quotient", 2, False)
    run("quotient", 3, False)
    for ratio in mf.VERDICTS:
        certificate(f"one-{ratio}", b"one", ratio)
    certificate("two", b"two", 1.0)
    lines = mf.readout(tmp_path, ["family", "quotient"], tmp_path / "certify")
    assert lines[1] == "seed1: certified at step 4, 7 epochs, 1.0 min; worst answer at best 1.00 (step 4), collapses 0"
    assert "NOT RECERTIFIED" in lines[2] and lines[3].startswith("seed10: spent")  # seeds in numeric order
    assert lines[4].startswith("family: 1 of 3 certified (Wilson 95%: 0.06-0.79)")
    assert lines[5] == "family's 2 failures: 1 came near a solution (worst answer >= 0.95), 0 collapsed"
    assert lines[-3].startswith("quotient: 0 of 3 certified")
    assert lines[-1] == "Fisher's exact, one-sided (family more often than quotient): p 0.5"
