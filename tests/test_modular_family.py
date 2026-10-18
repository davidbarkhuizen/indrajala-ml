"""The modular family's encoding and probe (the modular-family workplan, stage 2)."""

import hashlib
import json
import random
import shutil
from pathlib import Path
from typing import Any

import numpy as np
import pytest

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
    assert mf.exclusion_bound(25, [(16, 16), (16, 16)]) == 280  # the division workplan's table
    assert mf.exclusion_bound(25, [(32, 128)]) == 152
    assert mf.exclusion_bound(49, [(16, 16), (16, 16)]) == 304


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
    assert (tmp_path / "step-1.network.json").exists() and (tmp_path / "step-2.network.json").exists()
    starts = [r for r in rows if r.get("epoch") == 0 and r["step"] > 0]  # the step before's network, scored
    assert [r["step"] for r in starts] == [1, 2] and all(
        set(r) == {"step", "epoch", "screen", "kinds", "seconds"} for r in starts
    )


def test_every_epoch_logs_its_screen_score(tmp_path: Path):
    settings = mf.Settings(
        epoch_examples=64, max_epochs=3, patience=10, diagnose_every=2, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    mf.run(mf.Encoding(5, mf.QUOTIENT), [(4, 4)], 0, 1e9, tmp_path, settings, cap=1)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    epochs = [(r["epoch"], "factors" in r) for r in rows if r["step"] == 1 and "epoch" in r and "screen" in r]
    assert epochs[:4] == [(0, False), (1, False), (2, True), (3, False)]  # a diagnosis every 2 carries it too
    assert mf.trace(rows)[1].startswith("step   1 epoch   2")  # trace shows the diagnoses alone


def test_a_probe_run_stops_when_its_epochs_are_spent(tmp_path: Path):
    settings = mf.Settings(
        epoch_examples=64, max_epochs=2, diagnose_every=10, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    mf.run(mf.Encoding(5, mf.QUOTIENT), [(4, 4)], 0, 1e9, tmp_path, settings, budget_epochs=3)
    ends = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines() if '"end"' in line]
    assert ends[-1]["end"] == "spent" and len(ends) <= 3


def test_a_fork_restores_the_steps_best_network_cools_the_rate_and_stops_at_its_limit(tmp_path: Path):
    """A threshold of 2 calls every epoch that doesn't raise the peak a collapse (below twice it), so
    steps fork, each fork halving the step's rate, at most `limit` times a step."""
    settings = mf.Settings(
        epoch_examples=64, max_epochs=8, patience=10, diagnose_every=10, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    fork = mf.Fork(trigger=mf.Trigger(threshold=2.0), cool=0.5, limit=2)
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
    assert lines[0].startswith("step   1 epoch   5 codes [16, 15] share 0.000 sum 0.500")
    assert lines[-1].endswith("END plateau min 7")
    fork = {"step": 2, "epoch": 3, "fork": 1, "screen": 0.3, "peak": 0.9, "restored": 0.9, "rate": 0.0015}
    assert (
        mf.trace([row(1, [16, 15]), fork])[1]
        == "step   2 epoch   3 FORK 1: screen 0.300 peak 0.900 restored 0.900 rate 0.0015"
    )


def test_large_drops_and_whether_each_recovered():
    def row(step: int, worst: float) -> dict[str, object]:
        return {"step": step, "accuracy": {"sum": 1.0, "difference": worst}}

    rows = [row(1, 0.95), row(1, 0.5), row(2, 0.92), row(2, 0.7), row(3, 0.91), row(3, 0.2), row(4, 0.4)]
    assert mf.large_drops(rows) == [(1, True), (3, False)]  # 0.92 to 0.7 isn't large; 0.91 to 0.2 never back


def test_outcome_stops_at_the_first_certified_step_and_counts_its_epochs():
    def row(step: int, codes: int, worst: float, **extra: object) -> dict[str, object]:
        accuracy = {"sum": 1.0, "quotient-0": worst}
        return {"step": step, "accuracy": accuracy, "factors": [{"codes": codes}], **extra}

    rows = [
        row(0, 1, 0.1, epoch=0, seconds=0.2),
        row(1, 8, 0.6, end="passed", epochs=3, certified=False, seconds=30.0),
        row(2, 16, 0.997, epoch=5, seconds=200.0),  # kept to three decimals, not 1.0
        row(2, 7, 0.3, end="plateau", epochs=40, certified=False, seconds=300.0),  # a collapse
        row(3, 9, 1.0, end="passed", epochs=2, certified=True, seconds=330.0),
    ]
    assert mf.outcome(rows) == mf.Outcome(3, "certified", 45, 5.5, 1.0, 3, 1, 1, 0)
    assert mf.outcome(rows[:4]) == mf.Outcome(2, "plateau", 43, 5.0, 0.997, 2, 1, 1, 1)
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
    assert lines[1] == (
        "seed1: certified at step 4, 7 epochs, 1.0 min; worst answer at best 1.000 (step 4), collapses 0, "
        "large drops 0 (0 unrecovered)"
    )
    assert "NOT RECERTIFIED" in lines[2] and lines[3].startswith("seed10: spent")  # seeds in numeric order
    assert lines[4].startswith("family: 1 of 3 certified (Wilson 95%: 0.06-0.79)")
    assert lines[5] == "family's 2 failures: 1 came near a solution (worst answer >= 0.95), 0 collapsed"
    assert (
        lines[-4].startswith("quotient: 0 of 3 certified")
        and lines[-2] == "quotient: 0 large drops in 0 runs, 0 recovered"
    )
    assert lines[-1] == "Fisher's exact, one-sided (family more often than quotient): p 0.5"
    pairs = [("family", "quotient"), ("quotient", "family")]
    lines = mf.readout(tmp_path, ["family", "quotient"], tmp_path / "certify", pairs)
    assert lines[-2:] == [
        "Fisher's exact, one-sided (family more often than quotient): p 0.5, Holm-adjusted over 2: 1",
        "Fisher's exact, one-sided (quotient more often than family): p 1, Holm-adjusted over 2: 1",
    ]


def test_each_screen_row_logs_every_answer_kinds_share_beside_the_pooled_one(tmp_path: Path):
    settings = mf.Settings(
        epoch_examples=64, max_epochs=2, diagnose_every=10, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    mf.run(mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, 1e9, tmp_path, settings, cap=1)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    screened = [r for r in rows if "screen" in r]
    assert screened and all(set(r["kinds"]) >= {"quotient-nonzero"} for r in screened)
    assert all(0.0 <= share <= 1.0 for r in screened for share in r["kinds"].values())


def _point(step: int, epoch: int, screen: float, **kinds: float) -> dict[str, object]:
    return {"step": step, "epoch": epoch, "screen": screen, "kinds": kinds}


def test_the_trigger_scorer_counts_drops_catches_and_false_alarms_per_epoch():
    rows = [
        _point(1, 1, 0.5, a=0.4, b=0.6),  # still learning
        _point(1, 2, 0.95, a=0.92, b=0.98),
        _point(1, 3, 0.55, a=0.2, b=0.9),  # a large drop (0.92 to 0.2), pooled below 0.6 of 0.95
        _point(1, 4, 0.97, a=0.95, b=0.99),  # recovered
        _point(2, 1, 0.96, a=0.94, b=0.98),
        _point(2, 2, 0.7, a=0.85, b=0.6),  # a large drop (the worst 0.94 to 0.6) the pooled screen misses; never back
    ]
    pooled = mf.score_trigger(rows, mf.Trigger())
    assert (pooled.points, pooled.drops, pooled.caught, pooled.unrecovered) == (6, 2, 1, 1)
    assert pooled.false_alarms == 0 and pooled.early == 0
    worst = mf.score_trigger(rows, mf.Trigger(signal="worst", rule="fall", threshold=0.3))
    assert (worst.drops, worst.caught, worst.unrecovered_caught) == (2, 2, 1)


def test_the_vocabulary_keeps_every_earlier_studys_tokens_and_gives_the_product_its_own():
    assert mf.Encoding(5, mf.FAMILY).tokens == mf.COLUMNS + 2  # the studies before the product's
    assert mf.Encoding(5, mf.QUOTIENT).tokens == mf.Encoding(5, mf.FAMILY).tokens
    wide = [mf.Encoding(5, arm, mf.WIDE) for arm in (mf.QUOTIENT, mf.SUM, mf.FAMILY, mf.PRODUCT)]
    assert {e.tokens for e in wide} == {mf.COLUMNS + 3}
    assert wide[0].query_token("sum") == mf.Encoding(5, mf.SUM).query_token("sum")
    tokens, labels = mf.Encoding(5, mf.PRODUCT, mf.WIDE).example([7, 5])  # columns (2, 1) then (1, 2)
    states = mf.Encoding(5, mf.PRODUCT, mf.WIDE).states([7, 5])
    assert tokens[1::2] == (float(mf.COLUMNS + 2),) * 2
    assert labels[1::2] == tuple((a * b) % 5 for a, b in states)
    with pytest.raises(AssertionError):
        mf.Encoding(5, mf.PRODUCT)  # the product has no token in the earlier vocabulary


def _example_by_the_states(encoding: mf.Encoding, columns: list[int]) -> tuple[tuple[float, ...], tuple[int, ...]]:
    """Encoding.example as first written (before the window-program workplan's stage 1): the states and the operations."""
    tokens: list[float] = []
    labels: list[int] = []
    for token, (A, B) in zip(columns, encoding.states(columns), strict=True):
        tokens.append(float(token))
        labels.append(md.quotient(A, B, encoding.p))
        for operation in encoding.queries:
            tokens.append(float(encoding.query_token(operation)))
            labels.append(mf.OPERATIONS[operation](A, B, encoding.p))
    return tuple(tokens), tuple(labels)


@pytest.mark.parametrize("p", md.PRIMES)
def test_an_example_is_the_states_answers_for_every_encoding_and_length(p: int):
    rng = random.Random(p)
    encodings = [mf.Encoding(p, queries) for queries in (mf.QUOTIENT, mf.SUM, mf.FAMILY)] + [
        mf.Encoding(p, queries, mf.WIDE) for queries in (mf.QUOTIENT, mf.SUM, mf.PRODUCT, mf.FAMILY)
    ]
    for encoding in encodings:
        for n in [*range(65), 1024]:
            columns = mf.columns(n, rng)
            assert encoding.example(columns) == _example_by_the_states(encoding, columns)


def _rows(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in (path / "log.jsonl").read_text().splitlines()]


def test_a_guarded_run_logs_each_epochs_arming_and_rollbacks_and_each_steps_sum(tmp_path: Path):
    settings = mf.Settings(
        epoch_examples=64, max_epochs=2, diagnose_every=10, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    mf.run(mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, 1e9, tmp_path, settings, cap=2, guard=True)
    rows = _rows(tmp_path)
    epochs = [r for r in rows if r.get("epoch", 0) > 0 and "end" not in r]
    assert epochs and all(isinstance(r["armed"], bool) and r["rollbacks"] == 0 for r in epochs)
    assert all(r["updates"] == (1 if r["armed"] else 0) for r in epochs)  # one batch an epoch, counted when armed
    assert all(r["rollbacks"] == 0 for r in rows if "end" in r)


def _always_right(encoding: object, kinds: object) -> bool:
    return True


def test_the_guards_reads_leave_training_unchanged(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    """Armed every epoch, the machine read after every batch (never right, so nothing undone): the
    networks and logs are the unguarded run's, but for timings and the guard's fields."""
    settings = mf.Settings(
        epoch_examples=64, max_epochs=2, diagnose_every=1, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    encoding = mf.Encoding(5, mf.FAMILY)
    mf.run(encoding, [(4, 4)], 0, 1e9, tmp_path / "off", settings, cap=2, fork=mf.Fork())
    monkeypatch.setattr(mf, "certified_right", _always_right)
    mf.run(encoding, [(4, 4)], 0, 1e9, tmp_path / "on", settings, cap=2, fork=mf.Fork(), guard=True)
    on = _rows(tmp_path / "on")
    assert all(r["armed"] for r in on if r.get("epoch", 0) > 0 and "end" not in r)
    guard_fields = {"seconds", "armed", "updates", "rollbacks"}
    strip = [
        [{k: v for k, v in r.items() if k not in guard_fields} for r in rows] for rows in (_rows(tmp_path / "off"), on)
    ]
    assert strip[0] == strip[1]
    for name in ("step-1.network.json", "step-2.network.json"):
        assert (tmp_path / "off" / name).read_bytes() == (tmp_path / "on" / name).read_bytes()


def test_a_guard_that_undoes_nearly_every_update_for_its_stall_epochs_ends_the_run_stalled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
):
    """Armed every epoch and every update undone (right before, wrong after): the run stops
    "stalled" at its STALL_EPOCHS-th epoch, the step's network certified as any step's end."""
    from indrajala_ml.studies import certificate_guard as cg

    settings = mf.Settings(
        epoch_examples=64, max_epochs=20, patience=30, diagnose_every=30, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    reads = iter([True, False] * 100)  # each epoch: right when armed, wrong after its one update
    monkeypatch.setattr(mf, "certified_right", _always_right)

    def stub(machine: object, task: object) -> bool:
        return next(reads)

    monkeypatch.setattr(cg, "right", stub)
    mf.run(mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, 1e9, tmp_path, settings, cap=3, guard=True)
    rows = _rows(tmp_path)
    ends = [r for r in rows if "end" in r]
    assert [(r["step"], r["end"], r["epochs"]) for r in ends] == [(1, "stalled", mf.STALL_EPOCHS)]
    assert ends[0]["rollbacks"] == mf.STALL_EPOCHS and isinstance(ends[0]["certified"], bool)
    assert mf.outcome(rows).epochs == mf.STALL_EPOCHS


# the division ladder's feasibility table (docs/division-ladder-workplan.md): p, codes a factor, the cap
LADDER = {
    13: (38, 1612),
    17: (50, 2788),
    19: (56, 3496),
    23: (67, 5017),
    29: (85, 8065),
    31: (91, 9241),
    37: (108, 13032),
    41: (120, 16080),
}


@pytest.mark.parametrize("p", sorted(LADDER))
def test_the_ladders_cells_are_feasible_with_the_surplus_held(p: int):
    from fractions import Fraction

    codes, cap = LADDER[p]
    factors = mf.scaled_factors(p * p, Fraction(32, 11), 2)
    assert p in md.PRIMES and factors == [(16, codes), (16, codes)]
    assert codes >= p and codes * codes >= 8 * p * p  # a counter fits each factor; joint codes 8.5x the states
    assert mf.exclusion_bound(p * p, factors) == cap
    assert mf.scaled_factors(121, Fraction(32, 11), 2) == [(16, 32), (16, 32)]  # p 11's cell, exactly


# run --resume (the code-consolidation workplan's stage 3)

RESUMED = mf.Settings(
    epoch_examples=64, max_epochs=4, patience=10, diagnose_every=2, screen=4, cases=8, lengths=(2,), long=(4,)
)
# random-3-0 with A under it: step 2 passes, step 3 ends on a plateau
PLATEAUS = mf.Settings(
    epoch_examples=64, max_epochs=6, patience=2, diagnose_every=2, screen=4, cases=8, lengths=(2,), long=(4,)
)
RESUMED_FORK = mf.Fork(trigger=mf.Trigger(threshold=2.0), cool=0.5, limit=1)
# five runs: one forking, ending on its epochs; one guarded, ending steps at a plateau; one consolidated;
# one consolidated restoring its plateaued step's best; one forking without the commitment at step 1
RESUMABLE: dict[str, dict[str, Any]] = {
    "forked": {
        "encoding": mf.Encoding(5, mf.FAMILY),
        "settings": RESUMED,
        "cap": 2,
        "budget_epochs": 6,
        "fork": RESUMED_FORK,
    },
    "guarded": {
        "encoding": mf.Encoding(5, mf.QUOTIENT),
        "settings": mf.Settings(
            epoch_examples=64, max_epochs=6, patience=2, diagnose_every=2, screen=4, cases=8, lengths=(2,), long=(4,)
        ),
        "cap": 3,
        "budget_epochs": 10,
        "guard": True,
    },
    # the consolidation workplan's arms together: s0 2, the long tail, A's groups regrouped at a fork
    "consolidated": {
        "encoding": mf.target_named("random-3-0"),
        "settings": RESUMED,
        "cap": 3,
        "budget_epochs": 6,
        "fork": RESUMED_FORK,
        "long_tail": 4,
        "consolidate": True,
    },
    "restored": {
        "encoding": mf.target_named("random-3-0"),
        "settings": PLATEAUS,
        "cap": 3,
        "budget_epochs": 10,
        "consolidate": True,
        "restore_best": True,
    },
    "uncommitted": {
        "encoding": mf.Encoding(5, mf.FAMILY),
        "settings": RESUMED,
        "cap": 2,
        "budget_epochs": 6,
        "fork": RESUMED_FORK,
        "commitment_from": 2,
    },
}


class Stop(Exception):
    """A run stopped from outside, as a kill would stop it."""


def _resumable(out: Path, resume: bool = False, kind: str = "forked") -> None:
    config = dict(RESUMABLE[kind])
    encoding, settings = config.pop("encoding"), config.pop("settings")
    mf.run(encoding, [(4, 4)], 0, 1e9, out, settings, resume=resume, **config)


def _outcome(out: Path) -> tuple[list[dict[str, Any]], dict[str, str]]:
    # every row but its seconds, and every network's file
    rows = [json.loads(line) for line in (out / "log.jsonl").read_text().splitlines()]
    for row in rows:
        row.pop("seconds")
    return rows, {path.name: path.read_text() for path in sorted(out.glob("*.network.json"))}


def _stopped_at(monkeypatch: pytest.MonkeyPatch, module: Any, name: str, call: int, after: bool) -> None:
    # module.name raises Stop on its call-th call, before its work or after it
    original, calls = getattr(module, name), [0]

    def stopping(*args: Any, **kwargs: Any) -> Any:
        calls[0] += 1
        if calls[0] == call and not after:
            raise Stop
        result = original(*args, **kwargs)
        if calls[0] == call:
            raise Stop
        return result

    monkeypatch.setattr(module, name, stopping)


@pytest.mark.parametrize("kind", RESUMABLE)
def test_a_stopped_run_resumes_as_if_it_had_not_stopped(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, kind: str):
    # stopped just before and just after each checkpoint is written (the rows past the one before
    # cut back, or none), so within a step, at a fork and at a step's start and end; and part way
    # through an epoch
    # the guard armed by a looser rule than a perfect screen, so the kinds carried over a stop decide it
    def armed(_encoding: Any, kinds: dict[str, float]) -> bool:
        return max(kinds.values()) >= 0.6

    monkeypatch.setattr(mf, "certified_right", armed)
    writes = [0]
    write = mf._write_checkpoint

    def counted_write(*args: Any) -> None:
        writes[0] += 1
        write(*args)

    with monkeypatch.context() as counted:
        counted.setattr(mf, "_write_checkpoint", counted_write)
        _resumable(tmp_path / "whole", kind=kind)
    whole = _outcome(tmp_path / "whole")
    ends = [(r["step"], r["end"]) for r in whole[0] if "end" in r]
    if kind == "forked":
        assert any("fork" in r for r in whole[0]) and ends == [(1, "budget"), (2, "spent")]
    elif kind == "consolidated":
        assert [step for step, _ in ends] == [2, 3]  # from s0
        assert any("fork" in r and "consolidation" in r for r in whole[0])
    elif kind == "uncommitted":
        assert any("fork" in r for r in whole[0]) and [step for step, _ in ends] == [1, 2]
    elif kind == "restored":
        assert ends == [(2, "passed"), (3, "plateau")] and any("restore" in r for r in whole[0])
    else:
        assert any(r.get("armed") for r in whole[0]) and ("plateau" in dict(ends).values())
    stops = [("_write_checkpoint", n, after) for n in range(1, writes[0] + 1) for after in (False, True)]
    stops += [("screen", 3, False), ("diagnose", 4, False)]
    for name, call, after in stops[:-3] + stops[-2:]:  # stopped after the last write, the run had ended
        out = tmp_path / f"{name}-{call}-{after}"
        with monkeypatch.context() as patched:
            _stopped_at(patched, mf, name, call, after)
            with pytest.raises(Stop):
                _resumable(out, kind=kind)
        _resumable(out, resume=True, kind=kind)
        assert _outcome(out) == whole, (name, call, after)
        shutil.rmtree(out)  # /tmp may be memory: one stopped run's files at a time


def test_resuming_skips_an_ended_run_and_starts_one_not_begun(tmp_path: Path):
    _resumable(tmp_path / "new", resume=True)  # nothing to resume: it starts
    rows = (tmp_path / "new" / "log.jsonl").read_text()
    _resumable(tmp_path / "new", resume=True)  # it ended: nothing trains
    assert (tmp_path / "new" / "log.jsonl").read_text() == rows
    assert json.loads((tmp_path / "new" / mf.CHECKPOINT).read_text())["place"] == mf.ENDED


def test_resuming_refuses_another_runs_checkpoint_and_a_log_without_one(tmp_path: Path):
    _resumable(tmp_path / "run")
    with pytest.raises(AssertionError, match="another run"):
        mf.run(mf.Encoding(5, mf.FAMILY), [(4, 4)], 1, 1e9, tmp_path / "run", RESUMED, cap=2, resume=True)
    (tmp_path / "run" / mf.CHECKPOINT).unlink()
    with pytest.raises(AssertionError, match="no checkpoint"):
        _resumable(tmp_path / "run", resume=True)


# the code-consolidation workplan's stage 3: D3's s0, D4's long tail (B), A's groups and pull

# the general algebras' round-1 machines (E1), each s0 1 but random-3-0's
ROUND_1 = [f.format(n=n) for n in (3, 4, 5) for f in ("word-problem-s{n}", "move-to-front-{n}", "random-{n}-0")]


@pytest.mark.parametrize("name", [*ROUND_1, "modular-division-5", "modular-division-7"])
def test_s0_is_1_for_every_round_1_machine_but_random_3_0_s(name: str):
    from indrajala_ml.studies import window_tasks as wt

    assert mf.first_length(wt.task_named(name).machine) == (2 if name == "random-3-0" else 1)


def test_random_3_0_s_one_token_words_answer_alike_so_a_stateless_network_passes_them():
    from indrajala_ml.studies import window_tasks as wt

    machine = wt.task_named("random-3-0").machine
    assert {label for t in range(machine.tokens) for label in machine.run([t, 0])} == {2}
    two = {
        (t, label)
        for u in range(machine.tokens)
        for t in range(machine.tokens)
        for t_, label in zip((u, t, 0), machine.run([u, t, 0]), strict=True)
        if t_ == t
    }
    assert len(two) > len({t for t, _ in two})  # some token answers two ways: s0 is 2


# the plan's feasibility tables (docs/code-consolidation-workplan.md): codes a factor, joint codes,
# the cap (the exclusion bound), s0 (D3), d (D2); and stage 2b's probes
CONSOLIDATION_CELLS = {
    "random-3-0": (8, 64, 69, 2, 3),
    "word-problem-s4": (15, 225, 248, 1, 2),
    "random-4-0": (15, 225, 248, 1, 2),
}
B_PROBES = {
    "random-q8-1": (9, 81, 88),
    "random-q12-0": (11, 121, 132),
    "random-q16-0": (12, 144, 159),
    "random-perm-4-0": (15, 225, 248),
}


@pytest.mark.parametrize("name", sorted(CONSOLIDATION_CELLS))
def test_the_consolidation_cells_are_as_the_plan_s_table(name: str):
    from fractions import Fraction

    from indrajala_ml.studies import machine_reading as mr
    from indrajala_ml.studies import window_tasks as wt

    codes, joint, cap, s0, d = CONSOLIDATION_CELLS[name]
    machine = wt.task_named(name).machine
    factors = mf.scaled_factors(machine.states, Fraction(32, 11), 2)
    assert factors == [(16, codes)] * 2 and codes * codes == joint >= machine.states
    assert mf.exclusion_bound(machine.states, factors) == cap
    assert mf.first_length(machine) == s0 and mr.word_length(machine) == d


@pytest.mark.parametrize("name", sorted(B_PROBES))
def test_the_b_probes_are_as_the_plan_s_table(name: str):
    from fractions import Fraction

    from indrajala_ml.studies import window_tasks as wt

    codes, joint, cap = B_PROBES[name]
    machine = wt.task_named(name).machine
    factors = mf.scaled_factors(machine.states, Fraction(32, 11), 2)
    assert factors == [(16, codes)] * 2 and codes * codes == joint >= machine.states
    assert mf.exclusion_bound(machine.states, factors) == cap


def test_without_a_long_tail_an_epoch_s_lengths_are_the_curriculum_s_in_turn():
    assert mf.epoch_lengths(3, 7) == [1, 2, 3, 1, 2, 3, 1]


def test_the_long_tail_takes_every_other_example_beyond_the_step_to_its_factor_capped():
    lengths = mf.epoch_lengths(3, 16, long_tail=4, cap=10)
    assert lengths[::2] == [1, 2, 3, 1, 2, 3, 1, 2]  # half as before
    assert lengths[1::2] == [4, 5, 6, 7, 8, 9, 10, 4]  # 4..12, capped at 10
    assert mf.epoch_lengths(3, 16, long_tail=4, cap=40)[1::2] == [4, 5, 6, 7, 8, 9, 10, 11]
    assert mf.epoch_lengths(10, 4, long_tail=4, cap=10) == [1, 2, 3, 4]  # at the cap: the curriculum's alone


def test_a_networks_groups_count_its_reached_codes_and_its_table_is_what_it_consolidates():
    from indrajala_ml.studies import machine_reading as mr

    encoding = mf.target_named("random-3-0")
    net = mf.network(encoding, [(4, 4), (4, 4)], 0, "output")
    table, codes, groups = mf.consolidation(net, encoding, 1)
    assert codes >= groups >= 1 and len(table) == codes - groups
    assert all(duplicate != representative for duplicate, representative in table.items())
    net.consolidate(table)
    assert dict(net.consolidation) == table
    assert mf.consolidation(net, encoding, mr.word_length(encoding.task.machine))[1] == codes  # the same walk


def test_a_consolidated_run_starts_at_s0_and_logs_its_groups_at_each_diagnosis(tmp_path: Path):
    settings = mf.Settings(
        epoch_examples=64, max_epochs=4, patience=10, diagnose_every=2, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    mf.run(mf.target_named("random-3-0"), [(4, 4)], 0, 1e9, tmp_path, settings, cap=3, long_tail=4, consolidate=True)
    rows = _rows(tmp_path)
    assert [r["step"] for r in rows if "end" in r] == [2, 3]
    diagnoses = [r for r in rows if "factors" in r and isinstance(r.get("epoch"), int) and r["epoch"] > 0]
    assert diagnoses and all(set(r["consolidation"]) == {"codes", "groups", "duplicates"} for r in diagnoses)
    assert all(
        r["consolidation"]["codes"] == r["consolidation"]["groups"] + r["consolidation"]["duplicates"]
        for r in diagnoses
    )
    assert not any("consolidation" in r for r in rows if r not in diagnoses)
    saved = json.loads((tmp_path / mf.CHECKPOINT).read_text())
    assert saved["run"]["first"] == 2 and saved["run"]["long_tail"] == 4 and saved["run"]["consolidate"]


def test_a_run_without_the_plan_s_arms_keeps_its_identity_as_before(tmp_path: Path):
    _resumable(tmp_path / "run")
    identity = json.loads((tmp_path / "run" / mf.CHECKPOINT).read_text())["run"]
    assert not {"first", "long_tail", "consolidate", "restore_best", "commitment_from"} & set(identity)


# run --restore-best (the code-consolidation workplan's stage 2c, P2)


def _restoring(out: Path, cap: int, restore_best: bool) -> None:
    mf.run(
        mf.target_named("random-3-0"),
        [(4, 4)],
        0,
        1e9,
        out,
        PLATEAUS,
        cap=cap,
        consolidate=True,
        restore_best=restore_best,
    )


def test_restoring_the_best_leaves_a_run_without_a_plateau_as_it_was(tmp_path: Path):
    _restoring(tmp_path / "off", 2, False)
    _restoring(tmp_path / "on", 2, True)
    off, on = _outcome(tmp_path / "off"), _outcome(tmp_path / "on")
    assert [(r["step"], r["end"]) for r in on[0] if "end" in r] == [(2, "passed")]
    assert on[0] == off[0]
    assert {name: text for name, text in on[1].items() if name != "best.network.json"} == off[1]


def test_a_plateaued_step_ends_on_its_best_network_regrouped(tmp_path: Path):
    _restoring(tmp_path / "off", 3, False)
    _restoring(tmp_path / "on", 3, True)
    rows = _rows(tmp_path / "on")
    assert [(r["step"], r["end"]) for r in rows if "end" in r] == [(2, "passed"), (3, "plateau")]
    (restore,) = [r for r in rows if "restore" in r]
    assert restore["step"] == 3 and restore["restored"] == restore["restore"]
    assert restore["restore"] == max(
        r["screen"] for r in rows if r["step"] == 3 and r.get("epoch", 0) > 0 and "screen" in r
    )
    assert set(restore["consolidation"]) == {"codes", "groups", "duplicates"}
    # the step's saved network is its best, which the run without the restore didn't end on
    on = tmp_path / "on"
    assert (on / "step-3.network.json").read_bytes() == (on / "best.network.json").read_bytes()
    assert (on / "step-3.network.json").read_bytes() != (tmp_path / "off" / "step-3.network.json").read_bytes()
    assert any("RESTORE" in line for line in mf.trace(rows))
    assert json.loads((on / mf.CHECKPOINT).read_text())["run"]["restore_best"]


# run --commitment-from (the commitment-control workplan's diagnostic)


def test_the_commitment_from_the_first_step_is_the_run_without_it(tmp_path: Path):
    configs: dict[str, dict[str, Any]] = {"off": {}, "on": {"commitment_from": 1}}
    for name, config in configs.items():
        mf.run(
            mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, 1e9, tmp_path / name, RESUMED, cap=2, fork=RESUMED_FORK, **config
        )
    assert _outcome(tmp_path / "on") == _outcome(tmp_path / "off")


def test_the_steps_before_commitment_from_train_without_the_commitment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    # every epoch's training (a fork's too) reads the commitments its step should have
    seen: list[list[float]] = []
    train_epoch = mf.wr.train_epoch

    def recording(net: Any, *args: Any) -> None:
        seen.append(list(net.recurrence_layer.commitments))
        train_epoch(net, *args)

    monkeypatch.setattr(mf.wr, "train_epoch", recording)
    out = tmp_path / "run"
    mf.run(mf.Encoding(5, mf.FAMILY), [(4, 4)], 0, 1e9, out, RESUMED, cap=2, fork=RESUMED_FORK, commitment_from=2)
    rows = _rows(out)
    trained = [r["step"] for r in rows if r.get("epoch", 0) > 0 and "screen" in r]  # an epoch's row or its fork's
    assert len(trained) == len(seen) and set(trained) == {1, 2}
    assert [c for step, c in zip(trained, seen, strict=True) if step == 1] == [[0.0]] * trained.count(1)
    assert [c for step, c in zip(trained, seen, strict=True) if step == 2] == [[0.25]] * trained.count(2)
    assert json.loads((out / mf.CHECKPOINT).read_text())["run"]["commitment_from"] == 2
