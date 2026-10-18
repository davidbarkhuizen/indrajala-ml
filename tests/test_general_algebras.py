"""The general algebras' machines and their probe target (docs/general-algebras-workplan.md, D1-D3, D6, D7)."""

import json
import math
import random
from fractions import Fraction
from pathlib import Path

import numpy as np
import pytest

from indrajala_ml.studies import modular_family as mf
from indrajala_ml.studies import window_tasks as wt

CLASSES = ("word-problem-s{n}", "move-to-front-{n}", "random-{n}-0")
SIZES = (3, 4, 5)
# the workplan's feasibility table: n -> codes a factor, the exclusion bound (the cap)
FEASIBILITY = {3: (8, 69), 4: (15, 248), 5: (32, 1143)}


def _tasks() -> list[tuple[int, str]]:
    return [(n, form.format(n=n)) for n in SIZES for form in CLASSES]


@pytest.mark.parametrize(("n", "name"), _tasks())
def test_every_machine_has_n_factorial_reachable_minimal_states_n_plus_one_tokens_and_n_classes(n: int, name: str):
    machine = wt.task_named(name).machine
    assert machine.states == math.factorial(n) and machine.tokens == n + 1 and machine.classes == n
    assert len(machine.reachable()) == machine.states
    assert machine.distinguishing_length() is not None  # minimal (D2)
    assert (machine.transition[:, 0] == np.arange(machine.states)).all()  # token 0 the identity


@pytest.mark.parametrize(("n", "name"), _tasks())
def test_the_cells_are_feasible_with_the_surplus_held(n: int, name: str):
    q = wt.task_named(name).machine.states
    codes, cap = FEASIBILITY[n]
    factors = mf.scaled_factors(q, Fraction(32, 11), 2)
    assert factors == [(16, codes), (16, codes)]
    assert codes * codes >= 8 * q  # joint codes at least 8 times the states (8.5 at 120, division p 11's)
    assert mf.exclusion_bound(q, factors) == cap


@pytest.mark.parametrize("n", SIZES)
def test_the_group_s_tokens_are_permutations_and_move_to_front_s_are_idempotent(n: int):
    group, band = wt.word_problem_machine(n), wt.move_to_front_machine(n)
    for t in range(group.tokens):
        assert sorted(group.transition[:, t]) == list(range(group.states))  # a bijection
    for t in range(1, band.tokens):
        after = band.transition[:, t]
        assert (band.transition[after, t] == after).all()  # t t = t: no group part
        assert len(set(after)) < band.states


def test_the_group_s_output_is_the_item_at_position_1_and_move_to_front_s_the_last_item():
    states = wt.orderings(3)
    group, band = wt.word_problem_machine(3), wt.move_to_front_machine(3)
    # from (0, 1, 2): swap positions 1 and 3, then rotate: (2, 1, 0) -> (1, 0, 2)
    assert [states[s] for s in group.path([2, 3])] == [(2, 1, 0), (1, 0, 2)]
    assert group.run([2, 3]) == (2, 1)
    # access item 2, then item 1: (2, 0, 1) -> (1, 2, 0); the last item each time
    assert [states[s] for s in band.path([3, 2])] == [(2, 0, 1), (1, 2, 0)]
    assert band.run([3, 2]) == (1, 0)


def test_a_random_machine_is_the_same_for_its_seed_and_another_for_another():
    assert (wt.random_machine(4, 0).transition == wt.task_named("random-4-0").machine.transition).all()
    assert not (wt.random_machine(4, 0).transition == wt.random_machine(4, 1).transition).all()


def test_the_machine_task_passes_a_right_predictor_and_fails_a_wrong_one_and_s1_covers_every_state():
    task = wt.task_named("move-to-front-3")
    assert isinstance(task, wt.MachineTask)
    properties = task.catalogue(50, 0, 3)
    right = task.evaluate(lambda words: [task.machine.run((*w, 0)) for w in words], 8, properties)
    assert all(r.holds for r in right) and [r.cases for r in right] == [50, 18]
    assert set(right[1].strata) == set(range(6))
    wrong = task.evaluate(lambda words: [(0,) * (len(w) + 1) for w in words], 8, properties)
    assert not any(r.holds for r in wrong) and wrong[0].failures
    assert task.result_json(wrong[0])["failures"][0] == {"word": list(wrong[0].failures[0])}


def test_a_machine_target_is_one_token_a_window_its_kinds_by_token():
    encoding = mf.target_named("word-problem-s3")
    assert isinstance(encoding, mf.MachineEncoding)
    assert (encoding.width, encoding.tokens, encoding.classes, encoding.target_states) == (1, 4, 3, 6)
    word = encoding.draw(7, random.Random(0))
    assert len(word) == 8 and set(word) <= set(range(4)) and word[-1] == 0  # the final identity
    tokens, labels = encoding.example(word)
    assert tokens == tuple(float(t) for t in word) and labels == encoding.task.machine.run(word)
    assert encoding.kinds(tokens, labels) == [f"token-{t}" for t in word]
    assert set(encoding.quantities) == {"position-1", "position-2", "position-3"}
    assert isinstance(mf.target_named("modular-division-5"), mf.Encoding)


def test_a_probe_run_trains_a_machine_and_certifies_each_step_against_it(tmp_path: Path):
    settings = mf.Settings(
        epoch_examples=64, max_epochs=2, diagnose_every=1, screen=4, cases=8, lengths=(2,), long=(4,)
    )
    mf.run(mf.target_named("move-to-front-3"), [(4, 4), (4, 4)], 0, 1e9, tmp_path, settings, cap=2, fork=mf.Fork())
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    ends = [r for r in rows if "end" in r]
    assert [r["step"] for r in ends] == [1, 2]
    assert set(rows[0]["accuracy"]) <= {f"token-{t}" for t in range(4)}
    assert set(rows[0]["factors"][0]["shares"]) == {"position-1", "position-2", "position-3"}
    assert all(isinstance(r["certified"], bool) and r["minimised"] >= 1 for r in ends)
    certificate = mf.certify_file(tmp_path / "network.json", mf.target_named("move-to-front-3"), "move-to-front-3")
    assert certificate["task"] == "move-to-front-3" and certificate["slot"] == 0


# the random-class workplan's E1b: n 3's machines (D3: two random draws whose stateless guess is at
# most halfway from chance to 1), at the general algebras' cell, the curriculum from length 2 (D2)
E1B = ("word-problem-s3", "move-to-front-3", "random-3-1", "random-3-2")


@pytest.mark.parametrize("name", E1B)
def test_e1b_s_machines_are_reachable_minimal_and_feasible_at_n_3_s_cell(name: str):
    machine = wt.task_named(name).machine
    assert machine.states == 6 and machine.tokens == 4 and machine.classes == 3
    assert len(machine.reachable()) == 6 and machine.distinguishing_length() is not None
    codes, cap = FEASIBILITY[3]
    factors = mf.scaled_factors(machine.states, Fraction(32, 11), 2)
    assert factors == [(16, codes), (16, codes)] and mf.exclusion_bound(6, factors) == cap


def test_the_stateless_guess_is_chance_on_the_group_and_e1b_s_draws_are_within_the_bound():
    bound = 1 / 3 + (1 - 1 / 3) / 2
    assert wt.stateless_accuracy(wt.task_named("word-problem-s3").machine) == pytest.approx(1 / 3)
    assert wt.stateless_accuracy(wt.task_named("random-3-0").machine) > bound  # round 1's: little state needed
    assert wt.stateless_accuracy(wt.random_machine(3, 4)) == pytest.approx(1.0)  # none at all
    # the first two seeds from 1 within the bound
    assert [s for s in range(1, 4) if wt.stateless_accuracy(wt.random_machine(3, s)) <= bound][:2] == [1, 2]


def test_the_curriculum_starts_at_first_length_and_its_last_step_trains_to_the_cap(tmp_path: Path):
    assert mf.step_lengths(3, mf.PROBE) == [1, 2, 3]  # the default, as before
    settings = mf.Settings(
        epoch_examples=64, max_epochs=2, diagnose_every=1, screen=4, cases=8, lengths=(2,), long=(4,), first_length=2
    )
    assert mf.step_lengths(1, settings) == [1, 2]
    mf.run(mf.target_named("move-to-front-3"), [(4, 4), (4, 4)], 0, 1e9, tmp_path, settings, cap=3)
    rows = [json.loads(line) for line in (tmp_path / "log.jsonl").read_text().splitlines()]
    starts = [r for r in rows if r.get("epoch") == 0 and r["step"] > 0]
    assert [(r["step"], r["longest"]) for r in starts][:2] == [(1, 2), (2, 3)]
    assert max(r["step"] for r in rows if "end" in r) <= 2  # step 2 trains to the cap, 3
