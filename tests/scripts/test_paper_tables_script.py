"""scripts/paper_tables.py: the paper's tables regenerated and checked against a manuscript."""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts"))
import paper_tables as script  # scripts/ isn't a package

TEX = r"""
\begin{table}[t]
\caption{Another.}\label{tab:other}
\begin{tabular}{l}\toprule a \\ \midrule x & 1 \\ \bottomrule\end{tabular}
\end{table}
\begin{table}[t]
\caption{The one.}
\label{tab:toy}
\begin{tabular}{lr}
\toprule
arm & n \\
\midrule
first & 1 \\
second & 2 \\
\bottomrule
\end{tabular}
\end{table}
"""


def test_span_writes_ranges_as_the_manuscripts_do():
    assert script.span([]) == "--" and script.span([7]) == "7" and script.span([12, 6, 9]) == "6--12"


def rows(*lines: str) -> script.Table:
    return lambda data: list(lines)


def test_check_reads_the_labelled_tables_rows_and_flags_a_difference(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
):
    assert script.manuscript_rows(TEX, "tab:toy") == ["first & 1 \\\\", "second & 2 \\\\"]
    assert script.manuscript_rows(TEX, "tab:missing") is None
    tex = tmp_path / "paper.tex"
    tex.write_text(TEX)
    monkeypatch.setattr(script, "TABLES", {"tab:toy": rows("first & 1 \\\\", "second & 2 \\\\")})
    assert script.main(["check", str(tex)]) == 0
    monkeypatch.setattr(script, "TABLES", {"tab:toy": rows("first & 1 \\\\", "second & 3 \\\\")})
    assert script.main(["check", str(tex)]) == 1
    printed = capsys.readouterr().out
    assert "tab:toy: 2 rows, the same" in printed and "regenerated: second & 3" in printed


def test_median_is_whole_or_to_a_half():
    assert script.median([157, 182, 200, 394]) == "191" and script.median([152, 184, 207, 309]) == "195.5"
    assert script.median([95, 120, 230]) == "120"


def arm_run(epochs: int, end: str = "certified", certified: bool = True, long: bool | None = True, peak: float = 0.5):
    run = script.mf.Outcome(3, end, epochs, 1.0, peak, 2, 1 if peak < 0.5 else 0, 0, 0)
    return script.mf.ArmRun(300, run, certified, {}, True, long)


def test_rate_and_arm_tables_count_only_successes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path):
    runs = [
        arm_run(100),
        arm_run(300),
        arm_run(200, long=False),  # certified, a long test failed: a failure, but not an uncertified one
        arm_run(600, "spent", False, None, script.mf.NEAR),
        arm_run(600, "spent", False, None, 0.4),
    ]

    def batch_runs(data: Path, test: str, arm: str) -> list[script.mf.ArmRun]:
        return runs

    monkeypatch.setattr(script, "batch_runs", batch_runs)
    assert script.rates_table([("s4", 24)])(tmp_path) == [
        "\\texttt{s4} & 24 & 2 of 5 & 0.12--0.77 & 100--300, 200 & 1 of 3 \\\\"
    ]
    assert script.arms_table("t1", [("sum", "sum")])(tmp_path) == ["sum & 2 of 5 & 0.12--0.77 & 1 of 3 & 1 \\\\"]


LINE = (
    "/x/certified-step-19.network.json: 24 states, meet 4, cover False, cascade top 0; factor 0: 11 codes,"
    " approximation 4 blocks (congruence False, autonomous 0.545), determines places {3}; factor 1: 10 codes,"
    " approximation 1 blocks (congruence False, autonomous 0.440), determines -"
)


def test_a_decomposition_row_from_text_and_json_readings(tmp_path: Path):
    (tmp_path / "s4.txt").write_text(LINE + "\n")
    factors = [{"approximation": [5] * 5, "determines": [line]} for line in ("A + B", "A - B")]
    net = {"states": 25, "cover": True, "factors": factors}
    (tmp_path / "family").mkdir()
    (tmp_path / "family" / "a.json").write_text(script.json.dumps(net))
    assert script.readings(tmp_path / "s4.txt") == [script.Reading(24, False, [(4, ["places {3}"]), (1, [])])]
    rows = script.decomposition_row("s4", script.readings(tmp_path / "s4.txt") + script.readings(tmp_path / "family"))
    assert rows == "s4 & 2 & 3/4 & 2 (1) & 1 & \\texttt{A + B} 1, \\texttt{A - B} 1, \\texttt{places \\{3\\}} 1 \\\\"


def test_proof_outcomes_name_the_stop_and_chows_verdict():
    chow = {"equivalent": False, "words": 4096, "counterexample": [1, 0]}
    assert script.proof_outcome({"verdict": "silent"}) == "silent"
    assert script.proof_outcome({"verdict": "false-stop", "chow": chow}) == "false-stop refuted"
    assert script.proof_outcome({"verdict": "kept", "chow": {"refused": "limit"}}) == "kept out of reach"
    assert script.natural("d7-family-300") < script.natural("d11-family-300")
