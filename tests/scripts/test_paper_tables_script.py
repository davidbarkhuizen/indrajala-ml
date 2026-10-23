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
