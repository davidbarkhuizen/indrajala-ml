"""
A saved network's ONNX export (the export-and-hub workplan, D14): a format-2 file or a format-3
directory to OUT, checked against our numpy forward (D13), the check's report printed.

    .venv/bin/python scripts/export/export_onnx.py model.json model.onnx [--dtype float64]
"""

from __future__ import annotations

import argparse
import sys

from indrajala_ml.model.export.onnx.exporter import export_onnx


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("path", help="a format-2 file or a format-3 directory")
    parser.add_argument("out", help="the .onnx file to write")
    parser.add_argument(
        "--dtype", choices=("float32", "float64"), default=None, help="the export's dtype (default float32)"
    )
    parser.add_argument("--no-check", action="store_true", help="skip the check against our numpy forward")
    args = parser.parse_args(argv)
    export = export_onnx(args.path, args.out, dtype=args.dtype, check=not args.no_check)
    print(f"{args.out}: {export.dtype}", file=sys.stderr)
    if export.report is not None:
        for name, difference in export.report.max_difference.items():
            disagreements = export.report.argmax_disagreements[name]
            print(
                f"  {name}: largest difference {difference:.3g} (tolerance {export.report.tolerance:g}), "
                f"{disagreements} of {export.report.rows} argmaxes differ",
                file=sys.stderr,
            )
    return 0


if __name__ == "__main__":
    sys.exit(main())
