import importlib
import sys
import textwrap

from indrajala_ml.demos.registry import DEMOS

DESCRIPTION_WRAP_WIDTH = 92


def _print_table() -> None:

    title_width = max(len(demo.title) for demo in DEMOS)

    print()
    print(f"{'#':>2}  {'demo':<{title_width}}  description")
    print(f"{'--':>2}  {'-' * title_width}  {'-' * 11}")
    for index, demo in enumerate(DEMOS, start=1):
        print(f"{index:>2}  {demo.title:<{title_width}}  {demo.summary}")
    print()


def _read_selection() -> int | None:

    while True:
        try:
            raw = input("select a demo by number, or 'q' to quit: ").strip().lower()
        except (EOFError, KeyboardInterrupt):
            print()
            return None

        if raw in ("q", "quit", "exit"):
            return None

        if raw.isdigit() and 1 <= int(raw) <= len(DEMOS):
            return int(raw)

        print(f"'{raw}' isn't a valid choice - enter a number from 1 to {len(DEMOS)}, or 'q'.")


def _run_demo(index: int) -> None:

    demo = DEMOS[index - 1]

    print()
    print(demo.title)
    print("=" * len(demo.title))
    print(textwrap.fill(demo.description, DESCRIPTION_WRAP_WIDTH))
    print()
    print(f"launching {demo.module} ...")
    print()

    try:
        importlib.import_module(demo.module).main()
    except SystemExit as error:
        if error.code not in (None, 0):
            print(f"\n{demo.title} exited early (code {error.code}).")

    print()
    print(f"{demo.title} finished.")


def _parse_direct_selection(raw: str) -> int:

    if not raw.isdigit() or not (1 <= int(raw) <= len(DEMOS)):
        print(f"'{raw}' isn't a valid demo number - pass a number from 1 to {len(DEMOS)}.")
        _print_table()
        sys.exit(1)

    return int(raw)


def main(argv: list[str] | None = None) -> None:

    argv = sys.argv[1:] if argv is None else argv

    if argv:
        # `./cli demo 3`: run that demo and exit, without the menu, so scripts can launch one
        _run_demo(_parse_direct_selection(argv[0]))
        return

    print("indrajala-ml demos")
    print("select a demo to run; after it finishes you can run another or quit.")

    while True:
        _print_table()
        selection = _read_selection()
        if selection is None:
            print("goodbye.")
            return
        _run_demo(selection)


if __name__ == "__main__":
    main()
