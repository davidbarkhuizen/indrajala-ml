"""
Which functions' machine code differs between two builds of the crate's extension (`.so`): the
crate kernels workplan's stability check (D7). A change to one kernel should leave every other
kernel's code as it was; when it doesn't, an A/B's rows for code the change never touched move
with it, and the A/B can't be read (docs/measurement.md, Gotchas).

Each function's instructions come from `objdump -d -C --no-show-raw-insn`, normalized so that only
the code counts, not where the linker put it: addresses within the function become offsets from
its start, other addresses the symbol they point into, and the crate's disambiguation hashes are
dropped from symbol names. A function is "same" when its normalized instructions are equal.

    python scripts/measurement/kernel_code_diff.py OLD.so NEW.so [--match REGEX] [--all]

It prints the functions whose code changed (with both sizes in instructions), those only in one
build, and a count of those unchanged; `--match` keeps the functions whose name matches (default:
the crate's own, `indrajala_math_rust::`). It exits 1 if any matching function changed.
"""

import argparse
import re
import subprocess
import sys
from pathlib import Path

HEADER = re.compile(r"^([0-9a-f]+) <(.+)>:$")
ADDRESS = re.compile(r"\b([0-9a-f]{4,16})\b(?: <([^>]+)>)?")
HASH = re.compile(r"\[[0-9a-f]{16}\]|::h[0-9a-f]{16}\b")
# where the linker put data, not what the code does: x86's RIP-relative displacements and the
# targets objdump names after `#` (a GOT slot, a constant), aarch64's page offsets after an adrp
RIP = re.compile(r"-?0x[0-9a-f]+\(%rip\)")
COMMENT = re.compile(r"#\s*(?:(?:0x)?[0-9a-f]+\s*)?<[^>]*>|#\s*(?:0x)?[0-9a-f]+$")
ADRP = re.compile(r"^adrp\s+(\w+),")
PAGE_OFFSET = re.compile(r"#(?:0x[0-9a-f]+|\d+)")
DESTINATION = re.compile(r"^(?!st)\w+\s+(\w+),")


def functions(so: Path) -> dict[str, list[str]]:
    """Each function's normalized instructions, by demangled name."""
    return parse(
        subprocess.run(
            ["objdump", "-d", "-C", "--no-show-raw-insn", str(so)], capture_output=True, text=True, check=True
        ).stdout
    )


def parse(out: str) -> dict[str, list[str]]:
    """`objdump -d -C --no-show-raw-insn`'s output, each function's instructions normalized."""
    found: dict[str, list[str]] = {}
    name: str | None = None
    start = 0
    body: list[str] = []
    pages: set[str] = set()
    for line in out.splitlines():
        if header := HEADER.match(line):
            if name is not None:
                found[name] = body
            name, start, body, pages = HASH.sub("", header.group(2)), int(header.group(1), 16), [], set[str]()
            continue
        if name is None or ":" not in line or not line.startswith(" "):
            continue
        instruction = COMMENT.sub("# DATA", RIP.sub("REL(%rip)", line.split(":", 1)[1].strip()))
        if adrp := ADRP.match(instruction):
            pages.add(adrp.group(1))
            instruction = f"adrp {adrp.group(1)}, PAGE"
        else:
            if pages and any(f" {r}, #" in instruction or f"[{r}, #" in instruction for r in pages):
                instruction = PAGE_OFFSET.sub("#LO12", instruction)
            if (written := DESTINATION.match(instruction)) is not None:
                pages.discard(written.group(1))  # no longer a page's base

        def local(match: re.Match[str], start: int = start, name: str = name, size_hint: int = 1 << 20) -> str:
            address, symbol = int(match.group(1), 16), match.group(2)
            if symbol is not None:
                symbol = HASH.sub("", symbol)
                base = symbol.split("+")[0]
                return f"<{symbol}>" if base != name else f"<+{address - start:#x}>"
            return f"+{address - start:#x}" if 0 <= address - start < size_hint else match.group(0)

        body.append(HASH.sub("", ADDRESS.sub(local, instruction)))
    if name is not None:
        found[name] = body
    return found


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("old", type=Path)
    parser.add_argument("new", type=Path)
    parser.add_argument("--match", default=r"indrajala_math_rust::", help="regex on the demangled name")
    parser.add_argument("--all", action="store_true", help="also list the unchanged functions")
    args = parser.parse_args()
    keep = re.compile(args.match)
    old = {k: v for k, v in functions(args.old).items() if keep.search(k)}
    new = {k: v for k, v in functions(args.new).items() if keep.search(k)}
    changed = sorted(k for k in old.keys() & new.keys() if old[k] != new[k])
    same = sorted(k for k in old.keys() & new.keys() if old[k] == new[k])
    for name in changed:
        print(f"changed  {len(old[name]):5d} -> {len(new[name]):5d}  {name}")
    for name in sorted(old.keys() - new.keys()):
        print(f"only old {len(old[name]):5d}           {name}")
    for name in sorted(new.keys() - old.keys()):
        print(f"only new          {len(new[name]):5d}  {name}")
    if args.all:
        for name in same:
            print(f"same     {len(old[name]):5d}           {name}")
    print(
        f"{len(changed)} changed, {len(old.keys() - new.keys())} only old, "
        f"{len(new.keys() - old.keys())} only new, {len(same)} the same"
    )
    sys.exit(1 if changed else 0)


if __name__ == "__main__":
    main()
