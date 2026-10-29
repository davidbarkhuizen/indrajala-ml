"""
scripts/measurement/kernel_code_diff.py: a function's code is compared without where the linker put
it, so a moved function or constant reads as the same, and a changed instruction as changed.
"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent / "scripts" / "measurement"))
import kernel_code_diff  # scripts/ isn't a package

X86 = """
{start:016x} <crate::kernel>:
 {a:x}:\tvmovupd (%rdi),%ymm0
 {b:x}:\tvfmadd231pd {c:#x}(%rip),%ymm0,%ymm1        # {d:x} <anon.{h}.7+0x40>
 {e:x}:\tjne    {a:x} <crate::kernel+0x0>
 {f:x}:\tcall   {g:x} <crate::helper>
"""

AARCH64 = """
{start:016x} <crate::kernel>:
 {a:x}:\tadrp\tx8, {p:x} <anon+0x10>
 {b:x}:\tldr\tq0, [x8, #{o}]
 {c:x}:\tldr\tq1, [x9, #16]
 {d:x}:\tmov\tx8, x0
 {e:x}:\tldr\tq2, [x8, #32]
"""


def _x86(start: int, data: int, op: str = "vfmadd231pd") -> dict[str, list[str]]:
    text = X86.format(
        start=start, a=start, b=start + 4, c=data, d=data + 0x40, e=start + 13, f=start + 15,
        g=start + 0x900, h=f"{data:032x}",
    ).replace("vfmadd231pd", op)  # fmt: skip
    return kernel_code_diff.parse(text)


def _aarch64(start: int, page: int, offset: int, own: int = 16) -> dict[str, list[str]]:
    text = AARCH64.format(start=start, a=start, b=start + 4, c=start + 8, d=start + 12, e=start + 16, p=page, o=offset)
    return kernel_code_diff.parse(text.replace("[x9, #16]", f"[x9, #{own}]"))


def test_a_moved_function_and_constant_read_as_the_same_code():
    assert _x86(0x1000, 0x5000) == _x86(0x2340, 0x9100)
    assert _aarch64(0x1000, 0x40000, 1328) == _aarch64(0x8800, 0x52000, 1392)


def test_a_changed_instruction_or_offset_reads_as_changed():
    assert _x86(0x1000, 0x5000) != _x86(0x1000, 0x5000, op="vfmadd213pd")
    # x9 holds no page, so its offset is the code's own; x8 stops holding one once mov writes it
    assert _aarch64(0x1000, 0x40000, 1328) != _aarch64(0x1000, 0x40000, 1328, own=24)
    assert kernel_code_diff.parse(AARCH64.format(start=0, a=0, b=4, c=8, d=12, e=16, p=0x40000, o=8))["crate::kernel"][
        -1
    ].endswith("[x8, #32]")
