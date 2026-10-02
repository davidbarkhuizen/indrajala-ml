"""
tests/helpers.py's module walk: the class walks over indrajala_ml.model (all_subclasses, the saved
model fixtures) find every class, wherever docs/source-layout-workplan.md moves its module.
"""

import importlib
import pkgutil
import sys
from pathlib import Path
from types import ModuleType

import pytest

import indrajala_ml.model
from tests.helpers import model_modules, package_modules

# the classes indrajala_ml.model defined while it was one flat directory (source layout, stage 1).
# A walk that missed a subpackage would leave the tests that walk it passing on fewer classes.
FLAT_LAYOUT_CLASS_COUNT = 250


def _class_names(modules: list[ModuleType]) -> set[str]:
    return {
        name
        for module in modules
        for name, value in vars(module).items()
        if isinstance(value, type) and value.__module__ == module.__name__
    }


def test_model_modules_finds_every_class_the_flat_walk_finds() -> None:
    flat = [
        importlib.import_module(f"indrajala_ml.model.{info.name}")
        for info in pkgutil.iter_modules(indrajala_ml.model.__path__)
    ]
    found = _class_names(model_modules())
    assert _class_names(flat) <= found
    assert len(found) >= FLAT_LAYOUT_CLASS_COUNT


def test_package_modules_recurses_into_namespace_packages(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    root = tmp_path / "walk_probe"
    for module in ("a.py", "sub/b.py", "sub/deeper/c.py"):
        (root / module).parent.mkdir(parents=True, exist_ok=True)
        (root / module).write_text("")
    (root / "sub/notes.txt").write_text("")
    monkeypatch.setattr(sys, "path", [str(tmp_path), *sys.path])
    names = ["walk_probe.a", "walk_probe.sub.b", "walk_probe.sub.deeper.c"]
    try:
        package = importlib.import_module("walk_probe")
        assert [module.__name__ for module in package_modules(package)] == names
        assert [info.name for info in pkgutil.iter_modules(package.__path__)] == ["a"]  # what the old walks saw
    finally:
        for name in ["walk_probe", "walk_probe.sub", "walk_probe.sub.deeper", *names]:
            sys.modules.pop(name, None)
