"""tests/conftest.py's --shard I/N: CI's parallel jobs split tests/ between them, every test in one shard."""

from typing import Any

import conftest
import pytest


class _Hook:
    def __init__(self) -> None:
        self.deselected: list[Any] = []

    def pytest_deselected(self, items: list[Any]) -> None:
        self.deselected += items


class _Config:
    def __init__(self, shard: str | None) -> None:
        self.shard, self.hook = shard, _Hook()

    def getoption(self, name: str) -> str | None:
        assert name == "--shard"
        return self.shard


def _shard(shard: str | None, items: list[Any]) -> tuple[list[Any], list[Any]]:
    config = _Config(shard)
    kept = list(items)
    conftest.pytest_collection_modifyitems(config, kept)  # pyright: ignore[reportArgumentType]
    return kept, config.hook.deselected


def test_the_shards_partition_the_tests_every_nth_from_the_ith():
    items = list(range(23))
    shards = [_shard(f"{i}/4", items) for i in range(4)]
    assert [kept for kept, _ in shards][1] == [1, 5, 9, 13, 17, 21]
    assert sorted(x for kept, _ in shards for x in kept) == items  # each test in exactly one shard
    assert all(sorted(kept + deselected) == items for kept, deselected in shards)


def test_no_shard_keeps_every_test_and_a_bad_one_is_refused():
    assert _shard(None, [1, 2, 3]) == ([1, 2, 3], [])
    with pytest.raises(AssertionError):
        _shard("4/4", [1, 2, 3])
