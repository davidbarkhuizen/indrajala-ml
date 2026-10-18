from types import ModuleType
from typing import Any, cast

import pytest

from indrajala_ml.model.layers.array.array_backend import NUMPY, RUST
from indrajala_ml.model.protocols.array_protocols import ArrayBackend


def _backend_id(backend: ArrayBackend[Any]) -> str:
    return backend.name


@pytest.fixture(params=[NUMPY, RUST], ids=_backend_id)
def backend(request: pytest.FixtureRequest) -> ArrayBackend[Any]:
    # a test taking `backend` runs once per array backend, as test_x[numpy] and test_x[rust];
    # backend.owned wraps nested lists in that backend's array type
    return request.param


@pytest.fixture(params=["numpy", "rust", "python"])
def implementation(request: pytest.FixtureRequest) -> str:
    # a test taking `implementation` runs once per array backend and once on the pure-Python
    # networks, as test_x[numpy], test_x[rust] and test_x[python] (helpers.randomized builds them)
    return request.param


@pytest.fixture
def layer_cls(request: pytest.FixtureRequest, backend: ArrayBackend[Any]) -> Any:
    # the test module's LAYER_CLS ({backend name: layer class}) class for backend
    module = cast("ModuleType", request.module)  # pyright: ignore[reportUnknownMemberType]  (untyped in pytest)
    layer_classes = cast("dict[str, Any]", module.LAYER_CLS)
    return layer_classes[backend.name]


def pytest_addoption(parser: pytest.Parser) -> None:
    parser.addoption(
        "--shard",
        metavar="I/N",
        help="run only shard I of N (0-based): every Nth test from the Ith in collection order, so CI's "
        "parallel jobs split tests/ between them, the slow parametrized tests spread across all N",
    )


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    shard = cast("str | None", config.getoption("--shard"))
    if shard is None:
        return
    index, count = (int(x) for x in shard.split("/"))
    assert 0 <= index < count, f"--shard I/N needs 0 <= I < N; got {shard}"
    kept = [item for i, item in enumerate(items) if i % count == index]
    config.hook.pytest_deselected(items=[item for i, item in enumerate(items) if i % count != index])
    items[:] = kept
