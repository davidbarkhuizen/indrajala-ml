from types import ModuleType
from typing import Any, cast

import pytest

from indrajala_ml.model.array_backend import NUMPY, RUST
from indrajala_ml.model.array_protocols import ArrayBackend


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
