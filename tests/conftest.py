from typing import Any

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
