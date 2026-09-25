import pytest

from indrajala_ml.model.array_backend import NUMPY, RUST


@pytest.fixture(params=[NUMPY, RUST], ids=lambda backend: backend.name)
def backend(request):
    # a test taking `backend` runs once per array backend, as test_x[numpy] and test_x[rust];
    # backend.owned wraps nested lists in that backend's array type
    return request.param
