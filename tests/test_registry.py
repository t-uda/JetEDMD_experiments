import pytest

from dynid_benchmark.models import MODEL_REGISTRY, ensure_models_imported
from dynid_benchmark.models.base import Model, register_model


def test_registry_contains_expected_models():
    ensure_models_imported()
    expected = {"sindy_stlsq", "sindy_pi", "sindy_implicit", "edmd", "mean_dfdx", "neural_ode", "zero"}
    assert expected.issubset(set(MODEL_REGISTRY.keys()))


def test_ensure_models_imported_idempotent():
    ensure_models_imported()
    size_before = len(MODEL_REGISTRY)
    ensure_models_imported()
    assert len(MODEL_REGISTRY) == size_before


def test_register_model_rejects_duplicate_names():
    class Duplicate(Model):
        name = "sindy_stlsq"

        def fit(self, t, y, u=None):
            raise NotImplementedError

        def predict_derivative(self, t, x, u=None):
            raise NotImplementedError

    with pytest.raises(ValueError):
        register_model(Duplicate)
