import sys
import types

import pytest

from dynid_benchmark.models import MODEL_REGISTRY, ensure_models_imported
from dynid_benchmark.models.base import Model, register_model


@pytest.mark.parametrize(
    "expected_name",
    [
        "sindy_stlsq",
        "sindy_pi",
        "sindy_implicit",
        "edmd",
        "mean_dfdx",
        "neural_ode",
        "zero",
    ],
)
def test_registry_contains_expected_models(expected_name):
    ensure_models_imported()
    assert expected_name in MODEL_REGISTRY


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


def test_register_model_requires_name():
    class Nameless(Model):
        name = ""

        def fit(self, t, y, u=None):
            raise NotImplementedError

        def predict_derivative(self, t, x, u=None):
            raise NotImplementedError

    with pytest.raises(ValueError):
        register_model(Nameless)


def test_ensure_models_imported_accepts_extra_modules():
    module_name = "dynid_benchmark.models._demo_unit"
    module = types.ModuleType(module_name)
    exec(
        """
from dynid_benchmark.models.base import Model, register_model

@register_model
class Demo(Model):
    name = "demo_unit"
    def fit(self, t, y, u=None):
        pass
    def predict_derivative(self, t, x, u=None):
        return x
""",
        module.__dict__,
    )
    sys.modules[module_name] = module
    try:
        ensure_models_imported(extra_modules=["_demo_unit"])
        assert "demo_unit" in MODEL_REGISTRY
    finally:
        MODEL_REGISTRY.pop("demo_unit", None)
        sys.modules.pop(module_name, None)
