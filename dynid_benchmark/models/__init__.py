"""Model package init that guarantees registry population."""

from importlib import import_module
from typing import Iterable, Optional

from .base import MODEL_REGISTRY

# Explicit list keeps import order deterministic and avoids accidental duplicates.
_DEFAULT_MODEL_MODULES = (
    "null",
    "sindy_stlsq",
    "edmd",
    "sindy_pi",
    "sindy_implicit",
    "neural_ode_torch",
    "pysindy_adapter",
)

_LOADED = False


def ensure_models_imported(extra_modules: Optional[Iterable[str]] = None):
    """Import all known model modules so their classes register themselves.

    Parameters
    ----------
    extra_modules:
        Iterable of additional module names (relative to this package) to import.
    """
    global _LOADED
    modules = []
    if not _LOADED:
        modules.extend(_DEFAULT_MODEL_MODULES)
    if extra_modules:
        modules.extend(extra_modules)
    for module_name in modules:
        import_module(f"{__name__}.{module_name}")
    if not _LOADED:
        _LOADED = True
    return MODEL_REGISTRY


# Import defaults at module load so callers only need `import dynid_benchmark.models`.
ensure_models_imported()

__all__ = ["MODEL_REGISTRY", "ensure_models_imported"]
