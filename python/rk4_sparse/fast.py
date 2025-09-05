"""Fast path: re-export raw C++ functions for minimal overhead.

Import from here when you want to avoid even a thin Python wrapper.
The functions have argument names and brief docstrings from pybind11.
"""

from __future__ import annotations

try:
    from . import _rk4_sparse_cpp as _ext  # type: ignore
except Exception as _e:  # pragma: no cover
    raise ImportError(
        "C++ extension is not available. Build the module to use rk4_sparse.fast."
    ) from _e

# Re-export the C++ module's public API without star-imports (flake8 clean)
__doc__ = _ext.__doc__
__all__ = [name for name in dir(_ext) if not name.startswith("_")]
for name in __all__:
    globals()[name] = getattr(_ext, name)
del name, _ext
