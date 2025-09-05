"""Fast path: re-export raw C++ functions for minimal overhead.

Import from here when you want to avoid even a thin Python wrapper.
The functions have argument names and brief docstrings from pybind11.
"""

from __future__ import annotations

try:
    from ._rk4_sparse_cpp import *  # type: ignore
    from ._rk4_sparse_cpp import __doc__ as __doc__  # re-export module doc
except Exception as _e:  # pragma: no cover
    raise ImportError(
        "C++ extension is not available. Build the module to use rk4_sparse.fast."
    ) from _e

# Avoid leaking names beyond the C++ exports
__all__ = [name for name in globals().keys() if not name.startswith("_")]

