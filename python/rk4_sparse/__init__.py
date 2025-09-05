from __future__ import annotations

from typing import Any, Optional

from .rk4_py import rk4_sparse_py, rk4_numba_py
from .utils import create_test_matrices, create_test_pulse

# ──────────────────────────────────────────────────────────────
# C++ バックエンドは wheel に含まれていない可能性もあるので
# ImportError を握りつぶして None をエクスポートする。
# ──────────────────────────────────────────────────────────────

# 変数の型注釈を追加
rk4_sparse_eigen: Optional[Any] = None
rk4_sparse_eigen_cached: Optional[Any] = None
rk4_sparse_eigen_direct_csr: Optional[Any] = None
rk4_sparse_suitesparse: Optional[Any] = None
benchmark_implementations: Optional[Any] = None
rk4_sparse_suitesparse_mkl: Optional[Any] = None
fast: Optional[Any] = None
OPENBLAS_SUITESPARSE_AVAILABLE: bool = False
SUITESPARSE_MKL_AVAILABLE: bool = False

import os

# ヘルプ重視: Pythonラッパー（api.py）。速度重視: C++直（fast.py）。
try:
    from . import api as _api
    rk4_sparse_eigen = _api.rk4_sparse_eigen
    rk4_sparse_eigen_cached = _api.rk4_sparse_eigen_cached
    rk4_sparse_eigen_direct_csr = _api.rk4_sparse_eigen_direct_csr
    try:
        rk4_sparse_suitesparse = _api.rk4_sparse_suitesparse
    except Exception:
        rk4_sparse_suitesparse = None
    try:
        benchmark_implementations = _api.benchmark_implementations
    except Exception:
        benchmark_implementations = None
    OPENBLAS_SUITESPARSE_AVAILABLE = rk4_sparse_suitesparse is not None
except Exception as e:
    # Fallback: C++が無い環境でもパッケージは使える（Python実装のみ）
    rk4_sparse_eigen = None
    rk4_sparse_eigen_cached = None
    rk4_sparse_eigen_direct_csr = None
    rk4_sparse_suitesparse = None
    benchmark_implementations = None
    OPENBLAS_SUITESPARSE_AVAILABLE = False
    print(f"Warning: C++ backends not available: {e}")

# SuiteSparse-MKL実装（x86_64のみ）
try:
    from ._rk4_sparse_cpp import rk4_sparse_suitesparse_mkl as _mkl_func
    rk4_sparse_suitesparse_mkl = _mkl_func
    SUITESPARSE_MKL_AVAILABLE = True
except Exception:
    SUITESPARSE_MKL_AVAILABLE = False

# 速度優先の明示的パスも提供（rk4_sparse.fast）
try:
    from . import fast  # noqa: F401 - re-exported as a submodule
except Exception as _e:
    fast = None

# 環境変数でトップレベルを速度優先に切替可能（任意）。
# RK4_SPARSE_FAST=1 なら、トップレベル関数を fast 側に差し替える。
if os.environ.get("RK4_SPARSE_FAST", "0") in {"1", "true", "TRUE"} and fast is not None:
    try:
        # fastモジュールが利用可能な場合のみ属性にアクセス
        if hasattr(fast, 'rk4_sparse_eigen'):
            rk4_sparse_eigen = fast.rk4_sparse_eigen
        if hasattr(fast, 'rk4_sparse_eigen_cached'):
            rk4_sparse_eigen_cached = fast.rk4_sparse_eigen_cached
        if hasattr(fast, 'rk4_sparse_eigen_direct_csr'):
            rk4_sparse_eigen_direct_csr = fast.rk4_sparse_eigen_direct_csr
        if hasattr(fast, 'rk4_sparse_suitesparse'):
            rk4_sparse_suitesparse = fast.rk4_sparse_suitesparse
        if hasattr(fast, 'benchmark_implementations'):
            benchmark_implementations = fast.benchmark_implementations
    except Exception:
        # If fast import fails, keep API path
        pass

__all__ = [
    "rk4_sparse_py",
    "rk4_numba_py",
    "rk4_sparse_eigen",
    "rk4_sparse_eigen_cached",
    "rk4_sparse_eigen_direct_csr",
    "rk4_sparse_suitesparse",
    "benchmark_implementations",
    "create_test_matrices",
    "create_test_pulse",
    "OPENBLAS_SUITESPARSE_AVAILABLE",
    "SUITESPARSE_MKL_AVAILABLE",
]
