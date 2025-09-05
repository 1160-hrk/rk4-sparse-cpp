"""
rk4_sparse ― sparse 行列版 RK4 伝搬器
------------------------------------
* ``rk4_sparse_py``           : 100 % Python 実装
* ``rk4_numba_py``            : Numba JIT 実装（実験的）
* ``rk4_sparse_eigen``        : C++/Eigen + pybind11 実装
* ``rk4_sparse_eigen_cached`` : キャッシュ化された高速C++/Eigen実装
* ``rk4_sparse_eigen_direct_csr`` : 最適化されたC++/Eigen実装（直接CSR形式）
* ``rk4_sparse_suitesparse``  : C++/SuiteSparse-MKL + pybind11 実装
* ``benchmark_implementations``: 実装間の速度比較
* ``create_test_*``           : テスト用ユーティリティ
"""

from __future__ import annotations

from typing import Any, Optional

from .rk4_py import rk4_sparse_py, rk4_numba_py
from .utils import create_test_matrices, create_test_pulse

# ──────────────────────────────────────────────────────────────
# C++ バックエンドは wheel に含まれていない可能性もあるので
# ImportError を握りつぶして None をエクスポートする。
# ──────────────────────────────────────────────────────────────

rk4_sparse_eigen: Optional[Any] = None
rk4_sparse_eigen_cached: Optional[Any] = None
rk4_sparse_eigen_direct_csr: Optional[Any] = None
rk4_sparse_suitesparse: Optional[Any] = None
benchmark_implementations: Optional[Any] = None
rk4_sparse_suitesparse_mkl: Optional[Any] = None
OPENBLAS_SUITESPARSE_AVAILABLE: bool = False
SUITESPARSE_MKL_AVAILABLE: bool = False

# Eigen実装
try:
    from ._rk4_sparse_cpp import rk4_sparse_eigen  # バイナリ拡張を直接 import
    print("Info: Eigen version available")
except ImportError as e:                              # ビルド無しでもパッケージは使える
    rk4_sparse_eigen = None
    print(f"Warning: Eigen version not available: {e}")

# キャッシュ化された高速Eigen実装
try:
    from ._rk4_sparse_cpp import rk4_sparse_eigen_cached
    print("Info: Cached Eigen version available")
except ImportError as e:
    rk4_sparse_eigen_cached = None
    print(f"Warning: Cached Eigen version not available: {e}")

# 最適化されたEigen実装（直接CSR形式）
try:
    from ._rk4_sparse_cpp import rk4_sparse_eigen_direct_csr
    print("Info: Optimized Eigen version (direct CSR) available")
except ImportError as e:
    rk4_sparse_eigen_direct_csr = None
    print(f"Warning: Optimized Eigen version not available: {e}")

# OpenBLAS + SuiteSparse実装
try:
    from ._rk4_sparse_cpp import (
        rk4_sparse_suitesparse,
        benchmark_implementations
    )
    OPENBLAS_SUITESPARSE_AVAILABLE = True
    print("Info: OpenBLAS + SuiteSparse version available")
except ImportError as e:
    rk4_sparse_suitesparse = None
    benchmark_implementations = None
    OPENBLAS_SUITESPARSE_AVAILABLE = False
    print(f"Warning: OpenBLAS + SuiteSparse version not available: {e}")

# SuiteSparse-MKL実装（x86_64のみ）
try:
    from ._rk4_sparse_cpp import rk4_sparse_suitesparse_mkl
    SUITESPARSE_MKL_AVAILABLE = True
    print("Info: SuiteSparse-MKL version available")
except ImportError as e:
    rk4_sparse_suitesparse_mkl = None
    SUITESPARSE_MKL_AVAILABLE = False
    print(f"Warning: SuiteSparse-MKL version not available: {e}")

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
