"""
rk4_sparse ― sparse 行列版 RK4 伝搬器
------------------------------------
* ``rk4_sparse_py``     : 100 % Python 実装
* ``rk4_numba_py``      : Numba JIT 実装（実験的）
* ``rk4_sparse_cpp``    : C++/Eigen + pybind11 実装（ビルド済みなら自動ロード）
* ``create_test_*``     : テスト用ユーティリティ
"""

from __future__ import annotations

from .rk4_py import rk4_sparse_py, rk4_numba_py
from .utils import create_test_matrices, create_test_pulse

# ──────────────────────────────────────────────────────────────
# C++ バックエンドは wheel に含まれていない可能性もあるので
# ImportError を握りつぶして None をエクスポートする。
# ──────────────────────────────────────────────────────────────
try:
    from ._rk4_sparse_cpp import rk4_sparse_cpp  # バイナリ拡張を直接 import
except ImportError:                              # ビルド無しでもパッケージは使える
    rk4_sparse_cpp = None                        # type: ignore[assignment]

__all__ = [
    "rk4_sparse_py",
    "rk4_numba_py",
    "rk4_sparse_cpp",
    "create_test_matrices",
    "create_test_pulse",
]
