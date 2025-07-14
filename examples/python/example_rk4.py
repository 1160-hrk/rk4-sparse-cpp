#!/usr/bin/env python3
"""RK4スパース実装の使用例"""

import sys
import os
import numpy as np
from scipy.sparse import csr_matrix

# プロジェクトルートへのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

from rk4_sparse import rk4_sparse_py, rk4_sparse_cpp

def simple_two_level_example():
    """簡単な二準位系の例"""
    print("=== Simple Two-Level System Example ===")
    
    # 二準位系のハミルトニアン
    H0 = csr_matrix([[0, 0], [0, 1]], dtype=np.complex128)
    
    # 双極子演算子
    mux = csr_matrix([[0, 1], [1, 0]], dtype=np.complex128)
    muy = csr_matrix([[0, 0], [0, 0]], dtype=np.complex128)
    
    # 初期状態（基底状態）
    psi0 = np.array([1, 0], dtype=np.complex128)
    
    # 共鳴パルス
    steps = 100
    t = np.linspace(0, 10, steps)
    Ex = 0.1 * np.sin(t)  # 共鳴周波数
    Ey = np.zeros_like(Ex)
    dt = t[1] - t[0]
    
    print(f"Time steps: {steps}")
    print(f"Time step size: {dt:.3f}")
    print(f"Total time: {t[-1]:.3f}")
    
    # Python実装
    print("\nRunning Python implementation...")
    result_py = rk4_sparse_py(H0, mux, muy, Ex, Ey, psi0, dt*2, True, 1, False)
    
    # C++実装
    print("Running C++ implementation...")
    result_cpp = rk4_sparse_cpp(H0, mux, muy, Ex, Ey, psi0, dt*2, True, 1, False)
    
    # 結果の表示
    print(f"\nFinal state (Python): [{result_py[-1, 0]:.3f}, {result_py[-1, 1]:.3f}]")
    print(f"Final state (C++):    [{result_cpp[-1, 0]:.3f}, {result_cpp[-1, 1]:.3f}]")
    
    # 占有数の計算
    pop_ground_py = np.abs(result_py[:, 0])**2
    pop_excited_py = np.abs(result_py[:, 1])**2
    pop_ground_cpp = np.abs(result_cpp[:, 0])**2
    pop_excited_cpp = np.abs(result_cpp[:, 1])**2
    
    print(f"\nFinal populations:")
    print(f"Python  - Ground: {pop_ground_py[-1]:.6f}, Excited: {pop_excited_py[-1]:.6f}")
    print(f"C++     - Ground: {pop_ground_cpp[-1]:.6f}, Excited: {pop_excited_cpp[-1]:.6f}")
    
    # 実装間の差
    max_diff = np.max(np.abs(result_py - result_cpp))
    print(f"\nMaximum difference between implementations: {max_diff:.2e}")
    
    return True

if __name__ == "__main__":
    print("Example RK4 usage")
    success = simple_two_level_example()
    if success:
        print("\n✅ Example completed successfully!")
    else:
        print("\n❌ Example failed!")
