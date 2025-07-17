#!/usr/bin/env python3
"""
SuiteSparse-MKL版のテストスクリプト
"""

import sys
import os

# ビルドディレクトリのパスを追加
build_dir = "build-suitesparse"
if os.path.exists(build_dir):
    sys.path.append(os.path.join(build_dir, "python"))

import numpy as np
import scipy.sparse as sparse
from rk4_sparse import (
    rk4_sparse_eigen, 
    rk4_sparse_suitesparse, 
    benchmark_implementations,
    create_test_matrices, 
    create_test_pulse
)

def test_implementations():
    """両実装の動作確認とベンチマーク"""
    
    print("=== SuiteSparse-MKL版テスト ===")
    
    # テストデータの準備
    n = 100  # 行列サイズ
    steps = 1000  # 時間ステップ数
    
    print(f"行列サイズ: {n}x{n}")
    print(f"時間ステップ数: {steps}")
    
    # テスト行列の生成
    H0, mux, muy = create_test_matrices(n)
    
    # テストパルスの生成
    Ex, Ey = create_test_pulse(steps * 2 + 1)  # 奇数長にする
    
    # 初期状態
    psi0 = np.zeros(n, dtype=np.complex128)
    psi0[0] = 1.0
    
    # パラメータ
    dt = 0.01
    return_traj = True
    stride = 10
    renorm = False
    
    print("\n--- Eigen版の実行 ---")
    try:
        result_eigen = rk4_sparse_eigen(
            H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm
        )
        print(f"Eigen版成功: 結果サイズ = {result_eigen.shape}")
    except Exception as e:
        print(f"Eigen版エラー: {e}")
        return
    
    print("\n--- SuiteSparse-MKL版の実行 ---")
    try:
        result_suitesparse = rk4_sparse_suitesparse(
            H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm
        )
        print(f"SuiteSparse-MKL版成功: 結果サイズ = {result_suitesparse.shape}")
    except Exception as e:
        print(f"SuiteSparse-MKL版エラー: {e}")
        print("SuiteSparse-MKL版が利用できない可能性があります")
        return
    
    # 結果の比較
    print("\n--- 結果の比較 ---")
    diff = np.abs(result_eigen - result_suitesparse).max()
    print(f"最大差分: {diff}")
    
    if diff < 1e-10:
        print("✅ 両実装の結果が一致しました")
    else:
        print("❌ 結果に差異があります")
    
    # ベンチマーク実行
    print("\n--- ベンチマーク実行 ---")
    try:
        results = benchmark_implementations(
            H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm, num_runs=3
        )
        
        print("\nベンチマーク結果:")
        for result in results:
            print(f"  {result.implementation}: {result.total_time:.6f}秒 (Eigen比: {result.speedup_vs_eigen:.3f}x)")
            
    except Exception as e:
        print(f"ベンチマークエラー: {e}")

def test_availability():
    """利用可能な実装の確認"""
    
    print("=== 利用可能な実装の確認 ===")
    
    implementations = {
        "Eigen版": rk4_sparse_eigen,
        "SuiteSparse-MKL版": rk4_sparse_suitesparse,
        "ベンチマーク機能": benchmark_implementations
    }
    
    for name, impl in implementations.items():
        if impl is not None:
            print(f"✅ {name}: 利用可能")
        else:
            print(f"❌ {name}: 利用不可")

if __name__ == "__main__":
    test_availability()
    print()
    test_implementations() 