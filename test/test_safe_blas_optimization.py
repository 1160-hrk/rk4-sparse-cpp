#!/usr/bin/env python3
"""
安全なBLAS最適化版RK4実装のテストスクリプト
"""

import numpy as np
import scipy.sparse as sp
import time
from rk4_sparse._rk4_sparse_cpp import (
    rk4_sparse_eigen,
    rk4_sparse_blas_optimized_safe,
    test_blas_sparse_multiply
)

def create_test_matrices(dim):
    """テスト用のスパース行列を作成"""
    print(f"  {dim}次元のテスト行列を作成中...")
    
    # 対角成分
    diag_indices = np.arange(dim)
    diag_data = np.random.random(dim) + 1j * np.random.random(dim)
    
    # 非対角成分（スパース性を保つ）
    nnz_per_row = max(1, dim // 100)  # 1%の密度
    off_diag_data = []
    off_diag_indices = []
    off_diag_indptr = [0]
    
    for i in range(dim):
        # 各行の非ゼロ要素数を決定
        num_nnz = min(nnz_per_row, dim - 1)
        if num_nnz > 0:
            # ランダムな列を選択（対角成分を除く）
            cols = np.random.choice([j for j in range(dim) if j != i], 
                                  size=num_nnz, replace=False)
            data = np.random.random(num_nnz) + 1j * np.random.random(num_nnz)
            
            off_diag_data.extend(data)
            off_diag_indices.extend(cols)
            off_diag_indptr.append(off_diag_indptr[-1] + num_nnz)
        else:
            off_diag_indptr.append(off_diag_indptr[-1])
    
    # H0行列の作成
    H0_data = np.concatenate([diag_data, off_diag_data])
    H0_indices = np.concatenate([diag_indices, off_diag_indices])
    H0_indptr = np.arange(dim + 1)
    H0 = sp.csr_matrix((H0_data, H0_indices, H0_indptr), shape=(dim, dim))
    
    # mux, muy行列の作成（H0と同じパターン、異なる値）
    mux_data = np.random.random(len(H0_data)) + 1j * np.random.random(len(H0_data))
    muy_data = np.random.random(len(H0_data)) + 1j * np.random.random(len(H0_data))
    
    mux = sp.csr_matrix((mux_data, H0_indices, H0_indptr), shape=(dim, dim))
    muy = sp.csr_matrix((muy_data, H0_indices, H0_indptr), shape=(dim, dim))
    
    return H0, mux, muy

def test_safe_blas_optimization():
    """安全なBLAS最適化版のテスト"""
    print("=== 安全なBLAS最適化版のテスト ===")
    
    # テストパラメータ
    dimensions = [64, 128, 256, 512, 1024]
    num_steps = 50
    dt = 0.01
    
    results = {}
    
    for dim in dimensions:
        print(f"\n--- {dim}次元でのテスト ---")
        
        # テスト行列の作成
        H0, mux, muy = create_test_matrices(dim)
        
        # 電場と初期状態
        Ex = np.random.random(num_steps) * 0.1
        Ey = np.random.random(num_steps) * 0.1
        psi0 = np.random.random(dim) + 1j * np.random.random(dim)
        psi0 = psi0 / np.linalg.norm(psi0)
        
                # 標準版（Eigen）のベンチマーク
        print("  標準版（Eigen）を実行中...")
        start_time = time.time()
        result_eigen = None
        try:
            result_eigen = rk4_sparse_eigen(
                H0, mux, muy, Ex, Ey, psi0, dt, 
                return_traj=False, stride=1, renorm=False
            )
            eigen_time = time.time() - start_time
            print(f"    実行時間: {eigen_time:.4f}秒")
        except Exception as e:
            print(f"    エラー: {e}")
            eigen_time = float('inf')
        
        # 安全なBLAS最適化版のベンチマーク
        print("  安全なBLAS最適化版を実行中...")
        start_time = time.time()
        result_safe = None
        try:
            result_safe = rk4_sparse_blas_optimized_safe(
                H0, mux, muy, Ex, Ey, psi0, dt, 
                return_traj=False, stride=1, renorm=False
            )
            safe_time = time.time() - start_time
            print(f"    実行時間: {safe_time:.4f}秒")
            
            # 結果の比較
            if eigen_time != float('inf') and result_eigen is not None and result_safe is not None:
                diff = np.linalg.norm(result_eigen - result_safe)
                print(f"    結果の差: {diff:.2e}")
                if diff < 1e-10:
                    print("    ✓ 結果が一致しました")
                else:
                    print("    ✗ 結果が一致しません")
                
                speedup = eigen_time / safe_time
                print(f"    速度向上: {speedup:.2f}x")
        except Exception as e:
            print(f"    エラー: {e}")
            safe_time = float('inf')
        
        results[dim] = {
            'eigen_time': eigen_time,
            'safe_time': safe_time,
            'speedup': eigen_time / safe_time if safe_time != float('inf') else 0
        }
    
    # 結果の要約
    print("\n=== 結果要約 ===")
    for dim, result in results.items():
        if result['eigen_time'] != float('inf') and result['safe_time'] != float('inf'):
            print(f"{dim}次元: {result['speedup']:.2f}x 高速化")
        else:
            print(f"{dim}次元: エラーが発生")

def test_blas_sparse_multiply_safety():
    """BLASスパース行列-ベクトル積の安全性テスト"""
    print("\n=== BLASスパース行列-ベクトル積の安全性テスト ===")
    
    # 小さなテストケース
    dim = 8
    H0, _, _ = create_test_matrices(dim)
    psi0 = np.random.random(dim) + 1j * np.random.random(dim)
    
    print("  BLAS最適化版のスパース行列-ベクトル積をテスト中...")
    try:
        result = test_blas_sparse_multiply(H0, psi0)
        expected = -1j * H0.dot(psi0)
        
        diff = np.linalg.norm(result - expected)
        print(f"    結果の差: {diff:.2e}")
        
        if diff < 1e-10:
            print("    ✓ BLASスパース行列-ベクトル積が正常に動作")
        else:
            print("    ✗ BLASスパース行列-ベクトル積に問題があります")
            
    except Exception as e:
        print(f"    ✗ エラーが発生: {e}")

if __name__ == "__main__":
    test_safe_blas_optimization()
    test_blas_sparse_multiply_safety() 