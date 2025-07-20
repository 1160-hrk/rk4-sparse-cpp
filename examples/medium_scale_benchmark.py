#!/usr/bin/env python3
"""
中規模問題特化ベンチマーク
新しいSIMD最適化実装（rk4_sparse_medium_scale_optimized）の性能測定
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'build'))

import numpy as np
import scipy.sparse as sp
import time
import _rk4_sparse_cpp

def create_test_problem(dim, sparsity=0.1):
    """テスト問題の生成"""
    nnz = int(dim * dim * sparsity)
    
    # H0: ランダムスパース行列
    H0_data = np.random.randn(nnz) + 1j * np.random.randn(nnz)
    H0_indices = np.random.randint(0, dim, nnz)
    H0_indptr = np.sort(np.random.randint(0, nnz, dim + 1))
    H0_indptr[0] = 0
    H0_indptr[-1] = nnz
    H0 = sp.csr_matrix((H0_data, H0_indices, H0_indptr), shape=(dim, dim))
    H0 = H0 + H0.conj().T  # エルミート行列にする
    
    # mux, muy: より疎な行列
    mux_nnz = nnz // 4
    mux_data = np.random.randn(mux_nnz) + 1j * np.random.randn(mux_nnz) 
    mux_indices = np.random.randint(0, dim, mux_nnz)
    mux_indptr = np.sort(np.random.randint(0, mux_nnz, dim + 1))
    mux_indptr[0] = 0
    mux_indptr[-1] = mux_nnz
    mux = sp.csr_matrix((mux_data, mux_indices, mux_indptr), shape=(dim, dim))
    mux = mux + mux.conj().T
    
    muy = mux.copy()  # 簡単のため同じ構造を使用
    
    # 電場とパルス
    steps = 100
    t_total = 2 * steps + 1
    Ex = np.random.randn(t_total) * 0.1
    Ey = np.random.randn(t_total) * 0.1
    
    # 初期状態
    psi0 = np.random.randn(dim) + 1j * np.random.randn(dim)
    psi0 = psi0 / np.linalg.norm(psi0)
    
    return H0, mux, muy, Ex, Ey, psi0

def benchmark_implementations(dim_list, num_runs=3):
    """複数実装の性能比較"""
    results = {}
    
    for dim in dim_list:
        print(f"\n次元数: {dim}")
        print("=" * 50)
        
        # テスト問題生成
        H0, mux, muy, Ex, Ey, psi0 = create_test_problem(dim)
        dt = 0.01
        
        # 各実装のテスト
        implementations = [
            ("Eigen標準", "rk4_sparse_eigen"),
            ("Julia風", "rk4_sparse_julia_style"), 
            ("CSR最適化", "rk4_sparse_csr_optimized"),
            ("中規模SIMD最適化", "_rk4_sparse_medium_scale_optimized")
        ]
        
        dim_results = {}
        
        for name, func_name in implementations:
            if hasattr(_rk4_sparse_cpp, func_name):
                func = getattr(_rk4_sparse_cpp, func_name)
                times = []
                
                for run in range(num_runs):
                    try:
                        start_time = time.perf_counter()
                        result = func(H0, mux, muy, Ex, Ey, psi0, dt, False, 1, False)
                        end_time = time.perf_counter()
                        times.append(end_time - start_time)
                    except Exception as e:
                        print(f"  {name}: エラー - {e}")
                        times = [float('inf')]
                        break
                
                if times[0] != float('inf'):
                    avg_time = np.mean(times)
                    std_time = np.std(times)
                    print(f"  {name}: {avg_time:.4f}±{std_time:.4f}s")
                    dim_results[name] = avg_time
                else:
                    dim_results[name] = float('inf')
            else:
                print(f"  {name}: 関数が見つかりません")
                dim_results[name] = float('inf')
        
        results[dim] = dim_results
        
        # 改善率の計算
        if "Eigen標準" in dim_results and "中規模SIMD最適化" in dim_results:
            eigen_time = dim_results["Eigen標準"]
            simd_time = dim_results["中規模SIMD最適化"]
            if eigen_time < float('inf') and simd_time < float('inf'):
                improvement = eigen_time / simd_time
                print(f"  → 中規模SIMD最適化の改善率: {improvement:.2f}x")
        
        # Julia風との比較
        if "Julia風" in dim_results and "中規模SIMD最適化" in dim_results:
            julia_time = dim_results["Julia風"] 
            simd_time = dim_results["中規模SIMD最適化"]
            if julia_time < float('inf') and simd_time < float('inf'):
                improvement = julia_time / simd_time
                print(f"  → Julia風との比較: {improvement:.2f}x")
    
    return results

def main():
    print("中規模問題特化ベンチマーク")
    print("=" * 60)
    print("新しいSIMD最適化実装の性能測定")
    
    # 中規模問題のサイズリスト
    dim_list = [100, 200, 300, 500, 700, 1000]
    
    results = benchmark_implementations(dim_list)
    
    print("\n\n最終結果サマリー")
    print("=" * 60)
    
    # 結果テーブル
    print(f"{'次元数':<8} {'Eigen':<10} {'Julia風':<10} {'CSR':<10} {'中規模SIMD':<12} {'改善率':<8}")
    print("-" * 65)
    
    for dim in sorted(results.keys()):
        res = results[dim]
        eigen_time = res.get("Eigen標準", float('inf'))
        julia_time = res.get("Julia風", float('inf'))
        csr_time = res.get("CSR最適化", float('inf'))
        simd_time = res.get("中規模SIMD最適化", float('inf'))
        
        if eigen_time < float('inf') and simd_time < float('inf'):
            improvement = eigen_time / simd_time
        else:
            improvement = 0
        
        print(f"{dim:<8} {eigen_time:<10.4f} {julia_time:<10.4f} {csr_time:<10.4f} {simd_time:<12.4f} {improvement:<8.2f}x")
    
    print("\n分析:")
    print("- 中規模SIMD最適化は100-1000次元の範囲で最適化されています")
    print("- この範囲外では自動的にJulia風実装にフォールバックします")
    print("- 目標はJulia実装に近い性能の達成です")

if __name__ == "__main__":
    main() 