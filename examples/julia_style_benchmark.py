#!/usr/bin/env python3
"""
Julia風高速実装のベンチマークテスト
=====================================

C++実装がJuliaと同等の性能を達成できているかを検証します。
"""

import numpy as np
import time
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

try:
    from rk4_sparse._rk4_sparse_cpp import (
        rk4_sparse_eigen,           # 従来実装
        rk4_sparse_julia_style,     # Julia風高速実装
        rk4_sparse_csr_optimized,   # CSR最適化実装
    )
    print("C++実装が利用可能です")
except ImportError as e:
    print(f"C++実装が利用できません: {e}")
    rk4_sparse_eigen = None
    rk4_sparse_julia_style = None
    rk4_sparse_csr_optimized = None

def create_test_system(dim=512):
    """テスト用のハミルトニアンと双極子演算子を作成"""
    # 簡単な対角ハミルトニアン
    H0_diag = np.linspace(0, 1, dim)
    H0 = csr_matrix(np.diag(H0_diag), dtype=np.complex128)
    
    # 隣接要素間の結合
    off_diag = np.ones(dim-1) * 0.1
    mux = csr_matrix(np.diag(off_diag, k=1) + np.diag(off_diag, k=-1), dtype=np.complex128)
    
    # y方向は少し異なるパターン
    muy_data = np.ones(dim-2) * 0.05
    muy = csr_matrix(np.diag(muy_data, k=2) + np.diag(muy_data, k=-2), dtype=np.complex128)
    
    return H0, mux, muy

def create_test_fields(steps=1000):
    """テスト用の電場を作成"""
    t = np.linspace(0, 10, 2*steps + 1)
    Ex = np.sin(t) * np.exp(-0.1 * t)
    Ey = np.cos(t) * np.exp(-0.1 * t)
    return Ex, Ey

def benchmark_implementation(func, name, H0, mux, muy, Ex, Ey, psi0, dt, num_runs=5):
    """実装の性能を測定"""
    if func is None:
        return None, name
    
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        result = func(H0, mux, muy, Ex, Ey, psi0, dt, False, 1, False)
        end = time.perf_counter()
        times.append(end - start)
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"{name:25s}: {avg_time:.4f} ± {std_time:.4f} s")
    return avg_time, name

def main():
    """メインベンチマーク"""
    print("Julia風高速実装のベンチマークテスト")
    print("=" * 50)
    
    # テストサイズの設定
    test_dims = [256, 512, 1024, 2048]
    test_steps = 500
    
    results = {}
    
    for dim in test_dims:
        print(f"\n次元数: {dim}")
        print("-" * 30)
        
        # テストシステムの作成
        H0, mux, muy = create_test_system(dim)
        Ex, Ey = create_test_fields(test_steps)
        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0
        dt = 0.01
        
        dim_results = {}
        
        # 各実装のベンチマーク
        implementations = [
            (rk4_sparse_eigen, "従来Eigen実装"),
            (rk4_sparse_julia_style, "Julia風高速実装"),
            (rk4_sparse_csr_optimized, "CSR最適化実装"),
        ]
        
        baseline_time = None
        for func, name in implementations:
            avg_time, impl_name = benchmark_implementation(
                func, name, H0, mux, muy, Ex, Ey, psi0, dt
            )
            if avg_time is not None:
                dim_results[impl_name] = avg_time
                if baseline_time is None:
                    baseline_time = avg_time
        
        # 高速化比を計算
        if baseline_time:
            print("\n高速化比:")
            for name, time_val in dim_results.items():
                speedup = baseline_time / time_val
                print(f"  {name:25s}: {speedup:.2f}x")
        
        results[dim] = dim_results
    
    # 結果の可視化
    plot_results(results)

def plot_results(results):
    """結果をプロット"""
    if not results:
        print("プロット用のデータがありません")
        return
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    dims = list(results.keys())
    implementations = set()
    for dim_results in results.values():
        implementations.update(dim_results.keys())
    implementations = list(implementations)
    
    # 実行時間のプロット
    for impl in implementations:
        times = []
        valid_dims = []
        for dim in dims:
            if impl in results[dim]:
                times.append(results[dim][impl])
                valid_dims.append(dim)
        
        if times:
            ax1.plot(valid_dims, times, 'o-', label=impl, linewidth=2, markersize=6)
    
    ax1.set_xlabel('問題次元数')
    ax1.set_ylabel('実行時間 (秒)')
    ax1.set_title('実行時間比較')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # 高速化比のプロット（従来実装を基準）
    baseline_impl = "従来Eigen実装"
    if baseline_impl in implementations:
        for impl in implementations:
            if impl == baseline_impl:
                continue
            speedups = []
            valid_dims = []
            for dim in dims:
                if impl in results[dim] and baseline_impl in results[dim]:
                    speedup = results[dim][baseline_impl] / results[dim][impl]
                    speedups.append(speedup)
                    valid_dims.append(dim)
            
            if speedups:
                ax2.plot(valid_dims, speedups, 'o-', label=impl, linewidth=2, markersize=6)
    
    ax2.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='基準線')
    ax2.set_xlabel('問題次元数')
    ax2.set_ylabel('高速化比')
    ax2.set_title('従来実装に対する高速化比')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('examples/figures/julia_style_benchmark.png', dpi=150, bbox_inches='tight')
    plt.show()
    
    print(f"\nベンチマーク結果を examples/figures/julia_style_benchmark.png に保存しました")

if __name__ == "__main__":
    main() 