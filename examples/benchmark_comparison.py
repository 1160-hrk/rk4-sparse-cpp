import sys
import os
import time
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../build'))

from python.rk4_sparse_py import rk4_cpu_sparse as rk4_cpu_sparse_py
import _excitation_rk4_sparse as rk4_cpu_sparse_cpp

def create_test_system(dim):
    """指定したサイズのテスト系を生成"""
    # 対角要素のみを持つH0（スパース）
    H0_data = np.arange(dim, dtype=np.complex128)
    H0 = csr_matrix((H0_data, (np.arange(dim), np.arange(dim))), shape=(dim, dim))
    
    # 隣接要素のみを持つmux（スパース）
    mux_data = np.ones(dim-1, dtype=np.complex128)
    mux = csr_matrix((mux_data, (np.arange(dim-1), np.arange(1, dim))), shape=(dim, dim))
    mux = mux + mux.T
    
    # ゼロ行列のmuy（スパース）
    muy = csr_matrix((dim, dim), dtype=np.complex128)
    
    # 初期状態
    psi0 = np.zeros(dim, dtype=np.complex128)
    psi0[0] = 1.0
    
    return H0, mux, muy, psi0

def run_benchmark(dims, num_repeats=5, num_steps=1000):
    """ベンチマーク実行"""
    results = {
        'python': {dim: [] for dim in dims},
        'cpp': {dim: [] for dim in dims},
        'speedup': {dim: 0.0 for dim in dims}
    }
    
    # 電場パラメータ
    dt_E = 0.01
    E0 = 0.1
    omega_L = 1.0
    t = np.arange(0, dt_E * (num_steps+2), dt_E)
    Ex = E0 * np.sin(omega_L * t)
    Ey = np.zeros_like(Ex)
    
    for dim in dims:
        print(f"\n次元数: {dim}")
        H0, mux, muy, psi0 = create_test_system(dim)
        
        # Python実装
        print("Python実装の実行中...")
        for i in range(num_repeats):
            start_time = time.time()
            _ = rk4_cpu_sparse_py(H0, mux, muy, Ex, Ey, psi0, dt_E*2, True, 1, False)
            end_time = time.time()
            results['python'][dim].append(end_time - start_time)
            print(f"  反復 {i+1}/{num_repeats}: {results['python'][dim][-1]:.3f} 秒")
        
        # C++実装
        print("C++実装の実行中...")
        for i in range(num_repeats):
            start_time = time.time()
            _ = rk4_cpu_sparse_cpp.rk4_cpu_sparse(
                H0.data, H0.indices, H0.indptr,
                H0.shape[0], H0.shape[1],
                mux.data, mux.indices, mux.indptr,
                muy.data, muy.indices, muy.indptr,
                Ex, Ey, psi0, dt_E*2, True, 1, False
            )
            end_time = time.time()
            results['cpp'][dim].append(end_time - start_time)
            print(f"  反復 {i+1}/{num_repeats}: {results['cpp'][dim][-1]:.3f} 秒")
        
        # 平均速度向上率を計算
        py_mean = np.mean(results['python'][dim])
        cpp_mean = np.mean(results['cpp'][dim])
        results['speedup'][dim] = py_mean / cpp_mean
        print(f"平均速度向上率: {results['speedup'][dim]:.2f}倍")

    return results

def plot_results(results, dims):
    """結果をプロット"""
    plt.figure(figsize=(15, 5))
    
    # 実行時間の比較
    plt.subplot(121)
    x = np.arange(len(dims))
    width = 0.35
    
    py_means = [np.mean(results['python'][dim]) for dim in dims]
    py_stds = [np.std(results['python'][dim]) for dim in dims]
    cpp_means = [np.mean(results['cpp'][dim]) for dim in dims]
    cpp_stds = [np.std(results['cpp'][dim]) for dim in dims]
    
    plt.bar(x - width/2, py_means, width, label='Python', yerr=py_stds, capsize=5)
    plt.bar(x + width/2, cpp_means, width, label='C++', yerr=cpp_stds, capsize=5)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.xticks(x, dims)
    plt.legend()
    plt.grid(True)
    
    # 速度向上率
    plt.subplot(122)
    speedups = [results['speedup'][dim] for dim in dims]
    plt.plot(dims, speedups, 'bo-')
    plt.xlabel('Matrix Size')
    plt.ylabel('Speedup Ratio (Python/C++)')
    plt.title('C++ Implementation Speedup')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('examples/figures/benchmark_results.png')
    plt.close()

def main():
    # テストする行列サイズ
    dims = [2, 4, 8, 16, 32, 64]
    num_repeats = 5  # 各サイズでの繰り返し回数
    num_steps = 1000  # 時間発展のステップ数
    
    print("ベンチマーク開始")
    print(f"- 行列サイズ: {dims}")
    print(f"- 繰り返し回数: {num_repeats}")
    print(f"- 時間発展ステップ数: {num_steps}")
    
    results = run_benchmark(dims, num_repeats, num_steps)
    plot_results(results, dims)
    
    print("\nベンチマーク完了")
    print("結果は'examples/figures/benchmark_results.png'に保存されました")

if __name__ == "__main__":
    main() 