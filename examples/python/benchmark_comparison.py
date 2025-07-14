import sys
import os
import time
import json
import csv
from datetime import datetime
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# 現在のプロジェクト構造に対応
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../python'))

from rk4_sparse import rk4_sparse_py, rk4_sparse_cpp, rk4_numba_py

savepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(savepath, exist_ok=True)

def create_test_system(dim, num_steps=1000):
    """テストシステムを生成"""
    # ハミルトニアンと双極子演算子の生成
    H0 = csr_matrix(np.diag(np.arange(dim)), dtype=np.complex128)
    mux = csr_matrix(np.eye(dim, k=1) + np.eye(dim, k=-1), dtype=np.complex128)
    muy = csr_matrix((dim, dim), dtype=np.complex128)
    
    # 初期状態
    psi0 = np.zeros(dim, dtype=np.complex128)
    psi0[0] = 1.0
    
    # 電場パラメータ
    dt_E = 0.01
    E0 = 0.1
    omega_L = 1.0
    t = np.arange(0, dt_E * (num_steps+2), dt_E)
    Ex = E0 * np.sin(omega_L * t)
    Ey = np.zeros_like(Ex)
    
    return H0, mux, muy, Ex, Ey, psi0, dt_E

def run_benchmark(dims, num_repeats=100, num_steps=1000):
    """ベンチマークを実行"""
    results = {
        'numba': {dim: [] for dim in dims},
        'scipy.sparse': {dim: [] for dim in dims},
        'cpp': {dim: [] for dim in dims},
        'speedup_scipy.sparse': {dim: 0.0 for dim in dims},
        'speedup_cpp': {dim: 0.0 for dim in dims}
    }

    for dim in dims:
        print(f"\n次元数: {dim}")
        
        # テストシステムの生成
        H0, mux, muy, Ex, Ey, psi0, dt_E = create_test_system(dim)
        
        # scipy.sparse実装
        print("scipy.sparse実装の実行中...")
        for i in range(num_repeats):
            start_time = time.time()
            _ = rk4_sparse_py(H0, mux, muy, Ex, Ey, psi0, dt_E*2, True, 1, False)
            end_time = time.time()
            results['scipy.sparse'][dim].append(end_time - start_time)
            print(f"  反復 {i+1}/{num_repeats}: {results['scipy.sparse'][dim][-1]:.3f} 秒")
        
        # Numba実装
        print("Numba実装の実行中...")
        # H0, mux, muyをnp.ndarrayに変換
        H0_numba = H0.toarray()
        mux_numba = mux.toarray()
        muy_numba = muy.toarray()
        
        times = []
        for i in range(num_repeats):
            start_time = time.time()
            _ = rk4_numba_py(
                H0_numba, mux_numba, muy_numba,
                Ex.astype(np.float64), Ey.astype(np.float64),
                psi0,
                dt_E*2,
                True,
                1,
                False
            )
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"  反復 {i+1}/{num_repeats}: {times[-1]:.3f} 秒")
        results['numba'][dim] = times
        
        # C++実装
        print("C++実装の実行中...")
        times = []
        for i in range(num_repeats):
            start_time = time.time()
            _ = rk4_sparse_cpp(
                H0, mux, muy,
                Ex, Ey,
                psi0,
                dt_E*2,
                True,
                1,
                False
            )
            end_time = time.time()
            times.append(end_time - start_time)
            print(f"  反復 {i+1}/{num_repeats}: {times[-1]:.3f} 秒")
        results['cpp'][dim] = times
        
        # 平均速度向上率を計算
        py_mean = np.mean(results['scipy.sparse'][dim])
        numba_mean = np.mean(results['numba'][dim])
        cpp_mean = np.mean(results['cpp'][dim])
        results['speedup_scipy.sparse'][dim] = numba_mean / py_mean
        results['speedup_cpp'][dim] = numba_mean / cpp_mean
        print(f"Numba速度向上率: {results['speedup_scipy.sparse'][dim]:.2f}倍")
        print(f"C++速度向上率: {results['speedup_cpp'][dim]:.2f}倍")

    return results

def plot_results(results, dims):
    """結果をプロット"""
    plt.figure(figsize=(15, 10))
    
    # 実行時間の比較
    plt.subplot(221)
    x = np.arange(len(dims))
    width = 0.25
    
    py_means = [np.mean(results['scipy.sparse'][dim]) for dim in dims]
    py_stds = [np.std(results['scipy.sparse'][dim]) for dim in dims]
    numba_means = [np.mean(results['numba'][dim]) for dim in dims]
    numba_stds = [np.std(results['numba'][dim]) for dim in dims]
    cpp_means = [np.mean(results['cpp'][dim]) for dim in dims]
    cpp_stds = [np.std(results['cpp'][dim]) for dim in dims]
    
    plt.bar(x - width, py_means, width, label='scipy.sparse', yerr=py_stds, capsize=5)
    plt.bar(x, numba_means, width, label='Numba', yerr=numba_stds, capsize=5)
    plt.bar(x + width, cpp_means, width, label='C++', yerr=cpp_stds, capsize=5)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.xticks(x, dims)
    plt.legend()
    plt.grid(True)
    
    # scipy.sparse vs C++速度向上率
    plt.subplot(222)
    speedups_scipy_sparse = [results['speedup_scipy.sparse'][dim] for dim in dims]
    plt.plot(dims, speedups_scipy_sparse, 'go-', label='scipy.sparse')
    plt.xlabel('Matrix Size')
    plt.ylabel('Speedup Ratio (Numba/scipy.sparse)')
    plt.title('scipy.sparse Implementation Speedup')
    plt.grid(True)
    plt.legend()
    
    # C++速度向上率
    plt.subplot(223)
    speedups_cpp = [results['speedup_cpp'][dim] for dim in dims]
    plt.plot(dims, speedups_cpp, 'ro-', label='C++')
    plt.xlabel('Matrix Size')
    plt.ylabel('Speedup Ratio (Numba/C++)')
    plt.title('C++ Implementation Speedup')
    plt.grid(True)
    plt.legend()
    
    # 全体的な速度向上率の比較
    plt.subplot(224)
    plt.plot(dims, speedups_scipy_sparse, 'go-', label='scipy.sparse', linewidth=2)
    plt.plot(dims, speedups_cpp, 'ro-', label='C++', linewidth=2)
    plt.xlabel('Matrix Size')
    plt.ylabel('Speedup Ratio')
    plt.title('Overall Speedup Comparison')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'benchmark_results.png'))
    plt.close()

def save_benchmark_results(results, dims, savepath):
    """ベンチマーク結果をファイルに保存"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSONファイルに保存
    json_filename = os.path.join(savepath, f'benchmark_results_{timestamp}.json')
    results_data = {
        'timestamp': timestamp,
        'dimensions': dims,
        'results': results
    }
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    # CSVファイルに保存
    csv_filename = os.path.join(savepath, f'benchmark_results_{timestamp}.csv')
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # ヘッダー行
        writer.writerow([
            'Dimension', 'Implementation', 'Mean_Time_seconds', 'Std_Time_seconds',
            'Speedup_vs_scipy_sparse', 'Speedup_vs_cpp'
        ])
        
        # データ行
        for dim in dims:
            # scipy.sparse実装
            py_mean = np.mean(results['scipy.sparse'][dim])
            py_std = np.std(results['scipy.sparse'][dim])
            writer.writerow([
                dim, 'scipy.sparse', py_mean, py_std,
                results['speedup_scipy.sparse'][dim], results['speedup_cpp'][dim]
            ])
            
            # Numba実装
            numba_mean = np.mean(results['numba'][dim])
            numba_std = np.std(results['numba'][dim])
            writer.writerow([
                dim, 'numba', numba_mean, numba_std,
                results['speedup_scipy.sparse'][dim], results['speedup_cpp'][dim]
            ])
            
            # C++実装
            cpp_mean = np.mean(results['cpp'][dim])
            cpp_std = np.std(results['cpp'][dim])
            writer.writerow([
                dim, 'cpp', cpp_mean, cpp_std,
                results['speedup_scipy.sparse'][dim], results['speedup_cpp'][dim]
            ])
    
    print(f"Results saved to:")
    print(f"  JSON: {json_filename}")
    print(f"  CSV:  {csv_filename}")
    
    return json_filename, csv_filename

def main():
    # テストする行列サイズ
    dims = [2, 4, 8, 16, 32, 64, 128, 256]
    num_repeats = 100  # 各サイズでの繰り返し回数
    num_steps = 10000  # 時間発展のステップ数
    
    print("ベンチマーク開始")
    print(f"- 行列サイズ: {dims}")
    print(f"- 繰り返し回数: {num_repeats}")
    print(f"- 時間発展ステップ数: {num_steps}")
    
    results = run_benchmark(dims, num_repeats, num_steps)
    plot_results(results, dims)
    
    # 結果をファイルに保存
    print("\n=== Saving Results ===")
    save_benchmark_results(results, dims, savepath)
    
    print("\nベンチマーク完了")
    print("結果は{}に保存されました".format(os.path.join(savepath, 'benchmark_results.png')))

if __name__ == "__main__":
    main() 