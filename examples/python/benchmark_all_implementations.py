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

try:
    from rk4_sparse import rk4_sparse_py, rk4_numba_py, rk4_sparse_eigen, rk4_sparse_suitesparse
except ImportError as e:
    print(f"Warning: Could not import rk4_sparse: {e}")
    print("Please make sure the module is built and installed correctly.")
    sys.exit(1)

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
    implementations = ['python', 'numba', 'eigen', 'suitesparse']
    results = {impl: {dim: [] for dim in dims} for impl in implementations}
    speedups = {impl: {dim: 0.0 for dim in dims} for impl in implementations}

    for dim in dims:
        print(f"\n次元数: {dim}")
        
        # テストシステムの生成
        H0, mux, muy, Ex, Ey, psi0, dt_E = create_test_system(dim)
        
        # Python実装
        print("Python実装の実行中...")
        for i in range(num_repeats):
            start_time = time.time()
            _ = rk4_sparse_py(H0, mux, muy, Ex, Ey, psi0, dt_E*2, True, 1, False)
            end_time = time.time()
            results['python'][dim].append(end_time - start_time)
            print(f"  反復 {i+1}/{num_repeats}: {results['python'][dim][-1]:.3f} 秒")
        
        # Numba実装
        print("Numba実装の実行中...")
        # H0, mux, muyをnp.ndarrayに変換
        H0_numba = H0.toarray()
        mux_numba = mux.toarray()
        muy_numba = muy.toarray()
        
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
            results['numba'][dim].append(end_time - start_time)
            print(f"  反復 {i+1}/{num_repeats}: {results['numba'][dim][-1]:.3f} 秒")
        
        # C++ Eigen実装
        print("C++ Eigen実装の実行中...")
        for i in range(num_repeats):
            start_time = time.time()
            _ = rk4_sparse_eigen(
                H0, mux, muy,
                Ex, Ey,
                psi0,
                dt_E*2,
                True,
                1,
                False
            )
            end_time = time.time()
            results['eigen'][dim].append(end_time - start_time)
            print(f"  反復 {i+1}/{num_repeats}: {results['eigen'][dim][-1]:.3f} 秒")
        
        # C++ SuiteSparse実装
        print("C++ SuiteSparse実装の実行中...")
        try:
            for i in range(num_repeats):
                start_time = time.time()
                _ = rk4_sparse_suitesparse(
                    H0, mux, muy,
                    Ex, Ey,
                    psi0,
                    dt_E*2,
                    True,
                    1,
                    False,
                    1  # level=1 (STANDARD)
                )
                end_time = time.time()
                results['suitesparse'][dim].append(end_time - start_time)
                print(f"  反復 {i+1}/{num_repeats}: {results['suitesparse'][dim][-1]:.3f} 秒")
        except Exception as e:
            print(f"SuiteSparse実装でエラーが発生しました: {e}")
            results['suitesparse'][dim] = [float('inf')] * num_repeats
        
        # 速度向上率を計算（Python実装を基準）
        python_mean = np.mean(results['python'][dim])
        for impl in implementations:
            if impl != 'python':
                impl_mean = np.mean(results[impl][dim])
                if impl_mean > 0:
                    speedups[impl][dim] = python_mean / impl_mean
                else:
                    speedups[impl][dim] = 0.0
        
        print(f"速度向上率（Python基準）:")
        for impl in implementations:
            if impl != 'python':
                print(f"  {impl}: {speedups[impl][dim]:.2f}倍")

    return results, speedups

def plot_results(results, speedups, dims):
    """結果をプロット"""
    plt.figure(figsize=(20, 15))
    
    # 実行時間の比較
    plt.subplot(2, 3, 1)
    x = np.arange(len(dims))
    width = 0.2
    
    implementations = ['python', 'numba', 'eigen', 'suitesparse']
    colors = ['blue', 'green', 'red', 'orange']
    
    for i, impl in enumerate(implementations):
        means = [np.mean(results[impl][dim]) for dim in dims]
        stds = [np.std(results[impl][dim]) for dim in dims]
        plt.bar(x + i*width, means, width, label=impl, yerr=stds, capsize=5, color=colors[i])
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.xticks(x + width*1.5, dims)
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # 速度向上率の比較
    plt.subplot(2, 3, 2)
    for impl in ['numba', 'eigen', 'suitesparse']:
        speedup_values = [speedups[impl][dim] for dim in dims]
        plt.plot(dims, speedup_values, 'o-', label=impl, linewidth=2, markersize=6)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Speedup Ratio (vs Python)')
    plt.title('Speedup Comparison')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    
    # 各実装の詳細比較
    plt.subplot(2, 3, 3)
    for impl in implementations:
        means = [np.mean(results[impl][dim]) for dim in dims]
        plt.plot(dims, means, 'o-', label=impl, linewidth=2, markersize=6)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time by Implementation')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    
    # 速度向上率の詳細
    plt.subplot(2, 3, 4)
    for impl in ['numba', 'eigen', 'suitesparse']:
        speedup_values = [speedups[impl][dim] for dim in dims]
        plt.plot(dims, speedup_values, 'o-', label=impl, linewidth=2, markersize=6)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Speedup Ratio')
    plt.title('Speedup vs Python Implementation')
    plt.grid(True)
    plt.legend()
    
    # Eigen vs SuiteSparse比較
    plt.subplot(2, 3, 5)
    eigen_vs_suitesparse = []
    for dim in dims:
        eigen_mean = np.mean(results['eigen'][dim])
        suitesparse_mean = np.mean(results['suitesparse'][dim])
        if suitesparse_mean > 0:
            eigen_vs_suitesparse.append(eigen_mean / suitesparse_mean)
        else:
            eigen_vs_suitesparse.append(1.0)
    
    plt.plot(dims, eigen_vs_suitesparse, 'o-', label='Eigen/SuiteSparse', linewidth=2, markersize=6)
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Equal performance')
    plt.xlabel('Matrix Size')
    plt.ylabel('Speedup Ratio (Eigen/SuiteSparse)')
    plt.title('Eigen vs SuiteSparse Performance')
    plt.grid(True)
    plt.legend()
    
    # 統計情報
    plt.subplot(2, 3, 6)
    # 最大次元での各実装の統計
    max_dim = max(dims)
    stats_data = []
    labels = []
    
    for impl in implementations:
        times = results[impl][max_dim]
        if all(t != float('inf') for t in times):
            stats_data.append(times)
            labels.append(impl)
    
    if stats_data:
        plt.boxplot(stats_data)
        plt.xticks(range(1, len(labels) + 1), labels)
        plt.ylabel('Execution Time (seconds)')
        plt.title(f'Performance Distribution (dim={max_dim})')
        plt.grid(True)
        plt.yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'benchmark_all_implementations.png'), dpi=300, bbox_inches='tight')
    plt.close()

def save_benchmark_results(results, speedups, dims, savepath):
    """ベンチマーク結果をファイルに保存"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSONファイルに保存
    json_filename = os.path.join(savepath, f'benchmark_all_implementations_{timestamp}.json')
    results_data = {
        'timestamp': timestamp,
        'dimensions': dims,
        'results': results,
        'speedups': speedups
    }
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    # CSVファイルに保存
    csv_filename = os.path.join(savepath, f'benchmark_all_implementations_{timestamp}.csv')
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # ヘッダー行
        writer.writerow([
            'Dimension', 'Implementation', 'Mean_Time_seconds', 'Std_Time_seconds',
            'Min_Time_seconds', 'Max_Time_seconds', 'Speedup_vs_Python'
        ])
        
        # データ行
        for dim in dims:
            for impl in ['python', 'numba', 'eigen', 'suitesparse']:
                times = results[impl][dim]
                if all(t != float('inf') for t in times):
                    mean_time = np.mean(times)
                    std_time = np.std(times)
                    min_time = np.min(times)
                    max_time = np.max(times)
                    speedup = speedups[impl][dim] if impl != 'python' else 1.0
                else:
                    mean_time = std_time = min_time = max_time = float('inf')
                    speedup = 0.0
                
                writer.writerow([
                    dim, impl, mean_time, std_time, min_time, max_time, speedup
                ])
    
    print(f"Results saved to:")
    print(f"  JSON: {json_filename}")
    print(f"  CSV:  {csv_filename}")
    
    return json_filename, csv_filename

def print_summary(results, speedups, dims):
    """結果のサマリーを表示"""
    print("\n" + "="*60)
    print("ベンチマーク結果サマリー")
    print("="*60)
    
    implementations = ['python', 'numba', 'eigen', 'suitesparse']
    
    # 各次元での最速実装
    print("\n各次元での最速実装:")
    for dim in dims:
        best_impl = None
        best_time = float('inf')
        for impl in implementations:
            times = results[impl][dim]
            if all(t != float('inf') for t in times):
                mean_time = np.mean(times)
                if mean_time < best_time:
                    best_time = mean_time
                    best_impl = impl
        
        if best_impl:
            print(f"  次元 {dim}: {best_impl} ({best_time:.6f}秒)")
    
    # 平均速度向上率
    print("\n平均速度向上率（Python基準）:")
    for impl in ['numba', 'eigen', 'suitesparse']:
        avg_speedup = np.mean([speedups[impl][dim] for dim in dims])
        print(f"  {impl}: {avg_speedup:.2f}倍")
    
    # 最大次元での詳細比較
    max_dim = max(dims)
    print(f"\n最大次元（{max_dim}）での詳細比較:")
    for impl in implementations:
        times = results[impl][max_dim]
        if all(t != float('inf') for t in times):
            mean_time = np.mean(times)
            std_time = np.std(times)
            speedup = speedups[impl][max_dim] if impl != 'python' else 1.0
            print(f"  {impl}: {mean_time:.6f}±{std_time:.6f}秒 (速度向上率: {speedup:.2f}倍)")
        else:
            print(f"  {impl}: 実行失敗")

def main():
    # テストする行列サイズ
    dims = [2, 4, 8, 16, 32, 64, 128, 256]
    num_repeats = 50  # 各サイズでの繰り返し回数
    num_steps = 1000  # 時間発展のステップ数
    
    print("全実装ベンチマーク開始")
    print(f"- 行列サイズ: {dims}")
    print(f"- 繰り返し回数: {num_repeats}")
    print(f"- 時間発展ステップ数: {num_steps}")
    print(f"- 実装: Python, Numba, C++ Eigen, C++ SuiteSparse")
    
    results, speedups = run_benchmark(dims, num_repeats, num_steps)
    
    # 結果のサマリーを表示
    print_summary(results, speedups, dims)
    
    # 結果をプロット
    print("\n=== Plotting Results ===")
    plot_results(results, speedups, dims)
    
    # 結果をファイルに保存
    print("\n=== Saving Results ===")
    save_benchmark_results(results, speedups, dims, savepath)
    
    print("\nベンチマーク完了")
    print("結果は{}に保存されました".format(os.path.join(savepath, 'benchmark_all_implementations.png')))

if __name__ == "__main__":
    main() 