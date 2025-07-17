import sys
import os
import time
import json
import csv
from datetime import datetime
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
import psutil
import tracemalloc
import multiprocessing
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional

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

@dataclass
class DetailedBenchmarkResult:
    """細かなベンチマーク結果を格納するデータクラス"""
    implementation: str
    dimension: int
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    speedup_vs_python: float
    cpu_usage: float
    memory_usage: float
    memory_peak: float
    thread_count: int
    cache_misses: int = 0
    context_switches: int = 0
    cpu_migrations: int = 0
    timestamp: datetime = datetime.now()
    
    def to_dict(self) -> dict:
        """結果を辞書形式に変換（JSON保存用）"""
        result_dict = asdict(self)
        result_dict['timestamp'] = self.timestamp.isoformat()
        return result_dict

class DetailedPerformanceProfiler:
    """詳細な性能プロファイリングを行うクラス"""
    def __init__(self):
        self.process = psutil.Process()
        self.results = []
    
    def profile_execution(self, func, *args, **kwargs) -> DetailedBenchmarkResult:
        """関数の実行を詳細にプロファイリング"""
        # メモリトラッキング開始
        tracemalloc.start()
        
        # 初期状態の記録
        initial_cpu_samples = [self.process.cpu_percent() for _ in range(3)]
        initial_cpu = sum(initial_cpu_samples) / len(initial_cpu_samples)
        initial_memory = self.process.memory_info().rss / 1024**2 # MB
        initial_threads = self.process.num_threads()
        
        # 実行時間の計測
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time
        
        # 実行後の状態記録
        current_cpu_samples = [self.process.cpu_percent() for _ in range(3)]
        current_cpu = sum(current_cpu_samples) / len(current_cpu_samples)
        current_memory = self.process.memory_info().rss / 1024**2 # MB
        current_threads = self.process.num_threads()
        
        # メモリ使用量の計測
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        return DetailedBenchmarkResult(
            implementation=func.__name__,
            dimension=0,  # 後で設定
            mean_time=execution_time,
            std_time=0.0,  # 単回実行なので0
            min_time=execution_time,
            max_time=execution_time,
            speedup_vs_python=0.0,  # 後で計算
            cpu_usage=current_cpu,
            memory_usage=current_memory - initial_memory,
            memory_peak=peak / 1024 / 1024, # MB
            thread_count=current_threads
        )

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

def run_detailed_benchmark(dims, num_repeats=100, num_steps=1000):
    """詳細なベンチマークを実行"""
    implementations = ['python', 'numba', 'eigen', 'suitesparse']
    results = {impl: {dim: [] for dim in dims} for impl in implementations}
    detailed_results: Dict[str, Dict[int, Optional[DetailedBenchmarkResult]]] = {impl: {dim: None for dim in dims} for impl in implementations}
    
    profiler = DetailedPerformanceProfiler()

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
        
        # 詳細プロファイリング（最後の1回）
        detailed_result = profiler.profile_execution(
            rk4_sparse_py, H0, mux, muy, Ex, Ey, psi0, dt_E*2, True, 1, False
        )
        detailed_result.dimension = dim
        detailed_results['python'][dim] = detailed_result
        
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
        
        # 詳細プロファイリング
        detailed_result = profiler.profile_execution(
            rk4_numba_py, H0_numba, mux_numba, muy_numba,
            Ex.astype(np.float64), Ey.astype(np.float64),
            psi0, dt_E*2, True, 1, False
        )
        detailed_result.dimension = dim
        detailed_results['numba'][dim] = detailed_result
        
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
        
        # 詳細プロファイリング
        detailed_result = profiler.profile_execution(
            rk4_sparse_eigen, H0, mux, muy, Ex, Ey, psi0, dt_E*2, True, 1, False
        )
        detailed_result.dimension = dim
        detailed_results['eigen'][dim] = detailed_result
        
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
            
            # 詳細プロファイリング
            detailed_result = profiler.profile_execution(
                rk4_sparse_suitesparse, H0, mux, muy, Ex, Ey, psi0, 
                dt_E*2, True, 1, False, 1
            )
            detailed_result.dimension = dim
            detailed_results['suitesparse'][dim] = detailed_result
            
        except Exception as e:
            print(f"SuiteSparse実装でエラーが発生しました: {e}")
            results['suitesparse'][dim] = [float('inf')] * num_repeats
            detailed_results['suitesparse'][dim] = None
        
        # 速度向上率を計算（Python実装を基準）
        python_mean = np.mean(results['python'][dim])
        for impl in implementations:
            if impl != 'python':
                impl_mean = np.mean(results[impl][dim])
                if impl_mean > 0:
                    speedup = float(python_mean / impl_mean)
                    if detailed_results[impl][dim] is not None:
                        detailed_results[impl][dim].speedup_vs_python = speedup
                else:
                    if detailed_results[impl][dim] is not None:
                        detailed_results[impl][dim].speedup_vs_python = 0.0    
        print(f"速度向上率（Python基準）:")
        for impl in implementations:
            if impl != 'python':
                if detailed_results[impl][dim] is not None:
                    print(f"  {impl}: {detailed_results[impl][dim].speedup_vs_python:.2f}倍")
        
    return results, detailed_results

def plot_detailed_results(results, detailed_results, dims, num_steps=1000, num_repeats=10):
    """詳細な結果をプロット"""
    plt.figure(figsize=(24, 18))
    
    implementations = ['python', 'numba', 'eigen', 'suitesparse']
    colors = ['blue', 'green', 'red', 'orange']
    
    # 1行時間の比較
    plt.subplot(3, 4, 1)
    x = np.arange(len(dims))
    width = 0.2  
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
    
    # 2. CPU使用率の比較
    plt.subplot(3, 4, 2)
    for i, impl in enumerate(implementations):
        cpu_values = [detailed_results[impl][dim].cpu_usage if detailed_results[impl][dim] is not None else 0 for dim in dims]
        plt.plot(dims, cpu_values, 'o-', label=impl, color=colors[i], linewidth=2, markersize=6)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage Comparison')
    plt.grid(True)
    plt.legend()
    
    # 3. メモリ使用量の比較
    plt.subplot(3, 4, 3)
    for i, impl in enumerate(implementations):
        mem_values = [detailed_results[impl][dim].memory_usage if detailed_results[impl][dim] is not None else 0 for dim in dims]
        plt.plot(dims, mem_values, 'o-', label=impl, color=colors[i], linewidth=2, markersize=6)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    plt.grid(True)
    plt.legend()
    
    # 4. スレッド数の比較
    plt.subplot(3, 4, 4)
    for i, impl in enumerate(implementations):
        thread_values = [detailed_results[impl][dim].thread_count if detailed_results[impl][dim] is not None else 0 for dim in dims]
        plt.plot(dims, thread_values, 'o-', label=impl, color=colors[i], linewidth=2, markersize=6)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Thread Count')
    plt.title('Thread Count Comparison')
    plt.grid(True)
    plt.legend()
    
    # 5. 速度向上率の比較
    plt.subplot(3, 4, 5)
    for impl in ['numba', 'eigen', 'suitesparse']:
        speedup_values = [detailed_results[impl][dim].speedup_vs_python if detailed_results[impl][dim] is not None else 0 for dim in dims]
        plt.plot(dims, speedup_values, 'o-', label=impl, linewidth=2, markersize=6)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Speedup Ratio (vs Python)')
    plt.title('Speedup Comparison')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    
    # 6. メモリ効率（実行時間あたりのメモリ使用量）
    plt.subplot(3, 4, 6)
    for i, impl in enumerate(implementations):
        efficiency_values = []
        for dim in dims:
            if detailed_results[impl][dim] is not None:
                time_per_step = detailed_results[impl][dim].mean_time / 1000 # 正規化
                mem_per_step = detailed_results[impl][dim].memory_usage / 100
                efficiency_values.append(mem_per_step / time_per_step if time_per_step > 0 else 0)
            else:
                efficiency_values.append(0)
        plt.plot(dims, efficiency_values, 'o-', label=impl, color=colors[i], linewidth=2, markersize=6)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Memory/Time Ratio (MB/s)')
    plt.title('Memory Efficiency')
    plt.grid(True)
    plt.legend()
    
    # 7. Eigen vs SuiteSparse詳細比較
    plt.subplot(3, 4, 7)
    eigen_vs_suitesparse = []
    for dim in dims:
        if detailed_results['eigen'][dim] is not None and detailed_results['suitesparse'][dim] is not None:
            eigen_time = detailed_results['eigen'][dim].mean_time
            suitesparse_time = detailed_results['suitesparse'][dim].mean_time
            eigen_vs_suitesparse.append(eigen_time / suitesparse_time if suitesparse_time > 0 else 1.0)
        else:
            eigen_vs_suitesparse.append(1.0)
    
    plt.plot(dims, eigen_vs_suitesparse, 'o-', label='Eigen/SuiteSparse', linewidth=2, markersize=6)
    plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Equal performance')
    plt.xlabel('Matrix Size')
    plt.ylabel('Speedup Ratio (Eigen/SuiteSparse)')
    plt.title('Eigen vs SuiteSparse Performance')
    plt.grid(True)
    plt.legend()
    
    # 8. CPU使用率 vs 実行時間の相関
    plt.subplot(3, 4, 8)
    for i, impl in enumerate(implementations):
        times = []
        cpus = []
        for dim in dims:
            if detailed_results[impl][dim]:
                times.append(detailed_results[impl][dim].mean_time)
                cpus.append(detailed_results[impl][dim].cpu_usage)
        if times:
            plt.scatter(times, cpus, label=impl, color=colors[i], alpha=0.7)
    
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage vs Execution Time')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    
    # 9. メモリ使用量 vs 実行時間の相関
    plt.subplot(3, 4, 9)
    for i, impl in enumerate(implementations):
        times = []
        mems = []
        for dim in dims:
            if detailed_results[impl][dim]:
                times.append(detailed_results[impl][dim].mean_time)
                mems.append(detailed_results[impl][dim].memory_usage)
        if times:
            plt.scatter(times, mems, label=impl, color=colors[i], alpha=0.7)
    
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage vs Execution Time')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    
    # 10. スレッド数 vs 実行時間の相関
    plt.subplot(3, 4, 10)
    for i, impl in enumerate(implementations):
        times = []
        threads = []
        for dim in dims:
            if detailed_results[impl][dim]:
                times.append(detailed_results[impl][dim].mean_time)
                threads.append(detailed_results[impl][dim].thread_count)
        if times:
            plt.scatter(times, threads, label=impl, color=colors[i], alpha=0.7)
    
    plt.xlabel('Execution Time (seconds)')
    plt.ylabel('Thread Count')
    plt.title('Thread Count vs Execution Time')
    plt.grid(True)
    plt.legend()
    plt.xscale('log')
    
    # 11. 最大次元での統計情報
    plt.subplot(3, 4, 11)
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
    
    # 12. システム情報
    plt.subplot(3, 4, 12)
    plt.axis('off')
    system_info = f"""
    System Information:
    CPU Cores: {multiprocessing.cpu_count()}
    Total Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB
    Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB
    
    Benchmark Parameters:
    Dimensions: {dims}
    Steps: {num_steps}
    Repeats: {num_repeats}
    """
    plt.text(0.1, 0.5, system_info, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'detailed_benchmark_all_implementations.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def save_detailed_benchmark_results(results, detailed_results, dims, savepath):
    """詳細なベンチマーク結果をファイルに保存"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSONファイルに保存
    json_filename = os.path.join(savepath, f'detailed_benchmark_all_implementations_{timestamp}.json')
    
    # 結果を辞書形式に変換
    results_data = {
        'timestamp': timestamp,
        'dimensions': dims,
        'system_info': {
            'cpu_cores': multiprocessing.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3)
        },
        'detailed_results': {}
    }
    
    # 詳細結果を変換
    for impl in ['python', 'numba', 'eigen', 'suitesparse']:
        results_data['detailed_results'][impl] = {}
        for dim in dims:
            if detailed_results[impl][dim]:
                results_data['detailed_results'][impl][dim] = detailed_results[impl][dim].to_dict()
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    # CSVファイルに保存
    csv_filename = os.path.join(savepath, f'detailed_benchmark_all_implementations_{timestamp}.csv')
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # ヘッダー行
        writer.writerow([
            'Dimension', 'Implementation', 'Mean_Time_seconds', 'Std_Time_seconds',
            'Min_Time_seconds', 'Max_Time_seconds', 'Speedup_vs_Python',
            'CPU_Usage_percent', 'Memory_Usage_MB', 'Memory_Peak_MB', 'Thread_Count'
        ])
        
        # データ行
        for dim in dims:
            for impl in ['python', 'numba', 'eigen', 'suitesparse']:
                if detailed_results[impl][dim]:
                    result = detailed_results[impl][dim]
                    writer.writerow([
                        dim, impl, result.mean_time, result.std_time,
                        result.min_time, result.max_time, result.speedup_vs_python,
                        result.cpu_usage, result.memory_usage, result.memory_peak, result.thread_count
                    ])
    
    print(f"Detailed results saved to:")
    print(f"  JSON: {json_filename}")
    print(f"  CSV:  {csv_filename}")
    
    return json_filename, csv_filename

def print_detailed_summary(results, detailed_results, dims):
    """詳細な結果のサマリーを表示"""
    print("\n" + "="*80)
    print("詳細ベンチマーク結果サマリー")
    print("="*80)
    
    implementations = ['python', 'numba', 'eigen', 'suitesparse']
    
    # システム情報
    print(f"\nシステム情報:")
    print(f"  CPU コア数: {multiprocessing.cpu_count()}")
    print(f"  総メモリ: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"  利用可能メモリ: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    # 各次元での最速実装
    print("\n各次元での最速実装:")
    for dim in dims:
        best_impl = None
        best_time = float('inf')
        for impl in implementations:
            if detailed_results[impl][dim]:
                if detailed_results[impl][dim].mean_time < best_time:
                    best_time = detailed_results[impl][dim].mean_time
                    best_impl = impl
        
        if best_impl:
            print(f"  次元 {dim}: {best_impl} ({best_time:.6f}秒)")
    
    # 平均性能指標
    print("\n平均性能指標（全次元）:")
    for impl in implementations:
        times = []
        cpus = []
        mems = []
        threads = []
        for dim in dims:
            if detailed_results[impl][dim]:
                times.append(detailed_results[impl][dim].mean_time)
                cpus.append(detailed_results[impl][dim].cpu_usage)
                mems.append(detailed_results[impl][dim].memory_usage)
                threads.append(detailed_results[impl][dim].thread_count)
        
        if times:
            avg_time = np.mean(times)
            avg_cpu = np.mean(cpus)
            avg_mem = np.mean(mems)
            avg_threads = np.mean(threads)
            print(f"  {impl}:")
            print(f"    平均実行時間: {avg_time:.6f}秒")
            print(f"    平均CPU使用率: {avg_cpu:.1f}%")
            print(f"    平均メモリ使用量: {avg_mem:.1f} MB")
            print(f"    平均スレッド数: {avg_threads:.1f}")
    
    # 最大次元での詳細比較
    max_dim = max(dims)
    print(f"\n最大次元（{max_dim}）での詳細比較:")
    for impl in implementations:
        if detailed_results[impl][max_dim]:
            result = detailed_results[impl][max_dim]
            print(f"  {impl}:")
            print(f"    実行時間: {result.mean_time:.6f}秒")
            print(f"    CPU使用率: {result.cpu_usage:.1f}%")
            print(f"    メモリ使用量: {result.memory_usage:.1f} MB")
            print(f"    ピークメモリ: {result.memory_peak:.1f} MB")
            print(f"    スレッド数: {result.thread_count}")
            print(f"    速度向上率: {result.speedup_vs_python:.2f}倍")

def main():
    # テストする行列サイズ
    dims = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    num_repeats = 5 # 各サイズでの繰り返し回数
    num_steps = 1000 # 時間発展のステップ数
    
    print("詳細ベンチマーク開始")
    print(f"- 行列サイズ: {dims}")
    print(f"- 繰り返し回数: {num_repeats}")
    print(f"- 時間発展ステップ数: {num_steps}")
    print(f"- 実装: Python, Numba, C++ Eigen, C++ SuiteSparse")
    print(f"- 追加メトリクス: CPU使用率, メモリ使用量, スレッド数")
    
    results, detailed_results = run_detailed_benchmark(dims, num_repeats, num_steps)
    
    # 結果のサマリーを表示
    print_detailed_summary(results, detailed_results, dims)
    
    # 結果をプロット
    print("\n=== Plotting Detailed Results ===")
    plot_detailed_results(results, detailed_results, dims, num_steps, num_repeats)
    
    # 結果をファイルに保存
    print("\n=== Saving Detailed Results ===")
    save_detailed_benchmark_results(results, detailed_results, dims, savepath)
    
    print("\n詳細ベンチマーク完了")
    print(f"結果は{os.path.join(savepath, 'detailed_benchmark_all_implementations.png')}に保存されました")

if __name__ == "__main__":
    main() 