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
from typing import Dict, List, Tuple, Optional, Set

# 現在のプロジェクト構造に対応
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../python'))

try:
    from rk4_sparse import rk4_sparse_py, rk4_numba_py, rk4_sparse_eigen, rk4_sparse_suitesparse, rk4_sparse_eigen_direct_csr
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

def get_implementation_function(impl_name: str):
    """実装名から関数を取得"""
    implementation_map = {
        'python': rk4_sparse_py,
        'numba': rk4_numba_py,
        'eigen': rk4_sparse_eigen,
        'eigen_direct_csr': rk4_sparse_eigen_direct_csr,  # 新しい最適化実装
        'suitesparse': rk4_sparse_suitesparse
    }
    return implementation_map.get(impl_name)

def prepare_arguments_for_implementation(impl_name: str, H0, mux, muy, Ex, Ey, psi0, dt_E):
    """実装に応じて引数を準備"""
    if impl_name == 'numba':
        # Numba実装はnumpy配列を要求
        H0_numba = H0.toarray()
        mux_numba = mux.toarray()
        muy_numba = muy.toarray()
        return (H0_numba, mux_numba, muy_numba, 
                Ex.astype(np.float64), Ey.astype(np.float64), 
                psi0, dt_E*2, True, 1, False)
    elif impl_name == 'eigen_direct_csr':
        # 新しい直接CSR実装はCSRデータを直接渡す
        return (H0.data, H0.indices, H0.indptr,
                mux.data, mux.indices, mux.indptr,
                muy.data, muy.indices, muy.indptr,
                Ex, Ey, psi0, dt_E*2, True, 1, False)
    elif impl_name == 'suitesparse':
        # SuiteSparse実装は追加のlevelパラメータを要求
        return (H0, mux, muy, Ex, Ey, psi0, dt_E*2, True, 1, False, 1)
    else:
        # PythonとEigen実装は標準的な引数
        return (H0, mux, muy, Ex, Ey, psi0, dt_E*2, True, 1, False)

def run_detailed_benchmark(dims, selected_implementations: Set[str], num_repeats=100, num_steps=1000):
    """選択された実装で詳細なベンチマークを実行"""
    results = {impl: {dim: [] for dim in dims} for impl in selected_implementations}
    detailed_results: Dict[str, Dict[int, Optional[DetailedBenchmarkResult]]] = {
        impl: {dim: None for dim in dims} for impl in selected_implementations
    }
    
    profiler = DetailedPerformanceProfiler()

    for dim in dims:
        print(f"\n次元数: {dim}")
        
        # テストシステムの生成
        H0, mux, muy, Ex, Ey, psi0, dt_E = create_test_system(dim)
        
        # 各実装のベンチマーク実行
        for impl in selected_implementations:
            print(f"{impl}実装の実行中...")
            
            try:
                # 実装関数を取得
                impl_func = get_implementation_function(impl)
                if impl_func is None:
                    print(f"  警告: {impl}実装が見つかりません")
                    continue
                
                # 引数を準備
                args = prepare_arguments_for_implementation(impl, H0, mux, muy, Ex, Ey, psi0, dt_E)
                
                # 繰り返し実行
                for i in range(num_repeats):
                    start_time = time.time()
                    _ = impl_func(*args)
                    end_time = time.time()
                    results[impl][dim].append(end_time - start_time)
                    print(f"  反復 {i+1}/{num_repeats}: {results[impl][dim][-1]:.3f} 秒")
                
                # 詳細プロファイリング（最後の1回）
                detailed_result = profiler.profile_execution(impl_func, *args)
                detailed_result.dimension = dim
                detailed_results[impl][dim] = detailed_result
                
            except Exception as e:
                print(f"  {impl}実装でエラーが発生しました: {e}")
                results[impl][dim] = [float('inf')] * num_repeats
                detailed_results[impl][dim] = None
        
        # 速度向上率を計算（Python実装を基準）
        if 'python' in selected_implementations:
            python_mean = np.mean(results['python'][dim])
            for impl in selected_implementations:
                if impl != 'python' and detailed_results[impl][dim] is not None:
                    impl_mean = np.mean(results[impl][dim])
                    if impl_mean > 0:
                        speedup = float(python_mean / impl_mean)
                        detailed_results[impl][dim].speedup_vs_python = speedup
                    else:
                        detailed_results[impl][dim].speedup_vs_python = 0.0
            
            print(f"速度向上率（Python基準）:")
            for impl in selected_implementations:
                if impl != 'python' and detailed_results[impl][dim] is not None:
                    result = detailed_results[impl][dim]
                    if result is not None:
                        print(f"  {impl}: {result.speedup_vs_python:.2f}倍")
        
    return results, detailed_results

def plot_detailed_results(results, detailed_results, dims, selected_implementations, num_steps=1000, num_repeats=10):
    """詳細な結果をプロット"""
    impl_list = list(selected_implementations)
    colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']
    
    # プロット数を計算（選択された実装数に応じて）
    num_impls = len(impl_list)
    if num_impls <= 2:
        cols = 2
    elif num_impls <= 4:
        cols = 4
    else:
        cols = 4
    
    rows = (num_impls + cols - 1) // cols + 2  # 追加の行を確保
    
    plt.figure(figsize=(6*cols, 5*rows))
    
    # 1. 実行時間の比較
    plt.subplot(rows, cols, 1)
    x = np.arange(len(dims))
    width = 0.8 / len(impl_list)
    
    for i, impl in enumerate(impl_list):
        means = [np.mean(results[impl][dim]) for dim in dims]
        stds = [np.std(results[impl][dim]) for dim in dims]
        plt.bar(x + i*width, means, width, label=impl, yerr=stds, capsize=5, color=colors[i % len(colors)])
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.xticks(x + width*(len(impl_list)-1)/2, dims)
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # 2. CPU使用率の比較
    plt.subplot(rows, cols, 2)
    for i, impl in enumerate(impl_list):
        cpu_values = [detailed_results[impl][dim].cpu_usage if detailed_results[impl][dim] is not None else 0 for dim in dims]
        plt.plot(dims, cpu_values, 'o-', label=impl, color=colors[i % len(colors)], linewidth=2, markersize=6)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage Comparison')
    plt.grid(True)
    plt.legend()
    
    # 3. メモリ使用量の比較
    plt.subplot(rows, cols, 3)
    for i, impl in enumerate(impl_list):
        mem_values = [detailed_results[impl][dim].memory_usage if detailed_results[impl][dim] is not None else 0 for dim in dims]
        plt.plot(dims, mem_values, 'o-', label=impl, color=colors[i % len(colors)], linewidth=2, markersize=6)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    plt.grid(True)
    plt.legend()
    
    # 4. スレッド数の比較
    plt.subplot(rows, cols, 4)
    for i, impl in enumerate(impl_list):
        thread_values = [detailed_results[impl][dim].thread_count if detailed_results[impl][dim] is not None else 0 for dim in dims]
        plt.plot(dims, thread_values, 'o-', label=impl, color=colors[i % len(colors)], linewidth=2, markersize=6)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Thread Count')
    plt.title('Thread Count Comparison')
    plt.grid(True)
    plt.legend()
    
    # 5. 速度向上率の比較（Pythonが含まれている場合）
    if 'python' in selected_implementations:
        plt.subplot(rows, cols, 5)
        for impl in impl_list:
            if impl != 'python':
                speedup_values = [detailed_results[impl][dim].speedup_vs_python if detailed_results[impl][dim] is not None else 0 for dim in dims]
                plt.plot(dims, speedup_values, 'o-', label=impl, linewidth=2, markersize=6)
        
        plt.xlabel('Matrix Size')
        plt.ylabel('Speedup Ratio (vs Python)')
        plt.title('Speedup Comparison')
        plt.grid(True)
        plt.legend()
        plt.yscale('log')
    
    # 6. メモリ効率
    plt.subplot(rows, cols, 6)
    for i, impl in enumerate(impl_list):
        efficiency_values = []
        for dim in dims:
            if detailed_results[impl][dim] is not None:
                time_per_step = detailed_results[impl][dim].mean_time / 1000
                mem_per_step = detailed_results[impl][dim].memory_usage / 100
                efficiency_values.append(mem_per_step / time_per_step if time_per_step > 0 else 0)
            else:
                efficiency_values.append(0)
        plt.plot(dims, efficiency_values, 'o-', label=impl, color=colors[i % len(colors)], linewidth=2, markersize=6)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Memory/Time Ratio (MB/s)')
    plt.title('Memory Efficiency')
    plt.grid(True)
    plt.legend()
    
    # 7. 最大次元での統計情報
    plt.subplot(rows, cols, 7)
    max_dim = max(dims)
    stats_data = []
    labels = []
    
    for impl in impl_list:
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
    
    # 8. システム情報
    plt.subplot(rows, cols, 8)
    plt.axis('off')
    system_info = f"""
    System Information:
    CPU Cores: {multiprocessing.cpu_count()}
    Total Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB
    Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB
    
    Benchmark Parameters:
    Selected Implementations: {', '.join(impl_list)}
    Dimensions: {dims}
    Steps: {num_steps}
    Repeats: {num_repeats}
    """
    plt.text(0.1, 0.5, system_info, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    
    # ファイル名に選択された実装を含める
    impl_names = '_'.join(sorted(impl_list))
    plt.savefig(os.path.join(savepath, f'detailed_benchmark_{impl_names}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def save_detailed_benchmark_results(results, detailed_results, dims, selected_implementations, savepath):
    """詳細なベンチマーク結果をファイルに保存"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    impl_names = '_'.join(sorted(selected_implementations))
    
    # JSONファイルに保存
    json_filename = os.path.join(savepath, f'detailed_benchmark_{impl_names}_{timestamp}.json')
    
    # 結果を辞書形式に変換
    results_data = {
        'timestamp': timestamp,
        'selected_implementations': list(selected_implementations),
        'dimensions': dims,
        'system_info': {
            'cpu_cores': multiprocessing.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3)
        },
        'detailed_results': {}
    }
    
    # 詳細結果を変換
    for impl in selected_implementations:
        results_data['detailed_results'][impl] = {}
        for dim in dims:
            if detailed_results[impl][dim]:
                results_data['detailed_results'][impl][dim] = detailed_results[impl][dim].to_dict()
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    # CSVファイルに保存
    csv_filename = os.path.join(savepath, f'detailed_benchmark_{impl_names}_{timestamp}.csv')
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
            for impl in selected_implementations:
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

def print_detailed_summary(results, detailed_results, dims, selected_implementations):
    """詳細な結果のサマリーを表示"""
    print("\n" + "="*80)
    print("詳細ベンチマーク結果サマリー")
    print("="*80)
    
    impl_list = list(selected_implementations)
    
    # システム情報
    print(f"\nシステム情報:")
    print(f"  CPU コア数: {multiprocessing.cpu_count()}")
    print(f"  総メモリ: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"  利用可能メモリ: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"  選択された実装: {', '.join(impl_list)}")
    
    # 各次元での最速実装
    print("\n各次元での最速実装:")
    for dim in dims:
        best_impl = None
        best_time = float('inf')
        for impl in impl_list:
            if detailed_results[impl][dim]:
                if detailed_results[impl][dim].mean_time < best_time:
                    best_time = detailed_results[impl][dim].mean_time
                    best_impl = impl
        
        if best_impl:
            print(f"  次元 {dim}: {best_impl} ({best_time:.6f}秒)")
    
    # 平均性能指標
    print("\n平均性能指標（全次元）:")
    for impl in impl_list:
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
    for impl in impl_list:
        if detailed_results[impl][max_dim]:
            result = detailed_results[impl][max_dim]
            print(f"  {impl}:")
            print(f"    実行時間: {result.mean_time:.6f}秒")
            print(f"    CPU使用率: {result.cpu_usage:.1f}%")
            print(f"    メモリ使用量: {result.memory_usage:.1f} MB")
            print(f"    ピークメモリ: {result.memory_peak:.1f} MB")
            print(f"    スレッド数: {result.thread_count}")
            if 'python' in selected_implementations and impl != 'python':
                print(f"    速度向上率: {result.speedup_vs_python:.2f}倍")

def main():
    # 利用可能な実装
    available_implementations = {'python', 'numba', 'eigen', 'eigen_direct_csr', 'suitesparse'}
    
    # 比較する実装を選択（ここで変更可能）
    # selected_implementations = {'python', 'numba', 'eigen', 'eigen_direct_csr', 'suitesparse'}  # 全実装
    # selected_implementations = {'python', 'eigen', 'eigen_direct_csr'}  # PythonとEigen実装の比較
    selected_implementations = {'python', 'eigen', 'eigen_direct_csr', 'suitesparse'}  # 最適化効果の確認
    
    # 選択された実装の妥当性チェック
    invalid_impls = selected_implementations - available_implementations
    if invalid_impls:
        print(f"エラー: 無効な実装が指定されました: {invalid_impls}")
        print(f"利用可能な実装: {available_implementations}")
        return
    
    # テストする行列サイズ
    dims = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
    num_repeats = 10  # 各サイズでの繰り返し回数
    num_steps = 1000  # 時間発展のステップ数
    
    print("詳細ベンチマーク開始")
    print(f"- 行列サイズ: {dims}")
    print(f"- 繰り返し回数: {num_repeats}")
    print(f"- 時間発展ステップ数: {num_steps}")
    print(f"- 選択された実装: {', '.join(sorted(selected_implementations))}")
    print(f"- 追加メトリクス: CPU使用率, メモリ使用量, スレッド数")
    print(f"- 新機能: Phase 1-2最適化（データ変換削減 + 階層的並列化）")
    
    results, detailed_results = run_detailed_benchmark(dims, selected_implementations, num_repeats, num_steps)
    
    # 結果のサマリーを表示
    print_detailed_summary(results, detailed_results, dims, selected_implementations)
    
    # 結果をプロット
    print("\n=== Plotting Detailed Results ===")
    plot_detailed_results(results, detailed_results, dims, selected_implementations, num_steps, num_repeats)
    
    # 結果をファイルに保存
    print("\n=== Saving Detailed Results ===")
    save_detailed_benchmark_results(results, detailed_results, dims, selected_implementations, savepath)
    
    impl_names = '_'.join(sorted(selected_implementations))
    print(f"\n詳細ベンチマーク完了")
    print(f"結果は{os.path.join(savepath, f'detailed_benchmark_{impl_names}.png')}に保存されました")

if __name__ == "__main__":
    main() 