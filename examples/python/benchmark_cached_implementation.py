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
    from rk4_sparse import rk4_sparse_py, rk4_sparse_eigen, rk4_sparse_eigen_cached
    print("Info: All implementations imported successfully")
except ImportError as e:
    print(f"Warning: Could not import rk4_sparse: {e}")
    print("Please make sure the module is built and installed correctly.")
    # ダミー関数を作成してエラーを回避
    def dummy_function(*args, **kwargs):
        raise NotImplementedError("rk4_sparse module not available")
    rk4_sparse_py = dummy_function
    rk4_sparse_eigen = dummy_function
    rk4_sparse_eigen_cached = dummy_function

savepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(savepath, exist_ok=True)

@dataclass
class BenchmarkResult:
    """ベンチマーク結果を格納するデータクラス"""
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
    timestamp: datetime = datetime.now()
    
    def to_dict(self) -> dict:
        """結果を辞書形式に変換（JSON保存用）"""
        result_dict = asdict(self)
        result_dict['timestamp'] = self.timestamp.isoformat()
        return result_dict

class PerformanceProfiler:
    """性能プロファイリングを行うクラス"""
    def __init__(self):
        self.process = psutil.Process()
    
    def profile_execution(self, func, *args, **kwargs) -> BenchmarkResult:
        """関数の実行をプロファイリング"""
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
        
        return BenchmarkResult(
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
        'eigen': rk4_sparse_eigen,
        'eigen_cached': rk4_sparse_eigen_cached
    }
    return implementation_map.get(impl_name)

def run_benchmark(dims, num_repeats=10, num_steps=1000):
    """3つの実装でベンチマークを実行"""
    implementations = ['python', 'eigen', 'eigen_cached']
    results = {impl: {dim: [] for dim in dims} for impl in implementations}
    detailed_results: Dict[str, Dict[int, Optional[BenchmarkResult]]] = {
        impl: {dim: None for dim in dims} for impl in implementations
    }
    
    profiler = PerformanceProfiler()

    for dim in dims:
        print(f"\n次元数: {dim}")
        
        # テストシステムの生成
        H0, mux, muy, Ex, Ey, psi0, dt_E = create_test_system(dim)
        
        # 各実装のベンチマーク実行
        for impl in implementations:
            print(f"{impl}実装の実行中...")
            
            try:
                # 実装関数を取得
                impl_func = get_implementation_function(impl)
                if impl_func is None:
                    print(f"  警告: {impl}実装が見つかりません")
                    continue
                
                # 引数を準備（全実装で同じ形式）
                args = (H0, mux, muy, Ex, Ey, psi0, dt_E*2, True, 1, False)
                
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
        python_mean = np.mean(results['python'][dim])
        for impl in implementations:
            if impl != 'python' and detailed_results[impl][dim] is not None:
                impl_mean = np.mean(results[impl][dim])
                if impl_mean > 0:
                    speedup = float(python_mean / impl_mean)
                    detailed_results[impl][dim].speedup_vs_python = speedup
                else:
                    detailed_results[impl][dim].speedup_vs_python = 0.0
        
        print(f"速度向上率（Python基準）:")
        for impl in implementations:
            if impl != 'python' and detailed_results[impl][dim] is not None:
                result = detailed_results[impl][dim]
                if result is not None:
                    print(f"  {impl}: {result.speedup_vs_python:.2f}倍")
        
    return results, detailed_results

def plot_results(results, detailed_results, dims, num_steps=1000, num_repeats=10):
    """結果をプロット"""
    implementations = ['python', 'eigen', 'eigen_cached']
    colors = ['blue', 'red', 'green']
    
    plt.figure(figsize=(15, 10))
    
    # 1. 実行時間の比較
    plt.subplot(2, 3, 1)
    x = np.arange(len(dims))
    width = 0.25
    
    for i, impl in enumerate(implementations):
        means = [np.mean(results[impl][dim]) for dim in dims]
        stds = [np.std(results[impl][dim]) for dim in dims]
        plt.bar(x + i*width, means, width, label=impl, yerr=stds, capsize=5, color=colors[i])
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Execution Time Comparison')
    plt.xticks(x + width, dims)
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # 2. CPU使用率の比較
    plt.subplot(2, 3, 2)
    for i, impl in enumerate(implementations):
        cpu_values = [detailed_results[impl][dim].cpu_usage if detailed_results[impl][dim] is not None else 0 for dim in dims]
        plt.plot(dims, cpu_values, 'o-', label=impl, color=colors[i], linewidth=2, markersize=6)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('CPU Usage (%)')
    plt.title('CPU Usage Comparison')
    plt.grid(True)
    plt.legend()
    
    # 3. メモリ使用量の比較
    plt.subplot(2, 3, 3)
    for i, impl in enumerate(implementations):
        mem_values = [detailed_results[impl][dim].memory_usage if detailed_results[impl][dim] is not None else 0 for dim in dims]
        plt.plot(dims, mem_values, 'o-', label=impl, color=colors[i], linewidth=2, markersize=6)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Memory Usage (MB)')
    plt.title('Memory Usage Comparison')
    plt.grid(True)
    plt.legend()
    
    # 4. 速度向上率の比較
    plt.subplot(2, 3, 4)
    for impl in implementations:
        if impl != 'python':
            speedup_values = [detailed_results[impl][dim].speedup_vs_python if detailed_results[impl][dim] is not None else 0 for dim in dims]
            plt.plot(dims, speedup_values, 'o-', label=impl, linewidth=2, markersize=6)
    
    plt.xlabel('Matrix Size')
    plt.ylabel('Speedup Ratio (vs Python)')
    plt.title('Speedup Comparison')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')
    
    # 5. 最大次元での統計情報
    plt.subplot(2, 3, 5)
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
    
    # 6. システム情報
    plt.subplot(2, 3, 6)
    plt.axis('off')
    system_info = f"""
    System Information:
    CPU Cores: {multiprocessing.cpu_count()}
    Total Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB
    Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB
    
    Benchmark Parameters:
    Implementations: {', '.join(implementations)}
    Dimensions: {dims}
    Steps: {num_steps}
    Repeats: {num_repeats}
    """
    plt.text(0.1, 0.5, system_info, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='center')
    
    plt.tight_layout()
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(savepath, f'cached_benchmark_{timestamp}.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()

def save_benchmark_results(results, detailed_results, dims, savepath):
    """ベンチマーク結果をファイルに保存"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    implementations = ['python', 'eigen', 'eigen_cached']
    
    # JSONファイルに保存
    json_filename = os.path.join(savepath, f'cached_benchmark_{timestamp}.json')
    
    # 結果を辞書形式に変換
    results_data = {
        'timestamp': timestamp,
        'implementations': implementations,
        'dimensions': dims,
        'system_info': {
            'cpu_cores': multiprocessing.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3)
        },
        'detailed_results': {}
    }
    
    # 詳細結果を変換
    for impl in implementations:
        results_data['detailed_results'][impl] = {}
        for dim in dims:
            if detailed_results[impl][dim]:
                results_data['detailed_results'][impl][dim] = detailed_results[impl][dim].to_dict()
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    # CSVファイルに保存
    csv_filename = os.path.join(savepath, f'cached_benchmark_{timestamp}.csv')
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
            for impl in implementations:
                if detailed_results[impl][dim]:
                    result = detailed_results[impl][dim]
                    writer.writerow([
                        dim, impl, result.mean_time, result.std_time,
                        result.min_time, result.max_time, result.speedup_vs_python,
                        result.cpu_usage, result.memory_usage, result.memory_peak, result.thread_count
                    ])
    
    print(f"Results saved to:")
    print(f"  JSON: {json_filename}")
    print(f"  CSV:  {csv_filename}")
    
    return json_filename, csv_filename

def print_summary(results, detailed_results, dims):
    """結果のサマリーを表示"""
    print("\n" + "="*80)
    print("キャッシュ化実装のベンチマーク結果サマリー")
    print("="*80)
    
    implementations = ['python', 'eigen', 'eigen_cached']
    
    # システム情報
    print(f"\nシステム情報:")
    print(f"  CPU コア数: {multiprocessing.cpu_count()}")
    print(f"  総メモリ: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"  利用可能メモリ: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    print(f"  比較実装: {', '.join(implementations)}")
    
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
            if impl != 'python':
                print(f"    速度向上率: {result.speedup_vs_python:.2f}倍")
    
    # キャッシュ化の効果
    print(f"\nキャッシュ化の効果（eigen vs eigen_cached）:")
    for dim in dims:
        if (detailed_results['eigen'][dim] and 
            detailed_results['eigen_cached'][dim]):
            eigen_time = detailed_results['eigen'][dim].mean_time
            cached_time = detailed_results['eigen_cached'][dim].mean_time
            if eigen_time > 0:
                improvement = (eigen_time - cached_time) / eigen_time * 100
                print(f"  次元 {dim}: {improvement:.1f}% 高速化")

def main():
    # テストする行列サイズ
    dims = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    num_repeats = 10  # 各サイズでの繰り返し回数
    num_steps = 1000  # 時間発展のステップ数
    
    print("キャッシュ化実装のベンチマーク開始")
    print(f"- 行列サイズ: {dims}")
    print(f"- 繰り返し回数: {num_repeats}")
    print(f"- 時間発展ステップ数: {num_steps}")
    print(f"- 比較実装: python, eigen, eigen_cached")
    print(f"- 新機能: パターン構築・データ展開のキャッシュ化")
    
    results, detailed_results = run_benchmark(dims, num_repeats, num_steps)
    
    # 結果のサマリーを表示
    print_summary(results, detailed_results, dims)
    
    # 結果をプロット
    print("\n=== Plotting Results ===")
    plot_results(results, detailed_results, dims, num_steps, num_repeats)
    
    # 結果をファイルに保存
    print("\n=== Saving Results ===")
    save_benchmark_results(results, detailed_results, dims, savepath)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"\nベンチマーク完了")
    print(f"結果は{os.path.join(savepath, f'cached_benchmark_{timestamp}.png')}に保存されました")

if __name__ == "__main__":
    main() 