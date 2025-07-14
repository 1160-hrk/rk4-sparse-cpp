import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from python import rk4_sparse
import time
import cProfile
import pstats
from line_profiler import LineProfiler
import tracemalloc
import psutil
from typing import List, Tuple, Callable, Any, Optional, Union
import multiprocessing
from dataclasses import dataclass
from datetime import datetime

@dataclass
class ProfilingResult:
    """プロファイリング結果を格納するデータクラス"""
    execution_time: float
    cpu_usage: float
    memory_usage: float
    thread_count: int
    memory_peak: float
    function_stats: dict
    matrix_stats: dict  # 行列演算の統計情報を追加
    timestamp: datetime = datetime.now()

class PerformanceProfiler:
    """性能プロファイリングを行うクラス"""
    
    def __init__(self):
        self.process = psutil.Process()
        self.results = []
    
    def profile_execution(self, func: Callable, *args, **kwargs) -> ProfilingResult:
        """関数の実行をプロファイリング"""
        # メモリトラッキング開始
        tracemalloc.start()
        
        # CPU使用率の初期値を複数回測定して平均を取る
        initial_cpu_samples = [self.process.cpu_percent() for _ in range(5)]
        initial_cpu = sum(initial_cpu_samples) / len(initial_cpu_samples)
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # 行列演算の統計情報を収集
        H0, mux, muy = args[:3]
        matrix_stats = {
            'H0_nnz': H0.nnz,
            'H0_density': H0.nnz / (H0.shape[0] * H0.shape[1]),
            'mux_nnz': mux.nnz,
            'mux_density': mux.nnz / (mux.shape[0] * mux.shape[1]),
            'muy_nnz': muy.nnz,
            'muy_density': muy.nnz / (muy.shape[0] * muy.shape[1]),
            'dimension': H0.shape[0]
        }
        
        # 関数の実行時間を詳細に計測
        detailed_times = []
        start_time = time.perf_counter_ns()  # ナノ秒単位で計測
        
        result = func(*args, **kwargs)
        
        end_time = time.perf_counter_ns()
        execution_time = (end_time - start_time) / 1e9  # 秒に変換
        
        # メモリ使用量の計測
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # CPU使用率を複数回測定して平均を取る
        cpu_samples = [self.process.cpu_percent() for _ in range(5)]
        cpu_usage = sum(cpu_samples) / len(cpu_samples)
        memory_usage = self.process.memory_info().rss / 1024 / 1024  # MB
        thread_count = self.process.num_threads()
        
        # 関数統計の取得
        stats = {
            func.__name__: {
                'total_time': execution_time,
                'time_per_step': execution_time / (len(args[4]) - 1),  # Ex配列の長さからステップ数を推定
                'hits': 1,
                'average': execution_time
            }
        }
        
        # 結果を作成
        profile_result = ProfilingResult(
            execution_time=execution_time,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage - initial_memory,
            thread_count=thread_count,
            memory_peak=peak / 1024 / 1024,
            function_stats=stats,
            matrix_stats=matrix_stats
        )
        
        self.results.append(profile_result)
        return profile_result

def plot_performance_metrics(
    profiler: PerformanceProfiler,
    steps_list: List[int],
    save_dir: str
):
    """性能メトリクスをプロット"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 実行時間のプロット
    plt.figure(figsize=(10, 6))
    times = [r.execution_time for r in profiler.results]
    plt.plot(steps_list, times, 'b-', marker='o')
    plt.xlabel('Number of Steps')
    plt.ylabel('Execution Time [sec]')
    plt.title('Total Execution Time vs Steps')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'execution_time_{timestamp}.png'))
    plt.close()
    
    # ステップあたりの実行時間
    plt.figure(figsize=(10, 6))
    times_per_step = [r.function_stats['rk4_cpu_sparse']['time_per_step'] for r in profiler.results]
    plt.plot(steps_list, times_per_step, 'r-', marker='o')
    plt.xlabel('Number of Steps')
    plt.ylabel('Time per Step [sec]')
    plt.title('Time per Step vs Total Steps')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'time_per_step_{timestamp}.png'))
    plt.close()
    
    # CPU使用率のプロット
    plt.figure(figsize=(10, 6))
    cpu_usage = [r.cpu_usage for r in profiler.results]
    plt.plot(steps_list, cpu_usage, 'r-', marker='o')
    plt.xlabel('Number of Steps')
    plt.ylabel('CPU Usage [%]')
    plt.title('CPU Usage vs Steps')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'cpu_usage_{timestamp}.png'))
    plt.close()
    
    # メモリ使用量のプロット
    plt.figure(figsize=(10, 6))
    memory_usage = [r.memory_usage for r in profiler.results]
    plt.plot(steps_list, memory_usage, 'g-', marker='o')
    plt.xlabel('Number of Steps')
    plt.ylabel('Memory Usage [MB]')
    plt.title('Memory Usage vs Steps')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'memory_usage_{timestamp}.png'))
    plt.close()

def print_system_info():
    """システム情報を表示"""
    print("\n=== System Information ===")
    print(f"CPU Cores: {multiprocessing.cpu_count()}")
    print(f"Total Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB")
    print(f"Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB")
    
    # OpenMPの設定を確認
    print("\nOpenMP Configuration:")
    try:
        import os
        print(f"OMP_NUM_THREADS: {os.environ.get('OMP_NUM_THREADS', 'Not set')}")
        print(f"OMP_SCHEDULE: {os.environ.get('OMP_SCHEDULE', 'Not set')}")
        print(f"OMP_DYNAMIC: {os.environ.get('OMP_DYNAMIC', 'Not set')}")
    except Exception as e:
        print(f"Could not get OpenMP settings: {e}")
    
    # CPUの詳細情報
    try:
        import cpuinfo
        info = cpuinfo.get_cpu_info()
        print(f"\nCPU Information:")
        print(f"Brand: {info.get('brand_raw', 'Unknown')}")
        print(f"Architecture: {info.get('arch', 'Unknown')}")
        print(f"Flags: {', '.join(info.get('flags', []))[:100]}...")
    except ImportError:
        print("\nCPU Information: py-cpuinfo not available")
    
    print("========================\n")

def create_test_matrices(size: int = 2) -> Tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix]:
    """テスト用の行列を生成"""
    # 基底ハミルトニアン
    H0_data = np.array([1.0 + 0.0j])
    H0_indices = np.array([0])
    H0_indptr = np.array([0, 1, 1])
    H0 = sp.csc_matrix((H0_data, H0_indices, H0_indptr), shape=(size, size))
    
    # 双極子モーメント行列 (x方向)
    mux_data = np.array([0.1 + 0.0j, 0.1 + 0.0j])
    mux_indices = np.array([1, 0])
    mux_indptr = np.array([0, 1, 2])
    mux = sp.csc_matrix((mux_data, mux_indices, mux_indptr), shape=(size, size))
    
    # 双極子モーメント行列 (y方向)
    muy = sp.csc_matrix((size, size), dtype=np.complex128)
    
    return H0, mux, muy

def create_test_pulse(num_steps: int = 500) -> Tuple[np.ndarray, np.ndarray]:
    """テストパルスを生成"""
    t = np.linspace(0, 0.1, num_steps)
    Ex = 0.1 * np.sin(50.0 * t)
    Ey = np.zeros_like(Ex)
    return Ex, Ey

def run_profile(
    H0: sp.csc_matrix,
    mux: sp.csc_matrix,
    muy: sp.csc_matrix,
    psi0: np.ndarray,
    Ex: np.ndarray,
    Ey: np.ndarray,
    dt: float,
    steps: int,
    stride: int = 1,
    renorm: bool = False
) -> ProfilingResult:
    """プロファイリングを実行"""
    profiler = PerformanceProfiler()
    return profiler.profile_execution(
        rk4_cpu_sparse,
        H0, mux, muy,
        psi0.flatten(),
        Ex,
        Ey,
        dt,
        True,  # return_traj
        stride,
        renorm
    )

def main():
    """Main function"""
    print("Starting performance profiling...")
    
    # システム情報の表示
    print_system_info()
    
    # 出力ディレクトリの準備
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("examples", "figures", f"profile_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # テストケースの準備
    H0, mux, muy = create_test_matrices(2)
    psi0 = np.array([[1.0 + 0.0j], [0.0 + 0.0j]], dtype=np.complex128)
    dt = 0.02
    stride = 1
    
    # 異なるステップ数でプロファイリングを実行
    steps_list = [100, 200, 500, 1000, 2000, 5000]  # より大きなステップ数も試す
    profiler = PerformanceProfiler()
    
    print("\n=== Performance Metrics ===")
    print(f"{'Steps':>8} | {'Time [s]':>10} | {'Time/Step [µs]':>14} | {'CPU [%]':>8} | {'Mem [MB]':>8} | {'Threads':>7}")
    print("-" * 80)
    
    for steps in steps_list:
        Ex, Ey = create_test_pulse(steps)
        result = run_profile(
            H0, mux, muy, psi0, Ex, Ey, dt, steps - 1, stride, False
        )
        
        time_per_step = result.function_stats['rk4_cpu_sparse']['time_per_step'] * 1e6  # マイクロ秒に変換
        print(f"{steps:8d} | {result.execution_time:10.6f} | {time_per_step:14.2f} | "
              f"{result.cpu_usage:8.1f} | {result.memory_usage:8.1f} | {result.thread_count:7d}")
        
        profiler.results.append(result)
    
    # 行列情報の表示
    print("\n=== Matrix Information ===")
    matrix_stats = profiler.results[0].matrix_stats
    print(f"Matrix dimension: {matrix_stats['dimension']}x{matrix_stats['dimension']}")
    print(f"H0  - Non-zeros: {matrix_stats['H0_nnz']}, Density: {matrix_stats['H0_density']*100:.2f}%")
    print(f"mux - Non-zeros: {matrix_stats['mux_nnz']}, Density: {matrix_stats['mux_density']*100:.2f}%")
    print(f"muy - Non-zeros: {matrix_stats['muy_nnz']}, Density: {matrix_stats['muy_density']*100:.2f}%")
    
    # 結果をプロット
    plot_performance_metrics(profiler, steps_list, output_dir)
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
