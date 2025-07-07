import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from python import rk4_cpu_sparse
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
        
        # CPU使用率の初期値
        initial_cpu = self.process.cpu_percent()
        initial_memory = self.process.memory_info().rss / 1024 / 1024  # MB
        
        # 関数の実行時間を計測
        start_time = time.time()
        result = func(*args, **kwargs)
        execution_time = time.time() - start_time
        
        # メモリ使用量の計測
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        
        # CPU使用率と統計情報の収集
        cpu_usage = self.process.cpu_percent()
        memory_usage = self.process.memory_info().rss / 1024 / 1024  # MB
        thread_count = self.process.num_threads()
        
        # 関数統計の取得
        stats = {
            func.__name__: {
                'total_time': execution_time,
                'hits': 1,
                'average': execution_time
            }
        }
        
        # 結果を作成
        profile_result = ProfilingResult(
            execution_time=execution_time,
            cpu_usage=cpu_usage,
            memory_usage=memory_usage - initial_memory,  # 差分を記録
            thread_count=thread_count,
            memory_peak=peak / 1024 / 1024,  # MB
            function_stats=stats
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
    plt.title('Execution Time vs Steps')
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'execution_time_{timestamp}.png'))
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
    try:
        import torch
        print(f"PyTorch CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"GPU: {torch.cuda.get_device_name(0)}")
    except ImportError:
        print("PyTorch not available")
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
    steps_list = [100, 200, 500, 1000]
    profiler = PerformanceProfiler()
    
    print("\n=== Performance Metrics ===")
    print(f"{'Steps':>8} | {'Time [s]':>10} | {'CPU [%]':>8} | {'Mem [MB]':>8} | {'Threads':>7}")
    print("-" * 60)
    
    for steps in steps_list:
        Ex, Ey = create_test_pulse(steps)
        result = run_profile(
            H0, mux, muy, psi0, Ex, Ey, dt, steps - 1, stride, False
        )
        
        print(f"{steps:8d} | {result.execution_time:10.6f} | {result.cpu_usage:8.1f} | "
              f"{result.memory_usage:8.1f} | {result.thread_count:7d}")
        
        profiler.results.append(result)
    
    # 結果をプロット
    plot_performance_metrics(profiler, steps_list, output_dir)
    
    print(f"\nResults saved to: {output_dir}")

if __name__ == "__main__":
    main()
