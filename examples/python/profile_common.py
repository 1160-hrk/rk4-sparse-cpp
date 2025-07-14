"""共通のプロファイリング機能を提供するモジュール"""

import os
import time
import psutil
import tracemalloc
import multiprocessing
import json
import csv
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Callable, List, Dict
import matplotlib.pyplot as plt
import numpy as np

@dataclass
class ProfilingResult:
    """プロファイリング結果を格納するデータクラス"""
    execution_time: float
    cpu_usage: float
    memory_usage: float
    thread_count: int
    memory_peak: float
    function_stats: dict
    matrix_stats: dict
    implementation: str  # 実装の種類（'python' or 'cpp'）を追加
    timestamp: datetime = datetime.now()
    
    def to_dict(self) -> dict:
        """結果を辞書形式に変換（JSON保存用）"""
        result_dict = asdict(self)
        # datetimeオブジェクトを文字列に変換
        result_dict['timestamp'] = self.timestamp.isoformat()
        return result_dict

class PerformanceProfiler:
    """性能プロファイリングを行うクラス"""
    
    def __init__(self, implementation: str):
        self.process = psutil.Process()
        self.results = []
        self.implementation = implementation
    
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
            matrix_stats=matrix_stats,
            implementation=self.implementation
        )
        
        self.results.append(profile_result)
        return profile_result

def plot_comparison_metrics(
    python_profiler: PerformanceProfiler,
    cpp_profiler: PerformanceProfiler,
    steps_list: List[int],
    save_dir: str
):
    """Python実装とC++実装の性能メトリクスを比較プロット"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 実行時間の比較プロット
    plt.figure(figsize=(12, 6))
    py_times = [r.execution_time * 1000 for r in python_profiler.results]  # ミリ秒に変換
    cpp_times = [r.execution_time * 1000 for r in cpp_profiler.results]
    
    plt.plot(steps_list, py_times, 'b-', marker='o', label='Python')
    plt.plot(steps_list, cpp_times, 'r-', marker='s', label='C++')
    plt.xlabel('Number of Steps')
    plt.ylabel('Execution Time [ms]')
    plt.title('Total Execution Time Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'execution_time_comparison_{timestamp}.png'))
    plt.close()
    
    # ステップあたりの実行時間の比較
    plt.figure(figsize=(12, 6))
    py_times_per_step = [r.function_stats['run_python_profile']['time_per_step'] * 1e6 for r in python_profiler.results]
    cpp_times_per_step = [r.function_stats['run_cpp_profile']['time_per_step'] * 1e6 for r in cpp_profiler.results]
    
    plt.plot(steps_list, py_times_per_step, 'b-', marker='o', label='Python')
    plt.plot(steps_list, cpp_times_per_step, 'r-', marker='s', label='C++')
    plt.xlabel('Number of Steps')
    plt.ylabel('Time per Step [µs]')
    plt.title('Time per Step Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'time_per_step_comparison_{timestamp}.png'))
    plt.close()
    
    # CPU使用率の比較
    plt.figure(figsize=(12, 6))
    py_cpu = [r.cpu_usage for r in python_profiler.results]
    cpp_cpu = [r.cpu_usage for r in cpp_profiler.results]
    
    plt.plot(steps_list, py_cpu, 'b-', marker='o', label='Python')
    plt.plot(steps_list, cpp_cpu, 'r-', marker='s', label='C++')
    plt.xlabel('Number of Steps')
    plt.ylabel('CPU Usage [%]')
    plt.title('CPU Usage Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'cpu_usage_comparison_{timestamp}.png'))
    plt.close()
    
    # メモリ使用量の比較
    plt.figure(figsize=(12, 6))
    py_mem = [r.memory_usage for r in python_profiler.results]
    cpp_mem = [r.memory_usage for r in cpp_profiler.results]
    
    plt.plot(steps_list, py_mem, 'b-', marker='o', label='Python')
    plt.plot(steps_list, cpp_mem, 'r-', marker='s', label='C++')
    plt.xlabel('Number of Steps')
    plt.ylabel('Memory Usage [MB]')
    plt.title('Memory Usage Comparison')
    plt.legend()
    plt.grid(True)
    plt.savefig(os.path.join(save_dir, f'memory_usage_comparison_{timestamp}.png'))
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

def print_comparison_results(
    python_profiler: PerformanceProfiler,
    cpp_profiler: PerformanceProfiler,
    steps_list: List[int]
):
    """Python実装とC++実装の性能比較結果を表示"""
    print("\n=== Performance Comparison ===")
    print(f"{'Steps':>8} | {'Python [ms]':>12} | {'C++ [ms]':>12} | {'Speedup':>8} | {'Py µs/step':>10} | {'C++ µs/step':>10}")
    print("-" * 80)
    
    for i, steps in enumerate(steps_list):
        py_result = python_profiler.results[i]
        cpp_result = cpp_profiler.results[i]
        
        py_time = py_result.execution_time * 1000  # ミリ秒に変換
        cpp_time = cpp_result.execution_time * 1000
        speedup = py_time / cpp_time if cpp_time > 0 else float('inf')
        
        py_per_step = py_result.function_stats['run_python_profile']['time_per_step'] * 1e6  # マイクロ秒に変換
        cpp_per_step = cpp_result.function_stats['run_cpp_profile']['time_per_step'] * 1e6
        
        print(f"{steps:8d} | {py_time:12.3f} | {cpp_time:12.3f} | {speedup:8.2f}x | {py_per_step:10.2f} | {cpp_per_step:10.2f}")
    
    # 行列情報の表示（最初のケースのみ）
    print("\n=== Matrix Information ===")
    matrix_stats = python_profiler.results[0].matrix_stats
    print(f"Matrix dimension: {matrix_stats['dimension']}x{matrix_stats['dimension']}")
    print(f"H0  - Non-zeros: {matrix_stats['H0_nnz']}, Density: {matrix_stats['H0_density']*100:.2f}%")
    print(f"mux - Non-zeros: {matrix_stats['mux_nnz']}, Density: {matrix_stats['mux_density']*100:.2f}%")
    print(f"muy - Non-zeros: {matrix_stats['muy_nnz']}, Density: {matrix_stats['muy_density']*100:.2f}%") 

def save_profiling_results(
    profiler: PerformanceProfiler,
    steps_list: List[int],
    output_dir: str,
    implementation_name: str = ""
):
    """プロファイリング結果をファイルに保存"""
    if implementation_name == "":
        implementation_name = profiler.implementation
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # JSONファイルに保存
    json_filename = os.path.join(output_dir, f"{implementation_name}_profile_results_{timestamp}.json")
    results_data = {
        'implementation': implementation_name,
        'timestamp': timestamp,
        'system_info': {
            'cpu_cores': multiprocessing.cpu_count(),
            'total_memory_gb': psutil.virtual_memory().total / (1024**3),
            'available_memory_gb': psutil.virtual_memory().available / (1024**3)
        },
        'steps_list': steps_list,
        'results': [result.to_dict() for result in profiler.results]
    }
    
    with open(json_filename, 'w', encoding='utf-8') as f:
        json.dump(results_data, f, indent=2, ensure_ascii=False)
    
    # CSVファイルに保存
    csv_filename = os.path.join(output_dir, f"{implementation_name}_profile_results_{timestamp}.csv")
    with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # ヘッダー行
        writer.writerow([
            'Steps', 'Execution_Time_ms', 'CPU_Usage_percent', 'Memory_Usage_MB',
            'Thread_Count', 'Memory_Peak_MB', 'Time_Per_Step_us', 'Implementation'
        ])
        
        # データ行
        for i, result in enumerate(profiler.results):
            time_per_step = result.function_stats[f'run_{implementation_name}_profile']['time_per_step'] * 1e6
            writer.writerow([
                steps_list[i],
                result.execution_time * 1000,  # ミリ秒に変換
                result.cpu_usage,
                result.memory_usage,
                result.thread_count,
                result.memory_peak,
                time_per_step,
                implementation_name
            ])
    
    print(f"Results saved to:")
    print(f"  JSON: {json_filename}")
    print(f"  CSV:  {csv_filename}")
    
    return json_filename, csv_filename

def save_comparison_results(
    python_profiler: PerformanceProfiler,
    cpp_profiler: PerformanceProfiler,
    steps_list: List[int],
    output_dir: str
):
    """比較結果をファイルに保存"""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 比較結果のCSVファイル
    comparison_filename = os.path.join(output_dir, f"comparison_results_{timestamp}.csv")
    with open(comparison_filename, 'w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        # ヘッダー行
        writer.writerow([
            'Steps', 'Python_Time_ms', 'Cpp_Time_ms', 'Speedup',
            'Python_Time_Per_Step_us', 'Cpp_Time_Per_Step_us',
            'Python_CPU_percent', 'Cpp_CPU_percent',
            'Python_Memory_MB', 'Cpp_Memory_MB'
        ])
        
        # データ行
        for i, steps in enumerate(steps_list):
            py_result = python_profiler.results[i]
            cpp_result = cpp_profiler.results[i]
            
            py_time = py_result.execution_time * 1000
            cpp_time = cpp_result.execution_time * 1000
            speedup = py_time / cpp_time if cpp_time > 0 else float('inf')
            
            py_per_step = py_result.function_stats['run_python_profile']['time_per_step'] * 1e6
            cpp_per_step = cpp_result.function_stats['run_cpp_profile']['time_per_step'] * 1e6
            
            writer.writerow([
                steps, py_time, cpp_time, speedup,
                py_per_step, cpp_per_step,
                py_result.cpu_usage, cpp_result.cpu_usage,
                py_result.memory_usage, cpp_result.memory_usage
            ])
    
    print(f"Comparison results saved to: {comparison_filename}")
    return comparison_filename 