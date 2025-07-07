"""C++実装のプロファイリングを行うモジュール"""

import os
import sys
from datetime import datetime
import numpy as np
from scipy import sparse

# プロジェクトルートへのパスを追加
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from cpp.rk4_sparse_cpp import rk4_cpu_sparse as rk4_cpu_sparse_cpp
from python.utils import create_test_matrices, create_test_pulse
from profile_common import (
    PerformanceProfiler,
    print_system_info,
    print_comparison_results,
    plot_comparison_metrics
)

def run_cpp_profile(
    H0: sparse.csr_matrix,
    mux: sparse.csr_matrix,
    muy: sparse.csr_matrix,
    psi0: np.ndarray,
    Ex: np.ndarray,
    Ey: np.ndarray,
    dt: float,
    steps: int,
    stride: int,
    verbose: bool = False
) -> np.ndarray:
    """C++実装のプロファイリングを実行"""
    return rk4_cpu_sparse_cpp(H0, mux, muy, psi0, Ex, Ey, dt, steps, stride, verbose)

def main():
    """Main function for C++ profiling"""
    print("Starting C++ implementation profiling...")
    
    # システム情報の表示
    print_system_info()
    
    # 出力ディレクトリの準備
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join("examples", "figures", f"profile_results_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    
    # テストケースの準備
    H0, mux, muy = create_test_matrices(2)  # 2x2行列で開始
    psi0 = np.array([[1.0 + 0.0j], [0.0 + 0.0j]], dtype=np.complex128)
    dt = 0.02
    stride = 1
    
    # 異なるステップ数でプロファイリングを実行
    steps_list = [100, 200, 500, 1000, 2000, 5000]
    profiler = PerformanceProfiler(implementation='cpp')
    
    print("\n=== C++ Performance Metrics ===")
    print(f"{'Steps':>8} | {'Time [ms]':>10} | {'Time/Step [µs]':>14} | {'CPU [%]':>8} | {'Mem [MB]':>8} | {'Threads':>7}")
    print("-" * 80)
    
    for steps in steps_list:
        Ex, Ey = create_test_pulse(steps)
        result = profiler.profile_execution(
            run_cpp_profile,
            H0, mux, muy, psi0, Ex, Ey, dt, steps - 1, stride, False
        )
        
        time_per_step = result.function_stats['run_cpp_profile']['time_per_step'] * 1e6  # マイクロ秒に変換
        print(f"{steps:8d} | {result.execution_time*1000:10.3f} | {time_per_step:14.2f} | "
              f"{result.cpu_usage:8.1f} | {result.memory_usage:8.1f} | {result.thread_count:7d}")
    
    # 行列情報の表示
    print("\n=== Matrix Information ===")
    matrix_stats = profiler.results[0].matrix_stats
    print(f"Matrix dimension: {matrix_stats['dimension']}x{matrix_stats['dimension']}")
    print(f"H0  - Non-zeros: {matrix_stats['H0_nnz']}, Density: {matrix_stats['H0_density']*100:.2f}%")
    print(f"mux - Non-zeros: {matrix_stats['mux_nnz']}, Density: {matrix_stats['mux_density']*100:.2f}%")
    print(f"muy - Non-zeros: {matrix_stats['muy_nnz']}, Density: {matrix_stats['muy_density']*100:.2f}%")
    
    print(f"\nResults saved to: {output_dir}")
    return profiler, steps_list, output_dir

if __name__ == "__main__":
    main() 