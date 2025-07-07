import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import scipy.sparse as sp
import matplotlib.pyplot as plt
from python import rk4_propagate, rk4_propagate_v2, RK4Config
import time
import cProfile
import pstats
from line_profiler import LineProfiler
import tracemalloc
import psutil
from typing import List, Tuple, Callable, Any, Optional, Union

def measure_time(func: Callable, *args, **kwargs) -> float:
    """関数の実行時間を計測する"""
    start_time = time.time()
    func(*args, **kwargs)
    end_time = time.time()
    return end_time - start_time

def measure_detailed_time(func: Callable, *args, **kwargs) -> Tuple[float, List[float]]:
    """関数の実行時間と各ステップの時間を計測する"""
    start_time = time.time()
    result = func(*args, **kwargs)
    end_time = time.time()
    total_time = end_time - start_time
    
    # 結果から時間パターンを抽出
    time_pattern = []
    if isinstance(result, list):
        for i in range(len(result) - 1):
            step_time = time.time()
            time_pattern.append(step_time - start_time)
    
    return total_time, time_pattern

def analyze_memory_pattern(func, *args, **kwargs):
    """
    メモリアクセスパターンを分析
    """
    tracemalloc.start()
    
    # 実行前のメモリスナップショット
    snapshot1 = tracemalloc.take_snapshot()
    
    # 関数実行
    result = func(*args, **kwargs)
    
    # 実行後のメモリスナップショット
    snapshot2 = tracemalloc.take_snapshot()
    
    # スナップショットの差分を分析
    top_stats = snapshot2.compare_to(snapshot1, 'lineno')
    
    print(f"\n{func.__name__}のメモリアクセス分析:")
    for stat in top_stats[:10]:
        print(f"{stat.count:>10}: {stat.size/1024:.1f} KB")
        print(f"    {stat.traceback.format()[0]}")
    
    tracemalloc.stop()
    return result

def analyze_cpu_usage(func, *args, **kwargs):
    """
    CPU使用率を分析
    """
    process = psutil.Process()
    
    # 実行前のCPU時間
    cpu_percent = process.cpu_percent()
    
    # 関数実行
    start = time.perf_counter()
    result = func(*args, **kwargs)
    duration = time.perf_counter() - start
    
    # CPU使用率の計算
    cpu_percent = process.cpu_percent()
    
    print(f"\n{func.__name__}のCPU使用率分析:")
    print(f"CPU使用率: {cpu_percent:.1f}%")
    print(f"実行時間: {duration:.6f}秒")
    print(f"スレッド数: {process.num_threads()}")
    
    return result

def analyze_openmp_info():
    """
    OpenMPの設定と使用状況を分析
    """
    print("\n=== OpenMP情報 ===")
    try:
        import multiprocessing
        print(f"利用可能なCPUコア数: {multiprocessing.cpu_count()}")
        process = psutil.Process()
        print(f"現在のCPU使用率: {process.cpu_percent()}%")
        print(f"現在のスレッド数: {process.num_threads()}")
    except Exception as e:
        print(f"OpenMP情報を取得できません: {e}")

def detailed_profile(H0, mux, muy, psi0, Ex3, Ey3, dt, steps, stride):
    """
    特定のケースでの詳細なプロファイリングを実行
    """
    print("\n=== 詳細なプロファイリング ===")
    print(f"ステップ数: {steps}")
    print(f"行列サイズ: {H0.shape}")
    print(f"非ゼロ要素数: {H0.nnz}")
    
    # OpenMP情報の分析
    analyze_openmp_info()
    
    # メモリアクセスパターンの分析
    print("\n=== メモリアクセスパターンの分析 ===")
    analyze_memory_pattern(
        rk4_propagate,
        H0, mux, muy, psi0, Ex3, Ey3, dt, steps, stride, False, True
    )
    
    config = RK4Config()
    config.use_simd = True
    config.collect_metrics = True
    config.use_threading = True
    
    analyze_memory_pattern(
        rk4_propagate_v2,
        H0, mux, muy, psi0, Ex3, Ey3, dt, steps, stride, config
    )
    
    # CPU使用率の分析
    print("\n=== CPU使用率の分析 ===")
    analyze_cpu_usage(
        rk4_propagate,
        H0, mux, muy, psi0, Ex3, Ey3, dt, steps, stride, False, True
    )
    
    analyze_cpu_usage(
        rk4_propagate_v2,
        H0, mux, muy, psi0, Ex3, Ey3, dt, steps, stride, config
    )

def create_test_matrices(size: int = 2) -> Tuple[sp.csc_matrix, sp.csc_matrix, sp.csc_matrix]:
    """テスト用の行列を生成する"""
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
    """テストパルスを生成する"""
    t = np.linspace(0, 0.1, num_steps)
    Ex = 0.1 * np.sin(50.0 * t)
    Ey = np.zeros_like(Ex)
    return Ex, Ey

def run_benchmark(
    H0: sp.csc_matrix,
    mux: sp.csc_matrix,
    muy: sp.csc_matrix,
    psi0: np.ndarray,
    Ex: np.ndarray,
    Ey: np.ndarray,
    dt: float,
    steps: int,
    stride: int = 1,
    normalize: bool = False,
    collect_metrics: bool = True
) -> Tuple[Tuple[float, List[float]], Tuple[float, List[float]]]:
    """ベンチマークを実行する"""
    
    # 安定版の実行時間を計測
    timing_stable = measure_detailed_time(
        rk4_propagate,
        H0, mux, muy,
        psi0.flatten().tolist(),
        Ex.tolist(),
        Ey.tolist(),
        dt, steps, stride,
        normalize,
        collect_metrics
    )
    
    # 最適化版の設定
    config = RK4Config()
    config.use_simd = True
    config.collect_metrics = collect_metrics
    config.use_threading = True
    config.chunk_size = 64
    config.min_parallel_size = 1000
    config.tolerance = 1e-10
    
    # 最適化版の実行時間を計測
    timing_v2 = measure_detailed_time(
        rk4_propagate_v2,
        H0, mux, muy,
        psi0.flatten().tolist(),
        Ex.tolist(),
        Ey.tolist(),
        dt, steps, stride,
        config
    )
    
    return timing_stable, timing_v2

def plot_performance_comparison(
    steps_list: List[int],
    time_stable: List[float],
    time_v2: List[float],
    save_path: Optional[str] = None
):
    """性能比較のグラフを作成する"""
    plt.figure(figsize=(10, 6))
    
    # 実行時間のプロット
    plt.subplot(2, 1, 1)
    plt.plot(steps_list, time_stable, 'b-', label='rk4_propagate')
    plt.plot(steps_list, time_v2, 'r--', label='rk4_propagate_v2')
    plt.xlabel('ステップ数')
    plt.ylabel('実行時間 [秒]')
    plt.legend()
    plt.grid(True)
    
    # 速度比のプロット
    plt.subplot(2, 1, 2)
    speedup = np.array(time_stable) / np.array(time_v2)
    plt.plot(steps_list, speedup, 'g-')
    plt.xlabel('ステップ数')
    plt.ylabel('速度比 (stable/v2)')
    plt.grid(True)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

def main():
    """メイン関数"""
    print("ベンチマーク開始...")
    
    # テストケースの準備
    H0, mux, muy = create_test_matrices(2)
    psi0 = np.array([[1.0 + 0.0j], [0.0 + 0.0j]], dtype=np.complex128)
    dt = 0.02
    stride = 1
    
    # 異なるステップ数でベンチマークを実行
    steps_list = [100, 200, 500, 1000]
    time_stable_list = []
    time_v2_list = []
    
    print("     ステップ数 |   rk4_propagate | rk4_propagate_v2 |        速度比")
    print("------------------------------------------------------------")
    
    for steps in steps_list:
        Ex, Ey = create_test_pulse(steps)
        timing_stable, timing_v2 = run_benchmark(
            H0, mux, muy, psi0, Ex, Ey, dt, steps - 1, stride
        )
        
        time_stable_list.append(timing_stable[0])
        time_v2_list.append(timing_v2[0])
        
        speedup = timing_stable[0] / timing_v2[0]
        print(f"{steps:12d} | {timing_stable[0]:14.6f} | {timing_v2[0]:14.6f} | {speedup:14.6f}")
    
    # 結果をグラフ化
    plot_performance_comparison(
        steps_list,
        time_stable_list,
        time_v2_list,
        "benchmark_results.png"
    )

if __name__ == "__main__":
    main()
