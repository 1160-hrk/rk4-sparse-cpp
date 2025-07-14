"""
Benchmark module for harmonic oscillator system
=============================================

Measures:
- Total simulation time
- Matrix update time (C++ only)
- RK4 step time (C++ only)
"""

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '../../python'))

import time
import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from rk4_sparse import rk4_sparse_py, rk4_sparse_cpp
from harmonic_oscillator import create_ho_matrices, create_gaussian_pulse

savepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(savepath, exist_ok=True)

class HOBenchmark:
    def __init__(self, n_levels: int = 10, omega: float = 10.0):
        """Initialize benchmark parameters
        
        Args:
            n_levels (int): Number of energy levels
            omega (float): Angular frequency
        """
        self.n_levels = n_levels
        self.omega = omega
        self.hbar = 1.0
        self.mu0 = 1.0
        
        # パルスパラメータ
        self.sigma = 5.0
        self.amplitude = 1.0
        self.omega_L = omega
        
        # 時間パラメータ
        self.dt = 0.001
        self.t_max = self.sigma * 10
        self.t0 = self.t_max / 2
        self.t = np.arange(0, self.t_max, self.dt)
        self.steps = len(self.t)
        
        # 行列とパルスの生成
        self.H0, self.mux, self.muy = create_ho_matrices(
            self.n_levels, self.omega, self.hbar, self.mu0
        )
        self.Ex = create_gaussian_pulse(
            self.t, self.omega_L, self.t0, self.sigma, self.amplitude
        )
        self.Ey = np.zeros_like(self.Ex)
        
        # 初期状態（基底状態）
        self.psi0 = np.zeros(self.n_levels, dtype=np.complex128)
        self.psi0[0] = 1.0
        
        # 出力設定
        self.stride = 10
        self.return_traj = True
        self.renorm = False
    
    def run_python(self):
        """Run Python implementation and measure time"""
        start_time = time.time()
        result = rk4_sparse_py(
            self.H0, self.mux, self.muy,
            self.Ex, self.Ey,
            self.psi0,
            self.dt*2,
            self.return_traj,
            self.stride,
            self.renorm
        )
        end_time = time.time()
        return result, end_time - start_time
    
    def run_cpp(self):
        """Run C++ implementation and measure time"""
        start_time = time.time()
        result = rk4_sparse_cpp(
            self.H0, self.mux, self.muy,
            self.Ex, self.Ey,
            self.psi0,
            self.dt*2,
            self.return_traj,
            self.stride,
            self.renorm
        )
        end_time = time.time()
        return result, end_time - start_time

def run_benchmark(n_levels_list=[10, 50, 100, 200]):
    """Run benchmark for different system sizes"""
    results = []
    
    for n_levels in n_levels_list:
        print(f"\nBenchmarking {n_levels} levels system...")
        bench = HOBenchmark(n_levels=n_levels)
        
        # Python実装
        _, py_time = bench.run_python()
        print(f"Python implementation time: {py_time:.3f} seconds")
        
        # C++実装
        _, cpp_time = bench.run_cpp()
        print(f"C++ implementation time: {cpp_time:.3f} seconds")
        print(f"Speed-up ratio: {py_time/cpp_time:.1f}x")
        
        results.append({
            'n_levels': n_levels,
            'python_time': py_time,
            'cpp_time': cpp_time,
            'speedup': py_time/cpp_time
        })
    
    # 結果のプロット
    plot_results(results)

def plot_results(results):
    """Plot benchmark results"""
    n_levels = [r['n_levels'] for r in results]
    py_times = [r['python_time'] for r in results]
    cpp_times = [r['cpp_time'] for r in results]
    speedups = [r['speedup'] for r in results]
    
    plt.figure(figsize=(12, 5))
    
    # 実行時間のプロット
    plt.subplot(1, 2, 1)
    plt.plot(n_levels, py_times, 'o-', label='Python')
    plt.plot(n_levels, cpp_times, 's-', label='C++')
    plt.xlabel('Number of Levels')
    plt.ylabel('Execution Time (s)')
    plt.title('Execution Time vs System Size')
    plt.grid(True)
    plt.legend()
    
    # スピードアップ比のプロット
    plt.subplot(1, 2, 2)
    plt.plot(n_levels, speedups, 'o-')
    plt.xlabel('Number of Levels')
    plt.ylabel('Speed-up Ratio')
    plt.title('C++ Speed-up vs System Size')
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'ho_benchmark_results.png'))
    plt.close()

if __name__ == '__main__':
    run_benchmark() 