import sys
import os
import time
import numpy as np
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt

# プロジェクトルートへのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'python'))

from rk4_sparse import (
    rk4_sparse_py, rk4_sparse_eigen, rk4_sparse_suitesparse,
    rk4_sparse_suitesparse_optimized,  # 最適化版
    rk4_sparse_suitesparse_fast,       # 高速版も有効化
    OPENBLAS_SUITESPARSE_AVAILABLE
)

savepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(savepath, exist_ok=True)

# シミュレーションパラメータ
omega = 1.0  # 遷移周波数
E0 = 0.1     # 電場強度
omega_L = 1.0  # レーザー周波数（共鳴条件）
dt_E = 0.01     # 時間ステップ
steps_E = 10000  # 総ステップ数
stride = 1    # 出力間隔

# より大きな問題サイズでのテストも追加
test_sizes = [2, 4, 8, 16, 32]  # テストする行列サイズ

# 二準位系のハミルトニアン（基本サイズ）
H0 = csr_matrix([[0, 0],
                 [0, omega]], dtype=np.complex128)

# 双極子演算子（x方向）
mux = csr_matrix([[0, 1],
                  [1, 0]], dtype=np.complex128)

# 双極子演算子（y方向）- この例では使用しない
muy = csr_matrix([[0, 0],
                  [0, 0]], dtype=np.complex128)

# 初期状態 (基底状態)
psi0 = np.array([1, 0], dtype=np.complex128)

# 正弦波の電場を生成
t = np.arange(0, dt_E * (steps_E+2), dt_E)
Ex = E0 * np.sin(omega_L * t)
Ey = np.zeros_like(Ex)

# 結果を格納する辞書
results = {}
timings = {}
scaling_results = {}  # スケーリング結果を格納

# 各問題サイズでのテスト
for size in test_sizes:
    print(f"\n{'='*60}")
    print(f"問題サイズ: {size}x{size}")
    print(f"{'='*60}")

    # より大きなシステムの生成
    if size > 2:
        # 多準位系のハミルトニアン
        H0_large = csr_matrix((size, size), dtype=np.complex128)
        mux_large = csr_matrix((size, size), dtype=np.complex128)
        muy_large = csr_matrix((size, size), dtype=np.complex128)

        # 対角要素（エネルギー準位）
        for i in range(size):
            H0_large[i, i] = i * omega

        # 非対角要素（遷移）
        for i in range(size-1):
            mux_large[i, i+1] = 1.0
            mux_large[i+1, i] = 1.0

        H0_test = H0_large
        mux_test = mux_large
        muy_test = muy_large
        psi0_test = np.zeros(size, dtype=np.complex128)
        psi0_test[0] = 1.0
    else:
        H0_test = H0
        mux_test = mux
        muy_test = muy
        psi0_test = psi0

    # このサイズでの結果を格納
    size_results = {}
    size_timings = {}

    # Python実装
    print(f"\nRunning Python-sparse implementation (size {size})...")
    start_t = time.perf_counter()
    try:
        size_results['python'] = rk4_sparse_py(
            H0_test, mux_test, muy_test,
            Ex, Ey,
            psi0_test,
            dt_E*2,
            True,
            stride,
            False,
        )
        end_t = time.perf_counter()
        size_timings['python'] = end_t - start_t
        print(f"Python-sparse implementation time: "
              f"{size_timings['python']:.6f} seconds")
    except Exception as e:
        print(f"Python implementation failed: {e}")
        size_results['python'] = None
        size_timings['python'] = None

    # C++ Eigen実装
    print(f"\nRunning C++ Eigen implementation (size {size})...")
    start_t = time.perf_counter()
    try:
        size_results['eigen'] = rk4_sparse_eigen(
            H0_test, mux_test, muy_test,
            Ex, Ey,
            psi0_test,
            dt_E*2,
            True,
            stride,
            False,
        )
        end_t = time.perf_counter()
        size_timings['eigen'] = end_t - start_t
        print(f"C++ Eigen implementation time: "
              f"{size_timings['eigen']:.6f} seconds")
    except Exception as e:
        print(f"Eigen implementation failed: {e}")
        size_results['eigen'] = None
        size_timings['eigen'] = None

    # C++ SuiteSparse実装（基本版）
    print(f"\nRunning C++ SuiteSparse implementation (basic, size {size})...")
    start_t = time.perf_counter()
    try:
        size_results['suitesparse'] = rk4_sparse_suitesparse(
            H0_test, mux_test, muy_test,
            Ex, Ey,
            psi0_test,
            dt_E*2,
            True,
            stride,
            False,
        )
        end_t = time.perf_counter()
        size_timings['suitesparse'] = end_t - start_t
        print(f"C++ SuiteSparse basic implementation time: "
              f"{size_timings['suitesparse']:.6f} seconds")
    except Exception as e:
        print(f"SuiteSparse basic implementation failed: {e}")
        size_results['suitesparse'] = None
        size_timings['suitesparse'] = None

    # C++ SuiteSparse実装（最適化版）
    print(f"\nRunning C++ SuiteSparse implementation "
          f"(optimized, size {size})...")
    start_t = time.perf_counter()
    try:
        size_results['suitesparse_optimized'] = (
            rk4_sparse_suitesparse_optimized(
                H0_test, mux_test, muy_test,
                Ex, Ey,
                psi0_test,
                dt_E*2,
                True,
                stride,
                False,
            )
        )
        end_t = time.perf_counter()
        size_timings['suitesparse_optimized'] = end_t - start_t
        print(f"C++ SuiteSparse optimized implementation time: "
              f"{size_timings['suitesparse_optimized']:.6f} seconds")
    except Exception as e:
        print(f"SuiteSparse optimized implementation failed: {e}")
        size_results['suitesparse_optimized'] = None
        size_timings['suitesparse_optimized'] = None

    # C++ SuiteSparse実装（高速版）
    print(f"\nRunning C++ SuiteSparse implementation (fast, size {size})...")
    start_t = time.perf_counter()
    try:
        size_results['suitesparse_fast'] = rk4_sparse_suitesparse_fast(
            H0_test, mux_test, muy_test,
            Ex, Ey,
            psi0_test,
            dt_E*2,
            True,
            stride,
            False,
        )
        end_t = time.perf_counter()
        size_timings['suitesparse_fast'] = end_t - start_t
        print(f"C++ SuiteSparse fast implementation time: "
              f"{size_timings['suitesparse_fast']:.6f} seconds")
    except Exception as e:
        print(f"SuiteSparse fast implementation failed: {e}")
        size_results['suitesparse_fast'] = None
        size_timings['suitesparse_fast'] = None

    # このサイズでの結果を保存
    scaling_results[size] = {
        'results': size_results,
        'timings': size_timings
    }

    # 最初のサイズ（2x2）の結果をメインの結果として保存
    if size == 2:
        results = size_results
        timings = size_timings

# 各実装の基底・励起状態の占有数を計算
populations = {}
for name, result in results.items():
    if result is not None:
        populations[name] = {
            'P0': np.abs(result[:, 0])**2,
            'P1': np.abs(result[:, 1])**2
        }

# ラビ振動の解析解
Omega_R = E0 / 2
# 数値計算と同じ時間点を使用
t_analytical = np.arange(0, dt_E * (steps_E+1), 2*dt_E*stride)
P0_analytical = np.cos(Omega_R * t_analytical)**2
P1_analytical = np.sin(Omega_R * t_analytical)**2

# プロット
plt.figure(figsize=(20, 15))

# サブプロット1: 全実装の比較（2x2の場合）
plt.subplot(3, 3, 1)
colors = ['b', 'r', 'g', 'm', 'c', 'y', 'k']
linestyles = ['-', '-.', ':', '--', '-', '-.', ':']
alpha = 0.7

for i, (name, pop) in enumerate(populations.items()):
    color = colors[i % len(colors)]
    ls = linestyles[i % len(linestyles)]
    plt.plot(t_analytical, pop['P0'], color=color, linestyle=ls,
             label=f'Ground ({name})', alpha=alpha)
    plt.plot(t_analytical, pop['P1'], color=color, linestyle=ls,
             label=f'Excited ({name})', alpha=alpha)

plt.plot(t_analytical, P0_analytical, 'k--',
         label='Ground (analytical)', alpha=0.8, linewidth=2)
plt.plot(t_analytical, P1_analytical, 'g--',
         label='Excited (analytical)', alpha=0.8, linewidth=2)
plt.xlabel('Time (a.u.)')
plt.ylabel('Population')
plt.title('All Implementations vs Analytical (2x2)')
plt.grid(True)
plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')

# サブプロット2: 実装間の差分（Python vs 他）
plt.subplot(3, 3, 2)
if 'python' in populations:
    for name, pop in populations.items():
        if name != 'python':
            plt.plot(t_analytical,
                     np.abs(populations['python']['P0'] - pop['P0']),
                     label=f'Ground: Python-{name}', alpha=alpha)
            plt.plot(t_analytical,
                     np.abs(populations['python']['P1'] - pop['P1']),
                     label=f'Excited: Python-{name}', alpha=alpha)
    plt.xlabel('Time (a.u.)')
    plt.ylabel('|Population difference|')
    plt.title('Difference from Python Implementation')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')

# サブプロット3: 実装間の差分（Eigen vs 他）
plt.subplot(3, 3, 3)
if 'eigen' in populations:
    for name, pop in populations.items():
        if name != 'eigen':
            plt.plot(t_analytical,
                     np.abs(populations['eigen']['P0'] - pop['P0']),
                     label=f'Ground: Eigen-{name}', alpha=alpha)
            plt.plot(t_analytical,
                     np.abs(populations['eigen']['P1'] - pop['P1']),
                     label=f'Excited: Eigen-{name}', alpha=alpha)
    plt.xlabel('Time (a.u.)')
    plt.ylabel('|Population difference|')
    plt.title('Difference from Eigen Implementation')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')

# サブプロット4: 電場
plt.subplot(3, 3, 4)
plt.plot(t, Ex, label='Electric field')
plt.xlabel('Time (a.u.)')
plt.ylabel('Electric field (a.u.)')
plt.title('Applied electric field')
plt.grid(True)
plt.legend()

# サブプロット5: 実行時間比較（2x2）
plt.subplot(3, 3, 5)
valid_timings = {k: v for k, v in timings.items() if v is not None}
if valid_timings:
    names = list(valid_timings.keys())
    times = list(valid_timings.values())
    colors_plot = [colors[i % len(colors)] for i in range(len(names))]
    bars = plt.bar(names, times, color=colors_plot, alpha=0.7)
    plt.ylabel('Execution time (seconds)')
    plt.title('Performance Comparison (2x2)')
    plt.xticks(rotation=45)

    # バーの上に数値を表示
    for bar, time_val in zip(bars, times):
        plt.text(bar.get_x() + bar.get_width()/2,
                 bar.get_height() + max(times)*0.01,
                 f'{time_val:.4f}', ha='center', va='bottom', fontsize=8)

# サブプロット6: 速度向上率（2x2）
plt.subplot(3, 3, 6)
if 'python' in valid_timings and len(valid_timings) > 1:
    python_time = valid_timings['python']
    speedups = {}
    for name, time_val in valid_timings.items():
        if name != 'python':
            speedups[name] = python_time / time_val

    if speedups:
        names = list(speedups.keys())
        speedup_vals = list(speedups.values())
        colors_plot = [colors[i % len(colors)] for i in range(len(names))]
        bars = plt.bar(names, speedup_vals, color=colors_plot, alpha=0.7)
        plt.ylabel('Speedup vs Python')
        plt.title('Speedup Comparison (2x2)')
        plt.xticks(rotation=45)

        # バーの上に数値を表示
        for bar, speedup in zip(bars, speedup_vals):
            plt.text(bar.get_x() + bar.get_width()/2,
                     bar.get_height() + max(speedup_vals)*0.01,
                     f'{speedup:.2f}x', ha='center', va='bottom', fontsize=8)

# サブプロット7: スケーリング性能（問題サイズ vs 実行時間）
plt.subplot(3, 3, 7)
if scaling_results:
    sizes = list(scaling_results.keys())
    implementations = ['eigen', 'suitesparse', 'suitesparse_optimized',
                       'suitesparse_fast']

    for impl in implementations:
        times = []
        for size in sizes:
            timing = scaling_results[size]['timings'].get(impl)
            if timing is not None:
                times.append(timing)
            else:
                times.append(np.nan)

        if not all(np.isnan(times)):
            plt.plot(sizes, times, 'o-', label=impl, alpha=0.7)

    plt.xlabel('Matrix size')
    plt.ylabel('Execution time (seconds)')
    plt.title('Scaling Performance')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')

# サブプロット8: スケーリング性能（問題サイズ vs 速度向上率）
plt.subplot(3, 3, 8)
if scaling_results:
    sizes = list(scaling_results.keys())
    implementations = ['eigen', 'suitesparse', 'suitesparse_optimized',
                       'suitesparse_fast']

    for impl in implementations:
        speedups = []
        for size in sizes:
            python_time = scaling_results[size]['timings'].get('python')
            impl_time = scaling_results[size]['timings'].get(impl)
            if python_time is not None and impl_time is not None:
                speedups.append(python_time / impl_time)
            else:
                speedups.append(np.nan)

        if not all(np.isnan(speedups)):
            plt.plot(sizes, speedups, 'o-', label=impl, alpha=0.7)

    plt.xlabel('Matrix size')
    plt.ylabel('Speedup vs Python')
    plt.title('Scaling Speedup')
    plt.grid(True)
    plt.legend()

# サブプロット9: 実装状況のサマリー
plt.subplot(3, 3, 9)
plt.axis('off')
summary_text = "Implementation Status:\n\n"
for name in ['python', 'eigen', 'suitesparse', 'suitesparse_optimized',
             'suitesparse_fast']:
    status = "✓ Available" if results.get(name) is not None else "✗ Failed"
    timing = (f" ({timings.get(name):.4f}s)"
              if timings.get(name) is not None else "")
    summary_text += f"{name}: {status}{timing}\n"

if OPENBLAS_SUITESPARSE_AVAILABLE:
    summary_text += "\nOpenBLAS + SuiteSparse: ✓ Available"
else:
    summary_text += "\nOpenBLAS + SuiteSparse: ✗ Not available"

# スケーリング結果のサマリーを追加
summary_text += "\n\nScaling Test Results:\n"
for size in test_sizes:
    if size in scaling_results:
        eigen_time = scaling_results[size]['timings'].get('eigen')
        if eigen_time is not None:
            summary_text += f"Size {size}x{size}: Eigen {eigen_time:.4f}s\n"

plt.text(0.1, 0.9, summary_text, transform=plt.gca().transAxes,
         fontsize=10, verticalalignment='top', fontfamily='monospace')

plt.tight_layout()
plt.savefig(os.path.join(savepath, 'compare_eigen_suitesparse.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# 結果の表示
print("\n" + "="*60)
print("SIMULATION COMPLETED!")
print("="*60)

print("\nImplementation Status (2x2):")
for name in ['python', 'eigen', 'suitesparse', 'suitesparse_optimized',
             'suitesparse_fast']:
    if results.get(name) is not None:
        print(f"✓ {name}: {timings.get(name):.6f} seconds")
        if name in populations:
            print(f"  Final ground state: {populations[name]['P0'][-1]:.6f}")
            print(f"  Final excited state: {populations[name]['P1'][-1]:.6f}")
    else:
        print(f"✗ {name}: Failed")

print("\nAnalytical Solution:")
print(f"Final ground state population: {P0_analytical[-1]:.6f}")
print(f"Final excited state population: {P1_analytical[-1]:.6f}")

# 速度向上率の計算（2x2）
if 'python' in valid_timings and len(valid_timings) > 1:
    print("\nSpeedup Analysis (2x2):")
    python_time = valid_timings['python']
    for name, time_val in valid_timings.items():
        if name != 'python':
            speedup = python_time / time_val
            print(f"{name}: {speedup:.2f}x faster than Python")

# スケーリング結果の表示
print("\n" + "="*60)
print("SCALING ANALYSIS")
print("="*60)

for size in test_sizes:
    if size in scaling_results:
        print(f"\nMatrix Size: {size}x{size}")
        size_timings = scaling_results[size]['timings']

        # 各実装の実行時間
        for name in ['python', 'eigen', 'suitesparse', 'suitesparse_optimized',
             'suitesparse_fast']:
            timing = size_timings.get(name)
            if timing is not None:
                print(f"  {name}: {timing:.6f} seconds")

        # 速度向上率（Python基準）
        python_time = size_timings.get('python')
        if python_time is not None:
            print("  Speedup vs Python:")
            for name in ['eigen', 'suitesparse', 'suitesparse_optimized',
                         'suitesparse_fast']:
                impl_time = size_timings.get(name)
                if impl_time is not None:
                    speedup = python_time / impl_time
                    print(f"    {name}: {speedup:.2f}x")

        # Eigen比
        eigen_time = size_timings.get('eigen')
        if eigen_time is not None:
            print("  Speedup vs Eigen:")
            for name in ['suitesparse', 'suitesparse_optimized',
                         'suitesparse_fast']:
                impl_time = size_timings.get(name)
                if impl_time is not None:
                    speedup = eigen_time / impl_time
                    print(f"    {name}: {speedup:.2f}x")

print(f"\nPlot saved as: "
      f"{os.path.join(savepath, 'compare_eigen_suitesparse.png')}")
print("="*60)
