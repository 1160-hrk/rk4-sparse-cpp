import sys
import os

# プロジェクトルートへのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..',
                                'python'))

import numpy as np
import time
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from rk4_sparse import (rk4_sparse_py, rk4_sparse_eigen,
                        rk4_sparse_eigen_cached, rk4_numba_py)

savepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(savepath, exist_ok=True)


# シミュレーションパラメータ
omega = 1.0  # 遷移周波数
E0 = 0.1     # 電場強度
omega_L = 1.0  # レーザー周波数（共鳴条件）
dt_E = 0.01     # 時間ステップ
steps_E = 10000  # 総ステップ数
stride = 1    # 出力間隔

# 二準位系のハミルトニアン
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

# 入力型の確認
print("=== Input Types ===")
print(f"H0 type: {type(H0)}, shape: {H0.shape}, dtype: {H0.dtype}")
print(f"mux type: {type(mux)}, shape: {mux.shape}, dtype: {mux.dtype}")
print(f"muy type: {type(muy)}, shape: {muy.shape}, dtype: {muy.dtype}")
print(f"Ex type: {type(Ex)}, shape: {Ex.shape}, dtype: {Ex.dtype}")
print(f"Ey type: {type(Ey)}, shape: {Ey.shape}, dtype: {Ey.dtype}")
print(f"psi0 type: {type(psi0)}, shape: {psi0.shape}, dtype: {psi0.dtype}")
print(f"dt_E: {dt_E}")
print(f"stride: {stride}")

# 各実装での時間発展を計算
results = {}
timings = {}

# 1. Python-Numba実装
print("\nRunning Python-Numba implementation...")
H0_numba = H0.toarray()
mux_numba = mux.toarray()
muy_numba = muy.toarray()

start_t = time.perf_counter()
results['numba'] = rk4_numba_py(
    H0_numba, mux_numba, muy_numba,
    Ex.astype(np.float64), Ey.astype(np.float64),
    psi0,
    dt_E*2,
    True,
    stride,
    False,
)
end_t = time.perf_counter()
timings['numba'] = end_t - start_t
print(f"Python-Numba implementation completed in "
      f"{timings['numba']:.6f} seconds")
print(f"Result shape: {results['numba'].shape}")

# 2. Python-sparse実装
print("\nRunning Python-sparse implementation...")
start_t = time.perf_counter()
results['py_sparse'] = rk4_sparse_py(
    H0, mux, muy,
    Ex, Ey,
    psi0,
    dt_E*2,
    True,
    stride,
    False,
)
end_t = time.perf_counter()
timings['py_sparse'] = end_t - start_t
print(f"Python-sparse implementation completed in "
      f"{timings['py_sparse']:.6f} seconds")
print(f"Result shape: {results['py_sparse'].shape}")

# 3. C++ Eigen実装
print("\nRunning C++ Eigen implementation...")
start_t = time.perf_counter()
results['cpp_eigen'] = rk4_sparse_eigen(
    H0, mux, muy,
    Ex, Ey,
    psi0,
    dt_E*2,
    True,
    stride,
    False,
)
end_t = time.perf_counter()
timings['cpp_eigen'] = end_t - start_t
print(f"C++ Eigen implementation completed in "
      f"{timings['cpp_eigen']:.6f} seconds")
print(f"Result shape: {results['cpp_eigen'].shape}")

# 4. C++ Eigen Cached実装
print("\nRunning C++ Eigen Cached implementation...")
start_t = time.perf_counter()
results['cpp_eigen_cached'] = rk4_sparse_eigen_cached(
    H0, mux, muy,
    Ex, Ey,
    psi0,
    dt_E*2,
    True,
    stride,
    False,
)
end_t = time.perf_counter()
timings['cpp_eigen_cached'] = end_t - start_t
print(f"C++ Eigen Cached implementation completed in "
      f"{timings['cpp_eigen_cached']:.6f} seconds")
print(f"Result shape: {results['cpp_eigen_cached'].shape}")

# 各実装での基底状態と励起状態の占有数を計算
populations = {}
for name, result in results.items():
    populations[name] = {
        'ground': np.abs(result[:, 0])**2,  # 基底状態
        'excited': np.abs(result[:, 1])**2   # 励起状態
    }

# ラビ振動の解析解
Omega_R = E0 / 2  # 実効的なラビ周波数（回転波近似による因子1/2）
t_analytical = np.arange(0, dt_E * (steps_E+1), 2*dt_E*stride)  # 数値計算と同じ時間点を使用
P0_analytical = np.cos(Omega_R * t_analytical)**2
P1_analytical = np.sin(Omega_R * t_analytical)**2

# プロット
plt.figure(figsize=(20, 15))

# サブプロット1: 全実装の基底状態占有数比較
plt.subplot(3, 3, 1)
for name, pop in populations.items():
    plt.plot(t_analytical, pop['ground'], label=f'{name}', alpha=0.8)
plt.plot(t_analytical, P0_analytical, 'k--', label='Analytical', linewidth=2)
plt.xlabel('Time (a.u.)')
plt.ylabel('Ground State Population')
plt.title('Ground State Population Comparison')
plt.grid(True)
plt.legend()

# サブプロット2: 全実装の励起状態占有数比較
plt.subplot(3, 3, 2)
for name, pop in populations.items():
    plt.plot(t_analytical, pop['excited'], label=f'{name}', alpha=0.8)
plt.plot(t_analytical, P1_analytical, 'k--', label='Analytical', linewidth=2)
plt.xlabel('Time (a.u.)')
plt.ylabel('Excited State Population')
plt.title('Excited State Population Comparison')
plt.grid(True)
plt.legend()

# サブプロット3: 実行時間比較
plt.subplot(3, 3, 3)
names = list(timings.keys())
times = list(timings.values())
colors = ['blue', 'green', 'red', 'orange']
bars = plt.bar(names, times, color=colors, alpha=0.7)
plt.ylabel('Execution Time (seconds)')
plt.title('Performance Comparison')
plt.grid(True, axis='y')
# バーの上に数値を表示
for bar, time_val in zip(bars, times):
    plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.001,
             f'{time_val:.4f}', ha='center', va='bottom')

# サブプロット4-7: 各実装と解析解の差
implementations = ['numba', 'py_sparse', 'cpp_eigen', 'cpp_eigen_cached']
titles = ['Python-Numba', 'Python-Sparse', 'C++ Eigen', 'C++ Eigen Cached']

for i, (impl, title) in enumerate(zip(implementations, titles)):
    plt.subplot(3, 3, 4 + i)
    plt.plot(t_analytical,
             np.abs(populations[impl]['ground'] - P0_analytical),
             'b-', label='Ground state diff', alpha=0.7)
    plt.plot(t_analytical,
             np.abs(populations[impl]['excited'] - P1_analytical),
             'r-', label='Excited state diff', alpha=0.7)
    plt.xlabel('Time (a.u.)')
    plt.ylabel('|Population difference|')
    plt.title(f'{title} vs Analytical')
    plt.grid(True)
    plt.legend()
    plt.yscale('log')

# サブプロット8: 実装間の差（基準としてC++ Eigen Cachedを使用）
plt.subplot(3, 3, 8)
reference = 'cpp_eigen_cached'
for name in implementations:
    if name != reference:
        plt.plot(t_analytical,
                 np.abs(populations[name]['ground'] -
                        populations[reference]['ground']),
                 label=f'{name} vs {reference} (ground)', alpha=0.7)
        plt.plot(t_analytical,
                 np.abs(populations[name]['excited'] -
                        populations[reference]['excited']),
                 label=f'{name} vs {reference} (excited)', alpha=0.7)
plt.xlabel('Time (a.u.)')
plt.ylabel('|Population difference|')
plt.title(f'Implementation Differences (vs {reference})')
plt.grid(True)
plt.legend()
plt.yscale('log')

# サブプロット9: 印加した電場
plt.subplot(3, 3, 9)
plt.plot(t, Ex, label='Electric field', color='purple')
plt.xlabel('Time (a.u.)')
plt.ylabel('Electric field (a.u.)')
plt.title('Applied electric field')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(savepath, 'four_implementation_comparison.png'),
            dpi=300, bbox_inches='tight')
plt.close()

# 結果の表示
print("\n" + "="*60)
print("SIMULATION RESULTS SUMMARY")
print("="*60)

print("\n=== Execution Times ===")
for name, time_val in timings.items():
    print(f"{name:20}: {time_val:.6f} seconds")

print("\n=== Final Populations ===")
for name, pop in populations.items():
    print(f"{name:20}: Ground={pop['ground'][-1]:.6f}, "
          f"Excited={pop['excited'][-1]:.6f}")

print("\n=== Analytical Solution ===")
print(f"Ground state: {P0_analytical[-1]:.6f}")
print(f"Excited state: {P1_analytical[-1]:.6f}")

# 実装間の差の最大値を計算
print("\n=== Maximum Differences ===")
reference = 'cpp_eigen_cached'
for name in implementations:
    if name != reference:
        max_diff_ground = np.max(np.abs(populations[name]['ground'] -
                                        populations[reference]['ground']))
        max_diff_excited = np.max(np.abs(populations[name]['excited'] -
                                         populations[reference]['excited']))
        print(f"{name} vs {reference}: Ground={max_diff_ground:.2e}, "
              f"Excited={max_diff_excited:.2e}")

# 解析解との差の最大値を計算
print("\n=== Maximum Differences vs Analytical ===")
for name in implementations:
    max_diff_ground = np.max(np.abs(populations[name]['ground'] -
                                    P0_analytical))
    max_diff_excited = np.max(np.abs(populations[name]['excited'] -
                                     P1_analytical))
    print(f"{name:20}: Ground={max_diff_ground:.2e}, "
          f"Excited={max_diff_excited:.2e}")

# 性能比較
print("\n=== Performance Analysis ===")
fastest_time = min(timings.values())
for name, time_val in timings.items():
    speedup = fastest_time / time_val
    print(f"{name:20}: {speedup:.2f}x slower than fastest")

print(f"\nPlot saved as "
      f"{os.path.join(savepath, 'four_implementation_comparison.png')}")
print("="*60)
