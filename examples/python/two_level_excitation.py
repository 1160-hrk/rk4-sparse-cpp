import sys
import os

# プロジェクトルートへのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

savepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(savepath, exist_ok=True)

import numpy as np
import time
from scipy.sparse import csr_matrix
import matplotlib.pyplot as plt
from rk4_sparse import rk4_sparse_py, rk4_sparse_cpp, rk4_numba_py


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
psi0 = np.array([1, 0], dtype=np.complex128)  # 形状を修正

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

# Python実装での時間発展を計算

print("\nRunning Python-Numba implementation...")
H0_numba = H0.toarray()
mux_numba = mux.toarray()
muy_numba = muy.toarray()

start_t = time.perf_counter()
result_numba = rk4_numba_py(
    H0_numba, mux_numba, muy_numba,
    Ex.astype(np.float64), Ey.astype(np.float64),
    psi0,
    dt_E*2,
    True,
    stride,
    False,
)
end_t = time.perf_counter()
print("Python-Numba implementation completed.")
print(f"Python-Numba implementation time: {end_t - start_t} seconds")
print(f"Result shape (Python-Numba): {result_numba.shape}")

print("\nRunning Python-sparse implementation...")
start_t = time.perf_counter()
result_py = rk4_sparse_py(
    H0, mux, muy,
    Ex, Ey,
    psi0,
    dt_E*2,
    True,
    stride,
    False,
)
end_t = time.perf_counter()
print("Python-sparse implementation completed.")
print(f"Python-sparse implementation time: {end_t - start_t} seconds")
print(f"Result shape (Python-sparse): {result_py.shape}")

# CSRフォーマットのデータを取得
print("\nExtracting CSR format data for C++ implementation...")
print(f"H0 nnz: {H0.nnz}")
print(f"mux nnz: {mux.nnz}")
print(f"muy nnz: {muy.nnz}")

# C++実装での時間発展を計算
print("\nRunning C++ implementation...")
print("psi0.shape:", psi0.shape)
print("H0.shape:", H0.shape)
start_t = time.perf_counter()
result_cpp = rk4_sparse_cpp(
    H0, mux, muy,
    Ex, Ey,
    psi0,
    dt_E*2,
    True,
    stride,
    False,
)
end_t = time.perf_counter()
print("C++ implementation completed.")
print(f"C++ implementation time: {end_t - start_t} seconds")
print(f"Result shape (C++): {result_cpp.shape}")

# 基底状態と励起状態の占有数を計算（Python実装）
P0_py = np.abs(result_py[:, 0])**2  # 基底状態
P1_py = np.abs(result_py[:, 1])**2  # 励起状態

# 基底状態と励起状態の占有数を計算（C++実装）
P0_cpp = np.abs(result_cpp[:, 0])**2  # 基底状態
P1_cpp = np.abs(result_cpp[:, 1])**2  # 励起状態

print("\n=== Results ===")
print("Time points:", len(t))
print("Electric field points:", len(Ex))
print("Population points (Python):", len(P0_py))
print("Population points (C++):", len(P0_cpp))

# ラビ振動の解析解
# ラビ周波数 ΩR = μE0/ℏ （ℏ = 1, μ = 1 in atomic units）
Omega_R = E0 / 2  # 実効的なラビ周波数（回転波近似による因子1/2）
t_analytical = np.arange(0, dt_E * (steps_E+1), 2*dt_E*stride)  # 数値計算と同じ時間点を使用
P0_analytical = np.cos(Omega_R * t_analytical)**2
P1_analytical = np.sin(Omega_R * t_analytical)**2

# プロット
plt.figure(figsize=(15, 10))

# サブプロット1: Python実装と解析解の比較
plt.subplot(2, 2, 1)
plt.plot(t_analytical, P0_py, 'b-', label='Ground state (Python)', alpha=0.7)
plt.plot(t_analytical, P1_py, 'r-', label='Excited state (Python)', alpha=0.7)
plt.plot(t_analytical, P0_analytical, 'b--', label='Ground state (analytical)', alpha=0.7)
plt.plot(t_analytical, P1_analytical, 'r--', label='Excited state (analytical)', alpha=0.7)
plt.xlabel('Time (a.u.)')
plt.ylabel('Population')
plt.title('Python Implementation vs Analytical')
plt.grid(True)
plt.legend()

# サブプロット2: C++実装と解析解の比較
plt.subplot(2, 2, 2)
plt.plot(t_analytical, P0_cpp, 'b-', label='Ground state (C++)', alpha=0.7)
plt.plot(t_analytical, P1_cpp, 'r-', label='Excited state (C++)', alpha=0.7)
plt.plot(t_analytical, P0_analytical, 'b--', label='Ground state (analytical)', alpha=0.7)
plt.plot(t_analytical, P1_analytical, 'r--', label='Excited state (analytical)', alpha=0.7)
plt.xlabel('Time (a.u.)')
plt.ylabel('Population')
plt.title('C++ Implementation vs Analytical')
plt.grid(True)
plt.legend()

# サブプロット3: Python実装とC++実装の差
plt.subplot(2, 2, 3)
plt.plot(t_analytical, np.abs(P0_py - P0_cpp), 'b-', label='Ground state difference', alpha=0.7)
plt.plot(t_analytical, np.abs(P1_py - P1_cpp), 'r-', label='Excited state difference', alpha=0.7)
plt.xlabel('Time (a.u.)')
plt.ylabel('|Population difference|')
plt.title('Difference between Python and C++ implementations')
plt.grid(True)
plt.legend()
plt.yscale('log')  # 差分を対数スケールで表示

# サブプロット4: 印加した電場
plt.subplot(2, 2, 4)
plt.plot(t, Ex, label='Electric field')
plt.xlabel('Time (a.u.)')
plt.ylabel('Electric field (a.u.)')
plt.title('Applied electric field')
plt.grid(True)
plt.legend()

plt.tight_layout()
plt.savefig(os.path.join(savepath, 'two_level_excitation_comparison.png'))
plt.close()

# 結果の表示
print("\nSimulation completed!")
print("=== Python Implementation ===")
print(f"Final ground state population: {P0_py[-2]:.6f}")
print(f"Final excited state population: {P1_py[-2]:.6f}")
print("\n=== C++ Implementation ===")
print(f"Final ground state population: {P0_cpp[-2]:.6f}")
print(f"Final excited state population: {P1_cpp[-2]:.6f}")
print("\n=== Analytical Solution ===")
print(f"Final ground state population: {P0_analytical[-1]:.6f}")
print(f"Final excited state population: {P1_analytical[-1]:.6f}")

# 実装間の差の最大値を計算
max_diff_implementations = {
    'ground': np.max(np.abs(P0_py - P0_cpp)),
    'excited': np.max(np.abs(P1_py - P1_cpp))
}

# 解析解との差の最大値を計算
max_diff_analytical = {
    'python': {
        'ground': np.max(np.abs(P0_py - P0_analytical)),
        'excited': np.max(np.abs(P1_py - P1_analytical))
    },
    'cpp': {
        'ground': np.max(np.abs(P0_cpp - P0_analytical)),
        'excited': np.max(np.abs(P1_cpp - P1_analytical))
    }
}

print("\n=== Maximum Differences ===")
print("Between implementations:")
print(f"Ground state: {max_diff_implementations['ground']:.6e}")
print(f"Excited state: {max_diff_implementations['excited']:.6e}")
print("\nPython vs Analytical:")
print(f"Ground state: {max_diff_analytical['python']['ground']:.6e}")
print(f"Excited state: {max_diff_analytical['python']['excited']:.6e}")
print("\nC++ vs Analytical:")
print(f"Ground state: {max_diff_analytical['cpp']['ground']:.6e}")
print(f"Excited state: {max_diff_analytical['cpp']['excited']:.6e}")

print("\nPlot saved as {}".format(os.path.join(savepath, 'two_level_excitation_comparison.png'))) 