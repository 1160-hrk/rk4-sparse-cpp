"""
Quantum dynamics simulation of harmonic oscillator
===============================================

Energy levels: En = ℏω(n + 1/2)
Dipole moment: <n|μ|n±1> ∝ √(n + 1)
"""

import sys
import os
import time

# プロジェクトルートへのパスを追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

import numpy as np
from scipy import sparse
import matplotlib.pyplot as plt
from rk4_sparse import rk4_sparse_py, rk4_sparse_cpp

savepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(savepath, exist_ok=True)

def create_ho_matrices(n_levels: int, omega: float = 1.0, hbar: float = 1.0, mu0: float = 1.0):
    """調和振動子のハミルトニアンと双極子モーメント行列を生成

    Parameters
    ----------
    n_levels : int
        考慮する準位数
    omega : float
        角振動数
    hbar : float
        プランク定数
    mu0 : float
        双極子モーメント
    Returns
    -------
    H0, mux, muy : scipy.sparse.csr_matrix
        ハミルトニアンと双極子モーメント行列
    """
    # エネルギー固有値
    energies = hbar * omega * (np.arange(n_levels) + 0.5)
    
    # ハミルトニアン（対角行列）
    H0 = sparse.diags(energies, format='csr')
    H0 = H0.tocsr()  # 確実にCSR形式に変換
    
    # 双極子モーメント（隣接準位間の遷移のみ）
    # <n|μ|n+1> ∝ √(n + 1)
    dipole_elements = np.sqrt(np.arange(1, n_levels)) * mu0
    # mux = np.diag(dipole_elements, 1)
    mux = sparse.diags(dipole_elements, 1, format='csr')
    mux = mux + mux.T  # エルミート共役を加える
    mux = mux.tocsr()  # 確実にCSR形式に変換
    
    # y方向の双極子モーメントは0
    muy = sparse.csr_matrix((n_levels, n_levels))
    # muy = np.zeros_like(mux)
    
    return H0, mux, muy

def create_gaussian_pulse(t: np.ndarray, omega_L: float, t0: float, sigma: float, amplitude: float):
    """ガウシアンパルスを生成

    Parameters
    ----------
    t : array_like
        時間配列
    t0 : float
        パルス中心
    sigma : float
        パルス幅
    amplitude : float
        パルス振幅

    Returns
    -------
    Ex : ndarray
        電場波形
    """
    return amplitude * np.exp(-(t - t0)**2 / (2 * sigma**2)) * np.sin(omega_L * t)

def main():
    # パラメータ設定
    n_levels = 10  # 準位数
    omega = 10.0    # 角振動数
    hbar = 1.0     # プランク定数
    mu0 = 1.0      # 双極子モーメント
    
    # パルスパラメータ
    sigma = 5.0
    amplitude = 1
    omega_L = omega
    
    # 時間パラメータ
    dt = 0.001
    t_max = sigma * 10
    t0 = t_max / 2
    t = np.arange(0, t_max, dt)
    steps = len(t)
    
    # 行列の生成
    H0, mux, muy = create_ho_matrices(n_levels, omega, hbar, mu0)
    
    # パルスの生成
    Ex = create_gaussian_pulse(t, omega_L, t0, sigma, amplitude)
    Ey = np.zeros_like(Ex)  # y方向の電場は0
    
    # 初期状態（基底状態）
    psi0 = np.zeros(n_levels, dtype=np.complex128)
    psi0[0] = 1.0
    
    # シミュレーションパラメータ
    stride = 10  # 出力間隔
    
    print("=== Simulation Parameters ===")
    print(f"Number of levels: {n_levels}")
    print(f"Time step: {dt:.3f}")
    print(f"Total steps: {steps // 2}")
    print(f"Pulse center: {t0:.1f}")
    print(f"Pulse width: {sigma:.1f}")
    print(f"Pulse amplitude: {amplitude:.3f}")
    
    # Python実装での計算
    print("\nRunning Python implementation...")
    start_time = time.time()
    result_py = rk4_sparse_py(
        H0, mux, muy,
        Ex, Ey,
        psi0,
        dt*2,
        True,  # 軌跡を返す
        stride,
        False  # 規格化なし
    )
    py_time = time.time() - start_time
    print(f"Python implementation time: {py_time:.3f} seconds")
    
    # C++実装での計算
    print("\nRunning C++ implementation...")
    start_time = time.time()
    result_cpp = rk4_sparse_cpp(
        H0, mux, muy,
        Ex, Ey,
        psi0,
        dt*2,
        True,
        stride,
        False
    )
    cpp_time = time.time() - start_time
    print(f"C++ implementation time: {cpp_time:.3f} seconds")
    print(f"Speed-up ratio: {py_time/cpp_time:.1f}x")
    
    # 結果の可視化
    t_plot = t[::stride*2]  # strideに合わせて時間配列を間引く
    populations_py = np.abs(result_py)**2
    populations_cpp = np.abs(result_cpp)**2
    
    plt.figure(figsize=(12, 8))
    
    # Python実装の結果
    plt.subplot(2, 1, 1)
    for n in range(n_levels):
        plt.plot(t_plot, populations_py[:, n], label=f'n={n}')
    plt.title('Python Implementation: Population Dynamics')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.grid(True)
    plt.legend()
    
    # C++実装の結果
    plt.subplot(2, 1, 2)
    for n in range(n_levels):
        plt.plot(t_plot, populations_cpp[:, n], label=f'n={n}')
    plt.title('C++ Implementation: Population Dynamics')
    plt.xlabel('Time')
    plt.ylabel('Population')
    plt.grid(True)
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(os.path.join(savepath, 'harmonic_oscillator_results.png'))
    plt.close()
    
    # 実装間の差の確認
    max_diff = np.max(np.abs(populations_py - populations_cpp))
    print(f"\nMaximum difference between implementations: {max_diff:.2e}")
    
    # エネルギー期待値の計算と保存の確認
    energies = hbar * omega * (np.arange(n_levels) + 0.5)
    E_py = np.sum(populations_py * energies[None, :], axis=1)
    E_cpp = np.sum(populations_cpp * energies[None, :], axis=1)
    
    plt.figure(figsize=(10, 6))
    plt.plot(t_plot[:populations_py.shape[0]], E_py, label='Python')
    plt.plot(t_plot[:populations_cpp.shape[0]], E_cpp, '--', label='C++')
    plt.title('Time Evolution of Energy Expectation Value')
    plt.xlabel('Time')
    plt.ylabel('Energy')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(savepath, 'harmonic_oscillator_energy.png'))
    plt.close()
    
    # パルス波形の表示
    plt.figure(figsize=(10, 6))
    plt.plot(t, Ex, label='Ex')
    plt.title('Excitation Pulse')
    plt.xlabel('Time')
    plt.ylabel('Field Amplitude')
    plt.grid(True)
    plt.legend()
    plt.savefig(os.path.join(savepath, 'harmonic_oscillator_pulse.png'))
    plt.close()

if __name__ == '__main__':
    main() 