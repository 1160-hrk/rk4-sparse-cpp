# _rk4_schrodinger.py  ----------------------------------------------
"""
4-th order Runge–Kutta propagator
=================================
* backend="numpy"  →  CPU  (NumPy / Numba)
* backend="cupy"   →  GPU  (CuPy RawKernel)

電場配列は奇数・偶数どちらの長さでも OK。
"""

from __future__ import annotations

from typing import Literal, Union

import numpy as np
from scipy.sparse import csr_matrix

try:
    from numba import njit
except ImportError:
    njit = lambda func: func




def rk4_sparse_py(
    H0: Union[csr_matrix, np.ndarray],
    mux: Union[csr_matrix, np.ndarray],
    muy: Union[csr_matrix, np.ndarray],
    Ex: np.ndarray,
    Ey: np.ndarray,
    psi0: np.ndarray,
    dt: float,
    return_traj: bool,
    stride: int,
    renorm: bool = False,
) -> np.ndarray:
    """
    4th-order Runge-Kutta propagator with sparse matrices using precomputed pattern.

    Parameters
    ----------
    H0, mux, muy : csr_matrix
        Hamiltonian and dipole operators (sparse)
    Ex3, Ey3 : (steps, 3) ndarray
        Electric field triplets
    psi0 : (dim,) ndarray
        Initial wavefunction
    dt : float
        Time step
    steps : int
        Number of time steps
    stride : int
        Output stride
    renorm : bool
        Normalize wavefunction at each step

    Returns
    -------
    out : (n_out, dim) ndarray
        Time evolution
    """
    if not isinstance(H0, csr_matrix):
        H0 = csr_matrix(H0)
    if not isinstance(mux, csr_matrix):
        mux = csr_matrix(mux)
    if not isinstance(muy, csr_matrix):
        muy = csr_matrix(muy)

    steps = (Ex.size - 1) // 2
    def _field_to_triplets(field: np.ndarray) -> np.ndarray:
        """
        奇数長 → そのまま
        偶数長 → 末尾 1 点をバッサリ捨てる
        """
        ex1 = field[0:-2:2]
        ex2 = field[1:-1:2]
        ex4 = field[2::2]
        return np.column_stack((ex1, ex2, ex4)).astype(np.float64, copy=False)
    Ex3 = _field_to_triplets(Ex)
    Ey3 = _field_to_triplets(Ey)

    psi = psi0.copy()
    dim = psi.size
    n_out = steps // stride + 1
    out = np.empty((n_out, dim), dtype=np.complex128)
    out[0] = psi
    idx = 1

    # 1️⃣ 共通パターン（構造のみ）を作成
    pattern = ((H0 != 0) + (mux != 0) + (muy != 0))
    pattern = pattern.astype(np.complex128)  # 確実に複素数
    pattern.data[:] = 1.0 + 0j
    pattern = pattern.tocsr()

    # 2️⃣ パターンに合わせてデータを展開
    def expand_to_pattern(matrix: csr_matrix, pattern: csr_matrix) -> np.ndarray:
        result_data = np.zeros_like(pattern.data, dtype=np.complex128)
        m_csr = matrix.tocsr()
        pi, pj = pattern.nonzero()
        m_dict = {(i, j): v for i, j, v in zip(*m_csr.nonzero(), m_csr.data)}
        for idx_, (i, j) in enumerate(zip(pi, pj)):
            result_data[idx_] = m_dict.get((i, j), 0.0 + 0j)
        return result_data

    H0_data = expand_to_pattern(H0, pattern)
    mux_data = expand_to_pattern(mux, pattern)
    muy_data = expand_to_pattern(muy, pattern)

    # 3️⃣ 計算用行列
    H = pattern.copy()

    # 4️⃣ 作業バッファ
    buf = np.empty_like(psi)
    k1 = np.empty_like(psi)
    k2 = np.empty_like(psi)
    k3 = np.empty_like(psi)
    k4 = np.empty_like(psi)

    for s in range(steps):
        ex1, ex2, ex4 = Ex3[s]
        ey1, ey2, ey4 = Ey3[s]

        # H1
        H.data[:] = H0_data + mux_data * ex1 + muy_data * ey1
        k1[:] = -1j * H.dot(psi)
        buf[:] = psi + 0.5 * dt * k1

        # H2
        H.data[:] = H0_data + mux_data * ex2 + muy_data * ey2
        k2[:] = -1j * H.dot(buf)
        buf[:] = psi + 0.5 * dt * k2

        # H3
        H.data[:] = H0_data + mux_data * ex2 + muy_data * ey2
        k3[:] = -1j * H.dot(buf)
        buf[:] = psi + dt * k3

        # H4
        H.data[:] = H0_data + mux_data * ex4 + muy_data * ey4
        k4[:] = -1j * H.dot(buf)

        # update
        psi += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if renorm:
            norm = np.sqrt((psi.conj() @ psi).real)
            psi /= norm

        if return_traj and (s + 1) % stride == 0:
            out[idx] = psi
            idx += 1

    if return_traj:
        return out
    else:
        return psi


@njit(
    "c16[:, :]("
    "c16[:, :], c16[:, :], c16[:, :],"
    "f8[:], f8[:], c16[:], f8, b1, i8, b1"
    ")",
    cache=True
    )
def rk4_numba_py(
    H0: np.ndarray,
    mux: np.ndarray,
    muy: np.ndarray,
    Ex: np.ndarray,
    Ey: np.ndarray,
    psi0: np.ndarray,
    dt: float,
    return_traj: bool,
    stride: int,
    renorm: bool = False,
) -> np.ndarray:
    """
    4th-order Runge-Kutta propagator with sparse matrices using precomputed pattern.

    Parameters
    ----------
    H0, mux, muy : csr_matrix
        Hamiltonian and dipole operators (sparse)
    Ex3, Ey3 : (steps, 3) ndarray
        Electric field triplets
    psi0 : (dim,) ndarray
        Initial wavefunction
    dt : float
        Time step
    steps : int
        Number of time steps
    stride : int
        Output stride
    renorm : bool
        Normalize wavefunction at each step

    Returns
    -------
    out : (n_out, dim) ndarray
        Time evolution
    """


    steps = (Ex.size - 1) // 2
    Ex3, Ey3 = np.zeros((steps, 3), dtype=np.float64), np.zeros((steps, 3), dtype=np.float64)
    Ex3[:, 0], Ey3[:, 0] = Ex[0:-2:2], Ey[0:-2:2]
    Ex3[:, 1], Ey3[:, 1] = Ex[1:-1:2], Ey[1:-1:2]
    Ex3[:, 2], Ey3[:, 2] = Ex[2::2], Ey[2::2]
    
    psi = psi0.copy()
    dim = psi.size
    n_out = steps // stride + 1
    out = np.empty((n_out, dim), dtype=np.complex128)
    out[0] = psi
    idx = 1


    # 4️⃣ 作業バッファ
    buf = np.empty_like(psi)
    k1 = np.empty_like(psi)
    k2 = np.empty_like(psi)
    k3 = np.empty_like(psi)
    k4 = np.empty_like(psi)

    for s in range(steps):
        ex1, ex2, ex4 = Ex3[s]
        ey1, ey2, ey4 = Ey3[s]

        # H1
        H = H0 + mux * ex1 + muy * ey1
        k1[:] = -1j * H.dot(psi)
        buf[:] = psi + 0.5 * dt * k1

        # H2
        H = H0 + mux * ex2 + muy * ey2
        k2[:] = -1j * H.dot(buf)
        buf[:] = psi + 0.5 * dt * k2

        # H3
        H = H0 + mux * ex2 + muy * ey2
        k3[:] = -1j * H.dot(buf)
        buf[:] = psi + dt * k3

        # H4
        H = H0 + mux * ex4 + muy * ey4
        k4[:] = -1j * H.dot(buf)

        # update
        psi += (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        if renorm:
            norm = np.sqrt((psi.conj() @ psi).real)
            psi /= norm

        if return_traj and (s + 1) % stride == 0:
            out[idx] = psi
            idx += 1

    if return_traj:
        return out
    else:
        return psi.reshape((1, dim))