from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

try:
    # C++ extension (may not be available in pure-Python envs)
    from . import _rk4_sparse_cpp as _c  # type: ignore
except Exception:  # pragma: no cover - keep import-time failure non-fatal
    _c = None  # type: ignore

try:
    from scipy.sparse import csr_matrix  # type: ignore
except Exception:  # pragma: no cover
    csr_matrix = object  # type: ignore


def rk4_sparse_eigen(
    H0: "csr_matrix",
    mux: "csr_matrix",
    muy: "csr_matrix",
    Ex: NDArray[np.float64],
    Ey: NDArray[np.float64],
    psi0: NDArray[np.complex128],
    dt: float,
    return_traj: bool = False,
    stride: int = 1,
    renorm: bool = False,
) -> NDArray[np.complex128]:
    """RK4 time propagation using sparse operators (Eigen backend).

    Parameters
    ----------
    H0, mux, muy : scipy.sparse.csr_matrix (complex128)
        Hamiltonian and dipole operators in CSR format.
    Ex, Ey : (T,) float64 ndarray
        Electric field components per time step.
    psi0 : (N,) complex128 ndarray
        Initial state vector.
    dt : float
        Time step.
    return_traj : bool, optional
        If True, returns trajectory; otherwise final state only.
    stride : int, optional
        Keep every `stride` steps when returning trajectory.
    renorm : bool, optional
        Renormalize state at each step to mitigate drift.

    Returns
    -------
    numpy.ndarray (complex128)
        Final state or trajectory depending on `return_traj`.
    """
    if _c is None:
        raise ImportError("C++ extension is not available. Build the module to use this function.")
    return _c.rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm)


def rk4_sparse_eigen_cached(
    H0: "csr_matrix",
    mux: "csr_matrix",
    muy: "csr_matrix",
    Ex: NDArray[np.float64],
    Ey: NDArray[np.float64],
    psi0: NDArray[np.complex128],
    dt: float,
    return_traj: bool = False,
    stride: int = 1,
    renorm: bool = False,
) -> NDArray[np.complex128]:
    """RK4 propagation with cached pattern construction for faster runs.

    See `rk4_sparse_eigen` for parameter details.
    """
    if _c is None:
        raise ImportError("C++ extension is not available. Build the module to use this function.")
    return _c.rk4_sparse_eigen_cached(H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm)


def rk4_sparse_eigen_direct_csr(
    H0_data: NDArray[np.complex128],
    H0_indices: NDArray[np.int32],
    H0_indptr: NDArray[np.int32],
    mux_data: NDArray[np.complex128],
    mux_indices: NDArray[np.int32],
    mux_indptr: NDArray[np.int32],
    muy_data: NDArray[np.complex128],
    muy_indices: NDArray[np.int32],
    muy_indptr: NDArray[np.int32],
    Ex: NDArray[np.float64],
    Ey: NDArray[np.float64],
    psi0: NDArray[np.complex128],
    dt: float,
    return_traj: bool = False,
    stride: int = 1,
    renorm: bool = False,
) -> NDArray[np.complex128]:
    """RK4 propagation from raw CSR components to minimize conversion overhead.

    Parameters
    ----------
    H0_*, mux_*, muy_* : CSR components
        `.data (complex128)`, `.indices (int32)`, `.indptr (int32)` for each operator.
    Ex, Ey : (T,) float64 ndarray
        Electric field components per time step.
    psi0 : (N,) complex128 ndarray
        Initial state vector.
    dt, return_traj, stride, renorm : see `rk4_sparse_eigen`.
    """
    if _c is None:
        raise ImportError("C++ extension is not available. Build the module to use this function.")
    return _c.rk4_sparse_eigen_direct_csr(
        H0_data, H0_indices, H0_indptr,
        mux_data, mux_indices, mux_indptr,
        muy_data, muy_indices, muy_indptr,
        Ex, Ey, psi0,
        dt, return_traj, stride, renorm,
    )


def rk4_sparse_suitesparse(
    H0: "csr_matrix",
    mux: "csr_matrix",
    muy: "csr_matrix",
    Ex: NDArray[np.float64],
    Ey: NDArray[np.float64],
    psi0: NDArray[np.complex128],
    dt: float,
    return_traj: bool = False,
    stride: int = 1,
    renorm: bool = False,
    level: int = 1,
) -> NDArray[np.complex128]:
    """RK4 propagation using OpenBLAS + SuiteSparse backend (if available).

    `level` controls optimization level (0: BASIC, 1: STANDARD, 2: ENHANCED).
    See `rk4_sparse_eigen` for parameter details.
    """
    if _c is None or not hasattr(_c, "rk4_sparse_suitesparse"):
        raise ImportError("SuiteSparse backend is not available in this build.")
    return _c.rk4_sparse_suitesparse(H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm, level)


def benchmark_implementations(
    H0: "csr_matrix",
    mux: "csr_matrix",
    muy: "csr_matrix",
    Ex: NDArray[np.float64],
    Ey: NDArray[np.float64],
    psi0: NDArray[np.complex128],
    dt: float,
    num_steps: int,
    return_traj: bool = False,
    stride: int = 1,
    renorm: bool = False,
):
    """Run C++-side benchmark harness across implementations.

    Returns backend-defined performance metrics object or ndarray depending on build.
    """
    if _c is None or not hasattr(_c, "benchmark_implementations"):
        raise ImportError("benchmark_implementations is not available in this build.")
    return _c.benchmark_implementations(H0, mux, muy, Ex, Ey, psi0, dt, num_steps, return_traj, stride, renorm)


def get_omp_max_threads() -> int:
    """Get the maximum number of OpenMP threads configured for the C++ backend."""
    if _c is None or not hasattr(_c, "get_omp_max_threads"):
        return 1
    return _c.get_omp_max_threads()


__all__ = [
    "rk4_sparse_eigen",
    "rk4_sparse_eigen_cached",
    "rk4_sparse_eigen_direct_csr",
    "rk4_sparse_suitesparse",
    "benchmark_implementations",
    "get_omp_max_threads",
]

