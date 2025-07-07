"""C++実装のPythonインターフェース"""

import numpy as np
from scipy import sparse
from ._excitation_rk4_sparse import rk4_cpu_sparse_cpp

def rk4_cpu_sparse(
    H0: sparse.csr_matrix,
    mux: sparse.csr_matrix,
    muy: sparse.csr_matrix,
    psi0: np.ndarray,
    Ex: np.ndarray,
    Ey: np.ndarray,
    dt: float,
    steps: int,
    stride: int = 1,
    verbose: bool = False
) -> np.ndarray:
    """RK4法による時間発展計算（C++実装）
    
    Args:
        H0 (sparse.csr_matrix): ハミルトニアン
        mux (sparse.csr_matrix): x方向の双極子モーメント
        muy (sparse.csr_matrix): y方向の双極子モーメント
        psi0 (np.ndarray): 初期状態
        Ex (np.ndarray): x方向の電場
        Ey (np.ndarray): y方向の電場
        dt (float): 時間刻み
        steps (int): 総ステップ数
        stride (int, optional): 結果を保存する間隔. デフォルト1
        verbose (bool, optional): 詳細な出力を行うかどうか. デフォルト False
        
    Returns:
        np.ndarray: 時間発展後の状態ベクトル
    """
    # 入力チェック
    if not isinstance(H0, sparse.csr_matrix):
        H0 = sparse.csr_matrix(H0)
    if not isinstance(mux, sparse.csr_matrix):
        mux = sparse.csr_matrix(mux)
    if not isinstance(muy, sparse.csr_matrix):
        muy = sparse.csr_matrix(muy)
    
    # 配列の形状を確認
    if H0.shape != mux.shape or H0.shape != muy.shape:
        raise ValueError("All matrices must have the same shape")
    if H0.shape[0] != H0.shape[1]:
        raise ValueError("Matrices must be square")
    if psi0.shape[0] != H0.shape[0]:
        raise ValueError("Initial state dimension must match matrix dimension")
    if len(Ex) != len(Ey):
        raise ValueError("Electric field arrays must have the same length")
    
    # C++実装を呼び出し
    return rk4_cpu_sparse_cpp(
        H0.data, H0.indices, H0.indptr,
        mux.data, mux.indices, mux.indptr,
        muy.data, muy.indices, muy.indptr,
        psi0, Ex, Ey, dt, steps, stride,
        verbose
    ) 