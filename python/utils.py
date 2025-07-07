"""テスト用のユーティリティ関数を提供するモジュール"""

import numpy as np
from scipy import sparse

def create_test_matrices(n: int):
    """テスト用の行列を生成
    
    Args:
        n (int): 行列のサイズ
        
    Returns:
        tuple: (H0, mux, muy) - スパース行列のタプル
    """
    # H0: 対角要素のみを持つ行列
    H0_data = np.array([1.0])
    H0_row = np.array([0])
    H0_col = np.array([0])
    H0 = sparse.csr_matrix((H0_data, (H0_row, H0_col)), shape=(n, n))
    
    # mux: 非対角要素を持つ行列
    mux_data = np.array([0.1, 0.1])
    mux_row = np.array([0, 1])
    mux_col = np.array([1, 0])
    mux = sparse.csr_matrix((mux_data, (mux_row, mux_col)), shape=(n, n))
    
    # muy: ゼロ行列
    muy = sparse.csr_matrix((n, n))
    
    return H0, mux, muy

def create_test_pulse(steps: int):
    """テスト用のパルス波形を生成
    
    Args:
        steps (int): 時間ステップ数
        
    Returns:
        tuple: (Ex, Ey) - パルス波形の配列
    """
    # ガウシアンパルス
    t = np.linspace(0, 10, steps)
    sigma = 1.0
    t0 = 5.0
    
    Ex = np.exp(-(t - t0)**2 / (2 * sigma**2))
    Ey = np.zeros_like(Ex)
    
    return Ex, Ey 