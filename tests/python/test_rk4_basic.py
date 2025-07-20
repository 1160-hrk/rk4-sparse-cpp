#!/usr/bin/env python3
"""
基本的なRK4スパースソルバーのテスト
"""

import pytest
import numpy as np
import scipy.sparse as sp
from rk4_sparse import create_test_matrices, create_test_pulse


class TestBasicFunctionality:
    """基本的な機能のテスト"""
    
    def test_import_available(self):
        """必要なモジュールがインポートできることを確認"""
        # 基本的なインポートが可能であることを確認
        assert create_test_matrices is not None
        assert create_test_pulse is not None
    
    def test_test_matrices_creation(self):
        """テスト行列の作成が正常に動作することを確認"""
        dim = 5
        H0, mux, muy = create_test_matrices(dim)
        
        # 基本的な形状の確認
        assert H0.shape == (dim, dim)
        assert mux.shape == (dim, dim)
        assert muy.shape == (dim, dim)
        
        # スパース行列であることを確認
        assert sp.issparse(H0)
        assert sp.issparse(mux)
        assert sp.issparse(muy)
    
    def test_test_pulse_creation(self):
        """テストパルスの作成が正常に動作することを確認"""
        length = 10
        Ex, Ey = create_test_pulse(length)
        
        # 基本的な形状の確認
        assert len(Ex) == length
        assert len(Ey) == length
        
        # データ型の確認
        assert Ex.dtype == np.float64
        assert Ey.dtype == np.float64
    
    def test_simple_calculation(self):
        """簡単な計算が正常に動作することを確認"""
        dim = 3
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_pulse(5)
        
        # 簡単な行列-ベクトル積のテスト
        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0
        
        result = H0 @ psi0
        
        assert result.shape == (dim,)
        assert result.dtype == np.complex128
        assert np.all(np.isfinite(result))
    
    def test_physical_consistency(self):
        """物理的な一貫性の基本的な確認"""
        dim = 10
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_pulse(20)
        
        # ハミルトニアンがエルミート行列であることを確認
        H0_dense = H0.toarray()
        H0_hermitian = H0_dense.conj().T
        
        # 数値誤差を考慮して比較
        assert np.allclose(H0_dense, H0_hermitian, atol=1e-10)
        
        # 電場が非負であることを確認
        assert np.all(Ex >= 0)
        assert np.all(Ey >= 0)
    
    def test_edge_cases(self):
        """エッジケースのテスト"""
        # 最小の次元
        H0, mux, muy = create_test_matrices(1)
        assert H0.shape == (1, 1)
        
        # 最小のパルス長
        Ex, Ey = create_test_pulse(1)
        assert len(Ex) == 1
        assert len(Ey) == 1
    
    def test_data_types(self):
        """データ型の確認"""
        dim = 5
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_pulse(10)
        
        # 行列のデータ型
        assert H0.dtype == np.complex128
        assert mux.dtype == np.complex128
        assert muy.dtype == np.complex128
        
        # パルスのデータ型
        assert Ex.dtype == np.float64
        assert Ey.dtype == np.float64
