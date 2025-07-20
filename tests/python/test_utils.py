#!/usr/bin/env python3
"""
ユーティリティ関数のテスト
"""

import pytest
import numpy as np
import scipy.sparse as sp
from rk4_sparse import create_test_matrices, create_test_pulse


class TestCreateTestMatrices:
    """create_test_matrices関数のテスト"""
    
    def test_basic_functionality(self):
        """基本的な機能テスト"""
        dim = 10
        H0, mux, muy = create_test_matrices(dim)
        
        # 形状の確認
        assert H0.shape == (dim, dim)
        assert mux.shape == (dim, dim)
        assert muy.shape == (dim, dim)
        
        # スパース行列であることを確認
        assert sp.issparse(H0)
        assert sp.issparse(mux)
        assert sp.issparse(muy)
        
        # CSR形式であることを確認
        assert H0.format == 'csr'
        assert mux.format == 'csr'
        assert muy.format == 'csr'
    
    def test_different_dimensions(self):
        """異なる次元でのテスト"""
        dimensions = [5, 10, 20, 50, 100]
        
        for dim in dimensions:
            H0, mux, muy = create_test_matrices(dim)
            
            assert H0.shape == (dim, dim)
            assert mux.shape == (dim, dim)
            assert muy.shape == (dim, dim)
    
    def test_matrix_properties(self):
        """行列の性質のテスト"""
        dim = 20
        H0, mux, muy = create_test_matrices(dim)
        
        # 対角成分が存在することを確認
        assert H0.diagonal().any()
        assert mux.diagonal().any()
        assert muy.diagonal().any()
        
        # 非対角成分が存在することを確認（近接相互作用）
        off_diag_elements = H0 - sp.diags(H0.diagonal())
        assert off_diag_elements.nnz > 0
    
    def test_data_types(self):
        """データ型のテスト"""
        dim = 10
        H0, mux, muy = create_test_matrices(dim)
        
        # 複素数型であることを確認
        assert H0.dtype == np.complex128
        assert mux.dtype == np.complex128
        assert muy.dtype == np.complex128
    
    def test_hermitian_property(self):
        """エルミート性のテスト（H0はエルミート行列）"""
        dim = 10
        H0, _, _ = create_test_matrices(dim)
        
        # H0がエルミート行列であることを確認
        H0_dense = H0.toarray()
        H0_hermitian = H0_dense.conj().T
        
        # 数値誤差を考慮して比較
        assert np.allclose(H0_dense, H0_hermitian, atol=1e-10)
    
    def test_sparsity_pattern(self):
        """スパースパターンのテスト"""
        dim = 15
        H0, mux, muy = create_test_matrices(dim)
        
        # 非零要素の数が適切であることを確認
        # 対角成分 + 近接相互作用（各要素は最大2つの隣接要素と相互作用）
        expected_min_nnz = dim  # 対角成分
        expected_max_nnz = dim + 2 * (dim - 1)  # 対角成分 + 近接相互作用
        
        assert H0.nnz >= expected_min_nnz
        assert H0.nnz <= expected_max_nnz
        assert mux.nnz >= expected_min_nnz
        assert mux.nnz <= expected_max_nnz
        assert muy.nnz >= expected_min_nnz
        assert muy.nnz <= expected_max_nnz


class TestCreateTestPulse:
    """create_test_pulse関数のテスト"""
    
    def test_basic_functionality(self):
        """基本的な機能テスト"""
        length = 100
        Ex, Ey = create_test_pulse(length)
        
        # 形状の確認
        assert len(Ex) == length
        assert len(Ey) == length
        
        # データ型の確認
        assert Ex.dtype == np.float64
        assert Ey.dtype == np.float64
    
    def test_different_lengths(self):
        """異なる長さでのテスト"""
        lengths = [10, 50, 100, 500, 1000]
        
        for length in lengths:
            Ex, Ey = create_test_pulse(length)
            
            assert len(Ex) == length
            assert len(Ey) == length
    
    def test_pulse_properties(self):
        """パルスの性質のテスト"""
        length = 200
        Ex, Ey = create_test_pulse(length)
        
        # 値が有限であることを確認
        assert np.all(np.isfinite(Ex))
        assert np.all(np.isfinite(Ey))
        
        # 値が実数であることを確認
        assert np.all(np.isreal(Ex))
        assert np.all(np.isreal(Ey))
        
        # 値の範囲が適切であることを確認
        assert np.all(Ex >= 0)  # 通常、電場強度は正
        assert np.all(Ey >= 0)
    
    def test_pulse_shape(self):
        """パルス形状のテスト"""
        length = 100
        Ex, Ey = create_test_pulse(length)
        
        # パルスが時間的に変化することを確認
        # （完全に一定でないことを確認）
        assert not np.allclose(Ex, Ex[0])
        assert not np.allclose(Ey, Ey[0])
    
    def test_consistency(self):
        """一貫性のテスト"""
        length = 50
        
        # 同じ長さで複数回生成して一貫性を確認
        Ex1, Ey1 = create_test_pulse(length)
        Ex2, Ey2 = create_test_pulse(length)
        
        # 形状は同じ
        assert Ex1.shape == Ex2.shape
        assert Ey1.shape == Ey2.shape
        
        # 値は異なる（ランダム性）
        assert not np.allclose(Ex1, Ex2)
        assert not np.allclose(Ey1, Ey2)
    
    def test_edge_cases(self):
        """エッジケースのテスト"""
        # 最小の長さ
        Ex, Ey = create_test_pulse(1)
        assert len(Ex) == 1
        assert len(Ey) == 1
        
        # 大きな長さ
        Ex, Ey = create_test_pulse(10000)
        assert len(Ex) == 10000
        assert len(Ey) == 10000
    
    def test_input_validation(self):
        """入力検証のテスト"""
        # 負の長さ
        with pytest.raises(ValueError):
            create_test_pulse(-1)
        
        # ゼロの長さ
        with pytest.raises(ValueError):
            create_test_pulse(0)
        
        # 非整数の長さ
        with pytest.raises(TypeError):
            create_test_pulse(10.5)


class TestIntegration:
    """統合テスト"""
    
    def test_matrices_and_pulse_integration(self):
        """行列とパルスの統合テスト"""
        dim = 10
        pulse_length = 50
        
        # テストデータの生成
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_pulse(pulse_length)
        
        # 基本的な計算のテスト
        # 行列-ベクトル積が正常に動作することを確認
        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0
        
        # H0 * psi0 の計算
        result = H0 @ psi0
        
        assert result.shape == (dim,)
        assert result.dtype == np.complex128
        assert np.all(np.isfinite(result))
    
    def test_physical_consistency(self):
        """物理的な一貫性のテスト"""
        dim = 20
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_pulse(100)
        
        # ハミルトニアンのエルミート性を確認
        H0_dense = H0.toarray()
        H0_hermitian = H0_dense.conj().T
        
        assert np.allclose(H0_dense, H0_hermitian, atol=1e-10)
        
        # 電場の物理的な妥当性を確認
        assert np.all(Ex >= 0)  # 電場強度は非負
        assert np.all(Ey >= 0)
        
        # パルスの時間変化が滑らかであることを確認
        Ex_diff = np.diff(Ex)
        Ey_diff = np.diff(Ey)
        
        # 急激な変化がないことを確認
        assert np.all(np.abs(Ex_diff) < 1.0)
        assert np.all(np.abs(Ey_diff) < 1.0) 