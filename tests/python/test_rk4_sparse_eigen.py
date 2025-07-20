#!/usr/bin/env python3
"""
Eigen実装のRK4スパースソルバーのテスト
"""

import pytest
import numpy as np
import scipy.sparse as sp
from rk4_sparse import rk4_sparse_eigen, create_test_matrices, create_test_pulse


class TestRK4SparseEigen:
    """Eigen実装のテストクラス"""
    
    def test_import(self):
        """Eigen実装がインポートできることを確認"""
        assert rk4_sparse_eigen is not None
    
    def test_basic_functionality(self):
        """基本的な機能テスト"""
        # テストデータの準備
        dim = 10
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_pulse(100)
        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0
        
        # パラメータ
        dt = 0.01
        return_traj = True
        stride = 1
        renorm = False
        
        # 実行
        result = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm)
        
        # 結果の検証
        assert result is not None
        assert result.shape[0] == dim
        assert result.shape[1] == len(Ex) // stride
    
    def test_different_dimensions(self):
        """異なる次元でのテスト"""
        dimensions = [5, 10, 20, 50]
        
        for dim in dimensions:
            H0, mux, muy = create_test_matrices(dim)
            Ex, Ey = create_test_pulse(50)
            psi0 = np.zeros(dim, dtype=np.complex128)
            psi0[0] = 1.0
            
            result = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, False)
            
            assert result.shape[0] == dim
            assert result.shape[1] == len(Ex)
    
    def test_different_stride_values(self):
        """異なるstride値でのテスト"""
        dim = 10
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_pulse(100)
        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0
        
        strides = [1, 2, 5, 10]
        
        for stride in strides:
            result = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, 0.01, True, stride, False)
            expected_steps = len(Ex) // stride
            assert result.shape[1] == expected_steps
    
    def test_return_traj_options(self):
        """return_trajオプションのテスト"""
        dim = 10
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_pulse(50)
        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0
        
        # return_traj=True
        result_traj = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, False)
        assert result_traj.shape[1] == len(Ex)
        
        # return_traj=False
        result_final = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, 0.01, False, 1, False)
        assert result_final.shape[1] == 1
    
    def test_renorm_options(self):
        """renormオプションのテスト"""
        dim = 10
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_pulse(50)
        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0
        
        # renorm=False
        result_no_renorm = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, False)
        
        # renorm=True
        result_renorm = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, True)
        
        # 結果が異なることを確認（正規化の影響）
        assert not np.allclose(result_no_renorm, result_renorm)
    
    def test_input_validation(self):
        """入力検証のテスト"""
        dim = 10
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_pulse(50)
        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0
        
        # 不正な次元のpsi0
        with pytest.raises(Exception):
            wrong_psi0 = np.zeros(dim + 1, dtype=np.complex128)
            rk4_sparse_eigen(H0, mux, muy, Ex, Ey, wrong_psi0, 0.01, True, 1, False)
        
        # 不正なdt
        with pytest.raises(Exception):
            rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, -0.01, True, 1, False)
    
    def test_numerical_stability(self):
        """数値安定性のテスト"""
        dim = 100
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_pulse(1000)
        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0
        
        # 小さなdtでの長時間シミュレーション
        result = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, 0.001, True, 1, False)
        
        # 結果が有限値であることを確認
        assert np.all(np.isfinite(result))
        assert not np.any(np.isnan(result))
        assert not np.any(np.isinf(result))
    
    def test_conservation_properties(self):
        """保存量のテスト"""
        dim = 10
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_pulse(100)
        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0
        
        result = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, False)
        
        # 確率の保存（renorm=Falseの場合）
        probabilities = np.sum(np.abs(result)**2, axis=0)
        assert np.allclose(probabilities, probabilities[0], atol=1e-10)
    
    def test_complex_input_handling(self):
        """複素数入力の処理テスト"""
        dim = 10
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_pulse(50)
        
        # 複素数の初期状態
        psi0 = np.random.rand(dim) + 1j * np.random.rand(dim)
        psi0 = psi0 / np.linalg.norm(psi0)  # 正規化
        
        result = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, False)
        
        assert result.dtype == np.complex128
        assert result.shape[0] == dim 