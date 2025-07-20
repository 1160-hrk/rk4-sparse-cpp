#!/usr/bin/env python3
"""
実装間の比較テスト
"""

import pytest
import numpy as np
import scipy.sparse as sp
from rk4_sparse import (
    rk4_sparse_eigen,
    rk4_sparse_suitesparse,
    benchmark_implementations,
    create_test_matrices,
    create_test_pulse,
    OPENBLAS_SUITESPARSE_AVAILABLE
)


class TestImplementationComparison:
    """実装間の比較テスト"""
    
    def test_eigen_vs_suitesparse_accuracy(self):
        """Eigen実装とSuiteSparse実装の精度比較"""
        if not OPENBLAS_SUITESPARSE_AVAILABLE:
            pytest.skip("SuiteSparse実装が利用できません")
        
        # テストデータの準備
        dim = 20
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_pulse(100)
        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0
        
        # パラメータ
        dt = 0.01
        return_traj = True
        stride = 1
        renorm = False
        
        # 両実装で実行
        result_eigen = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm)
        result_suitesparse = rk4_sparse_suitesparse(H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm)
        
        # 結果の形状が一致することを確認
        assert result_eigen.shape == result_suitesparse.shape
        
        # 結果の精度を比較
        diff = np.abs(result_eigen - result_suitesparse).max()
        print(f"最大差分: {diff}")
        
        # 数値誤差の範囲内で一致することを確認
        assert diff < 1e-10
    
    def test_eigen_vs_suitesparse_different_parameters(self):
        """異なるパラメータでの比較"""
        if not OPENBLAS_SUITESPARSE_AVAILABLE:
            pytest.skip("SuiteSparse実装が利用できません")
        
        dim = 15
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_pulse(200)
        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0
        
        # 異なるstride値での比較
        strides = [1, 2, 5, 10]
        
        for stride in strides:
            result_eigen = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, 0.01, True, stride, False)
            result_suitesparse = rk4_sparse_suitesparse(H0, mux, muy, Ex, Ey, psi0, 0.01, True, stride, False)
            
            assert result_eigen.shape == result_suitesparse.shape
            diff = np.abs(result_eigen - result_suitesparse).max()
            assert diff < 1e-10
    
    def test_eigen_vs_suitesparse_renorm_options(self):
        """正規化オプションでの比較"""
        if not OPENBLAS_SUITESPARSE_AVAILABLE:
            pytest.skip("SuiteSparse実装が利用できません")
        
        dim = 10
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_pulse(50)
        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0
        
        # renorm=False
        result_eigen_no_renorm = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, False)
        result_suitesparse_no_renorm = rk4_sparse_suitesparse(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, False)
        
        diff_no_renorm = np.abs(result_eigen_no_renorm - result_suitesparse_no_renorm).max()
        assert diff_no_renorm < 1e-10
        
        # renorm=True
        result_eigen_renorm = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, True)
        result_suitesparse_renorm = rk4_sparse_suitesparse(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, True)
        
        diff_renorm = np.abs(result_eigen_renorm - result_suitesparse_renorm).max()
        assert diff_renorm < 1e-10
    
    def test_benchmark_results_consistency(self):
        """ベンチマーク結果の一貫性テスト"""
        if not OPENBLAS_SUITESPARSE_AVAILABLE:
            pytest.skip("ベンチマーク機能が利用できません")
        
        dim = 20
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_pulse(100)
        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0
        
        # ベンチマーク実行
        results = benchmark_implementations(
            H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, False, num_runs=3
        )
        
        # 結果の構造を確認
        assert len(results) > 0
        
        # 各実装の結果を確認
        implementations = [result.implementation for result in results]
        assert "Eigen" in implementations
        
        # 時間が正の値であることを確認
        for result in results:
            assert result.total_time > 0
            assert result.speedup_vs_eigen > 0
    
    def test_large_scale_comparison(self):
        """大規模問題での比較"""
        if not OPENBLAS_SUITESPARSE_AVAILABLE:
            pytest.skip("SuiteSparse実装が利用できません")
        
        dim = 100
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_pulse(500)
        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0
        
        # 大規模問題での実行
        result_eigen = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, False)
        result_suitesparse = rk4_sparse_suitesparse(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, False)
        
        # 結果の一致を確認
        diff = np.abs(result_eigen - result_suitesparse).max()
        assert diff < 1e-10
        
        # 結果が有限値であることを確認
        assert np.all(np.isfinite(result_eigen))
        assert np.all(np.isfinite(result_suitesparse))
    
    def test_numerical_stability_comparison(self):
        """数値安定性の比較"""
        if not OPENBLAS_SUITESPARSE_AVAILABLE:
            pytest.skip("SuiteSparse実装が利用できません")
        
        dim = 50
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_pulse(1000)
        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0
        
        # 小さなdtでの長時間シミュレーション
        dt = 0.001
        
        result_eigen = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, dt, True, 1, False)
        result_suitesparse = rk4_sparse_suitesparse(H0, mux, muy, Ex, Ey, psi0, dt, True, 1, False)
        
        # 両実装で数値安定性を確認
        assert np.all(np.isfinite(result_eigen))
        assert np.all(np.isfinite(result_suitesparse))
        assert not np.any(np.isnan(result_eigen))
        assert not np.any(np.isnan(result_suitesparse))
        assert not np.any(np.isinf(result_eigen))
        assert not np.any(np.isinf(result_suitesparse))
        
        # 結果の一致を確認
        diff = np.abs(result_eigen - result_suitesparse).max()
        assert diff < 1e-10
    
    def test_conservation_properties_comparison(self):
        """保存量の比較"""
        if not OPENBLAS_SUITESPARSE_AVAILABLE:
            pytest.skip("SuiteSparse実装が利用できません")
        
        dim = 15
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_pulse(200)
        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0
        
        # 両実装で実行
        result_eigen = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, False)
        result_suitesparse = rk4_sparse_suitesparse(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, False)
        
        # 確率の保存を確認（renorm=Falseの場合）
        prob_eigen = np.sum(np.abs(result_eigen)**2, axis=0)
        prob_suitesparse = np.sum(np.abs(result_suitesparse)**2, axis=0)
        
        # 初期確率との比較
        initial_prob = np.abs(psi0[0])**2
        
        assert np.allclose(prob_eigen, initial_prob, atol=1e-10)
        assert np.allclose(prob_suitesparse, initial_prob, atol=1e-10)
        
        # 両実装間で確率が一致することを確認
        assert np.allclose(prob_eigen, prob_suitesparse, atol=1e-10)
    
    def test_complex_input_handling_comparison(self):
        """複素数入力処理の比較"""
        if not OPENBLAS_SUITESPARSE_AVAILABLE:
            pytest.skip("SuiteSparse実装が利用できません")
        
        dim = 10
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_pulse(50)
        
        # 複素数の初期状態
        psi0 = np.random.rand(dim) + 1j * np.random.rand(dim)
        psi0 = psi0 / np.linalg.norm(psi0)  # 正規化
        
        # 両実装で実行
        result_eigen = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, False)
        result_suitesparse = rk4_sparse_suitesparse(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, False)
        
        # 結果の型と形状を確認
        assert result_eigen.dtype == np.complex128
        assert result_suitesparse.dtype == np.complex128
        assert result_eigen.shape == result_suitesparse.shape
        
        # 結果の一致を確認
        diff = np.abs(result_eigen - result_suitesparse).max()
        assert diff < 1e-10 