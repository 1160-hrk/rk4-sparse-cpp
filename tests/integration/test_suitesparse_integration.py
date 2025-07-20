#!/usr/bin/env python3
"""
SuiteSparse-MKL版の統合テスト
"""

import pytest
import numpy as np
import scipy.sparse as sparse
from rk4_sparse import (
    rk4_sparse_eigen, 
    rk4_sparse_suitesparse, 
    benchmark_implementations,
    create_test_matrices, 
    create_test_pulse,
    OPENBLAS_SUITESPARSE_AVAILABLE
)


class TestSuiteSparseIntegration:
    """SuiteSparse統合テストクラス"""
    
    def test_implementations_availability(self):
        """利用可能な実装の確認"""
        print("=== 利用可能な実装の確認 ===")
        
        implementations = {
            "Eigen版": rk4_sparse_eigen,
            "SuiteSparse版": rk4_sparse_suitesparse,
            "ベンチマーク機能": benchmark_implementations
        }
        
        for name, impl in implementations.items():
            if impl is not None:
                print(f"✅ {name}: 利用可能")
            else:
                print(f"❌ {name}: 利用不可")
        
        # 少なくともEigen実装は利用可能であることを確認
        assert rk4_sparse_eigen is not None
    
    def test_basic_functionality_comparison(self):
        """両実装の動作確認とベンチマーク"""
        if not OPENBLAS_SUITESPARSE_AVAILABLE:
            pytest.skip("SuiteSparse実装が利用できません")
        
        print("=== SuiteSparse-MKL版テスト ===")
        
        # テストデータの準備
        n = 100  # 行列サイズ
        steps = 1000  # 時間ステップ数
        
        print(f"行列サイズ: {n}x{n}")
        print(f"時間ステップ数: {steps}")
        
        # テスト行列の生成
        H0, mux, muy = create_test_matrices(n)
        
        # テストパルスの生成
        Ex, Ey = create_test_pulse(steps * 2 + 1)  # 奇数長にする
        
        # 初期状態
        psi0 = np.zeros(n, dtype=np.complex128)
        psi0[0] = 1.0
        
        # パラメータ
        dt = 0.01
        return_traj = True
        stride = 10
        renorm = False
        
        print("\n--- Eigen版の実行 ---")
        try:
            result_eigen = rk4_sparse_eigen(
                H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm
            )
            print(f"Eigen版成功: 結果サイズ = {result_eigen.shape}")
            
            # 結果の検証
            assert result_eigen.shape[0] == n
            assert result_eigen.dtype == np.complex128
            assert np.all(np.isfinite(result_eigen))
            
        except Exception as e:
            print(f"Eigen版エラー: {e}")
            pytest.fail(f"Eigen implementation failed: {e}")
        
        print("\n--- SuiteSparse-MKL版の実行 ---")
        try:
            result_suitesparse = rk4_sparse_suitesparse(
                H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm
            )
            print(f"SuiteSparse-MKL版成功: 結果サイズ = {result_suitesparse.shape}")
            
            # 結果の検証
            assert result_suitesparse.shape[0] == n
            assert result_suitesparse.dtype == np.complex128
            assert np.all(np.isfinite(result_suitesparse))
            
        except Exception as e:
            print(f"SuiteSparse-MKL版エラー: {e}")
            pytest.fail(f"SuiteSparse implementation failed: {e}")
        
        # 結果の比較
        print("\n--- 結果の比較 ---")
        diff = np.abs(result_eigen - result_suitesparse).max()
        print(f"最大差分: {diff}")
        
        if diff < 1e-10:
            print("✅ 両実装の結果が一致しました")
        else:
            print("❌ 結果に差異があります")
            pytest.fail(f"Results do not match. Max difference: {diff}")
    
    def test_benchmark_execution(self):
        """ベンチマーク実行のテスト"""
        if not OPENBLAS_SUITESPARSE_AVAILABLE:
            pytest.skip("ベンチマーク機能が利用できません")
        
        print("\n--- ベンチマーク実行 ---")
        
        # テストデータの準備
        n = 50  # 小さめのサイズでベンチマーク
        H0, mux, muy = create_test_matrices(n)
        Ex, Ey = create_test_pulse(100)
        psi0 = np.zeros(n, dtype=np.complex128)
        psi0[0] = 1.0
        
        dt = 0.01
        return_traj = True
        stride = 5
        renorm = False
        
        try:
            results = benchmark_implementations(
                H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm, num_runs=3
            )
            
            print("\nベンチマーク結果:")
            for result in results:
                print(f"  {result.implementation}: {result.total_time:.6f}秒 (Eigen比: {result.speedup_vs_eigen:.3f}x)")
            
            # 結果の検証
            assert len(results) > 0
            
            # 各実装の結果を確認
            implementations = [result.implementation for result in results]
            assert "Eigen" in implementations
            
            # 時間が正の値であることを確認
            for result in results:
                assert result.total_time > 0
                assert result.speedup_vs_eigen > 0
                
        except Exception as e:
            print(f"ベンチマークエラー: {e}")
            pytest.fail(f"Benchmark failed: {e}")
    
    def test_different_parameters(self):
        """異なるパラメータでのテスト"""
        if not OPENBLAS_SUITESPARSE_AVAILABLE:
            pytest.skip("SuiteSparse実装が利用できません")
        
        print("=== 異なるパラメータでのテスト ===")
        
        n = 30
        H0, mux, muy = create_test_matrices(n)
        Ex, Ey = create_test_pulse(50)
        psi0 = np.zeros(n, dtype=np.complex128)
        psi0[0] = 1.0
        
        # 異なるstride値でのテスト
        strides = [1, 2, 5, 10]
        
        for stride in strides:
            print(f"  stride = {stride}")
            
            result_eigen = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, 0.01, True, stride, False)
            result_suitesparse = rk4_sparse_suitesparse(H0, mux, muy, Ex, Ey, psi0, 0.01, True, stride, False)
            
            # 結果の一致を確認
            assert result_eigen.shape == result_suitesparse.shape
            diff = np.abs(result_eigen - result_suitesparse).max()
            assert diff < 1e-10
            
            print(f"    ✓ 結果一致 (差分: {diff:.2e})")
    
    def test_renorm_options(self):
        """正規化オプションでのテスト"""
        if not OPENBLAS_SUITESPARSE_AVAILABLE:
            pytest.skip("SuiteSparse実装が利用できません")
        
        print("=== 正規化オプションでのテスト ===")
        
        n = 20
        H0, mux, muy = create_test_matrices(n)
        Ex, Ey = create_test_pulse(30)
        psi0 = np.zeros(n, dtype=np.complex128)
        psi0[0] = 1.0
        
        # renorm=False
        result_eigen_no_renorm = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, False)
        result_suitesparse_no_renorm = rk4_sparse_suitesparse(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, False)
        
        diff_no_renorm = np.abs(result_eigen_no_renorm - result_suitesparse_no_renorm).max()
        assert diff_no_renorm < 1e-10
        print(f"  renorm=False: ✓ 結果一致 (差分: {diff_no_renorm:.2e})")
        
        # renorm=True
        result_eigen_renorm = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, True)
        result_suitesparse_renorm = rk4_sparse_suitesparse(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, True)
        
        diff_renorm = np.abs(result_eigen_renorm - result_suitesparse_renorm).max()
        assert diff_renorm < 1e-10
        print(f"  renorm=True: ✓ 結果一致 (差分: {diff_renorm:.2e})")
    
    def test_large_scale_comparison(self):
        """大規模問題での比較"""
        if not OPENBLAS_SUITESPARSE_AVAILABLE:
            pytest.skip("SuiteSparse実装が利用できません")
        
        print("=== 大規模問題での比較 ===")
        
        n = 200  # 大規模問題
        H0, mux, muy = create_test_matrices(n)
        Ex, Ey = create_test_pulse(100)
        psi0 = np.zeros(n, dtype=np.complex128)
        psi0[0] = 1.0
        
        print(f"  行列サイズ: {n}x{n}")
        print(f"  非零要素数: H0={H0.nnz}, mux={mux.nnz}, muy={muy.nnz}")
        
        # 両実装で実行
        result_eigen = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, False)
        result_suitesparse = rk4_sparse_suitesparse(H0, mux, muy, Ex, Ey, psi0, 0.01, True, 1, False)
        
        # 結果の一致を確認
        diff = np.abs(result_eigen - result_suitesparse).max()
        assert diff < 1e-10
        print(f"  ✓ 結果一致 (差分: {diff:.2e})")
        
        # 結果が有限値であることを確認
        assert np.all(np.isfinite(result_eigen))
        assert np.all(np.isfinite(result_suitesparse))
        print("  ✓ 数値安定性確認済み")
    
    def test_complete_integration(self):
        """完全な統合テスト"""
        print("=== 完全な統合テスト ===")
        
        # 利用可能な実装の確認
        self.test_implementations_availability()
        
        # 基本的な機能比較
        self.test_basic_functionality_comparison()
        
        # ベンチマーク実行
        self.test_benchmark_execution()
        
        # 異なるパラメータでのテスト
        self.test_different_parameters()
        
        # 正規化オプションでのテスト
        self.test_renorm_options()
        
        # 大規模問題での比較
        self.test_large_scale_comparison()
        
        print("\n=== 統合テスト完了 ===")
        print("✅ すべてのテストが正常に完了しました") 