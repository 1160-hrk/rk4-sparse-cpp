#!/usr/bin/env python3
"""
BLAS最適化版のRK4実装のテスト
8192次元での性能改善を確認
"""

import numpy as np
import scipy.sparse as sp
import time
import sys
import os
# sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
# sys.path.append(os.path.join(os.path.dirname(__file__), '../../python'))
# rk4_sparse_cppモジュールをインポート
try:
    from rk4_sparse._rk4_sparse_cpp import rk4_sparse_blas_optimized, rk4_sparse_eigen
    print("✓ rk4_sparse_cppモジュールのインポート成功")
except ImportError as e:
    print(f"✗ rk4_sparse_cppモジュールのインポート失敗: {e}")
    sys.exit(1)

def create_test_matrices(dim):
    """テスト用のスパース行列を作成"""
    print(f"  {dim}次元のテスト行列を作成中...")
    
    # 対角成分
    diag_indices = np.arange(dim)
    diag_data = np.random.random(dim) + 1j * np.random.random(dim)
    
    # 非対角成分（近接相互作用）
    off_diag_data = []
    off_diag_row = []
    off_diag_col = []
    
    for i in range(dim):
        # 隣接する要素との相互作用
        if i > 0:
            off_diag_row.append(i)
            off_diag_col.append(i-1)
            off_diag_data.append(0.1 * (np.random.random() + 1j * np.random.random()))
        
        if i < dim - 1:
            off_diag_row.append(i)
            off_diag_col.append(i+1)
            off_diag_data.append(0.1 * (np.random.random() + 1j * np.random.random()))
    
    # 行列の構築
    row_indices = np.concatenate([diag_indices, off_diag_row])
    col_indices = np.concatenate([diag_indices, off_diag_col])
    data = np.concatenate([diag_data, off_diag_data])
    
    # スパース行列の作成
    H0 = sp.csr_matrix((data, (row_indices, col_indices)), shape=(dim, dim))
    mux = sp.csr_matrix((data * 0.1, (row_indices, col_indices)), shape=(dim, dim))
    muy = sp.csr_matrix((data * 0.1, (row_indices, col_indices)), shape=(dim, dim))
    
    return H0, mux, muy

def benchmark_blas_optimization():
    """BLAS最適化版のベンチマーク"""
    print("=== BLAS最適化版のベンチマーク ===")
    
    # テストパラメータ
    dimensions = [1024, 2048, 4096, 8192, 16384]
    num_steps = 100
    dt = 0.01
    
    results = {}
    
    for dim in dimensions:
        print(f"\n--- {dim}次元でのテスト ---")
        
        # テスト行列の作成
        H0, mux, muy = create_test_matrices(dim)
        
        # 電場と初期状態
        Ex = np.random.random(num_steps) * 0.1
        Ey = np.random.random(num_steps) * 0.1
        psi0 = np.random.random(dim) + 1j * np.random.random(dim)
        psi0 = psi0 / np.linalg.norm(psi0)
        
        # 標準版（Eigen）のベンチマーク
        print("  標準版（Eigen）を実行中...")
        start_time = time.time()
        try:
            result_eigen = rk4_sparse_eigen(
                H0, mux, muy, Ex, Ey, psi0, dt, 
                return_traj=False, stride=1, renorm=False
            )
            eigen_time = time.time() - start_time
            print(f"    実行時間: {eigen_time:.4f}秒")
        except Exception as e:
            print(f"    エラー: {e}")
            eigen_time = float('inf')
        
        # BLAS最適化版のベンチマーク
        print("  BLAS最適化版を実行中...")
        start_time = time.time()
        try:
            result_blas = rk4_sparse_blas_optimized(
                H0, mux, muy, Ex, Ey, psi0, dt, 
                return_traj=False, stride=1, renorm=False
            )
            blas_time = time.time() - start_time
            print(f"    実行時間: {blas_time:.4f}秒")
        except Exception as e:
            print(f"    エラー: {e}")
            blas_time = float('inf')
        
        # 結果の比較
        if eigen_time != float('inf') and blas_time != float('inf'):
            speedup = eigen_time / blas_time
            print(f"    高速化率: {speedup:.2f}x")
            
            # 結果の精度チェック
            if 'result_eigen' in locals() and 'result_blas' in locals():
                if result_eigen.shape == result_blas.shape:
                    diff = np.linalg.norm(result_eigen - result_blas)
                    print(f"    結果の差: {diff:.2e}")
                    if diff < 1e-10:
                        print("    ✓ 結果は一致")
                    else:
                        print("    ⚠ 結果に差があります")
                else:
                    print("    ⚠ 結果の形状が異なります")
            else:
                print("    ⚠ 結果の比較ができません")
        else:
            speedup = 0.0
            print("    ⚠ 比較できません")
        
        results[dim] = {
            'eigen_time': eigen_time,
            'blas_time': blas_time,
            'speedup': speedup
        }
    
    # 結果の要約
    print("\n=== ベンチマーク結果の要約 ===")
    print("次元\t\t標準版(秒)\tBLAS版(秒)\t高速化率")
    print("-" * 50)
    for dim in dimensions:
        result = results[dim]
        if result['eigen_time'] != float('inf') and result['blas_time'] != float('inf'):
            print(f"{dim}\t\t{result['eigen_time']:.4f}\t\t{result['blas_time']:.4f}\t\t{result['speedup']:.2f}x")
        else:
            print(f"{dim}\t\tエラー\t\tエラー\t\t-")
    
    return results

def test_8192_dimension_specific():
    """8192次元での詳細テスト"""
    print("\n=== 8192次元での詳細テスト ===")
    
    dim = 8192
    num_steps = 1000
    dt = 0.01
    
    print(f"  {dim}次元、{num_steps}ステップでのテスト")
    
    # テスト行列の作成
    H0, mux, muy = create_test_matrices(dim)
    
    # 電場と初期状態
    Ex = np.random.random(num_steps) * 0.1
    Ey = np.random.random(num_steps) * 0.1
    psi0 = np.random.random(dim) + 1j * np.random.random(dim)
    psi0 = psi0 / np.linalg.norm(psi0)
    
    # 複数回実行して平均を取る
    num_runs = 5
    eigen_times = []
    blas_times = []
    
    for run in range(num_runs):
        print(f"  実行 {run + 1}/{num_runs}")
        
        # 標準版
        start_time = time.time()
        result_eigen = rk4_sparse_eigen(
            H0, mux, muy, Ex, Ey, psi0, dt, 
            return_traj=False, stride=1, renorm=False
        )
        eigen_times.append(time.time() - start_time)
        
        # BLAS最適化版
        start_time = time.time()
        result_blas = rk4_sparse_blas_optimized(
            H0, mux, muy, Ex, Ey, psi0, dt, 
            return_traj=False, stride=1, renorm=False
        )
        blas_times.append(time.time() - start_time)
    
    # 統計
    avg_eigen = np.mean(eigen_times)
    avg_blas = np.mean(blas_times)
    std_eigen = np.std(eigen_times)
    std_blas = np.std(blas_times)
    
    print(f"\n  標準版（Eigen）:")
    print(f"    平均時間: {avg_eigen:.4f} ± {std_eigen:.4f}秒")
    print(f"  BLAS最適化版:")
    print(f"    平均時間: {avg_blas:.4f} ± {std_blas:.4f}秒")
    print(f"  高速化率: {avg_eigen / avg_blas:.2f}x")
    
    # 結果の精度チェック
    if 'result_eigen' in locals() and 'result_blas' in locals():
        diff = np.linalg.norm(result_eigen - result_blas)
        print(f"  結果の差: {diff:.2e}")
        if diff < 1e-10:
            print("  ✓ 結果は一致")
        else:
            print("  ⚠ 結果に差があります")
    else:
        print("  ⚠ 結果の比較ができません")

def main():
    """メイン関数"""
    print("BLAS最適化版のRK4実装テスト")
    print("=" * 50)
    
    # システム情報の表示
    print(f"Python: {sys.version}")
    print(f"NumPy: {np.__version__}")
    import scipy
    print(f"SciPy: {scipy.__version__}")
    
    # OpenMP情報の確認
    try:
        from rk4_sparse._rk4_sparse_cpp import get_omp_max_threads
        max_threads = get_omp_max_threads()
        print(f"✓ OpenMP対応版がビルドされています（最大スレッド数: {max_threads}）")
    except Exception as e:
        print(f"⚠ OpenMP情報を取得できません: {e}")
    
    # BLAS情報の確認
    try:
        blas_info = np.__config__.show()
        print("BLAS情報:")
        print(blas_info)
    except Exception as e:
        print(f"⚠ BLAS情報を取得できません: {e}")
    
    # ベンチマーク実行
    results = benchmark_blas_optimization()
    
    # 8192次元での詳細テスト
    test_8192_dimension_specific()
    
    print("\n=== テスト完了 ===")
    print("BLAS最適化版の実装が完了しました。")
    print("8192次元での性能改善が期待されます。")

if __name__ == "__main__":
    main() 