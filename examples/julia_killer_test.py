#!/usr/bin/env python3
"""
Julia Killer Phase 1テスト - JuliaのSIMD実装を上回るC++実装のテスト
"""

import numpy as np
import time
import sys
import os

# プロジェクトのパスを追加
project_root = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'python')
sys.path.insert(0, project_root)

try:
    # C++実装のインポート
    import rk4_sparse._rk4_sparse_cpp as erk4
    print("✅ Julia Killer C++実装が利用可能です")
    print(f"利用可能な関数: julia_killer_rk4_phase1 = {'julia_killer_rk4_phase1' in dir(erk4)}")
except ImportError as e:
    print(f"❌ Julia Killer C++実装が見つかりません: {e}")
    sys.exit(1)

def create_test_matrices(dim: int):
    """テスト用のスパース行列を生成"""
    from scipy.sparse import csr_matrix
    
    # ハミルトニアン: 対角成分 E_n = n * hbar * omega
    H0_data = []
    H0_row = []
    H0_col = []
    
    for i in range(dim):
        H0_data.append(complex(i * 1.0, 0))  # エネルギー準位
        H0_row.append(i)
        H0_col.append(i)
        
    # 双極子行列: 最近接遷移のみ
    mux_data = []
    mux_row = []
    mux_col = []
    
    muy_data = []
    muy_row = []
    muy_col = []
    
    for i in range(dim - 1):
        # X方向双極子モーメント
        mux_data.extend([complex(np.sqrt(i+1), 0), complex(np.sqrt(i+1), 0)])
        mux_row.extend([i, i+1])
        mux_col.extend([i+1, i])
        
        # Y方向双極子モーメント  
        muy_data.extend([complex(0, np.sqrt(i+1)), complex(0, -np.sqrt(i+1))])
        muy_row.extend([i, i+1])
        muy_col.extend([i+1, i])
    
    # CSR形式のスパース行列として生成
    H0 = csr_matrix((H0_data, (H0_row, H0_col)), shape=(dim, dim), dtype=complex)
    mux = csr_matrix((mux_data, (mux_row, mux_col)), shape=(dim, dim), dtype=complex)
    muy = csr_matrix((muy_data, (muy_row, muy_col)), shape=(dim, dim), dtype=complex)
    
    return H0, mux, muy

def create_test_fields(steps: int):
    """テスト用の電場を生成"""
    # ガウシアンパルス
    t = np.linspace(0, 10, steps)
    omega = 1.0
    sigma = 2.0
    amplitude = 0.1
    
    Ex = amplitude * np.exp(-(t - 5)**2 / (2 * sigma**2)) * np.cos(omega * t)
    Ey = amplitude * np.exp(-(t - 5)**2 / (2 * sigma**2)) * np.sin(omega * t)
    
    return Ex, Ey

def run_benchmark():
    """Julia Killer Phase 1実装のベンチマークを実行"""
    print("🚀 Julia Killer Phase 1: SIMD最適化ベンチマーク")
    print("=" * 60)
    
    # テスト次元と条件
    dimensions = [16, 32, 64, 128, 256, 512, 1024]
    steps = 501  # 奇数にして3点セットが正確に作れるようにする
    dt = 0.01
    
    results = {}
    
    for dim in dimensions:
        print(f"\n📊 次元数 {dim} のテスト:")
        print(f"   時間ステップ数: {steps}")
        print(f"   時間刻み: {dt}")
        
        # テストシステムの作成
        print(f"🔧 次元数 {dim} のテストシステムを作成中...")
        H0, mux, muy = create_test_matrices(dim)
        Ex, Ey = create_test_fields(steps)
        
        print(f"   H0: {H0.nnz} 非ゼロ要素")
        print(f"   mux: {mux.nnz} 非ゼロ要素")
        print(f"   muy: {muy.nnz} 非ゼロ要素")
        
        # 初期状態（基底状態）
        psi0 = np.zeros(dim, dtype=complex)
        psi0[0] = 1.0
        
        # Julia Killer Phase 1テスト（3回実行して平均を取る）
        print(f"⏱️  Julia Killer Phase 1 をベンチマーク中...")
        times = []
        
        for run in range(3):
            start_time = time.time()
            try:
                result = erk4.julia_killer_rk4_phase1(
                    H0, mux, muy,
                    Ex, Ey,
                    psi0,
                    dt,
                    True,  # return_traj
                    1,      # stride
                    False   # renorm
                )
                elapsed = time.time() - start_time
                times.append(elapsed)
                print(f"   実行 {run+1}/3... {elapsed:.3f}s")
            except Exception as e:
                print(f"   ❌ エラー: {e}")
                times.append(float('inf'))
        
        if all(t < float('inf') for t in times):
            avg_time = np.mean(times)
            std_time = np.std(times)
            results[dim] = {'time': avg_time, 'std': std_time}
            print(f"   結果: {avg_time:.4f} ± {std_time:.4f} s")
        else:
            print(f"   結果: エラーのため測定不可")
    
    # 結果のまとめ
    print(f"\n📈 Julia Killer Phase 1 ベンチマーク結果:")
    print("-" * 60)
    print(f"{'次元数':<8} {'時間 (s)':<12} {'性能指標':<12}")
    print("-" * 60)
    
    for dim in dimensions:
        if dim in results:
            time_ms = results[dim]['time'] * 1000
            std_ms = results[dim]['std'] * 1000
            
            # Julia風の性能指標 (steps/ms)
            perf = steps / time_ms
            print(f"{dim:<8} {time_ms:.2f}±{std_ms:.2f} ms  {perf:.2f} steps/ms")
        else:
            print(f"{dim:<8} エラー              -")
    
    print("\n🎯 Julia比較（参考値）:")
    julia_times = {
        32: 0.218,    # Julia SIMD実装の参考値 (ms)
        64: 0.5,      
        128: 1.2,
        256: 2.8,
        512: 5.1,
        1024: 6.360
    }
    
    print("-" * 60)
    print(f"{'次元数':<8} {'C++ (ms)':<12} {'Julia (ms)':<12} {'比率':<12}")
    print("-" * 60)
    
    for dim in dimensions:
        if dim in results and dim in julia_times:
            cpp_time = results[dim]['time'] * 1000
            julia_time = julia_times[dim]
            ratio = cpp_time / julia_time
            
            status = "🔥" if ratio < 1.0 else "⚡" if ratio < 1.5 else "🐌"
            print(f"{dim:<8} {cpp_time:.2f}        {julia_time:.3f}       {ratio:.2f}x {status}")
    
    print("\n✅ Julia Killer Phase 1 ベンチマーク完了！")

if __name__ == "__main__":
    run_benchmark() 