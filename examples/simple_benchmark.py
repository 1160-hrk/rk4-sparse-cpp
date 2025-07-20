#!/usr/bin/env python3
"""
簡易ベンチマークテスト
=====================

新しく追加されたJulia風実装の性能をテストします。
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'python'))

import numpy as np
import time
from scipy.sparse import csr_matrix

try:
    from rk4_sparse._rk4_sparse_cpp import (
        rk4_sparse_eigen,           # 従来実装
        rk4_sparse_julia_style,     # Julia風高速実装
        rk4_sparse_csr_optimized,   # CSR最適化実装
    )
    print("✅ C++実装が利用可能です")
    cpp_available = True
except ImportError as e:
    print(f"❌ C++実装が利用できません: {e}")
    cpp_available = False

def create_test_system(dim=512):
    """テスト用のハミルトニアンと双極子演算子を作成"""
    print(f"🔧 次元数 {dim} のテストシステムを作成中...")
    
    # 簡単な対角ハミルトニアン
    H0_diag = np.linspace(0, 1, dim)
    H0 = csr_matrix(np.diag(H0_diag), dtype=np.complex128)
    
    # 隣接要素間の結合
    off_diag = np.ones(dim-1) * 0.1
    mux = csr_matrix(np.diag(off_diag, k=1) + np.diag(off_diag, k=-1), dtype=np.complex128)
    
    # y方向は少し異なるパターン
    muy_data = np.ones(dim-2) * 0.05
    muy = csr_matrix(np.diag(muy_data, k=2) + np.diag(muy_data, k=-2), dtype=np.complex128)
    
    print(f"   H0: {H0.nnz} 非ゼロ要素")
    print(f"   mux: {mux.nnz} 非ゼロ要素")
    print(f"   muy: {muy.nnz} 非ゼロ要素")
    
    return H0, mux, muy

def create_test_fields(steps=500):
    """テスト用の電場を作成"""
    t = np.linspace(0, 10, 2*steps + 1)
    Ex = np.sin(t) * np.exp(-0.1 * t)
    Ey = np.cos(t) * np.exp(-0.1 * t)
    return Ex, Ey

def benchmark_function(func, name, H0, mux, muy, Ex, Ey, psi0, dt, num_runs=3):
    """実装の性能を測定"""
    if func is None:
        return None, 0.0
    
    print(f"⏱️  {name} をベンチマーク中...")
    
    times = []
    for run in range(num_runs):
        print(f"   実行 {run+1}/{num_runs}...", end='', flush=True)
        
        start = time.perf_counter()
        try:
            result = func(H0, mux, muy, Ex, Ey, psi0, dt, False, 1, False)
            end = time.perf_counter()
            times.append(end - start)
            print(f" {end - start:.3f}s")
        except Exception as e:
            print(f" ❌ エラー: {e}")
            return None, 0.0
    
    avg_time = np.mean(times)
    std_time = np.std(times)
    
    print(f"   結果: {avg_time:.4f} ± {std_time:.4f} s")
    return avg_time, std_time

def main():
    """メインベンチマーク"""
    print("🚀 Julia風高速実装の簡易ベンチマークテスト")
    print("=" * 60)
    
    if not cpp_available:
        print("C++実装が利用できないため、ベンチマークを終了します。")
        return
    
    # テストサイズの設定
    dim = 1024
    steps = 500
    
    print(f"\n📊 テスト条件:")
    print(f"   - 次元数: {dim}")
    print(f"   - 時間ステップ数: {steps}")
    print(f"   - 時間刻み: 0.01")
    
    # テストシステムの作成
    H0, mux, muy = create_test_system(dim)
    Ex, Ey = create_test_fields(steps)
    psi0 = np.zeros(dim, dtype=np.complex128)
    psi0[0] = 1.0
    dt = 0.01
    
    print(f"\n🏁 ベンチマーク実行:")
    print("-" * 40)
    
    results = {}
    
    # 各実装のベンチマーク
    implementations = [
        (rk4_sparse_eigen, "従来Eigen実装"),
        (rk4_sparse_julia_style, "Julia風高速実装"),
        (rk4_sparse_csr_optimized, "CSR最適化実装"),
    ]
    
    baseline_time = None
    for func, name in implementations:
        avg_time, std_time = benchmark_function(
            func, name, H0, mux, muy, Ex, Ey, psi0, dt
        )
        if avg_time is not None:
            results[name] = avg_time
            if baseline_time is None:
                baseline_time = avg_time
    
    # 結果の比較
    if baseline_time and len(results) > 1:
        print(f"\n📈 高速化比較（{list(results.keys())[0]}を基準）:")
        print("-" * 40)
        for name, time_val in results.items():
            speedup = baseline_time / time_val
            if speedup > 1.0:
                print(f"   {name:25s}: {speedup:.2f}x 🚀")
            elif speedup == 1.0:
                print(f"   {name:25s}: {speedup:.2f}x 📊")
            else:
                print(f"   {name:25s}: {speedup:.2f}x 🐌")
    
    print(f"\n✅ ベンチマーク完了！")

if __name__ == "__main__":
    main() 