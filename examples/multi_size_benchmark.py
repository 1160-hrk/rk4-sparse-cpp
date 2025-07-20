#!/usr/bin/env python3
"""
多次元ベンチマークテスト
=======================

異なる問題サイズでの性能を比較します。
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
    # 簡単な対角ハミルトニアン
    H0_diag = np.linspace(0, 1, dim)
    H0 = csr_matrix(np.diag(H0_diag), dtype=np.complex128)
    
    # 隣接要素間の結合
    off_diag = np.ones(dim-1) * 0.1
    mux = csr_matrix(np.diag(off_diag, k=1) + np.diag(off_diag, k=-1), dtype=np.complex128)
    
    # y方向は少し異なるパターン
    muy_data = np.ones(dim-2) * 0.05
    muy = csr_matrix(np.diag(muy_data, k=2) + np.diag(muy_data, k=-2), dtype=np.complex128)
    
    return H0, mux, muy

def create_test_fields(steps=300):
    """テスト用の電場を作成"""
    t = np.linspace(0, 6, 2*steps + 1)
    Ex = np.sin(t) * np.exp(-0.1 * t)
    Ey = np.cos(t) * np.exp(-0.1 * t)
    return Ex, Ey

def benchmark_single(func, H0, mux, muy, Ex, Ey, psi0, dt, num_runs=3):
    """単一実装の性能を測定"""
    times = []
    for _ in range(num_runs):
        start = time.perf_counter()
        try:
            result = func(H0, mux, muy, Ex, Ey, psi0, dt, False, 1, False)
            end = time.perf_counter()
            times.append(end - start)
        except Exception as e:
            print(f"   ❌ エラー: {e}")
            return None
    
    return np.mean(times)

def main():
    """メインベンチマーク"""
    print("🚀 多次元ベンチマークテスト")
    print("=" * 50)
    
    if not cpp_available:
        print("C++実装が利用できないため、ベンチマークを終了します。")
        return
    
    # テストサイズの設定
    test_dims = [256, 512, 1024, 2048]
    steps = 300
    dt = 0.01
    
    print(f"\n📊 テスト条件:")
    print(f"   - 時間ステップ数: {steps}")
    print(f"   - 時間刻み: {dt}")
    print(f"   - 実行回数: 3回の平均")
    
    # 実装リスト
    implementations = [
        (rk4_sparse_eigen, "従来Eigen実装"),
        (rk4_sparse_julia_style, "Julia風高速実装"),
        (rk4_sparse_csr_optimized, "CSR最適化実装"),
    ]
    
    results = {}
    
    for dim in test_dims:
        print(f"\n🔧 次元数: {dim}")
        print("-" * 30)
        
        # テストシステムの作成
        H0, mux, muy = create_test_system(dim)
        Ex, Ey = create_test_fields(steps)
        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0
        
        print(f"   非ゼロ要素: H0={H0.nnz}, mux={mux.nnz}, muy={muy.nnz}")
        
        dim_results = {}
        baseline_time = None
        
        for func, name in implementations:
            print(f"   ⏱️  {name}...", end='', flush=True)
            avg_time = benchmark_single(func, H0, mux, muy, Ex, Ey, psi0, dt)
            
            if avg_time is not None:
                dim_results[name] = avg_time
                if baseline_time is None:
                    baseline_time = avg_time
                print(f" {avg_time:.4f}s")
            else:
                print(" Failed")
        
        # 高速化比を計算
        if baseline_time:
            print(f"\n   📈 高速化比:")
            for name, time_val in dim_results.items():
                speedup = baseline_time / time_val
                if speedup >= 1.05:
                    symbol = "🚀"
                elif speedup >= 0.95:
                    symbol = "📊"
                else:
                    symbol = "🐌"
                print(f"     {name:20s}: {speedup:.2f}x {symbol}")
        
        results[dim] = dim_results
    
    # 全体的な結果のサマリー
    print(f"\n📈 全体的な性能比較:")
    print("=" * 50)
    print(f"{'次元数':>8s} | {'従来':>8s} | {'Julia風':>8s} | {'CSR':>8s} | {'Julia倍率':>8s} | {'CSR倍率':>8s}")
    print("-" * 60)
    
    for dim in test_dims:
        if dim in results and len(results[dim]) >= 2:
            baseline = list(results[dim].values())[0]
            julia_time = results[dim].get("Julia風高速実装", baseline)
            csr_time = results[dim].get("CSR最適化実装", baseline)
            
            julia_speedup = baseline / julia_time if julia_time else 1.0
            csr_speedup = baseline / csr_time if csr_time else 1.0
            
            print(f"{dim:8d} | {baseline:8.4f} | {julia_time:8.4f} | {csr_time:8.4f} | {julia_speedup:8.2f} | {csr_speedup:8.2f}")
    
    print(f"\n✅ ベンチマーク完了！")

if __name__ == "__main__":
    main() 