#!/usr/bin/env python3
"""
大規模問題ベンチマークテスト
===========================

4096次元以上の大規模問題での性能を詳細に評価します。
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
    print(f"   🔧 {dim}次元システム構築中...", end='', flush=True)
    
    # 簡単な対角ハミルトニアン
    H0_diag = np.linspace(0, 1, dim)
    H0 = csr_matrix(np.diag(H0_diag), dtype=np.complex128)
    
    # 隣接要素間の結合
    off_diag = np.ones(dim-1) * 0.1
    mux = csr_matrix(np.diag(off_diag, k=1) + np.diag(off_diag, k=-1), dtype=np.complex128)
    
    # y方向は少し異なるパターン
    muy_data = np.ones(dim-2) * 0.05
    muy = csr_matrix(np.diag(muy_data, k=2) + np.diag(muy_data, k=-2), dtype=np.complex128)
    
    print(" ✅")
    return H0, mux, muy

def create_test_fields(steps=200):
    """テスト用の電場を作成（大規模問題用に短縮）"""
    t = np.linspace(0, 4, 2*steps + 1)
    Ex = np.sin(t) * np.exp(-0.1 * t)
    Ey = np.cos(t) * np.exp(-0.1 * t)
    return Ex, Ey

def benchmark_single(func, name, H0, mux, muy, Ex, Ey, psi0, dt):
    """単一実装の性能を測定（大規模問題用）"""
    print(f"   ⏱️  {name:20s}...", end='', flush=True)
    
    # ウォームアップ（小さなテスト）
    start = time.perf_counter()
    try:
        result = func(H0, mux, muy, Ex, Ey, psi0, dt, False, 1, False)
        end = time.perf_counter()
        runtime = end - start
        print(f" {runtime:8.4f}s")
        return runtime
    except Exception as e:
        print(f" ❌ エラー: {e}")
        return None

def main():
    """メインベンチマーク"""
    print("🚀 大規模問題ベンチマークテスト")
    print("=" * 60)
    
    if not cpp_available:
        print("C++実装が利用できないため、ベンチマークを終了します。")
        return
    
    # 大規模テストサイズの設定
    test_dims = [1024, 2048, 4096]
    steps = 200  # 大規模問題では時間ステップを削減
    dt = 0.01
    
    print(f"\n📊 テスト条件:")
    print(f"   - 時間ステップ数: {steps} (大規模問題用に短縮)")
    print(f"   - 時間刻み: {dt}")
    print(f"   - 各実装を1回ずつ実行")
    
    # 実装リスト
    implementations = [
        (rk4_sparse_eigen, "従来Eigen実装"),
        (rk4_sparse_julia_style, "Julia風高速実装"),
        (rk4_sparse_csr_optimized, "CSR最適化実装"),
    ]
    
    results = {}
    
    for dim in test_dims:
        print(f"\n🔧 次元数: {dim}")
        print("-" * 40)
        
        # テストシステムの作成
        H0, mux, muy = create_test_system(dim)
        Ex, Ey = create_test_fields(steps)
        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0
        
        print(f"   📈 非ゼロ要素: H0={H0.nnz}, mux={mux.nnz}, muy={muy.nnz}")
        
        dim_results = {}
        baseline_time = None
        
        for func, name in implementations:
            runtime = benchmark_single(func, name, H0, mux, muy, Ex, Ey, psi0, dt)
            
            if runtime is not None:
                dim_results[name] = runtime
                if baseline_time is None:
                    baseline_time = runtime
        
        # 高速化比を計算
        if baseline_time and len(dim_results) > 1:
            print(f"\n   📈 高速化比（{list(dim_results.keys())[0]}を基準）:")
            for name, time_val in dim_results.items():
                speedup = baseline_time / time_val
                improvement = (speedup - 1.0) * 100
                
                if speedup >= 1.1:
                    symbol = "🚀"
                    status = f"+{improvement:.1f}%"
                elif speedup >= 0.9:
                    symbol = "📊"
                    status = f"{improvement:+.1f}%"
                else:
                    symbol = "🐌"
                    status = f"{improvement:+.1f}%"
                    
                print(f"     {name:20s}: {speedup:.2f}x ({status}) {symbol}")
        
        results[dim] = dim_results
    
    # 全体的な結果のサマリー
    print(f"\n📈 大規模問題での性能サマリー:")
    print("=" * 60)
    print(f"{'次元数':>8s} | {'従来[s]':>9s} | {'Julia[s]':>9s} | {'CSR[s]':>7s} | {'Julia倍率':>9s} | {'CSR倍率':>8s}")
    print("-" * 60)
    
    total_speedup_julia = []
    total_speedup_csr = []
    
    for dim in test_dims:
        if dim in results and len(results[dim]) >= 2:
            baseline = list(results[dim].values())[0]
            julia_time = results[dim].get("Julia風高速実装", baseline)
            csr_time = results[dim].get("CSR最適化実装", baseline)
            
            julia_speedup = baseline / julia_time if julia_time else 1.0
            csr_speedup = baseline / csr_time if csr_time else 1.0
            
            total_speedup_julia.append(julia_speedup)
            total_speedup_csr.append(csr_speedup)
            
            print(f"{dim:8d} | {baseline:9.4f} | {julia_time:9.4f} | {csr_time:7.4f} | {julia_speedup:9.2f} | {csr_speedup:8.2f}")
    
    # 平均高速化比
    if total_speedup_julia:
        avg_julia = np.mean(total_speedup_julia)
        avg_csr = np.mean(total_speedup_csr)
        
        print("-" * 60)
        print(f"{'平均':>8s} | {'':>9s} | {'':>9s} | {'':>7s} | {avg_julia:9.2f} | {avg_csr:8.2f}")
        
        print(f"\n🎯 結論:")
        print(f"   - Julia風実装: 平均 {avg_julia:.2f}x ({(avg_julia-1)*100:+.1f}%)")
        print(f"   - CSR最適化実装: 平均 {avg_csr:.2f}x ({(avg_csr-1)*100:+.1f}%)")
        
        if avg_julia >= 1.5:
            print(f"   🚀 大規模問題で大幅な性能向上を達成！")
        elif avg_julia >= 1.1:
            print(f"   📈 大規模問題で良好な性能向上を達成")
        else:
            print(f"   📊 性能は同等レベル")
    
    print(f"\n✅ 大規模ベンチマーク完了！")

if __name__ == "__main__":
    main() 