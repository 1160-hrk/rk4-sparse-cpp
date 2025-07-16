#!/usr/bin/env python3
"""
SuiteSparse改善版のベンチマーク比較スクリプト

このスクリプトは、以下の実装を比較します：
1. rk4_sparse_eigen (Eigen版)
2. rk4_sparse_suitesparse (元のSuiteSparse版)
3. rk4_sparse_suitesparse_optimized (最適化版)
4. rk4_sparse_suitesparse_fast (高速版)

**問題点の分析と改善提案を含む**

使用方法:
    python benchmark_suitesparse_improvements.py
"""

import numpy as np
import scipy.sparse as sp
import time
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import os
import sys

# プロジェクトのルートディレクトリをPythonパスに追加
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'python'))

try:
    import rk4_sparse
    from rk4_sparse import (
        rk4_sparse_eigen,
        rk4_sparse_suitesparse,
        rk4_sparse_suitesparse_optimized,
        rk4_sparse_suitesparse_fast
    )
except ImportError as e:
    print(f"Error importing rk4_sparse: {e}")
    print("Please make sure the module is built and installed correctly.")
    sys.exit(1)

def create_two_level_system(dim):
    """多準位系のハミルトニアンと双極子演算子を生成"""
    # ハミルトニアン（対角要素）
    H0 = sp.csr_matrix((dim, dim), dtype=np.complex128)
    for i in range(dim):
        H0[i, i] = i * 1.0  # エネルギー準位
    
    # 双極子演算子（x方向）
    mux = sp.csr_matrix((dim, dim), dtype=np.complex128)
    for i in range(dim-1):
        mux[i, i+1] = 1.0
        mux[i+1, i] = 1.0
    
    # 双極子演算子（y方向）
    muy = sp.csr_matrix((dim, dim), dtype=np.complex128)
    
    return H0, mux, muy

def create_excitation_pulse(t_steps, dt):
    """励起パルスを生成"""
    t = np.arange(0, t_steps * dt, dt)
    E0 = 0.1
    omega_L = 1.0
    
    Ex = E0 * np.sin(omega_L * t)
    Ey = np.zeros_like(Ex)
    
    return Ex, Ey

def run_single_benchmark(H0, mux, muy, Ex, Ey, psi0, dt, implementations, num_runs=5):
    """単一のベンチマーク実行"""
    results = {}
    
    for name, func in implementations:
        print(f"  Testing {name}...")
        times = []
        
        for run in range(num_runs):
            start_time = time.perf_counter()
            try:
                result = func(H0, mux, muy, Ex, Ey, psi0, dt, True, 1, False)
                end_time = time.perf_counter()
                times.append(end_time - start_time)
            except Exception as e:
                print(f"    Error in {name}: {e}")
                times.append(np.nan)
        
        # 有効な結果のみを使用
        valid_times = [t for t in times if not np.isnan(t)]
        if valid_times:
            results[name] = {
                'mean_time': np.mean(valid_times),
                'std_time': np.std(valid_times),
                'min_time': np.min(valid_times),
                'max_time': np.max(valid_times),
                'success_rate': len(valid_times) / len(times)
            }
        else:
            results[name] = None
    
    return results

def analyze_performance_issues(results):
    """性能問題の分析"""
    print("\n" + "="*60)
    print("PERFORMANCE ANALYSIS")
    print("="*60)
    
    # 1. 命名と実装の不一致チェック
    print("\n1. 命名と実装の不一致:")
    if 'suitesparse_optimized' in results and 'suitesparse_fast' in results:
        opt_time = results['suitesparse_optimized']['mean_time']
        fast_time = results['suitesparse_fast']['mean_time']
        
        if opt_time < fast_time:
            print(f"  ✓ 最適化版 ({opt_time:.6f}s) < 高速版 ({fast_time:.6f}s)")
        else:
            print(f"  ❌ 問題: 最適化版 ({opt_time:.6f}s) > 高速版 ({fast_time:.6f}s)")
            print(f"     → 命名と実装が一致していません")
    
    # 2. Eigen版との比較
    print("\n2. Eigen版との比較:")
    if 'eigen' in results:
        eigen_time = results['eigen']['mean_time']
        print(f"  Eigen版: {eigen_time:.6f}s (基準)")
        
        for name in ['suitesparse', 'suitesparse_optimized', 'suitesparse_fast']:
            if name in results and results[name] is not None:
                impl_time = results[name]['mean_time']
                speedup = eigen_time / impl_time
                if speedup > 1.0:
                    print(f"  {name}: {impl_time:.6f}s ({speedup:.2f}x slower than Eigen)")
                else:
                    print(f"  {name}: {impl_time:.6f}s ({1/speedup:.2f}x faster than Eigen)")
    
    # 3. SuiteSparse版間の性能差
    print("\n3. SuiteSparse版間の性能差:")
    suite_versions = ['suitesparse', 'suitesparse_optimized', 'suitesparse_fast']
    suite_times = []
    
    for name in suite_versions:
        if name in results and results[name] is not None:
            suite_times.append((name, results[name]['mean_time']))
    
    if len(suite_times) >= 2:
        min_time = min(suite_times, key=lambda x: x[1])
        max_time = max(suite_times, key=lambda x: x[1])
        ratio = max_time[1] / min_time[1]
        
        print(f"  最速: {min_time[0]} ({min_time[1]:.6f}s)")
        print(f"  最遅: {max_time[0]} ({max_time[1]:.6f}s)")
        print(f"  性能差: {ratio:.2f}倍")
        
        if ratio < 1.1:
            print(f"  ⚠️  警告: 実装間の性能差が小さすぎます ({ratio:.2f}倍)")
            print(f"     → 実装が実質的に同じ可能性があります")
    
    # 4. 改善提案
    print("\n4. 改善提案:")
    print("  a) 実装の見直し:")
    print("     - 各実装の詳細な違いを明確化")
    print("     - 命名を実装内容に合わせて修正")
    print("     - または実装を命名に合わせて修正")
    
    print("  b) 大規模問題でのテスト:")
    print("     - より大きな行列サイズでのベンチマーク")
    print("     - 疎行列の密度による性能比較")
    print("     - メモリ使用量の測定")
    
    print("  c) SuiteSparseの真の利点を活かす:")
    print("     - より疎な行列でのテスト")
    print("     - SuiteSparse固有の最適化を確認")
    print("     - 並列化効率の測定")

def run_benchmark():
    """メインベンチマーク実行"""
    print("=== SuiteSparse改善版ベンチマーク ===")
    
    # パラメータ設定
    dimensions = [2, 4, 8, 16, 32, 64]
    dt = 0.01
    t_steps = 1000
    num_runs = 5
    
    # 実装リスト
    implementations = [
        ("eigen", rk4_sparse_eigen),
        ("suitesparse", rk4_sparse_suitesparse),
        ("suitesparse_optimized", rk4_sparse_suitesparse_optimized),
        ("suitesparse_fast", rk4_sparse_suitesparse_fast)
    ]
    
    # 結果を格納する辞書
    all_results = {}
    
    for dim in dimensions:
        print(f"\n次元 {dim} をテスト中...")
        
        # システムを生成
        H0, mux, muy = create_two_level_system(dim)
        Ex, Ey = create_excitation_pulse(t_steps, dt)
        psi0 = np.zeros(dim, dtype=complex)
        psi0[0] = 1.0
        
        # ベンチマーク実行
        results = run_single_benchmark(H0, mux, muy, Ex, Ey, psi0, dt, implementations, num_runs)
        all_results[dim] = results
        
        # 結果表示
        print(f"  結果:")
        for name, result in results.items():
            if result is not None:
                print(f"    {name}: {result['mean_time']:.6f} ± {result['std_time']:.6f}s")
            else:
                print(f"    {name}: Failed")
    
    # 性能問題の分析
    analyze_performance_issues(all_results[2])  # 2x2の結果で分析
    
    # スケーリング分析
    print("\n" + "="*60)
    print("SCALING ANALYSIS")
    print("="*60)
    
    # データフレームの作成
    data = []
    for dim in dimensions:
        for name in ['eigen', 'suitesparse', 'suitesparse_optimized', 'suitesparse_fast']:
            if name in all_results[dim] and all_results[dim][name] is not None:
                data.append({
                    'dimension': dim,
                    'implementation': name,
                    'mean_time': all_results[dim][name]['mean_time'],
                    'std_time': all_results[dim][name]['std_time']
                })
    
    df = pd.DataFrame(data)
    
    # スケーリングプロット
    plt.figure(figsize=(15, 10))
    
    # 実行時間のスケーリング
    plt.subplot(2, 2, 1)
    for name in ['eigen', 'suitesparse', 'suitesparse_optimized', 'suitesparse_fast']:
        subset = df[df['implementation'] == name]
        if not subset.empty:
            plt.errorbar(subset['dimension'], subset['mean_time'], 
                        yerr=subset['std_time'], label=name, marker='o')
    
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Execution Time (seconds)')
    plt.title('Scaling Performance')
    plt.legend()
    plt.grid(True)
    plt.yscale('log')
    
    # 速度向上率のスケーリング
    plt.subplot(2, 2, 2)
    eigen_times = df[df['implementation'] == 'eigen'].set_index('dimension')['mean_time']
    
    for name in ['suitesparse', 'suitesparse_optimized', 'suitesparse_fast']:
        subset = df[df['implementation'] == name]
        if not subset.empty:
            subset = subset.set_index('dimension')
            speedups = eigen_times / subset['mean_time']
            plt.plot(speedups.index, speedups.values, label=name, marker='o')
    
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Speedup vs Eigen')
    plt.title('Speedup Scaling')
    plt.legend()
    plt.grid(True)
    
    # 性能差の分析
    plt.subplot(2, 2, 3)
    suite_versions = ['suitesparse', 'suitesparse_optimized', 'suitesparse_fast']
    for dim in dimensions:
        times = []
        names = []
        for name in suite_versions:
            if name in all_results[dim] and all_results[dim][name] is not None:
                times.append(all_results[dim][name]['mean_time'])
                names.append(name)
        
        if len(times) >= 2:
            min_time = min(times)
            ratios = [t / min_time for t in times]
            x_pos = np.arange(len(ratios))
            plt.bar([f"{dim}x{dim}\n{name}" for name in names], ratios, alpha=0.7)
    
    plt.ylabel('Time Ratio (vs fastest)')
    plt.title('Performance Variation Among SuiteSparse Versions')
    plt.xticks(rotation=45)
    plt.grid(True)
    
    # 成功率の表示
    plt.subplot(2, 2, 4)
    success_rates = {}
    for name in ['eigen', 'suitesparse', 'suitesparse_optimized', 'suitesparse_fast']:
        rates = []
        for dim in dimensions:
            if name in all_results[dim] and all_results[dim][name] is not None:
                rates.append(all_results[dim][name]['success_rate'])
            else:
                rates.append(0.0)
        success_rates[name] = rates
    
    x = np.arange(len(dimensions))
    width = 0.2
    for i, (name, rates) in enumerate(success_rates.items()):
        plt.bar(x + i*width, rates, width, label=name, alpha=0.7)
    
    plt.xlabel('Matrix Dimension')
    plt.ylabel('Success Rate')
    plt.title('Implementation Success Rates')
    plt.legend()
    plt.xticks(x + width*1.5, dimensions)
    plt.grid(True)
    
    plt.tight_layout()
    
    # 保存
    savepath = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
    os.makedirs(savepath, exist_ok=True)
    plt.savefig(os.path.join(savepath, 'benchmark_suitesparse_improvements.png'), 
                dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"\nプロットを保存しました: {os.path.join(savepath, 'benchmark_suitesparse_improvements.png')}")
    
    return all_results

if __name__ == "__main__":
    results = run_benchmark() 