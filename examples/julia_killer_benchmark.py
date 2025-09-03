#!/usr/bin/env python3
"""
Julia Killer Benchmark - JuliaのSIMD実装を上回るC++実装のテスト
"""

import numpy as np
import time
import matplotlib.pyplot as plt
from typing import List, Tuple, Dict
import subprocess
import os
import sys

# プロジェクトのパスを追加
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

try:
    import excitation_rk4_sparse_python as erk4
except ImportError:
    print("Error: C++バインディングが見つかりません。先にビルドしてください。")
    sys.exit(1)

class JuliaKillerBenchmark:
    """JuliaのSIMD実装を上回るC++実装のベンチマーク"""
    
    def __init__(self):
        self.dimensions = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
        self.results = {}
        
    def create_test_matrices(self, dim: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """テスト用のスパース行列を生成"""
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
            # x方向双極子
            dipole_strength = np.sqrt((i + 1))  # <n|x|n+1> = sqrt(n+1)
            mux_data.extend([complex(dipole_strength, 0), complex(dipole_strength, 0)])
            mux_row.extend([i, i + 1])
            mux_col.extend([i + 1, i])
            
            # y方向双極子（位相90度ずれ）
            muy_data.extend([complex(0, dipole_strength), complex(0, -dipole_strength)])
            muy_row.extend([i, i + 1])
            muy_col.extend([i + 1, i])
        
        # スパース行列の構築
        from scipy.sparse import csr_matrix
        
        H0 = csr_matrix((H0_data, (H0_row, H0_col)), shape=(dim, dim), dtype=complex)
        mux = csr_matrix((mux_data, (mux_row, mux_col)), shape=(dim, dim), dtype=complex)
        muy = csr_matrix((muy_data, (muy_row, muy_col)), shape=(dim, dim), dtype=complex)
        
        return H0, mux, muy
    
    def create_electric_field(self, steps: int = 1001) -> Tuple[np.ndarray, np.ndarray]:
        """テスト用の電場を生成"""
        dt = 0.01
        t = np.linspace(0, (steps - 1) * dt, steps)
        
        # x方向のサイン波
        Ex = np.sin(t)
        # y方向はゼロ（簡単のため）
        Ey = np.zeros_like(t)
        
        return Ex, Ey
    
    def benchmark_implementation(self, func_name: str, H0, mux, muy, Ex, Ey, 
                                psi0, dt: float, warmup: int = 3, 
                                runs: int = 10) -> Dict:
        """実装のベンチマーク"""
        # 関数の取得
        func = getattr(erk4, func_name)
        
        # ウォームアップ
        for _ in range(warmup):
            try:
                _ = func(H0, mux, muy, Ex, Ey, psi0, dt, False, 1, False)
            except Exception as e:
                return {"error": str(e), "time": float('inf')}
        
        # 実際の測定
        times = []
        for _ in range(runs):
            start_time = time.perf_counter()
            try:
                result = func(H0, mux, muy, Ex, Ey, psi0, dt, False, 1, False)
                end_time = time.perf_counter()
                times.append((end_time - start_time) * 1000)  # ms
            except Exception as e:
                return {"error": str(e), "time": float('inf')}
        
        # 統計情報
        times = np.array(times)
        return {
            "mean": np.mean(times),
            "std": np.std(times),
            "min": np.min(times),
            "max": np.max(times),
            "median": np.median(times)
        }
    
    def run_benchmark(self):
        """全次元でのベンチマーク実行"""
        print("🚀 Julia Killer Benchmark 開始")
        print("=" * 80)
        
        # テスト対象の実装リスト
        implementations = [
            ("rk4_sparse_eigen", "標準Sparse Matrix"),
            ("rk4_sparse_eigen_cached", "Eigen Cached"),
            # 新しいJulia Killer実装を追加予定
        ]
        
        dt = 0.01
        
        for dim in self.dimensions:
            print(f"\n📊 次元: {dim}")
            print("-" * 40)
            
            # テストデータの準備
            H0, mux, muy = self.create_test_matrices(dim)
            Ex, Ey = self.create_electric_field()
            psi0 = np.zeros(dim, dtype=complex)
            psi0[0] = 1.0  # 基底状態からスタート
            
            self.results[dim] = {}
            
            for func_name, description in implementations:
                print(f"  {description}: ", end="", flush=True)
                
                result = self.benchmark_implementation(
                    func_name, H0, mux, muy, Ex, Ey, psi0, dt
                )
                
                if "error" in result:
                    print(f"❌ エラー: {result['error']}")
                    self.results[dim][func_name] = result
                else:
                    print(f"✅ {result['mean']:.3f} ± {result['std']:.3f} ms")
                    self.results[dim][func_name] = result
    
    def compare_with_julia_reference(self):
        """Julia実装との比較"""
        print("\n🔥 Julia実装との性能比較")
        print("=" * 80)
        
        # Julia参考値（multilevel_benchmark_results.mdから）
        julia_simd_times = {
            2: 0.059, 4: 0.076, 8: 0.088, 16: 0.130, 32: 0.218,
            64: 0.390, 128: 0.749, 256: 1.579, 512: 3.217, 1024: 6.360
        }
        
        print(f"{'次元':>6} {'Julia SIMD':>12} {'C++ Best':>12} {'高速化比':>10} {'判定':>8}")
        print("-" * 60)
        
        for dim in self.dimensions:
            if dim not in self.results:
                continue
                
            julia_time = julia_simd_times.get(dim, float('inf'))
            
            # C++実装の最速時間を取得
            cpp_times = []
            for impl_name, result in self.results[dim].items():
                if "error" not in result:
                    cpp_times.append(result['mean'])
            
            if not cpp_times:
                cpp_best = float('inf')
                speedup = 0
                status = "❌"
            else:
                cpp_best = min(cpp_times)
                speedup = julia_time / cpp_best if cpp_best > 0 else 0
                status = "🏆" if speedup < 1.0 else "🔥" if speedup < 1.2 else "⚡"
            
            print(f"{dim:>6} {julia_time:>10.3f}ms {cpp_best:>10.3f}ms {speedup:>8.2f}x {status:>8}")
    
    def create_performance_graph(self):
        """性能グラフの作成"""
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # グラフ1: 実行時間比較
        dimensions = []
        times_dict = {}
        
        for dim in self.dimensions:
            if dim not in self.results:
                continue
            dimensions.append(dim)
            
            for impl_name, result in self.results[dim].items():
                if "error" not in result:
                    if impl_name not in times_dict:
                        times_dict[impl_name] = []
                    times_dict[impl_name].append(result['mean'])
        
        # Julia参考データの追加
        julia_simd_times = [0.059, 0.076, 0.088, 0.130, 0.218, 0.390, 0.749, 1.579, 3.217, 6.360]
        
        ax1.loglog(dimensions, julia_simd_times[:len(dimensions)], 'ro-', 
                   label='Julia SIMD', linewidth=2, markersize=8)
        
        colors = ['blue', 'green', 'purple', 'orange']
        for i, (impl_name, times) in enumerate(times_dict.items()):
            if len(times) == len(dimensions):
                ax1.loglog(dimensions, times, f'{colors[i % len(colors)]}s-', 
                          label=f'C++ {impl_name}', linewidth=2, markersize=6)
        
        ax1.set_xlabel('Dimension', fontsize=12)
        ax1.set_ylabel('Execution Time (ms)', fontsize=12)
        ax1.set_title('Performance Comparison: C++ vs Julia SIMD', fontsize=14, fontweight='bold')
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # グラフ2: 高速化比
        julia_simd_times_dict = dict(zip(dimensions, julia_simd_times[:len(dimensions)]))
        
        for impl_name, times in times_dict.items():
            if len(times) == len(dimensions):
                speedups = [julia_simd_times_dict[dim] / time for dim, time in zip(dimensions, times)]
                ax2.semilogx(dimensions, speedups, f'{colors[list(times_dict.keys()).index(impl_name) % len(colors)]}s-', 
                            label=f'C++ {impl_name}', linewidth=2, markersize=6)
        
        ax2.axhline(y=1.0, color='red', linestyle='--', alpha=0.7, label='Julia SIMD baseline')
        ax2.set_xlabel('Dimension', fontsize=12)
        ax2.set_ylabel('Speedup vs Julia SIMD', fontsize=12)
        ax2.set_title('Speedup Factor: C++ / Julia SIMD', fontsize=14, fontweight='bold')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # グラフの保存 [[memory:2714886]]
        output_dir = "examples/figures"
        os.makedirs(output_dir, exist_ok=True)
        plt.savefig(f"{output_dir}/julia_killer_benchmark_comparison.png", dpi=300, bbox_inches='tight')
        plt.show()
        
        print(f"\n📈 性能グラフを保存しました: {output_dir}/julia_killer_benchmark_comparison.png")
    
    def generate_report(self):
        """詳細レポートの生成"""
        print("\n📊 詳細ベンチマーク結果")
        print("=" * 120)
        
        # ヘッダー
        header = f"{'次元':>6}"
        for impl in ["rk4_sparse_eigen", "rk4_sparse_eigen_cached"]:
            header += f" {impl:>20}"
        header += f" {'Julia SIMD':>12} {'Best Speedup':>14}"
        print(header)
        print("-" * 120)
        
        # Julia参考値
        julia_simd_times = {
            2: 0.059, 4: 0.076, 8: 0.088, 16: 0.130, 32: 0.218,
            64: 0.390, 128: 0.749, 256: 1.579, 512: 3.217, 1024: 6.360
        }
        
        total_improvements = []
        
        for dim in self.dimensions:
            if dim not in self.results:
                continue
                
            row = f"{dim:>6}"
            julia_time = julia_simd_times.get(dim, float('inf'))
            best_cpp_time = float('inf')
            
            for impl in ["rk4_sparse_eigen", "rk4_sparse_eigen_cached"]:
                if impl in self.results[dim] and "error" not in self.results[dim][impl]:
                    time_val = self.results[dim][impl]['mean']
                    row += f" {time_val:>18.3f}ms"
                    best_cpp_time = min(best_cpp_time, time_val)
                else:
                    row += f" {'ERROR':>20}"
            
            speedup = julia_time / best_cpp_time if best_cpp_time < float('inf') else 0
            total_improvements.append(speedup)
            
            row += f" {julia_time:>10.3f}ms {speedup:>12.2f}x"
            print(row)
        
        # 総合評価
        print("\n🎯 総合評価")
        print("-" * 50)
        avg_improvement = np.mean(total_improvements) if total_improvements else 0
        
        if avg_improvement > 1.0:
            print(f"🏆 C++実装がJuliaを上回りました！ 平均高速化: {avg_improvement:.2f}x")
        elif avg_improvement > 0.8:
            print(f"🔥 C++実装がJuliaに迫りました！ 平均性能比: {avg_improvement:.2f}x")
        else:
            print(f"⚡ C++実装は改善が必要です。 平均性能比: {avg_improvement:.2f}x")
        
        return avg_improvement

def main():
    """メイン実行関数"""
    print("🚀 Julia Killer Benchmark - JuliaのSIMD実装との対戦")
    print("=" * 80)
    
    benchmark = JuliaKillerBenchmark()
    
    try:
        # ベンチマーク実行
        benchmark.run_benchmark()
        
        # Julia実装との比較
        benchmark.compare_with_julia_reference()
        
        # 詳細レポート生成
        avg_improvement = benchmark.generate_report()
        
        # 性能グラフ作成
        benchmark.create_performance_graph()
        
        # 最終判定
        print("\n" + "=" * 80)
        if avg_improvement > 1.0:
            print("🏆 結論: C++実装がJuliaのSIMD実装を上回りました！")
        else:
            print("⚡ 結論: さらなる最適化が必要です。戦略を見直しましょう。")
        print("=" * 80)
        
    except KeyboardInterrupt:
        print("\n⚠️  ベンチマークが中断されました。")
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 