"""
SuiteSparse改善のベンチマーク（リファクタリング版）
"""

from benchmark_base import (
    BaseBenchmark, ImplementationManager, PlotGenerator, SAVEPATH
)
from typing import List, Dict, Any
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os
from scipy.sparse import csr_matrix

class SuiteSparseImprovementsBenchmark(BaseBenchmark):
    """SuiteSparse改善のベンチマーククラス"""

    def __init__(self, dims: List[int], num_repeats: int = 5, num_steps: int = 1000):
        # 利用可能な実装を取得
        available_implementations = ImplementationManager.get_available_implementations()

        # SuiteSparse改善比較用の実装を選択
        implementations = {'eigen', 'suitesparse'}
        # 利用可能な実装のみに絞り込み
        implementations = implementations.intersection(available_implementations)

        super().__init__(implementations, dims, num_repeats, num_steps)

    def create_two_level_system(self, dim: int) -> tuple:
        """2準位系のテストシステムを生成"""
        # ハミルトニアン（対角成分）
        H0 = csr_matrix(np.diag([0.0, 1.0] * (dim // 2)), dtype=np.complex128)

        # 双極子演算子（非対角成分）
        mux = csr_matrix((dim, dim), dtype=np.complex128)
        muy = csr_matrix((dim, dim), dtype=np.complex128)

        # 隣接する準位間の結合を設定
        for i in range(dim - 1):
            mux[i, i + 1] = 1.0
            mux[i + 1, i] = 1.0

        return H0, mux, muy

    def create_excitation_pulse(self, t_steps: int, dt: float) -> tuple:
        """励起パルスを生成"""
        t = np.arange(0, t_steps * dt, dt)
        # ガウシアンパルス
        sigma = 5.0
        t0 = t_steps * dt / 2
        E0 = 1.0
        omega_L = 1.0

        Ex = E0 * np.exp(-((t - t0) / sigma) ** 2) * np.cos(omega_L * t)
        Ey = np.zeros_like(Ex)

        return Ex, Ey

    def analyze_performance_issues(self, results: Dict[str, List[float]]) -> Dict[str, Any]:
        """性能問題を分析"""
        analysis = {}

        for impl, times in results.items():
            if all(t != float('inf') for t in times):
                mean_time = np.mean(times)
                std_time = np.std(times)
                min_time = np.min(times)
                max_time = np.max(times)

                analysis[impl] = {
                    'mean_time': mean_time,
                    'std_time': std_time,
                    'min_time': min_time,
                    'max_time': max_time,
                    'cv': std_time / mean_time if mean_time > 0 else 0,  # 変動係数
                    'stability': 'stable' if std_time / mean_time < 0.1 else 'unstable'
                }
            else:
                analysis[impl] = {
                    'mean_time': float('inf'),
                    'std_time': 0,
                    'min_time': float('inf'),
                    'max_time': float('inf'),
                    'cv': 0,
                    'stability': 'failed'
                }

        return analysis

    def create_suitesparse_improvement_plots(self) -> str:
        """SuiteSparse改善効果のプロットを作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(SAVEPATH, f'suitesparse_improvements_{timestamp}.png')

        implementations = sorted(self.implementations)

        plt.figure(figsize=(15, 10))

        # 1. 実行時間の比較
        plt.subplot(2, 3, 1)
        PlotGenerator.create_execution_time_plot(
            self.results, self.dims, self.implementations, plt.gca()
        )

        # 2. CPU使用率の比較
        plt.subplot(2, 3, 2)
        PlotGenerator.create_resource_usage_plot(
            self.detailed_results, self.dims, self.implementations,
            plt.gca(), 'cpu_usage', "CPU Usage Comparison"
        )

        # 3. メモリ使用量の比較
        plt.subplot(2, 3, 3)
        PlotGenerator.create_resource_usage_plot(
            self.detailed_results, self.dims, self.implementations,
            plt.gca(), 'memory_usage', "Memory Usage Comparison"
        )

        # 4. SuiteSparse vs Eigen速度向上率
        plt.subplot(2, 3, 4)
        if ('eigen' in self.implementations and
                'suitesparse' in self.implementations):
            speedup_values = []
            for dim in self.dims:
                eigen_result = self.detailed_results.get('eigen', {}).get(dim)
                suitesparse_result = self.detailed_results.get(
                    'suitesparse', {}).get(dim)

                if eigen_result is not None and suitesparse_result is not None:
                    eigen_time = eigen_result.mean_time
                    suitesparse_time = suitesparse_result.mean_time
                    speedup_values.append(
                        eigen_time / suitesparse_time
                        if suitesparse_time > 0 else 1.0)
                else:
                    speedup_values.append(1.0)

            plt.plot(self.dims, speedup_values, 'o-',
                     label='Eigen/SuiteSparse', linewidth=2, markersize=6)
            plt.axhline(y=1.0, color='black', linestyle='--',
                        alpha=0.5, label='Equal performance')
            plt.xlabel('Matrix Size')
            plt.ylabel('Speedup Ratio (Eigen/SuiteSparse)')
            plt.title('SuiteSparse Improvement Effect')
            plt.grid(True)
            plt.legend()

        # 5. 最大次元での統計情報
        plt.subplot(2, 3, 5)
        max_dim = max(self.dims)
        stats_data = []
        labels = []

        for impl in implementations:
            if impl in self.results and max_dim in self.results[impl]:
                times = self.results[impl][max_dim]
                if all(t != float('inf') for t in times):
                    stats_data.append(times)
                    labels.append(impl)

        if stats_data:
            plt.boxplot(stats_data)
            plt.xticks(range(1, len(labels) + 1), labels)
            plt.ylabel('Execution Time (seconds)')
            plt.title(f'Performance Distribution (dim={max_dim})')
            plt.grid(True)
            plt.yscale('log')

        # 6. システム情報
        plt.subplot(2, 3, 6)
        plt.axis('off')
        import multiprocessing
        import psutil
        system_info = f"""
        System Information:
        CPU Cores: {multiprocessing.cpu_count()}
        Total Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB
        Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB

        Benchmark Parameters:
        Implementations: {', '.join(implementations)}
        Dimensions: {self.dims}
        Steps: {self.num_steps}
        Repeats: {self.num_repeats}
        """
        plt.text(0.1, 0.5, system_info, transform=plt.gca().transAxes,
                 fontsize=10, verticalalignment='center')

        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()

        return plot_filename

    def print_suitesparse_summary(self):
        """SuiteSparse改善効果のサマリーを表示"""
        print("\n" + "="*80)
        print("SuiteSparse改善効果のベンチマーク結果サマリー")
        print("="*80)

        implementations = sorted(self.implementations)

        # システム情報
        import multiprocessing
        import psutil
        print("\nシステム情報:")
        print(f"  CPU コア数: {multiprocessing.cpu_count()}")
        print(f"  総メモリ: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"  利用可能メモリ: "
              f"{psutil.virtual_memory().available / (1024**3):.1f} GB")
        print(f"  比較実装: {', '.join(implementations)}")

        # 各次元での最速実装
        print("\n各次元での最速実装:")
        for dim in self.dims:
            best_impl = None
            best_time = float('inf')
            for impl in implementations:
                result = self.detailed_results.get(impl, {}).get(dim)
                if result is not None:
                    if result.mean_time < best_time:
                        best_time = result.mean_time
                        best_impl = impl

            if best_impl:
                print(f"  次元 {dim}: {best_impl} ({best_time:.6f}秒)")

        # SuiteSparse改善効果
        print("\nSuiteSparse改善効果（eigen vs suitesparse）:")
        for dim in self.dims:
            eigen_result = self.detailed_results.get('eigen', {}).get(dim)
            suitesparse_result = self.detailed_results.get(
                'suitesparse', {}).get(dim)

            if eigen_result is not None and suitesparse_result is not None:
                eigen_time = eigen_result.mean_time
                suitesparse_time = suitesparse_result.mean_time
                if eigen_time > 0:
                    improvement = ((eigen_time - suitesparse_time) /
                                   eigen_time * 100)
                    print(f"  次元 {dim}: {improvement:.1f}% 高速化")


def main():
    # テストする行列サイズ
    dims = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
    num_repeats = 5  # 各サイズでの繰り返し回数
    num_steps = 1000  # 時間発展のステップ数

    print("SuiteSparse改善効果のベンチマーク開始")
    print(f"- 行列サイズ: {dims}")
    print(f"- 繰り返し回数: {num_repeats}")
    print(f"- 時間発展ステップ数: {num_steps}")
    print("- 比較実装: eigen, suitesparse")
    print("- 新機能: SuiteSparse-MKL最適化")

    # ベンチマークの実行
    benchmark = SuiteSparseImprovementsBenchmark(dims, num_repeats, num_steps)
    results, detailed_results = benchmark.run_benchmark()

    # 結果のサマリーを表示
    benchmark.print_suitesparse_summary()

    # 結果をプロット
    print("\n=== Plotting Results ===")
    plot_file = benchmark.create_suitesparse_improvement_plots()

    # 結果をファイルに保存
    print("\n=== Saving Results ===")
    json_file, csv_file = benchmark.save_results("suitesparse_improvements")

    print("\nベンチマーク完了")
    print(f"結果は{plot_file}に保存されました")

if __name__ == "__main__":
    main()
