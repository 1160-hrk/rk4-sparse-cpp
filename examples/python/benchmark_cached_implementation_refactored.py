"""
キャッシュ化実装のベンチマーク（リファクタリング版）
"""

from benchmark_base import (
    BaseBenchmark, ImplementationManager, PlotGenerator, SAVEPATH
)
from typing import List
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

class CachedImplementationBenchmark(BaseBenchmark):
    """キャッシュ化実装のベンチマーククラス"""

    def __init__(self, dims: List[int], num_repeats: int = 10, num_steps: int = 1000):
        # 利用可能な実装を取得
        available_implementations = ImplementationManager.get_available_implementations()

        # キャッシュ化比較用の実装を選択
        implementations = {'python', 'eigen', 'eigen_cached'}
        # 利用可能な実装のみに絞り込み
        implementations = implementations.intersection(available_implementations)

        super().__init__(implementations, dims, num_repeats, num_steps)

    def create_cached_comparison_plots(self) -> str:
        """キャッシュ化効果の比較プロットを作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(SAVEPATH, f'cached_benchmark_{timestamp}.png')

        implementations = sorted(self.implementations)
        colors = ['blue', 'red', 'green']

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

        # 4. 速度向上率の比較
        plt.subplot(2, 3, 4)
        PlotGenerator.create_speedup_plot(
            self.detailed_results, self.dims, self.implementations, plt.gca()
        )

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

    def print_cached_summary(self):
        """キャッシュ化効果のサマリーを表示"""
        print("\n" + "="*80)
        print("キャッシュ化実装のベンチマーク結果サマリー")
        print("="*80)

        implementations = sorted(self.implementations)

        # システム情報
        import multiprocessing
        import psutil
        print("\nシステム情報:")
        print(f"  CPU コア数: {multiprocessing.cpu_count()}")
        print(f"  総メモリ: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"  利用可能メモリ: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        print(f"  比較実装: {', '.join(implementations)}")

        # 各次元での最速実装
        print("\n各次元での最速実装:")
        for dim in self.dims:
            best_impl = None
            best_time = float('inf')
            for impl in implementations:
                if (impl in self.detailed_results and dim in self.detailed_results[impl] and
                    self.detailed_results[impl][dim] is not None):
                    if self.detailed_results[impl][dim].mean_time < best_time:
                        best_time = self.detailed_results[impl][dim].mean_time
                        best_impl = impl

            if best_impl:
                print(f"  次元 {dim}: {best_impl} ({best_time:.6f}秒)")

        # 平均性能指標
        print("\n平均性能指標（全次元）:")
        for impl in implementations:
            times = []
            cpus = []
            mems = []
            threads = []
            for dim in self.dims:
                if (impl in self.detailed_results and dim in self.detailed_results[impl] and
                    self.detailed_results[impl][dim] is not None):
                    times.append(self.detailed_results[impl][dim].mean_time)
                    cpus.append(self.detailed_results[impl][dim].cpu_usage)
                    mems.append(self.detailed_results[impl][dim].memory_usage)
                    threads.append(self.detailed_results[impl][dim].thread_count)

            if times:
                avg_time = np.mean(times)
                avg_cpu = np.mean(cpus)
                avg_mem = np.mean(mems)
                avg_threads = np.mean(threads)
                print(f"  {impl}:")
                print(f"    平均実行時間: {avg_time:.6f}秒")
                print(f"    平均CPU使用率: {avg_cpu:.1f}%")
                print(f"    平均メモリ使用量: {avg_mem:.1f} MB")
                print(f"    平均スレッド数: {avg_threads:.1f}")

        # 最大次元での詳細比較
        max_dim = max(self.dims)
        print(f"\n最大次元（{max_dim}）での詳細比較:")
        for impl in implementations:
            if (impl in self.detailed_results and max_dim in self.detailed_results[impl] and
                self.detailed_results[impl][max_dim] is not None):
                result = self.detailed_results[impl][max_dim]
                print(f"  {impl}:")
                print(f"    実行時間: {result.mean_time:.6f}秒")
                print(f"    CPU使用率: {result.cpu_usage:.1f}%")
                print(f"    メモリ使用量: {result.memory_usage:.1f} MB")
                print(f"    ピークメモリ: {result.memory_peak:.1f} MB")
                print(f"    スレッド数: {result.thread_count}")
                if impl != 'python':
                    print(f"    速度向上率: {result.speedup_vs_python:.2f}倍")

        # キャッシュ化の効果
        print("\nキャッシュ化の効果（eigen vs eigen_cached）:")
        for dim in self.dims:
            if ('eigen' in self.detailed_results and dim in self.detailed_results['eigen'] and
                self.detailed_results['eigen'][dim] is not None and
                'eigen_cached' in self.detailed_results and dim in self.detailed_results['eigen_cached'] and
                self.detailed_results['eigen_cached'][dim] is not None):
                eigen_time = self.detailed_results['eigen'][dim].mean_time
                cached_time = self.detailed_results['eigen_cached'][dim].mean_time
                if eigen_time > 0:
                    improvement = (eigen_time - cached_time) / eigen_time * 100
                    print(f"  次元 {dim}: {improvement:.1f}% 高速化")

def main():
    # テストする行列サイズ
    dims = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    num_repeats = 10  # 各サイズでの繰り返し回数
    num_steps = 1000  # 時間発展のステップ数

    print("キャッシュ化実装のベンチマーク開始")
    print(f"- 行列サイズ: {dims}")
    print(f"- 繰り返し回数: {num_repeats}")
    print(f"- 時間発展ステップ数: {num_steps}")
    print("- 比較実装: python, eigen, eigen_cached")
    print("- 新機能: パターン構築・データ展開のキャッシュ化")

    # ベンチマークの実行
    benchmark = CachedImplementationBenchmark(dims, num_repeats, num_steps)
    results, detailed_results = benchmark.run_benchmark()

    # 結果のサマリーを表示
    benchmark.print_cached_summary()

    # 結果をプロット
    print("\n=== Plotting Results ===")
    plot_file = benchmark.create_cached_comparison_plots()

    # 結果をファイルに保存
    print("\n=== Saving Results ===")
    json_file, csv_file = benchmark.save_results("cached_benchmark")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    print("\nベンチマーク完了")
    print(f"結果は{plot_file}に保存されました")

if __name__ == "__main__":
    main()
