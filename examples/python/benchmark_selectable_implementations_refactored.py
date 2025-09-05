"""
選択可能な実装のベンチマーク（リファクタリング版）
"""

from benchmark_base import (
    BaseBenchmark, ImplementationManager, PlotGenerator, SAVEPATH
)
from typing import List, Set
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

class SelectableImplementationsBenchmark(BaseBenchmark):
    """選択可能な実装のベンチマーククラス"""

    def __init__(self, selected_implementations: Set[str], dims: List[int],
                 num_repeats: int = 10, num_steps: int = 1000):
        # 利用可能な実装を取得
        available_implementations = ImplementationManager.get_available_implementations()

        # 選択された実装の妥当性チェック
        invalid_impls = selected_implementations - available_implementations
        if invalid_impls:
            print(f"エラー: 無効な実装が指定されました: {invalid_impls}")
            print(f"利用可能な実装: {available_implementations}")
            raise ValueError(f"無効な実装: {invalid_impls}")

        # 利用可能な実装のみに絞り込み
        implementations = selected_implementations.intersection(available_implementations)

        super().__init__(implementations, dims, num_repeats, num_steps)

    def create_selectable_plots(self) -> str:
        """選択可能な実装のプロットを作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(SAVEPATH, f'selectable_benchmark_{timestamp}.png')

        implementations = sorted(self.implementations)
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown']

        plt.figure(figsize=(20, 15))

        # 1. 実行時間の比較
        plt.subplot(3, 4, 1)
        PlotGenerator.create_execution_time_plot(
            self.results, self.dims, self.implementations, plt.gca()
        )

        # 2. CPU使用率の比較
        plt.subplot(3, 4, 2)
        PlotGenerator.create_resource_usage_plot(
            self.detailed_results, self.dims, self.implementations,
            plt.gca(), 'cpu_usage', "CPU Usage Comparison"
        )

        # 3. メモリ使用量の比較
        plt.subplot(3, 4, 3)
        PlotGenerator.create_resource_usage_plot(
            self.detailed_results, self.dims, self.implementations,
            plt.gca(), 'memory_usage', "Memory Usage Comparison"
        )

        # 4. スレッド数の比較
        plt.subplot(3, 4, 4)
        PlotGenerator.create_resource_usage_plot(
            self.detailed_results, self.dims, self.implementations,
            plt.gca(), 'thread_count', "Thread Count Comparison"
        )

        # 5. 速度向上率の比較
        plt.subplot(3, 4, 5)
        PlotGenerator.create_speedup_plot(
            self.detailed_results, self.dims, self.implementations, plt.gca()
        )

        # 6. メモリ効率（実行時間あたりのメモリ使用量）
        plt.subplot(3, 4, 6)
        for i, impl in enumerate(implementations):
            efficiency_values = []
            for dim in self.dims:
                if (impl in self.detailed_results and dim in self.detailed_results[impl] and
                    self.detailed_results[impl][dim] is not None):
                    time_per_step = self.detailed_results[impl][dim].mean_time / 1000  # 正規化
                    mem_per_step = self.detailed_results[impl][dim].memory_usage / 100
                    efficiency_values.append(mem_per_step / time_per_step if time_per_step > 0 else 0)
                else:
                    efficiency_values.append(0)
            plt.plot(self.dims, efficiency_values, 'o-', label=impl, color=colors[i], linewidth=2, markersize=6)

        plt.xlabel('Matrix Size')
        plt.ylabel('Memory/Time Ratio (MB/s)')
        plt.title('Memory Efficiency')
        plt.grid(True)
        plt.legend()

        # 7. Eigen vs SuiteSparse詳細比較
        plt.subplot(3, 4, 7)
        if 'eigen' in self.implementations and 'suitesparse' in self.implementations:
            eigen_vs_suitesparse = []
            for dim in self.dims:
                if (self.detailed_results['eigen'][dim] is not None and
                    self.detailed_results['suitesparse'][dim] is not None):
                    eigen_time = self.detailed_results['eigen'][dim].mean_time
                    suitesparse_time = self.detailed_results['suitesparse'][dim].mean_time
                    eigen_vs_suitesparse.append(eigen_time / suitesparse_time if suitesparse_time > 0 else 1.0)
                else:
                    eigen_vs_suitesparse.append(1.0)

            plt.plot(self.dims, eigen_vs_suitesparse, 'o-', label='Eigen/SuiteSparse', linewidth=2, markersize=6)
            plt.axhline(y=1.0, color='black', linestyle='--', alpha=0.5, label='Equal performance')
            plt.xlabel('Matrix Size')
            plt.ylabel('Speedup Ratio (Eigen/SuiteSparse)')
            plt.title('Eigen vs SuiteSparse Performance')
            plt.grid(True)
            plt.legend()

        # 8. CPU使用率 vs 実行時間の相関
        plt.subplot(3, 4, 8)
        for i, impl in enumerate(implementations):
            times = []
            cpus = []
            for dim in self.dims:
                if (impl in self.detailed_results and dim in self.detailed_results[impl] and
                    self.detailed_results[impl][dim] is not None):
                    times.append(self.detailed_results[impl][dim].mean_time)
                    cpus.append(self.detailed_results[impl][dim].cpu_usage)
            if times:
                plt.scatter(times, cpus, label=impl, color=colors[i], alpha=0.7)

        plt.xlabel('Execution Time (seconds)')
        plt.ylabel('CPU Usage (%)')
        plt.title('CPU Usage vs Execution Time')
        plt.grid(True)
        plt.legend()
        plt.xscale('log')

        # 9. メモリ使用量 vs 実行時間の相関
        plt.subplot(3, 4, 9)
        for i, impl in enumerate(implementations):
            times = []
            mems = []
            for dim in self.dims:
                if (impl in self.detailed_results and dim in self.detailed_results[impl] and
                    self.detailed_results[impl][dim] is not None):
                    times.append(self.detailed_results[impl][dim].mean_time)
                    mems.append(self.detailed_results[impl][dim].memory_usage)
            if times:
                plt.scatter(times, mems, label=impl, color=colors[i], alpha=0.7)

        plt.xlabel('Execution Time (seconds)')
        plt.ylabel('Memory Usage (MB)')
        plt.title('Memory Usage vs Execution Time')
        plt.grid(True)
        plt.legend()
        plt.xscale('log')

        # 10. スレッド数 vs 実行時間の相関
        plt.subplot(3, 4, 10)
        for i, impl in enumerate(implementations):
            times = []
            threads = []
            for dim in self.dims:
                if (impl in self.detailed_results and dim in self.detailed_results[impl] and
                    self.detailed_results[impl][dim] is not None):
                    times.append(self.detailed_results[impl][dim].mean_time)
                    threads.append(self.detailed_results[impl][dim].thread_count)
            if times:
                plt.scatter(times, threads, label=impl, color=colors[i], alpha=0.7)

        plt.xlabel('Execution Time (seconds)')
        plt.ylabel('Thread Count')
        plt.title('Thread Count vs Execution Time')
        plt.grid(True)
        plt.legend()
        plt.xscale('log')

        # 11. 最大次元での統計情報
        plt.subplot(3, 4, 11)
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

        # 12. システム情報
        plt.subplot(3, 4, 12)
        plt.axis('off')
        import multiprocessing
        import psutil
        system_info = f"""
        System Information:
        CPU Cores: {multiprocessing.cpu_count()}
        Total Memory: {psutil.virtual_memory().total / (1024**3):.1f} GB
        Available Memory: {psutil.virtual_memory().available / (1024**3):.1f} GB

        Benchmark Parameters:
        Selected Implementations: {', '.join(sorted(self.implementations))}
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

    def print_selectable_summary(self):
        """選択可能な実装のサマリーを表示"""
        print("\n" + "="*80)
        print("選択可能な実装のベンチマーク結果サマリー")
        print("="*80)

        implementations = sorted(self.implementations)

        # システム情報
        import multiprocessing
        import psutil
        print("\nシステム情報:")
        print(f"  CPU コア数: {multiprocessing.cpu_count()}")
        print(f"  総メモリ: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"  利用可能メモリ: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        print(f"  選択された実装: {', '.join(sorted(self.implementations))}")

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

def main():
    # 利用可能な実装
    available_implementations = ImplementationManager.get_available_implementations()

    # 比較する実装を選択（ここで変更可能）
    # selected_implementations = {'python', 'numba', 'eigen', 'eigen_direct_csr', 'suitesparse'}  # 全実装
    # selected_implementations = {'python', 'eigen', 'eigen_direct_csr'}  # PythonとEigen実装の比較
    selected_implementations = {'python', 'eigen', 'eigen_direct_csr', 'suitesparse'}  # 最適化効果の確認

    # テストする行列サイズ
    dims = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192, 16384, 32768]
    num_repeats = 10  # 各サイズでの繰り返し回数
    num_steps = 1000  # 時間発展のステップ数

    print("詳細ベンチマーク開始")
    print(f"- 行列サイズ: {dims}")
    print(f"- 繰り返し回数: {num_repeats}")
    print(f"- 時間発展ステップ数: {num_steps}")
    print(f"- 選択された実装: {', '.join(sorted(selected_implementations))}")
    print("- 追加メトリクス: CPU使用率, メモリ使用量, スレッド数")
    print("- 新機能: Phase 1-2最適化（データ変換削減 + 階層的並列化）")

    # ベンチマークの実行
    benchmark = SelectableImplementationsBenchmark(selected_implementations, dims, num_repeats, num_steps)
    results, detailed_results = benchmark.run_benchmark()

    # 結果のサマリーを表示
    benchmark.print_selectable_summary()

    # 結果をプロット
    print("\n=== Plotting Detailed Results ===")
    plot_file = benchmark.create_selectable_plots()

    # 結果をファイルに保存
    print("\n=== Saving Detailed Results ===")
    json_file, csv_file = benchmark.save_results("selectable_benchmark")

    print("\n詳細ベンチマーク完了")
    print(f"結果は{plot_file}に保存されました")

if __name__ == "__main__":
    main()
