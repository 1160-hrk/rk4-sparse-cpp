"""
全実装のベンチマーク（リファクタリング版）
"""

from benchmark_base import (
    BaseBenchmark, ImplementationManager, ResultAnalyzer, 
    ResultSaver, PlotGenerator, SAVEPATH
)
from typing import Set, List
import matplotlib.pyplot as plt
import numpy as np
from datetime import datetime
import os

class AllImplementationsBenchmark(BaseBenchmark):
    """全実装のベンチマーククラス"""
    
    def __init__(self, dims: List[int], num_repeats: int = 5, num_steps: int = 1000):
        # 利用可能な実装を取得
        available_implementations = ImplementationManager.get_available_implementations()
        
        # 全実装を使用（利用可能なもののみ）
        all_implementations = {'python', 'numba', 'eigen', 'eigen_cached', 'eigen_direct_csr', 'suitesparse'}
        implementations = all_implementations.intersection(available_implementations)
        
        print(f"利用可能な実装: {sorted(implementations)}")
        
        super().__init__(implementations, dims, num_repeats, num_steps)
    
    def create_detailed_plots(self) -> str:
        """詳細なプロットを作成"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        plot_filename = os.path.join(SAVEPATH, f'detailed_benchmark_all_implementations_{timestamp}.png')
        
        plt.figure(figsize=(24, 18))
        
        implementations = sorted(self.implementations)
        colors = ['blue', 'green', 'red', 'orange', 'purple', 'brown', 'pink', 'gray']
        
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
                result = None
                if (impl in self.detailed_results and 
                    dim in self.detailed_results[impl]):
                    result = self.detailed_results[impl][dim]
                
                if result is not None:
                    time_per_step = result.mean_time / 1000  # 正規化
                    mem_per_step = result.memory_usage / 100
                    efficiency_values.append(mem_per_step / time_per_step if time_per_step > 0 else 0)
                else:
                    efficiency_values.append(0)
            plt.plot(self.dims, efficiency_values, 'o-', label=impl, color=colors[i], linewidth=2, markersize=6)
        
        plt.xlabel('Matrix Size')
        plt.ylabel('Memory/Time Ratio (MB/s)')
        plt.title('Memory Efficiency')
        plt.grid(True)
        plt.legend()
        
        # 7. Eigen系実装の詳細比較
        plt.subplot(3, 4, 7)
        eigen_implementations = [impl for impl in implementations if 'eigen' in impl]
        if len(eigen_implementations) > 1:
            for i, impl in enumerate(eigen_implementations):
                times = []
                for dim in self.dims:
                    result = None
                    if (impl in self.detailed_results and 
                        dim in self.detailed_results[impl]):
                        result = self.detailed_results[impl][dim]
                    
                    if result is not None:
                        times.append(result.mean_time)
                    else:
                        times.append(float('inf'))
                plt.plot(self.dims, times, 'o-', label=impl, color=colors[i], linewidth=2, markersize=6)
            
            plt.xlabel('Matrix Size')
            plt.ylabel('Execution Time (seconds)')
            plt.title('Eigen Implementations Comparison')
            plt.grid(True)
            plt.legend()
            plt.yscale('log')
        
        # 8. CPU使用率 vs 実行時間の相関
        plt.subplot(3, 4, 8)
        for i, impl in enumerate(implementations):
            times = []
            cpus = []
            for dim in self.dims:
                result = None
                if (impl in self.detailed_results and 
                    dim in self.detailed_results[impl]):
                    result = self.detailed_results[impl][dim]
                
                if result is not None:
                    times.append(result.mean_time)
                    cpus.append(result.cpu_usage)
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
                result = None
                if (impl in self.detailed_results and 
                    dim in self.detailed_results[impl]):
                    result = self.detailed_results[impl][dim]
                
                if result is not None:
                    times.append(result.mean_time)
                    mems.append(result.memory_usage)
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
                result = None
                if (impl in self.detailed_results and 
                    dim in self.detailed_results[impl]):
                    result = self.detailed_results[impl][dim]
                
                if result is not None:
                    times.append(result.mean_time)
                    threads.append(result.thread_count)
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
        Dimensions: {self.dims}
        Steps: {self.num_steps}
        Repeats: {self.num_repeats}
        Implementations: {len(implementations)}
        """
        plt.text(0.1, 0.5, system_info, transform=plt.gca().transAxes, 
                 fontsize=10, verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(plot_filename, dpi=300, bbox_inches='tight')
        plt.close()
        
        return plot_filename
    
    def print_detailed_summary(self):
        """詳細な結果のサマリーを表示"""
        print("\n" + "="*80)
        print("詳細ベンチマーク結果サマリー")
        print("="*80)
        
        implementations = sorted(self.implementations)
        
        # システム情報
        import multiprocessing
        import psutil
        print(f"\nシステム情報:")
        print(f"  CPU コア数: {multiprocessing.cpu_count()}")
        print(f"  総メモリ: {psutil.virtual_memory().total / (1024**3):.1f} GB")
        print(f"  利用可能メモリ: {psutil.virtual_memory().available / (1024**3):.1f} GB")
        print(f"  テスト実装数: {len(implementations)}")
        
        # 各次元での最速実装
        print("\n各次元での最速実装:")
        for dim in self.dims:
            best_impl = None
            best_time = float('inf')
            for impl in implementations:
                result = None
                if (impl in self.detailed_results and 
                    dim in self.detailed_results[impl]):
                    result = self.detailed_results[impl][dim]
                
                if result is not None:
                    if result.mean_time < best_time:
                        best_time = result.mean_time
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
                result = None
                if (impl in self.detailed_results and 
                    dim in self.detailed_results[impl]):
                    result = self.detailed_results[impl][dim]
                
                if result is not None:
                    times.append(result.mean_time)
                    cpus.append(result.cpu_usage)
                    mems.append(result.memory_usage)
                    threads.append(result.thread_count)
            
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
            result = None
            if (impl in self.detailed_results and 
                max_dim in self.detailed_results[impl]):
                result = self.detailed_results[impl][max_dim]
            
            if result is not None:
                print(f"  {impl}:")
                print(f"    実行時間: {result.mean_time:.6f}秒")
                print(f"    CPU使用率: {result.cpu_usage:.1f}%")
                print(f"    メモリ使用量: {result.memory_usage:.1f} MB")
                print(f"    ピークメモリ: {result.memory_peak:.1f} MB")
                print(f"    スレッド数: {result.thread_count}")
                if impl != 'python':
                    print(f"    速度向上率: {result.speedup_vs_python:.2f}倍")
        
        # 実装別の総合評価
        print("\n実装別総合評価:")
        impl_scores = {}
        for impl in implementations:
            if impl == 'python':
                continue  # Pythonは基準なので除外
            
            speedups = []
            for dim in self.dims:
                result = None
                if (impl in self.detailed_results and 
                    dim in self.detailed_results[impl]):
                    result = self.detailed_results[impl][dim]
                
                if result is not None:
                    speedups.append(result.speedup_vs_python)
            
            if speedups:
                avg_speedup = np.mean(speedups)
                max_speedup = np.max(speedups)
                impl_scores[impl] = (avg_speedup, max_speedup)
        
        # 速度向上率でソート
        sorted_impls = sorted(impl_scores.items(), key=lambda x: x[1][0], reverse=True)
        for impl, (avg_speedup, max_speedup) in sorted_impls:
            print(f"  {impl}: 平均{avg_speedup:.2f}倍, 最大{max_speedup:.2f}倍")

def main():
    # テストする行列サイズ（より広範囲でテスト）
    dims = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    num_repeats = 5  # 各サイズでの繰り返し回数
    num_steps = 1000  # 時間発展のステップ数
    
    print("全実装詳細ベンチマーク開始")
    print(f"- 行列サイズ: {dims}")
    print(f"- 繰り返し回数: {num_repeats}")
    print(f"- 時間発展ステップ数: {num_steps}")
    print(f"- 対象実装: 全利用可能実装（Python, Numba, C++ Eigen系, C++ SuiteSparse等）")
    print(f"- 追加メトリクス: CPU使用率, メモリ使用量, スレッド数, 速度向上率")
    
    # ベンチマークの実行
    benchmark = AllImplementationsBenchmark(dims, num_repeats, num_steps)
    results, detailed_results = benchmark.run_benchmark()
    
    # 結果のサマリーを表示
    benchmark.print_detailed_summary()
    
    # 結果をプロット
    print("\n=== Plotting Detailed Results ===")
    plot_file = benchmark.create_detailed_plots()
    
    # 結果をファイルに保存
    print("\n=== Saving Detailed Results ===")
    json_file, csv_file = benchmark.save_results("detailed_benchmark_all_implementations")
    
    print("\n全実装詳細ベンチマーク完了")
    print(f"結果は{plot_file}に保存されました")
    print(f"データファイル: {json_file}, {csv_file}")

if __name__ == "__main__":
    main() 