"""
共通のベンチマーク機能を提供するベースクラスとユーティリティ
"""

import sys
import os
import time
import json
import csv
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple, Optional, Set, Callable
import numpy as np
from scipy.sparse import csr_matrix
from matplotlib.axes import Axes
import psutil
import tracemalloc
import multiprocessing

# 現在のプロジェクト構造に対応
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../../python'))

# rk4_sparseモジュールのインポート
try:
    from rk4_sparse import (
        rk4_sparse_py, rk4_numba_py, rk4_sparse_eigen,
        rk4_sparse_suitesparse, rk4_sparse_eigen_cached,
        rk4_sparse_eigen_direct_csr
    )
    RK4_IMPORT_SUCCESS = True
except ImportError as e:
    print(f"Warning: Could not import rk4_sparse: {e}")
    print("Please make sure the module is built and installed correctly.")
    RK4_IMPORT_SUCCESS = False
    # ダミー関数を定義
    rk4_sparse_py = None
    rk4_numba_py = None
    rk4_sparse_eigen = None
    rk4_sparse_suitesparse = None
    rk4_sparse_eigen_cached = None
    rk4_sparse_eigen_direct_csr = None

# 保存先ディレクトリの設定
SAVEPATH = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'figures')
os.makedirs(SAVEPATH, exist_ok=True)

@dataclass
class BenchmarkResult:
    """ベンチマーク結果を格納するデータクラス"""
    implementation: str
    dimension: int
    mean_time: float
    std_time: float
    min_time: float
    max_time: float
    speedup_vs_python: float
    cpu_usage: float
    memory_usage: float
    memory_peak: float
    thread_count: int
    cache_misses: int = 0
    context_switches: int = 0
    cpu_migrations: int = 0
    timestamp: datetime = datetime.now()

    def to_dict(self) -> dict:
        """結果を辞書形式に変換（JSON保存用）"""
        result_dict = asdict(self)
        result_dict['timestamp'] = self.timestamp.isoformat()
        return result_dict

class PerformanceProfiler:
    """性能プロファイリングを行うクラス"""
    def __init__(self):
        self.process = psutil.Process()

    def profile_execution(self, func: Callable, *args, **kwargs) -> BenchmarkResult:
        """関数の実行をプロファイリング"""
        # メモリトラッキング開始
        tracemalloc.start()

        # 初期状態の記録
        initial_memory = self.process.memory_info().rss / 1024**2  # MB

        # 実行時間の計測
        start_time = time.perf_counter()
        func(*args, **kwargs)
        end_time = time.perf_counter()
        execution_time = end_time - start_time

        # 実行後の状態記録
        current_cpu_samples = [self.process.cpu_percent() for _ in range(3)]
        current_cpu = sum(current_cpu_samples) / len(current_cpu_samples)
        current_memory = self.process.memory_info().rss / 1024**2  # MB
        current_threads = self.process.num_threads()

        # メモリ使用量の計測
        current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()

        return BenchmarkResult(
            implementation=func.__name__,
            dimension=0,  # 後で設定
            mean_time=execution_time,
            std_time=0.0,  # 単回実行なので0
            min_time=execution_time,
            max_time=execution_time,
            speedup_vs_python=0.0,  # 後で計算
            cpu_usage=current_cpu,
            memory_usage=current_memory - initial_memory,
            memory_peak=peak / 1024 / 1024, # MB
            thread_count=current_threads
        )

class TestSystemGenerator:
    """テストシステムを生成するクラス"""

    @staticmethod
    def create_test_system(dim: int, num_steps: int = 1000) -> Tuple[csr_matrix, csr_matrix, csr_matrix, np.ndarray, np.ndarray, np.ndarray, float]:
        """テストシステムを生成"""
        # ハミルトニアンと双極子演算子の生成
        H0 = csr_matrix(np.diag(np.arange(dim)), dtype=np.complex128)
        mux = csr_matrix(np.eye(dim, k=1) + np.eye(dim, k=-1), dtype=np.complex128)
        muy = csr_matrix((dim, dim), dtype=np.complex128)

        # 初期状態
        psi0 = np.zeros(dim, dtype=np.complex128)
        psi0[0] = 1.0

        # 電場パラメータ
        dt_E = 0.01
        E0 = 0.1
        omega_L = 1.0
        t = np.arange(0, dt_E * (num_steps+2), dt_E)
        Ex = E0 * np.sin(omega_L * t)
        Ey = np.zeros_like(Ex)

        return H0, mux, muy, Ex, Ey, psi0, dt_E

class ImplementationManager:
    """実装の管理を行うクラス"""

    # 利用可能な実装のマッピング
    IMPLEMENTATIONS = {
        'python': rk4_sparse_py if RK4_IMPORT_SUCCESS else None,
        'numba': rk4_numba_py if RK4_IMPORT_SUCCESS else None,
        'eigen': rk4_sparse_eigen if RK4_IMPORT_SUCCESS else None,
        'eigen_cached': rk4_sparse_eigen_cached if RK4_IMPORT_SUCCESS else None,
        'eigen_direct_csr': rk4_sparse_eigen_direct_csr if RK4_IMPORT_SUCCESS else None,
        'suitesparse': rk4_sparse_suitesparse if RK4_IMPORT_SUCCESS else None,
    }

    @classmethod
    def get_available_implementations(cls) -> Set[str]:
        """利用可能な実装のリストを取得"""
        return {name for name, func in cls.IMPLEMENTATIONS.items() if func is not None}

    @classmethod
    def get_implementation_function(cls, impl_name: str) -> Optional[Callable]:
        """実装名から関数を取得"""
        return cls.IMPLEMENTATIONS.get(impl_name)

    @classmethod
    def prepare_arguments_for_implementation(cls, impl_name: str, H0: csr_matrix, mux: csr_matrix,
                                           muy: csr_matrix, Ex: np.ndarray, Ey: np.ndarray,
                                           psi0: np.ndarray, dt_E: float) -> Tuple[tuple, dict]:
        """実装に応じた引数を準備"""
        if impl_name == 'numba':
            # Numba実装用にnumpy配列に変換
            H0_numba = H0.toarray()
            mux_numba = mux.toarray()
            muy_numba = muy.toarray()
            args = (H0_numba, mux_numba, muy_numba, Ex.astype(np.float64), Ey.astype(np.float64),
                   psi0, dt_E*2, True, 1, False)
            kwargs = {}
        elif impl_name == 'eigen_direct_csr':
            # eigen_direct_csr実装用にCSR行列の個別コンポーネントを準備
            args = (
                H0.data.astype(np.complex128), H0.indices.astype(np.int32), H0.indptr.astype(np.int32),
                mux.data.astype(np.complex128), mux.indices.astype(np.int32), mux.indptr.astype(np.int32),
                muy.data.astype(np.complex128), muy.indices.astype(np.int32), muy.indptr.astype(np.int32),
                Ex.astype(np.float64), Ey.astype(np.float64), psi0.astype(np.complex128),
                dt_E*2, True, 1, False
            )
            kwargs = {}
        elif impl_name == 'suitesparse':
            # SuiteSparse実装用に追加パラメータ
            args = (H0, mux, muy, Ex, Ey, psi0, dt_E*2, True, 1, False, 1)
            kwargs = {}
        else:
            # その他の実装（Python, Eigen等）
            args = (H0, mux, muy, Ex, Ey, psi0, dt_E*2, True, 1, False)
            kwargs = {}

        return args, kwargs

class ResultAnalyzer:
    """結果分析を行うクラス"""

    @staticmethod
    def calculate_speedup(results: Dict[str, List[float]], baseline: str = 'python') -> Dict[str, float]:
        """速度向上率を計算"""
        if baseline not in results:
            return {}

        baseline_mean = np.mean(results[baseline])
        speedups = {}

        for impl, times in results.items():
            if impl != baseline and all(t != float('inf') for t in times):
                impl_mean = np.mean(times)
                if impl_mean > 0:
                    speedups[impl] = baseline_mean / impl_mean
                else:
                    speedups[impl] = 0.0

        return speedups

    @staticmethod
    def print_summary(results: Dict[str, List[float]], detailed_results: Dict[str, Optional[BenchmarkResult]],
                     dim: int, implementations: Set[str]):
        """結果のサマリーを表示"""
        print(f"\n次元 {dim} での結果サマリー:")

        # 実行時間の比較
        for impl in implementations:
            if impl in results and all(t != float('inf') for t in results[impl]):
                mean_time = np.mean(results[impl])
                std_time = np.std(results[impl])
                print(f"  {impl}: {mean_time:.6f} ± {std_time:.6f} 秒")

        # 速度向上率の計算と表示
        speedups = ResultAnalyzer.calculate_speedup(results)
        if speedups:
            print("速度向上率（Python基準）:")
            for impl, speedup in speedups.items():
                print(f"  {impl}: {speedup:.2f}倍")

class ResultSaver:
    """結果保存を行うクラス"""

    @staticmethod
    def save_to_json(results: Dict[str, Dict[int, List[float]]],
                    detailed_results: Dict[str, Dict[int, Optional[BenchmarkResult]]],
                    dims: List[int], implementations: Set[str],
                    filename_prefix: str = "benchmark") -> str:
        """結果をJSONファイルに保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        json_filename = os.path.join(SAVEPATH, f'{filename_prefix}_{timestamp}.json')

        # 結果を辞書形式に変換
        results_data = {
            'timestamp': timestamp,
            'implementations': list(implementations),
            'dimensions': dims,
            'system_info': {
                'cpu_cores': multiprocessing.cpu_count(),
                'total_memory_gb': psutil.virtual_memory().total / (1024**3),
                'available_memory_gb': psutil.virtual_memory().available / (1024**3)
            },
            'detailed_results': {}
        }

        # 詳細結果を変換
        for impl in implementations:
            results_data['detailed_results'][impl] = {}
            for dim in dims:
                if impl in detailed_results and dim in detailed_results[impl] and detailed_results[impl][dim]:
                    results_data['detailed_results'][impl][dim] = detailed_results[impl][dim].to_dict()

        with open(json_filename, 'w', encoding='utf-8') as f:
            json.dump(results_data, f, indent=2, ensure_ascii=False)

        return json_filename

    @staticmethod
    def save_to_csv(results: Dict[str, Dict[int, List[float]]],
                   detailed_results: Dict[str, Dict[int, Optional[BenchmarkResult]]],
                   dims: List[int], implementations: Set[str],
                   filename_prefix: str = "benchmark") -> str:
        """結果をCSVファイルに保存"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        csv_filename = os.path.join(SAVEPATH, f'{filename_prefix}_{timestamp}.csv')

        with open(csv_filename, 'w', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
            # ヘッダー行
            writer.writerow([
                'Dimension', 'Implementation', 'Mean_Time_seconds', 'Std_Time_seconds',
                'Min_Time_seconds', 'Max_Time_seconds', 'Speedup_vs_Python',
                'CPU_Usage_percent', 'Memory_Usage_MB', 'Memory_Peak_MB', 'Thread_Count'
            ])

            # データ行
            for dim in dims:
                for impl in implementations:
                    result = detailed_results.get(impl, {}).get(dim)
                    if result is not None:
                        writer.writerow([
                            dim, impl, result.mean_time, result.std_time,
                            result.min_time, result.max_time, result.speedup_vs_python,
                            result.cpu_usage, result.memory_usage, result.memory_peak, result.thread_count
                        ])

        return csv_filename

class PlotGenerator:
    """プロット生成を行うクラス"""

    @staticmethod
    def create_execution_time_plot(results: Dict[str, Dict[int, List[float]]],
                                 dims: List[int], implementations: Set[str],
                                 ax: Axes, title: str = "Execution Time Comparison"):
        """実行時間の比較プロットを作成"""
        x = np.arange(len(dims))
        width = 0.8 / len(implementations)

        for i, impl in enumerate(sorted(implementations)):
            means = []
            stds = []
            for dim in dims:
                if impl in results and dim in results[impl] and all(t != float('inf') for t in results[impl][dim]):
                    means.append(np.mean(results[impl][dim]))
                    stds.append(np.std(results[impl][dim]))
                else:
                    means.append(0)
                    stds.append(0)

            ax.bar(x + i*width, means, width, label=impl, yerr=stds, capsize=5)

        ax.set_xlabel('Matrix Size')
        ax.set_ylabel('Execution Time (seconds)')
        ax.set_title(title)
        ax.set_xticks(x + width * (len(implementations) - 1) / 2)
        ax.set_xticklabels([str(d) for d in dims])
        ax.legend()
        ax.grid(True)
        ax.set_yscale('log')

    @staticmethod
    def create_speedup_plot(detailed_results: Dict[str, Dict[int, Optional[BenchmarkResult]]],
                          dims: List[int], implementations: Set[str],
                          ax: Axes, baseline: str = 'python'):
        """速度向上率のプロットを作成"""
        for impl in implementations:
            if impl != baseline:
                speedup_values = []
                for dim in dims:
                    result = detailed_results.get(impl, {}).get(dim)
                    if result is not None:
                        speedup_values.append(result.speedup_vs_python)
                    else:
                        speedup_values.append(0)

                ax.plot(dims, speedup_values, 'o-', label=impl, linewidth=2, markersize=6)

        ax.set_xlabel('Matrix Size')
        ax.set_ylabel(f'Speedup Ratio (vs {baseline})')
        ax.set_title('Speedup Comparison')
        ax.grid(True)
        ax.legend()
        ax.set_yscale('log')

    @staticmethod
    def create_resource_usage_plot(detailed_results: Dict[str, Dict[int, Optional[BenchmarkResult]]],
                                 dims: List[int], implementations: Set[str],
                                 ax: Axes, metric: str = 'cpu_usage',
                                 title: str = "CPU Usage Comparison"):
        """リソース使用量のプロットを作成"""
        for impl in implementations:
            values = []
            for dim in dims:
                if (impl in detailed_results and dim in detailed_results[impl] and
                    detailed_results[impl][dim] is not None):
                    values.append(getattr(detailed_results[impl][dim], metric))
                else:
                    values.append(0)

            ax.plot(dims, values, 'o-', label=impl, linewidth=2, markersize=6)

        ax.set_xlabel('Matrix Size')
        if metric == 'cpu_usage':
            ax.set_ylabel('CPU Usage (%)')
        elif metric == 'memory_usage':
            ax.set_ylabel('Memory Usage (MB)')
        elif metric == 'thread_count':
            ax.set_ylabel('Thread Count')
        ax.set_title(title)
        ax.grid(True)
        ax.legend()

class BaseBenchmark:
    """ベンチマークの基底クラス"""

    def __init__(self, implementations: Set[str], dims: List[int],
                 num_repeats: int = 10, num_steps: int = 1000):
        self.implementations = implementations
        self.dims = dims
        self.num_repeats = num_repeats
        self.num_steps = num_steps
        self.profiler = PerformanceProfiler()

        # 結果の格納
        self.results: Dict[str, Dict[int, List[float]]] = {
            impl: {dim: [] for dim in dims} for impl in implementations
        }
        self.detailed_results: Dict[str, Dict[int, Optional[BenchmarkResult]]] = {
            impl: {dim: None for dim in dims} for impl in implementations
        }

    def run_single_benchmark(self, impl_name: str, dim: int, num_repeats: int, num_steps: int) -> Tuple[List[float], Optional[BenchmarkResult]]:
        """単一の実装でのベンチマークを実行"""
        print(f"  {impl_name}実装をテスト中...")

        # 実装関数を取得
        impl_func = ImplementationManager.get_implementation_function(impl_name)
        if impl_func is None:
            print(f"    {impl_name}実装が見つかりません")
            return [float('inf')] * num_repeats, None

        # テストシステムを生成
        H0, mux, muy, Ex, Ey, psi0, dt_E = TestSystemGenerator.create_test_system(dim, num_steps)

        # 実装に応じた引数を準備
        args, kwargs = ImplementationManager.prepare_arguments_for_implementation(
            impl_name, H0, mux, muy, Ex, Ey, psi0, dt_E
        )

        # 実行時間の計測
        times = []
        detailed_result = None

        try:
            # 最初の実行でプロファイリング
            detailed_result = self.profiler.profile_execution(impl_func, *args, **kwargs)
            detailed_result.implementation = impl_name
            detailed_result.dimension = dim

            # 繰り返し実行で統計を取得
            for i in range(num_repeats):
                start_time = time.perf_counter()
                result = impl_func(*args, **kwargs)
                end_time = time.perf_counter()
                times.append(end_time - start_time)

            # 統計情報を更新
            detailed_result.mean_time = float(np.mean(times))
            detailed_result.std_time = float(np.std(times))
            detailed_result.min_time = float(np.min(times))
            detailed_result.max_time = float(np.max(times))

            print(f"    実行時間: {detailed_result.mean_time:.6f} ± {detailed_result.std_time:.6f} 秒")

        except Exception as e:
            print(f"    {impl_name}実装でエラーが発生: {e}")
            times = [float('inf')] * num_repeats
            detailed_result = None

        return times, detailed_result

    def run_benchmark(self) -> Tuple[Dict[str, Dict[int, List[float]]],
                                   Dict[str, Dict[int, Optional[BenchmarkResult]]]]:
        """ベンチマークを実行"""
        print("ベンチマーク開始")
        print(f"- 行列サイズ: {self.dims}")
        print(f"- 繰り返し回数: {self.num_repeats}")
        print(f"- 時間発展ステップ数: {self.num_steps}")
        print(f"- 実装: {', '.join(sorted(self.implementations))}")

        for dim in self.dims:
            print(f"\n次元数: {dim}")

            for impl in self.implementations:
                times, detailed_result = self.run_single_benchmark(
                    impl, dim, self.num_repeats, self.num_steps
                )
                self.results[impl][dim] = times
                self.detailed_results[impl][dim] = detailed_result

            # 速度向上率を計算
            if 'python' in self.results and all(t != float('inf') for t in self.results['python'][dim]):
                python_mean = np.mean(self.results['python'][dim])
                for impl in self.implementations:
                    if impl != 'python':
                        result = self.detailed_results.get(impl, {}).get(dim)
                        if result is not None:
                            impl_mean = result.mean_time
                            if impl_mean > 0:
                                result.speedup_vs_python = float(python_mean / impl_mean)
                            else:
                                result.speedup_vs_python = 0.0

        return self.results, self.detailed_results

    def save_results(self, filename_prefix: str = "benchmark") -> Tuple[str, str]:
        """結果をファイルに保存"""
        json_file = ResultSaver.save_to_json(
            self.results, self.detailed_results, self.dims, self.implementations, filename_prefix
        )
        csv_file = ResultSaver.save_to_csv(
            self.results, self.detailed_results, self.dims, self.implementations, filename_prefix
        )
        return json_file, csv_file

