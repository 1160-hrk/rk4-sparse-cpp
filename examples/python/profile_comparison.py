"""Python実装とC++実装の性能比較を行うモジュール"""

import os
from datetime import datetime

from profile_python import main as python_main
from profile_cpp import main as cpp_main
from profile_common import plot_comparison_metrics, print_comparison_results

def main():
    """Main function for implementation comparison"""
    print("Starting implementation comparison...")
    
    # Python実装のプロファイリング
    print("\n=== Running Python Implementation ===")
    py_profiler, steps_list, py_output_dir = python_main()
    
    # C++実装のプロファイリング
    print("\n=== Running C++ Implementation ===")
    cpp_profiler, _, cpp_output_dir = cpp_main()
    
    # 比較結果の出力ディレクトリ
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    comparison_dir = os.path.join("examples", "figures", f"implementation_comparison_{timestamp}")
    os.makedirs(comparison_dir, exist_ok=True)
    
    # 比較結果の表示
    print_comparison_results(py_profiler, cpp_profiler, steps_list)
    
    # 比較グラフの作成
    plot_comparison_metrics(py_profiler, cpp_profiler, steps_list, comparison_dir)
    
    print(f"\nComparison results saved to: {comparison_dir}")

if __name__ == "__main__":
    main() 