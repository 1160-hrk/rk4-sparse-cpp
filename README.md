# Excitation RK4 Sparse

[![CI](https://github.com/1160-hrk/rk4-sparse-cpp/workflows/CI/badge.svg)](https://github.com/1160-hrk/rk4-sparse-cpp/actions) [![PyPI version](https://badge.fury.io/py/rk4-sparse-cpp.svg)](https://badge.fury.io/py/rk4-sparse-cpp) [![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT) [![C++17](https://img.shields.io/badge/C%2B%2B-17-blue.svg)](https://en.cppreference.com/w/cpp/17)

量子力学的な励起ダイナミクスを計算するための疎行列ベースのRK4ソルバー。

## 機能
- CSR形式の疎行列サポート
- OpenMPによる並列化（動的スケジューリング最適化）
- 複数の実装バリエーション
  - **Python実装**: 開発・デバッグ用
  - **Numba実装**: 小次元での高速化
  - **Eigen実装**: 標準的なC++実装
  - **Eigen_Cached実装**: 大次元での最適化
  - **Eigen_Direct_CSR実装**: 小次元での最高性能
  - **SuiteSparse実装**: メモリ効率重視
  - **SuiteSparse-MKL実装**: Intel MKLによる最速実装
- 包括的なベンチマーク機能
  - 実装間の詳細な性能比較
  - 次元別最適化推奨
  - メモリ使用量・CPU使用率分析
- メモリ最適化
  - キャッシュライン境界を考慮したアライメント
  - 疎行列パターンの再利用

## バージョン情報
- 現在のバージョン: v0.2.6
- ステータス: 安定版
- 最終更新: 2025-07-18
- **新機能**: 複数実装の統合、詳細なベンチマーク分析

## 必要条件
- Python 3.10以上
- C++17対応コンパイラ
- CMake 3.16以上
- pybind11
- Eigen3
- OpenMP（推奨）
- **オプション**: Intel MKL（SuiteSparse-MKL版を使用する場合）

## インストール

### pip install（推奨）
```bash
pip install rk4-sparse-cpp
```

この場合、`rk4_sparse`モジュールがsite-packagesにインストールされます。

### 開発用インストール
```bash
git clone https://github.com/1160-hrk/excitation-rk4-sparse.git
cd excitation-rk4-sparse

# Eigen版のビルド（デフォルト）
./tools/build.sh --clean

# SuiteSparse-MKL版のビルド（オプション）
./build_suitesparse.sh

# Pythonパッケージのインストール
pip install -e .

# または、直接パスを追加して使用
# sys.path.append('python')
```

### クイックテスト
```bash
# 2準位系のテスト
python examples/python/two_level_excitation.py

# 調和振動子のベンチマーク
python examples/python/benchmark_ho.py
```

## 使用例

### 基本的な使用法
```python
# pip installでインストールした場合
from rk4_sparse import (
    rk4_sparse_py, 
    rk4_sparse_eigen, 
    rk4_sparse_eigen_cached,
    rk4_sparse_eigen_direct_csr,
    rk4_sparse_suitesparse,
    rk4_sparse_suitesparse_mkl,
    benchmark_implementations
)

# 開発用インストールの場合
# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))
# from rk4_sparse import *

# Python実装（開発・デバッグ用）
result_py = rk4_sparse_py(H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm)

# Eigen版（標準的なC++実装）
result_eigen = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm)

# Eigen_Cached版（大次元での最適化）
if rk4_sparse_eigen_cached is not None:
    result_cached = rk4_sparse_eigen_cached(H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm)

# Eigen_Direct_CSR版（小次元での最高性能）
if rk4_sparse_eigen_direct_csr is not None:
    result_direct = rk4_sparse_eigen_direct_csr(H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm)

# SuiteSparse版（メモリ効率重視）
if rk4_sparse_suitesparse is not None:
    result_suitesparse = rk4_sparse_suitesparse(H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm)

# SuiteSparse-MKL版（最速、Intel MKL利用）
if rk4_sparse_suitesparse_mkl is not None:
    result_mkl = rk4_sparse_suitesparse_mkl(H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm)

# 実装間のベンチマーク
results = benchmark_implementations(H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm, num_runs=5)

# 利用可能な実装の確認
print(f"Eigen available: {rk4_sparse_eigen is not None}")
print(f"Eigen_Cached available: {rk4_sparse_eigen_cached is not None}")
print(f"Eigen_Direct_CSR available: {rk4_sparse_eigen_direct_csr is not None}")
print(f"SuiteSparse available: {rk4_sparse_suitesparse is not None}")
print(f"SuiteSparse-MKL available: {rk4_sparse_suitesparse_mkl is not None}")

### 例題
すべての例は`examples/python/`ディレクトリにあります：

1. **基本例**
```bash
python examples/python/two_level_excitation.py  # 2準位励起（Python/C++比較）
```

2. **ベンチマーク**
```bash
python examples/python/benchmark_ho.py         # 調和振動子系での全実装比較
```

3. **詳細分析**
```bash
# ベンチマーク結果の詳細分析
python examples/python/analyze_benchmark_results.py
```

## ベンチマーク

### 最新の性能結果（2025年1月）

#### 実装別性能比較
| 実装 | 小次元(<128) | 中次元(128-1024) | 大次元(>1024) | メモリ効率 | 推奨用途 |
|------|-------------|-----------------|---------------|-----------|----------|
| **Eigen_Direct_CSR** | **最高** | 良好 | 劣化 | 高 | 小次元リアルタイム |
| **Eigen_Cached** | 良好 | 良好 | **最高** | 高 | 汎用・大次元 |
| **Eigen** | 良好 | 良好 | 良好 | 高 | 標準 |
| **SuiteSparse** | 良好 | 良好 | 良好 | **最高** | メモリ制約環境 |
| **Numba** | 良好 | 劣化 | **不可** | 低 | 小次元のみ |
| **Python** | 基準 | 基準 | 基準 | 中 | 開発・デバッグ |

#### 高速化倍率（Python基準）
- **小次元（2-64）**: 100倍以上の高速化
- **中次元（128-512）**: 10-50倍の高速化
- **大次元（1024+）**: 1-5倍の高速化

### ベンチマーク実行
```bash
# 全実装の比較
python examples/python/benchmark_ho.py

# 2準位系のテスト
python examples/python/two_level_excitation.py

# 詳細分析
python examples/python/analyze_benchmark_results.py
```

### プログラム内での比較
```python
# 実装間の速度比較
results = benchmark_implementations(H0, mux, muy, Ex, Ey, psi0, dt, True, 1, False, 5)
for result in results:
    print(f"{result.implementation}: {result.total_time:.6f}秒 (Eigen比: {result.speedup_vs_eigen:.3f}x)")
```

## 性能

### 詳細な性能比較（2025年1月）

#### 実行時間比較（ミリ秒）
| 次元 | Python | Numba | Eigen | Eigen_Cached | Eigen_Direct_CSR | SuiteSparse |
|------|--------|-------|-------|--------------|------------------|-------------|
| 2    | 11.75  | 0.17  | 0.075 | 0.076        | **0.071**        | 0.081       |
| 4    | 12.18  | 0.26  | 0.090 | 0.089        | **0.086**        | 0.091       |
| 8    | 12.90  | 0.44  | 0.118 | 0.121        | 0.132            | 0.123       |
| 16   | 12.60  | 1.20  | 0.179 | 0.188        | **0.178**        | 0.180       |
| 32   | 12.65  | 3.72  | 0.304 | 0.308        | **0.302**        | 0.307       |
| 64   | 13.70  | 14.28 | 0.554 | **0.526**    | 0.549            | 0.539       |
| 128  | 15.09  | 59.99 | 1.104 | **1.056**    | 1.072            | 1.052       |
| 256  | 18.80  | 271.33| 2.299 | 2.622        | 2.858            | **2.204**    |
| 512  | 28.41  | 994.41| 5.792 | **5.059**    | 8.504            | 7.052       |
| 1024 | 38.49  | 3954.5| 14.785| **10.704**   | 14.288           | 14.944      |
| 2048 | 66.62  | -     | 34.796| **22.810**   | -                | 32.717      |
| 4096 | 103.88 | -     | 86.663| **42.129**   | -                | 86.115      |

#### 高速化倍率（Python基準）
| 次元 | Numba | Eigen | Eigen_Cached | Eigen_Direct_CSR | SuiteSparse |
|------|-------|-------|--------------|------------------|-------------|
| 2    | 67.4x | 155.9x| 154.0x       | **164.8x**       | 145.3x      |
| 4    | 46.5x | 134.8x| 136.8x       | **141.7x**       | 133.2x      |
| 8    | 29.2x | 109.1x| 106.3x       | 97.9x            | 105.2x      |
| 16   | 10.5x | 70.4x | 67.1x        | **70.7x**        | 70.2x       |
| 32   | 3.4x  | 41.7x | 41.1x        | **41.9x**        | 41.3x       |
| 64   | 1.0x  | 24.7x | **26.0x**    | 25.0x            | 25.4x       |
| 128  | 0.3x  | 13.7x | **14.3x**    | 14.1x            | 14.3x       |
| 256  | 0.07x | 8.2x  | 7.2x         | 6.6x             | **8.5x**    |
| 512  | 0.03x | 4.9x  | **5.6x**     | 3.3x            | 4.0x        |
| 1024 | 0.01x | 2.6x  | **3.6x**     | 2.7x            | 2.6x        |
| 2048 | -     | 1.9x  | **2.9x**     | -               | 2.0x        |
| 4096 | -     | 1.2x  | **2.5x**     | -               | 1.2x        |

## 最適化の特徴

### v0.2.6での主要改善
1. **複数実装の統合**: 6つの異なる実装バリエーション
2. **キャッシュ最適化**: 大次元での顕著な性能向上（最大50%改善）
3. **次元別最適化**: 用途に応じた最適実装の自動選択
4. **詳細なベンチマーク**: 包括的な性能分析機能

### コア技術
1. **メモリアライメント**
   - キャッシュライン境界（64バイト）に合わせたアライメント
   - 作業バッファの効率的な配置

2. **適応的並列化**
   - 閾値ベースの条件分岐（10,000要素以上で並列化）
   - 静的スケジューリング最適化

3. **疎行列最適化**
   - 非ゼロパターンの事前計算とキャッシュ
   - データ構造の再利用
   - 効率的な行列-ベクトル積

4. **実装別最適化**
   - **Eigen_Direct_CSR**: 小次元での直接CSR操作
   - **Eigen_Cached**: 大次元でのキャッシュ効果活用
   - **SuiteSparse**: メモリ効率重視の最適化

## ドキュメント

包括的なドキュメントが利用可能です：

- **開発ガイド**
  - [プロジェクト構造変更とマイグレーション](docs/development/project_restructure_migration.md)
  - [ビルドシステム設定](docs/development/build_configuration.md)

- **トラブルシューティング**
  - [性能回帰問題の分析と解決](docs/troubleshooting/performance_regression_analysis.md)

- **ベンチマーク結果**
  - [詳細な性能比較分析](docs/development/benchmark_comparison_analysis_20250718.md)
  - [統計サマリー](docs/development/benchmark_summary_statistics.csv)

- **実装選択ガイド**
  - **小次元（<128）**: Eigen_Direct_CSR
  - **中次元（128-1024）**: Eigen_Cached
  - **大次元（>1024）**: Eigen_Cached
  - **メモリ制約環境**: SuiteSparse
  - **開発・デバッグ**: Python実装

## ライセンス
MITライセンス

## 作者
- Hiroki Tsusaka
- IIS, UTokyo
- tsusaka4research "at" gmail.com

```bash
pip install -e .
