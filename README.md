# Excitation RK4 Sparse

量子力学的な励起ダイナミクスを計算するための疎行列ベースのRK4ソルバー。

## 機能
- CSR形式の疎行列サポート
- OpenMPによる並列化（動的スケジューリング最適化）
- Python/C++のハイブリッド実装
- 包括的なベンチマーク機能
  - 2準位系と調和振動子のテストケース
  - 詳細なパフォーマンス分析
  - 解析解との比較
- メモリ最適化
  - キャッシュライン境界を考慮したアライメント
  - 疎行列パターンの再利用

## バージョン情報
- 現在のバージョン: v0.2.5
- ステータス: 安定版
- 最終更新: 2024-07-15
- **新機能**: プロジェクト構造の大幅改善、性能問題の解決

## 必要条件
- Python 3.10以上
- C++17対応コンパイラ
- CMake 3.16以上
- pybind11
- Eigen3
- OpenMP（推奨）

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

# C++ライブラリのビルド
./tools/build.sh --clean

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
from rk4_sparse import rk4_sparse_py, rk4_sparse_cpp

# 開発用インストールの場合
# import sys
# import os
# sys.path.append(os.path.join(os.path.dirname(__file__), 'python'))
# from rk4_sparse import rk4_sparse_py, rk4_sparse_cpp

# Python実装
result_py = rk4_sparse_py(H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm)

# C++実装（高速）
result_cpp = rk4_sparse_cpp(H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm)
```

### 例題
すべての例は`examples/python/`ディレクトリにあります：

1. **基本例**
```bash
python examples/python/two_level_excitation.py  # 2準位励起
```

2. **ベンチマーク**
```bash
python examples/python/benchmark_ho.py         # 調和振動子系での比較
```

## ベンチマーク
以下のスクリプトで様々なベンチマークを実行できます：

1. 実装間の比較
```bash
python examples/python/benchmark_ho.py         # 調和振動子系での比較
```

2. 2準位系のテスト
```bash
python examples/python/two_level_excitation.py # 2準位励起のテスト
```

## 性能

最新のベンチマーク結果（2025年7月15日）による高性能を実現：

| システムサイズ | scipy.sparse [ms] | numba [ms] | C++ [ms] | C++ vs scipy | C++ vs numba |
|-------------:|------------------:|-----------:|----------:|-------------:|-------------:|
| 2レベル       | 11.6              | 0.2        | 0.1       | **110x**     | **2.0x**     |
| 4レベル       | 10.7              | 0.2        | 0.1       | **116x**     | **2.6x**     |
| 8レベル       | 11.0              | 0.4        | 0.1       | **75x**      | **2.9x**     |
| 16レベル      | 10.9              | 1.1        | 0.2       | **53x**      | **5.5x**     |
| 32レベル      | 11.5              | 3.8        | 0.3       | **34x**      | **11.3x**    |
| 64レベル      | 12.3              | 13.6       | 0.6       | **20x**      | **22.5x**    |
| 128レベル     | 13.9              | 55.1       | 1.2       | **12x**      | **46.9x**    |
| 256レベル     | 17.5              | 230.5      | 2.4       | **7.3x**     | **96.0x**    |

## 最適化の特徴

### v0.2.0での主要改善
1. **条件付きデバッグ出力**: I/Oオーバーヘッドの除去
2. **適応的並列化**: 小規模データでのOpenMPオーバーヘッド回避
3. **最適化されたスケジューリング**: 静的スケジューリングによる効率化

### コア技術
1. **メモリアライメント**
   - キャッシュライン境界（64バイト）に合わせたアライメント
   - 作業バッファの効率的な配置

2. **適応的並列化**
   - 閾値ベースの条件分岐（10,000要素以上で並列化）
   - 静的スケジューリング最適化

3. **疎行列最適化**
   - 非ゼロパターンの事前計算
   - データ構造の再利用
   - 効率的な行列-ベクトル積

## ドキュメント

包括的なドキュメントが利用可能です：

- **開発ガイド**
  - [プロジェクト構造変更とマイグレーション](docs/development/project_restructure_migration.md)
  - [ビルドシステム設定](docs/development/build_configuration.md)

- **トラブルシューティング**
  - [性能回帰問題の分析と解決](docs/troubleshooting/performance_regression_analysis.md)

- **ベンチマーク結果**
  - [性能測定結果](docs/benchmarks/performance_results.md)

## ライセンス
MITライセンス

## 作者
- Hiroki Tsusaka
- IIS, UTokyo
- tsusaka4research "at" gmail.com

```bash
pip install -e .
