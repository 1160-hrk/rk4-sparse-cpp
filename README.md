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
- 現在のバージョン: v0.2.0
- ステータス: 安定版
- 最終更新: 2024-01-08
- **新機能**: プロジェクト構造の大幅改善、性能問題の解決

## 必要条件
- Python 3.10以上
- C++17対応コンパイラ
- CMake 3.16以上
- pybind11
- Eigen3
- OpenMP（推奨）

## インストール

### 開発用インストール（推奨）
```bash
git clone https://github.com/1160-hrk/excitation-rk4-sparse.git
cd excitation-rk4-sparse

# C++ライブラリのビルド
./tools/build.sh --clean

# Pythonパッケージのインストール
pip install -e .
```

### クイックテスト
```bash
# 自動テストの実行
python tools/test_examples.py
```

## 使用例

### 基本的な使用法
```python
from excitation_rk4_sparse import ExcitationRK4Sparse
from excitation_rk4_sparse.bindings import ExcitationRK4SparseCpp

# Python実装
solver_py = ExcitationRK4Sparse()
result_py = solver_py.solve(H0, mux, muy, Ex, Ey, psi0, dt, True, stride, False)

# C++実装（高速）
solver_cpp = ExcitationRK4SparseCpp()
result_cpp = solver_cpp.solve(H0, mux, muy, Ex, Ey, psi0, dt, True, stride, False)
```

### 例題
すべての例は新しい構造で整理されています：

1. **基本例**
```bash
python examples/basic/example_rk4.py  # 簡単な2準位系
```

2. **量子システム例**
```bash
python examples/quantum/two_level_excitation.py   # 2準位励起
python examples/quantum/harmonic_oscillator.py   # 調和振動子
```

3. **ベンチマーク**
```bash
python examples/benchmarks/benchmark_comparison.py
```

## ベンチマーク
以下のスクリプトで様々なベンチマークを実行できます：

1. 実装間の比較
```bash
python examples/benchmark_comparison.py  # 基本的な比較
python examples/benchmark_ho.py         # 調和振動子系での比較
```

2. 詳細なプロファイリング
```bash
python examples/profile_comparison.py   # CPU使用率、メモリ使用量など
```

## 性能

最新の最適化により期待される高性能を実現：

| システムサイズ | Python [ms] | C++ [ms] | 速度比 |
|-------------:|------------:|----------:|-------:|
| 50レベル      | 12.8        | 0.5       | **23.65x** |
| 100レベル     | 14.5        | 0.9       | **15.69x** |
| 200レベル     | 17.3        | 1.8       | **9.81x** |
| 500レベル     | 12.2        | 2.9       | **4.29x** |

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
