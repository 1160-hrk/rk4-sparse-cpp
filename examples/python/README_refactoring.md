# ベンチマークコードのリファクタリング

## 概要

元のベンチマークコードを共通のベースクラスとユーティリティを使用してリファクタリングしました。これにより、コードの重複を削減し、保守性と拡張性を向上させました。

## リファクタリングの内容

### 1. 共通ベースクラス (`benchmark_base.py`)

以下の共通機能を提供するベースクラスを作成しました：

- **`BenchmarkResult`**: ベンチマーク結果を格納するデータクラス
- **`PerformanceProfiler`**: 性能プロファイリングを行うクラス
- **`TestSystemGenerator`**: テストシステムを生成するクラス
- **`ImplementationManager`**: 実装の管理を行うクラス
- **`BenchmarkRunner`**: ベンチマーク実行を管理するクラス
- **`ResultAnalyzer`**: 結果分析を行うクラス
- **`ResultSaver`**: 結果保存を行うクラス
- **`PlotGenerator`**: プロット生成を行うクラス
- **`BaseBenchmark`**: ベンチマークの基底クラス

### 2. リファクタリングされたファイル

#### 元のファイル → リファクタリング版

1. **`benchmark_all_implementations.py`** → **`benchmark_all_implementations_refactored.py`**
   - 全実装（Python, Numba, Eigen, SuiteSparse）の比較
   - 詳細なプロット（12個のサブプロット）

2. **`benchmark_cached_implementation.py`** → **`benchmark_cached_implementation_refactored.py`**
   - キャッシュ化実装の効果比較
   - Python, Eigen, Eigen Cachedの比較

3. **`benchmark_selectable_implementations.py`** → **`benchmark_selectable_implementations_refactored.py`**
   - 選択可能な実装の比較
   - カスタマイズ可能な実装選択

4. **`benchmark_suitesparse_improvements.py`** → **`benchmark_suitesparse_improvements_refactored.py`**
   - SuiteSparse改善効果の分析
   - Eigen vs SuiteSparseの比較

## 使用方法

### 基本的な使用方法

```python
from benchmark_all_implementations_refactored import AllImplementationsBenchmark

# ベンチマークの実行
dims = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096]
benchmark = AllImplementationsBenchmark(dims, num_repeats=5, num_steps=1000)
results, detailed_results = benchmark.run_benchmark()

# 結果の表示
benchmark.print_detailed_summary()

# プロットの作成
plot_file = benchmark.create_detailed_plots()

# 結果の保存
json_file, csv_file = benchmark.save_results("my_benchmark")
```

### カスタム実装の選択

```python
from benchmark_selectable_implementations_refactored import SelectableImplementationsBenchmark

# 特定の実装のみを選択
selected_implementations = {'python', 'eigen', 'suitesparse'}
benchmark = SelectableImplementationsBenchmark(
    selected_implementations, dims, num_repeats=10, num_steps=1000
)
results, detailed_results = benchmark.run_benchmark()
```

### キャッシュ化効果の確認

```python
from benchmark_cached_implementation_refactored import CachedImplementationBenchmark

# キャッシュ化実装の比較
benchmark = CachedImplementationBenchmark(dims, num_repeats=10, num_steps=1000)
results, detailed_results = benchmark.run_benchmark()
benchmark.print_cached_summary()
```

## 主な改善点

### 1. コードの重複削減
- 共通のベンチマークロジックを`BaseBenchmark`クラスに統合
- プロット生成、結果保存、分析機能を共通化

### 2. 保守性の向上
- 実装の追加・変更が容易
- 共通機能の修正が一箇所で可能

### 3. 拡張性の向上
- 新しいベンチマークタイプの追加が簡単
- カスタム分析機能の追加が容易

### 4. エラーハンドリングの改善
- 実装の存在チェック
- 適切なエラーメッセージの表示

### 5. 設定の柔軟性
- 実装の選択が自由
- パラメータのカスタマイズが容易

## ファイル構造

```
examples/python/
├── benchmark_base.py                           # 共通ベースクラス
├── benchmark_all_implementations_refactored.py # 全実装比較
├── benchmark_cached_implementation_refactored.py # キャッシュ化比較
├── benchmark_comparison_refactored.py          # 基本比較
├── benchmark_selectable_implementations_refactored.py # 選択可能比較
├── benchmark_suitesparse_improvements_refactored.py # SuiteSparse改善
└── README_refactoring.md                       # このファイル
```

## 実行例

### 全実装の比較
```bash
python benchmark_all_implementations_refactored.py
```

### キャッシュ化効果の確認
```bash
python benchmark_cached_implementation_refactored.py
```

### カスタム実装の選択
```bash
python benchmark_selectable_implementations_refactored.py
```

## 注意事項

1. **依存関係**: `rk4_sparse`モジュールが必要です
2. **利用可能な実装**: システムにインストールされている実装のみが利用可能です
3. **メモリ使用量**: 大きな行列サイズでは十分なメモリが必要です
4. **実行時間**: ベンチマークの実行には時間がかかる場合があります

## 今後の拡張

1. **新しい実装の追加**: `ImplementationManager`に新しい実装を追加
2. **新しい分析機能**: `ResultAnalyzer`に新しい分析メソッドを追加
3. **新しいプロットタイプ**: `PlotGenerator`に新しいプロット機能を追加
4. **並列実行**: 複数のベンチマークを並列実行する機能

## トラブルシューティング

### よくある問題

1. **ImportError**: `rk4_sparse`モジュールが見つからない
   - 解決策: モジュールをビルドしてインストール

2. **MemoryError**: メモリ不足
   - 解決策: 行列サイズを小さくするか、メモリを増やす

3. **実装が見つからない**: 特定の実装が利用できない
   - 解決策: 利用可能な実装を確認し、適切な実装を選択

### デバッグ

デバッグ情報を有効にするには、`BaseBenchmark`クラスの`run_benchmark`メソッドにログ出力を追加してください。 