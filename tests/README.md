# RK4 Sparse C++ テストスイート

このディレクトリには、rk4-sparse-cppプロジェクトの包括的なテストスイートが含まれています。

## テスト構成

### `python/` - Python単体テスト
- **`test_rk4_basic.py`** - 基本的な機能テスト
- **`test_rk4_sparse_eigen.py`** - Eigen実装の包括的テスト
- **`test_rk4_sparse_suitesparse.py`** - SuiteSparse実装のテスト
- **`test_utils.py`** - ユーティリティ関数のテスト

### `integration/` - 統合テスト
- **`test_implementation_comparison.py`** - 実装間の比較テスト
- **`test_segmentation_fault_fix.py`** - セグメンテーション違反修正のテスト
- **`test_suitesparse_integration.py`** - SuiteSparse統合テスト

### `cpp/` - C++単体テスト
- **`test_sparse_matrix_operations.cpp`** - C++スパース行列操作のテスト
- **`CMakeLists.txt`** - C++テスト用のビルド設定

## テストの実行

### Pythonテスト
```bash
# すべてのPythonテストを実行
pytest tests/python/ -v

# 特定のテストファイルを実行
pytest tests/python/test_rk4_sparse_eigen.py -v

# カバレッジ付きで実行
pytest tests/ -v --cov=rk4_sparse --cov-report=html
```

### 統合テスト
```bash
# 統合テストを実行
pytest tests/integration/ -v

# 特定の統合テストを実行
pytest tests/integration/test_implementation_comparison.py -v
```

### C++テスト
```bash
# C++テストをビルドして実行
cd tests/cpp
cmake .
make
./test_sparse_matrix_operations
```

## テストカバレッジ

テストは以下の領域をカバーしています：

### 機能テスト
- 基本的なRK4ソルバーの動作
- 異なる次元での計算
- 異なるパラメータ（stride、renorm等）での動作
- 入力検証とエラーハンドリング

### 数値テスト
- 数値安定性の確認
- 保存量（確率）の検証
- 複素数入力の処理
- エルミート性の確認

### 実装比較
- Eigen実装とSuiteSparse実装の結果一致
- ベンチマーク機能の動作確認
- 大規模問題での性能比較

### 統合テスト
- セグメンテーション違反修正の確認
- 完全なワークフローのテスト
- 異なるプラットフォームでの動作確認

## CI/CD統合

GitHub Actionsで以下のテストが自動実行されます：

1. **マルチプラットフォームテスト** - Ubuntu、macOS、Windows
2. **マルチPythonバージョンテスト** - Python 3.9-3.12
3. **コード品質チェック** - black、flake8、mypy
4. **ドキュメントビルド** - Sphinx
5. **カバレッジレポート** - Codecov

## トラブルシューティング

### インポートエラー
C++拡張がビルドされていない場合、一部のテストがスキップされます：
```python
if not CPP_AVAILABLE:
    pytest.skip("C++ extension not available")
```

### メモリ不足
大規模テストでメモリ不足が発生する場合、テストの次元を小さくしてください。

### プラットフォーム固有の問題
- **macOS**: Eigenのインストールが必要
- **Windows**: CMakeとVisual Studio Build Toolsが必要
- **Linux**: libeigen3-devパッケージが必要 