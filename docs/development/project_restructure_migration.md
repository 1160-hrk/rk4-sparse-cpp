# プロジェクト構造変更とマイグレーション

## 概要

excitation-rk4-sparseプロジェクトのディレクトリ構造を整理し、現代的なC++/Pythonハイブリッドプロジェクトの標準に準拠するための大規模なリファクタリングを実施しました。

## 変更前の構造

```
excitation-rk4-sparse/
├── excitation_rk4_sparse.cpp    # メインC++実装
├── excitation_rk4_sparse.py     # Python実装
├── rk4_sparse_py.py             # Python バインディング
├── build.sh                     # ビルドスクリプト
├── test_basic_import.py         # 簡単なテスト
├── examples/                    # 例とベンチマーク
├── tests/                       # テストファイル
└── data/                        # データと結果
```

**問題点**:
- ソースファイルがルートディレクトリに散在
- C++とPythonコードの分離が不十分
- ビルドシステムが単純すぎる
- 設定ファイルが不足

## 変更後の構造

```
excitation-rk4-sparse/
├── src/                         # C++ソースコード
│   ├── core/                    # コア実装
│   │   └── excitation_rk4_sparse.cpp
│   └── bindings/                # Pythonバインディング
│       └── python_bindings.cpp
├── python/                      # Pythonパッケージ
│   └── excitation_rk4_sparse/
│       ├── __init__.py
│       ├── rk4_sparse.py        # Python実装
│       └── bindings.py          # C++バインディング
├── tools/                       # ビルドとテストツール
│   ├── build.sh
│   └── test_examples.py
├── tests/                       # 言語別テスト
│   ├── cpp/                     # C++テスト
│   ├── python/                  # Pythonテスト
│   └── integration/             # 統合テスト
├── examples/                    # 分野別例
│   ├── basic/                   # 基本例
│   ├── quantum/                 # 量子システム例
│   └── benchmarks/              # 性能ベンチマーク
├── docs/                        # ドキュメント
│   ├── api/                     # API文書
│   ├── development/             # 開発ガイド
│   ├── troubleshooting/         # トラブルシューティング
│   └── benchmarks/              # ベンチマーク結果
├── data/                        # データと結果
│   ├── test_data/              # テストデータ
│   └── results/                # 出力結果
├── CMakeLists.txt              # CMakeビルド設定
├── CMakePresets.json           # CMake設定プリセット
├── pyproject.toml              # Python設定
└── .devcontainer/              # VS Code開発環境
```

## マイグレーション手順

### 1. 新しいディレクトリ構造の作成

```bash
# コアディレクトリの作成
mkdir -p src/core src/bindings
mkdir -p python/excitation_rk4_sparse
mkdir -p tools tests/{cpp,python,integration}
mkdir -p examples/{basic,quantum,benchmarks}
mkdir -p docs/{api,development,troubleshooting,benchmarks}
mkdir -p data/{test_data,results/figures}
```

### 2. ソースファイルの移動と整理

#### C++ファイル
```bash
# メイン実装
mv excitation_rk4_sparse.cpp src/core/

# Pythonバインディング（新規作成）
# → src/bindings/python_bindings.cpp
```

#### Pythonファイル
```bash
# Python実装
mv excitation_rk4_sparse.py python/excitation_rk4_sparse/rk4_sparse.py
mv rk4_sparse_py.py python/excitation_rk4_sparse/bindings.py

# パッケージ初期化ファイル作成
# → python/excitation_rk4_sparse/__init__.py
```

#### ツールとテスト
```bash
# ビルドツール
mv build.sh tools/

# テストファイルの分類
# → tests/python/test_import.py
# → tests/integration/test_performance.py
```

### 3. ビルドシステムの更新

#### CMakeLists.txt の作成
```cmake
cmake_minimum_required(VERSION 3.16)
project(excitation_rk4_sparse)

# 依存関係の設定
find_package(OpenMP REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(pybind11 REQUIRED)

# C++標準とコンパイルフラグ
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -march=native -ffast-math -fopenmp -flto")

# ライブラリのビルド
pybind11_add_module(_excitation_rk4_sparse
    src/core/excitation_rk4_sparse.cpp
    src/bindings/python_bindings.cpp
)
```

#### pyproject.toml の作成
```toml
[build-system]
requires = ["pybind11", "cmake"]
build-backend = "pybind11.setup_helpers.build_meta"

[project]
name = "excitation-rk4-sparse"
version = "0.1.0"
description = "High-performance sparse RK4 solver for quantum excitation dynamics"
```

### 4. 設定ファイルの追加

#### CMakePresets.json
```json
{
    "version": 3,
    "presets": [
        {
            "name": "release",
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Release"
            }
        },
        {
            "name": "debug", 
            "cacheVariables": {
                "CMAKE_BUILD_TYPE": "Debug",
                "ENABLE_PERFORMANCE_DEBUG": "ON"
            }
        }
    ]
}
```

#### .devcontainer/devcontainer.json
VS Code開発環境の設定を追加

### 5. パッケージインポートの修正

#### 変更前
```python
from excitation_rk4_sparse import ExcitationRK4Sparse
import rk4_sparse_py
```

#### 変更後
```python
from excitation_rk4_sparse import ExcitationRK4Sparse
from excitation_rk4_sparse.bindings import ExcitationRK4SparseCpp
```

### 6. 例とテストファイルの更新

全ての例ファイルで新しいインポートパスに対応：

```python
# 変更前
import sys
sys.path.append('..')
from excitation_rk4_sparse import ExcitationRK4Sparse

# 変更後  
from excitation_rk4_sparse import ExcitationRK4Sparse
```

## 発生した問題と解決

### 1. インポートエラー
**問題**: `__init__.py`でのファイル名不整合
```python
# 問題
from .rk4_sparse_py import ExcitationRK4Sparse

# 解決
from .rk4_sparse import ExcitationRK4Sparse
```

### 2. パス問題
**問題**: 例ファイルでの相対インポートエラー
**解決**: パッケージインストールによる絶対インポート

### 3. 性能回帰
**問題**: C++実装が期待より遅い
**解決**: デバッグ出力とOpenMP設定の最適化
（詳細は [performance_regression_analysis.md](../troubleshooting/performance_regression_analysis.md) を参照）

## テスト戦略

### 1. 自動テストスクリプト
`tools/test_examples.py` を作成：
- 基本インポートテスト
- 各例の実行テスト
- エラーレポート生成

### 2. 段階的テスト
1. **ビルドテスト**: `./tools/build.sh --clean`
2. **インポートテスト**: 基本的なPythonインポート
3. **機能テスト**: 数値的正確性の確認
4. **性能テスト**: ベンチマーク結果の検証

### 3. 継続的監視
- 性能回帰の早期検出
- ビルド設定の自動検証
- 例の動作確認

## 成果

### 1. コード組織化
- ✅ モジュラー構造による保守性向上
- ✅ 言語別・機能別の明確な分離
- ✅ 標準的なプロジェクト構造への準拠

### 2. 開発環境
- ✅ 現代的なビルドシステム (CMake + pybind11)
- ✅ VS Code統合開発環境
- ✅ 自動化されたテストとビルド

### 3. 性能
- ✅ 期待される高性能の維持/回復
- ✅ デバッグビルドとリリースビルドの分離
- ✅ 最適化設定の体系化

### 4. ドキュメント
- ✅ 包括的なドキュメント体系
- ✅ トラブルシューティングガイド
- ✅ 開発プロセスの記録

## 今後の開発

### 1. 継続的改善
- [ ] 自動化されたCI/CD パイプライン
- [ ] より包括的なテストスイート
- [ ] 性能監視ダッシュボード

### 2. 機能拡張
- [ ] より多くの数値解法の追加
- [ ] GPU加速対応
- [ ] 分散計算対応

### 3. ドキュメント充実
- [ ] API リファレンスの自動生成
- [ ] チュートリアルの追加
- [ ] ベンチマーク結果の定期更新

## 関連ドキュメント

- [Performance Regression Analysis](../troubleshooting/performance_regression_analysis.md)
- [Build System Configuration](build_configuration.md)
- [Testing Strategy](testing_strategy.md)
- [Development Environment Setup](development_setup.md)

## 更新履歴

- 2024-01-08: 初版作成 - プロジェクト構造変更の完全記録 