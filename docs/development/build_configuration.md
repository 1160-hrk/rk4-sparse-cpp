# ビルド設定ガイド

このドキュメントでは、rk4-sparse-cppプロジェクトのビルド設定について詳しく説明します。

## 概要

rk4-sparse-cppは複数の実装バリエーションを提供しており、それぞれ異なるビルド設定が必要です：

- **Eigen実装**: 標準的なC++/Eigen実装
- **SuiteSparse実装**: OpenBLAS + SuiteSparse実装
- **SuiteSparse-MKL実装**: Intel MKL + SuiteSparse実装

## 必要条件

### 基本要件
- **Python**: 3.10以上
- **C++コンパイラ**: C++17対応
  - GCC 7.0以上
  - Clang 5.0以上
  - MSVC 2019以上（Windows）
- **CMake**: 3.16以上
- **pybind11**: 2.10以上

### 依存ライブラリ

#### Eigen実装
- **Eigen3**: 3.3.0以上
- **OpenMP**: 推奨（並列化のため）

#### SuiteSparse実装
- **SuiteSparse**: 5.10.0以上
- **OpenBLAS**: 0.3.0以上
- **OpenMP**: 推奨

#### SuiteSparse-MKL実装
- **SuiteSparse**: 5.10.0以上
- **Intel MKL**: 2020.0以上
- **OpenMP**: Intel OpenMP（MKLと一緒にインストール）

## インストール方法

### 1. システム依存関係のインストール

#### Ubuntu/Debian
```bash
# 基本開発ツール
sudo apt update
sudo apt install build-essential cmake python3-dev

# Eigen3
sudo apt install libeigen3-dev

# OpenMP
sudo apt install libomp-dev

# SuiteSparse + OpenBLAS
sudo apt install libsuitesparse-dev libopenblas-dev

# Intel MKL（オプション）
# Intel oneAPI Base Toolkitをインストール
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html
```

#### CentOS/RHEL/Fedora
```bash
# 基本開発ツール
sudo yum groupinstall "Development Tools"
sudo yum install cmake3 python3-devel

# Eigen3
sudo yum install eigen3-devel

# OpenMP
sudo yum install libomp-devel

# SuiteSparse + OpenBLAS
sudo yum install suitesparse-devel openblas-devel
```

#### macOS
```bash
# Homebrewを使用
brew install cmake eigen openblas suite-sparse libomp

# Intel MKL（オプション）
brew install intel-oneapi-mkl
```

#### Windows
```bash
# vcpkgを使用
vcpkg install eigen3 openblas suitesparse

# Intel MKL（オプション）
# Intel oneAPI Base Toolkitをインストール
```

### 2. Python依存関係のインストール

```bash
# 基本依存関係
pip install numpy scipy pybind11

# 開発用依存関係
pip install pytest pytest-benchmark matplotlib
```

## ビルド設定

### 1. Eigen版のビルド（デフォルト）

```bash
# プロジェクトのクローン
git clone https://github.com/1160-hrk/excitation-rk4-sparse.git
cd excitation-rk4-sparse

# Eigen版のビルド
./tools/build.sh --clean

# または手動でビルド
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SUITESPARSE=OFF
make -j$(nproc)
```

### 2. SuiteSparse版のビルド

```bash
# SuiteSparse版のビルド
./build_suitesparse.sh

# または手動でビルド
mkdir build_suitesparse && cd build_suitesparse
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SUITESPARSE=ON -DBUILD_MKL=OFF
make -j$(nproc)
```

### 3. SuiteSparse-MKL版のビルド

```bash
# SuiteSparse-MKL版のビルド
./build_suitesparse_mkl.sh

# または手動でビルド
mkdir build_mkl && cd build_mkl
cmake .. -DCMAKE_BUILD_TYPE=Release -DBUILD_SUITESPARSE=ON -DBUILD_MKL=ON
make -j$(nproc)
```

## CMake設定オプション

### 基本オプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `CMAKE_BUILD_TYPE` | Debug | ビルドタイプ（Debug/Release/RelWithDebInfo） |
| `BUILD_SUITESPARSE` | OFF | SuiteSparse実装のビルド |
| `BUILD_MKL` | OFF | Intel MKLの使用 |
| `BUILD_TESTS` | ON | テストのビルド |
| `BUILD_BENCHMARKS` | ON | ベンチマークのビルド |

### 高度なオプション

| オプション | デフォルト | 説明 |
|-----------|-----------|------|
| `EIGEN3_INCLUDE_DIR` | 自動検出 | Eigen3のインクルードディレクトリ |
| `SUITESPARSE_INCLUDE_DIR` | 自動検出 | SuiteSparseのインクルードディレクトリ |
| `SUITESPARSE_LIBRARY_DIR` | 自動検出 | SuiteSparseのライブラリディレクトリ |
| `MKL_ROOT` | 自動検出 | Intel MKLのルートディレクトリ |
| `OPENMP_CXX_FLAGS` | 自動検出 | OpenMPのコンパイルフラグ |

### 使用例

```bash
# カスタムEigen3パスの指定
cmake .. -DEIGEN3_INCLUDE_DIR=/usr/local/include/eigen3

# カスタムSuiteSparseパスの指定
cmake .. -DSUITESPARSE_INCLUDE_DIR=/opt/suitesparse/include \
         -DSUITESPARSE_LIBRARY_DIR=/opt/suitesparse/lib

# Intel MKLパスの指定
cmake .. -DMKL_ROOT=/opt/intel/oneapi/mkl/latest

# デバッグビルド
cmake .. -DCMAKE_BUILD_TYPE=Debug

# テスト無効化
cmake .. -DBUILD_TESTS=OFF
```

## 環境変数

### コンパイラ設定

```bash
# GCC使用時
export CC=gcc
export CXX=g++

# Clang使用時
export CC=clang
export CXX=clang++

# Intel MKL使用時
export MKLROOT=/opt/intel/oneapi/mkl/latest
export LD_LIBRARY_PATH=$MKLROOT/lib/intel64:$LD_LIBRARY_PATH
```

### 並列化設定

```bash
# OpenMPスレッド数
export OMP_NUM_THREADS=4

# Intel MKLスレッド数
export MKL_NUM_THREADS=4
```

## トラブルシューティング

### よくある問題

#### 1. Eigen3が見つからない
```bash
# エラー: Could not find Eigen3
sudo apt install libeigen3-dev  # Ubuntu/Debian
# または
brew install eigen  # macOS
```

#### 2. SuiteSparseが見つからない
```bash
# エラー: Could not find SuiteSparse
sudo apt install libsuitesparse-dev  # Ubuntu/Debian
# または
brew install suite-sparse  # macOS
```

#### 3. Intel MKLが見つからない
```bash
# エラー: Could not find Intel MKL
# Intel oneAPI Base Toolkitをインストール
# https://www.intel.com/content/www/us/en/developer/tools/oneapi/base-toolkit.html
```

#### 4. OpenMPが見つからない
```bash
# エラー: Could not find OpenMP
sudo apt install libomp-dev  # Ubuntu/Debian
# または
brew install libomp  # macOS
```

#### 5. pybind11が見つからない
```bash
# エラー: Could not find pybind11
pip install pybind11
```

### デバッグビルド

```bash
# デバッグ情報付きでビルド
mkdir build_debug && cd build_debug
cmake .. -DCMAKE_BUILD_TYPE=Debug
make VERBOSE=1

# 詳細なログ出力
cmake .. -DCMAKE_BUILD_TYPE=Debug -DCMAKE_VERBOSE_MAKEFILE=ON
```

### パフォーマンス最適化

```bash
# 最適化レベルを上げる
cmake .. -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-O3 -march=native"

# 特定のアーキテクチャ向け最適化
cmake .. -DCMAKE_CXX_FLAGS="-O3 -march=skylake"  # Intel Skylake
cmake .. -DCMAKE_CXX_FLAGS="-O3 -march=znver2"   # AMD Zen2
```

## インストール

### 開発用インストール

```bash
# 現在のディレクトリにインストール
pip install -e .

# または、パスを追加して使用
export PYTHONPATH=$PYTHONPATH:$(pwd)/python
```

### システムインストール

```bash
# システム全体にインストール
pip install .

# または
python setup.py install
```

## テスト

### ビルド後のテスト

```bash
# 全テストの実行
python -m pytest tests/

# 特定のテストの実行
python -m pytest tests/test_rk4_sparse.py

# ベンチマークの実行
python examples/python/benchmark_ho.py
```

### 実装の可用性確認

```python
from rk4_sparse import (
    rk4_sparse_eigen,
    rk4_sparse_eigen_cached,
    rk4_sparse_eigen_direct_csr,
    rk4_sparse_suitesparse,
    rk4_sparse_suitesparse_mkl
)

print(f"Eigen: {rk4_sparse_eigen is not None}")
print(f"Eigen_Cached: {rk4_sparse_eigen_cached is not None}")
print(f"Eigen_Direct_CSR: {rk4_sparse_eigen_direct_csr is not None}")
print(f"SuiteSparse: {rk4_sparse_suitesparse is not None}")
print(f"SuiteSparse-MKL: {rk4_sparse_suitesparse_mkl is not None}")
```

## パッケージ配布

### Wheelビルド

```bash
# 全プラットフォーム向けwheelのビルド
python setup.py bdist_wheel

# 特定のPythonバージョン向け
python setup.py bdist_wheel --python-tag py310
```

### ソース配布

```bash
# ソース配布パッケージの作成
python setup.py sdist
```

## 参考資料

- [CMake公式ドキュメント](https://cmake.org/documentation/)
- [pybind11公式ドキュメント](https://pybind11.readthedocs.io/)
- [Eigen3公式ドキュメント](https://eigen.tuxfamily.org/)
- [SuiteSparse公式ドキュメント](https://people.engr.tamu.edu/davis/suitesparse.html)
- [Intel MKL公式ドキュメント](https://www.intel.com/content/www/us/en/developer/tools/oneapi/onemkl.html) 