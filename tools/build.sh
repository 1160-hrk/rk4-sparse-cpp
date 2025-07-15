#!/bin/bash

# エラー発生時にスクリプトを停止
set -e

# デフォルト値の設定
BUILD_TYPE="Release"
CLEAN_BUILD=0
INSTALL_PREFIX=""
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

# Pythonの実行可能ファイルパスを取得
PYTHON_EXECUTABLE=$(which python3)
if [ -z "$PYTHON_EXECUTABLE" ]; then
    echo "エラー: python3が見つかりません"
    exit 1
fi

# Python開発パッケージの確認
if ! python3 -c "import sysconfig; print(sysconfig.get_config_var('CONFINCLUDEPY'))" > /dev/null 2>&1; then
    echo "警告: Python開発パッケージが見つかりません。インストールを推奨します："
    echo "  sudo apt-get install python${PYTHON_VERSION}-dev"
    echo "続行しますが、ビルドが失敗する可能性があります..."
fi

# コマンドライン引数の解析
while [[ $# -gt 0 ]]; do
    case $1 in
        --debug)
            BUILD_TYPE="Debug"
            shift
            ;;
        --clean)
            CLEAN_BUILD=1
            shift
            ;;
        --prefix=*)
            INSTALL_PREFIX="${1#*=}"
            shift
            ;;
        --help)
            echo "使用方法: $0 [オプション]"
            echo "オプション:"
            echo "  --debug     デバッグビルドを実行"
            echo "  --clean     クリーンビルドを実行"
            echo "  --prefix=DIR インストール先のプレフィックスを指定"
            echo "  --help      このヘルプを表示"
            exit 0
            ;;
        *)
            echo "不明なオプション: $1"
            exit 1
            ;;
    esac
done

# クリーンビルドの場合
if [ $CLEAN_BUILD -eq 1 ]; then
    echo "クリーンビルドを実行中..."
    rm -rf build
fi

# ビルドディレクトリの作成と移動
echo "ビルドディレクトリを準備中..."
mkdir -p build
cd build

# CMakeの実行
echo "CMakeを実行中... (Build Type: $BUILD_TYPE)"

# Pythonの設定ファイルを探す
PYTHON_CONFIG_DIRS="/usr/lib/python${PYTHON_VERSION}/dist-packages /usr/local/lib/python${PYTHON_VERSION}/dist-packages /usr/lib/python${PYTHON_VERSION}/site-packages /usr/local/lib/python${PYTHON_VERSION}/site-packages"
PYTHON_CMAKE_DIR=""

for dir in $PYTHON_CONFIG_DIRS; do
    if [ -d "$dir" ]; then
        PYTHON_CMAKE_DIR="$dir"
        break
    fi
done

if [ -z "$PYTHON_CMAKE_DIR" ]; then
    echo "警告: PythonのCMake設定ディレクトリが見つかりません"
    PYTHON_CMAKE_DIR="/usr/lib/python${PYTHON_VERSION}/dist-packages"
fi

CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE -DPython_EXECUTABLE=$PYTHON_EXECUTABLE"

# pybind11の設定を追加
if [ -n "$PYBIND11_DIR" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -Dpybind11_DIR=$PYBIND11_DIR"
fi
if [ -n "$INSTALL_PREFIX" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX"
fi

# Python関連の環境変数を設定
export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:/usr/lib/python${PYTHON_VERSION}/dist-packages"
export Python_FIND_STRATEGY=LOCATION

# Pythonの設定情報を取得して環境変数に設定
PYTHON_INCLUDE_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_path('include'))" 2>/dev/null || echo "/usr/include/python${PYTHON_VERSION}")
PYTHON_LIBRARY_DIR=$(python3 -c "import sysconfig; print(sysconfig.get_config_var('LIBDIR'))" 2>/dev/null || echo "/usr/lib/python${PYTHON_VERSION}")

export Python_INCLUDE_DIRS="$PYTHON_INCLUDE_DIR"
export Python_LIBRARY_DIRS="$PYTHON_LIBRARY_DIR"

# pybind11の設定を直接指定
PYBIND11_DIR=$(python3 -c "import pybind11; import os; print(os.path.dirname(pybind11.__file__))" 2>/dev/null || echo "")
if [ -n "$PYBIND11_DIR" ]; then
    export CMAKE_PREFIX_PATH="${CMAKE_PREFIX_PATH}:$PYBIND11_DIR"
    echo "pybind11ディレクトリ: $PYBIND11_DIR"
fi

# PythonInterpの問題を回避するための追加設定
export PythonInterp_DIR="$PYBIND11_DIR"
export PythonInterp_ROOT="/usr"
export PythonInterp_FIND_STRATEGY=LOCATION

cmake .. $CMAKE_ARGS || {
    echo "CMakeの実行に失敗しました。エラーログ:"
    if [ -f CMakeFiles/CMakeError.log ]; then
        cat CMakeFiles/CMakeError.log
    else
        echo "CMakeError.logが見つかりません"
    fi
    exit 1
}

# ビルドの実行
echo "ビルドを実行中..."
cmake --build . -j$(nproc) || {
    echo "ビルドに失敗しました"
    exit 1
}

# ライブラリファイルの検索とコピー
echo "ライブラリファイルをPythonパッケージにコピー中..."
MODULE_PATH="lib/python/_rk4_sparse_cpp*.so"
SO_FILES=$(find . -name "_rk4_sparse_cpp*.so" -type f)

if [ -z "$SO_FILES" ]; then
    echo "エラー: .soファイルが見つかりません"
    echo "ビルドディレクトリの内容:"
    find . -type f -name "*.so"
    exit 1
fi

# Pythonパッケージディレクトリが存在することを確認
mkdir -p ../python/rk4_sparse

# 見つかった.soファイルをすべてコピー
for SO_FILE in $SO_FILES; do
    echo "コピー中: $SO_FILE"
    cp "$SO_FILE" ../python/rk4_sparse/
done

echo "ビルド成功！"

# インストール（プレフィックスが指定されている場合）
if [ -n "$INSTALL_PREFIX" ]; then
    echo "インストールを実行中..."
    cmake --install .
    echo "インストール完了: $INSTALL_PREFIX"
fi

# 最終確認
echo "Pythonパッケージディレクトリの内容:"
ls -la ../python/rk4_sparse/

# 新しい構造のメッセージ
echo ""
echo "新しいディレクトリ構造でのビルドが完了しました。"
echo "アルゴリズムの追加は以下の場所に行ってください："
echo "  - C++実装: src/core/"
echo "  - ヘッダー: include/excitation_rk4_sparse/"
echo "  - Pythonバインディング: src/bindings/python_bindings.cpp" 