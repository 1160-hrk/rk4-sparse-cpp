#!/bin/bash

# エラー発生時にスクリプトを停止
set -e

# デフォルト値の設定
BUILD_TYPE="Release"
CLEAN_BUILD=0
INSTALL_PREFIX=""
PYTHON_VERSION=$(python3 -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')")

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
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=$BUILD_TYPE"
if [ -n "$INSTALL_PREFIX" ]; then
    CMAKE_ARGS="$CMAKE_ARGS -DCMAKE_INSTALL_PREFIX=$INSTALL_PREFIX"
fi

cmake .. $CMAKE_ARGS || {
    echo "CMakeの実行に失敗しました。エラーログ:"
    cat CMakeFiles/CMakeError.log
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