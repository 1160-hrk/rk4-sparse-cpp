#!/bin/bash

# エラー発生時にスクリプトを停止
set -e

# ビルドディレクトリの作成と移動
echo "ビルドディレクトリを準備中..."
mkdir -p build
cd build

# CMakeの実行
echo "CMakeを実行中..."
cmake .. -DCMAKE_BUILD_TYPE=Release

# ビルドの実行
echo "ビルドを実行中..."
cmake --build . -j$(nproc)

# ビルド成功時のメッセージ
echo "ビルド成功！"

# ライブラリファイルの検索とコピー
echo "ライブラリファイルをPythonディレクトリにコピー中..."
if [ -f "_excitation_rk4_sparse.cpython-310-aarch64-linux-gnu.so" ]; then
    cp _excitation_rk4_sparse.cpython-310-aarch64-linux-gnu.so ../python/
else
    echo "警告: .soファイルが見つかりません"
    ls -la
    exit 1
fi

echo "完了！" 