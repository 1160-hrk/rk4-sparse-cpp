#!/bin/bash

# OpenBLAS + SuiteSparse版のビルドスクリプト

set -e

echo "=== OpenBLAS + SuiteSparse版のビルドを開始 ==="

# ビルドディレクトリを作成
BUILD_DIR="build-openblas-suitesparse"
mkdir -p $BUILD_DIR
cd $BUILD_DIR

# CMakeで設定
echo "CMakeで設定中..."
cmake .. \
    -DCMAKE_BUILD_TYPE=Release \
    -DUSE_OPENBLAS_SUITESPARSE=ON \
    -DUSE_SUITESPARSE_MKL=OFF \
    -DPython3_EXECUTABLE=$(which python3) \
    -G Ninja

# ビルド
echo "ビルド中..."
cmake --build . --config Release

echo "=== ビルド完了 ==="
echo "ビルド成果物: $BUILD_DIR/"
echo ""
echo "使用方法:"
echo "1. Pythonでインポート:"
echo "   from rk4_sparse import rk4_sparse_suitesparse, benchmark_implementations"
echo ""
echo "2. ベンチマーク実行:"
echo "   results = benchmark_implementations(H0, mux, muy, Ex, Ey, psi0, dt, True, 1, False, 5)" 