# セグメンテーション違反のデバッグ手順

## 問題の概要

BLAS最適化版の実装を適用した後、セグメンテーション違反が発生する問題が報告されています。この問題は、特に`rk4_sparse_blas_optimized`関数の実行時に発生します。

## 症状

### 発生状況
- **BLAS最適化版の実装適用後**に発生
- **1024次元でのテスト**でセグメンテーション違反
- **Pythonバインディング経由**での実行時に発生

### エラーメッセージ
```bash
$ python test_blas_optimization.py
✓ rk4_sparse_cppモジュールのインポート成功
BLAS最適化版のRK4実装テスト
==================================================
✓ OpenMP対応版がビルドされています（最大スレッド数: 14）
--- 1024次元でのテスト ---
  1024次元のテスト行列を作成中...
  標準版（Eigen）を実行中...
    実行時間: 0.0157秒
  BLAS最適化版を実行中...
Segmentation fault
```

## 根本原因の分析

### 1. BLAS関数の不適切な使用
```cpp
// 問題のあるコード
auto result = cblas_zdotc(length, 
                        reinterpret_cast<const double*>(&H_data[start]), 1, 
                        reinterpret_cast<const double*>(&x[H_indices[start]]), 1);
```

**問題点**:
- `reinterpret_cast<const double*>`で複素数配列を`double*`に変換（未定義動作）
- スパース行列の`indices`配列は連続していないため、BLAS関数が期待する連続メモリ領域ではない
- `stride=1`の設定が複素数型に対して不適切

### 2. メモリアクセス違反
```cpp
// 問題のあるコード
const double*>(&x[H_indices[start]])
```

**問題点**:
- `H_indices[start]`が行列の次元を超える可能性
- 境界チェックが行われていない

### 3. OpenBLASの条件分岐の問題
```cpp
#ifdef OPENBLAS_SUITESPARSE_AVAILABLE
    // BLAS実装
#else
    // フォールバック実装
#endif
```

**問題点**:
- `OPENBLAS_SUITESPARSE_AVAILABLE`が定義されていても、実際のBLASライブラリが正しくリンクされていない可能性

## デバッグ手順（ボトムアップアプローチ）

### ステップ1: BLAS関数の完全無効化

最も基本的なレベルから始めます。BLAS関数の呼び出しを完全に無効化して、基本的なスパース行列-ベクトル積が動作するか確認します。

```cpp
// デバッグ用：BLAS関数を完全に無効化
void blas_optimized_sparse_matrix_vector_multiply(
    const std::complex<double>* H_data,
    const int* H_indices,
    const int* H_indptr,
    const Eigen::VectorXcd& x,
    Eigen::VectorXcd& y,
    int dim) {
    
    using cplx = std::complex<double>;
    
    // デバッグ用：BLAS関数を完全に無効化して基本的な実装のみ使用
    y.setZero();
    for (int i = 0; i < dim; ++i) {
        int start = H_indptr[i];
        int end = H_indptr[i + 1];
        
        for (int j = start; j < end; ++j) {
            int col_idx = H_indices[j];
            if (col_idx >= 0 && col_idx < dim) {  // 境界チェック
                y[i] += H_data[j] * x[col_idx];
            }
        }
    }
    
    // 虚数単位を掛ける
    y *= cplx(0, -1);
}
```

### ステップ2: 基本的なテスト関数の作成

Pythonバインディングに簡単なテスト関数を追加します：

```cpp
// 簡単なテスト関数を追加
m.def("test_basic_sparse_multiply", [](
    const py::object& H0,
    const Eigen::VectorXcd& x
) {
    // 最適化されたCSR行列の構築
    Eigen::SparseMatrix<cplx> H0_mat = build_sparse_matrix_from_scipy(H0);
    
    // 基本的なスパース行列-ベクトル積
    Eigen::VectorXcd y = H0_mat * x;
    
    return y;
},
py::arg("H0"),
py::arg("x"),
"基本的なスパース行列-ベクトル積のテスト"
);
```

### ステップ3: 段階的なテスト

以下の順序でテストを実行します：

```python
import numpy as np
import scipy.sparse as sp
from rk4_sparse_cpp import test_basic_sparse_multiply, rk4_sparse_blas_optimized

# ステップ1: 基本的なスパース行列-ベクトル積のテスト
print("=== ステップ1: 基本的なスパース行列-ベクトル積のテスト ===")
dim = 1024
H0 = sp.random(dim, dim, density=0.01, format='csr', dtype=np.complex128)
x = np.random.rand(dim) + 1j * np.random.rand(dim)

try:
    y = test_basic_sparse_multiply(H0, x)
    print("✓ 基本的なスパース行列-ベクトル積: 成功")
except Exception as e:
    print(f"✗ 基本的なスパース行列-ベクトル積: 失敗 - {e}")

# ステップ2: BLAS無効化版のRK4テスト
print("\n=== ステップ2: BLAS無効化版のRK4テスト ===")
mux = sp.random(dim, dim, density=0.01, format='csr', dtype=np.complex128)
muy = sp.random(dim, dim, density=0.01, format='csr', dtype=np.complex128)
Ex = np.random.rand(10)
Ey = np.random.rand(10)
psi0 = np.random.rand(dim) + 1j * np.random.rand(dim)

try:
    result = rk4_sparse_blas_optimized(H0, mux, muy, Ex, Ey, psi0, 0.01, False, 1, False)
    print("✓ BLAS無効化版のRK4: 成功")
except Exception as e:
    print(f"✗ BLAS無効化版のRK4: 失敗 - {e}")
```

### ステップ4: BLAS関数の段階的有効化

各ステップが成功したら、BLAS関数を一つずつ有効化してテストします：

1. **境界チェック付きBLAS実装**
2. **連続メモリ領域確保版**
3. **正しいstride設定版**
4. **並列化版**

## 解決策

### 1. 安全なBLAS実装

```cpp
// 安全なBLAS実装
void blas_optimized_sparse_matrix_vector_multiply_safe(
    const std::complex<double>* H_data,
    const int* H_indices,
    const int* H_indptr,
    const Eigen::VectorXcd& x,
    Eigen::VectorXcd& y,
    int dim) {
    
    using cplx = std::complex<double>;
    
    #ifdef OPENBLAS_SUITESPARSE_AVAILABLE
    y.setZero();
    
    for (int i = 0; i < dim; ++i) {
        int start = H_indptr[i];
        int end = H_indptr[i + 1];
        int length = end - start;
        
        if (length > 0) {
            // 安全なBLAS実装：連続メモリ領域を確保
            std::vector<cplx> temp_x(length);
            for (int k = 0; k < length; ++k) {
                int col_idx = H_indices[start + k];
                if (col_idx >= 0 && col_idx < dim) {
                    temp_x[k] = x[col_idx];
                } else {
                    temp_x[k] = cplx(0.0, 0.0);
                }
            }
            
            // BLASのdot productを使用（正しい型変換）
            auto result = cblas_zdotc(length, 
                                    reinterpret_cast<const double*>(&H_data[start]), 2, 
                                    reinterpret_cast<const double*>(temp_x.data()), 2);
            y[i] = cplx(result.real, result.imag);
        }
    }
    #else
    // フォールバック実装
    y.setZero();
    for (int i = 0; i < dim; ++i) {
        int start = H_indptr[i];
        int end = H_indptr[i + 1];
        
        for (int j = start; j < end; ++j) {
            int col_idx = H_indices[j];
            if (col_idx >= 0 && col_idx < dim) {
                y[i] += H_data[j] * x[col_idx];
            }
        }
    }
    #endif
    
    // 虚数単位を掛ける
    y *= cplx(0, -1);
}
```

### 2. CMakeLists.txtの修正

```cmake
# OpenBLAS + SuiteSparseライブラリをリンク
if(USE_OPENBLAS_SUITESPARSE AND (OpenBLAS_FOUND OR OPENBLAS_LIB) AND (SuiteSparse_FOUND OR (SUITESPARSE_CHOLMOD AND SUITESPARSE_UMFPACK)))
    add_definitions(-DOPENBLAS_SUITESPARSE_AVAILABLE)
    message(STATUS "OpenBLAS + SuiteSparse optimization enabled")
    
    # ライブラリリストを構築
    set(LINK_LIBRARIES)
    if(OpenBLAS_LIBRARIES)
        list(APPEND LINK_LIBRARIES ${OpenBLAS_LIBRARIES})
    elseif(OPENBLAS_LIB)
        list(APPEND LINK_LIBRARIES ${OPENBLAS_LIB})
    endif()
    
    # 追加のBLASライブラリを明示的にリンク
    find_library(BLAS_LIB blas)
    if(BLAS_LIB)
        list(APPEND LINK_LIBRARIES ${BLAS_LIB})
        message(STATUS "BLAS library found: ${BLAS_LIB}")
    endif()
    
    find_library(CBLAS_LIB cblas)
    if(CBLAS_LIB)
        list(APPEND LINK_LIBRARIES ${CBLAS_LIB})
        message(STATUS "CBLAS library found: ${CBLAS_LIB}")
    endif()
    
    # 複数のBLAS実装を試行
    if(NOT BLAS_LIB)
        find_library(BLAS_LIB openblas)
    endif()
    if(NOT BLAS_LIB)
        find_library(BLAS_LIB libopenblas)
    endif()
    if(NOT BLAS_LIB)
        find_library(BLAS_LIB libblas)
    endif()
    
    if(BLAS_LIB)
        list(APPEND LINK_LIBRARIES ${BLAS_LIB})
        message(STATUS "BLAS library linked: ${BLAS_LIB}")
    else()
        message(WARNING "BLAS library not found, falling back to Eigen-only implementation")
        set(USE_OPENBLAS_SUITESPARSE OFF)
        add_definitions(-UOPENBLAS_SUITESPARSE_AVAILABLE)
    endif()
    
    target_link_libraries(_rk4_sparse_cpp PRIVATE ${LINK_LIBRARIES})
else()
    message(WARNING "OpenBLAS or SuiteSparse not found. Using Eigen-only implementation.")
    set(USE_OPENBLAS_SUITESPARSE OFF)
    add_definitions(-UOPENBLAS_SUITESPARSE_AVAILABLE)
endif()
```

## 予防策

### 1. 境界チェックの徹底
- すべての配列アクセスに境界チェックを追加
- `assert`文の活用

### 2. BLAS関数の適切な使用
- 複素数型の正しい変換
- 連続メモリ領域の確保
- 適切なstride設定

### 3. 段階的なテスト
- 小さな問題から始める
- 各段階での動作確認
- 自動化されたテストの活用

### 4. デバッグ情報の活用
```cpp
#ifdef DEBUG_SEGFAULT
    std::cout << "Debug: dim=" << dim << ", start=" << start << ", end=" << end << std::endl;
    std::cout << "Debug: col_idx=" << col_idx << std::endl;
#endif
```

## 関連ドキュメント

- [Performance Regression Analysis](performance_regression_analysis.md)
- [Build System Configuration](../development/build_configuration.md)
- [C++ Optimization Strategy](../development/cpp_optimization_strategy_20250717.md)

## 更新履歴

- 2024-01-08: 初版作成 - セグメンテーション違反のデバッグ手順 