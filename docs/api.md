# RK4 Sparse API ドキュメント

このドキュメントでは、rk4-sparse-cppライブラリのAPIについて詳しく説明します。

## 概要

rk4-sparse-cppは量子力学的な励起ダイナミクスを計算するための疎行列ベースのRK4ソルバーです。複数の実装バリエーションを提供し、用途に応じて最適な実装を選択できます。

## インポート

```python
from rk4_sparse import (
    # 基本実装
    rk4_sparse_py,           # Python実装
    rk4_numba_py,            # Numba JIT実装
    
    # C++実装
    rk4_sparse_eigen,        # 標準Eigen実装
    rk4_sparse_eigen_cached, # キャッシュ化Eigen実装
    rk4_sparse_eigen_direct_csr, # 直接CSR実装
    
    # SuiteSparse実装
    rk4_sparse_suitesparse,  # SuiteSparse実装
    rk4_sparse_suitesparse_mkl, # SuiteSparse-MKL実装
    
    # ベンチマーク機能
    benchmark_implementations,
    
    # ユーティリティ
    create_test_matrices,
    create_test_pulse,
    
    # 可用性フラグ
    OPENBLAS_SUITESPARSE_AVAILABLE,
    SUITESPARSE_MKL_AVAILABLE
)
```

## 主要関数

### 1. rk4_sparse_py

**Python実装** - 開発・デバッグ用の基本実装

```python
rk4_sparse_py(
    H0: Union[csr_matrix, np.ndarray],
    mux: Union[csr_matrix, np.ndarray],
    muy: Union[csr_matrix, np.ndarray],
    Ex: np.ndarray,
    Ey: np.ndarray,
    psi0: np.ndarray,
    dt: float,
    return_traj: bool,
    stride: int,
    renorm: bool = False
) -> np.ndarray
```

**パラメータ:**
- `H0`: ハミルトニアン行列（CSR形式または密行列）
- `mux`: x方向双極子演算子（CSR形式または密行列）
- `muy`: y方向双極子演算子（CSR形式または密行列）
- `Ex`: x方向電場配列（1次元numpy配列）
- `Ey`: y方向電場配列（1次元numpy配列）
- `psi0`: 初期波動関数（1次元複素numpy配列）
- `dt`: 時間ステップ
- `return_traj`: 軌道を返すかどうか
- `stride`: 出力間隔
- `renorm`: 各ステップで正規化するかどうか

**戻り値:**
- `return_traj=True`: `(n_out, dim)` の複素行列（時間軌道）
- `return_traj=False`: `(1, dim)` の複素行列（最終状態）

### 2. rk4_numba_py

**Numba JIT実装** - 小次元での高速化

```python
rk4_numba_py(
    H0: np.ndarray,
    mux: np.ndarray,
    muy: np.ndarray,
    Ex: np.ndarray,
    Ey: np.ndarray,
    psi0: np.ndarray,
    dt: float,
    return_traj: bool,
    stride: int,
    renorm: bool = False
) -> np.ndarray
```

**注意:** この実装は密行列（numpy配列）のみを受け付けます。

### 3. rk4_sparse_eigen

**標準Eigen実装** - 汎用的なC++実装

```python
rk4_sparse_eigen(
    H0: csr_matrix,
    mux: csr_matrix,
    muy: csr_matrix,
    Ex: np.ndarray,
    Ey: np.ndarray,
    psi0: np.ndarray,
    dt: float,
    return_traj: bool,
    stride: int,
    renorm: bool = False
) -> np.ndarray
```

### 4. rk4_sparse_eigen_cached

**キャッシュ化Eigen実装** - 大次元での最適化

```python
rk4_sparse_eigen_cached(
    H0: csr_matrix,
    mux: csr_matrix,
    muy: csr_matrix,
    Ex: np.ndarray,
    Ey: np.ndarray,
    psi0: np.ndarray,
    dt: float,
    return_traj: bool,
    stride: int,
    renorm: bool = False
) -> np.ndarray
```

**特徴:** 疎行列パターンをキャッシュし、大次元で顕著な性能向上を実現。

### 5. rk4_sparse_eigen_direct_csr

**直接CSR実装** - 小次元での最高性能

```python
rk4_sparse_eigen_direct_csr(
    H0_data: np.ndarray,
    H0_indices: np.ndarray,
    H0_indptr: np.ndarray,
    mux_data: np.ndarray,
    mux_indices: np.ndarray,
    mux_indptr: np.ndarray,
    muy_data: np.ndarray,
    muy_indices: np.ndarray,
    muy_indptr: np.ndarray,
    Ex: np.ndarray,
    Ey: np.ndarray,
    psi0: np.ndarray,
    dt: float,
    return_traj: bool,
    stride: int,
    renorm: bool = False
) -> np.ndarray
```

**特徴:** CSR行列の個別コンポーネントを直接受け取り、小次元で最高性能を実現。

### 6. rk4_sparse_suitesparse

**SuiteSparse実装** - メモリ効率重視

```python
rk4_sparse_suitesparse(
    H0: csr_matrix,
    mux: csr_matrix,
    muy: csr_matrix,
    Ex: np.ndarray,
    Ey: np.ndarray,
    psi0: np.ndarray,
    dt: float,
    return_traj: bool,
    stride: int,
    renorm: bool = False,
    level: int = 1
) -> np.ndarray
```

**パラメータ:**
- `level`: 最適化レベル（0: BASIC, 1: STANDARD, 2: ENHANCED）

### 7. rk4_sparse_suitesparse_mkl

**SuiteSparse-MKL実装** - Intel MKLによる最速実装

```python
rk4_sparse_suitesparse_mkl(
    H0: csr_matrix,
    mux: csr_matrix,
    muy: csr_matrix,
    Ex: np.ndarray,
    Ey: np.ndarray,
    psi0: np.ndarray,
    dt: float,
    return_traj: bool,
    stride: int,
    renorm: bool = False
) -> np.ndarray
```

**注意:** Intel MKLがインストールされている場合のみ利用可能。

## ベンチマーク機能

### benchmark_implementations

複数の実装の性能を比較する関数

```python
benchmark_implementations(
    H0: csr_matrix,
    mux: csr_matrix,
    muy: csr_matrix,
    Ex: np.ndarray,
    Ey: np.ndarray,
    psi0: np.ndarray,
    dt: float,
    num_steps: int,
    return_traj: bool,
    stride: int,
    renorm: bool = False,
    num_runs: int = 5
) -> List[BenchmarkResult]
```

**戻り値:** `BenchmarkResult`オブジェクトのリスト

```python
class BenchmarkResult:
    implementation: str      # 実装名
    total_time: float       # 総実行時間
    matrix_update_time: float  # 行列更新時間
    rk4_step_time: float    # RK4ステップ時間
    matrix_updates: int     # 行列更新回数
    rk4_steps: int          # RK4ステップ数
    speedup_vs_eigen: float # Eigen実装に対する高速化倍率
```

## ユーティリティ関数

### create_test_matrices

テスト用の行列を生成

```python
create_test_matrices(n: int) -> Tuple[csr_matrix, csr_matrix, csr_matrix]
```

**パラメータ:**
- `n`: 行列のサイズ

**戻り値:** `(H0, mux, muy)` - テスト用の疎行列タプル

### create_test_pulse

テスト用のパルス波形を生成

```python
create_test_pulse(steps: int) -> Tuple[np.ndarray, np.ndarray]
```

**パラメータ:**
- `steps`: 時間ステップ数

**戻り値:** `(Ex, Ey)` - ガウシアンパルス波形の配列

## データ型

### 入力データ型

| パラメータ | 型 | 説明 |
|-----------|----|------|
| `H0`, `mux`, `muy` | `scipy.sparse.csr_matrix` | 疎行列（複素数） |
| `Ex`, `Ey` | `numpy.ndarray` | 電場配列（実数、1次元） |
| `psi0` | `numpy.ndarray` | 初期波動関数（複素数、1次元） |
| `dt` | `float` | 時間ステップ |
| `return_traj` | `bool` | 軌道出力フラグ |
| `stride` | `int` | 出力間隔 |
| `renorm` | `bool` | 正規化フラグ |

### 出力データ型

| 戻り値 | 型 | 説明 |
|--------|----|------|
| `return_traj=True` | `numpy.ndarray` | `(n_out, dim)` 複素行列 |
| `return_traj=False` | `numpy.ndarray` | `(1, dim)` 複素行列 |

## 使用例

### 基本的な使用例

```python
import numpy as np
from scipy.sparse import csr_matrix
from rk4_sparse import rk4_sparse_eigen, create_test_matrices, create_test_pulse

# テストシステムの生成
H0, mux, muy = create_test_matrices(4)
Ex, Ey = create_test_pulse(1000)
psi0 = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.complex128)

# 時間発展の計算
result = rk4_sparse_eigen(
    H0, mux, muy, Ex, Ey, psi0,
    dt=0.01, return_traj=True, stride=10, renorm=False
)

print(f"結果の形状: {result.shape}")
```

### 実装の可用性確認

```python
from rk4_sparse import (
    rk4_sparse_eigen, rk4_sparse_eigen_cached, rk4_sparse_suitesparse
)

# 利用可能な実装の確認
implementations = {
    'Eigen': rk4_sparse_eigen,
    'Eigen_Cached': rk4_sparse_eigen_cached,
    'SuiteSparse': rk4_sparse_suitesparse
}

for name, impl in implementations.items():
    if impl is not None:
        print(f"{name}: 利用可能")
    else:
        print(f"{name}: 利用不可")
```

### ベンチマーク実行

```python
from rk4_sparse import benchmark_implementations

# ベンチマークの実行
results = benchmark_implementations(
    H0, mux, muy, Ex, Ey, psi0,
    dt=0.01, num_steps=1000, return_traj=True, stride=10, renorm=False,
    num_runs=5
)

# 結果の表示
for result in results:
    print(f"{result.implementation}: {result.total_time:.6f}秒 "
          f"(Eigen比: {result.speedup_vs_eigen:.3f}x)")
```

### 条件分岐による安全な使用

```python
from rk4_sparse import rk4_sparse_eigen_cached, rk4_sparse_suitesparse

# 最適な実装の選択
if rk4_sparse_eigen_cached is not None:
    # キャッシュ化実装を使用
    result = rk4_sparse_eigen_cached(H0, mux, muy, Ex, Ey, psi0, dt, True, 1, False)
elif rk4_sparse_suitesparse is not None:
    # SuiteSparse実装を使用
    result = rk4_sparse_suitesparse(H0, mux, muy, Ex, Ey, psi0, dt, True, 1, False)
else:
    # フォールバック
    from rk4_sparse import rk4_sparse_eigen
    result = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, dt, True, 1, False)
```

## エラーハンドリング

### よくあるエラー

1. **ImportError**: 実装が利用できない場合
   ```python
   if rk4_sparse_eigen_cached is None:
       print("Eigen_Cached実装が利用できません")
   ```

2. **ValueError**: 入力データの形状が不正な場合
   ```python
   # psi0は1次元配列である必要があります
   if psi0.ndim != 1:
       raise ValueError("psi0 must be a 1D array")
   ```

3. **RuntimeError**: CSR行列の形式が不正な場合
   ```python
   # H0, mux, muyはCSR形式である必要があります
   if not hasattr(H0, 'data') or not hasattr(H0, 'indices') or not hasattr(H0, 'indptr'):
       raise RuntimeError("H0 must be a scipy.sparse.csr_matrix")
   ```

## パフォーマンス最適化

### 実装選択ガイド

| 用途 | 推奨実装 | 特徴 |
|------|----------|------|
| 開発・デバッグ | `rk4_sparse_py` | 理解しやすく、デバッグしやすい |
| 小次元（<128） | `rk4_sparse_eigen_direct_csr` | 最高性能 |
| 中次元（128-1024） | `rk4_sparse_eigen_cached` | バランス良好 |
| 大次元（>1024） | `rk4_sparse_eigen_cached` | キャッシュ効果で高速 |
| メモリ制約環境 | `rk4_sparse_suitesparse` | メモリ効率最高 |
| 最高性能 | `rk4_sparse_suitesparse_mkl` | Intel MKL利用で最速 |

### パフォーマンスチップ

1. **行列の事前準備**: 可能な限り疎行列パターンを再利用
2. **メモリアライメント**: 大次元ではキャッシュ化実装を使用
3. **並列化**: OpenMPが有効な環境では自動的に並列化
4. **正規化**: 必要に応じて`renorm=True`を使用

## 制限事項

1. **Numba実装**: 密行列のみ対応、大次元で性能劣化
2. **SuiteSparse-MKL**: Intel MKLのインストールが必要
3. **メモリ使用量**: 大次元では大量のメモリを消費
4. **精度**: 浮動小数点演算による数値誤差が蓄積する可能性

## 参考資料

- [プロジェクトREADME](../README.md)
- [ビルド設定ガイド](development/build_configuration.md)
- [ベンチマーク結果](development/benchmark_comparison_analysis_20250718.md) 