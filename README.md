# Excitation RK4 Sparse

量子力学的な励起ダイナミクスを計算するための疎行列ベースのRK4ソルバー。

## 機能
- CSR形式の疎行列サポート
- OpenMPによる並列化
- Python/C++のハイブリッド実装
- ベンチマーク機能

## バージョン情報
- 現在のバージョン: v0.1.0
- ステータス: 安定版（初期リリース）
- 最終更新: 2024-07-07

## 必要条件
- Python 3.10以上
- C++17対応コンパイラ
- CMake 3.10以上
- pybind11
- Eigen3
- OpenMP（オプション）

## インストール
```bash
git clone https://github.com/yourusername/excitation-rk4-sparse.git
cd excitation-rk4-sparse
mkdir build && cd build
cmake ..
make
```

## 使用例
```python
from python.rk4_sparse_py import rk4_cpu_sparse as rk4_cpu_sparse_py
import _excitation_rk4_sparse as rk4_cpu_sparse_cpp

# Python実装
result_py = rk4_cpu_sparse_py(H0, mux, muy, Ex, Ey, psi0, dt, True, stride, False)

# C++実装
result_cpp = rk4_cpu_sparse_cpp.rk4_cpu_sparse(
    H0.data, H0.indices, H0.indptr,
    H0.shape[0], H0.shape[1],
    mux.data, mux.indices, mux.indptr,
    muy.data, muy.indices, muy.indptr,
    Ex, Ey, psi0, dt, True, stride, False
)
```

## ベンチマーク
`examples/benchmark_comparison.py`を実行することで、Python実装とC++実装のパフォーマンス比較が可能です：
```bash
python examples/benchmark_comparison.py
```

## 既知の課題
- 現時点ではPython実装の方が若干高速
- 小規模な行列での非効率性
- データ変換のオーバーヘッド

## ライセンス
MITライセンス

## 作者
- Hiroki Tsusaka
- IIS, UTokyo
- tsusaka4research "at" gmail.com

```bash
pip install -e .
