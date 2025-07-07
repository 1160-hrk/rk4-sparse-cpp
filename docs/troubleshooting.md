# トラブルシューティングガイド

## セグメンテーション違反の問題と解決策

### 1. Python/C++間のデータ変換の問題

#### 問題点
- `pybind11`のデフォルトの型変換に依存していた
- 入力配列（`psi0`）の形状チェックが不十分
- メモリの所有権と寿命の管理が不適切

#### 解決策
```cpp
// 改善前
m.def("rk4_propagate", &rk4_propagate,
    py::arg("psi0"),
    // ...
);

// 改善後
m.def("rk4_propagate",
    [](const SparseMatrixXcd& H0,
       // ...
       py::array_t<std::complex<double>>& psi0_arr,
       // ...) {
        // 明示的な形状チェック
        py::buffer_info buf = psi0_arr.request();
        if (buf.ndim != 2 || buf.shape[1] != 1) {
            throw std::runtime_error("psi0 must be a column vector (n x 1)");
        }
        // メモリマッピングによる効率的な変換
        Eigen::Map<VectorXcd> psi0(static_cast<std::complex<double>*>(buf.ptr), buf.shape[0]);
        // ...
    },
    py::arg("psi0").noconvert(),  // 暗黙的な型変換を防止
    // ...
);
```

### 2. 疎行列操作の非効率性

#### 問題点
- 疎行列を密行列に変換していた（`toDense()`の使用）
- 行列演算時の不要なメモリ確保と解放
- 一時オブジェクトの過剰な生成

#### 解決策
```cpp
// 改善前
auto build_H = [&](double ex, double ey) {
    Eigen::MatrixXcd H = H0.toDense();  // 密行列への変換
    H += mux.toDense() * ex;  // 密行列での演算
    H += muy.toDense() * ey;
    return H;
};

// 改善後
// メモリの事前確保
Eigen::SparseMatrix<cplx> H(dim, dim);
H.reserve(H0.nonZeros() + mux.nonZeros() + muy.nonZeros());

auto build_H = [&](double ex, double ey) -> const Eigen::SparseMatrix<cplx>& {
    H = H0;
    if (ex != 0.0) H += ex * mux;  // 疎行列のまま演算
    if (ey != 0.0) H += ey * muy;
    return H;
};
```

### 3. メモリ管理とパフォーマンスの最適化

#### 問題点
- 不要なコピーと一時オブジェクトの生成
- 行列演算の非効率な実装
- メモリリソースの過剰な使用

#### 解決策
```cpp
// 改善前
auto H1 = build_H(ex1, ey1);  // コピーが発生
k1 = matvec(H1, psi);

// 改善後
const auto& H1 = build_H(ex1, ey1);  // 参照を使用
k1 = matvec(H1, psi);

// matvec関数の最適化
auto matvec = [](const Eigen::SparseMatrix<cplx>& H,  // 参照で受け取り
                 const Eigen::VectorXcd& v) {
    return cplx(0, -1) * (H * v);
};
```

## ベストプラクティス

1. **型変換とチェック**
   - 入力データの型と形状を明示的にチェック
   - `.noconvert()`を使用して意図しない型変換を防止
   - メモリマッピングを活用して効率的なデータアクセスを実現

2. **疎行列の操作**
   - 可能な限り疎行列形式を維持
   - 必要な非ゼロ要素数を事前に予約
   - ゼロ係数の演算をスキップ

3. **メモリ管理**
   - 不要なコピーを避けるため参照を使用
   - 一時オブジェクトの生成を最小限に抑える
   - メモリの事前確保を活用

4. **エラー処理**
   - 入力データの検証を徹底
   - 適切なエラーメッセージを提供
   - 例外を使用して異常を通知

## 注意点

- Python側での入力データの準備時に、適切な形状（列ベクトル）を確保すること
- 大規模な疎行列演算を行う場合は、メモリ使用量に注意
- デバッグ出力を活用して問題の早期発見に努める 