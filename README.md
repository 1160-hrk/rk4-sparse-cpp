# Excitation RK4 Sparse

量子力学的な励起ダイナミクスを計算するための疎行列ベースのRK4ソルバー。

## 機能
- CSR形式の疎行列サポート
- OpenMPによる並列化（動的スケジューリング最適化）
- Python/C++のハイブリッド実装
- 包括的なベンチマーク機能
  - 2準位系と調和振動子のテストケース
  - 詳細なパフォーマンス分析
  - 解析解との比較
- メモリ最適化
  - キャッシュライン境界を考慮したアライメント
  - 疎行列パターンの再利用

## バージョン情報
- 現在のバージョン: v0.1.1
- ステータス: 安定版
- 最終更新: 2024-03-XX

## 必要条件
- Python 3.10以上
- C++17対応コンパイラ
- CMake 3.10以上
- pybind11
- Eigen3
- OpenMP（推奨）

## インストール
```bash
git clone https://github.com/1160-hrk/excitation-rk4-sparse.git
cd excitation-rk4-sparse
pip install -e .
```

## 使用例

### 基本的な使用法
```python
from python import rk4_cpu_sparse_py, rk4_cpu_sparse_cpp

# Python実装
result_py = rk4_cpu_sparse_py(H0, mux, muy, Ex, Ey, psi0, dt, True, stride, False)

# C++実装（高速）
result_cpp = rk4_cpu_sparse_cpp(H0, mux, muy, Ex, Ey, psi0, dt, True, stride, False)
```

### 例題
1. 2準位系の励起ダイナミクス
```bash
python examples/two_level_excitation.py
```

2. 調和振動子のダイナミクス
```bash
python examples/harmonic_oscillator.py
```

## ベンチマーク
以下のスクリプトで様々なベンチマークを実行できます：

1. 実装間の比較
```bash
python examples/benchmark_comparison.py  # 基本的な比較
python examples/benchmark_ho.py         # 調和振動子系での比較
```

2. 詳細なプロファイリング
```bash
python examples/profile_comparison.py   # CPU使用率、メモリ使用量など
```

## 性能
- 最大209倍の高速化を達成（C++ vs Python、2000ステップ時）
- 効率的なメモリ使用（<1MB）
- 優れたスケーリング特性
  - Python: 31.81µs → 22.39µs/step
  - C++: 3.08µs → 0.11µs/step

## 最適化の特徴
1. **メモリアライメント**
   - キャッシュライン境界（64バイト）に合わせたアライメント
   - 作業バッファの効率的な配置

2. **OpenMP並列化**
   - 行列更新処理の並列化
   - 動的スケジューリング（チャンク64）

3. **疎行列最適化**
   - 非ゼロパターンの事前計算
   - データ構造の再利用
   - 効率的な行列-ベクトル積

## ライセンス
MITライセンス

## 作者
- Hiroki Tsusaka
- IIS, UTokyo
- tsusaka4research "at" gmail.com

```bash
pip install -e .
