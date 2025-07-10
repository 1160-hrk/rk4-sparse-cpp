# PyPI登録手順

## 概要

excitation-rk4-sparseパッケージをPyPI（Python Package Index）に登録して、`pip install excitation-rk4-sparse`でインストール可能にする手順です。

## 📋 事前準備

### 1. アカウント作成

#### PyPI本番環境
1. [PyPI公式サイト](https://pypi.org/account/register/)でアカウント登録
2. メール認証を完了
3. **必須**: 2FA（二段階認証）を設定
   - TOTP（Google Authenticator、Authy等）
   - またはWebAuthn（ハードウェアキー）

#### TestPyPI（推奨）
1. [TestPyPI](https://test.pypi.org/account/register/)でテスト用アカウント登録
2. 本番前のテスト用途

### 2. APIトークンの設定

#### PyPIでAPIトークン生成
1. PyPIにログイン → Account settings
2. "API tokens" → "Add API token"
3. Scope: "Entire account" または特定プロジェクト
4. トークンを安全に保存

#### ローカル設定（推奨）
```bash
# ~/.pypirc ファイルを作成
cat > ~/.pypirc << EOF
[distutils]
index-servers = 
    pypi
    testpypi

[pypi]
repository: https://upload.pypi.org/legacy/
username: __token__
password: pypi-YOUR_TOKEN_HERE

[testpypi]
repository: https://test.pypi.org/legacy/
username: __token__
password: pypi-YOUR_TESTPYPI_TOKEN_HERE
EOF

# ファイル権限を制限
chmod 600 ~/.pypirc
```

## 🔨 パッケージビルド

### 1. 必要ツールのインストール
```bash
pip install build twine
sudo apt install python3.10-venv  # Ubuntu/Debianの場合
```

### 2. プロジェクト設定の確認

#### pyproject.toml
```toml
[project]
name = "excitation-rk4-sparse"
version = "0.2.0"
description = "High-performance sparse matrix RK4 solver for quantum excitation dynamics"
readme = "README.md"
license = {file = "LICENSE"}
authors = [
    {name = "Hiroki Tsusaka", email = "tsusaka4research@gmail.com"},
]
classifiers = [
    "Development Status :: 4 - Beta",
    "Intended Audience :: Science/Research",
    "License :: OSI Approved :: MIT License",
    "Programming Language :: Python :: 3",
    "Programming Language :: Python :: 3.8",
    "Programming Language :: Python :: 3.9",
    "Programming Language :: Python :: 3.10",
    "Programming Language :: Python :: 3.11",
    "Topic :: Scientific/Engineering :: Physics",
]
keywords = ["quantum", "dynamics", "rk4", "sparse", "physics", "simulation"]
requires-python = ">=3.8"
dependencies = [
    "numpy>=1.20.0",
    "scipy>=1.7.0",
    "matplotlib>=3.3.0",
]

[project.urls]
Homepage = "https://github.com/1160-hrk/excitation-rk4-sparse"
Repository = "https://github.com/1160-hrk/excitation-rk4-sparse"
Issues = "https://github.com/1160-hrk/excitation-rk4-sparse/issues"
Documentation = "https://github.com/1160-hrk/excitation-rk4-sparse/tree/main/docs"
```

### 3. 必要ファイルの確認
- ✅ `README.md` - プロジェクト説明
- ✅ `LICENSE` - ライセンスファイル（MIT）
- ✅ `pyproject.toml` - プロジェクト設定
- ✅ `python/excitation_rk4_sparse/` - Pythonパッケージ

### 4. パッケージビルド
```bash
# ビルドディレクトリのクリーンアップ
rm -rf dist/ build/ *.egg-info

# Pure Python版の場合（推奨：初回）
python setup_simple.py sdist bdist_wheel

# または pyproject.tomlベース（C++拡張含む）
python -m build
```

### 5. パッケージ検証
```bash
# PyPI要件の検証
python -m twine check dist/*
```

期待される出力:
```
Checking dist/excitation_rk4_sparse-0.2.0-py3-none-any.whl: PASSED
Checking dist/excitation-rk4-sparse-0.2.0.tar.gz: PASSED
```

## 🚀 アップロード手順

### 1. TestPyPIでテスト（推奨）
```bash
# テスト環境にアップロード
python -m twine upload --repository testpypi dist/*
```

**認証**:
- Username: `__token__`
- Password: TestPyPI APIトークン

### 2. テストパッケージの確認
```bash
# TestPyPIからインストールしてテスト
pip install -i https://test.pypi.org/simple/ excitation-rk4-sparse

# 基本テスト
python -c "from excitation_rk4_sparse import rk4_cpu_sparse_py; print('Success!')"
```

### 3. 本番PyPIへアップロード
```bash
# 本番環境にアップロード
python -m twine upload dist/*
```

### 4. 本番パッケージの確認
```bash
# 通常のpipでインストール
pip install excitation-rk4-sparse

# 機能テスト
python -c "from excitation_rk4_sparse import rk4_cpu_sparse_py, rk4_cpu_sparse_cpp; print('All systems go!')"
```

## 📝 トラブルシューティング

### 1. パッケージ名の重複
```
ERROR: The name 'excitation-rk4-sparse' is already in use.
```

**解決策**:
- パッケージ名を変更: `excitation-rk4-sparse-hrk`
- または既存パッケージが自分のものか確認

### 2. 認証エラー
```
ERROR: Invalid credentials
```

**解決策**:
- APIトークンを再確認
- `~/.pypirc`の設定を確認
- 2FA設定を確認

### 3. ファイルサイズエラー
```
ERROR: File too large
```

**解決策**:
- 不要ファイルを除外（`.gitignore`を参照）
- `MANIFEST.in`でファイル選択を制御

### 4. C++拡張のビルドエラー
```
ERROR: Microsoft Visual C++ 14.0 is required
```

**解決策**:
- まずPure Python版で登録
- C++拡張は別途wheel配布
- GitHub Actionsでクロスプラットフォームビルド

## 🔄 バージョン管理

### 1. セマンティックバージョニング
- `0.1.0` - 初期リリース
- `0.2.0` - 機能追加、API変更
- `0.2.1` - バグフィックス
- `1.0.0` - 安定版リリース

### 2. 新バージョンのリリース
```bash
# 1. バージョン番号更新
# pyproject.toml: version = "0.3.0"

# 2. リビルド
rm -rf dist/
python setup_simple.py sdist bdist_wheel

# 3. アップロード
python -m twine upload dist/*
```

## 📊 完成後の確認

### 1. PyPIページの確認
- パッケージ情報の表示
- インストール手順
- プロジェクトリンク

### 2. インストールテスト
```bash
# 新しい環境でテスト
python -m venv test_env
source test_env/bin/activate
pip install excitation-rk4-sparse
python -c "import excitation_rk4_sparse; print('Success!')"
```

### 3. 使用統計の確認
- PyPIダッシュボードでダウンロード数
- GitHub Actionsでの依存関係更新

## 🎯 今後の改善

### 1. CI/CDパイプライン
- GitHub Actionsで自動ビルド
- 複数Python版での自動テスト
- 自動PyPIアップロード

### 2. C++拡張の配布
- cibuildwheelでマルチプラットフォーム対応
- conda-forgeでの配布

### 3. ドキュメント
- Read the Docsでの公式ドキュメント
- APIリファレンスの自動生成

## 🔗 参考リンク

- [PyPI公式ドキュメント](https://packaging.python.org/)
- [Twineドキュメント](https://twine.readthedocs.io/)
- [setuptools ガイド](https://setuptools.pypa.io/)
- [Python Packaging User Guide](https://packaging.python.org/en/latest/)

---

**作成日**: 2024-01-09  
**最終更新**: 2024-01-09 