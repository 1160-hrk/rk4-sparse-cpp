# Excitation RK4 Sparse ドキュメント

excitation-rk4-sparseプロジェクトの包括的なドキュメントです。

## 📚 ドキュメント構成

### 🚀 クイックスタート
- [README](../README.md) - プロジェクト概要とインストール方法

### 👨‍💻 開発ガイド
- [プロジェクト構造変更とマイグレーション](development/project_restructure_migration.md) - v0.2.0での大規模リファクタリングの記録
- [ビルドシステム設定](development/build_configuration.md) - CMakeとビルド設定の詳細
- [テスト戦略](development/testing_strategy.md) - テストの方針と実装
- [開発環境セットアップ](development/development_setup.md) - VS Code dev containerの設定

### 🔧 トラブルシューティング
- [性能回帰問題の分析と解決](troubleshooting/performance_regression_analysis.md) - C++実装の性能問題の詳細分析と解決策

### 📊 ベンチマーク結果
- [性能測定結果](benchmarks/performance_results.md) - 最新の性能ベンチマーク
- [プロファイリング結果](benchmarks/profiling_results.md) - 詳細なプロファイリングデータ

### 📖 API リファレンス
- [Python API](api/python_api.md) - Pythonインターフェースの詳細
- [C++ API](api/cpp_api.md) - C++実装のAPI仕様

## 🔍 主要なトピック

### プロジェクト構造の変更（v0.2.0）
プロジェクトは大幅な構造変更を経て、現代的なC++/Pythonハイブリッドプロジェクトの標準に準拠しました：

```
旧構造 → 新構造
├── *.cpp (ルート)     → src/core/
├── *.py (ルート)      → python/excitation_rk4_sparse/
├── build.sh           → tools/build.sh
└── examples/          → examples/{basic,quantum,benchmarks}/
```

**詳細**: [project_restructure_migration.md](development/project_restructure_migration.md)

### 性能問題の解決
C++実装でPython実装より遅くなる深刻な性能回帰を発見・解決：

- **根本原因**: デバッグ出力のI/Oオーバーヘッド
- **解決結果**: 最大23.65倍の高速化を実現
- **学習事項**: 適切なプロファイリングとデバッグの重要性

**詳細**: [performance_regression_analysis.md](troubleshooting/performance_regression_analysis.md)

### 最新の性能結果

| システムサイズ | Python [ms] | C++ [ms] | 速度比 |
|-------------:|------------:|----------:|-------:|
| 50レベル      | 12.8        | 0.5       | **23.65x** |
| 100レベル     | 14.5        | 0.9       | **15.69x** |
| 200レベル     | 17.3        | 1.8       | **9.81x** |
| 500レベル     | 12.2        | 2.9       | **4.29x** |

## 🛠️ 開発者向け情報

### ビルドとテスト
```bash
# クリーンビルド
./tools/build.sh --clean

# デバッグビルド
./tools/build.sh --debug

# 全例の自動テスト
python tools/test_examples.py
```

### 開発環境
- **推奨**: VS Code + dev container
- **必要な拡張機能**: C++、Python、CMake Tools
- **設定ファイル**: `.devcontainer/devcontainer.json`

### パフォーマンス監視
```bash
# 性能回帰テスト
python examples/benchmarks/benchmark_comparison.py

# 詳細プロファイリング
python examples/benchmarks/profile_comparison.py
```

## 📝 ドキュメント作成について

### 原則
1. **実例重視**: 具体的なコード例と測定結果を含める
2. **問題志向**: 実際に発生した問題と解決策を記録
3. **継続更新**: 新しい発見や改善があれば随時更新
4. **相互リンク**: 関連ドキュメント間の適切なリンク

### 更新頻度
- **性能ベンチマーク**: 主要な変更時
- **API文書**: コード変更時
- **トラブルシューティング**: 問題発生・解決時
- **開発ガイド**: プロセス変更時

## 🤝 貢献方法

ドキュメントの改善にご協力ください：

1. **誤りの報告**: 不正確な情報や古い情報の指摘
2. **新しいトピック**: 不足している情報の追加提案
3. **実例の追加**: 具体的な使用例やベンチマーク結果
4. **翻訳**: 英語版ドキュメントの作成

## 📞 サポート

- **バグレポート**: GitHubのIssue
- **機能リクエスト**: GitHubのDiscussion
- **一般的な質問**: メール（tsusaka4research@gmail.com）

---

**最終更新**: 2024-01-08  
**バージョン**: v0.2.0 