# CMakeLists.txt コマンド早見表

Python 開発者が **pybind11 + CMake** で C++ 拡張を組むときに最低限押さえておきたいコマンドをまとめました。各ブロックを **コピペ**→コメントを調整して使えます。

---

## 1. CMake 本体のバージョン

```cmake
cmake_minimum_required(VERSION 3.15)
```

* **文法**: CMake 自体の “要求バージョン” を宣言。
* **役割**: `requires-python >= 3.x` と同じで、新しめのコマンドを使うための足切り。
* **忘れると**: 古い CMake が互換モードになり、後続で **Unknown command** やポリシー警告 → ビルド失敗。

---

## 2. プロジェクト名とバージョン

```cmake
project(rk4_sparse_cpp VERSION 1.0.0)
```

* **文法**: プロジェクト識別子を設定。
* **役割**: `name` / `version` に相当。`PROJECT_NAME`, `PROJECT_VERSION` 変数が使える。
* **忘れると**: 無名プロジェクト扱い。ビルドは通るがログや install 先が分かりにくい。

---

## 3. ポリシー設定

```cmake
cmake_policy(SET CMP0148 NEW)
```

* **文法**: 特定の挙動を“新仕様”に固定。
* **役割**: `from __future__ import ...` 的に旧モードを無効化。
* **忘れると**: CMake のバージョン依存で警告。基本は無害だがチームで差異が出る。

---

## 4. C++ 標準

```cmake
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
```

* **文法**: `-std=c++17` を付け、17 未満を禁止。
* **役割**: 17 機能 (`std::optional` など) を安心して使う。
* **忘れると**: デフォルトは 11/14 → 17 機能で **ビルドエラー**。

---

## 5. OpenMP (並列化オプション)

```cmake
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()
```

* **文法**: OpenMP を探し、見つかればフラグ付与。
* **役割**: “あれば並列化” のオプショナル依存。
* **忘れると**: シングルスレッド動作 or undefined reference (並列 API を呼べば)。

---

## 6. pybind11 を探す (必須)

```cmake
find_package(pybind11 REQUIRED)
```

* **文法**: `pybind11` の CMake パッケージをロード。
* **役割**: `pybind11_add_module` などヘルパー解禁。
* **忘れると**: 直後の `pybind11_add_module` が **Unknown command** → ビルド停止。

---

## 7. Eigen3 を探す & フォールバック

```cmake
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
  set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen")
  add_library(_eigen_interface INTERFACE)
  target_include_directories(_eigen_interface INTERFACE ${EIGEN3_INCLUDE_DIR})
  add_library(Eigen3::Eigen ALIAS _eigen_interface)
endif()
```

* **文法**: 既存の Eigen が無ければ vendoring。
* **役割**: CI やクロスビルドで “ヘッダだけ” を内蔵出来るように。
* **忘れると**: Eigen が無い環境で **fatal error: Eigen/...**。

---

## 8. ソースファイル一覧

```cmake
set(SOURCES
    src/core/excitation_rk4_sparse.cpp
    src/bindings/python_bindings.cpp
)
```

* **文法**: 変数にファイルパスを格納。
* **役割**: Python の `package_data` 的。
* **忘れると**: コンパイル対象ゼロ → 空モジュール or リンクエラー。

---

## 9. Python 拡張モジュールを生成

```cmake
pybind11_add_module(_rk4_sparse_cpp MODULE ${SOURCES})
```

* **文法**: 拡張モジュール (.so/.pyd) ターゲットを生成。
* **役割**: `setuptools.Extension` 相当。
* **忘れると**: C++ コードがビルドされず **import 不可**。

---

## 10. インクルードパス追加

```cmake
target_include_directories(_rk4_sparse_cpp PRIVATE include)
```

* **文法**: モジュール用にヘッダ検索パス（この場合は相対パス```include```）を追加。
* **役割**: `#include "..."` 解決。
* **忘れると**: **No such file or directory**。
* **PRIVATE 指定**:	```_rk4_sparse_cpp```自体のコンパイル時にのみヘッダ検索パスが追加される。リンク先には伝播しない。

---

## 11. ライブラリリンク

```cmake
target_link_libraries(_rk4_sparse_cpp PRIVATE Eigen3::Eigen)
# OpenMP は条件付きで同様に追加
```

* **文法**: 依存ライブラリをリンク。
* **役割**: シンボル解決。
* **忘れると**: **undefined reference** や並列なし。

---

## 12. 最適化フラグ

```cmake
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_rk4_sparse_cpp PRIVATE -O3 -march=native)
elseif(MSVC)
    target_compile_options(_rk4_sparse_cpp PRIVATE /O2)
endif()
```

* **文法**: コンパイラ別に最適化オプション。
* **役割**: 実行速度向上。
* **忘れると**: 動くが **遅い**。

---

## 13. インストール先

```cmake
install(TARGETS _rk4_sparse_cpp
    LIBRARY DESTINATION rk4_sparse  # Unix
    RUNTIME DESTINATION rk4_sparse) # Windows
```

* **文法**: `cmake --install` 時のコピー先を指定。
* **役割**: `pip install` 時に Python パッケージに .so/.pyd が含まれるように。
* **忘れると**: wheel にモジュールが入らず **ImportError**。

---

## 14. ビルドログ

```cmake
message(STATUS "Building ${PROJECT_NAME} version ${PROJECT_VERSION}")
```

* **文法**: コンソールに情報を出力。
* **役割**: デバッグ・CI の可読性向上。
* **忘れると**: 静かなビルドになるだけ。

---

### おまけ: 雛形まとめ

```cmake
cmake_minimum_required(VERSION 3.15)
project(my_module VERSION 0.1.0)
cmake_policy(SET CMP0148 NEW)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(OpenMP)
find_package(pybind11 REQUIRED)
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
  set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen")
  add_library(_eigen_interface INTERFACE)
  target_include_directories(_eigen_interface INTERFACE ${EIGEN3_INCLUDE_DIR})
  add_library(Eigen3::Eigen ALIAS _eigen_interface)
endif()

set(SOURCES src/core.cpp src/bindings.cpp)
pybind11_add_module(_my_module MODULE ${SOURCES})

target_include_directories(_my_module PRIVATE include)
target_link_libraries(_my_module PRIVATE Eigen3::Eigen)
if(OpenMP_CXX_FOUND)
    target_link_libraries(_my_module PRIVATE OpenMP::OpenMP_CXX)
endif()

if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_my_module PRIVATE -O3 -march=native)
elseif(MSVC)
    target_compile_options(_my_module PRIVATE /O2)
endif()

install(TARGETS _my_module LIBRARY DESTINATION my_module RUNTIME DESTINATION my_module)

message(STATUS "Building ${PROJECT_NAME} ${PROJECT_VERSION}")
```
- ```_my_module```: 共有ライブラリ (.so / .pyd) の基底名
  -> pythonで```import _my_module```と書く。
- ```src/core.cpp```, ```src/bindings.cpp```: ビルドするc++ソースファイル
このメモを土台に “コピペ → 名前とパスを調整” すれば、ほぼどの環境でも通る CMakeLists.txt を組めます。
