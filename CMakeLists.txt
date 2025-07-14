cmake_minimum_required(VERSION 3.15)  # バージョンを上げて新しいPython機能をサポート
project(rk4_sparse_cpp VERSION 1.0.0)

# CMakeポリシーの設定
cmake_policy(SET CMP0148 NEW)

# C++17を使用
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMPを有効化
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# Pythonを探す
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
message(STATUS "Using Python3 executable: ${Python3_EXECUTABLE}")

# pybind11を探す
find_package(pybind11 REQUIRED)

# Eigenを探す
find_package(Eigen3 3.3 REQUIRED NO_MODULE)


# ソースファイルの指定（新しいパス）
set(SOURCES
    src/core/excitation_rk4_sparse.cpp
    src/bindings/python_bindings.cpp
)

# インクルードディレクトリの設定（新しいパス）
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Python モジュールの作成
pybind11_add_module(_rk4_sparse_cpp MODULE ${SOURCES})

# Eigenをリンク
target_link_libraries(_rk4_sparse_cpp PRIVATE Eigen3::Eigen)

# OpenMPをリンク（利用可能な場合）
if(OpenMP_CXX_FOUND)
    target_link_libraries(_rk4_sparse_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# 最適化フラグの設定
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_rk4_sparse_cpp PRIVATE -O3 -march=native)
elseif(MSVC)
    target_compile_options(_rk4_sparse_cpp PRIVATE /O2)
endif()

# インストール設定
install(TARGETS _rk4_sparse_cpp
    LIBRARY DESTINATION rk4_sparse          # Unix (.so)
    RUNTIME DESTINATION rk4_sparse)         # Windows (.pyd/.dll)

# バージョン情報の出力
message(STATUS "Building ${PROJECT_NAME} version ${PROJECT_VERSION}")
message(STATUS "Python version: ${Python_VERSION}")
message(STATUS "pybind11 version: ${pybind11_VERSION}")

# 将来のアルゴリズム追加に備えた設定
message(STATUS "Source files: ${SOURCES}")
message(STATUS "Include directories: ${CMAKE_CURRENT_SOURCE_DIR}/include")
