cmake_minimum_required(VERSION 3.10)
project(excitation_rk4_sparse)

# C++17を使用
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# OpenMPを有効化
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# pybind11を探す
find_package(pybind11 REQUIRED)

# Eigenを探す
find_package(Eigen3 3.3 REQUIRED NO_MODULE)

# ソースファイルの指定
set(SOURCES
    src/excitation_rk4_sparse.cpp
    src/bindings.cpp
)

# インクルードディレクトリの設定
include_directories(
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${EIGEN3_INCLUDE_DIR}
)

# Python モジュールの作成
pybind11_add_module(_excitation_rk4_sparse ${SOURCES})

# Eigenをリンク
target_link_libraries(_excitation_rk4_sparse PRIVATE Eigen3::Eigen)

# OpenMPをリンク（利用可能な場合）
if(OpenMP_CXX_FOUND)
    target_link_libraries(_excitation_rk4_sparse PRIVATE OpenMP::OpenMP_CXX)
endif()

# 最適化フラグの設定
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_excitation_rk4_sparse PRIVATE -O3 -march=native)
elseif(MSVC)
    target_compile_options(_excitation_rk4_sparse PRIVATE /O2)
endif()

# インストール設定
install(TARGETS _excitation_rk4_sparse DESTINATION python)
