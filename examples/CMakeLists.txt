# Examples CMakeLists.txt
cmake_minimum_required(VERSION 3.15)

# C++17が必要
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigenライブラリのパスを設定
find_package(Eigen3 REQUIRED)

# OpenMPの検出
find_package(OpenMP)

# プロジェクトのヘッダーファイルディレクトリを追加
include_directories(${CMAKE_SOURCE_DIR}/include)

# ソースファイルディレクトリを追加  
include_directories(${CMAKE_SOURCE_DIR}/src)

# 多準位系ベンチマーク実行ファイル
add_executable(multilevel_benchmark 
    multilevel_benchmark.cpp
    ${CMAKE_SOURCE_DIR}/src/core/excitation_rk4_sparse.cpp
    ${CMAKE_SOURCE_DIR}/src/core/excitation_rk4_benchmark.cpp
)

# SuiteSparseが利用可能な場合は追加
if(TARGET SuiteSparse::CHOLMOD AND TARGET SuiteSparse::UMFPACK)
    target_sources(multilevel_benchmark PRIVATE
        ${CMAKE_SOURCE_DIR}/src/core/excitation_rk4_suitesparse.cpp
    )
    target_link_libraries(multilevel_benchmark 
        SuiteSparse::CHOLMOD
        SuiteSparse::UMFPACK
    )
    target_compile_definitions(multilevel_benchmark PRIVATE HAVE_SUITESPARSE)
endif()

# ライブラリとのリンク
target_link_libraries(multilevel_benchmark 
    Eigen3::Eigen
)

# OpenMPが利用可能な場合はリンク
if(OpenMP_CXX_FOUND)
    target_link_libraries(multilevel_benchmark OpenMP::OpenMP_CXX)
endif()

# デバッグ情報とコンパイラオプション
target_compile_definitions(multilevel_benchmark PRIVATE
    $<$<CONFIG:Debug>:DEBUG>
    $<$<CONFIG:Release>:NDEBUG>
)

# 最適化フラグ
if(CMAKE_BUILD_TYPE STREQUAL "Release")
    target_compile_options(multilevel_benchmark PRIVATE
        -O3 -march=native
    )
endif()

# ベンチマーク実行ターゲット
add_custom_target(run_multilevel_benchmark
    COMMAND multilevel_benchmark
    DEPENDS multilevel_benchmark
    WORKING_DIRECTORY ${CMAKE_BINARY_DIR}/examples
    COMMENT "Running multilevel system benchmark"
) 