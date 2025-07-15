# =============================  CMakeLists.txt  =============================
cmake_minimum_required(VERSION 3.15)

# ---------------------------------------------------------------------------
# 1. CMake ポリシー
# ---------------------------------------------------------------------------
cmake_policy(SET CMP0148 NEW)

project(${SKBUILD_PROJECT_NAME}
        VERSION ${SKBUILD_PROJECT_VERSION}
        LANGUAGES CXX)


# ---------------------------------------------------------------------------
# 2. C++17
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ---------------------------------------------------------------------------
# 3. OpenMP (任意)
# ---------------------------------------------------------------------------
find_package(OpenMP)
if(OpenMP_CXX_FOUND)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

# ---------------------------------------------------------------------------
# 4. pybind11
# ---------------------------------------------------------------------------
find_package(pybind11 REQUIRED)

# ---------------------------------------------------------------------------
# 5. Eigen (システムに無ければ同梱ヘッダを使用)
# ---------------------------------------------------------------------------
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, using bundled copy")
    set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen")

    add_library(_eigen_interface INTERFACE)
    target_include_directories(_eigen_interface INTERFACE ${EIGEN3_INCLUDE_DIR})
    add_library(Eigen3::Eigen ALIAS _eigen_interface)
endif()

# ---------------------------------------------------------------------------
# 6. バージョンファイルを自動生成
# ---------------------------------------------------------------------------
file(MAKE_DIRECTORY ${CMAKE_CURRENT_BINARY_DIR}/generated)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/include/excitation_rk4_sparse/version.hpp.in
    ${CMAKE_CURRENT_BINARY_DIR}/generated/version.hpp
    @ONLY
)

configure_file(
    ${CMAKE_CURRENT_SOURCE_DIR}/python/rk4_sparse/__init__.py.in
    ${CMAKE_CURRENT_BINARY_DIR}/python/rk4_sparse/__init__.py
    @ONLY
)

# ---------------------------------------------------------------------------
# 7. ソースファイル
# ---------------------------------------------------------------------------
set(SOURCES
    src/core/excitation_rk4_sparse.cpp
    src/bindings/python_bindings.cpp
)

# ---------------------------------------------------------------------------
# 8. 拡張モジュールをビルド
# ---------------------------------------------------------------------------
pybind11_add_module(_rk4_sparse_cpp MODULE ${SOURCES})

target_include_directories(_rk4_sparse_cpp
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/generated  # ← version.hpp を含む
)

target_link_libraries(_rk4_sparse_cpp PRIVATE Eigen3::Eigen)

if(OpenMP_CXX_FOUND)
    target_link_libraries(_rk4_sparse_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# 9. 最適化フラグ
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_rk4_sparse_cpp PRIVATE -O3 -march=native)
elseif(MSVC)
    target_compile_options(_rk4_sparse_cpp PRIVATE /O2)
endif()

# ---------------------------------------------------------------------------
# 10. インストール設定
# ---------------------------------------------------------------------------
install(TARGETS _rk4_sparse_cpp
    LIBRARY DESTINATION rk4_sparse          # Unix (.so)
    RUNTIME DESTINATION rk4_sparse)         # Windows (.pyd/.dll)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/python/rk4_sparse/__init__.py
    DESTINATION rk4_sparse
)

# ---------------------------------------------------------------------------
# 11. ビルドログ
# ---------------------------------------------------------------------------
message(STATUS "Building ${PROJECT_NAME} version ${PROJECT_VERSION}")
message(STATUS "Python version: ${Python_VERSION}")
message(STATUS "pybind11 version: ${pybind11_VERSION}")
message(STATUS "Source files: ${SOURCES}")
message(STATUS "Include directories: ${CMAKE_CURRENT_SOURCE_DIR}/include; ${CMAKE_CURRENT_BINARY_DIR}/generated")

# =============================  end of file  ===============================
