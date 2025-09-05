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
# 4. Python & pybind11
# ---------------------------------------------------------------------------
# Pythonの検出（複数の方法を試行）
find_package(Python3 COMPONENTS Interpreter Development QUIET)
if(NOT Python3_FOUND)
    find_package(Python3 QUIET)
endif()
if(NOT Python3_FOUND)
    find_package(Python COMPONENTS Interpreter Development QUIET)
endif()

# Pythonが見つからない場合のフォールバック
if(NOT Python3_FOUND AND NOT Python_FOUND)
    message(FATAL_ERROR "Python not found. Please install Python development headers.")
endif()

# pybind11の検出（複数の方法を試行）
find_package(pybind11 CONFIG QUIET)
if(NOT pybind11_FOUND)
    find_package(pybind11 MODULE QUIET)
endif()

if(NOT pybind11_FOUND)
    # 手動設定（フォールバック）
    find_path(PYBIND11_INCLUDE_DIR pybind11/pybind11.h
        PATHS
        /usr/include
        /usr/local/include
        /opt/homebrew/include
        ${CMAKE_CURRENT_SOURCE_DIR}/third_party/pybind11/include
        ${Python_SITELIB}/pybind11/include
        ${Python_SITEARCH}/pybind11/include
    )
    
    if(PYBIND11_INCLUDE_DIR)
        set(pybind11_INCLUDE_DIRS ${PYBIND11_INCLUDE_DIR})
        set(pybind11_FOUND TRUE)
        message(STATUS "Found pybind11 manually: ${PYBIND11_INCLUDE_DIR}")
        
        # pybind11のツールを手動で読み込み
        find_file(PYBIND11_TOOLS_CMAKE pybind11Tools.cmake
            PATHS
            /usr/lib/cmake/pybind11
            /usr/local/lib/cmake/pybind11
            /opt/homebrew/lib/cmake/pybind11
            ${Python_SITELIB}/pybind11/share/cmake/pybind11
            ${Python_SITEARCH}/pybind11/share/cmake/pybind11
        )
        
        if(PYBIND11_TOOLS_CMAKE)
            include(${PYBIND11_TOOLS_CMAKE})
            message(STATUS "Loaded pybind11 tools from: ${PYBIND11_TOOLS_CMAKE}")
        else()
            message(WARNING "pybind11 tools not found, trying alternative paths")
            # 代替パスを試行
            find_file(PYBIND11_TOOLS_CMAKE pybind11Tools.cmake
                PATHS
                /usr/share/cmake/pybind11
                /usr/local/share/cmake/pybind11
                /opt/homebrew/share/cmake/pybind11
            )
            if(PYBIND11_TOOLS_CMAKE)
                include(${PYBIND11_TOOLS_CMAKE})
                message(STATUS "Loaded pybind11 tools from alternative path: ${PYBIND11_TOOLS_CMAKE}")
            endif()
        endif()
    endif()
endif()

if(NOT pybind11_FOUND)
    message(FATAL_ERROR "pybind11 not found. Please install pybind11 development headers.")
endif()

message(STATUS "Found pybind11: ${pybind11_INCLUDE_DIRS}")

# Pythonの設定をpybind11に明示的に渡す
if(Python3_FOUND)
    set(PYTHON_EXECUTABLE ${Python3_EXECUTABLE})
    set(PYTHON_INCLUDE_DIRS ${Python3_INCLUDE_DIRS})
    set(PYTHON_LIBRARIES ${Python3_LIBRARIES})
elseif(Python_FOUND)
    set(PYTHON_EXECUTABLE ${Python_EXECUTABLE})
    set(PYTHON_INCLUDE_DIRS ${Python_INCLUDE_DIRS})
    set(PYTHON_LIBRARIES ${Python_LIBRARIES})
endif()

# ---------------------------------------------------------------------------
# 5. Eigen (システムに無ければ同梱ヘッダを使用)
# ---------------------------------------------------------------------------
find_package(Eigen3 3.3 QUIET NO_MODULE)
if(NOT Eigen3_FOUND)
    message(STATUS "Eigen3 not found, using bundled copy")
    set(EIGEN3_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/eigen")
    
    # 同梱Eigenの存在確認
    if(NOT EXISTS "${EIGEN3_INCLUDE_DIR}/Eigen/Core")
        message(FATAL_ERROR "Bundled Eigen not found at ${EIGEN3_INCLUDE_DIR}")
    endif()

    add_library(_eigen_interface INTERFACE)
    target_include_directories(_eigen_interface INTERFACE ${EIGEN3_INCLUDE_DIR})
    add_library(Eigen3::Eigen ALIAS _eigen_interface)
    message(STATUS "Using bundled Eigen from: ${EIGEN3_INCLUDE_DIR}")
else()
    message(STATUS "Using system Eigen3: ${EIGEN3_INCLUDE_DIR}")
endif()

# ---------------------------------------------------------------------------
# 6. OpenBLAS & SuiteSparse (高速化ライブラリ)
# ---------------------------------------------------------------------------
option(USE_OPENBLAS_SUITESPARSE "Enable OpenBLAS + SuiteSparse optimization" ON)

if(USE_OPENBLAS_SUITESPARSE)
    # OpenBLASの検出（強化版）
    find_package(OpenBLAS QUIET)
    if(NOT OpenBLAS_FOUND)
        # 複数のOpenBLASライブラリ名を試行
        find_library(OPENBLAS_LIB openblas)
        if(NOT OPENBLAS_LIB)
            find_library(OPENBLAS_LIB openblas64)
        endif()
        if(NOT OPENBLAS_LIB)
            find_library(OPENBLAS_LIB libopenblas)
        endif()
        if(NOT OPENBLAS_LIB)
            find_library(OPENBLAS_LIB libopenblas64)
        endif()
        
        if(OPENBLAS_LIB)
            set(OpenBLAS_LIBRARIES ${OPENBLAS_LIB})
            set(OpenBLAS_FOUND TRUE)
            message(STATUS "OpenBLAS found: ${OPENBLAS_LIB}")
        endif()
    endif()
    
    # SuiteSparseの検出
    find_package(SuiteSparse QUIET)
    if(NOT SuiteSparse_FOUND)
        # SuiteSparseの主要コンポーネントを個別に検索
        find_library(SUITESPARSE_CHOLMOD cholmod)
        find_library(SUITESPARSE_UMFPACK umfpack)
        find_library(SUITESPARSE_AMD amd)
        find_library(SUITESPARSE_COLAMD colamd)
        find_library(SUITESPARSE_CAMD camd)
        find_library(SUITESPARSE_CCOLAMD ccolamd)
        find_library(SUITESPARSE_CONFIG suitesparseconfig)
        
        if(SUITESPARSE_CHOLMOD AND SUITESPARSE_UMFPACK)
            set(SuiteSparse_LIBRARIES 
                ${SUITESPARSE_CHOLMOD}
                ${SUITESPARSE_UMFPACK}
                ${SUITESPARSE_AMD}
                ${SUITESPARSE_COLAMD}
                ${SUITESPARSE_CAMD}
                ${SUITESPARSE_CCOLAMD}
                ${SUITESPARSE_CONFIG}
            )
            set(SuiteSparse_FOUND TRUE)
            message(STATUS "SuiteSparse found: ${SuiteSparse_LIBRARIES}")
        endif()
    endif()
    
    # 個別検索でライブラリが見つかった場合の追加チェック
    if(NOT OpenBLAS_FOUND AND OPENBLAS_LIB)
        set(OpenBLAS_FOUND TRUE)
        set(OpenBLAS_LIBRARIES ${OPENBLAS_LIB})
        message(STATUS "OpenBLAS found via individual search: ${OPENBLAS_LIB}")
    endif()
    
    if(NOT SuiteSparse_FOUND AND SUITESPARSE_CHOLMOD AND SUITESPARSE_UMFPACK)
        set(SuiteSparse_FOUND TRUE)
        set(SuiteSparse_LIBRARIES 
            ${SUITESPARSE_CHOLMOD}
            ${SUITESPARSE_UMFPACK}
            ${SUITESPARSE_AMD}
            ${SUITESPARSE_COLAMD}
            ${SUITESPARSE_CAMD}
            ${SUITESPARSE_CCOLAMD}
            ${SUITESPARSE_CONFIG}
        )
        message(STATUS "SuiteSparse found via individual search: ${SuiteSparse_LIBRARIES}")
    endif()
    
    if((OpenBLAS_FOUND OR OPENBLAS_LIB) AND (SuiteSparse_FOUND OR (SUITESPARSE_CHOLMOD AND SUITESPARSE_UMFPACK)))
        add_definitions(-DOPENBLAS_SUITESPARSE_AVAILABLE)
        message(STATUS "OpenBLAS + SuiteSparse optimization enabled")
    else()
        message(WARNING "OpenBLAS or SuiteSparse not found. Using Eigen-only implementation.")
        set(USE_OPENBLAS_SUITESPARSE OFF)
    endif()
endif()

# ---------------------------------------------------------------------------
# 7. SuiteSparse-MKL (オプション、x86_64のみ)
# ---------------------------------------------------------------------------
option(USE_SUITESPARSE_MKL "Enable SuiteSparse-MKL support" OFF)

if(USE_SUITESPARSE_MKL)
    # Intel MKLの検出
    find_package(MKL QUIET)
    if(MKL_FOUND)
        message(STATUS "Intel MKL found: ${MKL_LIBRARIES}")
        add_definitions(-DSUITESPARSE_MKL_AVAILABLE)
        set(SUITESPARSE_MKL_LIBRARIES ${MKL_LIBRARIES})
        set(SUITESPARSE_MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIRS})
    else()
        # システムのMKLを検索
        find_library(MKL_CORE_LIB mkl_core)
        find_library(MKL_INTEL_LP64_LIB mkl_intel_lp64)
        find_library(MKL_SEQUENTIAL_LIB mkl_sequential)
        find_library(MKL_SPARSE_LIB mkl_sparse_blas)
        
        if(MKL_CORE_LIB AND MKL_INTEL_LP64_LIB AND MKL_SEQUENTIAL_LIB AND MKL_SPARSE_LIB)
            message(STATUS "Intel MKL libraries found")
            add_definitions(-DSUITESPARSE_MKL_AVAILABLE)
            set(SUITESPARSE_MKL_LIBRARIES 
                ${MKL_CORE_LIB} 
                ${MKL_INTEL_LP64_LIB} 
                ${MKL_SEQUENTIAL_LIB}
                ${MKL_SPARSE_LIB}
            )
        else()
            message(WARNING "SuiteSparse-MKL requested but Intel MKL not found. Falling back to Eigen-only implementation.")
            set(USE_SUITESPARSE_MKL OFF)
        endif()
    endif()
endif()

# ---------------------------------------------------------------------------
# 8. バージョンファイルを自動生成
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
# 9. ソースファイル
# ---------------------------------------------------------------------------
set(SOURCES
    src/core/excitation_rk4_sparse.cpp
    src/core/excitation_rk4_benchmark.cpp
    src/bindings/python_bindings.cpp
)

# OpenBLAS + SuiteSparseが有効な場合、追加のソースファイルを含める
if(USE_OPENBLAS_SUITESPARSE)
    list(APPEND SOURCES 
        src/core/excitation_rk4_suitesparse.cpp
    )
endif()

# SuiteSparse-MKLが有効な場合、追加のソースファイルを含める
if(USE_SUITESPARSE_MKL)
    list(APPEND SOURCES 
        src/core/excitation_rk4_suitesparse.cpp
    )
endif()

# ---------------------------------------------------------------------------
# 10. 拡張モジュールをビルド
# ---------------------------------------------------------------------------
pybind11_add_module(_rk4_sparse_cpp MODULE ${SOURCES})

target_include_directories(_rk4_sparse_cpp
    PRIVATE
        ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_BINARY_DIR}/generated  # ← version.hpp を含む
)

# OpenBLAS + SuiteSparseのインクルードディレクトリを追加
if(USE_OPENBLAS_SUITESPARSE AND (OpenBLAS_FOUND OR OPENBLAS_LIB))
    if(OpenBLAS_INCLUDE_DIRS)
        target_include_directories(_rk4_sparse_cpp
            PRIVATE ${OpenBLAS_INCLUDE_DIRS}
        )
    endif()
endif()

# SuiteSparse-MKLのインクルードディレクトリを追加
if(USE_SUITESPARSE_MKL AND SUITESPARSE_MKL_INCLUDE_DIRS)
    target_include_directories(_rk4_sparse_cpp
        PRIVATE ${SUITESPARSE_MKL_INCLUDE_DIRS}
    )
endif()

target_link_libraries(_rk4_sparse_cpp PRIVATE Eigen3::Eigen)

# OpenBLAS + SuiteSparseライブラリをリンク
if(USE_OPENBLAS_SUITESPARSE AND (OpenBLAS_FOUND OR OPENBLAS_LIB) AND (SuiteSparse_FOUND OR (SUITESPARSE_CHOLMOD AND SUITESPARSE_UMFPACK)))
    add_definitions(-DOPENBLAS_SUITESPARSE_AVAILABLE)
    message(STATUS "OpenBLAS + SuiteSparse optimization enabled")
    
    # ライブラリリストを構築
    set(LINK_LIBRARIES)
    if(OpenBLAS_LIBRARIES)
        list(APPEND LINK_LIBRARIES ${OpenBLAS_LIBRARIES})
    elseif(OPENBLAS_LIB)
        list(APPEND LINK_LIBRARIES ${OPENBLAS_LIB})
    endif()
    
    # 追加のBLASライブラリを明示的にリンク
    find_library(BLAS_LIB blas)
    if(BLAS_LIB)
        list(APPEND LINK_LIBRARIES ${BLAS_LIB})
        message(STATUS "BLAS library found: ${BLAS_LIB}")
    endif()
    
    find_library(CBLAS_LIB cblas)
    if(CBLAS_LIB)
        list(APPEND LINK_LIBRARIES ${CBLAS_LIB})
        message(STATUS "CBLAS library found: ${CBLAS_LIB}")
    endif()
    
    # 複数のBLAS実装を試行
    if(NOT BLAS_LIB)
        find_library(BLAS_LIB openblas)
    endif()
    if(NOT BLAS_LIB)
        find_library(BLAS_LIB libopenblas)
    endif()
    if(NOT BLAS_LIB)
        find_library(BLAS_LIB libblas)
    endif()
    
    if(BLAS_LIB)
        list(APPEND LINK_LIBRARIES ${BLAS_LIB})
        message(STATUS "BLAS library linked: ${BLAS_LIB}")
    else()
        message(WARNING "BLAS library not found, falling back to Eigen-only implementation")
        set(USE_OPENBLAS_SUITESPARSE OFF)
        add_definitions(-UOPENBLAS_SUITESPARSE_AVAILABLE)
    endif()
    
    if(SuiteSparse_LIBRARIES)
        list(APPEND LINK_LIBRARIES ${SuiteSparse_LIBRARIES})
    elseif(SUITESPARSE_CHOLMOD AND SUITESPARSE_UMFPACK)
        list(APPEND LINK_LIBRARIES 
            ${SUITESPARSE_CHOLMOD}
            ${SUITESPARSE_UMFPACK}
            ${SUITESPARSE_AMD}
            ${SUITESPARSE_COLAMD}
            ${SUITESPARSE_CAMD}
            ${SUITESPARSE_CCOLAMD}
            ${SUITESPARSE_CONFIG}
        )
    endif()
    
    target_link_libraries(_rk4_sparse_cpp PRIVATE ${LINK_LIBRARIES})
else()
    message(WARNING "OpenBLAS or SuiteSparse not found. Using Eigen-only implementation.")
    set(USE_OPENBLAS_SUITESPARSE OFF)
    add_definitions(-UOPENBLAS_SUITESPARSE_AVAILABLE)
endif()

# SuiteSparse-MKLライブラリをリンク
if(USE_SUITESPARSE_MKL AND SUITESPARSE_MKL_LIBRARIES)
    target_link_libraries(_rk4_sparse_cpp PRIVATE ${SUITESPARSE_MKL_LIBRARIES})
endif()

if(OpenMP_CXX_FOUND)
    target_link_libraries(_rk4_sparse_cpp PRIVATE OpenMP::OpenMP_CXX)
endif()

# ---------------------------------------------------------------------------
# 11. 最適化フラグ（BLAS最適化対応）
# ---------------------------------------------------------------------------
if(CMAKE_CXX_COMPILER_ID MATCHES "GNU|Clang")
    target_compile_options(_rk4_sparse_cpp PRIVATE
        -O3
        -march=native
        -mtune=native
        -ffast-math
        -funroll-loops
        -fomit-frame-pointer
        -DNDEBUG
    )
    
    # リンク時最適化
    set_property(TARGET _rk4_sparse_cpp PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
elseif(MSVC)
    target_compile_options(_rk4_sparse_cpp PRIVATE /O2 /GL)
    set_property(TARGET _rk4_sparse_cpp PROPERTY INTERPROCEDURAL_OPTIMIZATION TRUE)
endif()

# ---------------------------------------------------------------------------
# 12. インストール設定
# ---------------------------------------------------------------------------
install(TARGETS _rk4_sparse_cpp
    LIBRARY DESTINATION rk4_sparse          # Unix (.so)
    RUNTIME DESTINATION rk4_sparse)         # Windows (.pyd/.dll)

install(
    FILES ${CMAKE_CURRENT_BINARY_DIR}/python/rk4_sparse/__init__.py
    DESTINATION rk4_sparse
)

# Install additional pure-Python modules (api.py, fast.py, etc.) under rk4_sparse
install(
    DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/python/rk4_sparse/
    DESTINATION rk4_sparse
    FILES_MATCHING
        PATTERN "*.py"
        PATTERN "__init__.py" EXCLUDE
)

# ---------------------------------------------------------------------------
# 13. ビルドログ
# ---------------------------------------------------------------------------
message(STATUS "Building ${PROJECT_NAME} version ${PROJECT_VERSION}")
message(STATUS "Python version: ${Python_VERSION}")
message(STATUS "pybind11 version: ${pybind11_VERSION}")
message(STATUS "OpenBLAS + SuiteSparse support: ${USE_OPENBLAS_SUITESPARSE}")
if(USE_OPENBLAS_SUITESPARSE AND (OpenBLAS_FOUND OR OPENBLAS_LIB) AND (SuiteSparse_FOUND OR (SUITESPARSE_CHOLMOD AND SUITESPARSE_UMFPACK)))
    if(OpenBLAS_LIBRARIES)
        message(STATUS "OpenBLAS libraries: ${OpenBLAS_LIBRARIES}")
    elseif(OPENBLAS_LIB)
        message(STATUS "OpenBLAS library: ${OPENBLAS_LIB}")
    endif()
    if(SuiteSparse_LIBRARIES)
        message(STATUS "SuiteSparse libraries: ${SuiteSparse_LIBRARIES}")
    elseif(SUITESPARSE_CHOLMOD AND SUITESPARSE_UMFPACK)
        message(STATUS "SuiteSparse libraries: ${SUITESPARSE_CHOLMOD}, ${SUITESPARSE_UMFPACK}, etc.")
    endif()
endif()
message(STATUS "SuiteSparse-MKL support: ${USE_SUITESPARSE_MKL}")
if(USE_SUITESPARSE_MKL)
    message(STATUS "SuiteSparse-MKL libraries: ${SUITESPARSE_MKL_LIBRARIES}")
endif()
message(STATUS "Source files: ${SOURCES}")
message(STATUS "Include directories: ${CMAKE_CURRENT_SOURCE_DIR}/include; ${CMAKE_CURRENT_BINARY_DIR}/generated")

# =============================  end of file  ===============================
