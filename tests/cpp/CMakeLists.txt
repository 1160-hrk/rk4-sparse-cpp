cmake_minimum_required(VERSION 3.15)
project(rk4_sparse_cpp_tests)

# C++標準の設定
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Eigenの検索
find_package(Eigen3 REQUIRED)

# テスト実行ファイルの作成
add_executable(test_sparse_matrix_operations test_sparse_matrix_operations.cpp)
target_link_libraries(test_sparse_matrix_operations Eigen3::Eigen)

# テストの実行
add_custom_target(run_cpp_tests
    COMMAND ${CMAKE_CURRENT_BINARY_DIR}/test_sparse_matrix_operations
    DEPENDS test_sparse_matrix_operations
    COMMENT "Running C++ tests..."
)

# テスト結果の出力
add_custom_command(TARGET run_cpp_tests POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E echo "C++ tests completed successfully"
) 