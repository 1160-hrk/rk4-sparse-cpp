#pragma once

#include <Eigen/Sparse>
#include <vector>
#include <complex>
#include <chrono>

namespace excitation_rk4_sparse {

// 基本版SuiteSparse用のパフォーマンスメトリクス
struct SuiteSparsePerformanceMetrics {
    double matrix_update_time = 0.0;
    double rk4_step_time = 0.0;
    double sparse_solve_time = 0.0;
    size_t matrix_updates = 0;
    size_t rk4_steps = 0;
    size_t sparse_solves = 0;
};

// 最適化版SuiteSparse用のパフォーマンスメトリクス
struct SuiteSparseOptimizedPerformanceMetrics {
    double matrix_update_time = 0.0;
    double rk4_step_time = 0.0;
    double sparse_solve_time = 0.0;
    size_t matrix_updates = 0;
    size_t rk4_steps = 0;
    size_t sparse_solves = 0;
};

// 高速版SuiteSparse用のパフォーマンスメトリクス
struct SuiteSparseFastPerformanceMetrics {
    double matrix_update_time = 0.0;
    double rk4_step_time = 0.0;
    double sparse_solve_time = 0.0;
    size_t matrix_updates = 0;
    size_t rk4_steps = 0;
    size_t sparse_solves = 0;
};

enum class OptimizationLevel {
    BASIC,
    STANDARD,
    ENHANCED
};

Eigen::MatrixXcd rk4_sparse_suitesparse(
    const Eigen::SparseMatrix<std::complex<double>>& H0,
    const Eigen::SparseMatrix<std::complex<double>>& mux,
    const Eigen::SparseMatrix<std::complex<double>>& muy,
    const Eigen::VectorXd& Ex,
    const Eigen::VectorXd& Ey,
    const Eigen::VectorXcd& psi0,
    double dt,
    bool return_traj,
    int stride,
    bool renorm,
    OptimizationLevel level = OptimizationLevel::STANDARD);

} // namespace excitation_rk4_sparse 