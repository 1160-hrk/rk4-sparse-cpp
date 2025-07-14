#pragma once

// Eigenの最適化設定
#define EIGEN_VECTORIZE
#define EIGEN_DONT_ALIGN_STATICALLY

#include <Eigen/Sparse>
#include <vector>
#include <complex>

namespace excitation_rk4_sparse {

// キャッシュ性能の分析用構造体
struct PerformanceMetrics {
    double matrix_update_time = 0.0;
    double rk4_step_time = 0.0;
    size_t matrix_updates = 0;
    size_t rk4_steps = 0;
};

// 新しい関数 - Python APIと互換性のある実装
Eigen::MatrixXcd rk4_sparse_cpp(
    const Eigen::SparseMatrix<std::complex<double>>& H0,
    const Eigen::SparseMatrix<std::complex<double>>& mux,
    const Eigen::SparseMatrix<std::complex<double>>& muy,
    const Eigen::VectorXd& Ex,
    const Eigen::VectorXd& Ey,
    const Eigen::VectorXcd& psi0,
    double dt,
    bool return_traj,
    int stride,
    bool renorm = false);

// ヘルパー関数
std::vector<std::vector<double>> field_to_triplets(const Eigen::VectorXd& field);

} // namespace excitation_rk4_sparse
