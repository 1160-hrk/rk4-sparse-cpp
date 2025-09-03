#pragma once

// Eigenの最適化設定（重複定義を回避）
#ifndef EIGEN_VECTORIZE
#define EIGEN_VECTORIZE
#endif

#ifndef EIGEN_DONT_ALIGN_STATICALLY
#define EIGEN_DONT_ALIGN_STATICALLY
#endif

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

// Eigen版のRK4実装（関数名を明確化）
Eigen::MatrixXcd rk4_sparse_eigen(
    const Eigen::SparseMatrix<std::complex<double>>& H0,
    const Eigen::SparseMatrix<std::complex<double>>& mux,
    const Eigen::SparseMatrix<std::complex<double>>& muy,
    const Eigen::VectorXd& Ex,
    const Eigen::VectorXd& Ey,
    const Eigen::Ref<const Eigen::VectorXcd>& psi0,
    double dt,
    bool return_traj,
    int stride,
    bool renorm = false);

// パターン構築・データ展開のキャッシュ化を行う高速版
Eigen::MatrixXcd rk4_sparse_eigen_cached(
    const Eigen::SparseMatrix<std::complex<double>>& H0,
    const Eigen::SparseMatrix<std::complex<double>>& mux,
    const Eigen::SparseMatrix<std::complex<double>>& muy,
    const Eigen::VectorXd& Ex,
    const Eigen::VectorXd& Ey,
    const Eigen::Ref<const Eigen::VectorXcd>& psi0,
    double dt,
    bool return_traj,
    int stride,
    bool renorm = false);

// BLAS最適化版のスパース行列-ベクトル積
void blas_optimized_sparse_matrix_vector_multiply(
    const std::complex<double>* H_data,
    const int* H_indices,
    const int* H_indptr,
    const Eigen::VectorXcd& x,
    Eigen::VectorXcd& y,
    int dim);

// より効率的なBLAS実装（メモリ割り当てを最小化）
void blas_optimized_sparse_matrix_vector_multiply_efficient(
    const std::complex<double>* H_data,
    const int* H_indices,
    const int* H_indptr,
    const Eigen::VectorXcd& x,
    Eigen::VectorXcd& y,
    int dim);

// 安全なBLAS最適化版スパース行列-ベクトル積
void blas_optimized_sparse_matrix_vector_multiply_safe(
    const std::complex<double>* H_data,
    const int* H_indices,
    const int* H_indptr,
    const Eigen::VectorXcd& x,
    Eigen::VectorXcd& y,
    int dim);

// BLAS最適化版のRK4実装
Eigen::MatrixXcd rk4_sparse_blas_optimized(
    const Eigen::SparseMatrix<std::complex<double>>& H0,
    const Eigen::SparseMatrix<std::complex<double>>& mux,
    const Eigen::SparseMatrix<std::complex<double>>& muy,
    const Eigen::VectorXd& Ex,
    const Eigen::VectorXd& Ey,
    const Eigen::Ref<const Eigen::VectorXcd>& psi0,
    double dt,
    bool return_traj,
    int stride,
    bool renorm = false);

// より効率的なBLAS最適化版のRK4実装
Eigen::MatrixXcd rk4_sparse_blas_optimized_efficient(
    const Eigen::SparseMatrix<std::complex<double>>& H0,
    const Eigen::SparseMatrix<std::complex<double>>& mux,
    const Eigen::SparseMatrix<std::complex<double>>& muy,
    const Eigen::VectorXd& Ex,
    const Eigen::VectorXd& Ey,
    const Eigen::Ref<const Eigen::VectorXcd>& psi0,
    double dt,
    bool return_traj,
    int stride,
    bool renorm = false);

// 安全なBLAS最適化版のRK4実装
Eigen::MatrixXcd rk4_sparse_blas_optimized_safe(
    const Eigen::SparseMatrix<std::complex<double>>& H0,
    const Eigen::SparseMatrix<std::complex<double>>& mux,
    const Eigen::SparseMatrix<std::complex<double>>& muy,
    const Eigen::VectorXd& Ex,
    const Eigen::VectorXd& Ey,
    const Eigen::Ref<const Eigen::VectorXcd>& psi0,
    double dt,
    bool return_traj,
    int stride,
    bool renorm = false);

// 最適化されたSuiteSparse版のRK4実装
Eigen::MatrixXcd rk4_sparse_suitesparse_optimized(
    const Eigen::SparseMatrix<std::complex<double>>& H0,
    const Eigen::SparseMatrix<std::complex<double>>& mux,
    const Eigen::SparseMatrix<std::complex<double>>& muy,
    const Eigen::VectorXd& Ex,
    const Eigen::VectorXd& Ey,
    const Eigen::Ref<const Eigen::VectorXcd>& psi0,
    double dt,
    bool return_traj,
    int stride,
    bool renorm = false);

// 高速SuiteSparse版のRK4実装
Eigen::MatrixXcd rk4_sparse_suitesparse_fast(
    const Eigen::SparseMatrix<std::complex<double>>& H0,
    const Eigen::SparseMatrix<std::complex<double>>& mux,
    const Eigen::SparseMatrix<std::complex<double>>& muy,
    const Eigen::VectorXd& Ex,
    const Eigen::VectorXd& Ey,
    const Eigen::Ref<const Eigen::VectorXcd>& psi0,
    double dt,
    bool return_traj,
    int stride,
    bool renorm = false);

// ヘルパー関数
std::vector<std::vector<double>> field_to_triplets(const Eigen::VectorXd& field);

// Julia風の高速実装（シンプルで効率的）
Eigen::MatrixXcd rk4_sparse_julia_style(
    const Eigen::SparseMatrix<std::complex<double>>& H0,
    const Eigen::SparseMatrix<std::complex<double>>& mux,
    const Eigen::SparseMatrix<std::complex<double>>& muy,
    const Eigen::VectorXd& Ex,
    const Eigen::VectorXd& Ey,
    const Eigen::Ref<const Eigen::VectorXcd>& psi0,
    double dt,
    bool return_traj,
    int stride,
    bool renorm = false);

// CSR形式による超高速実装
Eigen::MatrixXcd rk4_sparse_csr_optimized(
    const Eigen::SparseMatrix<std::complex<double>>& H0,
    const Eigen::SparseMatrix<std::complex<double>>& mux,
    const Eigen::SparseMatrix<std::complex<double>>& muy,
    const Eigen::VectorXd& Ex,
    const Eigen::VectorXd& Ey,
    const Eigen::Ref<const Eigen::VectorXcd>& psi0,
    double dt,
    bool return_traj,
    int stride,
    bool renorm = false);

// 中規模問題特化のSIMD最適化実装 (100-1000次元に最適化)
Eigen::MatrixXcd rk4_sparse_medium_scale_optimized(
    const Eigen::SparseMatrix<std::complex<double>>& H0,
    const Eigen::SparseMatrix<std::complex<double>>& mux,
    const Eigen::SparseMatrix<std::complex<double>>& muy,
    const Eigen::VectorXd& Ex,
    const Eigen::VectorXd& Ey,
    const Eigen::Ref<const Eigen::VectorXcd>& psi0,
    double dt,
    bool return_traj,
    int stride,
    bool renorm = false);

// ユーティリティ関数
std::vector<std::vector<double>> field_to_triplets(const Eigen::VectorXd& field);

// Julia Killer SIMD実装
Eigen::MatrixXcd julia_killer_rk4_phase1(
    const Eigen::SparseMatrix<std::complex<double>>& H0,
    const Eigen::SparseMatrix<std::complex<double>>& mux,
    const Eigen::SparseMatrix<std::complex<double>>& muy,
    const Eigen::VectorXd& Ex,
    const Eigen::VectorXd& Ey,
    const Eigen::Ref<const Eigen::VectorXcd>& psi0,
    double dt,
    bool return_traj = true,
    int stride = 1,
    bool renorm = false);

// Julia Killer補助関数
void julia_killer_sparse_matvec(
    const std::complex<double>* H_values,
    const int* H_indices,
    const int* H_indptr,
    const std::complex<double>* x,
    std::complex<double>* y,
    int dim);

void julia_killer_vector_add_scaled(
    std::complex<double>* result,
    const std::complex<double>* a,
    const std::complex<double>* b,
    double scale,
    int dim);

void julia_killer_vector_normalize(
    std::complex<double>* vec,
    int dim);

} // namespace excitation_rk4_sparse
