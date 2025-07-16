#pragma once

#include "core.hpp"
#include "suitesparse.hpp"
#include <chrono>
#include <string>
#include <vector>

namespace excitation_rk4_sparse {

struct BenchmarkResult {
    std::string implementation;
    double total_time;
    double matrix_update_time;
    double rk4_step_time;
    size_t matrix_updates;
    size_t rk4_steps;
    double speedup_vs_eigen;
};

// 両実装の速度比較
std::vector<BenchmarkResult> benchmark_implementations(
    const Eigen::SparseMatrix<std::complex<double>>& H0,
    const Eigen::SparseMatrix<std::complex<double>>& mux,
    const Eigen::SparseMatrix<std::complex<double>>& muy,
    const Eigen::VectorXd& Ex,
    const Eigen::VectorXd& Ey,
    const Eigen::VectorXcd& psi0,
    double dt,
    bool return_traj,
    int stride,
    bool renorm = false,
    int num_runs = 5);

} // namespace excitation_rk4_sparse 