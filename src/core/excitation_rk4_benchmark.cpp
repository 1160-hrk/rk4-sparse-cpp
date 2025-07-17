#include "excitation_rk4_sparse/benchmark.hpp"
#include "excitation_rk4_sparse/core.hpp"
#include "excitation_rk4_sparse/suitesparse.hpp"
#include <iostream>
#include <algorithm>
#include <iomanip>
#include <numeric>

namespace excitation_rk4_sparse {

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
    bool renorm,
    int num_runs)
{
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double>;
    
    std::vector<BenchmarkResult> results;
    
    // Eigen版のベンチマーク
    std::vector<double> eigen_times;
    for (int run = 0; run < num_runs; ++run) {
        auto start = Clock::now();
        auto result = rk4_sparse_eigen(H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm);
        auto end = Clock::now();
        eigen_times.push_back(Duration(end - start).count());
    }
    
    // 平均時間を計算
    double eigen_avg_time = std::accumulate(eigen_times.begin(), eigen_times.end(), 0.0) / eigen_times.size();
    
    BenchmarkResult eigen_result;
    eigen_result.implementation = "Eigen";
    eigen_result.total_time = eigen_avg_time;
    eigen_result.speedup_vs_eigen = 1.0;
    results.push_back(eigen_result);
    
    // SuiteSparse版のベンチマーク（利用可能な場合）
    #ifdef SUITESPARSE_MKL_AVAILABLE
    std::vector<double> suitesparse_times;
    for (int run = 0; run < num_runs; ++run) {
        auto start = Clock::now();
        auto result = rk4_sparse_suitesparse(H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm);
        auto end = Clock::now();
        suitesparse_times.push_back(Duration(end - start).count());
    }
    
    double suitesparse_avg_time = std::accumulate(suitesparse_times.begin(), suitesparse_times.end(), 0.0) / suitesparse_times.size();
    
    BenchmarkResult suitesparse_result;
    suitesparse_result.implementation = "SuiteSparse-MKL";
    suitesparse_result.total_time = suitesparse_avg_time;
    suitesparse_result.speedup_vs_eigen = eigen_avg_time / suitesparse_avg_time;
    results.push_back(suitesparse_result);
    #endif
    
    // 結果を出力
    std::cout << "\n=== ベンチマーク結果 ===\n";
    std::cout << std::setw(20) << "実装" << std::setw(15) << "平均時間(秒)" << std::setw(15) << "Eigen比" << "\n";
    std::cout << std::string(50, '-') << "\n";
    
    for (const auto& result : results) {
        std::cout << std::setw(20) << result.implementation 
                  << std::setw(15) << std::fixed << std::setprecision(6) << result.total_time
                  << std::setw(15) << std::fixed << std::setprecision(3) << result.speedup_vs_eigen << "x\n";
    }
    
    return results;
}

} // namespace excitation_rk4_sparse 