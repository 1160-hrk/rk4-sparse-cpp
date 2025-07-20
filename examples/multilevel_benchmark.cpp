#include <iostream>
#include <vector>
#include <memory>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <cmath>
#include <Eigen/Dense>
#include <Eigen/Sparse>

// プロジェクトのヘッダーファイルをインクルード
#include "excitation_rk4_sparse/core.hpp"
#ifdef HAVE_SUITESPARSE
#include "excitation_rk4_sparse/suitesparse.hpp"
#endif
#include "excitation_rk4_sparse/benchmark.hpp"

namespace rk4_benchmark {

using Complex = std::complex<double>;
using SparseMatrix = Eigen::SparseMatrix<Complex>;
using DenseMatrix = Eigen::Matrix<Complex, Eigen::Dynamic, Eigen::Dynamic>;
using Vector = Eigen::VectorXcd;
using RealVector = Eigen::VectorXd;

// ベンチマーク結果を格納する構造体
struct BenchmarkResult {
    std::string method_name;
    double mean_time;
    double min_time;
    double max_time;
    double std_time;
    size_t memory_usage; // 推定メモリ使用量（バイト）
    size_t allocations; // アロケーション回数（推定）
    double speedup_ratio;
};

// 多準位系のハミルトニアンと双極子行列を生成
std::tuple<SparseMatrix, SparseMatrix, SparseMatrix> 
create_multilevel_system(int dim, double hbar = 1.0, double omega = 1.0, double mu = 0.1) {
    std::cout << "Creating " << dim << "-level system..." << std::flush;
    
    // ハミルトニアン（対角成分）
    std::vector<Eigen::Triplet<Complex>> H0_triplets;
    for (int i = 0; i < dim; ++i) {
        H0_triplets.emplace_back(i, i, i * hbar * omega);
    }
    
    SparseMatrix H0(dim, dim);
    H0.setFromTriplets(H0_triplets.begin(), H0_triplets.end());
    
    // 双極子行列（最近接遷移のみ）
    std::vector<Eigen::Triplet<Complex>> mux_triplets;
    for (int i = 0; i < dim - 1; ++i) {
        double coupling = mu * std::sqrt(i + 1);
        mux_triplets.emplace_back(i, i + 1, coupling);     // 上向き遷移
        mux_triplets.emplace_back(i + 1, i, coupling);     // 下向き遷移
    }
    
    SparseMatrix mux(dim, dim);
    mux.setFromTriplets(mux_triplets.begin(), mux_triplets.end());
    
    // y成分なし
    SparseMatrix muy(dim, dim);
    
    std::cout << " ✓" << std::endl;
    return std::make_tuple(H0, mux, muy);
}

// 密行列版のハミルトニアンと双極子行列を生成
std::tuple<DenseMatrix, DenseMatrix, DenseMatrix> 
create_multilevel_system_dense(int dim, double hbar = 1.0, double omega = 1.0, double mu = 0.1) {
    auto [H0_sparse, mux_sparse, muy_sparse] = create_multilevel_system(dim, hbar, omega, mu);
    
    DenseMatrix H0 = DenseMatrix(H0_sparse);
    DenseMatrix mux = DenseMatrix(mux_sparse);  
    DenseMatrix muy = DenseMatrix(muy_sparse);
    
    return std::make_tuple(H0, mux, muy);
}

// 電場を生成（Juliaと同じ形式）
std::pair<RealVector, RealVector> create_electric_field(double T, double dt, double E0, double omega_L) {
    // Juliaと同じように全ステップ分（奇数個）を生成
    int total_steps = static_cast<int>(T / dt) + 1;
    // steps = 1001のように奇数にして、RK4では(steps-1)/2を使用
    if (total_steps % 2 == 0) {
        total_steps += 1;
    }
    
    RealVector t = RealVector::LinSpaced(total_steps, 0.0, T);
    
    RealVector Ex(total_steps);
    RealVector Ey = RealVector::Zero(total_steps);
    
    for (int i = 0; i < total_steps; ++i) {
        Ex[i] = E0 * std::sin(omega_L * t[i]);
    }
    
    return std::make_pair(Ex, Ey);
}

// 初期状態を生成（基底状態）
Vector create_initial_state(int dim) {
    Vector psi0 = Vector::Zero(dim);
    psi0[0] = 1.0;
    return psi0;
}

// 密行列版のRK4実装（Juliaスタイルに合わせて修正）
Eigen::MatrixXcd rk4_dense(
    const DenseMatrix& H0,
    const DenseMatrix& mux,
    const DenseMatrix& muy,
    const RealVector& Ex,
    const RealVector& Ey,
    const Vector& psi0,
    double dt,
    bool return_traj = true,
    int stride = 1,
    bool renorm = false
) {
    // Juliaと同じステップ数計算
    const int steps = (Ex.size() - 1) / 2;
    const int dim = psi0.size();
    const int n_out = return_traj ? (steps + stride - 1) / stride : 1;
    
    Eigen::MatrixXcd out(n_out, dim);
    if (return_traj) {
        out.row(0) = psi0.transpose();
    }
    
    Vector psi = psi0;
    Vector buf(dim), k1(dim), k2(dim), k3(dim), k4(dim);
    Complex minus_i(0, -1);
    
    int idx = 1;
    for (int s = 0; s < steps; ++s) {
        // Juliaのように電場の3点を使用
        double ex1 = Ex[2*s];
        double ex2 = Ex[2*s + 1];
        double ex4 = Ex[2*s + 2];
        double ey1 = Ey[2*s];
        double ey2 = Ey[2*s + 1];
        double ey4 = Ey[2*s + 2];
        
        // H1 = H0 + ex1 * mux + ey1 * muy
        DenseMatrix H1 = H0 + ex1 * mux + ey1 * muy;
        k1 = minus_i * (H1 * psi);
        buf = psi + 0.5 * dt * k1;
        
        // H2 = H0 + ex2 * mux + ey2 * muy  
        DenseMatrix H2 = H0 + ex2 * mux + ey2 * muy;
        k2 = minus_i * (H2 * buf);
        buf = psi + 0.5 * dt * k2;
        
        // k3はH2を再利用
        k3 = minus_i * (H2 * buf);
        buf = psi + dt * k3;
        
        // H4 = H0 + ex4 * mux + ey4 * muy
        DenseMatrix H4 = H0 + ex4 * mux + ey4 * muy;
        k4 = minus_i * (H4 * buf);
        
        psi += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        
        if (renorm) {
            psi.normalize();
        }
        
        if (return_traj && (s % stride == 0)) {
            out.row(idx) = psi.transpose();
            idx++;
        }
    }
    
    if (!return_traj) {
        out.row(0) = psi.transpose();
    }
    
    return out;
}

// ベンチマーク実行関数
BenchmarkResult run_benchmark(
    const std::string& method_name,
    std::function<void()> method_func,
    int num_runs = 5
) {
    std::vector<double> times;
    times.reserve(num_runs);
    
    // ウォームアップ
    method_func();
    
    // 実際のベンチマーク
    for (int run = 0; run < num_runs; ++run) {
        auto start = std::chrono::high_resolution_clock::now();
        method_func();
        auto end = std::chrono::high_resolution_clock::now();
        
        auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
        times.push_back(duration.count() / 1000.0); // ms
    }
    
    BenchmarkResult result;
    result.method_name = method_name;
    result.mean_time = std::accumulate(times.begin(), times.end(), 0.0) / times.size();
    result.min_time = *std::min_element(times.begin(), times.end());
    result.max_time = *std::max_element(times.begin(), times.end());
    
    // 標準偏差の計算
    double variance = 0.0;
    for (double time : times) {
        variance += std::pow(time - result.mean_time, 2);
    }
    result.std_time = std::sqrt(variance / times.size());
    
    // メモリ使用量とアロケーション回数は推定値
    result.memory_usage = 0;
    result.allocations = 0;
    result.speedup_ratio = 1.0;
    
    return result;
}

// 単一次元でのベンチマーク実行
std::vector<BenchmarkResult> benchmark_single_dimension(int dim) {
    std::cout << "\n次元: " << dim << std::endl;
    std::cout << std::string(30, '-') << std::endl;
    
    // システム設定
    const double T = 10.0;
    const double dt = 0.01;
    const double E0 = 1.0;
    const double omega_L = 1.0;
    const int stride = 1;
    const bool renorm = false;
    
    // 電場生成
    auto [Ex, Ey] = create_electric_field(T, dt, E0, omega_L);
    Vector psi0 = create_initial_state(dim);
    
    std::vector<BenchmarkResult> results;
    
    // 一時的に密行列実装を無効化（メモリエラー対策）
    /*
    if (dim <= 64) {
        try {
            // Dense Matrix実装
            auto [H0_dense, mux_dense, muy_dense] = create_multilevel_system_dense(dim);
            auto dense_func = [&]() {
                rk4_dense(H0_dense, mux_dense, muy_dense, Ex, Ey, psi0, dt, true, stride, renorm);
            };
            
            std::cout << "Dense Matrix実装をテスト中..." << std::flush;
            auto dense_result = run_benchmark("Dense Matrix", dense_func);
            std::cout << " 平均時間: " << std::fixed << std::setprecision(3) 
                      << dense_result.mean_time << " ms" << std::endl;
            results.push_back(dense_result);
        } catch (const std::exception& e) {
            std::cout << " エラー: " << e.what() << std::endl;
        }
    }
    */
    
    try {
        // Sparse Matrix実装 (Eigen標準)
        auto [H0_sparse, mux_sparse, muy_sparse] = create_multilevel_system(dim);
        auto sparse_func = [&]() {
            excitation_rk4_sparse::rk4_sparse_eigen(H0_sparse, mux_sparse, muy_sparse, 
                                                   Ex, Ey, psi0, dt, true, stride, renorm);
        };
        
        std::cout << "Sparse Matrix実装をテスト中..." << std::flush;
        auto sparse_result = run_benchmark("Sparse Matrix", sparse_func);
        std::cout << " 平均時間: " << std::fixed << std::setprecision(3) 
                  << sparse_result.mean_time << " ms" << std::endl;
        results.push_back(sparse_result);
    } catch (const std::exception& e) {
        std::cout << " エラー: " << e.what() << std::endl;
    }
    
    // Julia Style実装 - 深刻なメモリ管理問題のため無効化
    // 詳細な調査と根本的な設計見直しが必要
    /*
    try {
        // Julia Style実装（修正版）
        auto [H0_sparse, mux_sparse, muy_sparse] = create_multilevel_system(dim);
        auto julia_func = [&]() {
            excitation_rk4_sparse::rk4_sparse_julia_style(H0_sparse, mux_sparse, muy_sparse, 
                                                         Ex, Ey, psi0, dt, true, stride, renorm);
        };
        
        std::cout << "Julia Style実装をテスト中..." << std::flush;
        auto julia_result = run_benchmark("Julia Style", julia_func);
        std::cout << " 平均時間: " << std::fixed << std::setprecision(3) 
                  << julia_result.mean_time << " ms" << std::endl;
        results.push_back(julia_result);
    } catch (const std::exception& e) {
        std::cout << " エラー: " << e.what() << std::endl;
    }
    */
    
    // CSR Optimized実装 - メモリ管理問題のため無効化
    /*
    try {
        // CSR Optimized実装（修正版）
        auto [H0_sparse, mux_sparse, muy_sparse] = create_multilevel_system(dim);
        auto csr_func = [&]() {
            excitation_rk4_sparse::rk4_sparse_csr_optimized(H0_sparse, mux_sparse, muy_sparse, 
                                                           Ex, Ey, psi0, dt, true, stride, renorm);
        };
        
        std::cout << "CSR Optimized実装をテスト中..." << std::flush;
        auto csr_result = run_benchmark("CSR Optimized", csr_func);
        std::cout << " 平均時間: " << std::fixed << std::setprecision(3) 
                  << csr_result.mean_time << " ms" << std::endl;
        results.push_back(csr_result);
    } catch (const std::exception& e) {
        std::cout << " エラー: " << e.what() << std::endl;
    }
    */
    
    try {
        // Eigen Cached実装
        auto [H0_sparse, mux_sparse, muy_sparse] = create_multilevel_system(dim);
        auto cached_func = [&]() {
            excitation_rk4_sparse::rk4_sparse_eigen_cached(H0_sparse, mux_sparse, muy_sparse, 
                                                          Ex, Ey, psi0, dt, true, stride, renorm);
        };
        
        std::cout << "Eigen Cached実装をテスト中..." << std::flush;
        auto cached_result = run_benchmark("Eigen Cached", cached_func);
        std::cout << " 平均時間: " << std::fixed << std::setprecision(3) 
                  << cached_result.mean_time << " ms" << std::endl;
        results.push_back(cached_result);
    } catch (const std::exception& e) {
        std::cout << " エラー: " << e.what() << std::endl;
    }
    
    // 一時的にSIMD実装を無効化（メモリアロケーション問題対策）
    /*
    if (dim >= 100 && dim <= 1000) {
        try {
            auto [H0_sparse, mux_sparse, muy_sparse] = create_multilevel_system(dim);
            auto simd_func = [&]() {
                excitation_rk4_sparse::rk4_sparse_medium_scale_optimized(H0_sparse, mux_sparse, muy_sparse, 
                                                                        Ex, Ey, psi0, dt, true, stride, renorm);
            };
            
            std::cout << "Medium Scale SIMD実装をテスト中..." << std::flush;
            auto simd_result = run_benchmark("Medium Scale SIMD", simd_func);
            std::cout << " 平均時間: " << std::fixed << std::setprecision(3) 
                      << simd_result.mean_time << " ms" << std::endl;
            results.push_back(simd_result);
        } catch (const std::exception& e) {
            std::cout << " エラー: " << e.what() << std::endl;
        }
    }
    */
    
    // 一時的にBLAS実装を無効化（セグメンテーションフォルト対策）
    /*
    if (dim <= 32) {
        try {
            auto [H0_sparse, mux_sparse, muy_sparse] = create_multilevel_system(dim);
            auto blas_safe_func = [&]() {
                excitation_rk4_sparse::rk4_sparse_blas_optimized_safe(H0_sparse, mux_sparse, muy_sparse, 
                                                                     Ex, Ey, psi0, dt, true, stride, renorm);
            };
            
            std::cout << "BLAS Safe実装をテスト中..." << std::flush;
            auto blas_safe_result = run_benchmark("BLAS Safe", blas_safe_func);
            std::cout << " 平均時間: " << std::fixed << std::setprecision(3) 
                      << blas_safe_result.mean_time << " ms" << std::endl;
            results.push_back(blas_safe_result);
        } catch (const std::exception& e) {
            std::cout << " エラー: " << e.what() << std::endl;
        }
        
        try {
            auto [H0_sparse, mux_sparse, muy_sparse] = create_multilevel_system(dim);
            auto blas_func = [&]() {
                excitation_rk4_sparse::rk4_sparse_blas_optimized(H0_sparse, mux_sparse, muy_sparse, 
                                                                Ex, Ey, psi0, dt, true, stride, renorm);
            };
            
            std::cout << "BLAS Standard実装をテスト中..." << std::flush;
            auto blas_result = run_benchmark("BLAS Standard", blas_func);
            std::cout << " 平均時間: " << std::fixed << std::setprecision(3) 
                      << blas_result.mean_time << " ms" << std::endl;
            results.push_back(blas_result);
        } catch (const std::exception& e) {
            std::cout << " エラー: " << e.what() << std::endl;
        }
    }
    */
    

    
    // スピードアップ比を計算（最初の実装を基準）
    if (!results.empty()) {
        double baseline_time = results[0].mean_time;
        for (auto& result : results) {
            result.speedup_ratio = baseline_time / result.mean_time;
        }
    }
    
    return results;
}

// CSVファイルに結果を保存
void save_results_to_csv(const std::map<int, std::vector<BenchmarkResult>>& all_results, 
                        const std::string& filename) {
    std::ofstream file(filename);
    file << "Dimension,Method,Mean_Time_ms,Min_Time_ms,Max_Time_ms,Std_Time_ms,Speedup_Ratio\n";
    
    for (const auto& [dim, results] : all_results) {
        for (const auto& result : results) {
            file << dim << ","
                 << result.method_name << ","
                 << std::fixed << std::setprecision(6) << result.mean_time << ","
                 << result.min_time << ","
                 << result.max_time << ","
                 << result.std_time << ","
                 << result.speedup_ratio << "\n";
        }
    }
    
    std::cout << "結果を " << filename << " に保存しました。" << std::endl;
}

// 結果の詳細レポートを出力
void print_detailed_report(const std::map<int, std::vector<BenchmarkResult>>& all_results) {
    std::cout << "\n詳細な結果" << std::endl;
    std::cout << std::string(120, '=') << std::endl;
    
    std::cout << std::left << std::setw(8) << "次元"
              << std::setw(20) << "手法"
              << std::setw(15) << "平均時間(ms)"
              << std::setw(15) << "最小時間(ms)"
              << std::setw(15) << "最大時間(ms)"
              << std::setw(15) << "標準偏差(ms)"
              << std::setw(12) << "高速化比" << std::endl;
    std::cout << std::string(120, '-') << std::endl;
    
    for (const auto& [dim, results] : all_results) {
        for (const auto& result : results) {
            std::cout << std::left << std::setw(8) << dim
                      << std::setw(20) << result.method_name
                      << std::setw(15) << std::fixed << std::setprecision(3) << result.mean_time
                      << std::setw(15) << result.min_time
                      << std::setw(15) << result.max_time
                      << std::setw(15) << result.std_time
                      << std::setw(12) << std::setprecision(2) << result.speedup_ratio << std::endl;
        }
    }
}

// スケーリング解析
void analyze_scaling(const std::map<int, std::vector<BenchmarkResult>>& all_results) {
    std::cout << "\nスケーリング解析" << std::endl;
    std::cout << std::string(50, '-') << std::endl;
    
    // 各手法のスケーリングを分析
    std::map<std::string, std::vector<std::pair<int, double>>> method_data;
    
    for (const auto& [dim, results] : all_results) {
        for (const auto& result : results) {
            method_data[result.method_name].emplace_back(dim, result.mean_time);
        }
    }
    
    for (const auto& [method, data] : method_data) {
        if (data.size() >= 2) {
            // 簡単な対数線形回帰でスケーリング指数を推定
            double sum_log_dim = 0, sum_log_time = 0, sum_log_dim_time = 0, sum_log_dim_sq = 0;
            int n = data.size();
            
            for (const auto& [dim, time] : data) {
                double log_dim = std::log10(dim);
                double log_time = std::log10(time);
                sum_log_dim += log_dim;
                sum_log_time += log_time;
                sum_log_dim_time += log_dim * log_time;
                sum_log_dim_sq += log_dim * log_dim;
            }
            
            double slope = (n * sum_log_dim_time - sum_log_dim * sum_log_time) / 
                          (n * sum_log_dim_sq - sum_log_dim * sum_log_dim);
            double intercept = (sum_log_time - slope * sum_log_dim) / n;
            
            std::cout << method << ":" << std::endl;
            std::cout << "  実行時間スケーリング: O(dim^" << std::fixed << std::setprecision(2) 
                      << slope << ")" << std::endl;
            std::cout << "  フィッティング式: time = " << std::setprecision(3) 
                      << std::pow(10, intercept) << " × dim^" << std::setprecision(2) 
                      << slope << std::endl << std::endl;
        }
    }
}

} // namespace rk4_benchmark

int main() {
    using namespace rk4_benchmark;
    
    std::cout << "多準位系ベンチマーク開始" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    
    // テストする次元（安全な実装のみなので全次元テスト）
    std::vector<int> dimensions = {2, 4, 8, 16, 32, 64, 128, 256, 512, 1024};
    
    std::map<int, std::vector<BenchmarkResult>> all_results;
    
    for (int dim : dimensions) {
        try {
            auto results = benchmark_single_dimension(dim);
            if (!results.empty()) {
                all_results[dim] = results;
            }
        } catch (const std::exception& e) {
            std::cout << "次元 " << dim << " でエラー: " << e.what() << std::endl;
        }
    }
    
    // 結果の可視化と保存
    std::cout << "\n結果の可視化" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    
    // 詳細レポート
    print_detailed_report(all_results);
    
    // スケーリング解析
    analyze_scaling(all_results);
    
    // CSVファイルに保存
    save_results_to_csv(all_results, "multilevel_benchmark_results.csv");
    
    // システム情報
    std::cout << "\nシステム情報" << std::endl;
    std::cout << std::string(50, '=') << std::endl;
    #ifdef _OPENMP
    std::cout << "OpenMP利用可能: はい" << std::endl;
    std::cout << "最大スレッド数: " << omp_get_max_threads() << std::endl;
    #else
    std::cout << "OpenMP利用可能: いいえ" << std::endl;
    #endif
    
    std::cout << "\nベンチマーク完了！" << std::endl;
    
    return 0;
} 