#include "excitation_rk4_sparse/core.hpp"
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <chrono>

namespace excitation_rk4_sparse {

static PerformanceMetrics current_metrics;

// field_to_triplets の実装
std::vector<std::vector<double>> field_to_triplets(const Eigen::VectorXd& field) {
    const int steps = (field.size() - 1) / 2;
    std::vector<std::vector<double>> triplets;
    triplets.reserve(steps);
    
    for (int i = 0; i < steps; ++i) {
        std::vector<double> triplet = {
            field[2*i],      // ex1
            field[2*i + 1],  // ex2
            field[2*i + 2]   // ex4
        };
        triplets.push_back(triplet);
    }
    
    return triplets;
}

// rk4_cpu_sparse実装
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
    bool renorm)
{
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double>;
    using cplx = std::complex<double>;

    // メトリクスをリセット
    current_metrics = PerformanceMetrics();

    #ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
    omp_set_num_threads(max_threads);
    #endif

    const int steps = (Ex.size() - 1) / 2;
    const int dim = psi0.size();
    const int n_out = steps / stride + 1;

    // メモリアライメントとキャッシュライン境界を考慮
    constexpr size_t CACHE_LINE = 64;

    // 出力行列の準備
    Eigen::MatrixXcd out;
    if (return_traj) {
        out.resize(n_out, dim);
        out.row(0) = psi0;
    } else {
        out.resize(1, dim);
    }

    // メモリアライメントを最適化
    alignas(CACHE_LINE) Eigen::VectorXcd psi = psi0;
    alignas(CACHE_LINE) Eigen::VectorXcd buf(dim);
    alignas(CACHE_LINE) Eigen::VectorXcd k1(dim);
    alignas(CACHE_LINE) Eigen::VectorXcd k2(dim);
    alignas(CACHE_LINE) Eigen::VectorXcd k3(dim);
    alignas(CACHE_LINE) Eigen::VectorXcd k4(dim);

    int idx = 1;

    // 1️⃣ 共通パターン（構造のみ）を作成
    const double threshold = 1e-12;
    Eigen::SparseMatrix<cplx> pattern = H0;
    pattern.setZero();
    
    // 非ゼロパターンを構築
    for (int k = 0; k < H0.outerSize(); ++k) {
        for (Eigen::SparseMatrix<cplx>::InnerIterator it(H0, k); it; ++it) {
            if (std::abs(it.value()) > threshold) {
                pattern.coeffRef(it.row(), it.col()) = cplx(1.0, 0.0);
            }
        }
    }
    
    for (int k = 0; k < mux.outerSize(); ++k) {
        for (Eigen::SparseMatrix<cplx>::InnerIterator it(mux, k); it; ++it) {
            if (std::abs(it.value()) > threshold) {
                pattern.coeffRef(it.row(), it.col()) = cplx(1.0, 0.0);
            }
        }
    }
    
    for (int k = 0; k < muy.outerSize(); ++k) {
        for (Eigen::SparseMatrix<cplx>::InnerIterator it(muy, k); it; ++it) {
            if (std::abs(it.value()) > threshold) {
                pattern.coeffRef(it.row(), it.col()) = cplx(1.0, 0.0);
            }
        }
    }
    pattern.makeCompressed();

    // 2️⃣ パターンに合わせてデータを展開
    auto expand_to_pattern = [](const Eigen::SparseMatrix<cplx>& mat, 
                               const Eigen::SparseMatrix<cplx>& pattern) -> std::vector<cplx> {
        std::vector<cplx> result(pattern.nonZeros(), cplx(0.0, 0.0));
        
        // パターンの非ゼロ要素のインデックスを取得
        Eigen::VectorXi pi(pattern.nonZeros());
        Eigen::VectorXi pj(pattern.nonZeros());
        int idx = 0;
        for (int k = 0; k < pattern.outerSize(); ++k) {
            for (Eigen::SparseMatrix<cplx>::InnerIterator it(pattern, k); it; ++it) {
                pi[idx] = it.row();
                pj[idx] = it.col();
                idx++;
            }
        }

        // データを展開（大きな行列の場合のみ並列化）
        if (pattern.nonZeros() > 10000) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < pattern.nonZeros(); ++i) {
                result[i] = mat.coeff(pi[i], pj[i]);
            }
        } else {
            for (int i = 0; i < pattern.nonZeros(); ++i) {
                result[i] = mat.coeff(pi[i], pj[i]);
            }
        }
        
        return result;
    };

    const size_t nnz = pattern.nonZeros();
    alignas(CACHE_LINE) std::vector<cplx> H0_data = expand_to_pattern(H0, pattern);
    alignas(CACHE_LINE) std::vector<cplx> mux_data = expand_to_pattern(mux, pattern);
    alignas(CACHE_LINE) std::vector<cplx> muy_data = expand_to_pattern(muy, pattern);

    // 3️⃣ 計算用行列
    Eigen::SparseMatrix<cplx> H = pattern;

    // 電場データを3点セットに変換
    auto Ex3 = field_to_triplets(Ex);
    auto Ey3 = field_to_triplets(Ey);

    // メインループ
    for (int s = 0; s < steps; ++s) {
        double ex1 = Ex3[s][0], ex2 = Ex3[s][1], ex4 = Ex3[s][2];
        double ey1 = Ey3[s][0], ey2 = Ey3[s][1], ey4 = Ey3[s][2];

        // H1
        #ifdef DEBUG_PERFORMANCE
        auto update_start = Clock::now();
        #endif
        if (nnz > 10000) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nnz; ++i) {
                H.valuePtr()[i] = H0_data[i] + ex1 * mux_data[i] + ey1 * muy_data[i];
            }
        } else {
            for (int i = 0; i < nnz; ++i) {
                H.valuePtr()[i] = H0_data[i] + ex1 * mux_data[i] + ey1 * muy_data[i];
            }
        }
        #ifdef DEBUG_PERFORMANCE
        auto update_end = Clock::now();
        current_metrics.matrix_update_time += Duration(update_end - update_start).count();
        current_metrics.matrix_updates++;
        #endif

        // RK4ステップの時間を計測
        #ifdef DEBUG_PERFORMANCE
        auto rk4_start = Clock::now();
        #endif
        k1 = cplx(0, -1) * (H * psi);
        buf = psi + 0.5 * dt * k1;

        // H2
        if (nnz > 10000) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nnz; ++i) {
                H.valuePtr()[i] = H0_data[i] + ex2 * mux_data[i] + ey2 * muy_data[i];
            }
        } else {
            for (int i = 0; i < nnz; ++i) {
                H.valuePtr()[i] = H0_data[i] + ex2 * mux_data[i] + ey2 * muy_data[i];
            }
        }
        k2 = cplx(0, -1) * (H * buf);
        buf = psi + 0.5 * dt * k2;

        // H3
        k3 = cplx(0, -1) * (H * buf);
        buf = psi + dt * k3;

        // H4
        if (nnz > 10000) {
            #pragma omp parallel for schedule(static)
            for (int i = 0; i < nnz; ++i) {
                H.valuePtr()[i] = H0_data[i] + ex4 * mux_data[i] + ey4 * muy_data[i];
            }
        } else {
            for (int i = 0; i < nnz; ++i) {
                H.valuePtr()[i] = H0_data[i] + ex4 * mux_data[i] + ey4 * muy_data[i];
            }
        }
        k4 = cplx(0, -1) * (H * buf);

        // 更新
        psi += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        #ifdef DEBUG_PERFORMANCE
        auto rk4_end = Clock::now();
        current_metrics.rk4_step_time += Duration(rk4_end - rk4_start).count();
        current_metrics.rk4_steps++;
        #endif

        if (renorm) {
            cplx norm_complex = psi.adjoint() * psi;
            double norm = std::sqrt(std::abs(norm_complex));
            if (norm > 1e-10) {
                psi /= norm;
            }
        }

        if (return_traj && (s + 1) % stride == 0) {
            out.row(idx) = psi;
            idx++;
        }
    }

    if (!return_traj) {
        out.row(0) = psi;
    }

    // パフォーマンスメトリクスを出力（デバッグ用）
    #ifdef DEBUG_PERFORMANCE
    std::cout << "\n=== パフォーマンスメトリクス ===\n";
    std::cout << "行列更新平均時間: " << current_metrics.matrix_update_time / current_metrics.matrix_updates * 1000 << " ms\n";
    std::cout << "RK4ステップ平均時間: " << current_metrics.rk4_step_time / current_metrics.rk4_steps * 1000 << " ms\n";
    #endif

    return out;
}

} // namespace excitation_rk4_sparse
