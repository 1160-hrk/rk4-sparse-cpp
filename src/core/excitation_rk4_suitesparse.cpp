#include "excitation_rk4_sparse/suitesparse.hpp"
#include "excitation_rk4_sparse/core.hpp"
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <chrono>
#include <limits>

// OpenBLAS + SuiteSparseのヘッダー
#ifdef OPENBLAS_SUITESPARSE_AVAILABLE
#include <cblas.h>
#include <suitesparse/cholmod.h>
#include <suitesparse/umfpack.h>
#endif

// SuiteSparse-MKLのヘッダー
#ifdef SUITESPARSE_MKL_AVAILABLE
#include <mkl.h>
#include <mkl_spblas.h>
#include <mkl_types.h>
#endif

namespace excitation_rk4_sparse {

static SuiteSparsePerformanceMetrics current_metrics;

// Phase 4: 適応的並列化閾値の計算関数（SuiteSparse版・8192次元対応・超厳格版）
inline int get_optimal_parallel_threshold_suitesparse() {
    #ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
    const int cache_line_size = 64;
    const int elements_per_cache_line = cache_line_size / sizeof(std::complex<double>);
    
    // 8192次元問題対応の超厳格な閾値設定：乗数を64から128に変更
    // 各スレッドが少なくとも128キャッシュライン分のデータを処理
    // 1024次元以下では実質的に並列化を無効化
    // 8192次元では完全に並列化を無効化
    return max_threads * elements_per_cache_line * 128;
    #else
    return std::numeric_limits<int>::max();  // 並列化しない
    #endif
}

// Phase 4: 最適化されたスパース行列-ベクトル積（SuiteSparse版）
inline void optimized_sparse_matrix_vector_multiply_suitesparse(
    const Eigen::SparseMatrix<std::complex<double>>& H,
    const Eigen::VectorXcd& x,
    Eigen::VectorXcd& y,
    int dim) {
    
    using cplx = std::complex<double>;
    
    #ifdef _OPENMP
    if (dim >= 8192) {
        // 8192次元以上：並列化を完全に無効化（シリアル実行）
        y = cplx(0, -1) * (H * x);
    } else if (dim > 4096) {
        // 大規模問題：列ベース並列化
        y.setZero();
        #pragma omp parallel for schedule(dynamic, 64)
        for (int k = 0; k < H.outerSize(); ++k) {
            for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(H, k); it; ++it) {
                y[it.row()] += it.value() * x[it.col()];
            }
        }
        y *= cplx(0, -1);
    } else {
        // 中規模問題：Eigenの最適化された実装を使用
        y = cplx(0, -1) * (H * x);
    }
    #else
    y = cplx(0, -1) * (H * x);
    #endif
}

// Phase 4: 適応的並列化戦略（SuiteSparse版・8192次元対応・超厳格版）
inline void adaptive_parallel_matrix_update_suitesparse(
    std::complex<double>* H_values,
    const std::complex<double>* H0_data,
    const std::complex<double>* mux_data,
    const std::complex<double>* muy_data,
    double ex, double ey, size_t nnz, int dim) {
    
    const int optimal_threshold = get_optimal_parallel_threshold_suitesparse();
    
    #ifdef _OPENMP
    if (dim >= 8192) {
        // 8192次元以上：並列化を完全に無効化（シリアル実行）
        for (size_t i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
        }
    } else if (dim >= 4096) {
        // 4096-8192次元：並列化を大幅に制限（極大規模問題のみ）
        if (nnz > optimal_threshold * 256) {
            const int chunk_size = std::max(2048, static_cast<int>(nnz) / (omp_get_max_threads() * 256));
            #pragma omp parallel for schedule(dynamic, chunk_size)
            for (size_t i = 0; i < nnz; ++i) {
                H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
            }
        } else {
            // シリアル実行
            for (size_t i = 0; i < nnz; ++i) {
                H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
            }
        }
    } else if (nnz > optimal_threshold * 256) {
        // 極大規模問題：動的スケジューリング
        const int chunk_size = std::max(2048, static_cast<int>(nnz) / (omp_get_max_threads() * 256));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (size_t i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 128) {
        // 超大規模問題：動的スケジューリング
        const int chunk_size = std::max(1024, static_cast<int>(nnz) / (omp_get_max_threads() * 128));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (size_t i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 64) {
        // 大規模問題：動的スケジューリング
        const int chunk_size = std::max(512, static_cast<int>(nnz) / (omp_get_max_threads() * 64));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (size_t i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 32) {
        // 中大規模問題：ガイド付きスケジューリング
        #pragma omp parallel for schedule(guided)
        for (size_t i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 16) {
        // 中規模問題：静的スケジューリング
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 8) {
        // 小中規模問題：小さなチャンクサイズでの静的スケジューリング
        const int chunk_size = std::max(1, static_cast<int>(nnz) / (omp_get_max_threads() * 8));
        #pragma omp parallel for schedule(static, chunk_size)
        for (size_t i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
        }
    } else {
        // 小規模問題：シリアル実行（並列化オーバーヘッドを回避）
        // 1024次元以下では実質的にシリアル実行
        for (size_t i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
        }
    }
    #else
    // OpenMPが利用できない場合のシリアル実行
    for (size_t i = 0; i < nnz; ++i) {
        H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
    }
    #endif
}

// Phase 4: 最適化された並列化戦略（SuiteSparse版・8192次元対応・超厳格版）
inline void optimized_parallel_matrix_update_suitesparse(
    std::complex<double>* H_values,
    const std::complex<double>* H0_data,
    const std::complex<double>* mux_data,
    const std::complex<double>* muy_data,
    double ex, double ey, size_t nnz) {
    
    const int optimal_threshold = get_optimal_parallel_threshold_suitesparse();
    
    #ifdef _OPENMP
    if (nnz > optimal_threshold * 256) {
        // 極大規模問題：動的スケジューリング（負荷分散最適化）
        const int chunk_size = std::max(2048, static_cast<int>(nnz) / (omp_get_max_threads() * 256));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (size_t i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 128) {
        // 超大規模問題：動的スケジューリング（負荷分散最適化）
        const int chunk_size = std::max(1024, static_cast<int>(nnz) / (omp_get_max_threads() * 128));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (size_t i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 64) {
        // 大規模問題：動的スケジューリング（負荷分散最適化）
        const int chunk_size = std::max(512, static_cast<int>(nnz) / (omp_get_max_threads() * 64));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (size_t i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 32) {
        // 中大規模問題：ガイド付きスケジューリング（適応的負荷分散）
        #pragma omp parallel for schedule(guided)
        for (size_t i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 16) {
        // 中規模問題：静的スケジューリング（低オーバーヘッド）
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 8) {
        // 小中規模問題：小さなチャンクサイズでの静的スケジューリング
        const int chunk_size = std::max(1, static_cast<int>(nnz) / (omp_get_max_threads() * 8));
        #pragma omp parallel for schedule(static, chunk_size)
        for (size_t i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
        }
    } else {
        // 小規模問題：シリアル実行（並列化オーバーヘッドを回避）
        // 1024次元以下では実質的にシリアル実行
        for (size_t i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
        }
    }
    #else
    // OpenMPが利用できない場合のシリアル実行
    for (size_t i = 0; i < nnz; ++i) {
        H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
    }
    #endif
}

// Phase 4: 最適化されたデータ展開関数（SuiteSparse版・8192次元対応・超厳格版）
inline std::vector<std::complex<double>> optimized_expand_to_pattern_suitesparse(
    const Eigen::SparseMatrix<std::complex<double>>& mat, 
    const Eigen::SparseMatrix<std::complex<double>>& pattern) {
    
    std::vector<std::complex<double>> result(pattern.nonZeros(), std::complex<double>(0.0, 0.0));
    
    // パターンの非ゼロ要素のインデックスを取得
    Eigen::VectorXi pi(pattern.nonZeros());
    Eigen::VectorXi pj(pattern.nonZeros());
    int idx = 0;
    for (int k = 0; k < pattern.outerSize(); ++k) {
        for (Eigen::SparseMatrix<std::complex<double>>::InnerIterator it(pattern, k); it; ++it) {
            pi[idx] = it.row();
            pj[idx] = it.col();
            idx++;
        }
    }

    // Phase 4: 適応的並列化によるデータ展開（SuiteSparse版・8192次元対応・超厳格版）
    const size_t nnz = pattern.nonZeros();
    const int optimal_threshold = get_optimal_parallel_threshold_suitesparse();
    
    #ifdef _OPENMP
    if (nnz > optimal_threshold * 128) {
        // 極大規模データ：動的スケジューリング
        const int chunk_size = std::max(1024, static_cast<int>(nnz) / (omp_get_max_threads() * 128));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (size_t i = 0; i < nnz; ++i) {
            result[i] = mat.coeff(pi[i], pj[i]);
        }
    } else if (nnz > optimal_threshold * 64) {
        // 超大規模データ：動的スケジューリング
        const int chunk_size = std::max(512, static_cast<int>(nnz) / (omp_get_max_threads() * 64));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (size_t i = 0; i < nnz; ++i) {
            result[i] = mat.coeff(pi[i], pj[i]);
        }
    } else if (nnz > optimal_threshold * 32) {
        // 大規模データ：動的スケジューリング
        const int chunk_size = std::max(256, static_cast<int>(nnz) / (omp_get_max_threads() * 32));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (size_t i = 0; i < nnz; ++i) {
            result[i] = mat.coeff(pi[i], pj[i]);
        }
    } else if (nnz > optimal_threshold * 16) {
        // 中大規模データ：動的スケジューリング
        const int chunk_size = std::max(128, static_cast<int>(nnz) / (omp_get_max_threads() * 16));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (size_t i = 0; i < nnz; ++i) {
            result[i] = mat.coeff(pi[i], pj[i]);
        }
    } else if (nnz > optimal_threshold * 8) {
        // 中規模データ：静的スケジューリング
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nnz; ++i) {
            result[i] = mat.coeff(pi[i], pj[i]);
        }
    } else if (nnz > optimal_threshold * 4) {
        // 小中規模データ：小さなチャンクサイズでの静的スケジューリング
        const int chunk_size = std::max(1, static_cast<int>(nnz) / (omp_get_max_threads() * 8));
        #pragma omp parallel for schedule(static, chunk_size)
        for (size_t i = 0; i < nnz; ++i) {
            result[i] = mat.coeff(pi[i], pj[i]);
        }
    } else {
        // 小規模データ：シリアル実行（並列化オーバーヘッドを回避）
        // 1024次元以下では実質的にシリアル実行
        for (size_t i = 0; i < nnz; ++i) {
            result[i] = mat.coeff(pi[i], pj[i]);
        }
    }
    #else
    for (size_t i = 0; i < nnz; ++i) {
        result[i] = mat.coeff(pi[i], pj[i]);
    }
    #endif
    
    return result;
}

// 旧来の3関数を統合した新しい関数（Phase 3: 並列化戦略の再設計）
// OptimizationLevelで分岐（現状は挙動同じだが将来拡張可能）
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
    OptimizationLevel /* level */)  // 未使用パラメータをコメント化
{
    using cplx = std::complex<double>;

    // メトリクスをリセット
    current_metrics = SuiteSparsePerformanceMetrics();

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

    // 1️⃣ 共通パターン（構造のみ）を作成（Eigen版と同じ）
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

    // 2️⃣ Phase 3: 最適化されたデータ展開
    const size_t nnz = pattern.nonZeros();
    alignas(CACHE_LINE) std::vector<cplx> H0_data = optimized_expand_to_pattern_suitesparse(H0, pattern);
    alignas(CACHE_LINE) std::vector<cplx> mux_data = optimized_expand_to_pattern_suitesparse(mux, pattern);
    alignas(CACHE_LINE) std::vector<cplx> muy_data = optimized_expand_to_pattern_suitesparse(muy, pattern);

    // 3️⃣ 計算用行列
    Eigen::SparseMatrix<cplx> H = pattern;

    #ifdef OPENBLAS_SUITESPARSE_AVAILABLE
    // OpenBLAS + SuiteSparse用の設定
    cholmod_common c;
    cholmod_start(&c);
    c.useGPU = 0;  // GPUは使用しない
    
    // 疎行列の並び替え用
    cholmod_sparse *cholmod_H = nullptr;
    #endif

    #ifdef SUITESPARSE_MKL_AVAILABLE
    // MKL Sparse BLAS用の行列記述子を準備
    sparse_matrix_t mkl_H = nullptr;
    sparse_index_base_t indexing = SPARSE_INDEX_BASE_ZERO;
    
    // MKL行列記述子を作成
    struct matrix_descr mkl_descr;
    mkl_descr.type = SPARSE_MATRIX_TYPE_GENERAL;
    mkl_descr.mode = SPARSE_FILL_MODE_FULL;
    mkl_descr.diag = SPARSE_DIAG_NON_UNIT;
    
    // Eigenの疎行列をMKL形式に変換するヘルパー関数
    auto eigen_to_mkl_sparse = [](const Eigen::SparseMatrix<cplx>& eigen_mat) -> sparse_matrix_t {
        sparse_matrix_t mkl_mat = nullptr;
        
        // Eigenのデータを取得
        const int rows = eigen_mat.rows();
        const int cols = eigen_mat.cols();
        const int nnz = eigen_mat.nonZeros();
        
        // 複素数データを実部・虚部に分離
        std::vector<double> real_data(nnz);
        std::vector<double> imag_data(nnz);
        
        for (int i = 0; i < nnz; ++i) {
            real_data[i] = eigen_mat.valuePtr()[i].real();
            imag_data[i] = eigen_mat.valuePtr()[i].imag();
        }
        
        // MKL Sparse BLAS行列を作成
        sparse_status_t status = mkl_sparse_z_create_csr(
            &mkl_mat,
            SPARSE_INDEX_BASE_ZERO,
            rows,
            cols,
            const_cast<int*>(eigen_mat.outerIndexPtr()),
            const_cast<int*>(eigen_mat.innerIndexPtr()),
            real_data.data(),
            imag_data.data()
        );
        
        if (status != SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("Failed to create MKL sparse matrix");
        }
        
        return mkl_mat;
    };
    #endif

    // 電場データを3点セットに変換
    auto Ex3 = field_to_triplets(Ex);
    auto Ey3 = field_to_triplets(Ey);

    // メインループ
    for (int s = 0; s < steps; ++s) {
        double ex1 = Ex3[s][0], ex2 = Ex3[s][1], ex4 = Ex3[s][2];
        double ey1 = Ey3[s][0], ey2 = Ey3[s][1], ey4 = Ey3[s][2];

        // H1 - Phase 4: 最適化された行列更新（8192次元対応）
        adaptive_parallel_matrix_update_suitesparse(H.valuePtr(), H0_data.data(), mux_data.data(), muy_data.data(), ex1, ey1, nnz, dim);

        // RK4ステップ
        optimized_sparse_matrix_vector_multiply_suitesparse(H, psi, k1, dim);
        
        buf = psi + 0.5 * dt * k1;

        // H2 - Phase 4: 最適化された行列更新（8192次元対応）
        adaptive_parallel_matrix_update_suitesparse(H.valuePtr(), H0_data.data(), mux_data.data(), muy_data.data(), ex2, ey2, nnz, dim);
        optimized_sparse_matrix_vector_multiply_suitesparse(H, buf, k2, dim);
        
        buf = psi + 0.5 * dt * k2;

        // H3 - Phase 4: 最適化されたスパース行列-ベクトル積
        optimized_sparse_matrix_vector_multiply_suitesparse(H, buf, k3, dim);
        buf = psi + dt * k3;

        // H4 - Phase 4: 最適化された行列更新（8192次元対応）
        adaptive_parallel_matrix_update_suitesparse(H.valuePtr(), H0_data.data(), mux_data.data(), muy_data.data(), ex4, ey4, nnz, dim);
        optimized_sparse_matrix_vector_multiply_suitesparse(H, buf, k4, dim);

        // 更新
        psi += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);

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

    #ifdef OPENBLAS_SUITESPARSE_AVAILABLE
    // SuiteSparseのクリーンアップ
    if (cholmod_H != nullptr) {
        cholmod_free_sparse(&cholmod_H, &c);
    }
    cholmod_finish(&c);
    #endif

    #ifdef SUITESPARSE_MKL_AVAILABLE
    // MKL行列のクリーンアップ
    if (mkl_H != nullptr) {
        mkl_sparse_destroy(mkl_H);
    }
    #endif

    // パフォーマンスメトリクスを出力（デバッグ用）
    #ifdef DEBUG_PERFORMANCE
    #ifdef OPENBLAS_SUITESPARSE_AVAILABLE
    std::cout << "\n=== OpenBLAS + SuiteSparse版パフォーマンスメトリクス ===\n";
    #elif defined(SUITESPARSE_MKL_AVAILABLE)
    std::cout << "\n=== SuiteSparse-MKL版パフォーマンスメトリクス ===\n";
    #else
    std::cout << "\n=== Eigen版パフォーマンスメトリクス ===\n";
    #endif
    std::cout << "行列更新平均時間: " << current_metrics.matrix_update_time / current_metrics.matrix_updates * 1000 << " ms\n";
    std::cout << "RK4ステップ平均時間: " << current_metrics.rk4_step_time / current_metrics.rk4_steps * 1000 << " ms\n";
    std::cout << "疎行列演算平均時間: " << current_metrics.sparse_solve_time / current_metrics.sparse_solves * 1000 << " ms\n";
    #endif

    return out;
}

} // namespace excitation_rk4_sparse 