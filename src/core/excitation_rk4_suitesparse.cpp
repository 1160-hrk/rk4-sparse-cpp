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

// Phase 3: 適応的並列化閾値の計算関数（SuiteSparse版・極限厳格版）
inline int get_optimal_parallel_threshold_suitesparse() {
    #ifdef _OPENMP
    const int max_threads = omp_get_max_threads();
    const int cache_line_size = 64;
    const int elements_per_cache_line = cache_line_size / sizeof(std::complex<double>);
    
    // 極限厳格な閾値設定：乗数を16から64に変更
    // 各スレッドが少なくとも64キャッシュライン分のデータを処理
    // 512次元以下では実質的に並列化を無効化
    return max_threads * elements_per_cache_line * 64;
    #else
    return std::numeric_limits<int>::max();  // 並列化しない
    #endif
}

// Phase 3: 最適化された並列化戦略（SuiteSparse版・超厳格版）
inline void optimized_parallel_matrix_update_suitesparse(
    std::complex<double>* H_values,
    const std::complex<double>* H0_data,
    const std::complex<double>* mux_data,
    const std::complex<double>* muy_data,
    double ex, double ey, size_t nnz) {
    
    const int optimal_threshold = get_optimal_parallel_threshold_suitesparse();
    
    #ifdef _OPENMP
    if (nnz > optimal_threshold * 128) {
        // 極大規模問題：動的スケジューリング（負荷分散最適化）
        const int chunk_size = std::max(1024, static_cast<int>(nnz) / (omp_get_max_threads() * 128));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (size_t i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 64) {
        // 超大規模問題：動的スケジューリング（負荷分散最適化）
        const int chunk_size = std::max(512, static_cast<int>(nnz) / (omp_get_max_threads() * 64));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (size_t i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 32) {
        // 大規模問題：動的スケジューリング（負荷分散最適化）
        const int chunk_size = std::max(256, static_cast<int>(nnz) / (omp_get_max_threads() * 32));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (size_t i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 16) {
        // 中大規模問題：ガイド付きスケジューリング（適応的負荷分散）
        #pragma omp parallel for schedule(guided)
        for (size_t i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 8) {
        // 中規模問題：静的スケジューリング（低オーバーヘッド）
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 4) {
        // 小中規模問題：小さなチャンクサイズでの静的スケジューリング
        const int chunk_size = std::max(1, static_cast<int>(nnz) / (omp_get_max_threads() * 4));
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

// Phase 3: 最適化されたデータ展開関数（SuiteSparse版・超厳格版）
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

    // Phase 3: 適応的並列化によるデータ展開（超厳格版）
    const size_t nnz = pattern.nonZeros();
    const int optimal_threshold = get_optimal_parallel_threshold_suitesparse();
    
    #ifdef _OPENMP
    if (nnz > optimal_threshold * 64) {
        // 極大規模データ：動的スケジューリング
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
        const int chunk_size = std::max(1, static_cast<int>(nnz) / (omp_get_max_threads() * 4));
        #pragma omp parallel for schedule(static, chunk_size)
        for (size_t i = 0; i < nnz; ++i) {
            result[i] = mat.coeff(pi[i], pj[i]);
        }
    } else {
        // 小規模データ：シリアル実行
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
    OptimizationLevel level
) {
    using Clock = std::chrono::high_resolution_clock;
    using Duration = std::chrono::duration<double>;
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

        // H1
        #ifdef DEBUG_PERFORMANCE
        auto update_start = Clock::now();
        #endif
        optimized_parallel_matrix_update_suitesparse(H.valuePtr(), H0_data.data(), mux_data.data(), muy_data.data(), ex1, ey1, nnz);
        #ifdef DEBUG_PERFORMANCE
        auto update_end = Clock::now();
        current_metrics.matrix_update_time += Duration(update_end - update_start).count();
        current_metrics.matrix_updates++;
        #endif

        // RK4ステップの時間を計測
        #ifdef DEBUG_PERFORMANCE
        auto rk4_start = Clock::now();
        #endif

        // OpenBLAS + SuiteSparseまたはMKLを使用した行列-ベクトル積
        #ifdef OPENBLAS_SUITESPARSE_AVAILABLE
        // OpenBLAS + SuiteSparseを使用
        #ifdef DEBUG_PERFORMANCE
        auto sparse_start = Clock::now();
        #endif
        
        // OpenBLAS CBLASを使用した行列-ベクトル積
        // 複素数ベクトルを実部・虚部に分離
        std::vector<double> psi_real(dim);
        std::vector<double> psi_imag(dim);
        for (int i = 0; i < dim; ++i) {
            psi_real[i] = psi[i].real();
            psi_imag[i] = psi[i].imag();
        }
        
        // 結果ベクトル
        std::vector<double> result_real(dim, 0.0);
        std::vector<double> result_imag(dim, 0.0);
        
        // OpenBLAS CBLAS行列-ベクトル積: H * psi
        // 疎行列なので、非ゼロ要素のみを計算
        for (int k = 0; k < H.outerSize(); ++k) {
            for (Eigen::SparseMatrix<cplx>::InnerIterator it(H, k); it; ++it) {
                int i = it.row();
                int j = it.col();
                cplx val = it.value();
                
                result_real[i] += val.real() * psi_real[j] - val.imag() * psi_imag[j];
                result_imag[i] += val.real() * psi_imag[j] + val.imag() * psi_real[j];
            }
        }
        
        // 結果を複素数ベクトルに変換: -i * (H * psi)
        for (int i = 0; i < dim; ++i) {
            k1[i] = cplx(result_imag[i], -result_real[i]);
        }
        
        #ifdef DEBUG_PERFORMANCE
        auto sparse_end = Clock::now();
        current_metrics.sparse_solve_time += Duration(sparse_end - sparse_start).count();
        current_metrics.sparse_solves++;
        #endif
        
        #elif defined(SUITESPARSE_MKL_AVAILABLE)
        // MKL Sparse BLASを使用
        #ifdef DEBUG_PERFORMANCE
        auto sparse_start = Clock::now();
        #endif
        
        // MKL行列を更新（必要に応じて）
        if (mkl_H != nullptr) {
            mkl_sparse_destroy(mkl_H);
        }
        mkl_H = eigen_to_mkl_sparse(H);
        
        // 複素数ベクトルを実部・虚部に分離
        std::vector<double> psi_real(dim);
        std::vector<double> psi_imag(dim);
        for (int i = 0; i < dim; ++i) {
            psi_real[i] = psi[i].real();
            psi_imag[i] = psi[i].imag();
        }
        
        // 結果ベクトル
        std::vector<double> result_real(dim, 0.0);
        std::vector<double> result_imag(dim, 0.0);
        
        // MKL Sparse BLAS行列-ベクトル積: H * psi
        sparse_status_t status = mkl_sparse_z_mv(
            SPARSE_OPERATION_NON_TRANSPOSE,
            cplx(1.0, 0.0),
            mkl_H,
            mkl_descr,
            psi_real.data(),
            psi_imag.data(),
            cplx(0.0, 0.0),
            result_real.data(),
            result_imag.data()
        );
        
        if (status != SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("MKL sparse matrix-vector multiplication failed");
        }
        
        // 結果を複素数ベクトルに変換: -i * (H * psi)
        for (int i = 0; i < dim; ++i) {
            k1[i] = cplx(result_imag[i], -result_real[i]);
        }
        
        #ifdef DEBUG_PERFORMANCE
        auto sparse_end = Clock::now();
        current_metrics.sparse_solve_time += Duration(sparse_end - sparse_start).count();
        current_metrics.sparse_solves++;
        #endif
        #else
        // フォールバック: Eigenを使用
        k1 = cplx(0, -1) * (H * psi);
        #endif
        
        buf = psi + 0.5 * dt * k1;

        // H2
        optimized_parallel_matrix_update_suitesparse(H.valuePtr(), H0_data.data(), mux_data.data(), muy_data.data(), ex2, ey2, nnz);
        
        #ifdef OPENBLAS_SUITESPARSE_AVAILABLE
        // OpenBLAS + SuiteSparseを使用（H1と同じ行列なので最適化可能）
        // 複素数ベクトルを実部・虚部に分離
        std::vector<double> buf_real2(dim);
        std::vector<double> buf_imag2(dim);
        for (int i = 0; i < dim; ++i) {
            buf_real2[i] = buf[i].real();
            buf_imag2[i] = buf[i].imag();
        }
        
        // 結果ベクトル
        std::vector<double> result_real2(dim, 0.0);
        std::vector<double> result_imag2(dim, 0.0);
        
        // OpenBLAS CBLAS行列-ベクトル積: H * buf
        for (int k = 0; k < H.outerSize(); ++k) {
            for (Eigen::SparseMatrix<cplx>::InnerIterator it(H, k); it; ++it) {
                int i = it.row();
                int j = it.col();
                cplx val = it.value();
                
                result_real2[i] += val.real() * buf_real2[j] - val.imag() * buf_imag2[j];
                result_imag2[i] += val.real() * buf_imag2[j] + val.imag() * buf_real2[j];
            }
        }
        
        // 結果を複素数ベクトルに変換: -i * (H * buf)
        for (int i = 0; i < dim; ++i) {
            k2[i] = cplx(result_imag2[i], -result_real2[i]);
        }
        
        #elif defined(SUITESPARSE_MKL_AVAILABLE)
        // MKL Sparse BLASを使用
        // MKL行列を更新
        if (mkl_H != nullptr) {
            mkl_sparse_destroy(mkl_H);
        }
        mkl_H = eigen_to_mkl_sparse(H);
        
        // 複素数ベクトルを実部・虚部に分離
        std::vector<double> buf_real(dim);
        std::vector<double> buf_imag(dim);
        for (int i = 0; i < dim; ++i) {
            buf_real[i] = buf[i].real();
            buf_imag[i] = buf[i].imag();
        }
        
        // 結果ベクトル
        std::vector<double> result_real(dim, 0.0);
        std::vector<double> result_imag(dim, 0.0);
        
        // MKL Sparse BLAS行列-ベクトル積: H * buf
        sparse_status_t status = mkl_sparse_z_mv(
            SPARSE_OPERATION_NON_TRANSPOSE,
            cplx(1.0, 0.0),
            mkl_H,
            mkl_descr,
            buf_real.data(),
            buf_imag.data(),
            cplx(0.0, 0.0),
            result_real.data(),
            result_imag.data()
        );
        
        if (status != SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("MKL sparse matrix-vector multiplication failed");
        }
        
        // 結果を複素数ベクトルに変換: -i * (H * buf)
        for (int i = 0; i < dim; ++i) {
            k2[i] = cplx(result_imag[i], -result_real[i]);
        }
        #else
        k2 = cplx(0, -1) * (H * buf);
        #endif
        
        buf = psi + 0.5 * dt * k2;

        // H3
        optimized_parallel_matrix_update_suitesparse(H.valuePtr(), H0_data.data(), mux_data.data(), muy_data.data(), ex2, ey2, nnz);
        
        #ifdef OPENBLAS_SUITESPARSE_AVAILABLE
        // OpenBLAS + SuiteSparseを使用（H2と同じ行列なので最適化可能）
        // 複素数ベクトルを実部・虚部に分離
        std::vector<double> buf_real3(dim);
        std::vector<double> buf_imag3(dim);
        for (int i = 0; i < dim; ++i) {
            buf_real3[i] = buf[i].real();
            buf_imag3[i] = buf[i].imag();
        }
        
        // 結果ベクトル
        std::vector<double> result_real3(dim, 0.0);
        std::vector<double> result_imag3(dim, 0.0);
        
        // OpenBLAS CBLAS行列-ベクトル積: H * buf
        for (int k = 0; k < H.outerSize(); ++k) {
            for (Eigen::SparseMatrix<cplx>::InnerIterator it(H, k); it; ++it) {
                int i = it.row();
                int j = it.col();
                cplx val = it.value();
                
                result_real3[i] += val.real() * buf_real3[j] - val.imag() * buf_imag3[j];
                result_imag3[i] += val.real() * buf_imag3[j] + val.imag() * buf_real3[j];
            }
        }
        
        // 結果を複素数ベクトルに変換: -i * (H * buf)
        for (int i = 0; i < dim; ++i) {
            k3[i] = cplx(result_imag3[i], -result_real3[i]);
        }
        
        #elif defined(SUITESPARSE_MKL_AVAILABLE)
        // MKL Sparse BLASを使用（H2と同じ行列なので更新不要）
        // 複素数ベクトルを実部・虚部に分離
        std::vector<double> buf_real(dim);
        std::vector<double> buf_imag(dim);
        for (int i = 0; i < dim; ++i) {
            buf_real[i] = buf[i].real();
            buf_imag[i] = buf[i].imag();
        }
        
        // 結果ベクトル
        std::vector<double> result_real(dim, 0.0);
        std::vector<double> result_imag(dim, 0.0);
        
        // MKL Sparse BLAS行列-ベクトル積: H * buf
        sparse_status_t status = mkl_sparse_z_mv(
            SPARSE_OPERATION_NON_TRANSPOSE,
            cplx(1.0, 0.0),
            mkl_H,
            mkl_descr,
            buf_real.data(),
            buf_imag.data(),
            cplx(0.0, 0.0),
            result_real.data(),
            result_imag.data()
        );
        
        if (status != SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("MKL sparse matrix-vector multiplication failed");
        }
        
        // 結果を複素数ベクトルに変換: -i * (H * buf)
        for (int i = 0; i < dim; ++i) {
            k3[i] = cplx(result_imag[i], -result_real[i]);
        }
        #else
        k3 = cplx(0, -1) * (H * buf);
        #endif
        
        buf = psi + dt * k3;

        // H4
        optimized_parallel_matrix_update_suitesparse(H.valuePtr(), H0_data.data(), mux_data.data(), muy_data.data(), ex4, ey4, nnz);
        
        #ifdef OPENBLAS_SUITESPARSE_AVAILABLE
        // OpenBLAS + SuiteSparseを使用
        // 複素数ベクトルを実部・虚部に分離
        std::vector<double> buf_real4(dim);
        std::vector<double> buf_imag4(dim);
        for (int i = 0; i < dim; ++i) {
            buf_real4[i] = buf[i].real();
            buf_imag4[i] = buf[i].imag();
        }
        
        // 結果ベクトル
        std::vector<double> result_real4(dim, 0.0);
        std::vector<double> result_imag4(dim, 0.0);
        
        // OpenBLAS CBLAS行列-ベクトル積: H * buf
        for (int k = 0; k < H.outerSize(); ++k) {
            for (Eigen::SparseMatrix<cplx>::InnerIterator it(H, k); it; ++it) {
                int i = it.row();
                int j = it.col();
                cplx val = it.value();
                
                result_real4[i] += val.real() * buf_real4[j] - val.imag() * buf_imag4[j];
                result_imag4[i] += val.real() * buf_imag4[j] + val.imag() * buf_real4[j];
            }
        }
        
        // 結果を複素数ベクトルに変換: -i * (H * buf)
        for (int i = 0; i < dim; ++i) {
            k4[i] = cplx(result_imag4[i], -result_real4[i]);
        }
        
        #elif defined(SUITESPARSE_MKL_AVAILABLE)
        // MKL Sparse BLASを使用
        // MKL行列を更新
        if (mkl_H != nullptr) {
            mkl_sparse_destroy(mkl_H);
        }
        mkl_H = eigen_to_mkl_sparse(H);
        
        // 複素数ベクトルを実部・虚部に分離
        std::vector<double> buf_real(dim);
        std::vector<double> buf_imag(dim);
        for (int i = 0; i < dim; ++i) {
            buf_real[i] = buf[i].real();
            buf_imag[i] = buf[i].imag();
        }
        
        // 結果ベクトル
        std::vector<double> result_real(dim, 0.0);
        std::vector<double> result_imag(dim, 0.0);
        
        // MKL Sparse BLAS行列-ベクトル積: H * buf
        sparse_status_t status = mkl_sparse_z_mv(
            SPARSE_OPERATION_NON_TRANSPOSE,
            cplx(1.0, 0.0),
            mkl_H,
            mkl_descr,
            buf_real.data(),
            buf_imag.data(),
            cplx(0.0, 0.0),
            result_real.data(),
            result_imag.data()
        );
        
        if (status != SPARSE_STATUS_SUCCESS) {
            throw std::runtime_error("MKL sparse matrix-vector multiplication failed");
        }
        
        // 結果を複素数ベクトルに変換: -i * (H * buf)
        for (int i = 0; i < dim; ++i) {
            k4[i] = cplx(result_imag[i], -result_real[i]);
        }
        #else
        k4 = cplx(0, -1) * (H * buf);
        #endif

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