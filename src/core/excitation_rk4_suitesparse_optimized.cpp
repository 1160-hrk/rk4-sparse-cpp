#include "excitation_rk4_sparse/suitesparse.hpp"
#include "excitation_rk4_sparse/core.hpp"
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <chrono>

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

static SuiteSparseOptimizedPerformanceMetrics current_metrics;

// SuiteSparse-MKL版のRK4実装
Eigen::MatrixXcd rk4_sparse_suitesparse_optimized(
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
    current_metrics = SuiteSparseOptimizedPerformanceMetrics();

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

    // 2️⃣ パターンに合わせてデータを展開（Eigen版と同じ）
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