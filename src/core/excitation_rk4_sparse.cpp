#include "excitation_rk4_sparse/core.hpp"
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <chrono>
#include <limits>

// BLASヘッダーのインクルード
#ifdef OPENBLAS_SUITESPARSE_AVAILABLE
extern "C" {
    #include <cblas.h>
}
#endif

namespace excitation_rk4_sparse {

static PerformanceMetrics current_metrics;

// Phase 4: 適応的並列化閾値の計算関数（8192次元対応・超厳格版）
inline int get_optimal_parallel_threshold() {
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

// Phase 4: 最適化されたスパース行列-ベクトル積
inline void optimized_sparse_matrix_vector_multiply(
    const Eigen::SparseMatrix<std::complex<double>>& H,
    const Eigen::VectorXcd& x,
    Eigen::VectorXcd& y,
    int dim) {
    
    using cplx = std::complex<double>;
    const int optimal_threshold = get_optimal_parallel_threshold();
    
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

// Phase 4: 適応的並列化戦略（8192次元対応・超厳格版）
inline void adaptive_parallel_matrix_update(
    std::complex<double>* H_values,
    const std::complex<double>* H0_data,
    const std::complex<double>* mux_data,
    const std::complex<double>* muy_data,
    double ex, double ey, size_t nnz, int dim) {
    
    const int optimal_threshold = get_optimal_parallel_threshold();
    
    #ifdef _OPENMP
    if (dim >= 8192) {
        // 8192次元以上：並列化を完全に無効化（シリアル実行）
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
        }
    } else if (dim >= 4096) {
        // 4096-8192次元：並列化を大幅に制限（極大規模問題のみ）
        if (nnz > optimal_threshold * 256) {
            const int chunk_size = std::max(2048, static_cast<int>(nnz) / (omp_get_max_threads() * 256));
            #pragma omp parallel for schedule(dynamic, chunk_size)
            for (int i = 0; i < static_cast<int>(nnz); ++i) {
                H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
            }
        } else {
            // シリアル実行
            for (int i = 0; i < static_cast<int>(nnz); ++i) {
                H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
            }
        }
    } else if (nnz > optimal_threshold * 256) {
        // 極大規模問題：動的スケジューリング
        const int chunk_size = std::max(2048, static_cast<int>(nnz) / (omp_get_max_threads() * 256));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 128) {
        // 超大規模問題：動的スケジューリング
        const int chunk_size = std::max(1024, static_cast<int>(nnz) / (omp_get_max_threads() * 128));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 64) {
        // 大規模問題：動的スケジューリング
        const int chunk_size = std::max(512, static_cast<int>(nnz) / (omp_get_max_threads() * 64));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 32) {
        // 中大規模問題：ガイド付きスケジューリング
        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 16) {
        // 中規模問題：静的スケジューリング
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 8) {
        // 小中規模問題：小さなチャンクサイズでの静的スケジューリング
        const int chunk_size = std::max(1, static_cast<int>(nnz) / (omp_get_max_threads() * 8));
        #pragma omp parallel for schedule(static, chunk_size)
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
        }
    } else {
        // 小規模問題：シリアル実行（並列化オーバーヘッドを回避）
        // 1024次元以下では実質的にシリアル実行
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
        }
    }
    #else
    // OpenMPが利用できない場合のシリアル実行
    for (int i = 0; i < static_cast<int>(nnz); ++i) {
        H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
    }
    #endif
}

// Phase 4: 最適化された並列化戦略（8192次元対応・超厳格版）
inline void optimized_parallel_matrix_update(
    std::complex<double>* H_values,
    const std::complex<double>* H0_data,
    const std::complex<double>* mux_data,
    const std::complex<double>* muy_data,
    double ex, double ey, size_t nnz) {
    
    const int optimal_threshold = get_optimal_parallel_threshold();
    
    #ifdef _OPENMP
    if (nnz > optimal_threshold * 256) {
        // 極大規模問題：動的スケジューリング（負荷分散最適化）
        const int chunk_size = std::max(2048, static_cast<int>(nnz) / (omp_get_max_threads() * 256));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 128) {
        // 超大規模問題：動的スケジューリング（負荷分散最適化）
        const int chunk_size = std::max(1024, static_cast<int>(nnz) / (omp_get_max_threads() * 128));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 64) {
        // 大規模問題：動的スケジューリング（負荷分散最適化）
        const int chunk_size = std::max(512, static_cast<int>(nnz) / (omp_get_max_threads() * 64));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 32) {
        // 中大規模問題：ガイド付きスケジューリング（適応的負荷分散）
        #pragma omp parallel for schedule(guided)
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 16) {
        // 中規模問題：静的スケジューリング（低オーバーヘッド）
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
        }
    } else if (nnz > optimal_threshold * 8) {
        // 小中規模問題：小さなチャンクサイズでの静的スケジューリング
        const int chunk_size = std::max(1, static_cast<int>(nnz) / (omp_get_max_threads() * 8));
        #pragma omp parallel for schedule(static, chunk_size)
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
        }
    } else {
        // 小規模問題：シリアル実行（並列化オーバーヘッドを回避）
        // 1024次元以下では実質的にシリアル実行
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
        }
    }
    #else
    // OpenMPが利用できない場合のシリアル実行
    for (int i = 0; i < static_cast<int>(nnz); ++i) {
        H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
    }
    #endif
}

// Phase 4: 最適化されたデータ展開関数（8192次元対応・超厳格版）
inline std::vector<std::complex<double>> optimized_expand_to_pattern(
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

    // Phase 4: 適応的並列化によるデータ展開（8192次元対応・超厳格版）
    const size_t nnz = pattern.nonZeros();
    const int optimal_threshold = get_optimal_parallel_threshold();
    
    #ifdef _OPENMP
    if (nnz > optimal_threshold * 128) {
        // 極大規模データ：動的スケジューリング
        const int chunk_size = std::max(1024, static_cast<int>(nnz) / (omp_get_max_threads() * 128));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            result[i] = mat.coeff(pi[i], pj[i]);
        }
    } else if (nnz > optimal_threshold * 64) {
        // 超大規模データ：動的スケジューリング
        const int chunk_size = std::max(512, static_cast<int>(nnz) / (omp_get_max_threads() * 64));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            result[i] = mat.coeff(pi[i], pj[i]);
        }
    } else if (nnz > optimal_threshold * 32) {
        // 大規模データ：動的スケジューリング
        const int chunk_size = std::max(256, static_cast<int>(nnz) / (omp_get_max_threads() * 32));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            result[i] = mat.coeff(pi[i], pj[i]);
        }
    } else if (nnz > optimal_threshold * 16) {
        // 中大規模データ：動的スケジューリング
        const int chunk_size = std::max(128, static_cast<int>(nnz) / (omp_get_max_threads() * 16));
        #pragma omp parallel for schedule(dynamic, chunk_size)
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            result[i] = mat.coeff(pi[i], pj[i]);
        }
    } else if (nnz > optimal_threshold * 8) {
        // 中規模データ：静的スケジューリング
        #pragma omp parallel for schedule(static)
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            result[i] = mat.coeff(pi[i], pj[i]);
        }
    } else if (nnz > optimal_threshold * 4) {
        // 小中規模データ：小さなチャンクサイズでの静的スケジューリング
        const int chunk_size = std::max(1, static_cast<int>(nnz) / (omp_get_max_threads() * 8));
        #pragma omp parallel for schedule(static, chunk_size)
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            result[i] = mat.coeff(pi[i], pj[i]);
        }
    } else {
        // 小規模データ：シリアル実行（並列化オーバーヘッドを回避）
        // 1024次元以下では実質的にシリアル実行
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            result[i] = mat.coeff(pi[i], pj[i]);
        }
    }
    #else
    for (int i = 0; i < static_cast<int>(nnz); ++i) {
        result[i] = mat.coeff(pi[i], pj[i]);
    }
    #endif
    
    return result;
}

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
    bool renorm) {
    
    using cplx = std::complex<double>;
    
    const int dim = H0.rows();
    const int steps = Ex.size();
    const int traj_size = return_traj ? (steps + stride - 1) / stride : 1;
    
    // 結果行列の初期化
    Eigen::MatrixXcd result(traj_size, dim);
    if (return_traj) {
        result.row(0) = psi0;
    }
    
    // スパース行列のCSR形式データを取得
    const std::complex<double>* H0_data = H0.valuePtr();
    const int* H0_indices = H0.innerIndexPtr();
    const int* H0_indptr = H0.outerIndexPtr();
    
    const std::complex<double>* mux_data = mux.valuePtr();
    const int* mux_indices = mux.innerIndexPtr();
    const int* mux_indptr = mux.outerIndexPtr();
    
    const std::complex<double>* muy_data = muy.valuePtr();
    const int* muy_indices = muy.innerIndexPtr();
    const int* muy_indptr = muy.outerIndexPtr();
    
    // 共通のパターンを取得（H0, mux, muyは同じパターンを持つと仮定）
    const int nnz = H0.nonZeros();
    
    // キャッシュラインサイズの定義
    constexpr size_t CACHE_LINE = 64;
    
    // 作業用ベクトルの初期化
    alignas(CACHE_LINE) Eigen::VectorXcd psi = psi0;
    alignas(CACHE_LINE) Eigen::VectorXcd buf(dim);
    alignas(CACHE_LINE) Eigen::VectorXcd k1(dim);
    alignas(CACHE_LINE) Eigen::VectorXcd k2(dim);
    alignas(CACHE_LINE) Eigen::VectorXcd k3(dim);
    alignas(CACHE_LINE) Eigen::VectorXcd k4(dim);
    
    // 行列更新用の一時バッファ
    alignas(CACHE_LINE) std::vector<cplx> H_values(nnz);
    
    // 時間発展ループ
    for (int s = 0; s < steps; ++s) {
        const double ex = Ex[s];
        const double ey = Ey[s];
        
        // 行列の更新（BLAS最適化版）
        #ifdef _OPENMP
        if (dim >= 8192) {
            // 8192次元以上：並列化を完全に無効化（シリアル実行）
            for (int i = 0; i < static_cast<int>(nnz); ++i) {
                H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
            }
        } else if (dim >= 4096) {
            // 4096-8192次元：並列化を大幅に制限
            const int optimal_threshold = get_optimal_parallel_threshold();
            if (nnz > optimal_threshold * 256) {
                const int chunk_size = std::max(2048, static_cast<int>(nnz) / (omp_get_max_threads() * 256));
                #pragma omp parallel for schedule(dynamic, chunk_size)
                for (int i = 0; i < static_cast<int>(nnz); ++i) {
                    H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
                }
            } else {
                // シリアル実行
                for (int i = 0; i < static_cast<int>(nnz); ++i) {
                    H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
                }
            }
        } else {
            // 小中規模問題：適応的並列化
            adaptive_parallel_matrix_update(H_values.data(), H0_data, mux_data, muy_data, ex, ey, nnz, dim);
        }
        #else
        // OpenMPが利用できない場合のシリアル実行
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            H_values[i] = H0_data[i] - ex * mux_data[i] - ey * muy_data[i];
        }
        #endif
        
        // RK4ステップ（効率的なBLAS実装を使用）
        // k1 = -i * H * psi
        blas_optimized_sparse_matrix_vector_multiply_efficient(H_values.data(), H0_indices, H0_indptr, psi, k1, dim);
        
        // k2 = -i * H * (psi + dt/2 * k1)
        buf = psi + (dt/2.0) * k1;
        blas_optimized_sparse_matrix_vector_multiply_efficient(H_values.data(), H0_indices, H0_indptr, buf, k2, dim);
        
        // k3 = -i * H * (psi + dt/2 * k2)
        buf = psi + (dt/2.0) * k2;
        blas_optimized_sparse_matrix_vector_multiply_efficient(H_values.data(), H0_indices, H0_indptr, buf, k3, dim);
        
        // k4 = -i * H * (psi + dt * k3)
        buf = psi + dt * k3;
        blas_optimized_sparse_matrix_vector_multiply_efficient(H_values.data(), H0_indices, H0_indptr, buf, k4, dim);
        
        // psi += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        psi += (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
        
        // 正規化（必要に応じて）
        if (renorm) {
            psi.normalize();
        }
        
        // 軌道の保存（必要に応じて）
        if (return_traj && s % stride == 0) {
            result.row(s / stride) = psi;
        }
    }
    
    // 最終結果の保存
    if (!return_traj) {
        result.row(0) = psi;
    }
    
    return result;
}

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

// BLAS最適化版のスパース行列-ベクトル積（デバッグ用：BLAS関数を完全に無効化）
void blas_optimized_sparse_matrix_vector_multiply(
    const std::complex<double>* H_data,
    const int* H_indices,
    const int* H_indptr,
    const Eigen::VectorXcd& x,
    Eigen::VectorXcd& y,
    int dim) {
    
    using cplx = std::complex<double>;
    
    // シンプルな実装：デバッグ出力を削除
    y.setZero();
    for (int i = 0; i < dim; ++i) {
        int start = H_indptr[i];
        int end = H_indptr[i + 1];
        for (int j = start; j < end; ++j) {
            int col_idx = H_indices[j];
            if (col_idx >= 0 && col_idx < dim) {  // 境界チェック
                y[i] += H_data[j] * x[col_idx];
            }
        }
    }
    // 虚数単位を掛ける（-iを掛ける）
    y *= cplx(0, -1);
}

// より効率的なBLAS実装（デバッグ用：BLAS関数を完全に無効化）
void blas_optimized_sparse_matrix_vector_multiply_efficient(
    const std::complex<double>* H_data,
    const int* H_indices,
    const int* H_indptr,
    const Eigen::VectorXcd& x,
    Eigen::VectorXcd& y,
    int dim) {
    
    using cplx = std::complex<double>;
    
    // デバッグ用：BLAS関数を完全に無効化して基本的な実装のみ使用
    y.setZero();
    for (int i = 0; i < dim; ++i) {
        int start = H_indptr[i];
        int end = H_indptr[i + 1];
        
        for (int j = start; j < end; ++j) {
            int col_idx = H_indices[j];
            if (col_idx >= 0 && col_idx < dim) {  // 境界チェック
                y[i] += H_data[j] * x[col_idx];
            }
        }
    }
    
    // 虚数単位を掛ける
    y *= cplx(0, 1);
}

// 安全なBLAS最適化版スパース行列-ベクトル積
void blas_optimized_sparse_matrix_vector_multiply_safe(
    const std::complex<double>* H_data,
    const int* H_indices,
    const int* H_indptr,
    const Eigen::VectorXcd& x,
    Eigen::VectorXcd& y,
    int dim) {
    
    using cplx = std::complex<double>;
    
    // デバッグ用：BLAS関数を完全に無効化して基本的な実装のみ使用
    y.setZero();
    for (int i = 0; i < dim; ++i) {
        int start = H_indptr[i];
        int end = H_indptr[i + 1];
        
        for (int j = start; j < end; ++j) {
            int col_idx = H_indices[j];
            if (col_idx >= 0 && col_idx < dim) {  // 境界チェック
                y[i] += H_data[j] * x[col_idx];
            }
        }
    }
    
    // 虚数単位を掛ける（物理式に合わせて-i）
    y *= cplx(0, -1);
}

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
    bool renorm) {
    
    using cplx = std::complex<double>;
    
    const int dim = H0.rows();
    const int steps = Ex.size();
    const int traj_size = return_traj ? (steps + stride - 1) / stride : 1;
    
    // 結果行列の初期化
    Eigen::MatrixXcd result(traj_size, dim);
    if (return_traj) {
        result.row(0) = psi0;
    }
    
    // スパース行列のCSR形式データを取得（コメントアウト - Eigen標準演算のみ使用）
    /*
    const std::complex<double>* H0_data = H0.valuePtr();
    const int* H0_indices = H0.innerIndexPtr();
    const int* H0_indptr = H0.outerIndexPtr();
    
    const std::complex<double>* mux_data = mux.valuePtr();
    const int* mux_indices = mux.innerIndexPtr();
    const int* mux_indptr = mux.outerIndexPtr();
    
    const std::complex<double>* muy_data = muy.valuePtr();
    const int* muy_indices = muy.innerIndexPtr();
    const int* muy_indptr = muy.outerIndexPtr();
    
    // 共通のパターンを取得（H0, mux, muyは同じパターンを持つと仮定）
    const int nnz = H0.nonZeros();
    */
    
    // キャッシュラインサイズの定義
    constexpr size_t CACHE_LINE = 64;
    
    // 作業用ベクトルの初期化
    alignas(CACHE_LINE) Eigen::VectorXcd psi = psi0;
    alignas(CACHE_LINE) Eigen::VectorXcd buf(dim);
    alignas(CACHE_LINE) Eigen::VectorXcd k1(dim);
    alignas(CACHE_LINE) Eigen::VectorXcd k2(dim);
    alignas(CACHE_LINE) Eigen::VectorXcd k3(dim);
    alignas(CACHE_LINE) Eigen::VectorXcd k4(dim);
    
    // 行列更新用の一時バッファ（Eigen標準演算用）
    Eigen::SparseMatrix<cplx> H_current(dim, dim);
    
    // 時間発展ループ
    for (int s = 0; s < steps; ++s) {
        const double ex = Ex[s];
        const double ey = Ey[s];
        
        // 行列の更新（Eigen標準演算）
        H_current = H0 + ex * mux + ey * muy;
        
        // RK4ステップ（Eigen標準演算）
        // k1 = -i * H_current * psi
        k1 = (-cplx(0,1)) * (H_current * psi);
        // k2 = -i * H_current * (psi + dt/2 * k1)
        buf = psi + (dt/2.0) * k1;
        k2 = (-cplx(0,1)) * (H_current * buf);
        // k3 = -i * H_current * (psi + dt/2 * k2)
        buf = psi + (dt/2.0) * k2;
        k3 = (-cplx(0,1)) * (H_current * buf);
        // k4 = -i * H_current * (psi + dt * k3)
        buf = psi + dt * k3;
        k4 = (-cplx(0,1)) * (H_current * buf);
        // psi += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        psi += (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
        // 正規化（必要に応じて）
        if (renorm) {
            psi.normalize();
        }
        // 軌道の保存（必要に応じて）
        if (return_traj && s % stride == 0) {
            result.row(s / stride) = psi;
        }
    }
    
    // 最終結果の保存
    if (!return_traj) {
        result.row(0) = psi;
    }
    
    return result;
}

// Eigen版のRK4実装（Phase 3: 並列化戦略の再設計）
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

    // 2️⃣ Phase 3: 最適化されたデータ展開
    const size_t nnz = pattern.nonZeros();
    alignas(CACHE_LINE) std::vector<cplx> H0_data = optimized_expand_to_pattern(H0, pattern);
    alignas(CACHE_LINE) std::vector<cplx> mux_data = optimized_expand_to_pattern(mux, pattern);
    alignas(CACHE_LINE) std::vector<cplx> muy_data = optimized_expand_to_pattern(muy, pattern);

    // 3️⃣ 計算用行列
    Eigen::SparseMatrix<cplx> H = pattern;

    // 電場データを3点セットに変換
    auto Ex3 = field_to_triplets(Ex);
    auto Ey3 = field_to_triplets(Ey);

    // メインループ
    for (int s = 0; s < steps; ++s) {
        double ex1 = Ex3[s][0], ex2 = Ex3[s][1], ex4 = Ex3[s][2];
        double ey1 = Ey3[s][0], ey2 = Ey3[s][1], ey4 = Ey3[s][2];

        // H1 - Phase 4: 最適化された行列更新（8192次元対応）
        #ifdef DEBUG_PERFORMANCE
        auto update_start = Clock::now();
        #endif
        adaptive_parallel_matrix_update(H.valuePtr(), H0_data.data(), mux_data.data(), muy_data.data(), ex1, ey1, nnz, dim);
        #ifdef DEBUG_PERFORMANCE
        auto update_end = Clock::now();
        current_metrics.matrix_update_time += Duration(update_end - update_start).count();
        current_metrics.matrix_updates++;
        #endif

        // RK4ステップの時間を計測
        #ifdef DEBUG_PERFORMANCE
        auto rk4_start = Clock::now();
        #endif
        optimized_sparse_matrix_vector_multiply(H, psi, k1, dim);
        buf = psi + 0.5 * dt * k1;

        // H2 - Phase 4: 最適化された行列更新（8192次元対応）
        adaptive_parallel_matrix_update(H.valuePtr(), H0_data.data(), mux_data.data(), muy_data.data(), ex2, ey2, nnz, dim);
        optimized_sparse_matrix_vector_multiply(H, buf, k2, dim);
        buf = psi + 0.5 * dt * k2;

        // H3
        optimized_sparse_matrix_vector_multiply(H, buf, k3, dim);
        buf = psi + dt * k3;

        // H4 - Phase 4: 最適化された行列更新（8192次元対応）
        adaptive_parallel_matrix_update(H.valuePtr(), H0_data.data(), mux_data.data(), muy_data.data(), ex4, ey4, nnz, dim);
        optimized_sparse_matrix_vector_multiply(H, buf, k4, dim);

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
    std::cout << "\n=== Eigen版パフォーマンスメトリクス ===\n";
    std::cout << "行列更新平均時間: " << current_metrics.matrix_update_time / current_metrics.matrix_updates * 1000 << " ms\n";
    std::cout << "RK4ステップ平均時間: " << current_metrics.rk4_step_time / current_metrics.rk4_steps * 1000 << " ms\n";
    #endif

    return out;
}

// パターン構築・データ展開のキャッシュ化を行う新規メソッド
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
    bool renorm)
{
    using cplx = std::complex<double>;
    constexpr size_t CACHE_LINE = 64;
    const int steps = (Ex.size() - 1) / 2;
    const int dim = psi0.size();
    const int n_out = steps / stride + 1;

    // 出力行列の準備
    Eigen::MatrixXcd out;
    if (return_traj) {
        out.resize(n_out, dim);
        out.row(0) = psi0;
    } else {
        out.resize(1, dim);
    }

    // --- キャッシュ用static変数 ---
    static int cached_dim = -1;
    static Eigen::SparseMatrix<cplx> cached_pattern;
    static std::vector<cplx> cached_H0_data, cached_mux_data, cached_muy_data;
    static size_t cached_nnz = 0;

    // パターンのキャッシュチェック
    if (cached_dim != dim || cached_pattern.rows() != dim || cached_pattern.cols() != dim) {
        // 共通パターンを構築
        const double threshold = 1e-12;
        Eigen::SparseMatrix<cplx> pattern(dim, dim);
        pattern.setZero();
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
        cached_pattern = pattern;
        cached_H0_data = optimized_expand_to_pattern(H0, pattern);
        cached_mux_data = optimized_expand_to_pattern(mux, pattern);
        cached_muy_data = optimized_expand_to_pattern(muy, pattern);
        cached_nnz = pattern.nonZeros();
        cached_dim = dim;
    }

    // 計算用行列
    Eigen::SparseMatrix<cplx> H = cached_pattern;

    // 電場データを3点セットに変換
    auto Ex3 = field_to_triplets(Ex);
    auto Ey3 = field_to_triplets(Ey);

    alignas(CACHE_LINE) Eigen::VectorXcd psi = psi0;
    alignas(CACHE_LINE) Eigen::VectorXcd buf(dim);
    alignas(CACHE_LINE) Eigen::VectorXcd k1(dim);
    alignas(CACHE_LINE) Eigen::VectorXcd k2(dim);
    alignas(CACHE_LINE) Eigen::VectorXcd k3(dim);
    alignas(CACHE_LINE) Eigen::VectorXcd k4(dim);

    int idx = 1;

    for (int s = 0; s < steps; ++s) {
        double ex1 = Ex3[s][0], ex2 = Ex3[s][1], ex4 = Ex3[s][2];
        double ey1 = Ey3[s][0], ey2 = Ey3[s][1], ey4 = Ey3[s][2];

        // H1
        adaptive_parallel_matrix_update(H.valuePtr(), cached_H0_data.data(), cached_mux_data.data(), cached_muy_data.data(), ex1, ey1, cached_nnz, dim);
        optimized_sparse_matrix_vector_multiply(H, psi, k1, dim);
        buf = psi + 0.5 * dt * k1;

        // H2
        adaptive_parallel_matrix_update(H.valuePtr(), cached_H0_data.data(), cached_mux_data.data(), cached_muy_data.data(), ex2, ey2, cached_nnz, dim);
        optimized_sparse_matrix_vector_multiply(H, buf, k2, dim);
        buf = psi + 0.5 * dt * k2;

        // H3
        optimized_sparse_matrix_vector_multiply(H, buf, k3, dim);
        buf = psi + dt * k3;

        // H4
        adaptive_parallel_matrix_update(H.valuePtr(), cached_H0_data.data(), cached_mux_data.data(), cached_muy_data.data(), ex4, ey4, cached_nnz, dim);
        optimized_sparse_matrix_vector_multiply(H, buf, k4, dim);

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

    return out;
}

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
    bool renorm) {
    
    using cplx = std::complex<double>;
    
    const int dim = H0.rows();
    const int steps = (Ex.size() - 1) / 2;
    const int traj_size = return_traj ? (steps + stride - 1) / stride : 1;
    
    // 結果行列の初期化
    Eigen::MatrixXcd result(traj_size, dim);
    if (return_traj) {
        result.row(0) = psi0;
    }
    
    // スパース行列のCSR形式データを取得
    const std::complex<double>* H0_data = H0.valuePtr();
    const int* H0_indices = H0.innerIndexPtr();
    const int* H0_indptr = H0.outerIndexPtr();
    
    const std::complex<double>* mux_data = mux.valuePtr();
    const int* mux_indices = mux.innerIndexPtr();
    const int* mux_indptr = mux.outerIndexPtr();
    
    const std::complex<double>* muy_data = muy.valuePtr();
    const int* muy_indices = muy.innerIndexPtr();
    const int* muy_indptr = muy.outerIndexPtr();
    
    // 共通のパターンを取得（H0, mux, muyは同じパターンを持つと仮定）
    const int nnz = H0.nonZeros();
    
    // キャッシュラインサイズの定義
    constexpr size_t CACHE_LINE = 64;
    
    // 作業用ベクトルの初期化
    alignas(CACHE_LINE) Eigen::VectorXcd psi = psi0;
    alignas(CACHE_LINE) Eigen::VectorXcd buf(dim);
    alignas(CACHE_LINE) Eigen::VectorXcd k1(dim);
    alignas(CACHE_LINE) Eigen::VectorXcd k2(dim);
    alignas(CACHE_LINE) Eigen::VectorXcd k3(dim);
    alignas(CACHE_LINE) Eigen::VectorXcd k4(dim);
    
    // 行列更新用の一時バッファ
    alignas(CACHE_LINE) std::vector<cplx> H_values(nnz);
    
    // 電場データを3点セットに変換
    auto Ex3 = field_to_triplets(Ex);
    auto Ey3 = field_to_triplets(Ey);
    
    // 時間発展ループ
    for (int s = 0; s < steps; ++s) {
        double ex1 = Ex3[s][0], ex2 = Ex3[s][1], ex4 = Ex3[s][2];
        double ey1 = Ey3[s][0], ey2 = Ey3[s][1], ey4 = Ey3[s][2];
        
        // H1 - k1の計算用
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            H_values[i] = H0_data[i] - ex1 * mux_data[i] - ey1 * muy_data[i];
        }
        
        // k1 = -i * H1 * psi
        blas_optimized_sparse_matrix_vector_multiply_safe(H_values.data(), H0_indices, H0_indptr, psi, k1, dim);
        buf = psi + 0.5 * dt * k1;
        
        // H2 - k2の計算用
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            H_values[i] = H0_data[i] - ex2 * mux_data[i] - ey2 * muy_data[i];
        }
        
        // k2 = -i * H2 * (psi + dt/2 * k1)
        blas_optimized_sparse_matrix_vector_multiply_safe(H_values.data(), H0_indices, H0_indptr, buf, k2, dim);
        buf = psi + 0.5 * dt * k2;
        
        // H3 - k3の計算用（ex2, ey2を使用）
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            H_values[i] = H0_data[i] - ex2 * mux_data[i] - ey2 * muy_data[i];
        }
        
        // k3 = -i * H3 * (psi + dt/2 * k2)
        blas_optimized_sparse_matrix_vector_multiply_safe(H_values.data(), H0_indices, H0_indptr, buf, k3, dim);
        buf = psi + dt * k3;
        
        // H4 - k4の計算用
        for (int i = 0; i < static_cast<int>(nnz); ++i) {
            H_values[i] = H0_data[i] - ex4 * mux_data[i] - ey4 * muy_data[i];
        }
        
        // k4 = -i * H4 * (psi + dt * k3)
        blas_optimized_sparse_matrix_vector_multiply_safe(H_values.data(), H0_indices, H0_indptr, buf, k4, dim);
        
        // psi += dt/6 * (k1 + 2*k2 + 2*k3 + k4)
        psi += (dt/6.0) * (k1 + 2.0*k2 + 2.0*k3 + k4);
        
        // 正規化（必要に応じて）
        if (renorm) {
            psi.normalize();
        }
        
        // 軌道の保存（必要に応じて）
        if (return_traj && s % stride == 0) {
            result.row(s / stride) = psi;
        }
    }
    
    // 最終結果の保存
    if (!return_traj) {
        result.row(0) = psi;
    }
    
    return result;
}

} // namespace excitation_rk4_sparse

