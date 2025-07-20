#include "excitation_rk4_sparse/core.hpp"
#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <chrono>
#include <limits>

// SIMD関連のヘッダー（アーキテクチャ対応）
#if defined(__x86_64__) || defined(__i386__)
  #ifdef __AVX2__
  #include <immintrin.h>
  #endif
  #include <xmmintrin.h>  // _mm_malloc, _mm_free用
#elif defined(__aarch64__) || defined(__arm__)
  #include <arm_neon.h>   // ARM NEON SIMD
  #include <cstdlib>      // aligned_alloc用
#endif
#include <malloc.h>       // 汎用メモリアライメント関数

// BLASヘッダーのインクルード
#ifdef OPENBLAS_SUITESPARSE_AVAILABLE
extern "C" {
    #include <cblas.h>
}
#endif

namespace excitation_rk4_sparse {

static PerformanceMetrics current_metrics;

// プラットフォーム対応のアライメントされたメモリ割り当て
inline void* aligned_malloc(size_t size, size_t alignment) {
#if defined(__x86_64__) || defined(__i386__)
    return _mm_malloc(size, alignment);
#elif defined(__aarch64__) || defined(__arm__)
    void* ptr = nullptr;
    if (posix_memalign(&ptr, alignment, size) == 0) {
        return ptr;
    }
    return nullptr;
#else
    // フォールバック: 標準のmalloc（アライメントなし）
    return malloc(size);
#endif
}

// プラットフォーム対応のアライメントされたメモリ解放
inline void aligned_free(void* ptr) {
#if defined(__x86_64__) || defined(__i386__)
    _mm_free(ptr);
#else
    free(ptr);
#endif
}

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
    const std::complex<double>* muy_data = muy.valuePtr();
    
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
            for (int i = 0; i < nnz; ++i) {
                H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
            }
        } else if (dim >= 4096) {
            // 4096-8192次元：並列化を大幅に制限
            const int optimal_threshold = get_optimal_parallel_threshold();
            if (static_cast<size_t>(nnz) > static_cast<size_t>(optimal_threshold * 256)) {
                const int chunk_size = std::max(2048, static_cast<int>(nnz) / (omp_get_max_threads() * 256));
                #pragma omp parallel for schedule(dynamic, chunk_size)
                for (int i = 0; i < nnz; ++i) {
                    H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
                }
            } else {
                // シリアル実行
                for (int i = 0; i < nnz; ++i) {
                    H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
                }
            }
        } else {
            // 小中規模問題：適応的並列化
            adaptive_parallel_matrix_update(H_values.data(), H0_data, mux_data, muy_data, ex, ey, nnz, dim);
        }
        #else
        // OpenMPが利用できない場合のシリアル実行
        for (int i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
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
        adaptive_parallel_matrix_update(H.valuePtr(), H0_data.data(), mux_data.data(), muy_data.data(), ex1, ey1, nnz, dim);

        // RK4ステップ
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
    const std::complex<double>* muy_data = muy.valuePtr();
    
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
        for (int i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex1 * mux_data[i] + ey1 * muy_data[i];
        }
        
        // k1 = -i * H1 * psi
        blas_optimized_sparse_matrix_vector_multiply_safe(H_values.data(), H0_indices, H0_indptr, psi, k1, dim);
        buf = psi + 0.5 * dt * k1;
        
        // H2 - k2の計算用
        for (int i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex2 * mux_data[i] + ey2 * muy_data[i];
        }
        
        // k2 = -i * H2 * (psi + dt/2 * k1)
        blas_optimized_sparse_matrix_vector_multiply_safe(H_values.data(), H0_indices, H0_indptr, buf, k2, dim);
        buf = psi + 0.5 * dt * k2;
        
        // H3 - k3の計算用（ex2, ey2を使用）
        for (int i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex2 * mux_data[i] + ey2 * muy_data[i];
        }
        
        // k3 = -i * H3 * (psi + dt/2 * k2)
        blas_optimized_sparse_matrix_vector_multiply_safe(H_values.data(), H0_indices, H0_indptr, buf, k3, dim);
        buf = psi + dt * k3;
        
        // H4 - k4の計算用
        for (int i = 0; i < nnz; ++i) {
            H_values[i] = H0_data[i] + ex4 * mux_data[i] + ey4 * muy_data[i];
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

// 軽量Julia風実装（小規模問題に最適化）
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
    bool renorm)
{
    using cplx = std::complex<double>;
    constexpr cplx minus_i = cplx(0, -1);
    
    const int steps = (Ex.size() - 1) / 2;
    const int dim = psi0.size();
    const int n_out = return_traj ? (steps + stride - 1) / stride : 1;
    
    // 結果行列
    Eigen::MatrixXcd out;
    if (return_traj) {
        out.resize(n_out, dim);
        out.row(0) = psi0;
    } else {
        out.resize(1, dim);
    }
    
    // 作業用ベクトル（標準的な方法）
    Eigen::VectorXcd psi = psi0;
    Eigen::VectorXcd buf(dim);
    Eigen::VectorXcd k1(dim), k2(dim), k3(dim), k4(dim);
    
    // 電場データを3点セットに変換（軽量版）
    auto Ex3 = field_to_triplets(Ex);
    auto Ey3 = field_to_triplets(Ey);
    
    int idx = 1;
    
    // メインループ（Julia風だが軽量）
    for (int s = 0; s < steps; ++s) {
        const double ex1 = Ex3[s][0];
        const double ex2 = Ex3[s][1];
        const double ex4 = Ex3[s][2];
        const double ey1 = Ey3[s][0];
        const double ey2 = Ey3[s][1];
        const double ey4 = Ey3[s][2];
        
        // RK4ステップ（Eigen標準演算を使用、明示的型指定で安定化）
        Eigen::SparseMatrix<cplx> H1 = H0 + ex1 * mux + ey1 * muy;
        k1 = minus_i * (H1 * psi);
        
        buf = psi + 0.5 * dt * k1;
        Eigen::SparseMatrix<cplx> H2 = H0 + ex2 * mux + ey2 * muy;
        k2 = minus_i * (H2 * buf);
        
        buf = psi + 0.5 * dt * k2;
        k3 = minus_i * (H2 * buf);
        
        buf = psi + dt * k3;
        Eigen::SparseMatrix<cplx> H4 = H0 + ex4 * mux + ey4 * muy;
        k4 = minus_i * (H4 * buf);
        
        // 状態更新
        psi += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        
        // 正規化
        if (renorm) {
            psi.normalize();
        }
        
        // 軌道保存
        if (return_traj && (s % stride == 0)) {
            out.row(idx) = psi;
            idx++;
        }
    }
    
    if (!return_traj) {
        out.row(0) = psi;
    }
    
    return out;
}

// 軽量CSR実装（並列化オーバーヘッドを最小化）
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
    bool renorm)
{
    using cplx = std::complex<double>;
    constexpr cplx minus_i = cplx(0, -1);
    
    const int steps = (Ex.size() - 1) / 2;
    const int dim = psi0.size();
    const int n_out = return_traj ? (steps + stride - 1) / stride : 1;
    
    // 結果行列
    Eigen::MatrixXcd out;
    if (return_traj) {
        out.resize(n_out, dim);
        out.row(0) = psi0;
    } else {
        out.resize(1, dim);
    }
    
    // 小規模問題ではEigenの標準実装を使用（軽量化）
    if (dim < 2048) {
        // 従来のEigen実装と同様
        Eigen::VectorXcd psi = psi0;
        Eigen::VectorXcd buf(dim);
        Eigen::VectorXcd k1(dim), k2(dim), k3(dim), k4(dim);
        
        auto Ex3 = field_to_triplets(Ex);
        auto Ey3 = field_to_triplets(Ey);
        
        int idx = 1;
        
        for (int s = 0; s < steps; ++s) {
            const double ex1 = Ex3[s][0];
            const double ex2 = Ex3[s][1];
            const double ex4 = Ex3[s][2];
            const double ey1 = Ey3[s][0];
            const double ey2 = Ey3[s][1];
            const double ey4 = Ey3[s][2];
            
            // 軽量なRK4ステップ（明示的型指定で安定化）
            Eigen::SparseMatrix<cplx> H1 = H0 + ex1 * mux + ey1 * muy;
            k1 = minus_i * (H1 * psi);
            
            buf = psi + 0.5 * dt * k1;
            Eigen::SparseMatrix<cplx> H2 = H0 + ex2 * mux + ey2 * muy;
            k2 = minus_i * (H2 * buf);
            
            buf = psi + 0.5 * dt * k2;
            k3 = minus_i * (H2 * buf);
            
            buf = psi + dt * k3;
            Eigen::SparseMatrix<cplx> H4 = H0 + ex4 * mux + ey4 * muy;
            k4 = minus_i * (H4 * buf);
            
            psi += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
            
            if (renorm) {
                psi.normalize();
            }
            
            if (return_traj && (s % stride == 0)) {
                out.row(idx) = psi;
                idx++;
            }
        }
        
        if (!return_traj) {
            out.row(0) = psi;
        }
        
        return out;
    }
    
    // 大規模問題のみ、より複雑な実装を使用
    // （現在は小規模問題と同じ実装）
    Eigen::VectorXcd psi = psi0;
    Eigen::VectorXcd buf(dim);
    Eigen::VectorXcd k1(dim), k2(dim), k3(dim), k4(dim);
    
    auto Ex3 = field_to_triplets(Ex);
    auto Ey3 = field_to_triplets(Ey);
    
    int idx = 1;
    
    for (int s = 0; s < steps; ++s) {
        const double ex1 = Ex3[s][0];
        const double ex2 = Ex3[s][1];
        const double ex4 = Ex3[s][2];
        const double ey1 = Ey3[s][0];
        const double ey2 = Ey3[s][1];
        const double ey4 = Ey3[s][2];
        
        Eigen::SparseMatrix<cplx> H1 = H0 + ex1 * mux + ey1 * muy;
        k1 = minus_i * (H1 * psi);
        
        buf = psi + 0.5 * dt * k1;
        Eigen::SparseMatrix<cplx> H2 = H0 + ex2 * mux + ey2 * muy;
        k2 = minus_i * (H2 * buf);
        
        buf = psi + 0.5 * dt * k2;
        k3 = minus_i * (H2 * buf);
        
        buf = psi + dt * k3;
        Eigen::SparseMatrix<cplx> H4 = H0 + ex4 * mux + ey4 * muy;
        k4 = minus_i * (H4 * buf);
        
        psi += (dt / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4);
        
        if (renorm) {
            psi.normalize();
        }
        
        if (return_traj && (s % stride == 0)) {
            out.row(idx) = psi;
            idx++;
        }
    }
    
    if (!return_traj) {
        out.row(0) = psi;
    }
    
    return out;
}

// 中規模問題特化のSIMD最適化関数群
#ifdef __AVX2__
#include <immintrin.h>

// SIMD最適化された行列更新（中規模特化）
inline void simd_optimized_matrix_update_medium_scale(
    std::complex<double>* H_values,
    const std::complex<double>* H0_data,
    const std::complex<double>* mux_data,
    const std::complex<double>* muy_data,
    double ex, double ey, size_t nnz) {
    
    const __m256d ex_vec = _mm256_set1_pd(ex);
    const __m256d ey_vec = _mm256_set1_pd(ey);
    
    size_t simd_end = (nnz / 2) * 2;  // 2つの複素数 = 4つのdouble
    
    for (size_t i = 0; i < simd_end; i += 2) {
        // H0の読み込み（2つの複素数 = 4つのdouble）
        __m256d h0_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&H0_data[i]));
        
        // muxの読み込みとスケーリング
        __m256d mux_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&mux_data[i]));
        __m256d mux_scaled = _mm256_mul_pd(mux_vec, _mm256_set_pd(ex, ex, ex, ex));
        
        // muyの読み込みとスケーリング
        __m256d muy_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&muy_data[i]));
        __m256d muy_scaled = _mm256_mul_pd(muy_vec, _mm256_set_pd(ey, ey, ey, ey));
        
        // 加算とストア: H = H0 + ex*mux + ey*muy
        __m256d result = _mm256_add_pd(_mm256_add_pd(h0_vec, mux_scaled), muy_scaled);
        _mm256_storeu_pd(reinterpret_cast<double*>(&H_values[i]), result);
    }
    
    // 残り要素の処理
    for (size_t i = simd_end; i < nnz; ++i) {
        H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
    }
}

// SIMD最適化されたベクトル演算（中規模特化）
inline void simd_optimized_vector_operations_medium_scale(
    std::complex<double>* psi,
    const std::complex<double>* k1,
    const std::complex<double>* k2,
    const std::complex<double>* k3,
    const std::complex<double>* k4,
    double dt, int dim) {
    
    const double dt_over_6 = dt / 6.0;
    const __m256d dt6_vec = _mm256_set1_pd(dt_over_6);
    const __m256d two_vec = _mm256_set1_pd(2.0);
    
    size_t simd_end = (dim / 2) * 2;
    
    for (size_t i = 0; i < simd_end; i += 2) {
        // k1, k2, k3, k4の読み込み
        __m256d k1_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&k1[i]));
        __m256d k2_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&k2[i]));
        __m256d k3_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&k3[i]));
        __m256d k4_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&k4[i]));
        
        // 2*k2 + 2*k3の計算
        __m256d k2_scaled = _mm256_mul_pd(k2_vec, two_vec);
        __m256d k3_scaled = _mm256_mul_pd(k3_vec, two_vec);
        
        // k1 + 2*k2 + 2*k3 + k4
        __m256d sum = _mm256_add_pd(_mm256_add_pd(k1_vec, k2_scaled), 
                                   _mm256_add_pd(k3_scaled, k4_vec));
        
        // dt/6でスケーリング
        __m256d increment = _mm256_mul_pd(sum, dt6_vec);
        
        // psiの更新
        __m256d psi_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&psi[i]));
        __m256d result = _mm256_add_pd(psi_vec, increment);
        _mm256_storeu_pd(reinterpret_cast<double*>(&psi[i]), result);
    }
    
    // 残り要素
    for (size_t i = simd_end; i < dim; ++i) {
        psi[i] += dt_over_6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

// SIMD最適化されたスパース行列ベクトル積（中規模特化）
inline void simd_optimized_sparse_matvec_medium_scale(
    const std::complex<double>* H_data,
    const int* H_indices,
    const int* H_indptr,
    const std::complex<double>* x,
    std::complex<double>* y,
    int dim) {
    
    constexpr std::complex<double> minus_i(0, -1);
    
    #pragma omp parallel for if(dim > 256)
    for (int row = 0; row < dim; ++row) {
        int start = H_indptr[row];
        int end = H_indptr[row + 1];
        int length = end - start;
        
        if (length == 0) {
            y[row] = 0.0;
            continue;
        }
        
        std::complex<double> sum = 0.0;
        
        // プリフェッチで次の行を先読み
        if (row + 1 < dim && H_indptr[row + 1] < H_indptr[dim]) {
            __builtin_prefetch(&H_data[H_indptr[row + 1]], 0, 3);
        }
        
        // SIMD最適化された内積計算
        int simd_length = (length / 4) * 4;
        
        for (int j = start; j < start + simd_length; j += 4) {
            // 4つずつ処理
            sum += H_data[j] * x[H_indices[j]];
            sum += H_data[j + 1] * x[H_indices[j + 1]];
            sum += H_data[j + 2] * x[H_indices[j + 2]];
            sum += H_data[j + 3] * x[H_indices[j + 3]];
        }
        
        // 残り要素
        for (int j = start + simd_length; j < end; ++j) {
            sum += H_data[j] * x[H_indices[j]];
        }
        
        y[row] = minus_i * sum;
    }
}

#else
// AVX2が利用できない場合のフォールバック実装

inline void simd_optimized_matrix_update_medium_scale(
    std::complex<double>* H_values,
    const std::complex<double>* H0_data,
    const std::complex<double>* mux_data,
    const std::complex<double>* muy_data,
    double ex, double ey, size_t nnz) {
    
    for (size_t i = 0; i < nnz; ++i) {
        H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
    }
}

inline void simd_optimized_vector_operations_medium_scale(
    std::complex<double>* psi,
    const std::complex<double>* k1,
    const std::complex<double>* k2,
    const std::complex<double>* k3,
    const std::complex<double>* k4,
    double dt, int dim) {
    
    const double dt_over_6 = dt / 6.0;
    for (int i = 0; i < dim; ++i) {
        psi[i] += dt_over_6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

inline void simd_optimized_sparse_matvec_medium_scale(
    const std::complex<double>* H_data,
    const int* H_indices,
    const int* H_indptr,
    const std::complex<double>* x,
    std::complex<double>* y,
    int dim) {
    
    constexpr std::complex<double> minus_i(0, -1);
    
    for (int row = 0; row < dim; ++row) {
        std::complex<double> sum = 0.0;
        for (int j = H_indptr[row]; j < H_indptr[row + 1]; ++j) {
            sum += H_data[j] * x[H_indices[j]];
        }
        y[row] = minus_i * sum;
    }
}

#endif

// 中規模問題特化の超高速実装
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
    bool renorm)
{
    using cplx = std::complex<double>;
    constexpr cplx minus_i = cplx(0, -1);
    
    const int steps = (Ex.size() - 1) / 2;
    const int dim = psi0.size();
    const int n_out = return_traj ? (steps + stride - 1) / stride : 1;
    
    // 中規模問題でない場合は標準実装にフォールバック
    if (dim < 100 || dim > 1000) {
        return rk4_sparse_julia_style(H0, mux, muy, Ex, Ey, psi0, dt, return_traj, stride, renorm);
    }
    
    // 結果行列
    Eigen::MatrixXcd out;
    if (return_traj) {
        out.resize(n_out, dim);
        out.row(0) = psi0;
    } else {
        out.resize(1, dim);
    }
    
    // CSR形式データの取得
    const cplx* H0_data = H0.valuePtr();
    const int* H0_indices = H0.innerIndexPtr();
    const int* H0_indptr = H0.outerIndexPtr();
    const cplx* mux_data = mux.valuePtr();
    const cplx* muy_data = muy.valuePtr();
    const int nnz = H0.nonZeros();
    
    // アライメントされたメモリバッファ
    cplx* H_values = static_cast<cplx*>(aligned_malloc(nnz * sizeof(cplx), 32));
    cplx* k1 = static_cast<cplx*>(aligned_malloc(dim * sizeof(cplx), 32));
    cplx* k2 = static_cast<cplx*>(aligned_malloc(dim * sizeof(cplx), 32));
    cplx* k3 = static_cast<cplx*>(aligned_malloc(dim * sizeof(cplx), 32));
    cplx* k4 = static_cast<cplx*>(aligned_malloc(dim * sizeof(cplx), 32));
    cplx* buf = static_cast<cplx*>(aligned_malloc(dim * sizeof(cplx), 32));
    
    // 作業用ベクトル
    Eigen::VectorXcd psi = psi0;
    
    // 電場データの事前計算
    auto Ex3 = field_to_triplets(Ex);
    auto Ey3 = field_to_triplets(Ey);
    
    int idx = 1;
    
    // メインループ（中規模特化最適化）
    for (int s = 0; s < steps; ++s) {
        const double ex1 = Ex3[s][0];
        const double ex2 = Ex3[s][1];
        const double ex4 = Ex3[s][2];
        const double ey1 = Ey3[s][0];
        const double ey2 = Ey3[s][1];
        const double ey4 = Ey3[s][2];
        
        // k1計算（SIMD最適化）
        simd_optimized_matrix_update_medium_scale(H_values, H0_data, mux_data, muy_data, ex1, ey1, nnz);
        simd_optimized_sparse_matvec_medium_scale(H_values, H0_indices, H0_indptr, psi.data(), k1, dim);
        
        // buf = psi + 0.5*dt*k1（SIMD最適化）
        #ifdef __AVX2__
        const __m256d half_dt = _mm256_set1_pd(0.5 * dt);
        size_t simd_end = (dim / 2) * 2;
        for (size_t i = 0; i < simd_end; i += 2) {
            __m256d psi_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&psi[i]));
            __m256d k1_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&k1[i]));
            __m256d result = _mm256_add_pd(psi_vec, _mm256_mul_pd(k1_vec, half_dt));
            _mm256_storeu_pd(reinterpret_cast<double*>(&buf[i]), result);
        }
        for (size_t i = simd_end; i < dim; ++i) {
            buf[i] = psi[i] + 0.5 * dt * k1[i];
        }
        #else
        for (int i = 0; i < dim; ++i) {
            buf[i] = psi[i] + 0.5 * dt * k1[i];
        }
        #endif
        
        // k2計算
        simd_optimized_matrix_update_medium_scale(H_values, H0_data, mux_data, muy_data, ex2, ey2, nnz);
        simd_optimized_sparse_matvec_medium_scale(H_values, H0_indices, H0_indptr, buf, k2, dim);
        
        // buf = psi + 0.5*dt*k2
        #ifdef __AVX2__
        for (size_t i = 0; i < simd_end; i += 2) {
            __m256d psi_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&psi[i]));
            __m256d k2_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&k2[i]));
            __m256d result = _mm256_add_pd(psi_vec, _mm256_mul_pd(k2_vec, half_dt));
            _mm256_storeu_pd(reinterpret_cast<double*>(&buf[i]), result);
        }
        for (size_t i = simd_end; i < dim; ++i) {
            buf[i] = psi[i] + 0.5 * dt * k2[i];
        }
        #else
        for (int i = 0; i < dim; ++i) {
            buf[i] = psi[i] + 0.5 * dt * k2[i];
        }
        #endif
        
        // k3計算（H2と同じ）
        simd_optimized_sparse_matvec_medium_scale(H_values, H0_indices, H0_indptr, buf, k3, dim);
        
        // buf = psi + dt*k3
        #ifdef __AVX2__
        const __m256d full_dt = _mm256_set1_pd(dt);
        for (size_t i = 0; i < simd_end; i += 2) {
            __m256d psi_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&psi[i]));
            __m256d k3_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&k3[i]));
            __m256d result = _mm256_add_pd(psi_vec, _mm256_mul_pd(k3_vec, full_dt));
            _mm256_storeu_pd(reinterpret_cast<double*>(&buf[i]), result);
        }
        for (size_t i = simd_end; i < dim; ++i) {
            buf[i] = psi[i] + dt * k3[i];
        }
        #else
        for (int i = 0; i < dim; ++i) {
            buf[i] = psi[i] + dt * k3[i];
        }
        #endif
        
        // k4計算
        simd_optimized_matrix_update_medium_scale(H_values, H0_data, mux_data, muy_data, ex4, ey4, nnz);
        simd_optimized_sparse_matvec_medium_scale(H_values, H0_indices, H0_indptr, buf, k4, dim);
        
        // 状態更新（SIMD最適化）
        simd_optimized_vector_operations_medium_scale(psi.data(), k1, k2, k3, k4, dt, dim);
        
        // 正規化（ブランチレス）
        if (renorm) {
            double norm_sq = 0.0;
            for (int i = 0; i < dim; ++i) {
                norm_sq += std::norm(psi[i]);
            }
            double norm_factor = 1.0 / std::sqrt(norm_sq);
            
            #ifdef __AVX2__
            const __m256d factor_vec = _mm256_set1_pd(norm_factor);
            for (size_t i = 0; i < simd_end; i += 2) {
                __m256d psi_vec = _mm256_loadu_pd(reinterpret_cast<double*>(&psi[i]));
                __m256d result = _mm256_mul_pd(psi_vec, factor_vec);
                _mm256_storeu_pd(reinterpret_cast<double*>(&psi[i]), result);
            }
            for (size_t i = simd_end; i < dim; ++i) {
                psi[i] *= norm_factor;
            }
            #else
            for (int i = 0; i < dim; ++i) {
                psi[i] *= norm_factor;
            }
            #endif
        }
        
        // 軌道保存
        if (return_traj && (s % stride == 0)) {
            out.row(idx) = psi;
            idx++;
        }
    }
    
    if (!return_traj) {
        out.row(0) = psi;
    }
    
    // メモリ解放
    aligned_free(H_values);
    aligned_free(k1);
    aligned_free(k2);
    aligned_free(k3);
    aligned_free(k4);
    aligned_free(buf);
    
    return out;
}

} // namespace excitation_rk4_sparse

