#include "excitation_rk4_sparse/core.hpp"
#include "simd_complex_ops.hpp"
#include <Eigen/Sparse>
#include <vector>
#include <complex>

namespace excitation_rk4_sparse {

/**
 * Julia Killer RK4実装
 * JuliaのSIMD実装（@turbo + LoopVectorization.jl）を上回る性能を目指す
 */
Eigen::MatrixXcd julia_killer_rk4_phase1(
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
    
    const int nnz = H0.nonZeros();
    
    // キャッシュアライメント最適化（64byte境界）
    constexpr size_t CACHE_LINE = 64;
    alignas(CACHE_LINE) std::vector<cplx> psi_vec(psi0.data(), psi0.data() + dim);
    alignas(CACHE_LINE) std::vector<cplx> buf(dim);
    alignas(CACHE_LINE) std::vector<cplx> k1(dim);
    alignas(CACHE_LINE) std::vector<cplx> k2(dim);
    alignas(CACHE_LINE) std::vector<cplx> k3(dim);
    alignas(CACHE_LINE) std::vector<cplx> k4(dim);
    
    // 行列更新用の一時バッファ（キャッシュアライメント）
    alignas(CACHE_LINE) std::vector<cplx> H_values(nnz);
    
    // 電場データを3点セットに変換（Juliaと同じパターン）
    auto Ex3 = field_to_triplets(Ex);
    auto Ey3 = field_to_triplets(Ey);
    
    int traj_idx = 1;
    
    // メインループ（Julia Killer最適化）
    for (int s = 0; s < steps; ++s) {
        const double ex1 = Ex3[s][0];
        const double ex2 = Ex3[s][1]; 
        const double ex4 = Ex3[s][2];
        const double ey1 = Ey3[s][0];
        const double ey2 = Ey3[s][1];
        const double ey4 = Ey3[s][2];
        
        // 🚀 Phase 1: SIMD最適化された行列更新
        // H1 = H0 + ex1*mux + ey1*muy (k1計算用)
        julia_killer::optimized_complex_matrix_update(
            H_values.data(), H0_data, mux_data, muy_data, ex1, ey1, nnz);
        
        // k1 = -i * H1 * psi (SIMD最適化されたスパース行列-ベクトル積)
        julia_killer_sparse_matvec(H_values.data(), H0_indices, H0_indptr, 
                                  psi_vec.data(), k1.data(), dim);
        
        // buf = psi + 0.5*dt*k1 (SIMD最適化されたベクトル演算)
        julia_killer_vector_add_scaled(buf.data(), psi_vec.data(), k1.data(), 
                                      0.5 * dt, dim);
        
        // H2 = H0 + ex2*mux + ey2*muy (k2計算用)
        julia_killer::optimized_complex_matrix_update(
            H_values.data(), H0_data, mux_data, muy_data, ex2, ey2, nnz);
        
        // k2 = -i * H2 * buf
        julia_killer_sparse_matvec(H_values.data(), H0_indices, H0_indptr,
                                  buf.data(), k2.data(), dim);
        
        // buf = psi + 0.5*dt*k2
        julia_killer_vector_add_scaled(buf.data(), psi_vec.data(), k2.data(),
                                      0.5 * dt, dim);
        
        // k3 = -i * H2 * buf (H2を再利用)
        julia_killer_sparse_matvec(H_values.data(), H0_indices, H0_indptr,
                                  buf.data(), k3.data(), dim);
        
        // buf = psi + dt*k3
        julia_killer_vector_add_scaled(buf.data(), psi_vec.data(), k3.data(),
                                      dt, dim);
        
        // H4 = H0 + ex4*mux + ey4*muy (k4計算用)
        julia_killer::optimized_complex_matrix_update(
            H_values.data(), H0_data, mux_data, muy_data, ex4, ey4, nnz);
        
        // k4 = -i * H4 * buf
        julia_killer_sparse_matvec(H_values.data(), H0_indices, H0_indptr,
                                  buf.data(), k4.data(), dim);
        
        // 🚀 Phase 1: SIMD最適化されたRK4ベクトル更新
        // psi += (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
        julia_killer::optimized_rk4_vector_update(
            psi_vec.data(), k1.data(), k2.data(), k3.data(), k4.data(), dt, dim);
        
        // 正規化（必要に応じて）
        if (renorm) {
            julia_killer_vector_normalize(psi_vec.data(), dim);
        }
        
        // 軌道の保存（必要に応じて）
        if (return_traj && (s % stride == 0)) {
            // std::vectorからEigenへの効率的なコピー
            std::copy(psi_vec.begin(), psi_vec.end(), 
                     result.row(traj_idx).data());
            traj_idx++;
        }
    }
    
    // 最終結果の保存
    if (!return_traj) {
        std::copy(psi_vec.begin(), psi_vec.end(), result.row(0).data());
    }
    
    return result;
}

/**
 * SIMD最適化されたスパース行列-ベクトル積
 * -i * H * x の計算を高速化
 */
void julia_killer_sparse_matvec(
    const std::complex<double>* H_values,
    const int* H_indices,
    const int* H_indptr,
    const std::complex<double>* x,
    std::complex<double>* y,
    int dim) {
    
    const std::complex<double> minus_i(0.0, -1.0);
    
    // ゼロ初期化
    std::fill_n(y, dim, std::complex<double>(0.0, 0.0));
    
    // CSR形式のスパース行列-ベクトル積
    #pragma omp parallel for if(dim > 256) schedule(static)
    for (int row = 0; row < dim; ++row) {
        std::complex<double> sum(0.0, 0.0);
        
        const int start = H_indptr[row];
        const int end = H_indptr[row + 1];
        
        // 内積計算（SIMDによる最適化は複雑なため、まずはスカラー実装）
        for (int j = start; j < end; ++j) {
            const int col = H_indices[j];
            sum += H_values[j] * x[col];
        }
        
        y[row] = minus_i * sum;
    }
}

/**
 * SIMD最適化されたベクトル演算
 * result = a + scale * b
 */
void julia_killer_vector_add_scaled(
    std::complex<double>* result,
    const std::complex<double>* a,
    const std::complex<double>* b,
    double scale,
    int dim) {
    
#ifdef SIMD_COMPLEX_AVX512
    const __m512d scale_vec = _mm512_set1_pd(scale);
    size_t simd_count = (dim / 4) * 4;
    
    for (size_t i = 0; i < simd_count; i += 4) {
        __m512d a_vec = _mm512_loadu_pd(reinterpret_cast<const double*>(&a[i]));
        __m512d b_vec = _mm512_loadu_pd(reinterpret_cast<const double*>(&b[i]));
        __m512d scaled_b = _mm512_mul_pd(b_vec, scale_vec);
        __m512d res = _mm512_add_pd(a_vec, scaled_b);
        _mm512_storeu_pd(reinterpret_cast<double*>(&result[i]), res);
    }
    
    // 残り要素の処理
    for (size_t i = simd_count; i < dim; ++i) {
        result[i] = a[i] + scale * b[i];
    }
    
#elif defined(SIMD_COMPLEX_AVX2)
    const __m256d scale_vec = _mm256_set1_pd(scale);
    size_t simd_count = (dim / 2) * 2;
    
    for (size_t i = 0; i < simd_count; i += 2) {
        __m256d a_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&a[i]));
        __m256d b_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&b[i]));
        __m256d scaled_b = _mm256_mul_pd(b_vec, scale_vec);
        __m256d res = _mm256_add_pd(a_vec, scaled_b);
        _mm256_storeu_pd(reinterpret_cast<double*>(&result[i]), res);
    }
    
    // 残り要素の処理
    for (size_t i = simd_count; i < dim; ++i) {
        result[i] = a[i] + scale * b[i];
    }
    
#else
    // フォールバック: スカラー実装
    for (int i = 0; i < dim; ++i) {
        result[i] = a[i] + scale * b[i];
    }
#endif
}

/**
 * SIMD最適化されたベクトル正規化
 */
void julia_killer_vector_normalize(std::complex<double>* vec, int dim) {
    // ノルムの計算
    double norm_sq = 0.0;
    
#ifdef SIMD_COMPLEX_AVX512
    __m512d norm_vec = _mm512_setzero_pd();
    size_t simd_count = (dim / 4) * 4;
    
    for (size_t i = 0; i < simd_count; i += 4) {
        __m512d v = _mm512_loadu_pd(reinterpret_cast<const double*>(&vec[i]));
        __m512d v_sq = _mm512_mul_pd(v, v);
        norm_vec = _mm512_add_pd(norm_vec, v_sq);
    }
    
    // 水平加算
    double temp[8];
    _mm512_storeu_pd(temp, norm_vec);
    for (int i = 0; i < 8; ++i) {
        norm_sq += temp[i];
    }
    
    // 残り要素の処理
    for (size_t i = simd_count; i < dim; ++i) {
        norm_sq += std::norm(vec[i]);
    }
    
#else
    // フォールバック: スカラー実装
    for (int i = 0; i < dim; ++i) {
        norm_sq += std::norm(vec[i]);
    }
#endif
    
    const double norm = std::sqrt(norm_sq);
    if (norm > 1e-10) {
        const double inv_norm = 1.0 / norm;
        
#ifdef SIMD_COMPLEX_AVX512
        const __m512d inv_norm_vec = _mm512_set1_pd(inv_norm);
        size_t simd_count = (dim / 4) * 4;
        
        for (size_t i = 0; i < simd_count; i += 4) {
            __m512d v = _mm512_loadu_pd(reinterpret_cast<const double*>(&vec[i]));
            __m512d normalized = _mm512_mul_pd(v, inv_norm_vec);
            _mm512_storeu_pd(reinterpret_cast<double*>(&vec[i]), normalized);
        }
        
        // 残り要素の処理
        for (size_t i = simd_count; i < dim; ++i) {
            vec[i] *= inv_norm;
        }
        
#else
        // フォールバック: スカラー実装
        for (int i = 0; i < dim; ++i) {
            vec[i] *= inv_norm;
        }
#endif
    }
}

} // namespace excitation_rk4_sparse 