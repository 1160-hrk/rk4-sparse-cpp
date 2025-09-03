#pragma once

#include <complex>
#include <cstddef>

// SIMD命令セットの検出
#if defined(__AVX512F__)
    #define SIMD_COMPLEX_AVX512
    #include <immintrin.h>
#elif defined(__AVX2__)
    #define SIMD_COMPLEX_AVX2
    #include <immintrin.h>
#elif defined(__ARM_NEON)
    #define SIMD_COMPLEX_NEON
    #include <arm_neon.h>
#endif

namespace julia_killer {

/**
 * Julia @turbo マクロに対抗するSIMD最適化
 * H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i]
 * の計算を高速化
 */

#ifdef SIMD_COMPLEX_AVX512
/**
 * AVX512による8つの複素数同時処理
 * Juliaの2-4倍のSIMD幅で処理
 */
inline void avx512_complex_matrix_update(
    std::complex<double>* __restrict__ H_values,
    const std::complex<double>* __restrict__ H0_data,
    const std::complex<double>* __restrict__ mux_data,
    const std::complex<double>* __restrict__ muy_data,
    double ex, double ey, size_t count) {
    
    // スカラー値をベクトルに展開
    const __m512d ex_vec = _mm512_set1_pd(ex);
    const __m512d ey_vec = _mm512_set1_pd(ey);
    
    // 8つの複素数 = 16個のdouble = 2つの__m512dレジスタ
    size_t simd_count = (count / 4) * 4;  // 4つの複素数ずつ処理
    
    for (size_t i = 0; i < simd_count; i += 4) {
        // H0の読み込み（4つの複素数 = 8個のdouble）
        __m512d h0_vec = _mm512_loadu_pd(reinterpret_cast<const double*>(&H0_data[i]));
        
        // muxの読み込みと複素数スケーリング
        __m512d mux_vec = _mm512_loadu_pd(reinterpret_cast<const double*>(&mux_data[i]));
        
        // 複素数乗算のために実部・虚部を分離処理
        // mux * ex: (a+bi) * ex = ax + bxi
        __m512d mux_real = _mm512_mul_pd(mux_vec, ex_vec);
        
        // muyの読み込みと複素数スケーリング
        __m512d muy_vec = _mm512_loadu_pd(reinterpret_cast<const double*>(&muy_data[i]));
        __m512d muy_real = _mm512_mul_pd(muy_vec, ey_vec);
        
        // 最終計算: H0 + ex*mux + ey*muy
        __m512d result = _mm512_add_pd(_mm512_add_pd(h0_vec, mux_real), muy_real);
        
        // 結果の格納
        _mm512_storeu_pd(reinterpret_cast<double*>(&H_values[i]), result);
    }
    
    // 残り要素の処理
    for (size_t i = simd_count; i < count; ++i) {
        H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
    }
}

/**
 * AVX512によるRK4ベクトル更新
 * psi += (dt/6) * (k1 + 2*k2 + 2*k3 + k4)
 */
inline void avx512_rk4_vector_update(
    std::complex<double>* __restrict__ psi,
    const std::complex<double>* __restrict__ k1,
    const std::complex<double>* __restrict__ k2,
    const std::complex<double>* __restrict__ k3,
    const std::complex<double>* __restrict__ k4,
    double dt, size_t dim) {
    
    const __m512d dt_sixth = _mm512_set1_pd(dt / 6.0);
    const __m512d two = _mm512_set1_pd(2.0);
    
    size_t simd_count = (dim / 4) * 4;
    
    for (size_t i = 0; i < simd_count; i += 4) {
        __m512d psi_vec = _mm512_loadu_pd(reinterpret_cast<const double*>(&psi[i]));
        __m512d k1_vec = _mm512_loadu_pd(reinterpret_cast<const double*>(&k1[i]));
        __m512d k2_vec = _mm512_loadu_pd(reinterpret_cast<const double*>(&k2[i]));
        __m512d k3_vec = _mm512_loadu_pd(reinterpret_cast<const double*>(&k3[i]));
        __m512d k4_vec = _mm512_loadu_pd(reinterpret_cast<const double*>(&k4[i]));
        
        // k1 + 2*k2 + 2*k3 + k4
        __m512d k_sum = _mm512_add_pd(k1_vec, k4_vec);
        k_sum = _mm512_add_pd(k_sum, _mm512_mul_pd(two, k2_vec));
        k_sum = _mm512_add_pd(k_sum, _mm512_mul_pd(two, k3_vec));
        
        // psi += dt/6 * k_sum
        __m512d update = _mm512_mul_pd(dt_sixth, k_sum);
        __m512d result = _mm512_add_pd(psi_vec, update);
        
        _mm512_storeu_pd(reinterpret_cast<double*>(&psi[i]), result);
    }
    
    // 残り要素の処理
    for (size_t i = simd_count; i < dim; ++i) {
        psi[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

#elif defined(SIMD_COMPLEX_AVX2)
/**
 * AVX2による4つの複素数同時処理
 * AVX512が利用できない場合のフォールバック
 */
inline void avx2_complex_matrix_update(
    std::complex<double>* __restrict__ H_values,
    const std::complex<double>* __restrict__ H0_data,
    const std::complex<double>* __restrict__ mux_data,
    const std::complex<double>* __restrict__ muy_data,
    double ex, double ey, size_t count) {
    
    const __m256d ex_vec = _mm256_set1_pd(ex);
    const __m256d ey_vec = _mm256_set1_pd(ey);
    
    size_t simd_count = (count / 2) * 2;  // 2つの複素数ずつ処理
    
    for (size_t i = 0; i < simd_count; i += 2) {
        __m256d h0_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&H0_data[i]));
        __m256d mux_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&mux_data[i]));
        __m256d muy_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&muy_data[i]));
        
        __m256d mux_scaled = _mm256_mul_pd(mux_vec, ex_vec);
        __m256d muy_scaled = _mm256_mul_pd(muy_vec, ey_vec);
        
        __m256d result = _mm256_add_pd(_mm256_add_pd(h0_vec, mux_scaled), muy_scaled);
        
        _mm256_storeu_pd(reinterpret_cast<double*>(&H_values[i]), result);
    }
    
    for (size_t i = simd_count; i < count; ++i) {
        H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
    }
}

inline void avx2_rk4_vector_update(
    std::complex<double>* __restrict__ psi,
    const std::complex<double>* __restrict__ k1,
    const std::complex<double>* __restrict__ k2,
    const std::complex<double>* __restrict__ k3,
    const std::complex<double>* __restrict__ k4,
    double dt, size_t dim) {
    
    const __m256d dt_sixth = _mm256_set1_pd(dt / 6.0);
    const __m256d two = _mm256_set1_pd(2.0);
    
    size_t simd_count = (dim / 2) * 2;
    
    for (size_t i = 0; i < simd_count; i += 2) {
        __m256d psi_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&psi[i]));
        __m256d k1_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&k1[i]));
        __m256d k2_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&k2[i]));
        __m256d k3_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&k3[i]));
        __m256d k4_vec = _mm256_loadu_pd(reinterpret_cast<const double*>(&k4[i]));
        
        __m256d k_sum = _mm256_add_pd(k1_vec, k4_vec);
        k_sum = _mm256_add_pd(k_sum, _mm256_mul_pd(two, k2_vec));
        k_sum = _mm256_add_pd(k_sum, _mm256_mul_pd(two, k3_vec));
        
        __m256d update = _mm256_mul_pd(dt_sixth, k_sum);
        __m256d result = _mm256_add_pd(psi_vec, update);
        
        _mm256_storeu_pd(reinterpret_cast<double*>(&psi[i]), result);
    }
    
    for (size_t i = simd_count; i < dim; ++i) {
        psi[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

#endif

/**
 * アーキテクチャ自動選択関数
 * コンパイル時に最適なSIMD実装を選択
 */
inline void optimized_complex_matrix_update(
    std::complex<double>* H_values,
    const std::complex<double>* H0_data,
    const std::complex<double>* mux_data,
    const std::complex<double>* muy_data,
    double ex, double ey, size_t count) {
    
#ifdef SIMD_COMPLEX_AVX512
    avx512_complex_matrix_update(H_values, H0_data, mux_data, muy_data, ex, ey, count);
#elif defined(SIMD_COMPLEX_AVX2)
    avx2_complex_matrix_update(H_values, H0_data, mux_data, muy_data, ex, ey, count);
#else
    // フォールバック: スカラー実装
    for (size_t i = 0; i < count; ++i) {
        H_values[i] = H0_data[i] + ex * mux_data[i] + ey * muy_data[i];
    }
#endif
}

inline void optimized_rk4_vector_update(
    std::complex<double>* psi,
    const std::complex<double>* k1,
    const std::complex<double>* k2,
    const std::complex<double>* k3,
    const std::complex<double>* k4,
    double dt, size_t dim) {
    
#ifdef SIMD_COMPLEX_AVX512
    avx512_rk4_vector_update(psi, k1, k2, k3, k4, dt, dim);
#elif defined(SIMD_COMPLEX_AVX2)
    avx2_rk4_vector_update(psi, k1, k2, k3, k4, dt, dim);
#else
    // フォールバック: スカラー実装
    for (size_t i = 0; i < dim; ++i) {
        psi[i] += (dt / 6.0) * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
#endif
}

} // namespace julia_killer 