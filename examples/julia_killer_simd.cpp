#include <Eigen/Sparse>
#include <complex>
#include <vector>
#include <chrono>
#include <iostream>

// SIMD関連のヘッダー
#if defined(__x86_64__) || defined(__i386__)
  #ifdef __AVX512F__
  #include <immintrin.h>
  #define SIMD_WIDTH 8  // AVX512: 8つの複素数
  #define USE_AVX512
  #elif defined(__AVX2__)
  #include <immintrin.h>
  #define SIMD_WIDTH 4  // AVX2: 4つの複素数
  #define USE_AVX2
  #elif defined(__SSE2__)
  #include <emmintrin.h>
  #define SIMD_WIDTH 2  // SSE2: 2つの複素数
  #define USE_SSE2
  #endif
#elif defined(__aarch64__) || defined(__arm__)
  #include <arm_neon.h>
  #define SIMD_WIDTH 2  // NEON: 2つの複素数
  #define USE_NEON
#endif

namespace julia_killer {

using cplx = std::complex<double>;

// ========================================
// Phase 1: Julia @turbo相当のSIMD最適化
// ========================================

#ifdef USE_AVX512
// AVX512による超高速複素数演算（8つの複素数を同時処理）
inline void avx512_complex_multiply_add(
    const cplx* __restrict__ a,    // H0_data
    const cplx* __restrict__ b,    // mux_data  
    const cplx* __restrict__ c,    // muy_data
    cplx* __restrict__ result,     // vals
    double scale_b,                // ex
    double scale_c,                // ey
    size_t count) {
    
    const __m512d scale_b_vec = _mm512_set1_pd(scale_b);
    const __m512d scale_c_vec = _mm512_set1_pd(scale_c);
    
    size_t simd_count = (count / 8) * 8;
    
    for (size_t i = 0; i < simd_count; i += 8) {
        // a, b, cの読み込み（8つの複素数 = 16のdouble）
        __m512d a_real = _mm512_loadu_pd(reinterpret_cast<const double*>(&a[i]));
        __m512d a_imag = _mm512_loadu_pd(reinterpret_cast<const double*>(&a[i]) + 8);
        
        __m512d b_real = _mm512_loadu_pd(reinterpret_cast<const double*>(&b[i]));
        __m512d b_imag = _mm512_loadu_pd(reinterpret_cast<const double*>(&b[i]) + 8);
        
        __m512d c_real = _mm512_loadu_pd(reinterpret_cast<const double*>(&c[i]));
        __m512d c_imag = _mm512_loadu_pd(reinterpret_cast<const double*>(&c[i]) + 8);
        
        // スケーリング: scale_b * b + scale_c * c
        __m512d b_scaled_real = _mm512_mul_pd(b_real, scale_b_vec);
        __m512d b_scaled_imag = _mm512_mul_pd(b_imag, scale_b_vec);
        
        __m512d c_scaled_real = _mm512_mul_pd(c_real, scale_c_vec);
        __m512d c_scaled_imag = _mm512_mul_pd(c_imag, scale_c_vec);
        
        // 結果計算: a + scale_b * b + scale_c * c
        __m512d result_real = _mm512_add_pd(_mm512_add_pd(a_real, b_scaled_real), c_scaled_real);
        __m512d result_imag = _mm512_add_pd(_mm512_add_pd(a_imag, b_scaled_imag), c_scaled_imag);
        
        // ストア
        _mm512_storeu_pd(reinterpret_cast<double*>(&result[i]), result_real);
        _mm512_storeu_pd(reinterpret_cast<double*>(&result[i]) + 8, result_imag);
    }
    
    // 残り要素の処理
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] + scale_b * b[i] + scale_c * c[i];
    }
}

// AVX512による複素数ベクトル演算（RK4ステップ用）
inline void avx512_rk4_vector_update(
    cplx* __restrict__ psi,
    const cplx* __restrict__ k1,
    const cplx* __restrict__ k2,
    const cplx* __restrict__ k3,
    const cplx* __restrict__ k4,
    double dt,
    size_t dim) {
    
    const double dt_over_6 = dt / 6.0;
    const __m512d dt6_vec = _mm512_set1_pd(dt_over_6);
    const __m512d two_vec = _mm512_set1_pd(2.0);
    
    size_t simd_count = (dim / 8) * 8;
    
    for (size_t i = 0; i < simd_count; i += 8) {
        // k1, k2, k3, k4の読み込み
        __m512d k1_real = _mm512_loadu_pd(reinterpret_cast<const double*>(&k1[i]));
        __m512d k1_imag = _mm512_loadu_pd(reinterpret_cast<const double*>(&k1[i]) + 8);
        
        __m512d k2_real = _mm512_loadu_pd(reinterpret_cast<const double*>(&k2[i]));
        __m512d k2_imag = _mm512_loadu_pd(reinterpret_cast<const double*>(&k2[i]) + 8);
        
        __m512d k3_real = _mm512_loadu_pd(reinterpret_cast<const double*>(&k3[i]));
        __m512d k3_imag = _mm512_loadu_pd(reinterpret_cast<const double*>(&k3[i]) + 8);
        
        __m512d k4_real = _mm512_loadu_pd(reinterpret_cast<const double*>(&k4[i]));
        __m512d k4_imag = _mm512_loadu_pd(reinterpret_cast<const double*>(&k4[i]) + 8);
        
        // 2*k2 + 2*k3の計算
        __m512d k2_scaled_real = _mm512_mul_pd(k2_real, two_vec);
        __m512d k2_scaled_imag = _mm512_mul_pd(k2_imag, two_vec);
        
        __m512d k3_scaled_real = _mm512_mul_pd(k3_real, two_vec);
        __m512d k3_scaled_imag = _mm512_mul_pd(k3_imag, two_vec);
        
        // k1 + 2*k2 + 2*k3 + k4
        __m512d sum_real = _mm512_add_pd(_mm512_add_pd(k1_real, k2_scaled_real), 
                                        _mm512_add_pd(k3_scaled_real, k4_real));
        __m512d sum_imag = _mm512_add_pd(_mm512_add_pd(k1_imag, k2_scaled_imag), 
                                        _mm512_add_pd(k3_scaled_imag, k4_imag));
        
        // dt/6でスケーリング
        __m512d increment_real = _mm512_mul_pd(sum_real, dt6_vec);
        __m512d increment_imag = _mm512_mul_pd(sum_imag, dt6_vec);
        
        // psiの更新
        __m512d psi_real = _mm512_loadu_pd(reinterpret_cast<const double*>(&psi[i]));
        __m512d psi_imag = _mm512_loadu_pd(reinterpret_cast<const double*>(&psi[i]) + 8);
        
        __m512d result_real = _mm512_add_pd(psi_real, increment_real);
        __m512d result_imag = _mm512_add_pd(psi_imag, increment_imag);
        
        _mm512_storeu_pd(reinterpret_cast<double*>(&psi[i]), result_real);
        _mm512_storeu_pd(reinterpret_cast<double*>(&psi[i]) + 8, result_imag);
    }
    
    // 残り要素
    const double dt_over_6_scalar = dt / 6.0;
    for (size_t i = simd_count; i < dim; ++i) {
        psi[i] += dt_over_6_scalar * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
    }
}

#elif defined(USE_AVX2)
// AVX2による高速複素数演算（4つの複素数を同時処理）
inline void avx2_complex_multiply_add(
    const cplx* __restrict__ a,
    const cplx* __restrict__ b,
    const cplx* __restrict__ c,
    cplx* __restrict__ result,
    double scale_b,
    double scale_c,
    size_t count) {
    
    const __m256d scale_b_vec = _mm256_set1_pd(scale_b);
    const __m256d scale_c_vec = _mm256_set1_pd(scale_c);
    
    size_t simd_count = (count / 4) * 4;
    
    for (size_t i = 0; i < simd_count; i += 4) {
        // aの読み込み（4つの複素数 = 8つのdouble）
        __m256d a_low = _mm256_loadu_pd(reinterpret_cast<const double*>(&a[i]));
        __m256d a_high = _mm256_loadu_pd(reinterpret_cast<const double*>(&a[i + 2]));
        
        // bの読み込みとスケーリング
        __m256d b_low = _mm256_loadu_pd(reinterpret_cast<const double*>(&b[i]));
        __m256d b_high = _mm256_loadu_pd(reinterpret_cast<const double*>(&b[i + 2]));
        
        __m256d b_scaled_low = _mm256_mul_pd(b_low, scale_b_vec);
        __m256d b_scaled_high = _mm256_mul_pd(b_high, scale_b_vec);
        
        // cの読み込みとスケーリング
        __m256d c_low = _mm256_loadu_pd(reinterpret_cast<const double*>(&c[i]));
        __m256d c_high = _mm256_loadu_pd(reinterpret_cast<const double*>(&c[i + 2]));
        
        __m256d c_scaled_low = _mm256_mul_pd(c_low, scale_c_vec);
        __m256d c_scaled_high = _mm256_mul_pd(c_high, scale_c_vec);
        
        // 結果計算とストア
        __m256d result_low = _mm256_add_pd(_mm256_add_pd(a_low, b_scaled_low), c_scaled_low);
        __m256d result_high = _mm256_add_pd(_mm256_add_pd(a_high, b_scaled_high), c_scaled_high);
        
        _mm256_storeu_pd(reinterpret_cast<double*>(&result[i]), result_low);
        _mm256_storeu_pd(reinterpret_cast<double*>(&result[i + 2]), result_high);
    }
    
    // 残り要素
    for (size_t i = simd_count; i < count; ++i) {
        result[i] = a[i] + scale_b * b[i] + scale_c * c[i];
    }
}

#else
// フォールバック実装（SIMD非対応環境）
inline void fallback_complex_multiply_add(
    const cplx* __restrict__ a,
    const cplx* __restrict__ b,
    const cplx* __restrict__ c,
    cplx* __restrict__ result,
    double scale_b,
    double scale_c,
    size_t count) {
    
    for (size_t i = 0; i < count; ++i) {
        result[i] = a[i] + scale_b * b[i] + scale_c * c[i];
    }
}
#endif

// ========================================
// Phase 2: キャッシュ最適化スパース行列演算
// ========================================

class OptimizedSparseMatrix {
public:
    struct Element {
        cplx H0_val;
        cplx mux_val;
        cplx muy_val;
        int row, col;
    };
    
    std::vector<Element> elements;
    std::vector<std::vector<size_t>> row_starts;
    int dim;
    
    OptimizedSparseMatrix(
        const Eigen::SparseMatrix<cplx>& H0,
        const Eigen::SparseMatrix<cplx>& mux,
        const Eigen::SparseMatrix<cplx>& muy) : dim(H0.rows()) {
        
        row_starts.resize(dim + 1);
        
        // 非ゼロ要素パターンを統合
        for (int k = 0; k < H0.outerSize(); ++k) {
            for (Eigen::SparseMatrix<cplx>::InnerIterator it(H0, k); it; ++it) {
                Element elem;
                elem.H0_val = it.value();
                elem.mux_val = mux.coeff(it.row(), it.col());
                elem.muy_val = muy.coeff(it.row(), it.col());
                elem.row = it.row();
                elem.col = it.col();
                elements.push_back(elem);
            }
        }
        
        // 行ごとのインデックス構築
        size_t current_row = 0;
        for (size_t i = 0; i < elements.size(); ++i) {
            while (current_row <= elements[i].row) {
                row_starts[current_row++] = i;
            }
        }
        while (current_row <= dim) {
            row_starts[current_row++] = elements.size();
        }
    }
    
    // 超高速スパース行列-ベクトル積（キャッシュ最適化）
    void multiply(const cplx* __restrict__ x, cplx* __restrict__ y, 
                 double ex, double ey) const {
        
        // ゼロ初期化
        std::fill(y, y + dim, cplx(0, 0));
        
        #ifdef _OPENMP
        #pragma omp parallel for schedule(static) if (dim > 64)
        #endif
        for (int row = 0; row < dim; ++row) {
            cplx sum(0, 0);
            size_t start = row_starts[row];
            size_t end = row_starts[row + 1];
            
            // プリフェッチ
            if (row + 1 < dim) {
                __builtin_prefetch(&elements[row_starts[row + 1]], 0, 3);
            }
            
            for (size_t i = start; i < end; ++i) {
                const Element& elem = elements[i];
                cplx H_val = elem.H0_val + ex * elem.mux_val + ey * elem.muy_val;
                sum += H_val * x[elem.col];
            }
            
            y[row] = cplx(0, -1) * sum;  // -i * sum
        }
    }
};

// ========================================
// Phase 3: Julia Killer RK4実装
// ========================================

Eigen::MatrixXcd julia_killer_rk4(
    const Eigen::SparseMatrix<cplx>& H0,
    const Eigen::SparseMatrix<cplx>& mux,
    const Eigen::SparseMatrix<cplx>& muy,
    const Eigen::VectorXd& Ex,
    const Eigen::VectorXd& Ey,
    const Eigen::VectorXcd& psi0,
    double dt,
    bool return_traj = true,
    int stride = 1) {
    
    const int dim = H0.rows();
    const int steps = (Ex.size() - 1) / 2;
    const int traj_size = return_traj ? (steps + stride - 1) / stride : 1;
    
    Eigen::MatrixXcd result(traj_size, dim);
    if (return_traj) {
        result.row(0) = psi0;
    }
    
    // キャッシュ最適化スパース行列の構築
    OptimizedSparseMatrix opt_matrix(H0, mux, muy);
    
    // アライメントされたワークバッファ
    alignas(64) std::vector<cplx> psi(psi0.data(), psi0.data() + dim);
    alignas(64) std::vector<cplx> buf(dim);
    alignas(64) std::vector<cplx> k1(dim);
    alignas(64) std::vector<cplx> k2(dim);
    alignas(64) std::vector<cplx> k3(dim);
    alignas(64) std::vector<cplx> k4(dim);
    
    // 電場データの3点セット変換
    std::vector<std::array<double, 3>> Ex3(steps), Ey3(steps);
    for (int s = 0; s < steps; ++s) {
        Ex3[s] = {Ex[2*s], Ex[2*s+1], Ex[2*s+2]};
        Ey3[s] = {Ey[2*s], Ey[2*s+1], Ey[2*s+2]};
    }
    
    int traj_idx = 1;
    
    // メインループ（Julia Killer最適化）
    for (int s = 0; s < steps; ++s) {
        const double ex1 = Ex3[s][0], ex2 = Ex3[s][1], ex4 = Ex3[s][2];
        const double ey1 = Ey3[s][0], ey2 = Ey3[s][1], ey4 = Ey3[s][2];
        
        // k1計算
        opt_matrix.multiply(psi.data(), k1.data(), ex1, ey1);
        
        // buf = psi + 0.5*dt*k1（SIMD最適化）
        const double half_dt = 0.5 * dt;
        #ifdef USE_AVX512
        for (size_t i = 0; i < dim; i += 8) {
            __m512d psi_real = _mm512_loadu_pd(reinterpret_cast<const double*>(&psi[i]));
            __m512d psi_imag = _mm512_loadu_pd(reinterpret_cast<const double*>(&psi[i]) + 8);
            
            __m512d k1_real = _mm512_loadu_pd(reinterpret_cast<const double*>(&k1[i]));
            __m512d k1_imag = _mm512_loadu_pd(reinterpret_cast<const double*>(&k1[i]) + 8);
            
            __m512d dt_vec = _mm512_set1_pd(half_dt);
            __m512d result_real = _mm512_add_pd(psi_real, _mm512_mul_pd(k1_real, dt_vec));
            __m512d result_imag = _mm512_add_pd(psi_imag, _mm512_mul_pd(k1_imag, dt_vec));
            
            _mm512_storeu_pd(reinterpret_cast<double*>(&buf[i]), result_real);
            _mm512_storeu_pd(reinterpret_cast<double*>(&buf[i]) + 8, result_imag);
        }
        #else
        for (int i = 0; i < dim; ++i) {
            buf[i] = psi[i] + half_dt * k1[i];
        }
        #endif
        
        // k2計算
        opt_matrix.multiply(buf.data(), k2.data(), ex2, ey2);
        
        // buf = psi + 0.5*dt*k2
        for (int i = 0; i < dim; ++i) {
            buf[i] = psi[i] + half_dt * k2[i];
        }
        
        // k3計算
        opt_matrix.multiply(buf.data(), k3.data(), ex2, ey2);
        
        // buf = psi + dt*k3
        for (int i = 0; i < dim; ++i) {
            buf[i] = psi[i] + dt * k3[i];
        }
        
        // k4計算
        opt_matrix.multiply(buf.data(), k4.data(), ex4, ey4);
        
        // 状態更新（超高速SIMD）
        #ifdef USE_AVX512
        avx512_rk4_vector_update(psi.data(), k1.data(), k2.data(), k3.data(), k4.data(), dt, dim);
        #elif defined(USE_AVX2)
        // AVX2版も実装可能
        const double dt_over_6 = dt / 6.0;
        for (int i = 0; i < dim; ++i) {
            psi[i] += dt_over_6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        #else
        const double dt_over_6 = dt / 6.0;
        for (int i = 0; i < dim; ++i) {
            psi[i] += dt_over_6 * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i]);
        }
        #endif
        
        // 軌跡の保存
        if (return_traj && (s % stride == 0)) {
            for (int i = 0; i < dim; ++i) {
                result(traj_idx, i) = psi[i];
            }
            traj_idx++;
        }
    }
    
    if (!return_traj) {
        for (int i = 0; i < dim; ++i) {
            result(0, i) = psi[i];
        }
    }
    
    return result;
}

} // namespace julia_killer 