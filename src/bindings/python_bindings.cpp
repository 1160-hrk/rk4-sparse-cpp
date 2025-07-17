#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "excitation_rk4_sparse/core.hpp"
#include "excitation_rk4_sparse/suitesparse.hpp"
#include "excitation_rk4_sparse/benchmark.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
using namespace excitation_rk4_sparse;
using cplx = std::complex<double>;

// 共通ユーティリティ関数
namespace detail {
    inline void ensure_vector_1d(const py::buffer_info& buf,
                                 const char* name) {
        if (buf.ndim != 1)
            throw py::value_error(std::string(name) + " must be 1-D");
    }
    inline void ensure_positive(double v, const char* name) {
        if (v <= 0.0)
            throw py::value_error(std::string(name) + " must be > 0");
    }
    inline void ensure_positive(int v, const char* name) {
        if (v <= 0)
            throw py::value_error(std::string(name) + " must be > 0");
    }
}  // namespace detail

// 最適化されたCSRデータからEigenの疎行列を構築するヘルパー関数
Eigen::SparseMatrix<std::complex<double>> build_sparse_matrix_from_scipy(
    const py::object& scipy_sparse_matrix)
{
    /* ★ 型チェックを強化：isspmatrix_csr を呼び出す */
    if (!py::module_::import("scipy.sparse").attr("isspmatrix_csr")(scipy_sparse_matrix)
            .cast<bool>())
        throw py::type_error("matrix must be scipy.sparse.csr_matrix");

    // scipy.sparseの行列からデータを取得
    py::array_t<std::complex<double>> data = scipy_sparse_matrix.attr("data").cast<py::array_t<std::complex<double>>>();
    py::array_t<int> indices = scipy_sparse_matrix.attr("indices").cast<py::array_t<int>>();
    py::array_t<int> indptr = scipy_sparse_matrix.attr("indptr").cast<py::array_t<int>>();
    int rows = scipy_sparse_matrix.attr("shape").attr("__getitem__")(0).cast<int>();
    int cols = scipy_sparse_matrix.attr("shape").attr("__getitem__")(1).cast<int>();

    /* ★ indptr[-1] と data.size を検証 */
    if (indptr.size() != rows + 1 ||
        indptr.at(indptr.size() - 1) != data.size())
        throw py::value_error("inconsistent CSR structure");

    // 最適化：直接CSR形式でEigen行列を構築
    Eigen::SparseMatrix<std::complex<double>> mat(rows, cols);
    mat.reserve(data.size());
    
    auto data_ptr = static_cast<std::complex<double>*>(data.request().ptr);
    auto indices_ptr = static_cast<int*>(indices.request().ptr);
    auto indptr_ptr = static_cast<int*>(indptr.request().ptr);
    
    // 最適化された構築：Triplet形式を避けて直接CSR形式を使用
    for (int i = 0; i < rows; ++i) {
        for (int j = indptr_ptr[i]; j < indptr_ptr[i + 1]; ++j) {
            mat.insert(i, indices_ptr[j]) = data_ptr[j];
        }
    }
    
    mat.makeCompressed();
    return mat;
}

// Phase 1: 直接CSR形式での処理を実装
// データ変換オーバーヘッドを完全に回避する新しい関数
Eigen::MatrixXcd rk4_sparse_eigen_direct_csr(
    const py::array_t<std::complex<double>>& H0_data,
    const py::array_t<int>& H0_indices,
    const py::array_t<int>& H0_indptr,
    const py::array_t<std::complex<double>>& mux_data,
    const py::array_t<int>& mux_indices,
    const py::array_t<int>& mux_indptr,
    const py::array_t<std::complex<double>>& muy_data,
    const py::array_t<int>& muy_indices,
    const py::array_t<int>& muy_indptr,
    const py::array_t<double>& Ex,
    const py::array_t<double>& Ey,
    const py::array_t<cplx>& psi0,
    double dt,
    bool return_traj,
    int stride,
    bool renorm)
{
    // 入力チェック
    if (H0_data.ndim() != 1 || mux_data.ndim() != 1 || muy_data.ndim() != 1) {
        throw std::runtime_error("Matrix data must be 1D arrays");
    }
    if (H0_indices.ndim() != 1 || mux_indices.ndim() != 1 || muy_indices.ndim() != 1) {
        throw std::runtime_error("Matrix indices must be 1D arrays");
    }
    if (H0_indptr.ndim() != 1 || mux_indptr.ndim() != 1 || muy_indptr.ndim() != 1) {
        throw std::runtime_error("Matrix indptr must be 1D arrays");
    }
    
    // バッファ情報の取得
    py::buffer_info H0_data_buf = H0_data.request();
    py::buffer_info H0_indices_buf = H0_indices.request();
    py::buffer_info H0_indptr_buf = H0_indptr.request();
    py::buffer_info mux_data_buf = mux_data.request();
    py::buffer_info mux_indices_buf = mux_indices.request();
    py::buffer_info mux_indptr_buf = mux_indptr.request();
    py::buffer_info muy_data_buf = muy_data.request();
    py::buffer_info muy_indices_buf = muy_indices.request();
    py::buffer_info muy_indptr_buf = muy_indptr.request();
    py::buffer_info Ex_buf = Ex.request();
    py::buffer_info Ey_buf = Ey.request();
    py::buffer_info psi0_buf = psi0.request();
    
    // データポインタの取得
    auto H0_data_ptr = static_cast<std::complex<double>*>(H0_data_buf.ptr);
    auto H0_indices_ptr = static_cast<int*>(H0_indices_buf.ptr);
    auto H0_indptr_ptr = static_cast<int*>(H0_indptr_buf.ptr);
    auto mux_data_ptr = static_cast<std::complex<double>*>(mux_data_buf.ptr);
    auto mux_indices_ptr = static_cast<int*>(mux_indices_buf.ptr);
    auto mux_indptr_ptr = static_cast<int*>(mux_indptr_buf.ptr);
    auto muy_data_ptr = static_cast<std::complex<double>*>(muy_data_buf.ptr);
    auto muy_indices_ptr = static_cast<int*>(muy_indices_buf.ptr);
    auto muy_indptr_ptr = static_cast<int*>(muy_indptr_buf.ptr);
    
    // 行列サイズの取得
    int rows = H0_indptr_buf.shape[0] - 1;
    int cols = psi0_buf.shape[0];
    
    // 電場とpsi0の変換
    Eigen::Map<const Eigen::VectorXd> Ex_vec(static_cast<double*>(Ex_buf.ptr), Ex_buf.shape[0]);
    Eigen::Map<const Eigen::VectorXd> Ey_vec(static_cast<double*>(Ey_buf.ptr), Ey_buf.shape[0]);
    Eigen::Map<const Eigen::VectorXcd> psi0_vec(static_cast<cplx*>(psi0_buf.ptr), psi0_buf.shape[0]);
    
    // 直接CSR形式での最適化された処理
    // 注意: この実装は既存のrk4_sparse_eigen関数を呼び出しますが、
    // 将来的には完全に直接CSR形式での処理に置き換える予定です
    
    // 一時的にEigen形式に変換（将来的には削除予定）
    Eigen::SparseMatrix<cplx> H0_mat(rows, cols);
    Eigen::SparseMatrix<cplx> mux_mat(rows, cols);
    Eigen::SparseMatrix<cplx> muy_mat(rows, cols);
    
    // 最適化された構築
    H0_mat.reserve(H0_data_buf.shape[0]);
    mux_mat.reserve(mux_data_buf.shape[0]);
    muy_mat.reserve(muy_data_buf.shape[0]);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = H0_indptr_ptr[i]; j < H0_indptr_ptr[i + 1]; ++j) {
            H0_mat.insert(i, H0_indices_ptr[j]) = H0_data_ptr[j];
        }
        for (int j = mux_indptr_ptr[i]; j < mux_indptr_ptr[i + 1]; ++j) {
            mux_mat.insert(i, mux_indices_ptr[j]) = mux_data_ptr[j];
        }
        for (int j = muy_indptr_ptr[i]; j < muy_indptr_ptr[i + 1]; ++j) {
            muy_mat.insert(i, muy_indices_ptr[j]) = muy_data_ptr[j];
        }
    }
    
    H0_mat.makeCompressed();
    mux_mat.makeCompressed();
    muy_mat.makeCompressed();
    
    // 既存の最適化された関数を呼び出し
    return rk4_sparse_eigen(
        H0_mat, mux_mat, muy_mat,
        Ex_vec, Ey_vec,
        psi0_vec,
        dt, return_traj, stride, renorm
    );
}

PYBIND11_MODULE(_rk4_sparse_cpp, m) {
    m.doc() = "Sparse matrix RK4 propagator for excitation dynamics (C++ implementation)";
    
    py::class_<PerformanceMetrics>(m, "PerformanceMetrics")
        .def_readonly("matrix_update_time", &PerformanceMetrics::matrix_update_time)
        .def_readonly("rk4_step_time", &PerformanceMetrics::rk4_step_time)
        .def_readonly("matrix_updates", &PerformanceMetrics::matrix_updates)
        .def_readonly("rk4_steps", &PerformanceMetrics::rk4_steps);
    
    // Phase 1: 最適化された直接CSR形式のバインディング
    m.def("rk4_sparse_eigen_direct_csr", &rk4_sparse_eigen_direct_csr,
        py::arg("H0_data"),
        py::arg("H0_indices"),
        py::arg("H0_indptr"),
        py::arg("mux_data"),
        py::arg("mux_indices"),
        py::arg("mux_indptr"),
        py::arg("muy_data"),
        py::arg("muy_indices"),
        py::arg("muy_indptr"),
        py::arg("Ex"),
        py::arg("Ey"),
        py::arg("psi0"),
        py::arg("dt"),
        py::arg("return_traj"),
        py::arg("stride"),
        py::arg("renorm"),
        "Optimized RK4 implementation with direct CSR format processing"
    );
    
    m.def("rk4_sparse_eigen", [](
        const py::object& H0,
        const py::object& mux,
        const py::object& muy,
        py::array_t<double,
            py::array::c_style | py::array::forcecast> Ex,
        py::array_t<double,
            py::array::c_style | py::array::forcecast> Ey,
        py::array_t<cplx,
            py::array::c_style | py::array::forcecast> psi0,
        double dt,
        bool return_traj,
        int stride,
        bool renorm
    ) {
        // 入力チェック
        if (!py::hasattr(H0, "data") || !py::hasattr(H0, "indices") || !py::hasattr(H0, "indptr")) {
            throw std::runtime_error("H0 must be a scipy.sparse.csr_matrix");
        }
        if (!py::hasattr(mux, "data") || !py::hasattr(mux, "indices") || !py::hasattr(mux, "indptr")) {
            throw std::runtime_error("mux must be a scipy.sparse.csr_matrix");
        }
        if (!py::hasattr(muy, "data") || !py::hasattr(muy, "indices") || !py::hasattr(muy, "indptr")) {
            throw std::runtime_error("muy must be a scipy.sparse.csr_matrix");
        }

        // バッファ情報の取得
        py::buffer_info Ex_buf = Ex.request();
        py::buffer_info Ey_buf = Ey.request();
        py::buffer_info psi0_buf = psi0.request();

        // 入力チェック
        if (psi0_buf.ndim != 1) {
            throw std::runtime_error("psi0 must be a 1D array");
        }

        // 最適化されたCSR行列の構築
        Eigen::SparseMatrix<cplx> H0_mat = build_sparse_matrix_from_scipy(H0);
        Eigen::SparseMatrix<cplx> mux_mat = build_sparse_matrix_from_scipy(mux);
        Eigen::SparseMatrix<cplx> muy_mat = build_sparse_matrix_from_scipy(muy);

        // 電場とpsi0の変換
        Eigen::Map<const Eigen::VectorXd> Ex_vec(static_cast<double*>(Ex_buf.ptr), Ex_buf.shape[0]);
        Eigen::Map<const Eigen::VectorXd> Ey_vec(static_cast<double*>(Ey_buf.ptr), Ey_buf.shape[0]);
        Eigen::Map<const Eigen::VectorXcd> psi0_vec(static_cast<cplx*>(psi0_buf.ptr), psi0_buf.shape[0]);

        // rk4_sparse_eigenの呼び出し
        return rk4_sparse_eigen(
            H0_mat, mux_mat, muy_mat,
            Ex_vec, Ey_vec,
            psi0_vec,
            dt, return_traj, stride, renorm
        );
    },
    py::arg("H0"),
    py::arg("mux"),
    py::arg("muy"),
    py::arg("Ex"),
    py::arg("Ey"),
    py::arg("psi0"),
    py::arg("dt"),
    py::arg("return_traj"),
    py::arg("stride"),
    py::arg("renorm")
    );
    
    // OpenBLAS + SuiteSparse版のバインディング（利用可能な場合）
    #ifdef OPENBLAS_SUITESPARSE_AVAILABLE
    m.def("rk4_sparse_suitesparse", [](
        const py::object& H0,
        const py::object& mux,
        const py::object& muy,
        py::array_t<double,
            py::array::c_style | py::array::forcecast> Ex,
        py::array_t<double,
            py::array::c_style | py::array::forcecast> Ey,
        py::array_t<cplx,
            py::array::c_style | py::array::forcecast> psi0,
        double dt,
        bool return_traj,
        int stride,
        bool renorm,
        int level // 0: BASIC, 1: STANDARD, 2: ENHANCED
    ) {
        // 入力チェック
        if (!py::hasattr(H0, "data") || !py::hasattr(H0, "indices") || !py::hasattr(H0, "indptr")) {
            throw std::runtime_error("H0 must be a scipy.sparse.csr_matrix");
        }
        if (!py::hasattr(mux, "data") || !py::hasattr(mux, "indices") || !py::hasattr(mux, "indptr")) {
            throw std::runtime_error("mux must be a scipy.sparse.csr_matrix");
        }
        if (!py::hasattr(muy, "data") || !py::hasattr(muy, "indices") || !py::hasattr(muy, "indptr")) {
            throw std::runtime_error("muy must be a scipy.sparse.csr_matrix");
        }

        // バッファ情報の取得
        py::buffer_info Ex_buf = Ex.request();
        py::buffer_info Ey_buf = Ey.request();
        py::buffer_info psi0_buf = psi0.request();

        // 入力チェック
        if (psi0_buf.ndim != 1) {
            throw std::runtime_error("psi0 must be a 1D array");
        }

        // 最適化されたCSR行列の構築
        Eigen::SparseMatrix<cplx> H0_mat = build_sparse_matrix_from_scipy(H0);
        Eigen::SparseMatrix<cplx> mux_mat = build_sparse_matrix_from_scipy(mux);
        Eigen::SparseMatrix<cplx> muy_mat = build_sparse_matrix_from_scipy(muy);

        // 電場とpsi0の変換
        Eigen::Map<const Eigen::VectorXd> Ex_vec(static_cast<double*>(Ex_buf.ptr), Ex_buf.shape[0]);
        Eigen::Map<const Eigen::VectorXd> Ey_vec(static_cast<double*>(Ey_buf.ptr), Ey_buf.shape[0]);
        Eigen::Map<const Eigen::VectorXcd> psi0_vec(static_cast<cplx*>(psi0_buf.ptr), psi0_buf.shape[0]);

        // SuiteSparse版の呼び出し
        return rk4_sparse_suitesparse(
            H0_mat, mux_mat, muy_mat,
            Ex_vec, Ey_vec,
            psi0_vec,
            dt, return_traj, stride, renorm, static_cast<OptimizationLevel>(level)
        );
    },
    py::arg("H0"),
    py::arg("mux"),
    py::arg("muy"),
    py::arg("Ex"),
    py::arg("Ey"),
    py::arg("psi0"),
    py::arg("dt"),
    py::arg("return_traj"),
    py::arg("stride"),
    py::arg("renorm"),
    py::arg("level")
    );
    #endif
    
    m.def("get_omp_max_threads", []() {
        #ifdef _OPENMP
        return omp_get_max_threads();
        #else
        return 1;
        #endif
    }, "Get the maximum number of OpenMP threads");
    
    // ベンチマーク実装関数
    m.def("benchmark_implementations", [](
        const py::object& H0,
        const py::object& mux,
        const py::object& muy,
        py::array_t<double,
            py::array::c_style | py::array::forcecast> Ex,
        py::array_t<double,
            py::array::c_style | py::array::forcecast> Ey,
        py::array_t<cplx,
            py::array::c_style | py::array::forcecast> psi0,
        double dt,
        int num_steps,
        bool return_traj,
        int stride,
        bool renorm
    ) {
        // 入力チェック
        if (!py::hasattr(H0, "data") || !py::hasattr(H0, "indices") || !py::hasattr(H0, "indptr")) {
            throw std::runtime_error("H0 must be a scipy.sparse.csr_matrix");
        }
        if (!py::hasattr(mux, "data") || !py::hasattr(mux, "indices") || !py::hasattr(mux, "indptr")) {
            throw std::runtime_error("mux must be a scipy.sparse.csr_matrix");
        }
        if (!py::hasattr(muy, "data") || !py::hasattr(muy, "indices") || !py::hasattr(muy, "indptr")) {
            throw std::runtime_error("muy must be a scipy.sparse.csr_matrix");
        }

        // バッファ情報の取得
        py::buffer_info Ex_buf = Ex.request();
        py::buffer_info Ey_buf = Ey.request();
        py::buffer_info psi0_buf = psi0.request();

        // 入力チェック
        if (psi0_buf.ndim != 1) {
            throw std::runtime_error("psi0 must be a 1D array");
        }

        // 最適化されたCSR行列の構築
        Eigen::SparseMatrix<cplx> H0_mat = build_sparse_matrix_from_scipy(H0);
        Eigen::SparseMatrix<cplx> mux_mat = build_sparse_matrix_from_scipy(mux);
        Eigen::SparseMatrix<cplx> muy_mat = build_sparse_matrix_from_scipy(muy);

        // 電場とpsi0の変換
        Eigen::Map<const Eigen::VectorXd> Ex_vec(static_cast<double*>(Ex_buf.ptr), Ex_buf.shape[0]);
        Eigen::Map<const Eigen::VectorXd> Ey_vec(static_cast<double*>(Ey_buf.ptr), Ey_buf.shape[0]);
        Eigen::Map<const Eigen::VectorXcd> psi0_vec(static_cast<cplx*>(psi0_buf.ptr), psi0_buf.shape[0]);

        // ベンチマーク実行
        return benchmark_implementations(
            H0_mat, mux_mat, muy_mat,
            Ex_vec, Ey_vec,
            psi0_vec,
            dt, num_steps, return_traj, stride, renorm
        );
    },
    py::arg("H0"),
    py::arg("mux"),
    py::arg("muy"),
    py::arg("Ex"),
    py::arg("Ey"),
    py::arg("psi0"),
    py::arg("dt"),
    py::arg("num_steps"),
    py::arg("return_traj"),
    py::arg("stride"),
    py::arg("renorm")
    );
} 