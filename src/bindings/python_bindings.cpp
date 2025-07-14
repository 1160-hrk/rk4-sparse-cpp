#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "excitation_rk4_sparse/core.hpp"
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

// CSRデータからEigenの疎行列を構築するヘルパー関数
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

    // Eigen形式の疎行列を構築
    Eigen::SparseMatrix<std::complex<double>> mat(rows, cols);
    std::vector<Eigen::Triplet<std::complex<double>>> triplets;
    
    auto data_ptr = static_cast<std::complex<double>*>(data.request().ptr);
    auto indices_ptr = static_cast<int*>(indices.request().ptr);
    auto indptr_ptr = static_cast<int*>(indptr.request().ptr);
    
    for (int i = 0; i < rows; ++i) {
        for (int j = indptr_ptr[i]; j < indptr_ptr[i + 1]; ++j) {
            triplets.emplace_back(i, indices_ptr[j], data_ptr[j]);
        }
    }
    
    mat.setFromTriplets(triplets.begin(), triplets.end());
    mat.makeCompressed();
    return mat;
}

PYBIND11_MODULE(_rk4_sparse_cpp, m) {
    m.doc() = "Sparse matrix RK4 propagator for excitation dynamics (C++ implementation)";
    
    py::class_<PerformanceMetrics>(m, "PerformanceMetrics")
        .def_readonly("matrix_update_time", &PerformanceMetrics::matrix_update_time)
        .def_readonly("rk4_step_time", &PerformanceMetrics::rk4_step_time)
        .def_readonly("matrix_updates", &PerformanceMetrics::matrix_updates)
        .def_readonly("rk4_steps", &PerformanceMetrics::rk4_steps);
    
    m.def("rk4_sparse_cpp", [](
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

        // CSR行列の構築
        Eigen::SparseMatrix<cplx> H0_mat = build_sparse_matrix_from_scipy(H0);
        Eigen::SparseMatrix<cplx> mux_mat = build_sparse_matrix_from_scipy(mux);
        Eigen::SparseMatrix<cplx> muy_mat = build_sparse_matrix_from_scipy(muy);

        // 電場とpsi0の変換
        Eigen::Map<const Eigen::VectorXd> Ex_vec(static_cast<double*>(Ex_buf.ptr), Ex_buf.shape[0]);
        Eigen::Map<const Eigen::VectorXd> Ey_vec(static_cast<double*>(Ey_buf.ptr), Ey_buf.shape[0]);
        Eigen::Map<const Eigen::VectorXcd> psi0_vec(static_cast<cplx*>(psi0_buf.ptr), psi0_buf.shape[0]);

        // rk4_cpu_sparseの呼び出し
        return rk4_sparse_cpp(
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
    
    m.def("get_omp_max_threads", []() {
        #ifdef _OPENMP
        return omp_get_max_threads();
        #else
        return 1;
        #endif
    }, "Get the maximum number of OpenMP threads");
} 