#include <pybind11/pybind11.h>
#include <pybind11/complex.h>
#include <pybind11/numpy.h>
#include <pybind11/stl.h>
#include <pybind11/eigen.h>
#include "excitation_rk4_sparse.hpp"
#ifdef _OPENMP
#include <omp.h>
#endif

namespace py = pybind11;
using namespace excitation_rk4_sparse;
using cplx = std::complex<double>;

// CSRデータからEigenの疎行列を構築するヘルパー関数
Eigen::SparseMatrix<std::complex<double>> build_sparse_matrix(
    py::array_t<std::complex<double>> data,
    py::array_t<int> indices,
    py::array_t<int> indptr,
    int rows, int cols)
{
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

PYBIND11_MODULE(_excitation_rk4_sparse, m) {
    m.doc() = "Sparse matrix RK4 propagator for excitation dynamics (C++ implementation)";
    
    py::class_<PerformanceMetrics>(m, "PerformanceMetrics")
        .def_readonly("matrix_update_time", &PerformanceMetrics::matrix_update_time)
        .def_readonly("rk4_step_time", &PerformanceMetrics::rk4_step_time)
        .def_readonly("matrix_updates", &PerformanceMetrics::matrix_updates)
        .def_readonly("rk4_steps", &PerformanceMetrics::rk4_steps);
    
    m.def("rk4_cpu_sparse", [](
        py::array_t<cplx> H0_data,
        py::array_t<int> H0_indices,
        py::array_t<int> H0_indptr,
        int H0_rows,
        int H0_cols,
        py::array_t<cplx> mux_data,
        py::array_t<int> mux_indices,
        py::array_t<int> mux_indptr,
        py::array_t<cplx> muy_data,
        py::array_t<int> muy_indices,
        py::array_t<int> muy_indptr,
        py::array_t<double> Ex,
        py::array_t<double> Ey,
        py::array_t<cplx> psi0,
        double dt,
        bool return_traj,
        int stride,
        bool renorm
    ) {
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

        // 入力チェック
        if (psi0_buf.ndim != 1) {
            throw std::runtime_error("psi0 must be a 1D array");
        }

        // CSR行列の構築
        Eigen::SparseMatrix<cplx> H0_mat(H0_rows, H0_cols);
        Eigen::SparseMatrix<cplx> mux_mat(H0_rows, H0_cols);
        Eigen::SparseMatrix<cplx> muy_mat(H0_rows, H0_cols);

        // H0の構築
        {
            std::vector<Eigen::Triplet<cplx>> triplets;
            auto H0_data_ptr = static_cast<cplx*>(H0_data_buf.ptr);
            auto H0_indices_ptr = static_cast<int*>(H0_indices_buf.ptr);
            auto H0_indptr_ptr = static_cast<int*>(H0_indptr_buf.ptr);

            for (int i = 0; i < H0_rows; i++) {
                for (int j = H0_indptr_ptr[i]; j < H0_indptr_ptr[i + 1]; j++) {
                    triplets.emplace_back(i, H0_indices_ptr[j], H0_data_ptr[j]);
                }
            }
            H0_mat.setFromTriplets(triplets.begin(), triplets.end());
        }

        // muxの構築
        {
            std::vector<Eigen::Triplet<cplx>> triplets;
            auto mux_data_ptr = static_cast<cplx*>(mux_data_buf.ptr);
            auto mux_indices_ptr = static_cast<int*>(mux_indices_buf.ptr);
            auto mux_indptr_ptr = static_cast<int*>(mux_indptr_buf.ptr);

            for (int i = 0; i < H0_rows; i++) {
                for (int j = mux_indptr_ptr[i]; j < mux_indptr_ptr[i + 1]; j++) {
                    triplets.emplace_back(i, mux_indices_ptr[j], mux_data_ptr[j]);
                }
            }
            mux_mat.setFromTriplets(triplets.begin(), triplets.end());
        }

        // muyの構築
        {
            std::vector<Eigen::Triplet<cplx>> triplets;
            auto muy_data_ptr = static_cast<cplx*>(muy_data_buf.ptr);
            auto muy_indices_ptr = static_cast<int*>(muy_indices_buf.ptr);
            auto muy_indptr_ptr = static_cast<int*>(muy_indptr_buf.ptr);

            for (int i = 0; i < H0_rows; i++) {
                for (int j = muy_indptr_ptr[i]; j < muy_indptr_ptr[i + 1]; j++) {
                    triplets.emplace_back(i, muy_indices_ptr[j], muy_data_ptr[j]);
                }
            }
            muy_mat.setFromTriplets(triplets.begin(), triplets.end());
        }

        // 電場とpsi0の変換
        Eigen::Map<const Eigen::VectorXd> Ex_vec(static_cast<double*>(Ex_buf.ptr), Ex_buf.shape[0]);
        Eigen::Map<const Eigen::VectorXd> Ey_vec(static_cast<double*>(Ey_buf.ptr), Ey_buf.shape[0]);
        Eigen::Map<const Eigen::VectorXcd> psi0_vec(static_cast<cplx*>(psi0_buf.ptr), psi0_buf.shape[0]);

        // rk4_cpu_sparseの呼び出し
        return rk4_cpu_sparse(
            H0_mat, mux_mat, muy_mat,
            Ex_vec, Ey_vec,
            psi0_vec,
            dt, return_traj, stride, renorm
        );
    },
    py::arg("H0_data"),
    py::arg("H0_indices"),
    py::arg("H0_indptr"),
    py::arg("H0_rows"),
    py::arg("H0_cols"),
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