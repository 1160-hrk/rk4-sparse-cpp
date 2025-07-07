#include "excitation_rk4_sparse.hpp"

Eigen::MatrixXcd rk4_propagate(
    const Eigen::SparseMatrix<std::complex<double>>& H0,
    const Eigen::SparseMatrix<std::complex<double>>& mux,
    const Eigen::SparseMatrix<std::complex<double>>& muy,
    const Eigen::VectorXcd& psi0,
    const std::vector<std::array<double,3>>& Ex3,
    const std::vector<std::array<double,3>>& Ey3,
    double dt,
    int steps,
    int stride,
    bool renorm,
    bool return_traj)
{
    // TODO: implement RK4 algorithm here
    return Eigen::MatrixXcd();
}
