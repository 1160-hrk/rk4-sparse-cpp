#include <pybind11/pybind11.h>
#include <pybind11/eigen.h>
#include <pybind11/stl.h>
#include "excitation_rk4_sparse.hpp"

namespace py = pybind11;

PYBIND11_MODULE(_excitation_rk4_sparse, m) {
    m.def("rk4_propagate", &rk4_propagate, "Runge-Kutta 4 propagator (C++)");
}
