cmake_minimum_required(VERSION 3.14)
project(excitation_rk4_sparse)

find_package(pybind11 REQUIRED)
find_package(Eigen3 REQUIRED)

pybind11_add_module(_excitation_rk4_sparse
    src/excitation_rk4_sparse.cpp
    src/bindings.cpp
)

target_include_directories(_excitation_rk4_sparse PRIVATE include Eigen3::Eigen)
