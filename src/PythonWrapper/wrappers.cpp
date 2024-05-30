#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "../MiniTorchLib/Tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(PythonWrapper, m) {
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def_readwrite("data", &Tensor::data)
        .def_readwrite("strides", &Tensor::strides)
        .def_readwrite("shape", &Tensor::shape)
        .def_readwrite("ndim", &Tensor::ndim)
        .def_readwrite("size", &Tensor::size)
        .def_readwrite("device", &Tensor::device);

    m.def("create_tensor", [](const std::vector<float>& data, const std::vector<int>& shape, int ndim) {
        return create_tensor(data, shape, ndim);
        }, py::return_value_policy::take_ownership);
}