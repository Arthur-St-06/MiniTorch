#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For vector handling

#include "../MiniTorchLib/Tensor.h"

namespace py = pybind11;

PYBIND11_MODULE(PythonWrapper, m)
{
    py::class_<Tensor>(m, "Tensor")
        .def(py::init<>())
        .def_readwrite("data", &Tensor::data)
        .def_readwrite("strides", &Tensor::strides)
        .def_readwrite("shape", &Tensor::shape)
        .def_readwrite("ndim", &Tensor::ndim)
        .def_readwrite("size", &Tensor::size)
        .def_readwrite("device", &Tensor::device);


    Tensor* create_tensor(const std::vector<float>&data, const std::vector<int>&shape, int ndim);
    m.def("create_tensor", &create_tensor);

    Tensor* add_tensor(Tensor* tensor1, Tensor* tensor2);
    m.def("add", &add_tensor);
}