#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For vector handling

#include "../MiniTorchLib/Tensor.h"
//#include "../MiniTorchLib/CPUTensorFunctions.h"

namespace py = pybind11;

PYBIND11_MODULE(PythonWrapper, m)
{
    py::class_<Tensor>(m, "tensor")
        .def(py::init<const std::vector<float>&, const std::vector<int>&, int>())
        .def_readwrite("data", &Tensor::data)
        .def_readwrite("strides", &Tensor::strides)
        .def_readwrite("shape", &Tensor::shape)
        .def_readwrite("ndim", &Tensor::ndim)
        .def_readwrite("size", &Tensor::size)
        .def_readwrite("device", &Tensor::device);

    m.def("add", [](Tensor* tensor1, Tensor* tensor2)
        {
            return Tensor::add_tensors(tensor1, tensor2);
        });
}