#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For vector handling

#include "../MiniTorchLib/Tensor.h"
#include "../MiniTorchLib/cuda_support.h"

namespace py = pybind11;

float python_get_item(Tensor& t, py::object obj_indicies)
{
    // If index is a single number e.g. [0]
    if (py::isinstance<py::int_>(obj_indicies))
    {
        int* index = new int[1];
        index[0] = py::cast<int>(obj_indicies);
        return t.get_item(index);
    }
    // If indicies are multiple numbers e.g. [0, 1]
    else if (py::isinstance<py::tuple>(obj_indicies))
    {
        py::tuple indicies_tuple = py::cast<py::tuple>(obj_indicies);

        if (indicies_tuple.size() > t.ndim)
        {
            printf("Too many indicies for tensor of dimension %d", t.ndim);
            exit(EXIT_FAILURE);
        }

        int* indicies = new int[indicies_tuple.size()];

        for (int i = 0; i < indicies_tuple.size(); i++)
        {
            if (py::isinstance<py::int_>(indicies_tuple[i]))
            {
                indicies[i] = (indicies_tuple[i].cast<int>());
            }
            else
            {
                printf("Indicies should be integers\n");
                exit(EXIT_FAILURE);
            }
        }
        return t.get_item(indicies);
    }
    else
    {
        printf("Indicies should be integers");
        exit(EXIT_FAILURE);
    }
}

PYBIND11_MODULE(PythonWrapper, m)
{
    /////////////////////////////
    // Tensor
    /////////////////////////////

    py::class_<Tensor>(m, "tensor")
        .def(py::init<const std::vector<float>&, const std::vector<int>&, int, std::string>())
        .def_readwrite("data", &Tensor::data)
        .def_readwrite("strides", &Tensor::strides)
        .def_readwrite("shape", &Tensor::shape)
        .def_readwrite("ndim", &Tensor::ndim)
        .def_readwrite("size", &Tensor::size)
        .def_readwrite("device", &Tensor::device)
        .def("to", &Tensor::to)
        .def("__getitem__", &python_get_item);

    m.def("add", [](Tensor* tensor1, Tensor* tensor2)
        {
            return Tensor::add_tensors(tensor1, tensor2);
        });

    /////////////////////////////
    // Cuda
    /////////////////////////////

    py::class_<cuda_support>(m, "cuda")
        .def(py::init<>())
        .def("is_available", &cuda_support::is_available);
}