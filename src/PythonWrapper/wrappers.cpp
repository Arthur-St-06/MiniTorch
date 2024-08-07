//#include <pybind11/pybind11.h>
#include <pybind11/stl.h> // For vector handling

#include "../MiniTorchLib/Tensor.h"
#include "../MiniTorchLib/cuda_support.h"
#include "../MiniTorchLib/common.h"
#include "../MiniTorchLib/cuda_common.h"

namespace py = pybind11;

floatX python_get_item(Tensor& t, py::object obj_indicies)
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
            throw_error("Too many indicies for tensor of dimension %d", t.ndim);
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
                throw_error("Indicies should be integers\n");
            }
        }
        return t.get_item(indicies);
    }
    else
    {
        throw_error("Indicies should be integers");
    }
}

/////////////////////////////
// Initializing tensors using just data with arbitrary number of dimensions
/////////////////////////////

int count_elements(const py::list& list)
{
    int total = 0;
    for (const auto& item : list)
    {
        if (py::isinstance<py::list>(item))
        {
            total += count_elements(item.cast<py::list>());
        }
        else
        {
            total++;
        }
    }
    return total;
}

void flatten_list(const py::list& _list, floatX* _data, int& _index)
{
    for (const auto& item : _list)
    {
        if (py::isinstance<py::list>(item))
        {
            flatten_list(item.cast<py::list>(), _data, _index);
        }
        else
        {
            if (py::isinstance<py::float_>(item) || py::isinstance<py::int_>(item))
            {
                _data[_index++] = item.cast<floatX>();
            }
            else
            {
                throw_error("List should contain numerical values\n");
            }
        }
    }
}

void set_shape(const py::object& _obj, std::vector<int>& _shape)
{
    if (py::isinstance<py::list>(_obj) == false)
        return;

    py::list list = _obj.cast<py::list>();

    if (list.size() == 0)
    {
        _shape.push_back(0);
        return;
    }

    _shape.push_back(list.size());

    py::object obj = py::reinterpret_borrow<py::object>(list[0]);
    set_shape(obj, _shape);
}

void verify_shape(const py::object& _obj, std::vector<int>& _shape, int _level)
{
    if (py::isinstance<py::list>(_obj) == false)
        return;

    py::list list = _obj.cast<py::list>();

    if (_shape.size() <= _level)
    {
        throw_error("Expected a list\n");
    }
    if (_shape[_level] != list.size())
    {
        throw_error("Amount of elements should be equal at all levels. Current number of elements in level %d: %d and %d\n", _level, _shape[_level], list.size());
    }

    for (const auto& inner_list : list)
    {
        py::object obj = py::reinterpret_borrow<py::object>(inner_list);
        verify_shape(obj, _shape, _level++);
    }
}

void get_shape(const py::list& _list, std::vector<int>& _shape)
{
    set_shape(_list, _shape);
    verify_shape(_list, _shape, 0);
}

Tensor* tensor_initializer(const py::list& list, std::string device)
{
    int num_elements = count_elements(list);
    floatX* data = new floatX[num_elements];

    int index = 0;
    flatten_list(list, data, index);

    std::vector<int> vec_shape;
    get_shape(list, vec_shape);

    for (int elem : vec_shape)
    {
        std::cout << elem << std::endl;
    }

    int* shape = new int[vec_shape.size()];
    std::copy(vec_shape.begin(), vec_shape.end(), shape);

    int ndim = vec_shape.size();

    return new Tensor(data, shape, ndim, device);
}

PYBIND11_MODULE(PythonWrapper, m)
{
    /////////////////////////////
    // Tensor
    /////////////////////////////

    py::class_<Tensor>(m, "tensor")
        .def(py::init([](const py::list& list, std::string device)
            {
                return tensor_initializer(list, device);
            }), py::arg("list"), py::arg("device") = "cpu")
        .def_readwrite("data", &Tensor::data)
        .def_readwrite("strides", &Tensor::strides)
        .def_readwrite("shape", &Tensor::shape)
        .def_readwrite("ndim", &Tensor::ndim)
        .def_readwrite("size", &Tensor::size)
        .def_readwrite("device", &Tensor::device)
        .def("to", &Tensor::to)
        .def("__getitem__", &python_get_item)
        .def("__repr__", &Tensor::tensor_to_string);

    m.def("add", &Tensor::add_tensors);
    m.def("arange", &Tensor::arange, py::arg("start") = 0, py::arg("end"), py::arg("device") = "cpu");

    /////////////////////////////
    // Cuda
    /////////////////////////////

    py::class_<cuda_support>(m, "cuda")
        .def(py::init<>())
        .def("is_available", &cuda_support::is_available);
}