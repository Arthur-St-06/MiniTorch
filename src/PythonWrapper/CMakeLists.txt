project(PythonWrapper)

include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
)

FetchContent_MakeAvailable(pybind11)

pybind11_add_module(PythonWrapper wrappers.cpp ../MiniTorchLib/Tensor.cpp ../MiniTorchLib/CPUTensorFunctions.cpp)