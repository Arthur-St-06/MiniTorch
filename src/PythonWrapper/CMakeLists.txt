cmake_minimum_required(VERSION 3.20)

project(PythonWrapper)

include(FetchContent)

FetchContent_Declare(
  pybind11
  GIT_REPOSITORY https://github.com/pybind/pybind11.git
)

FetchContent_MakeAvailable(pybind11)

pybind11_add_module(PythonWrapper wrappers.cpp ../MiniTorchLib/Tensor.cu ../MiniTorchLib/CPUTensorFunctions.cpp ../MiniTorchLib/cuda_support.cu ../MiniTorchLib/cuda_kernels.cu)

target_compile_definitions(PythonWrapper PRIVATE USE_PYTHON)