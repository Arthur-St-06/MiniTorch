#pragma once

#include <cuda_bf16.h>
#include <iostream>
#include <string>

/////////////////////////////
// For raising exceptions in python
/////////////////////////////
#ifdef USE_PYTHON
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

/////////////////////////////
// Precision settings
/////////////////////////////

#if defined(ENABLE_FP32)
typedef float floatX;
#elif defined(ENABLE_FP16)
typedef half floatX;
#else
typedef __nv_bfloat16 floatX;
#endif

/////////////////////////////
// Error handling
/////////////////////////////

// Checking CUDA functions
void inline cuda_check(cudaError_t _error, const char* _file, int _line)
{
	if (_error != cudaSuccess)
	{
		std::string error_message = "Cuda error: " + std::string(cudaGetErrorString(_error)) + " at file: " + std::string(_file) + " line: " + std::to_string(_line);
#ifdef USE_PYTHON
		throw py::value_error(error_message);
#else
		std::cout << error_message << std::endl;
		exit(EXIT_FAILURE);
#endif
	}
}
#define cuda_check(error) (cuda_check(error, __FILE__, __LINE__))