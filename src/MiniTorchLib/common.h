#pragma once

/////////////////////////////
// For raising exceptions in python
/////////////////////////////
#ifdef USE_PYTHON
#include <pybind11/pybind11.h>
namespace py = pybind11;
#endif

// Throw errors in python and c++ (use instead of printf() and exit())
template<typename... Args>
void throw_error(const char* _format, Args... _args)
{
	char buffer[1000];
	snprintf(buffer, sizeof(buffer), _format, _args...);

#ifdef USE_PYTHON
	throw py::value_error(buffer);
#else
	std::cout << buffer << std::endl;
	exit(EXIT_FAILURE);
#endif
}