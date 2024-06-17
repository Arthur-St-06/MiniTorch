#include <cuda_runtime.h>
#include "cuda_support.h"

bool cuda_support::is_available()
{
	int deviceCount = 0;
	cudaError_t error = cudaGetDeviceCount(&deviceCount);

	return error == cudaSuccess;
}