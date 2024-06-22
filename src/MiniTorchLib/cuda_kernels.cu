#include <cuda_runtime.h>
#include "cuda_kernels.h"

__global__ void add_cuda(float* _data1, float* _data2, float* _result_data, int _size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < _size)
	{
		_result_data[i] = _data1[i] + _data2[i];
	}
}

__global__ void arange_cuda(float* _data, int _start, int _size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < _size)
	{
		_data[i] = i + _start;
	}
}