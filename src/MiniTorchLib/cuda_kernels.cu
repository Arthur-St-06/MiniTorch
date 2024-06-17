#include <cuda_runtime.h>
#include "cuda_kernels.h"

__global__ void add_cuda(float* data1, float* data2, float* result_data, int size)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i < size)
	{
		result_data[i] = data1[i] + data2[i];
	}
}