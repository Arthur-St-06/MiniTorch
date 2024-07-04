#pragma once

#include "cuda_common.h"

__global__ void add_cuda(floatX* _data1, floatX* _data2, floatX* _result_data, int _size);
__global__ void arange_cuda(floatX* _data, int _start, int _size);