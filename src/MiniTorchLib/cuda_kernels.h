#pragma once

__global__ void add_cuda(float* _data1, float* _data2, float* _result_data, int _size);
__global__ void arange_cuda(float* _data, int _start, int _size);