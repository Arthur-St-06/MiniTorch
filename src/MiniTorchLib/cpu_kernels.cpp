#include "cpu_kernels.h"

void add_cpu(floatX* _data1, floatX* _data2, floatX* _result_data, int _size)
{
#pragma omp parallel for
    for (int i = 0; i < _size; i++)
    {
        _result_data[i] = _data1[i] + _data2[i];
    }
}

void arange_cpu(floatX* _data, int _start, int _size)
{
#pragma omp parallel for
    for (int i = 0; i < _size; i++)
    {
        _data[i] = i + _start;
    }
}