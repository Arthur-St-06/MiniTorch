#include "CPUTensorFunctions.h"

void add_cpu(float* data1, float* data2, float* result_data, int size)
{
#pragma omp parallel for
    for (int i = 0; i < size; i++)
    {
        result_data[i] = data1[i] + data2[i];
    }
}