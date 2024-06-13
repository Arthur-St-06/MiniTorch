#pragma once

#include <iostream>
#include <vector>
#include <immintrin.h>

class Tensor
{
public:
    float* data;

    int* strides;
    int* shape;
    int ndim;
    int size;
    char* device;

    static Tensor* add_tensors(Tensor* tensor1, Tensor* tensor2);
    //float get_item(Tensor* tensor, int* indicies);

    Tensor(float* data, int* shape, int ndim);
    Tensor(const std::vector<float>& data, const std::vector<int>& shape, int ndim);

    ~Tensor()
    {
        delete[] data;

        delete[] strides;
        delete[] shape;
        delete device;
    }
};