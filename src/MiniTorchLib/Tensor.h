#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <vector>

typedef struct Tensor
{
    float* data;
    int* strides;
    int* shape;
    int ndim;
    int size;
    char* device;

    Tensor()
    {
        data = nullptr;
        strides = nullptr;
        shape = nullptr;
        ndim = 0;
        size = 0;
        device = nullptr;
    }

    ~Tensor()
    {
        delete[] data;
        delete[] strides;
        delete[] shape;
        delete device;
    }
};

Tensor* create_tensor(float* data, int* shape, int ndim);
Tensor* create_tensor(const std::vector<float>& data, const std::vector<int>& shape, int ndim);
Tensor* add_tensor(Tensor* tensor1, Tensor* tensor2);
float get_item(Tensor* tensor, int* indicies);

#endif