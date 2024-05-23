#ifndef TENSOR_H
#define TENSOR_H

#include <iostream>
#include <cassert>

typedef struct Tensor
{
    float* data;
    int* strides;
    int* shape;
    int ndim;
    int size;
    char* device;
};

Tensor* create_tensor(float* data, int* shape, int ndim);
float get_item(Tensor* tensor, int* indicies);

#endif