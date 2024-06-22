#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>

class Tensor
{
public:
    float* data;
    int* strides;
    int* shape;
    int ndim;
    int size;
    std::string device;
    
    Tensor(float* _data, int* _shape, int _ndim, std::string _device = "cpu");

    ~Tensor();

    static Tensor* add_tensors(Tensor* _tensor1, Tensor* _tensor2);
    static Tensor* arange(int _start, int _end, std::string _device = "cpu");

    float get_item(int* _indicies);
    Tensor* to(std::string _device);
    
private:
    float* data_to_cuda(float* _data);
    // _delete_original = false prevents data deletion on cuda (needed if copying for printing)
    float* data_to_cpu(float* _data, bool _delete_original = true);

    // Returns "cpu" or "cuda" depending where ptr is store
    std::string check_pointer_location(void* _ptr);
};