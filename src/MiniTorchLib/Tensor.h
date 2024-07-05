#pragma once

#include <cuda_runtime.h>
#include <iostream>
#include <vector>
#include <iomanip> // For std::setw
#include "cuda_common.h"

class Tensor
{
public:
    floatX* data;
    int* strides;
    int* shape;
    int ndim;
    int size;
    std::string device;
    
    Tensor(floatX* _data, int* _shape, int _ndim, std::string _device = "cpu");
    ~Tensor();

    static Tensor* add_tensors(Tensor* _tensor1, Tensor* _tensor2);
    static Tensor* arange(int _start, int _end, std::string _device = "cpu");

    floatX get_item(int* _indicies);
    Tensor* to(std::string _device);

    std::string tensor_to_string();
    
private:
    std::string data_to_string(int _dim, int _offset, int _indentLevel);

    floatX* data_to_cuda(floatX* _data);
    // _delete_original = false prevents data deletion on cuda (needed if copying for printing)
    floatX* data_to_cpu(floatX* _data, bool _delete_original = true);

    // Returns "cpu" or "cuda" depending where _ptr is store
    std::string check_pointer_location(void* _ptr);
};