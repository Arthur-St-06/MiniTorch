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
    Tensor(const std::vector<float>& _data, const std::vector<int>& _shape, int _ndim, std::string _device = "cpu");

    ~Tensor()
    {
        if (device == "cuda")
        {
            cudaFree(data);
        }
        else if (device == "cpu")
        {
            delete[] data;
        }
        
        delete[] strides;
        delete[] shape;
    }

    static Tensor* add_tensors(Tensor* _tensor1, Tensor* _tensor2);
    //static Tensor* arange(int _size, );

    float get_item(int* _indicies);
    Tensor* to(std::string _device);
    float* data_to_cuda(float* _data);
    // Don't delete data on cuda if copying for printing
    float* data_to_cpu(float* _data, bool _delete_original = true);

private:
    // Returns "cpu" or "cuda" depending where ptr is store
    std::string check_pointer_location(void* _ptr);
};