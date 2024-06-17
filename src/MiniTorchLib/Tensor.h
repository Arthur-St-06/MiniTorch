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
    
    Tensor(float* data, int* shape, int ndim, std::string _device = "cpu");
    Tensor(const std::vector<float>& data, const std::vector<int>& shape, int ndim, std::string _device = "cpu");

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

    static Tensor* add_tensors(Tensor* tensor1, Tensor* tensor2);

    float get_item(int* indicies);
    Tensor* to(std::string device);
    float* data_to_cuda(float* data);
    float* data_to_cpu(float* data, bool delete_original = true);

private:
    std::string checkPointerLocation(void* ptr);
};