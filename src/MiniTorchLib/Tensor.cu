#include <chrono>

#include "Tensor.h"
#include "cpu_kernels.h"
#include "cuda_kernels.h"
#include "errors_support.h"

#define THREADS_PER_BLOCK 64*100000

Tensor::Tensor(float* _data, int* _shape, int _ndim, std::string _device)
{
    //throw py::value_error("Creating a tensor");
    shape = _shape;
    ndim = _ndim;

    // Calculate total amount of elements in the tensor
    size = 1;
    for (int i = 0; i < ndim; i++)
    {
        size *= shape[i];
    }

    device = _device;
    data = _data;
    // If device change is needed
    if (_device == "cuda" && check_pointer_location(data) == "cpu")
    {
        data = data_to_cuda(data);
    }

    // Allocate memory for strides which has "ndim" elements
    strides = new int[ndim];

    // Calculate stride for each dimension
    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--)
    {
        strides[i] = stride;
        stride *= shape[i];
    }
}

Tensor::~Tensor()
{
    if (device == "cuda")
    {
        cuda_check(cudaFree(data));
    }
    else if (device == "cpu")
    {
        delete[] data;
    }

    delete[] strides;
    delete[] shape;

    std::cout << "Tensor was deleted" << std::endl;
}

Tensor* Tensor::add_tensors(Tensor* _tensor1, Tensor* _tensor2)
{
    if (_tensor1->ndim != _tensor2->ndim)
    {
        throw_error("Tensors must have the same number of dimensions for addition. Current dimensions: %d and %d\n", _tensor1->ndim, _tensor2->ndim);
    }

    if (_tensor1->device != _tensor2->device)
    {
        throw_error("Tensors must be on the same device. Current devices: %s and %s\n", _tensor1->device.c_str(), _tensor2->device.c_str());
    }

    int ndim = _tensor1->ndim;
    std::string device = _tensor1->device;
    int* shape = new int[ndim];

    for (int i = 0; i < ndim; i++)
    {
        if (_tensor1->shape[i] != _tensor2->shape[i])
        {
            throw_error("Tensors must have the same shape for addition. Current shape at index %d: %d and %d\n", i, _tensor1->shape[i], _tensor2->shape[i]);
        }
        shape[i] = _tensor1->shape[i];
    }

    if (device == "cuda")
    {
        float* data;
        cuda_check(cudaMalloc((void**)&data, _tensor1->size * sizeof(float)));

        int num_blocks = (_tensor1->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        add_cuda << <num_blocks, THREADS_PER_BLOCK >> > (_tensor1->data, _tensor2->data, data, _tensor1->size);

        cuda_check(cudaGetLastError());
        cuda_check(cudaDeviceSynchronize());
        return new Tensor(data, shape, ndim, device);
    }
    else
    {
        float* data = new float[_tensor1->size];
        add_cpu(_tensor1->data, _tensor2->data, data, _tensor1->size);
        return new Tensor(data, shape, ndim, device);
    }
}

Tensor* Tensor::arange(int _start, int _end, std::string _device)
{
    int start = _start;
    int end = _end;
    int size = end - start;
    // Size and shape equal in 1d tensor
    int* shape = new int[1];
    shape[0] = size;

    std::string device = _device;

    if (_device == "cuda")
    {
        float* data;
        cuda_check(cudaMalloc((void**)&data, size * sizeof(float)));

        int num_blocks = (size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        arange_cuda << <num_blocks, THREADS_PER_BLOCK >> > (data, start, size);
        cuda_check(cudaGetLastError());
        return new Tensor(data, shape, 1, device);
    }
    else
    {
        float* data = new float[size];
        arange_cpu(data, start, size);
        return new Tensor(data, shape, 1, device);
    }
}

float Tensor::get_item(int* _indicies)
{
    // Convert n-dimensional indicies to 1 index to be used with a 1d array
    int index = 0;
    for (int i = 0; i < ndim; i++)
    {
        index += _indicies[i] * strides[i];
    }

    if ((index >= size) || (index < 0))
    {
        throw_error("Index should be less than the size of tensor and greater than 0, current index and size are: %d, %d\n", index, size);
    }

    float result;
    if (device == "cuda")
    {
        float* tmp_cpu_data = data_to_cpu(data, false);
        result = tmp_cpu_data[index];
        delete[] tmp_cpu_data;
    }
    else if (device == "cpu")
    {
        result = data[index];
    }

    return result;
}

Tensor* Tensor::to(std::string _device)
{
    if (device == "cpu" && _device == "cuda")
    {
        return new Tensor(data, shape, ndim, "cuda");
    }
    else if (device == "cuda" && _device == "cpu")
    {
        return new Tensor(data, shape, ndim, "cpu");
    }
    return this;
}

float* Tensor::data_to_cuda(float* _data)
{
    float* cuda_data;
    cuda_check(cudaMalloc((void**)&cuda_data, size * sizeof(float)));
    cuda_check(cudaMemcpy(cuda_data, _data, size * sizeof(float), cudaMemcpyHostToDevice));

    delete[] _data;

    printf("sent tensor to %s\n", device.c_str());

    return cuda_data;
}

float* Tensor::data_to_cpu(float* _data, bool _delete_original)
{
    float* cpu_data = new float[size];
    cuda_check(cudaMemcpy(cpu_data, _data, size * sizeof(float), cudaMemcpyDeviceToHost));
    if (_delete_original)
    {
        cuda_check(cudaFree(_data));
    }

    return cpu_data;
}

std::string Tensor::check_pointer_location(void* _ptr)
{
    cudaPointerAttributes attributes;
    cudaError_t error = cudaPointerGetAttributes(&attributes, _ptr);
    if (error == cudaSuccess)
    {
        if (attributes.type == cudaMemoryTypeDevice)
        {
            return "cuda";
        }
        else if (attributes.type == cudaMemoryTypeHost || attributes.type == cudaMemoryTypeUnregistered)
        {
            return "cpu";
        }
        else
        {
            throw_error("Can't determine poiner location.");
        }
    }
    else
    {
        throw_error("Cuda error: %s\n", cudaGetErrorString(error));
    }
}