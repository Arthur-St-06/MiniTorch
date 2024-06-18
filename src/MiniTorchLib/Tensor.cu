#include <chrono>

#include "Tensor.h"
#include "CPUTensorFunctions.h"
#include "cuda_kernels.h"

#define THREADS_PER_BLOCK 64

Tensor::Tensor(float* _data, int* _shape, int _ndim, std::string _device)
{
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
    if (_device == "cuda" && check_pointer_location(data) == "cpu") data = data_to_cuda(data);

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

std::string Tensor::check_pointer_location(void* _ptr)
{
    cudaPointerAttributes attributes;
    cudaError_t error = cudaPointerGetAttributes(&attributes, _ptr);
    if (error == cudaSuccess)
    {
        if (attributes.type == cudaMemoryTypeDevice) return "cuda";
        else if (attributes.type == cudaMemoryTypeHost || attributes.type == cudaMemoryTypeUnregistered) return "cpu";
        else
        {
            printf("Can't determine poiner location.\n");
            exit(EXIT_FAILURE);
        }
    }
    else
    {
        printf("Cuda error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

float* Tensor::data_to_cuda(float* _data)
{
    float* cuda_data;
    cudaMalloc((void**)&cuda_data, size * sizeof(float));
    cudaMemcpy(cuda_data, _data, size * sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Cuda error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    delete[] _data;

    printf("sent tensor to %s\n", device.c_str());

    return cuda_data;
}

float* Tensor::data_to_cpu(float* _data, bool _delete_original)
{
    float* cpu_data = new float[size];
    cudaMemcpy(cpu_data, _data, size * sizeof(float), cudaMemcpyDeviceToHost);
    if(_delete_original) cudaFree(_data);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess)
    {
        printf("Cuda error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    return cpu_data;
}

Tensor::Tensor(const std::vector<float>& _data, const std::vector<int>& _shape, int _ndim, std::string _device)
{
    shape = new int[_shape.size()];
    std::copy(_shape.begin(), _shape.end(), shape);

    ndim = _ndim;

    // Calculate total amount of elements in the tensor
    size = 1;
    for (int i = 0; i < ndim; i++)
    {
        size *= shape[i];
    }

    device = _device;
    data = new float[_data.size()];
    // Copies data of vector to the tensor from the beginning to end
    std::copy(_data.begin(), _data.end(), data);
    if (_device == "cuda") data = data_to_cuda(data);

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

Tensor* Tensor::add_tensors(Tensor* _tensor1, Tensor* _tensor2)
{
    if (_tensor1->device != _tensor2->device)
    {
        printf("Tensors must be on the same device. Current devices: %s and %s\n", _tensor1->device.c_str(), _tensor2->device.c_str());
        exit(EXIT_FAILURE);
    }

    std::string device = _tensor1->device;

    if (_tensor1->ndim != _tensor2->ndim)
    {
        printf("Tensors must have the same number of dimensions for addition. Current dimensions: %d and %d\n", _tensor1->ndim, _tensor2->ndim);
        exit(EXIT_FAILURE);
    }

    int ndim = _tensor1->ndim;
    int* shape = new int[ndim];

    for (int i = 0; i < ndim; i++)
    {
        if (_tensor1->shape[i] != _tensor2->shape[i])
        {
            printf("Tensors must have the same shape for addition. Current shape at index %d: %d and %d\n", i, _tensor1->shape[i], _tensor2->shape[i]);
            exit(EXIT_FAILURE);
        }
        shape[i] = _tensor1->shape[i];
    }

    if (device == "cuda")
    {
        float* result_data;
        cudaMalloc((void**)&result_data, _tensor1->size * sizeof(float));

        int num_blocks = (_tensor1->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        add_cuda <<<num_blocks, THREADS_PER_BLOCK>>> (_tensor1->data, _tensor2->data, result_data, _tensor1->size);

        cudaError_t error = cudaGetLastError();
        if (error != cudaSuccess)
        {
            printf("Cuda error: %s\n", cudaGetErrorString(error));
            exit(EXIT_FAILURE);
        }

        cudaDeviceSynchronize();
        return new Tensor(result_data, shape, ndim, device);
    }
    else
    {
        float* result_data = new float[_tensor1->size];
        add_cpu(_tensor1->data, _tensor2->data, result_data, _tensor1->size);
        return new Tensor(result_data, shape, ndim, device);
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
        printf("Index should be less than the size of tensor and greater than 0, current index and size are: %d, %d\n", index, size);
        exit(EXIT_FAILURE);
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