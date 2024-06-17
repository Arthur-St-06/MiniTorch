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
    if (_device == "cuda" && checkPointerLocation(data) == "cpu") data = data_to_cuda(data);

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

std::string Tensor::checkPointerLocation(void* ptr)
{
    cudaPointerAttributes attributes;
    cudaError_t error = cudaPointerGetAttributes(&attributes, ptr);
    if (error == cudaSuccess)
    {
        if (attributes.type == cudaMemoryTypeDevice) return "cuda";
        else if (attributes.type == cudaMemoryTypeHost) return "cpu";
    }
    else
    {
        printf("Cuda error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }
}

float* Tensor::data_to_cuda(float* data)
{
    float* cuda_data;
    cudaMalloc((void**)&cuda_data, size * sizeof(float));
    cudaMemcpy(cuda_data, data, size * sizeof(float), cudaMemcpyHostToDevice);

    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        printf("Cuda error: %s\n", cudaGetErrorString(error));
        exit(EXIT_FAILURE);
    }

    delete[] data;

    printf("sent tensor to %s\n", device.c_str());

    return cuda_data;
}

float* Tensor::data_to_cpu(float* data, bool delete_original)
{
    float* cpu_data = new float[size];
    cudaMemcpy(cpu_data, data, size * sizeof(float), cudaMemcpyDeviceToHost);
    if(delete_original) cudaFree(data);

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

Tensor* Tensor::add_tensors(Tensor* tensor1, Tensor* tensor2)
{
    if (tensor1->device != tensor2->device)
    {
        printf("Tensors must be on the same device. Current devices: %s and %s\n", tensor1->device.c_str(), tensor1->device.c_str());
        exit(EXIT_FAILURE);
    }

    std::string device = tensor1->device;

    if (tensor1->ndim != tensor2->ndim)
    {
        printf("Tensors must have the same number of dimensions for addition. Current dimensions: %d and %d\n", tensor1->ndim, tensor2->ndim);
        exit(EXIT_FAILURE);
    }

    int ndim = tensor1->ndim;
    int* shape = new int[ndim];

    for (int i = 0; i < ndim; i++)
    {
        if (tensor1->shape[i] != tensor2->shape[i])
        {
            printf("Tensors must have the same shape for addition. Current shape at index %d: %d and %d\n", i, tensor1->shape[i], tensor2->shape[i]);
            exit(EXIT_FAILURE);
        }
        shape[i] = tensor1->shape[i];
    }

    if (device == "cuda")
    {
        float* result_data;
        cudaMalloc((void**)&result_data, tensor1->size * sizeof(float));

        int num_blocks = (tensor1->size + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
        add_cuda <<<num_blocks, THREADS_PER_BLOCK>>> (tensor1->data, tensor2->data, result_data, tensor1->size);

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
        float* result_data = new float[tensor1->size];
        add_cpu(tensor1->data, tensor2->data, result_data, tensor1->size);
        return new Tensor(result_data, shape, ndim, device);
    }
}

float Tensor::get_item(int* indicies)
{
    // Convert n-dimensional indicies to 1 index to be used with a 1d array
    int index = 0;
    for (int i = 0; i < ndim; i++)
    {
        index += indicies[i] * strides[i];
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