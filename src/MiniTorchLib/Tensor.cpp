#include "Tensor.h"
#include "CPUTensorFunctions.h"

#include <chrono>

Tensor::Tensor(float* _data, int* _shape, int _ndim)
{
    data = _data;
    shape = _shape;
    ndim = _ndim;

    // Calculate total amount of elements in the tensor
    size = 1;
    for (int i = 0; i < ndim; i++)
    {
        size *= shape[i];
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

    device = nullptr;
}

Tensor::Tensor(const std::vector<float>& _data, const std::vector<int>& _shape, int _ndim)
{
    data = new float[_data.size()];
    // Copies data of vector to the tensor from the beginning to end
    std::copy(_data.begin(), _data.end(), data);

    shape = new int[_shape.size()];
    std::copy(_shape.begin(), _shape.end(), shape);

    ndim = _ndim;

    // Calculate total amount of elements in the tensor
    size = 1;
    for (int i = 0; i < ndim; i++)
    {
        size *= shape[i];
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
    device = nullptr;
}

Tensor* Tensor::add_tensors(Tensor* tensor1, Tensor* tensor2)
{
    if (tensor1->ndim != tensor2->ndim)
    {
        fprintf(stderr, "Tensors must have the same number of dimensions for addition. Current dimensions: %d and %d\n", tensor1->ndim, tensor2->ndim);
        exit(1);
    }

    int ndim = tensor1->ndim;
    int* shape = new int[ndim];

    for (int i = 0; i < ndim; i++)
    {
        if (tensor1->shape[i] != tensor2->shape[i])
        {
            fprintf(stderr, "Tensors must have the same shape for addition. Current shape at index %d: %d and %d\n", i, tensor1->shape[i], tensor2->shape[i]);
            exit(1);
        }
        shape[i] = tensor1->shape[i];
    }

    float* result_data = new float[tensor1->size];

    add_tensor_cpu(tensor1->data, tensor2->data, result_data, tensor1->size);

    return new Tensor(result_data, shape, ndim);
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
        fprintf(stderr, "Index should be less than the size of tensor and greater than 0, current index and size are: %d, %d\n", index, size);
        exit(1);
    }

    float result;
    result = data[index];

    return result;
}

float Tensor::get_item(std::vector<int> indicies)
{
    // Convert n-dimensional indicies to 1 index to be used with a 1d array
    int index = 0;
    for (int i = 0; i < ndim; i++)
    {
        index += indicies[i] * strides[i];
    }

    if ((index >= size) || (index < 0))
    {
        fprintf(stderr, "Index should be less than the size of tensor and greater than 0, current index and shape are: %d, %d\n", index, size);
        exit(1);
    }

    float result;
    result = data[index];

    return result;
}