#include "Tensor.h"

Tensor* create_tensor(float* data, int* shape, int ndim)
{
    // Allocate memory for 1 tensor and get its pointer
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (tensor == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    tensor->data = data;
    tensor->shape = shape;
    tensor->ndim = ndim;

    // Calculate total amount of elements in the tensor
    tensor->size = 1;
    for (int i = 0; i < ndim; i++)
    {
        tensor->size *= shape[i];
    }

    // Allocate memory for strides which has "ndim" elements
    tensor->strides = (int*)malloc(ndim * sizeof(int));
    if (tensor->strides == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    // Calculate stride for each dimension
    int stride = 1;
    for (int i = ndim - 1; i >= 0; i--)
    {
        tensor->strides[i] = stride;
        stride *= shape[i];
    }

    return tensor;
}

float get_item(Tensor* tensor, int* indicies)
{
    // Convert n-dimensional indicies to 1 index to be used with a 1d array
    int index = 0;
    for (int i = 0; i < tensor->ndim; i++)
    {
        index += indicies[i] * tensor->strides[i];
    }

    assert((index < tensor->size) && (index >= 0));

    float result;
    result = tensor->data[index];

    return result;
}