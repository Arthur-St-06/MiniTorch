#include "Tensor.h"
#include "CPUTensorFunctions.h"

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

Tensor* create_tensor(const std::vector<float>& data, const std::vector<int>& shape, int ndim)
{
    // Allocate memory for 1 tensor and get its pointer
    Tensor* tensor = (Tensor*)malloc(sizeof(Tensor));
    if (tensor == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }

    tensor->data = (float*)malloc(data.size() * sizeof(float));
    if (tensor->data == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    // Copies data of vector to the tensor from the beginning to end
    std::copy(data.begin(), data.end(), tensor->data);

    tensor->shape = (int*)malloc(shape.size() * sizeof(int));
    if (tensor->shape == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    std::copy(shape.begin(), shape.end(), tensor->shape);

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

Tensor* add_tensor(Tensor* tensor1, Tensor* tensor2)
{
    if (tensor1->ndim != tensor2->ndim)
    {
        fprintf(stderr, "Tensors must have the same number of dimensions for addition. Current dimensions: %d and %d\n", tensor1->ndim, tensor2->ndim);
        exit(1);
    }

    int ndim = tensor1->ndim;
    int* shape = (int*)malloc(ndim * sizeof(int));
    if (shape == NULL)
    {
        fprintf(stderr, "Memory allocation failed");
        exit(1);
    }

    for (int i = 0; i < ndim; i++)
    {
        if (tensor1->shape[i] != tensor2->shape[i])
        {
            fprintf(stderr, "Tensors must have the same shape for addition. Current shape at index %d: %d and %d\n", i, tensor1->shape[i], tensor2->shape[i]);
            exit(1);
        }
        shape[i] = tensor1->shape[i];
    }

    float* result_data = (float*)malloc(tensor1->size * sizeof(float));
    if (result_data == NULL)
    {
        fprintf(stderr, "Memory allocation failed\n");
        exit(1);
    }
    add_tensor_cpu(tensor1, tensor2, result_data);

    return create_tensor(result_data, shape, ndim);
}

float get_item(Tensor* tensor, int* indicies)
{
    // Convert n-dimensional indicies to 1 index to be used with a 1d array
    int index = 0;
    for (int i = 0; i < tensor->ndim; i++)
    {
        index += indicies[i] * tensor->strides[i];
    }

    if ((index >= tensor->size) || (index < 0))
    {
        fprintf(stderr, "Index should be less than the size of tensor and greater than 0, current index and shape are: %d, %d\n", index, tensor->size);
        exit(1);
    }

    float result;
    result = tensor->data[index];

    return result;
}