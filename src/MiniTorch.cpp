#include <iostream>
#include "MiniTorchLib/Tensor.h"

int main()
{
    const size_t data_size = 1000000;
    float* data_array = (float*)malloc(data_size * sizeof(float));

    if (data_array == NULL)
    {
        fprintf(stderr, "Memory allocation failed");
        exit(1);
    }

    for (size_t i = 0; i < data_size; i++)
    {
        data_array[i] = static_cast<float>(i);
    }

    int shape_array[2] = { 1000, 1000 };
    int ndim = 2;

    Tensor* tensor1 = create_tensor(data_array, shape_array, ndim);
    Tensor* tensor2 = create_tensor(data_array, shape_array, ndim);
    Tensor* result_tensor = add_tensor(tensor1, tensor2);

    int indices_array[] = { 999, 999 };
    float sum_at_index = get_item(result_tensor, indices_array);
    std::cout << sum_at_index;
}