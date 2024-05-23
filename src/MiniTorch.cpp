#include <iostream>
#include "MiniTorchLib/Tensor.h"

int main()
{
    float data_array[4] = { 1.0f, 2.0f,
                            3.0f, 4.0f };
    int shape_array[2] = { 2, 2 };
    int ndim = 2;

    Tensor* tensor = create_tensor(data_array, shape_array, ndim);
    int indices_array[] = { 1, 1 };
    get_item(tensor, indices_array);
    std::cout << tensor->size;
}