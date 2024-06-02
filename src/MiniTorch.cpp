#include <iostream>
#include "MiniTorchLib/Tensor.h"

// Check for memory leaks in debug mode
#ifdef _DEBUG
    #define _CRTDBG_MAP_ALLOC
    #include <crtdbg.h>
#endif

int main()
{
    #ifdef _DEBUG
        _CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
    #endif
    

    const size_t data_size = 8;

    float* first_tensor_data_array = new float[data_size];
    float* second_tensor_data_array = new float[data_size];

    for (size_t i = 0; i < data_size; i++)
    {
        first_tensor_data_array[i] = static_cast<float>(i);
        second_tensor_data_array[i] = static_cast<float>(i);
    }
    
   
    Tensor* tensor1 = create_tensor(first_tensor_data_array, new int[2] {8}, 1);
    Tensor* tensor2 = create_tensor(second_tensor_data_array, new int[2] {8}, 1);
    Tensor* result_tensor = add_tensor(tensor1, tensor2);
    //
    //int indices_array[] = { 999, 999 };
    //float sum_at_index = get_item(result_tensor, indices_array);
    //std::cout << sum_at_index;

    delete tensor1;
    delete tensor2;
    delete result_tensor;
}