#include <iostream>
#include "MiniTorchLib/Tensor.h"
#include <chrono>

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
    

    const size_t data_size = 8000000;

    //float* first_tensor_data_array = new float[data_size];
    //float* second_tensor_data_array = new float[data_size];
    //
    //for (size_t i = 0; i < data_size; i++)
    //{
    //    first_tensor_data_array[i] = static_cast<float>(i);
    //    second_tensor_data_array[i] = static_cast<float>(i);
    //}

    std::vector<float> data_vector;

    for (size_t i = 0; i < data_size; i++)
    {
        data_vector.push_back(i);
    }

    std::vector<int> shape = { data_size };
   
    Tensor* tensor1 = new Tensor(data_vector, shape, 1);

    Tensor* tensor2 = new Tensor(data_vector, shape, 1);
    
    auto start = std::chrono::high_resolution_clock::now();
    
    Tensor* result_tensor = Tensor::add_tensors(tensor1, tensor2);
    
    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    std::cout << "Time taken by function: " << duration.count() << " microseconds" << std::endl;
    
    int indices_array[] = { 999, 999 };
    float sum_at_index = result_tensor->get_item(indices_array);
    std::cout << sum_at_index;
    
    delete tensor1;
    delete tensor2;
    //delete result_tensor;
}