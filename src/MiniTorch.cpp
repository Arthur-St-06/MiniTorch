#include <iostream>
#include "MiniTorchLib/Tensor.h"
#include "MiniTorchLib/helper_functions.h"

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

    //float* first_tensor_data_array = new float[data_size];
    //float* second_tensor_data_array = new float[data_size];
    //
    //for (size_t i = 0; i < data_size; i++)
    //{
    //    first_tensor_data_array[i] = static_cast<float>(i);
    //    second_tensor_data_array[i] = static_cast<float>(i);
    //}

    // Initializing first tensor

    const int size = 10000;

    Tensor* tensor1;
    floatX* data1 = new floatX[8]{ 1, 2, 3, 4, 5, 6, 7, 8 };
    int* shape1 = new int[3] { 2, 2, 2 };
    int ndim1 = 3;

    tensor1 = new Tensor(data1, shape1, ndim1, "cuda");
    std::cout << tensor1->to_string();
    
    // Initializing second tensor
    Tensor* tensor2;
    floatX* data2 = new floatX[4]{ 1, 2, 3, 4 };
    int* shape2 = new int[2] { 2, 2 };
    int ndim2 = 2;
    
    // Result tensor
    Tensor* result_tensor;
    
    tensor1 = Tensor::arange(0, 10, "cpu");
    tensor2 = Tensor::arange(0, 10, "cpu");

    double execution_time = timeit(Tensor::add_tensors, tensor1, tensor2);
    printf("Average execution time: %f microseconds\n", execution_time);

    Tensor* res_tensor = Tensor::add_tensors(tensor1, tensor2);
    res_tensor = res_tensor->to("cpu");
    int a = 0;

    /*std::vector<float> data_vector;
    for (size_t i = 0; i < data_size; i++)
    {
        data_vector.push_back(i);
    }*/

    /*std::vector<int> shape = { data_size };
   
    Tensor* tensor1 = new Tensor(data_vector, shape, 1, "cuda");

    Tensor* tensor2 = new Tensor(data_vector, shape, 1, "cuda");

    auto start = std::chrono::high_resolution_clock::now();
    
    Tensor* result_tensor = Tensor::add_tensors(tensor1, tensor2);

    auto stop = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(stop - start);
    
    std::cout << "Time taken by function: " << duration.count() / 100 << " microseconds" << std::endl;
    
    int indices_array[] = { 1 };

    
    float sum_at_index = result_tensor->get_item(indices_array);
    std::cout << sum_at_index;
    
    delete tensor1;
    delete tensor2;
    delete result_tensor;*/
}