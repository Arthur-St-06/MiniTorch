#include "../MiniTorchLib/Tensor.h"

#include "gtest/gtest.h"

class IndexingTensorTest : public ::testing::Test {
protected:
    Tensor* tensor;
    float* data = new float[4] { 1.0f, 2.0f,
                                 3.0f, 4.0f };
    int* shape = new int[2] { 2, 2 };
    int ndim = 2;

    void SetUp() override {
        tensor = new Tensor(data, shape, ndim);
    }

    void TearDown() override {
        delete tensor;
    }
};

TEST_F(IndexingTensorTest, GetItemFromFirstIndex) {
    int indices[] = { 0, 0 };
    float item = tensor->get_item(indices);
    EXPECT_EQ(1.0f, item);
}

TEST_F(IndexingTensorTest, GetItemFromSecondIndex) {
    int indices[] = { 0, 1 };
    float item = tensor->get_item(indices);
    EXPECT_EQ(2.0f, item);
}

TEST_F(IndexingTensorTest, GetItemFromThirdIndex) {
    int indices[] = { 1, 0 };
    float item = tensor->get_item(indices);
    EXPECT_EQ(3.0f, item);
}

TEST_F(IndexingTensorTest, GetItemFromFourthIndex) {
    int indices[] = { 1, 1 };
    float item = tensor->get_item(indices);
    EXPECT_EQ(4.0f, item);
}

class AddingTensorsTest : public ::testing::Test {
protected:
    // Initializing first tensor
    Tensor* tensor1;
    float* data1 = new float[4] { 1.0f, 2.0f,
                                 3.0f, 4.0f };
    int* shape1 = new int[2] { 2, 2 };
    int ndim1 = 2;

    // Initializing second tensor
    Tensor* tensor2;
    float* data2 = new float[4] { -1.0f, 0.0f,
                                   1.0f, 4.0f };
    int* shape2 = new int[2] { 2, 2 };
    int ndim2 = 2;

    // Result tensor
    Tensor* result_tensor;

    void SetUp() override {
        tensor1 = new Tensor(data1, shape1, ndim1);
        tensor2 = new Tensor(data2, shape2, ndim2);
        result_tensor = Tensor::add_tensors(tensor1, tensor2);
    }

    void TearDown() override {
        delete tensor1;
        delete tensor2;
        delete result_tensor;
    }
};

TEST_F(AddingTensorsTest, FirstIndexCheck)
{
    int indices[] = { 0, 0 };
    float item = result_tensor->get_item(indices);
    EXPECT_EQ(0.0f, item);
}

TEST_F(AddingTensorsTest, SecondIndexCheck)
{
    int indices[] = { 0, 1 };
    float item = result_tensor->get_item(indices);
    EXPECT_EQ(2.0f, item);
}

TEST_F(AddingTensorsTest, ThirdIndexCheck)
{
    int indices[] = { 1, 0 };
    float item = result_tensor->get_item(indices);
    EXPECT_EQ(4.0f, item);
}

TEST_F(AddingTensorsTest, FourthIndexCheck)
{
    int indices[] = { 1, 1 };
    float item = result_tensor->get_item(indices);
    EXPECT_EQ(8.0f, item);
}

class CudaAddingTensorsTest : public ::testing::Test {
protected:
    // Initializing first tensor
    Tensor* tensor1;
    float* data1 = new float[4] { 1.0f, 2.0f,
                                  3.0f, 4.0f };
    int* shape1 = new int[2] { 2, 2 };
    int ndim1 = 2;

    // Initializing second tensor
    Tensor* tensor2;
    float* data2 = new float[4] { -1.0f, 0.0f,
                                   1.0f, 4.0f };
    int* shape2 = new int[2] { 2, 2 };
    int ndim2 = 2;

    // Result tensor
    Tensor* result_tensor;

    void SetUp() override {
        tensor1 = new Tensor(data1, shape1, ndim1, "cuda");
        tensor2 = new Tensor(data2, shape2, ndim2, "cuda");
        result_tensor = Tensor::add_tensors(tensor1, tensor2);
    }

    void TearDown() override {
        delete tensor1;
        delete tensor2;
        delete result_tensor;
    }
};

TEST_F(CudaAddingTensorsTest, FirstIndexCheck)
{
    int indices[] = { 0, 0 };
    float item = result_tensor->get_item(indices);
    EXPECT_EQ(0.0f, item);
}

TEST_F(CudaAddingTensorsTest, SecondIndexCheck)
{
    int indices[] = { 0, 1 };
    float item = result_tensor->get_item(indices);
    EXPECT_EQ(2.0f, item);
}

TEST_F(CudaAddingTensorsTest, ThirdIndexCheck)
{
    int indices[] = { 1, 0 };
    float item = result_tensor->get_item(indices);
    EXPECT_EQ(4.0f, item);
}

TEST_F(CudaAddingTensorsTest, FourthIndexCheck)
{
    int indices[] = { 1, 1 };
    float item = result_tensor->get_item(indices);
    EXPECT_EQ(8.0f, item);
}