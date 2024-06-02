#include "../MiniTorchLib/Tensor.h"

#include "gtest/gtest.h"

class TensorTest : public ::testing::Test {
protected:
    Tensor* tensor;
    float* data_array = new float[4] { 1.0f, 2.0f,
                                       3.0f, 4.0f };
    int* shape_array = new int[2] { 2, 2 };
    int ndim = 2;

    void SetUp() override {
        tensor = create_tensor(data_array, shape_array, ndim);
    }

    void TearDown() override {
        delete tensor;
    }
};

TEST_F(TensorTest, GetItemFromFirstIndex) {
    int indices_array[] = { 0, 0 };
    float item = get_item(tensor, indices_array);
    EXPECT_EQ(1.0f, item);
}

TEST_F(TensorTest, GetItemFromSecondIndex) {
    int indices_array[] = { 0, 1 };
    float item = get_item(tensor, indices_array);
    EXPECT_EQ(2.0f, item);
}

TEST_F(TensorTest, GetItemFromThirdIndex) {
    int indices_array[] = { 1, 0 };
    float item = get_item(tensor, indices_array);
    EXPECT_EQ(3.0f, item);
}

TEST_F(TensorTest, GetItemFromFourthIndex) {
    int indices_array[] = { 1, 1 };
    float item = get_item(tensor, indices_array);
    EXPECT_EQ(4.0f, item);
}