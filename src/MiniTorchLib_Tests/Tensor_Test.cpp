#include "../MiniTorchLib/Tensor.h"

#include "gtest/gtest.h"

class TensorTest : public ::testing::Test {
protected:
    Tensor* tensor;
    float data_array[4] = { 1.0f, 2.0f,
                            3.0f, 4.0f };
    int shape_array[2] = { 2, 2 };
    int ndim = 2;

    void SetUp() override {
        tensor = create_tensor(data_array, shape_array, ndim);
    }

    void TearDown() override {
        free(tensor->strides);
        free(tensor);
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
    EXPECT_EQ(5.0f, item);
}

TEST_F(TensorTest, GetItemFromOutOfUpperBoundIndex) {
    int indices_array[] = { 1, 2 };
    EXPECT_THROW(get_item(tensor, indices_array), std::out_of_range);
}

TEST_F(TensorTest, GetItemFromOutOfLowerBoundIndex) {
    int indices_array[] = { 0, 1 };
    EXPECT_THROW(get_item(tensor, indices_array), std::out_of_range);
}