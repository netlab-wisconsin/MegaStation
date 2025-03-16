/**
 * @file test_cuphy_tensor.cc
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Test cases for cuphy_tensor.h
 * @version 0.1
 * @date 2023-11-16
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <gtest/gtest.h>

#include <cstdint>

#include "matrix/cuphy_tensor.h"
#include "matrix/matrix.h"

TEST(CuphyTensor, VecConstructor) {
  mega::CuphyTensor t(mega::CuphyTensor::kHalf, 4, mega::Matrix::kHost);
  EXPECT_EQ(t.nDim(), 1);
  EXPECT_EQ(t.allocType(), mega::Matrix::kHost);
  EXPECT_NE(t.raw(), nullptr);
  EXPECT_EQ(t.dim(0), 4);
  EXPECT_EQ(t.stride(0), 1);
  EXPECT_EQ(t.szBytes(0), 1 * sizeof(half));
  EXPECT_EQ(t.szBytes(1), 4 * sizeof(half));
}

TEST(CuphyTensor, ArrConstructor) {
  mega::CuphyTensor t(mega::CuphyTensor::kFloat, 4, 5, mega::Matrix::kDevice,
                      mega::CuphyTensor::kCoalesce);
  EXPECT_EQ(t.nDim(), 2);
  EXPECT_EQ(t.allocType(), mega::Matrix::kDevice);
  EXPECT_NE(t.raw(), nullptr);
  EXPECT_EQ(t.dim(0), 4);
  EXPECT_EQ(t.dim(1), 5);
  EXPECT_EQ(t.stride(0), 1);
  EXPECT_EQ(t.stride(1), 32);
  EXPECT_EQ(t.szBytes(0), 1 * sizeof(float));
  EXPECT_EQ(t.szBytes(1), 32 * sizeof(float));
  EXPECT_EQ(t.szBytes(2), 32 * 5 * sizeof(float));
}

TEST(CuphyTensor, CubeConstructor) {
  mega::CuphyTensor t(mega::CuphyTensor::kBit, 8, 5, 6, mega::Matrix::kHost,
                      mega::CuphyTensor::kCoalesce);
  EXPECT_EQ(t.nDim(), 3);
  EXPECT_EQ(t.allocType(), mega::Matrix::kHost);
  EXPECT_NE(t.raw(), nullptr);
  EXPECT_EQ(t.dim(0), 8);
  EXPECT_EQ(t.dim(1), 5);
  EXPECT_EQ(t.dim(2), 6);
  EXPECT_EQ(t.stride(0), 1);
  EXPECT_EQ(t.stride(1), 1024);
  EXPECT_EQ(t.stride(2), 1024 * 5);
  EXPECT_EQ(t.szBytes(0), 0);
  EXPECT_EQ(t.szBytes(1), 1024 / 8);
  EXPECT_EQ(t.szBytes(2), 1024 * 5 / 8);
  EXPECT_EQ(t.szBytes(3), 1024 * 5 * 6 / 8);
}

TEST(CuphyTensor, SubscriptOP) {
  mega::CuphyTensor t(mega::CuphyTensor::kBit, 8, 5, 3, mega::Matrix::kHost,
                      mega::CuphyTensor::kTight);

  uint32_t ref_data[] = {
      0,  1,  2,  3,  4,

      5,  6,  7,  8,  9,

      10, 11, 12, 13, 14,
  };
  memcpy(t.ptr(), ref_data, sizeof(ref_data));

  int ref_id = 0;
  for (int i = 0; i < 3; ++i) {
    for (int j = 0; j < 5; ++j) {
      EXPECT_EQ(*(uint32_t *)(t[i][j].ptr()), ref_data[ref_id++]);
    }
  }

  EXPECT_THROW(t[3], std::out_of_range);
  EXPECT_THROW(t[-1], std::out_of_range);
  EXPECT_THROW(t[0][0][0], std::underflow_error);
}

TEST(CuphyTensor, BitAllocVec) {
  mega::CuphyTensor t(mega::CuphyTensor::kBit, 8, mega::Matrix::kHost);
  EXPECT_EQ(t.nDim(), 1);
  EXPECT_EQ(t.allocType(), mega::Matrix::kHost);
  EXPECT_NE(t.raw(), nullptr);
  EXPECT_EQ(t.dim(0), 8);
  EXPECT_EQ(t.stride(0), 1);
  EXPECT_EQ(t.szBytes(0), 0);
  EXPECT_EQ(t.szBytes(1), 4);
}

TEST(CuphyTensor, BitAllocArr) {
  mega::CuphyTensor t(mega::CuphyTensor::kBit, 8, 5, mega::Matrix::kHost);
  EXPECT_EQ(t.nDim(), 2);
  EXPECT_EQ(t.allocType(), mega::Matrix::kHost);
  EXPECT_NE(t.raw(), nullptr);
  EXPECT_EQ(t.dim(0), 8);
  EXPECT_EQ(t.dim(1), 5);
  EXPECT_EQ(t.stride(0), 1);
  EXPECT_EQ(t.stride(1), 32);
  EXPECT_EQ(t.szBytes(0), 0);
  EXPECT_EQ(t.szBytes(1), 4);
  EXPECT_EQ(t.szBytes(2), 20);
}