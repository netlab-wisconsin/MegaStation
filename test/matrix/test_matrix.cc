/**
 * @file test_matrix.cc
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Test cases for matrix.h
 * @version 0.1
 * @date 2023-11-16
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <gtest/gtest.h>

#include "matrix/matrix.h"

TEST(Matrix, VecConstructor) {
  mega::Matrix m(sizeof(short), 3, mega::Matrix::kHost);
  EXPECT_EQ(m.nDim(), 1);
  EXPECT_EQ(m.allocType(), mega::Matrix::kHost);
  EXPECT_NE(m.raw(), nullptr);
  EXPECT_EQ(m.dim(0), 3);
  EXPECT_EQ(m.stride(0), 1);
  EXPECT_EQ(m.szBytes(0), 1 * sizeof(short));
  EXPECT_EQ(m.szBytes(1), 3 * sizeof(short));
}

TEST(Matrix, ArrConstructor) {
  mega::Matrix m(sizeof(short), 3, 4, mega::Matrix::kDevice);
  EXPECT_EQ(m.nDim(), 2);
  EXPECT_EQ(m.allocType(), mega::Matrix::kDevice);
  EXPECT_NE(m.raw(), nullptr);
  EXPECT_EQ(m.dim(0), 3);
  EXPECT_EQ(m.dim(1), 4);
  EXPECT_EQ(m.stride(0), 1);
  EXPECT_EQ(m.stride(1), 3);
  EXPECT_EQ(m.szBytes(0), 1 * sizeof(short));
  EXPECT_EQ(m.szBytes(1), 3 * sizeof(short));
  EXPECT_EQ(m.szBytes(2), 3 * 4 * sizeof(short));
}

TEST(Matrix, CubeConstructor) {
  mega::Matrix m(sizeof(short), 3, 4, 5, mega::Matrix::kDevice);
  EXPECT_EQ(m.nDim(), 3);
  EXPECT_EQ(m.allocType(), mega::Matrix::kDevice);
  EXPECT_NE(m.raw(), nullptr);
  EXPECT_EQ(m.dim(0), 3);
  EXPECT_EQ(m.dim(1), 4);
  EXPECT_EQ(m.dim(2), 5);
  EXPECT_EQ(m.stride(0), 1);
  EXPECT_EQ(m.stride(1), 3);
  EXPECT_EQ(m.stride(2), 3 * 4);
  EXPECT_EQ(m.szBytes(0), 1 * sizeof(short));
  EXPECT_EQ(m.szBytes(1), 3 * sizeof(short));
  EXPECT_EQ(m.szBytes(2), 3 * 4 * sizeof(short));
  EXPECT_EQ(m.szBytes(3), 3 * 4 * 5 * sizeof(short));
}

TEST(Matrix, SubscriptOP) {
  mega::Matrix m(sizeof(int), 5, 3, 4, mega::Matrix::kHPin);

  int ref_data[] = {
      0,  1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14,

      15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29,

      30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44,

      45, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59,
  };
  memcpy(m.ptr(), ref_data, sizeof(ref_data));

  int ref_id = 0;
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 3; ++j) {
      for (int k = 0; k < 5; ++k) {
        EXPECT_EQ(*(m[i][j][k].ptr<int>()), ref_data[ref_id++]);
      }
    }
  }

  EXPECT_THROW(m[0][0][0][0], std::out_of_range);
  EXPECT_THROW(m[4], std::out_of_range);

  m[0][0][0].deref<int>(0) = -100;
  EXPECT_EQ(*(m[0][0][0].ptr<int>()), -100);
}