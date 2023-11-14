/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <gtest/gtest.h>
#include "cuphy.hpp"

// Uncomment to show the mean and standard deviation for sampled distributions
//#define TEST_RNG_PRINT_STATISTICS 1

namespace
{

////////////////////////////////////////////////////////////////////////
// Welford's online algorithm to compute mean and variance using a
// single pass over the data.
template <typename T>
class welford_variance
{
public:
    welford_variance(T firstValue) :
        mean_(firstValue),
        N_(1),
        delta_(0),
        m_sq_(0)
    {
        //printf("%f\n", firstValue);
    }
    void update(T val)
    {
        ++N_;
        delta_ =  val - mean_;
        mean_  += delta_ / N_;
        m_sq_  += delta_ * (val - mean_);
        //printf("%f\n", val);
    }
    std::pair<T, T> get()
    {
        //printf("N_ = %lu\n", N_);
        // Dividing by N-1 here to return the SAMPLE variance.
        // Divide instead by N to get the POPULATION variance.
        return std::pair<T, T>(mean_, m_sq_ / (N_ - 1));
    }
private:
    T      delta_, m_sq_, mean_;
    size_t N_;
       
};

////////////////////////////////////////////////////////////////////////
// normal_dist_test()
// Templated function to perform tests on normal distributions
template <cuphyDataType_t TType,
          typename        THostType>
void normal_dist_test(THostType m, THostType stddev, THostType tol)
{
    typedef cuphy::typed_tensor<TType, cuphy::pinned_alloc> tensor_p;
    //------------------------------------------------------------------
    // Allocate a tensor
    const int NUM_ELEMENTS = 1024 * 1024;
    tensor_p  t(NUM_ELEMENTS);
    //------------------------------------------------------------------
    // Generate random values
    cuphy::rng rng;
    rng.normal(t, m, stddev, 0);
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Calculate mean and variance using Welford's online algorithm
    welford_variance<THostType> w(t(0));
    for(int i = 1; i < NUM_ELEMENTS; ++i)
    {
        w.update(t(i));
    }
    std::pair<THostType, THostType> stats = w.get();
#if TEST_RNG_PRINT_STATISTICS
    printf("mean = %8.3f, variance = %8.3f, stddev = %8.3f, stddev_rel_error = %.3f\n",
           stats.first,
           stats.second,
           std::sqrt(stats.second),
           (std::sqrt(stats.second) - stddev) / stddev);
#endif
    // Using tolerance for both mean bias and relative STDDEV error.
    // TODO: Better metrics.
    EXPECT_LT(std::abs(stats.first - m),                   tol);
    EXPECT_LT((std::sqrt(stats.second) - stddev) / stddev, tol);
}

////////////////////////////////////////////////////////////////////////
// complex_normal_dist_test()
// Templated function to perform tests on normal distributions with
// complex data types
template <cuphyDataType_t TType,
          typename        TValueType,
          typename        THostScalarType>
void complex_normal_dist_test(TValueType m, TValueType stddev, THostScalarType tol)
{
    typedef cuphy::typed_tensor<TType, cuphy::pinned_alloc> tensor_p;
    //------------------------------------------------------------------
    // Allocate a tensor
    const int NUM_ELEMENTS = 1024 * 1024;
    //const int NUM_ELEMENTS = 8;
    tensor_p  t(NUM_ELEMENTS);
    //------------------------------------------------------------------
    // Generate random values
    cuphy::rng rng;
    rng.normal(t, m, stddev, 0);
    cudaStreamSynchronize(0);
    //for(int i = 0; i < 8; ++i)
    //{
    //    printf("[%i]: %f  %f\n", i, t(i).x, t(i).y);
    //}
    //------------------------------------------------------------------
    // Calculate mean and variance using Welford's online algorithm
    welford_variance<THostScalarType> wReal(t(0).x);
    welford_variance<THostScalarType> wImag(t(0).y);
    for(int i = 1; i < NUM_ELEMENTS; ++i)
    {
        wReal.update(t(i).x);
        wImag.update(t(i).y);
    }
    std::pair<THostScalarType, THostScalarType> statsReal = wReal.get();
    std::pair<THostScalarType, THostScalarType> statsImag = wImag.get();
#if TEST_RNG_PRINT_STATISTICS
    printf("real: mean = %f, variance = %f, stddev = %f, stddev_rel_error = %.3f\n",
           statsReal.first,
           statsReal.second,
           std::sqrt(statsReal.second),
           (std::sqrt(statsReal.second) - stddev.x) / stddev.x);
    printf("imag: mean = %f, variance = %f, stddev = %f, stddev_rel_error = %.3f\n",
           statsImag.first,
           statsImag.second,
           std::sqrt(statsImag.second),
           (std::sqrt(statsImag.second) - stddev.y) / stddev.y);
#endif
    // Using tolerance for both mean bias and relative STDDEV error.
    // TODO: Better metrics.
    EXPECT_LT(std::abs(statsReal.first - m.x),                     tol);
    EXPECT_LT((std::sqrt(statsReal.second) - stddev.x) / stddev.x, tol);
    EXPECT_LT(std::abs(statsImag.first - m.y),                     tol);
    EXPECT_LT((std::sqrt(statsImag.second) - stddev.y) / stddev.y, tol);
}

////////////////////////////////////////////////////////////////////////
// uniform_dist_test()
// Templated function to perform tests on uniform distributions
template <cuphyDataType_t TType,
          typename        THostType>
void uniform_dist_test(THostType a, THostType b, THostType tol)
{
    typedef cuphy::typed_tensor<TType, cuphy::pinned_alloc> tensor_p;
    //------------------------------------------------------------------
    // Allocate a tensor
    const int NUM_ELEMENTS = 128 * 1024;
    tensor_p  t(NUM_ELEMENTS);
    //------------------------------------------------------------------
    // Generate random values
    cuphy::rng rng;
    rng.uniform(t, a, b, 0);
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Calculate mean and variance using Welford's online algorithm
    welford_variance<THostType> w(t(0));
    for(int i = 1; i < NUM_ELEMENTS; ++i)
    {
        EXPECT_GE(static_cast<THostType>(t(i)), a);
        EXPECT_LE(static_cast<THostType>(t(i)), b);
        w.update(t(i));
    }
    std::pair<THostType, THostType> stats = w.get();
    THostType m = (a + b) / 2;
    THostType stddev = std::sqrt((b - a) * (b - a) / 12);
#if TEST_RNG_PRINT_STATISTICS
    printf("mean = %8.3f, variance = %8.3f, stddev = %8.3f (expected %8.3f), stddev_rel_error = %.3f\n",
           stats.first,
           stats.second,
           std::sqrt(stats.second),
           stddev,
           (std::sqrt(stats.second) - stddev) / stddev);
#endif
    // Using tolerance for both mean bias and relative STDDEV error.
    // TODO: Better metrics.
    EXPECT_LT(std::abs(stats.first - m),                   tol);
    EXPECT_LT((std::sqrt(stats.second) - stddev) / stddev, tol);
}
  
////////////////////////////////////////////////////////////////////////
// complex_uniform_dist_test()
// Templated function to perform tests on uniform distributions with
// complex data types
template <cuphyDataType_t TType,
          typename        TValueType,
          typename        THostScalarType>
void complex_uniform_dist_test(TValueType minVal, TValueType maxVal, THostScalarType tol)
{
    typedef cuphy::typed_tensor<TType, cuphy::pinned_alloc> tensor_p;
    //------------------------------------------------------------------
    // Allocate a tensor
    const int NUM_ELEMENTS = 1024 * 1024;
    //const int NUM_ELEMENTS = 8;
    tensor_p  t(NUM_ELEMENTS);
    //------------------------------------------------------------------
    // Generate random values
    cuphy::rng rng;
    rng.uniform(t, minVal, maxVal, 0);
    cudaStreamSynchronize(0);
    //for(int i = 0; i < 8; ++i)
    //{
    //    printf("[%i]: %f  %f\n", i, t(i).x, t(i).y);
    //}
    //------------------------------------------------------------------
    // Calculate mean and variance using Welford's online algorithm
    welford_variance<THostScalarType> wReal(t(0).x);
    welford_variance<THostScalarType> wImag(t(0).y);
    for(int i = 1; i < NUM_ELEMENTS; ++i)
    {
        wReal.update(t(i).x);
        wImag.update(t(i).y);
    }
    std::pair<THostScalarType, THostScalarType> statsReal = wReal.get();
    std::pair<THostScalarType, THostScalarType> statsImag = wImag.get();
    THostScalarType m_real      = (minVal.x + maxVal.x) / 2;
    THostScalarType m_imag      = (minVal.y + maxVal.y) / 2;
    THostScalarType stddev_real = std::sqrt((maxVal.x - minVal.x) * (maxVal.x - minVal.x) / 12);
    THostScalarType stddev_imag = std::sqrt((maxVal.y - minVal.y) * (maxVal.y - minVal.y) / 12);
#if TEST_RNG_PRINT_STATISTICS
    printf("real: mean = %f, variance = %f, stddev = %f, stddev_rel_error = %.3f\n",
           statsReal.first,
           statsReal.second,
           std::sqrt(statsReal.second),
           (std::sqrt(statsReal.second) - stddev_real) / stddev_real);
    printf("imag: mean = %f, variance = %f, stddev = %f, stddev_rel_error = %.3f\n",
           statsImag.first,
           statsImag.second,
           std::sqrt(statsImag.second),
           (std::sqrt(statsImag.second) - stddev_imag) / stddev_imag);
#endif
    // Using tolerance for both mean bias and relative STDDEV error.
    // TODO: Better metrics.
    EXPECT_LT(std::abs(statsReal.first - m_real),                        tol);
    EXPECT_LT((std::sqrt(statsReal.second) - stddev_real) / stddev_real, tol);
    EXPECT_LT(std::abs(statsImag.first - m_imag),                        tol);
    EXPECT_LT((std::sqrt(statsImag.second) - stddev_imag) / stddev_imag, tol);
}

} // namespace


////////////////////////////////////////////////////////////////////////
// RNG.Normal
TEST(RNG, Normal)
{
    normal_dist_test<CUPHY_R_64F, double>(0.0,  1.0,  0.1);
    normal_dist_test<CUPHY_R_32F, double>(0.0,  1.0,  0.1);
    normal_dist_test<CUPHY_R_16F, float> (0.0f, 1.0f, 0.1f);

    normal_dist_test<CUPHY_R_64F, double>(  0.0, 50.0,  0.1);
    normal_dist_test<CUPHY_R_32F, double>(-10.0,  5.0,  0.1);
    normal_dist_test<CUPHY_R_16F, float> ( 2.0f,  0.1f, 0.1f);
}

////////////////////////////////////////////////////////////////////////
// RNG.Uniform
TEST(RNG, Uniform)
{
    uniform_dist_test<CUPHY_R_8I,  float> (   -100.0f,    100.0f,   0.5f);
    uniform_dist_test<CUPHY_R_8I,  float> (      0.0f,    100.0f,   0.5f);
    uniform_dist_test<CUPHY_R_16I, float> (  -1000.0f,   1000.0f,  10.0f);
    uniform_dist_test<CUPHY_R_16U, float> (      0.0f,   1000.0f,   5.0f);
    uniform_dist_test<CUPHY_R_32I, float> (-100000.0f, 100000.0f, 500.0f);
    uniform_dist_test<CUPHY_R_32U, float> (      0.0f, 100000.0f, 500.0f);
    
    uniform_dist_test<CUPHY_R_16F, float> (0.0f,   1.0f, 0.1f);
    uniform_dist_test<CUPHY_R_32F, double>(0.0,    1.0,  0.1);
    uniform_dist_test<CUPHY_R_64F, double>(0.0,    1.0,  0.1);
    
    uniform_dist_test<CUPHY_R_32F, double>( 10.0, 20.0, 0.1);
    uniform_dist_test<CUPHY_R_32F, double>(-10.0, 10.0, 0.1);
    uniform_dist_test<CUPHY_R_16F, float> ( 1.0f, 5.0f, 0.1f);
}

////////////////////////////////////////////////////////////////////////
// RNG.Bits
TEST(RNG, Bits)
{
    typedef cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc> tensor_p;
    //------------------------------------------------------------------
    // Allocate a tensor
    const int NUM_ELEMENTS = (32 * 32) + 16;  // 1040
    const int NUM_COLUMNS  = 8;
    const int NUM_WORDS    = (NUM_ELEMENTS + 31) / 32;
    // Specify strides explicitly for CUPHY_BIT tensors
    tensor_p  t(NUM_ELEMENTS, NUM_COLUMNS, cuphy::tensor_flags::align_coalesce);
    //------------------------------------------------------------------
    // Generate random values
    cuphy::rng rng(0xF00DF00D);
    rng.uniform(t, 0, 1, 0);
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Count the total number of set bits
    size_t cBits = 0;
    for(int i = 0; i < NUM_WORDS; ++i)
    {
        //printf("[%i]: 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X 0x%08X\n",
        //       i,
        //       t(i, 0), t(i, 1), t(i, 2), t(i, 3), t(i, 4), t(i, 5), t(i, 6), t(i, 7));
        for(int j = 0; j < NUM_COLUMNS; ++j)
        {
            cBits += __builtin_popcount(t(i, j));
        }
    }
    //------------------------------------------------------------------
    // Compare the number of bit set to 50%
    float ratioSet = static_cast<float>(cBits) / (NUM_ELEMENTS * NUM_COLUMNS);
    //printf("Total bits: %i    Set bits: %lu   Ratio: %f\n",
    //       NUM_ELEMENTS * NUM_COLUMNS,
    //       cBits,
    //       ratioSet);
    EXPECT_LT(std::abs(ratioSet) - 0.5f, 0.1f);
    //------------------------------------------------------------------
    // Make sure that the high order bits in the last word are zero
    int      validBits   = (NUM_WORDS * 32) - NUM_ELEMENTS;
    uint32_t invalidMask = ~((1 << validBits) - 1);
    for(int j = 0; j < NUM_COLUMNS; ++j)
    {
        uint32_t val = t(NUM_WORDS - 1, j);
        EXPECT_TRUE(0 == (val & invalidMask));
    }
}

////////////////////////////////////////////////////////////////////////
// RNG.ComplexNormal
TEST(RNG, ComplexNormal)
{
    // Double precision, zero mean, unity standard deviation (each component)
    {
        cuDoubleComplex m64_0      = make_cuDoubleComplex(0.0, 0.0);
        cuDoubleComplex stddev64_1 = make_cuDoubleComplex(1.0, 1.0);
        complex_normal_dist_test<CUPHY_C_64F>(m64_0,  stddev64_1,  0.1);
    }
    // Single precision, zero mean, unity standard deviation (each component)
    {
        cuComplex m32_0      = make_cuFloatComplex(0.0, 0.0);
        cuComplex stddev32_1 = make_cuFloatComplex(1.0, 1.0);
        complex_normal_dist_test<CUPHY_C_32F>(m32_0,  stddev32_1,  0.1);
    }
    // Half precision, zero mean, unity standard deviation (each component)
    {
        cuComplex m32_0      = make_cuFloatComplex(0.0, 0.0);
        cuComplex stddev32_1 = make_cuFloatComplex(1.0, 1.0);
        complex_normal_dist_test<CUPHY_C_16F>(m32_0,  stddev32_1,  0.1);
    }
    // Single precision, zero mean, different standard deviations
    {
        cuComplex m32_0      = make_cuFloatComplex(0.0, 0.0);
        cuComplex stddev32_1 = make_cuFloatComplex(0.9, 0.1);
        complex_normal_dist_test<CUPHY_C_32F>(m32_0,  stddev32_1,  0.1);
    }
}

////////////////////////////////////////////////////////////////////////
// RNG.ComplexUniform
TEST(RNG, ComplexUniform)
{
    // Double precision
    {
        cuDoubleComplex min64 = make_cuDoubleComplex(0.0, 0.0);
        cuDoubleComplex max64 = make_cuDoubleComplex(1.0, 1.0);
        complex_uniform_dist_test<CUPHY_C_64F>(min64,  max64,  0.1);
    }
    // Single precision
    {
        cuComplex min32 = make_cuFloatComplex(0.0, 0.0);
        cuComplex max32 = make_cuFloatComplex(1.0, 1.0);
        complex_uniform_dist_test<CUPHY_C_32F>(min32,  max32,  0.1);
    }
    // Half precision
    {
        cuComplex min32 = make_cuFloatComplex(0.0, 0.0);
        cuComplex max32 = make_cuFloatComplex(1.0, 1.0);
        complex_uniform_dist_test<CUPHY_C_16F>(min32,  max32,  0.1);
    }
    // int32_t
    {
        int2 min32 = { 1000,   0};
        int2 max32 = { 2000, 500};
        complex_uniform_dist_test<CUPHY_C_32I>(min32,  max32,  1.0);
    }
    // int16_t
    {
        short2 min16 = { 1000,   0};
        short2 max16 = { 2000, 500};
        complex_uniform_dist_test<CUPHY_C_16I>(min16,  max16,  1.0);
    }
    // int8_t
    {
        char2 min8 = { 0,   0};
        char2 max8 = { 100, 100};
        complex_uniform_dist_test<CUPHY_C_8I>(min8,  max8,  1.0);
    }
    // uint32_t
    {
        uint2 min32 = { 1000,   0};
        uint2 max32 = { 2000, 500};
        complex_uniform_dist_test<CUPHY_C_32U>(min32,  max32,  1.0);
    }
    // uint16_t
    {
        ushort2 min16 = { 1000,   0};
        ushort2 max16 = { 2000, 500};
        complex_uniform_dist_test<CUPHY_C_16U>(min16,  max16,  1.0);
    }
    // uint8_t
    {
        uchar2 min8 = { 0,   0};
        uchar2 max8 = { 250, 250};
        complex_uniform_dist_test<CUPHY_C_8U>(min8,  max8,  1.0);
    }
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
