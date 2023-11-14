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

namespace
{

////////////////////////////////////////////////////////////////////////
// magnitude_squared()
// Returns the magnitude squared for a complex value
template <typename T> double magnitude_squared(T a)
{
    return (a.x * a.x) * (a.y * a.y);
}

////////////////////////////////////////////////////////////////////////
// host_add
// Template function for host addition of real values
template <typename T> struct host_add { static T operate(T a, T b) { return (a + b); } };

// Disable narrowing warnings here, since we are using inputs with a
// limited range.
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wnarrowing"
////////////////////////////////////////////////////////////////////////
// host_complex_add
// Template function for host addition of complex values
template <typename T> struct host_complex_add { static T operate(T a, T b) { return T{a.x + b.x, a.y + b.y}; } };
#pragma GCC diagnostic pop

////////////////////////////////////////////////////////////////////////
// host_xor
struct host_xor { static uint32_t operate(uint32_t a, uint32_t b) { return (a ^ b); } };

////////////////////////////////////////////////////////////////////////
// host_cmp_eq
// Host functor to compare real values
template <typename T> struct host_cmp_eq
{
    static void cmp(T a, T b)
    {
        EXPECT_EQ(a, b);
    }
};

////////////////////////////////////////////////////////////////////////
// host_cmp_abs_error
// Host functor to compare real values
template <typename T> struct host_cmp_abs_error
{
    template <typename TTol>
    static void cmp(T a, T b, TTol tol)
    {
        EXPECT_LE(std::abs(static_cast<double>(a) - static_cast<double>(b)),
                  tol);
    }
};
////////////////////////////////////////////////////////////////////////
// host_cmp_mag_squared
// Host functor to compare the magnitude squared of complex values
template <typename T> struct host_cmp_mag_squared
{
    template <typename TTol>
    static void cmp(T a, T b, TTol tol)
    {
        TTol mag_a = magnitude_squared(a);
        TTol mag_b = magnitude_squared(b);
        //printf("mag_a = %f, mag_b = %f\n", mag_a, mag_b);
        EXPECT_LT(std::abs(mag_a - mag_b), tol);
    }
};

////////////////////////////////////////////////////////////////////////
// range_gen_bits
// Range generator for bit tensors. (Values are ignored by the library)
struct range_gen_bits
{
    static int min() { return 0; }
    static int max() { return 1; }
};

////////////////////////////////////////////////////////////////////////
// range_gen_real
// Uniform distribution range generator for real values
struct range_gen_real
{
    static int min() { return 1; }
    static int max() { return 10; }
};

////////////////////////////////////////////////////////////////////////
// range_gen_complex
// Uniform distribution range generator for complex values
struct range_gen_complex
{
    static int2 min() { return int2{1, 1}; }
    static int2 max() { return int2{10, 10}; }
};

////////////////////////////////////////////////////////////////////////
// tensor_op_add
// Calls dst.add(A, B)
struct tensor_op_add
{
    template <class TDst, class TSrc>
    static void do_operation(TDst& dst, TSrc& srcA, TSrc& srcB)
    {
        dst.add(srcA, srcB);
    }
};

////////////////////////////////////////////////////////////////////////
// tensor_op_xor
// Calls dst.xor_op(A, B)
struct tensor_op_xor
{
    template <class TDst, class TSrcA, class TSrcB>
    static void do_operation(TDst& dst, TSrcA& srcA, TSrcB& srcB)
    {
        cuphy::tensor_xor(dst, srcA, srcB);
    }
};

////////////////////////////////////////////////////////////////////////
// do_elementwise_test()
template <cuphyDataType_t           TType,
          class                     TRangeGen,
          class                     TOp,
          template <typename> class THostOp,
          template <typename> class THostCompare>
void do_elementwise_test(int NUM_ROWS, int NUM_COLS)
{
    typedef cuphy::typed_tensor<TType, cuphy::pinned_alloc> tensor_p;
    typedef typename tensor_p::element_t                    element_t;
    typedef THostOp<element_t>                              host_op_t;
    typedef THostCompare<element_t>                         host_cmp_t;
    typedef TRangeGen                                       range_gen_t;
    //------------------------------------------------------------------
    // Allocate source and destination tensors
    tensor_p                 tSrcA(NUM_ROWS, NUM_COLS);
    tensor_p                 tSrcB(NUM_ROWS, NUM_COLS);
    tensor_p                 tDst(NUM_ROWS, NUM_COLS);
    //------------------------------------------------------------------
    // Initialize the source tensors with random values
    cuphy::rng rng;
    rng.uniform(tSrcA, range_gen_t::min(), range_gen_t::max());
    rng.uniform(tSrcB, range_gen_t::min(), range_gen_t::max());
    //------------------------------------------------------------------
    // Perform the requested operation
    TOp::do_operation(tDst, tSrcA, tSrcB);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Compare results
    for(int i = 0; i < NUM_ROWS; ++i)
    {
        for(int j = 0; j < NUM_COLS; ++j)
        {
            host_cmp_t::cmp(tDst(i, j),
                            host_op_t::operate(tSrcA(i, j), tSrcB(i, j)),
                            0.01f);
        }
    }
}

////////////////////////////////////////////////////////////////////////
// do_elementwise_bit_test()
template <class TOp,
          class THostOp>
void do_elementwise_bit_test(int NUM_ROWS, int NUM_COLS)
{
    typedef cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc> tensor_p;
    typedef THostOp                                             host_op_t;
    typedef host_cmp_eq<uint32_t>                               host_cmp_t;
    //------------------------------------------------------------------
    // Allocate source and destination tensors
    const std::array<int, 2> SRC_DIMS = {{NUM_ROWS, NUM_COLS}};
    tensor_p                 tSrcA(NUM_ROWS, NUM_COLS);
    tensor_p                 tSrcB(NUM_ROWS, NUM_COLS);
    tensor_p                 tDst(NUM_ROWS, NUM_COLS);
    //------------------------------------------------------------------
    // Initialize the source tensors with random values
    cuphy::rng rng;
    rng.uniform(tSrcA, 0, 1);
    rng.uniform(tSrcB, 0, 1);
    //------------------------------------------------------------------
    // Perform the requested operation
    TOp::do_operation(tDst, tSrcA, tSrcB);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Compare results
    int NUM_WORDS = (NUM_ROWS + 31) / 32;
    for(int i = 0; i < NUM_WORDS; ++i)
    {
        for(int j = 0; j < SRC_DIMS[1]; ++j)
        {
            host_cmp_t::cmp(tDst(i, j),
                            host_op_t::operate(tSrcA(i, j), tSrcB(i, j)));
            //printf("[%i, %i]: A: 0x%X, B: 0x%X, Output: 0x%X\n", i, j, tSrcA(i, j), tSrcB(i, j), tDst(i, j));
        }
    }
}

////////////////////////////////////////////////////////////////////////
// do_elementwise_bit_partial_word_test()
// Test for correct handling of elementwise operations at the end of
// columns for tensors of type CUPHY_BIT.
template <class TOp,
          class THostOp>
void do_elementwise_bit_partial_word_test(int NUM_ROWS_FULL,
                                          int NUM_ROWS_PARTIAL,
                                          int NUM_COLS)
{
    typedef cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc> tensor_p;
    typedef tensor_p::tensor_ref_t                              tensor_ref_p;
    typedef THostOp                                             host_op_t;
    typedef host_cmp_eq<uint32_t>                               host_cmp_t;
    //------------------------------------------------------------------
    // Allocate source and destination tensors
    tensor_p                 tSrcA(NUM_ROWS_FULL,    NUM_COLS);
    tensor_p                 tSrcB(NUM_ROWS_PARTIAL, NUM_COLS);
    tensor_p                 tDst(NUM_ROWS_PARTIAL,  NUM_COLS);
    //------------------------------------------------------------------
    // Initialize the source tensors with random values
    cuphy::rng rng;
    rng.uniform(tSrcA, 0, 1);
    rng.uniform(tSrcB, 0, 1);
    //------------------------------------------------------------------
    // Generate a tensor ref for a subset of tensor A. We want the
    // subset to contain a partial word at the end of the column.
    cuphy::index_group grp(cuphy::index_range(0, NUM_ROWS_PARTIAL),
                           cuphy::dim_all());
    tensor_ref_p       tSrcA_s = tSrcA.subset(grp);
    //------------------------------------------------------------------
    // Perform the requested operation
    TOp::do_operation(tDst, tSrcA_s, tSrcB);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Compare results
    int NUM_WORDS = (NUM_ROWS_PARTIAL + 31) / 32;
    for(int i = 0; i < NUM_WORDS; ++i)
    {
        uint32_t  A_mask    = 0xFFFFFFFF;
        const int ROW_COUNT = (i+1) * 32;
        if(ROW_COUNT > NUM_ROWS_PARTIAL)
        {
            const int NUM_BITS = NUM_ROWS_PARTIAL - (i * 32);
            A_mask = (1 << NUM_BITS) - 1;
        }
        for(int j = 0; j < NUM_COLS; ++j)
        {
            uint32_t A = tSrcA(i, j) & A_mask;
            uint32_t B = tSrcB(i, j);
            host_cmp_t::cmp(tDst(i, j),
                            host_op_t::operate(A, B));
            //printf("[%i, %i]: A: 0x%X, mask: 0x%X, B: 0x%X, Ref: 0x%X, Output: 0x%X\n",
            //       i,
            //       j,
            //       A,
            //       A_mask,
            //       B,
            //       host_op_t::operate(A, B),
            //       tDst(i, j));
        }
    }
}

////////////////////////////////////////////////////////////////////////
// do_elementwise_broadcast_test()
template <cuphyDataType_t           TType,
          class                     TRangeGen,
          class                     TOp,
          template <typename> class THostOp,
          template <typename> class THostCompare>
void do_elementwise_broadcast_test(int NUM_ROWS, int NUM_COLS)
{
    typedef cuphy::typed_tensor<TType, cuphy::pinned_alloc> tensor_p;
    typedef typename tensor_p::element_t                    element_t;
    typedef THostOp<element_t>                              host_op_t;
    typedef THostCompare<element_t>                         host_cmp_t;
    typedef TRangeGen                                       range_gen_t;
    //------------------------------------------------------------------
    // Allocate source and destination tensors
    tensor_p                 tSrcA(NUM_ROWS, NUM_COLS);
    tensor_p                 tSrcB(NUM_ROWS);
    tensor_p                 tDst(NUM_ROWS, NUM_COLS);
    //------------------------------------------------------------------
    // Initialize the source tensors with random values
    cuphy::rng rng;
    rng.uniform(tSrcA, range_gen_t::min(), range_gen_t::max());
    rng.uniform(tSrcB, range_gen_t::min(), range_gen_t::max());
    //------------------------------------------------------------------
    // Perform the requested operation
    TOp::do_operation(tDst, tSrcA, tSrcB);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Compare results
    for(int i = 0; i < NUM_ROWS; ++i)
    {
        for(int j = 0; j < NUM_COLS; ++j)
        {
            host_cmp_t::cmp(tDst(i, j),
                            host_op_t::operate(tSrcA(i, j), tSrcB(i)),
                            0.01f);
            //printf("[%i, %i]: A: %f, B: %f, Output: %f, Compare: %f\n",
            //       i,
            //       j,
            //       tSrcA(i, j),
            //       tSrcB(i),
            //       tDst(i, j),
            //       host_op_t::operate(tSrcA(i, j), tSrcB(i)));
        }
    }
}

////////////////////////////////////////////////////////////////////////
// do_elementwise_broadcast_bit_test()
template <class TOp,
          class THostOp>
void do_elementwise_broadcast_bit_test(int NUM_ROWS, int NUM_COLS)
{
    typedef cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc> tensor_p;
    typedef THostOp                                             host_op_t;
    typedef host_cmp_eq<uint32_t>                               host_cmp_t;
    //------------------------------------------------------------------
    // Allocate source and destination tensors
    tensor_p                 tSrcA(NUM_ROWS, NUM_COLS);
    tensor_p                 tSrcB(NUM_ROWS);
    tensor_p                 tDst(NUM_ROWS, NUM_COLS);
    //------------------------------------------------------------------
    // Initialize the source tensors with random values
    cuphy::rng rng;
    rng.uniform(tSrcA, 0, 1);
    rng.uniform(tSrcB, 0, 1);
    //------------------------------------------------------------------
    // Perform the requested operation
    TOp::do_operation(tDst, tSrcA, tSrcB);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Compare results
    int NUM_WORDS = (NUM_ROWS + 31) / 32;
    for(int i = 0; i < NUM_WORDS; ++i)
    {
        for(int j = 0; j < NUM_COLS; ++j)
        {
            host_cmp_t::cmp(tDst(i, j),
                            host_op_t::operate(tSrcA(i, j), tSrcB(i)));
            //printf("[%i, %i]: A: 0x%X, B: 0x%X, Output: 0x%X\n", i, j, tSrcA(i, j), tSrcB(i), tDst(i, j));
        }
    }
}

} // namespace


////////////////////////////////////////////////////////////////////////
// ElementWise.Add
TEST(ElementWise, Add)
{
    do_elementwise_test<CUPHY_R_8I,  range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (64,  16);
    do_elementwise_test<CUPHY_C_8I,  range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(128, 1);
    do_elementwise_test<CUPHY_R_8U,  range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (37,  19);
    do_elementwise_test<CUPHY_C_8U,  range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(8,   8);
    do_elementwise_test<CUPHY_R_16I, range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (64, 16);
    do_elementwise_test<CUPHY_C_16I, range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(64, 16);
    do_elementwise_test<CUPHY_R_16U, range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (64, 16);
    do_elementwise_test<CUPHY_C_16U, range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(64, 16);
    do_elementwise_test<CUPHY_R_32I, range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (64, 16);
    do_elementwise_test<CUPHY_C_32I, range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(64, 16);
    do_elementwise_test<CUPHY_R_32U, range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (64, 16);
    do_elementwise_test<CUPHY_C_32U, range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(64, 16);
    do_elementwise_test<CUPHY_R_16F, range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (64, 16);
    do_elementwise_test<CUPHY_C_16F, range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(64, 16);
    do_elementwise_test<CUPHY_R_32F, range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (64, 16);
    do_elementwise_test<CUPHY_C_32F, range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(64, 16);
    do_elementwise_test<CUPHY_R_64F, range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (64, 16);
    do_elementwise_test<CUPHY_C_64F, range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(64, 16);
}

////////////////////////////////////////////////////////////////////////
// ElementWise.XOR
TEST(ElementWise, XOR)
{
    do_elementwise_bit_test<tensor_op_xor, host_xor>(1024, 16);
}

////////////////////////////////////////////////////////////////////////
// ElementWise.XOR_Partial
TEST(ElementWise, XOR_Partial)
{
    do_elementwise_bit_partial_word_test<tensor_op_xor, host_xor>(64,  48,  16);
    do_elementwise_bit_partial_word_test<tensor_op_xor, host_xor>(264, 164, 8);
}

////////////////////////////////////////////////////////////////////////
// ElementWise.BroadcastBit
TEST(ElementWise, BroadcastBit)
{
    do_elementwise_broadcast_bit_test<tensor_op_xor, host_xor>(1024, 16);
}

////////////////////////////////////////////////////////////////////////
// ElementWise.Broadcast
TEST(ElementWise, Broadcast)
{
    do_elementwise_broadcast_test<CUPHY_R_32F, range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (64,   11);
    do_elementwise_broadcast_test<CUPHY_C_32F, range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(113,  66);
    do_elementwise_broadcast_test<CUPHY_R_16F, range_gen_real,    tensor_op_add, host_add,         host_cmp_abs_error>  (97,   3);
    do_elementwise_broadcast_test<CUPHY_C_16F, range_gen_complex, tensor_op_add, host_complex_add, host_cmp_mag_squared>(1024, 83);
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
