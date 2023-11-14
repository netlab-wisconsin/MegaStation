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

// Uncomment to show values being compared
//#define TEST_REDUCTION_PRINT_VALUES 1


namespace
{

template <typename T> class host_op_sum
{
public:
    host_op_sum() : accum_(0)  {}
    void apply(const T& value) { accum_ += value; }
    T    get_result()          { return accum_; }
private:
    T accum_;
};

////////////////////////////////////////////////////////////////////////
// do_reduction_test_2D()
// Performs a reduction test using a 2-D input. The result is 1-D.
template <cuphyDataType_t TType,
          template <typename> class THostOp,
          typename                  TTol>
void do_reduction_test_2D(int             NUM_ROWS,
                          int             NUM_COLS,
                          int             dim,
                          TTol            error_tol)
{
    typedef cuphy::typed_tensor<TType, cuphy::pinned_alloc> tensor_p;
    typedef typename cuphy::type_traits<TType>::type        value_t;
    typedef THostOp<value_t>                                host_op_t;
    //------------------------------------------------------------------
    // Allocate tensors
    std::array<int, 2> SRC_DIM = {NUM_ROWS, NUM_COLS};
    std::array<int, 2> DST_DIM = SRC_DIM;
    DST_DIM[dim] = 1;
    tensor_p  tSrc(cuphy::tensor_layout(SRC_DIM.size(), SRC_DIM.data(), nullptr));
    tensor_p  tDst(cuphy::tensor_layout(DST_DIM.size(), DST_DIM.data(), nullptr));
    //printf("tSrc: %s\n", tSrc.desc().get_info().to_string().c_str());
    //printf("tDst: %s\n", tDst.desc().get_info().to_string().c_str());
    //------------------------------------------------------------------
    // Initialize the source tensor with random values
    cuphy::rng rng;
    rng.uniform(tSrc, 1, 10);
    //------------------------------------------------------------------
    // Perform the reduction operation using the library function
    tSrc.sum(tDst, dim);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Perform the operation on the host
    if(0 == dim)
    {
        for(int j = 0; j < NUM_COLS; ++j)
        {
            host_op_t s;
            for(int i = 0; i < NUM_ROWS; ++i)
            {
                s.apply(tSrc(i, j));
            }
            EXPECT_LT(std::abs(s.get_result() - tDst(0, j)), error_tol)
                << "NUM_ROWS = "   << NUM_ROWS
                << ", NUM_COLS = " << NUM_COLS
                << ", column "     << j
                << ", host = "     << s.get_result()
                << ", library = "  << tDst(0, j)
                << std::endl;
#if TEST_REDUCTION_PRINT_VALUES
            std::cout << "host result: "      << s.get_result()
                      << ", library result: " << tDst(0, j)
                      << ", error = "         << std::abs(s.get_result() - tDst(0, j))
                      << std::endl;
#endif
        }
    }
    else
    {
        for(int i = 0; i < NUM_ROWS; ++i)
        {
            host_op_t s;
            for(int j = 0; j < NUM_COLS; ++j)
            {
                s.apply(tSrc(i, j));
            }
            EXPECT_LT(std::abs(s.get_result() - tDst(i, 0)), error_tol)
                << "NUM_ROWS = "   << NUM_ROWS
                << ", NUM_COLS = " << NUM_COLS
                << ", row "        << i
                << ", host = "     << s.get_result()
                << ", library = "  << tDst(i, 0)
                << std::endl;
#if TEST_REDUCTION_PRINT_VALUES
            std::cout << "host result: " << s.get_result()
                      << ", library result: " << tDst(i, 0)
                      << ", error = " << std::abs(s.get_result() - tDst(i, 0))
                      << std::endl;
#endif
        }
    }
}

////////////////////////////////////////////////////////////////////////
// do_reduction_test_2D_bits()
// Performs a reduction test using a 2-D input of type CUPHY_BIT. The
// result is 1-D tensor with the count of the number of bits in that
// column.
void do_reduction_test_2D_bits(int NUM_ROWS,
                               int NUM_COLS)
{

    typedef cuphy::typed_tensor<CUPHY_BIT,   cuphy::pinned_alloc> tensor_bits_p;
    typedef cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> tensor_sum_p;

    //typedef typename cuphy::type_traits<TType>::type        value_t;
    //typedef THostOp<value_t>                                host_op_t;
    //------------------------------------------------------------------
    // Allocate tensors
    std::array<int, 2> SRC_DIM = {NUM_ROWS, NUM_COLS};
    std::array<int, 2> DST_DIM = {1,        NUM_COLS};
    tensor_bits_p  tSrc(cuphy::tensor_layout(SRC_DIM.size(), SRC_DIM.data(), nullptr),
                        cuphy::tensor_flags::align_coalesce);
    tensor_sum_p   tDst(cuphy::tensor_layout(DST_DIM.size(), DST_DIM.data(), nullptr));
    //printf("tSrc: %s\n", tSrc.desc().get_info().to_string().c_str());
    //printf("tDst: %s\n", tDst.desc().get_info().to_string().c_str());
    //------------------------------------------------------------------
    // Initialize the source tensor with random values
    cuphy::rng rng;
    rng.uniform(tSrc, 0, 1);
    //------------------------------------------------------------------
    // Perform the reduction operation using the library function
    tSrc.sum(tDst, 0);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Perform the operation on the host
    const int NUM_WORDS = (NUM_ROWS + 31) / 32;
    for(int j = 0; j < NUM_COLS; ++j)
    {
        uint32_t host_count = 0;
        for(int i = 0; i < NUM_WORDS; ++i)
        {
            //printf("[%i]: 0x%X (%u)\n", i, tSrc(i, j), __builtin_popcount(tSrc(i, j)));
            host_count += __builtin_popcount(tSrc(i, j));
        }
        uint32_t lib_count = tDst(0,j);
        //printf("host = %u, library = %u\n", host_count, lib_count);
        EXPECT_EQ(host_count, lib_count)
            << "NUM_ROWS = "   << NUM_ROWS
            << ", NUM_COLS = " << NUM_COLS
            << ", column "     << j
            << ", host = "     << host_count
            << ", library = "  << lib_count
            << std::endl;
#if TEST_REDUCTION_PRINT_VALUES
        std::cout << "host result: "      << host_count
                  << ", library result: " << lib_count
                  << ", error = "         << std::abs(static_cast<int64_t>(host_count) - static_cast<int64_t>(lib_count))
                  << std::endl;
#endif
    }
}

////////////////////////////////////////////////////////////////////////
// do_reduction_test_2D_bits_partial_word()
// Performs a reduction test using a 2-D input of type CUPHY_BIT. The
// result is 1-D tensor with the count of the number of bits in that
// column.
void do_reduction_test_2D_bits_partial_word(int NUM_FULL_ROWS,
                                            int NUM_PARTIAL_ROWS,
                                            int NUM_COLS)
{

    typedef cuphy::typed_tensor<CUPHY_BIT,   cuphy::pinned_alloc> tensor_bits_p;
    typedef cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> tensor_sum_p;
    typedef tensor_bits_p::tensor_ref_t                           tensor_bits_p_ref;

    //typedef typename cuphy::type_traits<TType>::type        value_t;
    //typedef THostOp<value_t>                                host_op_t;
    //------------------------------------------------------------------
    // Allocate tensors
    tensor_bits_p  tSrc(NUM_FULL_ROWS, NUM_COLS);
    tensor_sum_p   tDst(1,             NUM_COLS);
    //printf("tSrc: %s\n", tSrc.desc().get_info().to_string().c_str());
    //printf("tDst: %s\n", tDst.desc().get_info().to_string().c_str());
    //------------------------------------------------------------------
    // Initialize the source tensor with random values
    cuphy::rng rng;
    rng.uniform(tSrc, 0, 1);
    //------------------------------------------------------------------
    // Generate a tensor ref for a subset of the source tensor. We want
    // the subset to contain a partial word at the end of each column.
    cuphy::index_group grp(cuphy::index_range(0, NUM_PARTIAL_ROWS),
                           cuphy::dim_all());
    tensor_bits_p_ref  tSrc_s = tSrc.subset(grp);
    //------------------------------------------------------------------
    // Perform the reduction operation using the library function
    cuphy::tensor_reduction_sum(tDst, tSrc_s);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    //------------------------------------------------------------------
    // Perform the operation on the host
    const int NUM_WORDS = (NUM_PARTIAL_ROWS + 31) / 32;
    for(int j = 0; j < NUM_COLS; ++j)
    {
        uint32_t host_count = 0;
        for(int i = 0; i < NUM_WORDS; ++i)
        {
            uint32_t  src_mask    = 0xFFFFFFFF;
            const int ROW_COUNT = (i+1) * 32;
            if(ROW_COUNT > NUM_PARTIAL_ROWS)
            {
                const int NUM_BITS = NUM_PARTIAL_ROWS - (i * 32);
                src_mask = (1 << NUM_BITS) - 1;
            }
            uint32_t value = tSrc(i, j) & src_mask;
            //printf("[%i]: src = 0x%X, mask = 0x%X, value = 0x%X, popcount = %u\n",
            //       i,
            //       tSrc(i, j),
            //       src_mask,
            //       value,
            //       __builtin_popcount(value));
            host_count += __builtin_popcount(value);
        }
        uint32_t lib_count = tDst(0,j);
        //printf("host = %u, library = %u\n", host_count, lib_count);
        EXPECT_EQ(host_count, lib_count)
            << "NUM_FULL_ROWS = "      << NUM_FULL_ROWS
            << ", NUM_PARTIAL_ROWS = " << NUM_PARTIAL_ROWS
            << ", NUM_COLS = "         << NUM_COLS
            << ", column "             << j
            << ", host = "             << host_count
            << ", library = "          << lib_count
            << std::endl;
    }

}

} // namespace


////////////////////////////////////////////////////////////////////////
// Reduction.Sum
TEST(Reduction, Sum)
{
    do_reduction_test_2D<CUPHY_R_32F, host_op_sum>(64, 8, 0, 0.1f);
    do_reduction_test_2D<CUPHY_R_32F, host_op_sum>(80, 8, 0, 0.1f);
    do_reduction_test_2D<CUPHY_R_32F, host_op_sum>(8, 64, 1, 0.1f);
    do_reduction_test_2D<CUPHY_R_32F, host_op_sum>(1,  1, 0, 0.1f);
}

////////////////////////////////////////////////////////////////////////
// Reduction.Sum_Bits
TEST(Reduction, Sum_Bits)
{
    do_reduction_test_2D_bits(64,    48);
    do_reduction_test_2D_bits(35000, 11);
}

////////////////////////////////////////////////////////////////////////
// Reduction.Sum_Bits_Partial_Word
TEST(Reduction, Sum_Bits_Partial_Word)
{
    do_reduction_test_2D_bits_partial_word(64,  48, 16);
    do_reduction_test_2D_bits_partial_word(128, 97, 3);
}


////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
