/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
// do_convert_test_to_bits()
template <cuphyDataType_t TType>
void do_convert_test_to_bits(int NUM_ROWS,
                             int NUM_COLS)
{
    typedef cuphy::typed_tensor<TType,     cuphy::pinned_alloc> tensor_p;
    typedef cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc> tensor_bit_p;

    //const std::array<int, 2> SRC_DIMS = {{NUM_ROWS, NUM_COLS}};
    //------------------------------------------------------------------
    // Allocate tensors
    tensor_p     tSrc(NUM_ROWS, NUM_COLS);
    tensor_bit_p tDst(NUM_ROWS, NUM_COLS, cuphy::tensor_flags::align_coalesce);
    //------------------------------------------------------------------
    // Initialize the source tensor with random values
    cuphy::rng rng;
    rng.uniform(tSrc, 0, 1);
    //------------------------------------------------------------------
    // Convert to the CUPHY_BIT type
    cuphy::tensor_convert(tDst, tSrc);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    
    //printf("Input:\n");
    //for(int i = 0; i < NUM_ROWS; ++i)
    //{
    //    printf("[%2i]: ", i);
    //    for(int j = 0; j < NUM_COLS; ++j)
    //    {
    //        printf("%i ", tSrc(i, j));
    //    }
    //    printf("\n");
    //}
    const int NUM_WORDS    = (NUM_ROWS + 31) / 32;
    for(int i = 0; i < NUM_WORDS; ++i)
    {
      //printf("[%2i]: ", i);
        for(int j = 0; j < NUM_COLS; ++j)
        {
            for(int k = 0; k < 32; ++k)
            {
                if(((i * 32) + k) < NUM_ROWS)
                {
                    uint32_t convert_bit = (tDst(i, j) >> k) & 0x1;
                    uint32_t expected_bit = (0 == tSrc(i * 32 + k, j)) ? 0 : 1;
                    EXPECT_EQ(convert_bit, expected_bit)
                      << "ROW = "   << (i * 32 + k)
                      << ", COL = " << j
                      <<", BIT = "  << k
                      <<", WORD = " << std::hex << tDst(i, j) << std::dec
                      <<", SRC = "  << tSrc(i*32+k, j)
                      << std::endl;
                }
                //printf("0x%X ", tDst(i, j));
            }
        }
        //printf("\n");
    }
}

////////////////////////////////////////////////////////////////////////
// do_convert_test_copy_bits()
void do_convert_test_copy_bits(int SRC_NUM_ROWS,
                               int DST_NUM_ROWS,
                               int NUM_COLS)
{
    typedef cuphy::typed_tensor<CUPHY_BIT, cuphy::pinned_alloc> tensor_bit_p;

    //const std::array<int, 2> SRC_DIMS = {{NUM_ROWS, NUM_COLS}};
    //------------------------------------------------------------------
    // Allocate tensors
    tensor_bit_p tSrc(SRC_NUM_ROWS, NUM_COLS);
    tensor_bit_p tDst(DST_NUM_ROWS, NUM_COLS);
    //------------------------------------------------------------------
    // Initialize the source tensor with random values
    cuphy::rng rng;
    rng.uniform(tSrc, 0, 1);
    //------------------------------------------------------------------
    // Copy from source to destination
    tensor_copy_range(tDst,
                      tSrc,
                      cuphy::index_group(cuphy::index_range(0, DST_NUM_ROWS),
                                         cuphy::dim_all()));
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    
    //printf("Input:\n");
    //for(int i = 0; i < NUM_ROWS; ++i)
    //{
    //    printf("[%2i]: ", i);
    //    for(int j = 0; j < NUM_COLS; ++j)
    //    {
    //        printf("%i ", tSrc(i, j));
    //    }
    //    printf("\n");
    //}
    const int DST_NUM_WORDS    = (DST_NUM_ROWS + 31) / 32;
    for(int i = 0; i < DST_NUM_WORDS; ++i)
    {
        for(int j = 0; j < NUM_COLS; ++j)
        {
            const uint32_t SRC_WORD = tSrc(i, j);
            const uint32_t DST_WORD = tDst(i, j);
            //printf("[%i, %i]: SRC = 0x%X, DST = 0x%X\n", i, j, SRC_WORD, DST_WORD);
            for(int k = 0; k < 32; ++k)
            {
                const uint32_t DST_BIT = (DST_WORD >> k) & 0x1U;
                if(((i * 32) + k) < DST_NUM_ROWS)
                {
                    const uint32_t SRC_BIT = (SRC_WORD >> k) & 0x1U;
                    EXPECT_EQ(DST_BIT, SRC_BIT)
                      << "ROW = "   << (i * 32 + k)
                      << ", COL = " << j
                      <<", BIT = "  << k
                      <<", WORD = " << std::hex << DST_WORD << std::dec
                      <<", SRC = "  << std::hex << SRC_WORD << std::dec
                      << std::endl;
                }
                else
                {
                    EXPECT_EQ(DST_BIT, 0U)
                      << "ROW = "   << (i * 32 + k)
                      << ", COL = " << j
                      <<", BIT = "  << k
                      <<", WORD = " << std::hex << DST_WORD << std::dec
                      <<", SRC = "  << std::hex << SRC_WORD << std::dec
                      << std::endl;
                }
            }
        }
        //printf("\n");
    }
}

} // namespace

////////////////////////////////////////////////////////////////////////
// Convert.ToBits
// Test conversion of scalar types to bits
TEST(Convert, ToBits)
{
    do_convert_test_to_bits<CUPHY_R_32U>(32, 8);
    do_convert_test_to_bits<CUPHY_R_8U>(100, 3);
}

////////////////////////////////////////////////////////////////////////
// Convert.BitCopySubset
// Test copy of subsets of bit tensor - in particular, the "zeroing" of
// bits in the last word of each column
TEST(Convert, BitCopySubset)
{
    do_convert_test_copy_bits(64,    33, 17);
    do_convert_test_copy_bits(64,    64, 1);
    do_convert_test_copy_bits(91,    51, 3);
    do_convert_test_copy_bits(1024, 511, 44);
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
