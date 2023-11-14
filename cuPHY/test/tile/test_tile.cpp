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

// comparison operator for complex floats. "Exact" comparison is OK here
// because we are just copying values.
//bool operator==(const cuFloatComplex& a, const cuFloatComplex& b)
//{
//    return (a.x == b.x) && (a.y == b.y);
//}

template <cuphyDataType_t TType>
void do_tile_test_2D(int             NUM_ROWS,
                     int             NUM_COLS,
                     int             NUM_TILE_ROWS,
                     int             NUM_TILE_COLS)
{
    typedef cuphy::typed_tensor<TType, cuphy::pinned_alloc> tensor_p;
    //------------------------------------------------------------------
    // Allocate a source tensor
    const std::array<int, 2> SRC_DIMS = {{NUM_ROWS, NUM_COLS}};
    tensor_p                 tSrc(cuphy::tensor_layout(SRC_DIMS.size(), SRC_DIMS.data(), nullptr));
    //------------------------------------------------------------------
    // Initialize the source tensor with random values
    cuphy::rng rng;
    rng.uniform(tSrc, 1, 10);
    //------------------------------------------------------------------
    // Allocate a destination tensor with appropriate dimensions
    const std::array<int, 2> TILE = {{NUM_TILE_ROWS,  NUM_TILE_COLS}};
    std::array<int, 2>       dst_dims;
    for(size_t i = 0; i < SRC_DIMS.size(); ++i)
    {
        dst_dims[i] = SRC_DIMS[i] * TILE[i];
    }
    tensor_p                 tDst(cuphy::tensor_layout(dst_dims.size(), dst_dims.data(), nullptr));
    //------------------------------------------------------------------
    // Perform the tile operation
    tSrc.tile(tDst, TILE[0], TILE[1]);
    //------------------------------------------------------------------
    // Wait for results to complete
    cudaStreamSynchronize(0);
    //printf("Input:\n");
    //for(int i = 0; i < SRC_DIMS[0]; ++i)
    //{
    //    printf("[%2i]: ", i);
    //    for(int j = 0; j < SRC_DIMS[1]; ++j)
    //    {
    //        printf("%f ", tSrc(i, j));
    //    }
    //    printf("\n");
    //}
    //printf("Output:\n");
    //for(int i = 0; i < dst_dims[0]; ++i)
    //{
    //    printf("[%2i]: ", i);
    //    for(int j = 0; j < dst_dims[1]; ++j)
    //    {
    //        printf("%f ", tDst(i, j));
    //    }
    //    printf("\n");
    //}
    for(int i = 0; i < dst_dims[0]; ++i)
    {
        for(int j = 0; j < dst_dims[1]; ++j)
        {
            EXPECT_EQ(tDst(i, j), tSrc(i % SRC_DIMS[0], j % SRC_DIMS[1]));
        }
    }
}

} // namespace


////////////////////////////////////////////////////////////////////////
// Tile.Basic2D
TEST(Tile, Basic2D)
{
    do_tile_test_2D<CUPHY_R_32F>(32, 8, 2, 3);
    do_tile_test_2D<CUPHY_R_32F>(32, 8, 2, 3);
    do_tile_test_2D<CUPHY_R_16F>(32, 8, 2, 3);
    do_tile_test_2D<CUPHY_R_8U> (32, 8, 2, 3);
    //do_tile_test_2D<CUPHY_C_64F>(32, 8, 2, 3);
}


////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
