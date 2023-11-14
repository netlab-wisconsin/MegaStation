/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


//#define CUPHY_DEBUG 1
#include "tensor_fill.hpp"
#include "cuphy_kernel_util.cuh"

namespace
{

const unsigned int NUM_TILE_THREADS = 1024;

template <unsigned int TSize> struct tile_elem_type;
template <>                   struct tile_elem_type<1>  { typedef uint8_t  type; };
template <>                   struct tile_elem_type<2>  { typedef uint16_t type; };
template <>                   struct tile_elem_type<4>  { typedef uint32_t type; };
template <>                   struct tile_elem_type<8>  { typedef uint64_t type; };
template <>                   struct tile_elem_type<16> { typedef uint4    type; };

////////////////////////////////////////////////////////////////////////
// tensor_tile_kernel()
// Kernel populate a tensor with a single value
template <unsigned int TSize>
__global__ __launch_bounds__(NUM_TILE_THREADS)
void tensor_tile_kernel(tensor_layout_any       layoutDst,
                        void*                   dst_v,
                        tensor_layout_any       layoutSrc,
                        const void*             src_v,
                        vec<int, CUPHY_DIM_MAX> tileExtents)
{
    static_assert(CUPHY_DIM_MAX == 5, "tensor_tile_kernel only defined for 5D tensors");
    //------------------------------------------------------------------
    // Determine the element type at compile time based on the size.
    typedef typename tile_elem_type<TSize>::type elem_t;
    elem_t*       dst = static_cast<elem_t*>(dst_v);
    const elem_t* src = static_cast<const elem_t*>(src_v);
    //------------------------------------------------------------------
    // Loop over source tensor (up to 5-D)
    for(int i4 = 0; i4 < layoutSrc.dimensions[4]; ++i4)
    {
        for(int i3 = 0; i3 < layoutSrc.dimensions[3]; ++i3)
        {        
            for(grid_stride_index<2> it2; it2 < layoutSrc.dimensions[2]; it2.next())
            {
                for(grid_stride_index<1> it1; it1 < layoutSrc.dimensions[1]; it1.next())
                {
                    for(grid_stride_index<0> it0; it0 < layoutSrc.dimensions[0]; it0.next())
                    {
                        //-    -    -    -    -    -    -    -    -    -
                        // Load the source value
                        int    nSrc[5] = {static_cast<int>(it0.value),
                                        static_cast<int>(it1.value),
                                        static_cast<int>(it2.value),
                                        i3,
                                        i4};
                        size_t in_idx  = layoutSrc.offset(nSrc);
                        elem_t value   = src[in_idx];
                        //-    -    -    -    -    -    -    -    -    -
                        // "Scatter" the source value to each output tile
                        for(int tile4 = 0; tile4 < tileExtents[4]; ++tile4)
                        {
                            for(int tile3 = 0; tile3 < tileExtents[3]; ++tile3)
                            {
                                for(int tile2 = 0; tile2 < tileExtents[2]; ++tile2)
                                {
                                    for(int tile1 = 0; tile1 < tileExtents[1]; ++tile1)
                                    {
                                        for(int tile0 = 0; tile0 < tileExtents[0]; ++tile0)
                                        {
                                            int nout[5] = {static_cast<int>(it0.value) + (tile0 * layoutSrc.dimensions[0]),
                                                        static_cast<int>(it1.value) + (tile1 * layoutSrc.dimensions[1]),
                                                        static_cast<int>(it2.value) + (tile2 * layoutSrc.dimensions[2]),
                                                        i3                          + (tile3 * layoutSrc.dimensions[3]),
                                                        i4                          + (tile4 * layoutSrc.dimensions[4])};
                                            size_t out_idx = layoutDst.offset(nout);
                                            dst[out_idx] = value;
                                        }
                                    }
                                }
                            }
                        }
                    } // it0
                }     // it1
            }         // it2
        }             // it3
    }                 // it4
}

////////////////////////////////////////////////////////////////////////
// launch_tile()
template <unsigned int TSize>
cuphyStatus_t launch_tile(const tensor_layout_any&       tDstLayout,
                          void*                          dst,
                          const tensor_layout_any&       tSrcLayout,
                          const void*                    src,
                          const vec<int, CUPHY_DIM_MAX>& tileExt,
                          cudaStream_t                   strm)
{
    dim3 gridDim(1);
    dim3 blockDim(NUM_TILE_THREADS);

    tensor_tile_kernel<TSize><<<gridDim, blockDim, 0, strm>>>(tDstLayout, // tensor dims
                                                              dst,        // output
                                                              tSrcLayout, // tensor dims
                                                              src,        // input
                                                              tileExt);   // tile extents
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

} // namespace

namespace cuphy_i
{

////////////////////////////////////////////////////////////////////////
// tensor_tile()
cuphyStatus_t tensor_tile(const tensor_desc&    tDst,
                          void*                 dstAddr,
                          const tensor_desc&    tSrc,
                          const void*           srcAddr,
                          int                   tileRank,
                          const int*            tileExtents,
                          cudaStream_t          strm)
{

    //------------------------------------------------------------------
    const tensor_layout_any& tDstLayout = tDst.layout();
    const tensor_layout_any& tSrcLayout = tSrc.layout();
    //------------------------------------------------------------------
    // Not supported yet. Need to deal with converting tensor layout to
    // equivalent "word" layout, and probably disallow row tiling unless
    // the source length is a multiple of 32.
    if(CUPHY_BIT == tSrc.type())
    {
        return CUPHY_STATUS_UNSUPPORTED_TYPE;
    }
    //------------------------------------------------------------------
    vec<int, CUPHY_DIM_MAX> tileExt;
    for(int i = 0; i < CUPHY_DIM_MAX; ++i)
    {
        if(i < tileRank)
        {
            // Make sure that the output tensor dimension matches the
            // combination of the input tensor dim and the desired
            // amount of tiling.
            if(tDstLayout.dimensions[i] != (tSrcLayout.dimensions[i] * tileExtents[i]))
            {
                return CUPHY_STATUS_SIZE_MISMATCH;
            }
            // Use the caller provided extent
            tileExt[i] = tileExtents[i];
        }
        else
        {
            // Extend the tiling to unit length in higher dimensions
            tileExt[i] = 1;
        }
    }
    //------------------------------------------------------------------
    // Get the size of the cuPHY data type. (Since we are just copying
    // values, we only generate one kernel for each size in bytes.)
    int elementSize =  get_cuphy_type_storage_element_size(tSrc.type());
    cuphyStatus_t s = CUPHY_STATUS_UNSUPPORTED_TYPE;
    switch(elementSize)
    {
    case 1:  s = launch_tile<1> (tDstLayout, dstAddr, tSrcLayout, srcAddr, tileExt, strm); break;
    case 2:  s = launch_tile<2> (tDstLayout, dstAddr, tSrcLayout, srcAddr, tileExt, strm); break;
    case 4:  s = launch_tile<4> (tDstLayout, dstAddr, tSrcLayout, srcAddr, tileExt, strm); break;
    case 8:  s = launch_tile<8> (tDstLayout, dstAddr, tSrcLayout, srcAddr, tileExt, strm); break;
    case 16: s = launch_tile<16>(tDstLayout, dstAddr, tSrcLayout, srcAddr, tileExt, strm); break;
    default:                                                                               break;
    } // switch 

    return s;
}

} // namespace cuphy_i