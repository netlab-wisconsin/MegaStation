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
#include "tensor_reduction.hpp"
#include "cuphy_kernel_util.cuh"
#include "type_convert.hpp"

namespace
{

template <typename T>
struct warp_reduce_op_sum
{
    __device__
    static void apply(T& a, const T& b)
    {
        a += b;
    }
};

template <typename T, template <typename> class TOp>
struct warp_reduce
{
    __device__
    static void reduce(T& value)
    {
        for(int offset = 16; offset > 0; offset /= 2)
        {
            TOp<T>::apply(value,
                          __shfl_down_sync(0xFFFFFFFF, value, offset));
        }
    }
};

////////////////////////////////////////////////////////////////////////
// For now, we will use a single warp for simplicity.
// TODO: implement workspace API interface and leverage CUB reduction ops
const unsigned int NUM_REDUCTION_THREADS = 32;

////////////////////////////////////////////////////////////////////////
// ReductionSum
template <typename TOut, typename TIn>
struct ReductionSum
{
    __device__ ReductionSum() : accum_(0)
    {
    }
    __device__ void apply(const TIn& value)
    {
        accum_ += value;
    }
    __device__ TOut combine()
    {
        warp_reduce<TOut, warp_reduce_op_sum>::reduce(accum_);
        return accum_;
    }
    //-------------------------------------------------------------
    // Data
    TOut accum_;
};

////////////////////////////////////////////////////////////////////////
// ReductionSumBits
template <typename TOut, typename TIn>
struct ReductionSumBits
{
    __device__ ReductionSumBits() : accum_(0)
    {
    }
    __device__ void apply(const TIn& value)
    {
        //printf("threadIdx.x = %u, value = 0x%X, count = %u\n", threadIdx.x, value, __popc(value));
        accum_ += __popc(value);
    }
    __device__ TOut combine()
    {
        warp_reduce<TOut, warp_reduce_op_sum>::reduce(accum_);
        return accum_;
    }
    //-------------------------------------------------------------
    // Data
    TOut accum_;
};

////////////////////////////////////////////////////////////////////////
// SrcOpNone
// "no op" source transformation, passing the source value through
// unmodified.
template <typename T>
struct SrcOpNone
{
    __device__
    T apply(const T& src, int /*idx0*/, int /*dim0*/) const
    {
        return src;
    }
};

////////////////////////////////////////////////////////////////////////
// SrcOpColumnEndBitMask
// Source operator to apply a bitwise AND operation with a previously
// defined mask. Used for CUPHY_BIT tensors with "partial word"
// dimensions, to avoid passing through bits that are past the end of
// a tensor subset.
template <typename T>
struct SrcOpColumnEndBitMask
{
    uint32_t mask;
    __device__
    T apply(const T& src, int idx0, int dim0) const
    {
        if(idx0 == (dim0 - 1))
        {
            return (src & mask);
        }
        else
        {
            return src;
        }
    }
};

////////////////////////////////////////////////////////////////////////
// tensor_reduction_kernel()
// Kernel to perform reduction along a dimension.
//
// We are not yet trying for "speed" with this kernel - just trying for
// correctness until the use case for a more performant version presents
// itself. The initial usage is expected to be "offline" error checking.
//
// A single warp CTA is used to perform reduction along the reduction
// dimension. The (3D) launch grid is used to assign a CTA to each
// output point.
//
// Template parameters:
// TOut: type of output value
// TIn: type of input value
// TReductionOp: struct template to perform the reduction operation,
//               instantiated as TReductionOp<TOut, TIn>();
// TSrcOp: struct template to transform source value before
//         conversion. Usually SrcOpNone, but a mask may be used for
//         tensors of type CUPHY_BIT to handle partial words.
template <typename TOut,
          typename TIn,
          template <typename, typename> class TReductionOp,
          template <typename> class TSrcOp>
__global__ __launch_bounds__(NUM_REDUCTION_THREADS)
void tensor_reduction_kernel(tensor_layout_any layoutDst,
                             void*             dst_v,
                             tensor_layout_any layoutSrc,
                             const void*       src_v,
                             TSrcOp<TIn>       sourceOp)
{
    static_assert(CUPHY_DIM_MAX == 5, "tensor_reduction_kernel() assumes 5D tensors");
    TOut*      dst = static_cast<TOut*>(dst_v);
    const TIn* src = static_cast<const TIn*>(src_v);
    //------------------------------------------------------------------
    // Assuming here that the layouts have been modified such that the
    // reduction dimension is dimenson 0.
    int srcidx[5] = {static_cast<int>(threadIdx.x),
                     static_cast<int>(blockIdx.x),
                     static_cast<int>(blockIdx.y),
                     static_cast<int>(blockIdx.z) % 32,
                     static_cast<int>(blockIdx.z) / 32};
    //------------------------------------------------------------------
    // Construct an operator instance
    TReductionOp<TOut, TIn> op;
    //------------------------------------------------------------------
    // Loop over the reduction dimension (per-thread)
    while(srcidx[0] < layoutSrc.dimensions[0])
    {
        size_t srcOffset = layoutSrc.offset(srcidx);
        TIn    srcValue  = sourceOp.apply(src[srcOffset],
                                          srcidx[0],
                                          layoutSrc.dimensions[0]);
        TOut   value     = type_convert<TOut>(srcValue);
        //printf("threadIdx.x = %u, idx= %i, value = %f\n", threadIdx.x, srcidx[0], value);
        op.apply(value);
        srcidx[0] += blockDim.x;
    }
    //------------------------------------------------------------------
    // Reduce across the warp
    TOut result = op.combine();
    //printf("threadIdx.x = %u, sum = %f\n", threadIdx.x, result);
    //------------------------------------------------------------------
    // Write output
    if(0 == threadIdx.x)
    {
        //printf("OUTPUT: %f\n", result);
        int dstidx[5] = { 0,
                          static_cast<int>(blockIdx.x),
                          static_cast<int>(blockIdx.y),
                          static_cast<int>(blockIdx.z) % 32,
                          static_cast<int>(blockIdx.z) / 32};
        size_t dstOffset = layoutDst.offset(dstidx);
        dst[dstOffset] = result;
    }
}

////////////////////////////////////////////////////////////////////////
// collapse_dim()
// Return a vector of dimensions with the reduction dimension set to 1
vec<int, CUPHY_DIM_MAX> collapse_dim(const vec<int, CUPHY_DIM_MAX>& src, int dim)
{
    vec<int, CUPHY_DIM_MAX> dst = src;
    dst[dim] = 1;
    return dst;
}

////////////////////////////////////////////////////////////////////////
// validate_tensor_descriptors()
cuphyStatus_t validate_tensor_descriptors(const tensor_desc& tDst,
                                          const tensor_desc& tSrc,
                                          int                dim,
                                          cuphyReductionOp_t redOp)
{
    cuphyStatus_t s = CUPHY_STATUS_INTERNAL_ERROR;
    //------------------------------------------------------------------
    // Check for void types on src and destination
    if((tDst.type() == CUPHY_VOID) || (tSrc.type() == CUPHY_VOID))
    {
        return CUPHY_STATUS_UNSUPPORTED_TYPE;
    }
    //------------------------------------------------------------------
    // For now, we will only support reductions in the first dimension
    // for tensors of type CUPHY_BIT
    if((CUPHY_BIT == tSrc.type()) && (0 != dim))
    {
        return CUPHY_STATUS_UNSUPPORTED_TYPE;
    }
    //------------------------------------------------------------------
    // Make sure that the destination dimensions are what we would
    // expect.
    vec<int, CUPHY_DIM_MAX> rdims = collapse_dim(tSrc.layout().dimensions, dim);
    if(rdims != tDst.layout().dimensions)
    {
        DEBUG_PRINTF("expected %i %i %i %i %i, received %i %i %i %i %i\n",
                     rdims[0], rdims[1], rdims[2], rdims[3], rdims[4],
                     tDst.layout().dimensions[0],
                     tDst.layout().dimensions[2],
                     tDst.layout().dimensions[2],
                     tDst.layout().dimensions[3],
                     tDst.layout().dimensions[4]);
        return CUPHY_STATUS_SIZE_MISMATCH;
    }
    //------------------------------------------------------------------
    // Perform operation-specific checking
    switch(redOp)
    {
    case CUPHY_REDUCTION_SUM:
        {
            // Only two supported options right now
            if(((CUPHY_R_32F == tDst.type()) && (CUPHY_R_32F == tSrc.type())) ||
               ((CUPHY_R_32U == tDst.type()) && (CUPHY_BIT   == tSrc.type())))
            {
                return CUPHY_STATUS_SUCCESS;
            }
            else
            {
                return CUPHY_STATUS_NOT_SUPPORTED;
            }
        }
    case CUPHY_REDUCTION_MIN: // fall through...
    case CUPHY_REDUCTION_MAX:
        {
#if 0
            if((tDst.type() != tSrcA.type()) || (tDst.type() != descB.type()))
                return CUPHY_STATUS_UNSUPPORTED_TYPE;
            if((tDst.layout().dimensions != tSrcA.layout().dimensions) ||
               (tDst.layout().dimensions != descB.layout().dimensions))
                return CUPHY_STATUS_SIZE_MISMATCH;
#endif
            return CUPHY_STATUS_NOT_SUPPORTED;
        }
    default:
        break;
    }
    return s;
}

////////////////////////////////////////////////////////////////////////
// tensor_reduction_sum()
cuphyStatus_t tensor_reduction_sum(const tensor_layout_any& tDstLayout,
                                   void*                    dstAddr,
                                   cuphyDataType_t          dstType,
                                   const tensor_layout_any& tSrcLayout,
                                   const void*              srcAddr,
                                   cuphyDataType_t          srcType,
                                   size_t                   /*workspaceSize*/,
                                   void*                    /*workspace*/,
                                   const dim3&              grdDim,
                                   const dim3&              blkDim,
                                   cudaStream_t             strm)
{
    
    if((dstType == CUPHY_R_32F) && (srcType == CUPHY_R_32F))
    {
        tensor_reduction_kernel<float, float, ReductionSum, SrcOpNone><<<grdDim, blkDim, 0, strm>>>(tDstLayout,
                                                                                                    dstAddr,
                                                                                                    tSrcLayout,
                                                                                                    srcAddr,
                                                                                                    SrcOpNone<float>());
        return CUPHY_STATUS_SUCCESS;
    }
    else if((CUPHY_R_32U == dstType) && (CUPHY_BIT == srcType))
    {
        // We are only supporting reductions on bits in dimension 0.
        // Note that the calling function set up the launch grid
        // dimensions based on the original layout, and the function
        // below modifies the layout to work with 32 bit words.
        // However, the grid launch should still be the same after
        // converting to a word layout because we only support
        // reductions in dimension 0 for CUPHY_BIT tensors.
        tensor_layout_any tSrcWordLayout = word_layout_from_bit_layout(tSrcLayout);
        
        // Create an operator to apply a bit mask to the last
        // word in each "column", based on the length of the
        // original dimension.
        SrcOpColumnEndBitMask<uint32_t> srcOp;
        srcOp.mask = bit_column_end_mask(tSrcLayout.dimensions[0]);
        
        tensor_reduction_kernel<uint32_t, uint32_t, ReductionSumBits, SrcOpColumnEndBitMask><<<grdDim, blkDim, 0, strm>>>(tDstLayout,
                                                                                                                          dstAddr,
                                                                                                                          tSrcWordLayout,
                                                                                                                          srcAddr,
                                                                                                                          srcOp);
        return CUPHY_STATUS_SUCCESS;
    }
    return CUPHY_STATUS_INTERNAL_ERROR;
}

} // namespace

namespace cuphy_i
{

////////////////////////////////////////////////////////////////////////
// tensor_tile()
cuphyStatus_t tensor_reduction(const tensor_desc& tDst,
                               void*              dstAddr,
                               const tensor_desc& tSrc,
                               const void*        srcAddr,
                               cuphyReductionOp_t redOp,
                               int                reductionDim,
                               size_t             workspaceSize,
                               void*              workspace,
                               cudaStream_t       strm)
{
    cuphyStatus_t s = validate_tensor_descriptors(tDst, tSrc, reductionDim, redOp);
    if(CUPHY_STATUS_SUCCESS != s)
    {
        return s;
    }
    //------------------------------------------------------------------
    // To simplify the kernel, we will modify the tensor layout such
    // that the reduction dimension is always the first dimension in
    // index space.
    tensor_layout_any tDstLayout = tDst.layout();
    tensor_layout_any tSrcLayout = tSrc.layout();
    if(0 != reductionDim)
    {
        tDstLayout.swap_dimensions(0, reductionDim);
        tSrcLayout.swap_dimensions(0, reductionDim);
    }

    dim3 grdDim(tSrcLayout.dimensions[1],
                tSrcLayout.dimensions[2],
                tSrcLayout.dimensions[3]);
    dim3 blkDim(NUM_REDUCTION_THREADS);
    //printf("grdDim = %u, %u, %u\n", grdDim.x, grdDim.y, grdDim.z);
    //------------------------------------------------------------------
    // Dispatch the appropriate operation
    s = CUPHY_STATUS_INTERNAL_ERROR;
    switch(redOp)
    {
    case CUPHY_REDUCTION_SUM:
        s = tensor_reduction_sum(tDstLayout,
                                 dstAddr,
                                 tDst.type(),
                                 tSrcLayout,
                                 srcAddr,
                                 tSrc.type(),
                                 workspaceSize,
                                 workspace,
                                 grdDim,
                                 blkDim,
                                 strm);
        break;
    case CUPHY_REDUCTION_MIN:
        break;
    case CUPHY_REDUCTION_MAX:
        break;
    }
    if(CUPHY_STATUS_SUCCESS != s)
    {
        return s;
    }
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

} // namespace cuphy_i
