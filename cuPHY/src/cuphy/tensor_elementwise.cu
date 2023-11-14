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
#include "tensor_elementwise.hpp"
#include "variant.hpp"
#include "cuphy_kernel_util.cuh"
#include "math_utils.cuh"

namespace
{

const unsigned int NUM_ELEMWISE_X = 32;
const unsigned int NUM_ELEMWISE_Y = 32;

////////////////////////////////////////////////////////////////////////
// get_default_scale()
// Returns a variant set with the default "scale" values, for cases when
// the caller does not provide a value. In general, the default scale
// would be "1" (in the appropriate type).
cuphyVariant_t get_default_scale(cuphyDataType_t t)
{
    cuphyVariant_t v;
    v.type = t;
    switch(t)
    {
    default:          // fall through to the
    case CUPHY_VOID:  // CUPHY_BIT case, just so that some value is set
    case CUPHY_BIT:   v.value.b1   = 1;                                               break;
    
    case CUPHY_R_8I:  v.value.r8i  = 1;                                               break;
    case CUPHY_C_8I:  v.value.c8i  = make_complex<char2>::create(1, 0);               break;
    case CUPHY_R_8U:  v.value.r8u  = 1;                                               break;
    case CUPHY_C_8U:  v.value.c8u  = make_complex<uchar2>::create(1, 0);              break;
    case CUPHY_R_16I: v.value.r16i = 1;                                               break;
    case CUPHY_C_16I: v.value.c16i = make_complex<short2>::create(1, 0);              break;
    case CUPHY_R_16U: v.value.r16u = 1;                                               break;
    case CUPHY_C_16U: v.value.c16u = make_complex<ushort2>::create(1, 0);             break;
    case CUPHY_R_32I: v.value.r32i = 1;                                               break;
    case CUPHY_C_32I: v.value.c32i = make_complex<int2>::create(1, 0);                break;
    case CUPHY_R_32U: v.value.r32u = 1;                                               break;
    case CUPHY_C_32U: v.value.c32u = make_complex<uint2>::create(1, 0);               break;
    case CUPHY_R_16F: v.value.r16f = __float2half(1.0f);                              break;
    case CUPHY_C_16F: v.value.c16f = __floats2half2_rn(1.0f, 0.0f);                   break;
    case CUPHY_R_32F: v.value.r32f = 1.0f;                                            break;
    case CUPHY_C_32F: v.value.c32f = make_complex<cuComplex>::create(1.0f, 0.0f);     break;
    case CUPHY_R_64F: v.value.r64f = 1.0;                                             break;
    case CUPHY_C_64F: v.value.c64f = make_complex<cuDoubleComplex>::create(1.0, 0.0); break;
    }
    return v;
}


////////////////////////////////////////////////////////////////////////
// binary_op_add()
template <typename T> struct binary_op_add
{
    __device__ static T apply(T alpha, T a, T beta, T b)
    {
        return (alpha * a) + (beta * b);
    }
};

////////////////////////////////////////////////////////////////////////
// binary_op_complex_add()
template <typename T> struct binary_op_complex_add
{
    __device__ static T apply(T alpha, T a, T beta, T b)
    {
        return complex_add(complex_mul(alpha, a), complex_mul(beta, b));
    }
};

////////////////////////////////////////////////////////////////////////
// binary_op_mul()
template <typename T> struct binary_op_mul
{
    __device__ static T apply(T alpha, T a, T beta, T b)
    {
        return (alpha * a) * (beta * b);
    }
};

////////////////////////////////////////////////////////////////////////
// binary_op_complex_mul()
template <typename T> struct binary_op_complex_mul
{
    __device__ static T apply(T alpha, T a, T beta, T b)
    {
        return complex_mul(complex_mul(alpha, a), complex_mul(beta, b));
    }
};

////////////////////////////////////////////////////////////////////////
// binary_op_xor()
template <typename T> struct binary_op_xor
{
    __device__ static T apply(T a, T b)
    {
        return (a ^ b);
    }
};

////////////////////////////////////////////////////////////////////////
// tensor_elementwise_unary_kernel()
// Kernel to perform an elementwise operation from 1 input
template <typename TDst, typename TSrc, template <typename> class TOpUnary>
__global__ __launch_bounds__(NUM_ELEMWISE_X * NUM_ELEMWISE_Y)
void tensor_elementwise_unary_kernel(tensor_layout_any       layoutDst,
                                     TDst*                   dst,
                                     tensor_layout_any       layoutSrcA,
                                     const TSrc*             srcA,
                                     TSrc                    alpha,
                                     tensor_layout_any       layoutSrcB,
                                     const TSrc*             srcB,
                                     TSrc                    beta)
{
    static_assert(CUPHY_DIM_MAX == 5, "tensor_tile_kernel only defined for 5D tensors");
    //------------------------------------------------------------------
    // Loop over tensor layout (up to 5-D)
    for(int i4 = 0; i4 < layoutDst.dimensions[4]; ++i4)
    {    
        for(int i3 = 0; i3 < layoutDst.dimensions[3]; ++i3)
        {
            for(grid_stride_index<2> it2; it2 < layoutDst.dimensions[2]; it2.next())
            {
                for(grid_stride_index<1> it1; it1 < layoutDst.dimensions[1]; it1.next())
                {
                    for(grid_stride_index<0> it0; it0 < layoutDst.dimensions[0]; it0.next())
                    {
                        //-    -    -    -    -    -    -    -    -    -
                        // Load the source value
                        int    nSrc[5]  = {static_cast<int>(it0.value),
                                        static_cast<int>(it1.value),
                                        static_cast<int>(it2.value),
                                        i3,
                                        i4};
                        size_t in_idx_A = layoutSrcA.offset(nSrc);
                        size_t out_idx  = layoutDst.offset(nSrc);
                        TSrc   a        = srcA[in_idx_A];
                        dst[out_idx]    = TOpUnary<TSrc>::apply(alpha, a); // op(a, b);
                    } // it0
                }     // it1
            }         // it2
        }             // it3
    }                 // it4
}


////////////////////////////////////////////////////////////////////////
// tensor_elementwise_binary_kernel()
// Kernel to perform an elementwise operation from 2 inputs
template <typename TDst, typename TSrc, template <typename> class TOpBinary>
__global__ __launch_bounds__(NUM_ELEMWISE_X * NUM_ELEMWISE_Y)
void tensor_elementwise_binary_kernel(tensor_layout_any       layoutDst,
                                      TDst*                   dst,
                                      tensor_layout_any       layoutSrcA,
                                      const TSrc*             srcA,
                                      TSrc                    alpha,
                                      tensor_layout_any       layoutSrcB,
                                      const TSrc*             srcB,
                                      TSrc                    beta)
{
    static_assert(CUPHY_DIM_MAX == 5, "tensor_tile_kernel only defined for 5D tensors");
    //------------------------------------------------------------------
    // Loop over tensor layout (up to 4-D)
    for(int i4 = 0; i4 < layoutDst.dimensions[4]; ++i4)
    {    
        for(int i3 = 0; i3 < layoutDst.dimensions[3]; ++i3)
        {
            for(grid_stride_index<2> it2; it2 < layoutDst.dimensions[2]; it2.next())
            {
                for(grid_stride_index<1> it1; it1 < layoutDst.dimensions[1]; it1.next())
                {
                    for(grid_stride_index<0> it0; it0 < layoutDst.dimensions[0]; it0.next())
                    {
                        //-    -    -    -    -    -    -    -    -    -
                        // Load the source value
                        int    nSrc[5]  = {static_cast<int>(it0.value),
                                        static_cast<int>(it1.value),
                                        static_cast<int>(it2.value),
                                        i3,
                                        i4};
                        size_t in_idx_A = layoutSrcA.offset(nSrc);
                        size_t in_idx_B = layoutSrcB.offset(nSrc);
                        size_t out_idx  = layoutDst.offset(nSrc);
                        TSrc a          = srcA[in_idx_A];
                        TSrc b          = srcB[in_idx_B];
                        //if((0 == threadIdx.x) && (0 == threadIdx.y))
                        //{
                        //    printf("a = %f, b = %f, alpha = %f, beta = %f, result = %f\n",
                        //           a,
                        //           b,
                        //           alpha,
                        //           beta,
                        //           TOpBinary<TSrc>::apply(alpha, a, beta, b));
                        //}
                        dst[out_idx]    = TOpBinary<TSrc>::apply(alpha, a, beta, b);
                    } // it0
                }     // it1
            }         // it2
        }             // it3
    }                 // it4
}

////////////////////////////////////////////////////////////////////////
// launch_elementwise_binary()
template <typename TDst, typename TSrc, template <typename> class TOperator>
cuphyStatus_t launch_elementwise_binary(const tensor_layout_any&       tDstLayout,
                                        void*                          dst,
                                        const tensor_layout_any&       tSrcALayout,
                                        const void*                    srcA,
                                        const cuphyVariant_t&          alpha,
                                        const tensor_layout_any&       tSrcBLayout,
                                        const void*                    srcB,
                                        const cuphyVariant_t&          beta,
                                        cudaStream_t                   strm)
{
    dim3 grdDim(1);
    dim3 blkDim(NUM_ELEMWISE_X, NUM_ELEMWISE_Y);
    
    tensor_elementwise_binary_kernel<TDst, TSrc, TOperator><<<grdDim, blkDim, 0, strm>>>(tDstLayout,                         // dst layout
                                                                                         static_cast<TDst*>(dst),            // output
                                                                                         tSrcALayout,                        // src A layout
                                                                                         static_cast<const TSrc*>(srcA),     // src A input
                                                                                         cuphy_i::variant_as_t<TSrc>(alpha), // alpha
                                                                                         tSrcBLayout,                        // src B layout
                                                                                         static_cast<const TSrc*>(srcB),     // src B input
                                                                                         cuphy_i::variant_as_t<TSrc>(beta)); // // beta
                                                                              
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

////////////////////////////////////////////////////////////////////////
// tensor_elementwise_binary_bit_kernel()
// Kernel to perform an elementwise operation from 2 inputs
template <template <typename> class TOpBinary>
__global__ __launch_bounds__(NUM_ELEMWISE_X * NUM_ELEMWISE_Y)
void tensor_elementwise_binary_bit_kernel(tensor_layout_any       layoutDst,
                                          uint32_t*               dst,
                                          tensor_layout_any       layoutSrcA,
                                          const uint32_t*         srcA,
                                          tensor_layout_any       layoutSrcB,
                                          const uint32_t*         srcB,
                                          uint32_t                columnEndMask)
{
    static_assert(CUPHY_DIM_MAX == 5, "tensor_tile_kernel only defined for 5D tensors");
    //------------------------------------------------------------------
    // Loop over tensor layout (up to 5-D)
    for(int i4 = 0; i4 < layoutDst.dimensions[4]; ++i4)
    {    
        for(int i3 = 0; i3 < layoutDst.dimensions[3]; ++i3)
        {
            for(grid_stride_index<2> it2; it2 < layoutDst.dimensions[2]; it2.next())
            {
                for(grid_stride_index<1> it1; it1 < layoutDst.dimensions[1]; it1.next())
                {
                    for(grid_stride_index<0> it0; it0 < layoutDst.dimensions[0]; it0.next())
                    {
                        //-    -    -    -    -    -    -    -    -    -
                        // Load the source value
                        int    nSrc[5]  = {static_cast<int>(it0.value),
                                        static_cast<int>(it1.value),
                                        static_cast<int>(it2.value),
                                        i3,
                                        i4};
                        size_t in_idx_A = layoutSrcA.offset(nSrc);
                        size_t in_idx_B = layoutSrcB.offset(nSrc);
                        size_t out_idx  = layoutDst.offset(nSrc);
                        uint32_t a      = srcA[in_idx_A];
                        uint32_t b      = srcB[in_idx_B];
                        if(it0.value == (layoutSrcA.dimensions[0] - 1))
                        {
                            a &= columnEndMask;
                            b &= columnEndMask;
                        }
                        //if((0 == threadIdx.x) && (0 == threadIdx.y))
                        //{
                        //    printf("a = %f, b = %f, alpha = %f, beta = %f, result = %f\n",
                        //           a,
                        //           b,
                        //           alpha,
                        //           beta,
                        //           TOpBinary<TSrc>::apply(alpha, a, beta, b));
                        //}
                        dst[out_idx]    = TOpBinary<uint32_t>::apply(a, b);
                    } // it0
                }     // it1
            }         // it2
        }             // it3
    }                 // it4
}

////////////////////////////////////////////////////////////////////////
// launch_elementwise_binary_bit()
template <template <typename> class TOperator>
cuphyStatus_t launch_elementwise_binary_bit(const tensor_layout_any&       tDstLayout,
                                            void*                          dst,
                                            const tensor_layout_any&       tSrcALayout,
                                            const void*                    srcA,
                                            const tensor_layout_any&       tSrcBLayout,
                                            const void*                    srcB,
                                            uint32_t                       columnEndMask,
                                            cudaStream_t                   strm)
{
    dim3 grdDim(1);
    dim3 blkDim(NUM_ELEMWISE_X, NUM_ELEMWISE_Y);
    
    tensor_elementwise_binary_bit_kernel<TOperator><<<grdDim, blkDim, 0, strm>>>(tDstLayout,                         // dst layout
                                                                                 static_cast<uint32_t*>(dst),        // output
                                                                                 tSrcALayout,                        // src A layout
                                                                                 static_cast<const uint32_t*>(srcA), // src A input
                                                                                 tSrcBLayout,                        // src B layout
                                                                                 static_cast<const uint32_t*>(srcB), // src B input
                                                                                 columnEndMask);                                                                              
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

////////////////////////////////////////////////////////////////////////
// update_descriptors_for_broadcast()
void update_descriptors_for_broadcast(tensor_desc& tSrcA,
                                      tensor_desc& tSrcB)
{
    vec<int, CUPHY_DIM_MAX> Adim     = tSrcA.layout().dimensions;
    vec<int, CUPHY_DIM_MAX> Bdim     = tSrcB.layout().dimensions;
    vec<int, CUPHY_DIM_MAX> Astrides = tSrcA.layout().strides;
    vec<int, CUPHY_DIM_MAX> Bstrides = tSrcB.layout().strides;
    for(int i = 0; i < CUPHY_DIM_MAX; ++i)
    {
        // Make sure that the stride is zero for a dimension with
        // only 1 element. This will allow indexing in the broadcast
        // dimension to resolve to a valid offset/address.
        if(1 == tSrcA.layout().dimensions[i]) Astrides[i] = 0;
        if(1 == tSrcB.layout().dimensions[i]) Bstrides[i] = 0;
   }
   tSrcA.set(tSrcA.type(), CUPHY_DIM_MAX, Adim.begin(), Astrides.begin());
   tSrcB.set(tSrcB.type(), CUPHY_DIM_MAX, Bdim.begin(), Bstrides.begin());
}

////////////////////////////////////////////////////////////////////////
// dimensions_match_with_broadcast()
bool dimensions_match_with_broadcast(const tensor_desc& tDst,
                                     const tensor_desc& tSrcA,
                                     const tensor_desc& tSrcB)
{
    const vec<int, CUPHY_DIM_MAX>& dimDst = tDst.layout().dimensions;
    const vec<int, CUPHY_DIM_MAX>& dimA   = tSrcA.layout().dimensions;
    const vec<int, CUPHY_DIM_MAX>& dimB   = tSrcB.layout().dimensions;
    for(int i = 0; i < CUPHY_DIM_MAX; ++i)
    {
        if(dimA[i] != dimB[i])
        {
            if((1 == dimA[i]) || (1 == dimB[i]))
            {
                if(dimDst[i] != std::max(dimA[i], dimB[i]))
                {
                    // A/B dimensions do not match and the output
                    // dimension it not max(dimA, dimB)
                    return false;
                }
            }
            else
            {
                // A/B dimensions do not match, and neither dimension is 1
                return false;
            }
        }
        else
        {
            // A and B dim are the same, destination should also match
            if(dimDst[i] != dimA[i])
            {
                return false;
            }
        }
    }
    return true;
}

////////////////////////////////////////////////////////////////////////
// validate_tensor_descriptors()
cuphyStatus_t validate_tensor_descriptors(const tensor_desc&   tDst,
                                          tensor_desc&         tSrcA,
                                          tensor_desc&         tSrcB,
                                          bool                 BIsValid,
                                          cuphyElementWiseOp_t elemOp)
{
    cuphyStatus_t s = CUPHY_STATUS_INTERNAL_ERROR;
    //------------------------------------------------------------------
    // Check for void types on srcA and destination
    if((tDst.type() == CUPHY_VOID) || (tSrcA.type() == CUPHY_VOID))
    {
        return CUPHY_STATUS_UNSUPPORTED_TYPE;
    }
    //------------------------------------------------------------------
    // Perform operation-specific checking
    switch(elemOp)
    {
    case CUPHY_ELEMWISE_ADD: // fall through...
    case CUPHY_ELEMWISE_MUL:
        {
            // Output and input types must match
            if(tDst.type() != tSrcA.type())
                return CUPHY_STATUS_UNSUPPORTED_TYPE;
            if(BIsValid)
            {
                if(tSrcB.type() != tSrcA.type())
                {
                    // Types of B and A do not match
                    return CUPHY_STATUS_UNSUPPORTED_TYPE;
                }
                if(!dimensions_match_with_broadcast(tDst, tSrcA, tSrcB))
                {
                    return CUPHY_STATUS_SIZE_MISMATCH;
                }
                update_descriptors_for_broadcast(tSrcA, tSrcB);
            }
            else
            {
                // Only 1 argument provided, no broadcasting possible
                if(tDst.layout().dimensions != tSrcA.layout().dimensions)
                    return CUPHY_STATUS_SIZE_MISMATCH;
            }
            return CUPHY_STATUS_SUCCESS;
        }
    case CUPHY_ELEMWISE_MIN: // fall through...
    case CUPHY_ELEMWISE_MAX:
        {
            if(!BIsValid)
                return CUPHY_STATUS_INVALID_ARGUMENT;
            if((tDst.type() != tSrcA.type()) || (tDst.type() != tSrcB.type()))
                return CUPHY_STATUS_UNSUPPORTED_TYPE;
            if((tDst.layout().dimensions != tSrcA.layout().dimensions) ||
               (tDst.layout().dimensions != tSrcB.layout().dimensions))
                return CUPHY_STATUS_SIZE_MISMATCH;
            return CUPHY_STATUS_SUCCESS;
        }
    case CUPHY_ELEMWISE_ABS:
        {
            if(BIsValid)
                return CUPHY_STATUS_INVALID_ARGUMENT;
            if(type_is_complex(tSrcA.type()) &&
               tDst.type() != scalar_type_from_complex_type(tSrcA.type()))
            {
                return CUPHY_STATUS_UNSUPPORTED_TYPE;
            }
            if(tDst.layout().dimensions != tSrcA.layout().dimensions)
            return CUPHY_STATUS_SUCCESS;
        }
    case CUPHY_ELEMWISE_BIT_XOR:
        {
            // Bitwise XOR requires both A and B inputs
            if(!BIsValid)
                return CUPHY_STATUS_INVALID_ARGUMENT;
            // Both inputs and the output must have type CUPHY_BIT
            if((tDst.type()  != CUPHY_BIT) ||
               (tSrcA.type() != CUPHY_BIT) ||
               (tSrcB.type() != CUPHY_BIT))
            {
                return CUPHY_STATUS_UNSUPPORTED_TYPE;
            }
            if(!dimensions_match_with_broadcast(tDst, tSrcA, tSrcB))
            {
                return CUPHY_STATUS_SIZE_MISMATCH;
            }
            update_descriptors_for_broadcast(tSrcA, tSrcB);
            return CUPHY_STATUS_SUCCESS;
        }
    default:
        break;
    }
    return s;
}

////////////////////////////////////////////////////////////////////////
// convert_scale_or_default()
cuphyStatus_t convert_scale_or_default(cuphyVariant_t&       dst,
                                       const cuphyVariant_t* pv,
                                       cuphyDataType_t       t)
{
    if(pv)
    {
        dst = *pv;
        return cuphy_i::convert_variant(dst, t);
    }
    else
    {
        dst = get_default_scale(t);
        return CUPHY_STATUS_SUCCESS;
    }
}

////////////////////////////////////////////////////////////////////////
// tensor_elementwise_binary_add()
cuphyStatus_t tensor_elementwise_binary_add(const tensor_desc&            tDst,
                                            void*                         dst,
                                            const tensor_desc&            tSrcA,
                                            const void*                   srcA,
                                            const cuphyVariant_t*         palpha,
                                            const tensor_desc&            tSrcB,
                                            const void*                   srcB,
                                            const cuphyVariant_t*         pbeta,
                                            cudaStream_t                  strm)
{
    //------------------------------------------------------------------
    // Set up scale values as required by the type-specific kernel
    cuphyVariant_t alpha, beta;
    cuphyStatus_t  s = convert_scale_or_default(alpha, palpha, tSrcA.type());
    if(CUPHY_STATUS_SUCCESS != s) return s;
    s                = convert_scale_or_default(beta,  pbeta , tSrcA.type());
    if(CUPHY_STATUS_SUCCESS != s) return s;
    //------------------------------------------------------------------
    s = CUPHY_STATUS_INTERNAL_ERROR;
    switch(tDst.type())
    {
    case CUPHY_R_8I:  s = launch_elementwise_binary<int8_t,   int8_t,   binary_op_add>        (tDst.layout(), dst, tSrcA.layout(), srcA, alpha, tSrcB.layout(), srcB, beta, strm);   break;
    case CUPHY_C_8I:  s = launch_elementwise_binary<char2,    char2,    binary_op_complex_add>(tDst.layout(), dst, tSrcA.layout(), srcA, alpha, tSrcB.layout(), srcB, beta, strm);   break;
    case CUPHY_R_8U:  s = launch_elementwise_binary<uint8_t,  uint8_t,  binary_op_add>        (tDst.layout(), dst, tSrcA.layout(), srcA, alpha, tSrcB.layout(), srcB, beta, strm);   break;
    case CUPHY_C_8U:  s = launch_elementwise_binary<uchar2,   uchar2,   binary_op_complex_add>(tDst.layout(), dst, tSrcA.layout(), srcA, alpha, tSrcB.layout(), srcB, beta, strm);   break;
    case CUPHY_R_16I: s = launch_elementwise_binary<int16_t,  int16_t,  binary_op_add>        (tDst.layout(), dst, tSrcA.layout(), srcA, alpha, tSrcB.layout(), srcB, beta, strm);   break;
    case CUPHY_C_16I: s = launch_elementwise_binary<short2,   short2,   binary_op_complex_add>(tDst.layout(), dst, tSrcA.layout(), srcA, alpha, tSrcB.layout(), srcB, beta, strm);   break;
    case CUPHY_R_16U: s = launch_elementwise_binary<uint16_t, uint16_t, binary_op_add>        (tDst.layout(), dst, tSrcA.layout(), srcA, alpha, tSrcB.layout(), srcB, beta, strm);   break;
    case CUPHY_C_16U: s = launch_elementwise_binary<ushort2,  ushort2,  binary_op_complex_add>(tDst.layout(), dst, tSrcA.layout(), srcA, alpha, tSrcB.layout(), srcB, beta, strm);   break;
    case CUPHY_R_32I: s = launch_elementwise_binary<int32_t,  int32_t,  binary_op_add>        (tDst.layout(), dst, tSrcA.layout(), srcA, alpha, tSrcB.layout(), srcB, beta, strm);   break;
    case CUPHY_C_32I: s = launch_elementwise_binary<int2,     int2,     binary_op_complex_add>(tDst.layout(), dst, tSrcA.layout(), srcA, alpha, tSrcB.layout(), srcB, beta, strm);   break;
    case CUPHY_R_32U: s = launch_elementwise_binary<uint32_t, uint32_t, binary_op_add>        (tDst.layout(), dst, tSrcA.layout(), srcA, alpha, tSrcB.layout(), srcB, beta, strm);   break;
    case CUPHY_C_32U: s = launch_elementwise_binary<uint2,    uint2,    binary_op_complex_add>(tDst.layout(), dst, tSrcA.layout(), srcA, alpha, tSrcB.layout(), srcB, beta, strm);   break;
    case CUPHY_R_16F: s = launch_elementwise_binary<__half,   __half,   binary_op_add>        (tDst.layout(), dst, tSrcA.layout(), srcA, alpha, tSrcB.layout(), srcB, beta, strm);   break;
    case CUPHY_C_16F: s = launch_elementwise_binary<__half2,  __half2,  binary_op_complex_add>(tDst.layout(), dst, tSrcA.layout(), srcA, alpha, tSrcB.layout(), srcB, beta, strm);   break;
    case CUPHY_R_32F: s = launch_elementwise_binary<float,    float,    binary_op_add>        (tDst.layout(), dst, tSrcA.layout(), srcA, alpha, tSrcB.layout(), srcB, beta, strm);   break;
    case CUPHY_C_32F: s = launch_elementwise_binary<float2,   float2,   binary_op_complex_add>(tDst.layout(), dst, tSrcA.layout(), srcA, alpha, tSrcB.layout(), srcB, beta, strm);   break;
    case CUPHY_R_64F: s = launch_elementwise_binary<double,   double,   binary_op_add>        (tDst.layout(), dst, tSrcA.layout(), srcA, alpha, tSrcB.layout(), srcB, beta, strm);   break;
    case CUPHY_C_64F: s = launch_elementwise_binary<double2,  double2,  binary_op_complex_add>(tDst.layout(), dst, tSrcA.layout(), srcA, alpha, tSrcB.layout(), srcB, beta, strm);   break;
    default:                                                                                                                                                                         break;
    }
    return s;
}

////////////////////////////////////////////////////////////////////////
// tensor_elementwise_binary_xor()
cuphyStatus_t tensor_elementwise_binary_xor(const tensor_desc&            tDst,
                                            void*                         dst,
                                            const tensor_desc&            tSrcA,
                                            const void*                   srcA,
                                            const tensor_desc&            tSrcB,
                                            const void*                   srcB,
                                            cudaStream_t                  strm)
{
    //------------------------------------------------------------------
    // We will use the "regular" kernel for bit-wise operations by
    // converting the CUPHY_BIT tensor to an equivalent uint32_t tensor.

    // 32-bit word layout from CUPHY_BIT type tensors
    tensor_layout_any layoutDst = word_layout_from_bit_layout(tDst.layout());
    tensor_layout_any layoutA   = word_layout_from_bit_layout(tSrcA.layout());
    tensor_layout_any layoutB   = word_layout_from_bit_layout(tSrcB.layout());
    // Mask for the end of the "column" of bits
    const uint32_t    COL_MASK  = bit_column_end_mask(tDst.layout().dimensions[0]);
    
    // Launch kernel
    cuphyStatus_t s = launch_elementwise_binary_bit<binary_op_xor>(layoutDst, dst, layoutA, srcA, layoutB, srcB, COL_MASK, strm);
    return s;
}

////////////////////////////////////////////////////////////////////////
// tensor_elementwise_binary()
cuphyStatus_t tensor_elementwise_binary(const tensor_desc&            tDst,
                                        void*                         dstAddr,
                                        const tensor_desc&            tSrcA,
                                        const void*                   srcAddrA,
                                        const cuphyVariant_t*         palpha,
                                        const tensor_desc&            tSrcB,
                                        const void*                   srcAddrB,
                                        const cuphyVariant_t*         pbeta,
                                        cuphyElementWiseOp_t          elemOp,
                                        cudaStream_t                  strm)
{
    cuphyStatus_t s = CUPHY_STATUS_INTERNAL_ERROR;;
    switch(elemOp)
    {
    case CUPHY_ELEMWISE_ADD:     s = tensor_elementwise_binary_add(tDst, dstAddr, tSrcA, srcAddrA, palpha, tSrcB, srcAddrB, pbeta, strm); break;
    case CUPHY_ELEMWISE_MUL:     break;
    case CUPHY_ELEMWISE_MIN:     break;
    case CUPHY_ELEMWISE_MAX:     break;
    case CUPHY_ELEMWISE_ABS:     break;
    case CUPHY_ELEMWISE_BIT_XOR: s = tensor_elementwise_binary_xor(tDst, dstAddr, tSrcA, srcAddrA,         tSrcB, srcAddrB,        strm); break;
    default:                     break;
    }
    return s;
}

////////////////////////////////////////////////////////////////////////
// tensor_elementwise_unary()
// Using "unnamed" arguments to avoid warnings until properly implemented
cuphyStatus_t tensor_elementwise_unary(const tensor_desc&            /*tDst*/,
                                       void*                         /*dstAddr*/,
                                       const tensor_desc&            /*tSrcA*/,
                                       const void*                   /*srcAddrA*/,
                                       const cuphyVariant_t*         /*palpha*/,
                                       cuphyElementWiseOp_t          /*elemOp*/,
                                       cudaStream_t                  /*strm*/)
{
    return CUPHY_STATUS_INTERNAL_ERROR;
}

} // namespace

namespace cuphy_i
{


////////////////////////////////////////////////////////////////////////
// tensor_elementwise()
cuphyStatus_t tensor_elementwise(const tensor_desc&      tDst,
                                 void*                   dstAddr,
                                 const tensor_desc&      tSrcA,
                                 const void*             srcAddrA,
                                 const cuphyVariant_t*   palpha,
                                 cuphyTensorDescriptor_t tSrcB,
                                 const void*             srcAddrB,
                                 const cuphyVariant_t*   pbeta,
                                 cuphyElementWiseOp_t    elemOp,
                                 cudaStream_t            strm)
{
    //------------------------------------------------------------------
    // Note: We are adopting NumPy-style "broadcast" semantics for
    // elementwise operations. We do so here by modifying the tensor
    // layout before attempting an element-wise operation. An
    // alternative would be to modify tensor descriptor internal data
    // so that all tensor descriptors "just work". (For the  most part,
    // it seems that this would entail setting the stride to 0 for
    // dimensions that have a size of 1.) For now, we restrict the
    // broadcast modifications to the tensor descriptor to temporary
    // instances created during this function all.
    //------------------------------------------------------------------
    // Copy tensor descriptors, since we may modify them to support
    // NumPy-style broadcast semantics for elementwise operations.
    tensor_desc tSrcALocal = tSrcA;
    tensor_desc tSrcBLocal;
    if(tSrcB)
    {
        tSrcBLocal = static_cast<const tensor_desc&>(*tSrcB);
    }
    //------------------------------------------------------------------
    // Validate types and sizes
    cuphyStatus_t s = validate_tensor_descriptors(tDst,
                                                  tSrcALocal,
                                                  tSrcBLocal,
                                                  tSrcB != nullptr,
                                                  elemOp);
    if(CUPHY_STATUS_SUCCESS != s)
    {
        return s;
    }
    //------------------------------------------------------------------
    if(tSrcB)
    {
        s = tensor_elementwise_binary(tDst,
                                      dstAddr,
                                      tSrcALocal,
                                      srcAddrA,
                                      palpha,
                                      tSrcBLocal,
                                      srcAddrB,
                                      pbeta,
                                      elemOp,
                                      strm);
    }
    else
    {
        s = tensor_elementwise_unary(tDst,
                                     dstAddr,
                                     tSrcALocal,
                                     srcAddrA,
                                     palpha,
                                     elemOp,
                                     strm);
    }
    return s;
}

} // namespace cuphy_i
