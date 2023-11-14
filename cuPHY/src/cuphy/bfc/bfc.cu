/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <algorithm>
#include "cuComplex.h"
#include <cooperative_groups.h>
#include "bfc.hpp"
#include "cuphy.hpp"
#include "type_convert.hpp"
#include "bfw_blockFP.cuh"
#include <vector>

using namespace cooperative_groups;

namespace bfw_coefComp
{
// #define ENABLE_DEBUG

static constexpr uint32_t N_THREADS_PER_WARP = 32; // cudaDeviceProp::warpSize;

template <typename TElem, int NDim>
struct tensor_ref_v0
{
    TElem* addr;
    int    dim[NDim];
    int    strides[NDim];
    size_t n_elem = 1;
    tensor_ref_v0(tensor_pair& tp) :
        addr(static_cast<TElem*>(tp.second)),
        n_elem(1)
    {
        const tensor_layout_any& layout = tp.first.get().layout();
#pragma unroll
        for(int i = 0; i < NDim; ++i)
        {
            dim[i]     = (layout.rank() > i) ? layout.dimensions[i] : 1;
            strides[i] = (layout.rank() > i) ? layout.strides[i] : 0;
            n_elem *= dim[i];
        }
    }
    tensor_ref_v0(const_tensor_pair& tp) :
        addr(static_cast<TElem*>(tp.second)),
        n_elem(1)
    {
        const tensor_layout_any& layout = tp.first.get().layout();
#pragma unroll
        for(int i = 0; i < NDim; ++i)
        {
            dim[i]     = (layout.rank() > i) ? layout.dimensions[i] : 1;
            strides[i] = (layout.rank() > i) ? layout.strides[i] : 0;
            n_elem *= dim[i];
        }
    }
    CUDA_BOTH long offset(int i0) const
    {
        return (strides[0] * (long)i0);
    }
    CUDA_BOTH long offset(int i0, int i1) const
    {
        return (strides[0] * (long)i0) + (strides[1] * (long)i1);
    }
    CUDA_BOTH long offset(int i0, int i1, int i2) const
    {
        return (strides[0] * (long)i0) + (strides[1] * (long)i1) + (strides[2] * (long)i2);
    };
    CUDA_BOTH long offset(int i0, int i1, int i2, int i3) const
    {
        return (strides[0] * (long)i0) + (strides[1] * (long)i1) + (strides[2] * (long)i2) + (strides[3] * (long)i3);
    };
    CUDA_BOTH TElem& operator()(int i0) { return *(addr + offset(i0)); }
    CUDA_BOTH const TElem& operator()(int i0) const { return *(addr + offset(i0)); }
    CUDA_BOTH TElem& operator()(int i0, int i1) { return *(addr + offset(i0, i1)); }
    CUDA_BOTH const TElem& operator()(int i0, int i1) const { return *(addr + offset(i0, i1)); }
    CUDA_BOTH TElem& operator()(int i0, int i1, int i2) { return *(addr + offset(i0, i1, i2)); }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2) const { return *(addr + offset(i0, i1, i2)); }
    CUDA_BOTH TElem& operator()(int i0, int i1, int i2, int i3) { return *(addr + offset(i0, i1, i2, i3)); }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3) const { return *(addr + offset(i0, i1, i2, i3)); }

    CUDA_BOTH size_t num_elem() { return n_elem; };
};

template <typename TElem>
struct tensor_ref
{
    TElem*         pAddr;
    const int32_t* strides;

    CUDA_BOTH
    tensor_ref(void* pAddr, const int32_t* pStrides) :
        pAddr(static_cast<TElem*>(pAddr)),
        strides(pStrides)
    {
    }
    CUDA_BOTH long offset(int i0) const
    {
        return (strides[0] * (long)i0);
    }
    CUDA_BOTH long offset(int i0, int i1) const
    {
        return (strides[0] * (long)i0) + (strides[1] * (long)i1);
    }
    CUDA_BOTH long offset(int i0, int i1, int i2) const
    {
        return (strides[0] * (long)i0) + (strides[1] * (long)i1) + (strides[2] * (long)i2);
    };
    CUDA_BOTH long offset(int i0, int i1, int i2, int i3) const
    {
        return (strides[0] * (long)i0) + (strides[1] * (long)i1) + (strides[2] * (long)i2) + (strides[3] * (long)i3);
    };
    CUDA_BOTH long offset(int i0, int i1, int i2, int i3, int i4) const
    {
        return (strides[0] * (long)i0) + (strides[1] * (long)i1) + (strides[2] * (long)i2) + (strides[3] * (long)i3) + (strides[4] * (long)i4);
    };    
    // clang-format off
    CUDA_BOTH TElem&       operator()(int i0)                                 { return *(pAddr + offset(i0));                 }
    CUDA_BOTH TElem&       operator()(int i0, int i1)                         { return *(pAddr + offset(i0, i1));             }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2)                 { return *(pAddr + offset(i0, i1, i2));         }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2, int i3)         { return *(pAddr + offset(i0, i1, i2, i3));     }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2, int i3, int i4) { return *(pAddr + offset(i0, i1, i2, i3, i4)); }

    CUDA_BOTH const TElem& operator()(int i0) const                                 { return *(pAddr + offset(i0));                 }
    CUDA_BOTH const TElem& operator()(int i0, int i1) const                         { return *(pAddr + offset(i0, i1));             }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2) const                 { return *(pAddr + offset(i0, i1, i2));         }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3) const         { return *(pAddr + offset(i0, i1, i2, i3));     }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3, int i4) const { return *(pAddr + offset(i0, i1, i2, i3, i4)); }

    // clang-format on
#if 0    
    if(std::is_const<TElem>::value) {
      CUDA_BOTH const TElem& operator()(int i0) const                         { return __ldg(addr + offset(i0));         }
      CUDA_BOTH const TElem& operator()(int i0, int i1) const                 { return __ldg(addr + offset(i0, i1));     }
      CUDA_BOTH const TElem& operator()(int i0, int i1, int i2) const         { return __ldg(addr + offset(i0, i1, i2)); }
      CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3) const { return __ldg(addr + offset(i0, i1, i2, i3)); }
    }
    else {
      CUDA_BOTH const TElem& operator()(int i0) const                         { return *(addr + offset(i0));         }
      CUDA_BOTH const TElem& operator()(int i0, int i1) const                 { return *(addr + offset(i0, i1));     }
      CUDA_BOTH const TElem& operator()(int i0, int i1, int i2) const         { return *(addr + offset(i0, i1, i2)); }
      CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3) const { return *(addr + offset(i0, i1, i2, i3)); }
    }
#endif
};

template <typename T, int M>
struct block_1D
{
    T         data[M];
    CUDA_BOTH T& operator()(int idx) { return data[idx]; }
};

template <typename T, int M, int N>
struct block_2D
{
    T         data[M * N];
    CUDA_BOTH T& operator()(int m, int n) { return data[(n * M) + m]; }
};

template <typename T, int L, int M, int N>
struct block_3D
{
    T         data[L * M * N];
    CUDA_BOTH T& operator()(int l, int m, int n) { return data[((n * M) + m) * L + l]; }
};

// Partial specialization of block_1D to use shared memory pointers
template <typename T, int M>
struct block_1D<T*, M>
{
    CUDA_BOTH block_1D(T* pData) :
        m_pData(pData){}; // static_assert(std::is_pointer<T>::value, "Must be a pointer type")
    block_1D()                    = delete;
    block_1D(block_1D const& blk) = delete;
    CUDA_BOTH block_1D& operator  =(block_1D const& block) { m_pData = block.m_pData; };
    ~block_1D()                   = default;

    CUDA_BOTH T&               operator()(int idx) { return m_pData[idx]; }
    static constexpr CUDA_BOTH size_t num_elem() { return M; }

private:
    T* m_pData = nullptr;
};

// Partial specialization of block_2D to use shared memory pointers
template <typename T, int M, int N>
struct block_2D<T*, M, N>
{
    CUDA_BOTH block_2D(T* pData) :
        m_pData(pData){};
    block_2D()                    = delete;
    block_2D(block_2D const& blk) = delete;
    CUDA_BOTH block_2D& operator  =(block_2D const& block) { m_pData = block.m_pData; };
    ~block_2D()                   = default;

    CUDA_BOTH T&               operator()(int m, int n) { return m_pData[(n * M) + m]; }
    static constexpr CUDA_BOTH size_t num_elem() { return M * N; }

private:
    T* m_pData = nullptr;
};

// Partial specialization of block_3D to use shared memory pointers
template <typename T, int L, int M, int N>
struct block_3D<T*, L, M, N>
{
    CUDA_BOTH block_3D(T* pData) :
        m_pData(pData){};
    block_3D()                    = delete;
    block_3D(block_3D const& blk) = delete;
    CUDA_BOTH block_3D& operator  =(block_3D const& block) { m_pData = block.m_pData; };
    ~block_3D()                   = default;

    CUDA_BOTH T&               operator()(int l, int m, int n) { return m_pData[((n * M) + m) * L + l]; }
    static constexpr CUDA_BOTH size_t num_elem() { return L * M * N; }

private:
    T* m_pData = nullptr;
};

// clang-format off
template <typename T> CUDA_BOTH_INLINE T         cuGet(int);
// template<>            CUDA_BOTH_INLINE __half    cuGet(int x) { return(__half(x)); }
template<>            CUDA_BOTH_INLINE float     cuGet(int x) { return(float(x)); }
// template<>            CUDA_BOTH_INLINE __half2 cuGet(int x) { return(make_cuComplex(float(x), 0.0f)); }
template<>            CUDA_BOTH_INLINE cuComplex cuGet(int x) { return(make_cuComplex(float(x), 0.0f)); }
// template<>             CUDA_BOTH_INLINE __half2 cuGet(int x) { return __halves2half2(__half(x), __half(0)); }

template <typename T> CUDA_BOTH_INLINE T         cuGet(float);
template<>            CUDA_BOTH_INLINE float     cuGet(float x) { return(float(x)); }
template<>            CUDA_BOTH_INLINE cuComplex cuGet(float x) { return(make_cuComplex(x, 0.0f)); }

template <typename T> CUDA_BOTH_INLINE T         cuAbs(T);
template<>            CUDA_BOTH_INLINE float     cuAbs(float x) { return(fabsf(x)); }

template <typename T> CUDA_INLINE T         cuRSqrt(T);
template<>            CUDA_INLINE float     cuRSqrt(float x) { return(rsqrtf(x)); }
template<>            CUDA_INLINE half      cuRSqrt(half x)  { return(hrsqrt(x)); }

static CUDA_BOTH_INLINE float     cuReal(cuComplex x)                   { return(cuCrealf(x)); }
static CUDA_BOTH_INLINE float     cuImag(cuComplex x)                   { return(cuCimagf(x)); }
static CUDA_BOTH_INLINE cuComplex cuConj(cuComplex x)                   { return(cuConjf(x)); }
static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, float y)       { return(make_cuComplex(cuCrealf(x)*y, cuCimagf(x)*y)); }
static CUDA_BOTH_INLINE cuComplex operator*(float x, cuComplex y)       { return(make_cuComplex(cuCrealf(y)*x, cuCimagf(y)*x)); }
static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, cuComplex y)   { return(cuCmulf(x, y)); }
static CUDA_BOTH_INLINE cuComplex operator+=(cuComplex &x, cuComplex y) { x = cuCaddf(x, y); return x; };
static CUDA_BOTH_INLINE cuComplex cuFma(cuComplex x, cuComplex y, cuComplex a) { return cuCfmaf(x,y,a); }// a = (x*y) + a;

#if 0
static CUDA_BOTH_INLINE cuComplex operator-(cuComplex x, cuComplex y)   { return(cuCsubf(x, y)); }
static CUDA_BOTH_INLINE cuComplex operator+=(cuComplex &x, float y)     { x = make_cuComplex(cuCrealf(x) + y, cuCimagf(x)); return x; }
static CUDA_BOTH_INLINE cuComplex operator*=(cuComplex &x, float y)     { x = make_cuComplex(cuCrealf(x)*y, cuCimagf(x)*y); return x; }
static CUDA_BOTH_INLINE float cuCRmul(cuComplex x, cuComplex y) { return((cuCrealf(x) * cuCrealf(y)) - (cuCimagf(x) * cuCimagf(y))); }
static CUDA_BOTH_INLINE float cuCImul(cuComplex x, cuComplex y) { return((cuCrealf(x) * cuCimagf(y)) + (cuCimagf(x) * cuCrealf(y))); }
static CUDA_BOTH_INLINE cuComplex cuNeg(cuComplex x)                  { return(make_cuComplex(-cuCrealf(x), -cuCimagf(x))); }
static CUDA_BOTH_INLINE cuComplex cuAdd(cuComplex x, cuComplex y)       { return(cuCaddf(x, y)); }
static CUDA_BOTH_INLINE cuComplex cuMul(cuComplex x, cuComplex y)       { return(cuCmulf(x, y)); }
static CUDA_BOTH_INLINE cuComplex cuDiv(cuComplex x, cuComplex y)       { return(cuCdivf(x, y)); }
static CUDA_BOTH_INLINE cuComplex operator+(cuComplex x, cuComplex y)   { return(cuCaddf(x, y)); }
static CUDA_BOTH_INLINE cuComplex operator-=(cuComplex x, cuComplex y)  { return(cuCsubf(x, y)); }
static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, cuComplex y)   { return(cuCmulf(x, y)); }
static CUDA_BOTH_INLINE cuComplex operator/(cuComplex x, float y)       { return(make_cuComplex(cuCrealf(x)/y, cuCimagf(x)/y)); }
static CUDA_BOTH_INLINE cuComplex operator*=(cuComplex &x, cuComplex y) { x = cuCmulf(x, y); return x; };
static CUDA_BOTH_INLINE cuComplex operator/=(cuComplex &x, float y)     { x = make_cuComplex(cuCrealf(x)/y, cuCimagf(x)/y); return x; }

// cuda_fp16.hpp
//__device__ __forceinline__ __half operator*(const __half &lh, const __half &rh) { return __hmul(lh, rh); }  
// __device__ __forceinline__ __half2& operator*=(__half2 &lh, const __half2 &rh) { lh = __hmul2(lh, rh); return lh; } 

// static CUDA_BOTH_INLINE __half2 cuConj(__half2 &hc) { __half2 t; t.x = hc.x; t.y = -hc.y; return t; }
// static CUDA_BOTH_INLINE __half2 cuGet(int x) {  __half2 t; t.x = __half(x); t.y = __float2half(0.0f); return t; } 
#endif
// clang-format on

template <typename T>
CUDA_BOTH_INLINE constexpr T div_round_up(T val, T divide_by)
{
    return ((val + (divide_by - 1)) / divide_by);
}

template <typename TStorageIn,
          typename TCompute,
          uint32_t N_ROWS_MAT,
          uint32_t N_COLS_MAT>
__device__ __forceinline__ void cmplxMatLoadColMjr(thread_block const&                                                                      thisThrdBlk,
                                                   block_2D<const typename complex_from_scalar<TStorageIn>::type*, N_ROWS_MAT, N_COLS_MAT>& srcMat,
                                                   block_2D<typename complex_from_scalar<TCompute>::type*, N_ROWS_MAT + 1, N_COLS_MAT>&     dstMat)
{
    typedef typename complex_from_scalar<TStorageIn>::type TComplexStorageIn;
    typedef typename complex_from_scalar<TCompute>::type   TComplexCompute;

    const uint32_t     N_THRDS                 = thisThrdBlk.size();
    const uint32_t     THRD_IDX                = thisThrdBlk.thread_rank();
    constexpr uint32_t N_MAT_ELEMS_TO_RD       = N_ROWS_MAT * N_COLS_MAT;
    const uint32_t     N_MAT_ELEMS_RD_PER_ITER = (N_MAT_ELEMS_TO_RD > N_THRDS) ? N_THRDS : N_MAT_ELEMS_TO_RD;
    const uint32_t     N_ITER_TO_RD_MAT        = div_round_up(N_MAT_ELEMS_TO_RD, N_MAT_ELEMS_RD_PER_ITER);

    for(uint32_t i = 0; i < N_ITER_TO_RD_MAT; ++i)
    {
        uint32_t matElemIdx = ((i * N_MAT_ELEMS_RD_PER_ITER) + THRD_IDX);
        uint32_t iRow       = matElemIdx % N_ROWS_MAT;
        uint32_t iCol       = matElemIdx / N_ROWS_MAT;
        // Not all threads would participate in the last iteration
        if(matElemIdx < N_MAT_ELEMS_TO_RD)
        {
            dstMat(iRow, iCol) =
                type_convert<TComplexCompute>(srcMat(iRow, iCol));

#ifdef ENABLE_DEBUG
            printf("Mat[%d][%d] = %f+j%f\n", iRow, iCol, dstMat(iRow, iCol).x, dstMat(iRow, iCol).y);
#endif
        }
    }
}

template <typename TStorageIn,
          typename TCompute,
          uint32_t N_ROWS_MAT,
          uint32_t N_COLS_MAT>
__device__ __forceinline__ void cmplxMatLoadRowMjr(thread_block const&                                                                  thisThrdBlk,
                                                   uint32_t                                                                             iPrb,
                                                   tensor_ref_v0<const typename complex_from_scalar<TStorageIn>::type, 3>                  tMatSrc, // (N_BS_ANTS, N_PRB, N_LAYERS)
                                                   block_2D<typename complex_from_scalar<TCompute>::type*, N_ROWS_MAT + 1, N_COLS_MAT>& matDst)
{
    typedef typename complex_from_scalar<TStorageIn>::type TComplexStorageIn;
    typedef typename complex_from_scalar<TCompute>::type   TComplexCompute;

    const uint32_t     N_THRDS                 = thisThrdBlk.size();
    const uint32_t     THRD_IDX                = thisThrdBlk.thread_rank();
    constexpr uint32_t N_MAT_ELEMS_TO_RD       = N_ROWS_MAT * N_COLS_MAT;
    const uint32_t     N_MAT_ELEMS_RD_PER_ITER = (N_MAT_ELEMS_TO_RD > N_THRDS) ? N_THRDS : N_MAT_ELEMS_TO_RD;
    const uint32_t     N_ITER_TO_RD_MAT        = div_round_up(N_MAT_ELEMS_TO_RD, N_MAT_ELEMS_RD_PER_ITER);

    for(uint32_t i = 0; i < N_ITER_TO_RD_MAT; ++i)
    {
        uint32_t matElemIdx = ((i * N_MAT_ELEMS_RD_PER_ITER) + THRD_IDX);
        uint32_t iCol       = matElemIdx % N_COLS_MAT;
        uint32_t iRow       = matElemIdx / N_COLS_MAT;
        // Not all threads may participate in the last iteration
        if(matElemIdx < N_MAT_ELEMS_TO_RD)
        {
#if 1 // Higher SRS_CH_EST store efficiency (but affects BFC load efficiency): (N_SRS_CH_EST_IN_FREQ, N_BS_ANTS, N_LAYERS)
            matDst(iRow, iCol) =
                type_convert<TComplexCompute>(tMatSrc(iPrb, iCol, iRow));
#else // BFW friendly load format (but negatively affects SRS_CH_EST store efficiency): (N_BS_ANTS, N_SRS_CH_EST_IN_FREQ, N_LAYERS)
            matDst(iRow, iCol) =
                type_convert<TComplexCompute>(tMatSrc(iCol, iPrb, iRow));
#endif

#ifdef ENABLE_DEBUG
            printf("Mat[%d][%d][%d] = %f+j%f\n", iPrb, iRow, iCol, matDst(iRow, iCol).x, matDst(iRow, iCol).y);
#endif
        }
    }
}

template <typename TStorageOut,
          typename TCompute,
          uint32_t N_ROWS_MAT,
          uint32_t N_COLS_MAT>
__device__ __forceinline__ void cmplxMatStore_v0(thread_block const&                                                                  thisThrdBlk,
                                                 uint32_t                                                                             iPrb,
                                                 TCompute                                                                             scale,
                                                 block_2D<typename complex_from_scalar<TCompute>::type*, N_ROWS_MAT + 1, N_COLS_MAT>& matSrc,
                                                 tensor_ref_v0<typename complex_from_scalar<TStorageOut>::type, 3>                       tMatDst) // (N_BS_ANTS, N_LAYERS, N_PRB)
{
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;
    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;

    const uint32_t     N_THRDS                 = thisThrdBlk.size();
    const uint32_t     THRD_IDX                = thisThrdBlk.thread_rank();
    constexpr uint32_t N_MAT_ELEMS_TO_WR       = N_ROWS_MAT * N_COLS_MAT;
    const uint32_t     N_MAT_ELEMS_WR_PER_ITER = (N_MAT_ELEMS_TO_WR > N_THRDS) ? N_THRDS : N_MAT_ELEMS_TO_WR;
    const uint32_t     N_ITER_TO_WR_MAT        = div_round_up(N_MAT_ELEMS_TO_WR, N_MAT_ELEMS_WR_PER_ITER);

    for(uint32_t i = 0; i < N_ITER_TO_WR_MAT; ++i)
    {
        uint32_t matElemIdx = ((i * N_MAT_ELEMS_WR_PER_ITER) + THRD_IDX);
        uint32_t iRow       = matElemIdx % N_ROWS_MAT;
        uint32_t iCol       = matElemIdx / N_ROWS_MAT;
        // Not all threads would participate in the last iteration
        if(matElemIdx < N_MAT_ELEMS_TO_WR)
        {
            tMatDst(iRow, iCol, iPrb) =
                type_convert<TComplexStorageOut>(matSrc(iRow, iCol) * scale);

#ifdef ENABLE_DEBUG
            printf("Mat[%d][%d][%d] = %f+j%f\n", iPrb, iRow, iCol, tMatDst(iRow, iCol, iPrb).x, tMatDst(iRow, iCol, iPrb).y);
#endif
        }
    }
}

template <typename TStorageOut,
          typename TCompute,
          uint32_t N_ROWS_MAT,
          uint32_t N_COLS_MAT>
__device__ __forceinline__ void cmplxMatStore(thread_block const&                                                                  thisThrdBlk,
                                              uint32_t                                                                             iPrb,
                                              TCompute                                                                             scale,
                                              block_2D<typename complex_from_scalar<TCompute>::type*, N_ROWS_MAT + 1, N_COLS_MAT>& matSrc,
                                              tensor_ref<typename complex_from_scalar<TStorageOut>::type>                          tMatDst) // (N_BS_ANTS, N_LAYERS, N_PRB)
{
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;
    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;

    const uint32_t     N_THRDS                 = thisThrdBlk.size();
    const uint32_t     THRD_IDX                = thisThrdBlk.thread_rank();
    constexpr uint32_t N_MAT_ELEMS_TO_WR       = N_ROWS_MAT * N_COLS_MAT;
    const uint32_t     N_MAT_ELEMS_WR_PER_ITER = (N_MAT_ELEMS_TO_WR > N_THRDS) ? N_THRDS : N_MAT_ELEMS_TO_WR;
    const uint32_t     N_ITER_TO_WR_MAT        = div_round_up(N_MAT_ELEMS_TO_WR, N_MAT_ELEMS_WR_PER_ITER);

    for(uint32_t i = 0; i < N_ITER_TO_WR_MAT; ++i)
    {
        uint32_t matElemIdx = ((i * N_MAT_ELEMS_WR_PER_ITER) + THRD_IDX);
        uint32_t iRow       = matElemIdx % N_ROWS_MAT;
        uint32_t iCol       = matElemIdx / N_ROWS_MAT;
        // Not all threads would participate in the last iteration
        if(matElemIdx < N_MAT_ELEMS_TO_WR)
        {
            tMatDst(iRow, iCol, iPrb) =
                type_convert<TComplexStorageOut>(matSrc(iRow, iCol) * scale);

#ifdef ENABLE_DEBUG
            printf("Mat[%d][%d][%d] = %f+j%f\n", iPrb, iRow, iCol, tMatDst(iRow, iCol, iPrb).x, tMatDst(iRow, iCol, iPrb).y);
#endif
        }
    }
}

template <typename TCompute, uint32_t N_ANT, uint32_t N_LAYERS, uint32_t N_THREADS>
__device__ void compMatStore(const typename complex_from_scalar<TCompute>::type* input,
                             uint8_t* output,
                             uint32_t ngrps,
                             float beta,
                             uint8_t compbits)
{
    uint32_t tid     = threadIdx.x;
    uint32_t iprbgrp = blockIdx.x;
    uint32_t input_index = iprbgrp * N_ANT * N_LAYERS;

    using TComplex = typename complex_from_scalar<TCompute>::type;

    int32_t compbytes = (compbits == 16) ? N_ANT * 4 : 2 * N_ANT / 8 * compbits + 1;
    if(compbits == 32) compbytes = N_ANT * sizeof(TComplex);
    uint32_t output_index = iprbgrp * compbytes;

    __shared__ TComplex smemBlkC[N_LAYERS][N_ANT + 1];
    uint32_t ty = tid / N_ANT;
    uint32_t tx = tid % N_ANT;
    for(uint32_t y = ty; y < N_LAYERS; y += N_THREADS / N_ANT)
    {
        TComplex v = input[input_index + tid];
        //v.x -= 0.5f;
        //v.y -= 0.5f;
        smemBlkC[y][tx] = v;
        input_index += N_THREADS;
    }
    __syncthreads();
    
    bfw_scale_compress_blockFP<TCompute, N_ANT + 1, N_ANT, N_LAYERS, N_THREADS>(
    &smemBlkC[0][0],       // Shared memory input pointer for the antennas
    output + output_index, // Output pointer for the first antenna
    beta,                  // Scaling factor
    compbits,              // Number of compressed bits, if 16=uncompressed, 32=FP pass-through
    tid,                   // 1D thread rank
    ngrps);                // Stride between 2 layers (number of PRB groups)
}


template <uint32_t THRD_GRP_SIZE>
__device__ __forceinline__
    __half2
    thrdGrpAllReduceSum(thread_block_tile<THRD_GRP_SIZE> const& thisThrdGrp, __half2 const& val)
{
    uint32_t thrdGrpSize = thisThrdGrp.size();
    __half2  sum         = val;
    for(int32_t i = thrdGrpSize / 2; i > 0; i /= 2)
    {
        sum.x += __float2half(thisThrdGrp.shfl_xor(sum.x, i));
        sum.y += __float2half(thisThrdGrp.shfl_xor(sum.y, i));
    }
    thisThrdGrp.sync();
    return sum;
}

template <uint32_t THRD_GRP_SIZE>
__device__ __forceinline__
    cuComplex
    thrdGrpAllReduceSum(thread_block_tile<THRD_GRP_SIZE> const& thisThrdGrp, cuComplex const& val)
{
    uint32_t  thrdGrpSize = thisThrdGrp.size();
    cuComplex sum         = val;
    for(int32_t i = thrdGrpSize / 2; i > 0; i /= 2)
    {
        sum.x += thisThrdGrp.shfl_xor(cuReal(sum), i);
        sum.y += thisThrdGrp.shfl_xor(cuImag(sum), i);
    }
    thisThrdGrp.sync();
    return sum;
}

template <uint32_t THRD_GRP_SIZE>
__device__ __forceinline__ __half
thrdGrpAllReduceSum(thread_block_tile<THRD_GRP_SIZE> const& thisThrdGrp, __half const& val)
{
    uint32_t thrdGrpSize = thisThrdGrp.size();
    __half   sum         = val;
    for(int32_t i = thrdGrpSize / 2; i > 0; i /= 2)
    {
        sum += __float2half(thisThrdGrp.shfl_xor(sum, i));
    }
    thisThrdGrp.sync();
    return sum;
}

template <uint32_t THRD_GRP_SIZE>
__device__ __forceinline__ float
thrdGrpAllReduceSum(thread_block_tile<THRD_GRP_SIZE> const& thisThrdGrp, float const& val)
{
    uint32_t thrdGrpSize = thisThrdGrp.size();
    float    sum         = val;
    for(int32_t i = thrdGrpSize / 2; i > 0; i /= 2)
    {
        sum += thisThrdGrp.shfl_xor(sum, i);
    }
    thisThrdGrp.sync();
    return sum;
}

// Inplace LU factorization of Matrix A - Iterative version (submatrix updates done one row at a time)
// Iterative version maybe used if the thread block size is around the length of a row (i.e. N_COLS_MAT) of
// the augmented matrix
template <typename TCompute,
          uint32_t N_ROWS_MAT,
          uint32_t N_COLS_MAT>
__device__ __forceinline__ void luFactorizeIter(thread_block const&                                                                  thisThrdBlk,
                                                block_2D<typename complex_from_scalar<TCompute>::type*, N_ROWS_MAT + 1, N_COLS_MAT>& matA)
{
    typedef typename complex_from_scalar<TCompute>::type TComplexCompute;

    constexpr uint32_t N_OUTER_ITER = (N_ROWS_MAT > 1) ? (N_ROWS_MAT - 1) : 1;    
    const uint32_t THRD_ABS_IDX = thisThrdBlk.thread_rank();

    // Iterate row by row of A applying Gaussian elimination. In each iteration Gaussian elimination
    // annihilates all elements of a column below main diagonal of G. In iteration k annihilate elements
    // G(k+1:n, k ). At the end of all iterations G is transformed to U
    // While transforming G to U, applying Gaussian elimination to other columns of A i.e. matrices I and M
    // produces matrices Linv and F respetively which can then be used to compute Ree and C via back
    // substitution
#pragma unroll
    for(uint32_t k = 0; k < N_OUTER_ITER; ++k)
    {
        // Gaussian elimination on submatrix A(k,k), since we know that A(k+1:n,k) will be annihilated we directly
        // proceed to applying Gaussian elimination on submatrix A(k+1:n,k+1:n)

        // Complex multiplication by inverse of real number is cheaper instead of complex division
        // @todo: add a safety check on Akk to avoid divide by zero
        TCompute minus_one_over_Akk = cuGet<TCompute>(-1) / cuReal(matA(k, k));

#ifdef ENABLE_DEBUG
        printf("Iteration: %d, A[%d][%d] = %f+j%f, inv = %f---------------\n", k, k, k, matA(k, k).x, matA(k, k).y, minus_one_over_Akk);
#endif

#pragma unroll
        for(uint32_t i = k + 1; i < N_ROWS_MAT; ++i)
        {
            // Compute multipliers needed for Gaussian elimination. For storage compactness the multipliers
            // (non-zero elements of Gauss vector/column of L) are stored in the annihilated zero location of
            // columns of U
#ifdef ENABLE_DEBUG
            printf("Before storing multiplier: A[%d][%d] = %f+j%f\n", i, k, matA(i, k).x, matA(i, k).y);
#endif
            // All threads compute multiplier Aik into a register and use it but only one thread stores it back
            TComplexCompute Aik = matA(i, k) * minus_one_over_Akk;

#ifdef ENABLE_DEBUG
            printf("After storing multiplier: A[%d][%d] = %f+j%f\n", i, k, Aik.x, Aik.y);
#endif
            // Perform Gaussian elimination:
            // linear combination of row k and row i starting from column element k+1:N_COLS_A
            if((THRD_ABS_IDX > k) && (THRD_ABS_IDX < N_COLS_MAT))
            {
                matA(i, THRD_ABS_IDX) = cuFma(Aik, matA(k, THRD_ABS_IDX), matA(i, THRD_ABS_IDX));

#ifdef ENABLE_DEBUG
                printf("A[%d][%d] = %f+j%f\n", i, THRD_ABS_IDX, matA(i, THRD_ABS_IDX).x, matA(i, THRD_ABS_IDX).y);
#endif
            }

            /*   
                 These are the updates to the lower triangular, which via Gaussian elimination, are being zeroed.
                 This is not needed.  The updates to matA would be zero if there was sufficient precision.  
                 As they are essentially zero, they are not used anywhere else in the computation.  (Or could be replaced by zeros if somehow needed.)
                 Removing the update removes the need for the frequent block synchronization which is the main performance improvement.

                 Keeping code here, but commented out, since this may be useful in the future for validation.

            // Ensure all threads (which may extend across multiple thrdGrps for nColsA > 32) have read and use shA(i,k) before writing into it
            thisThrdBlk.sync();
            if(0 == (THRD_ABS_IDX))
            {
                matA(i, k) = Aik;
            }
            */
        }

        // Wait for the entire submatrix update
        thisThrdBlk.sync();
    }
}

// Inplace LU factorization of Matrix A - Parallel version (entire sub-matrix update done one in parallel)
// - Parallel version maybe used (over iterative version) if the thread block size is much larger than the
// length of a row (i.e. N_COLS_MAT) of the augmented matrix resulting in fewer inactive threads during LU factorization.
// - This parallel version (luFactorizeParallel_v1) maximizes the number of active threads during the
// sub-matrix update by recomputing indices for each outerloop iteration and also reduces the number of inner
// loop iterations needed to update the submatrix. The index recomputation cost is paid once per outer loop
// iteration.
template <typename TCompute,
          uint32_t N_ROWS_MAT,
          uint32_t N_COLS_MAT>
__device__ __forceinline__ void luFactorizeParallel_v1(thread_block const&                                                                  thisThrdBlk,
                                                       block_2D<typename complex_from_scalar<TCompute>::type*, N_ROWS_MAT + 1, N_COLS_MAT>& matA)
{
    typedef typename complex_from_scalar<TCompute>::type TComplexCompute;

    constexpr uint32_t N_OUTER_ITER = (N_ROWS_MAT > 1) ? (N_ROWS_MAT - 1) : 1;

    const uint32_t N_THRDS      = thisThrdBlk.size();
    const uint32_t THRD_ABS_IDX = thisThrdBlk.thread_rank();

    // Iterate row by row of A applying Gaussian elimination. In each iteration Gaussian elimination
    // annihilates all elements of a column below main diagonal of G. In iteration k annihilate elements
    // G(k+1:n, k ). At the end of all iterations G is transformed to U
    // While transforming G to U, applying Gaussian elimination to other columns of A i.e. matrices I and M
    // produces matrices Linv and F respetively which can then be used to compute Ree and C via back
    // substitution
    // #pragma unroll
    for(uint32_t k = 0; k < N_OUTER_ITER; ++k)
    {
        // Gaussian elimination on submatrix A(k,k), since we know that A(k+1:n,k) will be annihilated we directly
        // proceed to applying Gaussian elimination on submatrix A(k+1:n,k+1:n)

        // Complex multiplication by inverse of real number is cheaper instead of complex division
        // @todo: add a safety check on Akk to avoid divide by zero
        TCompute minus_one_over_Akk = cuGet<TCompute>(-1) / cuReal(matA(k, k));

#ifdef ENABLE_DEBUG
        printf("Iteration: %d, A[%d][%d] = %f+j%f, inv = %f---------------\n", k, k, k, matA(k, k).x, matA(k, k).y, minus_one_over_Akk);
#endif

        // The entire sub-matrix can be updated in parallel (i.e. in addition to columns, the rows are also updated in parallel)
        // to extent permitted by parallelism (i.e. thread count) available in the thread block
        uint32_t subMatStartRowOffset = (k + 1);
        uint32_t subMatStartColOffset = (k + 1);
        uint32_t nRowsSubMat          = N_ROWS_MAT - subMatStartRowOffset;
        uint32_t nColsSubMat          = N_COLS_MAT - subMatStartColOffset;
        uint32_t subMatColIdx         = THRD_ABS_IDX % nColsSubMat;
        uint32_t matColIdx            = subMatStartColOffset + subMatColIdx; // process columns > k, note: matrix is in column major layout

        // Ensure whole rows are updated at a time
        // Assumes N_THRDS_X >= nColsSubMat
        uint32_t nRowsSubMatPerIter = N_THRDS / nColsSubMat;
        bool     thrdEnable         = (THRD_ABS_IDX < (nRowsSubMatPerIter * nColsSubMat)); // Disable threads which don't update full rows
        uint32_t nIterToProcSubMat  = div_round_up(nRowsSubMat, nRowsSubMatPerIter);
        for(uint32_t i = 0; i < nIterToProcSubMat; ++i)
        {
            uint32_t subMatRowIdx = (i * nRowsSubMatPerIter) + (THRD_ABS_IDX / nColsSubMat);
            uint32_t matRowIdx    = subMatStartRowOffset + subMatRowIdx; // process rows > k

            TComplexCompute Aik = cuGet<TComplexCompute>(0);
            if(thrdEnable && (matRowIdx < N_ROWS_MAT))
            {
                // Compute multipliers needed for Gaussian elimination. For storage compactness the multipliers
                // (non-zero elements of Gauss vector/column of L) are stored in the annihilated zero location of
                // columns of U

#ifdef ENABLE_DEBUG
                printf("Before storing multiplier: A[%d][%d] = %f+j%f\n", matRowIdx, k, matA(matRowIdx, k).x, matA(matRowIdx, k).y);
#endif
                // All threads compute multiplier Aik into a register and use it but only one thread stores it back
                Aik = matA(matRowIdx, k) * minus_one_over_Akk;

#ifdef ENABLE_DEBUG
                printf("After storing multiplier: A[%d][%d] = %f+j%f\n", matRowIdx, k, Aik.x, Aik.y);
#endif
                // Perform Gaussian elimination:
                // linear combination of row k and row i starting from column element k+1:N_COLS_A
                // if((THRD_ABS_IDX > k) && (THRD_ABS_IDX < N_COLS_MAT))
                if(matColIdx < N_COLS_MAT)
                {
                    matA(matRowIdx, matColIdx) = cuFma(Aik, matA(k, matColIdx), matA(matRowIdx, matColIdx));

#ifdef ENABLE_DEBUG
                    printf("A[%d][%d] = %f+j%f\n", matRowIdx, matColIdx, matA(matRowIdx, matColIdx).x, matA(matRowIdx, matColIdx).y);
#endif
                }
            }

            /*
                 These are the updates to the lower triangular, which via Gaussian elimination, are being zeroed.
                 This is not needed.  The updates to matA would be zero if there was sufficient precision.
                 As they are essentially zero, they are not used anywhere else in the computation.  (Or could be replaced by zeros if somehow needed.)
                 Removing the update removes the need for the frequent block synchronization which is the main performance improvement.

                 Keeping code here, but commented out, since this may be useful in the future for validation.

            // Ensure all threads (which may extend across multiple thrdGrps for nColsA > 32) have read and use shA(i,k) before writing into it
            thisThrdBlk.sync();
            if(thrdEnable && (matRowIdx < N_ROWS_MAT) && (subMatStartColOffset == matColIdx))
            {
                matA(matRowIdx, k) = Aik;
            }

            */
            
        }

        // Wait for the entire submatrix update
        thisThrdBlk.sync();
    }
}

// Inplace LU factorization of Matrix - Parallel version (entire sub-matrix update done one in parallel)
// - Parallel version maybe used (over iterative version) if the thread block size is much larger than the
// length of a row (i.e. N_COLS_MAT) of the augmented matrix resulting in fewer inactive threads during LU factorization.
// - This parallel version computes indices used in sub-matrix update once before the outerloop eliminating
// per outer loop index recomputation cost. Consequently (unlike luFactorizeParallel_v1) the smaller sub-matrix
// updates do not utilize all the availalbe threads (inactive thread count increases with smaller sub-matrices)
// while the number of inner loop iterations does not decrease with decrease in sub-matrix dimension.
template <typename TCompute,
          uint32_t N_ROWS_MAT,
          uint32_t N_COLS_MAT>
__device__ __forceinline__ void luFactorizeParallel_v2(thread_block const&                                                                  thisThrdBlk,
                                                       block_2D<typename complex_from_scalar<TCompute>::type*, N_ROWS_MAT + 1, N_COLS_MAT>& matA)
{
    typedef typename complex_from_scalar<TCompute>::type TComplexCompute;

    const uint32_t N_THRDS      = thisThrdBlk.size();
    const uint32_t THRD_ABS_IDX = thisThrdBlk.thread_rank();

    // Ensure whole rows are updated at a time
    // Assumes N_THRDS_X >= nColsSubMat
    uint32_t nRowsMatPerIter = N_THRDS / N_COLS_MAT;
    bool     thrdEnableMain  = (THRD_ABS_IDX < (nRowsMatPerIter * N_COLS_MAT)); // Disable threads which don't update full rows
    uint32_t nIterToProcMat  = div_round_up(N_ROWS_MAT, nRowsMatPerIter);
    uint32_t matColIdx       = THRD_ABS_IDX % N_COLS_MAT;
    uint32_t matRowOffset    = THRD_ABS_IDX / N_COLS_MAT;

    // Iterate row by row of A applying Gaussian elimination. In each iteration Gaussian elimination
    // annihilates all elements of a column below main diagonal of G. In iteration k annihilate elements
    // G(k+1:n, k ). At the end of all iterations G is transformed to U
    // While transforming G to U, applying Gaussian elimination to other columns of A i.e. matrices I and M
    // produces matrices Linv and F respetively which can then be used to compute Ree and C via back
    // substitution
    // #pragma unroll
    for(uint32_t k = 0; k < N_ROWS_MAT - 1; ++k)
    {
        // Gaussian elimination on submatrix A(k,k), since we know that A(k+1:n,k) will be annihilated we directly
        // proceed to applying Gaussian elimination on submatrix A(k+1:n,k+1:n)

        // Complex multiplication by inverse of real number is cheaper instead of complex division
        // @todo: add a safety check on Akk to avoid divide by zero
        TCompute minus_one_over_Akk = cuGet<TCompute>(-1) / cuReal(matA(k, k));

#ifdef ENABLE_DEBUG
        printf("Iteration: %d, A[%d][%d] = %f+j%f, inv = %f---------------\n", k, k, k, matA(k, k).x, matA(k, k).y, minus_one_over_Akk);
#endif

        // The entire sub-matrix can be updated in parallel (i.e. in addition to columns, the rows are also updated in parallel)
        // to extent permitted by parallelism (i.e. thread count) available in the thread block
        for(uint32_t i = 0; i < nIterToProcMat; ++i)
        {
            uint32_t matRowIdx = (i * nRowsMatPerIter) + matRowOffset;
            // process rows > k and process columns > k
            bool thrdEnable = thrdEnableMain && ((matRowIdx > k) && (matRowIdx < N_ROWS_MAT) &&
                                                 (matColIdx > k));

            TComplexCompute Aik = cuGet<TComplexCompute>(0);
            if(thrdEnable)
            {
                // Compute multipliers needed for Gaussian elimination. For storage compactness the multipliers
                // (non-zero elements of Gauss vector/column of L) are stored in the annihilated zero location of
                // columns of U

#ifdef ENABLE_DEBUG
                printf("Before storing multiplier: A[%d][%d] = %f+j%f\n", matRowIdx, k, matA(matRowIdx, k).x, matA(matRowIdx, k).y);
#endif
                // All threads compute multiplier Aik into a register and use it but only one thread stores it back
                Aik = matA(matRowIdx, k) * minus_one_over_Akk;

#ifdef ENABLE_DEBUG
                printf("After storing multiplier: A[%d][%d] = %f+j%f\n", matRowIdx, k, Aik.x, Aik.y);
#endif
                // Perform Gaussian elimination:
                // linear combination of row k and row i starting from column element k+1:N_COLS_A
                matA(matRowIdx, matColIdx) = cuFma(Aik, matA(k, matColIdx), matA(matRowIdx, matColIdx));

#ifdef ENABLE_DEBUG
                printf("A[%d][%d] = %f+j%f\n", matRowIdx, matColIdx, matA(matRowIdx, matColIdx).x, matA(matRowIdx, matColIdx).y);
#endif
            }

            // Ensure all threads (which may extend across multiple thrdGrps for nColsA > 32) have read and use shA(i,k) before writing into it
            thisThrdBlk.sync();
            if(thrdEnable && ((k + 1) == matColIdx))
            {
                matA(matRowIdx, k) = Aik;
            }
        }

        // Wait for the entire submatrix update
        thisThrdBlk.sync();
    }
}

// BeamFormingCancellation (BFC) coefficient computation kernel
// {N_LAYERS, N_BS_ANTS} = {16,64}
// Inputs and outputs assumed to be column major
// dimBlock: (N_THREADS_PER_WARP, N_LAYERS)
// dimGrid : (Nprb)
template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS, // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS,  // # of layers (# of cols in H matrix)
          uint32_t N_THRD_GRPS_PER_THRD_BLK,
          uint32_t N_THRDS_PER_GRP>
__global__ void
bfc_mmse_coef_comp_kernel_v0(tensor_ref_v0<const typename complex_from_scalar<TStorageIn>::type, 3> tH,      // (N_BS_ANTS, N_PRB, N_LAYERS)
                             tensor_ref_v0<const TStorageIn, 2>                                     tLambda, // (N_LAYERS, N_PRB)
                             tensor_ref_v0<typename complex_from_scalar<TStorageOut>::type, 3>      tCoef,   // (N_BS_ANTS, N_LAYERS, N_PRB)
                             tensor_ref_v0<typename complex_from_scalar<TStorageOut>::type, 4>      tDbg)
{
    // H is channel matrix
    // G is the enhanced Gram matrix
    // A is the augmented matrix, A = [ G | I | H ]

    //--------------------------------------------------------------------------------------------------------
    // Dimensions

    // H  : Channel matrix
    constexpr uint32_t N_ROWS_H = N_LAYERS;
    constexpr uint32_t N_COLS_H = N_BS_ANTS;

    // R  : Diagonal matrix (lambda) with per layer regularization coefficients
    constexpr uint32_t N_ROWS_R = N_LAYERS;
    // constexpr uint32_t N_COLS_R = N_LAYERS;

    // G  : Enhanced Gram matrix, G = H*H' + R
    constexpr uint32_t N_ROWS_G = N_LAYERS;
    constexpr uint32_t N_COLS_G = N_LAYERS;

    // I  : Identity matrix
    constexpr uint32_t N_ROWS_I = N_LAYERS;
    constexpr uint32_t N_COLS_I = N_LAYERS;

    // Linv: inverse lower trianuglar matrix in LU factorization
    constexpr uint32_t N_ROWS_LINV = N_LAYERS;

    // U  : Upper triangular matrix
    // constexpr uint32_t N_ROWS_U = N_ROWS_G;
    constexpr uint32_t N_COLS_U = N_COLS_G;

    // C  : MMSE coefficient matrix, C = H'*Ginv = H'*inv(H*H' + D)
    constexpr uint32_t N_ROWS_C = N_COLS_H;
    constexpr uint32_t N_COLS_C = N_COLS_G;

    // A  : Augmented result matrix, A = [ G | I | H ] -> [ U | Linv | F ]
    constexpr uint32_t N_ROWS_A = N_ROWS_G;
    constexpr uint32_t N_COLS_A = N_COLS_G + N_COLS_I + N_COLS_H;

    static_assert((N_THRDS_PER_GRP <= N_THREADS_PER_WARP), "Using co-operative groups");
    static_assert((0 == N_BS_ANTS % N_THRDS_PER_GRP) && (N_BS_ANTS >= N_THRDS_PER_GRP), "Expect BS antenna to be a multiple of thread group size");
    static_assert((0 == N_THRDS_PER_GRP % N_LAYERS) && (N_THRDS_PER_GRP >= N_LAYERS), "Expect thread group size to be a multiple of layer count"); 

    thread_block const& thisThrdBlk = this_thread_block();

    // Co-operative thread groups used in computation of inner products
    thread_block_tile<N_THRDS_PER_GRP> const& thrdGrp = tiled_partition<N_THRDS_PER_GRP>(thisThrdBlk);

    // G is Hermitian symmetric i.e. only the upper or lower diagonal elements need to be computed
    constexpr uint32_t N_TRI_ELEMS_G = N_ROWS_G * (N_ROWS_G + 1) / 2;

    // Iterations to compute one element of G. Each thread group computes the inner product needed to produce
    // one element of G
    constexpr uint32_t N_INNER_ITER_TO_COMP_G_ELEM = div_round_up(N_COLS_H, N_THRDS_PER_GRP);

    // Each thread group computes one element of G per outer loop iteration
    constexpr uint32_t N_OUTER_ITER_TO_COMP_G = div_round_up(N_TRI_ELEMS_G, N_THRD_GRPS_PER_THRD_BLK);

    // Each thrdGrp computes either part of or whole column of C
    constexpr uint32_t N_MAX_INNER_ITER_TO_COMP_C = N_ROWS_LINV;
    constexpr uint32_t N_THRD_GRPS_PER_C_COL_COMP = N_ROWS_C / N_THRDS_PER_GRP;
    constexpr uint32_t N_COLS_C_COMP_PER_THRD_BLK = N_THRD_GRPS_PER_THRD_BLK / N_THRD_GRPS_PER_C_COL_COMP;
    constexpr uint32_t N_OUTER_ITER_TO_COMP_C     = (N_COLS_C >= N_COLS_C_COMP_PER_THRD_BLK) ? (N_COLS_C / N_COLS_C_COMP_PER_THRD_BLK) : 1;

    
    // Number of iterations to compute Frobeius norm
    constexpr uint32_t N_INNER_ITER_TO_COMP_FNORM = N_ROWS_C / N_THRDS_PER_GRP;
    constexpr uint32_t N_OUTER_ITER_TO_COMP_FNORM = div_round_up(N_COLS_C, N_THRD_GRPS_PER_THRD_BLK);

    //--------------------------------------------------------------------------------------------------------
    // Compute indices used for element access
    const uint32_t THRD_IDX     = threadIdx.x; // thrdGrp.thread_rank()
    const uint32_t THRD_GRP_IDX = threadIdx.y;
    const uint32_t THRD_ABS_IDX = (threadIdx.y * blockDim.x) + threadIdx.x;

    const uint32_t PRB_IDX = blockIdx.x;

    const uint32_t ROW_IDX_R = THRD_ABS_IDX % N_ROWS_R;

    const uint32_t ROW_IDX_I = THRD_ABS_IDX % N_ROWS_I;
    const uint32_t COL_IDX_I = THRD_ABS_IDX / N_ROWS_I;

    // const uint32_t ROW_IDX_G = THRD_ABS_IDX % N_ROWS_G;
    // const uint32_t COL_IDX_G = THRD_ABS_IDX / N_ROWS_G; // COL_IDX_G needs a bounds check (since N_THRDS_X > # of G elements)

    // const uint32_t ROW_IDX_C = THRD_ABS_IDX % N_ROWS_C;
    // const uint32_t COL_IDX_C = THRD_ABS_IDX / N_ROWS_C;

    //--------------------------------------------------------------------------------------------------------
    // Shared memory allocation
    // H[N_TONES_PER_ITER*N_INST]

    // Shared memory contents as processing progresses:
    // A = [ G | I | H ] -> [ U | Linv | F ]

    constexpr uint32_t N_SMEM_R_ELEMS = N_ROWS_R;
    constexpr uint32_t N_SMEM_A_ELEMS = (N_ROWS_A + 1) * N_COLS_A; // (N_ROWS_A + 1) for SMEM padding to avoid bank conflicts
    constexpr uint32_t N_SMEM_C_ELEMS = (N_ROWS_C + 1) * N_COLS_C; // (N_ROWS_C + 1) for SMEM padding to avoid bank conflicts

    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    __shared__ TCompute smemBlkR[N_SMEM_R_ELEMS];
    __shared__ TComplexCompute smemBlkA[N_SMEM_A_ELEMS];
    __shared__ TComplexCompute smemBlkC[N_SMEM_C_ELEMS];
    __shared__ TCompute shCFrobeniusNorm;

    constexpr uint32_t            SMEM_START_OFFSET_R = 0;
    block_1D<TCompute*, N_ROWS_R> shR(&smemBlkR[SMEM_START_OFFSET_R]);

    constexpr uint32_t                                 SMEM_START_OFFSET_A = 0;
    block_2D<TComplexCompute*, N_ROWS_A + 1, N_COLS_A> shA(&smemBlkA[SMEM_START_OFFSET_A]);

    // SMEM overlay: A with [ G | I | H ]
    const uint32_t                                     SMEM_START_OFFSET_G = SMEM_START_OFFSET_A;
    block_2D<TComplexCompute*, N_ROWS_G + 1, N_COLS_G> shG(&smemBlkA[SMEM_START_OFFSET_G]);

    const uint32_t                                     SMEM_START_OFFSET_I = SMEM_START_OFFSET_G + shG.num_elem();
    block_2D<TComplexCompute*, N_ROWS_I + 1, N_COLS_I> shI(&smemBlkA[SMEM_START_OFFSET_I]);

    const uint32_t                                     SMEM_START_OFFSET_H = SMEM_START_OFFSET_I + shI.num_elem();
    block_2D<TComplexCompute*, N_ROWS_H + 1, N_COLS_H> shH(&smemBlkA[SMEM_START_OFFSET_H]);

    const uint32_t                                     SMEM_START_OFFSET_C = 0;
    block_2D<TComplexCompute*, N_ROWS_C + 1, N_COLS_C> shC(&smemBlkC[SMEM_START_OFFSET_C]);

    // SMEM overlay:
    // After LU - U replaces G, Linv replaces I and F replaces H
    auto& shU    = shG;
    auto& shLinv = shI;
    auto& shF    = shH;

    // Dinv overlays with R
    auto& shDinv = shR;

    //--------------------------------------------------------------------------------------------------------
    // Stage1: Load inputs

#ifdef ENABLE_DEBUG
    if(0 != blockIdx.x) return;
#endif

    cmplxMatLoadRowMjr<TStorageIn, TCompute, N_ROWS_H, N_COLS_H>(thisThrdBlk, PRB_IDX, tH, shH);

    if(THRD_ABS_IDX < N_ROWS_R)
    {
        shR(ROW_IDX_R) = type_convert<TCompute>(tLambda(ROW_IDX_R, PRB_IDX));
    }

    // Wait for loads to complete. Thread(s) processing an entry of H may not be the same ones loading it
    thisThrdBlk.sync();

#ifdef ENABLE_DEBUG
    // H
    for(uint32_t i = 0; i < N_ROWS_H; ++i)
    {
        if(THRD_ABS_IDX < N_COLS_H)
            tDbg(i, THRD_ABS_IDX, PRB_IDX) = type_convert<TComplexStorageOut>(shH(i, THRD_ABS_IDX));
    }
#endif

    //---------------------------------------------------------------------------------------------------
    // Stage1: Compute the enhanced Gram matrix: G = (H*H' + R),  G - N_LAYERS x N_LAYERS
    uint32_t matGRowEndMrkr = N_COLS_G;
    uint32_t iGRow          = 0;
    uint32_t iGCol          = 0;
    uint32_t iGIdx          = 0;
    for(uint32_t i = 0; i < N_OUTER_ITER_TO_COMP_G; ++i)
    {
        // linear index, each thread group computes one element of G per outer loop iteration
        iGIdx = (i * N_THRD_GRPS_PER_THRD_BLK) + THRD_GRP_IDX;

        if(iGIdx >= N_TRI_ELEMS_G) break;

        // Since G is Hermitian symmetric, its sufficient if only the upper (or lower) triangular elements of
        // H*H' are computed
        // Convert linear index to row and column indices of the upper triangular elements of matrix G
        while((iGIdx + iGRow) >= matGRowEndMrkr)
        {
            matGRowEndMrkr += (N_COLS_G - iGRow);
            ++iGRow;
        }
        iGCol = N_COLS_G - (matGRowEndMrkr - iGIdx) + iGRow;

        // Compute G(iGRow,iGCol) via N_BS_ANTS x N_BS_ANTS inner product
        TComplexCompute G = cuGet<TComplexCompute>(0);
        for(uint32_t j = 0; j < N_INNER_ITER_TO_COMP_G_ELEM; ++j)
        {
            uint32_t        iElem = (j * N_THRDS_PER_GRP) + THRD_IDX;
            TComplexCompute prod  = shH(iGRow, iElem) * cuConj(shH(iGCol, iElem));
            G += thrdGrpAllReduceSum<N_THRDS_PER_GRP>(thrdGrp, prod);
        }

        if(0 == THRD_IDX)
        {
            if(iGRow != iGCol)
            {
                shG(iGCol, iGRow) = cuConj(G);
            }
            else
            {
                G.x += shR(iGRow);
            }
            shG(iGRow, iGCol) = G;

            // printf("G[%d][%d] = %f+j%f, linIdx %d, threadIdx (%d,%d), blockIdx.x %d, matGRowEndMrkr %d\n", iGRow, iGCol, cuReal(G), cuImag(G), iGIdx, threadIdx.x, threadIdx.y, blockIdx.x, matGRowEndMrkr);
        }
    }

    if(COL_IDX_I < N_COLS_I)
    {
        shI(ROW_IDX_I, COL_IDX_I) =
            (ROW_IDX_I != COL_IDX_I) ? cuGet<TComplexCompute>(0) : cuGet<TComplexCompute>(1);
    }

    // Wait for G matrix compute and I matrix init to complete
    thisThrdBlk.sync();

#ifdef ENABLE_DEBUG
    // A0
    for(uint32_t i = 0; i < N_ROWS_A; ++i)
    {
        if(THRD_ABS_IDX < N_COLS_A)
            tDbg(i, THRD_ABS_IDX, PRB_IDX) = type_convert<TComplexStorageOut>(shA(i, THRD_ABS_IDX));
    }
#endif

    //---------------------------------------------------------------------------------------------------
    // Stage2: Perform joint LU factorization
    // A = [ G | I | H ] -> [ U | Linv | F ]
    // where U = L\G, Linv = L\I, F = L\H

    // bfc_mmse_coef_comp_kernel_v0: 
    // For Large layer count (e.g. 8, 16) thread block size >> # of columns of augmented matrix
    // (i.e. (N_THRDS_PER_GRP * N_LAYERS) >> (2*N_LAYERS + N_BS_ANTS)). Thus use parallel version of the
    // factorization algorithm to cut down iteration count and increase active threads during sub-matrix
    // updates
    // For small layer counts (e.g. 2, 4) thread block size >= # of columns of augmented matrix. Use iterative
    // version since the iteration count = N_ROWS_A = N_LAYERS is expected to be small and thread block size is 
    // not large relative to N_COLS_A

    ((2 != N_LAYERS) && (4 != N_LAYERS)) ? luFactorizeParallel_v1<TCompute, N_ROWS_A, N_COLS_A>(thisThrdBlk, shA) : 
                                           luFactorizeIter<TCompute, N_ROWS_A, N_COLS_A>(thisThrdBlk, shA);

#ifdef ENABLE_DEBUG
    // A1
    for(uint32_t i = 0; i < N_ROWS_A; ++i)
    {
        if(THRD_ABS_IDX < N_COLS_A)
            tDbg(i, THRD_ABS_IDX, PRB_IDX) = type_convert<TComplexStorageOut>(shA(i, THRD_ABS_IDX));
    }
#endif

    //---------------------------------------------------------------------------------------------------
    // Stage3: Multiply C = F'*(inv(D)*inv(L)), where D = I*(diag(U)), G - N_BS_ANTS x N_LAYERS

    // Compute inv(D)
    if(THRD_ABS_IDX < N_COLS_U)
    {
        shDinv(THRD_ABS_IDX) = cuGet<TCompute>(1) / cuReal(shU(THRD_ABS_IDX, THRD_ABS_IDX));
    }

    // Initialize matrix C Frobenius norm. Use a thread which was not used in above
    if(N_COLS_U == THRD_ABS_IDX)
    {
        shCFrobeniusNorm = cuGet<TCompute>(0);
    }

    thisThrdBlk.sync();

    // Each column of C maybe computed by one or more thread groups, C_COL_COMP_THRD_GRP_IDX is common index
    // of threads computing a column of C
    const uint32_t C_COL_COMP_THRD_GRP_IDX = (THRD_GRP_IDX / N_THRD_GRPS_PER_C_COL_COMP);

    // Due to the nature of lower triangular multiply, some inner products are longer than others (fewer
    // columns of F' to be combined when multpilying with later columns of Linv than initial columns).
    // To balance workload on each thread group, the outerloop iterations are divvied up to multiply as many
    // initial columns of Linv as its later columns
    constexpr uint32_t N_HALF_OUTER_ITER_TO_COMP_C = (N_OUTER_ITER_TO_COMP_C >= 2) ? (N_OUTER_ITER_TO_COMP_C / 2) : 1;

    // Offset to start column to be computed by this thread group
    const uint32_t C_COL_COMP_OFFSET = C_COL_COMP_THRD_GRP_IDX * N_HALF_OUTER_ITER_TO_COMP_C;
    const uint32_t C_ROW_COMP_OFFSET = (THRD_GRP_IDX % N_THRD_GRPS_PER_C_COL_COMP) * N_THRDS_PER_GRP;
    const uint32_t C_ROW_IDX         = C_ROW_COMP_OFFSET + THRD_IDX;

    // #pragma unroll
    for(uint32_t i = 0; i < N_OUTER_ITER_TO_COMP_C; ++i)
    {
        // Column index
        int32_t iCCol = C_COL_COMP_OFFSET + (i % N_HALF_OUTER_ITER_TO_COMP_C);
        if(iCCol >= N_COLS_C) continue;  // this condition holds when there are excess threads than needed i.e. N_COLS_C_COMP_PER_THRD_BLK > N_COLS_C (e.g. 1 layer case)

        // Process initial columns of Linv for first half iterations and later columns for the last half iterations
        if(i >= N_HALF_OUTER_ITER_TO_COMP_C) iCCol = N_COLS_C - iCCol - 1;

        // Due to the nature of lower triangular multiply, number of accumulations needed depends on the
        // column of Linv being multiplied
        TComplexCompute C = cuGet<TComplexCompute>(0);
        for(uint32_t iElem = iCCol; iElem < N_MAX_INNER_ITER_TO_COMP_C; ++iElem)
        {
            // Multiply inv(D)*inv(L))
            TComplexCompute DinvLinv = shDinv(iElem) * shLinv(iElem, iCCol);

            // Multiply F'*(inv(D)*inv(L))
            C = cuFma(cuConj(shF(iElem, C_ROW_IDX)), DinvLinv, C);
        }
        // TCompute absCSqr = cuReal(cuConj(C) * C);
        // atomicAdd(&shCFrobeniusNorm, absCSqr);

        shC(C_ROW_IDX, iCCol) = C;
        // printf("C[%d][%d] = %f+j%f, frobNorm %f, threadIdx (%d,%d), blockIdx.x %d i %d C_COL_COMP_OFFSET %d N_HALF_OUTER_ITER_TO_COMP_C %d\n", C_ROW_IDX, iCCol, cuReal(C), cuImag(C), shCFrobeniusNorm, threadIdx.x, threadIdx.y, blockIdx.x, i, C_COL_COMP_OFFSET, N_HALF_OUTER_ITER_TO_COMP_C);
    }

    // Wait for C matrix compute to complete
    thisThrdBlk.sync();

#ifdef ENABLE_DEBUG
    // Coefs pre-norm
    for(uint32_t i = 0; i < N_ROWS_C; ++i)
    {
        if(THRD_ABS_IDX < N_COLS_C)
            tDbg(i, THRD_ABS_IDX, PRB_IDX) = type_convert<TComplexStorageOut>(shC(i, THRD_ABS_IDX));
    }
#endif
    for(uint32_t i = 0; i < N_OUTER_ITER_TO_COMP_FNORM; ++i)
    {
        // Each thread group computes coefficient magnitude for one column of C per outer loop iteration
        uint32_t iColC = (i * N_THRD_GRPS_PER_THRD_BLK) + THRD_GRP_IDX;

        TCompute absCSqrSum = 0;
        for(uint32_t j = 0; j < N_INNER_ITER_TO_COMP_FNORM; ++j)
        {
            uint32_t iRowC = (j * N_THRDS_PER_GRP) + THRD_IDX;

            TCompute absCSqr = cuGet<TCompute>(0);
            if((iRowC < N_ROWS_C) && (iColC < N_COLS_C))
            {
                TComplexCompute C = shC(iRowC, iColC);
                absCSqr = cuReal(cuConj(C) * C);
            }
            absCSqrSum += thrdGrpAllReduceSum<N_THRDS_PER_GRP>(thrdGrp, absCSqr);
        }

        if(0 == THRD_IDX)
        {
            atomicAdd(&shCFrobeniusNorm, absCSqrSum);
        }
    }

    thisThrdBlk.sync();

    // Frobenius norm compute
    if(0 == THRD_ABS_IDX)
    {
#ifdef ENABLE_DEBUG
        printf("FrobNorm[%d] before = %f\n", PRB_IDX, shCFrobeniusNorm);
#endif
        shCFrobeniusNorm = cuRSqrt<TCompute>(shCFrobeniusNorm);
#ifdef ENABLE_DEBUG
        printf("FrobNorm[%d] after = %f\n", PRB_IDX, shCFrobeniusNorm);
#endif
    }

    // Wait for Frobenius norm to be computed
    thisThrdBlk.sync();

    //--------------------------------------------------------------------------------------------------------
    // Stage4: Write the result BFC coefficients C into device memory
    cmplxMatStore_v0<TStorageOut, TCompute, N_ROWS_C, N_COLS_C>(thisThrdBlk, PRB_IDX, shCFrobeniusNorm, shC, tCoef);

#ifdef ENABLE_DEBUG
    // C
    for(uint32_t i = 0; i < N_ROWS_C; ++i)
    {
        if(THRD_ABS_IDX < N_COLS_C)
            printf("C[%d][%d][%d] = %f+j%f\n", PRB_IDX, i, COL_IDX_C, shC(i, COL_IDX_C).x, shC(i, COL_IDX_C).y);
    }
#endif
}

template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS, // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS>  // # of layers (# of cols in H matrix)
void
bfc_mmse_coef_comp_kernel_launch(uint32_t           Nprb,
                                 const_tensor_pair& tH,
                                 const_tensor_pair& tLambda,
                                 tensor_pair&       tCoef,
                                 tensor_pair&       tDbg,
                                 cudaStream_t       strm)
{
    constexpr uint32_t N_THRDS_PER_GRP            = N_THREADS_PER_WARP;
    constexpr uint32_t N_THRD_GRPS_PER_THRD_BLK_1 = N_LAYERS/2; // Large layer count (e.g. 8,16)
    constexpr uint32_t N_THRD_GRPS_PER_THRD_BLK_2 = div_round_up(N_BS_ANTS+2*N_LAYERS, N_THRDS_PER_GRP); // Small layer count (e.g. 2,4)

    constexpr uint32_t N_THRD_GRPS_PER_THRD_BLK = (N_THRD_GRPS_PER_THRD_BLK_1 > N_THRD_GRPS_PER_THRD_BLK_2) ? N_THRD_GRPS_PER_THRD_BLK_1 : N_THRD_GRPS_PER_THRD_BLK_2;

    dim3 gridDim(Nprb);
    dim3 blockDim(N_THRDS_PER_GRP, N_THRD_GRPS_PER_THRD_BLK);

    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    tensor_ref_v0<const TComplexStorageIn, 3> H(tH);
    tensor_ref_v0<const TStorageIn, 2>        Lambda(tLambda);
    tensor_ref_v0<TComplexStorageOut, 3>      C(tCoef);
    tensor_ref_v0<TComplexStorageOut, 4>      Dbg(tDbg);

    // For V100, max permitted shared memory capacity is 96KB

#if 0
    constexpr int32_t  N_ITER = N_THRD_BLK_TONES / N_TONES_PER_ITER;
    constexpr uint32_t N_INST = (1 == N_ITER) ? 1 : 2; // double buffering for pipelining
    constexpr uint32_t N_SMEM_ELEMS =
        (((N_BS_ANTS + 1) * N_LAYERS * N_INST) +
         ((N_LAYERS + 1) * (N_LAYERS + N_LAYERS + N_BS_ANTS))) *
            N_TONES_PER_ITER;

    int nShmemBytes    = N_SMEM_ELEMS * sizeof(TComplexCompute);
    int nMaxShmemBytes = nShmemBytes;
    cudaFuncSetAttribute(bfc_mmse_coef_comp_kernel_v0<TStorageIn, TStorageOut, TCompute, N_THRD_BLK_TONES, N_TONES_PER_ITER, N_BS_ANTS, N_LAYERS, NH>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         nMaxShmemBytes);
#else

    int nShmemBytes = 0;
#endif
    bfc_mmse_coef_comp_kernel_v0<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS, N_THRD_GRPS_PER_THRD_BLK, N_THRDS_PER_GRP>
        <<<gridDim, blockDim, nShmemBytes, strm>>>(H,
                                                   Lambda,
                                                   C,
                                                   Dbg);
}

template <typename TStorageIn, typename TStorageOut, typename TCompute>
void bfc_coef_comp_kernel_launch(uint32_t           nBSAnts,
                                 uint32_t           nLayers,
                                 uint32_t           Nprb,
                                 const_tensor_pair& tH,
                                 const_tensor_pair& tLambda,
                                 tensor_pair&       tCoef,
                                 tensor_pair&       tDbg,
                                 cudaStream_t       strm)
{
    if(64 == nBSAnts)
    {
        constexpr uint32_t N_BS_ANTS = 64; // # of BS antenna (# of rows in H matrix)
        switch(nLayers)
        {
        // nLayers == 16
        case 16:
        {
            constexpr uint32_t N_LAYERS = 16; // # of layers (# of cols in H matrix)
            bfc_mmse_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(Nprb,
                                                                                                     tH,
                                                                                                     tLambda,
                                                                                                     tCoef,
                                                                                                     tDbg,
                                                                                                     strm);
            break;
        }
        // nLayers == 8
        case 8:
        {
            constexpr uint32_t N_LAYERS = 8; // # of layers (# of cols in H matrix)
            bfc_mmse_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(Nprb,
                                                                                                     tH,
                                                                                                     tLambda,
                                                                                                     tCoef,
                                                                                                     tDbg,
                                                                                                     strm);
            break;
        }        
        // nLayers == 4
        case 4:
        {
            constexpr uint32_t N_LAYERS = 4; // # of layers (# of cols in H matrix)
            bfc_mmse_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(Nprb,
                                                                                                     tH,
                                                                                                     tLambda,
                                                                                                     tCoef,
                                                                                                     tDbg,
                                                                                                     strm);
            break;
        }
        // nLayers == 2
        case 2:
        {
            constexpr uint32_t N_LAYERS = 2; // # of layers (# of cols in H matrix)
            bfc_mmse_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(Nprb,
                                                                                                     tH,
                                                                                                     tLambda,
                                                                                                     tCoef,
                                                                                                     tDbg,
                                                                                                     strm);
            break;
        }        
        default:
        {
            NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {}", 
                       __FUNCTION__, nBSAnts, nLayers);
            break;
        }
        }
    }
    // nBSAnts = 32
    else if(32 == nBSAnts)
    {
        constexpr uint32_t N_BS_ANTS = 32; // # of BS antenna (# of rows in H matrix)
        switch(nLayers)
        {
        // nLayers == 8
        case 8:
        {
            constexpr uint32_t N_LAYERS = 8; // # of layers (# of cols in H matrix)
            bfc_mmse_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(Nprb,
                                                                                                     tH,
                                                                                                     tLambda,
                                                                                                     tCoef,
                                                                                                     tDbg,
                                                                                                     strm);
            break;
        }
        // nLayers == 4
        case 4:
        {
            constexpr uint32_t N_LAYERS = 4; // # of layers (# of cols in H matrix)
            bfc_mmse_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(Nprb,
                                                                                                     tH,
                                                                                                     tLambda,
                                                                                                     tCoef,
                                                                                                     tDbg,
                                                                                                     strm);
            break;
        }
        // nLayers == 2
        case 2:
        {
            constexpr uint32_t N_LAYERS = 2; // # of layers (# of cols in H matrix)
            bfc_mmse_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(Nprb,
                                                                                                     tH,
                                                                                                     tLambda,
                                                                                                     tCoef,
                                                                                                     tDbg,
                                                                                                     strm);
            break;
        }
        // nLayers == 1
        case 1:
        {
            constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
            bfc_mmse_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(Nprb,
                                                                                                     tH,
                                                                                                     tLambda,
                                                                                                     tCoef,
                                                                                                     tDbg,
                                                                                                     strm);
            break;
        }        
        default:
        {
            NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {}", 
            __FUNCTION__, nBSAnts, nLayers);
            break;
        }
        }
    }
    else
    {
        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {}", 
            __FUNCTION__, nBSAnts, nLayers);
    }
}

void bfcCoefCompute(uint32_t           nBSAnts,
                    uint32_t           nLayers,
                    uint32_t           Nprb,
                    const_tensor_pair& tH,
                    const_tensor_pair& tLambda,
                    tensor_pair&       tCoef,
                    tensor_pair&       tDbg,
                    cudaStream_t       strm)
{
#ifdef ENABLE_DEBUG
    NVLOGD_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}() begin", __FUNCTION__);
#endif
    using TCompute = float;
    if(CUPHY_C_32F == tH.first.get().type())
    {
        using TStorageIn = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
        if(CUPHY_C_32F == tCoef.first.get().type())
        {
            using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
            bfc_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute>(nBSAnts,
                                                                           nLayers,
                                                                           Nprb,
                                                                           tH,
                                                                           tLambda,
                                                                           tCoef,
                                                                           tDbg,
                                                                           strm);
        }
        else if(CUPHY_C_16F == tCoef.first.get().type())
        {
            using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
            bfc_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute>(nBSAnts,
                                                                           nLayers,
                                                                           Nprb,
                                                                           tH,
                                                                           tLambda,
                                                                           tCoef,
                                                                           tDbg,
                                                                           strm);
        }
        else
        {
            NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type {}/{}", 
                       __FUNCTION__, tH.first.get().type(), tCoef.first.get().type());
        }
    }
    else if((CUPHY_C_16F == tH.first.get().type()) && (CUPHY_C_16F == tCoef.first.get().type()))
    {
        using TStorageIn  = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        bfc_coef_comp_kernel_launch<TStorageIn, TStorageOut, TCompute>(nBSAnts,
                                                                       nLayers,
                                                                       Nprb,
                                                                       tH,
                                                                       tLambda,
                                                                       tCoef,
                                                                       tDbg,
                                                                       strm);
    }
    else
    {
        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type {}/{}", 
            __FUNCTION__, tH.first.get().type(), tCoef.first.get().type());
    }
#ifdef ENABLE_DEBUG
        NVLOGD_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}() end", __FUNCTION__);
#endif
}

/* New Beamforming API -------------------------------------------------------------------------------------------------------------- */

template <typename TStorageIn,
          typename TCompute,
          uint32_t N_ROWS_H_MAT,
          uint32_t N_COLS_H_MAT>
__device__ __forceinline__ void srsChEstLoadRowMjr(thread_block const&                  thisThrdBlk,
                                                   uint32_t                             iPrbGrp,
                                                   bfwCoefCompKernelBfLayerPrm_t const* pBfLayerPrm,
                                                   block_2D<typename complex_from_scalar<TCompute>::type*, N_ROWS_H_MAT+1, N_COLS_H_MAT>& shSrsChEst)
{
    typedef typename complex_from_scalar<TStorageIn>::type TComplexStorageIn;
    typedef typename complex_from_scalar<TCompute>::type   TComplexCompute;

    const uint32_t     N_THRDS                 = thisThrdBlk.size();
    const uint32_t     THRD_IDX                = thisThrdBlk.thread_rank();
    constexpr uint32_t N_MAT_ELEMS_TO_RD       = N_ROWS_H_MAT * N_COLS_H_MAT;
    const uint32_t     N_MAT_ELEMS_RD_PER_ITER = (N_MAT_ELEMS_TO_RD > N_THRDS) ? N_THRDS : N_MAT_ELEMS_TO_RD;
    const uint32_t     N_ITER_TO_RD_MAT        = div_round_up(N_MAT_ELEMS_TO_RD, N_MAT_ELEMS_RD_PER_ITER);

    for(uint32_t i = 0; i < N_ITER_TO_RD_MAT; ++i)
    {
        uint32_t matElemIdx = ((i * N_MAT_ELEMS_RD_PER_ITER) + THRD_IDX);
        uint32_t iCol       = matElemIdx % N_COLS_H_MAT;
        uint32_t iRow       = matElemIdx / N_COLS_H_MAT;
        // Not all threads may participate in the last iteration
        if(matElemIdx < N_MAT_ELEMS_TO_RD)
        {
            bfwCoefCompKernelBfLayerPrm_t const& bfLayerPrm = pBfLayerPrm[iRow];
            uint32_t iSrcPrbGrp = bfLayerPrm.startPrbGrpOffset + iPrbGrp;
            uint8_t iSrcRow     = bfLayerPrm.ueLayerIdx;

            // Gather SRS channel estimates from different layers of potentially different UEs
            tensor_ref<const TComplexStorageIn> tSrsChEst(bfLayerPrm.tInfoSrsChEst.pAddr, bfLayerPrm.tInfoSrsChEst.strides); // (N_PRB_GRP, N_GNB_ANTS, N_LAYERS)
    
            shSrsChEst(iRow, iCol) =
                type_convert<TComplexCompute>(tSrsChEst(iSrcPrbGrp, iCol, iSrcRow));

#ifdef ENABLE_DEBUG
            printf("Mat[%d][%d][%d] = %f+j%f\n", iPrbGrp, iRow, iCol, shSrsChEst(iRow, iCol).x, shSrsChEst(iRow, iCol).y);
#endif
        }
    }
}

// BeamFormingWeight (BFW) coefficient computation kernel
// {N_LAYERS, N_BS_ANTS} = {16,64}
// Inputs and outputs assumed to be column major
// dimBlock: (N_THREADS_PER_WARP, N_LAYERS)
// dimGrid : (Nprb)
template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS, // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS,  // # of layers (# of cols in H matrix)
          uint32_t N_THRD_GRPS_PER_THRD_BLK,
          uint32_t N_THRDS_PER_GRP>
__global__ void
bfwMmseCoefCompKernel_v1(bfwCoefCompStatDescr_t* pStatDescr, bfwCoefCompDynDescr_t* pDynDescr)
{
    //--------------------------------------------------------------------------------------------------------
    // Setup local parameters based on descriptor
    bfwCoefCompStatDescr_t& statDescr = *(pStatDescr);

    // Early exit check
    // The grid is sized to process the max # of PRBs in a given heterogenous config. Exit if the PRB to be
    // processed by this thread block does not exist in the UE group
    // PRB index processed by this thread
    const uint32_t PRB_GRP_IDX = blockIdx.x;
    const uint32_t UE_GRP_IDX  = statDescr.pHetCfgUeGrpMap[pDynDescr->hetCfgIdx][blockIdx.y];

    bfwCoefCompKernelUeGrpPrm_t& ueGrpPrms = statDescr.pKernelUeGrpPrms[UE_GRP_IDX];
    bfwCoefCompKernelBfLayerPrm_t* pBfLayerPrms = ueGrpPrms.pBfLayerPrmGpu;

    const uint16_t nPrbGrp = ueGrpPrms.nPrbGrp;
    if(PRB_GRP_IDX >= nPrbGrp) return;

    //--------------------------------------------------------------------------------------------------------
    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    // clang-format off
    // tensor_ref<TComplexStorageOut>      tDbg (ueGrpPrms.tInfoDbg.pAddr     , ueGrpPrms.tInfoDbg.strides     );
#ifdef BFW_BOTH_COMP_FLOAT
    tensor_ref<TComplexStorageOut> tCoef(ueGrpPrms.tInfoBfwCoefs.pAddr, ueGrpPrms.tInfoBfwCoefs.strides); // (N_GNB_ANTS, N_LAYERS, N_PRB_GRP)
#endif
    // clang-format on

    TCompute lambda = statDescr.lambda;

    // H is channel matrix
    // G is the enhanced Gram matrix
    // A is the augmented matrix, A = [ G | I | H ]

    //--------------------------------------------------------------------------------------------------------
    // Dimensions

    // H  : Channel matrix
    constexpr uint32_t N_ROWS_H = N_LAYERS;
    constexpr uint32_t N_COLS_H = N_BS_ANTS;

    // R  : Diagonal matrix (lambda) with per layer regularization coefficients
    constexpr uint32_t N_ROWS_R = N_LAYERS;
    // constexpr uint32_t N_COLS_R = N_LAYERS;

    // G  : Enhanced Gram matrix, G = H*H' + R
    constexpr uint32_t N_ROWS_G = N_LAYERS;
    constexpr uint32_t N_COLS_G = N_LAYERS;

    // I  : Identity matrix
    constexpr uint32_t N_ROWS_I = N_LAYERS;
    constexpr uint32_t N_COLS_I = N_LAYERS;

    // Linv: inverse lower trianuglar matrix in LU factorization
    constexpr uint32_t N_ROWS_LINV = N_LAYERS;

    // U  : Upper triangular matrix
    constexpr uint32_t N_COLS_U = N_COLS_G;

    // C  : MMSE coefficient matrix, C = H'*Ginv = H'*inv(H*H' + D)
    constexpr uint32_t N_ROWS_C = N_COLS_H;
    constexpr uint32_t N_COLS_C = N_COLS_G;

    // A  : Augmented result matrix, A = [ G | I | H ] -> [ U | Linv | F ]
    constexpr uint32_t N_ROWS_A = N_ROWS_G;
    constexpr uint32_t N_COLS_A = N_COLS_G + N_COLS_I + N_COLS_H;

    static_assert((N_THRDS_PER_GRP <= N_THREADS_PER_WARP), "Using co-operative groups");
    static_assert((0 == N_BS_ANTS % N_THRDS_PER_GRP) && (N_BS_ANTS >= N_THRDS_PER_GRP), "Expect BS antenna to be a multiple of thread group size");
    static_assert(N_THRDS_PER_GRP >= N_LAYERS, "number of threads per group must be more than number of layers");

    thread_block const& thisThrdBlk = this_thread_block();

    // Co-operative thread groups used in computation of inner products
    thread_block_tile<N_THRDS_PER_GRP> const& thrdGrp = tiled_partition<N_THRDS_PER_GRP>(thisThrdBlk);

    // G is Hermitian symmetric i.e. only the upper or lower diagonal elements need to be computed
    constexpr uint32_t N_TRI_ELEMS_G = N_ROWS_G * (N_ROWS_G + 1) / 2;

    // Iterations to compute one element of G. Each thread group computes the inner product needed to produce
    // one element of G
    constexpr uint32_t N_INNER_ITER_TO_COMP_G_ELEM = div_round_up(N_COLS_H, N_THRDS_PER_GRP);

    // Each thread group computes one element of G per outer loop iteration
    constexpr uint32_t N_OUTER_ITER_TO_COMP_G = div_round_up(N_TRI_ELEMS_G, N_THRD_GRPS_PER_THRD_BLK);

    //--------------------------------------------------------------------------------------------------------
    // Compute indices used for element access
    const uint32_t THRD_IDX     = threadIdx.x; // thrdGrp.thread_rank()
    const uint32_t THRD_GRP_IDX = threadIdx.y;
    const uint32_t THRD_ABS_IDX = (threadIdx.y * blockDim.x) + threadIdx.x;

    const uint32_t PRB_IDX = blockIdx.x;

    const uint32_t ROW_IDX_R = THRD_ABS_IDX % N_ROWS_R;

    const uint32_t ROW_IDX_I = THRD_ABS_IDX % N_ROWS_I;
    const uint32_t COL_IDX_I = THRD_ABS_IDX / N_ROWS_I;

    //--------------------------------------------------------------------------------------------------------
    // Shared memory allocation
    // H[N_TONES_PER_ITER*N_INST]

    // Shared memory contents as processing progresses:
    // A = [ G | I | H ] -> [ U | Linv | F ]

    constexpr uint32_t N_SMEM_R_ELEMS = N_ROWS_R;
    constexpr uint32_t N_SMEM_A_ELEMS = (N_ROWS_A + 1) * N_COLS_A; // (N_ROWS_A + 1) for SMEM padding to avoid bank conflicts
    constexpr uint32_t N_SMEM_C_ELEMS = (N_ROWS_C + 1) * N_COLS_C; // (N_ROWS_C + 1) for SMEM padding to avoid bank conflicts

    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    __shared__ TCompute smemBlkR[N_SMEM_R_ELEMS];
    __shared__ TComplexCompute smemBlkA[N_SMEM_A_ELEMS];
    __shared__ TComplexCompute smemBlkC[N_SMEM_C_ELEMS];
    __shared__ TCompute shCFrobeniusNorm;

    constexpr uint32_t            SMEM_START_OFFSET_R = 0;
    block_1D<TCompute*, N_ROWS_R> shR(&smemBlkR[SMEM_START_OFFSET_R]);

    constexpr uint32_t                                 SMEM_START_OFFSET_A = 0;
    block_2D<TComplexCompute*, N_ROWS_A + 1, N_COLS_A> shA(&smemBlkA[SMEM_START_OFFSET_A]);

    // SMEM overlay: A with [ G | I | H ]
    const uint32_t                                     SMEM_START_OFFSET_G = SMEM_START_OFFSET_A;
    block_2D<TComplexCompute*, N_ROWS_G + 1, N_COLS_G> shG(&smemBlkA[SMEM_START_OFFSET_G]);

    const uint32_t                                     SMEM_START_OFFSET_I = SMEM_START_OFFSET_G + shG.num_elem();
    block_2D<TComplexCompute*, N_ROWS_I + 1, N_COLS_I> shI(&smemBlkA[SMEM_START_OFFSET_I]);

    const uint32_t                                     SMEM_START_OFFSET_H = SMEM_START_OFFSET_I + shI.num_elem();
    block_2D<TComplexCompute*, N_ROWS_H + 1, N_COLS_H> shH(&smemBlkA[SMEM_START_OFFSET_H]);

    const uint32_t                                     SMEM_START_OFFSET_C = 0;
    block_2D<TComplexCompute*, N_ROWS_C + 1, N_COLS_C> shC(&smemBlkC[SMEM_START_OFFSET_C]);

    // SMEM overlay:
    // After LU - U replaces G, Linv replaces I and F replaces H
    auto& shU    = shG;
    auto& shLinv = shI;
    auto& shF    = shH;

    // Dinv overlays with R
    auto& shDinv = shR;

    //--------------------------------------------------------------------------------------------------------
    // Stage1: Load inputs

#ifdef ENABLE_DEBUG
    if(0 != blockIdx.x) return;
#endif

    srsChEstLoadRowMjr<TStorageIn, TCompute, N_ROWS_H, N_COLS_H>(thisThrdBlk, PRB_IDX, pBfLayerPrms, shH);

    if(THRD_ABS_IDX < N_ROWS_R)
    {
        shR(ROW_IDX_R) = lambda;
    }

    // Wait for loads to complete. Thread(s) processing an entry of H may not be the same ones loading it
    thisThrdBlk.sync();

#ifdef ENABLE_DEBUG
    // H
    for(uint32_t i = 0; i < N_ROWS_H; ++i)
    {
        if(THRD_ABS_IDX < N_COLS_H)
            tDbg(i, THRD_ABS_IDX, PRB_IDX) = type_convert<TComplexStorageOut>(shH(i, THRD_ABS_IDX));
    }
#endif

    //---------------------------------------------------------------------------------------------------
    // Stage1: Compute the enhanced Gram matrix: G = (H*H' + R),  G - N_LAYERS x N_LAYERS

    if ( N_LAYERS < 4 ) {

        // For the case of N_LAYERS < 4, where the max number of independent elements of G is 6, it makes sense to use
        // warp-reductions to compute the elements of G.  These involve synchronizations at every step of the
        // reduction, but there are not enough elements of G to parallelize over otherwise.
        
        // Note that the parallel reduction naturally preserves the accuracy of the sum relatively well.
        
        uint32_t matGRowEndMrkr = N_COLS_G;
        uint32_t iGRow          = 0;
        uint32_t iGCol          = 0;
        uint32_t iGIdx          = 0;
        
        for(uint32_t i = 0; i < N_OUTER_ITER_TO_COMP_G; ++i)
        {
            // linear index, each thread group computes one element of G per outer loop iteration
            iGIdx = (i * N_THRD_GRPS_PER_THRD_BLK) + THRD_GRP_IDX;
            
            if(iGIdx >= N_TRI_ELEMS_G) break;
            
            // Since G is Hermitian, its sufficient if only the upper (or lower) triangular elements of G are computed.
            // Convert linear index to row and column indices of the upper triangular elements of matrix G
            while((iGIdx + iGRow) >= matGRowEndMrkr)
            {
             	matGRowEndMrkr += (N_COLS_G - iGRow);
                ++iGRow;
            }
            iGCol = N_COLS_G - (matGRowEndMrkr - iGIdx) + iGRow;

            // Compute G(iGRow,iGCol) via N_BS_ANTS x N_BS_ANTS inner product
            TComplexCompute G = cuGet<TComplexCompute>(0);
            for(uint32_t j = 0; j < N_INNER_ITER_TO_COMP_G_ELEM; ++j)
            {
             	uint32_t        iElem = (j * N_THRDS_PER_GRP) + THRD_IDX;
                TComplexCompute prod  = shH(iGRow, iElem) * cuConj(shH(iGCol, iElem));
                G += thrdGrpAllReduceSum<N_THRDS_PER_GRP>(thrdGrp, prod);
            }

            if(0 == THRD_IDX)
            {
                if(iGRow != iGCol)
                {
                    shG(iGCol, iGRow) = cuConj(G);
                }
                else
                {
                    G.x += shR(iGRow);
                }
                shG(iGRow, iGCol) = G;
#ifdef ENABLE_DEBUG                
                printf("G[%d][%d] = %f+j%f, linIdx %d, threadIdx (%d,%d), blockIdx.x %d, matGRowEndMrkr %d\n", iGRow, iGCol, cuReal(G), cuImag(G), iGIdx, threadIdx.x, threadIdx.y, blockIdx.x, matGRowEndMrkr);
#endif                
            }
        }
        
    }
    else {

        // For the case of N_LAYERS >= 4, there are sufficient independent elements (min 10) of G to parallelize over.  As well, multiple threads can be
        // assigned to an element of G.  The final reduction is performed using atomicAdd in shared, which is relatively slow.  So the max # of threads 
        // pere element is capped.  However, in practice, N_LAYERS == 8 uses only 3 'batches', so there are relatively few atomicAdds performed.

        // The serial sum (in this case) has poor accuracy properties relative to the parallel reduction (for the case of N_LAYERS <4).  For the case of
        // fp32, the SNR were, on average -0.578 dB lower than reference.  
        //      cuphy_ex_bfc -i .../BFW/TV_list_bfw_8cell_3slot_32Rx.yaml -r 1 -c -m 1
        // To remedy this, a version was written which would perform the accumulation in double-precision.  This resulted in an average SNR of +0.095 dB.
        // However, the fp64 version required an additional 8 registers.  As that seems more important at this time than -.578 dB, the fp32 version is
        // used here.

        // The fp64 version actually faster for some kernels.  This might be because the greater # of registers results in less occupancy and better
        // l1, l2 hit rates?  Retained as comment for future evaluation.

        TComplexCompute g;
        
        // Compute the batches.
        constexpr uint32_t HALF_G_ELEMS = (N_LAYERS*N_LAYERS + N_LAYERS)/2;         // lower triangular, including diagonal.
        constexpr uint32_t N_THREADS = N_THRDS_PER_GRP*N_THRD_GRPS_PER_THRD_BLK;
        constexpr uint32_t N_BATCHES1 = N_THREADS / HALF_G_ELEMS;
        const uint32_t N_BATCHES = N_BATCHES1 <= 8 ? N_BATCHES1 : 8;
        const uint32_t BATCH_SIZE = (N_BS_ANTS+N_BATCHES-1)/N_BATCHES;
        const uint32_t batch = THRD_ABS_IDX/HALF_G_ELEMS;

        // Assign each thread to a cell of G.
        const uint32_t i = THRD_ABS_IDX - batch * HALF_G_ELEMS;
        const uint32_t irow = (sqrtf(1+8*i)-1)/2;
        const uint32_t icol = i-(irow*irow+irow)/2;
        uint32_t kstart = batch * BATCH_SIZE;
        uint32_t kend = (batch+1)*BATCH_SIZE;
        if ( batch == N_BATCHES-1 ) kend = N_BS_ANTS;
        if ( batch >= N_BATCHES ) {
            kstart = 0;
            kend = 0;
        }

        // use batch zero to initialize lower triangular to zero.
        if ( batch == 0 ) {
            shG(irow,icol) = cuGet<TComplexCompute>(0);
        }

        // Sync threads to ensure shG is initialized to zero before atomic adds
        thisThrdBlk.sync();

        // Perform the GEMM segment
        if ( kend > 0 ) {
            
            /*

              // Double-precision accumulation
              // saved just in the event it is useful in the future

                cuDoubleComplex G1, G2, G;       
                TComplexCompute g1, g2;
                G.x = 0.;
                G.y = 0.;

                for ( int k=kstart; k<kend; ++k ) {
                    g1 = shH(irow,k);
                    g2 = shH(icol,k);
                    
                    // Accumulate in double-precision.
                    G1.x = g1.x;
                    G1.y = g1.y;
                    G2.x = g2.x;
                    G2.y = -g2.y;
                    G.x += (G1.x * G2.x - G1.y * G2.y);
                    G.y += (G1.x * G2.y + G1.y * G2.x);
                }
                
                g1.x = (float) G.x;
                g1.y = (float) G.y;
                
                atomicAdd( &(shG(irow,icol).x), g1.x);
                atomicAdd( &(shG(irow,icol).y), g1.y);
            */
            
            // Initialize register used for accumulation to zero.
            g = cuGet<TComplexCompute>(0);
            
            // Segment of inner product.
            for ( int k=kstart; k<kend; ++k ) {
                g = cuFma ( shH(irow,k), cuConj(shH(icol,k)), g );
            }
            // Atomic accumulation in SMEM
            atomicAdd( &(shG(irow,icol).x), g.x);
            atomicAdd( &(shG(irow,icol).y), g.y);
            
        }

        // ensure all atomicAdds have completed before reflecting the complex conjgate
        thisThrdBlk.sync();

        // use batch 0 to update the upper triangular
        if ( batch == 0 ) {

            if ( icol <= irow ) {

                g = shG(irow,icol);

                if ( irow != icol ) {
                    shG(icol, irow) = cuConj(g);
                }
                else {
                    g.x += shR(irow);
                    shG(irow,icol) = g;
                }

            }

        }

    }

    if(COL_IDX_I < N_COLS_I)
    {
        shI(ROW_IDX_I, COL_IDX_I) =
            (ROW_IDX_I != COL_IDX_I) ? cuGet<TComplexCompute>(0) : cuGet<TComplexCompute>(1);
    }

    // Wait for G matrix compute and I matrix init to complete
    thisThrdBlk.sync();

#ifdef ENABLE_DEBUG
    // A0
    for(uint32_t i = 0; i < N_ROWS_A; ++i)
    {
        if(THRD_ABS_IDX < N_COLS_A)
            tDbg(i, THRD_ABS_IDX, PRB_IDX) = type_convert<TComplexStorageOut>(shA(i, THRD_ABS_IDX));
    }
#endif

    //---------------------------------------------------------------------------------------------------
    // Stage2: Perform joint LU factorization
    // A = [ G | I | H ] -> [ U | Linv | F ]
    // where U = L\G, Linv = L\I, F = L\H

    // bfwMmseCoefCompKernel_v1: 
    // For Large layer count (e.g. 8, 16) thread block size >> # of columns of augmented matrix
    // (i.e. (N_THRDS_PER_GRP * N_LAYERS) >> (2*N_LAYERS + N_BS_ANTS)). Thus use parallel version of the
    // factorization algorithm to cut down iteration count and increase active threads during sub-matrix
    // updates
    // For small layer counts (e.g. 2, 4) thread block size >= # of columns of augmented matrix. Use iterative
    // version since the iteration count = N_ROWS_A = N_LAYERS is expected to be small and thread block size is 
    // not large relative to N_COLS_A


    /*

      Experimentation showed the luFactorizeParallel to never outperform luFactorizeIter.
      This had a big impact on performance.

    ((2 != N_LAYERS) && (4 != N_LAYERS)) ? luFactorizeParallel_v1<TCompute, N_ROWS_A, N_COLS_A>(thisThrdBlk, shA) : 
                                           luFactorizeIter<TCompute, N_ROWS_A, N_COLS_A>(thisThrdBlk, shA);
    */
    
    luFactorizeIter<TCompute, N_ROWS_A, N_COLS_A>(thisThrdBlk, shA);

#ifdef ENABLE_DEBUG
    // A1
    for(uint32_t i = 0; i < N_ROWS_A; ++i)
    {
        if(THRD_ABS_IDX < N_COLS_A)
            tDbg(i, THRD_ABS_IDX, PRB_IDX) = type_convert<TComplexStorageOut>(shA(i, THRD_ABS_IDX));
    }
#endif

    //---------------------------------------------------------------------------------------------------
    // Stage3: Multiply C = F'*(inv(D)*inv(L)), where D = I*(diag(U)), G - N_BS_ANTS x N_LAYERS


    // Compute inv(D)
    // Investigated applying shDiv to shLinv here vs. in the inv(D)*inv(L) loop.  Not exactly clear why, but his is faster.
    if(THRD_ABS_IDX < N_COLS_U)
    {
        shDinv(THRD_ABS_IDX) = cuGet<TCompute>(1) / cuReal(shU(THRD_ABS_IDX, THRD_ABS_IDX));
    }

    // Initialize matrix C Frobenius norm. Use a thread which was not used in above
    if(N_COLS_U == THRD_ABS_IDX)
    {
        shCFrobeniusNorm = cuGet<TCompute>(0);
    }
    
    thisThrdBlk.sync();

    TCompute absCSqr = cuGet<TCompute>(0);
      
    // the previous implementation of inv(D)*inv(L) unnecessarily repeated computations when gridDim.x was odd.
    // so this removes all of the inner and outer loops and replaces them with a block-stride loop.
    // This might be faster due to better resource usage and lower loop overhead.
    
    for ( uint32_t idx=THRD_ABS_IDX; idx<N_ROWS_C*N_COLS_C; idx+=blockDim.x*blockDim.y ) {

      // Compute the (row,col) for this index.
      uint32_t icol = idx/N_ROWS_C;
      uint32_t irow = idx - icol*N_ROWS_C;                              // avoid the use of %modulus
      
      TComplexCompute C = cuGet <TComplexCompute>(0);
      
      for ( uint32_t iElem=icol; iElem< N_ROWS_LINV; ++iElem ) {
	
	// Multiply inv(D)*inv(L)
	TComplexCompute DinvLinv = shDinv(iElem) * shLinv(iElem,icol);
	// Multiply F'*(inv(D)*inv(L))
	C = cuFma( cuConj(shF(iElem,irow)), DinvLinv, C );
	
      }
      
      // Store the computed element of C
      shC( irow, icol ) = C;
      
      // Collect inputs for the FrobeniusNorm in place - while we have C.
      // Sum here so we only need to do the thrdGrpAllReduceSum once.
      absCSqr += cuReal( cuConj(C)*C );
      
    }
      
    // Group (warp) - reduceSum absCSqr in place
    absCSqr = thrdGrpAllReduceSum<N_THRDS_PER_GRP>(thrdGrp,absCSqr);
    
    if ( 0 == THRD_IDX ) {
      // collect the inputs to the FrobeniusNorm from each group
      atomicAdd( &shCFrobeniusNorm, absCSqr );
    }
    
    // Wait for all the Frobenius terms to be collected
    thisThrdBlk.sync();
    
    // Compute the reciprocal of the Frobenius norm
    if(0 == THRD_ABS_IDX  ) {
      shCFrobeniusNorm = cuRSqrt<TCompute>( shCFrobeniusNorm );
    }
    
    // Wait for the reciprocal of the Frobenius norm to be computed
    thisThrdBlk.sync();

#ifdef ENABLE_DEBUG
    // Coefs pre-norm
    for(uint32_t i = 0; i < N_ROWS_C; ++i)
    {
        if(THRD_ABS_IDX < N_COLS_C)
            tDbg(i, THRD_ABS_IDX, PRB_IDX) = type_convert<TComplexStorageOut>(shC(i, THRD_ABS_IDX));
    }
#endif
    
    //--------------------------------------------------------------------------------------------------------
    // Stage4: Write the result BFC coefficients C into device memory
#ifdef BFW_BOTH_COMP_FLOAT
    cmplxMatStore<TStorageOut, TCompute, N_ROWS_C, N_COLS_C>(thisThrdBlk, PRB_IDX, shCFrobeniusNorm, shC, tCoef);
#endif

    /*compMatStore<TCompute,N_BS_ANTS,N_LAYERS,N_THREADS>(const typename complex_from_scalar<TCompute>::type* input,
                             uint8_t* output,
                             nPrbGrp,
                             float beta,
                             statDescr.compressBitwidth);*/

    // TODO parameterize beta based on compression bits   
    float beta           = 2048.0; // Scale factor for converting FP16 to integer
    uint8_t compressBits = statDescr.compressBitwidth;

    int32_t compbytes = (compressBits == 16) ? N_BS_ANTS * 4 : 2 * N_BS_ANTS / 8 * compressBits + 1;
    if(compressBits == 32)
    {
        compbytes = N_BS_ANTS * sizeof(TCompute);
        beta = 1.0f;
    }
    uint32_t output_index = PRB_GRP_IDX * compbytes; // (ANTENNAS, PRB_GRP, LAYERS)
    uint8_t* output = ueGrpPrms.pBfwCompCoef;
    constexpr uint32_t THRD_DIVS = N_BS_ANTS > 32 ? N_BS_ANTS : 32;
    constexpr uint32_t COMP_THRDS = THRD_DIVS*(N_THRDS_PER_GRP*N_THRD_GRPS_PER_THRD_BLK/THRD_DIVS);
    bfw_scale_compress_blockFP<TCompute, N_BS_ANTS + 1, N_BS_ANTS, N_LAYERS, COMP_THRDS>( 
        &smemBlkC[0],               // Shared memory input pointer for the antennas
        output + output_index,      // Output pointer for the first antenna
        //output,                     // Output pointer for the first antenna
        beta*shCFrobeniusNorm,      // Scaling factor
        compressBits,               // Number of compressed bits, if 16=uncompressed, 32=FP pass-through
        THRD_ABS_IDX,               // 1D thread rank
        nPrbGrp);                   // Stride between 2 layers (number of PRB groups)
        

#ifdef ENABLE_DEBUG
    // C
    for(uint32_t i = 0; i < N_ROWS_C; ++i)
    {
        if(THRD_ABS_IDX < N_COLS_C)
            printf("C[%d][%d][%d] = %f+j%f\n", PRB_IDX, i, COL_IDX_C, shC(i, COL_IDX_C).x, shC(i, COL_IDX_C).y);
    }
#endif
}

template <uint32_t N_BS_ANTS, // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS,  // # of layers (# of cols in H matrix)
          uint32_t N_THRD_GRPS_PER_THRD_BLK,
          uint32_t N_THRDS_PER_GRP>
void bfwCoefComp::bfwMmseCoefCompKernelLaunchGeo(uint16_t nMaxPrbGrp,
                                                 uint16_t nUeGrps,
                                                 dim3&    gridDim,
                                                 dim3&    blockDim)
{
    gridDim  = dim3(nMaxPrbGrp, nUeGrps);
    blockDim = dim3(N_THRDS_PER_GRP, N_THRD_GRPS_PER_THRD_BLK);

#ifdef ENABLE_DEBUG
    printf("blockDim (%d,%d,%d), gridDim (%d,%d,%d)\n", blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
#endif
}

template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS, // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS>  // # of layers (# of cols in H matrix)
void bfwCoefComp::bfwMmseCoefComp(bool                         getKernelFuncOnly,
                                  uint16_t                     nMaxPrbGrp,
                                  uint16_t                     nUeGrps,
                                  cuphyBfwCoefCompLaunchCfg_t& launchCfg)
{
    constexpr uint32_t N_THRDS_PER_GRP            = N_THREADS_PER_WARP;
    constexpr uint32_t N_THRD_GRPS_PER_THRD_BLK_1 = N_LAYERS/2; // Large layer count (e.g. 8,16)
    constexpr uint32_t N_THRD_GRPS_PER_THRD_BLK_2 = div_round_up(N_BS_ANTS+2*N_LAYERS, N_THRDS_PER_GRP); // Small layer count (e.g. 2,4)
    
    // original result for even layer counts - presume this provides best performance.
    constexpr uint32_t N_THRD_GRPS_PER_THRD_BLK_3 = (N_THRD_GRPS_PER_THRD_BLK_1 > N_THRD_GRPS_PER_THRD_BLK_2) ? N_THRD_GRPS_PER_THRD_BLK_1 : N_THRD_GRPS_PER_THRD_BLK_2;

    // odd layer counts must have N_LAYERS groups or code in bfwMmseCoefCompKernel_v1 'stage 3' will not work correctly.
    constexpr uint32_t N_THRD_GRPS_PER_THRD_BLK = (N_THRDS_PER_GRP%N_LAYERS==0) ? N_THRD_GRPS_PER_THRD_BLK_3 : N_LAYERS;

    void* kernelFunc = reinterpret_cast<void*>(bfwMmseCoefCompKernel_v1<TStorageIn,
                                                                        TStorageOut,
                                                                        TCompute,
                                                                        N_BS_ANTS,
                                                                        N_LAYERS,
                                                                        N_THRD_GRPS_PER_THRD_BLK,
                                                                        N_THRDS_PER_GRP>);

    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = launchCfg.kernelNodeParamsDriver;
    CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));

    if(! getKernelFuncOnly)
    {
        dim3 blockDim, gridDim;
        bfwMmseCoefCompKernelLaunchGeo<N_BS_ANTS, N_LAYERS, N_THRD_GRPS_PER_THRD_BLK, N_THRDS_PER_GRP>(nMaxPrbGrp, nUeGrps, gridDim, blockDim);
    
#if 0
        constexpr int32_t  N_ITER = N_THRD_BLK_TONES / N_TONES_PER_ITER;
        constexpr uint32_t N_INST = (1 == N_ITER) ? 1 : 2; // double buffering for pipelining
        constexpr uint32_t N_SMEM_ELEMS =
            (((N_BS_ANTS + 1) * N_LAYERS * N_INST) +
             ((N_LAYERS + 1) * (N_LAYERS + N_LAYERS + N_BS_ANTS))) *
                N_TONES_PER_ITER;
    
        int nShmemBytes    = N_SMEM_ELEMS * sizeof(TComplexCompute);
        int nMaxShmemBytes = nShmemBytes;
        cudaFuncSetAttribute(bfc_mmse_coef_comp_kernel_v1<TStorageIn, TStorageOut, TCompute, N_THRD_BLK_TONES, N_TONES_PER_ITER, N_BS_ANTS, N_LAYERS, NH>,
                             cudaFuncAttributeMaxDynamicSharedMemorySize,
                             nMaxShmemBytes);
#else
        int nShmemBytes = 0;
#endif
    
        kernelNodeParamsDriver.blockDimX = blockDim.x;
        kernelNodeParamsDriver.blockDimY = blockDim.y;
        kernelNodeParamsDriver.blockDimZ = blockDim.z;
        
        kernelNodeParamsDriver.gridDimX = gridDim.x;
        kernelNodeParamsDriver.gridDimY = gridDim.y;
        kernelNodeParamsDriver.gridDimZ = gridDim.z;
    
        kernelNodeParamsDriver.extra          = nullptr;
        kernelNodeParamsDriver.sharedMemBytes = nShmemBytes;    
    }
}

template <typename TStorageIn, typename TStorageOut, typename TCompute>
void bfwCoefComp::bfwCoefCompKernelSelL0(bool                         getKernelFuncOnly,
                                         uint16_t                     nMaxPrbGrp,
                                         uint16_t                     nUeGrps,
                                         uint16_t                     nRxAnts,
                                         uint8_t                      nLayers,
                                         cuphyBfwCoefCompLaunchCfg_t& launchCfg)
{
    if(64 == nRxAnts)
    {
        constexpr uint32_t N_BS_ANTS = 64; // # of BS antenna (# of rows in H matrix)
        switch(nLayers)
        {
            // nLayers == 16
            case 16:
            {
                constexpr uint32_t N_LAYERS = 16; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 15
            case 15:
            {
                constexpr uint32_t N_LAYERS = 15; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 14
            case 14:
            {
                constexpr uint32_t N_LAYERS = 14; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 13
            case 13:
            {
                constexpr uint32_t N_LAYERS = 13; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 12
            case 12:
            {
                constexpr uint32_t N_LAYERS = 12; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 11
            case 11:
            {
                constexpr uint32_t N_LAYERS = 11; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 10
            case 10:
            {
                constexpr uint32_t N_LAYERS = 10; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 9
            case 9:
            {
                constexpr uint32_t N_LAYERS = 9; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 8
            case 8:
            {
                constexpr uint32_t N_LAYERS = 8; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }        
            // nLayers == 7
            case 7:
            {
                constexpr uint32_t N_LAYERS = 7; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }        
            // nLayers == 6
            case 6:
            {
                constexpr uint32_t N_LAYERS = 6; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }        
            // nLayers == 5
            case 5:
            {
                constexpr uint32_t N_LAYERS = 5; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }        
            // nLayers == 4
            case 4:
            {
                constexpr uint32_t N_LAYERS = 4; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 3
            case 3:
            {
                constexpr uint32_t N_LAYERS = 3; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 2
            case 2:
            {
                constexpr uint32_t N_LAYERS = 2; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }        
            // nLayers == 1
            case 1:
            {
                constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }        
            default:
            {
                NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nRxAnts {} nLayers {}", 
                           __FUNCTION__, nRxAnts, nLayers);
                break;
            }
        }
    }
    // nBSAnts = 32
    else if(32 == nRxAnts)
    {
        constexpr uint32_t N_BS_ANTS = 32; // # of BS antenna (# of rows in H matrix)
        switch(nLayers)
        {
            // nLayers == 8
            case 8:
            {
                constexpr uint32_t N_LAYERS = 8; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 7
            case 7:
            {
                constexpr uint32_t N_LAYERS = 7; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 6
            case 6:
            {
                constexpr uint32_t N_LAYERS = 6; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 5
            case 5:
            {
                constexpr uint32_t N_LAYERS = 5; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 4
            case 4:
            {
                constexpr uint32_t N_LAYERS = 4; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 3
            case 3:
            {
                constexpr uint32_t N_LAYERS = 3; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 2
            case 2:
            {
                constexpr uint32_t N_LAYERS = 2; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 1
            case 1:
            {
                constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
                bfwMmseCoefComp<TStorageIn, TStorageOut, TCompute, N_BS_ANTS, N_LAYERS>(getKernelFuncOnly, nMaxPrbGrp, nUeGrps, launchCfg);
                break;
            }        
            default:
            {
                NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nRxAnts {} nLayers {}", 
                           __FUNCTION__, nRxAnts, nLayers);
                break;
            }
        }
    }
    else
    {
        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nRxAnts {} nLayers {}", 
                   __FUNCTION__, nRxAnts, nLayers);
    }
}

void bfwCoefComp::bfwCoefCompKernelSelL1(bool                         getKernelFuncOnly,
                                         uint16_t                     nMaxPrbGrp,
                                         uint16_t                     nUeGrps,
                                         uint16_t                     nRxAnts,
                                         uint8_t                      nLayers,
                                         cuphyDataType_t              srsChEstType,
                                         cuphyDataType_t              lambdaType,
#ifdef BFW_BOTH_COMP_FLOAT
                                         cuphyDataType_t              coefType,
#endif
                                         cuphyBfwCoefCompLaunchCfg_t& launchCfg)
{
#ifdef ENABLE_DEBUG    
    NVLOGD_FMT(NVLOG_BFW, "{}:{} Begin",__FUNCTION__, __LINE__);
#endif
#ifndef BFW_BOTH_COMP_FLOAT
    cuphyDataType_t coefType = CUPHY_C_16F;
#endif
    using TCompute = float;
    if((CUPHY_C_32F == srsChEstType) && (CUPHY_R_32F == lambdaType))
    {
        using TStorageIn = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
        if(CUPHY_C_32F == coefType)
        {
            using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
            bfwCoefCompKernelSelL0<TStorageIn, TStorageOut, TCompute>(getKernelFuncOnly,
                                                                      nMaxPrbGrp,
                                                                      nUeGrps,
                                                                      nRxAnts,
                                                                      nLayers,
                                                                      launchCfg);
        }
        else if(CUPHY_C_16F == coefType)
        {
            using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
            bfwCoefCompKernelSelL0<TStorageIn, TStorageOut, TCompute>(getKernelFuncOnly,
                                                                      nMaxPrbGrp,
                                                                      nUeGrps,
                                                                      nRxAnts,
                                                                      nLayers,
                                                                      launchCfg);
        }
        else
        {
            NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}:{} No kernel available to launch with requested data type srsChEstType ({}) lambdaType ({}) coefType ({})", 
                       __FUNCTION__, __LINE__, srsChEstType, lambdaType, coefType);
        }
    }
    else if((CUPHY_C_16F == srsChEstType) && (CUPHY_R_16F == lambdaType) && (CUPHY_C_16F == coefType))
    {
        using TStorageIn  = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        bfwCoefCompKernelSelL0<TStorageIn, TStorageOut, TCompute>(getKernelFuncOnly,
                                                                  nMaxPrbGrp,
                                                                  nUeGrps,
                                                                  nRxAnts,
                                                                  nLayers,
                                                                  launchCfg);    
    }
    else
    {
        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "{}:{} No kernel available to launch with requested data type srsChEstType ({}) lambdaType ({}) coefType ({})", 
                   __FUNCTION__, __LINE__, srsChEstType, lambdaType, coefType);
    }
#ifdef ENABLE_DEBUG
    NVLOGD_FMT(NVLOG_BFW, "{}:{} done",__FUNCTION__, __LINE__);
#endif
}

void bfwCoefComp::getDescrInfo(uint16_t nMaxUeGrps,
                               uint16_t nMaxTotalLayers,
                               size_t&  statDescrSizeBytes,
                               size_t&  statDescrAlignBytes,
                               size_t&  dynDescrSizeBytes,
                               size_t&  dynDescrAlignBytes,
                               size_t&  hetCfgUeGrpMapSizeBytes,
                               size_t&  hetCfgUeGrpMapAlignBytes,
                               size_t&  ueGrpPrmsSizeBytes,
                               size_t&  ueGrpPrmsAlignBytes,
                               size_t&  bfLayerPrmsSizeBytes,
                               size_t&  bfLayerPrmsAlignBytes)
{
    // Calculate sizes for various descriptor types
    statDescrSizeBytes  = sizeof(bfwCoefCompStatDescr_t);
    statDescrAlignBytes = alignof(bfwCoefCompStatDescr_t);

    dynDescrSizeBytes  = sizeof(bfwCoefCompDynDescrArr_t);
    dynDescrAlignBytes = alignof(bfwCoefCompDynDescrArr_t);

    hetCfgUeGrpMapSizeBytes = sizeof(decltype(m_pHetCfgUeGrpMapCpu)) * CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS * nMaxUeGrps;
    hetCfgUeGrpMapAlignBytes = alignof(decltype(m_pHetCfgUeGrpMapCpu));

    // Per UE group parameter descriptors
    ueGrpPrmsSizeBytes  = sizeof(bfwCoefCompKernelUeGrpPrm_t) * nMaxUeGrps;
    ueGrpPrmsAlignBytes = alignof(bfwCoefCompKernelUeGrpPrm_t);

    // Per layer parameter descriptors
    bfLayerPrmsSizeBytes  = sizeof(bfwCoefCompKernelBfLayerPrm_t) * nMaxTotalLayers;
    bfLayerPrmsAlignBytes = alignof(bfwCoefCompKernelBfLayerPrm_t);

    // dynDescrSizeBytes  = hetCfgUeGrpMapSizeBytes + perLayerDynDescrSizeBytes + perUeGrpDynDescrSizeBytes + dynDescrSizeBytes;    
    // dynDescrAlignBytes = std::max({hetCfgUeGrpMapAlignBytes, perLayerDynDescrAlignBytes, perUeGrpDynDescrAlignBytes, dynDescrAlignBytes});
}

cuphyStatus_t bfwCoefComp::init(bool         enableCpuToGpuDescrAsyncCpy,
                                uint8_t      compressBitwidth,
                                float        lambda,
                                void*        pStatDescrCpu,
                                void*        pStatDescrGpu,
                                void*        pDynDescrsCpu,
                                void*        pDynDescrsGpu,
                                void*        pHetCfgUeGrpMapCpu,
                                void*        pHetCfgUeGrpMapGpu,
                                void*        pUeGrpPrmsCpu,
                                void*        pUeGrpPrmsGpu,
                                void*        pBfLayerPrmsCpu,
                                void*        pBfLayerPrmsGpu,
                                cudaStream_t strm)
{
    if(!pStatDescrCpu || !pStatDescrGpu || !pDynDescrsCpu || !pDynDescrsGpu || !pHetCfgUeGrpMapCpu || !pHetCfgUeGrpMapGpu ||
       !pUeGrpPrmsCpu || !pUeGrpPrmsGpu || !pBfLayerPrmsCpu || !pBfLayerPrmsGpu) 
       return CUPHY_STATUS_INVALID_ARGUMENT;
              
    m_pHetCfgUeGrpMapCpu = (static_cast<decltype(m_pHetCfgUeGrpMapCpu)>(pHetCfgUeGrpMapCpu)); 
    m_pHetCfgUeGrpMapGpu = (static_cast<decltype(m_pHetCfgUeGrpMapGpu)>(pHetCfgUeGrpMapGpu)); 

    // Note: could use std::span as std::span kernelUeGrpPrmSpanCpu{pUeGrpDynDescrsCpu, m_nMaxTotalLayers}; for bounds-safe access
    m_pKernelUeGrpPrmCpu = (static_cast<bfwCoefCompKernelUeGrpPrm_t*>(pUeGrpPrmsCpu));
    m_pKernelUeGrpPrmGpu = (static_cast<bfwCoefCompKernelUeGrpPrm_t*>(pUeGrpPrmsGpu));

    m_pKernelBfLayerPrmCpu = (static_cast<bfwCoefCompKernelBfLayerPrm_t*>(pBfLayerPrmsCpu));
    m_pKernelBfLayerPrmGpu = (static_cast<bfwCoefCompKernelBfLayerPrm_t*>(pBfLayerPrmsGpu));

    // Setup static descriptor
    m_pStatDescrCpu = static_cast<bfwCoefCompStatDescr_t*>(pStatDescrCpu);
    m_pStatDescrGpu = static_cast<bfwCoefCompStatDescr_t*>(pStatDescrGpu);
    bfwCoefCompStatDescr_t& statDescrCpu = *m_pStatDescrCpu;
    statDescrCpu.compressBitwidth   = (0==compressBitwidth) ? 32 : compressBitwidth;
    statDescrCpu.lambda             = lambda;
    statDescrCpu.pKernelUeGrpPrms   = m_pKernelUeGrpPrmGpu;
    // statDescrCpu.pKernelBfLayerPrms = m_pKernelBfLayerPrmGpu;

    for(uint32_t hetCfgIdx = 0; hetCfgIdx < CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS; ++hetCfgIdx)
    {
        m_pHetCfgUeGrpMapArr[hetCfgIdx] = &m_pHetCfgUeGrpMapCpu[hetCfgIdx*m_nMaxUeGrps];

        // Setup pointers to heterogenous config to UE group map statically (the map values however change dynamically)
        statDescrCpu.pHetCfgUeGrpMap[hetCfgIdx] = &m_pHetCfgUeGrpMapGpu[hetCfgIdx*m_nMaxUeGrps];

        m_coefCompKernelArgsArr[hetCfgIdx].pStatDescr = m_pStatDescrGpu;
    }

    if(enableCpuToGpuDescrAsyncCpy)
    {
        CUDA_CHECK(cudaMemcpyAsync(m_pStatDescrGpu, m_pStatDescrCpu, sizeof(bfwCoefCompStatDescr_t), cudaMemcpyHostToDevice, strm));
    }

    // Save pointers to dynamic descriptors
    m_pDynDescrCpu = (static_cast<bfwCoefCompDynDescr_t*>(pDynDescrsCpu));
    m_pDynDescrGpu = (static_cast<bfwCoefCompDynDescr_t*>(pDynDescrsGpu));

    return CUPHY_STATUS_SUCCESS;
}

void bfwCoefComp::setupUeGrpDynDescr(cuphyBfwUeGrpPrm_t const&      ueGrpPrm,
                                     bfwCoefCompKernelUeGrpPrm_t&   kernelUeGrpPrm,
                                     bfwCoefCompKernelBfLayerPrm_t* pKernelLayerPrmCpu,
                                     bfwCoefCompKernelBfLayerPrm_t* pKernelLayerPrmGpu,
                                     cuphySrsChEstBuffInfo_t*       pChEstInfo,
#ifdef BFW_BOTH_COMP_FLOAT
                                     cuphyTensorPrm_t*              pTBfwCoef,
#endif
                                     uint8_t**                      pBfwCompCoef)
{   
    // Read UE group level parameters from API into descriptor
    kernelUeGrpPrm.nPrbGrp   = ueGrpPrm.nPrbGrp;
    kernelUeGrpPrm.nRxAnt    = ueGrpPrm.nRxAnt;
    kernelUeGrpPrm.nBfLayers = ueGrpPrm.nBfLayers;
#ifdef BFW_BOTH_COMP_FLOAT
    copyTensorPrm2Info(pTBfwCoef[ueGrpPrm.coefBufIdx], kernelUeGrpPrm.tInfoBfwCoefs);
#endif
    kernelUeGrpPrm.pBfwCompCoef = pBfwCompCoef[ueGrpPrm.coefBufIdx];

    // Setup per layer parameter in per UE group descriptor (bfwCoefCompKernelUeGrpPrm_t) to point to per layer CPU/GPU descriptors (bfwCoefCompKernelBfLayerPrm_t)
    kernelUeGrpPrm.pBfLayerPrmCpu = pKernelLayerPrmCpu;
    kernelUeGrpPrm.pBfLayerPrmGpu = pKernelLayerPrmGpu;

    // Beamforming layer level parameters from API into descriptor
    for(uint32_t layerIdx = 0; layerIdx < ueGrpPrm.nBfLayers; ++layerIdx)
    {
        // Copy per layer parameters into per layer CPU descriptor (bfwCoefCompKernelBfLayerPrm_t) which will then be copied to the GPU 
        // counterpart as part of bulk copy
        cuphyBfwLayerPrm_t const& bfLayerPrm            = ueGrpPrm.pBfLayerPrm[layerIdx];
        bfwCoefCompKernelBfLayerPrm_t& kernelBfLayerPrm = kernelUeGrpPrm.pBfLayerPrmCpu[layerIdx];
        kernelBfLayerPrm.ueLayerIdx                     = bfLayerPrm.ueLayerIndex;

        // Determine frequency (start PRB group) offset
        cuphySrsChEstBuffInfo_t const& chEstInfo = pChEstInfo[bfLayerPrm.chEstInfoBufIdx];        
        if(chEstInfo.startPrbGrp > ueGrpPrm.startPrbGrp)
        {
            throw std::runtime_error(std::string("bfwCoefComp::setupUeGrpDynDescr: SRS ChEst startPrb (" + std::to_string(chEstInfo.startPrbGrp) + ") beyond BFW startPrb (" + std::to_string(ueGrpPrm.startPrbGrp) + ")"));
        }
        kernelBfLayerPrm.startPrbGrpOffset = ueGrpPrm.startPrbGrp - chEstInfo.startPrbGrp;

        copyTensorPrm2Info(chEstInfo.tChEstBuffer, kernelBfLayerPrm.tInfoSrsChEst);
        // printf("layerIdx %d chEstInfoBufIdx %d ueLayerIdx %d elemType %d\n", layerIdx, bfLayerPrm.chEstInfoBufIdx, kernelBfLayerPrm.ueLayerIdx, kernelBfLayerPrm.tInfoSrsChEst.elemType);
    }
}

// Sweep through the UE groups, batch them into heterogenous configurations and setup kernel descriptors
void bfwCoefComp::setupAndBatchCoefComp(uint16_t                  nUeGrps,
                                        cuphyBfwUeGrpPrm_t const* pUeGrpPrms,
                                        uint32_t&                 nHetCfgs,
                                        cuphySrsChEstBuffInfo_t*  pChEstInfo,
#ifdef BFW_BOTH_COMP_FLOAT
                                        cuphyTensorPrm_t*         pTBfwCoef,
#endif
                                        uint8_t**                 pBfwCompCoef)
{
    //--------------------------------------------------------------------------------------------------------
    // Helper to find kernel function
    auto findKernelFunc = [](bfwCoefCompHetCfgArr_t const& hetCfgs, CUfunction func, int32_t& hetCfgIdx) {
        for(hetCfgIdx = 0; hetCfgIdx < hetCfgs.size(); ++hetCfgIdx)
        {
            // Check if kernel function is found
            if(func == hetCfgs[hetCfgIdx].func) break;

            // Check if no more kernel functions exist
            if(nullptr == hetCfgs[hetCfgIdx].func)
            {
                hetCfgIdx = -1;
                break;
            }
        }

        // Exhausted all heterogenous configs possible 
        if(hetCfgs.size() == hetCfgIdx) hetCfgIdx = -1;
    };

    //--------------------------------------------------------------------------------------------------------
    // Initialize the batch config data structure
    bfwCoefCompHetCfgArr_t& hetCfgs = m_coefCompHetCfgsArr;
    std::fill(hetCfgs.begin(), hetCfgs.end(), bfwCoefCompHetCfg_t{nullptr, 0, 0});

#ifdef ENABLE_DEBUG
    NVLOGD_FMT(NVLOG_BFW, "{}: # of UE groups {}",__FUNCTION__,nUeGrps);
#endif

    //--------------------------------------------------------------------------------------------------------
    // UE group sweep
    // Index into global descriptor pool of beamforming layers (across all UE groups)
    uint16_t globalBfLayerOffset = 0;
    nHetCfgs = 0;
    for(int32_t ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
    {
        //----------------------------------------------------------------------------------------------------
        // Absorb input API and setup kernel descriptor per UE group
        cuphyBfwUeGrpPrm_t const& ueGrpPrm = pUeGrpPrms[ueGrpIdx];
        bfwCoefCompKernelUeGrpPrm_t& kernelUeGrpPrm = m_pKernelUeGrpPrmCpu[ueGrpIdx];

        if((globalBfLayerOffset + ueGrpPrm.nBfLayers) >= m_nMaxTotalLayers) throw std::runtime_error(std::string("bfwCoefComp::setupAndBatchCoefComp: Exceeded limit (" + std::to_string(m_nMaxTotalLayers) + ") on total number of layers"));

        setupUeGrpDynDescr(ueGrpPrm, 
                           kernelUeGrpPrm,
                           &m_pKernelBfLayerPrmCpu[globalBfLayerOffset], 
                           &m_pKernelBfLayerPrmGpu[globalBfLayerOffset],
                           pChEstInfo,
#ifdef BFW_BOTH_COMP_FLOAT
                           pTBfwCoef,
#endif
                           pBfwCompCoef);
        globalBfLayerOffset += ueGrpPrm.nBfLayers;

        //----------------------------------------------------------------------------------------------------
        // Batch UE group into heterogenous configurations
        cuphyBfwCoefCompLaunchCfg_t launchCfg;
        bool getKernelFuncOnly = true;
        bfwCoefCompKernelSelL1(getKernelFuncOnly,
                               0,
                               0,
                               kernelUeGrpPrm.nRxAnt,
                               kernelUeGrpPrm.nBfLayers,
                               kernelUeGrpPrm.pBfLayerPrmCpu[0].tInfoSrsChEst.elemType,
                               type_to_cuphy_type<decltype(m_pStatDescrCpu->lambda)>::value,
#ifdef BFW_BOTH_COMP_FLOAT
                               kernelUeGrpPrm.tInfoBfwCoefs.elemType,
#endif
                               launchCfg);

        // Check if the heterogenous configuration already exists
        int32_t hetCfgIdx = 0;
        findKernelFunc(hetCfgs, launchCfg.kernelNodeParamsDriver.func, hetCfgIdx);

        uint16_t nPrbGrp = ueGrpPrm.nPrbGrp;
        // If a heterogenous configuration already exists then increment the # of UE groups for that config
        if(-1 != hetCfgIdx)
        {
            bfwCoefCompHetCfg_t& hetCfg = hetCfgs[hetCfgIdx];
            if(hetCfg.nUeGrps >= m_nMaxUeGrps)
            {
                throw std::runtime_error(std::string("bfwCoefComp::batchCoefComp: Exceeded limit (" + std::to_string(m_nMaxUeGrps) + ") on supported UE groups"));
            }

            if(nPrbGrp > hetCfg.nMaxPrbGrp) hetCfg.nMaxPrbGrp = nPrbGrp;

            m_pHetCfgUeGrpMapArr[hetCfgIdx][hetCfg.nUeGrps] = ueGrpIdx;
            hetCfg.nUeGrps++;

#ifdef ENABLE_DEBUG
            printf("bfwCoefComp::batchCoefComp: UE group %02d -> HetCfg %d (nUeGrps %02d nPrbGrp %03d nMaxPrbGrp %03d nRxAnt %02d nLayers %02d)\n", ueGrpIdx, hetCfgIdx, hetCfg.nUeGrps, nPrbGrp, hetCfg.nMaxPrbGrp, ueGrpPrm.nRxAnt, ueGrpPrm.nBfLayers);
#endif
        }
        // New heterogenous configuration found
        else
        {
            if(nHetCfgs >= CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS)
            {
                throw std::runtime_error("bfwCoefComp::batchCoefComp: Exceeded limit (" + std::to_string(CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS) + ") on supported heterogneous configurations");
            }

            int32_t newHetCfgIdx        = nHetCfgs++;
            bfwCoefCompHetCfg_t& hetCfg = hetCfgs[newHetCfgIdx];
            hetCfg.func                 = launchCfg.kernelNodeParamsDriver.func;
            hetCfg.nMaxPrbGrp           = nPrbGrp;

            m_pHetCfgUeGrpMapArr[newHetCfgIdx][hetCfg.nUeGrps] = ueGrpIdx;
            hetCfg.nUeGrps++;

#ifdef ENABLE_DEBUG
            printf("bfwCoefComp::setupCoefComp: UE group %02d -> HetCfg %d (nUeGrps %02d nPrbGrp %03d nMaxPrbGrp %03d nRxAnt %02d nLayers %02d)\n", ueGrpIdx, newHetCfgIdx, hetCfg.nUeGrps, nPrbGrp, hetCfg.nMaxPrbGrp, ueGrpPrm.nRxAnt, ueGrpPrm.nBfLayers);
#endif
        }
    }
}

cuphyStatus_t bfwCoefComp::setupCoefComp(uint16_t                      nUeGrps,
                                         cuphyBfwUeGrpPrm_t const*     pUeGrpPrms,
                                         bool                          enableCpuToGpuDescrAsyncCpy,
                                         cuphySrsChEstBuffInfo_t*      pChEstInfo,
#ifdef BFW_BOTH_COMP_FLOAT
                                         cuphyTensorPrm_t*             pTBfwCoef,
#endif
                                         uint8_t**                     pBfwCompCoef,
                                         cuphyBfwCoefCompLaunchCfgs_t* pLaunchCfgs,
                                         cudaStream_t                  strm)
{
    if(!pUeGrpPrms || !pChEstInfo || !pBfwCompCoef || !pLaunchCfgs) return CUPHY_STATUS_INVALID_ARGUMENT;

    cuphyBfwCoefCompLaunchCfgs_t& launchCfgs = *pLaunchCfgs;
    setupAndBatchCoefComp(nUeGrps,
                          pUeGrpPrms,
                          launchCfgs.nCfgs,
                          pChEstInfo,
#ifdef BFW_BOTH_COMP_FLOAT
                          pTBfwCoef,
#endif
                          pBfwCompCoef);
    
    for(uint32_t hetCfgIdx = 0; hetCfgIdx < launchCfgs.nCfgs; ++hetCfgIdx)
    {
        // Skip rest of the setup if there are no UE groups corresponding to the channel equalizer instance and hetCfg
        if(0 == m_coefCompHetCfgsArr[hetCfgIdx].nUeGrps) continue;
        
        bfwCoefCompDynDescr_t& dynDescr = m_pDynDescrCpu[hetCfgIdx];
        dynDescr.hetCfgIdx = hetCfgIdx;

        bfwCoefCompHetCfg_t const& hetCfg   = m_coefCompHetCfgsArr[hetCfgIdx];
        bfwCoefCompKernelArgs_t& kernelArgs = m_coefCompKernelArgsArr[hetCfgIdx];

        // Select kernel
        cuphyBfwCoefCompLaunchCfg_t& launchCfg = launchCfgs.cfgs[hetCfgIdx];

        // All UE groups within the a heterogenous config have the same gNB antenna and layer config
        int32_t ueGrpIdx = m_pHetCfgUeGrpMapArr[hetCfgIdx][0];
        bfwCoefCompKernelUeGrpPrm_t& kernelUeGrpPrm = m_pKernelUeGrpPrmCpu[ueGrpIdx];
        bool getKernelFuncOnly = false;
        bfwCoefCompKernelSelL1(getKernelFuncOnly,
                               hetCfg.nMaxPrbGrp,
                               hetCfg.nUeGrps,
                               kernelUeGrpPrm.nRxAnt,
                               kernelUeGrpPrm.nBfLayers,
                               kernelUeGrpPrm.pBfLayerPrmCpu[0].tInfoSrsChEst.elemType,
                               type_to_cuphy_type<decltype(m_pStatDescrCpu->lambda)>::value,
#ifdef BFW_BOTH_COMP_FLOAT
                               kernelUeGrpPrm.tInfoBfwCoefs.elemType,
#endif
                               launchCfg);

        if(hetCfg.func != launchCfg.kernelNodeParamsDriver.func)
        {
           throw std::runtime_error("bfwCoefComp::setupCoefComp: Kernel function mismatch");
        }                                   

        kernelArgs.pDynDescr    = &m_pDynDescrGpu[hetCfgIdx];
        launchCfg.kernelArgs[0] = &kernelArgs.pStatDescr;
        launchCfg.kernelArgs[1] = &kernelArgs.pDynDescr;
        launchCfg.kernelNodeParamsDriver.kernelParams = &(launchCfg.kernelArgs[0]);
    }

    // Optional descriptor copy to GPU memory
    if(enableCpuToGpuDescrAsyncCpy)
    {
        CUDA_CHECK(cudaMemcpyAsync(m_pDynDescrGpu        , m_pDynDescrCpu        , sizeof(bfwCoefCompDynDescrArr_t)                                                      , cudaMemcpyHostToDevice, strm));
        CUDA_CHECK(cudaMemcpyAsync(m_pHetCfgUeGrpMapGpu  , m_pHetCfgUeGrpMapCpu  , sizeof(decltype(m_pHetCfgUeGrpMapCpu))*CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS*m_nMaxUeGrps, cudaMemcpyHostToDevice, strm));
        CUDA_CHECK(cudaMemcpyAsync(m_pKernelUeGrpPrmGpu  , m_pKernelUeGrpPrmCpu  , sizeof(bfwCoefCompKernelUeGrpPrm_t)*m_nMaxUeGrps                                      , cudaMemcpyHostToDevice, strm));
        CUDA_CHECK(cudaMemcpyAsync(m_pKernelBfLayerPrmGpu, m_pKernelBfLayerPrmCpu, sizeof(bfwCoefCompKernelBfLayerPrm_t)*m_nMaxTotalLayers                               , cudaMemcpyHostToDevice, strm));    
    }
    return CUPHY_STATUS_SUCCESS;    
}

} // namespace bfw_coefComp
