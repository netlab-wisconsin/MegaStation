/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

 
//#define CUPHY_DEBUG 1
#include "cuphy_kernel_util.cuh"

//#define FAST_COMPILE

#include <algorithm>
#include "cuComplex.h"
#include "cuda_fp16.h"
#include <cooperative_groups.h>
#include "channel_eq.hpp"
#include "type_convert.hpp"
#include <cstddef>
#include <vector>
#include <type_traits>
#include "cuphy_context.hpp"
#include "soft_demapper.cuh"
#include "nvlog.hpp"
#include "cuphy.hpp"
#include "math_utils.cuh"

#include <cufftdx.hpp>
using namespace cooperative_groups;

namespace channel_eq
{
// #define ENABLE_PROFILING
// #define ENABLE_DEBUG
// #define ENABLE_DEBUG_SOFT_DEMAP

#define ENABLE_MULTI_SYMBS_PER_THRD_BLK  (2)  // (0) single symbol per slot
                                              // (1) NUM_SYMBOLS_PER_SLOT per slot
                                              // (2) automatically select the smallest number of symbols per slot such that cta size is multiple of 32
#if(ENABLE_MULTI_SYMBS_PER_THRD_BLK == 1)
//NUM_SYMBOLS_PER_SLOT can be set to minimum 1 and maximum OFDM_SYMBOLS_PER_SLOT
#define NUM_SYMBOLS_PER_SLOT (OFDM_SYMBOLS_PER_SLOT)
#endif

#define LEGACY_LLR_SCALE
#define WRITE_EQ_OUTPUT
#define EQ_COEF_COMP_H_MIMO_VER (2)
// Set 0 to use the explicit demapper, 1 to use the texture based demapper, 2 to use the simplified soft demapper
// note: if EQ_SOFT_DEMAP_USE_TEX is set to 0, i.e. using legacy demapper, it works only for single symbol per slot (ENABLE_MULTI_SYMBS_PER_THRD_BLK = 0)
#define EQ_SOFT_DEMAP_USE_TEX 2
#define MAX_ERROR_PRECISION 10000

#undef LOOP_DRMS_EQ

#if (ENABLE_MULTI_SYMBS_PER_THRD_BLK != 0)
static constexpr uint32_t CUDA_MAX_N_THRDS_PER_BLK = 1024;
#endif
static constexpr uint32_t N_THREADS_PER_WARP       = 32; // cudaDeviceProp::warpSize;

#if !EQ_SOFT_DEMAP_USE_TEX
// FP16 - largest normal number
static constexpr float LLR_LOW_LIM  = -65504.0f;// std::numeric_limits<__half>::lowest();
static constexpr float LLR_HIGH_LIM =  65504.0f;// std::numeric_limits<__half>::max();
#endif

static constexpr float INV_ZF_REGULARIZER = 3981.071705534973f; // Inverse of zero-forcing regularizer. Equivalent to diagoanl MMSE with SNR = 10^(3.6)


#if CUPHY_DEBUG
__device__
float debug_LLR_get_elem(const float4& Llr, int idx)
{
    switch(idx)
    {
    default:
    case 0: return Llr.x;
    case 1: return Llr.y;
    case 2: return Llr.z;
    case 3: return Llr.w;
    }
}

__device__
float debug_LLR_get_elem(const float2& Llr, int idx)
{
    cuphy_i::word_t w0, w1;
    w0.f32 = Llr.x;
    w1.f32 = Llr.y;
    switch(idx)
    {
    default:
    case 0: return __low2float(w0.f16x2);
    case 1: return __high2float(w0.f16x2);
    case 2: return __low2float(w1.f16x2);
    case 3: return __high2float(w1.f16x2);
    }
}
#endif

template <typename TElem>
struct tensor_ref
{
    TElem*     addr;
    const int32_t* strides;

    CUDA_BOTH
    tensor_ref(void* pAddr, const int32_t* pStrides) :
        addr(static_cast<TElem*>(pAddr)),
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
    CUDA_BOTH TElem&       operator()(int i0)                                   { return *(addr + offset(i0));         }
    CUDA_BOTH TElem&       operator()(int i0, int i1)                           { return *(addr + offset(i0, i1));     }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2)                   { return *(addr + offset(i0, i1, i2)); }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2, int i3)           { return *(addr + offset(i0, i1, i2, i3)); }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2, int i3, int i4)   { return *(addr + offset(i0, i1, i2, i3, i4)); }

    CUDA_BOTH const TElem& operator()(int i0) const                         { return *(addr + offset(i0));         }
    CUDA_BOTH const TElem& operator()(int i0, int i1) const                 { return *(addr + offset(i0, i1));     }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2) const         { return *(addr + offset(i0, i1, i2)); }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3) const { return *(addr + offset(i0, i1, i2, i3)); }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3, int i4) const { return *(addr + offset(i0, i1, i2, i3, i4)); }
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
template<>            CUDA_BOTH_INLINE float     cuGet(int x) { return(float(x)); }
template<>            CUDA_BOTH_INLINE __half    cuGet(int x) { return(__half(x)); }
template<>            CUDA_BOTH_INLINE cuComplex cuGet(int x) { return(make_cuComplex(float(x), 0.0f)); }
template<>            CUDA_BOTH_INLINE __half2   cuGet(int x) { return(make_half2(__half(x), 0.0));}

template <typename T> CUDA_BOTH_INLINE T         cuGet(float);
template<>            CUDA_BOTH_INLINE float     cuGet(float x) { return(float(x)); }
template<>            CUDA_BOTH_INLINE cuComplex cuGet(float x) { return(make_cuComplex(x, 0.0f)); }
template<>            CUDA_BOTH_INLINE __half2   cuGet(float x) { return(make_half2(x, 0.0f)); }

template <typename T> CUDA_BOTH_INLINE T         cuAbs(T);
template<>            CUDA_BOTH_INLINE float     cuAbs(float x) { return(fabsf(x)); }

static CUDA_BOTH_INLINE float     cuReal(cuComplex x)                   { return(cuCrealf(x)); }
static CUDA_BOTH_INLINE float     cuImag(cuComplex x)                   { return(cuCimagf(x)); }
static CUDA_BOTH_INLINE cuComplex cuConj(cuComplex x)                   { return(cuConjf(x)); }
static CUDA_BOTH_INLINE cuComplex operator-(cuComplex x, cuComplex y)   { return(cuCsubf(x, y)); }
static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, float y)       { return(make_cuComplex(cuCrealf(x)*y, cuCimagf(x)*y)); }
static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, cuComplex y)   { return(cuCmulf(x, y)); }
static CUDA_BOTH_INLINE cuComplex operator+=(cuComplex &x, float y)     { x = make_cuComplex(cuCrealf(x) + y, cuCimagf(x)); return x; }
static CUDA_BOTH_INLINE cuComplex operator+=(cuComplex &x, cuComplex y) { x = cuCaddf(x, y); return x; };
static CUDA_BOTH_INLINE cuComplex operator*=(cuComplex &x, float y)     { x = make_cuComplex(cuCrealf(x)*y, cuCimagf(x)*y); return x; }
static CUDA_BOTH_INLINE cuComplex operator*=(cuComplex &x, const cuComplex y) { x = cuCmulf(x, y); return x; };
static CUDA_BOTH_INLINE cuComplex cuFma(cuComplex x, cuComplex y, cuComplex a) { return cuCfmaf(x,y,a); }// a = (x*y) + a;

#ifndef FAST_COMPILE
// Arithmetic functions for half-precision complex
static CUDA_INLINE __half cuReal(__half2 x)                         { return x.x; }
static CUDA_INLINE __half cuImag(__half2 x)                         { return x.y; }
static CUDA_INLINE __half2 cuConj(__half2 x)                        { return make_half2(cuReal(x), -cuImag(x)); }
// static CUDA_INLINE cuComplex operator()(__half2 x)                  { return __half22float2(x); }
static CUDA_INLINE __half2 operator*(__half2 x, __half y)           { return __hmul2(x, make_half2(y, 1)); }
// static CUDA_INLINE cuComplex operator+=(cuComplex &x, __half y)     { x = make_cuComplex(cuCrealf(x) + y, cuCimagf(x)); return x; }
static CUDA_INLINE __half2 operator+=(__half2 &x, __half y)         { x = make_half2(cuReal(x) + y, cuImag(x)); return x; }
static CUDA_INLINE __half2 operator*=(__half2 &x, __half y)         { x = make_half2(cuReal(x)*y, cuReal(x)*y); return x; }
static CUDA_INLINE __half2 cuFma(__half2 x, __half2 y, __half2 a)   { return __hcmadd(x,y,a); }// a = (x*y) + a;
#endif

#if 0
static CUDA_BOTH_INLINE float cuCRmul(cuComplex x, cuComplex y) { return((cuCrealf(x) * cuCrealf(y)) - (cuCimagf(x) * cuCimagf(y))); }
static CUDA_BOTH_INLINE float cuCImul(cuComplex x, cuComplex y) { return((cuCrealf(x) * cuCimagf(y)) + (cuCimagf(x) * cuCrealf(y))); }
static CUDA_BOTH_INLINE cuComplex cuNeg(cuComplex x)                  { return(make_cuComplex(-cuCrealf(x), -cuCimagf(x))); }
static CUDA_BOTH_INLINE cuComplex cuAdd(cuComplex x, cuComplex y)       { return(cuCaddf(x, y)); }
static CUDA_BOTH_INLINE cuComplex cuMul(cuComplex x, cuComplex y)       { return(cuCmulf(x, y)); }
static CUDA_BOTH_INLINE cuComplex cuDiv(cuComplex x, cuComplex y)       { return(cuCdivf(x, y)); }
static CUDA_BOTH_INLINE cuComplex operator+(cuComplex x, cuComplex y)   { return(cuCaddf(x, y)); }

static CUDA_BOTH_INLINE cuComplex operator+=(cuComplex &x, cuComplex y)       { x = cuCaddf(x, y); return x; };
static CUDA_BOTH_INLINE cuComplex operator-=(cuComplex &x, cuComplex y)       { x = cuCsubf(x, y); return x; };
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

// Use tag dispatching to invoke different behaviours (different functions) for each QAM
template <QAM_t>
struct QAMEnumToTagMap;

// Only even QAMs supported by 3GPP
template <>
struct QAMEnumToTagMap<QAM_t::QAM_4>
{
    struct QAM4Tag
    {};
};
template <>
struct QAMEnumToTagMap<QAM_t::QAM_16>
{
    struct QAM16Tag
    {};
};
template <>
struct QAMEnumToTagMap<QAM_t::QAM_64>
{
    struct QAM64Tag
    {};
};
template <>
struct QAMEnumToTagMap<QAM_t::QAM_256>
{
    struct QAM256Tag
    {};
};



template <typename TStorageIn,
          typename TCompute,
          uint32_t N_ROWS_MAT,
          uint32_t N_COLS_MAT>
__device__ __forceinline__ void cmplxMatLoad(thread_block const&                                                                      thisThrdBlk,
                                             block_2D<const typename complex_from_scalar<TStorageIn>::type*, N_ROWS_MAT, N_COLS_MAT>& srcMat,
                                             block_2D<typename complex_from_scalar<TCompute>::type*, N_ROWS_MAT + 1, N_COLS_MAT>&     dstMat)
{
    typedef typename complex_from_scalar<TStorageIn>::type TComplexStorageIn;
    typedef typename complex_from_scalar<TCompute>::type   TComplexCompute;

    const uint32_t     N_THREADS               = thisThrdBlk.size();
    const uint32_t     THREAD_IDX              = thisThrdBlk.thread_rank();
    constexpr uint32_t N_MAT_ELEMS_TO_RD       = N_ROWS_MAT * N_COLS_MAT;
    const uint32_t     N_MAT_ELEMS_RD_PER_ITER = (N_MAT_ELEMS_TO_RD > N_THREADS) ? N_THREADS : N_MAT_ELEMS_TO_RD;
    const uint32_t     N_ITER_TO_RD_MAT        = div_round_up(N_MAT_ELEMS_TO_RD, N_MAT_ELEMS_RD_PER_ITER);

    for(uint32_t i = 0; i < N_ITER_TO_RD_MAT; ++i)
    {
        uint32_t matElemIdx = ((i * N_MAT_ELEMS_RD_PER_ITER) + THREAD_IDX);
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

    const uint32_t THREAD_X_IDX = threadIdx.x;

    // Iterate row by row of A applying Gaussian elimination. In each iteration Gaussian elimination
    // annihilates all elements of a column below main diagonal of G. In iteration k annihilate elements
    // G(k+1:n, k ). At the end of all iterations G is transformed to U
    // While transforming G to U, applying Gaussian elimination to other columns of A i.e. matrices I and M
    // produces matrices Linv and F respetively which can then be used to compute Ree and C via back
    // substitution
#pragma unroll
    for(int32_t k = 0; k < N_ROWS_MAT - 1; ++k)
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
            if((THREAD_X_IDX > k) && (THREAD_X_IDX < N_COLS_MAT))
            {
                matA(i, THREAD_X_IDX) = cuFma(Aik, matA(k, THREAD_X_IDX), matA(i, THREAD_X_IDX));

#ifdef ENABLE_DEBUG
                printf("A[%d][%d] = %f+j%f\n", i, THREAD_X_IDX, matA(i, THREAD_X_IDX).x, matA(i, THREAD_X_IDX).y);
#endif
            }

            // Ensure all threads (which may extend across multiple warps for nColsA > 32) have read and use shA(i,k) before writing into it
            thisThrdBlk.sync();
            if(0 == (THREAD_X_IDX))
            {
                matA(i, k) = Aik;
            }
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

    const uint32_t N_THREADS_X  = blockDim.x;
    const uint32_t THREAD_X_IDX = threadIdx.x;

    // Iterate row by row of A applying Gaussian elimination. In each iteration Gaussian elimination
    // annihilates all elements of a column below main diagonal of G. In iteration k annihilate elements
    // G(k+1:n, k ). At the end of all iterations G is transformed to U
    // While transforming G to U, applying Gaussian elimination to other columns of A i.e. matrices I and M
    // produces matrices Linv and F respetively which can then be used to compute Ree and C via back
    // substitution
#pragma unroll
    for(int32_t k = 0; k < N_ROWS_MAT - 1; ++k)
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
        uint32_t subMatColIdx         = THREAD_X_IDX % nColsSubMat;
        uint32_t matColIdx            = subMatStartColOffset + subMatColIdx; // process columns > k, note: matrix is in column major layout

        // Ensure whole rows are updated at a time
        // Assumes N_THREADS_X >= nColsSubMat
        uint32_t nRowsSubMatPerIter = N_THREADS_X / nColsSubMat;
        bool     thrdEnable         = (THREAD_X_IDX < (nRowsSubMatPerIter * nColsSubMat)); // Disable threads which don't update full rows
        uint32_t nIterToProcSubMat  = div_round_up(nRowsSubMat, nRowsSubMatPerIter);
        for(uint32_t i = 0; i < nIterToProcSubMat; ++i)
        {
            uint32_t subMatRowIdx = (i * nRowsSubMatPerIter) + (THREAD_X_IDX / nColsSubMat);
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
                // if((THREAD_X_IDX > k) && (THREAD_X_IDX < N_COLS_MAT))
                if(matColIdx < N_COLS_MAT)
                {
                    matA(matRowIdx, matColIdx) = cuFma(Aik, matA(k, matColIdx), matA(matRowIdx, matColIdx));

#ifdef ENABLE_DEBUG
                    printf("A[%d][%d] = %f+j%f\n", matRowIdx, matColIdx, matA(matRowIdx, matColIdx).x, matA(matRowIdx, matColIdx).y);
#endif
                }
            }

            // Ensure all threads (which may extend across multiple warps for nColsA > 32) have read and use shA(i,k) before writing into it
            thisThrdBlk.sync();
            if(thrdEnable && (matRowIdx < N_ROWS_MAT) && (subMatStartColOffset == matColIdx))
            {
                matA(matRowIdx, k) = Aik;
            }
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

    const uint32_t N_THREADS_X  = blockDim.x;
    const uint32_t THREAD_X_IDX = threadIdx.x;

    // Ensure whole rows are updated at a time
    // Assumes N_THREADS_X >= nColsSubMat
    uint32_t nRowsMatPerIter = N_THREADS_X / N_COLS_MAT;
    bool     thrdEnableMain  = (THREAD_X_IDX < (nRowsMatPerIter * N_COLS_MAT)); // Disable threads which don't update full rows
    uint32_t nIterToProcMat  = div_round_up(N_ROWS_MAT, nRowsMatPerIter);
    uint32_t matColIdx       = THREAD_X_IDX % N_COLS_MAT;
    uint32_t matRowOffset    = THREAD_X_IDX / N_COLS_MAT;

    // Iterate row by row of A applying Gaussian elimination. In each iteration Gaussian elimination
    // annihilates all elements of a column below main diagonal of G. In iteration k annihilate elements
    // G(k+1:n, k ). At the end of all iterations G is transformed to U
    // While transforming G to U, applying Gaussian elimination to other columns of A i.e. matrices I and M
    // produces matrices Linv and F respetively which can then be used to compute Ree and C via back
    // substitution
#pragma unroll
    for(int32_t k = 0; k < N_ROWS_MAT - 1; ++k)
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

            // Ensure all threads (which may extend across multiple warps for nColsA > 32) have read and use shA(i,k) before writing into it
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

template <typename TCompute,
          int32_t N_ROWS_U,
          int32_t N_COLS_U,
          int32_t N_ROWS_A,
          int32_t N_COLS_A>
__device__ __forceinline__ void backSub(thread_block const&                                                              thisThrdBlk,
                                        uint32_t                                                                         startCol,
                                        block_2D<typename complex_from_scalar<TCompute>::type*, N_ROWS_U + 1, N_COLS_U>& matU,
                                        block_2D<typename complex_from_scalar<TCompute>::type*, N_ROWS_A + 1, N_COLS_A>& matA)
{
    typedef typename complex_from_scalar<TCompute>::type TComplexCompute;

    const uint32_t THREAD_X_IDX = threadIdx.x;

    // Perform back substitution on last row first
    TCompute one_over_Uii = cuGet<TCompute>(1) / cuReal(matU(N_ROWS_U - 1, N_ROWS_U - 1));
    if((THREAD_X_IDX >= startCol) && (THREAD_X_IDX < N_COLS_A))
    {
        matA(N_ROWS_U - 1, THREAD_X_IDX) *= one_over_Uii;
    }

    // Now perform back substitution beginning from the row above last row and working upwards upto the first
    // row. Each thread solves for one column of the result (which are the columns of Ree and C)
#pragma unroll
    for(int32_t i = N_ROWS_U - 2; i >= 0; --i)
    {
        one_over_Uii = cuGet<TCompute>(1) / cuReal(matU(i, i));

        if((THREAD_X_IDX >= startCol) && (THREAD_X_IDX < N_COLS_A))
        {
            TComplexCompute sum = cuGet<TComplexCompute>(0);

#pragma unroll
            for(int32_t j = i + 1; j < N_ROWS_A; ++j)
            {
                sum = cuFma(matA(j, THREAD_X_IDX), matU(i, j), sum);
            }
            matA(i, THREAD_X_IDX) = (matA(i, THREAD_X_IDX) - sum) * one_over_Uii;
        }
    }

    thisThrdBlk.sync();
}

static constexpr uint32_t MAX_BITS_QAM = CUPHY_QAM_256;
// static constexpr uint32_t N_MAX_QAM_LLR   = MAX_BITS_QAM;
static constexpr uint32_t MAX_BITS_PAM = (MAX_BITS_QAM / 2);
// static constexpr uint32_t N_MAX_PAM_LLR   = (MAX_BITS_PAM/2);

#if !EQ_SOFT_DEMAP_USE_TEX
static constexpr uint32_t N_PAM_PER_QAM = (MAX_BITS_QAM / MAX_BITS_PAM);

__constant__ uint8_t LUT_PAM_OFFSET[] = {0, 1, 3, 6};
__constant__ float   LUT_SYMB_DIST_KPAM[] =
    {
        // PAM  OFFSET
        0.707106781186548f, //  1     0

        0.632455532033676f, //  2     1
        0.316227766016838f,

        0.617213399848368f, //  3     3
        0.308606699924184f,
        0.154303349962092f,

        0.613571991077897f, //  4     6
        0.306785995538948f,
        0.153392997769474f,
        0.076696498884737f};

template <typename TCompute,
          typename TStorageOut,
          typename TLlr>
__device__ __forceinline__ void computePamLlr(int32_t                                      nPamBits,
                                              uint32_t                                     iqIdx,
                                              TCompute                                     noiseInv,
                                              typename complex_from_scalar<TCompute>::type softEst,
                                              TCompute* __restrict__ pShWrkBuf,
                                              TLlr* __restrict__ pShLlr)
{
    constexpr uint32_t N_IQ = 2; // 2 samples: 1 I + 1 Q

    int32_t  pamIdx        = nPamBits - 1;
    int32_t  lastPamBitIdx = pamIdx;
    uint8_t  lutOffset     = LUT_PAM_OFFSET[pamIdx];
    TCompute dist          = LUT_SYMB_DIST_KPAM[lutOffset + lastPamBitIdx];

    // Inphase vs Quadrature branch
    TCompute pamSymb = (0 == iqIdx) ? cuReal(softEst) : cuImag(softEst);

    // Compute soft bits by soft slicing the received symbol and squared minimum distances of 2nd kind
    TCompute* pSoftBits = &pShWrkBuf[0];
    TCompute* pMinDist2 = &pShWrkBuf[MAX_BITS_PAM];

    pSoftBits[0] = pamSymb;

    uint8_t softBitSignBmsk = 0;
    for(int32_t i = 0; i < nPamBits - 1; ++i)
    {
        pSoftBits[i + 1] = LUT_SYMB_DIST_KPAM[lutOffset + i] - cuAbs(pSoftBits[i]);
        pMinDist2[i]     = dist + cuAbs(pSoftBits[i]);
        pMinDist2[i] *= pMinDist2[i];

        if(pSoftBits[i] < cuGet<TCompute>(0)) softBitSignBmsk |= (0x01 << i);

#ifdef ENABLE_DEBUG
        printf("computePamLLr: pamSymb = %f, pSoftBits[%d] = %f, pMinDist2[%d] = %f, dist = %f\n", pamSymb, i, pSoftBits[i], i, pMinDist2[i], dist);
#endif
    }
    pMinDist2[lastPamBitIdx] = dist + cuAbs(pSoftBits[lastPamBitIdx]);
    pMinDist2[lastPamBitIdx] *= pMinDist2[lastPamBitIdx];

    if(pSoftBits[lastPamBitIdx] < cuGet<TCompute>(0)) softBitSignBmsk |= (0x01 << lastPamBitIdx);

    // Minimum distance between received symbol and its closest constellation point (min distance of 1st kind)
    TCompute minDist1 = cuAbs(pSoftBits[lastPamBitIdx]) - dist;
    minDist1 *= minDist1;

    // Compute LLRs
    TStorageOut* pLlr = reinterpret_cast<TStorageOut*>(pShLlr);
    for(int32_t i = 0; i < nPamBits; ++i)
    {
        // noiseInv (i.e. ReeInv) is the inverse of QAM noise variance: noiseInv = 1/qamNoiseVariance
        // Since qamNoiseVariance = 2*pamNoiseVariance        
        // (minDist2 - minDist1)/(2*pamNoiseVariance) = (minDist2 - minDist1)/(qamNoiseVariance) = (minDist2 - minDist1)*noiseInv
        TCompute llr = ((pMinDist2[i] - minDist1) * noiseInv);
        // if(pSoftBits[i] < cuGet<TCompute>(0)) llr = -llr;
        if(softBitSignBmsk & (0x01 << i)) llr = -llr;

        uint32_t llrIdx = (i * N_IQ) + iqIdx;

        // Clamp LLR to max/min FP16 limits
        if(llr < LLR_LOW_LIM)  llr = __float2half_rz(LLR_LOW_LIM);
        if(llr > LLR_HIGH_LIM) llr = __float2half_rz(LLR_HIGH_LIM);
        pLlr[llrIdx] = llr;

#ifdef ENABLE_DEBUG
        printf("computePamLLr: llr = %f, pamSymb = %f, pSoftBits[%d] = %f, pMinDist2[%d] = %f, minDist1 = %f, N0Inv = %f\n", llr, pamSymb, i, pSoftBits[i], i, pMinDist2[i], minDist1, noiseInv);
#endif
    }
}
#endif

template <typename TComplexCompute,
          uint32_t THRD_GRP_SIZE>
__device__ __forceinline__
    TComplexCompute
    thrdGrpAllReduceSum(thread_block_tile<THRD_GRP_SIZE> const& thisThrdGrp, TComplexCompute const& val)
{
    uint32_t        thrdGrpSize = thisThrdGrp.size();
    TComplexCompute sum         = val;
    for(int32_t i = thrdGrpSize / 2; i > 0; i /= 2)
    {
        sum.x += thisThrdGrp.shfl_xor(cuReal(sum), i);
        sum.y += thisThrdGrp.shfl_xor(cuImag(sum), i);
    }
    thisThrdGrp.sync();
    return sum;
}

// Equalization kernel for high order MIMO and per PRB
// {N_LAYERS, N_BS_ANTS} = {8,16}, {16,16}
// Inputs and outputs assumed to be column major
// dimBlock: (N_BS_ANTS*N_LAYERS, N_TONES_PER_ITER)
// dimGrid : ((CUPHY_N_TONES_PER_PRB/N_THRD_BLK_TONES), Nprb)
template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS,        // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS,         // # of layers (# of cols in H matrix)  // used for static shared memory allocation
          uint32_t N_THRD_BLK_TONES, // # of frequency bins processed by a thread block
          uint32_t N_TONES_PER_ITER> // # of frequency bins processed in 1 iteration by the thread block
__device__ void
eqMmseCoefCompHighMimoKernel_v1(puschRxChEqStatDescr_t* pStatDescr, puschRxChEqCoefCompDynDescr_t* pDynDescr)
{
    //--------------------------------------------------------------------------------------------------------
    // Setup local parameters based on descriptor
    puschRxChEqStatDescr_t&        statDescr = *(pStatDescr);
    puschRxChEqCoefCompDynDescr_t& dynDescr  = *(pDynDescr);

    // Early exit check
    // The grid is sized to process the max # of PRBs in a given heterogenous config. Exit if the PRB to be
    // processed by this thread block does not exist in the UE group
    const uint32_t PRB_IDX = blockIdx.y;
    const uint32_t UE_GRP_IDX = dynDescr.hetCfgUeGrpMap[blockIdx.z];
    
    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];
    
    const uint16_t nPrb = drvdUeGrpPrms.nPrb;
    if(PRB_IDX >= nPrb) return;

    //--------------------------------------------------------------------------------------------------------
    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    // clang-format off
    tensor_ref<const TComplexStorageIn> tH         (drvdUeGrpPrms.tInfoHEst.pAddr       , drvdUeGrpPrms.tInfoHEst.strides       );// (N_BS_ANTS, N_LAYERS, NF, NH)
    tensor_ref<TStorageOut>             tReeDiagInv(drvdUeGrpPrms.tInfoReeDiagInv.pAddr , drvdUeGrpPrms.tInfoReeDiagInv.strides );// (N_SC, N_LAYERS, N_PRB, NH)   // (N_LAYERS, NF, NH)
    tensor_ref<TComplexStorageOut>      tCoef      (drvdUeGrpPrms.tInfoEqCoef.pAddr     , drvdUeGrpPrms.tInfoEqCoef.strides     );// (N_SC, N_LAYERS, N_BS_ANTS, N_PRB, NH) // (N_LAYERS, N_BS_ANTS, NF, NH)
    tensor_ref<TComplexStorageOut>      tDbg       (drvdUeGrpPrms.tInfoChEqDbg.pAddr    , drvdUeGrpPrms.tInfoChEqDbg.strides    );
    // clang-format on

    // set diagonal regularizer:
    bool   noiseRegFlag  = (drvdUeGrpPrms.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE) ? true : false;
    float  invDiagReg    = noiseRegFlag ? drvdUeGrpPrms.invNoiseVarLin : INV_ZF_REGULARIZER;

    //--------------------------------------------------------------------------------------------------------
    // Dimensions

    // H is channel matrix
    // G is the enhanced Gram matrix
    // A is the augmented matrix, A = [ G | I | M ]

    // H  : Channel matrix
    constexpr uint32_t N_ROWS_H = N_BS_ANTS;
    constexpr uint32_t N_COLS_H = N_LAYERS;

    // M  : Intermediate matrix, M = H'*RwwInv
    constexpr uint32_t N_ROWS_M = N_COLS_H;   // N_LAYERS
    constexpr uint32_t N_COLS_M = N_BS_ANTS; // N_BS_ANTS

    // G  : Enhanced Gram matrix, G = H'*RwwInv*H + inv(Rxx)
    constexpr uint32_t N_ROWS_G = N_LAYERS;
    constexpr uint32_t N_COLS_G = N_LAYERS;

    // U  : Upper triangular matrix
    constexpr uint32_t N_ROWS_U = N_ROWS_G;
    constexpr uint32_t N_COLS_U = N_COLS_G;

    // I  : Identity matrix, I = G*Ginv
    constexpr uint32_t N_ROWS_I = N_ROWS_G;
    constexpr uint32_t N_COLS_I = N_COLS_G;

    // Ree: Residual error covariance matrix, Ree = Ginv
    constexpr uint32_t N_ROWS_REE = N_ROWS_G;
    // constexpr uint32_t N_COLS_REE = N_COLS_G;

    // C  : MMSE coefficient matrix, C = Ree*H'*RwwInv = Ree*M
    constexpr uint32_t N_ROWS_C = N_ROWS_REE;
    constexpr uint32_t N_COLS_C = N_COLS_M;

    // A  : Augmented result matrix, A = [ G | I | M ] -> [ U | Linv | F ] -> [ U | Ginv | C ]
    constexpr uint32_t N_ROWS_A = N_ROWS_G;
    constexpr uint32_t N_COLS_A = N_COLS_G + N_COLS_I + N_COLS_M;

    // Column offsets to Ree (which is followed by C) matrix within the augmented matrix
    // which form the RHS of the back substition (and after back substitution the RHS is overwritten by the
    // unknowns being solved for)
    constexpr uint32_t START_COL_OFFSET_REE = N_COLS_U;

    constexpr int32_t N_ITER = N_THRD_BLK_TONES / N_TONES_PER_ITER;

    constexpr uint32_t N_INST = (1 == N_ITER) ? 1 : 2; // double buffering for pipelining

    // const uint32_t N_THREADS_X = blockDim.x; // Number of threads needed to process one frequency bin
    // const uint32_t N_THREADS = blockDim.x * blockDim.y; // N_TONES_PER_ITER == blockDim.y

    // Reciprocal of symbol energy
    const TCompute ES_INV = cuGet<TCompute>(1);

    //--------------------------------------------------------------------------------------------------------
    // Compute indices used for element access
    const uint32_t THREAD_X_IDX = threadIdx.x;
    // const uint32_t THREAD_IDX   = (threadIdx.y * blockDim.x) + threadIdx.x;

    // There are N_TONES_PER_ITER groups of threads with each group containing N_THREADS_X per group
    const uint32_t THRD_GRP_FREQ_OFFSET        = threadIdx.y;
    const uint32_t THRD_BLK_START_FREQ_OFFSET  = (blockIdx.x * N_THRD_BLK_TONES);
    const uint32_t THRD_GRP_START_FREQ_OFFSET  = THRD_BLK_START_FREQ_OFFSET + THRD_GRP_FREQ_OFFSET;
    const uint32_t THRD_BLK_ABS_START_FREQ_IDX = PRB_IDX * CUPHY_N_TONES_PER_PRB;

    // const uint32_t ROW_IDX_RWW = THREAD_IDX % N_ROWS_RWW;
    // const uint32_t COL_IDX_RWW = THREAD_IDX / N_ROWS_RWW; // COL_IDX_RWW needs a bounds check (since N_THREADS > # of Rww elements)

    const uint32_t ROW_IDX_H = THREAD_X_IDX % N_ROWS_H;
    const uint32_t COL_IDX_H = THREAD_X_IDX / N_ROWS_H;

    const uint32_t ROW_IDX_I = THREAD_X_IDX % N_ROWS_I;
    const uint32_t COL_IDX_I = THREAD_X_IDX / N_ROWS_I;

    const uint32_t ROW_IDX_M = THREAD_X_IDX % N_ROWS_M;
    const uint32_t COL_IDX_M = THREAD_X_IDX / N_ROWS_M;

    const uint32_t ROW_IDX_G = THREAD_X_IDX % N_ROWS_G;
    const uint32_t COL_IDX_G = THREAD_X_IDX / N_ROWS_G; // COL_IDX_G needs a bounds check (since N_THREADS_X > # of G elements)

    // const uint32_t ROW_IDX_REE = THREAD_X_IDX % N_ROWS_REE;
    // const uint32_t COL_IDX_REE = THREAD_X_IDX / N_ROWS_REE; // COL_IDX_REE needs a bounds check (since N_THREADS_X > # of Ree elements)

    const uint32_t ROW_IDX_C = THREAD_X_IDX % N_ROWS_C;
    const uint32_t COL_IDX_C = THREAD_X_IDX / N_ROWS_C;

    //--------------------------------------------------------------------------------------------------------
    // Shared memory allocation
    // H[N_TONES_PER_ITER*N_INST]

    // Shared memory contents as processing progresses:
    // A = [ G | I | M ] -> [ U | Linv | F ] -> [ U | Ree | C ]

    constexpr uint32_t N_SMEM_ELEMS =
        (((N_ROWS_H + 1) * N_COLS_H * N_INST) + // (N_ROWS_H + 1) for SMEM padding to avoid bank conflicts
         (N_ROWS_A + 1) * N_COLS_A) *
            N_TONES_PER_ITER;

    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    __shared__ TComplexCompute smemBlk[N_SMEM_ELEMS];

    constexpr uint32_t                                         SMEM_START_OFFSET_H_BLK = 0;
    const uint32_t                                             SMEM_START_OFFSET_H     = SMEM_START_OFFSET_H_BLK + THRD_GRP_FREQ_OFFSET * (N_ROWS_H + 1) * N_COLS_H * N_INST;
    block_3D<TComplexCompute*, N_ROWS_H + 1, N_COLS_H, N_INST> shH(&smemBlk[SMEM_START_OFFSET_H]);

    constexpr uint32_t                                 SMEM_START_OFFSET_A_BLK = SMEM_START_OFFSET_H_BLK + N_TONES_PER_ITER * shH.num_elem();
    const uint32_t                                     SMEM_START_OFFSET_A     = SMEM_START_OFFSET_A_BLK + THRD_GRP_FREQ_OFFSET * (N_ROWS_A + 1) * N_COLS_A;
    block_2D<TComplexCompute*, N_ROWS_A + 1, N_COLS_A> shA(&smemBlk[SMEM_START_OFFSET_A]);

    // SMEM overlay: A with [ G | I | M ]
    const uint32_t                                     SMEM_START_OFFSET_G = SMEM_START_OFFSET_A;
    block_2D<TComplexCompute*, N_ROWS_G + 1, N_COLS_G> shG(&smemBlk[SMEM_START_OFFSET_G]);

    const uint32_t                                     SMEM_START_OFFSET_I = SMEM_START_OFFSET_G + shG.num_elem();
    block_2D<TComplexCompute*, N_ROWS_I + 1, N_COLS_I> shI(&smemBlk[SMEM_START_OFFSET_I]);

    const uint32_t                                     SMEM_START_OFFSET_M = SMEM_START_OFFSET_I + shI.num_elem();
    block_2D<TComplexCompute*, N_ROWS_M + 1, N_COLS_M> shM(&smemBlk[SMEM_START_OFFSET_M]);

    // SMEM overlay: after LU - U replaces G, Linv replaces I and F replaces M
    auto& shU    = shG;
    auto& shLinv = shI;
    auto& shF    = shM;

    // SMEM overlay: after back substitution - Ree replaces Linv and C replaces F
    // (i.e. results are stored inplace)
    auto& shRee = shLinv;
    auto& shC   = shF;

    const uint32_t chEqTimeInstIdx = dynDescr.chEqTimeInstIdx;

    //--------------------------------------------------------------------------------------------------------
    // Stage1: Load inputs
    thread_block const& thisThrdBlk = this_thread_block();

#ifdef ENABLE_DEBUG
    if(0 != blockIdx.x) return;
#endif

    // Prologue
    // Prefetch H for first iteration
    uint32_t f       = 0;
    uint32_t currIdx = 0;
    if(COL_IDX_H < N_COLS_H)
    {
        const uint32_t FREQ_IDX     = THRD_GRP_START_FREQ_OFFSET + (f * N_TONES_PER_ITER);
        const uint32_t ABS_FREQ_IDX = THRD_BLK_ABS_START_FREQ_IDX + FREQ_IDX;

        shH(ROW_IDX_H, COL_IDX_H, currIdx) =
            type_convert<TComplexCompute>(tH(ROW_IDX_H, COL_IDX_H, ABS_FREQ_IDX, chEqTimeInstIdx));

        // tDbg(ROW_IDX_H,COL_IDX_H,ABS_FREQ_IDX, chEqTimeInstIdx) = shH(ROW_IDX_H,COL_IDX_H);
#ifdef ENABLE_DEBUG
        printf("H[%d][%d][%d] = %f+j%f\n", ABS_FREQ_IDX, ROW_IDX_H, COL_IDX_H, shH(ROW_IDX_H, COL_IDX_H, FREQ_IDX).x, shH(ROW_IDX_H, COL_IDX_H, FREQ_IDX).y);
#endif
    }

    for(int32_t f = 0; f < N_ITER; ++f)
    {
        const uint32_t FREQ_IDX     = THRD_GRP_START_FREQ_OFFSET + (f * N_TONES_PER_ITER);
        const uint32_t ABS_FREQ_IDX = THRD_BLK_ABS_START_FREQ_IDX + FREQ_IDX;

        // A thread can process elements different from what it loaded, so wait for all threads in the thread block
        // to complete
        thisThrdBlk.sync();

        // Prefetch H for next iteration
        if((COL_IDX_H < N_COLS_H) && (f < (N_ITER - 1)))
        {
            const uint32_t NXT_FREQ_IDX     = THRD_GRP_START_FREQ_OFFSET + ((f + 1) * N_TONES_PER_ITER);
            const uint32_t NXT_ABS_FREQ_IDX = THRD_BLK_ABS_START_FREQ_IDX + NXT_FREQ_IDX;
            shH(ROW_IDX_H, COL_IDX_H, currIdx ^ 1) =
                type_convert<TComplexCompute>(tH(ROW_IDX_H, COL_IDX_H, NXT_ABS_FREQ_IDX, chEqTimeInstIdx));

            // tDbg(ROW_IDX_H,COL_IDX_H,NXT_ABS_FREQ_IDX, chEqTimeInstIdx) = shH(ROW_IDX_H,COL_IDX_H);
#ifdef ENABLE_DEBUG
            printf("H[%d][%d][%d] = %f+j%f\n", NXT_ABS_FREQ_IDX, ROW_IDX_H, COL_IDX_H, shH(ROW_IDX_H, COL_IDX_H, NXT_FREQ_IDX).x, shH(ROW_IDX_H, COL_IDX_H, NXT_FREQ_IDX).y);
#endif
        }

        //---------------------------------------------------------------------------------------------------
        // Stage2: Compute the enhanced Gram matrix: M = H'*inv(Rww) and G = (H'*inv(Rww)*H + inv(Rxx))

        // Compute M = H'*inv(Rww): N_COLS_H x N_COLS_RWW = N_LAYERS x N_BS_ANTS
        shM(ROW_IDX_M, COL_IDX_M) = cuConj(shH(COL_IDX_M, ROW_IDX_M)) * invDiagReg;

        // Wait for matrix M computation to finish, before using it in computation of G
        thisThrdBlk.sync();

        // if((1 == f) && (COL_IDX_RWW < N_COLS_RWW)) tDbg(ROW_IDX_RWW,COL_IDX_RWW,PRB_IDX, chEqTimeInstIdx) = shRwwInv(ROW_IDX_RWW,COL_IDX_RWW);
        // tDbg(ROW_IDX_H,COL_IDX_H,THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shH(ROW_IDX_H,COL_IDX_H,currIdx);
        // tDbg(ROW_IDX_M,COL_IDX_M,THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shM(ROW_IDX_M,COL_IDX_M);

        // Compute G = (M*H + RxxInv)
        if(COL_IDX_G < N_COLS_G)
        {
            TComplexCompute G = cuGet<TComplexCompute>(0);

#pragma unroll
            for(uint32_t i = 0; i < N_ROWS_H; ++i)
            {
                G = cuFma(shM(ROW_IDX_G, i), shH(i, COL_IDX_G, currIdx), G);
            }

            if(ROW_IDX_G == COL_IDX_G)
            {
                G += ES_INV;
            }
            shG(ROW_IDX_G, COL_IDX_G) = G;

#ifdef ENABLE_DEBUG
            printf("After: M[%d][%d][%d] = %f+j%f\n", FREQ_IDX, ROW_IDX_M, COL_IDX_M, shM(ROW_IDX_M, COL_IDX_M).x, shM(ROW_IDX_M, COL_IDX_M).y);
            printf("G[%d][%d][%d] = %f+j%f\n", FREQ_IDX, ROW_IDX_G, COL_IDX_G, shG(ROW_IDX_G, COL_IDX_G).x, shG(ROW_IDX_G, COL_IDX_G).y);
#endif

            // tDbg(ROW_IDX_G,COL_IDX_G,THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = type_convert<TComplexStorageOut>(shG(ROW_IDX_G,COL_IDX_G));
        }

        // Initialize matrix I
        if(COL_IDX_I < N_COLS_I)
        {
            shI(ROW_IDX_I, COL_IDX_I) =
                (ROW_IDX_I != COL_IDX_I) ? cuGet<TComplexCompute>(0) : cuGet<TComplexCompute>(1);
        }

        // Wait for matrix G computation to finish, before using it in computation of Ginv
        thisThrdBlk.sync();

#ifdef ENABLE_DEBUG
        // A0
        for(uint32_t i = 0; i < N_ROWS_A; ++i)
        {
            if(THREAD_X_IDX < N_COLS_A)
                tDbg(i, THREAD_X_IDX, THRD_BLK_ABS_START_FREQ_IDX + FREQ_IDX, chEqTimeInstIdx) = type_convert<TComplexStorageOut>(shA(i, THREAD_X_IDX));
        }
#endif

        //---------------------------------------------------------------------------------------------------
        // Stage3: Perform joint LU factorization
        // A = [ G | I | M ] -> [ U | Linv | F ]
        // where U = L\G, Linv = L\I, F = L\M

        // eq_mmse_coef_comp_high_mimo_kernel_v1: thread block size >> # of columns of augmented matrix
        // (i.e. (N_LAYERS * N_LAYERS) >> (2*N_LAYERS + N_BS_ANTS)). Thus use parallel version of the
        // factorization algorithm to cut down iteration count and increase active threads during sub-matrix
        // updates
        // luFactorizeIter<TCompute, N_ROWS_A, N_COLS_A>(thisThrdBlk, shA);
        // luFactorizeParallel_v1<TCompute, N_ROWS_A, N_COLS_A>(thisThrdBlk, shA);
        luFactorizeParallel_v2<TCompute, N_ROWS_A, N_COLS_A>(thisThrdBlk, shA);

#ifdef ENABLE_DEBUG
        // A1
        for(uint32_t i = 0; i < N_ROWS_A; ++i)
        {
            if(THREAD_X_IDX < N_COLS_A)
                tDbg(i, THREAD_X_IDX, THRD_BLK_ABS_START_FREQ_IDX + FREQ_IDX, chEqTimeInstIdx) = type_convert<TComplexStorageOut>(shA(i, THREAD_X_IDX));
        }
#endif
        //---------------------------------------------------------------------------------------------------
        // Stage4: Solve by back substitution, compute residual error covariance matrix Ree as the inverse
        // of Gram matrix: Ginv = Ree and the MMSE coefficient matrix C
        // Solve U*[ Ree | C ] = [ Linv | F ] for Ginv and C where Ginv = U\(Linv) and C = U\F

        backSub<TCompute, N_ROWS_U, N_COLS_U, N_ROWS_A, N_COLS_A>(thisThrdBlk, START_COL_OFFSET_REE, shU, shA);

#ifdef ENABLE_DEBUG
        // A2
        for(uint32_t i = 0; i < N_ROWS_A; ++i)
        {
            if(THREAD_X_IDX < N_COLS_A)
                tDbg(i, THREAD_X_IDX, THRD_BLK_ABS_START_FREQ_IDX + FREQ_IDX, chEqTimeInstIdx) = type_convert<TComplexStorageOut>(shA(i, THREAD_X_IDX));
        }
#endif
        //--------------------------------------------------------------------------------------------------------
        // Stage5: Write the results (Ree and C) into device memory
        if(COL_IDX_C < N_COLS_C)
        {
            // Compute bias correction factor lambda and apply to coefficients
            // TCompute lambda = cuGet<TCompute>(1) / (cuGet<TCompute>(1) - cuReal(shRee(ROW_IDX_C, ROW_IDX_C)));
            TCompute one = cuGet<TCompute>(1);
            TCompute lambda = one;
            TCompute realRee = cuReal(shRee(ROW_IDX_C, ROW_IDX_C));
            if(realRee < one)
            {
                lambda = one / (one - realRee);
            }

            tCoef(COL_IDX_C, FREQ_IDX, ROW_IDX_C, PRB_IDX) =
                type_convert<TComplexStorageOut>(shC(ROW_IDX_C, COL_IDX_C) * lambda);
        }

        if(THREAD_X_IDX < N_ROWS_REE)
        {
            // Compute ReeInv while applying bias correction
            // TCompute reeInv = (cuGet<TCompute>(1)/shMemBlkReeDiag(THREAD_X_IDX, FREQ_IDX)) - cuGet<TCompute>(1);
            TCompute ree    = cuReal(shRee(THREAD_X_IDX, THREAD_X_IDX));
            TCompute reeInv = (cuGet<TCompute>(1) - ree) / ree;

            // saturate error precision
            if(reeInv > MAX_ERROR_PRECISION)
                reeInv = MAX_ERROR_PRECISION;

            tReeDiagInv(FREQ_IDX, THREAD_X_IDX, PRB_IDX, chEqTimeInstIdx) = type_convert<TStorageOut>(reeInv);
        }

#ifdef ENABLE_DEBUG
        printf("C[%d][%d][%d] = %f+j%f\n", ABS_FREQ_IDX, ROW_IDX_C, COL_IDX_C, shC(ROW_IDX_C, COL_IDX_C).x, shC(ROW_IDX_C, COL_IDX_C).y);
#endif
        // thisThrdBlk.sync();
        currIdx ^= 1;
    }
}

// Equalization kernel for high order MIMO and per PRB
// This flavor uses fewer threads (as many needed as solve the joint LU/back substitution processing)
// {N_LAYERS, N_BS_ANTS} = {8,16}, {16,16}
// Inputs and outputs assumed to be column major
// dimBlock: (N_BS_ANTS*N_LAYERS, N_TONES_PER_ITER)
// dimGrid : ((CUPHY_N_TONES_PER_PRB/N_THRD_BLK_TONES), Nprb)
template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS,        // # of BS antenna (# of rows in H matrix)  // used for static shared memory allocation
          uint32_t N_LAYERS,         // # of layers (# of cols in H matrix)
          uint32_t N_THRD_BLK_TONES, // # of frequency bins processed by a thread block
          uint32_t N_TONES_PER_ITER> // # of frequency bins processed in 1 iteration by the thread block
__device__ void
eqMmseCoefCompHighMimoKernel_v2(puschRxChEqStatDescr_t* pStatDescr, puschRxChEqCoefCompDynDescr_t* pDynDescr)
{
    KERNEL_PRINT_GRID_ONCE("%s\n grid = (%u %u %u), block = (%u %u %u), N_BS_ANTS = %u, N_LAYERS = %u, N_FREQ_BINS_PER_ITER = %u\n",
                           __PRETTY_FUNCTION__,
                           gridDim.x, gridDim.y, gridDim.z,
                           blockDim.x, blockDim.y, blockDim.z,
                           N_BS_ANTS,
                           N_LAYERS,
                           CUPHY_N_TONES_PER_PRB);
    //--------------------------------------------------------------------------------------------------------
    // Setup local parameters based on descriptor
    // puschRxChEqStatDescr_t&        statDescr = *(pStatDescr);
    puschRxChEqCoefCompDynDescr_t& dynDescr = *(pDynDescr);

    // Early exit check
    // The grid is sized to process the max # of PRBs in a given heterogenous config. Exit if the PRB to be
    // processed by this thread block does not exist in the UE group
    const uint32_t PRB_IDX = blockIdx.y;
    const uint32_t UE_GRP_IDX = dynDescr.hetCfgUeGrpMap[blockIdx.z];    

    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];

    const uint16_t nPrb = drvdUeGrpPrms.nPrb;
    if(PRB_IDX >= nPrb) return;

    //--------------------------------------------------------------------------------------------------------
    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    // clang-format off
    tensor_ref<const TComplexStorageIn> tH         (drvdUeGrpPrms.tInfoHEst.pAddr       , drvdUeGrpPrms.tInfoHEst.strides       );// (N_BS_ANTS, N_LAYERS, NF, NH)
    tensor_ref<TStorageOut>             tReeDiagInv(drvdUeGrpPrms.tInfoReeDiagInv.pAddr , drvdUeGrpPrms.tInfoReeDiagInv.strides );// (N_SC, N_LAYERS, N_PRB, NH)   // (N_LAYERS, NF, NH)
    tensor_ref<TComplexStorageOut>      tCoef      (drvdUeGrpPrms.tInfoEqCoef.pAddr     , drvdUeGrpPrms.tInfoEqCoef.strides     );// (N_SC, N_LAYERS, N_BS_ANTS, N_PRB, NH) // (N_LAYERS, N_BS_ANTS, NF, NH)
    tensor_ref<TComplexStorageOut>      tDbg       (drvdUeGrpPrms.tInfoChEqDbg.pAddr    , drvdUeGrpPrms.tInfoChEqDbg.strides    );
    // clang-format on

    // set diagonal regularizer:
    bool   noiseRegFlag  = (drvdUeGrpPrms.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE) ? true : false;
    float  invDiagReg    = noiseRegFlag ? drvdUeGrpPrms.invNoiseVarLin : INV_ZF_REGULARIZER;

    //--------------------------------------------------------------------------------------------------------
    // Dimensions

    // H is channel matrix
    // G is the enhanced Gram matrix
    // A is the augmented matrix, A = [ G | I | M ]

    // H  : Channel matrix
    constexpr uint32_t N_ROWS_H = N_BS_ANTS;
    constexpr uint32_t N_COLS_H = N_LAYERS;

    // M  : Intermediate matrix, M = H'*RwwInv
    constexpr uint32_t N_ROWS_M = N_COLS_H;   // N_LAYERS
    constexpr uint32_t N_COLS_M = N_BS_ANTS; // N_BS_ANTS

    // G  : Enhanced Gram matrix, G = H'*RwwInv*H + inv(Rxx)
    constexpr uint32_t N_ROWS_G = N_LAYERS;
    constexpr uint32_t N_COLS_G = N_LAYERS;

    // U  : Upper triangular matrix
    constexpr uint32_t N_ROWS_U = N_ROWS_G;
    constexpr uint32_t N_COLS_U = N_COLS_G;

    // I  : Identity matrix, I = G*Ginv
    constexpr uint32_t N_ROWS_I = N_ROWS_G;
    constexpr uint32_t N_COLS_I = N_COLS_G;

    // Ree: Residual error covariance matrix, Ree = Ginv
    constexpr uint32_t N_ROWS_REE = N_ROWS_G;
    // constexpr uint32_t N_COLS_REE = N_COLS_G;

    // C  : MMSE coefficient matrix, C = Ree*H'*RwwInv = Ree*M
    constexpr uint32_t N_ROWS_C = N_ROWS_REE;
    constexpr uint32_t N_COLS_C = N_COLS_M;

    // A  : Augmented result matrix, A = [ G | I | M ] -> [ U | Linv | F ] -> [ U | Ginv | C ]
    constexpr uint32_t N_ROWS_A = N_ROWS_G;
    constexpr uint32_t N_COLS_A = N_COLS_G + N_COLS_I + N_COLS_M;

    // Column offsets to Ree (which is followed by C) matrix within the augmented matrix
    // which form the RHS of the back substition (and after back substitution the RHS is overwritten by the
    // unknowns being solved for)
    constexpr uint32_t START_COL_OFFSET_REE = N_COLS_U;

    constexpr int32_t N_ITER = N_THRD_BLK_TONES / N_TONES_PER_ITER;

    constexpr uint32_t N_INST = 2; // double buffering for pipelining

    const uint32_t N_THREADS_X = blockDim.x; // Number of threads needed to process one frequency bin
    // const uint32_t N_THREADS   = blockDim.x * blockDim.y; // N_TONES_PER_ITER == blockDim.y

    // Reciprocal of symbol energy
    const TCompute ES_INV = cuGet<TCompute>(1);

    //--------------------------------------------------------------------------------------------------------
    // Compute indices used for element access
    const uint32_t THREAD_X_IDX = threadIdx.x;
    // const uint32_t THREAD_IDX   = (threadIdx.y * blockDim.x) + threadIdx.x;

    // There are N_TONES_PER_ITER groups of threads with each group containing N_THREADS_X per group
    const uint32_t THRD_GRP_FREQ_OFFSET        = threadIdx.y;
    const uint32_t THRD_BLK_START_FREQ_OFFSET  = (blockIdx.x * N_THRD_BLK_TONES);
    const uint32_t THRD_GRP_START_FREQ_OFFSET  = THRD_BLK_START_FREQ_OFFSET + THRD_GRP_FREQ_OFFSET;
    const uint32_t THRD_BLK_ABS_START_FREQ_IDX = PRB_IDX * CUPHY_N_TONES_PER_PRB;

    const uint32_t chEqTimeInstIdx = dynDescr.chEqTimeInstIdx;

    // const uint32_t ROW_IDX_RWW = THREAD_IDX % N_ROWS_RWW;
    // const uint32_t COL_IDX_RWW = THREAD_IDX / N_ROWS_RWW; // COL_IDX_RWW needs a bounds check (since N_THREADS > # of Rww elements)

    const uint32_t ROW_IDX_H = THREAD_X_IDX % N_ROWS_H;
    const uint32_t COL_IDX_H = THREAD_X_IDX / N_ROWS_H;

    const uint32_t ROW_IDX_I = THREAD_X_IDX % N_ROWS_I;
    // const uint32_t COL_IDX_I = THREAD_X_IDX / N_ROWS_I;

    // const uint32_t ROW_IDX_M = THREAD_X_IDX % N_ROWS_M;
    // const uint32_t COL_IDX_M = THREAD_X_IDX / N_ROWS_M;

    const uint32_t ROW_IDX_G = THREAD_X_IDX % N_ROWS_G;
    const uint32_t COL_IDX_G = THREAD_X_IDX / N_ROWS_G; // COL_IDX_G needs a bounds check (since N_THREADS_X > # of G elements)

    // const uint32_t ROW_IDX_REE = THREAD_X_IDX % N_ROWS_REE;
    // const uint32_t COL_IDX_REE = THREAD_X_IDX / N_ROWS_REE; // COL_IDX_REE needs a bounds check (since N_THREADS_X > # of Ree elements)

    const uint32_t ROW_IDX_C = THREAD_X_IDX % N_ROWS_C;
    const uint32_t COL_IDX_C = THREAD_X_IDX / N_ROWS_C;

    const uint32_t N_COLS_H_RD_PER_ITER = N_THREADS_X / N_ROWS_H;
    const uint32_t N_ITER_TO_RD_H       = div_round_up(N_COLS_H, N_COLS_H_RD_PER_ITER);

    const uint32_t N_COLS_G_COMPUTE_PER_ITER = N_THREADS_X / N_ROWS_G;
    const uint32_t N_ITER_TO_COMPUTE_G       = div_round_up(N_COLS_G, N_COLS_G_COMPUTE_PER_ITER);

    const uint32_t N_COLS_C_WR_PER_ITER = N_THREADS_X / N_ROWS_C;
    const uint32_t N_ITER_TO_WR_C       = div_round_up(N_COLS_C, N_COLS_C_WR_PER_ITER);

#ifdef ENABLE_DEBUG
 
    if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
    {
        printf("Addr: tH %lp  tReeDiagInv %lp tCoef %lp\n", tH.addr, tReeDiagInv.addr, tCoef.addr);
 
        printf("nPrb     : %d \n", nPrb);
        printf("tH          : addr %lp strides[0] %d strides[1] %d strides[2] %d strides[3] %d\n", static_cast<const TComplexStorageIn*>(tH.addr)     , tH.strides[0]         , tH.strides[1]         , tH.strides[2]         , tH.strides[3]         );
        printf("tReeDiagInv : addr %lp strides[0] %d strides[1] %d strides[2] %d strides[3] %d\n", static_cast<TStorageOut*>(tReeDiagInv.addr)        , tReeDiagInv.strides[0], tReeDiagInv.strides[1], tReeDiagInv.strides[2], tReeDiagInv.strides[3]);
        printf("tCoef       : addr %lp strides[0] %d strides[1] %d strides[2] %d strides[3] %d\n", static_cast<TComplexStorageOut*>(tCoef.addr)       , tCoef.strides[0]      , tCoef.strides[1]      , tCoef.strides[2]      , tCoef.strides[3]      );
        // printf("tDbg    strides[0] %d strides[1] %d strides[2] %d\n", tDbg.strides[0], tDbg.strides[1], tDbg.strides[2]);
    }
#endif   

    //--------------------------------------------------------------------------------------------------------
    // Shared memory allocation
    // H[N_TONES_PER_ITER*N_INST]

    // Shared memory contents as processing progresses:
    // A = [ G | I | M ] -> [ U | Linv | F ] -> [ U | Ree | C ]

    constexpr uint32_t N_SMEM_ELEMS =
        (((N_ROWS_H + 1) * N_COLS_H * N_INST) + // (N_ROWS_H + 1) for SMEM padding to avoid bank conflicts
         (N_ROWS_A + 1) * N_COLS_A) *
            N_TONES_PER_ITER;

    __shared__ TComplexCompute smemBlk[N_SMEM_ELEMS];

    constexpr uint32_t                                         SMEM_START_OFFSET_H_BLK = 0;
    const uint32_t                                             SMEM_START_OFFSET_H     = SMEM_START_OFFSET_H_BLK + THRD_GRP_FREQ_OFFSET * (N_ROWS_H + 1) * N_COLS_H * N_INST;
    block_3D<TComplexCompute*, N_ROWS_H + 1, N_COLS_H, N_INST> shH(&smemBlk[SMEM_START_OFFSET_H]);

    constexpr uint32_t                                 SMEM_START_OFFSET_A_BLK = SMEM_START_OFFSET_H_BLK + N_TONES_PER_ITER * shH.num_elem();
    const uint32_t                                     SMEM_START_OFFSET_A     = SMEM_START_OFFSET_A_BLK + THRD_GRP_FREQ_OFFSET * (N_ROWS_A + 1) * N_COLS_A;
    block_2D<TComplexCompute*, N_ROWS_A + 1, N_COLS_A> shA(&smemBlk[SMEM_START_OFFSET_A]);

    // SMEM overlay: A with [ G | I | M ]
    const uint32_t                                     SMEM_START_OFFSET_G = SMEM_START_OFFSET_A;
    block_2D<TComplexCompute*, N_ROWS_G + 1, N_COLS_G> shG(&smemBlk[SMEM_START_OFFSET_G]);

    const uint32_t                                     SMEM_START_OFFSET_I = SMEM_START_OFFSET_G + shG.num_elem();
    block_2D<TComplexCompute*, N_ROWS_I + 1, N_COLS_I> shI(&smemBlk[SMEM_START_OFFSET_I]);

    const uint32_t                                     SMEM_START_OFFSET_M = SMEM_START_OFFSET_I + shI.num_elem();
    block_2D<TComplexCompute*, N_ROWS_M + 1, N_COLS_M> shM(&smemBlk[SMEM_START_OFFSET_M]);

    // SMEM overlay: after LU - U replaces G, Linv replaces I and F replaces M
    auto& shU    = shG;
    auto& shLinv = shI;
    auto& shF    = shM;

    // SMEM overlay: after back substitution - Ree replaces Linv and C replaces F
    // (i.e. results are stored inplace)
    auto& shRee = shLinv;
    auto& shC   = shF;

    bool enableTdi = (0 != drvdUeGrpPrms.enablePuschTdi) ? true : false;    

    //--------------------------------------------------------------------------------------------------------
    // Stage1: Load inputs
    thread_block const& thisThrdBlk = this_thread_block();

    // Prologue
    // Prefetch H into shared memory
    uint32_t       f            = 0;
    uint32_t       currIdx      = 0;
    const uint32_t FREQ_IDX     = THRD_GRP_START_FREQ_OFFSET + (f * N_TONES_PER_ITER);
    const uint32_t ABS_FREQ_IDX = THRD_BLK_ABS_START_FREQ_IDX + FREQ_IDX;

#pragma unroll
    for(uint32_t i = 0; i < N_ITER_TO_RD_H; ++i)
    {
        uint32_t iCol = i * N_COLS_H_RD_PER_ITER + COL_IDX_H;
        // All threads may not participate in the last iteration
        if(iCol < N_COLS_H)
        {
            shH(ROW_IDX_H, iCol, currIdx) = type_convert<TComplexCompute>(tH(ROW_IDX_H, iCol, ABS_FREQ_IDX, chEqTimeInstIdx));

#ifdef ENABLE_DEBUG
            printf("H[%d][%d] = %f+j%f\n", ROW_IDX_H, iCol, shH(ROW_IDX_H, iCol, currIdx).x, shH(ROW_IDX_H, iCol, currIdx).y);
#endif
        }
    }

    for(int32_t f = 0; f < N_ITER; ++f)
    {
        const uint32_t FREQ_IDX = THRD_GRP_START_FREQ_OFFSET + (f * N_TONES_PER_ITER);
        //const uint32_t ABS_FREQ_IDX = THRD_BLK_ABS_START_FREQ_IDX + FREQ_IDX;

        // A thread can process elements different from what it loaded, so wait for all threads in the thread block
        // to complete
        thisThrdBlk.sync();

        // Prefetch H for next iteration
        if(f < (N_ITER - 1))
        {
            const uint32_t NXT_FREQ_IDX     = THRD_GRP_START_FREQ_OFFSET + ((f + 1) * N_TONES_PER_ITER);
            const uint32_t NXT_ABS_FREQ_IDX = THRD_BLK_ABS_START_FREQ_IDX + NXT_FREQ_IDX;
#pragma unroll
            for(uint32_t i = 0; i < N_ITER_TO_RD_H; ++i)
            {
                uint32_t iCol = i * N_COLS_H_RD_PER_ITER + COL_IDX_H;
                // All threads may not participate in the last iteration
                if(iCol < N_COLS_H)
                {
                    shH(ROW_IDX_H, iCol, currIdx ^ 1) = type_convert<TComplexCompute>(tH(ROW_IDX_H, iCol, NXT_ABS_FREQ_IDX, chEqTimeInstIdx));

#ifdef ENABLE_DEBUG
                    printf("H[%d][%d] = %f+j%f\n", ROW_IDX_H, iCol, shH(ROW_IDX_H, iCol, currIdx ^ 1).x, shH(ROW_IDX_H, iCol, currIdx ^ 1).y);
#endif
                }
            }
        }

        //---------------------------------------------------------------------------------------------------
        // Stage2: Compute the enhanced Gram matrix: M = H'*inv(Rww) and G = (H'*inv(Rww)*H + inv(Rxx))

        // Compute M = H'*inv(Rww): N_COLS_H x N_BS_ANTS = N_LAYERS x N_BS_ANTS
        // Select a subset of the threads in the warp to access the columns of RwwInv and each thread computes one
        // column of M
        if(THREAD_X_IDX < N_BS_ANTS)
        {
            // Each thread computes one column of M
#pragma unroll
            for(uint32_t j = 0; j < N_COLS_H; ++j)
            {
               shM(j, THREAD_X_IDX) = cuConj(shH(THREAD_X_IDX, j, currIdx)) * invDiagReg;

#ifdef ENABLE_DEBUG
                printf("M[%d][%d] = %f+j%f\n", j, THREAD_X_IDX, shM(j, THREAD_X_IDX).x, shM(j, THREAD_X_IDX).y);
#endif
                // tDbg(j,THREAD_X_IDX,ABS_FREQ_IDX,chEqTimeInstIdx) =
                //    type_convert<TComplexStorageOut>(shM(j,THREAD_X_IDX));
            }
        }

        // Wait for matrix M computation to finish, before using it in computation of G
        thisThrdBlk.sync();

        // Compute G = (M*H + RxxInv): N_LAYERS x N_LAYERS
#pragma unroll
        for(uint32_t i = 0; i < N_ITER_TO_COMPUTE_G; ++i)
        {
            uint32_t iCol = i * N_COLS_G_COMPUTE_PER_ITER + COL_IDX_G;

            // All threads may not participate in the last iteration
            if((ROW_IDX_G < N_ROWS_G) && (iCol < N_COLS_G))
            {
                TComplexCompute G{};

#pragma unroll
                for(uint32_t elem = 0; elem < N_BS_ANTS; ++elem)
                {
                    G = cuFma(shM(ROW_IDX_G, elem), shH(elem, iCol, currIdx), G);
                }

                // Add Noise_pwr*inv(Rxx) to the diagonal of H'*H, inv(Rxx) is assumed to be unity
                if(ROW_IDX_G == iCol)
                {
                    G += ES_INV;
                }
                shG(ROW_IDX_G, iCol) = G;

#ifdef ENABLE_DEBUG
                printf("G[%d][%d] = %f+j%f\n", ROW_IDX_G, iCol, shG(ROW_IDX_G, iCol).x, shG(ROW_IDX_G, iCol).y);
#endif

                // tDbg(ROW_IDX_G,iCol,ABS_FREQ_IDX,chEqTimeInstIdx) =
                //   type_convert<TComplexStorageOut>(shG(ROW_IDX_G,iCol));
            }

            // Initialize matrix I (which has the same dimenison as G)
            if(iCol < N_COLS_I)
            {
                shI(ROW_IDX_I, iCol) =
                    (ROW_IDX_I != iCol) ? cuGet<TComplexCompute>(0) : cuGet<TComplexCompute>(1);
            }
        }

        // if((1 == f) && (COL_IDX_RWW < N_COLS_RWW)) tDbg(ROW_IDX_RWW,COL_IDX_RWW,PRB_IDX, chEqTimeInstIdx) = shRwwInv(ROW_IDX_RWW,COL_IDX_RWW);
        // tDbg(ROW_IDX_H,COL_IDX_H,THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shH(ROW_IDX_H,COL_IDX_H,currIdx);
        // tDbg(ROW_IDX_M,COL_IDX_M,THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shM(ROW_IDX_M,COL_IDX_M);

        // Wait for matrix G computation to finish, before using it in computation of Ginv
        thisThrdBlk.sync();

        //---------------------------------------------------------------------------------------------------
        // Stage3: Perform joint LU factorization
        // A = [ G | I | M ] -> [ U | Linv | F ]
        // where U = L\G, Linv = L\I, F = L\M

        // eq_mmse_coef_comp_high_mimo_kernel_v2: thread block has size == # of columns of augmented matrix
        // (thread block size = row length of augmented matrix = (2*N_LAYERS + N_BS_ANTS)).
        // Thus no benefit in using the parallel version of factorization algorithm (for sub-matrix updates)
        luFactorizeIter<TCompute, N_ROWS_A, N_COLS_A>(thisThrdBlk, shA);
        // luFactorizeParallel_v1<TCompute, N_ROWS_A, N_COLS_A>(thisThrdBlk, shA);
        // luFactorizeParallel_v2<TCompute, N_ROWS_A, N_COLS_A>(thisThrdBlk, shA);

        //---------------------------------------------------------------------------------------------------
        // Stage4: Solve by back substitution, compute residual error covariance matrix Ree as the inverse
        // of Gram matrix: Ginv = Ree and the MMSE coefficient matrix C
        // Solve U*[ Ree | C ] = [ Linv | F ] for Ginv and C where Ginv = U\(Linv) and C = U\F

        backSub<TCompute, N_ROWS_U, N_COLS_U, N_ROWS_A, N_COLS_A>(thisThrdBlk, START_COL_OFFSET_REE, shU, shA);

        //--------------------------------------------------------------------------------------------------------
        // Stage5: Write the results (Ree and C) into device memory

#pragma unroll
        for(uint32_t i = 0; i < N_ITER_TO_WR_C; ++i)
        {
            uint32_t iCol = i * N_COLS_C_WR_PER_ITER + COL_IDX_C;

            // All threads may not participate in the last iteration
            if(iCol < N_COLS_C)
            {
                // Compute bias correction factor lambda and apply to coefficients
                // TCompute lambda = cuGet<TCompute>(1) / (cuGet<TCompute>(1) - cuReal(shRee(ROW_IDX_C, ROW_IDX_C)));
                TCompute one = cuGet<TCompute>(1);
                TCompute lambda = one;
                TCompute realRee = cuReal(shRee(ROW_IDX_C, ROW_IDX_C));
                if( realRee < one)
                {
                    lambda = one / (one - realRee);
                }
                

                if (enableTdi) {
                    tCoef(iCol, PRB_IDX * 12 + FREQ_IDX, ROW_IDX_C, chEqTimeInstIdx) =
                        type_convert<TComplexStorageOut>(shC(ROW_IDX_C, iCol) * lambda);
                }
                else {
                    tCoef(iCol, FREQ_IDX, ROW_IDX_C, PRB_IDX) =
                        type_convert<TComplexStorageOut>(shC(ROW_IDX_C, iCol) * lambda);              
                }
            }
        }

        if(THREAD_X_IDX < N_ROWS_REE)
        {
            // Compute ReeInv while applying bias correction
            // TCompute reeInv = (cuGet<TCompute>(1)/shMemBlkReeDiag(THREAD_X_IDX, FREQ_IDX)) - cuGet<TCompute>(1);
            TCompute ree    = cuReal(shRee(THREAD_X_IDX, THREAD_X_IDX));
            TCompute reeInv = (cuGet<TCompute>(1) - ree) / ree;

            // saturate error precision
            if(reeInv > static_cast<TCompute>(MAX_ERROR_PRECISION))
                reeInv = static_cast<TCompute>(MAX_ERROR_PRECISION);

            tReeDiagInv(FREQ_IDX, THREAD_X_IDX, PRB_IDX, chEqTimeInstIdx) = type_convert<TStorageOut>(reeInv);
        }

#ifdef ENABLE_DEBUG
        printf("C[%d][%d][%d] = %f+j%f\n", ABS_FREQ_IDX, ROW_IDX_C, COL_IDX_C, shC(ROW_IDX_C, COL_IDX_C).x, shC(ROW_IDX_C, COL_IDX_C).y);
#endif
        // thisThrdBlk.sync();
        currIdx ^= 1;
    }
}

// Equalization kernel for high order MIMO and per PRB
// This flavor uses fewer threads (as many needed as solve the joint LU/back substitution processing)
// {N_LAYERS, N_BS_ANTS} = {8,16}, {16,16}
// Inputs and outputs assumed to be column major
// dimBlock: (N_BS_ANTS*N_LAYERS, N_TONES_PER_ITER)
// dimGrid : ((CUPHY_N_TONES_PER_PRB/N_THRD_BLK_TONES), Nprb)
template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS,        // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS,         // # of layers (# of cols in H matrix)
          uint32_t N_THRD_BLK_TONES, // # of frequency bins processed by a thread block
          uint32_t N_TONES_PER_ITER> // # of frequency bins processed in 1 iteration by the thread block
__device__ void
eqMmseCoefCompHighMimoKernel_v3(puschRxChEqStatDescr_t* pStatDescr, puschRxChEqCoefCompDynDescr_t* pDynDescr)
{
    //--------------------------------------------------------------------------------------------------------
    // Setup local parameters based on descriptor
    // puschRxChEqStatDescr_t&        statDescr = *(args.pStatDescr);
    puschRxChEqCoefCompDynDescr_t& dynDescr = *(pDynDescr);

    // Early exit check
    // The grid is sized to process the max # of PRBs in a given heterogenous config. Exit if the PRB to be
    // processed by this thread block does not exist in the UE group
    const uint32_t PRB_IDX = blockIdx.y;
    const uint32_t UE_GRP_IDX = dynDescr.hetCfgUeGrpMap[blockIdx.z];    

    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];

    const uint16_t nPrb = drvdUeGrpPrms.nPrb;
    if(PRB_IDX >= nPrb) return;

    //--------------------------------------------------------------------------------------------------------
    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;
    typedef float                                           TCompute32;
    typedef cuComplex                                       TComplexCompute32;

    // clang-format off
    tensor_ref<const TComplexStorageIn> tH         (drvdUeGrpPrms.tInfoHEst.pAddr       , drvdUeGrpPrms.tInfoHEst.strides      );// (N_BS_ANTS, N_LAYERS, NF, NH)
    tensor_ref<TStorageOut>             tReeDiagInv(drvdUeGrpPrms.tInfoReeDiagInv.pAddr , drvdUeGrpPrms.tInfoReeDiagInv.strides);// (N_SC, N_LAYERS, N_PRB, NH)   // (N_LAYERS, NF, NH)
    tensor_ref<TComplexStorageOut>      tCoef      (drvdUeGrpPrms.tInfoEqCoef.pAddr     , drvdUeGrpPrms.tInfoEqCoef.strides      );// (N_SC, N_LAYERS, N_BS_ANTS, N_PRB, NH) // (N_LAYERS, N_BS_ANTS, NF, NH)
    tensor_ref<TComplexStorageOut>      tDbg       (drvdUeGrpPrms.tInfoChEqDbg.pAddr    , drvdUeGrpPrms.tInfoChEqDbg.strides       );
    // clang-format on

    // set diagonal regularizer:
    bool   noiseRegFlag  = (drvdUeGrpPrms.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE) ? true : false;
    float  invDiagReg    = noiseRegFlag ? drvdUeGrpPrms.invNoiseVarLin : INV_ZF_REGULARIZER;
    //--------------------------------------------------------------------------------------------------------
    // Dimensions

    // H is channel matrix
    // G is the enhanced Gram matrix
    // A is the augmented matrix, A = [ G | I | M ]

    // H  : Channel matrix
    constexpr uint32_t N_ROWS_H = N_BS_ANTS;
    constexpr uint32_t N_COLS_H = N_LAYERS;

    // M  : Intermediate matrix, M = H'*RwwInv
    constexpr uint32_t N_ROWS_M = N_COLS_H;   // N_LAYERS
    constexpr uint32_t N_COLS_M = N_BS_ANTS; // N_BS_ANTS

    // G  : Enhanced Gram matrix, G = H'*RwwInv*H + inv(Rxx)
    constexpr uint32_t N_ROWS_G = N_LAYERS;
    constexpr uint32_t N_COLS_G = N_LAYERS;

    // U  : Upper triangular matrix
    constexpr uint32_t N_ROWS_U = N_ROWS_G;
    constexpr uint32_t N_COLS_U = N_COLS_G;

    // I  : Identity matrix, I = G*Ginv
    constexpr uint32_t N_ROWS_I = N_ROWS_G;
    constexpr uint32_t N_COLS_I = N_COLS_G;

    // Ree: Residual error covariance matrix, Ree = Ginv
    constexpr uint32_t N_ROWS_REE = N_ROWS_G;
    // constexpr uint32_t N_COLS_REE = N_COLS_G;

    // C  : MMSE coefficient matrix, C = Ree*H'*RwwInv = Ree*M
    constexpr uint32_t N_ROWS_C = N_ROWS_REE;
    constexpr uint32_t N_COLS_C = N_COLS_M;

    // A  : Augmented result matrix, A = [ G | I | M ] -> [ U | Linv | F ] -> [ U | Ginv | C ]
    constexpr uint32_t N_ROWS_A = N_ROWS_G;
    constexpr uint32_t N_COLS_A = N_COLS_G + N_COLS_I + N_COLS_M;

    // Column offsets to Ree (which is followed by C) matrix within the augmented matrix
    // which form the RHS of the back substition (and after back substitution the RHS is overwritten by the
    // unknowns being solved for)
    constexpr uint32_t START_COL_OFFSET_REE = N_COLS_U;

    constexpr int32_t N_ITER = N_THRD_BLK_TONES / N_TONES_PER_ITER;

    constexpr uint32_t N_INST = 1; // double buffering for pipelining

    const uint32_t N_THREADS_X = blockDim.x; // Number of threads needed to process one frequency bin
    // const uint32_t N_THREADS   = blockDim.x * blockDim.y; // N_TONES_PER_ITER == blockDim.y

    // Reciprocal of symbol energy
    const TCompute ES_INV = cuGet<TCompute>(1);

    //--------------------------------------------------------------------------------------------------------
    // Compute indices used for element access
    const uint32_t THREAD_X_IDX = threadIdx.x;
    // const uint32_t THREAD_IDX   = (threadIdx.y * blockDim.x) + threadIdx.x;

    // There are N_TONES_PER_ITER groups of threads with each group containing N_THREADS_X per group
    const uint32_t THRD_GRP_FREQ_OFFSET        = threadIdx.y;
    const uint32_t THRD_BLK_START_FREQ_OFFSET  = (blockIdx.x * N_THRD_BLK_TONES);
    const uint32_t THRD_GRP_START_FREQ_OFFSET  = THRD_BLK_START_FREQ_OFFSET + THRD_GRP_FREQ_OFFSET;
    const uint32_t THRD_BLK_ABS_START_FREQ_IDX = PRB_IDX * CUPHY_N_TONES_PER_PRB;

    const uint32_t chEqTimeInstIdx = dynDescr.chEqTimeInstIdx;

    // const uint32_t ROW_IDX_RWW = THREAD_IDX % N_ROWS_RWW;
    // const uint32_t COL_IDX_RWW = THREAD_IDX / N_ROWS_RWW; // COL_IDX_RWW needs a bounds check (since N_THREADS > # of Rww elements)

    const uint32_t ROW_IDX_H = THREAD_X_IDX % N_ROWS_H;
    const uint32_t COL_IDX_H = THREAD_X_IDX / N_ROWS_H;

    const uint32_t ROW_IDX_I = THREAD_X_IDX % N_ROWS_I;
    // const uint32_t COL_IDX_I = THREAD_X_IDX / N_ROWS_I;

    // const uint32_t ROW_IDX_M = THREAD_X_IDX % N_ROWS_M;
    // const uint32_t COL_IDX_M = THREAD_X_IDX / N_ROWS_M;

    const uint32_t ROW_IDX_G = THREAD_X_IDX % N_ROWS_G;
    const uint32_t COL_IDX_G = THREAD_X_IDX / N_ROWS_G; // COL_IDX_G needs a bounds check (since N_THREADS_X > # of G elements)

    // const uint32_t ROW_IDX_REE = THREAD_X_IDX % N_ROWS_REE;
    // const uint32_t COL_IDX_REE = THREAD_X_IDX / N_ROWS_REE; // COL_IDX_REE needs a bounds check (since N_THREADS_X > # of Ree elements)

    const uint32_t ROW_IDX_C = THREAD_X_IDX % N_ROWS_C;
    const uint32_t COL_IDX_C = THREAD_X_IDX / N_ROWS_C;

    const uint32_t N_COLS_H_RD_PER_ITER = N_THREADS_X / N_ROWS_H;
    const uint32_t N_ITER_TO_RD_H       = div_round_up(N_COLS_H, N_COLS_H_RD_PER_ITER);

    const uint32_t N_COLS_G_COMPUTE_PER_ITER = N_THREADS_X / N_ROWS_G;
    const uint32_t N_ITER_TO_COMPUTE_G       = div_round_up(N_COLS_G, N_COLS_G_COMPUTE_PER_ITER);

    const uint32_t N_COLS_C_WR_PER_ITER = N_THREADS_X / N_ROWS_C;
    const uint32_t N_ITER_TO_WR_C       = div_round_up(N_COLS_C, N_COLS_C_WR_PER_ITER);

    //--------------------------------------------------------------------------------------------------------
    // Shared memory allocation
    // H[N_TONES_PER_ITER*N_INST]

    // Shared memory contents as processing progresses:
    // A = [ G | I | M ] -> [ U | Linv | F ] -> [ U | Ree | C ]

    constexpr uint32_t N_SMEM_BLK1_ELEMS =
        ((N_ROWS_H + 1) * N_COLS_H * N_INST) * N_TONES_PER_ITER; // (N_ROWS_H + 1) for SMEM padding to avoid bank conflicts
    
    constexpr uint32_t N_SMEM_BLK2_ELEMS = 
        (N_ROWS_A + 1) * N_COLS_A * N_TONES_PER_ITER;

    // Two smem block, one for fp16 another for fp32
    __shared__ TComplexCompute smemBlk1[N_SMEM_BLK1_ELEMS];
    __shared__ TComplexCompute32 smemBlk2[N_SMEM_BLK2_ELEMS];

    constexpr uint32_t                                         SMEM_START_OFFSET_H_BLK = 0;
    const uint32_t                                             SMEM_START_OFFSET_H     = SMEM_START_OFFSET_H_BLK + THRD_GRP_FREQ_OFFSET * (N_ROWS_H + 1) * N_COLS_H * N_INST;
    block_3D<TComplexCompute*, N_ROWS_H + 1, N_COLS_H, N_INST> shH(&smemBlk1[SMEM_START_OFFSET_H]);

    constexpr uint32_t                                 SMEM_START_OFFSET_A_BLK = 0;
    const uint32_t                                     SMEM_START_OFFSET_A     = SMEM_START_OFFSET_A_BLK + THRD_GRP_FREQ_OFFSET * (N_ROWS_A + 1) * N_COLS_A;
    // const uint32_t                                     SMEM_START_OFFSET_A     = 0;
    block_2D<TComplexCompute32*, N_ROWS_A + 1, N_COLS_A> shA(&smemBlk2[SMEM_START_OFFSET_A]);

    // SMEM overlay: A with [ G | I | M ]
    const uint32_t                                     SMEM_START_OFFSET_G = SMEM_START_OFFSET_A;
    block_2D<TComplexCompute32*, N_ROWS_G + 1, N_COLS_G> shG(&smemBlk2[SMEM_START_OFFSET_G]);

    const uint32_t                                     SMEM_START_OFFSET_I = SMEM_START_OFFSET_G + shG.num_elem();
    block_2D<TComplexCompute32*, N_ROWS_I + 1, N_COLS_I> shI(&smemBlk2[SMEM_START_OFFSET_I]);

    const uint32_t                                     SMEM_START_OFFSET_M = SMEM_START_OFFSET_I + shI.num_elem();
    block_2D<TComplexCompute32*, N_ROWS_M + 1, N_COLS_M> shM(&smemBlk2[SMEM_START_OFFSET_M]);

    // SMEM overlay: after LU - U replaces G, Linv replaces I and F replaces M
    auto& shU    = shG;
    auto& shLinv = shI;
    auto& shF    = shM;

    // SMEM overlay: after back substitution - Ree replaces Linv and C replaces F
    // (i.e. results are stored inplace)
    auto& shRee = shLinv;
    auto& shC   = shF;

    // cudaTextureObject_t rwwTexObj = statDescr.coefComp_tex;

    //--------------------------------------------------------------------------------------------------------
    // Stage1: Load inputs
    thread_block const& thisThrdBlk = this_thread_block();


    // Prologue
    // Prefetch H into shared memory
    uint32_t       currIdx      = 0;

    for(int32_t f = 0; f < N_ITER; ++f)
    {
        const uint32_t FREQ_IDX = THRD_GRP_START_FREQ_OFFSET + (f * N_TONES_PER_ITER);

        // Prefetch H for next iteration
        const uint32_t NXT_FREQ_IDX     = THRD_GRP_START_FREQ_OFFSET + ((f + 0) * N_TONES_PER_ITER);
        const uint32_t NXT_ABS_FREQ_IDX = THRD_BLK_ABS_START_FREQ_IDX + NXT_FREQ_IDX;
// #pragma unroll
        for(uint32_t i = 0; i < N_ITER_TO_RD_H; ++i)
        {
            uint32_t iCol = i * N_COLS_H_RD_PER_ITER + COL_IDX_H;
            // All threads may not participate in the last iteration
            if(iCol < N_COLS_H)
            {
                shH(ROW_IDX_H, iCol, currIdx) = type_convert<TComplexCompute>(tH(ROW_IDX_H, iCol, NXT_ABS_FREQ_IDX, chEqTimeInstIdx));

#ifdef ENABLE_DEBUG
                printf("H[%d][%d] = %f+j%f\n", ROW_IDX_H, iCol, shH(ROW_IDX_H, iCol, currIdx ^ 1).x, shH(ROW_IDX_H, iCol, currIdx ^ 1).y);
#endif
            }
        }

        // A thread can process elements different from what it loaded, so wait for all threads in the thread block
        // to complete
        thisThrdBlk.sync();

        //---------------------------------------------------------------------------------------------------
        // Stage2: Compute the enhanced Gram matrix: M = H'*inv(Rww) and G = (H'*inv(Rww)*H + inv(Rxx))

        // Compute M = H'*inv(Rww): N_COLS_H x N_BS_ANTS = N_LAYERS x N_BS_ANTS
        // Select a subset of the threads in the warp to access the columns of RwwInv and each thread computes one
        // column of M
        if(THREAD_X_IDX < N_BS_ANTS)
        {
            // Each thread computes one column of M
#pragma unroll
            for(uint32_t j = 0; j < N_COLS_H; ++j)
            {
                shM(j, THREAD_X_IDX) = type_convert<TComplexCompute32>(cuConj(shH(THREAD_X_IDX, j, currIdx)) * invDiagReg);

#ifdef ENABLE_DEBUG
                printf("M[%d][%d] = %f+j%f\n", j, THREAD_X_IDX, shM(j, THREAD_X_IDX).x, shM(j, THREAD_X_IDX).y);
#endif
                // tDbg(j,THREAD_X_IDX,ABS_FREQ_IDX,chEqTimeInstIdx) =
                  //    type_convert<TComplexStorageOut>(shM(j,THREAD_X_IDX));
            }
        }

        // Wait for matrix M computation to finish, before using it in computation of G
        // thisThrdBlk.sync();
        __syncthreads();

        // Compute G = (M*H + RxxInv): N_LAYERS x N_LAYERS
#pragma unroll
        for(uint32_t i = 0; i < N_ITER_TO_COMPUTE_G; ++i)
        {
            uint32_t iCol = i * N_COLS_G_COMPUTE_PER_ITER + COL_IDX_G;

            // All threads may not participate in the last iteration
            if((ROW_IDX_G < N_ROWS_G) && (iCol < N_COLS_G))
            {
                TComplexCompute G{};
                // if (threadIdx.x == 0 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0) printf("G={%f, %f}, i=%d\n", (float)G.x, (float)G.y, i);

#pragma unroll
                for(uint32_t elem = 0; elem < N_BS_ANTS; ++elem)
                {
                    // TComplexCompute32 H32 = type_convert<TComplexCompute32>(shH(elem, iCol, currIdx));
                    // TComplexCompute32 M32 = shM(ROW_IDX_G, elem);
                    TComplexCompute M16 = type_convert<TComplexCompute>(__float22half2_rn(shM(ROW_IDX_G, elem)));
                    TComplexCompute H16 = shH(elem, iCol, currIdx);
                    G = cuFma(M16, H16, G);
                    // if (THREAD_X_IDX==55 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0 && f==0) {
                    //     printf("tid=%d, M={%f, %f}, H={%f, %f}    G[%d,%d]={%f, %f}, elem=%d, f=%d\n", THREAD_X_IDX, shM(ROW_IDX_G, elem).x, shM(ROW_IDX_G, elem).y, H32.x, H32.y, ROW_IDX_G, iCol, G.x, G.y, elem, f);
                    //     // printf("G={%f, %f}, elem=%d, f=%d\n", (float)G.x, (float)G.y, elem, f);
                    // }
                }

                // Add Noise_pwr*inv(Rxx) to the diagonal of H'*H, inv(Rxx) is assumed to be unity
                if(ROW_IDX_G == iCol)
                {
                    G += ES_INV;
                }
                shG(ROW_IDX_G, iCol) = type_convert<TComplexCompute32>(G);

#ifdef ENABLE_DEBUG
                printf("G[%d][%d] = %f+j%f\n", ROW_IDX_G, iCol, shG(ROW_IDX_G, iCol).x, shG(ROW_IDX_G, iCol).y);
#endif

                // tDbg(ROW_IDX_G,iCol,ABS_FREQ_IDX,chEqTimeInstIdx) =
                //   type_convert<TComplexStorageOut>(shG(ROW_IDX_G,iCol));
            }

            // Initialize matrix I (which has the same dimenison as G)
            if(iCol < N_COLS_I)
            {
                    if(N_COLS_I > 1)
                    {
                        shI(ROW_IDX_I, iCol) =
                        (ROW_IDX_I != iCol) ? cuGet<TComplexCompute32>(0) : cuGet<TComplexCompute32>(1);
                    }
                    else
                    {
                        shI(ROW_IDX_I, iCol) = cuGet<TComplexCompute32>(1);
                    }

            }
        }

        // if((1 == f) && (COL_IDX_RWW < N_COLS_RWW)) tDbg(ROW_IDX_RWW,COL_IDX_RWW,PRB_IDX, chEqTimeInstIdx) = shRwwInv(ROW_IDX_RWW,COL_IDX_RWW);
        // tDbg(ROW_IDX_H,COL_IDX_H,THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shH(ROW_IDX_H,COL_IDX_H,currIdx);
        // tDbg(ROW_IDX_M,COL_IDX_M,THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shM(ROW_IDX_M,COL_IDX_M);

        // Wait for matrix G computation to finish, before using it in computation of Ginv
        thisThrdBlk.sync();

        //---------------------------------------------------------------------------------------------------
        // Stage3: Perform joint LU factorization
        // A = [ G | I | M ] -> [ U | Linv | F ]
        // where U = L\G, Linv = L\I, F = L\M

        // eq_mmse_coef_comp_high_mimo_kernel_v2: thread block size == # of columns of augmented matrix
        // (thread block size = row length of augmented matrix = (2*N_LAYERS + N_BS_ANTS)).
        // Thus no benefit in using the parallel version of factorization algorithm (for sub-matrix updates)
        // if (threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0 && f==0) {
        //     printf("shA[0, %d]={%f, %f}\n", threadIdx.x, (float)shA(0, threadIdx.x).x, (float)shA(0, threadIdx.x).y);
        // }
        // if (threadIdx.x<64 && threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0 && f==0) {
        //     printf("shM[0, %d]={%f, %f}\n", threadIdx.x, (float)shM(0, threadIdx.x).x, (float)shM(0, threadIdx.x).y);
        // }
        luFactorizeIter<TCompute32, N_ROWS_A, N_COLS_A>(thisThrdBlk, shA);
        // if (threadIdx.y==0 && blockIdx.x==0 && blockIdx.y==0 && f==0) {
        //     printf("shA[0, %d]={%f, %f}\n", threadIdx.x, (float)shA(0, threadIdx.x).x, (float)shA(0, threadIdx.x).y);
        // }
        // luFactorizeParallel_v1<TCompute, N_ROWS_A, N_COLS_A>(thisThrdBlk, shA);
        // luFactorizeParallel_v2<TCompute, N_ROWS_A, N_COLS_A>(thisThrdBlk, shA);

        //---------------------------------------------------------------------------------------------------
        // Stage4: Solve by back substitution, compute residual error covariance matrix Ree as the inverse
        // of Gram matrix: Ginv = Ree and the MMSE coefficient matrix C
        // Solve U*[ Ree | C ] = [ Linv | F ] for Ginv and C where Ginv = U\(Linv) and C = U\F

        backSub<TCompute32, N_ROWS_U, N_COLS_U, N_ROWS_A, N_COLS_A>(thisThrdBlk, START_COL_OFFSET_REE, shU, shA);

        //--------------------------------------------------------------------------------------------------------
        // Stage5: Write the results (Ree and C) into device memory

#pragma unroll
        for(uint32_t i = 0; i < N_ITER_TO_WR_C; ++i)
        {
            uint32_t iCol = i * N_COLS_C_WR_PER_ITER + COL_IDX_C;

            // All threads may not participate in the last iteration
            if(iCol < N_COLS_C)
            {
                // Compute bias correction factor lambda and apply to coefficients
                // TCompute32 lambda = cuGet<TCompute32>(1) / (cuGet<TCompute32>(1) - type_convert<TCompute32>(cuReal(shRee(ROW_IDX_C, ROW_IDX_C))));
                TCompute32 one = cuGet<TCompute32>(1);
                TCompute32 lambda = one;
                TCompute32 realRee = type_convert<TCompute32>(cuReal(shRee(ROW_IDX_C, ROW_IDX_C)));
                if(realRee < one)
                {
                    lambda = one / (one - realRee);
                }

                tCoef(iCol, PRB_IDX * 12 + FREQ_IDX, ROW_IDX_C, chEqTimeInstIdx) =
                    type_convert<TComplexStorageOut>(shC(ROW_IDX_C, iCol) * lambda);
            }
        }

        if(THREAD_X_IDX < N_ROWS_REE)
        {
            // Compute ReeInv while applying bias correction
            // TCompute reeInv = (cuGet<TCompute>(1)/shMemBlkReeDiag(THREAD_X_IDX, FREQ_IDX)) - cuGet<TCompute>(1);
            TCompute32 ree    = cuReal(shRee(THREAD_X_IDX, THREAD_X_IDX));
            TCompute32 reeInv = (cuGet<TCompute32>(1) - ree) / ree;

            // saturate error precision
            if(reeInv > MAX_ERROR_PRECISION)
                reeInv = MAX_ERROR_PRECISION;

            tReeDiagInv(FREQ_IDX, THREAD_X_IDX, PRB_IDX, chEqTimeInstIdx) = type_convert<TStorageOut>(reeInv);
        }

#ifdef ENABLE_DEBUG
        printf("C[%d][%d][%d] = %f+j%f\n", ABS_FREQ_IDX, ROW_IDX_C, COL_IDX_C, shC(ROW_IDX_C, COL_IDX_C).x, shC(ROW_IDX_C, COL_IDX_C).y);
#endif
        // thisThrdBlk.sync();
        // currIdx ^= 1;
    }
}

#ifndef FAST_COMPILE
template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS,        // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS,         // # of layers (# of cols in H matrix)
          uint32_t N_THRD_BLK_TONES, // # of frequency bins processed by a thread block
          uint32_t N_TONES_PER_ITER> // # of frequency bins processed in 1 iteration by the thread block
__global__ void
eqMmseCoefCompHighMimoKernel(puschRxChEqStatDescr_t* pStatDescr, puschRxChEqCoefCompDynDescr_t* pDynDescr)
{
  // Only for 64 ants
    if (N_BS_ANTS == 64)
    {
        eqMmseCoefCompHighMimoKernel_v3<TStorageIn,
                                        TStorageOut,
                                        TCompute,
                                        N_BS_ANTS,
                                        N_LAYERS,
                                        N_THRD_BLK_TONES,
                                        N_TONES_PER_ITER>(pStatDescr, pDynDescr);
    }
    else
    // v1 has a better memory BW utilization (85% v1 vs 71% v2) but 4us (58us vs 62us) slower for 100MHz (273PRB) usecase
#if(EQ_COEF_COMP_H_MIMO_VER == 1)
  {
    eqMmseCoefCompHighMimoKernel_v1<TStorageIn,
                                    TStorageOut,
                                    TCompute,
                                    N_BS_ANTS,
                                    N_LAYERS,
                                    N_THRD_BLK_TONES,
                                    N_TONES_PER_ITER>(pStatDescr, pDynDescr);
  }
#elif(EQ_COEF_COMP_H_MIMO_VER == 2)
  {
    eqMmseCoefCompHighMimoKernel_v2<TStorageIn,
                                    TStorageOut,
                                    TCompute,
                                    N_BS_ANTS,
                                    N_LAYERS,
                                    N_THRD_BLK_TONES,
                                    N_TONES_PER_ITER>(pStatDescr, pDynDescr);
  }
#endif
}

// Equalization kernel for massive MIMO
// {N_LAYERS, N_BS_ANTS} = {16,64}
// Inputs and outputs assumed to be column major
// dimBlock: (N_BS_ANTS*N_LAYERS, N_TONES_PER_ITER)
// dimGrid : ((CUPHY_N_TONES_PER_PRB/N_THRD_BLK_TONES), Nprb)
template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS,        // # of BS antenna (# of rows in H matrix)  // used in block_3D(). // used for too many instances requireing constant expressions.
          uint32_t N_LAYERS,         // # of layers (# of cols in H matrix)
          uint32_t N_THRD_BLK_TONES, // # of frequency bins processed by a thread block
          uint32_t N_TONES_PER_ITER> // # of frequency bins processed in 1 iteration by the thread block

__global__ void
eqMmseCoefCompMassiveMimoKernel_v1(puschRxChEqStatDescr_t* pStatDescr, puschRxChEqCoefCompDynDescr_t* pDynDescr)
{
    //--------------------------------------------------------------------------------------------------------
    // Setup local parameters based on descriptor
    //  puschRxChEqStatDescr_t&        statDescr = *(pStatDescr);
    puschRxChEqCoefCompDynDescr_t& dynDescr = *(pDynDescr);

    // Early exit check
    // The grid is sized to process the max # of PRBs in a given heterogenous config. Exit if the PRB to be
    // processed by this thread block does not exist in the UE group
    const uint32_t PRB_IDX = blockIdx.y;
    const uint32_t UE_GRP_IDX = dynDescr.hetCfgUeGrpMap[blockIdx.z];    

    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];

    const uint16_t nPrb = drvdUeGrpPrms.nPrb;
    if(PRB_IDX >= nPrb) return;

    //--------------------------------------------------------------------------------------------------------
    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    // clang-format off
    tensor_ref<const TComplexStorageIn> tH         (drvdUeGrpPrms.tInfoHEst.pAddr       , drvdUeGrpPrms.tInfoHEst.strides      );// (N_BS_ANTS, N_LAYERS, NF, NH)
    tensor_ref<TStorageOut>             tReeDiagInv(drvdUeGrpPrms.tInfoReeDiagInv.pAddr , drvdUeGrpPrms.tInfoReeDiagInv.strides);// (N_SC, N_LAYERS, N_PRB, NH)   // (N_LAYERS, NF, NH)
    tensor_ref<TComplexStorageOut>      tCoef      (drvdUeGrpPrms.tInfoEqCoef.pAddr     , drvdUeGrpPrms.tInfoEqCoef.strides      );// (N_SC, N_LAYERS, N_BS_ANTS, N_PRB, NH) // (N_LAYERS, N_BS_ANTS, NF, NH)
    tensor_ref<TComplexStorageOut>      tDbg       (drvdUeGrpPrms.tInfoChEqDbg.pAddr    , drvdUeGrpPrms.tInfoChEqDbg.strides       );
    // clang-format on

    // set diagonal regularizer:
    bool   noiseRegFlag  = (drvdUeGrpPrms.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE) ? true : false;
    float  invDiagReg    = noiseRegFlag ? drvdUeGrpPrms.invNoiseVarLin : INV_ZF_REGULARIZER;

    // H is channel matrix
    // G is the enhanced Gram matrix
    // A is the augmented matrix, A = [ G | I | M ]

    //--------------------------------------------------------------------------------------------------------
    // Dimensions

    // H  : Channel matrix
    constexpr uint32_t N_ROWS_H = N_BS_ANTS;
    constexpr uint32_t N_COLS_H = N_LAYERS;

    // M  : Intermediate matrix, M = H'*RwwInv
    constexpr uint32_t N_ROWS_M = N_COLS_H;   // N_LAYERS
    constexpr uint32_t N_COLS_M = N_BS_ANTS; // N_BS_ANTS

    // G  : Enhanced Gram matrix, G = H'*RwwInv*H + inv(Rxx)
    constexpr uint32_t N_ROWS_G = N_LAYERS;
    constexpr uint32_t N_COLS_G = N_LAYERS;

    // U  : Upper triangular matrix
    constexpr uint32_t N_ROWS_U = N_ROWS_G;
    constexpr uint32_t N_COLS_U = N_COLS_G;

    // I  : Identity matrix, I = G*Ginv
    constexpr uint32_t N_ROWS_I = N_ROWS_G;
    constexpr uint32_t N_COLS_I = N_COLS_G;

    // Ree: Residual error covariance matrix, Ree = Ginv
    constexpr uint32_t N_ROWS_REE = N_ROWS_G;
    // constexpr uint32_t N_COLS_REE = N_COLS_G;

    // C  : MMSE coefficient matrix, C = Ree*H'*RwwInv = Ree*M
    constexpr uint32_t N_ROWS_C = N_ROWS_REE;
    constexpr uint32_t N_COLS_C = N_COLS_M;

    // A  : Augmented result matrix, A = [ G | I | M ] -> [ U | Linv | F ] -> [ U | Ginv | C ]
    constexpr uint32_t N_ROWS_A = N_ROWS_G;
    constexpr uint32_t N_COLS_A = N_COLS_G + N_COLS_I + N_COLS_M;

    // Column offsets to Ree (which is followed by C) matrix within the augmented matrix
    // which form the RHS of the back substition (and after back substitution the RHS is overwritten by the
    // unknowns being solved for)
    constexpr uint32_t START_COL_OFFSET_REE = N_COLS_U;

    constexpr int32_t N_ITER = N_THRD_BLK_TONES / N_TONES_PER_ITER;

    constexpr uint32_t N_INST = 1; // buffering for pipelining

    const uint32_t N_THREADS_X = blockDim.x; // Number of threads needed to process one frequency bin
    // const uint32_t N_THREADS   = blockDim.x * blockDim.y; // N_TONES_PER_ITER == blockDim.y

    const uint32_t N_COLS_M_COMPUTE_PER_ITER = N_THREADS_X / N_ROWS_M;
    const uint32_t N_ITER_TO_COMPUTE_M       = div_round_up(N_COLS_M, N_COLS_M_COMPUTE_PER_ITER);

    const uint32_t N_COLS_G_COMPUTE_PER_ITER = N_THREADS_X / N_ROWS_G;
    const uint32_t N_ITER_TO_COMPUTE_G       = div_round_up(N_COLS_G, N_COLS_G_COMPUTE_PER_ITER);

    const uint32_t N_COLS_C_WR_PER_ITER = N_THREADS_X / N_ROWS_C;
    const uint32_t N_ITER_TO_WR_C       = div_round_up(N_COLS_C, N_COLS_C_WR_PER_ITER);

    // Reciprocal of symbol energy
    const TCompute ES_INV = cuGet<TCompute>(1);

    //--------------------------------------------------------------------------------------------------------
    // Compute indices used for element access
    const uint32_t THREAD_X_IDX = threadIdx.x;
    // const uint32_t THREAD_IDX   = (threadIdx.y * blockDim.x) + threadIdx.x;

    // There are N_TONES_PER_ITER groups of threads with each group containing N_THREADS_X per group
    const uint32_t THRD_GRP_FREQ_OFFSET        = threadIdx.y;
    const uint32_t THRD_BLK_START_FREQ_OFFSET  = (blockIdx.x * N_THRD_BLK_TONES);
    const uint32_t THRD_GRP_START_FREQ_OFFSET  = THRD_BLK_START_FREQ_OFFSET + THRD_GRP_FREQ_OFFSET;
    const uint32_t THRD_BLK_ABS_START_FREQ_IDX = PRB_IDX * CUPHY_N_TONES_PER_PRB;

    const uint32_t chEqTimeInstIdx = dynDescr.chEqTimeInstIdx;
    
    // const uint32_t ROW_IDX_RWW = THREAD_IDX % N_ROWS_RWW;
    // const uint32_t COL_IDX_RWW = THREAD_IDX / N_ROWS_RWW; // COL_IDX_RWW needs a bounds check (since N_THREADS > # of Rww elements)

    // const uint32_t ROW_IDX_H = THREAD_X_IDX % N_ROWS_H;
    // const uint32_t COL_IDX_H = THREAD_X_IDX / N_ROWS_H;

    const uint32_t ROW_IDX_I = THREAD_X_IDX % N_ROWS_I;
    // const uint32_t COL_IDX_I = THREAD_X_IDX / N_ROWS_I;

    const uint32_t ROW_IDX_M = THREAD_X_IDX % N_ROWS_M;
    const uint32_t COL_IDX_M = THREAD_X_IDX / N_ROWS_M;

    const uint32_t ROW_IDX_G = THREAD_X_IDX % N_ROWS_G;
    const uint32_t COL_IDX_G = THREAD_X_IDX / N_ROWS_G; // COL_IDX_G needs a bounds check (since N_THREADS_X > # of G elements)

    // const uint32_t ROW_IDX_REE = THREAD_X_IDX % N_ROWS_REE;
    // const uint32_t COL_IDX_REE = THREAD_X_IDX / N_ROWS_REE; // COL_IDX_REE needs a bounds check (since N_THREADS_X > # of Ree elements)

    const uint32_t ROW_IDX_C = THREAD_X_IDX % N_ROWS_C;
    const uint32_t COL_IDX_C = THREAD_X_IDX / N_ROWS_C;

    //--------------------------------------------------------------------------------------------------------
    // Shared memory allocation
    // H[N_TONES_PER_ITER*N_INST]

    // Shared memory contents as processing progresses:
    // A = [ G | I | M ] -> [ U | Linv | F ] -> [ U | Ree | C ]
    
    constexpr uint32_t N_SMEM_ELEMS = 
        (((N_ROWS_H + 1) * N_COLS_H * N_INST) + // (N_ROWS_H + 1) for SMEM padding to avoid bank conflicts
         (N_ROWS_A + 1) * N_COLS_A) *
        N_TONES_PER_ITER;
    
    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    __shared__ TComplexCompute smemBlk[N_SMEM_ELEMS];

#if 0
    constexpr uint32_t SMEM_START_OFFSET_RWW = 0;

    block_2D<TComplexCompute*, N_ROWS_RWW + 1, N_COLS_RWW> shRwwInv(&smemBlk[SMEM_START_OFFSET_RWW]);

    constexpr uint32_t SMEM_START_OFFSET_H_BLK = SMEM_START_OFFSET_RWW + shRwwInv.num_elem();
    const uint32_t     SMEM_START_OFFSET_H     = SMEM_START_OFFSET_H_BLK + THRD_GRP_FREQ_OFFSET * (N_ROWS_H + 1) * N_COLS_H * N_INST;
#endif
    constexpr uint32_t SMEM_START_OFFSET_H_BLK = 0;
    const uint32_t     SMEM_START_OFFSET_H     = SMEM_START_OFFSET_H_BLK + THRD_GRP_FREQ_OFFSET * (N_ROWS_H + 1) * N_COLS_H * N_INST;

    block_3D<TComplexCompute*, N_ROWS_H + 1, N_COLS_H, N_INST> shH(&smemBlk[SMEM_START_OFFSET_H]);

    constexpr uint32_t SMEM_START_OFFSET_A_BLK = SMEM_START_OFFSET_H_BLK + N_TONES_PER_ITER * shH.num_elem();
    const uint32_t     SMEM_START_OFFSET_A     = SMEM_START_OFFSET_A_BLK + THRD_GRP_FREQ_OFFSET * (N_ROWS_A + 1) * N_COLS_A;

    block_2D<TComplexCompute*, N_ROWS_A + 1, N_COLS_A> shA(&smemBlk[SMEM_START_OFFSET_A]);

    // SMEM overlay: A with [ G | I | M ]
    const uint32_t SMEM_START_OFFSET_G = SMEM_START_OFFSET_A;

    block_2D<TComplexCompute*, N_ROWS_G + 1, N_COLS_G> shG(&smemBlk[SMEM_START_OFFSET_G]);

    const uint32_t SMEM_START_OFFSET_I = SMEM_START_OFFSET_G + shG.num_elem();

    block_2D<TComplexCompute*, N_ROWS_I + 1, N_COLS_I>
        shI(&smemBlk[SMEM_START_OFFSET_I]);

    const uint32_t SMEM_START_OFFSET_M = SMEM_START_OFFSET_I + shI.num_elem();

    block_2D<TComplexCompute*, N_ROWS_M + 1, N_COLS_M>
        shM(&smemBlk[SMEM_START_OFFSET_M]);

    // SMEM overlay: after LU - U replaces G, Linv replaces I and F replaces M
    auto& shU    = shG;
    auto& shLinv = shI;
    auto& shF    = shM;

    // SMEM overlay: after back substitution - Ree replaces Linv and C replaces F
    // (i.e. results are stored inplace)
    auto& shRee = shLinv;
    auto& shC   = shF;

    //--------------------------------------------------------------------------------------------------------
    // Stage1: Load inputs
    thread_block const& thisThrdBlk = this_thread_block();

#ifdef ENABLE_DEBUG
    if(0 != blockIdx.x) return;
#endif

    // Prologue
    // Prefetch H for first iteration
    uint32_t f       = 0;
    uint32_t currIdx = 0;

    const uint32_t FREQ_IDX     = THRD_GRP_START_FREQ_OFFSET + (f * N_TONES_PER_ITER);
    const uint32_t ABS_FREQ_IDX = THRD_BLK_ABS_START_FREQ_IDX + FREQ_IDX;

    block_2D<const typename complex_from_scalar<TStorageIn>::type*, N_ROWS_H, N_COLS_H> srcH(tH.addr + tH.offset(0, 0, ABS_FREQ_IDX, chEqTimeInstIdx));
    block_2D<TComplexCompute*, N_ROWS_H + 1, N_COLS_H>                                  dstH(&shH(0, 0, currIdx));
    cmplxMatLoad<TStorageIn, TCompute, N_ROWS_H, N_COLS_H>(thisThrdBlk, srcH, dstH);

    for(int32_t f = 0; f < N_ITER; ++f)
    {
        const uint32_t FREQ_IDX = THRD_GRP_START_FREQ_OFFSET + (f * N_TONES_PER_ITER);
        //const uint32_t ABS_FREQ_IDX = THRD_BLK_ABS_START_FREQ_IDX + FREQ_IDX;

        // A thread can process elements different from what it loaded, so wait for all threads in the thread block
        // to complete
        thisThrdBlk.sync();

        // Prefetch H for next iteration
        if(f < (N_ITER - 1))
        {
            const uint32_t NXT_FREQ_IDX     = THRD_GRP_START_FREQ_OFFSET + ((f + 1) * N_TONES_PER_ITER);
            const uint32_t NXT_ABS_FREQ_IDX = THRD_BLK_ABS_START_FREQ_IDX + NXT_FREQ_IDX;

            block_2D<const typename complex_from_scalar<TStorageIn>::type*, N_ROWS_H, N_COLS_H> srcH(tH.addr + tH.offset(0, 0, NXT_ABS_FREQ_IDX, chEqTimeInstIdx));
            block_2D<TComplexCompute*, N_ROWS_H + 1, N_COLS_H>                                  dstH(&shH(0, 0, currIdx ^ 1));
            cmplxMatLoad<TStorageIn, TCompute, N_ROWS_H, N_COLS_H>(thisThrdBlk, srcH, dstH);
        }

        //---------------------------------------------------------------------------------------------------
        // Stage2: Compute the enhanced Gram matrix: M = H'*inv(Rww) and G = (H'*inv(Rww)*H + inv(Rxx))

        // Compute M = H'*inv(Rww): N_COLS_H x N_BS_ANTS = N_LAYERS x N_BS_ANTS
#pragma unroll
        for(uint32_t i = 0; i < N_ITER_TO_COMPUTE_M; ++i)
        {
            uint32_t iCol = i * N_COLS_M_COMPUTE_PER_ITER + COL_IDX_M;
            // All threads may not participate in the last iteration
            if((ROW_IDX_M < N_ROWS_M) && (iCol < N_COLS_M))
            {
                TComplexCompute prod{};

                // Compute H'*inv(Rww)
#pragma unroll
                for(uint32_t elem = 0; elem < N_BS_ANTS; ++elem)
                {
                    prod += cuConj(shH(elem, ROW_IDX_M, currIdx)) * invDiagReg;
                }
                shM(ROW_IDX_M, iCol) = prod;

#ifdef ENABLE_DEBUG
                printf("M[%d][%d] = %f+j%f\n", ROW_IDX_M, iCol, shM(ROW_IDX_M, iCol).x, shM(ROW_IDX_M, iCol).y);
#endif
            }
        }

        // Wait for matrix M computation to finish, before using it in computation of G
        thisThrdBlk.sync();

        // if((1 == f) && (COL_IDX_RWW < N_COLS_RWW)) tDbg(ROW_IDX_RWW,COL_IDX_RWW,PRB_IDX, chEqTimeInstIdx) = shRwwInv(ROW_IDX_RWW,COL_IDX_RWW);
        // tDbg(ROW_IDX_H,COL_IDX_H,THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shH(ROW_IDX_H,COL_IDX_H,currIdx);
        // tDbg(ROW_IDX_M,COL_IDX_M,THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shM(ROW_IDX_M,COL_IDX_M);

        // Compute G = (M*H + RxxInv): N_LAYERS x N_LAYERS
#pragma unroll
        for(uint32_t i = 0; i < N_ITER_TO_COMPUTE_G; ++i)
        {
            uint32_t iCol = i * N_COLS_G_COMPUTE_PER_ITER + COL_IDX_G;

            // All threads may not participate in the last iteration
            if((ROW_IDX_G < N_ROWS_G) && (iCol < N_COLS_G))
            {
                TComplexCompute G{};

#pragma unroll
                for(uint32_t elem = 0; elem < N_BS_ANTS; ++elem)
                {
                    G = cuFma(shM(ROW_IDX_G, elem), shH(elem, iCol, currIdx), G);
                }

                // Add Noise_pwr*inv(Rxx) to the diagonal of H'*H, inv(Rxx) is assumed to be unity
                if(ROW_IDX_G == iCol)
                {
                    G += ES_INV;
                }
                shG(ROW_IDX_G, iCol) = G;

#ifdef ENABLE_DEBUG
                printf("G[%d][%d] = %f+j%f\n", ROW_IDX_G, iCol, shG(ROW_IDX_G, iCol).x, shG(ROW_IDX_G, iCol).y);
#endif

                // tDbg(ROW_IDX_G,iCol,ABS_FREQ_IDX,chEqTimeInstIdx) =
                //   type_convert<TComplexStorageOut>(shG(ROW_IDX_G,iCol));
            }

            // Initialize matrix I (which has the same dimenison as G)
            if(iCol < N_COLS_I)
            {
                shI(ROW_IDX_I, iCol) =
                    (ROW_IDX_I != iCol) ? cuGet<TComplexCompute>(0) : cuGet<TComplexCompute>(1);
            }
        }

        // Wait for matrix G computation to finish, before using it in computation of Ginv
        thisThrdBlk.sync();

#ifdef ENABLE_DEBUG
        // A0
        for(uint32_t i = 0; i < N_ROWS_A; ++i)
        {
            if(THREAD_X_IDX < N_COLS_A)
                tDbg(i, THREAD_X_IDX, THRD_BLK_ABS_START_FREQ_IDX + FREQ_IDX, chEqTimeInstIdx) = type_convert<TComplexStorageOut>(shA(i, THREAD_X_IDX));
        }
#endif

        //---------------------------------------------------------------------------------------------------
        // Stage3: Perform joint LU factorization
        // A = [ G | I | M ] -> [ U | Linv | F ]
        // where U = L\G, Linv = L\I, F = L\M

        // eq_mmse_coef_comp_massive_mimo_kernel_v1: thread block size >> # of columns of augmented matrix
        // (i.e. (N_LAYERS * N_LAYERS) >> (2*N_LAYERS + N_BS_ANTS)). Thus use parallel version of the
        // factorization algorithm to cut down iteration count and increase active threads during sub-matrix
        // updates
        // luFactorizeIter<TCompute, N_ROWS_A, N_COLS_A>(thisThrdBlk, shA);
        luFactorizeParallel_v2<TCompute, N_ROWS_A, N_COLS_A>(thisThrdBlk, shA);
#ifdef ENABLE_DEBUG
        // A1
        for(uint32_t i = 0; i < N_ROWS_A; ++i)
        {
            if(THREAD_X_IDX < N_COLS_A)
                tDbg(i, THREAD_X_IDX, THRD_BLK_ABS_START_FREQ_IDX + FREQ_IDX, chEqTimeInstIdx) = type_convert<TComplexStorageOut>(shA(i, THREAD_X_IDX));
        }
#endif

        //---------------------------------------------------------------------------------------------------
        // Stage4: Solve by back substitution, compute residual error covariance matrix Ree as the inverse
        // of Gram matrix: Ginv = Ree and the MMSE coefficient matrix C
        // Solve U*[ Ree | C ] = [ Linv | F ] for Ginv and C where Ginv = U\(Linv) and C = U\F

        backSub<TCompute, N_ROWS_U, N_COLS_U, N_ROWS_A, N_COLS_A>(thisThrdBlk, START_COL_OFFSET_REE, shU, shA);
#ifdef ENABLE_DEBUG
        // A2
        for(uint32_t i = 0; i < N_ROWS_A; ++i)
        {
            if(THREAD_X_IDX < N_COLS_A)
                tDbg(i, THREAD_X_IDX, THRD_BLK_ABS_START_FREQ_IDX + FREQ_IDX, chEqTimeInstIdx) = type_convert<TComplexStorageOut>(shA(i, THREAD_X_IDX));
        }
#endif

        //--------------------------------------------------------------------------------------------------------
        // Stage5: Write the results (Ree and C) into device memory
#pragma unroll
        for(uint32_t i = 0; i < N_ITER_TO_WR_C; ++i)
        {
            uint32_t iCol = i * N_COLS_C_WR_PER_ITER + COL_IDX_C;

            // All threads may not participate in the last iteration
            if(iCol < N_COLS_C)
            {
                // Compute bias correction factor lambda and apply to coefficients
                // TCompute lambda = cuGet<TCompute>(1) / (cuGet<TCompute>(1) - cuReal(shRee(ROW_IDX_C, ROW_IDX_C)));
                TCompute one = cuGet<TCompute>(1);
                TCompute lambda = one;
                TCompute realRee = cuReal(shRee(ROW_IDX_C, ROW_IDX_C));
                if(realRee < one)
                {
                    lambda = one / (one - realRee);
                }

                tCoef(iCol, PRB_IDX * 12 + FREQ_IDX, ROW_IDX_C, chEqTimeInstIdx) =
                    type_convert<TComplexStorageOut>(shC(ROW_IDX_C, iCol) * lambda);
            }
        }

        if(THREAD_X_IDX < N_ROWS_REE)
        {
            // Compute ReeInv while applying bias correction
            // TCompute reeInv = (cuGet<TCompute>(1)/shMemBlkReeDiag(THREAD_X_IDX, FREQ_IDX)) - cuGet<TCompute>(1);
            TCompute ree    = cuReal(shRee(THREAD_X_IDX, THREAD_X_IDX));
            TCompute reeInv = (cuGet<TCompute>(1) - ree) / ree;

            // saturate error precision
            if(static_cast<int>(reeInv) > MAX_ERROR_PRECISION)
                reeInv = MAX_ERROR_PRECISION;

            tReeDiagInv(FREQ_IDX, THREAD_X_IDX, PRB_IDX, chEqTimeInstIdx) = type_convert<TStorageOut>(reeInv);
        }

#ifdef ENABLE_DEBUG
        printf("C[%d][%d][%d] = %f+j%f\n", ABS_FREQ_IDX, ROW_IDX_C, COL_IDX_C, shC(ROW_IDX_C, COL_IDX_C).x, shC(ROW_IDX_C, COL_IDX_C).y);
#endif
        currIdx ^= 1;
    }
}
#endif

// Equalizer coefficient compute kernel for low order MIMO per PRB
// {N_LAYERS, N_BS_ANTS} = {1,2}, {2,2}, {1,4}, {2,4}, {4,4}, {1,8}, {2,8} and {4,8}
// Inputs and outputs assumed to be column major
// dimBlock: (8,N_FREQ_BINS_PER_ITER) for N_LAYERS = 2, N_BS_ANTS = 4; (32,N_FREQ_BINS_PER_ITER) for N_LAYERS = 4, N_BS_ANTS = 8
//           N_FREQ_BINS_PER_ITER = 4
//           Essentially, there are N_FREQ_BINS_PER_ITER group of threads, each thread group contains:
//           8 threads for N_LAYERS = 2 and 32 threads for N_LAYERS = 4
// dimGrid : Nprb
template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS,            // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS,             // # of layers (# of cols in H matrix)  // used for static shared allocation
          uint32_t N_FREQ_BINS_PER_ITER> // # of frequency bins processed in 1 iteration by the thread block
__device__ void
eqLegacyMmseCoefCompLowMimoKernel(puschRxChEqStatDescr_t* pStatDescr, puschRxChEqCoefCompDynDescr_t* pDynDescr)
{
    KERNEL_PRINT_GRID_ONCE("%s\n grid = (%u %u %u), block = (%u %u %u), N_BS_ANTS = %u, N_LAYERS = %u, N_FREQ_BINS_PER_ITER = %u\n",
                           __PRETTY_FUNCTION__,
                           gridDim.x, gridDim.y, gridDim.z,
                           blockDim.x, blockDim.y, blockDim.z,
                           N_BS_ANTS,
                           N_LAYERS,
                           N_FREQ_BINS_PER_ITER);

    //--------------------------------------------------------------------------------------------------------
    // Setup local parameters based on descriptor
    // puschRxChEqStatDescr_t&        statDescr = *(pStatDescr);
    puschRxChEqCoefCompDynDescr_t& dynDescr = *(pDynDescr);

    // Early exit check
    // The grid is sized to process the max # of PRBs in a given heterogenous config. Exit if the PRB to be
    // processed by this thread block does not exist in the UE group
    const uint32_t PRB_IDX = blockIdx.x;
    const uint32_t UE_GRP_IDX = dynDescr.hetCfgUeGrpMap[blockIdx.y];
    
    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];

    const uint16_t nPrb = drvdUeGrpPrms.nPrb;
    if(PRB_IDX >= nPrb) return;

    //--------------------------------------------------------------------------------------------------------
    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    // clang-format off
    tensor_ref<const TComplexStorageIn> tH         (drvdUeGrpPrms.tInfoHEst.pAddr       , drvdUeGrpPrms.tInfoHEst.strides       );// (N_BS_ANTS, N_LAYERS, NF, NH)
    tensor_ref<TStorageOut>             tReeDiagInv(drvdUeGrpPrms.tInfoReeDiagInv.pAddr , drvdUeGrpPrms.tInfoReeDiagInv.strides );// (N_SC, N_LAYERS, N_PRB, NH)   // (N_LAYERS, NF, NH)
    tensor_ref<TComplexStorageOut>      tCoef      (drvdUeGrpPrms.tInfoEqCoef.pAddr     , drvdUeGrpPrms.tInfoEqCoef.strides     );// (N_SC, N_LAYERS, N_BS_ANTS, N_PRB, NH) // (N_LAYERS, N_BS_ANTS, NF, NH)
    tensor_ref<const TComplexStorageIn> tCfoEst    (drvdUeGrpPrms.tInfoCfoEst.pAddr     , drvdUeGrpPrms.tInfoCfoEst.strides    ); // (MAX_ND_SUPPORTED, N_LAYERS)
    tensor_ref<TComplexStorageOut>      tDbg       (drvdUeGrpPrms.tInfoChEqDbg.pAddr    , drvdUeGrpPrms.tInfoChEqDbg.strides    );
    // clang-format on

    // set diagonal regularizer:
    bool   noiseRegFlag  = (drvdUeGrpPrms.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE) ? true : false;
    float  invDiagReg    = noiseRegFlag ? drvdUeGrpPrms.invNoiseVarLin : INV_ZF_REGULARIZER;

    uint8_t* dmrsSymLoc = drvdUeGrpPrms.dmrsSymLoc;

    //--------------------------------------------------------------------------------------------------------
    // Dimensions

    // H  : Channel matrix
    constexpr uint32_t N_ROWS_H = N_BS_ANTS;
    constexpr uint32_t N_COLS_H = N_LAYERS;

    // M  : Intermediate matrix, M = H'*RwwInv
    constexpr uint32_t N_ROWS_M = N_COLS_H;   // N_LAYERS
    constexpr uint32_t N_COLS_M = N_BS_ANTS; // N_BS_ANTS

    // G  : Enhanced Gram matrix, G = H'*RwwInv*H + inv(Rxx)
    constexpr uint32_t N_ROWS_G = N_LAYERS;
    constexpr uint32_t N_COLS_G = N_LAYERS;

    // DU : Upper Triangular + Diagonal matrix in G = U'*(D*U)
    // constexpr uint32_t N_ROWS_DU = N_ROWS_G;
    // constexpr uint32_t N_COLS_DU = N_COLS_G;

    // Ree: Residual error covariance matrix, Ree = GInv
    constexpr uint32_t N_ROWS_REE = N_ROWS_G;
    constexpr uint32_t N_COLS_REE = N_COLS_G;

    // C  : MMSE coefficient matrix, C = Ree*H'*RwwInv = Ree*M
    constexpr uint32_t N_ROWS_C = N_ROWS_REE;
    constexpr uint32_t N_COLS_C = N_COLS_M;

    // I  : Identity matrix, I = G*GInv
    constexpr uint32_t N_ROWS_I = N_ROWS_G;
    constexpr uint32_t N_COLS_I = N_COLS_G;

    // J  : Intermediate matrix used in Ree = GInv computation, J = D*U*GInv
    // constexpr uint32_t N_ROWS_J = N_ROWS_DU;
    // constexpr uint32_t N_COLS_J = N_COLS_G;

    // K  : Intermediate matrix used in C (MMSE coefficients) computation, K = D*U*C
    // constexpr uint32_t N_ROWS_K = N_ROWS_DU;
    // constexpr uint32_t N_COLS_K = N_COLS_C;

    // A  : Augmented result matrix, A = [I | M] -> [J | K] -> [Ree | C]
    constexpr uint32_t N_ROWS_A = N_ROWS_I;
    constexpr uint32_t N_COLS_A = N_COLS_I + N_COLS_M;

    // Need to compute:
    // Residual error covariance matrix Ree = GInv and
    // MMSE coefficients C = Ree*H'*RwwInv = Ree*M

    // a. Factorize G = U'*D*U (transforms G to DU)

    // G*GInv = I => U'*D*U*GInv = I
    // C = Ree*M = GInv*M => G*C = M => U'*D*U*C = M

    // concatenating the two problems:
    // U'*D*U*[GInv | C] = [I | M]

    // Set D*U*GInv = J, D*U*C = K
    // b. Forward substitution : U'*[J    | K] = [I | M], solve for J and K
    // c. Backward substitution: DU*[GInv | C] = [J | K], solve for GInv and C

    // const uint32_t N_THREADS_X = blockDim.x;
    // const uint32_t N_THREADS = blockDim.x * blockDim.y; // N_FREQ_BINS_PER_ITER == blockDim.y

    static_assert(N_LAYERS <= N_BS_ANTS, "Recevied layer count should atmost equal base station antenna count");
    static_assert(0 == (CUPHY_N_TONES_PER_PRB % N_FREQ_BINS_PER_ITER), "Number of freq bins processed per iter must be a multiple of PRB size");
    constexpr uint32_t N_ITER_PER_PRB = CUPHY_N_TONES_PER_PRB / N_FREQ_BINS_PER_ITER;

    // Reciprocal of symbol energy
    const TCompute ES_INV = cuGet<TCompute>(1);

    //--------------------------------------------------------------------------------------------------------
    // Compute indices used for element access
    const uint32_t THREAD_X_IDX = threadIdx.x;
    // const uint32_t THREAD_IDX   = (threadIdx.y * blockDim.x) + threadIdx.x;

    // There are N_FREQ_BINS_PER_ITER groups of threads with each group containing N_THREADS_X per group
    const uint32_t THRD_GRP_FREQ_OFFSET        = threadIdx.y;
    const uint32_t THRD_BLK_ABS_START_FREQ_IDX = PRB_IDX * CUPHY_N_TONES_PER_PRB;

    const uint32_t chEqTimeInstIdx = dynDescr.chEqTimeInstIdx;

    // const uint32_t ROW_IDX_RWW = THREAD_IDX % N_ROWS_RWW;
    // const uint32_t COL_IDX_RWW = THREAD_IDX / N_ROWS_RWW; // COL_IDX_RWW needs a bounds check (since N_THREADS > # of Rww elements)

    const uint32_t ROW_IDX_H = THREAD_X_IDX % N_ROWS_H;
    const uint32_t COL_IDX_H = THREAD_X_IDX / N_ROWS_H;

    const uint32_t ROW_IDX_I = THREAD_X_IDX % N_ROWS_I;
    const uint32_t COL_IDX_I = THREAD_X_IDX / N_ROWS_I;

    const uint32_t ROW_IDX_M = THREAD_X_IDX % N_ROWS_M;
    const uint32_t COL_IDX_M = THREAD_X_IDX / N_ROWS_M;

    const uint32_t ROW_IDX_G = THREAD_X_IDX % N_ROWS_G;
    const uint32_t COL_IDX_G = THREAD_X_IDX / N_ROWS_G; // COL_IDX_G needs a bounds check (since N_THREADS_X > # of G elements)

    // const uint32_t ROW_IDX_REE = THREAD_X_IDX % N_ROWS_REE;
    // const uint32_t COL_IDX_REE = THREAD_X_IDX / N_ROWS_REE; // COL_IDX_REE needs a bounds check (since N_THREADS_X > # of Ree elements)

    const uint32_t ROW_IDX_C = THREAD_X_IDX % N_ROWS_C;
    const uint32_t COL_IDX_C = THREAD_X_IDX / N_ROWS_C;

    //--------------------------------------------------------------------------------------------------------
    // Shared memory allocation

    // SMEM allocation: C[N_TONES_PER_PRB]        , ReeDiag[N_TONES_PER_PRB],
    //                  Rww[1]                    ,
    //                  DInv[N_FREQ_BINS_PER_ITER], A[N_FREQ_BINS_PER_ITER]    , G[N_FREQ_BINS_PER_ITER]
    // @tdoo: ReeDiag and DInv are of type TCompute, consider allocating in TCompute and castinf to TComplexCompute
    constexpr uint32_t N_SMEM_ELEMS =
        (((N_ROWS_C + 1) * N_COLS_C) + N_ROWS_REE) * CUPHY_N_TONES_PER_PRB +
        (N_LAYERS + ((N_ROWS_G + 1) * N_COLS_G) + ((N_ROWS_A + 1) * N_COLS_A)) * N_FREQ_BINS_PER_ITER;

    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    __shared__ TComplexCompute smemBlk[N_SMEM_ELEMS];

    // H overlays C (note: while H and C both have the same size they have different dimensions. Since
    // N_COLS_C > N_COLS_H, padding the rows of C by 1 makes C bigger overall, thus H start is aligned to C
    // start boundary and not vice-versa)
    constexpr uint32_t                                                        SMEM_START_OFFSET_C_BLK = 0;
    block_3D<TComplexCompute*, N_ROWS_C + 1, N_COLS_C, CUPHY_N_TONES_PER_PRB> shMemBlkC(&smemBlk[SMEM_START_OFFSET_C_BLK]);

    constexpr uint32_t                                     SMEM_START_OFFSET_REE_DIAG_BLK = SMEM_START_OFFSET_C_BLK + shMemBlkC.num_elem();
    block_2D<TCompute*, N_ROWS_REE, CUPHY_N_TONES_PER_PRB> shMemBlkReeDiag(reinterpret_cast<TCompute*>(&smemBlk[SMEM_START_OFFSET_REE_DIAG_BLK]));

    const uint32_t                SMEM_START_OFFSET_DINV_BLK = SMEM_START_OFFSET_REE_DIAG_BLK + shMemBlkReeDiag.num_elem();
    const uint32_t                SMEM_START_OFFSET_DINV     = SMEM_START_OFFSET_DINV_BLK + THRD_GRP_FREQ_OFFSET * N_LAYERS;
    block_1D<TCompute*, N_LAYERS> shDInv(reinterpret_cast<TCompute*>(&smemBlk[SMEM_START_OFFSET_DINV]));

    const uint32_t                                     SMEM_START_OFFSET_G_BLK = SMEM_START_OFFSET_DINV_BLK + N_FREQ_BINS_PER_ITER * shDInv.num_elem();
    const uint32_t                                     SMEM_START_OFFSET_G     = SMEM_START_OFFSET_G_BLK + THRD_GRP_FREQ_OFFSET * (N_ROWS_G + 1) * N_COLS_G;
    block_2D<TComplexCompute*, N_ROWS_G + 1, N_COLS_G> shG(&smemBlk[SMEM_START_OFFSET_G]);

    const uint32_t                                     SMEM_START_OFFSET_A_BLK = SMEM_START_OFFSET_G_BLK + N_FREQ_BINS_PER_ITER * shG.num_elem();
    const uint32_t                                     SMEM_START_OFFSET_A     = SMEM_START_OFFSET_A_BLK + THRD_GRP_FREQ_OFFSET * (N_ROWS_A + 1) * N_COLS_A;
    block_2D<TComplexCompute*, N_ROWS_A + 1, N_COLS_A> shA(&smemBlk[SMEM_START_OFFSET_A]);

    // I and M overlay on A
    const uint32_t                                     SMEM_START_OFFSET_I = SMEM_START_OFFSET_A;
    block_2D<TComplexCompute*, N_ROWS_I + 1, N_COLS_I> shI(&smemBlk[SMEM_START_OFFSET_I]);

    const uint32_t                                     SMEM_START_OFFSET_M = SMEM_START_OFFSET_I + shI.num_elem();
    block_2D<TComplexCompute*, N_ROWS_M + 1, N_COLS_M> shM(&smemBlk[SMEM_START_OFFSET_M]);

    // SMEM overlays
    auto& shDU = shG; // G and DU have the same dimension

    bool     enableCfoCorrection = (0 != drvdUeGrpPrms.enableCfoCorrection) ? true : false;
    uint8_t* pUeGrpLayerToUeIdx  = drvdUeGrpPrms.ueGrpLayerToUeIdx;    
    const uint16_t dmrsMaxLen = drvdUeGrpPrms.dmrsMaxLen;

    // Overlays on A: A = [I | M] -> [J | K] -> [Ree | C]
    // I, J and Ree have the same dimension and are overlaid
    // M, K and C have the same dimension and are overlaid
    // auto& shJ   = shI;
    // auto& shRee = shJ;

    // auto& shK   = shM;
    // auto& shC   = shK;

    thread_block const& thisThrdBlk = this_thread_block();

#ifdef ENABLE_DEBUG
    if(0 != blockIdx.x) return;
#endif


#ifdef ENABLE_DEBUG
     if((0 == blockIdx.x) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     printf("%s\n grid = (%u %u %u) block = (%u %u %u) blockIdx.y %d UE_GRP_IDX %d hetCfgUeGrpMap[0] %d hetCfgUeGrpMap[1] %d hetCfgUeGrpMap[2] %d\n", __PRETTY_FUNCTION__, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.y, UE_GRP_IDX, dynDescr.hetCfgUeGrpMap[0], dynDescr.hetCfgUeGrpMap[1], dynDescr.hetCfgUeGrpMap[2]);

     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     {
         printf("Addr: tH %lp tReeDiagInv %lp tCoef %lp\n", tH.addr, tReeDiagInv.addr, tCoef.addr);
  
         printf("nPrb     : %d \n", nPrb);
         printf("tH          : addr %lp strides[0] %d strides[1] %d strides[2] %d strides[3] %d\n", static_cast<const TComplexStorageIn*>(tH.addr)     , tH.strides[0]         , tH.strides[1]         , tH.strides[2]         , tH.strides[3]         );
         printf("tReeDiagInv : addr %lp strides[0] %d strides[1] %d strides[2] %d strides[3] %d\n", static_cast<TStorageOut*>(tReeDiagInv.addr)        , tReeDiagInv.strides[0], tReeDiagInv.strides[1], tReeDiagInv.strides[2], tReeDiagInv.strides[3]);
         printf("tCoef       : addr %lp strides[0] %d strides[1] %d strides[2] %d strides[3] %d\n", static_cast<TComplexStorageOut*>(tCoef.addr)       , tCoef.strides[0]      , tCoef.strides[1]      , tCoef.strides[2]      , tCoef.strides[3]      );
         // printf("tDbg    strides[0] %d strides[1] %d strides[2] %d\n", tDbg.strides[0], tDbg.strides[1], tDbg.strides[2]);
     }
#endif     

    //--------------------------------------------------------------------------------------------------------
    // 1. Prefetch H into shared memory

    // Read H for whole PRB in one burst. Each N_THREADS_X group of threads can read one H matrix
#pragma unroll
    for(uint32_t f = 0; f < N_ITER_PER_PRB; ++f)
    {
        const uint32_t FREQ_IDX     = (f * N_FREQ_BINS_PER_ITER + THRD_GRP_FREQ_OFFSET);
        const uint32_t ABS_FREQ_IDX = THRD_BLK_ABS_START_FREQ_IDX + FREQ_IDX;

        // SMEM overlay, align H start to C start boundary
        const uint32_t                                     SMEM_START_OFFSET_C = SMEM_START_OFFSET_C_BLK + FREQ_IDX * (N_ROWS_C + 1) * N_COLS_C;
        const uint32_t                                     SMEM_START_OFFSET_H = SMEM_START_OFFSET_C;
        block_2D<TComplexCompute*, N_ROWS_H + 1, N_COLS_H> shH(&smemBlk[SMEM_START_OFFSET_H]);

        if(COL_IDX_H < N_COLS_H)
        {             
            // If needed, scale channel estimates by CFO correction
            if(enableCfoCorrection && (chEqTimeInstIdx > 0))
            {
                shH(ROW_IDX_H, COL_IDX_H) =
                        type_convert<TComplexCompute>(tH(ROW_IDX_H, COL_IDX_H, ABS_FREQ_IDX, chEqTimeInstIdx)) *
                        type_convert<TComplexCompute>(tCfoEst(dmrsSymLoc[chEqTimeInstIdx *dmrsMaxLen], pUeGrpLayerToUeIdx[COL_IDX_H]));
            }
            else
            {
                shH(ROW_IDX_H, COL_IDX_H) =
                    type_convert<TComplexCompute>(tH(ROW_IDX_H, COL_IDX_H, ABS_FREQ_IDX, chEqTimeInstIdx));
            }

            // tDbg(ROW_IDX_H,COL_IDX_H,ABS_FREQ_IDX, chEqTimeInstIdx) = shH(ROW_IDX_H,COL_IDX_H);
#ifdef ENABLE_DEBUG
            printf("H[%d][%d][%d] = %f+j%f\n", ABS_FREQ_IDX, ROW_IDX_H, COL_IDX_H, shH(ROW_IDX_H, COL_IDX_H).x, shH(ROW_IDX_H, COL_IDX_H).y);
#endif
        }
    }

    // Wait for loads to complete. Thread(s) processing an entry of H may not be the same ones loading
    // it
    thisThrdBlk.sync();

    for(uint32_t f = 0; f < N_ITER_PER_PRB; ++f)
    {
        const uint32_t FREQ_IDX = (f * N_FREQ_BINS_PER_ITER + THRD_GRP_FREQ_OFFSET);

        const uint32_t                                     SMEM_START_OFFSET_C = SMEM_START_OFFSET_C_BLK + FREQ_IDX * (N_ROWS_C + 1) * N_COLS_C;
        block_2D<TComplexCompute*, N_ROWS_C + 1, N_COLS_C> shCRes(&smemBlk[SMEM_START_OFFSET_C]);

        // SMEM overlay, align H start to C start boundary
        const uint32_t                                     SMEM_START_OFFSET_H = SMEM_START_OFFSET_C;
        block_2D<TComplexCompute*, N_ROWS_H + 1, N_COLS_H> shH(&smemBlk[SMEM_START_OFFSET_H]);

        //---------------------------------------------------------------------------------------------------
        // 2. Compute enhanced Gram matrix: G = (H'*RwwInv*H + RxxInv)



        // Compute intermediate matrix M = H'*RwwInv
        if(COL_IDX_M < N_COLS_M)
        {
            shM(ROW_IDX_M, COL_IDX_M) = cuConj(shH(COL_IDX_M, ROW_IDX_M)) * invDiagReg;
        }

        // Wait for matrix M computation to finish, before using it in computation of G
        thisThrdBlk.sync();

        // if((1 == f) && (COL_IDX_RWW < N_COLS_RWW)) tDbg(ROW_IDX_RWW,COL_IDX_RWW,PRB_IDX, chEqTimeInstIdx) = shRwwInv(ROW_IDX_RWW,COL_IDX_RWW);
        // tDbg(ROW_IDX_H,COL_IDX_H,THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shH(ROW_IDX_H,COL_IDX_H);
        // tDbg(ROW_IDX_M,COL_IDX_M,THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shM(ROW_IDX_M,COL_IDX_M);

        // Compute G = (M*H + RxxInv)
        if(COL_IDX_G < N_COLS_G)
        {
            TComplexCompute G = cuGet<TComplexCompute>(0);

#pragma unroll
            for(uint32_t i = 0; i < N_ROWS_H; ++i)
            {
                G = cuFma(shM(ROW_IDX_G, i), shH(i, COL_IDX_G), G);
            }

            if(ROW_IDX_G == COL_IDX_G)
            {
                G += ES_INV;
            }
            shG(ROW_IDX_G, COL_IDX_G) = G;

#ifdef ENABLE_DEBUG
            printf("After: M[%d][%d][%d] = %f+j%f\n", FREQ_IDX, ROW_IDX_M, COL_IDX_M, shM(ROW_IDX_M, COL_IDX_M).x, shM(ROW_IDX_M, COL_IDX_M).y);
            printf("G[%d][%d][%d] = %f+j%f\n", FREQ_IDX, ROW_IDX_G, COL_IDX_G, shG(ROW_IDX_G, COL_IDX_G).x, shG(ROW_IDX_G, COL_IDX_G).y);
#endif

            // tDbg(ROW_IDX_G,COL_IDX_G,THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shG(ROW_IDX_G,COL_IDX_G);
        }

        // Wait for matrix G computation to finish, before using it in computation of GInv
        thisThrdBlk.sync();

        //-----------------------------------------------------------------------------------------------
        // 3. Compute residual error covariance matrix Ree as the inverse of Gram matrix: GInv = Ree

        // 3a. LDL factorization: G = U'*D*U
        // Single threaded section (only N_FREQ_BINS_PER_ITER threads are active inside the if condition)

        if(0 == THREAD_X_IDX)
        {
#pragma unroll
            for(int32_t i = 0; i < N_LAYERS; ++i)
            {
                // compute ith diagonal entry of diagonal matrix D
                TCompute sum1 = cuGet<TCompute>(0);
                for(int32_t j = 0; j < i; ++j)
                {
                    sum1 += (cuReal(shDU(j, j)) * cuReal(cuConj(shDU(j, i)) * shDU(j, i)));

#ifdef ENABLE_DEBUG
                    printf("FREQ_IDX %d Row %d Iter %d DU[%d][%d][%d] = %f+j%f, DU[%d][%d][%d] = %f+j%f, sum1 = %f\n", FREQ_IDX, i, j, FREQ_IDX, j, j, shDU(j, j).x, shDU(j, j).y, FREQ_IDX, j, i, shDU(j, i).x, shDU(j, i).y, sum1);
#endif
                }
                shDU(i, i) = cuGet<TComplexCompute>(cuReal(shG(i, i)) - sum1);
                shDInv(i)  = cuGet<TCompute>(1) / cuReal(shDU(i, i));

                // tDbg(i,i,THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shDU(i,i);

#ifdef ENABLE_DEBUG
                printf("DU[%d][%d][%d] = %f+j%f, DInv[%d] = %f\n", FREQ_IDX, i, i, shDU(i, i).x, shDU(i, i).y, i, shDInv(i));
#endif
                // compute upper diagonal elements of ith row of matrix U (U is an upper triangular matrix)
                for(int32_t j = i + 1; j < N_LAYERS; ++j)
                {
                    TComplexCompute sum2 = cuGet<TComplexCompute>(0);
                    for(int32_t k = 0; k < i; ++k)
                    {
                        sum2 += ((cuConj(shDU(k, i)) * shDU(k, j)) * cuReal(shDU(k, k)));
                    }
                    shDU(i, j) = (shG(i, j) - sum2) * shDInv(i);

                    // tDbg(i,j,THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shDU(i,j);
#ifdef ENABLE_DEBUG
                    printf("FREQ_IDX %d Row %d Col %d DU[%d][%d][%d] = %f+j%f, sum2 = %f+j%f\n", FREQ_IDX, i, j, FREQ_IDX, i, j, shDU(i, j).x, shDU(i, j).y, sum2.x, sum2.y);
#endif
                }
            }
        }

        if(COL_IDX_I < N_COLS_I)
        {
            shI(ROW_IDX_I, COL_IDX_I) =
                (ROW_IDX_I != COL_IDX_I) ? cuGet<TComplexCompute>(0) : cuGet<TComplexCompute>(1);
        }

        // Wait for LDL factorization to complete
        thisThrdBlk.sync();

#ifdef ENABLE_DEBUG
        for(int32_t i = 0; i < N_ROWS_A; ++i)
        {
            const uint32_t COL_IDX_A = THREAD_X_IDX;
            if((COL_IDX_A < N_COLS_A) && (0 == FREQ_IDX)) printf("A[%d][%d][%d] = %f+j%f\n", FREQ_IDX, i, COL_IDX_A, shA(i, COL_IDX_A).x, shA(i, COL_IDX_A).y);
        }

        if((COL_IDX_I < N_COLS_I) && (0 == FREQ_IDX)) printf("I[%d][%d][%d] = %f+j%f\n", FREQ_IDX, ROW_IDX_I, COL_IDX_I, shI(ROW_IDX_I, COL_IDX_I).x, shI(ROW_IDX_I, COL_IDX_I).y);
        if((COL_IDX_M < N_COLS_M) && (0 == FREQ_IDX)) printf("M[%d][%d][%d] = %f+j%f\n", FREQ_IDX, ROW_IDX_M, COL_IDX_M, shM(ROW_IDX_M, COL_IDX_M).x, shM(ROW_IDX_M, COL_IDX_M).y);
#endif

        // 3b. Forward substitution: U'*[J | K] = [I | M], solve for J, K
        const uint32_t COL_IDX_A = THREAD_X_IDX;
        if(COL_IDX_A < N_COLS_A)
        {
#pragma unroll
            for(int32_t i = 0; i < N_LAYERS; ++i)
            {
                // Compute row i of intermediate matrices J, K
                TComplexCompute sum = cuGet<TComplexCompute>(0);
                for(int32_t j = 0; j < i; ++j)
                {
                    sum += (cuConj(shDU(j, i)) * shA(j, COL_IDX_A));
                }

                shA(i, COL_IDX_A) = shA(i, COL_IDX_A) - sum;
                // tDbg(i, COL_IDX_A, THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shA(i, COL_IDX_A);
            }
        }

        // Wait for Forward substitution to complete
        thisThrdBlk.sync();

        // 3c. Backward substitution: D*U*[GInv | C] = [J | K], solve for GInv, C
        if(COL_IDX_A < N_COLS_A)
        {
#pragma unroll
            for(int32_t i = N_LAYERS - 1; i >= 0; --i)
            {
                // Compute a row of intermediate matrices J, K
                TComplexCompute sum = cuGet<TComplexCompute>(0);
                for(int32_t j = i + 1; j < N_LAYERS; ++j)
                {
                    sum += (shA(j, COL_IDX_A) * shDU(i, j));
                }

                shA(i, COL_IDX_A) = (shA(i, COL_IDX_A) * shDInv(i)) - sum;
                // tDbg(i, COL_IDX_A, THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shA(i, COL_IDX_A);

#ifdef ENABLE_DEBUG
                printf("A[%d][%d][%d] = %f+j%f\n", FREQ_IDX, i, COL_IDX_A, shA(i, COL_IDX_A).x, shA(i, COL_IDX_A).y);
#endif

                // Store results to be available for writes later
                if((COL_IDX_A >= N_COLS_REE) && (COL_IDX_A < N_COLS_A))
                {
                    const uint32_t COL_IDX_C_DERIVED = COL_IDX_A - N_COLS_REE;
                    shCRes(i, COL_IDX_C_DERIVED)     = shA(i, COL_IDX_A);
                    // tDbg(i, COL_IDX_C_DERIVED, THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shA(i, COL_IDX_A);//  shCRes(i, COL_IDX_C_DERIVED);

#ifdef ENABLE_DEBUG
                    if((0 == blockIdx.x) && (4 == FREQ_IDX)) printf("C[%d][%d][%d] = %f+j%f\n", FREQ_IDX, i, COL_IDX_C_DERIVED, shCRes(i, COL_IDX_C_DERIVED).x, shCRes(i, COL_IDX_C_DERIVED).y);
#endif
                }
                if(COL_IDX_A == i)
                {
                    shMemBlkReeDiag(i, FREQ_IDX) = cuReal(shA(i, i));
                    // shMemBlkReeDiag(i, FREQ_IDX) = cuGet<TCompute>(1)/(cuReal(shA(i, i)));
                    // tDbg(i, THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx).x = shMemBlkReeDiag(i, FREQ_IDX);
                    // tDbg(i, THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx).y = 0;
#ifdef ENABLE_DEBUG
                    if((0 == blockIdx.x) && (4 == FREQ_IDX)) printf("ReeDiag[%d][%d] = %f\n", FREQ_IDX, i, shMemBlkReeDiag(i));
#endif
                }
            }
        }

        // Wait for Ree = GInv to be computed
        thisThrdBlk.sync();
    }

    //--------------------------------------------------------------------------------------------------------
    // 5. Apply bias correction and write MMSE coefficients out to global memory

    // Write C for whole PRB in one burst. Each N_THREADS_X group of threads can write one C matrix
#pragma unroll
    for(uint32_t f = 0; f < N_ITER_PER_PRB; ++f)
    {
        const uint32_t FREQ_IDX = (f * N_FREQ_BINS_PER_ITER + THRD_GRP_FREQ_OFFSET);

        if(COL_IDX_C < N_COLS_C)
        {
            // Compute bias correction factor lambda and apply to coefficients
            // TCompute lambda = cuGet<TCompute>(1) / (cuGet<TCompute>(1) - shMemBlkReeDiag(ROW_IDX_C, FREQ_IDX));
            TCompute one = cuGet<TCompute>(1);
            TCompute lambda = one;
            if(shMemBlkReeDiag(ROW_IDX_C, FREQ_IDX) < one)
            {
                lambda = one / (one - shMemBlkReeDiag(ROW_IDX_C, FREQ_IDX));
            }

            tCoef(COL_IDX_C, FREQ_IDX, ROW_IDX_C, PRB_IDX, chEqTimeInstIdx) =
                type_convert<TComplexStorageOut>(shMemBlkC(ROW_IDX_C, COL_IDX_C, FREQ_IDX) * lambda);

#ifdef ENABLE_DEBUG
            printf("C[%d][%d][%d][%d] = %f+j%f\n", PRB_IDX, FREQ_IDX, ROW_IDX_C, COL_IDX_C, shMemBlkC(ROW_IDX_C, COL_IDX_C, FREQ_IDX).x, shMemBlkC(ROW_IDX_C, COL_IDX_C, FREQ_IDX).y);
#endif
        }
        if(THREAD_X_IDX < N_ROWS_REE)
        {
            // Compute ReeInv while applying bias correction
            // TCompute reeInv = (cuGet<TCompute>(1)/shMemBlkReeDiag(THREAD_X_IDX, FREQ_IDX)) - cuGet<TCompute>(1);
            TCompute reeInv = (cuGet<TCompute>(1) - shMemBlkReeDiag(THREAD_X_IDX, FREQ_IDX)) / shMemBlkReeDiag(THREAD_X_IDX, FREQ_IDX);

            // saturate error precision
            if(reeInv > MAX_ERROR_PRECISION)
               reeInv = MAX_ERROR_PRECISION;

            tReeDiagInv(FREQ_IDX, THREAD_X_IDX, PRB_IDX, chEqTimeInstIdx) = type_convert<TStorageOut>(reeInv);
            //drvdUeGrpPrms.noiseVarForDtx = CUPHY_NOISE_RATIO_LEGACYMMSE*CUPHY_NOISE_REGULARIZER; //used for DTX detection when MMSEIRC and noise-interference estimation are not applied
        }
    }

    // thisThrdBlk.sync();
}

// Equalizer coefficient compute kernel for low order MIMO per PRB
// {N_LAYERS, N_BS_ANTS} = {1,2}, {2,2}, {1,4}, {2,4}, {4,4}, {1,8}, {2,8} and {4,8}
// Inputs and outputs assumed to be column major
// dimBlock: (8,N_FREQ_BINS_PER_ITER) for N_LAYERS = 2, N_BS_ANTS = 4; (32,N_FREQ_BINS_PER_ITER) for N_LAYERS = 4, N_BS_ANTS = 8
//           N_FREQ_BINS_PER_ITER = 4
//           Essentially, there are N_FREQ_BINS_PER_ITER group of threads, each thread group contains:
//           8 threads for N_LAYERS = 2 and 32 threads for N_LAYERS = 4
// dimGrid : Nprb
template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS,            // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS,             // # of layers (# of cols in H matrix)
          uint32_t N_FREQ_BINS_PER_ITER> // # of frequency bins processed in 1 iteration by the thread block
__device__ void
eqMmseIrcCoefCompLowMimoKernel(puschRxChEqStatDescr_t* pStatDescr, puschRxChEqCoefCompDynDescr_t* pDynDescr)
{
    KERNEL_PRINT_GRID_ONCE("%s\n grid = (%u %u %u), block = (%u %u %u), N_BS_ANTS = %u, N_LAYERS = %u, N_FREQ_BINS_PER_ITER = %u\n",
                           __PRETTY_FUNCTION__,
                           gridDim.x, gridDim.y, gridDim.z,
                           blockDim.x, blockDim.y, blockDim.z,
                           N_BS_ANTS,
                           N_LAYERS,
                           N_FREQ_BINS_PER_ITER);

    //--------------------------------------------------------------------------------------------------------
    // Setup local parameters based on descriptor
    // puschRxChEqStatDescr_t&        statDescr = *(pStatDescr);
    puschRxChEqCoefCompDynDescr_t& dynDescr = *(pDynDescr);

    // Early exit check
    // The grid is sized to process the max # of PRBs in a given heterogenous config. Exit if the PRB to be
    // processed by this thread block does not exist in the UE group
    const uint32_t PRB_IDX = blockIdx.x;
    const uint32_t UE_GRP_IDX = dynDescr.hetCfgUeGrpMap[blockIdx.y];
    
    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];

    const uint16_t nPrb = drvdUeGrpPrms.nPrb;
    if(PRB_IDX >= nPrb) return;

    //--------------------------------------------------------------------------------------------------------
    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    // clang-format off
    tensor_ref<const TComplexStorageIn> tH         (drvdUeGrpPrms.tInfoHEst.pAddr       , drvdUeGrpPrms.tInfoHEst.strides       );// (N_BS_ANTS, N_LAYERS, NF, NH)
    tensor_ref<const TComplexStorageIn> tLwwInv    (drvdUeGrpPrms.tInfoLwInv.pAddr      , drvdUeGrpPrms.tInfoLwInv.strides      );// (N_BS_ANTS, N_BS_ANTS, N_PRB)
    tensor_ref<TStorageOut>             tReeDiagInv(drvdUeGrpPrms.tInfoReeDiagInv.pAddr , drvdUeGrpPrms.tInfoReeDiagInv.strides );// (N_SC, N_LAYERS, N_PRB, NH)   // (N_LAYERS, NF, NH)
    tensor_ref<TComplexStorageOut>      tCoef      (drvdUeGrpPrms.tInfoEqCoef.pAddr     , drvdUeGrpPrms.tInfoEqCoef.strides     );// (N_SC, N_LAYERS, N_BS_ANTS, N_PRB, NH) // (N_LAYERS, N_BS_ANTS, NF, NH)
    tensor_ref<const TComplexStorageIn> tCfoEst    (drvdUeGrpPrms.tInfoCfoEst.pAddr     , drvdUeGrpPrms.tInfoCfoEst.strides    ); // (MAX_ND_SUPPORTED, N_LAYERS)
    tensor_ref<TComplexStorageOut>      tDbg       (drvdUeGrpPrms.tInfoChEqDbg.pAddr    , drvdUeGrpPrms.tInfoChEqDbg.strides    );

    // clang-format on

    uint8_t* dmrsSymLoc = drvdUeGrpPrms.dmrsSymLoc;

    //--------------------------------------------------------------------------------------------------------
    // Dimensions

    // H  : Channel matrix
    constexpr uint32_t N_ROWS_H = N_BS_ANTS;
    constexpr uint32_t N_COLS_H = N_LAYERS;

    // LwwInv: Inverse cholesky factor of Noise-interference covariance matrix
    constexpr uint32_t N_ROWS_LWW_INV = N_BS_ANTS;
    constexpr uint32_t N_COLS_LWW_INV = N_BS_ANTS;

    // N  : Intermediate matrix, N = LwwInv*H
    constexpr uint32_t N_ROWS_N = N_ROWS_LWW_INV;   // N_BS_ANTS
    constexpr uint32_t N_COLS_N = N_COLS_H;         // N_LAYERS

    // M  : Intermediate matrix, M = N'*LwwInv
    constexpr uint32_t N_ROWS_M = N_COLS_N;       // N_LAYERS
    constexpr uint32_t N_COLS_M = N_COLS_LWW_INV; // N_BS_ANTS

    // G  : Enhanced Gram matrix, G = N'*N + inv(Rxx)
    constexpr uint32_t N_ROWS_G = N_LAYERS;
    constexpr uint32_t N_COLS_G = N_LAYERS;

    // DU : Upper Triangular + Diagonal matrix in G = U'*(D*U)
    // constexpr uint32_t N_ROWS_DU = N_ROWS_G;
    // constexpr uint32_t N_COLS_DU = N_COLS_G;

    // Ree: Residual error covariance matrix, Ree = GInv
    constexpr uint32_t N_ROWS_REE = N_ROWS_G;
    constexpr uint32_t N_COLS_REE = N_COLS_G;

    // C  : MMSE coefficient matrix, C = Ree*N'*LwwInv = Ree*M
    constexpr uint32_t N_ROWS_C = N_ROWS_REE;
    constexpr uint32_t N_COLS_C = N_COLS_M;

    // I  : Identity matrix, I = G*GInv
    constexpr uint32_t N_ROWS_I = N_ROWS_G;
    constexpr uint32_t N_COLS_I = N_COLS_G;

    // J  : Intermediate matrix used in Ree = GInv computation, J = D*U*GInv
    // constexpr uint32_t N_ROWS_J = N_ROWS_DU;
    // constexpr uint32_t N_COLS_J = N_COLS_G;

    // K  : Intermediate matrix used in C (MMSE coefficients) computation, K = D*U*C
    // constexpr uint32_t N_ROWS_K = N_ROWS_DU;
    // constexpr uint32_t N_COLS_K = N_COLS_C;

    // A  : Augmented result matrix, A = [I | M] -> [J | K] -> [Ree | C]
    constexpr uint32_t N_ROWS_A = N_ROWS_I;
    constexpr uint32_t N_COLS_A = N_COLS_I + N_COLS_M;

    // Need to compute:
    // Residual error covariance matrix Ree = GInv and
    // MMSE coefficients C = Ree*N'*LwwInv = Ree*M

    // a. Factorize G = U'*D*U (transforms G to DU)

    // G*GInv = I => U'*D*U*GInv = I
    // C = Ree*M = GInv*M => G*C = M => U'*D*U*C = M

    // concatenating the two problems:
    // U'*D*U*[GInv | C] = [I | M]

    // Set D*U*GInv = J, D*U*C = K
    // b. Forward substitution : U'*[J    | K] = [I | M], solve for J and K
    // c. Backward substitution: DU*[GInv | C] = [J | K], solve for GInv and C

    // const uint32_t N_THREADS_X = blockDim.x;
    // const uint32_t N_THREADS = blockDim.x * blockDim.y; // N_FREQ_BINS_PER_ITER == blockDim.y

    static_assert(N_LAYERS <= N_BS_ANTS, "Recevied layer count should atmost equal base station antenna count");
    static_assert(0 == (CUPHY_N_TONES_PER_PRB % N_FREQ_BINS_PER_ITER), "Number of freq bins processed per iter must be a multiple of PRB size");
    constexpr uint32_t N_ITER_PER_PRB = CUPHY_N_TONES_PER_PRB / N_FREQ_BINS_PER_ITER;

    // Reciprocal of symbol energy
    const TCompute ES_INV = cuGet<TCompute>(1);

    //--------------------------------------------------------------------------------------------------------
    // Compute indices used for element access
    const uint32_t THREAD_X_IDX = threadIdx.x;
    // const uint32_t THREAD_IDX   = (threadIdx.y * blockDim.x) + threadIdx.x;

    // There are N_FREQ_BINS_PER_ITER groups of threads with each group containing N_THREADS_X per group
    const uint32_t THRD_GRP_FREQ_OFFSET        = threadIdx.y;
    const uint32_t THRD_BLK_ABS_START_FREQ_IDX = PRB_IDX * CUPHY_N_TONES_PER_PRB;

    const uint32_t chEqTimeInstIdx = dynDescr.chEqTimeInstIdx;

    // const uint32_t ROW_IDX_LWW_INV = THREAD_IDX % N_ROWS_LWW_INV;
    // const uint32_t COL_IDX_LWW_INV = THREAD_IDX / N_ROWS_LWW_INV; // COL_IDX_LWW_INV needs a bounds check (since N_THREADS > # of Lww elements)

    const uint32_t ROW_IDX_H = THREAD_X_IDX % N_ROWS_H;
    const uint32_t COL_IDX_H = THREAD_X_IDX / N_ROWS_H;

    const uint32_t ROW_IDX_I = THREAD_X_IDX % N_ROWS_I;
    const uint32_t COL_IDX_I = THREAD_X_IDX / N_ROWS_I;

    const uint32_t ROW_IDX_M = THREAD_X_IDX % N_ROWS_M;
    const uint32_t COL_IDX_M = THREAD_X_IDX / N_ROWS_M;

    const uint32_t ROW_IDX_N = THREAD_X_IDX % N_ROWS_N;
    const uint32_t COL_IDX_N = THREAD_X_IDX / N_ROWS_N;

    const uint32_t ROW_IDX_G = THREAD_X_IDX % N_ROWS_G;
    const uint32_t COL_IDX_G = THREAD_X_IDX / N_ROWS_G; // COL_IDX_G needs a bounds check (since N_THREADS_X > # of G elements)

    // const uint32_t ROW_IDX_REE = THREAD_X_IDX % N_ROWS_REE;
    // const uint32_t COL_IDX_REE = THREAD_X_IDX / N_ROWS_REE; // COL_IDX_REE needs a bounds check (since N_THREADS_X > # of Ree elements)

    const uint32_t ROW_IDX_C = THREAD_X_IDX % N_ROWS_C;
    const uint32_t COL_IDX_C = THREAD_X_IDX / N_ROWS_C;

    //--------------------------------------------------------------------------------------------------------
    // Shared memory allocation

    // SMEM allocation: C[N_TONES_PER_PRB]        , ReeDiag[N_TONES_PER_PRB],
    //                  Lww[1]                    ,
    //                  DInv[N_FREQ_BINS_PER_ITER], A[N_FREQ_BINS_PER_ITER] , G[N_FREQ_BINS_PER_ITER], N[N_FREQ_BINS_PER_ITER]
    // @tdoo: ReeDiag and DInv are of type TCompute, consider allocating in TCompute and cast into TComplexCompute
    // @todo: Check if N can be overlaid
    constexpr uint32_t N_SMEM_ELEMS =
        (((N_ROWS_C + 1) * N_COLS_C) + N_ROWS_REE) * CUPHY_N_TONES_PER_PRB +
        ((N_ROWS_LWW_INV + 1) * N_COLS_LWW_INV) +
        (N_LAYERS + ((N_ROWS_G + 1) * N_COLS_G) + ((N_ROWS_A + 1) * N_COLS_A) + ((N_ROWS_N + 1) * N_COLS_N)) * N_FREQ_BINS_PER_ITER;

    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    __shared__ TComplexCompute smemBlk[N_SMEM_ELEMS];

    // H overlays C (note: while H and C both have the same size they have different dimensions. Since
    // N_COLS_C > N_COLS_H, padding the rows of C by 1 makes C bigger overall, thus H start is aligned to C
    // start boundary and not vice-versa)
    constexpr uint32_t                                                        SMEM_START_OFFSET_C_BLK = 0;
    block_3D<TComplexCompute*, N_ROWS_C + 1, N_COLS_C, CUPHY_N_TONES_PER_PRB> shMemBlkC(&smemBlk[SMEM_START_OFFSET_C_BLK]);

    constexpr uint32_t                                     SMEM_START_OFFSET_REE_DIAG_BLK = SMEM_START_OFFSET_C_BLK + shMemBlkC.num_elem();
    block_2D<TCompute*, N_ROWS_REE, CUPHY_N_TONES_PER_PRB> shMemBlkReeDiag(reinterpret_cast<TCompute*>(&smemBlk[SMEM_START_OFFSET_REE_DIAG_BLK]));

    // LwwInv shared across all frequency bins in PRB
    constexpr uint32_t                                     SMEM_START_OFFSET_LWW_INV = SMEM_START_OFFSET_REE_DIAG_BLK + shMemBlkReeDiag.num_elem();
    block_2D<TComplexCompute*, N_ROWS_LWW_INV + 1, N_COLS_LWW_INV> shLwwInv(&smemBlk[SMEM_START_OFFSET_LWW_INV]);

    const uint32_t                SMEM_START_OFFSET_DINV_BLK = SMEM_START_OFFSET_LWW_INV + shLwwInv.num_elem();
    const uint32_t                SMEM_START_OFFSET_DINV     = SMEM_START_OFFSET_DINV_BLK + THRD_GRP_FREQ_OFFSET * N_LAYERS;
    block_1D<TCompute*, N_LAYERS> shDInv(reinterpret_cast<TCompute*>(&smemBlk[SMEM_START_OFFSET_DINV]));

    const uint32_t                                     SMEM_START_OFFSET_G_BLK = SMEM_START_OFFSET_DINV_BLK + N_FREQ_BINS_PER_ITER * shDInv.num_elem();
    const uint32_t                                     SMEM_START_OFFSET_G     = SMEM_START_OFFSET_G_BLK + THRD_GRP_FREQ_OFFSET * (N_ROWS_G + 1) * N_COLS_G;
    block_2D<TComplexCompute*, N_ROWS_G + 1, N_COLS_G> shG(&smemBlk[SMEM_START_OFFSET_G]);

    const uint32_t                                     SMEM_START_OFFSET_A_BLK = SMEM_START_OFFSET_G_BLK + N_FREQ_BINS_PER_ITER * shG.num_elem();
    const uint32_t                                     SMEM_START_OFFSET_A     = SMEM_START_OFFSET_A_BLK + THRD_GRP_FREQ_OFFSET * (N_ROWS_A + 1) * N_COLS_A;
    block_2D<TComplexCompute*, N_ROWS_A + 1, N_COLS_A> shA(&smemBlk[SMEM_START_OFFSET_A]);

    // I and M overlay on A
    const uint32_t                                     SMEM_START_OFFSET_I = SMEM_START_OFFSET_A;
    block_2D<TComplexCompute*, N_ROWS_I + 1, N_COLS_I> shI(&smemBlk[SMEM_START_OFFSET_I]);

    const uint32_t                                     SMEM_START_OFFSET_M = SMEM_START_OFFSET_I + shI.num_elem();
    block_2D<TComplexCompute*, N_ROWS_M + 1, N_COLS_M> shM(&smemBlk[SMEM_START_OFFSET_M]);

    const uint32_t                                     SMEM_START_OFFSET_N_BLK = SMEM_START_OFFSET_A_BLK + N_FREQ_BINS_PER_ITER * shA.num_elem();
    const uint32_t                                     SMEM_START_OFFSET_N     = SMEM_START_OFFSET_N_BLK + THRD_GRP_FREQ_OFFSET * (N_ROWS_N + 1) * N_COLS_N;
    block_2D<TComplexCompute*, N_ROWS_N + 1, N_COLS_N> shN(&smemBlk[SMEM_START_OFFSET_N]);

    // SMEM overlays
    auto& shDU = shG; // G and DU have the same dimension

    bool     enableCfoCorrection = (0 != drvdUeGrpPrms.enableCfoCorrection) ? true : false;
    uint8_t* pUeGrpLayerToUeIdx  = drvdUeGrpPrms.ueGrpLayerToUeIdx;    
    const uint16_t dmrsMaxLen = drvdUeGrpPrms.dmrsMaxLen;

    // Overlays on A: A = [I | M] -> [J | K] -> [Ree | C]
    // I, J and Ree have the same dimension and are overlaid
    // M, K and C have the same dimension and are overlaid
    // auto& shJ   = shI;
    // auto& shRee = shJ;

    // auto& shK   = shM;
    // auto& shC   = shK;

    thread_block const& thisThrdBlk = this_thread_block();

#ifdef ENABLE_DEBUG
    if(0 != blockIdx.x) return;
#endif


#ifdef ENABLE_DEBUG
     if((0 == blockIdx.x) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     printf("%s\n grid = (%u %u %u) block = (%u %u %u) blockIdx.y %d UE_GRP_IDX %d hetCfgUeGrpMap[0] %d hetCfgUeGrpMap[1] %d hetCfgUeGrpMap[2] %d\n", __PRETTY_FUNCTION__, gridDim.x, gridDim.y, gridDim.z, blockDim.x, blockDim.y, blockDim.z, blockIdx.y, UE_GRP_IDX, dynDescr.hetCfgUeGrpMap[0], dynDescr.hetCfgUeGrpMap[1], dynDescr.hetCfgUeGrpMap[2]);

     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     {
         printf("Addr: tH %lp tLwwInv %lp tReeDiagInv %lp tCoef %lp\n", tH.addr, tLwwInv.addr, tReeDiagInv.addr, tCoef.addr);
  
         printf("nPrb     : %d \n", nPrb);
         printf("tH          : addr %lp strides[0] %d strides[1] %d strides[2] %d strides[3] %d\n", static_cast<const TComplexStorageIn*>(tH.addr)     , tH.strides[0]         , tH.strides[1]         , tH.strides[2]         , tH.strides[3]         );
         printf("tLwwInv     : addr %lp strides[0] %d strides[1] %d strides[2] %d strides[3] %d\n", static_cast<const TComplexStorageIn*>(tLwwInv.addr), tLwwInv.strides[0]    , tLwwInv.strides[1]    , tLwwInv.strides[2]    , tLwwInv.strides[3]    );
         printf("tReeDiagInv : addr %lp strides[0] %d strides[1] %d strides[2] %d strides[3] %d\n", static_cast<TStorageOut*>(tReeDiagInv.addr)        , tReeDiagInv.strides[0], tReeDiagInv.strides[1], tReeDiagInv.strides[2], tReeDiagInv.strides[3]);
         printf("tCoef       : addr %lp strides[0] %d strides[1] %d strides[2] %d strides[3] %d\n", static_cast<TComplexStorageOut*>(tCoef.addr)       , tCoef.strides[0]      , tCoef.strides[1]      , tCoef.strides[2]      , tCoef.strides[3]      );
         // printf("tDbg    strides[0] %d strides[1] %d strides[2] %d\n", tDbg.strides[0], tDbg.strides[1], tDbg.strides[2]);
     }
#endif     

    //--------------------------------------------------------------------------------------------------------
    // 1. Prefetch H and LwwInv into shared memory

    // Read LwwInv (LwwInv is used by all subcarriers within the PRB)
    // @todo: suficient to load only the lower triangular elements
    block_2D<const typename complex_from_scalar<TStorageIn>::type*, N_ROWS_LWW_INV, N_COLS_LWW_INV> srcLwwInv(tLwwInv.addr + tLwwInv.offset(0, 0, PRB_IDX));
    cmplxMatLoad<TStorageIn, TCompute, N_ROWS_LWW_INV, N_COLS_LWW_INV>(thisThrdBlk, srcLwwInv, shLwwInv);

    // Read H for whole PRB in one burst. Each N_THREADS_X group of threads can read one H matrix
#pragma unroll
    for(uint32_t f = 0; f < N_ITER_PER_PRB; ++f)
    {
        const uint32_t FREQ_IDX     = (f * N_FREQ_BINS_PER_ITER + THRD_GRP_FREQ_OFFSET);
        const uint32_t ABS_FREQ_IDX = THRD_BLK_ABS_START_FREQ_IDX + FREQ_IDX;

        // SMEM overlay, align H start to C start boundary
        const uint32_t                                     SMEM_START_OFFSET_C = SMEM_START_OFFSET_C_BLK + FREQ_IDX * (N_ROWS_C + 1) * N_COLS_C;
        const uint32_t                                     SMEM_START_OFFSET_H = SMEM_START_OFFSET_C;
        block_2D<TComplexCompute*, N_ROWS_H + 1, N_COLS_H> shH(&smemBlk[SMEM_START_OFFSET_H]);

        if(COL_IDX_H < N_COLS_H)
        {             
            // If needed, scale channel estimates by CFO correction
            if(enableCfoCorrection && (chEqTimeInstIdx > 0))
            {
                shH(ROW_IDX_H, COL_IDX_H) =
                        type_convert<TComplexCompute>(tH(ROW_IDX_H, COL_IDX_H, ABS_FREQ_IDX, chEqTimeInstIdx)) *
                        type_convert<TComplexCompute>(tCfoEst(dmrsSymLoc[chEqTimeInstIdx *dmrsMaxLen], pUeGrpLayerToUeIdx[COL_IDX_H]));
            }
            else
            {
                shH(ROW_IDX_H, COL_IDX_H) =
                    type_convert<TComplexCompute>(tH(ROW_IDX_H, COL_IDX_H, ABS_FREQ_IDX, chEqTimeInstIdx));
            }

            // tDbg(ROW_IDX_H,COL_IDX_H,ABS_FREQ_IDX, chEqTimeInstIdx) = shH(ROW_IDX_H,COL_IDX_H);
#ifdef ENABLE_DEBUG
            printf("H[%d][%d][%d] = %f+j%f\n", ABS_FREQ_IDX, ROW_IDX_H, COL_IDX_H, shH(ROW_IDX_H, COL_IDX_H).x, shH(ROW_IDX_H, COL_IDX_H).y);
#endif
        }
    }

    // Wait for loads to complete. Thread(s) processing an entry of H or LwwInv may not be the same ones loading
    // it
    thisThrdBlk.sync();

    for(uint32_t f = 0; f < N_ITER_PER_PRB; ++f)
    {
        const uint32_t FREQ_IDX = (f * N_FREQ_BINS_PER_ITER + THRD_GRP_FREQ_OFFSET);

        const uint32_t                                     SMEM_START_OFFSET_C = SMEM_START_OFFSET_C_BLK + FREQ_IDX * (N_ROWS_C + 1) * N_COLS_C;
        block_2D<TComplexCompute*, N_ROWS_C + 1, N_COLS_C> shCRes(&smemBlk[SMEM_START_OFFSET_C]);

        // SMEM overlay, align H start to C start boundary
        const uint32_t                                     SMEM_START_OFFSET_H = SMEM_START_OFFSET_C;
        block_2D<TComplexCompute*, N_ROWS_H + 1, N_COLS_H> shH(&smemBlk[SMEM_START_OFFSET_H]);

        //---------------------------------------------------------------------------------------------------
        // 2. Compute enhanced Gram matrix: G = (N'*N + RxxInv), where N = LwwInv*H

        // 2a. Compute intermediate matrix N = LwwInv*H
        // Each thread computes one element of N
        if(COL_IDX_N < N_COLS_N)
        {
            TComplexCompute N = cuGet<TComplexCompute>(0);

            // Lower triangular multiply from the left, so inner product compute beginning at the first element 
            // (in a row or column) of input matrix stops at the diagonal element (i <= ROW_IDX_N)
            // Note: N_COLS_N <= N_ROWS_N
#pragma unroll            
            for(uint32_t i = 0; i <= ROW_IDX_N; ++i)
            {
                N = cuFma(shLwwInv(ROW_IDX_N, i), shH(i, COL_IDX_N), N);
                
#ifdef ENABLE_DEBUG
                if(0 == FREQ_IDX) printf("Iter %d: N[%d][%d][%d] = %f+j%f H[%d][%d] = %f+j%f LwwInv[%d][%d] = %f+j%f\n", i, FREQ_IDX, ROW_IDX_N, COL_IDX_N, cuReal(N), cuImag(N), ROW_IDX_N, i, cuReal(cuConj(shH(i, ROW_IDX_N))), cuImag(cuConj(shH(i, ROW_IDX_N))), i, COL_IDX_N, cuReal(shLwwInv(i, COL_IDX_N)), cuImag(shLwwInv(i, COL_IDX_N)));
#endif
            }
            shN(ROW_IDX_N, COL_IDX_N) = N;
        }

        // Wait for matrix N computation to finish, before using it in computation of G matrix (and subsequently matrix M)
        thisThrdBlk.sync();

        // 2b. Compute G = (N'*N + RxxInv)
        // Since G is Hermitian symmetric, it is sufficient to compute the lower triangular matrix elements and compute the
        // entries above the main diagonla via conjugation
        if((COL_IDX_G < N_COLS_G) && (ROW_IDX_G <= COL_IDX_G))
        {
            TComplexCompute G = cuGet<TComplexCompute>(0);

#pragma unroll
            for(uint32_t i = 0; i < N_ROWS_N; ++i)
            {
                G = cuFma(cuConj(shN(i, ROW_IDX_G)), shN(i, COL_IDX_G), G);
            }

            if(ROW_IDX_G != COL_IDX_G)
            {
                shG(COL_IDX_G, ROW_IDX_G) = cuConj(G);
            }
            else
            {
                G += ES_INV;
            }
            shG(ROW_IDX_G, COL_IDX_G) = G;

#ifdef ENABLE_DEBUG
            if(0 == FREQ_IDX) printf("G[%d][%d][%d] = %f+j%f\n", FREQ_IDX, ROW_IDX_G, COL_IDX_G, cuReal(shG(ROW_IDX_G, COL_IDX_G)), cuImag(shG(ROW_IDX_G, COL_IDX_G)));
#endif

            // tDbg(ROW_IDX_G,COL_IDX_G,THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shG(ROW_IDX_G,COL_IDX_G);
        }

        // Wait for matrix G computation to finish, before using it in computation of GInv
        thisThrdBlk.sync();

        //-----------------------------------------------------------------------------------------------
        // 3. LDL factorization: G = U'*D*U
        // Single threaded section (only N_FREQ_BINS_PER_ITER threads are active inside the if condition)

        if(0 == THREAD_X_IDX)
        {
#pragma unroll
            for(int32_t i = 0; i < N_LAYERS; ++i)
            {
                // compute ith diagonal entry of diagonal matrix D
                TCompute sum1 = cuGet<TCompute>(0);
                for(int32_t j = 0; j < i; ++j)
                {
                    sum1 += (cuReal(shDU(j, j)) * cuReal(cuConj(shDU(j, i)) * shDU(j, i)));

#ifdef ENABLE_DEBUG
                    printf("FREQ_IDX %d Row %d Iter %d DU[%d][%d][%d] = %f+j%f, DU[%d][%d][%d] = %f+j%f, sum1 = %f\n", FREQ_IDX, i, j, FREQ_IDX, j, j, shDU(j, j).x, shDU(j, j).y, FREQ_IDX, j, i, shDU(j, i).x, shDU(j, i).y, sum1);
#endif
                }
                shDU(i, i) = cuGet<TComplexCompute>(cuReal(shG(i, i)) - sum1);
                shDInv(i)  = cuGet<TCompute>(1) / cuReal(shDU(i, i));

                // tDbg(i,i,THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shDU(i,i);

#ifdef ENABLE_DEBUG
                printf("DU[%d][%d][%d] = %f+j%f, DInv[%d] = %f\n", FREQ_IDX, i, i, shDU(i, i).x, shDU(i, i).y, i, shDInv(i));
#endif
                // compute upper diagonal elements of ith row of matrix U (U is an upper triangular matrix)
                for(int32_t j = i + 1; j < N_LAYERS; ++j)
                {
                    TComplexCompute sum2 = cuGet<TComplexCompute>(0);
                    for(int32_t k = 0; k < i; ++k)
                    {
                        sum2 += ((cuConj(shDU(k, i)) * shDU(k, j)) * cuReal(shDU(k, k)));
                    }
                    shDU(i, j) = (shG(i, j) - sum2) * shDInv(i);

                    // tDbg(i,j,THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shDU(i,j);
#ifdef ENABLE_DEBUG
                    printf("FREQ_IDX %d Row %d Col %d DU[%d][%d][%d] = %f+j%f, sum2 = %f+j%f\n", FREQ_IDX, i, j, FREQ_IDX, i, j, shDU(i, j).x, shDU(i, j).y, sum2.x, sum2.y);
#endif
                }
            }
        }

        if(COL_IDX_I < N_COLS_I)
        {
            shI(ROW_IDX_I, COL_IDX_I) =
                (ROW_IDX_I != COL_IDX_I) ? cuGet<TComplexCompute>(0) : cuGet<TComplexCompute>(1);
        }

        // Note: No CTA sync here since G compute does not need M.        

        // Compute intermediate matrix M = N'*LwwInv
        if(COL_IDX_M < N_COLS_M)
        {
            TComplexCompute M = cuGet<TComplexCompute>(0);

            // Lower triangular multiply from the right, so inner product compute begins at the diagonal element (i = COL_IDX_M)
            // and ends at last element (of the row or column) of input matrix
            // Note: N_COLS_M >= N_ROWS_M
#pragma unroll
            for(uint32_t i = COL_IDX_M; i < N_ROWS_LWW_INV; ++i)
            {
                M = cuFma(cuConj(shN(i, ROW_IDX_M)), shLwwInv(i, COL_IDX_M), M);
#ifdef ENABLE_DEBUG
                if(0 == FREQ_IDX) printf("Iter %d: M[%d][%d][%d] = %f+j%f conjN[%d][%d] = %f+j%f LwwInv[%d][%d] = %f+j%f\n", i, FREQ_IDX, ROW_IDX_M, COL_IDX_M, cuReal(M), cuImag(M), ROW_IDX_M, i, cuReal(cuConj(shN(i, ROW_IDX_M))), cuImag(cuConj(shN(i, ROW_IDX_M))), i, COL_IDX_M, cuReal(shLwwInv(i, COL_IDX_M)), cuImag(shLwwInv(i, COL_IDX_M)));
#endif
            }
            shM(ROW_IDX_M, COL_IDX_M) = M;
        }

        // Wait for LDL factorization and computation of M to complete
        thisThrdBlk.sync();

#ifdef ENABLE_DEBUG
        for(int32_t i = 0; i < N_ROWS_A; ++i)
        {
            const uint32_t COL_IDX_A = THREAD_X_IDX;
            if((COL_IDX_A < N_COLS_A) && (0 == FREQ_IDX)) printf("A[%d][%d][%d] = %f+j%f\n", FREQ_IDX, i, COL_IDX_A, shA(i, COL_IDX_A).x, shA(i, COL_IDX_A).y);
        }

        if((COL_IDX_I < N_COLS_I) && (0 == FREQ_IDX)) printf("I[%d][%d][%d] = %f+j%f\n", FREQ_IDX, ROW_IDX_I, COL_IDX_I, shI(ROW_IDX_I, COL_IDX_I).x, shI(ROW_IDX_I, COL_IDX_I).y);
        if((COL_IDX_M < N_COLS_M) && (0 == FREQ_IDX)) printf("M[%d][%d][%d] = %f+j%f\n", FREQ_IDX, ROW_IDX_M, COL_IDX_M, shM(ROW_IDX_M, COL_IDX_M).x, shM(ROW_IDX_M, COL_IDX_M).y);
#endif

        //-----------------------------------------------------------------------------------------------
        // 4. Forward substitution: U'*[J | K] = [I | M], solve for J, K
        const uint32_t COL_IDX_A = THREAD_X_IDX;
        if(COL_IDX_A < N_COLS_A)
        {
#pragma unroll
            for(int32_t i = 0; i < N_LAYERS; ++i)
            {
                // Compute row i of intermediate matrices J, K
                TComplexCompute sum = cuGet<TComplexCompute>(0);
                for(int32_t j = 0; j < i; ++j)
                {
                    sum += (cuConj(shDU(j, i)) * shA(j, COL_IDX_A));
                }

                shA(i, COL_IDX_A) = shA(i, COL_IDX_A) - sum;
                // tDbg(i, COL_IDX_A, THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shA(i, COL_IDX_A);
            }
        }

        // Wait for Forward substitution to complete
        thisThrdBlk.sync();

        //-----------------------------------------------------------------------------------------------
        // 5. Backward substitution: D*U*[GInv | C] = [J | K], solve for GInv, C
        // Residual error covariance matrix Ree is the inverse of Gram matrix: Ree = GInv
        if(COL_IDX_A < N_COLS_A)
        {
#pragma unroll
            for(int32_t i = N_LAYERS - 1; i >= 0; --i)
            {
                // Compute a row of intermediate matrices J, K
                TComplexCompute sum = cuGet<TComplexCompute>(0);
                for(int32_t j = i + 1; j < N_LAYERS; ++j)
                {
                    sum += (shA(j, COL_IDX_A) * shDU(i, j));
                }

                shA(i, COL_IDX_A) = (shA(i, COL_IDX_A) * shDInv(i)) - sum;
                // tDbg(i, COL_IDX_A, THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shA(i, COL_IDX_A);

#ifdef ENABLE_DEBUG
                printf("A[%d][%d][%d] = %f+j%f\n", FREQ_IDX, i, COL_IDX_A, shA(i, COL_IDX_A).x, shA(i, COL_IDX_A).y);
#endif

                // Store results to be available for writes later
                if((COL_IDX_A >= N_COLS_REE) && (COL_IDX_A < N_COLS_A))
                {
                    const uint32_t COL_IDX_C_DERIVED = COL_IDX_A - N_COLS_REE;
                    shCRes(i, COL_IDX_C_DERIVED)     = shA(i, COL_IDX_A);
                    // tDbg(i, COL_IDX_C_DERIVED, THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx) = shA(i, COL_IDX_A);//  shCRes(i, COL_IDX_C_DERIVED);

#ifdef ENABLE_DEBUG
                    if((0 == blockIdx.x) && (4 == FREQ_IDX)) printf("C[%d][%d][%d] = %f+j%f\n", FREQ_IDX, i, COL_IDX_C_DERIVED, shCRes(i, COL_IDX_C_DERIVED).x, shCRes(i, COL_IDX_C_DERIVED).y);
#endif
                }
                if(COL_IDX_A == i)
                {
                    shMemBlkReeDiag(i, FREQ_IDX) = cuReal(shA(i, i));
                    // shMemBlkReeDiag(i, FREQ_IDX) = cuGet<TCompute>(1)/(cuReal(shA(i, i)));
                    // tDbg(i, THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx).x = shMemBlkReeDiag(i, FREQ_IDX);
                    // tDbg(i, THRD_BLK_ABS_START_FREQ_IDX+FREQ_IDX, chEqTimeInstIdx).y = 0;
#ifdef ENABLE_DEBUG
                    if((0 == blockIdx.x) && (4 == FREQ_IDX)) printf("ReeDiag[%d][%d] = %f\n", FREQ_IDX, i, shMemBlkReeDiag(i));
#endif
                }
            }
        }

        // Wait for Ree = GInv to be computed
        thisThrdBlk.sync();
    }

    //--------------------------------------------------------------------------------------------------------
    // 6. Apply bias correction and write MMSE coefficients out to global memory

    // Write C for whole PRB in one burst. Each N_THREADS_X group of threads can write one C matrix
#pragma unroll
    for(uint32_t f = 0; f < N_ITER_PER_PRB; ++f)
    {
        const uint32_t FREQ_IDX = (f * N_FREQ_BINS_PER_ITER + THRD_GRP_FREQ_OFFSET);

        if(COL_IDX_C < N_COLS_C)
        {
            // Compute bias correction factor lambda and apply to coefficients
            //TCompute lambda = cuGet<TCompute>(1) / (cuGet<TCompute>(1) - shMemBlkReeDiag(ROW_IDX_C, FREQ_IDX));
            TCompute one = cuGet<TCompute>(1);
            TCompute lambda = one;
            if(shMemBlkReeDiag(ROW_IDX_C, FREQ_IDX) < one)
            {
                lambda = one / (one - shMemBlkReeDiag(ROW_IDX_C, FREQ_IDX));
            }

            tCoef(COL_IDX_C, FREQ_IDX, ROW_IDX_C, PRB_IDX, chEqTimeInstIdx) =
                type_convert<TComplexStorageOut>(shMemBlkC(ROW_IDX_C, COL_IDX_C, FREQ_IDX) * lambda);
            //printf("w[%f %f %f]", lambda, (float)shMemBlkC(ROW_IDX_C, COL_IDX_C, FREQ_IDX).x, (float)tCoef(COL_IDX_C, FREQ_IDX, ROW_IDX_C, PRB_IDX, chEqTimeInstIdx).x);

#ifdef ENABLE_DEBUG
            printf("C[%d][%d][%d][%d] = %f+j%f\n", PRB_IDX, FREQ_IDX, ROW_IDX_C, COL_IDX_C, shMemBlkC(ROW_IDX_C, COL_IDX_C, FREQ_IDX).x, shMemBlkC(ROW_IDX_C, COL_IDX_C, FREQ_IDX).y);
#endif
        }
        if(THREAD_X_IDX < N_ROWS_REE)
        {
            // Compute ReeInv while applying bias correction
            // TCompute reeInv = (cuGet<TCompute>(1)/shMemBlkReeDiag(THREAD_X_IDX, FREQ_IDX)) - cuGet<TCompute>(1);
            TCompute reeInv = (cuGet<TCompute>(1) - shMemBlkReeDiag(THREAD_X_IDX, FREQ_IDX)) / shMemBlkReeDiag(THREAD_X_IDX, FREQ_IDX);
            //printf("n[%f %f]", shMemBlkReeDiag(THREAD_X_IDX, FREQ_IDX), (float)reeInv);
            // saturate error precision
            if(reeInv > MAX_ERROR_PRECISION)
               reeInv = MAX_ERROR_PRECISION;

            tReeDiagInv(FREQ_IDX, THREAD_X_IDX, PRB_IDX, chEqTimeInstIdx) = type_convert<TStorageOut>(reeInv);
        }
    }

    // thisThrdBlk.sync();
}

// equalizer coefficient compute kernel forwarding function to eqMmseIrcCoefCompLowMimoKernel or eqMmseCoefCompLowMimoKernel
// Used for low order MIMO and works per PRB
// {N_LAYERS, N_BS_ANTS} = {1,2}, {2,2}, {1,4}, {2,4}, {4,4}, {1,8}, {2,8} and {4,8}
// Inputs and outputs assumed to be column major
// dimBlock: (8,N_FREQ_BINS_PER_ITER) for N_LAYERS = 2, N_BS_ANTS = 4; (32,N_FREQ_BINS_PER_ITER) for N_LAYERS = 4, N_BS_ANTS = 8
//           N_FREQ_BINS_PER_ITER = 4
//           Essentially, there are N_FREQ_BINS_PER_ITER group of threads, each thread group contains:
//           8 threads for N_LAYERS = 2 and 32 threads for N_LAYERS = 4
// dimGrid : Nprb
template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS,            // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS,             // # of layers (# of cols in H matrix)
          uint32_t N_FREQ_BINS_PER_ITER> // # of frequency bins processed in 1 iteration by the thread block
__global__ void
eqMmseCoefCompLowMimoKernel(puschRxChEqStatDescr_t* pStatDescr, puschRxChEqCoefCompDynDescr_t* pDynDescr)
{
    const uint32_t UE_GRP_IDX = pDynDescr->hetCfgUeGrpMap[blockIdx.y];
    if(pDynDescr->pDrvdUeGrpPrms[UE_GRP_IDX].eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC)
    {
        eqMmseIrcCoefCompLowMimoKernel<TStorageIn,
                                       TStorageOut,
                                       TCompute,
                                       N_BS_ANTS,
                                       N_LAYERS,
                                       N_FREQ_BINS_PER_ITER>(pStatDescr, pDynDescr);
    }
    else
    {
        eqLegacyMmseCoefCompLowMimoKernel<TStorageIn, 
                                         TStorageOut,
                                         TCompute,
                                         N_BS_ANTS,
                                         N_LAYERS,
					 N_FREQ_BINS_PER_ITER>(pStatDescr, pDynDescr);
    }
}

#if (EQ_SOFT_DEMAP_USE_TEX == 0)

////////////////////////////////////////////////////////////////////////
// ch_eq_soft_demapper()
// Original soft demapper implementation
template <typename TStorageOut,
          typename TCompute,
          typename TLLR>
__device__ void ch_eq_soft_demapper(const int                                           PER_LAYER_THRD_IDX,
                                    int32_t                                             nPamBits,
                                    TCompute                                            noiseInv,
                                    const typename complex_from_scalar<TCompute>::type& softEst,
                                    int                                                 layerIdx,
                                    TCompute*                                           shWrkBuf,
                                    TLLR*                                               shLlr,
                                    const int                                           GMEM_ABS_WR_FREQ_IDX,
                                    const int                                           DATA_SYMB_ABS_IDX,
                                    const int                                           LLR_START_IDX,
                                    const int                                           WRK_BUF_START_IDX,
                                    TLLR*                                               LLRout)
{

    if(PER_LAYER_THRD_IDX < 2)
    {
        // PER_LAYER_THRD_IDX = 0 processes inphase sample and PER_LAYER_THRD_IDX = 1 processes
        // quadrature sample
        uint32_t iqIdx    = PER_LAYER_THRD_IDX;

#ifdef ENABLE_DEBUG
        if((0 == layerIdx) && (1 == GMEM_ABS_WR_FREQ_IDX) && (0 == dataSymLoc[DATA_SYMB_ABS_IDX]))
#endif
        computePamLlr<TCompute, TStorageOut>(nPamBits,
                                             iqIdx,
                                             noiseInv,
                                             softEst,
                                             &shWrkBuf[WRK_BUF_START_IDX],
                                             &shLlr[LLR_START_IDX]);

        // Only 2 (out of N_BS_ANTS) threads used to compute LLRs
        coalesced_group activeThrds = coalesced_threads();
        activeThrds.sync();

        KERNEL_PRINT_GRID_ONCE("layerIdx = %u, nPamBits = %i, noiseInv = %f, softEst = (%f, %f), LLR = (%f %f %f %f  %f %f %f %f)\n",
                               layerIdx,
                               nPamBits,
                               noiseInv,
                               softEst.x,
                               softEst.y,
                               debug_LLR_get_elem(shLlr[LLR_START_IDX + 0], 0),
                               debug_LLR_get_elem(shLlr[LLR_START_IDX + 0], 1),
                               debug_LLR_get_elem(shLlr[LLR_START_IDX + 0], 2),
                               debug_LLR_get_elem(shLlr[LLR_START_IDX + 0], 3),
                               debug_LLR_get_elem(shLlr[LLR_START_IDX + 1], 0),
                               debug_LLR_get_elem(shLlr[LLR_START_IDX + 1], 1),
                               debug_LLR_get_elem(shLlr[LLR_START_IDX + 1], 2),
                               debug_LLR_get_elem(shLlr[LLR_START_IDX + 1], 3));
#ifdef ENABLE_DEBUG
        // if((2 == layerIdx) && (3122 == GMEM_ABS_WR_FREQ_IDX) && (8 == dataSymLoc[DATA_SYMB_ABS_IDX]))
        if((0 == layerIdx) && (1 == GMEM_ABS_WR_FREQ_IDX) && (0 == dataSymLoc[DATA_SYMB_ABS_IDX]))
            printf("computePamLlr: [%d][%d][%d][%d] nPamBits = %d softEst = %f+j%f noiseInv = %f \n\t llr[0] = %f llr[1] = %f llr[2] = %f llr[3] = %f llr[4] = %f llr[5] = %f llr[6] = %f llr[7] = %f\n",
                   layerIdx,
                   FREQ_IDX,
                   PRB_IDX,
                   dataSymLoc[DATA_SYMB_ABS_IDX],
                   nPamBits,
                   cuReal(softEst),
                   cuImag(softEst),
                   noiseInv,
                   shLlr[LLR_START_IDX + 0].x,
                   shLlr[LLR_START_IDX + 0].y,
                   shLlr[LLR_START_IDX + 0].z,
                   shLlr[LLR_START_IDX + 0].w,
                   shLlr[LLR_START_IDX + 1].x,
                   shLlr[LLR_START_IDX + 1].y,
                   shLlr[LLR_START_IDX + 1].z,
                   shLlr[LLR_START_IDX + 1].w);
#endif

        // LLRs are packed in 2 vector float objects (float4 if FP32 and float2 if FP16). Enble writes
        // of the 2nd vector only when there are enough LLRs
        bool enableWrite = (PER_LAYER_THRD_IDX <= ((nPamBits - 1) / 2)) ? true : false;
        if(enableWrite)
        {
             // *reinterpret_cast<float4*>(tLlr.addr + llrStartOffsetGmem) = static_cast<float4>(shLlr[LLR_START_IDX + PER_LAYER_THRD_IDX]);
            *LLRout = static_cast<TLLR>(shLlr[LLR_START_IDX + PER_LAYER_THRD_IDX]);
        }
    }
}
#endif // #if (EQ_SOFT_DEMAP_USE_TEX == 0)

////////////////////////////////////////////////////////////////////////
// ch_eq_simplified_soft_demapper()
template <typename TStorageOut,
          typename TCompute>
__device__ void ch_eq_simplified_soft_demapper(const int                                    PER_LAYER_THRD_IDX,
                                        int32_t                                             nPamBits,
                                        TCompute                                            noiseInv,
                                        const typename complex_from_scalar<TCompute>::type& softEst,
                                        TStorageOut*                                        llr)
{
    // Only 1 thread performs soft demapping
    // TODO: Refactor calling kernel to have per-thread dot product + soft demapping

    if(0 == PER_LAYER_THRD_IDX)
    {
        typedef soft_demapper::soft_demapper_simplified<TCompute, TStorageOut> soft_demapper_t;
        typedef soft_demapper::LLR_group<TStorageOut, 8>                       llr_group_t;
        typedef soft_demapper::noise_type_map<TStorageOut>                     noise_type_map_t;

        // LLR output structure. Up to 8 LLRs may be required (for QAM256).
        llr_group_t grp;
        grp.ui32_4 = {0,0,0,0};

        if(nPamBits==0)
        {
            soft_demapper_t::symbol_to_LLR_group(grp,                                           // LLR output
                                                 softEst,                                       // symbol input
                                                 noise_type_map_t::scale(noiseInv, 1.0f),       // PAM noise var inverse  //TODO
                                                 1);                                            // pi/2 BPSK
        }
        else
        {
            // noiseInv input is the inverse of the (complex, QAM) noise variance
            // PAM_variance = QAM_variance / 2
            // 1 / PAM_variance = inv_PAM_variance = 2 / QAM_variance = 2 * inv_QAM_variance
            soft_demapper_t::symbol_to_LLR_group(grp,                                     // LLR output
                                             softEst,                                     // symbol input
                                             noise_type_map_t::scale(noiseInv, 2.0f),     // PAM noise var inverse
                                             nPamBits * 2);                               // QAM bits
        }
        KERNEL_PRINT_GRID_ONCE("LLR_tex = (%f %f %f %f  %f %f %f %f)\n",
                               grp[0],
                               grp[1],
                               grp[2],
                               grp[3],
                               grp[4],
                               grp[5],
                               grp[6],
                               grp[7]);

        // Write to output (global) memory. Note that as written below, this writes
        // all 8 values, whether valid for the given QAM or not.
        grp.write(llr);
    }
} // ch_eq_simplified_soft_demapper


////////////////////////////////////////////////////////////////////////
// ch_eq_soft_demapper_tex()
// Soft demapper implementation using the GPU texture unit (for some
// QAMs)
template <typename TStorageOut,
          typename TCompute>
__device__ void ch_eq_soft_demapper_tex(const int                                           PER_LAYER_THRD_IDX,
                                        int32_t                                             nPamBits,
                                        TCompute                                            noiseInv,
                                        const typename complex_from_scalar<TCompute>::type& softEst,
                                        TStorageOut*                                        llr,
                                        cudaTextureObject_t                                 texObj)
{
    // Only 1 thread performs soft demapping
    // TODO: Refactor calling kernel to have per-thread dot product + soft demapping
    
    if(0 == PER_LAYER_THRD_IDX)
    {
        typedef soft_demapper::soft_demapper_any<TCompute, TStorageOut> soft_demapper_t;
        typedef soft_demapper::LLR_group<TStorageOut, 8>                llr_group_t;
        typedef soft_demapper::noise_type_map<TStorageOut>              noise_type_map_t;

        // LLR output structure. Up to 8 LLRs may be required (for QAM256).
        llr_group_t grp;
        grp.ui32_4 = {0,0,0,0};

        if(nPamBits==0)
        {
            soft_demapper_t::symbol_to_LLR_group(grp,                                       // LLR output
                                             softEst,                                       // symbol input
                                             noise_type_map_t::scale(noiseInv, 1.0f),       // PAM noise var inverse  //TODO
                                             1,                                             // pi/2 BPSK
                                             texObj);                                       // CUDA texture object
        }
        else
        {
            // noiseInv input is the inverse of the (complex, QAM) noise variance
            // PAM_variance = QAM_variance / 2
            // 1 / PAM_variance = inv_PAM_variance = 2 / QAM_variance = 2 * inv_QAM_variance
            soft_demapper_t::symbol_to_LLR_group(grp,                                     // LLR output
                                             softEst,                                     // symbol input
                                             noise_type_map_t::scale(noiseInv, 2.0f),     // PAM noise var inverse
                                             nPamBits * 2,                                // QAM bits
                                             texObj);                                     // CUDA texture object
        }
        KERNEL_PRINT_GRID_ONCE("LLR_tex = (%f %f %f %f  %f %f %f %f)\n",
                               grp[0],
                               grp[1],
                               grp[2],
                               grp[3],
                               grp[4],
                               grp[5],
                               grp[6],
                               grp[7]);

        // Write to output (global) memory. Note that as written below, this writes
        // all 8 values, whether valid for the given QAM or not.
        grp.write(llr);
    }
}

// Per PRB equalizer coefficient application fused with soft demap
// Inputs and outputs assumed to be column major
// dimBlock: (N_BS_ANTS*N_PRB_TONES, N_THRD_BLK_DATA_SYMBS)
// dimGrid : (N_PRB, N_DATA_SYMBS/N_THRD_BLK_DATA_SYMBS, N_UE_GRPS)
// Note: NF = N_PRB_TONES * N_PRB
template <typename TStorageIn,
          typename TDataRx,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS,            // # of BS antenna (# of cols of C matrix) [difficult to remove as this is used by cooperative groups.]
          uint32_t N_SYMBS_PER_THRD_BLK, // # of data symbols processed by a thread block
          uint16_t SYMBOL_BITMASK>       
__device__ void
eqMmseSoftDemapKernel_v2(puschRxChEqStatDescr_t* pStatDescr, puschRxChEqSoftDemapDynDescr_t* pDynDescr)
{
    KERNEL_PRINT_GRID_ONCE("%s\n grid = (%u %u %u), block = (%u %u %u), N_BS_ANTS = %u, N_SYMBS_PER_THRD_BLK = %u\n",
                           __PRETTY_FUNCTION__,
                           gridDim.x, gridDim.y, gridDim.z,
                           blockDim.x, blockDim.y, blockDim.z,
                           N_BS_ANTS,
                           N_SYMBS_PER_THRD_BLK);                           
    //--------------------------------------------------------------------------------------------------------
    // Setup local parameters based on descriptor
    puschRxChEqSoftDemapDynDescr_t& dynDescr = *(pDynDescr);

    // Early exit check
    // The grid is sized to process the max # of PRBs in a given heterogenous config. Exit if the PRB to be
    // processed by this thread block does not exist in the UE group
    const uint32_t PRB_IDX  = blockIdx.x;
    const uint32_t UE_GRP_IDX = dynDescr.hetCfgUeGrpMap[blockIdx.z];

    // if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z)) printf("%s\n blockIdx.z %d UE_GRP_IDX %d\n", __PRETTY_FUNCTION__, blockIdx.z, UE_GRP_IDX);

    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];
    const uint32_t N_LAYERS = drvdUeGrpPrms.nLayers;

    const uint16_t nPrb = drvdUeGrpPrms.nPrb;
    uint16_t nDataSym = drvdUeGrpPrms.nDataSym;
    uint8_t nDmrsCdmGrpsNoData = drvdUeGrpPrms.nDmrsCdmGrpsNoData;
    if(nDmrsCdmGrpsNoData==1)
    {
        nDataSym += drvdUeGrpPrms.nDmrsSyms;
    }
    
    if(PRB_IDX >= nPrb) return;

    const uint16_t dmrsMaxLen  = drvdUeGrpPrms.dmrsMaxLen;
    const uint16_t nDmrsSym    = drvdUeGrpPrms.nDmrsSyms / dmrsMaxLen;

    //--------------------------------------------------------------------------------------------------------
    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TDataRx>::type     TComplexDataRx;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    typedef typename std::conditional<std::is_same<TStorageOut, __half>::value, float2, float4>::type TLlr; 
    // static_assert(sizeof(TLlr) == sizeof(float4));
    bool enableCfoCorrection = (0 != drvdUeGrpPrms.enableCfoCorrection) ? true : false;
    bool enableTdi = (0 != drvdUeGrpPrms.enablePuschTdi) ? true : false;
    if(SYMBOL_BITMASK==CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK)
    {
        enableCfoCorrection = false;
        enableTdi = false;
    }
    

    // clang-format off
    uint16_t startPrb           = drvdUeGrpPrms.startPrb;
    uint8_t *dataSymLoc         = drvdUeGrpPrms.dataSymLoc;
    uint8_t *dmrsSymLoc         = drvdUeGrpPrms.dmrsSymLoc;
    uint8_t *qam                = drvdUeGrpPrms.qam;
    uint8_t *pUeGrpLayerToUeIdx = drvdUeGrpPrms.ueGrpLayerToUeIdx;
    uint8_t  enableTfPrcd       = drvdUeGrpPrms.enableTfPrcd;

    tensor_ref<const TComplexStorageIn> tCoef       (drvdUeGrpPrms.tInfoEqCoef.pAddr         , drvdUeGrpPrms.tInfoEqCoef.strides         ); // (N_LAYERS, N_BS_ANTS, NF, NH)
    tensor_ref<const TComplexStorageIn> tCfoEst     (drvdUeGrpPrms.tInfoCfoEst.pAddr         , drvdUeGrpPrms.tInfoCfoEst.strides         ); // (MAX_ND_SUPPORTED, N_LAYERS)
    tensor_ref<const TStorageIn>        tReeDiagInv (drvdUeGrpPrms.tInfoReeDiagInv.pAddr     , drvdUeGrpPrms.tInfoReeDiagInv.strides     ); // (N_LAYERS, NF, NH)
    tensor_ref<const TComplexDataRx>    tDataRx     (drvdUeGrpPrms.tInfoDataRx.pAddr         , drvdUeGrpPrms.tInfoDataRx.strides         ); // (NF, ND, N_BS_ANTS)
    tensor_ref<TComplexStorageOut>      tDataEq     (drvdUeGrpPrms.tInfoDataEq.pAddr         , drvdUeGrpPrms.tInfoDataEq.strides         ); // (N_LAYERS, NF, ND)
    tensor_ref<TStorageOut>             tLlr        (drvdUeGrpPrms.tInfoLLR.pAddr            , drvdUeGrpPrms.tInfoLLR.strides            ); // (N_LLR, N_LAYERS, NF, ND)
    tensor_ref<TStorageOut>             tLlrCdm1    (drvdUeGrpPrms.tInfoLLRCdm1.pAddr        , drvdUeGrpPrms.tInfoLLRCdm1.strides        ); // (N_LLR, N_LAYERS, NF, ND)
#ifdef ENABLE_DEBUG
    tensor_ref<TComplexStorageOut>      tDbg        (drvdUeGrpPrms.tInfoChEqSoftDempDbg.pAddr, drvdUeGrpPrms.tInfoChEqSoftDempDbg.strides);
#endif
    // clang-format on
    

#ifdef ENABLE_DEBUG
     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     {  
         printf("Addr: tCoef %lp tCfoEst %lp tReeDiagInv %lp tDataRx %lp tDataEq %lp tLlr %lp\n", tCoef.addr, tCfoEst.addr, tReeDiagInv.addr, tDataRx.addr, tDataEq.addr, tLlr.addr);
         printf("nPrb        : %d \n", nPrb);

         printf("tCoef       : addr %lp strides[0] %d strides[1] %d strides[2] %d strides[3] %d\n", static_cast<const TComplexStorageIn*>(tCoef.addr)  , tCoef.strides[0]      , tCoef.strides[1]      , tCoef.strides[2]      , tCoef.strides[3]      );
         printf("tCfoEst     : addr %lp strides[0] %d strides[1] %d strides[2] %d strides[3] %d\n", static_cast<const TComplexStorageIn*>(tCfoEst.addr), tCfoEst.strides[0]    , tCfoEst.strides[1]    , tCfoEst.strides[2]    , tCfoEst.strides[3]    );
         printf("tReeDiagInv : addr %lp strides[0] %d strides[1] %d strides[2] %d strides[3] %d\n", static_cast<const TStorageIn*>(tReeDiagInv.addr)   , tReeDiagInv.strides[0], tReeDiagInv.strides[1], tReeDiagInv.strides[2], tReeDiagInv.strides[3]);
         printf("tDataRx     : addr %lp strides[0] %d strides[1] %d strides[2] %d strides[3] %d\n", static_cast<const TComplexDataRx*>(tDataRx.addr)   , tDataRx.strides[0]    , tDataRx.strides[1]    , tDataRx.strides[2]    , tDataRx.strides[3]    );
         printf("tDataEq     : addr %lp strides[0] %d strides[1] %d strides[2] %d strides[3] %d\n", static_cast<TComplexStorageOut*>(tDataEq.addr)     , tDataEq.strides[0]    , tDataEq.strides[1]    , tDataEq.strides[2]    , tDataEq.strides[3]    );
         printf("tLlr        : addr %lp strides[0] %d strides[1] %d strides[2] %d strides[3] %d\n", static_cast<TStorageOut*>(tLlr.addr)               , tLlr.strides[0]       , tLlr.strides[1]       , tLlr.strides[2]       , tLlr.strides[3]       );
         // printf("tDbg    strides[0] %d strides[1] %d strides[2] %d\n", tDbg.strides[0], tDbg.strides[1], tDbg.strides[2]);
     }
#endif  

    thread_block const& thisThrdBlk = this_thread_block();

    // @todos:
    // rename N_TONES_PER_PRB, N_SC to N_PRB_TONES
    // rename N_BS_ANTS to N_BB_PORTS
    // abbreviate _THREAD_ to _THRD_ to be consisten
    // replace FREQ with TONES

    //--------------------------------------------------------------------------------------------------------
    // Dimensions
    constexpr uint32_t N_IQ = std::min(2u,N_BS_ANTS); // 2 samples: 1 I + 1 Q, except for the case of N_BS_ANTS == 1

    constexpr uint32_t PRB_TONE_THRD_GRP_SIZE = N_BS_ANTS;
    static_assert(PRB_TONE_THRD_GRP_SIZE <= N_THREADS_PER_WARP, "using co-operative groups to compute an inner product across N_BS_ANTS");

    //--------------------------------------------------------------------------------------------------------
    // Compute indices used for element access
    const uint32_t N_THREADS = thisThrdBlk.size();
    const uint32_t THRD_IDX  = thisThrdBlk.thread_rank();

    // Indices used for kernel shared memory access and writing to global memory: BB port first, frequency next
    const uint32_t BB_PORT_IDX = threadIdx.x % N_BS_ANTS;
    const uint32_t FREQ_IDX    = threadIdx.x / N_BS_ANTS;

    const uint32_t GMEM_WR_FREQ_IDX = FREQ_IDX;

    // PRB index processed by this thread
    const uint32_t THRD_BLK_ABS_START_FREQ_IDX = PRB_IDX * CUPHY_N_TONES_PER_PRB;

    // Subcarrier sample location in global memory
    const uint32_t GMEM_ABS_WR_FREQ_IDX = THRD_BLK_ABS_START_FREQ_IDX + GMEM_WR_FREQ_IDX;
    const uint32_t GMEM_ABS_RD_FREQ_IDX = GMEM_ABS_WR_FREQ_IDX + 12 * startPrb;

    // const uint32_t chEqTimeInstIdx = 0; // @todo: add high mobility support i.e. handle multiple estimates of C in time

    const uint32_t DATA_SYMB_IDX           = threadIdx.y;
    const uint32_t DATA_SYMB_ABS_START_IDX = blockIdx.y * N_SYMBS_PER_THRD_BLK;
    const uint32_t DATA_SYMB_ABS_IDX       = DATA_SYMB_ABS_START_IDX + DATA_SYMB_IDX;

    if(DATA_SYMB_ABS_IDX >= nDataSym) return;
    
    if(SYMBOL_BITMASK==CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK)
    {
        if(!((SYMBOL_BITMASK>>dataSymLoc[DATA_SYMB_ABS_IDX])&1))
        {
            return;
        }
    }

    // 1 tile per PRB tone, each tile of size N_BS_ANTS
    // The PRB tone thread group is used to compute the inner product across BB ports
    thread_block_tile<PRB_TONE_THRD_GRP_SIZE> const& prbToneThrdGrp =
        tiled_partition<PRB_TONE_THRD_GRP_SIZE>(thisThrdBlk);

    const uint32_t PER_LAYER_THRD_IDX = prbToneThrdGrp.thread_rank();
    
    //--------------------------------------------------------------------------------------------------------
#if !EQ_SOFT_DEMAP_USE_TEX
    // Storage for LLRs
    __shared__ TLlr shLlr[CUPHY_N_TONES_PER_PRB * N_PAM_PER_QAM]; // [CUPHY_N_TONES_PER_PRB][N_PAM_PER_QAM];
    const uint32_t    LLR_START_IDX = FREQ_IDX * N_PAM_PER_QAM;

    // 2 scratchpad buffers, each of size MAX_BITS_PAM: one to store softBits and other to store minDist
    constexpr uint32_t PAM_LLR_COMP_WRK_BUF_LEN = 2 * MAX_BITS_PAM + 1;

    __shared__ TCompute shWrkBuf[CUPHY_N_TONES_PER_PRB * N_IQ * PAM_LLR_COMP_WRK_BUF_LEN];
    const uint32_t      WRK_BUF_START_IDX =
        ((FREQ_IDX * N_IQ) + PER_LAYER_THRD_IDX) * PAM_LLR_COMP_WRK_BUF_LEN;
#endif
    //--------------------------------------------------------------------------------------------------------
    // Each per PRB tone thread group co-operates to compute an inner product across N_BS_ANTS. However,
    // since data is laid out frequency first, we should transform from frequency first to BB port
    // first order.
    TComplexCompute regY = type_convert<TComplexCompute>(tDataRx(GMEM_ABS_RD_FREQ_IDX, dataSymLoc[DATA_SYMB_ABS_IDX], BB_PORT_IDX));

    //--------------------------------------------------------------------------------------------------------
    // Compute interpolated equalizer coefficients
    TCompute alpha1;
    TCompute alpha2;
    uint32_t dmrsIdx = 1;

    if (enableTdi && nDmrsSym > 1)
    {
         while ((dmrsIdx<(nDmrsSym-1))&&(dmrsSymLoc[(dmrsIdx+1) * dmrsMaxLen-1] < dataSymLoc[DATA_SYMB_ABS_IDX]))
         {
             dmrsIdx++;
         }


        alpha1 = (dmrsSymLoc[dmrsIdx*dmrsMaxLen] - dataSymLoc[DATA_SYMB_ABS_IDX]) / static_cast<TCompute>(dmrsSymLoc[dmrsIdx*dmrsMaxLen] - dmrsSymLoc[(dmrsIdx-1)*dmrsMaxLen]);
        alpha2 = (dataSymLoc[DATA_SYMB_ABS_IDX] - dmrsSymLoc[(dmrsIdx-1)*dmrsMaxLen]) / static_cast<TCompute>(dmrsSymLoc[dmrsIdx*dmrsMaxLen] - dmrsSymLoc[(dmrsIdx-1)*dmrsMaxLen]);
    }    

    //--------------------------------------------------------------------------------------------------------
    // Process
#ifdef ENABLE_DEBUG
    if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z))
    printf("eqMmseSoftDemapKernel_v2: %u\n", enableCfoCorrection);

    if((0 != blockIdx.x) || (0 != blockIdx.y)) return;
#endif

    for(int layerIdx = 0; layerIdx < N_LAYERS; ++layerIdx)
    {
        TComplexCompute prod;
        // Load QAM info for each layer
        uint8_t pamBitLen = qam[layerIdx] / 2;
        // Load noiseInv for each layer
        TCompute reeDiagInv;

        if (enableTdi && nDmrsSym > 1) {
            // Read coefficients, ReeDiagInv for the next iteration
            TComplexCompute regC1 = type_convert<TComplexCompute>(tCoef(BB_PORT_IDX, FREQ_IDX, layerIdx, PRB_IDX, (dmrsIdx-1)));
            TComplexCompute regC2 = type_convert<TComplexCompute>(tCoef(BB_PORT_IDX, FREQ_IDX, layerIdx, PRB_IDX, dmrsIdx));

            // Apply coefficients of a given layer to regY
            // Compute the product of coefficient vector C(0:N_BS_ANTS-1,freqIdx,layerIdx) with data vector
            // regY(0:N_BS_ANTS-1,freqIdx,symbIdx) using a co-operative group of N_BS_ANTS.
            // There is one such vector product per frequency bin
            TComplexCompute coef1Scaled = regC1 *alpha1; // complex * real - 2flops
            TComplexCompute coef2Scaled = regC2 *alpha2; // complex * real - 2flops
            prod = cuCaddf(coef1Scaled, coef2Scaled) * regY; // complex add (2flops) and complex * real (2flops)
            ////////////////////////////////////////////////////////
            //////// decide which DMRS symbol will be used /////////
            if(dataSymLoc[DATA_SYMB_ABS_IDX] <= dmrsSymLoc[dmrsMaxLen - 1])
            {
                reeDiagInv = type_convert<TCompute>(tReeDiagInv(FREQ_IDX, layerIdx, PRB_IDX, 0));
            }
            else
            {
                reeDiagInv = type_convert<TCompute>(tReeDiagInv(FREQ_IDX, layerIdx, PRB_IDX, dmrsIdx));
            }
        }
        else {
            // Read coefficients, ReeDiagInv for the next iteration
            TComplexCompute regC = type_convert<TComplexCompute>(tCoef(BB_PORT_IDX, FREQ_IDX, layerIdx, PRB_IDX, 0));

            // Apply coefficients of a given layer to regY
            // Compute the product of coefficient vector C(0:N_BS_ANTS-1,freqIdx,layerIdx) with data vector
            // regY(0:N_BS_ANTS-1,freqIdx,symbIdx) using a co-operative group of N_BS_ANTS.
            // There is one such vector product per frequency bin
            prod = regC * regY;
            reeDiagInv = type_convert<TCompute>(tReeDiagInv(FREQ_IDX, layerIdx, PRB_IDX, 0));
        }

#ifdef ENABLE_DEBUG
        printf("softDemap: C[%d][%d][%d][%d] = %f + j%f, y[%d][%d][%d] = %f + j%f, prod = %f + j%f\n", BB_PORT_IDX, FREQ_IDX, layerIdx, PRB_IDX, cuReal(shC(BB_PORT_IDX, FREQ_IDX)), cuImag(shC(BB_PORT_IDX, FREQ_IDX)), BB_PORT_IDX, FREQ_IDX, dataSymLoc(DATA_SYMB_ABS_IDX), cuReal(regY), cuImag(regY), cuReal(prod), cuImag(prod));
#endif

        // Accumulate within the co-operative group of N_BS_ANTS threads to produce one element
        // of the result: C*y
        // Note that the result should be visible to all threads
        TComplexCompute softEst = thrdGrpAllReduceSum<TComplexCompute, PRB_TONE_THRD_GRP_SIZE>(prbToneThrdGrp, prod);
        //if(PER_LAYER_THRD_IDX == 0) printf("total prod bx=%d by=%d bz=%d %d %f %f\n", blockIdx.x, blockIdx.y, blockIdx.z, prbToneThrdGrp.meta_group_rank(), softEst.x, softEst.y);

        if(enableCfoCorrection)
        {
            //printf("CFO: %d %d\n", dataSymLoc[DATA_SYMB_ABS_IDX], pUeGrpLayerToUeIdx[layerIdx]);
            softEst *= type_convert<TComplexCompute>(tCfoEst(dataSymLoc[DATA_SYMB_ABS_IDX], pUeGrpLayerToUeIdx[layerIdx]));
        }

#if 0
        if(enableTdi  && nDmrsSym > 1 && PER_LAYER_THRD_IDX == 0)
            printf("d1=%d d2=%d a1=%.3f a2=%.3f nDmrs=%d BB_PORT=%d layer=%d ABSF=%d dataSymLoc=%d dloc[d1]=%d dloc[d2]=%d shC1.x=%.3f shC1.y=%.3f shC2.x=%.3f shC2.y=%.3f data.x=%.3f, data.y=%.3f, m.x=%.3f m.y=%.3f m2.x=%.3f m2.y=%.3f softEst=%.3f %.3f %d %d %d threads=%d %d %d\n",
                   dmrsIdx-1,
                   dmrsIdx,
                   alpha1,
                   alpha2,
                   nDmrsSym,
                   BB_PORT_IDX,
                   layerIdx,
                   PRB_IDX * 12 + FREQ_IDX,
                   dataSymLoc[DATA_SYMB_ABS_IDX],
                   dmrsSymLoc[dmrsIdx-1],
                   dmrsSymLoc[dmrsIdx],
                   shC(BB_PORT_IDX, FREQ_IDX, dmrsIdx-1).x,
                   shC(BB_PORT_IDX, FREQ_IDX, dmrsIdx-1).y,
                   shC(BB_PORT_IDX, FREQ_IDX, dmrsIdx).x,
                   shC(BB_PORT_IDX, FREQ_IDX, dmrsIdx).y,
                   regY.x,
                   regY.y,
                   (make_cuFloatComplex(alpha1, 0) * shC(BB_PORT_IDX, FREQ_IDX, dmrsIdx-1)).x,
                   (make_cuFloatComplex(alpha1, 0) * shC(BB_PORT_IDX, FREQ_IDX, dmrsIdx-1)).y,
                   (make_cuFloatComplex(alpha2, 0) * shC(BB_PORT_IDX, FREQ_IDX, dmrsIdx)).x,
                   (make_cuFloatComplex(alpha2, 0) * shC(BB_PORT_IDX, FREQ_IDX, dmrsIdx)).y,
                   softEst.x,
                   softEst.y,
                   blockIdx.x, blockIdx.y,UE_GRP_IDX,threadIdx.x,threadIdx.y,threadIdx.z);  
#endif      
        if(enableTfPrcd==0)
        {          
            // Determine the output LLR address
            TStorageOut* LLRdst = tLlr.addr + tLlr.offset(PER_LAYER_THRD_IDX * MAX_BITS_PAM,
                                                          layerIdx,
                                                          GMEM_ABS_WR_FREQ_IDX,
                                                          DATA_SYMB_ABS_IDX);
            // Perform the soft demapping operation
            TStorageOut* LLRCdm1dst;
            uint8_t write_flag = 0;
            if((PER_LAYER_THRD_IDX==0)&&(nDmrsCdmGrpsNoData==1))
            {
                long addr_offset = 0;
                long addr_accum_per_data_sym = tLlr.strides[3];
                long addr_accum_per_dmrs_sym = (tLlr.strides[3]>>1);
                uint8_t dmrs_flag;
                for(uint8_t idx=0; idx < DATA_SYMB_ABS_IDX; idx++)
                {
                    dmrs_flag = 0;
                    for(uint8_t dmrs_idx=0; dmrs_idx<drvdUeGrpPrms.nDmrsSyms; dmrs_idx++)
                    {
                        if(dmrsSymLoc[dmrs_idx] == dataSymLoc[idx])
                        {
                            dmrs_flag = 1;
                            break;
                        }
                    }
                    if(dmrs_flag)
                    {
                        addr_offset += addr_accum_per_dmrs_sym;
                    }
                    else
                    {
                        addr_offset += addr_accum_per_data_sym;
                    }
                }

                dmrs_flag = 0;
                for(uint8_t dmrs_idx=0; dmrs_idx<drvdUeGrpPrms.nDmrsSyms; dmrs_idx++)
                {
                    if(dmrsSymLoc[dmrs_idx] == dataSymLoc[DATA_SYMB_ABS_IDX])
                    {
                        dmrs_flag = 1;
                        break;
                    }
                }
                if(dmrs_flag)
                {
                    if(GMEM_ABS_WR_FREQ_IDX%2)
                    {
                        addr_offset += (tLlr.strides[1]*layerIdx+tLlr.strides[2]*(GMEM_ABS_WR_FREQ_IDX>>1));
                        write_flag = 1;
                    }
                    else
                    {
                        write_flag = 0;
                    }
                }
                else
                {
                    addr_offset += (tLlr.strides[1]*layerIdx+tLlr.strides[2]*GMEM_ABS_WR_FREQ_IDX);
                    write_flag = 1;
                }
                LLRCdm1dst = tLlrCdm1.addr + addr_offset;
            }

#if(EQ_SOFT_DEMAP_USE_TEX == 2)

            ch_eq_simplified_soft_demapper<TStorageOut, TCompute>(PER_LAYER_THRD_IDX,        // PER_LAYER_THRD_IDX
                                                                  pamBitLen,                 // nPamBits
                                                                  reeDiagInv,                // noiseInv
                                                                  softEst,                   // softEst
                                                                  LLRdst);                   // LLR_output address

            if(write_flag)
            {
                ch_eq_simplified_soft_demapper<TStorageOut, TCompute>(PER_LAYER_THRD_IDX,     // PER_LAYER_THRD_IDX
                                                                      pamBitLen,              // nPamBits
                                                                      reeDiagInv,             // noiseInv
                                                                      softEst,                // softEst
                                                                      LLRCdm1dst);            // LLR_output address
            }

#elif(EQ_SOFT_DEMAP_USE_TEX == 1)

            ch_eq_soft_demapper_tex<TStorageOut, TCompute>(PER_LAYER_THRD_IDX,        // PER_LAYER_THRD_IDX
                                                           pamBitLen,              // nPamBits
                                                           reeDiagInv,             // noiseInv
                                                           softEst,                   // softEst
                                                           LLRdst,                    // LLR_output address
                                                           pStatDescr->demapper_tex); // texture object
                                                           
            if(write_flag)
            {
                ch_eq_soft_demapper_tex<TStorageOut, TCompute>(PER_LAYER_THRD_IDX,        // PER_LAYER_THRD_IDX
                                                               pamBitLen,              // nPamBits
                                                               reeDiagInv,             // noiseInv
                                                               softEst,                   // softEst
                                                               LLRCdm1dst,                // LLR_output address
                                                               pStatDescr->demapper_tex); // texture object
            }

#else
            ch_eq_soft_demapper<TStorageOut, TCompute>(PER_LAYER_THRD_IDX,               // PER_LAYER_THRD_IDX
                                                       pamBitLen,                     // nPamBits
                                                       reeDiagInv,                    // noiseInv
                                                       softEst,                          // softEst
                                                       layerIdx,                         // layerIdx
                                                       shWrkBuf,                         // shWrkBuf
                                                       shLlr,                            // shLlr
                                                       GMEM_ABS_WR_FREQ_IDX,             // GMEM_ABS_WR_FREQ_IDX
                                                       DATA_SYMB_ABS_IDX,                // DATA_SYMB_ABS_IDX
                                                       LLR_START_IDX,                    // LLR_START_IDX
                                                       WRK_BUF_START_IDX,                // WRK_BUF_START_IDX
                                                       reinterpret_cast<TLlr*>(LLRdst)); // LLR output address
#endif
    
#ifdef ENABLE_DEBUG
            // if((2 == LAYER_IDX) && (3122 == absFreqIdx) && (8 == dataSymLoc[DATA_COL_START_IDX + dataColIdx]))
            if((0 == layerIdx) && (GMEM_ABS_RD_FREQ_IDX < 10) && (0 == dataSymLoc[DATA_SYMB_ABS_IDX]))
                printf("computePamLlr: [%d][%d][%d] nPamBits = %d softEst = %f+j%f noiseInv = %f\n",
                       layerIdx,
                       GMEM_ABS_RD_FREQ_IDX,
                       dataSymLoc[DATA_SYMB_ABS_IDX],
                       pamBitLen,
                       cuReal(softEst),
                       cuImag(softEst),
                       reeDiagInv);
#endif
        }
        else if(enableTfPrcd==1)
        {
            tensor_ref<TComplexCompute>         tDataEqDft  (drvdUeGrpPrms.tInfoDataEqDft.pAddr      , drvdUeGrpPrms.tInfoDataEqDft.strides); // (NF*ND)
            if(0 == PER_LAYER_THRD_IDX)
            {
                tDataEqDft(GMEM_ABS_WR_FREQ_IDX + DATA_SYMB_ABS_IDX*nPrb*CUPHY_N_TONES_PER_PRB) = softEst;
            }
        }

#ifdef WRITE_EQ_OUTPUT
        // Pick one of the N_BS_ANTS threads to store the resulting soft estimate
        if(0 == PER_LAYER_THRD_IDX) tDataEq(layerIdx, GMEM_ABS_WR_FREQ_IDX, DATA_SYMB_ABS_IDX) = type_convert<TComplexStorageOut>(softEst);
        // if(0 == PER_LAYER_THRD_IDX) printf("DataEq[%d][%d][%d] = %f + j%f, %f + j%f\n", layerIdx, GMEM_ABS_WR_FREQ_IDX, DATA_SYMB_ABS_IDX, tDataEq(layerIdx, GMEM_ABS_WR_FREQ_IDX, DATA_SYMB_ABS_IDX).x, tDataEq(layerIdx, GMEM_ABS_WR_FREQ_IDX, DATA_SYMB_ABS_IDX).y, cuReal(softEst), cuImag(softEst));
        // if(((5 == UE_GRP_IDX) || (6 == UE_GRP_IDX) || (7 == UE_GRP_IDX) || (8 == UE_GRP_IDX)) && (PRB_IDX < 1) && (0 == PER_LAYER_THRD_IDX)) printf("ueGrpIdx %d blockIdx.z %d  DataEq[%d][%d][%d] = %f + j%f\n", UE_GRP_IDX, blockIdx.z, layerIdx, GMEM_ABS_WR_FREQ_IDX, DATA_SYMB_ABS_IDX, cuReal(softEst), cuImag(softEst));
#endif
    }
} //eqMmseSoftDemapKernel_v2


template <class FFT,
          typename TStorageIn,
          typename TDataRx,
          typename TStorageOut,
          typename TCompute> 
__global__ void bluestein_workspace_kernel(puschRxChEqStatDescr_t* pStatDescr, puschRxChEqSoftDemapDynDescr_t* pDynDescr)
{
    puschRxChEqSoftDemapDynDescr_t& dynDescr = *(pDynDescr); 
    const uint32_t UE_GRP_IDX = dynDescr.hetCfgUeGrpMap[blockIdx.z];
    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];
    uint8_t  enableTfPrcd       = drvdUeGrpPrms.enableTfPrcd;
    
    if(enableTfPrcd!=1) return;  
    
    uint16_t DFTSize = CUPHY_N_TONES_PER_PRB*drvdUeGrpPrms.nPrb;

    // caches current size of tDftBluesteinWorkspaceTime/Freq.  Initialized to zero.
    // static uint16_t dft_cache = 0;

    // if the size of the cached tDftBluesteinWorkspaceTime/Freq is correct, then don't bother re-generating.
    // if ( DFTSize != dft_cache ) 
    {
      
      extern __shared__ unsigned char shared_mem[];
      
      typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
      tensor_ref<TComplexCompute>         tDftBluesteinWorkspaceTime  (drvdUeGrpPrms.tInfoDftBluesteinWorkspaceTime.pAddr, drvdUeGrpPrms.tInfoDftBluesteinWorkspaceTime.strides); //FFT8192
      tensor_ref<TComplexCompute>         tDftBluesteinWorkspaceFreq  (drvdUeGrpPrms.tInfoDftBluesteinWorkspaceFreq.pAddr, drvdUeGrpPrms.tInfoDftBluesteinWorkspaceFreq.strides); //FFT8192
      using namespace cufftdx;
      
      /////////////////////////////blue workspace/////////////////////////////////////////////////////////////////////////////
      using blue_complex_type = typename FFT::value_type;
      uint16_t blue_fft_size = size_of<FFT>::value;
      blue_complex_type input[FFT::storage_size];
      
      // Generate w_time signal and store
      const unsigned int stride        = blue_fft_size / FFT::elements_per_thread;
      unsigned int       index         = threadIdx.x;
      unsigned int       compute_index = index;
      
      for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
	
	if (index >= DFTSize) {
	  compute_index = blue_fft_size - index;
        }
	
        blue_complex_type b_n = {0, 0};
	
        if (compute_index < DFTSize) {
	  // SR - the only reason to perform modulus here is to prevent a very large theta from reducing the accuracy of the subsequent sin and cos.
	  //      it doesn't seem to get THAT large - so removing since costly.  takes the kernel time (TVnr_7541) from 10us to 12us.
	  
	  const float theta = M_PI/DFTSize * ((compute_index * compute_index));  // % (2 * DFTSize));
	  b_n.x              = __cosf(theta); // SR - ensure that hardware intrinsics are being used.
	  b_n.y              = __sinf(theta); //      switching from (double precision) to (single precision plus intrinsics) dropped time from 18 us to 10 us.
        }
        // Store conjugated value in w_time
	input[i]      = b_n;
        b_n.y         = -b_n.y;
        tDftBluesteinWorkspaceTime(index) = (TComplexCompute){TCompute(b_n.x), TCompute(b_n.y)};
        index += stride;
        compute_index = index;
      }
      
      // Calculate w_freq
      FFT().execute(input, shared_mem);
      
      // Store w_freq
      index = threadIdx.x;
      for (unsigned int i = 0; i < FFT::elements_per_thread; i++) {
	tDftBluesteinWorkspaceFreq(index) = (TComplexCompute){TCompute(input[i].x), TCompute(input[i].y)}; 
	index += stride;
      }
      
      // dft_cache = DFTSize;

    }  // if ( DFTSize != dft_cache ) {

}



template <class FFT,
          typename TStorageIn,
          typename TDataRx,
          typename TStorageOut,
          typename TCompute,
          uint16_t SYMBOL_BITMASK>
__global__ void bluestein_Idft_kernel(puschRxChEqStatDescr_t* pStatDescr, puschRxChEqSoftDemapDynDescr_t* pDynDescr)
{
    puschRxChEqSoftDemapDynDescr_t& dynDescr = *(pDynDescr); 
    const uint32_t UE_GRP_IDX = dynDescr.hetCfgUeGrpMap[blockIdx.z];
    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];
    uint8_t  enableTfPrcd       = drvdUeGrpPrms.enableTfPrcd;

    if(enableTfPrcd!=1) return;
    
    uint16_t DFTSize = CUPHY_N_TONES_PER_PRB*drvdUeGrpPrms.nPrb;
    
    if(blockIdx.x>=drvdUeGrpPrms.nDataSym) return;
    
    uint8_t *dataSymLoc = drvdUeGrpPrms.dataSymLoc;
    if(SYMBOL_BITMASK==CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK)
    {
        if(!((SYMBOL_BITMASK>>dataSymLoc[blockIdx.x])&1))
        {
            return;
        }
    }
    
    extern __shared__ unsigned char shared_mem[];
    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    tensor_ref<TComplexCompute>         tDftBluesteinWorkspaceTime  (drvdUeGrpPrms.tInfoDftBluesteinWorkspaceTime.pAddr, drvdUeGrpPrms.tInfoDftBluesteinWorkspaceTime.strides); //FFT8192
    tensor_ref<TComplexCompute>         tDftBluesteinWorkspaceFreq  (drvdUeGrpPrms.tInfoDftBluesteinWorkspaceFreq.pAddr, drvdUeGrpPrms.tInfoDftBluesteinWorkspaceFreq.strides); //FFT8192
    tensor_ref<TComplexCompute>         tDataEqDft                  (drvdUeGrpPrms.tInfoDataEqDft.pAddr, drvdUeGrpPrms.tInfoDataEqDft.strides);                                 //(NF*ND)
    
    using namespace cufftdx;
    
    using complex_type = typename FFT::value_type;
    const unsigned int stride = size_of<FFT>::value / FFT::elements_per_thread;
    complex_type input[FFT::storage_size];
    
    assert(stride == FFT::max_threads_per_block);

    // SR - was Idft_1
    
    int indexIn = threadIdx.x + DFTSize * blockIdx.x;
    int indexWorkspaceTime = threadIdx.x;

    for (int i = 0; i < FFT::elements_per_thread; ++i){

      if((threadIdx.x + i * stride) < DFTSize) {
	// swap real<->imag for inverse FFT
	input[i].y = tDataEqDft(indexIn + i * stride).x;
	input[i].x = tDataEqDft(indexIn + i * stride).y;
        input[i] *= tDftBluesteinWorkspaceTime(indexWorkspaceTime);
        indexWorkspaceTime += stride;
      }
      else {
	input[i].x = 0.0f;
	input[i].y = 0.0f;
      }
    }
    
    FFT().execute(input, shared_mem);

    int indexOut = threadIdx.x + size_of<FFT>::value * blockIdx.x;
    int indexWorkspaceFreq = threadIdx.x;
    for (unsigned int i = 0; i < FFT::elements_per_thread; ++i) {
        input[i] *= tDftBluesteinWorkspaceFreq(indexWorkspaceFreq);
        input[i].y = -input[i].y; // conjugate
        indexWorkspaceFreq += stride;
    }

    FFT().execute(input, shared_mem);

    const double dftscale = 1.0/TCompute((size_of<FFT>::value)*sqrt(DFTSize));            // the compiler would do this just as well ... but looks cleaner this way.
    
    //////////////////////////////////////////////////////////////////////////////// 
    // We can limit the last loop to just max_meaningful_ept, other values are not needed.
    unsigned int max_meaningful_ept = (DFTSize + (stride - 1)) / stride;
    indexOut = threadIdx.x + DFTSize * blockIdx.x;
    indexWorkspaceTime = threadIdx.x;
    for (unsigned int i = 0; i < max_meaningful_ept; ++i) {

      if((threadIdx.x + i * stride) < DFTSize) {

        input[i].y = -input[i].y * dftscale; // conjugate and scale
        input[i].x = input[i].x*dftscale;

        input[i] *= tDftBluesteinWorkspaceTime(indexWorkspaceTime);

        indexWorkspaceTime += stride;

        // Make swap real<->imag for inverse FFT
        const auto tmp = input[i].x;
        input[i].x     = input[i].y;
        input[i].y     = tmp;
        tDataEqDft(indexOut + i * stride).x = input[i].x;
        tDataEqDft(indexOut + i * stride).y = input[i].y;

      }

    }
    
}

// Per PRB equalizer coefficient application fused with soft demap
// Inputs and outputs assumed to be column major
// dimBlock: (N_BS_ANTS*N_PRB_TONES, N_THRD_BLK_DATA_SYMBS)
// dimGrid : (N_PRB, N_DATA_SYMBS/N_THRD_BLK_DATA_SYMBS, N_UE_GRPS)
// Note: NF = N_PRB_TONES * N_PRB
template <typename TStorageIn,
          typename TDataRx,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS,            // # of BS antenna (# of cols of C matrix)
          uint32_t N_SYMBS_PER_THRD_BLK, // # of data symbols processed by a thread block
          uint16_t SYMBOL_BITMASK>
__device__ void
eqMmseSoftDemapAfterDftKernel_v2(puschRxChEqStatDescr_t* pStatDescr, puschRxChEqSoftDemapDynDescr_t* pDynDescr)
{
    //--------------------------------------------------------------------------------------------------------
    // Setup local parameters based on descriptor
    // puschRxChEqStatDescr_t&         statDescr = *(pStatDescr);
    puschRxChEqSoftDemapDynDescr_t& dynDescr = *(pDynDescr);

    // Early exit check
    // The grid is sized to process the max # of PRBs in a given heterogenous config. Exit if the PRB to be
    // processed by this thread block does not exist in the UE group
    const uint32_t PRB_IDX  = blockIdx.x;
    const uint32_t UE_GRP_IDX = dynDescr.hetCfgUeGrpMap[blockIdx.z];
    
    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];
    const uint32_t N_LAYERS = drvdUeGrpPrms.nLayers;

    uint8_t  enableTfPrcd       = drvdUeGrpPrms.enableTfPrcd;
    
    if(enableTfPrcd!=1) return;

    const uint16_t nPrb = drvdUeGrpPrms.nPrb;
    if(PRB_IDX >= nPrb) return;
    
    const uint16_t nDataSym = drvdUeGrpPrms.nDataSym;
    const uint16_t dmrsMaxLen = drvdUeGrpPrms.dmrsMaxLen;
    const uint16_t nDmrsSym   = drvdUeGrpPrms.nDmrsSyms / dmrsMaxLen;
    uint8_t *qam                = drvdUeGrpPrms.qam;
    uint8_t *dataSymLoc         = drvdUeGrpPrms.dataSymLoc;
    uint8_t *dmrsSymLoc         = drvdUeGrpPrms.dmrsSymLoc;

    //--------------------------------------------------------------------------------------------------------
    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TDataRx>::type     TComplexDataRx;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    typedef typename std::conditional<std::is_same<TStorageOut, __half>::value, float2, float4>::type TLlr; 
    // static_assert(sizeof(TLlr) == sizeof(float4));
    bool enableTdi = (0 != drvdUeGrpPrms.enablePuschTdi) ? true : false;
    if(SYMBOL_BITMASK==CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK)
    {
        enableTdi = false;
    }

    tensor_ref<const TStorageIn>        tReeDiagInv (drvdUeGrpPrms.tInfoReeDiagInv.pAddr     , drvdUeGrpPrms.tInfoReeDiagInv.strides     ); // (N_LAYERS, NF, NH)
    tensor_ref<TComplexCompute>         tDataEqDft  (drvdUeGrpPrms.tInfoDataEqDft.pAddr      , drvdUeGrpPrms.tInfoDataEqDft.strides      ); // (NF*ND)
    tensor_ref<TStorageOut>             tLlr        (drvdUeGrpPrms.tInfoLLR.pAddr            , drvdUeGrpPrms.tInfoLLR.strides            ); // (N_LLR, N_LAYERS, NF, ND)
    tensor_ref<TComplexStorageOut>      tDbg        (drvdUeGrpPrms.tInfoChEqSoftDempDbg.pAddr, drvdUeGrpPrms.tInfoChEqSoftDempDbg.strides);
    // clang-format on
    

    thread_block const& thisThrdBlk = this_thread_block();

    // @todos:
    // rename N_TONES_PER_PRB, N_SC to N_PRB_TONES
    // rename N_BS_ANTS to N_BB_PORTS
    // abbreviate _THREAD_ to _THRD_ to be consisten
    // replace FREQ with TONES

    //--------------------------------------------------------------------------------------------------------
    // Compute indices used for element access
    const uint32_t N_THREADS = thisThrdBlk.size();
    const uint32_t THRD_IDX  = thisThrdBlk.thread_rank();

    // Indices used for read access from GMEM: frequency first, BB port next
    // Y is laidout frequency first, so use threadIdx.x to read along frequency to perform a coalesced read
    const uint32_t GMEM_RD_FREQ_IDX    = threadIdx.x % CUPHY_N_TONES_PER_PRB;

    // Indices used for kernel shared memory access and writing to global memory: BB port first, frequency next
    const uint32_t FREQ_IDX    = threadIdx.x;

    const uint32_t GMEM_WR_FREQ_IDX = FREQ_IDX;

    // PRB index processed by this thread
    const uint32_t THRD_BLK_ABS_START_FREQ_IDX = PRB_IDX * CUPHY_N_TONES_PER_PRB;

    // Subcarrier sample location in global memory
    const uint32_t GMEM_ABS_WR_FREQ_IDX = THRD_BLK_ABS_START_FREQ_IDX + GMEM_WR_FREQ_IDX;

    // const uint32_t chEqTimeInstIdx = 0; // @todo: add high mobility support i.e. handle multiple estimates of C in time

    const uint32_t DATA_SYMB_IDX           = threadIdx.y;
    const uint32_t DATA_SYMB_ABS_START_IDX = blockIdx.y * N_SYMBS_PER_THRD_BLK;
    const uint32_t DATA_SYMB_ABS_IDX       = DATA_SYMB_ABS_START_IDX + DATA_SYMB_IDX;

    if (DATA_SYMB_ABS_IDX >= nDataSym) return;
    
    if(SYMBOL_BITMASK==CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK)
    {
        if(!((SYMBOL_BITMASK>>dataSymLoc[DATA_SYMB_ABS_IDX])&1))
        {
            return;
        }
    }

    uint16_t dmrsIdx = 0;
    if (enableTdi && nDmrsSym > 1)
    {
        while((dmrsIdx<(nDmrsSym-1))&&(dmrsSymLoc[(dmrsIdx+1) * dmrsMaxLen-1]<dataSymLoc[DATA_SYMB_ABS_IDX]))
        {
            dmrsIdx++;
        }
    }

    // Process
    for(int layerIdx = 0; layerIdx < N_LAYERS; ++layerIdx)
    {
        // Load QAM info for each layer
        uint8_t pamBitLen = qam[layerIdx] / 2;
        TCompute reeDiagInv = type_convert<TCompute>(tReeDiagInv(GMEM_RD_FREQ_IDX, layerIdx, PRB_IDX, dmrsIdx));
        TComplexCompute softEst = tDataEqDft(GMEM_ABS_WR_FREQ_IDX + DATA_SYMB_ABS_IDX*nPrb*CUPHY_N_TONES_PER_PRB);
//        /////DFT for DFT-s-OFDM/////
//        TComplexCompute softEst = cuGet<TComplexCompute>(0);
//        uint32_t sizeDft = nPrb*CUPHY_N_TONES_PER_PRB;
//        
//        for(uint32_t idxDft = 0; idxDft < sizeDft; idxDft++)
//        {
//            TComplexCompute coeffDft;
//            coeffDft.x = (TCompute)cos(2 * M_PI / sizeDft * idxDft * GMEM_ABS_WR_FREQ_IDX);
//            coeffDft.y = (TCompute)sin(2 * M_PI / sizeDft * idxDft * GMEM_ABS_WR_FREQ_IDX);
//            softEst += (tDataEqDft(idxDft + DATA_SYMB_ABS_IDX*nPrb*CUPHY_N_TONES_PER_PRB) * coeffDft);
//        }
//        softEst.x *= (TCompute)(1/sqrt(sizeDft));
//        softEst.y *= (TCompute)(1/sqrt(sizeDft));
        ////////////////////////////////////////////////    
        
        if(pamBitLen == 0)
        {
            // for pi/2 BPSK (cuphy_api.h: qamModOrder Value: 2,4,6,8 if transform precoding is disabled; 1,2,4,6,8 if transform precoding is enabled)
            if(GMEM_ABS_WR_FREQ_IDX%2==1)
            {
                TComplexCompute coeff;
                coeff.x = (TCompute)cos((-0.75f) * M_PI);
                coeff.y = (TCompute)sin((-0.75f) * M_PI);
                softEst = softEst * coeff;
            }
            else
            {
                TComplexCompute coeff;
                coeff.x = (TCompute)cos((-0.25f) * M_PI);
                coeff.y = (TCompute)sin((-0.25f) * M_PI);
                softEst = softEst * coeff;
            }
        }
        // Determine the output LLR address
        TStorageOut* LLRdst = tLlr.addr + tLlr.offset(0 * MAX_BITS_PAM,
                                                      layerIdx,
                                                      GMEM_ABS_WR_FREQ_IDX,
                                                      DATA_SYMB_ABS_IDX);
        // Perform the soft demapping operation
#if(EQ_SOFT_DEMAP_USE_TEX == 2)

        ch_eq_simplified_soft_demapper<TStorageOut, TCompute>(0,                         // PER_LAYER_THRD_IDX
                                                              pamBitLen,                 // nPamBits
                                                              reeDiagInv,                // noiseInv
                                                              softEst,                   // softEst
                                                              LLRdst);                   // LLR_output address


#elif(EQ_SOFT_DEMAP_USE_TEX == 1)
        // making sure layerIdx not
        // Overrunning array pamBitLen of 2 bytes at byte offset 2 using index layerIdx (which evaluates to 2).
        // coverity[overrun]
        ch_eq_soft_demapper_tex<TStorageOut, TCompute>(0,                         // PER_LAYER_THRD_IDX
                                                       pamBitLen,                 // nPamBits
                                                       reeDiagInv,                // noiseInv
                                                       softEst,                   // softEst
                                                       LLRdst,                    // LLR_output address
                                                       pStatDescr->demapper_tex); // texture object

#endif
    }
} //eqMmseSoftDemapAfterDftKernel_v2

#ifndef FAST_COMPILE
// Soft demapper for 64 antenna, one warp processes 64 ant's data
// Per PRB equalizer coefficient application fused with soft demap
// Inputs and outputs assumed to be column major
// dimBlock: (N_BS_ANTS*N_PRB_TONES, N_THRD_BLK_DATA_SYMBS)
// dimGrid : (N_PRB, N_DATA_SYMBS/N_THRD_BLK_DATA_SYMBS)
// Note: NF = N_PRB_TONES * N_PRB
template <typename TStorageIn,
          typename TDataRx,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_LAYERS,             // # of layers (# of rows of C matrix)
          uint32_t N_SYMBS_PER_THRD_BLK> // # of data symbols processed by a thread block
__device__ void
eqMmseSoftDemapKernel_v3(puschRxChEqStatDescr_t* pStatDescr, puschRxChEqSoftDemapDynDescr_t* pDynDescr)
{
    KERNEL_PRINT_GRID_ONCE("%s\n grid = (%u %u %u), block = (%u %u %u), N_BS_ANTS = %u, N_LAYERS = %u, N_SYMBS_PER_THRD_BLK = %u\n",
                            __PRETTY_FUNCTION__,
                            gridDim.x, gridDim.y, gridDim.z,
                            blockDim.x, blockDim.y, blockDim.z,
                            64,
                            N_LAYERS,
                            N_SYMBS_PER_THRD_BLK);
    //--------------------------------------------------------------------------------------------------------
    // Setup local parameters based on descriptor
    // puschRxChEqStatDescr_t&         statDescr = *(pStatDescr);
    puschRxChEqSoftDemapDynDescr_t& dynDescr = *(pDynDescr);

    // Early exit check
    // The grid is sized to process the max # of PRBs in a given heterogenous config. Exit if the PRB to be
    // processed by this thread block does not exist in the UE group
    // PRB index processed by this thread
    const uint32_t PRB_IDX  = blockIdx.x;
    const uint32_t UE_GRP_IDX = dynDescr.hetCfgUeGrpMap[blockIdx.z];

    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];

    const uint16_t nPrb = drvdUeGrpPrms.nPrb;
    if(PRB_IDX >= nPrb) return;

    //--------------------------------------------------------------------------------------------------------
    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TDataRx>::type     TComplexDataRx;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    typedef typename std::conditional<std::is_same<TStorageOut, __half>::value, float2, float4>::type TLlr; 
    // static_assert(sizeof(TLlr) == sizeof(float4));

    // clang-format off
    uint16_t startPrb   = drvdUeGrpPrms.startPrb;
    uint8_t *dataSymLoc = drvdUeGrpPrms.dataSymLoc;
    uint8_t *qam        = drvdUeGrpPrms.qam;
    tensor_ref<const TComplexStorageIn> tCoef       (drvdUeGrpPrms.tInfoEqCoef.pAddr         , drvdUeGrpPrms.tInfoEqCoef.strides         ); // (N_LAYERS, N_BS_ANTS, NF, NH)
    tensor_ref<const TStorageIn>        tReeDiagInv (drvdUeGrpPrms.tInfoReeDiagInv.pAddr     , drvdUeGrpPrms.tInfoReeDiagInv.strides     ); // (N_LAYERS, NF, NH)
    tensor_ref<const TComplexDataRx>    tDataRx     (drvdUeGrpPrms.tInfoDataRx.pAddr         , drvdUeGrpPrms.tInfoDataRx.strides         ); // (NF, ND, N_BS_ANTS)
    tensor_ref<TComplexStorageOut>      tDataEq     (drvdUeGrpPrms.tInfoDataEq.pAddr         , drvdUeGrpPrms.tInfoDataEq.strides         ); // (N_LAYERS, NF, ND)
    tensor_ref<TStorageOut>             tLlr        (drvdUeGrpPrms.tInfoLLR.pAddr            , drvdUeGrpPrms.tInfoLLR.strides            ); // (N_LLR, N_LAYERS, NF, ND)
    tensor_ref<TComplexStorageOut>      tDbg        (drvdUeGrpPrms.tInfoChEqSoftDempDbg.pAddr, drvdUeGrpPrms.tInfoChEqSoftDempDbg.strides);
    // clang-format on

    thread_block const& thisThrdBlk = this_thread_block();

    // @todos:
    // rename N_TONES_PER_PRB, N_SC to N_PRB_TONES
    // rename N_BS_ANTS to N_BB_PORTS
    // abbreviate _THREAD_ to _THRD_ to be consisten
    // replace FREQ with TONES

    //--------------------------------------------------------------------------------------------------------
    // Dimensions
    constexpr uint32_t N_BS_ANTS = 64; // Customized for 64 ANT
    constexpr uint32_t N_IQ = 2; // 2 samples: 1 I + 1 Q
    static_assert(N_BS_ANTS >= N_IQ, "Need N_IQ threads per PRB tone for soft demap compute");

    // Y         : Input data vector to be equalized
    constexpr uint32_t N_ROWS_Y = N_BS_ANTS;
    constexpr uint32_t N_COLS_Y = CUPHY_N_TONES_PER_PRB;

    // if bs_ants > 32, use one warp with a loop. For 64 ants, use one warps iterating twice to do the dot product
    constexpr uint32_t PRB_TONE_THRD_GRP_SIZE = (N_BS_ANTS > N_THREADS_PER_WARP) ? N_THREADS_PER_WARP : N_BS_ANTS;
    static_assert(PRB_TONE_THRD_GRP_SIZE <= N_THREADS_PER_WARP, "using co-operative groups to compute an inner product across N_BS_ANTS");

    // C         : coefficient matrix
    constexpr uint32_t N_ROWS_C = N_BS_ANTS;
    constexpr uint32_t N_COLS_C = CUPHY_N_TONES_PER_PRB;

    // ReeDiagInv: Inverse of the diagonal of residual error covariance matrix
    // constexpr uint32_t N_ELEMS_REE_DIAG_INV = CUPHY_N_TONES_PER_PRB;

    constexpr uint32_t N_INST = 2; // double buffer for pipelining

    //--------------------------------------------------------------------------------------------------------
    // Compute indices used for element access
    const uint32_t N_THREADS = thisThrdBlk.size();
    const uint32_t THRD_IDX  = thisThrdBlk.thread_rank();

    // Indices used for read access from GMEM: frequency first, BB port next
    // Y is laidout frequency first, so use threadIdx.x to read along frequency to perform a coalesced read
    const uint32_t GMEM_RD_FREQ_IDX    = threadIdx.x % CUPHY_N_TONES_PER_PRB;
    const uint32_t GMEM_RD_BB_PORT_IDX = threadIdx.x / CUPHY_N_TONES_PER_PRB;

    const uint32_t SMEM_WR_FREQ_IDX    = GMEM_RD_FREQ_IDX;
    const uint32_t SMEM_WR_BB_PORT_IDX = GMEM_RD_BB_PORT_IDX;
    // const uint32_t SMEM_WR_LAYER_IDX   = SMEM_WR_BB_PORT_IDX; // Needs boundary check with N_LAYERS

    // Indices used for kernel shared memory access and writing to global memory: BB port first, frequency next
    // uint32_t BB_PORT_IDX = threadIdx.x % N_BS_ANTS;
    // uint32_t FREQ_IDX    = threadIdx.x / N_BS_ANTS;
    const uint32_t BB_PORT_IDX = threadIdx.x & ((N_BS_ANTS >> 1) - 1);
    const uint32_t FREQ_IDX    = threadIdx.x >> 5;

    const uint32_t GMEM_WR_FREQ_IDX = FREQ_IDX;

    const uint32_t THRD_BLK_ABS_START_FREQ_IDX = PRB_IDX * CUPHY_N_TONES_PER_PRB;

    // Subcarrier sample location in global memory
    const uint32_t GMEM_ABS_RD_FREQ_IDX = THRD_BLK_ABS_START_FREQ_IDX + GMEM_RD_FREQ_IDX + 12*startPrb;
    const uint32_t GMEM_ABS_WR_FREQ_IDX = THRD_BLK_ABS_START_FREQ_IDX + GMEM_WR_FREQ_IDX;

    // const uint32_t chEqTimeInstIdx = 0; // @todo: add high mobility support i.e. handle multiple estimates of C in time

    const uint32_t DATA_SYMB_IDX           = threadIdx.y;
    const uint32_t DATA_SYMB_ABS_START_IDX = blockIdx.y * N_SYMBS_PER_THRD_BLK;
    const uint32_t DATA_SYMB_ABS_IDX       = DATA_SYMB_ABS_START_IDX + DATA_SYMB_IDX;

    const uint32_t SMEM_WR_DATA_SYMB_IDX = DATA_SYMB_IDX;

    // 1 tile per PRB tone, each tile of size N_BS_ANTS
    // The PRB tone thread group is used to compute the inner product across BB ports
    thread_block_tile<PRB_TONE_THRD_GRP_SIZE> const& prbToneThrdGrp =
        tiled_partition<PRB_TONE_THRD_GRP_SIZE>(thisThrdBlk);

    const uint32_t PER_LAYER_THRD_IDX = prbToneThrdGrp.thread_rank();

    //--------------------------------------------------------------------------------------------------------
    // SMEM1 allocation: C[N_INST], Y[N_SYMBS_PER_THRD_BLK] are overlaid (C, Y objects are both of the same size)
    constexpr uint32_t N_INST_SMEM       = (N_SYMBS_PER_THRD_BLK > N_INST) ? N_SYMBS_PER_THRD_BLK : N_INST;
    constexpr uint32_t N_SMEM_BLK1_ELEMS = (((N_ROWS_C + 1) * N_COLS_C) * N_INST_SMEM);
    __shared__ TComplexCompute smemBlk1[N_SMEM_BLK1_ELEMS];

    // SMEM2 allocation: ReeDiagInv
    // constexpr uint32_t N_SMEM_BLK2_ELEMS = (N_ELEMS_REE_DIAG_INV + 1) * N_INST;
    // __shared__ TCompute smemBlk2[N_SMEM_BLK2_ELEMS];

    constexpr uint32_t                                         SMEM_START_OFFSET_C = 0;
    block_3D<TComplexCompute*, N_ROWS_C + 1, N_COLS_C, N_INST> shC(&smemBlk1[SMEM_START_OFFSET_C]);

    // Y ooverlaid with  C
    constexpr uint32_t                                                       SMEM_START_OFFSET_Y = SMEM_START_OFFSET_C;
    block_3D<TComplexCompute*, N_ROWS_Y + 1, N_COLS_Y, N_SYMBS_PER_THRD_BLK> shY(&smemBlk1[SMEM_START_OFFSET_Y]);

    // constexpr uint32_t                                    SMEM_START_OFFSET_REE_DIAG_INV = 0;
    // block_2D<TCompute*, N_ELEMS_REE_DIAG_INV + 1, N_INST> shReeDiagInv(&smemBlk2[SMEM_START_OFFSET_REE_DIAG_INV]);
#if !EQ_SOFT_DEMAP_USE_TEX
    // Storage for LLRs
    __shared__ TLlr shLlr[CUPHY_N_TONES_PER_PRB * N_PAM_PER_QAM]; // [CUPHY_N_TONES_PER_PRB][N_PAM_PER_QAM];
    const uint32_t    LLR_START_IDX = FREQ_IDX * N_PAM_PER_QAM;

    // 2 scratchpad buffers, each of size MAX_BITS_PAM: one to store softBits and other to store minDist
    constexpr uint32_t PAM_LLR_COMP_WRK_BUF_LEN = 2 * MAX_BITS_PAM + 1;

    __shared__ TCompute shWrkBuf[CUPHY_N_TONES_PER_PRB * N_IQ * PAM_LLR_COMP_WRK_BUF_LEN];
    const uint32_t      WRK_BUF_START_IDX =
        ((FREQ_IDX * N_IQ) + PER_LAYER_THRD_IDX) * PAM_LLR_COMP_WRK_BUF_LEN;
#endif

    //--------------------------------------------------------------------------------------------------------
    // Load one time information: data to be equalized into shared memory

    // Each per PRB tone thread group co-operates to compute an inner product across N_BS_ANTS. However,
    // since data is laid out frequency first, use shared memory to transform from frequency first to BB port
    // first order.
    // Threads load gmemY[N_PRB_TONES][N_BS_ANTS][N_DATA_SYMBS_PER_THRD_BLK] to shmemY[N_BS_ANTS][N_PRB_TONES][N_DATA_SYMBS_PER_THRD_BLK]
    shY(SMEM_WR_BB_PORT_IDX, SMEM_WR_FREQ_IDX, SMEM_WR_DATA_SYMB_IDX) =
        type_convert<TComplexCompute>(tDataRx(GMEM_ABS_RD_FREQ_IDX, dataSymLoc[DATA_SYMB_ABS_IDX], GMEM_RD_BB_PORT_IDX));
    shY(SMEM_WR_BB_PORT_IDX + PRB_TONE_THRD_GRP_SIZE, SMEM_WR_FREQ_IDX, SMEM_WR_DATA_SYMB_IDX) =
        type_convert<TComplexCompute>(tDataRx(GMEM_ABS_RD_FREQ_IDX, dataSymLoc[DATA_SYMB_ABS_IDX], GMEM_RD_BB_PORT_IDX + PRB_TONE_THRD_GRP_SIZE));

    thisThrdBlk.sync();
    // TComplexCompute Y = shY(BB_PORT_IDX, FREQ_IDX, DATA_SYMB_IDX);

    //--------------------------------------------------------------------------------------------------------
    // Process
#ifdef ENABLE_DEBUG
    if((0 != blockIdx.x) || (0 != blockIdx.y)) return;
#endif

    //--------------------------------------------------------------------------------------------------------
    // Prefetch Coef for all N_SC subcarriers, all N_LAYERS and all N_BS_ANTS BB ports into registers

    // Prologue
    // Read coefficients, ReeDiagInv for the first iteration
    uint32_t currIdx     = 0;
    uint32_t nxtIdx      = currIdx ^ 0x1;
    uint32_t layerIdx    = 0;
    uint32_t nxtLayerIdx = layerIdx + 1;

    TComplexCompute C;

    // #pragma unroll
    for(nxtLayerIdx = layerIdx + 1; nxtLayerIdx <= N_LAYERS; ++nxtLayerIdx)
    {
        // Load QAM info for each layer
        uint8_t pamBitLen = qam[layerIdx] / 2;
        TComplexCompute prod = {0};
        // #pragma unroll
        for (uint32_t bbPortOffset = 0; bbPortOffset + PRB_TONE_THRD_GRP_SIZE <= N_BS_ANTS; bbPortOffset += PRB_TONE_THRD_GRP_SIZE)
        {
            TComplexCompute Y = shY(BB_PORT_IDX + bbPortOffset, FREQ_IDX, DATA_SYMB_IDX);

            // Read coefficients, ReeDiagInv for the next iteration
            if((0 == DATA_SYMB_IDX))
            {
                // shC(BB_PORT_IDX, FREQ_IDX, nxtIdx) =
                //     type_convert<TComplexCompute>(tCoef(BB_PORT_IDX, FREQ_IDX, nxtLayerIdx, PRB_IDX));
                C = type_convert<TComplexCompute>(tCoef(BB_PORT_IDX + bbPortOffset, PRB_IDX * 12 + FREQ_IDX, layerIdx, 0));
            }

            // Apply coefficients of a given layer to Y

            // Compute the product of coefficient vector C(0:N_BS_ANTS-1,freqIdx,layerIdx) with data vector
            // Y(0:N_BS_ANTS-1,freqIdx,symbIdx) using a co-operative group of N_BS_ANTS.
            // There is one such vector product per frequency bin
            // TComplexCompute prod = shC(BB_PORT_IDX, FREQ_IDX, currIdx) * Y;
            prod += C * Y;

#ifdef ENABLE_DEBUG
            printf("softDemap: C[%d][%d][%d][%d] = %f + j%f, y[%d][%d][%d] = %f + j%f, prod = %f + j%f\n", BB_PORT_IDX, FREQ_IDX, layerIdx, PRB_IDX, cuReal(shC(BB_PORT_IDX, FREQ_IDX, currIdx)), cuImag(shC(BB_PORT_IDX, FREQ_IDX, currIdx)), BB_PORT_IDX, FREQ_IDX, dataSymLoc(DATA_SYMB_ABS_IDX), cuReal(Y), cuImag(Y), cuReal(prod), cuImag(prod));
#endif
        }

        // Accumulate within the co-operative group of N_BS_ANTS threads to produce one element
        // of the result: C*y
        // Note that the result should be visible to all threads
        TComplexCompute softEst = thrdGrpAllReduceSum<TComplexCompute, PRB_TONE_THRD_GRP_SIZE>(prbToneThrdGrp, prod);
        TStorageOut* LLRdst = tLlr.addr + tLlr.offset(PER_LAYER_THRD_IDX * MAX_BITS_PAM,
                                layerIdx,
                                GMEM_ABS_WR_FREQ_IDX,
                                DATA_SYMB_ABS_IDX);

            TCompute noiseInv = tReeDiagInv(GMEM_RD_FREQ_IDX, layerIdx, PRB_IDX);

#if EQ_SOFT_DEMAP_USE_TEX
        ch_eq_soft_demapper_tex<TStorageOut, TCompute>(PER_LAYER_THRD_IDX,              // PER_LAYER_THRD_IDX
                                                       pamBitLen,                       // nPamBits
                                                       noiseInv,                        // noiseInv
                                                       softEst,                         // softEst
                                                       LLRdst,                          // LLR_output address
                                                       pStatDescr->demapper_tex);       // texture object
#else
        ch_eq_soft_demapper<TStorageOut, TCompute>(PER_LAYER_THRD_IDX,               // PER_LAYER_THRD_IDX
                                                   pamBitLen,                        // nPamBits
                                                   noiseInv,                         // noiseInv
                                                   softEst,                          // softEst
                                                   layerIdx,                         // layerIdx
                                                   shWrkBuf,                         // shWrkBuf
                                                   shLlr,                            // shLlr
                                                   GMEM_ABS_WR_FREQ_IDX,             // GMEM_ABS_WR_FREQ_IDX
                                                   DATA_SYMB_ABS_IDX,                // DATA_SYMB_ABS_IDX
                                                   LLR_START_IDX,                    // LLR_START_IDX
                                                   WRK_BUF_START_IDX,                // WRK_BUF_START_IDX
                                                   reinterpret_cast<TLlr*>(LLRdst)); // LLR output address
#endif

#ifdef WRITE_EQ_OUTPUT
        // Pick one of the N_BS_ANTS threads to store the resulting soft estimate
        if(2 == PER_LAYER_THRD_IDX) tDataEq(layerIdx, GMEM_ABS_WR_FREQ_IDX, DATA_SYMB_ABS_IDX) = type_convert<TComplexStorageOut>(softEst);
#endif

        currIdx = nxtIdx;
        nxtIdx ^= 0x1;
        layerIdx = nxtLayerIdx;
    }
}
#endif

template <typename TStorageIn,
          typename TDataRx,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS,            // # of BS antenna (# of cols of C matrix)
          uint32_t N_SYMBS_PER_THRD_BLK, // # of data symbols processed by a thread block
          uint16_t SYMBOL_BITMASK>       
// NOTE: currently using 48 as desired number of warps per SM in launch bound settings yields the best perf.
// However, there is risk if the code changes, even slightly, register reduction heuristics of the compiler fail to minimize registers the same way and
// this would result in larger register spills and overall perf regression.
// To be on the conservative side, 30 warps is used as desired number of resident warps, but if there is a need for further tuning in the future, 39 or 48 warps could be tested.
__launch_bounds__(CUPHY_N_TONES_PER_PRB * N_BS_ANTS * N_SYMBS_PER_THRD_BLK, (30/*desired number of warps per SM*/* N_THREADS_PER_WARP)/(CUPHY_N_TONES_PER_PRB * N_BS_ANTS * N_SYMBS_PER_THRD_BLK))
__global__ void
eqMmseSoftDemapKernel(puschRxChEqStatDescr_t* pStatDescr, puschRxChEqSoftDemapDynDescr_t* pDynDescr)
{
    KERNEL_PRINT_GRID_ONCE("%s\n grid = (%u %u %u), block = (%u %u %u), N_BS_ANTS = %u, N_SYMBS_PER_THRD_BLK = %u\n",
                           __PRETTY_FUNCTION__,
                           gridDim.x, gridDim.y, gridDim.z,
                           blockDim.x, blockDim.y, blockDim.z,
                           N_BS_ANTS,
                           N_SYMBS_PER_THRD_BLK);

    eqMmseSoftDemapKernel_v2<TStorageIn,
                             TDataRx,
                             TStorageOut,
                             TCompute,
                             N_BS_ANTS,
                             N_SYMBS_PER_THRD_BLK,
                             SYMBOL_BITMASK>(pStatDescr, pDynDescr);
}


template <typename TStorageIn,
          typename TDataRx,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS, // # of BS antenna (# of cols of C matrix)
          uint32_t N_SYMBS_PER_THRD_BLK, // # of data symbols processed by a thread block
          uint16_t SYMBOL_BITMASK>
__global__ void
eqMmseSoftDemapAfterDftKernel(puschRxChEqStatDescr_t* pStatDescr, puschRxChEqSoftDemapDynDescr_t* pDynDescr)
{
    KERNEL_PRINT_GRID_ONCE("%s\n grid = (%u %u %u), block = (%u %u %u), N_BS_ANTS = %u, N_SYMBS_PER_THRD_BLK = %u\n",
                           __PRETTY_FUNCTION__,
                           gridDim.x, gridDim.y, gridDim.z,
                           blockDim.x, blockDim.y, blockDim.z,
                           N_BS_ANTS,
                           N_SYMBS_PER_THRD_BLK);
                           
    eqMmseSoftDemapAfterDftKernel_v2<TStorageIn,
                                     TDataRx,
                                     TStorageOut,
                                     TCompute,
                                     N_BS_ANTS,
                                     N_SYMBS_PER_THRD_BLK,
                                     SYMBOL_BITMASK>(pStatDescr, pDynDescr);
} //eqMmseSoftDemapAfterDftKernel
#ifndef FAST_COMPILE
template <typename TStorageIn,
          typename TDataRx,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_LAYERS,  // # of layers (# of rows of C matrix)
          uint32_t N_SYMBS_PER_THRD_BLK> // # of data symbols processed by a thread block
__global__ void
eqMmseSoftDemapKernel_64R(puschRxChEqStatDescr_t* pStatDescr, puschRxChEqSoftDemapDynDescr_t* pDynDescr)
{
    eqMmseSoftDemapKernel_v3<TStorageIn,
                            TDataRx,
                            TStorageOut,
                            TCompute,
                            N_LAYERS,
                            N_SYMBS_PER_THRD_BLK>(pStatDescr, pDynDescr);
}
#endif

template <uint32_t N_LAYERS,
          uint32_t N_THRD_BLK_TONES,
          uint32_t N_TONES_PER_ITER> // # of layers (# of cols in H matrix)
void
puschRxChEq::coefCompMassiveMimoKernelLaunchGeo(uint16_t nPrb,
                                                uint16_t nUeGrps,
                                                dim3&    gridDim,
                                                dim3&    blockDim)
{
    constexpr uint32_t N_THRD_BLK_PER_PRB = (CUPHY_N_TONES_PER_PRB / N_THRD_BLK_TONES);
    // (N_BS_ANTS+N_LAYERS) > (N_BS_ANTS*N_LAYERS) if N_LAYERS = 1
    // const uint32_t N_THRDS_PER_TONE = std::max(N_BS_ANTS * N_LAYERS, N_BS_ANTS + N_LAYERS);
    // const uint32_t N_THRDS_PER_TONE = round_up_to_next<uint32_t>(N_BS_ANTS + (2 * N_LAYERS), N_THREADS_PER_WARP);
    // const uint32_t     N_THRDS_PER_TONE   = N_BS_ANTS + (2 * N_LAYERS);

    const uint32_t N_THRDS_PER_TONE = N_LAYERS * N_LAYERS;

    static_assert(0 == (CUPHY_N_TONES_PER_PRB % N_THRD_BLK_TONES), "Number of tones processed per thread block must be a multiple of PRB size");
    static_assert(0 == (N_THRD_BLK_TONES % N_TONES_PER_ITER), "Number of tones processed per iteration must be a multiple of number of tones processed by the thread block");

    gridDim  = dim3(N_THRD_BLK_PER_PRB, nPrb, nUeGrps);
    blockDim = dim3(N_THRDS_PER_TONE, N_TONES_PER_ITER);

#ifdef ENABLE_DEBUG
    NVLOGI_FMT(NVLOG_PUSCH, "{}: blockDim ({},{},{}), gridDim ({},{},{})", __FUNCTION__, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
#endif
}


#ifndef FAST_COMPILE
template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS, // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS>  // # of layers (# of cols in H matrix)
void
puschRxChEq::eqMmseCoefCompMassiveMimo(uint16_t                     nPrb,
                                       uint16_t                     nUeGrps,
                                       cuphyPuschRxChEqLaunchCfg_t& launchCfg)
{
    constexpr uint32_t N_THRD_BLK_TONES = 1; // CUPHY_N_TONES_PER_PRB;
    constexpr uint32_t N_TONES_PER_ITER = 1; // 12;

    static_assert(0 == (CUPHY_N_TONES_PER_PRB % N_THRD_BLK_TONES), "Number of tones processed per thread block must be a multiple of PRB size");
    static_assert(0 == (N_THRD_BLK_TONES % N_TONES_PER_ITER), "Number of tones processed per iteration must be a multiple of number of tones processed by the thread block");

    void* kernelFunc = reinterpret_cast<void*>(eqMmseCoefCompMassiveMimoKernel_v1<TStorageIn,
                                                                                  TStorageOut,
                                                                                  TCompute,
                                                                                  N_BS_ANTS,
                                                                                  N_LAYERS,
                                                                                  N_THRD_BLK_TONES,
                                                                                  N_TONES_PER_ITER>);

    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = launchCfg.kernelNodeParamsDriver;
    CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));

    dim3 blockDim, gridDim;
    coefCompMassiveMimoKernelLaunchGeo<N_LAYERS,
                                       N_THRD_BLK_TONES,
                                       N_TONES_PER_ITER>(nPrb, nUeGrps, gridDim, blockDim);

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;

    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;
    
    kernelNodeParamsDriver.extra          = nullptr;

    kernelNodeParamsDriver.sharedMemBytes = 0;
}

template <uint32_t N_THRD_BLK_PER_PRB, // # of thread blocks needed to process 1 PRB (12 subcarriers)
          uint32_t N_TONES_PER_ITER,   // # of frequency bins processed in 1 iteration by the thread block
          uint32_t N_THRDS_PER_TONE>   // # of threads needed to process 1 tone (1 subcarrier)
void
puschRxChEq::coefCompHighMimoKernelLaunchGeo(uint16_t nPrb,
                                             uint16_t nUeGrps,
                                             dim3&    gridDim,
                                             dim3&    blockDim)
{
    gridDim  = dim3(N_THRD_BLK_PER_PRB, nPrb, nUeGrps);
    blockDim = dim3(N_THRDS_PER_TONE, N_TONES_PER_ITER);

#ifdef ENABLE_DEBUG
    NVLOGI_FMT(NVLOG_PUSCH, "{}: blockDim ({},{},{}), gridDim ({},{},{})", __FUNCTION__, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
#endif
}

template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS, // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS>  // # of layers (# of cols in H matrix)
void
puschRxChEq::eqMmseCoefCompHighMimo(uint16_t                     nPrb,
                                    uint16_t                     nUeGrps,
                                    cuphyPuschRxChEqLaunchCfg_t& launchCfg)
{
#if(EQ_COEF_COMP_H_MIMO_VER == 1)
    constexpr uint32_t N_THRD_BLK_TONES   = 6; // CUPHY_N_TONES_PER_PRB;
    constexpr uint32_t N_TONES_PER_ITER   = 2; // 12;
    constexpr uint32_t N_THRD_BLK_PER_PRB = (CUPHY_N_TONES_PER_PRB / N_THRD_BLK_TONES);
    // (N_BS_ANTS+N_LAYERS) > (N_BS_ANTS*N_LAYERS) if N_LAYERS = 1
    const uint32_t N_THRDS_PER_TONE = std::max(N_BS_ANTS * N_LAYERS, N_BS_ANTS + N_LAYERS);
#elif(EQ_COEF_COMP_H_MIMO_VER == 2)
    constexpr uint32_t N_THRD_BLK_TONES   = 6; // 3;// 6;
    constexpr uint32_t N_TONES_PER_ITER   = 2; // 1;// 2;
    constexpr uint32_t N_THRD_BLK_PER_PRB = (CUPHY_N_TONES_PER_PRB / N_THRD_BLK_TONES);
    // const uint32_t N_THRDS_PER_TONE = round_up_to_next<uint32_t>(N_BS_ANTS + (2 * N_LAYERS), N_THREADS_PER_WARP);
    const uint32_t N_THRDS_PER_TONE = N_BS_ANTS + (2 * N_LAYERS); // width of the augmented matrix to be factorized
#endif
    static_assert(0 == (CUPHY_N_TONES_PER_PRB % N_THRD_BLK_TONES), "Number of tones processed per thread block must be a multiple of PRB size");
    static_assert(0 == (N_THRD_BLK_TONES % N_TONES_PER_ITER), "Number of tones processed per iteration must be a multiple of number of tones processed by the thread block");

    void* kernelFunc = reinterpret_cast<void*>(eqMmseCoefCompHighMimoKernel<TStorageIn,
                                                                            TStorageOut,
                                                                            TCompute,
                                                                            N_BS_ANTS,
                                                                            N_LAYERS,
                                                                            N_THRD_BLK_TONES,
                                                                            N_TONES_PER_ITER>);

    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = launchCfg.kernelNodeParamsDriver;
    CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));

    dim3 blockDim, gridDim;
    coefCompHighMimoKernelLaunchGeo<N_THRD_BLK_PER_PRB,
                                    N_TONES_PER_ITER,
                                    N_THRDS_PER_TONE>(nPrb, nUeGrps, gridDim, blockDim);

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;
    
    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;

    kernelNodeParamsDriver.extra          = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = 0;
}
#endif

template <uint32_t N_BS_ANTS, // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS,  // # of layers (# of cols in H matrix)
          uint32_t N_FREQ_BINS_PER_ITER>
void puschRxChEq::coefCompLowMimoKernelLaunchGeo(uint16_t nPrb,
                                                 uint16_t nUeGrps,
                                                 dim3&    gridDim,
                                                 dim3&    blockDim)
{
    // (N_BS_ANTS+N_LAYERS) > (N_BS_ANTS*N_LAYERS) if N_LAYERS = 1
    const uint32_t N_THREADS_PER_FREQ_BIN = std::max(N_BS_ANTS * N_LAYERS, N_BS_ANTS + N_LAYERS);

    gridDim  = dim3(nPrb, nUeGrps);
    blockDim = dim3(N_THREADS_PER_FREQ_BIN, N_FREQ_BINS_PER_ITER);

#ifdef ENABLE_DEBUG
    NVLOGI_FMT(NVLOG_PUSCH, "{}: blockDim ({},{},{}), gridDim ({},{},{})", __FUNCTION__, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
#endif
}

template <typename TStorageIn,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS, // # of BS antenna (# of rows in H matrix)
          uint32_t N_LAYERS>  // # of layers (# of cols in H matrix)
void
puschRxChEq::eqMmseCoefCompLowMimo(uint16_t                     nPrb,
                                   uint16_t                     nUeGrps,
                                   cuphyPuschRxChEqLaunchCfg_t& launchCfg)
{
    constexpr uint32_t N_FREQ_BINS_PER_ITER = 4; // 12;

    void* kernelFunc = reinterpret_cast<void*>(eqMmseCoefCompLowMimoKernel<TStorageIn,
                                                                           TStorageOut,
                                                                           TCompute,
                                                                           N_BS_ANTS,
                                                                           N_LAYERS,
                                                                           N_FREQ_BINS_PER_ITER>);
    
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = launchCfg.kernelNodeParamsDriver;
    CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));

    dim3 blockDim, gridDim;
    coefCompLowMimoKernelLaunchGeo<N_BS_ANTS,
                                   N_LAYERS,
                                   N_FREQ_BINS_PER_ITER>(nPrb, nUeGrps, gridDim, blockDim);

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;
    
    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;

    kernelNodeParamsDriver.extra          = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = 0;
}

template <uint32_t N_BS_ANTS, // # of BS antenna (# of cols of C matrix)
          uint32_t N_SYMBS_PER_THRD_BLK>
void puschRxChEq::softDemapKernelLaunchGeo(uint8_t  Nd,
                                           uint16_t nPrb,
                                           uint16_t nUeGrps,
                                           dim3&    gridDim,
                                           dim3&    blockDim)
{
    // Number of frequency bins processed by a thread block
    constexpr uint32_t N_FREQ_BINS_PER_THRD_BLK = CUPHY_N_TONES_PER_PRB;
    // Ensure max(N_BS_ANTS, N_LAYERS) threads exist per PRB tone
    constexpr uint32_t N_XTHRDS_PER_THRD_BLK = N_BS_ANTS * N_FREQ_BINS_PER_THRD_BLK;
    // Ensure total number of threads remains less than/equal to 1024
    static_assert((N_SYMBS_PER_THRD_BLK * N_XTHRDS_PER_THRD_BLK) <= 1024,"invalid launch configuration for soft demapper kernel");
    gridDim  = dim3(nPrb, (Nd + N_SYMBS_PER_THRD_BLK - 1) / N_SYMBS_PER_THRD_BLK, nUeGrps);
    blockDim = dim3(N_XTHRDS_PER_THRD_BLK, N_SYMBS_PER_THRD_BLK);

#ifdef ENABLE_DEBUG
    NVLOGI_FMT(NVLOG_PUSCH, "{}: blockDim ({},{},{}), gridDim ({},{},{})", __FUNCTION__, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
#endif
}


template <uint32_t N_SYMBS_PER_THRD_BLK>
void puschRxChEq::softDemapAfterDftKernelLaunchGeo(uint8_t  Nd,
                                           uint16_t nPrb,
                                           uint16_t nUeGrps,
                                           dim3&    gridDim,
                                           dim3&    blockDim)
{
    // Number of frequency bins processed by a thread block
    constexpr uint32_t N_FREQ_BINS_PER_THRD_BLK = CUPHY_N_TONES_PER_PRB;
    // Ensure max(N_BS_ANTS, N_LAYERS) threads exist per PRB tone
    constexpr uint32_t N_XTHRDS_PER_THRD_BLK = N_FREQ_BINS_PER_THRD_BLK;
    // Ensure total number of threads remains less than/equal to 1024
    static_assert((N_SYMBS_PER_THRD_BLK * N_XTHRDS_PER_THRD_BLK) <= 1024,"invalid launch configuration for soft demapper kernel");

    gridDim  = dim3(nPrb, (Nd + N_SYMBS_PER_THRD_BLK - 1) / N_SYMBS_PER_THRD_BLK, nUeGrps);
    blockDim = dim3(N_XTHRDS_PER_THRD_BLK, N_SYMBS_PER_THRD_BLK);

#ifdef ENABLE_DEBUG
    NVLOGI_FMT(NVLOG_PUSCH, "{}: blockDim ({},{},{}), gridDim ({},{},{})", __FUNCTION__, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
#endif
} // puschRxChEq::softDemapAfterDftKernelLaunchGeo
#ifndef FAST_COMPILE
template <uint32_t N_SYMBS_PER_THRD_BLK>
void puschRxChEq::softDemapKernelLaunchGeo_64R(uint8_t  Nd,
                                           uint16_t nPrb,
                                           uint16_t nUeGrps,
                                           dim3&    gridDim,
                                           dim3&    blockDim)
{
    constexpr uint32_t N_BS_ANTS = 64;
    // Number of frequency bins processed by a thread block
    constexpr uint32_t N_FREQ_BINS_PER_THRD_BLK = CUPHY_N_TONES_PER_PRB;
    // Ensure max(N_BS_ANTS, N_LAYERS) threads exist per PRB tone
    constexpr uint32_t N_XTHRDS_PER_THRD_BLK = (N_BS_ANTS / 2) * N_FREQ_BINS_PER_THRD_BLK;
    // Ensure total number of threads remains less than/equal to 1024
    static_assert((N_SYMBS_PER_THRD_BLK * N_XTHRDS_PER_THRD_BLK) <= 1024,"invalid launch configuration for soft demapper kernel");

    gridDim  = dim3(nPrb, (Nd + N_SYMBS_PER_THRD_BLK - 1) / N_SYMBS_PER_THRD_BLK, nUeGrps);
    blockDim = dim3(N_XTHRDS_PER_THRD_BLK, N_SYMBS_PER_THRD_BLK);

#ifdef ENABLE_DEBUG
    NVLOGI_FMT(NVLOG_PUSCH, "{}: blockDim ({},{},{}), gridDim ({},{},{})", __FUNCTION__, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
#endif
}
#endif

template <typename TStorageIn,
          typename TDataRx,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS, // # of BS antenna (# of cols of C matrix)
          uint32_t N_SYMBS_PER_THRD_BLK> // # of data symbols processed by a thread block
void
puschRxChEq::eqMmseSoftDemap(uint8_t                      Nd,
                             uint16_t                     nPrb,
                             uint16_t                     nUeGrps,
                             uint16_t                     symbolBitmask,
                             cuphyPuschRxChEqLaunchCfg_t& launchCfg)
{
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = launchCfg.kernelNodeParamsDriver;
    
    if(symbolBitmask==CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK)
    {
        static constexpr uint16_t SYMBOL_BITMASK    = CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK;
        void* kernelFunc = reinterpret_cast<void*>(eqMmseSoftDemapKernel<TStorageIn,
                                                                         TDataRx,
                                                                         TStorageOut,
                                                                         TCompute,
                                                                         N_BS_ANTS,
                                                                         N_SYMBS_PER_THRD_BLK,
                                                                         SYMBOL_BITMASK>); //full-slot processing

        CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));
    }
    else if(symbolBitmask==CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK)
    {
        static constexpr uint16_t SYMBOL_BITMASK    = CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK;
        void* kernelFunc = reinterpret_cast<void*>(eqMmseSoftDemapKernel<TStorageIn,
                                                                         TDataRx,
                                                                         TStorageOut,
                                                                         TCompute,
                                                                         N_BS_ANTS,
                                                                         N_SYMBS_PER_THRD_BLK,
                                                                         SYMBOL_BITMASK>); //early-HARQ processing

        CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));
    }

    dim3 blockDim, gridDim;
    softDemapKernelLaunchGeo<N_BS_ANTS,
                             N_SYMBS_PER_THRD_BLK>(Nd, nPrb, nUeGrps, gridDim, blockDim);

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;
    
    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;

    kernelNodeParamsDriver.extra          = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = 0;
} //eqMmseSoftDemap


template <typename TStorageIn,
          typename TDataRx,
          typename TStorageOut,
          typename TCompute>
void puschRxChEq::eqMmseSoftDemapBluesteinWorkspace(uint8_t            Nd,
                             uint16_t                     nPrb,
                             uint16_t                     nUeGrps,
                             cuphyPuschRxChEqLaunchCfg_t& launchCfg)
{
    using namespace cufftdx;
    dim3 grid_dim = dim3(1, 1, nUeGrps);  //bluestein workspace

    using FFT = decltype(Size<FFT8192>() + Precision<float>() + Type<fft_type::c2c>()
                    + Direction<fft_direction::inverse>() + FFTsPerBlock<1>()
                    + ElementsPerThread<16>() + SM<800>() + Block());
                              
    dim3 block_dim = FFT::block_dim;

    uint shared_memory_size = FFT::shared_memory_size;

#ifdef ENABLE_DEBUG
    NVLOGI_FMT(NVLOG_PUSCH, "{}: workspace[{}]FFT::max_threads_per_block[{}][{} {} {}]FFT::storage_size[{}]FFT::elements_per_thread[{}]grid_dim[{} {} {}]", __FUNCTION__, (int)FFT::requires_workspace, FFT::max_threads_per_block, block_dim.x, block_dim.y, block_dim.z, FFT::storage_size, FFT::elements_per_thread, grid_dim.x, grid_dim.y, grid_dim.z);
#endif

    void* kernelPtr = reinterpret_cast<void*>(bluestein_workspace_kernel<FFT, TStorageIn, TDataRx, TStorageOut, TCompute>);
  
    CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriver.func, kernelPtr));
    launchCfg.kernelNodeParamsDriver.blockDimX = block_dim.x;
    launchCfg.kernelNodeParamsDriver.blockDimY = block_dim.y;
    launchCfg.kernelNodeParamsDriver.blockDimZ = block_dim.z;
    launchCfg.kernelNodeParamsDriver.gridDimX  = grid_dim.x;
    launchCfg.kernelNodeParamsDriver.gridDimY  = grid_dim.y;
    launchCfg.kernelNodeParamsDriver.gridDimZ  = grid_dim.z;
    launchCfg.kernelNodeParamsDriver.sharedMemBytes = shared_memory_size;
    launchCfg.kernelNodeParamsDriver.extra     = NULL;
} //puschRxChEq::eqMmseSoftDemapBluesteinWorkspace

template <typename TStorageIn,
          typename TDataRx,
          typename TStorageOut,
          typename TCompute>
void puschRxChEq::eqMmseSoftDemapIdft(uint8_t                      Nd,
                                       uint16_t                     nPrb,
                                       uint16_t                     nUeGrps,
                                       uint16_t                     symbolBitmask, 
                                       cuphyPuschRxChEqLaunchCfg_t& launchCfg)
{

    using namespace cufftdx;
    dim3 grid_dim = dim3(Nd, 1, nUeGrps); //Idft

    using FFT = decltype(Size<FFT8192>() + Precision<float>() + Type<fft_type::c2c>()
                    + Direction<fft_direction::inverse>() + FFTsPerBlock<1>()
                    + ElementsPerThread<16>() + SM<800>() + Block());
                              
    dim3 block_dim = FFT::block_dim;

    uint shared_memory_size = FFT::shared_memory_size;

#ifdef ENABLE_DEBUG
    NVLOGI_FMT(NVLOG_PUSCH, "{}: workspace[{}]FFT::max_threads_per_block[{}][{} {} {}]FFT::storage_size[{}]FFT::elements_per_thread[{}]grid_dim[{} {} {}]", __FUNCTION__, (int)FFT::requires_workspace, FFT::max_threads_per_block, block_dim.x, block_dim.y, block_dim.z, FFT::storage_size, FFT::elements_per_thread, grid_dim.x, grid_dim.y, grid_dim.z);
#endif
  
    if(symbolBitmask==CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK)
    {   
        static constexpr uint16_t SYMBOL_BITMASK    = CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK;
        void* kernelPtr = reinterpret_cast<void*>(bluestein_Idft_kernel<FFT, TStorageIn, TDataRx, TStorageOut, TCompute, SYMBOL_BITMASK>);
        CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriver.func, kernelPtr));
    }
    else if(symbolBitmask==CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK)
    {   
        static constexpr uint16_t SYMBOL_BITMASK    = CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK;
        void* kernelPtr = reinterpret_cast<void*>(bluestein_Idft_kernel<FFT, TStorageIn, TDataRx, TStorageOut, TCompute, SYMBOL_BITMASK>);
        CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriver.func, kernelPtr));
    }
    launchCfg.kernelNodeParamsDriver.blockDimX = block_dim.x;
    launchCfg.kernelNodeParamsDriver.blockDimY = block_dim.y;
    launchCfg.kernelNodeParamsDriver.blockDimZ = block_dim.z;
    launchCfg.kernelNodeParamsDriver.gridDimX  = grid_dim.x;
    launchCfg.kernelNodeParamsDriver.gridDimY  = grid_dim.y;
    launchCfg.kernelNodeParamsDriver.gridDimZ  = grid_dim.z;
    launchCfg.kernelNodeParamsDriver.sharedMemBytes = shared_memory_size;
    launchCfg.kernelNodeParamsDriver.extra     = NULL;
} //puschRxChEq::eqMmseSoftDemapIdft

template <typename TStorageIn,
          typename TDataRx,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_BS_ANTS, // # of BS antenna (# of cols of C matrix)
          uint32_t N_SYMBS_PER_THRD_BLK> // # of data symbols processed by a thread block
void puschRxChEq::eqMmseSoftDemapAfterDft(uint8_t                      Nd,
                                          uint16_t                     nPrb,
                                          uint16_t                     nUeGrps,
                                          uint16_t                     symbolBitmask,
                                          cuphyPuschRxChEqLaunchCfg_t& launchCfg)
{
    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = launchCfg.kernelNodeParamsDriver;
    if(symbolBitmask==CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK)
    {
        static constexpr uint16_t SYMBOL_BITMASK    = CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK;
        void* kernelFunc = reinterpret_cast<void*>(eqMmseSoftDemapAfterDftKernel<TStorageIn,
                                                                                 TDataRx,
                                                                                 TStorageOut,
                                                                                 TCompute,
                                                                                 N_BS_ANTS,
                                                                                 N_SYMBS_PER_THRD_BLK,
                                                                                 SYMBOL_BITMASK>);
        CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));
    }
    else if(symbolBitmask==CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK)
    {
        static constexpr uint16_t SYMBOL_BITMASK    = CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK;
        void* kernelFunc = reinterpret_cast<void*>(eqMmseSoftDemapAfterDftKernel<TStorageIn,
                                                                                 TDataRx,
                                                                                 TStorageOut,
                                                                                 TCompute,
                                                                                 N_BS_ANTS,
                                                                                 N_SYMBS_PER_THRD_BLK,
                                                                                 SYMBOL_BITMASK>);
        CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));
    }

    dim3 blockDim, gridDim;
    softDemapAfterDftKernelLaunchGeo<N_SYMBS_PER_THRD_BLK>(Nd, nPrb, nUeGrps, gridDim, blockDim);

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;
    
    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;

    kernelNodeParamsDriver.extra          = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = 0;
} //puschRxChEq::eqMmseSoftDemapAfterDft

#ifndef FAST_COMPILE
template <typename TStorageIn,
          typename TDataRx,
          typename TStorageOut,
          typename TCompute,
          uint32_t N_LAYERS,  // # of layers (# of rows of C matrix)
          uint32_t N_SYMBS_PER_THRD_BLK> // # of data symbols processed by a thread block
void
puschRxChEq::eqMmseSoftDemap_64R(uint8_t                      Nd,
                                uint16_t                     nPrb,
                                uint16_t                     nUeGrps,
                                cuphyPuschRxChEqLaunchCfg_t& launchCfg)
{
    // constexpr N_BS_ANTS = 64;
    void* kernelFunc = reinterpret_cast<void*>(eqMmseSoftDemapKernel_64R<TStorageIn,
                                                                        TDataRx,
                                                                        TStorageOut,
                                                                        TCompute,
                                                                        N_LAYERS,
                                                                        N_SYMBS_PER_THRD_BLK>);

    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = launchCfg.kernelNodeParamsDriver;
    CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));

    dim3 blockDim, gridDim;
    softDemapKernelLaunchGeo_64R<N_SYMBS_PER_THRD_BLK>(Nd, nPrb, nUeGrps, gridDim, blockDim);

    kernelNodeParamsDriver.blockDimX = blockDim.x;
    kernelNodeParamsDriver.blockDimY = blockDim.y;
    kernelNodeParamsDriver.blockDimZ = blockDim.z;
    
    kernelNodeParamsDriver.gridDimX = gridDim.x;
    kernelNodeParamsDriver.gridDimY = gridDim.y;
    kernelNodeParamsDriver.gridDimZ = gridDim.z;

    kernelNodeParamsDriver.extra          = nullptr;
    kernelNodeParamsDriver.sharedMemBytes = 0;
}
#endif

template <typename TStorageIn, typename TStorageOut, typename TCompute>
void puschRxChEq::coefCompKernelSelectL0(uint16_t                     nBSAnts,
                                         uint8_t                      nLayers,
                                         uint8_t                      nTimeChEst,
                                         uint16_t                     nPrb,
                                         uint16_t                     nUeGrps,
                                         cuphyPuschRxChEqLaunchCfg_t& launchCfg)
{
    // Low MIMO regime
  if(((8 == nBSAnts) || (4 == nBSAnts) || (2 == nBSAnts) || (1 == nBSAnts) ))
    {
        switch(nBSAnts)
        {
        // nBSAnts == 8
        case 8: {
            constexpr uint32_t N_BS_ANTS = 8; // # of BS antenna (# of rows in H matrix)
            switch(nLayers)
            {
            // nLayers == 8
            case 8: {
                constexpr uint32_t N_LAYERS = 8; // # of layers (# of cols in H matrix)
		eqMmseCoefCompLowMimo<TStorageIn,
				      TStorageOut,
				      TCompute,
				      N_BS_ANTS,
				      N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 7
            case 7: {
                constexpr uint32_t N_LAYERS = 7; // # of layers (# of cols in H matrix)
		eqMmseCoefCompLowMimo<TStorageIn,
				      TStorageOut,
				      TCompute,
				      N_BS_ANTS,
				      N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 6
            case 6: {
                constexpr uint32_t N_LAYERS = 6; // # of layers (# of cols in H matrix)
		eqMmseCoefCompLowMimo<TStorageIn,
				      TStorageOut,
				      TCompute,
				      N_BS_ANTS,
				      N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 5
            case 5: {
                constexpr uint32_t N_LAYERS = 5; // # of layers (# of cols in H matrix)
		eqMmseCoefCompLowMimo<TStorageIn,
				      TStorageOut,
				      TCompute,
				      N_BS_ANTS,
				      N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 4
            case 4: {
                constexpr uint32_t N_LAYERS = 4; // # of layers (# of cols in H matrix)
		eqMmseCoefCompLowMimo<TStorageIn,
				      TStorageOut,
				      TCompute,
				      N_BS_ANTS,
				      N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 3
            case 3: {
                constexpr uint32_t N_LAYERS = 3; // # of layers (# of cols in H matrix)
		eqMmseCoefCompLowMimo<TStorageIn,
				      TStorageOut,
				      TCompute,
				      N_BS_ANTS,
				      N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 2
            case 2: {
                constexpr uint32_t N_LAYERS = 2; // # of layers (# of cols in H matrix)
		eqMmseCoefCompLowMimo<TStorageIn,
				      TStorageOut,
				      TCompute,
				      N_BS_ANTS,
				      N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }
	   // nLayers == 1
            case 1: {
                constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
		eqMmseCoefCompLowMimo<TStorageIn,
				      TStorageOut,
				      TCompute,
				      N_BS_ANTS,
				      N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }
            default: {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {} nTimeChEst {}", __FUNCTION__, nBSAnts, nLayers, nTimeChEst);
                break;
            }
            }

            break;
        }

        // nBSAnts == 4
        case 4: {
            constexpr uint32_t N_BS_ANTS = 4; // # of BS antenna (# of rows in H matrix)

            switch(nLayers)
            {
            // nLayers == 4
            case 4: {
                constexpr uint32_t N_LAYERS = 4; // # of layers (# of cols in H matrix)
		eqMmseCoefCompLowMimo<TStorageIn,
				      TStorageOut,
				      TCompute,
				      N_BS_ANTS,
				      N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }

            // nLayers == 3
            case 3: {
                constexpr uint32_t N_LAYERS = 3; // # of layers (# of cols in H matrix)
		eqMmseCoefCompLowMimo<TStorageIn,
				      TStorageOut,
				      TCompute,
				      N_BS_ANTS,
				      N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }

            // nLayers == 2
            case 2: {
                constexpr uint32_t N_LAYERS = 2; // # of layers (# of cols in H matrix)
		eqMmseCoefCompLowMimo<TStorageIn,
				      TStorageOut,
				      TCompute,
				      N_BS_ANTS,
				      N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }

            // nLayers == 1
            case 1: {
                constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
		eqMmseCoefCompLowMimo<TStorageIn,
				      TStorageOut,
				      TCompute,
				      N_BS_ANTS,
				      N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }
            default: {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {} nTimeChEst {}", __FUNCTION__, nBSAnts, nLayers, nTimeChEst);
                break;
            }
            }

            break;
        }

        // nBSAnts == 2
        case 2: {
            constexpr uint32_t N_BS_ANTS = 2; // # of BS antenna (# of rows in H matrix)
            switch(nLayers)
            {
            // nLayers == 2
            case 2: {
	      constexpr uint32_t N_LAYERS = 2; // # of layers (# of cols in H matrix)
	      eqMmseCoefCompLowMimo<TStorageIn,
				    TStorageOut,
				    TCompute,
				    N_BS_ANTS,
				    N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }

            // nLayers == 1
            case 1: {
                constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
		eqMmseCoefCompLowMimo<TStorageIn,
				      TStorageOut,
				      TCompute,
				      N_BS_ANTS,
				      N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }

            default: {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {} nTimeChEst {}", __FUNCTION__, nBSAnts, nLayers, nTimeChEst);
                break;
            }
            }
            break;
        }

	// nBSAnts == 1
	case 1: {
	  constexpr uint32_t N_BS_ANTS = 1; // # of BS antenna (# of rows in H matrix)
	  switch(nLayers)
            {

	      // nLayers == 1
              case 1: {
		constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
		eqMmseCoefCompLowMimo<TStorageIn,
				      TStorageOut,
				      TCompute,
				      N_BS_ANTS,
				      N_LAYERS>(nPrb, nUeGrps, launchCfg);
		break;
	      }

              default: {
		NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {} nTimeChEst {}", __FUNCTION__, nBSAnts, nLayers, nTimeChEst);
		break;
	      }
            }
	  break;
	}

        default: {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {} nTimeChEst {}", __FUNCTION__, nBSAnts, nLayers, nTimeChEst);
            break;
        }
        }
    }
#ifndef FAST_COMPILE
    // High MIMO regime
    else if((16 == nBSAnts))
    {
        switch(nBSAnts)
        {
        // nBSAnts == 16
        case 16: {
            constexpr uint32_t N_BS_ANTS = 16; // # of BS antenna (# of rows in H matrix)
            switch(nLayers)
            {
            // nLayers == 1
            case 1: {
                constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
		eqMmseCoefCompHighMimo<TStorageIn,
				       TStorageOut,
				       TCompute,
				       N_BS_ANTS,
				       N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 2
            case 2: {
                constexpr uint32_t N_LAYERS = 2; // # of layers (# of cols in H matrix)
		eqMmseCoefCompHighMimo<TStorageIn,
				       TStorageOut,
				       TCompute,
				       N_BS_ANTS,
				       N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 3
            case 3: {
	      constexpr uint32_t N_LAYERS = 3; // # of layers (# of cols in H matrix)
	      eqMmseCoefCompHighMimo<TStorageIn,
				     TStorageOut,
				     TCompute,
				     N_BS_ANTS,
				     N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 4
            case 4: {
                constexpr uint32_t N_LAYERS = 4; // # of layers (# of cols in H matrix)
		eqMmseCoefCompHighMimo<TStorageIn,
				       TStorageOut,
				       TCompute,
				       N_BS_ANTS,
				       N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 5
            case 5: {
                constexpr uint32_t N_LAYERS = 5; // # of layers (# of cols in H matrix)
		eqMmseCoefCompHighMimo<TStorageIn,
				       TStorageOut,
				       TCompute,
				       N_BS_ANTS,
				       N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 6
            case 6: {
                constexpr uint32_t N_LAYERS = 6; // # of layers (# of cols in H matrix)
		eqMmseCoefCompHighMimo<TStorageIn,
				       TStorageOut,
				       TCompute,
				       N_BS_ANTS,
				       N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 7
            case 7: {
                constexpr uint32_t N_LAYERS = 7; // # of layers (# of cols in H matrix)
		eqMmseCoefCompHighMimo<TStorageIn,
				       TStorageOut,
				       TCompute,
				       N_BS_ANTS,
				       N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 8
            case 8: {
                constexpr uint32_t N_LAYERS = 8; // # of layers (# of cols in H matrix)
		eqMmseCoefCompHighMimo<TStorageIn,
				       TStorageOut,
				       TCompute,
				       N_BS_ANTS,
				       N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }
            // nLayers == 16
            case 16: {
                constexpr uint32_t N_LAYERS = 16; // # of layers (# of cols in H matrix)
		eqMmseCoefCompHighMimo<TStorageIn,
				       TStorageOut,
				       TCompute,
				       N_BS_ANTS,
				       N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }
            default: {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {} nTimeChEst {}", __FUNCTION__, nBSAnts, nLayers, nTimeChEst);
                break;
            }
            }

            break;
        }

        default: {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {} nTimeChEst {}", __FUNCTION__, nBSAnts, nLayers, nTimeChEst);
            break;
        }
        }
    }
    // Massive MIMO regime
    else if((64 == nBSAnts))
    {
        switch(nBSAnts)
        {
        // nBSAnts == 64
        case 64: {
            constexpr uint32_t N_BS_ANTS = 64; // # of BS antenna (# of rows in H matrix)
            switch(nLayers)
            {
            // nLayers == 16
            case 16: {
                constexpr uint32_t N_LAYERS = 16; // # of layers (# of cols in H matrix)
		eqMmseCoefCompMassiveMimo<TStorageIn,
					  TStorageOut,
					  TCompute,
					  N_BS_ANTS,
					  N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }
            // 64R: add nLayers==8 and use highmimo, which out-performs massive mimo
            case 8: {
                using TComputeHalf          = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
                constexpr uint32_t N_LAYERS = 8; // # of layers (# of cols in H matrix)
		eqMmseCoefCompMassiveMimo<TStorageIn,
					  TStorageOut,
					  TComputeHalf,
					  N_BS_ANTS,
					  N_LAYERS>(nPrb, nUeGrps, launchCfg);
                break;
            }
            default: {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {} nTimeChEst {}", __FUNCTION__, nBSAnts, nLayers, nTimeChEst);
                break;
            }
            }

            break;
        }
        default: {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {} nTimeChEst {}", __FUNCTION__, nBSAnts, nLayers, nTimeChEst);
            break;
        }
        }
    }
#endif
    else
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {} nTimeChEst {}", __FUNCTION__, nBSAnts, nLayers, nTimeChEst);
    }
}

template <typename TStorageIn, typename TDataRx, typename TStorageOut, typename TCompute>
void puschRxChEq::softDemapKernelSelectL0(uint16_t                     nBSAnts,
                                          uint8_t                      nLayers,
                                          uint8_t                      nTimeChEsts,
                                          uint8_t                      Nd,
                                          uint16_t                     nPrb,
                                          uint16_t                     nUeGrps,
                                          uint16_t                     symbolBitmask,
                                          cuphyPuschRxChEqLaunchCfg_t& launchCfg)
{
    switch(nBSAnts)
    {
    case 64: {
        constexpr uint32_t N_LAYERS             = 8; // # of layers (# of cols in H matrix)
#ifndef ENABLE_MULTI_SYMBS_PER_THRD_BLK
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = 1; // # of data symbols processed by a thread block
#else
        constexpr uint32_t MAX_THRD_BLK_DIM_Y   = CUDA_MAX_N_THRDS_PER_BLK / (32 * CUPHY_N_TONES_PER_PRB);
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = OFDM_SYMBOLS_PER_SLOT < MAX_THRD_BLK_DIM_Y ? OFDM_SYMBOLS_PER_SLOT : MAX_THRD_BLK_DIM_Y;
#endif
        eqMmseSoftDemap_64R<TStorageIn,
                            TDataRx,
                            TStorageOut,
                            TCompute,
                            N_LAYERS,
                            N_SYMBS_PER_THRD_BLK>(Nd, nPrb, nUeGrps, launchCfg);
        break;
    }

    // nBSAnts == 16
    case 16: {
        constexpr uint32_t N_BS_ANTS = 16; // # of BS antenna (# of rows in H matrix)

#if ENABLE_MULTI_SYMBS_PER_THRD_BLK == 0   // single symbol per slot
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = 1;
#elif ENABLE_MULTI_SYMBS_PER_THRD_BLK == 1 // NUM_SYMBOLS_PER_SLOT per slot
        constexpr uint32_t MAX_THRD_BLK_DIM_Y   = CUDA_MAX_N_THRDS_PER_BLK / (N_BS_ANTS * CUPHY_N_TONES_PER_PRB);
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = NUM_SYMBOLS_PER_SLOT < MAX_THRD_BLK_DIM_Y ? NUM_SYMBOLS_PER_SLOT : MAX_THRD_BLK_DIM_Y;
#elif ENABLE_MULTI_SYMBS_PER_THRD_BLK == 2 // auto-select num symbols per slot
        constexpr uint32_t N_XTHRDS_PER_THRD_BLK = N_BS_ANTS * CUPHY_N_TONES_PER_PRB;
        // let's pick N_SYMBS_PER_THRD_BLK such that block size is multiple of warp size (but no more than OFDM_SYMBOLS_PER_SLOT)
        constexpr uint32_t MAX_THRD_BLK_DIM_Y    = std::min(CUDA_MAX_N_THRDS_PER_BLK, compute_lcm(N_XTHRDS_PER_THRD_BLK, N_THREADS_PER_WARP)) / N_XTHRDS_PER_THRD_BLK;
        constexpr uint32_t N_SYMBS_PER_THRD_BLK  = OFDM_SYMBOLS_PER_SLOT < MAX_THRD_BLK_DIM_Y ? OFDM_SYMBOLS_PER_SLOT : MAX_THRD_BLK_DIM_Y;
#endif

        eqMmseSoftDemap<TStorageIn,
                        TDataRx,
                        TStorageOut,
                        TCompute,
                        N_BS_ANTS,
                        N_SYMBS_PER_THRD_BLK>(Nd, nPrb, nUeGrps, symbolBitmask, launchCfg);
        break;
    }

    // nBSAnts == 8
    case 8: {
        constexpr uint32_t N_BS_ANTS = 8;           // # of BS antenna (# of rows in H matrix)
    #if ENABLE_MULTI_SYMBS_PER_THRD_BLK == 0        // single symbol per slot
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = 1;
    #elif ENABLE_MULTI_SYMBS_PER_THRD_BLK == 1      // NUM_SYMBOLS_PER_SLOT per slot
        constexpr uint32_t MAX_THRD_BLK_DIM_Y   = CUDA_MAX_N_THRDS_PER_BLK / (N_BS_ANTS * CUPHY_N_TONES_PER_PRB);
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = NUM_SYMBOLS_PER_SLOT < MAX_THRD_BLK_DIM_Y ? NUM_SYMBOLS_PER_SLOT : MAX_THRD_BLK_DIM_Y;
    #elif ENABLE_MULTI_SYMBS_PER_THRD_BLK == 2      // auto-select num symbols per slot
        constexpr uint32_t N_XTHRDS_PER_THRD_BLK = N_BS_ANTS * CUPHY_N_TONES_PER_PRB;
        // let's pick N_SYMBS_PER_THRD_BLK such that block size is multiple of warp size (but no more than OFDM_SYMBOLS_PER_SLOT)
        constexpr uint32_t MAX_THRD_BLK_DIM_Y   = std::min(CUDA_MAX_N_THRDS_PER_BLK, compute_lcm(N_XTHRDS_PER_THRD_BLK, N_THREADS_PER_WARP)) / N_XTHRDS_PER_THRD_BLK;
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = OFDM_SYMBOLS_PER_SLOT < MAX_THRD_BLK_DIM_Y ? OFDM_SYMBOLS_PER_SLOT : MAX_THRD_BLK_DIM_Y;
    #endif
        eqMmseSoftDemap<TStorageIn,
                        TDataRx,
                        TStorageOut,
                        TCompute,
                        N_BS_ANTS,
                        N_SYMBS_PER_THRD_BLK>(Nd, nPrb, nUeGrps, symbolBitmask, launchCfg);
        break;
    }

    // nBsAnts == 4
    case 4: {
        constexpr uint32_t N_BS_ANTS = 4; // # of BS antenna (# of rows in H matrix)

#if ENABLE_MULTI_SYMBS_PER_THRD_BLK == 0        // single symbol per slot
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = 1;
#elif ENABLE_MULTI_SYMBS_PER_THRD_BLK == 1      // NUM_SYMBOLS_PER_SLOT per slot
        constexpr uint32_t MAX_THRD_BLK_DIM_Y   = CUDA_MAX_N_THRDS_PER_BLK / (N_BS_ANTS * CUPHY_N_TONES_PER_PRB);
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = NUM_SYMBOLS_PER_SLOT < MAX_THRD_BLK_DIM_Y ? NUM_SYMBOLS_PER_SLOT : MAX_THRD_BLK_DIM_Y;
#elif ENABLE_MULTI_SYMBS_PER_THRD_BLK == 2      // auto-select num symbols per slot
        constexpr uint32_t N_XTHRDS_PER_THRD_BLK = N_BS_ANTS * CUPHY_N_TONES_PER_PRB;
        // let's pick N_SYMBS_PER_THRD_BLK such that block size is multiple of warp size (but no more than OFDM_SYMBOLS_PER_SLOT)
        constexpr uint32_t MAX_THRD_BLK_DIM_Y   = std::min(CUDA_MAX_N_THRDS_PER_BLK, compute_lcm(N_XTHRDS_PER_THRD_BLK, N_THREADS_PER_WARP)) / N_XTHRDS_PER_THRD_BLK;
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = OFDM_SYMBOLS_PER_SLOT < MAX_THRD_BLK_DIM_Y ? OFDM_SYMBOLS_PER_SLOT : MAX_THRD_BLK_DIM_Y;
#endif

        eqMmseSoftDemap<TStorageIn,
                        TDataRx,
                        TStorageOut,
                        TCompute,
                        N_BS_ANTS,
                        N_SYMBS_PER_THRD_BLK>(Nd, nPrb, nUeGrps, symbolBitmask, launchCfg);
        break;
    }

    // nBsAnts == 2
    case 2: {
        constexpr uint32_t N_BS_ANTS = 2; // # of BS antenna (# of rows in H matrix)

#if ENABLE_MULTI_SYMBS_PER_THRD_BLK == 0        // single symbol per slot
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = 1;
#elif ENABLE_MULTI_SYMBS_PER_THRD_BLK == 1      // NUM_SYMBOLS_PER_SLOT per slot
        constexpr uint32_t MAX_THRD_BLK_DIM_Y   = CUDA_MAX_N_THRDS_PER_BLK / (N_BS_ANTS * CUPHY_N_TONES_PER_PRB);
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = NUM_SYMBOLS_PER_SLOT < MAX_THRD_BLK_DIM_Y ? NUM_SYMBOLS_PER_SLOT : MAX_THRD_BLK_DIM_Y;
#elif ENABLE_MULTI_SYMBS_PER_THRD_BLK == 2      // auto-select num symbols per slot
        constexpr uint32_t N_XTHRDS_PER_THRD_BLK = N_BS_ANTS * CUPHY_N_TONES_PER_PRB;
        // let's pick N_SYMBS_PER_THRD_BLK such that block size is multiple of warp size (but no more than OFDM_SYMBOLS_PER_SLOT)
        constexpr uint32_t MAX_THRD_BLK_DIM_Y   = std::min(CUDA_MAX_N_THRDS_PER_BLK, compute_lcm(N_XTHRDS_PER_THRD_BLK, N_THREADS_PER_WARP)) / N_XTHRDS_PER_THRD_BLK;
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = OFDM_SYMBOLS_PER_SLOT < MAX_THRD_BLK_DIM_Y ? OFDM_SYMBOLS_PER_SLOT : MAX_THRD_BLK_DIM_Y;
#endif

        eqMmseSoftDemap<TStorageIn,
                    TDataRx,
                    TStorageOut,
                    TCompute,
                    N_BS_ANTS,
                    N_SYMBS_PER_THRD_BLK>(Nd, nPrb, nUeGrps, symbolBitmask, launchCfg);
        break;
    }

    // nBsAnts == 1
    case 1: {
      constexpr uint32_t N_BS_ANTS = 1; // # of BS antenna (# of rows in H matrix)

#if ENABLE_MULTI_SYMBS_PER_THRD_BLK == 0        // single symbol per slot
      constexpr uint32_t N_SYMBS_PER_THRD_BLK = 1;
#elif ENABLE_MULTI_SYMBS_PER_THRD_BLK == 1      // NUM_SYMBOLS_PER_SLOT per slot
      constexpr uint32_t MAX_THRD_BLK_DIM_Y   = CUDA_MAX_N_THRDS_PER_BLK / (N_BS_ANTS * CUPHY_N_TONES_PER_PRB);
      constexpr uint32_t N_SYMBS_PER_THRD_BLK = NUM_SYMBOLS_PER_SLOT < MAX_THRD_BLK_DIM_Y ? NUM_SYMBOLS_PER_SLOT : MAX_THRD_BLK_DIM_Y;
#elif ENABLE_MULTI_SYMBS_PER_THRD_BLK == 2      // auto-select num symbols per slot
      constexpr uint32_t N_XTHRDS_PER_THRD_BLK = N_BS_ANTS * CUPHY_N_TONES_PER_PRB;
      // let's pick N_SYMBS_PER_THRD_BLK such that block size is multiple of warp size (but no more than OFDM_SYMBOLS_PER_SLOT)
      constexpr uint32_t MAX_THRD_BLK_DIM_Y   = std::min(CUDA_MAX_N_THRDS_PER_BLK, compute_lcm(N_XTHRDS_PER_THRD_BLK, N_THREADS_PER_WARP)) / N_XTHRDS_PER_THRD_BLK;
      constexpr uint32_t N_SYMBS_PER_THRD_BLK = OFDM_SYMBOLS_PER_SLOT < MAX_THRD_BLK_DIM_Y ? OFDM_SYMBOLS_PER_SLOT : MAX_THRD_BLK_DIM_Y;
#endif

      eqMmseSoftDemap<TStorageIn,
		      TDataRx,
		      TStorageOut,
		      TCompute,
		      N_BS_ANTS,
		      N_SYMBS_PER_THRD_BLK>(Nd, nPrb, nUeGrps, symbolBitmask, launchCfg);
      break;
    }


    default: {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {} nTimeChEsts {} ND {}", __FUNCTION__, nBSAnts, nLayers, nTimeChEsts, Nd);
        break;
    }
    }

}  //puschRxChEq::softDemapKernelSelectL0

//#ifndef FAST_COMPILE
template <typename TStorageIn, typename TDataRx, typename TStorageOut, typename TCompute>
void puschRxChEq::softDemapBluesteinWorkspaceKernelSelectL0(uint16_t                     nBSAnts,
                                          uint8_t                      nLayers,
                                          uint8_t                      nTimeChEsts,
                                          uint8_t                      Nd,
                                          uint16_t                     nPrb,
                                          uint16_t                     nUeGrps,
                                          cuphyPuschRxChEqLaunchCfg_t& launchCfg)
{
    eqMmseSoftDemapBluesteinWorkspace<TStorageIn,
				      TDataRx,
				      TStorageOut,
				      TCompute>(Nd, nPrb, nUeGrps, launchCfg);
}//puschRxChEq::softDemapBluesteinWorkspaceKernelSelectL0

template <typename TStorageIn, typename TDataRx, typename TStorageOut, typename TCompute>
void puschRxChEq::softDemapIdftKernelSelectL0(//uint16_t                     nBSAnts,
					       //uint8_t                      nLayers,
					       //uint8_t                      nTimeChEsts,
                                               uint8_t                      Nd,
                                               uint16_t                     nPrb,
                                               uint16_t                     nUeGrps,
                                               uint16_t                     symbolBitmask,
                                               cuphyPuschRxChEqLaunchCfg_t& launchCfg)
{
    eqMmseSoftDemapIdft<TStorageIn,
			 TDataRx,
			 TStorageOut,
			 TCompute>(Nd, nPrb, nUeGrps, symbolBitmask, launchCfg);
} //puschRxChEq::softDemapIdftKernelSelectL0

template <typename TStorageIn, typename TDataRx, typename TStorageOut, typename TCompute>
void puschRxChEq::softDemapAfterDftKernelSelectL0(uint16_t                     nBSAnts,
                                                  uint8_t                      nLayers,
                                                  uint8_t                      nTimeChEsts,
                                                  uint8_t                      Nd,
                                                  uint16_t                     nPrb,
                                                  uint16_t                     nUeGrps,
                                                  uint16_t                     symbolBitmask,
                                                  cuphyPuschRxChEqLaunchCfg_t& launchCfg)
{
    switch(nBSAnts)
    {
    // nBSAnts == 64
    //TODO

    // nBSAnts == 16
    case 16: {
        constexpr uint32_t N_BS_ANTS = 16; // # of BS antenna (# of rows in H matrix)
#if ENABLE_MULTI_SYMBS_PER_THRD_BLK == 0   // single symbol per slot
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = 1;
#elif ENABLE_MULTI_SYMBS_PER_THRD_BLK == 1 // NUM_SYMBOLS_PER_SLOT per slot
        constexpr uint32_t MAX_THRD_BLK_DIM_Y   = CUDA_MAX_N_THRDS_PER_BLK / (N_BS_ANTS * CUPHY_N_TONES_PER_PRB);
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = NUM_SYMBOLS_PER_SLOT < MAX_THRD_BLK_DIM_Y ? NUM_SYMBOLS_PER_SLOT : MAX_THRD_BLK_DIM_Y;
#elif ENABLE_MULTI_SYMBS_PER_THRD_BLK == 2 // auto-select num symbols per slot
        constexpr uint32_t N_XTHRDS_PER_THRD_BLK = N_BS_ANTS * CUPHY_N_TONES_PER_PRB;
        // let's pick N_SYMBS_PER_THRD_BLK such that block size is multiple of warp size (but no more than OFDM_SYMBOLS_PER_SLOT)
        constexpr uint32_t MAX_THRD_BLK_DIM_Y    = std::min(CUDA_MAX_N_THRDS_PER_BLK, compute_lcm(N_XTHRDS_PER_THRD_BLK, N_THREADS_PER_WARP)) / N_XTHRDS_PER_THRD_BLK;
        constexpr uint32_t N_SYMBS_PER_THRD_BLK  = OFDM_SYMBOLS_PER_SLOT < MAX_THRD_BLK_DIM_Y ? OFDM_SYMBOLS_PER_SLOT : MAX_THRD_BLK_DIM_Y;
#endif
        eqMmseSoftDemapAfterDft<TStorageIn,
                                TDataRx,
                                TStorageOut,
                                TCompute,
                                N_BS_ANTS,
                                N_SYMBS_PER_THRD_BLK>(Nd, nPrb, nUeGrps, symbolBitmask, launchCfg);
    }

    // nBSAnts == 8
    case 8: {
        constexpr uint32_t N_BS_ANTS = 8;  // # of BS antenna (# of rows in H matrix)
#if ENABLE_MULTI_SYMBS_PER_THRD_BLK == 0   // single symbol per slot
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = 1;
#elif ENABLE_MULTI_SYMBS_PER_THRD_BLK == 1 // NUM_SYMBOLS_PER_SLOT per slot
        constexpr uint32_t MAX_THRD_BLK_DIM_Y   = CUDA_MAX_N_THRDS_PER_BLK / (N_BS_ANTS * CUPHY_N_TONES_PER_PRB);
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = NUM_SYMBOLS_PER_SLOT < MAX_THRD_BLK_DIM_Y ? NUM_SYMBOLS_PER_SLOT : MAX_THRD_BLK_DIM_Y;
#elif ENABLE_MULTI_SYMBS_PER_THRD_BLK == 2 // auto-select num symbols per slot
        constexpr uint32_t N_XTHRDS_PER_THRD_BLK = N_BS_ANTS * CUPHY_N_TONES_PER_PRB;
        // let's pick N_SYMBS_PER_THRD_BLK such that block size is multiple of warp size (but no more than OFDM_SYMBOLS_PER_SLOT)
        constexpr uint32_t MAX_THRD_BLK_DIM_Y    = std::min(CUDA_MAX_N_THRDS_PER_BLK, compute_lcm(N_XTHRDS_PER_THRD_BLK, N_THREADS_PER_WARP)) / N_XTHRDS_PER_THRD_BLK;
        constexpr uint32_t N_SYMBS_PER_THRD_BLK  = OFDM_SYMBOLS_PER_SLOT < MAX_THRD_BLK_DIM_Y ? OFDM_SYMBOLS_PER_SLOT : MAX_THRD_BLK_DIM_Y;
#endif
        eqMmseSoftDemapAfterDft<TStorageIn,
                                TDataRx,
                                TStorageOut,
                                TCompute,
                                N_BS_ANTS,
                                N_SYMBS_PER_THRD_BLK>(Nd, nPrb, nUeGrps, symbolBitmask, launchCfg);
        break;
    }

    // nBsAnts == 4
    case 4: {
        constexpr uint32_t N_BS_ANTS = 4;  // # of BS antenna (# of rows in H matrix)
#if ENABLE_MULTI_SYMBS_PER_THRD_BLK == 0   // single symbol per slot
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = 1;
#elif ENABLE_MULTI_SYMBS_PER_THRD_BLK == 1 // NUM_SYMBOLS_PER_SLOT per slot
        constexpr uint32_t MAX_THRD_BLK_DIM_Y   = CUDA_MAX_N_THRDS_PER_BLK / (N_BS_ANTS * CUPHY_N_TONES_PER_PRB);
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = NUM_SYMBOLS_PER_SLOT < MAX_THRD_BLK_DIM_Y ? NUM_SYMBOLS_PER_SLOT : MAX_THRD_BLK_DIM_Y;
#elif ENABLE_MULTI_SYMBS_PER_THRD_BLK == 2 // auto-select num symbols per slot
        constexpr uint32_t N_XTHRDS_PER_THRD_BLK = N_BS_ANTS * CUPHY_N_TONES_PER_PRB;
        // let's pick N_SYMBS_PER_THRD_BLK such that block size is multiple of warp size (but no more than OFDM_SYMBOLS_PER_SLOT)
        constexpr uint32_t MAX_THRD_BLK_DIM_Y    = std::min(CUDA_MAX_N_THRDS_PER_BLK, compute_lcm(N_XTHRDS_PER_THRD_BLK, N_THREADS_PER_WARP)) / N_XTHRDS_PER_THRD_BLK;
        constexpr uint32_t N_SYMBS_PER_THRD_BLK  = OFDM_SYMBOLS_PER_SLOT < MAX_THRD_BLK_DIM_Y ? OFDM_SYMBOLS_PER_SLOT : MAX_THRD_BLK_DIM_Y;
#endif
        eqMmseSoftDemapAfterDft<TStorageIn,
                                TDataRx,
                                TStorageOut,
                                TCompute,
                                N_BS_ANTS,
                                N_SYMBS_PER_THRD_BLK>(Nd, nPrb, nUeGrps, symbolBitmask, launchCfg);
        break;
    }

    // nBsAnts == 2
    case 2: {
        constexpr uint32_t N_BS_ANTS = 2;  // # of BS antenna (# of rows in H matrix)
#if ENABLE_MULTI_SYMBS_PER_THRD_BLK == 0   // single symbol per slot
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = 1;
#elif ENABLE_MULTI_SYMBS_PER_THRD_BLK == 1 // NUM_SYMBOLS_PER_SLOT per slot
        constexpr uint32_t MAX_THRD_BLK_DIM_Y   = CUDA_MAX_N_THRDS_PER_BLK / (N_BS_ANTS * CUPHY_N_TONES_PER_PRB);
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = NUM_SYMBOLS_PER_SLOT < MAX_THRD_BLK_DIM_Y ? NUM_SYMBOLS_PER_SLOT : MAX_THRD_BLK_DIM_Y;
#elif ENABLE_MULTI_SYMBS_PER_THRD_BLK == 2 // auto-select num symbols per slot
        constexpr uint32_t N_XTHRDS_PER_THRD_BLK = N_BS_ANTS * CUPHY_N_TONES_PER_PRB;
        // let's pick N_SYMBS_PER_THRD_BLK such that block size is multiple of warp size (but no more than OFDM_SYMBOLS_PER_SLOT)
        constexpr uint32_t MAX_THRD_BLK_DIM_Y   = std::min(CUDA_MAX_N_THRDS_PER_BLK, compute_lcm(N_XTHRDS_PER_THRD_BLK, N_THREADS_PER_WARP)) / N_XTHRDS_PER_THRD_BLK;
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = OFDM_SYMBOLS_PER_SLOT < MAX_THRD_BLK_DIM_Y ? OFDM_SYMBOLS_PER_SLOT : MAX_THRD_BLK_DIM_Y;
#endif
        eqMmseSoftDemapAfterDft<TStorageIn,
                                TDataRx,
                                TStorageOut,
                                TCompute,
                                N_BS_ANTS,
                                N_SYMBS_PER_THRD_BLK>(Nd, nPrb, nUeGrps, symbolBitmask, launchCfg);
        break;
    }

    // nBsAnts == 1
    case 1: {
        constexpr uint32_t N_BS_ANTS = 1;  // # of BS antenna (# of rows in H matrix)
#if ENABLE_MULTI_SYMBS_PER_THRD_BLK == 0   // single symbol per slot
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = 1;
#elif ENABLE_MULTI_SYMBS_PER_THRD_BLK == 1 // NUM_SYMBOLS_PER_SLOT per slot
        constexpr uint32_t MAX_THRD_BLK_DIM_Y   = CUDA_MAX_N_THRDS_PER_BLK / (N_BS_ANTS * CUPHY_N_TONES_PER_PRB);
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = NUM_SYMBOLS_PER_SLOT < MAX_THRD_BLK_DIM_Y ? NUM_SYMBOLS_PER_SLOT : MAX_THRD_BLK_DIM_Y;
#elif ENABLE_MULTI_SYMBS_PER_THRD_BLK == 2 // auto-select num symbols per slot
        constexpr uint32_t N_XTHRDS_PER_THRD_BLK = N_BS_ANTS * CUPHY_N_TONES_PER_PRB;
        // let's pick N_SYMBS_PER_THRD_BLK such that block size is multiple of warp size (but no more than OFDM_SYMBOLS_PER_SLOT)
        constexpr uint32_t MAX_THRD_BLK_DIM_Y   = std::min(CUDA_MAX_N_THRDS_PER_BLK, compute_lcm(N_XTHRDS_PER_THRD_BLK, N_THREADS_PER_WARP)) / N_XTHRDS_PER_THRD_BLK;
        constexpr uint32_t N_SYMBS_PER_THRD_BLK = OFDM_SYMBOLS_PER_SLOT < MAX_THRD_BLK_DIM_Y ? OFDM_SYMBOLS_PER_SLOT : MAX_THRD_BLK_DIM_Y;
#endif
        eqMmseSoftDemapAfterDft<TStorageIn,
                                TDataRx,
                                TStorageOut,
                                TCompute,
                                N_BS_ANTS,
                                N_SYMBS_PER_THRD_BLK>(Nd, nPrb, nUeGrps, symbolBitmask, launchCfg);
        break;
    }

    default: {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested configuration: nBSAnts {} nLayers {} nTimeChEsts {} ND {}", __FUNCTION__, nBSAnts, nLayers, nTimeChEsts, Nd);
        break;
    }
    }

}//puschRxChEq::softDemapAfterDftKernelSelectL0
//#endif

void puschRxChEq::coefCompKernelSelectL1(uint16_t                     nBSAnts,
                                         uint8_t                      nLayers,
                                         uint8_t                      nTimeChEst,
                                         uint16_t                     nPrb,
                                         uint16_t                     nUeGrps,
                                         cuphyDataType_t              hEstType,                                         
                                         cuphyDataType_t              coefType,
                                         cuphyPuschRxChEqLaunchCfg_t& launchCfg)
{
    //printf("channel_eq::eqCoefCompute() begin()\n");
    using TCompute = float;
    if(CUPHY_C_32F == hEstType)
    {
        using TStorageIn = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
        if(CUPHY_C_32F == coefType)
        {
            using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
            coefCompKernelSelectL0<TStorageIn, TStorageOut, TCompute>(nBSAnts,
                                                                      nLayers,
                                                                      nTimeChEst,
                                                                      nPrb,
                                                                      nUeGrps,
                                                                      launchCfg);
        }
#if 0  // Disabling unused types to reduce compile time
        else if(CUPHY_C_16F == coefType)
        {
            using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
            coefCompKernelSelectL0<TStorageIn, TStorageOut, TCompute>(nBSAnts,
                                                                        nLayers,
                                                                        nTimeChEst,
                                                                        nPrb,
                                                                        nUeGrps,
                                                                        launchCfg);
        }
#endif        
        else
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type (1)", __FUNCTION__);
        }
    }
#if 0  // Disabling unused types to reduce compile time
    else if((CUPHY_C_16F == hEstType) && (CUPHY_C_16F == coefType))
    {
        using TStorageIn  = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        coefCompKernelSelectL0<TStorageIn, TStorageOut, TCompute>(nBSAnts,
                                                                  nLayers,
                                                                  nTimeChEst,
                                                                  nPrb,
                                                                  nUeGrps,
                                                                  launchCfg);
    }
    else if((CUPHY_C_16F == hEstType) && (CUPHY_C_32F == coefType))
    {
        using TStorageIn  = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
        coefCompKernelSelectL0<TStorageIn, TStorageOut, TCompute>(nBSAnts,
                                                                  nLayers,
                                                                  nTimeChEst,
                                                                  nPrb,
                                                                  nUeGrps,
                                                                  launchCfg);
    }
#endif    
    else
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type (2)", __FUNCTION__);
    }
    //printf("channel_eq::eqCoefCompute end()\n");
}

void puschRxChEq::softDemapKernelSelectL1(uint16_t                     nBSAnts,
                                          uint8_t                      nLayers,
                                          uint8_t                      nTimeChEst,
                                          uint8_t                      Nd,
                                          uint16_t                     nPrb,
                                          uint16_t                     nUeGrps,
                                          uint16_t                     symbolBitmask,
                                          cuphyDataType_t              coefType,
                                          cuphyDataType_t              dataRxType,
                                          cuphyDataType_t              llrType,
                                          cuphyPuschRxChEqLaunchCfg_t& launchCfg)
{
    using TCompute = float;
    if(CUPHY_C_32F == coefType)
    {
        using TStorageIn = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
        // FP16 LLRs
        if((CUPHY_C_16F == dataRxType) && (CUPHY_R_16F == llrType))
        {
            using TDataRx     = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
            using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
            softDemapKernelSelectL0<TStorageIn, TDataRx, TStorageOut, TCompute>(nBSAnts,
                                                                                nLayers,
                                                                                nTimeChEst,
                                                                                Nd,
                                                                                nPrb,
                                                                                nUeGrps,
                                                                                symbolBitmask,
                                                                                launchCfg);
        }        
#if 0  // Disabling unused types to reduce compile time
        else if(CUPHY_C_32F == dataRxType)
        {
            using TDataRx = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
            if(CUPHY_R_32F == llrType)
            {
                using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
                softDemapKernelSelectL0<TStorageIn, TDataRx, TStorageOut, TCompute>(nBSAnts,
                                                                                    nLayers,
                                                                                    nTimeChEst,
                                                                                    Nd,
                                                                                    nPrb,
                                                                                    nUeGrps,
                                                                                    symbolBitmask,
                                                                                    launchCfg);
            }
            // FP16 LLRs
            else if(CUPHY_R_16F == llrType)
            {
                using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
                softDemapKernelSelectL0<TStorageIn, TDataRx, TStorageOut, TCompute>(nBSAnts,
                                                                                    nLayers,
                                                                                    nTimeChEst,
                                                                                    Nd,
                                                                                    nPrb,
                                                                                    nUeGrps,
                                                                                    symbolBitmask,
                                                                                    launchCfg);
            }
            else
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type (1)", __FUNCTION__);
            }
        }
        // supporting this mode until de-rate matching needs FP32 LLRs
        else if((CUPHY_C_16F == dataRxType) && (CUPHY_R_32F == llrType))
        {
            using TDataRx     = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
            using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
            softDemapKernelSelectL0<TStorageIn, TDataRx, TStorageOut, TCompute>(nBSAnts,
                                                                                nLayers,
                                                                                nTimeChEst,
                                                                                Nd,
                                                                                nPrb,
                                                                                nUeGrps,
                                                                                symbolBitmask,
                                                                                launchCfg);
        }
#endif        
        else
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type (2)", __FUNCTION__);
        }
    }
#if 0  // Disabling unused types to reduce compile time
    else if((CUPHY_C_16F == coefType) && (CUPHY_C_16F == dataRxType) &&
            (CUPHY_R_16F == llrType))
    {
        using TStorageIn  = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        using TDataRx     = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;

        softDemapKernelSelectL0<TStorageIn, TDataRx, TStorageOut, TCompute>(nBSAnts,
                                                                            nLayers,
                                                                            nTimeChEst,
                                                                            Nd,
                                                                            nPrb,
                                                                            nUeGrps,
                                                                            symbolBitmask,
                                                                            launchCfg);
    }
#endif    
    else
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type (3)", __FUNCTION__);
    }
    //printf("channel_eq::eqSoftDemap end()\n");
} //softDemapKernelSelectL1

//#ifndef FAST_COMPILE
void puschRxChEq::softDemapBluesteinWorkspaceKernelSelectL1(uint16_t                     nBSAnts,
                                          uint8_t                      nLayers,
                                          uint8_t                      nTimeChEst,
                                          uint8_t                      Nd,
                                          uint16_t                     nPrb,
                                          uint16_t                     nUeGrps,
                                          cuphyDataType_t              coefType,
                                          cuphyDataType_t              dataRxType,
                                          cuphyDataType_t              llrType,
                                          cuphyPuschRxChEqLaunchCfg_t& launchCfg)
{
    using TCompute = float;
    if(CUPHY_C_32F == coefType)
    {
        using TStorageIn = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
        // FP16 LLRs
        if((CUPHY_C_16F == dataRxType) && (CUPHY_R_16F == llrType))
        {
            using TDataRx     = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
            using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
            softDemapBluesteinWorkspaceKernelSelectL0<TStorageIn, TDataRx, TStorageOut, TCompute>(nBSAnts,
                                                                                nLayers,
                                                                                nTimeChEst,
                                                                                Nd,
                                                                                nPrb,
                                                                                nUeGrps,
                                                                                launchCfg);
        }        
#if 0  // Disabling unused types to reduce compile time
        else if(CUPHY_C_32F == dataRxType)
        {
            using TDataRx = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
            if(CUPHY_R_32F == llrType)
            {
                using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
                softDemapBluesteinWorkspaceKernelSelectL0<TStorageIn, TDataRx, TStorageOut, TCompute>(nBSAnts,
                                                                                    nLayers,
                                                                                    nTimeChEst,
                                                                                    Nd,
                                                                                    nPrb,
                                                                                    nUeGrps,
                                                                                    launchCfg);
            }
            // FP16 LLRs
            else if(CUPHY_R_16F == llrType)
            {
                using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
                softDemapBluesteinWorkspaceKernelSelectL0<TStorageIn, TDataRx, TStorageOut, TCompute>(nBSAnts,
                                                                                    nLayers,
                                                                                    nTimeChEst,
                                                                                    Nd,
                                                                                    nPrb,
                                                                                    nUeGrps,
                                                                                    launchCfg);
            }
            else
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type (1)", __FUNCTION__);
            }
        }
        // supporting this mode until de-rate matching needs FP32 LLRs
        else if((CUPHY_C_16F == dataRxType) && (CUPHY_R_32F == llrType))
        {
            using TDataRx     = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
            using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
            softDemapBluesteinWorkspaceKernelSelectL0<TStorageIn, TDataRx, TStorageOut, TCompute>(nBSAnts,
                                                                                nLayers,
                                                                                nTimeChEst,
                                                                                Nd,
                                                                                nPrb,
                                                                                nUeGrps,
                                                                                launchCfg);
        }
#endif        
        else
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type (2)", __FUNCTION__);
        }
    }
#if 0  // Disabling unused types to reduce compile time
    else if((CUPHY_C_16F == coefType) && (CUPHY_C_16F == dataRxType) &&
            (CUPHY_R_16F == llrType))
    {
        using TStorageIn  = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        using TDataRx     = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;

        softDemapBluesteinWorkspaceKernelSelectL0<TStorageIn, TDataRx, TStorageOut, TCompute>(nBSAnts,
                                                                            nLayers,
                                                                            nTimeChEst,
                                                                            Nd,
                                                                            nPrb,
                                                                            nUeGrps,
                                                                            launchCfg);
    }
#endif    
    else
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type (3)", __FUNCTION__);
    }

} //puschRxChEq::softDemapBluesteinWorkspaceKernelSelectL1

void puschRxChEq::softDemapIdftKernelSelectL1(uint8_t                      Nd,
                                               uint16_t                     nPrb,
                                               uint16_t                     nUeGrps,
                                               uint16_t                     symbolBitmask,
                                               cuphyDataType_t              coefType,
                                               cuphyDataType_t              dataRxType,
                                               cuphyDataType_t              llrType,
                                               cuphyPuschRxChEqLaunchCfg_t& launchCfg)
{

    using TCompute = float;
    if(CUPHY_C_32F == coefType)
    {
        using TStorageIn = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
        // FP16 LLRs
        if((CUPHY_C_16F == dataRxType) && (CUPHY_R_16F == llrType))
        {
            using TDataRx     = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
            using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
            softDemapIdftKernelSelectL0<TStorageIn, TDataRx, TStorageOut, TCompute>(Nd, nPrb, nUeGrps, symbolBitmask, launchCfg);
        }        
#if 0  // Disabling unused types to reduce compile time
        else if(CUPHY_C_32F == dataRxType)
        {
            using TDataRx = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
            if(CUPHY_R_32F == llrType)
            {
                using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
                softDemapIdftKernelSelectL0<TStorageIn, TDataRx, TStorageOut, TCompute>(Nd, nPrb, nUeGrps, symbolBitmask, launchCfg);
            }
            // FP16 LLRs
            else if(CUPHY_R_16F == llrType)
            {
                using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
                softDemapIdftKernelSelectL0<TStorageIn, TDataRx, TStorageOut, TCompute>(Nd, nPrb, nUeGrps, symbolBitmask, launchCfg);
            }
            else
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type (1)", __FUNCTION__);
            }
        }
        // supporting this mode until de-rate matching needs FP32 LLRs
        else if((CUPHY_C_16F == dataRxType) && (CUPHY_R_32F == llrType))
        {
            using TDataRx     = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
            using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
            softDemapIdftKernelSelectL0<TStorageIn, TDataRx, TStorageOut, TCompute>(Nd, nPrb, nUeGrps, symbolBitmask, launchCfg);
        }
#endif        
        else
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type (2)", __FUNCTION__);
        }
    }
#if 0  // Disabling unused types to reduce compile time
    else if((CUPHY_C_16F == coefType) && (CUPHY_C_16F == dataRxType) &&
            (CUPHY_R_16F == llrType))
    {
        using TStorageIn  = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        using TDataRx     = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;

        softDemapIdftKernelSelectL0<TStorageIn, TDataRx, TStorageOut, TCompute>(Nd, nPrb, nUeGrps, symbolBitmask, launchCfg);
    }
#endif    
    else
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type (3)", __FUNCTION__);
    }

} //puschRxChEq::softDemapIdftKernelSelectL1

void puschRxChEq::softDemapAfterDftKernelSelectL1(uint16_t                     nBSAnts,
                                                  uint8_t                      nLayers,
                                                  uint8_t                      nTimeChEst,
                                                  uint8_t                      Nd,
                                                  uint16_t                     nPrb,
                                                  uint16_t                     nUeGrps,
                                                  uint16_t                     symbolBitmask,
                                                  cuphyDataType_t              coefType,
                                                  cuphyDataType_t              dataRxType,
                                                  cuphyDataType_t              llrType,
                                                  cuphyPuschRxChEqLaunchCfg_t& launchCfg)
{
    using TCompute = float;
    if(CUPHY_C_32F == coefType)
    {
        using TStorageIn = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
        // FP16 LLRs
        if((CUPHY_C_16F == dataRxType) && (CUPHY_R_16F == llrType))
        {
            using TDataRx     = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
            using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
            softDemapAfterDftKernelSelectL0<TStorageIn, TDataRx, TStorageOut, TCompute>(nBSAnts,
                                                                                nLayers,
                                                                                nTimeChEst,
                                                                                Nd,
                                                                                nPrb,
                                                                                nUeGrps,
                                                                                symbolBitmask,
                                                                                launchCfg);
        }        
#if 0  // Disabling unused types to reduce compile time
        else if(CUPHY_C_32F == dataRxType)
        {
            using TDataRx = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
            if(CUPHY_R_32F == llrType)
            {
                using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
                softDemapAfterDftKernelSelectL0<TStorageIn, TDataRx, TStorageOut, TCompute>(nBSAnts,
                                                                                    nLayers,
                                                                                    nTimeChEst,
                                                                                    Nd,
                                                                                    nPrb,
                                                                                    nUeGrps,
                                                                                    symbolBitmask,
                                                                                    launchCfg);
            }
            // FP16 LLRs
            else if(CUPHY_R_16F == llrType)
            {
                using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
                softDemapAfterDftKernelSelectL0<TStorageIn, TDataRx, TStorageOut, TCompute>(nBSAnts,
                                                                                    nLayers,
                                                                                    nTimeChEst,
                                                                                    Nd,
                                                                                    nPrb,
                                                                                    nUeGrps,
                                                                                    symbolBitmask,
                                                                                    launchCfg);
            }
            else
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type (1)", __FUNCTION__);
            }
        }
        // supporting this mode until de-rate matching needs FP32 LLRs
        else if((CUPHY_C_16F == dataRxType) && (CUPHY_R_32F == llrType))
        {
            using TDataRx     = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
            using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
            softDemapAfterDftKernelSelectL0<TStorageIn, TDataRx, TStorageOut, TCompute>(nBSAnts,
                                                                                nLayers,
                                                                                nTimeChEst,
                                                                                Nd,
                                                                                nPrb,
                                                                                nUeGrps,
                                                                                symbolBitmask,
                                                                                launchCfg);
        }
#endif        
        else
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type (2)", __FUNCTION__);
        }
    }
#if 0  // Disabling unused types to reduce compile time
    else if((CUPHY_C_16F == coefType) && (CUPHY_C_16F == dataRxType) &&
            (CUPHY_R_16F == llrType))
    {
        using TStorageIn  = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        using TDataRx     = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        using TStorageOut = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;

        softDemapAfterDftKernelSelectL0<TStorageIn, TDataRx, TStorageOut, TCompute>(nBSAnts,
                                                                            nLayers,
                                                                            nTimeChEst,
                                                                            Nd,
                                                                            nPrb,
                                                                            nUeGrps,
                                                                            symbolBitmask,
                                                                            launchCfg);
    }
#endif    
    else
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type (3)", __FUNCTION__);
    }
} // puschRxChEq::softDemapAfterDftKernelSelectL1
//#endif

void puschRxChEq::init(cuphyContext_t hctx,
                       bool           enableCpuToGpuDescrAsyncCpy,
                       void**         ppStatDescrsCpu,
                       void**         ppStatDescrsGpu,
                       cudaStream_t   strm)
{
    for(int32_t chEqTimeInstIdx = 0; chEqTimeInstIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ; ++chEqTimeInstIdx)
    {
        puschRxChEqStatDescr_t& statDescrCpu = *(static_cast<puschRxChEqStatDescr_t*>(ppStatDescrsCpu[chEqTimeInstIdx]));

        //------------------------------------------------------------------
        // Soft demapper texture object initialization (static descriptor)
        cuphy_i::context& ctx     = static_cast<cuphy_i::context&>(*hctx);
        statDescrCpu.demapper_tex = ctx.soft_demapper_ctx().QAM_tex().tex_obj().handle();

        for(auto& kernelArgs : m_coefCompKernelArgsArr[chEqTimeInstIdx])
        {
            kernelArgs.pStatDescr = static_cast<puschRxChEqStatDescr_t*>(ppStatDescrsCpu[chEqTimeInstIdx]);
        }        

        if(enableCpuToGpuDescrAsyncCpy)
        {
            CUDA_CHECK(cudaMemcpyAsync(ppStatDescrsGpu[chEqTimeInstIdx], ppStatDescrsCpu[chEqTimeInstIdx], sizeof(puschRxChEqStatDescr_t), cudaMemcpyHostToDevice, strm));
        }
        // m_coefCompHetCfgsVecArr[chEqTimeInstIdx].resize(CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS, puschRxChEqCoefCompHetCfg_t{nullptr, 0, 0});
    }
    m_coefCompHetCfgsVecArr.fill({CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS, puschRxChEqCoefCompHetCfg_t{nullptr, 0, 0}});

    for(int32_t idx = 0; idx < 2; ++idx)
    {
        for(auto& kernelArgs : m_softDemapKernelArgsArr[idx])
        {
            kernelArgs.pStatDescr = static_cast<puschRxChEqStatDescr_t*>(ppStatDescrsCpu[idx]);
        }
        
        for(auto& kernelArgs : m_softDemapIdftKernelArgsArr[idx])
        {
            kernelArgs.pStatDescr = static_cast<puschRxChEqStatDescr_t*>(ppStatDescrsCpu[idx]);
        }
        
        for(auto& kernelArgs : m_softDemapAfterDftKernelArgsArr[idx])
        {
            kernelArgs.pStatDescr = static_cast<puschRxChEqStatDescr_t*>(ppStatDescrsCpu[idx]);
        }
    }
    m_softDemapHetCfgVec.fill({CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS, puschRxChEqSoftDemapHetCfg_t{nullptr, 0, 0, 0}});
    m_softDemapIdftHetCfgVec.fill({CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS, puschRxChEqSoftDemapHetCfg_t{nullptr, 0, 0, 0}});
    m_softDemapAfterDftHetCfgVec.fill({CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS, puschRxChEqSoftDemapHetCfg_t{nullptr, 0, 0, 0}});
    
    for(auto& kernelArgs : m_softDemapBluesteinWorkspaceKernelArgsArr)
    {
        kernelArgs.pStatDescr = static_cast<puschRxChEqStatDescr_t*>(ppStatDescrsCpu[0]);
    }   
}


void puschRxChEq::getDescrInfo(size_t& statDescrSizeBytes,
                               size_t& statDescrAlignBytes,
                               size_t& coefCompDynDescrSizeBytes,
                               size_t& coefCompDynDescrAlignBytes,
                               size_t& softDemapDynDescrSizeBytes,
                               size_t& softDemapDynDescrAlignBytes)
{
    statDescrSizeBytes  = sizeof(puschRxChEqStatDescr_t);
    statDescrAlignBytes = alignof(puschRxChEqStatDescr_t);

    coefCompDynDescrSizeBytes  = sizeof(puschRxChEqCoefCompDynDescrVec_t);
    coefCompDynDescrAlignBytes = alignof(puschRxChEqCoefCompDynDescrVec_t);

    softDemapDynDescrSizeBytes  = sizeof(puschRxChEqSoftDemapDynDescrVec_t);
    softDemapDynDescrAlignBytes = alignof(puschRxChEqSoftDemapDynDescrVec_t);
}

cuphyStatus_t puschRxChEq::batchEqCoefComp(uint32_t                          chEqTimeInstIdx,
                                  cuphyPuschRxUeGrpPrms_t*          pDrvdUeGrpPrms,
                                  uint16_t                          nUeGrps,
                                  uint32_t&                         nHetCfgs,
                                  puschRxChEqCoefCompDynDescrVec_t& dynDescrVecCpu)
{
    // Helper to find kernel function
    auto findKernelFunc = [](puschRxChEqCoefCompHetCfgVec_t const& hetCfgs, CUfunction func, int32_t& hetCfgIdx) {
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

    // Initialize the batch config data structure
    puschRxChEqCoefCompHetCfgVec_t& hetCfgs = m_coefCompHetCfgsVecArr[chEqTimeInstIdx];        
    std::fill(hetCfgs.begin(), hetCfgs.end(), puschRxChEqCoefCompHetCfg_t{nullptr, 0, 0});

#ifdef ENABLE_DEBUG
    NVLOGI_FMT(NVLOG_PUSCH, "{}: # of UE groups {}", __FUNCTION__, nUeGrps);
#endif

    nHetCfgs = 0;
    for(int32_t ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
    {
        cuphyPuschRxUeGrpPrms_t const& drvdUeGrpPrms = pDrvdUeGrpPrms[ueGrpIdx];

        // Skip UE group if there aren't enough DMRS additional positions
        // # of time domain channel estimates and equalizer coefficients is equal to the number of DMRS additional positions + 1
        if(chEqTimeInstIdx > drvdUeGrpPrms.dmrsAddlnPos)  continue;

        uint16_t nPrb = drvdUeGrpPrms.nPrb;

        // @todo: extend kernelSelectL2 to support a mode which only determines the kernel function
        cuphyPuschRxChEqLaunchCfg_t launchCfg;
        coefCompKernelSelectL1(drvdUeGrpPrms.nRxAnt,
                               drvdUeGrpPrms.nLayers,
                               chEqTimeInstIdx + 1,
                               nPrb,
                               nUeGrps,
                               drvdUeGrpPrms.tInfoHEst.elemType,
                               drvdUeGrpPrms.tInfoEqCoef.elemType,
                               launchCfg);

        // Check if the heterognous configuration already exists
        int32_t hetCfgIdx = 0;
        findKernelFunc(hetCfgs, launchCfg.kernelNodeParamsDriver.func, hetCfgIdx);

        // If a heterogenous configuration already exists then increment the # of UE groups for that config
        if(-1 != hetCfgIdx)
        {
            puschRxChEqCoefCompHetCfg_t& hetCfg = hetCfgs[hetCfgIdx];
            if(hetCfg.nUeGrps >= MAX_N_USER_GROUPS_SUPPORTED)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Exceeded limit on supported UE groups", __FUNCTION__);
                return CUPHY_STATUS_INTERNAL_ERROR;
            }

            if(nPrb > hetCfg.nMaxPrb) hetCfg.nMaxPrb = nPrb;

            dynDescrVecCpu[hetCfgIdx].hetCfgUeGrpMap[hetCfg.nUeGrps] = ueGrpIdx;
            hetCfg.nUeGrps++;

#ifdef ENABLE_DEBUG
            NVLOGI_FMT(NVLOG_PUSCH, "{}: chEqTimeInstIdx {} UE group {} -> HetCfg {} (nUeGrps {} nPrb {} nMaxPrb {} nRxAnt {} nLayers {})", __FUNCTION__, chEqTimeInstIdx, ueGrpIdx, hetCfgIdx, hetCfg.nUeGrps, nPrb, hetCfg.nMaxPrb, drvdUeGrpPrms.nRxAnt, drvdUeGrpPrms.nLayers);
#endif
        }
        // New heterogenous configuration found
        else
        {
            if(nHetCfgs >= CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Exceeded limit on supported heterogneous configurations", __FUNCTION__);
                return CUPHY_STATUS_INTERNAL_ERROR;
            }

            int32_t      newHetCfgIdx           = nHetCfgs++;
            puschRxChEqCoefCompHetCfg_t& hetCfg = hetCfgs[newHetCfgIdx];
            hetCfg.func                         = launchCfg.kernelNodeParamsDriver.func;
            hetCfg.nMaxPrb                      = nPrb;

            dynDescrVecCpu[newHetCfgIdx].hetCfgUeGrpMap[hetCfg.nUeGrps] = ueGrpIdx;
            hetCfg.nUeGrps++;

#ifdef ENABLE_DEBUG
            NVLOGI_FMT(NVLOG_PUSCH, "{}: chEqTimeInstIdx {} UE group {} -> HetCfg {} (nUeGrps {} nPrb {} nMaxPrb {} nRxAnt {} nLayers {})", __FUNCTION__, chEqTimeInstIdx, ueGrpIdx, newHetCfgIdx, hetCfg.nUeGrps, nPrb, hetCfg.nMaxPrb, drvdUeGrpPrms.nRxAnt, drvdUeGrpPrms.nLayers);
#endif
        }
    }
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t puschRxChEq::setupCoefCompute(cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsCpu,
                                            cuphyPuschRxUeGrpPrms_t*      pDrvdUeGrpPrmsGpu,
                                            uint16_t                      nUeGrps,
                                            uint16_t                      nMaxPrb,
                                            uint8_t                       enableCfoCorrection,
                                            uint8_t                       enablePuschTdi,
                                            bool                          enableCpuToGpuDescrAsyncCpy,
                                            void**                        ppDynDescrsCpu,
                                            void**                        ppDynDescrsGpu,
                                            cuphyPuschRxChEqLaunchCfgs_t* pLaunchCfgs,
                                            cudaStream_t                  strm)
{

    if(!pDrvdUeGrpPrmsCpu || !pDrvdUeGrpPrmsGpu || !ppDynDescrsCpu || !ppDynDescrsGpu || !pLaunchCfgs) return CUPHY_STATUS_INVALID_ARGUMENT;

    int32_t estimates = 1;
    if (enablePuschTdi) {
        estimates = CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ;
    }

    for(int32_t chEqTimeInstIdx = 0; chEqTimeInstIdx < estimates; ++chEqTimeInstIdx)
    {
        if(!ppDynDescrsCpu[chEqTimeInstIdx] || !ppDynDescrsGpu[chEqTimeInstIdx]) return CUPHY_STATUS_INVALID_ARGUMENT;

        cuphyPuschRxChEqLaunchCfgs_t&     launchCfgs     = pLaunchCfgs[chEqTimeInstIdx];
        puschRxChEqCoefCompDynDescrVec_t& dynDescrVecCpu = *(static_cast<puschRxChEqCoefCompDynDescrVec_t*>(ppDynDescrsCpu[chEqTimeInstIdx]));
        cuphyStatus_t status = batchEqCoefComp(chEqTimeInstIdx,
                                               pDrvdUeGrpPrmsCpu,
                                               nUeGrps,
                                               launchCfgs.nCfgs,
                                               dynDescrVecCpu);

        if(CUPHY_STATUS_SUCCESS != status)
        {
            return status;
        }


        puschRxChEqCoefCompDynDescrVec_t& dynDescrVecGpu = *(static_cast<puschRxChEqCoefCompDynDescrVec_t*>(ppDynDescrsGpu[chEqTimeInstIdx]));
        for(uint32_t hetCfgIdx = 0; hetCfgIdx < launchCfgs.nCfgs; ++hetCfgIdx)
        {
            // Skip rest of the setup if there are no UE groups corresponding to the channel equalizer instance and hetCfg
            if(0 == m_coefCompHetCfgsVecArr[chEqTimeInstIdx][hetCfgIdx].nUeGrps) continue;
            
            // Setup descriptor in CPU memory
            puschRxChEqCoefCompDynDescr_t&  dynDescr    = dynDescrVecCpu[hetCfgIdx];
            puschRxChEqCoefCompHetCfg_t const& hetCfg   = m_coefCompHetCfgsVecArr[chEqTimeInstIdx][hetCfgIdx];
            puschRxChEqCoefCompKernelArgs_t& kernelArgs = m_coefCompKernelArgsArr[chEqTimeInstIdx][hetCfgIdx];

            dynDescr.chEqTimeInstIdx = chEqTimeInstIdx;
            dynDescr.pDrvdUeGrpPrms = pDrvdUeGrpPrmsGpu;

            // Optional descriptor copy to GPU memory
            if(enableCpuToGpuDescrAsyncCpy)
            {
                CUDA_CHECK(cudaMemcpyAsync(&dynDescrVecGpu[hetCfgIdx], &dynDescr, sizeof(puschRxChEqCoefCompDynDescr_t), cudaMemcpyHostToDevice, strm));
            }

            // Select kernel
            cuphyPuschRxChEqLaunchCfg_t& launchCfg = launchCfgs.cfgs[hetCfgIdx];

            // TODO: Optimize function to determine kernel selection and launch geometry separately
            int32_t ueGrpIdx = dynDescr.hetCfgUeGrpMap[0];
            cuphyPuschRxUeGrpPrms_t const& drvdUeGrpPrms = pDrvdUeGrpPrmsCpu[ueGrpIdx];                
            coefCompKernelSelectL1(drvdUeGrpPrms.nRxAnt,
                                   drvdUeGrpPrms.nLayers,
                                   chEqTimeInstIdx + 1,
                                   hetCfg.nMaxPrb,
                                   hetCfg.nUeGrps,
                                   drvdUeGrpPrms.tInfoHEst.elemType,
                                   drvdUeGrpPrms.tInfoEqCoef.elemType,
                                   launchCfg);

            if(hetCfg.func != launchCfg.kernelNodeParamsDriver.func)
            {
               NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Kernel function mismatch", __FUNCTION__);
               return CUPHY_STATUS_INTERNAL_ERROR;
            }                                   

            kernelArgs.pDynDescr    = &dynDescrVecGpu[hetCfgIdx];            
            launchCfg.kernelArgs[0] = &kernelArgs.pStatDescr;
            launchCfg.kernelArgs[1] = &kernelArgs.pDynDescr;

            launchCfg.kernelNodeParamsDriver.kernelParams = &(launchCfg.kernelArgs[0]);
        }
    }
    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t puschRxChEq::batchEqSoftDemap(cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrms,
                                            uint16_t                           nUeGrps,
                                            uint16_t                           symbolBitmask,
                                            uint32_t&                          nHetCfgs,
                                            puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu)
{
    // Helper to find kernel function
    auto findKernelFunc = [](puschRxChEqSoftDemapHetCfgVec_t const& hetCfgs, CUfunction func, int32_t& hetCfgIdx) {
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

    // Initialize the batch config data structure    
    puschRxChEqSoftDemapHetCfgVec_t& hetCfgs = (symbolBitmask==CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK) ? m_softDemapHetCfgVec[0]:m_softDemapHetCfgVec[1];
    std::fill(hetCfgs.begin(), hetCfgs.end(), puschRxChEqSoftDemapHetCfg_t{nullptr, 0, 0, 0});

#if 0
    // Debug code to initialize hetCfgUeGrpMap with -1s   
    for(puschRxChEqSoftDemapDynDescr_t& dynDescrCpu : dynDescrVecCpu)
    {
        std::fill(std::begin(dynDescrCpu.hetCfgUeGrpMap), std::end(dynDescrCpu.hetCfgUeGrpMap), -1);
    }
#endif    
    

#ifdef ENABLE_DEBUG
    NVLOGI_FMT(NVLOG_PUSCH, "{}: # of UE groups {}", __FUNCTION__, nUeGrps);
#endif

    nHetCfgs = 0;
    for(int32_t ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
    {
        cuphyPuschRxUeGrpPrms_t const& drvdUeGrpPrms = pDrvdUeGrpPrms[ueGrpIdx];

        uint16_t nPrb    = drvdUeGrpPrms.nPrb;
        uint8_t nDataSym = drvdUeGrpPrms.nDataSym;
        if(drvdUeGrpPrms.nDmrsCdmGrpsNoData==1)
        {
            nDataSym += drvdUeGrpPrms.nDmrsSyms;
        }

        // @todo: extend kernelSelectL1 to support a mode which only determines the kernel function
        cuphyPuschRxChEqLaunchCfg_t launchCfg;
        softDemapKernelSelectL1(drvdUeGrpPrms.nRxAnt,
                                drvdUeGrpPrms.nLayers,
                                drvdUeGrpPrms.nTimeChEsts,
                                nDataSym,
                                nPrb,
                                nUeGrps,
                                symbolBitmask,
                                drvdUeGrpPrms.tInfoEqCoef.elemType,
                                drvdUeGrpPrms.tInfoDataRx.elemType,
                                drvdUeGrpPrms.tInfoLLR.elemType,
                                launchCfg);

        // Check if the heterognous configuration already exists
        int32_t hetCfgIdx = 0;
        findKernelFunc(hetCfgs, launchCfg.kernelNodeParamsDriver.func, hetCfgIdx);

        // If a heterogenous configuration already exists then increment the # of UE groups for that config
        if(-1 != hetCfgIdx)
        {
            puschRxChEqSoftDemapHetCfg_t& hetCfg = hetCfgs[hetCfgIdx];
            if(hetCfg.nUeGrps >= MAX_N_USER_GROUPS_SUPPORTED)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Exceeded limit on supported UE groups", __FUNCTION__);
                return CUPHY_STATUS_INTERNAL_ERROR;
            }

            if(nPrb > hetCfg.nMaxPrb) hetCfg.nMaxPrb             = nPrb;
            if(nDataSym > hetCfg.nMaxDataSym) hetCfg.nMaxDataSym = nDataSym;

            dynDescrVecCpu[hetCfgIdx].hetCfgUeGrpMap[hetCfg.nUeGrps] = ueGrpIdx;
            hetCfg.nUeGrps++;

#ifdef ENABLE_DEBUG
            NVLOGI_FMT(NVLOG_PUSCH, "{}: UE group {} -> HetCfg {} (nUeGrps {} nPrb {} nMaxPrb {} nDataSym {} nMaxDataSym {} nRxAnt {} nLayers {})", __FUNCTION__, ueGrpIdx, hetCfgIdx, hetCfg.nUeGrps, nPrb, hetCfg.nMaxPrb, nDataSym, hetCfg.nMaxDataSym, drvdUeGrpPrms.nRxAnt, drvdUeGrpPrms.nLayers);
#endif
        }
        // New heterogenous configuration found
        else
        {
            if(nHetCfgs >= CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Exceeded limit on supported heterogneous configurations", __FUNCTION__);
                return CUPHY_STATUS_INTERNAL_ERROR;
            }

            int32_t      newHetCfgIdx            = nHetCfgs++;
            puschRxChEqSoftDemapHetCfg_t& hetCfg = hetCfgs[newHetCfgIdx];
            hetCfg.func                          = launchCfg.kernelNodeParamsDriver.func;
            hetCfg.nMaxPrb                       = nPrb;
            hetCfg.nMaxDataSym                   = nDataSym;

            dynDescrVecCpu[newHetCfgIdx].hetCfgUeGrpMap[hetCfg.nUeGrps] = ueGrpIdx;
            hetCfg.nUeGrps++;

#ifdef ENABLE_DEBUG
            NVLOGI_FMT(NVLOG_PUSCH, "{}: UE group {} -> HetCfg {} (nUeGrps {} nPrb {} nMaxPrb {} nDataSym {} nMaxDataSym {} nRxAnt {} nLayers {})", __FUNCTION__, ueGrpIdx, newHetCfgIdx, hetCfg.nUeGrps, nPrb, hetCfg.nMaxPrb, nDataSym, hetCfg.nMaxDataSym, drvdUeGrpPrms.nRxAnt, drvdUeGrpPrms.nLayers);
#endif
        }
    }
    return CUPHY_STATUS_SUCCESS;
} //batchEqSoftDemap

//#ifndef FAST_COMPILE

cuphyStatus_t puschRxChEq::batchEqSoftDemapBluesteinWorkspace(cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrms,
                                   uint16_t                           nUeGrps,
                                   uint32_t&                          nHetCfgs,
                                   puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu)
{
    // Helper to find kernel function
    auto findKernelFunc = [](puschRxChEqSoftDemapHetCfgVec_t const& hetCfgs, CUfunction func, int32_t& hetCfgIdx) {
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

    // Initialize the batch config data structure    
    puschRxChEqSoftDemapHetCfgVec_t& hetCfgs = m_softDemapBluesteinWorkspaceHetCfgVec;
    std::fill(hetCfgs.begin(), hetCfgs.end(), puschRxChEqSoftDemapHetCfg_t{nullptr, 0, 0, 0});

#if 0
    // Debug code to initialize hetCfgUeGrpMap with -1s   
    for(puschRxChEqSoftDemapDynDescr_t& dynDescrCpu : dynDescrVecCpu)
    {
        std::fill(std::begin(dynDescrCpu.hetCfgUeGrpMap), std::end(dynDescrCpu.hetCfgUeGrpMap), -1);
    }
#endif    
    

#ifdef ENABLE_DEBUG
    NVLOGI_FMT(NVLOG_PUSCH, "{}: # of UE groups {}", __FUNCTION__, nUeGrps);
#endif

    nHetCfgs = 0;
    for(int32_t ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
    {
        cuphyPuschRxUeGrpPrms_t const& drvdUeGrpPrms = pDrvdUeGrpPrms[ueGrpIdx];

        uint16_t nPrb    = drvdUeGrpPrms.nPrb;
        uint8_t nDataSym = drvdUeGrpPrms.nDataSym;

        // @todo: extend kernelSelectL1 to support a mode which only determines the kernel function
        cuphyPuschRxChEqLaunchCfg_t launchCfg;
        softDemapBluesteinWorkspaceKernelSelectL1(drvdUeGrpPrms.nRxAnt,
                                drvdUeGrpPrms.nLayers,
                                drvdUeGrpPrms.nTimeChEsts,
                                nDataSym,
                                nPrb,
                                nUeGrps,
                                drvdUeGrpPrms.tInfoEqCoef.elemType,
                                drvdUeGrpPrms.tInfoDataRx.elemType,
                                drvdUeGrpPrms.tInfoLLR.elemType,
                                launchCfg);

        // Check if the heterognous configuration already exists
        int32_t hetCfgIdx = 0;
        findKernelFunc(hetCfgs, launchCfg.kernelNodeParamsDriver.func, hetCfgIdx);

        // If a heterogenous configuration already exists then increment the # of UE groups for that config
        if(-1 != hetCfgIdx)
        {
            puschRxChEqSoftDemapHetCfg_t& hetCfg = hetCfgs[hetCfgIdx];
            if(hetCfg.nUeGrps >= MAX_N_USER_GROUPS_SUPPORTED)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Exceeded limit on supported UE groups", __FUNCTION__);
                return CUPHY_STATUS_INTERNAL_ERROR;
            }

            if(nPrb > hetCfg.nMaxPrb) hetCfg.nMaxPrb             = nPrb;
            if(nDataSym > hetCfg.nMaxDataSym) hetCfg.nMaxDataSym = nDataSym;

            dynDescrVecCpu[hetCfgIdx].hetCfgUeGrpMap[hetCfg.nUeGrps] = ueGrpIdx;
            hetCfg.nUeGrps++;

#ifdef ENABLE_DEBUG
            NVLOGI_FMT(NVLOG_PUSCH, "{}: UE group {} -> HetCfg {} (nUeGrps {} nPrb {} nMaxPrb {} nDataSym {} nMaxDataSym {} nRxAnt {} nLayers {})\n", __FUNCTION__, ueGrpIdx, hetCfgIdx, hetCfg.nUeGrps, nPrb, hetCfg.nMaxPrb, nDataSym, hetCfg.nMaxDataSym, drvdUeGrpPrms.nRxAnt, drvdUeGrpPrms.nLayers);
#endif
        }
        // New heterogenous configuration found
        else
        {
            if(nHetCfgs >= CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Exceeded limit on supported heterogneous configurations", __FUNCTION__);
                return CUPHY_STATUS_INTERNAL_ERROR;
            }

            int32_t      newHetCfgIdx            = nHetCfgs++;
            puschRxChEqSoftDemapHetCfg_t& hetCfg = hetCfgs[newHetCfgIdx];
            hetCfg.func                          = launchCfg.kernelNodeParamsDriver.func;
            hetCfg.nMaxPrb                       = nPrb;
            hetCfg.nMaxDataSym                   = nDataSym;

            dynDescrVecCpu[newHetCfgIdx].hetCfgUeGrpMap[hetCfg.nUeGrps] = ueGrpIdx;
            hetCfg.nUeGrps++;

#ifdef ENABLE_DEBUG
            NVLOGI_FMT(NVLOG_PUSCH, "{}: UE group {} -> HetCfg {} (nUeGrps {} nPrb {} nMaxPrb {} nDataSym {} nMaxDataSym {} nRxAnt {} nLayers {})\n", __FUNCTION__, ueGrpIdx, newHetCfgIdx, hetCfg.nUeGrps, nPrb, hetCfg.nMaxPrb, nDataSym, hetCfg.nMaxDataSym, drvdUeGrpPrms.nRxAnt, drvdUeGrpPrms.nLayers);
#endif
        }
    }
    return CUPHY_STATUS_SUCCESS;
} //puschRxChEq::batchEqSoftDemapBluesteinWorkspace

cuphyStatus_t puschRxChEq::batchEqSoftDemapIdft(cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrms,
                                                 uint16_t                           nUeGrps,
                                                 uint16_t                           symbolBitmask,
                                                 uint32_t&                          nHetCfgs,
                                                 puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu)
{
    // Helper to find kernel function
    auto findKernelFunc = [](puschRxChEqSoftDemapHetCfgVec_t const& hetCfgs, CUfunction func, int32_t& hetCfgIdx) {
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

    // Initialize the batch config data structure    
    puschRxChEqSoftDemapHetCfgVec_t& hetCfgs = (symbolBitmask==CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK) ? m_softDemapIdftHetCfgVec[0]:m_softDemapIdftHetCfgVec[1];
    std::fill(hetCfgs.begin(), hetCfgs.end(), puschRxChEqSoftDemapHetCfg_t{nullptr, 0, 0, 0});

#if 0
    // Debug code to initialize hetCfgUeGrpMap with -1s   
    for(puschRxChEqSoftDemapDynDescr_t& dynDescrCpu : dynDescrVecCpu)
    {
        std::fill(std::begin(dynDescrCpu.hetCfgUeGrpMap), std::end(dynDescrCpu.hetCfgUeGrpMap), -1);
    }
#endif    
    

#ifdef ENABLE_DEBUG
    NVLOGI_FMT(NVLOG_PUSCH, "{}: # of UE groups {}", __FUNCTION__, nUeGrps);
#endif

    nHetCfgs = 0;
    for(int32_t ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
    {
        cuphyPuschRxUeGrpPrms_t const& drvdUeGrpPrms = pDrvdUeGrpPrms[ueGrpIdx];

        uint16_t nPrb    = drvdUeGrpPrms.nPrb;
        uint8_t nDataSym = drvdUeGrpPrms.nDataSym;

        // @todo: extend kernelSelectL1 to support a mode which only determines the kernel function
        cuphyPuschRxChEqLaunchCfg_t launchCfg;
        softDemapIdftKernelSelectL1(nDataSym,
                                     nPrb,
				                             nUeGrps,
                                     symbolBitmask,
				                             drvdUeGrpPrms.tInfoEqCoef.elemType,
				                             drvdUeGrpPrms.tInfoDataRx.elemType,
			                               drvdUeGrpPrms.tInfoLLR.elemType,
				                             launchCfg);

        // Check if the heterognous configuration already exists
        int32_t hetCfgIdx = 0;
        findKernelFunc(hetCfgs, launchCfg.kernelNodeParamsDriver.func, hetCfgIdx);

        // If a heterogenous configuration already exists then increment the # of UE groups for that config
        if(-1 != hetCfgIdx)
        {
            puschRxChEqSoftDemapHetCfg_t& hetCfg = hetCfgs[hetCfgIdx];
            if(hetCfg.nUeGrps >= MAX_N_USER_GROUPS_SUPPORTED)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Exceeded limit on supported UE groups", __FUNCTION__);
                return CUPHY_STATUS_INTERNAL_ERROR;
            }

            if(nPrb > hetCfg.nMaxPrb) hetCfg.nMaxPrb             = nPrb;
            if(nDataSym > hetCfg.nMaxDataSym) hetCfg.nMaxDataSym = nDataSym;

            dynDescrVecCpu[hetCfgIdx].hetCfgUeGrpMap[hetCfg.nUeGrps] = ueGrpIdx;
            hetCfg.nUeGrps++;

#ifdef ENABLE_DEBUG
            NVLOGI_FMT(NVLOG_PUSCH, "{}: UE group {} -> HetCfg {} (nUeGrps {} nPrb {} nMaxPrb {} nDataSym {} nMaxDataSym {} nRxAnt {} nLayers {})\n", __FUNCTION__, ueGrpIdx, hetCfgIdx, hetCfg.nUeGrps, nPrb, hetCfg.nMaxPrb, nDataSym, hetCfg.nMaxDataSym, drvdUeGrpPrms.nRxAnt, drvdUeGrpPrms.nLayers);
#endif
        }
        // New heterogenous configuration found
        else
        {
            if(nHetCfgs >= CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Exceeded limit on supported heterogneous configurations", __FUNCTION__);
                return CUPHY_STATUS_INTERNAL_ERROR;
            }

            int32_t      newHetCfgIdx            = nHetCfgs++;
            puschRxChEqSoftDemapHetCfg_t& hetCfg = hetCfgs[newHetCfgIdx];
            hetCfg.func                          = launchCfg.kernelNodeParamsDriver.func;
            hetCfg.nMaxPrb                       = nPrb;
            hetCfg.nMaxDataSym                   = nDataSym;

            dynDescrVecCpu[newHetCfgIdx].hetCfgUeGrpMap[hetCfg.nUeGrps] = ueGrpIdx;
            hetCfg.nUeGrps++;

#ifdef ENABLE_DEBUG
            NVLOGI_FMT(NVLOG_PUSCH, "{}: UE group {} -> HetCfg {} (nUeGrps {} nPrb {} nMaxPrb {} nDataSym {} nMaxDataSym {} nRxAnt {} nLayers {})\n", __FUNCTION__, ueGrpIdx, newHetCfgIdx, hetCfg.nUeGrps, nPrb, hetCfg.nMaxPrb, nDataSym, hetCfg.nMaxDataSym, drvdUeGrpPrms.nRxAnt, drvdUeGrpPrms.nLayers);
#endif
        }
    }
    return CUPHY_STATUS_SUCCESS;
} //puschRxChEq::batchEqSoftDemapIdft

cuphyStatus_t puschRxChEq::batchEqSoftDemapAfterDft(cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrms,
                                                    uint16_t                           nUeGrps,
                                                    uint16_t                           symbolBitmask,
                                                    uint32_t&                          nHetCfgs,
                                                    puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu)
{
    // Helper to find kernel function
    auto findKernelFunc = [](puschRxChEqSoftDemapHetCfgVec_t const& hetCfgs, CUfunction func, int32_t& hetCfgIdx) {
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

    // Initialize the batch config data structure    
    puschRxChEqSoftDemapHetCfgVec_t& hetCfgs = (symbolBitmask==CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK) ? m_softDemapAfterDftHetCfgVec[0]:m_softDemapAfterDftHetCfgVec[1];
    std::fill(hetCfgs.begin(), hetCfgs.end(), puschRxChEqSoftDemapHetCfg_t{nullptr, 0, 0, 0});

#if 0
    // Debug code to initialize hetCfgUeGrpMap with -1s   
    for(puschRxChEqSoftDemapDynDescr_t& dynDescrCpu : dynDescrVecCpu)
    {
        std::fill(std::begin(dynDescrCpu.hetCfgUeGrpMap), std::end(dynDescrCpu.hetCfgUeGrpMap), -1);
    }
#endif    
    

#ifdef ENABLE_DEBUG
    NVLOGI_FMT(NVLOG_PUSCH, "{}: # of UE groups {}", __FUNCTION__, nUeGrps);
#endif

    nHetCfgs = 0;
    for(int32_t ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
    {
        cuphyPuschRxUeGrpPrms_t const& drvdUeGrpPrms = pDrvdUeGrpPrms[ueGrpIdx];

        uint16_t nPrb    = drvdUeGrpPrms.nPrb;
        uint8_t nDataSym = drvdUeGrpPrms.nDataSym;

        // @todo: extend kernelSelectL1 to support a mode which only determines the kernel function
        cuphyPuschRxChEqLaunchCfg_t launchCfg;
        softDemapAfterDftKernelSelectL1(drvdUeGrpPrms.nRxAnt,
                                        drvdUeGrpPrms.nLayers,
                                        drvdUeGrpPrms.nTimeChEsts,
                                        nDataSym,
                                        nPrb,
                                        nUeGrps,
                                        symbolBitmask,
                                        drvdUeGrpPrms.tInfoEqCoef.elemType,
                                        drvdUeGrpPrms.tInfoDataRx.elemType,
                                        drvdUeGrpPrms.tInfoLLR.elemType,
                                        launchCfg);

        // Check if the heterognous configuration already exists
        int32_t hetCfgIdx = 0;
        findKernelFunc(hetCfgs, launchCfg.kernelNodeParamsDriver.func, hetCfgIdx);

        // If a heterogenous configuration already exists then increment the # of UE groups for that config
        if(-1 != hetCfgIdx)
        {
            puschRxChEqSoftDemapHetCfg_t& hetCfg = hetCfgs[hetCfgIdx];
            if(hetCfg.nUeGrps >= MAX_N_USER_GROUPS_SUPPORTED)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Exceeded limit on supported UE groups", __FUNCTION__);
                return CUPHY_STATUS_INTERNAL_ERROR;
            }

            if(nPrb > hetCfg.nMaxPrb) hetCfg.nMaxPrb             = nPrb;
            if(nDataSym > hetCfg.nMaxDataSym) hetCfg.nMaxDataSym = nDataSym;

            dynDescrVecCpu[hetCfgIdx].hetCfgUeGrpMap[hetCfg.nUeGrps] = ueGrpIdx;
            hetCfg.nUeGrps++;

#ifdef ENABLE_DEBUG
            NVLOGI_FMT(NVLOG_PUSCH, "{}: UE group {} -> HetCfg {} (nUeGrps {} nPrb {} nMaxPrb {} nDataSym {} nMaxDataSym {} nRxAnt {} nLayers {})\n", __FUNCTION__, ueGrpIdx, hetCfgIdx, hetCfg.nUeGrps, nPrb, hetCfg.nMaxPrb, nDataSym, hetCfg.nMaxDataSym, drvdUeGrpPrms.nRxAnt, drvdUeGrpPrms.nLayers);
#endif
        }
        // New heterogenous configuration found
        else
        {
            if(nHetCfgs >= CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Exceeded limit on supported heterogneous configurations", __FUNCTION__);
                return CUPHY_STATUS_INTERNAL_ERROR;
            }

            int32_t      newHetCfgIdx            = nHetCfgs++;
            puschRxChEqSoftDemapHetCfg_t& hetCfg = hetCfgs[newHetCfgIdx];
            hetCfg.func                          = launchCfg.kernelNodeParamsDriver.func;
            hetCfg.nMaxPrb                       = nPrb;
            hetCfg.nMaxDataSym                   = nDataSym;

            dynDescrVecCpu[newHetCfgIdx].hetCfgUeGrpMap[hetCfg.nUeGrps] = ueGrpIdx;
            hetCfg.nUeGrps++;

#ifdef ENABLE_DEBUG
            NVLOGI_FMT(NVLOG_PUSCH, "{}: UE group {} -> HetCfg {} (nUeGrps {} nPrb {} nMaxPrb {} nDataSym {} nMaxDataSym {} nRxAnt {} nLayers {})\n", __FUNCTION__, ueGrpIdx, newHetCfgIdx, hetCfg.nUeGrps, nPrb, hetCfg.nMaxPrb, nDataSym, hetCfg.nMaxDataSym, drvdUeGrpPrms.nRxAnt, drvdUeGrpPrms.nLayers);
#endif
        }
    }
    return CUPHY_STATUS_SUCCESS;
} //puschRxChEq::batchEqSoftDemapAfterDft
//#endif

cuphyStatus_t puschRxChEq::setupSoftDemap(cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrmsCpu,
                                          cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrmsGpu,
                                          uint16_t                           nUeGrps,
                                          uint16_t                           nMaxPrb,
                                          uint8_t                            enableCfoCorrection,
                                          uint8_t                            enablePuschTdi,
                                          uint16_t                           symbolBitmask,
                                          bool                               enableCpuToGpuDescrAsyncCpy,
                                          puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu,
                                          void*                              pDynDescrsGpu,
                                          cuphyPuschRxChEqLaunchCfgs_t*      pLaunchCfgs,
                                          cudaStream_t                       strm)
{
    if(!pDrvdUeGrpPrmsCpu || !pDrvdUeGrpPrmsGpu || !pDynDescrsGpu || !pLaunchCfgs) return CUPHY_STATUS_INVALID_ARGUMENT;
    
    if((symbolBitmask!=CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK)&&(symbolBitmask!=CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK))
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Invalid symbolBitmask {}", __FUNCTION__, symbolBitmask);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    cuphyPuschRxChEqLaunchCfgs_t& launchCfgs = *pLaunchCfgs;
    cuphyStatus_t status = batchEqSoftDemap(pDrvdUeGrpPrmsCpu,
                                            nUeGrps,
                                            symbolBitmask,
                                            launchCfgs.nCfgs,
                                            dynDescrVecCpu);
    
    if(CUPHY_STATUS_SUCCESS != status)
    {
        return status;
    }

    puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecGpu = *(static_cast<puschRxChEqSoftDemapDynDescrVec_t*>(pDynDescrsGpu));
    for(uint32_t hetCfgIdx = 0; hetCfgIdx < launchCfgs.nCfgs; ++hetCfgIdx)
    {
        // Skip rest of the setup if there are no UE groups corresponding to hetCfg
        if(symbolBitmask==CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK)
        {
          if(0 == m_softDemapHetCfgVec[0][hetCfgIdx].nUeGrps) continue;
        }
        else
        {
          if(0 == m_softDemapHetCfgVec[1][hetCfgIdx].nUeGrps) continue;
        }
        // Setup descriptor in CPU memory
        puschRxChEqSoftDemapDynDescr_t& dynDescr     = dynDescrVecCpu[hetCfgIdx];
        puschRxChEqSoftDemapHetCfg_t const& hetCfg   =  (symbolBitmask==CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK) ? m_softDemapHetCfgVec[0][hetCfgIdx]:m_softDemapHetCfgVec[1][hetCfgIdx];
        puschRxChEqSoftDemapKernelArgs_t& kernelArgs =  (symbolBitmask==CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK) ? m_softDemapKernelArgsArr[0][hetCfgIdx]:m_softDemapKernelArgsArr[1][hetCfgIdx];

        dynDescr.pDrvdUeGrpPrms = pDrvdUeGrpPrmsGpu;

        // Optional descriptor copy to GPU memory
        if(enableCpuToGpuDescrAsyncCpy)
        {
            // Unchecked return value
            CUDA_CHECK(cudaMemcpyAsync(&dynDescrVecGpu[hetCfgIdx], &dynDescr, sizeof(puschRxChEqSoftDemapDynDescr_t), cudaMemcpyHostToDevice, strm));
        }

        // Select kernel
        cuphyPuschRxChEqLaunchCfg_t& launchCfg = launchCfgs.cfgs[hetCfgIdx];

        int32_t ueGrpIdx = dynDescr.hetCfgUeGrpMap[0];
        cuphyPuschRxUeGrpPrms_t const& drvdUeGrpPrms = pDrvdUeGrpPrmsCpu[ueGrpIdx];
        softDemapKernelSelectL1(drvdUeGrpPrms.nRxAnt,
                                drvdUeGrpPrms.nLayers,
                                drvdUeGrpPrms.nTimeChEsts,
                                hetCfg.nMaxDataSym,
                                hetCfg.nMaxPrb,
                                hetCfg.nUeGrps,
                                symbolBitmask,
                                drvdUeGrpPrms.tInfoEqCoef.elemType,
                                drvdUeGrpPrms.tInfoDataRx.elemType,
                                drvdUeGrpPrms.tInfoLLR.elemType,
                                launchCfg);

        if(hetCfg.func != launchCfg.kernelNodeParamsDriver.func)
        {
           NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Kernel function mismatch", __FUNCTION__);
           return CUPHY_STATUS_INTERNAL_ERROR;
        }          
                                
        kernelArgs.pDynDescr    = &dynDescrVecGpu[hetCfgIdx];
        launchCfg.kernelArgs[0] = &kernelArgs.pStatDescr;
        launchCfg.kernelArgs[1] = &kernelArgs.pDynDescr;

        launchCfg.kernelNodeParamsDriver.kernelParams = &(launchCfg.kernelArgs[0]);
    }
    return CUPHY_STATUS_SUCCESS;
} //setupSoftDemap

cuphyStatus_t puschRxChEq::setupSoftDemapBluesteinWorkspace(cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrmsCpu,
                                          cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrmsGpu,
                                          uint16_t                           nUeGrps,
                                          uint16_t                           nMaxPrb,
                                          uint8_t                            enableCfoCorrection,
                                          uint8_t                            enablePuschTdi,
                                          bool                               enableCpuToGpuDescrAsyncCpy,
                                          puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu,
                                          void*                              pDynDescrsGpu,
                                          cuphyPuschRxChEqLaunchCfgs_t*      pLaunchCfgs,
                                          cudaStream_t                       strm)
{
#ifndef FAST_COMPILE
    if(!pDrvdUeGrpPrmsCpu || !pDrvdUeGrpPrmsGpu || !pDynDescrsGpu || !pLaunchCfgs) return CUPHY_STATUS_INVALID_ARGUMENT;
    
    cuphyPuschRxChEqLaunchCfgs_t& launchCfgs = *pLaunchCfgs;
    cuphyStatus_t status = batchEqSoftDemapBluesteinWorkspace(pDrvdUeGrpPrmsCpu,
                     nUeGrps,
                     launchCfgs.nCfgs,
                     dynDescrVecCpu);

    if(CUPHY_STATUS_SUCCESS != status)
    {
        return status;
    }

    puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecGpu = *(static_cast<puschRxChEqSoftDemapDynDescrVec_t*>(pDynDescrsGpu));
    for(uint32_t hetCfgIdx = 0; hetCfgIdx < launchCfgs.nCfgs; ++hetCfgIdx)
    {
        // Skip rest of the setup if there are no UE groups corresponding to hetCfg
        if(0 == m_softDemapBluesteinWorkspaceHetCfgVec[hetCfgIdx].nUeGrps) continue;

        // Setup descriptor in CPU memory
        puschRxChEqSoftDemapDynDescr_t& dynDescr     = dynDescrVecCpu[hetCfgIdx];
        puschRxChEqSoftDemapHetCfg_t const& hetCfg   = m_softDemapBluesteinWorkspaceHetCfgVec[hetCfgIdx];
        puschRxChEqSoftDemapKernelArgs_t& kernelArgs = m_softDemapBluesteinWorkspaceKernelArgsArr[hetCfgIdx];

        dynDescr.pDrvdUeGrpPrms = pDrvdUeGrpPrmsGpu;

        // Optional descriptor copy to GPU memory
        if(enableCpuToGpuDescrAsyncCpy)
        {
            // Unchecked return value
            CUDA_CHECK(cudaMemcpyAsync(&dynDescrVecGpu[hetCfgIdx], &dynDescr, sizeof(puschRxChEqSoftDemapDynDescr_t), cudaMemcpyHostToDevice, strm));
        }

        // Select kernel
        cuphyPuschRxChEqLaunchCfg_t& launchCfg = launchCfgs.cfgs[hetCfgIdx];

        int32_t ueGrpIdx = dynDescr.hetCfgUeGrpMap[0];
        cuphyPuschRxUeGrpPrms_t const& drvdUeGrpPrms = pDrvdUeGrpPrmsCpu[ueGrpIdx];
        softDemapBluesteinWorkspaceKernelSelectL1(drvdUeGrpPrms.nRxAnt,
                                drvdUeGrpPrms.nLayers,
                                drvdUeGrpPrms.nTimeChEsts,
                                hetCfg.nMaxDataSym,
                                hetCfg.nMaxPrb,
                                hetCfg.nUeGrps,
                                drvdUeGrpPrms.tInfoEqCoef.elemType,
                                drvdUeGrpPrms.tInfoDataRx.elemType,
                                drvdUeGrpPrms.tInfoLLR.elemType,
                                launchCfg);

        if(hetCfg.func != launchCfg.kernelNodeParamsDriver.func)
        {
           NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Kernel function mismatch", __FUNCTION__);
           return CUPHY_STATUS_INTERNAL_ERROR;
        }          
                                
        kernelArgs.pDynDescr    = &dynDescrVecGpu[hetCfgIdx];
        launchCfg.kernelArgs[0] = &kernelArgs.pStatDescr;
        launchCfg.kernelArgs[1] = &kernelArgs.pDynDescr;

        launchCfg.kernelNodeParamsDriver.kernelParams = &(launchCfg.kernelArgs[0]);
    }
#endif
    
    return CUPHY_STATUS_SUCCESS;
} //puschRxChEq::setupSoftDemapBluesteinWorkspace

cuphyStatus_t puschRxChEq::setupSoftDemapIdft(cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrmsCpu,
                                               cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrmsGpu,
                                               uint16_t                           nUeGrps,
                                               uint16_t                           nMaxPrb,
                                               uint8_t                            enableCfoCorrection,
                                               uint8_t                            enablePuschTdi,
                                               uint16_t                           symbolBitmask,
                                               bool                               enableCpuToGpuDescrAsyncCpy,
                                               puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu,
                                               void*                              pDynDescrsGpu,
                                               cuphyPuschRxChEqLaunchCfgs_t*      pLaunchCfgs,
                                               cudaStream_t                       strm)
{
//#ifndef FAST_COMPILE
    if(!pDrvdUeGrpPrmsCpu || !pDrvdUeGrpPrmsGpu || !pDynDescrsGpu || !pLaunchCfgs) return CUPHY_STATUS_INVALID_ARGUMENT;
    
    if((symbolBitmask!=CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK)&&(symbolBitmask!=CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK))
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Invalid symbolBitmask {}", __FUNCTION__, symbolBitmask);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    
    cuphyPuschRxChEqLaunchCfgs_t& launchCfgs = *pLaunchCfgs;
    cuphyStatus_t status = batchEqSoftDemapIdft(pDrvdUeGrpPrmsCpu,
                                                 nUeGrps,
                                                 symbolBitmask,
                                                 launchCfgs.nCfgs,
                                                 dynDescrVecCpu);

    if(CUPHY_STATUS_SUCCESS != status)
    {
        return status;
    }

    puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecGpu = *(static_cast<puschRxChEqSoftDemapDynDescrVec_t*>(pDynDescrsGpu));
    for(uint32_t hetCfgIdx = 0; hetCfgIdx < launchCfgs.nCfgs; ++hetCfgIdx)
    {
        // Skip rest of the setup if there are no UE groups corresponding to hetCfg
        if(symbolBitmask==CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK)
        {
            if(0 == m_softDemapIdftHetCfgVec[0][hetCfgIdx].nUeGrps) continue;
        }
        else
        {
            if(0 == m_softDemapIdftHetCfgVec[1][hetCfgIdx].nUeGrps) continue;
        }

        // Setup descriptor in CPU memory
        puschRxChEqSoftDemapDynDescr_t& dynDescr     = dynDescrVecCpu[hetCfgIdx];
        puschRxChEqSoftDemapHetCfg_t const& hetCfg   = (symbolBitmask==CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK) ? m_softDemapIdftHetCfgVec[0][hetCfgIdx]:m_softDemapIdftHetCfgVec[1][hetCfgIdx];
        puschRxChEqSoftDemapKernelArgs_t& kernelArgs = (symbolBitmask==CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK) ? m_softDemapIdftKernelArgsArr[0][hetCfgIdx]:m_softDemapIdftKernelArgsArr[1][hetCfgIdx];

        dynDescr.pDrvdUeGrpPrms = pDrvdUeGrpPrmsGpu;

        // Optional descriptor copy to GPU memory
        if(enableCpuToGpuDescrAsyncCpy)
        {
            // Unchecked return value
            CUDA_CHECK(cudaMemcpyAsync(&dynDescrVecGpu[hetCfgIdx], &dynDescr, sizeof(puschRxChEqSoftDemapDynDescr_t), cudaMemcpyHostToDevice, strm));
        }

        // Select kernel
        cuphyPuschRxChEqLaunchCfg_t& launchCfg = launchCfgs.cfgs[hetCfgIdx];

        int32_t ueGrpIdx = dynDescr.hetCfgUeGrpMap[0];
        cuphyPuschRxUeGrpPrms_t const& drvdUeGrpPrms = pDrvdUeGrpPrmsCpu[ueGrpIdx];
        softDemapIdftKernelSelectL1(hetCfg.nMaxDataSym,
                                     hetCfg.nMaxPrb,
				                             hetCfg.nUeGrps,
                                     symbolBitmask,
			                               drvdUeGrpPrms.tInfoEqCoef.elemType,
				                             drvdUeGrpPrms.tInfoDataRx.elemType,
				                             drvdUeGrpPrms.tInfoLLR.elemType,
				                             launchCfg);

        if(hetCfg.func != launchCfg.kernelNodeParamsDriver.func)
        {
           NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Kernel function mismatch", __FUNCTION__);
           return CUPHY_STATUS_INTERNAL_ERROR;
        }          
                                
        kernelArgs.pDynDescr    = &dynDescrVecGpu[hetCfgIdx];
        launchCfg.kernelArgs[0] = &kernelArgs.pStatDescr;
        launchCfg.kernelArgs[1] = &kernelArgs.pDynDescr;

        launchCfg.kernelNodeParamsDriver.kernelParams = &(launchCfg.kernelArgs[0]);
    }
//#endif
    return CUPHY_STATUS_SUCCESS;
} //puschRxChEq::setupSoftDemapIdft

cuphyStatus_t puschRxChEq::setupSoftDemapAfterDft(cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrmsCpu,
                                          cuphyPuschRxUeGrpPrms_t*           pDrvdUeGrpPrmsGpu,
                                          uint16_t                           nUeGrps,
                                          uint16_t                           nMaxPrb,
                                          uint8_t                            enableCfoCorrection,
                                          uint8_t                            enablePuschTdi,
                                          uint16_t                           symbolBitmask,
                                          bool                               enableCpuToGpuDescrAsyncCpy,
                                          puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecCpu,
                                          void*                              pDynDescrsGpu,
                                          cuphyPuschRxChEqLaunchCfgs_t*      pLaunchCfgs,
                                          cudaStream_t                       strm)
{
//#ifndef FAST_COMPILE
    if(!pDrvdUeGrpPrmsCpu || !pDrvdUeGrpPrmsGpu || !pDynDescrsGpu || !pLaunchCfgs) return CUPHY_STATUS_INVALID_ARGUMENT;
    
    if((symbolBitmask!=CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK)&&(symbolBitmask!=CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK))
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Invalid symbolBitmask {}", __FUNCTION__, symbolBitmask);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    cuphyPuschRxChEqLaunchCfgs_t& launchCfgs = *pLaunchCfgs;
    cuphyStatus_t status = batchEqSoftDemapAfterDft(pDrvdUeGrpPrmsCpu,
                                                    nUeGrps,
                                                    symbolBitmask,
                                                    launchCfgs.nCfgs,
                                                    dynDescrVecCpu);

    if(CUPHY_STATUS_SUCCESS != status)
    {
        return status;
    }

    puschRxChEqSoftDemapDynDescrVec_t& dynDescrVecGpu = *(static_cast<puschRxChEqSoftDemapDynDescrVec_t*>(pDynDescrsGpu));
    for(uint32_t hetCfgIdx = 0; hetCfgIdx < launchCfgs.nCfgs; ++hetCfgIdx)
    {
        // Skip rest of the setup if there are no UE groups corresponding to hetCfg
        if(symbolBitmask==CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK)
        {
            if(0 == m_softDemapAfterDftHetCfgVec[0][hetCfgIdx].nUeGrps) continue;
        }
        else
        {
            if(0 == m_softDemapAfterDftHetCfgVec[1][hetCfgIdx].nUeGrps) continue;
        }

        // Setup descriptor in CPU memory
        puschRxChEqSoftDemapDynDescr_t& dynDescr     = dynDescrVecCpu[hetCfgIdx];
        puschRxChEqSoftDemapHetCfg_t const& hetCfg   = (symbolBitmask==CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK) ? m_softDemapAfterDftHetCfgVec[0][hetCfgIdx]:m_softDemapAfterDftHetCfgVec[1][hetCfgIdx];
        puschRxChEqSoftDemapKernelArgs_t& kernelArgs = (symbolBitmask==CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK) ? m_softDemapAfterDftKernelArgsArr[0][hetCfgIdx]:m_softDemapAfterDftKernelArgsArr[1][hetCfgIdx];

        dynDescr.pDrvdUeGrpPrms = pDrvdUeGrpPrmsGpu;

        // Optional descriptor copy to GPU memory
        if(enableCpuToGpuDescrAsyncCpy)
        {
            // Unchecked return value
            CUDA_CHECK(cudaMemcpyAsync(&dynDescrVecGpu[hetCfgIdx], &dynDescr, sizeof(puschRxChEqSoftDemapDynDescr_t), cudaMemcpyHostToDevice, strm));
        }

        // Select kernel
        cuphyPuschRxChEqLaunchCfg_t& launchCfg = launchCfgs.cfgs[hetCfgIdx];

        int32_t ueGrpIdx = dynDescr.hetCfgUeGrpMap[0];
        cuphyPuschRxUeGrpPrms_t const& drvdUeGrpPrms = pDrvdUeGrpPrmsCpu[ueGrpIdx];
        softDemapAfterDftKernelSelectL1(drvdUeGrpPrms.nRxAnt,
                                        drvdUeGrpPrms.nLayers,
                                        drvdUeGrpPrms.nTimeChEsts,
                                        hetCfg.nMaxDataSym,
                                        hetCfg.nMaxPrb,
                                        hetCfg.nUeGrps,
                                        symbolBitmask,
                                        drvdUeGrpPrms.tInfoEqCoef.elemType,
                                        drvdUeGrpPrms.tInfoDataRx.elemType,
                                        drvdUeGrpPrms.tInfoLLR.elemType,
                                        launchCfg);

        if(hetCfg.func != launchCfg.kernelNodeParamsDriver.func)
        {
           NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Kernel function mismatch", __FUNCTION__);
           return CUPHY_STATUS_INTERNAL_ERROR;
        }          
                                
        kernelArgs.pDynDescr    = &dynDescrVecGpu[hetCfgIdx];
        launchCfg.kernelArgs[0] = &kernelArgs.pStatDescr;
        launchCfg.kernelArgs[1] = &kernelArgs.pDynDescr;

        launchCfg.kernelNodeParamsDriver.kernelParams = &(launchCfg.kernelArgs[0]);
    }
//#endif
    return CUPHY_STATUS_SUCCESS;
} //puschRxChEq::setupSoftDemapAfterDft




} // namespace channel_eq
