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
#include <functional>
#include <vector>
#include <bitset>
#include "cuComplex.h"
#include <cooperative_groups.h>
#include "srs_ch_est.hpp"
#include "type_convert.hpp"

using namespace cooperative_groups;

namespace srs_ch_est
{
// #define ENABLE_COEFS_IN_CONST_MEM
// #define ENABLE_PROFILING
// #define ENABLE_DEBUG

// 2 PRBs per channel estimate, per comb, per cyclic shift)
// Affects interpolation filter dimension
static constexpr uint32_t N_PRB_INP_PER_SRS_CH_EST = 2;

// PRBs used jointly for SRS channel estimation (affects interpolation filter dimension)
// interpolation filter dimension = N_SRS_CH_EST_PER_PRB_GRP x (N_PRB_INP_PER_PRB_GRP*N_TONES_PER_COMB)
// where N_SRS_TONES_PER_COMB = CUPHY_N_TONES_PER_PRB/N_SRS_COMBS
static constexpr uint32_t N_PRB_INP_PER_PRB_GRP = 4; // 4, 8

#ifdef ENABLE_COEFS_IN_CONST_MEM
// Note:  N_COMB_TONES_PER_PRB_GRP * N_SRS_COMBS = CUPHY_N_TONES_PER_PRB
// e.g. N_PRB_INP_PER_SRS_CH_EST = 2
// if N_SRS_COMBS = 4: N_COMB_TONES_PER_PRB_GRP = CUPHY_N_TONES_PER_PRB/N_SRS_COMBS = 3
// if N_SRS_COMBS = 2: N_COMB_TONES_PER_PRB_GRP = CUPHY_N_TONES_PER_PRB/N_SRS_COMBS = 6
// In both cases (N_COMB_TONES_PER_PRB_GRP * N_SRS_COMBS) == CUPHY_N_TONES_PER_PRB
static constexpr size_t FREQ_INTERP_COEF_LEN = N_PRB_INP_PER_SRS_CH_EST * CUPHY_N_TONES_PER_PRB * N_PRB_INP_PER_PRB_GRP;

// 2 filters (1st for 2 comb and 2nd for 4 comb) of dimension: N_PRB_INP_PER_SRS_CH_EST x (N_COMB_TONES_PER_PRB_GRP x N_SRS_COMBS) x N_PRB_INP_PER_PRB_GRP
__constant__ float GPU_CONST_MEM_FREQ_INTERP_COEFS[FREQ_INTERP_COEF_LEN * 2];
#endif

template <typename TElem>
struct tensor_ref
{
    TElem*     pAddr;
    const int* strides;

    CUDA_BOTH
    tensor_ref(void* pAddr, const int* pStrides) :
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
    // clang-format off
    CUDA_BOTH TElem&       operator()(int i0)                               { return *(pAddr + offset(i0));             }
    CUDA_BOTH TElem&       operator()(int i0, int i1)                       { return *(pAddr + offset(i0, i1));         }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2)               { return *(pAddr + offset(i0, i1, i2));     }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2, int i3)       { return *(pAddr + offset(i0, i1, i2, i3)); }

    CUDA_BOTH const TElem& operator()(int i0) const                         { return *(pAddr + offset(i0));             }
    CUDA_BOTH const TElem& operator()(int i0, int i1) const                 { return *(pAddr + offset(i0, i1));         }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2) const         { return *(pAddr + offset(i0, i1, i2));     }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3) const { return *(pAddr + offset(i0, i1, i2, i3)); }
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
    CUDA_BOTH T& operator[](int idx) { return data[idx]; }
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

template <typename T, int L, int M, int N, int O>
struct block_4D
{
    T         data[L * M * N * O];
    CUDA_BOTH T& operator()(int l, int m, int n, int o) { return data[(((o * N + n) * M) + m) * L + l]; }
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

// Partial specialization of block_4D to use shared memory pointers
template <typename T, int L, int M, int N, int O>
struct block_4D<T*, L, M, N, O>
{
    CUDA_BOTH block_4D(T* pData) :
        m_pData(pData){};
    block_4D()                    = delete;
    block_4D(block_4D const& blk) = delete;
    CUDA_BOTH block_4D& operator  =(block_4D const& block) { m_pData = block.m_pData; };
    ~block_4D()                   = default;

    CUDA_BOTH T&               operator()(int l, int m, int n, int o) { return m_pData[(((o * N + n) * M) + m) * L + l]; }
    static constexpr CUDA_BOTH size_t num_elem() { return L * M * N * O; }

private:
    T* m_pData = nullptr;
};

// clang-format off
template <typename T> CUDA_BOTH_INLINE constexpr T     cuGet(uint32_t);
template<>            CUDA_BOTH_INLINE constexpr float cuGet(uint32_t x) { return(float(x)); }

template <typename T> CUDA_BOTH_INLINE T         cuGet(int);
template<>            CUDA_BOTH_INLINE float     cuGet(int x) { return(float(x)); }
template<>            CUDA_BOTH_INLINE cuComplex cuGet(int x) { return(make_cuComplex(float(x), 0.0f)); }

template <typename T> CUDA_BOTH_INLINE T         cuGet(float);
template<>            CUDA_BOTH_INLINE float     cuGet(float x) { return x; }
template<>            CUDA_BOTH_INLINE cuComplex cuGet(float x) { return(make_cuComplex(x, 0.0f)); }

template <typename T> CUDA_BOTH_INLINE T         cuGet(float,float);
template <>           CUDA_BOTH_INLINE cuComplex cuGet<cuComplex>(float x, float y) { return make_cuComplex(x,y); }

static CUDA_BOTH_INLINE cuComplex operator+=(cuComplex &x, cuComplex y)       { x = cuCaddf(x, y); return x; };
static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, float y)             { return(make_cuComplex(cuCrealf(x)*y, cuCimagf(x)*y)); }
static CUDA_BOTH_INLINE cuComplex operator*=(cuComplex &x, const cuComplex y) { x = cuCmulf(x, y); return x; };

static CUDA_BOTH_INLINE float cuReal(cuComplex x) { return(cuCrealf(x)); }
static CUDA_BOTH_INLINE float cuImag(cuComplex x) { return(cuCimagf(x)); }

#if 0
static CUDA_BOTH_INLINE float cuReal(cuComplex x) { return(cuCrealf(x)); }
static CUDA_BOTH_INLINE float cuImag(cuComplex x) { return(cuCimagf(x)); }
static CUDA_BOTH_INLINE cuComplex cuConj(cuComplex x) { return(cuConjf(x)); }
static CUDA_BOTH_INLINE cuComplex cuAdd(cuComplex x, cuComplex y) { return(cuCaddf(x, y)); }
static CUDA_BOTH_INLINE cuComplex cuMul(cuComplex x, cuComplex y) { return(cuCmulf(x, y)); }
static CUDA_BOTH_INLINE cuComplex cuDiv(cuComplex x, cuComplex y) { return(cuCdivf(x, y)); }
static CUDA_BOTH_INLINE cuComplex operator+(cuComplex x, cuComplex y) { return(cuCaddf(x, y)); }
static CUDA_BOTH_INLINE cuComplex operator-(cuComplex x, cuComplex y) { return(cuCsubf(x, y)); }
static CUDA_BOTH_INLINE cuComplex operator+=(cuComplex &x, cuComplex y) { x = cuCaddf(x, y); return x; };
static CUDA_BOTH_INLINE cuComplex operator-=(cuComplex &x, cuComplex y) { x = cuCsubf(x, y); return x; };
static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, cuComplex y) { return(cuCmulf(x, y)); }
static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, int y) { return(make_cuComplex(cuCrealf(x)*float(y), cuCimagf(x)*float(y))); }
static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, float y) { return(make_cuComplex(cuCrealf(x)*y, cuCimagf(x)*y)); }
static CUDA_BOTH_INLINE cuComplex operator/(cuComplex x, float y) { return(make_cuComplex(cuCrealf(x)/y, cuCimagf(x)/y)); }
static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, cuComplex y) { return(cuCmulf(x, y)); }
static CUDA_BOTH_INLINE cuComplex operator*(const cuComplex x, const cuComplex y) { return(cuCmulf(x, y)); }
static CUDA_BOTH_INLINE cuComplex operator*=(cuComplex &x, float y) { x = make_cuComplex(cuCrealf(x)*y, cuCimagf(x)*y); return x; }
static CUDA_BOTH_INLINE cuComplex operator*=(cuComplex &x, cuComplex y) { x = cuCmulf(x, y); return x; };
static CUDA_BOTH_INLINE cuComplex cuFma(cuComplex x, cuComplex y, cuComplex a) { return cuCfmaf(x,y,a); }// a = (x*y) + a;

// cuda_fp16.hpp
//__device__ __forceinline__ __half operator*(const __half &lh, const __half &rh) { return __hmul(lh, rh); }  
// __device__ __forceinline__ __half2& operator*=(__half2 &lh, const __half2 &rh) { lh = __hmul2(lh, rh); return lh; } 

// static CUDA_BOTH_INLINE __half2 cuConj(__half2 &hc) { __half2 t; t.x = hc.x; t.y = -hc.y; return t; }
// static CUDA_BOTH_INLINE __half2 cuGet(int x) {  __half2 t; t.x = __half(x); t.y = __float2half(0.0f); return t; } 
#endif
// clang-format on

// SRS cyclic shifts always repeat every 4 tones
static constexpr uint8_t N_SRS_CYC_SHIFT_TONE_SPACING = 4;
__constant__ uint8_t SRS_CYC_SHIFT_ROT_IDXS[][N_SRS_CYC_SHIFT_TONE_SPACING] =
    {{0, 0, 0, 0}, // start offset for nSrsCycShifts 1 = (nSrsCycShifts - 1) = 0
     {0, 0, 0, 0}, // start offset for nSrsCycShifts 2 = (nSrsCycShifts - 1) = 1
     {0, 1, 0, 1},
     {0, 0, 0, 0}, // start offset for nSrsCycShifts 4 = (nSrsCycShifts - 1) = 3
     {0, 3, 1, 2},
     {0, 1, 0, 1},
     {0, 2, 1, 3}};

// Phase rotation in multiples of 90
template <typename TComplexCompute>
__device__ __forceinline__ void rotatePhase90(uint8_t rotIdx, const TComplexCompute& in, TComplexCompute& out)
{
    switch(rotIdx)
    {
    case 0: out = in; break;                                               // 0
    case 1: out = cuGet<TComplexCompute>(-cuReal(in), -cuImag(in)); break; // 180
    case 2: out = cuGet<TComplexCompute>(-cuImag(in), cuReal(in)); break;  // -90 (270)
    case 3: out = cuGet<TComplexCompute>(cuImag(in), -cuReal(in)); break;  // +90
    default: out = in; break;
    }
}

// #define ENABLE_DEBUG
template <typename TStorage,
          typename TDataRx,
          typename TCompute,
          uint32_t N_PRB_INP_PER_THRD_BLK_ITER, // # of PRBs processed by 1 iteration of a thread block
          uint32_t N_PRB_INP_PER_PRB_GRP,       // # of PRBs per processing group (4, 8), a group of PRBs on which SRS channel estimation processing is applied (e.g. interpolation)
          uint32_t N_PRB_INP_PER_SRS_CH_EST,    // expected to be 2 (2 PRBs per channel estimate, per comb, per cyclic shift)
          uint32_t N_SRS_COMBS,                 // # of SRS combs (1, 2 or 4)
          uint32_t N_SRS_CYC_SHIFTS>            // # of SRS cyclic shifts (1, 2 or 4)
static __global__ void
windowedSrsChEstKernel(srsChEstKernelArgs_t args)
{
    //--------------------------------------------------------------------------------------------------------
    // Setup local parameters based on descriptor
    srsChEstStatDescr_t& statDescr = *(args.pStatDescr);
    srsChEstDynDescr_t&  dynDescr  = *(args.pDynDescr);

#if 0    
    uint16_t nBSAnts         = dynDescr.nBSAnts; // gridDim.y;
    uint8_t  nLayers         = dynDescr.nLayers;
    uint16_t nPrb            = dynDescr.nPrb;
    uint8_t  nCombs          = dynDescr.nCombs;
    uint8_t  nCycShifts      = dynDescr.nCycShifts;
    uint8_t  nSrsSyms        = dynDescr.nSrsSyms; // gridDim.z;
#endif
    uint16_t scsKHz          = dynDescr.scsKHz;
    uint16_t nZc             = dynDescr.nZc;
    uint8_t  zcSeqNum        = dynDescr.zcSeqNum;
    TCompute delaySpreadSecs = cuGet<TCompute>(dynDescr.delaySpreadSecs);

    uint8_t* pSrsSymPos = dynDescr.srsSymPos;

#ifdef ENABLE_COEFS_IN_CONST_MEM
    constexpr size_t FREQ_INTERP_COEF_OFFSET = (2 == N_SRS_COMBS) ? 0 : FREQ_INTERP_COEF_LEN;
    TCompute*        pFreqInterpCoefs        = &GPU_CONST_MEM_FREQ_INTERP_COEFS[FREQ_INTERP_COEF_OFFSET];
#endif
#ifdef ENABLE_DEBUG
    if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
    {
        printf("windowedSrsChEstKernel: nIter %d nBSAnts %d nLayers %d nPrb %d scsKHz %d nCyclicShifts %d nCombs %d nSrsSyms %d nZc %d zcSeqNum %d delaySpreadSecs %e\n",
               dynDescr.nIter,
               nBSAnts,
               nLayers,
               nPrb,
               scsKHz,
               nCycShifts,
               nCombs,
               nSrsSyms,
               nZc,
               zcSeqNum,
               delaySpreadSecs);
        for(uint32_t i = 0; i < nSrsSyms; ++i)
        {
            printf("windowedSrsChEstKernel: srsSymPos[%d] = %d\n", i, dynDescr.srsSymPos[i]);
        }
    }

    if((0 != blockIdx.x) || (0 != blockIdx.y) || (0 != blockIdx.z)) return;
#endif
    // if((10 != blockIdx.x) || (17 != blockIdx.y) || (1 != blockIdx.z)) return;

    //--------------------------------------------------------------------------------------------------------
    typedef typename complex_from_scalar<TCompute>::type TComplexCompute;
    typedef typename complex_from_scalar<TDataRx>::type  TComplexDataRx;
    typedef typename complex_from_scalar<TStorage>::type TComplexStorage;

    // clang-format off
    tensor_ref<const TStorage>       tFreqInterpCoefs(statDescr.tPrmFreqInterpCoefs.pAddr, statDescr.tPrmFreqInterpCoefs.strides); // (N_SRS_CH_EST_PER_PRB_GRP, N_SRS_TONES_PER_PRB_GRP, N_SRS_COMBS) 
    tensor_ref<const TComplexDataRx> tDataRx         (dynDescr.tPrmDataRx.pAddr          , dynDescr.tPrmDataRx.strides);           // (NF, ND, N_BS_ANTS)
    tensor_ref<TComplexStorage>      tHEst           (dynDescr.tPrmHEst.pAddr            , dynDescr.tPrmHEst.strides);             // (N_SRS_CH_EST_IN_FREQ, N_BS_ANTS, N_LAYERS) or (N_BS_ANTS, N_SRS_CH_EST_IN_FREQ, N_LAYERS)
    tensor_ref<TComplexStorage>      tDbg            (dynDescr.tPrmDbg.pAddr             , dynDescr.tPrmDbg.strides);
    // clang-format on

    //--------------------------------------------------------------------------------------------------------
    // Dimensions and indices

    // Shared memory allocations:
    // 1. Phase shift/unshift sequence
    // 2. Interpolation filter (real)
    // 3. SRS tones/Channel estimates

    // Input counts (PRB/tone) per iteration of thread block
    constexpr uint32_t N_TONES_PROC_PER_THRD_BLK_ITER = N_PRB_INP_PER_THRD_BLK_ITER * CUPHY_N_TONES_PER_PRB;
    constexpr uint32_t N_PRB_GRPS_PER_THRD_BLK_ITER   = N_PRB_INP_PER_THRD_BLK_ITER / N_PRB_INP_PER_PRB_GRP;
    constexpr uint32_t N_COMB_TONES_PER_THRD_BLK_ITER = N_PRB_INP_PER_THRD_BLK_ITER * CUPHY_N_TONES_PER_PRB / N_SRS_COMBS;

    // Output (SRS channel estimate) counts per iteration of thread block
    constexpr uint32_t N_SRS_CH_EST_IN_FREQ_PER_THRD_BLK_ITER = N_PRB_INP_PER_THRD_BLK_ITER / N_PRB_INP_PER_SRS_CH_EST;
    // constexpr uint32_t N_SRS_CH_EST_PER_COMB      = N_SRS_CH_EST_IN_FREQ * N_SRS_CYC_SHIFTS;
    constexpr uint32_t N_SRS_CH_EST_IN_FREQ_AND_CYC_SHIFT_PER_THRD_BLK_ITER = N_SRS_CH_EST_IN_FREQ_PER_THRD_BLK_ITER * N_SRS_CYC_SHIFTS;
    constexpr uint32_t N_SRS_CH_EST_PER_THRD_BLK_ITER                       = N_SRS_CH_EST_IN_FREQ_AND_CYC_SHIFT_PER_THRD_BLK_ITER * N_SRS_COMBS;

    // Per PRB group counts
    constexpr uint32_t N_TONES_PER_PRB_GRP      = N_PRB_INP_PER_PRB_GRP * CUPHY_N_TONES_PER_PRB;
    constexpr uint32_t N_COMB_TONES_PER_PRB_GRP = N_TONES_PER_PRB_GRP / N_SRS_COMBS;
    constexpr uint32_t N_SRS_CH_EST_PER_PRB_GRP = N_PRB_INP_PER_PRB_GRP / N_PRB_INP_PER_SRS_CH_EST; // this is the number of SRS channel estimates per PRB group per cyclic shift per comb

    constexpr uint32_t N_SMEM_ELEMS_SHIFT_SEQ     = N_COMB_TONES_PER_THRD_BLK_ITER;
    constexpr uint32_t N_SMEM_ELEMS_UNSHIFT_SEQ   = N_SRS_CH_EST_IN_FREQ_PER_THRD_BLK_ITER * N_SRS_COMBS;
    constexpr uint32_t N_SMEM_ELEMS_PHASE_ROT_SEQ = N_SMEM_ELEMS_SHIFT_SEQ > N_SMEM_ELEMS_UNSHIFT_SEQ ? N_SMEM_ELEMS_SHIFT_SEQ : N_SMEM_ELEMS_UNSHIFT_SEQ; // using the longer of the two shift sequences

    constexpr uint32_t N_SMEM_ELEMS_SRS_TONES = N_TONES_PROC_PER_THRD_BLK_ITER * N_SRS_CYC_SHIFTS;

    // Sanity checks
    static_assert((N_TONES_PROC_PER_THRD_BLK_ITER * N_SRS_CYC_SHIFTS == N_COMB_TONES_PER_PRB_GRP * N_PRB_GRPS_PER_THRD_BLK_ITER * N_SRS_CYC_SHIFTS * N_SRS_COMBS), "Input SRS tone count mismatch");
    static_assert((N_SRS_CH_EST_PER_THRD_BLK_ITER == N_PRB_GRPS_PER_THRD_BLK_ITER * N_SRS_CH_EST_PER_PRB_GRP * N_SRS_CYC_SHIFTS * N_SRS_COMBS), "SRS channel estimate count mismatch");

    constexpr uint32_t N_SMEM_ELEMS = N_SMEM_ELEMS_PHASE_ROT_SEQ + N_SMEM_ELEMS_SRS_TONES;
    __shared__ TComplexCompute smemBlk[N_SMEM_ELEMS];

    // @todo: since the number of input SRS tones read per iteration are capped by N_THRDS, shShiftSeq can be capped to the same size as well
    constexpr uint32_t                                                              SMEM_START_OFFSET_PHASE_ROT_SEQ = 0;
    block_1D<TComplexCompute*, N_COMB_TONES_PER_THRD_BLK_ITER>                      shShiftSeq(&smemBlk[SMEM_START_OFFSET_PHASE_ROT_SEQ]);
    block_2D<TComplexCompute*, N_SRS_CH_EST_IN_FREQ_PER_THRD_BLK_ITER, N_SRS_COMBS> shUnShiftSeq(&smemBlk[SMEM_START_OFFSET_PHASE_ROT_SEQ]);

    constexpr uint32_t SMEM_START_OFFSET_SRS_TONES = SMEM_START_OFFSET_PHASE_ROT_SEQ + N_SMEM_ELEMS_PHASE_ROT_SEQ; // shPhaseRotSeq.num_elem();
    block_4D<TComplexCompute*, N_COMB_TONES_PER_PRB_GRP, N_PRB_GRPS_PER_THRD_BLK_ITER, N_SRS_CYC_SHIFTS, N_SRS_COMBS>
        shSrsTones(&smemBlk[SMEM_START_OFFSET_SRS_TONES]);

    thread_block const& thisThrdBlk = this_thread_block();
    const uint32_t      N_THRDS     = thisThrdBlk.size();
    const uint32_t      THRD_IDX    = thisThrdBlk.thread_rank();

    // BS antenna being processed by this thread block
    const uint32_t BS_ANT_IDX = blockIdx.y;

    // SRS symbol index
    const uint32_t SRS_SYM_IDX = blockIdx.z;

    //--------------------------------------------------------------------------------------------------------
    // Phase rotation sequence: delay shift (to center channel response) + ZC sequence removal
    constexpr TCompute DELAY_SHIFT   = 0.4f; // using 0.4 instead of 0.5 for margin (so that LOS path is within and not on interp filter edge)
    constexpr TCompute TWO_PI        = (2.0f * M_PI);
    TCompute           phaseDlyShift = TWO_PI * DELAY_SHIFT * delaySpreadSecs * (scsKHz * 1000);

    for(uint32_t iterIdx = 0; iterIdx < dynDescr.nIter; ++iterIdx)
    {
        // PRB cluster being processed by this thread block
        const uint32_t START_ABS_PRB_IDX        = (blockIdx.x + iterIdx) * N_PRB_INP_PER_THRD_BLK_ITER;
        const uint32_t START_ABS_TONE_IDX       = START_ABS_PRB_IDX * CUPHY_N_TONES_PER_PRB;
        const uint32_t START_ABS_SRS_CH_EST_IDX = START_ABS_PRB_IDX / N_PRB_INP_PER_SRS_CH_EST;

        constexpr uint32_t N_COMB_TONES_PER_PRB    = CUPHY_N_TONES_PER_PRB / N_SRS_COMBS;
        const uint32_t     START_ABS_COMB_TONE_IDX = START_ABS_PRB_IDX * N_COMB_TONES_PER_PRB;

        // The phase rotation sequence is of length N_COMB_TONES_PER_THRD_BLK_ITER
        if(THRD_IDX < N_COMB_TONES_PER_THRD_BLK_ITER)
        {
            const uint32_t IN_COMB_TONE_IDX     = THRD_IDX;
            const uint32_t IN_ABS_COMB_TONE_IDX = START_ABS_COMB_TONE_IDX + IN_COMB_TONE_IDX;
            uint16_t       zcIdx                = IN_ABS_COMB_TONE_IDX % nZc;
            TCompute       phaseZcCycShift      = TWO_PI * (cuGet<TCompute>((zcSeqNum * zcIdx * (zcIdx + 1))) / cuGet<TCompute>(2 * nZc));
            TCompute       phase                = ((phaseDlyShift * N_SRS_COMBS * IN_ABS_COMB_TONE_IDX) + phaseZcCycShift);
            shShiftSeq(IN_COMB_TONE_IDX)        = cuGet<TComplexCompute>(cosf(phase), sinf(phase));

            // shift sequence and phase
            // tDbg(IN_ABS_COMB_TONE_IDX) = type_convert<TComplexStorage>(shShiftSeq(IN_COMB_TONE_IDX)); // dim: nPrb * CUPHY_N_TONES_PER_PRB / nCombs
            // tDbg(IN_ABS_COMB_TONE_IDX) = type_convert<TComplexStorage>(cuGet<TComplexCompute>(phase, 0.0f)); // dim: nPrb * CUPHY_N_TONES_PER_PRB / nCombs
#ifdef ENABE_DEBUG
            printf("ShiftSeq[%d]: %f+j%f zcIdx %d phaseDlyShift %f phaseZcCycShift %f phase %f\n", IN_ABS_COMB_TONE_IDX, cuReal(shShiftSeq(IN_COMB_TONE_IDX)), cuImag(shShiftSeq(IN_COMB_TONE_IDX)), zcIdx, phaseDlyShift, phaseZcCycShift, phase);
#endif
        }

        // Wait for phase rotation sequence compute to complete
        thisThrdBlk.sync();

        //--------------------------------------------------------------------------------------------------------
        // SRS tone extraction, comb seperation, delay centering, ZC sequence removal, cyclic shift separation

        // Loop to read all input tones to be processed
        constexpr uint32_t N_TONES_TO_RD       = N_TONES_PROC_PER_THRD_BLK_ITER;
        const uint32_t     N_TONES_RD_PER_ITER = (N_TONES_TO_RD > N_THRDS) ? N_THRDS : N_TONES_TO_RD;
        const uint32_t     N_ITER_TO_RD_TONES  = div_round_up(N_TONES_TO_RD, N_TONES_RD_PER_ITER);

        for(uint32_t i = 0; i < N_ITER_TO_RD_TONES; ++i)
        {
            uint32_t inToneIdx    = i * N_TONES_RD_PER_ITER + THRD_IDX;
            uint32_t inAbsToneIdx = START_ABS_TONE_IDX + inToneIdx;

            if(inToneIdx < N_TONES_TO_RD)
            {
                // Contiguous load from global memory, scattered store into shared memory (tones are binned to
                // respective combs before store)
                uint32_t combIdx     = inToneIdx % N_SRS_COMBS;
                uint32_t combToneIdx = (inToneIdx / N_SRS_COMBS) % N_COMB_TONES_PER_PRB_GRP;
                uint32_t prbGrpIdx   = inToneIdx / N_TONES_PER_PRB_GRP;
                uint32_t cycShiftIdx = 0;

                uint32_t shiftSeqIdx = inToneIdx / N_SRS_COMBS; // shift sequence value common across tones

                // @todo:check performance by hoisting this load before phase rotation sequence compute
                TComplexCompute srsTone = type_convert<TComplexCompute>(tDataRx(inAbsToneIdx, pSrsSymPos[SRS_SYM_IDX], BS_ANT_IDX));

#ifdef ENABLE_DEBUG
                printf("PrePhaseShift: Input Tone[%d][%d[%d] %f+j%f\n", inAbsToneIdx, pSrsSymPos[SRS_SYM_IDX], BS_ANT_IDX, cuReal(srsTone), cuImag(srsTone));
#endif

                // delay shift and ZC removal
                srsTone *= shShiftSeq(shiftSeqIdx);

                // Seprate tones per comb
                shSrsTones(combToneIdx, prbGrpIdx, cycShiftIdx, combIdx) = srsTone;

#ifdef ENABLE_DEBUG
                printf("InterpIn: Tone[%d][%d][%d][%d] %f+j%f\n", combToneIdx, prbGrpIdx, cycShiftIdx, combIdx, cuReal(srsTone), cuImag(srsTone));
#endif

                // Interp in
                // tDbg(combToneIdx, (START_ABS_PRB_IDX / N_PRB_INP_PER_PRB_GRP) + prbGrpIdx, cycShiftIdx, combIdx) =
                //     type_convert<TComplexStorage>(shSrsTones(combToneIdx, prbGrpIdx, cycShiftIdx, combIdx));

                // Separate cyclic shift (SRS antenna ports) by applying phase rotations on SRS tones of each comb
                for(cycShiftIdx = 1; cycShiftIdx < N_SRS_CYC_SHIFTS; ++cycShiftIdx)
                {
                    uint8_t rotIdx =
                        SRS_CYC_SHIFT_ROT_IDXS[(N_SRS_CYC_SHIFTS - 1) + cycShiftIdx][combToneIdx % N_SRS_CYC_SHIFT_TONE_SPACING];

                    TComplexCompute srsCycShiftTone{};
                    rotatePhase90(rotIdx, srsTone, srsCycShiftTone);
                    shSrsTones(combToneIdx, prbGrpIdx, cycShiftIdx, combIdx) = srsCycShiftTone;

#ifdef ENABLE_DEBUG
                    printf("InterpIn: Tone[%d][%d][%d][%d] %f+j%f\n", combToneIdx, prbGrpIdx, cycShiftIdx, combIdx, cuReal(srsCycShiftTone), cuImag(srsCycShiftTone));
#endif

                    // Interp in
                    // tDbg(combToneIdx, (START_ABS_PRB_IDX / N_PRB_INP_PER_PRB_GRP) + prbGrpIdx, cycShiftIdx, combIdx) =
                    //     type_convert<TComplexStorage>(shSrsTones(combToneIdx, prbGrpIdx, cycShiftIdx, combIdx));
                }
            }
        }

        //--------------------------------------------------------------------------------------------------------
        // Phase rotation sequence: delay unshift
        constexpr uint32_t SRS_CH_EST_TONE_START_OFFSET = CUPHY_N_TONES_PER_PRB;
        constexpr uint32_t SRS_CH_EST_TONE_STRIDE       = 2 * CUPHY_N_TONES_PER_PRB;

        uint32_t unShiftSeqChEstIdx = THRD_IDX % N_SRS_CH_EST_IN_FREQ_PER_THRD_BLK_ITER;
        uint32_t unShiftSeqCombIdx  = (THRD_IDX / N_SRS_CH_EST_IN_FREQ_PER_THRD_BLK_ITER) % N_SRS_COMBS;
        if(THRD_IDX < (N_SRS_CH_EST_IN_FREQ_PER_THRD_BLK_ITER * N_SRS_COMBS))
        {
            int32_t  chEstAbsToneIdx = START_ABS_TONE_IDX + SRS_CH_EST_TONE_START_OFFSET + (unShiftSeqChEstIdx * SRS_CH_EST_TONE_STRIDE) - unShiftSeqCombIdx;
            TCompute phase           = -phaseDlyShift * chEstAbsToneIdx;

            shUnShiftSeq(unShiftSeqChEstIdx, unShiftSeqCombIdx) = cuGet<TComplexCompute>(cosf(phase), sinf(phase));

            //tDbg(START_ABS_SRS_CH_EST_IDX + unShiftSeqChEstIdx, unShiftSeqCombIdx) =
            //    type_convert<TComplexStorage>(shUnShiftSeq(unShiftSeqChEstIdx, unShiftSeqCombIdx));

#ifdef ENABLE_DEBUG
            printf("UnshiftSeq[%d][%d] %f+j%f, chEstAbsToneIdx %d\n", unShiftSeqChEstIdx, unShiftSeqCombIdx, cuReal(shUnShiftSeq(unShiftSeqChEstIdx, unShiftSeqCombIdx)), cuImag(shUnShiftSeq(unShiftSeqChEstIdx, unShiftSeqCombIdx)), chEstAbsToneIdx);
#endif
        }
        // Wait for phase rotation sequence compute to complete
        thisThrdBlk.sync();

        //--------------------------------------------------------------------------------------------------------
        // Interpolate (matrix-vector multiply)
        // Each thread produces 1 SRS channel estimate via interpolation
        // constexpr uint32_t N_SRS_CH_EST_PER_COMB      = N_SRS_CH_EST_PER_CYC_SHIFT_COMB * N_SRS_CYC_SHIFTS;
        // Reshape the threads into 4 sub-groups e.g. 2 SRS_CH_EST x 4 PRB_GRPS x 4 CYCLIC_SHIFTS x 4 COMBS
        uint32_t chEstIdx    = THRD_IDX % N_SRS_CH_EST_PER_PRB_GRP;
        uint32_t prbGrpIdx   = (THRD_IDX / N_SRS_CH_EST_PER_PRB_GRP) % N_PRB_GRPS_PER_THRD_BLK_ITER;
        uint32_t cycShiftIdx = (THRD_IDX / (N_SRS_CH_EST_PER_PRB_GRP * N_PRB_GRPS_PER_THRD_BLK_ITER)) % N_SRS_CYC_SHIFTS;
        uint32_t combIdx     = (THRD_IDX / (N_SRS_CH_EST_PER_PRB_GRP * N_PRB_GRPS_PER_THRD_BLK_ITER * N_SRS_CYC_SHIFTS)) % N_SRS_COMBS;

        // H = W x Y: (N_SRS_CH_EST_PER_PRB_GRP x N_COMB_TONES_PER_PRB_GRP) x (N_COMB_TONES_PER_PRB_GRP x 1)
        // Each thread selects one row of W and computes N_COMB_TONES_PER_PRB_GRP length inner product to produce one interpolated
        // tone of H
        TComplexCompute hEst{0};
        for(uint32_t toneIdx = 0; toneIdx < N_COMB_TONES_PER_PRB_GRP; ++toneIdx)
        {
#ifdef ENABLE_COEFS_IN_CONST_MEM
            TCompute interpCoef = type_convert<TCompute>(pFreqInterpCoefs[((combIdx * N_COMB_TONES_PER_PRB_GRP + toneIdx) * N_SRS_CH_EST_PER_PRB_GRP + chEstIdx)]);
#else
            TCompute interpCoef = type_convert<TCompute>(tFreqInterpCoefs(chEstIdx, toneIdx, combIdx));
#endif
            hEst += (shSrsTones(toneIdx, prbGrpIdx, cycShiftIdx, combIdx) * interpCoef);
#if 0        
        if((0 == chEstIdx) && (2 == combIdx) && (3 == cycShiftIdx) && ((1 == prbGrpIdx) || (2 == prbGrpIdx)))
        {
            TComplexCompute tone = shSrsTones(toneIdx, prbGrpIdx, cycShiftIdx, combIdx);
            printf("InterpProc: chEstIdx %d prbGrpIdx %d tone[%d] %f+j%f interpCoef %f innerProd %f+j%f\n", chEstIdx, prbGrpIdx, toneIdx, tone.x, tone.y, interpCoef, innerProd.x, innerProd.y);
        }
#endif
        }

        // Interp out
        // tDbg(chEstIdx, (START_ABS_PRB_IDX / N_PRB_INP_PER_PRB_GRP) + prbGrpIdx, cycShiftIdx, combIdx) = type_convert<TComplexStorage>(hEst);
#ifdef ENABLE_DEBUG
        // if((0 == chEstIdx) && (2 == combIdx) && (3 == cycShiftIdx) && ((1 == prbGrpIdx) || (2 == prbGrpIdx)))
        printf("InterpOut[%d][%d][%d][%d] = %f+j%f\n", chEstIdx, prbGrpIdx, cycShiftIdx, combIdx, innerProd.x, innerProd.y);
#endif

        hEst *= shUnShiftSeq((prbGrpIdx * N_SRS_CH_EST_PER_PRB_GRP) + chEstIdx, combIdx);

        // tDbg(chEstIdx, (START_ABS_PRB_IDX / N_PRB_INP_PER_PRB_GRP) + prbGrpIdx, cycShiftIdx, combIdx) =
        //    type_convert<TComplexStorage>(shHEst(chEstIdx, prbGrpIdx, cycShiftIdx, combIdx));

#ifdef ENABLE_DEBUG
        printf("HEst[%d][%d][%d][%d] %f+j%f\n", BS_ANT_IDX, START_ABS_SRS_CH_EST_IDX + (prbGrpIdx * N_SRS_CH_EST_PER_PRB_GRP) + chEstIdx, cycShiftIdx, (SRS_SYM_IDX * N_SRS_COMBS) + combIdx, cuReal(hEst), cuImag(hEst));
#endif

        //--------------------------------------------------------------------------------------------------------
        // Store
#if 1 // higher write efficiency: (N_SRS_CH_EST_IN_FREQ, N_BS_ANTS, N_LAYERS) o
        tHEst(START_ABS_SRS_CH_EST_IDX + (prbGrpIdx * N_SRS_CH_EST_PER_PRB_GRP) + chEstIdx, BS_ANT_IDX, cycShiftIdx, (SRS_SYM_IDX * N_SRS_COMBS) + combIdx) =
            type_convert<TComplexStorage>(hEst);
#else // BFW friendly writes: (N_BS_ANTS, N_SRS_CH_EST_IN_FREQ, N_LAYERS)
        tHEst(BS_ANT_IDX, START_ABS_SRS_CH_EST_IDX + (prbGrpIdx * N_SRS_CH_EST_PER_PRB_GRP) + chEstIdx, cycShiftIdx, (SRS_SYM_IDX * N_SRS_COMBS) + combIdx) =
            type_convert<TComplexStorage>(hEst);
#endif
    }
}

template <uint32_t N_PRB_INP_PER_THRD_BLK_ITER, // # of PRBs processed by 1 iteration of a thread block
          uint32_t N_PRB_INP_PER_PRB_GRP,       // # of PRBs per processing group (4, 8), a group of PRBs on which SRS channel estimation processing is applied (e.g. interpolation)
          uint32_t N_PRB_INP_PER_SRS_CH_EST,    // expected to be 2 (2 PRBs per channel estimate, per comb, per cyclic shift)
          uint32_t N_SRS_COMBS,                 // # of SRS combs (1, 2 or 4)
          uint32_t N_SRS_CYC_SHIFTS>            // # of SRS cyclic shifts (1, 2 or 4)
void
srsChEst::computeKernelLaunchGeo(uint8_t                    nIter,
                                 uint16_t                   nBSAnts,
                                 uint16_t                   nPrb,
                                 uint8_t                    nSrsSyms, // # of SRS symbols (1, 2, or 4)
                                 srsChEstKernelLaunchGeo_t& launchGeo)
{
    static_assert((N_PRB_INP_PER_SRS_CH_EST == 2), "Revisit the block dimensions (thread counts) since it assumes 2 SRS channel estimate per PRB, per comb, per cyclic shift");
    static_assert(((N_PRB_INP_PER_THRD_BLK_ITER % N_PRB_INP_PER_PRB_GRP) == 0), "Number of PRBs procesed by a thread block iteration should be a multiple of number of PRBs per PRB group");
    // static_assert(((N_PRB_INP_PER_PRB_GRP * CUPHY_N_TONES_PER_PRB % N_SRS_COMBS) == 0), "Number of tones per PRB group needs to be a multiple of number of SRS combs");

    constexpr uint32_t N_SRS_CH_EST_PER_THRD_BLK_ITER = (N_PRB_INP_PER_THRD_BLK_ITER / N_PRB_INP_PER_SRS_CH_EST) * N_SRS_COMBS * N_SRS_CYC_SHIFTS;

    // Thread block sized so that N_PRB_INP_PER_THRD_BLK_ITER, N_SRS_COMBS and N_SRS_CYC_SHIFTS are processed in parallel
    constexpr uint32_t N_THRDS_PER_THRD_BLK = N_SRS_CH_EST_PER_THRD_BLK_ITER;

    // Thread block sized so that N_PRB_INP_PER_THRD_BLK_ITER N_SRS_COMBS are processed in parallel with
    // N_SRS_CYC_SHIFTS processed sequentially
    // constexpr uint32_t N_THRDS_PER_THRD_BLK = N_SRS_CH_EST_PER_THRD_BLK_ITER / N_SRS_CYC_SHIFTS;

    // In each iteration N_PRB_INP_PER_THRD_BLK_ITER PRBs are processed to produce N_SRS_CH_EST_PER_THRD_BLK_ITER channel estimates
    const uint32_t N_SRS_CH_EST_PER_THRD_BLK = N_SRS_CH_EST_PER_THRD_BLK_ITER * nIter;

    const uint32_t N_SRS_CH_EST_PER_BS_ANT = (nPrb / N_PRB_INP_PER_SRS_CH_EST) * N_SRS_COMBS * N_SRS_CYC_SHIFTS;
    const uint32_t N_THRD_BLKS_PER_BS_ANT  = N_SRS_CH_EST_PER_BS_ANT / N_SRS_CH_EST_PER_THRD_BLK;

    launchGeo.gridDim  = dim3(N_THRD_BLKS_PER_BS_ANT, nBSAnts, nSrsSyms);
    launchGeo.blockDim = dim3(N_THRDS_PER_THRD_BLK);

#ifdef ENABLE_DEBUG
    printf("blockDim (%d,%d,%d), gridDim (%d,%d,%d) nIter %d\n", launchGeo.blockDim.x, launchGeo.blockDim.y, launchGeo.blockDim.z, launchGeo.gridDim.x, launchGeo.gridDim.y, launchGeo.gridDim.z, nIter);
#endif
}

template <typename TStorage,
          typename TDataRx,
          typename TCompute,
          uint32_t N_PRB_INP_PER_THRD_BLK_ITER, // # of PRBs processed by 1 iteration of a thread block
          uint32_t N_PRB_INP_PER_PRB_GRP,       // # of PRBs per processing group (4, 8), a group of PRBs on which SRS channel estimation processing is applied (e.g. interpolation)
          uint32_t N_PRB_INP_PER_SRS_CH_EST,    // expected to be 2 (2 PRBs per channel estimate, per comb, per cyclic shift)
          uint32_t N_SRS_COMBS,                 // # of SRS combs (1, 2 or 4)
          uint32_t N_SRS_CYC_SHIFTS>            // # of SRS cyclic shifts (1, 2 or 4)
void
windowedSrsChEstKernelLauncher(srsChEstKernelLaunchPrms& kernelLaunchPrms,
                               cudaStream_t&             strm)
{
    srsChEstKernelLaunchGeo_t& launchGeo = kernelLaunchPrms.launchGeo;
    windowedSrsChEstKernel<TStorage,
                           TDataRx,
                           TCompute,
                           N_PRB_INP_PER_THRD_BLK_ITER,
                           N_PRB_INP_PER_PRB_GRP,
                           N_PRB_INP_PER_SRS_CH_EST,
                           N_SRS_COMBS,
                           N_SRS_CYC_SHIFTS><<<launchGeo.gridDim, launchGeo.blockDim, 0, strm>>>(kernelLaunchPrms.args);
}

template <typename TStorage,
          typename TDataRx,
          typename TCompute,
          uint32_t N_PRB_INP_PER_THRD_BLK_ITER, // # of PRBs processed by 1 iteration of a thread block
          uint32_t N_PRB_INP_PER_PRB_GRP,       // # of PRBs per processing group (4, 8), a group of PRBs on which SRS channel estimation processing is applied (e.g. interpolation)
          uint32_t N_PRB_INP_PER_SRS_CH_EST,    // expected to be 2 (2 PRBs per channel estimate, per comb, per cyclic shift)
          uint32_t N_SRS_COMBS,                 // # of SRS combs (1, 2 or 4)
          uint32_t N_SRS_CYC_SHIFTS>            // # of SRS cyclic shifts (1, 2 or 4)
void
srsChEst::windowedSrsChEst(uint8_t                     nIter,
                           uint16_t                    nBSAnts,
                           uint16_t                    nPrb,
                           uint8_t                     nSrsSyms,
                           srsChEstKernelLaunchPrms_t& launchPrms)
{
    launchPrms.launcher = windowedSrsChEstKernelLauncher<TStorage,
                                                         TDataRx,
                                                         TCompute,
                                                         N_PRB_INP_PER_THRD_BLK_ITER,
                                                         N_PRB_INP_PER_PRB_GRP,
                                                         N_PRB_INP_PER_SRS_CH_EST,
                                                         N_SRS_COMBS,
                                                         N_SRS_CYC_SHIFTS>;

    launchPrms.kernelFunc = reinterpret_cast<void*>(windowedSrsChEstKernel<TStorage,
                                                                           TDataRx,
                                                                           TCompute,
                                                                           N_PRB_INP_PER_THRD_BLK_ITER,
                                                                           N_PRB_INP_PER_PRB_GRP,
                                                                           N_PRB_INP_PER_SRS_CH_EST,
                                                                           N_SRS_COMBS,
                                                                           N_SRS_CYC_SHIFTS>);

    computeKernelLaunchGeo<N_PRB_INP_PER_THRD_BLK_ITER,
                           N_PRB_INP_PER_PRB_GRP,
                           N_PRB_INP_PER_SRS_CH_EST,
                           N_SRS_COMBS,
                           N_SRS_CYC_SHIFTS>(nIter, nBSAnts, nPrb, nSrsSyms, launchPrms.launchGeo);
}

template <typename TStorage, typename TDataRx, typename TCompute>
void srsChEst::kernelSelectL0(bool                        enIter,
                              srsChEstDynDescr_t&         dynPrms,
                              srsChEstKernelLaunchPrms_t& launchPrms)
{
    bool noKernelFound = false;

    // # of PRB groups processed per thread block iter sized so that # of PRBs per thread block is 16
    // 16 is the largest multiple of 272
    constexpr uint32_t N_PRB_GRP_PER_THRD_BLK_ITER = 4;

    constexpr uint32_t N_PRB_INP_PER_THRD_BLK_ITER = N_PRB_GRP_PER_THRD_BLK_ITER * N_PRB_INP_PER_PRB_GRP; // 16                                                                                                                                                                                                                                                                                              -    constexpr uint32_t N_PRB_INP_PER_THRD_BLK = N_PRB_INP_PER_THRD_BLK_ITER * N_THRD_BLK_ITER;

    // # thread block iterations for SRS processing
    dynPrms.nIter = enIter ? (dynPrms.nPrb / N_PRB_INP_PER_THRD_BLK_ITER) : 1;

    // Check below ensures the parameters match the dimensions assumed in the kernel. Among others it ensures
    // that nPrb is divisible by N_PRB_INP_PER_THRD_BLK_ITER
    if((64 == dynPrms.nBSAnts) && (32 == dynPrms.nLayers) && (4 == dynPrms.nCombs) &&
       (4 == dynPrms.nCycShifts) && (2 == dynPrms.nSrsSyms) && (0 == (dynPrms.nPrb % N_PRB_INP_PER_THRD_BLK_ITER)))
    {
        constexpr uint32_t N_SRS_COMBS      = 4;
        constexpr uint32_t N_SRS_CYC_SHIFTS = 4;

        windowedSrsChEst<TStorage,
                         TDataRx,
                         TCompute,
                         N_PRB_INP_PER_THRD_BLK_ITER,
                         N_PRB_INP_PER_PRB_GRP,
                         N_PRB_INP_PER_SRS_CH_EST,
                         N_SRS_COMBS,
                         N_SRS_CYC_SHIFTS>(dynPrms.nIter, dynPrms.nBSAnts, dynPrms.nPrb, dynPrms.nSrsSyms, launchPrms);
    }
    else
    {
        noKernelFound = true;
    }

    if(noKernelFound)
    {
        printf("SRS Channel est: No kernel available to launch with requested configuration: nBSAnts %d nLayers %d "
               "nPrb %d scs (kHz) %d nCycShifts %d nCombs %d nSrsSyms %d nZc %d zcSeqNum %d delaySpread (s) %e \n",
               dynPrms.nBSAnts,
               dynPrms.nLayers,
               dynPrms.nPrb,
               dynPrms.scsKHz,
               dynPrms.nCycShifts,
               dynPrms.nCombs,
               dynPrms.nSrsSyms,
               dynPrms.nZc,
               dynPrms.zcSeqNum,
               dynPrms.delaySpreadSecs);

        launchPrms.launcher = srsChEstKernelLauncher_t();
    }
}

void srsChEst::kernelSelectL1(bool                        enIter,
                              srsChEstDynDescr_t&         dynPrms,
                              tensor_pair&                tDataRx,
                              tensor_pair&                tH,
                              srsChEstKernelLaunchPrms_t& launchPrms)
{
    using TCompute = float;
    if(CUPHY_C_32F == tH.first.get().type())
    {
        using TStorage = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
        if(CUPHY_C_32F == tDataRx.first.get().type())
        {
            using TDataRx = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
            kernelSelectL0<TStorage, TDataRx, TCompute>(enIter, dynPrms, launchPrms);
        }
        else if(CUPHY_C_16F == tDataRx.first.get().type())
        {
            using TDataRx = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
            kernelSelectL0<TStorage, TDataRx, TCompute>(enIter, dynPrms, launchPrms);
        }
        else
        {
            printf("SRS Channel Est: No kernel available to launch with requested data type\n");
        }
    }
    else if((CUPHY_C_16F == tH.first.get().type()) && (CUPHY_C_16F == tDataRx.first.get().type()))
    {
        using TStorage = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        using TDataRx  = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
        kernelSelectL0<TStorage, TDataRx, TCompute>(enIter, dynPrms, launchPrms);
    }
    else
    {
        printf("SRS Channel Est: No kernel available to launch with requested data type\n");
    }
}

void srsChEst::copyTensorPair(tensor_pair& tensorPair, void** ppDstAddr, int* pDstStrideArr)
{
    *ppDstAddr       = tensorPair.second;
    auto copyEntries = [](int* pDst, const int* pSrc, size_t nEntries) {
        for(uint32_t i = 0; i < nEntries; ++i) pDst[i] = pSrc[i];
    };

    const tensor_layout_any& tensorLayout = tensorPair.first.get().layout();
    copyEntries(pDstStrideArr, tensorLayout.strides.begin(), tensorLayout.rank());
}

void srsChEst::init(tensor_pair&         tFreqInterpCoefs,
                    bool                 enableCpuToGpuDescrAsyncCpy,
                    srsChEstStatDescr_t& statDescrCpu,
                    void*                pStatDescrGpu,
                    cudaStream_t         strm)
{
    copyTensorPair(tFreqInterpCoefs, &statDescrCpu.tPrmFreqInterpCoefs.pAddr, statDescrCpu.tPrmFreqInterpCoefs.strides);

    for(auto& kernelLaunchPrms : m_kernelLaunchPrmsVec)
    {
        kernelLaunchPrms.args.pStatDescr = static_cast<srsChEstStatDescr_t*>(pStatDescrGpu);
    }

    if(enableCpuToGpuDescrAsyncCpy)
    {
        CUDA_CHECK(cudaMemcpyAsync(pStatDescrGpu, &statDescrCpu, sizeof(srsChEstStatDescr_t), cudaMemcpyHostToDevice, strm));

#ifdef ENABLE_COEFS_IN_CONST_MEM
        // assumes tFreqInterpCoefs.first.get().layout().rank() ==  FREQ_INTERP_COEF_LEN
        uint32_t nFreqInterpCoefBytes = FREQ_INTERP_COEF_LEN * sizeof(GPU_CONST_MEM_FREQ_INTERP_COEFS[0]);
        // uint32_t comb2FilterOffset = 0;
        uint32_t comb4FilterByteOffset = nFreqInterpCoefBytes;

        CUDA_CHECK(cudaMemcpyToSymbolAsync(GPU_CONST_MEM_FREQ_INTERP_COEFS, tFreqInterpCoefs.second, nFreqInterpCoefBytes, comb4FilterByteOffset, cudaMemcpyHostToDevice, strm));
#endif
    }
}

void srsChEst::getDescrInfo(size_t& statDescrSizeBytes, size_t& statDescrAlignBytes, size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes)
{
    statDescrSizeBytes  = sizeof(srsChEstStatDescr_t);
    statDescrAlignBytes = alignof(srsChEstStatDescr_t);

    dynDescrSizeBytes  = sizeof(srsChEstDynDescrVec_t);
    dynDescrAlignBytes = alignof(srsChEstDynDescrVec_t);
}

void srsChEst::setup(cuphySrsChEstDynPrms_t const* pDynPrms,
                     tensor_pair&                  tDataRx,
                     tensor_pair&                  tHEst,
                     tensor_pair&                  tDbg,
                     bool                          enableCpuToGpuDescrAsyncCpy,
                     srsChEstDynDescrVec_t&        dynDescrVecCpu,
                     void*                         pDynDescrsGpu,
                     cudaStream_t                  strm)
{
    srsChEstDynDescr_t* pDynDescrVecGpu = static_cast<srsChEstDynDescr_t*>(pDynDescrsGpu);
    uint32_t            hetCfgIdx       = 0;
#if SRS_GRAPH_SUPPORT
    bool updateGraphNodePrms = (nullptr != pGraphNodePrms);
#endif

    for(auto& kernelLaunchPrms : m_kernelLaunchPrmsVec)
    {
        // Setup descriptor in CPU memory
        srsChEstDynDescr_t& dynDescr = dynDescrVecCpu[hetCfgIdx];

        dynDescr.nBSAnts         = pDynPrms->nBSAnts;
        dynDescr.nLayers         = pDynPrms->nLayers;
        dynDescr.nPrb            = pDynPrms->nPrb;
        dynDescr.scsKHz          = pDynPrms->scsKHz;
        dynDescr.nCycShifts      = pDynPrms->nCycShifts;
        dynDescr.nCombs          = pDynPrms->nCombs;
        dynDescr.nZc             = pDynPrms->nZc;
        dynDescr.zcSeqNum        = pDynPrms->zcSeqNum;
        dynDescr.delaySpreadSecs = pDynPrms->delaySpreadSecs;

        auto                       symLocBmsk = pDynPrms->srsSymLocBmsk;
        const decltype(symLocBmsk) ONE        = 1;
#if 1
        dynDescr.nSrsSyms = std::min(static_cast<int>(N_MAX_SRS_SYMS), __builtin_popcount(symLocBmsk));
#else
        dynDescr.nSrsSyms = std::min(static_cast<size_t>(N_MAX_SRS_SYMS),
                                     std::bitset<CHAR_BIT * sizeof(symLocBmsk)>(symLocBmsk).count());
#endif
        for(uint32_t i = 0; i < dynDescr.nSrsSyms; ++i)
        {
            // Using ctz for 0-based bit position (ffs provides 1-based)
            dynDescr.srsSymPos[i] = __builtin_ctz(symLocBmsk);
            symLocBmsk &= ~(ONE << dynDescr.srsSymPos[i]);
        }

        // clang-format off
        copyTensorPair(tDataRx, &dynDescr.tPrmDataRx.pAddr, dynDescr.tPrmDataRx.strides);
        copyTensorPair(tHEst  , &dynDescr.tPrmHEst.pAddr  , dynDescr.tPrmHEst.strides  );
        copyTensorPair(tDbg   , &dynDescr.tPrmDbg.pAddr   , dynDescr.tPrmDbg.strides   );
        // clang-format on

        kernelLaunchPrms.args.pDynDescr = &pDynDescrVecGpu[hetCfgIdx];
        // Select kernel, setup launch geometry etc
        bool enIter = (0 != pDynPrms->enIter) ? true : false;
        kernelSelectL1(enIter,
                       dynDescr,
                       tDataRx,
                       tHEst,
                       kernelLaunchPrms);

        // Optional descriptor copy to GPU memory
        if(enableCpuToGpuDescrAsyncCpy)
        {
            CUDA_CHECK(cudaMemcpyAsync(&pDynDescrVecGpu[hetCfgIdx], &dynDescr, sizeof(srsChEstDynDescr_t), cudaMemcpyHostToDevice, strm));
        }

#if SRS_GRAPH_SUPPORT
        // Optional graph node parameter update
        if(updateGraphNodePrms)
        {
            cudaKernelNodeParams graphNodePrms{0};
            kernelLaunchPrms.kernelArgs = std::array<void*, SRS_CH_EST_KERNEL_N_ARGS>{static_cast<void*>(&kernelLaunchPrms.args)};
            // memset(&graphNodePrms, 0, sizeof(graphNodePrms));

            graphNodePrms.func           = kernelLaunchPrms.kernelFunc;
            graphNodePrms.gridDim        = kernelLaunchPrms.launchGeo.gridDim;
            graphNodePrms.blockDim       = kernelLaunchPrms.launchGeo.blockDim;
            graphNodePrms.kernelParams   = kernelLaunchPrms.kernelArgs.data();
            graphNodePrms.sharedMemBytes = 0;
            graphNodePrms.extra          = nullptr;

            *(pGraphNodePrms->pSuccess) = static_cast<cuphyBool_t>(true);
            // update
            if(nullptr != pGraphNodePrms->pUpdatePrms)
            {
                cuphyPuschRxFeUpdateGraphNodePrms_t& updateGraphNodePrms = pGraphNodePrms->pUpdatePrms[hetCfgIdx];
                CUDA_CHECK(cudaGraphExecKernelNodeSetParams(*updateGraphNodePrms.pGraphExec,
                                                            *updateGraphNodePrms.pNode,
                                                            &graphNodePrms));
            }
            // create
            else if(nullptr != pGraphNodePrms->pCreatePrms)
            {
                cuphyPuschRxFeCreateGraphNodePrms_t& createGraphNodePrms = pGraphNodePrms->pCreatePrms[hetCfgIdx];
                CUDA_CHECK(cudaGraphAddKernelNode(createGraphNodePrms.pNode,
                                                  *createGraphNodePrms.pGraph,
                                                  createGraphNodePrms.pDependencies,
                                                  createGraphNodePrms.nDependencies,
                                                  &graphNodePrms));
            }
            else
            {
                *(pGraphNodePrms->pSuccess) = static_cast<cuphyBool_t>(false);
            }
        }
#endif
        hetCfgIdx++;
    }
}

void srsChEst::run(srsChEstStrmVec_t& strmVec)
{
    uint32_t hetCfgIdx = 0;
    for(auto& kernelLaunchPrms : m_kernelLaunchPrmsVec)
    {
        kernelLaunchPrms.launcher(kernelLaunchPrms, strmVec[hetCfgIdx]);
        hetCfgIdx++;
    }
}

} // namespace srs_ch_est
