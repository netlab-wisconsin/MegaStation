/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "channel_est.hpp"
#include "type_convert.hpp"
#include <vector>

namespace channel_est
{
template <typename TElem, int NDim>
struct tensor_ref
{
    TElem* addr;
    int    dim[NDim];
    int    strides[NDim];
    tensor_ref(tensor_pair& tp) :
        addr(static_cast<TElem*>(tp.second))
    {
        const tensor_layout_any& layout = tp.first.get().layout();
#pragma unroll
        for(int i = 0; i < NDim; ++i)
        {
            dim[i]     = (layout.rank() > i) ? layout.dimensions[i] : 1;
            strides[i] = (layout.rank() > i) ? layout.strides[i] : 0;
        }
    }
    tensor_ref(const_tensor_pair& tp) :
        addr(static_cast<TElem*>(tp.second))
    {
        const tensor_layout_any& layout = tp.first.get().layout();
#pragma unroll
        for(int i = 0; i < NDim; ++i)
        {
            dim[i]     = (layout.rank() > i) ? layout.dimensions[i] : 1;
            strides[i] = (layout.rank() > i) ? layout.strides[i] : 0;
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
    CUDA_BOTH long offset(int i0, int i1, int i2, int i3, int i4) const
    {
        return (strides[0] * (long)i0) + (strides[1] * (long)i1) + (strides[2] * (long)i2) + (strides[3] * (long)i3) + (strides[4] * (long)i4);
    };    
    // clang-format off
    CUDA_BOTH TElem&       operator()(int i0)                               { return *(addr + offset(i0));         }
    CUDA_BOTH const TElem& operator()(int i0) const                         { return *(addr + offset(i0));         }
    CUDA_BOTH TElem&       operator()(int i0, int i1)                       { return *(addr + offset(i0, i1));     }
    CUDA_BOTH const TElem& operator()(int i0, int i1) const                 { return *(addr + offset(i0, i1));     }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2)               { return *(addr + offset(i0, i1, i2)); }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2) const         { return *(addr + offset(i0, i1, i2)); }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2, int i3)       { return *(addr + offset(i0, i1, i2, i3)); }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3) const { return *(addr + offset(i0, i1, i2, i3)); }
    CUDA_BOTH TElem&       operator()(int i0, int i1, int i2, int i3, int i4)       { return *(addr + offset(i0, i1, i2, i3, i4)); }
    CUDA_BOTH const TElem& operator()(int i0, int i1, int i2, int i3, int i4) const { return *(addr + offset(i0, i1, i2, i3, i4)); }    
    // clang-format on
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

template <typename Tdst, typename Tsrc>
CUDA_INLINE void load_1D(Tdst* dst, const Tsrc* src, int num)
{
    for(int i = threadIdx.x; i < num; i += blockDim.x)
    {
        dst[i] = type_convert<Tdst>(src[i]);
    }
}

// Assumes column major, destination tightly packed
template <typename Tdst, typename Tsrc>
CUDA_INLINE void load_2D_transpose(Tdst* dst, const Tsrc* src, int srcM, int srcN, int srcStride)
{
    for(int i = threadIdx.x; i < (srcM * srcN); i += blockDim.x)
    {
        const int dstM = srcN;
        //const int dstN = srcM;
        const int dstCol = i / dstM; // Slow!
        const int dstRow = i % dstM;
        const int srcCol = dstRow;
        const int srcRow = dstCol;
        //printf("threadIdx.x = %i, dstRow = %i, dstCol = %i, srcRow = %i, srcCol = %i, dstOffset = %i, srcOffset = %i\n",
        //       threadIdx.x, dstRow, dstCol, srcRow, srcCol, (dstCol * dstM) + dstRow, (srcM * srcCol) + srcRow);
        dst[(dstCol * dstM) + dstRow] = type_convert<Tdst>(src[(srcStride * srcCol) + srcRow]);
    }
}

template <typename T, int M, int N>
struct block
{
    T         data[M * N];
    CUDA_BOTH T& operator[](int idx) { return data[idx]; }
};

template <typename T>
__device__ void print_block(const T* t, const char* name, const char* fmt, int M, int N = 1)
{
    if((0 == threadIdx.x) && (0 == threadIdx.y))
    {
        printf("%s:\n", name);
        for(int i = 0; i < M; ++i)
        {
            for(int j = 0; j < N; ++j)
            {
                printf(fmt, t[j * M + i]);
            }
            printf("\n");
        }
    }
}

template <typename T>
struct printer;
template <>
struct printer<struct __half2>
{
    static CUDA_INLINE void print(const struct __half2& chalf)
    {
        cuComplex c = type_convert<cuComplex>(chalf);
        printf("(%.4f, %.4f) ", c.x, c.y);
    }
};
// clang-format off
template <> struct printer<cuComplex> { static CUDA_INLINE void print(const cuComplex& c) { printf("(%.4f, %.4f) ", c.x, c.y); } };
template <> struct printer<float>     { static CUDA_INLINE void print(const float&     f) { printf("%.4f ", f); } };
template <> struct printer<int16_t>   { static CUDA_INLINE void print(const int16_t&   i) { printf("%i ", static_cast<int>(i)); } };
// clang-format on

template <typename T>
__device__ void dump_shared_mem(const char* desc, T* shmem, int N)
{
    __syncthreads();
    if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z))
    {
        printf("%s:\n", desc);
        for(int i = 0; i < N; ++i)
        {
            printf("%4i: ", i);
            printer<T>::print(shmem[i]);
            printf("\n");
        }
    }
}

template <typename TStorage,
          typename TCompute,
          int BLOCK_TILE_SIZE_X, // BLOCK DIMENSIONS IN WARP TILES (X = 8, Y = 4) or (X = 4, Y = 4)
          int BLOCK_TILE_SIZE_Y,
          int WARP_TILE_SIZE_X, // Must equal NTP (4)
          int WARP_TILE_SIZE_Y, // (8)
          int NFP,
          int NT>
__global__ void mmse_1D_time_frequency_kernel(tensor_ref<typename complex_from_scalar<TStorage>::type, 4>       tHinterp,
                                              tensor_ref<const typename complex_from_scalar<TStorage>::type, 3> tY,
                                              tensor_ref<const int16_t, 2>                                      tDMRS_index_freq,
                                              tensor_ref<const int16_t, 2>                                      tDMRS_index_time,
                                              tensor_ref<const TStorage, 3>                                     tW_freq,
                                              tensor_ref<const TStorage, 3>                                     tW_time,
                                              int*                                                              debug)

{
    typedef typename complex_from_scalar<TCompute>::type TComplexCompute;
    typedef typename complex_from_scalar<TStorage>::type TComplexStorage;
    const int                                            UE_IDX = blockIdx.x;
    //const int NFA        = tW_freq.dim[0];
    //const int NFP      = tW_freq.dim[1];
    const int NTP     = WARP_TILE_SIZE_X;
    const int NUM_ANT = tY.dim[2];

    const int THREADS_PER_WARP = 32;
    const int THREAD_LANE      = threadIdx.x % THREADS_PER_WARP;
    const int THREAD_TILE_X    = THREAD_LANE / WARP_TILE_SIZE_Y; // [0-3]
    const int THREAD_TILE_Y    = THREAD_LANE % WARP_TILE_SIZE_Y; // [0-7]
    //const int NUM_WARPS            = BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y;
    const int WARP_IDX             = threadIdx.x / THREADS_PER_WARP;
    const int WARP_TILE_X          = WARP_IDX / BLOCK_TILE_SIZE_Y; // [0-7]
    const int WARP_TILE_Y          = WARP_IDX % BLOCK_TILE_SIZE_Y; // [0-4]
    const int LOGICAL_X            = WARP_TILE_SIZE_X * WARP_TILE_X + THREAD_TILE_X;
    const int LOGICAL_Y            = WARP_TILE_SIZE_Y * WARP_TILE_Y + THREAD_TILE_Y;
    const int BLOCK_SIZE_THREADS_X = BLOCK_TILE_SIZE_X * WARP_TILE_SIZE_X;
    //const int BLOCK_SIZE_THREADS_Y = BLOCK_TILE_SIZE_Y * WARP_TILE_SIZE_Y;
    const int LOAD_FREQS_PER_WARP   = THREADS_PER_WARP / BLOCK_SIZE_THREADS_X;
    const int THREADS_PER_LOAD_FREQ = THREADS_PER_WARP / LOAD_FREQS_PER_WARP;
    const int FREQ_LOAD_ROW         = threadIdx.x / THREADS_PER_LOAD_FREQ;
    const int FREQ_LOAD_COL         = threadIdx.x % THREADS_PER_LOAD_FREQ;
    const int FREQ_LOAD_ANT_IDX     = FREQ_LOAD_COL / NTP;
    const int LOAD_FREQ_IDX         = tDMRS_index_freq(FREQ_LOAD_ROW, UE_IDX);
    const int TIME_IDX              = tDMRS_index_time(THREAD_LANE % NTP, UE_IDX);
    //printf("threadIdx.x = %i, FREQ_LOAD_ROW = %i, FREQ_LOAD_COL = %i\n", threadIdx.x, FREQ_LOAD_ROW, FREQ_LOAD_COL);

    //__shared__ block_1D<int16_t, NTP>                              DMRS_index_time;
    //__shared__ block_1D<int16_t, NFP>                              DMRS_index_freq;
    __shared__ block_2D<TComplexCompute, BLOCK_SIZE_THREADS_X, 32> Hp_block; // Stored transposed
    __shared__ block_2D<TCompute, 96, NFP> Wfreq_ue_block;
    //__shared__ block_2D<TCompute, NTP, 14>                         Wtime_t;
#if 1
    __shared__ block_2D<TComplexStorage, 32, BLOCK_TILE_SIZE_X * 14> Hinterp_out;
#endif

    // Load index data for this UE
    //load_1D(DMRS_index_time.data, &tDMRS_index_time(0, UE_IDX), NTP);
    //load_2D_transpose(Wtime_t.data, &tW_time(0, 0, UE_IDX), NT, NTP, tW_time.strides[1]);
    //if(debug) atomicAdd(debug + 0, );
    //load_1D(DMRS_index_freq.data, &tDMRS_index_freq(0, UE_IDX), NFP);
    //__syncthreads();

    //------------------------------------------------------------------
    // Load time filter coefficients into registers
    TCompute Wtime_t[14];
#pragma unroll
    for(int i = 0; i < 14; ++i)
    {
        Wtime_t[i] = tW_time(i, THREAD_TILE_X, UE_IDX);
    }

    //dump_shared_mem("DMRS_index_time", DMRS_index_time.data, NTP);
    //dump_shared_mem("DMRS_index_freq", DMRS_index_freq.data, NFP);
    //dump_shared_mem("Wtime_t", Wtime_t.data, NT * NTP);
    //printf("threadIdx.x = %i, WARP_IDX = %i, LOAD_FREQ_IDX = %i, TIME_INDEX = %i, STORE_ROW/COL = %i\n",
    //       threadIdx.x,
    //       WARP_IDX,
    //       LOAD_FREQ_IDX,
    //       THREAD_LANE % NTP,
    //       THREAD_LANE);
    //------------------------------------------------------------------
    // Double nested loop (antenna and frequency blocks) with prefetching
    // into registers followed by storage into shared memory.
    //------------------------------------------------------------------
    // Issue reads to load Wfreq and Hp data into registers...
    const int NUM_ANT_BLK  = NUM_ANT / BLOCK_TILE_SIZE_X;
    const int NUM_FREQ_BLK = 3;
    //TComplexStorage Hp_thread     = tY(threadIdx.x, 0, 0);   // TODO: Load using freq indices
    //printf("READ: threadIdx.x = %i, i0 = %i, i1 = %i, i2 = %i\n", threadIdx.x, LOAD_FREQ_IDX, DMRS_index_time[THREAD_LANE % NTP], THREAD_LANE / NTP);
    TComplexStorage Hp_thread = tY(LOAD_FREQ_IDX,
                                   TIME_IDX,
                                   FREQ_LOAD_ANT_IDX);
    if(debug) atomicAdd(debug + 0, 2);
    TStorage W_freq_thread[2];
#pragma unroll
    for(int i = 0; i < 2; ++i)
    {
        W_freq_thread[i] = tW_freq(THREAD_LANE, WARP_IDX + (i * BLOCK_SIZE_THREADS_X), UE_IDX);
    }
    if(debug) atomicAdd(debug + 0, 2);
    for(int iANTBLK = 0; iANTBLK < NUM_ANT_BLK; ++iANTBLK)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Store Hp in registers to shared memory
        //printf("Hp STORE: threadIdx.x = %i, i0 = %i, i1 = %i\n", threadIdx.x, FREQ_LOAD_COL, FREQ_LOAD_ROW);
        Hp_block(FREQ_LOAD_COL, FREQ_LOAD_ROW) = type_convert<TComplexCompute>(Hp_thread);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Issue reads for the next block of Hp
        if((iANTBLK + 1) < NUM_ANT_BLK)
        {
            //Hp_thread = tY(threadIdx.x, 0, 0);   // TODO: Load using freq indices
            Hp_thread = tY(LOAD_FREQ_IDX,
                           TIME_IDX,
                           ((iANTBLK + 1) * BLOCK_TILE_SIZE_X) + FREQ_LOAD_ANT_IDX);
            if(debug) atomicAdd(debug + 0, 2);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Loop over frequency blocks
        for(int iFREQBLK = 0; iFREQBLK < NUM_FREQ_BLK; ++iFREQBLK)
        {
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Store Wfreq in registers to shared memory and issue loads
            // for the next block
            if(0 == iANTBLK)
            {
#pragma unroll
                for(int i = 0; i < 2; ++i)
                {
                    Wfreq_ue_block(iFREQBLK * 32 + THREAD_LANE, WARP_IDX + (i * BLOCK_SIZE_THREADS_X)) = type_convert<TCompute>(W_freq_thread[i]);
                }
                if((iFREQBLK + 1) < NUM_FREQ_BLK)
                {
#pragma unroll
                    for(int i = 0; i < 2; ++i)
                    {
                        W_freq_thread[i] = tW_freq((iFREQBLK + 1) * 32 + THREAD_LANE,
                                                   WARP_IDX + (i * BLOCK_SIZE_THREADS_X),
                                                   UE_IDX);
                    }
                    if(debug) atomicAdd(debug + 0, 2);
                }
            }
            __syncthreads();
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Do computation. Matrix product C is computed as the sum of
            // the outer product of A.column[i] and B.row[i], for i = 0:31.
            // (Here A is a block of the Wfreq matrix and B is a block of
            // the Hp matrix.)
            TComplexCompute threadC[4]{};
#pragma unroll
            for(int i = 0; i < 4; ++i)
            {
                for(int j = 0; j < 8; ++j)
                {
                    TCompute        Wfreq = Wfreq_ue_block(iFREQBLK * 32 + LOGICAL_Y, (i * 8) + j);
                    TComplexCompute Hp    = Hp_block(LOGICAL_X, (i * 8) + j);
                    threadC[i].x += (Wfreq * Hp.x);
                    threadC[i].y += (Wfreq * Hp.y);
                    //if(0 == threadIdx.x) { printf("i = %i, Wfreq = %.4f, Hp = (%.4f, %.4fi), C = (%.4f, %.4f)\n", i, Wfreq, Hp.x, Hp.y, threadC.x, threadC.y); }
                }
            }
            threadC[0].x += threadC[2].x;
            threadC[0].y += threadC[2].y;
            threadC[1].x += threadC[3].x;
            threadC[1].y += threadC[3].y;
            threadC[0].x += threadC[1].x;
            threadC[0].y += threadC[1].y;
//-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
// Compute the product of the warp row with Wtime filter
// variables, and store results in shared memory.
#pragma unroll
            for(int iTime = 0; iTime < NT; ++iTime)
            {
                TCompute        Wtime = Wtime_t[iTime];
                TComplexCompute Hinterp;
                Hinterp.x = threadC[0].x * Wtime;
                Hinterp.y = threadC[0].y * Wtime;
                Hinterp.x += __shfl_down_sync(0xFFFFFFFF, Hinterp.x, WARP_TILE_SIZE_Y * 2);
                Hinterp.y += __shfl_down_sync(0xFFFFFFFF, Hinterp.y, WARP_TILE_SIZE_Y * 2);
                Hinterp.x += __shfl_down_sync(0xFFFFFFFF, Hinterp.x, WARP_TILE_SIZE_Y * 1);
                Hinterp.y += __shfl_down_sync(0xFFFFFFFF, Hinterp.y, WARP_TILE_SIZE_Y * 1);
                if(0 == THREAD_TILE_X)
                {
#if 1
                    Hinterp_out(LOGICAL_Y, (WARP_TILE_X * 14) + iTime) = type_convert<TComplexStorage>(Hinterp);
#endif
                }
            }
            __syncthreads();
//dump_shared_mem("Hinterp_out", Hinterp_out.data, 32 * BLOCK_TILE_SIZE_X * 14);
//-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
// Write output to global memory
#pragma unroll
            for(int iANT = 0; iANT < BLOCK_TILE_SIZE_X; ++iANT)
            {
                if(WARP_IDX < 14)
                {
                    //printf("threadIdx.x = %i: writing to (freq = %i, time = %i, ant = %i, ue = %i) from (%i, %i)\n",
                    //       threadIdx.x,
                    //       (iFREQBLK * 32) + (threadIdx.x % 32),
                    //       WARP_IDX,
                    //       (BLOCK_TILE_SIZE_X * iANTBLK) + iANT,
                    //       UE_IDX,
                    //       (threadIdx.x % 32), (iANT * 14) + WARP_IDX);
#if 1
                    tHinterp((iFREQBLK * 32) + (threadIdx.x % 32),
                             WARP_IDX,
                             (BLOCK_TILE_SIZE_X * iANTBLK) + iANT,
                             UE_IDX) = Hinterp_out(threadIdx.x % 32, (iANT * 14) + WARP_IDX);
                    if(debug) atomicAdd(debug + 1, 2);
#endif
                }
            }
        } // for each frequency block
    }     // for each antenna block
    //dump_shared_mem("Wfreq", Wfreq_ue_block.data, NFA * NFP);
    //dump_shared_mem("Hp_block", Hp_block.data, BLOCK_TILE_SIZE_X * WARP_TILE_SIZE_X * 32);
    //dump_shared_mem("Hinterp_out", Hinterp_out.data, 32 * BLOCK_TILE_SIZE_X * 14);

#if 0
    // Write dummy output
    for(int kAntenna = 0; kAntenna < 16; ++kAntenna)
    {
        TComplexStorage* Hinterp_ant = &tHinterp(0, 0, kAntenna, UE_IDX);
        for(int i = threadIdx.x; i < (NT * NFA); i+= blockDim.x)
        {
            Hinterp_ant[i] = type_convert<TComplexStorage>(make_complex<TComplexCompute>::create(threadIdx.x, kAntenna));
            if(debug) atomicAdd(debug + 1, 2);
        }
    }
#endif
}

////////////////////////////////////////////////////////////////////////
// mmes_1D_time_frequency_launch()
template <typename TStorage, typename TCompute>
void mmse_1D_time_frequency_launch(tensor_pair&       tDst,
                                   const_tensor_pair& tSymbols,
                                   const_tensor_pair& tFreqFilters,
                                   const_tensor_pair& tTimeFilters,
                                   const_tensor_pair& tFreqIndices,
                                   const_tensor_pair& tTimeIndices,
                                   cudaStream_t       strm)
{
    const int WARP_TILE_SIZE_X  = 4;
    const int WARP_TILE_SIZE_Y  = 8;
    const int BLOCK_TILE_SIZE_X = 4;
    const int BLOCK_TILE_SIZE_Y = 4;
    const int NFP               = 32;
    const int NT                = 14;
#if 0
    cudaFuncAttributes fAttr;
    if(cudaSuccess == cudaFuncGetAttributes(&fAttr, mmse_1D_time_frequency_kernel<TStorage,
                                                                                  TCompute,
                                                                                  BLOCK_TILE_SIZE_X,
                                                                                  BLOCK_TILE_SIZE_Y,
                                                                                  WARP_TILE_SIZE_X,
                                                                                  WARP_TILE_SIZE_Y,
                                                                                  NFP,
                                                                                  NT>))
    {
        printf("shared mem: %lu\n", fAttr.sharedSizeBytes);
    }
#endif

    int* debugInt = nullptr;
#if 0
    const int NUM_DEBUG_INT = 2;
    std::array<int, NUM_DEBUG_INT> hostDebug{}; // zero-initialize
    if(cudaSuccess != cudaMalloc(&debugInt, hostDebug.size() * sizeof(int)))
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "Error allocating");
    }
    cudaMemcpy(debugInt, hostDebug.data(), hostDebug.size() * sizeof(int), cudaMemcpyHostToDevice);
#endif

    typedef typename complex_from_scalar<TStorage>::type complex_storage_t;
    typedef typename complex_from_scalar<TCompute>::type complex_compute_t;
    typedef typename data_type_traits<CUPHY_R_16I>::type index_t;

    tensor_ref<const complex_storage_t, 3> Y(tSymbols);
    tensor_ref<complex_storage_t, 4>       Hinterp(tDst);
    tensor_ref<const index_t, 2>           DMRS_index_freq(tFreqIndices);
    tensor_ref<const index_t, 2>           DMRS_index_time(tTimeIndices);
    tensor_ref<const TStorage, 3>          W_freq(tFreqFilters);
    tensor_ref<const TStorage, 3>          W_time(tTimeFilters);

    int numUEs = DMRS_index_freq.dim[1];
#if 0
    // Override number of blocks (UEs)
    numUEs = 1;
#endif
    //printf("numUEs: %i\n", numUEs);

#if 0
    cudaEvent_t eStart, eFinish;
    cudaEventCreate(&eStart);
    cudaEventCreate(&eFinish);
#endif
    dim3 gridDim(numUEs);
    dim3 blockDim(BLOCK_TILE_SIZE_X * BLOCK_TILE_SIZE_Y * 32);
#if 1
    // WARMUP, also pass debug pointer to count reads + writes
    mmse_1D_time_frequency_kernel<TStorage,
                                  TCompute,
                                  BLOCK_TILE_SIZE_X,
                                  BLOCK_TILE_SIZE_Y,
                                  WARP_TILE_SIZE_X,
                                  WARP_TILE_SIZE_Y,
                                  NFP,
                                  NT><<<gridDim, blockDim, 0, strm>>>(Hinterp,
                                                                      Y,
                                                                      DMRS_index_freq,
                                                                      DMRS_index_time,
                                                                      W_freq,
                                                                      W_time,
                                                                      debugInt);
#endif

#if 0
    const int ITER_COUNT = 1000;
    cudaEventRecord(eStart, strm);
    for(int i = 0; i < ITER_COUNT; ++i)
    {
        mmse_1D_time_frequency_kernel<TStorage,
                                      TCompute,
                                      BLOCK_TILE_SIZE_X,
                                      BLOCK_TILE_SIZE_Y,
                                      WARP_TILE_SIZE_X,
                                      WARP_TILE_SIZE_Y,
                                      NFP,
                                      NT><<<gridDim, blockDim, 0, strm>>>(Hinterp,
                                                                          Y,
                                                                          DMRS_index_freq,
                                                                          DMRS_index_time,
                                                                          W_freq,
                                                                          W_time,
                                                                          nullptr);
    }
    cudaEventRecord(eFinish, strm);
    cudaEventSynchronize(eFinish);
    cudaDeviceSynchronize();
    float elapsed_ms = 0.0f;
    cudaEventElapsedTime(&elapsed_ms, eStart, eFinish);
#endif
#if 0
    // Read debug data, which contains read and write words
    cudaMemcpy(hostDebug.data(), debugInt, hostDebug.size() * sizeof(int), cudaMemcpyDeviceToHost);
    printf("debug int: ");
    for(int i = 0; i < hostDebug.size(); ++i)
    {
        printf("%i ", hostDebug[i]);
    }
    printf("\n");
    cudaFree(debugInt);
#endif

    //const size_t kernelReadBytes = sizeof(complex_storage_t) * 32 * 4 * 16 * 156;
    //const size_t kernelReadBytes = sizeof(complex_storage_t) * 1248 * 14 * 16 * numUEs;
    //printf("Kernel reads: %i words (%lu bytes, sizeof(word) = %lu)\n", hostDebug[0], hostDebug[0] * sizeof(TStorage), sizeof(TStorage));
    //printf("Kernel write: %i words (%lu bytes, sizeof(word) = %lu)\n", hostDebug[1], hostDebug[1] * sizeof(TStorage), sizeof(TStorage));
    //printf("Total:        %i words (%lu bytes)\n", hostDebug[0] + hostDebug[1], (hostDebug[0] + hostDebug[1]) * sizeof(TStorage));
    //size_t IO_BYTES_PER_ITER = (hostDebug[0] + hostDebug[1]) * sizeof(TStorage);

    //const size_t kernelWriteBytes = sizeof(complex_storage_t) * 96 * 14 * 16 * 156;
    //printf("kernelWriteBytes = %lu\n", kernelWriteBytes);

    //printf("Total time: %.3f ms   Average time: %.3f ms, BW: %f GB/s\n", elapsed_ms, elapsed_ms / ITER_COUNT, 1000.0f * 1.0e-9 * kernelWriteBytes / (elapsed_ms / ITER_COUNT));
    //printf("Total time: %.3f ms   Average time: %.3f ms, BW: %f GB/s, GFLOPS: %f\n",
    //       elapsed_ms,
    //       elapsed_ms / ITER_COUNT,
    //       1000.0f * 1.0e-9 * IO_BYTES_PER_ITER / (elapsed_ms / ITER_COUNT),
    //       (1000.0 * 1.0e-9 * 1130496 * numUEs) / (elapsed_ms / ITER_COUNT));

    std::vector<complex_storage_t> Hinterp_host(96 * 14 * 16);
    cudaMemcpy(Hinterp_host.data(), Hinterp.addr, 96 * 14 * 16 * sizeof(complex_storage_t), cudaMemcpyDeviceToHost);
    for(int i = 0; i < 16; ++i)
    {
        cuComplex c = type_convert<cuComplex>(Hinterp_host[i]);
        printf("Hinterp_host[%i] = (%.4f, %.4f)\n", i, c.x, c.y);
    }
}

////////////////////////////////////////////////////////////////////////
// mmse_1D_time_frequency()
void mmse_1D_time_frequency(tensor_pair&       tDst,
                            const_tensor_pair& tSymbols,
                            const_tensor_pair& tFreqFilters,
                            const_tensor_pair& tTimeFilters,
                            const_tensor_pair& tFreqIndices,
                            const_tensor_pair& tTimeIndices,
                            cudaStream_t       strm)
{
    //printf("channel_est::mmse_1D_time_frequency() begin()\n");
    switch(tDst.first.get().type())
    {
    case CUPHY_C_16F:
    {
        typedef scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type TStorage;
        mmse_1D_time_frequency_launch<TStorage, float>(tDst,
                                                       tSymbols,
                                                       tFreqFilters,
                                                       tTimeFilters,
                                                       tFreqIndices,
                                                       tTimeIndices,
                                                       strm);
    }
    break;
    case CUPHY_C_32F:
    {
        typedef scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type TStorage;
        mmse_1D_time_frequency_launch<TStorage, float>(tDst,
                                                       tSymbols,
                                                       tFreqFilters,
                                                       tTimeFilters,
                                                       tFreqIndices,
                                                       tTimeIndices,
                                                       strm);
    }
    break;
    default:
        break;
    }

    //printf("channel_est::mmse_1D_time_frequency() end()\n");
}

} // namespace channel_est
