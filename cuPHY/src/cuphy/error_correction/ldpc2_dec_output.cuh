/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_DEC_OUTPUT_CUH_INCLUDED_)
#define LDPC2_DEC_OUTPUT_CUH_INCLUDED_

#include "ldpc2.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// output_codeword_addr
// Provides the output address for a codeword when using the tensor-
// based LDPC decoder interface.
struct output_codeword_addr
{
    static __device__ uint32_t* get(const LDPC_kernel_params& params, int idx)
    {
        return reinterpret_cast<uint32_t*>(params.out + (idx * sizeof(uint32_t) * params.output_stride_words));
    }
};

////////////////////////////////////////////////////////////////////////
// decode_desc_output_addr()
// Provides the output address for a codeword when using the transport
// block-based LDPC decoder interface.
template <typename T> struct decode_desc_output_addr
{
    __device__
    static uint32_t* get(const cuphyLDPCDecodeDesc_t& decodeDesc, int cwIndex)
    {
        uint32_t* addr = nullptr;
        #pragma unroll
        for(int i = 0; i < CUPHY_LDPC_DECODE_DESC_MAX_TB; ++i)
        {
            if(i < decodeDesc.num_tbs)
            {
                if(cwIndex < decodeDesc.tb_output[i].num_codewords)
                {
                    addr = decodeDesc.tb_output[i].addr + (cwIndex * decodeDesc.tb_output[i].stride_words);
                    break;
                }
                cwIndex -= decodeDesc.tb_output[i].num_codewords;
            }
        }
        return addr;
    }
};

////////////////////////////////////////////////////////////////////////
// decode_desc_output_addr
// Specialization of decode_desc_output_addr for __half2
//template <> struct decode_desc_output_addr<__half2>
//{
//    __device__
//    static uint32_t* get()
//    {
//        return nullptr;
//    }
//};

////////////////////////////////////////////////////////////////////////
// num_cta_output_codewords()
// Returns the number of output codewords for a CTA.
// For 1x codeword at a time kernels, the return value is always 1.
// For 2x codeword at a time kernels, the return value will be 2 in all
// cases, except for the last CTA when the number of output codewords is
// odd. (In that case the return value will be 1.)
// This function is useful as written when the number of CTAs is equal
// to the number of codewords, but should not be used for looping.
template <typename T> __device__
int num_cta_output_codewords(int total_num_cw)
{
    if(1 == codewords_per_CTA<T>::value)
    {
        return 1;
    }
    else
    {
        if((blockIdx.x * codewords_per_CTA<T>::value + 1) < total_num_cw)
        {
            return 2;
        }
        else
        {
            return 1;
        }
    }
}

template <typename T> struct ldpc_dec_output_params;

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_params<>
template <typename T> struct ldpc_dec_output_params
{
    uint32_t* dst_gmem;         // output address for this CTA
    int       num_cw_bits;      // needed for variable outputs
    int       out_words_per_cw; // NOTE: can be derived from above...
    __device__
    ldpc_dec_output_params(const LDPC_kernel_params& params, int cwIndex) :
        dst_gmem(output_codeword_addr::get(params, cwIndex)),
        num_cw_bits(get_num_output_bits(params)),
        out_words_per_cw((num_cw_bits + 31) / 32)
    {
    }
    __device__
    ldpc_dec_output_params(uint32_t* dst,
                           int       nbits,
                           int       nwords) :
        dst_gmem(dst),
        num_cw_bits(nbits),
        out_words_per_cw(nwords)
    {
    }
    __device__
    ldpc_dec_output_params(const cuphyLDPCDecodeDesc_t& decodeDesc, int cwIndex) :
        dst_gmem(decode_desc_output_addr<T>::get(decodeDesc, cwIndex)),
        num_cw_bits(get_num_output_bits(decodeDesc)),
        out_words_per_cw((num_cw_bits + 31) / 32)
    {
    }
    __device__
    ldpc_dec_output_params(const cuphyLDPCDecodeDesc_t& decodeDesc,
                           tb_token                     tok) :
        dst_gmem(nullptr),
        num_cw_bits(get_num_output_bits(decodeDesc)),
        out_words_per_cw((num_cw_bits + 31) / 32)
    {
        int  tb     = tb_from_token(tok);
        int  offset = offset_from_token(tok);
        int  stride = decodeDesc.tb_output[tb].stride_words;
        dst_gmem    = decodeDesc.tb_output[tb].addr + (offset * stride);
    }
};

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_params<>
// specialization for __half2 (2 codewords at a time)
template <>
struct ldpc_dec_output_params<__half2>
{
    uint32_t* dst_gmem;
    int       num_cw_bits;      // needed for variable outputs
    int       out_words_per_cw; // NOTE: can be derived from above...
    int       out_stride_words; // needed for 2x codewords, where output stores two consecutive outputs
    int       num_out_cw;       // number of output codewords (by this CTA!), needed for 2x codeword kernels only
    __device__
    ldpc_dec_output_params(const LDPC_kernel_params& params) :
        dst_gmem(output_codeword_addr::get(params, blockIdx.x * codewords_per_CTA<__half2>::value)),
        num_cw_bits(get_num_output_bits(params)),
        out_words_per_cw((num_cw_bits + 31) / 32),
        out_stride_words(params.output_stride_words),
        num_out_cw(num_cta_output_codewords<__half2>(params.num_codewords))
    {
    }
    __device__
    ldpc_dec_output_params(const cuphyLDPCDecodeDesc_t& decodeDesc) :
        dst_gmem(nullptr),
        num_cw_bits(get_num_output_bits(decodeDesc)),
        out_words_per_cw((num_cw_bits + 31) / 32),
        out_stride_words(0),
        num_out_cw(0)
    {
        int blkIndex = blockIdx.x;
        #pragma unroll
        for(int i = 0; i < CUPHY_LDPC_DECODE_DESC_MAX_TB; ++i)
        {
            if(i < decodeDesc.num_tbs)
            {
                int iBlocksClaimed = (decodeDesc.tb_output[i].num_codewords + 1) / 2;
                if(blkIndex < iBlocksClaimed)
                {
                    out_stride_words = decodeDesc.tb_output[i].stride_words;
                    dst_gmem         = decodeDesc.tb_output[i].addr + (blkIndex * 2 * out_stride_words);
                    // Last block may have only 1 codeword...
                    num_out_cw       = ((blkIndex*2 + 1) == decodeDesc.tb_output[i].num_codewords) ? 1 : 2;
                    break;
                }
                blkIndex -= iBlocksClaimed;
            }
        }
    }
    __device__
    ldpc_dec_output_params(const cuphyLDPCDecodeDesc_t& decodeDesc,
                           tb_token                     tok) :
        dst_gmem(nullptr),
        num_cw_bits(get_num_output_bits(decodeDesc)),
        out_words_per_cw((num_cw_bits + 31) / 32),
        out_stride_words(0),
        num_out_cw(0)
    {
        int  tb          = tb_from_token(tok);
        int  offset      = offset_from_token(tok);
        bool is_partial  = is_partial_from_token(tok);
        out_stride_words = decodeDesc.tb_output[tb].stride_words;
        dst_gmem         = decodeDesc.tb_output[tb].addr + (offset * out_stride_words);
        num_out_cw       = is_partial ? 1 : 2;
    }
};


////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_fixed()
//template <typename T, int NODES, int Z>
//static inline __device__ void ldpc_dec_output_fixed(const ldpc_dec_output_params<T>& params,
//                                                    const float*                     app_smem)
//{
//    // The number of threads per warp.
//    enum
//    {
//        THREADS_PER_WARP = 32
//    };
//
//    // Decompose the thread indices into warp/lane.
//    int warp = threadIdx.x / THREADS_PER_WARP;
//    int lane = threadIdx.x % THREADS_PER_WARP;
//
//    // The output per thread.
//    uint32_t output = 0;
//
//    // Each warp reads 32*THREADS_PER_WARP elements.
//    int idx = warp * 32 * THREADS_PER_WARP + lane;
//    for(int ii = 0; ii < 32; ++ii)
//    {
//        float app = 0.f;
//        if(idx + ii * THREADS_PER_WARP < NODES * Z)
//        {
//            app = app_smem[idx + ii * THREADS_PER_WARP];
//        }
//
//        unsigned int vote = __ballot_sync(0xffffffff, signbit(app));
//        if(lane == ii)
//        {
//            output = vote;
//        }
//    }
//
//    // Output the result.
//    if(threadIdx.x < params.out_words_per_cw)
//    {
//        params.dst_gmem[threadIdx.x] = output;
//    }
//}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_fixed()
//template <typename T, int NODES, int Z>
//static inline __device__ void ldpc_dec_output_fixed(const ldpc_dec_output_params<T>& params,
//                                                    const __half*                    app_smem)
//{
//    // The number of threads per warp.
//    enum
//    {
//        THREADS_PER_WARP = 32
//    };
//
//    // Decompose the thread indices into warp/lane.
//    int warp = threadIdx.x / THREADS_PER_WARP;
//    int lane = threadIdx.x % THREADS_PER_WARP;
//
//    // The output per thread.
//    uint32_t output = 0;
//
//    // Each warp reads 32*THREADS_PER_WARP elements.
//    int idx = warp * 32 * THREADS_PER_WARP + lane;
//    for(int ii = 0; ii < 32; ++ii)
//    {
//        __half app = __float2half(0.0f);
//        if((idx + ii * THREADS_PER_WARP) < (NODES * Z))
//        {
//            app = app_smem[idx + ii * THREADS_PER_WARP];
//        }
//
//        unsigned int vote = __ballot_sync(0xffffffff, signbit(app));
//        if(lane == ii)
//        {
//            output = vote;
//        }
//    }
//
//    // Output the result.
//    if(threadIdx.x < params.out_words_per_cw)
//    {
//        params.dst_gmem[threadIdx.x] = output;
//    }
//}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
//template <typename T>
//static inline __device__ void ldpc_dec_output_variable(const ldpc_dec_output_params<T>& params,
//                                                       const float*                     app_smem)
//{
//    // The number of threads per warp.
//    enum
//    {
//        THREADS_PER_WARP = 32
//    };
//
//    // Decompose the thread indices into warp/lane.
//    int warp = threadIdx.x / THREADS_PER_WARP;
//    int lane = threadIdx.x % THREADS_PER_WARP;
//
//    // The output per thread.
//    uint32_t output = 0;
//
//    // Each warp reads 32*THREADS_PER_WARP elements.
//    int idx = warp * 32 * THREADS_PER_WARP + lane;
//    for(int ii = 0; ii < 32; ++ii)
//    {
//        float app = 0.f;
//        if((idx + ii * THREADS_PER_WARP) < params.num_cw_bits)
//        {
//            app = app_smem[idx + ii * THREADS_PER_WARP];
//        }
//
//        unsigned int vote = __ballot_sync(0xffffffff, signbit(app));
//        if(lane == ii)
//        {
//            output = vote;
//        }
//    }
//
//    // Output the result.
//    if(threadIdx.x < params.out_words_per_cw)
//    {
//        params.dst_gmem[threadIdx.x] = output;
//    }
//}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
template <typename T>
static inline __device__ void ldpc_dec_output_variable(const ldpc_dec_output_params<T>& params,
                                                       const __half*                   app_smem)
{
    // The number of threads per warp.
    enum
    {
        THREADS_PER_WARP = 32
    };
    //---------------------------------------------------------------
    // Each warp reads 32*THREADS_PER_WARP=1024 APP values and writes
    // 1024 bits in the form of 32 uint32_t values.
    const int WARP_IDX      = threadIdx.x / THREADS_PER_WARP;
    const int LANE          = threadIdx.x % THREADS_PER_WARP;
    const int BITS_PER_WARP = THREADS_PER_WARP * sizeof(uint32_t) * CHAR_BIT;
    //---------------------------------------------------------------
    // Check for early exit
    const int NUM_WARPS_REQ = (params.num_cw_bits + BITS_PER_WARP - 1) / BITS_PER_WARP;
    if(WARP_IDX >= NUM_WARPS_REQ)
    {
        return;
    }

    int output_idx = threadIdx.x;
    int start_idx  = WARP_IDX * BITS_PER_WARP + LANE;
    {
        uint32_t  output_value = 0;
        for(int ii = 0; ii < 32; ++ii)
        {
            const int    APP_IDX = start_idx + (ii * THREADS_PER_WARP);

            // Load soft decision from shared memory.
            // If index out of range, load value that is 0b as hard decision.
            const __half APP     = (APP_IDX < params.num_cw_bits) ?
                                   app_smem[APP_IDX]              :
                                   __float2half(1.0f);
            const uint32_t VOTE  = __ballot_sync(0xffffffff, llr_hard_decision(APP));
            if(LANE == ii)
            {
                output_value = VOTE;
            }
        }
        // Output the result.
        if(output_idx < params.out_words_per_cw)
        {
            params.dst_gmem[output_idx] = output_value;
        }
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
template <typename T>
static inline __device__ void ldpc_dec_output_variable_loop(const ldpc_dec_output_params<T>& params,
                                                            const __half*                   app_smem)
{
    // The number of threads per warp.
    enum
    {
        THREADS_PER_WARP = 32
    };
    //---------------------------------------------------------------
    // Each warp reads 32*THREADS_PER_WARP=1024 APP values and writes
    // 1024 bits in the form of 32 uint32_t values.
    const int WARP_IDX      = threadIdx.x / THREADS_PER_WARP;
    const int LANE          = threadIdx.x % THREADS_PER_WARP;
    const int BITS_PER_WARP = THREADS_PER_WARP * sizeof(uint32_t) * CHAR_BIT;
    //---------------------------------------------------------------
    // Check for early exit
    //const int NUM_WARPS          = (blockDim.x + 31) / 32;
    const int NUM_FULL_WARPS     = (blockDim.x / 32);
    const int NUM_FULL_WARPS_REQ = (params.num_cw_bits + BITS_PER_WARP - 1) / BITS_PER_WARP;
    const int NUM_ACTIVE_WARPS   = min(NUM_FULL_WARPS, NUM_FULL_WARPS_REQ);
    if(WARP_IDX >= NUM_ACTIVE_WARPS)
    {
        return;
    }

    int output_idx = threadIdx.x;
    int start_idx  = WARP_IDX * BITS_PER_WARP + LANE;
    do
    {
        uint32_t  output_value = 0;
        for(int ii = 0; ii < 32; ++ii)
        {
            const int    APP_IDX = start_idx + (ii * THREADS_PER_WARP);

            // Load soft decision from shared memory.
            // If index out of range, load value that is 0b as hard decision.
            const __half APP     = (APP_IDX < params.num_cw_bits) ?
                                   app_smem[APP_IDX]              :
                                   __float2half(1.0f);
            const uint32_t VOTE  = __ballot_sync(0xffffffff, llr_hard_decision(APP));
            if(LANE == ii)
            {
                output_value = VOTE;
            }
        }
        // Output the result.
        if(output_idx < params.out_words_per_cw)
        {
            params.dst_gmem[output_idx] = output_value;
        }
        // Advance
        output_idx += blockDim.x;
        start_idx  += (NUM_ACTIVE_WARPS * BITS_PER_WARP);
    } while(start_idx < params.num_cw_bits);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
static inline __device__ void ldpc_dec_output_variable(const LDPC_kernel_params& kernelParams,
                                                       const __half*             app_smem)
{
    ldpc_dec_output_params<__half> params(kernelParams, blockIdx.x);
    ldpc_dec_output_variable(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_multi()
static inline __device__ void ldpc_dec_output_variable_multi(const LDPC_kernel_params&    kernelParams,
                                                             const __half*                app_smem,
                                                             const multi_codeword_config& mconfig)
{
    const int LLR_STRIDE_VALUES   = round_up_to_next(get_num_LLRs(kernelParams),
                                                     static_cast<int>(sizeof(ldpc_traits<__half>::llr_sts_t) / sizeof(__half)));
    const int OUTPUT_STRIDE_BYTES = sizeof(uint32_t) * kernelParams.output_stride_words;
    for(int i = 0; i < mconfig.cta_codeword_count; ++i)
    {
        char* dst = kernelParams.out + ((mconfig.cta_start_index + i) * OUTPUT_STRIDE_BYTES);
        ldpc_dec_output_params<__half> params(reinterpret_cast<uint32_t*>(dst),               // output address
                                              get_num_output_bits(kernelParams),              // output bits
                                              (get_num_output_bits(kernelParams) + 31) / 32); // num output words
        ldpc_dec_output_variable(params, app_smem + (i * LLR_STRIDE_VALUES));
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
static inline __device__ void ldpc_dec_output_variable(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                       const __half*                app_smem)
{
    ldpc_dec_output_params<__half> params(decodeDesc, blockIdx.x);
    ldpc_dec_output_variable(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_multi()
static inline __device__ void ldpc_dec_output_variable_multi(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                             const __half*                app_smem,
                                                             const multi_codeword_config& mconfig)
{
    const int LLR_STRIDE_VALUES = round_up_to_next(get_num_LLRs(decodeDesc),
                                                   static_cast<int>(sizeof(ldpc_traits<__half>::llr_sts_t) / sizeof(__half)));
    for(int i = 0; i < mconfig.cta_codeword_count; ++i)
    {
        ldpc_dec_output_params<__half> params(decodeDesc,
                                              mconfig.cta_start_index + i);
        ldpc_dec_output_variable(params, app_smem + (i * LLR_STRIDE_VALUES));
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
static inline __device__ void ldpc_dec_output_variable_loop(const LDPC_kernel_params& kernelParams,
                                                            const __half*             app_smem)
{
    ldpc_dec_output_params<__half> params(kernelParams, blockIdx.x);
    ldpc_dec_output_variable_loop(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
static inline __device__ void ldpc_dec_output_variable_loop(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                            const __half*                app_smem)
{
    ldpc_dec_output_params<__half> params(decodeDesc, blockIdx.x);
    ldpc_dec_output_variable_loop(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
static inline __device__ void ldpc_dec_output_variable(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                       tb_token                     token,
                                                       const __half*                app_smem)
{
    ldpc_dec_output_params<__half> params(decodeDesc, token);
    ldpc_dec_output_variable(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
static inline __device__ void ldpc_dec_output_variable_loop(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                            tb_token                     token,
                                                            const __half*                app_smem)
{
    ldpc_dec_output_params<__half> params(decodeDesc, token);
    ldpc_dec_output_variable_loop(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
// Overload of hard decision output function for kernels that process
// two codewords at a time (using fp16x2 APP values in shared memory).
// "Variable" output functions use parameters that are not known at
// compile time.
template <typename T>
static inline __device__ void ldpc_dec_output_variable(const ldpc_dec_output_params<T>& params,
                                                       const __half2*                   app_smem)
{
    // The number of threads per warp.
    enum
    {
        THREADS_PER_WARP = 32
    };
    //---------------------------------------------------------------
    // Each warp reads 32*THREADS_PER_WARP=1024 APP values and writes
    // 1024 bits in the form of 32 uint32_t values.
    const int WARP_IDX      = threadIdx.x / THREADS_PER_WARP;
    const int LANE          = threadIdx.x % THREADS_PER_WARP;
    const int BITS_PER_WARP = THREADS_PER_WARP * sizeof(uint32_t) * CHAR_BIT;
    //---------------------------------------------------------------
    // Check for early exit
    const int NUM_WARPS_REQ = (params.num_cw_bits + BITS_PER_WARP - 1) / BITS_PER_WARP;
    if(WARP_IDX >= NUM_WARPS_REQ)
    {
        return;
    }

    int output_idx = threadIdx.x;
    int start_idx  = WARP_IDX * BITS_PER_WARP + LANE;
    {
        // The output per thread.
        uint32_t output[2] = {0, 0};

        for(int ii = 0; ii < 32; ++ii)
        {
            word_t       app;
            const int    APP_IDX = start_idx + (ii * THREADS_PER_WARP);
            // Load soft decision from shared memory.
            // If index out of range, load value that is 0b as hard decision.
            app.f16x2 = __half2_raw(__float2half2_rn(1.0));
            if(APP_IDX < params.num_cw_bits)
            {
                app.f16x2 = app_smem[APP_IDX];
            }
            //word_t app_sign_mask = fp16x2_sign_mask(app);
            //unsigned int vote0 = __ballot_sync(0xffffffff, (app_sign_mask.u32 & 0x00008000));
            //unsigned int vote1 = __ballot_sync(0xffffffff, (app_sign_mask.u32 & 0x80000000));
            __half2 fp16x2(app.f16x2);
            unsigned int vote0 = __ballot_sync(0xffffffff, (llr_hard_decision(fp16x2.x)));
            unsigned int vote1 = __ballot_sync(0xffffffff, (llr_hard_decision(fp16x2.y)));
            if(LANE == ii)
            {
                output[0] = vote0;
                output[1] = vote1;
            }
        }
        // Output the result.
        if(output_idx < params.out_words_per_cw)
        {
            params.dst_gmem[output_idx] = output[0];
            // Avoid writes past the end of the output when the number of
            // codewords is odd.
            if(2 == params.num_out_cw)
            {
                params.dst_gmem[output_idx + params.out_stride_words] = output[1];
            }
        }
    }
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
// Overload of hard decision output function for kernels that process
// two codewords at a time (using fp16x2 APP values in shared memory).
// "Variable" output functions use parameters that are not known at
// compile time.
template <typename T>
static inline __device__ void ldpc_dec_output_variable_loop(const ldpc_dec_output_params<T>& params,
                                                            const __half2*                   app_smem)
{
    // The number of threads per warp.
    enum
    {
        THREADS_PER_WARP = 32
    };
    //---------------------------------------------------------------
    // Each warp reads 32*THREADS_PER_WARP=1024 APP values and writes
    // 1024 bits in the form of 32 uint32_t values.
    const int WARP_IDX      = threadIdx.x / THREADS_PER_WARP;
    const int LANE          = threadIdx.x % THREADS_PER_WARP;
    const int BITS_PER_WARP = THREADS_PER_WARP * sizeof(uint32_t) * CHAR_BIT;
    //---------------------------------------------------------------
    // Check for early exit
    //const int NUM_WARPS          = (blockDim.x + 31) / 32;
    const int NUM_FULL_WARPS     = (blockDim.x / 32);
    const int NUM_FULL_WARPS_REQ = (params.num_cw_bits + BITS_PER_WARP - 1) / BITS_PER_WARP;
    const int NUM_ACTIVE_WARPS   = min(NUM_FULL_WARPS, NUM_FULL_WARPS_REQ);
    if(WARP_IDX >= NUM_ACTIVE_WARPS)
    {
        return;
    }

    int output_idx = threadIdx.x;
    int start_idx  = WARP_IDX * BITS_PER_WARP + LANE;
    do
    {
        // The output per thread.
        uint32_t output[2] = {0, 0};

        for(int ii = 0; ii < 32; ++ii)
        {
            word_t       app;
            const int    APP_IDX = start_idx + (ii * THREADS_PER_WARP);
            // Load soft decision from shared memory.
            // If index out of range, load value that is 0b as hard decision.
            app.f16x2 = __half2_raw(__float2half2_rn(1.0));
            if(APP_IDX < params.num_cw_bits)
            {
                app.f16x2 = app_smem[APP_IDX];
            }
            //word_t app_sign_mask = fp16x2_sign_mask(app);
            //unsigned int vote0 = __ballot_sync(0xffffffff, (app_sign_mask.u32 & 0x00008000));
            //unsigned int vote1 = __ballot_sync(0xffffffff, (app_sign_mask.u32 & 0x80000000));
            __half2 fp16x2(app.f16x2);
            unsigned int vote0 = __ballot_sync(0xffffffff, (llr_hard_decision(fp16x2.x)));
            unsigned int vote1 = __ballot_sync(0xffffffff, (llr_hard_decision(fp16x2.y)));
            if(LANE == ii)
            {
                output[0] = vote0;
                output[1] = vote1;
            }
        }
        // Output the result.
        if(output_idx < params.out_words_per_cw)
        {
            params.dst_gmem[output_idx] = output[0];
            // Avoid writes past the end of the output when the number of
            // codewords is odd.
            if(2 == params.num_out_cw)
            {
                params.dst_gmem[output_idx + params.out_stride_words] = output[1];
            }
        }
        // Advance
        output_idx += blockDim.x;
        start_idx  += (NUM_ACTIVE_WARPS * BITS_PER_WARP);
    } while(start_idx < params.num_cw_bits);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
static inline __device__ void ldpc_dec_output_variable(const LDPC_kernel_params& kernelParams,
                                                       const __half2*            app_smem)
{
    ldpc_dec_output_params<__half2> params(kernelParams);
    ldpc_dec_output_variable(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
static inline __device__ void ldpc_dec_output_variable(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                       const __half2*               app_smem)
{
    ldpc_dec_output_params<__half2> params(decodeDesc);
    ldpc_dec_output_variable(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable()
static inline __device__ void ldpc_dec_output_variable(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                       tb_token                     token,
                                                       const __half2*               app_smem)
{
    ldpc_dec_output_params<__half2> params(decodeDesc, token);
    ldpc_dec_output_variable(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
static inline __device__ void ldpc_dec_output_variable_loop(const LDPC_kernel_params& kernelParams,
                                                            const __half2*            app_smem)
{
    ldpc_dec_output_params<__half2> params(kernelParams);
    ldpc_dec_output_variable_loop(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
static inline __device__ void ldpc_dec_output_variable_loop(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                            const __half2*               app_smem)
{
    ldpc_dec_output_params<__half2> params(decodeDesc);
    ldpc_dec_output_variable_loop(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_variable_loop()
static inline __device__ void ldpc_dec_output_variable_loop(const cuphyLDPCDecodeDesc_t& decodeDesc,
                                                            tb_token                     token,
                                                            const __half2*               app_smem)
{
    ldpc_dec_output_params<__half2> params(decodeDesc, token);
    ldpc_dec_output_variable_loop(params, app_smem);
}

////////////////////////////////////////////////////////////////////////
// ldpc_dec_output_fixed()
// Overload of hard decision output function for kernels that process
// two codewords at a time (using fp16x2 APP values in shared memory).
// "Fixed" output functions use parameters (NODES, Z) that are known at
// compile time.
//template <typename T, int NODES, int Z>
//static inline __device__ void ldpc_dec_output_fixed(const ldpc_dec_output_params<T>& params,
//                                                    const __half2*                   app_smem)
//{
//    // The number of threads per warp.
//    enum
//    {
//        THREADS_PER_WARP = 32
//    };
//
//    // Decompose the thread indices into warp/lane.
//    const int WARP_IDX = threadIdx.x / THREADS_PER_WARP;
//    const int LANE     = threadIdx.x % THREADS_PER_WARP;
//
//    // The output per thread.
//    uint32_t output[2] = {0, 0};
//
//    // Each warp reads 32*THREADS_PER_WARP elements.
//    int idx = (WARP_IDX * 32 * THREADS_PER_WARP) + LANE;
//    for(int ii = 0; ii < 32; ++ii)
//    {
//        word_t app;
//        app.u32 = 0;
//        if(idx + (ii * THREADS_PER_WARP) < (NODES * Z))
//        {
//            app.f16x2 = app_smem[idx + (ii * THREADS_PER_WARP)];
//        }
//        word_t app_sign_mask = fp16x2_sign_mask(app);
//
//        unsigned int vote0 = __ballot_sync(0xffffffff, (app_sign_mask.u32 & 0x00008000));
//        unsigned int vote1 = __ballot_sync(0xffffffff, (app_sign_mask.u32 & 0x80000000));
//        if(LANE == ii)
//        {
//            output[0] = vote0;
//            output[1] = vote1;
//        }
//    }
//
//    // Output the result.
//    if(threadIdx.x < params.out_words_per_cw)
//    {
//        params.dst_gmem[threadIdx.x]                             = output[0];
//        // Avoid writes past the end of the output when the number of
//        // codewords is odd.
//        if(2 == params.num_out_cw)
//        {
//            params.dst_gmem[threadIdx.x + params.out_stride_words] = output[1];
//        }
//    }
//}

} // namespace ldpc2

#endif // !defined(LDPC2_DEC_OUTPUT_CUH_INCLUDED_)