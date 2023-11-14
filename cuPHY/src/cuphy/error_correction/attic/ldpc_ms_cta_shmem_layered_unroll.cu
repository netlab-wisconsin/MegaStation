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

#include "ldpc_ms_cta_shmem_layered_unroll.hpp"
#include "ldpc.cuh"
#include <float.h>

// TEMPORARY, for testing
//#include "nrLDPC_flat.cuh"

namespace
{

union word_t
{
    float    f32;
    uint32_t u32;
    int32_t  i32;
};

template <typename T> inline __device__ T smem_address_as(int oset);
//template <>           inline __device__ float smem_address_as(int oset)
//{
//    float f;
//    asm volatile("ld.shared.f32 %0, [%1];\n" : "=f"(f) : "r"(oset));
//    return f;
//}
template <>           inline __device__ word_t smem_address_as(int oset)
{
    word_t w;
    asm volatile("ld.shared.u32 %0, [%1];\n" : "=r"(w.u32) : "r"(oset));
    return w;
}

template <typename T> inline __device__ void smem_increment(int oset, T value);
template <>           inline __device__ void smem_increment(int oset, float inc)
{
    asm volatile("\t"
                 "{\n\t\t"
                 ".reg .f32 smem_value;\n\t\t"
                 "ld.shared.f32 smem_value, [%0];\n\t\t"
                 "add.f32 smem_value, smem_value, %1;\n\t\t"
                 "st.shared.f32 [%0], smem_value;\n\t"
                 "}\n"
                 : 
                 : "r"(oset), "f"(inc));
}

template <typename T> inline __device__ void smem_decrement(int oset, T value);
template <>           inline __device__ void smem_decrement(int oset, float value)
{
    asm volatile("\t"
                 "{\n\t\t"
                 ".reg .f32 smem_value;\n\t\t"
                 "ld.shared.f32 smem_value, [%0];\n\t\t"
                 "sub.f32 smem_value, smem_value, %1;\n\t\t"
                 "st.shared.f32 [%0], smem_value;\n\t"
                 "}\n"
                 : 
                 : "r"(oset), "f"(value));
}

template< int M, int N >
struct round_up {
  enum { value = (M + N-1) / N * N };
};

////////////////////////////////////////////////////////////////////////
// app_address()
// Calculate the BYTE ADDRESS (in shared memory)  of the APP value that
// maps to a thread.
// --base offset---->|
//     <---shift---->|
//     | - - - - - - - - - - - - - - - - - - - - |
//  0  |             x                           |
//  1  |               x                         |
//  2  |                 x                       |
//  3  |                   x                     |
//  4  |                     x                   |
//  5  |                       x                 |
//  6  |                         x               |
//  7  |                           x             |
//  8  |                             x           |
//  9  |                               x         |
// 10  |                                 x       |
// 11  |                                   x     |
// 12  |                                     x   |
// 13  |                                       x |
// 14  | x.......................................|<-- wrap index
// 15  |   x                                     |
// 16  |     x                                   |
// 17  |       x                                 |
// 18  |         x                               |
// 19  |           x                             |
//     | - - - - - - - - - - - - - - - - - - - - |
//
template <typename T, int Z, int COL_INDEX, int SHIFT>
inline __device__ int app_address()
{
    constexpr int BASE_OFFSET = ((COL_INDEX * Z) + SHIFT) * sizeof(T);
    constexpr int WRAP_INDEX  = (Z - SHIFT);
    int idx                   =  BASE_OFFSET + (threadIdx.x * sizeof(T));
    if(threadIdx.x >= WRAP_INDEX)
    {
        idx -= (Z * sizeof(T));
    }
    return idx;
}

template <int Z, int CHECK_INDEX> __device__ void bg1_calc_app_addresses(int (&app_addr)[19]);
template <> __device__ void bg1_calc_app_addresses<384, 0>(int (&app_addr)[19])
{
    constexpr int Z = 384;
    app_addr[0]  = app_address<float, Z, 0,  307>();
    app_addr[1]  = app_address<float, Z, 1,   19>();
    app_addr[2]  = app_address<float, Z, 2,   50>();
    app_addr[3]  = app_address<float, Z, 3,  369>();
    app_addr[4]  = app_address<float, Z, 5,  181>();
    app_addr[5]  = app_address<float, Z, 6,  216>();
    app_addr[6]  = app_address<float, Z, 9,  317>();
    app_addr[7]  = app_address<float, Z, 10, 288>();
    app_addr[8]  = app_address<float, Z, 11, 109>();
    app_addr[9]  = app_address<float, Z, 12,  17>();
    app_addr[10] = app_address<float, Z, 13, 357>();
    app_addr[11] = app_address<float, Z, 15, 215>();
    app_addr[12] = app_address<float, Z, 16, 106>();
    app_addr[13] = app_address<float, Z, 18, 242>();
    app_addr[14] = app_address<float, Z, 19, 180>();
    app_addr[15] = app_address<float, Z, 20, 330>();
    app_addr[16] = app_address<float, Z, 21, 346>();
    app_addr[17] = app_address<float, Z, 22,   1>();
    app_addr[18] = app_address<float, Z, 23,   0>();
}
template <> __device__ void bg1_calc_app_addresses<384, 1>(int (&app_addr)[19])
{
    constexpr int Z = 384;
    app_addr[0]  = app_address<float, Z, 0,   76>();
    app_addr[1]  = app_address<float, Z, 2,   76>();
    app_addr[2]  = app_address<float, Z, 3,   73>();
    app_addr[3]  = app_address<float, Z, 4,  288>();
    app_addr[4]  = app_address<float, Z, 5,  144>();
    app_addr[5]  = app_address<float, Z, 7,  331>();
    app_addr[6]  = app_address<float, Z, 8,  331>();
    app_addr[7]  = app_address<float, Z, 9,  178>();
    app_addr[8]  = app_address<float, Z, 11, 295>();
    app_addr[9]  = app_address<float, Z, 12, 342>();
    app_addr[10] = app_address<float, Z, 14, 217>();
    app_addr[11] = app_address<float, Z, 15,  99>();
    app_addr[12] = app_address<float, Z, 16, 354>();
    app_addr[13] = app_address<float, Z, 17, 114>();
    app_addr[14] = app_address<float, Z, 19, 331>();
    app_addr[15] = app_address<float, Z, 21, 112>();
    app_addr[16] = app_address<float, Z, 22,   0>();
    app_addr[17] = app_address<float, Z, 23,   0>();
    app_addr[18] = app_address<float, Z, 24,   0>();
}
template <> __device__ void bg1_calc_app_addresses<384, 2>(int (&app_addr)[19])
{
    constexpr int Z = 384;
    app_addr[0]  = app_address<float, Z, 0,  205>();
    app_addr[1]  = app_address<float, Z, 1,  250>();
    app_addr[2]  = app_address<float, Z, 2,  328>();
    app_addr[3]  = app_address<float, Z, 4,  332>();
    app_addr[4]  = app_address<float, Z, 5,  256>();
    app_addr[5]  = app_address<float, Z, 6,  161>();
    app_addr[6]  = app_address<float, Z, 7,  267>();
    app_addr[7]  = app_address<float, Z, 8,  160>();
    app_addr[8]  = app_address<float, Z, 9,   63>();
    app_addr[9]  = app_address<float, Z, 10, 129>();
    app_addr[10] = app_address<float, Z, 13, 200>();
    app_addr[11] = app_address<float, Z, 14,  88>();
    app_addr[12] = app_address<float, Z, 15,  53>();
    app_addr[13] = app_address<float, Z, 17, 131>();
    app_addr[14] = app_address<float, Z, 18, 240>();
    app_addr[15] = app_address<float, Z, 19, 205>();
    app_addr[16] = app_address<float, Z, 20,  13>();
    app_addr[17] = app_address<float, Z, 24,   0>();
    app_addr[18] = app_address<float, Z, 25,   0>();
}
template <> __device__ void bg1_calc_app_addresses<384, 3>(int (&app_addr)[19])
{
    constexpr int Z = 384;
    app_addr[0]  = app_address<float, Z, 0,  276>();
    app_addr[1]  = app_address<float, Z, 1,   87>();
    app_addr[2]  = app_address<float, Z, 3,    0>();
    app_addr[3]  = app_address<float, Z, 4,  275>();
    app_addr[4]  = app_address<float, Z, 6,  199>();
    app_addr[5]  = app_address<float, Z, 7,  153>();
    app_addr[6]  = app_address<float, Z, 8,   56>();
    app_addr[7]  = app_address<float, Z, 10, 132>();
    app_addr[8]  = app_address<float, Z, 11, 305>();
    app_addr[9]  = app_address<float, Z, 12, 231>();
    app_addr[10] = app_address<float, Z, 13, 341>();
    app_addr[11] = app_address<float, Z, 14, 212>();
    app_addr[12] = app_address<float, Z, 16, 304>();
    app_addr[13] = app_address<float, Z, 17, 300>();
    app_addr[14] = app_address<float, Z, 18, 271>();
    app_addr[15] = app_address<float, Z, 20,  39>();
    app_addr[16] = app_address<float, Z, 21, 357>();
    app_addr[17] = app_address<float, Z, 22,   1>();
    app_addr[18] = app_address<float, Z, 25,   0>();
}


template <typename T>
struct LDPC_kernel_params
{
    const char* input_llr;
    char*       out;
    int         input_llr_stride_elements;
    int         output_stride_words;
    int         max_iterations;
    int         outputs_per_codeword;       // The number of outputs/ints per codeword.

    T           inv_norm;
};

template <typename T> struct Traits;
template <>           struct Traits<float>
{
    // The type to load LLR.
    using Llr_ldg_type = float4;
    // The type to store LLR.
    using Llr_sts_type = float4;
    // The type to do the math.
    using App_type = float;
};

template <int NODES, int Z, int LLR_LDGS_>
inline __device__ void ldpc_llr(float4 (&llr)[LLR_LDGS_], const LDPC_kernel_params<float>& params)
{
    // clang-format off
    // The number of threads.
    enum { THREADS_PER_CTA = Z };
    // The number of LLR elements.
    enum { LLR_ELEMENTS = NODES * Z };
    // The number of bytes loaded by each thread per LDG -- we use LDG.128.
    enum { LLR_BYTES_PER_THREAD_PER_LDG = 16 };
    // The number of elements loaded by each thread per LDG.
    enum { LLR_ELEMENTS_PER_THREAD_PER_LDG = LLR_BYTES_PER_THREAD_PER_LDG / 4 };
    // The number of bytes loaded by the CTA per LDG.
    enum { LLR_BYTES_PER_CTA_PER_LDG = LLR_BYTES_PER_THREAD_PER_LDG * THREADS_PER_CTA };
    // The number of elements loaded by the CTA per LDG.
    enum { LLR_ELEMENTS_PER_CTA_PER_LDG = LLR_ELEMENTS_PER_THREAD_PER_LDG * THREADS_PER_CTA };
    // The number of LDGs needed to load the LLR array.
    enum { LLR_LDGS = (LLR_ELEMENTS + LLR_ELEMENTS_PER_CTA_PER_LDG-1) / LLR_ELEMENTS_PER_CTA_PER_LDG };
    // The number of elements for the last load.
    enum { LLR_REMAINING_ELEMENTS = LLR_ELEMENTS - (LLR_LDGS-1) * LLR_ELEMENTS_PER_CTA_PER_LDG };
    // clang-format on

    // Make sure the numbers match.
    static_assert(LLR_LDGS == LLR_LDGS_, "");

    // The offset in global memory for LLR elements.
    //int llr_gmem_offset = blockIdx.x*V*Z + threadIdx.x*LLR_ELEMENTS_PER_THREAD_PER_LDG;
    int llr_gmem_offset = blockIdx.x * params.input_llr_stride_elements + threadIdx.x * LLR_ELEMENTS_PER_THREAD_PER_LDG;

// Issue the loads to read LLR elements from global memory. Stage data in registers.
#pragma unroll
    for(int ii = 0; ii < LLR_LDGS - 1; ++ii)
    {
        const int imm    = ii * LLR_BYTES_PER_CTA_PER_LDG;
        int       offset = llr_gmem_offset * sizeof(float) + imm;
        llr[ii]          = *reinterpret_cast<const float4*>(&params.input_llr[offset]);
    }

    // Deal with the last (possibly) incomplete LDG.
    if(threadIdx.x * LLR_ELEMENTS_PER_THREAD_PER_LDG < LLR_REMAINING_ELEMENTS)
    {
        const int imm     = (LLR_LDGS - 1) * LLR_BYTES_PER_CTA_PER_LDG;
        int       offset  = llr_gmem_offset * sizeof(float) + imm;
        llr[LLR_LDGS - 1] = *reinterpret_cast<const float4*>(&params.input_llr[offset]);
    }

// Apply the normalization.
#pragma unroll
    for(int ii = 0; ii < LLR_LDGS; ++ii)
    {
        llr[ii].x *= params.inv_norm;
        llr[ii].y *= params.inv_norm;
        llr[ii].z *= params.inv_norm;
        llr[ii].w *= params.inv_norm;
    }
}

template <int NODES, int Z>
static inline __device__ void ldpc_output(const LDPC_kernel_params<float>& params, const float* app_smem, int offset)
{
    // The number of threads per warp.
    enum
    {
        THREADS_PER_WARP = 32
    };

    // Decompose the thread indices into warp/lane.
    int warp = threadIdx.x / THREADS_PER_WARP;
    int lane = threadIdx.x % THREADS_PER_WARP;

    // The output per thread.
    int output = 0;

    // Each warp reads 32*THREADS_PER_WARP elements.
    int idx = warp * 32 * THREADS_PER_WARP + lane;
    for(int ii = 0; ii < 32; ++ii)
    {
        float app = 0.f;
        if(idx + ii * THREADS_PER_WARP < NODES * Z)
        {
            app = app_smem[idx + ii * THREADS_PER_WARP];
        }

        int vote = __ballot_sync(0xffffffff, signbit(app));
        if(lane == ii)
        {
            output = vote;
        }
    }

    // Output the result.
    //int gmem_out_offset = blockIdx.x*params.outputs_per_codeword + offset;
    int gmem_out_offset = blockIdx.x * params.output_stride_words + offset;
    if(offset < params.outputs_per_codeword)
    {
        //KERNEL_PRINT_BLOCK_ONCE("blockIdx.x = %u, threadIdx.x = %u, offset = %i, gmem_out_offset = %i, output = 0x%X\n",
        //                        blockIdx.x, threadIdx.x, offset, gmem_out_offset, output);
        reinterpret_cast<int*>(params.out)[gmem_out_offset] = output;
    }
}


}

////////////////////////////////////////////////////////////////////////
// sign_mask()
// Returns a word with the sign bit of the input values, and all other
// bits 0.
__device__
word_t sign_mask(word_t w)
{
    word_t out;
    out.u32 = w.u32 & 0x80000000;
    return out;
}

__device__
void write_word(word_t w, int offset)
{
    asm volatile("st.shared.b32 [%0], %1;\n" :: "r"(offset), "r"(w.u32));
}

template <typename T> struct cC2V;

////////////////////////////////////////////////////////////////////////
// cC2V
// Compressed "check to variable" representation
template <>
struct cC2V<float>
{
    //------------------------------------------------------------------
    // init()
    // Initialization function
    __device__
    void init(word_t v0, int address0, word_t v1, int address1)
    {
        min0          = (fabsf(v0.f32) <= fabsf(v1.f32)) ? v0       : v1;
        min1_or_delta = (fabsf(v0.f32) <= fabsf(v1.f32)) ? v1       : v0;
        min0_address  = (fabsf(v0.f32) <= fabsf(v1.f32)) ? address0 : address1;
        word_t smask0 = sign_mask(v0);
        word_t smask1 = sign_mask(v1);
        signs         = (smask0.u32 >> 31) | (smask1.u32 >> 30);
    }
    //------------------------------------------------------------------
    // update()
    __device__
    void update(word_t v, int address, int index)
    {
        if(fabsf(v.f32) < fabsf(min0.f32))
        {
            // Note: storing values for min0 and min1, instead of
            // absolute values
            min1_or_delta = min0;
            min0          = v;
            min0_address  = address;
        }
        else if(fabsf(v.f32) < fabsf(min1_or_delta.f32))
        {
            // Note: storing value for min1, instead of
            // absolute value
            min1_or_delta = v;
        }
        word_t smask = sign_mask(v);
        signs |= (smask.u32 >> (31 - index));
    }
    //------------------------------------------------------------------
    // finalize()
    __device__
    void finalize(int row_count)
    {
        // The signs member has the current signs of each index
        // in the row. When we expand, we want the product of
        // signs, but without a specific index. We convert here
        // to what the actual sign will be, to simplify that
        // operation, by taking the XOR with a bitmask that has
        // each bit set to the overall parity (1s if odd, 0s if even).
        // For example, consider a row with 19 elements and the
        // following signs:
        //  18  16        12         8         4         0
        // x 0 0 1 | 1 0 1 0 | 1 0 1 0 | 0 0 0 0 | 1 1 1 1
        // The number of 1s is odd, so popc(signs) & 0x1 = 1,
        // and after shifting and subtracting we get:
        //  18  16        12         8         4         0
        // x 1 1 1 | 1 1 1 1 | 1 1 1 1 | 1 1 1 1 | 1 1 1 1
        // Taking the XOR with the original signs provides the
        // product of signs that will occur WITHOUT that specific
        // value:
        //  18  16        12         8         4         0
        // x 1 1 0 | 0 1 0 1 | 0 1 0 1 | 1 1 1 1 | 0 0 0 0
        const uint32_t signs_odd     = (__popc(signs) & 0x1);
        uint32_t       even_odd_mask = 0;
        if(signs_odd != 0)
        {
            even_odd_mask = (1 << row_count) - 1;
        }
        signs = even_odd_mask ^ signs;

        // We want to determine what to add to min0 to get min1,
        // taking the sign into account.
        // min0 + delta = min1
        // At this point, the sign of min0 is the "correct"
        // sign, i.e. the sign that we want when we expand the
        // compressed C2V value.
        // Note that at this point, min1 may be negative
        // or positive - it will be whatever the value was
        // when encountered during the update() function.
        // If min0 > 0:
        // <------------|---------|------|------->
        //              0       min0  abs(min1)
        //     delta = abs(min1) - min0
        // If min0 < 0:
        // <------------|---------|------|------->
        //        -abs(min1)     min0    0
        //     delta = -abs(min1) - min0
        //           = -(abs(min1) + min0)
        //           = -(abs(min1) - abs(min0))
        // So to handle both cases:
        // delta = [abs(min1) - abs(min0)] * sign(min0)
        //     where sign(min0) = +1 or -1
        min1_or_delta.f32 = fabsf(min1_or_delta.f32) - fabsf(min0.f32);
        min1_or_delta.u32 = min1_or_delta.u32 | ((signs_odd << 31) ^ sign_mask(min0).u32);
        // Store the absolute value of min0
        min0.f32 = fabsf(min0.f32);
    }
    //------------------------------------------------------------------
    // process_row_init()
    template <int ROW_LENGTH, int MAX_ROW_LENGTH>
    __device__
    void process_row_init(word_t (&app)[MAX_ROW_LENGTH],
                          int    (&app_addr)[MAX_ROW_LENGTH])
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Fetch app values and update the min0/min1/address fields
        #pragma unroll
        for(int i = 0; i < ROW_LENGTH; ++i)
        {
            // Load channel APP from given address
            app[i]  = smem_address_as<word_t>(app_addr[i]);
            if(1 == i)
            {
                // Initialize with first two values [0, 1]
                init(app[0], app_addr[0], app[1], app_addr[1]);
            }
            else if(i > 1)
            {
                // Update with subsequent values
                update(app[i], app_addr[i], i);
            }
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Post-process min0/min1/address fields
        finalize(ROW_LENGTH);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #pragma unroll
        for(int i = 0; i < ROW_LENGTH; ++i)
        {
            // Get the value, assuming that this isn't the min0 value.
            // (We'll account for that case below.)
            word_t inc = value_not_minimum(i);
            app[i].f32 += inc.f32;
            write_word(app[i], app_addr[i]);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Scatter an update to the min0 value so that it is now min1
        smem_increment(min0_address, min1_or_delta.f32);
    }
    //------------------------------------------------------------------
    // process_row()
    template <int ROW_LENGTH, int MAX_ROW_LENGTH>
    __device__
    void process_row(const cC2V& c2vInput,
                     word_t      (&app)[MAX_ROW_LENGTH],
                     int         (&app_addr)[MAX_ROW_LENGTH])
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Subtract the difference between min0 and min1 from the
        // appropriate address. Below, we will subtract min0 from ALL
        // APP values. By doing it this way, we can avoid comparing
        // each index to the min0 index.
        smem_decrement(c2vInput.min0_address, c2vInput.min1_or_delta.f32);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #pragma unroll
        for(int i = 0; i < ROW_LENGTH; ++i)
        {
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Load channel APP from given address
            app[i]  = smem_address_as<word_t>(app_addr[i]);
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Subtract the contribution from this check node, from the
            // previous iteration
            word_t dec = c2vInput.value_not_minimum(i);
            app[i].f32 -= dec.f32;
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Use the APP value to update the new C2V
            if(1 == i)
            {
                // Initialize with first two values [0, 1]
                init(app[0], app_addr[0], app[1], app_addr[1]);
            }
            else if(i > 1)
            {
                // Update with subsequent values
                update(app[i], app_addr[i], i);
            }
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Post-process min0/min1/address fields
        finalize(ROW_LENGTH);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        #pragma unroll
        for(int i = 0; i < ROW_LENGTH; ++i)
        {
            // Get the value, assuming that this isn't the min0 value.
            // (We'll account for that case below.)
            word_t inc = value_not_minimum(i);
            app[i].f32 += inc.f32;
            write_word(app[i], app_addr[i]);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Scatter an update to the min0 value so that it is now min1
        smem_increment(min0_address, min1_or_delta.f32);
    }
    //------------------------------------------------------------------
    // value_not_minimum()
    // Must only be called after finalize() function. (This function
    // assumes that the signs member has the "desired" sign, whereas
    // before the finalize() function is called, it contains the "input"
    // sign.)
    // The value assumes that the index is NOT the min0 index. For the
    // min0 index, we need to also add/subtract the difference between
    // min0 and min1.
    __device__
    word_t value_not_minimum(int index) const
    {
        const uint32_t sign_mask = signs << (31 - index);
        word_t         out;
        out.u32 = (sign_mask & 0x80000000) | min0.u32;
        return out;
    }
    //------------------------------------------------------------------
    // Data
    word_t    min0;
    word_t    min1_or_delta;
    uint32_t  signs;
    int       min0_address;
};

template <typename T, int C, int V, int Z>
__global__ __launch_bounds__(round_up<Z, 32>::value, 1)
void ldpc_layered_unroll(LDPC_kernel_params<T> params)
{
    // clang-format off
    // The type to load LLR.
    using Llr_ldg_type = typename Traits<T>::Llr_ldg_type;
    // The type to store LLR to shared memory.
    using Llr_sts_type = typename Traits<T>::Llr_sts_type;
    // The type to store APP in shared memory.
    using App_type = typename Traits<T>::App_type;

    // The number of threads per CTA.
    enum { THREADS_PER_WARP = 32, THREADS_PER_CTA = Z };

    // The number of LLR elements.
    enum { LLR_ELEMENTS = V * Z };
    // The number of bytes loaded by each thread per LDG -- we use LDG.128.
    enum { LLR_BYTES_PER_THREAD_PER_LDG = sizeof(Llr_ldg_type) };
    // The number of elements loaded by each thread per LDG.
    enum { LLR_ELEMENTS_PER_THREAD_PER_LDG = LLR_BYTES_PER_THREAD_PER_LDG / sizeof(T) };
    // The number of bytes loaded by the CTA per LDG.
    enum { LLR_BYTES_PER_CTA_PER_LDG = LLR_BYTES_PER_THREAD_PER_LDG * THREADS_PER_CTA };
    // The number of elements loaded by the CTA per LDG.
    enum { LLR_ELEMENTS_PER_CTA_PER_LDG = LLR_ELEMENTS_PER_THREAD_PER_LDG * THREADS_PER_CTA };
    // The number of LDGs needed to load the LLR array.
    enum { LLR_LDGS = (LLR_ELEMENTS + LLR_ELEMENTS_PER_CTA_PER_LDG-1) / LLR_ELEMENTS_PER_CTA_PER_LDG };
    // The number of elements for the last load.
    enum { LLR_REMAINING_ELEMENTS = LLR_ELEMENTS - (LLR_LDGS-1) * LLR_ELEMENTS_PER_CTA_PER_LDG };

    // The number of bytes loaded by each thread per STS.
    enum { LLR_BYTES_PER_THREAD_PER_STS = sizeof(Llr_sts_type) };
    // The number of elements loaded by each thread per STS.
    enum { LLR_ELEMENTS_PER_THREAD_PER_STS = LLR_BYTES_PER_THREAD_PER_STS / sizeof(T) };
    // The number of bytes loaded by the CTA per STS.
    enum { LLR_BYTES_PER_CTA_PER_STS = LLR_BYTES_PER_THREAD_PER_STS * THREADS_PER_CTA };
    // The number of elements loaded by the CTA per STS.
    enum { LLR_ELEMENTS_PER_CTA_PER_STS = LLR_ELEMENTS_PER_THREAD_PER_STS * THREADS_PER_CTA };
    // clang-format on
    //------------------------------------------------------------------
    __shared__ char smem_[LLR_ELEMENTS * sizeof(App_type)];
    App_type* app_smem = reinterpret_cast<App_type*>(smem_);
    //------------------------------------------------------------------
    // Copy LLR data from global memory to registers.
    Llr_sts_type llr[LLR_LDGS];
    ldpc_llr<V, Z>(llr, params);

    // The offset in shared memory for LLR elements.
    int llr_smem_offset = threadIdx.x * LLR_ELEMENTS_PER_THREAD_PER_STS;

// Copy the LLR elements to shared memory.
#pragma unroll
    for(int ii = 0; ii < LLR_LDGS - 1; ++ii)
    {
        const int imm                                                                 = ii * LLR_BYTES_PER_CTA_PER_STS;
        reinterpret_cast<Llr_sts_type*>(&smem_[llr_smem_offset * sizeof(T) + imm])[0] = llr[ii];
    }

    // Deal with the last (possibly) incomplete LDG.
    if(threadIdx.x * LLR_ELEMENTS_PER_THREAD_PER_LDG < LLR_REMAINING_ELEMENTS)
    {
        const int imm                                                                 = (LLR_LDGS - 1) * LLR_BYTES_PER_CTA_PER_STS;
        reinterpret_cast<Llr_sts_type*>(&smem_[llr_smem_offset * sizeof(T) + imm])[0] = llr[LLR_LDGS - 1];
    }

    // Make sure the data is in shared memory.
    __syncthreads();

    int         app_addr[19];  // shared memory (byte) addresses
    word_t      app[19];       // APP values
    cC2V<float> c2v[4];        // C2V messages
    //------------------------------------------------------------------
    // Iteration 0 (no previous C2V messages)
    //------------------------------------------------------------------
    // Check Node 0
    bg1_calc_app_addresses<Z, 0>(app_addr);
    c2v[0].process_row_init<19, 19>(app, app_addr);
    __syncthreads();
    //------------------------------------------------------------------
    // Check Node 1
    bg1_calc_app_addresses<Z, 1>(app_addr);
    c2v[1].process_row_init<19, 19>(app, app_addr);
    __syncthreads();
    //------------------------------------------------------------------
    // Check Node 2
    bg1_calc_app_addresses<Z, 2>(app_addr);
    c2v[2].process_row_init<19, 19>(app, app_addr);
    __syncthreads();
    //------------------------------------------------------------------
    // Check Node 3
    bg1_calc_app_addresses<Z, 3>(app_addr);
    c2v[3].process_row_init<19, 19>(app, app_addr);
    __syncthreads();
    //------------------------------------------------------------------
    // Iterations 1 through (N-1)
    for(int iter = 1; iter < params.max_iterations; ++iter)
    {
        cC2V<float> c2vNew;
        bg1_calc_app_addresses<Z, 0>(app_addr);
        c2vNew.process_row<19, 19>(c2v[0], app, app_addr);
        c2v[0] = c2vNew;
        __syncthreads();
        bg1_calc_app_addresses<Z, 1>(app_addr);
        c2vNew.process_row<19, 19>(c2v[1], app, app_addr);
        c2v[1] = c2vNew;
        __syncthreads();
        bg1_calc_app_addresses<Z, 2>(app_addr);
        c2vNew.process_row<19, 19>(c2v[2], app, app_addr);
        c2v[2] = c2vNew;
        __syncthreads();
        bg1_calc_app_addresses<Z, 3>(app_addr);
        c2vNew.process_row<19, 19>(c2v[3], app, app_addr);
        c2v[3] = c2vNew;
        __syncthreads();
    }
    //------------------------------------------------------------------
    // Write output based on APP values
    ldpc_output<V - C, Z>(params, app_smem, threadIdx.x);
}


namespace ldpc
{
////////////////////////////////////////////////////////////////////////
// decode_ms_cta_shmem_layered_unroll()
cuphyStatus_t decode_ms_cta_shmem_layered_unroll(LDPC_output_t&         tDst,
                                                 const_tensor_pair&     tLLR,
                                                 const LDPC_config&     config,
                                                 float                  normalization,
                                                 cuphyLDPCResults_t*    results,
                                                 void*                  workspace,
                                                 cuphyLDPCDiagnostic_t* diag,
                                                 cudaStream_t           strm)
{
    DEBUG_PRINTF("ldpc::decode_ms_cta_shmem_layered_unroll()\n");
    //------------------------------------------------------------------
    //const int       VNODES  = config.Kb + config.mb;
    cuphyDataType_t llrType = tLLR.first.get().type();
    // TODO: Examine expanding to other configurations
    if((config.Kb != 22) || (config.mb != 4) || (config.Z != 384))
    {
        return CUPHY_STATUS_NOT_SUPPORTED;
    }
    //------------------------------------------------------------------
    dim3 grdDim(config.num_codewords); // 1-D, NUM_CW for fp32, NUM_CW / 2 for fp16
    dim3 blkDim(config.Z);
    //------------------------------------------------------------------
    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc_layered_unroll<float, 4, 26, 384>));
    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc_layered_unroll<float, 4, 26, 384>), blkDim, 0);

    LDPC_kernel_params<float> params;
    // Note: Using generic tensor layout here, even though only
    // contiguous is supported. Use strides[0] after conversion to
    // contiguous layout.
    params.input_llr_stride_elements = tLLR.first.get().layout().strides[1];
    params.input_llr                 = (const char*)tLLR.second;
    params.output_stride_words       = tDst.layout().strides[0];;
    params.max_iterations            = config.max_iterations;;
    params.inv_norm                  = 1.f / normalization;
    params.outputs_per_codeword      = ((config.Kb * config.Z) + 31) / 32;
    params.out                       = (char*)tDst.addr();
    ldpc_layered_unroll<float, 4, 26, 384><<<grdDim, blkDim, 0, strm>>>(params);
    
#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

//----------------------------------------------------------------------
// decode_ms_cta_shmem_layered_unroll_workspace_size()
std::pair<bool, size_t> decode_ms_cta_shmem_layered_unroll_workspace_size(const LDPC_config& cfg)
{
    // TODO: Develop a more rigorous test to determine whether or not
    // this implementation is supported.
    if(cfg.mb <= 4)
    {
        // No global workspace data required
        return std::pair<bool, size_t>(true, 0);
    }
    else
    {
        return std::pair<bool, size_t>(true, 0);
    }
}

} // namespace ldpc
