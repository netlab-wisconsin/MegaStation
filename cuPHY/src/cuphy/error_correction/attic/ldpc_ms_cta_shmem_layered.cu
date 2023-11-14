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

#include "ldpc_ms_cta_layered.hpp"
#include "ldpc.cuh"
#include <float.h>
#include "nrLDPC_flat.cuh"

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint2 ldg64(const void* ptr)
{
    return *reinterpret_cast<const uint2*>(ptr);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

// static inline __device__ float ffma(float a, float b, float c) {
//   return a*b + c;
// }

////////////////////////////////////////////////////////////////////////////////////////////////////

// static inline __device__ float fltu(float a, float b) {
//   float c;
//   asm volatile("set.ltu.f32.f32 %0, %1, %2;\n" : "=f"(c) : "f"(a), "f"(b));
//   return c;
// }

////////////////////////////////////////////////////////////////////////////////////////////////////

//static inline __device__ int float2_to_half2(float a, float b)
//{
//    int c;
//    asm volatile(
//        "{\n"
//        "    .reg .f16 lo, hi;\n"
//        "    cvt.rn.f16.f32 lo, %1;\n"
//        "    cvt.rn.f16.f32 hi, %2;\n"
//        "    mov.b32 %0, {lo, hi};\n"
//        "}\n"
//        : "=r"(c)
//        : "f"(a), "f"(b));
//    return c;
//}

////////////////////////////////////////////////////////////////////////////////////////////////////

//static inline __device__ void half2_to_float2(float& a, float& b, int c)
//{
//    asm volatile(
//        "{\n"
//        "    .reg .f16 lo, hi;\n"
//        "    mov.b32 {lo, hi}, %2;\n"
//        "    cvt.f32.f16 %0, lo;\n"
//        "    cvt.f32.f16 %1, hi;\n"
//        "}\n"
//        : "=f"(a), "=f"(b)
//        : "r"(c));
//}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hadd2(uint32_t a, uint32_t b)
{
    uint32_t c;
    asm volatile("add.f16x2 %0, %1, %2;\n"
                 : "=r"(c)
                 : "r"(a), "r"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hequ2(uint32_t a, uint32_t b)
{
    uint32_t c;
    asm volatile("set.equ.f16x2.f16x2 %0, %1, %2;\n"
                 : "=r"(c)
                 : "r"(a), "r"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c)
{
    uint32_t d;
    asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n"
                 : "=r"(d)
                 : "r"(a), "r"(b), "r"(c));
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t h0_h0(uint32_t a)
{
    // It should map directly to Rx.H0_H0 in SASS!
    return a + (a << 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hltu2(uint32_t a, uint32_t b)
{
    uint32_t c;
    asm volatile("set.ltu.f16x2.f16x2 %0, %1, %2;\n"
                 : "=r"(c)
                 : "r"(a), "r"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hmul2(uint32_t a, uint32_t b)
{
    uint32_t c;
    asm volatile("mul.f16x2 %0, %1, %2;\n"
                 : "=r"(c)
                 : "r"(a), "r"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hneg2(uint32_t a)
{
    uint32_t b;
    asm volatile("neg.f16x2 %0, %1;\n"
                 : "=r"(b)
                 : "r"(a));
    return b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hsub2(uint32_t a, uint32_t b)
{
    uint32_t c;
    asm volatile("sub.f16x2 %0, %1, %2;\n"
                 : "=r"(c)
                 : "r"(a), "r"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float add(float a, float b)
{
    return a + b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t add(uint32_t a, uint32_t b)
{
    return hadd2(a, b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int SLICES>
static inline __device__ int compute_c_offset(int ci, int slice)
{
    if(SLICES == 1)
    {
        return ci;
    }
    else
    {
        return ci * SLICES + slice;
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int Z>
static inline __device__ int compute_app_offset(const ldpc_column_info_t& node, int zi)
{
    int offset = zi + node.shift;
    if(offset >= Z)
    {
        offset -= Z;
    }
    return node.index * Z + offset;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int Z>
static inline __device__ int2 compute_app_offset_x2(const ldpc_column_info_t& node, int zi)
{
    int col[2];
#pragma unroll
    for(int ii = 0; ii < 2; ++ii)
    {
        col[ii] = zi + ii * (Z / 2) + node.shift;
        if(col[ii] >= Z)
        {
            col[ii] -= Z;
        }
    }
    const int row = node.index;
    return make_int2(row * Z + col[0], row * Z + col[1]);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float sub(float a, float b)
{
    return a - b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t sub(uint32_t a, uint32_t b)
{
    return hsub2(a, b);
}

struct Ldpc_params
{
    // The input LLR (floats).
    const char* llr;
    // Stride (in elements) between inputs for consecutive codewords
    int llr_stride_elements;
    // The outputs (ints).
    char* out;
    // Stride (in words) between outputs for consecutive codewords
    int output_stride_words;
    // The number of outputs/ints per codeword.
    int outputs_per_codeword;
    // The number of iterations.
    int max_iterations;
    // The normalization factor.
    uint32_t inv_norm;
    // The number of loops over C.
    int loop_c;

    // The C2V messages. (Used only in spilling variants.)
    //char *c2v4, *c2v2;
    // The precomputed values for C2V spills. (Used only in spilling variants.)
    //int cz4, z4, cz2, z2;
};

template <typename T>
struct Compressed_c2v
{
};

template <typename T>
struct Traits
{
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Compressed_c2v<float>
{
    // Ctor.
    inline __device__ Compressed_c2v() :
        min0_(FLT_MAX),
        min1_(FLT_MAX),
        idx_(),
        sign_(0)
    {
    }

    // All updates were performed.
    inline __device__ void finalize_updates()
    {
        // odd_ = __popc(sign_) & 0x1;
        odd_ = __popc(sign_) & 0x1 ? -1.f : 1.f;
    }

    // Load a C2V message.
    //inline __device__ void load(const Ldpc_params &params, int ci, int zi) {
    //  const int offset2 = blockIdx.x*params.cz2 + ci*params.z2 + zi*sizeof(int2);
    //  int2 data2 = reinterpret_cast<const int2*>(&params.c2v2[offset2])[0];
    //
    //  // Extract the min0/min1 values.
    //  half2_to_float2(min0_, min1_, data2.x);
    //
    //  // Unpack idx/sign.
    //  idx_  = (data2.y & 0x000000ff);
    //  sign_ = (data2.y & 0xffffff00) >> 8;
    //
    //  // Rebuild the odd value.
    //  this->finalize_updates();
    //}

    // Store a C2V message.
    //inline __device__ void store(const Ldpc_params &params, int ci, int zi) const {
    //  const int offset2 = blockIdx.x*params.cz2 + ci*params.z2 + zi*sizeof(int2);
    //  int2 data2;
    //
    //  // Pack min0/min1 as 2x fp16s.
    //  data2.x = float2_to_half2(min0_, min1_);
    //  // Pack the sign and idx into a single 32b value.
    //  data2.y = (sign_ << 8) | idx_;

    //  // Store to memory.
    //  reinterpret_cast<int2*>(&params.c2v2[offset2])[0] = data2;
    //}

    // Update the compressed message using on V2C value.
    inline __device__ void update(int vi, float v2c)
    {
        float abs = fabsf(v2c);

        // // Is abs smaller than min0/min1?
        // float p0 = fltu(abs, min0_);
        // float p1 = fltu(abs, min1_);

        // // Update min0 and min1 based where abs can be inserted.
        // min1_ = ffma(p1, abs  , ffma(-p1, min1_, min1_));
        // min1_ = ffma(p0, min0_, ffma(-p0, min1_, min1_));
        // min0_ = ffma(p0, abs  , ffma(-p0, min0_, min0_));

        // // Update the index.
        // idx_ = ffma(p0, vi, ffma(-p0, idx_, idx_));

        if(abs < min0_)
        {
            min1_ = min0_;
            min0_ = abs;
            idx_  = vi;
        }
        else if(abs < min1_)
        { // Insert the value between min0 and min1.
            min1_ = abs;
        }

        // Update the sign.
        sign_ += signbit(v2c) << vi;
    }

    // Compute the value min * sign where min/sign are over all the neighbours \ vi.
    inline __device__ float value(int vi)
    {
        // Find the min. It should be FSETP + SEL.
        float min = vi == idx_ ? min1_ : min0_;
        // Determine the sign of the V2C message and move the bit to the MSB of an int.
        int sgn = (sign_ << (31 - vi)) & 0x80000000;
        // The toggle the sign bit of odd_ if needed.
        int odd = reinterpret_cast<int&>(odd_) ^ sgn;
        // The C2V value is now easy to compute.
        return min * reinterpret_cast<float&>(odd);
    }

    // The smallest absolute values.
    float min0_, min1_;
    // The index of the smallest value.
    int idx_;
    // The sign bits (1: negative, 0: positive).
    uint32_t sign_;
    // The record if the number of bits is odd.
    float odd_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Traits<float>
{
    // The type to load LLR.
    using Llr_ldg_type = float4;
    // The type to store LLR.
    using Llr_sts_type = float4;
    // The type to do the math.
    using App_type = float;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Compressed_c2v<uint16_t>
{
    // Ctor.
    inline __device__ Compressed_c2v() :
        min0_(0x7bff7bffu),
        min1_(0x7bff7bffu),
        idx_(0),
        sign_{0, 0}
    {
    }

    // All updates were performed.
    inline __device__ void finalize_updates()
    {
        odd_ = (__popc(sign_[0]) & 0x1) ? 0x0000bc00u : 0x00003c00u;
        odd_ |= (__popc(sign_[1]) & 0x1) ? 0xbc000000u : 0x3c000000u;
    }

    // Load a C2V message.
    //inline __device__ void load(const Ldpc_params &params, int ci, int zi) {
    //  const int offset4 = blockIdx.x*params.cz4 + ci*params.z4 + zi*sizeof(int4);
    //  int4 data4 = reinterpret_cast<const int4*>(&params.c2v4[offset4])[0];
    //  min0_    = reinterpret_cast<uint32_t&>(data4.x);
    //  min1_    = reinterpret_cast<uint32_t&>(data4.y);
    //  sign_[0] = data4.z >> 8;
    //  sign_[1] = data4.w >> 8;
    //
    //  // Repack the index as 2x denormalized fp16s.
    //  idx_ = ((data4.w & 0xff) << 16) | (data4.z & 0xff);
    //
    //  // Rebuild the odd_ value.
    //  this->finalize_updates();
    //}

    // Store a C2V message.
    //inline __device__ void store(const Ldpc_params &params, int ci, int zi) const {
    //  const int offset4 = blockIdx.x*params.cz4 + ci*params.z4 + zi*sizeof(int4);
    //  int4 data4;
    //  data4.x = reinterpret_cast<const int&>(min0_);
    //  data4.y = reinterpret_cast<const int&>(min1_);
    //  data4.z = (sign_[0] << 8) | ((idx_ >>  0) & 0xff);
    //  data4.w = (sign_[1] << 8) | ((idx_ >> 16) & 0xff);
    //  reinterpret_cast<int4*>(&params.c2v4[offset4])[0] = data4;
    //}

    // Update the compressed message using on V2C value.
    inline __device__ void update(int vi, uint32_t v2c)
    {
        // Extract the values without the sign bit.
        uint32_t abs = v2c & 0x7fff7fffu;

        // Is abs smaller than min0/min1?
        uint32_t p0 = hltu2(abs, min0_);
        uint32_t p1 = hltu2(abs, min1_);

        // Update min0 and min1 based where abs can be inserted.
        min1_ = hfma2(p1, abs, hfma2(hneg2(p1), min1_, min1_));
        min1_ = hfma2(p0, min0_, hfma2(hneg2(p0), min1_, min1_));
        min0_ = hfma2(p0, abs, hfma2(hneg2(p0), min0_, min0_));

        // Update the index -- we expect h0_h0 to map to .H0_H0 in SASS.
        idx_ = hfma2(p0, h0_h0(vi), hfma2(hneg2(p0), idx_, idx_));

        // Update the array of signs.
        sign_[0] |= ((v2c & 0x00008000u) >> 15) << vi;
        sign_[1] |= ((v2c & 0x80000000u) >> 31) << vi;
    }

    // Compute the value min * sign where min/sign are over all the neighbours \ vi.
    inline __device__ uint32_t value(int vi)
    {
        // Expand vi as 2x denorm FP16. We would want to use .H0_H0 in SASS.
        uint32_t vi2 = h0_h0(vi);
        // The predicate to determine if we pick the min or the 2nd min.
        uint32_t p = hequ2(idx_, vi2);
        // The min.
        uint32_t min = hfma2(p, min1_, hfma2(hneg2(p), min0_, min0_));

        // Determine the sign of the V2C message.
        uint32_t sgn = odd_;

        // Create a vector with the correct sign bits.
        sgn ^= ((sign_[0] >> vi) & 0x1) << 15;
        sgn ^= ((sign_[1] >> vi) & 0x1) << 31;

        // The C2V value is now easy to compute.
        return hmul2(min, sgn);
    }

    // The smallest absolute values. Each int is 2x fp16s.
    uint32_t min0_, min1_;
    // The indices of the smallest value. We store the indices in two denorm FP16s.
    uint32_t idx_;
    // The sign bits (1: negative, 0: positive).
    int sign_[2];
    // Is the number of bits odd?.
    uint32_t odd_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template <>
struct Traits<uint16_t>
{
    // The type to load LLR.
    using Llr_ldg_type = uint2;
    // The type to store LLR.
    using Llr_sts_type = uint4;
    // The type to do the math.
    using App_type = uint32_t;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

extern __shared__ char smem_[];

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NODES, int V, int Z, int LLR_LDGS_>
static inline __device__ void ldpc_llr(float4 (&llr)[LLR_LDGS_], const Ldpc_params& params)
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
    int llr_gmem_offset = blockIdx.x * params.llr_stride_elements + threadIdx.x * LLR_ELEMENTS_PER_THREAD_PER_LDG;

// Issue the loads to read LLR elements from global memory. Stage data in registers.
#pragma unroll
    for(int ii = 0; ii < LLR_LDGS - 1; ++ii)
    {
        const int imm    = ii * LLR_BYTES_PER_CTA_PER_LDG;
        int       offset = llr_gmem_offset * sizeof(float) + imm;
        llr[ii]          = *reinterpret_cast<const float4*>(&params.llr[offset]);
    }

    // Deal with the last (possibly) incomplete LDG.
    if(threadIdx.x * LLR_ELEMENTS_PER_THREAD_PER_LDG < LLR_REMAINING_ELEMENTS)
    {
        const int imm     = (LLR_LDGS - 1) * LLR_BYTES_PER_CTA_PER_LDG;
        int       offset  = llr_gmem_offset * sizeof(float) + imm;
        llr[LLR_LDGS - 1] = *reinterpret_cast<const float4*>(&params.llr[offset]);
    }

// Apply the normalization.
#pragma unroll
    for(int ii = 0; ii < LLR_LDGS; ++ii)
    {
        llr[ii].x *= reinterpret_cast<const float&>(params.inv_norm);
        llr[ii].y *= reinterpret_cast<const float&>(params.inv_norm);
        llr[ii].z *= reinterpret_cast<const float&>(params.inv_norm);
        llr[ii].w *= reinterpret_cast<const float&>(params.inv_norm);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint4 interleave_llr(uint2 llr0, uint2 llr1)
{
    uint4 d;
    asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n"
                 : "=r"(d.x)
                 : "r"(llr0.x), "r"(llr1.x));
    asm volatile("prmt.b32 %0, %1, %2, 0x7632;\n"
                 : "=r"(d.y)
                 : "r"(llr0.x), "r"(llr1.x));
    asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n"
                 : "=r"(d.z)
                 : "r"(llr0.y), "r"(llr1.y));
    asm volatile("prmt.b32 %0, %1, %2, 0x7632;\n"
                 : "=r"(d.w)
                 : "r"(llr0.y), "r"(llr1.y));
    return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NODES, int V, int Z, int LLR_LDGS_>
static inline __device__ void ldpc_llr(uint4 (&llr)[LLR_LDGS_], const Ldpc_params& params)
{
    // clang-format off
    // The number of threads.
    enum { THREADS_PER_CTA = Z };
    // The number of LLR elements.
    enum { LLR_ELEMENTS = NODES * Z };
    // The number of bytes loaded by each thread per LDG -- we use LDG.128.
    enum { LLR_BYTES_PER_THREAD_PER_LDG = 8 };
    // The number of elements loaded by each thread per LDG.
    enum { LLR_ELEMENTS_PER_THREAD_PER_LDG = LLR_BYTES_PER_THREAD_PER_LDG / 2 };
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
    //int llr_gmem_offset = blockIdx.x*V*Z*2 + threadIdx.x*LLR_ELEMENTS_PER_THREAD_PER_LDG;
    int llr_gmem_offset = blockIdx.x * params.llr_stride_elements * 2 + threadIdx.x * LLR_ELEMENTS_PER_THREAD_PER_LDG;

    // Issue the loads to read LLR elements from global memory. Stage data in registers.
    uint2 llr_[2][LLR_LDGS];
#pragma unroll
    for(int ii = 0; ii < 2; ++ii)
    {
        //const int base = ii*V*Z * sizeof(uint16_t);
        const int base = ii * params.llr_stride_elements * sizeof(uint16_t);
#pragma unroll
        for(int jj = 0; jj < LLR_LDGS - 1; ++jj)
        {
            const int imm = base + jj * LLR_BYTES_PER_CTA_PER_LDG;
            llr_[ii][jj]  = ldg64(&params.llr[llr_gmem_offset * sizeof(uint16_t) + imm]);
        }

        // Deal with the last (possibly) incomplete LDG.
        if(threadIdx.x * LLR_ELEMENTS_PER_THREAD_PER_LDG < LLR_REMAINING_ELEMENTS)
        {
            const int imm          = base + (LLR_LDGS - 1) * LLR_BYTES_PER_CTA_PER_LDG;
            llr_[ii][LLR_LDGS - 1] = ldg64(&params.llr[llr_gmem_offset * sizeof(uint16_t) + imm]);
        }
    }

// Apply the normalization.
#pragma unroll
    for(int ii = 0; ii < LLR_LDGS; ++ii)
    {
        llr_[0][ii].x = hmul2(llr_[0][ii].x, params.inv_norm);
        llr_[0][ii].y = hmul2(llr_[0][ii].y, params.inv_norm);
        llr_[1][ii].x = hmul2(llr_[1][ii].x, params.inv_norm);
        llr_[1][ii].y = hmul2(llr_[1][ii].y, params.inv_norm);
    }

// Interleave the FP16 numbers.
#pragma unroll
    for(int ii = 0; ii < LLR_LDGS; ++ii)
    {
        llr[ii] = interleave_llr(llr_[0][ii], llr_[1][ii]);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NODES, int Z, int THREADS_PER_CTA>
static inline __device__ void ldpc_output(const Ldpc_params& params, const float* app_smem, int offset)
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

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int NODES, int Z, int THREADS_PER_CTA>
static inline __device__ void ldpc_output(const Ldpc_params& params, const uint32_t* app_smem, int offset)
{
    // The number of threads per warp.
    enum
    {
        THREADS_PER_WARP = 32
    };

    // Decompose the thread indices into warp/lane.
    int warp = threadIdx.x / THREADS_PER_WARP;
    int lane = threadIdx.x % THREADS_PER_WARP;

    // The outputs per thread.
    int output0 = 0, output1 = 0;

    // Each warp reads 32*THREADS_PER_WARP elements.
    int idx = warp * 32 * THREADS_PER_WARP + lane;
    for(int ii = 0; ii < 32; ++ii)
    {
        uint32_t app = 0;
        if(idx + ii * THREADS_PER_WARP < NODES * Z)
        {
            app = app_smem[idx + ii * THREADS_PER_WARP];
        }

        int vote0 = __ballot_sync(0xffffffff, app & 0x00008000);
        int vote1 = __ballot_sync(0xffffffff, app & 0x80000000);
        if(lane == ii)
        {
            output0 = vote0;
            output1 = vote1;
        }
    }

    // Output the result.
    //int gmem_out_offset = blockIdx.x*2*params.outputs_per_codeword + offset;
    int gmem_out_offset = blockIdx.x * 2 * params.output_stride_words + offset;
    if(offset < params.outputs_per_codeword)
    {
        const int imm0                                             = 0;
        reinterpret_cast<int*>(params.out)[gmem_out_offset + imm0] = output0;
        //const int imm1 = params.outputs_per_codeword;
        const int imm1                                             = params.output_stride_words;
        reinterpret_cast<int*>(params.out)[gmem_out_offset + imm1] = output1;
        //KERNEL_PRINT_BLOCK_ONCE("blockIdx.x = %u, threadIdx.x = %u, offset = %i, gmem_out_offset = %i, imm0 = %i, imm1 = %i, output0 = 0x%X, output1 = 0x%X\n",
        //                        blockIdx.x, threadIdx.x, offset, gmem_out_offset, imm0, imm1, output0, output1);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <typename T, int C, int V, int Z>
static __global__ __launch_bounds__(Z) void ldpc_msa_layered_kernel(Ldpc_params params)
{
    // clang-format off
    // The type to load LLR.
    using Llr_ldg_type = typename Traits<T>::Llr_ldg_type;
    // The type to store LLR to shared memory.
    using Llr_sts_type = typename Traits<T>::Llr_sts_type;
    // The type to store APP in shared memory.
    using App_type = typename Traits<T>::App_type;

    // The amount of unrolling for V loops.
    enum { UNROLL_V_LOOPS = 8 };

    // The number of threads per CTA.
    enum { THREADS_PER_WARP = 32, THREADS_PER_CTA = Z };

    // The number of LLR elements.
    enum { LLR_ELEMENTS = /*BG1_NODES*/ V * Z };
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

    // The pointer to the buffer of APP in shared memory.
    App_type* app_smem = reinterpret_cast<App_type*>(smem_);

    // The thread index.
    int tidx = threadIdx.x;
    // The Z dimension.
    int zi = tidx;

    //
    // STAGE 1: Copy LLR data to APP in shared memory.
    //

    // Copy LLR data from global memory to registers.
    Llr_sts_type llr[LLR_LDGS];
    ldpc_llr</*BG1_NODES*/ V, V, Z>(llr, params);

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

    //
    // STAGE 2: Run the 1st iteration.
    //

    // Store one compressed C2V message in registers per check-node.
    Compressed_c2v<T> c2v[C];
#pragma unroll
    for(int ci = 0; ci < C; ++ci)
    {
        // Compute the compressed C2V for that iteration (and V2C on the fly).
        Compressed_c2v<T> next_c2v;

// Compute C2V messages for that check node.
#pragma unroll UNROLL_V_LOOPS
        for(int vi = 0; vi < bg1_row_degrees[ci]; ++vi)
        {
            int vzi = compute_app_offset<Z>(bg1_384[ci][vi], zi);
            next_c2v.update(vi, app_smem[vzi]);
        }

        // Finalize the compressed message.
        next_c2v.finalize_updates();

        // Make sure we are done reading APP before anyone starts touching it.
        __syncthreads();

// Update the APP buffer.
#pragma unroll UNROLL_V_LOOPS
        for(int vi = 0; vi < bg1_row_degrees[ci]; ++vi)
        {
            const int vzi = compute_app_offset<Z>(bg1_384[ci][vi], zi);
            app_smem[vzi] = add(app_smem[vzi], next_c2v.value(vi));
        }

        // Copy the C2V messages for the next iteration.
        c2v[ci] = next_c2v;

        // Make sure APP is ready for the next node.
        __syncthreads();
    }

//
// STAGE 3: Run the main loop.
//

// Do the loop.
#pragma unroll 1
    for(int iter = 1; iter < params.max_iterations; ++iter)
    {
#pragma unroll
        for(int ci = 0; ci < C; ++ci)
        {
            // Compute the compressed C2V for that iteration (and V2C on the fly).
            Compressed_c2v<T> next_c2v;

// Compute C2V messages for that check node.
#pragma unroll UNROLL_V_LOOPS
            for(int vi = 0; vi < bg1_row_degrees[ci]; ++vi)
            {
                int vzi = compute_app_offset<Z>(bg1_384[ci][vi], zi);
                next_c2v.update(vi, sub(app_smem[vzi], c2v[ci].value(vi)));
            }

            // Finalize the compressed message.
            next_c2v.finalize_updates();

            // Make sure we are done reading APP before anyone starts touching it.
            __syncthreads();

// Update the APP buffer.
#pragma unroll UNROLL_V_LOOPS
            for(int vi = 0; vi < bg1_row_degrees[ci]; ++vi)
            {
                const int vzi = compute_app_offset<Z>(bg1_384[ci][vi], zi);
                app_smem[vzi] = add(app_smem[vzi], sub(next_c2v.value(vi), c2v[ci].value(vi)));
            }

            // Copy the C2V messages for the next iteration.
            c2v[ci] = next_c2v;

            // Make sure APP is ready for the next node.
            __syncthreads();
        }

    } // (iter)

    //
    // STAGE 3: Hard-decision.
    //

    // Trigger the output of the 1st block that is already in SMEM.
    ldpc_output<V - C, Z, THREADS_PER_CTA>(params, app_smem, tidx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int V, int Z>
static inline __device__ float load_llr(const float* llr, int vzi)
{
    return llr[blockIdx.x * V * Z + vzi];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template <int V, int Z>
static inline __device__
    uint32_t
    load_llr(const uint16_t* llr, int vzi)
{
    int      imm0 = 0;
    uint16_t lo   = llr[blockIdx.x * V * Z * 2 + vzi + imm0];
    int      imm1 = V * Z;
    uint16_t hi   = llr[blockIdx.x * V * Z * 2 + vzi + imm1];

    return ((uint32_t)hi << 16) | (uint32_t)lo;
}

namespace ldpc
{
////////////////////////////////////////////////////////////////////////
// decode_ms_cta_shmem_layered()
cuphyStatus_t decode_ms_cta_shmem_layered(LDPC_output_t&         tDst,
                                          const_tensor_pair&     tLLR,
                                          const LDPC_config&     config,
                                          float                  normalization,
                                          cuphyLDPCResults_t*    results,
                                          void*                  workspace,
                                          cuphyLDPCDiagnostic_t* diag,
                                          cudaStream_t           strm)
{
    DEBUG_PRINTF("ldpc::decode_ms_cta_shmem_layered()\n");
    //------------------------------------------------------------------
    const int       VNODES  = config.Kb + config.mb;
    cuphyDataType_t llrType = tLLR.first.get().type();
    int             gridDimX;
    // TODO: Examine expanding to other configurations
    if((config.Kb != 22) || (config.mb != 4) || (config.Z != 384))
    {
        return CUPHY_STATUS_NOT_SUPPORTED;
    }
    //------------------------------------------------------------------
    // Prepare the kernel params.
    Ldpc_params params;
    memset(&params, 0, sizeof(params));
    params.llr = (const char*)tLLR.second;
    // Note: Using generic tensor layout here, even though only
    // contiguous is supported. Use strides[0] after conversion to
    // contiguous layout.
    params.llr_stride_elements  = tLLR.first.get().layout().strides[1];
    params.out                  = (char*)tDst.addr();
    params.output_stride_words  = tDst.layout().strides[0];
    params.outputs_per_codeword = ((config.Kb * config.Z) + 31) / 32;
    params.max_iterations       = config.max_iterations;
    if(CUPHY_R_16F == llrType)
    {
        const __half2 INV_NORM = __float2half2_rn(normalization);
        //params.inv_norm = 0x3c003c00;
        params.inv_norm        = reinterpret_cast<const uint32_t&>(INV_NORM);
        gridDimX               = (config.num_codewords + 1) / 2;
    }
    else
    {
        const float INV_NORM  = 1.f / normalization;
        params.inv_norm       = reinterpret_cast<const uint32_t&>(INV_NORM);
        gridDimX              = config.num_codewords;
    }
    DEBUG_PRINTF("LDPC Params:\n    llr_stride_elements = %i\n    output_stride_words = %i\n    outputs_per_codeword = %i\n    gridDimX = %i\n",
                 params.llr_stride_elements,
                 params.output_stride_words,
                 params.outputs_per_codeword,
                 gridDimX);
    //------------------------------------------------------------------
    // The amount of shared memory needed.
    // Note that for fp16, two codewords are decoded simultaneously.
    // Therefore, the amount of shared memory is the same for both cases.
    //int smem_sz = SMEM_ROWS * z * sizeof(int);
    int smem_sz = VNODES * config.Z * sizeof(int);

    //------------------------------------------------------------------
    // TODO: FP16 kernel decodes two codewords at a time, and assumes
    // that global memory will be addressable for an even number of
    // codewords. This may not be the case if memory for an odd number
    // of codewords was allocated.
    //------------------------------------------------------------------
    dim3 grdDim(gridDimX); // 1-D, NUM_CW for fp32, NUM_CW / 2 for fp16
    dim3 blkDim(config.Z);
    switch(llrType)
    {
    case CUPHY_R_16F:
        DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc_msa_layered_kernel<uint16_t, 4, 26, 384>));
        DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc_msa_layered_kernel<uint16_t, 4, 26, 384>), blkDim, smem_sz);
        DEBUG_PRINTF("grid = (%u, %u, %u), block = (%u, %u, %u), shmem = %i bytes\n",
                     grdDim.x,
                     grdDim.y,
                     grdDim.z,
                     blkDim.x,
                     blkDim.y,
                     blkDim.z,
                     smem_sz);
        ldpc_msa_layered_kernel<uint16_t, 4, 26, 384><<<grdDim, blkDim, smem_sz, strm>>>(params);
        break;
    case CUPHY_R_32F:
        DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc_msa_layered_kernel<float, 4, 26, 384>));
        DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc_msa_layered_kernel<float, 4, 26, 384>), blkDim, smem_sz);
        DEBUG_PRINTF("grid = (%u, %u, %u), block = (%u, %u, %u), shmem = %i bytes\n",
                     grdDim.x,
                     grdDim.y,
                     grdDim.z,
                     blkDim.x,
                     blkDim.y,
                     blkDim.z,
                     smem_sz);
        ldpc_msa_layered_kernel<float, 4, 26, 384><<<grdDim, blkDim, smem_sz, strm>>>(params);
        break;
    default:
        return CUPHY_STATUS_NOT_SUPPORTED;
    }
#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

//----------------------------------------------------------------------
// decode_ms_cta_shmem_layered_workspace_size()
std::pair<bool, size_t> decode_ms_cta_shmem_layered_workspace_size(const LDPC_config& cfg)
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
