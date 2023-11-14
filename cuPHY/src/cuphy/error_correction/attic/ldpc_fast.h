/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

#include <assert.h>
#include <float.h>
#include <stdint.h>
#include <stdio.h>
#include "cuda_fp16.h"

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK_CUDA(call) do { \
  cudaError_t status_ = call; \
  if( status_ != cudaSuccess ) { \
    fprintf(stderr, "Cuda error in file \"%s\" at line %d: %s\n", \
      __FILE__, __LINE__, cudaGetErrorString(status_)); \
    return 1; \
  } \
} while(0)

////////////////////////////////////////////////////////////////////////////////////////////////////

#define CHECK(cond) do { \
  if( !(cond) ) { \
    fprintf(stderr, "Error in file \"%s\" at line %d\n", __FILE__, __LINE__); \
    return 1; \
  } \
} while(0)

////////////////////////////////////////////////////////////////////////////////////////////////////

enum { BG1 = 1, BG1_NODES = 26, BG1_MAX_DEGREE = 19 };
enum { BG2 = 2, BG2_NODES = 22, BG2_MAX_DEGREE = 10 };

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int BG >
static inline __device__ int bg_nodes() {
  if( BG == BG1 ) {
    return BG1_NODES;
  } else if( BG == BG2 ) {
    return BG2_NODES;
  } else {
    assert(false);
    return 0;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int Z >
static inline __device__ int check_zi(int zi) {
  if( Z % 32 == 0 ) {
    return 1;
  } else {
    return zi < Z;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

struct Ldpc_params {
  // The input/output LLR (floats).
  const char *llr;
  // Stride (in elements) between inputs for consecutive codewords
  int llr_stride_elements;
  // The outputs (ints).
  char *out;
  // Stride (in words) between outputs for consecutive codewords
  int output_stride_words;
  // The problem size.
  int cw, bg, c, v, z, outputs_per_cw;
  // The number of iterations.
  int max_iters;
  // How many codewords a single CTA computes.
  //int cw_per_cta, cw_per_grid;
  // The normalization factor.
  uint32_t inv_norm;

  // // The queue to distribute the code-words between CTAs.
  // int *queue;
  // The barriers to synchronize the CTAs.
  int *barriers;
  // The number of CTAs per codeword.
  //int ctas_per_cw;
  // Precomputed values for faster division.
  //uint32_t mul_ctas_per_cw, shr_ctas_per_cw;

  // The C2V messages.
  char *c2v4, *c2v2;
  // The precomputed values for C2V spills.
  int cz4, z4, cz2, z2;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int V, int Z >
static inline __device__ void atomic_add(char *addr, int cw, int vzi, float val) {
  float *ptr = &reinterpret_cast<float*>(addr)[cw*V*Z + vzi];
  asm volatile("red.relaxed.gpu.global.add.f32 [%0], %1;\n" : : "l"(ptr), "f"(val));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int V, int Z >
static inline __device__ void atomic_add(char *addr, int cw, int vzi, uint32_t val) {
  uint32_t *ptr = &reinterpret_cast<uint32_t*>(addr)[cw*V*Z + vzi];
  asm volatile("red.relaxed.gpu.global.add.noftz.f16x2 [%0], %1;\n" : : "l"(ptr), "r"(val));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ 
void fast_divmod(int &div, int &mod, int x, int y, uint32_t mul, uint32_t shr) {
  if( y == 1 ) {
    div = x;
    mod = 0;
  } else {
    div = __umulhi((uint32_t) x, mul) >> shr;
    mod = x - div*y;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ int float2_to_half2(float a, float b) {
    int c;
    asm volatile( \
        "{\n" \
        "    .reg .f16 lo, hi;\n" \
        "    cvt.rn.f16.f32 lo, %1;\n" \
        "    cvt.rn.f16.f32 hi, %2;\n" \
        "    mov.b32 %0, {lo, hi};\n" \
        "}\n" : "=r"(c) : "f"(a), "f"(b));
    return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hadd2(uint32_t a, uint32_t b) {
  uint32_t c;
  asm volatile("add.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ void half2_to_float2(float &a, float &b, int c) {
    asm volatile( \
        "{\n" \
        "    .reg .f16 lo, hi;\n" \
        "    mov.b32 {lo, hi}, %2;\n" \
        "    cvt.f32.f16 %0, lo;\n" \
        "    cvt.f32.f16 %1, hi;\n" \
        "}\n" : "=f"(a), "=f"(b) : "r"(c));
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hequ2(uint32_t a, uint32_t b) {
  uint32_t c;
  asm volatile("set.equ.f16x2.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hfma2(uint32_t a, uint32_t b, uint32_t c) {
  uint32_t d;
  asm volatile("fma.rn.f16x2 %0, %1, %2, %3;\n" : "=r"(d) : "r"(a), "r"(b), "r"(c));
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t h0_h0(uint32_t a) {
  // It should map directly to Rx.H0_H0 in SASS!
  return a + (a << 16);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hltu2(uint32_t a, uint32_t b) {
  uint32_t c;
  asm volatile("set.ltu.f16x2.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hmul2(uint32_t a, uint32_t b) {
  uint32_t c;
  asm volatile("mul.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hneg2(uint32_t a) {
  uint32_t b;
  asm volatile("neg.f16x2 %0, %1;\n" : "=r"(b) : "r"(a));
  return b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t hsub2(uint32_t a, uint32_t b) {
  uint32_t c;
  asm volatile("sub.f16x2 %0, %1, %2;\n" : "=r"(c) : "r"(a), "r"(b));
  return c;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t inf_to_half_max(uint32_t a) {
  uint32_t c;
  uint32_t inf_inf        = 0x7C007C00;
  uint32_t fltmax_fltmax  = 0x7BFF7BFF;
  asm volatile("{\n"
               ".reg .pred eq_inf_lo, eq_inf_hi;\n"
               ".reg .u32  hi_val, lo_val;\n"
               "setp.equ.f16x2 eq_inf_lo|eq_inf_hi, %1, %2;\n" // compare a[0|1] to [inf|inf]
               "selp.u32 hi_val, %3, %1, eq_inf_hi; \n"        // set output hi word
               "selp.u32 lo_val, %3, %1, eq_inf_lo; \n"        // set output lo word
               "prmt.b32 %0, lo_val, hi_val, 0x00007610;\n"    // select lo and hi words
               "}\n"
               : "=r"(c)
               : "r"(a), "r"(inf_inf), "r"(fltmax_fltmax));
  return c;
}

inline __device__
bool half2_has_inf(const __half2 h)
{
    __half low = __low2half(h);
    __half high = __high2half(h);
    return (__hisinf(low) | __hisinf(high));
}

////////////////////////////////////////////////////////////////////////////////////////////////////
// clamp_to_half()
// Clamp the given floating point value to half_flt_max or half_flt_min
static inline __device__ float clamp_to_half(float f) {
  union half_u16
  {
      __half   h;
      uint16_t u16;
  };
  half_u16 u;
  u.u16 = 0x7BFF;
  const float abs_half_flt_max = __half2float(u.h);
  if(fabs(f) >= abs_half_flt_max)
  {
      return (f >= 0.0f) ? abs_half_flt_max : -abs_half_flt_max;
  }
  else
  {
      return f;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T >
struct Compressed_c2v {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Compressed_c2v<float> {

  // Ctor.
  inline __device__ Compressed_c2v() : min0_(FLT_MAX), min1_(FLT_MAX), idx_(0), sign_(0) {
  }

  // All updates were performed.
  inline __device__ void finalize_updates() {
    // Precompute signed values.
    if( __popc(sign_) & 0x1 ) {
      min2_ = -min1_;
      min1_ = -min0_;
    } else {
      min2_ =  min1_;
      min1_ =  min0_;
      min0_ = -min0_;
    }

    // Update the min1 value based on the sign of the indexed element.
    if( sign_ & (1 << idx_) ) {
      min2_ = -min2_;
    }
  }

  // Load a C2V message.
  inline __device__ void load(const Ldpc_params &params, int ci, int zi) {
    const int offset2 = blockIdx.x*params.cz2 + ci*params.z2 + zi*sizeof(int2);
    int2 data2 = reinterpret_cast<const int2*>(&params.c2v2[offset2])[0];

    // Extract the min0/min1 values.
    half2_to_float2(min0_, min2_, data2.x);

    // Unpack idx/sign.
    idx_  = (data2.y & 0x000000ff);
    sign_ = (data2.y & 0xffffff00) >> 8;

    // Rebuild the odd value.
    min1_ = -min0_;
  }

  // Store a C2V message.
  inline __device__ void store(const Ldpc_params &params, int ci, int zi) const {
    const int offset2 = blockIdx.x*params.cz2 + ci*params.z2 + zi*sizeof(int2);
    int2 data2;

    // Pack min0/min1 as 2x fp16s.
    // Clamp to +/- HALF_FLT_MAX before storing. If we don't do this, and
    // values are stored as INF in fp16, these get propagated to INF in
    // fp32 on loading. This causes problems when new values are less than
    // INF on the next iteration - it causes the BP algorithm to try and
    // reverse the bit.
    data2.x = float2_to_half2(clamp_to_half(min0_), clamp_to_half(min2_));
    // Pack the sign and idx into a single 32b value.
    data2.y = (sign_ << 8) | idx_;

    // Store to memory.
    reinterpret_cast<int2*>(&params.c2v2[offset2])[0] = data2;
  }

  // Update the compressed message using on V2C value.
  inline __device__ void update(int vi, float v2c) {
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

    if( abs < min0_ ) {
      min1_ = min0_;
      min0_ = abs;
      idx_  = vi;
    } else if( abs < min1_ ) { // Insert the value between min0 and min1.
      min1_ = abs;
    }

    // Update the sign.
    sign_ += signbit(v2c) << vi; 
  }
  // Compute the value min * sign where min/sign are over all the neighbours \ vi.
  inline __device__ float value(int vi) const {
    if( vi == idx_ ) {
      return min2_;
    }
    return (sign_ & (1 << vi)) ? min0_ : min1_; 
  }

  inline __device__
  bool operator==(const Compressed_c2v<float>& rhs) const {
      return (min0_    == rhs.min0_)  &&
             (min1_    == rhs.min1_)  &&
             (min2_    == rhs.min2_)  &&
             (idx_     == rhs.idx_)   &&
             (sign_    == rhs.sign_);
  }
  inline __device__
  bool operator!=(const Compressed_c2v<float>& rhs) const {
      return !(*this == rhs);
  }

  // The smallest absolute values.
  float min0_, min1_, min2_;
  // The index of the smallest value.
  int idx_;
  // The sign bits (1: negative, 0: positive).
  uint32_t sign_;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Compressed_c2v<uint16_t> {

  // Ctor.
  inline __device__ Compressed_c2v() 
    : min0_(0x7bff7bffu), min1_(0x7bff7bffu), idx_(0), sign_{0, 0} {
  }

  // All updates were performed.
  inline __device__ void finalize_updates() {
    odd_  = (__popc(sign_[0]) & 0x1) ? 0x0000bc00u : 0x00003c00u;
    odd_ |= (__popc(sign_[1]) & 0x1) ? 0xbc000000u : 0x3c000000u;
  }

  // Load a C2V message.
  inline __device__ void load(const Ldpc_params &params, int ci, int zi) {
    const int offset4 = blockIdx.x*params.cz4 + ci*params.z4 + zi*sizeof(int4);
    int4 data4 = reinterpret_cast<const int4*>(&params.c2v4[offset4])[0];
    min0_    = reinterpret_cast<uint32_t&>(data4.x);
    min1_    = reinterpret_cast<uint32_t&>(data4.y);
    sign_[0] = data4.z >> 8;
    sign_[1] = data4.w >> 8;

    // Repack the index as 2x denormalized fp16s.
    idx_ = ((data4.w & 0xff) << 16) | (data4.z & 0xff);

    // Rebuild the odd_ value.
    this->finalize_updates();
  }

  // Store a C2V message.
  inline __device__ void store(const Ldpc_params &params, int ci, int zi) const {
    const int offset4 = blockIdx.x*params.cz4 + ci*params.z4 + zi*sizeof(int4);
    int4 data4;
    data4.x = reinterpret_cast<const int&>(min0_);
    data4.y = reinterpret_cast<const int&>(min1_);
    data4.z = (sign_[0] << 8) | ((idx_ >>  0) & 0xff);
    data4.w = (sign_[1] << 8) | ((idx_ >> 16) & 0xff);
    reinterpret_cast<int4*>(&params.c2v4[offset4])[0] = data4;
  }

  // Update the compressed message using on V2C value.
  inline __device__ void update(int vi, uint32_t v2c) {
    // Extract the values without the sign bit.
    uint32_t abs = v2c & 0x7fff7fffu;
    // Infinite v2c values (which can occur for high SNRs and large
    // numbers of LDPC iterations) cause problems with the HFMA
    // formulation below. (Multiplication of inf with 0 results
    // in a NaN, which propagates to incorrect output values.)
    // Clamping inf values to the max float solves this.
    abs = inf_to_half_max(abs);

    // Is abs smaller than min0/min1?
    uint32_t p0 = hltu2(abs, min0_);
    uint32_t p1 = hltu2(abs, min1_);

    // Update min0 and min1 based where abs can be inserted.
    min1_ = hfma2(p1, abs  , hfma2(hneg2(p1), min1_, min1_));
    min1_ = hfma2(p0, min0_, hfma2(hneg2(p0), min1_, min1_));
    min0_ = hfma2(p0, abs  , hfma2(hneg2(p0), min0_, min0_));

    // Update the index -- we expect h0_h0 to map to .H0_H0 in SASS.
    idx_ = hfma2(p0, h0_h0(vi), hfma2(hneg2(p0), idx_, idx_));

    // Update the array of signs.
    sign_[0] |= ((v2c & 0x00008000u) >> 15) << vi; 
    sign_[1] |= ((v2c & 0x80000000u) >> 31) << vi; 
  }

  // Compute the value min * sign where min/sign are over all the neighbours \ vi.
  inline __device__ uint32_t value(int vi) const {
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

  // Note: ignoring odd_, since it isn't initialized at the moment...
  inline __device__
  bool operator==(const Compressed_c2v<uint16_t>& rhs) const {
      return (min0_    == rhs.min0_)    &&
             (min1_    == rhs.min1_)    &&
             (idx_     == rhs.idx_)     &&
             (sign_[0] == rhs.sign_[0]) &&
             (sign_[1] == rhs.sign_[1]);

  }
  inline __device__
  bool operator!=(const Compressed_c2v<uint16_t>& rhs) const {
      return !(*this == rhs);
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

template< int M, int N >
struct Round_up {
  enum { VALUE = (M + N-1) / N * N };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

inline int round_up(int m, int n) {
  return (m + n-1) / n * n;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

void check_results(int cw, int c, int v, int z, const int *found, const int *expected, int csv);
void find_divisor(uint32_t &mul, uint32_t &shr, int x);
void initialize_graph(int *degrees, int2 *nodes, int bg, int z);
void interleave_llr(int cw, int v, int z, uint16_t *interleaved_llr, const uint16_t *llr);
void msa_flooding(int cw, int bg, int c, int v, int z, int iters, const float *llr, int *out);
void msa_flooding(int cw, int bg, int c, int v, int z, int iters, const uint16_t *llr, int *out);
void msa_layered(int cw, int bg, int c, int v, int z, int iters, const float *llr, int *out);
void msa_layered(int cw, int bg, int c, int v, int z, int iters, const uint16_t *llr, int *out);
void random_llr(int cw, int v, int z, float *llr);
void random_llr(int cw, int v, int z, uint16_t *llr);
void read_llr(int cw, int v, int z, float *llr, const char *filename);
void read_llr(int cw, int v, int z, uint16_t *llr, const char *filename);
void read_ref(int cw, int c, int v, int z, int *ref, const char *filename);

////////////////////////////////////////////////////////////////////////////////////////////////////


