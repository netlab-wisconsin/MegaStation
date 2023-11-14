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

#include "ldpc_fast_layered.h"
#include "ldpc_fast.h"
#include "nrLDPC_flat.cuh"
#include <assert.h>

////////////////////////////////////////////////////////////////////////////////////////////////////

inline __device__ int2 gpu_node(int bg, int c, int v) {
  if( bg == BG1 ) {
    return make_int2(bg1_384[c][v].index, bg1_384[c][v].shift); // gpu_nodes[c*19 + v];
  } else if( bg == BG2 ) {
    return make_int2(-1, -1);                                   // gpu_nodes[c*10 + v];
  } else {
    assert(false);
    return make_int2(-1, -1);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

enum { SMEM_ROWS = 32 };

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int A, int B >
struct Min {
  enum { VALUE = A < B ? A : B };
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T >
struct Traits {
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Traits<float> {
  // The type to load LLR.
  using Llr_ldg_type = float4;
  // The type to store LLR.
  using Llr_sts_type = float4;
  // The type to do the math.
  using App_type = float;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template<>
struct Traits<uint16_t> {
  // The type to load LLR.
  using Llr_ldg_type = uint2;
  // The type to store LLR.
  using Llr_sts_type = uint4;
  // The type to do the math.
  using App_type = uint32_t;
};

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int Z >
static inline __device__ int compute_app_offset(const int2 &node, int zi) {
  int offset = zi + node.y;
  if( offset >= Z ) {
    offset -= Z;
  }
  return node.x * Z + offset;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float add(float a, float b) {
  return a + b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t add(uint32_t a, uint32_t b) {
  return hadd2(a, b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ float sub(float a, float b) {
  return a - b;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint32_t sub(uint32_t a, uint32_t b) {
  return hsub2(a, b);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NODES, int V, int Z, int LLR_LDGS_ >
static inline __device__
void ldpc_llr(float4 (&llr)[LLR_LDGS_], const Ldpc_params &params) {

  // The number of threads.
  enum { THREADS_PER_CTA = Round_up<Z, 32>::VALUE };
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

  // Make sure the numbers match.
  static_assert(LLR_LDGS == LLR_LDGS_, "");

  // The offset in global memory for LLR elements.
  //int llr_gmem_offset = blockIdx.x*V*Z + threadIdx.x*LLR_ELEMENTS_PER_THREAD_PER_LDG;
  int llr_gmem_offset = blockIdx.x*params.llr_stride_elements + threadIdx.x*LLR_ELEMENTS_PER_THREAD_PER_LDG;

  // Issue the loads to read LLR elements from global memory. Stage data in registers.
  #pragma unroll
  for( int ii = 0; ii < LLR_LDGS-1; ++ii ) {
    const int imm = ii*LLR_BYTES_PER_CTA_PER_LDG;
    int offset = llr_gmem_offset*sizeof(float) + imm;
    llr[ii] = *reinterpret_cast<const float4*>(&params.llr[offset]);
  }

  // Deal with the last (possibly) incomplete LDG.
  if( threadIdx.x*LLR_ELEMENTS_PER_THREAD_PER_LDG < LLR_REMAINING_ELEMENTS ) {
    const int imm = (LLR_LDGS-1)*LLR_BYTES_PER_CTA_PER_LDG;
    int offset = llr_gmem_offset*sizeof(float) + imm;
    llr[LLR_LDGS-1] = *reinterpret_cast<const float4*>(&params.llr[offset]);
  }

  // Apply the normalization.
  #pragma unroll
  for( int ii = 0; ii < LLR_LDGS; ++ii ) {
    llr[ii].x *= reinterpret_cast<const float&>(params.inv_norm);
    llr[ii].y *= reinterpret_cast<const float&>(params.inv_norm);
    llr[ii].z *= reinterpret_cast<const float&>(params.inv_norm);
    llr[ii].w *= reinterpret_cast<const float&>(params.inv_norm);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

static inline __device__ uint4 interleave_llr(uint2 llr0, uint2 llr1) {
  uint4 d;
  asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(d.x) : "r"(llr0.x), "r"(llr1.x));
  asm volatile("prmt.b32 %0, %1, %2, 0x7632;\n" : "=r"(d.y) : "r"(llr0.x), "r"(llr1.x));
  asm volatile("prmt.b32 %0, %1, %2, 0x5410;\n" : "=r"(d.z) : "r"(llr0.y), "r"(llr1.y));
  asm volatile("prmt.b32 %0, %1, %2, 0x7632;\n" : "=r"(d.w) : "r"(llr0.y), "r"(llr1.y));
  return d;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NODES, int V, int Z, int LLR_LDGS_ >
static inline __device__
void ldpc_llr(uint4 (&llr)[LLR_LDGS_], const Ldpc_params &params) {

  // The number of threads.
  enum { THREADS_PER_CTA = Round_up<Z, 32>::VALUE };
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

  // Make sure the numbers match.
  static_assert(LLR_LDGS == LLR_LDGS_, "");

  // The offset in global memory for LLR elements.
  //int llr_gmem_offset = blockIdx.x*V*Z*2 + threadIdx.x*LLR_ELEMENTS_PER_THREAD_PER_LDG;
  int llr_gmem_offset = blockIdx.x * params.llr_stride_elements * 2 + threadIdx.x*LLR_ELEMENTS_PER_THREAD_PER_LDG;

  // Issue the loads to read LLR elements from global memory. Stage data in registers.
  uint2 llr_[2][LLR_LDGS];
  #pragma unroll
  for( int ii = 0; ii < 2; ++ii ) {
    if((blockIdx.x * 2 + ii) < params.cw) { // Avoid reads past the end of input for odd number of codewords
      //const int base = ii*V*Z * sizeof(uint16_t);
      const int base = ii*params.llr_stride_elements * sizeof(uint16_t);
      #pragma unroll
      for( int jj = 0; jj < LLR_LDGS-1; ++jj ) {
        const int imm = base + jj*LLR_BYTES_PER_CTA_PER_LDG;
        const char *ptr = &params.llr[llr_gmem_offset*sizeof(uint16_t) + imm];
        llr_[ii][jj] = *reinterpret_cast<const uint2*>(ptr);
      }
  
      // Deal with the last (possibly) incomplete LDG.
      if( threadIdx.x*LLR_ELEMENTS_PER_THREAD_PER_LDG < LLR_REMAINING_ELEMENTS ) {
        const int imm = base + (LLR_LDGS-1)*LLR_BYTES_PER_CTA_PER_LDG;
        const char *ptr = &params.llr[llr_gmem_offset*sizeof(uint16_t) + imm];
        llr_[ii][LLR_LDGS-1] = *reinterpret_cast<const uint2*>(ptr);
      }
    }
  }

  // Apply the normalization.
  #pragma unroll
  for( int ii = 0; ii < LLR_LDGS; ++ii ) {
    llr_[0][ii].x = hmul2(llr_[0][ii].x, params.inv_norm);
    llr_[0][ii].y = hmul2(llr_[0][ii].y, params.inv_norm);
    llr_[1][ii].x = hmul2(llr_[1][ii].x, params.inv_norm);
    llr_[1][ii].y = hmul2(llr_[1][ii].y, params.inv_norm);
  }

  // Interleave the FP16 numbers.
  #pragma unroll
  for( int ii = 0; ii < LLR_LDGS; ++ii ) {
    llr[ii] = interleave_llr(llr_[0][ii], llr_[1][ii]);
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NODES, int Z, int THREADS_PER_CTA >
static inline __device__
void ldpc_output(const Ldpc_params &params, const float *app_smem, int offset) {

  // Make sure there are enough rows in SMEM.
  static_assert(NODES <= SMEM_ROWS, "");

  // The number of threads per warp.
  enum { THREADS_PER_WARP = 32 };

  // Decompose the thread indices into warp/lane.
  int warp = threadIdx.x / THREADS_PER_WARP;
  int lane = threadIdx.x % THREADS_PER_WARP;

  // The output per thread.
  int output = 0;

  // Each warp reads 32*THREADS_PER_WARP elements.
  int idx = warp*32*THREADS_PER_WARP + lane;
  for( int ii = 0; ii < 32; ++ii ) {
    float val = 0.f;
    if( idx + ii*THREADS_PER_WARP < NODES*Z ) {
      val = app_smem[idx + ii*THREADS_PER_WARP];
    }

    int vote = __ballot_sync(0xffffffff, val < 0.f);
    if( lane == ii ) {
      output = vote;
    }
  }

  // Output the result.
  //int gmem_out_offset = blockIdx.x*params.outputs_per_cw + offset;
  int gmem_out_offset = blockIdx.x * params.output_stride_words + offset;
  if( offset < params.outputs_per_cw ) {
    reinterpret_cast<int*>(params.out)[gmem_out_offset] = output;
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< int NODES, int Z, int THREADS_PER_CTA >
static inline __device__
void ldpc_output(const Ldpc_params &params, const uint32_t *app_smem, int offset) {

  // The number of threads per warp.
  enum { THREADS_PER_WARP = 32 };

  // Decompose the thread indices into warp/lane.
  int warp = threadIdx.x / THREADS_PER_WARP;
  int lane = threadIdx.x % THREADS_PER_WARP;

  // The outputs per thread.
  int output0 = 0, output1 = 0;

  // Each warp reads 32*THREADS_PER_WARP elements.
  int idx = warp*32*THREADS_PER_WARP + lane;
  for( int ii = 0; ii < 32; ++ii ) {
    uint32_t app = 0;
    if( idx + ii*THREADS_PER_WARP < NODES*Z ) {
      app = app_smem[idx + ii*THREADS_PER_WARP];
    }

    int vote0 = __ballot_sync(0xffffffff, app & 0x00008000);
    int vote1 = __ballot_sync(0xffffffff, app & 0x80000000);
    if( lane == ii ) {
      output0 = vote0;
      output1 = vote1;
    }
  }

  // Output the result.
  //int gmem_out_offset = blockIdx.x*2*params.outputs_per_cw + offset;
  int gmem_out_offset = blockIdx.x * 2 * params.output_stride_words + offset;
  if( offset < params.outputs_per_cw ) {
    const int imm0 = 0;
    reinterpret_cast<int*>(params.out)[gmem_out_offset + imm0] = output0;
    if((blockIdx.x * 2 + 1) < params.cw) { // Avoid writes past the end for odd number of codewords
      //const int imm1 = params.outputs_per_cw;
      const int imm1 = params.output_stride_words;
      reinterpret_cast<int*>(params.out)[gmem_out_offset + imm1] = output1;
    }
  }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T, int BG, int C, int V, int Z >
static __global__ __launch_bounds__(Round_up<Z, 32>::VALUE)
void ldpc_msa_layered_kernel(Ldpc_params params) {

  // The type to load LLR.
  using Llr_ldg_type = typename Traits<T>::Llr_ldg_type;
  // The type to store LLR to shared memory.
  using Llr_sts_type = typename Traits<T>::Llr_sts_type;
  // The type to store APP in shared memory.
  using App_type = typename Traits<T>::App_type;

  // The amount of unrolling for V loops.
  enum { UNROLL_V_LOOPS = 8 };

  // The number of threads per CTA.
  enum { THREADS_PER_WARP = 32, THREADS_PER_CTA = Round_up<Z, 32>::VALUE };

  // The number of LLR elements.
  enum { LLR_ELEMENTS = (BG == BG1 ? BG1_NODES : BG2_NODES) * Z };
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

  // The shared memory.
  __shared__ char smem_[SMEM_ROWS*Z*sizeof(int)];

  // The pointer to the buffer of APP in shared memory.
  App_type *app_smem = reinterpret_cast<App_type*>(smem_);

  // The thread index.
  int tidx = threadIdx.x;
  // The Z dimension.
  int zi = tidx;

  // 
  // STAGE 1: Copy LLR data to APP in shared memory.
  //
  
  // Copy LLR data from global memory to registers.
  Llr_sts_type llr[LLR_LDGS];
  ldpc_llr<(BG == BG1 ? BG1_NODES : BG2_NODES), V, Z>(llr, params);
  
  // The offset in shared memory for LLR elements.
  int llr_smem_offset = threadIdx.x*LLR_ELEMENTS_PER_THREAD_PER_STS;

  // Copy the LLR elements to shared memory.
  #pragma unroll
  for( int ii = 0; ii < LLR_LDGS-1; ++ii ) {
    const int imm = ii*LLR_BYTES_PER_CTA_PER_STS;
    reinterpret_cast<Llr_sts_type*>(&smem_[llr_smem_offset*sizeof(T) + imm])[0] = llr[ii];
  }

  // Deal with the last (possibly) incomplete LDG.
  if( threadIdx.x*LLR_ELEMENTS_PER_THREAD_PER_LDG < LLR_REMAINING_ELEMENTS ) {
    const int imm = (LLR_LDGS-1)*LLR_BYTES_PER_CTA_PER_STS;
    reinterpret_cast<Llr_sts_type*>(&smem_[llr_smem_offset*sizeof(T) + imm])[0] = llr[LLR_LDGS-1];
  }
  
  // Make sure the data is in shared memory.
  __syncthreads();

  // 
  // STAGE 2: Run the 1st iteration.
  //

  // Store one compressed C2V message in registers per check-node. 
  Compressed_c2v<T> c2v[C];
  #pragma unroll
  for( int ci = 0; ci < C; ++ci ) {

    // Compute the compressed C2V for that iteration (and V2C on the fly).
    Compressed_c2v<T> next_c2v;

    // Compute C2V messages for that check node.
    #pragma unroll UNROLL_V_LOOPS
    for( int vi = 0; vi < bg1_row_degrees[ci]; ++vi ) {
      int vzi = compute_app_offset<Z>(gpu_node(BG, ci, vi), zi);
      if( check_zi<Z>(zi) ) {
        next_c2v.update(vi, app_smem[vzi]);
      }
    } 

    // Finalize the compressed message.
    next_c2v.finalize_updates();

    // Make sure we are done reading APP before anyone starts touching it.
    __syncthreads();

    // Update the APP buffer.
    #pragma unroll UNROLL_V_LOOPS
    for( int vi = 0; vi < bg1_row_degrees[ci]; ++vi ) {
      const int vzi = compute_app_offset<Z>(gpu_node(BG, ci, vi), zi);
      if( check_zi<Z>(zi) ) {
        app_smem[vzi] = add(app_smem[vzi], next_c2v.value(vi));
      }
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
  for( int iter = 1; iter < params.max_iters; ++iter ) {

    #pragma unroll
    for( int ci = 0; ci < C; ++ci ) {

      // Compute the compressed C2V for that iteration (and V2C on the fly).
      Compressed_c2v<T> next_c2v;

      // Compute C2V messages for that check node.
      #pragma unroll UNROLL_V_LOOPS
      for( int vi = 0; vi < bg1_row_degrees[ci]; ++vi ) {
        int vzi = compute_app_offset<Z>(gpu_node(BG, ci, vi), zi);
        if( check_zi<Z>(zi) ) {
          next_c2v.update(vi, sub(app_smem[vzi], c2v[ci].value(vi)));
        }
      } 

      // Finalize the compressed message.
      next_c2v.finalize_updates();

      // Make sure we are done reading APP before anyone starts touching it.
      __syncthreads();

      // Update the APP buffer.
      #pragma unroll UNROLL_V_LOOPS
      for( int vi = 0; vi < bg1_row_degrees[ci]; ++vi ) {
        const int vzi = compute_app_offset<Z>(gpu_node(BG, ci, vi), zi);
        if( check_zi<Z>(zi) ) {
          app_smem[vzi] = add(app_smem[vzi], sub(next_c2v.value(vi), c2v[ci].value(vi)));
        }
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
  ldpc_output<V-C, Z, THREADS_PER_CTA>(params, app_smem, tidx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

//template< int V, int Z >
static inline __device__ 
float load_llr(const float *llr, int cw_stride, int vzi) {
  //return llr[blockIdx.x*V*Z + vzi];
  return llr[blockIdx.x*cw_stride + vzi];
}

////////////////////////////////////////////////////////////////////////////////////////////////////

//template< int V, int Z >
static inline __device__ 
uint32_t load_llr(const uint16_t *llr, int cw_stride, int vzi) {
  int imm0 = 0;
  //uint16_t lo = llr[blockIdx.x*V*Z*2 + vzi + imm0];
  uint16_t lo = llr[blockIdx.x*cw_stride*2 + vzi + imm0];
  //int imm1 = V*Z;
  //uint16_t hi = llr[blockIdx.x*V*Z*2 + vzi + imm1];
  uint16_t hi = llr[blockIdx.x*cw_stride*2 + vzi + cw_stride];

  return ((uint32_t) hi << 16) | (uint32_t) lo;
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T, int BG, int C, int V, int Z, int C_PER_THREAD >
static __global__ __launch_bounds__(Round_up<Z, 32>::VALUE)
void ldpc_msa_layered_with_spills_kernel(Ldpc_params params) {

  // The type to load LLR.
  using Llr_ldg_type = typename Traits<T>::Llr_ldg_type;
  // The type to store LLR to shared memory.
  using Llr_sts_type = typename Traits<T>::Llr_sts_type;
  // The type to store APP in shared memory.
  using App_type = typename Traits<T>::App_type;

  // The amount of unrolling for V loops.
  enum { UNROLL_V_LOOPS = 4 };

  // The number of threads per CTA.
  enum { THREADS_PER_WARP = 32, THREADS_PER_CTA = Round_up<Z, 32>::VALUE };

  // The number of LLR elements.
  enum { LLR_ELEMENTS = Min<SMEM_ROWS, V>::VALUE * Z };
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

  // Declare shared memory.
  __shared__ char smem_[SMEM_ROWS*Z*sizeof(int)];

  // The pointer to the buffer of APP in shared memory.
  App_type *app_smem = reinterpret_cast<App_type*>(smem_);

  // The thread index.
  int tidx = threadIdx.x;
  // The Z dimension.
  int zi = tidx;

  // 
  // STAGE 1: Copy LLR data to APP in shared memory.
  //
  
  // Copy LLR data from global memory to registers.
  Llr_sts_type llr[LLR_LDGS];
  ldpc_llr<Min<SMEM_ROWS, V>::VALUE, V, Z>(llr, params);
  
  // The offset in shared memory for LLR elements.
  int llr_smem_offset = threadIdx.x*LLR_ELEMENTS_PER_THREAD_PER_STS;

  // Copy the LLR elements to shared memory.
  #pragma unroll
  for( int ii = 0; ii < LLR_LDGS-1; ++ii ) {
    const int imm = ii*LLR_BYTES_PER_CTA_PER_STS;
    reinterpret_cast<Llr_sts_type*>(&smem_[llr_smem_offset*sizeof(T) + imm])[0] = llr[ii];
  }

  // Deal with the last (possibly) incomplete LDG.
  if( threadIdx.x*LLR_ELEMENTS_PER_THREAD_PER_LDG < LLR_REMAINING_ELEMENTS ) {
    const int imm = (LLR_LDGS-1)*LLR_BYTES_PER_CTA_PER_STS;
    reinterpret_cast<Llr_sts_type*>(&smem_[llr_smem_offset*sizeof(T) + imm])[0] = llr[LLR_LDGS-1];
  }
  
  // Make sure the data is in shared memory.
  __syncthreads();

  // 
  // STAGE 2: Run the 1st iteration.
  //

  // Store one compressed C2V message in registers per check-node. 
  #pragma unroll 1
  for( int co = 0; co < params.c; co += C_PER_THREAD ) {
    #pragma unroll
    for( int ci = 0; ci < C_PER_THREAD; ++ci ) {
      // The corresponding check-node.
      int c = co + ci;

      // Compute C2V messages for that check node.
      Compressed_c2v<T> c2v;
      #pragma unroll UNROLL_V_LOOPS
      for( int vi = 0; vi < bg1_row_degrees[c]; ++vi ) {
        const int2 &node = gpu_node(BG, c, vi);
        if( node.x < Min<SMEM_ROWS, V>::VALUE ) {
          int vzi = compute_app_offset<Z>(node, zi);
          if( check_zi<Z>(zi) ) {
            c2v.update(vi, app_smem[vzi]);
          }
        } else {
          if( check_zi<Z>(zi) ) {
            //c2v.update(vi, load_llr<V, Z>((const T*) params.llr, params.llr_stride_elements, node.x * Z + zi));
            c2v.update(vi, load_llr((const T*) params.llr, params.llr_stride_elements, node.x * Z + zi));
          }
        }
      } 

      // Finalize the compressed message.
      c2v.finalize_updates();

      // Make sure we are done reading APP before anyone starts touching it.
      __syncthreads();

      // Update the APP buffer.
      #pragma unroll UNROLL_V_LOOPS
      for( int vi = 0; vi < bg1_row_degrees[c]; ++vi ) {
        const int2 &node = gpu_node(BG, c, vi);
        if( node.x < Min<SMEM_ROWS, V>::VALUE ) {
          const int vzi = compute_app_offset<Z>(node, zi);
          if( check_zi<Z>(zi) ) {
            app_smem[vzi] = add(app_smem[vzi], c2v.value(vi));
          }
        }
      } 

      // Spill the C2V messages to global memory.
      if( check_zi<Z>(zi) ) {
        c2v.store(params, c, zi);
      }

      // Make sure APP is ready for the next node.
      __syncthreads();
    } 
  }

  // 
  // STAGE 3: Run the main loop.
  //

  // Do the loop.
  #pragma unroll 1
  for( int iter = 1; iter < params.max_iters; ++iter ) {
    #pragma unroll 1
    for( int co = 0; co < params.c; co += C_PER_THREAD ) {
      #pragma unroll
      for( int ci = 0; ci < C_PER_THREAD; ++ci ) {
        // The corresponding check-node.
        int c = co + ci;

        // Load the "spilled" value for C2V.
        Compressed_c2v<T> c2v;
        if( check_zi<Z>(zi) ) {
          c2v.load(params, c, zi);
        }

        // Compute C2V messages for that check node.
        Compressed_c2v<T> next_c2v;
        #pragma unroll UNROLL_V_LOOPS
        for( int vi = 0; vi < bg1_row_degrees[c]; ++vi ) {
          const int2 &node = gpu_node(BG, c, vi);
          if( node.x < Min<SMEM_ROWS, V>::VALUE ) {
            int vzi = compute_app_offset<Z>(node, zi);
            if( check_zi<Z>(zi) ) {
              next_c2v.update(vi, sub(app_smem[vzi], c2v.value(vi)));
            }
          } else {
            if( check_zi<Z>(zi) ) {
              //next_c2v.update(vi, load_llr<V, Z>((const T*) params.llr, params.llr_stride_elements, node.x * Z + zi));
              next_c2v.update(vi, load_llr((const T*) params.llr, params.llr_stride_elements, node.x * Z + zi));
            }
          }
        } 

        // Finalize the compressed message.
        next_c2v.finalize_updates();

        // Make sure we are done reading APP before anyone starts touching it.
        __syncthreads();

        // Update the APP buffer.
        #pragma unroll UNROLL_V_LOOPS
        for( int vi = 0; vi < bg1_row_degrees[c]; ++vi ) {
          const int2 &node = gpu_node(BG, c, vi);
          if( node.x < Min<SMEM_ROWS, V>::VALUE ) {
            const int vzi = compute_app_offset<Z>(node, zi);
            if( check_zi<Z>(zi) ) {
              app_smem[vzi] = add(app_smem[vzi], sub(next_c2v.value(vi), c2v.value(vi)));
            }
          }
        } 

        // Copy the C2V messages for the next iteration.
        if( check_zi<Z>(zi) ) {
          next_c2v.store(params, c, zi);
        }

        // Make sure APP is ready for the next node.
        __syncthreads();
      } 
    } // (loop)
  } // (iter)

  // 
  // STAGE 3: Hard-decision.
  //

  // The following code does not work for V >= 64.

  // Trigger the output of the 1st block that is already in SMEM.
  ldpc_output<V-C, Z, THREADS_PER_CTA>(params, app_smem, tidx);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

//#ifdef USE_AUTO_TUNING
//enum {
//  UC00_C_PER_THREAD_FP32 = USE_C_PER_THREAD,
//  UC01_C_PER_THREAD_FP32 = USE_C_PER_THREAD,
//  UC02_C_PER_THREAD_FP32 = USE_C_PER_THREAD,
//  UC03_C_PER_THREAD_FP32 = USE_C_PER_THREAD,
//  UC04_C_PER_THREAD_FP32 = USE_C_PER_THREAD,
//  UC05_C_PER_THREAD_FP32 = USE_C_PER_THREAD,
//  UC06_C_PER_THREAD_FP32 = USE_C_PER_THREAD,
//  UC07_C_PER_THREAD_FP32 = USE_C_PER_THREAD,
//  UC08_C_PER_THREAD_FP32 = USE_C_PER_THREAD,
//  UC09_C_PER_THREAD_FP32 = USE_C_PER_THREAD,
//  UC10_C_PER_THREAD_FP32 = USE_C_PER_THREAD,
//  UC11_C_PER_THREAD_FP32 = USE_C_PER_THREAD,
//  UC12_C_PER_THREAD_FP32 = USE_C_PER_THREAD,
//  UC13_C_PER_THREAD_FP32 = USE_C_PER_THREAD,
//  UC14_C_PER_THREAD_FP32 = USE_C_PER_THREAD,
//  UC15_C_PER_THREAD_FP32 = USE_C_PER_THREAD,
//  UC16_C_PER_THREAD_FP32 = USE_C_PER_THREAD,
//
//  UC00_C_PER_THREAD_FP16 = USE_C_PER_THREAD,
//  UC01_C_PER_THREAD_FP16 = USE_C_PER_THREAD,
//  UC02_C_PER_THREAD_FP16 = USE_C_PER_THREAD,
//  UC03_C_PER_THREAD_FP16 = USE_C_PER_THREAD,
//  UC04_C_PER_THREAD_FP16 = USE_C_PER_THREAD,
//  UC05_C_PER_THREAD_FP16 = USE_C_PER_THREAD,
//  UC06_C_PER_THREAD_FP16 = USE_C_PER_THREAD,
//  UC07_C_PER_THREAD_FP16 = USE_C_PER_THREAD,
//  UC08_C_PER_THREAD_FP16 = USE_C_PER_THREAD,
//  UC09_C_PER_THREAD_FP16 = USE_C_PER_THREAD,
//  UC10_C_PER_THREAD_FP16 = USE_C_PER_THREAD,
//  UC11_C_PER_THREAD_FP16 = USE_C_PER_THREAD,
//  UC12_C_PER_THREAD_FP16 = USE_C_PER_THREAD,
//  UC13_C_PER_THREAD_FP16 = USE_C_PER_THREAD,
//  UC14_C_PER_THREAD_FP16 = USE_C_PER_THREAD,
//  UC15_C_PER_THREAD_FP16 = USE_C_PER_THREAD,
//  UC16_C_PER_THREAD_FP16 = USE_C_PER_THREAD,
//};
//#else
//enum {
//  UC00_C_PER_THREAD_FP32 = 1,
//  UC01_C_PER_THREAD_FP32 = 4,
//  UC02_C_PER_THREAD_FP32 = 4,
//  UC03_C_PER_THREAD_FP32 = 7,
//  UC04_C_PER_THREAD_FP32 = 5,
//  UC05_C_PER_THREAD_FP32 = 3,
//  UC06_C_PER_THREAD_FP32 = 2,
//  UC07_C_PER_THREAD_FP32 = 1,
//  UC08_C_PER_THREAD_FP32 = 1,
//  UC09_C_PER_THREAD_FP32 = 7,
//  UC10_C_PER_THREAD_FP32 = 6,
//  UC11_C_PER_THREAD_FP32 = 5,
//  UC12_C_PER_THREAD_FP32 = 1,
//  UC13_C_PER_THREAD_FP32 = 5,
//  UC14_C_PER_THREAD_FP32 = 4,
//  UC15_C_PER_THREAD_FP32 = 2,
//  UC16_C_PER_THREAD_FP32 = 1,
//  UC17_C_PER_THREAD_FP32 = 1,
//
//  UC00_C_PER_THREAD_FP16 = 1,
//  UC01_C_PER_THREAD_FP16 = 1,
//  UC02_C_PER_THREAD_FP16 = 1,
//  UC03_C_PER_THREAD_FP16 = 1,
//  UC04_C_PER_THREAD_FP16 = 1,
//  UC05_C_PER_THREAD_FP16 = 1,
//  UC06_C_PER_THREAD_FP16 = 1,
//  UC07_C_PER_THREAD_FP16 = 1,
//  UC08_C_PER_THREAD_FP16 = 1,
//  UC09_C_PER_THREAD_FP16 = 1,
//  UC10_C_PER_THREAD_FP16 = 1,
//  UC11_C_PER_THREAD_FP16 = 1,
//  UC12_C_PER_THREAD_FP16 = 1,
//  UC13_C_PER_THREAD_FP16 = 1,
//  UC14_C_PER_THREAD_FP16 = 1,
//  UC15_C_PER_THREAD_FP16 = 1,
//  UC16_C_PER_THREAD_FP16 = 1,
//  UC17_C_PER_THREAD_FP16 = 1,
//};
//#endif

////////////////////////////////////////////////////////////////////////////////////////////////////
//
//int ldpc_fast_layered_launch_fp32(const Ldpc_params &params) {
//
//  #define UCx_LAUNCH(bg_, c_, v_, z_) \
//  if( params.bg == bg_ && params.c == c_ && params.v == v_ && params.z == z_ ) { \
//    assert(v_ - c_ <= 32); \
//    ldpc_msa_layered_kernel<float, bg_, c_, v_, z_> \
//      <<<params.cw, round_up(z_, 32)>>>(params); \
//  }
//
//  #define UCx_LAUNCH_WITH_SPILLS(bg_, c_, v_, z_, c_per_thread_) \
//  if( params.bg == bg_ && params.c == c_ && params.v == v_ && params.z == z_ ) { \
//    assert(v_ - c_ <= 32); \
//    ldpc_msa_layered_with_spills_kernel<float, bg_, c_, v_, z_, c_per_thread_> \
//      <<<params.cw, round_up(z_, 32)>>>(params); \
//  }
//
//       UCx_LAUNCH            (BG1,  4, 26, 384)
//  else UCx_LAUNCH_WITH_SPILLS(BG1, 28, 50, 384, UC01_C_PER_THREAD_FP32)
//  else UCx_LAUNCH_WITH_SPILLS(BG1, 40, 62, 384, UC02_C_PER_THREAD_FP32)
//  else UCx_LAUNCH_WITH_SPILLS(BG2, 14, 24, 144, UC03_C_PER_THREAD_FP32)
//  else UCx_LAUNCH_WITH_SPILLS(BG2, 10, 20, 176, UC04_C_PER_THREAD_FP32)
//  else UCx_LAUNCH_WITH_SPILLS(BG2,  9, 19, 192, UC05_C_PER_THREAD_FP32)
//  else UCx_LAUNCH            (BG2,  7, 17, 208)
//  else UCx_LAUNCH_WITH_SPILLS(BG2, 22, 32, 208, UC07_C_PER_THREAD_FP32)
//  else UCx_LAUNCH_WITH_SPILLS(BG2, 20, 30, 224, UC08_C_PER_THREAD_FP32)
//  else UCx_LAUNCH_WITH_SPILLS(BG2, 14, 24, 288, UC09_C_PER_THREAD_FP32)
//  else UCx_LAUNCH_WITH_SPILLS(BG2, 12, 22, 320, UC10_C_PER_THREAD_FP32)
//  else UCx_LAUNCH_WITH_SPILLS(BG2, 10, 20, 352, UC11_C_PER_THREAD_FP32)
//  else UCx_LAUNCH_WITH_SPILLS(BG2,  9, 19, 384, UC12_C_PER_THREAD_FP32)
//  else UCx_LAUNCH_WITH_SPILLS(BG1, 15, 37, 176, UC13_C_PER_THREAD_FP32)
//  else UCx_LAUNCH_WITH_SPILLS(BG1, 32, 54, 176, UC14_C_PER_THREAD_FP32)
//  else UCx_LAUNCH_WITH_SPILLS(BG1, 28, 50, 192, UC15_C_PER_THREAD_FP32)
//  else UCx_LAUNCH_WITH_SPILLS(BG1, 22, 44, 224, UC16_C_PER_THREAD_FP32)
//  else UCx_LAUNCH_WITH_SPILLS(BG1,  7, 29, 384, UC17_C_PER_THREAD_FP32)
//  else assert(false);
//  return 0;
//
//  #undef UCx_LAUNCH
//  #undef UCx_LAUNCH_WITH_SPILLS
//}
//
////////////////////////////////////////////////////////////////////////////////////////////////////
//
//int ldpc_fast_layered_launch_fp16(const Ldpc_params &params) {
//
//  #define UCx_LAUNCH(bg_, c_, v_, z_) \
//  if( params.bg == bg_ && params.c == c_ && params.v == v_ && params.z == z_ ) { \
//    assert(v_ - c_ <= 32); \
//    ldpc_msa_layered_kernel<uint16_t, bg_, c_, v_, z_> \
//      <<<params.cw / 2, round_up(z_, 32)>>>(params); \
//  }
//
//  #define UCx_LAUNCH_WITH_SPILLS(bg_, c_, v_, z_, c_per_thread_) \
//  if( params.bg == bg_ && params.c == c_ && params.v == v_ && params.z == z_ ) { \
//    assert(v_ - c_ <= 32); \
//    ldpc_msa_layered_with_spills_kernel<uint16_t, bg_, c_, v_, z_, c_per_thread_> \
//      <<<params.cw / 2, round_up(z_, 32)>>>(params); \
//  }
//
//       UCx_LAUNCH            (BG1,  4, 26, 384)
//  else UCx_LAUNCH_WITH_SPILLS(BG1, 28, 50, 384, UC01_C_PER_THREAD_FP16)
//  else UCx_LAUNCH_WITH_SPILLS(BG1, 40, 62, 384, UC02_C_PER_THREAD_FP16)
//  else UCx_LAUNCH_WITH_SPILLS(BG2, 14, 24, 144, UC03_C_PER_THREAD_FP16)
//  else UCx_LAUNCH_WITH_SPILLS(BG2, 10, 20, 176, UC04_C_PER_THREAD_FP16)
//  else UCx_LAUNCH_WITH_SPILLS(BG2,  9, 19, 192, UC05_C_PER_THREAD_FP16)
//  else UCx_LAUNCH            (BG2,  7, 17, 208)
//  else UCx_LAUNCH_WITH_SPILLS(BG2, 22, 32, 208, UC07_C_PER_THREAD_FP16)
//  else UCx_LAUNCH_WITH_SPILLS(BG2, 20, 30, 224, UC08_C_PER_THREAD_FP16)
//  else UCx_LAUNCH_WITH_SPILLS(BG2, 14, 24, 288, UC09_C_PER_THREAD_FP16)
//  else UCx_LAUNCH_WITH_SPILLS(BG2, 12, 22, 320, UC10_C_PER_THREAD_FP16)
//  else UCx_LAUNCH_WITH_SPILLS(BG2, 10, 20, 352, UC11_C_PER_THREAD_FP16)
//  else UCx_LAUNCH_WITH_SPILLS(BG2,  9, 19, 384, UC12_C_PER_THREAD_FP16)
//  else UCx_LAUNCH_WITH_SPILLS(BG1, 15, 37, 176, UC13_C_PER_THREAD_FP16)
//  else UCx_LAUNCH_WITH_SPILLS(BG1, 32, 54, 176, UC14_C_PER_THREAD_FP16)
//  else UCx_LAUNCH_WITH_SPILLS(BG1, 28, 50, 192, UC15_C_PER_THREAD_FP16)
//  else UCx_LAUNCH_WITH_SPILLS(BG1, 22, 44, 224, UC16_C_PER_THREAD_FP16)
//  else UCx_LAUNCH_WITH_SPILLS(BG1,  7, 29, 384, UC17_C_PER_THREAD_FP16)
//  else assert(false);
//  return 0;
//
//  #undef UCx_LAUNCH
//  #undef UCx_LAUNCH_WITH_SPILLS
//}

////////////////////////////////////////////////////////////////////////////////////////////////////

template< typename T, int BG, int C, int V, int Z >
void ldpc_fast_layered_launch(const dim3&        gridDim,
                              const dim3&        blockDim,
                              const Ldpc_params& params,
                              cudaStream_t       strm)
{
    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc_msa_layered_kernel<T, BG, C, V, Z>));
    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc_msa_layered_kernel<T, BG, C, V, Z>), blockDim, 0);
    DEBUG_PRINTF("grid = (%u, %u, %u), block = (%u, %u, %u)\n",
                     gridDim.x,
                     gridDim.y,
                     gridDim.z,
                     blockDim.x,
                     blockDim.y,
                     blockDim.z);
    ldpc_msa_layered_kernel<T, BG, C, V, Z><<<gridDim, blockDim, 0, strm>>>(params);
}

template< typename T, int BG, int C, int V, int Z, int C_PER_THREAD>
void ldpc_fast_layered_with_spills_launch(const dim3&        gridDim,
                                          const dim3&        blockDim,
                                          const Ldpc_params& params,
                                          cudaStream_t       strm)
{
    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc_msa_layered_with_spills_kernel<T, BG, C, V, Z, C_PER_THREAD>));
    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc_msa_layered_with_spills_kernel<T, BG, C, V, Z, C_PER_THREAD>), blockDim, 0);
    DEBUG_PRINTF("grid = (%u, %u, %u), block = (%u, %u, %u)\n",
                     gridDim.x,
                     gridDim.y,
                     gridDim.z,
                     blockDim.x,
                     blockDim.y,
                     blockDim.z);
    ldpc_msa_layered_with_spills_kernel<T, BG, C, V, Z, C_PER_THREAD><<<gridDim, blockDim, 0, strm>>>(params);
}

namespace ldpc
{
////////////////////////////////////////////////////////////////////////
// decode_fast_layered()
cuphyStatus_t decode_fast_layered(LDPC_output_t&         tDst,
                                  const_tensor_pair&     tLLR,
                                  const LDPC_config&     config,
                                  float                  normalization,
                                  cuphyLDPCResults_t*    results,
                                  void*                  workspace,
                                  cuphyLDPCDiagnostic_t* diag,
                                  cudaStream_t           strm)
{
    DEBUG_PRINTF("ldpc::decode_fast_layered()\n");
    //------------------------------------------------------------------
    //const int       VNODES  = config.Kb + config.mb;
    cuphyDataType_t llrType = tLLR.first.get().type();
    int             gridDimX;
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
    params.cw                   = config.num_codewords;
    params.bg                   = config.BG;
    params.c                    = config.mb;
    params.v                    = config.Kb;
    params.z                    = config.Z;
    params.outputs_per_cw       = ((config.Kb * config.Z) + 31) / 32;
    params.max_iters            = config.max_iterations;
    params.z4                   = config.Z * sizeof(int4);
    params.z2                   = config.Z * sizeof(int2);
    params.cz4                  = config.mb * params.z4;
    params.cz2                  = config.mb * params.z2;
    // c2v4 is used for fp16 kernels, c2v2 is used for fp32 kernels.
    // They aren't used at the same time (in the same kernel), so we set
    // them both to the workspace.
    params.c2v4                 = static_cast<char*>(workspace);
    params.c2v2                 = static_cast<char*>(workspace);
    if(CUPHY_R_16F == llrType)
    {
        const __half2 INV_NORM = __float2half2_rn(normalization);
        //params.inv_norm = 0x3c003c00;
        params.inv_norm = reinterpret_cast<const uint32_t&>(INV_NORM);
        gridDimX        = (config.num_codewords + 1) / 2;
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
                 params.outputs_per_cw,
                 gridDimX);

    //------------------------------------------------------------------
    // TODO: FP16 kernel decodes two codewords at a time, and assumes
    // that global memory will be addressable for an even number of
    // codewords. This may not be the case if memory for an odd number
    // of codewords was allocated.
    //------------------------------------------------------------------
    dim3 grdDim(gridDimX); // 1-D, NUM_CW for fp32, NUM_CW / 2 for fp16
    dim3 blkDim(round_up_to_next(config.Z, static_cast<int16_t>(32)));
    //------------------------------------------------------------------
    // TODO: Replace nested switch statements with lookup table over C, Z
    // (1 table for BG1, 4 tables for BG2 (different Kbs))
    switch(config.BG)
    {
    case 1:
        switch(llrType)
        {
        case CUPHY_R_16F:
            switch(config.Z)
            {
            case 384:
                switch(config.mb)
                {
                case 4:
                    ldpc_fast_layered_launch<uint16_t, 1, 4, 26, 384>(grdDim, blkDim, params, strm);
                    break;
                case 7:
                    ldpc_fast_layered_with_spills_launch<uint16_t, 1, 7, 29, 384, 1>(grdDim, blkDim, params, strm);
                    break;
                default:
                    return CUPHY_STATUS_UNSUPPORTED_CONFIG;
                }
                break;
            default:
                return CUPHY_STATUS_UNSUPPORTED_CONFIG;
            }
            break;
        case CUPHY_R_32F:
            switch(config.Z)
            {
            case 384:
                switch(config.mb)
                {
                case 4:
                    ldpc_fast_layered_launch<float, 1, 4, 26, 384>(grdDim, blkDim, params, strm);
                    break;
                case 7:
                    ldpc_fast_layered_with_spills_launch<float, 1, 7, 29, 384, 1>(grdDim, blkDim, params, strm);
                    break;
                default:
                    return CUPHY_STATUS_UNSUPPORTED_CONFIG;
                }
                break;
            default:
                return CUPHY_STATUS_UNSUPPORTED_CONFIG;
            }
            break;
        }
        break;
    case 2:
        switch(llrType)
        {
        case CUPHY_R_16F:
            switch(config.Z)
            {
            default:
                return CUPHY_STATUS_UNSUPPORTED_CONFIG;
            }
            break;
        case CUPHY_R_32F:
            switch(config.Z)
            {
            default:
                return CUPHY_STATUS_UNSUPPORTED_CONFIG;
            }
            break;
        }
        break;
    default:
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    //switch(llrType)
    //{
    //case CUPHY_R_16F:
    //    ldpc_msa_layered_kernel<uint16_t, 4, 26, 384><<<grdDim, blkDim, 0, strm>>>(params);
    //    break;
    //case CUPHY_R_32F:
    //    ldpc_msa_layered_kernel<float, 4, 26, 384><<<grdDim, blkDim, 0, strm>>>(params);
    //    break;
    //default:
    //    return CUPHY_STATUS_NOT_SUPPORTED;
    //}
#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

//----------------------------------------------------------------------
// decode_fast_layered_workspace_size()
std::pair<bool, size_t> decode_fast_layered_workspace_size(const LDPC_config& cfg)
{
    // TODO: Develop a more rigorous test to determine whether or not
    // this implementation is supported.
    if(cfg.mb == 4)
    {
        // ldpc_msa_layered_kernel: No global workspace data required
        return std::pair<bool, size_t>(true, 0);
    }
    else
    {
        const size_t MB_Z_NUMCW = cfg.num_codewords * cfg.mb * cfg.Z;
        // TODO: Julien's code allocated twice as much for fp16. Is that
        // right?
        if(CUPHY_R_16F == cfg.type)
        {
            return std::pair<bool, size_t>(true, MB_Z_NUMCW * sizeof(int4));
        }
        else if(CUPHY_R_32F == cfg.type)
        {
            return std::pair<bool, size_t>(true, MB_Z_NUMCW * sizeof(int2));
        }
        else
        {
            // Only fp16 and fp32 are supported
            return std::pair<bool, size_t>(false, 0);
        }

    }
}

} // namespace ldpc
