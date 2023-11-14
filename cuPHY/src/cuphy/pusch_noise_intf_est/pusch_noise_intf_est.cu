/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
 #include <cooperative_groups/reduce.h>
 #include <cooperative_groups/memcpy_async.h>
 #include <cuda/pipeline>
 #include "cuphy_kernel_util.cuh"
 #include "descrambling.cuh"
 #include "pusch_noise_intf_est.hpp"
 #include "type_convert.hpp"
 #include "utils.cuh"
 #include "nvlog.hpp"
 #include "cuphy.hpp"

 namespace cg = cooperative_groups;

 namespace pusch_noise_intf_est
 {
 // #define ENABLE_PROFILING
 // #define ENABLE_DEBUG
 // #define ENABLE_PRIME_WHILE_LOOP
 #define ENALBE_COMMON_DFTSOFDM_DESCRCODE_SUBROUTINE // if ENALBE_COMMON_DFTSOFDM_DESCRCODE_SUBROUTINE is defined, ENABLE_PRIME_WHILE_LOOP should not be defined. 

 // static constexpr uint32_t N_THREADS_PER_WARP = 32;
 // static constexpr float NOISE_REGULARIZER = 0.00001; //small noise regularizer corresponding to -50dB SNR
 
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
 
     CUDA_BOTH T&               operator[](int idx) { return m_pData[idx]; }
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
 template <typename T> CUDA_BOTH_INLINE constexpr T     cuGet(uint32_t);
 template<>            CUDA_BOTH_INLINE constexpr float cuGet(uint32_t x) { return(float(x)); }
 
 template <typename T> CUDA_BOTH_INLINE T         cuGet(int);
 template<>            CUDA_BOTH_INLINE float     cuGet(int x) { return(float(x)); }
 template<>            CUDA_BOTH_INLINE cuComplex cuGet(int x) { return(make_cuComplex(float(x), 0.0f)); }
 
 template <typename T> CUDA_BOTH_INLINE T         cuGet(float);
 template<>            CUDA_BOTH_INLINE cuComplex cuGet(float x) { return(make_cuComplex(x, 0.0f)); }
 
 template <typename T> CUDA_BOTH_INLINE T         cuGet(float,float);
 template <>           CUDA_BOTH_INLINE cuComplex cuGet<cuComplex>(float x, float y) { return make_cuComplex(x,y); }
 
 static CUDA_BOTH_INLINE float     cuReal(cuComplex x) { return(cuCrealf(x)); }
 static CUDA_BOTH_INLINE float     cuImag(cuComplex x) { return(cuCimagf(x)); }
 static CUDA_BOTH_INLINE cuComplex cuConj(cuComplex x) { return(cuConjf(x)); }
 
 static CUDA_BOTH_INLINE cuComplex operator-(cuComplex x, cuComplex y) { return(cuCsubf(x, y)); }
 
 static CUDA_BOTH_INLINE cuComplex operator+=(cuComplex &x, cuComplex y) { x = cuCaddf(x, y); return x; };

 static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, float y)       { return(make_cuComplex(cuCrealf(x)*y, cuCimagf(x)*y)); }
 // static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, double y)      { return(make_cuComplex(static_cast<float>(cuCrealf(x)*y), static_cast<float>(cuCimagf(x)*y))); }
 static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, cuComplex y)   { return(cuCmulf(x, y)); }

 static CUDA_BOTH_INLINE cuComplex operator/(cuComplex x, float y)       { return(make_cuComplex(cuCrealf(x)/y, cuCimagf(x)/y)); }
 // clang-format on
  
 template <typename T>
 CUDA_BOTH_INLINE constexpr T div_round_up(T val, T divide_by)
 {
     return ((val + (divide_by - 1)) / divide_by);
 } 
 
static __device__ __constant__ int8_t d_phi_6[30][6];
static int8_t                         phi_6[30][6] = {{-3, -1, 3, 3, -1, -3},
                              {-3, 3, -1, -1, 3, -3},
                              {-3, -3, -3, 3, 1, -3},
                              {1, 1, 1, 3, -1, -3},
                              {1, 1, 1, -3, -1, 3},
                              {-3, 1, -1, -3, -3, -3},
                              {-3, 1, 3, -3, -3, -3},
                              {-3, -1, 1, -3, 1, -1},
                              {-3, -1, -3, 1, -3, -3},
                              {-3, -3, 1, -3, 3, -3},
                              {-3, 1, 3, 1, -3, -3},
                              {-3, -1, -3, 1, 1, -3},
                              {1, 1, 3, -1, -3, 3},
                              {1, 1, 3, 3, -1, 3},
                              {1, 1, 1, -3, 3, -1},
                              {1, 1, 1, -1, 3, -3},
                              {-3, -1, -1, -1, 3, -1},
                              {-3, -3, -1, 1, -1, -3},
                              {-3, -3, -3, 1, -3, -1},
                              {-3, 1, 1, -3, -1, -3},
                              {-3, 3, -3, 1, 1, -3},
                              {-3, 1, -3, -3, -3, -1},
                              {1, 1, -3, 3, 1, 3},
                              {1, 1, -3, -3, 1, -3},
                              {1, 1, 3, -1, 3, 3},
                              {1, 1, -3, 1, 3, 3},
                              {1, 1, -1, -1, 3, -1},
                              {1, 1, -1, 3, -1, -1},
                              {1, 1, -1, 3, -3, -1},
                              {1, 1, -3, 1, -1, -1}};

static __device__ __constant__ int8_t d_phi_12[30][12];
static int8_t                         phi_12[30][12] = {{-3, 1, -3, -3, -3, 3, -3, -1, 1, 1, 1, -3},
                                {-3, 3, 1, -3, 1, 3, -1, -1, 1, 3, 3, 3},
                                {-3, 3, 3, 1, -3, 3, -1, 1, 3, -3, 3, -3},
                                {-3, -3, -1, 3, 3, 3, -3, 3, -3, 1, -1, -3},
                                {-3, -1, -1, 1, 3, 1, 1, -1, 1, -1, -3, 1},
                                {-3, -3, 3, 1, -3, -3, -3, -1, 3, -1, 1, 3},
                                {1, -1, 3, -1, -1, -1, -3, -1, 1, 1, 1, -3},
                                {-1, -3, 3, -1, -3, -3, -3, -1, 1, -1, 1, -3},
                                {-3, -1, 3, 1, -3, -1, -3, 3, 1, 3, 3, 1},
                                {-3, -1, -1, -3, -3, -1, -3, 3, 1, 3, -1, -3},
                                {-3, 3, -3, 3, 3, -3, -1, -1, 3, 3, 1, -3},
                                {-3, -1, -3, -1, -1, -3, 3, 3, -1, -1, 1, -3},
                                {-3, -1, 3, -3, -3, -1, -3, 1, -1, -3, 3, 3},
                                {-3, 1, -1, -1, 3, 3, -3, -1, -1, -3, -1, -3},
                                {1, 3, -3, 1, 3, 3, 3, 1, -1, 1, -1, 3},
                                {-3, 1, 3, -1, -1, -3, -3, -1, -1, 3, 1, -3},
                                {-1, -1, -1, -1, 1, -3, -1, 3, 3, -1, -3, 1},
                                {-1, 1, 1, -1, 1, 3, 3, -1, -1, -3, 1, -3},
                                {-3, 1, 3, 3, -1, -1, -3, 3, 3, -3, 3, -3},
                                {-3, -3, 3, -3, -1, 3, 3, 3, -1, -3, 1, -3},
                                {3, 1, 3, 1, 3, -3, -1, 1, 3, 1, -1, -3},
                                {-3, 3, 1, 3, -3, 1, 1, 1, 1, 3, -3, 3},
                                {-3, 3, 3, 3, -1, -3, -3, -1, -3, 1, 3, -3},
                                {3, -1, -3, 3, -3, -1, 3, 3, 3, -3, -1, -3},
                                {-3, -1, 1, -3, 1, 3, 3, 3, -1, -3, 3, 3},
                                {-3, 3, 1, -1, 3, 3, -3, 1, -1, 1, -1, 1},
                                {-1, 1, 3, -3, 1, -1, 1, -1, -1, -3, 1, -1},
                                {-3, -3, 3, 3, 3, -3, -1, 1, -3, 3, 1, -3},
                                {1, -1, 3, 1, 1, -1, -1, -1, 1, 3, -3, 1},
                                {-3, 3, -3, 3, -3, -3, 3, -1, -1, 1, 3, -3}};

static __device__ __constant__ int8_t d_phi_18[30][18];
static int8_t                         phi_18[30][18] = {{-1, 3, -1, -3, 3, 1, -3, -1, 3, -3, -1, -1, 1, 1, 1, -1, -1, -1},
                                {3, -3, 3, -1, 1, 3, -3, -1, -3, -3, -1, -3, 3, 1, -1, 3, -3, 3},
                                {-3, 3, 1, -1, -1, 3, -3, -1, 1, 1, 1, 1, 1, -1, 3, -1, -3, -1},
                                {-3, -3, 3, 3, 3, 1, -3, 1, 3, 3, 1, -3, -3, 3, -1, -3, -1, 1},
                                {1, 1, -1, -1, -3, -1, 1, -3, -3, -3, 1, -3, -1, -1, 1, -1, 3, 1},
                                {3, -3, 1, 1, 3, -1, 1, -1, -1, -3, 1, 1, -1, 3, 3, -3, 3, -1},
                                {-3, 3, -1, 1, 3, 1, -3, -1, 1, 1, -3, 1, 3, 3, -1, -3, -3, -3},
                                {1, 1, -3, 3, 3, 1, 3, -3, 3, -1, 1, 1, -1, 1, -3, -3, -1, 3},
                                {-3, 1, -3, -3, 1, -3, -3, 3, 1, -3, -1, -3, -3, -3, -1, 1, 1, 3},
                                {3, -1, 3, 1, -3, -3, -1, 1, -3, -3, 3, 3, 3, 1, 3, -3, 3, -3},
                                {-3, -3, -3, 1, -3, 3, 1, 1, 3, -3, -3, 1, 3, -1, 3, -3, -3, 3},
                                {-3, -3, 3, 3, 3, -1, -1, -3, -1, -1, -1, 3, 1, -3, -3, -1, 3, -1},
                                {-3, -1, -3, -3, 1, 1, -1, -3, -1, -3, -1, -1, 3, 3, -1, 3, 1, 3},
                                {1, 1, -3, -3, -3, -3, 1, 3, -3, 3, 3, 1, -3, -1, 3, -1, -3, 1},
                                {-3, 3, -1, -3, -1, -3, 1, 1, -3, -3, -1, -1, 3, -3, 1, 3, 1, 1},
                                {3, 1, -3, 1, -3, 3, 3, -1, -3, -3, -1, -3, -3, 3, -3, -1, 1, 3},
                                {-3, -1, -3, -1, -3, 1, 3, -3, -1, 3, 3, 3, 1, -1, -3, 3, -1, -3},
                                {-3, -1, 3, 3, -1, 3, -1, -3, -1, 1, -1, -3, -1, -1, -1, 3, 3, 1},
                                {-3, 1, -3, -1, -1, 3, 1, -3, -3, -3, -1, -3, -3, 1, 1, 1, -1, -1},
                                {3, 3, 3, -3, -1, -3, -1, 3, -1, 1, -1, -3, 1, -3, -3, -1, 3, 3},
                                {-3, 1, 1, -3, 1, 1, 3, -3, -1, -3, -1, 3, -3, 3, -1, -1, -1, -3},
                                {1, -3, -1, -3, 3, 3, -1, -3, 1, -3, -3, -1, -3, -1, 1, 3, 3, 3},
                                {-3, -3, 1, -1, -1, 1, 1, -3, -1, 3, 3, 3, 3, -1, 3, 1, 3, 1},
                                {3, -1, -3, 1, -3, -3, -3, 3, 3, -1, 1, -3, -1, 3, 1, 1, 3, 3},
                                {3, -1, -1, 1, -3, -1, -3, -1, -3, -3, -1, -3, 1, 1, 1, -3, -3, 3},
                                {-3, -3, 1, -3, 3, 3, 3, -1, 3, 1, 1, -3, -3, -3, 3, -3, -1, -1},
                                {-3, -1, -1, -3, 1, -3, 3, -1, -1, -3, 3, 3, -3, -1, 3, -1, -1, -1},
                                {-3, -3, 3, 3, -3, 1, 3, -1, -3, 1, -1, -3, 3, -3, -1, -1, -1, 3},
                                {-1, -3, 1, -3, -3, -3, 1, 1, 3, 3, -3, 3, 3, -3, -1, 3, -3, 1},
                                {-3, 3, 1, -1, -1, -1, -1, 1, -1, 3, 3, -3, -1, 1, 3, -1, 3, -1}};

static __device__ __constant__ int8_t d_phi_24[30][24];
static int8_t                         phi_24[30][24] = {{-1, -3, 3, -1, 3, 1, 3, -1, 1, -3, -1, -3, -1, 1, 3, -3, -1, -3, 3, 3, 3, -3, -3, -3},
                                {-1, -3, 3, 1, 1, -3, 1, -3, -3, 1, -3, -1, -1, 3, -3, 3, 3, 3, -3, 1, 3, 3, -3, -3},
                                {-1, -3, -3, 1, -1, -1, -3, 1, 3, -1, -3, -1, -1, -3, 1, 1, 3, 1, -3, -1, -1, 3, -3, -3},
                                {1, -3, 3, -1, -3, -1, 3, 3, 1, -1, 1, 1, 3, -3, -1, -3, -3, -3, -1, 3, -3, -1, -3, -3},
                                {-1, 3, -3, -3, -1, 3, -1, -1, 1, 3, 1, 3, -1, -1, -3, 1, 3, 1, -1, -3, 1, -1, -3, -3},
                                {-3, -1, 1, -3, -3, 1, 1, -3, 3, -1, -1, -3, 1, 3, 1, -1, -3, -1, -3, 1, -3, -3, -3, -3},
                                {-3, 3, 1, 3, -1, 1, -3, 1, -3, 1, -1, -3, -1, -3, -3, -3, -3, -1, -1, -1, 1, 1, -3, -3},
                                {-3, 1, 3, -1, 1, -1, 3, -3, 3, -1, -3, -1, -3, 3, -1, -1, -1, -3, -1, -1, -3, 3, 3, -3},
                                {-3, 1, -3, 3, -1, -1, -1, -3, 3, 1, -1, -3, -1, 1, 3, -1, 1, -1, 1, -3, -3, -3, -3, -3},
                                {1, 1, -1, -3, -1, 1, 1, -3, 1, -1, 1, -3, 3, -3, -3, 3, -1, -3, 1, 3, -3, 1, -3, -3},
                                {-3, -3, -3, -1, 3, -3, 3, 1, 3, 1, -3, -1, -1, -3, 1, 1, 3, 1, -1, -3, 3, 1, 3, -3},
                                {-3, 3, -1, 3, 1, -1, -1, -1, 3, 3, 1, 1, 1, 3, 3, 1, -3, -3, -1, 1, -3, 1, 3, -3},
                                {3, -3, 3, -1, -3, 1, 3, 1, -1, -1, -3, -1, 3, -3, 3, -1, -1, 3, 3, -3, -3, 3, -3, -3},
                                {-3, 3, -1, 3, -1, 3, 3, 1, 1, -3, 1, 3, -3, 3, -3, -3, -1, 1, 3, -3, -1, -1, -3, -3},
                                {-3, 1, -3, -1, -1, 3, 1, 3, -3, 1, -1, 3, 3, -1, -3, 3, -3, -1, -1, -3, -3, -3, 3, -3},
                                {-3, -1, -1, -3, 1, -3, -3, -1, -1, 3, -1, 1, -1, 3, 1, -3, -1, 3, 1, 1, -1, -1, -3, -3},
                                {-3, -3, 1, -1, 3, 3, -3, -1, 1, -1, -1, 1, 1, -1, -1, 3, -3, 1, -3, 1, -1, -1, -1, -3},
                                {3, -1, 3, -1, 1, -3, 1, 1, -3, -3, 3, -3, -1, -1, -1, -1, -1, -3, -3, -1, 1, 1, -3, -3},
                                {-3, 1, -3, 1, -3, -3, 1, -3, 1, -3, -3, -3, -3, -3, 1, -3, -3, 1, 1, -3, 1, 1, -3, -3},
                                {-3, -3, 3, 3, 1, -1, -1, -1, 1, -3, -1, 1, -1, 3, -3, -1, -3, -1, -1, 1, -3, 3, -1, -3},
                                {-3, -3, -1, -1, -1, -3, 1, -1, -3, -1, 3, -3, 1, -3, 3, -3, 3, 3, 1, -1, -1, 1, -3, -3},
                                {3, -1, 1, -1, 3, -3, 1, 1, 3, -1, -3, 3, 1, -3, 3, -1, -1, -1, -1, 1, -3, -3, -3, -3},
                                {-3, 1, -3, 3, -3, 1, -3, 3, 1, -1, -3, -1, -3, -3, -3, -3, 1, 3, -1, 1, 3, 3, 3, -3},
                                {-3, -1, 1, -3, -1, -1, 1, 1, 1, 3, 3, -1, 1, -1, 1, -1, -1, -3, -3, -3, 3, 1, -1, -3},
                                {-3, 3, -1, -3, -1, -1, -1, 3, -1, -1, 3, -3, -1, 3, -3, 3, -3, -1, 3, 1, 1, -1, -3, -3},
                                {-3, 1, -1, -3, -3, -1, 1, -3, -1, -3, 1, 1, -1, 1, 1, 3, 3, 3, -1, 1, -1, 1, -1, -3},
                                {-1, 3, -1, -1, 3, 3, -1, -1, -1, 3, -1, -3, 1, 3, 1, 1, -3, -3, -3, -1, -3, -1, -3, -3},
                                {3, -3, -3, -1, 3, 3, -3, -1, 3, 1, 1, 1, 3, -1, 3, -3, -1, 3, -1, 3, 1, -1, -3, -3},
                                {-3, 1, -3, 1, -3, 1, 1, 3, 1, -3, -3, -1, 1, 3, -1, -3, 3, 1, -1, -3, -3, -3, -3, -3},
                                {3, -3, -1, 1, 3, -1, -1, -3, -1, 3, -1, -3, -1, -3, 3, -1, 3, 1, 1, -3, 3, -3, -3, -3}};

#ifdef ENABLE_PRIME_WHILE_LOOP
    static __device__ __constant__ uint16_t d_primeNums[303];
    static uint16_t                         primeNums[303] = {2,3,5,7,11,13,17,19,23,29,31,37,41,43,47,53,59,61,67,71,73,79,83,89,97,101,103,107,109,113,127,131,137,139,149,151,157,163,167,173,179,181,191,193,197,199,211,223,227,229,233,239,241,251,257,263,269,271,277,281,283,293,307,311,313,317,331,337,347,349,353,359,367,373,379,383,389,397,401,409,419,421,431,433,439,443,449,457,461,463,467,479,487,491,499,503,509,521,523,541,547,557,563,569,571,577,587,593,599,601,607,613,617,619,631,641,643,647,653,659,661,673,677,683,691,701,709,719,727,733,739,743,751,757,761,769,773,787,797,809,811,821,823,827,829,839,853,857,859,863,877,881,883,887,907,911,919,929,937,941,947,953,967,971,977,983,991,997,1009,1013,1019,1021,1031,1033,1039,1049,1051,1061,1063,1069,1087,1091,1093,1097,1103,1109,1117,1123,1129,1151,1153,1163,1171,1181,1187,1193,1201,1213,1217,1223,1229,1231,1237,1249,1259,1277,1279,1283,1289,1291,1297,1301,1303,1307,1319,1321,1327,1361,1367,1373,1381,1399,1409,1423,1427,1429,1433,1439,1447,1451,1453,1459,1471,1481,1483,1487,1489,1493,1499,1511,1523,1531,1543,1549,1553,1559,1567,1571,1579,1583,1597,1601,1607,1609,1613,1619,1621,1627,1637,1657,1663,1667,1669,1693,1697,1699,1709,1721,1723,1733,1741,1747,1753,1759,1777,1783,1787,1789,1801,1811,1823,1831,1847,1861,1867,1871,1873,1877,1879,1889,1901,1907,1913,1931,1933,1949,1951,1973,1979,1987,1993,1997,1999};
#else
// d_primeNums[273] is a precalculated table of prime numbers per nPrb to avoid runtime search based on M_ZC implemented as a while loop when ENABLE_PRIME_WHILE_LOOP is not defined.
// M_ZC = N_DMRS_GRID_TONES_PER_PRB * nPrb;
    static __device__ __constant__ uint16_t d_primeNums[273]; 
    static uint16_t                         primeNums[273] = {5 ,11 ,17 ,23 ,29 ,31 ,41 ,47 ,53 ,59 ,61 ,71 ,73 ,83 ,89 ,89 ,101 ,107 ,113 ,113 ,113 ,131 ,137 ,139 ,149 ,151 ,157 ,167 ,173 ,179 ,181 ,191 ,197 ,199 ,199 ,211 ,211 ,227 ,233 ,239 ,241 ,251 ,257 ,263 ,269 ,271 ,281 ,283 ,293 ,293 ,293 ,311 ,317 ,317 ,317 ,331 ,337 ,347 ,353 ,359 ,359 ,367 ,373 ,383 ,389 ,389 ,401 ,401 ,409 ,419 ,421 ,431 ,433 ,443 ,449 ,449 ,461 ,467 ,467 ,479 ,479 ,491 ,491 ,503 ,509 ,509 ,521 ,523 ,523 ,523 ,541 ,547 ,557 ,563 ,569 ,571 ,577 ,587 ,593 ,599 ,601 ,607 ,617 ,619 ,619 ,631 ,641 ,647 ,653 ,659 ,661 ,661 ,677 ,683 ,683 ,691 ,701 ,701 ,709 ,719 ,719 ,727 ,733 ,743 ,743 ,751 ,761 ,761 ,773 ,773 ,773 ,787 ,797 ,797 ,809 ,811 ,821 ,827 ,829 ,839 ,839 ,839 ,857 ,863 ,863 ,863 ,881 ,887 ,887 ,887 ,887 ,911 ,911 ,919 ,929 ,929 ,941 ,947 ,953 ,953 ,953 ,971 ,977 ,983 ,983 ,991 ,997 ,997 ,1013 ,1019 ,1021 ,1031 ,1033 ,1039 ,1049 ,1051 ,1061 ,1063 ,1069 ,1069 ,1069 ,1091 ,1097 ,1103 ,1109 ,1109 ,1117 ,1123 ,1129 ,1129 ,1129 ,1151 ,1153 ,1163 ,1163 ,1171 ,1181 ,1187 ,1193 ,1193 ,1201 ,1201 ,1217 ,1223 ,1229 ,1231 ,1237 ,1237 ,1249 ,1259 ,1259 ,1259 ,1277 ,1283 ,1289 ,1291 ,1301 ,1307 ,1307 ,1319 ,1321 ,1327 ,1327 ,1327 ,1327 ,1327 ,1361 ,1367 ,1373 ,1373 ,1381 ,1381 ,1381 ,1399 ,1409 ,1409 ,1409 ,1427 ,1433 ,1439 ,1439 ,1451 ,1453 ,1459 ,1459 ,1471 ,1481 ,1487 ,1493 ,1499 ,1499 ,1511 ,1511 ,1523 ,1523 ,1531 ,1531 ,1543 ,1553 ,1559 ,1559 ,1571 ,1571 ,1583 ,1583 ,1583 ,1601 ,1607 ,1613 ,1619 ,1621 ,1627 ,1637};
#endif

#ifndef ENALBE_COMMON_DFTSOFDM_DESCRCODE_SUBROUTINE
static inline __device__ float2 gen_pusch_dftsofdm_descrcode(uint16_t M_ZC, uint16_t rIdx, int u, int v, uint16_t nPrb)
{
    float2 descrCode;
    if(M_ZC < 36)
    {
        if(rIdx < M_ZC)
        {
            switch(M_ZC)
            {
            case 6: {
                descrCode.x =(float)cos(M_PI * (d_phi_6[u][rIdx]) / 4.0f);
                descrCode.y= (float)sin(M_PI * (d_phi_6[u][rIdx]) / 4.0f);
                break;
            }
            case 12: {
                descrCode.x =(float)cos(M_PI * (d_phi_12[u][rIdx]) / 4.0f);
                descrCode.y= (float)sin(M_PI * (d_phi_12[u][rIdx]) / 4.0f);
                break;
            }
            case 18: {
                descrCode.x =(float)cos(M_PI * (d_phi_18[u][rIdx]) / 4.0f);
                descrCode.y= (float)sin(M_PI * (d_phi_18[u][rIdx]) / 4.0f);
                break;
            }
            case 24: {
                descrCode.x =(float)cos(M_PI * (d_phi_24[u][rIdx]) / 4.0f);
                descrCode.y= (float)sin(M_PI * (d_phi_24[u][rIdx]) / 4.0f);
                break;
            }
            case 30: {
                descrCode.x =(float)cos(M_PI * (u + 1) * (rIdx + 1) * (rIdx + 2) / 31.0f);
                descrCode.y= (float)(-sin(M_PI * (u + 1) * (rIdx + 1) * (rIdx + 2) / 31.0f));
                break;
            }  
            } 
        }
    }
    else 
    {
    #ifdef ENABLE_PRIME_WHILE_LOOP
        int idx = 0;
        while(M_ZC > d_primeNums[idx])
        {
            idx++;
        }
        idx--;
        uint16_t d_primeNum = d_primeNums[idx];
    #else
        uint16_t d_primeNum = d_primeNums[nPrb-1];
    #endif
        float qbar = d_primeNum * (u + 1) / 31.0f;
        float q    = (int)(qbar + 0.5f) + (v * (((int)(2 * qbar) & 1) * -2 + 1));
        uint32_t m = rIdx % d_primeNum;
        descrCode.x =(float)cos(M_PI * q * m * (m + 1) / d_primeNum);
        descrCode.y= (float)(-sin(M_PI * q * m * (m + 1) / d_primeNum));
    }
    return descrCode;
}
#endif


// #define ENABLE_DEBUG

// Inplace lower triangular Cholesky factorization (output matrix overwrites input matrix)
// 1. Parallelism: Assumes atleast (matInOrder - 1) threads per PRB since the rows of triL below diagonal are  
// computed in-parallel (iterations over columns).
// 2. The function does not produce the lower triangular Cholesky factor in the conventional sense. 2a and 2b below 
// describe how the output differs from lower triangular Cholesky matrix.
// 2a. The inverse of the diagonal elements are stored in the output Cholesky factor to avoid its
// recomputation (i.e. repeating the divide operation) in triLInv
// 2b. To enable in-place and in-parallel (along columns) compute of the inverse of the Cholesky factor
// in triLInv, the lower triangular Cholesky factor elements are also stored in the upper triangular section 
// so that triL(i,j) = triL(j,i) (i.e. at their reflected locations along the diagonal)
template <typename TCompute>
__device__ void cholFactor(cg::thread_block const&                                   thisThrdBlk,
                           bool                                                      thrdActive,
                           uint32_t                                                  prbIdx,
                           uint32_t                                                  perPrbThrdIdx,
                           uint32_t                                                  matInOrder, // matrix order = # of rows = # of rows
                           tensor_ref<typename complex_from_scalar<TCompute>::type>& tMatInOut)  // input must be hermitian symmetric
{
    typedef typename complex_from_scalar<TCompute>::type TComplexCompute;
    tensor_ref<TComplexCompute>& tHermSymMat = tMatInOut;
    tensor_ref<TComplexCompute>& tTriLMat    = tMatInOut;

    // Iterate through the columns of input matrix
    for(int32_t j = 0; j < matInOrder; ++j)
    {
        // Step 1: Compute diagonal element of j-th column
        // One thread active per PRB
        if(thrdActive && (0 == perPrbThrdIdx))
        {
            // Sum of the squared absolute values upto the diagonal element in the j-th row in matrix Lw
            TCompute sumAbsSqr = cuGet<TCompute>(0);
            for(int32_t k = 0; k <= j - 1; ++k)
            {
                TComplexCompute triLMat_jk = tTriLMat(j, k, prbIdx);
                sumAbsSqr += cuReal(triLMat_jk*cuConj(triLMat_jk));
            }
            TCompute matIn_jj = cuReal(tHermSymMat(j, j, prbIdx));
            TCompute triLMat_jj = sqrtf(matIn_jj - sumAbsSqr);

            // Note: Save inverse of diagonal and not the diagonal directly.
            // Inverse used in computation of rest of lower diagonal column elements and used in 
            tTriLMat(j, j, prbIdx) = cuGet<TComplexCompute>(cuGet<TCompute>(1) / triLMat_jj);

#ifdef ENABLE_DEBUG
            printf("cuPHY::Pusch::noiseIntfEstKernel::cholFactor: TriL[%02d][%02d][%03d] = %015.12f + %015.12f HermSymMat = %015.12f sumAbsSqr = %010.7f triLMat_jj = %010.7f threadIdx (%d %d %d) blockIdx (%d %d %d)\n", j, j, prbIdx, cuReal(tTriLMat(j, j, prbIdx)), cuImag(tTriLMat(j, j, prbIdx)), matIn_jj, sumAbsSqr, triLMat_jj, threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
#endif // ENABLE_DEBUG
        }
        thisThrdBlk.sync();
    
        // Step 2: Compute rest of the elements below diagonal of j-th column (i.e. i = j+1 row onwards) in parallel
        // (matInOrder - j - 1) threads active per PRB
        if(thrdActive && (perPrbThrdIdx > j))
        {
            int32_t i = perPrbThrdIdx;
            TComplexCompute matIn_ij = tHermSymMat(i, j, prbIdx);
            TComplexCompute sum = cuGet<TComplexCompute>(0);
            for(int32_t k = 0; k <= j-1; ++k)
            {
                TComplexCompute triLMat_ik = tTriLMat(i, k, prbIdx);
                TComplexCompute triLMat_jk = tTriLMat(j, k, prbIdx);
                sum += triLMat_ik * cuConj(triLMat_jk);
            }
            tTriLMat(i, j, prbIdx) = (matIn_ij - sum) * tTriLMat(j, j, prbIdx);

            // Save the element in upper triangular portion for use later in parallel inverse computation
            tTriLMat(j, i, prbIdx) = tTriLMat(i, j, prbIdx);

#ifdef ENABLE_DEBUG
            printf("cuPHY::Pusch::noiseIntfEstKernel::cholFactor: TriL[%02d][%02d][%03d] = %015.12f + %015.12f sum = %015.12f + %015.12f tHermSymMat = %015.12f + %015.12f triLMat_jj_inv = %015.12f\n", i, j, prbIdx, cuReal(tTriLMat(i, j, prbIdx)), cuImag(tTriLMat(i, j, prbIdx)), cuReal(sum), cuImag(sum), cuReal(matIn_ij), cuImag(matIn_ij),  cuReal(tTriLMat(j, j, prbIdx)));
#endif // ENABLE_DEBUG
        }
        thisThrdBlk.sync();
    }
}

// Inplace computation of the inverse of lower triangular matrix (Output matrix overwrites input matrix)
// 1. Parallelism: Assumes atleast matInOrder threads per PRB since the columns of triLInv are computed in-parallel (iterations over rows).
// 2. Input matrix assumption: Diagonal elements are expected to contain the inverse of the diagonal elements of the Cholesky factor
// 3. Input matrix assumption: The upper triangular portion of the matrix (normally 0) is assumed to contain a reflection of the lower 
// triangular portion to aid in-place and in-parallel computation
template <typename TCompute>
__device__ void triLInv(cg::thread_block const&                                   thisThrdBlk,
                        bool                                                      thrdActive,
                        uint32_t                                                  prbIdx,
                        uint32_t                                                  perPrbThrdIdx,
                        uint32_t                                                  matInOrder,    // matrix order = # of rows = # of rows
                        tensor_ref<typename complex_from_scalar<TCompute>::type>& tTriLMatInOut) // input/output matrix
{
    typedef typename complex_from_scalar<TCompute>::type TComplexCompute;
    tensor_ref<TComplexCompute>& tTriLMat    = tTriLMatInOut;
    tensor_ref<TComplexCompute>& tTriLInvMat = tTriLMatInOut;

    uint32_t j = perPrbThrdIdx;

    // Skip Step 1 since the inverse of diagonal elements are assumed to be pre-compued in the input
#if 0    
    // Step 1: Compute all the diagonal elements of the inverse matrix (which is simply inverse of input diagonal elements)
    if(thrdActive)
    {
        TCompute triLMat_jj = cuReal(tTriLMat(j, j, prbIdx));
        TCompute triLMat_jj_inv = cuGet<TCompute>(1)/triLMat_jj;
        tTriLInvMat(j, j, prbIdx) = cuGet<TComplexCompute>(triLMat_jj_inv);
    }
    thisThrdBlk.sync();
#endif

    // Step 2: Compute the columns of triLInvMat in parallel with each column calculated by iterating row-by-row
    if(thrdActive)
    {
        // Note: iteration count starts at j + 1 (versus j) since diagonal element is already available in the input or computed by step 1
        for(int32_t i = j + 1; i < matInOrder; ++i)
        {
            TComplexCompute sum = cuGet<TComplexCompute>(0);
            for(int32_t k = 0; k <= i - 1; ++k)
            {
                // Note: reads from upper triangular portion
                TComplexCompute triLMat_ik = tTriLMat(k, i, prbIdx);

                // Note: Assumes that the diagonal locations tTriLMat(j, j) contain the inverse of the diagonal
                TComplexCompute triLInvMat_kj = tTriLInvMat(k, j, prbIdx);
                sum += triLMat_ik * triLInvMat_kj;
            }
            TComplexCompute triLInvMat_ij = sum * (cuGet<TCompute>(-1) * cuReal(tTriLInvMat(i, i, prbIdx)));
            tTriLInvMat(i, j, prbIdx) = triLInvMat_ij;
#ifdef ENABLE_DEBUG
            printf("cuPHY::Pusch::noiseIntfEstKernel::triLInv: TriLInv[%02d][%02d][%03d] = %015.12f + %015.12f sum = %015.12f + %015.12f triLMat_jj_inv = %015.12f\n", i, j, prbIdx, cuReal(tTriLInvMat(i, j, prbIdx)), cuImag(tTriLInvMat(i, j, prbIdx)), cuReal(sum), cuImag(sum), cuReal(tTriLInvMat(i, i, prbIdx)));
#endif // ENABLE_DEBUG
        }
    }
    thisThrdBlk.sync();
}

// Assumes atleast matInOrder threads per PRB
// All computations occur inplace in tMatIn (which could be in shared memory).
// Final result is copied to tTriLInvMat
template <typename TCompute,
          typename TStorageOut>
__device__ void cholFactorInv(uint32_t                                                     startPrb,
                              uint32_t                                                     nPrbThisThrdBlk,
                              uint32_t                                                     matInOrder, // matrix order = # of rows = # of rows
                              tensor_ref<typename complex_from_scalar<TCompute>::type>&    tMatIn,     // must be hermitian symmetric
                              tensor_ref<typename complex_from_scalar<TStorageOut>::type>& tTriLInvMat)
{
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    cg::thread_block const& thisThrdBlk = cg::this_thread_block();
    const uint32_t thrdIdx = thisThrdBlk.thread_rank();

    uint32_t perPrbThrdIdx = thrdIdx % matInOrder;
    uint32_t relPrbIdx = thrdIdx / matInOrder;
    uint32_t absPrbIdx = startPrb + relPrbIdx;

    // thrdActive ensures that PRB index is within limits and perPrbThrdIdx is within dimensions of input matrix
    bool thrdActive = (relPrbIdx < 1) && (relPrbIdx < nPrbThisThrdBlk) && (perPrbThrdIdx < matInOrder) ?
                      true : false;

#ifdef ENABLE_DEBUG
    if(0 == thrdIdx)
    {
        printf("cuPHY::Pusch::noiseIntfEstKernel::cholFactorInv: ueGrpIdx %d prbGrpIdx %d thrdActive %u relPrbIdx %u prbIdx %u nPrbThisThrdBlk %u perPrbThrdIdx %u matInOrder %u\n", blockIdx.y, blockIdx.x, thrdActive, relPrbIdx, prbIdx, nPrbThisThrdBlk, perPrbThrdIdx, matInOrder);
    }
#endif // ENABLE_DEBUG

    cholFactor<TCompute>(thisThrdBlk, thrdActive, relPrbIdx, perPrbThrdIdx, matInOrder, tMatIn);
    triLInv<TCompute>(thisThrdBlk, thrdActive, relPrbIdx, perPrbThrdIdx, matInOrder, tMatIn);

    // Write out resulting lower triangular matrix from input matrix (potentially in shared memory) to output matrix
    // (potentially in global memory)
    // @todo: could use all the threads for writeout
    for(int32_t colIdx = 0; colIdx < matInOrder; ++colIdx)
    {
        uint32_t rowIdx = perPrbThrdIdx;
        if(thrdActive && (rowIdx >= colIdx))
        {
            tTriLInvMat(rowIdx, colIdx, absPrbIdx) = type_convert<TComplexStorageOut>(tMatIn(rowIdx, colIdx, relPrbIdx));
            if(rowIdx > colIdx)
                tTriLInvMat(colIdx, rowIdx, absPrbIdx) = cuGet<TComplexStorageOut>(0);
#ifdef ENABLE_DEBUG
            printf("cuPHY::Pusch::noiseIntfEstKernel::cholFactorInv: TriLInv[%02d][%02d][%03d] = %015.12f + %015.12f shTriLInv %015.12f + %015.12f threadIdx (%d %d %d) blockIdx (%d %d %d)\n", rowIdx, colIdx, absPrbIdx, cuReal(tTriLInvMat(rowIdx, colIdx, absPrbIdx)), cuImag(tTriLInvMat(rowIdx, colIdx, absPrbIdx)), cuReal(tMatIn(rowIdx, colIdx, relPrbIdx)), cuImag(tMatIn(rowIdx, colIdx, relPrbIdx)), threadIdx.x, threadIdx.y, threadIdx.z, blockIdx.x, blockIdx.y, blockIdx.z);
#endif // ENABLE_DEBUG  
        }
    }

    thisThrdBlk.sync();
}

// blockDim: (N_SC_PER_THRD_BLK*nRxAnt)
// gridDim:  (nThrdBlksPerUeGrp, nUeGrps)
// Note: tInfoPreEqNoiseVar and tInfoNoiseIntfEstInterCtaSyncCnt need to be reset to 0
 template <typename TStorageIn,
           typename TDataRx,
           typename TStorageOut,
           typename TCompute,
           uint32_t N_DMRS_GRIDS_PER_PRB,          // number of DMRS grids per PRB (2 or 3)        
           uint32_t N_PRB_PER_THRD_BLK,            // # of PRBs processed in a thread block
           uint8_t  DMRS_SYMBOL_IDX>               // the index of DMRS symbol to be processed
 __global__ void noiseIntfEstNoDftSOfdmKernel(puschRxNoiseIntfEstDynDescr_t* pDynDescr)
  {
    KERNEL_PRINT_GRID_ONCE("%s\n grid = (%u %u %u), block = (%u %u %u)\n",
                           __PRETTY_FUNCTION__,
                           gridDim.x, gridDim.y, gridDim.z,
                           blockDim.x, blockDim.y, blockDim.z);

    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TDataRx>::type     TComplexDataRx;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;    

    static_assert((2 == N_DMRS_GRIDS_PER_PRB), "DMRS grids per PRB other than 2 not supported");
    
    static_assert(((CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0 == DMRS_SYMBOL_IDX) || (CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT == DMRS_SYMBOL_IDX)), "Each kernel processes the first DMRS symbol for early-HARQ or all DMRS symbols for full-slot");

    // For DMRS config type 1, the LSB 3 bits of portId can be used to determine fOCC, grid, tOCC as follows:
    // bit 0 fOCC, bit 1 grid, bit 2 tOCC
    static constexpr uint32_t PORT_IDX_FOCC_MSK = 0x1;
    static constexpr uint32_t PORT_IDX_GRID_MSK = 0x2;
    static constexpr uint32_t PORT_IDX_TOCC_MSK = 0x4;

    //--------------------------------------------------------------------------------------------------------
    const uint32_t prbGrpIdx = blockIdx.x;
    const uint32_t ueGrpIdx  = blockIdx.y;

    cg::thread_block const& thisThrdBlk = cg::this_thread_block();
    const uint32_t thrdIdx = thisThrdBlk.thread_rank();
    const uint32_t nThrds = thisThrdBlk.size();

    puschRxNoiseIntfEstDynDescr_t& dynDescr = *(pDynDescr);
    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[ueGrpIdx];

    // Number of thread blocks needed to process this user group
    const uint32_t nPrb = drvdUeGrpPrms.nPrb;    
    uint32_t nUeGrpThrdBlksNeeded = div_round_up(nPrb, N_PRB_PER_THRD_BLK);

    // The number of thread blocks are sized to process UE group with largest PRB allocation
    // Early exit thread blocks which exceed those needed to process PRBs for current UE group
    if(blockIdx.x >= nUeGrpThrdBlksNeeded) return;

    // Pointer to DMRS symbol used for channel estimation (single-symbol if maxLen == 1, double-symbol if maxLen == 2)
    uint8_t const* pDmrsSymPos = drvdUeGrpPrms.dmrsSymLoc;
    const uint16_t slotNum  = drvdUeGrpPrms.slotNum;
    const uint16_t startPrb = drvdUeGrpPrms.startPrb;
    const uint16_t startSc = startPrb * CUPHY_N_TONES_PER_PRB;
    // const uint16_t nSc = nPrb * CUPHY_N_TONES_PER_PRB;

    // Number of PRBs already processed before this thread block
    const uint32_t nPrbProcessed       = prbGrpIdx * N_PRB_PER_THRD_BLK;
    const uint32_t startPrbThisThrdBlk = nPrbProcessed;
    const uint32_t absStartPrbThisThrdBlk = startPrb + startPrbThisThrdBlk;

    // Number of PRBs remaining to be processed from this thread block onwards
    const int32_t nPrbRemaining = nPrb - nPrbProcessed;
    // Calculate loop count - PRBs to be processed by this thread block
    const int32_t nPrbThisThrdBlk = (nPrbRemaining <= N_PRB_PER_THRD_BLK) ? nPrbRemaining : N_PRB_PER_THRD_BLK;  // std::min(nPrbRemaining, N_PRB_PER_THRD_BLK)
    const int32_t nScThisThrdBlk  = nPrbThisThrdBlk * CUPHY_N_TONES_PER_PRB;

    const uint8_t nRxAnt  = drvdUeGrpPrms.nRxAnt;
    const uint8_t nLayers = drvdUeGrpPrms.nLayers;

    const uint8_t dmrsMaxLen  = drvdUeGrpPrms.dmrsMaxLen;
    uint8_t nDmrsSyms   = drvdUeGrpPrms.nDmrsSyms; // nDmrsAddlnPos * dmrsMaxLen;
    uint8_t dmrsSymStart = 0;
    uint8_t dmrsSymStop  = nDmrsSyms;
    if(DMRS_SYMBOL_IDX==CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0)
    {
        dmrsSymStop    = dmrsMaxLen;
    }
    const uint16_t dmrsScrmId = drvdUeGrpPrms.dmrsScrmId;
    const uint8_t  nDmrsCdmGrpsNoData = drvdUeGrpPrms.nDmrsCdmGrpsNoData;
    
    uint8_t const* pDmrsPortIdxs = drvdUeGrpPrms.dmrsPortIdxs;  // Layer to port map
    const uint8_t   scid         = drvdUeGrpPrms.scid;
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
    uint16_t* pAbsUeIdxs         = &drvdUeGrpPrms.ueIdxs[0];
    uint32_t  nUes               = drvdUeGrpPrms.nUes; // Number of UEs in this UE group
#endif

    // Flag if noise covariance computed
    bool noiseCovFlag = (drvdUeGrpPrms.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC) ? true : false;


#ifdef ENABLE_DEBUG
    if(0 == thrdIdx)
    {
        printf("cuPHY::Pusch::noiseIntfEstKernel: ueGrpIdx %d nUeGrpThrdBlksNeeded %d prbGrpIdx %d nPrbRemaining %d nPrbThisThrdBlk %d nDmrsSyms %d dmrsMaxLen %d dmrsAddlnPos %d\n", ueGrpIdx, nUeGrpThrdBlksNeeded, prbGrpIdx, nPrbRemaining, nPrbThisThrdBlk, nDmrsSyms, dmrsMaxLen, drvdUeGrpPrms.dmrsAddlnPos);
    }
#endif // ENABLE_DEBUG  

    //--------------------------------------------------------------------------------------------------------
    // Number of DMRS tones to be processed by this thread block:

    // Number of tones per DMRS grid in a PRB
    static constexpr uint32_t N_DMRS_GRID_TONES_PER_PRB = CUPHY_N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB;

    // Number of tones per DMRS grid processed by this thread block
    static constexpr uint32_t N_DMRS_GRID_TONES = N_DMRS_GRID_TONES_PER_PRB * N_PRB_PER_THRD_BLK;

    // Total number of DMRS tones processed by this thread block
    static constexpr uint32_t N_DMRS_TONES = N_DMRS_GRID_TONES * N_DMRS_GRIDS_PER_PRB; // equal to CUPHY_N_TONES_PER_PRB * N_PRB_PER_THRD_BLK

    // Number of tones processed by the thread block per iteration
    static constexpr uint32_t N_SC_PER_THRD_BLK = CUPHY_N_TONES_PER_PRB;

    //--------------------------------------------------------------------------------------------------------
    // DMRS scrambling

    static constexpr uint32_t N_DMRS_SCR_BITS_PER_TONE = 2; // 1bit for I and 1 bit for Q
    static constexpr uint32_t N_DMRS_SCR_BITS = N_DMRS_SCR_BITS_PER_TONE * N_DMRS_GRID_TONES;
    
    // Number of DMRS scrambler bits generated by one call to gold32 by one thread
    static constexpr uint32_t N_DMRS_SCR_BITS_GEN_PER_THRD = 32;

    // Round up to the next multiple of N_DMRS_SCR_BITS_GEN_PER_THRD plus 1 (+1 because DMRS_SCR_PRB_CLUSTER_START_BIT_OFFSET
    // may be large enough to spill the scrambler bits to the next word)    
    static constexpr uint32_t N_DMRS_SCR_WORDS = div_round_up<uint32_t>(N_DMRS_SCR_BITS, N_DMRS_SCR_BITS_GEN_PER_THRD) + 1;

    // Section 5.2.1 in 3GPP TS 38.211
    // The fast-forward of 1600 prescribed by spec is already baked into the gold sequence generator
    static constexpr uint32_t DMRS_SCR_FF = 0; // 1600;
 
    // Absolute index of scrambling sequence start
    // Note:The DMRS scrambling sequence is the same for all the DMRS grids. There are 2 sequences one for
    // scid 0 and other for scid 1 but the same sequences is reused for all DMRS grids    
    const uint32_t dmrsScrSeqStartIdx = absStartPrbThisThrdBlk * N_DMRS_GRID_TONES_PER_PRB;

    // First scrambler bit index needed by this thread block
    const uint32_t dmrsScrStartBit = DMRS_SCR_FF + (dmrsScrSeqStartIdx * N_DMRS_SCR_BITS_PER_TONE);
 
    // The scrambling sequence generator outputs 32 scrambler bits at a time. Thus, compute the earliest
    // multiple of 32 bits which contains the scrambler bit of the first tone in the PRB cluster as the
    // start index
    const uint32_t dmrsScrGenAlignedStartBit = (dmrsScrStartBit / N_DMRS_SCR_BITS_GEN_PER_THRD) * N_DMRS_SCR_BITS_GEN_PER_THRD;
    
    // Offset to scrambler bit of the first tone in the PRB cluster
    const uint32_t dmrsScrStartBitOffset = dmrsScrStartBit - dmrsScrGenAlignedStartBit;

    // Since each thread generates N_DMRS_SCR_BITS_GEN_PER_THRD bits and each DMRS tone needs N_DMRS_SCR_BITS_PER_TONE bits to be scrambled,
    // compute the number of iterations needed to generated all the scrambler bits
    static constexpr uint32_t N_DMRS_SCR_BITS_NEEDED = N_DMRS_TONES*N_DMRS_SCR_BITS_PER_TONE;
    const uint32_t nDmrsScrBitsGenPerThrdBlkIter = nThrds*N_DMRS_SCR_BITS_GEN_PER_THRD;
    const uint32_t nDmrsScrBitGenIter = div_round_up(N_DMRS_SCR_BITS_NEEDED, nDmrsScrBitsGenPerThrdBlkIter);

    static constexpr TCompute RECIPROCAL_SQRT2 = 0.7071068f;

    // add 0.5dB to noise estimate to account for noise filtered out by the channel interpolation filter
    static constexpr TCompute NOISE_EST_CORRECTION_DB = 0.5f;

    // Note: if nDmrsCdmGrpsNoData = 2 then DMRS power = 2 (DMRS amplitude = sqrt(2)) 
    //       if nDmrsCdmGrpsNoData = 1 then DMRS power = 1
    const TCompute dmrsScale = (2 == nDmrsCdmGrpsNoData) ? 1.4142135f : 1.0f;
    
    //--------------------------------------------------------------------------------------------------------
    tensor_ref<const TComplexDataRx>    tDataRx         (drvdUeGrpPrms.tInfoDataRx.pAddr       , drvdUeGrpPrms.tInfoDataRx.strides);// (NF, ND, N_BS_ANTS)
    tensor_ref<const TComplexStorageIn> tHEst           (drvdUeGrpPrms.tInfoHEst.pAddr         , drvdUeGrpPrms.tInfoHEst.strides);  // (N_BS_ANTS, N_LAYERS, NF, NH)
    tensor_ref<volatile TStorageOut>    tNoiseIntfVar   (drvdUeGrpPrms.tInfoNoiseVarPreEq.pAddr, drvdUeGrpPrms.tInfoNoiseVarPreEq.strides); // (N_UE_GRP)
    tensor_ref<TComplexStorageOut>      tLwInv          (drvdUeGrpPrms.tInfoLwInv.pAddr        , drvdUeGrpPrms.tInfoLwInv.strides); // (N_BS_ANTS, N_BS_ANTS, N_PRB)
    tensor_ref<uint32_t>                tInterCtaSyncCnt(drvdUeGrpPrms.tInfoNoiseIntfEstInterCtaSyncCnt.pAddr, drvdUeGrpPrms.tInfoNoiseIntfEstInterCtaSyncCnt.strides);// (N_UE_GRPS)
    float&                              invNoiseVarLinear = drvdUeGrpPrms.invNoiseVarLin;
#ifdef ENABLE_DEBUG
    if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z))
    {
        printf("noiseIntfEstKernel - nPrb            : %d\n", nPrb);
        printf("noiseIntfEstKernel - nRxAnt          : %d\n", nRxAnt);
        printf("noiseIntfEstKernel - nLayers         : %d\n", nLayers);
        printf("noiseIntfEstKernel - tDataRx         : addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n", static_cast<uint16_t*>(drvdUeGrpPrms.tInfoDataRx.pAddr), drvdUeGrpPrms.tInfoDataRx.strides[0], drvdUeGrpPrms.tInfoDataRx.strides[1], drvdUeGrpPrms.tInfoDataRx.strides[2], drvdUeGrpPrms.tInfoDataRx.strides[3]);
        printf("noiseIntfEstKernel - tHEst           : addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n", static_cast<uint16_t*>(drvdUeGrpPrms.tInfoHEst.pAddr), drvdUeGrpPrms.tInfoHEst.strides[0], drvdUeGrpPrms.tInfoHEst.strides[1], drvdUeGrpPrms.tInfoHEst.strides[2], drvdUeGrpPrms.tInfoHEst.strides[3]);
        printf("noiseIntfEstKernel - tNoiseIntfVar   : addr 0x%llx strides[0] %d\n", static_cast<uint16_t*>(drvdUeGrpPrms.tInfoNoiseVarPreEq.pAddr), drvdUeGrpPrms.tInfoNoiseVarPreEq.strides[0]);
        printf("noiseIntfEstKernel - tLwInv          : addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n", static_cast<uint16_t*>(drvdUeGrpPrms.tInfoLwInv.pAddr), drvdUeGrpPrms.tInfoLwInv.strides[0], drvdUeGrpPrms.tInfoLwInv.strides[1], drvdUeGrpPrms.tInfoLwInv.strides[2], drvdUeGrpPrms.tInfoLwInv.strides[3]);
        printf("noiseIntfEstKernel - tInterCtaSyncCnt: addr 0x%llx strides[0] %d\n", static_cast<uint32_t*>(drvdUeGrpPrms.tInfoRsrpInterCtaSyncCnt.pAddr), drvdUeGrpPrms.tInfoRsrpInterCtaSyncCnt.strides[0]);
    }
    if((0 == thrdIdx) && (0 == blockIdx.x))
    {
        printf("%s\n: ueGrpIdx %d nRxAnt %d nLayers %d\n", __PRETTY_FUNCTION__, ueGrpIdx, nRxAnt, nLayers);
    }
#endif

#ifdef CH_EST_IN_SHARED_MEM
    const uint8_t nTimeChEst  = drvdUeGrpPrms.dmrsAddlnPos + 1;
    const uint32_t nShHEstElems = nRxAnt*nLayers*N_SC_PER_THRD_BLK*nTimeChEst;
#else
    const uint32_t nShHEstElems = 0;
#endif // CH_EST_IN_SHARED_MEM

    const uint32_t nShRwwElems = nRxAnt*nRxAnt;
    const uint32_t nShTxDmrsElems = nLayers*N_SC_PER_THRD_BLK;
    // const uint32_t nShNoiseIntfEstElems = nRxAnt*N_SC_PER_THRD_BLK*nDmrsSyms;

    const uint32_t nIterRwwProc = div_round_up(nShRwwElems, nThrds); // Number of iterations needed to process nShRwwElems with nThrds

    // Flexible shared memory tensors
    __shared__ int32_t shHEstStrides[4];
    __shared__ int32_t shRwwStrides[3];
    __shared__ int32_t shTxDmrsStrides[2];
    __shared__ int32_t shNoiseIntfEstStrides[2];

    extern __shared__ TComplexCompute smemBlk[];
    uint32_t smemOffset = 0;
    tensor_ref<TComplexCompute> shHEst(smemBlk + smemOffset, shHEstStrides);
    smemOffset += nShHEstElems;
    
    tensor_ref<TComplexCompute> shRww(smemBlk + smemOffset, shRwwStrides);
    smemOffset += nShRwwElems;

    tensor_ref<TComplexCompute> shTxDmrs(smemBlk + smemOffset, shTxDmrsStrides);    
    smemOffset += nShTxDmrsElems;

    tensor_ref<TComplexCompute> shNoiseIntfEst(smemBlk + smemOffset, shNoiseIntfEstStrides);

    // Scrambling sequence
    __shared__ uint32_t shScrWords[N_MAX_DMRS_SYMS][N_DMRS_SCR_WORDS];
    __shared__ bool isLastCtaDone;
    
    uint32_t& interCtaSyncCnt = tInterCtaSyncCnt(ueGrpIdx);

    auto isGridEnabled = [](const uint8_t portIdx, const uint8_t dmrsGridIdx) -> bool
    {
        // Is the port enabled on the grid of interest? (used to gate DMRS signal generation on the grid, gate layer specific processing for the grid etc)
        bool enableGrid = (((portIdx & PORT_IDX_GRID_MSK) >> 1) == dmrsGridIdx) ? true : false;
        return enableGrid;
    };

    //--------------------------------------------------------------------------------------------------------    
    // Initialization
    if(0 == thrdIdx)
    {
        isLastCtaDone = false;

        // Initialize strides for channel estimation matrix copy in shared memory
        // Size shHEst: nRxAnt*nLayers*N_SC_PER_THRD_BLK*nTimeChEst
        shHEstStrides[0] = 1;
        shHEstStrides[1] = shHEstStrides[0]*nRxAnt;
        shHEstStrides[2] = shHEstStrides[1]*nLayers;
        shHEstStrides[3] = shHEstStrides[2]*N_SC_PER_THRD_BLK;

        // Initialize strides for Rww matrix in shared memory
        // Size shRww: nRxAnt*nRxAnt
        shRwwStrides[0] = 1;
        shRwwStrides[1] = shRwwStrides[0]*nRxAnt;
        shRwwStrides[2] = shRwwStrides[1]*nRxAnt;

        // Initialize strides for TxDmrs in shared memory
        // Size shTxDmrs: N_SC_PER_THRD_BLK*nLayers
        shTxDmrsStrides[0] = 1;
        shTxDmrsStrides[1] = shTxDmrsStrides[0]*N_SC_PER_THRD_BLK;
        
        // Initialize strides for noise-interferece estimate in shared memory
        // Size shNoiseIntfEst: nRxAnt*N_SC_PER_THRD_BLK*nDmrsSyms
        shNoiseIntfEstStrides[0] = 1;
        shNoiseIntfEstStrides[1] = shNoiseIntfEstStrides[0]*nRxAnt;
    }
    // thisThrdBlk.sync();

    //--------------------------------------------------------------------------------------------------------
    // 1. Generate Gold sequence. This sequence is generated once for all PRBs and all DMRS symbols processed by this thread block
    for(int32_t dmrsSymIdx = dmrsSymStart; dmrsSymIdx < dmrsSymStop; ++dmrsSymIdx)
    {
        for(int32_t dmrsScrGenItr = 0; dmrsScrGenItr < nDmrsScrBitGenIter; ++dmrsScrGenItr)
        {
            const uint32_t dmrsScrSeqWrWordIdx = dmrsScrGenItr*nDmrsScrBitsGenPerThrdBlkIter + thrdIdx;
            if(dmrsScrSeqWrWordIdx < N_DMRS_SCR_WORDS)
            {
                // Compute the scrambler sequence
                const uint32_t TWO_POW_17 = bit(17);
        
                uint32_t dmrsAbsSymIdx = pDmrsSymPos[dmrsSymIdx];
        
                // see 38.211 section 6.4.1.1.1.1
                uint32_t cInit = TWO_POW_17 * (slotNum * OFDM_SYMBOLS_PER_SLOT + dmrsAbsSymIdx + 1) * (2 * dmrsScrmId + 1) + (2 * dmrsScrmId) + scid;
                cInit &= ~bit(31);
        
                shScrWords[dmrsSymIdx][dmrsScrSeqWrWordIdx] = 
                descrambling::gold32(cInit, (dmrsScrGenAlignedStartBit + dmrsScrSeqWrWordIdx * N_DMRS_SCR_BITS_GEN_PER_THRD));
#ifdef ENABLE_DEBUG
                printf("cuPHY::Pusch::noiseIntfEstKernel - dmrsAbsSymIdx %d, slotNum %d, dmrsScrmId %d, scid %d, dmrsScrSeqWrWordIdx %d, cInit 0x%08x, dmrsScrGenAlignedStartBit %d, shScrWords[%d][%d] 0x%08x\n", 
                dmrsAbsSymIdx, slotNum, dmrsScrmId, scid, dmrsScrSeqWrWordIdx, cInit, (dmrsScrGenAlignedStartBit + dmrsScrSeqWrWordIdx*N_DMRS_SCR_BITS_GEN_PER_THRD), 
                dmrsSymIdx, dmrsScrSeqWrWordIdx, shScrWords[dmrsSymIdx][dmrsScrSeqWrWordIdx]);
#endif // ENABLE_DEBUG
            }
        }
    }

    // Ensure scrambling sequence generation is complete
    thisThrdBlk.sync();
        
    //--------------------------------------------------------------------------------------------------------
    for(int32_t prbIdx = 0; prbIdx < nPrbThisThrdBlk; prbIdx++)
    {
        // Assumes N_SC_PER_THRD_BLK*nRxAnt threads

        const uint32_t scIdx   = thrdIdx % N_SC_PER_THRD_BLK; // subcarrier index used for per iteration indexing
        const uint8_t rxAntIdx = thrdIdx / N_SC_PER_THRD_BLK;
        const uint8_t layerIdx = rxAntIdx;

        // Index of DMRS grid in which the DMRS tone being processed by this thread resides
        const uint8_t dmrsGridIdx = scIdx % N_DMRS_GRIDS_PER_PRB;

        const uint32_t startPrbThisIter = startPrbThisThrdBlk + prbIdx;
        const uint32_t startScThisIter = startPrbThisIter*CUPHY_N_TONES_PER_PRB;
        const uint32_t intraThrdBlkRelScIdx = prbIdx*CUPHY_N_TONES_PER_PRB + scIdx; // 0 based relative subcarrier index (across all thread blocks) for HEst indexing       
        const uint32_t interThrdBlkRelScIdx = startScThisIter + scIdx; // 0 based relative subcarrier index (across all thread blocks) for HEst indexing       
        const uint32_t absScIdx = startSc + interThrdBlkRelScIdx; // startPrb based absoluate subcarrier index for DataRx indexing      

        // Frequency (subcarrier) based thread activation mask
        // Handle the case where intraThrdBlkRelScIdx >= nScThisThrdBlk or equivalently prbIdx >= nPrbThisThrdBlk
        bool thrdActive = intraThrdBlkRelScIdx < nScThisThrdBlk ? true : false;


        // Zero initialize shRww
        for(uint32_t rwwInitIterIdx = 0; rwwInitIterIdx < nIterRwwProc; ++rwwInitIterIdx)
        {
            uint32_t rwwLinIdx = rwwInitIterIdx * nThrds + thrdIdx;
            if(rwwLinIdx < nShRwwElems)
            {
                shRww.pAddr[rwwLinIdx] = cuGet<TComplexCompute>(0);
            }    
        }
        

        //--------------------------------------------------------------------------------------------------------
        // 2. DMRS signal generation
        // Note: All layers use the same scrambling sequence
        
        // DMRS scrambling bits generated correspond to subcarriers across frequency
        // e.g. 2 bits for tone 0(grid 0 or 1) | 2 bits for tone 1(grid 0 or 1) | 2 bits for tone 2(grid 0 or 1) | 2 bits for tone 3(grid 0 or 1) | ...
        const uint32_t dmrsGridToneIdx = (scIdx / N_DMRS_GRIDS_PER_PRB);
        const uint32_t dmrsRelGridToneIdx = prbIdx*N_DMRS_GRID_TONES_PER_PRB + dmrsGridToneIdx;
        const uint32_t dmrsToneScrSeqBitIdx = dmrsScrStartBitOffset + (dmrsRelGridToneIdx * N_DMRS_SCR_BITS_PER_TONE);
        const uint32_t dmrsScrSeqRdBitIdx = dmrsToneScrSeqBitIdx % N_DMRS_SCR_BITS_GEN_PER_THRD;
        const uint32_t dmrsScrSeqRdWordIdx = dmrsToneScrSeqBitIdx / N_DMRS_SCR_BITS_GEN_PER_THRD;

        for(int32_t dmrsSymIdx = dmrsSymStart; dmrsSymIdx < dmrsSymStop; ++dmrsSymIdx)
        {
            uint8_t chEstIdx = dmrsSymIdx / dmrsMaxLen;
            if(thrdActive && (layerIdx < nLayers))
            {
                // Layer to port map
                uint8_t portIdx = pDmrsPortIdxs[layerIdx];
                bool enableGrid = isGridEnabled(portIdx, dmrsGridIdx);

                shTxDmrs(scIdx, layerIdx) = cuGet<TComplexCompute>(0);
                if(enableGrid)
                {
                    //--------------------------------------------------------------------------------------------------------
                    // 2a. Generate scrambling sequence (QAM4 symbols)
                    int8_t scrIBit = (shScrWords[dmrsSymIdx][dmrsScrSeqRdWordIdx] >> dmrsScrSeqRdBitIdx) & 0x1;
                    int8_t scrQBit = (shScrWords[dmrsSymIdx][dmrsScrSeqRdWordIdx] >> (dmrsScrSeqRdBitIdx + 1)) & 0x1;
            
                    TComplexCompute scrSeq =
                        cuGet<TComplexCompute>((1 - 2 * scrIBit) * RECIPROCAL_SQRT2, (1 - 2 * scrQBit) * RECIPROCAL_SQRT2);
                                
                    //--------------------------------------------------------------------------------------------------------
                    // 2b. Apply cover codes (gridShift, fOCC, tOCC) to generate a copy of the transmitted DMRS signal
                    bool enableFOCC = (portIdx & PORT_IDX_FOCC_MSK) ? true : false; // fOCC enabled for odd ports only
                    bool enableTOCC = (portIdx & PORT_IDX_TOCC_MSK) ? true : false; // tOCC enabled for ports 4-7
                    
                    int8_t fOCC = (enableFOCC && (dmrsGridToneIdx & 0x1)) ? -1 : 1; // -1 for odd grid tones and +1 for even grid tones
                    int8_t tOCC = (enableTOCC && (dmrsSymIdx % dmrsMaxLen)) ? -1 : 1; // -1 for 2nd DMRS symbol in the double DMRS (maxLen = 1) and +1 for 1st DMRS symbol in the double DMRS (maxLen = 0 or 1)
    
                    shTxDmrs(scIdx, layerIdx) = scrSeq * (dmrsScale * (fOCC * tOCC));

#ifdef ENABLE_DEBUG
                    printf("cuPHY::Pusch::noiseIntfEstKernel - shDmrs[%04d][%1d][%1d] %010.7f+j%010.7f addr %p scrSeq %010.7f+j%010.7f scrIQ (%d,%d) dmrsScrSeqRdWordIdx %d dmrsScrSeqRdBitIdx %02d dmrsGridToneIdx %04d dmrsToneScrSeqBitIdx %02d portIdx %1d fOCC %2d tOCC %2d dmrsGridIdx %1d thrdIdx %04d blockIdx (%d %d %d) threadIdx (%d %d %d)\n",
                           interThrdBlkRelScIdx,
                           layerIdx,
                           dmrsSymIdx,
                           shTxDmrs(scIdx, layerIdx).x,
                           shTxDmrs(scIdx, layerIdx).y,
                           shTxDmrs.pAddr + shTxDmrs.offset(scIdx, layerIdx),
                           scrSeq.x,
                           scrSeq.y,
                           scrIBit,
                           scrQBit,
                           dmrsScrSeqRdWordIdx,
                           dmrsScrSeqRdBitIdx,
                           dmrsGridToneIdx,
                           dmrsToneScrSeqBitIdx,                           
                           portIdx,
                           fOCC,
                           tOCC,
                           dmrsGridIdx,
                           thrdIdx,
                           blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
#endif // ENABLE_DEBUG
                }

#ifdef ENABLE_DEBUG
                printf("cuPHY::Pusch::noiseIntfEstKernel - shDmrs[%04d][%1d][%1d] %010.7f+j%010.7f  addr %p enableGrid %1d portIdx %02d dmrsGridIdx %1d thrdIdx %04d blockIdx (%d %d %d) threadIdx (%d %d %d)\n", interThrdBlkRelScIdx, layerIdx, dmrsSymIdx, shTxDmrs(scIdx, layerIdx).x, shTxDmrs(scIdx, layerIdx).y, shTxDmrs.pAddr + shTxDmrs.offset(scIdx, layerIdx), enableGrid, portIdx, dmrsGridIdx, thrdIdx, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
#endif   
            }
            thisThrdBlk.sync();

            //--------------------------------------------------------------------------------------------------------
            // 3. Generate noise-interference free DMRS signal (to within channel estimation error) at the receiver by applying estimated 
            // channel to the generated DMRS signal
            if(thrdActive && (rxAntIdx < nRxAnt))
            {
                TComplexCompute rxPilotEstNoNoiseIntf = cuGet<TComplexCompute>(0);

                // Matrix-vector multiply
                // Inner product of row "rxAntIdx" of H with column vector shTxDmrs of length "nLayers"
                for(uint8_t layerIdx = 0; layerIdx < nLayers; ++layerIdx)
                {
                    bool enableGrid = isGridEnabled(pDmrsPortIdxs[layerIdx], dmrsGridIdx);
                    if(enableGrid)
                    {
                        rxPilotEstNoNoiseIntf += (type_convert<TComplexCompute>(tHEst(rxAntIdx, layerIdx, interThrdBlkRelScIdx, chEstIdx)) * shTxDmrs(scIdx, layerIdx)); // shHEst(rxAntIdx, layerIdx, scIdx, chEstIdx, currPipeStageIdx) * shTxDmrs[layerIdx];
                    }
    
#ifdef ENABLE_DEBUG
                    TComplexCompute chEst = type_convert<TComplexCompute>(tHEst(rxAntIdx, layerIdx, interThrdBlkRelScIdx, chEstIdx));
                    printf("cuPHY::Pusch::noiseIntfEstKernel - rxPilotEstNoNoiseIntf[%03d][%04d][%02d][%1d] = %010.7f+j%010.7f chEst[%03d][%02d][%1d][%1d] = %010.7f+j%010.7f txDmrs[%03d][%1d] = %010.7f+j%010.7f thrdIdx %04d\n", ueGrpIdx, interThrdBlkRelScIdx, rxAntIdx, dmrsSymIdx, cuReal(rxPilotEstNoNoiseIntf), cuImag(rxPilotEstNoNoiseIntf), interThrdBlkRelScIdx, rxAntIdx, layerIdx, chEstIdx, cuReal(chEst), cuImag(chEst), scIdx, layerIdx, cuReal(shTxDmrs(scIdx, layerIdx)), cuImag(shTxDmrs(scIdx, layerIdx)), thrdIdx);
#endif
                }

                TComplexCompute rxPilot = type_convert<TComplexCompute>(tDataRx(absScIdx, pDmrsSymPos[dmrsSymIdx], rxAntIdx));
                shNoiseIntfEst(rxAntIdx, scIdx) = rxPilot - rxPilotEstNoNoiseIntf;

#ifdef ENABLE_DEBUG
                printf("cuPHY::Pusch::noiseIntfEstKernel - noiseIntfEst[%03d][%04d][%02d][%1d] = %010.7f+j%010.7f rxPilot = %010.7f+j%010.7f rxPilotEstNoNoiseIntf = %010.7f+j%010.7f thrdIdx %04d\n", ueGrpIdx, interThrdBlkRelScIdx, rxAntIdx, dmrsSymIdx, cuReal(shNoiseIntfEst(rxAntIdx, scIdx, dmrsSymIdx)), cuImag(shNoiseIntfEst(rxAntIdx, scIdx, dmrsSymIdx)), cuReal(rxPilot), cuImag(rxPilot), cuReal(rxPilotEstNoNoiseIntf), cuImag(rxPilotEstNoNoiseIntf), thrdIdx);
#endif // ENABLE_DEBUG
            }
            thisThrdBlk.sync();

            //--------------------------------------------------------------------------------------------------------
            // 4. Compute and accumulate noise-interference co-variance matrix (accumulation across subcarriers and time i.e. nDmrsSyms)
            if(thrdActive && (rxAntIdx < nRxAnt))
            {
                uint32_t prbIdx2 = scIdx / CUPHY_N_TONES_PER_PRB;
                
                if(noiseCovFlag)
                {
                    // Noise-interference co-variance matrix is the outer product of noise-interference vector
                    for(uint8_t rxAntIdx2 = 0; rxAntIdx2 < nRxAnt; ++rxAntIdx2)
                    {
                        TComplexCompute rww = shNoiseIntfEst(rxAntIdx, scIdx) * cuConj(shNoiseIntfEst(rxAntIdx2, scIdx));

                        atomicAdd(&shRww(rxAntIdx, rxAntIdx2, prbIdx2).x, cuReal(rww));
                        atomicAdd(&shRww(rxAntIdx, rxAntIdx2, prbIdx2).y, cuImag(rww));

#ifdef ENABLE_DEBUG
                        __threadfence();
                        printf("cuPHY::Pusch::noiseIntfEstKernel - instRww[%03d][%03d][%04d][%02d][%02d] = %010.7f+j%010.7f accumRww = %010.7f+j%010.7f\n", ueGrpIdx, startPrbThisIter + prbIdx2, interThrdBlkRelScIdx, rxAntIdx, rxAntIdx2, cuReal(rww), cuImag(rww), cuReal(shRww(rxAntIdx, rxAntIdx2, prbIdx2)), cuImag(shRww(rxAntIdx, rxAntIdx2, prbIdx2)));
#endif // ENABLE_DEBUG                    
                    }
                }
                else
                {
                    for(uint8_t rxAntIdx2 = 0; rxAntIdx2 < nRxAnt; ++rxAntIdx2)
                    {
                        uint8_t flag = ((nDmrsCdmGrpsNoData == 1) && (scIdx % 2 == 1));
                        if(flag == 0)
                        {
                            TComplexCompute rww = shNoiseIntfEst(rxAntIdx, scIdx) * cuConj(shNoiseIntfEst(rxAntIdx2, scIdx));
    
                            atomicAdd(&shRww(rxAntIdx, rxAntIdx2, prbIdx2).x, cuReal(rww));
                            atomicAdd(&shRww(rxAntIdx, rxAntIdx2, prbIdx2).y, cuImag(rww));
    
#ifdef ENABLE_DEBUG
                            __threadfence();
                            printf("cuPHY::Pusch::noiseIntfEstKernel - instRww[%03d][%03d][%04d][%02d][%02d] = %010.7f+j%010.7f accumRww = %010.7f+j%010.7f\n", ueGrpIdx, startPrbThisIter + prbIdx2, interThrdBlkRelScIdx, rxAntIdx, rxAntIdx2, cuReal(rww), cuImag(rww), cuReal(shRww(rxAntIdx, rxAntIdx2, prbIdx2)), cuImag(shRww(rxAntIdx, rxAntIdx2, prbIdx2)));
#endif // ENABLE_DEBUG                    
                        }
                    }
                }

            }
            thisThrdBlk.sync();
        }

        //--------------------------------------------------------------------------------------------------------
        // Average noise-interference co-variance matrix and feed into inverse cholesky factor compute. 
        // Also calculate sum across its diagonal in prepration for noise variance compute
        // (sum of diagonal => accumulation across nRxAnt and additionally accumulate across all PRBs in the UE group)
        for(uint32_t rwwIterIdx = 0; rwwIterIdx < nIterRwwProc; ++rwwIterIdx)
        {
            uint32_t rwwLinIdx = rwwIterIdx * nThrds + thrdIdx;
            uint8_t rxAntIdx1 = rwwLinIdx % nRxAnt;
            uint8_t rxAntIdx2 = (rwwLinIdx / nRxAnt) % nRxAnt;            
            uint32_t prbIdx3 = rwwLinIdx / (nRxAnt * nRxAnt);
            
            // Frequency (PRB) based thread activation mask
            // Ensure prbIdx3 is within bounds
            bool thrdActive2 = ((prbIdx3 < 1) && ((prbIdx + prbIdx3) < nPrbThisThrdBlk)) ? true : false;

            if(thrdActive2)
            {
                TComplexCompute rww = shRww(rxAntIdx1, rxAntIdx2, prbIdx3) / (CUPHY_N_TONES_PER_PRB * nDmrsSyms);
                if(nDmrsCdmGrpsNoData==1)
                {
                    rww = rww * cuGet<TComplexCompute>(2.0f, 0.0f);
                }
                
                // Sum diagonal
                if(rxAntIdx1 == rxAntIdx2)
                {
                    rww.x += CUPHY_NOISE_REGULARIZER;
                    TCompute noisePwr = fabsf(cuReal(rww));
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
                    atomicAdd(const_cast<TCompute*>(tNoiseIntfVar.pAddr + tNoiseIntfVar.offset(pAbsUeIdxs[0])), noisePwr);
#else
                    atomicAdd(const_cast<TCompute*>(tNoiseIntfVar.pAddr + tNoiseIntfVar.offset(ueGrpIdx)), noisePwr);
#endif
                }

                shRww(rxAntIdx1, rxAntIdx2, prbIdx3) = rww;
                // tLwInv(rxAntIdx1, rxAntIdx2, startPrbThisIter + prbIdx3) = type_convert<TComplexStorageOut>(rww);
                
#ifdef ENABLE_DEBUG
                printf("cuPHY::Pusch::noiseIntfEstKernel - Rww[%03d][%03d][%02d][%02d] = %010.7f+j%010.7f accumRww[%03d][%02d][%02d] = %010.7f+j%010.7f\n", ueGrpIdx, startPrbThisIter + prbIdx3, rxAntIdx1, rxAntIdx2, cuReal(rww), cuImag(rww), startPrbThisIter + prbIdx3, rxAntIdx1, rxAntIdx2, cuReal(shRww(rxAntIdx1, rxAntIdx2, prbIdx3)), cuImag(shRww(rxAntIdx1, rxAntIdx2, prbIdx3)));
#endif // ENABLE_DEBUG
            }
            // printf("thrdIdx %d rwwIterIdx %d prbIdx %d prbIdx3 %d thrdActive %d thrdActive2 %d interThrdBlkRelScIdx %d nScThisThrdBlk %d intraThrdBlkRelScIdx %d\n", thrdIdx, rwwIterIdx, prbIdx, prbIdx3, thrdActive, thrdActive2, interThrdBlkRelScIdx , nScThisThrdBlk, intraThrdBlkRelScIdx);

        }
        thisThrdBlk.sync();
        
#ifdef ENABLE_DEBUG        
        if(0 == thrdIdx)
        {
            printf("cuPHY::Pusch::noiseIntfEstKernel: prbGrpIdx %d ueGrpIdx %d prbIdx %d startPrbThisIter %d nPrbThisThrdBlk %d\n", prbGrpIdx, ueGrpIdx, prbIdx, startPrbThisIter, nPrbThisThrdBlk);
        }
#endif // ENABLE_DEBUG

        if(noiseCovFlag)
        {
            cholFactorInv<TCompute, 
                          TStorageOut>(startPrbThisIter, nPrbThisThrdBlk, nRxAnt, shRww, tLwInv);

        }
    }

    // Logic to ensure all thread blocks in the grid complete (ending in writes to tNoiseIntfVar above)
    if(0 == thrdIdx)
    {
        // Ensure interCtaSyncCnt is incremented only after the tNoiseIntfVar global memory writes have been completed.
        // Note that while the global memory access above is atomic, it does not imply ordering constraints for memory operations,
        // hence a threadfence is still needed.
        __threadfence();

        uint32_t syncCnt = atomicInc(const_cast<uint32_t*>(&interCtaSyncCnt), nUeGrpThrdBlksNeeded);
        
        // Is this the last CTA to be processed for this user group?        
        isLastCtaDone = (syncCnt == (nUeGrpThrdBlksNeeded - 1)) ? true : false;

#ifdef ENABLE_DEBUG
        printf("cuPHY::Pusch::noiseIntfEstKernel: prbGrpIdx %d ueGrpIdx %d isLastCtaDone %u nUeGrpThrdBlksNeeded %d syncCnt %d interCtaSyncCnt %d\n", prbGrpIdx, ueGrpIdx, isLastCtaDone, nUeGrpThrdBlksNeeded, syncCnt, interCtaSyncCnt);
#endif
    }

    thisThrdBlk.sync();
 
    // Use one thread block per UE group to average noise-interference variance across nRxAnt, nPrb and convert from linear to dB
    if(isLastCtaDone)
    {
        if(0 == thrdIdx)
        {
            // Average across across Rx antenna and PRBs allocated to this UE group
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
            TCompute avgNoiseIntfVar     = type_convert<TCompute>(tNoiseIntfVar(pAbsUeIdxs[0])) / (nRxAnt * nPrb);
#else
            TCompute avgNoiseIntfVar     = type_convert<TCompute>(tNoiseIntfVar(ueGrpIdx)) / (nRxAnt * nPrb);
#endif
            invNoiseVarLinear            = float(1.0f / avgNoiseIntfVar);
            drvdUeGrpPrms.noiseVarForDtx = float(avgNoiseIntfVar);
            if(DMRS_SYMBOL_IDX==CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT)
            {
                TCompute noiseIntfVarDb      = 10*log10f(avgNoiseIntfVar) + NOISE_EST_CORRECTION_DB;
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
                // noiseVarPreEq per UE
                for(uint32_t i = 0; i < nUes; ++i)
                {      
                    tNoiseIntfVar(pAbsUeIdxs[i]) = type_convert<TStorageOut>(noiseIntfVarDb);  
                } 
#else
                // noiseVarPreEq per UEGRP
                tNoiseIntfVar(ueGrpIdx) = type_convert<TStorageOut>(noiseIntfVarDb);   
#endif
            }
            else if(DMRS_SYMBOL_IDX==CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0)
            {
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
                // reset for full-slot processing
                for(uint32_t i = 0; i < nUes; ++i)
                {      
                    tNoiseIntfVar(pAbsUeIdxs[i]) = type_convert<TStorageOut>(0);  
                } 
#else
                // noiseVarPreEq per UEGRP
                tNoiseIntfVar(ueGrpIdx) = type_convert<TStorageOut>(0);   
#endif
                interCtaSyncCnt = 0; //reset for full-slot processing  
            }

#ifdef ENABLE_DEBUG
            // printf("cuPHY::Pusch::noiseIntfEstKernel - nSymb %d nRxAnt %d nUeGrps %d nThrds %d nIter %d ueGrpIdx %d\n", nSymb, nRxAnt, nUeGrps, nThrds, nIter, ueGrpIdx2);
            printf("cuPHY::Pusch::noiseIntfEstKernel - noiseIntfVar: ueGrp[%d] %010.7f db (Lin: %010.7f)\n", ueGrpIdx, noiseIntfVarDb, avgNoiseIntfVar);
#endif
        }
    }
  } //noiseIntfEstNoDftSOfdmKernel

  // blockDim: (N_SC_PER_THRD_BLK*nRxAnt)
  // gridDim:  (nThrdBlksPerUeGrp, nUeGrps)
  // Note: tInfoPreEqNoiseVar and tInfoNoiseIntfEstInterCtaSyncCnt need to be reset to 0
  template <typename TStorageIn,
            typename TDataRx,
            typename TStorageOut,
            typename TCompute,
            uint32_t N_DMRS_GRIDS_PER_PRB,     // number of DMRS grids per PRB (2 or 3)
            uint32_t N_PRB_PER_THRD_BLK>       // # of PRBs processed in a thread block
  __global__ void noiseIntfEstNoDftSOfdmSubSlotKernel(puschRxNoiseIntfEstDynDescr_t* pDynDescr)
  {
    KERNEL_PRINT_GRID_ONCE("%s\n grid = (%u %u %u), block = (%u %u %u)\n",
                           __PRETTY_FUNCTION__,
                           gridDim.x, gridDim.y, gridDim.z,
                           blockDim.x, blockDim.y, blockDim.z);

    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TDataRx>::type     TComplexDataRx;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;

    static_assert((2 == N_DMRS_GRIDS_PER_PRB), "DMRS grids per PRB other than 2 not supported");

    // For DMRS config type 1, the LSB 3 bits of portId can be used to determine fOCC, grid, tOCC as follows:
    // bit 0 fOCC, bit 1 grid, bit 2 tOCC
    static constexpr uint32_t PORT_IDX_FOCC_MSK = 0x1;
    static constexpr uint32_t PORT_IDX_GRID_MSK = 0x2;
    static constexpr uint32_t PORT_IDX_TOCC_MSK = 0x4;

    //--------------------------------------------------------------------------------------------------------
    const uint32_t prbGrpIdx = blockIdx.x;
    const uint32_t ueGrpIdx  = blockIdx.y;

    cg::thread_block const& thisThrdBlk = cg::this_thread_block();
    const uint32_t thrdIdx = thisThrdBlk.thread_rank();
    const uint32_t nThrds = thisThrdBlk.size();

    puschRxNoiseIntfEstDynDescr_t& dynDescr = *(pDynDescr);
    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[ueGrpIdx];

    // Number of thread blocks needed to process this user group
    const uint32_t nPrb = drvdUeGrpPrms.nPrb;
    uint32_t nUeGrpThrdBlksNeeded = div_round_up(nPrb, N_PRB_PER_THRD_BLK);

    // The number of thread blocks are sized to process UE group with largest PRB allocation
    // Early exit thread blocks which exceed those needed to process PRBs for current UE group
    if(blockIdx.x >= nUeGrpThrdBlksNeeded) return;

    // Pointer to DMRS symbol used for channel estimation (single-symbol if maxLen == 1, double-symbol if maxLen == 2)
    uint8_t const* pDmrsSymPos = drvdUeGrpPrms.dmrsSymLoc;
    const uint16_t slotNum  = drvdUeGrpPrms.slotNum;
    const uint16_t startPrb = drvdUeGrpPrms.startPrb;
    const uint16_t startSc = startPrb * CUPHY_N_TONES_PER_PRB;
    // const uint16_t nSc = nPrb * CUPHY_N_TONES_PER_PRB;

    // Number of PRBs already processed before this thread block
    const uint32_t nPrbProcessed       = prbGrpIdx * N_PRB_PER_THRD_BLK;
    const uint32_t startPrbThisThrdBlk = nPrbProcessed;
    const uint32_t absStartPrbThisThrdBlk = startPrb + startPrbThisThrdBlk;

    // Number of PRBs remaining to be processed from this thread block onwards
    const int32_t nPrbRemaining = nPrb - nPrbProcessed;
    // Calculate loop count - PRBs to be processed by this thread block
    const int32_t nPrbThisThrdBlk = (nPrbRemaining <= N_PRB_PER_THRD_BLK) ? nPrbRemaining : N_PRB_PER_THRD_BLK;  // std::min(nPrbRemaining, N_PRB_PER_THRD_BLK)
    const int32_t nScThisThrdBlk  = nPrbThisThrdBlk * CUPHY_N_TONES_PER_PRB;

    const uint8_t nRxAnt  = drvdUeGrpPrms.nRxAnt;
    const uint8_t nLayers = drvdUeGrpPrms.nLayers;

    const uint8_t dmrsMaxLen  = drvdUeGrpPrms.dmrsMaxLen;
    const uint8_t nDmrsSyms   = drvdUeGrpPrms.nDmrsSyms; // nDmrsAddlnPos * dmrsMaxLen;
    const uint16_t dmrsScrmId = drvdUeGrpPrms.dmrsScrmId;
    const uint8_t  nDmrsCdmGrpsNoData = drvdUeGrpPrms.nDmrsCdmGrpsNoData;
    const uint8_t  subSlotStageIdx    = pDynDescr->subSlotStageIndex;    // index of current dmrs
    const bool isFirstDmrs = subSlotStageIdx == 0;
    const bool isLastDmrs = subSlotStageIdx == drvdUeGrpPrms.dmrsAddlnPos;

    uint8_t const* pDmrsPortIdxs = drvdUeGrpPrms.dmrsPortIdxs;  // Layer to port map
    const uint8_t   scid         = drvdUeGrpPrms.scid;
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
    uint16_t* pAbsUeIdxs   = &drvdUeGrpPrms.ueIdxs[0];
    uint32_t  nUes         = drvdUeGrpPrms.nUes; // Number of UEs in this UE group
#endif

    // Flag if noise covariance computed
    bool noiseCovFlag = (drvdUeGrpPrms.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC) ? true : false;


#ifdef ENABLE_DEBUG
    if(0 == thrdIdx)
    {
        printf("cuPHY::Pusch::noiseIntfEstKernel: ueGrpIdx %d nUeGrpThrdBlksNeeded %d prbGrpIdx %d nPrbRemaining %d nPrbThisThrdBlk %d nDmrsSyms %d dmrsMaxLen %d dmrsAddlnPos %d\n", ueGrpIdx, nUeGrpThrdBlksNeeded, prbGrpIdx, nPrbRemaining, nPrbThisThrdBlk, nDmrsSyms, dmrsMaxLen, drvdUeGrpPrms.dmrsAddlnPos);
    }
#endif // ENABLE_DEBUG

    //--------------------------------------------------------------------------------------------------------
    // Number of DMRS tones to be processed by this thread block:

    // Number of tones per DMRS grid in a PRB
    static constexpr uint32_t N_DMRS_GRID_TONES_PER_PRB = CUPHY_N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB;

    // Number of tones per DMRS grid processed by this thread block
    static constexpr uint32_t N_DMRS_GRID_TONES = N_DMRS_GRID_TONES_PER_PRB * N_PRB_PER_THRD_BLK;

    // Total number of DMRS tones processed by this thread block
    static constexpr uint32_t N_DMRS_TONES = N_DMRS_GRID_TONES * N_DMRS_GRIDS_PER_PRB; // equal to CUPHY_N_TONES_PER_PRB * N_PRB_PER_THRD_BLK

    // Number of tones processed by the thread block per iteration
    static constexpr uint32_t N_SC_PER_THRD_BLK = CUPHY_N_TONES_PER_PRB;

    //--------------------------------------------------------------------------------------------------------
    // DMRS scrambling

    static constexpr uint32_t N_DMRS_SCR_BITS_PER_TONE = 2; // 1bit for I and 1 bit for Q
    static constexpr uint32_t N_DMRS_SCR_BITS = N_DMRS_SCR_BITS_PER_TONE * N_DMRS_GRID_TONES;

    // Number of DMRS scrambler bits generated by one call to gold32 by one thread
    static constexpr uint32_t N_DMRS_SCR_BITS_GEN_PER_THRD = 32;

    // Round up to the next multiple of N_DMRS_SCR_BITS_GEN_PER_THRD plus 1 (+1 because DMRS_SCR_PRB_CLUSTER_START_BIT_OFFSET
    // may be large enough to spill the scrambler bits to the next word)
    static constexpr uint32_t N_DMRS_SCR_WORDS = div_round_up<uint32_t>(N_DMRS_SCR_BITS, N_DMRS_SCR_BITS_GEN_PER_THRD) + 1;

    // Section 5.2.1 in 3GPP TS 38.211
    // The fast-forward of 1600 prescribed by spec is already baked into the gold sequence generator
    static constexpr uint32_t DMRS_SCR_FF = 0; // 1600;

    // Absolute index of scrambling sequence start
    // Note:The DMRS scrambling sequence is the same for all the DMRS grids. There are 2 sequences one for
    // scid 0 and other for scid 1 but the same sequences is reused for all DMRS grids
    const uint32_t dmrsScrSeqStartIdx = absStartPrbThisThrdBlk * N_DMRS_GRID_TONES_PER_PRB;

    // First scrambler bit index needed by this thread block
    const uint32_t dmrsScrStartBit = DMRS_SCR_FF + (dmrsScrSeqStartIdx * N_DMRS_SCR_BITS_PER_TONE);

    // The scrambling sequence generator outputs 32 scrambler bits at a time. Thus, compute the earliest
    // multiple of 32 bits which contains the scrambler bit of the first tone in the PRB cluster as the
    // start index
    const uint32_t dmrsScrGenAlignedStartBit = (dmrsScrStartBit / N_DMRS_SCR_BITS_GEN_PER_THRD) * N_DMRS_SCR_BITS_GEN_PER_THRD;

    // Offset to scrambler bit of the first tone in the PRB cluster
    const uint32_t dmrsScrStartBitOffset = dmrsScrStartBit - dmrsScrGenAlignedStartBit;

    // Since each thread generates N_DMRS_SCR_BITS_GEN_PER_THRD bits and each DMRS tone needs N_DMRS_SCR_BITS_PER_TONE bits to be scrambled,
    // compute the number of iterations needed to generated all the scrambler bits
    static constexpr uint32_t N_DMRS_SCR_BITS_NEEDED = N_DMRS_TONES*N_DMRS_SCR_BITS_PER_TONE;
    const uint32_t nDmrsScrBitsGenPerThrdBlkIter = nThrds*N_DMRS_SCR_BITS_GEN_PER_THRD;
    const uint32_t nDmrsScrBitGenIter = div_round_up(N_DMRS_SCR_BITS_NEEDED, nDmrsScrBitsGenPerThrdBlkIter);

    static constexpr TCompute RECIPROCAL_SQRT2 = 0.7071068f;

    // add 0.5dB to noise estimate to account for noise filtered out by the channel interpolation filter
    static constexpr TCompute NOISE_EST_CORRECTION_DB = 0.5f;

    // Note: if nDmrsCdmGrpsNoData = 2 then DMRS power = 2 (DMRS amplitude = sqrt(2))
    //       if nDmrsCdmGrpsNoData = 1 then DMRS power = 1
    const TCompute dmrsScale = (2 == nDmrsCdmGrpsNoData) ? 1.4142135f : 1.0f;

    //--------------------------------------------------------------------------------------------------------
    tensor_ref<const TComplexDataRx>    tDataRx              (drvdUeGrpPrms.tInfoDataRx.pAddr       , drvdUeGrpPrms.tInfoDataRx.strides);// (NF, ND, N_BS_ANTS)
    tensor_ref<const TComplexStorageIn> tHEst                (drvdUeGrpPrms.tInfoHEst.pAddr         , drvdUeGrpPrms.tInfoHEst.strides);  // (N_BS_ANTS, N_LAYERS, NF, NH)
    tensor_ref<volatile TStorageOut>    tNoiseIntfVar        (drvdUeGrpPrms.tInfoNoiseVarPreEq.pAddr, drvdUeGrpPrms.tInfoNoiseVarPreEq.strides); // (N_UE_GRP)
    tensor_ref<volatile TStorageOut>    tNoiseIntfVarPerPrb  (drvdUeGrpPrms.tInfoNoiseVarPreEq.pAddr, drvdUeGrpPrms.tInfoNoiseVarPreEq.strides); // (nPrb)
    tensor_ref<TComplexStorageOut>      tLwInv               (drvdUeGrpPrms.tInfoLwInv.pAddr        , drvdUeGrpPrms.tInfoLwInv.strides); // (N_BS_ANTS, N_BS_ANTS, N_PRB)
    tensor_ref<uint32_t>                tInterCtaSyncCnt     (drvdUeGrpPrms.tInfoNoiseIntfEstInterCtaSyncCnt.pAddr, drvdUeGrpPrms.tInfoNoiseIntfEstInterCtaSyncCnt.strides);// (N_UE_GRPS)
    float&                              invNoiseVarLinear = drvdUeGrpPrms.invNoiseVarLin;
#ifdef ENABLE_DEBUG
    if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z))
    {
        printf("noiseIntfEstKernel - nPrb            : %d\n", nPrb);
        printf("noiseIntfEstKernel - nRxAnt          : %d\n", nRxAnt);
        printf("noiseIntfEstKernel - nLayers         : %d\n", nLayers);
        printf("noiseIntfEstKernel - tDataRx         : addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n", static_cast<uint16_t*>(drvdUeGrpPrms.tInfoDataRx.pAddr), drvdUeGrpPrms.tInfoDataRx.strides[0], drvdUeGrpPrms.tInfoDataRx.strides[1], drvdUeGrpPrms.tInfoDataRx.strides[2], drvdUeGrpPrms.tInfoDataRx.strides[3]);
        printf("noiseIntfEstKernel - tHEst           : addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n", static_cast<uint16_t*>(drvdUeGrpPrms.tInfoHEst.pAddr), drvdUeGrpPrms.tInfoHEst.strides[0], drvdUeGrpPrms.tInfoHEst.strides[1], drvdUeGrpPrms.tInfoHEst.strides[2], drvdUeGrpPrms.tInfoHEst.strides[3]);
        printf("noiseIntfEstKernel - tNoiseIntfVar   : addr 0x%llx strides[0] %d\n", static_cast<uint16_t*>(drvdUeGrpPrms.tInfoNoiseVarPreEq.pAddr), drvdUeGrpPrms.tInfoNoiseVarPreEq.strides[0]);
        printf("noiseIntfEstKernel - tLwInv          : addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n", static_cast<uint16_t*>(drvdUeGrpPrms.tInfoLwInv.pAddr), drvdUeGrpPrms.tInfoLwInv.strides[0], drvdUeGrpPrms.tInfoLwInv.strides[1], drvdUeGrpPrms.tInfoLwInv.strides[2], drvdUeGrpPrms.tInfoLwInv.strides[3]);
        printf("noiseIntfEstKernel - tInterCtaSyncCnt: addr 0x%llx strides[0] %d\n", static_cast<uint32_t*>(drvdUeGrpPrms.tInfoRsrpInterCtaSyncCnt.pAddr), drvdUeGrpPrms.tInfoRsrpInterCtaSyncCnt.strides[0]);
    }
    if((0 == thrdIdx) && (0 == blockIdx.x))
    {
        printf("%s\n: ueGrpIdx %d nRxAnt %d nLayers %d\n", __PRETTY_FUNCTION__, ueGrpIdx, nRxAnt, nLayers);
    }
#endif

#ifdef CH_EST_IN_SHARED_MEM
    const uint8_t nTimeChEst  = drvdUeGrpPrms.dmrsAddlnPos + 1;
    const uint32_t nShHEstElems = nRxAnt*nLayers*N_SC_PER_THRD_BLK*nTimeChEst;
#else
    const uint32_t nShHEstElems = 0;
#endif // CH_EST_IN_SHARED_MEM

    const uint32_t nShRwwElems = nRxAnt*nRxAnt;
    const uint32_t nShTxDmrsElems = nLayers*N_SC_PER_THRD_BLK;
    const uint32_t nShNoiseIntfEstElems = nRxAnt*N_SC_PER_THRD_BLK*nDmrsSyms;
    //const uint32_t nShNoiseIntfVar = nMaxPrb;

    const uint32_t nIterRwwProc = div_round_up(nShRwwElems, nThrds); // Number of iterations needed to process nShRwwElems with nThrds

    // Flexible shared memory tensors
    __shared__ int32_t shHEstStrides[4];
    __shared__ int32_t shRwwStrides[3];
    __shared__ int32_t shTxDmrsStrides[2];
    __shared__ int32_t shNoiseIntfEstStrides[2];

    extern __shared__ TComplexCompute smemBlk[];
    uint32_t smemOffset = 0;
    tensor_ref<TComplexCompute> shHEst(smemBlk + smemOffset, shHEstStrides);
    smemOffset += nShHEstElems;

    tensor_ref<TComplexCompute> shRww(smemBlk + smemOffset, shRwwStrides);
    smemOffset += nShRwwElems;

    tensor_ref<TComplexCompute> shTxDmrs(smemBlk + smemOffset, shTxDmrsStrides);
    smemOffset += nShTxDmrsElems;

    tensor_ref<TComplexCompute> shNoiseIntfEst(smemBlk + smemOffset, shNoiseIntfEstStrides);
    smemOffset += nShNoiseIntfEstElems;

    TCompute* shNoiseIntfVar = &smemBlk[smemOffset];

    // Scrambling sequence
    __shared__ uint32_t shScrWords[N_MAX_DMRS_SYMS][N_DMRS_SCR_WORDS];
    __shared__ bool isLastCtaDone;

    uint32_t& interCtaSyncCnt = tInterCtaSyncCnt(ueGrpIdx);

    auto isGridEnabled = [](const uint8_t portIdx, const uint8_t dmrsGridIdx) -> bool
    {
        // Is the port enabled on the grid of interest? (used to gate DMRS signal generation on the grid, gate layer specific processing for the grid etc)
        bool enableGrid = (((portIdx & PORT_IDX_GRID_MSK) >> 1) == dmrsGridIdx) ? true : false;
        return enableGrid;
    };

    //--------------------------------------------------------------------------------------------------------
    // Initialization
    if(0 == thrdIdx)
    {
        isLastCtaDone = false;

        // Initialize strides for channel estimation matrix copy in shared memory
        // Size shHEst: nRxAnt*nLayers*N_SC_PER_THRD_BLK*nTimeChEst
        shHEstStrides[0] = 1;
        shHEstStrides[1] = shHEstStrides[0]*nRxAnt;
        shHEstStrides[2] = shHEstStrides[1]*nLayers;
        shHEstStrides[3] = shHEstStrides[2]*N_SC_PER_THRD_BLK;

        // Initialize strides for Rww matrix in shared memory
        // Size shRww: nRxAnt*nRxAnt
        shRwwStrides[0] = 1;
        shRwwStrides[1] = shRwwStrides[0]*nRxAnt;
        shRwwStrides[2] = shRwwStrides[1]*nRxAnt;

        // Initialize strides for TxDmrs in shared memory
        // Size shTxDmrs: N_SC_PER_THRD_BLK*nLayers
        shTxDmrsStrides[0] = 1;
        shTxDmrsStrides[1] = shTxDmrsStrides[0]*N_SC_PER_THRD_BLK;

        // Initialize strides for noise-interferece estimate in shared memory
        // Size shNoiseIntfEst: nRxAnt*N_SC_PER_THRD_BLK*nDmrsSyms
        shNoiseIntfEstStrides[0] = 1;
        shNoiseIntfEstStrides[1] = shNoiseIntfEstStrides[0]*nRxAnt;

    }

    for (int i = thrdIdx; i < nPrb; i += nThrds)
    {
        shNoiseIntfVar[i] = 0;
    }
    // thisThrdBlk.sync();

    //--------------------------------------------------------------------------------------------------------
    // 1. Generate Gold sequence. This sequence is generated once for all PRBs and all DMRS symbols processed by this thread block
    for(int32_t dmrsSymIdx = subSlotStageIdx * dmrsMaxLen; dmrsSymIdx < (subSlotStageIdx + 1) * dmrsMaxLen; ++dmrsSymIdx)
    {
        for(int32_t dmrsScrGenItr = 0; dmrsScrGenItr < nDmrsScrBitGenIter; ++dmrsScrGenItr)
        {
            const uint32_t dmrsScrSeqWrWordIdx = dmrsScrGenItr*nDmrsScrBitsGenPerThrdBlkIter + thrdIdx;
            if(dmrsScrSeqWrWordIdx < N_DMRS_SCR_WORDS)
            {
                // Compute the scrambler sequence
                const uint32_t TWO_POW_17 = bit(17);

                uint32_t dmrsAbsSymIdx = pDmrsSymPos[dmrsSymIdx];

                // see 38.211 section 6.4.1.1.1.1
                uint32_t cInit = TWO_POW_17 * (slotNum * OFDM_SYMBOLS_PER_SLOT + dmrsAbsSymIdx + 1) * (2 * dmrsScrmId + 1) + (2 * dmrsScrmId) + scid;
                cInit &= ~bit(31);

                shScrWords[dmrsSymIdx][dmrsScrSeqWrWordIdx] =
                    descrambling::gold32(cInit, (dmrsScrGenAlignedStartBit + dmrsScrSeqWrWordIdx * N_DMRS_SCR_BITS_GEN_PER_THRD));
#ifdef ENABLE_DEBUG
                printf("cuPHY::Pusch::noiseIntfEstKernel - dmrsAbsSymIdx %d, slotNum %d, dmrsScrmId %d, scid %d, dmrsScrSeqWrWordIdx %d, cInit 0x%08x, dmrsScrGenAlignedStartBit %d, shScrWords[%d][%d] 0x%08x\n",
                dmrsAbsSymIdx, slotNum, dmrsScrmId, scid, dmrsScrSeqWrWordIdx, cInit, (dmrsScrGenAlignedStartBit + dmrsScrSeqWrWordIdx*N_DMRS_SCR_BITS_GEN_PER_THRD),
                dmrsSymIdx, dmrsScrSeqWrWordIdx, shScrWords[dmrsSymIdx][dmrsScrSeqWrWordIdx]);
#endif // ENABLE_DEBUG
            }
        }
    }

    // Ensure scrambling sequence generation is complete
    thisThrdBlk.sync();

    //--------------------------------------------------------------------------------------------------------
    for(int32_t prbIdx = 0; prbIdx < nPrbThisThrdBlk; prbIdx++)
    {
        // Assumes N_SC_PER_THRD_BLK*nRxAnt threads

        const uint32_t scIdx   = thrdIdx % N_SC_PER_THRD_BLK; // subcarrier index used for per iteration indexing
        const uint8_t rxAntIdx = thrdIdx / N_SC_PER_THRD_BLK;
        const uint8_t layerIdx = rxAntIdx;

        // Index of DMRS grid in which the DMRS tone being processed by this thread resides
        const uint8_t dmrsGridIdx = scIdx % N_DMRS_GRIDS_PER_PRB;

        const uint32_t startPrbThisIter = startPrbThisThrdBlk + prbIdx;
        const uint32_t startScThisIter = startPrbThisIter*CUPHY_N_TONES_PER_PRB;
        const uint32_t intraThrdBlkRelScIdx = prbIdx*CUPHY_N_TONES_PER_PRB + scIdx; // 0 based relative subcarrier index (across all thread blocks) for HEst indexing
        const uint32_t interThrdBlkRelScIdx = startScThisIter + scIdx; // 0 based relative subcarrier index (across all thread blocks) for HEst indexing
        const uint32_t absScIdx = startSc + interThrdBlkRelScIdx; // startPrb based absoluate subcarrier index for DataRx indexing

        // Frequency (subcarrier) based thread activation mask
        // Handle the case where intraThrdBlkRelScIdx >= nScThisThrdBlk or equivalently prbIdx >= nPrbThisThrdBlk
        bool thrdActive = intraThrdBlkRelScIdx < nScThisThrdBlk ? true : false;


        // Zero initialize shRww
        for(uint32_t rwwInitIterIdx = 0; rwwInitIterIdx < nIterRwwProc; ++rwwInitIterIdx)
        {
            uint32_t rwwLinIdx = rwwInitIterIdx * nThrds + thrdIdx;
            if(rwwLinIdx < nShRwwElems)
            {
                shRww.pAddr[rwwLinIdx] = cuGet<TComplexCompute>(0);
            }
        }

        //--------------------------------------------------------------------------------------------------------
        // 2. DMRS signal generation
        // Note: All layers use the same scrambling sequence

        // DMRS scrambling bits generated correspond to subcarriers across frequency
        // e.g. 2 bits for tone 0(grid 0 or 1) | 2 bits for tone 1(grid 0 or 1) | 2 bits for tone 2(grid 0 or 1) | 2 bits for tone 3(grid 0 or 1) | ...
        const uint32_t dmrsGridToneIdx = (scIdx / N_DMRS_GRIDS_PER_PRB);
        const uint32_t dmrsRelGridToneIdx = prbIdx*N_DMRS_GRID_TONES_PER_PRB + dmrsGridToneIdx;
        const uint32_t dmrsToneScrSeqBitIdx = dmrsScrStartBitOffset + (dmrsRelGridToneIdx * N_DMRS_SCR_BITS_PER_TONE);
        const uint32_t dmrsScrSeqRdBitIdx = dmrsToneScrSeqBitIdx % N_DMRS_SCR_BITS_GEN_PER_THRD;
        const uint32_t dmrsScrSeqRdWordIdx = dmrsToneScrSeqBitIdx / N_DMRS_SCR_BITS_GEN_PER_THRD;

        for(int32_t dmrsSymIdx = subSlotStageIdx * dmrsMaxLen; dmrsSymIdx < (subSlotStageIdx + 1) * dmrsMaxLen; ++dmrsSymIdx)
        {
            uint8_t chEstIdx = dmrsSymIdx / dmrsMaxLen;
            if(thrdActive && (layerIdx < nLayers))
            {
                // Layer to port map
                uint8_t portIdx = pDmrsPortIdxs[layerIdx];
                bool enableGrid = isGridEnabled(portIdx, dmrsGridIdx);

                shTxDmrs(scIdx, layerIdx) = cuGet<TComplexCompute>(0);
                if(enableGrid)
                {
                    //--------------------------------------------------------------------------------------------------------
                    // 2a. Generate scrambling sequence (QAM4 symbols)
                    int8_t scrIBit = (shScrWords[dmrsSymIdx][dmrsScrSeqRdWordIdx] >> dmrsScrSeqRdBitIdx) & 0x1;
                    int8_t scrQBit = (shScrWords[dmrsSymIdx][dmrsScrSeqRdWordIdx] >> (dmrsScrSeqRdBitIdx + 1)) & 0x1;

                    TComplexCompute scrSeq =
                        cuGet<TComplexCompute>((1 - 2 * scrIBit) * RECIPROCAL_SQRT2, (1 - 2 * scrQBit) * RECIPROCAL_SQRT2);

                    //--------------------------------------------------------------------------------------------------------
                    // 2b. Apply cover codes (gridShift, fOCC, tOCC) to generate a copy of the transmitted DMRS signal
                    bool enableFOCC = (portIdx & PORT_IDX_FOCC_MSK) ? true : false; // fOCC enabled for odd ports only
                    bool enableTOCC = (portIdx & PORT_IDX_TOCC_MSK) ? true : false; // tOCC enabled for ports 4-7

                    int8_t fOCC = (enableFOCC && (dmrsGridToneIdx & 0x1)) ? -1 : 1; // -1 for odd grid tones and +1 for even grid tones
                    int8_t tOCC = (enableTOCC && (dmrsSymIdx % dmrsMaxLen)) ? -1 : 1; // -1 for 2nd DMRS symbol in the double DMRS (maxLen = 1) and +1 for 1st DMRS symbol in the double DMRS (maxLen = 0 or 1)

                    shTxDmrs(scIdx, layerIdx) = scrSeq * (dmrsScale * (fOCC * tOCC));

#ifdef ENABLE_DEBUG
                    printf("cuPHY::Pusch::noiseIntfEstKernel - shDmrs[%04d][%1d][%1d] %010.7f+j%010.7f addr %p scrSeq %010.7f+j%010.7f scrIQ (%d,%d) dmrsScrSeqRdWordIdx %d dmrsScrSeqRdBitIdx %02d dmrsGridToneIdx %04d dmrsToneScrSeqBitIdx %02d portIdx %1d fOCC %2d tOCC %2d dmrsGridIdx %1d thrdIdx %04d blockIdx (%d %d %d) threadIdx (%d %d %d)\n",
                           interThrdBlkRelScIdx,
                           layerIdx,
                           dmrsSymIdx,
                           shTxDmrs(scIdx, layerIdx).x,
                           shTxDmrs(scIdx, layerIdx).y,
                           shTxDmrs.pAddr + shTxDmrs.offset(scIdx, layerIdx),
                           scrSeq.x,
                           scrSeq.y,
                           scrIBit,
                           scrQBit,
                           dmrsScrSeqRdWordIdx,
                           dmrsScrSeqRdBitIdx,
                           dmrsGridToneIdx,
                           dmrsToneScrSeqBitIdx,
                           portIdx,
                           fOCC,
                           tOCC,
                           dmrsGridIdx,
                           thrdIdx,
                           blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
#endif // ENABLE_DEBUG
                }

#ifdef ENABLE_DEBUG
                printf("cuPHY::Pusch::noiseIntfEstKernel - shDmrs[%04d][%1d][%1d] %010.7f+j%010.7f  addr %p enableGrid %1d portIdx %02d dmrsGridIdx %1d thrdIdx %04d blockIdx (%d %d %d) threadIdx (%d %d %d)\n", interThrdBlkRelScIdx, layerIdx, dmrsSymIdx, shTxDmrs(scIdx, layerIdx).x, shTxDmrs(scIdx, layerIdx).y, shTxDmrs.pAddr + shTxDmrs.offset(scIdx, layerIdx), enableGrid, portIdx, dmrsGridIdx, thrdIdx, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
#endif
            }
            thisThrdBlk.sync();

            //--------------------------------------------------------------------------------------------------------
            // 3. Generate noise-interference free DMRS signal (to within channel estimation error) at the receiver by applying estimated
            // channel to the generated DMRS signal
            if(thrdActive && (rxAntIdx < nRxAnt))
            {
                TComplexCompute rxPilotEstNoNoiseIntf = cuGet<TComplexCompute>(0);

                // Matrix-vector multiply
                // Inner product of row "rxAntIdx" of H with column vector shTxDmrs of length "nLayers"
                for(uint8_t layerIdx = 0; layerIdx < nLayers; ++layerIdx)
                {
                    bool enableGrid = isGridEnabled(pDmrsPortIdxs[layerIdx], dmrsGridIdx);
                    if(enableGrid)
                    {
                        rxPilotEstNoNoiseIntf += (type_convert<TComplexCompute>(tHEst(rxAntIdx, layerIdx, interThrdBlkRelScIdx, chEstIdx)) * shTxDmrs(scIdx, layerIdx)); // shHEst(rxAntIdx, layerIdx, scIdx, chEstIdx, currPipeStageIdx) * shTxDmrs[layerIdx];
                    }

#ifdef ENABLE_DEBUG
                    TComplexCompute chEst = type_convert<TComplexCompute>(tHEst(rxAntIdx, layerIdx, interThrdBlkRelScIdx, chEstIdx));
                    printf("cuPHY::Pusch::noiseIntfEstKernel - rxPilotEstNoNoiseIntf[%03d][%04d][%02d][%1d] = %010.7f+j%010.7f chEst[%03d][%02d][%1d][%1d] = %010.7f+j%010.7f txDmrs[%03d][%1d] = %010.7f+j%010.7f thrdIdx %04d\n", ueGrpIdx, interThrdBlkRelScIdx, rxAntIdx, dmrsSymIdx, cuReal(rxPilotEstNoNoiseIntf), cuImag(rxPilotEstNoNoiseIntf), interThrdBlkRelScIdx, rxAntIdx, layerIdx, chEstIdx, cuReal(chEst), cuImag(chEst), scIdx, layerIdx, cuReal(shTxDmrs(scIdx, layerIdx)), cuImag(shTxDmrs(scIdx, layerIdx)), thrdIdx);
#endif
                }

                TComplexCompute rxPilot = type_convert<TComplexCompute>(tDataRx(absScIdx, pDmrsSymPos[dmrsSymIdx], rxAntIdx));
                shNoiseIntfEst(rxAntIdx, scIdx) = rxPilot - rxPilotEstNoNoiseIntf;

#ifdef ENABLE_DEBUG
                printf("cuPHY::Pusch::noiseIntfEstKernel - noiseIntfEst[%03d][%04d][%02d][%1d] = %010.7f+j%010.7f rxPilot = %010.7f+j%010.7f rxPilotEstNoNoiseIntf = %010.7f+j%010.7f thrdIdx %04d\n", ueGrpIdx, interThrdBlkRelScIdx, rxAntIdx, dmrsSymIdx, cuReal(shNoiseIntfEst(rxAntIdx, scIdx, dmrsSymIdx)), cuImag(shNoiseIntfEst(rxAntIdx, scIdx, dmrsSymIdx)), cuReal(rxPilot), cuImag(rxPilot), cuReal(rxPilotEstNoNoiseIntf), cuImag(rxPilotEstNoNoiseIntf), thrdIdx);
#endif // ENABLE_DEBUG
            }
            thisThrdBlk.sync();

            //--------------------------------------------------------------------------------------------------------
            // 4. Compute and accumulate noise-interference co-variance matrix (accumulation across subcarriers and time i.e. nDmrsSyms)
            if(thrdActive && (rxAntIdx < nRxAnt))
            {
                uint32_t prbIdx2 = scIdx / CUPHY_N_TONES_PER_PRB;

                if(noiseCovFlag)
                {
                    // Noise-interference co-variance matrix is the outer product of noise-interference vector
                    for(uint8_t rxAntIdx2 = 0; rxAntIdx2 < nRxAnt; ++rxAntIdx2)
                    {
                        TComplexCompute rww = shNoiseIntfEst(rxAntIdx, scIdx) * cuConj(shNoiseIntfEst(rxAntIdx2, scIdx));

                        atomicAdd(&shRww(rxAntIdx, rxAntIdx2, prbIdx2).x, cuReal(rww));
                        atomicAdd(&shRww(rxAntIdx, rxAntIdx2, prbIdx2).y, cuImag(rww));

#ifdef ENABLE_DEBUG
                        __threadfence();
                        printf("cuPHY::Pusch::noiseIntfEstKernel - instRww[%03d][%03d][%04d][%02d][%02d] = %010.7f+j%010.7f accumRww = %010.7f+j%010.7f\n", ueGrpIdx, startPrbThisIter + prbIdx2, interThrdBlkRelScIdx, rxAntIdx, rxAntIdx2, cuReal(rww), cuImag(rww), cuReal(shRww(rxAntIdx, rxAntIdx2, prbIdx2)), cuImag(shRww(rxAntIdx, rxAntIdx2, prbIdx2)));
#endif // ENABLE_DEBUG
                    }
                }
                else
                {
                    for(uint8_t rxAntIdx2 = 0; rxAntIdx2 < nRxAnt; ++rxAntIdx2)
                    {
                        uint8_t flag = ((nDmrsCdmGrpsNoData == 1) && (scIdx % 2 == 1));
                        if(flag == 0)
                        {
                            TComplexCompute rww = shNoiseIntfEst(rxAntIdx, scIdx) * cuConj(shNoiseIntfEst(rxAntIdx2, scIdx));

                            atomicAdd(&shRww(rxAntIdx, rxAntIdx2, prbIdx2).x, cuReal(rww));
                            atomicAdd(&shRww(rxAntIdx, rxAntIdx2, prbIdx2).y, cuImag(rww));

#ifdef ENABLE_DEBUG
                            __threadfence();
                            printf("cuPHY::Pusch::noiseIntfEstKernel - instRww[%03d][%03d][%04d][%02d][%02d] = %010.7f+j%010.7f accumRww = %010.7f+j%010.7f\n", ueGrpIdx, startPrbThisIter + prbIdx2, interThrdBlkRelScIdx, rxAntIdx, rxAntIdx2, cuReal(rww), cuImag(rww), cuReal(shRww(rxAntIdx, rxAntIdx2, prbIdx2)), cuImag(shRww(rxAntIdx, rxAntIdx2, prbIdx2)));
#endif // ENABLE_DEBUG
                        }
                    }
                }

            }
            thisThrdBlk.sync();
        }

        //--------------------------------------------------------------------------------------------------------
        // Average noise-interference co-variance matrix and feed into inverse cholesky factor compute.
        // Also calculate sum across its diagonal in prepration for noise variance compute
        // (sum of diagonal => accumulation across nRxAnt and additionally accumulate across all PRBs in the UE group)
        for(uint32_t rwwIterIdx = 0; rwwIterIdx < nIterRwwProc; ++rwwIterIdx)
        {
            uint32_t rwwLinIdx = rwwIterIdx * nThrds + thrdIdx;
            uint8_t rxAntIdx1 = rwwLinIdx % nRxAnt;
            uint8_t rxAntIdx2 = (rwwLinIdx / nRxAnt) % nRxAnt;
            uint32_t prbIdx3 = rwwLinIdx / (nRxAnt * nRxAnt);

            // Frequency (PRB) based thread activation mask
            // Ensure prbIdx3 is within bounds
            bool thrdActive2 = ((prbIdx3 < 1) && ((prbIdx + prbIdx3) < nPrbThisThrdBlk)) ? true : false;

            if(thrdActive2)
            {
                TComplexCompute rww = shRww(rxAntIdx1, rxAntIdx2, prbIdx3) / (CUPHY_N_TONES_PER_PRB * dmrsMaxLen);
                if(nDmrsCdmGrpsNoData==1)
                {
                    rww = rww * cuGet<TComplexCompute>(2.0f, 0.0f);
                }

                // Sum diagonal
                if(rxAntIdx1 == rxAntIdx2)
                {
                    rww.x += CUPHY_NOISE_REGULARIZER;
                    TCompute noisePwr = fabsf(cuReal(rww));
                    atomicAdd(const_cast<TCompute*>(&shNoiseIntfVar[prbIdx], noisePwr));
                }

                shRww(rxAntIdx1, rxAntIdx2, prbIdx3) = rww;
                // tLwInv(rxAntIdx1, rxAntIdx2, startPrbThisIter + prbIdx3) = type_convert<TComplexStorageOut>(rww);

#ifdef ENABLE_DEBUG
                printf("cuPHY::Pusch::noiseIntfEstKernel - Rww[%03d][%03d][%02d][%02d] = %010.7f+j%010.7f accumRww[%03d][%02d][%02d] = %010.7f+j%010.7f\n", ueGrpIdx, startPrbThisIter + prbIdx3, rxAntIdx1, rxAntIdx2, cuReal(rww), cuImag(rww), startPrbThisIter + prbIdx3, rxAntIdx1, rxAntIdx2, cuReal(shRww(rxAntIdx1, rxAntIdx2, prbIdx3)), cuImag(shRww(rxAntIdx1, rxAntIdx2, prbIdx3)));
#endif // ENABLE_DEBUG
            }
            // printf("thrdIdx %d rwwIterIdx %d prbIdx %d prbIdx3 %d thrdActive %d thrdActive2 %d interThrdBlkRelScIdx %d nScThisThrdBlk %d intraThrdBlkRelScIdx %d\n", thrdIdx, rwwIterIdx, prbIdx, prbIdx3, thrdActive, thrdActive2, interThrdBlkRelScIdx , nScThisThrdBlk, intraThrdBlkRelScIdx);

        }
        thisThrdBlk.sync();

#ifdef ENABLE_DEBUG
        if(0 == thrdIdx)
        {
            printf("cuPHY::Pusch::noiseIntfEstKernel: prbGrpIdx %d ueGrpIdx %d prbIdx %d startPrbThisIter %d nPrbThisThrdBlk %d\n", prbGrpIdx, ueGrpIdx, prbIdx, startPrbThisIter, nPrbThisThrdBlk);
        }
#endif // ENABLE_DEBUG

        if(noiseCovFlag)
        {
            cholFactorInv<TCompute,
                          TStorageOut>(startPrbThisIter, nPrbThisThrdBlk, nRxAnt, shRww, tLwInv);

        }

        // update per PRB noise variance in global memory
        if(0 == thrdIdx)
        {        if(isFirstDmrs)
            {
                tNoiseIntfVarPerPrb(absStartPrbThisThrdBlk + prbIdx) = shNoiseIntfVar[prbIdx] / nRxAnt;
            }
            else
            {
                tNoiseIntfVarPerPrb(absStartPrbThisThrdBlk + prbIdx) = (tNoiseIntfVarPerPrb(absStartPrbThisThrdBlk + prbIdx) * subSlotStageIdx + shNoiseIntfVar[prbIdx] / nRxAnt) / (subSlotStageIdx + 1);
            }
        }

    }

    // Logic to ensure all thread blocks in the grid complete (ending in writes to tNoiseIntfVar above)
    if(0 == thrdIdx)
    {
        // Ensure interCtaSyncCnt is incremented only after the tNoiseIntfVarPerPrb global memory writes have been completed.
        // hence a threadfence is still needed.
        __threadfence();

        uint32_t syncCnt = atomicInc(const_cast<uint32_t*>(&interCtaSyncCnt), nUeGrpThrdBlksNeeded);

        // Is this the last CTA to be processed for this user group?
        isLastCtaDone = (syncCnt == (nUeGrpThrdBlksNeeded - 1)) ? true : false;

#ifdef ENABLE_DEBUG
        printf("cuPHY::Pusch::noiseIntfEstKernel: prbGrpIdx %d ueGrpIdx %d isLastCtaDone %u nUeGrpThrdBlksNeeded %d syncCnt %d interCtaSyncCnt %d\n", prbGrpIdx, ueGrpIdx, isLastCtaDone, nUeGrpThrdBlksNeeded, syncCnt, interCtaSyncCnt);
#endif
    }

    thisThrdBlk.sync();

    // Use one thread block per UE group to average noise-interference variance across nRxAnt, nPrb and convert from linear to dB
    if(isLastCtaDone)
    {
        if(0 == thrdIdx)
        {
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
            TCompute* noiseIntfVar = const_cast<TCompute*>(tNoiseIntfVar.pAddr + tNoiseIntfVar.offset(pAbsUeIdxs[0]));
#else
            TCompute* noiseIntfVar = const_cast<TCompute*>(tNoiseIntfVar.pAddr + tNoiseIntfVar.offset(ueGrpIdx));
#endif
            //FixMe!! verify if this is correct, if so then replace with parallel reduction
            TCompute sum = 0;
            // reduce sum tNoiseIntfVarPerPrb
            for (int i = 0; i < MAX_N_PRBS_SUPPORTED; i++)
            {
                sum += tNoiseIntfVarPerPrb(i); // tNoiseIntfVarPerPrb(i) contains moving average of per PRB
            }
            noiseIntfVar[0] = sum;

            // average over and PRBs allocated to this UE group and perform dB conversion (only for the last DMRS)
            if (isLastDmrs)
            {
                TCompute avgNoiseIntfVar     = type_convert<TCompute>(noiseIntfVar[0]) / nPrb;
                invNoiseVarLinear            = float(1.0f / avgNoiseIntfVar);
                drvdUeGrpPrms.noiseVarForDtx = float(avgNoiseIntfVar);
                TCompute noiseIntfVarDb      = 10 * log10f(avgNoiseIntfVar) + NOISE_EST_CORRECTION_DB;
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
            // noiseVarPreEq per UE
            for(uint32_t i = 0; i < nUes; ++i)
            {
                tNoiseIntfVar(pAbsUeIdxs[i]) = type_convert<TStorageOut>(noiseIntfVarDb);
            }
#else
                // noiseVarPreEq per UEGRP
                tNoiseIntfVar(ueGrpIdx) = type_convert<TStorageOut>(noiseIntfVarDb);
#endif
            }

#ifdef ENABLE_DEBUG
            // printf("cuPHY::Pusch::noiseIntfEstKernel - nSymb %d nRxAnt %d nUeGrps %d nThrds %d nIter %d ueGrpIdx %d\n", nSymb, nRxAnt, nUeGrps, nThrds, nIter, ueGrpIdx2);
            printf("cuPHY::Pusch::noiseIntfEstKernel - noiseIntfVar: ueGrp[%d] %010.7f db (Lin: %010.7f)\n", ueGrpIdx, noiseIntfVarDb, avgNoiseIntfVar);
#endif
        }
    }
  } //noiseIntfEstNoDftSOfdmSubSlotKernel

// blockDim: (N_SC_PER_THRD_BLK*nRxAnt)
// gridDim:  (nThrdBlksPerUeGrp, nUeGrps)
// Note: tInfoPreEqNoiseVar and tInfoNoiseIntfEstInterCtaSyncCnt need to be reset to 0
 template <typename TStorageIn,
           typename TDataRx,
           typename TStorageOut,
           typename TCompute,
           uint32_t N_DMRS_GRIDS_PER_PRB,          // number of DMRS grids (combs) per PRB (2 or 3)
           uint32_t N_PRB_PER_THRD_BLK,            // # of PRBs processed in a thread block
           uint8_t  DMRS_SYMBOL_IDX>               // the index of DMRS symbol to be processed
 __global__ void noiseIntfEstKernel(puschRxNoiseIntfEstDynDescr_t* pDynDescr)
  {
    KERNEL_PRINT_GRID_ONCE("%s\n grid = (%u %u %u), block = (%u %u %u)\n",
                           __PRETTY_FUNCTION__,
                           gridDim.x, gridDim.y, gridDim.z,
                           blockDim.x, blockDim.y, blockDim.z);

    typedef typename complex_from_scalar<TCompute>::type    TComplexCompute;
    typedef typename complex_from_scalar<TDataRx>::type     TComplexDataRx;
    typedef typename complex_from_scalar<TStorageIn>::type  TComplexStorageIn;
    typedef typename complex_from_scalar<TStorageOut>::type TComplexStorageOut;    

    static_assert((2 == N_DMRS_GRIDS_PER_PRB), "DMRS grids per PRB other than 2 not supported");
    
    static_assert(((CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0 == DMRS_SYMBOL_IDX) || (CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT == DMRS_SYMBOL_IDX)), "Each kernel processes the first DMRS symbol for early-HARQ or all DMRS symbols for full-slot");

    // For DMRS config type 1, the LSB 3 bits of portId can be used to determine fOCC, grid, tOCC as follows:
    // bit 0 fOCC, bit 1 grid, bit 2 tOCC
    static constexpr uint32_t PORT_IDX_FOCC_MSK = 0x1;
    static constexpr uint32_t PORT_IDX_GRID_MSK = 0x2;
    static constexpr uint32_t PORT_IDX_TOCC_MSK = 0x4;

    //--------------------------------------------------------------------------------------------------------
    const uint32_t prbGrpIdx = blockIdx.x;
    const uint32_t ueGrpIdx  = blockIdx.y;

    cg::thread_block const& thisThrdBlk = cg::this_thread_block();
    const uint32_t thrdIdx = thisThrdBlk.thread_rank();
    const uint32_t nThrds = thisThrdBlk.size();

    puschRxNoiseIntfEstDynDescr_t& dynDescr = *(pDynDescr);
    cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[ueGrpIdx];

    // Number of thread blocks needed to process this user group
    const uint32_t nPrb = drvdUeGrpPrms.nPrb;    
    uint32_t nUeGrpThrdBlksNeeded = div_round_up(nPrb, N_PRB_PER_THRD_BLK);

    // The number of thread blocks are sized to process UE group with largest PRB allocation
    // Early exit thread blocks which exceed those needed to process PRBs for current UE group
    if(blockIdx.x >= nUeGrpThrdBlksNeeded) return;

    // Pointer to DMRS symbol used for channel estimation (single-symbol if maxLen == 1, double-symbol if maxLen == 2)
    uint8_t const* pDmrsSymPos = drvdUeGrpPrms.dmrsSymLoc;
    const uint16_t slotNum  = drvdUeGrpPrms.slotNum;
    const uint16_t startPrb = drvdUeGrpPrms.startPrb;
    const uint16_t startSc = startPrb * CUPHY_N_TONES_PER_PRB;
    // const uint16_t nSc = nPrb * CUPHY_N_TONES_PER_PRB;

    // Number of PRBs already processed before this thread block
    const uint32_t nPrbProcessed       = prbGrpIdx * N_PRB_PER_THRD_BLK;
    const uint32_t startPrbThisThrdBlk = nPrbProcessed;
    const uint32_t absStartPrbThisThrdBlk = startPrb + startPrbThisThrdBlk;

    // Number of PRBs remaining to be processed from this thread block onwards
    const int32_t nPrbRemaining = nPrb - nPrbProcessed;
    // Calculate loop count - PRBs to be processed by this thread block
    const int32_t nPrbThisThrdBlk = (nPrbRemaining <= N_PRB_PER_THRD_BLK) ? nPrbRemaining : N_PRB_PER_THRD_BLK;  // std::min(nPrbRemaining, N_PRB_PER_THRD_BLK)
    const int32_t nScThisThrdBlk  = nPrbThisThrdBlk * CUPHY_N_TONES_PER_PRB;

    const uint8_t nRxAnt  = drvdUeGrpPrms.nRxAnt;
    const uint8_t nLayers = drvdUeGrpPrms.nLayers;

    const uint8_t dmrsMaxLen  = drvdUeGrpPrms.dmrsMaxLen;
    uint8_t nDmrsSyms   = drvdUeGrpPrms.nDmrsSyms; // nDmrsAddlnPos * dmrsMaxLen;
    uint8_t dmrsSymStart = 0;
    uint8_t dmrsSymStop  = nDmrsSyms;
    if(DMRS_SYMBOL_IDX==CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0)
    {
        dmrsSymStop    = dmrsMaxLen;
    }
    const uint16_t dmrsScrmId = drvdUeGrpPrms.dmrsScrmId;
    const uint8_t  nDmrsCdmGrpsNoData = drvdUeGrpPrms.nDmrsCdmGrpsNoData;
    
    uint8_t const* pDmrsPortIdxs = drvdUeGrpPrms.dmrsPortIdxs;  // Layer to port map
    const uint8_t   scid         = drvdUeGrpPrms.scid;
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
    uint16_t* pAbsUeIdxs         = &drvdUeGrpPrms.ueIdxs[0];
    uint32_t  nUes               = drvdUeGrpPrms.nUes; // Number of UEs in this UE group
#endif
    
    const uint8_t   enableTfPrcd = drvdUeGrpPrms.enableTfPrcd;

    // Flag if noise covariance computed
    bool noiseCovFlag = (drvdUeGrpPrms.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC) ? true : false;


#ifdef ENABLE_DEBUG
    if(0 == thrdIdx)
    {
        printf("cuPHY::Pusch::noiseIntfEstKernel: ueGrpIdx %d nUeGrpThrdBlksNeeded %d prbGrpIdx %d nPrbRemaining %d nPrbThisThrdBlk %d nDmrsSyms %d dmrsMaxLen %d dmrsAddlnPos %d\n", ueGrpIdx, nUeGrpThrdBlksNeeded, prbGrpIdx, nPrbRemaining, nPrbThisThrdBlk, nDmrsSyms, dmrsMaxLen, drvdUeGrpPrms.dmrsAddlnPos);
    }
#endif // ENABLE_DEBUG  

    //--------------------------------------------------------------------------------------------------------
    // Number of DMRS tones to be processed by this thread block:

    // Number of tones per DMRS grid in a PRB
    static constexpr uint32_t N_DMRS_GRID_TONES_PER_PRB = CUPHY_N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB;

    // Number of tones per DMRS grid processed by this thread block
    static constexpr uint32_t N_DMRS_GRID_TONES = N_DMRS_GRID_TONES_PER_PRB * N_PRB_PER_THRD_BLK;

    // Total number of DMRS tones processed by this thread block
    static constexpr uint32_t N_DMRS_TONES = N_DMRS_GRID_TONES * N_DMRS_GRIDS_PER_PRB; // equal to CUPHY_N_TONES_PER_PRB * N_PRB_PER_THRD_BLK

    // Number of tones processed by the thread block per iteration
    static constexpr uint32_t N_SC_PER_THRD_BLK = CUPHY_N_TONES_PER_PRB;

    //--------------------------------------------------------------------------------------------------------
    // DMRS scrambling

    static constexpr uint32_t N_DMRS_SCR_BITS_PER_TONE = 2; // 1bit for I and 1 bit for Q
    static constexpr uint32_t N_DMRS_SCR_BITS = N_DMRS_SCR_BITS_PER_TONE * N_DMRS_GRID_TONES;
    
    // Number of DMRS scrambler bits generated by one call to gold32 by one thread
    static constexpr uint32_t N_DMRS_SCR_BITS_GEN_PER_THRD = 32;

    // Round up to the next multiple of N_DMRS_SCR_BITS_GEN_PER_THRD plus 1 (+1 because DMRS_SCR_PRB_CLUSTER_START_BIT_OFFSET
    // may be large enough to spill the scrambler bits to the next word)    
    static constexpr uint32_t N_DMRS_SCR_WORDS = div_round_up<uint32_t>(N_DMRS_SCR_BITS, N_DMRS_SCR_BITS_GEN_PER_THRD) + 1;

    // Section 5.2.1 in 3GPP TS 38.211
    // The fast-forward of 1600 prescribed by spec is already baked into the gold sequence generator
    static constexpr uint32_t DMRS_SCR_FF = 0; // 1600;
 
    // Absolute index of scrambling sequence start
    // Note:The DMRS scrambling sequence is the same for all the DMRS grids. There are 2 sequences one for
    // scid 0 and other for scid 1 but the same sequences is reused for all DMRS grids    
    const uint32_t dmrsScrSeqStartIdx = absStartPrbThisThrdBlk * N_DMRS_GRID_TONES_PER_PRB;

    // First scrambler bit index needed by this thread block
    const uint32_t dmrsScrStartBit = DMRS_SCR_FF + (dmrsScrSeqStartIdx * N_DMRS_SCR_BITS_PER_TONE);
 
    // The scrambling sequence generator outputs 32 scrambler bits at a time. Thus, compute the earliest
    // multiple of 32 bits which contains the scrambler bit of the first tone in the PRB cluster as the
    // start index
    const uint32_t dmrsScrGenAlignedStartBit = (dmrsScrStartBit / N_DMRS_SCR_BITS_GEN_PER_THRD) * N_DMRS_SCR_BITS_GEN_PER_THRD;
    
    // Offset to scrambler bit of the first tone in the PRB cluster
    const uint32_t dmrsScrStartBitOffset = dmrsScrStartBit - dmrsScrGenAlignedStartBit;

    // Since each thread generates N_DMRS_SCR_BITS_GEN_PER_THRD bits and each DMRS tone needs N_DMRS_SCR_BITS_PER_TONE bits to be scrambled,
    // compute the number of iterations needed to generated all the scrambler bits
    static constexpr uint32_t N_DMRS_SCR_BITS_NEEDED = N_DMRS_TONES*N_DMRS_SCR_BITS_PER_TONE;
    const uint32_t nDmrsScrBitsGenPerThrdBlkIter = nThrds*N_DMRS_SCR_BITS_GEN_PER_THRD;
    const uint32_t nDmrsScrBitGenIter = div_round_up(N_DMRS_SCR_BITS_NEEDED, nDmrsScrBitsGenPerThrdBlkIter);

    static constexpr TCompute RECIPROCAL_SQRT2 = 0.7071068f;

    // add 0.5dB to noise estimate to account for noise filtered out by the channel interpolation filter
    static constexpr TCompute NOISE_EST_CORRECTION_DB = 0.5f;

    // Note: if nDmrsCdmGrpsNoData = 2 then DMRS power = 2 (DMRS amplitude = sqrt(2)) 
    //       if nDmrsCdmGrpsNoData = 1 then DMRS power = 1
    const TCompute dmrsScale = (2 == nDmrsCdmGrpsNoData) ? 1.4142135f : 1.0f;
    
    //--------------------------------------------------------------------------------------------------------    
    tensor_ref<const TComplexDataRx>    tDataRx        (drvdUeGrpPrms.tInfoDataRx.pAddr       , drvdUeGrpPrms.tInfoDataRx.strides);// (NF, ND, N_BS_ANTS)
    tensor_ref<const TComplexStorageIn> tHEst          (drvdUeGrpPrms.tInfoHEst.pAddr         , drvdUeGrpPrms.tInfoHEst.strides);  // (N_BS_ANTS, N_LAYERS, NF, NH)
    tensor_ref<volatile TStorageOut>    tNoiseIntfVar  (drvdUeGrpPrms.tInfoNoiseVarPreEq.pAddr, drvdUeGrpPrms.tInfoNoiseVarPreEq.strides); // (N_UE_GRP)
    tensor_ref<TComplexStorageOut>      tLwInv         (drvdUeGrpPrms.tInfoLwInv.pAddr        , drvdUeGrpPrms.tInfoLwInv.strides); // (N_BS_ANTS, N_BS_ANTS, N_PRB)
    tensor_ref<uint32_t>                tInterCtaSyncCnt(drvdUeGrpPrms.tInfoNoiseIntfEstInterCtaSyncCnt.pAddr, drvdUeGrpPrms.tInfoNoiseIntfEstInterCtaSyncCnt.strides);// (N_UE_GRPS)
    float&                              invNoiseVarLinear = drvdUeGrpPrms.invNoiseVarLin;
#ifdef ENABLE_DEBUG
    if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z))
    {
        printf("noiseIntfEstKernel - nPrb            : %d\n", nPrb);
        printf("noiseIntfEstKernel - nRxAnt          : %d\n", nRxAnt);
        printf("noiseIntfEstKernel - nLayers         : %d\n", nLayers);
        printf("noiseIntfEstKernel - tDataRx         : addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n", static_cast<uint16_t*>(drvdUeGrpPrms.tInfoDataRx.pAddr), drvdUeGrpPrms.tInfoDataRx.strides[0], drvdUeGrpPrms.tInfoDataRx.strides[1], drvdUeGrpPrms.tInfoDataRx.strides[2], drvdUeGrpPrms.tInfoDataRx.strides[3]);
        printf("noiseIntfEstKernel - tHEst           : addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n", static_cast<uint16_t*>(drvdUeGrpPrms.tInfoHEst.pAddr), drvdUeGrpPrms.tInfoHEst.strides[0], drvdUeGrpPrms.tInfoHEst.strides[1], drvdUeGrpPrms.tInfoHEst.strides[2], drvdUeGrpPrms.tInfoHEst.strides[3]);
        printf("noiseIntfEstKernel - tNoiseIntfVar   : addr 0x%llx strides[0] %d\n", static_cast<uint16_t*>(drvdUeGrpPrms.tInfoNoiseVarPreEq.pAddr), drvdUeGrpPrms.tInfoNoiseVarPreEq.strides[0]);
        printf("noiseIntfEstKernel - tLwInv          : addr 0x%llx strides[0] %d, strides[1] %d, strides[2] %d, strides[3] %d\n", static_cast<uint16_t*>(drvdUeGrpPrms.tInfoLwInv.pAddr), drvdUeGrpPrms.tInfoLwInv.strides[0], drvdUeGrpPrms.tInfoLwInv.strides[1], drvdUeGrpPrms.tInfoLwInv.strides[2], drvdUeGrpPrms.tInfoLwInv.strides[3]);
        printf("noiseIntfEstKernel - tInterCtaSyncCnt: addr 0x%llx strides[0] %d\n", static_cast<uint32_t*>(drvdUeGrpPrms.tInfoRsrpInterCtaSyncCnt.pAddr), drvdUeGrpPrms.tInfoRsrpInterCtaSyncCnt.strides[0]);
    }
    if((0 == thrdIdx) && (0 == blockIdx.x))
    {
        printf("%s\n: ueGrpIdx %d nRxAnt %d nLayers %d\n", __PRETTY_FUNCTION__, ueGrpIdx, nRxAnt, nLayers);
    }
#endif

#ifdef CH_EST_IN_SHARED_MEM
    const uint8_t nTimeChEst  = drvdUeGrpPrms.dmrsAddlnPos + 1;
    const uint32_t nShHEstElems = nRxAnt*nLayers*N_SC_PER_THRD_BLK*nTimeChEst;
#else
    const uint32_t nShHEstElems = 0;
#endif // CH_EST_IN_SHARED_MEM

    const uint32_t nShRwwElems = nRxAnt*nRxAnt;
    const uint32_t nShTxDmrsElems = nLayers*N_SC_PER_THRD_BLK;
    // const uint32_t nShNoiseIntfEstElems = nRxAnt*N_SC_PER_THRD_BLK*nDmrsSyms;

    const uint32_t nIterRwwProc = div_round_up(nShRwwElems, nThrds); // Number of iterations needed to process nShRwwElems with nThrds

    // Flexible shared memory tensors
    __shared__ int32_t shHEstStrides[4];
    __shared__ int32_t shRwwStrides[3];
    __shared__ int32_t shTxDmrsStrides[2];
    __shared__ int32_t shNoiseIntfEstStrides[2];

    extern __shared__ TComplexCompute smemBlk[];
    uint32_t smemOffset = 0;
    tensor_ref<TComplexCompute> shHEst(smemBlk + smemOffset, shHEstStrides);
    smemOffset += nShHEstElems;
    
    tensor_ref<TComplexCompute> shRww(smemBlk + smemOffset, shRwwStrides);
    smemOffset += nShRwwElems;

    tensor_ref<TComplexCompute> shTxDmrs(smemBlk + smemOffset, shTxDmrsStrides);    
    smemOffset += nShTxDmrsElems;

    tensor_ref<TComplexCompute> shNoiseIntfEst(smemBlk + smemOffset, shNoiseIntfEstStrides);

    // Scrambling sequence
    __shared__ uint32_t shScrWords[N_MAX_DMRS_SYMS][N_DMRS_SCR_WORDS];
    __shared__ bool isLastCtaDone;
    
    uint32_t& interCtaSyncCnt = tInterCtaSyncCnt(ueGrpIdx);

    auto isGridEnabled = [](const uint8_t portIdx, const uint8_t dmrsGridIdx) -> bool
    {
        // Is the port enabled on the grid of interest? (used to gate DMRS signal generation on the grid, gate layer specific processing for the grid etc)
        bool enableGrid = (((portIdx & PORT_IDX_GRID_MSK) >> 1) == dmrsGridIdx) ? true : false;
        return enableGrid;
    };

    //--------------------------------------------------------------------------------------------------------    
    // Initialization
    if(0 == thrdIdx)
    {
        isLastCtaDone = false;

        // Initialize strides for channel estimation matrix copy in shared memory
        // Size shHEst: nRxAnt*nLayers*N_SC_PER_THRD_BLK*nTimeChEst
        shHEstStrides[0] = 1;
        shHEstStrides[1] = shHEstStrides[0]*nRxAnt;
        shHEstStrides[2] = shHEstStrides[1]*nLayers;
        shHEstStrides[3] = shHEstStrides[2]*N_SC_PER_THRD_BLK;

        // Initialize strides for Rww matrix in shared memory
        // Size shRww: nRxAnt*nRxAnt
        shRwwStrides[0] = 1;
        shRwwStrides[1] = shRwwStrides[0]*nRxAnt;
        shRwwStrides[2] = shRwwStrides[1]*nRxAnt;

        // Initialize strides for TxDmrs in shared memory
        // Size shTxDmrs: N_SC_PER_THRD_BLK*nLayers
        shTxDmrsStrides[0] = 1;
        shTxDmrsStrides[1] = shTxDmrsStrides[0]*N_SC_PER_THRD_BLK;
        
        // Initialize strides for noise-interferece estimate in shared memory
        // Size shNoiseIntfEst: nRxAnt*N_SC_PER_THRD_BLK*nDmrsSyms
        shNoiseIntfEstStrides[0] = 1;
        shNoiseIntfEstStrides[1] = shNoiseIntfEstStrides[0]*nRxAnt;
    }
    // thisThrdBlk.sync();

    //--------------------------------------------------------------------------------------------------------
    // 1. Generate Gold sequence. This sequence is generated once for all PRBs and all DMRS symbols processed by this thread block
    if(enableTfPrcd==0)
    {
    for(int32_t dmrsSymIdx = dmrsSymStart; dmrsSymIdx < dmrsSymStop; ++dmrsSymIdx)
    {
        for(int32_t dmrsScrGenItr = 0; dmrsScrGenItr < nDmrsScrBitGenIter; ++dmrsScrGenItr)
        {
            const uint32_t dmrsScrSeqWrWordIdx = dmrsScrGenItr*nDmrsScrBitsGenPerThrdBlkIter + thrdIdx;
            if(dmrsScrSeqWrWordIdx < N_DMRS_SCR_WORDS)
            {
                // Compute the scrambler sequence
                const uint32_t TWO_POW_17 = bit(17);
        
                uint32_t dmrsAbsSymIdx = pDmrsSymPos[dmrsSymIdx];
        
                // see 38.211 section 6.4.1.1.1.1
                uint32_t cInit = TWO_POW_17 * (slotNum * OFDM_SYMBOLS_PER_SLOT + dmrsAbsSymIdx + 1) * (2 * dmrsScrmId + 1) + (2 * dmrsScrmId) + scid;
                cInit &= ~bit(31);
        
                shScrWords[dmrsSymIdx][dmrsScrSeqWrWordIdx] = 
                descrambling::gold32(cInit, (dmrsScrGenAlignedStartBit + dmrsScrSeqWrWordIdx * N_DMRS_SCR_BITS_GEN_PER_THRD));
#ifdef ENABLE_DEBUG
                printf("cuPHY::Pusch::noiseIntfEstKernel - dmrsAbsSymIdx %d, slotNum %d, dmrsScrmId %d, scid %d, dmrsScrSeqWrWordIdx %d, cInit 0x%08x, dmrsScrGenAlignedStartBit %d, shScrWords[%d][%d] 0x%08x\n", 
                dmrsAbsSymIdx, slotNum, dmrsScrmId, scid, dmrsScrSeqWrWordIdx, cInit, (dmrsScrGenAlignedStartBit + dmrsScrSeqWrWordIdx*N_DMRS_SCR_BITS_GEN_PER_THRD), 
                dmrsSymIdx, dmrsScrSeqWrWordIdx, shScrWords[dmrsSymIdx][dmrsScrSeqWrWordIdx]);
#endif // ENABLE_DEBUG
            }
        }
    }

    // Ensure scrambling sequence generation is complete
    thisThrdBlk.sync();
    } //if(enableTfPrcd==0)
        
    //--------------------------------------------------------------------------------------------------------
    for(int32_t prbIdx = 0; prbIdx < nPrbThisThrdBlk; prbIdx++)
    {
        // Assumes N_SC_PER_THRD_BLK*nRxAnt threads

        const uint32_t scIdx   = thrdIdx % N_SC_PER_THRD_BLK; // subcarrier index used for per iteration indexing
        const uint8_t rxAntIdx = thrdIdx / N_SC_PER_THRD_BLK;
        const uint8_t layerIdx = rxAntIdx;

        // Index of DMRS grid in which the DMRS tone being processed by this thread resides
        const uint8_t dmrsGridIdx = scIdx % N_DMRS_GRIDS_PER_PRB;

        const uint32_t startPrbThisIter = startPrbThisThrdBlk + prbIdx;
        const uint32_t startScThisIter = startPrbThisIter*CUPHY_N_TONES_PER_PRB;
        const uint32_t intraThrdBlkRelScIdx = prbIdx*CUPHY_N_TONES_PER_PRB + scIdx; // 0 based relative subcarrier index (across all thread blocks) for HEst indexing       
        const uint32_t interThrdBlkRelScIdx = startScThisIter + scIdx; // 0 based relative subcarrier index (across all thread blocks) for HEst indexing       
        const uint32_t absScIdx = startSc + interThrdBlkRelScIdx; // startPrb based absoluate subcarrier index for DataRx indexing      

        // Frequency (subcarrier) based thread activation mask
        // Handle the case where intraThrdBlkRelScIdx >= nScThisThrdBlk or equivalently prbIdx >= nPrbThisThrdBlk
        bool thrdActive = intraThrdBlkRelScIdx < nScThisThrdBlk ? true : false;


        // Zero initialize shRww
        for(uint32_t rwwInitIterIdx = 0; rwwInitIterIdx < nIterRwwProc; ++rwwInitIterIdx)
        {
            uint32_t rwwLinIdx = rwwInitIterIdx * nThrds + thrdIdx;
            if(rwwLinIdx < nShRwwElems)
            {
                shRww.pAddr[rwwLinIdx] = cuGet<TComplexCompute>(0);
            }    
        }
        

        //--------------------------------------------------------------------------------------------------------
        // 2. DMRS signal generation
        // Note: All layers use the same scrambling sequence
        
        // DMRS scrambling bits generated correspond to subcarriers across frequency
        // e.g. 2 bits for tone 0(grid 0 or 1) | 2 bits for tone 1(grid 0 or 1) | 2 bits for tone 2(grid 0 or 1) | 2 bits for tone 3(grid 0 or 1) | ...
        const uint32_t dmrsGridToneIdx = (scIdx / N_DMRS_GRIDS_PER_PRB);
        const uint32_t dmrsRelGridToneIdx = prbIdx*N_DMRS_GRID_TONES_PER_PRB + dmrsGridToneIdx;
        const uint32_t dmrsToneScrSeqBitIdx = dmrsScrStartBitOffset + (dmrsRelGridToneIdx * N_DMRS_SCR_BITS_PER_TONE);
        const uint32_t dmrsScrSeqRdBitIdx = dmrsToneScrSeqBitIdx % N_DMRS_SCR_BITS_GEN_PER_THRD;
        const uint32_t dmrsScrSeqRdWordIdx = dmrsToneScrSeqBitIdx / N_DMRS_SCR_BITS_GEN_PER_THRD;

        for(int32_t dmrsSymIdx = dmrsSymStart; dmrsSymIdx < dmrsSymStop; ++dmrsSymIdx)
        {
            uint8_t chEstIdx = dmrsSymIdx / dmrsMaxLen;
            if(thrdActive && (layerIdx < nLayers))
            {
                // Layer to port map
                uint8_t portIdx = pDmrsPortIdxs[layerIdx];
                bool enableGrid = isGridEnabled(portIdx, dmrsGridIdx);

                shTxDmrs(scIdx, layerIdx) = cuGet<TComplexCompute>(0);
                
                if(enableGrid)
                {
                    if(enableTfPrcd==0)
                    {
                        //--------------------------------------------------------------------------------------------------------
                        // 2a. Generate scrambling sequence (QAM4 symbols)
                        int8_t scrIBit = (shScrWords[dmrsSymIdx][dmrsScrSeqRdWordIdx] >> dmrsScrSeqRdBitIdx) & 0x1;
                        int8_t scrQBit = (shScrWords[dmrsSymIdx][dmrsScrSeqRdWordIdx] >> (dmrsScrSeqRdBitIdx + 1)) & 0x1;
                
                        TComplexCompute scrSeq =
                            cuGet<TComplexCompute>((1 - 2 * scrIBit) * RECIPROCAL_SQRT2, (1 - 2 * scrQBit) * RECIPROCAL_SQRT2);
                                    
                        //--------------------------------------------------------------------------------------------------------
                        // 2b. Apply cover codes (gridShift, fOCC, tOCC) to generate a copy of the transmitted DMRS signal
                        bool enableFOCC = (portIdx & PORT_IDX_FOCC_MSK) ? true : false; // fOCC enabled for odd ports only
                        bool enableTOCC = (portIdx & PORT_IDX_TOCC_MSK) ? true : false; // tOCC enabled for ports 4-7
                        
                        int8_t fOCC = (enableFOCC && (dmrsGridToneIdx & 0x1)) ? -1 : 1; // -1 for odd grid tones and +1 for even grid tones
                        int8_t tOCC = (enableTOCC && (dmrsSymIdx % dmrsMaxLen)) ? -1 : 1; // -1 for 2nd DMRS symbol in the double DMRS (maxLen = 1) and +1 for 1st DMRS symbol in the double DMRS (maxLen = 0 or 1)
        
                        shTxDmrs(scIdx, layerIdx) = scrSeq * (dmrsScale * (fOCC * tOCC));
    
#ifdef ENABLE_DEBUG
                        printf("cuPHY::Pusch::noiseIntfEstKernel - shDmrs[%04d][%1d][%1d] %010.7f+j%010.7f addr %p scrSeq %010.7f+j%010.7f scrIQ (%d,%d) dmrsScrSeqRdWordIdx %d dmrsScrSeqRdBitIdx %02d dmrsGridToneIdx %04d dmrsToneScrSeqBitIdx %02d portIdx %1d fOCC %2d tOCC %2d dmrsGridIdx %1d thrdIdx %04d blockIdx (%d %d %d) threadIdx (%d %d %d)\n",
                               interThrdBlkRelScIdx,
                               layerIdx,
                               dmrsSymIdx,
                               shTxDmrs(scIdx, layerIdx).x,
                               shTxDmrs(scIdx, layerIdx).y,
                               shTxDmrs.pAddr + shTxDmrs.offset(scIdx, layerIdx),
                               scrSeq.x,
                               scrSeq.y,
                               scrIBit,
                               scrQBit,
                               dmrsScrSeqRdWordIdx,
                               dmrsScrSeqRdBitIdx,
                               dmrsGridToneIdx,
                               dmrsToneScrSeqBitIdx,                           
                               portIdx,
                               fOCC,
                               tOCC,
                               dmrsGridIdx,
                               thrdIdx,
                               blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
#endif // ENABLE_DEBUG
                    }
                    else if(enableTfPrcd==1)
                    {
                        uint16_t M_ZC  = N_DMRS_GRID_TONES_PER_PRB * nPrb; //different from DMRS_ABS_TONE_IDX
                        int u = 0;
                        int v = 0;
                        if(drvdUeGrpPrms.optionalDftSOfdm)
                        {
                            u = (int)drvdUeGrpPrms.lowPaprGroupNumber;
                            v = (int)drvdUeGrpPrms.lowPaprSequenceNumber;
                        }
                        else
                        {
                            const uint32_t puschIdentity          = drvdUeGrpPrms.puschIdentity;
                            const uint8_t  groupOrSequenceHopping = drvdUeGrpPrms.groupOrSequenceHopping;
                            const uint8_t  N_symb_slot            = drvdUeGrpPrms.N_symb_slot;
                      
                            int f_gh       = 0;
                            
                            if(groupOrSequenceHopping==1)
                            {
                                uint32_t cInit = floor(puschIdentity/30);
                                for(int m = 0; m < 8; m++)
                                {
                                    uint32_t idxSeq = 8 * (slotNum * N_symb_slot + pDmrsSymPos[dmrsSymIdx]) + m;
                                    f_gh = f_gh + ((descrambling::gold32(cInit, idxSeq) >> (idxSeq % 32)) & 0x1) * (1 << m);
                                }
                                f_gh = f_gh % 30;
    //                            if((blockIdx.x==0)&&(blockIdx.y==0)&&(threadIdx.x==0)&&(threadIdx.y==0))
    //                            {
    //                                printf("f_gh[%d]\n", f_gh);
    //                            }
                            }
                            else if(groupOrSequenceHopping==2)
                            {
                                if(M_ZC > 6 * CUPHY_N_TONES_PER_PRB)
                                {
                                    uint32_t idxSeq = slotNum * N_symb_slot + pDmrsSymPos[dmrsSymIdx];
                                    v = (descrambling::gold32(puschIdentity, idxSeq) >> (idxSeq % 32)) & 0x1;
                                    
    //                                if((blockIdx.x==0)&&(blockIdx.y==0)&&(threadIdx.x==0)&&(threadIdx.y==0))
    //                                {
    //                                    printf("idxSeq[%d]v[%d]\n", idxSeq, v);
    //                                }
                                }
                            }
                            
                            u = (f_gh + puschIdentity)%30;
                        }
                        uint16_t rIdx = startPrbThisIter * N_DMRS_GRID_TONES_PER_PRB + dmrsGridToneIdx;
#ifdef ENALBE_COMMON_DFTSOFDM_DESCRCODE_SUBROUTINE
                        float2 descrCode = gen_pusch_dftsofdm_descrcode(M_ZC, rIdx, u, v, nPrb, d_phi_6[u][rIdx], d_phi_12[u][rIdx], d_phi_18[u][rIdx], d_phi_24[u][rIdx], d_primeNums);
#else
                        float2 descrCode = gen_pusch_dftsofdm_descrcode(M_ZC, rIdx, u, v, nPrb);
#endif
                        shTxDmrs(scIdx, layerIdx) = cuGet<TComplexCompute>(descrCode.x, descrCode.y) * dmrsScale;
                    }
                }

#ifdef ENABLE_DEBUG
                printf("cuPHY::Pusch::noiseIntfEstKernel - shDmrs[%04d][%1d][%1d] %010.7f+j%010.7f  addr %p enableGrid %1d portIdx %02d dmrsGridIdx %1d thrdIdx %04d blockIdx (%d %d %d) threadIdx (%d %d %d)\n", interThrdBlkRelScIdx, layerIdx, dmrsSymIdx, shTxDmrs(scIdx, layerIdx).x, shTxDmrs(scIdx, layerIdx).y, shTxDmrs.pAddr + shTxDmrs.offset(scIdx, layerIdx), enableGrid, portIdx, dmrsGridIdx, thrdIdx, blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
#endif   
            }
            thisThrdBlk.sync();

            //--------------------------------------------------------------------------------------------------------
            // 3. Generate noise-interference free DMRS signal (to within channel estimation error) at the receiver by applying estimated 
            // channel to the generated DMRS signal
            if(thrdActive && (rxAntIdx < nRxAnt))
            {
                TComplexCompute rxPilotEstNoNoiseIntf = cuGet<TComplexCompute>(0);

                // Matrix-vector multiply
                // Inner product of row "rxAntIdx" of H with column vector shTxDmrs of length "nLayers"
                for(uint8_t layerIdx = 0; layerIdx < nLayers; ++layerIdx)
                {
                    bool enableGrid = isGridEnabled(pDmrsPortIdxs[layerIdx], dmrsGridIdx);
                    if(enableGrid)
                    {
                        rxPilotEstNoNoiseIntf += (type_convert<TComplexCompute>(tHEst(rxAntIdx, layerIdx, interThrdBlkRelScIdx, chEstIdx)) * shTxDmrs(scIdx, layerIdx)); // shHEst(rxAntIdx, layerIdx, scIdx, chEstIdx, currPipeStageIdx) * shTxDmrs[layerIdx];
                    }
    
#ifdef ENABLE_DEBUG
                    TComplexCompute chEst = type_convert<TComplexCompute>(tHEst(rxAntIdx, layerIdx, interThrdBlkRelScIdx, chEstIdx));
                    printf("cuPHY::Pusch::noiseIntfEstKernel - rxPilotEstNoNoiseIntf[%03d][%04d][%02d][%1d] = %010.7f+j%010.7f chEst[%03d][%02d][%1d][%1d] = %010.7f+j%010.7f txDmrs[%03d][%1d] = %010.7f+j%010.7f thrdIdx %04d\n", ueGrpIdx, interThrdBlkRelScIdx, rxAntIdx, dmrsSymIdx, cuReal(rxPilotEstNoNoiseIntf), cuImag(rxPilotEstNoNoiseIntf), interThrdBlkRelScIdx, rxAntIdx, layerIdx, chEstIdx, cuReal(chEst), cuImag(chEst), scIdx, layerIdx, cuReal(shTxDmrs(scIdx, layerIdx)), cuImag(shTxDmrs(scIdx, layerIdx)), thrdIdx);
#endif
                }

                TComplexCompute rxPilot = type_convert<TComplexCompute>(tDataRx(absScIdx, pDmrsSymPos[dmrsSymIdx], rxAntIdx));
                shNoiseIntfEst(rxAntIdx, scIdx) = rxPilot - rxPilotEstNoNoiseIntf;

#ifdef ENABLE_DEBUG
                printf("cuPHY::Pusch::noiseIntfEstKernel - noiseIntfEst[%03d][%04d][%02d][%1d] = %010.7f+j%010.7f rxPilot = %010.7f+j%010.7f rxPilotEstNoNoiseIntf = %010.7f+j%010.7f thrdIdx %04d\n", ueGrpIdx, interThrdBlkRelScIdx, rxAntIdx, dmrsSymIdx, cuReal(shNoiseIntfEst(rxAntIdx, scIdx, dmrsSymIdx)), cuImag(shNoiseIntfEst(rxAntIdx, scIdx, dmrsSymIdx)), cuReal(rxPilot), cuImag(rxPilot), cuReal(rxPilotEstNoNoiseIntf), cuImag(rxPilotEstNoNoiseIntf), thrdIdx);
#endif // ENABLE_DEBUG
            }
            thisThrdBlk.sync();

            //--------------------------------------------------------------------------------------------------------
            // 4. Compute and accumulate noise-interference co-variance matrix (accumulation across subcarriers and time i.e. nDmrsSyms)
            if(thrdActive && (rxAntIdx < nRxAnt))
            {
                uint32_t prbIdx2 = scIdx / CUPHY_N_TONES_PER_PRB;
                
                if(noiseCovFlag)
                {
                    // Noise-interference co-variance matrix is the outer product of noise-interference vector
                    for(uint8_t rxAntIdx2 = 0; rxAntIdx2 < nRxAnt; ++rxAntIdx2)
                    {
                        TComplexCompute rww = shNoiseIntfEst(rxAntIdx, scIdx) * cuConj(shNoiseIntfEst(rxAntIdx2, scIdx));

                        atomicAdd(&shRww(rxAntIdx, rxAntIdx2, prbIdx2).x, cuReal(rww));
                        atomicAdd(&shRww(rxAntIdx, rxAntIdx2, prbIdx2).y, cuImag(rww));

#ifdef ENABLE_DEBUG
                        __threadfence();
                        printf("cuPHY::Pusch::noiseIntfEstKernel - instRww[%03d][%03d][%04d][%02d][%02d] = %010.7f+j%010.7f accumRww = %010.7f+j%010.7f\n", ueGrpIdx, startPrbThisIter + prbIdx2, interThrdBlkRelScIdx, rxAntIdx, rxAntIdx2, cuReal(rww), cuImag(rww), cuReal(shRww(rxAntIdx, rxAntIdx2, prbIdx2)), cuImag(shRww(rxAntIdx, rxAntIdx2, prbIdx2)));
#endif // ENABLE_DEBUG                    
                    }
                }
                else
                {
                    for(uint8_t rxAntIdx2 = 0; rxAntIdx2 < nRxAnt; ++rxAntIdx2)
                    {
                        uint8_t flag = ((nDmrsCdmGrpsNoData == 1) && (scIdx % 2 == 1));
                        if(flag == 0)
                        {
                            TComplexCompute rww = shNoiseIntfEst(rxAntIdx, scIdx) * cuConj(shNoiseIntfEst(rxAntIdx2, scIdx));
    
                            atomicAdd(&shRww(rxAntIdx, rxAntIdx2, prbIdx2).x, cuReal(rww));
                            atomicAdd(&shRww(rxAntIdx, rxAntIdx2, prbIdx2).y, cuImag(rww));
    
#ifdef ENABLE_DEBUG
                            __threadfence();
                            printf("cuPHY::Pusch::noiseIntfEstKernel - instRww[%03d][%03d][%04d][%02d][%02d] = %010.7f+j%010.7f accumRww = %010.7f+j%010.7f\n", ueGrpIdx, startPrbThisIter + prbIdx2, interThrdBlkRelScIdx, rxAntIdx, rxAntIdx2, cuReal(rww), cuImag(rww), cuReal(shRww(rxAntIdx, rxAntIdx2, prbIdx2)), cuImag(shRww(rxAntIdx, rxAntIdx2, prbIdx2)));
#endif // ENABLE_DEBUG                    
                        }
                    }
                }

            }
            thisThrdBlk.sync();
        }

        //--------------------------------------------------------------------------------------------------------
        // Average noise-interference co-variance matrix and feed into inverse cholesky factor compute. 
        // Also calculate sum across its diagonal in prepration for noise variance compute
        // (sum of diagonal => accumulation across nRxAnt and additionally accumulate across all PRBs in the UE group)
        for(uint32_t rwwIterIdx = 0; rwwIterIdx < nIterRwwProc; ++rwwIterIdx)
        {
            uint32_t rwwLinIdx = rwwIterIdx * nThrds + thrdIdx;
            uint8_t rxAntIdx1 = rwwLinIdx % nRxAnt;
            uint8_t rxAntIdx2 = (rwwLinIdx / nRxAnt) % nRxAnt;            
            uint32_t prbIdx3 = rwwLinIdx / (nRxAnt * nRxAnt);
            
            // Frequency (PRB) based thread activation mask
            // Ensure prbIdx3 is within bounds
            bool thrdActive2 = ((prbIdx3 < 1) && ((prbIdx + prbIdx3) < nPrbThisThrdBlk)) ? true : false;

            if(thrdActive2)
            {
                TComplexCompute rww = shRww(rxAntIdx1, rxAntIdx2, prbIdx3) / (CUPHY_N_TONES_PER_PRB * nDmrsSyms);
                if(nDmrsCdmGrpsNoData==1)
                {
                    rww = rww * cuGet<TComplexCompute>(2.0f, 0.0f);
                }
                
                // Sum diagonal
                if(rxAntIdx1 == rxAntIdx2)
                {
                    rww.x += CUPHY_NOISE_REGULARIZER;
                    TCompute noisePwr = fabsf(cuReal(rww));
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
                    atomicAdd(const_cast<TCompute*>(tNoiseIntfVar.pAddr + tNoiseIntfVar.offset(pAbsUeIdxs[0])), noisePwr);
#else
                    atomicAdd(const_cast<TCompute*>(tNoiseIntfVar.pAddr + tNoiseIntfVar.offset(ueGrpIdx)), noisePwr);
#endif
                }

                shRww(rxAntIdx1, rxAntIdx2, prbIdx3) = rww;
                // tLwInv(rxAntIdx1, rxAntIdx2, startPrbThisIter + prbIdx3) = type_convert<TComplexStorageOut>(rww);
                
#ifdef ENABLE_DEBUG
                printf("cuPHY::Pusch::noiseIntfEstKernel - Rww[%03d][%03d][%02d][%02d] = %010.7f+j%010.7f accumRww[%03d][%02d][%02d] = %010.7f+j%010.7f\n", ueGrpIdx, startPrbThisIter + prbIdx3, rxAntIdx1, rxAntIdx2, cuReal(rww), cuImag(rww), startPrbThisIter + prbIdx3, rxAntIdx1, rxAntIdx2, cuReal(shRww(rxAntIdx1, rxAntIdx2, prbIdx3)), cuImag(shRww(rxAntIdx1, rxAntIdx2, prbIdx3)));
#endif // ENABLE_DEBUG
            }
            // printf("thrdIdx %d rwwIterIdx %d prbIdx %d prbIdx3 %d thrdActive %d thrdActive2 %d interThrdBlkRelScIdx %d nScThisThrdBlk %d intraThrdBlkRelScIdx %d\n", thrdIdx, rwwIterIdx, prbIdx, prbIdx3, thrdActive, thrdActive2, interThrdBlkRelScIdx , nScThisThrdBlk, intraThrdBlkRelScIdx);

        }
        thisThrdBlk.sync();
        
#ifdef ENABLE_DEBUG        
        if(0 == thrdIdx)
        {
            printf("cuPHY::Pusch::noiseIntfEstKernel: prbGrpIdx %d ueGrpIdx %d prbIdx %d startPrbThisIter %d nPrbThisThrdBlk %d\n", prbGrpIdx, ueGrpIdx, prbIdx, startPrbThisIter, nPrbThisThrdBlk);
        }
#endif // ENABLE_DEBUG

        if(noiseCovFlag)
        {
            cholFactorInv<TCompute, 
                          TStorageOut>(startPrbThisIter, nPrbThisThrdBlk, nRxAnt, shRww, tLwInv);

        }
    }

    // Logic to ensure all thread blocks in the grid complete (ending in writes to tNoiseIntfVar above)
    if(0 == thrdIdx)
    {
        // Ensure interCtaSyncCnt is incremented only after the tNoiseIntfVar global memory writes have been completed.
        // Note that while the global memory access above is atomic, it does not imply ordering constraints for memory operations,
        // hence a threadfence is still needed.
        __threadfence(); //<--

        uint32_t syncCnt = atomicInc(const_cast<uint32_t*>(&interCtaSyncCnt), nUeGrpThrdBlksNeeded);
        
        // Is this the last CTA to be processed for this user group?        
        isLastCtaDone = (syncCnt == (nUeGrpThrdBlksNeeded - 1)) ? true : false;

#ifdef ENABLE_DEBUG
        printf("cuPHY::Pusch::noiseIntfEstKernel: prbGrpIdx %d ueGrpIdx %d isLastCtaDone %u nUeGrpThrdBlksNeeded %d syncCnt %d interCtaSyncCnt %d\n", prbGrpIdx, ueGrpIdx, isLastCtaDone, nUeGrpThrdBlksNeeded, syncCnt, interCtaSyncCnt);
#endif
    }

    thisThrdBlk.sync();
 
    // Use one thread block per UE group to average noise-interference variance across nRxAnt, nPrb and convert from linear to dB
    if(isLastCtaDone)
    {
        if(0 == thrdIdx)
        {
            // Average across across Rx antenna and PRBs allocated to this UE group
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
            TCompute avgNoiseIntfVar     = type_convert<TCompute>(tNoiseIntfVar(pAbsUeIdxs[0])) / (nRxAnt * nPrb);
#else
            TCompute avgNoiseIntfVar     = type_convert<TCompute>(tNoiseIntfVar(ueGrpIdx)) / (nRxAnt * nPrb);
#endif
            invNoiseVarLinear            = float(1.0f / avgNoiseIntfVar);
            drvdUeGrpPrms.noiseVarForDtx = float(avgNoiseIntfVar);
            if(DMRS_SYMBOL_IDX==CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT)
            {
                TCompute noiseIntfVarDb      = 10*log10f(avgNoiseIntfVar) + NOISE_EST_CORRECTION_DB;
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
                // noiseVarPreEq per UE
                for(uint32_t i = 0; i < nUes; ++i)
                {      
                    tNoiseIntfVar(pAbsUeIdxs[i]) = type_convert<TStorageOut>(noiseIntfVarDb);  
                } 
#else
                // noiseVarPreEq per UEGRP
                tNoiseIntfVar(ueGrpIdx) = type_convert<TStorageOut>(noiseIntfVarDb);   
#endif
            }
            else if(DMRS_SYMBOL_IDX==CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0)
            {
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
                // reset for full-slot processing
                for(uint32_t i = 0; i < nUes; ++i)
                {      
                    tNoiseIntfVar(pAbsUeIdxs[i]) = type_convert<TStorageOut>(0);  
                } 
#else
                // noiseVarPreEq per UEGRP
                tNoiseIntfVar(ueGrpIdx) = type_convert<TStorageOut>(0);   
#endif
                interCtaSyncCnt = 0; //reset for full-slot processing  
            }
#ifdef ENABLE_DEBUG
            // printf("cuPHY::Pusch::noiseIntfEstKernel - nSymb %d nRxAnt %d nUeGrps %d nThrds %d nIter %d ueGrpIdx %d\n", nSymb, nRxAnt, nUeGrps, nThrds, nIter, ueGrpIdx2);
            printf("cuPHY::Pusch::noiseIntfEstKernel - noiseIntfVar: ueGrp[%d] %010.7f db (Lin: %010.7f)\n", ueGrpIdx, noiseIntfVarDb, avgNoiseIntfVar);
#endif
        }
    }
  } //noiseIntfEstKernel

 template<uint32_t N_PRB_PER_THRD_BLK, typename TCompute>           // N_PRB_PER_THRD_BLK # of PRBs processed in a thread block
 void puschRxNoiseIntfEst::noiseIntfEstLaunchGeo(uint16_t nMaxPrb,
                                                 uint16_t nRxAnt,
                                                 uint16_t nUeGrps,
                                                 dim3&    gridDim,
                                                 dim3&    blockDim,
                                                 size_t&  sharedMemBytes)
 { 
     // Number of thread blocks needed to process a UE group with nMaxPrb allocation
     uint32_t nThrdBlksPerUeGrp = div_round_up<uint32_t>(nMaxPrb, N_PRB_PER_THRD_BLK);

     static constexpr uint32_t N_SC_PER_THRD_BLK = CUPHY_N_TONES_PER_PRB;
     blockDim.x = N_SC_PER_THRD_BLK*nRxAnt;
     blockDim.y = 1;
     blockDim.z = 1;
 
     gridDim.x = nThrdBlksPerUeGrp;
     gridDim.y = nUeGrps;
     gridDim.z = 1;

#ifdef CH_EST_IN_SHARED_MEM
    const uint32_t nShHEstElems = nRxAnt*nRxAnt*N_SC_PER_THRD_BLK*CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; // actual size: nRxAnt*nLayers*N_SC_PER_THRD_BLK*nTimeChEst
#else
    const uint32_t nShHEstElems = 0;
#endif
    const uint32_t nShRwwElems = nRxAnt*nRxAnt;
    const uint32_t nShTxDmrsElems = nRxAnt*N_SC_PER_THRD_BLK; // actual size: nLayers*N_SC_PER_THRD_BLK
    const uint32_t nShNoiseIntfEstElems = nRxAnt*N_SC_PER_THRD_BLK; // nRxAnt*N_SC_PER_THRD_BLK*nDmrsSymbols
    const uint32_t nShNoiseIntfVar = CUPHY_ENABLE_SUB_SLOT_PROCESSING == 1 ? nMaxPrb : 0; // only for sub-slot processing kernel

    sharedMemBytes = (nShHEstElems + nShRwwElems + nShTxDmrsElems + nShNoiseIntfEstElems) * sizeof(typename complex_from_scalar<TCompute>::type);
    sharedMemBytes+= nShNoiseIntfVar * sizeof(TCompute);

 #ifdef ENABLE_DEBUG
     NVLOGI_FMT(NVLOG_PUSCH, "{}: blockDim ({},{},{}), gridDim ({},{},{}), sharedMemBytes {} N_SC_PER_THRD_BLK {} nRxAnt {} nThrdBlksPerUeGrp {} nMaxPrb {}", __FUNCTION__, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z, sharedMemBytes, N_SC_PER_THRD_BLK, nRxAnt, nThrdBlksPerUeGrp, nMaxPrb);
 #endif
 }
 
 template <typename TStorageIn,
           typename TDataRx,
           typename TStorageOut,
           typename TCompute>      
 void puschRxNoiseIntfEst::noiseIntfEst(uint16_t                             nMaxPrb,
                                        uint16_t                             nRxAnt,
                                        uint16_t                             nUeGrps,
                                        uint8_t                              enableDftSOfdm,
                                        uint8_t                              dmrsSymbolIdx,
                                        cuphyPuschRxNoiseIntfEstLaunchCfg_t& launchCfg)
 {
     static constexpr uint32_t N_DMRS_GRIDS_PER_PRB    = 2;
     static constexpr uint32_t N_PRB_PER_THRD_BLK      = 2; // N_PRB_PER_THRD_BLK*12*nRxAnt threads per thread block
     CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = launchCfg.kernelNodeParamsDriver;
     
     if(enableDftSOfdm==1)
     {
         if(dmrsSymbolIdx==CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT)
         {
             static constexpr uint8_t DMRS_SYMBOL_IDX    = CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT;
             void* kernelFunc = reinterpret_cast<void*>(noiseIntfEstKernel<TStorageIn,
                                                                           TDataRx,
                                                                           TStorageOut,
                                                                           TCompute,
                                                                           N_DMRS_GRIDS_PER_PRB,
                                                                           N_PRB_PER_THRD_BLK,
                                                                           DMRS_SYMBOL_IDX>);
             CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));
         }
         else if(dmrsSymbolIdx==CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0)
         {
             static constexpr uint8_t DMRS_SYMBOL_IDX    = CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0;
             void* kernelFunc = reinterpret_cast<void*>(noiseIntfEstKernel<TStorageIn,
                                                                           TDataRx,
                                                                           TStorageOut,
                                                                           TCompute,
                                                                           N_DMRS_GRIDS_PER_PRB,
                                                                           N_PRB_PER_THRD_BLK,
                                                                           DMRS_SYMBOL_IDX>);
             CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));
         }
     }
     else if(enableDftSOfdm==0)
     {
         if(dmrsSymbolIdx==CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT)
         {
             static constexpr uint8_t DMRS_SYMBOL_IDX    = CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT;
             void* kernelFunc = reinterpret_cast<void*>(noiseIntfEstNoDftSOfdmKernel<TStorageIn,
                                                                                     TDataRx,
                                                                                     TStorageOut,
                                                                                     TCompute,
                                                                                     N_DMRS_GRIDS_PER_PRB,
                                                                                     N_PRB_PER_THRD_BLK,
                                                                                     DMRS_SYMBOL_IDX>);
             CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));
         }
         if(dmrsSymbolIdx==CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0)
         {
             static constexpr uint8_t DMRS_SYMBOL_IDX    = CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0;
             void* kernelFunc = reinterpret_cast<void*>(noiseIntfEstNoDftSOfdmKernel<TStorageIn,
                                                                                     TDataRx,
                                                                                     TStorageOut,
                                                                                     TCompute,
                                                                                     N_DMRS_GRIDS_PER_PRB,
                                                                                     N_PRB_PER_THRD_BLK,
                                                                                     DMRS_SYMBOL_IDX>);
             CUDA_CHECK(cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc));
         }
     }
     
     size_t sharedMemBytes = 0;
     dim3 blockDim, gridDim;
     noiseIntfEstLaunchGeo<N_PRB_PER_THRD_BLK, TCompute>(nMaxPrb, nRxAnt, nUeGrps, gridDim, blockDim, sharedMemBytes);
 
     kernelNodeParamsDriver.blockDimX = blockDim.x;
     kernelNodeParamsDriver.blockDimY = blockDim.y;
     kernelNodeParamsDriver.blockDimZ = blockDim.z;
     
     kernelNodeParamsDriver.gridDimX = gridDim.x;
     kernelNodeParamsDriver.gridDimY = gridDim.y;
     kernelNodeParamsDriver.gridDimZ = gridDim.z;
 
     kernelNodeParamsDriver.extra          = nullptr;
     kernelNodeParamsDriver.sharedMemBytes = sharedMemBytes;
 }
 
 void puschRxNoiseIntfEst::kernelSelect(uint16_t                     nMaxPrb,
                                        uint16_t                     nRxAnt,
                                        uint16_t                     nUeGrps,
                                        uint8_t                      enableDftSOfdm,
                                        uint8_t                      dmrsSymbolIdx,
                                        cuphyDataType_t              chEstType,
                                        cuphyDataType_t              dataRxType,
                                        cuphyDataType_t              noiseIntfVarType,
                                        cuphyDataType_t              lwInvType,
                                        cuphyPuschRxNoiseIntfEstLaunchCfg_t& launchCfg)
 {
     // Note: Output tensor tNoiseIntfVar used in global memory accumulation. Hence using single precision
 
     using TCompute = float;
     if((CUPHY_C_16F == dataRxType) && (CUPHY_C_32F == chEstType) && (CUPHY_R_32F == noiseIntfVarType) && (CUPHY_C_32F == lwInvType))
     {
         using TDataRx = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
         using TStorageIn  = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
         using TStorageOut = data_type_traits<CUPHY_R_32F>::type; 
         noiseIntfEst<TStorageIn, TDataRx, TStorageOut, TCompute>(nMaxPrb,
                                                                  nRxAnt,
                                                                  nUeGrps,
                                                                  enableDftSOfdm,
                                                                  dmrsSymbolIdx,
                                                                  launchCfg);
     }     
#if 0  // Disable unused templates to save code space and compile time
     else if(CUPHY_C_32F == dataRxType)
     {
         using TDataRx = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
         if((CUPHY_C_32F == chEstType) && (CUPHY_R_32F == noiseIntfVarType) && (CUPHY_C_32F == lwInvType))
         {
             using TStorageIn = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;        
             using TStorageOut = data_type_traits<CUPHY_R_32F>::type;
             noiseIntfEst<TStorageIn, TDataRx, TStorageOut, TCompute>(nMaxPrb,
                                                                      nRxAnt,
                                                                      nUeGrps,
                                                                      enableDftSOfdm,
                                                                      dmrsSymbolIdx
                                                                      launchCfg);
         }
         else
         {
             NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type - DataRx {} ChEst {} NoiseIntfVar {} LwInv {}", __FUNCTION__, cuphyGetDataTypeString(dataRxType), cuphyGetDataTypeString(chEstType), cuphyGetDataTypeString(noiseIntfVarType), cuphyGetDataTypeString(lwInvType));
         }
     }
#endif     
     else
     {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested data type - DataRx {} ChEst {} NoiseIntfVar {} LwInv {}", __FUNCTION__, cuphyGetDataTypeString(dataRxType), cuphyGetDataTypeString(chEstType), cuphyGetDataTypeString(noiseIntfVarType), cuphyGetDataTypeString(lwInvType));
     }
 }
 
 void puschRxNoiseIntfEst::getDescrInfo(size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes)
 {
    dynDescrSizeBytes  = sizeof(puschRxNoiseIntfEstDynDescrVec_t);
    dynDescrAlignBytes = alignof(puschRxNoiseIntfEstDynDescrVec_t);
 }

 cuphyStatus_t puschRxNoiseIntfEst::setup(cuphyPuschRxUeGrpPrms_t*              pDrvdUeGrpPrmsCpu,
                                          cuphyPuschRxUeGrpPrms_t*              pDrvdUeGrpPrmsGpu,
                                          uint16_t                              nUeGrps,
                                          uint32_t		                          nMaxPrb,
                                          uint8_t                               enableDftSOfdm,
                                          uint8_t                               dmrsSymbolIdx,
                                          uint8_t                               enableCpuToGpuDescrAsyncCpy,
                                          void*                                 pDynDescrsCpu,
                                          void*                                 pDynDescrsGpu,
                                          cuphyPuschRxNoiseIntfEstLaunchCfgs_t* pLaunchCfgs,
                                          cudaStream_t                          strm,
                                          uint8_t                               subSlotStageIdx)
 {
     if(enableDftSOfdm > 1)
     {
         NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Invalid enableDftSOfdm {}", __FUNCTION__, enableDftSOfdm);
         return CUPHY_STATUS_INVALID_ARGUMENT;
     }
     // only support dmrsSymbolIdx = CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0 or dmrsSymbolIdx = CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT
     if((dmrsSymbolIdx!=CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0)&&(dmrsSymbolIdx!=CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT))
     {
         NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Invalid dmrsSymbolIdx {}", __FUNCTION__, dmrsSymbolIdx);
         return CUPHY_STATUS_INVALID_ARGUMENT;
     }
     
     puschRxNoiseIntfEstDynDescr_t* pDynDescrVecGpu = static_cast<puschRxNoiseIntfEstDynDescr_t*>(pDynDescrsGpu);
     puschRxNoiseIntfEstDynDescr_t* pDynDescrVecCpu = static_cast<puschRxNoiseIntfEstDynDescr_t*>(pDynDescrsCpu);
     // index offset for sub-slot processing (this offset is 0 for full-slot processing mode)
     int sspOffset = subSlotStageIdx * CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS;

     if(!pDrvdUeGrpPrmsCpu || !pDrvdUeGrpPrmsGpu || !pDynDescrVecGpu || !pLaunchCfgs) return CUPHY_STATUS_INVALID_ARGUMENT;
      
     for(uint32_t hetCfgIdx = 0; hetCfgIdx < pLaunchCfgs->nCfgs; ++hetCfgIdx)
     {
         // Setup descriptor in CPU memory
         puschRxNoiseIntfEstDynDescr_t& dynDescr = pDynDescrVecCpu[hetCfgIdx + sspOffset];
         dynDescr.pDrvdUeGrpPrms = pDrvdUeGrpPrmsGpu;
         dynDescr.subSlotStageIndex = subSlotStageIdx;
 
         puschRxNoiseIntfEstKernelArgs_t& kernelArgs = (dmrsSymbolIdx==CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT) ? m_noiseIntfKernelArgsArr[0][hetCfgIdx + sspOffset] : m_noiseIntfKernelArgsArr[1][hetCfgIdx + sspOffset];
         kernelArgs.pDynDescr = &pDynDescrVecGpu[hetCfgIdx + sspOffset];
 
         // Optional descriptor copy to GPU memory
         if(enableCpuToGpuDescrAsyncCpy)
         {
             CUDA_CHECK(cudaMemcpyAsync(&pDynDescrVecGpu[hetCfgIdx + sspOffset], &dynDescr, sizeof(puschRxNoiseIntfEstDynDescr_t), cudaMemcpyHostToDevice, strm));
         }
 
         // Select kernel
         //TODO: supporting variable antenna counts
         cuphyPuschRxNoiseIntfEstLaunchCfg_t& launchCfg = pLaunchCfgs->cfgs[hetCfgIdx + sspOffset];
         kernelSelect(nMaxPrb,
                      pDrvdUeGrpPrmsCpu[0].nRxAnt,
                      nUeGrps,
                      enableDftSOfdm,
                      dmrsSymbolIdx,
                      pDrvdUeGrpPrmsCpu[0].tInfoHEst.elemType,
                      pDrvdUeGrpPrmsCpu[0].tInfoDataRx.elemType,
                      pDrvdUeGrpPrmsCpu[0].tInfoNoiseVarPreEq.elemType,
                      pDrvdUeGrpPrmsCpu[0].tInfoLwInv.elemType,
                      launchCfg);    
         
         launchCfg.kernelArgs[0] = &kernelArgs.pDynDescr;        
         launchCfg.kernelNodeParamsDriver.kernelParams = &(launchCfg.kernelArgs[0]);

         // printf("cuPHY::puschRxNoiseIntfEst::setup - kernelAddr %p grid(x y z) %d %d %d block(x y z) %d %d %d sharedMemBytes %d kernelParams %p kernelArgs[0] %p gpuDescAddr %p hetCfgIdx %d\n", launchCfg.kernelNodeParamsDriver.func, launchCfg.kernelNodeParamsDriver.gridDimX, launchCfg.kernelNodeParamsDriver.gridDimY,  launchCfg.kernelNodeParamsDriver.gridDimZ,  launchCfg.kernelNodeParamsDriver.blockDimX,  launchCfg.kernelNodeParamsDriver.blockDimY,  launchCfg.kernelNodeParamsDriver.blockDimZ, launchCfg.kernelNodeParamsDriver.sharedMemBytes, &(launchCfg.kernelArgs[0]), &(kernelArgs.pDynDescr), &pDynDescrVecGpu[hetCfgIdx], hetCfgIdx);
     }
     if(enableDftSOfdm==1)
     {
         for(uint32_t idx = 0; idx < nUeGrps; idx++)
         {
             if(pDrvdUeGrpPrmsCpu[idx].enableTfPrcd==1)
             {
                 CUDA_CHECK(cudaMemcpyToSymbolAsync(d_phi_6, phi_6, sizeof(phi_6), 0, cudaMemcpyHostToDevice, strm));
                 CUDA_CHECK(cudaMemcpyToSymbolAsync(d_phi_12, phi_12, sizeof(phi_12), 0, cudaMemcpyHostToDevice, strm));
                 CUDA_CHECK(cudaMemcpyToSymbolAsync(d_phi_18, phi_18, sizeof(phi_18), 0, cudaMemcpyHostToDevice, strm));
                 CUDA_CHECK(cudaMemcpyToSymbolAsync(d_phi_24, phi_24, sizeof(phi_24), 0, cudaMemcpyHostToDevice, strm));
                 CUDA_CHECK(cudaMemcpyToSymbolAsync(d_primeNums, primeNums, sizeof(primeNums), 0, cudaMemcpyHostToDevice, strm));
                 
                 break;
             }
        }
     }
     return CUPHY_STATUS_SUCCESS;
 }

 } // namespace pusch_noise_intf_est
 