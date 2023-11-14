/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
 #include <sstream>  
 #include "cuComplex.h"
 #include <cooperative_groups.h>
 #include "descrambling.cuh"
 #include "ch_est.hpp"
 #include "type_convert.hpp"
 #include <vector>
 #include "utils.cuh"
 #include "nvlog.hpp"
 #include "cuphy.hpp"
 
 using namespace cooperative_groups;
 using namespace descrambling;
 
 namespace ch_est
 {
 // #define ENABLE_PROFILING
 // #define ENABLE_DEBUG
 
 // #define ENABLE_PRIME_WHILE_LOOP
 #define ENALBE_COMMON_DFTSOFDM_DESCRCODE_SUBROUTINE // if ENALBE_COMMON_DFTSOFDM_DESCRCODE_SUBROUTINE is defined, ENABLE_PRIME_WHILE_LOOP should not be defined.

 // PRB size in tones
 static constexpr uint32_t N_TONES_PER_PRB    = 12;
 static constexpr uint32_t N_TOCC   = 2;

 // Total # of symbols after tOCC and fOCC removal
 static constexpr uint32_t N_DMRS_SYMS_OCC = 4; // reserve memory for the worst-case 

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
 
 static CUDA_BOTH_INLINE cuComplex operator+=(cuComplex &x, cuComplex y)       { x = cuCaddf(x, y); return x; };
 static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, int y)               { return(make_cuComplex(cuCrealf(x)*float(y), cuCimagf(x)*float(y))); }
 static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, float y)             { return(make_cuComplex(cuCrealf(x)*y, cuCimagf(x)*y)); }
 static CUDA_BOTH_INLINE cuComplex operator*=(cuComplex &x, const cuComplex y) { x = cuCmulf(x, y); return x; };
 
 // #ifdef ENABLE_DEBUG
 static CUDA_BOTH_INLINE cuComplex operator*(cuComplex x, cuComplex y) { return(cuCmulf(x, y)); }
 // #endif
 
 //static CUDA_BOTH_INLINE float cuReal(cuComplex x) { return(cuCrealf(x)); }
 //static CUDA_BOTH_INLINE float cuImag(cuComplex x) { return(cuCimagf(x)); }
 static CUDA_BOTH_INLINE cuComplex cuConj(cuComplex x) { return(cuConjf(x)); }
 
 #if 0
 static CUDA_BOTH_INLINE float cuReal(cuComplex x) { return(cuCrealf(x)); }
 static CUDA_BOTH_INLINE float cuImag(cuComplex x) { return(cuCimagf(x)); }
 static CUDA_BOTH_INLINE cuComplex cuAdd(cuComplex x, cuComplex y) { return(cuCaddf(x, y)); }
 static CUDA_BOTH_INLINE cuComplex cuMul(cuComplex x, cuComplex y) { return(cuCmulf(x, y)); }
 static CUDA_BOTH_INLINE cuComplex cuDiv(cuComplex x, cuComplex y) { return(cuCdivf(x, y)); }
 static CUDA_BOTH_INLINE cuComplex operator+(cuComplex x, cuComplex y) { return(cuCaddf(x, y)); }
 static CUDA_BOTH_INLINE cuComplex operator-(cuComplex x, cuComplex y) { return(cuCsubf(x, y)); }
 static CUDA_BOTH_INLINE cuComplex operator+=(cuComplex &x, cuComplex y)  { x = cuCaddf(x, y); return x; }; 
 static CUDA_BOTH_INLINE cuComplex operator-=(cuComplex x, cuComplex y) { return(cuCsubf(x, y)); }
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
 
 CUDA_INLINE constexpr uint32_t get_inter_dmrs_grid_freq_shift(const uint32_t nDmrsGridsPerPrb)
 {
     return (2 == nDmrsGridsPerPrb) ? 1 : 2;
 }
 
 CUDA_INLINE constexpr uint32_t get_inter_dmrs_grid_freq_shift_idx(const uint32_t nDmrsGridsPerPrb, const uint32_t gridIdx)
 {
     return ((nDmrsGridsPerPrb - 1) - gridIdx) * get_inter_dmrs_grid_freq_shift(nDmrsGridsPerPrb);
 }
 
 CUDA_INLINE constexpr uint32_t get_smem_dmrs_tone_idx(const uint32_t nDmrsGridsPerPrb, const uint32_t nInterDmrsGridFreqShift, const uint32_t tIdx)
 {
     return (2 == nDmrsGridsPerPrb) ? (tIdx / nDmrsGridsPerPrb) :
                                      (nInterDmrsGridFreqShift * (tIdx / (nInterDmrsGridFreqShift * nDmrsGridsPerPrb)) +
                                       (tIdx % nInterDmrsGridFreqShift));
 }
 
 CUDA_INLINE constexpr uint32_t get_smem_dmrs_grid_idx(const uint32_t nDmrsGridsPerPrb, const uint32_t nInterDmrsGridFreqShift, const uint32_t tIdx)
 {
     return (2 == nDmrsGridsPerPrb) ? (tIdx % nDmrsGridsPerPrb) :
                                      (tIdx / nInterDmrsGridFreqShift) % nDmrsGridsPerPrb;
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

 
 // Channel Estimation kernel:
 // Performs frequency domain interpolation: Uses DMRS tones in N_DMRS_PRB_IN_PER_CLUSTER PRBs and generates
 // channel estimate over N_DMRS_INTERP_PRB_OUT_PER_CLUSTER PRBs for all the layers present in N_DMRS_PRB_IN_PER_CLUSTER PRBs
 // Each thread block consumes a pilot chunk: N_DMRS_PRB_IN_PER_CLUSTER x N_DMRS_SYMS pilot and outputs H as:
 // N_DMRS_SYMS_FOCC x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB x N_DMRS_INTERP_PRB_OUT_PER_CLUSTER = N_LAYERS x N_DMRS_INTERP_PRB_OUT_PER_CLUSTER
 // Inputs and outputs assumed to be column major
 
 // Note: The shift and unshift sequences are the same for all DMRS time resources (symbols) but different for
 // DMRS frequency resources (tones). The descrambling sequence is different for each time (symbol) and
 // frequencey (tone) resource
 
 // Since each thread block estimates channel H for a PRB cluster of size N_DMRS_INTERP_PRB_OUT_PER_CLUSTER PRBs
 // # of thread blocks needed = gridDim = N_DATA_PRB/N_DMRS_INTERP_PRB_OUT_PER_CLUSTER
 // dimBlock: (N_DMRS_PRB_IN_PER_CLUSTER*N_TONES_PER_PRB) dimGrid: (N_DATA_PRB/N_DMRS_INTERP_PRB_OUT_PER_CLUSTER, nBSAnt)
 // Tested for: N_DATA_PRB = 64, N_DMRS_INTERP_PRB_OUT_PER_CLUSTER = 4, N_DMRS_PRB_IN_PER_CLUSTER = 8, dimBlock(96) and dimGrid(68, 16)
 // where N_DATA_PRB is the total # of PRBs bearing data i.e. total number interpolatd DMRS PRBs produced by
 
 // Channel estimates from only the active DMRS grids are saved. Table below contains mapping of active DMRS grid
 // index bitmask to DMRS grid write index (-1 for inactive grids). Choose data type to be smallest possible type
 // to save on table size (and hence the memory foot print)
 __constant__ int8_t DMRS_GRID_WR_IDX_TBL[][3]{{-1, -1, -1}, {0, -1, -1}, {-1, 0, -1}, {0, 1, -1}, {-1, -1, 0}, {0, -1, 1}, {-1, 0, 1}, {0, 1, 2}};
 
 #if 0
 template <typename TStorage,
           typename TDataRx,
           typename TCompute,
           uint32_t N_LAYERS,                          // # of layers (# of cols in H matrix)
           uint32_t N_DMRS_GRIDS_PER_PRB,              // # of DMRS grids per PRB (2 or 3)
           uint32_t N_DMRS_PRB_IN_PER_CLUSTER,         // # of PRBs bearing DMRS tones to be processed by each thread block (i.e. used in channel estimation)
           uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER, // # of PRBs bearing channel estimates (interpolated tones) at output
           uint32_t N_DMRS_SYMS>                       // # of time domain DMRS symbols (1,2 or 4)
 static __global__ void
 windowedChEstKernel(puschRxChEstStatDescr_t* pStatDescr, puschRxChEstDynDescr_t* pDynDescr)
 {
     // PRB cluster being processed by this thread block
     const uint32_t PRB_CLUSTER_IDX = blockIdx.x;
     // BS antenna being processed by this thread block
     const uint32_t BS_ANT_IDX = blockIdx.y;
 
     if((0 != BS_ANT_IDX) || (0 != PRB_CLUSTER_IDX)) return;
 
     if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     {
         puschRxChEstDynDescr_t& dynDescr      = *(pDynDescr);
         uint16_t                slotNum            = dynDescr.slotNum;
         uint8_t                 activeDmrsGridBmsk = dynDescr.activeDmrsGridBmsk;
         tensor_ref<const uint16_t>       tDmrsScId(dynDescr.tPrmDmrsScId.pAddr             , dynDescr.tPrmDmrsScId.strides);        // (N_UE_GRPS) 
         uint16_t dmrsScId = tDmrsScId(blockIdx.z);
         printf("dmrsScId %d slotNum %d activeDmrsGridBmsk 0x%08x\n", dmrsScId, slotNum, activeDmrsGridBmsk);
     }
 }
 #else
 template <typename TStorage,
           typename TDataRx,
           typename TCompute,
           uint32_t N_LAYERS,                          // # of layers (# of cols in H matrix) ** may be larger than the actual number of layers in the group
           uint32_t N_DMRS_GRIDS_PER_PRB,              // # of DMRS grids per PRB (2 or 3)
           uint32_t N_DMRS_PRB_IN_PER_CLUSTER,         // # of PRBs bearing DMRS tones to be processed by each thread block (i.e. used in channel estimation)
           uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER, // # of PRBs bearing channel estimates (interpolated tones) at output
           uint32_t N_DMRS_SYMS>                       // # of consecutive DMRS symbols (1 or 2)
 static __global__ void
 windowedChEstNoDftSOfdmKernel(puschRxChEstStatDescr_t* pStatDescr, puschRxChEstDynDescr_t* pDynDescr)
 {
     //--------------------------------------------------------------------------------------------------------     
     puschRxChEstDynDescr_t& dynDescr = *pDynDescr;

     // UE group processed by this thread block
     const uint32_t UE_GRP_IDX = dynDescr.hetCfgUeGrpMap[blockIdx.z];
     // PRB cluster being processed by this thread block
     const uint32_t PRB_CLUSTER_IDX = blockIdx.x;     
     // BS antenna being processed by this thread block
     const uint32_t BS_ANT_IDX = blockIdx.y;

     // Early exit check
     // The grid is sized to process the max # of PRB clusters in a given heterogenous config. Exit if the PRB cluster to be
     // processed by this thread block does not exist in the UE group
     cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];
     const uint16_t nPrb   = drvdUeGrpPrms.nPrb;
     const uint16_t nRxAnt = drvdUeGrpPrms.nRxAnt;
     const uint32_t N_PRB_CLUSTERS_PER_BS_ANT = div_round_up(nPrb, static_cast<uint16_t>(N_DMRS_INTERP_PRB_OUT_PER_CLUSTER));
     if((PRB_CLUSTER_IDX >= N_PRB_CLUSTERS_PER_BS_ANT) || (BS_ANT_IDX >= nRxAnt)) return;

#ifdef ENABLE_DEBUG     
     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z)) 
     printf("%s\n blockIdx.z %d UE_GRP_IDX %d blockDim (%d,%d,%d) gridDim (%d,%d,%d)\n", __PRETTY_FUNCTION__, blockIdx.z, UE_GRP_IDX, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);

     if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.y) && (0 == blockIdx.z)) 
     printf("PRB_CLUSTER_IDX %d N_PRB_CLUSTERS_PER_BS_ANT %d chEstTimeInst %d\n", PRB_CLUSTER_IDX, N_PRB_CLUSTERS_PER_BS_ANT, dynDescr.chEstTimeInst);
#endif

     //--------------------------------------------------------------------------------------------------------
     // Setup local parameters based on descriptor
     puschRxChEstStatDescr_t& statDescr = *pStatDescr;
     const uint16_t  slotNum            = drvdUeGrpPrms.slotNum;
     const uint8_t   chEstTimeInst      = dynDescr.chEstTimeInst;
     const uint8_t   activeDmrsGridBmsk = drvdUeGrpPrms.activeDMRSGridBmsk;
     uint8_t*        OCCIdx             = drvdUeGrpPrms.OCCIdx;
     const uint16_t  nLayers            = drvdUeGrpPrms.nLayers;
     const uint8_t   dmrsMaxLen         = drvdUeGrpPrms.dmrsMaxLen;
     const uint8_t   nPrbsMod2          = nPrb & 0x1;
     const uint32_t  N_DATA_PRB         = nPrb;
     const uint8_t   scid               = drvdUeGrpPrms.scid;

     // Pointer to DMRS symbol used for channel estimation (single-symbol if maxLen = 1, double-symbol if maxLen = 2)
     uint8_t*        pDmrsSymPos   = &drvdUeGrpPrms.dmrsSymLoc[chEstTimeInst*dmrsMaxLen];
     const uint16_t  startPrb = drvdUeGrpPrms.startPrb;
     const uint16_t  dmrsScId = drvdUeGrpPrms.dmrsScrmId;
     const uint8_t   nDmrsCdmGrpsNoData = drvdUeGrpPrms.nDmrsCdmGrpsNoData;
     
     //--------------------------------------------------------------------------------------------------------
     typedef typename complex_from_scalar<TCompute>::type TComplexCompute;
     typedef typename complex_from_scalar<TDataRx>::type TComplexDataRx;
     typedef typename complex_from_scalar<TStorage>::type TComplexStorage;
 
     // clang-format off
     tensor_ref<const TCompute>       tFreqInterpCoefs((8==N_DMRS_PRB_IN_PER_CLUSTER) ? statDescr.tPrmFreqInterpCoefs.pAddr : statDescr.tPrmFreqInterpCoefs4.pAddr, 
                                                       (8==N_DMRS_PRB_IN_PER_CLUSTER) ? statDescr.tPrmFreqInterpCoefs.strides : statDescr.tPrmFreqInterpCoefs4.strides); // (N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER + N_INTER_DMRS_GRID_FREQ_SHIFT, N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER, 3), 3 filters: 1 for middle, 1 lower edge and 1 upper edge
 
   //  tensor_ref<const TStorage>       tFreqInterpCoefs(statDescr.tPrmFreqInterpCoefs.pAddr, statDescr.tPrmFreqInterpCoefs.strides);
 #if 1 // shift/unshift sequences same precision as data (FP16 or FP32)
     tensor_ref<const TComplexDataRx> tShiftSeq       (statDescr.tPrmShiftSeq.pAddr       , statDescr.tPrmShiftSeq.strides);        // (N_DATA_PRB*N_DMRS_GRID_TONES_PER_PRB, N_DMRS_SYMS)
     tensor_ref<const TComplexDataRx> tUnShiftSeq     (statDescr.tPrmUnShiftSeq.pAddr     , statDescr.tPrmUnShiftSeq.strides);      // (N_DATA_PRB*N_DMRS_INTERP_TONES_PER_GRID*N_DMRS_GRIDS_PER_PRB + N_INTER_DMRS_GRID_FREQ_SHIFT)
 #else // shift/unshift sequences same precision as channel estimates (typically FP32)
     tensor_ref<const TComplexStorage> tShiftSeq       (statDescr.tPrmShiftSeq.pAddr       , statDescr.tPrmShiftSeq.strides);        // (N_DATA_PRB*N_DMRS_GRID_TONES_PER_PRB, N_DMRS_SYMS)
     tensor_ref<const TComplexStorage> tUnShiftSeq     (statDescr.tPrmUnShiftSeq.pAddr     , statDescr.tPrmUnShiftSeq.strides);      // (N_DATA_PRB*N_DMRS_INTERP_TONES_PER_GRID*N_DMRS_GRIDS_PER_PRB + N_INTER_DMRS_GRID_FREQ_SHIFT)
 #endif    
     tensor_ref<const TComplexDataRx> tDataRx        (drvdUeGrpPrms.tInfoDataRx.pAddr  , drvdUeGrpPrms.tInfoDataRx.strides);// (NF, ND, N_BS_ANTS)
     tensor_ref<TComplexStorage>      tHEst          (drvdUeGrpPrms.tInfoHEst.pAddr    , drvdUeGrpPrms.tInfoHEst.strides); 
     tensor_ref<TComplexStorage>      tDbg           (drvdUeGrpPrms.tInfoChEstDbg.pAddr, drvdUeGrpPrms.tInfoChEstDbg.strides); 
     // clang-format on

#ifdef ENABLE_DEBUG
     if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.x) && (0 == blockIdx.y))
     printf("%s\n: NH_IDX %d hetCfgUeGrpIdx %d ueGrpIdx %d nPrb %d startPb %d pDmrsSymPos[0] %d pDmrsSymPos[1] %d dmrsScId %d\n", __PRETTY_FUNCTION__, dynDescr.chEstTimeInst, blockIdx.z, UE_GRP_IDX, nPrb, startPrb, pDmrsSymPos[0], pDmrsSymPos[1], dmrsScId);
#endif

     //--------------------------------------------------------------------------------------------------------
     // Dimensions and indices
 
     // Estimates of H in time supported
     const uint32_t NH_IDX = chEstTimeInst;
 
     // Channel estimation expands tones in a DMRS grid (4 or 6, given by N_DMRS_GRID_TONES_PER_PRB) into a full PRB
     constexpr uint32_t N_DMRS_INTERP_TONES_PER_GRID = N_TONES_PER_PRB;
 
     // # of tones per DMRS grid in a PRB
     constexpr uint32_t N_DMRS_GRID_TONES_PER_PRB = N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB;
     // Max permissible DMRS grids within a PRB based on spec
     constexpr uint32_t N_DMRS_TYPE1_GRIDS_PER_PRB = 2;
     constexpr uint32_t N_DMRS_TYPE2_GRIDS_PER_PRB = 3;
     static_assert(((N_DMRS_TYPE1_GRIDS_PER_PRB == N_DMRS_GRIDS_PER_PRB) || (N_DMRS_TYPE2_GRIDS_PER_PRB == N_DMRS_GRIDS_PER_PRB)),
                   "DMRS grid count exceeds max value");
 
     // Within a PRB, successive DMRS grids are shifted by 2 tones
     constexpr uint32_t N_INTER_DMRS_GRID_FREQ_SHIFT = get_inter_dmrs_grid_freq_shift(N_DMRS_GRIDS_PER_PRB);
 
     // Per grid tone counts present in input and output PRB clusters. These tones counts are expected to be equal
     constexpr uint32_t N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER        = N_DMRS_GRID_TONES_PER_PRB * N_DMRS_PRB_IN_PER_CLUSTER;
     constexpr uint32_t N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER = N_DMRS_INTERP_TONES_PER_GRID * N_DMRS_INTERP_PRB_OUT_PER_CLUSTER;
 
     // Total # of DMRS tones consumed by this thread block (this number should equal number of threads in
     // thread block since each DMRS tone is processed by a thread)
     constexpr uint32_t N_DMRS_TONES = N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER * N_DMRS_GRIDS_PER_PRB; // blockDim.x
     // Total # of interpolated DMRS tones produced by this thread block (this number should also equal number
     // of threads in thread block)
     constexpr uint32_t N_INTERP_TONES = N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER * N_DMRS_GRIDS_PER_PRB; // blockDim.x
 
     static_assert((N_DMRS_PRB_IN_PER_CLUSTER * N_TONES_PER_PRB == N_DMRS_TONES),
                   "Mismatch in expected vs calcualted DMRS tone count");
     static_assert((N_DMRS_TONES == N_INTERP_TONES),
                   "Thread allocation assumes input DMRS tone count and interpolated tone count are equal, ensure sufficient threads are allocated for interpoloation etc");
 
     // Ensure configured symbol count does not exceed max value prescribed by spec
     static_assert((N_DMRS_SYMS <= N_MAX_DMRS_SYMS), "DMRS symbol count exceeds max value");

     // Interpolation filter indices for middle and edge PRBs
     constexpr uint32_t MIDDLE_INTERP_FILT_IDX     = 0;
     constexpr uint32_t LOWER_EDGE_INTERP_FILT_IDX = 1;
     constexpr uint32_t UPPER_EDGE_INTERP_FILT_IDX = 2;
 
     // DMRS descrambling
     constexpr uint32_t N_DMRS_DESCR_BITS_PER_TONE    = 2; // 1bit for I and 1 bit for Q
     constexpr uint32_t N_DMRS_DESCR_BITS_PER_CLUSTER = N_DMRS_DESCR_BITS_PER_TONE * N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
     // # of DMRS descrambler bits generated at one time
     constexpr uint32_t N_DMRS_DESCR_BITS_GEN = 32;
     // Round up to the next multiple of N_DMRS_DESCR_BITS_GEN plus 1 (+1 because DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET
     // may be large enough to spill the descrambler bits to the next word)
     constexpr uint32_t N_DMRS_DESCR_WORDS =
         ((N_DMRS_DESCR_BITS_PER_CLUSTER + N_DMRS_DESCR_BITS_GEN - 1) / N_DMRS_DESCR_BITS_GEN) + 1;
     // round_up_to_next(N_DMRS_DESCR_BITS_PER_CLUSTER, N_DMRS_DESCR_BITS_GEN) + 1;
 
     // number of "edge" tones, not estimated but used to extract additional dmrs 
     constexpr uint32_t HALF_N_EDGE_TONES = N_TONES_PER_PRB * (N_DMRS_PRB_IN_PER_CLUSTER - N_DMRS_INTERP_PRB_OUT_PER_CLUSTER) / 2;   
 
     const uint32_t ACTIVE_DMRS_GRID_BMSK = 0x3;
 
     // Total number of PRB clusers to be processed (N_PRB_CLUSTERS*N_DMRS_INTERP_PRB_OUT_PER_CLUSTER = N_DATA_PRB)
     const uint32_t N_PRB_CLUSTERS = N_PRB_CLUSTERS_PER_BS_ANT;
 
     // Per UE group descrambling ID
     uint16_t dmrsScramId = dmrsScId;

#ifdef ENABLE_DEBUG
 
     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     {
         printf("Addr: tFreqInterpCoefs %lp tShiftSeq %lp tUnShiftSeq %lp tDataRx %lp tHEst %lp \n", tFreqInterpCoefs.pAddr, tShiftSeq.pAddr, tUnShiftSeq.pAddr, tDataRx.pAddr, tHEst.pAddr);
 
         printf("tFreqInterpCoefs: addr %lp strides[0] %d strides[1] %d strides[2] %d\n", static_cast<const TCompute*>(tFreqInterpCoefs.pAddr) , tFreqInterpCoefs.strides[0], tFreqInterpCoefs.strides[1], tFreqInterpCoefs.strides[2]);
         printf("tShiftSeq       : addr %lp strides[0] %d strides[1] %d strides[2] %d\n", static_cast<const TComplexDataRx*>(tShiftSeq.pAddr)  , tShiftSeq.strides[0]       , tShiftSeq.strides[1]       , tShiftSeq.strides[2]       );
         printf("tUnShiftSeq     : addr %lp strides[0] %d strides[1] %d strides[2] %d\n", static_cast<const TComplexDataRx*>(tUnShiftSeq.pAddr), tUnShiftSeq.strides[0]     , tUnShiftSeq.strides[1]     , tUnShiftSeq.strides[2]     );
 
         printf("startPrb       : %d \n", startPrb);
         printf("dmrsScId       : %d\n", dmrsScId);
         printf("tDataRx         : addr %lp strides[0] %d strides[1] %d strides[2] %d\n", static_cast<const TComplexDataRx*>(tDataRx.pAddr)    , tDataRx.strides[0]         , tDataRx.strides[1]         , tDataRx.strides[2]         );
         printf("tHEst           : addr %lp strides[0] %d strides[1] %d strides[2] %d\n", static_cast<TComplexStorage*>(tHEst.pAddr)     , tHEst.strides[0]                 , tHEst.strides[1]           , tHEst.strides[2]           );
         // printf("tDbg    strides[0] %d strides[1] %d strides[2] %d\n", tDbg.strides[0], tDbg.strides[1], tDbg.strides[2]);
 
         printf("dmrsScramId %d slotNum %d activeDmrsGridBmsk 0x%08x chEstTimeInst %d\n", dmrsScramId, slotNum, activeDmrsGridBmsk, NH_IDX);
     }
 
     // printf("Block(%d %d %d) Thread(%d %d %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
     // if((0 != BS_ANT_IDX) || (0 != PRB_CLUSTER_IDX)) return;
     // printf("dmrsScramId %d slotNum %d activeDmrsGridBmsk 0x%08x", dmrsScramId, slotNum, activeDmrsGridBmsk);
     // if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     //    printf("tDataRx strides[0] %d strides[1] %d strides[2] %d\n", tDataRx.strides[0], tDataRx.strides[1], tDataRx.strides[2]);
 #if 0    
     printf("InterpCoefs[%d][%d][%d] = %f ShiftSeq[%d][%d] = %f+j%f UnShiftSeq[%d] %f+j%f DataRx[%d][%d][%d]= %f+j%f\n",
            0,0,0,
            tFreqInterpCoefs(0,0,0),
            0,0,
            tShiftSeq(0,0).x,
            tShiftSeq(0,0).y,
            0,
            tUnShiftSeq(0).x,
            tUnShiftSeq(0).y,
            0,0,0,
            tDataRx(0,0,0).x,
            tDataRx(0,0,0).y);
 #endif
#endif
 
     const uint32_t THREAD_IDX = threadIdx.x;
 
     // # of PRBs for which channel must be estimated
     const uint32_t N_EDGE_PRB = (N_DMRS_PRB_IN_PER_CLUSTER - N_DMRS_INTERP_PRB_OUT_PER_CLUSTER) / 2; // Lower and Upper edge PRBs
 
     // Determine first PRB in the cluster being processed
     uint32_t prbClusterStartIdx = (PRB_CLUSTER_IDX * N_DMRS_INTERP_PRB_OUT_PER_CLUSTER) - N_EDGE_PRB;
     if(0 == PRB_CLUSTER_IDX) prbClusterStartIdx = 0;                                                             // Lower edge
     if((N_PRB_CLUSTERS - 1) == PRB_CLUSTER_IDX) prbClusterStartIdx = N_DATA_PRB - N_DMRS_PRB_IN_PER_CLUSTER;     // Upper edge
 
     uint32_t prbAbsStartIdx = prbClusterStartIdx + startPrb;
     // Absolute index of DMRS tone within the input OFDM symbol (used as index when loading tone from OFDM
     // symbol)
     const uint32_t DMRS_ABS_TONE_IDX = prbAbsStartIdx * N_TONES_PER_PRB + THREAD_IDX;
 
     // This index calculation intends to divvy up threads in the thread block for processing as follows:
     // the first group of N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER to process the first DMRS grid, the second group
     // to process the second DMRS grid and so on
     // Relative index of DMRS tone (within a DMRS grid) being processed by this thread
     const uint32_t DMRS_TONE_IDX        = THREAD_IDX % N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
     const uint32_t DMRS_INTERP_TONE_IDX = THREAD_IDX % N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER;
 
     // Index of DMRS grid in which the DMRS tone being processed by this thread resides
     // Note: although the grid index is calculated using total number of DMRS grid tones in the cluster, its
     // used as an index in context of both input DMRS tones and interpolated DMRS tones under the assumption:
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER == N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER
     const uint32_t DMRS_GRID_IDX = THREAD_IDX / N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
 
     const uint32_t tmpGridIdx = DMRS_GRID_IDX > 1 ? 1 : DMRS_GRID_IDX;
     const uint8_t  activeTOCCBmsk     = drvdUeGrpPrms.activeTOCCBmsk[tmpGridIdx];
     const uint8_t  activeFOCCBmsk     = drvdUeGrpPrms.activeFOCCBmsk[tmpGridIdx];
     const uint32_t N_DMRS_SYMS_FOCC = activeFOCCBmsk == 3 ? 2 : 1;
     // Index which enables extraction of DMRS tones of a given DMRS grid scattered within the PRB into one
     // contiguous set for processing. Note that the read from GMEM is coalesced and write into SMEM is scattered
     // @todo: check if index calculation can be simplified
     const uint32_t SMEM_DMRS_TONE_IDX = get_smem_dmrs_tone_idx(N_DMRS_GRIDS_PER_PRB, N_INTER_DMRS_GRID_FREQ_SHIFT, THREAD_IDX);
     const uint32_t SMEM_DMRS_GRID_IDX = get_smem_dmrs_grid_idx(N_DMRS_GRIDS_PER_PRB, N_INTER_DMRS_GRID_FREQ_SHIFT, THREAD_IDX);
 
     // Absolute index of descrambling + shift sequence element
     const uint32_t DMRS_DESCR_SHIFT_SEQ_START_IDX = prbAbsStartIdx * N_DMRS_GRID_TONES_PER_PRB;
     // const uint32_t DMRS_DESCR_SHIFT_SEQ_ABS_IDX   = DMRS_DESCR_SHIFT_SEQ_START_IDX + DMRS_TONE_IDX;
     const uint32_t DMRS_DESCR_SHIFT_SEQ_ABS_IDX = DMRS_TONE_IDX;
 
     // Select one of 3 interpolation filters for middle section, lower and upper edges of the frequency band
     uint32_t filtIdx = MIDDLE_INTERP_FILT_IDX;                                        // All tones in between lower and upper edges
     if(0 == PRB_CLUSTER_IDX) filtIdx = LOWER_EDGE_INTERP_FILT_IDX;                    // Lower edge
     if((N_PRB_CLUSTERS - 1) == PRB_CLUSTER_IDX) filtIdx = UPPER_EDGE_INTERP_FILT_IDX; // Upper edge
 
     // Absolute index of interpolated tone produced by this thread
     const uint32_t INTERP_PRB_CLUSTER_IDX   = blockIdx.x;
     uint32_t INTERP_DMRS_ABS_TONE_IDX = INTERP_PRB_CLUSTER_IDX * N_DMRS_INTERP_PRB_OUT_PER_CLUSTER * N_TONES_PER_PRB + DMRS_INTERP_TONE_IDX;
 
     if((N_DMRS_PRB_IN_PER_CLUSTER == 4) && (nPrbsMod2 == 1) && (filtIdx == UPPER_EDGE_INTERP_FILT_IDX))
         INTERP_DMRS_ABS_TONE_IDX = INTERP_DMRS_ABS_TONE_IDX - N_TONES_PER_PRB;
 
     // Select the shift in interpolation filter coefficients and delay shift based on grid index
     // (for e.g. for 2 DMRS grids and 48 tones per grid, multiply DMRS tone vector with top 48 rows for
     // DMRS_GRID_IDX 0 and bottom 48 rows for DMRS_GRID_IDX 1 to acheieve the effect of shift)
     uint32_t gridShiftIdx = get_inter_dmrs_grid_freq_shift_idx(N_DMRS_GRIDS_PER_PRB, DMRS_GRID_IDX);
 
     // Section 5.2.1 in 3GPP TS 38.211
     // The fast-forward of 1600 prescribed by spec is already baked into the gold sequence generator
     constexpr uint32_t DMRS_DESCR_FF = 0; // 1600;
 
     // First descrambler bit index needed by this thread block
     // Note:The DMRS scrambling sequence is the same for all the DMRS grids. There are 2 sequences one for
     // scid 0 and other for scid 1 but the same sequences is reused for all DMRS grids
     const uint32_t DMRS_DESCR_PRB_CLUSTER_START_BIT =
         DMRS_DESCR_FF + (DMRS_DESCR_SHIFT_SEQ_START_IDX * N_DMRS_DESCR_BITS_PER_TONE);
 
     // The descrambling sequence generator outputs 32 descrambler bits at a time. Thus, compute the earliest
     // multiple of 32 bits which contains the descrambler bit of the first tone in the PRB cluster as the
     // start index
     const uint32_t DMRS_DESCR_GEN_ALIGNED_START_BIT =
         (DMRS_DESCR_PRB_CLUSTER_START_BIT / N_DMRS_DESCR_BITS_GEN) * N_DMRS_DESCR_BITS_GEN;
     // Offset to descrambler bit of the first tone in the PRB cluster
     const uint32_t DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET =
         DMRS_DESCR_PRB_CLUSTER_START_BIT - DMRS_DESCR_GEN_ALIGNED_START_BIT;
 
     // DMRS descrambling bits generated correspond to subcarriers across frequency
     // e.g. 2 bits for tone0(grid 0) | 2 bits for tone1(grid 1) | 2 bits for tone 2(grid 0) | 2 bits for tone 3(grid 1) | ...
     const uint32_t DMRS_TONE_DESCR_BIT_IDX = DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET +
                                              (DMRS_TONE_IDX * N_DMRS_DESCR_BITS_PER_TONE);
     const uint32_t DMRS_DESCR_SEQ_RD_BIT_IDX  = DMRS_TONE_DESCR_BIT_IDX % N_DMRS_DESCR_BITS_GEN;
     const uint32_t DMRS_DESCR_SEQ_RD_WORD_IDX = DMRS_TONE_DESCR_BIT_IDX / N_DMRS_DESCR_BITS_GEN;
 
     const uint32_t DMRS_DESCR_SEQ_WR_WORD_IDX = THREAD_IDX % N_DMRS_DESCR_WORDS;
     const uint32_t DMRS_DESCR_SEQ_WR_SYM_IDX  = THREAD_IDX / N_DMRS_DESCR_WORDS;
 
 #if 0
     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z))
        printf("N_DMRS_DESCR_BITS_PER_CLUSTER %d N_DMRS_DESCR_WORDS %d DMRS_DESCR_PRB_CLUSTER_START_BIT %d DMRS_DESCR_GEN_ALIGNED_START_BIT %d, "
               "DMRS_DESCR_SEQ_RD_WORD_IDX %d, DMRS_DESCR_SEQ_RD_BIT_IDX %d, DMRS_DESCR_SEQ_WR_WORD_IDX %d, DMRS_DESCR_SEQ_WR_SYM_IDX %d\n",
               N_DMRS_DESCR_BITS_PER_CLUSTER, N_DMRS_DESCR_WORDS, DMRS_DESCR_PRB_CLUSTER_START_BIT, DMRS_DESCR_GEN_ALIGNED_START_BIT, 
               DMRS_DESCR_SEQ_RD_WORD_IDX, DMRS_DESCR_SEQ_RD_BIT_IDX, DMRS_DESCR_SEQ_WR_WORD_IDX, DMRS_DESCR_SEQ_WR_SYM_IDX);
 #endif
 
     // Data layouts:
     // Global memory read into shared memory
     // N_DMRS_TONES x N_DMRS_SYMS -> N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS x N_DMRS_GRIDS_PER_PRB
 
     // tOCC removal
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB
 
     // fOCC removal
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB
 
     // Interpolation
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB =
     // N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER x N_LAYERS x N_DMRS_GRIDS_PER_PRB
 
     //--------------------------------------------------------------------------------------------------------
     // Allocate shared memory
 
     constexpr uint32_t N_SMEM_ELEMS1 = N_DMRS_TONES * N_DMRS_SYMS; // (N_DMRS_TONES + N_DMRS_GRIDS_PER_PRB)*N_DMRS_SYMS;
     constexpr uint32_t N_SMEM_ELEMS2 = N_INTERP_TONES * N_DMRS_SYMS_OCC;
     constexpr uint32_t N_SMEM_ELEMS  = (N_SMEM_ELEMS1 > N_SMEM_ELEMS2) ? N_SMEM_ELEMS1 : N_SMEM_ELEMS2;
     // constexpr uint32_t N_SMEM_ELEMS  = (N_SMEM_ELEMS1 + N_SMEM_ELEMS2);
     // constexpr uint32_t N_SMEM_ELEMS  = max(N_SMEM_ELEMS1, N_SMEM_ELEMS2);
 
     __shared__ TComplexCompute smemBlk[N_SMEM_ELEMS];
     // overlay1
     block_3D<TComplexCompute*, N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS, N_DMRS_GRIDS_PER_PRB> shPilots(&smemBlk[0]);
     // overlay2
     block_3D<TComplexCompute*, N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS_OCC, N_DMRS_GRIDS_PER_PRB> shH(&smemBlk[0]);
     // block_3D<TComplexCompute*, N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS_OCC, N_DMRS_GRIDS_PER_PRB> shH(&smemBlk[shPilots.num_elem()]);
     static_assert((shPilots.num_elem() <= N_SMEM_ELEMS) && (shH.num_elem() <= N_SMEM_ELEMS), "Insufficient shared memory");
 
     __shared__ uint32_t descrWords[N_DMRS_SYMS][N_DMRS_DESCR_WORDS];
 
     //--------------------------------------------------------------------------------------------------------
     // Read DMRS tones into shared memory (separate the tones into different DMRS grids during the write)
 
     // Cache shift sequence in register
     TComplexCompute shiftSeq = type_convert<TComplexCompute>(tShiftSeq(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, 0));
 
 #pragma unroll
     for(uint32_t i = 0; i < N_DMRS_SYMS; ++i)
     {
         shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX) =
             type_convert<TComplexCompute>(tDataRx(DMRS_ABS_TONE_IDX, pDmrsSymPos[i], BS_ANT_IDX));
 
 #ifdef ENABLE_DEBUG
         printf("Pilots[%d][%d][%d] -> shPilots[%d][%d][%d] = %f+j%f, ShiftSeq[%d][%d] = %f+j%f\n",
                DMRS_ABS_TONE_IDX,
                pDmrsSymPos[i],
                BS_ANT_IDX,
                SMEM_DMRS_TONE_IDX,
                i,
                SMEM_DMRS_GRID_IDX,
                shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX).x,
                shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX).y,
                DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                0,
                shiftSeq.x,
                shiftSeq.y);
 #endif
     }
 
     // Compute the descsrambler sequence
     const uint32_t TWO_POW_17 = bit(17);
 
     if(DMRS_DESCR_SEQ_WR_SYM_IDX < N_DMRS_SYMS)
     {
         uint32_t symIdx = pDmrsSymPos[DMRS_DESCR_SEQ_WR_SYM_IDX];
 
         // see 38.211 section 6.4.1.1.1.1
         uint32_t cInit = TWO_POW_17 * (slotNum * OFDM_SYMBOLS_PER_SLOT + symIdx + 1) * (2 * dmrsScramId + 1) + (2 * dmrsScramId) + scid;
         cInit &= ~bit(31);
 
         // descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX] =
         //  __brev(gold32(cInit, (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX*N_DMRS_DESCR_BITS_GEN)));
 
         descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX] =
             gold32(cInit, (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX * N_DMRS_DESCR_BITS_GEN));
 #if 0
         printf("symIdx %d, DMRS_DESCR_SEQ_WR_WORD_IDX %d, cInit 0x%08x, DMRS_DESCR_GEN_ALIGNED_START_BIT %d, descrWords[%d][%d] 0x%08x\n", 
                symIdx, DMRS_DESCR_SEQ_WR_WORD_IDX, cInit, 
                (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX*N_DMRS_DESCR_BITS_GEN), 
                DMRS_DESCR_SEQ_WR_SYM_IDX, DMRS_DESCR_SEQ_WR_WORD_IDX, descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX]);
 #endif
     }
 
     // To ensure coalesced reads, input DMRS tones are read preserving input order but swizzled while writing
     // to shared memory. Thus each thread may not process the same tone which it wrote to shared memory
     thread_block const& thisThrdBlk = this_thread_block();
     thisThrdBlk.sync();

     //--------------------------------------------------------------------------------------------------------
     // Apply de-scrambling + delay domain centering sequence for tone index processed by this thread across all
     // DMRS symbols
     const TCompute RECIPROCAL_SQRT2 = 0.7071068f;
     const TCompute SQRT2            = 1.41421356f;
 
 #pragma unroll
     for(uint32_t i = 0; i < N_DMRS_SYMS; ++i)
     {
         int8_t descrIBit = (descrWords[i][DMRS_DESCR_SEQ_RD_WORD_IDX] >> DMRS_DESCR_SEQ_RD_BIT_IDX) & 0x1;
         int8_t descrQBit = (descrWords[i][DMRS_DESCR_SEQ_RD_WORD_IDX] >> (DMRS_DESCR_SEQ_RD_BIT_IDX + 1)) & 0x1;
 
         TComplexCompute descrCode =
             cuConj(cuGet<TComplexCompute>((1 - 2 * descrIBit) * RECIPROCAL_SQRT2, (1 - 2 * descrQBit) * RECIPROCAL_SQRT2));
         TComplexCompute descrShiftSeq = shiftSeq * descrCode;
 
 #ifdef ENABLE_DEBUG
         TComplexCompute descrShiftPilot = shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) * descrShiftSeq;
         printf("descrShiftAbsIdx: %d, shPilots[%d][%d][%d] (%f+j%f) * DescrShiftSeq[%d][%d] (%f+j%f) = %f+j%f, ShiftSeq = %f+j%f, DescrCode = %f+j%f, descrIQ (%d,%d) descrWordIdx %d descrBitIdx %d\n",
                DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                DMRS_TONE_IDX,
                i,
                DMRS_GRID_IDX,
                shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).x,
                shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).y,
                DMRS_TONE_IDX,
                i,
                descrShiftSeq.x,
                descrShiftSeq.y,
                descrShiftPilot.x,
                descrShiftPilot.y,
                shiftSeq.x,
                shiftSeq.y,
                descrCode.x,
                descrCode.y,
                descrIBit,
                descrQBit,
                DMRS_DESCR_SEQ_RD_WORD_IDX,
                DMRS_DESCR_SEQ_RD_BIT_IDX);
         if((0 == DMRS_GRID_IDX) && (((0 == prbAbsStartIdx) && (DMRS_TONE_IDX < (N_EDGE_PRB * N_DMRS_GRID_TONES_PER_PRB))) || ((0 != prbAbsStartIdx) && (prbAbsStartIdx + N_DMRS_PRB_IN_PER_CLUSTER) <= N_DATA_PRB)))
         {
 #if 0
            TComplexCompute descrShiftPilot = shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) * descrShiftSeq;
            printf("descrShiftAbsIdx: %d shPilots[%d][%d][%d] (%f+j%f) * DescrShiftSeq[%d][%d] (%f+j%f) = %f+j%f, ShiftSeq = %f+j%f, DescrCode = %f+j%f, descrIQ (%d,%d) descrWordIdx %d descrBitIdx %d\n",
                  DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                  DMRS_TONE_IDX,
                  i,
                  DMRS_GRID_IDX,
                  shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).x,
                  shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).y,
                  DMRS_TONE_IDX,
                  i,
                  descrShiftSeq.x,
                  descrShiftSeq.y,
                  descrShiftPilot.x,
                  descrShiftPilot.y,
                  shiftSeq.x,
                  shiftSeq.y,
                  descrCode.x,
                  descrCode.y,
                  descrIBit,
                  descrQBit,
                  DMRS_DESCR_SEQ_RD_WORD_IDX,
                  DMRS_DESCR_SEQ_RD_BIT_IDX);
 #endif
 
             // tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(shiftSeq);
             // tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(descrShiftSeq);
             tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(descrCode);             
         }
 #endif // ENABLE_DEBUG
 
         shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) *= descrShiftSeq;
     }
        
     //--------------------------------------------------------------------------------------------------------
     // Time domain cover code removal
     constexpr TCompute AVG_SCALE = cuGet<TCompute>(1u) / cuGet<TCompute>(N_DMRS_SYMS);
     TComplexCompute    avg[N_TOCC]{};
     
     int32_t tOCCIdx = 0;

 #pragma unroll
     for(int32_t i = 0; i < 2; ++i)
     {
        int32_t temp = (activeTOCCBmsk >> i) & 0x1;
        if (!temp) {
            continue;
        }
 #pragma unroll
         for(int32_t j = 0; j < N_DMRS_SYMS; ++j)
         {
             // For first tOCC (i = 0) output, multiply all DMRS symbols with +1 and average
             // For second tOCC (i = 1) output, multiply even DMRS symbols with +1, odd DMRS symbols with -1 and average
             int32_t sign = (-(i & j)) | 1;
             avg[tOCCIdx] += (shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX) * (sign * AVG_SCALE));
 #ifdef ENABLE_DEBUG
             TComplexCompute prod = (shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX) * (sign * AVG_SCALE));
             printf("sign*AVG_SCALE %f Pilot[%d][%d][%d] = %f+j%f avg[%d] = %f+j%f, prod = %f+j%f\n",
                    sign * AVG_SCALE,
                    DMRS_TONE_IDX,
                    j,
                    DMRS_GRID_IDX,
                    shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX).x,
                    shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX).y,
                    i,
                    avg[i].x,
                    avg[i].y,
                    prod.x,
                    prod.y);
 #endif
         }
         tOCCIdx++;
     }
 
     // shPilots and shH are overlaid in shared memory and can have different sizes (based on config). For this reason
     // ensure shPilots access from all threads is completed before writing into shH
     thisThrdBlk.sync();
 
     //--------------------------------------------------------------------------------------------------------
     // Apply frequecy domain cover code and store inplace in shared memory
     // Multiply even tones with +1 and odd tones with -1
 
     // Note that the loop termination count below is tOCC symbol count
 #pragma unroll
     for(int32_t i = 0; i < tOCCIdx; ++i)
     {
        int32_t fOCCIdx = 0;

 #pragma unroll
         for(int32_t j = 0; j < 2; ++j)
         {
            int32_t temp = (activeFOCCBmsk >> j) & 0x1;
            if (!temp) {
                continue;
            }
             // First fOCC output: multiply all tones by +1s
             // Second fOCC output: multiply even tones by +1s and odd tones by -1s
             int32_t sign                                                  = (-(DMRS_TONE_IDX & j)) | 1;
             shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + fOCCIdx, DMRS_GRID_IDX) = avg[i] * sign;
 
 #ifdef ENABLE_DEBUG
             printf("PilotsPostOCC[%d][%d][%d] = %f+j%f\n",
                    DMRS_TONE_IDX,
                    (N_DMRS_SYMS_FOCC * i) + j,
                    DMRS_GRID_IDX,
                    cuReal(shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + j, DMRS_GRID_IDX)),
                    cuImag(shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + j, DMRS_GRID_IDX)));
 #endif
             fOCCIdx++;
         }
     }
 
     // Ensure all threads complete writing results to shared memory since each thread computing an inner product
     // during interpolation stage will use results from other threads in the thread block
     thisThrdBlk.sync();
 
     //--------------------------------------------------------------------------------------------------------
     // Interpolate (matrix-vector multiply)
     for(uint32_t i = 0; i < (N_DMRS_SYMS_FOCC*tOCCIdx); ++i)
     {
         TComplexCompute innerProd{};
 
         // H = W x Y: (N_INTERP_TONES x N_DMRS_TONES) x (N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_OCC)
         // Each thread selects one row of W and computes N_DMRS_TONES length inner product to produce one interpolated
         // tone of H
         for(uint32_t j = 0; j < N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER; ++j)
         {
             TCompute interpCoef = type_convert<TCompute>(tFreqInterpCoefs(DMRS_INTERP_TONE_IDX + gridShiftIdx, j, filtIdx));
             innerProd += (shH(j, i, DMRS_GRID_IDX) * interpCoef);
         }
         // Wait for all threads to complete their inner products before updating the shared memory inplace
         // The sync is needed because shPilots and shH are overlaid
         thisThrdBlk.sync();
 
         shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX) = innerProd;
 
 #ifdef ENABLE_DEBUG
         printf("InterpPilots[%d][%d][%d] = %f+j%f\n", DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX, innerProd.x, innerProd.y);
 #endif
     }
 
     // Wait for shared memory writes to complete from all threads
     thisThrdBlk.sync();
 
     // Write channel estimates for active grids only
     const int32_t DMRS_GRID_WR_IDX = DMRS_GRID_WR_IDX_TBL[activeDmrsGridBmsk & ACTIVE_DMRS_GRID_BMSK][DMRS_GRID_IDX];
     // if(!is_set(bit(DMRS_GRID_IDX), activeDmrsGridBmsk) || (DMRS_GRID_WR_IDX < 0)) return;
     if(DMRS_GRID_WR_IDX < 0) return;
 
     //--------------------------------------------------------------------------------------------------------
     // Unshift the channel in delay back to its original location and write to GMEM. This is a scattered write
     // (@todo: any opportunities to make it coalesced?)
     // Output format is N_BS_ANT x (N_DMRS_SYMS_FOCC x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB) x N_DMRS_INTERP_PRB_OUT_PER_CLUSTER
     // where N_DMRS_SYMS_FOCC x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB = N_LAYERS
     //const uint32_t DMRS_GRID_OFFSET_H = DMRS_GRID_WR_IDX * N_DMRS_SYMS_OCC;
 
 #ifdef CH_EST_COALESCED_WRITE
     // H (N_DATA_PRB*N_TONES_PER_PRB, N_LAYERS, N_BS_ANTS, NH)
     //read the number of rx antennas
     uint32_t N_BS_ANTS  = drvdUeGrpPrms.nRxAnt;     
     TComplexStorage* pHEst = tHEst.addr + ((NH_IDX * N_BS_ANTS + BS_ANT_IDX) * N_LAYERS * N_DATA_PRB * N_TONES_PER_PRB);
 #endif
 
 
    // index of interpolated tone within cluster
    uint32_t CLUSTER_INTERP_TONE_IDX = DMRS_INTERP_TONE_IDX + HALF_N_EDGE_TONES;                                             
    if(0 == PRB_CLUSTER_IDX) CLUSTER_INTERP_TONE_IDX = CLUSTER_INTERP_TONE_IDX - HALF_N_EDGE_TONES;                        // Lower edge
    if((N_PRB_CLUSTERS - 1) == PRB_CLUSTER_IDX) CLUSTER_INTERP_TONE_IDX = CLUSTER_INTERP_TONE_IDX + HALF_N_EDGE_TONES;     // Upper edge
 
    // check if estimated tone dropped
    if(N_DMRS_PRB_IN_PER_CLUSTER == 4)
    {
        if((filtIdx == UPPER_EDGE_INTERP_FILT_IDX) && (nPrbsMod2 == 1) && (DMRS_INTERP_TONE_IDX < 12))
            return;
    }
 
 #pragma unroll
     for(uint32_t i = 0; i < N_LAYERS; ++i)
     {
         if (i < nLayers) {
            uint32_t j = OCCIdx[i] & 0x3;
            uint32_t k = (OCCIdx[i] >> 2) & 0x1;
            if (DMRS_GRID_IDX == k) {
                shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX) *=
                type_convert<TComplexCompute>(tUnShiftSeq(CLUSTER_INTERP_TONE_IDX + gridShiftIdx)); //INTERP_DMRS_ABS_TONE_IDX
                if(nDmrsCdmGrpsNoData==1)
                {
                    shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX) *= cuGet<TComplexCompute>(SQRT2, 0.0f);
                }
#ifndef CH_EST_COALESCED_WRITE
                tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX) =
                     type_convert<TComplexStorage>(shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX));

                ////Test/////
                //if (BS_ANT_IDX == 0 && i == 3 && INTERP_DMRS_ABS_TONE_IDX && !NH_IDX)
                //printf("minus itHEst = %f+j%f\n", tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x-tHEst(BS_ANT_IDX, 1, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x, tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y-tHEst(BS_ANT_IDX, 1, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y);
                /////////////

#else //fix me for flexbile DMRS port mapping
                pHEst[(DMRS_GRID_OFFSET_H + i) * N_DATA_PRB * N_TONES_PER_PRB + INTERP_DMRS_ABS_TONE_IDX] =
                     type_convert<TComplexStorage>(shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX));
#endif
            }
#ifdef ENABLE_DEBUG
#if 0 
     printf("shH[%d][%d][%d] = %f+j%f\n", DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX,
         shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX).x, shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX).y);
#endif
#if 0
     if(((6 == UE_GRP_IDX) || (7 == UE_GRP_IDX) || (8 == UE_GRP_IDX)) && (PRB_CLUSTER_IDX < 1))
     {
        TCompute hEstReal = type_convert<TCompute>(tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x);
        TCompute hEstImag = type_convert<TCompute>(tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y);
        printf("ueGrpIdx %d blockIdx.z %d tH[%d][%d][%d][%d] = %f+j%f\n", UE_GRP_IDX, blockIdx.z, BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX, hEstReal, hEstImag);
     }
      
#endif
#if 0
       printf("tUnshift[%d] = %f+j%f\n", INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx,tUnShiftSeq(INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx).x, tUnShiftSeq(INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx).y);
 
       printf("tH[%d][%d][%d][%d] = %f+j%f\n", BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX,
         tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x,
         tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y);
#endif
#endif         
        }
     }
 } //windowedChEstNoDftSOfdmKernel
 
 template <typename TStorage,
           typename TDataRx,
           typename TCompute,
           uint32_t N_LAYERS,                          // # of layers (# of cols in H matrix) ** may be larger than the actual number of layers in the group
           uint32_t N_DMRS_GRIDS_PER_PRB,              // # of DMRS grids per PRB (2 or 3)
           uint32_t N_DMRS_PRB_IN_PER_CLUSTER,         // # of PRBs bearing DMRS tones to be processed by each thread block (i.e. used in channel estimation)
           uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER, // # of PRBs bearing channel estimates (interpolated tones) at output
           uint32_t N_DMRS_SYMS>                       // # of consecutive DMRS symbols (1 or 2)
 static __global__ void
 windowedChEstKernel(puschRxChEstStatDescr_t* pStatDescr, puschRxChEstDynDescr_t* pDynDescr)
 {
     //--------------------------------------------------------------------------------------------------------     
     puschRxChEstDynDescr_t& dynDescr = *pDynDescr;

     // UE group processed by this thread block
     const uint32_t UE_GRP_IDX = dynDescr.hetCfgUeGrpMap[blockIdx.z];
     // PRB cluster being processed by this thread block
     const uint32_t PRB_CLUSTER_IDX = blockIdx.x;     
     // BS antenna being processed by this thread block
     const uint32_t BS_ANT_IDX = blockIdx.y;

     // Early exit check
     // The grid is sized to process the max # of PRB clusters in a given heterogenous config. Exit if the PRB cluster to be
     // processed by this thread block does not exist in the UE group
     cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];
     const uint16_t nPrb   = drvdUeGrpPrms.nPrb;
     const uint16_t nRxAnt = drvdUeGrpPrms.nRxAnt;
     const uint32_t N_PRB_CLUSTERS_PER_BS_ANT = div_round_up(nPrb, static_cast<uint16_t>(N_DMRS_INTERP_PRB_OUT_PER_CLUSTER));
     if((PRB_CLUSTER_IDX >= N_PRB_CLUSTERS_PER_BS_ANT) || (BS_ANT_IDX >= nRxAnt)) return;

#ifdef ENABLE_DEBUG     
     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z)) 
     printf("%s\n blockIdx.z %d UE_GRP_IDX %d blockDim (%d,%d,%d) gridDim (%d,%d,%d)\n", __PRETTY_FUNCTION__, blockIdx.z, UE_GRP_IDX, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);

     if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.y) && (0 == blockIdx.z)) 
     printf("PRB_CLUSTER_IDX %d N_PRB_CLUSTERS_PER_BS_ANT %d chEstTimeInst %d\n", PRB_CLUSTER_IDX, N_PRB_CLUSTERS_PER_BS_ANT, dynDescr.chEstTimeInst);
#endif

     //--------------------------------------------------------------------------------------------------------
     // Setup local parameters based on descriptor
     puschRxChEstStatDescr_t& statDescr    = *pStatDescr;
     const uint16_t  slotNum               = drvdUeGrpPrms.slotNum;
     const uint8_t   chEstTimeInst         = dynDescr.chEstTimeInst;
     const uint8_t   activeDmrsGridBmsk    = drvdUeGrpPrms.activeDMRSGridBmsk;
     uint8_t*        OCCIdx                = drvdUeGrpPrms.OCCIdx;
     const uint16_t  nLayers               = drvdUeGrpPrms.nLayers;
     const uint8_t   dmrsMaxLen            = drvdUeGrpPrms.dmrsMaxLen;
     const uint8_t   nPrbsMod2             = nPrb & 0x1;
     const uint32_t  N_DATA_PRB            = nPrb;
     const uint8_t   scid                  = drvdUeGrpPrms.scid;
     const uint8_t   enableTfPrcd          = drvdUeGrpPrms.enableTfPrcd;
     
     // Pointer to DMRS symbol used for channel estimation (single-symbol if maxLen = 1, double-symbol if maxLen = 2)
     uint8_t*        pDmrsSymPos   = &drvdUeGrpPrms.dmrsSymLoc[chEstTimeInst*dmrsMaxLen];
     const uint16_t  startPrb = drvdUeGrpPrms.startPrb;
     const uint16_t  dmrsScId = drvdUeGrpPrms.dmrsScrmId;
     const uint8_t   nDmrsCdmGrpsNoData = drvdUeGrpPrms.nDmrsCdmGrpsNoData;
     
     //--------------------------------------------------------------------------------------------------------
     typedef typename complex_from_scalar<TCompute>::type TComplexCompute;
     typedef typename complex_from_scalar<TDataRx>::type TComplexDataRx;
     typedef typename complex_from_scalar<TStorage>::type TComplexStorage;
 
     // clang-format off
     tensor_ref<const TCompute>       tFreqInterpCoefs((8==N_DMRS_PRB_IN_PER_CLUSTER) ? statDescr.tPrmFreqInterpCoefs.pAddr : statDescr.tPrmFreqInterpCoefs4.pAddr, 
                                                       (8==N_DMRS_PRB_IN_PER_CLUSTER) ? statDescr.tPrmFreqInterpCoefs.strides : statDescr.tPrmFreqInterpCoefs4.strides); // (N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER + N_INTER_DMRS_GRID_FREQ_SHIFT, N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER, 3), 3 filters: 1 for middle, 1 lower edge and 1 upper edge
 
   //  tensor_ref<const TStorage>       tFreqInterpCoefs(statDescr.tPrmFreqInterpCoefs.pAddr, statDescr.tPrmFreqInterpCoefs.strides);
 #if 1 // shift/unshift sequences same precision as data (FP16 or FP32)
     tensor_ref<const TComplexDataRx> tShiftSeq       (statDescr.tPrmShiftSeq.pAddr       , statDescr.tPrmShiftSeq.strides);        // (N_DATA_PRB*N_DMRS_GRID_TONES_PER_PRB, N_DMRS_SYMS)
     tensor_ref<const TComplexDataRx> tUnShiftSeq     (statDescr.tPrmUnShiftSeq.pAddr     , statDescr.tPrmUnShiftSeq.strides);      // (N_DATA_PRB*N_DMRS_INTERP_TONES_PER_GRID*N_DMRS_GRIDS_PER_PRB + N_INTER_DMRS_GRID_FREQ_SHIFT)
 #else // shift/unshift sequences same precision as channel estimates (typically FP32)
     tensor_ref<const TComplexStorage> tShiftSeq       (statDescr.tPrmShiftSeq.pAddr       , statDescr.tPrmShiftSeq.strides);        // (N_DATA_PRB*N_DMRS_GRID_TONES_PER_PRB, N_DMRS_SYMS)
     tensor_ref<const TComplexStorage> tUnShiftSeq     (statDescr.tPrmUnShiftSeq.pAddr     , statDescr.tPrmUnShiftSeq.strides);      // (N_DATA_PRB*N_DMRS_INTERP_TONES_PER_GRID*N_DMRS_GRIDS_PER_PRB + N_INTER_DMRS_GRID_FREQ_SHIFT)
 #endif    
     tensor_ref<const TComplexDataRx> tDataRx        (drvdUeGrpPrms.tInfoDataRx.pAddr  , drvdUeGrpPrms.tInfoDataRx.strides);// (NF, ND, N_BS_ANTS)
     tensor_ref<TComplexStorage>      tHEst          (drvdUeGrpPrms.tInfoHEst.pAddr    , drvdUeGrpPrms.tInfoHEst.strides); 
     tensor_ref<TComplexStorage>      tDbg           (drvdUeGrpPrms.tInfoChEstDbg.pAddr, drvdUeGrpPrms.tInfoChEstDbg.strides); 
     // clang-format on

#ifdef ENABLE_DEBUG
     if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.x) && (0 == blockIdx.y))
     printf("%s\n: NH_IDX %d hetCfgUeGrpIdx %d ueGrpIdx %d nPrb %d startPb %d pDmrsSymPos[0] %d pDmrsSymPos[1] %d dmrsScId %d\n", __PRETTY_FUNCTION__, dynDescr.chEstTimeInst, blockIdx.z, UE_GRP_IDX, nPrb, startPrb, pDmrsSymPos[0], pDmrsSymPos[1], dmrsScId);
#endif

     //--------------------------------------------------------------------------------------------------------
     // Dimensions and indices
 
     // Estimates of H in time supported
     const uint32_t NH_IDX = chEstTimeInst;
 
     // Channel estimation expands tones in a DMRS grid (4 or 6, given by N_DMRS_GRID_TONES_PER_PRB) into a full PRB
     constexpr uint32_t N_DMRS_INTERP_TONES_PER_GRID = N_TONES_PER_PRB;
 
     // # of tones per DMRS grid in a PRB
     constexpr uint32_t N_DMRS_GRID_TONES_PER_PRB = N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB;
     // Max permissible DMRS grids within a PRB based on spec
     constexpr uint32_t N_DMRS_TYPE1_GRIDS_PER_PRB = 2;
     constexpr uint32_t N_DMRS_TYPE2_GRIDS_PER_PRB = 3;
     static_assert(((N_DMRS_TYPE1_GRIDS_PER_PRB == N_DMRS_GRIDS_PER_PRB) || (N_DMRS_TYPE2_GRIDS_PER_PRB == N_DMRS_GRIDS_PER_PRB)),
                   "DMRS grid count exceeds max value");
 
     // Within a PRB, successive DMRS grids are shifted by 2 tones
     constexpr uint32_t N_INTER_DMRS_GRID_FREQ_SHIFT = get_inter_dmrs_grid_freq_shift(N_DMRS_GRIDS_PER_PRB);
 
     // Per grid tone counts present in input and output PRB clusters. These tones counts are expected to be equal
     constexpr uint32_t N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER        = N_DMRS_GRID_TONES_PER_PRB * N_DMRS_PRB_IN_PER_CLUSTER;
     constexpr uint32_t N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER = N_DMRS_INTERP_TONES_PER_GRID * N_DMRS_INTERP_PRB_OUT_PER_CLUSTER;
 
     // Total # of DMRS tones consumed by this thread block (this number should equal number of threads in
     // thread block since each DMRS tone is processed by a thread)
     constexpr uint32_t N_DMRS_TONES = N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER * N_DMRS_GRIDS_PER_PRB; // blockDim.x
     // Total # of interpolated DMRS tones produced by this thread block (this number should also equal number
     // of threads in thread block)
     constexpr uint32_t N_INTERP_TONES = N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER * N_DMRS_GRIDS_PER_PRB; // blockDim.x
 
     static_assert((N_DMRS_PRB_IN_PER_CLUSTER * N_TONES_PER_PRB == N_DMRS_TONES),
                   "Mismatch in expected vs calcualted DMRS tone count");
     static_assert((N_DMRS_TONES == N_INTERP_TONES),
                   "Thread allocation assumes input DMRS tone count and interpolated tone count are equal, ensure sufficient threads are allocated for interpoloation etc");
 
     // Ensure configured symbol count does not exceed max value prescribed by spec
     static_assert((N_DMRS_SYMS <= N_MAX_DMRS_SYMS), "DMRS symbol count exceeds max value");

     // Interpolation filter indices for middle and edge PRBs
     constexpr uint32_t MIDDLE_INTERP_FILT_IDX     = 0;
     constexpr uint32_t LOWER_EDGE_INTERP_FILT_IDX = 1;
     constexpr uint32_t UPPER_EDGE_INTERP_FILT_IDX = 2;
 
     // DMRS descrambling
     constexpr uint32_t N_DMRS_DESCR_BITS_PER_TONE    = 2; // 1bit for I and 1 bit for Q
     constexpr uint32_t N_DMRS_DESCR_BITS_PER_CLUSTER = N_DMRS_DESCR_BITS_PER_TONE * N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
     // # of DMRS descrambler bits generated at one time
     constexpr uint32_t N_DMRS_DESCR_BITS_GEN = 32;
     // Round up to the next multiple of N_DMRS_DESCR_BITS_GEN plus 1 (+1 because DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET
     // may be large enough to spill the descrambler bits to the next word)
     constexpr uint32_t N_DMRS_DESCR_WORDS =
         ((N_DMRS_DESCR_BITS_PER_CLUSTER + N_DMRS_DESCR_BITS_GEN - 1) / N_DMRS_DESCR_BITS_GEN) + 1;
     // round_up_to_next(N_DMRS_DESCR_BITS_PER_CLUSTER, N_DMRS_DESCR_BITS_GEN) + 1;
 
     // number of "edge" tones, not estimated but used to extract additional dmrs 
     constexpr uint32_t HALF_N_EDGE_TONES = N_TONES_PER_PRB * (N_DMRS_PRB_IN_PER_CLUSTER - N_DMRS_INTERP_PRB_OUT_PER_CLUSTER) / 2;   
 
     const uint32_t ACTIVE_DMRS_GRID_BMSK = 0x3;
 
     // Total number of PRB clusers to be processed (N_PRB_CLUSTERS*N_DMRS_INTERP_PRB_OUT_PER_CLUSTER = N_DATA_PRB)
     const uint32_t N_PRB_CLUSTERS = N_PRB_CLUSTERS_PER_BS_ANT;
 
     // Per UE group descrambling ID
     uint16_t dmrsScramId = dmrsScId;

#ifdef ENABLE_DEBUG
 
     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     {
         printf("Addr: tFreqInterpCoefs %lp tShiftSeq %lp tUnShiftSeq %lp tDataRx %lp tHEst %lp \n", tFreqInterpCoefs.pAddr, tShiftSeq.pAddr, tUnShiftSeq.pAddr, tDataRx.pAddr, tHEst.pAddr);
 
         printf("tFreqInterpCoefs: addr %lp strides[0] %d strides[1] %d strides[2] %d\n", static_cast<const TCompute*>(tFreqInterpCoefs.pAddr) , tFreqInterpCoefs.strides[0], tFreqInterpCoefs.strides[1], tFreqInterpCoefs.strides[2]);
         printf("tShiftSeq       : addr %lp strides[0] %d strides[1] %d strides[2] %d\n", static_cast<const TComplexDataRx*>(tShiftSeq.pAddr)  , tShiftSeq.strides[0]       , tShiftSeq.strides[1]       , tShiftSeq.strides[2]       );
         printf("tUnShiftSeq     : addr %lp strides[0] %d strides[1] %d strides[2] %d\n", static_cast<const TComplexDataRx*>(tUnShiftSeq.pAddr), tUnShiftSeq.strides[0]     , tUnShiftSeq.strides[1]     , tUnShiftSeq.strides[2]     );
 
         printf("startPrb       : %d \n", startPrb);
         printf("dmrsScId       : %d\n", dmrsScId);
         printf("tDataRx         : addr %lp strides[0] %d strides[1] %d strides[2] %d\n", static_cast<const TComplexDataRx*>(tDataRx.pAddr)    , tDataRx.strides[0]         , tDataRx.strides[1]         , tDataRx.strides[2]         );
         printf("tHEst           : addr %lp strides[0] %d strides[1] %d strides[2] %d\n", static_cast<TComplexStorage*>(tHEst.pAddr)     , tHEst.strides[0]                 , tHEst.strides[1]           , tHEst.strides[2]           );
         // printf("tDbg    strides[0] %d strides[1] %d strides[2] %d\n", tDbg.strides[0], tDbg.strides[1], tDbg.strides[2]);
 
         printf("dmrsScramId %d slotNum %d activeDmrsGridBmsk 0x%08x chEstTimeInst %d\n", dmrsScramId, slotNum, activeDmrsGridBmsk, NH_IDX);
     }
 
     // printf("Block(%d %d %d) Thread(%d %d %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
     // if((0 != BS_ANT_IDX) || (0 != PRB_CLUSTER_IDX)) return;
     // printf("dmrsScramId %d slotNum %d activeDmrsGridBmsk 0x%08x", dmrsScramId, slotNum, activeDmrsGridBmsk);
     // if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     //    printf("tDataRx strides[0] %d strides[1] %d strides[2] %d\n", tDataRx.strides[0], tDataRx.strides[1], tDataRx.strides[2]);
 #if 0    
     printf("InterpCoefs[%d][%d][%d] = %f ShiftSeq[%d][%d] = %f+j%f UnShiftSeq[%d] %f+j%f DataRx[%d][%d][%d]= %f+j%f\n",
            0,0,0,
            tFreqInterpCoefs(0,0,0),
            0,0,
            tShiftSeq(0,0).x,
            tShiftSeq(0,0).y,
            0,
            tUnShiftSeq(0).x,
            tUnShiftSeq(0).y,
            0,0,0,
            tDataRx(0,0,0).x,
            tDataRx(0,0,0).y);
 #endif
#endif
 
     const uint32_t THREAD_IDX = threadIdx.x;
 
     // # of PRBs for which channel must be estimated
     const uint32_t N_EDGE_PRB = (N_DMRS_PRB_IN_PER_CLUSTER - N_DMRS_INTERP_PRB_OUT_PER_CLUSTER) / 2; // Lower and Upper edge PRBs
 
     // Determine first PRB in the cluster being processed
     uint32_t prbClusterStartIdx = (PRB_CLUSTER_IDX * N_DMRS_INTERP_PRB_OUT_PER_CLUSTER) - N_EDGE_PRB;
     if(0 == PRB_CLUSTER_IDX) prbClusterStartIdx = 0;                                                             // Lower edge
     if((N_PRB_CLUSTERS - 1) == PRB_CLUSTER_IDX) prbClusterStartIdx = N_DATA_PRB - N_DMRS_PRB_IN_PER_CLUSTER;     // Upper edge
 
     uint32_t prbAbsStartIdx = prbClusterStartIdx + startPrb;
     // Absolute index of DMRS tone within the input OFDM symbol (used as index when loading tone from OFDM
     // symbol)
     const uint32_t DMRS_ABS_TONE_IDX = prbAbsStartIdx * N_TONES_PER_PRB + THREAD_IDX;
 
     // This index calculation intends to divvy up threads in the thread block for processing as follows:
     // the first group of N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER to process the first DMRS grid, the second group
     // to process the second DMRS grid and so on
     // Relative index of DMRS tone (within a DMRS grid) being processed by this thread
     const uint32_t DMRS_TONE_IDX        = THREAD_IDX % N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
     const uint32_t DMRS_INTERP_TONE_IDX = THREAD_IDX % N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER;
 
     // Index of DMRS grid in which the DMRS tone being processed by this thread resides
     // Note: although the grid index is calculated using total number of DMRS grid tones in the cluster, its
     // used as an index in context of both input DMRS tones and interpolated DMRS tones under the assumption:
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER == N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER
     const uint32_t DMRS_GRID_IDX = THREAD_IDX / N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
 
     const uint32_t tmpGridIdx = DMRS_GRID_IDX > 1 ? 1 : DMRS_GRID_IDX;
     const uint8_t  activeTOCCBmsk     = drvdUeGrpPrms.activeTOCCBmsk[tmpGridIdx];
     const uint8_t  activeFOCCBmsk     = drvdUeGrpPrms.activeFOCCBmsk[tmpGridIdx];
     const uint32_t N_DMRS_SYMS_FOCC = activeFOCCBmsk == 3 ? 2 : 1;
     // Index which enables extraction of DMRS tones of a given DMRS grid scattered within the PRB into one
     // contiguous set for processing. Note that the read from GMEM is coalesced and write into SMEM is scattered
     // @todo: check if index calculation can be simplified
     const uint32_t SMEM_DMRS_TONE_IDX = get_smem_dmrs_tone_idx(N_DMRS_GRIDS_PER_PRB, N_INTER_DMRS_GRID_FREQ_SHIFT, THREAD_IDX);
     const uint32_t SMEM_DMRS_GRID_IDX = get_smem_dmrs_grid_idx(N_DMRS_GRIDS_PER_PRB, N_INTER_DMRS_GRID_FREQ_SHIFT, THREAD_IDX);
 
     // Absolute index of descrambling + shift sequence element
     const uint32_t DMRS_DESCR_SHIFT_SEQ_START_IDX = prbAbsStartIdx * N_DMRS_GRID_TONES_PER_PRB;
     // const uint32_t DMRS_DESCR_SHIFT_SEQ_ABS_IDX   = DMRS_DESCR_SHIFT_SEQ_START_IDX + DMRS_TONE_IDX;
     const uint32_t DMRS_DESCR_SHIFT_SEQ_ABS_IDX = DMRS_TONE_IDX;
 
     // Select one of 3 interpolation filters for middle section, lower and upper edges of the frequency band
     uint32_t filtIdx = MIDDLE_INTERP_FILT_IDX;                                        // All tones in between lower and upper edges
     if(0 == PRB_CLUSTER_IDX) filtIdx = LOWER_EDGE_INTERP_FILT_IDX;                    // Lower edge
     if((N_PRB_CLUSTERS - 1) == PRB_CLUSTER_IDX) filtIdx = UPPER_EDGE_INTERP_FILT_IDX; // Upper edge
 
     // Absolute index of interpolated tone produced by this thread
     const uint32_t INTERP_PRB_CLUSTER_IDX   = blockIdx.x;
     uint32_t INTERP_DMRS_ABS_TONE_IDX = INTERP_PRB_CLUSTER_IDX * N_DMRS_INTERP_PRB_OUT_PER_CLUSTER * N_TONES_PER_PRB + DMRS_INTERP_TONE_IDX;
 
     if((N_DMRS_PRB_IN_PER_CLUSTER == 4) && (nPrbsMod2 == 1) && (filtIdx == UPPER_EDGE_INTERP_FILT_IDX))
         INTERP_DMRS_ABS_TONE_IDX = INTERP_DMRS_ABS_TONE_IDX - N_TONES_PER_PRB;
 
     // Select the shift in interpolation filter coefficients and delay shift based on grid index
     // (for e.g. for 2 DMRS grids and 48 tones per grid, multiply DMRS tone vector with top 48 rows for
     // DMRS_GRID_IDX 0 and bottom 48 rows for DMRS_GRID_IDX 1 to acheieve the effect of shift)
     uint32_t gridShiftIdx = get_inter_dmrs_grid_freq_shift_idx(N_DMRS_GRIDS_PER_PRB, DMRS_GRID_IDX);
 
     // Section 5.2.1 in 3GPP TS 38.211
     // The fast-forward of 1600 prescribed by spec is already baked into the gold sequence generator
     constexpr uint32_t DMRS_DESCR_FF = 0; // 1600;
 
     // First descrambler bit index needed by this thread block
     // Note:The DMRS scrambling sequence is the same for all the DMRS grids. There are 2 sequences one for
     // scid 0 and other for scid 1 but the same sequences is reused for all DMRS grids
     const uint32_t DMRS_DESCR_PRB_CLUSTER_START_BIT =
         DMRS_DESCR_FF + (DMRS_DESCR_SHIFT_SEQ_START_IDX * N_DMRS_DESCR_BITS_PER_TONE);
 
     // The descrambling sequence generator outputs 32 descrambler bits at a time. Thus, compute the earliest
     // multiple of 32 bits which contains the descrambler bit of the first tone in the PRB cluster as the
     // start index
     const uint32_t DMRS_DESCR_GEN_ALIGNED_START_BIT =
         (DMRS_DESCR_PRB_CLUSTER_START_BIT / N_DMRS_DESCR_BITS_GEN) * N_DMRS_DESCR_BITS_GEN;
     // Offset to descrambler bit of the first tone in the PRB cluster
     const uint32_t DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET =
         DMRS_DESCR_PRB_CLUSTER_START_BIT - DMRS_DESCR_GEN_ALIGNED_START_BIT;
 
     // DMRS descrambling bits generated correspond to subcarriers across frequency
     // e.g. 2 bits for tone0(grid 0) | 2 bits for tone1(grid 1) | 2 bits for tone 2(grid 0) | 2 bits for tone 3(grid 1) | ...
     const uint32_t DMRS_TONE_DESCR_BIT_IDX = DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET +
                                              (DMRS_TONE_IDX * N_DMRS_DESCR_BITS_PER_TONE);
     const uint32_t DMRS_DESCR_SEQ_RD_BIT_IDX  = DMRS_TONE_DESCR_BIT_IDX % N_DMRS_DESCR_BITS_GEN;
     const uint32_t DMRS_DESCR_SEQ_RD_WORD_IDX = DMRS_TONE_DESCR_BIT_IDX / N_DMRS_DESCR_BITS_GEN;
 
     const uint32_t DMRS_DESCR_SEQ_WR_WORD_IDX = THREAD_IDX % N_DMRS_DESCR_WORDS;
     const uint32_t DMRS_DESCR_SEQ_WR_SYM_IDX  = THREAD_IDX / N_DMRS_DESCR_WORDS;
 
 #if 0
     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z))
        printf("N_DMRS_DESCR_BITS_PER_CLUSTER %d N_DMRS_DESCR_WORDS %d DMRS_DESCR_PRB_CLUSTER_START_BIT %d DMRS_DESCR_GEN_ALIGNED_START_BIT %d, "
               "DMRS_DESCR_SEQ_RD_WORD_IDX %d, DMRS_DESCR_SEQ_RD_BIT_IDX %d, DMRS_DESCR_SEQ_WR_WORD_IDX %d, DMRS_DESCR_SEQ_WR_SYM_IDX %d\n",
               N_DMRS_DESCR_BITS_PER_CLUSTER, N_DMRS_DESCR_WORDS, DMRS_DESCR_PRB_CLUSTER_START_BIT, DMRS_DESCR_GEN_ALIGNED_START_BIT, 
               DMRS_DESCR_SEQ_RD_WORD_IDX, DMRS_DESCR_SEQ_RD_BIT_IDX, DMRS_DESCR_SEQ_WR_WORD_IDX, DMRS_DESCR_SEQ_WR_SYM_IDX);
 #endif
 
     // Data layouts:
     // Global memory read into shared memory
     // N_DMRS_TONES x N_DMRS_SYMS -> N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS x N_DMRS_GRIDS_PER_PRB
 
     // tOCC removal
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB
 
     // fOCC removal
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB
 
     // Interpolation
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB =
     // N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER x N_LAYERS x N_DMRS_GRIDS_PER_PRB
 
     //--------------------------------------------------------------------------------------------------------
     // Allocate shared memory
 
     constexpr uint32_t N_SMEM_ELEMS1 = N_DMRS_TONES * N_DMRS_SYMS; // (N_DMRS_TONES + N_DMRS_GRIDS_PER_PRB)*N_DMRS_SYMS;
     constexpr uint32_t N_SMEM_ELEMS2 = N_INTERP_TONES * N_DMRS_SYMS_OCC;
     constexpr uint32_t N_SMEM_ELEMS  = (N_SMEM_ELEMS1 > N_SMEM_ELEMS2) ? N_SMEM_ELEMS1 : N_SMEM_ELEMS2;
     // constexpr uint32_t N_SMEM_ELEMS  = (N_SMEM_ELEMS1 + N_SMEM_ELEMS2);
     // constexpr uint32_t N_SMEM_ELEMS  = max(N_SMEM_ELEMS1, N_SMEM_ELEMS2);
 
     __shared__ TComplexCompute smemBlk[N_SMEM_ELEMS];
     // overlay1
     block_3D<TComplexCompute*, N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS, N_DMRS_GRIDS_PER_PRB> shPilots(&smemBlk[0]);
     // overlay2
     block_3D<TComplexCompute*, N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS_OCC, N_DMRS_GRIDS_PER_PRB> shH(&smemBlk[0]);
     // block_3D<TComplexCompute*, N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS_OCC, N_DMRS_GRIDS_PER_PRB> shH(&smemBlk[shPilots.num_elem()]);
     static_assert((shPilots.num_elem() <= N_SMEM_ELEMS) && (shH.num_elem() <= N_SMEM_ELEMS), "Insufficient shared memory");
 
     __shared__ uint32_t descrWords[N_DMRS_SYMS][N_DMRS_DESCR_WORDS];
 
     //--------------------------------------------------------------------------------------------------------
     // Read DMRS tones into shared memory (separate the tones into different DMRS grids during the write)
 
     // Cache shift sequence in register
     TComplexCompute shiftSeq = type_convert<TComplexCompute>(tShiftSeq(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, 0));
 
 #pragma unroll
     for(uint32_t i = 0; i < N_DMRS_SYMS; ++i)
     {
         shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX) =
             type_convert<TComplexCompute>(tDataRx(DMRS_ABS_TONE_IDX, pDmrsSymPos[i], BS_ANT_IDX));
 
#ifdef ENABLE_DEBUG
         printf("Pilots[%d][%d][%d] -> shPilots[%d][%d][%d] = %f+j%f, ShiftSeq[%d][%d] = %f+j%f\n",
                DMRS_ABS_TONE_IDX,
                pDmrsSymPos[i],
                BS_ANT_IDX,
                SMEM_DMRS_TONE_IDX,
                i,
                SMEM_DMRS_GRID_IDX,
                shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX).x,
                shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX).y,
                DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                0,
                shiftSeq.x,
                shiftSeq.y);
#endif
     }
     if(enableTfPrcd==0)
     {
         // Compute the descsrambler sequence
         const uint32_t TWO_POW_17 = bit(17);
     
         if(DMRS_DESCR_SEQ_WR_SYM_IDX < N_DMRS_SYMS)
         {
             uint32_t symIdx = pDmrsSymPos[DMRS_DESCR_SEQ_WR_SYM_IDX];
     
             // see 38.211 section 6.4.1.1.1.1
             uint32_t cInit = TWO_POW_17 * (slotNum * OFDM_SYMBOLS_PER_SLOT + symIdx + 1) * (2 * dmrsScramId + 1) + (2 * dmrsScramId) + scid;
             cInit &= ~bit(31);
     
             // descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX] =
             //  __brev(gold32(cInit, (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX*N_DMRS_DESCR_BITS_GEN)));
     
             descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX] =
                 gold32(cInit, (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX * N_DMRS_DESCR_BITS_GEN));
 #if 0
             printf("symIdx %d, DMRS_DESCR_SEQ_WR_WORD_IDX %d, cInit 0x%08x, DMRS_DESCR_GEN_ALIGNED_START_BIT %d, descrWords[%d][%d] 0x%08x\n", 
                    symIdx, DMRS_DESCR_SEQ_WR_WORD_IDX, cInit, 
                    (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX*N_DMRS_DESCR_BITS_GEN), 
                    DMRS_DESCR_SEQ_WR_SYM_IDX, DMRS_DESCR_SEQ_WR_WORD_IDX, descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX]);
 #endif
         }
     }
 
     // To ensure coalesced reads, input DMRS tones are read preserving input order but swizzled while writing
     // to shared memory. Thus each thread may not process the same tone which it wrote to shared memory
     thread_block const& thisThrdBlk = this_thread_block();
     thisThrdBlk.sync();

     //--------------------------------------------------------------------------------------------------------
     // Apply de-scrambling + delay domain centering sequence for tone index processed by this thread across all
     // DMRS symbols
     const TCompute RECIPROCAL_SQRT2 = 0.7071068f;
     const TCompute SQRT2            = 1.41421356f;
 
 #pragma unroll
     for(uint32_t i = 0; i < N_DMRS_SYMS; ++i)
     {
         if(enableTfPrcd==0)
         {
             int8_t descrIBit = (descrWords[i][DMRS_DESCR_SEQ_RD_WORD_IDX] >> DMRS_DESCR_SEQ_RD_BIT_IDX) & 0x1;
             int8_t descrQBit = (descrWords[i][DMRS_DESCR_SEQ_RD_WORD_IDX] >> (DMRS_DESCR_SEQ_RD_BIT_IDX + 1)) & 0x1;
     
             TComplexCompute descrCode =
                 cuConj(cuGet<TComplexCompute>((1 - 2 * descrIBit) * RECIPROCAL_SQRT2, (1 - 2 * descrQBit) * RECIPROCAL_SQRT2));
             TComplexCompute descrShiftSeq = shiftSeq * descrCode;
     
#ifdef ENABLE_DEBUG
             TComplexCompute descrShiftPilot = shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) * descrShiftSeq;
             printf("descrShiftAbsIdx: %d, shPilots[%d][%d][%d] (%f+j%f) * DescrShiftSeq[%d][%d] (%f+j%f) = %f+j%f, ShiftSeq = %f+j%f, DescrCode = %f+j%f, descrIQ (%d,%d) descrWordIdx %d descrBitIdx %d\n",
                    DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                    DMRS_TONE_IDX,
                    i,
                    DMRS_GRID_IDX,
                    shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).x,
                    shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).y,
                    DMRS_TONE_IDX,
                    i,
                    descrShiftSeq.x,
                    descrShiftSeq.y,
                    descrShiftPilot.x,
                    descrShiftPilot.y,
                    shiftSeq.x,
                    shiftSeq.y,
                    descrCode.x,
                    descrCode.y,
                    descrIBit, 
                    descrQBit,
                    DMRS_DESCR_SEQ_RD_WORD_IDX,
                    DMRS_DESCR_SEQ_RD_BIT_IDX);}
             if((0 == DMRS_GRID_IDX) && (((0 == prbAbsStartIdx) && (DMRS_TONE_IDX < (N_EDGE_PRB * N_DMRS_GRID_TONES_PER_PRB))) || ((0 != prbAbsStartIdx) && (prbAbsStartIdx + N_DMRS_PRB_IN_PER_CLUSTER) <= N_DATA_PRB)))
             {
 #if 0
                TComplexCompute descrShiftPilot = shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) * descrShiftSeq;
                printf("descrShiftAbsIdx: %d shPilots[%d][%d][%d] (%f+j%f) * DescrShiftSeq[%d][%d] (%f+j%f) = %f+j%f, ShiftSeq = %f+j%f, DescrCode = %f+j%f, descrIQ (%d,%d) descrWordIdx %d descrBitIdx %d\n",
                      DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                      DMRS_TONE_IDX,
                      i,
                      DMRS_GRID_IDX,
                      shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).x,
                      shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).y,
                      DMRS_TONE_IDX,
                      i,
                      descrShiftSeq.x,
                      descrShiftSeq.y,
                      descrShiftPilot.x,
                      descrShiftPilot.y,
                      shiftSeq.x,
                      shiftSeq.y,
                      descrCode.x,
                      descrCode.y,
                      descrIBit,
                      descrQBit,
                      DMRS_DESCR_SEQ_RD_WORD_IDX,
                      DMRS_DESCR_SEQ_RD_BIT_IDX);
#endif
     
                 // tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(shiftSeq);
                 // tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(descrShiftSeq);
                 tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(descrCode);             
             }
#endif // ENABLE_DEBUG
             shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) *= descrShiftSeq;
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
                           uint32_t idxSeq = 8 * (slotNum * N_symb_slot + pDmrsSymPos[i]) + m;
                           f_gh = f_gh + ((gold32(cInit, idxSeq) >> (idxSeq % 32)) & 0x1) * (1 << m);
                       }
                       f_gh = f_gh % 30;
    //                   if((blockIdx.x==0)&&(blockIdx.y==0)&&(threadIdx.x==0)&&(threadIdx.y==0))
    //                   {
    //                       printf("f_gh[%d]\n", f_gh);
    //                   }
                   }
                   else if(groupOrSequenceHopping==2)
                   {
                       if(M_ZC > 6 * N_TONES_PER_PRB)
                       {
                           uint32_t idxSeq = slotNum * N_symb_slot + pDmrsSymPos[i];
                           v = (gold32(puschIdentity, idxSeq) >> (idxSeq % 32)) & 0x1;
                           
    //                       if((blockIdx.x==0)&&(blockIdx.y==0)&&(threadIdx.x==0)&&(threadIdx.y==0))
    //                       {
    //                           printf("idxSeq[%d]v[%d]\n", idxSeq, v);
    //                       }
                       }
                   }
                   
                   u = (f_gh + puschIdentity)%30;
               }
               uint16_t rIdx = prbClusterStartIdx * N_DMRS_GRID_TONES_PER_PRB + DMRS_TONE_IDX;
#ifdef ENALBE_COMMON_DFTSOFDM_DESCRCODE_SUBROUTINE
               float2 descrCode = gen_pusch_dftsofdm_descrcode(M_ZC, rIdx, u, v, nPrb, d_phi_6[u][rIdx], d_phi_12[u][rIdx], d_phi_18[u][rIdx], d_phi_24[u][rIdx], d_primeNums);
#else
               float2 descrCode = gen_pusch_dftsofdm_descrcode(M_ZC, rIdx, u, v, nPrb);
#endif
               TComplexCompute descrShiftSeq = shiftSeq * cuConj(cuGet<TComplexCompute>(descrCode.x, descrCode.y));
               shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) *= descrShiftSeq;     
         }  
     } // for(uint32_t i = 0; i < N_DMRS_SYMS; ++i)
        
     //--------------------------------------------------------------------------------------------------------
     // Time domain cover code removal
     constexpr TCompute AVG_SCALE = cuGet<TCompute>(1u) / cuGet<TCompute>(N_DMRS_SYMS);
     TComplexCompute    avg[N_TOCC]{};
     
     int32_t tOCCIdx = 0;

 #pragma unroll
     for(int32_t i = 0; i < 2; ++i)
     {
        int32_t temp = (activeTOCCBmsk >> i) & 0x1;
        if (!temp) {
            continue;
        }
 #pragma unroll
         for(int32_t j = 0; j < N_DMRS_SYMS; ++j)
         {
             // For first tOCC (i = 0) output, multiply all DMRS symbols with +1 and average
             // For second tOCC (i = 1) output, multiply even DMRS symbols with +1, odd DMRS symbols with -1 and average
             int32_t sign = (-(i & j)) | 1;
             avg[tOCCIdx] += (shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX) * (sign * AVG_SCALE));
 #ifdef ENABLE_DEBUG
             TComplexCompute prod = (shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX) * (sign * AVG_SCALE));
             printf("sign*AVG_SCALE %f Pilot[%d][%d][%d] = %f+j%f avg[%d] = %f+j%f, prod = %f+j%f\n",
                    sign * AVG_SCALE,
                    DMRS_TONE_IDX,
                    j,
                    DMRS_GRID_IDX,
                    shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX).x,
                    shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX).y,
                    i,
                    avg[i].x,
                    avg[i].y,
                    prod.x,
                    prod.y);
 #endif
         }
         tOCCIdx++;
     }
 
     // shPilots and shH are overlaid in shared memory and can have different sizes (based on config). For this reason
     // ensure shPilots access from all threads is completed before writing into shH
     thisThrdBlk.sync();
 
     //--------------------------------------------------------------------------------------------------------
     // Apply frequecy domain cover code and store inplace in shared memory
     // Multiply even tones with +1 and odd tones with -1
 
     // Note that the loop termination count below is tOCC symbol count
 #pragma unroll
     for(int32_t i = 0; i < tOCCIdx; ++i)
     {
        int32_t fOCCIdx = 0;

 #pragma unroll
         for(int32_t j = 0; j < 2; ++j)
         {
            int32_t temp = (activeFOCCBmsk >> j) & 0x1;
            if (!temp) {
                continue;
            }
             // First fOCC output: multiply all tones by +1s
             // Second fOCC output: multiply even tones by +1s and odd tones by -1s
             int32_t sign                                                  = (-(DMRS_TONE_IDX & j)) | 1;
             shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + fOCCIdx, DMRS_GRID_IDX) = avg[i] * sign;
 
 #ifdef ENABLE_DEBUG
             printf("PilotsPostOCC[%d][%d][%d] = %f+j%f\n",
                    DMRS_TONE_IDX,
                    (N_DMRS_SYMS_FOCC * i) + j,
                    DMRS_GRID_IDX,
                    cuReal(shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + j, DMRS_GRID_IDX)),
                    cuImag(shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + j, DMRS_GRID_IDX)));
 #endif
             fOCCIdx++;
         }
     }
 
     // Ensure all threads complete writing results to shared memory since each thread computing an inner product
     // during interpolation stage will use results from other threads in the thread block
     thisThrdBlk.sync();
 
     //--------------------------------------------------------------------------------------------------------
     // Interpolate (matrix-vector multiply)
     for(uint32_t i = 0; i < (N_DMRS_SYMS_FOCC*tOCCIdx); ++i)
     {
         TComplexCompute innerProd{};
 
         // H = W x Y: (N_INTERP_TONES x N_DMRS_TONES) x (N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_OCC)
         // Each thread selects one row of W and computes N_DMRS_TONES length inner product to produce one interpolated
         // tone of H
         for(uint32_t j = 0; j < N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER; ++j)
         {
             TCompute interpCoef = type_convert<TCompute>(tFreqInterpCoefs(DMRS_INTERP_TONE_IDX + gridShiftIdx, j, filtIdx));
             innerProd += (shH(j, i, DMRS_GRID_IDX) * interpCoef);
         }
         // Wait for all threads to complete their inner products before updating the shared memory inplace
         // The sync is needed because shPilots and shH are overlaid
         thisThrdBlk.sync();
 
         shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX) = innerProd;
 
 #ifdef ENABLE_DEBUG
         printf("InterpPilots[%d][%d][%d] = %f+j%f\n", DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX, innerProd.x, innerProd.y);
 #endif
     }
 
     // Wait for shared memory writes to complete from all threads
     thisThrdBlk.sync();
 
     // Write channel estimates for active grids only
     const int32_t DMRS_GRID_WR_IDX = DMRS_GRID_WR_IDX_TBL[activeDmrsGridBmsk & ACTIVE_DMRS_GRID_BMSK][DMRS_GRID_IDX];
     // if(!is_set(bit(DMRS_GRID_IDX), activeDmrsGridBmsk) || (DMRS_GRID_WR_IDX < 0)) return;
     if(DMRS_GRID_WR_IDX < 0) return;
 
     //--------------------------------------------------------------------------------------------------------
     // Unshift the channel in delay back to its original location and write to GMEM. This is a scattered write
     // (@todo: any opportunities to make it coalesced?)
     // Output format is N_BS_ANT x (N_DMRS_SYMS_FOCC x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB) x N_DMRS_INTERP_PRB_OUT_PER_CLUSTER
     // where N_DMRS_SYMS_FOCC x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB = N_LAYERS
     //const uint32_t DMRS_GRID_OFFSET_H = DMRS_GRID_WR_IDX * N_DMRS_SYMS_OCC;
 
 #ifdef CH_EST_COALESCED_WRITE
     // H (N_DATA_PRB*N_TONES_PER_PRB, N_LAYERS, N_BS_ANTS, NH)
     //read the number of rx antennas
     uint32_t N_BS_ANTS  = drvdUeGrpPrms.nRxAnt;     
     TComplexStorage* pHEst = tHEst.addr + ((NH_IDX * N_BS_ANTS + BS_ANT_IDX) * N_LAYERS * N_DATA_PRB * N_TONES_PER_PRB);
 #endif
 
 
    // index of interpolated tone within cluster
    uint32_t CLUSTER_INTERP_TONE_IDX = DMRS_INTERP_TONE_IDX + HALF_N_EDGE_TONES;                                             
    if(0 == PRB_CLUSTER_IDX) CLUSTER_INTERP_TONE_IDX = CLUSTER_INTERP_TONE_IDX - HALF_N_EDGE_TONES;                        // Lower edge
    if((N_PRB_CLUSTERS - 1) == PRB_CLUSTER_IDX) CLUSTER_INTERP_TONE_IDX = CLUSTER_INTERP_TONE_IDX + HALF_N_EDGE_TONES;     // Upper edge
 
    // check if estimated tone dropped
    if(N_DMRS_PRB_IN_PER_CLUSTER == 4)
    {
        if((filtIdx == UPPER_EDGE_INTERP_FILT_IDX) && (nPrbsMod2 == 1) && (DMRS_INTERP_TONE_IDX < 12))
            return;
    }
 
 #pragma unroll
     for(uint32_t i = 0; i < N_LAYERS; ++i)
     {
         if (i < nLayers) {
            uint32_t j = OCCIdx[i] & 0x3;
            uint32_t k = (OCCIdx[i] >> 2) & 0x1;
            if (DMRS_GRID_IDX == k) {
                shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX) *=
                type_convert<TComplexCompute>(tUnShiftSeq(CLUSTER_INTERP_TONE_IDX + gridShiftIdx)); //INTERP_DMRS_ABS_TONE_IDX
                if(nDmrsCdmGrpsNoData==1)
                {
                    shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX) *= cuGet<TComplexCompute>(SQRT2, 0.0f);
                }
#ifndef CH_EST_COALESCED_WRITE
                tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX) =
                     type_convert<TComplexStorage>(shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX));

                ////Test/////
                //if (BS_ANT_IDX == 0 && i == 3 && INTERP_DMRS_ABS_TONE_IDX && !NH_IDX)
                //printf("minus itHEst = %f+j%f\n", tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x-tHEst(BS_ANT_IDX, 1, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x, tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y-tHEst(BS_ANT_IDX, 1, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y);
                /////////////

#else //fix me for flexbile DMRS port mapping
                pHEst[(DMRS_GRID_OFFSET_H + i) * N_DATA_PRB * N_TONES_PER_PRB + INTERP_DMRS_ABS_TONE_IDX] =
                     type_convert<TComplexStorage>(shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX));
#endif
            }
#ifdef ENABLE_DEBUG
#if 0 
     printf("shH[%d][%d][%d] = %f+j%f\n", DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX,
         shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX).x, shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX).y);
#endif
#if 0
     if(((6 == UE_GRP_IDX) || (7 == UE_GRP_IDX) || (8 == UE_GRP_IDX)) && (PRB_CLUSTER_IDX < 1))
     {
        TCompute hEstReal = type_convert<TCompute>(tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x);
        TCompute hEstImag = type_convert<TCompute>(tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y);
        printf("ueGrpIdx %d blockIdx.z %d tH[%d][%d][%d][%d] = %f+j%f\n", UE_GRP_IDX, blockIdx.z, BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX, hEstReal, hEstImag);
     }
      
#endif
#if 0
       printf("tUnshift[%d] = %f+j%f\n", INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx,tUnShiftSeq(INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx).x, tUnShiftSeq(INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx).y);
 
       printf("tH[%d][%d][%d][%d] = %f+j%f\n", BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX,
         tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x,
         tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y);
#endif
#endif         
        }
     }
 }
 #endif
 
 
 template <typename TStorage,
           typename TDataRx,
           typename TCompute,
           uint32_t N_LAYERS,                          // # of layers (# of cols in H matrix) ** may be larger than the actual number of layers in the group
           uint32_t N_PRBS,
           uint32_t N_DMRS_GRIDS_PER_PRB,              // # of DMRS grids per PRB (2 or 3)
           uint32_t N_DMRS_SYMS>                       // # of time domain DMRS symbols (1,2 or 4)
 static __global__ void
 smallChEstKernel(puschRxChEstStatDescr_t* pStatDescr, puschRxChEstDynDescr_t* pDynDescr)
 {
     //--------------------------------------------------------------------------------------------------------
     // Setup local parameters based on descriptor
     puschRxChEstStatDescr_t& statDescr = *pStatDescr;
     // UE group processed by this thread block
     puschRxChEstDynDescr_t& dynDescr   = *pDynDescr;
     const uint32_t  UE_GRP_IDX         = dynDescr.hetCfgUeGrpMap[blockIdx.z];
     
     // BS antenna being processed by this thread block
     const uint32_t BS_ANT_IDX = blockIdx.y;
     
     cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];
     const uint16_t nRxAnt = drvdUeGrpPrms.nRxAnt;
     if(BS_ANT_IDX >= nRxAnt) return;

     const uint16_t  slotNum            = drvdUeGrpPrms.slotNum;
     const uint8_t   chEstTimeInst      = dynDescr.chEstTimeInst;
     const uint8_t   activeDmrsGridBmsk = drvdUeGrpPrms.activeDMRSGridBmsk;
     uint8_t*        OCCIdx             = drvdUeGrpPrms.OCCIdx;
     uint16_t        nLayers            = drvdUeGrpPrms.nLayers; 
     const uint8_t   dmrsMaxLen         = drvdUeGrpPrms.dmrsMaxLen;
     const uint16_t  startPrb           = drvdUeGrpPrms.startPrb;
     const uint16_t  dmrsScId           = drvdUeGrpPrms.dmrsScrmId;
     const uint8_t   scid               = drvdUeGrpPrms.scid;
     const uint8_t   nDmrsCdmGrpsNoData = drvdUeGrpPrms.nDmrsCdmGrpsNoData;
     const uint8_t   enableTfPrcd       = drvdUeGrpPrms.enableTfPrcd;
     
     
     // Pointer to DMRS symbol used for channel estimation (single-symbol if maxLen = 1, double-symbol if maxLen = 2)
     uint8_t*        pDmrsSymPos        = &drvdUeGrpPrms.dmrsSymLoc[chEstTimeInst*dmrsMaxLen];

     //--------------------------------------------------------------------------------------------------------
     typedef typename complex_from_scalar<TCompute>::type TComplexCompute;
     typedef typename complex_from_scalar<TDataRx>::type  TComplexDataRx;
     typedef typename complex_from_scalar<TStorage>::type TComplexStorage;
 
     // clang-format off
     tensor_ref<const TStorage>       tFreqInterpCoefs(statDescr.tPrmFreqInterpCoefsSmall.pAddr, statDescr.tPrmFreqInterpCoefsSmall.strides); // (N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER + N_INTER_DMRS_GRID_FREQ_SHIFT, N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER, 3), 3 filters: 1 for middle, 1 lower edge and 1 upper edge
 #if 1 // shift/unshift sequences same precision as data (FP16 or FP32)
     tensor_ref<const TComplexDataRx> tShiftSeq       (statDescr.tPrmShiftSeq.pAddr       , statDescr.tPrmShiftSeq.strides);        // (N_DATA_PRB*N_DMRS_GRID_TONES_PER_PRB, N_DMRS_SYMS)
     tensor_ref<const TComplexDataRx> tUnShiftSeq     (statDescr.tPrmUnShiftSeq.pAddr     , statDescr.tPrmUnShiftSeq.strides);      // (N_DATA_PRB*N_DMRS_INTERP_TONES_PER_GRID*N_DMRS_GRIDS_PER_PRB + N_INTER_DMRS_GRID_FREQ_SHIFT)
 #else // shift/unshift sequences same precision as channel estimates (typically FP32)
     tensor_ref<const TComplexStorage> tShiftSeq       (statDescr.tPrmShiftSeq.pAddr       , statDescr.tPrmShiftSeq.strides);        // (N_DATA_PRB*N_DMRS_GRID_TONES_PER_PRB, N_DMRS_SYMS)
     tensor_ref<const TComplexStorage> tUnShiftSeq     (statDescr.tPrmUnShiftSeq.pAddr     , statDescr.tPrmUnShiftSeq.strides);      // (N_DATA_PRB*N_DMRS_INTERP_TONES_PER_GRID*N_DMRS_GRIDS_PER_PRB + N_INTER_DMRS_GRID_FREQ_SHIFT)
 #endif    
     tensor_ref<const TComplexDataRx> tDataRx        (drvdUeGrpPrms.tInfoDataRx.pAddr  , drvdUeGrpPrms.tInfoDataRx.strides);// (NF, ND, N_BS_ANTS)
     tensor_ref<TComplexStorage>      tHEst          (drvdUeGrpPrms.tInfoHEst.pAddr    , drvdUeGrpPrms.tInfoHEst.strides); 
     tensor_ref<TComplexStorage>      tDbg           (drvdUeGrpPrms.tInfoChEstDbg.pAddr, drvdUeGrpPrms.tInfoChEstDbg.strides); 
     // clang-format on

#ifdef ENABLE_DEBUG
     if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.x) && (0 == blockIdx.y))
     {
        printf("%s\n: hetCfgUeGrpIdx %d ueGrpIdx %d nPrb %d startPb %d\n", __PRETTY_FUNCTION__, blockIdx.z, UE_GRP_IDX, N_PRBS, startPrb);
     }
#endif

     //--------------------------------------------------------------------------------------------------------
     // Dimensions and indices
 
     const uint32_t NH_IDX = chEstTimeInst;
 
     // Channel estimation expands tones in a DMRS grid (4 or 6, given by N_DMRS_GRID_TONES_PER_PRB) into a full PRB
     constexpr uint32_t N_DMRS_INTERP_TONES_PER_GRID = N_TONES_PER_PRB;
 
     // # of tones per DMRS grid in a PRB
     constexpr uint32_t N_DMRS_GRID_TONES_PER_PRB = N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB;
     // Max permissible DMRS grids within a PRB based on spec
     constexpr uint32_t N_DMRS_TYPE1_GRIDS_PER_PRB = 2;
     constexpr uint32_t N_DMRS_TYPE2_GRIDS_PER_PRB = 3;
     static_assert(((N_DMRS_TYPE1_GRIDS_PER_PRB == N_DMRS_GRIDS_PER_PRB) || (N_DMRS_TYPE2_GRIDS_PER_PRB == N_DMRS_GRIDS_PER_PRB)),
                   "DMRS grid count exceeds max value");
 
     // Within a PRB, successive DMRS grids are shifted by 2 tones
     constexpr uint32_t N_INTER_DMRS_GRID_FREQ_SHIFT = get_inter_dmrs_grid_freq_shift(N_DMRS_GRIDS_PER_PRB);
 
     // Per grid tone counts present in input and output PRB clusters. These tones counts are expected to be equal
     constexpr uint32_t N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER        = N_DMRS_GRID_TONES_PER_PRB * N_PRBS;
     constexpr uint32_t N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER = N_DMRS_INTERP_TONES_PER_GRID * N_PRBS;
 
     // Total # of DMRS tones consumed by this thread block (this number should equal number of threads in
     // thread block since each DMRS tone is processed by a thread)
     constexpr uint32_t N_DMRS_TONES = N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER * N_DMRS_GRIDS_PER_PRB; // blockDim.x
     // Total # of interpolated DMRS tones produced by this thread block (this number should also equal number
     // of threads in thread block)
     constexpr uint32_t N_INTERP_TONES = N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER * N_DMRS_GRIDS_PER_PRB; // blockDim.x
 
     // DMRS descrambling
     constexpr uint32_t N_DMRS_DESCR_BITS_PER_TONE    = 2; // 1bit for I and 1 bit for Q
     constexpr uint32_t N_DMRS_DESCR_BITS_PER_CLUSTER = N_DMRS_DESCR_BITS_PER_TONE * N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
     // # of DMRS descrambler bits generated at one time
     constexpr uint32_t N_DMRS_DESCR_BITS_GEN = 32;
     // Round up to the next multiple of N_DMRS_DESCR_BITS_GEN plus 1 (+1 because DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET
     // may be large enough to spill the descrambler bits to the next word)
     constexpr uint32_t N_DMRS_DESCR_WORDS =
         ((N_DMRS_DESCR_BITS_PER_CLUSTER + N_DMRS_DESCR_BITS_GEN - 1) / N_DMRS_DESCR_BITS_GEN) + 1;
     // round_up_to_next(N_DMRS_DESCR_BITS_PER_CLUSTER, N_DMRS_DESCR_BITS_GEN) + 1;
 
     const uint32_t ACTIVE_DMRS_GRID_BMSK = 0x3; 

     // Per UE group descrambling ID
     uint16_t dmrsScramId = dmrsScId;

 #ifdef ENABLE_DEBUG
 
     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     {
         printf("Addr: tFreqInterpCoefs %lp tShiftSeq %lp tUnShiftSeq %lp tDataRx %lp tHEst %lp\n", tFreqInterpCoefs.pAddr, tShiftSeq.pAddr, tUnShiftSeq.pAddr, tDataRx.pAddr, tHEst.pAddr);
 
         printf("tFreqInterpCoefs strides[0] %d strides[1] %d strides[2] %d\n", tFreqInterpCoefs.strides[0], tFreqInterpCoefs.strides[1], tFreqInterpCoefs.strides[2]);
         printf("tShiftSeq        strides[0] %d strides[1] %d\n", tShiftSeq.strides[0], tShiftSeq.strides[1]);
         printf("tUnShiftSeq      strides[0] %d strides[1] %d\n", tUnShiftSeq.strides[0], tUnShiftSeq.strides[1]);
 
         printf("tDataRx strides[0] %d strides[1] %d strides[2] %d\n", tDataRx.strides[0], tDataRx.strides[1], tDataRx.strides[2]);
         printf("tHEst   strides[0] %d strides[1] %d strides[2] %d\n", tHEst.strides[0], tHEst.strides[1], tHEst.strides[2]);
         // printf("tDbg    strides[0] %d strides[1] %d strides[2] %d\n", tDbg.strides[0], tDbg.strides[1], tDbg.strides[2]);
 
         printf("dmrsScramId %d slotNum %d activeDmrsGridBmsk 0x%08x\n", dmrsScramId, slotNum, activeDmrsGridBmsk);
     }
 
     // printf("Block(%d %d %d) Thread(%d %d %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
     // if((0 != BS_ANT_IDX) || (0 != PRB_CLUSTER_IDX)) return;
     // printf("dmrsScramId %d slotNum %d activeDmrsGridBmsk 0x%08x", dmrsScramId, slotNum, activeDmrsGridBmsk);
     // if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     //    printf("tDataRx strides[0] %d strides[1] %d strides[2] %d\n", tDataRx.strides[0], tDataRx.strides[1], tDataRx.strides[2]);
 #if 0    
     printf("InterpCoefs[%d][%d][%d] = %f ShiftSeq[%d][%d] = %f+j%f UnShiftSeq[%d] %f+j%f DataRx[%d][%d][%d]= %f+j%f\n",
            0,0,0,
            tFreqInterpCoefs(0,0,0),
            0,0,
            tShiftSeq(0,0).x,
            tShiftSeq(0,0).y,
            0,
            tUnShiftSeq(0).x,
            tUnShiftSeq(0).y,
            0,0,0,
            tDataRx(0,0,0).x,
            tDataRx(0,0,0).y);
 #endif
 #endif
 
     const uint32_t THREAD_IDX = threadIdx.x;
 
 
 
     // Determine first PRB in the cluster being processed
     uint32_t prbClusterStartIdx = 0;
 
     uint32_t prbAbsStartIdx = prbClusterStartIdx + startPrb;
     // Absolute index of DMRS tone within the input OFDM symbol (used as index when loading tone from OFDM
     // symbol)
     const uint32_t DMRS_ABS_TONE_IDX = prbAbsStartIdx * N_TONES_PER_PRB + THREAD_IDX;
 
     // This index calculation intends to divvy up threads in the thread block for processing as follows:
     // the first group of N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER to process the first DMRS grid, the second group
     // to process the second DMRS grid and so on
     // Relative index of DMRS tone (within a DMRS grid) being processed by this thread
     const uint32_t DMRS_TONE_IDX        = THREAD_IDX % N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
     const uint32_t DMRS_INTERP_TONE_IDX = THREAD_IDX % N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER;
 
     // Index of DMRS grid in which the DMRS tone being loaded by this thread resides
     const uint32_t DMRS_GRID_IDX = THREAD_IDX / N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;

     const uint32_t tempGridIdx = DMRS_GRID_IDX > 1 ? 1 : DMRS_GRID_IDX;

     uint8_t   activeTOCCBmsk     = drvdUeGrpPrms.activeTOCCBmsk[tempGridIdx];
     uint8_t   activeFOCCBmsk     = drvdUeGrpPrms.activeFOCCBmsk[tempGridIdx];
     uint32_t  N_DMRS_SYMS_TOCC   = activeTOCCBmsk == 3 ? 2 : 1;
     uint32_t  N_DMRS_SYMS_FOCC   = activeFOCCBmsk == 3 ? 2 : 1;
 
     // Index of DMRS grid which thread computes
     const uint32_t DMRS_GRID_IDX_COMPUTE = THREAD_IDX / N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER;
 
     // Index which enables extraction of DMRS tones of a given DMRS grid scattered within the PRB into one
     // contiguous set for processing. Note that the read from GMEM is coalesced and write into SMEM is scattered
     // @todo: check if index calculation can be simplified
     const uint32_t SMEM_DMRS_TONE_IDX = get_smem_dmrs_tone_idx(N_DMRS_GRIDS_PER_PRB, N_INTER_DMRS_GRID_FREQ_SHIFT, THREAD_IDX);
     const uint32_t SMEM_DMRS_GRID_IDX = get_smem_dmrs_grid_idx(N_DMRS_GRIDS_PER_PRB, N_INTER_DMRS_GRID_FREQ_SHIFT, THREAD_IDX);
 
     // Absolute index of descrambling + shift sequence element
     const uint32_t DMRS_DESCR_SHIFT_SEQ_START_IDX = prbAbsStartIdx * N_DMRS_GRID_TONES_PER_PRB;
     const uint32_t DMRS_DESCR_SHIFT_SEQ_ABS_IDX = DMRS_TONE_IDX;
 
     // Absolute index of interpolated tone produced by this thread
     const uint32_t INTERP_DMRS_ABS_TONE_IDX =  DMRS_INTERP_TONE_IDX;
 
     // Select which estimation filter to used based on size of prb cluster
     constexpr uint32_t filtIdx = N_PRBS - 1;                                        
 
 
     // Select the shift in interpolation filter coefficients and delay shift based on grid index
     // (for e.g. for 2 DMRS grids and 48 tones per grid, multiply DMRS tone vector with top 48 rows for
     // DMRS_GRID_IDX 0 and bottom 48 rows for DMRS_GRID_IDX 1 to acheieve the effect of shift)
     uint32_t gridShiftIdx = get_inter_dmrs_grid_freq_shift_idx(N_DMRS_GRIDS_PER_PRB, DMRS_GRID_IDX_COMPUTE);
 
     // Section 5.2.1 in 3GPP TS 38.211
     // The fast-forward of 1600 prescribed by spec is already baked into the gold sequence generator
     constexpr uint32_t DMRS_DESCR_FF = 0; // 1600;
 
     // First descrambler bit index needed by this thread block
     // Note:The DMRS scrambling sequence is the same for all the DMRS grids. There are 2 sequences one for
     // scid 0 and other for scid 1 but the same sequences is reused for all DMRS grids
     const uint32_t DMRS_DESCR_PRB_CLUSTER_START_BIT =
         DMRS_DESCR_FF + (DMRS_DESCR_SHIFT_SEQ_START_IDX * N_DMRS_DESCR_BITS_PER_TONE);
 
     // The descrambling sequence generator outputs 32 descrambler bits at a time. Thus, compute the earliest
     // multiple of 32 bits which contains the descrambler bit of the first tone in the PRB cluster as the
     // start index
     const uint32_t DMRS_DESCR_GEN_ALIGNED_START_BIT =
         (DMRS_DESCR_PRB_CLUSTER_START_BIT / N_DMRS_DESCR_BITS_GEN) * N_DMRS_DESCR_BITS_GEN;
     // Offset to descrambler bit of the first tone in the PRB cluster
     const uint32_t DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET =
         DMRS_DESCR_PRB_CLUSTER_START_BIT - DMRS_DESCR_GEN_ALIGNED_START_BIT;
 
     // DMRS descrambling bits generated correspond to subcarriers across frequency
     // e.g. 2 bits for tone0(grid 0) | 2 bits for tone1(grid 1) | 2 bits for tone 2(grid 0) | 2 bits for tone 3(grid 1) | ...
     const uint32_t DMRS_TONE_DESCR_BIT_IDX = DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET +
                                              (DMRS_TONE_IDX * N_DMRS_DESCR_BITS_PER_TONE);
     const uint32_t DMRS_DESCR_SEQ_RD_BIT_IDX  = DMRS_TONE_DESCR_BIT_IDX % N_DMRS_DESCR_BITS_GEN;
     const uint32_t DMRS_DESCR_SEQ_RD_WORD_IDX = DMRS_TONE_DESCR_BIT_IDX / N_DMRS_DESCR_BITS_GEN;
 
     const uint32_t DMRS_DESCR_SEQ_WR_WORD_IDX = THREAD_IDX % N_DMRS_DESCR_WORDS;
     const uint32_t DMRS_DESCR_SEQ_WR_SYM_IDX  = THREAD_IDX / N_DMRS_DESCR_WORDS;
 
     // determine if thread used to load dmrs:
     const bool loadFlag = (THREAD_IDX < N_PRBS*N_TONES_PER_PRB) ? true : false;
 
 #if 0
     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z))
        printf("N_DMRS_DESCR_BITS_PER_CLUSTER %d N_DMRS_DESCR_WORDS %d DMRS_DESCR_PRB_CLUSTER_START_BIT %d DMRS_DESCR_GEN_ALIGNED_START_BIT %d, "
               "DMRS_DESCR_SEQ_RD_WORD_IDX %d, DMRS_DESCR_SEQ_RD_BIT_IDX %d, DMRS_DESCR_SEQ_WR_WORD_IDX %d, DMRS_DESCR_SEQ_WR_SYM_IDX %d\n",
               N_DMRS_DESCR_BITS_PER_CLUSTER, N_DMRS_DESCR_WORDS, DMRS_DESCR_PRB_CLUSTER_START_BIT, DMRS_DESCR_GEN_ALIGNED_START_BIT, 
               DMRS_DESCR_SEQ_RD_WORD_IDX, DMRS_DESCR_SEQ_RD_BIT_IDX, DMRS_DESCR_SEQ_WR_WORD_IDX, DMRS_DESCR_SEQ_WR_SYM_IDX);
 #endif
 
     // Data layouts:
     // Global memory read into shared memory
     // N_DMRS_TONES x N_DMRS_SYMS -> N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS x N_DMRS_GRIDS_PER_PRB
 
     // tOCC removal
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB
 
     // fOCC removal
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB
 
     // Interpolation
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB =
     // N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER x N_LAYERS x N_DMRS_GRIDS_PER_PRB
 
     //--------------------------------------------------------------------------------------------------------
     // Allocate shared memory
 
     constexpr uint32_t N_SMEM_ELEMS1 = N_DMRS_TONES * N_DMRS_SYMS; // (N_DMRS_TONES + N_DMRS_GRIDS_PER_PRB)*N_DMRS_SYMS;
     constexpr uint32_t N_SMEM_ELEMS2 = N_INTERP_TONES * N_DMRS_SYMS_OCC;
     constexpr uint32_t N_SMEM_ELEMS  = (N_SMEM_ELEMS1 > N_SMEM_ELEMS2) ? N_SMEM_ELEMS1 : N_SMEM_ELEMS2;
     // constexpr uint32_t N_SMEM_ELEMS  = (N_SMEM_ELEMS1 + N_SMEM_ELEMS2);
     // constexpr uint32_t N_SMEM_ELEMS  = max(N_SMEM_ELEMS1, N_SMEM_ELEMS2);
 
     __shared__ TComplexCompute smemBlk[N_SMEM_ELEMS];
     // overlay1
     block_3D<TComplexCompute*, N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS, N_DMRS_GRIDS_PER_PRB> shPilots(&smemBlk[0]);
     // overlay2
     block_3D<TComplexCompute*, N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS_OCC, N_DMRS_GRIDS_PER_PRB> shH(&smemBlk[0]);
     // block_3D<TComplexCompute*, N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS_OCC, N_DMRS_GRIDS_PER_PRB> shH(&smemBlk[shPilots.num_elem()]);
     static_assert((shPilots.num_elem() <= N_SMEM_ELEMS) && (shH.num_elem() <= N_SMEM_ELEMS), "Insufficient shared memory");
 
     __shared__ uint32_t descrWords[N_DMRS_SYMS][N_DMRS_DESCR_WORDS];
 
     //--------------------------------------------------------------------------------------------------------
     // Read DMRS tones into shared memory (separate the tones into different DMRS grids during the write)
 
     // Cache shift sequence in register
     TComplexCompute shiftSeq = type_convert<TComplexCompute>(tShiftSeq(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, 0));
 
     if(loadFlag)
     {
 
     #pragma unroll
         for(uint32_t i = 0; i < N_DMRS_SYMS; ++i)
         {
             shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX) =
                 type_convert<TComplexCompute>(tDataRx(DMRS_ABS_TONE_IDX, pDmrsSymPos[i], BS_ANT_IDX));
 
     #ifdef ENABLE_DEBUG
             printf("Pilots[%d][%d][%d] -> shPilots[%d][%d][%d] = %f+j%f, ShiftSeq[%d][%d] = %f+j%f\n",
                 DMRS_ABS_TONE_IDX,
                 pDmrsSymPos[i],
                 BS_ANT_IDX,
                 SMEM_DMRS_TONE_IDX,
                 i,
                 SMEM_DMRS_GRID_IDX,
                 shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX).x,
                 shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX).y,
                 DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                 0,
                 shiftSeq.x,
                 shiftSeq.y);
     #endif
         }
 
         // Compute the descsrambler sequence
         const uint32_t TWO_POW_17 = bit(17);
 
         if(DMRS_DESCR_SEQ_WR_SYM_IDX < N_DMRS_SYMS)
         {
             uint32_t symIdx = pDmrsSymPos[DMRS_DESCR_SEQ_WR_SYM_IDX];
 
             // see 38.211 section 6.4.1.1.1.1
             uint32_t cInit = TWO_POW_17 * (slotNum * OFDM_SYMBOLS_PER_SLOT + symIdx + 1) * (2 * dmrsScramId + 1) + (2 * dmrsScramId) + scid;
             cInit &= ~bit(31);
 
             // descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX] =
             //  __brev(gold32(cInit, (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX*N_DMRS_DESCR_BITS_GEN)));
 
             descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX] =
                 gold32(cInit, (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX * N_DMRS_DESCR_BITS_GEN));
     #if 0
             printf("symIdx %d, DMRS_DESCR_SEQ_WR_WORD_IDX %d, cInit 0x%08x, DMRS_DESCR_GEN_ALIGNED_START_BIT %d, descrWords[%d][%d] 0x%08x\n", 
                 symIdx, DMRS_DESCR_SEQ_WR_WORD_IDX, cInit, 
                 (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX*N_DMRS_DESCR_BITS_GEN), 
                 DMRS_DESCR_SEQ_WR_SYM_IDX, DMRS_DESCR_SEQ_WR_WORD_IDX, descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX]);
     #endif
         }
     } 
 
     // To ensure coalesced reads, input DMRS tones are read preserving input order but swizzled while writing
     // to shared memory. Thus each thread may not process the same tone which it wrote to shared memory
     thread_block const& thisThrdBlk = this_thread_block();
     thisThrdBlk.sync();
 
     //--------------------------------------------------------------------------------------------------------
     // Apply de-scrambling + delay domain centering sequence for tone index processed by this thread across all
     // DMRS symbols
     const TCompute RECIPROCAL_SQRT2 = 0.7071068f;
     const TCompute SQRT2            = 1.41421356f;
     TComplexCompute    avg[N_TOCC]{};
     

     if(loadFlag)
     {
     #pragma unroll
         for(uint32_t i = 0; i < N_DMRS_SYMS; ++i)
         {
             if(enableTfPrcd==0)
             {
                 int8_t descrIBit = (descrWords[i][DMRS_DESCR_SEQ_RD_WORD_IDX] >> DMRS_DESCR_SEQ_RD_BIT_IDX) & 0x1;
                 int8_t descrQBit = (descrWords[i][DMRS_DESCR_SEQ_RD_WORD_IDX] >> (DMRS_DESCR_SEQ_RD_BIT_IDX + 1)) & 0x1;
     
                 TComplexCompute descrCode =
                     cuConj(cuGet<TComplexCompute>((1 - 2 * descrIBit) * RECIPROCAL_SQRT2, (1 - 2 * descrQBit) * RECIPROCAL_SQRT2));
                 TComplexCompute descrShiftSeq = shiftSeq * descrCode;
     
    #ifdef ENABLE_DEBUG             
                 TComplexCompute descrShiftPilot = shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) * descrShiftSeq;
                 printf("descrShiftAbsIdx: %d, shPilots[%d][%d][%d] (%f+j%f) * DescrShiftSeq[%d][%d] (%f+j%f) = %f+j%f, ShiftSeq = %f+j%f, DescrCode = %f+j%f, descrIQ (%d,%d) descrWordIdx %d descrBitIdx %d\n",
                     DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                     DMRS_TONE_IDX,
                     i,
                     DMRS_GRID_IDX,
                     shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).x,
                     shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).y,
                     DMRS_TONE_IDX,
                     i,
                     descrShiftSeq.x,
                     descrShiftSeq.y,
                     descrShiftPilot.x,
                     descrShiftPilot.y,
                     shiftSeq.x,
                     shiftSeq.y,
                     descrCode.x,
                     descrCode.y,
                     descrIBit,
                     descrQBit,
                     DMRS_DESCR_SEQ_RD_WORD_IDX,
                     DMRS_DESCR_SEQ_RD_BIT_IDX);
    
                 if((0 == DMRS_GRID_IDX) && (((0 == prbAbsStartIdx) && (DMRS_TONE_IDX < (N_EDGE_PRB * N_DMRS_GRID_TONES_PER_PRB))) || ((0 != prbAbsStartIdx) && (prbAbsStartIdx + N_DMRS_PRB_IN_PER_CLUSTER) <= N_DATA_PRB)))
                 {
         #if 0
                 TComplexCompute descrShiftPilot = shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) * descrShiftSeq;
                 printf("descrShiftAbsIdx: %d shPilots[%d][%d][%d] (%f+j%f) * DescrShiftSeq[%d][%d] (%f+j%f) = %f+j%f, ShiftSeq = %f+j%f, DescrCode = %f+j%f, descrIQ (%d,%d) descrWordIdx %d descrBitIdx %d\n",
                         DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                         DMRS_TONE_IDX,
                         i,
                         DMRS_GRID_IDX,
                         shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).x,
                         shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).y,
                         DMRS_TONE_IDX,
                         i,
                         descrShiftSeq.x,
                         descrShiftSeq.y,
                         descrShiftPilot.x,
                         descrShiftPilot.y,
                         shiftSeq.x,
                         shiftSeq.y,
                         descrCode.x,
                         descrCode.y,
                         descrIBit,
                         descrQBit,
                         DMRS_DESCR_SEQ_RD_WORD_IDX,
                         DMRS_DESCR_SEQ_RD_BIT_IDX);
#endif
     
                     // tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(shiftSeq);
                     // tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(descrShiftSeq);
                     tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(descrCode);
                 }
#endif // ENABLE_DEBUG
     
                 shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) *= descrShiftSeq;
             }
             else if(enableTfPrcd==1)
             {
                 uint16_t M_ZC  = N_DMRS_GRID_TONES_PER_PRB * N_PRBS; //different from DMRS_ABS_TONE_IDX
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
                             uint32_t idxSeq = 8 * (slotNum * N_symb_slot + pDmrsSymPos[i]) + m;
                             f_gh = f_gh + ((gold32(cInit, idxSeq) >> (idxSeq % 32)) & 0x1) * (1 << m);
                         }
                         f_gh = f_gh % 30;
      //                   if((blockIdx.x==0)&&(blockIdx.y==0)&&(threadIdx.x==0)&&(threadIdx.y==0))
      //                   {
      //                       printf("f_gh[%d]\n", f_gh);
      //                   }
                     }
                     else if(groupOrSequenceHopping==2)
                     {
                         if(M_ZC > 6 * N_TONES_PER_PRB)
                         {
                             uint32_t idxSeq = slotNum * N_symb_slot + pDmrsSymPos[i];
                             v = (gold32(puschIdentity, idxSeq) >> (idxSeq % 32)) & 0x1;
                             
      //                       if((blockIdx.x==0)&&(blockIdx.y==0)&&(threadIdx.x==0)&&(threadIdx.y==0))
      //                       {
      //                           printf("idxSeq[%d]v[%d]\n", idxSeq, v);
      //                       }
                         }
                     }
                     
                     u = (f_gh + puschIdentity)%30;
                 }
                 uint16_t rIdx = prbClusterStartIdx * N_DMRS_GRID_TONES_PER_PRB + DMRS_TONE_IDX;
#ifdef ENALBE_COMMON_DFTSOFDM_DESCRCODE_SUBROUTINE
                 float2 descrCode = gen_pusch_dftsofdm_descrcode(M_ZC, rIdx, u, v, N_PRBS, d_phi_6[u][rIdx], d_phi_12[u][rIdx], d_phi_18[u][rIdx], d_phi_24[u][rIdx], d_primeNums);
#else
                 float2 descrCode = gen_pusch_dftsofdm_descrcode(M_ZC, rIdx, u, v, N_PRBS);
#endif
                 TComplexCompute descrShiftSeq = shiftSeq * cuConj(cuGet<TComplexCompute>(descrCode.x, descrCode.y));
                 shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) *= descrShiftSeq;  
             }
         }
 
         //--------------------------------------------------------------------------------------------------------
         // Time domain cover code removal
         constexpr TCompute AVG_SCALE = cuGet<TCompute>(1u) / cuGet<TCompute>(N_DMRS_SYMS);
         // TComplexCompute    avg[N_DMRS_SYMS_TOCC]{};
         
         int32_t tOCCIdx = 0;
         
     #pragma unroll
         for(int32_t i = 0; i < 2; ++i)
         {
            int32_t temp = (activeTOCCBmsk >> i) & 0x1;
            if (!temp) {
                continue;
            }
     #pragma unroll
             for(int32_t j = 0; j < N_DMRS_SYMS; ++j)
             {
                 // For first tOCC (i = 0) output, multiply all DMRS symbols with +1 and average
                 // For second tOCC (i = 1) output, multiply even DMRS symbols with +1, odd DMRS symbols with -1 and average
                 int32_t sign = (-(i & j)) | 1;
                 avg[tOCCIdx] += (shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX) * (sign * AVG_SCALE));
     #ifdef ENABLE_DEBUG
                 TComplexCompute prod = (shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX) * (sign * AVG_SCALE));
                 printf("sign*AVG_SCALE %f Pilot[%d][%d][%d] = %f+j%f avg[%d] = %f+j%f, prod = %f+j%f\n",
                     sign * AVG_SCALE,
                     DMRS_TONE_IDX,
                     j,
                     DMRS_GRID_IDX,
                     shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX).x,
                     shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX).y,
                     i,
                     avg[i].x,
                     avg[i].y,
                     prod.x,
                     prod.y);
     #endif
             }
             tOCCIdx++;
         }
     }
 
     // shPilots and shH are overlaid in shared memory and can have different sizes (based on config). For this reason
     // ensure shPilots access from all threads is completed before writing into shH
     thisThrdBlk.sync();
 
     //--------------------------------------------------------------------------------------------------------
     // Apply frequecy domain cover code and store inplace in shared memory
     // Multiply even tones with +1 and odd tones with -1
 
     if(loadFlag)
     {
         // Note that the loop termination count below is tOCC symbol count
     #pragma unroll
         for(int32_t i = 0; i < N_DMRS_SYMS_TOCC; ++i)
         {
            int32_t fOCCIdx = 0;

     #pragma unroll
             for(int32_t j = 0; j < 2; ++j)
             {
                int32_t temp = (activeFOCCBmsk >> j) & 0x1;
                if (!temp) {
                    continue;
                }
                 // First fOCC output: multiply all tones by +1s
                 // Second fOCC output: multiply even tones by +1s and odd tones by -1s
                 int32_t sign                                                  = (-(DMRS_TONE_IDX & j)) | 1;
                 shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + fOCCIdx, DMRS_GRID_IDX) = avg[i] * sign;
     #ifdef ENABLE_DEBUG
                 printf("PilotsPostOCC[%d][%d][%d] = %f+j%f\n",
                     DMRS_TONE_IDX,
                     (N_DMRS_SYMS_FOCC * i) + j,
                     DMRS_GRID_IDX,
                     cuReal(shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + j, DMRS_GRID_IDX)),
                     cuImag(shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + j, DMRS_GRID_IDX)));
     #endif
                 fOCCIdx++;
             }
         }
     }
 
     // Ensure all threads complete writing results to shared memory since each thread computing an inner product
     // during interpolation stage will use results from other threads in the thread block
     thisThrdBlk.sync();
 
     //--------------------------------------------------------------------------------------------------------
     // Interpolate (matrix-vector multiply)

     // early exit for inactive grids
     const int32_t DMRS_GRID_WR_IDX = DMRS_GRID_WR_IDX_TBL[activeDmrsGridBmsk & ACTIVE_DMRS_GRID_BMSK][DMRS_GRID_IDX_COMPUTE];
     // if(!is_set(bit(DMRS_GRID_IDX), activeDmrsGridBmsk) || (DMRS_GRID_WR_IDX < 0)) return;
     if(DMRS_GRID_WR_IDX < 0) return;

     activeTOCCBmsk     = drvdUeGrpPrms.activeTOCCBmsk[DMRS_GRID_IDX_COMPUTE];
     activeFOCCBmsk     = drvdUeGrpPrms.activeFOCCBmsk[DMRS_GRID_IDX_COMPUTE];
     N_DMRS_SYMS_TOCC   = activeTOCCBmsk == 3 ? 2 : 1;
     N_DMRS_SYMS_FOCC   = activeFOCCBmsk == 3 ? 2 : 1;

     for(uint32_t i = 0; i < (N_DMRS_SYMS_FOCC*N_DMRS_SYMS_TOCC); ++i)
     {
         TComplexCompute innerProd{};
 
         // H = W x Y: (N_INTERP_TONES x N_DMRS_TONES) x (N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_OCC)
         // Each thread selects one row of W and computes N_DMRS_TONES length inner product to produce one interpolated
         // tone of H
         for(uint32_t j = 0; j < N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER; ++j)
         {
             TCompute interpCoef = type_convert<TCompute>(tFreqInterpCoefs(DMRS_INTERP_TONE_IDX + gridShiftIdx, j, filtIdx));
             innerProd += (shH(j, i, DMRS_GRID_IDX_COMPUTE) * interpCoef);
         }
         // Wait for all threads to complete their inner products before updating the shared memory inplace
         // The sync is needed because shPilots and shH are overlaid
         thisThrdBlk.sync();
 
         shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX_COMPUTE) = innerProd;
 
 #ifdef ENABLE_DEBUG
         printf("InterpPilots[%d][%d][%d] = %f+j%f\n", DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX, innerProd.x, innerProd.y);
 #endif
     }
 
     // Wait for shared memory writes to complete from all threads
     thisThrdBlk.sync();
 
     //--------------------------------------------------------------------------------------------------------
     // Unshift the channel in delay back to its original location and write to GMEM. This is a scattered write
     // (@todo: any opportunities to make it coalesced?)
     // Output format is N_BS_ANT x (N_DMRS_SYMS_FOCC x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB) x N_DMRS_INTERP_PRB_OUT_PER_CLUSTER
     // where N_DMRS_SYMS_FOCC x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB = N_LAYERS
     //const uint32_t DMRS_GRID_OFFSET_H = DMRS_GRID_WR_IDX * N_DMRS_SYMS_OCC;
 
 // #ifdef CH_EST_COALESCED_WRITE
 //     // H (N_DATA_PRB*N_TONES_PER_PRB, N_LAYERS, N_BS_ANTS, NH)
 //     // read the number of rx antennas
 //     uint32_t N_BS_ANTS = drvdUeGrpPrms.nRxAnt;
 //     TComplexStorage* pHEst = tHEst.addr + ((NH_IDX * N_BS_ANTS + BS_ANT_IDX) * N_LAYERS * N_DATA_PRB * N_TONES_PER_PRB);
 // #endif
 
 
 // index of interpolated tone within cluster
 uint32_t CLUSTER_INTERP_TONE_IDX = DMRS_INTERP_TONE_IDX;  
 
 #pragma unroll
     for(uint32_t i = 0; i < N_LAYERS; ++i)
     {
        if (i < nLayers) {
            uint32_t j = OCCIdx[i] & 0x3;
            uint32_t k = (OCCIdx[i] >> 2) & 0x1;
            if (DMRS_GRID_IDX_COMPUTE == k) {
                shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX_COMPUTE) *=
                    type_convert<TComplexCompute>(tUnShiftSeq(CLUSTER_INTERP_TONE_IDX + gridShiftIdx)); //INTERP_DMRS_ABS_TONE_IDX
                if(nDmrsCdmGrpsNoData==1)
                {
                    shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX) *= cuGet<TComplexCompute>(SQRT2, 0.0f);
                }
 
 #ifndef CH_EST_COALESCED_WRITE
     tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX) =
             type_convert<TComplexStorage>(shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX_COMPUTE));
 #else //fix me for flexbile DMRS port mapping
         pHEst[(DMRS_GRID_OFFSET_H + i) * N_DATA_PRB * N_TONES_PER_PRB + INTERP_DMRS_ABS_TONE_IDX] =
             type_convert<TComplexStorage>(shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX_COMPUTE));
 #endif
            }
 #ifdef ENABLE_DEBUG
 #if 0 
      printf("shH[%d][%d][%d] = %f+j%f\n", DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX,
          shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX).x, shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX).y);
 #endif
 #if 0     
       printf("tH[%d][%d][%d][%d] = %f+j%f\n", BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX,
          tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x,
          tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y);
 #endif
 #if 0
        printf("tUnshift[%d] = %f+j%f\n", INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx,tUnShiftSeq(INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx).x, tUnShiftSeq(INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx).y);
  
        printf("tH[%d][%d][%d][%d] = %f+j%f\n", BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX,
          tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x,
          tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y);
 #endif
 #endif
        }
     }
 } //smallChEstKernel
 // #endif
 
 template <typename TStorage,
           typename TDataRx,
           typename TCompute,
           uint32_t N_LAYERS,                          // # of layers (# of cols in H matrix) ** may be larger than the actual number of layers in the group
           uint32_t N_PRBS,
           uint32_t N_DMRS_GRIDS_PER_PRB,              // # of DMRS grids per PRB (2 or 3)
           uint32_t N_DMRS_SYMS>                       // # of time domain DMRS symbols (1,2 or 4)
 static __global__ void
 smallChEstNoDftSOfdmKernel(puschRxChEstStatDescr_t* pStatDescr, puschRxChEstDynDescr_t* pDynDescr)
 {
     //--------------------------------------------------------------------------------------------------------
     // Setup local parameters based on descriptor
     puschRxChEstStatDescr_t& statDescr = *pStatDescr;
     // UE group processed by this thread block
     puschRxChEstDynDescr_t& dynDescr   = *pDynDescr;
     const uint32_t  UE_GRP_IDX         = dynDescr.hetCfgUeGrpMap[blockIdx.z];
     
     // BS antenna being processed by this thread block
     const uint32_t BS_ANT_IDX = blockIdx.y;
     
     cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrms = dynDescr.pDrvdUeGrpPrms[UE_GRP_IDX];
     const uint16_t nRxAnt = drvdUeGrpPrms.nRxAnt;
     if(BS_ANT_IDX >= nRxAnt) return;

     const uint16_t  slotNum            = drvdUeGrpPrms.slotNum;
     const uint8_t   chEstTimeInst      = dynDescr.chEstTimeInst;
     const uint8_t   activeDmrsGridBmsk = drvdUeGrpPrms.activeDMRSGridBmsk;
     uint8_t*        OCCIdx             = drvdUeGrpPrms.OCCIdx;
     uint16_t        nLayers            = drvdUeGrpPrms.nLayers; 
     const uint8_t   dmrsMaxLen         = drvdUeGrpPrms.dmrsMaxLen;
     const uint16_t  startPrb           = drvdUeGrpPrms.startPrb;
     const uint16_t  dmrsScId           = drvdUeGrpPrms.dmrsScrmId;
     const uint8_t   scid               = drvdUeGrpPrms.scid;
     const uint8_t   nDmrsCdmGrpsNoData = drvdUeGrpPrms.nDmrsCdmGrpsNoData;
     
     
     // Pointer to DMRS symbol used for channel estimation (single-symbol if maxLen = 1, double-symbol if maxLen = 2)
     uint8_t*        pDmrsSymPos        = &drvdUeGrpPrms.dmrsSymLoc[chEstTimeInst*dmrsMaxLen];

     //--------------------------------------------------------------------------------------------------------
     typedef typename complex_from_scalar<TCompute>::type TComplexCompute;
     typedef typename complex_from_scalar<TDataRx>::type  TComplexDataRx;
     typedef typename complex_from_scalar<TStorage>::type TComplexStorage;
 
     // clang-format off
     tensor_ref<const TStorage>       tFreqInterpCoefs(statDescr.tPrmFreqInterpCoefsSmall.pAddr, statDescr.tPrmFreqInterpCoefsSmall.strides); // (N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER + N_INTER_DMRS_GRID_FREQ_SHIFT, N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER, 3), 3 filters: 1 for middle, 1 lower edge and 1 upper edge
 #if 1 // shift/unshift sequences same precision as data (FP16 or FP32)
     tensor_ref<const TComplexDataRx> tShiftSeq       (statDescr.tPrmShiftSeq.pAddr       , statDescr.tPrmShiftSeq.strides);        // (N_DATA_PRB*N_DMRS_GRID_TONES_PER_PRB, N_DMRS_SYMS)
     tensor_ref<const TComplexDataRx> tUnShiftSeq     (statDescr.tPrmUnShiftSeq.pAddr     , statDescr.tPrmUnShiftSeq.strides);      // (N_DATA_PRB*N_DMRS_INTERP_TONES_PER_GRID*N_DMRS_GRIDS_PER_PRB + N_INTER_DMRS_GRID_FREQ_SHIFT)
 #else // shift/unshift sequences same precision as channel estimates (typically FP32)
     tensor_ref<const TComplexStorage> tShiftSeq       (statDescr.tPrmShiftSeq.pAddr       , statDescr.tPrmShiftSeq.strides);        // (N_DATA_PRB*N_DMRS_GRID_TONES_PER_PRB, N_DMRS_SYMS)
     tensor_ref<const TComplexStorage> tUnShiftSeq     (statDescr.tPrmUnShiftSeq.pAddr     , statDescr.tPrmUnShiftSeq.strides);      // (N_DATA_PRB*N_DMRS_INTERP_TONES_PER_GRID*N_DMRS_GRIDS_PER_PRB + N_INTER_DMRS_GRID_FREQ_SHIFT)
 #endif    
     tensor_ref<const TComplexDataRx> tDataRx        (drvdUeGrpPrms.tInfoDataRx.pAddr  , drvdUeGrpPrms.tInfoDataRx.strides);// (NF, ND, N_BS_ANTS)
     tensor_ref<TComplexStorage>      tHEst          (drvdUeGrpPrms.tInfoHEst.pAddr    , drvdUeGrpPrms.tInfoHEst.strides); 
     tensor_ref<TComplexStorage>      tDbg           (drvdUeGrpPrms.tInfoChEstDbg.pAddr, drvdUeGrpPrms.tInfoChEstDbg.strides); 
     // clang-format on

#ifdef ENABLE_DEBUG
     if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z) && (0 == blockIdx.x) && (0 == blockIdx.y))
     {
        printf("%s\n: hetCfgUeGrpIdx %d ueGrpIdx %d nPrb %d startPb %d\n", __PRETTY_FUNCTION__, blockIdx.z, UE_GRP_IDX, N_PRBS, startPrb);
     }
#endif

     //--------------------------------------------------------------------------------------------------------
     // Dimensions and indices
 
     const uint32_t NH_IDX = chEstTimeInst;
 
     // Channel estimation expands tones in a DMRS grid (4 or 6, given by N_DMRS_GRID_TONES_PER_PRB) into a full PRB
     constexpr uint32_t N_DMRS_INTERP_TONES_PER_GRID = N_TONES_PER_PRB;
 
     // # of tones per DMRS grid in a PRB
     constexpr uint32_t N_DMRS_GRID_TONES_PER_PRB = N_TONES_PER_PRB / N_DMRS_GRIDS_PER_PRB;
     // Max permissible DMRS grids within a PRB based on spec
     constexpr uint32_t N_DMRS_TYPE1_GRIDS_PER_PRB = 2;
     constexpr uint32_t N_DMRS_TYPE2_GRIDS_PER_PRB = 3;
     static_assert(((N_DMRS_TYPE1_GRIDS_PER_PRB == N_DMRS_GRIDS_PER_PRB) || (N_DMRS_TYPE2_GRIDS_PER_PRB == N_DMRS_GRIDS_PER_PRB)),
                   "DMRS grid count exceeds max value");
 
     // Within a PRB, successive DMRS grids are shifted by 2 tones
     constexpr uint32_t N_INTER_DMRS_GRID_FREQ_SHIFT = get_inter_dmrs_grid_freq_shift(N_DMRS_GRIDS_PER_PRB);
 
     // Per grid tone counts present in input and output PRB clusters. These tones counts are expected to be equal
     constexpr uint32_t N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER        = N_DMRS_GRID_TONES_PER_PRB * N_PRBS;
     constexpr uint32_t N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER = N_DMRS_INTERP_TONES_PER_GRID * N_PRBS;
 
     // Total # of DMRS tones consumed by this thread block (this number should equal number of threads in
     // thread block since each DMRS tone is processed by a thread)
     constexpr uint32_t N_DMRS_TONES = N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER * N_DMRS_GRIDS_PER_PRB; // blockDim.x
     // Total # of interpolated DMRS tones produced by this thread block (this number should also equal number
     // of threads in thread block)
     constexpr uint32_t N_INTERP_TONES = N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER * N_DMRS_GRIDS_PER_PRB; // blockDim.x
 
     // DMRS descrambling
     constexpr uint32_t N_DMRS_DESCR_BITS_PER_TONE    = 2; // 1bit for I and 1 bit for Q
     constexpr uint32_t N_DMRS_DESCR_BITS_PER_CLUSTER = N_DMRS_DESCR_BITS_PER_TONE * N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
     // # of DMRS descrambler bits generated at one time
     constexpr uint32_t N_DMRS_DESCR_BITS_GEN = 32;
     // Round up to the next multiple of N_DMRS_DESCR_BITS_GEN plus 1 (+1 because DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET
     // may be large enough to spill the descrambler bits to the next word)
     constexpr uint32_t N_DMRS_DESCR_WORDS =
         ((N_DMRS_DESCR_BITS_PER_CLUSTER + N_DMRS_DESCR_BITS_GEN - 1) / N_DMRS_DESCR_BITS_GEN) + 1;
     // round_up_to_next(N_DMRS_DESCR_BITS_PER_CLUSTER, N_DMRS_DESCR_BITS_GEN) + 1;
 
     const uint32_t ACTIVE_DMRS_GRID_BMSK = 0x3; 

     // Per UE group descrambling ID
     uint16_t dmrsScramId = dmrsScId;

 #ifdef ENABLE_DEBUG
 
     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     {
         printf("Addr: tFreqInterpCoefs %lp tShiftSeq %lp tUnShiftSeq %lp tDataRx %lp tHEst %lp\n", tFreqInterpCoefs.pAddr, tShiftSeq.pAddr, tUnShiftSeq.pAddr, tDataRx.pAddr, tHEst.pAddr);
 
         printf("tFreqInterpCoefs strides[0] %d strides[1] %d strides[2] %d\n", tFreqInterpCoefs.strides[0], tFreqInterpCoefs.strides[1], tFreqInterpCoefs.strides[2]);
         printf("tShiftSeq        strides[0] %d strides[1] %d\n", tShiftSeq.strides[0], tShiftSeq.strides[1]);
         printf("tUnShiftSeq      strides[0] %d strides[1] %d\n", tUnShiftSeq.strides[0], tUnShiftSeq.strides[1]);
 
         printf("tDataRx strides[0] %d strides[1] %d strides[2] %d\n", tDataRx.strides[0], tDataRx.strides[1], tDataRx.strides[2]);
         printf("tHEst   strides[0] %d strides[1] %d strides[2] %d\n", tHEst.strides[0], tHEst.strides[1], tHEst.strides[2]);
         // printf("tDbg    strides[0] %d strides[1] %d strides[2] %d\n", tDbg.strides[0], tDbg.strides[1], tDbg.strides[2]);
 
         printf("dmrsScramId %d slotNum %d activeDmrsGridBmsk 0x%08x\n", dmrsScramId, slotNum, activeDmrsGridBmsk);
     }
 
     // printf("Block(%d %d %d) Thread(%d %d %d)\n", blockIdx.x, blockIdx.y, blockIdx.z, threadIdx.x, threadIdx.y, threadIdx.z);
     // if((0 != BS_ANT_IDX) || (0 != PRB_CLUSTER_IDX)) return;
     // printf("dmrsScramId %d slotNum %d activeDmrsGridBmsk 0x%08x", dmrsScramId, slotNum, activeDmrsGridBmsk);
     // if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z) && (0 == threadIdx.x) && (0 == threadIdx.y) && (0 == threadIdx.z))
     //    printf("tDataRx strides[0] %d strides[1] %d strides[2] %d\n", tDataRx.strides[0], tDataRx.strides[1], tDataRx.strides[2]);
 #if 0    
     printf("InterpCoefs[%d][%d][%d] = %f ShiftSeq[%d][%d] = %f+j%f UnShiftSeq[%d] %f+j%f DataRx[%d][%d][%d]= %f+j%f\n",
            0,0,0,
            tFreqInterpCoefs(0,0,0),
            0,0,
            tShiftSeq(0,0).x,
            tShiftSeq(0,0).y,
            0,
            tUnShiftSeq(0).x,
            tUnShiftSeq(0).y,
            0,0,0,
            tDataRx(0,0,0).x,
            tDataRx(0,0,0).y);
 #endif
 #endif
 
     const uint32_t THREAD_IDX = threadIdx.x;
 
 
 
     // Determine first PRB in the cluster being processed
     uint32_t prbClusterStartIdx = 0;
 
     uint32_t prbAbsStartIdx = prbClusterStartIdx + startPrb;
     // Absolute index of DMRS tone within the input OFDM symbol (used as index when loading tone from OFDM
     // symbol)
     const uint32_t DMRS_ABS_TONE_IDX = prbAbsStartIdx * N_TONES_PER_PRB + THREAD_IDX;
 
     // This index calculation intends to divvy up threads in the thread block for processing as follows:
     // the first group of N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER to process the first DMRS grid, the second group
     // to process the second DMRS grid and so on
     // Relative index of DMRS tone (within a DMRS grid) being processed by this thread
     const uint32_t DMRS_TONE_IDX        = THREAD_IDX % N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;
     const uint32_t DMRS_INTERP_TONE_IDX = THREAD_IDX % N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER;
 
     // Index of DMRS grid in which the DMRS tone being loaded by this thread resides
     const uint32_t DMRS_GRID_IDX = THREAD_IDX / N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER;

     const uint32_t tempGridIdx = DMRS_GRID_IDX > 1 ? 1 : DMRS_GRID_IDX;

     uint8_t   activeTOCCBmsk     = drvdUeGrpPrms.activeTOCCBmsk[tempGridIdx];
     uint8_t   activeFOCCBmsk     = drvdUeGrpPrms.activeFOCCBmsk[tempGridIdx];
     uint32_t  N_DMRS_SYMS_TOCC   = activeTOCCBmsk == 3 ? 2 : 1;
     uint32_t  N_DMRS_SYMS_FOCC   = activeFOCCBmsk == 3 ? 2 : 1;
 
     // Index of DMRS grid which thread computes
     const uint32_t DMRS_GRID_IDX_COMPUTE = THREAD_IDX / N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER;
 
     // Index which enables extraction of DMRS tones of a given DMRS grid scattered within the PRB into one
     // contiguous set for processing. Note that the read from GMEM is coalesced and write into SMEM is scattered
     // @todo: check if index calculation can be simplified
     const uint32_t SMEM_DMRS_TONE_IDX = get_smem_dmrs_tone_idx(N_DMRS_GRIDS_PER_PRB, N_INTER_DMRS_GRID_FREQ_SHIFT, THREAD_IDX);
     const uint32_t SMEM_DMRS_GRID_IDX = get_smem_dmrs_grid_idx(N_DMRS_GRIDS_PER_PRB, N_INTER_DMRS_GRID_FREQ_SHIFT, THREAD_IDX);
 
     // Absolute index of descrambling + shift sequence element
     const uint32_t DMRS_DESCR_SHIFT_SEQ_START_IDX = prbAbsStartIdx * N_DMRS_GRID_TONES_PER_PRB;
     const uint32_t DMRS_DESCR_SHIFT_SEQ_ABS_IDX = DMRS_TONE_IDX;
 
     // Absolute index of interpolated tone produced by this thread
     const uint32_t INTERP_DMRS_ABS_TONE_IDX =  DMRS_INTERP_TONE_IDX;
 
     // Select which estimation filter to used based on size of prb cluster
     constexpr uint32_t filtIdx = N_PRBS - 1;                                        
 
 
     // Select the shift in interpolation filter coefficients and delay shift based on grid index
     // (for e.g. for 2 DMRS grids and 48 tones per grid, multiply DMRS tone vector with top 48 rows for
     // DMRS_GRID_IDX 0 and bottom 48 rows for DMRS_GRID_IDX 1 to acheieve the effect of shift)
     uint32_t gridShiftIdx = get_inter_dmrs_grid_freq_shift_idx(N_DMRS_GRIDS_PER_PRB, DMRS_GRID_IDX_COMPUTE);
 
     // Section 5.2.1 in 3GPP TS 38.211
     // The fast-forward of 1600 prescribed by spec is already baked into the gold sequence generator
     constexpr uint32_t DMRS_DESCR_FF = 0; // 1600;
 
     // First descrambler bit index needed by this thread block
     // Note:The DMRS scrambling sequence is the same for all the DMRS grids. There are 2 sequences one for
     // scid 0 and other for scid 1 but the same sequences is reused for all DMRS grids
     const uint32_t DMRS_DESCR_PRB_CLUSTER_START_BIT =
         DMRS_DESCR_FF + (DMRS_DESCR_SHIFT_SEQ_START_IDX * N_DMRS_DESCR_BITS_PER_TONE);
 
     // The descrambling sequence generator outputs 32 descrambler bits at a time. Thus, compute the earliest
     // multiple of 32 bits which contains the descrambler bit of the first tone in the PRB cluster as the
     // start index
     const uint32_t DMRS_DESCR_GEN_ALIGNED_START_BIT =
         (DMRS_DESCR_PRB_CLUSTER_START_BIT / N_DMRS_DESCR_BITS_GEN) * N_DMRS_DESCR_BITS_GEN;
     // Offset to descrambler bit of the first tone in the PRB cluster
     const uint32_t DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET =
         DMRS_DESCR_PRB_CLUSTER_START_BIT - DMRS_DESCR_GEN_ALIGNED_START_BIT;
 
     // DMRS descrambling bits generated correspond to subcarriers across frequency
     // e.g. 2 bits for tone0(grid 0) | 2 bits for tone1(grid 1) | 2 bits for tone 2(grid 0) | 2 bits for tone 3(grid 1) | ...
     const uint32_t DMRS_TONE_DESCR_BIT_IDX = DMRS_DESCR_PRB_CLUSTER_START_BIT_OFFSET +
                                              (DMRS_TONE_IDX * N_DMRS_DESCR_BITS_PER_TONE);
     const uint32_t DMRS_DESCR_SEQ_RD_BIT_IDX  = DMRS_TONE_DESCR_BIT_IDX % N_DMRS_DESCR_BITS_GEN;
     const uint32_t DMRS_DESCR_SEQ_RD_WORD_IDX = DMRS_TONE_DESCR_BIT_IDX / N_DMRS_DESCR_BITS_GEN;
 
     const uint32_t DMRS_DESCR_SEQ_WR_WORD_IDX = THREAD_IDX % N_DMRS_DESCR_WORDS;
     const uint32_t DMRS_DESCR_SEQ_WR_SYM_IDX  = THREAD_IDX / N_DMRS_DESCR_WORDS;
 
     // determine if thread used to load dmrs:
     const bool loadFlag = (THREAD_IDX < N_PRBS*N_TONES_PER_PRB) ? true : false;
 
 #if 0
     if((0 == blockIdx.x) && (0 == blockIdx.y) && (0 == blockIdx.z))
        printf("N_DMRS_DESCR_BITS_PER_CLUSTER %d N_DMRS_DESCR_WORDS %d DMRS_DESCR_PRB_CLUSTER_START_BIT %d DMRS_DESCR_GEN_ALIGNED_START_BIT %d, "
               "DMRS_DESCR_SEQ_RD_WORD_IDX %d, DMRS_DESCR_SEQ_RD_BIT_IDX %d, DMRS_DESCR_SEQ_WR_WORD_IDX %d, DMRS_DESCR_SEQ_WR_SYM_IDX %d\n",
               N_DMRS_DESCR_BITS_PER_CLUSTER, N_DMRS_DESCR_WORDS, DMRS_DESCR_PRB_CLUSTER_START_BIT, DMRS_DESCR_GEN_ALIGNED_START_BIT, 
               DMRS_DESCR_SEQ_RD_WORD_IDX, DMRS_DESCR_SEQ_RD_BIT_IDX, DMRS_DESCR_SEQ_WR_WORD_IDX, DMRS_DESCR_SEQ_WR_SYM_IDX);
 #endif
 
     // Data layouts:
     // Global memory read into shared memory
     // N_DMRS_TONES x N_DMRS_SYMS -> N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS x N_DMRS_GRIDS_PER_PRB
 
     // tOCC removal
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB
 
     // fOCC removal
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB
 
     // Interpolation
     // N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB ->
     // N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_FOCC x NUM_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB =
     // N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER x N_LAYERS x N_DMRS_GRIDS_PER_PRB
 
     //--------------------------------------------------------------------------------------------------------
     // Allocate shared memory
 
     constexpr uint32_t N_SMEM_ELEMS1 = N_DMRS_TONES * N_DMRS_SYMS; // (N_DMRS_TONES + N_DMRS_GRIDS_PER_PRB)*N_DMRS_SYMS;
     constexpr uint32_t N_SMEM_ELEMS2 = N_INTERP_TONES * N_DMRS_SYMS_OCC;
     constexpr uint32_t N_SMEM_ELEMS  = (N_SMEM_ELEMS1 > N_SMEM_ELEMS2) ? N_SMEM_ELEMS1 : N_SMEM_ELEMS2;
     // constexpr uint32_t N_SMEM_ELEMS  = (N_SMEM_ELEMS1 + N_SMEM_ELEMS2);
     // constexpr uint32_t N_SMEM_ELEMS  = max(N_SMEM_ELEMS1, N_SMEM_ELEMS2);
 
     __shared__ TComplexCompute smemBlk[N_SMEM_ELEMS];
     // overlay1
     block_3D<TComplexCompute*, N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS, N_DMRS_GRIDS_PER_PRB> shPilots(&smemBlk[0]);
     // overlay2
     block_3D<TComplexCompute*, N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS_OCC, N_DMRS_GRIDS_PER_PRB> shH(&smemBlk[0]);
     // block_3D<TComplexCompute*, N_TOTAL_DMRS_INTERP_GRID_TONES_PER_CLUSTER, N_DMRS_SYMS_OCC, N_DMRS_GRIDS_PER_PRB> shH(&smemBlk[shPilots.num_elem()]);
     static_assert((shPilots.num_elem() <= N_SMEM_ELEMS) && (shH.num_elem() <= N_SMEM_ELEMS), "Insufficient shared memory");
 
     __shared__ uint32_t descrWords[N_DMRS_SYMS][N_DMRS_DESCR_WORDS];
 
     //--------------------------------------------------------------------------------------------------------
     // Read DMRS tones into shared memory (separate the tones into different DMRS grids during the write)
 
     // Cache shift sequence in register
     TComplexCompute shiftSeq = type_convert<TComplexCompute>(tShiftSeq(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, 0));
 
     if(loadFlag)
     {
 
     #pragma unroll
         for(uint32_t i = 0; i < N_DMRS_SYMS; ++i)
         {
             shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX) =
                 type_convert<TComplexCompute>(tDataRx(DMRS_ABS_TONE_IDX, pDmrsSymPos[i], BS_ANT_IDX));
 
     #ifdef ENABLE_DEBUG
             printf("Pilots[%d][%d][%d] -> shPilots[%d][%d][%d] = %f+j%f, ShiftSeq[%d][%d] = %f+j%f\n",
                 DMRS_ABS_TONE_IDX,
                 pDmrsSymPos[i],
                 BS_ANT_IDX,
                 SMEM_DMRS_TONE_IDX,
                 i,
                 SMEM_DMRS_GRID_IDX,
                 shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX).x,
                 shPilots(SMEM_DMRS_TONE_IDX, i, SMEM_DMRS_GRID_IDX).y,
                 DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                 0,
                 shiftSeq.x,
                 shiftSeq.y);
     #endif
         }
 
         // Compute the descsrambler sequence
         const uint32_t TWO_POW_17 = bit(17);
 
         if(DMRS_DESCR_SEQ_WR_SYM_IDX < N_DMRS_SYMS)
         {
             uint32_t symIdx = pDmrsSymPos[DMRS_DESCR_SEQ_WR_SYM_IDX];
 
             // see 38.211 section 6.4.1.1.1.1
             uint32_t cInit = TWO_POW_17 * (slotNum * OFDM_SYMBOLS_PER_SLOT + symIdx + 1) * (2 * dmrsScramId + 1) + (2 * dmrsScramId) + scid;
             cInit &= ~bit(31);
 
             // descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX] =
             //  __brev(gold32(cInit, (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX*N_DMRS_DESCR_BITS_GEN)));
 
             descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX] =
                 gold32(cInit, (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX * N_DMRS_DESCR_BITS_GEN));
     #if 0
             printf("symIdx %d, DMRS_DESCR_SEQ_WR_WORD_IDX %d, cInit 0x%08x, DMRS_DESCR_GEN_ALIGNED_START_BIT %d, descrWords[%d][%d] 0x%08x\n", 
                 symIdx, DMRS_DESCR_SEQ_WR_WORD_IDX, cInit, 
                 (DMRS_DESCR_GEN_ALIGNED_START_BIT + DMRS_DESCR_SEQ_WR_WORD_IDX*N_DMRS_DESCR_BITS_GEN), 
                 DMRS_DESCR_SEQ_WR_SYM_IDX, DMRS_DESCR_SEQ_WR_WORD_IDX, descrWords[DMRS_DESCR_SEQ_WR_SYM_IDX][DMRS_DESCR_SEQ_WR_WORD_IDX]);
     #endif
         }
     } 
 
     // To ensure coalesced reads, input DMRS tones are read preserving input order but swizzled while writing
     // to shared memory. Thus each thread may not process the same tone which it wrote to shared memory
     thread_block const& thisThrdBlk = this_thread_block();
     thisThrdBlk.sync();
 
     //--------------------------------------------------------------------------------------------------------
     // Apply de-scrambling + delay domain centering sequence for tone index processed by this thread across all
     // DMRS symbols
     const TCompute RECIPROCAL_SQRT2 = 0.7071068f;
     const TCompute SQRT2            = 1.41421356f;
     TComplexCompute    avg[N_TOCC]{};
     

     if(loadFlag)
     {
     #pragma unroll
         for(uint32_t i = 0; i < N_DMRS_SYMS; ++i)
         {
             int8_t descrIBit = (descrWords[i][DMRS_DESCR_SEQ_RD_WORD_IDX] >> DMRS_DESCR_SEQ_RD_BIT_IDX) & 0x1;
             int8_t descrQBit = (descrWords[i][DMRS_DESCR_SEQ_RD_WORD_IDX] >> (DMRS_DESCR_SEQ_RD_BIT_IDX + 1)) & 0x1;
 
             TComplexCompute descrCode =
                 cuConj(cuGet<TComplexCompute>((1 - 2 * descrIBit) * RECIPROCAL_SQRT2, (1 - 2 * descrQBit) * RECIPROCAL_SQRT2));
             TComplexCompute descrShiftSeq = shiftSeq * descrCode;
 
#ifdef ENABLE_DEBUG             
             TComplexCompute descrShiftPilot = shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) * descrShiftSeq;
             printf("descrShiftAbsIdx: %d, shPilots[%d][%d][%d] (%f+j%f) * DescrShiftSeq[%d][%d] (%f+j%f) = %f+j%f, ShiftSeq = %f+j%f, DescrCode = %f+j%f, descrIQ (%d,%d) descrWordIdx %d descrBitIdx %d\n",
                 DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                 DMRS_TONE_IDX,
                 i,
                 DMRS_GRID_IDX,
                 shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).x,
                 shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).y,
                 DMRS_TONE_IDX,
                 i,
                 descrShiftSeq.x,
                 descrShiftSeq.y,
                 descrShiftPilot.x,
                 descrShiftPilot.y,
                 shiftSeq.x,
                 shiftSeq.y,
                 descrCode.x,
                 descrCode.y,
                 descrIBit,
                 descrQBit,
                 DMRS_DESCR_SEQ_RD_WORD_IDX,
                 DMRS_DESCR_SEQ_RD_BIT_IDX);

             if((0 == DMRS_GRID_IDX) && (((0 == prbAbsStartIdx) && (DMRS_TONE_IDX < (N_EDGE_PRB * N_DMRS_GRID_TONES_PER_PRB))) || ((0 != prbAbsStartIdx) && (prbAbsStartIdx + N_DMRS_PRB_IN_PER_CLUSTER) <= N_DATA_PRB)))
             {
     #if 0
             TComplexCompute descrShiftPilot = shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) * descrShiftSeq;
             printf("descrShiftAbsIdx: %d shPilots[%d][%d][%d] (%f+j%f) * DescrShiftSeq[%d][%d] (%f+j%f) = %f+j%f, ShiftSeq = %f+j%f, DescrCode = %f+j%f, descrIQ (%d,%d) descrWordIdx %d descrBitIdx %d\n",
                     DMRS_DESCR_SHIFT_SEQ_ABS_IDX,
                     DMRS_TONE_IDX,
                     i,
                     DMRS_GRID_IDX,
                     shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).x,
                     shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX).y,
                     DMRS_TONE_IDX,
                     i,
                     descrShiftSeq.x,
                     descrShiftSeq.y,
                     descrShiftPilot.x,
                     descrShiftPilot.y,
                     shiftSeq.x,
                     shiftSeq.y,
                     descrCode.x,
                     descrCode.y,
                     descrIBit,
                     descrQBit,
                     DMRS_DESCR_SEQ_RD_WORD_IDX,
                     DMRS_DESCR_SEQ_RD_BIT_IDX);
     #endif
 
                 // tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(shiftSeq);
                 // tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(descrShiftSeq);
                 tDbg(DMRS_DESCR_SHIFT_SEQ_ABS_IDX, i, 0, 0) = type_convert<TComplexStorage>(descrCode);
             }
     #endif // ENABLE_DEBUG
 
             shPilots(DMRS_TONE_IDX, i, DMRS_GRID_IDX) *= descrShiftSeq;
         }
 
         //--------------------------------------------------------------------------------------------------------
         // Time domain cover code removal
         constexpr TCompute AVG_SCALE = cuGet<TCompute>(1u) / cuGet<TCompute>(N_DMRS_SYMS);
         // TComplexCompute    avg[N_DMRS_SYMS_TOCC]{};
         
         int32_t tOCCIdx = 0;
         
     #pragma unroll
         for(int32_t i = 0; i < 2; ++i)
         {
            int32_t temp = (activeTOCCBmsk >> i) & 0x1;
            if (!temp) {
                continue;
            }
     #pragma unroll
             for(int32_t j = 0; j < N_DMRS_SYMS; ++j)
             {
                 // For first tOCC (i = 0) output, multiply all DMRS symbols with +1 and average
                 // For second tOCC (i = 1) output, multiply even DMRS symbols with +1, odd DMRS symbols with -1 and average
                 int32_t sign = (-(i & j)) | 1;
                 avg[tOCCIdx] += (shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX) * (sign * AVG_SCALE));
     #ifdef ENABLE_DEBUG
                 TComplexCompute prod = (shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX) * (sign * AVG_SCALE));
                 printf("sign*AVG_SCALE %f Pilot[%d][%d][%d] = %f+j%f avg[%d] = %f+j%f, prod = %f+j%f\n",
                     sign * AVG_SCALE,
                     DMRS_TONE_IDX,
                     j,
                     DMRS_GRID_IDX,
                     shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX).x,
                     shPilots(DMRS_TONE_IDX, j, DMRS_GRID_IDX).y,
                     i,
                     avg[i].x,
                     avg[i].y,
                     prod.x,
                     prod.y);
     #endif
             }
             tOCCIdx++;
         }
     }
 
     // shPilots and shH are overlaid in shared memory and can have different sizes (based on config). For this reason
     // ensure shPilots access from all threads is completed before writing into shH
     thisThrdBlk.sync();
 
     //--------------------------------------------------------------------------------------------------------
     // Apply frequecy domain cover code and store inplace in shared memory
     // Multiply even tones with +1 and odd tones with -1
 
     if(loadFlag)
     {
         // Note that the loop termination count below is tOCC symbol count
     #pragma unroll
         for(int32_t i = 0; i < N_DMRS_SYMS_TOCC; ++i)
         {
            int32_t fOCCIdx = 0;

     #pragma unroll
             for(int32_t j = 0; j < 2; ++j)
             {
                int32_t temp = (activeFOCCBmsk >> j) & 0x1;
                if (!temp) {
                    continue;
                }
                 // First fOCC output: multiply all tones by +1s
                 // Second fOCC output: multiply even tones by +1s and odd tones by -1s
                 int32_t sign                                                  = (-(DMRS_TONE_IDX & j)) | 1;
                 shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + fOCCIdx, DMRS_GRID_IDX) = avg[i] * sign;
     #ifdef ENABLE_DEBUG
                 printf("PilotsPostOCC[%d][%d][%d] = %f+j%f\n",
                     DMRS_TONE_IDX,
                     (N_DMRS_SYMS_FOCC * i) + j,
                     DMRS_GRID_IDX,
                     cuReal(shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + j, DMRS_GRID_IDX)),
                     cuImag(shH(DMRS_TONE_IDX, (N_DMRS_SYMS_FOCC * i) + j, DMRS_GRID_IDX)));
     #endif
                 fOCCIdx++;
             }
         }
     }
 
     // Ensure all threads complete writing results to shared memory since each thread computing an inner product
     // during interpolation stage will use results from other threads in the thread block
     thisThrdBlk.sync();
 
     //--------------------------------------------------------------------------------------------------------
     // Interpolate (matrix-vector multiply)

     // early exit for inactive grids
     const int32_t DMRS_GRID_WR_IDX = DMRS_GRID_WR_IDX_TBL[activeDmrsGridBmsk & ACTIVE_DMRS_GRID_BMSK][DMRS_GRID_IDX_COMPUTE];
     // if(!is_set(bit(DMRS_GRID_IDX), activeDmrsGridBmsk) || (DMRS_GRID_WR_IDX < 0)) return;
     if(DMRS_GRID_WR_IDX < 0) return;

     activeTOCCBmsk     = drvdUeGrpPrms.activeTOCCBmsk[DMRS_GRID_IDX_COMPUTE];
     activeFOCCBmsk     = drvdUeGrpPrms.activeFOCCBmsk[DMRS_GRID_IDX_COMPUTE];
     N_DMRS_SYMS_TOCC   = activeTOCCBmsk == 3 ? 2 : 1;
     N_DMRS_SYMS_FOCC   = activeFOCCBmsk == 3 ? 2 : 1;

     for(uint32_t i = 0; i < (N_DMRS_SYMS_FOCC*N_DMRS_SYMS_TOCC); ++i)
     {
         TComplexCompute innerProd{};
 
         // H = W x Y: (N_INTERP_TONES x N_DMRS_TONES) x (N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER x N_DMRS_SYMS_OCC)
         // Each thread selects one row of W and computes N_DMRS_TONES length inner product to produce one interpolated
         // tone of H
         for(uint32_t j = 0; j < N_TOTAL_DMRS_GRID_TONES_PER_CLUSTER; ++j)
         {
             TCompute interpCoef = type_convert<TCompute>(tFreqInterpCoefs(DMRS_INTERP_TONE_IDX + gridShiftIdx, j, filtIdx));
             innerProd += (shH(j, i, DMRS_GRID_IDX_COMPUTE) * interpCoef);
         }
         // Wait for all threads to complete their inner products before updating the shared memory inplace
         // The sync is needed because shPilots and shH are overlaid
         thisThrdBlk.sync();
 
         shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX_COMPUTE) = innerProd;
 
 #ifdef ENABLE_DEBUG
         printf("InterpPilots[%d][%d][%d] = %f+j%f\n", DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX, innerProd.x, innerProd.y);
 #endif
     }
 
     // Wait for shared memory writes to complete from all threads
     thisThrdBlk.sync();
 
     //--------------------------------------------------------------------------------------------------------
     // Unshift the channel in delay back to its original location and write to GMEM. This is a scattered write
     // (@todo: any opportunities to make it coalesced?)
     // Output format is N_BS_ANT x (N_DMRS_SYMS_FOCC x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB) x N_DMRS_INTERP_PRB_OUT_PER_CLUSTER
     // where N_DMRS_SYMS_FOCC x N_DMRS_SYMS_TOCC x N_DMRS_GRIDS_PER_PRB = N_LAYERS
     //const uint32_t DMRS_GRID_OFFSET_H = DMRS_GRID_WR_IDX * N_DMRS_SYMS_OCC;
 
 // #ifdef CH_EST_COALESCED_WRITE
 //     // H (N_DATA_PRB*N_TONES_PER_PRB, N_LAYERS, N_BS_ANTS, NH)
 //     // read the number of rx antennas
 //     uint32_t N_BS_ANTS = drvdUeGrpPrms.nRxAnt;
 //     TComplexStorage* pHEst = tHEst.addr + ((NH_IDX * N_BS_ANTS + BS_ANT_IDX) * N_LAYERS * N_DATA_PRB * N_TONES_PER_PRB);
 // #endif
 
 
 // index of interpolated tone within cluster
 uint32_t CLUSTER_INTERP_TONE_IDX = DMRS_INTERP_TONE_IDX;  
 
 #pragma unroll
     for(uint32_t i = 0; i < N_LAYERS; ++i)
     {
        if (i < nLayers) {
            uint32_t j = OCCIdx[i] & 0x3;
            uint32_t k = (OCCIdx[i] >> 2) & 0x1;
            if (DMRS_GRID_IDX_COMPUTE == k) {
                shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX_COMPUTE) *=
                    type_convert<TComplexCompute>(tUnShiftSeq(CLUSTER_INTERP_TONE_IDX + gridShiftIdx)); //INTERP_DMRS_ABS_TONE_IDX
                if(nDmrsCdmGrpsNoData==1)
                {
                    shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX) *= cuGet<TComplexCompute>(SQRT2, 0.0f);
                }
 
 #ifndef CH_EST_COALESCED_WRITE
     tHEst(BS_ANT_IDX, i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX) =
             type_convert<TComplexStorage>(shH(DMRS_INTERP_TONE_IDX, j, DMRS_GRID_IDX_COMPUTE));
 #else //fix me for flexbile DMRS port mapping
         pHEst[(DMRS_GRID_OFFSET_H + i) * N_DATA_PRB * N_TONES_PER_PRB + INTERP_DMRS_ABS_TONE_IDX] =
             type_convert<TComplexStorage>(shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX_COMPUTE));
 #endif
            }
 #ifdef ENABLE_DEBUG
 #if 0 
      printf("shH[%d][%d][%d] = %f+j%f\n", DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX,
          shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX).x, shH(DMRS_INTERP_TONE_IDX, i, DMRS_GRID_IDX).y);
 #endif
 #if 0     
       printf("tH[%d][%d][%d][%d] = %f+j%f\n", BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX,
          tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x,
          tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y);
 #endif
 #if 0
        printf("tUnshift[%d] = %f+j%f\n", INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx,tUnShiftSeq(INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx).x, tUnShiftSeq(INTERP_DMRS_ABS_TONE_IDX + gridShiftIdx).y);
  
        printf("tH[%d][%d][%d][%d] = %f+j%f\n", BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX,
          tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).x,
          tHEst(BS_ANT_IDX, DMRS_GRID_OFFSET_H + i, INTERP_DMRS_ABS_TONE_IDX, NH_IDX).y);
 #endif
 #endif
        }
     }
 } //smallChEstNoDftSOfdmKernel


 __device__ __forceinline__ unsigned long long get_ptimer_ns()
 {
     unsigned long long globaltimer;
     // 64-bit global nanosecond timer
     asm volatile("mov.u64 %0, %globaltimer;"
                  : "=l"(globaltimer));
     return globaltimer;
 }

 // kernel to wait until first SYMIDX bits are ready
 template <uint32_t SYMIDX>
 static __global__ void
 preEarlyHarqWaitKernel(puschRxChEstStatDescr_t* pStatDescr, puschRxChEstDynDescr_t* pDynDescr)
 {
     if(threadIdx.x==0)
     {
         bool      bitsReady  = false;
         bool      timedOut   = false;
         uint64_t startTime   = get_ptimer_ns();
         uint64_t maxWaitTime = startTime + static_cast<uint64_t>(pDynDescr->waitTimeOutPreEarlyHarqUs) * 1000;
    
         pDynDescr->mPuschStartTimeNs = startTime; // record the start time to be used in postEarlyHarqWaitKernel
    
         while (!bitsReady && !timedOut)
         {
            bitsReady = true;
            for(int i = 0; i < SYMIDX; i++)
            {
                volatile uint32_t* pstat = (volatile uint32_t*)&(pStatDescr->pSymbolRxStatus[i]);
                if (*pstat != SYM_RX_DONE)
                {
                    bitsReady = false;
                    break;
                }
            }
            timedOut = get_ptimer_ns() > maxWaitTime;   // to ensure the kernel won't get stuck in while loop
         }
    
         if(timedOut && (!bitsReady))
         {
             *(pDynDescr->pPreEarlyHarqWaitKernelStatus_d) = PUSCH_RX_WAIT_KERNEL_STATUS_TIMEOUT;
         }
         else
         {
             *(pDynDescr->pPreEarlyHarqWaitKernelStatus_d) = PUSCH_RX_WAIT_KERNEL_STATUS_DONE;
         }
     }
 }

 template <uint32_t SYMIDX>
 static __global__ void
 postEarlyHarqWaitKernel(puschRxChEstStatDescr_t* pStatDescr, puschRxChEstDynDescr_t* pDynDescr)
 {
     if(threadIdx.x==0)
     {
         bool      bitsReady  = false;
         bool      timedOut   = false;
         uint64_t startTime   = pDynDescr->mPuschStartTimeNs;
         uint64_t maxWaitTime = startTime + static_cast<uint64_t>(pDynDescr->waitTimeOutPostEarlyHarqUs) * 1000;
    
         while (!bitsReady && !timedOut)
         {
            bitsReady = true;
            for(int i = 0; i < SYMIDX; i++)
            {
                volatile uint32_t* pstat = (volatile uint32_t*)&(pStatDescr->pSymbolRxStatus[i]);
                if (*pstat != SYM_RX_DONE)
                {
                    bitsReady = false;
                    break;
                }
            }
            timedOut = get_ptimer_ns() > maxWaitTime;   // to ensure the kernel won't get stuck in while loop
         }

         if(timedOut && (!bitsReady))
         {
             *(pDynDescr->pPostEarlyHarqWaitKernelStatus_d) = PUSCH_RX_WAIT_KERNEL_STATUS_TIMEOUT;
         }
         else
         {
             *(pDynDescr->pPostEarlyHarqWaitKernelStatus_d) = PUSCH_RX_WAIT_KERNEL_STATUS_DONE;
         }
     }
 }
 
 
 template <uint32_t N_DMRS_GRIDS_PER_PRB,              // # of DMRS grids per PRB (2 or 3)
           uint32_t N_DMRS_PRB_IN_PER_CLUSTER,         // # of PRBs bearing DMRS tones to be processed by each thread block (i.e. used in channel estimation)
           uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER> // # of PRBs bearing interpolated tones at output
 void
 puschRxChEst::computeKernelLaunchGeo(uint16_t nTotalDataPrb,
                                      uint16_t nUeGrps,
                                      uint32_t nRxAnt,
                                      dim3&    gridDim,
                                      dim3&    blockDim)
 {
     constexpr uint32_t N_TOTAL_INPUT_TONES = N_DMRS_PRB_IN_PER_CLUSTER * N_TONES_PER_PRB;
     constexpr uint32_t N_TOTAL_OUTPUT_TONES = N_DMRS_INTERP_PRB_OUT_PER_CLUSTER * N_DMRS_GRIDS_PER_PRB * N_TONES_PER_PRB;
     static_assert((N_TOTAL_INPUT_TONES == N_TOTAL_OUTPUT_TONES),
                   "Thread allocation assumes input DMRS tone count and interpolated tone count are equal, ensure sufficient threads are allocated for interpoloation etc");
 
     const uint32_t N_THREAD_BLKS_PER_BS_ANT = div_round_up(nTotalDataPrb, static_cast<uint16_t>(N_DMRS_INTERP_PRB_OUT_PER_CLUSTER));
     gridDim  = dim3(N_THREAD_BLKS_PER_BS_ANT, nRxAnt, nUeGrps);
     blockDim = dim3(N_TOTAL_OUTPUT_TONES);
 
#ifdef ENABLE_DEBUG
     NVLOGI_FMT(NVLOG_PUSCH, "{}: blockDim ({},{},{}), gridDim ({},{},{})", __FUNCTION__, blockDim.x, blockDim.y, blockDim.z, gridDim.x, gridDim.y, gridDim.z);
#endif
 }
 
 template <typename TStorage,
           typename TDataRx,
           typename TCompute,
           uint32_t N_LAYERS,                          // # of layers (# of cols in H matrix)
           uint32_t N_DMRS_GRIDS_PER_PRB,              // # of DMRS grids per PRB (2 or 3)
           uint32_t N_DMRS_PRB_IN_PER_CLUSTER,         // # of PRBs bearing DMRS tones to be processed by each thread block (i.e. used in channel estimation)
           uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER, // # of PRBs bearing channel estimates (interpolated tones) at output
           uint32_t N_DMRS_SYMS>                       // # of time domain DMRS symbols (1,2 or 4)
 void
 puschRxChEst::windowedChEst(uint16_t                        nTotalDataPrb,
                             uint16_t                        nUeGrps,
                             uint32_t                        nRxAnt,
                             uint8_t                         enableDftSOfdm,
                             cuphyPuschRxChEstLaunchCfg_t&   launchCfg)
 {
     if(enableDftSOfdm==1)
     {
         void* kernelFunc = reinterpret_cast<void*>(windowedChEstKernel<TStorage,
                                                                    TDataRx,
                                                                    TCompute,
                                                                    N_LAYERS,
                                                                    N_DMRS_GRIDS_PER_PRB,
                                                                    N_DMRS_PRB_IN_PER_CLUSTER,
                                                                    N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
                                                                    N_DMRS_SYMS>);
  
         CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriver.func, kernelFunc));
     }
     else if(enableDftSOfdm==0)
     {
         void* kernelFunc = reinterpret_cast<void*>(windowedChEstNoDftSOfdmKernel<TStorage,
                                                                    TDataRx,
                                                                    TCompute,
                                                                    N_LAYERS,
                                                                    N_DMRS_GRIDS_PER_PRB,
                                                                    N_DMRS_PRB_IN_PER_CLUSTER,
                                                                    N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
                                                                    N_DMRS_SYMS>);
  
         CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriver.func, kernelFunc));
     
     }
     // else
     // {
     //     assert(false);
     // }
 
     dim3 blockDim, gridDim;
     computeKernelLaunchGeo<N_DMRS_GRIDS_PER_PRB,
                            N_DMRS_PRB_IN_PER_CLUSTER,
                            N_DMRS_INTERP_PRB_OUT_PER_CLUSTER>(nTotalDataPrb, nUeGrps, nRxAnt, gridDim, blockDim);
                            
     CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = launchCfg.kernelNodeParamsDriver;
     kernelNodeParamsDriver.blockDimX = blockDim.x;
     kernelNodeParamsDriver.blockDimY = blockDim.y;
     kernelNodeParamsDriver.blockDimZ = blockDim.z;
 
     kernelNodeParamsDriver.gridDimX = gridDim.x;
     kernelNodeParamsDriver.gridDimY = gridDim.y;
     kernelNodeParamsDriver.gridDimZ = gridDim.z;
     
     kernelNodeParamsDriver.extra          = nullptr;
     kernelNodeParamsDriver.sharedMemBytes = 0;
 }
 
 
 template <typename TStorage,
           typename TDataRx,
           typename TCompute,
           uint32_t N_LAYERS,                          // # of layers (# of cols in H matrix)
           uint32_t N_PRBS,                            // 1, 2 or 3
           uint32_t N_DMRS_GRIDS_PER_PRB,              // # of DMRS grids per PRB (2 or 3)
           uint32_t N_DMRS_SYMS>                       // # of time domain DMRS symbols (1,2 or 4)
 void
 puschRxChEst::smallChEst(uint16_t                        nUeGrps,
                          uint32_t                        nRxAnt,
                          uint8_t                         enableDftSOfdm,
                          cuphyPuschRxChEstLaunchCfg_t&   launchCfg)
 {
 
     if(enableDftSOfdm == 1)
     {
         void* kernelFunc = reinterpret_cast<void*>(smallChEstKernel<TStorage,
                                                                    TDataRx,
                                                                    TCompute,
                                                                    N_LAYERS,
                                                                    N_PRBS,
                                                                    N_DMRS_GRIDS_PER_PRB,
                                                                    N_DMRS_SYMS>);
  
         CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriver.func, kernelFunc));
     }
     else if(enableDftSOfdm==0)
     {
         void* kernelFunc = reinterpret_cast<void*>(smallChEstNoDftSOfdmKernel<TStorage,
                                                                               TDataRx,
                                                                               TCompute,
                                                                               N_LAYERS,
                                                                               N_PRBS,
                                                                               N_DMRS_GRIDS_PER_PRB,
                                                                               N_DMRS_SYMS>);
  
         CUDA_CHECK(cudaGetFuncBySymbol(&launchCfg.kernelNodeParamsDriver.func, kernelFunc));
     }
     
     // compute launch geometry:
     uint32_t  N_THREADS = 2 * N_PRBS * N_TONES_PER_PRB;
 
     dim3 gridDim(1, nRxAnt, nUeGrps);
     dim3 blockDim(N_THREADS);
                            
     CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = launchCfg.kernelNodeParamsDriver;
     kernelNodeParamsDriver.blockDimX = blockDim.x;
     kernelNodeParamsDriver.blockDimY = blockDim.y;
     kernelNodeParamsDriver.blockDimZ = blockDim.z;
 
     kernelNodeParamsDriver.gridDimX = gridDim.x;
     kernelNodeParamsDriver.gridDimY = gridDim.y;
     kernelNodeParamsDriver.gridDimZ = gridDim.z;
     
     kernelNodeParamsDriver.extra          = nullptr;
     kernelNodeParamsDriver.sharedMemBytes = 0;
 }
 
 template <typename TStorage, 
           typename TDataRx, 
           typename TCompute, 
           uint32_t N_LAYERS, 
           uint32_t N_DMRS_GRIDS_PER_PRB, 
           uint32_t N_DMRS_SYMS>
 void puschRxChEst::kernelSelectL0(uint16_t                        nTotalDataPrb,
                                   uint16_t                        nUeGrps,
                                   uint32_t                        nRxAnt,
                                   uint8_t                         enableDftSOfdm,
                                   cuphyPuschRxChEstLaunchCfg_t&   launchCfg)
 {
     bool noKernelFound = false;
     if((0 == (nTotalDataPrb % 4)) && (nTotalDataPrb > 7)) // (nTotalDataPrb >= 8)
     {
         constexpr uint32_t N_DMRS_PRB_IN_PER_CLUSTER         = 8; // # of DMRS PRBs processed by a thread block
         constexpr uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER = 4; // # of DMRS interpolated PRBs produced by a thread block
 
         windowedChEst<TStorage,
                         TDataRx,
                         TCompute,
                         N_LAYERS,
                         N_DMRS_GRIDS_PER_PRB,
                         N_DMRS_PRB_IN_PER_CLUSTER,
                         N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
                         N_DMRS_SYMS>(nTotalDataPrb, nUeGrps, nRxAnt, enableDftSOfdm, launchCfg);
     }
     else if(((0 != (nTotalDataPrb % 4)) || (nTotalDataPrb == 4)) && (nTotalDataPrb > 3))  // (nTotalDataPrb >= 4)
     {
         constexpr uint32_t N_DMRS_PRB_IN_PER_CLUSTER         = 4; // # of DMRS PRBs processed by a thread block
         constexpr uint32_t N_DMRS_INTERP_PRB_OUT_PER_CLUSTER = 2; // # of DMRS interpolated PRBs produced by a thread block
 
         windowedChEst<TStorage,
                         TDataRx,
                         TCompute,
                         N_LAYERS,
                         N_DMRS_GRIDS_PER_PRB,
                         N_DMRS_PRB_IN_PER_CLUSTER,
                         N_DMRS_INTERP_PRB_OUT_PER_CLUSTER,
                         N_DMRS_SYMS>(nTotalDataPrb, nUeGrps, nRxAnt, enableDftSOfdm, launchCfg);
     }
     else if(nTotalDataPrb < 4) // (nTotalDataPrb < 4)
     {
         switch(nTotalDataPrb)
         {
             case 3:
             {
                 constexpr uint32_t N_PRBS = 3; 
                 smallChEst<TStorage,
                             TDataRx,
                             TCompute,
                             N_LAYERS,
                             N_PRBS,
                             N_DMRS_GRIDS_PER_PRB,
                             N_DMRS_SYMS>(nUeGrps, nRxAnt, enableDftSOfdm, launchCfg);
                 break; 
             }
             
             case 2:
             {
                 constexpr uint32_t N_PRBS = 2; 
                 smallChEst<TStorage,
                             TDataRx,
                             TCompute,
                             N_LAYERS,
                             N_PRBS,
                             N_DMRS_GRIDS_PER_PRB,
                             N_DMRS_SYMS>(nUeGrps, nRxAnt, enableDftSOfdm, launchCfg);
                 break;
             }
 
             case 1:
             {
                 constexpr uint32_t N_PRBS = 1; 
                 smallChEst<TStorage,
                             TDataRx,
                             TCompute,
                             N_LAYERS,
                             N_PRBS,
                             N_DMRS_GRIDS_PER_PRB,
                             N_DMRS_SYMS>(nUeGrps, nRxAnt, enableDftSOfdm, launchCfg);
                 break;
             }
         }
     }
     else
     {
         noKernelFound = true;
     }
     if(noKernelFound)
     {
         NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available nTotalDataPrb {}", __FUNCTION__, nTotalDataPrb);
     }  
 }
 
 template <typename TStorage, typename TDataRx, typename TCompute>
 void puschRxChEst::kernelSelectL1(uint16_t                        nBSAnts,
                                   uint8_t                         nLayers,
                                   uint8_t                         nDmrsSyms,
                                   uint8_t                         nDmrsGridsPerPrb,
                                   uint16_t                        nTotalDataPrb,
                                   uint8_t                         Nh,
                                   uint16_t                        nUeGrps,
                                   uint8_t                         enableDftSOfdm,
                                   cuphyPuschRxChEstLaunchCfg_t&   launchCfg)
 {
     bool noKernelFound = false;
 
     // Check below ensures the parameters match the dimensions assumed in the kernel. Among others it ensures
     // that nTotalDataPrb is divisible by N_DMRS_INTERP_PRB_OUT_PER_CLUSTER and divisible by N_DMRS_PRB_IN_PER_CLUSTER

     if((nBSAnts >= nLayers) && (nLayers <= 8) && (2 == nDmrsGridsPerPrb) && (2 == nDmrsSyms) && (1 == Nh))
     {
         constexpr uint32_t N_DMRS_GRIDS_PER_PRB = 2; // 2 grids => 6 grid tones per PRB
         constexpr uint32_t N_DMRS_SYMS          = 2; // # of DMRS symbols

         switch(nLayers)
         {
             case 8:
             case 7:
             case 6:
             case 5:
             {
                constexpr uint32_t N_LAYERS = 8; // # of layers (# of cols in H matrix)
                kernelSelectL0<TStorage, 
                               TDataRx, 
                               TCompute, 
                               N_LAYERS, 
                               N_DMRS_GRIDS_PER_PRB, 
                               N_DMRS_SYMS>(nTotalDataPrb, nUeGrps, nBSAnts, enableDftSOfdm, launchCfg);
                break;
             }

             case 4:
             case 3:
             case 2:
             {
                 constexpr uint32_t N_LAYERS = 4; // # of layers (# of cols in H matrix)
                 kernelSelectL0<TStorage, 
                                TDataRx, 
                                TCompute, 
                                N_LAYERS, 
                                N_DMRS_GRIDS_PER_PRB, 
                                N_DMRS_SYMS>(nTotalDataPrb, nUeGrps, nBSAnts, enableDftSOfdm, launchCfg);
                 break;
             } // nLayers = 2,4
 
             case 1:
             {
                 constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
                 kernelSelectL0<TStorage, 
                                TDataRx, 
                                TCompute, 
                                N_LAYERS, 
                                N_DMRS_GRIDS_PER_PRB, 
                                N_DMRS_SYMS>(nTotalDataPrb, nUeGrps, nBSAnts, enableDftSOfdm, launchCfg);
                 break;
             } // nLayers = 1
 
             default: noKernelFound = true; break;
         } // nLayers
     }
     else if((nBSAnts >= nLayers) && (nLayers < 8) && (2 == nDmrsGridsPerPrb) && (1 == nDmrsSyms) && (1 == Nh))     
     {
         constexpr uint32_t N_DMRS_GRIDS_PER_PRB = 2; // 2 grids => 6 grid tones per PRB
         constexpr uint32_t N_DMRS_SYMS          = 1; // # of DMRS symbols

         switch(nLayers)
         {
             case 4:
             case 3:
             case 2:
             {
                 constexpr uint32_t N_LAYERS = 4; // # of layers (# of cols in H matrix)
                 kernelSelectL0<TStorage, 
                                TDataRx, 
                                TCompute, 
                                N_LAYERS, 
                                N_DMRS_GRIDS_PER_PRB, 
                                N_DMRS_SYMS>(nTotalDataPrb, nUeGrps, nBSAnts, enableDftSOfdm, launchCfg);
                 break;
             } // nLayers = 2,4
 
             case 1:
             {
                 constexpr uint32_t N_LAYERS = 1; // # of layers (# of cols in H matrix)
                 kernelSelectL0<TStorage, 
                                TDataRx, 
                                TCompute, 
                                N_LAYERS, 
                                N_DMRS_GRIDS_PER_PRB, 
                                N_DMRS_SYMS>(nTotalDataPrb, nUeGrps, nBSAnts, enableDftSOfdm, launchCfg);
                 break;
             } // nLayers = 1
 
             default: noKernelFound = true; break;
         } // nLayers
     }     
     else
     {
         noKernelFound = true;
     }
 
     if(noKernelFound)
     {
         NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available (L1 stage) to launch with requested configuration: nBSAnts {} nLayers {} nDmrsGridsPerPrb {} nDmrsSyms {} Nh {} nTotalDataPrb {}", __FUNCTION__, nBSAnts, nLayers, nDmrsGridsPerPrb, nDmrsSyms, Nh, nTotalDataPrb);
     }
 }
 
 void puschRxChEst::kernelSelectL2(uint16_t                        nBSAnts,
                                   uint8_t                         nLayers,
                                   uint8_t                         nDmrsSyms,
                                   uint8_t                         nDmrsGridsPerPrb,
                                   uint16_t                        nTotalDataPrb,
                                   uint8_t                         Nh,
                                   uint16_t                        nUeGrps,
                                   uint8_t                         enableDftSOfdm,
                                   cuphyDataType_t                 dataRxType,
                                   cuphyDataType_t                 hEstType,
                                   cuphyPuschRxChEstLaunchCfg_t&   launchCfg)
 {
     using TCompute = float;
     if(CUPHY_C_32F == hEstType)
     {
         using TStorage = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
         if(CUPHY_C_32F == dataRxType)
         {
             using TDataRx = scalar_from_complex<data_type_traits<CUPHY_C_32F>::type>::type;
             kernelSelectL1<TStorage, TDataRx, TCompute>(nBSAnts,
                                                         nLayers,
                                                         nDmrsSyms,
                                                         nDmrsGridsPerPrb,
                                                         nTotalDataPrb,
                                                         Nh,
                                                         nUeGrps,
                                                         enableDftSOfdm,
                                                         launchCfg);
         }
         else if(CUPHY_C_16F == dataRxType)
         {
             using TDataRx = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
             kernelSelectL1<TStorage, TDataRx, TCompute>(nBSAnts,
                                                         nLayers,
                                                         nDmrsSyms,
                                                         nDmrsGridsPerPrb,
                                                         nTotalDataPrb,
                                                         Nh,
                                                         nUeGrps,
                                                         enableDftSOfdm,
                                                         launchCfg);
         }
         else
         {
             NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested date type", __FUNCTION__);
         }
     }
     else if((CUPHY_C_16F == hEstType) && (CUPHY_C_16F == dataRxType))
     {
         using TStorage = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
         using TDataRx  = scalar_from_complex<data_type_traits<CUPHY_C_16F>::type>::type;
         kernelSelectL1<TStorage, TDataRx, TCompute>(nBSAnts,
                                                     nLayers,
                                                     nDmrsSyms,
                                                     nDmrsGridsPerPrb,
                                                     nTotalDataPrb,
                                                     Nh,
                                                     nUeGrps,
                                                     enableDftSOfdm,
                                                     launchCfg);
     }
     else
     {
         NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: No kernel available to launch with requested date type", __FUNCTION__);
     }
 }
 
 void puschRxChEst::init(tensor_pair&             tFreqInterpCoefs,
                         tensor_pair&             tFreqInterpCoefs4,
                         tensor_pair&             tFreqInterpCoefsSmall,
                         tensor_pair&             tShiftSeq,
                         tensor_pair&             tShiftSeq4,
                         tensor_pair&             tUnShiftSeq,
                         tensor_pair&             tUnShiftSeq4,
                         const uint32_t*          pSymStat,
                         bool                     enableCpuToGpuDescrAsyncCpy,
                         void**                   ppStatDescrsCpu,
                         void**                   ppStatDescrsGpu,
                         cudaStream_t             strm)
 {
     // TODO: use std::copy_n
     auto copyEntries = [](int* pDst, const int* pSrc, size_t nEntries) {for(uint32_t i = 0; i < nEntries; ++i) pDst[i] = pSrc[i]; };

     for(int32_t chEstTimeInstIdx = 0; chEstTimeInstIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEstTimeInstIdx)
     {
        puschRxChEstStatDescr_t& statDescrCpu = *(static_cast<puschRxChEstStatDescr_t*>(ppStatDescrsCpu[chEstTimeInstIdx]));

        statDescrCpu.pSymbolRxStatus = pSymStat;

        statDescrCpu.tPrmFreqInterpCoefs.pAddr          = tFreqInterpCoefs.second;
        const tensor_layout_any& tFreqInterpCoefsLayout = tFreqInterpCoefs.first.get().layout();
        copyEntries(statDescrCpu.tPrmFreqInterpCoefs.strides, tFreqInterpCoefsLayout.strides.begin(), tFreqInterpCoefsLayout.rank());
    
        statDescrCpu.tPrmFreqInterpCoefs4.pAddr          = tFreqInterpCoefs4.second;
        const tensor_layout_any& tFreqInterpCoefsLayout4 = tFreqInterpCoefs4.first.get().layout();
        copyEntries(statDescrCpu.tPrmFreqInterpCoefs4.strides, tFreqInterpCoefsLayout4.strides.begin(), tFreqInterpCoefsLayout4.rank());
    
        statDescrCpu.tPrmFreqInterpCoefsSmall.pAddr          = tFreqInterpCoefsSmall.second;
        const tensor_layout_any& tFreqInterpCoefsLayoutSmall = tFreqInterpCoefsSmall.first.get().layout();
        copyEntries(statDescrCpu.tPrmFreqInterpCoefsSmall.strides, tFreqInterpCoefsLayoutSmall.strides.begin(), tFreqInterpCoefsLayoutSmall.rank());
    
        statDescrCpu.tPrmShiftSeq.pAddr          = tShiftSeq.second;
        const tensor_layout_any& tShiftSeqLayout = tShiftSeq.first.get().layout();
        copyEntries(statDescrCpu.tPrmShiftSeq.strides, tShiftSeqLayout.strides.begin(), tShiftSeqLayout.rank());
    
        statDescrCpu.tPrmShiftSeq4.pAddr          = tShiftSeq4.second;
        const tensor_layout_any& tShiftSeqLayout4 = tShiftSeq4.first.get().layout();
        copyEntries(statDescrCpu.tPrmShiftSeq4.strides, tShiftSeqLayout4.strides.begin(), tShiftSeqLayout4.rank());
    
        statDescrCpu.tPrmUnShiftSeq.pAddr          = tUnShiftSeq.second;
        const tensor_layout_any& tUnShiftSeqLayout = tUnShiftSeq.first.get().layout();
        copyEntries(statDescrCpu.tPrmUnShiftSeq.strides, tUnShiftSeqLayout.strides.begin(), tUnShiftSeqLayout.rank());
    
        statDescrCpu.tPrmUnShiftSeq4.pAddr          = tUnShiftSeq4.second;
        const tensor_layout_any& tUnShiftSeqLayout4 = tUnShiftSeq4.first.get().layout();
        copyEntries(statDescrCpu.tPrmUnShiftSeq4.strides, tUnShiftSeqLayout4.strides.begin(), tUnShiftSeqLayout4.rank());
    
        for(auto& kernelArgs : m_kernelArgsArr[chEstTimeInstIdx])
        {
            kernelArgs.pStatDescr = static_cast<puschRxChEstStatDescr_t*>(ppStatDescrsGpu[chEstTimeInstIdx]);
        }

        if(enableCpuToGpuDescrAsyncCpy)
        {
            //Unchecked return value
            CUDA_CHECK(cudaMemcpyAsync(ppStatDescrsGpu[chEstTimeInstIdx], ppStatDescrsCpu[chEstTimeInstIdx], sizeof(puschRxChEstStatDescr_t), cudaMemcpyHostToDevice, strm));
        }   
     }
 }
 
 void puschRxChEst::getDescrInfo(size_t& statDescrSizeBytes, size_t& statDescrAlignBytes, size_t& dynDescrSizeBytes, size_t& dynDescrAlignBytes)
 {
     statDescrSizeBytes  = sizeof(puschRxChEstStatDescr_t);
     statDescrAlignBytes = alignof(puschRxChEstStatDescr_t);
 
     dynDescrSizeBytes  = sizeof(puschRxChEstDynDescrVec_t);
     dynDescrAlignBytes = alignof(puschRxChEstDynDescrVec_t);
 }

 cuphyStatus_t puschRxChEst::batch(uint32_t                       chEstTimeInstIdx,
                          cuphyPuschRxUeGrpPrms_t*       pDrvdUeGrpPrms,
                          uint16_t                       nUeGrps,
                          uint8_t                        enableDftSOfdm,
                          uint32_t&                      nHetCfgs,
                          puschRxChEstDynDescrVec_t&     dynDescrVecCpu)
 {
     // Initialize the batch config data structure
     puschRxChEstHetCfgArr_t& hetCfgs = m_hetCfgsArr[chEstTimeInstIdx];
     hetCfgs.fill({nullptr, 0, 0});
 
     // Helper to find kernel function
     auto findKernelFunc = [](puschRxChEstHetCfgArr_t const& hetCfgs, CUfunction func, int32_t& hetCfgIdx)
     {
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
 
 #ifdef ENABLE_DEBUG  
     NVLOGI_FMT(NVLOG_PUSCH, "{}: # of UE groups {}", __FUNCTION__, nUeGrps);  
 #endif

     nHetCfgs = 0;
     for(int32_t ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
     {
        cuphyPuschRxUeGrpPrms_t const& drvdUeGrpPrms = pDrvdUeGrpPrms[ueGrpIdx];

        // Skip UE group if there aren't enough DMRS additional positions
        // # of time domain channel estimates is equal to the number of DMRS additional positions + 1
        if(chEstTimeInstIdx > drvdUeGrpPrms.dmrsAddlnPos)  continue;

         uint16_t nPrb        = drvdUeGrpPrms.nPrb;
         uint16_t nRxAnt      = drvdUeGrpPrms.nRxAnt;
         auto  symLocBmsk     = drvdUeGrpPrms.dmrsSymLocBmsk;
         int32_t nMinDmrsSyms = std::min(static_cast<int32_t>(drvdUeGrpPrms.nDmrsSyms), __builtin_popcount(symLocBmsk));
         
         // @todo: extend kernelSelectL2 to support a mode which only determines the kernel function
         cuphyPuschRxChEstLaunchCfg_t launchCfg;
         kernelSelectL2(drvdUeGrpPrms.nRxAnt,
                        drvdUeGrpPrms.nLayers,
                        drvdUeGrpPrms.dmrsMaxLen,
                        drvdUeGrpPrms.nDmrsGridsPerPrb,
                        nPrb,
                        1,
                        nUeGrps,
                        enableDftSOfdm,
                        drvdUeGrpPrms.tInfoDataRx.elemType,
                        drvdUeGrpPrms.tInfoHEst.elemType,
                        launchCfg);
 
         // Check if the heterognous configuration already exists
         int32_t hetCfgIdx = 0;
         findKernelFunc(hetCfgs, launchCfg.kernelNodeParamsDriver.func, hetCfgIdx);
 
         // If a heterogenous configuration already exists then increment the # of UE groups for that config
         if(-1 != hetCfgIdx)
         {
             puschRxChEstHetCfg_t& hetCfg = hetCfgs[hetCfgIdx];
             if(hetCfg.nUeGrps >= MAX_N_USER_GROUPS_SUPPORTED)
             {
                 NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Exceeded limit on supported UE groups", __FUNCTION__);
                 return CUPHY_STATUS_INTERNAL_ERROR;
             }
 
             if(nPrb   > hetCfg.nMaxPrb)   hetCfg.nMaxPrb   = nPrb;
             if(nRxAnt > hetCfg.nMaxRxAnt) hetCfg.nMaxRxAnt = nRxAnt;
 
             dynDescrVecCpu[hetCfgIdx].hetCfgUeGrpMap[hetCfg.nUeGrps] = ueGrpIdx;
             hetCfg.nUeGrps++;
 
 #ifdef ENABLE_DEBUG
            NVLOGI_FMT(NVLOG_PUSCH, "{}: UE group {} -> HetCfg {} funcPtr {:p} (nHetCfgs {} nUeGrps {} nPrb {} nMaxPrb {} nRxAnt {} nLayers {} dmrsAddlnPos {} dmrsMaxLen {} nDmrsGridsPerPrb {})", __FUNCTION__, ueGrpIdx, newHetCfgIdx, static_cast<void*>(hetCfg.func), nHetCfgs, hetCfg.nUeGrps, nPrb, hetCfg.nMaxPrb, drvdUeGrpPrms.nRxAnt, drvdUeGrpPrms.nLayers, drvdUeGrpPrms.dmrsAddlnPos, drvdUeGrpPrms.dmrsMaxLen, drvdUeGrpPrms.nDmrsGridsPerPrb);
 #endif            
         }
         // New heterogenous configuration found
         else
         {
             if(nHetCfgs >= CUPHY_PUSCH_RX_CH_EST_N_MAX_HET_CFGS)
             {
                 NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Exceeded limit on supported heterogneous configurations", __FUNCTION__);
                 return CUPHY_STATUS_INTERNAL_ERROR;
             }
 
             int32_t newHetCfgIdx = nHetCfgs++;
             puschRxChEstHetCfg_t& hetCfg = hetCfgs[newHetCfgIdx];
             hetCfg.func = launchCfg.kernelNodeParamsDriver.func;
             hetCfg.nMaxPrb = nPrb;
             hetCfg.nMaxRxAnt = nRxAnt;
 
             dynDescrVecCpu[newHetCfgIdx].hetCfgUeGrpMap[hetCfg.nUeGrps] = ueGrpIdx;
             hetCfg.nUeGrps++;
 
 #ifdef ENABLE_DEBUG  
             NVLOGI_FMT(NVLOG_PUSCH, "{}: UE group {} -> HetCfg {} funcPtr {:p} (nHetCfgs {} nUeGrps {} nPrb {} nMaxPrb {} nRxAnt {} nLayers {} dmrsAddlnPos {} dmrsMaxLen {} nDmrsGridsPerPrb {})", __FUNCTION__, ueGrpIdx, newHetCfgIdx, static_cast<void*>(hetCfg.func), nHetCfgs, hetCfg.nUeGrps, nPrb, hetCfg.nMaxPrb, drvdUeGrpPrms.nRxAnt, drvdUeGrpPrms.nLayers, drvdUeGrpPrms.dmrsAddlnPos, drvdUeGrpPrms.dmrsMaxLen, drvdUeGrpPrms.nDmrsGridsPerPrb);
 #endif            
         }
     }
     return CUPHY_STATUS_SUCCESS;
 }
 
 cuphyStatus_t
 puschRxChEst::setup(cuphyPuschRxUeGrpPrms_t*              pDrvdUeGrpPrmsCpu,
                     cuphyPuschRxUeGrpPrms_t*              pDrvdUeGrpPrmsGpu,
                     uint16_t                              nUeGrps,
                     uint8_t                               enableDftSOfdm,
                     uint8_t*                              pPreEarlyHarqWaitKernelStatus_d,
                     uint8_t*                              pPostEarlyHarqWaitKernelStatus_d,
                     const uint16_t                        waitTimeOutPreEarlyHarqUs,
                     const uint16_t                        waitTimeOutPostEarlyHarqUs,
                     bool                                  enableCpuToGpuDescrAsyncCpy,
                     void**                                ppDynDescrsCpu,
                     void**                                ppDynDescrsGpu,
                     cuphyPuschRxChEstLaunchCfgs_t*        pLaunchCfgs,
                     uint8_t                               enableEarlyHarqProc,
                     cuphyPuschRxEarlyHarqWaitLaunchCfg_t* pLaunchCfgsPreEHQ,
                     cuphyPuschRxEarlyHarqWaitLaunchCfg_t* pLaunchCfgsPostEHQ,
                     cudaStream_t                          strm)

 {
    if(!pDrvdUeGrpPrmsCpu || !pDrvdUeGrpPrmsGpu || !ppDynDescrsCpu || !ppDynDescrsGpu || !pLaunchCfgs) return CUPHY_STATUS_INVALID_ARGUMENT;

    for(int32_t chEstTimeInstIdx = 0; chEstTimeInstIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEstTimeInstIdx)
    {
        if(!ppDynDescrsCpu[chEstTimeInstIdx] || !ppDynDescrsGpu[chEstTimeInstIdx]) return CUPHY_STATUS_INVALID_ARGUMENT;

        puschRxChEstDynDescrVec_t& dynDescrVecCpu = *(static_cast<ch_est::puschRxChEstDynDescrVec_t*>(ppDynDescrsCpu[chEstTimeInstIdx]));
        cuphyPuschRxChEstLaunchCfgs_t& launchCfgs = pLaunchCfgs[chEstTimeInstIdx];
        cuphyStatus_t status = batch(chEstTimeInstIdx,
                                     pDrvdUeGrpPrmsCpu,
                                     nUeGrps,
                                     enableDftSOfdm,
                                     launchCfgs.nCfgs,
                                     dynDescrVecCpu);
                                     
        if(CUPHY_STATUS_SUCCESS != status)
        {
            return status;
        }

        puschRxChEstDynDescr_t* pDynDescrVecGpu = static_cast<puschRxChEstDynDescr_t*>(ppDynDescrsGpu[chEstTimeInstIdx]);
        for(uint32_t hetCfgIdx = 0; hetCfgIdx < launchCfgs.nCfgs; ++hetCfgIdx)
        { 
            // Skip rest of the setup if there are no UE groups corresponding to the channel estimation time instance and hetCfg
            if(0 == m_hetCfgsArr[chEstTimeInstIdx][hetCfgIdx].nUeGrps) continue;

            // Setup descriptor in CPU memory
            puschRxChEstDynDescr_t& dynDescr   = dynDescrVecCpu[hetCfgIdx];
            puschRxChEstHetCfg_t const& hetCfg = m_hetCfgsArr[chEstTimeInstIdx][hetCfgIdx];
            dynDescr.chEstTimeInst             = chEstTimeInstIdx;
            dynDescr.pDrvdUeGrpPrms            = pDrvdUeGrpPrmsGpu;
            
            dynDescr.pPreEarlyHarqWaitKernelStatus_d = pPreEarlyHarqWaitKernelStatus_d;
            dynDescr.pPostEarlyHarqWaitKernelStatus_d= pPostEarlyHarqWaitKernelStatus_d;
            dynDescr.waitTimeOutPreEarlyHarqUs = waitTimeOutPreEarlyHarqUs;
            dynDescr.waitTimeOutPostEarlyHarqUs= waitTimeOutPostEarlyHarqUs;

 #ifdef ENABLE_DEBUG 
            NVLOGI_FMT(NVLOG_PUSCH, "{}: startPrb {} dmrsScId {}", __FUNCTION__, pDrvdUeGrpPrmsCpu[hetCfgIdx].startPrb, pDrvdUeGrpPrmsCpu[hetCfgIdx].dmrsScrmId);
 #endif // ENABLE_DEBUG

            puschRxChEstKernelArgs_t& kernelArgs = m_kernelArgsArr[chEstTimeInstIdx][hetCfgIdx];
            kernelArgs.pDynDescr = &pDynDescrVecGpu[hetCfgIdx];
    
            // Optional descriptor copy to GPU memory
            if(enableCpuToGpuDescrAsyncCpy)
            {
                CUDA_CHECK(cudaMemcpyAsync(&pDynDescrVecGpu[hetCfgIdx], &dynDescr, sizeof(puschRxChEstDynDescr_t), cudaMemcpyHostToDevice, strm));
            }
    
            // Select kernel
            cuphyPuschRxChEstLaunchCfg_t& launchCfg = launchCfgs.cfgs[hetCfgIdx];
  
            // TODO: Optimize function to determine kernel selection and launch geometry separately
            // TODO: for supporting per UE group layer and antenna count. Also per UE group DMRS config (nDmrsGridsPerPrb)
            int32_t ueGrpIdx = dynDescr.hetCfgUeGrpMap[0];
            cuphyPuschRxUeGrpPrms_t const& drvdUeGrpPrms = pDrvdUeGrpPrmsCpu[ueGrpIdx];            
            kernelSelectL2(hetCfg.nMaxRxAnt,
                           drvdUeGrpPrms.nLayers,
                           drvdUeGrpPrms.dmrsMaxLen,
                           drvdUeGrpPrms.nDmrsGridsPerPrb,
                           hetCfg.nMaxPrb,
                           1,
                           hetCfg.nUeGrps,
                           enableDftSOfdm,
                           drvdUeGrpPrms.tInfoDataRx.elemType,
                           drvdUeGrpPrms.tInfoHEst.elemType,
                           launchCfg);
                           
            if(hetCfg.func != launchCfg.kernelNodeParamsDriver.func)
            {
               NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: HetCfg {} (nUeGrps {} nMaxPrb {} nMaxRxAnt {} nLayers {} dmrsAddlnPos {} dmrsMaxLen {} nDmrsGridsPerPrb {})", __FUNCTION__, hetCfgIdx, hetCfg.nUeGrps, hetCfg.nMaxPrb, hetCfg.nMaxRxAnt, drvdUeGrpPrms.nLayers, drvdUeGrpPrms.dmrsAddlnPos, drvdUeGrpPrms.dmrsMaxLen, drvdUeGrpPrms.nDmrsGridsPerPrb);
               return CUPHY_STATUS_INTERNAL_ERROR;
            }
    
            launchCfg.kernelArgs[0] = &kernelArgs.pStatDescr;
            launchCfg.kernelArgs[1] = &kernelArgs.pDynDescr;
            
            launchCfg.kernelNodeParamsDriver.kernelParams = &(launchCfg.kernelArgs[0]);

            // pLaunchCfgsEHQ needs to be configured only once, hence running for hetCfgIdx == 0
            if (hetCfgIdx == 0)
            {
               // setup launch configs for wait kernel used in early HARQ processing
               if(enableEarlyHarqProc)
               {
                    CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pLaunchCfgsPreEHQ->kernelNodeParamsDriver;
                    // kernel input
                    pLaunchCfgsPreEHQ->kernelArgs[0] = &kernelArgs.pStatDescr;
                    pLaunchCfgsPreEHQ->kernelArgs[1] = &kernelArgs.pDynDescr;
                    kernelNodeParamsDriver.kernelParams = &(pLaunchCfgsPreEHQ->kernelArgs[0]);

                    // set and populate the remaining kernel parameters
                    dim3 gridDim(1);
                    dim3 blockDim(32);
                    //assumption is early HARQ bits are only on symbols 0-3
                    //!!ToDo consider cases where DMRS max length is 2
                    constexpr int HARQ_SYM_IDX_UB = 4; // = 3 + dmrsMaxLength
                    void* kernelFunc = reinterpret_cast<void*>(preEarlyHarqWaitKernel<HARQ_SYM_IDX_UB>);
                    cudaGetFuncBySymbol(&kernelNodeParamsDriver.func, kernelFunc);

                    kernelNodeParamsDriver.blockDimX = blockDim.x;
                    kernelNodeParamsDriver.blockDimY = blockDim.y;
                    kernelNodeParamsDriver.blockDimZ = blockDim.z;

                    kernelNodeParamsDriver.gridDimX = gridDim.x;
                    kernelNodeParamsDriver.gridDimY = gridDim.y;
                    kernelNodeParamsDriver.gridDimZ = gridDim.z;

                    kernelNodeParamsDriver.extra          = nullptr;
                    kernelNodeParamsDriver.sharedMemBytes = 0;
                    
                    /////////////////////////////////////////////
                    CUDA_KERNEL_NODE_PARAMS& kernelNodePostEHQParamsDriver = pLaunchCfgsPostEHQ->kernelNodeParamsDriver;
                    // kernel input
                    pLaunchCfgsPostEHQ->kernelArgs[0] = &kernelArgs.pStatDescr;
                    pLaunchCfgsPostEHQ->kernelArgs[1] = &kernelArgs.pDynDescr;
                    kernelNodePostEHQParamsDriver.kernelParams = &(pLaunchCfgsPostEHQ->kernelArgs[0]);

                    constexpr int POST_HARQ_SYM_IDX_UB = OFDM_SYMBOLS_PER_SLOT;
                    void* kernelFuncPostEHQ = reinterpret_cast<void*>(postEarlyHarqWaitKernel<POST_HARQ_SYM_IDX_UB>);
                    cudaGetFuncBySymbol(&kernelNodePostEHQParamsDriver.func, kernelFuncPostEHQ);

                    kernelNodePostEHQParamsDriver.blockDimX = blockDim.x;
                    kernelNodePostEHQParamsDriver.blockDimY = blockDim.y;
                    kernelNodePostEHQParamsDriver.blockDimZ = blockDim.z;

                    kernelNodePostEHQParamsDriver.gridDimX = gridDim.x;
                    kernelNodePostEHQParamsDriver.gridDimY = gridDim.y;
                    kernelNodePostEHQParamsDriver.gridDimZ = gridDim.z;

                    kernelNodePostEHQParamsDriver.extra          = nullptr;
                    kernelNodePostEHQParamsDriver.sharedMemBytes = 0;
                    
               }
            }
        }
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
 
 } // namespace ch_est
 
