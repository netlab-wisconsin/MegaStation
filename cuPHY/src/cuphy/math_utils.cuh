/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#pragma once

#include "cuphy.h"

namespace 
{
  ////////////////////////////////////////////////////////////////////////
  // complex_mul()
  template <typename T>
__forceinline__ __device__ T complex_mul(const T& a, const T& b)
  {
      // Use explicit cast to avoid warnings for narrowing. We assume the
      // caller knows the expected range is "safe".
      typedef typename scalar_from_complex<T>::type scalar_t;
      return T{static_cast<scalar_t>((a.x * b.x) - (a.y * b.y)),
              static_cast<scalar_t>((a.x * b.y) + (a.y * b.x))};
  }
  template <>
  __forceinline__ __device__ __half2 complex_mul(const __half2& a, const __half2& b)
  {
    __half2 c{0.0, 0.0};
    return __hcmadd(a, b, c);
  }

  ////////////////////////////////////////////////////////////////////////
  // real_mul() multiplies complex value with a real value
  template <typename T1, typename T2>
__forceinline__ __device__ T1 real_mul(const T1& a, const T2& b)
  {
      // Use explicit cast to avoid warnings for narrowing. We assume the
      // caller knows the expected range is "safe".
      using scalar_t = typename scalar_from_complex<T1>::type;
      return T1{static_cast<scalar_t>(a.x * b),
              static_cast<scalar_t>(a.y * b)};
  }
  template <>
  __forceinline__ __device__ __half2 real_mul(const __half2& a, const __half& b)
  {
      return __halves2half2(a.x*b,a.y*b);
  }

  ////////////////////////////////////////////////////////////////////////
  // complex_conjmul()
  template <typename T>
  __forceinline__ __device__ T complex_conjmul(const T& a, const T& bc)
  {
      // Use explicit cast to avoid warnings for narrowing. We assume the
      // caller knows the expected range is "safe".
      typedef typename scalar_from_complex<T>::type scalar_t;
      return T{static_cast<scalar_t>((a.x * bc.x) + (a.y * bc.y)),
               static_cast<scalar_t>((a.y * bc.x) - (a.x * bc.y))};
  }
  template <>
  __forceinline__ __device__ __half2 complex_conjmul(const __half2& a, const __half2& bc)
  {
      // (a.re, a.im) * (bc.re, -bc.im)
      // acc.re =  (a.re*b.re) + a.im*b.im
      // acc.im = -(a.re*b.im) + a.im*b.re
      const __half2 a_im   = __half2half2(a.y);
      const __half2 b_swap = __lowhigh2highlow(bc);
      __half2 acc          = __hmul2(a_im, b_swap);
      const __half2 a_re   = __half2(a.x,__hneg(a.x));
      acc                  = __hfma2(a_re, bc, acc);
      return acc;
  }

  ////////////////////////////////////////////////////////////////////////
  // complex_add()
  template <typename T>
  __device__ T complex_add(const T& a, const T& b)
  {
      // Use explicit cast to avoid warnings for narrowing. We assume the
      // caller knows the expected range is "safe".
      typedef typename scalar_from_complex<T>::type scalar_t;

      return T{static_cast<scalar_t>(a.x + b.x), static_cast<scalar_t>(a.y + b.y)};
  }

  template <>
  __device__ __half2 complex_add(const __half2& a, const __half2& b)
  {
      return __hadd2(a, b);
  }

  ////////////////////////////////////////////////////////////////////////
  //least common multiple and greatest common divisor
  template <typename T>
  constexpr __host__ __device__ uint32_t compute_gcd(T m, T n)
  {
      static_assert(std::is_unsigned_v<T>, "only unsigned values supported");
      if (n == 0) return m;
      return compute_gcd(n, m % n);
  }

  template <typename T>
  constexpr __host__ __device__ uint32_t compute_lcm(T m, T n)
  {
      static_assert(std::is_unsigned_v<T>, "only unsigned values supported");
      //return std::lcm(m, n); //nvcc shipped with CUDA 12.2 gives compiler error due to not handling constexpr input properly
      return (m * n) / compute_gcd(m, n);
  }

};
