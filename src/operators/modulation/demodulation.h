/**
 * @file demodulation.cuh
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Demodulation function pointers for both CPU and GPU
 * @version 0.1
 * @date 2023-11-26
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cuda_fp16.h>

#include "mega_complex.h"

// scale factors for demodulation
#define SCALE_BYTE_QPSK 20
#define SCALE_BYTE_QAM16 100
#define SCALE_BYTE_QAM64 100
#define SCALE_BYTE_QAM256 100

namespace mega {

#ifndef __CUDA_ARCH__ /* host cannot find __habs */
#define __habs(x) ((x) < half(0.0) ? half(-(x)) : (x))
#endif

__device__ __host__ inline void demodQPSK(const Complex &c, half llr[]) {
  llr[0] = half(c.re * -SCALE_BYTE_QPSK * sqrtf(2.f));
  llr[1] = half(c.im * -SCALE_BYTE_QPSK * sqrtf(2.f));
}

__device__ __host__ inline void demod16QAM(const Complex &c, half llr[]) {
  llr[0] = half(SCALE_BYTE_QAM16 * c.re);
  llr[1] = half(SCALE_BYTE_QAM16 * c.im);
  llr[2] = half(2 * SCALE_BYTE_QAM16 / sqrtf(10.f)) - __habs(llr[0]);
  llr[3] = half(2 * SCALE_BYTE_QAM16 / sqrtf(10.f)) - __habs(llr[1]);
}

__device__ __host__ inline void demod64QAM(const Complex &c, half llr[]) {
  const half t1 = half(4 * SCALE_BYTE_QAM64 / sqrtf(42.f));
  const half t2 = half(2 * SCALE_BYTE_QAM64 / sqrtf(42.f));

  llr[0] = half(SCALE_BYTE_QAM64 * c.re);
  llr[1] = half(SCALE_BYTE_QAM64 * c.im);
  llr[2] = t1 - __habs(llr[0]);
  llr[3] = t1 - __habs(llr[1]);
  llr[4] = t2 - __habs(llr[2]);
  llr[5] = t2 - __habs(llr[3]);
}

__device__ __host__ inline void demod256QAM(const Complex &c, half llr[]) {
  const half t1 = half(8 * SCALE_BYTE_QAM256 / sqrtf(170.f));
  const half t2 = half(4 * SCALE_BYTE_QAM256 / sqrtf(170.f));
  const half t3 = half(2 * SCALE_BYTE_QAM256 / sqrtf(170.f));

  llr[0] = half(SCALE_BYTE_QAM256 * c.re);
  llr[1] = half(SCALE_BYTE_QAM256 * c.im);
  llr[2] = t1 - __habs(llr[0]);
  llr[3] = t1 - __habs(llr[1]);
  llr[4] = t2 - __habs(llr[2]);
  llr[5] = t2 - __habs(llr[3]);
  llr[6] = t3 - __habs(llr[4]);
  llr[7] = t3 - __habs(llr[5]);
}

/**
 * @brief Demodulation function pointers for both CPU and GPU
 *
 * @param c input complex number
 * @param llr output llr (soft bits / log likelihood ratio)
 */
typedef void (*demodPtr)(const Complex &, half[]);

}  // namespace mega