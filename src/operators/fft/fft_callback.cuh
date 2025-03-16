/**
 * @file fft_callback.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief fft callback functions
 * @version 0.1
 * @date 2023-12-10
 *
 * @copyright Copyright (c) 2023
 *
 */
#pragma once

#include <stdint.h>

#include "fft_utils.h"
#include "mega_complex.h"

namespace mega {

template <typename QuanT>
__global__ void fft_load_kernel(const QuanT *in, Complex *out,
                                const struct baseInfo fftInfo) {
  const uint32_t batch_count = fftInfo.bsAnt;
  const uint32_t num_element = fftInfo.ofdmCAnum;

  for (uint64_t batch_idx = blockIdx.x; batch_idx < batch_count;
       batch_idx += gridDim.x) {
    for (uint64_t idx = threadIdx.x; idx < num_element; idx += blockDim.x) {
      uint64_t offset = batch_idx * num_element + idx;

      uint64_t in_offset = offset << 1;
      QuanT re_quant = in[in_offset];
      QuanT im_quant = in[in_offset + 1];

      float re = float(re_quant) / kQuantFloat<QuanT>();
      float im = float(im_quant) / kQuantFloat<QuanT>();

      out[offset] = {re, im};
    }
  }
}

__global__ void __launch_bounds__(1024)
    fft_store_uplink_kernel(const Complex *in, Complex *out,
                            const struct baseInfo fftInfo) {
  const uint32_t batch_count = fftInfo.bsAnt;
  const uint32_t cyclic_shift = fftInfo.ofdmCAnum / 2;

  for (uint64_t bsAnt_id = blockIdx.x; bsAnt_id < batch_count;
       bsAnt_id += gridDim.x) {
    for (uint64_t idx = threadIdx.x; idx < fftInfo.ofdmNum; idx += blockDim.x) {
      int64_t carrier_offset =
          (idx + fftInfo.ofdmStart + cyclic_shift) % fftInfo.ofdmCAnum;

      out[bsAnt_id * fftInfo.ofdmNum + idx] =
          in[bsAnt_id * fftInfo.ofdmCAnum + carrier_offset];
    }
  }
  /**** old performant code with transpose
  const uint32_t size = fftInfo.bsAnt * fftInfo.ofdmNum;
  const uint32_t stride = gridDim.x * blockDim.x;
  const uint32_t cyclic_shift = fftInfo.ofdmCAnum / 2;

  const uint64_t thr_id = blockIdx.x * blockDim.x + threadIdx.x;

  for (uint64_t idx = thr_id; idx < size; idx += stride) {
    uint64_t bsAnt_id = idx % fftInfo.bsAnt;
    uint64_t carrier_offset =
        (idx / fftInfo.bsAnt + fftInfo.ofdmStart + cyclic_shift) %
        fftInfo.ofdmCAnum;
    out[idx] = in[bsAnt_id * fftInfo.ofdmCAnum + carrier_offset];
  }
  ****/
}

__global__ void __launch_bounds__(1024)
    fft_store_pilot_kernel(const Complex *in, Complex *out,
                           const struct pilotInfo fftInfo) {
  const uint32_t cyclic_shift = fftInfo.ofdmCAnum / 2;

  for (uint64_t bsAnt_id = blockIdx.x; bsAnt_id < fftInfo.bsAnt;
       bsAnt_id += gridDim.x) {
    for (uint64_t idx = threadIdx.x; idx < fftInfo.ofdmNum; idx += blockDim.x) {
      int64_t carrier_offset =
          (idx + fftInfo.ofdmStart + cyclic_shift) % fftInfo.ofdmCAnum;

      Complex pilot = fftInfo.pilotSign[idx];
      Complex element = in[bsAnt_id * fftInfo.ofdmCAnum + carrier_offset];
      element = {
          element.re * pilot.re + element.im * pilot.im,
          element.re * pilot.im - element.im * pilot.re,
      };

      uint64_t cg_offset = idx / fftInfo.scGroup;
      uint64_t ue_offset = fftInfo.ueStart + idx % fftInfo.scGroup;

      if (ue_offset >= fftInfo.ueAnt) {
        continue;
      }

      uint64_t block_size = fftInfo.bsAnt * fftInfo.ueAnt;
      uint64_t block_start = cg_offset * block_size;
      uint64_t block_offset = bsAnt_id * fftInfo.ueAnt + ue_offset;

      out[block_start + block_offset] = element;
    }
  }
}

}  // namespace mega