/**
 * @file ifft_callback.cuh
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief ifft callback functions
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
__global__ void ifft_store_downlink_kernel(const Complex *in, QuanT *out,
                                           const struct baseInfo ifftInfo) {
  const uint32_t num_element = ifftInfo.ofdmCAnum;
  const uint32_t batch_count = ifftInfo.bsAnt;

  for (uint64_t batch_idx = blockIdx.x; batch_idx < batch_count;
       batch_idx += gridDim.x) {
    for (uint64_t idx = threadIdx.x; idx < num_element; idx += blockDim.x) {
      uint64_t offset = batch_idx * num_element + idx;

      Complex element = in[offset];

      float re = element.re;
      float im = element.im;

      float re_scaled = re * (kQuantFloat<QuanT>() / ifftInfo.ofdmCAnum);
      float im_scaled = im * (kQuantFloat<QuanT>() / ifftInfo.ofdmCAnum);

      int re_quant = __float2int_rn(re_scaled);
      int im_quant = __float2int_rn(im_scaled);

      if (re_scaled > std::numeric_limits<QuanT>::max()) {
        re_quant = std::numeric_limits<QuanT>::max();
      } else if (re_scaled < std::numeric_limits<QuanT>::min()) {
        re_quant = std::numeric_limits<QuanT>::min();
      }
      if (im_scaled > std::numeric_limits<QuanT>::max()) {
        im_quant = std::numeric_limits<QuanT>::max();
      } else if (im_scaled < std::numeric_limits<QuanT>::min()) {
        im_quant = std::numeric_limits<QuanT>::min();
      }

      uint64_t out_offset = offset << 1;  // offset * 2
      out[out_offset] = re_quant;
      out[out_offset + 1] = im_quant;
    }
  }
}

}  // namespace mega