/**
 * @file modulation_kernel.cuh
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Modulation kernel for the GPU
 * @version 0.1
 * @date 2023-11-26
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include "mega_complex.h"
#include "modulation.h"

namespace mega {

__constant__ float2
    modTable[Modulation::kMaxM];  //!< Modulation table in GPU constant memory
                                  // Can't use Complex type here, because
                                  // Complex is not a POD type

/**
 * @brief Modulation kernel, together with the pilot filling for DM-RS
 *
 * @param in input bits to be modulated
 * @param out output complex symbols
 * @param mod_params modulation parameters
 * @param in_bytes input bytes
 * @param num_carriers number of subcarriers of one UE (OFDM symbols)
 * @param num_ues batch count (number of UEs)
 */
__global__ void modulate_kernel(const uint8_t *in, Complex *out,
                                ModParams mod_params, uint64_t in_bytes,
                                uint64_t num_carriers, uint64_t num_ues) {
  for (int ue_id = blockIdx.z; ue_id < num_ues;  // UE index
       ue_id += gridDim.z) {
    uint64_t carrier_id =
        blockIdx.x * blockDim.x + threadIdx.x;  // subcarrier index
    uint64_t shrink_idx =
        mod_params.pilot_spacing
            ? carrier_id - ((carrier_id / mod_params.pilot_spacing) + 1)
            : carrier_id;

    const uint8_t *input_data_ptr = in + ue_id * in_bytes;
    Complex *output_data_ptr = out + ue_id * num_carriers;

    uint8_t mod_byte = 0;

    for (int i = 0, mod_idx = shrink_idx * mod_params.order;
         i < mod_params.order && (mod_idx / 8) < in_bytes; i++, mod_idx++) {
      mod_byte = ((input_data_ptr[mod_idx / 8] >> (mod_idx % 8)) & 0x1) |
                 (mod_byte << 1);
    }

    if (carrier_id < num_carriers) {
      output_data_ptr[carrier_id] =
          mod_params.pilot_spacing &&
                  (carrier_id % mod_params.pilot_spacing) == 0
              ? mod_params.pilot_table[ue_id * num_carriers + carrier_id]
              : (Complex)modTable[mod_byte];
    }
  }
}

/**
 * @brief Initialize the modulation table in GPU constant memory
 *
 * @param modTable_cpu modulation table in CPU memory
 * @param stream CUDA stream
 */
__host__ void init_modulation_cuda_table(Complex *modTable_cpu,
                                         cudaStream_t stream = nullptr) {
  cudaMemcpyToSymbolAsync(modTable, modTable_cpu,
                          Modulation::kMaxM * sizeof(Complex), 0,
                          cudaMemcpyHostToDevice, stream);
}

}  // namespace mega