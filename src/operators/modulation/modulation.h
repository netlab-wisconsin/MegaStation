/**
 * @file modulation.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Initialize modulation table and launch modulation kernel
 * @version 0.1
 * @date 2023-11-26
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <vector>

#include "matrix/matrix.h"
#include "mega_complex.h"

namespace mega {

struct ModParams {
  uint8_t order;               //!< Modulation order (M = 2^order)
  const Complex* pilot_table;  //!< Pilot table for DM-RS
  uint8_t pilot_spacing;       //!< Pilot spacing for DM-RS
};

class Modulation {
 public:
  static constexpr uint16_t kMaxM = 256;  //!< Maximum modulation number `M`QAM
 private:
  static std::vector<Complex*> pilot_tables;  //!< Pilot tables for each device
  static inline Complex modTable_cpu[kMaxM];  //!< CPU modulation table
  static ModParams mod_params;                //!< Modulation parameters

  static void initQPSK();
  static void init16QAM();
  static void init64QAM();
  static void init256QAM();

  /**
   * @brief Launch modulation kernel
   *
   * @param in input bits data
   * @param out output complex modulation data
   * @param in_bytes the number of bytes in input data
   * @param num_carriers number of OFDM symbols of one UE
   * @param num_ues number of UEs
   * @param stream CUDA stream
   */
  static void modulate(const uint8_t* in, Complex* out, uint64_t in_bytes,
                       uint64_t num_carriers, uint64_t num_ues,
                       cudaStream_t stream = nullptr);

 public:
  /**
   * @brief Initialize modulation table and modulation parameters
   *
   * @param order Modulation order (M = 2^order) for `ModParams`
   * @param seq_len Sequence length of Zadoff-Chu sequence for generating
   * `pilot_table` of `ModParams`
   * @param num_ues Number of UEs for `ModParams`
   * @param pilot_spacing Pilot spacing for DM-RS for `ModParams`
   * @param num_device Number of devices
   */
  static void init(const uint8_t order, const uint64_t seq_len,
                   const uint64_t num_ues, const uint8_t pilot_spacing,
                   const uint8_t num_device = 1);

  /**
   * @brief Launch modulation kernel
   *
   * @param in input bits data (input_bits * num_ues, column-major)
   * @param out output complex modulation data (num_ues * num_carriers,
   * column-major)
   * @param stream CUDA stream
   */
  static void modulate(const Matrix& in, const Matrix& out,
                       cudaStream_t stream = nullptr);

  /**
   * @brief Free Pilot Table
   *
   */
  static inline void destroy() {
    if (mod_params.pilot_table != nullptr) {
      cudaFree(const_cast<void*>((const void*)mod_params.pilot_table));
      mod_params.pilot_table = nullptr;
    }
  }
};

}  // namespace mega