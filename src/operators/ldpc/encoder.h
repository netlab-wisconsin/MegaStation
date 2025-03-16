/**
 * @file encoder.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief LDPC encoder with cuphy
 * @version 0.1
 * @date 2023-11-16
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cuda_runtime.h>
#include <cuphy.h>

#include <stdexcept>

#include "config.h"
#include "matrix/cuphy_tensor.h"

namespace mega {

/**
 * @brief LDPC Encoder Wrapper of cuPHY's LDPC encoder
 *
 */
class LDPCEncoder {
 private:
  LDPCConfig config;  //!< LDPC configuration
                      /*
                       * tIn Input bits (bit)[Zc * Kb, batch_count]
                       * tOut Output bits (bit)[Zc * (Kb + mb - punc), batch_count]
                       */

 public:
  /**
   * @brief Default constructor of LDPCEncoder
   *
   */
  LDPCEncoder() = default;
  /**
   * @brief Construct a new LDPCEncoder object
   *
   * @param config_ LDPC configuration
   * @param max_batch_count_ Maximum Batch count
   */
  LDPCEncoder(const LDPCConfig &config_) : config(config_) {}

  /**
   * @brief Encode bits to LLRs
   *
   * @param stream CUDA stream
   */
  void operator()(const CuphyTensor &tIn, const CuphyTensor &tOut,
                  cudaStream_t stream = nullptr) {
    if (tIn.dim(1) != tOut.dim(1) || tIn.nDim() > 2 || tOut.nDim() > 2) {
      throw std::invalid_argument(
          "Input and output tensor are 1 or 2 dimensional and must have the "
          "same batch size (dim(1))");
    }

    int batch_count = tIn.nDim() == 1 ? 0 : tIn.dim(1);

    cuphySetupLDPCEncode(
        batch_count ? tIn[0].get_desc() : tIn.get_desc(), tIn.ptr(),
        batch_count ? tOut[0].get_desc() : tOut.get_desc(), tOut.ptr(),
        config.base_graph,      // base graph
        config.lifting_factor,  // lifting size
        true,                   // puncture output bits
        config.parity_nodes,    // max parity nodes
        0,                      // redundancy version
        batch_count,            // batching
        batch_count,            // batch count
        tIn.szBytes(tIn.nDim() - 1), tOut.szBytes(tOut.nDim() - 1), stream);
  }
};

class EncoderFactory {
 private:
  static LDPCEncoder encoder;

 public:
  static void init(const LDPCConfig &config);
  static inline void destroy(){};
  static void encode(const CuphyTensor &tIn, const CuphyTensor &tOut,
                     cudaStream_t stream = nullptr);
};

}  // namespace mega