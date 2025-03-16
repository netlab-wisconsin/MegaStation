/**
 * @file decoder.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief LDPC Decoder with cuPHY
 * @version 0.1
 * @date 2023-11-16
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cuda_runtime.h>
#include <cuphy.h>

#include <memory>
#include <vector>

#include "config.h"
#include "matrix/cuphy_tensor.h"

namespace mega {
/**
 * @brief LDPC Decoder Wrapper of cuPHY's LDPC decoder
 *
 */
class LDPCDecoder {
 private:
  std::shared_ptr<cuphyContext> ctx;          //!< cuPHY context
  std::shared_ptr<cuphyLDPCDecoder> decoder;  //!< cuPHY LDPC decoder
  cuphyLDPCDecodeConfigDesc_t
      config_desc;  //!< cuPHY LDPC decoder config descriptor

 public:
  /**
   * @brief Default constructor of LDPCDecoder
   *
   */
  LDPCDecoder() = default;
  /**
   * @brief Construct a new LDPCDecoder object
   *
   * @param llr_type_ Data type of LLRs
   * @param config LDPC configuration
   */
  LDPCDecoder(const CuphyTensor::cuphy_dtype_t llr_type_,
              const LDPCConfig& config) {
    cuphyContext_t ctx_;
    cuphyLDPCDecoder_t decoder_;
    cuphyCreateContext(&ctx_, 0);
    cuphyCreateLDPCDecoder(ctx_, &decoder_, 0);

    ctx = std::shared_ptr<cuphyContext>(ctx_, cuphyDestroyContext);
    decoder =
        std::shared_ptr<cuphyLDPCDecoder>(decoder_, cuphyDestroyLDPCDecoder);

    config_desc.llr_type = (cuphyDataType_t)llr_type_;
    config_desc.num_parity_nodes = config.parity_nodes;
    config_desc.Z = config.lifting_factor;
    config_desc.max_iterations = config.max_decode_iter;
    config_desc.Kb = config.info_nodes;
    config_desc.flags = config.max_decode_iter > 2
                            ? CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT
                            : CUPHY_LDPC_DECODE_DEFAULT;
    config_desc.BG = config.base_graph;
    config_desc.algo = 0;             // Automatically choose the best algorithm
    config_desc.workspace = nullptr;  // of no use now

    cuphyErrorCorrectionLDPCDecodeSetNormalization(decoder.get(), &config_desc);
  }

  /**
   * @brief Decode LLRs to bits
   *
   * @param tLLRs (half/float)[Zc * (Kb + mb), batch_count]\nNeeds to include
   * prefix puntured bits (zero out)
   * @param tOut  (bit)[Zc * Kb, batch_count]
   * @param stream cuda stream
   */
  void operator()(const CuphyTensor& tLLRs, const CuphyTensor& tOut,
                  cudaStream_t stream = nullptr) {
    cuphyErrorCorrectionLDPCDecode(decoder.get(), tOut.get_desc(), tOut.ptr(),
                                   tLLRs.get_desc(), tLLRs.ptr(), &config_desc,
                                   stream);
  }
};

class DecoderFactory {
 private:
  static std::vector<LDPCDecoder> decoders;  //!< LDPC decoders for each device

 public:
  /**
   * @brief Initialize LDPCDecoder
   *
   * @param llr_type_ Data type of LLRs
   * @param config LDPC configuration
   * @return LDPCDecoder
   */
  static void init(const CuphyTensor::cuphy_dtype_t llr_type_,
                   const LDPCConfig& config, uint8_t num_devices);
  static void decode(const CuphyTensor& tLLRs, const CuphyTensor& tOut,
                     cudaStream_t stream = nullptr);
  static void destroy();
};

}  // namespace mega