/**
 * @file config.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Configurations for LDPC
 * @version 0.1
 * @date 2023-11-16
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cstdint>

#include "cuphy.h"

namespace mega {

/**
 * @brief LDPC configuration for cuphy
 * used both in encoder and decoder.\n
 * Should stay constant after initialization
 * since some of the parameters are computed from others
 *
 */
struct LDPCConfig {
  uint16_t parity_nodes;     //!< nRows/mb
  uint16_t base_graph;       //!< BG: 1 or 2
  uint16_t lifting_factor;   //!< Zc
  uint16_t max_decode_iter;  //!< maximum number of decoding iterations
  uint16_t info_nodes;       //!< info nodes (Kb)

  uint64_t encoded_bits;    //!< number of bits after encoding
  uint64_t decoded_bits;    //!< number of bits after decoding / input bits
  uint64_t punctured_bits;  //!< number of punctured bits

  static inline uint16_t num_input_cols(uint16_t base_graph) {
    return base_graph == 1 ? CUPHY_LDPC_BG1_INFO_NODES
                           : CUPHY_LDPC_MAX_BG2_INFO_NODES;
  }

  static inline uint64_t num_decoded_bits(uint16_t base_graph,
                                          uint16_t lifting_factor) {
    return num_input_cols(base_graph) * uint64_t(lifting_factor);
  }

  static inline constexpr uint16_t num_punc_cols() {
    return CUPHY_LDPC_NUM_PUNCTURED_NODES;
  }

  LDPCConfig() = default;
  /**
   * @brief Construct a new LDPCConfig object
   *
   * @param mb_ number of parity nodes (nRows)
   * @param bg_ base graph
   * @param zc_ lifting factor
   * @param md_iter_ maximum number of decoding iterations
   */
  LDPCConfig(uint16_t mb_, uint16_t bg_, uint16_t zc_, uint16_t md_iter_)
      : parity_nodes(mb_),
        base_graph(bg_),
        lifting_factor(zc_),
        max_decode_iter(md_iter_) {
    info_nodes = num_input_cols(base_graph);
    encoded_bits =
        (info_nodes - CUPHY_LDPC_NUM_PUNCTURED_NODES + parity_nodes) *
        uint64_t(lifting_factor);
    decoded_bits = info_nodes * uint64_t(lifting_factor);
    punctured_bits = CUPHY_LDPC_NUM_PUNCTURED_NODES * uint64_t(lifting_factor);
  }
};

}  // namespace mega