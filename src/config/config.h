/**
 * @file config.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Global configuration
 * @version 0.1
 * @date 2023-12-16
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <spdlog/spdlog.h>

#include <cstdint>
#include <string>

#include "ldpc/config.h"

namespace mega {

/**
 * @brief Global configuration
 *
 */
struct Config {
  uint32_t antennas;    //!< number of base station antennas
  uint32_t users;       //!< number of user equipment antennas
  uint32_t symbols;     //!< number of symbols
  uint64_t ofdm_data;   //!< number of OFDM data
  uint64_t ofdm_ca;     //!< number of OFDM channel access
  uint64_t ofdm_start;  //!< valid OFDM start point
  uint32_t sc_group;    //!< size of subcarrier groups

  LDPCConfig ldpc_uconfig;  //!< LDPC uplink configuration
  LDPCConfig ldpc_dconfig;  //!< LDPC downlink configuration

  uint8_t mod_order;       //!< modulation order
  uint32_t pilot_spacing;  //!< pilot spacing

  enum Symbol : uint8_t {
    NLPilot,  //!< not last pilot
    Pilot,    //!< last pilot
    Uplink,
    Downlink,
  };

  //!< frame configuration
  struct Frame {
    std::string frame;  //!< frame symbols representation

    uint32_t pilot_syms;     //!< number of pilot symbols
    uint32_t uplink_syms;    //!< number of uplink symbols
    uint32_t downlink_syms;  //!< number of downlink symbols

    std::vector<Symbol> gidx_sym;     //!< Index in global frame to symbol type
    std::vector<uint32_t> gidx_lidx;  //!< Index in global frame to local index
                                      //!< of each symbol type map

    std::vector<uint16_t> pilots;  //!< pilot symbols global index
  } frame;

  uint16_t num_devices;  //!< number of devices

  bool running;  //!< running status

  Config() = default;
  Config(uint32_t antennas_, uint32_t users_, uint64_t ofdm_data_,
         uint64_t ofdm_ca_, uint32_t sc_group_, std::string frame_,
         uint8_t mod_order_, double code_rate_, uint16_t base_graph_,
         uint16_t max_iter_, uint32_t pilot_spacing_);

  void open_peer_access();
};

extern Config gconfig;  //!< global configuration declaration

}  // namespace mega