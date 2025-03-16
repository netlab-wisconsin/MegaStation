/**
 * @file config.cc
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Configuration Initialization
 * @version 0.1
 * @date 2023-12-16
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "config.h"

#include <algorithm>
#include <vector>

#include "utils/consts.h"

mega::Config mega::gconfig;  //!< global configuration definition

using namespace mega;

Config::Config(uint32_t antennas_, uint32_t users_, uint64_t ofdm_data_,
               uint64_t ofdm_ca_, uint32_t sc_group_, std::string frame_,
               uint8_t mod_order_, double code_rate_, uint16_t base_graph_,
               uint16_t max_iter_, uint32_t pilot_spacing_)
    : antennas(antennas_),
      users(users_),
      ofdm_data(ofdm_data_),
      ofdm_ca(ofdm_ca_),
      ofdm_start((ofdm_ca_ - ofdm_data_) / 2),
      sc_group(sc_group_),
      mod_order(mod_order_),
      pilot_spacing(pilot_spacing_) {
  symbols = frame_.size();
  frame.frame = frame_;
  frame.pilot_syms = 0;
  frame.uplink_syms = 0;
  frame.downlink_syms = 0;
  frame.gidx_sym.resize(symbols);
  frame.gidx_lidx.resize(symbols);

  frame.pilots.reserve(symbols);

  for (uint16_t symbol_id = 0; symbol_id < symbols; symbol_id++) {
    switch (frame_[symbol_id]) {
      case 'P':
        frame.gidx_sym[symbol_id] = Symbol::NLPilot;
        frame.gidx_lidx[symbol_id] = frame.pilot_syms;
        frame.pilot_syms++;
        frame.pilots.push_back(symbol_id);
        break;
      case 'U':
        frame.gidx_sym[symbol_id] = Symbol::Uplink;
        frame.gidx_lidx[symbol_id] = frame.uplink_syms;
        frame.uplink_syms++;
        break;
      case 'D':
        frame.gidx_sym[symbol_id] = Symbol::Downlink;
        frame.gidx_lidx[symbol_id] = frame.downlink_syms;
        frame.downlink_syms++;
        break;
      default:
        break;
    }
  }
  frame.gidx_sym[frame.pilots.back()] = Symbol::Pilot;

  if (sc_group_ > 32) {
    spdlog::error("Subcarrier group size must be less than or equal to 32");
    throw std::runtime_error(
        "Subcarrier group size must be less than or equal to 32");
  }

  if (users_ > sc_group_ * frame.pilot_syms) {
    spdlog::error(
        "Number of users must be less than or equal to subcarrier "
        "group size x # of pilot symbols");
    throw std::runtime_error(
        "Number of users must be less than or equal to subcarrier group size x "
        "# of pilot symbols");
  }

  // Set of LDPC lifting size Zc, from TS38.212 Table 5.3.2-1
  std::vector<uint64_t> zc_vec = {
      2,   4,   8,   16, 32, 64,  128, 256, 3,   6,   12,  24, 48,
      96,  192, 384, 5,  10, 20,  40,  80,  160, 320, 7,   14, 28,
      56,  112, 224, 9,  18, 36,  72,  144, 288, 11,  22,  44, 88,
      176, 352, 13,  26, 52, 104, 208, 15,  30,  60,  120, 240};
  std::sort(zc_vec.begin(), zc_vec.end());

  uint64_t uncoded_bits =
      static_cast<uint64_t>(ofdm_data * mod_order * code_rate_);

  uint64_t zc = zc_vec.back();
  for (uint32_t i = 0; i < zc_vec.size(); i++) {
    if ((LDPCConfig::num_decoded_bits(base_graph_, zc_vec[i]) * kCbPerSymbol <=
         uncoded_bits) &&
        (LDPCConfig::num_decoded_bits(base_graph_, zc_vec[i + 1]) *
             kCbPerSymbol >
         uncoded_bits)) {
      zc = zc_vec[i];
      break;
    }
  }

  uint64_t num_rows =
      static_cast<uint64_t>(
          std::round(LDPCConfig::num_input_cols(base_graph_) / code_rate_)) -
      (LDPCConfig::num_input_cols(base_graph_) - 2);

  ldpc_uconfig = LDPCConfig(num_rows, base_graph_, zc, max_iter_);

  uncoded_bits = static_cast<uint64_t>(
      (ofdm_data - static_cast<uint64_t>(ofdm_data / pilot_spacing)) *
      mod_order * code_rate_);

  zc = zc_vec.back();
  for (uint32_t i = 0; i < zc_vec.size(); i++) {
    if ((LDPCConfig::num_decoded_bits(base_graph_, zc_vec[i]) * kCbPerSymbol <=
         uncoded_bits) &&
        (LDPCConfig::num_decoded_bits(base_graph_, zc_vec[i + 1]) *
             kCbPerSymbol >
         uncoded_bits)) {
      zc = zc_vec[i];
      break;
    }
  }

  num_rows =
      static_cast<uint64_t>(
          std::round(LDPCConfig::num_input_cols(base_graph_) / code_rate_)) -
      (LDPCConfig::num_input_cols(base_graph_) - LDPCConfig::num_punc_cols());

  ldpc_dconfig = LDPCConfig(num_rows, base_graph_, zc, max_iter_);

  int dev_count;
  cudaGetDeviceCount(&dev_count);
  num_devices = static_cast<uint16_t>(dev_count);

  running = true;

  // Report configuration
  spdlog::info("Number of Devices: {}", num_devices);
  spdlog::info("Basestation Antennas: {}", antennas);
  spdlog::info("User Antennas: {}", users);
  spdlog::info("Number of Symbols: {}", symbols);
  spdlog::info("OFDM Data Size: {}", ofdm_data);
  spdlog::info("OFDM CA Size: {}", ofdm_ca);
  spdlog::info("OFDM Start Postion: {}", ofdm_start);
  spdlog::info("Subcarrier Group Size: {}", sc_group);
  spdlog::info("Frame: {}", frame.frame);
  spdlog::info("Pilot Symbols: {}", frame.pilot_syms);
  spdlog::info("Uplink Symbols: {}", frame.uplink_syms);
  spdlog::info("Downlink Symbols: {}", frame.downlink_syms);
  spdlog::info("Modulation Order: {}", mod_order);
  spdlog::info("Code Rate: {}", code_rate_);
  spdlog::info("Uplink & Downlink LDPC Base Graph: {}", base_graph_);
  spdlog::info("Uplink & Downlink LDPC Max Iterations: {}", max_iter_);
  spdlog::info("Uplink LDPC Parity Nodes: {}", ldpc_uconfig.parity_nodes);
  spdlog::info("Uplink LDPC Lifting Factor: {}", ldpc_uconfig.lifting_factor);
  spdlog::info("Uplink LDPC Info Nodes: {}", ldpc_uconfig.info_nodes);
  spdlog::info("Uplink LDPC Encoded Bits: {}", ldpc_uconfig.encoded_bits);
  spdlog::info("Uplink LDPC Decoded Bits: {}", ldpc_uconfig.decoded_bits);
  spdlog::info("Uplink LDPC Punctured Bits: {}", ldpc_uconfig.punctured_bits);
  spdlog::info("Downlink LDPC Parity Nodes: {}", ldpc_dconfig.parity_nodes);
  spdlog::info("Downlink LDPC Lifting Factor: {}", ldpc_dconfig.lifting_factor);
  spdlog::info("Downlink LDPC Info Nodes: {}", ldpc_dconfig.info_nodes);
  spdlog::info("Downlink LDPC Encoded Bits: {}", ldpc_dconfig.encoded_bits);
  spdlog::info("Downlink LDPC Decoded Bits: {}", ldpc_dconfig.decoded_bits);
  spdlog::info("Downlink LDPC Punctured Bits: {}", ldpc_dconfig.punctured_bits);
}

void Config::open_peer_access() {
  for (uint32_t i = 0; i < num_devices; i++) {
    cudaSetDevice(i);
    for (uint32_t j = 0; j < num_devices; j++) {
      if (i != j) {
        int can_access_peer = 0;
        cudaDeviceCanAccessPeer(&can_access_peer, i, j);
        spdlog::info("{} Can access peer {}: {}", i, j, bool(can_access_peer));
        if (can_access_peer) cudaDeviceEnablePeerAccess(j, 0);
      }
    }
  }
}