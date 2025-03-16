/**
 * @file decoder.cc
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Decoder Factory implementation
 * @version 0.1
 * @date 2024-04-08
 *
 * @copyright Copyright (c) 2024
 *
 */
#include "decoder.h"

#include <cstdint>

#include "../scrambler/scrambler.h"
#include "cuda_runtime_api.h"

using namespace mega;

std::vector<LDPCDecoder> DecoderFactory::decoders;

void DecoderFactory::init(const CuphyTensor::cuphy_dtype_t llr_type_,
                          const LDPCConfig& config, uint8_t num_devices) {
  for (uint8_t i = 0; i < num_devices; i++) {
    cudaSetDevice(i);
    decoders.push_back(LDPCDecoder(llr_type_, config));
  }
}

void DecoderFactory::destroy() { decoders.clear(); }

void DecoderFactory::decode(const CuphyTensor& tLLRs, const CuphyTensor& tOut,
                            cudaStream_t stream) {
  int device;
  cudaGetDevice(&device);
  decoders[device](tLLRs, tOut, stream);
  Scrambler::scrambler(tOut, tOut, stream);
}