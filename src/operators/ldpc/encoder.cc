/**
 * @file encoder.cc
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Encoder Factory implementation
 * @version 0.1
 * @date 2024-04-08
 *
 * @copyright Copyright (c) 2024
 *
 */

#include "encoder.h"

#include "../scrambler/scrambler.h"

using namespace mega;

LDPCEncoder EncoderFactory::encoder;

void EncoderFactory::init(const LDPCConfig &config) {
  encoder = LDPCEncoder(config);
}

void EncoderFactory::encode(const CuphyTensor &tIn, const CuphyTensor &tOut,
                            cudaStream_t stream) {
  Scrambler::scrambler(tIn, tIn, stream);
  encoder(tIn, tOut, stream);
}