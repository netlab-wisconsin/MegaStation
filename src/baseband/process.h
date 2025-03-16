/**
 * @file process.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Baseband processing header file
 * @version 0.1
 * @date 2023-12-16
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cstdint>

#include "beamform/beamform.h"
#include "fft/fft_op.h"
#include "ldpc/decoder.h"
#include "ldpc/encoder.h"
#include "utils/counter.h"
#include "utils/types.h"

namespace mega {

class BandProcess {
 private:
  // Pilot Processing
  static ArrFrameSymbol<PilotFFT<QuanT>> pfft;
  static ArrFrame<Beamform> beam;

  // Uplink Processing
  static ArrFrameSymbol<UplinkFFT<QuanT>> ufft;
  static ArrFrame<LDPCDecoder> decoder;

  // Downlink Processing
  static ArrFrame<LDPCEncoder> encoder;
  static ArrFrameSymbol<DownlinkIFFT<QuanT>> difft;

  static const uint32_t base_thread_id = kRecvThreads;
  static std::atomic<uint32_t> thread_offset;

 protected:
  static void process(uint32_t frame_cid, uint32_t symbol_id);
  static void noprocess(uint32_t frame_cid, uint32_t symbol_id);
  static void frame(uint32_t frame_cid);
  static void symbol(uint32_t frame_cid, uint32_t symbol_id);
  static void task(uint32_t frame_cid, uint32_t symbol_id, uint32_t task_id);
  static void copy(uint32_t frame_sid, uint32_t frame_did, uint32_t symbol_id,
                   uint32_t task_id);
  static void sync(uint32_t frame_cid, uint32_t symbol_id);

 public:
  static void start(uint32_t frame_cid);
  static void init();

  static Counter counters;
};

}  // namespace mega