/**
 * @file dodecode.h
 * @brief Declaration file for the DoDecode class.
 */

#ifndef DODECODE_H_
#define DODECODE_H_

#include <cstdint>
#include <memory>

#define half flex_half
#include "config.h"
#include "doer.h"
#include "memory_manage.h"
#include "message.h"
#include "phy_stats.h"
#include "scrambler.h"
#include "stats.h"
#undef half

#include "ldpc_cuda.h"

class DoDecode : public Doer {
 public:
  DoDecode(Config* in_config, int in_tid,
           PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t>& demod_buffers,
           PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t>& decoded_buffers,
           PhyStats* in_phy_stats,
           Table<cudaStream_t>& cuda_streams,
           int16_t* cuda_demul_buffer,
           int8_t* cuda_decoded_buffer,
           Stats* in_stats_manager);
  ~DoDecode() override;

  EventData Launch(size_t tag) override;

 private:
  int16_t* resp_var_nodes_;
  PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t>& demod_buffers_;
  PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t>& decoded_buffers_;
  PhyStats* phy_stats_;
  DurationStat* duration_stat_;
  std::unique_ptr<AgoraScrambler::Scrambler> scrambler_;

  // GPU
  Table<cudaStream_t>& cuda_streams_;
  LDPC_decode ldpc_decoder_;
  int16_t *cuda_demul_budffer_;
  int8_t *cuda_decoded_buffer_;
  cudaStream_t cuda_stream_;
};

#endif  // DODECODE_H_
