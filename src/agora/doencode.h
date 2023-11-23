/**
 * @file doencode.h
 * @brief Declaration file for the Docoding class.  Includes the DoEncode and
 * DoDecode classes.
 */

#ifndef DOENCODE_H_
#define DOENCODE_H_

#include <cstdint>
#include <memory>

#define half flex_half
#include "config.h"
#include "doer.h"
#include "memory_manage.h"
#include "message.h"
#include "scrambler.h"
#include "stats.h"
#undef half

#define half cuda_half
#include "ldpc_cuda.h"
#include <cuda.h>
#include <cuda_runtime.h>
#undef half

class DoEncode : public Doer {
 public:
  DoEncode(Config* in_config, int in_tid, Direction dir,
           Table<int8_t>& in_raw_data_buffer, size_t in_buffer_rollover,
           Table<int8_t>& in_mod_bits_buffer,
           Table<cudaStream_t>& cuda_streams,
           int8_t *cuda_encoded_buffer,
           float2 *cuda_mod_buffer,
           Stats* in_stats_manager);
  ~DoEncode() override;

  EventData Launch(size_t tag) override;

 private:
  Direction dir_;

  // References to buffers allocated pre-construction
  Table<int8_t>& raw_data_buffer_;
  size_t raw_buffer_rollover_;
  Table<int8_t>& mod_bits_buffer_;

  // Intermediate buffer to hold LDPC encoding parity
  int8_t* parity_buffer_;

  // Intermediate buffer to hold LDPC encoding output
  int8_t* encoded_buffer_temp_;

  // Intermediate buffer to hold pre/post scrambled data
  int8_t* scrambler_buffer_;

  DurationStat* duration_stat_;
  std::unique_ptr<AgoraScrambler::Scrambler> scrambler_;

  // GPU
  Table<cudaStream_t>& cuda_streams_;
  LDPC_encode ldpc_encoder_;
  int8_t *cuda_encoded_buffer_;
  // uint8_t *cuda_encoded_buffer_local_;
  float2 *cuda_mod_buffer_;
  int8_t *cuda_input_buffer_;
  uint8_t *cpu_input_buffer_;
  float2 *cuda_ue_specific_;
  cudaStream_t cuda_stream_;
};

#endif  // DOENCODE_H_
