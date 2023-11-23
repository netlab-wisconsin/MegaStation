/**
 * @file doencode.cc
 * @brief Implmentation file for the DoEncode class.  Currently, just supports
 * basestation
 */

#include "doencode.h"

#include "concurrent_queue_wrapper.h"
#include "encoder.h"
#include "logger.h"
#include "phy_ldpc_decoder_5gnr.h"

#include "temp_launch.h"

static constexpr bool kPrintEncodedData = false;
static constexpr bool kPrintRawMacData = false;

DoEncode::DoEncode(Config* in_config, int in_tid, Direction dir,
                   Table<int8_t>& in_raw_data_buffer, size_t in_buffer_rollover,
                   Table<int8_t>& in_mod_bits_buffer,
                   Table<cudaStream_t>& cuda_streams,
                   int8_t *cuda_encoded_buffer,
                   float2 *cuda_mod_buffer,
                   Stats* in_stats_manager)
    : Doer(in_config, in_tid),
      dir_(dir),
      raw_data_buffer_(in_raw_data_buffer),
      raw_buffer_rollover_(in_buffer_rollover),
      mod_bits_buffer_(in_mod_bits_buffer),
      scrambler_(std::make_unique<AgoraScrambler::Scrambler>()),
      cuda_streams_(cuda_streams),
      ldpc_encoder_(in_config->LdpcConfig(dir).BaseGraph(),
                  in_config->LdpcConfig(dir).ExpansionFactor(),
                  in_config->LdpcConfig(dir).NumRows(),
                  in_config->LdpcConfig(dir).NumBlocksInSymbol() * cfg_->UeAntNum()),
      cuda_encoded_buffer_(cuda_encoded_buffer),
      cuda_mod_buffer_(cuda_mod_buffer) {
  duration_stat_ = in_stats_manager->GetDurationStat(DoerType::kEncode, in_tid);
  parity_buffer_ = static_cast<int8_t*>(Agora_memory::PaddedAlignedAlloc(
      Agora_memory::Alignment_t::kAlign64,
      LdpcEncodingParityBufSize(cfg_->LdpcConfig(dir).BaseGraph(),
                                cfg_->LdpcConfig(dir).ExpansionFactor())));
  assert(parity_buffer_ != nullptr);
  encoded_buffer_temp_ = static_cast<int8_t*>(Agora_memory::PaddedAlignedAlloc(
      Agora_memory::Alignment_t::kAlign64,
      LdpcEncodingEncodedBufSize(cfg_->LdpcConfig(dir).BaseGraph(),
                                 cfg_->LdpcConfig(dir).ExpansionFactor())));
  assert(encoded_buffer_temp_ != nullptr);

  const size_t scrambler_buffer_bytes =
      cfg_->NumBytesPerCb(dir) + cfg_->NumPaddingBytesPerCb(dir);

  scrambler_buffer_ = static_cast<int8_t*>(Agora_memory::PaddedAlignedAlloc(
      Agora_memory::Alignment_t::kAlign64, scrambler_buffer_bytes));
  std::memset(scrambler_buffer_, 0u, scrambler_buffer_bytes);

  const LDPCconfig& ldpc_config = cfg_->LdpcConfig(dir_);
  int dims[2] = {
    int(ldpc_config.NumCbCodewLen()),
    int(ldpc_config.NumBlocksInSymbol() * cfg_->UeAntNum())
  };
  tensor_desc encoded_desc(CUPHY_BIT, 2, dims, CUPHY_TENSOR_ALIGN_DEFAULT);
  // cudaMalloc((void **)&cuda_encoded_buffer_local_, encoded_desc.sz_bytes);

  dims[0] = int(ldpc_config.NumCbLen());
  tensor_desc input_desc(CUPHY_BIT, 2, dims, CUPHY_TENSOR_ALIGN_DEFAULT);
  cpu_input_buffer_ = (uint8_t *)malloc(input_desc.sz_bytes);
  cudaMalloc((void **)&cuda_input_buffer_, input_desc.sz_bytes);

  init_modulation_launch(cfg_->ModTable(dir_)[0], pow(2, cfg_->ModOrderBits(dir_)) * sizeof(cuComplex));
  // cudaMalloc(reinterpret_cast<void **>(&cuda_mod_buffer_), 
  //     sizeof(cuComplex) * cfg_->UeAntNum() *
  //     cfg_->OfdmDataNum());
  cudaMalloc(reinterpret_cast<void **>(&cuda_ue_specific_), 
      sizeof(float2) * cfg_->UeAntNum() *
      cfg_->OfdmDataNum());
  cudaMemcpy(cuda_ue_specific_, cfg_->UeSpecificPilot()[0], sizeof(float2) * cfg_->UeAntNum() * cfg_->OfdmDataNum(), cudaMemcpyHostToDevice);

  assert(scrambler_buffer_ != nullptr);
}

DoEncode::~DoEncode() {
  std::free(parity_buffer_);
  std::free(encoded_buffer_temp_);
  std::free(scrambler_buffer_);
}

EventData DoEncode::Launch(size_t tag) {
  const LDPCconfig& ldpc_config = cfg_->LdpcConfig(dir_);
  // size_t frame_id = gen_tag_t(tag).frame_id_;
  size_t symbol_id = gen_tag_t(tag).symbol_id_;
  // size_t cb_id = gen_tag_t(tag).cb_id_;
  // size_t cur_cb_id = cb_id % ldpc_config.NumBlocksInSymbol();
  // size_t ue_id = cb_id / ldpc_config.NumBlocksInSymbol();

  size_t start_tsc = GetTime::WorkerRdtsc();

  size_t symbol_idx;
  size_t symbol_idx_data;
  if (dir_ == Direction::kDownlink) {
    symbol_idx = cfg_->Frame().GetDLSymbolIdx(symbol_id);
    assert(symbol_idx >= cfg_->Frame().ClientDlPilotSymbols());
    symbol_idx_data = symbol_idx - cfg_->Frame().ClientDlPilotSymbols();
  } else {
    symbol_idx = cfg_->Frame().GetULSymbolIdx(symbol_id);
    assert(symbol_idx >= cfg_->Frame().ClientUlPilotSymbols());
    symbol_idx_data = symbol_idx - cfg_->Frame().ClientUlPilotSymbols();
  }
  cuda_stream_ = cuda_streams_[symbol_id][0];
  size_t input_bit = ldpc_config.NumCbLen();
  size_t encode_bit = ldpc_config.NumCbCodewLen();
  size_t num_blocks = ldpc_config.NumBlocksInSymbol() * cfg_->UeAntNum();

  int dims_input[1] = { (int)input_bit };
  tensor_desc tIn(CUPHY_BIT, 1, dims_input);
  int dims_encode[1] = { (int)encode_bit };
  tensor_desc tOut(CUPHY_BIT, 1, dims_encode);
  // int8_t *input_gpu;
  // cudaMalloc(&input_gpu, tIn.sz_bytes * num_blocks);
  // int8_t *encode_gpu;
  // cudaMalloc(&encode_gpu, tOut.sz_bytes * num_blocks);
  /*for (size_t i = 0; i < num_blocks; i++) {
    cudaMemcpy(input_gpu + i * tIn.sz_bytes, input[i], tIn.sz_bytes, cudaMemcpyHostToDevice);
  }*/
  tIn.data = cuda_input_buffer_;
  // tOut.data = cuda_encoded_buffer_local_;
  tOut.data = cuda_encoded_buffer_ + symbol_idx * tOut.sz_bytes * num_blocks;

  // LDPC_encode ldpc_encode(ldpc_config.BaseGraph(),
  //   ldpc_config.ExpansionFactor(), ldpc_config.NumRows(), num_blocks);
  // ldpc_encode.encode(tIn, tOut, cuda_stream_);

  // return EventData(EventType::kEncode, tag);

  // if (kDebugPrintInTask) {
  //   std::printf(
  //       "In doEncode thread %d: frame: %zu, symbol: %zu:%zu:%zu, code block "
  //       "%zu, ue_id: %zu\n",
  //       tid_, frame_id, symbol_id, symbol_idx, symbol_idx_data, cur_cb_id,
  //       ue_id);
  // }

  int dims[2] = {
    int(ldpc_config.NumCbLen()),
    int(ldpc_config.NumBlocksInSymbol() * cfg_->UeAntNum())
  };
  tensor_desc input_desc(CUPHY_BIT, 2, dims, CUPHY_TENSOR_ALIGN_DEFAULT);
  //uint8_t* input_cpu_buffer = (uint8_t *)malloc(input_desc.sz_bytes);
  size_t start_tsc1 = GetTime::WorkerRdtsc();
  duration_stat_->task_duration_[1] += start_tsc1 - start_tsc;

  // int8_t* tx_data_ptr = nullptr;
  ///\todo Make GetMacBits and GetInfoBits
  /// universal with raw_buffer_rollover_ the parameter.
  if (kEnableMac) {
    // All cb's per symbol are included in 1 mac packet
    RtAssert(false, "kEnableMac is not supported");
    // tx_data_ptr = cfg_->GetMacBits(raw_data_buffer_, dir_,
    //                                (frame_id % raw_buffer_rollover_),
    //                                symbol_idx_data, ue_id, cur_cb_id);

    // if (kPrintRawMacData) {
    //   auto* pkt = reinterpret_cast<MacPacketPacked*>(tx_data_ptr);
    //   std::printf(
    //       "In doEncode [%d] mac packet frame: %d, symbol: %zu:%d, ue_id: %d, "
    //       "data length %d, crc %d size %zu:%zu\n",
    //       tid_, pkt->Frame(), symbol_idx_data, pkt->Symbol(), pkt->Ue(),
    //       pkt->PayloadLength(), pkt->Crc(), cfg_->MacPacketLength(dir_),
    //       cfg_->NumBytesPerCb(dir_));
    //   std::printf("Data: ");
    //   for (size_t i = 0; i < cfg_->MacPayloadMaxLength(dir_); i++) {
    //     std::printf(" %02x", (uint8_t)(pkt->Data()[i]));
    //   }
    //   std::printf("\n");
    // }
  } else {
    for (size_t i = 0; i < cfg_->UeAntNum(); i++) {
      uint8_t* input_buffer_ptr =
        (uint8_t*)cfg_->GetInfoBits(raw_data_buffer_, dir_, symbol_idx, i, 0);
      uint8_t* input_cpu_buffer_ptr = cpu_input_buffer_ + (i * ldpc_config.NumBlocksInSymbol() + 0) * (input_desc.strides[1] / 8);
      memcpy(input_cpu_buffer_ptr, input_buffer_ptr, cfg_->NumBytesPerCb(dir_));
    }
    // tx_data_ptr =
    //     cfg_->GetInfoBits(raw_data_buffer_, dir_, symbol_idx, ue_id, cur_cb_id);
  }

  // int8_t* ldpc_input = tx_data_ptr;
  // const size_t num_bytes_per_cb = cfg_->NumBytesPerCb(dir_);
  // const size_t num_padding_bytes_per_cb = cfg_->NumPaddingBytesPerCb(dir_);
  size_t start_tsc2 = GetTime::WorkerRdtsc();
  duration_stat_->task_duration_[2] += start_tsc2 - start_tsc1;

  // if (this->cfg_->ScrambleEnabled()) {
  //   scrambler_->Scramble(scrambler_buffer_, ldpc_input, num_bytes_per_cb);
  //   ldpc_input = scrambler_buffer_;
  // }
  // if (num_padding_bytes_per_cb > 0) {
  //   std::memset(&ldpc_input[num_bytes_per_cb], 0u, num_padding_bytes_per_cb);
  // }

  // size_t sz_line_ue = input_desc.sz_bytes / sizeof(uint8_t);
  int8_t *cuda_input_ptr = cuda_input_buffer_;// + symbol_idx * sz_line_ue;
  cudaMemcpyAsync(cuda_input_ptr, cpu_input_buffer_, input_desc.sz_bytes, cudaMemcpyHostToDevice, cuda_stream_);
  scrambler_launch(
    (uint8_t *)cuda_input_ptr,
    (uint8_t *)cuda_input_ptr,
    (input_desc.strides[1] / 8) * sizeof(uint8_t),
    ldpc_config.NumBlocksInSymbol() * cfg_->UeAntNum(),
    cuda_stream_);
  //input_desc.data = cuda_input_buffer_;

  // dims[0] = int(ldpc_config.NumCbCodewLen());
  //tensor_desc encoded_desc(CUPHY_BIT, 2, dims, CUPHY_TENSOR_ALIGN_DEFAULT);
  //encoded_desc.data = cuda_encoded_buffer_;

  // if (kDebugTxData) {
  //   std::stringstream dataprint;
  //   dataprint << std::setfill('0') << std::hex;
  //   for (size_t i = 0; i < num_bytes_per_cb; i++) {
  //     dataprint << " " << std::setw(2)
  //               << std::to_integer<int>(
  //                      reinterpret_cast<std::byte*>(ldpc_input)[i]);
  //   }
  //   AGORA_LOG_INFO("ldpc input (%zu %zu %zu): %s\n", frame_id, symbol_idx,
  //                  ue_id, dataprint.str().c_str());
  // }

  // LdpcEncodeHelper(ldpc_config.BaseGraph(), ldpc_config.ExpansionFactor(),
  //                  ldpc_config.NumRows(), encoded_buffer_temp_, parity_buffer_,
  //                  ldpc_input);
  // LDPC_encode ldpc_encoder(ldpc_config.BaseGraph(),
  //   ldpc_config.ExpansionFactor(),
  //   ldpc_config.NumRows(), ldpc_config.NumBlocksInSymbol() * cfg_->UeAntNum());
  ldpc_encoder_.encode(tIn, tOut, cuda_stream_);
  //cudaStreamSynchronize(cuda_stream_);

  float2 *cuda_mod_buffer_local = cuda_mod_buffer_ + symbol_idx * cfg_->UeAntNum() * cfg_->OfdmDataNum();
  modulation_launch(
    (const uint8_t *)tOut.data,
    (void *)(cuda_mod_buffer_local),
    cuda_ue_specific_,
    cfg_->OfdmPilotSpacing(),
    cfg_->ModOrderBits(dir_),
    ldpc_config.NumBlocksInSymbol() * (ldpc_config.NumCbCodewLen() / 8),
    cfg_->OfdmDataNum(),
    cfg_->UeAntNum(),
    cuda_stream_
  );
  // cudaStreamSynchronize(cuda_stream_);
  // float2 *modulated_cpy = (float2 *)malloc(cfg_->UeAntNum() * cfg_->OfdmDataNum() * sizeof(float2));
  // cudaMemcpy(modulated_cpy, cuda_mod_buffer_local, cfg_->UeAntNum() * cfg_->OfdmDataNum() * sizeof(float2), cudaMemcpyDeviceToHost);
  // spdlog::warn("symbol {}-{}, modulated_cpy[1] = ({}, {})\n", symbol_idx, cfg_->OfdmPilotSpacing(), modulated_cpy[16].x, modulated_cpy[16].y);
  // free(modulated_cpy);
  duration_stat_->task_duration_[3] += GetTime::WorkerRdtsc() - start_tsc2;
  // cudaMemcpyAsync(encoded_buffer_temp_, tOut.data, 20, cudaMemcpyDeviceToHost, cuda_stream_);
  // cudaMemcpyAsync(cuda_encoded_buffer_ + symbol_idx * tOut.sz_bytes * num_blocks, tOut.data, tOut.sz_bytes * num_blocks, cudaMemcpyDeviceToHost, cuda_stream_);
  // // cudaStreamSynchronize(cuda_stream_);
  // cudaMemcpyAsync(encoded_buffer_temp_, cuda_encoded_buffer_ + symbol_idx * tOut.sz_bytes * num_blocks, 20, cudaMemcpyDeviceToHost, cuda_stream_);
  // cudaStreamSynchronize(cuda_stream_);
  // spdlog::warn("Encoded {}: {} {}\n", symbol_idx, (void *)tOut.data, (uint8_t)encoded_buffer_temp_[0]);
  // uint8_t a[1] = {1};
  // cudaMemcpyAsync(tOut.data, a, 1, cudaMemcpyHostToDevice, cuda_stream_);
  //cudaMemset(tOut.data, 12, tOut.sz_bytes * num_blocks);
  //free(input_cpu_buffer);
  // if (kDebugTxData) {
  //   std::stringstream dataprint;
  //   dataprint << std::setfill('0') << std::hex;
  //   for (size_t i = 0; i < BitsToBytes(ldpc_config.NumCbCodewLen()); i++) {
  //     dataprint << " " << std::setw(2)
  //               << std::to_integer<int>(
  //                      reinterpret_cast<std::byte*>(encoded_buffer_temp_)[i]);
  //   }
  //   AGORA_LOG_INFO("ldpc output (%zu %zu %zu): %s\n", frame_id, symbol_idx,
  //                  ue_id, dataprint.str().c_str());
  // }
  // int8_t* mod_buffer_ptr = cfg_->GetModBitsBuf(mod_bits_buffer_, dir_, frame_id,
  //                                              symbol_idx, ue_id, cur_cb_id);

  // if (kPrintRawMacData && dir_ == Direction::kUplink) {
  //   std::printf("Encoded data - placed at location (%zu %zu %zu) %zu\n",
  //               frame_id, symbol_idx, ue_id,
  //               reinterpret_cast<intptr_t>(mod_buffer_ptr));
  // }
  // AdaptBitsForMod(reinterpret_cast<uint8_t*>(encoded_buffer_temp_),
  //                 reinterpret_cast<uint8_t*>(mod_buffer_ptr),
  //                 BitsToBytes(ldpc_config.NumCbCodewLen()),
  //                 cfg_->ModOrderBits(dir_));

  // if (kPrintEncodedData) {
  //   std::printf("Encoded data\n");
  //   const size_t num_mod = cfg_->SubcarrierPerCodeBlock(dir_);
  //   for (size_t i = 0; i < num_mod; i++) {
  //     std::printf("%u ", *(mod_buffer_ptr + i));
  //   }
  //   std::printf("\n");
  // }

  const size_t duration = GetTime::WorkerRdtsc() - start_tsc;
  duration_stat_->task_duration_[0] += duration;
  duration_stat_->task_count_++;
  if (GetTime::CyclesToUs(duration, cfg_->FreqGhz()) > 500) {
    std::printf("Thread %d Encode takes %.2f\n", tid_,
                GetTime::CyclesToUs(duration, cfg_->FreqGhz()));
  }
  return EventData(EventType::kEncode, tag);
}
