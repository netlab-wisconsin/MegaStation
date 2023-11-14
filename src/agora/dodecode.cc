/**
 * @file dodecode.cc
 * @brief Implmentation file for the DoDecode class. Currently, just supports
 * basestation
 */
#include "dodecode.h"

#include "concurrent_queue_wrapper.h"
#include "phy_ldpc_decoder_5gnr.h"

#include "cuda_fp16.h"
#include "temp_launch.h"

static constexpr bool kPrintLLRData = false;
static constexpr bool kPrintDecodedData = false;

static constexpr size_t kVarNodesSize = 1024 * 1024 * sizeof(int16_t);

DoDecode::DoDecode(
    Config* in_config, int in_tid,
    PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t>& demod_buffers,
    PtrCube<kFrameWnd, kMaxSymbols, kMaxUEs, int8_t>& decoded_buffers,
    PhyStats* in_phy_stats,
    Table<cudaStream_t>& cuda_streams,
    int16_t* cuda_demul_buffer,
    int8_t* cuda_decoded_buffer,
    Stats* in_stats_manager)
    : Doer(in_config, in_tid),
      demod_buffers_(demod_buffers),
      decoded_buffers_(decoded_buffers),
      phy_stats_(in_phy_stats),
      scrambler_(std::make_unique<AgoraScrambler::Scrambler>()),
      cuda_streams_(cuda_streams),
      ldpc_decoder_(CUPHY_R_16F,
        (short)in_config->LdpcConfig(Direction::kUplink).NumRows(),
        (short)in_config->LdpcConfig(Direction::kUplink).BaseGraph(),
        (short)in_config->LdpcConfig(Direction::kUplink).ExpansionFactor(),
        (short)in_config->LdpcConfig(Direction::kUplink).MaxDecoderIter()),
      cuda_demul_budffer_(cuda_demul_buffer),
      cuda_decoded_buffer_(cuda_decoded_buffer) {
  duration_stat_ = in_stats_manager->GetDurationStat(DoerType::kDecode, in_tid);
  resp_var_nodes_ = static_cast<int16_t*>(Agora_memory::PaddedAlignedAlloc(
      Agora_memory::Alignment_t::kAlign64, kVarNodesSize));
  uint8_t scrambler_init = AgoraScrambler::kScramblerInitState;
  uint8_t res_xor = 0;
  uint8_t scram_buffer_cpu[AgoraScrambler::kScramblerlength];
  for (size_t i = 0; i < AgoraScrambler::kScramblerlength; i++) {
    res_xor = (scrambler_init ^ (scrambler_init >> 3)) & 0x01;
    scram_buffer_cpu[i] = res_xor;
    scrambler_init >>= 1;
    scrambler_init |= res_xor << 6;
  }
  init_scrambler_launch(scram_buffer_cpu);
}

DoDecode::~DoDecode() { std::free(resp_var_nodes_); }

EventData DoDecode::Launch(size_t tag) {
  const LDPCconfig& ldpc_config = cfg_->LdpcConfig(Direction::kUplink);
  const size_t frame_id = gen_tag_t(tag).frame_id_;
  const size_t symbol_id = gen_tag_t(tag).symbol_id_;
  const size_t symbol_idx_ul = cfg_->Frame().GetULSymbolIdx(symbol_id);
  // const size_t cb_id = gen_tag_t(tag).cb_id_;
  const size_t symbol_offset =
      cfg_->GetTotalDataSymbolIdxUl(frame_id, symbol_idx_ul);
  // const size_t cur_cb_id = (cb_id % ldpc_config.NumBlocksInSymbol());
  // const size_t ue_id = (cb_id / ldpc_config.NumBlocksInSymbol());
  const size_t frame_slot = (frame_id % kFrameWnd);
  const size_t num_bytes_per_cb = cfg_->NumBytesPerCb(Direction::kUplink);
  // if (kDebugPrintInTask == true) {
  //   std::printf(
  //       "In doDecode thread %d: frame: %zu, symbol: %zu, code block: "
  //       "%zu, ue: %zu offset %zu\n",
  //       tid_, frame_id, symbol_id, cur_cb_id, ue_id, symbol_offset);
  // }

  size_t start_tsc = GetTime::WorkerRdtsc();
  cuda_stream_ = cuda_streams_[symbol_id][0];

  // struct bblib_ldpc_decoder_5gnr_request ldpc_decoder_5gnr_request {};
  // struct bblib_ldpc_decoder_5gnr_response ldpc_decoder_5gnr_response {};

  // Decoder setup
  // int16_t num_filler_bits = 0;
  // int16_t num_channel_llrs = ldpc_config.NumCbCodewLen();

  // ldpc_decoder_5gnr_request.numChannelLlrs = num_channel_llrs;
  // ldpc_decoder_5gnr_request.numFillerBits = num_filler_bits;
  // ldpc_decoder_5gnr_request.maxIterations = ldpc_config.MaxDecoderIter();
  // ldpc_decoder_5gnr_request.enableEarlyTermination =
  //     ldpc_config.EarlyTermination();
  // ldpc_decoder_5gnr_request.Zc = ldpc_config.ExpansionFactor();
  // ldpc_decoder_5gnr_request.baseGraph = ldpc_config.BaseGraph();
  // ldpc_decoder_5gnr_request.nRows = ldpc_config.NumRows();

  // int num_msg_bits = ldpc_config.NumCbLen() - num_filler_bits;
  // ldpc_decoder_5gnr_response.numMsgBits = num_msg_bits;
  // ldpc_decoder_5gnr_response.varNodes = resp_var_nodes_;

  // int8_t* llr_buffer_ptr = demod_buffers_[frame_slot][symbol_idx_ul][ue_id] +
  //                          (cfg_->ModOrderBits(Direction::kUplink) *
  //                           (ldpc_config.NumCbCodewLen() * cur_cb_id));
  size_t prefix_zeros = ldpc_config.ExpansionFactor() * 2;
  // int8_t *cuda_decoded_ptr = cuda_decoded_buffer_ +
  //   (symbol_idx_ul * cfg_->UeAntNum() *
  //   cfg_->LdpcConfig(Direction::kUplink).NumBlocksInSymbol() *
  //     Roundup<64>(cfg_->NumBytesPerCb(Direction::kUplink))) +
  //   (ue_id * cfg_->LdpcConfig(Direction::kUplink).NumBlocksInSymbol() *
  //     Roundup<64>(cfg_->NumBytesPerCb(Direction::kUplink)));
  // __half *cpu_demul_ptr = (__half *)malloc(sizeof(__half) *
  //   cfg_->ModOrderBits(Direction::kUplink) * cfg_->OfdmDataNum());
  // cudaMemcpy(cpu_demul_ptr, cuda_demul_ptr,
  //   sizeof(__half) * cfg_->ModOrderBits(Direction::kUplink) * cfg_->OfdmDataNum(),
  //   cudaMemcpyDeviceToHost);
  // for (size_t i = 0; i < cfg_->ModOrderBits(Direction::kUplink) * cfg_->OfdmDataNum(); i++) {
  //   llr_buffer_ptr[i] = (short)cpu_demul_ptr[i];
  // }
  int dims[2] = {
    int(ldpc_config.NumCbCodewLen() + prefix_zeros),
    int(ldpc_config.NumBlocksInSymbol() * cfg_->UeAntNum())
  };
  tensor_desc llr_desc(CUPHY_R_16F, 2, dims, CUPHY_TENSOR_ALIGN_COALESCE);
  size_t sz_line_ue = llr_desc.sz_bytes / sizeof(half);
  half *cuda_demul_ptr = (half *)cuda_demul_budffer_ + symbol_idx_ul * sz_line_ue;
  llr_desc.data = cuda_demul_ptr;
  // printf("llr_dims: %d, %d\n", llr_desc.dims[0], llr_desc.dims[1]);
  // printf("llr_strides: %d, %d\n", llr_desc.strides[0], llr_desc.strides[1]);
  // dims[0] = cfg_->LdpcConfig(Direction::kUplink).NumBlocksInSymbol() *
  //     Roundup<64>(cfg_->NumBytesPerCb(Direction::kUplink)) * 8;
  // tensor_desc decoded_desc(CUPHY_BIT, 2, dims, CUPHY_TENSOR_ALIGN_COALESCE);
  // decoded_desc.data = cuda_decoded_ptr;
  // printf("decoded_dims: %d, %d\n", decoded_desc.dims[0], decoded_desc.dims[1]);
  // printf("decoded_strides: %d, %d\n", decoded_desc.strides[0], decoded_desc.strides[1]);
  // printf("num_blocks_in_symbol: %ld\n", cfg_->LdpcConfig(Direction::kUplink).NumBlocksInSymbol());
  dims[0] = int(ldpc_config.NumCbLen());
  tensor_desc decoded_desc(CUPHY_BIT, 2, dims, CUPHY_TENSOR_ALIGN_COALESCE);
  sz_line_ue = decoded_desc.sz_bytes / sizeof(uint8_t);
  int8_t *cuda_decoded_ptr = cuda_decoded_buffer_ + symbol_idx_ul * sz_line_ue;
  decoded_desc.data = cuda_decoded_ptr;
  // printf("decoded_dims: %d, %d\n", decoded_desc.dims[0], decoded_desc.dims[1]);
  // printf("decoded_strides: %d, %d\n", decoded_desc.strides[0], decoded_desc.strides[1]);
  // printf("num_blocks_in_symbol: %ld\n", cfg_->LdpcConfig(Direction::kUplink).NumBlocksInSymbol());
  // printf("size_line_ue: %ld, %ld\n", sz_line_ue, cfg_->LdpcConfig(Direction::kUplink).NumBlocksInSymbol() * Roundup<64>(cfg_->NumBytesPerCb(Direction::kUplink)));
  ldpc_decoder_.decode(llr_desc, decoded_desc, cuda_stream_);
  scrambler_launch(
    (uint8_t *)cuda_decoded_ptr,
    (uint8_t *)cuda_decoded_ptr,
    (decoded_desc.strides[1] / 8) * sizeof(uint8_t),
    ldpc_config.NumBlocksInSymbol() * cfg_->UeAntNum(),
    cuda_stream_);

  uint8_t *decoded_cpu_buffer = (uint8_t *)malloc(decoded_desc.sz_bytes);
  cudaMemcpyAsync(decoded_cpu_buffer, cuda_decoded_ptr, decoded_desc.sz_bytes, cudaMemcpyDeviceToHost, cuda_stream_);
  cudaStreamSynchronize(cuda_stream_);

  size_t start_tsc1 = GetTime::WorkerRdtsc();
  duration_stat_->task_duration_[1] += start_tsc1 - start_tsc;
  // cudaMemcpy(decoded_buffer_ptr, cuda_decoded_ptr,
  //   sizeof(uint8_t) * cfg_->LdpcConfig(Direction::kUplink).NumBlocksInSymbol() *
  //     Roundup<64>(cfg_->NumBytesPerCb(Direction::kUplink)),
  //   cudaMemcpyDeviceToHost);
  // cudaMemcpyAsync(decoded_buffer_ptr, cuda_decoded_ptr,
  //   num_bytes_per_cb * cfg_->LdpcConfig(Direction::kUplink).NumBlocksInSymbol(),
  //   cudaMemcpyDeviceToHost, cuda_stream_);
  // cudaStreamSynchronize(cuda_stream_);
  // size_t sz_cpu_buffer = sizeof(uint8_t)
  //   * cfg_->LdpcConfig(Direction::kUplink).NumBlocksInSymbol()
  //   * Roundup<64>(cfg_->NumBytesPerCb(Direction::kUplink));
  for (size_t i = 0; i < cfg_->UeAntNum(); i++) {
    uint8_t* decoded_buffer_ptr =
      (uint8_t*)decoded_buffers_[frame_slot][symbol_idx_ul][i];
    memcpy(decoded_buffer_ptr,
      decoded_cpu_buffer
        + i * (decoded_desc.strides[1] / 8) * sizeof(uint8_t)
        * ldpc_config.NumBlocksInSymbol(),
      num_bytes_per_cb * ldpc_config.NumBlocksInSymbol());
  }
  free(decoded_cpu_buffer);

  // ldpc_decoder_5gnr_request.varNodes = llr_buffer_ptr;
  // ldpc_decoder_5gnr_response.compactedMessageBytes = decoded_buffer_ptr;

  //bblib_ldpc_decoder_5gnr(&ldpc_decoder_5gnr_request,
  //                        &ldpc_decoder_5gnr_response);

  // if (cfg_->ScrambleEnabled()) {
  //   scrambler_->Descramble(decoded_buffer_ptr, num_bytes_per_cb);
  // }

  size_t start_tsc2 = GetTime::WorkerRdtsc();
  duration_stat_->task_duration_[2] += start_tsc2 - start_tsc1;

  // if (kPrintLLRData) {
  //   std::printf("LLR data, symbol_offset: %zu\n", symbol_offset);
  //   for (size_t i = 0; i < ldpc_config.NumCbCodewLen(); i++) {
  //     std::printf("%d ", *(llr_buffer_ptr + i));
  //   }
  //   std::printf("\n");
  // }

  // if (kPrintDecodedData) {
  //   std::printf("Decoded data\n");
  //   for (size_t i = 0; i < (ldpc_config.NumCbLen() >> 3); i++) {
  //     std::printf("%u ", *(decoded_buffer_ptr + i));
  //   }
  //   std::printf("\n");
  // }

  for (size_t ue_id = 0; ue_id < cfg_->UeAntNum(); ue_id++) {
  if ((kEnableMac == false) && (kPrintPhyStats == true) &&
      (symbol_idx_ul >= cfg_->Frame().ClientUlPilotSymbols())) {
    uint8_t* decoded_buffer_ptr =
      (uint8_t*)decoded_buffers_[frame_slot][symbol_idx_ul][ue_id];
    phy_stats_->UpdateDecodedBits(ue_id, symbol_offset, frame_slot,
                                  num_bytes_per_cb * 8);
    phy_stats_->IncrementDecodedBlocks(ue_id, symbol_offset, frame_slot);
    size_t block_error(0);
    for (size_t i = 0; i < num_bytes_per_cb; i++) {
      uint8_t rx_byte = decoded_buffer_ptr[i];
      auto tx_byte = static_cast<uint8_t>(
          cfg_->GetInfoBits(cfg_->UlBits(), Direction::kUplink, symbol_idx_ul,
                            ue_id, 0)[i]);
      phy_stats_->UpdateBitErrors(ue_id, symbol_offset, frame_slot, tx_byte,
                                  rx_byte);
      if (rx_byte != tx_byte) {
        block_error++;
      }
    }
    phy_stats_->UpdateBlockErrors(ue_id, symbol_offset, frame_slot,
                                  block_error);
  }
  }
  size_t start_tsc3 = GetTime::WorkerRdtsc();
  duration_stat_->task_duration_[3] += start_tsc3 - start_tsc2;

  size_t duration = GetTime::WorkerRdtsc() - start_tsc;
  duration_stat_->task_duration_[0] += duration;
  duration_stat_->task_count_++;
  if (GetTime::CyclesToUs(duration, cfg_->FreqGhz()) > 500) {
    std::printf("Thread %d Decode takes %.2f\n", tid_,
                GetTime::CyclesToUs(duration, cfg_->FreqGhz()));
  }

  return EventData(EventType::kDecode, tag);
}
