/**
 * @file agora_buffer.h
 * @brief Defination file for the AgoraBuffer class
 */
#include "agora_buffer.h"

#define half cuda_half
#include "ldpc_cuda.h"
#undef half

AgoraBuffer::AgoraBuffer(Config* const cfg)
    : config_(cfg),
      ul_socket_buf_size_(cfg->PacketLength() * cfg->BsAntNum() * kFrameWnd *
                          cfg->Frame().NumTotalSyms()),
      csi_buffer_(kFrameWnd, cfg->UeAntNum(),
                  cfg->BsAntNum() * cfg->OfdmDataNum()),
      ul_beam_matrix_(kFrameWnd, cfg->OfdmDataNum(),
                      cfg->BsAntNum() * cfg->UeAntNum()),
      dl_beam_matrix_(kFrameWnd, cfg->OfdmDataNum(),
                      cfg->UeAntNum() * cfg->BsAntNum()),
      demod_buffer_(kFrameWnd, cfg->Frame().NumULSyms(), cfg->UeAntNum(),
                    kMaxModType * cfg->OfdmDataNum()),
      decoded_buffer_(kFrameWnd, cfg->Frame().NumULSyms(), cfg->UeAntNum(),
                      cfg->LdpcConfig(Direction::kUplink).NumBlocksInSymbol() *
                          Roundup<64>(cfg->NumBytesPerCb(Direction::kUplink))) {
  AllocateTables();
}

AgoraBuffer::~AgoraBuffer() { FreeTables(); }

void AgoraBuffer::AllocateTables() {
  // Uplink
  const size_t task_buffer_symbol_num_ul =
      config_->Frame().NumULSyms() * kFrameWnd;

  ul_socket_buffer_.Malloc(config_->SocketThreadNum() /* RX */,
                           ul_socket_buf_size_,
                           Agora_memory::Alignment_t::kAlign64);

  fft_buffer_.Malloc(task_buffer_symbol_num_ul,
                     config_->OfdmDataNum() * config_->BsAntNum(),
                     Agora_memory::Alignment_t::kAlign64);

  equal_buffer_.Malloc(task_buffer_symbol_num_ul,
                       config_->OfdmDataNum() * config_->UeAntNum(),
                       Agora_memory::Alignment_t::kAlign64);
  ue_spec_pilot_buffer_.Calloc(
      kFrameWnd, config_->Frame().ClientUlPilotSymbols() * config_->UeAntNum(),
      Agora_memory::Alignment_t::kAlign64);
  
  // GPU
  size_t total_ul_sym = config_->Frame().NumPilotSyms() + config_->Frame().NumULSyms();
  fft_gather_cpu_.Malloc(total_ul_sym,
                     2 * config_->OfdmCaNum() * config_->BsAntNum(),
                     Agora_memory::Alignment_t::kAlign64);
  cudaMalloc(reinterpret_cast<void **>(&packet_buffer_),
      2 * sizeof(short) * config_->OfdmCaNum() * total_ul_sym * config_->BsAntNum());
  cudaMalloc(reinterpret_cast<void **>(&fft_out_),
      sizeof(cufftComplex) * config_->OfdmDataNum() * total_ul_sym * config_->BsAntNum());
  pilot_fft_out_ = fft_out_;
  uplink_fft_out_ = fft_out_
    + config_->OfdmDataNum() * config_->Frame().NumPilotSyms() * config_->BsAntNum();
  size_t prefix_zeros = config_->LdpcConfig(Direction::kUplink).ExpansionFactor() * 2;
  int dims[2] = {
    int(config_->LdpcConfig(Direction::kUplink).NumCbCodewLen() + prefix_zeros),
    int(config_->LdpcConfig(Direction::kUplink).NumBlocksInSymbol())
  };
  tensor_desc desc_encoded(CUPHY_R_16F, 2, dims, CUPHY_TENSOR_ALIGN_COALESCE);
  cudaMalloc(reinterpret_cast<void **>(&demul_out_),
    config_->Frame().NumULSyms() * config_->UeAntNum() * desc_encoded.sz_bytes);
  // cudaMalloc(&decoded_out_,
  //     sizeof(int8_t) * config_->Frame().NumULSyms() * config_->UeAntNum() *
  //     config_->LdpcConfig(Direction::kUplink).NumBlocksInSymbol() *
  //     Roundup<64>(config_->NumBytesPerCb(Direction::kUplink)));
  dims[0] = int(config_->LdpcConfig(Direction::kUplink).NumCbLen());
  tensor_desc desc_decoded(CUPHY_BIT, 2, dims, CUPHY_TENSOR_ALIGN_COALESCE);
  cudaMalloc(reinterpret_cast<void **>(&decoded_out_),
      config_->Frame().NumULSyms() * config_->UeAntNum() * desc_decoded.sz_bytes);
  cuda_streams_.Malloc(total_ul_sym, 1, Agora_memory::Alignment_t::kAlign64);
  for (size_t i = 0; i < total_ul_sym; i++) {
    cudaStreamCreateWithFlags(cuda_streams_[i], cudaStreamNonBlocking);
  }

  struct storeInfo stInfo = {
      .ofdmStart = config_->OfdmDataStart(),
      .ofdmNum = config_->OfdmDataNum(),
      .ofdmCAnum = config_->OfdmCaNum(),
      .bsAnt = config_->BsAntNum(),
	    .ueAnt = config_->UeAntNum(),
      .scGroup = config_->PilotScGroupSize(),
      .ueStart = 0,
      .pilotSign = NULL,
  };
  cudaMalloc(reinterpret_cast<void **>(&(stInfo.pilotSign)),
    sizeof(cufftComplex) * config_->OfdmDataNum());
  cudaMemcpy(stInfo.pilotSign, config_->PilotsSgn(),
    sizeof(cufftComplex) * config_->OfdmDataNum(), cudaMemcpyHostToDevice);
  cudaMalloc(reinterpret_cast<void **>(&stInfoPtr_), sizeof(stInfo));
  cudaMemcpy(stInfoPtr_, &stInfo, sizeof(stInfo), cudaMemcpyHostToDevice);

  // Downlink
  if (config_->Frame().NumDLSyms() > 0) {
    const size_t task_buffer_symbol_num =
        config_->Frame().NumDLSyms() * kFrameWnd;

    size_t dl_socket_buffer_status_size =
        config_->BsAntNum() * task_buffer_symbol_num;
    size_t dl_socket_buffer_size =
        config_->DlPacketLength() * dl_socket_buffer_status_size;
    AllocBuffer1d(&dl_socket_buffer_, dl_socket_buffer_size,
                  Agora_memory::Alignment_t::kAlign64, 1);

    size_t dl_bits_buffer_size =
        kFrameWnd * config_->MacBytesNumPerframe(Direction::kDownlink);
    dl_bits_buffer_.Calloc(config_->UeAntNum(), dl_bits_buffer_size,
                           Agora_memory::Alignment_t::kAlign64);
    dl_bits_buffer_status_.Calloc(config_->UeAntNum(), kFrameWnd,
                                  Agora_memory::Alignment_t::kAlign64);

    dl_ifft_buffer_.Calloc(config_->BsAntNum() * task_buffer_symbol_num,
                           config_->OfdmCaNum(),
                           Agora_memory::Alignment_t::kAlign64);
    calib_dl_buffer_.Malloc(kFrameWnd,
                            config_->BfAntNum() * config_->OfdmDataNum(),
                            Agora_memory::Alignment_t::kAlign64);
    calib_ul_buffer_.Malloc(kFrameWnd,
                            config_->BfAntNum() * config_->OfdmDataNum(),
                            Agora_memory::Alignment_t::kAlign64);
    calib_dl_msum_buffer_.Malloc(kFrameWnd,
                                 config_->BfAntNum() * config_->OfdmDataNum(),
                                 Agora_memory::Alignment_t::kAlign64);
    calib_ul_msum_buffer_.Malloc(kFrameWnd,
                                 config_->BfAntNum() * config_->OfdmDataNum(),
                                 Agora_memory::Alignment_t::kAlign64);
    calib_buffer_.Malloc(kFrameWnd,
                         config_->BfAntNum() * config_->OfdmDataNum(),
                         Agora_memory::Alignment_t::kAlign64);
    //initialize the calib buffers
    const complex_float complex_init = {0.0f, 0.0f};
    //const complex_float complex_init = {1.0f, 0.0f};
    for (size_t frame = 0u; frame < kFrameWnd; frame++) {
      for (size_t i = 0; i < (config_->OfdmDataNum() * config_->BfAntNum());
           i++) {
        calib_dl_buffer_[frame][i] = complex_init;
        calib_ul_buffer_[frame][i] = complex_init;
        calib_dl_msum_buffer_[frame][i] = complex_init;
        calib_ul_msum_buffer_[frame][i] = complex_init;
        calib_buffer_[frame][i] = complex_init;
      }
    }
    dl_mod_bits_buffer_.Calloc(
        task_buffer_symbol_num,
        Roundup<64>(config_->GetOFDMDataNum()) * config_->UeAntNum(),
        Agora_memory::Alignment_t::kAlign64);
  }
}

void AgoraBuffer::FreeTables() {
  // Uplink
  ul_socket_buffer_.Free();
  fft_buffer_.Free();
  equal_buffer_.Free();
  ue_spec_pilot_buffer_.Free();

  // Downlink
  if (config_->Frame().NumDLSyms() > 0) {
    FreeBuffer1d(&dl_socket_buffer_);
    dl_ifft_buffer_.Free();
    calib_dl_buffer_.Free();
    calib_ul_buffer_.Free();
    calib_dl_msum_buffer_.Free();
    calib_ul_msum_buffer_.Free();
    calib_buffer_.Free();
    dl_mod_bits_buffer_.Free();
    dl_bits_buffer_.Free();
    dl_bits_buffer_status_.Free();
  }
}
