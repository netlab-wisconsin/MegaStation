/**
 * @file test_ldpc.cc
 *
 * @brief Accuracy and performance test for LDPC. The encoder is Agora's
 * avx2enc - unlike FlexRAN's encoder, avx2enc works with AVX2 (i.e., unlike
 * FlexRAN's encoder, avx2enc does not require AVX-512). The decoder is
 * FlexRAN's decoder, which supports AVX2.
 */

#include <algorithm>
#include <bitset>
#include <fstream>
#include <vector>

#define half flex_half
#include "encoder.h"
#include "gettime.h"
#include "memory_manage.h"
#include "phy_ldpc_decoder_5gnr.h"
#include "symbols.h"
#include "utils_ldpc.h"
#undef half

#define half cuda_half
#include "cuphy.h"
#include "ldpc_cuda.h"
#undef half

void ldpc_decode_cuda(
  uint16_t Zc,
  int8_t **llr,
  uint8_t **decoded,
  size_t llr_bit,
  size_t decode_bit
);

void ldpc_encode_cuda(
  uint16_t Zc,
  int8_t **input,
  int8_t **encode,
  size_t input_bit,
  size_t encode_bit
);

static constexpr size_t kNumCodeBlocks = 2;
static constexpr size_t kBaseGraph = 1;
static constexpr bool kEnableEarlyTermination = false;
static constexpr size_t kNumFillerBits = 0;
static constexpr size_t kMaxDecoderIters = 8;
static constexpr size_t kK5GnrNumPunctured = 2;
static constexpr size_t kNumRows = 46;

int main() {
  double freq_ghz = GetTime::MeasureRdtscFreq();
  std::printf("Spinning for one second for Turbo Boost\n");
  GetTime::NanoSleep(1000 * 1000 * 1000, freq_ghz);
  int8_t* input[kNumCodeBlocks];
  int8_t* parity[kNumCodeBlocks];
  int8_t* encoded[kNumCodeBlocks];
  uint8_t* decoded[kNumCodeBlocks];

  std::printf("Code rate: %.3f (nRows = %zu)\n", 22.f / (20 + kNumRows),
              kNumRows);

  /*std::vector<size_t> zc_vec = {
      2,   4,   8,   16, 32, 64,  128, 256, 3,   6,   12,  24, 48,
      96,  192, 384, 5,  10, 20,  40,  80,  160, 320, 7,   14, 28,
      56,  112, 224, 9,  18, 36,  72,  144, 288, 11,  22,  44, 88,
      176, 352, 13,  26, 52, 104, 208, 15,  30,  60,  120, 240};*/
  std::vector<size_t> zc_vec = { 64 };
  std::sort(zc_vec.begin(), zc_vec.end());
  for (const size_t& zc : zc_vec) {
    if (zc < LdpcGetMinZc() || zc > LdpcGetMaxZc()) {
      std::fprintf(stderr, "Zc value %zu not supported. Skipping.\n", zc);
      continue;
    }
    const size_t num_input_bits = LdpcNumInputBits(kBaseGraph, zc);
    const size_t num_encoded_bits =
        LdpcNumEncodedBits(kBaseGraph, zc, kNumRows);
    const size_t size_encoded_buf = LdpcEncodingEncodedBufSize(kBaseGraph, zc);

    for (size_t i = 0; i < kNumCodeBlocks; i++) {
      input[i] = new int8_t[LdpcEncodingInputBufSize(kBaseGraph, zc)];
      parity[i] = new int8_t[LdpcEncodingParityBufSize(kBaseGraph, zc)];
      encoded[i] = new int8_t[size_encoded_buf];
      decoded[i] = new uint8_t[size_encoded_buf];
    }

    // Randomly generate input
    //srand(time(nullptr));
    srand(0);
    for (auto& n : input) {
      for (size_t i = 0; i < BitsToBytes(num_input_bits); i++) {
        n[i] = static_cast<int8_t>(rand());
        //n[i] = static_cast<int8_t>(0xaa);
      }
    }

    const size_t encoding_start_tsc = GetTime::Rdtsc();
    for (size_t n = 0; n < kNumCodeBlocks; n++) {
      LdpcEncodeHelper(kBaseGraph, zc, kNumRows, encoded[n], parity[n],
                       input[n]);
    }
    //printf("input[0]: %d\n", input[0][0]);
    /*LDPC_encode ldpc_encode(kBaseGraph, zc, kNumRows, 1);
    for (size_t n = 0; n < kNumCodeBlocks; n++) {
      ldpc_encode_cuda(ldpc_encode, zc, &(input[n]), &(encoded[n]), num_input_bits, num_encoded_bits);
    }*/
    //ldpc_encode_cuda(zc, input, encoded, num_input_bits, num_encoded_bits);
    //printf("encoded[0]: %d\n", encoded[0][0]);

    const double encoding_us =
        GetTime::CyclesToUs(GetTime::Rdtsc() - encoding_start_tsc, freq_ghz);

    // For decoding, generate log-likelihood ratios, one byte per input bit
    int8_t* llrs[kNumCodeBlocks];
    for (size_t n = 0; n < kNumCodeBlocks; n++) {
      llrs[n] = static_cast<int8_t*>(Agora_memory::PaddedAlignedAlloc(
          Agora_memory::Alignment_t::kAlign32, num_encoded_bits));
      for (size_t i = 0; i < num_encoded_bits; i++) {
        uint8_t bit_i = (encoded[n][i / 8] >> (i % 8)) & 1;
        llrs[n][i] = (bit_i == 1 ? -127 : 127);
      }
    }

    // Decoder setup
    struct bblib_ldpc_decoder_5gnr_request ldpc_decoder_5gnr_request = {};
    struct bblib_ldpc_decoder_5gnr_response ldpc_decoder_5gnr_response = {};
    ldpc_decoder_5gnr_request.numChannelLlrs = num_encoded_bits;
    ldpc_decoder_5gnr_request.numFillerBits = kNumFillerBits;
    ldpc_decoder_5gnr_request.maxIterations = kMaxDecoderIters;
    ldpc_decoder_5gnr_request.enableEarlyTermination = kEnableEarlyTermination;
    ldpc_decoder_5gnr_request.Zc = zc;
    ldpc_decoder_5gnr_request.baseGraph = kBaseGraph;
    ldpc_decoder_5gnr_request.nRows = kNumRows;

    const size_t buffer_len = 1024 * 1024;
    const size_t num_msg_bits = num_input_bits - kNumFillerBits;
    ldpc_decoder_5gnr_response.numMsgBits = num_msg_bits;
    ldpc_decoder_5gnr_response.varNodes =
        static_cast<int16_t*>(Agora_memory::PaddedAlignedAlloc(
            Agora_memory::Alignment_t::kAlign32, buffer_len * sizeof(int16_t)));

    // Decoding
    const size_t decoding_start_tsc = GetTime::Rdtsc();
    /*for (size_t n = 0; n < kNumCodeBlocks; n++) {
      ldpc_decoder_5gnr_request.varNodes = llrs[n];
      ldpc_decoder_5gnr_response.compactedMessageBytes = decoded[n];
      bblib_ldpc_decoder_5gnr(&ldpc_decoder_5gnr_request,
                              &ldpc_decoder_5gnr_response);
    }*/
    ldpc_decode_cuda(zc, llrs, decoded, num_encoded_bits + 2*zc, num_input_bits);

    const double decoding_us =
        GetTime::CyclesToUs(GetTime::Rdtsc() - decoding_start_tsc, freq_ghz);

    // Check for errors
    size_t err_cnt = 0;
    for (size_t n = 0; n < kNumCodeBlocks; n++) {
      auto* input_buffer = reinterpret_cast<uint8_t*>(input[n]);
      uint8_t* output_buffer = decoded[n];
      for (size_t i = 0; i < BitsToBytes(num_input_bits); i++) {
        uint8_t error = input_buffer[i] ^ output_buffer[i];
        printf("%ld\terror: %i, input_buffer: %i, output_buffer: %i\n", i, error, input_buffer[i], output_buffer[i]);
        for (size_t j = 0; j < 8; j++) {
          if (i * 8 + j >= num_input_bits) {
            continue;  // Don't compare beyond end of input bits
          }
          err_cnt += error & 1;
          error >>= 1;
        }
      }
    }

    std::printf(
        "Zc = %zu, {encoding, decoding}: {%.2f, %.2f} Mbps, {%.2f, "
        "%.2f} us per code block. Bit errors = %zu, BER = %.3f\n",
        zc, num_input_bits * kNumCodeBlocks / encoding_us,
        num_input_bits * kNumCodeBlocks / decoding_us,
        encoding_us / kNumCodeBlocks, decoding_us / kNumCodeBlocks, err_cnt,
        err_cnt * 1.0 / (kNumCodeBlocks * num_input_bits));

    for (size_t i = 0; i < kNumCodeBlocks; i++) {
      delete[] input[i];
      delete[] parity[i];
      delete[] encoded[i];
      delete[] decoded[i];
      std::free(llrs[i]);
    }
    std::free(ldpc_decoder_5gnr_response.varNodes);
  }

  return 0;
}

void ldpc_encode_cuda(
  uint16_t Zc,
  int8_t **input,
  int8_t **encode,
  size_t input_bit,
  size_t encode_bit
) {
  //cudaStream_t stream;
  //cudaStreamCreate(&stream);
  int dims_input[1] = { (int)input_bit };
  tensor_desc tIn(CUPHY_BIT, 1, dims_input);
  int dims_encode[1] = { (int)encode_bit };
  tensor_desc tOut(CUPHY_BIT, 1, dims_encode);
  int8_t *input_gpu;
  cudaMalloc(&input_gpu, tIn.sz_bytes * kNumCodeBlocks);
  int8_t *encode_gpu;
  cudaMalloc(&encode_gpu, tOut.sz_bytes * kNumCodeBlocks);
  for (size_t i = 0; i < kNumCodeBlocks; i++) {
    cudaMemcpy(input_gpu + i * tIn.sz_bytes, input[i], tIn.sz_bytes, cudaMemcpyHostToDevice);
  }
  tIn.data = input_gpu;
  tOut.data = encode_gpu;

  LDPC_encode ldpc_encode(kBaseGraph, Zc, kNumRows, kNumCodeBlocks);
  ldpc_encode.encode(tIn, tOut);
  //cudaStreamSynchronize(stream);

  for (size_t i = 0; i < kNumCodeBlocks; i++) {
    cudaMemcpy(encode[i] + BitsToBytes(2 * Zc), encode_gpu + i * tOut.sz_bytes, tOut.sz_bytes, cudaMemcpyDeviceToHost);
    memset(encode[i], 0, BitsToBytes(2*Zc));
  }
  cudaFree(input_gpu);
  cudaFree(encode_gpu);
  return;

  /*cuphyStatus_t status;

  size_t desc_size  = 0;
  size_t align_size = 0;
  size_t workspace_size = 0;
  int max_UEs = PDSCH_MAX_UES_PER_CELL_GROUP;
  status = cuphyLDPCEncodeGetDescrInfo(&desc_size, &align_size, max_UEs, &workspace_size);
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphyLDPCEncodeGetDescrInfo failed %d\n", status);
    exit(-1);
  }

  uint8_t *d_ldpc_desc;
  cudaMalloc(&d_ldpc_desc, desc_size);
  uint8_t *d_workspace;
  cudaMalloc(&d_workspace, workspace_size);
  uint8_t *h_ldpc_desc = (uint8_t *)malloc(desc_size);
  //cudaHostAlloc(&h_ldpc_desc, desc_size, 0);
  uint8_t *h_workspace = (uint8_t *)malloc(workspace_size);
  //cudaHostAlloc(&h_workspace, workspace_size, 0);

  printf("desc_size: %ld\n", desc_size);
  printf("align_size: %ld\n", align_size);
  printf("workspace_size: %ld\n", workspace_size);

  cuphyTensorDescriptor_t desc_input;
  status = cuphyCreateTensorDescriptor(&desc_input);
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphyCreateTensorDescriptor failed %d\n", status);
    exit(-1);
  }
  std::printf("input bit %ld\n", input_bit);
  int dims_input[CUPHY_DIM_MAX] = { (int)input_bit, (int)1 };
  int strides_input[CUPHY_DIM_MAX];
  cuphyDataType_t type_input = CUPHY_BIT;
  int num_dims_input = 2;
  status = cuphySetTensorDescriptor(desc_input, type_input, num_dims_input, dims_input, nullptr, CUPHY_TENSOR_ALIGN_DEFAULT);
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphySetTensorDescriptor failed %d\n", status);
    exit(-1);
  }
  status = cuphyGetTensorDescriptor(desc_input, CUPHY_DIM_MAX, &type_input, &num_dims_input, dims_input, strides_input);
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphyGetTensorDescriptor failed %d\n", status);
    exit(-1);
  }
  std::printf("input tensor: type_input %d, num_dims_input %d\n", type_input, num_dims_input);
  for (int i = 0; i < CUPHY_DIM_MAX; i++) {
    std::printf("dims_input[%d] = %d\n", i, dims_input[i]);
  }
  for (int i = 0; i < CUPHY_DIM_MAX; i++) {
    std::printf("strides_input[%d] = %d\n", i, strides_input[i]);
  }
  size_t sz_input;
  status = cuphyGetTensorSizeInBytes(desc_input, &sz_input);
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphyGetTensorSizeInBytes failed %d\n", status);
    exit(-1);
  }
  std::printf("sz_input = %ld\n\n", sz_input);
  void *input_gpus[kNumCodeBlocks];
  for (int i = 0; i < kNumCodeBlocks; i++) {
    cudaMalloc(&(input_gpus[i]), sz_input);
    cudaMemcpy(input_gpus[i], input[i], BitsToBytes(input_bit), cudaMemcpyHostToDevice);
  }

  const int MAXV = (1 == kBaseGraph) ? CUPHY_LDPC_MAX_BG1_VAR_NODES : CUPHY_LDPC_MAX_BG2_VAR_NODES;
  size_t encode_buf_bit = Zc * MAXV;
  cuphyTensorDescriptor_t desc_encode;
  status = cuphyCreateTensorDescriptor(&desc_encode);
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphyCreateTensorDescriptor failed %d\n", status);
    exit(-1);
  }
  std::printf("encode bit %ld\n", encode_buf_bit);
  int dims_encode[CUPHY_DIM_MAX] = { (int)encode_buf_bit, (int)1 };
  int strides_encode[CUPHY_DIM_MAX];
  cuphyDataType_t type_encode = CUPHY_BIT;
  int num_dims_encode = 2;
  status = cuphySetTensorDescriptor(desc_encode, type_encode, num_dims_encode, dims_encode, nullptr, CUPHY_TENSOR_ALIGN_DEFAULT);
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphySetTensorDescriptor failed %d\n", status);
    exit(-1);
  }
  status = cuphyGetTensorDescriptor(desc_encode, CUPHY_DIM_MAX, &type_encode, &num_dims_encode, dims_encode, strides_encode);
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphyGetTensorDescriptor failed %d\n", status);
    exit(-1);
  }
  std::printf("encode tensor: type_encode %d, num_dims_encode %d\n", type_encode, num_dims_encode);
  for (int i = 0; i < CUPHY_DIM_MAX; i++) {
    std::printf("dims_encode[%d] = %d\n", i, dims_encode[i]);
  }
  for (int i = 0; i < CUPHY_DIM_MAX; i++) {
    std::printf("strides_encode[%d] = %d\n", i, strides_encode[i]);
  }
  size_t sz_encode;
  status = cuphyGetTensorSizeInBytes(desc_encode, &sz_encode);
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphyGetTensorSizeInBytes failed %d\n", status);
    exit(-1);
  }
  std::printf("sz_encode = %ld\n\n", sz_encode);
  void *encode_gpus[kNumCodeBlocks];
  for (int i = 0; i < kNumCodeBlocks; i++) {
    cudaMalloc(&(encode_gpus[i]), sz_encode);
  }

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  //cuphyLDPCEncodeLaunchConfig_t launchConfig_ptr;// = (cuphyLDPCEncodeLaunchConfig_t)malloc(sizeof(cuphyLDPCEncodeLaunchConfig) * 10);
  cuphyLDPCEncodeLaunchConfig launchConfig;
  //launchConfig_ptr = &launchConfig;
  //launchConfig.m_desc = d_ldpc_desc;
  //std::printf("launchConfig = %ld, %ld, %ld\n", sizeof(launchConfig), sizeof(cuphyLDPCEncodeLaunchConfig), sizeof(CUDA_KERNEL_NODE_PARAMS));
  status = cuphySetupLDPCEncode(&launchConfig,
                             desc_input, // source descriptor
                             nullptr,          // source address
                             desc_encode, // destination descriptor
                             nullptr,          // destination address
                             kBaseGraph,                  // base graph
                             Zc,                   // lifting size
                             true,            // puncture output bits
                             kNumRows,      // max parity nodes
                             0,                  // redundancy version
                             true,
                             kNumCodeBlocks,
                             input_gpus,
                             encode_gpus,
                             h_workspace,
                             d_workspace,
                             h_ldpc_desc,   // host descriptor
                             d_ldpc_desc,   // device descriptor
                             true,                   // do async copy during setup
                             stream);
  launchConfig.m_desc = d_ldpc_desc;
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphySetupLDPCEncode failed %d\n", status);
    exit(-1);
  }
  const CUDA_KERNEL_NODE_PARAMS& kernelNodeParams = launchConfig.m_kernelNodeParams;
  cuLaunchKernel(kernelNodeParams.func,
                kernelNodeParams.gridDimX,
                kernelNodeParams.gridDimY,
                kernelNodeParams.gridDimZ,
                kernelNodeParams.blockDimX,
                kernelNodeParams.blockDimY,
                kernelNodeParams.blockDimZ,
                kernelNodeParams.sharedMemBytes,
                stream,
                kernelNodeParams.kernelParams,
                kernelNodeParams.extra);

  cudaStreamSynchronize(stream);

  for (int i = 0; i < kNumCodeBlocks; i++) {
    cudaMemcpy(encode[i], encode_gpus[i], BitsToBytes(encode_bit), cudaMemcpyDeviceToHost);
  }
  for (int i = 0; i < kNumCodeBlocks; i++) {
    cudaFree(input_gpus[i]);
    cudaFree(encode_gpus[i]);
  }
  for (int i = 0; i < kNumCodeBlocks; i++) {
    memset(encode[i], 0, BitsToBytes(2*Zc));
    for (int j = 0; j < BitsToBytes(2 * Zc); j++) {
      encode[i][j] = 0;
    }
  }
  cudaFree(d_ldpc_desc);
  cudaFree(d_workspace);
  free(h_ldpc_desc);
  free(h_workspace);
  cudaStreamDestroy(stream);*/
}

void ldpc_decode_cuda(
  uint16_t Zc,
  int8_t **llr,
  uint8_t **decoded,
  size_t llr_bit,
  size_t decode_bit
) {
  LDPC_decode decoder(CUPHY_R_32F, kNumRows, kBaseGraph, Zc, kMaxDecoderIters);
  int dims_llr[2] = { (int)llr_bit, (int)kNumCodeBlocks };
  tensor_desc tLLR(CUPHY_R_32F, 2, dims_llr, CUPHY_TENSOR_ALIGN_COALESCE);
  int dims_decode[2] = { (int)decode_bit, (int)kNumCodeBlocks };
  tensor_desc tDecode(CUPHY_BIT, 2, dims_decode, CUPHY_TENSOR_ALIGN_COALESCE);
  float *llr_gpu;
  cudaMalloc(&llr_gpu, tLLR.sz_bytes);
  uint32_t *decoded_gpu;
  cudaMalloc(&decoded_gpu, tDecode.sz_bytes);
  tLLR.data = llr_gpu;
  tDecode.data = decoded_gpu;
  printf("llr_size: %ld, decode_size: %ld\n", tLLR.sz_bytes, tDecode.sz_bytes);
  printf("llr_dims: %d, %d, decode_dims: %d, %d\n", tLLR.dims[0], tLLR.dims[1], tDecode.dims[0], tDecode.dims[1]);
  printf("llr_strides: %d, %d, decode_strides: %d, %d\n", tLLR.strides[0], tLLR.strides[1], tDecode.strides[0], tDecode.strides[1]);

  float *llr_cpu = (float *)malloc(tLLR.sz_bytes);
  for (int i = 0; i < dims_llr[1]; i++) {
    for (int j = 0; j < dims_llr[0]; j++) {
      //printf("%d, %d, %d\n", i, j, llr[i][j]);
      llr_cpu[i * tLLR.strides[1] + j *  tLLR.strides[0]] = float(llr[i][j]) / 8.f;
    }
  }
  cudaMemcpy(llr_gpu, llr_cpu, tLLR.sz_bytes, cudaMemcpyHostToDevice);

  cudaStream_t stream;
  cudaStreamCreate(&stream);
  decoder.decode(tLLR, tDecode, stream);
  cudaStreamSynchronize(stream);

  uint32_t *decoded_cpu = (uint32_t *)malloc(tDecode.sz_bytes);
  cudaMemcpy(decoded_cpu, decoded_gpu, tDecode.sz_bytes, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < kNumCodeBlocks; i++) {
    uint8_t* output_buffer = decoded[i];
    uint8_t* decoded_buffer = (uint8_t*)(decoded_cpu + i * (tDecode.strides[1] / 32));
    for (size_t j = 0; j < BitsToBytes(decode_bit); j++) {
      output_buffer[j] = decoded_buffer[j];
    }
  }
  free(llr_cpu);
  free(decoded_cpu);
  cudaFree(llr_gpu);
  cudaFree(decoded_gpu);
  return;

  /*cuphyStatus_t status;

  cuphyContext_t ctx;
  status = cuphyCreateContext(&ctx, 0);
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphyCreateContext failed %d\n", status);
    exit(-1);
  }

  cuphyLDPCDecoder_t decoder = nullptr;
  status = cuphyCreateLDPCDecoder(ctx, &decoder, 0);
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphyCreateLDPCDecoder failed %d\n", status);
    exit(-1);
  }

  cuphyLDPCDecodeConfigDesc_t config = {
    .llr_type = CUPHY_R_32F,
    .num_parity_nodes = kNumRows,
    .Z = Zc,
    .max_iterations = kMaxDecoderIters,
    .Kb = LdpcNumInputCols(kBaseGraph),
    .flags = CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT,
    .BG = kBaseGraph,
    .algo = 0,
    .workspace = nullptr,
  };

  status = cuphyErrorCorrectionLDPCDecodeSetNormalization(decoder, &config);
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphyErrorCorrectionLDPCDecodeSetNormalization failed %d\n", status);
    exit(-1);
  }

  cuphyTensorDescriptor_t desc_llr;
  status = cuphyCreateTensorDescriptor(&desc_llr);
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphyCreateTensorDescriptor failed %d\n", status);
    exit(-1);
  }
  std::printf("llr bit %ld\n", llr_bit);
  int dims_llr[CUPHY_DIM_MAX] = { (int)llr_bit, (int)kNumCodeBlocks };
  int strides_llr[CUPHY_DIM_MAX];
  cuphyDataType_t type_llr = CUPHY_R_32F;
  int num_dims_llr = 2;
  status = cuphySetTensorDescriptor(desc_llr, type_llr, num_dims_llr, dims_llr, nullptr, CUPHY_TENSOR_ALIGN_COALESCE);
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphySetTensorDescriptor failed %d\n", status);
    exit(-1);
  }
  status = cuphyGetTensorDescriptor(desc_llr, CUPHY_DIM_MAX, &type_llr, &num_dims_llr, dims_llr, strides_llr);
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphyGetTensorDescriptor failed %d\n", status);
    exit(-1);
  }
  std::printf("llr tensor: type_llr %d, num_dims_llr %d\n", type_llr, num_dims_llr);
  for (int i = 0; i < CUPHY_DIM_MAX; i++) {
    std::printf("dims_llr[%d] = %d\n", i, dims_llr[i]);
  }
  for (int i = 0; i < CUPHY_DIM_MAX; i++) {
    std::printf("strides_llr[%d] = %d\n", i, strides_llr[i]);
  }
  size_t sz_llr;
  status = cuphyGetTensorSizeInBytes(desc_llr, &sz_llr);
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphyGetTensorSizeInBytes failed %d\n", status);
    exit(-1);
  }
  std::printf("sz_llr = %ld\n\n", sz_llr);

  cuphyLDPCDecodeDesc_t desc = {
    .config = config,
    .num_tbs = kNumCodeBlocks,
  };

  float *llr_cpu = (float *)malloc(sz_llr);
  for (int i = 0; i < dims_llr[1]; i++) {
    for (int j = 0; j < dims_llr[0]; j++) {
      llr_cpu[i * strides_llr[1] + j * strides_llr[0]] = float(llr[i][j]) / 8.f;
    }
  }
  float *llr_gpu;
  cudaMalloc(&llr_gpu, sz_llr);
  cudaMemcpy(llr_gpu, llr_cpu, sz_llr, cudaMemcpyHostToDevice);

  for (int i = 0; i < kNumCodeBlocks; i++) {
    desc.llr_input[i] = {
      .addr = llr_gpu + i * strides_llr[1],
      .stride_elements = (int32_t)strides_llr[1],
      .num_codewords = 1,
    };
  }
  desc.llr_input[0] = {
    .addr = llr_gpu,
    .stride_elements = (int32_t)strides_llr[1],
    .num_codewords = 1,
  };


  cuphyTensorDescriptor_t desc_decode;
  status = cuphyCreateTensorDescriptor(&desc_decode);
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphyCreateTensorDescriptor failed %d\n", status);
    exit(-1);
  }
  std::printf("decode bit %ld\n", decode_bit);
  int dims_decode[CUPHY_DIM_MAX] = { (int)decode_bit, (int)kNumCodeBlocks };
  int strides_decode[CUPHY_DIM_MAX];
  cuphyDataType_t type_decode = CUPHY_BIT;
  int num_dims_decode = 2;
  status = cuphySetTensorDescriptor(desc_decode, type_decode, num_dims_decode, dims_decode, nullptr, CUPHY_TENSOR_ALIGN_COALESCE);
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphySetTensorDescriptor failed %d\n", status);
    exit(-1);
  }
  status = cuphyGetTensorDescriptor(desc_decode, CUPHY_DIM_MAX, &type_decode, &num_dims_decode, dims_decode, strides_decode);
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphyGetTensorDescriptor failed %d\n", status);
    exit(-1);
  }
  std::printf("decode tensor: type_decode %d, num_dims_decode %d\n", type_decode, num_dims_decode);
  for (int i = 0; i < CUPHY_DIM_MAX; i++) {
    std::printf("dims_decode[%d] = %d\n", i, dims_decode[i]);
  }
  for (int i = 0; i < CUPHY_DIM_MAX; i++) {
    std::printf("strides_decode[%d] = %d\n", i, strides_decode[i]);
  }
  size_t sz_decode;
  status = cuphyGetTensorSizeInBytes(desc_decode, &sz_decode);
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphyGetTensorSizeInBytes failed %d\n", status);
    exit(-1);
  }
  std::printf("sz_decode = %ld\n\n", sz_decode);


  uint32_t *decoded_gpu;
  cudaMalloc(&decoded_gpu, sz_decode);
  for (int i = 0; i < kNumCodeBlocks; i++) {
    desc.tb_output[i] = {
      .addr = decoded_gpu + i * (strides_decode[1] / 32),
      .stride_words = strides_decode[1] / 32,
      .num_codewords = 1,
    };
  }
  desc.tb_output[0] = {
    .addr = decoded_gpu,
    .stride_words = strides_decode[1] / 32,
    .num_codewords = dims_decode[1],
  };

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  //status = cuphyErrorCorrectionLDPCTransportBlockDecode(decoder, &desc, stream);
  status = cuphyErrorCorrectionLDPCDecode(decoder, desc_decode, decoded_gpu, desc_llr, llr_gpu, &config, stream);
  if (status != CUPHY_STATUS_SUCCESS) {
    std::fprintf(stderr, "cuphyErrorCorrectionLDPCTransportBlockDecode failed %d\n", status);
    exit(-1);
  }

  cudaStreamSynchronize(stream);

  uint32_t *decoded_cpu = (uint32_t *)malloc(sz_decode);
  cudaMemcpy(decoded_cpu, decoded_gpu, sz_decode, cudaMemcpyDeviceToHost);
  for (size_t i = 0; i < kNumCodeBlocks; i++) {
    uint8_t* output_buffer = decoded[i];
    uint8_t* decoded_buffer = (uint8_t*)(decoded_cpu + i * (strides_decode[1] / 32));
    for (size_t j = 0; j < BitsToBytes(decode_bit); j++) {
      output_buffer[j] = decoded_buffer[j];
    }
  }

  free(llr_cpu);
  free(decoded_cpu);
  cudaFree(llr_gpu);
  cudaFree(decoded_gpu);
  cuphyDestroyLDPCDecoder(decoder);
  cuphyDestroyContext(ctx);
  cudaStreamDestroy(stream);*/
}