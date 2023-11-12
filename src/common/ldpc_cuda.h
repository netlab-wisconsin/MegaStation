#pragma once

#include "cuphy.h"

struct tensor_desc {
  cuphyTensorDescriptor_t desc;
  int nDims;
  int dims[CUPHY_DIM_MAX];
  int strides[CUPHY_DIM_MAX];
  int sz_bytes;
  cuphyDataType_t dtype;
  int rank;

  void *data;

  tensor_desc(cuphyDataType_t dtype_, int nDims_, int dims_[], int flags_ = CUPHY_TENSOR_ALIGN_DEFAULT) :
    nDims(nDims_), dtype(dtype_) {
    cuphyCreateTensorDescriptor(&desc);
    cuphySetTensorDescriptor(desc, dtype, nDims, dims_, nullptr, flags_);
    cuphyGetTensorDescriptor(desc, CUPHY_DIM_MAX, &dtype, &rank, dims, strides);
    cuphyGetTensorSizeInBytes(desc, &sz_bytes);
  }

  tensor_desc(tensor_desc &other) :
    nDims(other.nDims), dtype(other.dtype) {
    cuphyCreateTensorDescriptor(&desc);
    cuphySetTensorDescriptor(desc, dtype, nDims, dims, nullptr, flags_);
    cuphyGetTensorDescriptor(desc, CUPHY_DIM_MAX, &dtype, &rank, dims, strides);
    cuphyGetTensorSizeInBytes(desc, &sz_bytes);
  }

  ~tensor_desc() {
    cuphyDestroyTensorDescriptor(desc);
  }
};

class LDPC_decode {
private:
  cuphyContext_t ctx;
  cuphyLDPCDecoder_t decoder;

  static inline const int compute_kb(int bg) {
    return bg == 1 ? CUPHY_LDPC_BG1_INFO_NODES : CUPHY_LDPC_MAX_BG2_INFO_NODES;
  }

public:
  cuphyLDPCDecodeConfigDesc_t config_desc;

  LDPC_decode(cuphyDataType_t llr_type_, int nParity_, int bG_, int zc_, int max_iter_) {
    cuphyCreateContext(&ctx, 0);
    cuphyCreateLDPCDecoder(ctx, &decoder, 0);
    config_desc = {
      .llr_type = dtype_,
      .num_parity_nodes = nParity_,
      .Z = zc_,
      .max_iterations = max_iter_,
      .Kb = compute_kb(bG_),
      .norm = 0.0,
      .flags = CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT,
      .BG = bG_,
      .algo = 0,
      .workspace = nullptr
    };
    cuphyErrorCorrectionLDPCDecodeSetNormalization(decoder, &config_desc);
  }

  ~LDPC_decode() {
    cuphyDestroyLDPCDecoder(decoder);
    cuphyDestroyContext(ctx);
  }

  /**
   * @brief
   * tLLRs: [batch_count, Zc * (Kb - 2 + mb)]
   * tOut: [batch_count, Zc * Kb]
  */
  void decode(tensor_desc& tLLRs, tensor_desc& tOut, cudaStream_t stream) {
    cuphyErrorCorrectionLDPCDecode(decoder, tOut.desc, tOut.data, tLLRs.desc, tLLRs.data, &config_desc, stream);
  }
};

class LDPC_encode {
private:
  void *h_workspace;
  void *d_workspace;
  void *h_ldpc_desc;
  void *d_ldpc_desc;
  int BG;
  int Zc;
  int nParity;

public:
  LDPC_encode(int bG_, int zc_, int nParity_) :
    BG(bG_), Zc(zc_), nParity(nParity_),
    maxV(bG_ == 1 ? CUPHY_LDPC_MAX_BG1_VAR_NODES : CUPHY_LDPC_MAX_BG2_VAR_NODES) {
    size_t desc_size;
    size_t alloc_size;
    size_t workspace_size;
    int max_UEs = PDSCH_MAX_UES_PER_CELL_GROUP;
    cuphyLDPCEncodeGetDescrInfo(&desc_size, &alloc_size, max_UEs, &workspace_size);

    cudaMalloc(&d_workspace, workspace_size);
    cudaMalloc(&d_ldpc_desc, desc_size);
    h_workspace = malloc(workspace_size);
    h_ldpc_desc = malloc(desc_size);
  }

  ~LDPC_encode() {
    cudaFree(d_workspace);
    cudaFree(d_ldpc_desc);
    free(h_workspace);
    free(h_ldpc_desc);
  }

  /**
   * @brief
   * tIn: [batch_count, Zc * Kb]
   * tOut: [batch_count, Zc * (Kb - 2 + mb)] // TODO: maxV is needed or we can use Kb + mb - 2? Give max parity nodes? puncture output bits?
   * CUPHY_LDPC_NUM_PUNCTURED_NODES = 2
  */
  void encode(tensor_desc& tIn, tensor_desc& tOut, int batch_count, cudaStream_t stream) {
    cuphyLDPCEncodeLaunchConfig launchConfig;
    cuphySetupLDPCEncode(&launchConfig, // launch config (output)
                        tIn.desc, // source descriptor
                        nullptr,  // no batch source address
                        tOut.desc, // destination descriptor
                        nullptr,  // no batch destination address
                        BG, // base graph
                        Zc, // lifting size
                        true, // puncture output bits
                        nParity, // max parity nodes
                        0, // redundancy version
                        true,        // batching
                        batch_count, // batch count
                        tIn.addr(),  // batch in address
                        tOut.addr(), // batch out address
                        h_workspace,
                        d_workspace,
                        h_ldpc_desc,
                        d_ldpc_desc,
                        true, // do async copy during setup
                        stream);
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
  }
};