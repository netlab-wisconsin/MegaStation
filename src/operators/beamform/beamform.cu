/**
 * @file beamform.cu
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Beamforming kernels (mainly zero-forcing)
 * @version 0.1
 * @date 2023-12-07
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "beam_utils.cuh"
#include "beamform.h"
#include "cgemm.h"
#include "mmlt.cuh"

using namespace mega;

Beamform::Beamform(uint32_t num_bs_, uint32_t num_ue_, uint32_t batch_count_)
    : num_bs(num_bs_),
      num_ue(num_ue_),
      batch_count(batch_count_),
      lifetime_ptr(std::make_shared<int>()) {
  cublasCreate(&blas_handle);
  cusolverDnCreate(&solver_handle);

  cudaMalloc(&buffer, sizeof(cuComplex) * num_ue * num_ue * batch_count);
  cudaMalloc(&buf_ptr, sizeof(cuComplex*) * num_bs * batch_count);
  cudaMalloc(&csi_ptr, sizeof(cuComplex*) * num_bs * batch_count);
  cudaMalloc(&info_ptr, sizeof(int32_t) * batch_count);
}

Beamform::~Beamform() {
  if (lifetime_ptr.use_count() == 1) {
    cublasDestroy(blas_handle);
    cusolverDnDestroy(solver_handle);

    cudaFree(buffer);
    cudaFree(buf_ptr);
    cudaFree(csi_ptr);
    cudaFree(info_ptr);

    buffer = nullptr;
    buf_ptr = nullptr;
    csi_ptr = nullptr;
    info_ptr = nullptr;
  }
}

void set_batch_ptr_launch(cuComplex** ptr_array, cuComplex* mat,
                          uint32_t num_ptrs, uint32_t stride, uint32_t skip,
                          cudaStream_t stream = nullptr) {
  dim3 block = get_block_shape();
  dim3 grid = get_grid_shape({num_ptrs, 1, 1}, block.x);

  set_batch_ptr<cuComplex>
      <<<grid, block, 0, stream>>>(ptr_array, mat, num_ptrs, stride, skip);
}

void scale_absmax_launch(const Complex* input, Complex* output,
                         uint64_t num_elements, uint64_t batch_count,
                         uint32_t row, uint32_t col, cudaStream_t stream) {
  uint32_t num_threads =
      num_elements < 1024 ? ((num_elements / 32 + 1) * 32) : 1024;
  dim3 block = dim3(num_threads, 1, 1);
  dim3 grid = dim3(batch_count % 65536, 1, 1);

  size_t smem_size = sizeof(float) * num_threads;

  scale_absmax<<<grid, block, smem_size, stream>>>(input, output, num_elements,
                                                   batch_count, row, col);
}

void Beamform::beamformer_uplink(Complex* csi, uint32_t batch,
                                 cudaStream_t stream) {
  cublasSetStream(blas_handle, stream);
  cusolverDnSetStream(solver_handle, stream);

  cuComplex* csi_blas = reinterpret_cast<cuComplex*>(csi);

  cuComplex alpha = make_cuComplex(1.0f, 0.0f);
  cuComplex beta = make_cuComplex(0.0f, 0.0f);

  int64_t lda = num_ue, ldb = num_ue, ldc = num_ue;

  int64_t stride_in = num_ue * num_bs, stride_out = num_ue * num_ue;

  // Compute buffer = csi * csi^H
  cublasCgemmStridedBatched(blas_handle, CUBLAS_OP_N, CUBLAS_OP_C, num_ue,
                            num_ue, num_bs, &alpha, csi_blas, lda, stride_in,
                            csi_blas, ldb, stride_in, &beta, buffer, ldc,
                            stride_out, batch);
  // CgemmStridedBatched()({{static_cast<int>(num_ue), static_cast<int>(num_ue),
  //                         static_cast<int>(num_bs)},
  //                        {reinterpret_cast<cutComplex*>(csi_blas), lda},
  //                        stride_in,
  //                        {reinterpret_cast<cutComplex*>(csi_blas), ldb},
  //                        stride_in,
  //                        {reinterpret_cast<cutComplex*>(buffer), ldc},
  //                        stride_out,
  //                        {reinterpret_cast<cutComplex*>(buffer), ldc},
  //                        stride_out,
  //                        static_cast<int>(batch)});

  // Compute (buffer)^-1 * csi using Cholesky decomposition

  //// Compute Cholesky decomposition of buffer
  set_batch_ptr_launch(buf_ptr, buffer, batch, stride_out, 1, stream);

  cusolverDnCpotrfBatched(solver_handle, CUBLAS_FILL_MODE_LOWER, num_ue,
                          buf_ptr, lda, info_ptr, batch);

  //// Compute Inverse of buffer via Cholesky decomposition
  ////// (A[i] * X[i] = B[i], A is buffer and B is csi, only support single RHS,
  ////// where X[i] and B[i] should be vectors)
  set_batch_ptr_launch(buf_ptr, buffer, batch * num_bs, stride_out, num_bs,
                       stream);
  set_batch_ptr_launch(csi_ptr, csi_blas, batch * num_bs, num_ue, 1, stream);

  cusolverDnCpotrsBatched(solver_handle, CUBLAS_FILL_MODE_LOWER, num_ue, 1,
                          buf_ptr, lda, csi_ptr, ldb, info_ptr, batch * num_bs);
}

void Beamform::beamformer(Complex* csi, Complex* precode, uint32_t batch,
                          cudaStream_t stream) {
  beamformer_uplink(csi, batch, stream);
  scale_absmax_launch(csi, precode, num_ue * num_bs, batch, num_ue, num_bs,
                      stream);
}

void Beamform::beamformer(const Matrix& csi, const Matrix& precode,
                          cudaStream_t stream) {
  uint32_t batch = csi.dim(csi.nDim() - 1);
  if (precode.nDim() != csi.nDim() ||
      batch != precode.dim(precode.nDim() - 1)) {
    throw std::runtime_error("Csi & Precode mismatch");
  }
  if (batch > batch_count) {
    throw std::runtime_error("Batch size exceeds the maximum batch size");
  }
  beamformer(csi.ptr<Complex>(), precode.ptr<Complex>(), batch, stream);
}

void Beamform::beamform_uplink(const Matrix& csi, cudaStream_t stream) {
  uint32_t batch = csi.dim(csi.nDim() - 1);
  if (batch > batch_count) {
    throw std::runtime_error("Batch size " + std::to_string(batch) +
                             " exceeds the maximum batch size " +
                             std::to_string(batch_count));
  }
  beamformer_uplink(csi.ptr<Complex>(), batch, stream);
}

void Beamform::beamform_downlink(const Matrix& csi, const Matrix& precode,
                                 cudaStream_t stream) {
  uint32_t batch = csi.dim(csi.nDim() - 1);
  uint32_t num_ue = csi.dim(0);
  uint32_t num_bs = csi.dim(1);
  if (precode.nDim() != csi.nDim() ||
      batch != precode.dim(precode.nDim() - 1)) {
    throw std::runtime_error("Csi & Precode mismatch");
  }
  if (num_bs != precode.dim(0) || num_ue != precode.dim(1)) {
    throw std::runtime_error("Precode dimension mismatch");
  }
  scale_absmax_launch(csi.ptr<Complex>(), precode.ptr<Complex>(),
                      num_ue * num_bs, batch, num_ue, num_bs, stream);
}