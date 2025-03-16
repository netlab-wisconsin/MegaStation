/**
 * @file beamform.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Beamform launcher (zero-forcing)
 * @version 0.1
 * @date 2023-12-07
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cublas_v2.h>
#include <cuda.h>
#include <cuda_runtime.h>
#include <cusolverDn.h>

#include <cstdint>
#include <memory>

#include "matrix/matrix.h"
#include "mega_complex.h"

namespace mega {

class Beamform {
 private:
  cuComplex* buffer;    //!< Aggregated Intermediate Buffer for zf computation
  cuComplex** buf_ptr;  //!< Pointer Array to the intermediate buffers
  cuComplex** csi_ptr;  //!< Pointer Array to the CSI matrices
  int32_t* info_ptr;    //!< Pointer Array to the info array
  cublasHandle_t blas_handle;        //!< cuBLAS handle
  cusolverDnHandle_t solver_handle;  //!< cuSOLVER handle
  uint32_t num_bs;                   //!< Number of base stations
  uint32_t num_ue;                   //!< Number of user equipments
  uint32_t batch_count;              //!< Number of batches

  std::shared_ptr<int> lifetime_ptr;  //!< beamform lifetime

  /**
   * @brief Beamform launcher (zero-forcing)
   *
   * @param csi Complex channel state information
   * @param precode Precode matrix
   * @param batch Number of batches
   * @param stream CUDA stream
   */
  void beamformer(Complex* csi, Complex* precode, uint32_t batch,
                  cudaStream_t stream = nullptr);
  void beamformer_uplink(Complex* csi, uint32_t batch,
                         cudaStream_t stream = nullptr);

 public:
  Beamform() : lifetime_ptr(nullptr){};
  /**
   * @brief Construct a new Beamform object
   *
   * @param num_bs_ Number of base stations
   * @param num_ue_ Number of user equipments
   * @param batch_count_ Number of batches
   */
  Beamform(uint32_t num_bs_, uint32_t num_ue_, uint32_t batch_count_);

  /**
   * @brief Destroy the Beamform object
   *
   */
  ~Beamform();

  /**
   * @brief Beamform launcher (zero-forcing)
   *
   * @param csi Complex channel state information
   * @param precode Precode matrix
   * @param batch_count Number of batches
   * @param stream CUDA stream
   */
  void beamformer(const Matrix& csi, const Matrix& precode,
                  cudaStream_t stream = nullptr);
  void beamform_uplink(const Matrix& csi, cudaStream_t stream = nullptr);
  static void beamform_downlink(const Matrix& csi, const Matrix& precode,
                                cudaStream_t stream = nullptr);
};

}  // namespace mega