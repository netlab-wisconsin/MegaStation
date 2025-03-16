/**
 * @file fft_op.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief fft header file (include fft & ifft)
 * @version 0.1
 * @date 2023-12-10
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cufftXt.h>

#include <memory>
#include <stdexcept>
#include <vector>

#include "fft_utils.h"
#include "matrix/matrix.h"
#include "mega_complex.h"

namespace mega {

class BaseFFT {
 protected:
  cufftHandle plan;    //!< cufft fft plan
  Complex* work_area;  //!< work area
  pilotInfo hinfo;     //!< fft info (host)

  std::shared_ptr<int> lifetime_ptr;  //!< fft lifetime
 public:
  BaseFFT() : lifetime_ptr(std::make_shared<int>()) {}
  virtual void operator()(const Matrix& in, const Matrix& out,
                          cudaStream_t stream = nullptr) {
    throw std::runtime_error("fft not implemented");
  }
  virtual ~BaseFFT() {
    if (lifetime_ptr.use_count() == 1) {
      if (work_area) {
        cudaFree(work_area);
        work_area = nullptr;
        cufftDestroy(plan);
      }
    }
  }
};

template <typename QuanT = short>
class UplinkFFT : public BaseFFT {
 public:
  UplinkFFT() = default;
  UplinkFFT(uint32_t ofdm_start, uint32_t ofdm_ca, uint32_t ofdm_num,
            uint32_t bs_num);
  /**
   * @brief Uplink FFT (ensure out is not truncated)
   *
   * @param in input data (short)
   * @param out output data (Complex)
   * @param stream cuda stream
   */
  void operator()(const Matrix& in, const Matrix& out,
                  cudaStream_t stream = nullptr) override;
};

struct PilotSign {
  static std::vector<Complex*>
      pilot_sign_ptrs;  //!< a vector of pilot sign (device ptr, different
                        //!< devices)

  /**
   * @brief Initialize pilot sign
   *
   * @param ofdm_num number of OFDM symbols
   * @param num_device number of devices
   */
  static void init(uint32_t ofdm_num, uint8_t num_device = 1);
  /**
   * @brief Destroy pilot sign
   *
   */
  static void destroy() {
    for (auto& pilot_sign_ptr : PilotSign::pilot_sign_ptrs) {
      if (pilot_sign_ptr) {
        cudaFree(pilot_sign_ptr);
        pilot_sign_ptr = nullptr;
      }
    }
  }
};

template <typename QuanT = short>
class PilotFFT : public BaseFFT {
 public:
  PilotFFT() = default;
  PilotFFT(uint32_t ofdm_start, uint32_t ofdm_ca, uint32_t ofdm_num,
           uint32_t bs_num, uint32_t ue_num, uint32_t sc_group);
  /**
   * @brief Pliot FFT (ensure out is not truncated)
   *
   * @param in input data (short)
   * @param out output data (Complex)
   * @param stream cuda stream
   */
  void operator()(const Matrix& in, const Matrix& out,
                  cudaStream_t stream = nullptr) override;
  void operator()(const Matrix& in, const Matrix& out, uint32_t ue_start,
                  cudaStream_t stream = nullptr);
};

template <typename QuanT = short>
class DownlinkIFFT : public BaseFFT {
 public:
  DownlinkIFFT() = default;
  DownlinkIFFT(uint32_t ofdm_start, uint32_t ofdm_ca, uint32_t ofdm_num,
               uint32_t bs_num);
  void operator()(const Matrix& in, const Matrix& out,
                  cudaStream_t stream = nullptr) override;
};

}  // namespace mega