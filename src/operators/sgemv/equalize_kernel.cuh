/**
 * @file equalize_kernel.cuh
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Equalization + Demodulation with modified (singular) GEMV (generalized
 * matrix-vector multiplication)
 * @version 0.1
 * @date 2023-12-01
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <sys/types.h>

#include <cstdint>

#include "mega_complex.h"
#include "mmlt.cuh"
#include "modulation/demodulation.h"

namespace mega {

class EqualizeKernel {
 public:
  struct Params {
    // Equalization Related
    coord<3> problem_size;       //!< [ueAntNum, bsAntNum, subcarrierNum]
    coord<2> csi_shape;          //!< [ueAntNum, bsAntNum]
    const Complex* batched_csi;  //!< batched CSI matrix, each Column Major
                                 //!< [ueAntNum, bsAntNum, subcarrierNum]
    const Complex*
        batched_bsVec;  //!< batched input vector [bsAntNum, 1, subcarrierNum]
    half*
        batched_ueVec;  //!< batched output vector [ueAntNum, 1, subcarrierNum]
    uint64_t csi_stride;    //!< stride of each batch of batched_csi
    uint64_t bsVec_stride;  //!< stride of each batch of batched_bsVec
    uint64_t ueVec_stride;  //!< stride of each batch of batched_ueVec
    uint64_t csi_skip;  //!< move to next batch of batched_csi every csi_skip

    // Demodulation Related
    uint8_t mod_order;   //!< modulation order
    uint64_t cb_size;    //!< code block size
    uint64_t cb_stride;  //!< code block stride (including punctured zeros)
    uint64_t zeros;  //!< number of prefix punctured zeros in each code block
                     //!< (prepared for LDPC Decoder)
    uint64_t valid_bits;  //!< number of valid bits of one transport block (sum
                          //!< of all code blocks, excluding punctured zeros)

    /**
     * @brief Construct a new Params object (General case)
     *
     */
    Params(coord<3> problem_size, coord<2> csi_shape,
           const Complex* batched_csi, const Complex* batched_bsVec,
           half* batched_ueVec, uint64_t csi_stride, uint64_t bsVec_stride,
           uint64_t ueVec_stride, uint64_t csi_skip, uint8_t mod_order,
           uint64_t cb_size, uint64_t cb_stride, uint64_t zeros,
           uint64_t valid_bits)
        : problem_size(problem_size),
          csi_shape(csi_shape),
          batched_csi(batched_csi),
          batched_bsVec(batched_bsVec),
          batched_ueVec(batched_ueVec),
          csi_stride(csi_stride),
          bsVec_stride(bsVec_stride),
          ueVec_stride(ueVec_stride),
          csi_skip(csi_skip),
          mod_order(mod_order),
          cb_size(cb_size),
          cb_stride(cb_stride),
          zeros(zeros),
          valid_bits(valid_bits) {}

    /**
     * @brief Construct a new Params object (Common case)
     *
     */
    Params(coord<3> problem_size, const Complex* batched_csi,
           const Complex* batched_bsVec, half* batched_ueVec,
           uint64_t ueVec_stride, uint64_t csi_skip, uint8_t mod_order,
           uint64_t cb_size, uint64_t cb_stride, uint64_t zeros,
           uint64_t valid_bits)
        : problem_size(problem_size),
          csi_shape({problem_size.row, problem_size.col}),
          batched_csi(batched_csi),
          batched_bsVec(batched_bsVec),
          batched_ueVec(batched_ueVec),
          csi_stride(problem_size.row * problem_size.col),
          bsVec_stride(problem_size.dep),
          ueVec_stride(ueVec_stride),
          csi_skip(csi_skip),
          mod_order(mod_order),
          cb_size(cb_size),
          cb_stride(cb_stride),
          zeros(zeros),
          valid_bits(valid_bits) {}
  };

  /**
   * @brief Construct a new Equalize object
   *
   */
  MMLT_DEVICE
  EqualizeKernel() {}

  static constexpr uint8_t kK = 8;
  static constexpr uint8_t kMaxDemod = 8;

  template <const uint8_t layout, const uint8_t elem>
  union EqualSharedMem {
    struct {
      MMLTBuf<Complex, 8 * layout * elem * kK> csi;
      MMLTBuf<Complex, 8 * elem * kK> bsVec;
    };
    MMLTBuf<half, 8 * 8 * layout * elem * kMaxDemod> out;

    MMLT_DEVICE
    EqualSharedMem() {}  // default constructor deleted?
  };

  template <const uint8_t layout>
  struct ThreadID {
    uint32_t warp_id;
    uint32_t lane_id;
    uint32_t warp_idx;
    uint32_t warp_idy;
    uint32_t lane_idx;
    uint32_t lane_idy;

    MMLT_DEVICE
    ThreadID()
        : warp_id(threadIdx.x / kThreadCount),
          lane_id(threadIdx.x % kThreadCount),
          warp_idx(warp_id % layout),
          warp_idy(warp_id / layout),
          lane_idx(lane_id % 8),
          lane_idy(lane_id / 8) {}
  };

  template <const uint8_t layout, const uint8_t elem>
  struct InIter {
    const uint16_t block_size_csi = 8 * layout * elem;
    const uint16_t block_size_bsVec = (4 + 4) * elem;

    const uint16_t total_thr = 64 * layout;
    const uint16_t total_elem_csi = block_size_csi * kK;
    const uint16_t total_elem_bsVec = block_size_bsVec * kK;

    const uint16_t load_iter_csi = total_elem_csi / total_thr;
    const uint16_t load_interval_csi = total_thr / block_size_csi;
    const uint16_t load_iter_bsVec = total_elem_bsVec / total_thr;
    const uint16_t load_interval_bsVec = total_thr / block_size_bsVec;

    const coord<2>& pstride;  // {csi stride, bsVec stride}
    const coord<3>& psize;    // {ue, bs, carrier}
    struct GMem {
      const Complex* csi;
      const Complex* bsVec;
    } gmem;
    struct SMem {
      Complex* csi;
      Complex* bsVec;
    } smem;
    bool pred_csi_ue;
    bool pred_bs_carrier;
    uint64_t off_csi_bs;
    uint64_t off_bsVec_bs;

    struct {
      MMLTArr<Complex, elem>* csi;
      MMLTArr<Complex, elem>* bsVec;
    } rsmem;

    MMLT_DEVICE
    InIter(const coord<2>& pstride_, const coord<3>& psize_, const GMem& gmem_,
           const SMem& smem_, const ThreadID<layout>& tid)
        : pstride(pstride_), psize(psize_) {
      uint64_t ue_id =
          blockIdx.x * block_size_csi + threadIdx.x % block_size_csi;
      off_csi_bs = threadIdx.x / block_size_csi;
      gmem.csi = gmem_.csi + ue_id + off_csi_bs * pstride.row;

      uint64_t carrier_id =
          blockIdx.y * block_size_bsVec + threadIdx.x % block_size_bsVec;
      off_bsVec_bs = threadIdx.x / block_size_bsVec;
      gmem.bsVec = gmem_.bsVec + carrier_id + off_bsVec_bs * pstride.col;

      smem.csi = smem_.csi + threadIdx.x;
      smem.bsVec = smem_.bsVec + threadIdx.x;

      pred_csi_ue = ue_id < psize.row;
      pred_bs_carrier = carrier_id < psize.dep;

      rsmem.csi = reinterpret_cast<MMLTArr<Complex, elem>*>(smem_.csi) +
                  8 * tid.warp_idx + tid.lane_idx;
      rsmem.bsVec = reinterpret_cast<MMLTArr<Complex, elem>*>(smem_.bsVec) +
                    4 * tid.warp_idy + tid.lane_idy;
    }

    MMLT_DEVICE
    void load_next_smem() {
      MMLT_UNROLL
      for (int i = 0; i < load_iter_csi; i++) {
        Complex tmp = Complex();
        if (pred_csi_ue && off_csi_bs < psize.col) {
          tmp = gmem.csi[i * load_interval_csi * pstride.row];
        }
        smem.csi[i * load_interval_csi * block_size_csi] = tmp;
        off_csi_bs += load_interval_csi;
      }
      gmem.csi += kK * pstride.row;

      MMLT_UNROLL
      for (int i = 0; i < load_iter_bsVec; i++) {
        Complex tmp = Complex();
        if (pred_bs_carrier && off_bsVec_bs < psize.col) {
          tmp = gmem.bsVec[i * load_interval_bsVec * pstride.col];
        }
        smem.bsVec[i * load_interval_bsVec * block_size_bsVec] = tmp;
        off_bsVec_bs += load_interval_bsVec;
      }
      gmem.bsVec += kK * pstride.col;
    }

    MMLT_DEVICE
    void load_next_reg(MMLTArr<Complex, elem>& csi_reg,
                       MMLTArr<Complex, elem>& bsVec_reg, const int& k_iter) {
      csi_reg = rsmem.csi[8 * layout * k_iter];
      bsVec_reg = rsmem.bsVec[8 * k_iter];
    }

    MMLT_DEVICE
    void reset_smem_offset() {}
  };

  template <const uint8_t layout, const uint8_t elem>
  struct OutIter {
    const uint16_t block_count_ue = 8 * layout;
    const uint16_t block_count_car = 8;

    const uint16_t block_size_ue = block_count_ue * elem;
    const uint16_t block_size_car = block_count_car * elem;

    const uint16_t total_thr = 64 * layout;

    const uint16_t store_iter = (block_count_ue * block_size_car) / total_thr;
    const uint16_t store_interval = total_thr / block_size_car;

    const uint8_t& mod_order;
    const coord<2>& shape;

    bool pred;
    uint64_t off_ue;

    half* rsmem;
    half* smem;
    half* gmem;

    MMLT_DEVICE
    OutIter(const coord<2>& shape_, half* gmem_, half* smem_,
            const uint8_t& mod_order_, const coord<2>& cb,
            const uint64_t& zeros, const uint64_t& skip,
            const ThreadID<layout>& tid)
        : shape(shape_), mod_order(mod_order_) {
      rsmem = smem_ + ((8 * tid.warp_idx + tid.lane_idx) * block_count_car +
                       (4 * tid.warp_idy + tid.lane_idy)) *
                          mod_order * elem;
      uint64_t ue_id =
          block_size_ue * blockIdx.x + (threadIdx.x / block_size_car) * elem;
      uint64_t carrier_id = blockIdx.z * skip + block_size_car * blockIdx.y +
                            threadIdx.x % block_size_car;

      uint32_t bit_count = carrier_id * mod_order;
      uint32_t codeblock_count = bit_count / cb.row;
      uint32_t bit_offset = bit_count % cb.row;

      uint32_t carrier_offset = codeblock_count * cb.col + zeros + bit_offset;

      gmem = gmem_ + ue_id * shape.col + carrier_offset;
      smem = smem_ + threadIdx.x * mod_order;

      pred = carrier_offset < shape.col;
      off_ue = ue_id;
    }

    MMLT_DEVICE
    void store_next_reg(MMLTArr<Complex, elem * elem>& accum_reg,
                        const int& i_iter) {
      MMLT_UNROLL
      for (int j = 0; j < elem; j++) {
        half* smem_demod = rsmem + j * mod_order;
        switch (mod_order) {
          case 2:
            demodQPSK(accum_reg.storage[j + i_iter * elem], smem_demod);
            break;
          case 4:
            demod16QAM(accum_reg.storage[j + i_iter * elem], smem_demod);
            break;
          case 6:
            demod64QAM(accum_reg.storage[j + i_iter * elem], smem_demod);
            break;
          case 8:
            demod256QAM(accum_reg.storage[j + i_iter * elem], smem_demod);
            break;
          default:
            break;
        }
      }
    }

    MMLT_DEVICE
    void store_next_smem() {
      MMLT_UNROLL
      for (int i = 0; i < store_iter; i++) {
        if (pred && ((off_ue + i * store_interval * elem) < shape.row)) {
          switch (mod_order) {
            case 2:
              reinterpret_cast<float*>(gmem + i * store_interval * elem *
                                                  shape.col)[0] =
                  reinterpret_cast<float*>(
                      smem)[i * store_interval * block_size_car];
              break;
            case 4:
              reinterpret_cast<double*>(gmem + i * store_interval * elem *
                                                   shape.col)[0] =
                  reinterpret_cast<double*>(
                      smem)[i * store_interval * block_size_car];
              break;
            case 6:
              reinterpret_cast<MMLTArr<float, 3>*>(
                  gmem + i * store_interval * elem * shape.col)[0] =
                  reinterpret_cast<MMLTArr<float, 3>*>(
                      smem)[i * store_interval * block_size_car];
              break;
            case 8:
              reinterpret_cast<MMLTArr<float, 4>*>(
                  gmem + i * store_interval * elem * shape.col)[0] =
                  reinterpret_cast<MMLTArr<float, 4>*>(
                      smem)[i * store_interval * block_size_car];
              break;
            default:
              break;
          }
        }
      }
      gmem += shape.col;
      off_ue += 1;
    }
  };

  template <const uint8_t layout, const uint8_t elem>
  MMLT_DEVICE void mmlt(const Params params) {
    // One layer: 8 * (4 + 4) * elem matrix
    // layout: how many rows of layers (max 4)
    // 4: elements along k axis
    __shared__ EqualSharedMem<layout, elem> smem;

    ThreadID<layout> tid;

    MMLTArr<Complex, elem> csi_reg;
    MMLTArr<Complex, elem> bsVec_reg;
    MMLTArr<Complex, elem * elem> accum_reg;

    InIter<layout, elem> in_iter(
        {params.csi_shape.row, params.bsVec_stride},
        {params.problem_size.row, params.problem_size.col, params.csi_skip},
        {params.batched_csi + blockIdx.z * params.csi_stride,
         params.batched_bsVec + blockIdx.z * params.csi_skip},
        {smem.csi.storage, smem.bsVec.storage}, tid);

    accum_reg.fill(Complex());

    int k_iterations = (params.problem_size.col + kK - 1) / kK;

    MMLT_LOOP
    for (; k_iterations > 0; k_iterations--) {
      in_iter.load_next_smem();

      __syncthreads();

      MMLT_UNROLL
      for (int k = 0; k < kK; k++) {
        in_iter.load_next_reg(csi_reg, bsVec_reg, k);
        mmlt_op(accum_reg, csi_reg, bsVec_reg);
      }
      in_iter.reset_smem_offset();

      __syncthreads();
    }

    OutIter<layout, elem> out_iter(
        {params.problem_size.row, params.ueVec_stride}, params.batched_ueVec,
        smem.out.storage, params.mod_order, {params.cb_size, params.cb_stride},
        params.zeros, params.csi_skip, tid);
    MMLT_UNROLL
    for (int i = 0; i < elem; i++) {
      out_iter.store_next_reg(accum_reg, i);
      __syncthreads();

      out_iter.store_next_smem();
      __syncthreads();
    }
  }

  /**
   * @brief Equalization + Demodulation with modified (singular) GEMV
   *
   * @param params parameters of the equalization kernel
   */
  /*** old naive GEMV implementation ***/
  /* MMLT_DEVICE
  void operator()(const Params& params) {
    for (uint32_t subcarrier_id = blockIdx.z;
         subcarrier_id < params.problem_size.dep; subcarrier_id += gridDim.z) {
      uint32_t ue_id = blockIdx.x * kThreadCount + threadIdx.x;

      const Complex* csi_ptr = params.batched_csi + ue_id;
      const Complex* bsVec_ptr = params.batched_bsVec;

      csi_ptr += (subcarrier_id / params.csi_skip) * params.csi_stride;
      bsVec_ptr += subcarrier_id * params.bsVec_stride;

      Complex accum = Complex();

      MMLT_LOOP
      for (uint32_t bs_id = 0; bs_id < params.problem_size.col; ++bs_id) {
        Complex a = Complex();
        if (ue_id < params.problem_size.row) {
          a = *csi_ptr;
        }
        csi_ptr += params.csi_shape.row;

        Complex b = *bsVec_ptr;
        bsVec_ptr += 1;

        accum += a * b;
      }

      half reg_demod[kMaxDemod];
      switch (params.mod_order) {
        case 2:
          demodQPSK(accum, reg_demod);
          break;
        case 4:
          demod16QAM(accum, reg_demod);
          break;
        case 6:
          demod64QAM(accum, reg_demod);
          break;
        case 8:
          demod256QAM(accum, reg_demod);
          break;
        default:
          break;
      }

      uint32_t bit_count = subcarrier_id * params.mod_order;
      uint32_t codeblock_count = bit_count / params.cb_size;
      uint32_t bit_offset = bit_count % params.cb_size;

      half* ueVec_ptr = params.batched_ueVec + ue_id * params.ueVec_stride +
                        codeblock_count * params.cb_stride + params.zeros +
                        bit_offset;

      MMLT_UNROLL
      for (uint32_t out_bit = 0; out_bit < kMaxDemod; ++out_bit) {
        if (out_bit >= params.mod_order ||
            (bit_count + out_bit) >= params.valid_bits) {
          break;
        }
        if (ue_id < params.problem_size.row) {
          ueVec_ptr[out_bit] = reg_demod[out_bit];
        }
      }
    }
  } */
};

/**
 * @brief Launch the equalization kernel wrapper
 *
 * @param params parameters of the equalization kernel
 */
template <const uint8_t layout, const uint8_t elem>
__global__ __launch_bounds__(256) void equalize_kernel(
    const EqualizeKernel::Params params) {
  EqualizeKernel().mmlt<layout, elem>(params);
}

}  // namespace mega