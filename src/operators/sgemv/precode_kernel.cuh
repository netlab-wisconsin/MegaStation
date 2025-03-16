/**
 * @file precode_kernel.cuh
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Precoding + cyclic shift + Expand kernel with modified (singular) GEMV
 * (generalized matrix-vector multiplication)
 * @version 0.1
 * @date 2023-12-02
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cstdint>

#include "mega_complex.h"
#include "mmlt.cuh"

namespace mega {

class PrecodeKernel {
 public:
  struct Params {
    // Precoding Related
    coord<3> problem_size;   //!< [bsAntNum, ueAntNum, subcarrierNum]
    coord<2> precode_shape;  //!< [bsAntNum, ueAntNum]
    const Complex*
        batched_precode;  //!< batched Precode matrix, each Column Major
                          //!< [ueAntNum, bsAntNum, subcarrierNum]
    const Complex*
        batched_ueVec;  //!< batched input vector [ueAntNum, 1, subcarrierNum]
    Complex*
        batched_bsVec;  //!< batched output vector [bsAntNum, 1, subcarrierNum]
    uint64_t precode_stride;  //!< stride of each batch of batched_precode
    uint64_t ueVec_stride;    //!< stride of each batch of batched_ueVec
    uint64_t bsVec_stride;    //!< stride of each batch of batched_bsVec
    uint64_t precode_skip;    //!< move to next batch of batched_precode every
                              //!< precode_skip

    // Pre-fft Expansion Related
    uint64_t bsVec_start;  //!< start index of valid subcarrier in each row of
                           //!< batched_bsVec

    /**
     * @brief Construct a new Params object (General case)
     *
     */
    Params(coord<3> problem_size, coord<2> precode_shape,
           const Complex* batched_precode, const Complex* batched_ueVec,
           Complex* batched_bsVec, uint64_t precode_stride,
           uint64_t ueVec_stride, uint64_t bsVec_stride, uint64_t precode_skip,
           uint64_t bsVec_start)
        : problem_size(problem_size),
          precode_shape(precode_shape),
          batched_precode(batched_precode),
          batched_ueVec(batched_ueVec),
          batched_bsVec(batched_bsVec),
          precode_stride(precode_stride),
          ueVec_stride(ueVec_stride),
          bsVec_stride(bsVec_stride),
          precode_skip(precode_skip),
          bsVec_start(bsVec_start) {}

    /**
     * @brief Construct a new Params object (Common case)
     *
     */
    Params(coord<3> problem_size, const Complex* batched_precode,
           const Complex* batched_ueVec, Complex* batched_bsVec,
           uint64_t bsVec_stride, uint64_t precode_skip, uint64_t bsVec_start)
        : problem_size(problem_size),
          precode_shape({problem_size.row, problem_size.col}),
          batched_precode(batched_precode),
          batched_ueVec(batched_ueVec),
          batched_bsVec(batched_bsVec),
          precode_stride(problem_size.row * problem_size.col),
          ueVec_stride(problem_size.dep),
          bsVec_stride(bsVec_stride),
          precode_skip(precode_skip),
          bsVec_start(bsVec_start) {}
  };

  /**
   * @brief Construct a new Equalize object
   *
   */
  MMLT_DEVICE
  PrecodeKernel() {}

  static constexpr uint8_t kK = 8;

  template <const uint8_t layout, const uint8_t elem>
  union PrecSharedMem {
    struct {
      MMLTBuf<Complex, 8 * layout * elem * kK> prec;
      MMLTBuf<Complex, 8 * elem * kK> ueVec;
    };
    MMLTBuf<Complex, 8 * 8 * layout * elem> out;

    MMLT_DEVICE
    PrecSharedMem() {}  // default constructor deleted?
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
          warp_idx(warp_id % (2 * layout)),
          warp_idy(warp_id / (2 * layout)),
          lane_idx(lane_id % 4),
          lane_idy(lane_id / 4) {}
  };

  template <const uint8_t layout, const uint8_t elem>
  struct InIter {
    const uint16_t block_size_prec = (4 + 4) * layout * elem;
    const uint16_t block_size_ueVec = 8 * elem;

    const uint16_t total_thr = 64 * layout;
    const uint16_t total_elem_prec = block_size_prec * kK;
    const uint16_t total_elem_ueVec = block_size_ueVec * kK;

    const uint16_t load_iter_prec = total_elem_prec / total_thr;
    const uint16_t load_interval_prec = total_thr / block_size_prec;
    const uint16_t load_iter_ueVec = total_elem_ueVec / total_thr;
    const uint16_t load_interval_ueVec = total_thr / block_size_ueVec;

    const coord<2>& pstride;  // {prec stride, ueVec stride}
    const coord<3>& psize;    // {bs, ue, carrier}
    struct GMem {
      const Complex* prec;
      const Complex* ueVec;
    } gmem;
    struct SMem {
      Complex* prec;
      Complex* ueVec;
    } smem;
    bool pred_prec_bs;
    bool pred_ue_carrier;
    uint64_t off_prec_ue;
    uint64_t off_ueVec_ue;

    struct {
      MMLTArr<Complex, elem>* prec;
      MMLTArr<Complex, elem>* ueVec;
    } rsmem;

    MMLT_DEVICE
    InIter(const coord<2>& pstride_, const coord<3>& psize_, const GMem& gmem_,
           const SMem& smem_, const ThreadID<layout>& tid)
        : pstride(pstride_), psize(psize_) {
      uint64_t bs_id =
          blockIdx.x * block_size_prec + threadIdx.x % block_size_prec;
      off_prec_ue = threadIdx.x / block_size_prec;
      gmem.prec = gmem_.prec + bs_id + off_prec_ue * pstride.row;

      uint64_t carrier_id =
          blockIdx.y * block_size_ueVec + threadIdx.x % block_size_ueVec;
      off_ueVec_ue = threadIdx.x / block_size_ueVec;
      gmem.ueVec = gmem_.ueVec + carrier_id + off_ueVec_ue * pstride.col;

      smem.prec = smem_.prec + threadIdx.x;
      smem.ueVec = smem_.ueVec + threadIdx.x;

      pred_prec_bs = bs_id < psize.row;
      pred_ue_carrier = carrier_id < psize.dep;

      rsmem.prec = reinterpret_cast<MMLTArr<Complex, elem>*>(smem_.prec) +
                   4 * tid.warp_idx + tid.lane_idx;
      rsmem.ueVec = reinterpret_cast<MMLTArr<Complex, elem>*>(smem_.ueVec) +
                    8 * tid.warp_idy + tid.lane_idy;
    }

    MMLT_DEVICE
    void load_next_smem() {
      MMLT_UNROLL
      for (int i = 0; i < load_iter_prec; i++) {
        Complex tmp = Complex();
        if (pred_prec_bs && off_prec_ue < psize.col) {
          tmp = gmem.prec[i * load_interval_prec * pstride.row];
        }
        smem.prec[i * load_interval_prec * block_size_prec] = tmp;
        off_prec_ue += load_interval_prec;
      }
      gmem.prec += kK * pstride.row;

      MMLT_UNROLL
      for (int i = 0; i < load_iter_ueVec; i++) {
        Complex tmp = Complex();
        if (pred_ue_carrier && off_ueVec_ue < psize.col) {
          tmp = gmem.ueVec[i * load_interval_ueVec * pstride.col];
        }
        smem.ueVec[i * load_interval_ueVec * block_size_ueVec] = tmp;
        off_ueVec_ue += load_interval_ueVec;
      }
      gmem.ueVec += kK * pstride.col;
    }

    MMLT_DEVICE
    void load_next_reg(MMLTArr<Complex, elem>& prec_reg,
                       MMLTArr<Complex, elem>& ueVec_reg, const int& k_iter) {
      prec_reg = rsmem.prec[8 * layout * k_iter];
      ueVec_reg = rsmem.ueVec[8 * k_iter];
    }

    MMLT_DEVICE
    void reset_smem_offset() {}
  };

  template <const uint8_t layout, const uint8_t elem>
  struct OutIter {
    const uint16_t block_count_bs = 4 * layout * 2;
    const uint16_t block_count_car = 8;

    const uint16_t block_size_bs = block_count_bs * elem;
    const uint16_t block_size_car = block_count_car * elem;

    const uint16_t total_thr = 64 * layout;

    const uint16_t store_iter = (block_count_bs * block_size_car) / total_thr;
    const uint16_t store_interval = total_thr / block_size_car;

    const coord<2>& shape;

    bool pred;
    uint64_t off_bs;

    MMLTArr<Complex, elem>* rsmem;
    Complex* smem;
    Complex* gmem;

    MMLT_DEVICE
    OutIter(const coord<2>& shape_, Complex* gmem_, Complex* smem_,
            const uint64_t& car_start, const uint64_t& skip,
            const ThreadID<layout>& tid)
        : shape(shape_) {
      rsmem = reinterpret_cast<MMLTArr<Complex, elem>*>(smem_) +
              (4 * tid.warp_idx + tid.lane_idx) * block_count_car +
              (8 * tid.warp_idy + tid.lane_idy);
      uint64_t bs_id =
          block_size_bs * blockIdx.x + (threadIdx.x / block_size_car) * elem;
      uint64_t carrier_offset =
          blockIdx.y * block_size_car + threadIdx.x % block_size_car;
      uint64_t carrier_id = blockIdx.z * skip + carrier_offset;

      int out_carrier_id =
          (carrier_id + car_start + (shape.col / 2)) % shape.col;

      gmem = gmem_ + bs_id * shape.col + out_carrier_id;
      smem = smem_ + threadIdx.x;

      pred = carrier_offset < skip;
      off_bs = bs_id;
    }

    MMLT_DEVICE
    void store_next_reg(MMLTArr<Complex, elem * elem>& accum_reg,
                        const int& i_iter) {
      rsmem[0] = reinterpret_cast<MMLTArr<Complex, elem>*>(&accum_reg)[i_iter];
    }

    MMLT_DEVICE
    void store_next_smem() {
      MMLT_UNROLL
      for (int i = 0; i < store_iter; i++) {
        if (pred && ((off_bs + i * store_interval * elem) < shape.row)) {
          gmem[i * store_interval * elem * shape.col] =
              smem[i * store_interval * block_size_car];
        }
      }
      gmem += shape.col;
      off_bs += 1;
    }
  };

  template <const uint8_t layout, const uint8_t elem>
  MMLT_DEVICE void mmlt(const Params params) {
    // One layer: (4 + 4) * 8 * elem matrix
    // layout: how many rows of layers (max 4)
    // 4: elements along k axis
    __shared__ PrecSharedMem<layout, elem> smem;

    ThreadID<layout> tid;

    MMLTArr<Complex, elem> prec_reg;
    MMLTArr<Complex, elem> ueVec_reg;
    MMLTArr<Complex, elem * elem> accum_reg;

    InIter<layout, elem> in_iter(
        {params.precode_shape.row, params.ueVec_stride},
        {params.problem_size.row, params.problem_size.col, params.precode_skip},
        {params.batched_precode + blockIdx.z * params.precode_stride,
         params.batched_ueVec + blockIdx.z * params.precode_skip},
        {smem.prec.storage, smem.ueVec.storage}, tid);

    accum_reg.fill(Complex());

    int k_iterations = (params.problem_size.col + kK - 1) / kK;

    MMLT_LOOP
    for (; k_iterations > 0; k_iterations--) {
      in_iter.load_next_smem();

      __syncthreads();

      MMLT_UNROLL
      for (int k = 0; k < kK; k++) {
        in_iter.load_next_reg(prec_reg, ueVec_reg, k);
        mmlt_op(accum_reg, prec_reg, ueVec_reg);
      }
      in_iter.reset_smem_offset();

      __syncthreads();
    }

    OutIter<layout, elem> out_iter(
        {params.problem_size.row, params.bsVec_stride}, params.batched_bsVec,
        smem.out.storage, params.bsVec_start, params.precode_skip, tid);
    MMLT_UNROLL
    for (int i = 0; i < elem; i++) {
      out_iter.store_next_reg(accum_reg, i);
      __syncthreads();

      out_iter.store_next_smem();
      __syncthreads();
    }
  }

  /**
   * @brief Precode + pre-fft transformation with modified (singular) GEMV
   *
   * @param params parameters of the precoding kernel
   */
  /*** old naive GEMV implementation ***/
  /* MMLT_DEVICE
  void operator()(const Params& params) {
    for (uint32_t subcarrier_id = blockIdx.z;
         subcarrier_id < params.problem_size.dep; subcarrier_id += gridDim.z) {
      uint32_t bs_id = blockIdx.x * kThreadCount + threadIdx.x;

      const Complex* precode_ptr = params.batched_precode + bs_id;
      const Complex* ueVec_ptr = params.batched_ueVec;

      precode_ptr +=
          (subcarrier_id / params.precode_skip) * params.precode_stride;
      ueVec_ptr += subcarrier_id * params.ueVec_stride;

      Complex accum = Complex();

      MMLT_LOOP
      for (uint32_t ue_id = 0; ue_id < params.problem_size.col; ++ue_id) {
        Complex a = Complex();
        if (bs_id < params.problem_size.row) {
          a = *precode_ptr;
        }
        precode_ptr += params.precode_shape.row;

        Complex b = *ueVec_ptr;
        ueVec_ptr += 1;

        accum += a * b;
      }

      int out_subcarrier_id =
          (subcarrier_id + params.bsVec_start + (params.bsVec_stride / 2)) %
          params.bsVec_stride;
      Complex* bsVec_ptr = params.batched_bsVec + bs_id * params.bsVec_stride +
                           out_subcarrier_id;
      if (bs_id < params.problem_size.row) {
        *bsVec_ptr = accum;
      }
    }
  } */
};

/**
 * @brief Launch the precoding kernel wrapper
 *
 * @param params parameters of the precoding kernel
 */
template <const uint8_t layout, const uint8_t elem>
__global__ __launch_bounds__(256) void precode_kernel(
    const PrecodeKernel::Params params) {
  PrecodeKernel().mmlt<layout, elem>(params);
}

};  // namespace mega