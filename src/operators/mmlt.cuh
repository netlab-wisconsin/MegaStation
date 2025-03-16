/**
 * @file mmlt.cuh
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Export Macros from MMLT library
 * @version 0.1
 * @date 2023-12-01
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cstdint>

#define MMLT_DEVICE __forceinline__ __device__
#define MMLT_UNROLL _Pragma("unroll")
#define MMLT_LOOP _Pragma("unroll 1")

namespace mega {

/**
 * @brief Coordinate of a multi-dimensional array
 *
 * @tparam N dimension
 */
template <int N>
struct coord;

/**
 * @brief Coordinate of 2D array
 *
 */
template <>
struct coord<2> {
  uint64_t row, col;  //!< row, column
};

/**
 * @brief Coordinate of problem size
 *
 */
template <>
struct coord<3> {
  uint64_t row, col, dep;  //!< row, column, depth(batch_size)
};

static constexpr uint8_t kThreadCount = 32;  //!< 32 threads per warp

/**
 * @brief Get the block shape object
 *
 * @return dim3 block shape
 */
static inline dim3 get_block_shape() { return dim3(kThreadCount, 1, 1); }

/**
 * @brief Get the grid shape object
 *
 * @param psize problem size
 * @param block block shape
 * @return dim3 grid shape
 */
static inline dim3 get_grid_shape(const coord<3> &psize, const dim3 &block) {
  return dim3((psize.row + block.x - 1) / block.x, 1, psize.dep % 65536);
}

template <typename T, int N>
struct MMLTArr {
  T storage[N];

  MMLT_DEVICE
  void fill(T val) {
    MMLT_UNROLL
    for (int i = 0; i < N; i++) {
      storage[i] = val;
    }
  }
};

template <typename T, int N>
struct MMLTBuf {
  alignas(16) T storage[N];
};

template <typename T, int NA, int NB>
MMLT_DEVICE void mmlt_op(MMLTArr<T, NA * NB> &outC, const MMLTArr<T, NA> &inA,
                         const MMLTArr<T, NB> &inB) {
  MMLT_UNROLL
  for (int i = 0; i < NA; i++) {
    MMLT_UNROLL
    for (int j = 0; j < NB; j++) {
      outC.storage[i * NB + j] += inA.storage[i] * inB.storage[j];
    }
  }
}

}  // namespace mega
