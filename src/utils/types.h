/**
 * @file types.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Self-defined types
 * @version 0.1
 * @date 2023-12-16
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cuda_runtime.h>

#include <array>
#include <atomic>
#include <cstdint>

#include "consts.h"
#include "matrix/cuphy_tensor.h"
#include "matrix/matrix.h"

namespace mega {

using QuanT = char;  //!< quantization type

template <typename T>
using ArrFrame = std::array<T, kFrameWindow>;  //!< array type for each frame

template <typename T>
using ArrFrameSymbol = ArrFrame<std::array<T, kMaxSymbolNum>>;  //!< array type
                                                                //!< for each
                                                                //!< frame, each
                                                                //!< symbol

using ArrCounter =
    ArrFrameSymbol<std::atomic<uint32_t>>;  //!< counter type for each frame,
                                            //!< each symbol
using ArrStream = ArrFrameSymbol<cudaStream_t>;  //!< CUDA stream array type for
                                                 //!< each frame, each symbol

using Buffer = ArrFrame<Matrix>;          //!< buffer for each frame
using BufferPhy = ArrFrame<CuphyTensor>;  //!< buffer for each frame

}  // namespace mega