/**
 * @file consts.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Common constants
 * @version 0.1
 * @date 2023-12-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cstdint>

namespace mega {

constexpr uint32_t kFrameWindow = 30;   //!< circular frame window size
constexpr uint32_t kMaxSymbolNum = 20;  //!< symbol number
constexpr uint32_t kCbPerSymbol = 1;    //!< code blocks per symbol

constexpr uint32_t kLogThreads = 1;           //!< log threads
constexpr uint32_t kLogQueueSize = 33554432;  //!< log queue size

constexpr uint32_t kRecvThreads = 2;  //!< count of receive threads
constexpr uint8_t kMaxDeviceNum = 2;  //!< max device number

}  // namespace mega