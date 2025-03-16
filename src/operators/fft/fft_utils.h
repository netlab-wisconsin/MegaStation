/**
 * @file fft_utils.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief fft struct used in callbacks
 * @version 0.1
 * @date 2023-12-10
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cstdint>
#include <limits>

#include "mega_complex.h"

template <typename QuanT>
static constexpr float kQuantFloat() {
  return static_cast<float>(std::numeric_limits<QuanT>::max()) + 1.0f;
}

struct baseInfo {
  uint32_t ofdmStart;
  uint32_t ofdmNum;
  uint32_t ofdmCAnum;
  uint32_t bsAnt;
};

struct pilotInfo : baseInfo {
  uint32_t ueAnt;
  uint32_t scGroup;
  uint32_t ueStart;
  mega::Complex *pilotSign;
};
