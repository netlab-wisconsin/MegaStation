/**
 * @file counter.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Atomic counter for each frame, each symbol
 * @version 0.1
 * @date 2023-12-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <spdlog/spdlog.h>

#include <atomic>

#include "consts.h"
#include "types.h"

namespace mega {

/**
 * @brief Atomic counter for each frame, each symbol
 *
 */
class Counter {
 private:
  ArrCounter counter;  //!< counter for each frame, each symbol
  uint64_t max_count;  //!< max count

 public:
  /**
   * @brief Construct a new Counter object
   *
   */
  Counter(uint64_t max_count_ = 0) : max_count(max_count_) {
    for (uint8_t frame_id = 0; frame_id < kFrameWindow; frame_id++) {
      for (uint8_t symbol_id = 0; symbol_id < kMaxSymbolNum; symbol_id++) {
        counter[frame_id][symbol_id].store(0, std::memory_order_seq_cst);
      }
    }
  }

  /**
   * @brief Change the max count
   *
   */
  void set_max_count(uint64_t max_count_) { max_count = max_count_; }

  /**
   * @brief Increment the counter for a specific frame and symbol
   *
   * @param frame_id frame id
   * @param symbol_id symbol id
   * @return uint32_t counter value
   */
  bool inc(uint32_t frame_id, uint32_t symbol_id) {
    auto &lcounter = counter[frame_id % kFrameWindow][symbol_id];

    uint64_t ccount = lcounter.fetch_add(1, std::memory_order_acq_rel) + 1;

    if (ccount == max_count) {
      return true;
    } else if (ccount < max_count && ccount > 0) {
      return false;
    } else {
      spdlog::error(
          "Counter Overflow: frame_id: {}, frame_cid: {}, symbol_id: {}, "
          "current_count: {}, max_count: {}",
          frame_id, frame_id % kFrameWindow, symbol_id, ccount, max_count);
      throw std::runtime_error(
          "Counter Overflow: Maybe because you did't initialize the gconfig "
          "with the correct antenna number. Please call Config::init(...) "
          "before creating the UdpServer object. Or maybe the system is too "
          "slow to process the packets.");
    }
  }

  /**
   * @brief Prepare for processing a new symbol
   *
   * @param frame_id frame id
   * @param symbol_id symbol id
   * @return uint32_t counter value
   */
  void prepare(uint32_t frame_id, uint32_t symbol_id) {
    counter[frame_id % kFrameWindow][symbol_id].store(
        -1, std::memory_order_release);
  }

  /**
   * @brief Check if the symbol is ready to be processed
   *
   * @param frame_id frame id
   * @param symbol_id symbol id
   * @return bool ready or not
   */
  bool ready(uint32_t frame_id, uint32_t symbol_id) {
    return counter[frame_id % kFrameWindow][symbol_id].load(
               std::memory_order_acquire) == -1;
  }
  /**
   * @brief Done processing and Reset the counter
   *
   * @param frame_id frame id
   * @param symbol_id symbol id
   */
  void done(uint32_t frame_id, uint32_t symbol_id) {
    counter[frame_id % kFrameWindow][symbol_id].store(
        0, std::memory_order_release);
  }
};

}  // namespace mega