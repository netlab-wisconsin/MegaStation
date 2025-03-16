/**
 * @file pin_threads.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Pin threads to cores
 * @version 0.1
 * @date 2024-01-29
 *
 * @copyright Copyright (c) 2024
 *
 */

#pragma once

#include <cstdint>
#include <mutex>
#include <unordered_set>
#include <vector>

namespace mega {

class PinThreads {
 private:
  static std::vector<std::vector<uint32_t>> cpu_layout;
  static std::mutex mtx;
  static std::unordered_set<uint32_t> assigned_cores;

 public:
  /**
   * @brief Get the cpu layout
   *
   * @param exclude_cores cores to exclude
   */
  static void init(const std::unordered_set<uint32_t>& exclude_cores = {});
  /**
   * @brief Pin the current thread to a core
   *
   * @param base_thread_id base thread id for the same thread workers type
   * @param thread_offset requested thread offset for the current thread
   */
  static void pin_thread(uint32_t node_id, uint32_t thread_id);
  /**
   * @brief Release the current thread from the core
   *
   * @param base_thread_id base thread id for the same thread workers type
   * @param thread_offset requested thread offset for the current thread
   */
  static void release_thread(uint32_t node_id, uint32_t thread_id);
};

};  // namespace mega