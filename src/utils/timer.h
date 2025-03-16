/**
 * @file timer.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Timer header file for timer (GPU and CPU)
 * @version 0.1
 * @date 2023-12-15
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cuda_runtime.h>
#include <spdlog/spdlog.h>
#include <time.h>

#include <cassert>
#include <chrono>

namespace mega {

/**
 * @brief Base Timer class
 *
 */
class Timer {
 public:
  /**
   * @brief Start the timer
   *
   */
  virtual void start(void *args = nullptr) = 0;
  /**
   * @brief Stop the timer
   *
   */
  virtual void stop(void *args = nullptr) = 0;
  /**
   * @brief Get the duration in ms
   *
   * @return double duration in ms
   */
  virtual double get_duration_ms() = 0;
  /**
   * @brief Destroy the Timer object
   *
   */
  virtual ~Timer() = default;
};

/**
 * @brief Timer class for timing (CPU)
 *
 */
class TimerCPU : public Timer {
 private:
  std::chrono::time_point<std::chrono::high_resolution_clock>
      start_time;  //!< start time
  std::chrono::time_point<std::chrono::high_resolution_clock>
      end_time;  //!< end time

 public:
  /**
   * @brief Construct a new TimerCPU object
   *
   */
  TimerCPU() {}
  /**
   * @brief Start the timer
   *
   */
  void start(void * = nullptr) override {
    start_time = std::chrono::high_resolution_clock::now();
  }
  /**
   * @brief Stop the timer
   *
   */
  void stop(void * = nullptr) override {
    end_time = std::chrono::high_resolution_clock::now();
  }
  /**
   * @brief Get the duration in ms
   *
   * @return double duration in ms
   */
  double get_duration_ms() override {
    return std::chrono::duration_cast<std::chrono::microseconds>(end_time -
                                                                 start_time)
               .count() /
           1000.0;
  }
  double checkpoint() {
    return std::chrono::duration_cast<std::chrono::microseconds>(
               std::chrono::high_resolution_clock::now() - start_time)
               .count() /
           1000.0;
  }
  /**
   * @brief Destroy the TimerCPU object
   *
   */
  ~TimerCPU() override = default;
};

/**
 * @brief  Timer class for timing (GPU)
 *
 */
class TimerGPU : public Timer {
 private:
  cudaEvent_t start_time;  //!< start time
  cudaEvent_t end_time;    //!< end time

 public:
  /**
   * @brief Construct a new TimerGPU object
   *
   */
  TimerGPU() {
    cudaEventCreateWithFlags(&start_time, cudaEventBlockingSync);
    cudaEventCreateWithFlags(&end_time, cudaEventBlockingSync);
  }
  /**
   * @brief Start the timer
   *
   */
  void start(void *stream = nullptr) override {
    cudaEventRecord(start_time, reinterpret_cast<cudaStream_t>(stream));
  }
  void start(cudaStream_t stream) { cudaEventRecord(start_time, stream); }
  /**
   * @brief Stop the timer
   *
   */
  void stop(void *stream = nullptr) override {
    cudaEventRecord(end_time, reinterpret_cast<cudaStream_t>(stream));
  }
  void stop(cudaStream_t stream) { cudaEventRecord(end_time, stream); }
  /**
   * @brief Get the duration in ms
   *
   * @return double duration in ms
   */
  double get_duration_ms() override {
    cudaEventSynchronize(end_time);
    float ms;
    cudaEventElapsedTime(&ms, start_time, end_time);
    return ms;
  }
  /**
   * @brief Destroy the TimerGPU object
   *
   */
  ~TimerGPU() override {
    cudaEventDestroy(start_time);
    cudaEventDestroy(end_time);
  }
};

class TimerTsc : public Timer {
 private:
  uint64_t start_time;
  uint64_t end_time;
  static inline double freq_ghz;

  static inline uint64_t __rdtsc() {
    uint64_t lo, hi;
    asm volatile("rdtsc" : "=a"(lo), "=d"(hi));
    return static_cast<uint64_t>((hi << 32) | lo);
  }

 public:
  static void init() {
    uint64_t start, end;
    timespec start_ts, end_ts;

    clock_gettime(CLOCK_REALTIME, &start_ts);
    start = __rdtsc();

    uint64_t sum = 5;
    for (uint64_t i = 0; i < 1000000; i++) {
      sum += i + (sum + i) * (i % sum);
    }
    assert(sum == 13580802877818827968ull);

    clock_gettime(CLOCK_REALTIME, &end_ts);
    end = __rdtsc();

    uint64_t clock_ns =
        static_cast<uint64_t>(end_ts.tv_sec - start_ts.tv_sec) * 1000000000ul +
        static_cast<uint64_t>(end_ts.tv_nsec - start_ts.tv_nsec);
    freq_ghz = static_cast<double>(end - start) / clock_ns;

    spdlog::info("TSC frequency: {} GHz", freq_ghz);
  }
  TimerTsc() {}
  void start(void * = nullptr) override { start_time = __rdtsc(); }
  void stop(void * = nullptr) override { end_time = __rdtsc(); }
  double get_duration_ms() override {
    return (end_time - start_time) / (freq_ghz * 1e6);
  }
  ~TimerTsc() override = default;
};

}  // namespace mega