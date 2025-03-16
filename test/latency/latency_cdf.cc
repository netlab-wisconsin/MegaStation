/**
 * @file integral_test.cc
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Test Whole Baseband Processing
 * @version 0.1
 * @date 2024-04-02
 *
 * @copyright Copyright (c) 2024
 *
 */
#include <gtest/gtest.h>
#include <spdlog/sinks/basic_file_sink.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <fstream>
#include <vector>

#include "baseband/buffers.h"
#include "baseband/process.h"
#include "config/config.h"
#include "utils/timer.h"

using namespace mega;

constexpr uint32_t kIterations = 1000;

class BP : public BandProcess {
 public:
  static void frame() { BandProcess::frame(0); }
  static void symbol(uint32_t symbol_id) { BandProcess::symbol(0, symbol_id); }
  static void symbol2(uint32_t symbol_id, uint32_t frame_id) {
    BandProcess::symbol(frame_id, symbol_id);
  }

  static void task(uint32_t symbol_id, uint32_t task_id) {
    BandProcess::task(0, symbol_id, task_id);
  }

  static void process(uint32_t symbol_id) {
    BandProcess::process(0, symbol_id);
  }
  static void process2(uint32_t symbol_id, uint32_t frame_id) {
    BandProcess::process(frame_id, symbol_id);
  }

  static void copy(uint32_t frame_sid, uint32_t frame_did, uint32_t symbol_id,
                   uint32_t task_id) {
    BandProcess::copy(frame_sid, frame_did, symbol_id, task_id);
  }
};

std::pair<double, double> statics(std::vector<double> &data) {
  double sum = 0;
  for (auto &d : data) {
    sum += d;
  }
  double mean = sum / data.size();

  double sq_sum = 0;
  for (auto &d : data) {
    sq_sum += (d - mean) * (d - mean);
  }
  double stdev = std::sqrt(sq_sum / data.size());

  return {mean, stdev};
}

int ants = 64;
int users = 16;
int ofdm = 1200;
int sg = 16;

std::string dir = "../../../test/latency/";

TEST(LatencyBreakdown, Uplink) {
  int num_pilots = (users + sg - 1) / sg;
  std::string frame_str =
      std::string(num_pilots, 'P') + std::string(14 - num_pilots, 'U');

  gconfig =
      Config(ants, users, ofdm, 2048, sg, frame_str, 6, 1. / 3., 1, 1, 16);

  BandBuffers::init();
  BandProcess::init();

  auto &h_recv_ref = BandBuffers::HostBufferRecv[0];

  std::string file_name = std::string(dir) + "tx_" + std::to_string(ants) +
                          "x" + std::to_string(users) + ".data";

  std::ifstream input(file_name, std::ios::binary);
  input.seekg(0, std::ios::end);
  std::streampos fileSize = input.tellg();
  input.seekg(0, std::ios::beg);

  assert(fileSize == h_recv_ref.szBytes(h_recv_ref.nDim()));

  input.read(h_recv_ref.ptr<char>(), fileSize);
  input.close();

  std::string log_file = std::string(dir) + "result" + std::to_string(ants) +
                         "x" + std::to_string(users) + "cdf.csv";
  auto my_logger = spdlog::basic_logger_mt("basic_logger", log_file, true);
  my_logger->set_pattern("%v");
  my_logger->flush_on(spdlog::level::info);
  my_logger->info("uplink,,");
  my_logger->info("symbol_id,task_id,latency");

  // cold start
  cudaSetDevice(0);
  for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols; symbol_id++) {
    BP::process(symbol_id);
  }

  auto &local_streams = BandBuffers::Streams[0];

  uint32_t start_task = num_pilots - 1;
  uint32_t end_task = num_pilots - 1 + 2;

  for (uint32_t symbol_id = start_task; symbol_id < end_task; symbol_id++) {
    uint32_t total_tasks = symbol_id == start_task ? 3 : 5;
    for (uint32_t task_id = 0; task_id < total_tasks; task_id++) {
      for (uint32_t i = 0; i < kIterations; i++) {
        TimerGPU gtimer;
        gtimer.start(local_streams[symbol_id]);
        BP::task(symbol_id, task_id);
        gtimer.stop(local_streams[symbol_id]);
        my_logger->info("{},{},{}", symbol_id, task_id,
                        gtimer.get_duration_ms());
      }
    }
  }

  BandBuffers::destroy();
}

TEST(LatencyBreakdown, Downlink) {
  int num_pilots = (users + sg - 1) / sg;
  std::string frame_str =
      std::string(num_pilots, 'P') + std::string(14 - num_pilots, 'D');

  gconfig =
      Config(ants, users, ofdm, 2048, sg, frame_str, 6, 1. / 3., 1, 1, 16);

  BandBuffers::init();
  BandProcess::init();

  auto &dl_data = BandBuffers::HostBufferMacRecv[0];

  std::string file_name_mac = std::string(dir) + "mac_" + std::to_string(ants) +
                              "x" + std::to_string(users) + ".data";

  std::ifstream input(file_name_mac, std::ios::binary);
  input.seekg(0, std::ios::end);
  std::streampos fileSize = input.tellg();
  input.seekg(0, std::ios::beg);

  assert(fileSize == dl_data.szBytes(dl_data.nDim()));

  input.read(dl_data.ptr<char>(), fileSize);
  input.close();

  auto &h_recv_ref = BandBuffers::HostBufferRecv[0];

  std::string file_name_tx = std::string(dir) + "tx_" + std::to_string(ants) +
                             "x" + std::to_string(users) + ".data";

  input = std::ifstream(file_name_tx, std::ios::binary);
  input.seekg(0, std::ios::end);
  fileSize = input.tellg();
  input.seekg(0, std::ios::beg);

  assert(fileSize / 14 == h_recv_ref.szBytes(h_recv_ref.nDim()) / num_pilots);

  input.read(h_recv_ref.ptr<char>(), fileSize / 14);
  input.close();

  std::string log_file = std::string(dir) + "result" + std::to_string(ants) +
                         "x" + std::to_string(users) + "cdf.csv";
  auto my_logger = spdlog::basic_logger_mt("basic_logger2", log_file, false);
  my_logger->set_pattern("%v");
  my_logger->flush_on(spdlog::level::info);
  my_logger->info("downlink,,");
  my_logger->info("symbol_id,task_id,latency");

  cudaSetDevice(0);
  for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols; symbol_id++) {
    BP::process(symbol_id);
  }

  auto &local_streams = BandBuffers::Streams[0];

  uint32_t start_task = num_pilots - 1;
  uint32_t end_task = num_pilots - 1 + 2;

  for (uint32_t symbol_id = start_task; symbol_id < end_task; symbol_id++) {
    uint32_t total_tasks = symbol_id == start_task ? 3 : 6;
    for (uint32_t task_id = 0; task_id < total_tasks; task_id++) {
      std::vector<double> times;
      for (uint32_t i = 0; i < kIterations; i++) {
        TimerGPU gtimer;
        gtimer.start(local_streams[symbol_id]);
        BP::task(symbol_id, task_id);
        gtimer.stop(local_streams[symbol_id]);
        my_logger->info("{},{},{}", symbol_id, task_id,
                        gtimer.get_duration_ms());
      }
    }
  }

  BandBuffers::destroy();
}

TEST(LatencyBreakdown, Communication) {
  int num_pilots = (users + sg - 1) / sg;
  std::string frame_str =
      std::string(num_pilots, 'P') + std::string(14 - num_pilots, 'U');

  gconfig =
      Config(ants, users, ofdm, 2048, sg, frame_str, 6, 1. / 3., 1, 1, 16);
  gconfig.open_peer_access();

  BandBuffers::init();
  BandProcess::init();

  auto &h_recv_ref = BandBuffers::HostBufferRecv[0];

  std::string file_name = std::string(dir) + "tx_" + std::to_string(ants) +
                          "x" + std::to_string(users) + ".data";

  std::ifstream input(file_name, std::ios::binary);
  input.seekg(0, std::ios::end);
  std::streampos fileSize = input.tellg();
  input.seekg(0, std::ios::beg);

  assert(fileSize == h_recv_ref.szBytes(h_recv_ref.nDim()));

  input.read(h_recv_ref.ptr<char>(), fileSize);
  input.close();

  std::string log_file = "commu_cdf.csv";
  auto my_logger = spdlog::basic_logger_mt("basic_logger3", log_file, true);
  my_logger->set_pattern("%v");
  my_logger->flush_on(spdlog::level::info);

  // cold start
  for (uint32_t frame_id = 0; frame_id < 30; frame_id++) {
    memcpy(BandBuffers::HostBufferRecv[frame_id].ptr<char>(),
           h_recv_ref.ptr<char>(), h_recv_ref.szBytes());
  }

  cudaSetDevice(0);
  for (uint32_t frame_id = 0; frame_id < 30; frame_id += 2) {
    for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols; symbol_id++) {
      BP::process2(symbol_id, frame_id);
    }
  }

  cudaSetDevice(1);
  for (uint32_t frame_id = 1; frame_id < 30; frame_id += 2) {
    for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols; symbol_id++) {
      BP::process2(symbol_id, frame_id);
    }
  }

  for (uint32_t i = 0; i < kIterations; i++) {
    TimerCPU ctimer;
    cudaSetDevice(0);
    int prev_dev = 0;
    for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols; symbol_id++) {
      int dev = symbol_id < 7 ? 0 : 1;
      if (dev != prev_dev) {
        cudaSetDevice(dev);
        ctimer.start();
        BP::copy(prev_dev, dev, 0, 1);
        ctimer.stop();
        my_logger->info("{}", ctimer.get_duration_ms());
        prev_dev = dev;
      }
      BP::symbol2(symbol_id, dev);
    }
  }

  BandBuffers::destroy();
}

int main(int argc, char **argv) {
  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-ants") == 0) {
      ants = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-users") == 0) {
      users = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-ofdm") == 0) {
      ofdm = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-sg") == 0) {
      sg = atoi(argv[++i]);
    } else if (strcmp(argv[i], "-dir") == 0) {
      dir = std::string(argv[++i]);
    }
  }

  ::testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
