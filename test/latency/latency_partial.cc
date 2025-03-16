/**
 * @file partial.cc
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Partial parallelism motivation analysis
 * @version 0.1
 * @date 2024-07-05
 *
 * @copyright Copyright (c) 2024
 *
 */
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
#include <thread>
#include <vector>

#include "arti_kernel.h"
#include "baseband/buffers.h"
#include "baseband/process.h"
#include "config/config.h"
#include "utils/timer.h"

using namespace mega;

constexpr uint32_t kIterations = 100;

class BP : public BandProcess {
 public:
  static void symbol(uint32_t symbol_id, uint32_t frame_id) {
    BandProcess::symbol(frame_id, symbol_id);
  }

  static void task(uint32_t task_id, uint32_t symbol_id, uint32_t frame_id) {
    BandProcess::task(frame_id, symbol_id, task_id);
  }

  static void frame(uint32_t frame_id) { BandProcess::frame(frame_id); }

  static void process(uint32_t symbol_id, uint32_t frame_id) {
    BandProcess::process(frame_id, symbol_id);
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

TEST(LatencyPartial, Uplink) {
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

  // cold start
  cudaSetDevice(0);
  for (uint32_t frame_id = 0; frame_id < 20; frame_id += 2) {
    memcpy(BandBuffers::HostBufferRecv[frame_id].ptr<char>(),
           h_recv_ref.ptr<char>(), h_recv_ref.szBytes());
  }
  for (uint32_t frame_id = 0; frame_id < 20; frame_id += 2) {
    for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols; symbol_id++) {
      BP::process(symbol_id, frame_id);
    }
  }

  std::string log_file = std::string(dir) + "partial" + std::to_string(ants) +
                         "x" + std::to_string(users) + ".csv";
  auto my_logger = spdlog::basic_logger_mt("basic_logger", log_file, true);
  my_logger->set_pattern("%v");
  my_logger->flush_on(spdlog::level::info);
  my_logger->info("uplink,,,");
  my_logger->info("occupied vSM,type,mean,stdev");

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int *flag;
  cudaMallocManaged(&flag, sizeof(int));

  char *data0, *result0;

  cudaMalloc(&data0, 49152 * sizeof(char));
  cudaMalloc(&result0, 49152 * sizeof(char));

  char *data_cpu = (char *)malloc(49152 * sizeof(char));
  for (int i = 0; i < 49152; i++) {
    data_cpu[i] = rand() % 256;
  }

  cudaMemcpy(data0, data_cpu, 49152 * sizeof(char), cudaMemcpyHostToDevice);

  for (int vsm = 0; vsm < 80; vsm++) {
    *flag = 0;
    cudaDeviceSynchronize();

    *flag = 1;
    arti_launch(flag, data0, result0, vsm, stream);

    {
      std::vector<double> throughputs;
      for (uint32_t i = 0; i < kIterations; i++) {
        // std::vector<std::thread> threads;
        TimerCPU ctimer;
        ctimer.start();
        for (uint32_t j = 0; j < 1; j++) {
          // threads.push_back(std::thread([j]() { BP::frame(j * 2); }));
          BP::frame(j * 2);
        }
        // for (auto &t : threads) {
        //   t.join();
        // }
        ctimer.stop();
        throughputs.push_back(ctimer.get_duration_ms());
      }
      auto cal_stat = statics(throughputs);
      printf("vSM: %d, frame, mean: %f, stdev: %f\n", vsm, cal_stat.first,
             cal_stat.second);
      my_logger->info("{},frame,{},{}", vsm, cal_stat.first, cal_stat.second);
    }

    {
      std::vector<double> throughputs;
      for (uint32_t i = 0; i < kIterations; i++) {
        TimerCPU ctimer;
        ctimer.start();
        for (uint32_t j = 0; j < 1; j++) {
          for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols;
               symbol_id++) {
            BP::symbol(symbol_id, j * 2);
          }
        }
        ctimer.stop();
        throughputs.push_back(ctimer.get_duration_ms());
      }
      auto cal_stat = statics(throughputs);
      printf("vSM: %d, symbol, mean: %f, stdev: %f\n", vsm, cal_stat.first,
             cal_stat.second);
      my_logger->info("{},symbol,{},{}", vsm, cal_stat.first, cal_stat.second);
    }

    {
      std::vector<double> throughputs;
      for (uint32_t i = 0; i < kIterations; i++) {
        TimerCPU ctimer;
        ctimer.start();
        for (uint32_t j = 0; j < 1; j++) {
          for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols;
               symbol_id++) {
            uint32_t total_tasks;
            switch (gconfig.frame.gidx_sym[symbol_id]) {
              case Config::Pilot:
                total_tasks = 3;
                break;
              case Config::Uplink:
                total_tasks = 5;
                break;
              case Config::Downlink:
                total_tasks = 6;
                break;
              default:
                total_tasks = 0;
            }
            for (uint32_t task_id = 0; task_id < total_tasks; task_id++) {
              BP::task(task_id, symbol_id, j * 2);
            }
          }
        }
        ctimer.stop();
        throughputs.push_back(ctimer.get_duration_ms());
      }
      auto cal_stat = statics(throughputs);
      printf("vSM: %d, task, mean: %f, stdev: %f\n", vsm, cal_stat.first,
             cal_stat.second);
      my_logger->info("{},task,{},{}", vsm, cal_stat.first, cal_stat.second);
    }
  }

  *flag = 0;
  cudaDeviceSynchronize();

  cudaStreamDestroy(stream);
  cudaFree(flag);
  cudaFree(data0);
  cudaFree(result0);
  free(data_cpu);
  BandBuffers::destroy();
}

TEST(LatencyPartial, Downlink) {
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

  cudaSetDevice(0);
  for (uint32_t frame_id = 0; frame_id < 20; frame_id += 2) {
    memcpy(BandBuffers::HostBufferMacRecv[frame_id].ptr<char>(),
           dl_data.ptr<char>(), dl_data.szBytes());
    memcpy(BandBuffers::HostBufferRecv[frame_id].ptr<char>(),
           h_recv_ref.ptr<char>(), h_recv_ref.szBytes());
  }
  for (uint32_t frame_id = 0; frame_id < 20; frame_id += 2) {
    for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols; symbol_id++) {
      BP::process(symbol_id, frame_id);
    }
  }

  std::string log_file = std::string(dir) + "partial" + std::to_string(ants) +
                         "x" + std::to_string(users) + ".csv";
  auto my_logger = spdlog::basic_logger_mt("basic_logger2", log_file, false);
  my_logger->set_pattern("%v");
  my_logger->flush_on(spdlog::level::info);
  my_logger->info("downlink,,,");
  my_logger->info("occupied vSM,type,mean,stdev");

  cudaStream_t stream;
  cudaStreamCreate(&stream);

  int *flag;
  cudaMallocManaged(&flag, sizeof(int));

  char *data0, *result0;

  cudaMalloc(&data0, 49152 * sizeof(char));
  cudaMalloc(&result0, 49152 * sizeof(char));

  char *data_cpu = (char *)malloc(49152 * sizeof(char));
  for (int i = 0; i < 49152; i++) {
    data_cpu[i] = rand() % 256;
  }

  cudaMemcpy(data0, data_cpu, 49152 * sizeof(char), cudaMemcpyHostToDevice);

  for (int vsm = 0; vsm < 80; vsm++) {
    *flag = 0;
    cudaDeviceSynchronize();

    *flag = 1;
    arti_launch(flag, data0, result0, vsm, stream);

    {
      std::vector<double> throughputs;
      for (uint32_t i = 0; i < kIterations; i++) {
        // std::vector<std::thread> threads;
        TimerCPU ctimer;
        ctimer.start();
        for (uint32_t j = 0; j < 1; j++) {
          // threads.push_back(std::thread([j]() { BP::frame(j * 2); }));
          BP::frame(j * 2);
        }
        // for (auto &t : threads) {
        //   t.join();
        // }
        ctimer.stop();
        throughputs.push_back(ctimer.get_duration_ms());
      }
      auto cal_stat = statics(throughputs);
      printf("vSM: %d, frame, mean: %f, stdev: %f\n", vsm, cal_stat.first,
             cal_stat.second);
      my_logger->info("{},frame,{},{}", vsm, cal_stat.first, cal_stat.second);
    }

    {
      std::vector<double> throughputs;
      for (uint32_t i = 0; i < kIterations; i++) {
        TimerCPU ctimer;
        ctimer.start();
        for (uint32_t j = 0; j < 1; j++) {
          for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols;
               symbol_id++) {
            BP::symbol(symbol_id, j * 2);
          }
        }
        ctimer.stop();
        throughputs.push_back(ctimer.get_duration_ms());
      }
      auto cal_stat = statics(throughputs);
      printf("vSM: %d, symbol, mean: %f, stdev: %f\n", vsm, cal_stat.first,
             cal_stat.second);
      my_logger->info("{},symbol,{},{}", vsm, cal_stat.first, cal_stat.second);
    }

    {
      std::vector<double> throughputs;
      for (uint32_t i = 0; i < kIterations; i++) {
        TimerCPU ctimer;
        ctimer.start();
        for (uint32_t j = 0; j < 1; j++) {
          for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols;
               symbol_id++) {
            uint32_t total_tasks;
            switch (gconfig.frame.gidx_sym[symbol_id]) {
              case Config::Pilot:
                total_tasks = 3;
                break;
              case Config::Uplink:
                total_tasks = 5;
                break;
              case Config::Downlink:
                total_tasks = 6;
                break;
              default:
                total_tasks = 0;
            }
            for (uint32_t task_id = 0; task_id < total_tasks; task_id++) {
              BP::task(task_id, symbol_id, j * 2);
            }
          }
        }
        ctimer.stop();
        throughputs.push_back(ctimer.get_duration_ms());
      }
      auto cal_stat = statics(throughputs);
      printf("vSM: %d, task, mean: %f, stdev: %f\n", vsm, cal_stat.first,
             cal_stat.second);
      my_logger->info("{},task,{},{}", vsm, cal_stat.first, cal_stat.second);
    }
  }

  *flag = 0;
  cudaDeviceSynchronize();

  cudaStreamDestroy(stream);
  cudaFree(flag);
  cudaFree(data0);
  cudaFree(result0);
  free(data_cpu);
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
