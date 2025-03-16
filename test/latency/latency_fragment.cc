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

  static void copy(uint32_t frame_sid, uint32_t frame_did, uint32_t symbol_id,
                   uint32_t task_id) {
    BandProcess::copy(frame_sid, frame_did, symbol_id, task_id);
  }
  static void sync(uint32_t frame_id, uint32_t symbol_id) {
    BandProcess::sync(frame_id, symbol_id);
  }

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

TEST(LatencyBreakdown, Uplink) {
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

  std::string log_file = std::string(dir) + "fragment" + std::to_string(ants) +
                         "x" + std::to_string(users) + ".csv";
  auto my_logger = spdlog::basic_logger_mt("basic_logger", log_file, true);
  my_logger->set_pattern("%v");
  my_logger->flush_on(spdlog::level::info);
  my_logger->info("uplink,,,");
  my_logger->info("#/#,type,mean,stdev");

  // cold start
  for (uint32_t frame_id = 0; frame_id < 30; frame_id++) {
    memcpy(BandBuffers::HostBufferRecv[frame_id].ptr<char>(),
           h_recv_ref.ptr<char>(), h_recv_ref.szBytes());
  }

  char *data_cpu = (char *)malloc(49152 * sizeof(char));
  for (int i = 0; i < 49152; i++) {
    data_cpu[i] = rand() % 256;
  }

  cudaSetDevice(0);
  for (uint32_t frame_id = 0; frame_id < 30; frame_id += 2) {
    for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols; symbol_id++) {
      BP::process(symbol_id, frame_id);
    }
  }

  cudaStream_t stream0;
  cudaStreamCreate(&stream0);

  int *flag0;
  cudaMallocManaged(&flag0, sizeof(int));

  char *data0, *result0;

  cudaMalloc(&data0, 49152 * sizeof(char));
  cudaMalloc(&result0, 49152 * sizeof(char));

  cudaMemcpy(data0, data_cpu, 49152 * sizeof(char), cudaMemcpyHostToDevice);

  cudaSetDevice(1);
  for (uint32_t frame_id = 1; frame_id < 30; frame_id += 2) {
    for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols; symbol_id++) {
      BP::process(symbol_id, frame_id);
    }
  }

  cudaStream_t stream1;
  cudaStreamCreate(&stream1);

  int *flag1;
  cudaMallocManaged(&flag1, sizeof(int));

  char *data1, *result1;

  cudaMalloc(&data1, 49152 * sizeof(char));
  cudaMalloc(&result1, 49152 * sizeof(char));

  cudaMemcpy(data1, data_cpu, 49152 * sizeof(char), cudaMemcpyHostToDevice);

  for (int vsm = 14; vsm >= 7; vsm--) {
    cudaSetDevice(0);
    *flag0 = 0;
    cudaDeviceSynchronize();
    *flag0 = 1;
    arti_launch(flag0, data0, result0, 80 - vsm, stream0);

    cudaSetDevice(1);
    *flag1 = 0;
    cudaDeviceSynchronize();
    *flag1 = 1;
    arti_launch(flag1, data1, result1, 80 - (14 - vsm), stream1);

    {
      std::vector<double> throughputs;
      for (uint32_t i = 0; i < kIterations; i++) {
        TimerCPU ctimer;
        std::vector<std::thread> threads;
        int flag = 0;
        for (uint32_t j = 0; j < 14; j++) {
          threads.push_back(std::thread([&flag, j, &vsm]() {
            int dev = j < vsm ? 0 : 1;
            int frm_id = (dev == 0) ? (j * 2) : ((j - vsm) * 2 + 1);
            cudaSetDevice(dev);
            while (flag == 0);
            BP::frame(frm_id);
          }));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ctimer.start();
        flag = 1;
        for (auto &t : threads) {
          t.join();
        }
        ctimer.stop();
        throughputs.push_back(14.0 / ctimer.get_duration_ms());
      }
      auto cal_stat = statics(throughputs);
      printf("vSM: %d, frame, mean: %f, stdev: %f\n", vsm, cal_stat.first,
             cal_stat.second);
      my_logger->info("{}/{},frame,{},{}", 14 - vsm, vsm, cal_stat.first,
                      cal_stat.second);
    }

    {
      std::vector<double> throughputs;
      for (uint32_t i = 0; i < kIterations; i++) {
        TimerCPU ctimer;
        std::vector<std::thread> threads;
        int flag = 0;
        for (uint32_t j = 0; j < 14; j++) {
          threads.push_back(std::thread([&flag, j, &vsm]() {
            cudaSetDevice(0);
            int prev_dev = 0;
            while (flag == 0);
            for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols;
                 symbol_id++) {
              int dev = symbol_id < vsm ? 0 : 1;
              if (dev != prev_dev) {
                cudaSetDevice(dev);
                BP::copy(j * 2 + prev_dev, j * 2 + dev, 0, 1);
                prev_dev = dev;
              }
              BP::symbol(symbol_id, j * 2 + dev);
            }
          }));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        flag = 1;
        ctimer.start();
        for (auto &t : threads) {
          t.join();
        }
        ctimer.stop();
        throughputs.push_back(14.0 / ctimer.get_duration_ms());
      }
      auto cal_stat = statics(throughputs);
      printf("vSM: %d, symbol, mean: %f, stdev: %f\n", vsm, cal_stat.first,
             cal_stat.second);
      my_logger->info("{}/{},symbol,{},{}", 14 - vsm, vsm, cal_stat.first,
                      cal_stat.second);
    }

    {
      std::vector<double> throughputs;
      for (uint32_t i = 0; i < kIterations; i++) {
        TimerCPU ctimer;

        std::vector<std::thread> threads;
        int flag = 0;
        for (uint32_t j = 0; j < 14; j++) {
          threads.push_back(std::thread([&flag, j, &vsm]() {
            int main_dev = j < vsm ? 0 : 1;
            cudaSetDevice(main_dev);
            int prev_dev = main_dev;
            while (flag == 0);
            for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols;
                 symbol_id++) {
              if (prev_dev != main_dev) {
                cudaSetDevice(main_dev);
                prev_dev = main_dev;
              }
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
                int dev = (task_id > 2 && vsm != 14) ? 1 - main_dev : main_dev;
                if (dev != prev_dev) {
                  BP::sync(j * 2 + prev_dev, symbol_id);
                  cudaSetDevice(dev);
                  BP::copy(j * 2 + prev_dev, j * 2 + dev, symbol_id, 2);
                  prev_dev = dev;
                }
                BP::task(task_id, symbol_id, j * 2 + dev);
              }
            }
          }));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        flag = 1;
        ctimer.start();
        for (auto &t : threads) {
          t.join();
        }
        ctimer.stop();
        throughputs.push_back(14.0 / ctimer.get_duration_ms());
      }
      auto cal_stat = statics(throughputs);
      printf("vSM: %d, task, mean: %f, stdev: %f\n", vsm, cal_stat.first,
             cal_stat.second);
      my_logger->info("{}/{},task,{},{}", 14 - vsm, vsm, cal_stat.first,
                      cal_stat.second);
    }
  }

  cudaSetDevice(0);
  *flag0 = 0;
  cudaDeviceSynchronize();

  cudaSetDevice(1);
  *flag1 = 0;
  cudaDeviceSynchronize();

  cudaStreamDestroy(stream0);
  cudaStreamDestroy(stream1);
  cudaFree(flag0);
  cudaFree(flag1);
  cudaFree(data0);
  cudaFree(result0);
  cudaFree(data1);
  cudaFree(result1);
  free(data_cpu);
  BandBuffers::destroy();
}

TEST(LatencyBreakdown, Downlink) {
  int num_pilots = (users + sg - 1) / sg;
  std::string frame_str =
      std::string(num_pilots, 'P') + std::string(14 - num_pilots, 'D');

  gconfig =
      Config(ants, users, ofdm, 2048, sg, frame_str, 6, 1. / 3., 1, 1, 16);
  gconfig.open_peer_access();

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

  std::string log_file = std::string(dir) + "fragment" + std::to_string(ants) +
                         "x" + std::to_string(users) + ".csv";
  auto my_logger = spdlog::basic_logger_mt("basic_logger2", log_file, false);
  my_logger->set_pattern("%v");
  my_logger->flush_on(spdlog::level::info);
  my_logger->info("downlink,,,");
  my_logger->info("#/#,type,mean,stdev");

  // cold start
  for (uint32_t frame_id = 0; frame_id < 30; frame_id++) {
    memcpy(BandBuffers::HostBufferMacRecv[frame_id].ptr<char>(),
           dl_data.ptr<char>(), dl_data.szBytes());
    memcpy(BandBuffers::HostBufferRecv[frame_id].ptr<char>(),
           h_recv_ref.ptr<char>(), h_recv_ref.szBytes());
  }

  char *data_cpu = (char *)malloc(49152 * sizeof(char));
  for (int i = 0; i < 49152; i++) {
    data_cpu[i] = rand() % 256;
  }

  cudaSetDevice(0);
  for (uint32_t frame_id = 0; frame_id < 30; frame_id += 2) {
    for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols; symbol_id++) {
      BP::process(symbol_id, frame_id);
    }
  }

  cudaStream_t stream0;
  cudaStreamCreate(&stream0);

  int *flag0;
  cudaMallocManaged(&flag0, sizeof(int));

  char *data0, *result0;

  cudaMalloc(&data0, 49152 * sizeof(char));
  cudaMalloc(&result0, 49152 * sizeof(char));

  cudaMemcpy(data0, data_cpu, 49152 * sizeof(char), cudaMemcpyHostToDevice);

  cudaSetDevice(1);
  for (uint32_t frame_id = 1; frame_id < 30; frame_id += 2) {
    for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols; symbol_id++) {
      BP::process(symbol_id, frame_id);
    }
  }

  cudaStream_t stream1;
  cudaStreamCreate(&stream1);

  int *flag1;
  cudaMallocManaged(&flag1, sizeof(int));

  char *data1, *result1;

  cudaMalloc(&data1, 49152 * sizeof(char));
  cudaMalloc(&result1, 49152 * sizeof(char));

  cudaMemcpy(data1, data_cpu, 49152 * sizeof(char), cudaMemcpyHostToDevice);

  for (int vsm = 14; vsm >= 7; vsm--) {
    cudaSetDevice(0);
    *flag0 = 0;
    cudaDeviceSynchronize();
    *flag0 = 1;
    arti_launch(flag0, data0, result0, 80 - vsm, stream0);

    cudaSetDevice(1);
    *flag1 = 0;
    cudaDeviceSynchronize();
    *flag1 = 1;
    arti_launch(flag1, data1, result1, 80 - (14 - vsm), stream1);

    {
      std::vector<double> throughputs;
      for (uint32_t i = 0; i < kIterations; i++) {
        TimerCPU ctimer;
        std::vector<std::thread> threads;
        int flag = 0;
        for (uint32_t j = 0; j < 14; j++) {
          threads.push_back(std::thread([&flag, j, &vsm]() {
            int dev = j < vsm ? 0 : 1;
            int frm_id = (dev == 0) ? (j * 2) : ((j - vsm) * 2 + 1);
            cudaSetDevice(dev);
            while (flag == 0);
            BP::frame(frm_id);
          }));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        ctimer.start();
        flag = 1;
        for (auto &t : threads) {
          t.join();
        }
        ctimer.stop();
        throughputs.push_back(14.0 / ctimer.get_duration_ms());
      }
      auto cal_stat = statics(throughputs);
      printf("vSM: %d, frame, mean: %f, stdev: %f\n", vsm, cal_stat.first,
             cal_stat.second);
      my_logger->info("{}/{},frame,{},{}", 14 - vsm, vsm, cal_stat.first,
                      cal_stat.second);
    }

    {
      std::vector<double> throughputs;
      for (uint32_t i = 0; i < kIterations; i++) {
        TimerCPU ctimer;
        std::vector<std::thread> threads;
        int flag = 0;
        for (uint32_t j = 0; j < 14; j++) {
          threads.push_back(std::thread([&flag, j, &vsm]() {
            cudaSetDevice(0);
            int prev_dev = 0;
            while (flag == 0);
            for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols;
                 symbol_id++) {
              int dev = symbol_id < vsm ? 0 : 1;
              if (dev != prev_dev) {
                cudaSetDevice(dev);
                BP::copy(j * 2 + prev_dev, j * 2 + dev, 0, 1);
                prev_dev = dev;
              }
              BP::symbol(symbol_id, j * 2 + dev);
            }
          }));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        flag = 1;
        ctimer.start();
        for (auto &t : threads) {
          t.join();
        }
        ctimer.stop();
        throughputs.push_back(14.0 / ctimer.get_duration_ms());
      }
      auto cal_stat = statics(throughputs);
      printf("vSM: %d, symbol, mean: %f, stdev: %f\n", vsm, cal_stat.first,
             cal_stat.second);
      my_logger->info("{}/{},symbol,{},{}", 14 - vsm, vsm, cal_stat.first,
                      cal_stat.second);
    }

    {
      std::vector<double> throughputs;
      for (uint32_t i = 0; i < kIterations; i++) {
        TimerCPU ctimer;

        std::vector<std::thread> threads;
        int flag = 0;
        for (uint32_t j = 0; j < 14; j++) {
          threads.push_back(std::thread([&flag, j, &vsm, &num_pilots]() {
            int main_dev = j < vsm ? 0 : 1;
            cudaSetDevice(main_dev);
            int prev_dev = main_dev;
            while (flag == 0);
            for (uint32_t symbol_id = 0; symbol_id < num_pilots; symbol_id++) {
              for (uint32_t task_id = 0; task_id < 3; task_id++) {
                BP::task(task_id, symbol_id, j * 2);
              }
            }
            if (vsm != 14) {
              cudaSetDevice(1 - main_dev);
              prev_dev = 1 - main_dev;
            }
            for (uint32_t symbol_id = num_pilots - 1;
                 symbol_id < gconfig.symbols; symbol_id++) {
              if (vsm != 14 && prev_dev != 1 - main_dev) {
                cudaSetDevice(1 - main_dev);
                prev_dev = 1 - main_dev;
              }
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
                int dev = (task_id > 2 || vsm == 14) ? main_dev : 1 - main_dev;
                if (dev != prev_dev) {
                  BP::sync(j * 2 + prev_dev, symbol_id);
                  cudaSetDevice(dev);
                  BP::copy(j * 2 + prev_dev, j * 2 + dev, symbol_id, 2);
                  prev_dev = dev;
                }
                BP::task(task_id, symbol_id, j * 2 + dev);
              }
            }
          }));
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(1));
        flag = 1;
        ctimer.start();
        for (auto &t : threads) {
          t.join();
        }
        ctimer.stop();
        throughputs.push_back(14.0 / ctimer.get_duration_ms());
      }
      auto cal_stat = statics(throughputs);
      printf("vSM: %d, task, mean: %f, stdev: %f\n", vsm, cal_stat.first,
             cal_stat.second);
      my_logger->info("{}/{},task,{},{}", 14 - vsm, vsm, cal_stat.first,
                      cal_stat.second);
    }
  }

  cudaSetDevice(0);
  *flag0 = 0;
  cudaDeviceSynchronize();

  cudaSetDevice(1);
  *flag1 = 0;
  cudaDeviceSynchronize();

  cudaStreamDestroy(stream0);
  cudaStreamDestroy(stream1);
  cudaFree(flag0);
  cudaFree(flag1);
  cudaFree(data0);
  cudaFree(result0);
  cudaFree(data1);
  cudaFree(result1);
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
