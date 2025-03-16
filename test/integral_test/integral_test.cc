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

#include <fstream>

#include "baseband/buffers.h"
#include "baseband/process.h"
#include "config/config.h"
#include "utils/timer.h"

using namespace mega;

class BP : public BandProcess {
 public:
  static void process(uint32_t symbol_id) {
    BandProcess::process(0, symbol_id);
  }
};

TEST(IntegralTest, Uplink) {
  gconfig =
      Config(128, 32, 1216, 2048, 32, "PUUUUUUUUUUUUU", 6, 1. / 3., 1, 5, 16);

  BandBuffers::init();
  BandProcess::init();

  cudaSetDevice(0);

  for (uint32_t symbol_id = 0; symbol_id < 2; symbol_id++) {
    BP::process(symbol_id);
  }

  return;

  auto &h_recv_ref = BandBuffers::HostBufferRecv[0];

  std::ifstream input("../../../test/integral_test/tx_data_128_32.data",
                      std::ios::binary);
  input.seekg(0, std::ios::end);
  std::streampos fileSize = input.tellg();
  input.seekg(0, std::ios::beg);

  assert(fileSize == h_recv_ref.szBytes(h_recv_ref.nDim()));

  input.read(h_recv_ref.ptr<char>(), fileSize);
  input.close();

  cudaSetDevice(0);

  TimerCPU timer;

  timer.start();
  for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols; symbol_id++) {
    BP::process(symbol_id);
  }
  timer.stop();

  EXPECT_EQ(BandBuffers::HostBufferRecv[0].ptr<uint8_t>()[13], 254);
  EXPECT_EQ(BandBuffers::HostBufferMacSend[0][12].ptr<uint8_t>()[13], 64);

  double time = timer.get_duration_ms();
  printf("One Frame Time: %f\n", time);
}

TEST(IntegralTest, Downlink) {
  gconfig =
      Config(128, 32, 1216, 2048, 32, "PDDDDDDDDDDDDD", 6, 1. / 3., 1, 5, 16);

  BandBuffers::init();
  BandProcess::init();

  cudaSetDevice(0);

  for (uint32_t symbol_id = 0; symbol_id < 2; symbol_id++) {
    BP::process(symbol_id);
  }

  return;

  auto &dl_data = BandBuffers::HostBufferMacRecv[0];

  std::ifstream input("../../../test/integral_test/mac_data_128_32.data",
                      std::ios::binary);
  input.seekg(0, std::ios::end);
  std::streampos fileSize = input.tellg();
  input.seekg(0, std::ios::beg);

  assert(fileSize == dl_data.szBytes(dl_data.nDim()));

  input.read(dl_data.ptr<char>(), fileSize);
  input.close();

  auto &h_recv_ref = BandBuffers::HostBufferRecv[0];

  input = std::ifstream("../../../test/integral_test/tx_data_128_32.data",
                        std::ios::binary);
  input.seekg(0, std::ios::end);
  fileSize = input.tellg();
  input.seekg(0, std::ios::beg);

  assert(fileSize / 14 == h_recv_ref.szBytes(h_recv_ref.nDim()));

  input.read(h_recv_ref.ptr<char>(), fileSize / 14);
  input.close();

  cudaSetDevice(0);

  TimerCPU timer;
  timer.start();
  for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols; symbol_id++) {
    BP::process(symbol_id);
  }
  timer.stop();

  EXPECT_EQ(BandBuffers::HostBufferMacRecv[0].ptr<uint8_t>()[13], 146);
  EXPECT_EQ(BandBuffers::HostBufferSend[0][12].ptr<uint8_t>()[14], 103);
  double time = timer.get_duration_ms();
  printf("One Frame Time: %f\n", time);
}
