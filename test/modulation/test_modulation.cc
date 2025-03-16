#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "complex.h"
#include "matrix/matrix.h"
#include "modulation/demodulation.h"
#include "modulation/modulation.h"

TEST(ModulateTest, ModulateSingle) {
  mega::Matrix h_input(sizeof(uint8_t), (64 * 6) / 8, 4, mega::Matrix::kHost);
  mega::Matrix h_mod(sizeof(mega::Complex), 64, 4, mega::Matrix::kHost);
  mega::Matrix d_input(sizeof(uint8_t), (64 * 6) / 8, 4, mega::Matrix::kDevice);
  mega::Matrix d_mod(sizeof(mega::Complex), 64, 4, mega::Matrix::kDevice);

  uint8_t input_arr[64];
  for (int i = 0; i < 64; ++i) {
    input_arr[i] = static_cast<uint8_t>(i);
  }
  for (int i = 0; i < 4; ++i) {
    uint8_t byte = 0;
    for (int bit = 0; bit < 64 * 6; ++bit) {
      byte |= ((input_arr[bit / 6] >> (bit % 6)) & 0x1) << (bit % 8);
      if ((bit + 1) % 8 == 0) {
        ((uint8_t *)h_input[i].ptr())[bit / 8] = byte;
        byte = 0;
      }
    }
  }
  cudaMemcpy(d_input.ptr(), h_input.ptr(), h_input.szBytes(),
             cudaMemcpyHostToDevice);

  mega::Modulation::init(6, 64, 4, 0);
  mega::Modulation::modulate(d_input, d_mod);
  mega::Modulation::destroy();
  cudaMemcpy(h_mod.ptr(), d_mod.ptr(), h_mod.szBytes(), cudaMemcpyDeviceToHost);

  uint8_t output_arr[4][64];
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 64; ++j) {
      half llr[6];
      mega::demod64QAM(*h_mod[i][j].ptr<mega::Complex>(), llr);
      uint8_t byte = 0;
      for (int bit = 0; bit < 6; ++bit) {
        byte |= (llr[bit] < half(0.0)) << bit;
      }
      output_arr[i][j] = byte;
    }
  }

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 64; ++j) {
      EXPECT_EQ(input_arr[j], output_arr[i][j]);
    }
  }
}

TEST(ModulateTest, ModulateMutiThread) {
  mega::Matrix h_input(sizeof(uint8_t), (64 * 6) / 8, 4, mega::Matrix::kHost);
  mega::Matrix h_mod(sizeof(mega::Complex), 64 + 1, 4, mega::Matrix::kHost);
  // mega::Matrix d_input(sizeof(uint8_t), (64 * 6) / 8, 4,
  // mega::Matrix::kDevice); mega::Matrix d_mod(sizeof(mega::Complex), 4, 64 +
  // 1, mega::Matrix::kDevice);

  uint8_t input_arr[64];
  for (int i = 0; i < 64; ++i) {
    input_arr[i] = static_cast<uint8_t>(i);
  }
  for (int i = 0; i < 4; ++i) {
    uint8_t byte = 0;
    for (int bit = 0; bit < 64 * 6; ++bit) {
      byte |= ((input_arr[bit / 6] >> (bit % 6)) & 0x1) << (bit % 8);
      if ((bit + 1) % 8 == 0) {
        ((uint8_t *)h_input[i].ptr())[bit / 8] = byte;
        byte = 0;
      }
    }
  }

  std::vector<std::thread> threads;
  std::vector<mega::Matrix> d_inputs;
  std::vector<mega::Matrix> d_mods;
  std::vector<cudaStream_t> streams;
  mega::Modulation::init(6, 64, 4, 64 + 1);
  for (int i = 0; i < 8; ++i) {
    d_inputs.emplace_back(sizeof(uint8_t), (64 * 6) / 8, 4,
                          mega::Matrix::kDevice);
    d_mods.emplace_back(sizeof(mega::Complex), 64 + 1, 4,
                        mega::Matrix::kDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    streams.emplace_back(stream);

    threads.emplace_back([&, i]() {
      cudaMemcpyAsync(d_inputs[i].ptr(), h_input.ptr(), h_input.szBytes(),
                      cudaMemcpyHostToDevice, streams[i]);
      mega::Modulation::modulate(d_inputs[i], d_mods[i], streams[i]);
      cudaStreamSynchronize(streams[i]);
    });
  }

  for (int i = 0; i < 8; ++i) {
    threads[i].join();
  }

  mega::Modulation::destroy();
  for (auto &stream : streams) {
    cudaStreamDestroy(stream);
  }
}