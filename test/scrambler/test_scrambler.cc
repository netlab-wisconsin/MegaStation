#include <gtest/gtest.h>

#include <thread>
#include <vector>

#include "matrix/matrix.h"
#include "scrambler/scrambler.h"

TEST(ScramblerTest, ScrambleIntegratedSingle) {
  char h_input[] = "Hello World!";
  mega::Matrix d_input(sizeof(char), sizeof(h_input) / sizeof(char),
                       mega::Matrix::kDevice);
  cudaMemcpy(d_input.ptr(), h_input, sizeof(h_input), cudaMemcpyHostToDevice);
  mega::Scrambler::scrambler(d_input, d_input);
  mega::Scrambler::scrambler(d_input, d_input);
  cudaMemcpy(h_input, d_input.ptr(), sizeof(h_input), cudaMemcpyDeviceToHost);
  EXPECT_STREQ("Hello World!", h_input);
}

TEST(ScramblerTest, ScrambleIntegratedMulti) {
  mega::Matrix h_input(sizeof(uint8_t), 137, 4, mega::Matrix::kHost);
  mega::Matrix h_return(sizeof(uint8_t), 137, 4, mega::Matrix::kHost);
  mega::Matrix d_input(sizeof(uint8_t), 137, 4, mega::Matrix::kDevice);
  cudaStream_t stream;
  cudaStreamCreate(&stream);
  srand(time(NULL));
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 137; ++j) {
      *(uint8_t *)(h_input[i][j].ptr()) = static_cast<uint8_t>(rand());
    }
  }
  cudaMemcpyAsync(d_input.ptr(), h_input.ptr(), h_input.szBytes(),
                  cudaMemcpyHostToDevice, stream);
  mega::Scrambler::scrambler(d_input, d_input, stream);
  mega::Scrambler::scrambler(d_input, d_input, stream);
  cudaMemcpyAsync(h_return.ptr(), d_input.ptr(), h_input.szBytes(),
                  cudaMemcpyDeviceToHost, stream);
  cudaStreamSynchronize(stream);

  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < 137; ++j) {
      EXPECT_EQ(*(uint8_t *)(h_input[i][j].ptr()),
                *(uint8_t *)(h_return[i][j].ptr()));
    }
  }
}

TEST(ScramblerTest, ScrambleMultiThread) {
  std::vector<std::thread> threads;
  std::vector<mega::Matrix> d_matrix;
  std::vector<cudaStream_t> streams;

  for (int i = 0; i < 8; ++i) {
    d_matrix.emplace_back(sizeof(uint8_t), 128, 4, mega::Matrix::kDevice);
    cudaStream_t stream;
    cudaStreamCreate(&stream);
    streams.emplace_back(stream);
  }

  for (int i = 0; i < 8; ++i) {
    threads.emplace_back([&, i]() {
      mega::Scrambler::scrambler(d_matrix[i], d_matrix[i], streams[i]);
      mega::Scrambler::scrambler(d_matrix[i], d_matrix[i], streams[i]);
      cudaStreamSynchronize(streams[i]);
    });
  }

  for (int i = 0; i < 8; ++i) {
    threads[i].join();
  }

  for (int i = 0; i < 8; ++i) {
    cudaStreamDestroy(streams[i]);
  }
}