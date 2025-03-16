#include <gtest/gtest.h>

#include <fstream>

#include "complex.h"
#include "matrix/matrix.h"
#include "mega_complex.h"
#include "sgemv/equalize.h"
#include "sgemv/precode.h"

class SGemvTest : public testing::Test {
 protected:
  std::shared_ptr<mega::Complex> csi;
  size_t csi_sz;
  std::shared_ptr<mega::Complex> bsVec;
  size_t bsVec_sz;
  std::shared_ptr<half> equalize;
  size_t equalize_sz;

  std::shared_ptr<mega::Complex> prec;
  size_t prec_sz;
  std::shared_ptr<mega::Complex> ueVec;
  size_t ueVec_sz;
  std::shared_ptr<mega::Complex> precoded;
  size_t precoded_sz;

  uint32_t ue;
  uint32_t bs;
  uint32_t ofdm;
  uint32_t ofdmE;

  SGemvTest() : ue(4), bs(8), ofdm(16), ofdmE(32) {
    // read csi data (generated by numpy)
    std::ifstream input("../../../test/sgemv/csi.data", std::ios::binary);
    input.seekg(0, std::ios::end);
    std::streampos fileSize = input.tellg();
    input.seekg(0, std::ios::beg);

    assert(fileSize == ue * bs * sizeof(mega::Complex));
    csi_sz = fileSize;

    std::shared_ptr<char> temp = std::shared_ptr<char>(new char[fileSize]);
    input.read(temp.get(), fileSize);
    input.close();

    csi = reinterpret_cast<std::shared_ptr<mega::Complex> &&>(temp);

    // read bsVec data (generated by numpy)
    input = std::ifstream("../../../test/sgemv/bsVec.data", std::ios::binary);
    input.seekg(0, std::ios::end);
    fileSize = input.tellg();
    input.seekg(0, std::ios::beg);

    assert(fileSize == bs * ofdm * sizeof(mega::Complex));
    bsVec_sz = fileSize;

    temp = std::shared_ptr<char>(new char[fileSize]);
    input.read(temp.get(), fileSize);
    input.close();

    bsVec = reinterpret_cast<std::shared_ptr<mega::Complex> &&>(temp);

    // read equalized data (generated by numpy)
    std::ifstream output("../../../test/sgemv/equalize.data", std::ios::binary);
    output.seekg(0, std::ios::end);
    fileSize = output.tellg();
    output.seekg(0, std::ios::beg);

    assert(fileSize == ue * ofdm * sizeof(half) * 2);
    equalize_sz = fileSize;

    temp = std::shared_ptr<char>(new char[fileSize]);
    output.read(temp.get(), fileSize);
    output.close();

    equalize = reinterpret_cast<std::shared_ptr<half> &&>(temp);

    // read prec data (generated by numpy)
    input = std::ifstream("../../../test/sgemv/prec.data", std::ios::binary);
    input.seekg(0, std::ios::end);
    fileSize = input.tellg();
    input.seekg(0, std::ios::beg);

    assert(fileSize == ue * bs * sizeof(mega::Complex));
    prec_sz = fileSize;

    temp = std::shared_ptr<char>(new char[fileSize]);
    input.read(temp.get(), fileSize);
    input.close();

    prec = reinterpret_cast<std::shared_ptr<mega::Complex> &&>(temp);

    // read ueVec data (generated by numpy)
    input = std::ifstream("../../../test/sgemv/ueVec.data", std::ios::binary);
    input.seekg(0, std::ios::end);
    fileSize = input.tellg();
    input.seekg(0, std::ios::beg);

    assert(fileSize == ue * ofdm * sizeof(mega::Complex));
    ueVec_sz = fileSize;

    temp = std::shared_ptr<char>(new char[fileSize]);
    input.read(temp.get(), fileSize);
    input.close();

    ueVec = reinterpret_cast<std::shared_ptr<mega::Complex> &&>(temp);

    // read precoded data (generated by numpy)
    output =
        std::ifstream("../../../test/sgemv/precoded.data", std::ios::binary);
    output.seekg(0, std::ios::end);
    fileSize = output.tellg();
    output.seekg(0, std::ios::beg);

    assert(fileSize == bs * ofdmE * sizeof(mega::Complex));
    precoded_sz = fileSize;

    temp = std::shared_ptr<char>(new char[fileSize]);
    output.read(temp.get(), fileSize);
    output.close();

    precoded = reinterpret_cast<std::shared_ptr<mega::Complex> &&>(temp);
  }
};

TEST_F(SGemvTest, Equalize) {
  mega::Matrix d_csi(sizeof(mega::Complex), ue, bs, mega::Matrix::kDevice);
  mega::Matrix d_bsVec(sizeof(mega::Complex), ofdm, bs, mega::Matrix::kDevice);
  mega::Matrix d_equalize(sizeof(half), 2 * ofdm, 1 * ue,
                          mega::Matrix::kDevice);

  EXPECT_EQ(csi_sz, d_csi.szBytes());
  EXPECT_EQ(bsVec_sz, d_bsVec.szBytes());
  EXPECT_EQ(equalize_sz, d_equalize.szBytes());

  mega::Complex *bsVec_ptr = bsVec.get();
  mega::Complex *bsVec_transpose_ptr = new mega::Complex[bs * ofdm];
  for (int i = 0; i < ofdm; i++) {
    for (int j = 0; j < bs; j++) {
      bsVec_transpose_ptr[j * ofdm + i] = bsVec_ptr[i * bs + j];
    }
  }

  cudaMemcpy(d_csi.ptr(), csi.get(), d_csi.szBytes(), cudaMemcpyHostToDevice);
  cudaMemcpy(d_bsVec.ptr(), bsVec_transpose_ptr, d_bsVec.szBytes(),
             cudaMemcpyHostToDevice);

  mega::Equalize::equalize(d_csi, d_bsVec, d_equalize, ofdm, 2, 2 * ofdm, 0, 1);

  mega::Matrix h_equalize(sizeof(half), 2 * ofdm, ue, mega::Matrix::kHost);
  cudaMemcpy(h_equalize.ptr(), d_equalize.ptr(), h_equalize.szBytes(),
             cudaMemcpyDeviceToHost);

  half *h_equalize_ptr = h_equalize.ptr<half>();
  half *ref_equalize_ptr = equalize.get();
  for (int i = 0; i < 2 * ue * ofdm; ++i) {
    EXPECT_EQ(h_equalize_ptr[i], ref_equalize_ptr[i]);
  }
}

TEST_F(SGemvTest, Precode) {
  mega::Matrix d_prec(sizeof(mega::Complex), bs, ue, mega::Matrix::kDevice);
  mega::Matrix d_ueVec(sizeof(mega::Complex), ofdm, ue, mega::Matrix::kDevice);
  mega::Matrix d_precoded(sizeof(mega::Complex), ofdmE, bs,
                          mega::Matrix::kDevice);

  EXPECT_EQ(prec_sz, d_prec.szBytes());
  EXPECT_EQ(ueVec_sz, d_ueVec.szBytes());
  EXPECT_EQ(precoded_sz, d_precoded.szBytes());

  mega::Complex *ueVec_ptr = ueVec.get();
  mega::Complex *ueVec_transpose_ptr = new mega::Complex[ue * ofdm];
  for (int i = 0; i < ofdm; i++) {
    for (int j = 0; j < ue; j++) {
      ueVec_transpose_ptr[j * ofdm + i] = ueVec_ptr[i * ue + j];
    }
  }

  cudaMemcpy(d_prec.ptr(), prec.get(), d_prec.szBytes(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_ueVec.ptr(), ueVec_transpose_ptr, d_ueVec.szBytes(),
             cudaMemcpyHostToDevice);

  mega::Precode::precode(d_prec, d_ueVec, d_precoded, ofdm, 8);

  mega::Matrix h_precoded(sizeof(mega::Complex), ofdmE, bs,
                          mega::Matrix::kHost);
  cudaMemcpy(h_precoded.ptr(), d_precoded.ptr(), h_precoded.szBytes(),
             cudaMemcpyDeviceToHost);

  float *h_precoded_ptr = h_precoded.ptr<float>();
  float *ref_precoded_ptr = (float *)precoded.get();
  for (int i = 0; i < bs * ofdmE * 2; ++i) {
    EXPECT_NEAR(h_precoded_ptr[i], ref_precoded_ptr[i], 1e-5);
  }
}

TEST_F(SGemvTest, PrecodeLarge) {
  mega::Matrix d_prec(sizeof(mega::Complex), 64, 16, 75, mega::Matrix::kDevice);
  mega::Matrix d_ueVec(sizeof(mega::Complex), 1200, 16, mega::Matrix::kDevice);
  mega::Matrix d_precoded(sizeof(mega::Complex), 2048, 64,
                          mega::Matrix::kDevice);

  mega::Matrix h_prec(sizeof(mega::Complex), 64, 16, 75, mega::Matrix::kHost);
  mega::Matrix h_ueVec(sizeof(mega::Complex), 1200, 16, mega::Matrix::kHost);
  mega::Matrix h_precoded_ref(sizeof(mega::Complex), 2048, 64,
                              mega::Matrix::kHost);

  // random data
  for (int i = 0; i < 64 * 16 * 75; i++) {
    h_prec.ptr<float>()[i * 2] = (float)rand() / RAND_MAX;
    h_prec.ptr<float>()[i * 2 + 1] = (float)rand() / RAND_MAX;
  }

  for (int i = 0; i < 1200 * 16; i++) {
    h_ueVec.ptr<float>()[i * 2] = (float)rand() / RAND_MAX;
    h_ueVec.ptr<float>()[i * 2 + 1] = (float)rand() / RAND_MAX;
  }

  memset(h_precoded_ref.ptr(), 0, h_precoded_ref.szBytes());
  for (int i = 0; i < 64; i++) {
    for (int j = 0; j < 1200; j++) {
      mega::Matrix h_prec_j = h_prec[j / 16];
      mega::Complex accum = {0, 0};
      for (int k = 0; k < 16; k++) {
        mega::Complex prec_e = *h_prec_j[k][i].ptr<mega::Complex>();
        mega::Complex ueVec_e = *h_ueVec[k][j].ptr<mega::Complex>();
        accum += prec_e * ueVec_e;
      }
      int j_cs = (j + 424 + 1024) % 2048;
      *h_precoded_ref[i][j_cs].ptr<mega::Complex>() = accum;
    }
  }

  cudaMemcpy(d_prec.ptr(), h_prec.ptr(), h_prec.szBytes(),
             cudaMemcpyHostToDevice);
  cudaMemcpy(d_ueVec.ptr(), h_ueVec.ptr(), h_ueVec.szBytes(),
             cudaMemcpyHostToDevice);
  cudaMemset(d_precoded.ptr(), 0, d_precoded.szBytes());

  mega::Precode::precode(d_prec, d_ueVec, d_precoded, 16, 424);

  mega::Matrix h_precoded(sizeof(mega::Complex), 2048, 64, mega::Matrix::kHost);
  cudaMemcpy(h_precoded.ptr(), d_precoded.ptr(), h_precoded.szBytes(),
             cudaMemcpyDeviceToHost);

  float *h_precoded_ptr = h_precoded.ptr<float>();
  float *ref_precoded_ptr = h_precoded_ref.ptr<float>();
  // for (int i = 0; i < 64 * 2048 * 2; ++i) {
  //   EXPECT_NEAR(h_precoded_ptr[i], ref_precoded_ptr[i], 1e-5);
  // }
  EXPECT_NEAR(h_precoded[0][1624].ptr<float>()[0],
              h_precoded_ref[0][1624].ptr<float>()[0], 1e-5);
}