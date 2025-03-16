/**
 * @file modulation.cc
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Modulation Structure Static Members
 * @version 0.1
 * @date 2023-11-26
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <algorithm>
#include <cmath>
#include <stdexcept>

#include "mmlt.cuh"
#include "modulation.h"
#include "modulation_kernel.cuh"
#include "utils.h"

using namespace mega;

void Modulation::initQPSK() {
  float scale = 1 / sqrt(2);
  float mod_qpsk[2] = {-scale, scale};
  for (int i = 0; i < 4; i++) {
    modTable_cpu[i] = {mod_qpsk[i / 2], mod_qpsk[i % 2]};
  }
}

void Modulation::init16QAM() {
  float scale = 1 / sqrt(10);
  float mod_16qam[4] = {1 * scale, 3 * scale, (-1) * scale, (-3) * scale};
  for (int i = 0; i < 16; i++) {
    /* get bit 2 and 0 */
    int imag_i = (((i >> 2) & 0x1) << 1) + (i & 0x1);
    /* get bit 3 and 1 */
    int real_i = (((i >> 3) & 0x1) << 1) + ((i >> 1) & 0x1);
    modTable_cpu[i] = {mod_16qam[real_i], mod_16qam[imag_i]};
  }
}

void Modulation::init64QAM() {
  float scale = 1 / sqrt(42);
  float mod_64qam[8] = {3 * scale,    1 * scale,    5 * scale,    7 * scale,
                        (-3) * scale, (-1) * scale, (-5) * scale, (-7) * scale};
  for (int i = 0; i < 64; i++) {
    /* get bit 4, 2, 0 */
    int imag_i = (((i >> 4) & 0x1) << 2) + (((i >> 2) & 0x1) << 1) + (i & 0x1);
    /* get bit 5, 3, 1 */
    int real_i =
        (((i >> 5) & 0x1) << 2) + (((i >> 3) & 0x1) << 1) + ((i >> 1) & 0x1);
    modTable_cpu[i] = {mod_64qam[real_i], mod_64qam[imag_i]};
  }
}

void Modulation::init256QAM() {
  float scale = 1 / sqrt(170);
  float mod_256qam[16] = {
      (-15) * scale, (-13) * scale, (-9) * scale, (-11) * scale,
      (-1) * scale,  (-3) * scale,  (-7) * scale, (-5) * scale,
      15 * scale,    13 * scale,    9 * scale,    11 * scale,
      1 * scale,     3 * scale,     7 * scale,    5 * scale};
  for (int i = 0; i < 256; i++) {
    // Get bits 6, 4, 2, 0 (and pack into 4 bit integer)
    int imag_i = (((i & 0x40) >> 3)) + (((i & 0x10) >> 2)) +
                 (((i & 0x4) >> 1)) + (i & 0x1);
    // Get bits 7, 5, 3, 1 (and pack into 4 bit integer)
    int real_i = (((i & 0x80) >> 4)) + (((i & 0x20) >> 3)) +
                 (((i & 0x8) >> 2)) + (((i & 0x2) >> 1));
    modTable_cpu[i] = {mod_256qam[real_i], mod_256qam[imag_i]};
  }
}

ModParams Modulation::mod_params = {
    .order = 0, .pilot_table = nullptr, .pilot_spacing = 0};
std::vector<Complex*> Modulation::pilot_tables;

void Modulation::init(const uint8_t order, const uint64_t seq_len,
                      const uint64_t num_ues, const uint8_t pilot_spacing,
                      const uint8_t num_device) {
  mod_params.order = order;
  Matrix pilot_table_cpu(sizeof(Complex), seq_len, num_ues, Matrix::kHost);
  Matrix seq = zadoff_chu_sequence(seq_len);
  for (uint64_t i = 0; i < num_ues; i++) {
    cyclic_shift(seq, pilot_table_cpu[i], i * M_PI / 6);
  }
  mod_params.pilot_table = nullptr;
  mod_params.pilot_spacing = pilot_spacing;
  switch (order) {
    case 2:
      initQPSK();
      break;
    case 4:
      init16QAM();
      break;
    case 6:
      init64QAM();
      break;
    case 8:
      init256QAM();
      break;
    default:
      throw std::invalid_argument("Invalid modulation order");
  }

  pilot_tables.reserve(num_device);
  std::fill(pilot_tables.begin(), pilot_tables.end(), nullptr);
  for (uint8_t i = 0; i < num_device; i++) {
    cudaSetDevice(i);
    pilot_tables[i] = seq_to_gpu(pilot_table_cpu);
    init_modulation_cuda_table(modTable_cpu);
  }
}

void Modulation::modulate(const uint8_t* in, Complex* out, uint64_t in_bytes,
                          uint64_t num_carriers, uint64_t num_ues,
                          cudaStream_t stream) {
  if (mod_params.order == 0) {
    throw std::runtime_error("Modulation order not set, use init() to set.");
  }

  ModParams local_mp = mod_params;
  local_mp.pilot_table = pilot_tables[get_device_id()];
  dim3 block = {256, 1, 1};
  dim3 grid = get_grid_shape({num_carriers, 1, num_ues}, block);
  modulate_kernel<<<grid, block, 0, stream>>>(in, out, local_mp, in_bytes,
                                              num_carriers, num_ues);
}

void Modulation::modulate(const Matrix& in, const Matrix& out,
                          cudaStream_t stream) {
  const uint64_t in_bytes =
      in.szBytes(0) > 1 ? in.szBytes(1) / in.szBytes(0) : in.szBytes(1);
  const uint64_t num_carriers = out.dim(0);
  const uint64_t num_ues = out.dim(1);
  modulate(in.ptr<uint8_t>(), out.ptr<Complex>(), in_bytes, num_carriers,
           num_ues, stream);
}