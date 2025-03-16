#include "cpu_buffer.h"

using namespace mega;

CPUMemBuffer::CPUMemBuffer() {
  const uint64_t &ofdm_cnum = gconfig.ofdm_ca;
  const uint64_t &ofdm_dnum = gconfig.ofdm_data;
  const uint32_t &ant_num = gconfig.antennas;
  const uint32_t &ue_num = gconfig.users;
  const uint32_t &pilot_syms = gconfig.frame.pilot_syms;
  const uint32_t &uplink_syms = gconfig.frame.uplink_syms;
  const uint64_t &udecoded_bits = gconfig.ldpc_uconfig.decoded_bits;
  const uint64_t &ddecoded_bits = gconfig.ldpc_dconfig.decoded_bits;
  const uint32_t &downlink_syms = gconfig.frame.downlink_syms;

  for (uint32_t i = 0; i < kFrameWindow; i++) {
    Matrix h_recv_buffer;
    CuphyTensor h_uncode_buffer;
    CuphyTensor h_decode_buffer;
    Matrix h_send_buffer;

    h_recv_buffer = Matrix(2 * sizeof(QuanT), ofdm_cnum, ant_num,
                           pilot_syms + uplink_syms, Matrix::kHPin);
    if (downlink_syms > 0)
      h_uncode_buffer =
          CuphyTensor(CuphyTensor::kBit, ddecoded_bits, ue_num * kCbPerSymbol,
                      downlink_syms, Matrix::kHPin);
    if (uplink_syms > 0)
      h_decode_buffer =
          CuphyTensor(CuphyTensor::kBit, udecoded_bits, ue_num * kCbPerSymbol,
                      uplink_syms, Matrix::kHPin, CuphyTensor::kCoalesce);
    if (downlink_syms > 0)
      h_send_buffer = Matrix(2 * sizeof(QuanT), ofdm_cnum, ant_num,
                             downlink_syms, Matrix::kHPin);

    for (uint16_t j = 0; j < gconfig.symbols; j++) {
      uint32_t local_j = gconfig.frame.gidx_lidx[j];
      if (gconfig.frame.gidx_sym[j] <= Config::Pilot) {
        buf_in[{i, j}] = h_recv_buffer[local_j];
      } else if (gconfig.frame.gidx_sym[j] == Config::Uplink) {
        buf_in[{i, j}] = h_recv_buffer[local_j + pilot_syms];
        buf_out[{i, j}] = h_decode_buffer[local_j];
      } else if (gconfig.frame.gidx_sym[j] == Config::Downlink) {
        buf_in[{i, j}] = h_uncode_buffer[local_j];
        buf_out[{i, j}] = h_send_buffer[local_j];
      }
    }
  }
}