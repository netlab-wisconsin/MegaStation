/**
 * @file process.cc
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Baseband buffers source file
 * @version 0.1
 * @date 2023-12-17
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "buffers.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include "config/config.h"
#include "mega_complex.h"
#include "utils/types.h"

using namespace mega;

ArrStream BandBuffers::Streams;
Buffer BandBuffers::HostBufferRecv;
Buffer BandBuffers::DeviceBufferRecv;
Buffer BandBuffers::DeviceBufferCsi;
Buffer BandBuffers::DeviceBufferUFFT;
Buffer BandBuffers::DeviceBufferNmCsi;
BufferPhy BandBuffers::DeviceBufferEqualize;
BufferPhy BandBuffers::DeviceBufferDecode;
BufferPhy BandBuffers::HostBufferMacSend;
BufferPhy BandBuffers::HostBufferMacRecv;
BufferPhy BandBuffers::DeviceBufferUncode;
BufferPhy BandBuffers::DeviceBufferEncode;
Buffer BandBuffers::DeviceBufferModulate;
Buffer BandBuffers::DeviceBufferPrecode;
Buffer BandBuffers::DeviceBufferIFFT;
Buffer BandBuffers::HostBufferSend;

void BandBuffers::init() {
  const uint64_t &ofdm_cnum = gconfig.ofdm_ca;
  const uint64_t &ofdm_dnum = gconfig.ofdm_data;
  const uint32_t &ant_num = gconfig.antennas;
  const uint32_t &ue_num = gconfig.users;
  const uint32_t &symbol_num = gconfig.symbols;

  uint8_t frame_id = 0;
  for (auto &stream_arr : Streams) {
    uint32_t symbol_id = 0;
    cudaSetDevice(frame_id % kMaxDeviceNum);
    for (auto &s : stream_arr) {
      if (symbol_id >= symbol_num) break;
      cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
      symbol_id++;
    }
    frame_id++;
  }

  for (uint8_t frame_id = 0; frame_id < kFrameWindow; frame_id++) {
    cudaSetDevice(frame_id % kMaxDeviceNum);

    const uint32_t &pilot_syms = gconfig.frame.pilot_syms;
    const uint32_t &uplink_syms = gconfig.frame.uplink_syms;

    Matrix h_recv_buffer(2 * sizeof(QuanT), ofdm_cnum, ant_num,
                         pilot_syms + uplink_syms, Matrix::kHPin);
    Matrix d_recv_buffer(2 * sizeof(QuanT), ofdm_cnum, ant_num,
                         pilot_syms + uplink_syms, Matrix::kDevice);

    HostBufferRecv[frame_id] = h_recv_buffer;
    DeviceBufferRecv[frame_id] = d_recv_buffer;

    const uint32_t &sc_group = gconfig.sc_group;
    const uint32_t scg_count = ofdm_dnum / sc_group;

    // Don't use ofdm_dnum here to create enough space for FFT
    Matrix d_csi_buffer(sizeof(Complex), ue_num, ant_num, scg_count,
                        Matrix::kDevice);
    Matrix d_nmcsi_buffer(sizeof(Complex), ant_num, ue_num, scg_count,
                          Matrix::kDevice);

    DeviceBufferCsi[frame_id] = d_csi_buffer;
    DeviceBufferNmCsi[frame_id] = d_nmcsi_buffer;

    if (uplink_syms > 0) {
      // After Transpose
      Matrix d_ufft_buffer(sizeof(Complex), ofdm_dnum, ant_num, uplink_syms,
                           Matrix::kDevice);

      DeviceBufferUFFT[frame_id] = d_ufft_buffer;

      const uint64_t &encoded_bits = gconfig.ldpc_uconfig.encoded_bits;
      const uint64_t &punctured_bits = gconfig.ldpc_uconfig.punctured_bits;
      const uint64_t &decoded_bits = gconfig.ldpc_uconfig.decoded_bits;

      CuphyTensor d_equalize_buffer(CuphyTensor::kHalf,
                                    encoded_bits + punctured_bits,
                                    ue_num * kCbPerSymbol, uplink_syms,
                                    Matrix::kDevice, CuphyTensor::kCoalesce);

      DeviceBufferEqualize[frame_id] = d_equalize_buffer;

      CuphyTensor d_decode_buffer(CuphyTensor::kBit, decoded_bits,
                                  ue_num * kCbPerSymbol, uplink_syms,
                                  Matrix::kDevice, CuphyTensor::kCoalesce);
      CuphyTensor h_decode_buffer(CuphyTensor::kBit, decoded_bits,
                                  ue_num * kCbPerSymbol, uplink_syms,
                                  Matrix::kHPin, CuphyTensor::kCoalesce);

      DeviceBufferDecode[frame_id] = d_decode_buffer;
      HostBufferMacSend[frame_id] = h_decode_buffer;
    }

    const uint32_t &downlink_syms = gconfig.frame.downlink_syms;
    if (downlink_syms > 0) {
      const uint64_t &encoded_bits = gconfig.ldpc_dconfig.encoded_bits;
      const uint64_t &decoded_bits = gconfig.ldpc_dconfig.decoded_bits;

      CuphyTensor h_uncode_buffer(CuphyTensor::kBit, decoded_bits,
                                  ue_num * kCbPerSymbol, downlink_syms,
                                  Matrix::kHPin);
      CuphyTensor d_uncode_buffer(CuphyTensor::kBit, decoded_bits,
                                  ue_num * kCbPerSymbol, downlink_syms,
                                  Matrix::kDevice);

      HostBufferMacRecv[frame_id] = h_uncode_buffer;
      DeviceBufferUncode[frame_id] = d_uncode_buffer;

      CuphyTensor d_encode_buffer(CuphyTensor::kBit, encoded_bits,
                                  ue_num * kCbPerSymbol, downlink_syms,
                                  Matrix::kDevice);

      DeviceBufferEncode[frame_id] = d_encode_buffer;

      Matrix d_modulate_buffer(sizeof(Complex), ofdm_dnum, ue_num,
                               downlink_syms, Matrix::kDevice);

      DeviceBufferModulate[frame_id] = d_modulate_buffer;

      Matrix d_precode_buffer(sizeof(Complex), ofdm_cnum, ant_num,
                              downlink_syms, Matrix::kDevice);

      DeviceBufferPrecode[frame_id] = d_precode_buffer;

      Matrix d_ifft_buffer(2 * sizeof(QuanT), ofdm_cnum, ant_num, downlink_syms,
                           Matrix::kDevice);

      DeviceBufferIFFT[frame_id] = d_ifft_buffer;

      Matrix h_send_buffer(2 * sizeof(QuanT), ofdm_cnum, ant_num, downlink_syms,
                           Matrix::kHPin);

      HostBufferSend[frame_id] = h_send_buffer;
    }
  }
}

void BandBuffers::destroy() {
  for (auto &stream_arr : Streams) {
    uint32_t symbol_id = 0;
    for (auto &s : stream_arr) {
      if (symbol_id >= gconfig.symbols) break;
      cudaStreamDestroy(s);
    }
  }
}