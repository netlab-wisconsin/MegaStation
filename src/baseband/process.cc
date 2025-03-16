/**
 * @file process.cc
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Baseband processing source file
 * @version 0.1
 * @date 2023-12-17
 *
 * @copyright Copyright (c) 2023
 *
 */

#include "process.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <spdlog/spdlog.h>

#include <atomic>
#include <stdexcept>

#include "buffers.h"
#include "config/config.h"
#include "fft/fft_op.h"
#include "modulation/modulation.h"
#include "scrambler/scrambler.h"
#include "sgemv/equalize.h"
#include "sgemv/precode.h"
#include "utils/pin_threads.h"
#include "utils/timer.h"
#include "utils/types.h"

using namespace mega;

// #define NOCOPY

ArrFrameSymbol<PilotFFT<QuanT>> BandProcess::pfft;
ArrFrame<Beamform> BandProcess::beam;

ArrFrameSymbol<UplinkFFT<QuanT>> BandProcess::ufft;
ArrFrame<LDPCDecoder> BandProcess::decoder;

ArrFrame<LDPCEncoder> BandProcess::encoder;
ArrFrameSymbol<DownlinkIFFT<QuanT>> BandProcess::difft;

std::atomic<uint32_t> BandProcess::thread_offset(0);

void BandProcess::init() {
  const uint64_t &ofdm_cnum = gconfig.ofdm_ca;
  const uint64_t &ofdm_dnum = gconfig.ofdm_data;
  const uint64_t &ofdm_start = gconfig.ofdm_start;
  const uint32_t &ant_num = gconfig.antennas;
  const uint32_t &ue_num = gconfig.users;
  const uint32_t &symbol_num = gconfig.symbols;
  const uint32_t &sc_group = gconfig.sc_group;

  // TODO check buffers for different devices
  PilotSign::init(ofdm_dnum, kMaxDeviceNum);
  Scrambler::init(kMaxDeviceNum);
  Modulation::init(gconfig.mod_order, ofdm_dnum, ue_num, gconfig.pilot_spacing,
                   kMaxDeviceNum);

  for (uint8_t frame_id = 0; frame_id < kFrameWindow; frame_id++) {
    cudaSetDevice(frame_id % kMaxDeviceNum);

    // Pilot Processing
    for (uint8_t pilot_id = 0; pilot_id < gconfig.frame.pilot_syms;
         pilot_id++) {
      pfft[frame_id][pilot_id] = PilotFFT<QuanT>(
          ofdm_start, ofdm_cnum, ofdm_dnum, ant_num, ue_num, sc_group);
    }
    beam[frame_id] = Beamform(ant_num, ue_num, ofdm_dnum / sc_group);

    // Uplink Processing
    for (uint8_t uplink_id = 0; uplink_id < gconfig.frame.uplink_syms;
         uplink_id++) {
      ufft[frame_id][uplink_id] =
          UplinkFFT<QuanT>(ofdm_start, ofdm_cnum, ofdm_dnum, ant_num);
    }
    decoder[frame_id] =
        LDPCDecoder(CuphyTensor::kHalf, gconfig.ldpc_uconfig);  // per-device

    // Downlink Processing
    for (uint8_t downlink_id = 0; downlink_id < gconfig.frame.downlink_syms;
         downlink_id++) {
      difft[frame_id][downlink_id] =
          DownlinkIFFT<QuanT>(ofdm_start, ofdm_cnum, ofdm_dnum, ant_num);
    }
    encoder[frame_id] = LDPCEncoder(gconfig.ldpc_dconfig);
  }
}

constexpr double kPrintThreshold = 3.5;
constexpr bool kPrintSymbol = false;

Counter BandProcess::counters;

void BandProcess::start(uint32_t frame_cid) {
  uint32_t thread_id = thread_offset.fetch_add(1);
  PinThreads::pin_thread(base_thread_id, thread_id);
  spdlog::info("START Processing Thread {} for frame_cid: {}",
               base_thread_id + thread_id, frame_cid);

  cudaSetDevice(frame_cid % kMaxDeviceNum);

  TimerCPU timer_frame, timer_symbol;

  while (true) {
    for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols; symbol_id++) {
      while (!counters.ready(frame_cid, symbol_id));
      if (symbol_id == 0) {
        timer_frame.start();
      }
      if (kPrintSymbol) timer_symbol.start();

      process(frame_cid, symbol_id);
      counters.done(frame_cid, symbol_id);

      if (kPrintSymbol) {
        timer_symbol.stop();
        double symbol_time = timer_symbol.get_duration_ms();
        spdlog::warn("COMPLETE symbol in {} ms: frame_cid: {}, symbol_id: {}",
                     symbol_time, frame_cid, symbol_id);
      }
    }
    timer_frame.stop();
    double frame_time = timer_frame.get_duration_ms();
    if (frame_time > kPrintThreshold) {
      spdlog::warn("COMPLETE frame in {} ms: frame_cid: {}",
                   timer_frame.get_duration_ms(), frame_cid);
    }
  }
}

void BandProcess::process(uint32_t frame_cid, uint32_t symbol_id) {
  auto &local_streams = BandBuffers::Streams[frame_cid];

  auto &local_hrecv = BandBuffers::HostBufferRecv[frame_cid];
  auto &local_recv = BandBuffers::DeviceBufferRecv[frame_cid];
  auto &local_ufft = BandBuffers::DeviceBufferUFFT[frame_cid];
  auto &local_csi = BandBuffers::DeviceBufferCsi[frame_cid];
  auto &local_nmcsi = BandBuffers::DeviceBufferNmCsi[frame_cid];
  auto &local_equalize = BandBuffers::DeviceBufferEqualize[frame_cid];
  auto &local_decode = BandBuffers::DeviceBufferDecode[frame_cid];
  auto &local_txmac = BandBuffers::HostBufferMacSend[frame_cid];
  auto &local_rxmac = BandBuffers::HostBufferMacRecv[frame_cid];
  auto &local_uncode = BandBuffers::DeviceBufferUncode[frame_cid];
  auto &local_encode = BandBuffers::DeviceBufferEncode[frame_cid];
  auto &local_modulate = BandBuffers::DeviceBufferModulate[frame_cid];
  auto &local_precode = BandBuffers::DeviceBufferPrecode[frame_cid];
  auto &local_ifft = BandBuffers::DeviceBufferIFFT[frame_cid];
  auto &local_hsend = BandBuffers::HostBufferSend[frame_cid];

  auto &uplink_fft = ufft[frame_cid];
  auto &pilot_fft = pfft[frame_cid];
  auto &beamform = beam[frame_cid];
  auto &ldpc_decoder = decoder[frame_cid];
  auto &ldpc_encoder = encoder[frame_cid];
  auto &downlink_ifft = difft[frame_cid];

  uint32_t local_symid = gconfig.frame.gidx_lidx[symbol_id];
  if (gconfig.frame.gidx_sym[symbol_id] <= Config::Pilot) {
    cudaMemcpyAsync(local_recv[symbol_id].ptr(), local_hrecv[symbol_id].ptr(),
                    local_hrecv[symbol_id].szBytes(), cudaMemcpyHostToDevice,
                    local_streams[symbol_id]);
    pilot_fft[local_symid](local_recv[symbol_id], local_csi,
                           local_streams[symbol_id]);
    if (local_symid == gconfig.frame.pilot_syms - 1) {
      for (uint32_t i = 0; i < symbol_id; i++) {
        if (gconfig.frame.gidx_sym[i] > Config::Pilot) continue;
        cudaStreamSynchronize(local_streams[i]);
      }
      beamform.beamformer(local_csi, local_nmcsi, local_streams[symbol_id]);
      cudaStreamSynchronize(local_streams[symbol_id]);
    }
  } else if (gconfig.frame.gidx_sym[symbol_id] == Config::Uplink) {
    cudaMemcpyAsync(local_recv[symbol_id].ptr(), local_hrecv[symbol_id].ptr(),
                    local_hrecv[symbol_id].szBytes(), cudaMemcpyHostToDevice,
                    local_streams[symbol_id]);
    uplink_fft[local_symid](local_recv[symbol_id], local_ufft[local_symid],
                            local_streams[symbol_id]);
    Equalize::equalize(local_csi, local_ufft[local_symid],
                       local_equalize[local_symid], gconfig.sc_group,
                       gconfig.mod_order, gconfig.ldpc_uconfig.encoded_bits,
                       gconfig.ldpc_uconfig.punctured_bits, kCbPerSymbol,
                       local_streams[symbol_id]);
    ldpc_decoder(local_equalize[local_symid], local_decode[local_symid],
                 local_streams[symbol_id]);
    Scrambler::scrambler(local_decode[local_symid], local_decode[local_symid],
                         local_streams[symbol_id]);
    cudaMemcpyAsync(local_txmac[local_symid].ptr(),
                    local_decode[local_symid].ptr(),
                    local_decode[local_symid].szBytes(), cudaMemcpyDeviceToHost,
                    local_streams[symbol_id]);
  } else if (gconfig.frame.gidx_sym[symbol_id] == Config::Downlink) {
    cudaMemcpyAsync(local_uncode[local_symid].ptr(),
                    local_rxmac[local_symid].ptr(),
                    local_rxmac[local_symid].szBytes(), cudaMemcpyHostToDevice,
                    local_streams[symbol_id]);
    Scrambler::scrambler(local_uncode[local_symid], local_uncode[local_symid],
                         local_streams[symbol_id]);
    ldpc_encoder(local_uncode[local_symid], local_encode[local_symid],
                 local_streams[symbol_id]);
    Modulation::modulate(local_encode[local_symid], local_modulate[local_symid],
                         local_streams[symbol_id]);
    Precode::precode(local_nmcsi, local_modulate[local_symid],
                     local_precode[local_symid], gconfig.sc_group,
                     gconfig.ofdm_start, local_streams[symbol_id]);
    downlink_ifft[local_symid](local_precode[local_symid],
                               local_ifft[local_symid],
                               local_streams[symbol_id]);
    cudaMemcpyAsync(local_hsend[local_symid].ptr(),
                    local_ifft[local_symid].ptr(),
                    local_ifft[local_symid].szBytes(), cudaMemcpyDeviceToHost,
                    local_streams[symbol_id]);
  } else {
    throw std::runtime_error("Can't Process now. Symbol type not defined");
  }
done:
  if (symbol_id == gconfig.symbols - 1) {
    for (uint32_t i = 0; i < gconfig.symbols; i++) {
      if (gconfig.frame.gidx_sym[i] <= Config::Pilot) continue;
      cudaStreamSynchronize(local_streams[i]);
    }
  }
}

void BandProcess::sync(uint32_t frame_cid, uint32_t symbol_id) {
  auto &local_streams = BandBuffers::Streams[frame_cid];
  cudaStreamSynchronize(local_streams[symbol_id]);
}

void BandProcess::copy(uint32_t frame_sid, uint32_t frame_did,
                       uint32_t symbol_id, uint32_t task_id) {
  auto &local_streams = BandBuffers::Streams[frame_did];
  uint32_t local_symid = gconfig.frame.gidx_lidx[symbol_id];

  if (gconfig.frame.gidx_sym[symbol_id] <= Config::Pilot) {
    if (task_id == 0) {
      auto &local_srecv = BandBuffers::DeviceBufferRecv[frame_sid];
      auto &local_drecv = BandBuffers::DeviceBufferRecv[frame_did];
      cudaMemcpyAsync(local_drecv[local_symid].ptr(),
                      local_srecv[local_symid].ptr(),
                      local_srecv[local_symid].szBytes(),
                      cudaMemcpyDeviceToDevice, local_streams[symbol_id]);
    } else if (task_id == 1) {
      auto &local_scsi = BandBuffers::DeviceBufferCsi[frame_sid];
      auto &local_dcsi = BandBuffers::DeviceBufferCsi[frame_did];
      cudaMemcpyAsync(local_dcsi.ptr(), local_scsi.ptr(), local_scsi.szBytes(),
                      cudaMemcpyDeviceToDevice, local_streams[symbol_id]);
    }
  } else if (gconfig.frame.gidx_sym[symbol_id] == Config::Uplink) {
    if (task_id == 0) {
      auto &local_srecv = BandBuffers::DeviceBufferRecv[frame_sid];
      auto &local_drecv = BandBuffers::DeviceBufferRecv[frame_did];
      cudaMemcpyAsync(local_drecv[local_symid].ptr(),
                      local_srecv[local_symid].ptr(),
                      local_srecv[local_symid].szBytes(),
                      cudaMemcpyDeviceToDevice, local_streams[symbol_id]);
    } else if (task_id == 1) {
      auto &local_sufft = BandBuffers::DeviceBufferUFFT[frame_sid];
      auto &local_dufft = BandBuffers::DeviceBufferUFFT[frame_did];
      cudaMemcpyAsync(local_dufft[local_symid].ptr(),
                      local_sufft[local_symid].ptr(),
                      local_sufft[local_symid].szBytes(),
                      cudaMemcpyDeviceToDevice, local_streams[symbol_id]);
    } else if (task_id == 2) {
      auto &local_sequalize = BandBuffers::DeviceBufferEqualize[frame_sid];
      auto &local_dequalize = BandBuffers::DeviceBufferEqualize[frame_did];
      cudaMemcpyAsync(local_dequalize[local_symid].ptr(),
                      local_sequalize[local_symid].ptr(),
                      local_sequalize[local_symid].szBytes(),
                      cudaMemcpyDeviceToDevice, local_streams[symbol_id]);
    } else if (task_id == 3) {
      auto &local_sdecode = BandBuffers::DeviceBufferDecode[frame_sid];
      auto &local_ddecode = BandBuffers::DeviceBufferDecode[frame_did];
      cudaMemcpyAsync(local_ddecode[local_symid].ptr(),
                      local_sdecode[local_symid].ptr(),
                      local_sdecode[local_symid].szBytes(),
                      cudaMemcpyDeviceToDevice, local_streams[symbol_id]);
    }
  } else if (gconfig.frame.gidx_sym[symbol_id] == Config::Downlink) {
    if (task_id == 0) {
      auto &local_suncode = BandBuffers::DeviceBufferUncode[frame_sid];
      auto &local_duncode = BandBuffers::DeviceBufferUncode[frame_did];
      cudaMemcpyAsync(local_duncode[local_symid].ptr(),
                      local_suncode[local_symid].ptr(),
                      local_suncode[local_symid].szBytes(),
                      cudaMemcpyDeviceToDevice, local_streams[symbol_id]);
    } else if (task_id == 1) {
      auto &local_sencode = BandBuffers::DeviceBufferEncode[frame_sid];
      auto &local_dencode = BandBuffers::DeviceBufferEncode[frame_did];
      cudaMemcpyAsync(local_dencode[local_symid].ptr(),
                      local_sencode[local_symid].ptr(),
                      local_sencode[local_symid].szBytes(),
                      cudaMemcpyDeviceToDevice, local_streams[symbol_id]);
    } else if (task_id == 2) {
      auto &local_smodulate = BandBuffers::DeviceBufferModulate[frame_sid];
      auto &local_dmodulate = BandBuffers::DeviceBufferModulate[frame_did];
      cudaMemcpyAsync(local_dmodulate[local_symid].ptr(),
                      local_smodulate[local_symid].ptr(),
                      local_smodulate[local_symid].szBytes(),
                      cudaMemcpyDeviceToDevice, local_streams[symbol_id]);
    } else if (task_id == 3) {
      auto &local_sprecode = BandBuffers::DeviceBufferPrecode[frame_sid];
      auto &local_dprecode = BandBuffers::DeviceBufferPrecode[frame_did];
      cudaMemcpyAsync(local_dprecode[local_symid].ptr(),
                      local_sprecode[local_symid].ptr(),
                      local_sprecode[local_symid].szBytes(),
                      cudaMemcpyDeviceToDevice, local_streams[symbol_id]);
    } else if (task_id == 4) {
      auto &local_sifft = BandBuffers::DeviceBufferIFFT[frame_sid];
      auto &local_difft = BandBuffers::DeviceBufferIFFT[frame_did];
      cudaMemcpyAsync(local_difft[local_symid].ptr(),
                      local_sifft[local_symid].ptr(),
                      local_sifft[local_symid].szBytes(),
                      cudaMemcpyDeviceToDevice, local_streams[symbol_id]);
    }
  }
}

void BandProcess::symbol(uint32_t frame_cid, uint32_t symbol_id) {
  auto &local_streams = BandBuffers::Streams[frame_cid];

  auto &local_hrecv = BandBuffers::HostBufferRecv[frame_cid];
  auto &local_recv = BandBuffers::DeviceBufferRecv[frame_cid];
  auto &local_ufft = BandBuffers::DeviceBufferUFFT[frame_cid];
  auto &local_csi = BandBuffers::DeviceBufferCsi[frame_cid];
  auto &local_nmcsi = BandBuffers::DeviceBufferNmCsi[frame_cid];
  auto &local_equalize = BandBuffers::DeviceBufferEqualize[frame_cid];
  auto &local_decode = BandBuffers::DeviceBufferDecode[frame_cid];
  auto &local_txmac = BandBuffers::HostBufferMacSend[frame_cid];
  auto &local_rxmac = BandBuffers::HostBufferMacRecv[frame_cid];
  auto &local_uncode = BandBuffers::DeviceBufferUncode[frame_cid];
  auto &local_encode = BandBuffers::DeviceBufferEncode[frame_cid];
  auto &local_modulate = BandBuffers::DeviceBufferModulate[frame_cid];
  auto &local_precode = BandBuffers::DeviceBufferPrecode[frame_cid];
  auto &local_ifft = BandBuffers::DeviceBufferIFFT[frame_cid];
  auto &local_hsend = BandBuffers::HostBufferSend[frame_cid];

  auto &uplink_fft = ufft[frame_cid];
  auto &pilot_fft = pfft[frame_cid];
  auto &beamform = beam[frame_cid];
  auto &ldpc_decoder = decoder[frame_cid];
  auto &ldpc_encoder = encoder[frame_cid];
  auto &downlink_ifft = difft[frame_cid];

  uint32_t local_symid = gconfig.frame.gidx_lidx[symbol_id];
  if (gconfig.frame.gidx_sym[symbol_id] <= Config::Pilot) {
#ifndef NOCOPY
    cudaMemcpyAsync(local_recv[symbol_id].ptr(), local_hrecv[symbol_id].ptr(),
                    local_hrecv[symbol_id].szBytes(), cudaMemcpyHostToDevice,
                    local_streams[symbol_id]);
#endif
    pilot_fft[local_symid](local_recv[symbol_id], local_csi,
                           local_streams[symbol_id]);
    if (local_symid == gconfig.frame.pilot_syms - 1) {
      for (uint32_t i = 0; i < symbol_id; i++) {
        if (gconfig.frame.gidx_sym[i] > Config::Pilot) continue;
        cudaStreamSynchronize(local_streams[i]);
      }
      beamform.beamformer(local_csi, local_nmcsi, local_streams[symbol_id]);
      cudaStreamSynchronize(local_streams[symbol_id]);
    }
  } else if (gconfig.frame.gidx_sym[symbol_id] == Config::Uplink) {
#ifndef NOCOPY
    cudaMemcpyAsync(local_recv[symbol_id].ptr(), local_hrecv[symbol_id].ptr(),
                    local_hrecv[symbol_id].szBytes(), cudaMemcpyHostToDevice,
                    local_streams[symbol_id]);
#endif
    uplink_fft[local_symid](local_recv[symbol_id], local_ufft[local_symid],
                            local_streams[symbol_id]);
    Equalize::equalize(local_csi, local_ufft[local_symid],
                       local_equalize[local_symid], gconfig.sc_group,
                       gconfig.mod_order, gconfig.ldpc_uconfig.encoded_bits,
                       gconfig.ldpc_uconfig.punctured_bits, kCbPerSymbol,
                       local_streams[symbol_id]);
    ldpc_decoder(local_equalize[local_symid], local_decode[local_symid],
                 local_streams[symbol_id]);
    Scrambler::scrambler(local_decode[local_symid], local_decode[local_symid],
                         local_streams[symbol_id]);
#ifndef NOCOPY
    cudaMemcpyAsync(local_txmac[local_symid].ptr(),
                    local_decode[local_symid].ptr(),
                    local_decode[local_symid].szBytes(), cudaMemcpyDeviceToHost,
                    local_streams[symbol_id]);
#endif
  } else if (gconfig.frame.gidx_sym[symbol_id] == Config::Downlink) {
#ifndef NOCOPY
    cudaMemcpyAsync(local_uncode[local_symid].ptr(),
                    local_rxmac[local_symid].ptr(),
                    local_rxmac[local_symid].szBytes(), cudaMemcpyHostToDevice,
                    local_streams[symbol_id]);
#endif
    Scrambler::scrambler(local_uncode[local_symid], local_uncode[local_symid],
                         local_streams[symbol_id]);
    ldpc_encoder(local_uncode[local_symid], local_encode[local_symid],
                 local_streams[symbol_id]);
    Modulation::modulate(local_encode[local_symid], local_modulate[local_symid],
                         local_streams[symbol_id]);
    Precode::precode(local_nmcsi, local_modulate[local_symid],
                     local_precode[local_symid], gconfig.sc_group,
                     gconfig.ofdm_start, local_streams[symbol_id]);
    downlink_ifft[local_symid](local_precode[local_symid],
                               local_ifft[local_symid],
                               local_streams[symbol_id]);
#ifndef NOCOPY
    cudaMemcpyAsync(local_hsend[local_symid].ptr(),
                    local_ifft[local_symid].ptr(),
                    local_ifft[local_symid].szBytes(), cudaMemcpyDeviceToHost,
                    local_streams[symbol_id]);
#endif
  } else {
    throw std::runtime_error("Can't Process now. Symbol type not defined");
  }
  if (symbol_id == gconfig.symbols - 1) {
    for (uint32_t i = 0; i < gconfig.symbols; i++) {
      if (gconfig.frame.gidx_sym[i] <= Config::Pilot) continue;
      cudaStreamSynchronize(local_streams[i]);
    }
  }
}

void BandProcess::frame(uint32_t frame_cid) {
  auto &local_stream = BandBuffers::Streams[frame_cid][0];

  auto &local_hrecv = BandBuffers::HostBufferRecv[frame_cid];
  auto &local_recv = BandBuffers::DeviceBufferRecv[frame_cid];
  auto &local_ufft = BandBuffers::DeviceBufferUFFT[frame_cid];
  auto &local_csi = BandBuffers::DeviceBufferCsi[frame_cid];
  auto &local_nmcsi = BandBuffers::DeviceBufferNmCsi[frame_cid];
  auto &local_equalize = BandBuffers::DeviceBufferEqualize[frame_cid];
  auto &local_decode = BandBuffers::DeviceBufferDecode[frame_cid];
  auto &local_txmac = BandBuffers::HostBufferMacSend[frame_cid];
  auto &local_rxmac = BandBuffers::HostBufferMacRecv[frame_cid];
  auto &local_uncode = BandBuffers::DeviceBufferUncode[frame_cid];
  auto &local_encode = BandBuffers::DeviceBufferEncode[frame_cid];
  auto &local_modulate = BandBuffers::DeviceBufferModulate[frame_cid];
  auto &local_precode = BandBuffers::DeviceBufferPrecode[frame_cid];
  auto &local_ifft = BandBuffers::DeviceBufferIFFT[frame_cid];
  auto &local_hsend = BandBuffers::HostBufferSend[frame_cid];

  auto &uplink_fft = ufft[frame_cid];
  auto &pilot_fft = pfft[frame_cid];
  auto &beamform = beam[frame_cid];
  auto &ldpc_decoder = decoder[frame_cid];
  auto &ldpc_encoder = encoder[frame_cid];
  auto &downlink_ifft = difft[frame_cid];

  for (uint32_t symbol_id = 0; symbol_id < gconfig.symbols; symbol_id++) {
    uint32_t local_symid = gconfig.frame.gidx_lidx[symbol_id];
    if (gconfig.frame.gidx_sym[symbol_id] <= Config::Pilot) {
#ifndef NOCOPY
      cudaMemcpyAsync(local_recv[symbol_id].ptr(), local_hrecv[symbol_id].ptr(),
                      local_hrecv[symbol_id].szBytes(), cudaMemcpyHostToDevice,
                      local_stream);
#endif
      pilot_fft[local_symid](local_recv[symbol_id], local_csi, local_stream);
      if (local_symid == gconfig.frame.pilot_syms - 1) {
        beamform.beamformer(local_csi, local_nmcsi, local_stream);
      }
    } else if (gconfig.frame.gidx_sym[symbol_id] == Config::Uplink) {
#ifndef NOCOPY
      cudaMemcpyAsync(local_recv[symbol_id].ptr(), local_hrecv[symbol_id].ptr(),
                      local_hrecv[symbol_id].szBytes(), cudaMemcpyHostToDevice,
                      local_stream);
#endif
      uplink_fft[local_symid](local_recv[symbol_id], local_ufft[local_symid],
                              local_stream);
      Equalize::equalize(local_csi, local_ufft[local_symid],
                         local_equalize[local_symid], gconfig.sc_group,
                         gconfig.mod_order, gconfig.ldpc_uconfig.encoded_bits,
                         gconfig.ldpc_uconfig.punctured_bits, kCbPerSymbol,
                         local_stream);
      ldpc_decoder(local_equalize[local_symid], local_decode[local_symid],
                   local_stream);
      Scrambler::scrambler(local_decode[local_symid], local_decode[local_symid],
                           local_stream);
#ifndef NOCOPY
      cudaMemcpyAsync(local_txmac[local_symid].ptr(),
                      local_decode[local_symid].ptr(),
                      local_decode[local_symid].szBytes(),
                      cudaMemcpyDeviceToHost, local_stream);
#endif
    } else if (gconfig.frame.gidx_sym[symbol_id] == Config::Downlink) {
#ifndef NOCOPY
      cudaMemcpyAsync(local_uncode[local_symid].ptr(),
                      local_rxmac[local_symid].ptr(),
                      local_rxmac[local_symid].szBytes(),
                      cudaMemcpyHostToDevice, local_stream);
#endif
      Scrambler::scrambler(local_uncode[local_symid], local_uncode[local_symid],
                           local_stream);
      ldpc_encoder(local_uncode[local_symid], local_encode[local_symid],
                   local_stream);
      Modulation::modulate(local_encode[local_symid],
                           local_modulate[local_symid], local_stream);
      Precode::precode(local_nmcsi, local_modulate[local_symid],
                       local_precode[local_symid], gconfig.sc_group,
                       gconfig.ofdm_start, local_stream);
      downlink_ifft[local_symid](local_precode[local_symid],
                                 local_ifft[local_symid], local_stream);
#ifndef NOCOPY
      cudaMemcpyAsync(local_hsend[local_symid].ptr(),
                      local_ifft[local_symid].ptr(),
                      local_ifft[local_symid].szBytes(), cudaMemcpyDeviceToHost,
                      local_stream);
#endif
    } else {
      throw std::runtime_error("Can't Process now. Symbol type not defined");
    }
  }
  cudaStreamSynchronize(local_stream);
}

void BandProcess::task(uint32_t frame_cid, uint32_t symbol_id,
                       uint32_t task_id) {
  auto &local_streams = BandBuffers::Streams[frame_cid];

  auto &local_hrecv = BandBuffers::HostBufferRecv[frame_cid];
  auto &local_recv = BandBuffers::DeviceBufferRecv[frame_cid];
  auto &local_ufft = BandBuffers::DeviceBufferUFFT[frame_cid];
  auto &local_csi = BandBuffers::DeviceBufferCsi[frame_cid];
  auto &local_nmcsi = BandBuffers::DeviceBufferNmCsi[frame_cid];
  auto &local_equalize = BandBuffers::DeviceBufferEqualize[frame_cid];
  auto &local_decode = BandBuffers::DeviceBufferDecode[frame_cid];
  auto &local_txmac = BandBuffers::HostBufferMacSend[frame_cid];
  auto &local_rxmac = BandBuffers::HostBufferMacRecv[frame_cid];
  auto &local_uncode = BandBuffers::DeviceBufferUncode[frame_cid];
  auto &local_encode = BandBuffers::DeviceBufferEncode[frame_cid];
  auto &local_modulate = BandBuffers::DeviceBufferModulate[frame_cid];
  auto &local_precode = BandBuffers::DeviceBufferPrecode[frame_cid];
  auto &local_ifft = BandBuffers::DeviceBufferIFFT[frame_cid];
  auto &local_hsend = BandBuffers::HostBufferSend[frame_cid];

  auto &uplink_fft = ufft[frame_cid];
  auto &pilot_fft = pfft[frame_cid];
  auto &beamform = beam[frame_cid];
  auto &ldpc_decoder = decoder[frame_cid];
  auto &ldpc_encoder = encoder[frame_cid];
  auto &downlink_ifft = difft[frame_cid];

  uint32_t last_pilot = gconfig.frame.pilot_syms - 1;

  uint32_t local_symid = gconfig.frame.gidx_lidx[symbol_id];
  if (gconfig.frame.gidx_sym[symbol_id] <= Config::Pilot) {
    if (task_id == 0) {
#ifndef NOCOPY
      cudaMemcpyAsync(local_recv[symbol_id].ptr(), local_hrecv[symbol_id].ptr(),
                      local_hrecv[symbol_id].szBytes(), cudaMemcpyHostToDevice,
                      local_streams[symbol_id]);
#endif
    } else if (task_id == 1) {
      pilot_fft[local_symid](local_recv[symbol_id], local_csi,
                             local_streams[symbol_id]);
    } else if (task_id == 2) {
      if (local_symid == gconfig.frame.pilot_syms - 1) {
        for (uint32_t i = 0; i < symbol_id; i++) {
          if (gconfig.frame.gidx_sym[i] > Config::Pilot) continue;
          cudaStreamSynchronize(local_streams[i]);
        }
        beamform.beamformer(local_csi, local_nmcsi, local_streams[symbol_id]);
      }
    }
  } else if (gconfig.frame.gidx_sym[symbol_id] == Config::Uplink) {
    if (task_id == 0) {
#ifndef NOCOPY
      cudaMemcpyAsync(local_recv[symbol_id].ptr(), local_hrecv[symbol_id].ptr(),
                      local_hrecv[symbol_id].szBytes(), cudaMemcpyHostToDevice,
                      local_streams[symbol_id]);
#endif
    } else if (task_id == 1) {
      uplink_fft[local_symid](local_recv[symbol_id], local_ufft[local_symid],
                              local_streams[symbol_id]);
    } else if (task_id == 2) {
      cudaStreamSynchronize(local_streams[last_pilot]);
      Equalize::equalize(local_csi, local_ufft[local_symid],
                         local_equalize[local_symid], gconfig.sc_group,
                         gconfig.mod_order, gconfig.ldpc_uconfig.encoded_bits,
                         gconfig.ldpc_uconfig.punctured_bits, kCbPerSymbol,
                         local_streams[symbol_id]);
    } else if (task_id == 3) {
      ldpc_decoder(local_equalize[local_symid], local_decode[local_symid],
                   local_streams[symbol_id]);
      Scrambler::scrambler(local_decode[local_symid], local_decode[local_symid],
                           local_streams[symbol_id]);
    } else if (task_id == 4) {
#ifndef NOCOPY
      cudaMemcpyAsync(local_txmac[local_symid].ptr(),
                      local_decode[local_symid].ptr(),
                      local_decode[local_symid].szBytes(),
                      cudaMemcpyDeviceToHost, local_streams[symbol_id]);
#endif
      if (symbol_id == gconfig.symbols - 1) {
        for (uint32_t i = 0; i < gconfig.symbols; i++) {
          if (gconfig.frame.gidx_sym[i] <= Config::Pilot) continue;
          cudaStreamSynchronize(local_streams[i]);
        }
      }
    }
  } else if (gconfig.frame.gidx_sym[symbol_id] == Config::Downlink) {
    if (task_id == 0) {
#ifndef NOCOPY
      cudaMemcpyAsync(local_uncode[local_symid].ptr(),
                      local_rxmac[local_symid].ptr(),
                      local_rxmac[local_symid].szBytes(),
                      cudaMemcpyHostToDevice, local_streams[symbol_id]);
#endif
    } else if (task_id == 1) {
      Scrambler::scrambler(local_uncode[local_symid], local_uncode[local_symid],
                           local_streams[symbol_id]);
      ldpc_encoder(local_uncode[local_symid], local_encode[local_symid],
                   local_streams[symbol_id]);
    } else if (task_id == 2) {
      Modulation::modulate(local_encode[local_symid],
                           local_modulate[local_symid],
                           local_streams[symbol_id]);
    } else if (task_id == 3) {
      cudaStreamSynchronize(local_streams[last_pilot]);
      Precode::precode(local_nmcsi, local_modulate[local_symid],
                       local_precode[local_symid], gconfig.sc_group,
                       gconfig.ofdm_start, local_streams[symbol_id]);
    } else if (task_id == 4) {
      downlink_ifft[local_symid](local_precode[local_symid],
                                 local_ifft[local_symid],
                                 local_streams[symbol_id]);
    } else if (task_id == 5) {
#ifndef NOCOPY
      cudaMemcpyAsync(local_hsend[local_symid].ptr(),
                      local_ifft[local_symid].ptr(),
                      local_ifft[local_symid].szBytes(), cudaMemcpyDeviceToHost,
                      local_streams[symbol_id]);
#endif
      if (symbol_id == gconfig.symbols - 1) {
        for (uint32_t i = 0; i < gconfig.symbols; i++) {
          if (gconfig.frame.gidx_sym[i] <= Config::Pilot) continue;
          cudaStreamSynchronize(local_streams[i]);
        }
      }
    }
  } else {
    throw std::runtime_error("Can't Process now. Symbol type not defined");
  }
}

void BandProcess::noprocess(uint32_t frame_cid, uint32_t symbol_id) {
  auto &local_streams = BandBuffers::Streams[frame_cid];

  auto &local_hrecv = BandBuffers::HostBufferRecv[frame_cid];
  auto &local_recv = BandBuffers::DeviceBufferRecv[frame_cid];
  auto &local_ufft = BandBuffers::DeviceBufferUFFT[frame_cid];
  auto &local_csi = BandBuffers::DeviceBufferCsi[frame_cid];
  auto &local_nmcsi = BandBuffers::DeviceBufferNmCsi[frame_cid];
  auto &local_equalize = BandBuffers::DeviceBufferEqualize[frame_cid];
  auto &local_decode = BandBuffers::DeviceBufferDecode[frame_cid];
  auto &local_txmac = BandBuffers::HostBufferMacSend[frame_cid];
  auto &local_rxmac = BandBuffers::HostBufferMacRecv[frame_cid];
  auto &local_uncode = BandBuffers::DeviceBufferUncode[frame_cid];
  auto &local_encode = BandBuffers::DeviceBufferEncode[frame_cid];
  auto &local_modulate = BandBuffers::DeviceBufferModulate[frame_cid];
  auto &local_precode = BandBuffers::DeviceBufferPrecode[frame_cid];
  auto &local_ifft = BandBuffers::DeviceBufferIFFT[frame_cid];
  auto &local_hsend = BandBuffers::HostBufferSend[frame_cid];

  auto &uplink_fft = ufft[frame_cid];
  auto &pilot_fft = pfft[frame_cid];
  auto &beamform = beam[frame_cid];
  auto &ldpc_decoder = decoder[frame_cid];
  auto &ldpc_encoder = encoder[frame_cid];
  auto &downlink_ifft = difft[frame_cid];

  uint32_t local_symid = gconfig.frame.gidx_lidx[symbol_id];
  if (gconfig.frame.gidx_sym[symbol_id] <= Config::Pilot) {
    cudaMemcpyAsync(local_recv[symbol_id].ptr(), local_hrecv[symbol_id].ptr(),
                    local_hrecv[symbol_id].szBytes(), cudaMemcpyHostToDevice,
                    local_streams[symbol_id]);
    pilot_fft[local_symid](local_recv[symbol_id], local_csi,
                           local_streams[symbol_id]);
    if (local_symid == gconfig.frame.pilot_syms - 1) {
      for (uint32_t i = 0; i < symbol_id; i++) {
        if (gconfig.frame.gidx_sym[i] > Config::Pilot) continue;
        cudaStreamSynchronize(local_streams[i]);
      }
      beamform.beamformer(local_csi, local_nmcsi, local_streams[symbol_id]);
      cudaStreamSynchronize(local_streams[symbol_id]);
    }
  } else if (gconfig.frame.gidx_sym[symbol_id] == Config::Uplink) {
    cudaMemcpyAsync(local_recv[symbol_id].ptr(), local_hrecv[symbol_id].ptr(),
                    local_hrecv[symbol_id].szBytes(), cudaMemcpyHostToDevice,
                    local_streams[symbol_id]);
    uplink_fft[local_symid](local_recv[symbol_id], local_ufft[local_symid],
                            local_streams[symbol_id]);
    Equalize::equalize(local_csi, local_ufft[local_symid],
                       local_equalize[local_symid], gconfig.sc_group,
                       gconfig.mod_order, gconfig.ldpc_uconfig.encoded_bits,
                       gconfig.ldpc_uconfig.punctured_bits, kCbPerSymbol,
                       local_streams[symbol_id]);
    ldpc_decoder(local_equalize[local_symid], local_decode[local_symid],
                 local_streams[symbol_id]);
    Scrambler::scrambler(local_decode[local_symid], local_decode[local_symid],
                         local_streams[symbol_id]);
    cudaMemcpyAsync(local_txmac[local_symid].ptr(),
                    local_decode[local_symid].ptr(),
                    local_decode[local_symid].szBytes(), cudaMemcpyDeviceToHost,
                    local_streams[symbol_id]);
  } else if (gconfig.frame.gidx_sym[symbol_id] == Config::Downlink) {
    cudaMemcpyAsync(local_uncode[local_symid].ptr(),
                    local_rxmac[local_symid].ptr(),
                    local_rxmac[local_symid].szBytes(), cudaMemcpyHostToDevice,
                    local_streams[symbol_id]);
    Scrambler::scrambler(local_uncode[local_symid], local_uncode[local_symid],
                         local_streams[symbol_id]);
    ldpc_encoder(local_uncode[local_symid], local_encode[local_symid],
                 local_streams[symbol_id]);
    Modulation::modulate(local_encode[local_symid], local_modulate[local_symid],
                         local_streams[symbol_id]);
    Precode::precode(local_nmcsi, local_modulate[local_symid],
                     local_precode[local_symid], gconfig.sc_group,
                     gconfig.ofdm_start, local_streams[symbol_id]);
    downlink_ifft[local_symid](local_precode[local_symid],
                               local_ifft[local_symid],
                               local_streams[symbol_id]);
    cudaMemcpyAsync(local_hsend[local_symid].ptr(),
                    local_ifft[local_symid].ptr(),
                    local_ifft[local_symid].szBytes(), cudaMemcpyDeviceToHost,
                    local_streams[symbol_id]);
  } else {
    throw std::runtime_error("Can't Process now. Symbol type not defined");
  }
done:
  if (symbol_id == gconfig.symbols - 1) {
    for (uint32_t i = 0; i < gconfig.symbols; i++) {
      if (gconfig.frame.gidx_sym[i] <= Config::Pilot) continue;
      cudaStreamSynchronize(local_streams[i]);
    }
  }
}