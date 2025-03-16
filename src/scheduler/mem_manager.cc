#include "mem_manager.h"

#include "beamform/beamform.h"
#include "fft/fft_op.h"
#include "ldpc/decoder.h"
#include "ldpc/encoder.h"
#include "matrix/cuphy_tensor.h"
#include "matrix/matrix.h"
#include "mega_complex.h"
#include "modulation/modulation.h"
#include "scrambler/scrambler.h"
#include "utils/consts.h"
#include "utils/types.h"

using namespace mega;

std::unique_ptr<RefTable> MemManager::ref_table;
std::array<MemWrapper, MemOpTypeProp::kNumMemType> MemManager::mem_warp;

void MemManager::init(const uint32_t vframe_counts) {
  ref_table = std::make_unique<RefTable>(vframe_counts);

  const uint64_t &ofdm_cnum = gconfig.ofdm_ca;
  const uint64_t &ofdm_dnum = gconfig.ofdm_data;
  const uint32_t &ant_num = gconfig.antennas;
  const uint32_t &ue_num = gconfig.users;
  const uint32_t &sc_group = gconfig.sc_group;
  const uint32_t scg_count = ofdm_dnum / sc_group;

  get_mem_wrapper(MemOpType::kPreFFT) =
      Matrix(2 * sizeof(QuanT), ofdm_cnum, ant_num, Matrix::kDesc);
  get_mem_wrapper(MemOpType::kCSI) =
      Matrix(sizeof(Complex), ue_num, ant_num, scg_count, Matrix::kDesc);
  get_mem_wrapper(MemOpType::kNmCSI) =
      Matrix(sizeof(Complex), ant_num, ue_num, scg_count, Matrix::kDesc);
  get_mem_wrapper(MemOpType::kUFFT) =
      Matrix(sizeof(Complex), ofdm_dnum, ant_num, Matrix::kDesc);
  {
    const uint64_t &encoded_bits = gconfig.ldpc_uconfig.encoded_bits;
    const uint64_t &punctured_bits = gconfig.ldpc_uconfig.punctured_bits;
    const uint64_t &decoded_bits = gconfig.ldpc_uconfig.decoded_bits;
    get_mem_wrapper(MemOpType::kEqual) = CuphyTensor(
        CuphyTensor::kHalf, encoded_bits + punctured_bits,
        ue_num * kCbPerSymbol, Matrix::kDesc, CuphyTensor::kCoalesce);
    get_mem_wrapper(MemOpType::kDecode) =
        CuphyTensor(CuphyTensor::kBit, decoded_bits, ue_num * kCbPerSymbol,
                    Matrix::kDesc, CuphyTensor::kCoalesce);
  }
  {
    const uint64_t &encoded_bits = gconfig.ldpc_dconfig.encoded_bits;
    const uint64_t &decoded_bits = gconfig.ldpc_dconfig.decoded_bits;
    get_mem_wrapper(MemOpType::kUncode) = CuphyTensor(
        CuphyTensor::kBit, decoded_bits, ue_num * kCbPerSymbol, Matrix::kDesc);
    get_mem_wrapper(MemOpType::kEncode) = CuphyTensor(
        CuphyTensor::kBit, encoded_bits, ue_num * kCbPerSymbol, Matrix::kDesc);
  }
  get_mem_wrapper(MemOpType::kModulate) =
      Matrix(sizeof(Complex), ofdm_dnum, ue_num, Matrix::kDesc);
  get_mem_wrapper(MemOpType::kPrecode) =
      Matrix(sizeof(Complex), ofdm_cnum, ant_num, Matrix::kDesc);
  get_mem_wrapper(MemOpType::kDiFFT) =
      Matrix(2 * sizeof(QuanT), ofdm_cnum, ant_num, Matrix::kDevice);

  PilotSign::init(ofdm_dnum, gconfig.num_devices);
  Scrambler::init(gconfig.num_devices);
  Modulation::init(gconfig.mod_order, ofdm_dnum, ue_num, gconfig.pilot_spacing,
                   gconfig.num_devices);
  DecoderFactory::init(CuphyTensor::kHalf, gconfig.ldpc_uconfig,
                       gconfig.num_devices);
  EncoderFactory::init(gconfig.ldpc_dconfig);
}

template <class T, class... Args>
inline std::shared_ptr<void> get_void_ptr(Args... args) {
  T *func = new T(args...);
  void *func_ptr = reinterpret_cast<void *>(func);
  return std::shared_ptr<void>(
      func_ptr, [](void *ptr) { delete reinterpret_cast<T *>(ptr); });
}

MemManager::MemManager(const uint32_t vframe_counts)
    : free_memop_list(MemOpTypeProp::kNumMemOpType), mem_table(vframe_counts) {
  const uint64_t ofdm_cnum = gconfig.ofdm_ca;
  const uint64_t ofdm_dnum = gconfig.ofdm_data;
  const uint32_t ant_num = gconfig.antennas;
  const uint32_t ue_num = gconfig.users;
  const uint32_t sc_group = gconfig.sc_group;
  const uint32_t scg_count = ofdm_dnum / sc_group;
  const uint32_t pilot_syms = gconfig.frame.pilot_syms;
  const uint32_t uplink_syms = gconfig.frame.uplink_syms;
  const uint32_t downlink_syms = gconfig.frame.downlink_syms;

  uint64_t total_mem_per_frame = 0;
  uint64_t total, free, before_alloc;
  available_frames = 0;
  cudaMemGetInfo(&before_alloc, &total);
  // TODO: Do module loading during first time
  do {
    available_frames++;

    Matrix d_recv_buffer(2 * sizeof(QuanT), ofdm_cnum, ant_num,
                         uplink_syms + pilot_syms, Matrix::kDevice);
    all_memop_list.push_back(d_recv_buffer.raw());

    for (uint32_t i = 0; i < pilot_syms + uplink_syms; i++)
      get_free_list(MemOpType::kPreFFT).enqueue(d_recv_buffer[i].ptr());

    Matrix d_csi_buffer(sizeof(Complex), ue_num, ant_num, scg_count,
                        Matrix::kDevice);
    all_memop_list.push_back(d_csi_buffer.raw());

    get_free_list(MemOpType::kCSI).enqueue(d_csi_buffer.ptr());

    if (downlink_syms > 0) {
      Matrix d_nmcsi_buffer(sizeof(Complex), ant_num, ue_num, scg_count,
                            Matrix::kDevice);
      all_memop_list.push_back(d_nmcsi_buffer.raw());

      get_free_list(MemOpType::kNmCSI).enqueue(d_nmcsi_buffer.ptr());
    }

    if (uplink_syms > 0) {
      Matrix d_ufft_buffer(sizeof(Complex), ofdm_dnum, ant_num, uplink_syms,
                           Matrix::kDevice);
      all_memop_list.push_back(d_ufft_buffer.raw());

      for (uint32_t i = 0; i < uplink_syms; i++)
        get_free_list(MemOpType::kUFFT).enqueue(d_ufft_buffer[i].ptr());

      const uint64_t &encoded_bits = gconfig.ldpc_uconfig.encoded_bits;
      const uint64_t &punctured_bits = gconfig.ldpc_uconfig.punctured_bits;
      const uint64_t &decoded_bits = gconfig.ldpc_uconfig.decoded_bits;

      CuphyTensor d_equalize_buffer(CuphyTensor::kHalf,
                                    encoded_bits + punctured_bits,
                                    ue_num * kCbPerSymbol, uplink_syms,
                                    Matrix::kDevice, CuphyTensor::kCoalesce);
      all_memop_list.push_back(d_equalize_buffer.raw());

      for (uint32_t i = 0; i < uplink_syms; i++)
        get_free_list(MemOpType::kEqual).enqueue(d_equalize_buffer[i].ptr());

      CuphyTensor d_decode_buffer(CuphyTensor::kBit, decoded_bits,
                                  ue_num * kCbPerSymbol, uplink_syms,
                                  Matrix::kDevice, CuphyTensor::kCoalesce);
      all_memop_list.push_back(d_decode_buffer.raw());

      for (uint32_t i = 0; i < uplink_syms; i++)
        get_free_list(MemOpType::kDecode).enqueue(d_decode_buffer[i].ptr());
    }
    if (downlink_syms > 0) {
      const uint64_t &encoded_bits = gconfig.ldpc_dconfig.encoded_bits;
      const uint64_t &decoded_bits = gconfig.ldpc_dconfig.decoded_bits;
      CuphyTensor d_uncode_buffer(CuphyTensor::kBit, decoded_bits,
                                  ue_num * kCbPerSymbol, downlink_syms,
                                  Matrix::kDevice);
      all_memop_list.push_back(d_uncode_buffer.raw());

      for (uint32_t i = 0; i < downlink_syms; i++)
        get_free_list(MemOpType::kUncode).enqueue(d_uncode_buffer[i].ptr());

      CuphyTensor d_encode_buffer(CuphyTensor::kBit, encoded_bits,
                                  ue_num * kCbPerSymbol, downlink_syms,
                                  Matrix::kDevice);
      all_memop_list.push_back(d_encode_buffer.raw());

      for (uint32_t i = 0; i < downlink_syms; i++)
        get_free_list(MemOpType::kEncode).enqueue(d_encode_buffer[i].ptr());

      Matrix d_modulate_buffer(sizeof(Complex), ofdm_dnum, ue_num,
                               downlink_syms, Matrix::kDevice);
      all_memop_list.push_back(d_modulate_buffer.raw());

      for (uint32_t i = 0; i < downlink_syms; i++)
        get_free_list(MemOpType::kModulate).enqueue(d_modulate_buffer[i].ptr());

      Matrix d_precode_buffer(sizeof(Complex), ofdm_cnum, ant_num,
                              downlink_syms, Matrix::kDevice);
      all_memop_list.push_back(d_precode_buffer.raw());

      for (uint32_t i = 0; i < downlink_syms; i++)
        get_free_list(MemOpType::kPrecode).enqueue(d_precode_buffer[i].ptr());

      Matrix d_ifft_buffer(2 * sizeof(QuanT), ofdm_cnum, ant_num, downlink_syms,
                           Matrix::kDevice);
      all_memop_list.push_back(d_ifft_buffer.raw());

      for (uint32_t i = 0; i < downlink_syms; i++)
        get_free_list(MemOpType::kDiFFT).enqueue(d_ifft_buffer[i].ptr());
    }

    std::shared_ptr<void> shared_op;
    const uint64_t &ofdm_start = gconfig.ofdm_start;
    for (uint32_t i = 0; i < pilot_syms; i++) {
      shared_op = get_void_ptr<PilotFFT<QuanT>>(
          ofdm_start, ofdm_cnum, ofdm_dnum, ant_num, ue_num, sc_group);
      all_memop_list.push_back(shared_op);

      get_free_list(MemOpType::kPFFTOp).enqueue(shared_op.get());
    }

    shared_op = get_void_ptr<Beamform>(ant_num, ue_num, scg_count);
    all_memop_list.push_back(shared_op);

    get_free_list(MemOpType::kBeamOp).enqueue(shared_op.get());

    for (uint32_t i = 0; i < uplink_syms; i++) {
      shared_op = get_void_ptr<UplinkFFT<QuanT>>(ofdm_start, ofdm_cnum,
                                                 ofdm_dnum, ant_num);
      all_memop_list.push_back(shared_op);

      get_free_list(MemOpType::kUFFTOp).enqueue(shared_op.get());
    }
    for (uint32_t i = 0; i < downlink_syms; i++) {
      shared_op = get_void_ptr<DownlinkIFFT<QuanT>>(ofdm_start, ofdm_cnum,
                                                    ofdm_dnum, ant_num);
      all_memop_list.push_back(shared_op);

      get_free_list(MemOpType::kDiFFTOp).enqueue(shared_op.get());
    }

    cudaMemGetInfo(&free, &total);
    if (total_mem_per_frame == 0) {
      total_mem_per_frame = before_alloc - free;
    }
  } while (free > total_mem_per_frame && available_frames < vframe_counts);

  int device_id;
  cudaGetDevice(&device_id);
  spdlog::info("Device {} Allocated {} frames with {} MB", device_id,
               available_frames, total_mem_per_frame / 1024 / 1024);
}