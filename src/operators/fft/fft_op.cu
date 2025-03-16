/**
 * @file fft_op.cu
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Wrapper for FFT operations
 * @version 0.1
 * @date 2023-12-10
 *
 * @copyright Copyright (c) 2023
 *
 */
#include <algorithm>

#include "fft_callback.cuh"
#include "fft_op.h"
#include "ifft_callback.cuh"
#include "utils.h"

using namespace mega;

std::vector<Complex *> PilotSign::pilot_sign_ptrs;

template <typename QuanT>
void fft_loader(const QuanT *in, Complex *out, struct baseInfo fftInfo,
                cudaStream_t stream = nullptr) {
  const uint32_t num_element = fftInfo.ofdmCAnum;
  const uint32_t batch_count = fftInfo.bsAnt;

  uint32_t num_threads =
      num_element < 1024 ? ((num_element / 32 + 1) * 32) : 1024;
  dim3 block(num_threads, 1, 1);
  dim3 grid(batch_count % 65536, 1, 1);

  fft_load_kernel<<<grid, block, 0, stream>>>(in, out, fftInfo);
}

void fft_uplink_storer(const Complex *in, Complex *out, struct baseInfo fftInfo,
                       cudaStream_t stream = nullptr) {
  const uint32_t num_element = fftInfo.ofdmNum;
  const uint32_t batch_count = fftInfo.bsAnt;

  uint32_t num_threads =
      num_element < 1024 ? ((num_element / 32 + 1) * 32) : 1024;
  dim3 block(num_threads, 1, 1);
  dim3 grid(batch_count % 65536, 1, 1);

  fft_store_uplink_kernel<<<grid, block, 0, stream>>>(in, out, fftInfo);
}

void fft_pilot_storer(const Complex *in, Complex *out, struct pilotInfo fftInfo,
                      cudaStream_t stream = nullptr) {
  const uint32_t num_element = fftInfo.ofdmNum;
  const uint32_t batch_count = fftInfo.bsAnt;

  uint32_t num_threads =
      num_element < 1024 ? ((num_element / 32 + 1) * 32) : 1024;
  dim3 block(num_threads, 1, 1);
  dim3 grid(batch_count % 65536, 1, 1);

  fft_store_pilot_kernel<<<grid, block, 0, stream>>>(in, out, fftInfo);
}

template <typename QuanT>
void ifft_downlink_storer(const Complex *in, QuanT *out,
                          struct baseInfo ifftInfo,
                          cudaStream_t stream = nullptr) {
  const uint32_t num_element = ifftInfo.ofdmCAnum;
  const uint32_t batch_count = ifftInfo.bsAnt;

  uint32_t num_threads =
      num_element < 1024 ? ((num_element / 32 + 1) * 32) : 1024;
  dim3 block(num_threads, 1, 1);
  dim3 grid(batch_count % 65536, 1, 1);

  ifft_store_downlink_kernel<<<grid, block, 0, stream>>>(in, out, ifftInfo);
}

template <typename QuanT>
UplinkFFT<QuanT>::UplinkFFT(uint32_t ofdm_start, uint32_t ofdm_ca,
                            uint32_t ofdm_num, uint32_t bs_num) {
  baseInfo uplink_info = {
      .ofdmStart = ofdm_start,
      .ofdmNum = ofdm_num,
      .ofdmCAnum = ofdm_ca,
      .bsAnt = bs_num,
  };
  static_cast<baseInfo &>(hinfo) = uplink_info;

  cufftCreate(&plan);
  cufftPlan1d(&plan, ofdm_ca, CUFFT_C2C, bs_num);

  cudaMalloc(&work_area, sizeof(Complex) * ofdm_ca * bs_num);
}

template <typename QuanT>
PilotFFT<QuanT>::PilotFFT(uint32_t ofdm_start, uint32_t ofdm_ca,
                          uint32_t ofdm_num, uint32_t bs_num, uint32_t ue_num,
                          uint32_t sc_group) {
  pilotInfo pilot_info = {
      {
          .ofdmStart = ofdm_start,
          .ofdmNum = ofdm_num,
          .ofdmCAnum = ofdm_ca,
          .bsAnt = bs_num,
      },
      ue_num,
      sc_group,
      0,
      PilotSign::pilot_sign_ptrs[get_device_id()],
  };
  hinfo = pilot_info;

  cufftCreate(&plan);
  cufftPlan1d(&plan, ofdm_ca, CUFFT_C2C, bs_num);

  cudaMalloc(&work_area, sizeof(Complex) * ofdm_ca * bs_num);
}

template <typename QuanT>
DownlinkIFFT<QuanT>::DownlinkIFFT(uint32_t ofdm_start, uint32_t ofdm_ca,
                                  uint32_t ofdm_num, uint32_t bs_num) {
  baseInfo downlink_info = {
      .ofdmStart = ofdm_start,
      .ofdmNum = ofdm_num,
      .ofdmCAnum = ofdm_ca,
      .bsAnt = bs_num,
  };
  static_cast<baseInfo &>(hinfo) = downlink_info;

  cufftCreate(&plan);
  cufftPlan1d(&plan, ofdm_ca, CUFFT_C2C, bs_num);

  cudaMalloc(&work_area, sizeof(Complex) * ofdm_ca * bs_num);
}

template <typename QuanT>
void UplinkFFT<QuanT>::operator()(const Matrix &in, const Matrix &out,
                                  cudaStream_t stream) {
  cufftSetStream(plan, stream);
  fft_loader(in.ptr<QuanT>(), work_area, hinfo, stream);
  cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(work_area),
               reinterpret_cast<cufftComplex *>(work_area), CUFFT_FORWARD);
  fft_uplink_storer(work_area, out.ptr<Complex>(), hinfo, stream);
}

template <typename QuanT>
void PilotFFT<QuanT>::operator()(const Matrix &in, const Matrix &out,
                                 cudaStream_t stream) {
  cufftSetStream(plan, stream);
  fft_loader(in.ptr<QuanT>(), work_area, hinfo, stream);
  cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(work_area),
               reinterpret_cast<cufftComplex *>(work_area), CUFFT_FORWARD);
  fft_pilot_storer(work_area, out.ptr<Complex>(), hinfo, stream);
}

template <typename QuanT>
void PilotFFT<QuanT>::operator()(const Matrix &in, const Matrix &out,
                                 uint32_t ue_start, cudaStream_t stream) {
  cufftSetStream(plan, stream);
  pilotInfo hinfo_tmp = hinfo;
  hinfo_tmp.ueStart = ue_start;
  fft_loader(in.ptr<QuanT>(), work_area, hinfo_tmp, stream);
  cufftExecC2C(plan, reinterpret_cast<cufftComplex *>(work_area),
               reinterpret_cast<cufftComplex *>(work_area), CUFFT_FORWARD);
  fft_pilot_storer(work_area, out.ptr<Complex>(), hinfo_tmp, stream);
}

template <typename QuanT>
void DownlinkIFFT<QuanT>::operator()(const Matrix &in, const Matrix &out,
                                     cudaStream_t stream) {
  cufftSetStream(plan, stream);
  cufftExecC2C(plan, in.ptr<cufftComplex>(),
               reinterpret_cast<cufftComplex *>(work_area), CUFFT_INVERSE);
  ifft_downlink_storer(work_area, out.ptr<QuanT>(), hinfo, stream);
}

void PilotSign::init(uint32_t ofdm_num, uint8_t num_device) {
  PilotSign::pilot_sign_ptrs.resize(num_device);
  std::fill(PilotSign::pilot_sign_ptrs.begin(),
            PilotSign::pilot_sign_ptrs.end(), nullptr);

  Matrix pilot_sign_table = zadoff_chu_sequence(ofdm_num);
  cyclic_shift(pilot_sign_table, pilot_sign_table, M_PI / 4);
  Complex *pilot_sign_cpu = pilot_sign_table.ptr<Complex>();
  for (uint32_t i = 0; i < ofdm_num; i++) {
    pilot_sign_cpu[i] /= pilot_sign_cpu[i].pow2();
  }

  for (uint8_t i = 0; i < num_device; i++) {
    cudaSetDevice(i);
    PilotSign::pilot_sign_ptrs[i] = seq_to_gpu(pilot_sign_table);
  }
}

template class mega::PilotFFT<short>;
template class mega::UplinkFFT<short>;
template class mega::DownlinkIFFT<short>;

template class mega::PilotFFT<char>;
template class mega::UplinkFFT<char>;
template class mega::DownlinkIFFT<char>;