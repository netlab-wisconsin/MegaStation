#pragma once

//#include <math.h>
#include <cuda_fp16.h>
#include <stdio.h>

struct myComplex {
  float re;
  float im;

  __device__ __host__
  myComplex(float re, float im) : re(re), im(im) {}
  __device__ __host__
  myComplex() : re(0), im(0) {}
  __device__ __host__
  inline myComplex operator+(const myComplex &other) const {
    return myComplex(re + other.re, im + other.im);
  }
  __device__ __host__
  inline myComplex operator*(const myComplex &other) const {
    return myComplex(re * other.re - im * other.im, re * other.im + im * other.re);
  }
  __device__ __host__
  inline myComplex operator-(const myComplex &other) const {
    return myComplex(re - other.re, im - other.im);
  }
  __device__ __host__
  inline myComplex operator-() const {
    return myComplex(-re, -im);
  }
  __device__ __host__
  inline myComplex operator+=(const myComplex &other) {
    re += other.re;
    im += other.im;
    return *this;
  }
  __device__ __host__
  inline myComplex operator-=(const myComplex &other) {
    re -= other.re;
    im -= other.im;
    return *this;
  }
  __device__ __host__
  inline myComplex operator*=(const myComplex &other) {
    float tmp = re;
    re = re * other.re - im * other.im;
    im = tmp * other.im + im * other.re;
    return *this;
  }
  __device__ __host__
  inline bool operator==(const myComplex &other) const {
    return re == other.re && im == other.im;
  }
  __device__ __host__
  inline bool operator!=(const myComplex &other) const {
    return re != other.re || im != other.im;
  }
  __device__ __host__
  inline myComplex conj() const {
    return myComplex(re, -im);
  }
  __device__ __host__
  inline float abs() const {
    return sqrtf(re * re + im * im);
  }
};


#define SCALE_BYTE_CONV_QPSK_CU 20
#define SCALE_BYTE_CONV_QAM16_CU 100
#define SCALE_BYTE_CONV_QAM64_CU 100
#define SCALE_BYTE_CONV_QAM256_CU 100


__device__ __host__
inline void demodQPSK(const myComplex &c, half llr[]) {
  llr[0] = half(c.re * -SCALE_BYTE_CONV_QPSK_CU * sqrtf(2.f));
  llr[1] = half(c.im * -SCALE_BYTE_CONV_QPSK_CU * sqrtf(2.f));
}

__device__ __host__
inline void demod16QAM(const myComplex &c, half llr[]) {
  llr[0] = half(SCALE_BYTE_CONV_QAM16_CU * c.re);
  llr[1] = half(SCALE_BYTE_CONV_QAM16_CU * c.im);
  llr[2] = half(2 * SCALE_BYTE_CONV_QAM16_CU / sqrtf(10.f)) - __habs(llr[0]);
  llr[3] = half(2 * SCALE_BYTE_CONV_QAM16_CU / sqrtf(10.f)) - __habs(llr[1]);
}

__device__ __host__
inline void demod64QAM(const myComplex &c, half llr[]) {
  const half t1 = half(4 * SCALE_BYTE_CONV_QAM64_CU / sqrtf(42.f));
  const half t2 = half(2 * SCALE_BYTE_CONV_QAM64_CU / sqrtf(42.f));

  llr[0] = half(SCALE_BYTE_CONV_QAM64_CU * c.re);
  llr[1] = half(SCALE_BYTE_CONV_QAM64_CU * c.im);
  llr[2] = t1 - __habs(llr[0]);
  llr[3] = t1 - __habs(llr[1]);
  llr[4] = t2 - __habs(llr[2]);
  llr[5] = t2 - __habs(llr[3]);
}

__device__ __host__
inline void demod256QAM(const myComplex &c, half llr[]) {
  const half t1 = half(8 * SCALE_BYTE_CONV_QAM256_CU / sqrtf(170.f));
  const half t2 = half(4 * SCALE_BYTE_CONV_QAM256_CU / sqrtf(170.f));
  const half t3 = half(2 * SCALE_BYTE_CONV_QAM256_CU / sqrtf(170.f));

  llr[0] = half(SCALE_BYTE_CONV_QAM256_CU * c.re);
  llr[1] = half(SCALE_BYTE_CONV_QAM256_CU * c.im);
  llr[2] = t1 - __habs(llr[0]);
  llr[3] = t1 - __habs(llr[1]);
  llr[4] = t2 - __habs(llr[2]);
  llr[5] = t2 - __habs(llr[3]);
  llr[6] = t3 - __habs(llr[4]);
  llr[7] = t3 - __habs(llr[5]);
}

typedef void (*demodPtr)(const myComplex &, half[]);
// static const demodPtr kDemodPtrs[] = {demodQPSK, demod16QAM, demod64QAM, demod256QAM};

__constant__ float2 modTable[256];

__global__ void modulateKernel(const uint8_t *input, myComplex *output,
  const myComplex *pilot_table, int pilot_spacing,
  uint8_t mod, size_t in_bytes, size_t out_bytes, size_t batch_count) {
  const myComplex *modTable_ptr = (myComplex *)modTable;
  for (int batch_idx = blockIdx.z; batch_idx < batch_count; batch_idx += gridDim.z) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    size_t idx_shrink = idx - ((idx / pilot_spacing) + 1);

    const uint8_t *input_data_ptr = input + batch_idx * in_bytes;
    myComplex *output_data_ptr = output + batch_idx;

    uint8_t mod_byte = 0;

    for(int i = 0, mod_idx = idx_shrink * mod;
      i < mod && (mod_idx / 8) < in_bytes;
      i++, mod_idx++) {
      mod_byte = ((input_data_ptr[mod_idx / 8] >> (mod_idx % 8)) & 0x1) | (mod_byte << 1);
    }

    if (idx < out_bytes) {
      output_data_ptr[idx * batch_count] = ((idx % pilot_spacing) == 0) ?
        pilot_table[batch_idx * out_bytes + idx] : modTable_ptr[mod_byte];
    }
  }
}

__host__ void init_modulation_table(myComplex *modulation, size_t sz, cudaStream_t stream) {
  cudaMemcpyToSymbol(modTable, modulation, sz);
}