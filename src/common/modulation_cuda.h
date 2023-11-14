#pragma once

//#include <math.h>
#include <cuda_fp16.h>

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