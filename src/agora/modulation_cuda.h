#pragma once

#include <math.h>

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
inline void demodQPSK(const myComplex &c, signed char llr[]) {
  llr[0] = (signed char)(c.re * -SCALE_BYTE_CONV_QPSK_CU * sqrtf(2.f));
  llr[1] = (signed char)(c.im * -SCALE_BYTE_CONV_QPSK_CU * sqrtf(2.f));
}

__device__ __host__
inline void demod16QAM(const myComplex &c, signed char llr[]) {
  llr[0] = (signed char)(SCALE_BYTE_CONV_QAM16_CU * c.re);
  llr[1] = (signed char)(SCALE_BYTE_CONV_QAM16_CU * c.im);
  llr[2] = 2 * SCALE_BYTE_CONV_QAM16_CU / sqrtf(10.f) - abs(llr[0]);
  llr[3] = 2 * SCALE_BYTE_CONV_QAM16_CU / sqrtf(10.f) - abs(llr[1]);
}

__device__ __host__
inline void demod64QAM(const myComplex &c, signed char llr[]) {
  const signed char t1 = 4 * SCALE_BYTE_CONV_QAM64_CU / sqrtf(42.f);
  const signed char t2 = 2 * SCALE_BYTE_CONV_QAM64_CU / sqrtf(42.f);

  llr[0] = (signed char)lrintf(SCALE_BYTE_CONV_QAM64_CU * c.re);
  llr[1] = (signed char)lrintf(SCALE_BYTE_CONV_QAM64_CU * c.im);
  llr[2] = t1 - abs(llr[0]);
  llr[3] = t1 - abs(llr[1]);
  llr[4] = t2 - abs(llr[2]);
  llr[5] = t2 - abs(llr[3]);
}

__device__ __host__
inline void demod256QAM(const myComplex &c, signed char llr[]) {
  const unsigned char t1 = 8 * SCALE_BYTE_CONV_QAM256_CU / sqrtf(170.f);
  const unsigned char t2 = 4 * SCALE_BYTE_CONV_QAM256_CU / sqrtf(170.f);
  const unsigned char t3 = 2 * SCALE_BYTE_CONV_QAM256_CU / sqrtf(170.f);

  llr[0] = (signed char)(SCALE_BYTE_CONV_QAM256_CU * c.re);
  llr[1] = (signed char)(SCALE_BYTE_CONV_QAM256_CU * c.im);
  llr[2] = t1 - abs(llr[0]);
  llr[3] = t1 - abs(llr[1]);
  llr[4] = t2 - abs(llr[2]);
  llr[5] = t2 - abs(llr[3]);
  llr[6] = t3 - abs(llr[4]);
  llr[7] = t3 - abs(llr[5]);
}

typedef void (*demodPtr)(const myComplex &, signed char[]);
// static const demodPtr kDemodPtrs[] = {demodQPSK, demod16QAM, demod64QAM, demod256QAM};