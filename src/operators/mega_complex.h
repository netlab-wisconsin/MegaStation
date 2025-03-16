/**
 * @file mega_complex.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Complex number class
 * @version 0.1
 * @date 2023-11-26
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>

namespace mega {

/**
 * @brief Complex number class
 *
 */
struct Complex {
  float re;  //!< Real part
  float im;  //!< Imaginary part

  __device__ __host__ Complex(float re, float im) : re(re), im(im) {}
  __device__ __host__ Complex() : re(0), im(0) {}
  __device__ __host__ Complex(float2 &val) : re(val.x), im(val.y) {}
  __device__ __host__ inline Complex operator+(const Complex &other) const {
    return Complex(re + other.re, im + other.im);
  }
  __device__ __host__ inline Complex operator*(const Complex &other) const {
    return Complex(re * other.re - im * other.im,
                   re * other.im + im * other.re);
  }
  __device__ __host__ inline Complex operator-(const Complex &other) const {
    return Complex(re - other.re, im - other.im);
  }
  __device__ __host__ inline Complex operator-() const {
    return Complex(-re, -im);
  }
  __device__ __host__ inline Complex operator+=(const Complex &other) {
    re += other.re;
    im += other.im;
    return *this;
  }
  __device__ __host__ inline Complex operator-=(const Complex &other) {
    re -= other.re;
    im -= other.im;
    return *this;
  }
  __device__ __host__ inline Complex operator*=(const Complex &other) {
    float tmp = re;
    re = re * other.re - im * other.im;
    im = tmp * other.im + im * other.re;
    return *this;
  }
  __device__ __host__ inline bool operator==(const Complex &other) const {
    return re == other.re && im == other.im;
  }
  __device__ __host__ inline bool operator!=(const Complex &other) const {
    return re != other.re || im != other.im;
  }
  __device__ __host__ inline Complex conj() const { return Complex(re, -im); }
  __device__ __host__ inline float abs() const {
    return sqrtf(re * re + im * im);
  }
  __device__ __host__ inline float pow2() const { return re * re + im * im; }
  __device__ __host__ inline operator float2() const {
    return make_float2(re, im);
  }
  __device__ __host__ inline Complex operator/(const float &denom) const {
    return Complex(re / denom, im / denom);
  }
  __device__ __host__ inline Complex operator/=(const float &denom) {
    re /= denom;
    im /= denom;
    return *this;
  }
};

}  // namespace mega