/**
 * @file utils.h
 * @author Xincheng Xie (xxc@cs.wisc.edu)
 * @brief Utility functions and some constants that commonly used
 * @version 0.1
 * @date 2023-11-27
 *
 * @copyright Copyright (c) 2023
 *
 */

#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdint>
#include <stdexcept>

#include "matrix/matrix.h"
#include "mega_complex.h"

namespace mega {

static constexpr uint16_t kPrimeArray[309] = {
    2,    3,    5,    7,    11,   13,   17,   19,   23,   29,   31,   37,
    41,   43,   47,   53,   59,   61,   67,   71,   73,   79,   83,   89,
    97,   101,  103,  107,  109,  113,  127,  131,  137,  139,  149,  151,
    157,  163,  167,  173,  179,  181,  191,  193,  197,  199,  211,  223,
    227,  229,  233,  239,  241,  251,  257,  263,  269,  271,  277,  281,
    283,  293,  307,  311,  313,  317,  331,  337,  347,  349,  353,  359,
    367,  373,  379,  383,  389,  397,  401,  409,  419,  421,  431,  433,
    439,  443,  449,  457,  461,  463,  467,  479,  487,  491,  499,  503,
    509,  521,  523,  541,  547,  557,  563,  569,  571,  577,  587,  593,
    599,  601,  607,  613,  617,  619,  631,  641,  643,  647,  653,  659,
    661,  673,  677,  683,  691,  701,  709,  719,  727,  733,  739,  743,
    751,  757,  761,  769,  773,  787,  797,  809,  811,  821,  823,  827,
    829,  839,  853,  857,  859,  863,  877,  881,  883,  887,  907,  911,
    919,  929,  937,  941,  947,  953,  967,  971,  977,  983,  991,  997,
    1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069,
    1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163,
    1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249,
    1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321,
    1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439,
    1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511,
    1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601,
    1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693,
    1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783,
    1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877,
    1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987,
    1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039};

inline Matrix zadoff_chu_sequence(uint64_t seq_len) {
  Matrix seq(sizeof(Complex), seq_len, Matrix::kHost);
  Complex *seq_cpu = seq.ptr<Complex>();

  // Get the length of the zadoff-chu sequence based of \p seq_len
  uint16_t len = kPrimeArray[308];
  for (int j = 0; j < 308; j++) {
    if (kPrimeArray[j] <= seq_len && kPrimeArray[j + 1] > seq_len) {
      len = kPrimeArray[j];
      break;
    }
  }

  double root = floor(len * 2.0 / 31.0 + 0.5);
  for (uint64_t i = 0; i < seq_len; i++) {
    uint16_t m_loop = i % len;
    sincosf(-M_PI * root * m_loop * (m_loop + 1) / len, &seq_cpu[i].im,
            &seq_cpu[i].re);
  }

  return seq;
}

inline void cyclic_shift(const Matrix &seq_in, const Matrix &seq_out,
                         float alpha) {
  if (seq_in.nDim() != 1 || seq_out.nDim() != 1 ||
      seq_in.dim(0) != seq_out.dim(0)) {
    throw std::runtime_error(
        "Input and Output sequence must be a vector and must have the same "
        "length");
  }

  uint64_t seq_len = seq_in.dim(0);
  Complex *seq_cpu = seq_in.ptr<Complex>();
  Complex *seq_cyclic_shift_cpu = seq_out.ptr<Complex>();

  for (uint64_t i = 0; i < seq_len; i++) {
    Complex c;
    sincosf(alpha * i, &c.im, &c.re);
    seq_cyclic_shift_cpu[i] = seq_cpu[i] * c;
  }
}

inline Complex *seq_to_gpu(const Matrix &seq) {
  Complex *seq_gpu;
  cudaMalloc(&seq_gpu, seq.szBytes());
  cudaMemcpy(seq_gpu, seq.ptr(), seq.szBytes(), cudaMemcpyHostToDevice);
  return seq_gpu;
}

inline int get_device_id() {
  int device_id;
  cudaGetDevice(&device_id);
  return device_id;
}

}  // namespace mega