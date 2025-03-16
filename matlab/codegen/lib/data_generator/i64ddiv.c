/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: i64ddiv.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "i64ddiv.h"
#include "data_generator_rtwutil.h"
#include "rt_nonfinite.h"
#include "rt_nonfinite.h"
#include <math.h>
#include <stddef.h>
#include <string.h>

/* Function Definitions */
/*
 * Arguments    : unsigned long long x
 *                double y
 * Return Type  : unsigned long long
 */
unsigned long long b_i64ddiv(unsigned long long x, double y)
{
  unsigned long long n;
  unsigned long long z;
  int shiftAmount;
  n = x;
  if (y == 0.0) {
    if (x != 0ULL) {
      unsigned int ux[2];
      memcpy((void *)&ux[0], (void *)&y,
             (unsigned int)((size_t)2 * sizeof(unsigned int)));
      if ((ux[0] != 0U) || (ux[1] != 0U)) {
        z = 0ULL;
      } else {
        z = MAX_uint64_T;
      }
    } else {
      z = 0ULL;
    }
  } else if ((x == 0ULL) || (y < 0.0) || (rtIsInf(y) || rtIsNaN(y))) {
    z = 0ULL;
  } else {
    double b_x;
    unsigned long long xint;
    int xexp;
    b_x = frexp(y, &shiftAmount);
    xint = (unsigned long long)rt_roundd_snf(b_x * 9.007199254740992E+15);
    xexp = shiftAmount - 53;
    if (shiftAmount - 53 > 12) {
      z = 0ULL;
    } else if (shiftAmount - 53 < -116) {
      z = MAX_uint64_T;
    } else {
      if (xint == 0ULL) {
        z = MAX_uint64_T;
      } else {
        z = x / xint;
      }
      if (shiftAmount - 53 > 0) {
        n = z >> (shiftAmount - 53);
        z = n + (z >> (shiftAmount - 54) & 1ULL);
        if (z < n) {
          z = MAX_uint64_T;
        }
      } else {
        unsigned long long u;
        if (xint != 0ULL) {
          if (xint == 0ULL) {
            u = MAX_uint64_T;
          } else {
            u = x / xint;
          }
          n = x - u * xint;
        }
        int exitg1;
        do {
          exitg1 = 0;
          if (xexp < 0) {
            shiftAmount = -xexp;
            if (shiftAmount > 11) {
              shiftAmount = 11;
            }
            if ((z >> (64 - shiftAmount)) > 0ULL) {
              z = MAX_uint64_T;
              exitg1 = 1;
            } else {
              z <<= shiftAmount;
              n <<= shiftAmount;
              xexp += shiftAmount;
              if (xint == 0ULL) {
                u = MAX_uint64_T;
              } else {
                u = n / xint;
              }
              if (MAX_uint64_T - u <= z) {
                z = MAX_uint64_T;
                exitg1 = 1;
              } else {
                z += u;
                if (xint != 0ULL) {
                  if (xint == 0ULL) {
                    u = MAX_uint64_T;
                  } else {
                    u = n / xint;
                  }
                  n -= u * xint;
                }
              }
            }
          } else {
            if ((n << 1) >= xint) {
              z++;
            }
            exitg1 = 1;
          }
        } while (exitg1 == 0);
      }
    }
  }
  return z;
}

/*
 * Arguments    : unsigned long long x
 * Return Type  : unsigned long long
 */
unsigned long long i64ddiv(unsigned long long x)
{
  unsigned long long z;
  int xexp;
  if (x == 0ULL) {
    z = 0ULL;
  } else {
    unsigned long long n;
    frexp(2.0, &xexp);
    xexp = -51;
    z = x >> 52;
    n = x - (z << 52);
    int exitg1;
    do {
      exitg1 = 0;
      if (xexp < 0) {
        int shiftAmount;
        shiftAmount = -xexp;
        if (shiftAmount > 11) {
          shiftAmount = 11;
        }
        if ((z >> (64 - shiftAmount)) > 0ULL) {
          z = MAX_uint64_T;
          exitg1 = 1;
        } else {
          unsigned long long t_tmp;
          z <<= shiftAmount;
          n <<= shiftAmount;
          xexp += shiftAmount;
          t_tmp = n >> 52;
          if (MAX_uint64_T - t_tmp <= z) {
            z = MAX_uint64_T;
            exitg1 = 1;
          } else {
            z += t_tmp;
            n -= t_tmp << 52;
          }
        }
      } else {
        if ((n << 1) >= 4503599627370496ULL) {
          z++;
        }
        exitg1 = 1;
      }
    } while (exitg1 == 0);
  }
  return z;
}

/*
 * File trailer for i64ddiv.c
 *
 * [EOF]
 */
