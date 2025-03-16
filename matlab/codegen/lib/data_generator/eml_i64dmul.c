/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: eml_i64dmul.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "eml_i64dmul.h"
#include "data_generator_rtwutil.h"
#include "rt_nonfinite.h"
#include "rt_nonfinite.h"
#include <math.h>

/* Function Definitions */
/*
 * Arguments    : unsigned long long x
 *                double y
 * Return Type  : unsigned long long
 */
unsigned long long times(unsigned long long x, double y)
{
  unsigned long long z;
  int ex_t;
  if (rtIsNaN(y) || (y <= 0.0)) {
    z = 0ULL;
  } else {
    double yd;
    yd = frexp(y, &ex_t);
    if (ex_t <= -64) {
      z = 0ULL;
    } else {
      unsigned long long b_y0;
      unsigned long long b_y1;
      unsigned long long yint;
      yint = (unsigned long long)rt_roundd_snf(yd * 9.007199254740992E+15);
      b_y1 = yint >> 32;
      b_y0 = yint & 4294967295ULL;
      if (x == 0ULL) {
        z = 0ULL;
      } else if (rtIsInf(y)) {
        z = MAX_uint64_T;
      } else if (ex_t - 53 > 64) {
        z = MAX_uint64_T;
      } else {
        unsigned long long ldword;
        unsigned long long n1;
        unsigned long long temp0;
        n1 = x >> 32;
        yint = x & 4294967295ULL;
        ldword = yint * b_y0;
        temp0 = yint * b_y1;
        yint = n1 * b_y0;
        b_y0 =
            ((temp0 & 4294967295ULL) + (ldword >> 32)) + (yint & 4294967295ULL);
        ldword = (ldword & 4294967295ULL) + (b_y0 << 32);
        yint = ((n1 * b_y1 + (temp0 >> 32)) + (yint >> 32)) + (b_y0 >> 32);
        if (ex_t - 53 >= 0) {
          if (yint > 0ULL) {
            z = MAX_uint64_T;
          } else {
            short i;
            boolean_T guard1;
            guard1 = false;
            if (117 - ex_t < 64) {
              i = (short)(117 - ex_t);
              if (117 - ex_t < 0) {
                i = 0;
              }
              if ((ldword >> (unsigned char)i) > 0ULL) {
                z = MAX_uint64_T;
              } else {
                guard1 = true;
              }
            } else {
              guard1 = true;
            }
            if (guard1) {
              i = (short)(ex_t - 53);
              if (ex_t - 53 < 0) {
                i = 0;
              } else if (ex_t - 53 > 255) {
                i = 255;
              }
              z = ldword << (unsigned char)i;
            }
          }
        } else {
          short i;
          short i1;
          short i2;
          short i3;
          short i4;
          short i5;
          i = (short)(53 - ex_t);
          if (53 - ex_t < 0) {
            i = 0;
          }
          i1 = (short)(53 - ex_t);
          if (53 - ex_t < 0) {
            i1 = 0;
          }
          i2 = (short)(ex_t + 11);
          if (ex_t + 11 < 0) {
            i2 = 0;
          } else if (ex_t + 11 > 255) {
            i2 = 255;
          }
          i3 = (short)(52 - ex_t);
          if (52 - ex_t < 0) {
            i3 = 0;
          }
          i4 = (short)-(ex_t + 11);
          if (-(ex_t + 11) < 0) {
            i4 = 0;
          }
          i5 = (short)-(ex_t + 12);
          if (-(ex_t + 12) < 0) {
            i5 = 0;
          }
          if (ex_t - 53 > -64) {
            if ((yint >> (unsigned char)i) > 0ULL) {
              z = MAX_uint64_T;
            } else {
              z = ((ldword >> (unsigned char)i1) +
                   (yint << (unsigned char)i2)) +
                  (ldword >> (unsigned char)i3 & 1ULL);
            }
          } else if (ex_t - 53 == -64) {
            z = yint + (ldword >> 63 & 1ULL);
          } else {
            z = (yint >> (unsigned char)i4) +
                (yint >> (unsigned char)i5 & 1ULL);
          }
        }
      }
    }
  }
  return z;
}

/*
 * File trailer for eml_i64dmul.c
 *
 * [EOF]
 */
