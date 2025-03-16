/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: randn.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "randn.h"
#include "data_generator_data.h"
#include "data_generator_emxutil.h"
#include "data_generator_types.h"
#include "eml_rand_mt19937ar.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : unsigned long long varargin_1
 *                unsigned long long varargin_2
 *                emxArray_creal32_T *r
 * Return Type  : void
 */
void b_complexLike(unsigned long long varargin_1, unsigned long long varargin_2,
                   emxArray_creal32_T *r)
{
  creal32_T *r_data;
  int k;
  int loop_ub;
  k = r->size[0] * r->size[1];
  r->size[0] = (int)varargin_1;
  r->size[1] = (int)varargin_2;
  emxEnsureCapacity_creal32_T(r, k);
  r_data = r->data;
  loop_ub = (int)varargin_1 * (int)varargin_2;
  for (k = 0; k < loop_ub; k++) {
    double im;
    double re;
    re = eml_rand_mt19937ar(state);
    im = eml_rand_mt19937ar(state);
    r_data[k].re = (float)re;
    r_data[k].im = (float)im;
  }
  for (k = 0; k < loop_ub; k++) {
    float ai;
    float b_im;
    float b_re;
    b_im = r_data[k].re;
    ai = r_data[k].im;
    if (ai == 0.0F) {
      b_re = b_im / 1.41421354F;
      b_im = 0.0F;
    } else if (b_im == 0.0F) {
      b_re = 0.0F;
      b_im = ai / 1.41421354F;
    } else {
      b_re = b_im / 1.41421354F;
      b_im = ai / 1.41421354F;
    }
    r_data[k].re = b_re;
    r_data[k].im = b_im;
  }
}

/*
 * File trailer for randn.c
 *
 * [EOF]
 */
