/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: exp.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "exp.h"
#include "data_generator_types.h"
#include "rt_nonfinite.h"
#include "rt_nonfinite.h"
#include <math.h>

/* Function Definitions */
/*
 * Arguments    : emxArray_creal32_T *x
 * Return Type  : void
 */
void b_exp(emxArray_creal32_T *x)
{
  creal32_T *x_data;
  int k;
  int nx;
  x_data = x->data;
  nx = x->size[0];
  for (k = 0; k < nx; k++) {
    float im;
    float r;
    r = x_data[k].re;
    im = x_data[k].im;
    if (r == 0.0F) {
      x_data[k].re = cosf(im);
      x_data[k].im = sinf(im);
    } else if (im == 0.0F) {
      x_data[k].re = expf(r);
      x_data[k].im = 0.0F;
    } else if (rtIsInfF(im) && rtIsInfF(r) && (r < 0.0F)) {
      x_data[k].re = 0.0F;
      x_data[k].im = 0.0F;
    } else {
      r = expf(r / 2.0F);
      x_data[k].re = r * (r * cosf(im));
      x_data[k].im = r * (r * sinf(im));
    }
  }
}

/*
 * File trailer for exp.c
 *
 * [EOF]
 */
