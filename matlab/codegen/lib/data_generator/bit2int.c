/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: bit2int.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "bit2int.h"
#include "data_generator_emxutil.h"
#include "data_generator_types.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : const emxArray_real_T *x
 *                emxArray_real_T *y
 * Return Type  : void
 */
void bit2int(const emxArray_real_T *x, emxArray_real_T *y)
{
  static const unsigned char uv[8] = {1U, 2U, 4U, 8U, 16U, 32U, 64U, 128U};
  emxArray_real_T *r;
  const double *x_data;
  double inSize_idx_0;
  double *r1;
  double *y_data;
  int boffset;
  int emptyDimValue;
  int j;
  int k;
  x_data = x->data;
  inSize_idx_0 = (double)x->size[0] / 8.0;
  emptyDimValue =
      (int)((unsigned int)(x->size[0] * x->size[1] * x->size[2]) >> 3);
  emxInit_real_T(&r, 2);
  j = r->size[0] * r->size[1];
  r->size[0] = 1;
  r->size[1] = emptyDimValue;
  emxEnsureCapacity_real_T(r, j);
  r1 = r->data;
  for (j = 0; j < emptyDimValue; j++) {
    double s;
    boffset = j << 3;
    s = 0.0;
    for (k = 0; k < 8; k++) {
      s += (double)uv[k] * x_data[boffset + k];
    }
    r1[j] = s;
  }
  j = y->size[0] * y->size[1] * y->size[2];
  y->size[0] = (int)inSize_idx_0;
  y->size[1] = x->size[1];
  y->size[2] = x->size[2];
  emxEnsureCapacity_real_T(y, j);
  y_data = y->data;
  boffset = (int)inSize_idx_0 * x->size[1] * x->size[2];
  for (j = 0; j < boffset; j++) {
    y_data[j] = r1[j];
  }
  emxFree_real_T(&r);
}

/*
 * File trailer for bit2int.c
 *
 * [EOF]
 */
