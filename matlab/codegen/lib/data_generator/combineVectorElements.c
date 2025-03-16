/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: combineVectorElements.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "combineVectorElements.h"
#include "data_generator_emxutil.h"
#include "data_generator_types.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : const emxArray_boolean_T *x_d
 *                const emxArray_int32_T *x_colidx
 *                int x_n
 *                emxArray_real_T *y_d
 *                emxArray_int32_T *y_colidx
 *                emxArray_int32_T *y_rowidx
 * Return Type  : int
 */
int combineVectorElements(const emxArray_boolean_T *x_d,
                          const emxArray_int32_T *x_colidx, int x_n,
                          emxArray_real_T *y_d, emxArray_int32_T *y_colidx,
                          emxArray_int32_T *y_rowidx)
{
  double *y_d_data;
  const int *x_colidx_data;
  int col;
  int outidx;
  int xp;
  int y_n;
  int *y_colidx_data;
  int *y_rowidx_data;
  const boolean_T *x_d_data;
  x_colidx_data = x_colidx->data;
  x_d_data = x_d->data;
  if (x_n == 0) {
    int i;
    y_n = 0;
    i = y_colidx->size[0];
    y_colidx->size[0] = 1;
    emxEnsureCapacity_int32_T(y_colidx, i);
    y_colidx_data = y_colidx->data;
    y_colidx_data[0] = 1;
    i = y_d->size[0];
    y_d->size[0] = 1;
    emxEnsureCapacity_real_T(y_d, i);
    y_d_data = y_d->data;
    y_d_data[0] = 0.0;
    i = y_rowidx->size[0];
    y_rowidx->size[0] = 1;
    emxEnsureCapacity_int32_T(y_rowidx, i);
    y_rowidx_data = y_rowidx->data;
    y_rowidx_data[0] = 1;
  } else {
    int i;
    outidx = x_colidx_data[x_colidx->size[0] - 1] - 1;
    if (x_n <= outidx) {
      outidx = x_n;
    }
    y_n = x_n;
    if (outidx < 1) {
      outidx = 1;
    }
    i = y_d->size[0];
    y_d->size[0] = outidx;
    emxEnsureCapacity_real_T(y_d, i);
    y_d_data = y_d->data;
    i = y_rowidx->size[0];
    y_rowidx->size[0] = outidx;
    emxEnsureCapacity_int32_T(y_rowidx, i);
    y_rowidx_data = y_rowidx->data;
    i = y_colidx->size[0];
    y_colidx->size[0] = x_n + 1;
    emxEnsureCapacity_int32_T(y_colidx, i);
    y_colidx_data = y_colidx->data;
    y_colidx_data[0] = 1;
    outidx = 1;
    i = (unsigned short)x_n;
    for (col = 0; col < i; col++) {
      double r;
      int xend;
      int xstart;
      xstart = x_colidx_data[col];
      xend = x_colidx_data[col + 1] - 1;
      r = 0.0;
      for (xp = xstart; xp <= xend; xp++) {
        r += (double)x_d_data[xp - 1];
      }
      if (r != 0.0) {
        y_d_data[outidx - 1] = r;
        outidx++;
      }
      y_colidx_data[col + 1] = outidx;
    }
    i = y_colidx_data[y_colidx->size[0] - 1];
    for (outidx = 0; outidx <= i - 2; outidx++) {
      y_rowidx_data[outidx] = 1;
    }
  }
  return y_n;
}

/*
 * File trailer for combineVectorElements.c
 *
 * [EOF]
 */
