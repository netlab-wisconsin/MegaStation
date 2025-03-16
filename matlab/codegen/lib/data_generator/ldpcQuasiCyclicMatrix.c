/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: ldpcQuasiCyclicMatrix.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "ldpcQuasiCyclicMatrix.h"
#include "colon.h"
#include "data_generator_emxutil.h"
#include "data_generator_types.h"
#include "rt_nonfinite.h"
#include "sparse.h"
#include "rt_nonfinite.h"
#include <math.h>

/* Function Definitions */
/*
 * Arguments    : double blockSize
 *                const double P_data[]
 *                const int P_size[2]
 *                emxArray_boolean_T *H_d
 *                emxArray_int32_T *H_colidx
 *                emxArray_int32_T *H_rowidx
 *                int *H_n
 * Return Type  : int
 */
int ldpcQuasiCyclicMatrix(double blockSize, const double P_data[],
                          const int P_size[2], emxArray_boolean_T *H_d,
                          emxArray_int32_T *H_colidx,
                          emxArray_int32_T *H_rowidx, int *H_n)
{
  c_sparse expl_temp;
  emxArray_real_T *b_y;
  emxArray_real_T *columnIndex;
  emxArray_real_T *rowIndex;
  emxArray_real_T *y;
  double ind;
  double n;
  double *b_y_data;
  double *columnIndex_data;
  double *rowIndex_data;
  double *y_data;
  int H_m;
  int b_i;
  int i;
  int i1;
  int idx;
  int ii;
  int loop_ub_tmp;
  int *H_colidx_data;
  boolean_T x_data[3128];
  boolean_T exitg1;
  boolean_T *H_d_data;
  loop_ub_tmp = P_size[0] * P_size[1];
  for (i = 0; i < loop_ub_tmp; i++) {
    x_data[i] = (P_data[i] != -1.0);
  }
  idx = 0;
  ii = 0;
  exitg1 = false;
  while ((!exitg1) && (ii <= loop_ub_tmp - 1)) {
    if (x_data[ii]) {
      idx++;
      if (idx >= loop_ub_tmp) {
        exitg1 = true;
      } else {
        ii++;
      }
    } else {
      ii++;
    }
  }
  if (idx < 1) {
    idx = 0;
  }
  n = (double)idx * blockSize;
  emxInit_real_T(&rowIndex, 1);
  i = rowIndex->size[0];
  rowIndex->size[0] = (int)n;
  emxEnsureCapacity_real_T(rowIndex, i);
  rowIndex_data = rowIndex->data;
  emxInit_real_T(&columnIndex, 1);
  i = columnIndex->size[0];
  columnIndex->size[0] = (int)n;
  emxEnsureCapacity_real_T(columnIndex, i);
  columnIndex_data = columnIndex->data;
  ind = 0.0;
  i = P_size[1];
  idx = P_size[0];
  emxInit_real_T(&y, 2);
  y_data = y->data;
  emxInit_real_T(&b_y, 2);
  b_y_data = b_y->data;
  for (loop_ub_tmp = 0; loop_ub_tmp < i; loop_ub_tmp++) {
    for (b_i = 0; b_i < idx; b_i++) {
      n = P_data[b_i + P_size[0] * loop_ub_tmp];
      if (n != -1.0) {
        double tmp_data[384];
        double j;
        boolean_T b;
        b = rtIsNaN(blockSize);
        if (b) {
          i1 = y->size[0] * y->size[1];
          y->size[0] = 1;
          y->size[1] = 1;
          emxEnsureCapacity_real_T(y, i1);
          y_data = y->data;
          y_data[0] = rtNaN;
        } else if (blockSize < 1.0) {
          y->size[0] = 1;
          y->size[1] = 0;
        } else {
          i1 = y->size[0] * y->size[1];
          y->size[0] = 1;
          y->size[1] = (int)(blockSize - 1.0) + 1;
          emxEnsureCapacity_real_T(y, i1);
          y_data = y->data;
          ii = (int)(blockSize - 1.0);
          for (i1 = 0; i1 <= ii; i1++) {
            y_data[i1] = (double)i1 + 1.0;
          }
        }
        ii = y->size[1];
        j = (((double)loop_ub_tmp + 1.0) - 1.0) * blockSize;
        for (i1 = 0; i1 < ii; i1++) {
          double d;
          double d1;
          d = y_data[i1];
          d1 = ind + d;
          tmp_data[i1] = d1;
          columnIndex_data[(int)d1 - 1] = j + d;
        }
        n = blockSize - n;
        if (rtIsNaN(n + 1.0) || b) {
          i1 = b_y->size[0] * b_y->size[1];
          b_y->size[0] = 1;
          b_y->size[1] = 1;
          emxEnsureCapacity_real_T(b_y, i1);
          b_y_data = b_y->data;
          b_y_data[0] = rtNaN;
        } else if (blockSize < n + 1.0) {
          b_y->size[0] = 1;
          b_y->size[1] = 0;
        } else if ((rtIsInf(n + 1.0) || rtIsInf(blockSize)) &&
                   (n + 1.0 == blockSize)) {
          i1 = b_y->size[0] * b_y->size[1];
          b_y->size[0] = 1;
          b_y->size[1] = 1;
          emxEnsureCapacity_real_T(b_y, i1);
          b_y_data = b_y->data;
          b_y_data[0] = rtNaN;
        } else if (floor(n + 1.0) == n + 1.0) {
          i1 = b_y->size[0] * b_y->size[1];
          b_y->size[0] = 1;
          ii = (int)(blockSize - (n + 1.0));
          b_y->size[1] = ii + 1;
          emxEnsureCapacity_real_T(b_y, i1);
          b_y_data = b_y->data;
          for (i1 = 0; i1 <= ii; i1++) {
            b_y_data[i1] = (n + 1.0) + (double)i1;
          }
        } else {
          eml_float_colon(n + 1.0, blockSize, b_y);
          b_y_data = b_y->data;
        }
        if (rtIsNaN(n)) {
          i1 = y->size[0] * y->size[1];
          y->size[0] = 1;
          y->size[1] = 1;
          emxEnsureCapacity_real_T(y, i1);
          y_data = y->data;
          y_data[0] = rtNaN;
        } else if (n < 1.0) {
          y->size[0] = 1;
          y->size[1] = 0;
        } else {
          i1 = y->size[0] * y->size[1];
          y->size[0] = 1;
          y->size[1] = (int)(n - 1.0) + 1;
          emxEnsureCapacity_real_T(y, i1);
          y_data = y->data;
          ii = (int)(n - 1.0);
          for (i1 = 0; i1 <= ii; i1++) {
            y_data[i1] = (double)i1 + 1.0;
          }
        }
        n = (((double)b_i + 1.0) - 1.0) * blockSize;
        ii = b_y->size[1];
        for (i1 = 0; i1 < ii; i1++) {
          rowIndex_data[(int)tmp_data[i1] - 1] = n + b_y_data[i1];
        }
        ii = y->size[1];
        for (i1 = 0; i1 < ii; i1++) {
          rowIndex_data[(int)tmp_data[i1 + b_y->size[1]] - 1] = n + y_data[i1];
        }
        ind += blockSize;
      }
    }
  }
  emxFree_real_T(&b_y);
  emxFree_real_T(&y);
  emxInitStruct_sparse(&expl_temp);
  sparse(rowIndex, columnIndex, (double)P_size[0] * blockSize,
         (double)P_size[1] * blockSize, &expl_temp);
  emxFree_real_T(&columnIndex);
  emxFree_real_T(&rowIndex);
  i = H_d->size[0];
  H_d->size[0] = expl_temp.d->size[0];
  emxEnsureCapacity_boolean_T(H_d, i);
  H_d_data = H_d->data;
  ii = expl_temp.d->size[0];
  for (i = 0; i < ii; i++) {
    H_d_data[i] = expl_temp.d->data[i];
  }
  i = H_colidx->size[0];
  H_colidx->size[0] = expl_temp.colidx->size[0];
  emxEnsureCapacity_int32_T(H_colidx, i);
  H_colidx_data = H_colidx->data;
  ii = expl_temp.colidx->size[0];
  for (i = 0; i < ii; i++) {
    H_colidx_data[i] = expl_temp.colidx->data[i];
  }
  i = H_rowidx->size[0];
  H_rowidx->size[0] = expl_temp.rowidx->size[0];
  emxEnsureCapacity_int32_T(H_rowidx, i);
  H_colidx_data = H_rowidx->data;
  ii = expl_temp.rowidx->size[0];
  for (i = 0; i < ii; i++) {
    H_colidx_data[i] = expl_temp.rowidx->data[i];
  }
  emxFreeStruct_sparse(&expl_temp);
  H_m = expl_temp.m;
  *H_n = expl_temp.n;
  return H_m;
}

/*
 * File trailer for ldpcQuasiCyclicMatrix.c
 *
 * [EOF]
 */
