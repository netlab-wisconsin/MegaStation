/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: triu.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "triu.h"
#include "data_generator_emxutil.h"
#include "data_generator_types.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : const emxArray_boolean_T *A_d
 *                const emxArray_int32_T *A_colidx
 *                const emxArray_int32_T *A_rowidx
 *                int A_m
 *                int A_n
 *                emxArray_boolean_T *U_d
 *                emxArray_int32_T *U_colidx
 *                emxArray_int32_T *U_rowidx
 *                int *U_n
 * Return Type  : int
 */
int sparse_triu(const emxArray_boolean_T *A_d, const emxArray_int32_T *A_colidx,
                const emxArray_int32_T *A_rowidx, int A_m, int A_n,
                emxArray_boolean_T *U_d, emxArray_int32_T *U_colidx,
                emxArray_int32_T *U_rowidx, int *U_n)
{
  const int *A_colidx_data;
  const int *A_rowidx_data;
  int U_m;
  int col;
  int didx;
  int i;
  int *U_colidx_data;
  int *U_rowidx_data;
  const boolean_T *A_d_data;
  boolean_T *U_d_data;
  A_rowidx_data = A_rowidx->data;
  A_colidx_data = A_colidx->data;
  A_d_data = A_d->data;
  didx = A_colidx_data[A_colidx->size[0] - 1] - 1;
  if (didx < 1) {
    didx = 1;
  }
  i = U_d->size[0];
  U_d->size[0] = didx;
  emxEnsureCapacity_boolean_T(U_d, i);
  U_d_data = U_d->data;
  i = U_rowidx->size[0];
  U_rowidx->size[0] = didx;
  emxEnsureCapacity_int32_T(U_rowidx, i);
  U_rowidx_data = U_rowidx->data;
  i = U_colidx->size[0];
  U_colidx->size[0] = A_n + 1;
  emxEnsureCapacity_int32_T(U_colidx, i);
  U_colidx_data = U_colidx->data;
  U_colidx_data[0] = 1;
  didx = 0;
  i = (unsigned short)A_n;
  for (col = 0; col < i; col++) {
    int ridx;
    ridx = A_colidx_data[col] - 1;
    while ((ridx + 1 < A_colidx_data[col + 1]) &&
           (A_rowidx_data[ridx] <= col)) {
      U_rowidx_data[didx] = A_rowidx_data[ridx];
      U_d_data[didx] = A_d_data[ridx];
      didx++;
      ridx++;
    }
    U_colidx_data[col + 1] = didx + 1;
  }
  U_m = A_m;
  *U_n = A_n;
  return U_m;
}

/*
 * File trailer for triu.c
 *
 * [EOF]
 */
