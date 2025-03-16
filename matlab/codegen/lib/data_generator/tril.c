/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: tril.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "tril.h"
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
 *                emxArray_boolean_T *L_d
 *                emxArray_int32_T *L_colidx
 *                emxArray_int32_T *L_rowidx
 *                int *L_n
 * Return Type  : int
 */
int sparse_tril(const emxArray_boolean_T *A_d, const emxArray_int32_T *A_colidx,
                const emxArray_int32_T *A_rowidx, int A_m, int A_n,
                emxArray_boolean_T *L_d, emxArray_int32_T *L_colidx,
                emxArray_int32_T *L_rowidx, int *L_n)
{
  const int *A_colidx_data;
  const int *A_rowidx_data;
  int L_m;
  int col;
  int didx;
  int i;
  int ridx;
  int *L_colidx_data;
  int *L_rowidx_data;
  const boolean_T *A_d_data;
  boolean_T *L_d_data;
  A_rowidx_data = A_rowidx->data;
  A_colidx_data = A_colidx->data;
  A_d_data = A_d->data;
  didx = A_colidx_data[A_colidx->size[0] - 1] - 1;
  if (didx < 1) {
    didx = 1;
  }
  i = L_d->size[0];
  L_d->size[0] = didx;
  emxEnsureCapacity_boolean_T(L_d, i);
  L_d_data = L_d->data;
  i = L_rowidx->size[0];
  L_rowidx->size[0] = didx;
  emxEnsureCapacity_int32_T(L_rowidx, i);
  L_rowidx_data = L_rowidx->data;
  i = L_colidx->size[0];
  L_colidx->size[0] = A_n + 1;
  emxEnsureCapacity_int32_T(L_colidx, i);
  L_colidx_data = L_colidx->data;
  L_colidx_data[0] = 1;
  didx = 0;
  i = (unsigned short)A_n;
  for (col = 0; col < i; col++) {
    for (ridx = A_colidx_data[col] - 1; ridx + 1 < A_colidx_data[col + 1];
         ridx++) {
      if (A_rowidx_data[ridx] >= col + 2) {
        L_rowidx_data[didx] = A_rowidx_data[ridx];
        L_d_data[didx] = A_d_data[ridx];
        didx++;
      }
    }
    L_colidx_data[col + 1] = didx + 1;
  }
  L_m = A_m;
  *L_n = A_n;
  return L_m;
}

/*
 * File trailer for tril.c
 *
 * [EOF]
 */
