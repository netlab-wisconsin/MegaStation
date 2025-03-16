/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: sparse.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "sparse.h"
#include "data_generator_emxutil.h"
#include "data_generator_types.h"
#include "fillIn.h"
#include "rt_nonfinite.h"
#include "sparse1.h"

/* Function Definitions */
/*
 * Arguments    : const emxArray_boolean_T *varargin_1
 *                emxArray_boolean_T *y_d
 *                emxArray_int32_T *y_colidx
 *                emxArray_int32_T *y_rowidx
 *                int *y_n
 * Return Type  : int
 */
int b_sparse(const emxArray_boolean_T *varargin_1, emxArray_boolean_T *y_d,
             emxArray_int32_T *y_colidx, emxArray_int32_T *y_rowidx, int *y_n)
{
  int col;
  int ctr;
  int i;
  int numalloc;
  int row;
  int y_m;
  int *y_colidx_data;
  int *y_rowidx_data;
  const boolean_T *varargin_1_data;
  boolean_T *y_d_data;
  varargin_1_data = varargin_1->data;
  numalloc = 0;
  i = varargin_1->size[0] * varargin_1->size[1];
  for (ctr = 0; ctr < i; ctr++) {
    if (varargin_1_data[ctr]) {
      numalloc++;
    }
  }
  y_m = varargin_1->size[0];
  *y_n = varargin_1->size[1];
  if (numalloc < 1) {
    numalloc = 1;
  }
  i = y_d->size[0];
  y_d->size[0] = numalloc;
  emxEnsureCapacity_boolean_T(y_d, i);
  y_d_data = y_d->data;
  for (i = 0; i < numalloc; i++) {
    y_d_data[i] = false;
  }
  i = y_colidx->size[0];
  y_colidx->size[0] = varargin_1->size[1] + 1;
  emxEnsureCapacity_int32_T(y_colidx, i);
  y_colidx_data = y_colidx->data;
  ctr = varargin_1->size[1];
  for (i = 0; i <= ctr; i++) {
    y_colidx_data[i] = 0;
  }
  y_colidx_data[0] = 1;
  i = y_rowidx->size[0];
  y_rowidx->size[0] = numalloc;
  emxEnsureCapacity_int32_T(y_rowidx, i);
  y_rowidx_data = y_rowidx->data;
  for (i = 0; i < numalloc; i++) {
    y_rowidx_data[i] = 0;
  }
  y_rowidx_data[0] = 1;
  ctr = 0;
  i = (unsigned short)varargin_1->size[1];
  for (col = 0; col < i; col++) {
    numalloc = (unsigned short)varargin_1->size[0];
    for (row = 0; row < numalloc; row++) {
      if (varargin_1_data[row + varargin_1->size[0] * col]) {
        y_rowidx_data[ctr] = row + 1;
        y_d_data[ctr] = true;
        ctr++;
      }
    }
    y_colidx_data[col + 1] = ctr + 1;
  }
  return y_m;
}

/*
 * Arguments    : const emxArray_real_T *varargin_1
 *                const emxArray_real_T *varargin_2
 *                double varargin_4
 *                double varargin_5
 *                c_sparse *y
 * Return Type  : void
 */
void sparse(const emxArray_real_T *varargin_1,
            const emxArray_real_T *varargin_2, double varargin_4,
            double varargin_5, c_sparse *y)
{
  emxArray_int32_T *b_sint;
  emxArray_int32_T *sint;
  emxArray_int32_T *sortedIndices;
  const double *varargin_1_data;
  const double *varargin_2_data;
  int i;
  int nc;
  int ns;
  int numalloc;
  int *sint_data;
  int *sortedIndices_data;
  varargin_2_data = varargin_2->data;
  varargin_1_data = varargin_1->data;
  nc = varargin_2->size[0];
  emxInit_int32_T(&sortedIndices, 1);
  i = sortedIndices->size[0];
  sortedIndices->size[0] = varargin_2->size[0];
  emxEnsureCapacity_int32_T(sortedIndices, i);
  sortedIndices_data = sortedIndices->data;
  emxInit_int32_T(&sint, 1);
  i = sint->size[0];
  sint->size[0] = varargin_2->size[0];
  emxEnsureCapacity_int32_T(sint, i);
  sint_data = sint->data;
  for (numalloc = 0; numalloc < nc; numalloc++) {
    sortedIndices_data[numalloc] = numalloc + 1;
    sint_data[numalloc] = (int)varargin_2_data[numalloc];
  }
  ns = varargin_1->size[0];
  emxInit_int32_T(&b_sint, 1);
  i = b_sint->size[0];
  b_sint->size[0] = varargin_1->size[0];
  emxEnsureCapacity_int32_T(b_sint, i);
  sortedIndices_data = b_sint->data;
  for (numalloc = 0; numalloc < ns; numalloc++) {
    sortedIndices_data[numalloc] = (int)varargin_1_data[numalloc];
  }
  locSortrows(sortedIndices, sint, b_sint);
  sortedIndices_data = b_sint->data;
  sint_data = sint->data;
  emxFree_int32_T(&sortedIndices);
  y->m = (int)varargin_4;
  ns = (int)varargin_5;
  y->n = (int)varargin_5;
  numalloc = varargin_2->size[0];
  if (numalloc < 1) {
    numalloc = 1;
  }
  i = y->d->size[0];
  y->d->size[0] = numalloc;
  emxEnsureCapacity_boolean_T(y->d, i);
  for (i = 0; i < numalloc; i++) {
    y->d->data[i] = false;
  }
  i = y->colidx->size[0];
  y->colidx->size[0] = (int)varargin_5 + 1;
  emxEnsureCapacity_int32_T(y->colidx, i);
  for (i = 0; i <= ns; i++) {
    y->colidx->data[i] = 0;
  }
  y->colidx->data[0] = 1;
  i = y->rowidx->size[0];
  y->rowidx->size[0] = numalloc;
  emxEnsureCapacity_int32_T(y->rowidx, i);
  for (i = 0; i < numalloc; i++) {
    y->rowidx->data[i] = 0;
  }
  ns = 0;
  i = (unsigned short)(int)varargin_5;
  for (numalloc = 0; numalloc < i; numalloc++) {
    while ((ns + 1 <= nc) && (sint_data[ns] == numalloc + 1)) {
      y->rowidx->data[ns] = sortedIndices_data[ns];
      ns++;
    }
    y->colidx->data[numalloc + 1] = ns + 1;
  }
  emxFree_int32_T(&b_sint);
  emxFree_int32_T(&sint);
  for (numalloc = 0; numalloc < nc; numalloc++) {
    y->d->data[numalloc] = true;
  }
  sparse_fillIn(y);
}

/*
 * File trailer for sparse.c
 *
 * [EOF]
 */
