/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: sparse1.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "sparse1.h"
#include "data_generator_emxutil.h"
#include "data_generator_types.h"
#include "fillIn.h"
#include "introsort.h"
#include "rt_nonfinite.h"

/* Function Declarations */
static void permuteVector(const emxArray_int32_T *idx, emxArray_int32_T *y);

/* Function Definitions */
/*
 * Arguments    : const emxArray_int32_T *idx
 *                emxArray_int32_T *y
 * Return Type  : void
 */
static void permuteVector(const emxArray_int32_T *idx, emxArray_int32_T *y)
{
  emxArray_int32_T *t;
  const int *idx_data;
  int i;
  int loop_ub;
  int ny;
  int *t_data;
  int *y_data;
  y_data = y->data;
  idx_data = idx->data;
  ny = y->size[0];
  emxInit_int32_T(&t, 1);
  i = t->size[0];
  t->size[0] = y->size[0];
  emxEnsureCapacity_int32_T(t, i);
  t_data = t->data;
  loop_ub = y->size[0];
  for (i = 0; i < loop_ub; i++) {
    t_data[i] = y_data[i];
  }
  for (loop_ub = 0; loop_ub < ny; loop_ub++) {
    y_data[loop_ub] = t_data[idx_data[loop_ub] - 1];
  }
  emxFree_int32_T(&t);
}

/*
 * Arguments    : emxArray_int32_T *idx
 *                emxArray_int32_T *a
 *                emxArray_int32_T *b
 * Return Type  : void
 */
void locSortrows(emxArray_int32_T *idx, emxArray_int32_T *a,
                 emxArray_int32_T *b)
{
  introsort(idx, a->size[0], a, b);
  permuteVector(idx, a);
  permuteVector(idx, b);
}

/*
 * Arguments    : const emxArray_boolean_T *this_d
 *                const emxArray_int32_T *this_colidx
 *                const emxArray_int32_T *this_rowidx
 *                int this_m
 *                int this_n
 *                emxArray_boolean_T *y
 * Return Type  : void
 */
void sparse_full(const emxArray_boolean_T *this_d,
                 const emxArray_int32_T *this_colidx,
                 const emxArray_int32_T *this_rowidx, int this_m, int this_n,
                 emxArray_boolean_T *y)
{
  const int *this_colidx_data;
  const int *this_rowidx_data;
  int c;
  int cend;
  int i;
  int idx;
  const boolean_T *this_d_data;
  boolean_T *y_data;
  this_rowidx_data = this_rowidx->data;
  this_colidx_data = this_colidx->data;
  this_d_data = this_d->data;
  i = y->size[0] * y->size[1];
  y->size[0] = this_m;
  y->size[1] = this_n;
  emxEnsureCapacity_boolean_T(y, i);
  y_data = y->data;
  cend = this_m * this_n;
  for (i = 0; i < cend; i++) {
    y_data[i] = false;
  }
  i = (unsigned short)this_n;
  for (c = 0; c < i; c++) {
    int i1;
    cend = this_colidx_data[c + 1] - 1;
    i1 = this_colidx_data[c];
    for (idx = i1; idx <= cend; idx++) {
      y_data[(this_rowidx_data[idx - 1] + y->size[0] * c) - 1] =
          this_d_data[idx - 1];
    }
  }
}

/*
 * Arguments    : const emxArray_boolean_T *this_d
 *                const emxArray_int32_T *this_colidx
 *                const emxArray_int32_T *this_rowidx
 *                int this_m
 *                const emxArray_real_T *varargin_2
 *                c_sparse *s
 * Return Type  : void
 */
void sparse_parenReference(const emxArray_boolean_T *this_d,
                           const emxArray_int32_T *this_colidx,
                           const emxArray_int32_T *this_rowidx, int this_m,
                           const emxArray_real_T *varargin_2, c_sparse *s)
{
  const double *varargin_2_data;
  double d;
  const int *this_colidx_data;
  const int *this_rowidx_data;
  int cidx;
  int i;
  int k;
  int loop_ub;
  int nd;
  int numalloc;
  int sn;
  const boolean_T *this_d_data;
  varargin_2_data = varargin_2->data;
  this_rowidx_data = this_rowidx->data;
  this_colidx_data = this_colidx->data;
  this_d_data = this_d->data;
  sn = varargin_2->size[1] - 1;
  nd = 0;
  for (cidx = 0; cidx <= sn; cidx++) {
    d = varargin_2_data[cidx];
    nd = (nd + this_colidx_data[(int)d]) - this_colidx_data[(int)d - 1];
  }
  s->n = varargin_2->size[1];
  s->m = this_m;
  if (nd >= 1) {
    numalloc = nd;
  } else {
    numalloc = 1;
  }
  i = s->d->size[0];
  s->d->size[0] = numalloc;
  emxEnsureCapacity_boolean_T(s->d, i);
  for (i = 0; i < numalloc; i++) {
    s->d->data[i] = false;
  }
  i = s->colidx->size[0];
  s->colidx->size[0] = varargin_2->size[1] + 1;
  emxEnsureCapacity_int32_T(s->colidx, i);
  loop_ub = varargin_2->size[1];
  for (i = 0; i <= loop_ub; i++) {
    s->colidx->data[i] = 0;
  }
  s->colidx->data[0] = 1;
  i = s->rowidx->size[0];
  s->rowidx->size[0] = numalloc;
  emxEnsureCapacity_int32_T(s->rowidx, i);
  for (i = 0; i < numalloc; i++) {
    s->rowidx->data[i] = 0;
  }
  i = varargin_2->size[1];
  for (numalloc = 0; numalloc < i; numalloc++) {
    s->colidx->data[numalloc + 1] = 1;
  }
  sparse_fillIn(s);
  if (nd != 0) {
    numalloc = 0;
    for (cidx = 0; cidx <= sn; cidx++) {
      d = varargin_2_data[cidx];
      loop_ub = this_colidx_data[(int)d - 1];
      nd = this_colidx_data[(int)d] - loop_ub;
      for (k = 0; k < nd; k++) {
        int i1;
        i = (loop_ub + k) - 1;
        i1 = numalloc + k;
        s->d->data[i1] = this_d_data[i];
        s->rowidx->data[i1] = this_rowidx_data[i];
      }
      if (nd - 1 >= 0) {
        numalloc += nd;
      }
      s->colidx->data[cidx + 1] = s->colidx->data[cidx] + nd;
    }
  }
}

/*
 * File trailer for sparse1.c
 *
 * [EOF]
 */
