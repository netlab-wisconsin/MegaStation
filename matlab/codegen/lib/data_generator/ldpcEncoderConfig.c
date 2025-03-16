/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: ldpcEncoderConfig.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "ldpcEncoderConfig.h"
#include "combineVectorElements.h"
#include "data_generator_emxutil.h"
#include "data_generator_rtwutil.h"
#include "data_generator_types.h"
#include "eml_setop.h"
#include "find.h"
#include "locBsearch.h"
#include "rt_nonfinite.h"
#include "sort.h"
#include "sparse.h"
#include "sparse1.h"
#include "tril.h"
#include "triu.h"

/* Function Declarations */
static void ConvertMatrixFormat(const emxArray_boolean_T *X_d,
                                const emxArray_int32_T *X_colidx,
                                const emxArray_int32_T *X_rowidx, int X_n,
                                emxArray_int32_T *RowIndices,
                                emxArray_int32_T *RowStartLoc,
                                emxArray_int32_T *ColumnSum);

static int gf2factorize(const emxArray_boolean_T *X_d,
                        const emxArray_int32_T *X_colidx,
                        const emxArray_int32_T *X_rowidx, int X_m, int X_n,
                        emxArray_boolean_T *A_d, emxArray_int32_T *A_colidx,
                        emxArray_int32_T *A_rowidx, emxArray_boolean_T *B_d,
                        emxArray_int32_T *B_colidx, emxArray_int32_T *B_rowidx,
                        emxArray_real_T *chosen_pivots, int *A_n, int *B_n,
                        boolean_T *invertible);

static double isfulldiagtriangular(const emxArray_int32_T *X_colidx,
                                   const emxArray_int32_T *X_rowidx, int X_m,
                                   int X_n);

/* Function Definitions */
/*
 * Arguments    : const emxArray_boolean_T *X_d
 *                const emxArray_int32_T *X_colidx
 *                const emxArray_int32_T *X_rowidx
 *                int X_n
 *                emxArray_int32_T *RowIndices
 *                emxArray_int32_T *RowStartLoc
 *                emxArray_int32_T *ColumnSum
 * Return Type  : void
 */
static void ConvertMatrixFormat(const emxArray_boolean_T *X_d,
                                const emxArray_int32_T *X_colidx,
                                const emxArray_int32_T *X_rowidx, int X_n,
                                emxArray_int32_T *RowIndices,
                                emxArray_int32_T *RowStartLoc,
                                emxArray_int32_T *ColumnSum)
{
  emxArray_int32_T *expl_temp;
  emxArray_int32_T *y_colidx;
  emxArray_real_T *CumulativeSum;
  emxArray_real_T *y_d;
  double d;
  double *CumulativeSum_data;
  double *y_d_data;
  const int *X_colidx_data;
  const int *X_rowidx_data;
  int c;
  int col;
  int i;
  int idx;
  int nx_tmp;
  int *RowIndices_data;
  RowIndices_data = RowIndices->data;
  X_rowidx_data = X_rowidx->data;
  X_colidx_data = X_colidx->data;
  nx_tmp = X_colidx_data[X_colidx->size[0] - 1] - 1;
  if (nx_tmp == 0) {
    RowIndices->size[0] = 0;
  } else {
    i = RowIndices->size[0];
    RowIndices->size[0] = nx_tmp;
    emxEnsureCapacity_int32_T(RowIndices, i);
    RowIndices_data = RowIndices->data;
    for (idx = 0; idx < nx_tmp; idx++) {
      RowIndices_data[idx] = X_rowidx_data[idx];
    }
    idx = 0;
    col = 1;
    while (idx < nx_tmp) {
      if (idx == X_colidx_data[col] - 1) {
        col++;
      } else {
        idx++;
      }
    }
    if (nx_tmp == 1) {
      if (idx == 0) {
        RowIndices->size[0] = 0;
      }
    } else {
      i = RowIndices->size[0];
      RowIndices->size[0] = idx;
      emxEnsureCapacity_int32_T(RowIndices, i);
      RowIndices_data = RowIndices->data;
    }
  }
  if (RowIndices->size[0] != 0) {
    col = RowIndices->size[0];
    for (i = 0; i < col; i++) {
      d = (double)RowIndices_data[i] - 1.0;
      if (d >= -2.147483648E+9) {
        nx_tmp = (int)d;
      } else {
        nx_tmp = MIN_int32_T;
      }
      RowIndices_data[i] = nx_tmp;
    }
  } else {
    i = RowIndices->size[0];
    RowIndices->size[0] = 1;
    emxEnsureCapacity_int32_T(RowIndices, i);
    RowIndices_data = RowIndices->data;
    RowIndices_data[0] = 0;
  }
  emxInit_real_T(&y_d, 1);
  emxInit_int32_T(&y_colidx, 1);
  emxInit_int32_T(&expl_temp, 1);
  col = combineVectorElements(X_d, X_colidx, X_n, y_d, y_colidx, expl_temp);
  RowIndices_data = y_colidx->data;
  y_d_data = y_d->data;
  emxFree_int32_T(&expl_temp);
  emxInit_real_T(&CumulativeSum, 2);
  i = CumulativeSum->size[0] * CumulativeSum->size[1];
  CumulativeSum->size[0] = 1;
  CumulativeSum->size[1] = col;
  emxEnsureCapacity_real_T(CumulativeSum, i);
  CumulativeSum_data = CumulativeSum->data;
  for (i = 0; i < col; i++) {
    CumulativeSum_data[i] = 0.0;
  }
  i = (unsigned short)col;
  for (c = 0; c < i; c++) {
    col = RowIndices_data[c + 1] - 1;
    nx_tmp = RowIndices_data[c];
    for (idx = nx_tmp; idx <= col; idx++) {
      CumulativeSum_data[c] = y_d_data[idx - 1];
    }
  }
  emxFree_int32_T(&y_colidx);
  emxFree_real_T(&y_d);
  i = ColumnSum->size[0] * ColumnSum->size[1];
  ColumnSum->size[0] = 1;
  ColumnSum->size[1] = CumulativeSum->size[1];
  emxEnsureCapacity_int32_T(ColumnSum, i);
  RowIndices_data = ColumnSum->data;
  col = CumulativeSum->size[1];
  for (i = 0; i < col; i++) {
    d = rt_roundd_snf(CumulativeSum_data[i]);
    if (d < 2.147483648E+9) {
      if (d >= -2.147483648E+9) {
        nx_tmp = (int)d;
      } else {
        nx_tmp = MIN_int32_T;
      }
    } else if (d >= 2.147483648E+9) {
      nx_tmp = MAX_int32_T;
    } else {
      nx_tmp = 0;
    }
    RowIndices_data[i] = nx_tmp;
  }
  i = CumulativeSum->size[0] * CumulativeSum->size[1];
  CumulativeSum->size[0] = 1;
  CumulativeSum->size[1] = ColumnSum->size[1];
  emxEnsureCapacity_real_T(CumulativeSum, i);
  CumulativeSum_data = CumulativeSum->data;
  col = ColumnSum->size[1];
  for (i = 0; i < col; i++) {
    CumulativeSum_data[i] = RowIndices_data[i];
  }
  if (CumulativeSum->size[1] != 0) {
    i = CumulativeSum->size[1];
    for (col = 0; col <= i - 2; col++) {
      CumulativeSum_data[col + 1] += CumulativeSum_data[col];
    }
  }
  if (CumulativeSum->size[1] - 1 < 1) {
    col = 1;
  } else {
    col = CumulativeSum->size[1];
  }
  i = RowStartLoc->size[0] * RowStartLoc->size[1];
  RowStartLoc->size[0] = 1;
  RowStartLoc->size[1] = col;
  emxEnsureCapacity_int32_T(RowStartLoc, i);
  RowIndices_data = RowStartLoc->data;
  RowIndices_data[0] = 0;
  for (i = 0; i <= col - 2; i++) {
    d = CumulativeSum_data[i];
    if (d < 2.147483648E+9) {
      if (d >= -2.147483648E+9) {
        nx_tmp = (int)d;
      } else {
        nx_tmp = MIN_int32_T;
      }
    } else if (d >= 2.147483648E+9) {
      nx_tmp = MAX_int32_T;
    } else {
      nx_tmp = 0;
    }
    RowIndices_data[i + 1] = nx_tmp;
  }
  emxFree_real_T(&CumulativeSum);
}

/*
 * Arguments    : const emxArray_boolean_T *X_d
 *                const emxArray_int32_T *X_colidx
 *                const emxArray_int32_T *X_rowidx
 *                int X_m
 *                int X_n
 *                emxArray_boolean_T *A_d
 *                emxArray_int32_T *A_colidx
 *                emxArray_int32_T *A_rowidx
 *                emxArray_boolean_T *B_d
 *                emxArray_int32_T *B_colidx
 *                emxArray_int32_T *B_rowidx
 *                emxArray_real_T *chosen_pivots
 *                int *A_n
 *                int *B_n
 *                boolean_T *invertible
 * Return Type  : int
 */
static int gf2factorize(const emxArray_boolean_T *X_d,
                        const emxArray_int32_T *X_colidx,
                        const emxArray_int32_T *X_rowidx, int X_m, int X_n,
                        emxArray_boolean_T *A_d, emxArray_int32_T *A_colidx,
                        emxArray_int32_T *A_rowidx, emxArray_boolean_T *B_d,
                        emxArray_int32_T *B_colidx, emxArray_int32_T *B_rowidx,
                        emxArray_real_T *chosen_pivots, int *A_n, int *B_n,
                        boolean_T *invertible)
{
  emxArray_boolean_T b_candidate_rows_data;
  emxArray_boolean_T c_candidate_rows_data;
  emxArray_boolean_T *Y1;
  emxArray_boolean_T *Y2;
  emxArray_boolean_T *b_Y2;
  emxArray_boolean_T *s_d;
  emxArray_boolean_T *tmpd;
  emxArray_int16_T *b_y;
  emxArray_int32_T *ia;
  emxArray_int32_T *ii;
  emxArray_int32_T *r2;
  emxArray_int32_T *s_colidx;
  emxArray_int32_T *s_rowidx;
  emxArray_int8_T *r;
  emxArray_real_T *c_y;
  emxArray_real_T *candidate_rows;
  emxArray_real_T *r4;
  emxArray_uint32_T *r3;
  emxArray_uint32_T *y;
  double *chosen_pivots_data;
  double *d_candidate_rows_data;
  const int *X_colidx_data;
  const int *X_rowidx_data;
  int A_m;
  int c;
  int candidate_rows_size;
  int exitg1;
  int i;
  int i1;
  int idx;
  int numalloc;
  int nzs_tmp_tmp;
  int ridx;
  int *ii_data;
  unsigned int *r5;
  int *s_rowidx_data;
  unsigned int *y_data;
  short *b_y_data;
  signed char *r1;
  boolean_T candidate_rows_data[17664];
  const boolean_T *X_d_data;
  boolean_T val;
  boolean_T *Y1_data;
  boolean_T *Y2_data;
  boolean_T *tmpd_data;
  X_rowidx_data = X_rowidx->data;
  X_colidx_data = X_colidx->data;
  X_d_data = X_d->data;
  emxInit_int8_T(&r, 2);
  i = r->size[0] * r->size[1];
  r->size[0] = X_m;
  r->size[1] = X_m;
  emxEnsureCapacity_int8_T(r, i);
  r1 = r->data;
  ridx = X_m * X_m;
  for (i = 0; i < ridx; i++) {
    r1[i] = 0;
  }
  i = (unsigned short)X_m;
  for (numalloc = 0; numalloc < i; numalloc++) {
    r1[numalloc + r->size[0] * numalloc] = 1;
  }
  emxInit_boolean_T(&Y1, 2);
  i = Y1->size[0] * Y1->size[1];
  Y1->size[0] = r->size[0];
  Y1->size[1] = r->size[1];
  emxEnsureCapacity_boolean_T(Y1, i);
  Y1_data = Y1->data;
  for (i = 0; i < ridx; i++) {
    Y1_data[i] = (r1[i] != 0);
  }
  emxFree_int8_T(&r);
  nzs_tmp_tmp = X_colidx_data[X_colidx->size[0] - 1];
  if (nzs_tmp_tmp - 1 < 1) {
    idx = 0;
  } else {
    idx = nzs_tmp_tmp - 1;
  }
  emxInit_boolean_T(&tmpd, 1);
  i = tmpd->size[0];
  tmpd->size[0] = idx;
  emxEnsureCapacity_boolean_T(tmpd, i);
  tmpd_data = tmpd->data;
  for (i = 0; i < idx; i++) {
    tmpd_data[i] = X_d_data[i];
  }
  emxInit_boolean_T(&s_d, 1);
  emxInit_int32_T(&s_colidx, 1);
  emxInit_int32_T(&s_rowidx, 1);
  if (nzs_tmp_tmp - 1 >= 1) {
    numalloc = nzs_tmp_tmp - 2;
  } else {
    numalloc = 0;
  }
  i = s_d->size[0];
  s_d->size[0] = numalloc + 1;
  emxEnsureCapacity_boolean_T(s_d, i);
  Y2_data = s_d->data;
  for (i = 0; i <= numalloc; i++) {
    Y2_data[i] = false;
  }
  i = s_colidx->size[0];
  s_colidx->size[0] = X_n + 1;
  emxEnsureCapacity_int32_T(s_colidx, i);
  ii_data = s_colidx->data;
  for (i = 0; i <= X_n; i++) {
    ii_data[i] = 0;
  }
  ii_data[0] = 1;
  i = s_rowidx->size[0];
  s_rowidx->size[0] = numalloc + 1;
  emxEnsureCapacity_int32_T(s_rowidx, i);
  s_rowidx_data = s_rowidx->data;
  for (i = 0; i <= numalloc; i++) {
    s_rowidx_data[i] = 0;
  }
  i = (unsigned short)X_n;
  for (c = 0; c < i; c++) {
    ii_data[c + 1] = 1;
  }
  idx = 0;
  i = s_colidx->size[0];
  for (c = 0; c <= i - 2; c++) {
    ridx = ii_data[c];
    ii_data[c] = idx + 1;
    do {
      exitg1 = 0;
      i1 = ii_data[c + 1];
      if (ridx < i1) {
        val = false;
        while (ridx < i1) {
          if (val || Y2_data[ridx - 1]) {
            val = true;
          }
          ridx++;
        }
        if (val) {
          Y2_data[idx] = true;
          s_rowidx_data[idx] = 0;
          idx++;
        }
      } else {
        exitg1 = 1;
      }
    } while (exitg1 == 0);
  }
  if (nzs_tmp_tmp - 1 < 1) {
    idx = 1;
  } else {
    idx = nzs_tmp_tmp;
  }
  for (i = 0; i <= idx - 2; i++) {
    s_rowidx_data[i] = X_rowidx_data[i];
  }
  i = s_colidx->size[0];
  s_colidx->size[0] = X_colidx->size[0];
  emxEnsureCapacity_int32_T(s_colidx, i);
  ii_data = s_colidx->data;
  idx = X_colidx->size[0];
  for (i = 0; i < idx; i++) {
    ii_data[i] = X_colidx_data[i];
  }
  for (numalloc = 0; numalloc <= nzs_tmp_tmp - 2; numalloc++) {
    Y2_data[numalloc] = tmpd_data[numalloc];
  }
  emxFree_boolean_T(&tmpd);
  idx = 1;
  i = X_colidx->size[0];
  for (c = 0; c <= i - 2; c++) {
    ridx = ii_data[c];
    ii_data[c] = idx;
    while (ridx < ii_data[c + 1]) {
      numalloc = s_rowidx_data[ridx - 1];
      val = Y2_data[ridx - 1];
      ridx++;
      if (val) {
        Y2_data[idx - 1] = true;
        s_rowidx_data[idx - 1] = numalloc;
        idx++;
      }
    }
  }
  ii_data[s_colidx->size[0] - 1] = idx;
  emxInit_boolean_T(&Y2, 2);
  sparse_full(s_d, s_colidx, s_rowidx, X_m, X_n, Y2);
  Y2_data = Y2->data;
  emxFree_boolean_T(&s_d);
  i = chosen_pivots->size[0];
  chosen_pivots->size[0] = X_m;
  emxEnsureCapacity_real_T(chosen_pivots, i);
  chosen_pivots_data = chosen_pivots->data;
  for (i = 0; i < X_m; i++) {
    chosen_pivots_data[i] = 0.0;
  }
  *invertible = true;
  nzs_tmp_tmp = 0;
  emxInit_real_T(&candidate_rows, 1);
  emxInit_int32_T(&r2, 2);
  emxInit_int32_T(&ii, 2);
  emxInit_uint32_T(&y, 2);
  y_data = y->data;
  emxInit_int16_T(&b_y);
  b_y_data = b_y->data;
  emxInit_int32_T(&ia, 1);
  emxInit_uint32_T(&r3, 1);
  emxInit_real_T(&r4, 1);
  emxInit_real_T(&c_y, 1);
  emxInit_boolean_T(&b_Y2, 2);
  do {
    exitg1 = 0;
    if (nzs_tmp_tmp <= X_m - 1) {
      candidate_rows_size = Y2->size[0];
      idx = Y2->size[0];
      for (i = 0; i < idx; i++) {
        candidate_rows_data[i] = Y2_data[i + Y2->size[0] * nzs_tmp_tmp];
      }
      if (((double)nzs_tmp_tmp + 1.0) - 1.0 < 1.0) {
        ridx = 0;
      } else {
        ridx = (int)(((double)nzs_tmp_tmp + 1.0) - 1.0);
      }
      i = r2->size[0] * r2->size[1];
      r2->size[0] = 1;
      r2->size[1] = ridx;
      emxEnsureCapacity_int32_T(r2, i);
      ii_data = r2->data;
      for (i = 0; i < ridx; i++) {
        ii_data[i] = (int)chosen_pivots_data[i];
      }
      idx = r2->size[1];
      for (i = 0; i < idx; i++) {
        candidate_rows_data[ii_data[i] - 1] = false;
      }
      b_candidate_rows_data.data = &candidate_rows_data[0];
      b_candidate_rows_data.size = &candidate_rows_size;
      b_candidate_rows_data.allocatedSize = 17664;
      b_candidate_rows_data.numDimensions = 1;
      b_candidate_rows_data.canFreeData = false;
      eml_find(&b_candidate_rows_data, s_colidx);
      ii_data = s_colidx->data;
      i = s_rowidx->size[0];
      s_rowidx->size[0] = s_colidx->size[0];
      emxEnsureCapacity_int32_T(s_rowidx, i);
      s_rowidx_data = s_rowidx->data;
      idx = s_colidx->size[0];
      for (i = 0; i < idx; i++) {
        s_rowidx_data[i] = ii_data[i];
      }
      c_candidate_rows_data.data = &candidate_rows_data[0];
      c_candidate_rows_data.size = &candidate_rows_size;
      c_candidate_rows_data.allocatedSize = 17664;
      c_candidate_rows_data.numDimensions = 1;
      c_candidate_rows_data.canFreeData = false;
      eml_find(&c_candidate_rows_data, s_colidx);
      ii_data = s_colidx->data;
      i = candidate_rows->size[0];
      candidate_rows->size[0] = s_colidx->size[0];
      emxEnsureCapacity_real_T(candidate_rows, i);
      d_candidate_rows_data = candidate_rows->data;
      idx = s_colidx->size[0];
      for (i = 0; i < idx; i++) {
        d_candidate_rows_data[i] = ii_data[i];
      }
      if (s_rowidx->size[0] == 0) {
        short tmp_data[17664];
        *invertible = false;
        A_m = b_sparse(Y1, A_d, A_colidx, A_rowidx, A_n);
        b_sparse(Y2, B_d, B_colidx, B_rowidx, B_n);
        if (X_m < 1) {
          y->size[0] = 1;
          y->size[1] = 0;
        } else {
          i = y->size[0] * y->size[1];
          y->size[0] = 1;
          y->size[1] = X_m;
          emxEnsureCapacity_uint32_T(y, i);
          y_data = y->data;
          idx = X_m - 1;
          for (i = 0; i <= idx; i++) {
            y_data[i] = (unsigned int)i + 1U;
          }
        }
        i = r4->size[0];
        r4->size[0] = ridx;
        emxEnsureCapacity_real_T(r4, i);
        d_candidate_rows_data = r4->data;
        for (i = 0; i < ridx; i++) {
          d_candidate_rows_data[i] = chosen_pivots_data[i];
        }
        sort(r4);
        i = c_y->size[0];
        c_y->size[0] = y->size[1];
        emxEnsureCapacity_real_T(c_y, i);
        d_candidate_rows_data = c_y->data;
        idx = y->size[1];
        for (i = 0; i < idx; i++) {
          d_candidate_rows_data[i] = y_data[i];
        }
        do_vectors(c_y, r4, candidate_rows, ia);
        d_candidate_rows_data = candidate_rows->data;
        if (candidate_rows->size[0] - 1 < 0) {
          b_y->size[0] = 1;
          b_y->size[1] = 0;
        } else {
          i = b_y->size[0] * b_y->size[1];
          b_y->size[0] = 1;
          b_y->size[1] = candidate_rows->size[0];
          emxEnsureCapacity_int16_T(b_y, i);
          b_y_data = b_y->data;
          idx = candidate_rows->size[0] - 1;
          for (i = 0; i <= idx; i++) {
            b_y_data[i] = (short)i;
          }
        }
        numalloc = b_y->size[1];
        idx = b_y->size[1];
        for (i = 0; i < idx; i++) {
          tmp_data[i] =
              (short)(((double)nzs_tmp_tmp + 1.0) + (double)b_y_data[i]);
        }
        for (i = 0; i < numalloc; i++) {
          chosen_pivots_data[tmp_data[i] - 1] = d_candidate_rows_data[i];
        }
        exitg1 = 1;
      } else {
        boolean_T x_data[43724];
        boolean_T exitg2;
        chosen_pivots_data[nzs_tmp_tmp] = d_candidate_rows_data[0];
        idx = Y2->size[1];
        for (i = 0; i < idx; i++) {
          x_data[i] =
              Y2_data[((int)d_candidate_rows_data[0] + Y2->size[0] * i) - 1];
        }
        ridx = Y2->size[1];
        idx = 0;
        i = ii->size[0] * ii->size[1];
        ii->size[0] = 1;
        ii->size[1] = Y2->size[1];
        emxEnsureCapacity_int32_T(ii, i);
        ii_data = ii->data;
        numalloc = 0;
        exitg2 = false;
        while ((!exitg2) && (numalloc <= ridx - 1)) {
          if (x_data[numalloc]) {
            idx++;
            ii_data[idx - 1] = numalloc + 1;
            if (idx >= ridx) {
              exitg2 = true;
            } else {
              numalloc++;
            }
          } else {
            numalloc++;
          }
        }
        if (Y2->size[1] == 1) {
          if (idx == 0) {
            ii->size[0] = 1;
            ii->size[1] = 0;
          }
        } else {
          i = ii->size[0] * ii->size[1];
          if (idx < 1) {
            ii->size[1] = 0;
          } else {
            ii->size[1] = idx;
          }
          emxEnsureCapacity_int32_T(ii, i);
          ii_data = ii->data;
        }
        if (s_rowidx->size[0] != 1) {
          i = r3->size[0];
          r3->size[0] = ii->size[1];
          emxEnsureCapacity_uint32_T(r3, i);
          r5 = r3->data;
          idx = ii->size[1];
          for (i = 0; i < idx; i++) {
            r5[i] = (unsigned int)ii_data[i];
          }
          i = b_Y2->size[0] * b_Y2->size[1];
          b_Y2->size[0] = s_rowidx->size[0] - 1;
          b_Y2->size[1] = r3->size[0];
          emxEnsureCapacity_boolean_T(b_Y2, i);
          tmpd_data = b_Y2->data;
          idx = r3->size[0];
          for (i = 0; i < idx; i++) {
            numalloc = s_rowidx->size[0];
            for (i1 = 0; i1 <= numalloc - 2; i1++) {
              tmpd_data[i1 + b_Y2->size[0] * i] =
                  !Y2_data[(s_rowidx_data[i1 + 1] +
                            Y2->size[0] * ((int)r5[i] - 1)) -
                           1];
            }
          }
          idx = b_Y2->size[1];
          for (i = 0; i < idx; i++) {
            numalloc = b_Y2->size[0];
            for (i1 = 0; i1 < numalloc; i1++) {
              Y2_data[((int)d_candidate_rows_data[i1 + 1] +
                       Y2->size[0] * ((int)r5[i] - 1)) -
                      1] = tmpd_data[i1 + b_Y2->size[0] * i];
            }
          }
        }
        if (s_rowidx->size[0] < 2) {
          i = 0;
          i1 = 0;
        } else {
          i = 1;
          i1 = candidate_rows->size[0];
        }
        idx = i1 - i;
        i1 = ia->size[0];
        ia->size[0] = idx;
        emxEnsureCapacity_int32_T(ia, i1);
        ii_data = ia->data;
        for (i1 = 0; i1 < idx; i1++) {
          ii_data[i1] = s_rowidx_data[i + i1];
        }
        idx = ia->size[0];
        for (i = 0; i < idx; i++) {
          Y1_data[(ii_data[i] +
                   Y1->size[0] * ((int)d_candidate_rows_data[0] - 1)) -
                  1] = true;
        }
        nzs_tmp_tmp++;
      }
    } else {
      A_m = b_sparse(Y1, A_d, A_colidx, A_rowidx, A_n);
      b_sparse(Y2, B_d, B_colidx, B_rowidx, B_n);
      exitg1 = 1;
    }
  } while (exitg1 == 0);
  emxFree_boolean_T(&b_Y2);
  emxFree_real_T(&c_y);
  emxFree_real_T(&r4);
  emxFree_int32_T(&s_rowidx);
  emxFree_int32_T(&s_colidx);
  emxFree_uint32_T(&r3);
  emxFree_int32_T(&ia);
  emxFree_int16_T(&b_y);
  emxFree_uint32_T(&y);
  emxFree_int32_T(&ii);
  emxFree_int32_T(&r2);
  emxFree_real_T(&candidate_rows);
  emxFree_boolean_T(&Y2);
  emxFree_boolean_T(&Y1);
  return A_m;
}

/*
 * Arguments    : const emxArray_int32_T *X_colidx
 *                const emxArray_int32_T *X_rowidx
 *                int X_m
 *                int X_n
 * Return Type  : double
 */
static double isfulldiagtriangular(const emxArray_int32_T *X_colidx,
                                   const emxArray_int32_T *X_rowidx, int X_m,
                                   int X_n)
{
  emxArray_int32_T *y_colidx;
  emxArray_int8_T *b_y_colidx;
  emxArray_int8_T *out_colidx;
  double shape;
  const int *X_colidx_data;
  const int *X_rowidx_data;
  int M;
  int col;
  int i;
  int toFill;
  int *y_colidx_data;
  signed char *b_y_colidx_data;
  signed char *out_colidx_data;
  boolean_T found;
  X_rowidx_data = X_rowidx->data;
  X_colidx_data = X_colidx->data;
  if (X_m == X_n) {
    M = X_n;
  } else if (X_n > X_m) {
    if (X_n - X_m < 0) {
      M = X_n;
    } else {
      M = X_m;
    }
  } else if (X_n - X_m > 0) {
    M = X_m;
  } else {
    M = X_n;
  }
  emxInit_int32_T(&y_colidx, 1);
  toFill = 1;
  i = (unsigned short)X_n;
  for (col = 0; col < i; col++) {
    sparse_locBsearch(X_rowidx, col + 1, X_colidx_data[col],
                      X_colidx_data[col + 1], &found);
    if (found) {
      toFill++;
    }
  }
  col = y_colidx->size[0];
  y_colidx->size[0] = 2;
  emxEnsureCapacity_int32_T(y_colidx, col);
  y_colidx_data = y_colidx->data;
  y_colidx_data[0] = 1;
  y_colidx_data[1] = toFill;
  emxInit_int8_T(&b_y_colidx, 1);
  col = b_y_colidx->size[0];
  b_y_colidx->size[0] = 2;
  emxEnsureCapacity_int8_T(b_y_colidx, col);
  b_y_colidx_data = b_y_colidx->data;
  b_y_colidx_data[0] = 1;
  if (M == 0) {
    b_y_colidx_data[1] = 2;
  } else if (y_colidx_data[1] - y_colidx_data[0] == M) {
    b_y_colidx_data[1] = 2;
  } else {
    b_y_colidx_data[1] = 1;
  }
  emxInit_int8_T(&out_colidx, 1);
  col = out_colidx->size[0];
  out_colidx->size[0] = 2;
  emxEnsureCapacity_int8_T(out_colidx, col);
  out_colidx_data = out_colidx->data;
  out_colidx_data[1] = 1;
  out_colidx_data[0] = 1;
  if (b_y_colidx_data[0] == b_y_colidx_data[1]) {
    out_colidx_data[1] = 2;
  }
  emxFree_int8_T(&b_y_colidx);
  found = false;
  toFill = out_colidx_data[1] - 1;
  col = out_colidx_data[0];
  emxFree_int8_T(&out_colidx);
  for (M = col; M <= toFill; M++) {
    found = true;
  }
  if (found) {
    shape = 0.0;
  } else {
    col = y_colidx->size[0];
    y_colidx->size[0] = X_n + 1;
    emxEnsureCapacity_int32_T(y_colidx, col);
    y_colidx_data = y_colidx->data;
    y_colidx_data[0] = 1;
    toFill = 1;
    for (col = 0; col < i; col++) {
      for (M = X_colidx_data[col]; M < X_colidx_data[col + 1]; M++) {
        if (X_rowidx_data[M - 1] >= col + 1) {
          toFill++;
        }
      }
      y_colidx_data[col + 1] = toFill;
    }
    toFill = y_colidx_data[y_colidx->size[0] - 1] - 1;
    if (toFill == X_colidx_data[X_colidx->size[0] - 1] - 1) {
      shape = 1.0;
    } else if (toFill == X_m) {
      shape = -1.0;
    } else {
      shape = 0.0;
    }
  }
  emxFree_int32_T(&y_colidx);
  return shape;
}

/*
 * Arguments    : const emxArray_boolean_T *obj_ParityCheckMatrix_d
 *                const emxArray_int32_T *obj_ParityCheckMatrix_colidx
 *                const emxArray_int32_T *obj_ParityCheckMatrix_rowidx
 *                int obj_ParityCheckMatrix_m
 *                int obj_ParityCheckMatrix_n
 *                emxArray_int32_T *c_derivedParams_MatrixL_RowIndi
 *                emxArray_int32_T *c_derivedParams_MatrixL_RowStar
 *                emxArray_int32_T *derivedParams_MatrixL_ColumnSum
 *                emxArray_int32_T *derivedParams_RowOrder
 *                emxArray_int32_T *c_derivedParams_MatrixA_RowIndi
 *                emxArray_int32_T *c_derivedParams_MatrixA_RowStar
 *                emxArray_int32_T *derivedParams_MatrixA_ColumnSum
 *                emxArray_int32_T *c_derivedParams_MatrixB_RowIndi
 *                emxArray_int32_T *c_derivedParams_MatrixB_RowStar
 *                emxArray_int32_T *derivedParams_MatrixB_ColumnSum
 * Return Type  : signed char
 */
signed char c_ldpcEncoderConfig_CalcDerived(
    const emxArray_boolean_T *obj_ParityCheckMatrix_d,
    const emxArray_int32_T *obj_ParityCheckMatrix_colidx,
    const emxArray_int32_T *obj_ParityCheckMatrix_rowidx,
    int obj_ParityCheckMatrix_m, int obj_ParityCheckMatrix_n,
    emxArray_int32_T *c_derivedParams_MatrixL_RowIndi,
    emxArray_int32_T *c_derivedParams_MatrixL_RowStar,
    emxArray_int32_T *derivedParams_MatrixL_ColumnSum,
    emxArray_int32_T *derivedParams_RowOrder,
    emxArray_int32_T *c_derivedParams_MatrixA_RowIndi,
    emxArray_int32_T *c_derivedParams_MatrixA_RowStar,
    emxArray_int32_T *derivedParams_MatrixA_ColumnSum,
    emxArray_int32_T *c_derivedParams_MatrixB_RowIndi,
    emxArray_int32_T *c_derivedParams_MatrixB_RowStar,
    emxArray_int32_T *derivedParams_MatrixB_ColumnSum)
{
  c_sparse expl_temp;
  emxArray_boolean_T *P_d;
  emxArray_boolean_T *Reversed_PB_d;
  emxArray_int32_T *P_colidx;
  emxArray_int32_T *P_rowidx;
  emxArray_int32_T *Reversed_PB_colidx;
  emxArray_int32_T *Reversed_PB_rowidx;
  emxArray_real_T *b_y;
  emxArray_real_T *c_y;
  emxArray_real_T *rowOrder;
  emxArray_real_T *y;
  double K;
  double *y_data;
  int PB_n;
  int P_n;
  int cidx;
  int i;
  int k;
  int ridx;
  int *P_colidx_data;
  int *Reversed_PB_colidx_data;
  int *Reversed_PB_rowidx_data;
  signed char derivedParams_EncodingMethod;
  boolean_T found;
  boolean_T *P_d_data;
  boolean_T *Reversed_PB_d_data;
  K = (double)obj_ParityCheckMatrix_n - (double)obj_ParityCheckMatrix_m;
  emxInit_real_T(&y, 2);
  if (obj_ParityCheckMatrix_n < K + 1.0) {
    y->size[0] = 1;
    y->size[1] = 0;
  } else {
    i = y->size[0] * y->size[1];
    y->size[0] = 1;
    k = (int)((double)obj_ParityCheckMatrix_n - (K + 1.0));
    y->size[1] = k + 1;
    emxEnsureCapacity_real_T(y, i);
    y_data = y->data;
    for (i = 0; i <= k; i++) {
      y_data[i] = (K + 1.0) + (double)i;
    }
  }
  emxInitStruct_sparse(&expl_temp);
  sparse_parenReference(obj_ParityCheckMatrix_d, obj_ParityCheckMatrix_colidx,
                        obj_ParityCheckMatrix_rowidx, obj_ParityCheckMatrix_m,
                        y, &expl_temp);
  emxFree_real_T(&y);
  PB_n = expl_temp.n;
  emxInit_boolean_T(&P_d, 1);
  emxInit_int32_T(&P_colidx, 1);
  emxInit_int32_T(&P_rowidx, 1);
  emxInit_real_T(&b_y, 2);
  switch ((int)isfulldiagtriangular(expl_temp.colidx, expl_temp.rowidx,
                                    expl_temp.m, expl_temp.n)) {
  case 1:
    derivedParams_EncodingMethod = 1;
    sparse_tril(expl_temp.d, expl_temp.colidx, expl_temp.rowidx, expl_temp.m,
                expl_temp.n, P_d, P_colidx, P_rowidx, &P_n);
    i = c_derivedParams_MatrixL_RowIndi->size[0];
    c_derivedParams_MatrixL_RowIndi->size[0] = 1;
    emxEnsureCapacity_int32_T(c_derivedParams_MatrixL_RowIndi, i);
    Reversed_PB_colidx_data = c_derivedParams_MatrixL_RowIndi->data;
    Reversed_PB_colidx_data[0] = 0;
    i = c_derivedParams_MatrixL_RowStar->size[0] *
        c_derivedParams_MatrixL_RowStar->size[1];
    c_derivedParams_MatrixL_RowStar->size[0] = 1;
    c_derivedParams_MatrixL_RowStar->size[1] = 1;
    emxEnsureCapacity_int32_T(c_derivedParams_MatrixL_RowStar, i);
    Reversed_PB_colidx_data = c_derivedParams_MatrixL_RowStar->data;
    Reversed_PB_colidx_data[0] = 0;
    i = derivedParams_MatrixL_ColumnSum->size[0] *
        derivedParams_MatrixL_ColumnSum->size[1];
    derivedParams_MatrixL_ColumnSum->size[0] = 1;
    derivedParams_MatrixL_ColumnSum->size[1] = 1;
    emxEnsureCapacity_int32_T(derivedParams_MatrixL_ColumnSum, i);
    Reversed_PB_colidx_data = derivedParams_MatrixL_ColumnSum->data;
    Reversed_PB_colidx_data[0] = 0;
    i = derivedParams_RowOrder->size[0];
    derivedParams_RowOrder->size[0] = 1;
    emxEnsureCapacity_int32_T(derivedParams_RowOrder, i);
    Reversed_PB_colidx_data = derivedParams_RowOrder->data;
    Reversed_PB_colidx_data[0] = -2;
    if (K < 1.0) {
      b_y->size[0] = 1;
      b_y->size[1] = 0;
    } else {
      i = b_y->size[0] * b_y->size[1];
      b_y->size[0] = 1;
      b_y->size[1] = (int)(K - 1.0) + 1;
      emxEnsureCapacity_real_T(b_y, i);
      y_data = b_y->data;
      k = (int)(K - 1.0);
      for (i = 0; i <= k; i++) {
        y_data[i] = (double)i + 1.0;
      }
    }
    sparse_parenReference(obj_ParityCheckMatrix_d, obj_ParityCheckMatrix_colidx,
                          obj_ParityCheckMatrix_rowidx, obj_ParityCheckMatrix_m,
                          b_y, &expl_temp);
    ConvertMatrixFormat(expl_temp.d, expl_temp.colidx, expl_temp.rowidx,
                        expl_temp.n, c_derivedParams_MatrixA_RowIndi,
                        c_derivedParams_MatrixA_RowStar,
                        derivedParams_MatrixA_ColumnSum);
    ConvertMatrixFormat(
        P_d, P_colidx, P_rowidx, P_n, c_derivedParams_MatrixB_RowIndi,
        c_derivedParams_MatrixB_RowStar, derivedParams_MatrixB_ColumnSum);
    break;
  case -1:
    derivedParams_EncodingMethod = -1;
    sparse_triu(expl_temp.d, expl_temp.colidx, expl_temp.rowidx, expl_temp.m,
                expl_temp.n, P_d, P_colidx, P_rowidx, &P_n);
    i = c_derivedParams_MatrixL_RowIndi->size[0];
    c_derivedParams_MatrixL_RowIndi->size[0] = 1;
    emxEnsureCapacity_int32_T(c_derivedParams_MatrixL_RowIndi, i);
    Reversed_PB_colidx_data = c_derivedParams_MatrixL_RowIndi->data;
    Reversed_PB_colidx_data[0] = 0;
    i = c_derivedParams_MatrixL_RowStar->size[0] *
        c_derivedParams_MatrixL_RowStar->size[1];
    c_derivedParams_MatrixL_RowStar->size[0] = 1;
    c_derivedParams_MatrixL_RowStar->size[1] = 1;
    emxEnsureCapacity_int32_T(c_derivedParams_MatrixL_RowStar, i);
    Reversed_PB_colidx_data = c_derivedParams_MatrixL_RowStar->data;
    Reversed_PB_colidx_data[0] = 0;
    i = derivedParams_MatrixL_ColumnSum->size[0] *
        derivedParams_MatrixL_ColumnSum->size[1];
    derivedParams_MatrixL_ColumnSum->size[0] = 1;
    derivedParams_MatrixL_ColumnSum->size[1] = 1;
    emxEnsureCapacity_int32_T(derivedParams_MatrixL_ColumnSum, i);
    Reversed_PB_colidx_data = derivedParams_MatrixL_ColumnSum->data;
    Reversed_PB_colidx_data[0] = 0;
    i = derivedParams_RowOrder->size[0];
    derivedParams_RowOrder->size[0] = 1;
    emxEnsureCapacity_int32_T(derivedParams_RowOrder, i);
    Reversed_PB_colidx_data = derivedParams_RowOrder->data;
    Reversed_PB_colidx_data[0] = -2;
    if (K < 1.0) {
      b_y->size[0] = 1;
      b_y->size[1] = 0;
    } else {
      i = b_y->size[0] * b_y->size[1];
      b_y->size[0] = 1;
      b_y->size[1] = (int)(K - 1.0) + 1;
      emxEnsureCapacity_real_T(b_y, i);
      y_data = b_y->data;
      k = (int)(K - 1.0);
      for (i = 0; i <= k; i++) {
        y_data[i] = (double)i + 1.0;
      }
    }
    sparse_parenReference(obj_ParityCheckMatrix_d, obj_ParityCheckMatrix_colidx,
                          obj_ParityCheckMatrix_rowidx, obj_ParityCheckMatrix_m,
                          b_y, &expl_temp);
    ConvertMatrixFormat(expl_temp.d, expl_temp.colidx, expl_temp.rowidx,
                        expl_temp.n, c_derivedParams_MatrixA_RowIndi,
                        c_derivedParams_MatrixA_RowStar,
                        derivedParams_MatrixA_ColumnSum);
    ConvertMatrixFormat(
        P_d, P_colidx, P_rowidx, P_n, c_derivedParams_MatrixB_RowIndi,
        c_derivedParams_MatrixB_RowStar, derivedParams_MatrixB_ColumnSum);
    break;
  default: {
    int Reversed_PB_m;
    int colNnz;
    int i1;
    int sm;
    emxInit_real_T(&c_y, 2);
    y_data = c_y->data;
    if (expl_temp.m < 1) {
      c_y->size[0] = 1;
      c_y->size[1] = 0;
    } else {
      i = c_y->size[0] * c_y->size[1];
      c_y->size[0] = 1;
      c_y->size[1] = expl_temp.m;
      emxEnsureCapacity_real_T(c_y, i);
      y_data = c_y->data;
      k = expl_temp.m - 1;
      for (i = 0; i <= k; i++) {
        y_data[i] = expl_temp.m - i;
      }
    }
    sm = c_y->size[1];
    emxInit_boolean_T(&Reversed_PB_d, 1);
    Reversed_PB_d->size[0] = 0;
    emxInit_int32_T(&Reversed_PB_rowidx, 1);
    Reversed_PB_rowidx->size[0] = 0;
    emxInit_int32_T(&Reversed_PB_colidx, 1);
    i = Reversed_PB_colidx->size[0];
    Reversed_PB_colidx->size[0] = expl_temp.n + 1;
    emxEnsureCapacity_int32_T(Reversed_PB_colidx, i);
    Reversed_PB_colidx_data = Reversed_PB_colidx->data;
    for (i = 0; i <= PB_n; i++) {
      Reversed_PB_colidx_data[i] = 0;
    }
    Reversed_PB_colidx_data[0] = 1;
    colNnz = 1;
    k = 0;
    i = (unsigned short)expl_temp.n;
    for (cidx = 0; cidx < i; cidx++) {
      for (ridx = 0; ridx < sm; ridx++) {
        PB_n = sparse_locBsearch(expl_temp.rowidx, (int)y_data[ridx],
                                 expl_temp.colidx->data[cidx],
                                 expl_temp.colidx->data[cidx + 1], &found);
        if (found) {
          i1 = Reversed_PB_d->size[0];
          Reversed_PB_m = Reversed_PB_d->size[0];
          Reversed_PB_d->size[0]++;
          emxEnsureCapacity_boolean_T(Reversed_PB_d, Reversed_PB_m);
          Reversed_PB_d_data = Reversed_PB_d->data;
          found = expl_temp.d->data[PB_n - 1];
          Reversed_PB_d_data[i1] = found;
          i1 = Reversed_PB_rowidx->size[0];
          Reversed_PB_m = Reversed_PB_rowidx->size[0];
          Reversed_PB_rowidx->size[0]++;
          emxEnsureCapacity_int32_T(Reversed_PB_rowidx, Reversed_PB_m);
          Reversed_PB_rowidx_data = Reversed_PB_rowidx->data;
          Reversed_PB_rowidx_data[i1] = ridx + 1;
          Reversed_PB_d_data[k] = found;
          Reversed_PB_rowidx_data[k] = ridx + 1;
          k++;
          colNnz++;
        }
      }
      Reversed_PB_colidx_data[cidx + 1] = colNnz;
    }
    if (Reversed_PB_colidx_data[Reversed_PB_colidx->size[0] - 1] - 1 == 0) {
      i = Reversed_PB_rowidx->size[0];
      Reversed_PB_rowidx->size[0] = 1;
      emxEnsureCapacity_int32_T(Reversed_PB_rowidx, i);
      Reversed_PB_rowidx_data = Reversed_PB_rowidx->data;
      Reversed_PB_rowidx_data[0] = 1;
      i = Reversed_PB_d->size[0];
      Reversed_PB_d->size[0] = 1;
      emxEnsureCapacity_boolean_T(Reversed_PB_d, i);
      Reversed_PB_d_data = Reversed_PB_d->data;
      Reversed_PB_d_data[0] = false;
    }
    Reversed_PB_m = c_y->size[1];
    switch ((int)isfulldiagtriangular(Reversed_PB_colidx, Reversed_PB_rowidx,
                                      c_y->size[1], expl_temp.n)) {
    case 1: {
      double a;
      a = (double)obj_ParityCheckMatrix_n - K;
      if (a < 1.0) {
        c_y->size[0] = 1;
        c_y->size[1] = 0;
      } else {
        i = c_y->size[0] * c_y->size[1];
        c_y->size[0] = 1;
        c_y->size[1] = (int)-(1.0 - a) + 1;
        emxEnsureCapacity_real_T(c_y, i);
        y_data = c_y->data;
        k = (int)-(1.0 - a);
        for (i = 0; i <= k; i++) {
          y_data[i] = a - (double)i;
        }
      }
      derivedParams_EncodingMethod = 1;
      sparse_tril(Reversed_PB_d, Reversed_PB_colidx, Reversed_PB_rowidx,
                  Reversed_PB_m, expl_temp.n, P_d, P_colidx, P_rowidx, &P_n);
      i = c_derivedParams_MatrixL_RowIndi->size[0];
      c_derivedParams_MatrixL_RowIndi->size[0] = 1;
      emxEnsureCapacity_int32_T(c_derivedParams_MatrixL_RowIndi, i);
      Reversed_PB_colidx_data = c_derivedParams_MatrixL_RowIndi->data;
      Reversed_PB_colidx_data[0] = 0;
      i = c_derivedParams_MatrixL_RowStar->size[0] *
          c_derivedParams_MatrixL_RowStar->size[1];
      c_derivedParams_MatrixL_RowStar->size[0] = 1;
      c_derivedParams_MatrixL_RowStar->size[1] = 1;
      emxEnsureCapacity_int32_T(c_derivedParams_MatrixL_RowStar, i);
      Reversed_PB_colidx_data = c_derivedParams_MatrixL_RowStar->data;
      Reversed_PB_colidx_data[0] = 0;
      i = derivedParams_MatrixL_ColumnSum->size[0] *
          derivedParams_MatrixL_ColumnSum->size[1];
      derivedParams_MatrixL_ColumnSum->size[0] = 1;
      derivedParams_MatrixL_ColumnSum->size[1] = 1;
      emxEnsureCapacity_int32_T(derivedParams_MatrixL_ColumnSum, i);
      Reversed_PB_colidx_data = derivedParams_MatrixL_ColumnSum->data;
      Reversed_PB_colidx_data[0] = 0;
      i = derivedParams_RowOrder->size[0];
      derivedParams_RowOrder->size[0] = c_y->size[1];
      emxEnsureCapacity_int32_T(derivedParams_RowOrder, i);
      Reversed_PB_colidx_data = derivedParams_RowOrder->data;
      k = c_y->size[1];
      for (i = 0; i < k; i++) {
        a = y_data[i] - 1.0;
        if (a < 2.147483648E+9) {
          i1 = (int)a;
        } else {
          i1 = MAX_int32_T;
        }
        Reversed_PB_colidx_data[i] = i1;
      }
      if (K < 1.0) {
        b_y->size[0] = 1;
        b_y->size[1] = 0;
      } else {
        i = b_y->size[0] * b_y->size[1];
        b_y->size[0] = 1;
        b_y->size[1] = (int)(K - 1.0) + 1;
        emxEnsureCapacity_real_T(b_y, i);
        y_data = b_y->data;
        k = (int)(K - 1.0);
        for (i = 0; i <= k; i++) {
          y_data[i] = (double)i + 1.0;
        }
      }
      sparse_parenReference(obj_ParityCheckMatrix_d,
                            obj_ParityCheckMatrix_colidx,
                            obj_ParityCheckMatrix_rowidx,
                            obj_ParityCheckMatrix_m, b_y, &expl_temp);
      ConvertMatrixFormat(expl_temp.d, expl_temp.colidx, expl_temp.rowidx,
                          expl_temp.n, c_derivedParams_MatrixA_RowIndi,
                          c_derivedParams_MatrixA_RowStar,
                          derivedParams_MatrixA_ColumnSum);
      ConvertMatrixFormat(
          P_d, P_colidx, P_rowidx, P_n, c_derivedParams_MatrixB_RowIndi,
          c_derivedParams_MatrixB_RowStar, derivedParams_MatrixB_ColumnSum);
    } break;
    case -1: {
      double a;
      a = (double)obj_ParityCheckMatrix_n - K;
      if (a < 1.0) {
        c_y->size[0] = 1;
        c_y->size[1] = 0;
      } else {
        i = c_y->size[0] * c_y->size[1];
        c_y->size[0] = 1;
        c_y->size[1] = (int)-(1.0 - a) + 1;
        emxEnsureCapacity_real_T(c_y, i);
        y_data = c_y->data;
        k = (int)-(1.0 - a);
        for (i = 0; i <= k; i++) {
          y_data[i] = a - (double)i;
        }
      }
      derivedParams_EncodingMethod = -1;
      sparse_triu(Reversed_PB_d, Reversed_PB_colidx, Reversed_PB_rowidx,
                  Reversed_PB_m, expl_temp.n, P_d, P_colidx, P_rowidx, &P_n);
      i = c_derivedParams_MatrixL_RowIndi->size[0];
      c_derivedParams_MatrixL_RowIndi->size[0] = 1;
      emxEnsureCapacity_int32_T(c_derivedParams_MatrixL_RowIndi, i);
      Reversed_PB_colidx_data = c_derivedParams_MatrixL_RowIndi->data;
      Reversed_PB_colidx_data[0] = 0;
      i = c_derivedParams_MatrixL_RowStar->size[0] *
          c_derivedParams_MatrixL_RowStar->size[1];
      c_derivedParams_MatrixL_RowStar->size[0] = 1;
      c_derivedParams_MatrixL_RowStar->size[1] = 1;
      emxEnsureCapacity_int32_T(c_derivedParams_MatrixL_RowStar, i);
      Reversed_PB_colidx_data = c_derivedParams_MatrixL_RowStar->data;
      Reversed_PB_colidx_data[0] = 0;
      i = derivedParams_MatrixL_ColumnSum->size[0] *
          derivedParams_MatrixL_ColumnSum->size[1];
      derivedParams_MatrixL_ColumnSum->size[0] = 1;
      derivedParams_MatrixL_ColumnSum->size[1] = 1;
      emxEnsureCapacity_int32_T(derivedParams_MatrixL_ColumnSum, i);
      Reversed_PB_colidx_data = derivedParams_MatrixL_ColumnSum->data;
      Reversed_PB_colidx_data[0] = 0;
      i = derivedParams_RowOrder->size[0];
      derivedParams_RowOrder->size[0] = c_y->size[1];
      emxEnsureCapacity_int32_T(derivedParams_RowOrder, i);
      Reversed_PB_colidx_data = derivedParams_RowOrder->data;
      k = c_y->size[1];
      for (i = 0; i < k; i++) {
        a = y_data[i] - 1.0;
        if (a < 2.147483648E+9) {
          i1 = (int)a;
        } else {
          i1 = MAX_int32_T;
        }
        Reversed_PB_colidx_data[i] = i1;
      }
      if (K < 1.0) {
        b_y->size[0] = 1;
        b_y->size[1] = 0;
      } else {
        i = b_y->size[0] * b_y->size[1];
        b_y->size[0] = 1;
        b_y->size[1] = (int)(K - 1.0) + 1;
        emxEnsureCapacity_real_T(b_y, i);
        y_data = b_y->data;
        k = (int)(K - 1.0);
        for (i = 0; i <= k; i++) {
          y_data[i] = (double)i + 1.0;
        }
      }
      sparse_parenReference(obj_ParityCheckMatrix_d,
                            obj_ParityCheckMatrix_colidx,
                            obj_ParityCheckMatrix_rowidx,
                            obj_ParityCheckMatrix_m, b_y, &expl_temp);
      ConvertMatrixFormat(expl_temp.d, expl_temp.colidx, expl_temp.rowidx,
                          expl_temp.n, c_derivedParams_MatrixA_RowIndi,
                          c_derivedParams_MatrixA_RowStar,
                          derivedParams_MatrixA_ColumnSum);
      ConvertMatrixFormat(
          P_d, P_colidx, P_rowidx, P_n, c_derivedParams_MatrixB_RowIndi,
          c_derivedParams_MatrixB_RowStar, derivedParams_MatrixB_ColumnSum);
    } break;
    default: {
      derivedParams_EncodingMethod = 0;
      emxInit_real_T(&rowOrder, 1);
      Reversed_PB_m = gf2factorize(
          expl_temp.d, expl_temp.colidx, expl_temp.rowidx, expl_temp.m,
          expl_temp.n, Reversed_PB_d, Reversed_PB_colidx, Reversed_PB_rowidx,
          P_d, P_colidx, P_rowidx, rowOrder, &k, &P_n, &found);
      y_data = rowOrder->data;
      P_colidx_data = P_colidx->data;
      P_d_data = P_d->data;
      if (found) {
        sparse_tril(Reversed_PB_d, Reversed_PB_colidx, Reversed_PB_rowidx,
                    Reversed_PB_m, k, expl_temp.d, expl_temp.colidx,
                    expl_temp.rowidx, &PB_n);
        ConvertMatrixFormat(expl_temp.d, expl_temp.colidx, expl_temp.rowidx,
                            PB_n, c_derivedParams_MatrixL_RowIndi,
                            c_derivedParams_MatrixL_RowStar,
                            derivedParams_MatrixL_ColumnSum);
        sm = rowOrder->size[0];
        Reversed_PB_d->size[0] = 0;
        Reversed_PB_rowidx->size[0] = 0;
        i = Reversed_PB_colidx->size[0];
        Reversed_PB_colidx->size[0] = P_n + 1;
        emxEnsureCapacity_int32_T(Reversed_PB_colidx, i);
        Reversed_PB_colidx_data = Reversed_PB_colidx->data;
        for (i = 0; i <= P_n; i++) {
          Reversed_PB_colidx_data[i] = 0;
        }
        Reversed_PB_colidx_data[0] = 1;
        colNnz = 1;
        k = 0;
        i = (unsigned short)P_n;
        for (cidx = 0; cidx < i; cidx++) {
          for (ridx = 0; ridx < sm; ridx++) {
            PB_n = sparse_locBsearch(P_rowidx, (int)y_data[ridx],
                                     P_colidx_data[cidx],
                                     P_colidx_data[cidx + 1], &found);
            if (found) {
              i1 = Reversed_PB_d->size[0];
              Reversed_PB_m = Reversed_PB_d->size[0];
              Reversed_PB_d->size[0]++;
              emxEnsureCapacity_boolean_T(Reversed_PB_d, Reversed_PB_m);
              Reversed_PB_d_data = Reversed_PB_d->data;
              found = P_d_data[PB_n - 1];
              Reversed_PB_d_data[i1] = found;
              i1 = Reversed_PB_rowidx->size[0];
              Reversed_PB_m = Reversed_PB_rowidx->size[0];
              Reversed_PB_rowidx->size[0]++;
              emxEnsureCapacity_int32_T(Reversed_PB_rowidx, Reversed_PB_m);
              Reversed_PB_rowidx_data = Reversed_PB_rowidx->data;
              Reversed_PB_rowidx_data[i1] = ridx + 1;
              Reversed_PB_d_data[k] = found;
              Reversed_PB_rowidx_data[k] = ridx + 1;
              k++;
              colNnz++;
            }
          }
          Reversed_PB_colidx_data[cidx + 1] = colNnz;
        }
        if (Reversed_PB_colidx_data[Reversed_PB_colidx->size[0] - 1] - 1 == 0) {
          i = Reversed_PB_rowidx->size[0];
          Reversed_PB_rowidx->size[0] = 1;
          emxEnsureCapacity_int32_T(Reversed_PB_rowidx, i);
          Reversed_PB_rowidx_data = Reversed_PB_rowidx->data;
          Reversed_PB_rowidx_data[0] = 1;
          i = Reversed_PB_d->size[0];
          Reversed_PB_d->size[0] = 1;
          emxEnsureCapacity_boolean_T(Reversed_PB_d, i);
          Reversed_PB_d_data = Reversed_PB_d->data;
          Reversed_PB_d_data[0] = false;
        }
        sparse_triu(Reversed_PB_d, Reversed_PB_colidx, Reversed_PB_rowidx,
                    rowOrder->size[0], P_n, P_d, P_colidx, P_rowidx, &P_n);
        i = derivedParams_RowOrder->size[0];
        derivedParams_RowOrder->size[0] = rowOrder->size[0];
        emxEnsureCapacity_int32_T(derivedParams_RowOrder, i);
        Reversed_PB_colidx_data = derivedParams_RowOrder->data;
        k = rowOrder->size[0];
        for (i = 0; i < k; i++) {
          double a;
          a = rt_roundd_snf(y_data[i] - 1.0);
          if (a < 2.147483648E+9) {
            if (a >= -2.147483648E+9) {
              i1 = (int)a;
            } else {
              i1 = MIN_int32_T;
            }
          } else if (a >= 2.147483648E+9) {
            i1 = MAX_int32_T;
          } else {
            i1 = 0;
          }
          Reversed_PB_colidx_data[i] = i1;
        }
        if (K < 1.0) {
          b_y->size[0] = 1;
          b_y->size[1] = 0;
        } else {
          i = b_y->size[0] * b_y->size[1];
          b_y->size[0] = 1;
          b_y->size[1] = (int)(K - 1.0) + 1;
          emxEnsureCapacity_real_T(b_y, i);
          y_data = b_y->data;
          k = (int)(K - 1.0);
          for (i = 0; i <= k; i++) {
            y_data[i] = (double)i + 1.0;
          }
        }
        sparse_parenReference(obj_ParityCheckMatrix_d,
                              obj_ParityCheckMatrix_colidx,
                              obj_ParityCheckMatrix_rowidx,
                              obj_ParityCheckMatrix_m, b_y, &expl_temp);
        ConvertMatrixFormat(expl_temp.d, expl_temp.colidx, expl_temp.rowidx,
                            expl_temp.n, c_derivedParams_MatrixA_RowIndi,
                            c_derivedParams_MatrixA_RowStar,
                            derivedParams_MatrixA_ColumnSum);
        ConvertMatrixFormat(
            P_d, P_colidx, P_rowidx, P_n, c_derivedParams_MatrixB_RowIndi,
            c_derivedParams_MatrixB_RowStar, derivedParams_MatrixB_ColumnSum);
      }
      emxFree_real_T(&rowOrder);
    } break;
    }
    emxFree_real_T(&c_y);
    emxFree_int32_T(&Reversed_PB_rowidx);
    emxFree_int32_T(&Reversed_PB_colidx);
    emxFree_boolean_T(&Reversed_PB_d);
  } break;
  }
  emxFreeStruct_sparse(&expl_temp);
  emxFree_real_T(&b_y);
  emxFree_int32_T(&P_rowidx);
  emxFree_int32_T(&P_colidx);
  emxFree_boolean_T(&P_d);
  return derivedParams_EncodingMethod;
}

/*
 * File trailer for ldpcEncoderConfig.c
 *
 * [EOF]
 */
