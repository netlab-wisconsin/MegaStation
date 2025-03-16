/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: sortIdx.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "sortIdx.h"
#include "data_generator_types.h"
#include "rt_nonfinite.h"

/* Function Declarations */
static void merge(emxArray_int32_T *idx, emxArray_real_T *x, int offset, int np,
                  int nq, emxArray_int32_T *iwork, emxArray_real_T *xwork);

/* Function Definitions */
/*
 * Arguments    : emxArray_int32_T *idx
 *                emxArray_real_T *x
 *                int offset
 *                int np
 *                int nq
 *                emxArray_int32_T *iwork
 *                emxArray_real_T *xwork
 * Return Type  : void
 */
static void merge(emxArray_int32_T *idx, emxArray_real_T *x, int offset, int np,
                  int nq, emxArray_int32_T *iwork, emxArray_real_T *xwork)
{
  double *x_data;
  double *xwork_data;
  int j;
  int *idx_data;
  int *iwork_data;
  xwork_data = xwork->data;
  iwork_data = iwork->data;
  x_data = x->data;
  idx_data = idx->data;
  if (nq != 0) {
    int iout;
    int n_tmp;
    int p;
    int q;
    n_tmp = np + nq;
    for (j = 0; j < n_tmp; j++) {
      iout = offset + j;
      iwork_data[j] = idx_data[iout];
      xwork_data[j] = x_data[iout];
    }
    p = 0;
    q = np;
    iout = offset - 1;
    int exitg1;
    do {
      exitg1 = 0;
      iout++;
      if (xwork_data[p] <= xwork_data[q]) {
        idx_data[iout] = iwork_data[p];
        x_data[iout] = xwork_data[p];
        if (p + 1 < np) {
          p++;
        } else {
          exitg1 = 1;
        }
      } else {
        idx_data[iout] = iwork_data[q];
        x_data[iout] = xwork_data[q];
        if (q + 1 < n_tmp) {
          q++;
        } else {
          q = iout - p;
          for (j = p + 1; j <= np; j++) {
            iout = q + j;
            idx_data[iout] = iwork_data[j - 1];
            x_data[iout] = xwork_data[j - 1];
          }
          exitg1 = 1;
        }
      }
    } while (exitg1 == 0);
  }
}

/*
 * Arguments    : emxArray_int32_T *idx
 *                emxArray_real_T *x
 *                int offset
 *                int n
 *                int preSortLevel
 *                emxArray_int32_T *iwork
 *                emxArray_real_T *xwork
 * Return Type  : void
 */
void merge_block(emxArray_int32_T *idx, emxArray_real_T *x, int offset, int n,
                 int preSortLevel, emxArray_int32_T *iwork,
                 emxArray_real_T *xwork)
{
  int bLen;
  int nPairs;
  int nTail;
  nPairs = n >> preSortLevel;
  bLen = 1 << preSortLevel;
  while (nPairs > 1) {
    int tailOffset;
    if ((nPairs & 1) != 0) {
      nPairs--;
      tailOffset = bLen * nPairs;
      nTail = n - tailOffset;
      if (nTail > bLen) {
        merge(idx, x, offset + tailOffset, bLen, nTail - bLen, iwork, xwork);
      }
    }
    tailOffset = bLen << 1;
    nPairs >>= 1;
    for (nTail = 0; nTail < nPairs; nTail++) {
      merge(idx, x, offset + nTail * tailOffset, bLen, bLen, iwork, xwork);
    }
    bLen = tailOffset;
  }
  if (n > bLen) {
    merge(idx, x, offset, bLen, n - bLen, iwork, xwork);
  }
}

/*
 * File trailer for sortIdx.c
 *
 * [EOF]
 */
