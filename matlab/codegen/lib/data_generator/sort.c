/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: sort.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "sort.h"
#include "data_generator_emxutil.h"
#include "data_generator_types.h"
#include "rt_nonfinite.h"
#include "sortIdx.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : emxArray_real_T *x
 * Return Type  : void
 */
void sort(emxArray_real_T *x)
{
  emxArray_int32_T *iidx;
  emxArray_int32_T *iwork;
  emxArray_real_T *vwork;
  emxArray_real_T *xwork;
  double *vwork_data;
  double *x_data;
  double *xwork_data;
  int b;
  int bLen;
  int b_b;
  int b_i;
  int dim;
  int i;
  int i1;
  int j;
  int k;
  int vlen;
  int vstride;
  int *iidx_data;
  int *iwork_data;
  x_data = x->data;
  dim = 0;
  if (x->size[0] != 1) {
    dim = -1;
  }
  if (dim + 2 <= 1) {
    i = x->size[0];
  } else {
    i = 1;
  }
  vlen = i - 1;
  emxInit_real_T(&vwork, 1);
  bLen = vwork->size[0];
  vwork->size[0] = i;
  emxEnsureCapacity_real_T(vwork, bLen);
  vwork_data = vwork->data;
  vstride = 1;
  for (k = 0; k <= dim; k++) {
    vstride *= x->size[0];
  }
  emxInit_int32_T(&iidx, 1);
  emxInit_int32_T(&iwork, 1);
  emxInit_real_T(&xwork, 1);
  for (b_i = 0; b_i < 1; b_i++) {
    for (j = 0; j < vstride; j++) {
      for (k = 0; k <= vlen; k++) {
        vwork_data[k] = x_data[j + k * vstride];
      }
      i = iidx->size[0];
      iidx->size[0] = vwork->size[0];
      emxEnsureCapacity_int32_T(iidx, i);
      iidx_data = iidx->data;
      dim = vwork->size[0];
      for (i = 0; i < dim; i++) {
        iidx_data[i] = 0;
      }
      if (vwork->size[0] != 0) {
        double x4[4];
        int bLen2;
        int i2;
        int i3;
        int i4;
        int iidx_tmp;
        int n;
        int wOffset_tmp;
        short idx4[4];
        n = vwork->size[0];
        x4[0] = 0.0;
        idx4[0] = 0;
        x4[1] = 0.0;
        idx4[1] = 0;
        x4[2] = 0.0;
        idx4[2] = 0;
        x4[3] = 0.0;
        idx4[3] = 0;
        i = iwork->size[0];
        iwork->size[0] = vwork->size[0];
        emxEnsureCapacity_int32_T(iwork, i);
        iwork_data = iwork->data;
        dim = vwork->size[0];
        i = xwork->size[0];
        xwork->size[0] = vwork->size[0];
        emxEnsureCapacity_real_T(xwork, i);
        xwork_data = xwork->data;
        for (i = 0; i < dim; i++) {
          iwork_data[i] = 0;
          xwork_data[i] = 0.0;
        }
        bLen2 = 0;
        dim = 0;
        for (k = 0; k < n; k++) {
          if (rtIsNaN(vwork_data[k])) {
            iidx_tmp = (n - bLen2) - 1;
            iidx_data[iidx_tmp] = k + 1;
            xwork_data[iidx_tmp] = vwork_data[k];
            bLen2++;
          } else {
            dim++;
            idx4[dim - 1] = (short)(k + 1);
            x4[dim - 1] = vwork_data[k];
            if (dim == 4) {
              double d;
              double d1;
              dim = k - bLen2;
              if (x4[0] <= x4[1]) {
                i1 = 1;
                i2 = 2;
              } else {
                i1 = 2;
                i2 = 1;
              }
              if (x4[2] <= x4[3]) {
                i3 = 3;
                i4 = 4;
              } else {
                i3 = 4;
                i4 = 3;
              }
              d = x4[i3 - 1];
              d1 = x4[i1 - 1];
              if (d1 <= d) {
                d1 = x4[i2 - 1];
                if (d1 <= d) {
                  i = i1;
                  bLen = i2;
                  i1 = i3;
                  i2 = i4;
                } else if (d1 <= x4[i4 - 1]) {
                  i = i1;
                  bLen = i3;
                  i1 = i2;
                  i2 = i4;
                } else {
                  i = i1;
                  bLen = i3;
                  i1 = i4;
                }
              } else {
                d = x4[i4 - 1];
                if (d1 <= d) {
                  if (x4[i2 - 1] <= d) {
                    i = i3;
                    bLen = i1;
                    i1 = i2;
                    i2 = i4;
                  } else {
                    i = i3;
                    bLen = i1;
                    i1 = i4;
                  }
                } else {
                  i = i3;
                  bLen = i4;
                }
              }
              iidx_data[dim - 3] = idx4[i - 1];
              iidx_data[dim - 2] = idx4[bLen - 1];
              iidx_data[dim - 1] = idx4[i1 - 1];
              iidx_data[dim] = idx4[i2 - 1];
              vwork_data[dim - 3] = x4[i - 1];
              vwork_data[dim - 2] = x4[bLen - 1];
              vwork_data[dim - 1] = x4[i1 - 1];
              vwork_data[dim] = x4[i2 - 1];
              dim = 0;
            }
          }
        }
        wOffset_tmp = vwork->size[0] - bLen2;
        if (dim > 0) {
          signed char perm[4];
          perm[1] = 0;
          perm[2] = 0;
          perm[3] = 0;
          if (dim == 1) {
            perm[0] = 1;
          } else if (dim == 2) {
            if (x4[0] <= x4[1]) {
              perm[0] = 1;
              perm[1] = 2;
            } else {
              perm[0] = 2;
              perm[1] = 1;
            }
          } else if (x4[0] <= x4[1]) {
            if (x4[1] <= x4[2]) {
              perm[0] = 1;
              perm[1] = 2;
              perm[2] = 3;
            } else if (x4[0] <= x4[2]) {
              perm[0] = 1;
              perm[1] = 3;
              perm[2] = 2;
            } else {
              perm[0] = 3;
              perm[1] = 1;
              perm[2] = 2;
            }
          } else if (x4[0] <= x4[2]) {
            perm[0] = 2;
            perm[1] = 1;
            perm[2] = 3;
          } else if (x4[1] <= x4[2]) {
            perm[0] = 2;
            perm[1] = 3;
            perm[2] = 1;
          } else {
            perm[0] = 3;
            perm[1] = 2;
            perm[2] = 1;
          }
          i = (unsigned char)dim;
          for (k = 0; k < i; k++) {
            iidx_tmp = (wOffset_tmp - dim) + k;
            bLen = perm[k];
            iidx_data[iidx_tmp] = idx4[bLen - 1];
            vwork_data[iidx_tmp] = x4[bLen - 1];
          }
        }
        dim = bLen2 >> 1;
        for (k = 0; k < dim; k++) {
          i1 = wOffset_tmp + k;
          i2 = iidx_data[i1];
          iidx_tmp = (n - k) - 1;
          iidx_data[i1] = iidx_data[iidx_tmp];
          iidx_data[iidx_tmp] = i2;
          vwork_data[i1] = xwork_data[iidx_tmp];
          vwork_data[iidx_tmp] = xwork_data[i1];
        }
        if ((bLen2 & 1) != 0) {
          dim += wOffset_tmp;
          vwork_data[dim] = xwork_data[dim];
        }
        dim = 2;
        if (wOffset_tmp > 1) {
          if (vwork->size[0] >= 256) {
            n = wOffset_tmp >> 8;
            if (n > 0) {
              for (b = 0; b < n; b++) {
                double b_xwork[256];
                short b_iwork[256];
                i4 = (b << 8) - 1;
                for (b_b = 0; b_b < 6; b_b++) {
                  bLen = 1 << (b_b + 2);
                  bLen2 = bLen << 1;
                  i = 256 >> (b_b + 3);
                  for (k = 0; k < i; k++) {
                    i2 = (i4 + k * bLen2) + 1;
                    for (i1 = 0; i1 < bLen2; i1++) {
                      dim = i2 + i1;
                      b_iwork[i1] = (short)iidx_data[dim];
                      b_xwork[i1] = vwork_data[dim];
                    }
                    i3 = 0;
                    i1 = bLen;
                    dim = i2 - 1;
                    int exitg1;
                    do {
                      exitg1 = 0;
                      dim++;
                      if (b_xwork[i3] <= b_xwork[i1]) {
                        iidx_data[dim] = b_iwork[i3];
                        vwork_data[dim] = b_xwork[i3];
                        if (i3 + 1 < bLen) {
                          i3++;
                        } else {
                          exitg1 = 1;
                        }
                      } else {
                        iidx_data[dim] = b_iwork[i1];
                        vwork_data[dim] = b_xwork[i1];
                        if (i1 + 1 < bLen2) {
                          i1++;
                        } else {
                          dim -= i3;
                          for (i1 = i3 + 1; i1 <= bLen; i1++) {
                            iidx_tmp = dim + i1;
                            iidx_data[iidx_tmp] = b_iwork[i1 - 1];
                            vwork_data[iidx_tmp] = b_xwork[i1 - 1];
                          }
                          exitg1 = 1;
                        }
                      }
                    } while (exitg1 == 0);
                  }
                }
              }
              dim = n << 8;
              i1 = wOffset_tmp - dim;
              if (i1 > 0) {
                merge_block(iidx, vwork, dim, i1, 2, iwork, xwork);
              }
              dim = 8;
            }
          }
          merge_block(iidx, vwork, 0, wOffset_tmp, dim, iwork, xwork);
          vwork_data = vwork->data;
        }
      }
      for (k = 0; k <= vlen; k++) {
        x_data[j + k * vstride] = vwork_data[k];
      }
    }
  }
  emxFree_real_T(&xwork);
  emxFree_int32_T(&iwork);
  emxFree_int32_T(&iidx);
  emxFree_real_T(&vwork);
}

/*
 * File trailer for sort.c
 *
 * [EOF]
 */
