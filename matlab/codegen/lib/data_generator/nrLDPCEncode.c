/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: nrLDPCEncode.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "nrLDPCEncode.h"
#include "data_generator_emxutil.h"
#include "data_generator_types.h"
#include "encode.h"
#include "find.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : emxArray_real_T *in
 *                double bgn
 *                emxArray_real_T *out
 * Return Type  : void
 */
void nrLDPCEncode(emxArray_real_T *in, double bgn, emxArray_real_T *out)
{
  emxArray_boolean_T *b_in;
  emxArray_int32_T *locs;
  emxArray_real_T *outCBall;
  double *in_data;
  double *out_data;
  int i;
  int i1;
  int i2;
  int i3;
  int *locs_data;
  boolean_T *b_in_data;
  in_data = in->data;
  if ((in->size[0] == 0) || (in->size[1] == 0)) {
    out->size[0] = 0;
    out->size[1] = in->size[1];
  } else {
    double N;
    double Zc;
    int ncwnodes;
    int nsys;
    if (bgn == 1.0) {
      nsys = 22;
      ncwnodes = 66;
    } else {
      nsys = 10;
      ncwnodes = 50;
    }
    Zc = (double)in->size[0] / (double)nsys;
    N = Zc * (double)ncwnodes;
    emxInit_boolean_T(&b_in, 1);
    i = b_in->size[0];
    b_in->size[0] = in->size[0];
    emxEnsureCapacity_boolean_T(b_in, i);
    b_in_data = b_in->data;
    nsys = in->size[0];
    for (i = 0; i < nsys; i++) {
      b_in_data[i] = (in_data[i] == -1.0);
    }
    emxInit_int32_T(&locs, 1);
    eml_find(b_in, locs);
    locs_data = locs->data;
    emxFree_boolean_T(&b_in);
    nsys = in->size[1];
    for (i = 0; i < nsys; i++) {
      ncwnodes = locs->size[0];
      for (i1 = 0; i1 < ncwnodes; i1++) {
        in_data[(locs_data[i1] + in->size[0] * i) - 1] = 0.0;
      }
    }
    emxInit_real_T(&outCBall, 2);
    encode(in, bgn, Zc, outCBall);
    in_data = outCBall->data;
    nsys = outCBall->size[1];
    for (i = 0; i < nsys; i++) {
      ncwnodes = locs->size[0];
      for (i1 = 0; i1 < ncwnodes; i1++) {
        in_data[(locs_data[i1] + outCBall->size[0] * i) - 1] = -1.0;
      }
    }
    emxFree_int32_T(&locs);
    i = out->size[0] * out->size[1];
    out->size[0] = (int)N;
    out->size[1] = in->size[1];
    emxEnsureCapacity_real_T(out, i);
    out_data = out->data;
    nsys = (int)N * in->size[1];
    for (i = 0; i < nsys; i++) {
      out_data[i] = 0.0;
    }
    Zc = 2.0 * Zc + 1.0;
    if (Zc > outCBall->size[0]) {
      i = 0;
      i1 = 0;
    } else {
      i = (int)Zc - 1;
      i1 = outCBall->size[0];
    }
    nsys = outCBall->size[1];
    for (i2 = 0; i2 < nsys; i2++) {
      ncwnodes = i1 - i;
      for (i3 = 0; i3 < ncwnodes; i3++) {
        out_data[i3 + out->size[0] * i2] =
            in_data[(i + i3) + outCBall->size[0] * i2];
      }
    }
    emxFree_real_T(&outCBall);
  }
}

/*
 * File trailer for nrLDPCEncode.c
 *
 * [EOF]
 */
