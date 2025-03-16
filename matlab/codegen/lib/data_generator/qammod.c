/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: qammod.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "qammod.h"
#include "data_generator_emxutil.h"
#include "data_generator_rtwutil.h"
#include "data_generator_types.h"
#include "getSquareConstellation.h"
#include "log2.h"
#include "rt_nonfinite.h"
#include "rt_nonfinite.h"
#include <math.h>

/* Function Definitions */
/*
 * Arguments    : const emxArray_real_T *x
 *                double M
 *                emxArray_creal_T *y
 * Return Type  : void
 */
void qammod(const emxArray_real_T *x, double M, emxArray_creal_T *y)
{
  static const signed char iv[64] = {
      48, 47, 43, 44, 60, 59, 63, 64, 46, 45, 41, 42, 58, 57, 61, 62,
      38, 37, 33, 34, 50, 49, 53, 54, 40, 39, 35, 36, 52, 51, 55, 56,
      8,  7,  3,  4,  20, 19, 23, 24, 6,  5,  1,  2,  18, 17, 21, 22,
      14, 13, 9,  10, 26, 25, 29, 30, 16, 15, 11, 12, 28, 27, 31, 32};
  emxArray_creal_T *newConst;
  emxArray_real_T *b_y;
  emxArray_real_T *msg;
  emxArray_real_T *newSymbolOrderMap;
  emxArray_real_T *powOf2;
  emxArray_real_T *sym;
  emxArray_real_T *symbolOrderMap;
  creal_T *newConst_data;
  creal_T *y_data;
  const double *x_data;
  double defaultAveragePower;
  double nBits;
  double ndbl;
  double *msg_data;
  double *powOf2_data;
  double *sym_data;
  double *symbolOrderMap_data;
  int i;
  int k;
  int n;
  int nm1d2;
  x_data = x->data;
  emxInit_creal_T(&newConst, 2);
  getSquareConstellation(M, newConst);
  if ((M == 2.0) || (M == 8.0)) {
    defaultAveragePower = (5.0 * M / 4.0 - 1.0) * 2.0 / 3.0;
  } else {
    defaultAveragePower = b_log2(M);
    if (rtIsNaN(defaultAveragePower) || rtIsInf(defaultAveragePower)) {
      ndbl = rtNaN;
    } else if (defaultAveragePower == 0.0) {
      ndbl = 0.0;
    } else {
      ndbl = fmod(defaultAveragePower, 2.0);
      if (ndbl == 0.0) {
        ndbl = 0.0;
      } else if (defaultAveragePower < 0.0) {
        ndbl += 2.0;
      }
    }
    if (ndbl != 0.0) {
      defaultAveragePower = (31.0 * M / 32.0 - 1.0) * 2.0 / 3.0;
    } else {
      defaultAveragePower = (M - 1.0) * 2.0 / 3.0;
    }
  }
  defaultAveragePower = sqrt(1.0 / defaultAveragePower);
  nm1d2 = newConst->size[1];
  i = newConst->size[0] * newConst->size[1];
  newConst->size[0] = 1;
  emxEnsureCapacity_creal_T(newConst, i);
  newConst_data = newConst->data;
  for (i = 0; i < nm1d2; i++) {
    newConst_data[i].re *= defaultAveragePower;
    newConst_data[i].im *= defaultAveragePower;
  }
  if ((x->size[2] == 1) && (x->size[1] == 1)) {
    i = newConst->size[0] * newConst->size[1];
    newConst->size[0] = newConst->size[1];
    newConst->size[1] = 1;
    emxEnsureCapacity_creal_T(newConst, i);
    newConst_data = newConst->data;
  }
  nBits = b_log2(M);
  defaultAveragePower = (double)x->size[0] / nBits;
  emxInit_real_T(&sym, 3);
  i = sym->size[0] * sym->size[1] * sym->size[2];
  sym->size[0] = (int)defaultAveragePower;
  sym->size[1] = x->size[1];
  sym->size[2] = x->size[2];
  emxEnsureCapacity_real_T(sym, i);
  sym_data = sym->data;
  emxInit_real_T(&b_y, 2);
  msg_data = b_y->data;
  if (rtIsNaN(nBits - 1.0)) {
    i = b_y->size[0] * b_y->size[1];
    b_y->size[0] = 1;
    b_y->size[1] = 1;
    emxEnsureCapacity_real_T(b_y, i);
    msg_data = b_y->data;
    msg_data[0] = rtNaN;
  } else if (nBits - 1.0 < 0.0) {
    b_y->size[0] = 1;
    b_y->size[1] = 0;
  } else if (floor(nBits - 1.0) == nBits - 1.0) {
    i = b_y->size[0] * b_y->size[1];
    b_y->size[0] = 1;
    b_y->size[1] = (int)-(0.0 - (nBits - 1.0)) + 1;
    emxEnsureCapacity_real_T(b_y, i);
    msg_data = b_y->data;
    nm1d2 = (int)-(0.0 - (nBits - 1.0));
    for (i = 0; i <= nm1d2; i++) {
      msg_data[i] = (nBits - 1.0) - (double)i;
    }
  } else {
    double apnd;
    ndbl = floor(-(0.0 - (nBits - 1.0)) + 0.5);
    apnd = (nBits - 1.0) - ndbl;
    if (fabs(0.0 - apnd) < 4.4408920985006262E-16 * fmax(nBits - 1.0, 0.0)) {
      ndbl++;
      apnd = 0.0;
    } else if (0.0 - apnd > 0.0) {
      apnd = (nBits - 1.0) - (ndbl - 1.0);
    } else {
      ndbl++;
    }
    if (ndbl >= 0.0) {
      n = (int)ndbl;
    } else {
      n = 0;
    }
    i = b_y->size[0] * b_y->size[1];
    b_y->size[0] = 1;
    b_y->size[1] = n;
    emxEnsureCapacity_real_T(b_y, i);
    msg_data = b_y->data;
    if (n > 0) {
      msg_data[0] = nBits - 1.0;
      if (n > 1) {
        msg_data[n - 1] = apnd;
        nm1d2 = (n - 1) / 2;
        for (k = 0; k <= nm1d2 - 2; k++) {
          msg_data[k + 1] = (nBits - 1.0) - ((double)k + 1.0);
          msg_data[(n - k) - 2] = apnd - (-((double)k + 1.0));
        }
        if (nm1d2 << 1 == n - 1) {
          msg_data[nm1d2] = ((nBits - 1.0) + apnd) / 2.0;
        } else {
          msg_data[nm1d2] = (nBits - 1.0) - (double)nm1d2;
          msg_data[nm1d2 + 1] = apnd - (-(double)nm1d2);
        }
      }
    }
  }
  emxInit_real_T(&symbolOrderMap, 1);
  i = symbolOrderMap->size[0];
  symbolOrderMap->size[0] = b_y->size[1];
  emxEnsureCapacity_real_T(symbolOrderMap, i);
  symbolOrderMap_data = symbolOrderMap->data;
  nm1d2 = b_y->size[1];
  for (i = 0; i < nm1d2; i++) {
    symbolOrderMap_data[i] = msg_data[i];
  }
  nm1d2 = symbolOrderMap->size[0];
  emxInit_real_T(&powOf2, 1);
  i = powOf2->size[0];
  powOf2->size[0] = symbolOrderMap->size[0];
  emxEnsureCapacity_real_T(powOf2, i);
  powOf2_data = powOf2->data;
  for (k = 0; k < nm1d2; k++) {
    powOf2_data[k] = rt_powd_snf(2.0, symbolOrderMap_data[k]);
  }
  i = (int)(defaultAveragePower * (double)x->size[1] * (double)x->size[2]);
  for (n = 0; n < i; n++) {
    defaultAveragePower = nBits * (((double)n + 1.0) - 1.0);
    ndbl = 0.0;
    nm1d2 = (int)nBits;
    for (k = 0; k < nm1d2; k++) {
      ndbl += x_data[(int)(defaultAveragePower + ((double)k + 1.0)) - 1] *
              powOf2_data[k];
    }
    sym_data[n] = ndbl;
  }
  emxFree_real_T(&powOf2);
  i = symbolOrderMap->size[0];
  symbolOrderMap->size[0] = (int)M;
  emxEnsureCapacity_real_T(symbolOrderMap, i);
  symbolOrderMap_data = symbolOrderMap->data;
  if (M - 1.0 < 0.0) {
    b_y->size[0] = 1;
    b_y->size[1] = 0;
  } else {
    i = b_y->size[0] * b_y->size[1];
    b_y->size[0] = 1;
    b_y->size[1] = (int)(M - 1.0) + 1;
    emxEnsureCapacity_real_T(b_y, i);
    msg_data = b_y->data;
    nm1d2 = (int)(M - 1.0);
    for (i = 0; i <= nm1d2; i++) {
      msg_data[i] = i;
    }
  }
  for (i = 0; i < 64; i++) {
    symbolOrderMap_data[iv[i] - 1] = msg_data[i];
  }
  emxFree_real_T(&b_y);
  emxInit_real_T(&newSymbolOrderMap, 2);
  if ((sym->size[2] == 1) && (sym->size[0] == 1)) {
    i = newSymbolOrderMap->size[0] * newSymbolOrderMap->size[1];
    newSymbolOrderMap->size[0] = 1;
    newSymbolOrderMap->size[1] = symbolOrderMap->size[0];
    emxEnsureCapacity_real_T(newSymbolOrderMap, i);
    powOf2_data = newSymbolOrderMap->data;
    nm1d2 = symbolOrderMap->size[0];
    for (i = 0; i < nm1d2; i++) {
      powOf2_data[newSymbolOrderMap->size[0] * i] = symbolOrderMap_data[i];
    }
  } else {
    i = newSymbolOrderMap->size[0] * newSymbolOrderMap->size[1];
    newSymbolOrderMap->size[0] = symbolOrderMap->size[0];
    newSymbolOrderMap->size[1] = 1;
    emxEnsureCapacity_real_T(newSymbolOrderMap, i);
    powOf2_data = newSymbolOrderMap->data;
    nm1d2 = symbolOrderMap->size[0];
    for (i = 0; i < nm1d2; i++) {
      powOf2_data[i] = symbolOrderMap_data[i];
    }
  }
  emxFree_real_T(&symbolOrderMap);
  emxInit_real_T(&msg, 3);
  i = msg->size[0] * msg->size[1] * msg->size[2];
  msg->size[0] = sym->size[0];
  msg->size[1] = sym->size[1];
  msg->size[2] = sym->size[2];
  emxEnsureCapacity_real_T(msg, i);
  msg_data = msg->data;
  nm1d2 = sym->size[0] * sym->size[1] * sym->size[2];
  for (i = 0; i < nm1d2; i++) {
    msg_data[i] = powOf2_data[(int)(sym_data[i] + 1.0) - 1];
  }
  emxFree_real_T(&newSymbolOrderMap);
  emxFree_real_T(&sym);
  i = y->size[0] * y->size[1] * y->size[2];
  y->size[0] = msg->size[0];
  y->size[1] = msg->size[1];
  y->size[2] = msg->size[2];
  emxEnsureCapacity_creal_T(y, i);
  y_data = y->data;
  for (i = 0; i < nm1d2; i++) {
    y_data[i] = newConst_data[(int)msg_data[i]];
  }
  emxFree_real_T(&msg);
  emxFree_creal_T(&newConst);
}

/*
 * File trailer for qammod.c
 *
 * [EOF]
 */
