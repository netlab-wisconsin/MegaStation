/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: ifft.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "ifft.h"
#include "FFTImplementationCallback.h"
#include "data_generator_emxutil.h"
#include "data_generator_types.h"
#include "rt_nonfinite.h"
#include <math.h>

/* Function Definitions */
/*
 * Arguments    : const emxArray_creal32_T *x
 *                emxArray_creal32_T *y
 * Return Type  : void
 */
void ifft(const emxArray_creal32_T *x, emxArray_creal32_T *y)
{
  emxArray_real32_T *costab;
  emxArray_real32_T *costab1q;
  emxArray_real32_T *sintab;
  emxArray_real32_T *sintabinv;
  creal32_T *y_data;
  float *costab1q_data;
  float *costab_data;
  float *sintab_data;
  float *sintabinv_data;
  int k;
  int pow2p;
  if ((x->size[0] == 0) || (x->size[1] == 0) || (x->size[2] == 0)) {
    int pmax;
    pow2p = y->size[0] * y->size[1] * y->size[2];
    y->size[0] = x->size[0];
    y->size[1] = x->size[1];
    y->size[2] = x->size[2];
    emxEnsureCapacity_creal32_T(y, pow2p);
    y_data = y->data;
    pmax = x->size[0] * x->size[1] * x->size[2];
    for (pow2p = 0; pow2p < pmax; pow2p++) {
      y_data[pow2p].re = 0.0F;
      y_data[pow2p].im = 0.0F;
    }
  } else {
    float e;
    int n;
    int pmax;
    int pmin;
    boolean_T useRadix2;
    useRadix2 = ((x->size[0] & (x->size[0] - 1)) == 0);
    pmin = 1;
    if (useRadix2) {
      pmax = x->size[0];
    } else {
      n = (x->size[0] + x->size[0]) - 1;
      pmax = 31;
      if (n <= 1) {
        pmax = 0;
      } else {
        boolean_T exitg1;
        pmin = 0;
        exitg1 = false;
        while ((!exitg1) && (pmax - pmin > 1)) {
          k = (pmin + pmax) >> 1;
          pow2p = 1 << k;
          if (pow2p == n) {
            pmax = k;
            exitg1 = true;
          } else if (pow2p > n) {
            pmax = k;
          } else {
            pmin = k;
          }
        }
      }
      pmin = 1 << pmax;
      pmax = pmin;
    }
    e = 6.28318548F / (float)pmax;
    n = (int)((unsigned int)pmax >> 1) >> 1;
    emxInit_real32_T(&costab1q, 2);
    pow2p = costab1q->size[0] * costab1q->size[1];
    costab1q->size[0] = 1;
    costab1q->size[1] = n + 1;
    emxEnsureCapacity_real32_T(costab1q, pow2p);
    costab1q_data = costab1q->data;
    costab1q_data[0] = 1.0F;
    pmax = n / 2 - 1;
    for (k = 0; k <= pmax; k++) {
      costab1q_data[k + 1] = cosf(e * (float)(k + 1));
    }
    pow2p = pmax + 2;
    pmax = n - 1;
    for (k = pow2p; k <= pmax; k++) {
      costab1q_data[k] = sinf(e * (float)(n - k));
    }
    costab1q_data[n] = 0.0F;
    emxInit_real32_T(&costab, 2);
    emxInit_real32_T(&sintab, 2);
    emxInit_real32_T(&sintabinv, 2);
    if (!useRadix2) {
      n = costab1q->size[1] - 1;
      pmax = (costab1q->size[1] - 1) << 1;
      pow2p = costab->size[0] * costab->size[1];
      costab->size[0] = 1;
      costab->size[1] = pmax + 1;
      emxEnsureCapacity_real32_T(costab, pow2p);
      costab_data = costab->data;
      pow2p = sintab->size[0] * sintab->size[1];
      sintab->size[0] = 1;
      sintab->size[1] = pmax + 1;
      emxEnsureCapacity_real32_T(sintab, pow2p);
      sintab_data = sintab->data;
      costab_data[0] = 1.0F;
      sintab_data[0] = 0.0F;
      pow2p = sintabinv->size[0] * sintabinv->size[1];
      sintabinv->size[0] = 1;
      sintabinv->size[1] = pmax + 1;
      emxEnsureCapacity_real32_T(sintabinv, pow2p);
      sintabinv_data = sintabinv->data;
      for (k = 0; k < n; k++) {
        sintabinv_data[k + 1] = costab1q_data[(n - k) - 1];
      }
      pow2p = costab1q->size[1];
      for (k = pow2p; k <= pmax; k++) {
        sintabinv_data[k] = costab1q_data[k - n];
      }
      for (k = 0; k < n; k++) {
        costab_data[k + 1] = costab1q_data[k + 1];
        sintab_data[k + 1] = -costab1q_data[(n - k) - 1];
      }
      pow2p = costab1q->size[1];
      for (k = pow2p; k <= pmax; k++) {
        costab_data[k] = -costab1q_data[pmax - k];
        sintab_data[k] = -costab1q_data[k - n];
      }
    } else {
      n = costab1q->size[1] - 1;
      pmax = (costab1q->size[1] - 1) << 1;
      pow2p = costab->size[0] * costab->size[1];
      costab->size[0] = 1;
      costab->size[1] = pmax + 1;
      emxEnsureCapacity_real32_T(costab, pow2p);
      costab_data = costab->data;
      pow2p = sintab->size[0] * sintab->size[1];
      sintab->size[0] = 1;
      sintab->size[1] = pmax + 1;
      emxEnsureCapacity_real32_T(sintab, pow2p);
      sintab_data = sintab->data;
      costab_data[0] = 1.0F;
      sintab_data[0] = 0.0F;
      for (k = 0; k < n; k++) {
        costab_data[k + 1] = costab1q_data[k + 1];
        sintab_data[k + 1] = costab1q_data[(n - k) - 1];
      }
      pow2p = costab1q->size[1];
      for (k = pow2p; k <= pmax; k++) {
        costab_data[k] = -costab1q_data[pmax - k];
        sintab_data[k] = costab1q_data[k - n];
      }
      sintabinv->size[0] = 1;
      sintabinv->size[1] = 0;
    }
    emxFree_real32_T(&costab1q);
    if (useRadix2) {
      c_FFTImplementationCallback_r2b(x, x->size[0], costab, sintab, y);
    } else {
      c_FFTImplementationCallback_dob(x, pmin, x->size[0], costab, sintab,
                                      sintabinv, y);
    }
    emxFree_real32_T(&sintabinv);
    emxFree_real32_T(&sintab);
    emxFree_real32_T(&costab);
  }
}

/*
 * File trailer for ifft.c
 *
 * [EOF]
 */
