/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: wlanScramble.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "wlanScramble.h"
#include "data_generator_emxutil.h"
#include "data_generator_types.h"
#include "rt_nonfinite.h"
#include <math.h>

/* Function Definitions */
/*
 * Arguments    : const emxArray_real_T *x
 *                emxArray_real_T *y
 * Return Type  : void
 */
void wlanScramble(const emxArray_real_T *x, emxArray_real_T *y)
{
  static const signed char iv[7] = {1, 0, 1, 1, 1, 0, 1};
  const double *x_data;
  double *y_data;
  int b_i;
  int i;
  int j;
  int loop_ub;
  signed char I_data[127];
  x_data = x->data;
  i = y->size[0] * y->size[1];
  y->size[0] = x->size[0];
  y->size[1] = x->size[1];
  emxEnsureCapacity_real_T(y, i);
  y_data = y->data;
  loop_ub = x->size[0] * x->size[1];
  for (i = 0; i < loop_ub; i++) {
    y_data[i] = 0.0;
  }
  if ((x->size[0] != 0) && (x->size[1] != 0)) {
    int buffSize;
    signed char scramblerInitBits[7];
    for (b_i = 0; b_i < 7; b_i++) {
      scramblerInitBits[b_i] = iv[b_i];
    }
    buffSize = (int)fmin(127.0, x->size[0]);
    for (loop_ub = 0; loop_ub < buffSize; loop_ub++) {
      signed char i1;
      i1 = (signed char)((scramblerInitBits[0] != 0) !=
                         (scramblerInitBits[3] != 0));
      I_data[loop_ub] = i1;
      for (i = 0; i < 6; i++) {
        scramblerInitBits[i] = scramblerInitBits[i + 1];
      }
      scramblerInitBits[6] = i1;
    }
    i = y->size[0] * y->size[1];
    y->size[0] = x->size[0];
    y->size[1] = x->size[1];
    emxEnsureCapacity_real_T(y, i);
    y_data = y->data;
    i = x->size[1];
    loop_ub = x->size[0];
    for (j = 0; j < i; j++) {
      int k;
      k = 0;
      for (b_i = 0; b_i < loop_ub; b_i++) {
        if (k == buffSize) {
          k = 1;
        } else {
          k++;
        }
        y_data[b_i + y->size[0] * j] =
            ((x_data[b_i + x->size[0] * j] != 0.0) != (I_data[k - 1] != 0));
      }
    }
  }
}

/*
 * File trailer for wlanScramble.c
 *
 * [EOF]
 */
