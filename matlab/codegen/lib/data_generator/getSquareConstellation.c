/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: getSquareConstellation.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "getSquareConstellation.h"
#include "data_generator_emxutil.h"
#include "data_generator_rtwutil.h"
#include "data_generator_types.h"
#include "log2.h"
#include "rt_nonfinite.h"
#include "rt_nonfinite.h"
#include <math.h>

/* Function Definitions */
/*
 * Arguments    : double M
 *                emxArray_creal_T *constellation
 * Return Type  : void
 */
void getSquareConstellation(double M, emxArray_creal_T *constellation)
{
  static const cint8_T icv[8] = {{
                                     -3, /* re */
                                     1   /* im */
                                 },
                                 {
                                     -3, /* re */
                                     -1  /* im */
                                 },
                                 {
                                     -1, /* re */
                                     1   /* im */
                                 },
                                 {
                                     -1, /* re */
                                     -1  /* im */
                                 },
                                 {
                                     1, /* re */
                                     1  /* im */
                                 },
                                 {
                                     1, /* re */
                                     -1 /* im */
                                 },
                                 {
                                     3, /* re */
                                     1  /* im */
                                 },
                                 {
                                     3, /* re */
                                     -1 /* im */
                                 }};
  emxArray_real_T *a;
  emxArray_real_T *x;
  emxArray_real_T *xPoints;
  emxArray_real_T *y;
  emxArray_real_T *yPoints;
  creal_T *constellation_data;
  double nbits;
  double *a_data;
  double *xPoints_data;
  double *x_data;
  double *yPoints_data;
  int i;
  int ibmat;
  int itilerow;
  int k;
  int kd;
  int nm1d2;
  nbits = b_log2(M);
  i = constellation->size[0] * constellation->size[1];
  constellation->size[0] = 1;
  kd = (int)M;
  constellation->size[1] = (int)M;
  emxEnsureCapacity_creal_T(constellation, i);
  constellation_data = constellation->data;
  if (nbits == 1.0) {
    i = constellation->size[0] * constellation->size[1];
    constellation->size[0] = 1;
    constellation->size[1] = (int)M;
    emxEnsureCapacity_creal_T(constellation, i);
    constellation_data = constellation->data;
    for (i = 0; i < kd; i++) {
      constellation_data[i].re = 2.0 * (double)i - 1.0;
      constellation_data[i].im = -0.0;
    }
  } else {
    double sqrtM;
    if (rtIsNaN(nbits) || rtIsInf(nbits)) {
      sqrtM = rtNaN;
    } else if (nbits == 0.0) {
      sqrtM = 0.0;
    } else {
      sqrtM = fmod(nbits, 2.0);
      if (sqrtM == 0.0) {
        sqrtM = 0.0;
      } else if (nbits < 0.0) {
        sqrtM += 2.0;
      }
    }
    emxInit_real_T(&xPoints, 2);
    xPoints_data = xPoints->data;
    emxInit_real_T(&yPoints, 2);
    yPoints_data = yPoints->data;
    emxInit_real_T(&x, 2);
    emxInit_real_T(&y, 1);
    emxInit_real_T(&a, 1);
    if ((sqrtM != 0.0) && (nbits > 3.0)) {
      double mI;
      double mQ;
      double tmp2;
      double tmp3;
      double tmp4;
      double tmp5;
      mI = rt_powd_snf(2.0, (nbits + 1.0) / 2.0);
      mQ = rt_powd_snf(2.0, (nbits - 1.0) / 2.0);
      nbits = trunc((M - 1.0) / mI);
      tmp2 = 3.0 * mI / 4.0;
      tmp3 = mI / 2.0;
      tmp4 = mQ / 2.0;
      tmp5 = 2.0 * mQ;
      i = (int)((M - 1.0) + 1.0);
      if (i - 1 >= 0) {
        if (nbits < 2.147483648E+9) {
          if (nbits >= -2.147483648E+9) {
            k = (int)nbits;
          } else {
            k = MIN_int32_T;
          }
        } else if (nbits >= 2.147483648E+9) {
          k = MAX_int32_T;
        } else {
          k = 0;
        }
      }
      for (ibmat = 0; ibmat < i; ibmat++) {
        double apnd;
        nbits = 2.0 * trunc((double)ibmat / mQ) + (1.0 - mI);
        sqrtM = -(2.0 * (double)(ibmat & k) + (1.0 - mQ));
        apnd = fabs(floor(nbits));
        if (apnd > tmp2) {
          double cdiff;
          cdiff = fabs(floor(sqrtM));
          if (nbits < 0.0) {
            kd = -1;
          } else {
            kd = (nbits > 0.0);
          }
          if (sqrtM < 0.0) {
            nm1d2 = -1;
          } else {
            nm1d2 = (sqrtM > 0.0);
          }
          if (cdiff > tmp4) {
            nbits = (double)kd * (apnd - tmp3);
            sqrtM = (double)nm1d2 * (tmp5 - cdiff);
          } else {
            nbits = (double)kd * (mI - apnd);
            sqrtM = (double)nm1d2 * (mQ + cdiff);
          }
        }
        constellation_data[ibmat].re = nbits;
        constellation_data[ibmat].im = sqrtM;
      }
    } else if (nbits == 3.0) {
      i = constellation->size[0] * constellation->size[1];
      constellation->size[0] = 1;
      constellation->size[1] = (int)M;
      emxEnsureCapacity_creal_T(constellation, i);
      constellation_data = constellation->data;
      for (i = 0; i < kd; i++) {
        constellation_data[i].re = icv[i].re;
        constellation_data[i].im = icv[i].im;
      }
    } else {
      double apnd;
      double cdiff;
      boolean_T b;
      sqrtM = rt_powd_snf(2.0, nbits / 2.0);
      b = rtIsNaN(-(sqrtM - 1.0));
      if (b || rtIsNaN(sqrtM - 1.0)) {
        i = xPoints->size[0] * xPoints->size[1];
        xPoints->size[0] = 1;
        xPoints->size[1] = 1;
        emxEnsureCapacity_real_T(xPoints, i);
        xPoints_data = xPoints->data;
        xPoints_data[0] = rtNaN;
      } else if (sqrtM - 1.0 < -(sqrtM - 1.0)) {
        xPoints->size[0] = 1;
        xPoints->size[1] = 0;
      } else if ((rtIsInf(-(sqrtM - 1.0)) || rtIsInf(sqrtM - 1.0)) &&
                 (-(sqrtM - 1.0) == sqrtM - 1.0)) {
        i = xPoints->size[0] * xPoints->size[1];
        xPoints->size[0] = 1;
        xPoints->size[1] = 1;
        emxEnsureCapacity_real_T(xPoints, i);
        xPoints_data = xPoints->data;
        xPoints_data[0] = rtNaN;
      } else if (floor(-(sqrtM - 1.0)) == -(sqrtM - 1.0)) {
        i = xPoints->size[0] * xPoints->size[1];
        xPoints->size[0] = 1;
        kd = (int)(((sqrtM - 1.0) - (-(sqrtM - 1.0))) / 2.0);
        xPoints->size[1] = kd + 1;
        emxEnsureCapacity_real_T(xPoints, i);
        xPoints_data = xPoints->data;
        for (i = 0; i <= kd; i++) {
          xPoints_data[i] = -(sqrtM - 1.0) + 2.0 * (double)i;
        }
      } else {
        nbits = floor(((sqrtM - 1.0) - (-(sqrtM - 1.0))) / 2.0 + 0.5);
        apnd = -(sqrtM - 1.0) + nbits * 2.0;
        cdiff = apnd - (sqrtM - 1.0);
        if (fabs(cdiff) < 4.4408920985006262E-16 *
                              fmax(fabs(-(sqrtM - 1.0)), fabs(sqrtM - 1.0))) {
          nbits++;
          apnd = sqrtM - 1.0;
        } else if (cdiff > 0.0) {
          apnd = -(sqrtM - 1.0) + (nbits - 1.0) * 2.0;
        } else {
          nbits++;
        }
        if (nbits >= 0.0) {
          ibmat = (int)nbits;
        } else {
          ibmat = 0;
        }
        i = xPoints->size[0] * xPoints->size[1];
        xPoints->size[0] = 1;
        xPoints->size[1] = ibmat;
        emxEnsureCapacity_real_T(xPoints, i);
        xPoints_data = xPoints->data;
        if (ibmat > 0) {
          xPoints_data[0] = -(sqrtM - 1.0);
          if (ibmat > 1) {
            xPoints_data[ibmat - 1] = apnd;
            nm1d2 = (ibmat - 1) / 2;
            for (k = 0; k <= nm1d2 - 2; k++) {
              kd = (k + 1) << 1;
              xPoints_data[k + 1] = -(sqrtM - 1.0) + (double)kd;
              xPoints_data[(ibmat - k) - 2] = apnd - (double)kd;
            }
            i = nm1d2 << 1;
            if (i == ibmat - 1) {
              xPoints_data[nm1d2] = (-(sqrtM - 1.0) + apnd) / 2.0;
            } else {
              xPoints_data[nm1d2] = -(sqrtM - 1.0) + (double)i;
              xPoints_data[nm1d2 + 1] = apnd - (double)i;
            }
          }
        }
      }
      if (rtIsNaN(sqrtM - 1.0) || b) {
        i = yPoints->size[0] * yPoints->size[1];
        yPoints->size[0] = 1;
        yPoints->size[1] = 1;
        emxEnsureCapacity_real_T(yPoints, i);
        yPoints_data = yPoints->data;
        yPoints_data[0] = rtNaN;
      } else if (sqrtM - 1.0 < -(sqrtM - 1.0)) {
        yPoints->size[0] = 1;
        yPoints->size[1] = 0;
      } else if ((rtIsInf(sqrtM - 1.0) || rtIsInf(-(sqrtM - 1.0))) &&
                 (sqrtM - 1.0 == -(sqrtM - 1.0))) {
        i = yPoints->size[0] * yPoints->size[1];
        yPoints->size[0] = 1;
        yPoints->size[1] = 1;
        emxEnsureCapacity_real_T(yPoints, i);
        yPoints_data = yPoints->data;
        yPoints_data[0] = rtNaN;
      } else if (floor(sqrtM - 1.0) == sqrtM - 1.0) {
        i = yPoints->size[0] * yPoints->size[1];
        yPoints->size[0] = 1;
        kd = (int)((-(sqrtM - 1.0) - (sqrtM - 1.0)) / -2.0);
        yPoints->size[1] = kd + 1;
        emxEnsureCapacity_real_T(yPoints, i);
        yPoints_data = yPoints->data;
        for (i = 0; i <= kd; i++) {
          yPoints_data[i] = (sqrtM - 1.0) + -2.0 * (double)i;
        }
      } else {
        nbits = floor((-(sqrtM - 1.0) - (sqrtM - 1.0)) / -2.0 + 0.5);
        apnd = (sqrtM - 1.0) + nbits * -2.0;
        cdiff = -(sqrtM - 1.0) - apnd;
        if (fabs(cdiff) < 4.4408920985006262E-16 *
                              fmax(fabs(sqrtM - 1.0), fabs(-(sqrtM - 1.0)))) {
          nbits++;
          apnd = -(sqrtM - 1.0);
        } else if (cdiff > 0.0) {
          apnd = (sqrtM - 1.0) + (nbits - 1.0) * -2.0;
        } else {
          nbits++;
        }
        if (nbits >= 0.0) {
          ibmat = (int)nbits;
        } else {
          ibmat = 0;
        }
        i = yPoints->size[0] * yPoints->size[1];
        yPoints->size[0] = 1;
        yPoints->size[1] = ibmat;
        emxEnsureCapacity_real_T(yPoints, i);
        yPoints_data = yPoints->data;
        if (ibmat > 0) {
          yPoints_data[0] = sqrtM - 1.0;
          if (ibmat > 1) {
            yPoints_data[ibmat - 1] = apnd;
            nm1d2 = (ibmat - 1) / 2;
            for (k = 0; k <= nm1d2 - 2; k++) {
              kd = (k + 1) * -2;
              yPoints_data[k + 1] = (sqrtM - 1.0) + (double)kd;
              yPoints_data[(ibmat - k) - 2] = apnd - (double)kd;
            }
            if (nm1d2 << 1 == ibmat - 1) {
              yPoints_data[nm1d2] = ((sqrtM - 1.0) + apnd) / 2.0;
            } else {
              kd = nm1d2 * -2;
              yPoints_data[nm1d2] = (sqrtM - 1.0) + (double)kd;
              yPoints_data[nm1d2 + 1] = apnd - (double)kd;
            }
          }
        }
      }
      i = (int)sqrtM;
      k = x->size[0] * x->size[1];
      x->size[0] = (int)sqrtM;
      x->size[1] = xPoints->size[1];
      emxEnsureCapacity_real_T(x, k);
      x_data = x->data;
      kd = xPoints->size[1];
      for (nm1d2 = 0; nm1d2 < kd; nm1d2++) {
        ibmat = nm1d2 * (int)sqrtM;
        for (itilerow = 0; itilerow < i; itilerow++) {
          x_data[ibmat + itilerow] = xPoints_data[nm1d2];
        }
      }
      k = a->size[0];
      a->size[0] = yPoints->size[1];
      emxEnsureCapacity_real_T(a, k);
      a_data = a->data;
      kd = yPoints->size[1];
      for (k = 0; k < kd; k++) {
        a_data[k] = yPoints_data[k];
      }
      k = y->size[0];
      y->size[0] = a->size[0] * (int)sqrtM;
      emxEnsureCapacity_real_T(y, k);
      xPoints_data = y->data;
      kd = a->size[0];
      for (itilerow = 0; itilerow < i; itilerow++) {
        nm1d2 = itilerow * kd;
        for (k = 0; k < kd; k++) {
          xPoints_data[nm1d2 + k] = a_data[k];
        }
      }
      kd = x->size[0] * x->size[1];
      i = constellation->size[0] * constellation->size[1];
      constellation->size[0] = 1;
      constellation->size[1] = kd;
      emxEnsureCapacity_creal_T(constellation, i);
      constellation_data = constellation->data;
      for (i = 0; i < kd; i++) {
        constellation_data[i].re = x_data[i];
        constellation_data[i].im = xPoints_data[i];
      }
    }
    emxFree_real_T(&a);
    emxFree_real_T(&y);
    emxFree_real_T(&x);
    emxFree_real_T(&yPoints);
    emxFree_real_T(&xPoints);
    i = constellation->size[0] * constellation->size[1];
    constellation->size[0] = 1;
    emxEnsureCapacity_creal_T(constellation, i);
    constellation_data = constellation->data;
    kd = constellation->size[1] - 1;
    for (i = 0; i <= kd; i++) {
      sqrtM = constellation_data[i].re;
      nbits = constellation_data[i].im;
      constellation_data[i].re = sqrtM - nbits * 0.0;
      constellation_data[i].im = sqrtM * 0.0 + nbits;
    }
  }
}

/*
 * File trailer for getSquareConstellation.c
 *
 * [EOF]
 */
