/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: FFTImplementationCallback.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "FFTImplementationCallback.h"
#include "data_generator_data.h"
#include "data_generator_emxutil.h"
#include "data_generator_types.h"
#include "rt_nonfinite.h"
#include "omp.h"
#include <math.h>

/* Function Declarations */
static void d_FFTImplementationCallback_r2b(const emxArray_creal32_T *x,
                                            int unsigned_nRows,
                                            const emxArray_real32_T *costab,
                                            const emxArray_real32_T *sintab,
                                            emxArray_creal32_T *y);

/* Function Definitions */
/*
 * Arguments    : const emxArray_creal32_T *x
 *                int unsigned_nRows
 *                const emxArray_real32_T *costab
 *                const emxArray_real32_T *sintab
 *                emxArray_creal32_T *y
 * Return Type  : void
 */
static void d_FFTImplementationCallback_r2b(const emxArray_creal32_T *x,
                                            int unsigned_nRows,
                                            const emxArray_real32_T *costab,
                                            const emxArray_real32_T *sintab,
                                            emxArray_creal32_T *y)
{
  const creal32_T *x_data;
  creal32_T *y_data;
  const float *costab_data;
  const float *sintab_data;
  float temp_im;
  float temp_re;
  float temp_re_tmp;
  float twid_re;
  int i;
  int iDelta;
  int iDelta2;
  int iheight;
  int ihi;
  int iy;
  int j;
  int ju;
  int k;
  int nRowsD2;
  sintab_data = sintab->data;
  costab_data = costab->data;
  x_data = x->data;
  iy = y->size[0];
  y->size[0] = unsigned_nRows;
  emxEnsureCapacity_creal32_T(y, iy);
  y_data = y->data;
  if (unsigned_nRows > x->size[0]) {
    iy = y->size[0];
    y->size[0] = unsigned_nRows;
    emxEnsureCapacity_creal32_T(y, iy);
    y_data = y->data;
    for (iy = 0; iy < unsigned_nRows; iy++) {
      y_data[iy].re = 0.0F;
      y_data[iy].im = 0.0F;
    }
  }
  j = x->size[0];
  if (j > unsigned_nRows) {
    j = unsigned_nRows;
  }
  ihi = unsigned_nRows - 2;
  nRowsD2 = (int)((unsigned int)unsigned_nRows >> 1);
  k = nRowsD2 / 2;
  iy = 0;
  ju = 0;
  for (i = 0; i <= j - 2; i++) {
    boolean_T tst;
    y_data[iy] = x_data[i];
    iy = unsigned_nRows;
    tst = true;
    while (tst) {
      iy >>= 1;
      ju ^= iy;
      tst = ((ju & iy) == 0);
    }
    iy = ju;
  }
  if (j - 2 < 0) {
    j = 0;
  } else {
    j--;
  }
  y_data[iy] = x_data[j];
  if (unsigned_nRows > 1) {
    for (i = 0; i <= ihi; i += 2) {
      temp_re_tmp = y_data[i + 1].re;
      temp_im = y_data[i + 1].im;
      temp_re = y_data[i].re;
      twid_re = y_data[i].im;
      y_data[i + 1].re = temp_re - temp_re_tmp;
      y_data[i + 1].im = twid_re - temp_im;
      y_data[i].re = temp_re + temp_re_tmp;
      y_data[i].im = twid_re + temp_im;
    }
  }
  iDelta = 2;
  iDelta2 = 4;
  iheight = ((k - 1) << 2) + 1;
  while (k > 0) {
    for (i = 0; i < iheight; i += iDelta2) {
      iy = i + iDelta;
      temp_re = y_data[iy].re;
      temp_im = y_data[iy].im;
      y_data[iy].re = y_data[i].re - temp_re;
      y_data[iy].im = y_data[i].im - temp_im;
      y_data[i].re += temp_re;
      y_data[i].im += temp_im;
    }
    iy = 1;
    for (j = k; j < nRowsD2; j += k) {
      float twid_im;
      twid_re = costab_data[j];
      twid_im = sintab_data[j];
      i = iy;
      ihi = iy + iheight;
      while (i < ihi) {
        ju = i + iDelta;
        temp_re_tmp = y_data[ju].im;
        temp_im = y_data[ju].re;
        temp_re = twid_re * temp_im - twid_im * temp_re_tmp;
        temp_im = twid_re * temp_re_tmp + twid_im * temp_im;
        y_data[ju].re = y_data[i].re - temp_re;
        y_data[ju].im = y_data[i].im - temp_im;
        y_data[i].re += temp_re;
        y_data[i].im += temp_im;
        i += iDelta2;
      }
      iy++;
    }
    k /= 2;
    iDelta = iDelta2;
    iDelta2 += iDelta2;
    iheight -= iDelta;
  }
}

/*
 * Arguments    : const emxArray_creal32_T *x
 *                int n2blue
 *                int nfft
 *                const emxArray_real32_T *costab
 *                const emxArray_real32_T *sintab
 *                const emxArray_real32_T *sintabinv
 *                emxArray_creal32_T *y
 * Return Type  : void
 */
void c_FFTImplementationCallback_dob(const emxArray_creal32_T *x, int n2blue,
                                     int nfft, const emxArray_real32_T *costab,
                                     const emxArray_real32_T *sintab,
                                     const emxArray_real32_T *sintabinv,
                                     emxArray_creal32_T *y)
{
  emxArray_creal32_T *fv;
  emxArray_creal32_T *fy;
  emxArray_creal32_T *r1;
  emxArray_creal32_T *wwc;
  const creal32_T *x_data;
  creal32_T *fv_data;
  creal32_T *fy_data;
  creal32_T *r;
  creal32_T *wwc_data;
  creal32_T *y_data;
  const float *costab_data;
  const float *sintab_data;
  float nt_im;
  float re_tmp;
  float temp_im;
  float temp_re;
  float temp_re_tmp;
  float twid_im;
  float twid_re;
  int b_i;
  int b_k;
  int b_y;
  int chan;
  int i;
  int iDelta;
  int iDelta2;
  int iheight;
  int iy;
  int ju;
  int k;
  int minNrowsNx;
  int nInt2;
  int nInt2m1;
  int nRowsD2;
  int rt;
  int xoff;
  boolean_T tst;
  sintab_data = sintab->data;
  costab_data = costab->data;
  x_data = x->data;
  nInt2m1 = (nfft + nfft) - 1;
  emxInit_creal32_T(&wwc, 1);
  i = wwc->size[0];
  wwc->size[0] = nInt2m1;
  emxEnsureCapacity_creal32_T(wwc, i);
  wwc_data = wwc->data;
  rt = 0;
  wwc_data[nfft - 1].re = 1.0F;
  wwc_data[nfft - 1].im = 0.0F;
  nInt2 = nfft << 1;
  for (k = 0; k <= nfft - 2; k++) {
    b_y = ((k + 1) << 1) - 1;
    if (nInt2 - rt <= b_y) {
      rt += b_y - nInt2;
    } else {
      rt += b_y;
    }
    nt_im = 3.14159274F * (float)rt / (float)nfft;
    i = (nfft - k) - 2;
    wwc_data[i].re = cosf(nt_im);
    wwc_data[i].im = -sinf(nt_im);
  }
  i = nInt2m1 - 1;
  for (k = i; k >= nfft; k--) {
    wwc_data[k] = wwc_data[(nInt2m1 - k) - 1];
  }
  nInt2m1 = x->size[0];
  i = y->size[0] * y->size[1] * y->size[2];
  y->size[0] = nfft;
  y->size[1] = x->size[1];
  y->size[2] = x->size[2];
  emxEnsureCapacity_creal32_T(y, i);
  y_data = y->data;
  if (nfft > x->size[0]) {
    i = y->size[0] * y->size[1] * y->size[2];
    y->size[0] = nfft;
    y->size[1] = x->size[1];
    y->size[2] = x->size[2];
    emxEnsureCapacity_creal32_T(y, i);
    y_data = y->data;
    b_y = nfft * x->size[1] * x->size[2];
    for (i = 0; i < b_y; i++) {
      y_data[i].re = 0.0F;
      y_data[i].im = 0.0F;
    }
  }
  b_y = x->size[1] * x->size[2] - 1;
#pragma omp parallel num_threads(omp_get_max_threads()) private(               \
    r, fy_data, fv_data, fv, fy, r1, xoff, ju, minNrowsNx, b_k, iy, temp_re,   \
    temp_im, twid_im, re_tmp, iDelta, nRowsD2, b_i, tst, temp_re_tmp, twid_re, \
    iDelta2, iheight)
  {
    emxInit_creal32_T(&fv, 1);
    emxInit_creal32_T(&fy, 1);
    emxInit_creal32_T(&r1, 1);
#pragma omp for nowait
    for (chan = 0; chan <= b_y; chan++) {
      xoff = chan * nInt2m1;
      ju = r1->size[0];
      r1->size[0] = nfft;
      emxEnsureCapacity_creal32_T(r1, ju);
      r = r1->data;
      if (nfft > x->size[0]) {
        ju = r1->size[0];
        r1->size[0] = nfft;
        emxEnsureCapacity_creal32_T(r1, ju);
        r = r1->data;
        for (ju = 0; ju < nfft; ju++) {
          r[ju].re = 0.0F;
          r[ju].im = 0.0F;
        }
      }
      minNrowsNx = x->size[0];
      if (nfft <= minNrowsNx) {
        minNrowsNx = nfft;
      }
      for (b_k = 0; b_k < minNrowsNx; b_k++) {
        iy = (nfft + b_k) - 1;
        temp_re = wwc_data[iy].re;
        temp_im = wwc_data[iy].im;
        ju = xoff + b_k;
        twid_im = x_data[ju].im;
        re_tmp = x_data[ju].re;
        r[b_k].re = temp_re * re_tmp + temp_im * twid_im;
        r[b_k].im = temp_re * twid_im - temp_im * re_tmp;
      }
      ju = minNrowsNx + 1;
      for (b_k = ju; b_k <= nfft; b_k++) {
        r[b_k - 1].re = 0.0F;
        r[b_k - 1].im = 0.0F;
      }
      ju = fy->size[0];
      fy->size[0] = n2blue;
      emxEnsureCapacity_creal32_T(fy, ju);
      fy_data = fy->data;
      if (n2blue > r1->size[0]) {
        ju = fy->size[0];
        fy->size[0] = n2blue;
        emxEnsureCapacity_creal32_T(fy, ju);
        fy_data = fy->data;
        for (ju = 0; ju < n2blue; ju++) {
          fy_data[ju].re = 0.0F;
          fy_data[ju].im = 0.0F;
        }
      }
      iy = r1->size[0];
      iDelta = n2blue;
      if (iy <= n2blue) {
        iDelta = iy;
      }
      minNrowsNx = n2blue - 2;
      nRowsD2 = (int)((unsigned int)n2blue >> 1);
      b_k = nRowsD2 / 2;
      iy = 0;
      ju = 0;
      for (b_i = 0; b_i <= iDelta - 2; b_i++) {
        fy_data[iy] = r[b_i];
        xoff = n2blue;
        tst = true;
        while (tst) {
          xoff >>= 1;
          ju ^= xoff;
          tst = ((ju & xoff) == 0);
        }
        iy = ju;
      }
      if (iDelta - 2 < 0) {
        xoff = 0;
      } else {
        xoff = iDelta - 1;
      }
      fy_data[iy] = r[xoff];
      if (n2blue > 1) {
        for (b_i = 0; b_i <= minNrowsNx; b_i += 2) {
          temp_re_tmp = fy_data[b_i + 1].re;
          temp_im = fy_data[b_i + 1].im;
          re_tmp = fy_data[b_i].re;
          twid_re = fy_data[b_i].im;
          fy_data[b_i + 1].re = re_tmp - temp_re_tmp;
          fy_data[b_i + 1].im = twid_re - temp_im;
          fy_data[b_i].re = re_tmp + temp_re_tmp;
          fy_data[b_i].im = twid_re + temp_im;
        }
      }
      iDelta = 2;
      iDelta2 = 4;
      iheight = ((b_k - 1) << 2) + 1;
      while (b_k > 0) {
        for (b_i = 0; b_i < iheight; b_i += iDelta2) {
          iy = b_i + iDelta;
          temp_re = fy_data[iy].re;
          temp_im = fy_data[iy].im;
          fy_data[iy].re = fy_data[b_i].re - temp_re;
          fy_data[iy].im = fy_data[b_i].im - temp_im;
          fy_data[b_i].re += temp_re;
          fy_data[b_i].im += temp_im;
        }
        xoff = 1;
        for (minNrowsNx = b_k; minNrowsNx < nRowsD2; minNrowsNx += b_k) {
          twid_re = costab_data[minNrowsNx];
          twid_im = sintab_data[minNrowsNx];
          b_i = xoff;
          iy = xoff + iheight;
          while (b_i < iy) {
            ju = b_i + iDelta;
            temp_re_tmp = fy_data[ju].im;
            temp_im = fy_data[ju].re;
            temp_re = twid_re * temp_im - twid_im * temp_re_tmp;
            temp_im = twid_re * temp_re_tmp + twid_im * temp_im;
            fy_data[ju].re = fy_data[b_i].re - temp_re;
            fy_data[ju].im = fy_data[b_i].im - temp_im;
            fy_data[b_i].re += temp_re;
            fy_data[b_i].im += temp_im;
            b_i += iDelta2;
          }
          xoff++;
        }
        b_k /= 2;
        iDelta = iDelta2;
        iDelta2 += iDelta2;
        iheight -= iDelta;
      }
      d_FFTImplementationCallback_r2b(wwc, n2blue, costab, sintab, fv);
      fv_data = fv->data;
      iy = fy->size[0];
      for (ju = 0; ju < iy; ju++) {
        re_tmp = fy_data[ju].re;
        twid_im = fv_data[ju].im;
        temp_im = fy_data[ju].im;
        twid_re = fv_data[ju].re;
        fy_data[ju].re = re_tmp * twid_re - temp_im * twid_im;
        fy_data[ju].im = re_tmp * twid_im + temp_im * twid_re;
      }
      d_FFTImplementationCallback_r2b(fy, n2blue, costab, sintabinv, fv);
      fv_data = fv->data;
      if (fv->size[0] > 1) {
        temp_im = 1.0F / (float)fv->size[0];
        iy = fv->size[0];
        for (ju = 0; ju < iy; ju++) {
          fv_data[ju].re *= temp_im;
          fv_data[ju].im *= temp_im;
        }
      }
      ju = (int)(float)nfft;
      xoff = wwc->size[0];
      for (b_k = ju; b_k <= xoff; b_k++) {
        twid_im = wwc_data[b_k - 1].re;
        re_tmp = fv_data[b_k - 1].im;
        temp_im = wwc_data[b_k - 1].im;
        twid_re = fv_data[b_k - 1].re;
        iy = b_k - (int)(float)nfft;
        r[iy].re = twid_im * twid_re + temp_im * re_tmp;
        r[iy].im = twid_im * re_tmp - temp_im * twid_re;
        temp_im = r[iy].re;
        twid_im = r[iy].im;
        if (twid_im == 0.0F) {
          twid_re = temp_im / (float)nfft;
          temp_im = 0.0F;
        } else if (temp_im == 0.0F) {
          twid_re = 0.0F;
          temp_im = twid_im / (float)nfft;
        } else {
          twid_re = temp_im / (float)nfft;
          temp_im = twid_im / (float)nfft;
        }
        r[iy].re = twid_re;
        r[iy].im = temp_im;
      }
      xoff = y->size[0];
      iy = r1->size[0];
      for (ju = 0; ju < iy; ju++) {
        y_data[ju + xoff * chan] = r[ju];
      }
    }
    emxFree_creal32_T(&r1);
    emxFree_creal32_T(&fy);
    emxFree_creal32_T(&fv);
  }
  emxFree_creal32_T(&wwc);
}

/*
 * Arguments    : const emxArray_creal32_T *x
 *                int n1_unsigned
 *                const emxArray_real32_T *costab
 *                const emxArray_real32_T *sintab
 *                emxArray_creal32_T *y
 * Return Type  : void
 */
void c_FFTImplementationCallback_r2b(const emxArray_creal32_T *x,
                                     int n1_unsigned,
                                     const emxArray_real32_T *costab,
                                     const emxArray_real32_T *sintab,
                                     emxArray_creal32_T *y)
{
  emxArray_creal32_T *r1;
  const creal32_T *x_data;
  creal32_T *r;
  creal32_T *y_data;
  const float *costab_data;
  const float *sintab_data;
  float b;
  float temp_im;
  float temp_re;
  float temp_re_tmp;
  float twid_im;
  float twid_re;
  int b_i;
  int chan;
  int i;
  int iDelta;
  int iheight;
  int ihi;
  int iy;
  int j;
  int ju;
  int k;
  int loop_ub;
  int nRowsD2;
  int nrows;
  int xoff;
  boolean_T tst;
  sintab_data = sintab->data;
  costab_data = costab->data;
  x_data = x->data;
  nrows = x->size[0];
  i = y->size[0] * y->size[1] * y->size[2];
  y->size[0] = n1_unsigned;
  y->size[1] = x->size[1];
  y->size[2] = x->size[2];
  emxEnsureCapacity_creal32_T(y, i);
  y_data = y->data;
  if (n1_unsigned > x->size[0]) {
    i = y->size[0] * y->size[1] * y->size[2];
    y->size[0] = n1_unsigned;
    y->size[1] = x->size[1];
    y->size[2] = x->size[2];
    emxEnsureCapacity_creal32_T(y, i);
    y_data = y->data;
    loop_ub = n1_unsigned * x->size[1] * x->size[2];
    for (i = 0; i < loop_ub; i++) {
      y_data[i].re = 0.0F;
      y_data[i].im = 0.0F;
    }
  }
  loop_ub = x->size[1] * x->size[2] - 1;
#pragma omp parallel num_threads(omp_get_max_threads()) private(               \
    r, r1, xoff, ju, iy, iDelta, ihi, nRowsD2, k, b_i, j, tst, temp_re_tmp,    \
    temp_im, temp_re, twid_re, iheight, twid_im)
  {
    emxInit_creal32_T(&r1, 1);
#pragma omp for nowait
    for (chan = 0; chan <= loop_ub; chan++) {
      xoff = chan * nrows;
      ju = r1->size[0];
      r1->size[0] = n1_unsigned;
      emxEnsureCapacity_creal32_T(r1, ju);
      r = r1->data;
      if (n1_unsigned > x->size[0]) {
        ju = r1->size[0];
        r1->size[0] = n1_unsigned;
        emxEnsureCapacity_creal32_T(r1, ju);
        r = r1->data;
        for (ju = 0; ju < n1_unsigned; ju++) {
          r[ju].re = 0.0F;
          r[ju].im = 0.0F;
        }
      }
      iy = x->size[0];
      iDelta = n1_unsigned;
      if (iy <= n1_unsigned) {
        iDelta = iy;
      }
      ihi = n1_unsigned - 2;
      nRowsD2 = (int)((unsigned int)n1_unsigned >> 1);
      k = nRowsD2 / 2;
      iy = 0;
      ju = 0;
      for (b_i = 0; b_i <= iDelta - 2; b_i++) {
        r[iy] = x_data[xoff + b_i];
        j = n1_unsigned;
        tst = true;
        while (tst) {
          j >>= 1;
          ju ^= j;
          tst = ((ju & j) == 0);
        }
        iy = ju;
      }
      if (iDelta - 2 >= 0) {
        xoff = (xoff + iDelta) - 1;
      }
      r[iy] = x_data[xoff];
      if (n1_unsigned > 1) {
        for (b_i = 0; b_i <= ihi; b_i += 2) {
          temp_re_tmp = r[b_i + 1].re;
          temp_im = r[b_i + 1].im;
          temp_re = r[b_i].re;
          twid_re = r[b_i].im;
          r[b_i + 1].re = temp_re - temp_re_tmp;
          r[b_i + 1].im = twid_re - temp_im;
          r[b_i].re = temp_re + temp_re_tmp;
          r[b_i].im = twid_re + temp_im;
        }
      }
      iDelta = 2;
      xoff = 4;
      iheight = ((k - 1) << 2) + 1;
      while (k > 0) {
        for (b_i = 0; b_i < iheight; b_i += xoff) {
          iy = b_i + iDelta;
          temp_re = r[iy].re;
          temp_im = r[iy].im;
          r[iy].re = r[b_i].re - temp_re;
          r[iy].im = r[b_i].im - temp_im;
          r[b_i].re += temp_re;
          r[b_i].im += temp_im;
        }
        iy = 1;
        for (j = k; j < nRowsD2; j += k) {
          twid_re = costab_data[j];
          twid_im = sintab_data[j];
          b_i = iy;
          ihi = iy + iheight;
          while (b_i < ihi) {
            ju = b_i + iDelta;
            temp_re_tmp = r[ju].im;
            temp_im = r[ju].re;
            temp_re = twid_re * temp_im - twid_im * temp_re_tmp;
            temp_im = twid_re * temp_re_tmp + twid_im * temp_im;
            r[ju].re = r[b_i].re - temp_re;
            r[ju].im = r[b_i].im - temp_im;
            r[b_i].re += temp_re;
            r[b_i].im += temp_im;
            b_i += xoff;
          }
          iy++;
        }
        k /= 2;
        iDelta = xoff;
        xoff += xoff;
        iheight -= iDelta;
      }
      iy = y->size[0];
      j = r1->size[0];
      for (ju = 0; ju < j; ju++) {
        y_data[ju + iy * chan] = r[ju];
      }
    }
    emxFree_creal32_T(&r1);
  }
  if (y->size[0] > 1) {
    b = 1.0F / (float)y->size[0];
    loop_ub = y->size[0] * y->size[1] * y->size[2];
    for (i = 0; i < loop_ub; i++) {
      y_data[i].re *= b;
      y_data[i].im *= b;
    }
  }
}

/*
 * File trailer for FFTImplementationCallback.c
 *
 * [EOF]
 */
