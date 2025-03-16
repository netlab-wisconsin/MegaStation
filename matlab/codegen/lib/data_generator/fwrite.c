/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: fwrite.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "fwrite.h"
#include "data_generator_data.h"
#include "data_generator_emxutil.h"
#include "data_generator_rtwutil.h"
#include "data_generator_types.h"
#include "rt_nonfinite.h"
#include <stddef.h>
#include <stdio.h>

/* Function Declarations */
static FILE *getFileStar(double fileID, boolean_T *autoflush);

/* Function Definitions */
/*
 * Arguments    : double fileID
 *                boolean_T *autoflush
 * Return Type  : FILE *
 */
static FILE *getFileStar(double fileID, boolean_T *autoflush)
{
  FILE *filestar;
  signed char fileid;
  fileid = (signed char)fileID;
  if (((signed char)fileID < 0) || (fileID != (signed char)fileID)) {
    fileid = -1;
  }
  if (fileid >= 3) {
    *autoflush = eml_autoflush[fileid - 3];
    filestar = eml_openfiles[fileid - 3];
  } else if (fileid == 0) {
    filestar = stdin;
    *autoflush = true;
  } else if (fileid == 1) {
    filestar = stdout;
    *autoflush = true;
  } else if (fileid == 2) {
    filestar = stderr;
    *autoflush = true;
  } else {
    filestar = NULL;
    *autoflush = true;
  }
  if (!(fileID != 0.0)) {
    filestar = NULL;
  }
  return filestar;
}

/*
 * Arguments    : double fileID
 *                const emxArray_creal32_T *x
 * Return Type  : void
 */
void b_fwrite(double fileID, const emxArray_creal32_T *x)
{
  FILE *filestar;
  emxArray_real32_T *b_x;
  const creal32_T *x_data;
  float *b_x_data;
  int i;
  int loop_ub;
  boolean_T autoflush;
  x_data = x->data;
  emxInit_real32_T(&b_x, 3);
  i = b_x->size[0] * b_x->size[1] * b_x->size[2];
  b_x->size[0] = x->size[0];
  b_x->size[1] = x->size[1];
  b_x->size[2] = x->size[2];
  emxEnsureCapacity_real32_T(b_x, i);
  b_x_data = b_x->data;
  loop_ub = x->size[0] * x->size[1] * x->size[2];
  for (i = 0; i < loop_ub; i++) {
    b_x_data[i] = x_data[i].re;
  }
  filestar = getFileStar(fileID, &autoflush);
  if ((!(filestar == NULL)) &&
      ((b_x->size[0] != 0) && (b_x->size[1] != 0) && (b_x->size[2] != 0))) {
    size_t bytesOutSizet;
    bytesOutSizet =
        fwrite(&b_x_data[0], sizeof(float),
               (size_t)(b_x->size[0] * b_x->size[1] * b_x->size[2]), filestar);
    if (((double)bytesOutSizet > 0.0) && autoflush) {
      fflush(filestar);
    }
  }
  emxFree_real32_T(&b_x);
}

/*
 * Arguments    : double fileID
 *                const emxArray_real_T *x
 * Return Type  : void
 */
void c_fwrite(double fileID, const emxArray_real_T *x)
{
  FILE *filestar;
  emxArray_uint8_T *xout;
  const double *x_data;
  int i;
  unsigned char *xout_data;
  boolean_T autoflush;
  x_data = x->data;
  filestar = getFileStar(fileID, &autoflush);
  emxInit_uint8_T(&xout);
  if ((!(filestar == NULL)) &&
      ((x->size[0] != 0) && (x->size[1] != 0) && (x->size[2] != 0))) {
    size_t bytesOutSizet;
    int loop_ub;
    i = xout->size[0] * xout->size[1] * xout->size[2];
    xout->size[0] = x->size[0];
    xout->size[1] = x->size[1];
    xout->size[2] = x->size[2];
    emxEnsureCapacity_uint8_T(xout, i);
    xout_data = xout->data;
    loop_ub = x->size[0] * x->size[1] * x->size[2];
    for (i = 0; i < loop_ub; i++) {
      double d;
      unsigned char u;
      d = rt_roundd_snf(x_data[i]);
      if (d < 256.0) {
        if (d >= 0.0) {
          u = (unsigned char)d;
        } else {
          u = 0U;
        }
      } else if (d >= 256.0) {
        u = MAX_uint8_T;
      } else {
        u = 0U;
      }
      xout_data[i] = u;
    }
    bytesOutSizet =
        fwrite(&xout_data[0], sizeof(unsigned char),
               (size_t)(x->size[0] * x->size[1] * x->size[2]), filestar);
    if (((double)bytesOutSizet > 0.0) && autoflush) {
      fflush(filestar);
    }
  }
  emxFree_uint8_T(&xout);
}

/*
 * File trailer for fwrite.c
 *
 * [EOF]
 */
