/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: eml_setop.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "eml_setop.h"
#include "data_generator_emxutil.h"
#include "data_generator_types.h"
#include "rt_nonfinite.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : const emxArray_real_T *a
 *                const emxArray_real_T *b
 *                emxArray_real_T *c
 *                emxArray_int32_T *ia
 * Return Type  : int
 */
int do_vectors(const emxArray_real_T *a, const emxArray_real_T *b,
               emxArray_real_T *c, emxArray_int32_T *ia)
{
  const double *a_data;
  const double *b_data;
  double *c_data;
  int b_ialast;
  int iafirst;
  int ialast;
  int ib_size;
  int iblast;
  int na;
  int nc;
  int nia;
  int *ia_data;
  b_data = b->data;
  a_data = a->data;
  na = a->size[0];
  nia = c->size[0];
  c->size[0] = a->size[0];
  emxEnsureCapacity_real_T(c, nia);
  c_data = c->data;
  nia = ia->size[0];
  ia->size[0] = a->size[0];
  emxEnsureCapacity_int32_T(ia, nia);
  ia_data = ia->data;
  ib_size = 0;
  nc = 0;
  nia = 0;
  iafirst = 0;
  ialast = 0;
  iblast = 1;
  while ((ialast + 1 <= na) && (iblast <= b->size[0])) {
    double ak;
    double bk;
    b_ialast = ialast + 1;
    ak = a_data[ialast];
    while ((b_ialast < a->size[0]) && (a_data[b_ialast] == ak)) {
      b_ialast++;
    }
    ialast = b_ialast - 1;
    bk = b_data[iblast - 1];
    while ((iblast < b->size[0]) && (b_data[iblast] == bk)) {
      iblast++;
    }
    if (ak == bk) {
      ialast = b_ialast;
      iafirst = b_ialast;
      iblast++;
    } else {
      boolean_T p;
      if (rtIsNaN(bk)) {
        p = !rtIsNaN(ak);
      } else if (rtIsNaN(ak)) {
        p = false;
      } else {
        p = (ak < bk);
      }
      if (p) {
        nia = nc + 1;
        nc++;
        c_data[nia - 1] = ak;
        ia_data[nia - 1] = iafirst + 1;
        ialast = b_ialast;
        iafirst = b_ialast;
      } else {
        iblast++;
      }
    }
  }
  while (ialast + 1 <= na) {
    b_ialast = ialast + 1;
    while ((b_ialast < a->size[0]) && (a_data[b_ialast] == a_data[ialast])) {
      b_ialast++;
    }
    nia = nc + 1;
    nc++;
    c_data[nia - 1] = a_data[ialast];
    ia_data[nia - 1] = iafirst + 1;
    ialast = b_ialast;
    iafirst = b_ialast;
  }
  if (nia < 1) {
    nia = 0;
  }
  nc = ia->size[0];
  ia->size[0] = nia;
  emxEnsureCapacity_int32_T(ia, nc);
  nc = c->size[0];
  c->size[0] = nia;
  emxEnsureCapacity_real_T(c, nc);
  return ib_size;
}

/*
 * File trailer for eml_setop.c
 *
 * [EOF]
 */
