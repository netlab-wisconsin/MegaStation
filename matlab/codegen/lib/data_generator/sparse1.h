/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: sparse1.h
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

#ifndef SPARSE1_H
#define SPARSE1_H

/* Include Files */
#include "data_generator_types.h"
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
void locSortrows(emxArray_int32_T *idx, emxArray_int32_T *a,
                 emxArray_int32_T *b);

void sparse_full(const emxArray_boolean_T *this_d,
                 const emxArray_int32_T *this_colidx,
                 const emxArray_int32_T *this_rowidx, int this_m, int this_n,
                 emxArray_boolean_T *y);

void sparse_parenReference(const emxArray_boolean_T *this_d,
                           const emxArray_int32_T *this_colidx,
                           const emxArray_int32_T *this_rowidx, int this_m,
                           const emxArray_real_T *varargin_2, c_sparse *s);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for sparse1.h
 *
 * [EOF]
 */
