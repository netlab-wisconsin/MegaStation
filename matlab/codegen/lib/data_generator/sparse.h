/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: sparse.h
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

#ifndef SPARSE_H
#define SPARSE_H

/* Include Files */
#include "data_generator_types.h"
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
int b_sparse(const emxArray_boolean_T *varargin_1, emxArray_boolean_T *y_d,
             emxArray_int32_T *y_colidx, emxArray_int32_T *y_rowidx, int *y_n);

void sparse(const emxArray_real_T *varargin_1,
            const emxArray_real_T *varargin_2, double varargin_4,
            double varargin_5, c_sparse *y);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for sparse.h
 *
 * [EOF]
 */
