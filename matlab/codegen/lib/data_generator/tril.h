/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: tril.h
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

#ifndef TRIL_H
#define TRIL_H

/* Include Files */
#include "data_generator_types.h"
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
int sparse_tril(const emxArray_boolean_T *A_d, const emxArray_int32_T *A_colidx,
                const emxArray_int32_T *A_rowidx, int A_m, int A_n,
                emxArray_boolean_T *L_d, emxArray_int32_T *L_colidx,
                emxArray_int32_T *L_rowidx, int *L_n);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for tril.h
 *
 * [EOF]
 */
