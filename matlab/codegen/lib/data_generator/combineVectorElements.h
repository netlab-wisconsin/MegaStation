/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: combineVectorElements.h
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

#ifndef COMBINEVECTORELEMENTS_H
#define COMBINEVECTORELEMENTS_H

/* Include Files */
#include "data_generator_types.h"
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
int combineVectorElements(const emxArray_boolean_T *x_d,
                          const emxArray_int32_T *x_colidx, int x_n,
                          emxArray_real_T *y_d, emxArray_int32_T *y_colidx,
                          emxArray_int32_T *y_rowidx);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for combineVectorElements.h
 *
 * [EOF]
 */
