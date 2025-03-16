/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: heapsort.h
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

#ifndef HEAPSORT_H
#define HEAPSORT_H

/* Include Files */
#include "data_generator_types.h"
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
void b_heapsort(emxArray_int32_T *x, int xstart, int xend,
                const emxArray_int32_T *cmp_workspace_a,
                const emxArray_int32_T *cmp_workspace_b);

void c_heapsort(emxArray_int32_T *x, int xstart, int xend,
                const emxArray_int32_T *cmp_workspace_x);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for heapsort.h
 *
 * [EOF]
 */
