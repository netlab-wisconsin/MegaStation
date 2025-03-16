/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: locBsearch.h
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

#ifndef LOCBSEARCH_H
#define LOCBSEARCH_H

/* Include Files */
#include "data_generator_types.h"
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
int sparse_locBsearch(const emxArray_int32_T *x, int xi, int xstart, int xend,
                      boolean_T *found);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for locBsearch.h
 *
 * [EOF]
 */
