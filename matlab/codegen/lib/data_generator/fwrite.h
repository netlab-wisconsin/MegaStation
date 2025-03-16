/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: fwrite.h
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

#ifndef FWRITE_H
#define FWRITE_H

/* Include Files */
#include "data_generator_types.h"
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
void b_fwrite(double fileID, const emxArray_creal32_T *x);

void c_fwrite(double fileID, const emxArray_real_T *x);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for fwrite.h
 *
 * [EOF]
 */
