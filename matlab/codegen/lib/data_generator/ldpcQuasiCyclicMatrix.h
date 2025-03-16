/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: ldpcQuasiCyclicMatrix.h
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

#ifndef LDPCQUASICYCLICMATRIX_H
#define LDPCQUASICYCLICMATRIX_H

/* Include Files */
#include "data_generator_types.h"
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
int ldpcQuasiCyclicMatrix(double blockSize, const double P_data[],
                          const int P_size[2], emxArray_boolean_T *H_d,
                          emxArray_int32_T *H_colidx,
                          emxArray_int32_T *H_rowidx, int *H_n);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for ldpcQuasiCyclicMatrix.h
 *
 * [EOF]
 */
