/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: ldpcEncode.h
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

#ifndef LDPCENCODE_H
#define LDPCENCODE_H

/* Include Files */
#include "data_generator_types.h"
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
void ldpcEncode(const emxArray_real_T *informationBits,
                double c_encoderConfig_NumParityCheckB,
                signed char c_encoderConfig_derivedParams_E,
                const emxArray_int32_T *c_encoderConfig_derivedParams_M,
                const emxArray_int32_T *d_encoderConfig_derivedParams_M,
                const emxArray_int32_T *e_encoderConfig_derivedParams_M,
                const emxArray_int32_T *c_encoderConfig_derivedParams_R,
                const emxArray_int32_T *f_encoderConfig_derivedParams_M,
                const emxArray_int32_T *g_encoderConfig_derivedParams_M,
                const emxArray_int32_T *h_encoderConfig_derivedParams_M,
                const emxArray_int32_T *i_encoderConfig_derivedParams_M,
                const emxArray_int32_T *j_encoderConfig_derivedParams_M,
                const emxArray_int32_T *k_encoderConfig_derivedParams_M,
                emxArray_real_T *output);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for ldpcEncode.h
 *
 * [EOF]
 */
