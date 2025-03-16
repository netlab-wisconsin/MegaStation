/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: ldpcEncoderConfig.h
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

#ifndef LDPCENCODERCONFIG_H
#define LDPCENCODERCONFIG_H

/* Include Files */
#include "data_generator_types.h"
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
signed char c_ldpcEncoderConfig_CalcDerived(
    const emxArray_boolean_T *obj_ParityCheckMatrix_d,
    const emxArray_int32_T *obj_ParityCheckMatrix_colidx,
    const emxArray_int32_T *obj_ParityCheckMatrix_rowidx,
    int obj_ParityCheckMatrix_m, int obj_ParityCheckMatrix_n,
    emxArray_int32_T *c_derivedParams_MatrixL_RowIndi,
    emxArray_int32_T *c_derivedParams_MatrixL_RowStar,
    emxArray_int32_T *derivedParams_MatrixL_ColumnSum,
    emxArray_int32_T *derivedParams_RowOrder,
    emxArray_int32_T *c_derivedParams_MatrixA_RowIndi,
    emxArray_int32_T *c_derivedParams_MatrixA_RowStar,
    emxArray_int32_T *derivedParams_MatrixA_ColumnSum,
    emxArray_int32_T *c_derivedParams_MatrixB_RowIndi,
    emxArray_int32_T *c_derivedParams_MatrixB_RowStar,
    emxArray_int32_T *derivedParams_MatrixB_ColumnSum);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for ldpcEncoderConfig.h
 *
 * [EOF]
 */
