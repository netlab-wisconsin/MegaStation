/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: data_generator_emxutil.h
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

#ifndef DATA_GENERATOR_EMXUTIL_H
#define DATA_GENERATOR_EMXUTIL_H

/* Include Files */
#include "data_generator_types.h"
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
extern void emxCopyStruct_ldpcEncoderConfig(ldpcEncoderConfig *dst,
                                            const ldpcEncoderConfig *src);

extern void emxCopyStruct_sparse(c_sparse *dst, const c_sparse *src);

extern void emxCopyStruct_struct_T(c_struct_T *dst, const c_struct_T *src);

extern void emxCopy_boolean_T(emxArray_boolean_T **dst,
                              emxArray_boolean_T *const *src);

extern void emxCopy_int32_T(emxArray_int32_T **dst,
                            emxArray_int32_T *const *src);

extern void emxEnsureCapacity_boolean_T(emxArray_boolean_T *emxArray,
                                        int oldNumel);

extern void emxEnsureCapacity_creal32_T(emxArray_creal32_T *emxArray,
                                        int oldNumel);

extern void emxEnsureCapacity_creal_T(emxArray_creal_T *emxArray, int oldNumel);

extern void emxEnsureCapacity_int16_T(emxArray_int16_T *emxArray, int oldNumel);

extern void emxEnsureCapacity_int32_T(emxArray_int32_T *emxArray, int oldNumel);

extern void emxEnsureCapacity_int8_T(emxArray_int8_T *emxArray, int oldNumel);

extern void emxEnsureCapacity_real32_T(emxArray_real32_T *emxArray,
                                       int oldNumel);

extern void emxEnsureCapacity_real_T(emxArray_real_T *emxArray, int oldNumel);

extern void emxEnsureCapacity_uint32_T(emxArray_uint32_T *emxArray,
                                       int oldNumel);

extern void emxEnsureCapacity_uint64_T(emxArray_uint64_T *emxArray,
                                       int oldNumel);

extern void emxEnsureCapacity_uint8_T(emxArray_uint8_T *emxArray, int oldNumel);

extern void emxFreeMatrix_ldpcEncoderConfig(ldpcEncoderConfig pMatrix[6144]);

extern void emxFreeStruct_ldpcEncoderConfig(ldpcEncoderConfig *pStruct);

extern void emxFreeStruct_sparse(c_sparse *pStruct);

extern void emxFreeStruct_struct_T(c_struct_T *pStruct);

extern void emxFree_boolean_T(emxArray_boolean_T **pEmxArray);

extern void emxFree_creal32_T(emxArray_creal32_T **pEmxArray);

extern void emxFree_creal_T(emxArray_creal_T **pEmxArray);

extern void emxFree_int16_T(emxArray_int16_T **pEmxArray);

extern void emxFree_int32_T(emxArray_int32_T **pEmxArray);

extern void emxFree_int8_T(emxArray_int8_T **pEmxArray);

extern void emxFree_real32_T(emxArray_real32_T **pEmxArray);

extern void emxFree_real_T(emxArray_real_T **pEmxArray);

extern void emxFree_uint32_T(emxArray_uint32_T **pEmxArray);

extern void emxFree_uint64_T(emxArray_uint64_T **pEmxArray);

extern void emxFree_uint8_T(emxArray_uint8_T **pEmxArray);

extern void emxInitMatrix_ldpcEncoderConfig(ldpcEncoderConfig pMatrix[6144]);

extern void emxInitStruct_ldpcEncoderConfig(ldpcEncoderConfig *pStruct);

extern void emxInitStruct_sparse(c_sparse *pStruct);

extern void emxInitStruct_struct_T(c_struct_T *pStruct);

extern void emxInit_boolean_T(emxArray_boolean_T **pEmxArray,
                              int numDimensions);

extern void emxInit_creal32_T(emxArray_creal32_T **pEmxArray,
                              int numDimensions);

extern void emxInit_creal_T(emxArray_creal_T **pEmxArray, int numDimensions);

extern void emxInit_int16_T(emxArray_int16_T **pEmxArray);

extern void emxInit_int32_T(emxArray_int32_T **pEmxArray, int numDimensions);

extern void emxInit_int8_T(emxArray_int8_T **pEmxArray, int numDimensions);

extern void emxInit_real32_T(emxArray_real32_T **pEmxArray, int numDimensions);

extern void emxInit_real_T(emxArray_real_T **pEmxArray, int numDimensions);

extern void emxInit_uint32_T(emxArray_uint32_T **pEmxArray, int numDimensions);

extern void emxInit_uint64_T(emxArray_uint64_T **pEmxArray);

extern void emxInit_uint8_T(emxArray_uint8_T **pEmxArray);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for data_generator_emxutil.h
 *
 * [EOF]
 */
