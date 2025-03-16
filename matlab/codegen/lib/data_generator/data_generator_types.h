/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: data_generator_types.h
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

#ifndef DATA_GENERATOR_TYPES_H
#define DATA_GENERATOR_TYPES_H

/* Include Files */
#include "rtwtypes.h"

/* Type Definitions */
#ifndef struct_emxArray_boolean_T
#define struct_emxArray_boolean_T
struct emxArray_boolean_T {
  boolean_T *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
};
#endif /* struct_emxArray_boolean_T */
#ifndef typedef_emxArray_boolean_T
#define typedef_emxArray_boolean_T
typedef struct emxArray_boolean_T emxArray_boolean_T;
#endif /* typedef_emxArray_boolean_T */

#ifndef struct_emxArray_int32_T
#define struct_emxArray_int32_T
struct emxArray_int32_T {
  int *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
};
#endif /* struct_emxArray_int32_T */
#ifndef typedef_emxArray_int32_T
#define typedef_emxArray_int32_T
typedef struct emxArray_int32_T emxArray_int32_T;
#endif /* typedef_emxArray_int32_T */

#ifndef typedef_c_sparse
#define typedef_c_sparse
typedef struct {
  emxArray_boolean_T *d;
  emxArray_int32_T *colidx;
  emxArray_int32_T *rowidx;
  int m;
  int n;
} c_sparse;
#endif /* typedef_c_sparse */

#ifndef typedef_c_struct_T
#define typedef_c_struct_T
typedef struct {
  signed char EncodingMethod;
  emxArray_int32_T *MatrixL_RowIndices;
  emxArray_int32_T *MatrixL_RowStartLoc;
  emxArray_int32_T *MatrixL_ColumnSum;
  emxArray_int32_T *RowOrder;
  emxArray_int32_T *MatrixA_RowIndices;
  emxArray_int32_T *MatrixA_RowStartLoc;
  emxArray_int32_T *MatrixA_ColumnSum;
  emxArray_int32_T *MatrixB_RowIndices;
  emxArray_int32_T *MatrixB_RowStartLoc;
  emxArray_int32_T *MatrixB_ColumnSum;
} c_struct_T;
#endif /* typedef_c_struct_T */

#ifndef typedef_ldpcEncoderConfig
#define typedef_ldpcEncoderConfig
typedef struct {
  c_sparse ParityCheckMatrix;
  double BlockLength;
  double NumInformationBits;
  double NumParityCheckBits;
  c_struct_T derivedParams;
} ldpcEncoderConfig;
#endif /* typedef_ldpcEncoderConfig */

#ifndef typedef_emxArray_creal32_T
#define typedef_emxArray_creal32_T
typedef struct {
  creal32_T *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
} emxArray_creal32_T;
#endif /* typedef_emxArray_creal32_T */

#ifndef struct_emxArray_real_T
#define struct_emxArray_real_T
struct emxArray_real_T {
  double *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
};
#endif /* struct_emxArray_real_T */
#ifndef typedef_emxArray_real_T
#define typedef_emxArray_real_T
typedef struct emxArray_real_T emxArray_real_T;
#endif /* typedef_emxArray_real_T */

#ifndef struct_emxArray_int8_T
#define struct_emxArray_int8_T
struct emxArray_int8_T {
  signed char *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
};
#endif /* struct_emxArray_int8_T */
#ifndef typedef_emxArray_int8_T
#define typedef_emxArray_int8_T
typedef struct emxArray_int8_T emxArray_int8_T;
#endif /* typedef_emxArray_int8_T */

#ifndef typedef_emxArray_creal_T
#define typedef_emxArray_creal_T
typedef struct {
  creal_T *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
} emxArray_creal_T;
#endif /* typedef_emxArray_creal_T */

#ifndef struct_emxArray_uint64_T
#define struct_emxArray_uint64_T
struct emxArray_uint64_T {
  unsigned long long *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
};
#endif /* struct_emxArray_uint64_T */
#ifndef typedef_emxArray_uint64_T
#define typedef_emxArray_uint64_T
typedef struct emxArray_uint64_T emxArray_uint64_T;
#endif /* typedef_emxArray_uint64_T */

#ifndef struct_emxArray_uint32_T
#define struct_emxArray_uint32_T
struct emxArray_uint32_T {
  unsigned int *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
};
#endif /* struct_emxArray_uint32_T */
#ifndef typedef_emxArray_uint32_T
#define typedef_emxArray_uint32_T
typedef struct emxArray_uint32_T emxArray_uint32_T;
#endif /* typedef_emxArray_uint32_T */

#ifndef struct_emxArray_int16_T
#define struct_emxArray_int16_T
struct emxArray_int16_T {
  short *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
};
#endif /* struct_emxArray_int16_T */
#ifndef typedef_emxArray_int16_T
#define typedef_emxArray_int16_T
typedef struct emxArray_int16_T emxArray_int16_T;
#endif /* typedef_emxArray_int16_T */

#ifndef struct_emxArray_real32_T
#define struct_emxArray_real32_T
struct emxArray_real32_T {
  float *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
};
#endif /* struct_emxArray_real32_T */
#ifndef typedef_emxArray_real32_T
#define typedef_emxArray_real32_T
typedef struct emxArray_real32_T emxArray_real32_T;
#endif /* typedef_emxArray_real32_T */

#ifndef struct_emxArray_uint8_T
#define struct_emxArray_uint8_T
struct emxArray_uint8_T {
  unsigned char *data;
  int *size;
  int allocatedSize;
  int numDimensions;
  boolean_T canFreeData;
};
#endif /* struct_emxArray_uint8_T */
#ifndef typedef_emxArray_uint8_T
#define typedef_emxArray_uint8_T
typedef struct emxArray_uint8_T emxArray_uint8_T;
#endif /* typedef_emxArray_uint8_T */

#endif
/*
 * File trailer for data_generator_types.h
 *
 * [EOF]
 */
