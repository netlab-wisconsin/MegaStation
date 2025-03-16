/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: ldpcEncode.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "ldpcEncode.h"
#include "data_generator_emxutil.h"
#include "data_generator_types.h"
#include "rt_nonfinite.h"

/* Function Declarations */
static void GF2MatrixMul(const emxArray_real_T *source, double destLen,
                         const emxArray_int32_T *RowIndices,
                         const emxArray_int32_T *rowloc,
                         const emxArray_int32_T *ColumnSum,
                         emxArray_real_T *dest);

static void GF2Subst(emxArray_real_T *srcdest,
                     const emxArray_int32_T *RowIndices,
                     const emxArray_int32_T *rowloc,
                     const emxArray_int32_T *ColumnSum, signed char direction);

/* Function Definitions */
/*
 * Arguments    : const emxArray_real_T *source
 *                double destLen
 *                const emxArray_int32_T *RowIndices
 *                const emxArray_int32_T *rowloc
 *                const emxArray_int32_T *ColumnSum
 *                emxArray_real_T *dest
 * Return Type  : void
 */
static void GF2MatrixMul(const emxArray_real_T *source, double destLen,
                         const emxArray_int32_T *RowIndices,
                         const emxArray_int32_T *rowloc,
                         const emxArray_int32_T *ColumnSum,
                         emxArray_real_T *dest)
{
  emxArray_int32_T *rowindex;
  emxArray_int32_T *y;
  emxArray_real_T *r;
  const double *source_data;
  double *dest_data;
  double *r1;
  const int *ColumnSum_data;
  const int *RowIndices_data;
  const int *rowloc_data;
  int columnindex;
  int i;
  int i1;
  int k;
  int yk;
  int *rowindex_data;
  int *y_data;
  ColumnSum_data = ColumnSum->data;
  rowloc_data = rowloc->data;
  RowIndices_data = RowIndices->data;
  source_data = source->data;
  yk = (int)destLen;
  i = dest->size[0];
  dest->size[0] = (int)destLen;
  emxEnsureCapacity_real_T(dest, i);
  dest_data = dest->data;
  for (i = 0; i < yk; i++) {
    dest_data[i] = 0.0;
  }
  i = source->size[0];
  emxInit_int32_T(&rowindex, 1);
  emxInit_int32_T(&y, 2);
  emxInit_real_T(&r, 1);
  for (columnindex = 0; columnindex < i; columnindex++) {
    if (source_data[columnindex] != 0.0) {
      int n;
      if (ColumnSum_data[columnindex] < 1) {
        n = 0;
      } else {
        n = ColumnSum_data[columnindex];
      }
      i1 = y->size[0] * y->size[1];
      y->size[0] = 1;
      y->size[1] = n;
      emxEnsureCapacity_int32_T(y, i1);
      y_data = y->data;
      if (n > 0) {
        y_data[0] = 1;
        yk = 1;
        for (k = 2; k <= n; k++) {
          yk++;
          y_data[k - 1] = yk;
        }
      }
      n = rowloc_data[columnindex];
      i1 = rowindex->size[0];
      rowindex->size[0] = y->size[1];
      emxEnsureCapacity_int32_T(rowindex, i1);
      rowindex_data = rowindex->data;
      k = y->size[1];
      for (i1 = 0; i1 < k; i1++) {
        yk = y_data[i1];
        if ((n < 0) && (yk < MIN_int32_T - n)) {
          yk = MIN_int32_T;
        } else if ((n > 0) && (yk > MAX_int32_T - n)) {
          yk = MAX_int32_T;
        } else {
          yk += n;
        }
        yk = RowIndices_data[yk - 1];
        if (yk > 2147483646) {
          yk = MAX_int32_T;
        } else {
          yk++;
        }
        rowindex_data[i1] = yk;
      }
      i1 = r->size[0];
      r->size[0] = rowindex->size[0];
      emxEnsureCapacity_real_T(r, i1);
      r1 = r->data;
      k = rowindex->size[0];
      for (i1 = 0; i1 < k; i1++) {
        r1[i1] = 1.0 - dest_data[rowindex_data[i1] - 1];
      }
      k = r->size[0];
      for (i1 = 0; i1 < k; i1++) {
        dest_data[rowindex_data[i1] - 1] = r1[i1];
      }
    }
  }
  emxFree_real_T(&r);
  emxFree_int32_T(&y);
  emxFree_int32_T(&rowindex);
}

/*
 * Arguments    : emxArray_real_T *srcdest
 *                const emxArray_int32_T *RowIndices
 *                const emxArray_int32_T *rowloc
 *                const emxArray_int32_T *ColumnSum
 *                signed char direction
 * Return Type  : void
 */
static void GF2Subst(emxArray_real_T *srcdest,
                     const emxArray_int32_T *RowIndices,
                     const emxArray_int32_T *rowloc,
                     const emxArray_int32_T *ColumnSum, signed char direction)
{
  emxArray_int32_T *rowindex;
  emxArray_int32_T *y;
  emxArray_real_T *r;
  double *r1;
  double *srcdest_data;
  const int *ColumnSum_data;
  const int *RowIndices_data;
  const int *rowloc_data;
  int columnindex;
  int k;
  int *rowindex_data;
  int *y_data;
  ColumnSum_data = ColumnSum->data;
  rowloc_data = rowloc->data;
  RowIndices_data = RowIndices->data;
  srcdest_data = srcdest->data;
  emxInit_int32_T(&rowindex, 1);
  emxInit_int32_T(&y, 2);
  emxInit_real_T(&r, 1);
  if (direction == 1) {
    int i;
    i = srcdest->size[0];
    for (columnindex = 0; columnindex < i; columnindex++) {
      if (srcdest_data[columnindex] != 0.0) {
        int b_columnindex;
        int n;
        int yk;
        if (ColumnSum_data[columnindex] < 1) {
          n = 0;
        } else {
          n = ColumnSum_data[columnindex];
        }
        k = y->size[0] * y->size[1];
        y->size[0] = 1;
        y->size[1] = n;
        emxEnsureCapacity_int32_T(y, k);
        y_data = y->data;
        if (n > 0) {
          y_data[0] = 1;
          yk = 1;
          for (k = 2; k <= n; k++) {
            yk++;
            y_data[k - 1] = yk;
          }
        }
        b_columnindex = rowloc_data[columnindex];
        k = rowindex->size[0];
        rowindex->size[0] = y->size[1];
        emxEnsureCapacity_int32_T(rowindex, k);
        rowindex_data = rowindex->data;
        n = y->size[1];
        for (k = 0; k < n; k++) {
          yk = y_data[k];
          if ((b_columnindex < 0) && (yk < MIN_int32_T - b_columnindex)) {
            yk = MIN_int32_T;
          } else if ((b_columnindex > 0) &&
                     (yk > MAX_int32_T - b_columnindex)) {
            yk = MAX_int32_T;
          } else {
            yk += b_columnindex;
          }
          yk = RowIndices_data[yk - 1];
          if (yk > 2147483646) {
            yk = MAX_int32_T;
          } else {
            yk++;
          }
          rowindex_data[k] = yk;
        }
        k = r->size[0];
        r->size[0] = rowindex->size[0];
        emxEnsureCapacity_real_T(r, k);
        r1 = r->data;
        n = rowindex->size[0];
        for (k = 0; k < n; k++) {
          r1[k] = 1.0 - srcdest_data[rowindex_data[k] - 1];
        }
        n = r->size[0];
        for (k = 0; k < n; k++) {
          srcdest_data[rowindex_data[k] - 1] = r1[k];
        }
      }
    }
  } else {
    int i;
    i = srcdest->size[0];
    for (columnindex = 0; columnindex < i; columnindex++) {
      int b_columnindex;
      b_columnindex = (srcdest->size[0] - columnindex) - 1;
      if (srcdest_data[b_columnindex] != 0.0) {
        int n;
        int yk;
        if (ColumnSum_data[b_columnindex] < 1) {
          n = 0;
        } else {
          n = ColumnSum_data[b_columnindex];
        }
        k = y->size[0] * y->size[1];
        y->size[0] = 1;
        y->size[1] = n;
        emxEnsureCapacity_int32_T(y, k);
        y_data = y->data;
        if (n > 0) {
          y_data[0] = 1;
          yk = 1;
          for (k = 2; k <= n; k++) {
            yk++;
            y_data[k - 1] = yk;
          }
        }
        b_columnindex = rowloc_data[b_columnindex];
        k = rowindex->size[0];
        rowindex->size[0] = y->size[1];
        emxEnsureCapacity_int32_T(rowindex, k);
        rowindex_data = rowindex->data;
        n = y->size[1];
        for (k = 0; k < n; k++) {
          yk = y_data[k];
          if ((b_columnindex < 0) && (yk < MIN_int32_T - b_columnindex)) {
            yk = MIN_int32_T;
          } else if ((b_columnindex > 0) &&
                     (yk > MAX_int32_T - b_columnindex)) {
            yk = MAX_int32_T;
          } else {
            yk += b_columnindex;
          }
          yk = RowIndices_data[yk - 1];
          if (yk > 2147483646) {
            yk = MAX_int32_T;
          } else {
            yk++;
          }
          rowindex_data[k] = yk;
        }
        k = r->size[0];
        r->size[0] = rowindex->size[0];
        emxEnsureCapacity_real_T(r, k);
        r1 = r->data;
        n = rowindex->size[0];
        for (k = 0; k < n; k++) {
          r1[k] = 1.0 - srcdest_data[rowindex_data[k] - 1];
        }
        n = r->size[0];
        for (k = 0; k < n; k++) {
          srcdest_data[rowindex_data[k] - 1] = r1[k];
        }
      }
    }
  }
  emxFree_real_T(&r);
  emxFree_int32_T(&y);
  emxFree_int32_T(&rowindex);
}

/*
 * Arguments    : const emxArray_real_T *informationBits
 *                double c_encoderConfig_NumParityCheckB
 *                signed char c_encoderConfig_derivedParams_E
 *                const emxArray_int32_T *c_encoderConfig_derivedParams_M
 *                const emxArray_int32_T *d_encoderConfig_derivedParams_M
 *                const emxArray_int32_T *e_encoderConfig_derivedParams_M
 *                const emxArray_int32_T *c_encoderConfig_derivedParams_R
 *                const emxArray_int32_T *f_encoderConfig_derivedParams_M
 *                const emxArray_int32_T *g_encoderConfig_derivedParams_M
 *                const emxArray_int32_T *h_encoderConfig_derivedParams_M
 *                const emxArray_int32_T *i_encoderConfig_derivedParams_M
 *                const emxArray_int32_T *j_encoderConfig_derivedParams_M
 *                const emxArray_int32_T *k_encoderConfig_derivedParams_M
 *                emxArray_real_T *output
 * Return Type  : void
 */
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
                emxArray_real_T *output)
{
  emxArray_int32_T *rowindex;
  emxArray_int32_T *y;
  emxArray_real_T *MatrixProductBuffer;
  emxArray_real_T *ParityCheckBits;
  emxArray_real_T *b_informationBits;
  emxArray_real_T *r;
  const double *informationBits_data;
  double *MatrixProductBuffer_data;
  double *ParityCheckBits_data;
  double *output_data;
  const int *d_encoderConfig_derivedParams_R;
  const int *l_encoderConfig_derivedParams_M;
  const int *m_encoderConfig_derivedParams_M;
  const int *n_encoderConfig_derivedParams_M;
  int b_i;
  int columnindex;
  int i;
  int i1;
  int i2;
  int k;
  int n;
  int yk;
  int *rowindex_data;
  int *y_data;
  short sizes_idx_0;
  boolean_T b;
  boolean_T empty_non_axis_sizes;
  d_encoderConfig_derivedParams_R = c_encoderConfig_derivedParams_R->data;
  l_encoderConfig_derivedParams_M = e_encoderConfig_derivedParams_M->data;
  m_encoderConfig_derivedParams_M = d_encoderConfig_derivedParams_M->data;
  n_encoderConfig_derivedParams_M = c_encoderConfig_derivedParams_M->data;
  informationBits_data = informationBits->data;
  emxInit_real_T(&ParityCheckBits, 2);
  i = ParityCheckBits->size[0] * ParityCheckBits->size[1];
  ParityCheckBits->size[0] = (int)c_encoderConfig_NumParityCheckB;
  ParityCheckBits->size[1] = informationBits->size[1];
  emxEnsureCapacity_real_T(ParityCheckBits, i);
  ParityCheckBits_data = ParityCheckBits->data;
  i = informationBits->size[1];
  emxInit_real_T(&MatrixProductBuffer, 1);
  emxInit_real_T(&r, 1);
  emxInit_int32_T(&rowindex, 1);
  emxInit_int32_T(&y, 2);
  emxInit_real_T(&b_informationBits, 1);
  for (b_i = 0; b_i < i; b_i++) {
    if (d_encoderConfig_derivedParams_R[0] >= 0) {
      n = informationBits->size[0];
      i1 = b_informationBits->size[0];
      b_informationBits->size[0] = informationBits->size[0];
      emxEnsureCapacity_real_T(b_informationBits, i1);
      output_data = b_informationBits->data;
      for (i1 = 0; i1 < n; i1++) {
        output_data[i1] =
            informationBits_data[i1 + informationBits->size[0] * b_i];
      }
      GF2MatrixMul(b_informationBits, c_encoderConfig_NumParityCheckB,
                   f_encoderConfig_derivedParams_M,
                   g_encoderConfig_derivedParams_M,
                   h_encoderConfig_derivedParams_M, MatrixProductBuffer);
      MatrixProductBuffer_data = MatrixProductBuffer->data;
      if (c_encoderConfig_derivedParams_E == 0) {
        i1 = MatrixProductBuffer->size[0];
        for (columnindex = 0; columnindex < i1; columnindex++) {
          if (MatrixProductBuffer_data[columnindex] != 0.0) {
            int o_encoderConfig_derivedParams_M;
            if (l_encoderConfig_derivedParams_M[columnindex] < 1) {
              n = 0;
            } else {
              n = l_encoderConfig_derivedParams_M[columnindex];
            }
            i2 = y->size[0] * y->size[1];
            y->size[0] = 1;
            y->size[1] = n;
            emxEnsureCapacity_int32_T(y, i2);
            y_data = y->data;
            if (n > 0) {
              y_data[0] = 1;
              yk = 1;
              for (k = 2; k <= n; k++) {
                yk++;
                y_data[k - 1] = yk;
              }
            }
            o_encoderConfig_derivedParams_M =
                m_encoderConfig_derivedParams_M[columnindex];
            i2 = rowindex->size[0];
            rowindex->size[0] = y->size[1];
            emxEnsureCapacity_int32_T(rowindex, i2);
            rowindex_data = rowindex->data;
            n = y->size[1];
            for (i2 = 0; i2 < n; i2++) {
              k = y_data[i2];
              if ((o_encoderConfig_derivedParams_M < 0) &&
                  (k < MIN_int32_T - o_encoderConfig_derivedParams_M)) {
                yk = MIN_int32_T;
              } else if ((o_encoderConfig_derivedParams_M > 0) &&
                         (k > MAX_int32_T - o_encoderConfig_derivedParams_M)) {
                yk = MAX_int32_T;
              } else {
                yk = o_encoderConfig_derivedParams_M + k;
              }
              yk = n_encoderConfig_derivedParams_M[yk - 1];
              if (yk > 2147483646) {
                yk = MAX_int32_T;
              } else {
                yk++;
              }
              rowindex_data[i2] = yk;
            }
            i2 = b_informationBits->size[0];
            b_informationBits->size[0] = rowindex->size[0];
            emxEnsureCapacity_real_T(b_informationBits, i2);
            output_data = b_informationBits->data;
            n = rowindex->size[0];
            for (i2 = 0; i2 < n; i2++) {
              output_data[i2] =
                  1.0 - MatrixProductBuffer_data[rowindex_data[i2] - 1];
            }
            n = b_informationBits->size[0];
            for (i2 = 0; i2 < n; i2++) {
              MatrixProductBuffer_data[rowindex_data[i2] - 1] = output_data[i2];
            }
          }
        }
      }
      i1 = r->size[0];
      r->size[0] = c_encoderConfig_derivedParams_R->size[0];
      emxEnsureCapacity_real_T(r, i1);
      output_data = r->data;
      n = c_encoderConfig_derivedParams_R->size[0];
      for (i1 = 0; i1 < n; i1++) {
        yk = d_encoderConfig_derivedParams_R[i1];
        if (yk > 2147483646) {
          yk = MAX_int32_T;
        } else {
          yk++;
        }
        output_data[i1] = MatrixProductBuffer_data[yk - 1];
      }
      n = r->size[0];
      for (i1 = 0; i1 < n; i1++) {
        ParityCheckBits_data[i1 + ParityCheckBits->size[0] * b_i] =
            output_data[i1];
      }
      n = ParityCheckBits->size[0];
      i1 = MatrixProductBuffer->size[0];
      MatrixProductBuffer->size[0] = ParityCheckBits->size[0];
      emxEnsureCapacity_real_T(MatrixProductBuffer, i1);
      MatrixProductBuffer_data = MatrixProductBuffer->data;
      for (i1 = 0; i1 < n; i1++) {
        MatrixProductBuffer_data[i1] =
            ParityCheckBits_data[i1 + ParityCheckBits->size[0] * b_i];
      }
      GF2Subst(MatrixProductBuffer, i_encoderConfig_derivedParams_M,
               j_encoderConfig_derivedParams_M, k_encoderConfig_derivedParams_M,
               c_encoderConfig_derivedParams_E);
      MatrixProductBuffer_data = MatrixProductBuffer->data;
      n = MatrixProductBuffer->size[0];
      for (i1 = 0; i1 < n; i1++) {
        ParityCheckBits_data[i1 + ParityCheckBits->size[0] * b_i] =
            MatrixProductBuffer_data[i1];
      }
    } else {
      n = informationBits->size[0];
      i1 = b_informationBits->size[0];
      b_informationBits->size[0] = informationBits->size[0];
      emxEnsureCapacity_real_T(b_informationBits, i1);
      output_data = b_informationBits->data;
      for (i1 = 0; i1 < n; i1++) {
        output_data[i1] =
            informationBits_data[i1 + informationBits->size[0] * b_i];
      }
      GF2MatrixMul(b_informationBits, c_encoderConfig_NumParityCheckB,
                   f_encoderConfig_derivedParams_M,
                   g_encoderConfig_derivedParams_M,
                   h_encoderConfig_derivedParams_M, MatrixProductBuffer);
      MatrixProductBuffer_data = MatrixProductBuffer->data;
      n = MatrixProductBuffer->size[0];
      for (i1 = 0; i1 < n; i1++) {
        ParityCheckBits_data[i1 + ParityCheckBits->size[0] * b_i] =
            MatrixProductBuffer_data[i1];
      }
      n = ParityCheckBits->size[0];
      i1 = MatrixProductBuffer->size[0];
      MatrixProductBuffer->size[0] = ParityCheckBits->size[0];
      emxEnsureCapacity_real_T(MatrixProductBuffer, i1);
      MatrixProductBuffer_data = MatrixProductBuffer->data;
      for (i1 = 0; i1 < n; i1++) {
        MatrixProductBuffer_data[i1] =
            ParityCheckBits_data[i1 + ParityCheckBits->size[0] * b_i];
      }
      GF2Subst(MatrixProductBuffer, i_encoderConfig_derivedParams_M,
               j_encoderConfig_derivedParams_M, k_encoderConfig_derivedParams_M,
               c_encoderConfig_derivedParams_E);
      MatrixProductBuffer_data = MatrixProductBuffer->data;
      n = MatrixProductBuffer->size[0];
      for (i1 = 0; i1 < n; i1++) {
        ParityCheckBits_data[i1 + ParityCheckBits->size[0] * b_i] =
            MatrixProductBuffer_data[i1];
      }
    }
  }
  emxFree_real_T(&b_informationBits);
  emxFree_int32_T(&y);
  emxFree_int32_T(&rowindex);
  emxFree_real_T(&r);
  emxFree_real_T(&MatrixProductBuffer);
  b = ((informationBits->size[0] != 0) && (informationBits->size[1] != 0));
  if (b) {
    yk = informationBits->size[1];
  } else if ((ParityCheckBits->size[0] != 0) &&
             (ParityCheckBits->size[1] != 0)) {
    yk = ParityCheckBits->size[1];
  } else {
    yk = informationBits->size[1];
    if (ParityCheckBits->size[1] > informationBits->size[1]) {
      yk = ParityCheckBits->size[1];
    }
  }
  empty_non_axis_sizes = (yk == 0);
  if (empty_non_axis_sizes || b) {
    n = informationBits->size[0];
  } else {
    n = 0;
  }
  if (empty_non_axis_sizes ||
      ((ParityCheckBits->size[0] != 0) && (ParityCheckBits->size[1] != 0))) {
    sizes_idx_0 = (short)ParityCheckBits->size[0];
  } else {
    sizes_idx_0 = 0;
  }
  k = sizes_idx_0;
  i = output->size[0] * output->size[1];
  output->size[0] = n + sizes_idx_0;
  output->size[1] = yk;
  emxEnsureCapacity_real_T(output, i);
  output_data = output->data;
  for (i = 0; i < yk; i++) {
    for (i1 = 0; i1 < n; i1++) {
      output_data[i1 + output->size[0] * i] = informationBits_data[i1 + n * i];
    }
    for (i1 = 0; i1 < k; i1++) {
      output_data[(i1 + n) + output->size[0] * i] =
          ParityCheckBits_data[i1 + sizes_idx_0 * i];
    }
  }
  emxFree_real_T(&ParityCheckBits);
}

/*
 * File trailer for ldpcEncode.c
 *
 * [EOF]
 */
