/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: insertionsort.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "insertionsort.h"
#include "data_generator_types.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : emxArray_int32_T *x
 *                int xstart
 *                int xend
 *                const emxArray_int32_T *cmp_workspace_x
 * Return Type  : void
 */
void b_insertionsort(emxArray_int32_T *x, int xstart, int xend,
                     const emxArray_int32_T *cmp_workspace_x)
{
  const int *cmp_workspace_x_data;
  int i;
  int k;
  int *x_data;
  cmp_workspace_x_data = cmp_workspace_x->data;
  x_data = x->data;
  i = xstart + 1;
  for (k = i; k <= xend; k++) {
    int idx;
    int xc;
    boolean_T exitg1;
    xc = x_data[k - 1];
    idx = k - 1;
    exitg1 = false;
    while ((!exitg1) && (idx >= xstart)) {
      int i1;
      i1 = x_data[idx - 1];
      if (cmp_workspace_x_data[xc - 1] < cmp_workspace_x_data[i1 - 1]) {
        x_data[idx] = i1;
        idx--;
      } else {
        exitg1 = true;
      }
    }
    x_data[idx] = xc;
  }
}

/*
 * Arguments    : emxArray_int32_T *x
 *                int xstart
 *                int xend
 *                const emxArray_int32_T *cmp_workspace_a
 *                const emxArray_int32_T *cmp_workspace_b
 * Return Type  : void
 */
void insertionsort(emxArray_int32_T *x, int xstart, int xend,
                   const emxArray_int32_T *cmp_workspace_a,
                   const emxArray_int32_T *cmp_workspace_b)
{
  const int *cmp_workspace_a_data;
  const int *cmp_workspace_b_data;
  int i;
  int k;
  int *x_data;
  cmp_workspace_b_data = cmp_workspace_b->data;
  cmp_workspace_a_data = cmp_workspace_a->data;
  x_data = x->data;
  i = xstart + 1;
  for (k = i; k <= xend; k++) {
    int idx;
    int xc;
    boolean_T exitg1;
    xc = x_data[k - 1] - 1;
    idx = k - 2;
    exitg1 = false;
    while ((!exitg1) && (idx + 1 >= xstart)) {
      int i1;
      boolean_T varargout_1;
      i1 = cmp_workspace_a_data[x_data[idx] - 1];
      if (cmp_workspace_a_data[xc] < i1) {
        varargout_1 = true;
      } else if (cmp_workspace_a_data[xc] == i1) {
        varargout_1 =
            (cmp_workspace_b_data[xc] < cmp_workspace_b_data[x_data[idx] - 1]);
      } else {
        varargout_1 = false;
      }
      if (varargout_1) {
        x_data[idx + 1] = x_data[idx];
        idx--;
      } else {
        exitg1 = true;
      }
    }
    x_data[idx + 1] = xc + 1;
  }
}

/*
 * File trailer for insertionsort.c
 *
 * [EOF]
 */
