/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: heapsort.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "heapsort.h"
#include "data_generator_types.h"
#include "rt_nonfinite.h"

/* Function Declarations */
static void b_heapify(emxArray_int32_T *x, int idx, int xstart, int xend,
                      const emxArray_int32_T *cmp_workspace_x);

static void heapify(emxArray_int32_T *x, int idx, int xstart, int xend,
                    const emxArray_int32_T *cmp_workspace_a,
                    const emxArray_int32_T *cmp_workspace_b);

/* Function Definitions */
/*
 * Arguments    : emxArray_int32_T *x
 *                int idx
 *                int xstart
 *                int xend
 *                const emxArray_int32_T *cmp_workspace_x
 * Return Type  : void
 */
static void b_heapify(emxArray_int32_T *x, int idx, int xstart, int xend,
                      const emxArray_int32_T *cmp_workspace_x)
{
  const int *cmp_workspace_x_data;
  int cmpIdx;
  int extremum;
  int extremumIdx;
  int leftIdx;
  int *x_data;
  boolean_T changed;
  boolean_T exitg1;
  cmp_workspace_x_data = cmp_workspace_x->data;
  x_data = x->data;
  changed = true;
  extremumIdx = (idx + xstart) - 2;
  leftIdx = ((idx << 1) + xstart) - 1;
  exitg1 = false;
  while ((!exitg1) && (leftIdx < xend)) {
    int xcmp;
    changed = false;
    extremum = x_data[extremumIdx];
    cmpIdx = leftIdx - 1;
    xcmp = x_data[leftIdx - 1];
    if (cmp_workspace_x_data[xcmp - 1] <
        cmp_workspace_x_data[x_data[leftIdx] - 1]) {
      cmpIdx = leftIdx;
      xcmp = x_data[leftIdx];
    }
    if (cmp_workspace_x_data[x_data[extremumIdx] - 1] <
        cmp_workspace_x_data[xcmp - 1]) {
      x_data[extremumIdx] = xcmp;
      x_data[cmpIdx] = extremum;
      extremumIdx = cmpIdx;
      leftIdx = ((((cmpIdx - xstart) + 2) << 1) + xstart) - 1;
      changed = true;
    } else {
      exitg1 = true;
    }
  }
  if (changed && (leftIdx <= xend)) {
    extremum = x_data[extremumIdx];
    cmpIdx = x_data[leftIdx - 1];
    if (cmp_workspace_x_data[x_data[extremumIdx] - 1] <
        cmp_workspace_x_data[cmpIdx - 1]) {
      x_data[extremumIdx] = cmpIdx;
      x_data[leftIdx - 1] = extremum;
    }
  }
}

/*
 * Arguments    : emxArray_int32_T *x
 *                int idx
 *                int xstart
 *                int xend
 *                const emxArray_int32_T *cmp_workspace_a
 *                const emxArray_int32_T *cmp_workspace_b
 * Return Type  : void
 */
static void heapify(emxArray_int32_T *x, int idx, int xstart, int xend,
                    const emxArray_int32_T *cmp_workspace_a,
                    const emxArray_int32_T *cmp_workspace_b)
{
  const int *cmp_workspace_a_data;
  const int *cmp_workspace_b_data;
  int extremum;
  int extremumIdx;
  int i;
  int i1;
  int leftIdx;
  int *x_data;
  boolean_T changed;
  boolean_T exitg1;
  boolean_T varargout_1;
  cmp_workspace_b_data = cmp_workspace_b->data;
  cmp_workspace_a_data = cmp_workspace_a->data;
  x_data = x->data;
  changed = true;
  extremumIdx = (idx + xstart) - 2;
  leftIdx = ((idx << 1) + xstart) - 2;
  exitg1 = false;
  while ((!exitg1) && (leftIdx + 1 < xend)) {
    int cmpIdx;
    int i2;
    int xcmp;
    changed = false;
    extremum = x_data[extremumIdx];
    cmpIdx = leftIdx;
    xcmp = x_data[leftIdx];
    i = cmp_workspace_a_data[x_data[leftIdx] - 1];
    i1 = x_data[leftIdx + 1] - 1;
    i2 = cmp_workspace_a_data[i1];
    if (i < i2) {
      varargout_1 = true;
    } else if (i == i2) {
      varargout_1 = (cmp_workspace_b_data[x_data[leftIdx] - 1] <
                     cmp_workspace_b_data[i1]);
    } else {
      varargout_1 = false;
    }
    if (varargout_1) {
      cmpIdx = leftIdx + 1;
      xcmp = x_data[leftIdx + 1];
    }
    i = cmp_workspace_a_data[x_data[extremumIdx] - 1];
    i1 = cmp_workspace_a_data[xcmp - 1];
    if (i < i1) {
      varargout_1 = true;
    } else if (i == i1) {
      varargout_1 = (cmp_workspace_b_data[x_data[extremumIdx] - 1] <
                     cmp_workspace_b_data[xcmp - 1]);
    } else {
      varargout_1 = false;
    }
    if (varargout_1) {
      x_data[extremumIdx] = xcmp;
      x_data[cmpIdx] = extremum;
      extremumIdx = cmpIdx;
      leftIdx = ((((cmpIdx - xstart) + 2) << 1) + xstart) - 2;
      changed = true;
    } else {
      exitg1 = true;
    }
  }
  if (changed && (leftIdx + 1 <= xend)) {
    extremum = x_data[extremumIdx];
    i = cmp_workspace_a_data[x_data[extremumIdx] - 1];
    i1 = cmp_workspace_a_data[x_data[leftIdx] - 1];
    if (i < i1) {
      varargout_1 = true;
    } else if (i == i1) {
      varargout_1 = (cmp_workspace_b_data[x_data[extremumIdx] - 1] <
                     cmp_workspace_b_data[x_data[leftIdx] - 1]);
    } else {
      varargout_1 = false;
    }
    if (varargout_1) {
      x_data[extremumIdx] = x_data[leftIdx];
      x_data[leftIdx] = extremum;
    }
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
void b_heapsort(emxArray_int32_T *x, int xstart, int xend,
                const emxArray_int32_T *cmp_workspace_a,
                const emxArray_int32_T *cmp_workspace_b)
{
  int idx;
  int k;
  int n;
  int *x_data;
  x_data = x->data;
  n = (xend - xstart) - 1;
  for (idx = n + 2; idx >= 1; idx--) {
    heapify(x, idx, xstart, xend, cmp_workspace_a, cmp_workspace_b);
    x_data = x->data;
  }
  for (k = 0; k <= n; k++) {
    int t;
    idx = (xend - k) - 1;
    t = x_data[idx];
    x_data[idx] = x_data[xstart - 1];
    x_data[xstart - 1] = t;
    heapify(x, 1, xstart, idx, cmp_workspace_a, cmp_workspace_b);
    x_data = x->data;
  }
}

/*
 * Arguments    : emxArray_int32_T *x
 *                int xstart
 *                int xend
 *                const emxArray_int32_T *cmp_workspace_x
 * Return Type  : void
 */
void c_heapsort(emxArray_int32_T *x, int xstart, int xend,
                const emxArray_int32_T *cmp_workspace_x)
{
  int idx;
  int k;
  int n;
  int *x_data;
  x_data = x->data;
  n = (xend - xstart) - 1;
  for (idx = n + 2; idx >= 1; idx--) {
    b_heapify(x, idx, xstart, xend, cmp_workspace_x);
    x_data = x->data;
  }
  for (k = 0; k <= n; k++) {
    int t;
    idx = (xend - k) - 1;
    t = x_data[idx];
    x_data[idx] = x_data[xstart - 1];
    x_data[xstart - 1] = t;
    b_heapify(x, 1, xstart, idx, cmp_workspace_x);
    x_data = x->data;
  }
}

/*
 * File trailer for heapsort.c
 *
 * [EOF]
 */
