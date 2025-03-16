/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: locBsearch.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "locBsearch.h"
#include "data_generator_types.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : const emxArray_int32_T *x
 *                int xi
 *                int xstart
 *                int xend
 *                boolean_T *found
 * Return Type  : int
 */
int sparse_locBsearch(const emxArray_int32_T *x, int xi, int xstart, int xend,
                      boolean_T *found)
{
  const int *x_data;
  int n;
  x_data = x->data;
  if (xstart < xend) {
    if (xi < x_data[xstart - 1]) {
      n = xstart - 1;
      *found = false;
    } else {
      int high_i;
      int low_ip1;
      high_i = xend;
      n = xstart;
      low_ip1 = xstart;
      while (high_i > low_ip1 + 1) {
        int mid_i;
        mid_i = (n >> 1) + (high_i >> 1);
        if (((n & 1) == 1) && ((high_i & 1) == 1)) {
          mid_i++;
        }
        if (xi >= x_data[mid_i - 1]) {
          n = mid_i;
          low_ip1 = mid_i;
        } else {
          high_i = mid_i;
        }
      }
      *found = (x_data[n - 1] == xi);
    }
  } else if (xstart == xend) {
    n = xstart - 1;
    *found = false;
  } else {
    n = 0;
    *found = false;
  }
  return n;
}

/*
 * File trailer for locBsearch.c
 *
 * [EOF]
 */
