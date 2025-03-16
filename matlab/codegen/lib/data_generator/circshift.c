/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: circshift.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "circshift.h"
#include "data_generator_emxutil.h"
#include "data_generator_types.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : emxArray_creal32_T *a
 *                unsigned long long p
 * Return Type  : void
 */
void circshift(emxArray_creal32_T *a, unsigned long long p)
{
  emxArray_creal32_T *buffer;
  creal32_T *a_data;
  creal32_T *buffer_data;
  int b_i;
  int dim;
  int i;
  int j;
  int k;
  a_data = a->data;
  dim = 1;
  if (a->size[0] != 1) {
    dim = 0;
  } else if ((a->size[1] == 1) && (a->size[2] != 1)) {
    dim = 2;
  }
  emxInit_creal32_T(&buffer, 2);
  if ((a->size[0] != 0) && (a->size[1] != 0) && (a->size[2] != 0) &&
      ((a->size[0] != 1) || (a->size[1] != 1) || (a->size[2] != 1))) {
    int lowerDim;
    int npages;
    int ns;
    int nv;
    int pagesize;
    int stride;
    boolean_T shiftright;
    ns = (int)p;
    shiftright = true;
    if (a->size[dim] <= 1) {
      ns = 0;
    } else {
      if ((int)p > a->size[dim]) {
        unsigned int u;
        u = (unsigned int)a->size[dim];
        if (u == 0U) {
          i = MAX_int32_T;
        } else {
          i = (int)((unsigned int)p / u);
        }
        ns = (int)p - a->size[dim] * i;
      }
      if (ns > (a->size[dim] >> 1)) {
        ns = a->size[dim] - ns;
        shiftright = false;
      }
    }
    lowerDim = a->size[0];
    if ((a->size[0] > 0) && ((a->size[1] == 0) || (a->size[1] > a->size[0]))) {
      lowerDim = a->size[1];
    }
    if ((lowerDim > 0) && ((a->size[2] == 0) || (a->size[2] > lowerDim))) {
      lowerDim = a->size[2];
    }
    lowerDim /= 2;
    i = buffer->size[0] * buffer->size[1];
    buffer->size[0] = 1;
    buffer->size[1] = lowerDim;
    emxEnsureCapacity_creal32_T(buffer, i);
    buffer_data = buffer->data;
    for (i = 0; i < lowerDim; i++) {
      buffer_data[i].re = 0.0F;
      buffer_data[i].im = 0.0F;
    }
    i = a->size[dim];
    nv = a->size[dim];
    stride = 1;
    for (k = 0; k < dim; k++) {
      stride *= a->size[k];
    }
    npages = 1;
    lowerDim = dim + 2;
    for (k = lowerDim; k < 4; k++) {
      npages *= a->size[k - 1];
    }
    pagesize = stride * a->size[dim];
    if ((a->size[dim] > 1) && (ns > 0)) {
      for (b_i = 0; b_i < npages; b_i++) {
        lowerDim = b_i * pagesize;
        for (j = 0; j < stride; j++) {
          dim = lowerDim + j;
          if (shiftright) {
            int i1;
            for (k = 0; k < ns; k++) {
              buffer_data[k] = a_data[dim + ((k + i) - ns) * stride];
            }
            i1 = ns + 1;
            for (k = nv; k >= i1; k--) {
              a_data[dim + (k - 1) * stride] =
                  a_data[dim + ((k - ns) - 1) * stride];
            }
            for (k = 0; k < ns; k++) {
              a_data[dim + k * stride] = buffer_data[k];
            }
          } else {
            int i1;
            for (k = 0; k < ns; k++) {
              buffer_data[k] = a_data[dim + k * stride];
            }
            i1 = (i - ns) - 1;
            for (k = 0; k <= i1; k++) {
              a_data[dim + k * stride] = a_data[dim + (k + ns) * stride];
            }
            for (k = 0; k < ns; k++) {
              a_data[dim + ((k + i) - ns) * stride] = buffer_data[k];
            }
          }
        }
      }
    }
  }
  emxFree_creal32_T(&buffer);
}

/*
 * File trailer for circshift.c
 *
 * [EOF]
 */
