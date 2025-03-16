/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: introsort.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "introsort.h"
#include "data_generator_types.h"
#include "heapsort.h"
#include "insertionsort.h"
#include "rt_nonfinite.h"

/* Type Definitions */
#ifndef typedef_struct_T
#define typedef_struct_T
typedef struct {
  int xstart;
  int xend;
  int depth;
} struct_T;
#endif /* typedef_struct_T */

/* Function Definitions */
/*
 * Arguments    : emxArray_int32_T *x
 *                int xend
 *                const emxArray_int32_T *cmp_workspace_x
 * Return Type  : void
 */
void b_introsort(emxArray_int32_T *x, int xend,
                 const emxArray_int32_T *cmp_workspace_x)
{
  struct_T frame;
  const int *cmp_workspace_x_data;
  int i;
  int pmin;
  int *x_data;
  cmp_workspace_x_data = cmp_workspace_x->data;
  x_data = x->data;
  if (xend > 1) {
    if (xend <= 32) {
      b_insertionsort(x, 1, xend, cmp_workspace_x);
    } else {
      struct_T st_d_data[120];
      int MAXDEPTH;
      int pmax;
      int pow2p;
      int st_n;
      int t;
      boolean_T exitg1;
      pmax = 31;
      pmin = 0;
      exitg1 = false;
      while ((!exitg1) && (pmax - pmin > 1)) {
        t = (pmin + pmax) >> 1;
        pow2p = 1 << t;
        if (pow2p == xend) {
          pmax = t;
          exitg1 = true;
        } else if (pow2p > xend) {
          pmax = t;
        } else {
          pmin = t;
        }
      }
      MAXDEPTH = (pmax - 1) << 1;
      frame.xstart = 1;
      frame.xend = xend;
      frame.depth = 0;
      pmax = MAXDEPTH << 1;
      for (i = 0; i < pmax; i++) {
        st_d_data[i] = frame;
      }
      st_d_data[0] = frame;
      st_n = 1;
      while (st_n > 0) {
        frame = st_d_data[st_n - 1];
        st_n--;
        i = frame.xend - frame.xstart;
        if (i + 1 <= 32) {
          b_insertionsort(x, frame.xstart, frame.xend, cmp_workspace_x);
          x_data = x->data;
        } else if (frame.depth == MAXDEPTH) {
          c_heapsort(x, frame.xstart, frame.xend, cmp_workspace_x);
          x_data = x->data;
        } else {
          pmin = (frame.xstart + i / 2) - 1;
          i = x_data[frame.xstart - 1];
          if (cmp_workspace_x_data[x_data[pmin] - 1] <
              cmp_workspace_x_data[i - 1]) {
            x_data[frame.xstart - 1] = x_data[pmin];
            x_data[pmin] = i;
          }
          i = x_data[frame.xstart - 1];
          pmax = x_data[frame.xend - 1];
          if (cmp_workspace_x_data[pmax - 1] < cmp_workspace_x_data[i - 1]) {
            x_data[frame.xstart - 1] = pmax;
            x_data[frame.xend - 1] = i;
          }
          i = x_data[frame.xend - 1];
          if (cmp_workspace_x_data[i - 1] <
              cmp_workspace_x_data[x_data[pmin] - 1]) {
            t = x_data[pmin];
            x_data[pmin] = i;
            x_data[frame.xend - 1] = t;
          }
          pow2p = x_data[pmin];
          x_data[pmin] = x_data[frame.xend - 2];
          x_data[frame.xend - 2] = pow2p;
          pmax = frame.xstart - 1;
          pmin = frame.xend - 2;
          int exitg2;
          do {
            exitg2 = 0;
            pmax++;
            int exitg3;
            do {
              exitg3 = 0;
              i = cmp_workspace_x_data[pow2p - 1];
              if (cmp_workspace_x_data[x_data[pmax] - 1] < i) {
                pmax++;
              } else {
                exitg3 = 1;
              }
            } while (exitg3 == 0);
            for (pmin--; i < cmp_workspace_x_data[x_data[pmin] - 1]; pmin--) {
            }
            if (pmax + 1 >= pmin + 1) {
              exitg2 = 1;
            } else {
              t = x_data[pmax];
              x_data[pmax] = x_data[pmin];
              x_data[pmin] = t;
            }
          } while (exitg2 == 0);
          x_data[frame.xend - 2] = x_data[pmax];
          x_data[pmax] = pow2p;
          if (pmax + 2 < frame.xend) {
            st_d_data[st_n].xstart = pmax + 2;
            st_d_data[st_n].xend = frame.xend;
            st_d_data[st_n].depth = frame.depth + 1;
            st_n++;
          }
          if (frame.xstart < pmax + 1) {
            st_d_data[st_n].xstart = frame.xstart;
            st_d_data[st_n].xend = pmax + 1;
            st_d_data[st_n].depth = frame.depth + 1;
            st_n++;
          }
        }
      }
    }
  }
}

/*
 * Arguments    : emxArray_int32_T *x
 *                int xend
 *                const emxArray_int32_T *cmp_workspace_a
 *                const emxArray_int32_T *cmp_workspace_b
 * Return Type  : void
 */
void introsort(emxArray_int32_T *x, int xend,
               const emxArray_int32_T *cmp_workspace_a,
               const emxArray_int32_T *cmp_workspace_b)
{
  struct_T frame;
  const int *cmp_workspace_a_data;
  const int *cmp_workspace_b_data;
  int i;
  int *x_data;
  cmp_workspace_b_data = cmp_workspace_b->data;
  cmp_workspace_a_data = cmp_workspace_a->data;
  x_data = x->data;
  if (xend > 1) {
    if (xend <= 32) {
      insertionsort(x, 1, xend, cmp_workspace_a, cmp_workspace_b);
    } else {
      struct_T st_d_data[120];
      int MAXDEPTH;
      int pmax;
      int pmin;
      int pow2p;
      int st_n;
      int t;
      boolean_T exitg1;
      pmax = 31;
      pmin = 0;
      exitg1 = false;
      while ((!exitg1) && (pmax - pmin > 1)) {
        t = (pmin + pmax) >> 1;
        pow2p = 1 << t;
        if (pow2p == xend) {
          pmax = t;
          exitg1 = true;
        } else if (pow2p > xend) {
          pmax = t;
        } else {
          pmin = t;
        }
      }
      MAXDEPTH = (pmax - 1) << 1;
      frame.xstart = 1;
      frame.xend = xend;
      frame.depth = 0;
      pmax = MAXDEPTH << 1;
      for (i = 0; i < pmax; i++) {
        st_d_data[i] = frame;
      }
      st_d_data[0] = frame;
      st_n = 1;
      while (st_n > 0) {
        frame = st_d_data[st_n - 1];
        st_n--;
        i = frame.xend - frame.xstart;
        if (i + 1 <= 32) {
          insertionsort(x, frame.xstart, frame.xend, cmp_workspace_a,
                        cmp_workspace_b);
          x_data = x->data;
        } else if (frame.depth == MAXDEPTH) {
          b_heapsort(x, frame.xstart, frame.xend, cmp_workspace_a,
                     cmp_workspace_b);
          x_data = x->data;
        } else {
          int xmid;
          boolean_T varargout_1;
          xmid = (frame.xstart + i / 2) - 1;
          i = cmp_workspace_a_data[x_data[xmid] - 1];
          pmax = x_data[frame.xstart - 1];
          pmin = cmp_workspace_a_data[pmax - 1];
          if (i < pmin) {
            varargout_1 = true;
          } else if (i == pmin) {
            varargout_1 = (cmp_workspace_b_data[x_data[xmid] - 1] <
                           cmp_workspace_b_data[pmax - 1]);
          } else {
            varargout_1 = false;
          }
          if (varargout_1) {
            x_data[frame.xstart - 1] = x_data[xmid];
            x_data[xmid] = pmax;
          }
          i = x_data[frame.xend - 1];
          pmax = cmp_workspace_a_data[i - 1];
          pmin = x_data[frame.xstart - 1];
          t = cmp_workspace_a_data[pmin - 1];
          if (pmax < t) {
            varargout_1 = true;
          } else if (pmax == t) {
            varargout_1 =
                (cmp_workspace_b_data[i - 1] < cmp_workspace_b_data[pmin - 1]);
          } else {
            varargout_1 = false;
          }
          if (varargout_1) {
            x_data[frame.xstart - 1] = i;
            x_data[frame.xend - 1] = pmin;
          }
          i = x_data[frame.xend - 1];
          pmax = cmp_workspace_a_data[i - 1];
          pmin = cmp_workspace_a_data[x_data[xmid] - 1];
          if (pmax < pmin) {
            varargout_1 = true;
          } else if (pmax == pmin) {
            varargout_1 = (cmp_workspace_b_data[i - 1] <
                           cmp_workspace_b_data[x_data[xmid] - 1]);
          } else {
            varargout_1 = false;
          }
          if (varargout_1) {
            t = x_data[xmid];
            x_data[xmid] = i;
            x_data[frame.xend - 1] = t;
          }
          pow2p = x_data[xmid] - 1;
          x_data[xmid] = x_data[frame.xend - 2];
          x_data[frame.xend - 2] = pow2p + 1;
          pmax = frame.xstart - 1;
          pmin = frame.xend - 2;
          int exitg2;
          do {
            int exitg3;
            exitg2 = 0;
            pmax++;
            do {
              exitg3 = 0;
              i = cmp_workspace_a_data[x_data[pmax] - 1];
              if (i < cmp_workspace_a_data[pow2p]) {
                varargout_1 = true;
              } else if (i == cmp_workspace_a_data[pow2p]) {
                varargout_1 = (cmp_workspace_b_data[x_data[pmax] - 1] <
                               cmp_workspace_b_data[pow2p]);
              } else {
                varargout_1 = false;
              }
              if (varargout_1) {
                pmax++;
              } else {
                exitg3 = 1;
              }
            } while (exitg3 == 0);
            pmin--;
            do {
              exitg3 = 0;
              i = cmp_workspace_a_data[x_data[pmin] - 1];
              if (cmp_workspace_a_data[pow2p] < i) {
                varargout_1 = true;
              } else if (cmp_workspace_a_data[pow2p] == i) {
                varargout_1 = (cmp_workspace_b_data[pow2p] <
                               cmp_workspace_b_data[x_data[pmin] - 1]);
              } else {
                varargout_1 = false;
              }
              if (varargout_1) {
                pmin--;
              } else {
                exitg3 = 1;
              }
            } while (exitg3 == 0);
            if (pmax + 1 >= pmin + 1) {
              exitg2 = 1;
            } else {
              t = x_data[pmax];
              x_data[pmax] = x_data[pmin];
              x_data[pmin] = t;
            }
          } while (exitg2 == 0);
          x_data[frame.xend - 2] = x_data[pmax];
          x_data[pmax] = pow2p + 1;
          if (pmax + 2 < frame.xend) {
            st_d_data[st_n].xstart = pmax + 2;
            st_d_data[st_n].xend = frame.xend;
            st_d_data[st_n].depth = frame.depth + 1;
            st_n++;
          }
          if (frame.xstart < pmax + 1) {
            st_d_data[st_n].xstart = frame.xstart;
            st_d_data[st_n].xend = pmax + 1;
            st_d_data[st_n].depth = frame.depth + 1;
            st_n++;
          }
        }
      }
    }
  }
}

/*
 * File trailer for introsort.c
 *
 * [EOF]
 */
