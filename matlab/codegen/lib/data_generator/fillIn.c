/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: fillIn.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "fillIn.h"
#include "data_generator_types.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : c_sparse *this
 * Return Type  : void
 */
void sparse_fillIn(c_sparse *this)
{
  int c;
  int i;
  int idx;
  idx = 1;
  i = this->colidx->size[0];
  for (c = 0; c <= i - 2; c++) {
    int ridx;
    ridx = this->colidx->data[c];
    this->colidx->data[c] = idx;
    int exitg1;
    int i1;
    do {
      exitg1 = 0;
      i1 = this->colidx->data[c + 1];
      if (ridx < i1) {
        int currRowIdx;
        boolean_T val;
        val = false;
        currRowIdx = this->rowidx->data[ridx - 1];
        while ((ridx < i1) && (this->rowidx->data[ridx - 1] == currRowIdx)) {
          if (val || this->d->data[ridx - 1]) {
            val = true;
          }
          ridx++;
        }
        if (val) {
          this->d->data[idx - 1] = true;
          this->rowidx->data[idx - 1] = currRowIdx;
          idx++;
        }
      } else {
        exitg1 = 1;
      }
    } while (exitg1 == 0);
  }
  this->colidx->data[this->colidx->size[0] - 1] = idx;
}

/*
 * File trailer for fillIn.c
 *
 * [EOF]
 */
