/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: data_generator_terminate.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "data_generator_terminate.h"
#include "data_generator_data.h"
#include "encode.h"
#include "rt_nonfinite.h"
#include "omp.h"

/* Function Definitions */
/*
 * Arguments    : void
 * Return Type  : void
 */
void data_generator_terminate(void)
{
  encode_free();
  omp_destroy_nest_lock(&data_generator_nestLockGlobal);
  isInitialized_data_generator = false;
}

/*
 * File trailer for data_generator_terminate.c
 *
 * [EOF]
 */
