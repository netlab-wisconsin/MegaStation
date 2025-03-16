/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: data_generator_initialize.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "data_generator_initialize.h"
#include "data_generator_data.h"
#include "eml_rand_mt19937ar_stateful.h"
#include "encode.h"
#include "fileManager.h"
#include "rt_nonfinite.h"
#include "omp.h"

/* Function Definitions */
/*
 * Arguments    : void
 * Return Type  : void
 */
void data_generator_initialize(void)
{
  omp_init_nest_lock(&data_generator_nestLockGlobal);
  c_eml_rand_mt19937ar_stateful_i();
  filedata_init();
  encode_init();
  isInitialized_data_generator = true;
}

/*
 * File trailer for data_generator_initialize.c
 *
 * [EOF]
 */
