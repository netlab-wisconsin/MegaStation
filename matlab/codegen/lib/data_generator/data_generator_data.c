/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: data_generator_data.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "data_generator_data.h"
#include "rt_nonfinite.h"

/* Variable Definitions */
unsigned int state[625];

FILE *eml_openfiles[20];

boolean_T eml_autoflush[20];

omp_nest_lock_t data_generator_nestLockGlobal;

boolean_T isInitialized_data_generator = false;

/*
 * File trailer for data_generator_data.c
 *
 * [EOF]
 */
