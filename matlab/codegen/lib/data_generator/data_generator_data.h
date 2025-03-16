/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: data_generator_data.h
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

#ifndef DATA_GENERATOR_DATA_H
#define DATA_GENERATOR_DATA_H

/* Include Files */
#include "rtwtypes.h"
#include "omp.h"
#include <stddef.h>
#include <stdio.h>
#include <stdlib.h>

/* Variable Declarations */
extern unsigned int state[625];
extern FILE *eml_openfiles[20];
extern boolean_T eml_autoflush[20];
extern omp_nest_lock_t data_generator_nestLockGlobal;
extern boolean_T isInitialized_data_generator;

#endif
/*
 * File trailer for data_generator_data.h
 *
 * [EOF]
 */
