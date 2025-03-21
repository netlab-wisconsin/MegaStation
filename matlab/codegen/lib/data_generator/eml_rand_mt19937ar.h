/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: eml_rand_mt19937ar.h
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

#ifndef EML_RAND_MT19937AR_H
#define EML_RAND_MT19937AR_H

/* Include Files */
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
double eml_rand_mt19937ar(unsigned int b_state[625]);

void genrand_uint32_vector(unsigned int mt[625], unsigned int u[2]);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for eml_rand_mt19937ar.h
 *
 * [EOF]
 */
