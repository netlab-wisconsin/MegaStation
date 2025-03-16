/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: data_generator.h
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

#ifndef DATA_GENERATOR_H
#define DATA_GENERATOR_H

/* Include Files */
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
extern void
data_generator(unsigned long long ue, unsigned long long bs,
               unsigned long long ofdm_ca, unsigned long long ofdm_da,
               unsigned long long sc_group, unsigned long long num_pilots,
               unsigned long long num_uplinks, unsigned long long num_downlinks,
               double modulation_order, double code_rate, double base_graph,
               unsigned long long pilot_spacing);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for data_generator.h
 *
 * [EOF]
 */
