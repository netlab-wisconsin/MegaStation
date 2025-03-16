/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: main.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/*************************************************************************/
/* This automatically generated example C main file shows how to call    */
/* entry-point functions that MATLAB Coder generated. You must customize */
/* this file for your application. Do not modify this file directly.     */
/* Instead, make a copy of this file, modify it, and integrate it into   */
/* your development environment.                                         */
/*                                                                       */
/* This file initializes entry-point function arguments to a default     */
/* size and value before calling the entry-point functions. It does      */
/* not store or use any values returned from the entry-point functions.  */
/* If necessary, it does pre-allocate memory for returned values.        */
/* You can use this file as a starting point for a main function that    */
/* you can deploy in your application.                                   */
/*                                                                       */
/* After you copy the file, and before you deploy it, you must make the  */
/* following changes:                                                    */
/* * For variable-size function arguments, change the example sizes to   */
/* the sizes that your application requires.                             */
/* * Change the example values of function arguments to the values that  */
/* your application requires.                                            */
/* * If the entry-point functions return values, store these values or   */
/* otherwise use them as required by your application.                   */
/*                                                                       */
/*************************************************************************/

/* Include Files */
#include <stdlib.h>
#include <string.h>

#include "data_generator.h"
#include "data_generator_terminate.h"
#include "rt_nonfinite.h"

/* Function Declarations */
static double argInit_real_T(char *arg);

static unsigned long long argInit_uint64_T(char *arg);

/* Function Definitions */
/*
 * Arguments    : char *
 * Return Type  : double
 */
static double argInit_real_T(char *arg) {
  char *end;
  return strtod(arg, &end);
}

/*
 * Arguments    : char *
 * Return Type  : unsigned long long
 */
static unsigned long long argInit_uint64_T(char *arg) {
  char *end;
  return strtoull(arg, &end, 10);
}

/*
 * Arguments    : int argc
 *                char **argv
 * Return Type  : int
 */
int main(int argc, char **argv) {
  unsigned long long ue = 16;
  unsigned long long bs = 64;
  unsigned long long ofdm_ca = 2048;
  unsigned long long ofdm_da = 1200;
  unsigned long long sc_group = 16;
  unsigned long long num_pilots = 1;
  unsigned long long num_uplinks = 13;
  unsigned long long num_downlinks = 13;
  double modulation_order = 6.0;
  double code_rate = 1.0 / 3.0;
  double base_graph = 1.0;
  unsigned long long pilot_spacing = 16;

  for (int i = 1; i < argc; i++) {
    if (strcmp(argv[i], "-ue") == 0) {
      ue = argInit_uint64_T(argv[++i]);
    } else if (strcmp(argv[i], "-bs") == 0) {
      bs = argInit_uint64_T(argv[++i]);
    } else if (strcmp(argv[i], "-ofdm_ca") == 0) {
      ofdm_ca = argInit_uint64_T(argv[++i]);
    } else if (strcmp(argv[i], "-ofdm_da") == 0) {
      ofdm_da = argInit_uint64_T(argv[++i]);
    } else if (strcmp(argv[i], "-sc_group") == 0) {
      sc_group = argInit_uint64_T(argv[++i]);
    } else if (strcmp(argv[i], "-num_pilots") == 0) {
      num_pilots = argInit_uint64_T(argv[++i]);
    } else if (strcmp(argv[i], "-num_uplinks") == 0) {
      num_uplinks = argInit_uint64_T(argv[++i]);
    } else if (strcmp(argv[i], "-num_downlinks") == 0) {
      num_downlinks = argInit_uint64_T(argv[++i]);
    } else if (strcmp(argv[i], "-modulation_order") == 0) {
      modulation_order = argInit_real_T(argv[++i]);
    } else if (strcmp(argv[i], "-code_rate") == 0) {
      code_rate = argInit_real_T(argv[++i]);
    } else if (strcmp(argv[i], "-base_graph") == 0) {
      base_graph = argInit_real_T(argv[++i]);
    } else if (strcmp(argv[i], "-pilot_spacing") == 0) {
      pilot_spacing = argInit_uint64_T(argv[++i]);
    }
  }

  data_generator(ue, bs, ofdm_ca, ofdm_da, sc_group, num_pilots, num_uplinks,
                 num_downlinks, modulation_order, code_rate, base_graph,
                 pilot_spacing);

  /* Terminate the application.
You do not need to do this more than one time. */
  data_generator_terminate();
  return 0;
}

/*
 * File trailer for main.c
 *
 * [EOF]
 */
