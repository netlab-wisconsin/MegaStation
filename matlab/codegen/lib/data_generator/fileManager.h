/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: fileManager.h
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

#ifndef FILEMANAGER_H
#define FILEMANAGER_H

/* Include Files */
#include "rtwtypes.h"
#include <stddef.h>
#include <stdlib.h>

#ifdef __cplusplus
extern "C" {
#endif

/* Function Declarations */
int cfclose(double fid);

signed char cfopen(const char *cfilename);

void filedata_init(void);

#ifdef __cplusplus
}
#endif

#endif
/*
 * File trailer for fileManager.h
 *
 * [EOF]
 */
