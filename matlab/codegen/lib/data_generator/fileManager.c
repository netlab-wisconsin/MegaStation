/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: fileManager.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "fileManager.h"
#include "data_generator_data.h"
#include "rt_nonfinite.h"
#include <stdio.h>

/* Function Declarations */
static signed char filedata(void);

/* Function Definitions */
/*
 * Arguments    : void
 * Return Type  : signed char
 */
static signed char filedata(void)
{
  int k;
  signed char f;
  boolean_T exitg1;
  f = 0;
  k = 0;
  exitg1 = false;
  while ((!exitg1) && (k < 20)) {
    if (eml_openfiles[k] == NULL) {
      f = (signed char)(k + 1);
      exitg1 = true;
    } else {
      k++;
    }
  }
  return f;
}

/*
 * Arguments    : double fid
 * Return Type  : int
 */
int cfclose(double fid)
{
  FILE *f;
  int st;
  signed char b_fileid;
  signed char fileid;
  st = -1;
  fileid = (signed char)fid;
  if (((signed char)fid < 0) || (fid != (signed char)fid)) {
    fileid = -1;
  }
  b_fileid = fileid;
  if (fileid < 0) {
    b_fileid = -1;
  }
  if (b_fileid >= 3) {
    f = eml_openfiles[b_fileid - 3];
  } else if (b_fileid == 0) {
    f = stdin;
  } else if (b_fileid == 1) {
    f = stdout;
  } else if (b_fileid == 2) {
    f = stderr;
  } else {
    f = NULL;
  }
  if ((f != NULL) && (fileid >= 3)) {
    int cst;
    cst = fclose(f);
    if (cst == 0) {
      st = 0;
      eml_openfiles[fileid - 3] = NULL;
      eml_autoflush[fileid - 3] = true;
    }
  }
  return st;
}

/*
 * Arguments    : const char *cfilename
 * Return Type  : signed char
 */
signed char cfopen(const char *cfilename)
{
  FILE *filestar;
  signed char fileid;
  signed char j;
  fileid = -1;
  j = filedata();
  if (j >= 1) {
    filestar = fopen(cfilename, "wb");
    if (filestar != NULL) {
      int i;
      eml_openfiles[j - 1] = filestar;
      eml_autoflush[j - 1] = true;
      i = j + 2;
      if (j + 2 > 127) {
        i = 127;
      }
      fileid = (signed char)i;
    }
  }
  return fileid;
}

/*
 * Arguments    : void
 * Return Type  : void
 */
void filedata_init(void)
{
  int i;
  for (i = 0; i < 20; i++) {
    eml_autoflush[i] = false;
  }
  for (i = 0; i < 20; i++) {
    eml_openfiles[i] = NULL;
  }
}

/*
 * File trailer for fileManager.c
 *
 * [EOF]
 */
