/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: rand.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "rand.h"
#include "data_generator_data.h"
#include "data_generator_emxutil.h"
#include "data_generator_types.h"
#include "eml_rand_mt19937ar.h"
#include "rt_nonfinite.h"

/* Function Definitions */
/*
 * Arguments    : unsigned long long varargin_1
 *                unsigned long long varargin_2
 *                unsigned long long varargin_3
 *                emxArray_real_T *r
 * Return Type  : void
 */
void b_rand(unsigned long long varargin_1, unsigned long long varargin_2,
            unsigned long long varargin_3, emxArray_real_T *r)
{
  double *r_data;
  int i;
  int k;
  i = r->size[0] * r->size[1] * r->size[2];
  r->size[0] = (int)varargin_1;
  r->size[1] = (int)varargin_2;
  r->size[2] = (int)varargin_3;
  emxEnsureCapacity_real_T(r, i);
  r_data = r->data;
  i = (int)varargin_1 * (int)varargin_2 * (int)varargin_3;
  for (k = 0; k < i; k++) {
    double b_r;
    /* ========================= COPYRIGHT NOTICE ============================
     */
    /*  This is a uniform (0,1) pseudorandom number generator based on: */
    /*                                                                         */
    /*  A C-program for MT19937, with initialization improved 2002/1/26. */
    /*  Coded by Takuji Nishimura and Makoto Matsumoto. */
    /*                                                                         */
    /*  Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura, */
    /*  All rights reserved. */
    /*                                                                         */
    /*  Redistribution and use in source and binary forms, with or without */
    /*  modification, are permitted provided that the following conditions */
    /*  are met: */
    /*                                                                         */
    /*    1. Redistributions of source code must retain the above copyright */
    /*       notice, this list of conditions and the following disclaimer. */
    /*                                                                         */
    /*    2. Redistributions in binary form must reproduce the above copyright
     */
    /*       notice, this list of conditions and the following disclaimer */
    /*       in the documentation and/or other materials provided with the */
    /*       distribution. */
    /*                                                                         */
    /*    3. The names of its contributors may not be used to endorse or */
    /*       promote products derived from this software without specific */
    /*       prior written permission. */
    /*                                                                         */
    /*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS */
    /*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT */
    /*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR */
    /*  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT */
    /*  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, */
    /*  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT */
    /*  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, */
    /*  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY */
    /*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT */
    /*  (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
     */
    /*  OF THIS  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */
    /*                                                                         */
    /* =============================   END   =================================
     */
    unsigned int u[2];
    do {
      genrand_uint32_vector(state, u);
      u[0] >>= 5U;
      u[1] >>= 6U;
      b_r =
          1.1102230246251565E-16 * ((double)u[0] * 6.7108864E+7 + (double)u[1]);
    } while (b_r == 0.0);
    r_data[k] = b_r;
  }
}

/*
 * Arguments    : unsigned long long varargin_1
 *                unsigned long long varargin_2
 *                emxArray_creal32_T *r
 * Return Type  : void
 */
void complexLike(unsigned long long varargin_1, unsigned long long varargin_2,
                 emxArray_creal32_T *r)
{
  creal32_T *r_data;
  int i;
  int k;
  i = r->size[0] * r->size[1];
  r->size[0] = (int)varargin_1;
  r->size[1] = (int)varargin_2;
  emxEnsureCapacity_creal32_T(r, i);
  r_data = r->data;
  i = (int)varargin_1 * (int)varargin_2;
  for (k = 0; k < i; k++) {
    float b_r;
    float c_r;
    unsigned int u[2];
    /* ========================= COPYRIGHT NOTICE ============================
     */
    /*  This is a uniform (0,1) pseudorandom number generator based on: */
    /*                                                                         */
    /*  A C-program for MT19937, with initialization improved 2002/1/26. */
    /*  Coded by Takuji Nishimura and Makoto Matsumoto. */
    /*                                                                         */
    /*  Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura, */
    /*  All rights reserved. */
    /*                                                                         */
    /*  Redistribution and use in source and binary forms, with or without */
    /*  modification, are permitted provided that the following conditions */
    /*  are met: */
    /*                                                                         */
    /*    1. Redistributions of source code must retain the above copyright */
    /*       notice, this list of conditions and the following disclaimer. */
    /*                                                                         */
    /*    2. Redistributions in binary form must reproduce the above copyright
     */
    /*       notice, this list of conditions and the following disclaimer */
    /*       in the documentation and/or other materials provided with the */
    /*       distribution. */
    /*                                                                         */
    /*    3. The names of its contributors may not be used to endorse or */
    /*       promote products derived from this software without specific */
    /*       prior written permission. */
    /*                                                                         */
    /*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS */
    /*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT */
    /*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR */
    /*  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT */
    /*  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, */
    /*  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT */
    /*  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, */
    /*  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY */
    /*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT */
    /*  (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
     */
    /*  OF THIS  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */
    /*                                                                         */
    /* =============================   END   =================================
     */
    do {
      genrand_uint32_vector(state, u);
      u[0] >>= 5U;
      u[1] >>= 6U;
      b_r = (float)u[0] * 7.4505806E-9F;
    } while (((int)u[0] == 0) && ((int)u[1] == 0));
    if (b_r < 5.96046448E-8F) {
      b_r = 5.96046448E-8F;
    } else if (b_r > 0.99999994F) {
      b_r = 0.99999994F;
    }
    /* ========================= COPYRIGHT NOTICE ============================
     */
    /*  This is a uniform (0,1) pseudorandom number generator based on: */
    /*                                                                         */
    /*  A C-program for MT19937, with initialization improved 2002/1/26. */
    /*  Coded by Takuji Nishimura and Makoto Matsumoto. */
    /*                                                                         */
    /*  Copyright (C) 1997 - 2002, Makoto Matsumoto and Takuji Nishimura, */
    /*  All rights reserved. */
    /*                                                                         */
    /*  Redistribution and use in source and binary forms, with or without */
    /*  modification, are permitted provided that the following conditions */
    /*  are met: */
    /*                                                                         */
    /*    1. Redistributions of source code must retain the above copyright */
    /*       notice, this list of conditions and the following disclaimer. */
    /*                                                                         */
    /*    2. Redistributions in binary form must reproduce the above copyright
     */
    /*       notice, this list of conditions and the following disclaimer */
    /*       in the documentation and/or other materials provided with the */
    /*       distribution. */
    /*                                                                         */
    /*    3. The names of its contributors may not be used to endorse or */
    /*       promote products derived from this software without specific */
    /*       prior written permission. */
    /*                                                                         */
    /*  THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS */
    /*  "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT */
    /*  LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR */
    /*  A PARTICULAR PURPOSE ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT */
    /*  OWNER OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, */
    /*  SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT */
    /*  LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, */
    /*  DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY */
    /*  THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT */
    /*  (INCLUDING  NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
     */
    /*  OF THIS  SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE. */
    /*                                                                         */
    /* =============================   END   =================================
     */
    do {
      genrand_uint32_vector(state, u);
      u[0] >>= 5U;
      u[1] >>= 6U;
      c_r = (float)u[0] * 7.4505806E-9F;
    } while (((int)u[0] == 0) && ((int)u[1] == 0));
    if (c_r < 5.96046448E-8F) {
      c_r = 5.96046448E-8F;
    } else if (c_r > 0.99999994F) {
      c_r = 0.99999994F;
    }
    r_data[k].re = b_r;
    r_data[k].im = c_r;
  }
}

/*
 * File trailer for rand.c
 *
 * [EOF]
 */
