/*
 * Academic License - for use in teaching, academic research, and meeting
 * course requirements at degree granting institutions only.  Not for
 * government, commercial, or other organizational use.
 * File: data_generator.c
 *
 * MATLAB Coder version            : 23.2
 * C/C++ source code generated on  : 05-Jun-2024 18:40:37
 */

/* Include Files */
#include "data_generator.h"
#include "bit2int.h"
#include "circshift.h"
#include "data_generator_data.h"
#include "data_generator_emxutil.h"
#include "data_generator_initialize.h"
#include "data_generator_rtwutil.h"
#include "data_generator_types.h"
#include "eml_i64dmul.h"
#include "exp.h"
#include "fileManager.h"
#include "fwrite.h"
#include "i64ddiv.h"
#include "ifft.h"
#include "nrLDPCEncode.h"
#include "qammod.h"
#include "rand.h"
#include "randn.h"
#include "rt_nonfinite.h"
#include "wlanScramble.h"
#include "rt_nonfinite.h"
#include <math.h>

/* Function Declarations */
static void binary_expand_op(emxArray_creal32_T *in1, int in2,
                             unsigned long long in4,
                             const emxArray_creal32_T *in5,
                             const emxArray_int8_T *in6);

static void binary_expand_op_1(emxArray_creal32_T *in1, unsigned long long in2,
                               const emxArray_creal32_T *in3,
                               const emxArray_creal32_T *in4);

static void binary_expand_op_2(emxArray_creal32_T *in1, int in2,
                               const emxArray_creal32_T *in3,
                               const emxArray_creal32_T *in4);

static void binary_expand_op_3(emxArray_creal32_T *in1,
                               const creal_T in2_data[], const int *in2_size,
                               int in4, const emxArray_creal32_T *in5);

static unsigned long long get_ldpc_config(unsigned long long ubits, double bg,
                                          double crate,
                                          unsigned long long *dbits);

static unsigned long long mul_u64_sat(unsigned long long a,
                                      unsigned long long b);

static unsigned long long mul_wide_u64(unsigned long long in0,
                                       unsigned long long in1,
                                       unsigned long long *ptrOutBitsLo);

/* Function Definitions */
/*
 * Arguments    : emxArray_creal32_T *in1
 *                int in2
 *                unsigned long long in4
 *                const emxArray_creal32_T *in5
 *                const emxArray_int8_T *in6
 * Return Type  : void
 */
static void binary_expand_op(emxArray_creal32_T *in1, int in2,
                             unsigned long long in4,
                             const emxArray_creal32_T *in5,
                             const emxArray_int8_T *in6)
{
  const creal32_T *in5_data;
  creal32_T *in1_data;
  int i;
  int loop_ub;
  int stride_0_0;
  int stride_1_0;
  const signed char *in6_data;
  in6_data = in6->data;
  in5_data = in5->data;
  in1_data = in1->data;
  stride_0_0 = (in5->size[0] != 1);
  stride_1_0 = (in6->size[0] != 1);
  if (in6->size[0] == 1) {
    loop_ub = in5->size[0];
  } else {
    loop_ub = in6->size[0];
  }
  for (i = 0; i < loop_ub; i++) {
    int i1;
    int i2;
    i1 = in6_data[i * stride_1_0];
    i2 = (in2 + i) - 1;
    in1_data[i2 + in1->size[0] * ((int)in4 - 1)].re =
        (float)i1 * in5_data[i * stride_0_0].re;
    in1_data[i2 + in1->size[0] * ((int)in4 - 1)].im =
        (float)i1 * in5_data[i * stride_0_0].im;
  }
}

/*
 * Arguments    : emxArray_creal32_T *in1
 *                unsigned long long in2
 *                const emxArray_creal32_T *in3
 *                const emxArray_creal32_T *in4
 * Return Type  : void
 */
static void binary_expand_op_1(emxArray_creal32_T *in1, unsigned long long in2,
                               const emxArray_creal32_T *in3,
                               const emxArray_creal32_T *in4)
{
  const creal32_T *in3_data;
  const creal32_T *in4_data;
  creal32_T *in1_data;
  int aux_0_1;
  int aux_1_1;
  int i;
  int i1;
  int loop_ub;
  int stride_0_0;
  int stride_0_1;
  int stride_1_0;
  int stride_1_1;
  in4_data = in4->data;
  in3_data = in3->data;
  in1_data = in1->data;
  stride_0_0 = (in3->size[0] != 1);
  stride_0_1 = (in3->size[1] != 1);
  stride_1_0 = (in4->size[0] != 1);
  stride_1_1 = (in4->size[1] != 1);
  aux_0_1 = 0;
  aux_1_1 = 0;
  if (in4->size[1] == 1) {
    loop_ub = in3->size[1];
  } else {
    loop_ub = in4->size[1];
  }
  for (i = 0; i < loop_ub; i++) {
    int b_loop_ub;
    if (in4->size[0] == 1) {
      b_loop_ub = in3->size[0];
    } else {
      b_loop_ub = in4->size[0];
    }
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      int in4_re_tmp;
      in4_re_tmp = i1 * stride_1_0;
      in1_data[(i1 + in1->size[0] * i) +
               in1->size[0] * in1->size[1] * ((int)in2 - 1)]
          .re = in3_data[i1 * stride_0_0 + in3->size[0] * aux_0_1].re +
                0.03F * in4_data[in4_re_tmp + in4->size[0] * aux_1_1].re *
                    0.707106769F;
      in1_data[(i1 + in1->size[0] * i) +
               in1->size[0] * in1->size[1] * ((int)in2 - 1)]
          .im = in3_data[i1 * stride_0_0 + in3->size[0] * aux_0_1].im +
                0.03F * in4_data[in4_re_tmp + in4->size[0] * aux_1_1].im *
                    0.707106769F;
    }
    aux_1_1 += stride_1_1;
    aux_0_1 += stride_0_1;
  }
}

/*
 * Arguments    : emxArray_creal32_T *in1
 *                int in2
 *                const emxArray_creal32_T *in3
 *                const emxArray_creal32_T *in4
 * Return Type  : void
 */
static void binary_expand_op_2(emxArray_creal32_T *in1, int in2,
                               const emxArray_creal32_T *in3,
                               const emxArray_creal32_T *in4)
{
  const creal32_T *in3_data;
  const creal32_T *in4_data;
  creal32_T *in1_data;
  int aux_0_1;
  int aux_1_1;
  int i;
  int i1;
  int loop_ub;
  int stride_0_0;
  int stride_0_1;
  int stride_1_0;
  int stride_1_1;
  in4_data = in4->data;
  in3_data = in3->data;
  in1_data = in1->data;
  stride_0_0 = (in3->size[0] != 1);
  stride_0_1 = (in3->size[1] != 1);
  stride_1_0 = (in4->size[0] != 1);
  stride_1_1 = (in4->size[1] != 1);
  aux_0_1 = 0;
  aux_1_1 = 0;
  if (in4->size[1] == 1) {
    loop_ub = in3->size[1];
  } else {
    loop_ub = in4->size[1];
  }
  for (i = 0; i < loop_ub; i++) {
    int b_loop_ub;
    if (in4->size[0] == 1) {
      b_loop_ub = in3->size[0];
    } else {
      b_loop_ub = in4->size[0];
    }
    for (i1 = 0; i1 < b_loop_ub; i1++) {
      int in4_re_tmp;
      in4_re_tmp = i1 * stride_1_0;
      in1_data[(i1 + in1->size[0] * i) +
               in1->size[0] * in1->size[1] * (in2 - 1)]
          .re = in3_data[i1 * stride_0_0 + in3->size[0] * aux_0_1].re +
                0.03F * in4_data[in4_re_tmp + in4->size[0] * aux_1_1].re *
                    0.707106769F;
      in1_data[(i1 + in1->size[0] * i) +
               in1->size[0] * in1->size[1] * (in2 - 1)]
          .im = in3_data[i1 * stride_0_0 + in3->size[0] * aux_0_1].im +
                0.03F * in4_data[in4_re_tmp + in4->size[0] * aux_1_1].im *
                    0.707106769F;
    }
    aux_1_1 += stride_1_1;
    aux_0_1 += stride_0_1;
  }
}

/*
 * Arguments    : emxArray_creal32_T *in1
 *                const creal_T in2_data[]
 *                const int *in2_size
 *                int in4
 *                const emxArray_creal32_T *in5
 * Return Type  : void
 */
static void binary_expand_op_3(emxArray_creal32_T *in1,
                               const creal_T in2_data[], const int *in2_size,
                               int in4, const emxArray_creal32_T *in5)
{
  emxArray_creal32_T *b_in1;
  const creal32_T *in5_data;
  creal32_T *b_in1_data;
  creal32_T *in1_data;
  int i;
  int in2_re_tmp;
  int loop_ub;
  in5_data = in5->data;
  i = in1->size[0];
  in1->size[0] = *in2_size + in4;
  emxEnsureCapacity_creal32_T(in1, i);
  in1_data = in1->data;
  for (i = 0; i < *in2_size; i++) {
    in1_data[i].re = (float)in2_data[i].re;
    in1_data[i].im = (float)in2_data[i].im;
  }
  loop_ub = in4 - 1;
  for (in2_re_tmp = 0; in2_re_tmp <= loop_ub; in2_re_tmp++) {
    i = in2_re_tmp + *in2_size;
    in1_data[i].re = (float)in2_data[in2_re_tmp].re;
    in1_data[i].im = (float)in2_data[in2_re_tmp].im;
  }
  emxInit_creal32_T(&b_in1, 1);
  i = b_in1->size[0];
  b_in1->size[0] = in1->size[0];
  emxEnsureCapacity_creal32_T(b_in1, i);
  b_in1_data = b_in1->data;
  in2_re_tmp = (in5->size[0] != 1);
  loop_ub = in1->size[0];
  for (i = 0; i < loop_ub; i++) {
    float f;
    float f1;
    float f2;
    float f3;
    int i1;
    f = in1_data[i].re;
    i1 = i * in2_re_tmp;
    f1 = in5_data[i1].im;
    f2 = in1_data[i].im;
    f3 = in5_data[i1].re;
    b_in1_data[i].re = f * f3 - f2 * f1;
    b_in1_data[i].im = f * f1 + f2 * f3;
  }
  i = in1->size[0];
  in1->size[0] = b_in1->size[0];
  emxEnsureCapacity_creal32_T(in1, i);
  in1_data = in1->data;
  loop_ub = b_in1->size[0];
  for (i = 0; i < loop_ub; i++) {
    in1_data[i] = b_in1_data[i];
  }
  emxFree_creal32_T(&b_in1);
}

/*
 * Arguments    : unsigned long long ubits
 *                double bg
 *                double crate
 *                unsigned long long *dbits
 * Return Type  : unsigned long long
 */
static unsigned long long get_ldpc_config(unsigned long long ubits, double bg,
                                          double crate,
                                          unsigned long long *dbits)
{
  static const short iv[51] = {
      2,   3,   4,   5,   6,   7,   8,   9,   10,  11,  12,  13,  14,
      15,  16,  18,  20,  22,  24,  26,  28,  30,  32,  36,  40,  44,
      48,  52,  56,  60,  64,  72,  80,  88,  96,  104, 112, 120, 128,
      144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384};
  unsigned long long ebits;
  int i;
  int iter;
  int zc;
  if (bg == 1.0) {
    i = 22;
  } else {
    i = 10;
  }
  zc = 384;
  iter = -1;
  while (iter + 2 < 51) {
    boolean_T p;
    iter++;
    if (ubits >= 4503599627370496ULL) {
      p = true;
    } else {
      p = ((double)(i * iv[iter]) <= ubits);
    }
    if (p) {
      p = false;
      if (ubits < 4503599627370496ULL) {
        p = ((double)ubits < i * iv[iter + 1]);
      }
      if (p) {
        zc = iv[iter];
      }
    }
  }
  unsigned long long z;
  z = b_i64ddiv((unsigned long long)i, crate);
  ebits = mul_u64_sat(z, (unsigned long long)zc);
  *dbits = mul_u64_sat((unsigned long long)i, (unsigned long long)zc);
  return ebits;
}

/*
 * Arguments    : unsigned long long a
 *                unsigned long long b
 * Return Type  : unsigned long long
 */
static unsigned long long mul_u64_sat(unsigned long long a,
                                      unsigned long long b)
{
  unsigned long long result;
  unsigned long long u64_chi;
  u64_chi = mul_wide_u64(a, b, &result);
  if (u64_chi) {
    result = MAX_uint64_T;
  }
  return result;
}

/*
 * Arguments    : unsigned long long in0
 *                unsigned long long in1
 *                unsigned long long *ptrOutBitsLo
 * Return Type  : unsigned long long
 */
static unsigned long long mul_wide_u64(unsigned long long in0,
                                       unsigned long long in1,
                                       unsigned long long *ptrOutBitsLo)
{
  unsigned long long in0Hi;
  unsigned long long in0Lo;
  unsigned long long in1Hi;
  unsigned long long in1Lo;
  unsigned long long productHiLo;
  unsigned long long productLoHi;
  in0Hi = in0 >> 32ULL;
  in0Lo = in0 & 4294967295ULL;
  in1Hi = in1 >> 32ULL;
  in1Lo = in1 & 4294967295ULL;
  productHiLo = in0Hi * in1Lo;
  productLoHi = in0Lo * in1Hi;
  in0Lo *= in1Lo;
  in1Lo = 0ULL;
  *ptrOutBitsLo = in0Lo + (productLoHi << 32ULL);
  if (*ptrOutBitsLo < in0Lo) {
    in1Lo = 1ULL;
  }
  in0Lo = *ptrOutBitsLo;
  *ptrOutBitsLo += productHiLo << 32ULL;
  if (*ptrOutBitsLo < in0Lo) {
    in1Lo++;
  }
  return ((in1Lo + in0Hi * in1Hi) + (productLoHi >> 32ULL)) +
         (productHiLo >> 32ULL);
}

/*
 * Arguments    : unsigned long long ue
 *                unsigned long long bs
 *                unsigned long long ofdm_ca
 *                unsigned long long ofdm_da
 *                unsigned long long sc_group
 *                unsigned long long num_pilots
 *                unsigned long long num_uplinks
 *                unsigned long long num_downlinks
 *                double modulation_order
 *                double code_rate
 *                double base_graph
 *                unsigned long long pilot_spacing
 * Return Type  : void
 */
void data_generator(unsigned long long ue, unsigned long long bs,
                    unsigned long long ofdm_ca, unsigned long long ofdm_da,
                    unsigned long long sc_group, unsigned long long num_pilots,
                    unsigned long long num_uplinks,
                    unsigned long long num_downlinks, double modulation_order,
                    double code_rate, double base_graph,
                    unsigned long long pilot_spacing)
{
  static const short iv[309] = {
      2,    3,    5,    7,    11,   13,   17,   19,   23,   29,   31,   37,
      41,   43,   47,   53,   59,   61,   67,   71,   73,   79,   83,   89,
      97,   101,  103,  107,  109,  113,  127,  131,  137,  139,  149,  151,
      157,  163,  167,  173,  179,  181,  191,  193,  197,  199,  211,  223,
      227,  229,  233,  239,  241,  251,  257,  263,  269,  271,  277,  281,
      283,  293,  307,  311,  313,  317,  331,  337,  347,  349,  353,  359,
      367,  373,  379,  383,  389,  397,  401,  409,  419,  421,  431,  433,
      439,  443,  449,  457,  461,  463,  467,  479,  487,  491,  499,  503,
      509,  521,  523,  541,  547,  557,  563,  569,  571,  577,  587,  593,
      599,  601,  607,  613,  617,  619,  631,  641,  643,  647,  653,  659,
      661,  673,  677,  683,  691,  701,  709,  719,  727,  733,  739,  743,
      751,  757,  761,  769,  773,  787,  797,  809,  811,  821,  823,  827,
      829,  839,  853,  857,  859,  863,  877,  881,  883,  887,  907,  911,
      919,  929,  937,  941,  947,  953,  967,  971,  977,  983,  991,  997,
      1009, 1013, 1019, 1021, 1031, 1033, 1039, 1049, 1051, 1061, 1063, 1069,
      1087, 1091, 1093, 1097, 1103, 1109, 1117, 1123, 1129, 1151, 1153, 1163,
      1171, 1181, 1187, 1193, 1201, 1213, 1217, 1223, 1229, 1231, 1237, 1249,
      1259, 1277, 1279, 1283, 1289, 1291, 1297, 1301, 1303, 1307, 1319, 1321,
      1327, 1361, 1367, 1373, 1381, 1399, 1409, 1423, 1427, 1429, 1433, 1439,
      1447, 1451, 1453, 1459, 1471, 1481, 1483, 1487, 1489, 1493, 1499, 1511,
      1523, 1531, 1543, 1549, 1553, 1559, 1567, 1571, 1579, 1583, 1597, 1601,
      1607, 1609, 1613, 1619, 1621, 1627, 1637, 1657, 1663, 1667, 1669, 1693,
      1697, 1699, 1709, 1721, 1723, 1733, 1741, 1747, 1753, 1759, 1777, 1783,
      1787, 1789, 1801, 1811, 1823, 1831, 1847, 1861, 1867, 1871, 1873, 1877,
      1879, 1889, 1901, 1907, 1913, 1931, 1933, 1949, 1951, 1973, 1979, 1987,
      1993, 1997, 1999, 2003, 2011, 2017, 2027, 2029, 2039};
  emxArray_creal32_T *A;
  emxArray_creal32_T *C;
  emxArray_creal32_T *a;
  emxArray_creal32_T *b_r;
  emxArray_creal32_T *csi;
  emxArray_creal32_T *pilot_iter;
  emxArray_creal32_T *pilots;
  emxArray_creal32_T *tx_symbols;
  emxArray_creal32_T *uplinks;
  emxArray_creal32_T *zachu_seq;
  emxArray_creal_T *r2;
  emxArray_creal_T *uplinks_modulated;
  emxArray_int8_T *b_a;
  emxArray_int8_T *mask;
  emxArray_real_T *b_uplinks_bits;
  emxArray_real_T *mac_symbols;
  emxArray_real_T *r1;
  emxArray_real_T *uplinks_bits;
  emxArray_real_T *uplinks_encoded;
  emxArray_uint64_T *y;
  creal_T zchu_seq_data[2039];
  creal_T *r3;
  creal_T *uplinks_modulated_data;
  creal32_T *A_data;
  creal32_T *C_data;
  creal32_T *a_data;
  creal32_T *csi_data;
  creal32_T *pilots_data;
  creal32_T *tx_symbols_data;
  creal32_T *uplinks_data;
  creal32_T *zachu_seq_data;
  double ai;
  double r;
  double *mac_symbols_data;
  double *uplinks_bits_data;
  double *uplinks_encoded_data;
  unsigned long long b_varargin_1;
  unsigned long long encoded_uplink;
  unsigned long long iter;
  unsigned long long num_scgroups;
  unsigned long long ofdm_start;
  unsigned long long qY;
  unsigned long long varargin_1;
  unsigned long long x;
  unsigned long long *y_data;
  float A_re_tmp;
  float b_A_re_tmp;
  float b_re_tmp;
  float re_tmp;
  float s_im;
  float s_re;
  int N;
  int b_iter;
  int b_loop_ub;
  int b_n;
  int c_loop_ub;
  int d_loop_ub;
  int i;
  int i1;
  int i2;
  int i3;
  int inner;
  int k;
  int loop_ub;
  int loop_ub_tmp;
  int n;
  int ntilerows;
  int tile_size_idx_0;
  signed char b_fileid;
  signed char fileid;
  signed char *b_a_data;
  signed char *mask_data;
  if (!isInitialized_data_generator) {
    data_generator_initialize();
  }
  /*  This file prepares data for baseband simulation. */
  /* %%%%% CONFIG BEG %%%%%% */
  qY = ofdm_ca - ofdm_da;
  if (qY > ofdm_ca) {
    qY = 0ULL;
  }
  ofdm_start = i64ddiv(qY);
  encoded_uplink =
      get_ldpc_config(times(times(ofdm_da, modulation_order), code_rate),
                      base_graph, code_rate, &varargin_1);
  if (pilot_spacing == 0ULL) {
    if (ofdm_da == 0ULL) {
      num_scgroups = 0ULL;
    } else {
      num_scgroups = MAX_uint64_T;
    }
  } else {
    if (pilot_spacing == 0ULL) {
      num_scgroups = MAX_uint64_T;
    } else {
      num_scgroups = ofdm_da / pilot_spacing;
    }
    x = ofdm_da - num_scgroups * pilot_spacing;
    if ((x > 0ULL) && (x >= (pilot_spacing >> 1ULL) + (pilot_spacing & 1ULL))) {
      num_scgroups++;
    }
  }
  qY = ofdm_da - num_scgroups;
  if (qY > ofdm_da) {
    qY = 0ULL;
  }
  get_ldpc_config(times(times(qY, modulation_order), code_rate), base_graph,
                  code_rate, &b_varargin_1);
  /* %%%%% CONFIG END %%%%%% */
  /* %%%%% CSI GENERATION BEG %%%%%% */
  if (sc_group == 0ULL) {
    if (ofdm_ca == 0ULL) {
      num_scgroups = 0ULL;
    } else {
      num_scgroups = MAX_uint64_T;
    }
  } else {
    if (sc_group == 0ULL) {
      num_scgroups = MAX_uint64_T;
    } else {
      num_scgroups = ofdm_ca / sc_group;
    }
    x = ofdm_ca - num_scgroups * sc_group;
    if ((x > 0ULL) && (x >= (sc_group >> 1ULL) + (sc_group & 1ULL))) {
      num_scgroups++;
    }
  }
  emxInit_creal32_T(&csi, 3);
  i = csi->size[0] * csi->size[1] * csi->size[2];
  csi->size[0] = (int)ue;
  csi->size[1] = (int)bs;
  csi->size[2] = (int)num_scgroups;
  emxEnsureCapacity_creal32_T(csi, i);
  csi_data = csi->data;
  loop_ub = (int)ue * (int)bs * (int)num_scgroups;
  for (i = 0; i < loop_ub; i++) {
    csi_data[i].re = 0.0F;
    csi_data[i].im = 0.0F;
  }
  iter = 1ULL;
  emxInit_creal32_T(&a, 2);
  while (iter <= num_scgroups) {
    complexLike(ue, bs, a);
    a_data = a->data;
    loop_ub = a->size[1];
    for (i = 0; i < loop_ub; i++) {
      b_iter = a->size[0];
      for (n = 0; n < b_iter; n++) {
        csi_data[(n + csi->size[0] * i) +
                 csi->size[0] * csi->size[1] * ((int)iter - 1)]
            .re = 0.707106769F * (2.0F * a_data[n + a->size[0] * i].re - 1.0F);
        csi_data[(n + csi->size[0] * i) +
                 csi->size[0] * csi->size[1] * ((int)iter - 1)]
            .im = 0.707106769F * (2.0F * a_data[n + a->size[0] * i].im);
      }
    }
    iter++;
  }
  /* %%%%% CSI GENERATION END %%%%%% */
  emxInit_creal32_T(&tx_symbols, 3);
  i = tx_symbols->size[0] * tx_symbols->size[1] * tx_symbols->size[2];
  tx_symbols->size[0] = (int)ofdm_ca;
  tx_symbols->size[1] = (int)bs;
  emxEnsureCapacity_creal32_T(tx_symbols, i);
  num_scgroups = num_pilots + num_uplinks;
  qY = num_scgroups;
  if (num_scgroups < num_pilots) {
    qY = MAX_uint64_T;
  }
  i = tx_symbols->size[0] * tx_symbols->size[1] * tx_symbols->size[2];
  tx_symbols->size[2] = (int)qY;
  emxEnsureCapacity_creal32_T(tx_symbols, i);
  tx_symbols_data = tx_symbols->data;
  if (num_scgroups < num_pilots) {
    num_scgroups = MAX_uint64_T;
  }
  loop_ub_tmp = (int)ofdm_ca * (int)bs;
  loop_ub = loop_ub_tmp * (int)num_scgroups;
  for (i = 0; i < loop_ub; i++) {
    tx_symbols_data[i].re = 0.0F;
    tx_symbols_data[i].im = 0.0F;
  }
  emxInit_real_T(&mac_symbols, 3);
  b_rand(b_varargin_1, ue, num_downlinks, mac_symbols);
  mac_symbols_data = mac_symbols->data;
  i = mac_symbols->size[0] * mac_symbols->size[1] * mac_symbols->size[2];
  for (k = 0; k < i; k++) {
    mac_symbols_data[k] = floor(mac_symbols_data[k] * 2.0);
  }
  fileid = cfopen("ant_data.data");
  b_fileid = cfopen("mac_data.data");
  /* %%%%% PILOTS GENERATION BEG %%%%%% */
  /* %%%%% FUNCTIONS %%%%%% */
  b_iter = -1;
  N = 2039;
  int exitg1;
  do {
    exitg1 = 0;
    if (b_iter + 2 < 309) {
      boolean_T p;
      b_iter++;
      if (ofdm_da >= 4503599627370496ULL) {
        p = true;
      } else {
        p = ((double)iv[b_iter] <= ofdm_da);
      }
      if (p) {
        p = false;
        if (ofdm_da < 4503599627370496ULL) {
          p = ((double)ofdm_da < iv[b_iter + 1]);
        }
        if (p) {
          N = iv[b_iter];
          exitg1 = 1;
        }
      }
    } else {
      exitg1 = 1;
    }
  } while (exitg1 == 0);
  r = floor((double)N * 2.0 / 31.0 + 0.5) * -3.1415926535897931;
  loop_ub = N - 1;
  for (i = 0; i <= loop_ub; i++) {
    ai = (double)i * r * ((double)i + 1.0);
    if (ai == 0.0) {
      zchu_seq_data[i].re = -0.0;
      zchu_seq_data[i].im = 0.0;
    } else {
      zchu_seq_data[i].re = 0.0;
      zchu_seq_data[i].im = ai / (double)N;
    }
  }
  for (k = 0; k < N; k++) {
    r = zchu_seq_data[k].re;
    if (r == 0.0) {
      ai = zchu_seq_data[k].im;
      zchu_seq_data[k].re = cos(ai);
      ai = sin(ai);
      zchu_seq_data[k].im = ai;
    } else {
      ai = zchu_seq_data[k].im;
      if (ai == 0.0) {
        r = exp(r);
        zchu_seq_data[k].re = r;
        zchu_seq_data[k].im = 0.0;
      } else if (rtIsInf(ai) && rtIsInf(r) && (r < 0.0)) {
        zchu_seq_data[k].re = 0.0;
        zchu_seq_data[k].im = 0.0;
      } else {
        r = exp(r / 2.0);
        zchu_seq_data[k].re = r * (r * cos(ai));
        ai = r * (r * sin(ai));
        zchu_seq_data[k].im = ai;
      }
    }
  }
  qY = ofdm_da - (unsigned long long)N;
  if (qY > ofdm_da) {
    qY = 0ULL;
  }
  if (qY < 1ULL) {
    loop_ub = 0;
  } else {
    loop_ub = (int)qY;
  }
  qY = ofdm_da - 1ULL;
  if (ofdm_da - 1ULL > ofdm_da) {
    qY = 0ULL;
  }
  n = (int)qY;
  emxInit_uint64_T(&y);
  i = y->size[0] * y->size[1];
  y->size[0] = 1;
  y->size[1] = (int)qY + 1;
  emxEnsureCapacity_uint64_T(y, i);
  y_data = y->data;
  for (k = 0; k <= n; k++) {
    y_data[k] = (unsigned long long)k;
  }
  emxInit_creal32_T(&b_r, 1);
  i = b_r->size[0];
  b_r->size[0] = y->size[1];
  emxEnsureCapacity_creal32_T(b_r, i);
  uplinks_data = b_r->data;
  b_iter = y->size[1];
  for (i = 0; i < b_iter; i++) {
    s_re = (float)y_data[i];
    s_im = s_re * 0.0F * 3.14159274F;
    s_re *= 3.14159274F;
    if (s_re == 0.0F) {
      uplinks_data[i].re = s_im / 4.0F;
      uplinks_data[i].im = 0.0F;
    } else if (s_im == 0.0F) {
      uplinks_data[i].re = 0.0F;
      uplinks_data[i].im = s_re / 4.0F;
    } else {
      uplinks_data[i].re = rtNaNF;
      uplinks_data[i].im = s_re / 4.0F;
    }
  }
  emxFree_uint64_T(&y);
  b_exp(b_r);
  uplinks_data = b_r->data;
  i = N + loop_ub;
  emxInit_creal32_T(&zachu_seq, 1);
  if (i == b_r->size[0]) {
    n = zachu_seq->size[0];
    zachu_seq->size[0] = i;
    emxEnsureCapacity_creal32_T(zachu_seq, n);
    zachu_seq_data = zachu_seq->data;
    for (i = 0; i < N; i++) {
      zachu_seq_data[i].re = (float)zchu_seq_data[i].re;
      zachu_seq_data[i].im = (float)zchu_seq_data[i].im;
    }
    for (i = 0; i < loop_ub; i++) {
      n = i + N;
      zachu_seq_data[n].re = (float)zchu_seq_data[i].re;
      zachu_seq_data[n].im = (float)zchu_seq_data[i].im;
    }
    loop_ub = zachu_seq->size[0];
    for (i = 0; i < loop_ub; i++) {
      s_re = zachu_seq_data[i].re;
      s_im = uplinks_data[i].im;
      re_tmp = zachu_seq_data[i].im;
      b_re_tmp = uplinks_data[i].re;
      zachu_seq_data[i].re = s_re * b_re_tmp - re_tmp * s_im;
      zachu_seq_data[i].im = s_re * s_im + re_tmp * b_re_tmp;
    }
  } else {
    binary_expand_op_3(zachu_seq, zchu_seq_data, &N, loop_ub, b_r);
    zachu_seq_data = zachu_seq->data;
  }
  emxFree_creal32_T(&b_r);
  emxInit_creal32_T(&pilots, 2);
  i = pilots->size[0] * pilots->size[1];
  pilots->size[0] = (int)ofdm_ca;
  pilots->size[1] = (int)sc_group;
  emxEnsureCapacity_creal32_T(pilots, i);
  pilots_data = pilots->data;
  loop_ub = (int)ofdm_ca * (int)sc_group;
  for (i = 0; i < loop_ub; i++) {
    pilots_data[i].re = 0.0F;
    pilots_data[i].im = 0.0F;
  }
  if (sc_group >= 1ULL) {
    b_loop_ub = (int)sc_group;
    if (sc_group == 0ULL) {
      if (ofdm_da == 0ULL) {
        num_scgroups = 0ULL;
      } else {
        num_scgroups = MAX_uint64_T;
      }
    } else {
      if (sc_group == 0ULL) {
        num_scgroups = MAX_uint64_T;
      } else {
        num_scgroups = ofdm_da / sc_group;
      }
      x = ofdm_da - num_scgroups * sc_group;
      if ((x > 0ULL) && (x >= (sc_group >> 1ULL) + (sc_group & 1ULL))) {
        num_scgroups++;
      }
    }
    tile_size_idx_0 = (int)num_scgroups;
    ntilerows = (int)num_scgroups;
    qY = ofdm_start + 1ULL;
    if (ofdm_start + 1ULL < ofdm_start) {
      qY = MAX_uint64_T;
    }
    num_scgroups = ofdm_start + ofdm_da;
    if (num_scgroups < ofdm_start) {
      num_scgroups = MAX_uint64_T;
    }
    if (qY > num_scgroups) {
      i1 = 1;
    } else {
      i1 = (int)qY;
    }
  }
  iter = 1ULL;
  emxInit_int8_T(&mask, 1);
  emxInit_int8_T(&b_a, 1);
  while (iter <= sc_group) {
    i = mask->size[0];
    mask->size[0] = (int)sc_group;
    emxEnsureCapacity_int8_T(mask, i);
    mask_data = mask->data;
    for (i = 0; i < b_loop_ub; i++) {
      mask_data[i] = 0;
    }
    mask_data[(int)iter - 1] = 1;
    i = b_a->size[0];
    b_a->size[0] = mask->size[0];
    emxEnsureCapacity_int8_T(b_a, i);
    b_a_data = b_a->data;
    loop_ub = mask->size[0];
    for (i = 0; i < loop_ub; i++) {
      b_a_data[i] = mask_data[i];
    }
    b_iter = mask->size[0] * tile_size_idx_0;
    i = mask->size[0];
    mask->size[0] = b_iter;
    emxEnsureCapacity_int8_T(mask, i);
    mask_data = mask->data;
    b_iter = b_a->size[0];
    for (n = 0; n < ntilerows; n++) {
      N = n * b_iter;
      for (k = 0; k < b_iter; k++) {
        mask_data[N + k] = b_a_data[k];
      }
    }
    if (zachu_seq->size[0] == mask->size[0]) {
      loop_ub = zachu_seq->size[0];
      for (i = 0; i < loop_ub; i++) {
        n = mask_data[i];
        b_iter = (i1 + i) - 1;
        pilots_data[b_iter + pilots->size[0] * ((int)iter - 1)].re =
            (float)n * zachu_seq_data[i].re;
        pilots_data[b_iter + pilots->size[0] * ((int)iter - 1)].im =
            (float)n * zachu_seq_data[i].im;
      }
    } else {
      binary_expand_op(pilots, i1, iter, zachu_seq, mask);
      pilots_data = pilots->data;
    }
    iter++;
  }
  emxFree_int8_T(&b_a);
  emxFree_int8_T(&mask);
  emxFree_creal32_T(&zachu_seq);
  iter = 1ULL;
  emxInit_creal32_T(&pilot_iter, 2);
  emxInit_creal32_T(&C, 2);
  emxInit_creal32_T(&A, 2);
  while (iter <= num_pilots) {
    i = pilot_iter->size[0] * pilot_iter->size[1];
    pilot_iter->size[0] = (int)ofdm_ca;
    pilot_iter->size[1] = (int)bs;
    emxEnsureCapacity_creal32_T(pilot_iter, i);
    zachu_seq_data = pilot_iter->data;
    for (i = 0; i < loop_ub_tmp; i++) {
      zachu_seq_data[i].re = 0.0F;
      zachu_seq_data[i].im = 0.0F;
    }
    if (ofdm_ca >= 1ULL) {
      qY = iter - 1ULL;
      if (iter - 1ULL > iter) {
        qY = 0ULL;
      }
      num_scgroups = mul_u64_sat(qY, sc_group);
      qY = num_scgroups + 1ULL;
      if (num_scgroups + 1ULL < num_scgroups) {
        qY = MAX_uint64_T;
      }
      num_scgroups = mul_u64_sat(iter, sc_group);
      if (qY > num_scgroups) {
        i2 = 0;
        i3 = 0;
      } else {
        i2 = (int)qY - 1;
        i3 = (int)num_scgroups;
      }
      c_loop_ub = csi->size[1];
      d_loop_ub = pilots->size[1];
      inner = pilots->size[1];
      b_n = csi->size[1];
    }
    for (x = 1ULL; x <= ofdm_ca; x++) {
      if (sc_group == 0ULL) {
        num_scgroups = MAX_uint64_T;
      } else {
        qY = x - 1ULL;
        if (x - 1ULL > x) {
          qY = 0ULL;
        }
        num_scgroups = qY / sc_group;
      }
      qY = num_scgroups + 1ULL;
      if (num_scgroups + 1ULL < num_scgroups) {
        qY = MAX_uint64_T;
      }
      loop_ub = i3 - i2;
      i = a->size[0] * a->size[1];
      a->size[0] = loop_ub;
      a->size[1] = csi->size[1];
      emxEnsureCapacity_creal32_T(a, i);
      a_data = a->data;
      for (i = 0; i < c_loop_ub; i++) {
        for (n = 0; n < loop_ub; n++) {
          a_data[n + a->size[0] * i] =
              csi_data[((i2 + n) + csi->size[0] * i) +
                       csi->size[0] * csi->size[1] * ((int)qY - 1)];
        }
      }
      i = A->size[0] * A->size[1];
      A->size[0] = 1;
      A->size[1] = pilots->size[1];
      emxEnsureCapacity_creal32_T(A, i);
      A_data = A->data;
      for (i = 0; i < d_loop_ub; i++) {
        A_data[i] = pilots_data[((int)x + pilots->size[0] * i) - 1];
      }
      i = C->size[0] * C->size[1];
      C->size[0] = 1;
      C->size[1] = csi->size[1];
      emxEnsureCapacity_creal32_T(C, i);
      C_data = C->data;
      for (b_loop_ub = 0; b_loop_ub < b_n; b_loop_ub++) {
        b_iter = b_loop_ub * loop_ub;
        s_re = 0.0F;
        s_im = 0.0F;
        for (k = 0; k < inner; k++) {
          re_tmp = A_data[k].re;
          N = b_iter + k;
          b_re_tmp = a_data[N].im;
          A_re_tmp = A_data[k].im;
          b_A_re_tmp = a_data[N].re;
          s_re += re_tmp * b_A_re_tmp - A_re_tmp * b_re_tmp;
          s_im += re_tmp * b_re_tmp + A_re_tmp * b_A_re_tmp;
        }
        C_data[b_loop_ub].re = s_re;
        C_data[b_loop_ub].im = s_im;
      }
      loop_ub = C->size[1];
      for (i = 0; i < loop_ub; i++) {
        zachu_seq_data[((int)x + pilot_iter->size[0] * i) - 1] = C_data[i];
      }
    }
    b_complexLike(ofdm_ca, bs, a);
    a_data = a->data;
    if ((pilot_iter->size[0] == a->size[0]) &&
        (pilot_iter->size[1] == a->size[1])) {
      loop_ub = pilot_iter->size[1];
      for (i = 0; i < loop_ub; i++) {
        b_iter = pilot_iter->size[0];
        for (n = 0; n < b_iter; n++) {
          tx_symbols_data[(n + tx_symbols->size[0] * i) +
                          tx_symbols->size[0] * tx_symbols->size[1] *
                              ((int)iter - 1)]
              .re = zachu_seq_data[n + pilot_iter->size[0] * i].re +
                    0.03F * a_data[n + a->size[0] * i].re * 0.707106769F;
          tx_symbols_data[(n + tx_symbols->size[0] * i) +
                          tx_symbols->size[0] * tx_symbols->size[1] *
                              ((int)iter - 1)]
              .im = zachu_seq_data[n + pilot_iter->size[0] * i].im +
                    0.03F * a_data[n + a->size[0] * i].im * 0.707106769F;
        }
      }
    } else {
      binary_expand_op_1(tx_symbols, iter, pilot_iter, a);
      tx_symbols_data = tx_symbols->data;
    }
    iter++;
  }
  emxFree_creal32_T(&pilot_iter);
  /* %%%%% PILOTS GENERATION END %%%%%% */
  /*  smap = [61, 29, 21, 53, 55, 23, 31, 63,... */
  /*          45, 13,  5, 37, 39,  7, 15, 47,... */
  /*          41,  9,  1, 33, 35,  3, 11, 43,... */
  /*          57, 25, 17, 49, 51, 19, 27, 59,... */
  /*          56, 24, 16, 48, 50, 18, 26, 58,... */
  /*          40,  8,  0, 32, 34,  2, 10, 42,... */
  /*          44, 12,  4, 36, 38,  6, 14, 46,... */
  /*          60, 28, 20, 52, 54, 22, 30, 62]; */
  /* %%%%% UPLINKS GENERATION BEG %%%%%% */
  emxInit_real_T(&uplinks_bits, 3);
  b_rand(varargin_1, ue, num_uplinks, uplinks_bits);
  uplinks_bits_data = uplinks_bits->data;
  i = uplinks_bits->size[0] * uplinks_bits->size[1] * uplinks_bits->size[2];
  for (k = 0; k < i; k++) {
    uplinks_bits_data[k] = floor(uplinks_bits_data[k] * 2.0);
  }
  emxInit_real_T(&uplinks_encoded, 3);
  i = uplinks_encoded->size[0] * uplinks_encoded->size[1] *
      uplinks_encoded->size[2];
  uplinks_encoded->size[0] = (int)encoded_uplink;
  uplinks_encoded->size[1] = (int)ue;
  uplinks_encoded->size[2] = (int)num_uplinks;
  emxEnsureCapacity_real_T(uplinks_encoded, i);
  uplinks_encoded_data = uplinks_encoded->data;
  emxInit_creal_T(&uplinks_modulated, 3);
  i = uplinks_modulated->size[0] * uplinks_modulated->size[1] *
      uplinks_modulated->size[2];
  uplinks_modulated->size[0] = (int)ofdm_da;
  uplinks_modulated->size[1] = (int)ue;
  uplinks_modulated->size[2] = (int)num_uplinks;
  emxEnsureCapacity_creal_T(uplinks_modulated, i);
  uplinks_modulated_data = uplinks_modulated->data;
  loop_ub = (int)ofdm_da * (int)ue * (int)num_uplinks;
  for (i = 0; i < loop_ub; i++) {
    uplinks_modulated_data[i].re = 0.0;
    uplinks_modulated_data[i].im = 0.0;
  }
  iter = 1ULL;
  emxInit_real_T(&b_uplinks_bits, 2);
  emxInit_real_T(&r1, 2);
  while (iter <= num_uplinks) {
    i = b_uplinks_bits->size[0] * b_uplinks_bits->size[1];
    b_uplinks_bits->size[0] = uplinks_bits->size[0];
    b_uplinks_bits->size[1] = uplinks_bits->size[1];
    emxEnsureCapacity_real_T(b_uplinks_bits, i);
    mac_symbols_data = b_uplinks_bits->data;
    loop_ub = uplinks_bits->size[1];
    for (i = 0; i < loop_ub; i++) {
      b_iter = uplinks_bits->size[0];
      for (n = 0; n < b_iter; n++) {
        mac_symbols_data[n + b_uplinks_bits->size[0] * i] =
            uplinks_bits_data[(n + uplinks_bits->size[0] * i) +
                              uplinks_bits->size[0] * uplinks_bits->size[1] *
                                  ((int)iter - 1)];
      }
    }
    wlanScramble(b_uplinks_bits, r1);
    mac_symbols_data = r1->data;
    loop_ub = r1->size[1];
    for (i = 0; i < loop_ub; i++) {
      b_iter = r1->size[0];
      for (n = 0; n < b_iter; n++) {
        uplinks_bits_data[(n + uplinks_bits->size[0] * i) +
                          uplinks_bits->size[0] * uplinks_bits->size[1] *
                              ((int)iter - 1)] =
            mac_symbols_data[n + r1->size[0] * i];
      }
    }
    i = b_uplinks_bits->size[0] * b_uplinks_bits->size[1];
    b_uplinks_bits->size[0] = uplinks_bits->size[0];
    b_uplinks_bits->size[1] = uplinks_bits->size[1];
    emxEnsureCapacity_real_T(b_uplinks_bits, i);
    mac_symbols_data = b_uplinks_bits->data;
    loop_ub = uplinks_bits->size[1];
    for (i = 0; i < loop_ub; i++) {
      b_iter = uplinks_bits->size[0];
      for (n = 0; n < b_iter; n++) {
        mac_symbols_data[n + b_uplinks_bits->size[0] * i] =
            uplinks_bits_data[(n + uplinks_bits->size[0] * i) +
                              uplinks_bits->size[0] * uplinks_bits->size[1] *
                                  ((int)iter - 1)];
      }
    }
    nrLDPCEncode(b_uplinks_bits, base_graph, r1);
    mac_symbols_data = r1->data;
    loop_ub = r1->size[1];
    for (i = 0; i < loop_ub; i++) {
      b_iter = r1->size[0];
      for (n = 0; n < b_iter; n++) {
        uplinks_encoded_data[(n + uplinks_encoded->size[0] * i) +
                             uplinks_encoded->size[0] *
                                 uplinks_encoded->size[1] * ((int)iter - 1)] =
            mac_symbols_data[n + r1->size[0] * i];
      }
    }
    iter++;
  }
  emxFree_real_T(&r1);
  emxFree_real_T(&b_uplinks_bits);
  emxInit_creal_T(&r2, 3);
  qammod(uplinks_encoded, rt_powd_snf(2.0, modulation_order), r2);
  r3 = r2->data;
  emxFree_real_T(&uplinks_encoded);
  loop_ub = r2->size[2];
  for (i = 0; i < loop_ub; i++) {
    b_iter = r2->size[1];
    for (n = 0; n < b_iter; n++) {
      b_loop_ub = r2->size[0];
      for (i1 = 0; i1 < b_loop_ub; i1++) {
        uplinks_modulated_data[(i1 + uplinks_modulated->size[0] * n) +
                               uplinks_modulated->size[0] *
                                   uplinks_modulated->size[1] * i] =
            r3[(i1 + r2->size[0] * n) + r2->size[0] * r2->size[1] * i];
      }
    }
  }
  emxFree_creal_T(&r2);
  emxInit_creal32_T(&uplinks, 3);
  i = uplinks->size[0] * uplinks->size[1] * uplinks->size[2];
  uplinks->size[0] = (int)ofdm_ca;
  uplinks->size[1] = (int)ue;
  uplinks->size[2] = (int)num_uplinks;
  emxEnsureCapacity_creal32_T(uplinks, i);
  uplinks_data = uplinks->data;
  loop_ub = (int)ofdm_ca * (int)ue * (int)num_uplinks;
  for (i = 0; i < loop_ub; i++) {
    uplinks_data[i].re = 0.0F;
    uplinks_data[i].im = 0.0F;
  }
  qY = ofdm_start + 1ULL;
  if (ofdm_start + 1ULL < ofdm_start) {
    qY = MAX_uint64_T;
  }
  num_scgroups = ofdm_start + ofdm_da;
  if (num_scgroups < ofdm_start) {
    num_scgroups = MAX_uint64_T;
  }
  if (qY > num_scgroups) {
    i = 0;
  } else {
    i = (int)qY - 1;
  }
  loop_ub = uplinks_modulated->size[2];
  for (n = 0; n < loop_ub; n++) {
    b_iter = uplinks_modulated->size[1];
    for (i1 = 0; i1 < b_iter; i1++) {
      b_loop_ub = uplinks_modulated->size[0];
      for (i2 = 0; i2 < b_loop_ub; i2++) {
        i3 = i + i2;
        uplinks_data[(i3 + uplinks->size[0] * i1) +
                     uplinks->size[0] * uplinks->size[1] * n]
            .re =
            (float)
                uplinks_modulated_data[(i2 + uplinks_modulated->size[0] * i1) +
                                       uplinks_modulated->size[0] *
                                           uplinks_modulated->size[1] * n]
                    .re;
        uplinks_data[(i3 + uplinks->size[0] * i1) +
                     uplinks->size[0] * uplinks->size[1] * n]
            .im =
            (float)
                uplinks_modulated_data[(i2 + uplinks_modulated->size[0] * i1) +
                                       uplinks_modulated->size[0] *
                                           uplinks_modulated->size[1] * n]
                    .im;
      }
    }
  }
  emxFree_creal_T(&uplinks_modulated);
  for (iter = 1ULL; iter <= num_uplinks; iter++) {
    i = pilots->size[0] * pilots->size[1];
    pilots->size[0] = (int)ofdm_ca;
    pilots->size[1] = (int)bs;
    emxEnsureCapacity_creal32_T(pilots, i);
    pilots_data = pilots->data;
    for (i = 0; i < loop_ub_tmp; i++) {
      pilots_data[i].re = 0.0F;
      pilots_data[i].im = 0.0F;
    }
    for (x = 1ULL; x <= ofdm_ca; x++) {
      if (sc_group == 0ULL) {
        num_scgroups = MAX_uint64_T;
      } else {
        qY = x - 1ULL;
        if (x - 1ULL > x) {
          qY = 0ULL;
        }
        num_scgroups = qY / sc_group;
      }
      qY = num_scgroups + 1ULL;
      if (num_scgroups + 1ULL < num_scgroups) {
        qY = MAX_uint64_T;
      }
      i = a->size[0] * a->size[1];
      a->size[0] = csi->size[0];
      a->size[1] = csi->size[1];
      emxEnsureCapacity_creal32_T(a, i);
      a_data = a->data;
      loop_ub = csi->size[1];
      for (i = 0; i < loop_ub; i++) {
        b_iter = csi->size[0];
        for (n = 0; n < b_iter; n++) {
          a_data[n + a->size[0] * i] =
              csi_data[(n + csi->size[0] * i) +
                       csi->size[0] * csi->size[1] * ((int)qY - 1)];
        }
      }
      i = A->size[0] * A->size[1];
      A->size[0] = 1;
      A->size[1] = uplinks->size[1];
      emxEnsureCapacity_creal32_T(A, i);
      A_data = A->data;
      loop_ub = uplinks->size[1];
      for (i = 0; i < loop_ub; i++) {
        A_data[i] = uplinks_data[(((int)x + uplinks->size[0] * i) +
                                  uplinks->size[0] * uplinks->size[1] *
                                      ((int)iter - 1)) -
                                 1];
      }
      inner = uplinks->size[1];
      n = csi->size[1];
      i = C->size[0] * C->size[1];
      C->size[0] = 1;
      C->size[1] = csi->size[1];
      emxEnsureCapacity_creal32_T(C, i);
      C_data = C->data;
      for (b_loop_ub = 0; b_loop_ub < n; b_loop_ub++) {
        b_iter = b_loop_ub * csi->size[0];
        s_re = 0.0F;
        s_im = 0.0F;
        for (k = 0; k < inner; k++) {
          re_tmp = A_data[k].re;
          N = b_iter + k;
          b_re_tmp = a_data[N].im;
          A_re_tmp = A_data[k].im;
          b_A_re_tmp = a_data[N].re;
          s_re += re_tmp * b_A_re_tmp - A_re_tmp * b_re_tmp;
          s_im += re_tmp * b_re_tmp + A_re_tmp * b_A_re_tmp;
        }
        C_data[b_loop_ub].re = s_re;
        C_data[b_loop_ub].im = s_im;
      }
      loop_ub = C->size[1];
      for (i = 0; i < loop_ub; i++) {
        pilots_data[((int)x + pilots->size[0] * i) - 1] = C_data[i];
      }
    }
    b_complexLike(ofdm_ca, bs, a);
    a_data = a->data;
    qY = num_pilots + iter;
    if (qY < num_pilots) {
      qY = MAX_uint64_T;
    }
    if ((pilots->size[0] == a->size[0]) && (pilots->size[1] == a->size[1])) {
      loop_ub = pilots->size[1];
      for (i = 0; i < loop_ub; i++) {
        b_iter = pilots->size[0];
        for (n = 0; n < b_iter; n++) {
          tx_symbols_data[(n + tx_symbols->size[0] * i) +
                          tx_symbols->size[0] * tx_symbols->size[1] *
                              ((int)qY - 1)]
              .re = pilots_data[n + pilots->size[0] * i].re +
                    0.03F * a_data[n + a->size[0] * i].re * 0.707106769F;
          tx_symbols_data[(n + tx_symbols->size[0] * i) +
                          tx_symbols->size[0] * tx_symbols->size[1] *
                              ((int)qY - 1)]
              .im = pilots_data[n + pilots->size[0] * i].im +
                    0.03F * a_data[n + a->size[0] * i].im * 0.707106769F;
        }
      }
    } else {
      binary_expand_op_2(tx_symbols, (int)qY, pilots, a);
      tx_symbols_data = tx_symbols->data;
    }
  }
  emxFree_creal32_T(&A);
  emxFree_creal32_T(&C);
  emxFree_creal32_T(&a);
  emxFree_creal32_T(&uplinks);
  emxFree_creal32_T(&pilots);
  /* %%%%% UPLINKS GENERATION END %%%%%% */
  /* %%%%% IFFT TX SYMBOLS BEG %%%%%% */
  circshift(tx_symbols, i64ddiv(ofdm_ca));
  tx_symbols_data = tx_symbols->data;
  i = csi->size[0] * csi->size[1] * csi->size[2];
  csi->size[0] = tx_symbols->size[0];
  csi->size[1] = tx_symbols->size[1];
  csi->size[2] = tx_symbols->size[2];
  emxEnsureCapacity_creal32_T(csi, i);
  csi_data = csi->data;
  loop_ub = tx_symbols->size[0] * tx_symbols->size[1] * tx_symbols->size[2] - 1;
  for (i = 0; i <= loop_ub; i++) {
    csi_data[i] = tx_symbols_data[i];
  }
  ifft(csi, tx_symbols);
  emxFree_creal32_T(&csi);
  /* %%%%% IFFT TX SYMBOLS END %%%%%% */
  /* %%%%% WRITE DATA BEG %%%%%% */
  b_fwrite(fileid, tx_symbols);
  emxFree_creal32_T(&tx_symbols);
  bit2int(mac_symbols, uplinks_bits);
  emxFree_real_T(&mac_symbols);
  c_fwrite(b_fileid, uplinks_bits);
  emxFree_real_T(&uplinks_bits);
  cfclose(fileid);
  cfclose(b_fileid);
  /* %%%%% WRITE DATA END %%%%%% */
}

/*
 * File trailer for data_generator.c
 *
 * [EOF]
 */
