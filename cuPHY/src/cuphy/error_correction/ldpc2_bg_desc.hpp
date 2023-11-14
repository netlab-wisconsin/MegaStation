/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_BG_DESC_HPP_INCLUDED_)
#define LDPC2_BG_DESC_HPP_INCLUDED_

#include "nrLDPC_templates.cuh"
#include "cuphy_internal.h"
#include "cuda_fp16.h"

namespace ldpc2
{
    
////////////////////////////////////////////////////////////////////////
// row_pair_index
// Struct to provide the index of a struct or scalar that provides info
// for the first "row pair" in a given parity check row. (A row pair
// refers to a pair of elements in a parity check row.)
// When paired, rows with odd degree will have a padded last element.
// As an example, in BG1, the first row has 19 elements. The row pair
// index of row 0 will be 0, and the row pair index of row 1 will be
// 10 = (19 + 1) / 2.
template <int BG, int CHECK_IDX> struct row_pair_index
{
    static const int value = row_pair_index<BG, CHECK_IDX-1>::value + div_round_up_t<row_degree<BG, CHECK_IDX-1>::value, 2>::value;
};
template <int BG> struct row_pair_index<BG, 0>
{
    static const int value = 0;
};

////////////////////////////////////////////////////////////////////////
// BG1_ROW_ADDR_PAIR_IDX
// Provides array offsets when parity nodes are processes in PAIRS. For
// example, a row with degree 19 will contain ceil(19/2) nonzero variable
// nodes, and therefore we will allocate 10 "pair" descriptors.
enum BG1_ROW_ADDR_PAIR_IDX
{
    BG1_ADDR_PAIR_IDX_ROW_0  = row_pair_index<1, 0>::value,
    BG1_ADDR_PAIR_IDX_ROW_1  = row_pair_index<1, 1>::value,
    BG1_ADDR_PAIR_IDX_ROW_2  = row_pair_index<1, 2>::value,
    BG1_ADDR_PAIR_IDX_ROW_3  = row_pair_index<1, 3>::value,
    BG1_ADDR_PAIR_IDX_ROW_4  = row_pair_index<1, 4>::value,
    BG1_ADDR_PAIR_IDX_ROW_5  = row_pair_index<1, 5>::value,
    BG1_ADDR_PAIR_IDX_ROW_6  = row_pair_index<1, 6>::value,
    BG1_ADDR_PAIR_IDX_ROW_7  = row_pair_index<1, 7>::value,
    BG1_ADDR_PAIR_IDX_ROW_8  = row_pair_index<1, 8>::value,
    BG1_ADDR_PAIR_IDX_ROW_9  = row_pair_index<1, 9>::value,
    BG1_ADDR_PAIR_IDX_ROW_10 = row_pair_index<1, 10>::value,
    BG1_ADDR_PAIR_IDX_ROW_11 = row_pair_index<1, 11>::value,
    BG1_ADDR_PAIR_IDX_ROW_12 = row_pair_index<1, 12>::value,
    BG1_ADDR_PAIR_IDX_ROW_13 = row_pair_index<1, 13>::value,
    BG1_ADDR_PAIR_IDX_ROW_14 = row_pair_index<1, 14>::value,
    BG1_ADDR_PAIR_IDX_ROW_15 = row_pair_index<1, 15>::value,
    BG1_ADDR_PAIR_IDX_ROW_16 = row_pair_index<1, 16>::value,
    BG1_ADDR_PAIR_IDX_ROW_17 = row_pair_index<1, 17>::value,
    BG1_ADDR_PAIR_IDX_ROW_18 = row_pair_index<1, 18>::value,
    BG1_ADDR_PAIR_IDX_ROW_19 = row_pair_index<1, 19>::value,
    BG1_ADDR_PAIR_IDX_ROW_20 = row_pair_index<1, 20>::value,
    BG1_ADDR_PAIR_IDX_ROW_21 = row_pair_index<1, 21>::value,
    BG1_ADDR_PAIR_IDX_ROW_22 = row_pair_index<1, 22>::value,
    BG1_ADDR_PAIR_IDX_ROW_23 = row_pair_index<1, 23>::value,
    BG1_ADDR_PAIR_IDX_ROW_24 = row_pair_index<1, 24>::value,
    BG1_ADDR_PAIR_IDX_ROW_25 = row_pair_index<1, 25>::value,
    BG1_ADDR_PAIR_IDX_ROW_26 = row_pair_index<1, 26>::value,
    BG1_ADDR_PAIR_IDX_ROW_27 = row_pair_index<1, 27>::value,
    BG1_ADDR_PAIR_IDX_ROW_28 = row_pair_index<1, 28>::value,
    BG1_ADDR_PAIR_IDX_ROW_29 = row_pair_index<1, 29>::value,
    BG1_ADDR_PAIR_IDX_ROW_30 = row_pair_index<1, 30>::value,
    BG1_ADDR_PAIR_IDX_ROW_31 = row_pair_index<1, 31>::value,
    BG1_ADDR_PAIR_IDX_ROW_32 = row_pair_index<1, 32>::value,
    BG1_ADDR_PAIR_IDX_ROW_33 = row_pair_index<1, 33>::value,
    BG1_ADDR_PAIR_IDX_ROW_34 = row_pair_index<1, 34>::value,
    BG1_ADDR_PAIR_IDX_ROW_35 = row_pair_index<1, 35>::value,
    BG1_ADDR_PAIR_IDX_ROW_36 = row_pair_index<1, 36>::value,
    BG1_ADDR_PAIR_IDX_ROW_37 = row_pair_index<1, 37>::value,
    BG1_ADDR_PAIR_IDX_ROW_38 = row_pair_index<1, 38>::value,
    BG1_ADDR_PAIR_IDX_ROW_39 = row_pair_index<1, 39>::value,
    BG1_ADDR_PAIR_IDX_ROW_40 = row_pair_index<1, 40>::value,
    BG1_ADDR_PAIR_IDX_ROW_41 = row_pair_index<1, 41>::value,
    BG1_ADDR_PAIR_IDX_ROW_42 = row_pair_index<1, 42>::value,
    BG1_ADDR_PAIR_IDX_ROW_43 = row_pair_index<1, 43>::value,
    BG1_ADDR_PAIR_IDX_ROW_44 = row_pair_index<1, 44>::value,
    BG1_ADDR_PAIR_IDX_ROW_45 = row_pair_index<1, 45>::value,
    BG1_ADDR_PAIR_COUNT      = row_pair_index<1, 45>::value + div_round_up_t<row_degree<1,45>::value, 2>::value
};

////////////////////////////////////////////////////////////////////////
// BG2_ROW_ADDR_PAIR_IDX
// Provides array offsets when parity nodes are processes in PAIRS. For
// example, a row with degree 19 will contain ceil(19/2) nonzero variable
// nodes, and therefore we will allocate 10 "pair" descriptors.
enum BG2_ROW_ADDR_PAIR_IDX
{
    BG2_ADDR_PAIR_IDX_ROW_0  = row_pair_index<2, 0>::value,
    BG2_ADDR_PAIR_IDX_ROW_1  = row_pair_index<2, 1>::value,
    BG2_ADDR_PAIR_IDX_ROW_2  = row_pair_index<2, 2>::value,
    BG2_ADDR_PAIR_IDX_ROW_3  = row_pair_index<2, 3>::value,
    BG2_ADDR_PAIR_IDX_ROW_4  = row_pair_index<2, 4>::value,
    BG2_ADDR_PAIR_IDX_ROW_5  = row_pair_index<2, 5>::value,
    BG2_ADDR_PAIR_IDX_ROW_6  = row_pair_index<2, 6>::value,
    BG2_ADDR_PAIR_IDX_ROW_7  = row_pair_index<2, 7>::value,
    BG2_ADDR_PAIR_IDX_ROW_8  = row_pair_index<2, 8>::value,
    BG2_ADDR_PAIR_IDX_ROW_9  = row_pair_index<2, 9>::value,
    BG2_ADDR_PAIR_IDX_ROW_10 = row_pair_index<2, 10>::value,
    BG2_ADDR_PAIR_IDX_ROW_11 = row_pair_index<2, 11>::value,
    BG2_ADDR_PAIR_IDX_ROW_12 = row_pair_index<2, 12>::value,
    BG2_ADDR_PAIR_IDX_ROW_13 = row_pair_index<2, 13>::value,
    BG2_ADDR_PAIR_IDX_ROW_14 = row_pair_index<2, 14>::value,
    BG2_ADDR_PAIR_IDX_ROW_15 = row_pair_index<2, 15>::value,
    BG2_ADDR_PAIR_IDX_ROW_16 = row_pair_index<2, 16>::value,
    BG2_ADDR_PAIR_IDX_ROW_17 = row_pair_index<2, 17>::value,
    BG2_ADDR_PAIR_IDX_ROW_18 = row_pair_index<2, 18>::value,
    BG2_ADDR_PAIR_IDX_ROW_19 = row_pair_index<2, 19>::value,
    BG2_ADDR_PAIR_IDX_ROW_20 = row_pair_index<2, 20>::value,
    BG2_ADDR_PAIR_IDX_ROW_21 = row_pair_index<2, 21>::value,
    BG2_ADDR_PAIR_IDX_ROW_22 = row_pair_index<2, 22>::value,
    BG2_ADDR_PAIR_IDX_ROW_23 = row_pair_index<2, 23>::value,
    BG2_ADDR_PAIR_IDX_ROW_24 = row_pair_index<2, 24>::value,
    BG2_ADDR_PAIR_IDX_ROW_25 = row_pair_index<2, 25>::value,
    BG2_ADDR_PAIR_IDX_ROW_26 = row_pair_index<2, 26>::value,
    BG2_ADDR_PAIR_IDX_ROW_27 = row_pair_index<2, 27>::value,
    BG2_ADDR_PAIR_IDX_ROW_28 = row_pair_index<2, 28>::value,
    BG2_ADDR_PAIR_IDX_ROW_29 = row_pair_index<2, 29>::value,
    BG2_ADDR_PAIR_IDX_ROW_30 = row_pair_index<2, 30>::value,
    BG2_ADDR_PAIR_IDX_ROW_31 = row_pair_index<2, 31>::value,
    BG2_ADDR_PAIR_IDX_ROW_32 = row_pair_index<2, 32>::value,
    BG2_ADDR_PAIR_IDX_ROW_33 = row_pair_index<2, 33>::value,
    BG2_ADDR_PAIR_IDX_ROW_34 = row_pair_index<2, 34>::value,
    BG2_ADDR_PAIR_IDX_ROW_35 = row_pair_index<2, 35>::value,
    BG2_ADDR_PAIR_IDX_ROW_36 = row_pair_index<2, 36>::value,
    BG2_ADDR_PAIR_IDX_ROW_37 = row_pair_index<2, 37>::value,
    BG2_ADDR_PAIR_IDX_ROW_38 = row_pair_index<2, 38>::value,
    BG2_ADDR_PAIR_IDX_ROW_39 = row_pair_index<2, 39>::value,
    BG2_ADDR_PAIR_IDX_ROW_40 = row_pair_index<2, 40>::value,
    BG2_ADDR_PAIR_IDX_ROW_41 = row_pair_index<2, 41>::value,
    BG2_ADDR_PAIR_COUNT      = row_pair_index<2, 41>::value + div_round_up_t<row_degree<2,41>::value, 2>::value
};

////////////////////////////////////////////////////////////////////////
// ldpc2::LDPC_node_desc
// 3GPP 5G Base Graph Node Descriptor
struct LDPC_node_desc
{
    uint32_t shift_mod;
    uint32_t col_Z_sz;
};

template <int BG> struct BG_desc;
template <> struct BG_desc<1>
{
    LDPC_node_desc nodes[BG1_ADDR_PAIR_COUNT];
};
template <> struct BG_desc<2>
{
    LDPC_node_desc nodes[BG2_ADDR_PAIR_COUNT];
};

typedef BG_desc<1> BG1_desc_t;
typedef struct BG_desc<2> BG2_desc_t;

// Host descriptor declarations
extern const BG1_desc_t BG1_desc_Z32_half;
extern const BG1_desc_t BG1_desc_Z36_half;
extern const BG1_desc_t BG1_desc_Z40_half;
extern const BG1_desc_t BG1_desc_Z44_half;
extern const BG1_desc_t BG1_desc_Z48_half;
extern const BG1_desc_t BG1_desc_Z52_half;
extern const BG1_desc_t BG1_desc_Z56_half;
extern const BG1_desc_t BG1_desc_Z60_half;
extern const BG1_desc_t BG1_desc_Z64_half;
extern const BG1_desc_t BG1_desc_Z72_half;
extern const BG1_desc_t BG1_desc_Z80_half;
extern const BG1_desc_t BG1_desc_Z88_half;
extern const BG1_desc_t BG1_desc_Z96_half;
extern const BG1_desc_t BG1_desc_Z104_half;
extern const BG1_desc_t BG1_desc_Z112_half;
extern const BG1_desc_t BG1_desc_Z120_half;
extern const BG1_desc_t BG1_desc_Z128_half;
extern const BG1_desc_t BG1_desc_Z144_half;
extern const BG1_desc_t BG1_desc_Z160_half;
extern const BG1_desc_t BG1_desc_Z176_half;
extern const BG1_desc_t BG1_desc_Z192_half;
extern const BG1_desc_t BG1_desc_Z208_half;
extern const BG1_desc_t BG1_desc_Z224_half;
extern const BG1_desc_t BG1_desc_Z240_half;
extern const BG1_desc_t BG1_desc_Z256_half;
extern const BG1_desc_t BG1_desc_Z288_half;
extern const BG1_desc_t BG1_desc_Z320_half;
extern const BG1_desc_t BG1_desc_Z352_half;
extern const BG1_desc_t BG1_desc_Z384_half;

extern const BG1_desc_t BG1_desc_Z32_half2;
extern const BG1_desc_t BG1_desc_Z36_half2;
extern const BG1_desc_t BG1_desc_Z40_half2;
extern const BG1_desc_t BG1_desc_Z44_half2;
extern const BG1_desc_t BG1_desc_Z48_half2;
extern const BG1_desc_t BG1_desc_Z52_half2;
extern const BG1_desc_t BG1_desc_Z56_half2;
extern const BG1_desc_t BG1_desc_Z60_half2;
extern const BG1_desc_t BG1_desc_Z64_half2;
extern const BG1_desc_t BG1_desc_Z72_half2;
extern const BG1_desc_t BG1_desc_Z80_half2;
extern const BG1_desc_t BG1_desc_Z88_half2;
extern const BG1_desc_t BG1_desc_Z96_half2;
extern const BG1_desc_t BG1_desc_Z104_half2;
extern const BG1_desc_t BG1_desc_Z112_half2;
extern const BG1_desc_t BG1_desc_Z120_half2;
extern const BG1_desc_t BG1_desc_Z128_half2;
extern const BG1_desc_t BG1_desc_Z144_half2;
extern const BG1_desc_t BG1_desc_Z160_half2;
extern const BG1_desc_t BG1_desc_Z176_half2;
extern const BG1_desc_t BG1_desc_Z192_half2;
extern const BG1_desc_t BG1_desc_Z208_half2;
extern const BG1_desc_t BG1_desc_Z224_half2;
extern const BG1_desc_t BG1_desc_Z240_half2;
extern const BG1_desc_t BG1_desc_Z256_half2;
extern const BG1_desc_t BG1_desc_Z288_half2;
extern const BG1_desc_t BG1_desc_Z320_half2;
extern const BG1_desc_t BG1_desc_Z352_half2;
extern const BG1_desc_t BG1_desc_Z384_half2;


extern const BG2_desc_t BG2_desc_Z32_half;
extern const BG2_desc_t BG2_desc_Z36_half;
extern const BG2_desc_t BG2_desc_Z40_half;
extern const BG2_desc_t BG2_desc_Z44_half;
extern const BG2_desc_t BG2_desc_Z48_half;
extern const BG2_desc_t BG2_desc_Z52_half;
extern const BG2_desc_t BG2_desc_Z56_half;
extern const BG2_desc_t BG2_desc_Z60_half;
extern const BG2_desc_t BG2_desc_Z64_half;
extern const BG2_desc_t BG2_desc_Z72_half;
extern const BG2_desc_t BG2_desc_Z80_half;
extern const BG2_desc_t BG2_desc_Z88_half;
extern const BG2_desc_t BG2_desc_Z96_half;
extern const BG2_desc_t BG2_desc_Z104_half;
extern const BG2_desc_t BG2_desc_Z112_half;
extern const BG2_desc_t BG2_desc_Z120_half;
extern const BG2_desc_t BG2_desc_Z128_half;
extern const BG2_desc_t BG2_desc_Z144_half;
extern const BG2_desc_t BG2_desc_Z160_half;
extern const BG2_desc_t BG2_desc_Z176_half;
extern const BG2_desc_t BG2_desc_Z192_half;
extern const BG2_desc_t BG2_desc_Z208_half;
extern const BG2_desc_t BG2_desc_Z224_half;
extern const BG2_desc_t BG2_desc_Z240_half;
extern const BG2_desc_t BG2_desc_Z256_half;
extern const BG2_desc_t BG2_desc_Z288_half;
extern const BG2_desc_t BG2_desc_Z320_half;
extern const BG2_desc_t BG2_desc_Z352_half;
extern const BG2_desc_t BG2_desc_Z384_half;

extern const BG2_desc_t BG2_desc_Z32_half2;
extern const BG2_desc_t BG2_desc_Z36_half2;
extern const BG2_desc_t BG2_desc_Z40_half2;
extern const BG2_desc_t BG2_desc_Z44_half2;
extern const BG2_desc_t BG2_desc_Z48_half2;
extern const BG2_desc_t BG2_desc_Z52_half2;
extern const BG2_desc_t BG2_desc_Z56_half2;
extern const BG2_desc_t BG2_desc_Z60_half2;
extern const BG2_desc_t BG2_desc_Z64_half2;
extern const BG2_desc_t BG2_desc_Z72_half2;
extern const BG2_desc_t BG2_desc_Z80_half2;
extern const BG2_desc_t BG2_desc_Z88_half2;
extern const BG2_desc_t BG2_desc_Z96_half2;
extern const BG2_desc_t BG2_desc_Z104_half2;
extern const BG2_desc_t BG2_desc_Z112_half2;
extern const BG2_desc_t BG2_desc_Z120_half2;
extern const BG2_desc_t BG2_desc_Z128_half2;
extern const BG2_desc_t BG2_desc_Z144_half2;
extern const BG2_desc_t BG2_desc_Z160_half2;
extern const BG2_desc_t BG2_desc_Z176_half2;
extern const BG2_desc_t BG2_desc_Z192_half2;
extern const BG2_desc_t BG2_desc_Z208_half2;
extern const BG2_desc_t BG2_desc_Z224_half2;
extern const BG2_desc_t BG2_desc_Z240_half2;
extern const BG2_desc_t BG2_desc_Z256_half2;
extern const BG2_desc_t BG2_desc_Z288_half2;
extern const BG2_desc_t BG2_desc_Z320_half2;
extern const BG2_desc_t BG2_desc_Z352_half2;
extern const BG2_desc_t BG2_desc_Z384_half2;

template <typename T, int BG> const BG_desc<BG>* get_BG_desc(int Z);

template <>
inline
const BG_desc<1>* get_BG_desc<__half, 1>(int Z)
{
    const BG1_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG1_desc_Z32_half;  break;
    case 36:  bgd = &BG1_desc_Z36_half;  break;
    case 40:  bgd = &BG1_desc_Z40_half;  break;
    case 44:  bgd = &BG1_desc_Z44_half;  break;
    case 48:  bgd = &BG1_desc_Z48_half;  break;
    case 52:  bgd = &BG1_desc_Z52_half;  break;
    case 56:  bgd = &BG1_desc_Z56_half;  break;
    case 60:  bgd = &BG1_desc_Z60_half;  break;
    case 64:  bgd = &BG1_desc_Z64_half;  break;
    case 72:  bgd = &BG1_desc_Z72_half;  break;
    case 80:  bgd = &BG1_desc_Z80_half;  break;
    case 88:  bgd = &BG1_desc_Z88_half;  break;
    case 96:  bgd = &BG1_desc_Z96_half;  break;
    case 104: bgd = &BG1_desc_Z104_half; break;
    case 112: bgd = &BG1_desc_Z112_half; break;
    case 120: bgd = &BG1_desc_Z120_half; break;
    case 128: bgd = &BG1_desc_Z128_half; break;
    case 144: bgd = &BG1_desc_Z144_half; break;
    case 160: bgd = &BG1_desc_Z160_half; break;
    case 176: bgd = &BG1_desc_Z176_half; break;
    case 192: bgd = &BG1_desc_Z192_half; break;
    case 208: bgd = &BG1_desc_Z208_half; break;
    case 224: bgd = &BG1_desc_Z224_half; break;
    case 240: bgd = &BG1_desc_Z240_half; break;
    case 256: bgd = &BG1_desc_Z256_half; break;
    case 288: bgd = &BG1_desc_Z288_half; break;
    case 320: bgd = &BG1_desc_Z320_half; break;
    case 352: bgd = &BG1_desc_Z352_half; break;
    case 384: bgd = &BG1_desc_Z384_half; break;
    }
    return bgd;
}

template <>
inline
const BG_desc<1>* get_BG_desc<__half2, 1>(int Z)
{
    const BG1_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG1_desc_Z32_half2;  break;
    case 36:  bgd = &BG1_desc_Z36_half2;  break;
    case 40:  bgd = &BG1_desc_Z40_half2;  break;
    case 44:  bgd = &BG1_desc_Z44_half2;  break;
    case 48:  bgd = &BG1_desc_Z48_half2;  break;
    case 52:  bgd = &BG1_desc_Z52_half2;  break;
    case 56:  bgd = &BG1_desc_Z56_half2;  break;
    case 60:  bgd = &BG1_desc_Z60_half2;  break;
    case 64:  bgd = &BG1_desc_Z64_half2;  break;
    case 72:  bgd = &BG1_desc_Z72_half2;  break;
    case 80:  bgd = &BG1_desc_Z80_half2;  break;
    case 88:  bgd = &BG1_desc_Z88_half2;  break;
    case 96:  bgd = &BG1_desc_Z96_half2;  break;
    case 104: bgd = &BG1_desc_Z104_half2; break;
    case 112: bgd = &BG1_desc_Z112_half2; break;
    case 120: bgd = &BG1_desc_Z120_half2; break;
    case 128: bgd = &BG1_desc_Z128_half2; break;
    case 144: bgd = &BG1_desc_Z144_half2; break;
    case 160: bgd = &BG1_desc_Z160_half2; break;
    case 176: bgd = &BG1_desc_Z176_half2; break;
    case 192: bgd = &BG1_desc_Z192_half2; break;
    case 208: bgd = &BG1_desc_Z208_half2; break;
    case 224: bgd = &BG1_desc_Z224_half2; break;
    case 240: bgd = &BG1_desc_Z240_half2; break;
    case 256: bgd = &BG1_desc_Z256_half2; break;
    case 288: bgd = &BG1_desc_Z288_half2; break;
    case 320: bgd = &BG1_desc_Z320_half2; break;
    case 352: bgd = &BG1_desc_Z352_half2; break;
    case 384: bgd = &BG1_desc_Z384_half2; break;
    }
    return bgd;
}

template <>
inline
const BG_desc<2>* get_BG_desc<__half, 2>(int Z)
{
    const BG2_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG2_desc_Z32_half;  break;
    case 36:  bgd = &BG2_desc_Z36_half;  break;
    case 40:  bgd = &BG2_desc_Z40_half;  break;
    case 44:  bgd = &BG2_desc_Z44_half;  break;
    case 48:  bgd = &BG2_desc_Z48_half;  break;
    case 52:  bgd = &BG2_desc_Z52_half;  break;
    case 56:  bgd = &BG2_desc_Z56_half;  break;
    case 60:  bgd = &BG2_desc_Z60_half;  break;
    case 64:  bgd = &BG2_desc_Z64_half;  break;
    case 72:  bgd = &BG2_desc_Z72_half;  break;
    case 80:  bgd = &BG2_desc_Z80_half;  break;
    case 88:  bgd = &BG2_desc_Z88_half;  break;
    case 96:  bgd = &BG2_desc_Z96_half;  break;
    case 104: bgd = &BG2_desc_Z104_half; break;
    case 112: bgd = &BG2_desc_Z112_half; break;
    case 120: bgd = &BG2_desc_Z120_half; break;
    case 128: bgd = &BG2_desc_Z128_half; break;
    case 144: bgd = &BG2_desc_Z144_half; break;
    case 160: bgd = &BG2_desc_Z160_half; break;
    case 176: bgd = &BG2_desc_Z176_half; break;
    case 192: bgd = &BG2_desc_Z192_half; break;
    case 208: bgd = &BG2_desc_Z208_half; break;
    case 224: bgd = &BG2_desc_Z224_half; break;
    case 240: bgd = &BG2_desc_Z240_half; break;
    case 256: bgd = &BG2_desc_Z256_half; break;
    case 288: bgd = &BG2_desc_Z288_half; break;
    case 320: bgd = &BG2_desc_Z320_half; break;
    case 352: bgd = &BG2_desc_Z352_half; break;
    case 384: bgd = &BG2_desc_Z384_half; break;
    }
    return bgd;
}

template <>
inline
const BG_desc<2>* get_BG_desc<__half2, 2>(int Z)
{
    const BG2_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG2_desc_Z32_half2;  break;
    case 36:  bgd = &BG2_desc_Z36_half2;  break;
    case 40:  bgd = &BG2_desc_Z40_half2;  break;
    case 44:  bgd = &BG2_desc_Z44_half2;  break;
    case 48:  bgd = &BG2_desc_Z48_half2;  break;
    case 52:  bgd = &BG2_desc_Z52_half2;  break;
    case 56:  bgd = &BG2_desc_Z56_half2;  break;
    case 60:  bgd = &BG2_desc_Z60_half2;  break;
    case 64:  bgd = &BG2_desc_Z64_half2;  break;
    case 72:  bgd = &BG2_desc_Z72_half2;  break;
    case 80:  bgd = &BG2_desc_Z80_half2;  break;
    case 88:  bgd = &BG2_desc_Z88_half2;  break;
    case 96:  bgd = &BG2_desc_Z96_half2;  break;
    case 104: bgd = &BG2_desc_Z104_half2; break;
    case 112: bgd = &BG2_desc_Z112_half2; break;
    case 120: bgd = &BG2_desc_Z120_half2; break;
    case 128: bgd = &BG2_desc_Z128_half2; break;
    case 144: bgd = &BG2_desc_Z144_half2; break;
    case 160: bgd = &BG2_desc_Z160_half2; break;
    case 176: bgd = &BG2_desc_Z176_half2; break;
    case 192: bgd = &BG2_desc_Z192_half2; break;
    case 208: bgd = &BG2_desc_Z208_half2; break;
    case 224: bgd = &BG2_desc_Z224_half2; break;
    case 240: bgd = &BG2_desc_Z240_half2; break;
    case 256: bgd = &BG2_desc_Z256_half2; break;
    case 288: bgd = &BG2_desc_Z288_half2; break;
    case 320: bgd = &BG2_desc_Z320_half2; break;
    case 352: bgd = &BG2_desc_Z352_half2; break;
    case 384: bgd = &BG2_desc_Z384_half2; break;
    }
    return bgd;
}

////////////////////////////////////////////////////////////////////////
// ldpc2::LDPC_adj_node_desc
// 3GPP 5G Base Graph Node Descriptor (alternate "adjusted" data storage
// format)
struct LDPC_adj_node_desc
{
    uint32_t wrap_index;
    int32_t  col_Z_shift_low;
    int32_t  col_Z_shift_high;
};

template <int BG> struct BG_adj_desc;
template <> struct BG_adj_desc<1>
{
    LDPC_adj_node_desc nodes[BG1_ADDR_PAIR_COUNT];
};
template <> struct BG_adj_desc<2>
{
    LDPC_adj_node_desc nodes[BG2_ADDR_PAIR_COUNT];
};

typedef BG_adj_desc<1> BG1_adj_desc_t;
typedef struct BG_adj_desc<2> BG2_adj_desc_t;

// Host descriptor declarations
extern const BG1_adj_desc_t BG1_adj_desc_Z2_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z3_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z4_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z5_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z6_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z7_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z8_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z9_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z10_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z11_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z12_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z13_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z14_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z15_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z16_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z18_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z20_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z22_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z24_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z26_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z28_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z30_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z32_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z36_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z40_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z44_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z48_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z52_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z56_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z60_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z64_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z72_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z80_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z88_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z96_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z104_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z112_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z120_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z128_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z144_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z160_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z176_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z192_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z208_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z224_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z240_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z256_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z288_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z320_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z352_half;
extern const BG1_adj_desc_t BG1_adj_desc_Z384_half;

extern const BG1_adj_desc_t BG1_adj_desc_Z32_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z36_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z40_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z44_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z48_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z52_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z56_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z60_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z64_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z72_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z80_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z88_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z96_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z104_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z112_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z120_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z128_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z144_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z160_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z176_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z192_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z208_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z224_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z240_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z256_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z288_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z320_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z352_half2;
extern const BG1_adj_desc_t BG1_adj_desc_Z384_half2;


extern const BG2_adj_desc_t BG2_adj_desc_Z2_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z3_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z4_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z5_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z6_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z7_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z8_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z9_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z10_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z11_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z12_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z13_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z14_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z15_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z16_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z18_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z20_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z22_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z24_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z26_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z28_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z30_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z32_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z36_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z40_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z44_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z48_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z52_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z56_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z60_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z64_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z72_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z80_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z88_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z96_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z104_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z112_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z120_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z128_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z144_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z160_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z176_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z192_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z208_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z224_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z240_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z256_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z288_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z320_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z352_half;
extern const BG2_adj_desc_t BG2_adj_desc_Z384_half;

extern const BG2_adj_desc_t BG2_adj_desc_Z32_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z36_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z40_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z44_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z48_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z52_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z56_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z60_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z64_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z72_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z80_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z88_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z96_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z104_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z112_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z120_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z128_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z144_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z160_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z176_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z192_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z208_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z224_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z240_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z256_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z288_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z320_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z352_half2;
extern const BG2_adj_desc_t BG2_adj_desc_Z384_half2;

template <typename T, int BG> const BG_adj_desc<BG>* get_adj_BG_desc(int Z);
template <typename T, int BG> const BG_adj_desc<BG>* get_adj_BG_desc_small(int Z);

template <>
inline
const BG_adj_desc<1>* get_adj_BG_desc<__half, 1>(int Z)
{
    const BG1_adj_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG1_adj_desc_Z32_half;  break;
    case 36:  bgd = &BG1_adj_desc_Z36_half;  break;
    case 40:  bgd = &BG1_adj_desc_Z40_half;  break;
    case 44:  bgd = &BG1_adj_desc_Z44_half;  break;
    case 48:  bgd = &BG1_adj_desc_Z48_half;  break;
    case 52:  bgd = &BG1_adj_desc_Z52_half;  break;
    case 56:  bgd = &BG1_adj_desc_Z56_half;  break;
    case 60:  bgd = &BG1_adj_desc_Z60_half;  break;
    case 64:  bgd = &BG1_adj_desc_Z64_half;  break;
    case 72:  bgd = &BG1_adj_desc_Z72_half;  break;
    case 80:  bgd = &BG1_adj_desc_Z80_half;  break;
    case 88:  bgd = &BG1_adj_desc_Z88_half;  break;
    case 96:  bgd = &BG1_adj_desc_Z96_half;  break;
    case 104: bgd = &BG1_adj_desc_Z104_half; break;
    case 112: bgd = &BG1_adj_desc_Z112_half; break;
    case 120: bgd = &BG1_adj_desc_Z120_half; break;
    case 128: bgd = &BG1_adj_desc_Z128_half; break;
    case 144: bgd = &BG1_adj_desc_Z144_half; break;
    case 160: bgd = &BG1_adj_desc_Z160_half; break;
    case 176: bgd = &BG1_adj_desc_Z176_half; break;
    case 192: bgd = &BG1_adj_desc_Z192_half; break;
    case 208: bgd = &BG1_adj_desc_Z208_half; break;
    case 224: bgd = &BG1_adj_desc_Z224_half; break;
    case 240: bgd = &BG1_adj_desc_Z240_half; break;
    case 256: bgd = &BG1_adj_desc_Z256_half; break;
    case 288: bgd = &BG1_adj_desc_Z288_half; break;
    case 320: bgd = &BG1_adj_desc_Z320_half; break;
    case 352: bgd = &BG1_adj_desc_Z352_half; break;
    case 384: bgd = &BG1_adj_desc_Z384_half; break;
    }
    return bgd;
}

template <>
inline
const BG_adj_desc<1>* get_adj_BG_desc_small<__half, 1>(int Z)
{
    const BG1_adj_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 2:   bgd = &BG1_adj_desc_Z2_half;   break;
    case 3:   bgd = &BG1_adj_desc_Z3_half;   break;
    case 4:   bgd = &BG1_adj_desc_Z4_half;   break;
    case 5:   bgd = &BG1_adj_desc_Z5_half;   break;
    case 6:   bgd = &BG1_adj_desc_Z6_half;   break;
    case 7:   bgd = &BG1_adj_desc_Z7_half;   break;
    case 8:   bgd = &BG1_adj_desc_Z8_half;   break;
    case 9:   bgd = &BG1_adj_desc_Z9_half;   break;
    case 10:  bgd = &BG1_adj_desc_Z10_half;  break;
    case 11:  bgd = &BG1_adj_desc_Z11_half;  break;
    case 12:  bgd = &BG1_adj_desc_Z12_half;  break;
    case 13:  bgd = &BG1_adj_desc_Z13_half;  break;
    case 14:  bgd = &BG1_adj_desc_Z14_half;  break;
    case 15:  bgd = &BG1_adj_desc_Z15_half;  break;
    case 16:  bgd = &BG1_adj_desc_Z16_half;  break;
    case 18:  bgd = &BG1_adj_desc_Z18_half;  break;
    case 20:  bgd = &BG1_adj_desc_Z20_half;  break;
    case 22:  bgd = &BG1_adj_desc_Z22_half;  break;
    case 24:  bgd = &BG1_adj_desc_Z24_half;  break;
    case 26:  bgd = &BG1_adj_desc_Z26_half;  break;
    case 28:  bgd = &BG1_adj_desc_Z28_half;  break;
    case 30:  bgd = &BG1_adj_desc_Z30_half;  break;
    }
    return bgd;
}

template <>
inline
const BG_adj_desc<1>* get_adj_BG_desc<__half2, 1>(int Z)
{
    const BG1_adj_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG1_adj_desc_Z32_half2;  break;
    case 36:  bgd = &BG1_adj_desc_Z36_half2;  break;
    case 40:  bgd = &BG1_adj_desc_Z40_half2;  break;
    case 44:  bgd = &BG1_adj_desc_Z44_half2;  break;
    case 48:  bgd = &BG1_adj_desc_Z48_half2;  break;
    case 52:  bgd = &BG1_adj_desc_Z52_half2;  break;
    case 56:  bgd = &BG1_adj_desc_Z56_half2;  break;
    case 60:  bgd = &BG1_adj_desc_Z60_half2;  break;
    case 64:  bgd = &BG1_adj_desc_Z64_half2;  break;
    case 72:  bgd = &BG1_adj_desc_Z72_half2;  break;
    case 80:  bgd = &BG1_adj_desc_Z80_half2;  break;
    case 88:  bgd = &BG1_adj_desc_Z88_half2;  break;
    case 96:  bgd = &BG1_adj_desc_Z96_half2;  break;
    case 104: bgd = &BG1_adj_desc_Z104_half2; break;
    case 112: bgd = &BG1_adj_desc_Z112_half2; break;
    case 120: bgd = &BG1_adj_desc_Z120_half2; break;
    case 128: bgd = &BG1_adj_desc_Z128_half2; break;
    case 144: bgd = &BG1_adj_desc_Z144_half2; break;
    case 160: bgd = &BG1_adj_desc_Z160_half2; break;
    case 176: bgd = &BG1_adj_desc_Z176_half2; break;
    case 192: bgd = &BG1_adj_desc_Z192_half2; break;
    case 208: bgd = &BG1_adj_desc_Z208_half2; break;
    case 224: bgd = &BG1_adj_desc_Z224_half2; break;
    case 240: bgd = &BG1_adj_desc_Z240_half2; break;
    case 256: bgd = &BG1_adj_desc_Z256_half2; break;
    case 288: bgd = &BG1_adj_desc_Z288_half2; break;
    case 320: bgd = &BG1_adj_desc_Z320_half2; break;
    case 352: bgd = &BG1_adj_desc_Z352_half2; break;
    case 384: bgd = &BG1_adj_desc_Z384_half2; break;
    }
    return bgd;
}

template <>
inline
const BG_adj_desc<2>* get_adj_BG_desc<__half, 2>(int Z)
{
    const BG2_adj_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG2_adj_desc_Z32_half;  break;
    case 36:  bgd = &BG2_adj_desc_Z36_half;  break;
    case 40:  bgd = &BG2_adj_desc_Z40_half;  break;
    case 44:  bgd = &BG2_adj_desc_Z44_half;  break;
    case 48:  bgd = &BG2_adj_desc_Z48_half;  break;
    case 52:  bgd = &BG2_adj_desc_Z52_half;  break;
    case 56:  bgd = &BG2_adj_desc_Z56_half;  break;
    case 60:  bgd = &BG2_adj_desc_Z60_half;  break;
    case 64:  bgd = &BG2_adj_desc_Z64_half;  break;
    case 72:  bgd = &BG2_adj_desc_Z72_half;  break;
    case 80:  bgd = &BG2_adj_desc_Z80_half;  break;
    case 88:  bgd = &BG2_adj_desc_Z88_half;  break;
    case 96:  bgd = &BG2_adj_desc_Z96_half;  break;
    case 104: bgd = &BG2_adj_desc_Z104_half; break;
    case 112: bgd = &BG2_adj_desc_Z112_half; break;
    case 120: bgd = &BG2_adj_desc_Z120_half; break;
    case 128: bgd = &BG2_adj_desc_Z128_half; break;
    case 144: bgd = &BG2_adj_desc_Z144_half; break;
    case 160: bgd = &BG2_adj_desc_Z160_half; break;
    case 176: bgd = &BG2_adj_desc_Z176_half; break;
    case 192: bgd = &BG2_adj_desc_Z192_half; break;
    case 208: bgd = &BG2_adj_desc_Z208_half; break;
    case 224: bgd = &BG2_adj_desc_Z224_half; break;
    case 240: bgd = &BG2_adj_desc_Z240_half; break;
    case 256: bgd = &BG2_adj_desc_Z256_half; break;
    case 288: bgd = &BG2_adj_desc_Z288_half; break;
    case 320: bgd = &BG2_adj_desc_Z320_half; break;
    case 352: bgd = &BG2_adj_desc_Z352_half; break;
    case 384: bgd = &BG2_adj_desc_Z384_half; break;
    }
    return bgd;
}

template <>
inline
const BG_adj_desc<2>* get_adj_BG_desc_small<__half, 2>(int Z)
{
    const BG2_adj_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 2:   bgd = &BG2_adj_desc_Z2_half;   break;
    case 3:   bgd = &BG2_adj_desc_Z3_half;   break;
    case 4:   bgd = &BG2_adj_desc_Z4_half;   break;
    case 5:   bgd = &BG2_adj_desc_Z5_half;   break;
    case 6:   bgd = &BG2_adj_desc_Z6_half;   break;
    case 7:   bgd = &BG2_adj_desc_Z7_half;   break;
    case 8:   bgd = &BG2_adj_desc_Z8_half;   break;
    case 9:   bgd = &BG2_adj_desc_Z9_half;   break;
    case 10:  bgd = &BG2_adj_desc_Z10_half;  break;
    case 11:  bgd = &BG2_adj_desc_Z11_half;  break;
    case 12:  bgd = &BG2_adj_desc_Z12_half;  break;
    case 13:  bgd = &BG2_adj_desc_Z13_half;  break;
    case 14:  bgd = &BG2_adj_desc_Z14_half;  break;
    case 15:  bgd = &BG2_adj_desc_Z15_half;  break;
    case 16:  bgd = &BG2_adj_desc_Z16_half;  break;
    case 18:  bgd = &BG2_adj_desc_Z18_half;  break;
    case 20:  bgd = &BG2_adj_desc_Z20_half;  break;
    case 22:  bgd = &BG2_adj_desc_Z22_half;  break;
    case 24:  bgd = &BG2_adj_desc_Z24_half;  break;
    case 26:  bgd = &BG2_adj_desc_Z26_half;  break;
    case 28:  bgd = &BG2_adj_desc_Z28_half;  break;
    case 30:  bgd = &BG2_adj_desc_Z30_half;  break;
    }
    return bgd;
}

template <>
inline
const BG_adj_desc<2>* get_adj_BG_desc<__half2, 2>(int Z)
{
    const BG2_adj_desc_t* bgd = nullptr;
    switch(Z)
    {
    case 32:  bgd = &BG2_adj_desc_Z32_half2;  break;
    case 36:  bgd = &BG2_adj_desc_Z36_half2;  break;
    case 40:  bgd = &BG2_adj_desc_Z40_half2;  break;
    case 44:  bgd = &BG2_adj_desc_Z44_half2;  break;
    case 48:  bgd = &BG2_adj_desc_Z48_half2;  break;
    case 52:  bgd = &BG2_adj_desc_Z52_half2;  break;
    case 56:  bgd = &BG2_adj_desc_Z56_half2;  break;
    case 60:  bgd = &BG2_adj_desc_Z60_half2;  break;
    case 64:  bgd = &BG2_adj_desc_Z64_half2;  break;
    case 72:  bgd = &BG2_adj_desc_Z72_half2;  break;
    case 80:  bgd = &BG2_adj_desc_Z80_half2;  break;
    case 88:  bgd = &BG2_adj_desc_Z88_half2;  break;
    case 96:  bgd = &BG2_adj_desc_Z96_half2;  break;
    case 104: bgd = &BG2_adj_desc_Z104_half2; break;
    case 112: bgd = &BG2_adj_desc_Z112_half2; break;
    case 120: bgd = &BG2_adj_desc_Z120_half2; break;
    case 128: bgd = &BG2_adj_desc_Z128_half2; break;
    case 144: bgd = &BG2_adj_desc_Z144_half2; break;
    case 160: bgd = &BG2_adj_desc_Z160_half2; break;
    case 176: bgd = &BG2_adj_desc_Z176_half2; break;
    case 192: bgd = &BG2_adj_desc_Z192_half2; break;
    case 208: bgd = &BG2_adj_desc_Z208_half2; break;
    case 224: bgd = &BG2_adj_desc_Z224_half2; break;
    case 240: bgd = &BG2_adj_desc_Z240_half2; break;
    case 256: bgd = &BG2_adj_desc_Z256_half2; break;
    case 288: bgd = &BG2_adj_desc_Z288_half2; break;
    case 320: bgd = &BG2_adj_desc_Z320_half2; break;
    case 352: bgd = &BG2_adj_desc_Z352_half2; break;
    case 384: bgd = &BG2_adj_desc_Z384_half2; break;
    }
    return bgd;
}

} // namespace ldpc2


#endif // !defined(LDPC2_BG_DESC_HPP_INCLUDED_)
