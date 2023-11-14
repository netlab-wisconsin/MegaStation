/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(SEQUENCE_HPP_INCLUDED_)
#define SEQUENCE_HPP_INCLUDED_
#include <stdint.h>

#define MAX_INPUT_TB_SIZE_IN_WORDS ((MAX_ENCODED_CODE_BLOCK_BIT_SIZE * MAX_N_CBS_PER_TB_SUPPORTED) / 32)

namespace descrambling
{
const uint32_t     BITS_PROCESSED_PER_LUT_ENTRY      = 32;
const uint32_t     BITS_PROCESSED_PER_LUT_ENTRY_MASK = (1UL << BITS_PROCESSED_PER_LUT_ENTRY) - 1;
const uint32_t     POLY_1                            = 0x80000009;
const uint32_t     POLY_2                            = 0x8000000F;
const uint32_t     POLY_1_GMASK                      = 0x00000009;
const uint32_t     POLY_2_GMASK                      = 0x0000000F;
const uint32_t     GLOBAL_BLOCK_SIZE                 = 512;
const uint32_t     WARP_SIZE                         = 32;
constexpr uint32_t WORD_SIZE                         = sizeof(uint32_t) * 8;
// Nc value from 5G spec: number of bits skipped for both LFSRs
const uint32_t NC = 1600;
} // namespace descrambling
#endif
