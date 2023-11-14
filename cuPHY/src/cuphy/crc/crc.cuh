/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(CRC_CUH_INCLUDED_)
#define CRC_CUH_INCLUDED_

#include "crc.hpp"
#include "cuphy.h"
#include "cuphy_internal.h"

#define TABLE 1

namespace crc
{
/// swap endianess
template <int bitsize>
static __device__ inline uint32_t swap(uint32_t val)
{
    int size = bitsize >> 3;
    switch(size)
    {
    case 4:
	  return __byte_perm(val, 0, 0x0123);
    case 3:
        return ((val & 0xFF) << 16) + (val & 0xFF00) + ((val & 0xFF0000) >> 16);
    case 2:
        return ((val & 0xFF) << 8) + ((val & 0xFF00) >> 8);
    default:
        return val;
    }
}

const int LUT_SIZE = 256;
// CRC polynomial type uintCRC_t has to be an unsigned int type and
// uintCRCBitLength at most the bit size of the unsigned int type stride is in
// bytes

// CRC polynomial type uintCRC_t has to be an unsigned int type and
// uintCRCBitLength at most the bit size of the unsigned int type stride is in
// bytes

#define FULL_MASK 0xffffffff

// XOR warp level reduction
template <typename uintCRC_t>
__inline__ __device__ uintCRC_t warpReduceSum(uintCRC_t val)
{
    for(int offset = WARP_SIZE / 2; offset > 0; offset /= 2)
        val ^= __shfl_down_sync(FULL_MASK, val, offset, WARP_SIZE);
    return val;
}

// Shared memory XOR reduction
template <typename uintCRC_t>
__device__ inline uintCRC_t xorReductionWarpShared(uintCRC_t  input,
                                                   uintCRC_t* shared)
{
    int lane = threadIdx.x % WARP_SIZE;
    int wid  = threadIdx.x / WARP_SIZE;

    input =
        warpReduceSum<uintCRC_t>(input); // Each warp performs partial reduction

    if(lane == 0) shared[wid] = input; // Write reduced value to shared memory

    __syncthreads(); // Wait for all partial reductions

    // read from shared memory only if that warp existed
    input = (threadIdx.x < blockDim.x / WARP_SIZE) ? shared[lane] : 0;

    if(wid == 0)
        input = warpReduceSum<uintCRC_t>(input); // Final reduce within first warp

    return input;
}

template <typename T, uint32_t size>
__device__ T mulModPoly(T a, T b, T poly)
{
    T prod;
#pragma unroll
    for(int i = 0; i < size; i++)
    {
        prod ^= (b & 1) ? a : 0;
        a = (a << 1) ^ ((a & (1 << (size - 1))) ? poly : 0);
        b >>= 1;
    }
    return prod;
}

template <typename T, uint32_t size>
__device__ uint32_t mulModCRCPolyLUT(uint32_t a,
                                     T        b,
                                     uint32_t LUT[4][256],
                                     T        poly)
{
    T prod = 0;
    T crc  = LUT[3][static_cast<uint8_t>(a)] ^ LUT[2][static_cast<uint8_t>(a >> 8)] ^ LUT[1][static_cast<uint8_t>(a >> 16)] ^ LUT[0][static_cast<uint8_t>(a >> 24)];

/*	uint32_t shift = 1;
	for(int i = 0; i < 32; i++)
	{
	         if(pow & 1)
                 shift = mulModPoly<T, size>(shift, b, poly);
		 shift = mulModPoly<T, size>(shift, shift, poly);
		 pow >>= 1;
}
*/
#pragma unroll
    for(int i = 0; i < size; i++)
    {
        prod ^= (crc & 1) ? b : 0;
        b = (b << 1) ^ (b & (1 << (size - 1)) ? poly : 0);
        crc >>= 1;
    }
    return prod;
}

// Big-endian 32-bit Modular GF2 polynomial multiplication by monomials
// using coalesced precomputed x^{32i}, x^{32i +8}, x^{32i + 16}, x^{32i + 24}
// values
template <typename uintCRC_t, int uintCRCBitLength>
CUDA_BOTH uintCRC_t mulModCRCPoly32_1Coalesced(const uint32_t   a,
                                               const uintCRC_t* table,
                                               uint32_t         tableCoalescedOffset,
                                               uintCRC_t        poly)
{
    uintCRC_t    prod    = 0;
    uintCRC_t    msbMask = (1 << (uintCRCBitLength - 1));
    unsigned int offset  = 0;
    // need to take care of little-endiannes within a byte
    uint32_t step = BITS_PROCESSED_PER_LUT_ENTRY > 8 ? BITS_PROCESSED_PER_LUT_ENTRY : 8;
    for(int bitsProcessed = sizeof(uint32_t) * 8 - step; bitsProcessed >= 0; bitsProcessed -= step)
    {
        // take care of byte level little-endiannes for BIT windows smaller than 8-bits
        for(int bStep = bitsProcessed; bStep < bitsProcessed + 8; bStep += BITS_PROCESSED_PER_LUT_ENTRY)
        {
            uint8_t inputByte = a >> (bStep);
            for(unsigned bit = 0; bit < BITS_PROCESSED_PER_LUT_ENTRY; bit++)
            {
                uintCRC_t pred  = ((inputByte >> bit) & 1);
                uintCRC_t pprod = table[(offset)] * pred;
                for(unsigned shift = 0; shift < bit; shift++)
                {
                    uintCRC_t pred = (pprod & msbMask) != 0;
                    pprod <<= 1;
                    pprod ^= (poly * pred);
                }
                prod ^= pprod;
            }
            offset += tableCoalescedOffset;
        }
    }
    return prod;
}

// Little-endian 32-bit Modular GF2 polynomial multiplication
template <typename uintCRC_t, int uintCRCBitLength>
CUDA_BOTH uintCRC_t mulModCRCPoly32LR(const uint32_t  a,
                                      const uintCRC_t b,
                                      uintCRC_t       poly,
                                      int             msb = 3)
{
    uintCRC_t prod        = 0;
    uintCRC_t msbMask     = (1 << (uintCRCBitLength));
    uintCRC_t allOnesMask = -1;
    allOnesMask >>= (sizeof(uintCRC_t) * 8 - uintCRCBitLength - 1);
    for(int byte = 0; byte < msb; byte++)
    {
        uint8_t inputByte = a >> (8 * (byte));
        for(int bit = 0; bit < 8; bit++)
        {
            uintCRC_t pred  = ((inputByte >> bit) & 1) == 0;
            uintCRC_t pprod = (b & (pred + allOnesMask));
            int       s     = (byte)*8;
            for(int shift = 0; shift < s + bit; shift++)
            {
                pprod <<= 1;
                uintCRC_t pred = (pprod & msbMask) == 0;
                pprod ^= (poly & (pred + allOnesMask));
            }
            prod ^= pprod;
        }
    }
    return prod;
}

// Little-endian 32-bit Modular GF2 polynomial multiplication by monomials using
// coalesced precomputed x^{32i}, x^{32i + 8}, x^{32i + 16}, x^{32i + 24} values
template <typename uintCRC_t, int uintCRCBitLength>
CUDA_BOTH uintCRC_t mulModCRCPoly32_1LR(const uint32_t   a,
                                        const uintCRC_t* table,
                                        uintCRC_t        poly,
                                        int              msB = 3)
{
    uintCRC_t prod        = 0;
    uintCRC_t msbMask     = (1 << (uintCRCBitLength - 1));
    uintCRC_t allOnesMask = -1;
    allOnesMask >>= (sizeof(uintCRC_t) * 8 - uintCRCBitLength - 1);
    int byte = 0;
    for(int bitsProcessed = 0; bitsProcessed < msB * 8; bitsProcessed += BITS_PROCESSED_PER_LUT_ENTRY)
    {
        uint8_t inputByte = a >> (bitsProcessed);
        for(int bit = 0; bit < BITS_PROCESSED_PER_LUT_ENTRY; bit++)
        {
            uintCRC_t pred  = ((inputByte >> bit) & 1);
            uintCRC_t pprod = table[(byte)] * pred;
            for(int shift = 0; shift < bit; shift++)
            {
                uintCRC_t pred = (pprod & msbMask) != 0;
                pprod <<= 1;
                pprod ^= (poly * pred);
            }
            prod ^= pprod;
        }
        byte++;
        // printf("prod:%x \n", prod);
    }
    return prod;
}

// Big-endian 32-bit Modular GF2 polynomial multiplication by monomials
// using contiguous precomputed x^{32i}, x^{33i}, x^{34i}, x^{35i} values
template <typename uintCRC_t, int uintCRCBitLength>
__device__ uintCRC_t mulModCRCPoly32_1(const uint32_t   a,
                                       const uintCRC_t* table,
                                       uintCRC_t        poly,
                                       int              msB = 3)
{
    uintCRC_t prod        = 0;
    uintCRC_t msbMask     = (1 << (uintCRCBitLength - 1));
    uintCRC_t allOnesMask = -1;
    allOnesMask >>= (sizeof(uintCRC_t) * 8 - uintCRCBitLength - 1);

    for(int byte = msB; byte >= 0; byte--)
    {
        uint8_t inputByte = a >> (8 * byte);
        for(int bit = 0; bit < 8; bit++)
        {
            uintCRC_t pred  = ((inputByte >> bit) & 1) == 0;
            uintCRC_t pprod = (table[(3 - byte)] & (pred + allOnesMask));
            for(int shift = 0; shift < bit; shift++)
            {
                uintCRC_t pred = (pprod & msbMask) == 0;
                pprod <<= 1;
                pprod ^= (poly & (pred + allOnesMask));
            }
            prod ^= pprod;
        }
    }
    return prod;
}
} // namespace crc
#endif
