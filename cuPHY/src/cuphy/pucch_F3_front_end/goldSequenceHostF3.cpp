/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(SEQUENCE_CUH_CPU_F3_INCLUDED_)
#define SEQUENCE_CUH_CPU_F3_INCLUDED_

#include "cuphy.h"
#include "GOLD_2_32_P_LUT_CPU.h"
#include "GOLD_1_SEQ_LUT_CPU.h"




namespace seqCpuF3
{
const uint32_t     POLY_2                            = 0x8000000F;
const uint32_t     POLY_2_GMASK                      = 0x0000000F;

const uint32_t     BITS_PROCESSED_PER_LUT_ENTRY      = 32;
const uint32_t     BITS_PROCESSED_PER_LUT_ENTRY_MASK = (1UL << BITS_PROCESSED_PER_LUT_ENTRY) - 1;
constexpr uint32_t WORD_SIZE                         = sizeof(uint32_t) * 8;



uint32_t bitRev(uint32_t x)
{
    x = ((x >> 1) & 0x55555555u) | ((x & 0x55555555u) << 1);
    x = ((x >> 2) & 0x33333333u) | ((x & 0x33333333u) << 2);
    x = ((x >> 4) & 0x0f0f0f0fu) | ((x & 0x0f0f0f0fu) << 4);
    x = ((x >> 8) & 0x00ff00ffu) | ((x & 0x00ff00ffu) << 8);
    x = ((x >> 16) & 0xffffu) | ((x & 0xffffu) << 16);
    return x;
 
	return 0;
}






// Fibonacci LFSR for second polynomial for Gold sequence generation
CUDA_BOTH inline uint32_t fibonacciLFSR2_CPU(uint32_t& state, uint32_t n, uint32_t resInit = 0)
{
    uint32_t res = resInit;
    // x^{31} + x^3 + x^2 + x + 1
    for(int i = 0; i < n; i++)
    {
        uint32_t bit = (state) ^ ((state >> 1)) ^ ((state >> 2)) ^ (state >> 3);
        bit          = bit & 1;
        res >>= 1;
        res ^= (state & 1) << 31; //(% 32);
        state >>= 1;
        state ^= (bit << 30);
    }
    return res;
}
// Fibonacci LFSR for second polynomial for Gold sequence generation
CUDA_BOTH inline uint32_t fibonacciLFSR2_1bit_CPU(uint32_t& state)
{
    uint32_t res = state;
    // x^{31} + x^3 + x^2 + x + 1
    uint32_t bit = (state) ^ ((state >> 1)) ^ ((state >> 2)) ^ (state >> 3);
    bit          = bit & 1;
    state >>= 1;
    state ^= (bit << 30);
    res ^= (state >> 30) << 31;
    return res;
}
// Fibonacci LFSR for first polynomial for Gold sequence generation
// inverted output
CUDA_BOTH inline uint32_t fibonacciLFSR1_CPU(uint32_t& state, uint32_t n)
{
    uint32_t res = 0;

    // x^{31} + x^3 + 1
    for(int i = 0; i < n; i++)
    {
        uint32_t bit = (state) ^ (state >> 3);
        bit          = bit & 1;
        res >>= 1;
        res ^= (state & 1) << 31; //(% 32);
        state >>= 1;
        state ^= (bit << 30);
    }
    return res;
}
// Galois LFSR for Gold sequence generation
// computes between 1 and 32 bits, n must be at most 32
CUDA_BOTH inline uint32_t galois31LFSRWord_CPU(uint32_t state, uint32_t galoisMask, uint32_t n = 31)
{
    uint32_t res = 0;

    uint32_t msbMask = (1 << 30);
    uint32_t bit;
    uint32_t pred;
#pragma unroll
    for(int i = 0; i < n; i++)
    {
        bit  = (msbMask & state);
        pred = bit != 0;
        state <<= 1;
        state ^= pred * galoisMask;
        res ^= pred << i;
    }
    return res;
}

CUDA_BOTH inline uint32_t polyMulHigh31_CPU(uint32_t a, uint32_t b)
{
    uint32_t prodHi = 0;
#pragma unroll
    for(int i = 1; i < 32; i++)
    {
        uint32_t pred = ((b >> i) & 1);
        prodHi ^= (pred * a) >> (31 - i);
    }
    return prodHi;
}
/*
CUDA_BOTH inline uint32_t mulModPoly31(uint32_t a,
                                       uint32_t pow,
                                       uint32_t poly)
{
    // a moduloe POLY_2, 31 BITs
    uint32_t crc = a ^ (a >= poly) * poly;
    uint32_t r = 1;
    uint32_t y = crc;
    while(pow > 1)
    {
        if(pow & 1)
            r = mulModPoly<uint32_t, 31>(r, y, poly);
        y = mulModPoly<uint32_t, 31>(y, y, poly);
        pow >>= 1;
    }

    return mulModPoly<uint32_t, 31>(r, y, poly);
}
*/

CUDA_BOTH inline uint32_t mulModPoly31LUT_CPU(uint32_t a,
                                              uint32_t b,
                                              uint32_t poly)
{
    uint32_t prod = 0;
    // a moduloe POLY_2, 31 BITs
    uint32_t crc = a ^ (a >= POLY_2) * POLY_2;
#pragma unroll
    for(int i = 0; i < 31; i++)
    {
        prod ^= (crc & 1) * b;
        b = (b << 1) ^ (b & (1 << (30)) ? poly : 0);
        crc >>= 1;
    }

    return prod;
}

// Little-endian 31-bit Modular GF2 polynomial multiplication by monomials
// using coalesced precomputed x^{32i}, x^{32i +8}, x^{32i + 16}, x^{32i + 24}
// values
CUDA_BOTH inline uint32_t mulModPoly31_Coalesced_CPU(const uint32_t  a,
                                                     const uint32_t* table,
                                                     uint32_t        tableWordOffset,
                                                     uint32_t        poly)
{
    uint32_t     prod    = 0;
    uint32_t     msbMask = (1UL << 31);
    unsigned int offset  = 0;

#pragma unroll
    for(int bitsProcessed = 0; bitsProcessed < sizeof(uint32_t) * 8; bitsProcessed += BITS_PROCESSED_PER_LUT_ENTRY)
    {
        uint32_t inputByte = a >> (bitsProcessed)&BITS_PROCESSED_PER_LUT_ENTRY_MASK;
        for(unsigned bit = 0; bit < BITS_PROCESSED_PER_LUT_ENTRY; bit++)
        {
            uint32_t pred  = ((inputByte >> (bit)) & 1);
            uint32_t pprod = table[(offset)] * pred;
            for(unsigned shift = 0; shift < bit; shift++)
            {
                pprod <<= 1;
                uint32_t pred = (pprod & msbMask) == 0;
                pprod ^= (poly * pred);
            }
            prod ^= pprod;
        }
        offset += tableWordOffset;
    }

    return prod;
}


// Compute 32 bits of the Gold sequence starting from bit n
__host__  inline uint32_t gold32n_CPU(uint32_t seed2, uint32_t n)
{
    uint32_t prod2;

    //    uint32_t state1 = 0x40000000;         // reverse of 0x1
    uint32_t state2 = bitRev(seed2) >> 1; // reverse 31 bits

    state2 = polyMulHigh31_CPU(state2, POLY_2);
    prod2  = mulModPoly31LUT_CPU(state2,
                            GOLD_2_32_P_LUT_CPU[(n) / WORD_SIZE],
                            POLY_2);

    uint32_t fstate2 = galois31LFSRWord_CPU(prod2, POLY_2_GMASK, 31);

    uint32_t output2 = fibonacciLFSR2_CPU(fstate2, 32);
    output2          = (output2 >> (n % 32));
    output2 |= (n % 32) ? (fstate2 << (32 - (n % 32))) : 0;

    uint32_t fstate1 = GOLD_1_SEQ_LUT_CPU[n / WORD_SIZE] & 0x7FFFFFFF;
    uint32_t seq1f   = fibonacciLFSR1_CPU(fstate1, 32);
    seq1f            = (seq1f >> (n % 32));
    seq1f |= (n % 32) ? (fstate1 << (32 - (n % 32))) : 0;

    return seq1f ^ output2;
}

} // namespace descrambling

#endif
