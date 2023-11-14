/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#pragma once

// Pack the compressed data ("compbits" bits per value), with 4 (I,Q) pairs per thread.
// Each thread writes 4 IQ pairs into shared memory.

// If only certain values are used (e.g. 9, 14, 16), the other tests could be optimized
// out using a compile-time macro, to save SASS footprint.

__device__ inline void packPRB(
    uint8_t*       sm,        // Shared memory pointer for the current PRB
    const int32_t& tx_prb,    // Sub-PRB id inside a subgroup of threads working on the same PRB
    const int32_t  vi[4],     // I values
    const int32_t  vq[4],     // Q values
    const int32_t& compParam, // Compression parameter
    const uint8_t& compbits)  // Number of compressed bits. 16 means no compression
{
    int offset = tx_prb * compbits;
    // Compression parameter. Don't write if no compression is used.
    if(tx_prb == 0 && compbits < 16)
        sm[offset] = compParam;

    // Write the data in shared memory.
    if(compbits == 9)
    {
        sm[offset + 1] = (vi[0] >> 1);                         // 8, remains 1
        sm[offset + 2] = (vi[0] << 7) | ((vq[0] >> 2) & 0x7f); // 1 + 7, remains 2
        sm[offset + 3] = (vq[0] << 6) | ((vi[1] >> 3) & 0x3f); // 2 + 6, remains 3
        sm[offset + 4] = (vi[1] << 5) | ((vq[1] >> 4) & 0x1f); // 3 + 5, remains 4
        sm[offset + 5] = (vq[1] << 4) | ((vi[2] >> 5) & 0x0f); // 4 + 4, remains 5
        sm[offset + 6] = (vi[2] << 3) | ((vq[2] >> 6) & 0x07); // 5 + 3, remains 6
        sm[offset + 7] = (vq[2] << 2) | ((vi[3] >> 7) & 0x03); // 6 + 2, remains 7
        sm[offset + 8] = (vi[3] << 1) | ((vq[3] >> 8) & 0x01); // 7 + 1, remains 8
        sm[offset + 9] = vq[3];                                // 8
    }
    else if(compbits == 14)
    {
        sm[offset + 1]  = (vi[0] >> 6);                          // 8, remains 6
        sm[offset + 2]  = (vi[0] << 2) | ((vq[0] >> 12) & 0x03); // 6 + 2, remains 12
        sm[offset + 3]  = (vq[0] >> 4);                          // 8, remains 4
        sm[offset + 4]  = (vq[0] << 4) | ((vi[1] >> 10) & 0x0f); // 4 + 4, remains 10
        sm[offset + 5]  = (vi[1] >> 2);                          // 8, remains 2
        sm[offset + 6]  = (vi[1] << 6) | ((vq[1] >> 8) & 0x3f);  // 2 + 6, Remains 8
        sm[offset + 7]  = vq[1];                                 // 8, remains 0
        sm[offset + 8]  = (vi[2] >> 6);                          // 8, remains 6
        sm[offset + 9]  = (vi[2] << 2) | ((vq[2] >> 12) & 0x03); // 6 + 2, remains 12
        sm[offset + 10] = (vq[2] >> 4);                          // 8, remains 4
        sm[offset + 11] = (vq[2] << 4) | ((vi[3] >> 10) & 0x0f); // 4 + 4, remains 10
        sm[offset + 12] = (vi[3] >> 2);                          // 8, remains 2
        sm[offset + 13] = (vi[3] << 6) | ((vq[3] >> 8) & 0x3f);  // 2 + 6, remains 8
        sm[offset + 14] = vq[3];
    }
    else if(compbits == 7)
    {
        sm[offset + 1] = (vi[0] << 1) | ((vq[0] >> 6) & 0x01); // 7 + 1, remains 6
        sm[offset + 2] = (vq[0] << 2) | ((vi[1] >> 5) & 0x03); // 6 + 2, remains 5
        sm[offset + 3] = (vi[1] << 3) | ((vq[1] >> 4) & 0x07); // 5 + 3, remains 4
        sm[offset + 4] = (vq[1] << 4) | ((vi[2] >> 3) & 0x0f); // 4 + 4, remains 3
        sm[offset + 5] = (vi[2] << 5) | ((vq[2] >> 2) & 0x1f); // 3 + 5, remains 2
        sm[offset + 6] = (vq[2] << 6) | ((vi[3] >> 1) & 0x3f); // 2 + 6, remains 1
        sm[offset + 7] = (vi[3] << 7) | (vq[3] & 0x7f);        // 1 + 7
    }
    else if(compbits == 8)
    {
        for(int i = 0; i < 4; i++)
        {
            sm[offset + 2 * i + 1] = vi[i];
            sm[offset + 2 * i + 2] = vq[i];
        }
    }
    else if(compbits == 10)
    {
        sm[offset + 1]  = (vi[0] >> 2);                         // 8, remains 2
        sm[offset + 2]  = (vi[0] << 6) | ((vq[0] >> 4) & 0x3f); // 2 + 6, remains 4
        sm[offset + 3]  = (vq[0] << 4) | ((vi[1] >> 6) & 0x0f); // 4 + 4, remains 6
        sm[offset + 4]  = (vi[1] << 2) | ((vq[1] >> 8) & 0x03); // 6 + 2, remains 8
        sm[offset + 5]  = vq[1];                                // 8
        sm[offset + 6]  = (vi[2] >> 2);                         // 8, remains 2
        sm[offset + 7]  = (vi[2] << 6) | ((vq[2] >> 4) & 0x3f); // 2 + 6, remains 4
        sm[offset + 8]  = (vq[2] << 4) | ((vi[3] >> 6) & 0x0f); // 4 + 4, remains 6
        sm[offset + 9]  = (vi[3] << 2) | ((vq[3] >> 8) & 0x03); // 6 + 2, remains 8
        sm[offset + 10] = vq[3];                                // 8
    }
    else if(compbits == 11)
    {
        sm[offset + 1]  = (vi[0] >> 3);                          // 8, remains 3
        sm[offset + 2]  = (vi[0] << 5) | ((vq[0] >> 6) & 0x1f);  // 3 + 5, remains 6
        sm[offset + 3]  = (vq[0] << 2) | ((vi[1] >> 9) & 0x03);  // 6 + 2, remains 9
        sm[offset + 4]  = (vi[1] >> 1);                          // 8, remains 1
        sm[offset + 5]  = (vi[1] << 7) | ((vq[1] >> 4) & 0x7f);  // 1 + 7, remains 4
        sm[offset + 6]  = (vq[1] << 4) | ((vi[2] >> 7) & 0x0f);  // 4 + 4, remains 7
        sm[offset + 7]  = (vi[2] << 1) | ((vq[2] >> 10) & 0x01); // 7 + 1, remains 10
        sm[offset + 8]  = (vq[2] >> 2);                          // 8, remains 2
        sm[offset + 9]  = (vq[2] << 6) | ((vi[3] >> 5) & 0x3f);  // 2 + 6, remains 5
        sm[offset + 10] = (vi[3] << 3) | ((vq[3] >> 8) & 0x07);  // 5 + 3, remains 8
        sm[offset + 11] = vq[3];                                 // 8
    }
    else if(compbits == 12)
    {
        for(int i = 0; i < 4; i++)
        {
            sm[offset + 3 * i + 1] = vi[i] >> 4;
            sm[offset + 3 * i + 2] = (vi[i] << 4) | ((vq[i] >> 8) & 0x0f);
            sm[offset + 3 * i + 3] = vq[i];
        }
    }
    else if(compbits == 13)
    {
        sm[offset + 1]  = (vi[0] >> 5);                          // 8, remains 5
        sm[offset + 2]  = (vi[0] << 3) | ((vq[0] >> 10) & 0x07); // 5 + 3, remains 10
        sm[offset + 3]  = (vq[0] >> 2);                          // 8, remains 2
        sm[offset + 4]  = (vq[0] << 6) | ((vi[1] >> 7) & 0x3f);  // 2 + 6, remains 7
        sm[offset + 5]  = (vi[1] << 1) | ((vq[1] >> 12) & 0x01); // 7 + 1, remains 12
        sm[offset + 6]  = (vq[1] >> 4);                          // 8, Remains 4
        sm[offset + 7]  = (vq[1] << 4) | ((vi[2] >> 9) & 0x0f);  // 4 + 4, remains 9
        sm[offset + 8]  = (vi[2] >> 1);                          // 8, remains 1
        sm[offset + 9]  = (vi[2] << 7) | ((vq[2] >> 6) & 0x7f);  // 1 + 7, remains 6
        sm[offset + 10] = (vq[2] << 2) | ((vi[3] >> 11) & 0x03); // 6 + 2, remains 11
        sm[offset + 11] = (vi[3] >> 3);                          // 8, remains 3
        sm[offset + 12] = (vi[3] << 5) | ((vq[3] >> 8) & 0x1f);  // 3 + 5, remains 8
        sm[offset + 13] = vq[3];
    }
    else if(compbits == 15)
    {
        sm[offset + 1]  = (vi[0] >> 7);                          // 8, remains 7
        sm[offset + 2]  = (vi[0] << 1) | ((vq[0] >> 14) & 0x01); // 7 + 1, remains 14
        sm[offset + 3]  = (vq[0] >> 6);                          // 8, remains 6
        sm[offset + 4]  = (vq[0] << 2) | ((vi[1] >> 13) & 0x03); // 6 + 2, remains 13
        sm[offset + 5]  = (vi[1] >> 5);                          // 8, remains 5
        sm[offset + 6]  = (vi[1] << 3) | ((vq[1] >> 12) & 0x07); // 5 + 3, Remains 12
        sm[offset + 7]  = (vq[1] >> 4);                          // 8, remains 4
        sm[offset + 8]  = (vq[1] << 4) | ((vi[2] >> 11) & 0x0f); // 4 + 4, remains 11
        sm[offset + 9]  = (vi[2] >> 3);                          // 8, remains 3
        sm[offset + 10] = (vi[2] << 5) | ((vq[2] >> 10) & 0x1f); // 3 + 5, remains 10
        sm[offset + 11] = (vq[2] >> 2);                          // 8, remains 2
        sm[offset + 12] = (vq[2] << 6) | ((vi[3] >> 9) & 0x3f);  // 2 + 6, Remains 9
        sm[offset + 13] = (vi[3] >> 1);                          // 8, remains 1
        sm[offset + 14] = (vi[3] << 7) | ((vq[3] >> 8) & 0x7f);  // 1 + 7, remains 8
        sm[offset + 15] = vq[3];
    }
    else if(compbits == 16) // No compression.
    {
        // Just reorder the bytes to network order
        int* smi = (int*)(sm + offset);
        for(int i = 0; i < 4; i++)
            smi[i] = __byte_perm(vi[i], vq[i], 0x4501);
    }
}

// Unpack the compressed data (compbits per value), with 4 (I,Q) pairs per thread.
// Warning: Only the lower "compbits" bits of vi and vq are populated,
// negative numbers are not properly signed!
__device__ inline void unpackInput(
    const uint8_t* __restrict__ input, // Address of the compressed PRB
    uint32_t tx_prb,                   // Rank inside sub-group working on same PRB
    int32_t  vi[4],                    // Decompressed I values
    int32_t  vq[4],                    // Deompressed Q values
    int32_t& compParam,                // Compression parameter
    uint8_t  compbits)                  // Number of compressed bits
{
    if(compbits < 16)
    {
        compParam  = input[0];
        int offset = tx_prb * compbits + 1;
        if(compbits == 7)
        {
            vi[0] = input[offset + 0] >> 1;                                       // 7, remains 1
            vq[0] = ((input[offset + 0] & 0x01) << 6) | (input[offset + 1] >> 2); // 1 + 6, remains 2
            vi[1] = ((input[offset + 1] & 0x03) << 5) | (input[offset + 2] >> 3); // 2 + 5, remains 3
            vq[1] = ((input[offset + 2] & 0x07) << 4) | (input[offset + 3] >> 4); // 3 + 4, remains 4
            vi[2] = ((input[offset + 3] & 0x0f) << 3) | (input[offset + 4] >> 5); // 4 + 3, remains 5
            vq[2] = ((input[offset + 4] & 0x1f) << 2) | (input[offset + 5] >> 6); // 5 + 2, remains 6
            vi[3] = ((input[offset + 5] & 0x3f) << 1) | (input[offset + 6] >> 7); // 6 + 1, remains 7
            vq[3] = input[offset + 6] & 0x7f;                                     // 7
        }
        else if(compbits == 8)
        {
            for(int i = 0; i < 4; i++)
            {
                vi[i] = input[offset + 2 * i];
                vq[i] = input[offset + 2 * i + 1];
            }
        }
        else if(compbits == 9)
        {
            vi[0] = (input[offset] << 1) | (input[offset + 1] >> 7);              // 8 + 1, remains 7
            vq[0] = ((input[offset + 1] & 0x7f) << 2) | (input[offset + 2] >> 6); // 7 + 2, remains 6
            vi[1] = ((input[offset + 2] & 0x3f) << 3) | (input[offset + 3] >> 5); // 6 + 3, remains 5
            vq[1] = ((input[offset + 3] & 0x1f) << 4) | (input[offset + 4] >> 4); // 5 + 4, remains 4
            vi[2] = ((input[offset + 4] & 0x0f) << 5) | (input[offset + 5] >> 3); // 4 + 5, remains 3
            vq[2] = ((input[offset + 5] & 0x07) << 6) | (input[offset + 6] >> 2); // 3 + 6, remains 2
            vi[3] = ((input[offset + 6] & 0x03) << 7) | (input[offset + 7] >> 1); // 2 + 7, remains 1
            vq[3] = ((input[offset + 7] & 0x01) << 8) | input[offset + 8];        // 1 + 8
        }
        else if(compbits == 10)
        {
            vi[0] = (input[offset] << 2) | (input[offset + 1] >> 6);              // 8 + 2, remains 6
            vq[0] = ((input[offset + 1] & 0x3f) << 4) | (input[offset + 2] >> 4); // 6 + 4, remains 4
            vi[1] = ((input[offset + 2] & 0x0f) << 6) | (input[offset + 3] >> 2); // 4 + 6, remains 2
            vq[1] = ((input[offset + 3] & 0x03) << 8) | (input[offset + 4]);      // 2 + 8
            vi[2] = (input[offset + 5] << 2) | (input[offset + 6] >> 6);          // 8 + 2, remains 6
            vq[2] = ((input[offset + 6] & 0x3f) << 4) | (input[offset + 7] >> 4); // 6 + 4, remains 4
            vi[3] = ((input[offset + 7] & 0x0f) << 6) | (input[offset + 8] >> 2); // 4 + 6, remains 2
            vq[3] = ((input[offset + 8] & 0x03) << 8) | (input[offset + 9]);      // 2 + 8
        }
        else if(compbits == 11)
        {
            vi[0] = (input[offset] << 3) | (input[offset + 1] >> 5);                                          // 8 + 3, remains 5
            vq[0] = ((input[offset + 1] & 0x1f) << 6) | (input[offset + 2] >> 2);                             // 5 + 6, remains 2
            vi[1] = ((input[offset + 2] & 0x03) << 9) | (input[offset + 3] << 1) | (input[offset + 4] >> 7);  // 2 + 8 + 1, remains 7
            vq[1] = ((input[offset + 4] & 0x7f) << 4) | (input[offset + 5] >> 4);                             // 7 + 4, remains 4
            vi[2] = ((input[offset + 5] & 0x0f) << 7) | (input[offset + 6] >> 1);                             // 4 + 7, remains 1
            vq[2] = ((input[offset + 6] & 0x01) << 10) | (input[offset + 7] << 2) | (input[offset + 8] >> 6); // 1 + 8 + 2, remains 6
            vi[3] = ((input[offset + 8] & 0x3f) << 5) | (input[offset + 9] >> 3);                             // 6 + 5, remains 3
            vq[3] = ((input[offset + 9] & 0x07) << 8) | (input[offset + 10]);                                 // 3 + 8
        }
        else if(compbits == 12)
        {
            for(int i = 0; i < 4; i++)
            {
                vi[i] = (input[offset + 3 * i] << 4) | (input[offset + 3 * i + 1] >> 4);
                vq[i] = ((input[offset + 3 * i + 1] & 0xf) << 8) | input[offset + 3 * i + 2];
            }
        }
        else if(compbits == 13)
        {
            vi[0] = (input[offset] << 5) | (input[offset + 1] >> 3);                                            // 8 + 5, remains 3
            vq[0] = ((input[offset + 1] & 0x07) << 10) | (input[offset + 2] << 2) | (input[offset + 3] >> 6);   // 3 + 8 + 2, remains 6
            vi[1] = ((input[offset + 3] & 0x3f) << 7) | (input[offset + 4] >> 1);                               // 6 + 7, remains 1
            vq[1] = ((input[offset + 4] & 0x01) << 12) | (input[offset + 5] << 4) | (input[offset + 6] >> 4);   // 1 + 8 + 4, remains 4
            vi[2] = ((input[offset + 6] & 0x0f) << 9) | (input[offset + 7] << 1) | (input[offset + 8] >> 7);    // 4 + 8 + 1, remains 7
            vq[2] = ((input[offset + 8] & 0x7f) << 6) | (input[offset + 9] >> 2);                               // 7 + 6, remains 2
            vi[3] = ((input[offset + 9] & 0x03) << 11) | (input[offset + 10] << 3) | (input[offset + 11] >> 5); // 2 + 8 + 3, remains 5
            vq[3] = ((input[offset + 11] & 0x1f) << 8) | input[offset + 12];                                    // 5 + 8
        }
        else if(compbits == 14)
        {
            vi[0] = (input[offset] << 6) | (input[offset + 1] >> 2);                                             // 8 + 6, remains 2
            vq[0] = ((input[offset + 1] & 0x03) << 12) | (input[offset + 2] << 4) | (input[offset + 3] >> 4);    // 2 + 8 + 4, remains 4
            vi[1] = ((input[offset + 3] & 0x0f) << 10) | (input[offset + 4] << 2) | (input[offset + 5] >> 6);    // 4 + 8 + 2, remains 6
            vq[1] = ((input[offset + 5] & 0x3f) << 8) | input[offset + 6];                                       // 6 + 8
            vi[2] = (input[offset + 7] << 6) | (input[offset + 8] >> 2);                                         // 8 + 6, remains 2
            vq[2] = ((input[offset + 8] & 0x03) << 12) | (input[offset + 9] << 4) | (input[offset + 10] >> 4);   // 2 + 8 + 4, remains 4
            vi[3] = ((input[offset + 10] & 0x0f) << 10) | (input[offset + 11] << 2) | (input[offset + 12] >> 6); // 4 + 8 + 2, remains 6
            vq[3] = ((input[offset + 12] & 0x3f) << 8) | input[offset + 13];                                     // 6 + 8
        }
        else if(compbits == 15)
        {
            vi[0] = (input[offset] << 7) | (input[offset + 1] >> 1);                                            // 8 + 7, remains 1
            vq[0] = ((input[offset + 1] & 0x01) << 14) | (input[offset + 2] << 6) | (input[offset + 3] >> 2);   // 1 + 8 + 6, remains 2
            vi[1] = ((input[offset + 3] & 0x03) << 13) | (input[offset + 4] << 5) | (input[offset + 5] >> 3);   // 2 + 8 + 5, remains 3
            vq[1] = ((input[offset + 5] & 0x07) << 12) | (input[offset + 6] << 4) | (input[offset + 7] >> 4);   // 3 + 8 + 4, remains 4
            vi[2] = ((input[offset + 7] & 0x0f) << 11) | (input[offset + 8] << 3) | (input[offset + 9] >> 5);   // 4 + 8 + 3, remains 5
            vq[2] = ((input[offset + 9] & 0x1f) << 10) | (input[offset + 10] << 2) | (input[offset + 11] >> 6); // 5 + 8 + 2, remains 6
            vi[3] = ((input[offset + 11] & 0x3f) << 9) | (input[offset + 12] << 1) | (input[offset + 13] >> 7); // 6 + 8 + 1, remains 7
            vq[3] = ((input[offset + 13] & 0x7f) << 8) | input[offset + 14];                                    // 7 + 8
        }
    }
    else // compbits == 16, no compression
    {
        const uint* input_vec = reinterpret_cast<const uint*>(input + tx_prb * 16);
        for(int i = 0; i < 4; i++)
        {
            uint v = input_vec[i];
            v      = __byte_perm(v, v, 0x2301); // Remove network order
            vi[i]  = v & 0xffff;
            vq[i]  = v >> 16;
        }
        compParam = 0;
    }
}