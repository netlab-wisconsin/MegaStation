/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */
#pragma once

// Calculate the derate match output index per algorithm in 38.212 5.4.2.1 using fast non-iterative method
inline __host__ __device__ int derate_match_fast_calc_modulo(int inIdx, int E, int K, int Kd, int F, int k0, int Ncb, int Zc)
{
    int outIdx;

    // Scenario 0: We start before the hole
    if (k0 < Kd)
    {
        outIdx = k0 + inIdx;

        while (outIdx >= Ncb)
        {
            outIdx -= Ncb;
            outIdx += F;
        }
        if (outIdx >= Kd)
        {
            outIdx += F;
        }
        if (outIdx >= Ncb)
        {
            outIdx -= Ncb;
        }

    }
     // Scenario 1: We start after the hole
     else if (k0 > K)
     {
        outIdx = k0 + inIdx;

        while (outIdx >= Ncb)
        {
            outIdx -= Ncb;
            if (outIdx >= Kd)
            {
                outIdx += F;
            }
        }
    }
    // Scenario 2: We start in the hole
    else //if ((k0 >= Kd) && (k0 <= K))
    {
        int Fmin = K-k0;
        outIdx = k0 + inIdx + Fmin;

        while (outIdx >= Ncb)
        {
            outIdx -= Ncb;
            if (outIdx >= Kd)
            {
                outIdx += F;
            }
        }
    }

    return outIdx;
}
