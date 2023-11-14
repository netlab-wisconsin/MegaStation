/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <gtest/gtest.h>
#include <cstdlib>
#include <iostream>
#include <vector>
#include "descrambling.cuh"
#include "descrambling.hpp"

using namespace descrambling;

uint32_t reverse(uint32_t x, int bits)
{
    x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1);   // Swap _<>_
    x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2);   // Swap __<>__
    x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4);   // Swap ____<>____
    x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8);   // Swap ...
    x = ((x & 0x0000FFFF) << 16) | ((x & 0xFFFF0000) >> 16); // Swap ...
    return x >> (32 - bits);
}

int testDescrambling(uint32_t nTBs, uint32_t TBSize)
{
    // test result
    int      res  = 0;
    uint32_t size = nTBs * TBSize;

    std::vector<float>    llrs(size);
    std::vector<float>    gpuOut(size);
    std::vector<uint32_t> cinitArray(nTBs);
    std::vector<uint32_t> cpuSeq(size);
    std::vector<uint32_t> tbBoundaryArray(nTBs + 1);

    // populate input parameters

    for(int i = 0; i < size; i++) llrs[i] = i;

    for(int i = 0; i < nTBs; i++) cinitArray[i] = 1507348;

    tbBoundaryArray[0] = 0;
    for(int i = 1; i <= nTBs; i++)
        tbBoundaryArray[i] = tbBoundaryArray[i - 1] + TBSize;

    uint32_t maxNCodeBlocks = TBSize / (GLOBAL_BLOCK_SIZE);

    // generate CPU scrambling sequence

    std::vector<uint32_t> x1(NC + size + 31);
    std::vector<uint32_t> x2(NC + size + 31);

    for(int i = 0; i < nTBs; i++)
    {
        for(int n = 0; n < 31; n++)
        {
            x2[n] = (cinitArray[i] >> n) & 0x1;
        }

        x1[0] = 1;

        for(int j = 0; j < NC + tbBoundaryArray[i + 1] - tbBoundaryArray[i]; j++)
        {
            x1[j + 31] = (x1[j + 3] + x1[j]) & 0x1;

            x2[j + 31] = (x2[j + 3] + x2[j + 2] + +x2[j + 1] + x2[j]) & 0x1;
        }
        /*
     for (n = 0; n < len; n++) printf("%d, ", x1[n]);
    */

        for(int j = 0; j < tbBoundaryArray[i + 1] - tbBoundaryArray[i]; j++)
        {
            cpuSeq[tbBoundaryArray[i] + j] = (x1[j + NC] + x2[j + NC]) & 0x1;
        }
    }

    void* descrambleEnv[1];
    cuphyDescrambleInit(descrambleEnv);

    cuphyStatus_t status;

    status = cuphyDescrambleLoadParams(descrambleEnv,
                                       nTBs,
                                       maxNCodeBlocks,
                                       tbBoundaryArray.data(),
                                       cinitArray.data());

    status = cuphyDescrambleLoadInput(descrambleEnv, llrs.data());

    status = cuphyDescramble(descrambleEnv, nullptr, true, 1000, 0);
    status = cuphyDescrambleStoreOutput(descrambleEnv, gpuOut.data());

    // CPU computation
    for(int i = 0; i < nTBs; i++)
    {
        for(int j = 0; j < tbBoundaryArray[i + 1] - tbBoundaryArray[i]; j++)
        {
            if(cpuSeq[tbBoundaryArray[i] + j])
            {
                llrs[tbBoundaryArray[i] + j] = -llrs[tbBoundaryArray[i] + j];
            }
        }
    }

    res = 1;
    for(int i = 0; i < size; i++)
    {
        if(gpuOut[i] != llrs[i])
        {
            std::cout << "Error: not equal at " << i << "(" << size << ") "
                      << gpuOut[i] << " " << llrs[i] << "\n";
            res = 0;
        }
    }
    return res;
}

int DESCRAMBLE_TEST()
{
    int tbSize = 117504;
    int nTBs   = 8;

    int res = testDescrambling(nTBs, tbSize);

    return res;
}

TEST(DESCRAMBLE, SAME_SIZE_TBS) { EXPECT_EQ(DESCRAMBLE_TEST(), 1); }

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
