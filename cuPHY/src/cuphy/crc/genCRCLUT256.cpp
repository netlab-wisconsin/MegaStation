/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


//#define _TEST_ 1
// each thread reads and process this many input bytes
#define BYTES_PER_THREAD 4

#define POLYQ (0x814141ABu)
#include "crc.hpp"
#include "descrambling.hpp"
#include <iostream>
#include <fstream>
#include <stdint.h>
#include <iomanip>

using namespace crc;

int main()
{
#ifndef _TEST_
    int         nPolys                 = 3;
    const char* POLY_NAMES[nPolys]     = {"G_CRC_24_A", "G_CRC_24_B", "G_CRC_16", "POLY_2"};
    uint32_t    POLYS[nPolys]          = {G_CRC_24_A, G_CRC_24_B, G_CRC_16, descrambling::POLY_2};
    uint32_t    POLY_BIT_SIZES[nPolys] = {24, 24, 16, 31};
    uint32_t    LUT[nPolys][BYTES_PER_THREAD][256];
    uint32_t    MSB_MASKS[nPolys] = {1 << 23, 1 << 23, 1 << 15, 1 << 30};
#else
    int         nPolys                 = 1;
    const char* POLY_NAMES[nPolys]     = {"POLYQ"};
    uint32_t    POLYS[nPolys]          = {POLYQ};
    uint32_t    POLY_BIT_SIZES[nPolys] = {32};
    uint32_t    LUT[nPolys][BYTES_PER_THREAD][256];
    uint32_t    MSB_MASKS[nPolys] = {1U << 31};
#endif
    std::ofstream of("CRC_256_LUTS.h");
    of << "#ifndef _CRC_256_LUTS_H_\n";
    of << "#define _CRC_256_LUTS_H_\n\n";

    for(int p = 0; p < nPolys; p++)
    {
        of << "static __device__ uint32_t " << POLY_NAMES[p] << "_256_LUT[" << BYTES_PER_THREAD << "][256] = {\n";

        for(int i = 0; i < BYTES_PER_THREAD; i++)
        {

                    of << "{\n";
            for(int j = 0; j < 256; j++)
            {
                uint32_t val = (i == 0) ? static_cast<uint32_t>(j << (POLY_BIT_SIZES[p] - 8)) : LUT[p][i - 1][j];
                for(int bit = 0; bit < 8; bit++)
                {
                    if((val & MSB_MASKS[p]) != 0)
                    {
                        val <<= 1;
                        val ^= POLYS[p];
                    }
                    else
                    {
                        val <<= 1;
                    }
                }

                LUT[p][i][j] = val;
                of << "0x" << std::setfill('0') << std::setw(8) << std::uppercase << std::hex << val;

                if(j != 256 - 1)
                    of << ",\n";
                else
                    of << "\n";
            }

            if(i != BYTES_PER_THREAD - 1)
                of << "},\n";
            else
                of << "}\n";
        }

        of << "};\n\n";
    }

    of << "#endif\n";
    of.close();

    return 0;
}
