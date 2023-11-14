/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <exception>
#include <fstream>
#include <iostream>
#include <string>
#include <vector>
#include "../descrambling/descrambling.hpp"
#include "crc.hpp"

std::string LUT_DIR;

// Fibonacci LFSR for first polynomial for Gold sequence generation
// inverted output
inline uint32_t fibonacciLFSR1(uint32_t& state, uint32_t n)
{
    uint32_t res = 0;
// x^{31} + x^3 + 1
#pragma unroll
    for(int i = 0; i < n; i++)
    {
        uint32_t bit = (state) ^ (state >> 3);
        bit          = bit & 1;
        res ^= (state & 1) << i; //(% 32);
        state >>= 1;
        state ^= (bit << 30);
    }
    return res;
}
// Compute (k * expConst)-th power of initVal modulo polynomial poly
template <typename uintCRC_t, int uintCRCBitLength>
uintCRC_t computeModPowK(int k, uintCRC_t poly, uintCRC_t initVal = 1, uint32_t expConst = 8)
{
    int       l           = k;
    uintCRC_t crc         = initVal;
    uintCRC_t msbMask     = (1 << (uintCRCBitLength - 1));
    uintCRC_t allOnesMask = -1;

    if((sizeof(uintCRC_t) * 8 - uintCRCBitLength) > 0)
        allOnesMask >>= (sizeof(uintCRC_t) * 8 - uintCRCBitLength - 1);

    for(int i = 0; i < l; i++)
    {
        for(int b = 0; b < expConst; b++)
        {
            uintCRC_t pred = (crc & msbMask) == 0;
            crc <<= 1;
            crc ^= (poly & (pred + allOnesMask));
        }
    }
    return crc;
}

using namespace std;
using namespace crc;
using namespace descrambling;

//compute Fibonacci LFSR sequence for descrambling LFSR1
void genGoldSeqLFSR1()
{
    uint32_t         state = 0x1;
    vector<uint32_t> seq;
    const uint32_t   wordBitLength = sizeof(uint32_t) * 8;
    for(int i = 0; i < MAX_INPUT_TB_SIZE_IN_WORDS; i++)
    {
        uint32_t seqWord = fibonacciLFSR1(state, 32);
        if((i)*wordBitLength >= NC)
            seq.push_back(seqWord);
    }

    std::string fname = std::string(LUT_DIR) + "/GOLD_1_SEQ_LUT.h";
    std::cout << "Opening '" << fname << "' for writing..." << std::endl;
    ofstream of(fname);

    of << "#ifndef _GOLD_1_SEQ_LUT_H_\n";
    of << "#define _GOLD_1_SEQ_LUT_H_\n";
    of << "#include <stdint.h>\n\n";
    of << "const uint32_t GOLD_1_SEQ_LUT_SIZE = " << MAX_INPUT_TB_SIZE_IN_WORDS << ";\n";
    of << "static __device__ uint32_t GOLD_1_SEQ_LUT[GOLD_1_SEQ_LUT_SIZE] = {\n";

    for(int j = 0; j < MAX_INPUT_TB_SIZE_IN_WORDS; j++)
    {
        of << "0x" << std::hex << seq[j];
        if(j < MAX_INPUT_TB_SIZE_IN_WORDS - 1) of << ",\n";
    }
    of << "};\n"
          "#endif";
}

template <typename polyType, size_t polyBitLength>
void computeAndStorePowers(string       polyName,
                           string       polyTypeString,
                           polyType     poly,
                           unsigned int tableSize,
                           bool         invert = true)
{
    std::string fname = std::string(LUT_DIR) + "/" + polyName +
                        "_P_LUT.h";

    std::cout << "Opening '" << fname << "' for writing..." << std::endl;
    ofstream of(fname);

    const int entryBitLength = sizeof(polyType) * 8;
    polyType* LUT            = new polyType[tableSize];

    uint8_t power[1] = {1};

    // for dynamic programming precompute
    polyType lastVal;

    // this values works for both downlink and uplink
    lastVal = computeCRC<polyType, polyBitLength>(power, 0, poly, (0x1 << (8)));
    if(polyBitLength == 16)
        lastVal = computeModPowK<polyType, polyBitLength>(1, poly, 1, 16);

    if(invert)
    {
        LUT[tableSize - 1] = lastVal;

        for(int i = tableSize - 2; i >= 0; i--)
        {
            lastVal = computeModPowK<polyType, polyBitLength>(1, poly, lastVal, 32);
            LUT[i]  = lastVal;
        }
    }
    else
    {
        LUT[0] = lastVal;

        for(int i = 1; i < tableSize; i++)
        {
            lastVal = computeModPowK<polyType, polyBitLength>(1, poly, lastVal, 32);
            LUT[i]  = lastVal;
        }
    }

    of << "#ifndef _" << polyName
       << "_P_LUT_H_\n";
    of << "#define _" << polyName
       << "_P_LUT_H_\n";
    of << "#include <stdint.h>\n\n";
    of << "const uint32_t " << polyName
       << "_P_LUT_SIZE = " << tableSize << ";\n";

    of << "static __device__ " << polyTypeString << " " << polyName
       << "_P_LUT[" << polyName
       << "_P_LUT_SIZE] = {\n";

    for(uint32_t j = 0; j < tableSize; j++)
    {
        of << "0x" << std::hex << LUT[j];
        if(j < tableSize - 1) of << ",\n";
    }
    of << "};\n"
          "#endif";
    delete[] LUT;
    of.close();
}
// Compute x^{84i} values modulo poly (uint type)
// and store into header file
// each framgment of the resulting uint elements is laid out for coalesced access

// uint type of poly, poly length in bits
template <typename polyType, size_t polyBitLength>
void computeAndStorePowerLUTCoalesced(string       polyName,
                                      string       polyTypeString,
                                      polyType     poly,
                                      unsigned int tableSize,
                                      const char*  sizeLabel        = nullptr,
                                      bool         invert           = true,
                                      uint32_t     skip             = 0,
                                      uint32_t     wordFragmentSize = 8)
{
    uint32_t    wordFragmentByteSize = (wordFragmentSize + 8 - 1) / 8;
    std::string sizeStr;
    if(sizeLabel == nullptr)
        sizeStr = std::to_string(wordFragmentByteSize);
    else
        sizeStr = std::string(sizeLabel);

    std::string fname = std::string(LUT_DIR) + "/" + polyName + "_" + sizeStr +
                        "_P_LUT.h";
    std::cout << "Opening '" << fname << "' for writing..." << std::endl;
    ofstream of(fname);

    if(wordFragmentSize > 8 * wordFragmentByteSize)
        throw std::out_of_range("Fragment size of table words(" + std::to_string(wordFragmentSize) + ") can't be larger than table size in bytes(" + std::to_string(8 * wordFragmentByteSize) + ")");

    const int entryBitLength = sizeof(polyType) * 8;
    int       polyFragments  = (polyBitLength + wordFragmentSize - 1) / wordFragmentSize;
    polyType* LUT            = new polyType[tableSize];

    // Handle x^{8i} values less than poly first
    uint64_t     t                    = 0;
    uint64_t     i                    = 0;
    uint8_t      power[1]             = {1};
    unsigned int tableCoalescedOffset = tableSize / ((entryBitLength) / wordFragmentSize);
    uint32_t     nFragmentsPerWord    = (entryBitLength) / wordFragmentSize;

    // for dynamic programming precompute
    polyType lastVal;
    uint64_t s = 0;
    while(t < polyFragments)
    {
        lastVal =
            computeCRC<polyType, polyBitLength>(power, 0, poly, (0x1 << (i * wordFragmentSize)));
        if(i >= skip)
        {
            LUT[i / nFragmentsPerWord + (i % nFragmentsPerWord) * tableCoalescedOffset] = lastVal;
            s++;
        }
        i++;
        t += 1;
    }

    for(; i < tableSize + skip; i++)
    {
        lastVal = computeModPowK<polyType, polyBitLength>(1, poly, lastVal, wordFragmentSize);

        if(i >= skip)
        {
            LUT[s / nFragmentsPerWord + (s % nFragmentsPerWord) * tableCoalescedOffset] = lastVal;
            s++;
        }
    }

    // invert table for big endian access
    if(invert)
    {
        for(int base = 0; base <= tableCoalescedOffset * (nFragmentsPerWord - 1); base += tableCoalescedOffset)
        {
            for(int j = 0; j < tableCoalescedOffset / 2; j++)
            {
                polyType temp                            = LUT[base + j];
                LUT[base + j]                            = LUT[base + tableCoalescedOffset - j - 1];
                LUT[base + tableCoalescedOffset - j - 1] = temp;
            }
        }
    }
    of << "#ifndef _" << polyName << "_" << sizeStr << "_P_LUT_H_\n";
    of << "#define _" << polyName << "_" << sizeStr << "_P_LUT_H_\n";
    of << "#include <stdint.h>\n\n";
    of << "const uint32_t " << polyName << "_" << sizeStr
       << "_P_LUT_SIZE = " << tableSize << ";\n";
    of << "const uint32_t " << polyName << "_" << sizeStr
       << "_P_LUT_OFFSET = " << tableCoalescedOffset << ";\n";

    of << "static __device__ " << polyTypeString << " " << polyName << "_" << sizeStr
       << "_P_LUT[" << polyName << "_" << sizeStr << "_P_LUT_SIZE] = {\n";

    for(uint32_t j = 0; j < tableSize; j++)
    {
        of << "0x" << std::hex << LUT[j];
        if(j < tableSize - 1) of << ",\n";
    }
    of << "};\n"
          "#endif";
    delete[] LUT;
    of.close();
}

// Compute x^8i values modulo poly (uint type) and store into file
template <typename polyType, size_t polyBitLength, size_t wordFragmentByteSize>
void computeAndStorePowerLUT8(string       polyName,
                              string       polyTypeString,
                              polyType     poly,
                              unsigned int tableSize,
                              const char*  sizeLabel = nullptr)
{
    std::string sizeStr;
    if(sizeLabel == nullptr)
        sizeStr = std::to_string(wordFragmentByteSize);
    else
        sizeStr = std::string(sizeLabel);

    std::string fname = std::string(LUT_DIR) + "/" + polyName + "_" + sizeStr +
                        "_P_LUT.h";
    std::cout << "Opening '" << fname << "' for writing..." << std::endl;
    ofstream of(fname);

    if(tableSize > MAX_PTABLE_SIZE)
        throw std::out_of_range("Power table size can be at most " +
                                std::to_string(MAX_PTABLE_SIZE));

    const int entryBitLength = sizeof(polyType);
    int       polyBytes      = (polyBitLength + 8 - 1) / 8;
    polyType* LUT            = new polyType[tableSize];
    uint8_t   power[1]       = {0};

    // HANDLE x^{8i} less than poly first
    unsigned t = 0;
    uint64_t i = 0;

    //  Uninitialized scalar variable
    LUT[0] = 0x0;

    while(t < polyBytes)
    {
        LUT[i] = computeCRC<polyType, polyBitLength>(power,
                                                     static_cast<int>(0),
                                                     poly,
                                                     static_cast<polyType>((0x1 << (i * 8))));
        i++;
        t += wordFragmentByteSize;
    }
    for(; i < tableSize; i++)
    {
        LUT[i] = computeModPowK<polyType, polyBitLength>(wordFragmentByteSize,
                                                         poly,
                                                         LUT[i - 1]);
    }

    // invert table for big endian access

    for(int i = 0; i < tableSize / 2; i++)
    {
        polyType temp          = LUT[i];
        LUT[i]                 = LUT[tableSize - i - 1];
        LUT[tableSize - i - 1] = temp;
    }

    of << "#ifndef _" << polyName << "_" << sizeStr << "_P_LUT_H_\n";
    of << "#define _" << polyName << "_" << sizeStr << "_P_LUT_H_\n";
    of << "#include <stdint.h>\n\n";
    of << "const uint32_t " << polyName << "_" << sizeStr
       << "_P_LUT_SIZE = " << tableSize << ";\n";
    of << "static " << polyTypeString << " " << polyName << "_" << sizeStr
       << "_P_LUT[" << polyName << "_" << sizeStr << "_P_LUT_SIZE] = {\n";

    for(int j = 0; j < tableSize; j++)
    {
        of << "0x" << std::hex << LUT[j];
        if(j < tableSize - 1) of << ",\n";
    }
    of << "};\n"
          "#endif";
    delete[] LUT;
    of.close();
}

// Precompute large values x^i modulo poly and store into header file
template <typename polyType, size_t polyBitLength, size_t wordFragmentByteSize>
void computeAndStorePowerLUTLARGE(string       polyName,
                                  string       polyTypeString,
                                  polyType     poly,
                                  unsigned int tableSize,
                                  const char*  sizeLabel = nullptr)
{
    std::string sizeStr;
    if(sizeLabel == nullptr)
        sizeStr = std::to_string(wordFragmentByteSize);
    else
        sizeStr = std::string(sizeLabel);

    std::string fname = std::string(LUT_DIR) + "/" + polyName + "_" + sizeStr +
                        "_P_LUT.h";
    std::cout << "Opening '" << fname << "' for writing..." << std::endl;
    ofstream of(fname);

    if(tableSize > MAX_PTABLE_SIZE)
        throw std::out_of_range("Power table size can be at most " +
                                std::to_string(MAX_PTABLE_SIZE));
    const int entryBitLength = sizeof(polyType);

    int       polyBytes = (polyBitLength + 8 - 1) / 8;
    polyType* LUT       = new polyType[tableSize];

    uint64_t i = 0;

    LUT[0] = 0x1;
    for(i = 1; i < tableSize; i++)
    {
        LUT[i] = computeModPowK<polyType, polyBitLength>(wordFragmentByteSize,
                                                         poly,
                                                         LUT[i - 1]);
    }

    for(int i = 0; i < tableSize / 2; i++)
    {
        polyType temp          = LUT[i];
        LUT[i]                 = LUT[tableSize - i - 1];
        LUT[tableSize - i - 1] = temp;
    }

    of << "#ifndef _" << polyName << "_" << sizeStr << "_P_LUT_H_\n";
    of << "#define _" << polyName << "_" << sizeStr << "_P_LUT_H_\n";
    of << "#include <stdint.h>\n\n";
    of << "const uint32_t " << polyName << "_" << sizeStr
       << "_P_LUT_SIZE = " << tableSize << ";\n";
    of << "static __device__ " << polyTypeString << " " << polyName << "_" << sizeStr
       << "_P_LUT[" << polyName << "_" << sizeStr << "_P_LUT_SIZE] = {\n";

    for(int j = 0; j < tableSize; j++)
    {
        of << "0x" << std::hex << LUT[j];
        if(j < tableSize - 1) of << ",\n";
    }
    of << "};\n"
          "#endif";
    delete[] LUT;
    of.close();
}

int main(int argc, char** argv)
{
    if(argc < 2)
    {
        std::cout
            << "Missing LUT directory parameter. Usage: ./genLUT [LUT dirname]\n";
        return -1;
    }
    LUT_DIR = std::string(argv[1]);

    std::cout << "Generate look-up tables for 3GPP NR CRC "
                 "and descrambling: \n";
    std::cout << LUT_DIR << "\n";

    uint32_t nWordFragmentsCRC = sizeof(uint32_t) * 8 / crc::BITS_PROCESSED_PER_LUT_ENTRY;
    // clang-format off
    computeAndStorePowers<uint32_t, 16>               ("G_CRC_16", "uint16_t", G_CRC_16, (MAX_WORDS_PER_SMALL_CODE_BLOCK), true);
    computeAndStorePowers<uint32_t, 24>               ("G_CRC_24_A", "uint32_t", G_CRC_24_A, MAX_WORDS_PER_TRANSPORT_BLOCK, true);
    computeAndStorePowers<uint32_t, 24>               ("G_CRC_24_B", "uint32_t", G_CRC_24_B, (MAX_WORDS_PER_CODE_BLOCK), true);
    // GOLD SEQUENCE / DESCRAMBLING LUTS
    genGoldSeqLFSR1();
    // skip NC = 1600 bits as per 5G spec
    uint32_t skipNC = NC / descrambling::BITS_PROCESSED_PER_LUT_ENTRY;
    uint32_t nWordFragments = 1;//sizeof(uint32_t) * 8 / descrambling::BITS_PROCESSED_PER_LUT_ENTRY;
    computeAndStorePowerLUTCoalesced<uint32_t, 31>("GOLD_2", "uint32_t", descrambling::POLY_2, MAX_INPUT_TB_SIZE_IN_WORDS * nWordFragments, "32", false, skipNC, descrambling::BITS_PROCESSED_PER_LUT_ENTRY);
    // clang-format on
}
