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
#include <fstream>
#include <iostream>
#include <string>

#include "crc.hpp"
#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_internal.h"

using namespace crc;
using namespace cuphy_i;

// utility function for unit test
template <typename baseType>
unsigned long equalCount(baseType* a, baseType* b, unsigned long nElements, const std::string& label = "")
{
    unsigned long popCount = 0;
    for(int i = 0; i < nElements; i++)
    {
        popCount += a[i] != b[i];
        if(a[i] != b[i])
        {
            std::cout << label << "NOT EQUAL (" << std::dec << i << ") a: " << std::hex << a[i]
                      << " b: " << std::hex << b[i] << "\n";
        }
    }
    return popCount == 0;
}

template <typename baseType>
void linearToCoalesced(baseType*     coalescedData,
                       baseType*     linearData,
                       unsigned long nElements,
                       unsigned long elementSize,
                       unsigned long stride)
{
    for(int i = 0; i < nElements; i++)
    {
        for(int j = 0; j < elementSize; j++)
            coalescedData[j * stride + i] = linearData[i * elementSize + j];
    }
}

int CRC_GPU_UPLINK_PUSCH_TEST(bool timeIt)
{
    uint32_t  nTBs                    = MAX_N_TBS_SUPPORTED;
    uint32_t* firstCodeBlockIdxArray  = new uint32_t[nTBs];
    uint32_t* nCodeBlocks             = new uint32_t[nTBs];
    uint32_t* codeBlockByteSizes      = new uint32_t[nTBs]; // 56;
    uint32_t* codeBlockWordSizes      = new uint32_t[nTBs]; // 56;
    uint32_t* codeBlockDataByteSizes  = new uint32_t[nTBs]; // 1053;
    uint32_t* CBPaddingByteSizes      = new uint32_t[nTBs]; // pad to 32-bit boundary
    uint32_t* crcByteSizes            = new uint32_t[nTBs];
    uint32_t* totalCodeBlockByteSizes = new uint32_t[nTBs];
    uint32_t  totalByteSize           = 0;
    uint32_t  totalNCodeBlocks        = 0;
    uint32_t* tbPaddedByteSizes       = new uint32_t[nTBs];
    uint32_t  totalTBPaddedByteSize   = 0;
    uint32_t  ratio                   = sizeof(uint32_t) / sizeof(uint8_t);
    // Same CRC value for each code block, code blocks are all the same
    // linear input layout : cb1|crc1, cb2|crc2, ...

    for(int i = 0; i < nTBs; i++)
    {
        if(i == 1)
        {
            nCodeBlocks[i]        = 1;
            crcByteSizes[i]       = 2;
            codeBlockByteSizes[i] = 333;
        }
        else if(i == 2)
        {
            nCodeBlocks[i]        = 1;
            crcByteSizes[i]       = 3;
            codeBlockByteSizes[i] = 945;
        }

        else
        {
            codeBlockByteSizes[i] = 1007;
            nCodeBlocks[i]        = 6;
            crcByteSizes[i]       = 3;
        }
        totalNCodeBlocks += nCodeBlocks[i];
        codeBlockDataByteSizes[i]  = codeBlockByteSizes[i] - crcByteSizes[i];
        CBPaddingByteSizes[i]      = (MAX_BYTES_PER_CODE_BLOCK - (codeBlockByteSizes[i] % MAX_BYTES_PER_CODE_BLOCK)) % MAX_BYTES_PER_CODE_BLOCK;
        totalCodeBlockByteSizes[i] = codeBlockByteSizes[i] + CBPaddingByteSizes[i];
        codeBlockWordSizes[i]      = totalCodeBlockByteSizes[i] / ratio;
        totalByteSize += totalCodeBlockByteSizes[i] * nCodeBlocks[i];
        tbPaddedByteSizes[i] = (codeBlockDataByteSizes[i] + (nCodeBlocks[i] == 1 ? crcByteSizes[i] : 0)) * nCodeBlocks[i] +
                               (4 - (nCodeBlocks[i] * (codeBlockDataByteSizes[i] + (nCodeBlocks[i] == 1 ? crcByteSizes[i] : 0)) % 4)) % 4;
        totalTBPaddedByteSize += tbPaddedByteSizes[i];
    }

    PerTbParams* tbPrmsArray           = new PerTbParams[nTBs];
    uint8_t*     linearInput           = new uint8_t[totalByteSize];
    uint32_t*    goldenCRCs            = new uint32_t[totalNCodeBlocks];
    uint8_t*     goldenTransportBlocks = new uint8_t[totalTBPaddedByteSize];
    uint32_t*    transportBlocks       = new uint32_t[totalTBPaddedByteSize / ratio];
    uint32_t*    codeBlocks            = (uint32_t*)linearInput;
    uint32_t*    crcs                  = new uint32_t[totalNCodeBlocks];
    uint32_t*    tbCRCs                = new uint32_t[nTBs];
    memset(goldenTransportBlocks, 0, totalTBPaddedByteSize);
    memset(firstCodeBlockIdxArray, 0, nTBs * sizeof(uint32_t));
    uint32_t tbBytes      = 0;
    uint32_t totalCBBytes = 0;
    uint32_t totalCBs     = 0;
    for(int t = 0; t < nTBs; t++)
    {
        // Build transport block
        uint32_t cbBytes = 0;

        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            memset(goldenTransportBlocks + tbBytes + cbBytes,
                   rand(),
                   codeBlockDataByteSizes[t]);
            cbBytes += codeBlockDataByteSizes[t];
        }

        // last code block contains TB CRC in the last 3 bytes
        if(nCodeBlocks[t] > 1)
        {
            uint32_t golden_tbCRC = computeCRC<uint32_t, 24>(goldenTransportBlocks + tbBytes,
                                                             codeBlockDataByteSizes[t] * nCodeBlocks[t] - crcByteSizes[t],
                                                             G_CRC_24_A,
                                                             0,
                                                             1);
            for(int j = 0; j < crcByteSizes[t]; j++)
                goldenTransportBlocks[tbBytes + nCodeBlocks[t] * codeBlockDataByteSizes[t] - crcByteSizes[t] + j] =
                    (golden_tbCRC >> (crcByteSizes[t] - 1 - j) * 8) & 0xFF;
        }
        // compute CB crcs
        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            uint8_t* cbPtr  = linearInput + i * totalCodeBlockByteSizes[t] + totalCBBytes;
            uint8_t* crcPtr = (cbPtr + codeBlockDataByteSizes[t]);
            memcpy(cbPtr,
                   goldenTransportBlocks + i * codeBlockDataByteSizes[t] + tbBytes,
                   codeBlockDataByteSizes[t]);
            uint32_t crc;
            if(nCodeBlocks[t] == 1)
            {
                if(codeBlockDataByteSizes[t] <= MAX_SMALL_A_BYTES)
                {
                    crc = computeCRC<uint32_t, 16>((uint8_t*)cbPtr,
                                                   codeBlockDataByteSizes[t],
                                                   G_CRC_16,
                                                   0,
                                                   1);
                }
                else
                    crc = computeCRC<uint32_t, 24>((uint8_t*)cbPtr,
                                                   codeBlockDataByteSizes[t],
                                                   G_CRC_24_A,
                                                   0,
                                                   1);
                for(int j = 0; j < crcByteSizes[t]; j++)
                    goldenTransportBlocks[tbBytes + nCodeBlocks[t] * codeBlockDataByteSizes[t] /*- crcByteSizes[t]*/ + j] =
                        (crc >> (crcByteSizes[t] - 1 - j) * 8) & 0xFF;
            }

            else

                crc = computeCRC<uint32_t, 24>((uint8_t*)cbPtr,
                                               codeBlockDataByteSizes[t],
                                               G_CRC_24_B,
                                               0,
                                               1);

            for(int j = 0; j < crcByteSizes[t]; j++)
                crcPtr[j] = (crc >> (crcByteSizes[t] - 1 - j) * 8) & 0xFF;
            goldenCRCs[totalCBs] = 0;
            totalCBs++;
            memset(cbPtr + codeBlockByteSizes[t], 0, CBPaddingByteSizes[t]);
        }
        tbPrmsArray[t].num_CBs             = nCodeBlocks[t];
        tbPrmsArray[t].K                   = codeBlockByteSizes[t] * 8;
        tbPrmsArray[t].F                   = 0;
        tbPrmsArray[t].firstCodeBlockIndex = 0;
        tbBytes += tbPaddedByteSizes[t];
        totalCBBytes += nCodeBlocks[t] * totalCodeBlockByteSizes[t];
    }

#if 0
    std::cout << "CBs:\n";
    for (int i = 0; i < totalByteSize; i++)
        std::cout << std::hex << (unsigned short)linearInput[i] << ",";
    std::cout << "\n";

    std::cout << "TB:\n";
    for (int i = 0; i < totalTBPaddedByteSize; i++)
        std::cout << std::hex << (unsigned short)goldenTransportBlocks[i] << ",";
    std::cout << "\n";
#endif

    //input
    unique_device_ptr<uint32_t> d_codeBlocks = make_unique_device<uint32_t>(totalByteSize / sizeof(uint32_t));

    unique_device_ptr<PerTbParams> d_tbPrmsArray = make_unique_device<PerTbParams>(nTBs);
    //output

    unique_device_ptr<uint32_t> d_CBCRCs = make_unique_device<uint32_t>(nTBs * nCodeBlocks[0]);
    unique_device_ptr<uint32_t> d_TBCRCs = make_unique_device<uint32_t>(nTBs);
    unique_device_ptr<uint8_t>  d_TBs    = make_unique_device<uint8_t>(nTBs * tbPaddedByteSizes[0]);

    cudaMemcpy(d_codeBlocks.get(), codeBlocks, totalByteSize, cudaMemcpyHostToDevice);

    cudaMemcpy(d_tbPrmsArray.get(), tbPrmsArray, sizeof(PerTbParams) * nTBs, cudaMemcpyHostToDevice);

    cuphyStatus_t status = cuphyCRCDecode(
        d_CBCRCs.get(),
        d_TBCRCs.get(),
        d_TBs.get(),
        d_codeBlocks.get(),
        d_tbPrmsArray.get(),
        nTBs,
        nCodeBlocks[0],
        tbPaddedByteSizes[0],
        0,
        timeIt,
        0,
        false,
        0);

    cudaMemcpy(crcs, d_CBCRCs.get(), sizeof(uint32_t) * totalNCodeBlocks, cudaMemcpyDeviceToHost);

    cudaMemcpy(tbCRCs, d_TBCRCs.get(), sizeof(uint32_t) * nTBs, cudaMemcpyDeviceToHost);

    cudaMemcpy(transportBlocks, d_TBs.get(), totalTBPaddedByteSize, cudaMemcpyDeviceToHost);

    int passed = 0;

    if(status != CUPHY_STATUS_SUCCESS)
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "CRC: CUPHY ERROR");
    else
    {
        uint32_t* gt = (uint32_t*)goldenTransportBlocks;
        passed       = equalCount(crcs, goldenCRCs, totalNCodeBlocks, "CB CRC ");
        passed &= equalCount(transportBlocks, gt, totalTBPaddedByteSize / ratio, "TB DATA");

        for(int i = 0; i < nTBs; i++)
        {
            passed &= (tbCRCs[i] == 0);
            if(tbCRCs[i] != 0)
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "TB[{}] CRC not equal to 0: {}", i, tbCRCs[i]);
        }
    }

    delete[] tbPrmsArray;
    delete[] crcs;
    delete[] tbCRCs;
    delete[] goldenCRCs;
    delete[] goldenTransportBlocks;
    delete[] linearInput;
    delete[] transportBlocks;
    delete[] firstCodeBlockIdxArray;
    delete[] nCodeBlocks;
    delete[] codeBlockByteSizes;
    delete[] crcByteSizes;
    delete[] codeBlockWordSizes;
    delete[] codeBlockDataByteSizes;
    delete[] CBPaddingByteSizes; // pad to 32-bit boundary
    delete[] totalCodeBlockByteSizes;
    delete[] tbPaddedByteSizes;

    return passed;
}

int CRC_GPU_DOWNLINK_PDSCH(bool timeIt, int numCBs = 2) // FIXME numCBs has no effect!
{
    uint32_t  nTBs                    = 2;
    uint32_t* nCodeBlocks             = new uint32_t[nTBs];
    uint32_t* crcByteSizes            = new uint32_t[nTBs];
    uint32_t* codeBlockByteSizes      = new uint32_t[nTBs];
    uint32_t* codeBlockDataByteSizes  = new uint32_t[nTBs];
    uint32_t* cbPaddingByteSizes      = new uint32_t[nTBs];
    uint32_t* fillerByteSizes         = new uint32_t[nTBs];
    uint32_t* tensorStrideByteSizes   = new uint32_t[nTBs];
    uint32_t* totalCodeBlockByteSizes = new uint32_t[nTBs]; // The CB size that include every thing
    uint32_t* tbPaddedByteSizes       = new uint32_t[nTBs];
    uint32_t  totalNCodeBlocks        = 0;
    uint32_t  totalInTBByteSize       = 0;
    uint32_t  totalOutTBByteSizes     = 0;
    uint32_t  ratio                   = sizeof(uint32_t) / sizeof(uint8_t);

    for(int i = 0; i < nTBs; i++)
    {
        codeBlockByteSizes[i]    = 1056;
        crcByteSizes[i]          = 3;
        nCodeBlocks[i]           = 2;
        fillerByteSizes[i]       = 0;
        tensorStrideByteSizes[i] = 0;

        codeBlockDataByteSizes[i] = codeBlockByteSizes[i] - crcByteSizes[i] - fillerByteSizes[i];
        totalNCodeBlocks += nCodeBlocks[i];
        tbPaddedByteSizes[i] = codeBlockDataByteSizes[i] * nCodeBlocks[i];
        tbPaddedByteSizes[i] += (4 - (tbPaddedByteSizes[i] % 4)) % 4;
        totalInTBByteSize += tbPaddedByteSizes[i];
        totalCodeBlockByteSizes[i] = codeBlockByteSizes[i] + tensorStrideByteSizes[i];
        cbPaddingByteSizes[i]      = (4 - totalCodeBlockByteSizes[i] % 4) % 4;
        totalCodeBlockByteSizes[i] += cbPaddingByteSizes[i];
        totalOutTBByteSizes += totalCodeBlockByteSizes[i] * nCodeBlocks[i]; // should be a multiple of 4
    }

    PdschPerTbParams* tbPrmsArray      = new PdschPerTbParams[nTBs];
    uint8_t*     goldenTransportBlocks = new uint8_t[totalInTBByteSize];
    uint32_t*    crcs                  = new uint32_t[totalNCodeBlocks];
    uint32_t*    tbCRCs                = new uint32_t[nTBs];
    uint32_t*    goldenTBCRCs          = new uint32_t[nTBs];
    uint32_t*    goldenCRCs            = new uint32_t[totalNCodeBlocks];
    uint32_t*    codeBlocks            = new uint32_t[totalOutTBByteSizes >> 2];

    memset(goldenTransportBlocks, 0, totalInTBByteSize);
    memset(codeBlocks, 0, totalOutTBByteSizes);

    uint32_t tbOutBytes = 0;
    uint32_t tbBytes    = 0;
    uint8_t* outputs    = (uint8_t*)codeBlocks;
    for(int t = 0; t < nTBs; t++)
    {
        uint32_t cbBytes    = 0;
        uint32_t cbOutBytes = 0;
        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            // Set the input
            memset(goldenTransportBlocks + tbBytes + cbBytes,
                   rand(),
                   codeBlockDataByteSizes[t]);
            memcpy(outputs + tbOutBytes + cbOutBytes,
                   goldenTransportBlocks + tbBytes + cbBytes,
                   codeBlockDataByteSizes[t]);
            cbBytes += codeBlockDataByteSizes[t];
            cbOutBytes += totalCodeBlockByteSizes[t];
        }
        tbPrmsArray[t].num_CBs             = nCodeBlocks[t];
        tbPrmsArray[t].K                   = codeBlockByteSizes[t] * 8;
        tbPrmsArray[t].F                   = 0;
        tbPrmsArray[t].firstCodeBlockIndex = 0;
        // Add tbSize, tbStartOffset and paddingBytes fields
        tbPrmsArray[t].tbSize        = codeBlockDataByteSizes[t] * nCodeBlocks[t] - 3;
        tbPrmsArray[t].tbStartOffset = (t == 0) ? 0 : (tbPrmsArray[t - 1].tbStartOffset + tbPrmsArray[t - 1].tbSize);
        tbPrmsArray[t].tbStartAddr   = nullptr; // not used in CRC encode
        //tbPrmsArray[t].paddingBytes  = tbPaddedByteSizes[t] - tbPrmsArray[t].tbSize;
        tbPrmsArray[t].cumulativeTbSizePadding =  tbPaddedByteSizes[t] + ((t == 0) ? 0 : tbPrmsArray[t-1].cumulativeTbSizePadding);

        // The last CB in one TB is 3 bytes shorter than other CBs
        memset(goldenTransportBlocks + tbBytes +
                   nCodeBlocks[t] * codeBlockDataByteSizes[t] - 3,
               0,
               3);
        memset(outputs + tbOutBytes + (nCodeBlocks[t] - 1) * totalCodeBlockByteSizes[t] +
                   codeBlockDataByteSizes[t] - 3,
               0,
               3);
        tbBytes += tbPaddedByteSizes[t];
        tbOutBytes += totalCodeBlockByteSizes[t] * nCodeBlocks[t];
    }

    //input
    unique_device_ptr<uint32_t>    d_transportBlocks = make_unique_device<uint32_t>(totalInTBByteSize / ratio);
    unique_device_ptr<PdschPerTbParams> d_tbPrmsArray     = make_unique_device<PdschPerTbParams>(nTBs);

    //output

    unique_device_ptr<uint32_t> d_CBCRCs     = make_unique_device<uint32_t>(nTBs * nCodeBlocks[0]);
    unique_device_ptr<uint32_t> d_TBCRCs     = make_unique_device<uint32_t>(nTBs);
    unique_device_ptr<uint8_t>  d_codeBlocks = make_unique_device<uint8_t>(totalOutTBByteSizes);

    cudaMemcpy(d_transportBlocks.get(), (uint32_t*)goldenTransportBlocks, totalInTBByteSize, cudaMemcpyHostToDevice);

    //cudaMemcpy(d_codeBlocks.get(), (uint8_t*)codeBlocks, totalOutTBByteSizes, cudaMemcpyHostToDevice);

    CUDA_CHECK(cudaMemcpy(d_tbPrmsArray.get(), tbPrmsArray, sizeof(PdschPerTbParams) * nTBs, cudaMemcpyHostToDevice));

    // Allocate launch config struct.
    std::unique_ptr<cuphyCrcEncodeLaunchConfig> crc_hndl = std::make_unique<cuphyCrcEncodeLaunchConfig>();

    // Allocate descriptors and setup rate matching component
    uint8_t       desc_async_copy = 1; // Copy descriptor to the GPU during setup. And set TB-CRCs to 0.
    uint8_t*      h_crc_encode_desc;
    size_t        desc_size = 0, alloc_size = 0;
    cuphyStatus_t status = cuphyCrcEncodeGetDescrInfo(&desc_size, &alloc_size);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        printf("cuphyCrcEncodeGetDescrInfo error %d\n", status);
    }
    unique_device_ptr<uint8_t> d_crc_encode_desc = make_unique_device<uint8_t>(desc_size);
    CUDA_CHECK(cudaHostAlloc((void**)&h_crc_encode_desc, desc_size, cudaHostAllocDefault));

    cudaStream_t cuda_strm = 0;

    status = cuphySetupCrcEncode(crc_hndl.get(),
                                 d_CBCRCs.get(),
                                 d_TBCRCs.get(),
                                 d_transportBlocks.get(),
                                 d_codeBlocks.get(),
                                 d_tbPrmsArray.get(),
                                 nTBs,
                                 nCodeBlocks[0],
                                 tbPaddedByteSizes[0],
                                 0,
                                 false,
                                 h_crc_encode_desc,
                                 d_crc_encode_desc.get(),
                                 desc_async_copy,
                                 cuda_strm);

    if(status != CUPHY_STATUS_SUCCESS)
    {
        throw std::runtime_error("Invalid argument(s) for cuphySetupCrcEncode");
    }

    // CRC has 2 kernels right now
    CUresult status_k1 = launch_kernel(crc_hndl.get()->m_kernelNodeParams[0], cuda_strm);
    CUresult status_k2 = launch_kernel(crc_hndl.get()->m_kernelNodeParams[1], cuda_strm);
    if((status_k1 != CUDA_SUCCESS) ||
       (status_k2 != CUDA_SUCCESS))
    {
        throw std::runtime_error("CRC Encode error(s)");
    }

    CUDA_CHECK(cudaStreamSynchronize(cuda_strm));

    CUDA_CHECK(cudaMemcpy(crcs, d_CBCRCs.get(), totalNCodeBlocks * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(tbCRCs, d_TBCRCs.get(), nTBs * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy((uint8_t*)codeBlocks, d_codeBlocks.get(), totalOutTBByteSizes, cudaMemcpyDeviceToHost));

    tbOutBytes = 0;
    tbBytes    = 0;

    int passed = 1;

    for(int t = 0; t < nTBs; t++)
    {
        // compute TB crcs
        uint32_t tbcrc  = computeCRC<uint32_t, 24>(goldenTransportBlocks + tbBytes,
                                                  codeBlockDataByteSizes[t] * nCodeBlocks[t] - 3,
                                                  G_CRC_24_A,
                                                  0,
                                                  1);
        goldenTBCRCs[t] = tbcrc;

        // Compare standalone per-TB CRC w/ per-TB CRC inserted in code blocks buffer
        // Compute pointer to per-TB CRC in the CB buffer. Subtract 6 because of 3B for per-TB CRC
        // and 3 for per-CB CRC.
        uint8_t* tmp           = (uint8_t*)codeBlocks + nCodeBlocks[t] * totalCodeBlockByteSizes[t] + tbOutBytes - 6;
        uint32_t gpu_perTB_crc = 0;
        for(int byte_id = 0; byte_id < 3; byte_id++)
        {
            gpu_perTB_crc |= ((*(tmp + byte_id)) << (byte_id * 8)); // Assume CRC written least significant byte of 24bits first (lower addr)
        }

        if(tbcrc != gpu_perTB_crc)
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "TB {}: Standalone per-TB CRC {} mismatch w/ CRC inserted in CB buffer {}", t, tbcrc, gpu_perTB_crc);
            passed = 0;
        }

        // compute CB crcs
        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            uint8_t* cbPtr = (uint8_t*)codeBlocks + i * totalCodeBlockByteSizes[t] + tbOutBytes;
            // TODO It'd be better not to use the buffer (pointed by cbPtr) w/ the GPU generated CRCs to compute the CPU CRCs.
            uint32_t crc = computeCRC<uint32_t, 24>((uint8_t*)cbPtr,
                                                    codeBlockDataByteSizes[t],
                                                    G_CRC_24_B,
                                                    0,
                                                    1);

            goldenCRCs[t * nCodeBlocks[t] + i] = crc;

            // Also compare the per-CB CRCs CRC written in the CB buffer w/ the standalone per-CB CRC.
            uint8_t* tmp           = cbPtr + codeBlockDataByteSizes[t];
            uint32_t gpu_perCB_crc = 0;
            for(int byte_id = 0; byte_id < 3; byte_id++)
            {
                gpu_perCB_crc |= ((*(tmp + byte_id)) << (byte_id * 8)); // Assume CRC written least significant byte of 24bits first (lower addr)
            }
            if(crc != gpu_perCB_crc)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "TB {}, CB {}: Standalone per-CB CRC %x mismatch w/ CRC inserted in CB buffer {}", t, i, crc, gpu_perCB_crc);
                passed = 0;
            }
        }
        tbBytes += tbPaddedByteSizes[t];
        tbOutBytes += nCodeBlocks[t] * totalCodeBlockByteSizes[t];
    }

    if(status != CUPHY_STATUS_SUCCESS)
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "CRC: CUPHY ERROR");
    else
    {
        // uint32_t* gt = (uint32_t*)goldenTransportBlocks;
        passed &= equalCount(crcs, goldenCRCs, totalNCodeBlocks, "CB CRC ");
        // passed &= equalCount(transportBlocks, gt, totalTBPaddedByteSize / ratio, "TB DATA");

        for(int i = 0; i < nTBs; i++)
        {
            passed &= (tbCRCs[i] == goldenTBCRCs[i]);
            if(tbCRCs[i] != goldenTBCRCs[i])
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "TB[{}] CRC: {} not equal to goldenTBCRCs: {}", i, tbCRCs[i], goldenTBCRCs[i]);
        }
    }

    delete[] nCodeBlocks;
    delete[] crcByteSizes;
    delete[] codeBlockByteSizes;
    delete[] codeBlockDataByteSizes;
    delete[] cbPaddingByteSizes;
    delete[] fillerByteSizes;
    delete[] tensorStrideByteSizes;
    delete[] totalCodeBlockByteSizes;
    delete[] goldenTransportBlocks;
    delete[] crcs;
    delete[] tbCRCs;
    delete[] codeBlocks;

    CUDA_CHECK(cudaFreeHost(h_crc_encode_desc));

    return passed;
}

int CRC_SINGLE_CB_GPU_UPLINK_PUSCH_TEST(bool timeIt)
{
    uint32_t  nTBs                    = MAX_N_TBS_SUPPORTED;
    uint32_t* nCodeBlocks             = new uint32_t[nTBs];
    uint32_t  crcByteSize             = 3; // 24 bits
    uint32_t* firstCodeBlockIdxArray  = new uint32_t[nTBs];
    uint32_t* codeBlockByteSizes      = new uint32_t[nTBs]; // 56;
    uint32_t* codeBlockWordSizes      = new uint32_t[nTBs]; // 56;
    uint32_t* codeBlockDataByteSizes  = new uint32_t[nTBs]; // 1053;
    uint32_t* CBPaddingByteSizes      = new uint32_t[nTBs]; // pad to 32-bit boundary
    uint32_t* totalCodeBlockByteSizes = new uint32_t[nTBs];
    uint32_t  totalByteSize           = 0;
    uint32_t  totalNCodeBlocks        = 0;
    uint32_t* tbPaddedByteSizes       = new uint32_t[nTBs];
    uint32_t  totalTBPaddedByteSize   = 0;
    uint32_t  ratio                   = sizeof(uint32_t) / sizeof(uint8_t);
    // Same CRC value for each code block, code blocks are all the same
    // linear input layout : cb1|crc1, cb2|crc2, ...

    for(int i = 0; i < nTBs; i++)
    {
        if(i == 2)
        {
            codeBlockByteSizes[i] = 1056;
            nCodeBlocks[i]        = 1;
        }
        codeBlockByteSizes[i] = 1000;
        nCodeBlocks[i]        = 1;
        totalNCodeBlocks += nCodeBlocks[i];
        codeBlockDataByteSizes[i]  = codeBlockByteSizes[i] - crcByteSize;
        CBPaddingByteSizes[i]      = (MAX_BYTES_PER_CODE_BLOCK - (codeBlockByteSizes[i] % MAX_BYTES_PER_CODE_BLOCK)) % MAX_BYTES_PER_CODE_BLOCK;
        totalCodeBlockByteSizes[i] = codeBlockByteSizes[i] + CBPaddingByteSizes[i];
        codeBlockWordSizes[i]      = totalCodeBlockByteSizes[i] / ratio;
        totalByteSize += totalCodeBlockByteSizes[i] * nCodeBlocks[i];
        tbPaddedByteSizes[i] = (codeBlockDataByteSizes[i] + (nCodeBlocks[i] == 1 ? crcByteSize : 0)) * nCodeBlocks[i] +
                               (4 - (nCodeBlocks[i] * (codeBlockDataByteSizes[i] + (nCodeBlocks[i] == 1 ? crcByteSize : 0)) % 4)) % 4;
        totalTBPaddedByteSize += tbPaddedByteSizes[i];
    }

    PerTbParams* tbPrmsArray           = new PerTbParams[nTBs];
    uint8_t*     linearInput           = new uint8_t[totalByteSize];
    uint32_t*    goldenCRCs            = new uint32_t[totalNCodeBlocks];
    uint8_t*     goldenTransportBlocks = new uint8_t[totalTBPaddedByteSize];
    uint32_t*    transportBlocks       = new uint32_t[totalTBPaddedByteSize / ratio];
    uint32_t*    codeBlocks            = (uint32_t*)linearInput;
    uint32_t*    crcs                  = new uint32_t[totalNCodeBlocks];
    uint32_t*    tbCRCs                = new uint32_t[nTBs];
    memset(goldenTransportBlocks, 0, totalTBPaddedByteSize);
    memset(firstCodeBlockIdxArray, 0, nTBs * sizeof(uint32_t));
    uint32_t tbBytes      = 0;
    uint32_t totalCBBytes = 0;
    for(int t = 0; t < nTBs; t++)
    {
        // Build transport block
        uint32_t cbBytes = 0;
        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            memset(goldenTransportBlocks + tbBytes + cbBytes,
                   rand(),
                   codeBlockDataByteSizes[t]);
            cbBytes += codeBlockDataByteSizes[t];
        }

        // just compute CB crcs using TB polynomial, as TBs contain only one CB
        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            uint8_t* cbPtr  = linearInput + i * totalCodeBlockByteSizes[t] + totalCBBytes;
            uint8_t* crcPtr = (cbPtr + codeBlockDataByteSizes[t]);
            memcpy(cbPtr,
                   goldenTransportBlocks + i * codeBlockDataByteSizes[t] + tbBytes,
                   codeBlockDataByteSizes[t]);
            uint32_t crc = computeCRC<uint32_t, 24>((uint8_t*)cbPtr,
                                                    codeBlockDataByteSizes[t],
                                                    G_CRC_24_A,
                                                    0,
                                                    1);
            for(int j = 0; j < crcByteSize; j++)
                crcPtr[j] = (crc >> (crcByteSize - 1 - j) * 8) & 0xFF;
            for(int j = 0; j < crcByteSize; j++)
                goldenTransportBlocks[tbBytes + nCodeBlocks[t] * codeBlockDataByteSizes[t] /*- crcByteSizes[t]*/ + j] =
                    (crc >> (crcByteSize - 1 - j) * 8) & 0xFF;

            goldenCRCs[t * nCodeBlocks[t] + i] = 0;
            memset(cbPtr + codeBlockByteSizes[t], 0, CBPaddingByteSizes[t]);
        }
        tbPrmsArray[t].num_CBs             = nCodeBlocks[t];
        tbPrmsArray[t].K                   = codeBlockByteSizes[t] * 8;
        tbPrmsArray[t].F                   = 0;
        tbPrmsArray[t].firstCodeBlockIndex = 0;
        tbBytes += tbPaddedByteSizes[t];
        totalCBBytes += nCodeBlocks[t] * totalCodeBlockByteSizes[t];
    }
#if 0
    std::cout << "CBs:\n";
    for (int i = 0; i < totalByteSize; i++)
        std::cout << std::hex << (unsigned short)linearInput[i] << ",";
    std::cout << "\n";

    std::cout << "TB:\n";
    for (int i = 0; i < totalTBPaddedByteSize; i++)
        std::cout << std::hex << (unsigned short)goldenTransportBlocks[i] << ",";
    std::cout << "\n";
#endif
    /*
       cuphyStatus_t status = cuphyCRCDecode(
       crcs, tbCRCs, transportBlocks, nTBs, (const uint32_t*)codeBlocks,
       nCodeBlocks, codeBlockWordSizes, codeBlockDataByteSizes, timeIt, 10000);
     */

    //input
    unique_device_ptr<uint32_t>    d_codeBlocks  = make_unique_device<uint32_t>(totalByteSize / sizeof(uint32_t));
    unique_device_ptr<PerTbParams> d_tbPrmsArray = make_unique_device<PerTbParams>(nTBs);

    //output

    unique_device_ptr<uint32_t> d_CBCRCs = make_unique_device<uint32_t>(nTBs * nCodeBlocks[0]);
    unique_device_ptr<uint32_t> d_TBCRCs = make_unique_device<uint32_t>(nTBs);
    unique_device_ptr<uint8_t>  d_TBs    = make_unique_device<uint8_t>(nTBs * tbPaddedByteSizes[0]);

    cudaMemcpy(d_codeBlocks.get(), codeBlocks, totalByteSize, cudaMemcpyHostToDevice);

    cudaMemcpy(d_tbPrmsArray.get(), tbPrmsArray, sizeof(PerTbParams) * nTBs, cudaMemcpyHostToDevice);

    cuphyStatus_t status = cuphyCRCDecode(
        d_CBCRCs.get(),
        d_TBCRCs.get(),
        d_TBs.get(),
        d_codeBlocks.get(),
        d_tbPrmsArray.get(),
        nTBs,
        nCodeBlocks[0],
        tbPaddedByteSizes[0],
        0,
        timeIt,
        0,
        false,
        0);

    cudaMemcpy(crcs, d_CBCRCs.get(), sizeof(uint32_t) * totalNCodeBlocks, cudaMemcpyDeviceToHost);

    cudaMemcpy(tbCRCs, d_TBCRCs.get(), sizeof(uint32_t) * nTBs, cudaMemcpyDeviceToHost);

    cudaMemcpy(transportBlocks, d_TBs.get(), totalTBPaddedByteSize, cudaMemcpyDeviceToHost);

    int passed = 0;
    if(status != CUPHY_STATUS_SUCCESS)
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "CRC: CUPHY ERROR");
    else
    {
        uint32_t* gt = (uint32_t*)goldenTransportBlocks;
        passed       = equalCount(crcs, goldenCRCs, totalNCodeBlocks, "CB CRC ");
        passed &= equalCount(transportBlocks, gt, totalTBPaddedByteSize / ratio, "TB DATA ");

        for(int i = 0; i < nTBs; i++)
        {
            passed &= (tbCRCs[i] == 0);
            if(tbCRCs[i] != 0)
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "TB[{}] CRC not equal to 0: {}", i, tbCRCs[i]);
        }
    }

    delete[] tbPrmsArray;
    delete[] crcs;
    delete[] tbCRCs;
    delete[] goldenCRCs;
    delete[] goldenTransportBlocks;
    delete[] linearInput;
    delete[] transportBlocks;
    delete[] firstCodeBlockIdxArray;
    delete[] nCodeBlocks;
    delete[] codeBlockByteSizes;
    delete[] codeBlockWordSizes;
    delete[] codeBlockDataByteSizes;
    delete[] CBPaddingByteSizes; // pad to 32-bit boundary
    delete[] totalCodeBlockByteSizes;
    delete[] tbPaddedByteSizes;

    return passed;
}

int CRC_SINGLE_SMALL_CB_GPU_UPLINK_PUSCH_TEST(bool timeIt)
{
    uint32_t  nTBs                    = MAX_N_TBS_SUPPORTED;
    uint32_t* firstCodeBlockIdxArray  = new uint32_t[nTBs];
    uint32_t* nCodeBlocks             = new uint32_t[nTBs];
    uint32_t  crcByteSize             = 2;                  // 24 bits
    uint32_t* codeBlockByteSizes      = new uint32_t[nTBs]; // 56;
    uint32_t* codeBlockWordSizes      = new uint32_t[nTBs]; // 56;
    uint32_t* codeBlockDataByteSizes  = new uint32_t[nTBs]; // 1053;
    uint32_t* CBPaddingByteSizes      = new uint32_t[nTBs]; // pad to 32-bit boundary
    uint32_t* totalCodeBlockByteSizes = new uint32_t[nTBs];
    uint32_t  totalByteSize           = 0;
    uint32_t  totalNCodeBlocks        = 0;
    uint32_t* tbPaddedByteSizes       = new uint32_t[nTBs];
    uint32_t  totalTBPaddedByteSize   = 0;
    uint32_t  ratio                   = sizeof(uint32_t) / sizeof(uint8_t);
    // Same CRC value for each code block, code blocks are all the same
    // linear input layout : cb1|crc1, cb2|crc2, ...

    for(int i = 0; i < nTBs; i++)
    {
        if(i == 2)
        {
            codeBlockByteSizes[i] = 333;
            nCodeBlocks[i]        = 1;
        }
        codeBlockByteSizes[i] = 476;
        nCodeBlocks[i]        = 1;
        totalNCodeBlocks += nCodeBlocks[i];
        codeBlockDataByteSizes[i]  = codeBlockByteSizes[i] - crcByteSize;
        CBPaddingByteSizes[i]      = (MAX_BYTES_PER_CODE_BLOCK - (codeBlockByteSizes[i] % MAX_BYTES_PER_CODE_BLOCK)) % MAX_BYTES_PER_CODE_BLOCK;
        totalCodeBlockByteSizes[i] = codeBlockByteSizes[i] + CBPaddingByteSizes[i];
        codeBlockWordSizes[i]      = totalCodeBlockByteSizes[i] / ratio;
        totalByteSize += totalCodeBlockByteSizes[i] * nCodeBlocks[i];
        tbPaddedByteSizes[i] = (codeBlockDataByteSizes[i] + (nCodeBlocks[i] == 1 ? crcByteSize : 0)) * nCodeBlocks[i] +
                               (4 - (nCodeBlocks[i] * (codeBlockDataByteSizes[i] + (nCodeBlocks[i] == 1 ? crcByteSize : 0)) % 4)) % 4;
        totalTBPaddedByteSize += tbPaddedByteSizes[i];
    }

    PerTbParams* tbPrmsArray           = new PerTbParams[nTBs];
    uint8_t*     linearInput           = new uint8_t[totalByteSize];
    uint32_t*    goldenCRCs            = new uint32_t[totalNCodeBlocks];
    uint8_t*     goldenTransportBlocks = new uint8_t[totalTBPaddedByteSize];
    uint32_t*    transportBlocks       = new uint32_t[totalTBPaddedByteSize / ratio];
    uint32_t*    codeBlocks            = (uint32_t*)linearInput;
    uint32_t*    crcs                  = new uint32_t[totalNCodeBlocks];
    uint32_t*    tbCRCs                = new uint32_t[nTBs];
    memset(goldenTransportBlocks, 0, totalTBPaddedByteSize);
    memset(firstCodeBlockIdxArray, 0, nTBs * sizeof(uint32_t));

    uint32_t tbBytes      = 0;
    uint32_t totalCBBytes = 0;
    for(int t = 0; t < nTBs; t++)
    {
        // Build transport block
        uint32_t cbBytes = 0;
        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            memset(goldenTransportBlocks + tbBytes + cbBytes,
                   rand(),
                   codeBlockDataByteSizes[t]);
            cbBytes += codeBlockDataByteSizes[t];
        }

        // just compute CB crcs using TB polynomial, as TBs contain only one CB
        for(int i = 0; i < nCodeBlocks[t]; i++)
        {
            uint8_t* cbPtr  = linearInput + i * totalCodeBlockByteSizes[t] + totalCBBytes;
            uint8_t* crcPtr = (cbPtr + codeBlockDataByteSizes[t]);
            memcpy(cbPtr,
                   goldenTransportBlocks + i * codeBlockDataByteSizes[t] + tbBytes,
                   codeBlockDataByteSizes[t]);
            uint32_t crc = computeCRC<uint32_t, 16>((uint8_t*)cbPtr,
                                                    codeBlockDataByteSizes[t],
                                                    G_CRC_16,
                                                    0,
                                                    1);
            for(int j = 0; j < crcByteSize; j++)
                crcPtr[j] = (crc >> (crcByteSize - 1 - j) * 8) & 0xFF;
            for(int j = 0; j < crcByteSize; j++)
                goldenTransportBlocks[tbBytes + nCodeBlocks[t] * codeBlockDataByteSizes[t] /*- crcByteSizes[t]*/ + j] =
                    (crc >> (crcByteSize - 1 - j) * 8) & 0xFF;
            goldenCRCs[t * nCodeBlocks[t] + i] = 0;
            memset(cbPtr + codeBlockByteSizes[t], 0, CBPaddingByteSizes[t]);
        }

        tbPrmsArray[t].num_CBs             = nCodeBlocks[t];
        tbPrmsArray[t].K                   = codeBlockByteSizes[t] * 8;
        tbPrmsArray[t].F                   = 0;
        tbPrmsArray[t].firstCodeBlockIndex = 0;
        tbBytes += tbPaddedByteSizes[t];
        totalCBBytes += nCodeBlocks[t] * totalCodeBlockByteSizes[t];
    }
#if 0
    std::cout << "CBs:\n";
    for (int i = 0; i < totalByteSize; i++)
        std::cout << std::hex << (unsigned short)linearInput[i] << ",";
    std::cout << "\n";

    std::cout << "TB:\n";
    for (int i = 0; i < totalTBPaddedByteSize; i++)
        std::cout << std::hex << (unsigned short)goldenTransportBlocks[i] << ",";
    std::cout << "\n";
#endif
    /*
       cuphyStatus_t status = cuphyCRCDecode(
       crcs, tbCRCs, transportBlocks, nTBs, (const uint32_t*)codeBlocks,
       nCodeBlocks, codeBlockWordSizes, codeBlockDataByteSizes, timeIt, 10000);
     */

    //input
    unique_device_ptr<uint32_t>    d_codeBlocks  = make_unique_device<uint32_t>(totalByteSize / sizeof(uint32_t));
    unique_device_ptr<PerTbParams> d_tbPrmsArray = make_unique_device<PerTbParams>(nTBs);

    //output

    unique_device_ptr<uint32_t> d_CBCRCs = make_unique_device<uint32_t>(nTBs * nCodeBlocks[0]);
    unique_device_ptr<uint32_t> d_TBCRCs = make_unique_device<uint32_t>(nTBs);
    unique_device_ptr<uint8_t>  d_TBs    = make_unique_device<uint8_t>(nTBs * tbPaddedByteSizes[0]);

    cudaMemcpy(d_codeBlocks.get(), codeBlocks, totalByteSize, cudaMemcpyHostToDevice);

    cudaMemcpy(d_tbPrmsArray.get(), tbPrmsArray, sizeof(PerTbParams) * nTBs, cudaMemcpyHostToDevice);

    cuphyStatus_t status = cuphyCRCDecode(
        d_CBCRCs.get(),
        d_TBCRCs.get(),
        d_TBs.get(),
        d_codeBlocks.get(),
        d_tbPrmsArray.get(),
        nTBs,
        nCodeBlocks[0],
        tbPaddedByteSizes[0],
        0,
        timeIt,
        0,
        false,
        0);

    cudaMemcpy(crcs, d_CBCRCs.get(), sizeof(uint32_t) * totalNCodeBlocks, cudaMemcpyDeviceToHost);

    cudaMemcpy(tbCRCs, d_TBCRCs.get(), sizeof(uint32_t) * nTBs, cudaMemcpyDeviceToHost);

    cudaMemcpy(transportBlocks, d_TBs.get(), totalTBPaddedByteSize, cudaMemcpyDeviceToHost);

    int passed = 0;
    if(status != CUPHY_STATUS_SUCCESS)
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "CRC: CUPHY ERROR");
    else
    {
        uint32_t* gt = (uint32_t*)goldenTransportBlocks;
        passed       = equalCount(crcs, goldenCRCs, totalNCodeBlocks, "CB CRC ");
        passed &= equalCount(transportBlocks, gt, totalTBPaddedByteSize / ratio, "TB DATA ");

        for(int i = 0; i < nTBs; i++)
        {
            passed &= (tbCRCs[i] == 0);
            if(tbCRCs[i] != 0)
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "TB[{}] CRC not equal to 0: {}", i, tbCRCs[i]);
        }
    }

    delete[] tbPrmsArray;
    delete[] crcs;
    delete[] tbCRCs;
    delete[] goldenCRCs;
    delete[] goldenTransportBlocks;
    delete[] linearInput;
    delete[] transportBlocks;
    delete[] nCodeBlocks;
    delete[] firstCodeBlockIdxArray;
    delete[] codeBlockByteSizes;
    delete[] codeBlockWordSizes;
    delete[] codeBlockDataByteSizes;
    delete[] CBPaddingByteSizes; // pad to 32-bit boundary
    delete[] totalCodeBlockByteSizes;
    delete[] tbPaddedByteSizes;

    return passed;
}

TEST(CRC_GPU_UPLINK_PUSCH, 24B_24A)
{
    EXPECT_EQ(CRC_GPU_UPLINK_PUSCH_TEST(false), 1);
}

TEST(CRC_SINGLE_CB_GPU_UPLINK_PUSCH, 24A)
{
    EXPECT_EQ(CRC_SINGLE_CB_GPU_UPLINK_PUSCH_TEST(false), 1);
}

TEST(CRC_SINGLE_SMALL_CB_GPU_UPLINK_PUSCH, 16)
{
    EXPECT_EQ(CRC_SINGLE_SMALL_CB_GPU_UPLINK_PUSCH_TEST(false), 1);
}

//int cb_num_array[20] = {51, 44, 26, 18, 10, 383, 330, 195, 135, 75, 501, 429, 251, 174, 91, 752, 644, 377, 261, 137};
TEST(CRC_GPU_DOWNLINK_PDSCH, 24B_24A)
{
    /*for (int i = 0; i < 20; i++){
        EXPECT_EQ(CRC_GPU_DOWNLINK_PDSCH(false, cb_num_array[i]), 1); // FIXME the 2nd argument is not used.
    }*/
    EXPECT_EQ(CRC_GPU_DOWNLINK_PDSCH(false), 1);
}

int main(int argc, char** argv)
{
    testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
