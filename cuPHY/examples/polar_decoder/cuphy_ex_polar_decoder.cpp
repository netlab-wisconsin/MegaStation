/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "cuphy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "datasets.hpp"

#include "cuphy_internal.h"
#include "utils.cuh"

#include <cstring>
#include <iostream>
#include <unistd.h> // for getcwd()
#include <dirent.h> // opendir, readdir
#include <errno.h>
#include <sys/stat.h> // for mkdir

////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("polar_decoder [options]\n");
    printf("  Options:\n");
    printf("    -i  Input HDF5 filename\n");
    printf("    -L  list size for polar decoder\n");
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    int polarListSz = 1;
    cuphyNvlogFmtHelper nvlog_fmt("polar_decoder.log");
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int         iArg          = 1;
        std::string inputFilename = std::string();

        while(iArg < argc)
        {
            if('-' == argv[iArg][0])
            {
                switch(argv[iArg][1])
                {
                case 'i':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided");
                    }
                    inputFilename.assign(argv[iArg++]);
                    break;
                case 'L':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &polarListSz)) || ((polarListSz != 1) && (polarListSz != 2) && (polarListSz != 4) && (polarListSz != 8)))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid list size {}, list size for Polar decoder can be 1, 2, 4, or 8", polarListSz);
                        exit(1);
                    }
                    ++iArg;
                    break;
                default:
                    NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                    usage();
                    exit(1);
                }
            }
            else
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
                exit(1);
            }
        }
        if(inputFilename.empty())
        {
            usage();
            exit(1);
        }

        // ---------------------------------------------------------------
        // Initialize main stream

        cuphy::stream cuStrmMain;

        //-----------------------------------------------------------------
        // Allocate GPU memory

        size_t max_cwPrms_mem      = sizeof(cuphyPolarCwPrm_t) * CUPHY_MAX_N_POL_CWS;
        size_t max_nCwBits         = CUPHY_MAX_N_POL_CWS * 1024;
        size_t max_cwTreeLLRs_mem  = (2 * max_nCwBits) * polarListSz * sizeof(__half);
        size_t max_N_cw            = CUPHY_POLAR_DECODER_MAX_BITS;
        size_t max_cwTreeTypes_mem = sizeof(uint8_t) * (2 * max_N_cw) * polarListSz * CUPHY_MAX_N_POL_UCI_SEGS;
        size_t max_scratch_buffer  = polarListSz > 1 ? sizeof(bool) * (2 * max_N_cw) * polarListSz : 0;
        size_t max_out_mem         = (max_N_cw / 32) * CUPHY_MAX_N_POL_CWS * sizeof(uint32_t);
        size_t max_mem             = max_cwPrms_mem + max_cwTreeLLRs_mem + max_cwTreeTypes_mem + max_scratch_buffer + max_out_mem;

        cuphy::linear_alloc<128, cuphy::device_alloc> linearAlloc(max_mem);

        //------------------------------------------------------------------
        // Load uci polar dataset

        UciPolarDataset uciPolarDataset(inputFilename, cuStrmMain.handle());
        cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied

        //----------------------------------------------------------------------
        // GPU input buffers

        uint16_t               nPolUciSegs         = uciPolarDataset.nPolUciSegs;
        cuphyPolarUciSegPrm_t* pPolarUciSegPrmsCpu = uciPolarDataset.polUciSegPrmsVec.data();

        uint16_t              nPolCws = 0;
        std::vector<__half*>  cwTreeLLRsGpuAddrVec(CUPHY_MAX_N_POL_CWS);
        std::vector<uint8_t*> cwTreeTypesGpuAddrVec(CUPHY_MAX_N_POL_UCI_SEGS);

        for(int segIdx = 0; segIdx < nPolUciSegs; ++segIdx)
        {
            uint16_t  N_cw               = pPolarUciSegPrmsCpu[segIdx].N_cw;
            uint8_t   nCbs               = pPolarUciSegPrmsCpu[segIdx].nCbs;
            size_t    nBytesCwTree       = sizeof(uint8_t) * (2 * N_cw);
            size_t    nBytesCwLLRs       = sizeof(__half) * N_cw;
            size_t    nBytesCwTreeLLRs   = sizeof(__half) * (2 * N_cw) * polarListSz; // more storage is required for list decoder, proportional to the list size
            const int treeTypesIdxOffset = 2;

            cwTreeTypesGpuAddrVec[segIdx] = static_cast<uint8_t*>(linearAlloc.alloc(nBytesCwTree));
            cudaMemcpyAsync(cwTreeTypesGpuAddrVec[segIdx] + treeTypesIdxOffset, uciPolarDataset.refCwTreeTypesVec[segIdx].addr(), nBytesCwTree - treeTypesIdxOffset * sizeof(uint8_t), cudaMemcpyHostToDevice, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());

            for(int i = 0; i < nCbs; ++i)
            {
                cwTreeLLRsGpuAddrVec[nPolCws] = static_cast<__half*>(linearAlloc.alloc(nBytesCwTreeLLRs));
                uint16_t offset               = N_cw;
                // LLRs are copied only for the first decoder in the list
                cudaMemcpyAsync(cwTreeLLRsGpuAddrVec[nPolCws] + offset, uciPolarDataset.refCwLLRsVec[nPolCws].addr(), nBytesCwLLRs, cudaMemcpyHostToDevice, cuStrmMain.handle());
                cudaStreamSynchronize(cuStrmMain.handle());
                nPolCws += 1;
            }
        }

        //----------------------------------------------------------------------
        // GPU output buffers

        uint8_t*               pCrcErrorFlags = static_cast<uint8_t*>(linearAlloc.alloc(nPolCws));
        std::vector<uint32_t*> cbEstsGpuAddrVec(nPolCws);
        std::vector<uint32_t*> uciSegEstsGpuAddrVec(nPolUciSegs);

        uint32_t cbIdx = 0;
        for(int uciSegIdx = 0; uciSegIdx < nPolUciSegs; ++uciSegIdx)
        {
            uint32_t nUciSegBits  = pPolarUciSegPrmsCpu[uciSegIdx].nCbs * (pPolarUciSegPrmsCpu[uciSegIdx].K_cw - pPolarUciSegPrmsCpu[uciSegIdx].nCrcBits) - pPolarUciSegPrmsCpu[uciSegIdx].zeroInsertFlag;
            uint32_t nUciSegWords = div_round_up(nUciSegBits, static_cast<uint32_t>(32));

            uint32_t nDecodedCbBits  = pPolarUciSegPrmsCpu[uciSegIdx].K_cw - pPolarUciSegPrmsCpu[uciSegIdx].nCrcBits;
            uint32_t nDecodedCbWords = div_round_up(nUciSegBits, static_cast<uint32_t>(32));

            for(int cbIdxWithUciSeg = 0; cbIdxWithUciSeg < pPolarUciSegPrmsCpu[uciSegIdx].nCbs; ++cbIdxWithUciSeg)
            {
                cbEstsGpuAddrVec[cbIdx] = static_cast<uint32_t*>(linearAlloc.alloc(nDecodedCbWords * sizeof(uint32_t)));
                cbIdx++;
            }

            if(pPolarUciSegPrmsCpu[uciSegIdx].nCbs == 1)
            {
                uciSegEstsGpuAddrVec[uciSegIdx] = cbEstsGpuAddrVec[cbIdx - 1];
            }else
            {
                uciSegEstsGpuAddrVec[uciSegIdx] = static_cast<uint32_t*>(linearAlloc.alloc(nUciSegWords * sizeof(uint32_t)));
            }
        }

        // ----------------------------------------------------------------------
        // Codeword parameters

        std::vector<cuphyPolarCwPrm_t> cwPrmsCpuVec(nPolCws);
        uint16_t                       cwIdx = 0;

        for(int segIdx = 0; segIdx < nPolUciSegs; ++segIdx)
        {
            for(int i = 0; i < pPolarUciSegPrmsCpu[segIdx].nCbs; ++i)
            {
                cwPrmsCpuVec[cwIdx].N_cw         = pPolarUciSegPrmsCpu[segIdx].N_cw;
                cwPrmsCpuVec[cwIdx].nCrcBits     = pPolarUciSegPrmsCpu[segIdx].nCrcBits;
                cwPrmsCpuVec[cwIdx].A_cw         = pPolarUciSegPrmsCpu[segIdx].K_cw - cwPrmsCpuVec[cwIdx].nCrcBits;
                cwPrmsCpuVec[cwIdx].pCwTreeTypes = cwTreeTypesGpuAddrVec[segIdx];

                cwPrmsCpuVec[cwIdx].nCbsInUciSeg       = pPolarUciSegPrmsCpu[segIdx].nCbs;
                cwPrmsCpuVec[cwIdx].cbIdxWithinUciSeg  = i;
                cwPrmsCpuVec[cwIdx].zeroInsertFlag     = pPolarUciSegPrmsCpu[segIdx].zeroInsertFlag;
                cwPrmsCpuVec[cwIdx].pUciSegEst         = uciSegEstsGpuAddrVec[segIdx];

                cwIdx += 1;
            }
        }


        uint8_t   nCbsInUciSeg;      // number of codeblocks in parent UCI seg. 1 or 2
        uint8_t   cbIdxWithinUciSeg; // index of clodeblock within parent UCI seg. 0 or 1
        uint8_t   zeroInsertFlag;    // flag, indicates if zero inserted at start of first codeblock. 0 or 1
        uint32_t* pUciSegEst;        // pointer to estimated UCI segment (GPU).

        cuphyPolarCwPrm_t* pCwPrmsGpu = static_cast<cuphyPolarCwPrm_t*>(linearAlloc.alloc(nPolCws * sizeof(cuphyPolarCwPrm_t)));
        cudaMemcpyAsync(pCwPrmsGpu, cwPrmsCpuVec.data(), nPolCws * sizeof(cuphyPolarCwPrm_t), cudaMemcpyHostToDevice, cuStrmMain.handle());
        cudaStreamSynchronize(cuStrmMain.handle());

        //----------------------------------------------------------------------
        // GPU scratch buffers used in list decoder

        std::vector<bool*> listPolScratchGpuAddrVec;

        if (polarListSz > 1) {
            listPolScratchGpuAddrVec.resize(nPolCws);
            for(int cbIdx = 0; cbIdx < nPolCws; ++cbIdx)
            {
                size_t   nBytesScratch = sizeof(bool) * (2 * cwPrmsCpuVec[cbIdx].N_cw) * polarListSz;
                listPolScratchGpuAddrVec[cbIdx] = static_cast<bool*>(linearAlloc.alloc(nBytesScratch));
            }
        }

        //------------------------------------------------------------------
        // polarDecoder descriptors

        size_t dynDescrSizeBytes, dynDescrAlignBytes;

        // descriptors hold Kernel parameters in GPU
        cuphyStatus_t statusGetWorkspaceSize = cuphyPolarDecoderGetDescrInfo(&dynDescrSizeBytes,
                                                                             &dynDescrAlignBytes);
        if(CUPHY_STATUS_SUCCESS != statusGetWorkspaceSize) throw cuphy::cuphy_exception(statusGetWorkspaceSize);

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

        // for CW LLR addresses in CPU descriptor
        size_t   dynDescrLLRAddrsSizeBytes = sizeof(__half*) * nPolCws;
        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrLLRAddrsBufCpu(dynDescrLLRAddrsSizeBytes);

        // for CB estimate addresses in CPU descriptor
        size_t   dynDescrCbAddrsSizeBytes = sizeof(uint32_t*) * nPolCws;
        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrCbAddrsBufCpu(dynDescrCbAddrsSizeBytes);

        // for scratch buffer addresses for list decoder in CPU descriptor
        size_t   dynDescrScratchAddrsSizeBytes = sizeof(bool*) * nPolCws;
        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrScratchAddrsBufCpu(dynDescrScratchAddrsSizeBytes);

        //------------------------------------------------------------------
        // Create polarDecoder object

        cuphyPolarDecoderHndl_t polarDecoderHndl;
        cuphyStatus_t           statusCreate = cuphyCreatePolarDecoder(&polarDecoderHndl);

        if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);

        //------------------------------------------------------------------
        // setup polarDecoder object

        // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
        cuphyPolarDecoderLaunchCfg_t polarDecoderLaunchCfg;

        // setup function populates dynamic descriptor and launch config
        bool enableCpuToGpuDescrAsyncCpy = false;

        cuphyStatus_t polarDecoderSetupStatus = cuphySetupPolarDecoder(polarDecoderHndl,
                                                                       nPolCws,
                                                                       cwTreeLLRsGpuAddrVec.data(),
                                                                       pCwPrmsGpu,
                                                                       cwPrmsCpuVec.data(),
                                                                       cbEstsGpuAddrVec.data(),
                                                                       listPolScratchGpuAddrVec.data(),
                                                                       polarListSz,
                                                                       pCrcErrorFlags,
                                                                       static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                                       dynDescrBufCpu.addr(),
                                                                       dynDescrBufGpu.addr(),
                                                                       dynDescrLLRAddrsBufCpu.addr(),
                                                                       dynDescrCbAddrsBufCpu.addr(),
                                                                       dynDescrScratchAddrsBufCpu.addr(),
                                                                       &polarDecoderLaunchCfg,
                                                                       cuStrmMain.handle());

        if(CUPHY_STATUS_SUCCESS != polarDecoderSetupStatus) throw cuphy::cuphy_exception(polarDecoderSetupStatus);

        if(!enableCpuToGpuDescrAsyncCpy)
        {
            cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());
        }

        //------------------------------------------------------------------
        // run polarDecoder

        // launch kernel using the CUDA driver API
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = polarDecoderLaunchCfg.kernelNodeParamsDriver;
        CUresult                 polarDecoderRunStatus  = cuLaunchKernel(kernelNodeParamsDriver.func,
                                                        kernelNodeParamsDriver.gridDimX,
                                                        kernelNodeParamsDriver.gridDimY,
                                                        kernelNodeParamsDriver.gridDimZ,
                                                        kernelNodeParamsDriver.blockDimX,
                                                        kernelNodeParamsDriver.blockDimY,
                                                        kernelNodeParamsDriver.blockDimZ,
                                                        kernelNodeParamsDriver.sharedMemBytes,
                                                        static_cast<CUstream>(cuStrmMain.handle()),
                                                        kernelNodeParamsDriver.kernelParams,
                                                        kernelNodeParamsDriver.extra);
        if(CUDA_SUCCESS != polarDecoderRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
        cudaStreamSynchronize(cuStrmMain.handle()); // synch to make sure kernel finishes

        // -------------------------------------------------------------------
        // Evaluate decoder output

        uciPolarDataset.evalDecoderOutput(nPolCws, cwPrmsCpuVec.data(), cbEstsGpuAddrVec.data(), pCrcErrorFlags, nPolUciSegs, pPolarUciSegPrmsCpu, uciSegEstsGpuAddrVec.data(), cuStrmMain.handle());

        // -------------------------------------------------------------------
        // cleanup

        cuphyStatus_t statusDestroy = cuphyDestroyPolarDecoder(polarDecoderHndl);
        if(CUPHY_STATUS_SUCCESS != statusDestroy) throw cuphy::cuphy_exception(statusDestroy);
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    return returnValue;
}
