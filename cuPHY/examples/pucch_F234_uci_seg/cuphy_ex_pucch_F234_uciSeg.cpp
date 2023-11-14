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
#include "test_config.hpp"

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
    printf("pucch_F234_uciSeg [options]\n");
    printf("  Options:\n");
    printf("    -i  input_filename         Input yaml file for slot/cell config or HDF5 file for single cell example\n");
    printf("    -G                         Enable graph processing mode\n");
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    cuphyNvlogFmtHelper nvlog_fmt("pucch_F234_uciSeg.log");
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int         iArg = 1;
        std::string inputFilename  = std::string();
        uint64_t    procModeBmsk   = 0;

        while(iArg < argc)
        {
            if('-' == argv[iArg][0])
            {
                switch(argv[iArg][1])
                {
                case 'i':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided");
                    }
                    inputFilename.assign(argv[iArg++]);
                    break;
                case 'G':
                    ++iArg;
                    procModeBmsk = PUCCH_PROC_MODE_FULL_SLOT_GRAPHS;
                    NVLOGI_FMT(NVLOG_PUCCH, "CUDA graph enabled!");
                    break;
                default:
                    NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                    usage();
                    exit(1);
                    break;
                }
            }
            else
            {
                NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
                exit(1);
            }
        }
        if(inputFilename.empty())
        {
            usage();
            exit(1);
        }

        //------------------------------------------------------------------
        // input files
        std::vector<std::string> inputFileNameVec;
        std::string              inFileExtn = inputFilename.substr(inputFilename.find_last_of(".") + 1);
        if(inFileExtn == "yaml")
        {
            // yaml parsing
            cuphy::test_config testCfg(inputFilename.c_str());
            int                nCells           = testCfg.num_cells();
            int                nSlots           = testCfg.num_slots();
            const std::string  pucchChannelName = "PUCCH";

            try
            {
                for(size_t idxSlot = 0; idxSlot < nSlots; idxSlot++)
                {
                    for(int idxCell = 0; idxCell < nCells; idxCell++)
                    {
                        auto fname = testCfg.slots()[idxSlot].at(pucchChannelName)[idxCell];
                        inputFileNameVec.emplace_back(fname);
                    }
                }
            }
            catch(...)
            {
                throw std::runtime_error("PUCCH channel name not found in the input file");
            }
            assert(inputFileNameVec.size() == nCells);
        }
        else
        {
            inputFileNameVec.emplace_back(inputFilename);
        }

        //------------------------------------------------------------------
        // Load API parameters
        cuphy::stream cuStrmMain;
        
        pucchStaticApiDataset  statPucchApiDataset(inputFileNameVec, cuStrmMain.handle());
        pucchDynApiDataset     dynPucchApiDataset (inputFileNameVec, cuStrmMain.handle(), procModeBmsk);
        EvalPucchDataset       evalPucchDataset   (inputFileNameVec, cuStrmMain.handle());

        cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied

        uint16_t nF2Ucis = evalPucchDataset.nF2Ucis;
        uint16_t nF3Ucis = evalPucchDataset.nF3Ucis;

        if ((nF2Ucis == 0) && (nF3Ucis == 0)) {
            printf("No PF2/3/4 UCI exists.\n");
            return returnValue;
        }
        cuphyPucchF234OutOffsets_t* F2OutOffsetsCpu = new cuphyPucchF234OutOffsets_t[nF2Ucis];
        cuphyPucchF234OutOffsets_t* F3OutOffsetsCpu = new cuphyPucchF234OutOffsets_t[nF3Ucis];
        cuphyPucchUciPrm_t* F2UciPrms               = new cuphyPucchUciPrm_t[nF2Ucis];
        cuphyPucchUciPrm_t* F3UciPrms               = new cuphyPucchUciPrm_t[nF3Ucis];

        int numBytesUciPayloads = evalPucchDataset.tRefPayloadBytes[0].desc().get_size_in_bytes();
        uint8_t* uciPayloadsCpu = new uint8_t[numBytesUciPayloads];

        size_t max_mem = numBytesUciPayloads * sizeof(uint8_t);
        memcpy(uciPayloadsCpu, evalPucchDataset.tRefPayloadBytes[0].addr(), max_mem);// [0] - consider a single cell
        
        cuphy::linear_alloc<128, cuphy::device_alloc> linearAlloc(max_mem);
        uint8_t* uciPayloadsGpu = static_cast<uint8_t*>(linearAlloc.alloc(numBytesUciPayloads * sizeof(uint8_t)));

        for (int uciIdx = 0; uciIdx < nF2Ucis; uciIdx++) {
            F2OutOffsetsCpu[uciIdx].uciSeg1PayloadByteOffset = evalPucchDataset.pucchF2bufferOffsetsVec[uciIdx].uciSeg1PayloadByteOffset;
            F2OutOffsetsCpu[uciIdx].harqPayloadByteOffset    = evalPucchDataset.pucchF2bufferOffsetsVec[uciIdx].harqPayloadByteOffset;
            F2OutOffsetsCpu[uciIdx].srPayloadByteOffset      = evalPucchDataset.pucchF2bufferOffsetsVec[uciIdx].srPayloadByteOffset;
            F2OutOffsetsCpu[uciIdx].csi1PayloadByteOffset    = evalPucchDataset.pucchF2bufferOffsetsVec[uciIdx].csiP1PayloadByteOffset;

            uint32_t nHarqBytes  = evalPucchDataset.pucchF2bufferOffsetsVec[uciIdx].nHarqBytes;
            uint32_t nSrBytes    = evalPucchDataset.pucchF2bufferOffsetsVec[uciIdx].nSrBytes;
            uint32_t nCsiP1Bytes = evalPucchDataset.pucchF2bufferOffsetsVec[uciIdx].nCsiP1Bytes;

            for (int byteIdx = 0; byteIdx < nHarqBytes; byteIdx++) {
                uciPayloadsCpu[F2OutOffsetsCpu[uciIdx].harqPayloadByteOffset+byteIdx] = 0;
            }
            for (int byteIdx = 0; byteIdx < nSrBytes; byteIdx++) {
                uciPayloadsCpu[F2OutOffsetsCpu[uciIdx].srPayloadByteOffset+byteIdx] = 0;
            }
            for (int byteIdx = 0; byteIdx < nCsiP1Bytes; byteIdx++) {
                uciPayloadsCpu[F2OutOffsetsCpu[uciIdx].csi1PayloadByteOffset+byteIdx] = 0;
            }

            F2UciPrms[uciIdx].bitLenHarq                     = dynPucchApiDataset.F2UciPrmsVec[uciIdx].bitLenHarq;
            F2UciPrms[uciIdx].bitLenSr                       = dynPucchApiDataset.F2UciPrmsVec[uciIdx].bitLenSr;
            F2UciPrms[uciIdx].bitLenCsiPart1                 = dynPucchApiDataset.F2UciPrmsVec[uciIdx].bitLenCsiPart1;
        }
        for (int uciIdx = 0; uciIdx < nF3Ucis; uciIdx++) {
            F3OutOffsetsCpu[uciIdx].uciSeg1PayloadByteOffset = evalPucchDataset.pucchF3bufferOffsetsVec[uciIdx].uciSeg1PayloadByteOffset;
            F3OutOffsetsCpu[uciIdx].harqPayloadByteOffset    = evalPucchDataset.pucchF3bufferOffsetsVec[uciIdx].harqPayloadByteOffset;
            F3OutOffsetsCpu[uciIdx].srPayloadByteOffset      = evalPucchDataset.pucchF3bufferOffsetsVec[uciIdx].srPayloadByteOffset;
            F3OutOffsetsCpu[uciIdx].csi1PayloadByteOffset    = evalPucchDataset.pucchF3bufferOffsetsVec[uciIdx].csiP1PayloadByteOffset;

            uint32_t nHarqBytes  = evalPucchDataset.pucchF3bufferOffsetsVec[uciIdx].nHarqBytes;
            uint32_t nSrBytes    = evalPucchDataset.pucchF3bufferOffsetsVec[uciIdx].nSrBytes;
            uint32_t nCsiP1Bytes = evalPucchDataset.pucchF3bufferOffsetsVec[uciIdx].nCsiP1Bytes;

            for (int byteIdx = 0; byteIdx < nHarqBytes; byteIdx++) {
                uciPayloadsCpu[F3OutOffsetsCpu[uciIdx].harqPayloadByteOffset+byteIdx] = 0;
            }
            for (int byteIdx = 0; byteIdx < nSrBytes; byteIdx++) {
                uciPayloadsCpu[F3OutOffsetsCpu[uciIdx].srPayloadByteOffset+byteIdx] = 0;
            }
            for (int byteIdx = 0; byteIdx < nCsiP1Bytes; byteIdx++) {
                uciPayloadsCpu[F3OutOffsetsCpu[uciIdx].csi1PayloadByteOffset+byteIdx] = 0;
            }

            F3UciPrms[uciIdx].bitLenHarq                     = dynPucchApiDataset.F3UciPrmsVec[uciIdx].bitLenHarq;
            F3UciPrms[uciIdx].bitLenSr                       = dynPucchApiDataset.F3UciPrmsVec[uciIdx].bitLenSr;
            F3UciPrms[uciIdx].bitLenCsiPart1                 = dynPucchApiDataset.F3UciPrmsVec[uciIdx].bitLenCsiPart1;
        }
        
        CUDA_CHECK(cudaMemcpyAsync(uciPayloadsGpu, uciPayloadsCpu, numBytesUciPayloads * sizeof(uint8_t), cudaMemcpyHostToDevice, cuStrmMain.handle()));
        cudaStreamSynchronize(cuStrmMain.handle());
        //------------------------------------------------------------------
        // Pucch F2/3/4 UciSeg kernel descriptors

        // descriptors hold Kernel parameters in GPU
        size_t   dynDescrSizeBytes, dynDescrAlignBytes;
        cuphyStatus_t statusGetWorkspaceSize = cuphyPucchF234UciSegGetDescrInfo(&dynDescrSizeBytes,
                                                                                &dynDescrAlignBytes);
        if(CUPHY_STATUS_SUCCESS != statusGetWorkspaceSize) throw cuphy::cuphy_exception(statusGetWorkspaceSize);

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

        //------------------------------------------------------------------
        // Create Pucch F2/3/4 UciSeg kernel object

        cuphyPucchF234UciSegHndl_t pucchF234UciSegHndl;
        cuphyStatus_t statusCreate = cuphyCreatePucchF234UciSeg(&pucchF234UciSegHndl);

        if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);

        //------------------------------------------------------------------
        // setup Pucch F2/3/4 UciSeg kernel
        // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
        cuphyPucchF234UciSegLaunchCfg_t  pucchF234UciSegLaunchCfg;

        // setup function populates dynamic descriptor and launch config
        bool enableCpuToGpuDescrAsyncCpy = false;

        cuphyStatus_t pucchF234UciSegSetupStatus = cuphySetupPucchF234UciSeg(pucchF234UciSegHndl,
                                                                             nF2Ucis,
                                                                             nF3Ucis,
                                                                             F2UciPrms,
                                                                             F3UciPrms,
                                                                             F2OutOffsetsCpu,
                                                                             F3OutOffsetsCpu,
                                                                             uciPayloadsGpu,
                                                                             dynDescrBufCpu.addr(),
                                                                             dynDescrBufGpu.addr(),
                                                                             enableCpuToGpuDescrAsyncCpy,
                                                                             &pucchF234UciSegLaunchCfg,
                                                                             cuStrmMain.handle()); 

        if(CUPHY_STATUS_SUCCESS != pucchF234UciSegSetupStatus) throw cuphy::cuphy_exception(pucchF234UciSegSetupStatus);              

        if(!enableCpuToGpuDescrAsyncCpy) {
            cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());
        }            

        //------------------------------------------------------------------
        // run pucch F2/3/4 UciSeg kernel

        // launch kernel using the CUDA driver API
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pucchF234UciSegLaunchCfg.kernelNodeParamsDriver;
        CUresult pucchF234UciSegRunStatus = cuLaunchKernel(kernelNodeParamsDriver.func,
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
        if (CUDA_SUCCESS != pucchF234UciSegRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
        cudaStreamSynchronize(cuStrmMain.handle()); // synch to make sure kernel finishes
        
        // -------------------------------------------------------------------
        // Evaluate Pucch F3 SegLLRs kernel
        uint16_t nMismatches = evalPucchDataset.evalPucchF234UciSeg(uciPayloadsGpu, nF2Ucis, nF3Ucis, F2OutOffsetsCpu, F3OutOffsetsCpu, cuStrmMain.handle());
        printf("\n comparing cuPHY F2/F3 UCI output to reference output: %d mismatches out of %d UCIs\n", nMismatches, nF2Ucis + nF3Ucis);

        if (nMismatches) {
            returnValue = 1;
        }
        // -------------------------------------------------------------------
        // cleanup

        cuphyStatus_t statusDestroy = cuphyDestroyPucchF234UciSeg(pucchF234UciSegHndl);
        if(CUPHY_STATUS_SUCCESS != statusDestroy) throw cuphy::cuphy_exception(statusDestroy);

        delete uciPayloadsCpu;
        delete F2OutOffsetsCpu;
        delete F3OutOffsetsCpu;
        delete F2UciPrms;
        delete F3UciPrms;
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    return returnValue;
}