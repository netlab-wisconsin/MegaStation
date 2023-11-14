/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <bitset>
#include <vector>
#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "datasets.hpp"
#include <cstring>
#include <iostream>
#include <dirent.h> // opendir, readdir
#include <errno.h>
#include <sys/stat.h> // for mkdir


const int TRIALS_PER_TV = 1;

////////////////////////////////////////////////////////////////////////
// usage()
void usage() {
    printf("Simplex [options]\n");
    printf("  Options:\n");
    printf("    -h                  Display usage information\n");
    printf("    -i  input_filename  Input HDF5 filename, which must contain the following datasets:\n");
    printf("    -s  synthetic CWs   Number of 'synthetic' CWs to test.  Will repeat CWs in TV up to this number\n");
    printf("    -p  print CW parameters\n");
}

using namespace cuphy;
////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[]) {
    int NUM_SYNTHETIC_CW = 0;
    int returnValue = 0;
    int PRINT_PARAMETERS = 0;
    cuphyNvlogFmtHelper nvlog_fmt("simplex_decoder.log");
    try {
        //------------------------------------------------------------------
        // Parse command line arguments
        int         iArg = 1;
        std::string inputFilename  = std::string();

        while(iArg < argc) {
            if('-' == argv[iArg][0]) {
                switch(argv[iArg][1]) {
                case 'i':
                    if(++iArg >= argc) {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided");
                        exit(1);
                    }
                    inputFilename.assign(argv[iArg++]);
                    break;
                case 's':
                    if (++iArg >= argc) {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: No number of synthetic codewords provided");
                        exit(1);
                    }
                    NUM_SYNTHETIC_CW = atoi(argv[iArg++]);
                    break;
                case 'p':
                    iArg++;
                    PRINT_PARAMETERS=1;
                    break;
                case 'h':
                    usage();
                    exit(0);
                    break;
                default:
                    NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                    usage();
                    exit(1);
                    break;
                }
            } else {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
                exit(1);
            }
        }
        if(inputFilename.empty()) {
            usage();
            exit(1);
        }

        //-----------------------------------------------------------------
        // Allocate GPU memory
        
        size_t max_cwPrms_mem  =  sizeof(cuphyPolarCwPrm_t) * CUPHY_MAX_N_SPX_CWS;
        size_t max_nRmBits     =  273 * 12 * 14 * 8 * 2; // maxPrb x scPerPrb x symbolPerSlot x maxLayers x maxQamBits
        size_t max_out_mem     =  CUPHY_MAX_N_SPX_CWS * sizeof(uint32_t);
        size_t max_mem         =  max_cwPrms_mem + max_nRmBits + max_out_mem;

        cuphy::linear_alloc<128, cuphy::device_alloc> linearAlloc(max_mem);

        //------------------------------------------------------------------
        // Load dataset
        
        cuphy::stream cuStrmMain;

        simplexDataset spxDataset(inputFilename, cuStrmMain.handle());
        cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied

        //----------------------------------------------------------------------
        // GPU input buffers

        uint16_t             nCws               = spxDataset.nCws;
        cuphySimplexCwPrm_t* pSimplexCwPrmsCpu  = spxDataset.simplexCwPrmsVec.data();


        for(int cwIdx = 0; cwIdx < nCws; ++cwIdx)
        {
            uint32_t E = pSimplexCwPrmsCpu[cwIdx].E;

            pSimplexCwPrmsCpu[cwIdx].d_LLRs = static_cast<__half*>(linearAlloc.alloc(E * sizeof(__half)));
            cudaMemcpyAsync(pSimplexCwPrmsCpu[cwIdx].d_LLRs, spxDataset.refCwLLRsVec[cwIdx].addr(), E * sizeof(__half), cudaMemcpyHostToDevice, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());
        }

        //----------------------------------------------------------------------
        // GPU output buffers

        for(int cbIdx = 0; cbIdx < nCws; ++cbIdx)
        {
            pSimplexCwPrmsCpu[cbIdx].d_cbEst = static_cast<uint32_t*>(linearAlloc.alloc(sizeof(uint32_t)));
        }

        //----------------------------------------------------------------------
        // copy codeword prms to GPU


        size_t nPrmBytes = nCws * sizeof(cuphySimplexCwPrm_t);
        cuphySimplexCwPrm_t* pSimplexCwPrmsGpu = static_cast<cuphySimplexCwPrm_t*>(linearAlloc.alloc(nPrmBytes));
        cudaMemcpyAsync(pSimplexCwPrmsGpu, pSimplexCwPrmsCpu, nPrmBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
        cudaStreamSynchronize(cuStrmMain.handle());


        //------------------------------------------------------------------
        // simplex decoder descriptors

        size_t dynDescrSizeBytes, dynDescrAlignBytes;

        // descriptors hold Kernel parameters in GPU
        cuphyStatus_t statusGetWorkspaceSize = cuphySimplexDecoderGetDescrInfo(&dynDescrSizeBytes,
                                                                               &dynDescrAlignBytes);
        if(CUPHY_STATUS_SUCCESS != statusGetWorkspaceSize) throw cuphy::cuphy_exception(statusGetWorkspaceSize);

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

        //------------------------------------------------------------------
        // Create simplex decoder object

        cuphySimplexDecoderHndl_t simplexDecoderHndl;
        cuphyStatus_t            statusCreate = cuphyCreateSimplexDecoder(&simplexDecoderHndl);

        if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);

        //------------------------------------------------------------------
        // Setup simplex decoder

        // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
        cuphySimplexDecoderLaunchCfg_t simplexDecoderLaunchCfg;

        // setup function populates dynamic descriptor and launch config
        bool enableCpuToGpuDescrAsyncCpy = false;


        cuphyStatus_t statusSetup =  cuphySetupSimplexDecoder(simplexDecoderHndl,
                                                              nCws,
                                                              pSimplexCwPrmsCpu, 
                                                              pSimplexCwPrmsGpu, 
                                                              static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                              dynDescrBufCpu.addr(),
                                                              dynDescrBufGpu.addr(),
                                                              &simplexDecoderLaunchCfg,                 
                                                              cuStrmMain.handle());

    if(CUPHY_STATUS_SUCCESS != statusSetup) throw cuphy::cuphy_exception(statusSetup);

        if(!enableCpuToGpuDescrAsyncCpy)
        {
            cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());
        }                       

        //------------------------------------------------------------------
        // Run simplex decoder

        // launch kernel using the CUDA driver API
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = simplexDecoderLaunchCfg.kernelNodeParamsDriver;
        CUresult                 simplexDecoderRunStatus  = cuLaunchKernel( kernelNodeParamsDriver.func,
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
        if(CUDA_SUCCESS != simplexDecoderRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
        cudaStreamSynchronize(cuStrmMain.handle()); // synch to make sure kernel finishes

        // -------------------------------------------------------------------
        // Evaluate decoder output

        spxDataset.evalDecoderOutput(pSimplexCwPrmsCpu, cuStrmMain.handle());

        // -------------------------------------------------------------------
        // cleanup

        cuphyStatus_t statusDestroy = cuphyDestroySimplexDecoder(simplexDecoderHndl);
        if(CUPHY_STATUS_SUCCESS != statusDestroy) throw cuphy::cuphy_exception(statusDestroy);


    }
    catch(std::exception& e) {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
        returnValue = -1;
    }
    catch(...) {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "UNKNOWN EXCEPTION");
        returnValue = -2;
    }
    return returnValue;
}