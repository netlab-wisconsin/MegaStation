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
    printf("polar_seg_deRm_deItl [options]\n");
    printf("  Options:\n");
    printf("    -i  Input HDF5 filename\n");
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    cuphyNvlogFmtHelper nvlog_fmt("polar_seg_deRm_deItl.log");
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int         iArg = 1;
        std::string inputFilename  = std::string();

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
                default:
                    NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                    usage();
                    exit(1);
                    break;
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

        size_t max_PolUciSegPrms_mem = sizeof(cuphyPolarUciSegPrm_t) * CUPHY_MAX_N_POL_UCI_SEGS;
        size_t max_nRmBits           = 273 * 12 * 14 * 8 * 2;      // maxPrb x scPerPrb x symbolPerSlot x maxLayers x maxQamBits 
        size_t max_uciSegLLRs_mem    = max_nRmBits * sizeof(__half);
        size_t max_cwLLRs_mem        = max_nRmBits * sizeof(__half);
        size_t max_mem               = max_PolUciSegPrms_mem + max_uciSegLLRs_mem + max_cwLLRs_mem;

        cuphy::linear_alloc<128, cuphy::device_alloc> linearAlloc(max_mem);

        //------------------------------------------------------------------
        // Load uci polar dataset

        UciPolarDataset uciPolarDataset(inputFilename, cuStrmMain.handle());
        cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied

        //----------------------------------------------------------------------
        // GPU output buffers

        uint16_t nPolUciSegs                       = uciPolarDataset.nPolUciSegs;
        cuphyPolarUciSegPrm_t* pPolarUciSegPrmsCpu = uciPolarDataset.polUciSegPrmsVec.data();
        
        uint16_t nPolCws = 0;
        std::vector<__half*>cwLLRsAddrVec(CUPHY_MAX_N_POL_CWS);
        std::vector<cuphyPolarCwPrm_t>cwPrmsVec(CUPHY_MAX_N_POL_CWS);

        for(int segIdx = 0; segIdx < nPolUciSegs; ++segIdx)
        {
            for(int i = 0; i < pPolarUciSegPrmsCpu[segIdx].nCbs; ++i)
            {
                size_t nBytes          = pPolarUciSegPrmsCpu[segIdx].N_cw * sizeof(__half);
                cwLLRsAddrVec[nPolCws] = static_cast<__half*>(linearAlloc.alloc(nBytes));
                cwPrmsVec[nPolCws].N_cw = pPolarUciSegPrmsCpu[segIdx].N_cw;

                pPolarUciSegPrmsCpu[segIdx].childCbIdxs[i] = nPolCws;
                nPolCws += 1;
            }
        }

        //----------------------------------------------------------------------
        // GPU input buffers 

        // copy uciSegPrms from CPU to GPU
        size_t nUciSegPrmBytes = nPolUciSegs * sizeof(cuphyPolarUciSegPrm_t);
        cuphyPolarUciSegPrm_t* pPolarUciSegPrmsGpu = static_cast<cuphyPolarUciSegPrm_t*>(linearAlloc.alloc(nUciSegPrmBytes));
        cudaMemcpyAsync(pPolarUciSegPrmsGpu, pPolarUciSegPrmsCpu, nUciSegPrmBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
        cudaStreamSynchronize(cuStrmMain.handle());

        // copy cwPrms from CPU to GPU
        size_t nCwPrmBytes = nPolCws * sizeof(cuphyPolarCwPrm_t);
        cuphyPolarCwPrm_t* pPolarCwPrmsGpu = static_cast<cuphyPolarCwPrm_t*>(linearAlloc.alloc(nCwPrmBytes));
        cudaMemcpyAsync(pPolarCwPrmsGpu, cwPrmsVec.data(), nCwPrmBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());

        // copy uciSegLLRs from CPU to GPU
        std::vector<__half*> uciSegLLRsAddrVec(nPolUciSegs);
        std::vector<cuphyPolarUciSegPrm_t>& polUciSegPrmsVec                                 = uciPolarDataset.polUciSegPrmsVec;
        std::vector<cuphy::typed_tensor<CUPHY_R_16F, cuphy::pinned_alloc>>& refUciSegLLRsVec = uciPolarDataset.refUciSegLLRsVec;

        for(int segIdx = 0; segIdx < nPolUciSegs; ++segIdx)
        {
            size_t nBytes             = polUciSegPrmsVec[segIdx].E_seg * sizeof(__half);
            uciSegLLRsAddrVec[segIdx] = static_cast<__half*>(linearAlloc.alloc(nBytes));

           cudaMemcpyAsync(uciSegLLRsAddrVec[segIdx], refUciSegLLRsVec[segIdx].addr(), nBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
           cudaStreamSynchronize(cuStrmMain.handle()); 
        }

        //------------------------------------------------------------------
        // polSegDeRmDeItl descriptors

        // descriptors hold Kernel parameters in GPU
        size_t   dynDescrSizeBytes, dynDescrAlignBytes;
        cuphyStatus_t statusGetWorkspaceSize = cuphyPolSegDeRmDeItlGetDescrInfo(&dynDescrSizeBytes,
                                                                          &dynDescrAlignBytes);
        if(CUPHY_STATUS_SUCCESS != statusGetWorkspaceSize) throw cuphy::cuphy_exception(statusGetWorkspaceSize);

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

        // for UCI Seg LLR addresses in CPU descriptor
        size_t   dynDescrUciAddrsSizeBytes = sizeof(__half*) * nPolUciSegs;
        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrUciAddrsBufCpu(dynDescrUciAddrsSizeBytes);

        // for CW LLR addresses in CPU descriptor
        size_t   dynDescrCwAddrsSizeBytes = sizeof(__half*) * nPolCws;
        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrCwAddrsBufCpu(dynDescrCwAddrsSizeBytes);

        //------------------------------------------------------------------
        // Create polSegDeRmDeItl object

        cuphyPolSegDeRmDeItlHndl_t polSegDeRmDeItlHndl;
        cuphyStatus_t statusCreate = cuphyCreatePolSegDeRmDeItl(&polSegDeRmDeItlHndl);

        if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);

        //------------------------------------------------------------------
        // setup polSegDeRmDeItl object

        // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
        cuphyPolSegDeRmDeItlLaunchCfg_t  polSegDeRmDeItlLaunchCfg;

        // setup function populates dynamic descriptor and launch config
        bool enableCpuToGpuDescrAsyncCpy = false;

        cuphyStatus_t polSegDeRmDeItlSetupStatus = cuphySetupPolSegDeRmDeItl(polSegDeRmDeItlHndl,
                                                                             nPolUciSegs,  
                                                                             nPolCws,              
                                                                             pPolarUciSegPrmsCpu,           
                                                                             pPolarUciSegPrmsGpu,
                                                                             cwPrmsVec.data(),
                                                                             pPolarCwPrmsGpu,            
                                                                             uciSegLLRsAddrVec.data(), 
                                                                             cwLLRsAddrVec.data(),       
                                                                             dynDescrBufCpu.addr(),                 
                                                                             dynDescrBufGpu.addr(),
                                                                             dynDescrCwAddrsBufCpu.addr(),
                                                                             dynDescrUciAddrsBufCpu.addr(),
                                                                             enableCpuToGpuDescrAsyncCpy, 
                                                                             &polSegDeRmDeItlLaunchCfg,                  
                                                                             cuStrmMain.handle());   

        if(CUPHY_STATUS_SUCCESS != polSegDeRmDeItlSetupStatus) throw cuphy::cuphy_exception(polSegDeRmDeItlSetupStatus);

        if(!enableCpuToGpuDescrAsyncCpy) {
            cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());
        }

        //------------------------------------------------------------------
        // run polSegDeRmDeItl

        // launch kernel using the CUDA driver API
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = polSegDeRmDeItlLaunchCfg.kernelNodeParamsDriver;
        CUresult polSegDeRmDeItlRunStatus = cuLaunchKernel(kernelNodeParamsDriver.func,
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
        if(CUDA_SUCCESS != polSegDeRmDeItlRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
        cudaStreamSynchronize(cuStrmMain.handle()); // synch to make sure kernel finishes


        // -------------------------------------------------------------------
        // Evaluate Pucch F0 reciver

        uciPolarDataset.evalCwLLRs(nPolCws, cwPrmsVec.data(), cwLLRsAddrVec.data(), cuStrmMain.handle());


        // -------------------------------------------------------------------
        // cleanup

        cuphyStatus_t statusDestroy = cuphyDestroyPolSegDeRmDeItl(polSegDeRmDeItlHndl);
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
