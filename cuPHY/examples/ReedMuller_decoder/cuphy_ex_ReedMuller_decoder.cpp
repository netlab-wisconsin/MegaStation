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
#include "pusch_utils.hpp"


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
    printf("pucch_F0_reciever [options]\n");
    printf("  Options:\n");
    printf("    -i  Input HDF5 filename\n");
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    cuphyNvlogFmtHelper nvlog_fmt("ReedMuller_decoder.log");
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int         iArg = 1;
        std::string inputFilename;

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

        //------------------------------------------------------------------
        // Load H5 file

        hdf5hpp::hdf5_file      fInput = hdf5hpp::hdf5_file::open(inputFilename.c_str());
        cuphy::cuphyHDF5_struct RMPar  = cuphy::get_HDF5_struct(fInput, "RMPar");

        uint16_t E = RMPar.get_value_as<uint32_t>("E");
        uint8_t  K = RMPar.get_value_as<uint8_t>("K");

        cuphyRmCwPrm_t cwPrm;
        cwPrm.E        = E;
        cwPrm.K        = K;
        cwPrm.exitFlag = 0;

       cuphy::typed_tensor<CUPHY_R_16F , cuphy::pinned_alloc>  refCwLLRs  = cuphy::typed_tensor_from_dataset<CUPHY_R_16F, cuphy::pinned_alloc>(fInput.open_dataset("rmBitLLRs"), cuphy::tensor_flags::align_default, cuStrmMain.handle());                      
       cuphy::typed_tensor<CUPHY_R_8U  , cuphy::pinned_alloc>  refPayload = cuphy::typed_tensor_from_dataset<CUPHY_R_8U , cuphy::pinned_alloc>(fInput.open_dataset("payload_uint8"), cuphy::tensor_flags::align_default, cuStrmMain.handle());
       
        //---------------------------------------------------------------------
        // Allocate memory

        size_t max_LLR_bytes = 273*12*sizeof(__half)*8;
        size_t max_out_bytes = MAX_N_TBS_SUPPORTED * sizeof(uint32_t);
        size_t max_prm_bytes = 2 * MAX_N_TBS_SUPPORTED * sizeof(int) + MAX_N_TBS_SUPPORTED * sizeof(void*);
        size_t max_bytes     = max_LLR_bytes + max_out_bytes + max_prm_bytes;

        cuphy::linear_alloc<128, cuphy::device_alloc>  linearAlloc(max_bytes); 

        //---------------------------------------------------------------------
        // Input parameters

        cwPrm.d_cbEst = static_cast<uint32_t*> (linearAlloc.alloc(sizeof(uint32_t)));
        cwPrm.d_LLRs  = static_cast<__half*>(linearAlloc.alloc(E * sizeof(__half)));

        // -----------------------------------------------------------
        // Copy buffers Cpu to Gpu

        cuphyRmCwPrm_t* pCwPrmsGpu  = static_cast<cuphyRmCwPrm_t*>(linearAlloc.alloc(sizeof(cuphyRmCwPrm_t)));

        cudaMemcpyAsync(pCwPrmsGpu      ,  &cwPrm           , sizeof(cuphyRmCwPrm_t) , cudaMemcpyHostToDevice, cuStrmMain.handle());
        cudaMemcpyAsync(cwPrm.d_LLRs,  refCwLLRs.addr() , sizeof(__half) * E     , cudaMemcpyHostToDevice, cuStrmMain.handle());
        cudaStreamSynchronize(cuStrmMain.handle());

        //------------------------------------------------------------------
        // Rm decoder descriptors

        size_t dynDescrSizeBytes, dynDescrAlignBytes;

        // descriptors hold Kernel parameters in GPU
        cuphyStatus_t statusGetWorkspaceSize = cuphyRmDecoderGetDescrInfo(&dynDescrSizeBytes,
                                                                          &dynDescrAlignBytes);
        if(CUPHY_STATUS_SUCCESS != statusGetWorkspaceSize) throw cuphy::cuphy_exception(statusGetWorkspaceSize);

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

        //---------------------------------------------------------------------
        // Create Decoder

        cuphyRmDecoderHndl_t  rmDecoderHndl;
        cuphy::context        ctx;
        unsigned int          rmFlags = 0;

        cuphyStatus_t status = cuphyCreateRmDecoder(ctx.handle(), &rmDecoderHndl, rmFlags, nullptr /* don't care about memory footprint tracking */);
        cudaStreamSynchronize(cuStrmMain.handle()); 
        if(CUPHY_STATUS_SUCCESS != status) 
        {
            throw cuphy::cuphy_fn_exception(status, "cuphyCreateRmDecoder()");
        }

        //------------------------------------------------------------------
        // Setup Rm decoder

        // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
        cuphyRmDecoderLaunchCfg_t rmDecoderLaunchCfg;

        // setup function populates dynamic descriptor and launch config
        bool enableCpuToGpuDescrAsyncCpy = false;

        uint16_t nRmCws = 1;
        cuphyStatus_t setupStatus = cuphySetupRmDecoder(rmDecoderHndl,
                                                        nRmCws,
                                                        pCwPrmsGpu,
                                                        static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                        dynDescrBufCpu.addr(),
                                                        dynDescrBufGpu.addr(),
                                                        &rmDecoderLaunchCfg,                 
                                                        cuStrmMain.handle());

        if(CUPHY_STATUS_SUCCESS != setupStatus) throw cuphy::cuphy_exception(setupStatus);

        if(!enableCpuToGpuDescrAsyncCpy)
        {
            cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());
        } 

        //------------------------------------------------------------------
        // Run simplex decoder

        // launch kernel using the CUDA driver API
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = rmDecoderLaunchCfg.kernelNodeParamsDriver;
        CUresult                 runStatus              = cuLaunchKernel(   kernelNodeParamsDriver.func,
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
        if(CUDA_SUCCESS != runStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
        cudaStreamSynchronize(cuStrmMain.handle()); // synch to make sure kernel finishes

        //-------------------------------------------------------------------
        //Evaluate Decoder  (first codeblock)

        uint32_t firstCb;
        cudaMemcpyAsync(&firstCb, cwPrm.d_cbEst, sizeof(uint32_t), cudaMemcpyDeviceToHost, cuStrmMain.handle());
        cudaStreamSynchronize(cuStrmMain.handle()); 

        uint8_t nMismatchs = 0;
        for(uint32_t bitIdx = 0; bitIdx < K; ++bitIdx)
        {
            uint8_t decodedBit = 0;
            if((firstCb >> bitIdx) & static_cast<uint32_t>(1))
            {
                decodedBit = 1;
            }

            if(decodedBit != refPayload(bitIdx))
            {
                nMismatchs += 1;
            }
        }
        printf("\n\n %d mismatches out of %d bits \n\n", nMismatchs, K);


        //-------------------------------------------------------------------
        // Cleanup Decoder  

        status = cuphyDestroyRmDecoder(rmDecoderHndl);

        if(CUPHY_STATUS_SUCCESS != status) 
        {
            throw cuphy::cuphy_fn_exception(status, "cuphyDestroyRmDecoder()");
        }
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
