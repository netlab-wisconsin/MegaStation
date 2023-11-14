/*
 * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
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
#include "pusch_rx.hpp"

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
    cuphyNvlogFmtHelper nvlog_fmt("pusch_rateMatch.log");
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int         iArg = 1;
        std::vector<std::string> inputFilenameVec;

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
                    inputFilenameVec.emplace_back(argv[iArg++]);
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
        if(inputFilenameVec.empty())
        {
            usage();
            exit(1);
        }

        //----------------------------------------------------------------
        // Initialize CPU memory

        cuphy::buffer<PerTbParams, cuphy::pinned_alloc>              tbPrmsCpu_buffer(MAX_N_TBS_SUPPORTED);
        cuphy::buffer<cuphyPuschRxUeGrpPrms_t, cuphy::pinned_alloc>  drvdUeGrpPrmsBuffer(MAX_N_USER_GROUPS_SUPPORTED);

        //------------------------------------------------------------------
        // Load API parameters

        cuphy::stream cuStrmMain(cudaStreamDefault, PUSCH_STREAM_PRIORITY);

        StaticApiDataset  staticApiDataset(inputFilenameVec, cuStrmMain.handle());
        DynApiDataset     dynApiDataset(inputFilenameVec,   cuStrmMain.handle());
        EvalDataset       evalDataset(inputFilenameVec, cuStrmMain.handle());

        uint32_t nUes    = dynApiDataset.cellGrpDynPrm.nUes;
        uint16_t nUeGrps = dynApiDataset.cellGrpDynPrm.nUeGrps;

        cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied

        //------------------------------------------------------------------
        // Derive API parameters

        PuschRx puschRx(&staticApiDataset.puschStatPrms, cuStrmMain.handle());

        uint8_t                 enableRssiMeasurement = 0;
        cuphyLDPCParams         ldpcPrms(&staticApiDataset.puschStatPrms);

        puschRx.expandFrontEndParameters(&dynApiDataset.puschDynPrm, drvdUeGrpPrmsBuffer.addr(), enableRssiMeasurement);
        puschRx.expandBackEndParameters(&dynApiDataset.puschDynPrm, drvdUeGrpPrmsBuffer.addr(), tbPrmsCpu_buffer.addr(), ldpcPrms);


        //----------------------------------------------------------------------
        // GPU Uci on pusch input buffers

        std::vector<cuphy::tensor_device>  schLLRsVec;
        uint32_t nMaxCbsPerTb = 0;
        uint32_t nMaxTbs = nUes;

        for(int ueIdx = 0; ueIdx < nUes; ueIdx++)
        {
            if(tbPrmsCpu_buffer[ueIdx].uciOnPuschFlag)
            {
                schLLRsVec.emplace_back(CUPHY_R_16F, evalDataset.schLLRsRef[ueIdx].layout());
                schLLRsVec[ueIdx].convert(evalDataset.schLLRsRef[ueIdx], cuStrmMain.handle());

                tbPrmsCpu_buffer[ueIdx].d_schAndCsi2LLRs = static_cast<__half*>(schLLRsVec[ueIdx].addr());
            }
            if(tbPrmsCpu_buffer[ueIdx].isDataPresent)
            {
                nMaxCbsPerTb = std::max(nMaxCbsPerTb, tbPrmsCpu_buffer[ueIdx].num_CBs);
                printf("nMaxCbsPerTb %u num_CBs %u\n", nMaxCbsPerTb, tbPrmsCpu_buffer[ueIdx].num_CBs);
            }
        }

        //----------------------------------------------------------------------
        // GPU UE Grp input buffers

        cuphy::buffer<PerTbParams, cuphy::device_alloc>          tbPrmsGpu_buffer(nUes);
        std::vector<cuphy::tensor_device>                        tEqOutLLRsVec;
        std::vector<cuphyTensorPrm_t>                            tPrmEqOutLLRsVec(nUeGrps);

        for(int ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
        {
            tEqOutLLRsVec.emplace_back(CUPHY_R_16F, evalDataset.eqOutLLRsRef[ueGrpIdx].layout());
            tEqOutLLRsVec[ueGrpIdx].convert(evalDataset.eqOutLLRsRef[ueGrpIdx], cuStrmMain.handle());

            tPrmEqOutLLRsVec[ueGrpIdx].desc  = tEqOutLLRsVec[ueGrpIdx].desc().handle();
            tPrmEqOutLLRsVec[ueGrpIdx].pAddr = tEqOutLLRsVec[ueGrpIdx].addr();                   
        }

        cudaMemcpyAsync(tbPrmsGpu_buffer.addr(), tbPrmsCpu_buffer.addr(), sizeof(PerTbParams), cudaMemcpyHostToDevice, cuStrmMain.handle());
        cudaStreamSynchronize(cuStrmMain.handle());



        //----------------------------------------------------------------------
        // GPU output buffers

        uint32_t NUM_BYTES_PER_LLR = 2;
        uint32_t maxBytesRateMatch = NUM_BYTES_PER_LLR * nMaxTbs * nMaxCbsPerTb * MAX_N_RM_LLRS_PER_CB;
        cuphy::linear_alloc<128, cuphy::device_alloc>  linearAlloc(maxBytesRateMatch); 
        printf("nMaxTbs %u nMaxCbsPerTb %u maxBytesRateMatch %u\n", nMaxTbs, nMaxCbsPerTb, maxBytesRateMatch);

        void** ppRmOut;
         cudaError_t status = cudaHostAlloc(&ppRmOut, sizeof(uint8_t*)*nMaxTbs, cudaHostAllocPortable | cudaHostAllocMapped);
         if (status != cudaSuccess)
         {
             printf("Failure with cudaHostAlloc %d\n",status);
         }

        for(int ueIdx = 0; ueIdx < nUes; ++ueIdx)
        {
            size_t nBytesDeRm  = NUM_BYTES_PER_LLR * (tbPrmsCpu_buffer[ueIdx].Ncb + 2 * tbPrmsCpu_buffer[ueIdx].Zc) * tbPrmsCpu_buffer[ueIdx].num_CBs;
            ppRmOut[ueIdx]     = linearAlloc.alloc(nBytesDeRm);
        }

        //---------------------------------------------------------------------
        // Extract PUSCH rateMatch parameters

        
        const PerTbParams* pTbPrmsCpu = tbPrmsCpu_buffer.addr();
        const PerTbParams* pTbPrmsGpu = tbPrmsGpu_buffer.addr();
        cuphyTensorPrm_t*  pTPrmRmIn  = tPrmEqOutLLRsVec.data();

        uint16_t nSchUes = 0;
        std::vector<uint16_t> schUserIdxsVec(nUes);
        for(int ueIdx = 0; ueIdx < nUes; ++ueIdx)
        {
            if(pTbPrmsCpu[ueIdx].isDataPresent)
            {
                schUserIdxsVec[nSchUes]  =  ueIdx;
                nSchUes                 += 1;
            }
        }
        // cudaMemsetAsync(pRmOut, 0, maxBytesRateMatch, cuStrmMain.handle()); // make sure punctured LLRs set to zero
        // cudaStreamSynchronize(cuStrmMain.handle());

        //------------------------------------------------------------------
        // Pusch rateMatch descriptors

        // descriptors hold Kernel parameters in GPU
        size_t   dynDescrSizeBytes, dynDescrAlignBytes;

	cuphyStatus_t statusGetWorkspaceSize = cuphyPuschRxRateMatchGetDescrInfo(&dynDescrSizeBytes,
                                                                            &dynDescrAlignBytes);
        if(CUPHY_STATUS_SUCCESS != statusGetWorkspaceSize) throw cuphy::cuphy_exception(statusGetWorkspaceSize);

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);


        //------------------------------------------------------------------
        // Create Pusch rateMatch object
        
        cuphyPuschRxRateMatchHndl_t puschRmHndl;

        int FPconfig       = 3;  // 0: FP32 in, FP32 out; 1: FP16 in, FP32 out; 2: FP32 in, FP16 out; 3: FP16 in, FP16 out; other values: invalid
        int descramblingOn = 1;  // enable/disable descrambling

        cuphyCreatePuschRxRateMatch(&puschRmHndl, FPconfig,  descramblingOn);

        //------------------------------------------------------------------
        // Setup Pusch rateMatch object

        // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
        cuphyPuschRxRateMatchLaunchCfg_t puschRmLaunchCfg;

        // setup function populates dynamic descriptor and launch config
        bool enableCpuToGpuDescrAsyncCpy = false;

        cuphyStatus_t puschRmSetupStatus = cuphySetupPuschRxRateMatch( puschRmHndl,
                                                                       nSchUes,
                                                                       schUserIdxsVec.data(),
                                                                       pTbPrmsCpu,
                                                                       pTbPrmsGpu,
                                                                       pTPrmRmIn,
                                                                       pTPrmRmIn,
                                                                       ppRmOut,
                                                                       dynDescrBufCpu.addr(),
                                                                       dynDescrBufGpu.addr(),
                                                                       enableCpuToGpuDescrAsyncCpy,
                                                                       &puschRmLaunchCfg,
                                                                       cuStrmMain.handle());   
        if(CUPHY_STATUS_SUCCESS != puschRmSetupStatus) throw cuphy::cuphy_exception(puschRmSetupStatus);

        if(!enableCpuToGpuDescrAsyncCpy) {
            cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());
        }

        //------------------------------------------------------------------
        // Run Pusch rate match

        // launch kernel using the CUDA driver API
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = puschRmLaunchCfg.kernelNodeParamsDriver;
        CUresult pucchF0RxRunStatus = cuLaunchKernel(kernelNodeParamsDriver.func,
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
        if(CUDA_SUCCESS != pucchF0RxRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
        cudaStreamSynchronize(cuStrmMain.handle()); // synch to make sure kernel finishes



       evalDataset.evalPuschRm(ppRmOut, pTbPrmsCpu, cuStrmMain.handle());

    



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
