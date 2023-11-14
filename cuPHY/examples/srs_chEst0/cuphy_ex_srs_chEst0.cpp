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



/////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("srs_chEst [options]\n");
    printf("  Options:\n");
    printf("    -i  Input HDF5 filename\n");
    printf("    -G\n");
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    cuphyNvlogFmtHelper nvlog_fmt("srs_chEst0.log");
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int         iArg = 1;
        std::vector<std::string> inputFilenameVec;
        uint64_t procModeBmsk = 0;

        while(iArg < argc)
        {
            if('-' == argv[iArg][0])
            {
                switch(argv[iArg][1])
                {
                case 'i':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided");
                    }
                    inputFilenameVec.push_back(argv[iArg++]);
                    break;
                case 'G':
                    ++iArg;
                    procModeBmsk = SRS_PROC_MODE_FULL_SLOT_GRAPHS;
                    NVLOGI_FMT(NVLOG_SRS, "CUDA graph enabled!");
                    break;
                default:
                    NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                    usage();
                    exit(1);
                    break;
                }
            }
            else
            {
                NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
                exit(1);
            }
        }
        if(inputFilenameVec.empty())
        {
            usage();
            exit(1);
        }

        // ---------------------------------------------------------------
        // Initialize main stream

        cuphy::stream cuStrmMain;

        //-----------------------------------------------------------------
        // Initialize GPU memory

        size_t max_nSrsUes        =  1000;  
        size_t max_cells          =  32;    
        size_t max_rbSnr_mem      =  max_nSrsUes * 273 * sizeof(float);
        size_t max_srsReport_mem  =  max_nSrsUes * sizeof(cuphySrsReport_t);
        size_t max_chEstToL2_mem  =  max_cells * 273 * 128 * 16 * CUPHY_SRS_MAX_FULL_BAND_SRS_ANT_PORTS_SLOT_PER_CELL; // max_cells * max_prbs * max_ants * CUPHY_SRS_MAX_FULL_BAND_SRS_ANT_PORTS_SLOT_PER_CELL
        size_t max_mem            =  max_rbSnr_mem + max_srsReport_mem + max_chEstToL2_mem;

        cuphy::linear_alloc<128, cuphy::device_alloc> linearAlloc(max_mem);
        // memset() is to suppress compute-sanitizer initcheck errors.
        // It can be removed in the future, once compute-sanitizer allows to suppress errors/warnings.
        linearAlloc.memset(0, cuStrmMain.handle());
        CUDA_CHECK(cudaStreamSynchronize(cuStrmMain.handle()));

        //------------------------------------------------------------------
        // Load API parameters

        srsStaticApiDataset  srsStaticApiDataset(inputFilenameVec, cuStrmMain.handle());
        srsDynApiDataset     srsDynApiDataset(inputFilenameVec,    cuStrmMain.handle(), procModeBmsk);
        srsEvalDataset       srsEvalDataset(inputFilenameVec,      cuStrmMain.handle());  

        uint16_t                           nSrsUes        = srsDynApiDataset.cellGrpDynPrm.nSrsUes;
        uint16_t                           nCells         = srsDynApiDataset.cellGrpDynPrm.nCells;
        std::vector<cuphySrsCellDynPrm_t>& cellDynPrmVec  = srsDynApiDataset.cellDynPrmVec;
        std::vector<cuphyUeSrsPrm_t>&      ueSrsPrmVec    = srsDynApiDataset.ueSrsPrmVec;
        std::vector<cuphyCellStatPrm_t>&   cellStatPrmVec = srsStaticApiDataset.cellStatPrmVec;
        cuphySrsChEstBuffInfo_t*           pChEstBuffInfo = srsDynApiDataset.dataOut.pChEstBuffInfo;
        cuphyTensorPrm_t*                  pTDataRx       = srsDynApiDataset.dataIn.pTDataRx;
        cuphySrsFilterPrms_t&              srsFilterPrms  = srsStaticApiDataset.srsStatPrms.srsFilterPrms;
        cuphySrsChEstToL2_t*               pSrsChEstToL2  = srsDynApiDataset.chEstToL2Vec.data();

        //---------------------------------------------------------------------
        // Combine static and Dynamic cell parameters

        std::vector<cuphySrsCellPrms_t> srsCellPrmsVec(nCells);

        for(int cellIdx = 0; cellIdx < nCells; ++cellIdx)
        {
            srsCellPrmsVec[cellIdx].slotNum     = cellDynPrmVec[cellIdx].slotNum;
            srsCellPrmsVec[cellIdx].frameNum    = cellDynPrmVec[cellIdx].frameNum;
            srsCellPrmsVec[cellIdx].srsStartSym = cellDynPrmVec[cellIdx].srsStartSym;
            srsCellPrmsVec[cellIdx].nSrsSym     = cellDynPrmVec[cellIdx].nSrsSym;
            srsCellPrmsVec[cellIdx].nRxAntSrs   = cellStatPrmVec[cellIdx].nRxAntSrs;
            srsCellPrmsVec[cellIdx].mu          = cellStatPrmVec[cellIdx].mu;
        }

        //-----------------------------------------------------------------
        // Allocate output memory

        uint32_t            rbSnrBufferSize = nSrsUes * 273;
        float*              d_rbSnrBuffer   = static_cast<float*>(linearAlloc.alloc(sizeof(float) * rbSnrBufferSize));
        cuphySrsReport_t*   d_srsReports    = static_cast<cuphySrsReport_t*>(linearAlloc.alloc(sizeof(cuphySrsReport_t) * nSrsUes));
        std::vector<void*>  addrGpuChEstToL2Vec(nSrsUes);
        // since cuphySrsReport_t has few parameters that need to be initialized to 0, perform the following copy
        // ToDo?? any better approach?
        std::vector<cuphySrsReport_t> h_srsReports(nSrsUes);
        for (auto& srs : h_srsReports)
        {
            srs.widebandSignalEnergy = 0.f;
            srs.widebandNoiseEnergy = 0.f;
            srs.widebandScCorr = __floats2half2_rn(0.f, 0.f);
        }
        CUDA_CHECK(cudaMemcpyAsync (d_srsReports, h_srsReports.data(), sizeof(cuphySrsReport_t) * nSrsUes, cudaMemcpyHostToDevice, cuStrmMain.handle()));
        CUDA_CHECK(cudaStreamSynchronize(cuStrmMain.handle()));

        
        for(int ueIdx = 0; ueIdx < nSrsUes; ++ueIdx){
            size_t maxChEstSize = 273*128*4*sizeof(float2);
            addrGpuChEstToL2Vec[ueIdx] = linearAlloc.alloc(maxChEstSize);
        }

        //----------------------------------------------------------------------------------------------------------

        std::vector<uint32_t> rbSnrBuffOffsetsVec(nSrsUes);
        for(int ueIdx = 0; ueIdx < nSrsUes; ++ueIdx)
        {
            rbSnrBuffOffsetsVec[ueIdx] = ueIdx * 273;
        }

       // ------------------------------------------------------------------
       // srsChEst0 descriptors

        // descriptors hold Kernel parameters in GPU
        size_t statDescrSizeBytes, statDescrAlignBytes, dynDescrSizeBytes, dynDescrAlignBytes;
        cuphyStatus_t statusGetWorkspaceSize = cuphySrsChEst0GetDescrInfo(&statDescrSizeBytes,
                                                                          &statDescrAlignBytes,
                                                                          &dynDescrSizeBytes,
                                                                          &dynDescrAlignBytes);
        if(CUPHY_STATUS_SUCCESS != statusGetWorkspaceSize) throw cuphy::cuphy_exception(statusGetWorkspaceSize);

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> statDescrBufCpu(statDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> statDescrBufGpu(statDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

        //------------------------------------------------------------------
        // Create srsChEst0 object

        cuphySrsChEst0Hndl_t srsChEst0Hndl;
        bool enableCpuToGpuDescrAsyncCpy = false;

        cuphyStatus_t statusCreate = cuphyCreateSrsChEst0(&srsChEst0Hndl, 
                                                          &srsFilterPrms,
                                                          static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                          statDescrBufCpu.addr(),                     
                                                          statDescrBufGpu.addr(), 
                                                          cuStrmMain.handle());

        if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);

        if(!enableCpuToGpuDescrAsyncCpy){
            CUDA_CHECK(cudaMemcpyAsync(statDescrBufGpu.addr(), statDescrBufCpu.addr(), statDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle()));
            cudaStreamSynchronize(cuStrmMain.handle());
        }

        //------------------------------------------------------------------
        // setup srsChEst0

        // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
        cuphySrsChEst0LaunchCfg_t  srsChEst0LaunchCfg;

        // Setup function populates dynamic descriptor and launch config. Option to copy descriptors to GPU during setup call.
        cuphyStatus_t setupStatus = cuphySetupSrsChEst0(srsChEst0Hndl,
                                                        nSrsUes,
                                                        ueSrsPrmVec.data(),
                                                        nCells,
                                                        pTDataRx, 
                                                        srsCellPrmsVec.data(),
                                                        d_rbSnrBuffer,
                                                        rbSnrBuffOffsetsVec.data(),
                                                        d_srsReports,
                                                        pChEstBuffInfo,
                                                        addrGpuChEstToL2Vec.data(),
                                                        pSrsChEstToL2,
                                                        static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                        dynDescrBufCpu.addr(),                     
                                                        dynDescrBufGpu.addr(), 
                                                        &srsChEst0LaunchCfg,
                                                        cuStrmMain.handle());

        if(CUPHY_STATUS_SUCCESS != setupStatus) throw cuphy::cuphy_exception(setupStatus);
        if(!enableCpuToGpuDescrAsyncCpy) {
            cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());
        }

        //------------------------------------------------------------------
        // run srs ChEst

        // launch kernel using the CUDA driver API
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = srsChEst0LaunchCfg.kernelNodeParamsDriver;
        CUresult srsChEst0RunStatus = cuLaunchKernel(kernelNodeParamsDriver.func,
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
        if(CUDA_SUCCESS != srsChEst0RunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
        cudaStreamSynchronize(cuStrmMain.handle()); // synch to make sure kernel finishes

        //------------------------------------------------------------------
        // Evaluate results    

        // copy to buffers:
        CUDA_CHECK(cudaMemcpyAsync(srsDynApiDataset.dataOut.pSrsReports,  d_srsReports , sizeof(cuphySrsReport_t) * nSrsUes, cudaMemcpyDeviceToHost, cuStrmMain.handle()));
        CUDA_CHECK(cudaMemcpyAsync(srsDynApiDataset.dataOut.pRbSnrBuffer, d_rbSnrBuffer, sizeof(float) * rbSnrBufferSize   , cudaMemcpyDeviceToHost, cuStrmMain.handle()));
        cudaStreamSynchronize(cuStrmMain.handle());

        srsEvalDataset.evalSrsRx(srsDynApiDataset.srsDynPrm, srsDynApiDataset.tSrsChEstVec, srsDynApiDataset.dataOut.pRbSnrBuffer, srsDynApiDataset.dataOut.pSrsReports, cuStrmMain.handle());

        //------------------------------------------------------------------
        // cleanup

        cuphyStatus_t statusDestroy = cuphyDestroySrsChEst0(srsChEst0Hndl);
        if(CUPHY_STATUS_SUCCESS != statusDestroy) throw cuphy::cuphy_exception(statusDestroy);

    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    return returnValue;
}
