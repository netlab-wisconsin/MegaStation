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
#include "pusch_rx.hpp"

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
    printf("pucch_F0_reciever [options]\n");
    printf("  Options:\n");
    printf("    -i  Input HDF5 filename\n");
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    cuphyNvlogFmtHelper nvlog_fmt("uci_on_pusch_deRm.log");
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
                    inputFilenameVec.push_back(argv[iArg++]);
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

        // ---------------------------------------------------------------
        // Initialize main stream

        cuphy::stream cuStrmMain(cudaStreamDefault, PUSCH_STREAM_PRIORITY);

        //------------------------------------------------------------------
        // Load API parameters

        uint16_t maxTbs = 100;

        StaticApiDataset  staticApiDataset(inputFilenameVec, cuStrmMain.handle());
        DynApiDataset     dynApiDataset(inputFilenameVec,   cuStrmMain.handle());
        EvalDataset       evalDataset(inputFilenameVec, cuStrmMain.handle());

        cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied

        //----------------------------------------------------------------
        // Initialize CPU memory

        cuphy::buffer<PerTbParams, cuphy::pinned_alloc>              tbPrmsCpu_buffer(maxTbs);
        cuphy::buffer<uint16_t, cuphy::pinned_alloc>                 uciUserIdxs_buffer(maxTbs);
        cuphy::buffer<cuphyPuschRxUeGrpPrms_t, cuphy::pinned_alloc>  drvdUeGrpPrmsBuffer(PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP);

        //-----------------------------------------------------------------
        // Initialize GPU memory

        size_t max_nEqOutLLRs        = 273 * 12 * 14 * 8 * 2;      // maxPrb x scPerPrb x symbolPerSlot x maxLayers x maxQamBits 
        size_t max_nDeSegLLRs        = max_nEqOutLLRs;
        size_t max_eqOutLLRs_mem     = max_nEqOutLLRs * sizeof(__half);
        size_t max_nDeSegLLRs_mem    = max_nDeSegLLRs * sizeof(__half);
        size_t max_tbPrms_mem        = sizeof(PerTbParams) * maxTbs;
        size_t max_mem               = max_eqOutLLRs_mem + max_nDeSegLLRs_mem + max_tbPrms_mem;

        cuphy::linear_alloc<128, cuphy::device_alloc> linearAlloc(max_mem);

        //------------------------------------------------------------------
        // Derive API parameters

        PuschRx puschRx(&staticApiDataset.puschStatPrms, cuStrmMain.handle());

        uint8_t                 enableRssiMeasurement = 0;
        cuphyLDPCParams         ldpcPrms(&staticApiDataset.puschStatPrms);

        puschRx.expandFrontEndParameters(&dynApiDataset.puschDynPrm, drvdUeGrpPrmsBuffer.addr(), enableRssiMeasurement);
        puschRx.expandBackEndParameters(&dynApiDataset.puschDynPrm, drvdUeGrpPrmsBuffer.addr(), tbPrmsCpu_buffer.addr(), ldpcPrms);
        
        //------------------------------------------------------------------
        // Allocate GPU output buffers

        uint16_t            nUes   = dynApiDataset.puschDynPrm.pCellGrpDynPrm->nUes;
        cuphyPuschUePrm_t* pUePrms = dynApiDataset.puschDynPrm.pCellGrpDynPrm->pUePrms;
        uint16_t nUciUes = 0;

        for(int ueIdx = 0; ueIdx < nUes; ++ueIdx)
        {
            if(pUePrms[ueIdx].pduBitmap & 2) // check 1st bit for UCI transmission
            {
                tbPrmsCpu_buffer[ueIdx].d_schAndCsi2LLRs = static_cast<__half*>(linearAlloc.alloc(tbPrmsCpu_buffer[ueIdx].G * sizeof(__half)));
                tbPrmsCpu_buffer[ueIdx].d_csi1LLRs       = static_cast<__half*>(linearAlloc.alloc(tbPrmsCpu_buffer[ueIdx].G_csi1 * sizeof(__half)));
                tbPrmsCpu_buffer[ueIdx].d_harqLLrs       = static_cast<__half*>(linearAlloc.alloc(tbPrmsCpu_buffer[ueIdx].G_harq * sizeof(__half)));
                uciUserIdxs_buffer[nUciUes] = ueIdx;
                nUciUes += 1;
            }
        }

        //----------------------------------------------------------------------
        // Load GPU input buffers

        // equalizer output LLRs
        uint16_t nUeGrps = dynApiDataset.cellGrpDynPrm.nUeGrps;
        std::vector<cuphy::tensor_device>                        tEqOutLLRsVec;
        std::vector<cuphyTensorPrm_t>                            tPrmEqOutLLRsVec(nUeGrps);

        for(int ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
        {
            tEqOutLLRsVec.emplace_back(CUPHY_R_16F, evalDataset.eqOutLLRsRef[ueGrpIdx].layout());
            tEqOutLLRsVec[ueGrpIdx].convert(evalDataset.eqOutLLRsRef[ueGrpIdx], cuStrmMain.handle());

            tPrmEqOutLLRsVec[ueGrpIdx].desc  = tEqOutLLRsVec[ueGrpIdx].desc().handle();
            tPrmEqOutLLRsVec[ueGrpIdx].pAddr = tEqOutLLRsVec[ueGrpIdx].addr();                   
        }

        // copy perTbParams to GPU
        PerTbParams* pTbPrmsGpu = static_cast<PerTbParams*>(linearAlloc.alloc(sizeof(PerTbParams) * nUes));
        cudaMemcpyAsync(pTbPrmsGpu, tbPrmsCpu_buffer.addr(), sizeof(PerTbParams), cudaMemcpyHostToDevice, cuStrmMain.handle());

        // synch to ensure copies 
        cudaStreamSynchronize(cuStrmMain.handle());

        //------------------------------------------------------------------
        // uciOnPuschSegLLRs1 descriptors

        // descriptors hold Kernel parameters in GPU
        size_t   dynDescrSizeBytes, dynDescrAlignBytes;
        cuphyStatus_t statusGetWorkspaceSize = cuphyUciOnPuschSegLLRs1GetDescrInfo(&dynDescrSizeBytes,
                                                                                   &dynDescrAlignBytes);
        if(CUPHY_STATUS_SUCCESS != statusGetWorkspaceSize) throw cuphy::cuphy_exception(statusGetWorkspaceSize);

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

        //------------------------------------------------------------------
        // Create uciOnPuschSegLLRs1 object

        cuphyUciOnPuschSegLLRs1Hndl_t uciOnPuschSegLLRs1Hndl;
        cuphyStatus_t statusCreate = cuphyCreateUciOnPuschSegLLRs1(&uciOnPuschSegLLRs1Hndl);

        if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);

        //------------------------------------------------------------------
        // setup uciOnPuschSegLLRs1

        // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
        cuphyUciOnPuschSegLLRs1LaunchCfg_t  uciOnPuschSegLLRs1LaunchCfg;

        // Setup function populates dynamic descriptor and launch config. Option to copy descriptors to GPU during setup call.
        bool enableCpuToGpuDescrAsyncCpy = false;

        // Parameters needed for setup function
        std::vector<uint16_t> nPrbsVec(nUeGrps);
        for(int grpIdx = 0; grpIdx < nUeGrps; ++grpIdx){
            nPrbsVec[grpIdx] = drvdUeGrpPrmsBuffer[grpIdx].nPrb;
        }
        uint8_t   startSym      = dynApiDataset.ueGrpPrmsVec[0].puschStartSym;
        uint8_t   nPuschSym     = dynApiDataset.ueGrpPrmsVec[0].nPuschSym;
        uint8_t   nPuschDataSym = drvdUeGrpPrmsBuffer[0].nDataSym;
        uint8_t*  pDataSymIdxs  = drvdUeGrpPrmsBuffer[0].dataSymLoc;
        uint8_t   nPuschDmrsSym = drvdUeGrpPrmsBuffer[0].nDmrsSyms;
        uint8_t*  pDmrsSymIdxs  = drvdUeGrpPrmsBuffer[0].dmrsSymLoc;

        // call setup function
        cuphyStatus_t setupStatus = cuphySetupUciOnPuschSegLLRs1(uciOnPuschSegLLRs1Hndl,
                                                                nUciUes,                          
                                                                uciUserIdxs_buffer.addr(), 
                                                                tbPrmsCpu_buffer.addr(),
                                                                pTbPrmsGpu,
                                                                nUeGrps,
                                                                tPrmEqOutLLRsVec.data(),
                                                                nPrbsVec.data(),
                                                                startSym,
                                                                nPuschSym,
                                                                nPuschDataSym,
                                                                pDataSymIdxs,
                                                                nPuschDmrsSym,
                                                                pDmrsSymIdxs,    
                                                                dynDescrBufCpu.addr(),                     
                                                                dynDescrBufGpu.addr(), 
                                                                enableCpuToGpuDescrAsyncCpy,            
                                                                &uciOnPuschSegLLRs1LaunchCfg,                      
                                                                cuStrmMain.handle());  

        if(CUPHY_STATUS_SUCCESS != setupStatus) throw cuphy::cuphy_exception(setupStatus);

        if(!enableCpuToGpuDescrAsyncCpy) {
            cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());
        }

        //------------------------------------------------------------------
        // run uciOnPuschSegLLRs1

        // launch kernel using the CUDA driver API
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = uciOnPuschSegLLRs1LaunchCfg.kernelNodeParamsDriver;
        CUresult runStatus = cuLaunchKernel(kernelNodeParamsDriver.func,
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
        
        // -------------------------------------------------------------------
        // Evaluate computed uci Rm sizes

        std::vector<PerTbParams>& perTbPrmsRef    = evalDataset.perTbPrmsRef;
        std::vector<cuphyPuschUePrm_t> &uePrmsVec = dynApiDataset.uePrmsVec;

        evalDataset.evalUciRmSizes(tbPrmsCpu_buffer.addr(), perTbPrmsRef.data(), uePrmsVec.data(), nUes);

        evalDataset.evalUciOnPuschSegLLRs1(nUciUes, uciUserIdxs_buffer.addr(), tbPrmsCpu_buffer.addr(), cuStrmMain.handle());

        // -------------------------------------------------------------------
        // cleanup

        cuphyStatus_t statusDestroy = cuphyDestroyUciOnPuschSegLLRs1(uciOnPuschSegLLRs1Hndl);
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
