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
    printf("uciOnPusch_csi2_ctrl [options]\n");
    printf("  Options:\n");
    printf("    -i  Input HDF5 filename\n");
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    cuphyNvlogFmtHelper nvlog_fmt("uciOnPusch_csi2_ctrl.log");
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int         iArg = 1;
        std::vector<std::string> inputFilenameVec;
        std::string              outputFilename =  std::string();

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
                case 'o':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: No output file name given");
                    }
                    outputFilename.assign(argv[iArg++]);
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

        //----------------------------------------------------------------
        // Initialize CPU memory

        cuphy::buffer<PerTbParams, cuphy::pinned_alloc>              tbPrmsCpu_buffer(MAX_N_TBS_SUPPORTED);
        cuphy::buffer<cuphyPuschRxUeGrpPrms_t, cuphy::pinned_alloc>  drvdUeGrpPrmsBuffer(PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP);

        // // //-----------------------------------------------------------------
        // // // Initialize GPU memory

        // // size_t max_nEqOutLLRs       =  273 * 12 * 14 * 8 * 2;      // maxPrb x scPerPrb x symbolPerSlot x maxLayers x maxQamBits 
        // // size_t max_nDeSegLLRs       =  max_nEqOutLLRs;
        // // size_t max_eqOutLLRs_mem    =  max_nEqOutLLRs * sizeof(__half);
        // // size_t max_nDeSegLLRs_mem   =  max_nDeSegLLRs * sizeof(__half);
        // // size_t max_tbPrms_mem       =  sizeof(PerTbParams) * MAX_N_TBS_SUPPORTED;
        // // size_t max_mem              =  max_eqOutLLRs_mem + max_nDeSegLLRs_mem + max_tbPrms_mem;
        size_t max_mem = 10000000;

        cuphy::linear_alloc<128, cuphy::device_alloc> linearAlloc(max_mem);

        //------------------------------------------------------------------
        // Load API parameters

       StaticApiDataset  staticApiDataset(inputFilenameVec, cuStrmMain.handle());
       DynApiDataset     dynApiDataset(inputFilenameVec,    cuStrmMain.handle());
       EvalDataset       evalDataset(inputFilenameVec,      cuStrmMain.handle());

        cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied

        uint16_t                 nUes                  = dynApiDataset.puschDynPrm.pCellGrpDynPrm->nUes;
        uint16_t                 nUeGrps               = dynApiDataset.cellGrpDynPrm.nUeGrps;
        cuphyPuschUePrm_t*       pUePrmsCpu            = dynApiDataset.puschDynPrm.pCellGrpDynPrm->pUePrms;
        uint16_t                 nDynCells             = dynApiDataset.cellGrpDynPrm.nCells;
        cuphyPuschCellDynPrm_t*  pPuschCellDynPrmsCpu  = dynApiDataset.cellGrpDynPrm.pCellPrms;
        uint16_t                 nStatCells            = staticApiDataset.puschStatPrms.nMaxCells;
        cuphyPuschCellStatPrm_t* pPuschCellStatPrmsCpu = staticApiDataset.puschCellStatPrmVec.data();
        uint16_t                 forcedNumCsi2Bits     = staticApiDataset.dbgPrm.forcedNumCsi2Bits;

        //------------------------------------------------------------------
        // Derive API parameters

        PuschRx puschRx(&staticApiDataset.puschStatPrms, cuStrmMain.handle());

        uint8_t          enableRssiMeasurement = 0;
        cuphyLDPCParams  ldpcPrms(&staticApiDataset.puschStatPrms);

        puschRx.expandFrontEndParameters(&dynApiDataset.puschDynPrm, drvdUeGrpPrmsBuffer.addr(), enableRssiMeasurement);
        puschRx.expandBackEndParameters(&dynApiDataset.puschDynPrm, drvdUeGrpPrmsBuffer.addr(), tbPrmsCpu_buffer.addr(), ldpcPrms);

        //------------------------------------------------------------------
        // Load CSI-P2 parameters  

        uint16_t nCsi2Ues                     =  evalDataset.nCsi2Ues;
        std::vector<uint16_t>& csi2UeIdxsVec  =  evalDataset.csi2UeIdxsVec; 

        //----------------------------------------------------------------------
        // Load GPU input buffers

        PerTbParams* pTbPrmsGpu = static_cast<PerTbParams*>(linearAlloc.alloc(nUes * sizeof(PerTbParams)));
        cudaMemcpyAsync(pTbPrmsGpu, tbPrmsCpu_buffer.addr(), sizeof(PerTbParams) * nUes, cudaMemcpyHostToDevice, cuStrmMain.handle());

        cuphyPuschCellStatPrm_t* pPuschCellStatPrmsGpu = static_cast<cuphyPuschCellStatPrm_t*>(linearAlloc.alloc(nStatCells * sizeof(cuphyPuschCellStatPrm_t)));
        cudaMemcpyAsync(pPuschCellStatPrmsGpu, pPuschCellStatPrmsCpu, sizeof(cuphyPuschCellStatPrm_t) * nStatCells, cudaMemcpyHostToDevice, cuStrmMain.handle());

        size_t   nUciBytes      = evalDataset.tRefUciPayloadBytes.layout().dimensions()[0];
        uint8_t* pUciPayloadGpu = static_cast<uint8_t*>(linearAlloc.alloc(nUciBytes));
        cudaMemcpyAsync(pUciPayloadGpu, evalDataset.tRefUciPayloadBytes.addr(), nUciBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());

        // synch to ensure copies 
        cudaStreamSynchronize(cuStrmMain.handle());

        //------------------------------------------------------------------
        // Allocate GPU output buffers

        cuphySimplexCwPrm_t*    pCsi2SpxCwPrmsGpu  = static_cast<cuphySimplexCwPrm_t*>   (linearAlloc.alloc(nCsi2Ues   * sizeof(cuphySimplexCwPrm_t)));
        cuphyRmCwPrm_t*         pCsi2RmCwPrmsGpu   = static_cast<cuphyRmCwPrm_t*>        (linearAlloc.alloc(nCsi2Ues   * sizeof(cuphyRmCwPrm_t)));
        cuphyPolarUciSegPrm_t*  pCsi2PolSegPrmsGpu = static_cast<cuphyPolarUciSegPrm_t*> (linearAlloc.alloc(nCsi2Ues   * sizeof(cuphyPolarUciSegPrm_t)));
        cuphyPolarCwPrm_t*      pCsi2PolCwPrmsGpu  = static_cast<cuphyPolarCwPrm_t*>     (linearAlloc.alloc(2*nCsi2Ues * sizeof(cuphyPolarCwPrm_t)));

        uint16_t* pNumCsi2Bits = static_cast<uint16_t*>(linearAlloc.alloc(nCsi2Ues * sizeof(uint16_t)));

        //------------------------------------------------------------------
        // uciOnPusch Buffer offsets

        std::vector<cuphyUciOnPuschOutOffsets_t> uciOffsetVec(nUes);
        EvalDataset::ueRefBufferOffsets*         pUeRefBufferOffsets = evalDataset.ueRefBuffOffsetsVec.data();

        for(int ueIdx = 0; ueIdx < nUes; ++ueIdx)
        {
            uciOffsetVec[ueIdx].csi1PayloadByteOffset = pUeRefBufferOffsets[ueIdx].csi1PayloadByteOffset;
            uciOffsetVec[ueIdx].numCsi2BitsOffset     = ueIdx;
        }

        //------------------------------------------------------------------
        // uciOnPuschCsi2Ctrl descriptors

        // descriptors hold Kernel parameters in GPU
        size_t   dynDescrSizeBytes, dynDescrAlignBytes;
        cuphyStatus_t statusGetWorkspaceSize = cuphyUciOnPuschCsi2CtrlGetDescrInfo(&dynDescrSizeBytes,
                                                                                   &dynDescrAlignBytes);
        if(CUPHY_STATUS_SUCCESS != statusGetWorkspaceSize) throw cuphy::cuphy_exception(statusGetWorkspaceSize);

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

        //------------------------------------------------------------------
        // Create uciOnPuschCsi2Ctrl object

        cuphyUciOnPuschCsi2CtrlHndl_t uciOnPuschCsi2CtrlHndl;
        cuphyStatus_t statusCreate = cuphyCreateUciOnPuschCsi2Ctrl(&uciOnPuschCsi2CtrlHndl);

        if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);

        //------------------------------------------------------------------
        // setup uciOnPuschCsi2Ctrl

        // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
        cuphyUciOnPuschCsi2CtrlLaunchCfg_t  uciOnPuschCsi2CtrlLaunchCfg;

        // Setup function populates dynamic descriptor and launch config. Option to copy descriptors to GPU during setup call.
        bool enableCpuToGpuDescrAsyncCpy = false;

        cuphyStatus_t setupStatus = cuphySetupUciOnPuschCsi2Ctrl(uciOnPuschCsi2CtrlHndl,
                                                                 nCsi2Ues,
                                                                 csi2UeIdxsVec.data(),
                                                                 tbPrmsCpu_buffer.addr(),
                                                                 pTbPrmsGpu,
                                                                 drvdUeGrpPrmsBuffer.addr(),
                                                                 pPuschCellStatPrmsGpu,
                                                                 uciOffsetVec.data(),
                                                                 pUciPayloadGpu,
                                                                 pNumCsi2Bits,
                                                                 pCsi2PolSegPrmsGpu,
                                                                 pCsi2PolCwPrmsGpu,
                                                                 pCsi2RmCwPrmsGpu,
                                                                 pCsi2SpxCwPrmsGpu,
                                                                 forcedNumCsi2Bits,
                                                                 dynDescrBufCpu.addr(),                     
                                                                 dynDescrBufGpu.addr(), 
                                                                 static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),            
                                                                 &uciOnPuschCsi2CtrlLaunchCfg,                      
                                                                 cuStrmMain.handle());


        if(CUPHY_STATUS_SUCCESS != setupStatus) throw cuphy::cuphy_exception(setupStatus);
        if(!enableCpuToGpuDescrAsyncCpy) {
            cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());
        }

        //------------------------------------------------------------------
        // run uciOnPuschCsi2Ctrl

        // Launch kernel using the CUDA driver API
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = uciOnPuschCsi2CtrlLaunchCfg.kernelNodeParamsDriver;
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
        // Compare cuphy results to reference

        evalDataset.evalUciOnPuschCsi2Ctrl(pTbPrmsGpu, cuStrmMain.handle());

        // -------------------------------------------------------------------
        // Cleanup

        cuphyStatus_t statusDestroy = cuphyDestroyUciOnPuschCsi2Ctrl(uciOnPuschCsi2CtrlHndl);
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
