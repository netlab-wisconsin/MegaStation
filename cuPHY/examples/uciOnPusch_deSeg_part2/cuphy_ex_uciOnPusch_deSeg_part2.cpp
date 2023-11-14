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
    cuphyNvlogFmtHelper nvlog_fmt("uciOnPusch_deSeg_part2.log");
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
        cuphy::buffer<uint16_t, cuphy::pinned_alloc>                 uciUserIdxs_buffer(MAX_N_TBS_SUPPORTED);
        cuphy::buffer<cuphyPuschRxUeGrpPrms_t, cuphy::pinned_alloc>  drvdUeGrpPrmsBuffer(PDSCH_MAX_UE_GROUPS_PER_CELL_GROUP);

        //-----------------------------------------------------------------
        // Initialize GPU memory

        size_t max_nEqOutLLRs       =  273 * 12 * 14 * 8 * 2 * 10;      // maxPrb x scPerPrb x symbolPerSlot x maxLayers x maxQamBits x maxCells
        size_t max_nDeSegLLRs       =  max_nEqOutLLRs;
        size_t max_eqOutLLRs_mem    =  max_nEqOutLLRs * sizeof(__half);
        size_t max_nDeSegLLRs_mem   =  max_nDeSegLLRs * sizeof(__half);
        size_t max_tbPrms_mem       =  sizeof(PerTbParams) * MAX_N_TBS_SUPPORTED;
        size_t max_mem              =  max_eqOutLLRs_mem + max_nDeSegLLRs_mem + max_tbPrms_mem;

        cuphy::linear_alloc<128, cuphy::device_alloc> linearAlloc(max_mem);

        //------------------------------------------------------------------
        // Load API parameters

        StaticApiDataset  staticApiDataset(inputFilenameVec, cuStrmMain.handle());
        DynApiDataset     dynApiDataset(inputFilenameVec,    cuStrmMain.handle());
        EvalDataset       evalDataset(inputFilenameVec,      cuStrmMain.handle());

        cudaStreamSynchronize(cuStrmMain.handle()); // synch to ensure data copied

        uint16_t           nUes    = dynApiDataset.puschDynPrm.pCellGrpDynPrm->nUes;
        uint16_t           nUeGrps = dynApiDataset.cellGrpDynPrm.nUeGrps;
        cuphyPuschUePrm_t* pUePrms = dynApiDataset.puschDynPrm.pCellGrpDynPrm->pUePrms;

        //------------------------------------------------------------------
        // Derive API parameters

        PuschRx puschRx(&staticApiDataset.puschStatPrms, cuStrmMain.handle());

        uint8_t                 enableRssiMeasurement = 0;
        cuphyLDPCParams         ldpcPrms(&staticApiDataset.puschStatPrms);

        puschRx.expandFrontEndParameters(&dynApiDataset.puschDynPrm, drvdUeGrpPrmsBuffer.addr(), enableRssiMeasurement);
        puschRx.expandBackEndParameters(&dynApiDataset.puschDynPrm, drvdUeGrpPrmsBuffer.addr(), tbPrmsCpu_buffer.addr(), ldpcPrms);

        //------------------------------------------------------------------
        // Load CSI-P2 parameters  
        // In PUSCH reciever pipeline these will be computed using the CSI-P1 payload

        uint16_t nCsi2Ues                                =  evalDataset.nCsi2Ues;
        std::vector<uint16_t>& csi2UeIdxsVec             =  evalDataset.csi2UeIdxsVec; 
        std::vector<EvalDataset::uciSizes>& uciSizesVec  =  evalDataset.uciSizesVec; 

        for(int csi2Idx = 0; csi2Idx < nCsi2Ues; ++csi2Idx)
        {
            uint16_t ueIdx                     =  csi2UeIdxsVec[csi2Idx];
            tbPrmsCpu_buffer[ueIdx].G          =  uciSizesVec[csi2Idx].G;
            tbPrmsCpu_buffer[ueIdx].G_csi2     =  uciSizesVec[csi2Idx].G_csi2;
            tbPrmsCpu_buffer[ueIdx].nBitsCsi2  =  uciSizesVec[csi2Idx].nBitsCsi2;
        }

        //----------------------------------------------------------------------
        // Load GPU input buffers

        // equalizer output LLRs
        std::vector<cuphy::tensor_device>  tEqOutLLRsVec;
        std::vector<cuphy::tensor_ref>     tRefEqOutLLRsVec(nUeGrps);
        std::vector<cuphyTensorPrm_t>      tPrmEqOutLLRsVec(nUeGrps);

        for(int ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
        {
            int nLayers  = static_cast<int>(drvdUeGrpPrmsBuffer[ueGrpIdx].nLayers);
            int nSc      = static_cast<int>(CUPHY_N_TONES_PER_PRB * drvdUeGrpPrmsBuffer[ueGrpIdx].nPrb);
            int nDataSym = static_cast<int>(drvdUeGrpPrmsBuffer[ueGrpIdx].nDataSym);

            tEqOutLLRsVec.emplace_back(CUPHY_R_16F, evalDataset.eqOutLLRsRef[ueGrpIdx].layout());
            tEqOutLLRsVec[ueGrpIdx].convert(evalDataset.eqOutLLRsRef[ueGrpIdx], cuStrmMain.handle());

            tRefEqOutLLRsVec[ueGrpIdx].desc().set(CUPHY_R_16F, CUPHY_QAM_256, nLayers, nSc, nDataSym, cuphy::tensor_flags::align_tight);
            tRefEqOutLLRsVec[ueGrpIdx].set_addr(tEqOutLLRsVec[ueGrpIdx].addr());    

            tPrmEqOutLLRsVec[ueGrpIdx].desc  = tRefEqOutLLRsVec[ueGrpIdx].desc().handle();
            tPrmEqOutLLRsVec[ueGrpIdx].pAddr = tRefEqOutLLRsVec[ueGrpIdx].addr();
        }

        // copy ueGrp prms to GPU
        cuphyPuschRxUeGrpPrms_t* pUeGrpPrmsGpu = static_cast<cuphyPuschRxUeGrpPrms_t*>(linearAlloc.alloc(sizeof(cuphyPuschRxUeGrpPrms_t) * nUeGrps));
        cudaMemcpyAsync(pUeGrpPrmsGpu, drvdUeGrpPrmsBuffer.addr(), sizeof(cuphyPuschRxUeGrpPrms_t) * nUeGrps, cudaMemcpyHostToDevice, cuStrmMain.handle());

        // synch to ensure copies 
        cudaStreamSynchronize(cuStrmMain.handle());

        //------------------------------------------------------------------
        // Allocate GPU output buffers

        std::vector<__half*> schLLRsAddrVec(nCsi2Ues);
        std::vector<__half*> csi2LLRsAddrVec(nCsi2Ues);

        for(int csi2Idx = 0; csi2Idx < nCsi2Ues; ++csi2Idx)
        {
            uint16_t ueIdx  = csi2UeIdxsVec[csi2Idx];
            uint32_t G      = tbPrmsCpu_buffer[ueIdx].G;
            uint32_t G_csi2 = tbPrmsCpu_buffer[ueIdx].G_csi2;
            uint16_t nBitsHarq = tbPrmsCpu_buffer[ueIdx].nBitsHarq;

            tbPrmsCpu_buffer[ueIdx].d_schAndCsi2LLRs = static_cast<__half*>(linearAlloc.alloc((G + G_csi2) * sizeof(__half)));
        }

        //------------------------------------------------------------------
        // copy perTbParams to GPU
        
        PerTbParams* pTbPrmsGpu = static_cast<PerTbParams*>(linearAlloc.alloc(sizeof(PerTbParams) * nUes));
        cudaMemcpyAsync(pTbPrmsGpu, tbPrmsCpu_buffer.addr(), sizeof(PerTbParams) * nUes, cudaMemcpyHostToDevice, cuStrmMain.handle());

        //------------------------------------------------------------------
        // uciOnPuschSegLLRs2 descriptors

        // descriptors hold Kernel parameters in GPU
        size_t   dynDescrSizeBytes, dynDescrAlignBytes;
        cuphyStatus_t statusGetWorkspaceSize = cuphyUciOnPuschSegLLRs2GetDescrInfo(&dynDescrSizeBytes,
                                                                                   &dynDescrAlignBytes);
        if(CUPHY_STATUS_SUCCESS != statusGetWorkspaceSize) throw cuphy::cuphy_exception(statusGetWorkspaceSize);

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

        //------------------------------------------------------------------
        // Create uciOnPuschSegLLRs2 object

        cuphyUciOnPuschSegLLRs2Hndl_t uciOnPuschSegLLRs2Hndl;
        cuphyStatus_t statusCreate = cuphyCreateUciOnPuschSegLLRs2(&uciOnPuschSegLLRs2Hndl);

        if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);

        //------------------------------------------------------------------
        // setup uciOnPuschSegLLRs2

        // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
        cuphyUciOnPuschSegLLRs2LaunchCfg_t  uciOnPuschSegLLRs2LaunchCfg;

        // Setup function populates dynamic descriptor and launch config. Option to copy descriptors to GPU during setup call.
        bool enableCpuToGpuDescrAsyncCpy = false;

        cuphyStatus_t setupStatus = cuphySetupUciOnPuschSegLLRs2(uciOnPuschSegLLRs2Hndl,
                                                                 nCsi2Ues,
                                                                 csi2UeIdxsVec.data(),
                                                                 tbPrmsCpu_buffer.addr(),
                                                                 pTbPrmsGpu,
                                                                 nUeGrps,
                                                                 tPrmEqOutLLRsVec.data(),
                                                                 drvdUeGrpPrmsBuffer.addr(),
                                                                 pUeGrpPrmsGpu,               
                                                                 dynDescrBufCpu.addr(),                     
                                                                 dynDescrBufGpu.addr(), 
                                                                 enableCpuToGpuDescrAsyncCpy,            
                                                                 &uciOnPuschSegLLRs2LaunchCfg,                      
                                                                 cuStrmMain.handle());


        // if(CUPHY_STATUS_SUCCESS != setupStatus) throw cuphy::cuphy_exception(setupStatus);
        if(!enableCpuToGpuDescrAsyncCpy) {
            cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());
        }

        //------------------------------------------------------------------
        // run uciOnPuschSegLLRs2

        // Launch kernel using the CUDA driver API
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = uciOnPuschSegLLRs2LaunchCfg.kernelNodeParamsDriver;
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

        evalDataset.evalUciOnPuschSegLLRs2(nCsi2Ues, csi2UeIdxsVec.data(), tbPrmsCpu_buffer.addr(), cuStrmMain.handle());

        
        //------------------------------------------------------------------
        // save ChEst to h5

        std::unique_ptr<hdf5hpp::hdf5_file> dbgProbeUqPtr;
        if(!outputFilename.empty())
        {
            dbgProbeUqPtr.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(outputFilename.c_str())));

            uint32_t G_sch = tbPrmsCpu_buffer[0].G;
            if(G_sch > 0)
            {
                cuphy::tensor_ref tCuphySchLLR;
                tCuphySchLLR.desc().set(CUPHY_R_16F, static_cast<int>(G_sch), cuphy::tensor_flags::align_tight);
                tCuphySchLLR.set_addr(static_cast<void*>(schLLRsAddrVec[0]));

                cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tCuphySchLLR, "cuphySchLLRs");
            }

            uint32_t G_csi2 = tbPrmsCpu_buffer[0].G_csi2;
            cuphy::tensor_ref tCuphyCsi2LLR;
            tCuphyCsi2LLR.desc().set(CUPHY_R_16F, static_cast<int>(G_csi2), cuphy::tensor_flags::align_tight);
            tCuphyCsi2LLR.set_addr(static_cast<void*>(csi2LLRsAddrVec[0]));

            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tCuphyCsi2LLR, "cuphyCsi2LLRs");
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
