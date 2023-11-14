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

#define CUPHY_PUCCH_F3_MAX_E (4608)

////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("pucch_F3_front_end [options]\n");
    printf("  Options:\n");
    printf("    -i  input_filename         Input yaml file for slot/cell config or HDF5 file for single cell example\n");
    printf("    -m  processing mode        PUCCH proc mode: streams (0x0), graphs (0x1)\n");
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    cuphyNvlogFmtHelper nvlog_fmt("pucch_F3_front_end.log");
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
                case 'm':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%lu", &procModeBmsk)) || ((procModeBmsk != PUCCH_PROC_MODE_FULL_SLOT) && (procModeBmsk != PUCCH_PROC_MODE_FULL_SLOT_GRAPHS)))
                    {
                        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid processing mode ({})", procModeBmsk);
                        exit(1);
                    }
                    ++iArg;
                    if (procModeBmsk == PUCCH_PROC_MODE_FULL_SLOT_GRAPHS)  NVLOGI_FMT(NVLOG_PUCCH, "CUDA graph enabled!");
                    break;
                default:
                    NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                    usage();
                    exit(1);
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

        //---------------------------------------------------------------------
        // F3 Uci parameters
        uint16_t                nF3Ucis         = dynPucchApiDataset.cellGrpDynPrm.nF3Ucis;
        cuphyPucchUciPrm_t*     pF3UciPrms      = dynPucchApiDataset.cellGrpDynPrm.pF3UciPrms;

        if(!nF3Ucis)
        {
            printf("\n No PF3 UCI received.\n");
            return 0;
        }

        // Input
        std::vector<cuphyTensorPrm_t>& tPrmDataRxVec = dynPucchApiDataset.tPrmDataRxVec;

        //------------------------------------------------------------------
        // Allocate F3 UCI output buffer in GPU memory
        size_t max_mem = CUPHY_PUCCH_F3_MAX_UCI + CUPHY_PUCCH_F3_MAX_UCI * CUPHY_PUCCH_F3_MAX_E * sizeof(__half);
        cuphy::linear_alloc<128, cuphy::device_alloc> linearAlloc(max_mem);
        std::vector<__half*>  descramLLRaddrsVec(CUPHY_PUCCH_F3_MAX_UCI);
    //    CUDA_CHECK(cudaHostAlloc( &descramLLRaddrsVec, nF3Ucis*sizeof(__half*), 0));

        uint8_t* pDTXflags;
        pDTXflags = static_cast<uint8_t*>(linearAlloc.alloc(nF3Ucis*sizeof(uint8_t)));

        float* pSinr;
        pSinr = static_cast<float*>(linearAlloc.alloc(nF3Ucis*sizeof(float)));

        float* pRssi;
        pRssi = static_cast<float*>(linearAlloc.alloc(nF3Ucis*sizeof(float)));

        float* pRsrp;
        pRsrp = static_cast<float*>(linearAlloc.alloc(nF3Ucis*sizeof(float)));

        float* pInterf;
        pInterf = static_cast<float*>(linearAlloc.alloc(nF3Ucis*sizeof(float)));

        float* pNoiseVar;
        pNoiseVar = static_cast<float*>(linearAlloc.alloc(nF3Ucis*sizeof(float)));

        float* pTaEst;
        pTaEst = static_cast<float*>(linearAlloc.alloc(nF3Ucis*sizeof(float)));

        uint16_t* E_tot;
        CUDA_CHECK(cudaHostAlloc( &E_tot, nF3Ucis*sizeof(uint16_t), 0));

        for (int uciIdx = 0; uciIdx<nF3Ucis; uciIdx++)
        {
            uint8_t nSym_data  = 0;
            uint8_t nSym       = pF3UciPrms[uciIdx].nSym;
            switch (int(nSym))
            {
                case 4:
                    if (pF3UciPrms[uciIdx].freqHopFlag) {
                        nSym_data = 2;
                    } else {
                        nSym_data = 3;
                    }
                    break;
                case 5:
                case 6:
                case 7:
                case 8:
                case 9:
                    nSym_data = nSym - 2;  
                    break;
                case 10:
                case 11:
                case 12:
                case 13:
                case 14:
                    if (pF3UciPrms[uciIdx].AddDmrsFlag) {
                        nSym_data = nSym - 4;
                    } else {
                        nSym_data = nSym - 2;
                    }
                    break;
                default:
                    NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of symbols for PUCCH format 3");
                    exit(1);
                    break;
            }

            if (pF3UciPrms[uciIdx].pi2Bpsk) {
                E_tot[uciIdx] = 12 * nSym_data * pF3UciPrms[uciIdx].prbSize;
            } else {
                E_tot[uciIdx] = 24 * nSym_data * pF3UciPrms[uciIdx].prbSize;
            }
            descramLLRaddrsVec[uciIdx] = static_cast<__half*>(linearAlloc.alloc(E_tot[uciIdx]*sizeof(__half)));

        }


        //------------------------------------------------------------------
        // Pucch F3 descriptors

        // descriptors hold Kernel parameters in GPU
        size_t   dynDescrSizeBytes, dynDescrAlignBytes;
        cuphyStatus_t statusGetWorkspaceSize = cuphyPucchF3RxGetDescrInfo(&dynDescrSizeBytes,
                                                                          &dynDescrAlignBytes);
        if(CUPHY_STATUS_SUCCESS != statusGetWorkspaceSize) throw cuphy::cuphy_exception(statusGetWorkspaceSize);

        // increase memory size to account for cell parameters
        int nCells = statPucchApiDataset.pucchStatPrms.nMaxCellsPerSlot;
        dynDescrSizeBytes += nCells * sizeof(cuphyPucchCellPrm_t);

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

        //------------------------------------------------------------------
        // could use extra allocated space in dynDescrBufCpu instead of allocating cellCmnBufCpu
        cuphy::buffer<cuphyPucchCellPrm_t , cuphy::pinned_alloc> cellCmnBufCpu(nCells);

        // extract cell parameters and copy to GPU
        for(int i = 0; i < nCells; i++)
        {
            cellCmnBufCpu[i].nRxAnt         = statPucchApiDataset.cellStatPrm[i].nRxAnt;
            cellCmnBufCpu[i].pucchHoppingId = dynPucchApiDataset.cellDynPrm[i].pucchHoppingId;
            cellCmnBufCpu[i].slotNum        = dynPucchApiDataset.cellDynPrm[i].slotNum;
        }

        //------------------------------------------------------------------
        // Create Pucch F3 receiver object

        cuphyPucchF3RxHndl_t pucchF3RxHndl;
        cuphyStatus_t statusCreate = cuphyCreatePucchF3Rx(&pucchF3RxHndl, cuStrmMain.handle());

        if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);

        //------------------------------------------------------------------
        // setup Pucch F3 reciver
        // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
        cuphyPucchF3RxLaunchCfg_t  pucchF3RxLaunchCfg;

        // setup function populates dynamic descriptor and launch config
        bool enableCpuToGpuDescrAsyncCpy = false;

        cuphyStatus_t pucchF3RxSetupStatus = cuphySetupPucchF3Rx( pucchF3RxHndl,
                                                                  tPrmDataRxVec.data(),
                                                                  descramLLRaddrsVec.data(),
                                                                  pDTXflags,
                                                                  pSinr,
                                                                  pRssi,
                                                                  pRsrp,
                                                                  pInterf,
                                                                  pNoiseVar,
                                                                  pTaEst,
                                                                  nCells,
                                                                  nF3Ucis,
                                                                  pF3UciPrms,
                                                                  cellCmnBufCpu.addr(),
                                                                  enableCpuToGpuDescrAsyncCpy,
                                                                  dynDescrBufCpu.addr(),                     
                                                                  dynDescrBufGpu.addr(),                     
                                                                  &pucchF3RxLaunchCfg,                      
                                                                  cuStrmMain.handle()); 

        if(CUPHY_STATUS_SUCCESS != pucchF3RxSetupStatus) throw cuphy::cuphy_exception(pucchF3RxSetupStatus);              

        if(!enableCpuToGpuDescrAsyncCpy) {
            cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());
        }                                            

        //------------------------------------------------------------------
        // run pucch F3 reciever

        // launch kernel using the CUDA driver API
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pucchF3RxLaunchCfg.kernelNodeParamsDriver;
        CUresult pucchF3RxRunStatus = cuLaunchKernel(kernelNodeParamsDriver.func,
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
        if(CUDA_SUCCESS != pucchF3RxRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
        cudaStreamSynchronize(cuStrmMain.handle()); // synch to make sure kernel finishes
        
        // -------------------------------------------------------------------
        // Evaluate Pucch F3 receiver
        uint16_t nMismatches = evalPucchDataset.evalPucchF3FrontEnd(descramLLRaddrsVec.data(), pDTXflags, E_tot, cuStrmMain.handle());
        printf("\n comparing cuPHY F3 UCI output to reference output: %d mismatches out of %d UCIs\n", nMismatches, nF3Ucis);

        if (nMismatches) {
            returnValue = 1;
        }        

        // -------------------------------------------------------------------
        // cleanup

        cuphyStatus_t statusDestroy = cuphyDestroyPucchF3Rx(pucchF3RxHndl);
        if(CUPHY_STATUS_SUCCESS != statusDestroy) throw cuphy::cuphy_exception(statusDestroy);

        CUDA_CHECK(cudaFreeHost(E_tot));
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