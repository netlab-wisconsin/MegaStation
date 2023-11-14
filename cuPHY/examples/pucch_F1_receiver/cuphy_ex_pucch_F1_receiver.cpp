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
    printf("pucch_F1_reciever [options]\n");
    printf("  Options:\n");
    printf("    -i  input_filename         Input yaml file for slot/cell config or HDF5 file for single cell example\n");
    printf("    -m  processing mode        PUCCH proc mode: streams (0x0), graphs (0x1)\n");
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    cuphyNvlogFmtHelper nvlog_fmt("pucch_F1_reciever.log");
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
        // F1 Uci parameters
        uint16_t            nF1Ucis    = dynPucchApiDataset.cellGrpDynPrm.nF1Ucis;
        cuphyPucchUciPrm_t* pF1UciPrms = dynPucchApiDataset.cellGrpDynPrm.pF1UciPrms;

        if(!nF1Ucis)
        {
            printf("\n No PF1 UCI received.\n");
            return 0;
        }

        // Input
        std::vector<cuphyTensorPrm_t>& tPrmDataRxVec = dynPucchApiDataset.tPrmDataRxVec;

        //------------------------------------------------------------------
        // Allocate F1 UCI output buffer in GPU memory

        cuphy::buffer<cuphyPucchF0F1UciOut_t, cuphy::device_alloc> bF1OutGpu(nF1Ucis);

        //------------------------------------------------------------------
        // Pucch F1 descriptors

        // descriptors hold Kernel parameters in GPU
        size_t   dynDescrSizeBytes, dynDescrAlignBytes;
        cuphyStatus_t statusGetWorkspaceSize = cuphyPucchF1RxGetDescrInfo(&dynDescrSizeBytes,
                                                                          &dynDescrAlignBytes);
        if(CUPHY_STATUS_SUCCESS != statusGetWorkspaceSize) throw cuphy::cuphy_exception(statusGetWorkspaceSize);

        // increase memory size to account for cell parameters
        int nCells = statPucchApiDataset.pucchStatPrms.nMaxCellsPerSlot;
        dynDescrSizeBytes += nCells * sizeof(cuphyPucchCellPrm_t);

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

        //-----------------------------------------------------------------------------------------
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
        // Create Pucch F1 receiver object

        cuphyPucchF1RxHndl_t pucchF1RxHndl;
        cuphyStatus_t statusCreate = cuphyCreatePucchF1Rx(&pucchF1RxHndl, cuStrmMain.handle());

        if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);

        //------------------------------------------------------------------
        // setup Pucch F1 reciver

        // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
        cuphyPucchF1RxLaunchCfg_t  pucchF1RxLaunchCfg;

        // setup function populates dynamic descriptor and launch config
        bool enableCpuToGpuDescrAsyncCpy = false;

        cuphyStatus_t pucchF1RxSetupStatus = cuphySetupPucchF1Rx( pucchF1RxHndl,
                                                                  tPrmDataRxVec.data(),
                                                                  bF1OutGpu.addr(),
                                                                  nCells,
                                                                  nF1Ucis,
                                                                  pF1UciPrms,
                                                                  cellCmnBufCpu.addr(),
                                                                  enableCpuToGpuDescrAsyncCpy,
                                                                  dynDescrBufCpu.addr(),                     
                                                                  dynDescrBufGpu.addr(),                     
                                                                  &pucchF1RxLaunchCfg,                      
                                                                  cuStrmMain.handle());     

        if(CUPHY_STATUS_SUCCESS != pucchF1RxSetupStatus) throw cuphy::cuphy_exception(pucchF1RxSetupStatus);

        if(!enableCpuToGpuDescrAsyncCpy) {
            cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());
        }

        //------------------------------------------------------------------
        // run pucch F1 reciever

        // launch kernel using the CUDA driver API
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pucchF1RxLaunchCfg.kernelNodeParamsDriver;
        CUresult pucchF1RxRunStatus = cuLaunchKernel(kernelNodeParamsDriver.func,
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
        if(CUDA_SUCCESS != pucchF1RxRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
        cudaStreamSynchronize(cuStrmMain.handle()); // synch to make sure kernel finishes

        // -------------------------------------------------------------------
        // Evaluate Pucch F1 reciver

        uint16_t nMismatches = evalPucchDataset.evalPucchF1Receiver(bF1OutGpu.addr(), cuStrmMain.handle());
        printf("\n comparing cuPHY F1 UCI output to reference output: %d mismatches out of %d UCIs\n", nMismatches, nF1Ucis);

        if (nMismatches) {
            returnValue = 1;
        }        

        // -------------------------------------------------------------------
        // cleanup

        cuphyStatus_t statusDestroy = cuphyDestroyPucchF1Rx(pucchF1RxHndl);
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
