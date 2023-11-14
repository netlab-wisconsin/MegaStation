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
    printf("pucch_F3_segLLRs [options]\n");
    printf("  Options:\n");
    printf("    -i  input_filename         Input yaml file for slot/cell config or HDF5 file for single cell example\n");
    printf("    -m  processing mode        PUCCH proc mode: streams (0x0), graphs (0x1)\n");
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    cuphyNvlogFmtHelper nvlog_fmt("pucch_F3_segLLRs.log");
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
        // Extract PUCCH F3 parameters

        // F3 Uci parameters
        uint16_t                nF3Ucis         = dynPucchApiDataset.cellGrpDynPrm.nF3Ucis;
        cuphyPucchUciPrm_t*     pF3UciPrms      = dynPucchApiDataset.cellGrpDynPrm.pF3UciPrms;

        if(!nF3Ucis)
        {
            printf("\n No PF3 UCI received.\n");
            return 0;
        }

        // load descrambled LLR array for each PF3 UCI
        hdf5hpp::hdf5_file fInput                                      = hdf5hpp::hdf5_file::open(inputFilename.c_str());
        cuphy::typed_tensor<CUPHY_R_16F, cuphy::pinned_alloc> tRefLLRs = cuphy::typed_tensor_from_dataset<CUPHY_R_16F, cuphy::pinned_alloc>(fInput.open_dataset("pucchF234_refLLRbuffer"), cuphy::tensor_flags::align_tight, cuStrmMain.handle());
        hdf5hpp::hdf5_dataset F3offsetsH5                              = fInput.open_dataset("pucchF3_refBufferOffsets");

        std::vector<uint32_t> LLRsoffset; 
        LLRsoffset.resize(nF3Ucis);

        for(int uciIdx = 0; uciIdx < nF3Ucis; ++uciIdx)
        {
            cuphy::cuphyHDF5_struct offsets = cuphy::get_HDF5_struct_index(F3offsetsH5, uciIdx);
            LLRsoffset[uciIdx]              = offsets.get_value_as<uint32_t>("LLRsoffset");
        }

        //------------------------------------------------------------------
        // Allocate F3 UCI output buffer in GPU memory
        size_t max_mem = CUPHY_PUCCH_F3_MAX_UCI * CUPHY_PUCCH_F3_MAX_E * sizeof(__half);
        cuphy::linear_alloc<128, cuphy::device_alloc> linearAlloc(max_mem);
        std::vector<__half*>  descramLLRaddrsVecGPU(CUPHY_PUCCH_F3_MAX_UCI);
        std::vector<__half*>  descramLLRaddrsVecCPU(CUPHY_PUCCH_F3_MAX_UCI);

        uint16_t* E_tot;
        uint16_t* E_seg1;
        uint16_t* E_seg2;
        CUDA_CHECK(cudaHostAlloc( &E_tot, nF3Ucis*sizeof(uint16_t), 0));
        CUDA_CHECK(cudaHostAlloc( &E_seg1, nF3Ucis*sizeof(uint16_t), 0));
        CUDA_CHECK(cudaHostAlloc( &E_seg2, nF3Ucis*sizeof(uint16_t), 0));

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

            uint16_t A_seg1 = pF3UciPrms[uciIdx].bitLenHarq + pF3UciPrms[uciIdx].bitLenCsiPart1;

            uint8_t nBitsPerRe = pF3UciPrms[uciIdx].pi2Bpsk == 1? 1 : 2;

            E_tot[uciIdx] = nBitsPerRe * 12 * nSym_data * pF3UciPrms[uciIdx].prbSize;

            descramLLRaddrsVecCPU[uciIdx] = new __half[E_tot[uciIdx]];
            for (int b = 0; b < E_tot[uciIdx]; b++) {
                descramLLRaddrsVecCPU[uciIdx][b] = tRefLLRs(LLRsoffset[uciIdx]+b);
            }
            descramLLRaddrsVecGPU[uciIdx] = static_cast<__half*>(linearAlloc.alloc(E_tot[uciIdx]*sizeof(__half)));
            CUDA_CHECK(cudaMemcpyAsync(descramLLRaddrsVecGPU[uciIdx], descramLLRaddrsVecCPU[uciIdx], sizeof(__half) * E_tot[0], cudaMemcpyHostToDevice, cuStrmMain.handle()));
            cudaStreamSynchronize(cuStrmMain.handle());
        }


        //------------------------------------------------------------------
        // Pucch F3 SegLLRs kernel descriptors

        // descriptors hold Kernel parameters in GPU
        size_t   dynDescrSizeBytes, dynDescrAlignBytes;
        cuphyStatus_t statusGetWorkspaceSize = cuphyPucchF3SegLLRsGetDescrInfo(&dynDescrSizeBytes,
                                                                               &dynDescrAlignBytes);
        if(CUPHY_STATUS_SUCCESS != statusGetWorkspaceSize) throw cuphy::cuphy_exception(statusGetWorkspaceSize);

        cuphy::buffer<uint8_t, cuphy::pinned_alloc> dynDescrBufCpu(dynDescrSizeBytes);
        cuphy::buffer<uint8_t, cuphy::device_alloc> dynDescrBufGpu(dynDescrSizeBytes);

        //------------------------------------------------------------------
        // Create Pucch F3 SegLLRs kernel object

        cuphyPucchF3SegLLRsHndl_t pucchF3SegLLRsHndl;
        cuphyStatus_t statusCreate = cuphyCreatePucchF3SegLLRs(&pucchF3SegLLRsHndl);

        if(CUPHY_STATUS_SUCCESS != statusCreate) throw cuphy::cuphy_exception(statusCreate);

        //------------------------------------------------------------------
        // setup Pucch F3 SegLLRs kernel
        // Launch config holds everything needed to launch kernel using CUDA driver API or add it to a graph
        cuphyPucchF3SegLLRsLaunchCfg_t  pucchF3SegLLRsLaunchCfg;

        // setup function populates dynamic descriptor and launch config
        bool enableCpuToGpuDescrAsyncCpy = false;

        cuphyStatus_t pucchF3SegLLRsSetupStatus = cuphySetupPucchF3SegLLRs(pucchF3SegLLRsHndl,
                                                                           nF3Ucis,
                                                                           pF3UciPrms,
                                                                           descramLLRaddrsVecGPU.data(),
                                                                           dynDescrBufCpu.addr(),                     
                                                                           dynDescrBufGpu.addr(),
                                                                           enableCpuToGpuDescrAsyncCpy,                   
                                                                           &pucchF3SegLLRsLaunchCfg,                      
                                                                           cuStrmMain.handle()); 

        if(CUPHY_STATUS_SUCCESS != pucchF3SegLLRsSetupStatus) throw cuphy::cuphy_exception(pucchF3SegLLRsSetupStatus);              

        if(!enableCpuToGpuDescrAsyncCpy) {
            cudaMemcpyAsync(dynDescrBufGpu.addr(), dynDescrBufCpu.addr(), dynDescrSizeBytes, cudaMemcpyHostToDevice, cuStrmMain.handle());
            cudaStreamSynchronize(cuStrmMain.handle());
        }                                            

        //------------------------------------------------------------------
        // run pucch F3 SegLLRs kernel

        // launch kernel using the CUDA driver API
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = pucchF3SegLLRsLaunchCfg.kernelNodeParamsDriver;
        CUresult pucchF3SegLLRsRunStatus = cuLaunchKernel(kernelNodeParamsDriver.func,
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
        if(CUDA_SUCCESS != pucchF3SegLLRsRunStatus) throw cuphy::cuphy_exception(CUPHY_STATUS_INTERNAL_ERROR);
        cudaStreamSynchronize(cuStrmMain.handle()); // synch to make sure kernel finishes
        
        // -------------------------------------------------------------------
        // Evaluate Pucch F3 SegLLRs kernel
        uint16_t nMismatches = evalPucchDataset.evalPucchF3SegLLRs(descramLLRaddrsVecGPU.data(), E_seg1, E_seg2, cuStrmMain.handle());
        printf("\n comparing cuPHY F3 UCI output to reference output: %d mismatches out of %d UCIs\n", nMismatches, nF3Ucis);

        if (nMismatches) {
            returnValue = 1;
        }        

        // -------------------------------------------------------------------
        // cleanup

        cuphyStatus_t statusDestroy = cuphyDestroyPucchF3SegLLRs(pucchF3SegLLRsHndl);
        if(CUPHY_STATUS_SUCCESS != statusDestroy) throw cuphy::cuphy_exception(statusDestroy);

        CUDA_CHECK(cudaFreeHost(E_tot));
        CUDA_CHECK(cudaFreeHost(E_seg1));
        CUDA_CHECK(cudaFreeHost(E_seg2));
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