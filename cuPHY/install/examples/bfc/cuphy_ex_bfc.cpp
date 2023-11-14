/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
#include <numeric>
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "util.hpp"
#include "test_config.hpp"
#include "datasets.hpp"
#include "cuphy_channels.hpp"

#include <chrono>
using Clock     = std::chrono::high_resolution_clock;
using TimePoint = std::chrono::time_point<Clock>;
template <typename T, typename unit>
using duration = std::chrono::duration<T, unit>;
template <typename T>
using ms = std::chrono::milliseconds;
template <typename T>
using us = std::chrono::microseconds;

////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("cuphy_ex_bfc [options]\n");
    printf("  Options:\n");
    printf("    -h                     Display usage information\n");
    printf("    -l  log_filename       filename to save log output\n");
    printf("    -i  input_filename     Input HDF5 or yaml filename\n");
    printf("    -m  processing mode    PUSCH proc mode: streams(0x0), graphs (0x1) (default = 0x0)\n");
    printf("    -r  # of iterations    Number of run iterations to run (default = 1000)\n");    
    printf("    -o  outfile            Write pipeline tensors to an HDF5 output file.\n");
    printf("                           (Not recommended for use during timing runs.)\n");
    printf("    -c  <ref check snr>    Enable reference check with optional SNR threshold used for reference checks (default: disabled, refCheckSnr = 30dB), [0, 300dB]\n");
    printf("    -w  delayMs            CPU latency hiding delay in milliseconds (default: 5)\n");
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    cuphyNvlogFmtHelper nvlog_fmt("bfw.log");
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int         iArg = 1;
        std::string inputFileName;
        std::string outputFileName;
        int32_t     nIterations = 1000;
        int32_t     delayMs     = 5;
        int         gpuId       = 0;
        bool        enableOutputFileLog = false;
        int         slotIdxInYaml = 0; // only 1 slot supported
        int         nCells = 0;
        uint32_t    nSlots = 1;
        float       refCheckSnrThd = 30.0f, inRefCheckSnrThd = 0.0f;
        bool        enableRefChecks = false;
        uint64_t    procModeBmsk    = 0;

        while(iArg < argc)
        {
            if('-' == argv[iArg][0])
            {
                switch(argv[iArg][1])
                {
                case 'i':
                    if(++iArg >= argc)
                    {
                        throw std::invalid_argument(fmt::format("No filename provided."));
                    }
                    inputFileName.assign(argv[iArg++]);
                    break;
                case 'h':
                    usage();
                    exit(0);
                    break;
                case 'o':
                    enableOutputFileLog = true;
                    if(++iArg < argc)
                    {
                        outputFileName.assign(argv[iArg++]);
                    }
                    break;                    
                case 'r':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nIterations)) || ((nIterations <= 0)))
                    {
                        throw std::invalid_argument(fmt::format("Invalid number of run iterations: {}", nIterations));
                    }
                    ++iArg;
                    break;
                case 'm':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%lu", &procModeBmsk)) || ((BFW_PROC_MODE_NO_GRAPH != procModeBmsk) && (BFW_PROC_MODE_WITH_GRAPH != procModeBmsk)))
                    {
                        throw std::invalid_argument(fmt::format("Invalid processing mode (0x{:x})", procModeBmsk));
                    }
                    ++iArg;
                    break;
                case 'c':
                    enableRefChecks = true;
                    if((++iArg < argc) && (1 == sscanf(argv[iArg], "%f", &inRefCheckSnrThd)) && (inRefCheckSnrThd > 0) && (inRefCheckSnrThd < 300))
                    {
                        refCheckSnrThd = inRefCheckSnrThd;
                        ++iArg;
                    }
                    break;
                case 'd':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &delayMs)) || ((delayMs <= 0)))
                    {
                        throw std::invalid_argument(fmt::format("Invalid delay value: {})", delayMs));
                    }
                    ++iArg;
                    break;                    
                case '-':
                    switch(argv[iArg][2])
                    {
                    default:
                        usage();
                        throw std::invalid_argument(fmt::format("Unknown option: {})", argv[iArg]));
                    }
                    break;
                default:
                    usage();
                    throw std::invalid_argument(fmt::format("Unknown option: {})", argv[iArg]));
                }
            }
            else
            {
                throw std::invalid_argument(fmt::format("Invalid command line argument: {})", argv[iArg]));
            }
        }
        if(inputFileName.empty())
        {
            throw std::invalid_argument("No valid filename provided");
        }

        //-------------------------------------------------------------------------------
        // Parse inputs
        std::vector<std::vector<std::string>> inFileNamesBfc; 
        std::string inFileExtn = inputFileName.substr(inputFileName.find_last_of(".") + 1);
        NVLOGC_FMT(NVLOG_BFW, "File extension: {}", inFileExtn);
        if(inFileExtn == "yaml")
        {
            cuphy::test_config testCfg(inputFileName.c_str());
            testCfg.print();
            nCells = testCfg.num_cells();
            nSlots = testCfg.num_slots();
            const std:: string bfcChannelName = "BFC";
            
            std::vector<std::string> bfcHdf5Filenames;
    
            inFileNamesBfc.resize(nSlots);
            for (int iSlot = 0; iSlot < nSlots; iSlot++) 
            {
                inFileNamesBfc[iSlot].resize(nCells);
                for (int iCell = 0; iCell < nCells; iCell += 1) 
                {
                    std::string bfcTvFilename = testCfg.slots()[iSlot].at(bfcChannelName)[iCell];        
                    inFileNamesBfc[iSlot][iCell] = bfcTvFilename;
                }
            }
            
            if(slotIdxInYaml >= nSlots)
            {
                NVLOGW_FMT(NVLOG_BFW, "Need slot index < number of slots in yaml ({}). Got slot index {}. Using slot index 0", nSlots, slotIdxInYaml);
                slotIdxInYaml = 0;
            }             
        }
        else
        {
            // Only single slot, single cell supported in vanilla HDF5 mode
            nCells = 1;
            inFileNamesBfc.resize(nCells);
            inFileNamesBfc[0].emplace_back(inputFileName);            
        }

        //-------------------------------------------------------------------------------

        // Stream for workload submission
        cuphy::stream cuphyStrm(cudaStreamNonBlocking);

        // Events for syncrhonization across streams
        std::vector<float> elapsedTimeUsCuphyEvts(nIterations, 0.0f);
        std::vector<cuphy::event_timer> cuphyEvtTimers(nIterations);
    
        //-------------------------------------------------------------------------------
        // Pipeline creation and setup
        // Note: test bench consumes the lambda value from the first file. lambda value assumed to be same across all TVs 
        bfwStaticApiDataset staticApiDataset(inFileNamesBfc[0], cuphyStrm.handle(), outputFileName);

#if 1  
        //------------------------------------------------------------------
        // Write outputs
        std::unique_ptr<hdf5hpp::hdf5_file> dbgProbeUqPtr;
        if(enableOutputFileLog)
        {
            std::string inFileName = inFileNamesBfc[0][0];//[cellIdx];

            // If possible use input file name as suffix for output
            size_t pos = inFileName.rfind('/', inFileName.length());
            if(pos != std::string::npos)
            {
//                outputFileName = "gpu_out_cell_" + std::to_string(cellIdx) + "_" + inFileName.substr(pos + 1, inFileName.length() - pos);
                printf("outputFileName: %s\n", outputFileName.c_str());
            }
            dbgProbeUqPtr.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(outputFileName.c_str())));
    
            // Write channel equalizer outputs
//            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutCoef, "Coef");
//            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutDbg, "Dbg");
            
            // Wait for writes to complete
//            cudaDeviceSynchronize();
        }
#endif        

        cuphyStrm.synchronize();

        std::vector<bfwDynApiDataset> dynApiDatasets; 
        std::vector<bfwEvalDataset> evalDatasets;
        std::vector<cuphy::bfw_tx> bfwTxPipelines;

        dynApiDatasets.reserve(nSlots);
        evalDatasets.reserve(nSlots);
        bfwTxPipelines.reserve(nSlots);
        for(int32_t iSlot = 0; iSlot < nSlots; ++iSlot)
        {
            dynApiDatasets.emplace_back(inFileNamesBfc[iSlot], cuphyStrm.handle(), procModeBmsk);
            evalDatasets.emplace_back(inFileNamesBfc[iSlot], cuphyStrm.handle());
            bfwTxPipelines.emplace_back(staticApiDataset.bfwStatPrms, cuphyStrm.handle());
            bfwTxPipelines[iSlot].setup(dynApiDatasets[iSlot].bfwDynPrms);
        }
        cuphyStrm.synchronize();

#if 0
        //------------------------------------------------------------------
        // 1. Vectors above invoke tensor copy constructor, tensor copy constructor's invocation of convert function uses default stream        
        // 2. Ensure all prior work on the GPU is completed before launching delay kernel (free up the internal FIFOs to accomodate as much
        // of the workload burst that follows the delay kernel)        
        cudaDeviceSynchronize();
#endif

        //-------------------------------------------------------------------------------
        // Execute pipeline
        for(int32_t iSlot = 0; iSlot < nSlots; ++iSlot)
        {
            // Insert short delay kernel
            gpu_ms_delay(delayMs, gpuId, cuphyStrm.handle());

            auto startWallClock = Clock::now();
            for(int32_t i = 0; i < nIterations; ++i)
            {
                // run launches on the same CUDA stream setup uses
                cuphyEvtTimers[i].record_begin(cuphyStrm.handle());
                bfwTxPipelines[iSlot].run(procModeBmsk);
                cuphyEvtTimers[i].record_end(cuphyStrm.handle());
            }

            // Wait for work to complete
            float totalElapsedTimeUsCuphyEvt = 0.0f;
            for(int32_t i = 0; i < nIterations; ++i)
            {
                cuphyEvtTimers[i].synchronize();
                elapsedTimeUsCuphyEvts[i] = cuphyEvtTimers[i].elapsed_time_ms()*1000;
                totalElapsedTimeUsCuphyEvt += elapsedTimeUsCuphyEvts[i];
            }
            cuphyStrm.synchronize();
            auto stopWallClock = Clock::now();
            duration<float, std::micro> diff = stopWallClock - startWallClock;            
            float elapsedTimeUsWallClk = diff.count();

            NVLOGC_FMT(NVLOG_BFW, "---------------------------------------------------------------");
            NVLOGC_FMT(NVLOG_BFW, "Slot[{}]: Average ({} runs) elapsed time in usec (CUDA event w/ {} ms delay kernel) = {:07.4f}",
                   iSlot,
                   nIterations,
                   delayMs,
                   totalElapsedTimeUsCuphyEvt / nIterations);
            //NVLOGC_FMT(NVLOG_BFW, "Average elapsed time wall clock {:07.4f}", elapsedTimeUsWallClk/nIterations); 

            //-------------------------------------------------------------------------------
            // Evaluate results
            if(enableRefChecks)
            {
                evalDatasets[iSlot].bfwEvalCoefs(dynApiDatasets[iSlot], cuphyStrm.handle(), refCheckSnrThd, enableRefChecks);
            }
            bfwTxPipelines[0].writeDbgSynch();
            cuphyStrm.synchronize();
            NVLOGC_FMT(NVLOG_BFW, "---------------------------------------------------------------");
        }

#if 0    
        //------------------------------------------------------------------
        // Write outputs
        std::unique_ptr<hdf5hpp::hdf5_file> dbgProbeUqPtr;
        if(enableOutputFileLog)
        {
            std::string inFileName = inFileNamesBfc[cellIdx];

            // If possible use input file name as suffix for output
            size_t pos = inFileName.rfind('/', inFileName.length());
            if(pos != std::string::npos)
            {
                outputFileName = "gpu_out_cell_" + std::to_string(cellIdx) + "_" + inFileName.substr(pos + 1, inFileName.length() - pos);
                NVLOGC_FMT(NVLOG_BFW, "outputFileName: {}", outputFileName);
            }
            dbgProbeUqPtr.reset(new hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::create(outputFileName.c_str())));
    
            // Write channel equalizer outputs
            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutCoef, "Coef");
            cuphy::write_HDF5_dataset(*dbgProbeUqPtr, tOutDbg, "Dbg");
            
            // Wait for writes to complete
            cudaDeviceSynchronize();
        }
#endif        
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    return returnValue;
}
