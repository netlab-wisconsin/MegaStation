/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <atomic>
#include "nvlog.hpp"

#include "cuda_profiler_api.h"
#include "util.hpp"
#include "pusch_rx_test.hpp"
#include "test_config.hpp"

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
    printf("cuphy_ex_pusch_rx_multi_pipe [options]\n");
    printf("  Options:\n");
    printf("    -h                     Display usage information\n");
    printf("    -d                     Enable debug\n");
    printf("    -D                     Enable de-rate-match debug\n");
    printf("    -i  input_filenames    Input filename (h5 for single-cell, yaml for multi-cell)\n");
    printf("    -l  log_filename       filename to save log output\n");
    printf("    -n  # of pipelines     Number of pipelines to run (currently only 1 supported)\n");
    printf("    -c  CPU Id             CPU Id used to run the first pipeline, cpuIdPipeline[i] = cpuIdFirstPipeline + i\n");
    printf("    -g  GPU Id             GPU Id used to run all the pipelines\n");
    printf("    -m  processing mode    PUSCH proc mode: streams (0x0), graphs (0x1), early-HARQ sub-slot on streams (0x2), early-HARQ on graphs (0x3)\n");
    printf("    -K  LDPC launch mode   LDPC kernel launch mode: single stream driver api (0) [default], stream-pool (1), single stream (2), single stream opt (3); This option will be ignored if graph mode selected.\n");
    printf("    -o  outfile            Write pipeline tensors to an HDF5 output file.\n");
    printf("                           (Not recommended for use during timing runs.)\n");
    printf("    -r  # of iterations    Number of run iterations to run\n");
    printf("    -v                     Enable nvprof profiling with 1 iteration\n");
    printf("    -w  delay_ms           Set the initial GPU delay in milliseconds (default: 2000)\n");
    printf("    --M <activeThrdPcts>   Enables use of cuda-contexts (one per thread) and uses a list of comma seperated values for active thread percentage\n");
    printf("    --H                    Use half precision (FP16)\n");
    printf("    -H <harq attempts>     Tests PUSCH HARQ by incrementally testing input HDF5 filenames of the form filename_s#p0.h5. \n");
    printf("    --S slot index in yaml Slot index in yaml file to use\n");
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;

    char nvlog_yaml_file[1024];
    // Relative path from binary to default nvlog_config.yaml
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    std::string log_name = "pusch.log";
    nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
    pthread_t log_thread_id = -1;

    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        constexpr uint32_t N_MAX_INST = 1;// 36;
        int                iArg       = 1;
        std::string        inputFileName;
        std::string        outputFileName;
        bool               enableOutputFileLog = false;
        bool               b16                 = false;
        uint32_t           nInst = 1;
        int32_t            nGPUs = 0;
        CUDA_CHECK(cudaGetDeviceCount(&nGPUs));
        int32_t  nMaxConcurentThrds = std::thread::hardware_concurrency();
        int32_t  cpuIdFirstInst     = 0;
        int32_t  gpuId              = 0;
        uint32_t nIterations        = 1000;
        bool     enable_nvprof      = false;
        bool     debug              = false;
        int      descramblingOn     = 1;
        uint32_t fp16Mode           = 1;
        bool     useCuCtxs          = false;
        uint64_t procModeBmsk       = 0;
        uint32_t ldpcLaunchMode     = 0;
        std::vector<uint32_t> mpsActiveThrdPcts;
        uint32_t defaultMpsActiveThrdPct = 100;
        uint32_t harq_attempts = 1;
        bool drmDebug = false;
        int slotIdxInYaml = -1; // to specify a single slot

        // Note: this value needs to be derived empirically by measuring maxIdealDelayKernelTimeUs (see PuschRxTest::DisplayTiming)
        // For e.g. for a 16 cell F14 TV this value is ~200us on DGX-A100
        uint32_t delay_ms = 1;
                                         
        while(iArg < argc)
        {
            if('-' == argv[iArg][0])
            {
                switch(argv[iArg][1])
                {
                case 'i':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided.");
                    }
                    inputFileName.assign(argv[iArg++]);
                    break;
                case 'l':
                    if(++iArg < argc)
                    {
                        log_name.assign(argv[iArg++]);
                    }
                    break;
                case 'h':
                    usage();
                    exit(0);
                    break;
                case 'p':
                    b16 = true;
                    ++iArg;
                    break;
                case 'n':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &nInst)) ||
                       ((nInst <= 0) || (nInst > N_MAX_INST)))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of instances (should be within [1,{}])", N_MAX_INST);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'r':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nIterations)) || ((nIterations <= 0)))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of run iterations");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'v':
                    enable_nvprof = true;
                    nIterations   = 1;
                    ++iArg;
                    break;
                case 'g':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &gpuId)) ||
                       ((gpuId < 0) || (gpuId >= nGPUs)))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid GPU Id (should be within [0,{}])", nGPUs - 1);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'w':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%u", &delay_ms)))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid delay");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'c':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &cpuIdFirstInst)) ||
                       ((cpuIdFirstInst < 0) || (cpuIdFirstInst > nMaxConcurentThrds - nInst)))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid CPU Id (should be within [0,{}], nMaxConcurrentThrds {})", nMaxConcurentThrds - nInst, nMaxConcurentThrds);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'd':
                    debug = true;
                    ++iArg;
                    break;
                case 'D':
                    drmDebug = true;
                    ++iArg;
                    break;
                case 'o':
                    enableOutputFileLog = true;
                    if(++iArg < argc)
                    {
                        outputFileName.assign(argv[iArg++]);
                    }
                    break;
                case 'm':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%lu", &procModeBmsk)) || (procModeBmsk > PUSCH_MAX_PROC_MODES))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid processing mode ({})", procModeBmsk);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'K':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%u", &ldpcLaunchMode)) || (3 < ldpcLaunchMode))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid LDPC kernel launch mode ({})", ldpcLaunchMode);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'H':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: No number of HARQ attempts provided");
                    }
                    if (1 != sscanf(argv[iArg], "%u", &harq_attempts))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Unable to parse harq attempts");
                        exit(1);
                    }
                    printf("HARQ attempts: %u\n",harq_attempts);
                    ++iArg;
                    break;
                case '-':
                    switch(argv[iArg][2])
                    {
                    case 'H':
                        if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &fp16Mode)) || (2 < fp16Mode))
                        {
                            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid FP16 mode {}", fp16Mode);
                            exit(1);
                        }
                        ++iArg;
                        break;
                    case 'M':
                        useCuCtxs = true;
                        if(++iArg >= argc || (1 != sscanf(argv[iArg], "%i", &fp16Mode)))
                        {
                            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "Using default ({}) MPS thread percentages for all sub-contexts", fp16Mode, defaultMpsActiveThrdPct);
                        }
                        else
                        {
                            std::string mpsActiveThrdPctsStr;
                            mpsActiveThrdPctsStr.assign(argv[iArg++]);
                            std::vector<char> str(mpsActiveThrdPctsStr.begin(), mpsActiveThrdPctsStr.end());
                            // const char* str = mpsActiveThrdPctsStr.c_str();
                            char* pChar = strtok(str.data(),",");
                            while (pChar != NULL)
                            {
                                mpsActiveThrdPcts.emplace_back(std::stoul(pChar));
                                // printf("%s %d\n",pChar, mpsActiveThrdPcts.back());
                                pChar = strtok(NULL, ",");
                            }
                            // printf("MPS percentages:\n");
                            // for(auto &mpsActiveThrdPct : mpsActiveThrdPcts) {printf("%d\n", mpsActiveThrdPct);}
                        }
                        break;
                    case 'S':
                        if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &slotIdxInYaml)) || (slotIdxInYaml < 0))
                        {
                            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid yaml slot index {}", slotIdxInYaml);
                            exit(1);
                        }
                        ++iArg;
                        break;                                                
                    default:
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                        usage();
                        exit(1);
                        break;
                    }
                    break;
                default:
                    NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                    usage();
                    exit(1);
                    break;
                }
            }
            else // if('-' == argv[iArg][0])
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
                exit(1);
            }
        } // while (iArg < argc)

        if(inputFileName.empty())
        {
            usage();
            exit(1);
        }
        
        log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, log_name.c_str(),NULL);
        nvlog_fmtlog_thread_init();

        if(procModeBmsk & PUSCH_PROC_MODE_FULL_SLOT_GRAPHS)
        {
            NVLOGI_FMT(NVLOG_PUSCH, "CUDA graph enabled!");
        } else {
            NVLOGI_FMT(NVLOG_PUSCH, "CUDA stream mode");
        }

        if(procModeBmsk & PUSCH_PROC_MODE_SUB_SLOT_EARLY_HARQ)
        {
            NVLOGI_FMT(NVLOG_PUSCH, "early-HARQ sub-slot processing enabled!");
        }

        std::vector<std::vector<std::string>> inputFileNameVec;
        std::string inFileExtn = inputFileName.substr(inputFileName.find_last_of(".") + 1);
        printf("File extension: %s\n", inFileExtn.c_str());
        if(inFileExtn == "yaml")
        {
            cuphy::test_config testCfg(inputFileName.c_str());
            testCfg.print();
            int nCells = testCfg.num_cells();
            int nSlots = testCfg.num_slots();
            const std:: string channelName = "PUSCH";

            if (slotIdxInYaml != -1)
            {
                inputFileNameVec.resize(1);
                if(slotIdxInYaml >= nSlots)
                {
                    NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "WARNING: Need slot index < number of slots in yaml ({}). Got slot index {}. Using slot index 0", nSlots, slotIdxInYaml);
                    slotIdxInYaml = 0;
                }

                for(int i = 0; i < nCells; i += 1)
                {
                    std::string tvFilename = testCfg.slots()[slotIdxInYaml].at(channelName)[i];
                    inputFileNameVec[0].emplace_back(tvFilename);
                }
            }
            else
            {
                inputFileNameVec.resize(nSlots);
                for (int slotIdx = 0; slotIdx < nSlots; slotIdx++)
                {
                    for(int i = 0; i < nCells; i += 1)
                    {
                        std::string tvFilename = testCfg.slots()[slotIdx].at(channelName)[i];
                        inputFileNameVec[slotIdx].emplace_back(tvFilename);
                    }
                }
            }
#if 0
            if(nInst != nCells)
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Mismatch in number of cells ({}) and pipeline instances ({})", nCells, nInst);
                exit(1);
            }
#endif
        }
        else
        {
            inputFileNameVec.resize(1);
            for(int i = 0; i < nInst; ++i)
                inputFileNameVec[0].emplace_back(inputFileName);
        }

        if(enableOutputFileLog && outputFileName.empty())
        {
            // If possible use input file name as suffix for output
            size_t startPos = inputFileName.rfind('/', inputFileName.length());
            if(startPos != std::string::npos)
            {
                size_t extnLen = inputFileName.length() - inputFileName.find_last_of(".") + 1;
                size_t endPos  = inputFileName.length() - startPos - extnLen;
                outputFileName = "gpu_out_" + inputFileName.substr(startPos + 1, endPos) + ".h5";
                
                printf("outputFileName: %s\n", outputFileName.c_str());
            }
        }

        if((useCuCtxs) && (mpsActiveThrdPcts.empty() || mpsActiveThrdPcts.size() < nInst))
        {
            for(uint32_t i = mpsActiveThrdPcts.size(); i < nInst; ++i) mpsActiveThrdPcts.emplace_back(defaultMpsActiveThrdPct);
        }

        cudaStream_t cuStream;
        cudaStreamCreateWithFlags(&cuStream, cudaStreamNonBlocking);

        // DynApiDataset a(inputFilename,cuStream);
        // StaticApiDataset b(inputFilename, cuStream);
        // EvalDataset c(inputFilename, cuStream);

        //output file names (For now only 0'th pipeline)
        std::vector<std::string> outputFileNameVec(nInst);
        outputFileNameVec[0] = outputFileName;

        int              cuStrmPrio = PUSCH_STREAM_PRIORITY;
        std::vector<int> cuStrmPrios(nInst, cuStrmPrio);
        cudaStream_t     delayCuStrm = 0;

        bool                       startSyncPt = false;
        std::mutex                 cvStartSyncPtMutex;
        std::condition_variable    cvStartSyncPt;
        std::atomic<std::uint32_t> atmSyncPtWaitCnt(0);
        PuschRxTest                puschRxTest("PuschRx", nInst, useCuCtxs, delayCuStrm, cuStrmPrios, startSyncPt, cvStartSyncPtMutex, cvStartSyncPt, atmSyncPtWaitCnt, mpsActiveThrdPcts, harq_attempts, 1<<ldpcLaunchMode, drmDebug, debug);

        cudaSetDevice(gpuId);


        // GPU and CPU Ids
        std::vector<int> gpuIds(nInst, gpuId);
        std::vector<int> cpuIds(nInst);
        uint32_t         instCpuId = 0;
        for(auto& cpuId : cpuIds) { cpuId = instCpuId + cpuIdFirstInst; }

        // Scheduling policy and priorities
        std::vector<int> schdPolicyVec(nInst, SCHED_RR); // SCHED_FIFO // SCHED_RR
        std::vector<int> prioVec;                        // 99
        for(auto const& policy : schdPolicyVec)
        {
            prioVec.emplace_back(sched_get_priority_max(policy));
        }

        puschRxTest.Setup(nIterations, enable_nvprof, descramblingOn, delay_ms, 0, cpuIds, schdPolicyVec, prioVec, gpuIds, inputFileNameVec, outputFileNameVec, procModeBmsk, fp16Mode);

        // Allow all worker threads to arrive at start sync point
        uint32_t syncPtWaitCnt = 0;
        uint32_t nSyncPtWaits  = nInst;
        while((syncPtWaitCnt = atmSyncPtWaitCnt.load(std::memory_order_seq_cst)) < nSyncPtWaits)
        {
            std::this_thread::sleep_for(std::chrono::milliseconds(100));
        }

        printf("Run config: GPU Id %d, # of pipelines %d\n", gpuId, nInst);

        // Release start sync point
        {
            std::lock_guard<std::mutex> cvStartSyncPtMutexGuard(cvStartSyncPtMutex);
            startSyncPt = true;
        }
        cvStartSyncPt.notify_all();

        puschRxTest.WaitForCompletion();
       // puschRxTest.DisplayMetricsApi(debug);

    }

    catch(std::exception& e)
    {
        if(log_thread_id < 0)
        {
            log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, log_name.c_str(),NULL);
            nvlog_fmtlog_thread_init();
        }
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {   
        if(log_thread_id < 0)
        {
            log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, log_name.c_str(),NULL);
            nvlog_fmtlog_thread_init();
        }
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    nvlog_fmtlog_close(log_thread_id);
    return returnValue;
}
