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
#include <iostream>
#include <sstream>
#include <fstream>
#include <string>
#include <vector>
#include <thread>
#include <mutex>
#include <memory>
#include <queue>
#include <unordered_map>
#include <condition_variable>
#include <atomic>
#include <unistd.h>  /* For SYS_xxx definitions */
#include <syscall.h> /* For SYS_xxx definitions */

#include "util.hpp"
#include "cuphy.hpp"
#include "cuphy_channels.hpp"
#include "psch_rx_tx_cmn.hpp"
#include "test_config.hpp"

// using namespace cuphy;

// NOTE: cannot define vectors of cuPHYTestWorker yet (need copy constructors). Until then, need use seperate function to finish simulation after creation.
typedef std::vector<std::vector<std::vector<std::string>>> string3Dvec_t;

void finishPschSim(std::vector<cuPHYTestWorker*>& pTestWorkers, string3Dvec_t& inCtxFileNamesPuschRx, string3Dvec_t& inCtxFileNamesPdschTx, uint32_t nTimingItrs, uint32_t nPowerItrs, uint32_t num_patterns, uint32_t nCtxts, bool printCbErrors, cuphy::stream& mainStream, int32_t gpuId, uint32_t delayUs, uint32_t powerDelayUs, bool ref_check_pdsch, bool identical_ldpc_configs, cuphyPdschProcMode_t pdsch_proc_mode, uint64_t pusch_proc_mode, uint32_t fp16Mode, int descramblingOn, uint32_t nCellsPerCtxt, uint32_t nStrmsPerCtxt, uint32_t nPschItrsPerStrm, cuphy::event_timer& slotPatternTimer, float tot_slotPattern_time, float avg_slotPattern_time_us, std::shared_ptr<cuphy::buffer<uint32_t, cuphy::pinned_alloc>>& shPtrGpuStartSyncFlag, std::map<std::string, int>& cuStrmPrioMap, uint32_t syncUpdateIntervalCnt, bool enableLdpcThroughputMode, bool printCellMetrics, bool pdsch_group_cells, uint32_t pdsch_cells_per_stream, bool pusch_group_cells, maxPDSCHPrms pdschPrms, maxPUSCHPrms puschPrms, uint32_t ldpcLaunchMode);
void readSmIds(std::vector<cuPHYTestWorker*>& pTestWorkers, cuphy::stream& cuphyStrm);

//----------------------------------------------------------------------------------------------------------
// usage
//----------------------------------------------------------------------------------------------------------
void usage()
{
    printf("cuphy_ex_sch_rx_tx [options]\n");
    printf("  Options:\n");
    printf("    -h                     Display usage information\n");
    printf("    -d                     Debug message setting: Disable(0), Enable(> 0)\n");
    printf("    -k                     ref_check for PDSCH\n");
    printf("    -l                     Force separate LDPC kernels, one per transport block, for PDSCH\n");
    printf("    -m  process mode       streams(0), graphs (1).\n");
    printf("    -u                     uldl mode: -u 3 uses one UL and one DL context at a miminum, DL context processes 4 slots in serial\n");
    printf("                           uldl mode: -u 4 uses -C UL/DL contexts. Per slot each context has -S pipelines run -I times sequentially\n");
    printf("                           uldl mode: -u 5 DDDSUUDDDD\n");
    printf("    -i  yaml input_file    Input yaml filename\n");
    printf("    -c  CPU Id             CPU Id used to run the first pipeline (default 0), (cpuIdPipeline[ii] = CPU Id, or cpuIdPipeline[ii] = CPU Id + ii if multi-core enabled)\n");
    printf("    -t                     Enables multi-core threading\n");
    printf("    -g  GPU Id             GPU Id used to run all the pipelines\n");
    printf("    -o  outfile            Write pipeline tensors to an HDF5 output file.\n");
    printf("                           (Not recommended for use during timing runs.)\n");
    printf("    -r  # of iterations    Number of run iterations to run (set to 1 internally for power measurements mode -P)\n");
    printf("    -w  delayUs            Set the initial GPU delay in microseconds (default: 10000)\n");
    printf("    -b                     Option to print codeblock errors\n");
    printf("    -S  <streams>          number of streams per worker\n");
    printf("    -I  <iterations>       number of iterations per stream\n");
    printf("    -C  <contexts>         number of contexts \n");
    printf("    -P  <iterations>       if > 0 enables power measurement mode (default: disabled). When enabled, min iteration count forced to 100)\n");
    printf("    -W  <delay (us)>       Inter slot pattern workload delay in microseconds for power measurement mode (default: 30)\n");
    printf("    -L                     Force selection of throughput mode for PUSCH LDPC decoder\n");
    printf("    -K                     LDPC kernel launch mode: -K 0, uses single stream using driver api (default);-K 1, uses multi-stream launch;\n");
    printf("                           -K 2, uses single-stream launch via tensor interface; -K 3, single stream opt;\n");
    printf("                           Note: option '-K' will be ignored if graph mode selected.\n");
    printf("    --H                    Use half precision (FP16) for back-end\n");
    printf("    --P                    Use separate context for PRACH\n");
    printf("    --Q                    Use separate context for PDCCH\n");
    printf("    --X                    Use separate context for PUCCH\n");
    printf("    --Z                    Use separate context for SRS\n");
    printf("    --S                    Split SRS: run first half of SRS cells at the beginning and the remaining half after PUSCH\n");
    printf("    -B                     Enable pattern mode(s) B for DDDSUUDDDD (u5) (mode(s) A enabled by default)\n");
    printf("    --G                    Group together all PDSCH cells in a context in a slot and execute via single PDSCH pipeline object; same for PDCCH if present\n");
    printf("    --g                    Group together all PUSCH cells in a context in a slot and execute via single PUSCH pipeline object; same for PUCCH and PRACH if present\n");
    printf("    --b                    [Deprecated] Inter-cell batching for PDSCH. No effect as inter-cell batching is always enabled with --G. \n");
    printf("    --M <ctx1,..,ctxtN>    Max number of SMs used per context in comma separated list. Order is as follows: [PRACH if --P], [PDCCH if --Q], [PUCCH if --X], PDSCH, PUSCH\n");
    printf("    --k                    ref_check for PDCCH\n");
    printf("    --c <ch_name_1,...,ch_name_N> enable reference checks for channels whose names are provided as a comma separated list\n");
}

//----------------------------------------------------------------------------------------------------------
// main - Instantiates one or more workers objects. Responsible for accepting commands from user,
// orchestrating the tests
//----------------------------------------------------------------------------------------------------------
int main(int argc, char* argv[])
{
    int returnValue = 0;
    cuphyNvlogFmtHelper nvlog_fmt("cuphy_ex_psch_rx_tx.log");
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        constexpr uint32_t N_MAX_INST    = 8;
        constexpr uint32_t N_WORKER_INST = 2; // e.g. F13 test - 1 PUSCH worker + 1 PDSCH worker
        int                iArg          = 1;
        std::string        inputFileName;
        std::string        inputFileNamePdschTx;
        std::string        outputFileName;

        int32_t              nMaxConcurrentThrds        = std::thread::hardware_concurrency();
        int32_t              baseCpuId                 = 0;
        bool                 enableMultiCore      = false;
        int32_t              gpuId                     = 0;
        uint32_t             nTimingItrs               = 1;
        uint32_t             nPowerItrs                = 0;
        int                  dbgMsgLevel               = 0; // no debug messages
        int                  descramblingOn            = 1;
        bool                 ref_check_pdsch           = false; // enabled with -k or --c ...,PDSCH,...
        bool                 ref_check_pdcch           = false; // enabled with --k or --c ...,PDCCH,...
        bool                 ref_check_pucch           = false; // enabled with --c ...,PUCCH,...
        bool                 ref_check_prach           = false; // enabled with --c ...,PRACH,...
        bool                 ref_check_srs             = false; // enabled with --c ...,SRS,...
        bool                 ref_check_bwc             = false; // enabled with --c ...,BWC,...
        bool                 ref_check_csirs           = false; // enabled with --c ...,CSIRS,...
        bool                 ref_check_ssb             = false; // enabled with --c ...,SSB,...
        int                  cfg_process_mode          = 0;
        bool                 identical_ldpc_configs    = true;
        bool                 enableLdpcThroughputMode  = false;
        cuphyPdschProcMode_t pdsch_proc_mode           = PDSCH_PROC_MODE_NO_GRAPHS;
        uint64_t             pusch_proc_mode           = 0;
        uint32_t             ldpcLaunchMode            = 1;
        uint32_t             delayUs                   = 10000;
        uint32_t             powerDelayUs              = 30; // emperically measured values from nsight-sys indicate a lower bound of around 15-20us
        uint32_t             fp16Mode                  = 1;
        bool                 enableHighPdschCuStrmPrio = false;
        uint32_t             nCtxts                    = 2;
        uint32_t             nStrmsPerCtxt             = 0;
        uint32_t             nPdschStrmsPerCtxt        = 0;
        uint32_t             nPuschStrmsPerCtxt        = 0;
        uint32_t             nSSBStrmsPerCtxt          = 0;
        uint32_t             nPdschCellsPerStrm        = 1; // Number of cells per pipeline object per context. 1 unless group_pdsch_cells is set.
        uint32_t             nPdcchCellsPerStrm        = 1; // Number of cells per pipeline object per context. 1 unless group_pdsch_cells is set. Could be different than nPdschCellsPerStrm
        uint32_t             nPrachCellsPerStrm        = 1; // Number of cells per pipeline object per context. 1 unless group_pusch_cells is set.
        uint32_t             nPschItrsPerStrm          = 0;
        uint32_t             nPdschItrsPerStrm         = 0;
        uint32_t             nPuschItrsPerStrm         = 0;
        int                  pipelineExec              = 0;
        int                  uldl                      = 3;
        bool                 printCbErrors             = false; // PUSCH checks
        bool                 printCellMetrics          = false;
        uint32_t             num_patterns              = 0;
        uint32_t             slots_per_pattern         = 0;
        uint32_t             pusch_slots_per_pattern   = 1;
        uint32_t             nPdschCellsPerPattern     = 0;
        uint32_t             nPuschCellsPerPattern     = 0;
        uint32_t             nPschCellsPerPattern      = 0;
        int                  mpsVals                   = 0;
        bool                 heterogenousMpsPartitions = false;
        std::vector<int32_t> mpsSubctxSmCounts;
        uint32_t             BWCplaceholderDelay       = 0;
        bool                 group_pdsch_cells         = false;
        bool                 group_pusch_cells         = false;
        bool                 group_pucch_cells         = false;
        bool                 splitSRScells50_50        = false;
        bool                 prachCtx                  = false;
        bool                 pdcchCtx                  = false;
        bool                 pucchCtx                  = false;
        bool                 srsCtx                    = false; // currently only applicable for pattern B2
        bool                 pdsch_inter_cell_batching = true; // deprecated
        uint32_t             mode                      = 0; // 0 is mode A (default), 1 is mode B
        uint32_t             nSsbSlots                 = 0; // SSB slots 1 for u3 and 4 for u5

        while(iArg < argc)
        {
            if('-' == argv[iArg][0])
            {
                switch(argv[iArg][1])
                {
                case 'i':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided");
                    }
                    inputFileName.assign(argv[iArg++]);
                    break;
                case 'h':
                    usage();
                    exit(0);
                    break;
                case 'a':
                    enableHighPdschCuStrmPrio = true;
                    ++iArg;
                    break;
                case 'k':
                    ref_check_pdsch = true;
                    ++iArg;
                    break;
                case 'l':
                    identical_ldpc_configs = false; // Launch separate LDPC kernels, one per TB.
                    ++iArg;
                    break;
                case 'L':
                    enableLdpcThroughputMode = true;
                    ++iArg;
                    break;
                case 'm':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &cfg_process_mode)) || ((cfg_process_mode < 0)) || ((cfg_process_mode > 1)))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid process mode");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'K':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%u", &ldpcLaunchMode)) || (3 < ldpcLaunchMode))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid LDPC kernel launch mode ({})", ldpcLaunchMode);
                        exit(1);
                    }
                    ldpcLaunchMode = 1 << ldpcLaunchMode;
                    ++iArg;
                    break;
                case 'b':
                    printCbErrors = true;
                    ++iArg;
                    break;
                case 'r':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nTimingItrs)) || ((nTimingItrs <= 0)))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of run iterations");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'B':
                    mode = 1;
                    ++iArg;
                    break;
                case 'C':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nCtxts)) || (nCtxts <= 0))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of contexts");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'S':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nStrmsPerCtxt)) || ((nStrmsPerCtxt <= 0)))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of streams per context");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'I':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nPschItrsPerStrm)) || ((nPschItrsPerStrm <= 0)))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of iterations per stream");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'P':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nPowerItrs)) || ((nPowerItrs <= 0)))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of iterations");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'W':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%u", &powerDelayUs)) || (powerDelayUs < 0))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid power delay value (should be atleast 30us)");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'g':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &gpuId)) ||
                       ((gpuId < 0)))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid GPU Id {}", gpuId);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'w':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%u", &delayUs))) {
                        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid delay");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'c':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &baseCpuId)) ||
                       ((baseCpuId < 0) || (baseCpuId >= nMaxConcurrentThrds)))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid base CPU Id (should be within [0,{}])", nMaxConcurrentThrds-1);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 't':
                    enableMultiCore = true;
                    ++iArg;
                    break;
                case 'd':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &dbgMsgLevel)) || ((dbgMsgLevel < 0)))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid debug message level {}, disabling debug messages", dbgMsgLevel);
                        dbgMsgLevel = 0;
                    }
                    ++iArg;
                    break;
                case 'o':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: No output file name given");
                    }
                    outputFileName.assign(argv[iArg++]);
                    break;
                case 'u':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &uldl)) ||
                       ((uldl < 3) || (uldl > 5)))
                    {
                        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid uldl mode {}", uldl);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case '-':
                    switch(argv[iArg][2])
                    {
                    case 'H':
                        if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &fp16Mode)) || (1 < fp16Mode))
                        {
                            NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid FP16 mode {}\n", fp16Mode);
                            exit(1);
                        }
                        ++iArg;
                        break;
                    case 'c':
                        if(++iArg >= argc)
                        {
                            NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "List of channel names not provided for reference check option --c");
                        }
                        else
                        {
                            // Input -> std::string
                            std::string chanelNameListStr(argv[iArg++]);
                            // std::string -> std::stringstream
                            std::stringstream channelNameListStrStrm;
                            channelNameListStrStrm.str(chanelNameListStr);

                            std::unordered_map<std::string, int> chNameMap = {{"PUSCH", 0}, {"PDSCH", 1}, {"PDCCH", 2}, {"PUCCH", 3}, {"PRACH", 4}, {"BWC", 5}, {"SSB", 6}, {"CSIRS", 7}, {"SRS", 8}};

                            // Extract from std::stringsream into mpsSubctxSmCounts
                            while(channelNameListStrStrm.good())
                            {
                                std::string subStr;
                                getline(channelNameListStrStrm, subStr, ',');
                                switch(chNameMap[subStr])
                                {
                                case 0:
                                    printCbErrors = true;
                                    NVLOGI_FMT(NVLOG_PUSCH, "Reference checks for PUSCH enabled");
                                    break;
                                case 1:
                                    ref_check_pdsch = true;
                                    NVLOGI_FMT(NVLOG_PDSCH, "Reference checks for PDSCH enabled");
                                    break;
                                case 2:
                                    ref_check_pdcch = true;
                                    NVLOGI_FMT(NVLOG_PDCCH, "Reference checks for PDCCH enabled");
                                    break;
                                case 3:
                                    ref_check_pucch = true;
                                    NVLOGI_FMT(NVLOG_PUCCH, "Reference checks for PUCCH enabled");
                                    break;
                                case 4:
                                    ref_check_prach = true;
                                    NVLOGI_FMT(NVLOG_PRACH, "Reference checks for PRACH enabled");
                                    break;
                                case 5:
                                    ref_check_bwc = true;
                                    NVLOGI_FMT(NVLOG_BFW, "Reference checks for BWC enabled");
                                    break;
                                case 6:
                                    ref_check_ssb = true;
                                    NVLOGI_FMT(NVLOG_SSB, "Reference checks for SSB enabled");
                                    break;
                                case 7:
                                    ref_check_csirs = true;
                                    NVLOGI_FMT(NVLOG_CSIRS, "Reference checks for CSIRS enabled");
                                    break;
                                case 8:
                                    ref_check_srs = true;
                                    NVLOGI_FMT(NVLOG_SRS, "Reference checks for SRS enabled");
                                    break;
                                default:
                                    break;
                                }
                            }
                        }
                        break;

                    case 'M':
                        heterogenousMpsPartitions = true;
                        if(++iArg >= argc || (1 != sscanf(argv[iArg], "%i", &mpsVals)))
                        {
                            NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "Using default SM counts for all sub-contexts");
                        }
                        else
                        {
                            // Input -> std::string
                            std::string mpsSubctxSmCountsStr(argv[iArg++]);
                            // std::string -> std::stringstream
                            std::stringstream mpsSubctxSmCountsStrStrm;
                            mpsSubctxSmCountsStrStrm.str(mpsSubctxSmCountsStr);

                            // Extract from std::stringsream into mpsSubctxSmCounts
                            int i = 0;
                            while(mpsSubctxSmCountsStrStrm.good())
                            {
                                std::string subStr;
                                getline(mpsSubctxSmCountsStrStrm, subStr, ',');
                                mpsSubctxSmCounts.push_back(std::stoul(subStr));
                                //printf("mpsSubctxSmCounts[%d] str %s subStr %s int %d\n", i++, mpsSubctxSmCountsStrStrm.str().c_str(), subStr.c_str(), mpsSubctxSmCounts.back());
                            }
                            // printf("MPS sub-context SM counts:\n");
                            // for(auto &mpsSubctxSmCount : mpsSubctxSmCounts) {printf("%d\n", mpsSubctxSmCount);}
                        }
                        break;
                    case 'P':
                        prachCtx = true;
                        nCtxts++;
                        ++iArg;
                        break;
                    case 'Q':
                        pdcchCtx = true;
                        nCtxts++;
                        ++iArg;
                        break;
                    case 'X':
                        pucchCtx = true;
                        nCtxts++;
                        ++iArg;
                        break;
                    case 'Z':
                        srsCtx = true;
                        nCtxts++;
                        ++iArg;
                        break;
                    case 'S':
                        splitSRScells50_50 = true;
                        ++iArg;
                        break;
                    case 'g':
                        group_pusch_cells = group_pucch_cells = true;
                        ++iArg;
                        break;
                    case 'G':
                        group_pdsch_cells = true;
                        ++iArg;
                        break;
                    case 'b':
                        pdsch_inter_cell_batching = true; // no effect, true by default
                        ++iArg;
                        break;
                    case 'k':
                        ref_check_pdcch = true;
                        ++iArg;
                        break;
                    default:
                        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                        usage();
                        exit(1);
                        break;
                    }
                    break;
                default:
                    NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                    usage();
                    exit(1);
                    break;
                }
            }

            else // if('-' == argv[iArg][0])
            {
                NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
                exit(1);
            }
        } // while (iArg < argc)

        if(inputFileName.empty())
        {
            usage();
            exit(1);
        }

        if(prachCtx && !(uldl == 3 || uldl == 5))
        {
            NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT,  "ERROR: PRACH context mode --P can only be used in conjunction with -u 3 or -u 5");
            exit(1);
        }
        if(pdcchCtx && !(uldl == 3 || uldl == 5))
        {
            NVLOGE_FMT(NVLOG_PDCCH, AERIAL_CUPHY_EVENT,  "ERROR: PDCCH context mode --Q can only be used in conjunction with -u 3 or -u 5");
            exit(1);
        }
        if(pucchCtx && !(uldl == 3 || uldl == 5))
        {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "ERROR: PUCCH context mode --X can only be used in conjunction with -u 3 or -u 5");
            exit(1);
        }
        if(srsCtx && (uldl != 5 || !mode))
        {
            NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: SRS context mode --Z can only be used in conjunction with -u 5 and -B");
            exit(1);
        }

        //---------------------------------------------------------------------------
        // input files

        // yaml parsing
        cuphy::test_config testCfg(inputFileName.c_str());
        //testCfg.print();
        //   testCfg.print();
        uint32_t cells_per_slot  = static_cast<uint32_t>(testCfg.num_cells());
        uint32_t num_slots       = static_cast<uint32_t>(testCfg.num_slots());
        uint32_t num_SRS_cells   = 0;
        uint32_t num_PRACH_cells = 0;
        uint32_t num_BWC_cells   = 0;
        uint32_t num_PDCCH_cells = 0;
        uint32_t num_PDSCH_cells = 0;
        uint32_t num_PUSCH_cells = 0;
        uint32_t num_PUCCH_cells = 0;
        uint32_t num_SSB_cells   = 0;
        uint32_t num_CSIRS_cells = 0;

        // convert slots to slot patterns
        switch(uldl)
        {
        case 3: {
            if(num_slots % 4 != 0)
            {
                NVLOGW_FMT(NVLOG_TAG_BASE_CUPHY, "Warning! For F13 mode (u = 4) the number of slots in YAML must be a multiple of four");
                num_patterns          = num_slots;
                slots_per_pattern     = 1;
                nPdschCellsPerPattern = cells_per_slot;
                nPuschCellsPerPattern = cells_per_slot;
            }
            else
            {
                num_patterns          = num_slots / 4;
                slots_per_pattern     = 4;
                nPdschCellsPerPattern = 4 * cells_per_slot;
                nPuschCellsPerPattern = cells_per_slot;
            }
            nSsbSlots = 1;
            break;
        }

        case 4: {
            num_patterns            = num_slots;
            slots_per_pattern       = 1;
            nPschCellsPerPattern    = cells_per_slot;
            nPuschCellsPerPattern   = cells_per_slot;
            pusch_slots_per_pattern = 1;
            nSsbSlots = 1;
            break;
        }

        case 5: {
            if(num_slots % 8 != 0)
            {
                NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "Error! For DDDSUUDDDD mode (u = 5) the number of slots in YAML must be a multiple of eight");
                exit(1);
            }

            num_patterns            = num_slots / 8;
            slots_per_pattern       = 8;
            nPdschCellsPerPattern   = 8 * cells_per_slot;
            nPuschCellsPerPattern   = 2 * cells_per_slot;
            pusch_slots_per_pattern = 2;
            nSsbSlots = 4;
            break;
        }
        }

        bool graphs_mode = (cfg_process_mode >= 1);
        pdsch_proc_mode  = (graphs_mode) ? PDSCH_PROC_MODE_GRAPHS : PDSCH_PROC_MODE_NO_GRAPHS;
        if(pdsch_inter_cell_batching)
        {
            pdsch_proc_mode = (cuphyPdschProcMode_t)((uint64_t)pdsch_proc_mode | (uint64_t)PDSCH_INTER_CELL_BATCHING);
        }
        pusch_proc_mode = (graphs_mode) ? 1 : 0;

        // read H5 files
        const std::string puschChannelName  = "PUSCH";
        const std::string pusch2ChannelName = "PUSCH2";
        const std::string pdschChannelName  = "PDSCH";
        const std::string bfcChannelName    = "BWC";
        const std::string srsChannelName    = "SRS";
        const std::string prachChannelName  = "PRACH";
        const std::string pdcchChannelName  = "PDCCH";
        const std::string pucchChannelName  = "PUCCH";
        const std::string pucch2ChannelName = "PUCCH2";
        const std::string ssbChannelName    = "SSB";
        const std::string csirsChannelName  = "CSIRS";

        std::vector<std::vector<std::string>> inFileNamesPuschRx(num_patterns); // Dim: num_patterns x num_cells
        std::vector<std::vector<std::string>> inFileNamesPdschTx(num_patterns); // Dim: num_patterns x (num_cells * slots_per_pattern)
        std::vector<std::vector<std::string>> inFileNamesBFC(num_patterns);
        std::vector<std::vector<std::string>> inFileNamesSRS(num_patterns);
        std::vector<std::vector<std::string>> inFileNamesPRACH(num_patterns);
        std::vector<std::vector<std::string>> inFileNamesPdcchTx(num_patterns);
        std::vector<std::vector<std::string>> inFileNamesPucchRx(num_patterns);
        std::vector<std::vector<std::string>> inFileNamesSSB(num_patterns);
        std::vector<std::vector<std::string>> inFileNamesCSIRS(num_patterns);

        std::vector<bool> runSRSVec(num_patterns, false);
        std::vector<bool> runPRACHVec(num_patterns, false);
        std::vector<bool> runPUSCHVec(num_patterns, false);
        std::vector<bool> runPDSCHVec(num_patterns, false);
        std::vector<bool> runPDCCHVec(num_patterns, false);
        std::vector<bool> runPUCCHVec(num_patterns, false);
        std::vector<bool> runBWCVec(num_patterns, false);
        std::vector<bool> runSSBVec(num_patterns, false);
        std::vector<bool> runCSIRSVec(num_patterns, false);

        std::vector<uint32_t> patternMode(num_patterns, 0); // not u5 by default

        int leastPriority, greatestPriority;
        CUDA_CHECK(cudaDeviceGetStreamPriorityRange(&leastPriority, &greatestPriority));

#ifndef ENABLE_F01_STREAM_PRIO
        if(enableHighPdschCuStrmPrio && (4 == uldl))
        {
            NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "ERROR: F01 mode with priorities is not supported");
            exit(1);
        }
#endif

        std::map<std::string, int> cuStrmPrioMap;
        if(enableHighPdschCuStrmPrio)
        {
            // These priorities can be overridden by the ones specified in the yaml file
            cuStrmPrioMap["PUSCH"]  = greatestPriority + 1 + prachCtx;
            cuStrmPrioMap["PUSCH2"] = greatestPriority + 2 + prachCtx;
            cuStrmPrioMap["PUCCH"]  = greatestPriority + 1 + prachCtx;
            cuStrmPrioMap["PUCCH2"] = greatestPriority + 2 + prachCtx;
            cuStrmPrioMap["PDSCH"]  = greatestPriority + prachCtx;
            cuStrmPrioMap["PDCCH"]  = greatestPriority + prachCtx;
            cuStrmPrioMap["CSIRS"]  = greatestPriority + prachCtx;
            cuStrmPrioMap["SRS"]    = greatestPriority + 3 + prachCtx;
            cuStrmPrioMap["PRACH"]  = greatestPriority + 4 * (!prachCtx);
            cuStrmPrioMap["SSB"]    = greatestPriority + 1 + prachCtx;
        }

        // try reading priorities from yaml file; the priorities there are all relative to this
        // GPU's greatest priority (reminder greatest priority is negative, e.g., -5)
        {
            yaml::file_parser fp(inputFileName.c_str());
            yaml::document    d = fp.next_document();
            yaml::node        r = d.root();
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // number of cells (scalar)
            try
            {
                cuStrmPrioMap["PUSCH"] = greatestPriority + r["PUSCH_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["PUSCH2"] = greatestPriority + r["PUSCH2_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["PUCCH"] = greatestPriority + r["PUCCH_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["PUCCH2"] = greatestPriority + r["PUCCH2_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["PDSCH"] = greatestPriority + r["PDSCH_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["PDCCH"] = greatestPriority + r["PDCCH_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["SRS"] = greatestPriority + r["SRS_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["PRACH"] = greatestPriority + r["PRACH_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["SSB"] = greatestPriority + r["SSB_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
            try
            {
                cuStrmPrioMap["CSIRS"] = greatestPriority + r["CSIRS_PRIO"].as<unsigned int>();
            }
            catch(...)
            {}
        }

        // parse maximum parameters for memory allocation
        maxPDSCHPrms pdschPrms;
        maxPUSCHPrms puschPrms;

        pdschPrms.maxNTbs      = 0;
        pdschPrms.maxNCbs      = 0;
        pdschPrms.maxNCbsPerTb = 0;
        pdschPrms.maxNPrbs     = 0;
        pdschPrms.maxNTx       = 0;

        puschPrms.maxNTbs          = 0;
        puschPrms.maxNCbs          = 0;
        puschPrms.maxNCbsPerTb     = 0;
        puschPrms.maxNPrbs         = 0;
        puschPrms.maxNRx           = 0;
        puschPrms.maxNCellsPerSlot = 1; // no cell-groups by default

        yaml::file_parser fp(inputFileName.c_str());
        yaml::document    d           = fp.next_document();
        yaml::node        r           = d.root();
        bool              prmsPresent = false;
        try
        {
            r           = r["parameters"];
            prmsPresent = true;
        }
        catch(...)
        {
            NVLOGW_FMT(NVLOG_TAG_BASE_CUPHY, "WARNING: Parameters field not found in YAML");
        }

        if(prmsPresent)
        {
            try
            {
                yaml::node p = r["PDSCH"];
                try
                {
                    pdschPrms.maxNTbs = group_pdsch_cells ? p["Max #TB per slot"] : p["Max #TB per slot per cell"];
                }
                catch(...)
                {
                    pdschPrms.maxNTbs = 0;
                }
                try
                {
                    pdschPrms.maxNCbs = group_pdsch_cells ? p["Max #CB per slot"] : p["Max #CB per slot per cell"];
                }
                catch(...)
                {
                    pdschPrms.maxNCbs = 0;
                }
                try
                {
                    pdschPrms.maxNCbsPerTb = p["Max #CB per slot per cell per TB"];
                }
                catch(...)
                {
                    pdschPrms.maxNCbsPerTb = 0;
                }
                try
                {
                    pdschPrms.maxNPrbs = p["Max #PRB per cell"];
                }
                catch(...)
                {
                    pdschPrms.maxNPrbs = 0;
                }
                try
                {
                    pdschPrms.maxNTx = p["Max #TX per cell"];
                }
                catch(...)
                {
                    pdschPrms.maxNTx = 0;
                }
            }
            catch(...)
            {
                ("WARNING: Parameters / PDSCH field not found in YAML\n");
            }

            try
            {
                yaml::node p = r["PUSCH"];
                try
                {
                    puschPrms.maxNTbs = group_pusch_cells ? p["Max #TB per slot"] : p["Max #TB per slot per cell"];
                }
                catch(...)
                {
                    puschPrms.maxNTbs = 0;
                }
                try
                {
                    puschPrms.maxNCbs = group_pusch_cells ? p["Max #CB per slot"] : p["Max #CB per slot per cell"];
                }
                catch(...)
                {
                    puschPrms.maxNCbs = 0;
                }
                try
                {
                    puschPrms.maxNCbsPerTb = p["Max #CB per slot per cell per TB"];
                }
                catch(...)
                {
                    puschPrms.maxNCbsPerTb = 0;
                }
                try
                {
                    puschPrms.maxNPrbs = p["Max #PRB per cell"];
                }
                catch(...)
                {
                    puschPrms.maxNPrbs = 0;
                }
                try
                {
                    puschPrms.maxNRx = p["Max #RX per cell"];
                }
                catch(...)
                {
                    puschPrms.maxNRx = 0;
                }
            }
            catch(...)
            {
                ("WARNING: Parameters / PUSCH field not found in YAML\n");
            }
        }
        bool addedSSBCtx = false;
        for(uint32_t patternIdx = 0; patternIdx < num_patterns; patternIdx++)
        {
            uint32_t slotIdx = patternIdx * slots_per_pattern;
            num_SRS_cells    = 0;
            num_PRACH_cells  = 0;
            num_BWC_cells    = 0;
            num_PDSCH_cells  = 0;
            num_PUSCH_cells  = 0;
            num_PUCCH_cells  = 0;
            num_SSB_cells    = 0;
            num_CSIRS_cells  = 0;

            try
            {
                uint32_t ncells = testCfg.slots()[slotIdx].at(srsChannelName).size();
                num_SRS_cells   = ncells;
                if(num_SRS_cells == 0)
                {
                    throw std::out_of_range("SRS #cells == 0");
                }
                runSRSVec[patternIdx] = true;
            }
            catch(const std::out_of_range& ex)
            {
                runSRSVec[patternIdx] = false;
                NVLOGI_FMT(NVLOG_SRS, "NO SRS detected for pattern {}", patternIdx);
                if(srsCtx && num_SRS_cells == 0)
                {
                    NVLOGE_FMT(NVLOG_SRS, AERIAL_CUPHY_EVENT,  "ERROR: SRS context enabled with --Z but no SRS test vectors in yaml for pattern {}", patternIdx);
                    exit(-1);
                }
            }

            try
            {
                uint32_t ncells = testCfg.slots()[slotIdx].at(ssbChannelName).size();
                num_SSB_cells   = ncells;
                if(num_SSB_cells == 0)
                {
                    throw std::out_of_range("SSB #cells == 0");
                }
                runSSBVec[patternIdx] = true;
                if(!addedSSBCtx)
                {
                    nCtxts++; // SSB always uns in dedicated context
                    addedSSBCtx = true;
                }
            }
            catch(const std::out_of_range& ex)
            {
                runSSBVec[patternIdx] = false;
                printf("NO SSB detected for pattern %d\n", patternIdx);
            }

            try
            {
                uint32_t ncells = testCfg.slots()[slotIdx].at(prachChannelName).size();
                num_PRACH_cells = ncells;
                if(num_PRACH_cells == 0)
                {
                    throw std::out_of_range("PRACH #cells == 0");
                }
                runPRACHVec[patternIdx] = true;
            }
            catch(const std::out_of_range& ex)
            {
                runPRACHVec[patternIdx] = false;
                printf("NO PRACH detected for pattern %d\n", patternIdx);
            }

            if(prachCtx && num_PRACH_cells == 0)
            {
                printf("ERROR: PRACH context enabled with --P but no PRACH test vectors in yaml for pattern %d\n", patternIdx);
                exit(-1);
            }

            try
            {
                uint32_t ncells = testCfg.slots()[slotIdx].at(puschChannelName).size();
                num_PUSCH_cells = ncells;
                if(num_PUSCH_cells == 0)
                {
                    throw std::out_of_range("PUSCH #cells == 0");
                }
                runPUSCHVec[patternIdx] = true;
            }
            catch(const std::out_of_range& ex)
            {
                runPUSCHVec[patternIdx] = false;
                printf("NO PUSCH detected for pattern %d\n", patternIdx);
            }

            try
            {
                uint32_t ncells = testCfg.slots()[slotIdx].at(pucchChannelName).size();
                num_PUCCH_cells = ncells;
                if(num_PUCCH_cells == 0)
                {
                    throw std::out_of_range("PUCCH #cells == 0");
                }
                runPUCCHVec[patternIdx] = true;
            }
            catch(const std::out_of_range& ex)
            {
                runPUCCHVec[patternIdx] = false;
                printf("NO PUCCH detected for pattern %d\n", patternIdx);

                if(pucchCtx && num_PUCCH_cells == 0)
                {
                    printf("ERROR: PUCCH context enabled with --X but no PUCCH test vectors in yaml for pattern %d\n", patternIdx);
                    exit(-1);
                }
            }

            try
            {
                uint32_t ncells = testCfg.slots()[slotIdx].at(pdschChannelName).size();
                num_PDSCH_cells = ncells;
                if(num_PDSCH_cells == 0)
                {
                    throw std::out_of_range("PDSCH #cells == 0");
                }
                runPDSCHVec[patternIdx] = true;
            }
            catch(const std::out_of_range& ex)
            {
                runPDSCHVec[patternIdx] = false;
                printf("NO PDSCH detected for pattern %d\n", patternIdx);
            }

            try
            {
                uint32_t ncells = testCfg.slots()[slotIdx].at(bfcChannelName).size();
                num_BWC_cells   = ncells;
                if(num_BWC_cells == 0)
                {
                    throw std::out_of_range("BWC #cells == 0");
                }
                runBWCVec[patternIdx] = true;
                if(uldl == 5) // long pattern mode B
                {
                    if(mode)
                    {
                        patternMode[patternIdx] = 6;
                        printf("Pattern %d: DDDSUUDDDD mode B2\n", patternIdx);
                    }
                    else
                    {
                        patternMode[patternIdx] = 3;
                        printf("Pattern %d: DDDSUUDDDD mode A2\n", patternIdx);
                    }
                }
            }
            catch(const std::out_of_range& ex)
            {
                printf("NO BWC detected for pattern %d\n", patternIdx);
            }

            try
            {
                uint32_t ncells = testCfg.slots()[slotIdx].at(pdcchChannelName).size();
                num_PDCCH_cells = ncells;
                if(num_PDCCH_cells == 0)
                {
                    throw std::out_of_range("PDCCH #cells == 0");
                }
                runPDCCHVec[patternIdx] = true;
            }
            catch(const std::out_of_range& ex)
            {
                runPDCCHVec[patternIdx] = false;
                printf("NO PDCCH detected for pattern %d\n", patternIdx);
            }
            try
            {
                uint32_t ncells = testCfg.slots()[slotIdx].at(csirsChannelName).size();
                num_CSIRS_cells = ncells;
                if(num_CSIRS_cells == 0)
                {
                    throw std::out_of_range("CSIRS #cells == 0");
                }
                runCSIRSVec[patternIdx] = true;
            }
            catch(const std::out_of_range& ex)
            {
                runCSIRSVec[patternIdx] = false;
                printf("NO CSIRS detected for pattern %d\n", patternIdx);
            }

            if(pdcchCtx && num_PDCCH_cells == 0)
            {
                printf("ERROR: PDCCH context enabled with --Q but no PDCCH test vectors in yaml for pattern %d\n", patternIdx);
                exit(-1);
            }

            for(uint32_t cellIdx = 0; cellIdx < num_SRS_cells; ++cellIdx)
            {
                std::string srs_tv_filename = testCfg.slots()[slotIdx].at(srsChannelName)[cellIdx];
                inFileNamesSRS[patternIdx].emplace_back(srs_tv_filename);
            }

            for(uint32_t cellIdx = 0; cellIdx < num_SSB_cells; ++cellIdx)
            {
                std::string ssb_tv_filename = testCfg.slots()[slotIdx].at(ssbChannelName)[cellIdx];
                inFileNamesSSB[patternIdx].emplace_back(ssb_tv_filename);
            }

            for(uint32_t cellIdx = 0; cellIdx < num_PRACH_cells; ++cellIdx)
            {
                std::string prach_tv_filename = testCfg.slots()[slotIdx].at(prachChannelName)[cellIdx];
                inFileNamesPRACH[patternIdx].emplace_back(prach_tv_filename);
            }

            // pusch
            for(uint32_t i = 0; i < pusch_slots_per_pattern; ++i)
            {
                for(uint32_t cellIdx = 0; cellIdx < num_PUSCH_cells; ++cellIdx)
                {
                    std::string pusch_tv_filename = testCfg.slots()[slotIdx + i].at(puschChannelName)[cellIdx];
                    inFileNamesPuschRx[patternIdx].emplace_back(pusch_tv_filename);
                }
                // pucch
                for(uint32_t cellIdx = 0; cellIdx < num_PUCCH_cells; ++cellIdx)
                {
                    std::string pucch_tv_filename = testCfg.slots()[slotIdx + i].at(pucchChannelName)[cellIdx];
                    inFileNamesPucchRx[patternIdx].emplace_back(pucch_tv_filename);
                }
            }

            // pdsch + BWC
            for(uint32_t i = 0; i < slots_per_pattern; ++i)
            {
                for(int cellIdx = 0; cellIdx < num_PDSCH_cells; ++cellIdx)
                {
                    std::string pdsch_tv_filename = testCfg.slots()[slotIdx + i].at(pdschChannelName)[cellIdx];
                    inFileNamesPdschTx[patternIdx].emplace_back(pdsch_tv_filename);
                }

                for(int cellIdx = 0; cellIdx < num_PDCCH_cells; ++cellIdx)
                {
                    std::string pdcch_tv_filename = testCfg.slots()[slotIdx + i].at(pdcchChannelName)[cellIdx];
                    inFileNamesPdcchTx[patternIdx].emplace_back(pdcch_tv_filename);
                    //printf("cell %d, pattern %d filename %s\n", cellIdx, patternIdx, pdcch_tv_filename.c_str());
                }

                for(int cellIdx = 0; cellIdx < num_CSIRS_cells; ++cellIdx)
                {
                    std::string csirs_tv_filename = testCfg.slots()[slotIdx + i].at(csirsChannelName)[cellIdx];
                    inFileNamesCSIRS[patternIdx].emplace_back(csirs_tv_filename);
                    //printf("cell %d, pattern %d filename %s\n", cellIdx, patternIdx, pdcch_tv_filename.c_str());
                }

                if(runBWCVec[patternIdx])
                {
                    for(int cellIdx = 0; cellIdx < num_BWC_cells; ++cellIdx)
                    {
                        std::string bfc_tv_filename = testCfg.slots()[slotIdx + i].at(bfcChannelName)[cellIdx];
                        inFileNamesBFC[patternIdx].emplace_back(bfc_tv_filename);
                    }
                }
            }
            // disable channels not present in selected mode for u == 5

            //FixMe?? the following if statement is always false
            if(uldl == 5 && patternMode[patternIdx] == 0)
            {
                if(mode == 0) // mode A
                {
                    if(runPDCCHVec[patternIdx] || runPRACHVec[patternIdx])
                    {
                        patternMode[patternIdx] = 2;
                        printf("Pattern %d: DDDSUUDDDD mode A1\n", patternIdx);
                    }
                    else
                    {
                        printf("Pattern %d: DDDSUUDDDD mode A0\n", patternIdx);
                        patternMode[patternIdx] = 1;
                    }
                }
                else
                {
                    if(runPDCCHVec[patternIdx] || runPRACHVec[patternIdx])
                    {
                        patternMode[patternIdx] = 5;
                        printf("Pattern %d: DDDSUUDDDD mode B1\n", patternIdx);
                    }
                    else
                    {
                        patternMode[patternIdx] = 4;
                        printf("Pattern %d: DDDSUUDDDD mode B0\n", patternIdx);
                    }
                }
            }
            if(uldl == 3)
            {
                if(runPDCCHVec[patternIdx])
                {
                    if(runBWCVec[patternIdx])
                        printf("Pattern %d: DDDSU mode A5\n", patternIdx);
                    else
                        printf("Pattern %d: DDDSU mode A4\n", patternIdx);
                }
                else
                {
                    if(runBWCVec[patternIdx])
                    {
                        if(runPRACHVec[patternIdx])
                            printf("Pattern %d: DDDSU mode A3\n", patternIdx);
                        else
                            printf("Pattern %d: DDDSU mode A2\n", patternIdx);
                    }
                    else
                    {
                        if(runPRACHVec[patternIdx])
                            printf("Pattern %d: DDDSU mode A1\n", patternIdx);
                        else
                            printf("Pattern %d: DDDSU mode A0\n", patternIdx);
                    }
                }
            }
        }

        //---------------------------------------------------------------------------
        // worker run configurations

        if(uldl == 4)
        {
            // use a default cfg if not specified at input
            if(nStrmsPerCtxt == 0)
            {
                nStrmsPerCtxt    = cells_per_slot / nCtxts;
                nPschItrsPerStrm = 1;
            }

            nPdschCellsPerStrm         = group_pdsch_cells ? nStrmsPerCtxt : 1;
            nPdschStrmsPerCtxt         = group_pdsch_cells ? 1 : nStrmsPerCtxt;
            nPuschStrmsPerCtxt         = group_pusch_cells ? 1 : nStrmsPerCtxt;
            puschPrms.maxNCellsPerSlot = group_pusch_cells ? nStrmsPerCtxt : 1;
            // check that run configuration is legal
            if((nCtxts * nStrmsPerCtxt * nPschItrsPerStrm) != nPschCellsPerPattern)
            {
                NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "Error! nCtxts*nStrmsPerCtxt*nPschItrsPerStrm must be equal to number of cells");
                exit(1);
            }
        }

        if(uldl != 4)
        {
            nPdschCellsPerStrm         = group_pdsch_cells ? cells_per_slot : 1;
            nPdcchCellsPerStrm         = group_pdsch_cells ? num_PDCCH_cells : 1; // group_pdsch_cells used intentionally
            nPrachCellsPerStrm         = group_pusch_cells ? num_PRACH_cells : 1;
            nStrmsPerCtxt              = cells_per_slot;
            nPdschItrsPerStrm          = (uldl == 3) ? 4 : /* U5 */ 8;
            nPuschItrsPerStrm          = 1;
            nPdschStrmsPerCtxt         = group_pdsch_cells ? 1 : cells_per_slot;
            nSSBStrmsPerCtxt           = group_pdsch_cells ? 1 : num_SSB_cells;
            nPuschStrmsPerCtxt         = group_pusch_cells ? uldl == 5 ? 2 : 1 : cells_per_slot;
            puschPrms.maxNCellsPerSlot = group_pusch_cells ? cells_per_slot : 1;
            if(uldl == 5)
                num_PUCCH_cells *= 2;
        }
        if(group_pdsch_cells && (nPdschCellsPerStrm > PDSCH_MAX_CELLS_PER_CELL_GROUP))
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Error! nPdschCellsPerStrm {} > PDSCH_MAX_CELLS_PER_CELL_GROUP {}. Update the max. limit in the header file", nPdschCellsPerStrm, PDSCH_MAX_CELLS_PER_CELL_GROUP);
            exit(1);
        }

        //---------------------------------------------------------------------------
        // Partition GPU and CPU

        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, gpuId));
        int32_t gpuMaxSmCount = 0;
        CU_CHECK(cuDeviceGetAttribute(&gpuMaxSmCount, CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT, device));

        if(heterogenousMpsPartitions)
        {
            if(nCtxts != mpsSubctxSmCounts.size())
            {
                NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "Number of input MPS sub-context SM counts ({}) does not match # of worker instances {}", mpsSubctxSmCounts.size(), nCtxts);
                exit(1);
            }

            bool     validSmCount = true;
            uint32_t subCtxIdx    = 0;
            std::for_each(mpsSubctxSmCounts.begin(), mpsSubctxSmCounts.end(), [&gpuMaxSmCount, &subCtxIdx, &validSmCount](auto& mpsSubCtxSmCount) {
                if(mpsSubCtxSmCount > gpuMaxSmCount)
                {
                    validSmCount = false;
                    NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "SM count ({}) for sub-context {} exceeds device limit ({})\n", mpsSubCtxSmCount, subCtxIdx, gpuMaxSmCount);
                }
                subCtxIdx++;
            });
            if(!validSmCount)
            {
                exit(1);
            }
        }
        else if(!heterogenousMpsPartitions)
        {
            mpsSubctxSmCounts.clear();
            for(uint32_t ctxIdx = 0; ctxIdx < nCtxts; ++ctxIdx)
            {
                mpsSubctxSmCounts.emplace_back(gpuMaxSmCount);
            }
        }

        // GPU and CPU Ids
        std::vector<int> gpuIds(nCtxts, gpuId);
        std::vector<int> cpuIds(nCtxts, baseCpuId);
        if(enableMultiCore) {
            for(int ii=0; ii<nCtxts; ii++) {
                int current_core = baseCpuId + ii;
                int max_core = nMaxConcurrentThrds-1;
                cpuIds[ii] = std::min(current_core, max_core);
                if(current_core > max_core) {
                    NVLOGW_FMT(NVLOG_TAG_BASE_CUPHY, "WARNING!  Not enough cores available to fit thread {}.  Aliasing cpu {}",current_core,max_core);
                }
            }
        }

        int              cpuThrdSchdPolicy = SCHED_RR; // SCHED_FIFO // SCHED_RR
        int              cpuThrdPrio       = sched_get_priority_max(cpuThrdSchdPolicy);
        std::vector<int> cpuThrdPrios(nCtxts, cpuThrdPrio);

        // Main cuda stream and events for synchronizing GPU work
        cuphy::stream mainStream(cudaStreamNonBlocking);

        // For power test bench, the syncUpdateIntervalCnt is the number of iterations between
        // CPU-GPU to syncup (i.e. max number of iterations where the workload is pushed to the GPU before its ungated)
        uint32_t syncUpdateIntervalCnt = graphs_mode ? 50 : 5;
        if(0 != nPowerItrs)
        {
            nPowerItrs  = (nPowerItrs >= 100) ? nPowerItrs : 100;
            nTimingItrs = 1; // Force timing iterations to 1 for power measurements
        }
        // Create and initialize (clear) CPU to GPU start sync flag
        auto shPtrGpuStartSyncFlag  = std::make_shared<cuphy::buffer<uint32_t, cuphy::pinned_alloc>>(1);
        (*shPtrGpuStartSyncFlag)[0] = 0;

        // Command/Response queues for communication between main (orchestration) thread and worker thread(s)
        std::vector<cuphySharedWrkrCmdQ> cmdQVec;
        std::vector<cuphySharedWrkrRspQ> rspQVec;

        for(uint32_t ctxIdx = 0; ctxIdx < nCtxts; ++ctxIdx)
        {
            std::string commandStr;
            commandStr = "CommandQueue" + std::to_string(ctxIdx);
            cmdQVec.emplace_back(std::make_shared<cuphyTestWrkrCmdQ>(commandStr.c_str()));

            std::string responseStr;
            responseStr = "ResponseQueue" + std::to_string(ctxIdx);
            rspQVec.emplace_back(std::make_shared<cuphyTestWrkrRspQ>(responseStr.c_str()));
        }

        //---------------------------------------------------------------------------
        // Timing objects
        cuphy::event_timer slotPatternTimer;
        float              tot_slotPattern_time    = 0;
        float              avg_slotPattern_time_us = 0;

        //--------------------------------------------------------------------------
        // Print simulation setup

        // printf("\n ----------------------------------------------------------");
        // printf("\n Notes on  simulation setup:");
        // printf("\n--> %d cells", cells_per_slot);
        // if(uldl == 3){
        //     printf("\n--> TDD slot pattern. Consists of %d PDSCH and %d PUSCH slots", nPdschCellsPerPattern, nPuschCellsPerPattern);
        // }else{
        //     printf("\n--> FDD slot pattern. Consists of 1 PUSCH + PDSCH slot");
        // }


        //----------\--------------------------------------------------
        //PDSCH + PUSCH
        if((uldl == 3) || (uldl == 5))
        {
            //------------------------------------------------------------------------------------
            // create workers
            // uint32_t ctxtIdx = 0;
            //testWorkerVec.reserve(nCtxts); // needed to avoid call to copy constructor
            std::vector<std::string> testWorkerStringVec;
            //= {"PdschTxTestWorker", "PuschRxTestWorker", "PrachTestWorker", "PdcchTestWorker", "PucchTestWorker"};

            std::unordered_map<std::string, int> channelWorkerMap;

            int c = 0;
            if(prachCtx)
            {
                channelWorkerMap.insert({"PRACH", c});
                testWorkerStringVec.push_back("PrachTestWorker");
                c++;
            }

            if(pdcchCtx)
            {
                channelWorkerMap.insert({"PDCCH", c});
                testWorkerStringVec.push_back("PdcchTestWorker");
                c++;
            }

            if(pucchCtx)
            {
                channelWorkerMap.insert({"PUCCH", c});
                testWorkerStringVec.push_back("PucchTestWorker");
                c++;
            }

            channelWorkerMap.insert({"PDSCH", c});
            testWorkerStringVec.push_back("PdschTxTestWorker");
            if(!pdcchCtx)
                channelWorkerMap.insert({"PDCCH", c});
            c++;

            channelWorkerMap.insert({"PUSCH", c});
            testWorkerStringVec.push_back("PuschRxTestWorker");
            if(!prachCtx)
                channelWorkerMap.insert({"PRACH", c});
            if(!pucchCtx)
                channelWorkerMap.insert({"PUCCH", c});
            if(!srsCtx)
                channelWorkerMap.insert({"SRS", c});

            if(num_SSB_cells > 0)
            {
                c++;
                channelWorkerMap.insert({"SSB", c});
                testWorkerStringVec.push_back("SsbTestWorker");
            }

            if (srsCtx)
            {
                c++;
                channelWorkerMap.insert({"SRS", c});
                testWorkerStringVec.push_back("SrsTestWorker");
            }

            //--------------------------------------------------------------------------
            // Print run setup
            std::string ctxt_run_string            = "parallel";
            std::string graphs_streams_mode_string = (graphs_mode) ? "Graphs" : "Streams";

            printf("\n----------------------------------------------------------");
            printf("\nNotes on  run setup:");
            printf("\n--> %d CUDA contexts (workers) are run in %s and in %s mode\n\n", nCtxts, ctxt_run_string.c_str(), graphs_streams_mode_string.c_str());
            for(int ctxtIdx = 0; ctxtIdx < nCtxts; ctxtIdx++)
            {
                printf("requested SMs for context [%d] %-20s : %d\n", ctxtIdx, testWorkerStringVec[ctxtIdx].c_str(), mpsSubctxSmCounts[ctxtIdx]);
            }

            CUcontext primaryCtx;
            CU_CHECK(cuCtxGetCurrent(&primaryCtx));
            printf("\nPrimary Context Id 0x%0lx\n", reinterpret_cast<uint64_t>(primaryCtx));
            printf("----------------------------------------------------------\n");


            std::vector<cuPHYTestWorker> testWorkerVec;
            testWorkerVec.reserve(nCtxts);

            for(int ctxtIdx = 0; ctxtIdx < nCtxts; ctxtIdx++)
            {
                testWorkerVec.emplace_back(testWorkerStringVec[ctxtIdx].c_str(), ctxtIdx, cpuIds[ctxtIdx], gpuIds[ctxtIdx], cpuThrdSchdPolicy, cpuThrdPrios[ctxtIdx], mpsSubctxSmCounts[ctxtIdx], cmdQVec[ctxtIdx], rspQVec[ctxtIdx], uldl, dbgMsgLevel);
            }

            cuphy::event                               startEvent(cudaEventDisableTiming);
            std::vector<std::shared_ptr<cuphy::event>> shPtrStopEvents;
            shPtrStopEvents.resize(nCtxts);

            //------------------------------------------------------------------------------------
            // initialize workers
            if(prachCtx)
            {
                testWorkerVec[channelWorkerMap["PRACH"]].init(nStrmsPerCtxt, nPuschItrsPerStrm, nTimingItrs, cuStrmPrioMap, shPtrGpuStartSyncFlag, 0, 0, 0, 0, num_PRACH_cells, 0, 0, false, true, patternMode[0]);
            }
            if(pdcchCtx) // It's possible to have PDCCH in the yaml but not a separate PDCCH context. In this case PDCCH will share PDSCH's context, see PDSCH init below
            {
                // Note num_PDCCH_cells need not be identical to num_PDSCH_cells
                testWorkerVec[channelWorkerMap["PDCCH"]].init(nStrmsPerCtxt, nPdschItrsPerStrm, nTimingItrs, cuStrmPrioMap, shPtrGpuStartSyncFlag, num_PDCCH_cells, num_CSIRS_cells, 0, 0, 0, 0, 0, false, true, patternMode[0]);
            }
            if(pucchCtx)
            {
                testWorkerVec[channelWorkerMap["PUCCH"]].init(nStrmsPerCtxt, nPuschItrsPerStrm, nTimingItrs, cuStrmPrioMap, shPtrGpuStartSyncFlag, 0, 0, 0, 0, 0, num_PUCCH_cells, 0, false, true, patternMode[0]);
            }
            if(srsCtx)
            {
                testWorkerVec[channelWorkerMap["SRS"]].init(nStrmsPerCtxt, nPuschItrsPerStrm, nTimingItrs, cuStrmPrioMap, shPtrGpuStartSyncFlag, 0, 0, 0, num_SRS_cells, 0, 0, 0, splitSRScells50_50, true, patternMode[0], true);
            }

            if(num_SSB_cells)
            {
                testWorkerVec[channelWorkerMap["SSB"]].init(nSSBStrmsPerCtxt, nSsbSlots, nTimingItrs, cuStrmPrioMap, shPtrGpuStartSyncFlag, 0, 0, 0, 0, 0, 0, num_SSB_cells, false, true, patternMode[0]);
            }

            testWorkerVec[channelWorkerMap["PDSCH"]].init(nPdschStrmsPerCtxt, nPdschItrsPerStrm, nTimingItrs, cuStrmPrioMap, shPtrGpuStartSyncFlag, pdcchCtx ? 0 : num_PDCCH_cells, pdcchCtx ? 0 : num_CSIRS_cells, num_BWC_cells, 0, 0, 0, 0, false, true, patternMode[0]);

            testWorkerVec[channelWorkerMap["PUSCH"]].init(nPuschStrmsPerCtxt, nPuschItrsPerStrm, nTimingItrs, cuStrmPrioMap, shPtrGpuStartSyncFlag, 0, 0, 0, srsCtx ? 0 : num_SRS_cells, prachCtx ? 0 : num_PRACH_cells, pucchCtx ? 0 : num_PUCCH_cells, 0, splitSRScells50_50, true, patternMode[0], srsCtx);

            testWorkerVec[channelWorkerMap["PDSCH"]].pdschTxInit(inFileNamesPdschTx[0], ref_check_pdsch, identical_ldpc_configs, pdsch_proc_mode, group_pdsch_cells, nPdschCellsPerStrm, pdschPrms);

            testWorkerVec[channelWorkerMap["PDSCH"]].bfcInit(inFileNamesBFC[0], ref_check_bwc);

            testWorkerVec[channelWorkerMap["PUSCH"]].puschRxInit(inFileNamesPuschRx[0], fp16Mode, descramblingOn, printCbErrors, pusch_proc_mode, enableLdpcThroughputMode, group_pusch_cells, puschPrms, ldpcLaunchMode);

            testWorkerVec[channelWorkerMap["SRS"]].srsInit(inFileNamesSRS[0], ref_check_srs, pusch_proc_mode, splitSRScells50_50);

            testWorkerVec[channelWorkerMap["PRACH"]].prachInit(inFileNamesPRACH[0], pusch_proc_mode, ref_check_prach, group_pusch_cells, nPrachCellsPerStrm);

            testWorkerVec[channelWorkerMap["PUCCH"]].pucchRxInit(inFileNamesPucchRx[0], ref_check_pucch, group_pucch_cells, pusch_proc_mode);

            testWorkerVec[channelWorkerMap["PDCCH"]].pdcchTxInit(inFileNamesPdcchTx[0], group_pdsch_cells, nPdcchCellsPerStrm /* can be different than nPdschCellsPerStrm */, ref_check_pdcch, (uint64_t)(pdsch_proc_mode & 0x1));
            testWorkerVec[channelWorkerMap["PDCCH"]].csirsInit(inFileNamesCSIRS[0], ref_check_csirs, group_pdsch_cells, (uint64_t)(pdsch_proc_mode & 0x1));

            if(num_SSB_cells)
            {
                testWorkerVec[channelWorkerMap["SSB"]].ssbInit(inFileNamesSSB[0], ref_check_ssb, group_pdsch_cells, nSsbSlots, (uint64_t)(pdsch_proc_mode & 0x1));
            }

            std::vector<cuPHYTestWorker*> pTestWorkers; //{&testWorkerVec[0], &testWorkerVec[1], &testWorkerVec[2]};
            for(int i = 0; i < nCtxts; i++)
                pTestWorkers.push_back(&testWorkerVec[i]);

            readSmIds(pTestWorkers, mainStream);

            //------------------------------------------------------------------------------------
            // Loop over slot-patterns
            //
            for(int patternIdx = 0; patternIdx < num_patterns; patternIdx++)
            {
                // timing iterations
                // setup pipelines
	    	    testWorkerVec[channelWorkerMap["PDSCH"]].pdschTxSetup(inFileNamesPdschTx[patternIdx]);

                testWorkerVec[channelWorkerMap["PUCCH"]].pucchRxSetup(inFileNamesPucchRx[patternIdx]);

                testWorkerVec[channelWorkerMap["PRACH"]].prachSetup(inFileNamesPRACH[patternIdx]);

                testWorkerVec[channelWorkerMap["PDSCH"]].bfcSetup(inFileNamesBFC[patternIdx]);

                testWorkerVec[channelWorkerMap["PDCCH"]].pdcchTxSetup(inFileNamesPdcchTx[patternIdx]);

		        testWorkerVec[channelWorkerMap["PDCCH"]].csirsSetup(inFileNamesCSIRS[patternIdx]);

                testWorkerVec[channelWorkerMap["PUSCH"]].puschRxSetup(inFileNamesPuschRx[patternIdx]);
                
		        testWorkerVec[channelWorkerMap["SRS"]].srsSetup(inFileNamesSRS[patternIdx]);
                
		        if(num_SSB_cells)
                {
                    testWorkerVec[channelWorkerMap["SSB"]].ssbSetup(inFileNamesSSB[patternIdx]);
                }
#if USE_NVTX
                nvtxRangePush("PATTERN");
#endif
                for(uint32_t itrIdx = 0; itrIdx < nTimingItrs; ++itrIdx)
                {
                    if(0 != nPowerItrs)
                    {
                        // Initialize CPU-GPU start sync flag
                        uint32_t syncFlagVal        = 1;
                        (*shPtrGpuStartSyncFlag)[0] = 0;

                        for(int i = 0; i < nCtxts; i++)
                            testWorkerVec[i].setWaitVal(syncFlagVal);

                        for(uint32_t powerItrIdx = 1; powerItrIdx <= nPowerItrs; ++powerItrIdx)
                        {
                            gpu_us_delay(powerDelayUs, gpuId, mainStream.handle(), 0);

                            // Drop event on main stream for kernels to queue behind
                            startEvent.record(mainStream.handle());
                            // run pipeline
                            testWorkerVec[channelWorkerMap["PDSCH"]].pdschTxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PDSCH"]], true /*waitRsp*/, pdcchCtx ? testWorkerVec[channelWorkerMap["PDCCH"]].getPdcchStopEventVecPtr() : nullptr /* PDCCH stop event vec*/ /*, PDSCH inter slot event vec = nullptr*/);
                            if(pdcchCtx)
                            {
                                testWorkerVec[channelWorkerMap["PDCCH"]].pdschTxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PDCCH"]], true, nullptr, testWorkerVec[channelWorkerMap["PDSCH"]].getPdschInterSlotEventVecPtr());
                            }
                            testWorkerVec[channelWorkerMap["PUSCH"]].puschRxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PUSCH"]]);
                            if(prachCtx)
                            {
                                testWorkerVec[channelWorkerMap["PRACH"]].puschRxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PRACH"]], testWorkerVec[channelWorkerMap["PUSCH"]].getPusch2StartEvent());
                            }
                            if(pucchCtx)
                            {
                                testWorkerVec[channelWorkerMap["PUCCH"]].puschRxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PUCCH"]], testWorkerVec[channelWorkerMap["PUSCH"]].getPusch2StartEvent(), testWorkerVec[channelWorkerMap["PUSCH"]].getPuschStartEvent());
                            }
                            if(srsCtx)
                            {
                                testWorkerVec[channelWorkerMap["SRS"]].puschRxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["SRS"]]);
                            }
                            // SSB starts at beginning of 5th slot in uldl == case
                            if(num_SSB_cells)
                            {
                                testWorkerVec[channelWorkerMap["SSB"]].pdschTxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["SSB"]], true /*waitRsp*/, nullptr, uldl == 3 ? nullptr : testWorkerVec[channelWorkerMap["PDSCH"]].getPdschInterSlotEventVecPtr() /*uldl == 5:  start at 5th slot*/);
                            }
                            for(int i = 0; i < nCtxts; i++)
                                CUDA_CHECK(cudaStreamWaitEvent(mainStream.handle(), shPtrStopEvents[i]->handle()));

                            // ungate GPU to run workload every syncUpdateIntervalCnt iterations to ensure the driver does not block on GPU queues being full
                            if(0 == (powerItrIdx % syncUpdateIntervalCnt))
                            {
                                // Signal GPU to get started
                                // CU_CHECK(cuStreamWriteValue32(mainStream.handle(), reinterpret_cast<CUdeviceptr>(shPtrGpuStartSyncFlag->addr()), syncFlagVal, CU_STREAM_WRITE_VALUE_DEFAULT));
                                (*shPtrGpuStartSyncFlag)[0] = syncFlagVal;

                                // Setup wait for next set of workloads
                                syncFlagVal++;

                                for(int i = 0; i < nCtxts; i++)
                                    testWorkerVec[i].setWaitVal(syncFlagVal);
                            }
                        }

                        // Ungate the last wait request
                        (*shPtrGpuStartSyncFlag)[0] = syncFlagVal;
                    }

                    // Launch delay kernel on main stream on every iteration to ensure timeline is preserved
                    gpu_us_delay(delayUs, gpuId, mainStream.handle(), 0);

                    // Drop event on main stream for kernels to queue behind
                    startEvent.record(mainStream.handle());
                    // start timer
                    slotPatternTimer.record_begin(mainStream.handle());

                    // run pipeline
                    testWorkerVec[channelWorkerMap["PDSCH"]].pdschTxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PDSCH"]], true, pdcchCtx ? testWorkerVec[channelWorkerMap["PDCCH"]].getPdcchStopEventVecPtr() : nullptr);

                    if(pdcchCtx)
                    {
                        testWorkerVec[channelWorkerMap["PDCCH"]].pdschTxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PDCCH"]], true, nullptr, testWorkerVec[channelWorkerMap["PDSCH"]].getPdschInterSlotEventVecPtr());
                    }

                    testWorkerVec[channelWorkerMap["PUSCH"]].puschRxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PUSCH"]]);

                    if(prachCtx)
                    {
                        testWorkerVec[channelWorkerMap["PRACH"]].puschRxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PRACH"]], testWorkerVec[channelWorkerMap["PUSCH"]].getPusch2StartEvent());
                    }

                    if(pucchCtx)
                    {
                        testWorkerVec[channelWorkerMap["PUCCH"]].puschRxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["PUCCH"]], testWorkerVec[channelWorkerMap["PUSCH"]].getPusch2StartEvent(), testWorkerVec[channelWorkerMap["PUSCH"]].getPuschStartEvent());
                    }

                    if(srsCtx)
                    {
                        testWorkerVec[channelWorkerMap["SRS"]].puschRxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["SRS"]]);
                    }

                    // SSB starts at beginning of 5th slot in uldl == case
                    if(num_SSB_cells)
                    {
                        testWorkerVec[channelWorkerMap["SSB"]].pdschTxRun(startEvent.handle(), shPtrStopEvents[channelWorkerMap["SSB"]], true /*waitRsp*/, nullptr, uldl == 3 ? nullptr : testWorkerVec[channelWorkerMap["PDSCH"]].getPdschInterSlotEventVecPtr() /*uldl == 5:  start at 5th slot*/);
                    }

                    for(int i = 0; i < nCtxts; i++)
                        CUDA_CHECK(cudaStreamWaitEvent(mainStream.handle(), shPtrStopEvents[i]->handle()));

                    // end timer
                    slotPatternTimer.record_end(mainStream.handle());
                    mainStream.synchronize();
                    float et = slotPatternTimer.elapsed_time_ms();
                    tot_slotPattern_time += et;

                    // Evaluate result
                    testWorkerVec[channelWorkerMap["PDSCH"]].eval();

                    testWorkerVec[channelWorkerMap["PUSCH"]].eval(printCbErrors);

                    if(prachCtx)
                        testWorkerVec[channelWorkerMap["PRACH"]].eval();

                    if(pdcchCtx)
                        testWorkerVec[channelWorkerMap["PDCCH"]].eval();

                    if(pucchCtx)
                        testWorkerVec[channelWorkerMap["PUCCH"]].eval();

                    if(srsCtx)
                        testWorkerVec[channelWorkerMap["SRS"]].eval();

                    if(num_SSB_cells)
                        testWorkerVec[channelWorkerMap["SSB"]].eval();
                }
                // print results
                avg_slotPattern_time_us = tot_slotPattern_time * 1000 / static_cast<float>(nTimingItrs);
                printf("\n-----------------------------------------------------------\n");
                printf("Slot pattern # %d\n", patternIdx);
                printf("average slot pattern run time: %.2f us (averaged over %d iterations) \n", avg_slotPattern_time_us, nTimingItrs);

                if(runPUSCHVec[patternIdx])
                {
                    float startTimePusch = testWorkerVec[channelWorkerMap["PUSCH"]].getTotPuschStartTime() * 1000 / static_cast<float>(nTimingItrs);
                    float timePusch      = testWorkerVec[channelWorkerMap["PUSCH"]].getTotPuschRunTime() * 1000 / static_cast<float>(nTimingItrs);
                    printf("Average PUSCH run time: %.2f us from %.2f (averaged over %d iterations) \n", timePusch, startTimePusch, nTimingItrs);
                    if(uldl == 5)
                    {
                        float startTimePusch2 = testWorkerVec[channelWorkerMap["PUSCH"]].getTotPusch2StartTime() * 1000 / static_cast<float>(nTimingItrs);
                        float timePusch2      = testWorkerVec[channelWorkerMap["PUSCH"]].getTotPusch2RunTime() * 1000 / static_cast<float>(nTimingItrs);
                        printf("Average PUSCH2 run time: %.2f us from %.2f (averaged over %d iterations) \n", timePusch2, startTimePusch2, nTimingItrs);
                    }
                }
                if(runPUCCHVec[patternIdx])
                {
                    float startTimePUCCH = testWorkerVec[channelWorkerMap["PUCCH"]].getTotPucchStartTime() * 1000 / static_cast<float>(nTimingItrs);
                    float timePUCCH      = testWorkerVec[channelWorkerMap["PUCCH"]].getTotPucchRunTime() * 1000 / static_cast<float>(nTimingItrs);
                    printf("Average PUCCH run time: %.2f us from %.2f (averaged over %d iterations) \n", timePUCCH, startTimePUCCH, nTimingItrs);
                    if(uldl == 5)
                    {
                        float startTimePucch2 = testWorkerVec[channelWorkerMap["PUCCH"]].getTotPucch2StartTime() * 1000 / static_cast<float>(nTimingItrs);
                        float timePucch2      = testWorkerVec[channelWorkerMap["PUCCH"]].getTotPucch2RunTime() * 1000 / static_cast<float>(nTimingItrs);
                        printf("Average PUCCH2 run time: %.2f us from %.2f (averaged over %d iterations) \n", timePucch2, startTimePucch2, nTimingItrs);
                    }
                }

                if(runSRSVec[patternIdx])
                {
                    float startTimeSRS = testWorkerVec[channelWorkerMap["SRS"]].getTotSRSStartTime() * 1000 / static_cast<float>(nTimingItrs);
                    float timeSRS      = testWorkerVec[channelWorkerMap["SRS"]].getTotSRSRunTime() * 1000 / static_cast<float>(nTimingItrs);
                    printf("Slot # %d: average SRS1  run time: %.2f us from %.2f (averaged over %d iterations) \n", 0, timeSRS, startTimeSRS, nTimingItrs);

                    if(splitSRScells50_50)
                    {
                        float startTimeSRS2 = testWorkerVec[channelWorkerMap["SRS"]].getTotSRS2StartTime() * 1000 / static_cast<float>(nTimingItrs);
                        float timeSRS2      = testWorkerVec[channelWorkerMap["SRS"]].getTotSRS2RunTime() * 1000 / static_cast<float>(nTimingItrs);
                        printf("Slot # %d: average SRS2  run time: %.2f us from %.2f (averaged over %d iterations) \n", 4, timeSRS2, startTimeSRS2, nTimingItrs);
                    }
                }
                if(runPRACHVec[patternIdx])
                {
                    float startTimePRACH = testWorkerVec[channelWorkerMap["PRACH"]].getTotPrachStartTime() * 1000 / static_cast<float>(nTimingItrs);
                    float timePRACH      = testWorkerVec[channelWorkerMap["PRACH"]].getTotPrachRunTime() * 1000 / static_cast<float>(nTimingItrs);
                    printf("Slot # %d: average PRACH run time: %.2f us from %.2f (averaged over %d iterations) \n", 0, timePRACH, startTimePRACH, nTimingItrs);
                }

                if(runPDCCHVec[patternIdx])
                {
                    for(int sl = 0; sl < slots_per_pattern; sl++)
                    {
                        float startTimePdcch = testWorkerVec[channelWorkerMap["PDCCH"]].getPdcchStartTimes()[sl] * 1000 / static_cast<float>(nTimingItrs);
                        float timePdcch      = testWorkerVec[channelWorkerMap["PDCCH"]].getPdcchIterTimes()[sl] * 1000 / static_cast<float>(nTimingItrs);
                        printf("Slot # %d: average PDCCH run time: %.2f us from %.2f (averaged over %d iterations) \n", sl, timePdcch, startTimePdcch, nTimingItrs);
                    }
                }

                if(runCSIRSVec[patternIdx])
                {
                    for(int sl = 0; sl < slots_per_pattern; sl++)
                    {
                        float startTimeCSIRS = testWorkerVec[channelWorkerMap["PDCCH"]].getCSIRSStartTimes()[sl] * 1000 / static_cast<float>(nTimingItrs);
                        float timeCSIRS      = testWorkerVec[channelWorkerMap["PDCCH"]].getCSIRSIterTimes()[sl] * 1000 / static_cast<float>(nTimingItrs);
                        printf("Slot # %d: average CSIRS run time: %.2f us from %.2f (averaged over %d iterations) \n", sl, timeCSIRS, startTimeCSIRS, nTimingItrs);
                    }
                }

                if(runPDSCHVec[patternIdx])
                {
                    for(int sl = 0; sl < slots_per_pattern; sl++)
                    {
                        if(uldl == 5)
                        {
                            if(runBWCVec[patternIdx] && !(sl == slots_per_pattern - 1 + (patternMode[patternIdx] == 3 || patternMode[patternIdx] == 6)))
                            {
                                float  timeBwcStart = testWorkerVec[channelWorkerMap["PDSCH"]].getBWCIterStartTimes()[sl] * 1000 / static_cast<float>(nTimingItrs);                                
                                float timeBWC            = testWorkerVec[channelWorkerMap["PDSCH"]].getBWCIterTimes()[sl] * 1000 / static_cast<float>(nTimingItrs);
                                printf("Slot # %d: average BWC   run time: %.2f us from %.2f us (averaged over %d iterations) \n", sl, timeBWC, timeBwcStart, nTimingItrs);
                            }

                            float timePdsch          = testWorkerVec[channelWorkerMap["PDSCH"]].getPdschIterTimes()[sl] * 1000 / static_cast<float>(nTimingItrs);
                            float timePdschNextStart = testWorkerVec[channelWorkerMap["PDSCH"]].getPdschSlotStartTimes()[sl + (patternMode[patternIdx] == 3 || patternMode[patternIdx] == 6)] * 1000 / static_cast<float>(nTimingItrs);
                            printf("Slot # %d: average PDSCH run time: %.2f us from %.2f (averaged over %d iterations) \n", sl + (patternMode[patternIdx] == 3 || patternMode[patternIdx] == 6), timePdsch, timePdschNextStart, nTimingItrs);
                        }
                        else
                        {
                            float timePdsch          = testWorkerVec[channelWorkerMap["PDSCH"]].getPdschIterTimes()[sl] * 1000 / static_cast<float>(nTimingItrs);
                            float timePdschNextStart = testWorkerVec[channelWorkerMap["PDSCH"]].getPdschSlotStartTimes()[sl] * 1000 / static_cast<float>(nTimingItrs);
                            printf("Slot # %d: average PDSCH run time: %.2f us from %.2f (averaged over %d iterations) \n", sl, timePdsch, timePdschNextStart, nTimingItrs);

                            if(runBWCVec[patternIdx])
                            {
                                float timePdschNextStart = testWorkerVec[channelWorkerMap["PDSCH"]].getPdschSlotStartTimes()[sl + (sl == slots_per_pattern - 1)] * 1000 / static_cast<float>(nTimingItrs);
                                float timeBWC            = testWorkerVec[channelWorkerMap["PDSCH"]].getBWCIterTimes()[sl] * 1000 / static_cast<float>(nTimingItrs);
                                printf("Slot # %d: average BWC   run time: %.2f us from %.2f (averaged over %d iterations) \n", sl == slots_per_pattern - 1 ? slots_per_pattern : sl, timeBWC, timePdschNextStart, nTimingItrs);
                            }
                        }
                    }
                }

                if(runSSBVec[patternIdx])
                {
                    for(uint32_t ssbSlotIdx = 0; ssbSlotIdx < nSsbSlots; ssbSlotIdx++)
                    {
                        float startTimeSSB = testWorkerVec[channelWorkerMap["SSB"]].getTotSSBStartTime()[ssbSlotIdx] * 1000 / static_cast<float>(nTimingItrs);
                        float timeSSB      = testWorkerVec[channelWorkerMap["SSB"]].getTotSSBRunTime()[ssbSlotIdx] * 1000 / static_cast<float>(nTimingItrs);
                        printf("Slot # %d: average SSB   run time: %.2f us from %.2f (averaged over %d iterations) \n", 5 * (patternMode[patternIdx] == 3 || patternMode[patternIdx] == 6) + ssbSlotIdx, timeSSB, startTimeSSB, nTimingItrs);
                    }
                }

                if(printCellMetrics)
                {
                    testWorkerVec[channelWorkerMap["PDSCH"]].print();
                    testWorkerVec[channelWorkerMap["PUSCH"]].print(printCbErrors);
                    if(prachCtx)
                    {
                        testWorkerVec[channelWorkerMap["PRACH"]].print();
                    }
                    if(pdcchCtx)
                        testWorkerVec[channelWorkerMap["PDCCH"]].print();
                    if(pucchCtx)
                        testWorkerVec[channelWorkerMap["PUCCH"]].print();
                    if(srsCtx)
                        testWorkerVec[channelWorkerMap["SRS"]].print();
                    if(num_SSB_cells)
                        testWorkerVec[channelWorkerMap["SSB"]].print();
                }

                // clean up params
                testWorkerVec[channelWorkerMap["PDSCH"]].pdschTxClean();
                testWorkerVec[channelWorkerMap["PDSCH"]].resetEvalBuffers();
                if(prachCtx)
                    testWorkerVec[channelWorkerMap["PRACH"]].resetEvalBuffers();
                if(pdcchCtx)
                    testWorkerVec[channelWorkerMap["PDCCH"]].resetEvalBuffers();
                if(pucchCtx)
                    testWorkerVec[channelWorkerMap["PUCCH"]].resetEvalBuffers();
                if(srsCtx)
                    testWorkerVec[channelWorkerMap["SRS"]].resetEvalBuffers();
                testWorkerVec[channelWorkerMap["PUSCH"]].resetEvalBuffers(printCbErrors);
                if(num_SSB_cells)
                    testWorkerVec[channelWorkerMap["SSB"]].resetEvalBuffers();
                // reset time
                tot_slotPattern_time    = 0;
                avg_slotPattern_time_us = 0;
            }
            // deinitialize workers
            for(int i = 0; i < nCtxts; i++)
                testWorkerVec[i].deinit();
#if USE_NVTX
            nvtxRangePop();
#endif
        }
        //------------------------------------------------------------
        //PSCH
        else if(uldl == 4)
        {
            //------------------------------------------------------------------------------------
            // Divide cells between contexts

            printf("\n\n nPschCellsPerPattern: %d, num_patterns: %d\n\n", nPschCellsPerPattern, num_patterns);

            uint32_t      nCellsPerCtxt = nPschCellsPerPattern / nCtxts;
            string3Dvec_t inCtxFileNamesPuschRx(num_patterns); // Dim: num_patterns x nCtxts x nCellsPerCtxt
            string3Dvec_t inCtxFileNamesPdschTx(num_patterns); // Dim: num_patterns x nCtxts x nCellsPerCtxt

            for(uint32_t patternIdx = 0; patternIdx < num_patterns; ++patternIdx)
            {
                inCtxFileNamesPuschRx[patternIdx].resize(nCtxts);
                inCtxFileNamesPdschTx[patternIdx].resize(nCtxts);

                for(uint32_t ctxtIdx = 0; ctxtIdx < nCtxts; ++ctxtIdx)
                {
                    for(uint32_t i = 0; i < nCellsPerCtxt; ++i)
                    {
                        uint32_t cellIdx = ctxtIdx * nCellsPerCtxt + i;
                        inCtxFileNamesPuschRx[patternIdx][ctxtIdx].emplace_back(inFileNamesPuschRx[patternIdx][cellIdx]);
                        inCtxFileNamesPdschTx[patternIdx][ctxtIdx].emplace_back(inFileNamesPdschTx[patternIdx][cellIdx]);
                    }
                }
            }

            //-----------------------------------------------------------------------------------
            // Create workers
            std::vector<cuPHYTestWorker*> pTestWorkers(nCtxts);

            for(uint32_t ii = 0; ii < nCtxts; ii++)
            {
                pTestWorkers[ii] = new cuPHYTestWorker(std::string("PschTxRxTestWorker"), ii, cpuIds[ii], gpuIds[ii], cpuThrdSchdPolicy, cpuThrdPrios[ii], mpsSubctxSmCounts[ii], cmdQVec[ii], rspQVec[ii], uldl, dbgMsgLevel);
            }
            finishPschSim(pTestWorkers, inCtxFileNamesPuschRx, inCtxFileNamesPdschTx, nTimingItrs, nPowerItrs, num_patterns, nCtxts, printCbErrors, mainStream, gpuId, delayUs, powerDelayUs, ref_check_pdsch, identical_ldpc_configs, pdsch_proc_mode, pusch_proc_mode, fp16Mode, descramblingOn, nCellsPerCtxt, nStrmsPerCtxt, nPschItrsPerStrm, slotPatternTimer, tot_slotPattern_time, avg_slotPattern_time_us, shPtrGpuStartSyncFlag, cuStrmPrioMap, syncUpdateIntervalCnt, enableLdpcThroughputMode, printCellMetrics, group_pdsch_cells, nPdschCellsPerStrm, group_pusch_cells, pdschPrms, puschPrms, ldpcLaunchMode);

            for(uint32_t ii = 0; ii < nCtxts; ii++)
            {
                delete pTestWorkers[ii];
            }
        }
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_TAG_BASE_CUPHY, AERIAL_CUPHY_EVENT,  "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    return returnValue;
}

void pschRunCore(std::vector<cuPHYTestWorker*>& pTestWorkers, uint32_t nCtxts, cuphy::stream& mainStream, cuphy::event& startEvent, std::vector<std::shared_ptr<cuphy::event>>& shPtrStopEvents)
{
    // Drop event on main stream for kernels to queue behind
    startEvent.record(mainStream.handle());

    // run pipeline
    for(uint32_t ctxIdx = 0; ctxIdx < nCtxts; ++ctxIdx)
    {
        pTestWorkers[ctxIdx]->pschTxRxRun(startEvent.handle(), shPtrStopEvents[ctxIdx]);
    }

    for(uint32_t ctxIdx = 0; ctxIdx < nCtxts; ++ctxIdx)
    {
        CUDA_CHECK(cudaStreamWaitEvent(mainStream.handle(), shPtrStopEvents[ctxIdx]->handle()));
    }
}

void finishPschSim(std::vector<cuPHYTestWorker*>& pTestWorkers, string3Dvec_t& inCtxFileNamesPuschRx, string3Dvec_t& inCtxFileNamesPdschTx, uint32_t nTimingItrs, uint32_t nPowerItrs, uint32_t num_patterns, uint32_t nCtxts, bool printCbErrors, cuphy::stream& mainStream, int32_t gpuId, uint32_t delayUs, uint32_t powerDelayUs, bool ref_check_pdsch, bool identical_ldpc_configs, cuphyPdschProcMode_t pdsch_proc_mode, uint64_t pusch_proc_mode, uint32_t fp16Mode, int descramblingOn, uint32_t nCellsPerCtxt, uint32_t nStrmsPerCtxt, uint32_t nPschItrsPerStrm, cuphy::event_timer& slotPatternTimer, float tot_slotPattern_time, float avg_slotPattern_time_us, std::shared_ptr<cuphy::buffer<uint32_t, cuphy::pinned_alloc>>& shPtrGpuStartSyncFlag, std::map<std::string, int>& cuStrmPrioMap, uint32_t syncUpdateIntervalCnt, bool enableLdpcThroughputMode, bool printCellMetrics, bool pdsch_group_cells, uint32_t pdsch_cells_per_stream, bool pusch_group_cells, maxPDSCHPrms pdschPrms, maxPUSCHPrms puschPrms, uint32_t ldpcLaunchMode)
{
    //------------------------------------------------------------------------------------
    // initialize workers
    for(uint32_t ctxIdx = 0; ctxIdx < nCtxts; ++ctxIdx)
    {
        pTestWorkers[ctxIdx]->init(nStrmsPerCtxt, nPschItrsPerStrm, nTimingItrs, cuStrmPrioMap, shPtrGpuStartSyncFlag);
        pTestWorkers[ctxIdx]->pdschTxInit(inCtxFileNamesPdschTx[0][ctxIdx], ref_check_pdsch, identical_ldpc_configs, pdsch_proc_mode, pdsch_group_cells, pdsch_cells_per_stream, pdschPrms);
        pTestWorkers[ctxIdx]->puschRxInit(inCtxFileNamesPuschRx[0][ctxIdx], 1, 1, true, pusch_proc_mode, enableLdpcThroughputMode, pusch_group_cells, puschPrms, ldpcLaunchMode);
    }

    //------------------------------------------------------------------------------------
    // Loop over slot patterns

    // For serial execution of workers, N_WORKER_INST+1 are needed
    cuphy::event                               startEvent(cudaEventDisableTiming);
    std::vector<std::shared_ptr<cuphy::event>> shPtrStopEvents;
    for(uint32_t i = 0; i < nCtxts; ++i) shPtrStopEvents.emplace_back(std::make_shared<cuphy::event>(cudaEventDisableTiming));

    bool isPschTxRx = true;
    for(int patternIdx = 0; patternIdx < num_patterns; patternIdx++)
    {
        // timing iterations
        for(uint32_t itrIdx = 0; itrIdx < nTimingItrs; ++itrIdx)
        {
            // setup workers
            for(uint32_t ctxIdx = 0; ctxIdx < nCtxts; ++ctxIdx)
            {
                pTestWorkers[ctxIdx]->pdschTxSetup(inCtxFileNamesPdschTx[patternIdx][ctxIdx]);
                pTestWorkers[ctxIdx]->puschRxSetup(inCtxFileNamesPuschRx[patternIdx][ctxIdx]);
            }

            if(0 != nPowerItrs)
            {
                // Initialize CPU-GPU start sync flag
                uint32_t syncFlagVal        = 1;
                (*shPtrGpuStartSyncFlag)[0] = 0;
                for(uint32_t ctxIdx = 0; ctxIdx < nCtxts; ++ctxIdx)
                {
                    pTestWorkers[ctxIdx]->setWaitVal(syncFlagVal);
                }

                for(uint32_t powerItrIdx = 1; powerItrIdx <= nPowerItrs; ++powerItrIdx)
                {
                    if(powerDelayUs > 0) gpu_us_delay(powerDelayUs, gpuId, mainStream.handle(), 0);
                    pschRunCore(pTestWorkers, nCtxts, mainStream, startEvent, shPtrStopEvents);

                    // ungate GPU to run workload every syncUpdateIntervalCnt iterations to ensure the driver does not block on GPU queues being full
                    if(0 == (powerItrIdx % syncUpdateIntervalCnt))
                    {
                        // Signal GPU to get started
                        // CU_CHECK(cuStreamWriteValue32(mainStream.handle(), reinterpret_cast<CUdeviceptr>(shPtrGpuStartSyncFlag->addr()), syncFlagVal, CU_STREAM_WRITE_VALUE_DEFAULT));
                        (*shPtrGpuStartSyncFlag)[0] = syncFlagVal;

                        // Setup wait for next set of workloads
                        syncFlagVal++;
                        for(uint32_t ctxIdx = 0; ctxIdx < nCtxts; ++ctxIdx)
                        {
                            pTestWorkers[ctxIdx]->setWaitVal(syncFlagVal);
                        }
                    }
                }
                // Ungate the last wait request
                (*shPtrGpuStartSyncFlag)[0] = syncFlagVal;
            }
            else
            {
                // Launch delay kernel on main stream
                gpu_us_delay(delayUs, gpuId, mainStream.handle(), 0);
            }

            // start timer
            slotPatternTimer.record_begin(mainStream.handle());

            pschRunCore(pTestWorkers, nCtxts, mainStream, startEvent, shPtrStopEvents);

            // end timer
            slotPatternTimer.record_end(mainStream.handle());
            mainStream.synchronize();
            tot_slotPattern_time += slotPatternTimer.elapsed_time_ms();

            // Evaluate result
            for(uint32_t ctxIdx = 0; ctxIdx < nCtxts; ++ctxIdx)
            {
                pTestWorkers[ctxIdx]->eval(printCbErrors, isPschTxRx);
            }
        }
        // print results
        avg_slotPattern_time_us = tot_slotPattern_time * 1000 / static_cast<float>(nTimingItrs);
        printf("\n-----------------------------------------------------------\n");
        printf("Slot # %d\n", patternIdx);
        printf("average slot run time: %.2f us (averaged over %d iterations) \n", avg_slotPattern_time_us, nTimingItrs);

        for(uint32_t ctxIdx = 0; ctxIdx < nCtxts; ++ctxIdx)
        {
            float timePdsch = pTestWorkers[ctxIdx]->getPdschIterTimes()[0] * 1000 / static_cast<float>(nTimingItrs);
            printf("Ctx # %d: average PDSCH run time: %.2f us (averaged over %d iterations) \n", ctxIdx, timePdsch, nTimingItrs);

            float timePusch = pTestWorkers[ctxIdx]->getTotPuschRunTime() * 1000 / static_cast<float>(nTimingItrs);
            printf("Ctx # %d: average PUSCH run time: %.2f us (averaged over %d iterations) \n", ctxIdx, timePusch, nTimingItrs);
        }
        if(printCellMetrics)
        {
            for(uint32_t ctxIdx = 0; ctxIdx < nCtxts; ++ctxIdx)
            {
                pTestWorkers[ctxIdx]->print(printCbErrors, isPschTxRx);
            }
        }

        // clean up params
        for(uint32_t ctxIdx = 0; ctxIdx < nCtxts; ++ctxIdx)
        {
            pTestWorkers[ctxIdx]->pdschTxClean();
            pTestWorkers[ctxIdx]->resetEvalBuffers(printCbErrors);
        }

        // reset time
        tot_slotPattern_time    = 0;
        avg_slotPattern_time_us = 0;
    }

    // deinitialize workers
    for(uint32_t ctxIdx = 0; ctxIdx < nCtxts; ++ctxIdx)
    {
        pTestWorkers[ctxIdx]->deinit();
    }
}

void readSmIds(std::vector<cuPHYTestWorker*>& pTestWorkers, cuphy::stream& cuphyStrm)
{
    std::vector<std::shared_ptr<cuphy::event>> shPtrRdSmIdWaitEvents(pTestWorkers.size());

    uint32_t wrkrIdx = 0;
    for(auto& pTestWorker : pTestWorkers)
    {
        pTestWorker->readSmIds(shPtrRdSmIdWaitEvents[wrkrIdx++]);
    }

    for(auto& shPtrRdSmIdWaitEvent : shPtrRdSmIdWaitEvents)
    {
        CUDA_CHECK(cudaStreamWaitEvent(cuphyStrm.handle(), shPtrRdSmIdWaitEvent->handle()));
    }

    std::vector<std::vector<uint32_t>> smIds(pTestWorkers.size());
    wrkrIdx = 0;
    for(auto& pTestWorker : pTestWorkers)
    {
        uint32_t  nSmIds    = 0;
        uint32_t* pSmIdsGpu = pTestWorker->getSmIdsGpu(nSmIds);
        smIds[wrkrIdx]      = std::move(std::vector<uint32_t>(nSmIds));

        cudaMemcpyAsync(smIds[wrkrIdx].data(), pSmIdsGpu, sizeof(uint32_t) * nSmIds, cudaMemcpyDeviceToHost, cuphyStrm.handle());
        cuphyStrm.synchronize();
        std::sort(smIds[wrkrIdx].begin(), smIds[wrkrIdx].end(), std::less<uint32_t>());

        printf("Pusch Worker[%d]: SM Id counts %lu\n", wrkrIdx, smIds[wrkrIdx].size());
        for(int i = 0; i < smIds[wrkrIdx].size(); ++i)
        {
            printf("%02u ", smIds[wrkrIdx][i]);
        }
        printf("\n");
        wrkrIdx++;
    }
}
