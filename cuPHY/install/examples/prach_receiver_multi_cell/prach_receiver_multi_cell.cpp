/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "cuphy.hpp"
#include "cuphy_channels.hpp"
#include "util.hpp"
#include "datasets.hpp"
#include "test_config.hpp"

#define NUM_PREAMBLE 64

using namespace std;
using namespace cuphy;

void usage() {

    printf("  Options:\n");
    printf("    -h                          Display usage information\n");
    printf("    -l  log_filename            filename to save log output\n");
    printf("    -i  input_filename          Input yaml file for slot/cell config \n");
    printf("    -n  # of cell               Number of cells to run. Used only if input file has extension h5\n");
    printf("    -r  # of iterations         Number of run iterations to run\n");
    printf("    -d  # of microseconds       Delay kernel duration in us\n");
    printf("    -m  processing mode         Proc mode: streams(0x0), graphs (0x1) (default = 0x0)\n");
    printf("    -k                          Enable reference check. Compare GPU output with test vector.\n");
    printf("    -g                          Execute all cells in a slot on the same PRACH object.\n"); // Reminder: they should all have identical static parameters.
}

int main(int argc, char* argv[])
{

    int returnValue = 0;
    char nvlog_yaml_file[1024];
    // Relative path from binary to default nvlog_config.yaml
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    std::string log_name = "prach.log";
    nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
    pthread_t log_thread_id = -1;

    //------------------------------------------------------------------
    // Parse command line arguments
    constexpr uint32_t N_MAX_INST = 36;
    int         iArg = 1;
    std::string inputFileName;
    uint32_t    num_iterations      = 1;
    uint32_t    nInst               = 1;
    bool        ref_check_prach     = false;
    uint32_t    delayUs             = 10000;
    bool        group_cells         = false; // Group cells in the same cell-group.
    uint64_t    procModeBmsk    = 0;

    try
    {
        while(iArg < argc)
        {
            if('-' == argv[iArg][0])
            {
                switch(argv[iArg][1])
                {
                case 'h':
                    usage();
                    throw std::invalid_argument(fmt::format("Unknown option: {}", argv[iArg]));
                case 'l':
                    if(++iArg < argc)
                    {
                        log_name.assign(argv[iArg++]);
                    }
                    break;
                case 'i':
                    if(++iArg >= argc)
                    {
                        throw std::invalid_argument(fmt::format("ERROR: No filename provided."));
                    }
                    inputFileName.assign(argv[iArg++]);
                    break;
                case 'n':
                        if((++iArg >= argc) ||
                        (1 != sscanf(argv[iArg], "%i", &nInst)) ||
                        ((nInst <= 0) || (nInst > N_MAX_INST)))
                        {
                            throw std::invalid_argument(fmt::format("Invalid number of instances (should be within [1,{}}]", N_MAX_INST));
                        }
                        ++iArg;
                        break;
                case 'r':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &num_iterations)) || ((num_iterations <= 0)))
                    {
                        throw std::invalid_argument(fmt::format("Invalid number of iterations: {}", argv[iArg]));
                    }
                    ++iArg;
                    break;
                case 'd':
                        delayUs = std::stoi(argv[++iArg]);
                        ++iArg;
                        break;
                case 'g':
                    group_cells = true; // Group all cells in a slot and run them on a single PRACH channel.
                    ++iArg;
                    break;
                case 'm':
                        if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%lu", &procModeBmsk)) || ((PRACH_PROC_MODE_NO_GRAPH != procModeBmsk) && (PRACH_PROC_MODE_WITH_GRAPH != procModeBmsk)))
                        {
                            throw std::invalid_argument(fmt::format("Invalid processing mode: {:x}", procModeBmsk));
                        }
                        ++iArg;
                        break;
                case 'k':
                    ref_check_prach = true;
                    ++iArg;
                    break;
                default:
                    usage();
                    throw std::invalid_argument(fmt::format("Unknown option: {}", argv[iArg]));
                }
            }
            else
            {
                throw std::invalid_argument(fmt::format("Invalid command line argument: {}", argv[iArg]));
            }
        }
        if(inputFileName.empty())
        {
            usage();
            throw std::invalid_argument(fmt::format("No filename provided."));
            
        }
        log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, log_name.c_str(),NULL);
        nvlog_fmtlog_thread_init();

        std::unique_ptr<cuphy::test_config> testCfg;
        int num_cells = 1;
        int num_slots = 1;
        
        std::string inFileExtn = inputFileName.substr(inputFileName.find_last_of(".") + 1);
        NVLOGC_FMT(NVLOG_PRACH, "File extension: {}", inFileExtn);
        if(inFileExtn == "yaml")
        {
            testCfg = make_unique<cuphy::test_config>(inputFileName.c_str());
            testCfg->print();

            num_cells = testCfg->num_cells(); // The same number of cells is present across all slots.
            num_slots = testCfg->num_slots();
        }
        else
        {
            num_cells = nInst;
            NVLOGC_FMT(NVLOG_PRACH, "number of cells: {}", nInst);
            NVLOGC_FMT(NVLOG_PRACH, "File: {}", inputFileName);
        }
        
        const std:: string channelName = "PRACH";

        std::vector<std::vector<prach_rx>>          m_prachRxPipes;
        std::vector<std::vector<PrachApiDataset>>   m_prachApiDataSets;

        std::vector<stream>        streams;
        cudaEvent_t                start_streams_event;
        std::vector<cudaEvent_t> stop_streams_events(num_cells);

        const int num_prach_objects = group_cells ? 1 : num_cells;

        m_prachRxPipes.resize(num_slots);
        m_prachApiDataSets.resize(num_slots);

        for(int idxSlot = 0; idxSlot < num_slots; idxSlot++) {
            // Reminder num_prach_objects is the number of cells, if group_cells is false, or 1 otherwise
            m_prachApiDataSets[idxSlot].reserve(num_prach_objects);

            // Loop over all cells (not num_prach_objects) to populate the datasets
            for(int i = 0; i < num_cells; i += 1)
            {
                std::string tv_filename;
                if(testCfg)
                {
                    tv_filename = testCfg->slots()[idxSlot].at(channelName)[i];
                }
                else
                {
                    tv_filename = inputFileName;
                }

                if((idxSlot == 0) && (i < num_prach_objects)) {
                    streams.emplace_back(cudaStreamNonBlocking);
                }
                if (group_cells) {
                    if (i == 0) {
                        m_prachApiDataSets[idxSlot].emplace_back(tv_filename, streams[i].handle(), procModeBmsk, ref_check_prach);
                    } else {
                        // Nothing to cumulative update for the static parameters.
                        // Update the dyanmic parameters
                        m_prachApiDataSets[idxSlot][0].cumulativeUpdate(tv_filename, streams[0].handle());
                    }
                } else {
                    m_prachApiDataSets[idxSlot].emplace_back(tv_filename, streams[i].handle(), procModeBmsk, ref_check_prach);
                    m_prachApiDataSets[idxSlot][i].finalize(streams[i].handle());
                    m_prachRxPipes[idxSlot].emplace_back(m_prachApiDataSets[idxSlot][i].prachStatPrms);
                }
            }

            if (group_cells) {
                m_prachApiDataSets[idxSlot][0].finalize(streams[0].handle());
                m_prachRxPipes[idxSlot].emplace_back(m_prachApiDataSets[idxSlot][0].prachStatPrms);
            }
        }

        NVLOGC_FMT(NVLOG_PRACH, "Timing {} PRACH pipeline(s) Run(). ",num_cells);
        if (group_cells)
        {
            NVLOGC_FMT(NVLOG_PRACH, "Grouping all cells in a slot.");
        }
        NVLOGC_FMT(NVLOG_PRACH, "- NB: Allocations, setup processing not included.\n");

        /* Prach::Run for all num_cells pipelines will be timed on streams[0].handle() CUDA stream (note, that is NOT stream 0).
        Have that stream wait for all other streams to complete their work too. */

        for(int i = 0; i < num_prach_objects; i++)
        {
            CUDA_CHECK(cudaEventCreateWithFlags(&stop_streams_events[i], cudaEventDisableTiming));
        }
        CUDA_CHECK(cudaEventCreateWithFlags(&start_streams_event, cudaEventDisableTiming));

        float total_time_slot[num_slots]; // Total time of a slot divided by number of iterations (num_iterations)
        float total_time_single_cell_slot[num_slots][num_prach_objects];

        for(int idxSlot = 0; idxSlot < num_slots; idxSlot++)
        {
            float                    total_time = 0;
            std::vector<float>       total_time_single_cell(num_prach_objects, 0);
            std::vector<event_timer> cuphy_timer_single_cell(num_prach_objects);

            for(int i = 0; i < num_prach_objects; i++)
            {
                m_prachRxPipes[idxSlot][i].setup(m_prachApiDataSets[idxSlot][i].prachDynPrms);
            }
            gpu_us_delay(delayUs, 0, streams[0].handle());
            CUDA_CHECK(cudaEventRecord(start_streams_event, streams[0].handle()));

            for(int iter = 0; iter < num_iterations; iter++)
            {
                event_timer cuphy_timer;
                cuphy_timer.record_begin(streams[0].handle());

                for(int i = 0; i < num_prach_objects; i++)
                {
                    cudaStream_t strm_handle = streams[i].handle();

                    if(i != 0)
                    {
                        CUDA_CHECK(cudaStreamWaitEvent(streams[i].handle(), start_streams_event, 0));
                    }

                    cuphy_timer_single_cell[i].record_begin(streams[i].handle());

                    m_prachRxPipes[idxSlot][i].run();
                    cuphy_timer_single_cell[i].record_end(streams[i].handle());

                    /* Record a stop event on all streams but the streams[0].handle() stream. */
                    /* Have streams[0].handle() stream wait for all other streams to complete their work before stopping the timer. */
                    if(i != 0)
                    {
                        CUDA_CHECK(cudaEventRecord(stop_streams_events[i], strm_handle));
                        CUDA_CHECK(cudaStreamWaitEvent(streams[0].handle(), stop_streams_events[i], 0));
                    }
                }

                cuphy_timer.record_end(streams[0].handle());
                cuphy_timer.synchronize();
                total_time += cuphy_timer.elapsed_time_ms();

                for(int i = 0; i < num_prach_objects; i++)
                {
                    cuphy_timer_single_cell[i].synchronize(); // To be safe
                    total_time_single_cell[i] += cuphy_timer_single_cell[i].elapsed_time_ms();
                }

                if (ref_check_prach && (iter == 0)) {
                    int errors = 0;
                for(int i = 0; i < num_prach_objects; i++)
                {
                    errors += m_prachApiDataSets[idxSlot][i].evaluateOutput();
                }

                if (errors != 0) {
                        exit(1);
                    }
                }
                gpu_us_delay(delayUs, 0, streams[0].handle()); // 10ms delay kernel. Can update/comment out.
                CUDA_CHECK(cudaEventRecord(start_streams_event, streams[0].handle()));
            }
            total_time_slot[idxSlot] = total_time / num_iterations;

            for(int i = 0; i < num_prach_objects; i++)
            {
                total_time_single_cell_slot[idxSlot][i] = total_time_single_cell[i] / num_iterations;
            }
        }

        for(int idxSlot = 0; idxSlot < num_slots; idxSlot++)
        {
            NVLOGC_FMT(NVLOG_PRACH, "Slot # {},  PRACH pipeline(s): {:.2f} us (avg. over {} iterations)", idxSlot, total_time_slot[idxSlot] * 1000, num_iterations);
            for(int i = 0; i < num_prach_objects; i++)
            {
                if (group_cells) {
                    NVLOGC_FMT(NVLOG_PRACH, "--> PRACH object # {} with {} cells: {:.2f} us (avg over {} iterations)", i, num_cells, total_time_single_cell_slot[idxSlot][i] * 1000, num_iterations);
                } else {
                    NVLOGC_FMT(NVLOG_PRACH, "--> Cell # {} : {:.2f} us (avg over {} iterations)", i, total_time_single_cell_slot[idxSlot][i] * 1000, num_iterations);
                }
            }
        }
    }
    catch(std::exception& e)
    {
        if(log_thread_id < 0)
        {
            log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, log_name.c_str(),NULL);
            nvlog_fmtlog_thread_init();
        }
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        if(log_thread_id < 0)
        {
            log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, log_name.c_str(),NULL);
            nvlog_fmtlog_thread_init();
        }
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT, "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    nvlog_fmtlog_close(log_thread_id);
    return returnValue;

}
