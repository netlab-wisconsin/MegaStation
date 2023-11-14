/*
 * Copyright (c) 2022-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "util.hpp"
#include "ssb_tx.hpp"
#include "cuphy_channels.hpp"
#include "datasets.hpp"
#include <list>
#include <fstream>
#include "test_config.hpp"

#define SSB_STREAM_PRIORITY 0
#define PRINT_SSB_CONFIG 0 // Set to 1 for debugging purposes

using namespace cuphy;

/**
 *  @brief Print usage information for the DL pipeline example.
 */
void usage()
{
    printf("  Options:\n");
    printf("    -h                          Display usage information\n");
    printf("    -i  input_filename          Input yaml file for slot/cell config \n");
    printf("    -r  # of iterations         Number of run iterations to run\n");
    printf("    -d  # of microseconds       Delay kernel duration in us\n");
    printf("    -k                          Enable reference check. Compare GPU output with test vector (first iteration only).\n");
    printf("    -m  proc_mode               Processing mode: streams(0), graphs (1)\n");
    printf("    -g                          Execute all cells in a slot on the same SSB object with batching too.\n");
    printf("    -s  setup_mode              0 (default) - setup is not timed; 1 - time setup only; no run is run; 2 - time both setup and run, back to back.\n");
}

int main(int argc, char* argv[])
{
    int returnValue = 0;

    char nvlog_yaml_file[1024];
    // Relative path from binary to default nvlog_config.yaml
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
    pthread_t log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, "ssb_tx_multicell.log",NULL);
    nvlog_fmtlog_thread_init();

    //------------------------------------------------------------------
    // Parse command line arguments
    int         iArg = 1;
    std::string inputFileName;
    uint32_t    num_iterations  = 1;
    bool        ref_check_ssb   = false;
    uint64_t    procModeBmsk    = 0;
    uint32_t    delayUs         = 10000;
    bool        group_cells     = false; // Group cells and if possible, also process all cells in a single kernel per component.
    int         time_setup_mode = 0;     // default mode: only time run, not setup
    std::string setup_modes[3]  = {"GPU-run only", "GPU-setup only", "GPU-setup-and-run"};
    std::string proc_modes[2]   = {"in Stream mode", "in Graphs mode"};

    while(iArg < argc)
    {
        if('-' == argv[iArg][0])
        {
            switch(argv[iArg][1])
            {
            case 'h':
                usage();
                exit(0);
                break;
            case 'i':
                if(++iArg >= argc)
                {
                    NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided.");
                }
                inputFileName.assign(argv[iArg++]);
                break;
            case 'r':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &num_iterations)) || ((num_iterations <= 0)))
                {
                    NVLOGF_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of iterations");
                }
                ++iArg;
                break;
            case 'd':
                    delayUs = std::stoi(argv[++iArg]);
                    ++iArg;
                    break;
            case 'g':
                group_cells = true;
                ++iArg;
                break;
            case 'k':
                ref_check_ssb = true;
                ++iArg;
                break;
            case 'm':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%lu", &procModeBmsk)) || ((SSB_PROC_MODE_STREAMS != procModeBmsk) && (SSB_PROC_MODE_GRAPHS != procModeBmsk)))
                {
                    NVLOGF_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT,  "ERROR: Invalid processing mode ({:#x})", procModeBmsk);
                }
                ++iArg;
                break;
            case 's':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &time_setup_mode)) || ((time_setup_mode < 0)) || ((time_setup_mode > 2)))
                {
                    NVLOGF_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT,  "ERROR: Invalid process mode");
                }
                ++iArg;
                break;
            default:
                NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                usage();
                exit(1);
                break;
            }
        }
        else
        {
            NVLOGF_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
        }
    }
    if(inputFileName.empty())
    {
        usage();
        exit(1);
    }

    cuphy::test_config testCfg(inputFileName.c_str());
    const std:: string channelName = "SSB";
    testCfg.print_channel(channelName); // print only SSB channel related parts of the input YAML file
    //testCfg.print(); // print entire YAML file contents, incl. other channels not run with this example
    int num_cells = testCfg.num_cells(); // The same number of cells is present across all slots.
    int num_slots = testCfg.num_slots();

    //NVLOGC_FMT(NVLOG_SSB, "SSB multi-cell with {} cells and {} slots",  num_cells, num_slots);
    CUDA_CHECK(cudaSetDevice(0)); // Select GPU device 0

    std::vector<std::vector<cuphy::ssb_tx>>       m_ssbTxPipes;
    std::vector<std::vector<ssbStaticApiDataset>> m_ssbTxStaticApiDataSets;
    std::vector<std::vector<ssbDynApiDataset>>    m_ssbTxDynamicApiDataSets;

    std::vector<stream>        streams;
    cudaEvent_t                start_streams_event;
    std::vector<cudaEvent_t> stop_streams_events(num_cells);
    const int num_ssb_objects = group_cells ? 1 : num_cells;

    m_ssbTxPipes.resize(num_slots);
    m_ssbTxStaticApiDataSets.resize(num_slots);
    m_ssbTxDynamicApiDataSets.resize(num_slots);

    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++) {
        int cells_in_slot = testCfg.slots()[idxSlot].at(channelName).size();
        if (cells_in_slot != num_cells) {
            NVLOGF_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "Slot {} error: expected {} cells but got {}", idxSlot, num_cells, cells_in_slot);
        }
        // Reminder num_ssb_object is the number of cells, if group_cells is false, or 1 otherwise
        m_ssbTxStaticApiDataSets[idxSlot].reserve(num_ssb_objects);
        m_ssbTxDynamicApiDataSets[idxSlot].reserve(num_ssb_objects);

        // Loop over all cells (not num_ssb_objects) to populate the static and dynamic datasets
        for(int i = 0; i < num_cells; i += 1)
        {
            std::string tv_filename = testCfg.slots()[idxSlot].at(channelName)[i];
            if((idxSlot == 0) && (i < num_ssb_objects)) {
                streams.emplace_back(cudaStreamNonBlocking, SSB_STREAM_PRIORITY);
            }
            if (group_cells) {
                if (i == 0) {
                    m_ssbTxStaticApiDataSets[idxSlot].emplace_back(num_cells);
                    m_ssbTxDynamicApiDataSets[idxSlot].emplace_back(tv_filename, m_ssbTxStaticApiDataSets[idxSlot][i].ssbStatPrms.nMaxCellsPerSlot, streams[i].handle(), procModeBmsk);
                    m_ssbTxPipes[idxSlot].emplace_back(m_ssbTxStaticApiDataSets[idxSlot][i].ssbStatPrms);
                } else {
                    // Nothing to cumulative update for the static parameters.
                    // Update the dyanmic parameters
                    m_ssbTxDynamicApiDataSets[idxSlot][0].cumulativeUpdate(tv_filename, streams[0].handle());
                }
            } else {
                m_ssbTxStaticApiDataSets[idxSlot].emplace_back();
                m_ssbTxDynamicApiDataSets[idxSlot].emplace_back(tv_filename, m_ssbTxStaticApiDataSets[idxSlot][i].ssbStatPrms.nMaxCellsPerSlot, streams[i].handle(), procModeBmsk);
                m_ssbTxPipes[idxSlot].emplace_back(m_ssbTxStaticApiDataSets[idxSlot][i].ssbStatPrms);
            }
        }
    }

#if PRINT_SSB_CONFIG
    // Print dynamic dataset params for each SSB object. For dbg. purposes
    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++) {
        for(int i = 0; i < num_ssb_objects; i += 1) {
            m_ssbTxPipes[idxSlot][i].handle()->pipeline->printSsbConfig(m_ssbTxDynamicApiDataSets[idxSlot][i].ssb_dyn_params);
        }
    }
#endif

    NVLOGC_FMT(NVLOG_SSB, "");
    NVLOGC_FMT(NVLOG_SSB, "Timing {} SSB pipeline(s). ", num_cells);
    if (group_cells)
    {
        NVLOGC_FMT(NVLOG_SSB, "Grouping all cells in a slot.");
    } else {
        NVLOGC_FMT(NVLOG_SSB, "");
    }
    if (time_setup_mode== 0) {
        NVLOGC_FMT(NVLOG_SSB, "- NB: Allocations, setup processing not included.");
        NVLOGC_FMT(NVLOG_SSB, "");
    }

    /* SSB Tx Run for all num_cells pipelines will be timed on streams[0].handle() CUDA stream (note, that is NOT stream 0).
    Have that stream wait for all other streams to complete their work too. */

    for(int i = 0; i < num_ssb_objects; i++)
    {
        CUDA_CHECK(cudaEventCreateWithFlags(&stop_streams_events[i], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&start_streams_event, cudaEventDisableTiming));

    float total_time_slot[num_slots]; // Total time of a slot divided by number of iterations (num_iterations)
    float total_time_single_cell_slot[num_slots][num_ssb_objects];
    //NVLOGC_FMT(NVLOG_SSB, "num slots {}, num_ssb_objects {}", num_slots, num_ssb_objects);

    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++)
    {
        float                    total_time = 0;
        std::vector<float>       total_time_single_cell(num_ssb_objects, 0);
        std::vector<event_timer> cuphy_timer_single_cell(num_ssb_objects);

        if (time_setup_mode == 0){ // In this mode, setup is not timed
            for(int i = 0; i < num_ssb_objects; i++)
            {
                m_ssbTxPipes[idxSlot][i].setup(m_ssbTxDynamicApiDataSets[idxSlot][i].ssb_dyn_params);
            }
        }
        gpu_us_delay(delayUs, 0, streams[0].handle());
        CUDA_CHECK(cudaEventRecord(start_streams_event, streams[0].handle()));

        for(int iter = 0; iter < num_iterations; iter++)
        {
            event_timer cuphy_timer;
            cuphy_timer.record_begin(streams[0].handle());

            for(int i = 0; i < num_ssb_objects; i++)
            {
                cudaStream_t strm_handle = streams[i].handle();

                if(i != 0)
                {
                    CUDA_CHECK(cudaStreamWaitEvent(streams[i].handle(), start_streams_event, 0));
                }

                cuphy_timer_single_cell[i].record_begin(streams[i].handle());
                if (time_setup_mode != 0) { // If time_setup_mode is 1 or 2, it is timed
                    m_ssbTxPipes[idxSlot][i].setup(m_ssbTxDynamicApiDataSets[idxSlot][i].ssb_dyn_params);
                }
                if (time_setup_mode != 1)
                {
                    m_ssbTxPipes[idxSlot][i].run(0 /*ssb_proc_mode*/);
                }
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

            for(int i = 0; i < num_ssb_objects; i++)
            {
                cuphy_timer_single_cell[i].synchronize(); // To be safe
                total_time_single_cell[i] += cuphy_timer_single_cell[i].elapsed_time_ms();
            }

            if (ref_check_ssb && (iter == 0)) {
               for(int i = 0; i < num_ssb_objects; i++)
               {
                  int errors = m_ssbTxDynamicApiDataSets[idxSlot][i].refCheck(true);
                  if (errors != 0) {
                      exit(1);
                  }
               }
            }
            gpu_us_delay(delayUs, 0, streams[0].handle()); // 10ms delay kernel. Can update/comment out.
            CUDA_CHECK(cudaEventRecord(start_streams_event, streams[0].handle()));
        }
        total_time_slot[idxSlot] = total_time / num_iterations;

        for(int i = 0; i < num_ssb_objects; i++)
        {
            total_time_single_cell_slot[idxSlot][i] = total_time_single_cell[i] / num_iterations;
        }
    }

    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++)
    {
        NVLOGC_FMT(NVLOG_SSB, "Slot # {},  SSB pipeline(s) {} {}: {:.2f} us (avg. over {} iterations)", idxSlot, setup_modes[time_setup_mode].c_str(), proc_modes[procModeBmsk].c_str(), total_time_slot[idxSlot] * 1000, num_iterations);
        for(int i = 0; i < num_ssb_objects; i++)
        {
            if (group_cells) {
                NVLOGC_FMT(NVLOG_SSB, "--> SSB object # {} with {} cells: {:.2f} us (avg over {} iterations)", i, num_cells, total_time_single_cell_slot[idxSlot][i] * 1000, num_iterations);
            } else {
                NVLOGC_FMT(NVLOG_SSB, "--> Cell # {} : {:.2f} us (avg over {} iterations)", i, total_time_single_cell_slot[idxSlot][i] * 1000, num_iterations);
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    nvlog_fmtlog_close(log_thread_id);

    return 0;
}
