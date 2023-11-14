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
#include "csirs_tx.hpp"
#include "cuphy_channels.hpp"
#include "datasets.hpp"
#include <list>
#include <fstream>
#include "test_config.hpp"
#include "memtrace.h"

#define CSIRS_STREAM_PRIORITY 0
#define PRINT_CSIRS_CONFIG 0 // Set to 1 for debugging purposes

using namespace cuphy;

/**
 *  @brief Print usage information for the DL pipeline example.
 */
void usage()
{
    printf("  Options:\n");
    printf("    -h                          Display usage information\n");
    printf("    -i  input_filename          Input yaml file for slot/cell config or HDF5 file for single cell example\n");
    printf("    -r  # of iterations         Number of run iterations to run\n");
    printf("    -d  # of microseconds       Delay kernel duration in us\n");
    printf("    -k                          Enable reference check. Compare GPU output with test vector (first iteration only).\n");
    printf("    -m  proc_mode               Processing mode: streams(0), graphs (1)\n");
    printf("    -g                          Execute all cells in a slot on the same CSIRS object with batching too.\n");
    printf("    -s  setup_mode              0 (default) - setup is not timed; 1 - time setup only; no run is run; 2 - time both setup and run, back to back.\n");
}

int main(int argc, char* argv[])
{
    char nvlog_yaml_file[1024];
    // Relative path from binary to default nvlog_config.yaml
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
    pthread_t log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, "csirs_tx_multicell.log",NULL);
    nvlog_fmtlog_thread_init();

    //------------------------------------------------------------------
    // Parse command line arguments
    int         iArg = 1;
    std::string inputFileName;
    uint32_t    num_iterations  = 1;
    bool        ref_check_csirs = false;
    uint64_t    procModeBmsk    = 0;
    uint32_t    delayUs         = 10000;
    bool        group_cells     = false; // Group cells and if possible, also process all cells in a single kernel per component.
    int         time_setup_mode = 0;     // default mode: only time run, not setup
    std::string setup_modes[3]  = {"GPU-run only", "GPU-setup only", "GPU-setup-and-run"};
    std::string proc_modes[2]  = {"in Stream mode", "in Graphs mode"};

    CUDA_CHECK(cudaSetDevice(0));

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
                    NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided.");
                }
                inputFileName.assign(argv[iArg++]);
                break;
            case 'r':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &num_iterations)) || ((num_iterations <= 0)))
                {
                    NVLOGF_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of iterations");
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
                ref_check_csirs = true; // Note: enabled by default when input file is single HDF5 TV and not a yaml
                ++iArg;
                break;
            case 'm':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%lu", &procModeBmsk)) || ((CSIRS_PROC_MODE_STREAMS != procModeBmsk) && (CSIRS_PROC_MODE_GRAPHS != procModeBmsk)))
                {
                    NVLOGF_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT,  "ERROR: Invalid processing mode ({:#x})", procModeBmsk);
                }
                ++iArg;
                break;
            case 's':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &time_setup_mode)) || ((time_setup_mode < 0)) || ((time_setup_mode > 2)))
                {
                    NVLOGF_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT,  "ERROR: Invalid process mode");
                }
                ++iArg;
                break;
            default:
                NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                usage();
                exit(1);
                break;
            }
        }
        else
        {
            NVLOGF_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
        }
    }
    if(inputFileName.empty())
    {
        usage();
        exit(1);
    }

    std::unique_ptr<cuphy::test_config> testCfg;

    // Default value of 1 for num_cells and num_slots used when input file is HDF5 file and not a yaml.
    int num_cells = 1;
    int num_slots = 1;

    const std:: string channelName = "CSIRS";
    std::string inFileExtn = inputFileName.substr(inputFileName.find_last_of(".") + 1);
    //NVLOGC_FMT(NVLOG_CSIRS, "File extension: {}", inFileExtn.c_str());
    if(inFileExtn == "yaml")
    {
        testCfg = std::make_unique<cuphy::test_config>(inputFileName.c_str());
        testCfg->print_channel(channelName); // print only CSIRS channel related parts of the input YAML file
        //testCfg->print(); // print entire YAML file contents, incl. other channels not run with this example
        num_cells = testCfg->num_cells(); // The same number of cells is present across all slots.
        num_slots = testCfg->num_slots();
        //NVLOGC_FMT(NVLOG_CSIRS, "CSIRS multi-cell with {} cells and {} slots",  num_cells, num_slots);
    } else {
        NVLOGC_FMT(NVLOG_CSIRS, "CSI-RS TX Pipeline single-cell example with TV {} and ref. checks enabled",  inputFileName.c_str());
        ref_check_csirs = true; // override -k option in single cell case
    }

    CUDA_CHECK(cudaSetDevice(0)); // Select GPU device 0

    std::vector<std::vector<cuphy::csirs_tx>>       m_csirsTxPipes;
    std::vector<std::vector<csirsStaticApiDataset>> m_csirsTxStaticApiDataSets;
    std::vector<std::vector<csirsDynApiDataset>>    m_csirsTxDynamicApiDataSets;

    std::vector<stream>        streams;
    cudaEvent_t                start_streams_event;
    std::vector<cudaEvent_t> stop_streams_events(num_cells);
    const int num_csirs_objects = group_cells ? 1 : num_cells;

    m_csirsTxPipes.resize(num_slots);
    m_csirsTxStaticApiDataSets.resize(num_slots);
    m_csirsTxDynamicApiDataSets.resize(num_slots);

    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++) {
        if (testCfg) {
            int cells_in_slot = testCfg->slots()[idxSlot].at(channelName).size();
            if (cells_in_slot != num_cells) {
                NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Slot {} error: expected {} cells but got {}", idxSlot, num_cells, cells_in_slot);
                exit(1);
            }
        }
        // Reminder num_csirs_object is the number of cells, if group_cells is false, or 1 otherwise
        m_csirsTxStaticApiDataSets[idxSlot].reserve(num_csirs_objects);
        m_csirsTxDynamicApiDataSets[idxSlot].reserve(num_csirs_objects);

        // Loop over all cells (not num_csirs_objects) to populate the static and dynamic datasets
        for(int i = 0; i < num_cells; i += 1)
        {
            std::string tv_filename = (testCfg) ? testCfg->slots()[idxSlot].at(channelName)[i] : inputFileName;
            if((idxSlot == 0) && (i < num_csirs_objects)) {
                streams.emplace_back(cudaStreamNonBlocking, CSIRS_STREAM_PRIORITY);
            }
            if (group_cells) {
                if (i == 0) {
                    m_csirsTxStaticApiDataSets[idxSlot].emplace_back(tv_filename, num_cells);
                    m_csirsTxDynamicApiDataSets[idxSlot].emplace_back(tv_filename, m_csirsTxStaticApiDataSets[idxSlot][i].csirsStatPrms.nMaxCellsPerSlot, streams[i].handle(), procModeBmsk);
                    m_csirsTxPipes[idxSlot].emplace_back(m_csirsTxStaticApiDataSets[idxSlot][i].csirsStatPrms);
                } else {
                    // Update the static parameters
                    m_csirsTxStaticApiDataSets[idxSlot][0].cumulativeUpdate(tv_filename);
                    // Update the dynamic parameters
                    m_csirsTxDynamicApiDataSets[idxSlot][0].cumulativeUpdate(tv_filename, streams[0].handle());
                }
            } else {
                m_csirsTxStaticApiDataSets[idxSlot].emplace_back(tv_filename);
                m_csirsTxDynamicApiDataSets[idxSlot].emplace_back(tv_filename, m_csirsTxStaticApiDataSets[idxSlot][i].csirsStatPrms.nMaxCellsPerSlot, streams[i].handle(), procModeBmsk);
                m_csirsTxPipes[idxSlot].emplace_back(m_csirsTxStaticApiDataSets[idxSlot][i].csirsStatPrms);
            }
        }
    }

#if PRINT_CSIRS_CONFIG
    // Print dynamic dataset params for each CSIRS object. For dbg. purposes
    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++) {
        for(int i = 0; i < num_csirs_objects; i += 1) {
            static_cast<CsirsTx*>(m_csirsTxPipes[idxSlot][i].handle())->printCsirsConfig(m_csirsTxDynamicApiDataSets[idxSlot][i].csirs_dyn_params);
        }
    }
#endif

    NVLOGC_FMT(NVLOG_CSIRS, "Timing {} CSIRS pipeline(s). ", num_cells);
    if (group_cells)
    {
        NVLOGC_FMT(NVLOG_CSIRS, "Grouping all cells in a slot.");
    } else {
        NVLOGC_FMT(NVLOG_CSIRS, "");
    }
    if (time_setup_mode== 0) {
        NVLOGC_FMT(NVLOG_CSIRS, "- NB: Allocations, setup processing not included.");
        NVLOGC_FMT(NVLOG_CSIRS, "");
    }

    /* CSIRS Tx Run for all num_cells pipelines will be timed on streams[0].handle() CUDA stream (note, that is NOT stream 0).
    Have that stream wait for all other streams to complete their work too. */

    for(int i = 0; i < num_csirs_objects; i++)
    {
        CUDA_CHECK(cudaEventCreateWithFlags(&stop_streams_events[i], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&start_streams_event, cudaEventDisableTiming));

    float total_time_slot[num_slots]; // Total time of a slot divided by number of iterations (num_iterations)
    float total_time_single_cell_slot[num_slots][num_csirs_objects];
    //NVLOGC_FMT(NVLOG_CSIRS, "num slots {}, num_csirs_objects {}", num_slots, num_csirs_objects);

    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++)
    {
        float                    total_time = 0;
        std::vector<float>       total_time_single_cell(num_csirs_objects, 0);
        std::vector<event_timer> cuphy_timer_single_cell(num_csirs_objects);

        if (time_setup_mode == 0){ // In this mode, setup is not timed
            for(int i = 0; i < num_csirs_objects; i++)
            {
                m_csirsTxPipes[idxSlot][i].setup(m_csirsTxDynamicApiDataSets[idxSlot][i].csirs_dyn_params);
            }
        }
#if 0
        if (idxSlot == 0) { // why warmup?
            m_csirsTxPipes[idxSlot][0].setup(m_csirsTxDynamicApiDataSets[idxSlot][0].csirs_dyn_params);
            m_csirsTxPipes[idxSlot][0].run();
        }
#endif
        if (testCfg) {
            gpu_us_delay(delayUs, 0, streams[0].handle());
        }
        CUDA_CHECK(cudaEventRecord(start_streams_event, streams[0].handle()));

        // Enable dynamic memory allocation tracing in real-time code path; only applicable when running with LD_PRELOAD=<PATH to libmimalloc.so>
        memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);

        for(int iter = 0; iter < num_iterations; iter++)
        {
            event_timer cuphy_timer;
            cuphy_timer.record_begin(streams[0].handle());

            for(int i = 0; i < num_csirs_objects; i++)
            {
                cudaStream_t strm_handle = streams[i].handle();

                if(i != 0)
                {
                    CUDA_CHECK(cudaStreamWaitEvent(streams[i].handle(), start_streams_event, 0));
                }

                cuphy_timer_single_cell[i].record_begin(streams[i].handle());
                if (time_setup_mode != 0) { // If time_setup_mode is 1 or 2, it is timed
                    m_csirsTxPipes[idxSlot][i].setup(m_csirsTxDynamicApiDataSets[idxSlot][i].csirs_dyn_params);
                }
                if (time_setup_mode != 1)
                {
                    m_csirsTxPipes[idxSlot][i].run();
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

            for(int i = 0; i < num_csirs_objects; i++)
            {
                cuphy_timer_single_cell[i].synchronize(); // To be safe
                total_time_single_cell[i] += cuphy_timer_single_cell[i].elapsed_time_ms();
            }

            if (ref_check_csirs && (iter == 0)) {
               for(int i = 0; i < num_csirs_objects; i++)
               {
                  int errors = m_csirsTxDynamicApiDataSets[idxSlot][i].refCheck(true);
                  if (errors != 0) {
                      exit(1);
                  }
               }
            }
            if (testCfg) {
                gpu_us_delay(delayUs, 0, streams[0].handle()); // 10ms delay kernel. Can update/comment out.
            }
            CUDA_CHECK(cudaEventRecord(start_streams_event, streams[0].handle()));
        }
        total_time_slot[idxSlot] = total_time / num_iterations;
        memtrace_set_config(0); // disable memory allocation tracing beyond this point

        for(int i = 0; i < num_csirs_objects; i++)
        {
            total_time_single_cell_slot[idxSlot][i] = total_time_single_cell[i] / num_iterations;
        }
    }

    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++)
    {
        NVLOGC_FMT(NVLOG_CSIRS, "Slot # {}, CSIRS pipeline(s) {} {}: {:.2f} us (avg. over {} iterations).", idxSlot, setup_modes[time_setup_mode].c_str(), proc_modes[procModeBmsk].c_str(), total_time_slot[idxSlot] * 1000, num_iterations);
        for(int i = 0; i < num_csirs_objects; i++)
        {
            if (group_cells) {
                NVLOGC_FMT(NVLOG_CSIRS, "--> CSIRS object # {} with {} cells: {:.2f} us (avg over {} iterations)", i, num_cells, total_time_single_cell_slot[idxSlot][i] * 1000, num_iterations);
            } else {
                NVLOGC_FMT(NVLOG_CSIRS, "--> Cell # {} : {:.2f} us (avg over {} iterations)", i, total_time_single_cell_slot[idxSlot][i] * 1000, num_iterations);
            }
        }
    }

    CUDA_CHECK(cudaDeviceSynchronize());
    nvlog_fmtlog_close(log_thread_id);

    return 0;
}
