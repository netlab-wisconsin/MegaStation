/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "util.hpp"
#include "pdsch_tx.hpp"
#include <list>
#include <fstream>
#include "test_config.hpp"

//#define _READ_TB_CRC_ 1

#define PDSCH_TB_INPUT_ON_GPU 0 //FIXME set to 1 if you want to test the TB buffers in GPU mode for cuPHY standalone

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
    printf("    -k                          Enable reference check. Compare GPU output with test vector.\n");
    printf("    -m  process mode            streams(0), graphs (1).\n");
    printf("    -g                          Execute all cells in a slot on the same PDSCH object.\n"); // Reminder: they should all have identical static parameters.
    printf("    -s  setup_mode              0 (default) - setup is not timed; 1 - time setup only; no run is run; 2 - time both setup and run, back to back.\n");
    printf("    -c  cpu_id                  cpu_id used for CPU affinity setting.\n");
    printf("    -p  priority                Thread priority.\n");
}

int main(int argc, char* argv[])
{
    int returnValue = 0;

    char nvlog_yaml_file[1024];
    // Relative path from binary to default nvlog_config.yaml
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
    pthread_t log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, "pdsch_tx_multicell.log",NULL);
    nvlog_fmtlog_thread_init();

    //------------------------------------------------------------------
    // Parse command line arguments
    int         iArg = 1;
    std::string inputFileName;
    uint32_t    num_iterations      = 1;
    bool        ref_check_pdsch     = false;
    int         cfg_process_mode    = 0;
    int         cfg_priority        = 0;
    int         cfg_cpu_id          = -1;
    uint32_t    delayUs             = 10000;
    bool        group_cells            = false; // Group cells in the same cell-group and if possible process them all in a single kernel per component.
    int         time_setup_mode        = 0; // default mode: only time run, not setup
    std::string setup_modes[3] = {"GPU-run only", "GPU-setup only", "GPU-setup-and-run"};

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
                    NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided.");
                }
                inputFileName.assign(argv[iArg++]);
                break;
            case 'r':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &num_iterations)) || ((num_iterations <= 0)))
                {
                    NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of iterations");
                }
                ++iArg;
                break;
            case 'd':
                    delayUs = std::stoi(argv[++iArg]);
                    ++iArg;
                    break;
            case 'g':
                group_cells = true; // Group all cells in a slot and run them on a single PdschTx channel.
                ++iArg;
                break;
            case 'k':
                ref_check_pdsch = true;
                ++iArg;
                break;
            case 's':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &time_setup_mode)) || ((time_setup_mode < 0)) || ((time_setup_mode > 2)))
                {
                    NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid process mode");
                }
                ++iArg;
                break;
            case 'm':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &cfg_process_mode)) || ((cfg_process_mode < 0)) || ((cfg_process_mode > 1)))
                {
                    NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid process mode");
                }
                ++iArg;
                break;
            case 'c':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &cfg_cpu_id)) || ((cfg_cpu_id < 0)))
                {
                    NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid cpu_id");
                }
                ++iArg;
                break;
            case 'p':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &cfg_priority)) || ((cfg_priority <= 0)) || ((cfg_priority >99)))
                {
                    NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid thread priority");
                }
                ++iArg;
                break;
            default:
                NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                usage();
                exit(1);
                break;
            }
        }
        else
        {
            NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
        }
    }
    if(inputFileName.empty())
    {
        usage();
        exit(1);
    }

    cuphy::test_config testCfg(inputFileName.c_str());
    testCfg.print();
    int num_cells = testCfg.num_cells(); // The same number of cells is present across all slots.
    int num_slots = testCfg.num_slots();
    const std:: string channelName = "PDSCH";

    std::vector<std::string>  hdf5_filenames;
    std::ifstream tvs_list_file;

    int enable_homogeneous_mode = 0;
    int cfg_aas_mode = 0;
    bool ref_check = false;
    bool aas_mode = (cfg_aas_mode >= 1);
    bool graphs_mode = (cfg_process_mode >= 1);
    std::string pipeline_mode = (aas_mode) ? "AAS" : "non AAS";
    std::string graphs_streams_mode_string = (graphs_mode) ? "Graphs" : "Streams";
    cuphyPdschProcMode_t tmp_pdsch_proc_mode = (graphs_mode) ? PDSCH_PROC_MODE_GRAPHS : PDSCH_PROC_MODE_NO_GRAPHS;
    uint64_t pdsch_proc_mode = tmp_pdsch_proc_mode;
    pdsch_proc_mode |= PDSCH_INTER_CELL_BATCHING; // not needed; inter cell batching is applied regardless of this flag

    CUDA_CHECK(cudaSetDevice(0)); // Select GPU device 0
    CUmoduleLoadingMode mode;
    CUresult status = cuModuleGetLoadingMode(&mode);
    if (status != CUDA_SUCCESS) NVLOGC_FMT(NVLOG_PDSCH, "cuModuleGetLoading returned {}", status);
    NVLOGC_FMT(NVLOG_PDSCH, "mode {} (reminder EAGER_LOADING is {} while lazy is {})", mode, CU_MODULE_EAGER_LOADING, CU_MODULE_LAZY_LOADING);

    // The Downlink pipeline includes: (a) CRC, (b) LDPC  encoder, (c) Rate-Matching,
    // (d) Modulation Mapper and (e) DMRS components.

    std::vector<stream>        streams;
    std::vector<tensor_device> data_tx_tensor(num_cells); // Even if groups_cells is set, num_cells output tensors will exist.

    // Large buffer added as cuPHY-CP needed a buffer with a power of 2 size.  Buffer below is not.
    size_t                               large_buffer_elements = MAX_N_PRBS_SUPPORTED * CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT * MAX_DL_PORTS; // Per cell
    size_t                               large_buffer_bytes    = large_buffer_elements * sizeof(__half2); // Per cell
    unique_device_ptr<__half2> large_buffer                    = make_unique_device<__half2>(num_cells * large_buffer_elements);

    // Reset the entire output buffer.
    CUDA_CHECK(cudaMemset(large_buffer.get(), 0, num_cells * large_buffer_bytes)); // This is done in the default stream

    const int num_pdsch_objects = group_cells ? 1 : num_cells;

    cudaEvent_t              start_streams_event;
    std::vector<cudaEvent_t> stop_streams_events(num_cells);

    std::string                         first_filename = testCfg.slots()[0].at(channelName)[0];
    std::unique_ptr<hdf5hpp::hdf5_file> tmp_first_file = std::make_unique<hdf5hpp::hdf5_file>(hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::open(first_filename.c_str())));

    bool use_new_api = tmp_first_file->is_valid_dataset("ue_pars");
    if(!use_new_api)
    {
        NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "PDSCH examples no longer support old API TVs.");
    }
    tmp_first_file.reset();


    std::unique_ptr<cuphyPdschTxHndl_t[]>                pdsch_handle = std::make_unique<cuphyPdschTxHndl_t[]>(num_pdsch_objects);
    std::vector<cuphyPdschStatPrms_t>                    pdsch_static_params(num_slots * num_pdsch_objects); // Full vector only used in ref_check. Otherwise, only the first num_pdsch_objects elements are used.
    std::vector<cuphyTracker_t>                          pdsch_trackers(num_slots * num_pdsch_objects);
    typedef std::vector<cuphyPdschCellGrpDynPrm_t>       dyn_params_vector_t;
    dyn_params_vector_t                                  cell_dyn_params(1);
    std::vector<dyn_params_vector_t>                     pdsch_cell_grp_dyn_params(num_slots * num_cells, cell_dyn_params); // Limited to one cell group per pipeline
    std::vector<std::unique_ptr<hdf5hpp::hdf5_file>>     input_file(num_slots * num_cells); // Even under group_cells, we need to read all TVs and stitch them together.
#if PDSCH_TB_INPUT_ON_GPU
    std::vector<typed_tensor<CUPHY_R_8U, device_alloc>>  crc_input_data;
#else
    std::vector<typed_tensor<CUPHY_R_8U, pinned_alloc>>  crc_input_data;
#endif
    std::vector<typed_tensor<CUPHY_R_32U, pinned_alloc>> tb_crc_input_data;

    // When group_cells is set, every num_cells elements for each of data_in, tb_crc_data_in and output_data will be processed by a single pipeline.
    std::vector<cuphyPdschDataIn_t>  data_in(num_slots * num_cells);
    std::vector<cuphyPdschDataIn_t>  tb_crc_data_in(num_slots * num_cells);
    std::vector<cuphyPdschDataOut_t> output_data(num_slots * num_cells);
    std::vector<cuphyPdschStatusOut_t> output_status(num_slots * num_cells);

    std::vector<cuphyPdschDynPrms_t> pdsch_dyn_params(num_slots * num_cells);
    //NVLOGC_FMT(NVLOG_PDSCH, "num_slots {}, num_cells {}", num_slots, num_cells);

    auto data_in_ptr = std::make_unique<uint8_t*[]>(num_slots * num_cells);
    auto tb_crc_data_in_ptr = std::make_unique<uint8_t*[]>(num_slots * num_cells);

    if (group_cells) {

        for(int idxSlot = 0; idxSlot < num_slots; idxSlot++)
        {
            for(int i = 0; i < num_cells; i += 1) {
                if(idxSlot == 0)
                {
                    if (i < num_pdsch_objects) {
                        streams.emplace_back(cudaStreamNonBlocking, PDSCH_STREAM_PRIORITY);
                    }
                    //Even though we're grouping all cells in a slot, each cell will still have its own output tensor buffer.
                    data_tx_tensor[i] = tensor_device((void*)((__half2*)large_buffer.get() + i * large_buffer_elements),
                                                               CUPHY_C_16F,
                                                               CUPHY_N_TONES_PER_PRB * 273,
                                                               OFDM_SYMBOLS_PER_SLOT, MAX_DL_LAYERS,
                                                               cuphy::tensor_flags::align_tight); //FIXME For now assuming 273PRBs (max possible)

                    int data_tx_tensor_bytes = data_tx_tensor[i].desc().get_size_in_bytes();
                    if(data_tx_tensor_bytes > large_buffer_bytes)
                    {
                        NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Buffer ({} bytes) is smaller than data_tx_tensor ({})", large_buffer_bytes, data_tx_tensor_bytes);
                    }
                }
                std::string tv_filename = testCfg.slots()[idxSlot].at(channelName)[i];
                hdf5_filenames.emplace_back(tv_filename);
                // NVLOGC_FMT(NVLOG_PDSCH, "PdschTx Pipeline {} uses TV {}", i, hdf5_filenames[idxSlot*num_cells+i]);

                input_file[idxSlot * num_cells + i] = std::make_unique<hdf5hpp::hdf5_file>(hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::open(hdf5_filenames[idxSlot * num_cells + i].c_str())));

                //Read static and dynamic parameters per cell

                // Accumulate the static parameters across all cells in each slot idxSlot.
                // Only the first pdsch_static_params element will be used in the timing runs below. These parameters should be valid for all TVs in subsequent slots.
                // All pdsch_static_params elements will be used for the reference check before run.
                cumulative_read_pdsch_static_pars_from_file(pdsch_static_params[idxSlot],
                                                             *input_file[idxSlot * num_cells + i],
                                                             hdf5_filenames[idxSlot * num_cells + i].c_str(),
                                                             ref_check,
                                                             (i == 0) /* allocate memory for first cell in each slot based on some predefined max values */);
                pdsch_static_params[idxSlot].pOutInfo = &pdsch_trackers[idxSlot];

                //FIXME Would it help to have an std::map that maps static physical cell Ids to static cell index?
                //FIXME also some of the dbg params read above (e.g., ref-checks, ldpc nodes) should be the same across all cells

                // Accumulate the cell group dynamic parameters across all cells in each slot idxSlot.
                cumulative_read_cell_group_dynamic_pars_from_file(pdsch_cell_grp_dyn_params[idxSlot],
                                                                   *input_file[idxSlot * num_cells + i],
                                                                   (i == 0) /* allocate memory for first cell in each slot based on some predefined max values */);

                //Parse input dataset
                hdf5hpp::hdf5_dataset crc_dataset = (*input_file[idxSlot * num_cells + i]).open_dataset("InputData");
#if PDSCH_TB_INPUT_ON_GPU
                crc_input_data.emplace_back(typed_tensor_from_dataset<CUPHY_R_8U, device_alloc>(crc_dataset, cuphy::tensor_flags::align_default, streams[0].handle()));
#else
                crc_input_data.emplace_back(typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(crc_dataset, cuphy::tensor_flags::align_default, streams[0].handle()));
#endif
#ifdef _READ_TB_CRC_
                hdf5hpp::hdf5_dataset tb_crc_dataset = (*input_file[idxSlot * num_cells + i]).open_dataset("tbCrcBuffer");
                tb_crc_input_data.emplace_back(typed_tensor_from_dataset<CUPHY_R_32U, pinned_alloc>(tb_crc_dataset, cuphy::tensor_flags::align_default, streams[0].handle()));
#endif

                data_in_ptr[idxSlot * num_cells + i] = crc_input_data[idxSlot * num_cells + i].addr();
#if PDSCH_TB_INPUT_ON_GPU
                data_in[idxSlot * num_cells + i] = {&data_in_ptr[idxSlot * num_cells + i], cuphyPdschDataIn_t::GPU_BUFFER};
#else
                data_in[idxSlot * num_cells + i] = {&data_in_ptr[idxSlot * num_cells + i], cuphyPdschDataIn_t::CPU_BUFFER};
#endif
                if (i == 0) pdsch_static_params[idxSlot].full_slot_processing = (!aas_mode);
#ifdef _READ_TB_CRC_
                tb_crc_data_in_ptr[idxSlot * num_cells + i] = (uint8_t*)tb_crc_input_data[idxSlot * num_cells + i].addr();
                tb_crc_data_in[idxSlot * num_cells + i] = {&tb_crc_data_in_ptr[idxSlot * num_cells + i], cuphyPdschDataIn_t::CPU_BUFFER};
                if (i == 0) pdsch_static_params[idxSlot].read_TB_CRC = true;
#else
                tb_crc_data_in[idxSlot * num_cells + i] = {nullptr, cuphyPdschDataIn_t::CPU_BUFFER};
                if (i == 0) pdsch_static_params[idxSlot].read_TB_CRC = false;
#endif

                input_file[idxSlot * num_cells + i].reset(); // Close the file; will reopen it later

                if (i == 0) {
                    output_data[idxSlot] = {new cuphyTensorPrm_t[num_cells]};
                    pdsch_dyn_params[idxSlot] = {streams[0].handle(),
                                                 pdsch_proc_mode,
                                                 pdsch_cell_grp_dyn_params[idxSlot].data(),
                                                 &data_in[idxSlot * num_cells + i] /* pointer to contiguous num_cells elements for data input */,
                                                 &tb_crc_data_in[idxSlot * num_cells + i] /* pointer to contiguous num_cells elements for PDSCH CRC input */,
                                                 &output_data[idxSlot] /* pointer to an array of num_cells output tensors */,
                                                 &output_status[idxSlot] /* pointer to cell group status */};
                }
                pdsch_dyn_params[idxSlot].pDataOut->pTDataTx[i].desc  = data_tx_tensor[i].desc().handle();
                pdsch_dyn_params[idxSlot].pDataOut->pTDataTx[i].pAddr = data_tx_tensor[i].addr();
            }
            // Print stiched together static and dynamic parameters for each slot
            //cuphy::print_pdsch_static(&pdsch_static_params[idxSlot]);
            //cuphy::print_pdsch_dynamic_cell_group(&pdsch_cell_grp_dyn_params[idxSlot][0]);
            //cuphy::print_pdsch_dynamic(&pdsch_dyn_params[idxSlot]);
            }
        }
        else
        {
            //Not grouping any cells. Could consolidate with if clause later

            for(int idxSlot = 0; idxSlot < num_slots; idxSlot++)
            {
                for(int i = 0; i < num_cells; i += 1)
                {
                    std::string tv_filename = testCfg.slots()[idxSlot].at(channelName)[i];
                    hdf5_filenames.emplace_back(tv_filename);
                    //NVLOGC_FMT(NVLOG_PDSCH, "PdschTx Pipeline {} uses TV {}", i, hdf5_filenames[idxSlot*num_cells+i]);
                    input_file[idxSlot * num_cells + i] = std::make_unique<hdf5hpp::hdf5_file>(hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::open(hdf5_filenames[idxSlot * num_cells + i].c_str())));

                    if(idxSlot == 0)
                    {
                        streams.emplace_back(cudaStreamNonBlocking, PDSCH_STREAM_PRIORITY);
                        int num_REs = cuphy::get_HDF5_dataset_info((*input_file[idxSlot * num_cells + i]).open_dataset("Xtf")).layout().dimensions()[0];
                        data_tx_tensor[i] = tensor_device((void*)((__half2*)large_buffer.get() + i * large_buffer_elements),
                                                                  CUPHY_C_16F, num_REs, OFDM_SYMBOLS_PER_SLOT, MAX_DL_LAYERS,
                                                                  cuphy::tensor_flags::align_tight);

                        int data_tx_tensor_bytes = data_tx_tensor[i].desc().get_size_in_bytes();
                        if(data_tx_tensor_bytes > large_buffer_bytes)
                        {
                            NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Buffer ({} bytes) is smaller than data_tx_tensor ({})", large_buffer_bytes, data_tx_tensor_bytes);
                        }
                    }

                    //Read static and dynamic parameters per cell
                    read_pdsch_static_pars_from_file(pdsch_static_params[idxSlot * num_cells + i], *input_file[idxSlot * num_cells + i],
                                                     hdf5_filenames[idxSlot * num_cells + i].c_str(), ref_check, true);
                    read_cell_group_dynamic_pars_from_file(pdsch_cell_grp_dyn_params[idxSlot * num_cells + i], *input_file[idxSlot * num_cells + i]);

                    pdsch_static_params[idxSlot*num_cells + i].pOutInfo = &pdsch_trackers[idxSlot*num_cells + i];
                    int cfg_num_cells = pdsch_cell_grp_dyn_params[idxSlot * num_cells + i][0].nCells;
                    if(cfg_num_cells != 1)
                    {
                        NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Error! # cells = {}, but this example only supports a single cell per pipeline", num_cells);
                    }

                    //Parse input dataset
                    hdf5hpp::hdf5_dataset crc_dataset = (*input_file[idxSlot * num_cells + i]).open_dataset("InputData");
#if PDSCH_TB_INPUT_ON_GPU
                    crc_input_data.emplace_back(typed_tensor_from_dataset<CUPHY_R_8U, device_alloc>(crc_dataset, cuphy::tensor_flags::align_default, streams[i].handle()));
#else
                    crc_input_data.emplace_back(typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(crc_dataset, cuphy::tensor_flags::align_default, streams[i].handle()));
#endif
#ifdef _READ_TB_CRC_
                    hdf5hpp::hdf5_dataset tb_crc_dataset = (*input_file[idxSlot * num_cells + i]).open_dataset("tbCrcBuffer");
                    tb_crc_input_data.emplace_back(typed_tensor_from_dataset<CUPHY_R_32U, pinned_alloc>(tb_crc_dataset, cuphy::tensor_flags::align_default, streams[i].handle()));
#endif

                    data_in_ptr[idxSlot * num_cells + i] = crc_input_data[idxSlot * num_cells + i].addr();
#if PDSCH_TB_INPUT_ON_GPU
                    data_in[idxSlot * num_cells + i] = {&data_in_ptr[idxSlot * num_cells + i], cuphyPdschDataIn_t::GPU_BUFFER};
#else
                    data_in[idxSlot * num_cells + i] = {&data_in_ptr[idxSlot * num_cells + i], cuphyPdschDataIn_t::CPU_BUFFER};
#endif
                    pdsch_static_params[idxSlot * num_cells + i].full_slot_processing = (!aas_mode);
#ifdef _READ_TB_CRC_
                    tb_crc_data_in_ptr[idxSlot * num_cells + i] = (uint8_t*)tb_crc_input_data[idxSlot * num_cells + i].addr();
                    tb_crc_data_in[idxSlot * num_cells + i] = {&tb_crc_data_in_ptr[idxSlot * num_cells + i], cuphyPdschDataIn_t::CPU_BUFFER};
                    pdsch_static_params[idxSlot * num_cells + i].read_TB_CRC = true;
#else
                    tb_crc_data_in[idxSlot * num_cells + i] = {nullptr, cuphyPdschDataIn_t::CPU_BUFFER};
                    pdsch_static_params[idxSlot * num_cells + i].read_TB_CRC = false;
#endif

                    input_file[idxSlot * num_cells + i].reset();                      // Close the file; will reopen it later
                    output_data[idxSlot * num_cells + i] = {new cuphyTensorPrm_t[1]}; //single cell per pipeline for now

                    pdsch_dyn_params[idxSlot * num_cells + i]                             = {streams[i].handle(), pdsch_proc_mode, pdsch_cell_grp_dyn_params[idxSlot * num_cells + i].data(), &data_in[idxSlot * num_cells + i], &tb_crc_data_in[idxSlot * num_cells + i], &output_data[idxSlot * num_cells + i], &output_status[idxSlot * num_cells + i]};
                    pdsch_dyn_params[idxSlot * num_cells + i].pDataOut->pTDataTx[0].desc  = data_tx_tensor[i].desc().handle();
                    pdsch_dyn_params[idxSlot * num_cells + i].pDataOut->pTDataTx[0].pAddr = data_tx_tensor[i].addr();

                    //cuphy::print_pdsch_static(&pdsch_static_params[idxSlot*num_cells + i]);
                    //cuphy::print_pdsch_dynamic_cell_group(&pdsch_cell_grp_dyn_params[idxSlot * num_cells + i][0]);
                    //cuphy::print_pdsch_dynamic(&pdsch_dyn_params[idxSlot*num_cells + i]);
                }
        }
    }


    if(ref_check_pdsch)
    {
        // Run all PDSCH pipeline(s) once w/ ref checks enabled
        NVLOGC_FMT(NVLOG_PDSCH, "");
        NVLOGC_FMT(NVLOG_PDSCH, "Running all {} DL pipelines once w/ reference checks enabled in {} and {} mode.", num_cells, pipeline_mode, graphs_streams_mode_string);

        for(int idxSlot = 0; idxSlot < num_slots; idxSlot++)
        {
            for(int i = 0; i < num_pdsch_objects; i++)
            {
                int grouped_cell_index = (group_cells) ? idxSlot : idxSlot * num_cells + i;

                if (group_cells) {
                  NVLOGC_FMT(NVLOG_PDSCH, "--> idxSlot = {}, idx PDSCH object = {}", idxSlot, i);
                } else {
                  NVLOGC_FMT(NVLOG_PDSCH, "--> idxSlot = {}, idxCell = {}", idxSlot, i);
                }

                /*NVLOGC_FMT(NVLOG_PDSCH, "Pdsch Object {} has {} static cells:", i, pdsch_static_params[grouped_cell_index].nCells);
                for (int k = 0; k <  pdsch_static_params[grouped_cell_index].nCells; k++)
                {
                    NVLOGC_FMT(NVLOG_PDSCH, " - static cell {} with TV name {}", k, pdsch_static_params[grouped_cell_index].pDbg[k].pCfgFileName);
                }*/

                cuphyStatus_t status = cuphyCreatePdschTx(&pdsch_handle[i], &pdsch_static_params[grouped_cell_index]);
                if(status != CUPHY_STATUS_SUCCESS)
                {
                    NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Pipeline {}: Error! cuphyCreatePdschTx(): {}", i, cuphyGetErrorString(status));
                }

                pdsch_dyn_params[grouped_cell_index] = {streams[i].handle(),
                                                        pdsch_proc_mode,
                                                        pdsch_cell_grp_dyn_params[grouped_cell_index].data(),
                                                        &data_in[idxSlot * num_cells + i],
                                                        &tb_crc_data_in[idxSlot * num_cells + i],
                                                        &output_data[grouped_cell_index],
                                                        &output_status[grouped_cell_index]};

                int tensor_index = (group_cells) ? i : 0;
                pdsch_dyn_params[grouped_cell_index].pDataOut->pTDataTx[tensor_index].desc  = data_tx_tensor[i].desc().handle();
                pdsch_dyn_params[grouped_cell_index].pDataOut->pTDataTx[tensor_index].pAddr = data_tx_tensor[i].addr();

                if (group_cells) {
                    updateRefCheckMultipleCells(pdsch_handle[i], true);
                } else {
                    updateRefCheck(pdsch_handle[i], true);
                }

                status = cuphySetupPdschTx(pdsch_handle[i], &pdsch_dyn_params[grouped_cell_index], nullptr);
                if(status != CUPHY_STATUS_SUCCESS)
                {
                    if (pdsch_dyn_params[grouped_cell_index].pStatusInfo->status == cuphyPdschStatusType_t::CUPHY_PDSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB) {
                        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Pipeline {}: CUPHY_PDSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB error in cuphySetupPdschTx(): {}. Triggered by TB {} in cell group and cellPrmStatIdx {}",
                                      i, cuphyGetErrorString(status), pdsch_dyn_params[grouped_cell_index].pStatusInfo->ueIdx, pdsch_dyn_params[grouped_cell_index].pStatusInfo->cellPrmStatIdx);
                    } else {
                        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Pipeline {}: Error! cuphySetupPdschTx(): {}", i, cuphyGetErrorString(status));
                    }
                    exit(1);
                }

                status = cuphyRunPdschTx(pdsch_handle[i], pdsch_proc_mode);
                if(status != CUPHY_STATUS_SUCCESS)
                {
                    NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Pipeline {}: Error! cuphyRunPdschTx(): {}", i, cuphyGetErrorString(status));
                }

                streams[i].synchronize();

                status = cuphyDestroyPdschTx(pdsch_handle[i]);
                if(status != CUPHY_STATUS_SUCCESS)
                {
                    NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Pipeline {}: Error! cuphyDestroyPdschTx(): {}", i, cuphyGetErrorString(status));
                }

            } // end of work for one slot
            CUDA_CHECK(cudaMemset(large_buffer.get(), 0, num_cells * large_buffer_bytes)); // This is done in the default stream
        }
    }
    if (cfg_cpu_id >=0)
    {
        cpu_set_t cpuset;
        CPU_ZERO(&cpuset);
        CPU_SET(cfg_cpu_id, &cpuset);
        int ret = pthread_setaffinity_np(pthread_self(), sizeof(cpu_set_t), &cpuset);
        if (ret)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "pthread_setaffinity_np error: {}", ret);
	    return -1;
        }
    }
    if (cfg_priority > 0)
    {
        struct sched_param params;
        params.__sched_priority = cfg_priority;
        int ret = pthread_setschedparam(pthread_self(), SCHED_FIFO, &params);
        if (ret != 0)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "pthread_setschedparam error: {}", ret);
            return -1;
        }
    }

    // There will either be 1 or num_cells PDSCH channel objects depending on if group_cells is set or not.
    for(int i = 0; i < num_pdsch_objects; i++)
    {
        cuphyStatus_t status = cuphyCreatePdschTx(&pdsch_handle[i], &pdsch_static_params[i]);
        if(status != CUPHY_STATUS_SUCCESS)
        {
            NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Pipeline {}: Error! cuphyCreatePdschTx(): {}", i, cuphyGetErrorString(status));
        }
        // Ref_check here has to be false (a) because no buffers are reset and (b)
        // because the static parameters which contain filename are not update on every slot, so only the ones from the first slot are used
        // See commented out printfs below for (b).
        if (group_cells) {
            updateRefCheckMultipleCells(pdsch_handle[i], ref_check);
        } else {
            updateRefCheck(pdsch_handle[i], ref_check);
        }
#if 0
        NVLOGC_FMT(NVLOG_PDSCH, "Pdsch Object {} has {} static cells:", i, pdsch_static_params[i].nCells);
        for (int k = 0; k <  pdsch_static_params[i].nCells; k++)
        {
            NVLOGC_FMT(NVLOG_PDSCH, " - static cell {} with TV name {}", k, pdsch_static_params[i].pDbg[k].pCfgFileName);
        }
#endif
    }

    // gpu_ms_delay(10, 0, streams[0].handle()); // 10ms delay kernel. Can update/comment out.
    // CUDA_CHECK(cudaEventRecord(start_streams_event, streams[0].handle()));

    /* Time PDSCH pipeline(s) Run().
        NB: The following are not timed: Reading the HDF5 file, allocations, memory transfers.
        Also, the timing code does not include any reference checks as these will currently
        fail (some buffers need to be reset across iterations).
    */
    NVLOGC_FMT(NVLOG_PDSCH, "");
    NVLOGC_FMT(NVLOG_PDSCH, "Timing {} PDSCH DL pipeline(s) Run() w/o reference checks in {} and {} mode.", num_cells, pipeline_mode, graphs_streams_mode_string);
    if (group_cells)
    {
        NVLOGC_FMT(NVLOG_PDSCH, "Grouping all cells in a slot.");
    }
    NVLOGC_FMT(NVLOG_PDSCH, "- NB: Allocations not included. Ref. checks will fail!");
    NVLOGC_FMT(NVLOG_PDSCH, "");

    /* PdschTx::Run for all num_cells pipelines will be timed on streams[0].handle() CUDA stream (note, that is NOT stream 0).
    Have that stream wait for all other streams to complete their work too. */

    for(int i = 0; i < num_pdsch_objects; i++)
    {
        CUDA_CHECK(cudaEventCreateWithFlags(&stop_streams_events[i], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&start_streams_event, cudaEventDisableTiming));

    float total_time_slot[num_slots];
    float total_time_single_cell_slot[num_slots][num_pdsch_objects];

    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++)
    {
        float                    total_time = 0;
        std::vector<float>       total_time_single_cell(num_pdsch_objects, 0);
        std::vector<event_timer> cuphy_timer_single_cell(num_pdsch_objects);

        for(int i = 0; i < num_pdsch_objects; i++)
        {

            int grouped_cell_index = (group_cells) ? idxSlot : idxSlot * num_cells + i;
            pdsch_dyn_params[grouped_cell_index]  = {streams[i].handle(),
                                                     pdsch_proc_mode,
                                                     pdsch_cell_grp_dyn_params[grouped_cell_index].data(),
                                                     &data_in[idxSlot * num_cells + i],
                                                     &tb_crc_data_in[idxSlot * num_cells + i],
                                                     &output_data[grouped_cell_index],
                                                     &output_status[grouped_cell_index]};
            /*NVLOGC_FMT(NVLOG_PDSCH, "idxSlot {}, PDSCH object {} processing {} cells, {} UE groups, {} UEs, {} CWs and {} CSI-RS.",
                   idxSlot, i, pdsch_dyn_params[grouped_cell_index].pCellGrpDynPrm->nCells,
                   pdsch_dyn_params[grouped_cell_index].pCellGrpDynPrm->nUeGrps,
                   pdsch_dyn_params[grouped_cell_index].pCellGrpDynPrm->nUes,
                   pdsch_dyn_params[grouped_cell_index].pCellGrpDynPrm->nCws,
                   pdsch_dyn_params[grouped_cell_index].pCellGrpDynPrm->nCsiRsPrms);*/

            if (group_cells) {
               for (int cell_id = 0; cell_id < num_cells; cell_id++) {
                   pdsch_dyn_params[grouped_cell_index].pDataOut->pTDataTx[cell_id].desc  = data_tx_tensor[cell_id].desc().handle();
                   pdsch_dyn_params[grouped_cell_index].pDataOut->pTDataTx[cell_id].pAddr = data_tx_tensor[cell_id].addr();
               }
            } else {
                pdsch_dyn_params[grouped_cell_index].pDataOut->pTDataTx[0].desc  = data_tx_tensor[i].desc().handle();
                pdsch_dyn_params[grouped_cell_index].pDataOut->pTDataTx[0].pAddr = data_tx_tensor[i].addr();
            }

            if (time_setup_mode == 0){ // In this mode, setup is not timed
                cuphyStatus_t status = cuphySetupPdschTx(pdsch_handle[i], &pdsch_dyn_params[grouped_cell_index], nullptr);
                if(status != CUPHY_STATUS_SUCCESS)
                {
                    if (pdsch_dyn_params[grouped_cell_index].pStatusInfo->status == cuphyPdschStatusType_t::CUPHY_PDSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB) {
                        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Pipeline {}: CUPHY_PDSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB error in cuphySetupPdschTx(): {}. Triggered by TB {} in cell group and cellPrmStatIdx {}",
                                      i, cuphyGetErrorString(status), pdsch_dyn_params[grouped_cell_index].pStatusInfo->ueIdx, pdsch_dyn_params[grouped_cell_index].pStatusInfo->cellPrmStatIdx);
                    } else {
                        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Pipeline {}: Error! cuphySetupPdschTx(): {}", i, cuphyGetErrorString(status));
                    }
                    exit(1);
                }
            }
        }
        gpu_us_delay(delayUs, 0, streams[0].handle());
        CUDA_CHECK(cudaEventRecord(start_streams_event, streams[0].handle()));

        for(int iter = 0; iter < num_iterations; iter++)
        {
            event_timer cuphy_timer;
            cuphy_timer.record_begin(streams[0].handle());

            for(int i = 0; i < num_pdsch_objects; i++)
            {
                cudaStream_t strm_handle = streams[i].handle();

                if(i != 0)
                {
                    CUDA_CHECK(cudaStreamWaitEvent(streams[i].handle(), start_streams_event, 0));
                }
                cuphy_timer_single_cell[i].record_begin(streams[i].handle());

                if (time_setup_mode != 0) { // If time_setup_mode is 1 or 2, it is timed
                    int grouped_cell_index = (group_cells) ? idxSlot : idxSlot * num_cells + i;
                    cuphyStatus_t status = cuphySetupPdschTx(pdsch_handle[i], &pdsch_dyn_params[grouped_cell_index], nullptr);
                    if(status != CUPHY_STATUS_SUCCESS)
                    {
                       if (pdsch_dyn_params[grouped_cell_index].pStatusInfo->status == cuphyPdschStatusType_t::CUPHY_PDSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB) {
                           NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Pipeline {}: CUPHY_PDSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB error in cuphySetupPdschTx(): {}. Triggered by TB {} in cell group and cellPrmStatIdx {}",
                                      i, cuphyGetErrorString(status), pdsch_dyn_params[grouped_cell_index].pStatusInfo->ueIdx, pdsch_dyn_params[grouped_cell_index].pStatusInfo->cellPrmStatIdx);
                       } else {
                           NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Pipeline {}: Error! cuphySetupPdschTx(): {}", i, cuphyGetErrorString(status));
                       }
                       exit(1);
                    }
                }
                if (time_setup_mode != 1)
                {

                    cuphyStatus_t status = cuphyRunPdschTx(pdsch_handle[i], pdsch_proc_mode);
                    if(status != CUPHY_STATUS_SUCCESS)
                    {
                        NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Pipeline {}: Error! cuphyRunPdschTx(): {}", i, cuphyGetErrorString(status));
                    }
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

            for(int i = 0; i < num_pdsch_objects; i++)
            {
                cuphy_timer_single_cell[i].synchronize(); // To be safe
                total_time_single_cell[i] += cuphy_timer_single_cell[i].elapsed_time_ms();
            }

            gpu_us_delay(delayUs, 0, streams[0].handle()); // 10ms delay kernel. Can update/comment out.
            CUDA_CHECK(cudaEventRecord(start_streams_event, streams[0].handle()));
        }
        total_time_slot[idxSlot] = total_time / num_iterations;

        for(int i = 0; i < num_pdsch_objects; i++)
        {
            total_time_single_cell_slot[idxSlot][i] = total_time_single_cell[i] / num_iterations;
        }
    }

    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++)
    {
        NVLOGC_FMT(NVLOG_PDSCH, "Slot # {},  PDSCH pipeline(s) {}: {:.2f} us (avg. over {} iterations)", idxSlot, setup_modes[time_setup_mode].c_str(), total_time_slot[idxSlot] * 1000, num_iterations);
        for(int i = 0; i < num_pdsch_objects; i++)
        {
            if (group_cells) {
                NVLOGC_FMT(NVLOG_PDSCH, "--> PDSCH object # {} with {} cells: {:.2f} us (avg over {} iterations)", i, num_cells, total_time_single_cell_slot[idxSlot][i] * 1000, num_iterations);
            } else {
                NVLOGC_FMT(NVLOG_PDSCH, "--> Cell # {} : {:.2f} us (avg over {} iterations)", i, total_time_single_cell_slot[idxSlot][i] * 1000, num_iterations);
            }
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());


    // Cleanup
    for(int idxSlot = 0; idxSlot < num_slots; idxSlot++)
    {
        for(int i = 0; i < num_pdsch_objects; i++)
        {
            int grouped_cell_index = (group_cells) ? idxSlot : idxSlot * num_cells + i;

            pdsch_params_cleanup(pdsch_static_params[grouped_cell_index], pdsch_cell_grp_dyn_params[grouped_cell_index]);
            if(idxSlot == 0)
            {
                cuphyStatus_t status = cuphyDestroyPdschTx(pdsch_handle[i]);
                if(status != CUPHY_STATUS_SUCCESS)
                {
                    NVLOGF_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Pipeline {}: Error! cuphyDestroyPdschTx(): {}", i, cuphyGetErrorString(status));
                }
            }
            if (!group_cells)
            {
                delete[] output_data[grouped_cell_index].pTDataTx;
            }
        }
        if (group_cells)
        {
            delete[] output_data[idxSlot].pTDataTx;
        }
    }
    nvlog_fmtlog_close(log_thread_id);

    return 0;
}
