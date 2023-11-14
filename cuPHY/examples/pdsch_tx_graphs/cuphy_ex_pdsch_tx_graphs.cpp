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

//#define _READ_TB_CRC_ 1

#define REF_CHECK_PDSCH 0 //TODO Set to 1 if you want to run ref. checks once before timing the pipeline.

using namespace cuphy;

/**
 *  @brief Print usage information for the DL pipeline example.
 */
void usage()
{
    std::cout << "cuphy_ex_pdsch_tx_graphs [options]" << std::endl;
    std::cout << "  Options:" << std::endl;
    std::cout << "     -h                Display usage information" << std::endl;
    std::cout << "     homogeneous_mode num_cells input_filename  num_iterations  AAS_mode (Enable homogeneous mode=1, number of cells, Input HDF5 filename (to be replicated), Number of iterations, Enable AAS mode)" << std::endl;
    std::cout << "     homogeneous_mode input_filename num_iterations AAS_mode (Enable homogeneous mode=0, Input filename listing HDF5 TVs (an HDF5 filename per line), Number of iterations, Enable AAS mode)" << std::endl;

    std::cout << std::endl;
    std::cout << "  Examples:" << std::endl;
    std::cout << "      ./cuphy_ex_pdsch_tx_graphs -h" << std::endl;
    std::cout << "      ./cuphy_ex_pdsch_tx_graphs 1 1 ~/dl_pipeline.h5 20 0" << std::endl;
    std::cout << "      ./cuphy_ex_pdsch_tx_graphs 1 1 ~/dl_pipeline.h5 20 1" << std::endl;
    std::cout << "      ./cuphy_ex_pdsch_tx_graphs 0 ~/example_tv_list 20 0" << std::endl;
}

int main(int argc, char* argv[])
{
    std::vector<std::string> hdf5_filenames;
    std::ifstream            tvs_list_file;
    
    cuphyNvlogFmtHelper nvlog_fmt("pdsch_tx_graphs.log");

    if(((argc != 6) && (argc != 5)) || ((argc == 2) && (argv[1][0] == '-') && (argv[1][1] == 'h')))
    {
        usage();
        exit(1);
    }

    int enable_homogeneous_mode = std::stoi(argv[1]);
    if((enable_homogeneous_mode != 1) && (enable_homogeneous_mode != 0))
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Invalid homogeneous mode value: {}. Should be 0 or 1", enable_homogeneous_mode);
        exit(1);
    }

    int num_cells = 0;
    if(!enable_homogeneous_mode)
    {
        tvs_list_file.open(argv[2]);
        if(!tvs_list_file)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Could not open file {} containing list of TVs", argv[2]);
            exit(1);
        }
        std::string tv_filename;
        while(tvs_list_file >> tv_filename)
        {
            hdf5_filenames.emplace_back(tv_filename);
        }
        num_cells = hdf5_filenames.size();
        if(num_cells == 0)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "File containing list of TVs is empty!");
            exit(1);
        }
        tvs_list_file.close();
    }
    else
    {
        num_cells = std::stoi(argv[2]);
        if(num_cells < 1)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Invalid cell count: {}. Should be >= 1", num_cells);
            exit(1);
        }
        hdf5_filenames.resize(num_cells);
    }

    int num_iterations = (enable_homogeneous_mode) ? std::stoi(argv[4]) : std::stoi(argv[3]);
    if(num_iterations <= 0)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Invalid number of iterations: : {}. Should be > 0", num_iterations);
        exit(1);
    }

    int cfg_aas_mode = (enable_homogeneous_mode) ? std::stoi(argv[5]) : std::stoi(argv[4]);
    if(cfg_aas_mode < 0)
    {
        std::cout << "Negative AAS mode value treated as 0." << std::endl;
    }
    else if(cfg_aas_mode > 1)
    {
        std::cout << "AAS mode value > 1 treated as 1." << std::endl;
    }
    bool                 ref_check              = false;
    bool                 aas_mode               = (cfg_aas_mode >= 1);
    std::string          pipeline_mode          = (aas_mode) ? "AAS" : "non AAS";
    cuphyPdschProcMode_t pdsch_proc_mode        = PDSCH_PROC_MODE_NO_GRAPHS;
    bool                 identical_LDPC_configs = true; // A runtime check resets LDPC configs to non-identical if they are not.

    CUDA_CHECK(cudaSetDevice(0)); // Select GPU device 0

    // The Downlink pipeline includes: (a) CRC, (b) LDPC  encoder, (c) Rate-Matching,
    // (d) Modulation Mapper and (e) DMRS components.

    std::vector<stream>        streams;
    std::vector<tensor_device> data_tx_tensor(num_cells);

    // Large buffer added as cuPHY tools needs a buffer with a power of 2 size.
    //int large_buffer_bytes = 4194304;
    int                               large_buffer_bytes    = 2935296;
    int                               large_buffer_elements = large_buffer_bytes / sizeof(cuFloatComplex);
    unique_device_ptr<cuFloatComplex> large_buffer          = make_unique_device<cuFloatComplex>(num_cells * large_buffer_elements);

    // Reset the entire output buffer.
    CUDA_CHECK(cudaMemset(large_buffer.get(), 0, num_cells * large_buffer_bytes)); // This is done in the default stream

    cudaEvent_t              start_streams_event;
    std::vector<cudaEvent_t> stop_streams_events(num_cells);

    std::string                         first_filename = (enable_homogeneous_mode) ? argv[3] : hdf5_filenames[0];
    std::unique_ptr<hdf5hpp::hdf5_file> tmp_first_file = std::make_unique<hdf5hpp::hdf5_file>(hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::open(first_filename.c_str())));
    bool                                use_new_api    = tmp_first_file->is_valid_dataset("ue_pars");
    if(!use_new_api)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "PDSCH examples no longer support old API TVs");
        exit(1);
    }
    tmp_first_file.reset();

    std::unique_ptr<cuphyPdschTxHndl_t[]>                pdsch_handle = std::make_unique<cuphyPdschTxHndl_t[]>(num_cells);
    std::vector<cuphyPdschStatPrms_t>                    pdsch_static_params(num_cells);
    typedef std::vector<cuphyPdschCellGrpDynPrm_t>       dyn_params_vector_t;
    dyn_params_vector_t                                  cell_dyn_params(1);
    std::vector<dyn_params_vector_t>                     pdsch_cell_grp_dyn_params(num_cells, cell_dyn_params); // Limited to one cell group per pipeline
    std::vector<std::unique_ptr<hdf5hpp::hdf5_file>>     input_file(num_cells);
    std::vector<typed_tensor<CUPHY_R_8U, pinned_alloc>>  crc_input_data;
    std::vector<typed_tensor<CUPHY_R_32U, pinned_alloc>> tb_crc_input_data;

    cudaGraph_t     graph;
    cudaGraphExec_t graph_instance;

    for(int i = 0; i < num_cells; i += 1)
    {
        streams.emplace_back(cudaStreamNonBlocking, PDSCH_STREAM_PRIORITY);
        data_tx_tensor[i] = tensor_device((void*)((cuFloatComplex*)large_buffer.get() + i * large_buffer_elements), CUPHY_C_16F, CUPHY_N_TONES_PER_PRB * 273, OFDM_SYMBOLS_PER_SLOT, MAX_DL_LAYERS, cuphy::tensor_flags::align_tight);
        ;

        int data_tx_tensor_bytes = data_tx_tensor[i].desc().get_size_in_bytes();
        if(data_tx_tensor_bytes > large_buffer_bytes)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Buffer ({} bytes) is smaller than data_tx_tensor ({})", large_buffer_bytes, data_tx_tensor_bytes);
            exit(1);
        }
        if(enable_homogeneous_mode)
        {
            hdf5_filenames[i] = argv[3];
        }
        std::cout << " PdschTx Pipeline " << i << " uses TV " << hdf5_filenames[i] << std::endl;

        input_file[i] = std::make_unique<hdf5hpp::hdf5_file>(hdf5hpp::hdf5_file(hdf5hpp::hdf5_file::open(hdf5_filenames[i].c_str())));

        //Read static and dynamic parameters per cell
        read_pdsch_static_pars_from_file(pdsch_static_params[i], *input_file[i], hdf5_filenames[i].c_str(), ref_check, identical_LDPC_configs);
        read_cell_group_dynamic_pars_from_file(pdsch_cell_grp_dyn_params[i], *input_file[i]);
        int cfg_num_cells = pdsch_cell_grp_dyn_params[i][0].nCells;
        if(cfg_num_cells != 1)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Error! # cells = {}, but this example only supports a single cell per pipeline", num_cells);
            exit(1);
        }

        //Parse input dataset
        hdf5hpp::hdf5_dataset crc_dataset = (*input_file[i]).open_dataset("InputData");
        crc_input_data.emplace_back(typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(crc_dataset, cuphy::tensor_flags::align_default, streams[i].handle()));
        pdsch_static_params[i].full_slot_processing = (!aas_mode);
#ifdef _READ_TB_CRC_
        hdf5hpp::hdf5_dataset tb_crc_dataset = (*input_file[i]).open_dataset("tbCrcBuffer");
        tb_crc_input_data.emplace_back(typed_tensor_from_dataset<CUPHY_R_32U, pinned_alloc>(tb_crc_dataset, cuphy::tensor_flags::align_default, streams[i].handle()));
        pdsch_static_params[i].read_TB_CRC = true;
#else
        pdsch_static_params[i].read_TB_CRC = false;
#endif
        CUDA_CHECK(cudaEventCreateWithFlags(&stop_streams_events[i], cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventCreateWithFlags(&start_streams_event, cudaEventDisableTiming));

    std::vector<cuphyPdschDataIn_t>  data_in(num_cells);
    std::vector<cuphyPdschDataIn_t>  tb_crc_data_in(num_cells);
    std::vector<cuphyPdschDataOut_t> output_data(num_cells);
    std::vector<cuphyPdschDynPrms_t> pdsch_dyn_params(num_cells);

    auto crc_input_data_ptr = std::make_unique<uint8_t*[]>(num_cells);
#ifdef _READ_TB_CRC_
    auto tb_crc_input_data_ptr = std::make_unique<uint8_t*[]>(num_cells);
#endif

    for(int i = 0; i < num_cells; i += 1)
    {
        crc_input_data_ptr[i] = (uint8_t*)crc_input_data[i].addr();
        data_in[i] = {&crc_input_data_ptr[i], cuphyPdschDataIn_t::CPU_BUFFER};

#ifdef _READ_TB_CRC_
        tb_crc_input_data_ptr[i] = (uint8_t*)tb_crc_input_data[i].addr();
        tb_crc_data_in[i] = {&tb_crc_input_data_ptr[i], cuphyPdschDataIn_t::CPU_BUFFER};
#else
        tb_crc_data_in[i] = {nullptr, cuphyPdschDataIn_t::CPU_BUFFER};
#endif
        input_file[i].reset(); // Close the file; will reopen it later

        output_data[i] = {new cuphyTensorPrm_t[1]}; //single cell per pipeline for now

        pdsch_dyn_params[i]                             = {streams[i].handle(), pdsch_proc_mode, pdsch_cell_grp_dyn_params[i].data(), &data_in[i], &tb_crc_data_in[i], &output_data[i]};
        pdsch_dyn_params[i].pDataOut->pTDataTx[0].desc  = data_tx_tensor[i].desc().handle();
        pdsch_dyn_params[i].pDataOut->pTDataTx[0].pAddr = data_tx_tensor[i].addr();

        cuphyStatus_t status = cuphyCreatePdschTx(&pdsch_handle[i], &pdsch_static_params[i]); // Currently calling PdschTx constructor with empty filename
        if(status != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Pipeline {}: Error! cuphyCreatePdschTx(): {}", i, cuphyGetErrorString(status));
            exit(1);
        }
    }

    for(int i = 0; i < num_cells; i++)
    {
        cuphyStatus_t status = cuphySetupPdschTx(pdsch_handle[i], &pdsch_dyn_params[i], nullptr);
        if(status != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Pipeline {}: Error! cuphySetupPdschTx(): {}", i, cuphyGetErrorString(status));
            exit(1);
        }
    }
    CUDA_CHECK(cudaDeviceSynchronize());

#if REF_CHECK_PDSCH
    // Run all PDSCH pipeline(s) once w/ ref checks enabled
    std::cout << std::endl;
    std::cout << "Running all " << num_cells << " DL pipelines once w/ reference checks enabled in " << pipeline_mode << " mode." << std::endl;
    for(int i = 0; i < num_cells; i++)
    {
        updateRefCheck(pdsch_handle[i], true);
        cudaStream_t  strm_handle = streams[i].handle();
        cuphyStatus_t status      = cuphyRunPdschTx(pdsch_handle[i], pdsch_proc_mode);
        if(status != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Pipeline {}: Error! cuphyRunPdschTx(): {}", i, cuphyGetErrorString(status));
            exit(1);
        }
        updateRefCheck(pdsch_handle[i], ref_check);
    }
#endif

    // Start stream capture on streams[0]
    CUDA_CHECK(cudaStreamBeginCapture(streams[0].handle(), cudaStreamCaptureModeGlobal));
    gpu_empty_kernel(streams[0].handle()); // empty kernel as single-node graph entry point
    CUDA_CHECK(cudaEventRecord(start_streams_event, streams[0].handle()));

    for(int i = 0; i < num_cells; i++)
    {
        if(i != 0)
        {
            CUDA_CHECK(cudaStreamWaitEvent(streams[i].handle(), start_streams_event, 0));
        }

        cuphyStatus_t status = cuphyRunPdschTx(pdsch_handle[i], pdsch_proc_mode);
        if(status != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Pipeline {}: Error! cuphyRunPdschTx(): {}", i, cuphyGetErrorString(status));
            exit(1);
        }

        /* Record a stop event on all streams but the streams[0].handle() stream. */
        /* Have streams[0].handle() stream wait for all other streams to complete their work. */
        if(i != 0)
        {
            CUDA_CHECK(cudaEventRecord(stop_streams_events[i], streams[i].handle()));
            CUDA_CHECK(cudaStreamWaitEvent(streams[0].handle(), stop_streams_events[i], 0));
        }
    }

    CUDA_CHECK(cudaStreamEndCapture(streams[0].handle(), &graph));
    streams[0].synchronize();

    /* Time PDSCH pipeline(s) Run().
       NB: The following are not timed: Reading the HDF5 file, allocations, memory transfers.
       Also, the timing code does not include any reference checks as these will currently
       fail (some buffers need to be reset across iterations).
    */
    std::cout << std::endl;
    std::cout << "Timing " << num_cells << " PDSCH DL pipeline(s) Run() w/o reference checks in " << pipeline_mode << " mode." << std::endl;
    std::cout << "- NB: Allocations not included. Ref. checks will fail!" << std::endl
              << std::endl;

    float total_time = 0;
    CUDA_CHECK(cudaGraphInstantiate(&graph_instance, graph, NULL /*error node*/, NULL /*log buffer*/, 0 /*log buffer size*/));

#if CUDART_VERSION >= 11010
    CUDA_CHECK(cudaGraphUpload(graph_instance, streams[0].handle()));
#endif

    event_timer cuphy_timer;
    cuphy_timer.record_begin(streams[0].handle());

    // Run executable graph num_iterations times on streams[0] CUDA stream.
    for(int iter = 0; iter < num_iterations; iter++)
    {
        CUDA_CHECK(cudaGraphLaunch(graph_instance, streams[0].handle()));
    }

    cuphy_timer.record_end(streams[0].handle());
    cuphy_timer.synchronize();
    total_time += cuphy_timer.elapsed_time_ms();

    total_time /= num_iterations;

    printf("%d PDSCH pipeline(s): %.2f us (avg. over %d iterations)\n", num_cells, total_time * 1000, num_iterations);

    CUDA_CHECK(cudaDeviceSynchronize());

    // Cleanup
    for(int i = 0; i < num_cells; i++)
    {
        pdsch_params_cleanup(pdsch_static_params[i], pdsch_cell_grp_dyn_params[i]);
        cuphyStatus_t status = cuphyDestroyPdschTx(pdsch_handle[i]);
        if(status != CUPHY_STATUS_SUCCESS)
        {
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Pipeline {}: Error! cuphyDestroyPdschTx(): {}", i, cuphyGetErrorString(status));
            exit(1);
        }
        delete[] output_data[i].pTDataTx;
    }

    CUDA_CHECK(cudaGraphExecDestroy(graph_instance));
    CUDA_CHECK(cudaGraphDestroy(graph));

    return 0;
}
