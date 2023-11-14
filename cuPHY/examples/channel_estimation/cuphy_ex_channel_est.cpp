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
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"

////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("cuphy_ex_channel_est [options]\n");
    printf("  Options:\n");
    printf("    -i  input_filename     Input HDF5 filename, which must contain the following datasets:\n");
    printf("                           DMRS_index_freq: Per-UE frequency indices for DMRS symbols\n");
    printf("                           DMRS_index_time: Per-UE time indices for DMRS symbols\n");
    printf("                           W_freq:          Per-UE 1-D interpolation filter (frequency)\n");
    printf("                           W_time:          Per-UE 1-D interpolation filter (time)\n");
    printf("                           Y:               Symbol grid (time/frequency) for each antennas\n");
    printf("    -p                     Convert fp32 inputs to fp16 for storage in memory ('pseudo-fp16' mode)\n");
    printf("    -h                     Display usage information\n");
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    cuphyNvlogFmtHelper nvlog_fmt("channel_est.log");
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int         iArg = 1;
        std::string inputFilename;
        bool        b16 = false;
        while(iArg < argc)
        {
            if('-' == argv[iArg][0])
            {
                switch(argv[iArg][1])
                {
                case 'i':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided");
                    }
                    inputFilename.assign(argv[iArg++]);
                    break;
                case 'h':
                    usage();
                    exit(0);
                    break;
                case 'p':
                    b16 = true;
                    ++iArg;
                    break;
                default:
                    NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                    usage();
                    exit(1);
                    break;
                }
            }
            else
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
                exit(1);
            }
        }
        if(inputFilename.empty())
        {
            usage();
            exit(1);
        }
        //------------------------------------------------------------------
        // Open the input file
        hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFilename.c_str());

        //------------------------------------------------------------------
        // Allocate tensors in device memory
        // clang-format off
        cuphy::tensor_device tY               = cuphy::tensor_from_dataset(fInput.open_dataset("Y"),               cuphy::tensor_flags::align_coalesce);
        cuphy::tensor_device tDMRS_index_freq = cuphy::tensor_from_dataset(fInput.open_dataset("DMRS_index_freq"), cuphy::tensor_flags::align_tight);
        cuphy::tensor_device tDMRS_index_time = cuphy::tensor_from_dataset(fInput.open_dataset("DMRS_index_time"), cuphy::tensor_flags::align_tight);
        cuphy::tensor_device tW_freq          = cuphy::tensor_from_dataset(fInput.open_dataset("W_freq"),          cuphy::tensor_flags::align_coalesce);
        cuphy::tensor_device tW_time          = cuphy::tensor_from_dataset(fInput.open_dataset("W_time"),          cuphy::tensor_flags::align_coalesce);
        // clang-format on

        // clang-format off
        printf("Input tensors:\n");
        printf("---------------------------------------------------------------\n");
        printf("Y:               %s\n",   tY.desc().get_info().to_string(false).c_str());
        printf("DMRS_index_freq: %s\n",   tDMRS_index_freq.desc().get_info().to_string(false).c_str());
        printf("DMRS_index_time: %s\n",   tDMRS_index_time.desc().get_info().to_string(false).c_str());
        printf("W_freq:          %s\n",   tW_freq.desc().get_info().to_string(false).c_str());
        printf("W_time:          %s\n\n", tW_time.desc().get_info().to_string(false).c_str());
        // clang-format on

        //------------------------------------------------------------------
        // Determine parameters derived from the input data
        int numUEs              = tDMRS_index_freq.dimensions()[1];
        int numOFDMSymbols      = tY.dimensions()[1];
        int numAntennas         = tY.rank() > 2 ? tY.dimensions()[2] : 1;
        int numSubcarriers      = tY.dimensions()[0];
        int numSubcarriersPerUE = tW_freq.dimensions()[0];

        printf("Derived parameters:\n");
        printf("---------------------------------------------------------------\n");
        printf("numSubcarriers:      %i\n", numSubcarriers);
        printf("numUEs:              %i\n", numUEs);
        printf("numOFDMSymbols:      %i\n", numOFDMSymbols);
        printf("numSubcarriersPerUE: %i\n", numSubcarriersPerUE);
        printf("numAntennas:         %i\n\n", numAntennas);

        // Allocate an output buffer based on the input dimensions
        cuphy::tensor_device tH_interp(b16 ? CUPHY_C_16F : CUPHY_C_32F,
                                       numSubcarriersPerUE,
                                       numOFDMSymbols,
                                       numAntennas,
                                       numUEs,
                                       cuphy::tensor_flags::align_coalesce);

        printf("Tensor layout:\n");
        printf("---------------------------------------------------------------\n");
        printf("Y:               addr: %p, %s, size: %.1f kB\n",
               tY.addr(),
               tY.desc().get_info().to_string().c_str(),
               tY.desc().get_size_in_bytes() / 1024.0);
        printf("DMRS_index_freq: addr: %p, %s, size: %.1f kB\n",
               tDMRS_index_freq.addr(),
               tDMRS_index_freq.desc().get_info().to_string().c_str(),
               tDMRS_index_freq.desc().get_size_in_bytes() / 1024.0);
        printf("DMRS_index_time: addr: %p, %s, size: %.1f kB\n",
               tDMRS_index_time.addr(),
               tDMRS_index_time.desc().get_info().to_string().c_str(),
               tDMRS_index_time.desc().get_size_in_bytes() / 1024.0);
        printf("W_freq:          addr: %p, %s, size: %.1f kB\n",
               tW_freq.addr(),
               tW_freq.desc().get_info().to_string().c_str(),
               tW_freq.desc().get_size_in_bytes() / 1024.0);
        printf("W_time:          addr: %p, %s, size: %.1f kB\n",
               tW_time.addr(),
               tW_time.desc().get_info().to_string().c_str(),
               tW_time.desc().get_size_in_bytes() / 1024.0);
        printf("H_interp:        addr: %p, %s, size: %.1f kB\n\n",
               tH_interp.addr(),
               tH_interp.desc().get_info().to_string().c_str(),
               tH_interp.desc().get_size_in_bytes() / 1024.0);
#if 0
        cuphy::tensor_info tinfo = tW_time.desc().get_info();
        std::vector<float> W_time_host(tW_time.desc().get_size_in_bytes() / sizeof(float));
        if(cudaSuccess != cudaMemcpy(W_time_host.data(), tW_time.addr(), tW_time.desc().get_size_in_bytes(), cudaMemcpyDeviceToHost))
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "Error performing memcpy");
        }

        for(int j = 0; j < 4; ++j)
        {
            for(int i = 0; i < 14; ++i)
            {
                size_t offset = tinfo.layout.get_offset(i, j, 11);
                printf("offset: i = %i, j = %i, %lu, value: %f\n", i, j, offset, W_time_host[offset]);
            }
        }
#endif
#if 1
        cuphyStatus_t s = cuphyChannelEst1DTimeFrequency(tH_interp.desc().handle(),
                                                         tH_interp.addr(),
                                                         tY.desc().handle(),
                                                         tY.addr(),
                                                         tW_freq.desc().handle(),
                                                         tW_freq.addr(),
                                                         tW_time.desc().handle(),
                                                         tW_time.addr(),
                                                         tDMRS_index_freq.desc().handle(),
                                                         tDMRS_index_freq.addr(),
                                                         tDMRS_index_time.desc().handle(),
                                                         tDMRS_index_time.addr(),
                                                         0);
        if(CUPHY_STATUS_SUCCESS != s)
        {
            throw cuphy::cuphy_exception(s);
        }
#endif
#if 0
        printf("Writing H_interp: %s, size: %.1f kB\n\n",
               tH_interp.desc().get_info().to_string().c_str(),
               tH_interp.desc().get_size_in_bytes() / 1024.0);
        hdf5hpp::hdf5_file fOutput = hdf5hpp::hdf5_file::create("test_output.h5");
        cuphy::write_HDF5_dataset(fOutput, tH_interp, "H_interp");
        // If successful, test_output.h5 can be loaded from MATLAB using
        // the hdf5_load_nv.m script:
        // s = hdf5_load_nv('test_output.h5');
        // or
        // hdf5_load_nv('test_output.h5');
#endif
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    return returnValue;
}
