/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <iostream>
#include <vector>
#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "util.hpp"
#include "cuphy_internal.h"
#include "datasets.hpp"
#include "ssb_tx.hpp"

#include "memtrace.h"
#include "nvlog.h"


using namespace std;
using namespace hdf5hpp;
using namespace cuphy;

typedef __half2 fp16_complex_t;

#if 0
int checkReference(float2 * expTensorData, __half2 * outTensorData, uint32_t n_t, uint32_t n_f)
{
    float err_threshold = 0.001;
    int  err_cnt = 0;

    printf("\nComparing matlab and gpu test vectors: ");
    printf("\n-------------------------------------\n");

    for(int i = 0; i < n_t; i++)  {
        for(int j = 0; j < n_f; j++)  {
            __half2        outData      = outTensorData[i * n_f + j]; // from GPU
            float2         outDataFloat;
            outDataFloat.x = float (outData.x);
            outDataFloat.y = float (outData.y);
            float2         expDataFloat = expTensorData[i * n_f + j]; // from matlab
            float err_x = abs(outDataFloat.x - expDataFloat.x);
            float err_y = abs(outDataFloat.y - expDataFloat.y);
            if (err_x > err_threshold || err_y > err_threshold) {
                printf("tfSignal mismatch at t = %2d f = %4d, matlab = (%6.3f, %6.3f) vs gpu = (%6.3f, %6.3f)\n",
                i, j, expDataFloat.x, expDataFloat.y, outDataFloat.x, outDataFloat.y);
                err_cnt ++;
            }
        }
    }
    if (err_cnt == 0) {
        printf("====> Test PASS\n\n");
        return 0;
    }
    else {
        printf("====> Test FAIL\n\n");
        return 1;
    }
}
#endif

void usage() {

    std::cout << "  testSS [options]" << std::endl;
    std::cout << "  Options:" << std::endl;
    std::cout << "    -h                          (Display usage information)" << std::endl;
    std::cout << "    -i  input_filename          Input HDF5 filename" << std::endl;
    std::cout << "    -r  # of iterations         Number of iterations to run" << std::endl;
    std::cout << "    -m  proc_mode               Processing mode: streams(0), graphs (1)" << std::endl;

    std::cout << std::endl;
    std::cout << "  Examples:" << std::endl;
    std::cout << "      ./testSS -i ~/TV_cuphy_SS_TC-1_SSBlock.h5 -r 10 -m 1" << std::endl;
    std::cout << "      ./testSS -h" << std::endl;
}

int main(int argc, char* argv[])
{
    int         iArg = 1;
    std::string inputFileName;
    uint64_t    procModeBmsk   = 0;
    int         num_iterations = 1;
    std::string proc_modes[2]   = {"in Stream mode", "in Graphs mode"};

    char nvlog_yaml_file[1024];
    // Relative path from binary to default nvlog_config.yaml
    std::string relative_path = std::string("../../../../").append(NVLOG_DEFAULT_CONFIG_FILE);
    nv_get_absolute_path(nvlog_yaml_file, relative_path.c_str());
    pthread_t log_thread_id = nvlog_fmtlog_init(nvlog_yaml_file, "ssb_tx.log",NULL);
    nvlog_fmtlog_thread_init();

    while(iArg < argc)
    {
        if('-' == argv[iArg][0])
        {
            switch(argv[iArg][1])
            {
            case 'h':
                usage();
                exit(0);
            case 'i':
                if(++iArg >= argc)
                {
                    NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "ERROR: No filename provided.");
                }
                inputFileName.assign(argv[iArg++]);
                break;
            case 'r':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &num_iterations)) || ((num_iterations <= 0)))
                {
                    NVLOGF_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "ERROR: Invalid number of iterations");
                }
                ++iArg;
                break;
            case 'm':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%lu", &procModeBmsk)) || ((CSIRS_PROC_MODE_STREAMS != procModeBmsk) && (CSIRS_PROC_MODE_GRAPHS != procModeBmsk)))
                {
                    NVLOGF_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "ERROR: Invalid processing mode ({:#x})", procModeBmsk);
                }
                ++iArg;
                break;
            default:
                NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "ERROR: Unknown option: {}", argv[iArg]);
                usage();
                exit(1);
            }
        }
        else
        {
            NVLOGF_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "ERROR: Invalid command line argument: {}", argv[iArg]);
        }
    }
    if(inputFileName.empty())
    {
        usage();
        exit(1);
    }

    cuphyStatus_t status = CUPHY_STATUS_SUCCESS;

    // Create CUDA stream
    stream strm(cudaStreamNonBlocking);
    cudaStream_t strm_handle = strm.handle();

    // Allocate SSB handle
    std::unique_ptr<cuphySsbTxHndl_t> ssb_handle = std::make_unique<cuphySsbTxHndl_t>();

    // Static and dynamic datasets
    ssbStaticApiDataset static_dataset;
    ssbDynApiDataset    dynamic_dataset(inputFileName, 1 /* number of cells*/, strm_handle, procModeBmsk);
    status = cuphyCreateSsbTx(ssb_handle.get(), &(static_dataset.ssbStatPrms));
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGF_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "Error! cuphyCreateSsbTx(): {}", cuphyGetErrorString(status));
    }

    event_timer cuphy_timer;

    // Enable dynamic memory allocation tracing in real-time code path; only applicable when running with LD_PRELOAD=<PATH to libmimalloc.so>
    memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);

    status = cuphySetupSsbTx(*ssb_handle, &(dynamic_dataset.ssb_dyn_params));
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGF_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "Error! cuphySetupSsbTx(): {}", cuphyGetErrorString(status));
    }

    strm.synchronize();

    // Time SSB Run.
    cuphy_timer.record_begin(strm_handle);

    for (int iter = 0; iter < num_iterations; iter++) {
        status = cuphyRunSsbTx(*ssb_handle, 0);
    }

    if (status != CUPHY_STATUS_SUCCESS) {
        NVLOGF_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "Error! cuphyRunSsbTx(): {}", cuphyGetErrorString(status));
    }
    memtrace_set_config(0); // disable memory allocation tracing beyond this point

    cuphy_timer.record_end(strm_handle);
    cuphy_timer.synchronize();
    float time1 = cuphy_timer.elapsed_time_ms();
    time1 /= num_iterations;
    NVLOGC_FMT(NVLOG_SSB, "SSB TX Pipeline {}: {:.2f} us (avg. over {} iterations)", proc_modes[procModeBmsk].c_str(), time1 * 1000, num_iterations);
    NVLOGC_FMT(NVLOG_SSB, "- NB: Allocations/PBCH payload generation and CRC computation/mem. copies not included!");

    int rslt = (dynamic_dataset.refCheck(true) == 0) ? 0 : 1;

    status = cuphyDestroySsbTx(*ssb_handle);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGF_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "Error! cuphyDestroySsbTx(): {}", cuphyGetErrorString(status));
    }

    nvlog_fmtlog_close(log_thread_id);
    return rslt;
}
