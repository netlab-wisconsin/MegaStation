/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_internal.h"
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "util.hpp"
#include "datasets.hpp"

#include <cstdlib>
#include <string>
#include <iostream>
#include <fstream>
#include <sstream>
#include <vector>

#include "memtrace.h"
#include "nvlog.h"


using namespace std;
using namespace cuphy;
using namespace hdf5hpp;

#if 0
int checkReference(float2 * expTensorData, __half2 * outTensorData, uint32_t n_p, uint32_t n_t, uint32_t n_f, string tv_name)
{
    float err_threshold = 0.001;
    int  err_cnt = 0;

    printf("\nComparing matlab and gpu test vectors: ");
    printf("\nTensor dimension: %d x %d x %d", n_f, n_t, n_p);
    printf("\n-------------------------------------\n");

    for(int k = 0; k < n_p; k++)  {
        for(int i = 0; i < n_t; i++)  {
            for(int j = 0; j < n_f; j++)  {
                __half2        outData      = outTensorData[k * n_t * n_f + i * n_f + j]; // from GPU
                float2         outDataFloat;
                outDataFloat.x = float (outData.x);
                outDataFloat.y = float (outData.y);
                float2         expDataFloat = expTensorData[k * n_t * n_f + i * n_f + j]; // from matlab
                float err_x = abs(outDataFloat.x - expDataFloat.x);
                float err_y = abs(outDataFloat.y - expDataFloat.y);
                if (err_x > err_threshold || err_y > err_threshold)
                {
                    // printf("tfSignal mismatch at p = %2d t = %2d f = %4d, matlab = (%6.3f, %6.3f) vs gpu = (%6.3f, %6.3f)\n",
                    //     k, i, j, expDataFloat.x, expDataFloat.y, outDataFloat.x, outDataFloat.y);
                    err_cnt++;
                }
                // else if(abs(expDataFloat.x) > err_threshold || abs(expDataFloat.y) > err_threshold)
                // {
                //     printf("tfSignal match at p = %2d t = %2d f = %4d, matlab = (%6.3f, %6.3f) vs gpu = (%6.3f, %6.3f)\n",
                //         k, i, j, expDataFloat.x, expDataFloat.y, outDataFloat.x, outDataFloat.y);
                // }
            }
        }
    }
    if (err_cnt == 0) {
        printf("====> TV %s: Test PASS\n\n", tv_name.c_str());
        return 0;
    }
    else {
        printf("====> TV %s: Test FAIL. Found %d mismatched symbols\n\n", tv_name.c_str(), err_cnt);
        return 1;
    }
}
#endif

void usage() {

    std::cout << "  nzp_csi_rs_test [options]" << std::endl;
    std::cout << "  Options:" << std::endl;
    printf("    -h                          Display usage information\n");
    printf("    -i  input_filename          Input HDF5 filename\n");
    printf("    -r  # of iterations         Number of iterations to run\n");
    printf("    -m  proc_mode               Processing mode: streams(0), graphs (1)\n");

    std::cout << std::endl;
    std::cout << "  Examples:" << std::endl;
    std::cout << "      ./nzp_csi_rs_test -i ~/TVnr_4001_CSIRS_gNB_CUPHY_s0p0.h5 -r 10 -m 1" << std::endl;
    std::cout << "      ./nzp_csi_rs_test -h" << std::endl;
}

int main(int argc, char* argv[])
{
    int         iArg = 1;
    std::string inputFileName;
    uint64_t    procModeBmsk   = 0;
    int         num_iterations = 1;

    cuphyNvlogFmtHelper nvlog_fmt("csirs_tx.log");

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
            case 'm':
                if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%lu", &procModeBmsk)) || ((CSIRS_PROC_MODE_STREAMS != procModeBmsk) && (CSIRS_PROC_MODE_GRAPHS != procModeBmsk)))
                {
                    NVLOGF_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT,  "ERROR: Invalid processing mode {:#x}", procModeBmsk);
                }
                ++iArg;
                break;
            default:
                NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                usage();
                exit(1);
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

    cuphyStatus_t status = CUPHY_STATUS_SUCCESS;

    CUDA_CHECK(cudaSetDevice(0));

    // Create CUDA stream
    stream strm(cudaStreamNonBlocking);
    cudaStream_t strm_handle = strm.handle();

    // Allocate CSI-RS handle
    std::unique_ptr<cuphyCsirsTxHndl_t> csirs_handle = std::make_unique<cuphyCsirsTxHndl_t>();

    csirsStaticApiDataset static_dataset(inputFileName);
    csirsDynApiDataset    dynamic_dataset(inputFileName, 1 /* number of cells*/, strm_handle, procModeBmsk);

    status = cuphyCreateCsirsTx(csirs_handle.get(), &(static_dataset.csirsStatPrms));
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGF_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT,  "Error! cuphyCreateCsirsTx(): {}", cuphyGetErrorString(status));
    }

#if 0
    status = cuphySetupCsirsTx(*csirs_handle, &(dynamic_dataset.csirs_dyn_params));
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT,  "Error! cuphySetupCsirsTx(): {}", cuphyGetErrorString(status));
    }

    // warm-up call
    status = cuphyRunCsirsTx(*csirs_handle);
    if (status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT,  "Error! cuphyRunCsirsTx(): {}", cuphyGetErrorString(status));
    }
#endif

    strm.synchronize();

    // Time NZP CSI-RS Setup/run combined.
    event_timer cuphy_timer;

    // Enable dynamic memory allocation tracing in real-time code path; only applicable when running with LD_PRELOAD=<PATH to libmimalloc.so>
    memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE | MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);

    cuphy_timer.record_begin(strm_handle);

    for (int iter = 0; iter < num_iterations; iter++) {

        status = cuphySetupCsirsTx(*csirs_handle, &(dynamic_dataset.csirs_dyn_params));

        // generate CSI-RS signal
        status = cuphyRunCsirsTx(*csirs_handle);
    }

    if (status != CUPHY_STATUS_SUCCESS) {
        NVLOGF_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT,  "Error! cuphyRunCsirsTx(): {}", cuphyGetErrorString(status));
    }
    memtrace_set_config(0); // disable memory allocation tracing beyond this point

    cuphy_timer.record_end(strm_handle);
    cuphy_timer.synchronize();
    float time1 = cuphy_timer.elapsed_time_ms();
    time1 /= num_iterations;
    NVLOGC_FMT(NVLOG_CSIRS, "CSI-RS TX Pipeline Both Setup and Run (in {} mode): {:.2f} us (avg. over {} iterations)",
              (procModeBmsk == 0) ? "Stream" : "Graphs", time1 * 1000, num_iterations);
    NVLOGC_FMT(NVLOG_CSIRS, "- NB: Allocations not included!");

    strm.synchronize();

    // Time NZP CSI-RS Run. NB: scrambling sequence generation, async copies etc are not counted!
    
    cuphy_timer.record_begin(strm_handle);

    for (int iter = 0; iter < num_iterations; iter++) {
        // generate CSI-RS signal
        status = cuphyRunCsirsTx(*csirs_handle);
    }

    if (status != CUPHY_STATUS_SUCCESS) {
        NVLOGF_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Error! cuphyRunCsirsTx(): ", cuphyGetErrorString(status));
    }

    cuphy_timer.record_end(strm_handle);
    cuphy_timer.synchronize();
    time1 = cuphy_timer.elapsed_time_ms();
    time1 /= num_iterations;
    NVLOGC_FMT(NVLOG_CSIRS, "CSI-RS TX Pipeline Only Run (in {} mode): {:.2f} us (avg. over {} iterations)", (procModeBmsk == 0) ? "Stream": "Graphs", time1 * 1000, num_iterations);
    NVLOGC_FMT(NVLOG_CSIRS, "- NB: Allocations/mem. copies not included!");

    // Code for ref. checks
    strm.synchronize();
    int rslt = (dynamic_dataset.refCheck(true) == 0) ? 0 : 1;

    // Cleanup
    status = cuphyDestroyCsirsTx(*csirs_handle);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGF_FMT(NVLOG_CSIRS, AERIAL_CUPHY_EVENT, "Error! cuphyDestroyCsirsTx(): {}", cuphyGetErrorString(status));
    }

    return rslt;
}
