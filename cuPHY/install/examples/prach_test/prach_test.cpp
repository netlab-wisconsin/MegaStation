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
#include <random>

#include "cuphy.hpp"
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "util.hpp"
#include "datasets.hpp"

#define NUM_PREAMBLE 64

using namespace std;
using namespace cuphy;


struct DetectSummary {
    int false_detection_count;
    int miss_detection_count;
    int timing_error_count;
    int detected_count;
    int total_count;
}; 

template <typename T>
T div_round_up(T val, T divide_by) {
    return ((val + (divide_by - 1)) / divide_by);
}

void check_results(void * num_detectedPrmb,
                          void * prmbIndex_estimates,
                          void * prmbDelay_estimates,
                          hdf5hpp::hdf5_file & prach_file,
                          float  delay_error_limit,
                          DetectSummary * detect_summary) {

    // Copy output from GPU to CPU for reference comparison   
    std::vector<uint32_t> gpu_num_detectedPrmb(1);
    CUDA_CHECK(cudaMemcpy(gpu_num_detectedPrmb.data(), num_detectedPrmb, sizeof(uint32_t), cudaMemcpyDeviceToHost));

    int numPrmb = gpu_num_detectedPrmb[0];

    std::vector<uint32_t> gpu_prmbIndex_estimates(numPrmb);
    std::vector<float> gpu_prmbDelay_estimates(numPrmb);

    CUDA_CHECK(cudaMemcpy(gpu_prmbIndex_estimates.data(), prmbIndex_estimates, numPrmb * sizeof(uint32_t), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(gpu_prmbDelay_estimates.data(), prmbDelay_estimates, numPrmb * sizeof(float), cudaMemcpyDeviceToHost));        
   
    // Read results from matlab test vector
    using tensor_pinned_R_32U = typed_tensor<CUPHY_R_32U, pinned_alloc>;
    using tensor_pinned_R_32F = typed_tensor<CUPHY_R_32F, pinned_alloc>;
    tensor_pinned_R_32U matlab_prmbIndex_estimates = typed_tensor_from_dataset<CUPHY_R_32U,
        pinned_alloc>(prach_file.open_dataset("prmbIdx_det_0"));
    tensor_pinned_R_32F matlab_prmbDelay_estimates = typed_tensor_from_dataset<CUPHY_R_32F,
        pinned_alloc>(prach_file.open_dataset("delay_time_det_0"));
   
    if (numPrmb == 0) {
        tensor_pinned_R_32U matlab_numPrmb = typed_tensor_from_dataset<CUPHY_R_32U,
                                                pinned_alloc>(prach_file.open_dataset("detIdx_0"));
        int matlab_numPrmb_value = *(matlab_numPrmb.addr());
        if(matlab_numPrmb_value != numPrmb)
        {
            detect_summary->miss_detection_count = detect_summary->miss_detection_count + 1;
        }
        else
        {
            detect_summary->detected_count = detect_summary->detected_count + 1;
        }
    } 
    else if (numPrmb > 1) {
        detect_summary->false_detection_count = detect_summary->false_detection_count + 1;
    }
    else {
        int matlab_prmbIndex_val = matlab_prmbIndex_estimates({0});
        int gpu_prmbIndex_val = gpu_prmbIndex_estimates[0];
        float matlab_prmbDelay_val = matlab_prmbDelay_estimates({0});
        float gpu_prmbDelay_val = gpu_prmbDelay_estimates[0];
            
        if (gpu_prmbIndex_val != matlab_prmbIndex_val) {
            detect_summary->miss_detection_count = detect_summary->miss_detection_count + 1;
        }
        else if (abs(gpu_prmbDelay_val*1e6-matlab_prmbDelay_val*1e6) > delay_error_limit) {
            detect_summary->timing_error_count = detect_summary->timing_error_count + 1;
        }
        else {
            detect_summary->detected_count = detect_summary->detected_count + 1;
        }
    }
}

void usage() {

    std::cout << "prach_test [options]" << std::endl;
    std::cout << "  Options:" << std::endl;
    std::cout << "     -h                              (Display usage information)" << std::endl;
    std::cout << "     input_filename(.h5) test_case(1-4) num_iterations(>0))" << std::endl;

    std::cout << std::endl;
    std::cout << "  Examples:" << std::endl;
    std::cout << "      ./prach_test ~/TV_PRACH_1251.h5 1 1000" << std::endl;
    std::cout << "      ./prach_test -h" << std::endl;
}


int main(int argc, char* argv[]) {  

    if ((argc != 4) || ((argc == 2) && (argv[1][0] == '-') && (argv[1][1] == 'h'))) {
        usage();
        exit(1);
    }

// printf("readinput is done ...\n");

    cudaStream_t strm = 0;

    // Read input HDF5 file that contains config params, input data, and intermediate results for reference comparison.
    std::string prach_filename = "";
    prach_filename.assign(argv[1]);

    int test_case = stoi(argv[2])-1;
    if ((test_case < 0) || (test_case > 3)) {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT,  "Invalid test case: {}. Should be 1-4", test_case);
        exit(1);
    }

    int num_iterations = stoi(argv[3]);
    if (num_iterations <= 0) {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT,  "Invalid number of iterations:  {}. Should be > 0", num_iterations);
        exit(1);
    }

    PrachApiDataset dataset(prach_filename, strm, 0, true);
    dataset.finalize(strm);

    hdf5hpp::hdf5_file prach_file = hdf5hpp::hdf5_file::open(prach_filename.c_str());

// printf("open prach_file is done ...\n");

    // Allocate PRACH handle
    std::unique_ptr<cuphyPrachRxHndl_t> prach_handle = std::make_unique<cuphyPrachRxHndl_t>();
    cuphyStatus_t status = cuphyCreatePrachRx(prach_handle.get(),  &(dataset.prachStatPrms));
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT,  "Error! cuphyCreatePrachRx(): {}", cuphyGetErrorString(status));
        exit(1);
    }

    status = cuphySetupPrachRx(*prach_handle, &(dataset.prachDynPrms));
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT,  "Error! cuphySetupPrachRx(): {}", cuphyGetErrorString(status));
        exit(1);
    }

// printf("allocate output is done ...\n");

    float snr_req[4] = {-16.4, -18.6, -18.7, -20.8}; // Required SNR level in dB 
    float delay_error[4] = {1.04, 1.04, 0.52, 0.26}; // Allowed timing error in us
    float snr_step = 1; // SNR step for test
    int snr_num = 4;    // Number of SNR points for test

    int rxDimension = dataset.dataRxTensor[0].dimensions()[0] * dataset.dataRxTensor[0].dimensions()[1];
    DetectSummary detect_summary;

    __half2 rx_buf_clean [rxDimension];
    __half2 rx_buf_noisy [rxDimension];
    std::default_random_engine generator;
    std::normal_distribution<float> distribution(0,1.0);

    // copy clean rx data from device to host
    CUDA_CHECK(cudaMemcpy(rx_buf_clean, dataset.dataRxTensor[0].addr(), rxDimension*sizeof(__half2), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaDeviceSynchronize());


    float delay_error_limit = delay_error[test_case];
    printf("\nPRACH Detection Results:\n");
    printf("---------------------------------------------------------------\n");
    printf("SNR     Total    Detect    Miss   False   Timing_error\n");
    for (int snrIdx = 0; snrIdx < snr_num; snrIdx++) {
        
        float snr_chan = snr_req[test_case] - (snr_num-1-snrIdx) * snr_step;    
        float snr_lin = exp10(-snr_chan/20)*sqrt(0.5);

        detect_summary.miss_detection_count = 0;
        detect_summary.false_detection_count = 0;
        detect_summary.timing_error_count = 0;
        detect_summary.detected_count = 0;

        for (int iter = 0; iter < num_iterations; iter++) {

            // add noise to rx data and copy from host to device
            for (int idx = 0; idx < rxDimension; idx++ ) {
                float randnum = distribution(generator);
                rx_buf_noisy[idx].x = rx_buf_clean[idx].x + snr_lin*randnum;
                randnum = distribution(generator);
                rx_buf_noisy[idx].y = rx_buf_clean[idx].y + snr_lin*randnum;                
            }
            CUDA_CHECK(cudaMemcpy(dataset.dataRxTensor[0].addr(), rx_buf_noisy, rxDimension*sizeof(__half2), cudaMemcpyHostToDevice));
            CUDA_CHECK(cudaDeviceSynchronize());

            // detect rx data
            status = cuphyRunPrachRx(*prach_handle);
            if (status != CUPHY_STATUS_SUCCESS) {
                NVLOGE_FMT(NVLOG_PRACH, AERIAL_CUPHY_EVENT,  "Error! cuphyRunPrachRx(): {}", cuphyGetErrorString(status));
                exit(1);
            }
            CUDA_CHECK(cudaDeviceSynchronize());
            check_results(dataset.num_detectedPrmb.addr(),
                dataset.prmbIndex_estimates.addr(), dataset.prmbDelay_estimates.addr(), dataset.prach_file,
                delay_error_limit, & detect_summary);      
            CUDA_CHECK(cudaDeviceSynchronize());
        }
        printf("%4.1f   %4d     %4d     %4d     %4d     %4d\n", snr_chan, num_iterations, detect_summary.detected_count, 
            detect_summary.miss_detection_count, detect_summary.false_detection_count, detect_summary.timing_error_count);
    }
    
    printf("\nPRACH Test Summary:\n");
    printf("---------------------------------------------------------------\n");
    printf("At SNR = %4.1f dB: \n", snr_req[test_case]);
    printf("Required  detection rate > 0.99,  false alarm rate < 0.001\n");        
    float detect_rate = ((float) detect_summary.detected_count)/((float) num_iterations);
    float false_rate = ((float) detect_summary.false_detection_count)/((float) num_iterations);
    printf("Simulated detection rate = %.3f, false alarm rate = %.4f\n", detect_rate, false_rate);
    if (detect_rate > 0.99 && false_rate < 0.001) {
        printf("=====> Test PASS \n\n");
    }
    else {
        printf("=====> Test FAIL \n\n");
    }

    cuphyDestroyPrachRx(*prach_handle);
// printf("cuphyPrachReceiver is done ...\n");

    return 0;
}
