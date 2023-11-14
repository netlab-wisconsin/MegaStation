/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
#include "cuphy.h"
#include "cuphy.hpp"
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "util.hpp"
#include "cuphy_internal.h"

using namespace std;
using namespace cuphy;

void usage() {
    std::cout << "modulation_mapper [options]" << std::endl;
    std::cout << "  Options:" << std::endl;
    std::cout << "     -h                              (Display usage information)" << std::endl;
    std::cout << "     input_filename  num_iterations  (Input HDF5 filename, Number of iterations)" << std::endl;


    std::cout << std::endl;
    std::cout << "  Examples:" << std::endl;
    std::cout << "      ./modulation_mapper ~/input_file.h5 20" << std::endl;

}

int main(int argc, char* argv[]) {

    using tensor_pinned_R_64F = typed_tensor<CUPHY_R_64F, pinned_alloc>;
    using tensor_pinned_C_64F = typed_tensor<CUPHY_C_64F, pinned_alloc>;
    
    cuphyNvlogFmtHelper nvlog_fmt("modulation_mapper.log");

    const int ELEMENT_SIZE = sizeof(uint32_t) * 8; // 32 bits
    cudaStream_t strm = 0;

    if ((argc != 3) || ((argc == 2) && (argv[1][0] == '-') && (argv[1][1] == 'h'))) {
        usage();
        exit(1);
    }

    int num_iterations = stoi(argv[2]);
    if (num_iterations <= 0) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Invalid number of iterations: {}. Should be > 0", num_iterations);
        exit(1);
    }

    // Read input HDF5 file to read rate-matching output.
    hdf5hpp::hdf5_file input_file = hdf5hpp::hdf5_file::open(argv[1]);
    hdf5hpp::hdf5_dataset ue_dataset = input_file.open_dataset("ue_pars");

    // This example only processes the first TB (Transport Block) of the input HDF file.
    int num_TBs = 1;
    tensor_pinned_R_64F input_data = typed_tensor_from_dataset<CUPHY_R_64F, pinned_alloc>(input_file.open_dataset("tb0_layer_mapped"));
    tensor_pinned_C_64F output_data = typed_tensor_from_dataset<CUPHY_C_64F, pinned_alloc>(input_file.open_dataset("tb0_qams"));

    const int rate_matched_bits = input_data.layout().dimensions()[0];
    const int qam_elements = output_data.layout().dimensions()[0];
    int modulation_order  = rate_matched_bits / qam_elements;

    tensor_device d_in_tensor(CUPHY_BIT, rate_matched_bits);
    buffer<uint32_t, pinned_alloc> h_in_tensor(d_in_tensor.desc().get_size_in_bytes());

    tensor_device modulation_output(CUPHY_C_16F, qam_elements,
                                    cuphy::tensor_flags::align_tight);

    for (int element_start = 0; element_start < rate_matched_bits; element_start += ELEMENT_SIZE)  {
            uint32_t bits = 0;
            for (int offset = 0; ((offset < ELEMENT_SIZE) && (element_start + offset < rate_matched_bits)); offset++) {
                uint32_t bit = (input_data({element_start + offset}) == 1) ? 1 : 0;
                // 1st element of HDF5 file's tb_codedcbs datatset will map to the
		// least significant bit of a tensor element
		bits |= (bit << offset);
           }
           uint32_t* word = h_in_tensor.addr() + (element_start / ELEMENT_SIZE);
           *word          = bits;
    }

    std::vector<PdschPerTbParams> h_workspace(num_TBs);
    std::vector<PdschDmrsParams> h_pdschDmrsParams(num_TBs);
    for (int i = 0; i < num_TBs; i++) {
        h_workspace[i].G = rate_matched_bits;
        h_workspace[i].Qm = modulation_order;
        //remaining h_workspace fields unitialized. Unused in modulation kernel.

        // Need to read beta_qam from the dataset, since it was used on the reference output
        hdf5hpp::hdf5_dataset_elem ue_config = ue_dataset[i];
        h_pdschDmrsParams[i].beta_qam = ue_config["beta_qam"].as<float>();
        h_pdschDmrsParams[i].num_Rbs = 0; // force legacy flat input/output - we will likely deprecate that mode
        //TODO remaining h_pdschDmrsParams fields uninitialized. Unused in modulation kernel.
    }
    unique_device_ptr<PdschPerTbParams> d_workspace = make_unique_device<PdschPerTbParams>(num_TBs);
    unique_device_ptr<PdschDmrsParams> d_pdschDmrsParams = make_unique_device<PdschDmrsParams>(num_TBs);

    CUDA_CHECK(cudaMemcpy(d_in_tensor.addr(), h_in_tensor.addr(), d_in_tensor.desc().get_size_in_bytes(), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_workspace.get(), h_workspace.data(), num_TBs * sizeof(PdschPerTbParams), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_pdschDmrsParams.get(), h_pdschDmrsParams.data(), num_TBs * sizeof(PdschDmrsParams), cudaMemcpyHostToDevice));

    std::cout << "Will run modulation_mapper for Transport Block (TB) 0 (HDF5 file: " << argv[1] << ")" << std::endl;
    std::cout << std::endl << "# rate matched bits (input) = " << rate_matched_bits << std::endl;
    std::cout << "modulation order = " << modulation_order << std::endl;
    std::cout << "# symbols (output) = " << qam_elements << std::endl << std::endl;

    int max_bits_per_layer = 0; // Does not matter when cuphyModulation's 1st arg. is nullptr. Otherwise it's max(G/Nl) across all TBs.

    // Allocate launch config struct.
    std::unique_ptr<cuphyModulationLaunchConfig> modulation_hndl = make_unique<cuphyModulationLaunchConfig>();

    // Allocate descriptors and setup modulation mapper component
    uint8_t desc_async_copy = 1; // Copy descriptor to the GPU during setup

    size_t desc_size=0, alloc_size=0;
    cuphyStatus_t status = cuphyModulationGetDescrInfo(&desc_size, &alloc_size);
    if (status != CUPHY_STATUS_SUCCESS) {
        printf("cuphyModulationGetDescrInfo error %d\n", status);
        throw cuphy::cuphy_exception(status);
    }
    cuphy::unique_device_ptr<uint8_t> d_modulation_desc = make_unique_device<uint8_t>(desc_size);
    cuphy::unique_pinned_ptr<uint8_t> h_modulation_desc = make_unique_pinned<uint8_t>(desc_size);

    // Calling cuphySetupModulation with d_params == nullptr is permitted, and assumes
    // output is a contiguous buffer rather than a 3D {3276, 14, 4} tensor as in the PdschTx example.
    status = cuphySetupModulation(modulation_hndl.get(),
                                  d_pdschDmrsParams.get(),
                                  d_in_tensor.desc().handle(),
                                  d_in_tensor.addr(),
                                  qam_elements,
                                  max_bits_per_layer,
                                  num_TBs,
                                  d_workspace.get(),
                                  modulation_output.desc().handle(),
                                  modulation_output.addr(),
                                  h_modulation_desc.get(),
                                  d_modulation_desc.get(),
                                  desc_async_copy,
                                  strm);
    if (status != CUPHY_STATUS_SUCCESS) {
        throw std::runtime_error("Invalid argument(s) for cuphySetupModulation");
    }

    event_timer cuphy_timer;
    cuphy_timer.record_begin();

    for (int iter = 0; iter < num_iterations; iter++) {
         // Run Modulation
         launch_kernel(modulation_hndl.get()->m_kernelNodeParams, strm);
    }

    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "CUDA Error {}", cudaGetErrorString(cuda_error));
        exit(1);
    }

    cuphy_timer.record_end();
    cuphy_timer.synchronize();
    float time1 = cuphy_timer.elapsed_time_ms();

    if (status != CUPHY_STATUS_SUCCESS) {
        throw std::runtime_error("Invalid argument(s) for cuphyRunModulation");
    }

    time1 /= num_iterations;

    printf("Modulation Mapper Kernel: %.2f us (avg. over %d iterations)\n", time1 * 1000, num_iterations);


    std::vector<__half2> h_modulation_output(qam_elements);
    CUDA_CHECK(cudaMemcpy(h_modulation_output.data(), modulation_output.addr(), modulation_output.desc().get_size_in_bytes(), cudaMemcpyDeviceToHost));

    //Reference comparison
    uint32_t gpu_mismatch = 0;
    const __half tolerance = __float2half(0.0001f); //update tolerance as needed.
    for (int symbol_id = 0; symbol_id < qam_elements; symbol_id += 1) {
        __half2 ref_symbol;
        ref_symbol.x = (half) output_data({symbol_id}).x;
        ref_symbol.y = (half) output_data({symbol_id}).y;

        if (!complex_approx_equal<__half2, __half>(h_modulation_output[symbol_id], ref_symbol, tolerance)) {
            /*printf("Error! Mismatch for QAM symbol %d - expected=%f + i %f vs. gpu=%f + i %f\n", symbol_id,
                   (float) ref_symbol.x, (float) ref_symbol.y,
                   (float) h_modulation_output[symbol_id].x, (float) h_modulation_output[symbol_id].y);*/
            gpu_mismatch += 1;
        }
    }

    std::cout << "Found " << gpu_mismatch << " mismatched QAM symbols out of " << qam_elements << std::endl;

    if (gpu_mismatch != 0) {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "Reference check failed!");
        exit(1);
    }

    return 0;
}
