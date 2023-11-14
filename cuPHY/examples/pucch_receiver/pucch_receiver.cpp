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

bool compare_approx(const float &a, const float &b) {
    const float tolerance = 0.0001f; // update tolerance as needed.
    float diff = abs(a - b);
    float m = std::max(abs(a), abs(b));
    float ratio = (diff >= tolerance) ? diff / m : diff;

    return (ratio <= tolerance);
}


bool complex_approx_equal(cuFloatComplex & a, cuFloatComplex & b) {
    return (compare_approx(a.x, b.x) && compare_approx(a.y, b.y));
}

int compare_intermediate_step3_results(PucchParams & pucch_params, int symbols,
                                       std::vector<cuFloatComplex> & gpu_values,
                                       hdf5hpp::hdf5_file & pucch_file,
                                       std::string dataset_name) {
    //std::cout << "dataset_name " << dataset_name << ": antennas " << pucch_params.num_bs_antennas << ", symbols " << symbols << std::endl;

    using tensor_pinned_C_32F = typed_tensor<CUPHY_C_32F, pinned_alloc>;
    tensor_pinned_C_32F matlab_ref_values = typed_tensor_from_dataset<CUPHY_C_32F, pinned_alloc>(pucch_file.open_dataset(dataset_name.c_str()));

    int cnt = 0;
    int mismatched_cnt = 0;
    for (int antenna = 0; antenna < pucch_params.num_bs_antennas; antenna++) {
        for (int time_index = 0; time_index < symbols; time_index++) {
            for (int freq_index = 0; freq_index < CUPHY_N_TONES_PER_PRB; freq_index++) {
                if (!complex_approx_equal(gpu_values[cnt], matlab_ref_values(freq_index, time_index, antenna), 0.0001f)) {
                    mismatched_cnt += 1;
                    std::cout << "Data mismatch for [" << cnt << "]. GPU computed data data: ";
                    std::cout << gpu_values[cnt].x << " + i " << gpu_values[cnt].y << std::endl;
                    std::cout << " vs. H5 ref " << matlab_ref_values(freq_index, time_index, antenna).x;
                    std::cout << " + i " << matlab_ref_values(freq_index, time_index, antenna).y << std::endl;
                }
                cnt += 1;
            }
        }
    }

    std::cout << "Dataset " << dataset_name << ": " << mismatched_cnt << " mismatches out of " << cnt << " elements." << std::endl;
    return mismatched_cnt;
}

void reference_comparison(PucchParams &pucch_params, cuFloatComplex * pucch_workspace,
                          void * bit_estimates,
                          hdf5hpp::hdf5_file & pucch_file) {

    int mult_factor = pucch_params.num_bs_antennas * CUPHY_N_TONES_PER_PRB;
    int pucch_data_elements = mult_factor * pucch_params.num_data_symbols;
    int pucch_dmrs_elements = mult_factor * pucch_params.num_dmrs_symbols;

    cuFloatComplex * d_ref_dmrs = pucch_workspace;
    cuFloatComplex * d_ref_data = d_ref_dmrs + pucch_dmrs_elements;

    int num_ues = pucch_params.num_pucch_ue;
    std::vector<uint32_t> gpu_bit_estimates(num_ues * 2);
    std::vector<cuFloatComplex> per_ue_data_qam(num_ues);

    // Copy output from GPU to CPU for reference comparison
    CUDA_CHECK(cudaMemcpy(gpu_bit_estimates.data(), bit_estimates, num_ues * 2 * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Also copy the per-UE sums, as an extra check, as widely different qam values can result in the same bit estimate.
    cuFloatComplex * d_per_ue_sums = d_ref_dmrs +  (pucch_dmrs_elements + pucch_data_elements);
    CUDA_CHECK(cudaMemcpy(per_ue_data_qam.data(), d_per_ue_sums, num_ues * sizeof(cuFloatComplex), cudaMemcpyDeviceToHost));

    // Check per-ue sums and b_estimates. Note, when pucch_receiver is run multiple times, only the results of the last iteration are compared.
    using tensor_pinned_R_32U = typed_tensor<CUPHY_R_32U, pinned_alloc>;
    using tensor_pinned_C_32F = typed_tensor<CUPHY_C_32F, pinned_alloc>;
    tensor_pinned_R_32U matlab_bit_estimates = typed_tensor_from_dataset<CUPHY_R_32U,
        pinned_alloc>(pucch_file.open_dataset("bit_est_step6"));

    int b_est_mismatch = 0;
    int qam_sum_mismatch = 0;
    for (int ue = 0; ue < num_ues; ue++) {
        for (int bit = 0; bit < 2; bit++) { // not all bits are used, but both matlab and gpu code set unused ones to zero
	    int matlab_val = matlab_bit_estimates(ue, bit);
	    int gpu_val = gpu_bit_estimates[ue * 2 + bit];
	    if (matlab_val != gpu_val) {
	        printf("Error! Mismatch for ue %d and bit %d - matlab=%d vs. gpu=%d\n", ue, bit, matlab_val, gpu_val);
		b_est_mismatch += 1;
	    }
	}

        //Compare sum for this ue
        stringstream qam_ue_name;
        qam_ue_name << "qam_est_ue_" << ue;
        tensor_pinned_C_32F matlab_per_ue_data_qam = typed_tensor_from_dataset<CUPHY_C_32F,
            pinned_alloc>(pucch_file.open_dataset(qam_ue_name.str().c_str()));

        if (!complex_approx_equal(per_ue_data_qam[ue], matlab_per_ue_data_qam(0), 0.0001f)) {
            printf("Error! Mismatch per-ue qam sum for ue %d - matlab=%f + i %f vs. gpu=%f + i %f\n", ue,
                   matlab_per_ue_data_qam(0).x, matlab_per_ue_data_qam(0).y,
                   per_ue_data_qam[ue].x, per_ue_data_qam[ue].y);
            qam_sum_mismatch += 1;
        }
    }

    std::cout << "Compared bit_estimates & qam_sums w/ matlab. Found " << b_est_mismatch;
    std::cout << " and " << qam_sum_mismatch << " mismatches respectively." << std::endl;
}


/** @brief: Update pucch_params with data read from HDF5 pucch_file.
 *  @param[in] pucch_params: PucchParams struct on the host.
 *  @param[in] pucch_file:   HDF5 file containing datasets for all required PUCCH params values.
 */
void update_pucch_params(PucchParams & pucch_params, hdf5hpp::hdf5_file & pucch_file) {

    using tensor_pinned_R_32U = typed_tensor<CUPHY_R_32U, pinned_alloc>;
    using tensor_pinned_R_32F = typed_tensor<CUPHY_R_32F, pinned_alloc>;

    pucch_params.format = CUPHY_PUCCH_FORMAT1; // only format currently supported

    tensor_pinned_R_32U start_symbol = typed_tensor_from_dataset<CUPHY_R_32U, pinned_alloc>(pucch_file.open_dataset("startSym"));
    pucch_params.start_symbol =  start_symbol(0);

    tensor_pinned_R_32U n_symbols = typed_tensor_from_dataset<CUPHY_R_32U, pinned_alloc>(pucch_file.open_dataset("nSym"));
    pucch_params.num_symbols = n_symbols(0);

    // Update # DMRS and data symbols for CUPHY_PUCCH_FORMAT 1. Derived values.
    pucch_params.num_dmrs_symbols = ceil(pucch_params.num_symbols * 1.0f/ 2);
    pucch_params.num_data_symbols = pucch_params.num_symbols - pucch_params.num_dmrs_symbols;

    tensor_pinned_R_32U PRB_index = typed_tensor_from_dataset<CUPHY_R_32U, pinned_alloc>(pucch_file.open_dataset("prbIdx"));
    pucch_params.PRB_index = PRB_index(0);

    tensor_pinned_R_32U low_PAPR_seq_index = typed_tensor_from_dataset<CUPHY_R_32U, pinned_alloc>(pucch_file.open_dataset("u"));
    pucch_params.low_PAPR_seq_index = low_PAPR_seq_index(0);

    tensor_pinned_R_32U num_bs_antennas = typed_tensor_from_dataset<CUPHY_R_32U, pinned_alloc>(pucch_file.open_dataset("L_BS"));
    pucch_params.num_bs_antennas = num_bs_antennas(0);

    tensor_pinned_R_32U numerology = typed_tensor_from_dataset<CUPHY_R_32U, pinned_alloc>(pucch_file.open_dataset("mu"));
    pucch_params.mu = numerology(0);

    tensor_pinned_R_32U slot_number = typed_tensor_from_dataset<CUPHY_R_32U, pinned_alloc>(pucch_file.open_dataset("slotNumber"));
    pucch_params.slot_number = slot_number(0);

    tensor_pinned_R_32U hopping_id = typed_tensor_from_dataset<CUPHY_R_32U, pinned_alloc>(pucch_file.open_dataset("hoppingId"));
    pucch_params.hopping_id = hopping_id(0);

    tensor_pinned_R_32U num_pucch_ue = typed_tensor_from_dataset<CUPHY_R_32U, pinned_alloc>(pucch_file.open_dataset("nUe_pucch"));
    pucch_params.num_pucch_ue = num_pucch_ue(0);
    if ((pucch_params.num_pucch_ue > MAX_UE_CNT) || (pucch_params.num_pucch_ue <= 0)) {
        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "Invalid number of PUCCH UEs {} Should be in (0, {}]", pucch_params.num_pucch_ue, MAX_UE_CNT);
        throw std::runtime_error("Invalid number of PUCCH UEs.");
    }

    // Process configuration parameters for per-UE processing
    tensor_pinned_R_32F Wf = typed_tensor_from_dataset<CUPHY_R_32F, pinned_alloc>(pucch_file.open_dataset("Wf"));

    stringstream wt_cell_name;
    wt_cell_name << "Wt_cell_"  << pucch_params.num_symbols - 4;
    tensor_pinned_R_32F Wt_cell = typed_tensor_from_dataset<CUPHY_R_32F, pinned_alloc>(pucch_file.open_dataset(wt_cell_name.str().c_str()));
    int wt_rows = ceil(pucch_params.num_symbols * 1.0f/ 2); // dmrs_symbols
    int wt_cols = pucch_params.num_symbols - wt_rows; // data_symbols
    for (int i = 0; i < CUPHY_N_TONES_PER_PRB; i++) {
        for (int j = 0; j < CUPHY_N_TONES_PER_PRB; j++) {
            pucch_params.Wf[i * CUPHY_N_TONES_PER_PRB + j] = Wf(i, j);
	    if ((i < wt_rows) && (j < wt_cols)) {
	        pucch_params.Wt_cell[j * wt_rows + i] = Wt_cell(i, j); // i is # dmrs symbols, j is # data symbols in Wt_cell; Wt_cell has column major order
	    }
        }
    }

    // Params to add: (a) tOCCidx, (b) nBits, (c) initial cyclic_shift_index
    for (int ue = 0; ue < pucch_params.num_pucch_ue; ue++) {
        PucchUeCellParams ue_cell_params;

        stringstream tocc_idx_ue_name;
        tocc_idx_ue_name << "pucch_ue_" << ue << "_tOCCidx";
        tensor_pinned_R_32U time_cover_index = typed_tensor_from_dataset<CUPHY_R_32U, pinned_alloc>(pucch_file.open_dataset(tocc_idx_ue_name.str().c_str()));
         ue_cell_params.time_cover_code_index = time_cover_index(0);

        stringstream nbits_ue_name;
        nbits_ue_name << "pucch_ue_" << ue << "_nBits";
        tensor_pinned_R_32U num_bits = typed_tensor_from_dataset<CUPHY_R_32U, pinned_alloc>(pucch_file.open_dataset(nbits_ue_name.str().c_str()));
        ue_cell_params.num_bits = num_bits(0);
        if ((ue_cell_params.num_bits != 1) && (ue_cell_params.num_bits != 2)) 
        {
            NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "Invalid number of bits for UE {}: {}. Should be either 1 or 2", ue, ue_cell_params.num_bits);
            throw std::runtime_error("Invalid number of bit for PUCCH UE.");
        }

        stringstream init_cs_ue_name;
        init_cs_ue_name << "pucch_ue_" << ue << "_cs0";
        tensor_pinned_R_32U cs0 = typed_tensor_from_dataset<CUPHY_R_32U, pinned_alloc>(pucch_file.open_dataset(init_cs_ue_name.str().c_str()));
        ue_cell_params.init_cyclic_shift_index = cs0(0);

        pucch_params.cell_params[ue] = ue_cell_params;
    }

}

void usage() {

    std::cout << "pucch_receiver [options]" << std::endl;
    std::cout << "  Options:" << std::endl;
    std::cout << "     -h                              (Display usage information)" << std::endl;
    std::cout << "     input_filename  num_iterations  (Input HDF5 filename, Number of iterations)" << std::endl;

    std::cout << std::endl;
    std::cout << "  Examples:" << std::endl;
    std::cout << "      ./pucch_receiver ~/TC1_pucch.h5 20" << std::endl;
    std::cout << "      ./pucch_receiver -h" << std::endl;
}

int main(int argc, char* argv[]) {

    if ((argc != 3) || ((argc == 2) && (argv[1][0] == '-') && (argv[1][1] == 'h'))) {
        usage();
        exit(1);
    }

    // Yan: temporarily disabled legacy PUCCH receiver
    printf("\n Legacy PUCCH receiver example is temporarily disabled.\n");
    return 0;

    cudaStream_t strm = 0;

    // Read input HDF5 file that contaings config params, input data, and intermediate results for reference comparison.
    std::string pucch_filename = "";
    pucch_filename.assign(argv[1]);

    int num_iterations = stoi(argv[2]);
    if (num_iterations <= 0) 
    {
        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT,  "Invalid number of iterations: {}. Should be > 0", num_iterations);
        exit(1);
    }

    hdf5hpp::hdf5_file pucch_file = hdf5hpp::hdf5_file::open(pucch_filename.c_str());
    cuphy::tensor_device pucch_data_rx = tensor_from_dataset(pucch_file.open_dataset("Y_input"), CUPHY_C_32F, cuphy::tensor_flags::align_tight, strm);

    // Set pucch_params from HDF5 file.
    PucchParams pucch_params;
    update_pucch_params(pucch_params, pucch_file);

    // Allocate workspace size: includes config parameters as well as space for intermediate results
    // Worspace will contain the following elements (in the listed order):
    // (a) d_ref_dmrs, (b) d_ref_data, (c) ue, (d) param.
    size_t pucch_workspace_size = cuphyPucchReceiverWorkspaceSize(pucch_params.num_pucch_ue,
                                                                pucch_params.num_bs_antennas,
                                                                pucch_params.num_symbols,
								CUPHY_C_32F); // in bytes
    buffer<cuFloatComplex, device_alloc> pucch_workspace_buffer(div_round_up(pucch_workspace_size, sizeof(cuFloatComplex)));

    // Copy PUCCH params to allocated workspace
    cuphyCopyPucchParamsToWorkspace(&pucch_params, pucch_workspace_buffer.addr(), CUPHY_C_32F);

    // Allocate output
    // TODO Potentially pack bit estimates into bits, rather than using an uint32_t per bit.
    tensor_device bit_estimates(CUPHY_R_32U,
                                pucch_params.num_pucch_ue,
                                2,
                                cuphy::tensor_flags::align_tight);

    // Run PUCCH receiver for num_iterations iterations and print timing output (if time_kernel is set).
    int time_kernel = 1;

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float time1 = 0.0;
    cudaEventRecord(start);

    for (int iter = 0; iter < num_iterations; iter++) {

        cuphyPucchReceiver(pucch_data_rx.desc().handle(), pucch_data_rx.addr(),
                           bit_estimates.desc().handle(), bit_estimates.addr(),
                           CUPHY_PUCCH_FORMAT1,
                           &pucch_params,
                           strm,
			   (void*)pucch_workspace_buffer.addr(),
			   pucch_workspace_size, CUPHY_C_32F);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time1, start, stop);

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    time1 /= num_iterations;

    if (time_kernel != 0) {
        printf("PUCCH Receiver Kernels: %.2f us (avg. over %d iterations)\n", time1 * 1000, num_iterations);
    }

    // Compare device and matlab reference values. Comparison for cuFloatComplex numbers is approximate.
    // Can update tolerance in compare_approx function as needed.
    reference_comparison(pucch_params, pucch_workspace_buffer.addr(), bit_estimates.addr(), pucch_file);

    return 0;
}
