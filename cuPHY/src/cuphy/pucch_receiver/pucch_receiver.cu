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
#include "cuphy_internal.h"
#include "cuphy.hpp"
#include <vector>
#include <iostream>
#include <cassert>
#include "tensor_desc.hpp"
#include "descrambling.hpp" // for POLY_* masks etc.
#include "descrambling.cuh"
#include "crc.hpp"
#include "type_convert.hpp"

#include "PUCCH_RECEIVER_F1_TOCC_VALUES_LUT.h"
#include "PUCCH_RECEIVER_F1_TIME_SHIFT_SEQ_VALUES_LUT.h"
#include "PUCCH_RECEIVER_F1_PAPR_SEQ_VALUES_LUT.h"

//#define ENABLE_DEBUG_PUCCH

using namespace cuphy_i;
using namespace descrambling; // for POLY_ etc.

void cuphyUpdatePucchParamsFormat1(PucchParams * pucch_params, const gnb_pars * gnb_params,
                                   const tb_pars * tb_params) {

    pucch_params->format = CUPHY_PUCCH_FORMAT1; // only format currently supported

    // Update with relevant info from the tb_pars struct.
    pucch_params->start_symbol = tb_params->startSym;
    pucch_params->num_symbols = tb_params->numSym;
    pucch_params->PRB_index = tb_params->startPrb;

    // Update with relevant info from the gnb_pars struct.
    pucch_params->num_bs_antennas = gnb_params->numBsAnt;
    pucch_params->mu = gnb_params->mu;
    pucch_params->slot_number = gnb_params->slotNumber;

    // Update # DMRS and data symbols for CUPHY_PUCCH_FORMAT 1. Derived values.
    pucch_params->num_dmrs_symbols = ceil(pucch_params->num_symbols * 1.0f/ 2);
    pucch_params->num_data_symbols = pucch_params->num_symbols - pucch_params->num_dmrs_symbols;
}

template<typename Tcomplex>
__device__ Tcomplex complex_mult(Tcomplex num1, Tcomplex num2);

template<>
__device__ cuFloatComplex complex_mult(cuFloatComplex num1, cuFloatComplex num2) {
    return cuCmulf(num1, num2);
};

template<>
__device__ __half2 complex_mult(__half2 num1, __half2 num2) {
    return __hmul2(num1, num2);
};


void check_pucch_dtype(cuphyDataType_t pucch_complex_data_type) {

    if ((pucch_complex_data_type != CUPHY_C_32F) && (pucch_complex_data_type != CUPHY_C_16F)) {
        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUPHY_EVENT, "Unsupported data type for PUCCH receiver.  Must be complex float or complex half");
        assert(false);
    }
}

size_t cuphyPucchReceiverWorkspaceSize(int num_ues, int num_bs_antennas, int num_symbols,
		                       cuphyDataType_t pucch_complex_data_type) {

    check_pucch_dtype(pucch_complex_data_type);
    int intermediate_buffer_elements = num_bs_antennas * num_symbols * CUPHY_N_TONES_PER_PRB + num_ues;
    int pucch_element_size = get_cuphy_type_storage_element_size(pucch_complex_data_type);
    return (sizeof(PucchParams) + intermediate_buffer_elements * pucch_element_size);
}

void cuphyCopyPucchParamsToWorkspace(const PucchParams * pucch_params, void* pucch_workspace,
		                     cuphyDataType_t pucch_complex_data_type) {

    check_pucch_dtype(pucch_complex_data_type);
    int pucch_elements = pucch_params->num_bs_antennas * pucch_params->num_symbols * CUPHY_N_TONES_PER_PRB;
    int num_ues = pucch_params->num_pucch_ue;

    // d_pucch_params's position in the workspace is implementation dependent.
    PucchParams * d_pucch_params = (pucch_complex_data_type == CUPHY_C_32F) ?
	                           (PucchParams *)(((cuFloatComplex *)pucch_workspace) + pucch_elements + num_ues) :
	                           (PucchParams *)(((__half2 *)pucch_workspace) + pucch_elements + num_ues);
    CUDA_CHECK(cudaMemcpy(d_pucch_params, pucch_params, sizeof(PucchParams), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaDeviceSynchronize());
}

/** @brief: Compute bit estimates from the per-UE data QAM
 *  @param[in] d_per_ue_sums: per-UE sums that make up data QAM; each element is a complex number.
 *  @param[in] pucch_params: PUCCH receiver config. parameters.
 *  @param[in, out] bit_estimates: array of bit-estimates; 2-bits per UE; second bit 0 when not used.
 */
template<typename Tcomplex, typename Tscalar>
__global__ void qam_to_bits(const Tcomplex * __restrict__ d_per_ue_sums,
                            const PucchParams * __restrict__ pucch_params,
	                    uint32_t * __restrict__ bit_estimates) {

    // bit_estimates has num_ues complex numbers
    if (threadIdx.x < pucch_params->num_pucch_ue) {
	Tcomplex per_ue_sum = d_per_ue_sums[threadIdx.x];
        //printf("ue %d has m_iue_sum = %f + i %f\n", threadIdx.x, per_ue_sum.x, per_ue_sum.y);
	int num_bits = pucch_params->cell_params[threadIdx.x].num_bits;
	int bit_index = threadIdx.x << 1;

	if (num_bits == 2) {
	    bit_estimates[bit_index] = (per_ue_sum.x <= (Tscalar) 0) ? 1 : 0;
	    bit_estimates[bit_index | 0x1ULL] = (per_ue_sum.y <= (Tscalar) 0) ? 1 : 0;
	} else { // num_bits == 1
        bit_estimates[bit_index] = ((per_ue_sum.x + per_ue_sum.y) <= (Tscalar) 0) ? 1 : 0;
        bit_estimates[bit_index | 0x1ULL] = 0; //unused, but CPU expects it to be 0 for comparison.
	}

	//printf("ue %d has bits %d and vals = [%d, %d]\n", threadIdx.x, num_bits, bit_estimates[2*threadIdx.x], bit_estimates[2*threadIdx.x + 1]);
    }
}


/** @brief: Compute cyclic shift index; used to remove user's code
 *  @param[in] pucch_params:    PUCCH receiver parameters
 *  @param[in] symbol_id:       PUCCH symbol Id
 */
__device__ uint32_t compute_cyclic_shift(const PucchParams * __restrict__ pucch_params, int symbol_id) {
    int symbol_offset = pucch_params->start_symbol + symbol_id;
    uint32_t cs0 = pucch_params->cell_params[blockIdx.y].init_cyclic_shift_index;
    uint32_t hopping_id = pucch_params->hopping_id;
    uint32_t cs = 0;
    for (int m = 0; m <= 7; m++) {
        int gold_index = (14 * 8 * pucch_params->slot_number) + 8 * symbol_offset + m;
        uint32_t gold_value = gold32(hopping_id, gold_index);
        uint32_t gold_bit = ((gold_value >> (gold_index & 31)) & 0x1);
        cs = cs + ((1 << m) * gold_bit);
        //cs(i) = cs(i) + 2^m * c(14*8*slotNumber + 8*(i - 1) + m);
    }
    uint32_t  cyclic_shift = (cs0 + cs) % 12;
    return cyclic_shift;

}


/** @brief: Remover user code, as part of per-UE processing. Called twice,
 *          once for dmrs and once for data.
 *  @param[in] d_step3_addr:    DMRS or data address of PUCCH signal after common processing.
 *  @param[in] pucch_params:    PUCCH receiver parameters
 *  @param[in] symbols:         number of dmrs or data symbols
 *  @param[in] symbol_id_offset: offset used in cyclic_shift_index; 0 for dmrs and 1 for data.
 *  @param[in] cyclic_shift_index: array in shared memory that stores the cyclic shift indices
 */
template<typename Tcomplex>
__device__ Tcomplex remove_user_codes(const Tcomplex * __restrict__ d_step3_addr,
                                      const PucchParams * __restrict__ pucch_params,
                                      int symbols, int symbol_id_offset,
                                      const uint32_t * cyclic_shift_index) {

    Tcomplex step4_val = make_complex<Tcomplex>::create(0.0f, 0.0f);
    int time_cover_code_index = pucch_params->cell_params[blockIdx.y].time_cover_code_index;
    int phi_row_start = time_cover_code_index * (OFDM_SYMBOLS_PER_SLOT >> 1);

    int symbol_id = threadIdx.x / CUPHY_N_TONES_PER_PRB;
    int freq_id = threadIdx.x % CUPHY_N_TONES_PER_PRB;

    uint32_t antenna_offset = blockIdx.x * symbols * CUPHY_N_TONES_PER_PRB;
    if (threadIdx.x < CUPHY_N_TONES_PER_PRB * symbols) {
           // Read orthogonal cover codes
           float * tocc_ptr = &TOCC_VALUES_LUT[symbols-1][(phi_row_start + symbol_id) << 1];
           Tcomplex conj_tocc_val = make_complex<Tcomplex>::create(*tocc_ptr, *(tocc_ptr + 1));

           uint32_t time_shift = cyclic_shift_index[(symbol_id << 1) + symbol_id_offset];

           // Compute frequency representation of cyclic shifts; could potentially store it in constant memory directly.
           float cs_val = -2.0f * (float) M_PI * freq_id * time_shift / CUPHY_N_TONES_PER_PRB;
	   Tcomplex cs_freq_val =
		   make_complex<Tcomplex>::create(cosf(cs_val), sinf(cs_val));
	   step4_val = complex_mult<Tcomplex>(complex_mult<Tcomplex>(conj_tocc_val, cs_freq_val),
			   d_step3_addr[antenna_offset + threadIdx.x]);
    }
    return step4_val;
}


/** @brief: Remove user code and then estimate channel and data. Produce data QAM.
 *          Kernel config: gridDim.x is # antennas, gridDim.y is # UEs.
 *  @param[in] d_step3_dmrs:        PUCCH DMRS signal after common processing.
 *  @param[in] d_step3_data:        PUCCH data signal after common processing.
 *  @param[in] pucch_params:        PUCCH config parameters.
 *  @param[in, out] d_per_ue_sums:  Estimated QAM data, an element per UE.
 */
template<typename Tcomplex, typename Tscalar>
__global__ void pucch_fused_per_ue_processing(const Tcomplex * __restrict__ d_step3_dmrs,
		                              const Tcomplex * __restrict__ d_step3_data,
                                              const PucchParams * __restrict__ pucch_params,
                                              Tcomplex * __restrict__ d_per_ue_sums) {

    __shared__ Tcomplex first_matrix_multiply[CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT / 2]; // overprovisioned; contains Wf * w/ Yiue_dmrs
    __shared__ Tcomplex tmp_h_iue[CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT / 2]; // overprovisioned
    __shared__ Tcomplex sum_m; //step 6
    __shared__ uint32_t cyclic_shift_index[OFDM_SYMBOLS_PER_SLOT];

    const int ue_id = blockIdx.y;

    const int dmrs_symbols = pucch_params->num_dmrs_symbols;
    const int data_symbols = pucch_params->num_data_symbols;
    int gnb_mu = pucch_params->mu;

    // Initialize shared memory
    if (threadIdx.x < CUPHY_N_TONES_PER_PRB * dmrs_symbols) { // shared mem init for step 5
        first_matrix_multiply[threadIdx.x] = make_complex<Tcomplex>::create(0.0f, 0.0f);
        tmp_h_iue[threadIdx.x] = make_complex<Tcomplex>::create(0.0f, 0.0f); // only CUPHY_N_TONES_PER_PRB * data_symbols are valid
    }

    if (threadIdx.x == 0) { // shared mem init for step 6
        sum_m = make_complex<Tcomplex>::create(0.0f, 0.0f);
    }

    if (threadIdx.x < pucch_params->num_symbols) {
        cyclic_shift_index[threadIdx.x] =  compute_cyclic_shift(pucch_params, threadIdx.x);
    }
    __syncthreads();

    int symbol_id = threadIdx.x / CUPHY_N_TONES_PER_PRB;
    int freq_id = threadIdx.x % CUPHY_N_TONES_PER_PRB;

    // Remove user code
    Tcomplex step4_dmrs_val = remove_user_codes<Tcomplex>(d_step3_dmrs, pucch_params, dmrs_symbols, 0, cyclic_shift_index);
    Tcomplex step4_data_val = remove_user_codes<Tcomplex>(d_step3_data, pucch_params, data_symbols, 1, cyclic_shift_index);

    float * time_shift_ptr = &TIME_SHIFT_SEQ_VALUES_LUT[gnb_mu][freq_id << 1];
    Tcomplex time_shift_val = make_complex<Tcomplex>::create(*time_shift_ptr, *(time_shift_ptr + 1));

    // intermediate result = Wf * Yiue_dmrs
    if (threadIdx.x < CUPHY_N_TONES_PER_PRB * dmrs_symbols) {
        for (int tone = 0; tone < CUPHY_N_TONES_PER_PRB; tone++) {
            int Wf_index = CUPHY_N_TONES_PER_PRB * tone + freq_id;
            Tscalar Wf_val = type_convert<Tscalar>(pucch_params->Wf[Wf_index]);
            atomicAdd(&first_matrix_multiply[symbol_id * CUPHY_N_TONES_PER_PRB + tone].x, step4_dmrs_val.x * Wf_val);
            atomicAdd(&first_matrix_multiply[symbol_id * CUPHY_N_TONES_PER_PRB + tone].y, step4_dmrs_val.y * Wf_val);
        }
    }
    __syncthreads();

    // H_iue = intermediate result * Wt
    // Doing a matrix matrix multiply of a CUPHY_N_TONES_PER_PRB x dmrs_symbols with a
    // dmrs_symbols * data_symbols array.
    if (threadIdx.x < CUPHY_N_TONES_PER_PRB * dmrs_symbols) { // Size of first_matrix_multiply; it has to be dmrs_symbols

        Tcomplex first_matrix_multiply_val = first_matrix_multiply[threadIdx.x];
        for (int Wt_symbol = 0; Wt_symbol < data_symbols; Wt_symbol++) {
            int Wt_index = dmrs_symbols * Wt_symbol + symbol_id;
            Tscalar Wt_val = type_convert<Tscalar>(pucch_params->Wt_cell[Wt_index]);
            atomicAdd(&tmp_h_iue[Wt_symbol * CUPHY_N_TONES_PER_PRB + freq_id].x, first_matrix_multiply_val.x * Wt_val);
            atomicAdd(&tmp_h_iue[Wt_symbol * CUPHY_N_TONES_PER_PRB + freq_id].y, first_matrix_multiply_val.y * Wt_val);
        }
    }
    __syncthreads();

    // Wrap up channel estimation and do data estimation
    if (threadIdx.x < CUPHY_N_TONES_PER_PRB * data_symbols) {
	//time shift; conj_tmp_h_est contains step5 output
        Tcomplex conj_tmp_h_est = complex_mult<Tcomplex>(make_complex<Tcomplex>::create(tmp_h_iue[threadIdx.x].x,
				                -tmp_h_iue[threadIdx.x].y),
                                                time_shift_val);

        Tcomplex m_iue = complex_mult<Tcomplex>(step4_data_val, conj_tmp_h_est);
        atomicAdd(&sum_m.x, m_iue.x);
        atomicAdd(&sum_m.y, m_iue.y);
    }
    __syncthreads();

    // Sum per ue across all antennas. Each UE will have a single complex number.
    if (threadIdx.x == 0) {
        atomicAdd(&d_per_ue_sums[ue_id].x, sum_m.x);
        atomicAdd(&d_per_ue_sums[ue_id].y, sum_m.y);
    }
}

#ifdef ENABLE_DEBUG_PUCCH
/** @brief: Print per UE config parameters: initial cyclic shift index, index of time cover code,
 *          number of transmitted bits.
 *  @param[in] pucch_params: pointer to PucchParams struct.
 */
__device__  void print_ue_params(const PucchParams * pucch_params) {

    if ((blockIdx.x == 0) && (threadIdx.x == 0)) {
        for (int ue = 0; ue < pucch_params->num_pucch_ue; ue++) {
           printf("ue %d, tOCC_idx %d, num_bits %d, cs0 %d\n", ue, pucch_params->cell_params[ue].time_cover_code_index,
                  pucch_params->cell_params[ue].num_bits, pucch_params->cell_params[ue].init_cyclic_shift_index);
        }
    }
    __syncthreads();
}
#endif

/** @brief: Common (i.e., same for all UEs) processing: extract PUCCH signal,
 *          remove cell code, center DMRS and separate dmrs and data.
 *  @param[in] Nf:
 *  @param[in] Nt:
 *  @param[in] num_ues: number of UEs
 *  @param[in] d_pucch_rx: received signal, allocated in antennas first, frequency second, time last format.
 *  @param[in] pucch_params: control channel configuration parameters, some are separate per UE.
 *  @param[in, out] d_ref_dmrs: DRMS signal; allocated in antennas first, freq. second, time last layout.
 *  @param[in, out] d_ref_data: data signal, allocated in antennas first, freq. second, time last layout.
 *  @param[in, out] d_per_ue_sums: per-UE sums that will make up data QAM; each element is a complex number.
 */
template<typename Tcomplex, typename Tscalar>
__global__ void pucch_common_processing(const int Nf, const int Nt, const int num_ues,
                                        const Tcomplex * __restrict__ d_pucch_rx,
                                        const PucchParams * __restrict__ pucch_params,
                                        Tcomplex * __restrict__ d_ref_dmrs,
                                        Tcomplex * __restrict__ d_ref_data,
                                        Tcomplex * __restrict__ d_per_ue_sums) {

#ifdef ENABLE_DEBUG_PUCCH
    print_ue_params(pucch_params);
#endif

    // Initialize per-ue sums to zero. Needed in pucch_fused_per_ue_processing kernel.
    if (blockIdx.x == 0)   {
        for (int i  = threadIdx.x; i < num_ues; i+= blockDim.x) {
            d_per_ue_sums[i] = make_complex<Tcomplex>::create(0.0f, 0.0f);
        }
    }

    //Each block works on an antenna; OFDM_SYMBOLS_PER_SLOT  is max number of symbols; overprovisioned.
    const size_t shmem_elements = CUPHY_N_TONES_PER_PRB * OFDM_SYMBOLS_PER_SLOT;
    __shared__ Tcomplex Y_pucch[shmem_elements];

    uint32_t antenna_offset = blockIdx.x * Nf * Nt;

    int total_symbols = pucch_params->num_symbols; // sum of dmrs and data symbols
    int time_index = threadIdx.x / CUPHY_N_TONES_PER_PRB;
    int freq_index = threadIdx.x % CUPHY_N_TONES_PER_PRB;
    int start_symbol = pucch_params->start_symbol;
    int PRB_index = pucch_params->PRB_index;
    int PAPR_seq_index = pucch_params->low_PAPR_seq_index;
    int gnb_mu = pucch_params->mu;


    // Initialize shared memory
    if (threadIdx.x < CUPHY_N_TONES_PER_PRB * total_symbols) {
        int freq_idx = (PRB_index * CUPHY_N_TONES_PER_PRB) +  freq_index;
        int time_idx = start_symbol + time_index;
        int index = antenna_offset + (time_idx *  Nf) +  freq_idx;
        Y_pucch[threadIdx.x] = d_pucch_rx[index];
    }
    __syncthreads();


   // Per block freq_index (which PRAQ or shift sequence element we modify) and time_index (DMRS iff time_index % 2 == 0)
   int not_dmrs = time_index & 0x1ULL; // set to 1 when time_index is odd
   int current_symbols = (not_dmrs) ? pucch_params->num_data_symbols : pucch_params->num_dmrs_symbols;
   int global_index = blockIdx.x * current_symbols * CUPHY_N_TONES_PER_PRB + (time_index >> 1) * CUPHY_N_TONES_PER_PRB + freq_index;

   // Separate dmrs from data.
   if (threadIdx.x < CUPHY_N_TONES_PER_PRB * total_symbols) {
      float * PAPR_ptr = &PAPR_SEQ_VALUES_LUT[PAPR_seq_index][freq_index << 1];
      Tcomplex conj_PAPR_val = make_complex<Tcomplex>::create(*PAPR_ptr, *(PAPR_ptr + 1));

      if (not_dmrs) {
         d_ref_data[global_index]  = complex_mult<Tcomplex>(Y_pucch[threadIdx.x], conj_PAPR_val);
      } else {
         float * time_shift_ptr = &TIME_SHIFT_SEQ_VALUES_LUT[gnb_mu][freq_index << 1];
         Tcomplex time_shift_val = make_complex<Tcomplex>::create(*time_shift_ptr, *(time_shift_ptr + 1));
         d_ref_dmrs[global_index]  = complex_mult<Tcomplex>(Y_pucch[threadIdx.x],
                                     complex_mult<Tcomplex>(conj_PAPR_val, time_shift_val));
      }
   }
}


template<typename Tcomplex, typename Tscalar>
void launch_templated_pucch_receiver_kernels(const cuphyTensorDescriptor_t data_rx_desc,
                                  const void* data_rx_addr,
                                  cuphyTensorDescriptor_t bit_estimates_desc, /* unused */
                                  void* bit_estimates_addr,
                                  const uint32_t pucch_format,
                                  const PucchParams * pucch_params,
                                  cudaStream_t strm,
				  void* pucch_workspace,
				  size_t allocated_pucch_workspace_size) {

    // PUCCH format check. Currently only PUCCH Format 1 is supported.
    assert(pucch_params->format == CUPHY_PUCCH_FORMAT1);

    // Get layout of input vector.
    const_tensor_pair pucch_data_rx_pair(static_cast<const tensor_desc&>(*data_rx_desc), data_rx_addr);
    const int Nf = pucch_data_rx_pair.first.get().layout().dimensions[0];
    const int Nt = pucch_data_rx_pair.first.get().layout().dimensions[1];


    /* Get device pointers within the allocated pucch_workspace memory pool.
       The order is as follows: d_ref_dmrs, d_ref_data, d_per_ue_sums, d_pucch_params.
       All but the last buffer are of cuFloatComplex type. */
    const int num_ues = pucch_params->num_pucch_ue;
    const int num_antennas = pucch_params->num_bs_antennas;
    const int num_symbols = pucch_params->num_symbols;
    const int num_dmrs_symbols = pucch_params->num_dmrs_symbols;
    const int pucch_data_elements = num_antennas * pucch_params->num_data_symbols * CUPHY_N_TONES_PER_PRB;
    const int pucch_dmrs_elements = num_antennas * num_dmrs_symbols * CUPHY_N_TONES_PER_PRB;
    const int WARP_SIZE = 32;

    Tcomplex * d_ref_dmrs = (Tcomplex *)pucch_workspace;
    Tcomplex * d_ref_data = d_ref_dmrs + pucch_dmrs_elements;
    Tcomplex * d_per_ue_sums = d_ref_data + pucch_data_elements;
    PucchParams * d_pucch_params = (PucchParams *)(d_per_ue_sums + num_ues);

    // Kernel Configurations. Note that max(num_antennas) = 16 and max(num_ues) = 42.
    dim3 common_grid_dim(num_antennas);
    dim3 ue_grid_dim(num_antennas, num_ues);

    dim3 common_block_dim(div_round_up(CUPHY_N_TONES_PER_PRB * num_symbols, WARP_SIZE) * WARP_SIZE);
    // ue_dmrs_block_dim suffices for fused per-UE kernel, as #data symbols is at most the same as # dmrs symbols
    // in PUCCH format 1
    dim3 ue_dmrs_block_dim(div_round_up(CUPHY_N_TONES_PER_PRB * num_dmrs_symbols, WARP_SIZE) * WARP_SIZE);
    dim3 rounded_ues = div_round_up(num_ues, WARP_SIZE) * WARP_SIZE;


    // Common, across all UEs, processing for pucch signal (input).
    pucch_common_processing<Tcomplex, Tscalar><<<common_grid_dim, common_block_dim, 0, strm>>>(Nf,
                                                     Nt,
                                                     num_ues,
                                                     (Tcomplex *)data_rx_addr,
                                                     d_pucch_params,
                                                     d_ref_dmrs,
                                                     d_ref_data,
                                                     d_per_ue_sums);

    // Per-UE processing

    // TODO Potentially revisit implementation (e.g., consider a block per UE to reduce kernel launch
    // overhead.
    pucch_fused_per_ue_processing<Tcomplex, Tscalar><<<ue_grid_dim, ue_dmrs_block_dim, 0, strm>>>(d_ref_dmrs,
			                      d_ref_data,
                                              d_pucch_params,
                                              d_per_ue_sums);

    // TODO Potentially pack bit estimates into bits, instead of using an uint32_t
    // for each bit.
    qam_to_bits<Tcomplex, Tscalar><<<1, rounded_ues, 0, strm>>>(d_per_ue_sums,
                            d_pucch_params,
	                    (uint32_t *) bit_estimates_addr);

    cudaError_t cuda_error = cudaGetLastError();
    if (cuda_error != cudaSuccess) {
        NVLOGE_FMT(NVLOG_PUCCH, AERIAL_CUDA_API_EVENT, "CUDA ERROR: {}",cudaGetErrorString(cuda_error));
    }

}


void cuphyPucchReceiver(const cuphyTensorDescriptor_t data_rx_desc,
                        const void* data_rx_addr,
                        cuphyTensorDescriptor_t bit_estimates_desc, /* unused */
                        void* bit_estimates_addr,
                        const uint32_t pucch_format,
                        const PucchParams * pucch_params,
                        cudaStream_t strm,
		        void* pucch_workspace,
		        size_t allocated_pucch_workspace_size,
		        cuphyDataType_t pucch_complex_data_type) {

    check_pucch_dtype(pucch_complex_data_type);

    if (pucch_complex_data_type == CUPHY_C_32F) {
	launch_templated_pucch_receiver_kernels<cuFloatComplex, float>(data_rx_desc, data_rx_addr, bit_estimates_desc,
			                        bit_estimates_addr, pucch_format, pucch_params, strm, pucch_workspace,
						allocated_pucch_workspace_size);
    } else if (pucch_complex_data_type == CUPHY_C_16F) {
	launch_templated_pucch_receiver_kernels<struct __half2, __half>(data_rx_desc, data_rx_addr, bit_estimates_desc,
			                        bit_estimates_addr, pucch_format, pucch_params, strm, pucch_workspace,
						allocated_pucch_workspace_size);
   }
}
