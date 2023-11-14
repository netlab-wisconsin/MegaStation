/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


/**
 * @brief Update PdschPerTbParams struct that tracks configuration information at per TB
 *        granularity. Check that configuration values are valid.
 * @param[in] cfg_rv: redundancy version
 * @param[in] cfg_Qm: modulation order
 * @param[in] cfg_bg: base graph
 * @param[in] cfg_Nl: number of layers per Tb (at most MAX_DL_LAYERS_PER_TB for downlink)
 * @param[in] cfg_num_CBs: number of code blocks
 * @param[in] cfg_Zc: lifting factor
 * @param[in] cfg_G: number of rated matched bits available for TB transmission
 * @param[in] cfg_F: number of filler bits
 * @param[in] cfg_cinit: seed used for scrambling sequence
 * @param[in] cfg_Nref: used to determine Ncb if smaller than N.
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT.
 */
cuphyStatus_t cuphySetTBParams(PdschPerTbParams * tb_params_struct, uint32_t cfg_rv, uint32_t cfg_Qm,
		               uint32_t cfg_bg, uint32_t cfg_Nl, uint32_t cfg_num_CBs,
		               uint32_t cfg_Zc, uint32_t cfg_G, uint32_t cfg_F, uint32_t cfg_cinit, uint32_t cfg_Nref);

/**
 * @brief Compute starting position, ko, for redundancy version rv as per
 *        Table 5.4.2.1-2 from the 3GPP reference.
 * @param[in] rv: redundancy version; [0, 3] valid range
 * @param[in] bg_num: base graph number; 1 or 2.
 * @param[in] Ncb: circular buffer length for LDPC encoder's output
 * @param[in] Zc: lifting size
 * @return starting position k0
 */
int compute_k0(int rv, int bg_num, int Ncb, int Zc);


/**
 * @brief Compute rate matching sequence length, in bits, for all C code blocks in a TB.
 @param[in, out] Er[]: will hold 2 uint32_t elements per-TB. First is the id of the CB_split for Er (num_CBs if no split), and second is
                       the smallest supported Er. The Er for CBs from CB_split and beyond is Er + Nl * Qm.
 * @param[in] C: number of code blocks
 * @param[in] Qm: modulation order
 * @param[in] Nl: number of transmission layers
 * @param[in] G: number of coded bits available for TB's transmission.
 * @param[in, out] Emax: set to true if Emax was updated
 * @param[in, out] Emax: maximum Er across all TBs processed in a kernel; divisible by word_size
 * @param[in] no_rate_matching: Er computed differently for TBs whose cells are in TM (testing mode)
 * @param[in] word_size: element size, in bits, for rate matching's input and output;
 *                       default = sizeof(uint32_t)*8 = 32
 */
void compute_rate_matching_length(uint32_t Er[], int C, int Qm, int Nl, int G, uint32_t & Emax, bool& updated_Emax, bool no_rate_matching=false, int word_size=32);
