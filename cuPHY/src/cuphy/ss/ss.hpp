/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#if !defined(SS_HPP_INCLUDED_)
#define SS_HPP_INCLUDED_

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

/**
 * @brief: Generate time/freq domain subcarriers for SSB pipeline.
 * @param[in]  d_x_tx: pointer to the PBCH rate matched output
 * @param[out] d_tfSignal: array of all num_cells output buffers; one buffer per cell
 * @param[in]  d_ssb_params: SSB parameters for all num_SSB SSBs
 * @param[in] d_per_cell_params: cell specific parameters for all num_cells cells
 * @param[in] num_SSBs: number of SSBs
 * @param[in] num_cells: number of cells
 * @param[in] stream: CUDA stream for kernel launch
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT or CUPHY_STATUS_INTERNAL_ERROR.
 */
cuphyStatus_t cuphyRunSsbMapper(const uint8_t*                  d_x_tx,
                                __half2**                       d_tfSignal,
                                const cuphyPerSsBlockDynPrms_t* d_ssb_params,
                                const cuphyPerCellSsbDynPrms_t* d_per_cell_params,
                                const cuphyPmWOneLayer_t*       d_pmw_params,
                                uint16_t                        num_SSBs,
                                uint16_t                        num_cells,
                                cudaStream_t                    stream,
                                cuphySsbMapperLaunchCfg_t*      pSsbMapperCfg);

/**
 * @brief: Set kernel launch configurations for SSB Mapper kernel
 * @param[in, out] pLaunchCfg: pointer to launch configuration for SSB mapper kernel
 * @param[in] num_SSBs: number of SSBs
 * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT.
 */
cuphyStatus_t kernelSelectSsbMapper(cuphySsbMapperLaunchCfg_t* pLaunchCfg,
                                    uint16_t                   num_SSBs);

#if defined(__cplusplus)
} /* extern "C" */
#endif /* defined(__cplusplus) */

#endif // SS_HPP_INCLUDED_
