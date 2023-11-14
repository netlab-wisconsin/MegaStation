/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#if !defined(PDSCH_DMRS_HPP_INCLUDED_)
#define PDSCH_DMRS_HPP_INCLUDED_

#if defined(__cplusplus)
extern "C" {
#endif /* defined(__cplusplus) */

struct pdschDmrsDescr
{
    PdschDmrsParams* dmrs_params;
    int num_TBs;
};
typedef struct pdschDmrsDescr pdschDmrsDescr_t;

struct pdschCsirsPrepDescr
{
    uint16_t*          reMapArray;
    int                bufferSizeInBytes;
    cuphyCsirsRrcDynPrm_t * csirsParams;
    int                numParams;
    uint32_t*          offsets;
    uint32_t           totalOffsets;
    uint32_t*          cellIndexArray;
    PdschUeGrpParams*  ueGrpParams;
    PdschDmrsParams*   dmrsParams;
    uint16_t           maxBWP;
};
typedef struct pdschCsirsPrepDescr pdschCsirsPrepDescr_t;

cuphyStatus_t CUPHYWINAPI cuphySetupPdschCsirsPreprocessing(cuphyPdschCsirsPrepLaunchConfig_t pdschCsirsLaunchConfig,
                                                          void*        re_map_array_addr,
                                                          cuphyCsirsRrcDynPrm_t* d_params,
                                                          size_t       numParams,
                                                          uint32_t     total_offsets,
                                                          uint32_t*    d_offsets,
                                                          uint32_t*    d_cellIndex,
                                                          uint16_t                num_ue_groups,
                                                          PdschUeGrpParams*       d_ue_grp_params,
                                                          PdschDmrsParams*        d_dmrs_params,
                                                          uint16_t                max_BWP,
                                                          uint16_t                num_cells,
                                                          void*                   cpu_desc,
                                                          void*                   gpu_desc,
                                                          uint8_t                 enable_desc_async_copy,
                                                          cudaStream_t stream);

#if defined(__cplusplus)
} /* extern "C" */
#endif /* defined(__cplusplus) */

#endif // PDSCH_DMRS_HPP_INCLUDED_
