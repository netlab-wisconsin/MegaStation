/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <cstddef>
#include <string>
#include "pusch_rx.hpp"
#include "cuphy.hpp"
#include "cuphy_api.h"
#include "util.hpp"
#include "convert_tensor.cuh"

#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"


// #include "graph_group_count.h"
// #define USE_NVTX 1
//#define MEMTRACE      //FixMe uncomment to enable memtrace  in cuPHY runs, but note that call to memtrace_set_config(0);
//                      //will disable mem tracing on that thread onward

// Enable multi-stream kernel launches in Pusch pipeline using m_G1streamPool and m_G2streamPool
#define PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH

void PuschRx::allocateDescr()
{
    // zero-initialize
    std::array<size_t, N_PUSCH_DESCR_TYPES> statDescrSizeBytes{};
    std::array<size_t, N_PUSCH_DESCR_TYPES> statDescrAlignBytes{};
    std::array<size_t, N_PUSCH_DESCR_TYPES> dynDescrSizeBytes{};
    std::array<size_t, N_PUSCH_DESCR_TYPES> dynDescrAlignBytes{};

    size_t* pStatDescrSizeBytes  = statDescrSizeBytes.data();
    size_t* pStatDescrAlignBytes = statDescrAlignBytes.data();
    size_t* pDynDescrSizeBytes   = dynDescrSizeBytes.data();
    size_t* pDynDescrAlignBytes  = dynDescrAlignBytes.data();

    cuphyStatus_t status = cuphyPuschRxChEstGetDescrInfo(&pStatDescrSizeBytes[PUSCH_CH_EST],
                                                         &pStatDescrAlignBytes[PUSCH_CH_EST],
                                                         &pDynDescrSizeBytes[PUSCH_CH_EST],
                                                         &pDynDescrAlignBytes[PUSCH_CH_EST]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyPuschRxChEstGetDescrInfo()");
    }

    status = cuphyPuschRxNoiseIntfEstGetDescrInfo(&pDynDescrSizeBytes[PUSCH_NOISE_INTF_EST],
                                                  &pDynDescrAlignBytes[PUSCH_NOISE_INTF_EST]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyPuschRxNoiseIntfEstGetDescrInfo()");
    }

    for(uint32_t chEstTimeIdx = 1; chEstTimeIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEstTimeIdx)
    {
        pStatDescrSizeBytes[PUSCH_CH_EST + chEstTimeIdx]  = pStatDescrSizeBytes[PUSCH_CH_EST];
        pStatDescrAlignBytes[PUSCH_CH_EST + chEstTimeIdx] = pStatDescrAlignBytes[PUSCH_CH_EST];
        pDynDescrSizeBytes[PUSCH_CH_EST + chEstTimeIdx]   = pDynDescrSizeBytes[PUSCH_CH_EST];
        pDynDescrAlignBytes[PUSCH_CH_EST + chEstTimeIdx]  = pDynDescrAlignBytes[PUSCH_CH_EST];
    }

    status = cuphyPuschRxCfoTaEstGetDescrInfo(&pStatDescrSizeBytes[PUSCH_CFO_TA_EST],
                                              &pStatDescrAlignBytes[PUSCH_CFO_TA_EST],
                                              &pDynDescrSizeBytes[PUSCH_CFO_TA_EST],
                                              &pDynDescrAlignBytes[PUSCH_CFO_TA_EST]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyPuschRxCfoTaEstGetDescrInfo()");
    }

    status = cuphyPuschRxChEqGetDescrInfo(&pStatDescrSizeBytes[PUSCH_CH_EQ_COEF], // same stat descriptor is reused for Soft-demap as well
                                          &pStatDescrAlignBytes[PUSCH_CH_EQ_COEF],
                                          &pDynDescrSizeBytes[PUSCH_CH_EQ_COEF],
                                          &pDynDescrAlignBytes[PUSCH_CH_EQ_COEF],
                                          &pDynDescrSizeBytes[PUSCH_CH_EQ_SOFT_DEMAP],
                                          &pDynDescrAlignBytes[PUSCH_CH_EQ_SOFT_DEMAP]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyPuschRxChEqGetDescrInfo()");
    }

    for(uint32_t chEqInstIdx = 1; chEqInstIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ; ++chEqInstIdx)
    {
        pStatDescrSizeBytes[PUSCH_CH_EQ_COEF + chEqInstIdx]  = pStatDescrSizeBytes[PUSCH_CH_EQ_COEF];
        pStatDescrAlignBytes[PUSCH_CH_EQ_COEF + chEqInstIdx] = pStatDescrAlignBytes[PUSCH_CH_EQ_COEF];
        pDynDescrSizeBytes[PUSCH_CH_EQ_COEF + chEqInstIdx]   = pDynDescrSizeBytes[PUSCH_CH_EQ_COEF];
        pDynDescrAlignBytes[PUSCH_CH_EQ_COEF + chEqInstIdx]  = pDynDescrAlignBytes[PUSCH_CH_EQ_COEF];
    }

    status = cuphyPuschRxRateMatchGetDescrInfo(&pDynDescrSizeBytes[PUSCH_RATE_MATCH], &pDynDescrAlignBytes[PUSCH_RATE_MATCH]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyPuschRxRateMatchGetDescrInfo()");
    }

    status = cuphyPuschRxCrcDecodeGetDescrInfo(&pDynDescrSizeBytes[PUSCH_CRC], &pDynDescrAlignBytes[PUSCH_CRC]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyPuschRxCrcDecodeGetDescrInfo()");
    }

    status = cuphyPuschRxRssiGetDescrInfo(&pDynDescrSizeBytes[PUSCH_RSSI], &pDynDescrAlignBytes[PUSCH_RSSI], &pDynDescrSizeBytes[PUSCH_RSRP], &pDynDescrAlignBytes[PUSCH_RSRP]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyPuschRxRssiGetDescrInfo()");
    }

    status = cuphyUciOnPuschSegLLRs0GetDescrInfo(&pDynDescrSizeBytes[PUSCH_SEG_UCI_LLRS0], &pDynDescrAlignBytes[PUSCH_SEG_UCI_LLRS0]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyUciOnPuschSegLLRs0GetDescrInfo()");
    }

    status = cuphyUciOnPuschSegLLRs0GetDescrInfo(&pDynDescrSizeBytes[PUSCH_SEG_EARLY_UCI_LLRS0], &pDynDescrAlignBytes[PUSCH_SEG_EARLY_UCI_LLRS0]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyUciOnPuschEarlySegLLRs0GetDescrInfo()");
    }

    status = cuphyUciOnPuschSegLLRs2GetDescrInfo(&pDynDescrSizeBytes[PUSCH_SEG_UCI_LLRS2], &pDynDescrAlignBytes[PUSCH_SEG_UCI_LLRS2]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyUciOnPuschSegLLRs2GetDescrInfo()");
    }

    status = cuphyUciOnPuschCsi2CtrlGetDescrInfo(&pDynDescrSizeBytes[PUSCH_UCI_CSI2_CTRL], &pDynDescrAlignBytes[PUSCH_UCI_CSI2_CTRL]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyUciOnPuschCsi2CtrlGetDescrInfo()");
    }

    status = cuphyCompCwTreeTypesGetDescrInfo(&pDynDescrSizeBytes[POL_COMP_CW_TREE], &pDynDescrAlignBytes[POL_COMP_CW_TREE]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCompCwTreeTypesGetDescrInfo()");
    }

    pDynDescrSizeBytes[POL_COMP_CW_TREE_ADDRS]  = sizeof(uint8_t*) * CUPHY_MAX_N_POL_UCI_SEGS * m_cuphyPuschStatPrms.nMaxCellsPerSlot;
    pDynDescrAlignBytes[POL_COMP_CW_TREE_ADDRS] = alignof(uint8_t*);

    status = cuphyCompCwTreeTypesGetDescrInfo(&pDynDescrSizeBytes[POL_COMP_CW_TREE_EARLY], &pDynDescrAlignBytes[POL_COMP_CW_TREE_EARLY]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCompCwTreeTypesGetDescrInfo_early()");
    }

    pDynDescrSizeBytes[POL_COMP_CW_TREE_ADDRS_EARLY]  = sizeof(uint8_t*) * CUPHY_MAX_N_POL_UCI_SEGS * m_cuphyPuschStatPrms.nMaxCellsPerSlot;
    pDynDescrAlignBytes[POL_COMP_CW_TREE_ADDRS_EARLY] = alignof(uint8_t*);

    status = cuphyPolSegDeRmDeItlGetDescrInfo(&pDynDescrSizeBytes[POL_SEG_DERM_DEITL], &pDynDescrAlignBytes[POL_SEG_DERM_DEITL]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyPolSegDeRmDeItlGetDescrInfo()");
    }

    pDynDescrSizeBytes[POL_SEG_DERM_DEITL_CW_ADDRS]  = sizeof(__half*) * CUPHY_MAX_N_POL_UCI_SEGS * m_cuphyPuschStatPrms.nMaxCellsPerSlot;
    pDynDescrAlignBytes[POL_SEG_DERM_DEITL_CW_ADDRS] = alignof(__half*);

    pDynDescrSizeBytes[POL_SEG_DERM_DEITL_UCI_ADDRS]  = sizeof(__half*) * CUPHY_MAX_N_POL_UCI_SEGS * m_cuphyPuschStatPrms.nMaxCellsPerSlot;
    pDynDescrAlignBytes[POL_SEG_DERM_DEITL_UCI_ADDRS] = alignof(__half*);

    status = cuphyPolSegDeRmDeItlGetDescrInfo(&pDynDescrSizeBytes[POL_SEG_DERM_DEITL_EARLY], &pDynDescrAlignBytes[POL_SEG_DERM_DEITL_EARLY]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyPolSegDeRmDeItlGetDescrInfo()");
    }

    pDynDescrSizeBytes[POL_SEG_DERM_DEITL_CW_ADDRS_EARLY]  = sizeof(__half*) * CUPHY_MAX_N_POL_UCI_SEGS * m_cuphyPuschStatPrms.nMaxCellsPerSlot;
    pDynDescrAlignBytes[POL_SEG_DERM_DEITL_CW_ADDRS_EARLY] = alignof(__half*);

    pDynDescrSizeBytes[POL_SEG_DERM_DEITL_UCI_ADDRS_EARLY]  = sizeof(__half*) * CUPHY_MAX_N_POL_UCI_SEGS * m_cuphyPuschStatPrms.nMaxCellsPerSlot;
    pDynDescrAlignBytes[POL_SEG_DERM_DEITL_UCI_ADDRS_EARLY] = alignof(__half*);

    status = cuphyPolarDecoderGetDescrInfo(&pDynDescrSizeBytes[POL_DECODE], &pDynDescrAlignBytes[POL_DECODE]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyPolarDecoderGetDescrInfo()");
    }

    pDynDescrSizeBytes[POL_DECODE_LLR_ADDRS]  = sizeof(__half*) * CUPHY_MAX_N_POL_CWS * m_cuphyPuschStatPrms.nMaxCellsPerSlot;
    pDynDescrAlignBytes[POL_DECODE_LLR_ADDRS] = alignof(__half*);

    pDynDescrSizeBytes[POL_DECODE_CB_ADDRS]  = sizeof(uint32_t*) * CUPHY_MAX_N_POL_CWS * m_cuphyPuschStatPrms.nMaxCellsPerSlot;
    pDynDescrAlignBytes[POL_DECODE_CB_ADDRS] = alignof(uint32_t*);

    pDynDescrSizeBytes[LIST_POL_DECODE_SCRATCH_ADDRS]  = sizeof(bool*) * CUPHY_MAX_N_POL_CWS * m_cuphyPuschStatPrms.nMaxCellsPerSlot;
    pDynDescrAlignBytes[LIST_POL_DECODE_SCRATCH_ADDRS] = alignof(bool*);

    status = cuphyPolarDecoderGetDescrInfo(&pDynDescrSizeBytes[POL_DECODE_EARLY], &pDynDescrAlignBytes[POL_DECODE_EARLY]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyPolarDecoderGetDescrInfo()");
    }

    pDynDescrSizeBytes[POL_DECODE_LLR_ADDRS_EARLY]  = sizeof(__half*) * CUPHY_MAX_N_POL_CWS * m_cuphyPuschStatPrms.nMaxCellsPerSlot;
    pDynDescrAlignBytes[POL_DECODE_LLR_ADDRS_EARLY] = alignof(__half*);

    pDynDescrSizeBytes[POL_DECODE_CB_ADDRS_EARLY]  = sizeof(uint32_t*) * CUPHY_MAX_N_POL_CWS * m_cuphyPuschStatPrms.nMaxCellsPerSlot;
    pDynDescrAlignBytes[POL_DECODE_CB_ADDRS_EARLY] = alignof(uint32_t*);

    pDynDescrSizeBytes[LIST_POL_DECODE_SCRATCH_ADDRS_EARLY]  = sizeof(bool*) * CUPHY_MAX_N_POL_CWS * m_cuphyPuschStatPrms.nMaxCellsPerSlot;
    pDynDescrAlignBytes[LIST_POL_DECODE_SCRATCH_ADDRS_EARLY] = alignof(bool*);

    status = cuphySimplexDecoderGetDescrInfo(&pDynDescrSizeBytes[SPX_DECODE], &pDynDescrAlignBytes[SPX_DECODE]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphySimplexDecoderGetDescrInfo()");
    }

    status = cuphyRmDecoderGetDescrInfo(&pDynDescrSizeBytes[RM_DECODE], &pDynDescrAlignBytes[RM_DECODE]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyRmDecoderGetDescrInfo()");
    }

    status = cuphySimplexDecoderGetDescrInfo(&pDynDescrSizeBytes[SPX_DECODE_CSI2], &pDynDescrAlignBytes[SPX_DECODE_CSI2]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphySimplexDecoderGetDescrInfo_csi2()");
    }

    status = cuphyRmDecoderGetDescrInfo(&pDynDescrSizeBytes[RM_DECODE_CSI2], &pDynDescrAlignBytes[RM_DECODE_CSI2]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyRmDecoderGetDescrInfo_csi2()");
    }

    status = cuphySimplexDecoderGetDescrInfo(&pDynDescrSizeBytes[SPX_DECODE_EARLY], &pDynDescrAlignBytes[SPX_DECODE_EARLY]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphySimplexDecoderGetDescrInfo_early()");
    }

    status = cuphyRmDecoderGetDescrInfo(&pDynDescrSizeBytes[RM_DECODE_EARLY], &pDynDescrAlignBytes[RM_DECODE_EARLY]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyRmDecoderGetDescrInfo_early()");
    }

    pDynDescrSizeBytes[PUSCH_FRONT_END_PARAMS]  = sizeof(cuphyPuschRxUeGrpPrms_t) * MAX_N_USER_GROUPS_SUPPORTED;
    pDynDescrAlignBytes[PUSCH_FRONT_END_PARAMS] = alignof(cuphyPuschRxUeGrpPrms_t);

    if(m_cuphyPuschStatPrms.pDbg->forcedNumCsi2Bits > 0)
    {
        status = cuphyCompCwTreeTypesGetDescrInfo(&pDynDescrSizeBytes[POL_COMP_CW_TREE_CSI2], &pDynDescrAlignBytes[POL_COMP_CW_TREE_CSI2]);
        if(CUPHY_STATUS_SUCCESS != status)
        {
            throw cuphy::cuphy_fn_exception(status, "cuphyCompCwTreeTypesGetDescrInfo()");
        }

        pDynDescrSizeBytes[POL_COMP_CW_TREE_ADDRS_CSI2]  = sizeof(uint8_t*) * CUPHY_MAX_N_POL_UCI_SEGS_CSI2 * m_cuphyPuschStatPrms.nMaxCellsPerSlot;
        pDynDescrAlignBytes[POL_COMP_CW_TREE_ADDRS_CSI2] = alignof(uint8_t*);

        status = cuphyPolSegDeRmDeItlGetDescrInfo(&pDynDescrSizeBytes[POL_SEG_DERM_DEITL_CSI2], &pDynDescrAlignBytes[POL_SEG_DERM_DEITL_CSI2]);
        if(CUPHY_STATUS_SUCCESS != status)
        {
            throw cuphy::cuphy_fn_exception(status, "cuphyPolSegDeRmDeItlGetDescrInfo()");
        }

        pDynDescrSizeBytes[POL_SEG_DERM_DEITL_CW_ADDRS_CSI2]  = sizeof(__half*) * CUPHY_MAX_N_POL_CWS_CSI2 * m_cuphyPuschStatPrms.nMaxCellsPerSlot;
        pDynDescrAlignBytes[POL_SEG_DERM_DEITL_CW_ADDRS_CSI2] = alignof(__half*);

        pDynDescrSizeBytes[POL_SEG_DERM_DEITL_UCI_ADDRS_CSI2]  = sizeof(__half*) * CUPHY_MAX_N_POL_CWS_CSI2 * m_cuphyPuschStatPrms.nMaxCellsPerSlot;
        pDynDescrAlignBytes[POL_SEG_DERM_DEITL_UCI_ADDRS_CSI2] = alignof(__half*);

        status = cuphyPolarDecoderGetDescrInfo(&pDynDescrSizeBytes[POL_DECODE_CSI2], &pDynDescrAlignBytes[POL_DECODE_CSI2]);
        if(CUPHY_STATUS_SUCCESS != status)
        {
            throw cuphy::cuphy_fn_exception(status, "cuphyPolarDecoderGetDescrInfo()");
        }

        pDynDescrSizeBytes[POL_DECODE_LLR_ADDRS_CSI2]  = sizeof(__half*) * CUPHY_MAX_N_POL_CWS_CSI2 * m_cuphyPuschStatPrms.nMaxCellsPerSlot;
        pDynDescrAlignBytes[POL_DECODE_LLR_ADDRS_CSI2] = alignof(__half*);

        pDynDescrSizeBytes[POL_DECODE_CB_ADDRS_CSI2]  = sizeof(uint32_t*) * CUPHY_MAX_N_POL_CWS_CSI2 * m_cuphyPuschStatPrms.nMaxCellsPerSlot;
        pDynDescrAlignBytes[POL_DECODE_CB_ADDRS_CSI2] = alignof(uint32_t*);

        pDynDescrSizeBytes[LIST_POL_DECODE_SCRATCH_ADDRS_CSI2]  = sizeof(bool*) * CUPHY_MAX_N_POL_CWS_CSI2 * m_cuphyPuschStatPrms.nMaxCellsPerSlot;
        pDynDescrAlignBytes[LIST_POL_DECODE_SCRATCH_ADDRS_CSI2] = alignof(bool*);
    }

    // Allocate descriptor
    m_kernelStatDescr.alloc(statDescrSizeBytes, statDescrAlignBytes, &m_memoryFootprint);
    // m_kernelStatDescr.displayDescrSizes();

    m_kernelDynDescr.alloc(dynDescrSizeBytes, dynDescrAlignBytes, &m_memoryFootprint);
    // m_kernelDynDescr.displayDescrSizes();

}

void PuschRx::allocateInputBuf(uint32_t nMaxTbs, uint32_t maxNumTbsSupported)
{
    size_t* pInputBufSizeBytes  = m_inputBufSizeBytes.data();
    size_t* pInputBufAlignBytes = m_inputBufAlignBytes.data();

    pInputBufSizeBytes[PUSCH_TB_PRMS]  = nMaxTbs == 0 ? sizeof(PerTbParams) * maxNumTbsSupported : sizeof(PerTbParams) * nMaxTbs;
    pInputBufAlignBytes[PUSCH_TB_PRMS] = alignof(PerTbParams);

    pInputBufSizeBytes[PUSCH_SPX_PRMS]  = sizeof(cuphySimplexCwPrm_t) * CUPHY_MAX_N_SPX_CWS; //m_nSpxCws
    pInputBufAlignBytes[PUSCH_SPX_PRMS] = alignof(cuphySimplexCwPrm_t);

    pInputBufSizeBytes[PUSCH_RM_CW_PRMS]  = sizeof(cuphyRmCwPrm_t) * CUPHY_MAX_N_SPX_CWS;//m_nRmCws;
    pInputBufAlignBytes[PUSCH_RM_CW_PRMS] = alignof(cuphyRmCwPrm_t);

    pInputBufSizeBytes[PUSCH_UCI_SEG_PRMS]  = sizeof(cuphyPolarUciSegPrm_t) * CUPHY_MAX_N_POL_UCI_SEGS; //m_nPolUciSegs;
    pInputBufAlignBytes[PUSCH_UCI_SEG_PRMS] = alignof(cuphyPolarUciSegPrm_t);
    pInputBufSizeBytes[PUSCH_UCI_CW_PRMS]   = sizeof(cuphyPolarCwPrm_t) * CUPHY_MAX_N_POL_CWS; //m_nPolCbs;
    pInputBufAlignBytes[PUSCH_UCI_CW_PRMS]  = alignof(cuphyPolarCwPrm_t);
 
    if(m_cuphyPuschStatPrms.pDbg->forcedNumCsi2Bits > 0)
    {
        pInputBufSizeBytes[PUSCH_UCI_SEG_CSI2_PRMS]  = sizeof(cuphyPolarUciSegPrm_t) * CUPHY_MAX_N_PUSCH_CSI2 ; 
        pInputBufAlignBytes[PUSCH_UCI_SEG_CSI2_PRMS] = alignof(cuphyPolarUciSegPrm_t);
        pInputBufSizeBytes[PUSCH_UCI_CW_CSI2_PRMS]   = sizeof(cuphyPolarCwPrm_t) * CUPHY_MAX_N_PUSCH_CSI2 * 2; 
        pInputBufAlignBytes[PUSCH_UCI_CW_CSI2_PRMS]  = alignof(cuphyPolarCwPrm_t);
    }

    pInputBufSizeBytes[PUSCH_SPX_CSI2_PRMS]    = sizeof(cuphySimplexCwPrm_t) * CUPHY_MAX_N_PUSCH_CSI2 ; //m_nCsi2Ues;
    pInputBufAlignBytes[PUSCH_SPX_CSI2_PRMS]   = alignof(cuphySimplexCwPrm_t);
    pInputBufSizeBytes[PUSCH_RM_CW_CSI2_PRMS]  = sizeof(cuphyRmCwPrm_t) * CUPHY_MAX_N_PUSCH_CSI2 ; //m_nCsi2Ues;
    pInputBufAlignBytes[PUSCH_RM_CW_CSI2_PRMS] = alignof(cuphyRmCwPrm_t);

    pInputBufSizeBytes[PUSCH_SPX_EARLY_PRMS]  = sizeof(cuphySimplexCwPrm_t) * CUPHY_MAX_N_SPX_CWS; //m_nSpxCws
    pInputBufAlignBytes[PUSCH_SPX_EARLY_PRMS] = alignof(cuphySimplexCwPrm_t);

    pInputBufSizeBytes[PUSCH_RM_CW_EARLY_PRMS]  = sizeof(cuphyRmCwPrm_t) * CUPHY_MAX_N_SPX_CWS;//m_nRmCws;
    pInputBufAlignBytes[PUSCH_RM_CW_EARLY_PRMS] = alignof(cuphyRmCwPrm_t);

    pInputBufSizeBytes[PUSCH_UCI_SEG_EARLY_PRMS]  = sizeof(cuphyPolarUciSegPrm_t) * CUPHY_MAX_N_POL_UCI_SEGS; //m_nPolUciSegs;
    pInputBufAlignBytes[PUSCH_UCI_SEG_EARLY_PRMS] = alignof(cuphyPolarUciSegPrm_t);
    pInputBufSizeBytes[PUSCH_UCI_CW_EARLY_PRMS]   = sizeof(cuphyPolarCwPrm_t) * CUPHY_MAX_N_POL_CWS; //m_nPolCbs;
    pInputBufAlignBytes[PUSCH_UCI_CW_EARLY_PRMS]  = alignof(cuphyPolarCwPrm_t);

    // Allocate input buffer
    m_h2dBuffer.alloc(m_inputBufSizeBytes, m_inputBufAlignBytes, &m_memoryFootprint);

    // update pointers in pusch
    auto h2dBufCpuStartAddrs = m_h2dBuffer.getCpuStartAddrs();

    m_pTbPrmsCpu           = reinterpret_cast<PerTbParams*>(h2dBufCpuStartAddrs[PUSCH_TB_PRMS]);
    m_pSpxCwPrmsCpu        = reinterpret_cast<cuphySimplexCwPrm_t*>(h2dBufCpuStartAddrs[PUSCH_SPX_PRMS]);
    m_pRmCwPrmsCpu         = reinterpret_cast<cuphyRmCwPrm_t*>(h2dBufCpuStartAddrs[PUSCH_RM_CW_PRMS]);
    m_pUciSegPrmsCpu       = reinterpret_cast<cuphyPolarUciSegPrm_t*>(h2dBufCpuStartAddrs[PUSCH_UCI_SEG_PRMS]);
    m_pUciCwPrmsCpu        = reinterpret_cast<cuphyPolarCwPrm_t*>(h2dBufCpuStartAddrs[PUSCH_UCI_CW_PRMS]);
    m_pSpxCwPrmsCpu_csi2   = reinterpret_cast<cuphySimplexCwPrm_t*>(h2dBufCpuStartAddrs[PUSCH_SPX_CSI2_PRMS]);
    m_pRmCwPrmsCpu_csi2    = reinterpret_cast<cuphyRmCwPrm_t*>(h2dBufCpuStartAddrs[PUSCH_RM_CW_CSI2_PRMS]);
    m_pUciSegPrmsCpu_csi2  = reinterpret_cast<cuphyPolarUciSegPrm_t*>(h2dBufCpuStartAddrs[PUSCH_UCI_SEG_CSI2_PRMS]);
    m_pUciCwPrmsCpu_csi2   = reinterpret_cast<cuphyPolarCwPrm_t*>(h2dBufCpuStartAddrs[PUSCH_UCI_CW_CSI2_PRMS]);
    m_pSpxCwPrmsCpu_early  = reinterpret_cast<cuphySimplexCwPrm_t*>(h2dBufCpuStartAddrs[PUSCH_SPX_EARLY_PRMS]);
    m_pRmCwPrmsCpu_early   = reinterpret_cast<cuphyRmCwPrm_t*>(h2dBufCpuStartAddrs[PUSCH_RM_CW_EARLY_PRMS]);
    m_pUciSegPrmsCpu_early = reinterpret_cast<cuphyPolarUciSegPrm_t*>(h2dBufCpuStartAddrs[PUSCH_UCI_SEG_EARLY_PRMS]);
    m_pUciCwPrmsCpu_early  = reinterpret_cast<cuphyPolarCwPrm_t*>(h2dBufCpuStartAddrs[PUSCH_UCI_CW_EARLY_PRMS]);
}

void PuschRx::updateInputBuf()
{
    // before updating addresses, it is required to first update number of uci parameters
    expandUciParameters(true);

    size_t* pInputBufSizeBytes  = m_inputBufSizeBytes.data();

    pInputBufSizeBytes[PUSCH_TB_PRMS]             = sizeof(PerTbParams) * m_cuphyPuschCellGrpDynPrm.nUes;
    pInputBufSizeBytes[PUSCH_SPX_PRMS]            = sizeof(cuphySimplexCwPrm_t) * m_nSpxCws;
    pInputBufSizeBytes[PUSCH_RM_CW_PRMS]          = sizeof(cuphyRmCwPrm_t) * m_nRmCws;
    pInputBufSizeBytes[PUSCH_UCI_SEG_PRMS]        = sizeof(cuphyPolarUciSegPrm_t) * m_nPolUciSegs;
    pInputBufSizeBytes[PUSCH_UCI_CW_PRMS]         = sizeof(cuphyPolarCwPrm_t) * m_nPolCbs;
    pInputBufSizeBytes[PUSCH_SPX_EARLY_PRMS]      = sizeof(cuphySimplexCwPrm_t) * m_nSpxCws_early;
    pInputBufSizeBytes[PUSCH_RM_CW_EARLY_PRMS]    = sizeof(cuphyRmCwPrm_t) * m_nRmCws_early;
    pInputBufSizeBytes[PUSCH_UCI_SEG_EARLY_PRMS]  = sizeof(cuphyPolarUciSegPrm_t) * m_nPolUciSegs_early;
    pInputBufSizeBytes[PUSCH_UCI_CW_EARLY_PRMS]   = sizeof(cuphyPolarCwPrm_t) * m_nPolCbs_early;
    pInputBufSizeBytes[PUSCH_SPX_CSI2_PRMS]       = sizeof(cuphySimplexCwPrm_t) * m_nCsi2Ues;
    pInputBufSizeBytes[PUSCH_RM_CW_CSI2_PRMS]     = sizeof(cuphyRmCwPrm_t) * m_nCsi2Ues;

    if(m_cuphyPuschStatPrms.pDbg->forcedNumCsi2Bits > 0)
    {
        pInputBufSizeBytes[PUSCH_UCI_SEG_CSI2_PRMS] = sizeof(cuphyPolarUciSegPrm_t) * m_nCsi2Ues;
        pInputBufSizeBytes[PUSCH_UCI_CW_CSI2_PRMS]  = sizeof(cuphyPolarCwPrm_t) * m_nCsi2Ues * 2;
    }

    // Allocate input buffer
    m_h2dBuffer.update(m_inputBufSizeBytes, m_inputBufAlignBytes);

    // Update host pointers and shallow copy data if needed
    auto h2dBufCpuStartAddrs = m_h2dBuffer.getCpuStartAddrs();

    // since m_pTbPrmsCpu does not change as the first buffer in m_h2dBuffer, no need to be updated

    auto pSpxCwPrmsCpu     = reinterpret_cast<cuphySimplexCwPrm_t*>(h2dBufCpuStartAddrs[PUSCH_SPX_PRMS]);
    auto pSpxCwPrmsCpu_old = m_pSpxCwPrmsCpu;
    m_pSpxCwPrmsCpu        = pSpxCwPrmsCpu;
    if(pSpxCwPrmsCpu_old != m_pSpxCwPrmsCpu)
    {
        for(int i = 0; i < m_nSpxCws; i++)
        {
            m_pSpxCwPrmsCpu[i] = pSpxCwPrmsCpu_old[i];
        }
    }

    auto pSpxCwPrmsCpu_early     = reinterpret_cast<cuphySimplexCwPrm_t*>(h2dBufCpuStartAddrs[PUSCH_SPX_EARLY_PRMS]);
    auto pSpxCwPrmsCpu_old_early = m_pSpxCwPrmsCpu_early;
    m_pSpxCwPrmsCpu_early        = pSpxCwPrmsCpu_early;
    if(pSpxCwPrmsCpu_old_early != m_pSpxCwPrmsCpu_early)
    {
        for(int i = 0; i < m_nSpxCws_early; i++)
        {
            m_pSpxCwPrmsCpu_early[i] = pSpxCwPrmsCpu_old_early[i];
        }
    }

    auto pRmCwPrmsCpu     = reinterpret_cast<cuphyRmCwPrm_t*>(h2dBufCpuStartAddrs[PUSCH_RM_CW_PRMS]);
    auto pRmCwPrmsCpu_old = m_pRmCwPrmsCpu;
    m_pRmCwPrmsCpu        = pRmCwPrmsCpu;
    if(pRmCwPrmsCpu_old != m_pRmCwPrmsCpu)
    {
        for(int i = 0; i < m_nRmCws; i++)
        {
            m_pRmCwPrmsCpu[i] = pRmCwPrmsCpu_old[i];
        }
    }

    auto pRmCwPrmsCpu_early     = reinterpret_cast<cuphyRmCwPrm_t*>(h2dBufCpuStartAddrs[PUSCH_RM_CW_EARLY_PRMS]);
    auto pRmCwPrmsCpu_old_early = m_pRmCwPrmsCpu_early;
    m_pRmCwPrmsCpu_early        = pRmCwPrmsCpu_early;
    if(pRmCwPrmsCpu_old_early != m_pRmCwPrmsCpu_early)
    {
        for(int i = 0; i < m_nRmCws_early; i++)
        {
            m_pRmCwPrmsCpu_early[i] = pRmCwPrmsCpu_old_early[i];
        }
    }

    auto pUciSegPrmsCpu     = reinterpret_cast<cuphyPolarUciSegPrm_t*>(h2dBufCpuStartAddrs[PUSCH_UCI_SEG_PRMS]);
    auto pUciSegPrmsCpu_old = m_pUciSegPrmsCpu;
    m_pUciSegPrmsCpu        = pUciSegPrmsCpu;
    if(pUciSegPrmsCpu_old != m_pUciSegPrmsCpu)
    {
        for(int i = 0; i < m_nPolUciSegs; i++)
        {
            m_pUciSegPrmsCpu[i] = pUciSegPrmsCpu_old[i];
        }
    }

    auto pUciSegPrmsCpu_early     = reinterpret_cast<cuphyPolarUciSegPrm_t*>(h2dBufCpuStartAddrs[PUSCH_UCI_SEG_EARLY_PRMS]);
    auto pUciSegPrmsCpu_old_early = m_pUciSegPrmsCpu_early;
    m_pUciSegPrmsCpu_early        = pUciSegPrmsCpu_early;
    if(pUciSegPrmsCpu_old_early != m_pUciSegPrmsCpu_early)
    {
        for(int i = 0; i < m_nPolUciSegs_early; i++)
        {
            m_pUciSegPrmsCpu_early[i] = pUciSegPrmsCpu_old_early[i];
        }
    }

    auto pUciCwPrmsCpu     = reinterpret_cast<cuphyPolarCwPrm_t*>(h2dBufCpuStartAddrs[PUSCH_UCI_CW_PRMS]);
    auto pUciCwPrmsCpu_old = m_pUciCwPrmsCpu;
    m_pUciCwPrmsCpu        = pUciCwPrmsCpu;
    if(pUciCwPrmsCpu_old != m_pUciCwPrmsCpu)
    {
        for(int i = 0; i < m_nPolCbs; i++)
        {
            m_pUciCwPrmsCpu[i] = pUciCwPrmsCpu_old[i];
        }
    }

    auto pUciCwPrmsCpu_early     = reinterpret_cast<cuphyPolarCwPrm_t*>(h2dBufCpuStartAddrs[PUSCH_UCI_CW_EARLY_PRMS]);
    auto pUciCwPrmsCpu_old_early = m_pUciCwPrmsCpu_early;
    m_pUciCwPrmsCpu_early        = pUciCwPrmsCpu_early;
    if(pUciCwPrmsCpu_old_early != m_pUciCwPrmsCpu_early)
    {
        for(int i = 0; i < m_nPolCbs_early; i++)
        {
            m_pUciCwPrmsCpu_early[i] = pUciCwPrmsCpu_old_early[i];
        }
    }

    auto pSpxCwPrmsCpu_csi2     = reinterpret_cast<cuphySimplexCwPrm_t*>(h2dBufCpuStartAddrs[PUSCH_SPX_CSI2_PRMS]);
    auto pSpxCwPrmsCpu_csi2_old = m_pSpxCwPrmsCpu_csi2;
    m_pSpxCwPrmsCpu_csi2        = pSpxCwPrmsCpu_csi2;
    if(pSpxCwPrmsCpu_csi2_old != m_pSpxCwPrmsCpu_csi2)
    {
        for(int i = 0; i < m_nCsi2Ues; i++)
        {
            m_pSpxCwPrmsCpu_csi2[i] = pSpxCwPrmsCpu_csi2_old[i];
        }
    }

    auto pRmCwPrmsCpu_csi2     = reinterpret_cast<cuphyRmCwPrm_t*>(h2dBufCpuStartAddrs[PUSCH_RM_CW_CSI2_PRMS]);
    auto pRmCwPrmsCpu_csi2_old = m_pRmCwPrmsCpu_csi2;
    m_pRmCwPrmsCpu_csi2        = pRmCwPrmsCpu_csi2;
    if(pRmCwPrmsCpu_csi2_old != m_pRmCwPrmsCpu_csi2)
    {
        for(int i = 0; i < m_nCsi2Ues; i++)
        {
            m_pRmCwPrmsCpu_csi2[i] = pRmCwPrmsCpu_csi2_old[i];
        }
    }

    if(m_cuphyPuschStatPrms.pDbg->forcedNumCsi2Bits > 0)
    {
        auto pUciSegPrmsCpu_csi2     = reinterpret_cast<cuphyPolarUciSegPrm_t*>(h2dBufCpuStartAddrs[PUSCH_UCI_SEG_CSI2_PRMS]);
        auto pUciSegPrmsCpu_csi2_old = pUciSegPrmsCpu_csi2;
        m_pUciSegPrmsCpu_csi2        = pUciSegPrmsCpu_csi2;
        if(pUciSegPrmsCpu_csi2_old != m_pUciSegPrmsCpu_csi2)
        {
            for(int i = 0; i < m_nCsi2Ues; i++)
            {
                m_pUciSegPrmsCpu_csi2[i] = pUciSegPrmsCpu_csi2_old[i];
            }
        }

        auto pUciCwPrmsCpu_csi2     = reinterpret_cast<cuphyPolarCwPrm_t*>(h2dBufCpuStartAddrs[PUSCH_UCI_CW_CSI2_PRMS]);
        auto pUciCwPrmsCpu_csi2_old = m_pUciCwPrmsCpu_csi2;
        m_pUciCwPrmsCpu_csi2        = pUciCwPrmsCpu_csi2;
        if(pUciCwPrmsCpu_csi2_old != m_pUciCwPrmsCpu_csi2)
        {
            for(int i = 0; i < (2*m_nCsi2Ues); i++)
            {
                m_pUciCwPrmsCpu_csi2[i] = pUciCwPrmsCpu_csi2_old[i];
            }
        }
    }
}

void PuschRx::createComponents(cudaStream_t cuStrm,
                               int          rmFPconfig,
                               int          descramblingOn)
{
    // printf("Begin creating components\n");

    auto statCpuDescrStartAddrs      = m_kernelStatDescr.getCpuStartAddrs();
    auto statGpuDescrStartAddrs      = m_kernelStatDescr.getGpuStartAddrs();
    bool enableCpuToGpuDescrAsyncCpy = true;

    cuphyStatus_t status = cuphyCreatePuschRxChEst(&m_chEstHndl,
                                                   &m_chEstSettings.tPrmWFreq,
                                                   &m_chEstSettings.tPrmWFreq4,
                                                   &m_chEstSettings.tPrmWFreqSmall,
                                                   &m_chEstSettings.tPrmShiftSeq,
                                                   &m_chEstSettings.tPrmShiftSeq4,
                                                   &m_chEstSettings.tPrmUnShiftSeq,
                                                   &m_chEstSettings.tPrmUnShiftSeq4,
                                                   m_chEstSettings.pSymbolRxStatus,
                                                   enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                   reinterpret_cast<void**>(&statCpuDescrStartAddrs[PUSCH_CH_EST]),
                                                   reinterpret_cast<void**>(&statGpuDescrStartAddrs[PUSCH_CH_EST]),
                                                   cuStrm);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePuschRxChEst()");
    }

    status = cuphyCreatePuschRxNoiseIntfEst(&m_noiseIntfEstHndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePuschRxNoiseIntfEst()");
    }

    status = cuphyCreatePuschRxCfoTaEst(&m_cfoTaEstHndl,
                                        enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                        static_cast<void*>(statCpuDescrStartAddrs[PUSCH_CFO_TA_EST]),
                                        static_cast<void*>(statGpuDescrStartAddrs[PUSCH_CFO_TA_EST]),
                                        cuStrm);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePuschRxCfoTaEst()");
    }

    status = cuphyCreatePuschRxChEq(m_ctx.handle(),
                                    &m_chEqHndl,
                                    enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                    reinterpret_cast<void**>(&statCpuDescrStartAddrs[PUSCH_CH_EQ_COEF]),
                                    reinterpret_cast<void**>(&statGpuDescrStartAddrs[PUSCH_CH_EQ_COEF]),
                                    cuStrm);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePuschRxChEq()");
    }

    status = cuphyCreatePuschRxRateMatch(&m_rateMatchHndl, rmFPconfig, descramblingOn);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePuschRxRateMatch()");
    }

    status = cuphyCreatePuschRxCrcDecode(&m_crcDecodeHndl, 1);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePuschRxCrcDecode()");
    }

    status = cuphyCreatePuschRxRssi(&m_rssiHndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePuschRxRssi()");
    }

    status = cuphyCreateUciOnPuschSegLLRs0(&m_uciOnPuschSegLLRs0Hndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateUciOnPuschSegLLRs0()");
    }

    status = cuphyCreateUciOnPuschSegLLRs0(&m_uciOnPuschEarlySegLLRs0Hndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateUciOnPuschEarlySegLLRs0()");
    }

    status = cuphyCreateUciOnPuschSegLLRs2(&m_uciOnPuschSegLLRs2Hndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateUciOnPuschSegLLRs2()");
    }

    status = cuphyCreateUciOnPuschCsi2Ctrl(&m_uciOnPuschCsi2CtrlHndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateUciOnPuschCsi2Ctrl()");
    }

    status = cuphyCreateSimplexDecoder(&m_spxDecoderHndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateSimplexDecoder()");
    }

    unsigned int rmFlags = 0;
    status               = cuphyCreateRmDecoder(m_ctx.handle(), &m_rmDecodeHndl, rmFlags, &m_memoryFootprint);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateRmDecoder()");
    }

    status = cuphyCreateSimplexDecoder(&m_spxDecoderHndl_csi2);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateSimplexDecoder_csi2()");
    }

    status = cuphyCreateRmDecoder(m_ctx.handle(), &m_rmDecodeHndl_csi2, rmFlags, &m_memoryFootprint);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateRmDecoder_csi2()");
    }

    status = cuphyCreateSimplexDecoder(&m_spxDecoderHndl_early);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateSimplexDecoder_early()");
    }

    status = cuphyCreateRmDecoder(m_ctx.handle(), &m_rmDecodeHndl_early, rmFlags, &m_memoryFootprint);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateRmDecoder_early()");
    }

    status = cuphyCreateCompCwTreeTypes(&m_compCwTreeTypesHndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateCompCwTreeTypes()");
    }

    status = cuphyCreatePolSegDeRmDeItl(&m_polSegDeRmDeItlHndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePolSegDeRmDeItl()");
    }

    status = cuphyCreatePolarDecoder(&m_polarDecoderHndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePolarDecoder()");
    }
    
    status = cuphyCreateCompCwTreeTypes(&m_compCwTreeTypesHndl_csi2);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateCompCwTreeTypes(CSI 2)");
    }

    status = cuphyCreatePolSegDeRmDeItl(&m_polSegDeRmDeItlHndl_csi2);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePolSegDeRmDeItl(CSI 2)");
    }

    status = cuphyCreatePolarDecoder(&m_polarDecoderHndl_csi2);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePolarDecoder(CSI 2)");
    }

    status = cuphyCreateCompCwTreeTypes(&m_compCwTreeTypesHndl_early);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateCompCwTreeTypes(early)");
    }

    status = cuphyCreatePolSegDeRmDeItl(&m_polSegDeRmDeItlHndl_early);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePolSegDeRmDeItl(early)");
    }

    status = cuphyCreatePolarDecoder(&m_polarDecoderHndl_early);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreatePolarDecoder(early)");
    }

    // printf("Done creating components\n");
}

void PuschRx::destroyComponents()
{
    // printf("Begin destroying components\n");

    cuphyStatus_t status = cuphyDestroyPuschRxChEst(m_chEstHndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyPuschRxChEst() error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyPuschRxChEst()");
    }

    status = cuphyDestroyPuschRxNoiseIntfEst(m_noiseIntfEstHndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyPuschRxNoiseIntfEst() error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyPuschRxNoiseIntfEst()");
    }

    status = cuphyDestroyPuschRxCfoTaEst(m_cfoTaEstHndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyPuschRxCfoTaEst() error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyPuschRxCfoTaEst()");
    }

    status = cuphyDestroyPuschRxChEq(m_chEqHndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyPuschRxChEq() error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyPuschRxChEq()");
    }

    status = cuphyDestroyPuschRxRateMatch(m_rateMatchHndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyPuschRxRateMatch() error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyPuschRxRateMatch()");
    }

    status = cuphyDestroyPuschRxCrcDecode(m_crcDecodeHndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyPuschRxCrcDecode() error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyPuschRxCrcDecode()");
    }

    status = cuphyDestroyPuschRxRssi(m_rssiHndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyPuschRxRssi() error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyPuschRxRssi()");
    }

    status = cuphyDestroyUciOnPuschSegLLRs0(m_uciOnPuschSegLLRs0Hndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyUciOnPuschSegLLRs0() error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyUciOnPuschSegLLRs0()");
    }

    status = cuphyDestroyUciOnPuschSegLLRs0(m_uciOnPuschEarlySegLLRs0Hndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyUciOnPuschEarlySegLLRs0() error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyUciOnPuschEarlySegLLRs0()");
    }

    status = cuphyDestroyUciOnPuschSegLLRs2(m_uciOnPuschSegLLRs2Hndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyUciOnPuschSegLLRs2() error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyUciOnPuschSegLLRs2()");
    }

    status = cuphyDestroyUciOnPuschCsi2Ctrl(m_uciOnPuschCsi2CtrlHndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyUciOnPuschCsi2Ctrl() error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyUciOnPuschCsi2Ctrl()");
    }

    status = cuphyDestroySimplexDecoder(m_spxDecoderHndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroySimplexDecoder() error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroySimplexDecoder()");
    }

    status = cuphyDestroyRmDecoder(m_rmDecodeHndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyRmDecoder() error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyRmDecoder()");
    }

    status = cuphyDestroySimplexDecoder(m_spxDecoderHndl_csi2);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroySimplexDecoder_csi2() error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroySimplexDecoder_csi2()");
    }

    status = cuphyDestroyRmDecoder(m_rmDecodeHndl_csi2);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyRmDecoder_csi2() error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyRmDecoder_csi2()");
    }

    status = cuphyDestroyCompCwTreeTypes(m_compCwTreeTypesHndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyCompCwTreeTypes() error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyCompCwTreeTypes()");
    }

    status = cuphyDestroyPolSegDeRmDeItl(m_polSegDeRmDeItlHndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyPolSegDeRmDeItl() error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyPolSegDeRmDeItl()");
    }

    status = cuphyDestroyPolarDecoder(m_polarDecoderHndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyPolarDecoder() error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyPolarDecoder()");
    }

    status = cuphyDestroyCompCwTreeTypes(m_compCwTreeTypesHndl_csi2);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyCompCwTreeTypes(CSI 2) error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyCompCwTreeTypes(CSI 2)");
    }

    status = cuphyDestroyPolSegDeRmDeItl(m_polSegDeRmDeItlHndl_csi2);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyPolSegDeRmDeItl(CSI 2) error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyPolSegDeRmDeItl(CSI 2)");
    }

    status = cuphyDestroyPolarDecoder(m_polarDecoderHndl_csi2);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyPolarDecoder(CSI 2) error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyPolarDecoder(CSI 2)");
    }

    status = cuphyDestroySimplexDecoder(m_spxDecoderHndl_early);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroySimplexDecoder_early() error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroySimplexDecoder_early()");
    }

    status = cuphyDestroyRmDecoder(m_rmDecodeHndl_early);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyRmDecoder_early() error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyRmDecoder_early()");
    }
    status = cuphyDestroyCompCwTreeTypes(m_compCwTreeTypesHndl_early);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyCompCwTreeTypes(early) error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyCompCwTreeTypes(early)");
    }

    status = cuphyDestroyPolSegDeRmDeItl(m_polSegDeRmDeItlHndl_early);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyPolSegDeRmDeItl(early) error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyPolSegDeRmDeItl(early)");
    }

    status = cuphyDestroyPolarDecoder(m_polarDecoderHndl_early);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphyDestroyPolarDecoder(early) error {}", cuphyGetErrorString(status));
        throw cuphy::cuphy_fn_exception(status, "cuphyDestroyPolarDecoder(early)");
    }
}

PuschRx::PuschRx(cuphyPuschStatPrms_t const* pStatPrms, cudaStream_t cuStream) :
    m_puschCellStatPrmBufGpu(pStatPrms->nMaxCells, &m_memoryFootprint),
    m_puschCellStatPrmCpu(pStatPrms->nMaxCells),
    m_cuphyCellStatPrmVecCpu(pStatPrms->pCellStatPrms, pStatPrms->pCellStatPrms + pStatPrms->nMaxCells),
    m_h2dBuffer("PuschInputPrms"),
    m_ldpcWorkspaceSize(0),
    m_LDPCdecoder(m_ctx),
    m_LDPCkernelLaunchMode(pStatPrms->ldpcKernelLaunch),
    m_polDcdrListSz(pStatPrms->polarDcdrListSz),
    m_LinearAlloc(getBufferSize(pStatPrms), &m_memoryFootprint),
    m_kernelStatDescr("PuschStatDescr"),
    m_kernelDynDescr("PuschDynDescr"),
    m_cuphyPuschStatPrms(*(pStatPrms)),
    m_cudaGraphModeEnabled(false),
    m_ldpcPrms(pStatPrms),
    m_chEstSettings(pStatPrms, cuStream, &m_memoryFootprint),
    m_maxNTbs(pStatPrms->nMaxTbs == 0 ? (pStatPrms->nMaxCellsPerSlot) > 1 ? MAX_N_TBS_PER_CELL_GROUP_SUPPORTED : MAX_N_TBS_SUPPORTED : pStatPrms->nMaxTbs),
    m_maxNCbs(pStatPrms->nMaxTotCbs == 0 ? (pStatPrms->nMaxTbs == 0 ? (pStatPrms->nMaxCellsPerSlot) > 1 ? MAX_N_TBS_PER_CELL_GROUP_SUPPORTED : MAX_N_TBS_SUPPORTED : pStatPrms->nMaxTbs) * (pStatPrms->nMaxCbsPerTb == 0 ? (pStatPrms->nMaxCellsPerSlot) > 1 ? MAX_N_CBS_PER_TB_PER_CELL_GROUP_SUPPORTED : MAX_N_CBS_PER_TB_SUPPORTED : pStatPrms->nMaxCbsPerTb) : pStatPrms->nMaxTotCbs),
    m_maxNCbsPerTb(pStatPrms->nMaxCbsPerTb == 0 ? (pStatPrms->nMaxCellsPerSlot) > 1 ? MAX_N_CBS_PER_TB_PER_CELL_GROUP_SUPPORTED : MAX_N_CBS_PER_TB_SUPPORTED : pStatPrms->nMaxCbsPerTb),
    m_maxNPrbAlloc(pStatPrms->nMaxPrb == 0 ? MAX_N_PRBS_SUPPORTED : pStatPrms->nMaxPrb),
    m_maxNRx(pStatPrms->nMaxRx == 0 ? MAX_N_ANTENNAS_SUPPORTED : pStatPrms->nMaxRx),
    m_ldpcStreamPool(0),
    m_G1streamPool(0),
    m_G2streamPool(0),
    m_uciOnPuschSegLLRs0Event{cudaEventDisableTiming},
    m_compCwTreeTypesEvent{cudaEventDisableTiming},
    m_rateMatchEvent(cudaEventDisableTiming)
{
    pStatPrms->pOutInfo->pMemoryFootprint = &m_memoryFootprint; // update  static parameter field that points to the cuphyMemoryFootprintTracker object for this channel

    //m_cuphyPuschStatPrms.pDbg->forcedNumCsi2Bits = 0; //TODO: remove this after CP team populates this API paramater.
    uint32_t MAX_N_TBs_SUPPORTED        = (pStatPrms->nMaxCellsPerSlot) > 1 ? MAX_N_TBS_PER_CELL_GROUP_SUPPORTED : MAX_N_TBS_SUPPORTED;
    uint32_t MAX_N_CBs_PER_TB_SUPPORTED = (pStatPrms->nMaxCellsPerSlot) > 1 ? MAX_N_CBS_PER_TB_PER_CELL_GROUP_SUPPORTED : MAX_N_CBS_PER_TB_SUPPORTED;

    // Allocate buffer for input parameters
    allocateInputBuf(pStatPrms->nMaxTbs, MAX_N_TBs_SUPPORTED);

    if(m_maxNTbs > MAX_N_TBs_SUPPORTED)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "nMaxTbs provided {} is larger than supported max {}", m_maxNTbs, MAX_N_TBs_SUPPORTED);
        std::string err("PUSCH: nMaxTbs provided (" + std::to_string(m_maxNTbs) + ") is larger than supported max (" + std::to_string(MAX_N_TBs_SUPPORTED) + ")");
        throw std::out_of_range(err);
    }

    if(m_maxNCbsPerTb > MAX_N_CBs_PER_TB_SUPPORTED)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "nMaxCbsPerTb provided {} is larger than supported max {}", m_maxNCbsPerTb, MAX_N_CBs_PER_TB_SUPPORTED);
        std::string err("PUSCH: nMaxCbsPerTb provided (" + std::to_string(m_maxNCbsPerTb) + ") is larger than supported max (" + std::to_string(MAX_N_CBs_PER_TB_SUPPORTED) + ")");
        throw std::out_of_range(err);
    }

    if(m_maxNCbs > m_maxNTbs * m_maxNCbsPerTb)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "nMaxTotCbs provided {} is larger than provided nMaxTbs * nMaxCbsPerTb {}", m_maxNCbs, m_maxNTbs * m_maxNCbsPerTb);
        std::string err("PUSCH: nMaxTotCbs provided (" + std::to_string(m_maxNCbs) + ") is larger than provided nMaxTbs * nMaxCbsPerTb (" + std::to_string(m_maxNTbs * m_maxNCbsPerTb) + ")");
        throw std::out_of_range(err);
    }

    if(m_maxNPrbAlloc > MAX_N_PRBS_SUPPORTED)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "nMaxPrb provided {} is larger than supported max {}", m_maxNPrbAlloc, MAX_N_PRBS_SUPPORTED);
        std::string err("PUSCH: nMaxPrb provided (" + std::to_string(m_maxNPrbAlloc) + ") is larger than supported max (" + std::to_string(MAX_N_PRBS_SUPPORTED) + ")");
        throw std::out_of_range(err);
    }

    if(m_maxNRx > MAX_N_ANTENNAS_SUPPORTED)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "nMaxRx provided {} is larger than supported max {}", m_maxNRx, MAX_N_ANTENNAS_SUPPORTED);
        std::string err("PUSCH: nMaxRx provided (" + std::to_string(m_maxNRx) + ") is larger than supported max (" + std::to_string(MAX_N_ANTENNAS_SUPPORTED) + ")");
        throw std::out_of_range(err);
    }

    if ((m_LDPCkernelLaunchMode & PUSCH_RX_LDPC_STREAM_POOL) || (m_LDPCkernelLaunchMode & PUSCH_RX_ENABLE_LDPC_DEC_SINGLE_STREAM_OPT))
    {
        // stream priority of the stream pool needs to be consistent with priority of cuStream
        int tmp_priority = 0;
        CUDA_CHECK(cudaStreamGetPriority(cuStream, &tmp_priority));
        m_ldpcStreamPool.resize(N_MAX_LDPC_HET_CFGS, tmp_priority);
    }

#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
    // create stream pool for to launch 1-rssiMeasKernel, 2-rsrpMeasKernel_v1, 3-uciOnPuschSeqLLRs0 and 4-compCwTreeTypes,
    // also to launch 1-simplex_decoder 2-rm_decoder and 3-polSegDeRmDeIt in CSI-P1,
    // and to launch 1-rm_decoder 2-simplex_decoder, 3-de_rate_matching_global2, and 4 polar backend in CSI-P2
    int priority = 0;
    CUDA_CHECK(cudaStreamGetPriority(cuStream, &priority));
    size_t max_num_streams = 2 + CUPHY_PUSCH_RX_RSSI_N_MAX_HET_CFGS + CUPHY_PUSCH_RX_RSRP_N_MAX_HET_CFGS;
    m_G1streamPool.resize(max_num_streams , priority);
    m_G2streamPool.resize(4 , priority);
#endif

    // Resize (not reserve) front end tensors to avoid constructor/destructor calls later. Unused
    // tensors are OK.
    m_tRefDataRx.resize(pStatPrms->nMaxCellsPerSlot);

    m_tRefLLRVec.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tRefLLRCdm1Vec.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tRefHEstVec.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tRefPerPrbNoiseVarVec.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tRefLwInvVec.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tRefChEstDbgVec.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tRefCfoEstVec.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tRefReeDiagInvVec.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tRefDataEqVec.resize(MAX_N_USER_GROUPS_SUPPORTED);
    if(m_chEstSettings.enableDftSOfdm==1)
    {
        m_tRefDataEqDftVec.resize(MAX_N_USER_GROUPS_SUPPORTED);
        m_tRefDataEqDftIntermediateVec.resize(MAX_N_USER_GROUPS_SUPPORTED);
        m_tRefDftBluesteinWorkspaceTimeVec.resize(MAX_N_USER_GROUPS_SUPPORTED);
        m_tRefDftBluesteinWorkspaceFreqVec.resize(MAX_N_USER_GROUPS_SUPPORTED);
    }
    m_tRefCoefVec.resize(MAX_N_USER_GROUPS_SUPPORTED);
    m_tRefEqDbgVec.resize(MAX_N_USER_GROUPS_SUPPORTED);

    // Allocate front-end tensor parameters.
    m_tPrmLLRVec.reserve(MAX_N_USER_GROUPS_SUPPORTED);
    m_tPrmLLRCdm1Vec.reserve(MAX_N_USER_GROUPS_SUPPORTED);
    m_nPrbsVec.reserve(MAX_N_USER_GROUPS_SUPPORTED);

    // back end parameters
    m_schUserIdxsVec.resize(MAX_N_TBS_PER_CELL_GROUP_SUPPORTED);
    m_uciUserIdxsVec.resize(CUPHY_MAX_N_POL_UCI_SEGS);
    m_csi2UeIdxsVec.resize(CUPHY_MAX_N_POL_UCI_SEGS);
    m_cwTreeTypesAddrVec.resize(CUPHY_MAX_N_POL_UCI_SEGS);
    m_uciSegLLRsAddrVec.resize(CUPHY_MAX_N_POL_UCI_SEGS);
    m_cwLLRsAddrVec.resize(CUPHY_MAX_N_POL_CWS);
    m_cbEstAddrVec.resize(CUPHY_MAX_N_POL_CWS);
    m_cwTreeTypesAddrVec_csi2.resize(CUPHY_MAX_N_POL_CWS_CSI2);
    m_uciSegLLRsAddrVec_csi2.resize(CUPHY_MAX_N_POL_UCI_SEGS_CSI2);
    m_cwLLRsAddrVec_csi2.resize(CUPHY_MAX_N_POL_CWS_CSI2);
    m_cbEstAddrVec_csi2.resize(CUPHY_MAX_N_POL_CWS_CSI2);
    m_cwTreeTypesAddrVec_early.resize(CUPHY_MAX_N_POL_CWS);
    m_uciSegLLRsAddrVec_early.resize(CUPHY_MAX_N_POL_UCI_SEGS);
    m_cwLLRsAddrVec_early.resize(CUPHY_MAX_N_POL_CWS);
    m_cbEstAddrVec_early.resize(CUPHY_MAX_N_POL_CWS);
    if (m_polDcdrListSz > 1) {
        m_listPolScratchAddrVec.resize(CUPHY_MAX_N_POL_CWS);
        m_listPolScratchAddrVec_csi2.resize(CUPHY_MAX_N_POL_CWS_CSI2);
        m_listPolScratchAddrVec_early.resize(CUPHY_MAX_N_POL_CWS);
    }
    m_cwTreeLLRsAddrVec.resize(CUPHY_MAX_N_POL_CWS);
    m_cwTreeLLRsAddrVec_csi2.resize(CUPHY_MAX_N_POL_CWS_CSI2);
    m_cwTreeLLRsAddrVec_early.resize(CUPHY_MAX_N_POL_CWS);
    m_pUciSegEst.resize(CUPHY_MAX_N_POL_UCI_SEGS);
    m_pUciSegEst_csi2.resize(CUPHY_MAX_N_POL_UCI_SEGS_CSI2);
    m_pUciSegEst_early.resize(CUPHY_MAX_N_POL_UCI_SEGS);

    // store number of launch config state; to be used in updateGraph() to reduce CUDA api calls
    m_chEstNodesEnabled.resize(CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST, std::vector<uint8_t>(CUPHY_PUSCH_RX_CH_EST_N_MAX_HET_CFGS, std::numeric_limits<uint8_t>::max()));
    m_noiseIntfEstNodesEnabled.resize(CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS, std::numeric_limits<uint8_t>::max());
    m_cfoTaEstNodesEnabled.resize(CUPHY_PUSCH_RX_CFO_EST_N_MAX_HET_CFGS, std::numeric_limits<uint8_t>::max());
    m_chEqCoefCompNodesEnabled.resize(CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST, std::vector<uint8_t>(CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS, std::numeric_limits<uint8_t>::max()));
    m_chEqSoftDemapNodesEnabled.resize(CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS, std::numeric_limits<uint8_t>::max());
    if(m_chEstSettings.enableDftSOfdm==1)
    {
        m_chEqSoftDemapBluesteinWorkspaceNodesEnabled.resize(CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS, std::numeric_limits<uint8_t>::max());
        m_chEqSoftDemapIdftNodesEnabled.resize(CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS, std::numeric_limits<uint8_t>::max());
        m_chEqSoftDemapAfterDftNodesEnabled.resize(CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS, std::numeric_limits<uint8_t>::max());
    }
    m_ldpcDecoderNodesEnabled.resize(N_MAX_LDPC_HET_CFGS, std::numeric_limits<uint8_t>::max());
    m_rssiNodesEnabled.resize(CUPHY_PUSCH_RX_RSSI_N_MAX_HET_CFGS, std::numeric_limits<uint8_t>::max());
    m_rsrpNodesEnabled.resize(CUPHY_PUSCH_RX_RSRP_N_MAX_HET_CFGS, std::numeric_limits<uint8_t>::max());
    m_rateMatchNodeEnabled          = std::numeric_limits<uint8_t>::max();
    m_crcNodesEnabled               = std::numeric_limits<uint8_t>::max();
    m_uciSegLLRs0NodeEnabled        = std::numeric_limits<uint8_t>::max();
    m_simplexDecoderNodeEnabled     = std::numeric_limits<uint8_t>::max();
    m_rmDecoderNodeEnabled          = std::numeric_limits<uint8_t>::max();
    m_polarNodeEnabled              = std::numeric_limits<uint8_t>::max();
    m_csi2NodeEnabled               = std::numeric_limits<uint8_t>::max();
    // enable state flags for early HARQ path
    m_ehqAllNodesDisabled   = 0;
    m_ehqPreWaitNodeEnabled = std::numeric_limits<uint8_t>::max();
    m_ehqChEstNodesEnabled.resize(CUPHY_PUSCH_RX_CH_EST_N_MAX_HET_CFGS, std::numeric_limits<uint8_t>::max());
    m_ehqNoiseIntfEstNodesEnabled.resize(CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS, std::numeric_limits<uint8_t>::max());
    m_ehqChEqCoefCompNodesEnabled.resize(CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS, std::numeric_limits<uint8_t>::max());
    m_ehqChEqSoftDemapNodesEnabled.resize(CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS, std::numeric_limits<uint8_t>::max());
    if(m_chEstSettings.enableDftSOfdm==1)
    {
        m_ehqChEqSoftDemapBluesteinWorkspaceNodesEnabled.resize(CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS, std::numeric_limits<uint8_t>::max());
        m_ehqChEqSoftDemapIdftNodesEnabled.resize(CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS, std::numeric_limits<uint8_t>::max());
        m_ehqChEqSoftDemapAfterDftNodesEnabled.resize(CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS, std::numeric_limits<uint8_t>::max());
    }
    m_ehqUciSegLLRs0NodeEnabled         = std::numeric_limits<uint8_t>::max();
    m_ehqSimplexDecoderNodeEnabled      = std::numeric_limits<uint8_t>::max();
    m_ehqRmDecoderNodeEnabled           = std::numeric_limits<uint8_t>::max();
    m_ehqPolarNodeEnabled               = std::numeric_limits<uint8_t>::max();
    m_ehqMemcpyUciPayloadNodeEnabled    = std::numeric_limits<uint8_t>::max();
    m_ehqMemcpyUciCrcNodeEnabled        = std::numeric_limits<uint8_t>::max();
    m_ehqMemcpyDetectionStatNodeEnabled = std::numeric_limits<uint8_t>::max();
    m_ehqEventReadyNodeEnabled          = std::numeric_limits<uint8_t>::max();
    m_ehqPostWaitNodeEnabled            = std::numeric_limits<uint8_t>::max();

    // Allocate descriptors for pipeline usage
    allocateDescr();

    // Initalize componets
    int rmFPconfig;
#ifdef LLR_FP16
    rmFPconfig = pStatPrms->ldpcUseHalf * 2 + 1;
#else
    rmFPconfig = pStatPrms->ldpcUseHalf * 2 + 0;
#endif

    // Keep descrambling on by default. Unless disabled explictly via debug
    int descrmOn                 = true;
    m_outputPrms.debugOutputFlag = false;
    if(nullptr != pStatPrms->pDbg)
    {
        descrmOn = static_cast<int>(pStatPrms->pDbg->descrmOn);

        // set debug prms
        if(pStatPrms->pDbg->pOutFileName != nullptr)
        {
            m_outputPrms.debugOutputFlag = true;
            m_outputPrms.outHdf5File     = hdf5hpp::hdf5_file::open(pStatPrms->pDbg->pOutFileName);
        }
    }

    // copy static cell parameters to GPU memory:
    uint16_t nMaxCells = pStatPrms->nMaxCells;
    for(int cellIdx = 0; cellIdx < nMaxCells; ++cellIdx)
    {
        m_puschCellStatPrmCpu[cellIdx] = *(pStatPrms->pCellStatPrms[cellIdx].pPuschCellStatPrms);
    }

    CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(static_cast<void*>(m_puschCellStatPrmBufGpu.addr()), static_cast<void*>(m_puschCellStatPrmCpu.addr()), nMaxCells * sizeof(cuphyPuschCellStatPrm_t), cudaMemcpyHostToDevice, cuStream));


    createComponents(cuStream, rmFPconfig, descrmOn);

    createGraph();
    CUDA_CHECK_EXCEPTION(cudaGraphInstantiate(&m_graphExec, m_graph, NULL, NULL, 0));
    disableAllEarlyHarqNodes();

    if(PRINT_GPU_MEMORY_CUPHY_CHANNEL == 1)
    {
        m_memoryFootprint.printMemoryFootprint(this, "PUSCH");
    }
}

cuphyStatus_t PuschRx::setupCmnPhase1(cuphyPuschDynPrms_t* pDynPrm)
{
    m_cuphyPuschCellGrpDynPrm = *(pDynPrm->pCellGrpDynPrm);

    if(m_cuphyPuschCellGrpDynPrm.nUes > m_maxNTbs)
    {
        pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_NTBS_PERCELLGROUP_OUT_OF_RANGE;
        pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
        pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Number of TBs in cell group is {} but current max. specified for PUSCH is {}", __FUNCTION__, m_cuphyPuschCellGrpDynPrm.nUes, m_maxNTbs);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    if(m_cuphyPuschCellGrpDynPrm.nUeGrps > MAX_N_USER_GROUPS_SUPPORTED)
    {
        pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_NUEGRPS_PERCELLGROUP_OUT_OF_RANGE;
        pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
        pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Number of UE groups in cell group is {} but current max. specified for PUSCH is {}", __FUNCTION__, m_cuphyPuschCellGrpDynPrm.nUeGrps, MAX_N_USER_GROUPS_SUPPORTED);
        return CUPHY_STATUS_VALUE_OUT_OF_RANGE;  
    }
    // reset linear allocation
    m_LinearAlloc.reset();

    // pusch parameters
    m_harqBufferSizeInBytes    = pDynPrm->pDataOut->h_harqBufferSizeInBytes;
    m_pHarqBuffers             = reinterpret_cast<void**>(pDynPrm->pDataInOut->pHarqBuffersInOut);
    auto dynCpuDescrStartAddrs = m_kernelDynDescr.getCpuStartAddrs();
    auto dynGpuDescrStartAddrs = m_kernelDynDescr.getGpuStartAddrs();
    m_drvdUeGrpPrmsCpu         = (cuphyPuschRxUeGrpPrms_t*)(dynCpuDescrStartAddrs[PUSCH_FRONT_END_PARAMS]);
    m_drvdUeGrpPrmsGpu         = (cuphyPuschRxUeGrpPrms_t*)(dynGpuDescrStartAddrs[PUSCH_FRONT_END_PARAMS]);
    m_nMaxPrb                  = expandFrontEndParameters(pDynPrm, m_drvdUeGrpPrmsCpu, m_cuphyPuschStatPrms.enableRssiMeasurement);
    if((pDynPrm->pStatusOut->status == cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_NPRBS_OUT_OF_RANGE) && (m_nMaxPrb == 0xFFFFFFFF))
    {
        return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
    }
    //expandParameters(m_tbPrmsArrayCpu.addr(), &m_cuphyPuschStatPrms, pDynPrm, m_cuphyCellStatPrm, m_cmnPrms, m_ueGrpPrmsPrime, m_ldpcPrms);
    cuphyStatus_t status = expandBackEndParameters(pDynPrm, m_drvdUeGrpPrmsCpu, m_pTbPrmsCpu, m_ldpcPrms);
    if(status != CUPHY_STATUS_SUCCESS) return status;


    PerTbParams* pPerTbPrms = m_pTbPrmsCpu;
    for(uint32_t iterator = 0; iterator < m_cuphyPuschCellGrpDynPrm.nUeGrps; iterator++)
    {
        cuphyPuschUeGrpPrm_t ueGrpPrms = m_cuphyPuschCellGrpDynPrm.pUeGrpPrms[iterator];
        m_nPrbsVec[iterator]           = ueGrpPrms.nPrb;
        for(int i = 0; i < ueGrpPrms.nUes; i++)
        {
            uint16_t ueIdx = ueGrpPrms.pUePrmIdxs[i];
            uint8_t  Qm    = static_cast<uint8_t>(pPerTbPrms[ueIdx].Qm);
            for(int j = 0; j < pPerTbPrms[ueIdx].Nl; ++j)
            {
                m_drvdUeGrpPrmsCpu[iterator].qam[pPerTbPrms[ueIdx].layer_map_array[j]] = Qm;
            }
        }
    }

    // output parameters
    m_outputPrms.cpuCopyOn                  = pDynPrm->cpuCopyOn;
    m_outputPrms.pUciOnPuschOutOffsets      = pDynPrm->pDataOut->pUciOnPuschOutOffsets;

    cuphyPuschUePrm_t* pUePrms = m_cuphyPuschCellGrpDynPrm.pUePrms;

    m_outputPrms.totNumTbs          = 0;
    m_outputPrms.totNumCbs          = 0;
    m_outputPrms.totNumPayloadBytes = 0;
    m_nSchUes                       = 0;
    m_nEarlyHarqUes                 = 0;

    bool earlyHarqMode = (pDynPrm->procModeBmsk & PUSCH_PROC_MODE_SUB_SLOT_EARLY_HARQ)? true : false;

    for(int ueIdx = 0; ueIdx < m_cuphyPuschCellGrpDynPrm.nUes; ++ueIdx)
    {
        if(pUePrms[ueIdx].pduBitmap & 2)
        {
            m_outputPrms.pUciOnPuschOutOffsets[ueIdx].HarqDetectionStatusOffset  = ueIdx;
            m_outputPrms.pUciOnPuschOutOffsets[ueIdx].CsiP1DetectionStatusOffset = ueIdx;
            m_outputPrms.pUciOnPuschOutOffsets[ueIdx].CsiP2DetectionStatusOffset = ueIdx;
            m_outputPrms.pUciOnPuschOutOffsets[ueIdx].isEarlyHarq = 0;
            m_pTbPrmsCpu[ueIdx].isEarlyHarq = 0;
            if(pUePrms[ueIdx].pUciPrms->nBitsHarq>0)
            {
                //****************** determine isEarlyHarq for each UE ***************************//
                //****************** HARQ bits in symbol 0 ~ symbol 3  ***************************//
                uint32_t G_harq = pPerTbPrms[ueIdx].G_harq;
                uint16_t ueGrpIdx = pUePrms[ueIdx].ueGrpIdx;
                cuphyPuschRxUeGrpPrms_t* drvdUeGrpPrms = &m_drvdUeGrpPrmsCpu[ueGrpIdx];
                uint8_t nEarlyHarqSymbols = 0;
                uint8_t firstHarqSymbol = drvdUeGrpPrms->dmrsSymLoc[0] + drvdUeGrpPrms->dmrsMaxLen;
                if(firstHarqSymbol <= CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_UPPER_BOUND)
                {
                    nEarlyHarqSymbols = CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_UPPER_BOUND + 1 - firstHarqSymbol;
                }
                uint32_t nAvailableEaryHarqRmBits = static_cast<uint32_t>(nEarlyHarqSymbols)*(static_cast<uint32_t>(pUePrms[ueIdx].nUeLayers))*(static_cast<uint32_t>(pUePrms[ueIdx].pUeGrpPrm->nPrb))*CUPHY_N_TONES_PER_PRB*(static_cast<uint32_t>(pUePrms[ueIdx].qamModOrder));
                if(G_harq<=nAvailableEaryHarqRmBits)
                {
                    m_outputPrms.pUciOnPuschOutOffsets[ueIdx].isEarlyHarq = 1;
                    m_nEarlyHarqUes += 1;
                    m_pTbPrmsCpu[ueIdx].isEarlyHarq = static_cast<uint8_t>(earlyHarqMode);
                }
                //printf("ueIdx[%d]G_harq[%d]nAvailableEaryHarqRmBits[%d]firstHarqSymbol[%d]nEarlyHarqSymbols[%d]\n", ueIdx, G_harq, nAvailableEaryHarqRmBits, firstHarqSymbol, nEarlyHarqSymbols);
                //********************************************************************************//
            }
        }

        if(pUePrms[ueIdx].pduBitmap & 1)
        {
            //While creating the output buffer pTbPayloads, cuPhy copies the TB first, followed by the CRC (3 bytes)
            //Then, before copying the next TB into pTbPayloads, it aligns the start of next TB to 4 byte address
            //space. Hence there is some additional bytes between 2 TBs. The totNumPayloadBytes need to take into
            //account these additional bytes.
            pDynPrm->pDataOut->pStartOffsetsTbPayload[ueIdx] = m_outputPrms.totNumPayloadBytes;
            pDynPrm->pDataOut->pStartOffsetsTbCrc[ueIdx]     = m_outputPrms.totNumTbs;
            pDynPrm->pDataOut->pStartOffsetsCbCrc[ueIdx]     = m_outputPrms.totNumCbs;

            m_outputPrms.totNumTbs      += 1;
            m_schUserIdxsVec[m_nSchUes] =  ueIdx;
            m_nSchUes                   += 1;

            m_outputPrms.totNumCbs += m_pTbPrmsCpu[ueIdx].num_CBs;

            uint8_t  crcSizeBytes = m_pTbPrmsCpu[ueIdx].tbSize > 3824 ? 3 : 2;     // 38.212, section 7.2.1
            uint32_t tbSizeBytes  = m_pTbPrmsCpu[ueIdx].tbSize / 8 + crcSizeBytes; // in cuPHY each TB includes TB payload + TB CRC
            m_outputPrms.totNumPayloadBytes += tbSizeBytes;

            uint32_t tbWordAlignPaddingBytes = (sizeof(uint32_t) - (tbSizeBytes % sizeof(uint32_t))) % sizeof(uint32_t);
            m_outputPrms.totNumPayloadBytes += tbWordAlignPaddingBytes;

            /* printf("ueIdx %d tbSize %d tbSizeBytes %d totNumPayloadBytes %d pStartOffsetsTbPayload %d\n",
                ueIdx, m_tbPrmsArrayCpu[ueIdx].tbSize, m_tbPrmsArrayCpu[ueIdx].tbSize / 8, m_outputPrms.totNumPayloadBytes,
                pDynPrm->pDataOut->pStartOffsetsTbPayload[ueIdx]); */
        }
        else
        {
            //TODO
            pDynPrm->pDataOut->pStartOffsetsTbPayload[ueIdx] = 0;
            pDynPrm->pDataOut->pStartOffsetsTbCrc[ueIdx]     = 0;
            pDynPrm->pDataOut->pStartOffsetsCbCrc[ueIdx]     = 0;
        }
    }

    // Bundled data is allocated based on max case scenario; update start addresses based on actual size within m_h2dBuffer
    // updateInputBuf() should be called after expandBackEndParameters() and before expandUciParameters()
    updateInputBuf();

    expandUciParameters(false);

    pDynPrm->pDataOut->totNumUciSegs        = m_outputPrms.totNumUciSegs;
    m_outputPrms.pTbCrcsHost                = pDynPrm->pDataOut->pTbCrcs;
    m_outputPrms.pCbCrcsHost                = pDynPrm->pDataOut->pCbCrcs;
    m_outputPrms.pTbPayloadsHost            = pDynPrm->pDataOut->pTbPayloads;
    m_outputPrms.pTaEstsHost                = pDynPrm->pDataOut->pTaEsts;
    m_outputPrms.pRssiHost                  = pDynPrm->pDataOut->pRssi;
    m_outputPrms.pRsrpHost                  = pDynPrm->pDataOut->pRsrp;
    m_outputPrms.pNoiseVarPreEqHost         = pDynPrm->pDataOut->pNoiseVarPreEq;
    m_outputPrms.pNoiseVarPostEqHost        = pDynPrm->pDataOut->pNoiseVarPostEq;
    m_outputPrms.pSinrPreEqHost             = pDynPrm->pDataOut->pSinrPreEq;
    m_outputPrms.pSinrPostEqHost            = pDynPrm->pDataOut->pSinrPostEq;
    m_outputPrms.pCfoHzHost                 = pDynPrm->pDataOut->pCfoHz;
    m_outputPrms.pUciPayloadsHost           = pDynPrm->pDataOut->pUciPayloads;
    m_outputPrms.pHarqDetectionStatusHost   = pDynPrm->pDataOut->HarqDetectionStatus;
    m_outputPrms.pCsiP1DetectionStatusHost  = pDynPrm->pDataOut->CsiP1DetectionStatus;
    m_outputPrms.pCsiP2DetectionStatusHost  = pDynPrm->pDataOut->CsiP2DetectionStatus;
    m_outputPrms.pUciCrcFlagsHost           = pDynPrm->pDataOut->pUciCrcFlags;
    m_outputPrms.pNumCsi2BitsHost           = pDynPrm->pDataOut->pNumCsi2Bits;

    pDynPrm->pDataOut->totNumTbs            = m_outputPrms.totNumTbs;
    pDynPrm->pDataOut->totNumCbs            = m_outputPrms.totNumCbs;
    pDynPrm->pDataOut->totNumPayloadBytes   = m_outputPrms.totNumPayloadBytes;
    //m_outputPrms.pUciDTXsHost          = pDynPrm->pDataOut->pUciDTXs;


    // Input tensors
    m_tPrmDataRx   = *(pDynPrm->pDataIn->pTDataRx);

    // memory
    allocateDeviceMemory(pDynPrm);
    
    //early-HARQ flags and memories
    m_earlyHarqModeEnabled = (pDynPrm->procModeBmsk & PUSCH_PROC_MODE_SUB_SLOT_EARLY_HARQ)? true : false;
    //over-write m_earlyHarqModeEnabled if necessary
    if(m_nEarlyHarqUes==0 || m_cuphyPuschStatPrms.enableEarlyHarq==0)
    {
        m_earlyHarqModeEnabled = false;
    }
    
    if(m_earlyHarqModeEnabled)
    {
        pDynPrm->pDataOut->isEarlyHarqPresent = 1;
    }
    else
    {
        pDynPrm->pDataOut->isEarlyHarqPresent = 0;
    }
    
    return CUPHY_STATUS_SUCCESS;
}

void PuschRx::setupCmnPhase2(cuphyPuschDynPrms_t* pDynPrm)
{
    // ChEst and CFO Est initialize nCfgs within the component
    for(int32_t chEstTimeInst = 0; chEstTimeInst < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEstTimeInst)
    {
        m_chEstLaunchCfgs[chEstTimeInst].nCfgs = 0;
    }
    for(int32_t chEqTimeInst = 0; chEqTimeInst < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEqTimeInst)
    {
        m_chEqCoefCompLaunchCfgs[chEqTimeInst].nCfgs = 0;
    }
    for(int32_t idx = 0; idx < CUPHY_MAX_PUSCH_EXECUTION_PATHS; ++idx)
    {
        m_noiseIntfEstLaunchCfgs[idx].nCfgs          = CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS;
        
        m_chEqSoftDemapLaunchCfgs[idx].nCfgs         = 0;
        m_chEqSoftDemapIdftLaunchCfgs[idx].nCfgs    = 0;
        m_chEqSoftDemapAfterDftLaunchCfgs[idx].nCfgs = 0;
    }
    m_cfoTaEstLaunchCfgs.nCfgs                         = 0;
    m_chEqSoftDemapBluesteinWorkspaceLaunchCfgs.nCfgs  = 0;    
    m_rssiLaunchCfgs.nCfgs                             = CUPHY_PUSCH_RX_RSSI_N_MAX_HET_CFGS;
    m_rsrpLaunchCfgs.nCfgs                             = CUPHY_PUSCH_RX_RSRP_N_MAX_HET_CFGS;

    
    ///////////////////////////////////////////////////////////////////
    ////   update ndi /////////////////////////////////////////////////
    PerTbParams* pPerTbPrms = m_pTbPrmsCpu;
    cuphyPuschCellGrpDynPrm_t cellGrpDynPrm = *(pDynPrm->pCellGrpDynPrm);
    cuphyPuschUePrm_t*        uePrmsArray   = pDynPrm->pCellGrpDynPrm->pUePrms;
    for(int i = 0; i < cellGrpDynPrm.nUes; i++)
    {
       // HARQ parameters
        pPerTbPrms[i].ndi = uePrmsArray[i].ndi;
    }
    ////////////////////////////////////////////////////////////////////   
    
    // LDPC kernel setup
    prepareLDPCStreamsTB();

    for(int i = 0; i < m_LDPCDecodeDescSet.count(); ++i)
    {
        m_ldpcLaunchCfgs[i].decode_desc = m_LDPCDecodeDescSet[i];
        m_LDPCdecoder.get_launch_config(m_ldpcLaunchCfgs[i]);
        // printf("LDPC: decSetIdx %d gridDim (x y z) (%d %d %d) blockDim (x y z) (%d %d %d) \n", i, m_ldpcLaunchCfgs[i].kernel_node_params_driver.gridDimX, m_ldpcLaunchCfgs[i].kernel_node_params_driver.gridDimY, m_ldpcLaunchCfgs[i].kernel_node_params_driver.gridDimZ, m_ldpcLaunchCfgs[i].kernel_node_params_driver.blockDimX, m_ldpcLaunchCfgs[i].kernel_node_params_driver.blockDimY, m_ldpcLaunchCfgs[i].kernel_node_params_driver.blockDimZ);
    }
    // printf("PuschRx setupCmn: LDPC node count %d\n", m_LDPCDecodeDescSet.count());

    // debug output
    if(m_outputPrms.debugOutputFlag)
    {
        cuphyPuschCellGrpDynPrm_t const& cellGrpDynPrm = *(pDynPrm->pCellGrpDynPrm);
        cuphyTensorPrm_t* pTDataRx = pDynPrm->pDataIn->pTDataRx;

        for(int32_t cellIdx = 0; cellIdx < cellGrpDynPrm.nCells; ++cellIdx)
        {
            int32_t cellPrmStatIdx = cellGrpDynPrm.pCellPrms[cellIdx].cellPrmStatIdx;
            int32_t cellPrmDynIdx  = cellGrpDynPrm.pCellPrms[cellIdx].cellPrmDynIdx;

            const int NF       = CUPHY_N_TONES_PER_PRB * m_cuphyCellStatPrmVecCpu[cellPrmStatIdx].nPrbUlBwp;
            const int NT       = 14;
            const int N_BS_ANT = m_cuphyCellStatPrmVecCpu[cellPrmStatIdx].nRxAnt;

            m_tRefDataRx[cellIdx].desc().set(CUPHY_C_16F, NF, NT, N_BS_ANT, cuphy::tensor_flags::align_tight);
            m_tRefDataRx[cellIdx].set_addr(pTDataRx[cellPrmDynIdx].pAddr);
        }
    }
}

void PuschRx::expandUciCodingPrms(uint32_t nInfoBits, uint32_t nRmBits, uint8_t Qm, float DTXthreshold, bool updateOnlyNumInputPrms,
    uint16_t& nRmCws, cuphyRmCwPrm_t* pRmCwPrms, uint16_t& nSpxCws, cuphySimplexCwPrm_t* pSpxCwPrms,
    uint16_t& nPolUciSegs, uint16_t& nPolCbs, cuphyPolarUciSegPrm_t* pUciSegPrms, cuphyPolarCwPrm_t* pUciCwPrms)
{
    if(nInfoBits <= CUPHY_N_MAX_UCI_BITS_SIMPLEX)
    {
        if(!updateOnlyNumInputPrms)
        {
            m_outputPrms.totNumUciPayloadBytes += sizeof(uint32_t);
            pSpxCwPrms[nSpxCws].E               = nRmBits;
            pSpxCwPrms[nSpxCws].K               = nInfoBits;
            pSpxCwPrms[nSpxCws].nBitsPerQam     = Qm;
            pSpxCwPrms[nSpxCws].exitFlag        = 0;
            pSpxCwPrms[nSpxCws].en_DTXest       = CUPHY_DTX_EN + CUPHY_DET_EN;
            pSpxCwPrms[nSpxCws].DTXthreshold    = DTXthreshold;
        }
        nSpxCws += 1;
    }
    else if(nInfoBits <= CUPHY_N_MAX_UCI_BITS_RM)
    {
        if(!updateOnlyNumInputPrms)
        {
            m_outputPrms.totNumUciPayloadBytes    += sizeof(uint32_t);
            pRmCwPrms[nRmCws].K             = nInfoBits;
            pRmCwPrms[nRmCws].E             = nRmBits;
            pRmCwPrms[nRmCws].exitFlag      = 0;
            pRmCwPrms[nRmCws].en_DTXest     = CUPHY_DTX_EN + CUPHY_DET_EN;
            pRmCwPrms[nRmCws].DTXthreshold  = DTXthreshold;
        }
        nRmCws += 1;
    }
    else
    {
        if (updateOnlyNumInputPrms)
        {
            int nCbs = 0;
            if (((nInfoBits >= 360) && (nRmBits >= 1088)) || (nInfoBits >= 1013)) {
                nCbs = 2;
            } else {
                nCbs = 1;
            }

            nPolCbs     += nCbs;
            nPolUciSegs += 1;

        }
        else
        {
            cuphyPolarUciSegPrm_t& uciSegPrms = pUciSegPrms[nPolUciSegs];
            uciSegPrms.exitFlag = 0;

            // crc size (38.212 6.3.1.2.1)
            uciSegPrms.nCrcBits = (nInfoBits <= 19) ? 6 : 11;

            // code block segmentation (38.212 6.3.1.3.1)
            // code block size         (38.212 5.2.1)
            if(((nInfoBits >= 360) && (nRmBits >= 1088)) || (nInfoBits >= 1013))
            {
                uciSegPrms.nCbs           = 2;
                uciSegPrms.K_cw           = div_round_up(nInfoBits, static_cast<uint32_t>(2)) + uciSegPrms.nCrcBits;
                uciSegPrms.E_cw           = nRmBits / 2;
                uciSegPrms.zeroInsertFlag = nInfoBits % 2;
            }
            else
            {
                uciSegPrms.nCbs           = 1;
                uciSegPrms.K_cw           = nInfoBits + uciSegPrms.nCrcBits;
                uciSegPrms.E_cw           = nRmBits;
                uciSegPrms.zeroInsertFlag = 0;
            }

            // encoded cb(s) size (38.212 5.3.1)
            uint32_t n_temp        = static_cast<uint32_t>(ceil(log2(static_cast<double>(uciSegPrms.E_cw))) - 1);
            uint32_t two_to_n_temp = 1 << n_temp;

            uint32_t n_1;
            if((8 * uciSegPrms.E_cw <= 9 * two_to_n_temp) && (16 * uciSegPrms.K_cw <= 9 * uciSegPrms.E_cw))
            {
                n_1 = n_temp;
            }
            else
            {
                n_1 = n_temp + 1;
            }

            uint32_t n_2   = static_cast<uint32_t>(ceil(log2(static_cast<double>(uciSegPrms.K_cw) * 8)));
            uint32_t n_min = 5;
            uint32_t n_max = 10;

            uciSegPrms.n_cw = std::max(std::min(std::min(n_1, n_2), n_max), n_min);
            uciSegPrms.N_cw = 1 << uciSegPrms.n_cw;

            // child cb(s)
            for(int i = 0; i < uciSegPrms.nCbs; ++i)
            {
                cuphyPolarCwPrm_t& cwPrms = pUciCwPrms[nPolCbs];
                cwPrms.exitFlag = 0;

                cwPrms.N_cw     = uciSegPrms.N_cw;
                cwPrms.nCrcBits = uciSegPrms.nCrcBits;
                cwPrms.A_cw     = uciSegPrms.K_cw - uciSegPrms.nCrcBits;

                cwPrms.nCbsInUciSeg      = uciSegPrms.nCbs;
                cwPrms.cbIdxWithinUciSeg = i;
                cwPrms.zeroInsertFlag    = uciSegPrms.zeroInsertFlag;

                uciSegPrms.childCbIdxs[i] = nPolCbs;
                nPolCbs += 1;
            }
            uint16_t nUciSegBits = uciSegPrms.nCbs * (uciSegPrms.K_cw - uciSegPrms.nCrcBits) - uciSegPrms.zeroInsertFlag;
            m_outputPrms.totNumUciPayloadBytes += 4 * div_round_up(nUciSegBits, static_cast<uint16_t>(32));
            nPolUciSegs += 1;
        }
    }
}

void PuschRx::expandUciParameters(bool updateOnlyNumInputPrms)
{
    m_outputPrms.totNumUciSegs         = 0;
    m_outputPrms.totNumUciPayloadBytes = 0;
    m_nSpxCws                          = 0;
    m_nRmCws                           = 0;
    m_nPolUciSegs                      = 0;
    m_nPolCbs                          = 0;
    m_nUciUes                          = 0;
    m_nCsi2Ues                         = 0;
    m_nRmCws_csi2                      = 0;
    m_nSpxCws_csi2                     = 0;
    m_nPolUciSegs_csi2                 = 0;
    m_nPolCbs_csi2                     = 0;
    m_nRmCws_early                     = 0;
    m_nSpxCws_early                    = 0;
    m_nPolUciSegs_early                = 0;
    m_nPolCbs_early                    = 0;

    uint32_t           nUes    = m_cuphyPuschCellGrpDynPrm.nUes;
    cuphyPuschUePrm_t* pUePrms = m_cuphyPuschCellGrpDynPrm.pUePrms;

    for(int ueIdx = 0; ueIdx < nUes; ++ueIdx)
    {
        if(m_pTbPrmsCpu[ueIdx].uciOnPuschFlag)
        {
            m_uciUserIdxsVec[m_nUciUes] = ueIdx;
            m_nUciUes += 1;

            float DTXthreshold = pUePrms[ueIdx].pUciPrms->DTXthreshold;
            // Per 3GPP section 6.3.2.1.1, if
            // UCI is transmitted on PUSCH without SCH without CSI-P2 and with CSI-P1
            // then even if there is one HARQ bit, the UCI bit sequence contains 2 bits (a0 and a1).
            if((m_pTbPrmsCpu[ueIdx].isDataPresent == 0) && (m_pTbPrmsCpu[ueIdx].csi2Flag == 0) && (pUePrms[ueIdx].pUciPrms->nBitsCsi1 > 0) && (pUePrms[ueIdx].pUciPrms->nBitsHarq == 1))
            {
                pUePrms[ueIdx].pUciPrms->nBitsHarq = 2;
            }
            uint32_t nBitsHarq = pUePrms[ueIdx].pUciPrms->nBitsHarq;
            if(nBitsHarq > 0)
            {
                if(m_pTbPrmsCpu[ueIdx].isEarlyHarq)
                {
                    expandUciCodingPrms(nBitsHarq, m_pTbPrmsCpu[ueIdx].G_harq, m_pTbPrmsCpu[ueIdx].Qm, DTXthreshold, updateOnlyNumInputPrms,
                    m_nRmCws_early, m_pRmCwPrmsCpu_early, m_nSpxCws_early, m_pSpxCwPrmsCpu_early, m_nPolUciSegs_early, m_nPolCbs_early, m_pUciSegPrmsCpu_early, m_pUciCwPrmsCpu_early);
                    m_outputPrms.totNumUciSegs += 1;
                }else
                {
                    expandUciCodingPrms(nBitsHarq, m_pTbPrmsCpu[ueIdx].G_harq, m_pTbPrmsCpu[ueIdx].Qm, DTXthreshold, updateOnlyNumInputPrms,
                    m_nRmCws, m_pRmCwPrmsCpu, m_nSpxCws, m_pSpxCwPrmsCpu, m_nPolUciSegs, m_nPolCbs, m_pUciSegPrmsCpu, m_pUciCwPrmsCpu);
                    m_outputPrms.totNumUciSegs += 1;
                }
            }

            uint32_t nBitsCsi1 = pUePrms[ueIdx].pUciPrms->nBitsCsi1;
            if(nBitsCsi1 > 0)
            {
                expandUciCodingPrms(nBitsCsi1, m_pTbPrmsCpu[ueIdx].G_csi1, m_pTbPrmsCpu[ueIdx].Qm, DTXthreshold, updateOnlyNumInputPrms,
                m_nRmCws, m_pRmCwPrmsCpu, m_nSpxCws, m_pSpxCwPrmsCpu, m_nPolUciSegs, m_nPolCbs, m_pUciSegPrmsCpu, m_pUciCwPrmsCpu);                
                m_outputPrms.totNumUciSegs += 1;
            }

            if((pUePrms[ueIdx].pduBitmap >> 5) & 1)
            {
                m_csi2UeIdxsVec[m_nCsi2Ues] = ueIdx;
                m_nCsi2Ues         += 1;
                m_nRmCws_csi2      += 1;
                m_nSpxCws_csi2     += 1;
                m_nPolUciSegs_csi2 += 1;
                m_nPolCbs_csi2     += 2;
                m_outputPrms.totNumUciPayloadBytes += CUPHY_MAX_N_CSI2_WORDS * sizeof(uint32_t);
            }
        }
    }
}

cuphyStatus_t PuschRx::setupComponents(bool enableCpuToGpuDescrAsyncCpy, cuphyPuschDynPrms_t* pDynPrm)
{
    auto dynCpuDescrStartAddrs = m_kernelDynDescr.getCpuStartAddrs();
    auto dynGpuDescrStartAddrs = m_kernelDynDescr.getGpuStartAddrs();

    cuphyStatus_t chEstSetupStatus = cuphySetupPuschRxChEst(m_chEstHndl,
                                                            m_drvdUeGrpPrmsCpu,
                                                            m_drvdUeGrpPrmsGpu,
                                                            m_cuphyPuschCellGrpDynPrm.nUeGrps,
                                                            m_chEstSettings.enableDftSOfdm,
                                                            pDynPrm->pDataOut->pPreEarlyHarqWaitKernelStatus_d,
                                                            pDynPrm->pDataOut->pPostEarlyHarqWaitKernelStatus_d,
                                                            pDynPrm->waitTimeOutPreEarlyHarqUs,
                                                            pDynPrm->waitTimeOutPostEarlyHarqUs,
                                                            enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                            reinterpret_cast<void**>(&dynCpuDescrStartAddrs[PUSCH_CH_EST]),
                                                            reinterpret_cast<void**>(&dynGpuDescrStartAddrs[PUSCH_CH_EST]),
                                                            m_chEstLaunchCfgs,
                                                            m_earlyHarqModeEnabled,
                                                            &m_preEarlyHarqWaitCfgs,
                                                            &m_postEarlyHarqWaitCfgs,
                                                            m_cuStream);

    cuphyStatus_t noiseIntfEstSetupStatus = CUPHY_STATUS_SUCCESS;
    cuphyStatus_t noiseIntfEstEarlyHarqSetupStatus = CUPHY_STATUS_SUCCESS;
    if(m_chEstSettings.enableSinrMeasurement || (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE)|| (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC))
    {
        if(m_earlyHarqModeEnabled)
        {
            noiseIntfEstEarlyHarqSetupStatus = cuphySetupPuschRxNoiseIntfEst(m_noiseIntfEstHndl,
                                                                                      m_drvdUeGrpPrmsCpu,
                                                                                      m_drvdUeGrpPrmsGpu,
                                                                                      m_cuphyPuschCellGrpDynPrm.nUeGrps,
                                                                                      m_nMaxPrb,
                                                                                      m_chEstSettings.enableDftSOfdm,
                                                                                      CUPHY_PUSCH_NOISE_EST_DMRS_ADDITIONAL_POS_0, // for the early-HARQ processing
                                                                                      enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                                      static_cast<void*>(dynCpuDescrStartAddrs[PUSCH_NOISE_INTF_EST]),
                                                                                      static_cast<void*>(dynGpuDescrStartAddrs[PUSCH_NOISE_INTF_EST]),
                                                                                      &m_noiseIntfEstLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH],
                                                                                      m_cuStream);
        }
                                                                
        noiseIntfEstSetupStatus = cuphySetupPuschRxNoiseIntfEst(m_noiseIntfEstHndl,
                                                                m_drvdUeGrpPrmsCpu,
                                                                m_drvdUeGrpPrmsGpu,
                                                                m_cuphyPuschCellGrpDynPrm.nUeGrps,
                                                                m_nMaxPrb,
                                                                m_chEstSettings.enableDftSOfdm,
                                                                CUPHY_PUSCH_NOISE_EST_DMRS_FULL_SLOT, // for full-slot processing
                                                                enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                static_cast<void*>(dynCpuDescrStartAddrs[PUSCH_NOISE_INTF_EST]),
                                                                static_cast<void*>(dynGpuDescrStartAddrs[PUSCH_NOISE_INTF_EST]),
                                                                &m_noiseIntfEstLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH],
                                                                m_cuStream);
    }

    cuphyStatus_t cfoTaEstSetupStatus = CUPHY_STATUS_SUCCESS;
    if(m_chEstSettings.enableCfoCorrection || m_chEstSettings.enableToEstimation)
    {
        cfoTaEstSetupStatus = cuphySetupPuschRxCfoTaEst(
            m_cfoTaEstHndl,
            m_drvdUeGrpPrmsCpu,
            m_drvdUeGrpPrmsGpu,
            m_cuphyPuschCellGrpDynPrm.nUeGrps,
            m_nMaxPrb,
            0,
            enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
            static_cast<void*>(dynCpuDescrStartAddrs[PUSCH_CFO_TA_EST]),
            static_cast<void*>(dynGpuDescrStartAddrs[PUSCH_CFO_TA_EST]),
            &m_cfoTaEstLaunchCfgs,
            m_cuStream);
    }

    cuphyStatus_t coefComputeSetupStatus = cuphySetupPuschRxChEqCoefCompute(m_chEqHndl,
                                                                            m_drvdUeGrpPrmsCpu,
                                                                            m_drvdUeGrpPrmsGpu,
                                                                            m_cuphyPuschCellGrpDynPrm.nUeGrps,
                                                                            m_nMaxPrb,
                                                                            m_chEstSettings.enableCfoCorrection,
                                                                            m_chEstSettings.enablePuschTdi,
                                                                            enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                            reinterpret_cast<void**>(&dynCpuDescrStartAddrs[PUSCH_CH_EQ_COEF]),
                                                                            reinterpret_cast<void**>(&dynGpuDescrStartAddrs[PUSCH_CH_EQ_COEF]),
                                                                            m_chEqCoefCompLaunchCfgs,
                                                                            m_cuStream);
    cuphyStatus_t setupSoftDemapEarlyHarqStatus = CUPHY_STATUS_SUCCESS;
    if(m_earlyHarqModeEnabled)
    {                                                                     
        setupSoftDemapEarlyHarqStatus = cuphySetupPuschRxChEqSoftDemap(m_chEqHndl,
                                                                       m_drvdUeGrpPrmsCpu,
                                                                       m_drvdUeGrpPrmsGpu,
                                                                       static_cast<uint16_t>(m_cuphyPuschCellGrpDynPrm.nUeGrps),
                                                                       m_nMaxPrb,
                                                                       m_chEstSettings.enableCfoCorrection,
                                                                       m_chEstSettings.enablePuschTdi,
                                                                       CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK, // for the early-HARQ processing at symbol 3
                                                                       enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                       static_cast<void*>(dynCpuDescrStartAddrs[PUSCH_CH_EQ_SOFT_DEMAP]),
                                                                       static_cast<void*>(dynGpuDescrStartAddrs[PUSCH_CH_EQ_SOFT_DEMAP]),
                                                                       &m_chEqSoftDemapLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH],
                                                                       m_cuStream);
    }

    cuphyStatus_t setupSoftDemapStatus = cuphySetupPuschRxChEqSoftDemap(m_chEqHndl,
                                                                        m_drvdUeGrpPrmsCpu,
                                                                        m_drvdUeGrpPrmsGpu,
                                                                        static_cast<uint16_t>(m_cuphyPuschCellGrpDynPrm.nUeGrps),
                                                                        m_nMaxPrb,
                                                                        m_chEstSettings.enableCfoCorrection,
                                                                        m_chEstSettings.enablePuschTdi,
                                                                        CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK, // for full-slot processing
                                                                        enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                        static_cast<void*>(dynCpuDescrStartAddrs[PUSCH_CH_EQ_SOFT_DEMAP]),
                                                                        static_cast<void*>(dynGpuDescrStartAddrs[PUSCH_CH_EQ_SOFT_DEMAP]),
                                                                        &m_chEqSoftDemapLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH],
                                                                        m_cuStream);

    cuphyStatus_t setupSoftDemapBluesteinWorkspaceStatus = CUPHY_STATUS_SUCCESS;
    cuphyStatus_t setupSoftDemapIdftStatus = CUPHY_STATUS_SUCCESS;
    cuphyStatus_t setupSoftDemapAfterDftStatus = CUPHY_STATUS_SUCCESS;
    cuphyStatus_t setupSoftDemapIdftEarlyHarqStatus = CUPHY_STATUS_SUCCESS;
    cuphyStatus_t setupSoftDemapAfterDftEarlyHarqStatus = CUPHY_STATUS_SUCCESS;
    if(m_chEstSettings.enableDftSOfdm==1)
    {
        setupSoftDemapBluesteinWorkspaceStatus = cuphySetupPuschRxChEqSoftDemapBluesteinWorkspace(m_chEqHndl,
                                                                            m_drvdUeGrpPrmsCpu,
                                                                            m_drvdUeGrpPrmsGpu,
                                                                            static_cast<uint16_t>(m_cuphyPuschCellGrpDynPrm.nUeGrps),
                                                                            m_nMaxPrb,
                                                                            m_chEstSettings.enableCfoCorrection,
                                                                            m_chEstSettings.enablePuschTdi,
                                                                            enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                            static_cast<void*>(dynCpuDescrStartAddrs[PUSCH_CH_EQ_SOFT_DEMAP]),
                                                                            static_cast<void*>(dynGpuDescrStartAddrs[PUSCH_CH_EQ_SOFT_DEMAP]),
                                                                            &m_chEqSoftDemapBluesteinWorkspaceLaunchCfgs,
                                                                            m_cuStream);
                                                                            
        if(m_earlyHarqModeEnabled)
        {   
            setupSoftDemapIdftEarlyHarqStatus = cuphySetupPuschRxChEqSoftDemapIdft(m_chEqHndl,
                                                                            m_drvdUeGrpPrmsCpu,
                                                                            m_drvdUeGrpPrmsGpu,
                                                                            static_cast<uint16_t>(m_cuphyPuschCellGrpDynPrm.nUeGrps),
                                                                            m_nMaxPrb,
                                                                            m_chEstSettings.enableCfoCorrection,
                                                                            m_chEstSettings.enablePuschTdi,
                                                                            CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK, // for the early-HARQ processing at symbol 3
                                                                            enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                            static_cast<void*>(dynCpuDescrStartAddrs[PUSCH_CH_EQ_SOFT_DEMAP]),
                                                                            static_cast<void*>(dynGpuDescrStartAddrs[PUSCH_CH_EQ_SOFT_DEMAP]),
                                                                            &m_chEqSoftDemapIdftLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH],
                                                                            m_cuStream);

            setupSoftDemapAfterDftEarlyHarqStatus = cuphySetupPuschRxChEqSoftDemapAfterDft(m_chEqHndl,
                                                                            m_drvdUeGrpPrmsCpu,
                                                                            m_drvdUeGrpPrmsGpu,
                                                                            static_cast<uint16_t>(m_cuphyPuschCellGrpDynPrm.nUeGrps),
                                                                            m_nMaxPrb,
                                                                            m_chEstSettings.enableCfoCorrection,
                                                                            m_chEstSettings.enablePuschTdi,
                                                                            CUPHY_PUSCH_RX_SOFT_DEMAPPER_EARLY_HARQ_SYMBOL_BITMASK, // for the early-HARQ processing at symbol 3
                                                                            enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                            static_cast<void*>(dynCpuDescrStartAddrs[PUSCH_CH_EQ_SOFT_DEMAP]),
                                                                            static_cast<void*>(dynGpuDescrStartAddrs[PUSCH_CH_EQ_SOFT_DEMAP]),
                                                                            &m_chEqSoftDemapAfterDftLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH],
                                                                            m_cuStream);
        
        
        } 

        setupSoftDemapIdftStatus = cuphySetupPuschRxChEqSoftDemapIdft(m_chEqHndl,
                                                                            m_drvdUeGrpPrmsCpu,
                                                                            m_drvdUeGrpPrmsGpu,
                                                                            static_cast<uint16_t>(m_cuphyPuschCellGrpDynPrm.nUeGrps),
                                                                            m_nMaxPrb,
                                                                            m_chEstSettings.enableCfoCorrection,
                                                                            m_chEstSettings.enablePuschTdi,
                                                                            CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK, // for full-slot processing
                                                                            enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                            static_cast<void*>(dynCpuDescrStartAddrs[PUSCH_CH_EQ_SOFT_DEMAP]),
                                                                            static_cast<void*>(dynGpuDescrStartAddrs[PUSCH_CH_EQ_SOFT_DEMAP]),
                                                                            &m_chEqSoftDemapIdftLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH],
                                                                            m_cuStream);

        setupSoftDemapAfterDftStatus = cuphySetupPuschRxChEqSoftDemapAfterDft(m_chEqHndl,
                                                                            m_drvdUeGrpPrmsCpu,
                                                                            m_drvdUeGrpPrmsGpu,
                                                                            static_cast<uint16_t>(m_cuphyPuschCellGrpDynPrm.nUeGrps),
                                                                            m_nMaxPrb,
                                                                            m_chEstSettings.enableCfoCorrection,
                                                                            m_chEstSettings.enablePuschTdi,
                                                                            CUPHY_PUSCH_RX_SOFT_DEMAPPER_FULL_SLOT_SYMBOL_BITMASK, // for full-slot processing
                                                                            enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                            static_cast<void*>(dynCpuDescrStartAddrs[PUSCH_CH_EQ_SOFT_DEMAP]),
                                                                            static_cast<void*>(dynGpuDescrStartAddrs[PUSCH_CH_EQ_SOFT_DEMAP]),
                                                                            &m_chEqSoftDemapAfterDftLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH],
                                                                            m_cuStream);
    }

    if(m_nUciUes > 0)
    {
        cuphyUciToSeg_t uciToSeg = SEG_ALL_UCI;
        cuphySetupUciOnPuschSegLLRs0(m_uciOnPuschSegLLRs0Hndl,
                                     m_nUciUes,
                                     m_uciUserIdxsVec.data(),
                                     m_pTbPrmsCpu,
                                     m_pTbPrmsGpu,
                                     m_cuphyPuschCellGrpDynPrm.nUeGrps,
                                     m_tPrmLLRVec.data(),
                                     m_drvdUeGrpPrmsCpu,
                                     m_drvdUeGrpPrmsGpu,
                                     uciToSeg,
                                     static_cast<void*>(dynCpuDescrStartAddrs[PUSCH_SEG_UCI_LLRS0]),
                                     static_cast<void*>(dynGpuDescrStartAddrs[PUSCH_SEG_UCI_LLRS0]),
                                     enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                     &m_uciOnPuschSegLLRs0LaunchCfg,
                                     m_cuStream);
    }

    if(m_nEarlyHarqUes > 0)
    {
        cuphyUciToSeg_t uciToSeg = SEG_ONLY_EARLY_UCI;
        cuphySetupUciOnPuschSegLLRs0(m_uciOnPuschEarlySegLLRs0Hndl,
                                     m_nUciUes,
                                     m_uciUserIdxsVec.data(),
                                     m_pTbPrmsCpu,
                                     m_pTbPrmsGpu,
                                     m_cuphyPuschCellGrpDynPrm.nUeGrps,
                                     m_tPrmLLRVec.data(),
                                     m_drvdUeGrpPrmsCpu,
                                     m_drvdUeGrpPrmsGpu,
                                     uciToSeg,
                                     static_cast<void*>(dynCpuDescrStartAddrs[PUSCH_SEG_EARLY_UCI_LLRS0]),
                                     static_cast<void*>(dynGpuDescrStartAddrs[PUSCH_SEG_EARLY_UCI_LLRS0]),
                                     enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                     &m_uciOnPuschEarlySegLLRs0LaunchCfg,
                                     m_cuStream);
    }

    if(m_nSpxCws > 0)
    {
        cuphySetupSimplexDecoder(m_spxDecoderHndl,
                                 m_nSpxCws,
                                 m_pSpxCwPrmsCpu,
                                 m_pSpxPrmsGpu,
                                 static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                 static_cast<void*>(dynCpuDescrStartAddrs[SPX_DECODE]),
                                 static_cast<void*>(dynGpuDescrStartAddrs[SPX_DECODE]),
                                 &m_simplexDecoderLaunchCfg,
                                 m_cuStream);
    }

    if(m_nSpxCws_early > 0)
    {
        cuphySetupSimplexDecoder(m_spxDecoderHndl_early,
                                 m_nSpxCws_early,
                                 m_pSpxCwPrmsCpu_early,
                                 m_pSpxPrmsGpu_early,
                                 static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                 static_cast<void*>(dynCpuDescrStartAddrs[SPX_DECODE_EARLY]),
                                 static_cast<void*>(dynGpuDescrStartAddrs[SPX_DECODE_EARLY]),
                                 &m_simplexDecoderLaunchCfg_early,
                                 m_cuStream);
    }

    if(m_nRmCws > 0)
    {
        cuphySetupRmDecoder(m_rmDecodeHndl,
                            m_nRmCws,
                            m_pRmCwPrmsGpu,
                            static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                            static_cast<void*>(dynCpuDescrStartAddrs[RM_DECODE]),
                            static_cast<void*>(dynGpuDescrStartAddrs[RM_DECODE]),
                            &m_rmDecoderLaunchCfg,
                            m_cuStream);
    }

    if(m_nRmCws_early > 0)
    {
        cuphySetupRmDecoder(m_rmDecodeHndl_early,
                            m_nRmCws_early,
                            m_pRmCwPrmsGpu_early,
                            static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                            static_cast<void*>(dynCpuDescrStartAddrs[RM_DECODE_EARLY]),
                            static_cast<void*>(dynGpuDescrStartAddrs[RM_DECODE_EARLY]),
                            &m_rmDecoderLaunchCfg_early,
                            m_cuStream);
    }

    if(m_nPolUciSegs > 0)
    {
        cuphyStatus_t compCwTreeTypesSetupStatus = cuphySetupCompCwTreeTypes(m_compCwTreeTypesHndl,
                                                                             m_nPolUciSegs,
                                                                             m_pUciSegPrmsCpu,
                                                                             m_pUciSegPrmsGpu,
                                                                             m_cwTreeTypesAddrVec.data(),
                                                                             static_cast<void*>(dynCpuDescrStartAddrs[POL_COMP_CW_TREE]),
                                                                             static_cast<void*>(dynGpuDescrStartAddrs[POL_COMP_CW_TREE]),
                                                                             static_cast<void*>(dynCpuDescrStartAddrs[POL_COMP_CW_TREE_ADDRS]),
                                                                             enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                             &m_compCwTreeTypesLaunchCfg,
                                                                             m_cuStream);

        cuphyStatus_t polSegDeRmDeItlSetupStatus = cuphySetupPolSegDeRmDeItl(m_polSegDeRmDeItlHndl,
                                                                             m_nPolUciSegs,
                                                                             m_nPolCbs,
                                                                             m_pUciSegPrmsCpu,
                                                                             m_pUciSegPrmsGpu,
                                                                             m_pUciCwPrmsCpu,
                                                                             m_pUciCwPrmsGpu, 
                                                                             m_uciSegLLRsAddrVec.data(),
                                                                             m_cwLLRsAddrVec.data(),
                                                                             static_cast<void*>(dynCpuDescrStartAddrs[POL_SEG_DERM_DEITL]),
                                                                             static_cast<void*>(dynGpuDescrStartAddrs[POL_SEG_DERM_DEITL]),
                                                                             static_cast<void*>(dynCpuDescrStartAddrs[POL_SEG_DERM_DEITL_CW_ADDRS]),
                                                                             static_cast<void*>(dynCpuDescrStartAddrs[POL_SEG_DERM_DEITL_UCI_ADDRS]),
                                                                             enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                             &m_polSegDeRmDeItlLaunchCfg,
                                                                             m_cuStream);

        cuphyStatus_t polarDecoderSetupStatus = cuphySetupPolarDecoder(m_polarDecoderHndl,
                                                                       m_nPolCbs,
                                                                       m_cwTreeLLRsAddrVec.data(),
                                                                       m_pUciCwPrmsGpu,
                                                                       m_pUciCwPrmsCpu,
                                                                       m_cbEstAddrVec.data(),
                                                                       m_listPolScratchAddrVec.data(),
                                                                       m_polDcdrListSz,
                                                                       m_pPolCrcFlags,
                                                                       static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                                       static_cast<void*>(dynCpuDescrStartAddrs[POL_DECODE]),
                                                                       static_cast<void*>(dynGpuDescrStartAddrs[POL_DECODE]),
                                                                       static_cast<void*>(dynCpuDescrStartAddrs[POL_DECODE_LLR_ADDRS]),
                                                                       static_cast<void*>(dynCpuDescrStartAddrs[POL_DECODE_CB_ADDRS]),
                                                                       static_cast<void*>(dynCpuDescrStartAddrs[LIST_POL_DECODE_SCRATCH_ADDRS]),
                                                                       &m_polarDecoderLaunchCfg,
                                                                       m_cuStream);
    }

    if(m_nPolUciSegs_early > 0)
    {
        cuphyStatus_t compCwTreeTypesSetupStatus_early = cuphySetupCompCwTreeTypes(m_compCwTreeTypesHndl_early,
                                                                                    m_nPolUciSegs_early,
                                                                                    m_pUciSegPrmsCpu_early,
                                                                                    m_pUciSegPrmsGpu_early,
                                                                                    m_cwTreeTypesAddrVec_early.data(),
                                                                                    static_cast<void*>(dynCpuDescrStartAddrs[POL_COMP_CW_TREE_EARLY]),
                                                                                    static_cast<void*>(dynGpuDescrStartAddrs[POL_COMP_CW_TREE_EARLY]),
                                                                                    static_cast<void*>(dynCpuDescrStartAddrs[POL_COMP_CW_TREE_ADDRS_EARLY]),
                                                                                    enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                                    &m_compCwTreeTypesLaunchCfg_early,
                                                                                    m_cuStream);

        cuphyStatus_t polSegDeRmDeItlSetupStatus_early = cuphySetupPolSegDeRmDeItl(m_polSegDeRmDeItlHndl_early,
                                                                             m_nPolUciSegs_early,
                                                                             m_nPolCbs_early,
                                                                             m_pUciSegPrmsCpu_early,
                                                                             m_pUciSegPrmsGpu_early,
                                                                             m_pUciCwPrmsCpu_early,
                                                                             m_pUciCwPrmsGpu_early, 
                                                                             m_uciSegLLRsAddrVec_early.data(),
                                                                             m_cwLLRsAddrVec_early.data(),
                                                                             static_cast<void*>(dynCpuDescrStartAddrs[POL_SEG_DERM_DEITL_EARLY]),
                                                                             static_cast<void*>(dynGpuDescrStartAddrs[POL_SEG_DERM_DEITL_EARLY]),
                                                                             static_cast<void*>(dynCpuDescrStartAddrs[POL_SEG_DERM_DEITL_CW_ADDRS_EARLY]),
                                                                             static_cast<void*>(dynCpuDescrStartAddrs[POL_SEG_DERM_DEITL_UCI_ADDRS_EARLY]),
                                                                             enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                             &m_polSegDeRmDeItlLaunchCfg_early,
                                                                             m_cuStream);

        cuphyStatus_t polarDecoderSetupStatus_early = cuphySetupPolarDecoder(m_polarDecoderHndl_early,
                                                                            m_nPolCbs_early,
                                                                            m_cwTreeLLRsAddrVec_early.data(),
                                                                            m_pUciCwPrmsGpu_early,
                                                                            m_pUciCwPrmsCpu_early,
                                                                            m_cbEstAddrVec_early.data(),
                                                                            m_listPolScratchAddrVec_early.data(),
                                                                            m_polDcdrListSz,
                                                                            m_pPolCrcFlags_early,
                                                                            static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                                            static_cast<void*>(dynCpuDescrStartAddrs[POL_DECODE_EARLY]),
                                                                            static_cast<void*>(dynGpuDescrStartAddrs[POL_DECODE_EARLY]),
                                                                            static_cast<void*>(dynCpuDescrStartAddrs[POL_DECODE_LLR_ADDRS_EARLY]),
                                                                            static_cast<void*>(dynCpuDescrStartAddrs[POL_DECODE_CB_ADDRS_EARLY]),
                                                                            static_cast<void*>(dynCpuDescrStartAddrs[LIST_POL_DECODE_SCRATCH_ADDRS_EARLY]),
                                                                            &m_polarDecoderLaunchCfg_early,
                                                                            m_cuStream);
    }

    if(m_earlyHarqModeEnabled)
    {
        cuphySetD2HMemcpyNodeParams(&m_ehqUciPayloadMemcpyD2Hparams, m_outputPrms.pUciPayloadsDevice, m_outputPrms.pUciPayloadsHost, m_outputPrms.totNumUciPayloadBytes);
        cuphySetD2HMemcpyNodeParams(&m_ehqUciCrcMemcpyD2Hparams, m_outputPrms.pUciCrcFlagsDevice, m_outputPrms.pUciCrcFlagsHost, m_nPolUciSegs);
        cuphySetD2HMemcpyNodeParams(&m_ehqDetectionStatMemcpyD2Hparams, m_outputPrms.pHarqDetectionStatusDevice, m_outputPrms.pHarqDetectionStatusHost, m_nUes);
    }

    if(m_nCsi2Ues > 0)
    {
        cuphyStatus_t csi2CtrlSetupStatus = cuphySetupUciOnPuschCsi2Ctrl(m_uciOnPuschCsi2CtrlHndl,
                                                                         m_nCsi2Ues,
                                                                         m_csi2UeIdxsVec.data(),
                                                                         m_pTbPrmsCpu,
                                                                         m_pTbPrmsGpu,
                                                                         m_drvdUeGrpPrmsCpu,
                                                                         m_puschCellStatPrmBufGpu.addr(),
                                                                         m_outputPrms.pUciOnPuschOutOffsets,
                                                                         m_outputPrms.pUciPayloadsDevice,
                                                                         m_outputPrms.pNumCsi2BitsDevice,
                                                                         m_pUciSegPrmsGpu_csi2,
                                                                         m_pUciCwPrmsGpu_csi2,
                                                                         m_pRmCwPrmsGpu_csi2,
                                                                         m_pSpxPrmsGpu_csi2,
                                                                         m_cuphyPuschStatPrms.pDbg->forcedNumCsi2Bits,
                                                                         static_cast<void*>(dynCpuDescrStartAddrs[PUSCH_UCI_CSI2_CTRL]),
                                                                         static_cast<void*>(dynGpuDescrStartAddrs[PUSCH_UCI_CSI2_CTRL]),
                                                                         static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                                         &m_uciOnPuschCsi2CtrlLaunchCfg,
                                                                         m_cuStream);

        cuphyStatus_t segLLRs2SetupStatus = cuphySetupUciOnPuschSegLLRs2(m_uciOnPuschSegLLRs2Hndl,
                                                                         m_nCsi2Ues,
                                                                         m_csi2UeIdxsVec.data(),
                                                                         m_pTbPrmsCpu,
                                                                         m_pTbPrmsGpu,
                                                                         m_cuphyPuschCellGrpDynPrm.nUeGrps,
                                                                         m_tPrmLLRVec.data(),
                                                                         m_drvdUeGrpPrmsCpu,
                                                                         m_drvdUeGrpPrmsGpu,
                                                                         static_cast<void*>(dynCpuDescrStartAddrs[PUSCH_SEG_UCI_LLRS2]),
                                                                         static_cast<void*>(dynGpuDescrStartAddrs[PUSCH_SEG_UCI_LLRS2]),
                                                                         static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                                         &m_uciOnPuschSegLLRs2LaunchCfg,
                                                                         m_cuStream);

        cuphyStatus_t rmSetupStatus_csi2 = cuphySetupRmDecoder(m_rmDecodeHndl_csi2,
                                                               m_nCsi2Ues,
                                                               m_pRmCwPrmsGpu_csi2,
                                                               static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                               static_cast<void*>(dynCpuDescrStartAddrs[RM_DECODE_CSI2]),
                                                               static_cast<void*>(dynGpuDescrStartAddrs[RM_DECODE_CSI2]),
                                                               &m_rmDecoderLaunchCfg_csi2,
                                                               m_cuStream);

        cuphyStatus_t spxSetupStatus_csi2 = cuphySetupSimplexDecoder(m_spxDecoderHndl_csi2,
                                                                     m_nSpxCws_csi2,
                                                                     m_pSpxCwPrmsCpu_csi2,
                                                                     m_pSpxPrmsGpu_csi2,
                                                                     static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                                     static_cast<void*>(dynCpuDescrStartAddrs[SPX_DECODE_CSI2]),
                                                                     static_cast<void*>(dynGpuDescrStartAddrs[SPX_DECODE_CSI2]),
                                                                     &m_simplexDecoderLaunchCfg_csi2,
                                                                     m_cuStream);

        if(m_cuphyPuschStatPrms.pDbg->forcedNumCsi2Bits > 0)
        {  
            
            cuphyStatus_t compCwTreeTypesSetupStatus_csi2 = cuphySetupCompCwTreeTypes(m_compCwTreeTypesHndl_csi2,
                                                                                        m_nPolUciSegs_csi2,
                                                                                        m_pUciSegPrmsCpu_csi2,
                                                                                        m_pUciSegPrmsGpu_csi2,
                                                                                        m_cwTreeTypesAddrVec_csi2.data(),
                                                                                        static_cast<void*>(dynCpuDescrStartAddrs[POL_COMP_CW_TREE_CSI2]),
                                                                                        static_cast<void*>(dynGpuDescrStartAddrs[POL_COMP_CW_TREE_CSI2]),
                                                                                        static_cast<void*>(dynCpuDescrStartAddrs[POL_COMP_CW_TREE_ADDRS_CSI2]),
                                                                                        enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                                        &m_compCwTreeTypesLaunchCfg_csi2,
                                                                                        m_cuStream);

            cuphyStatus_t polSegDeRmDeItlSetupStatus_csi2 = cuphySetupPolSegDeRmDeItl(m_polSegDeRmDeItlHndl_csi2,
                                                                                        m_nPolUciSegs_csi2,
                                                                                        m_nPolCbs_csi2,
                                                                                        m_pUciSegPrmsCpu_csi2,
                                                                                        m_pUciSegPrmsGpu_csi2,
                                                                                        m_pUciCwPrmsCpu_csi2,
                                                                                        m_pUciCwPrmsGpu_csi2,       
                                                                                        m_uciSegLLRsAddrVec_csi2.data(),
                                                                                        m_cwLLRsAddrVec_csi2.data(),
                                                                                        static_cast<void*>(dynCpuDescrStartAddrs[POL_SEG_DERM_DEITL_CSI2]),
                                                                                        static_cast<void*>(dynGpuDescrStartAddrs[POL_SEG_DERM_DEITL_CSI2]),
                                                                                        static_cast<void*>(dynCpuDescrStartAddrs[POL_SEG_DERM_DEITL_CW_ADDRS_CSI2]),
                                                                                        static_cast<void*>(dynCpuDescrStartAddrs[POL_SEG_DERM_DEITL_UCI_ADDRS_CSI2]),
                                                                                        enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                                                        &m_polSegDeRmDeItlLaunchCfg_csi2,
                                                                                        m_cuStream);

            cuphyStatus_t polarDecoderSetupStatus_csi2 = cuphySetupPolarDecoder(m_polarDecoderHndl_csi2,
                                                                                m_nPolCbs_csi2,
                                                                                m_cwTreeLLRsAddrVec_csi2.data(),
                                                                                m_pUciCwPrmsGpu_csi2,
                                                                                m_pUciCwPrmsCpu_csi2,
                                                                                m_cbEstAddrVec_csi2.data(),
                                                                                m_listPolScratchAddrVec_csi2.data(),
                                                                                m_polDcdrListSz,
                                                                                m_pPolCrcFlags_csi2,
                                                                                static_cast<uint8_t>(enableCpuToGpuDescrAsyncCpy),
                                                                                static_cast<void*>(dynCpuDescrStartAddrs[POL_DECODE_CSI2]),
                                                                                static_cast<void*>(dynGpuDescrStartAddrs[POL_DECODE_CSI2]),
                                                                                static_cast<void*>(dynCpuDescrStartAddrs[POL_DECODE_LLR_ADDRS_CSI2]),
                                                                                static_cast<void*>(dynCpuDescrStartAddrs[POL_DECODE_CB_ADDRS_CSI2]),
                                                                                static_cast<void*>(dynCpuDescrStartAddrs[LIST_POL_DECODE_SCRATCH_ADDRS_CSI2]),
                                                                                &m_polarDecoderLaunchCfg_csi2,
                                                                                m_cuStream);
        }   
    }

    cuphyStatus_t setupRateMatchStatus;
    cuphyStatus_t setupCrcDecodeStatus;
    if(m_nSchUes > 0)
    {
        setupRateMatchStatus = cuphySetupPuschRxRateMatch(m_rateMatchHndl,                         // handle to rate-matching class
                                                          m_nSchUes,                               // number of users w/h SCH data
                                                          m_schUserIdxsVec.data(),                 // indicies of users w/h SCH data
                                                          m_pTbPrmsCpu,                            // starting adress of transport block paramters (CPU)
                                                          m_pTbPrmsGpu,                            // starting adress of transport block paramters (GPU)
                                                          m_tPrmLLRVec.data(),                     // starting adress of input LLR tensor parameters
                                                          m_tPrmLLRCdm1Vec.data(),
                                                          m_pHarqBuffers,                          // array of rm outputs in gpu
                                                          dynCpuDescrStartAddrs[PUSCH_RATE_MATCH], // pointer to descriptor in cpu
                                                          dynGpuDescrStartAddrs[PUSCH_RATE_MATCH], // pointer to descriptor in gpu
                                                          1,                                       // option to copy cpu descriptors from cpu to gpu
                                                          &m_rateMatchLaunchCfg,
                                                          m_cuStream); // stream to perform copy

        setupCrcDecodeStatus = cuphySetupPuschRxCrcDecode(m_crcDecodeHndl,
                                                          m_nSchUes,
                                                          m_schUserIdxsVec.data(),
                                                          m_outputPrms.pCbCrcsDevice,
                                                          m_outputPrms.pTbPayloadsDevice,
                                                          static_cast<uint32_t*>(d_pLDPCOut),
                                                          m_outputPrms.pTbCrcsDevice,
                                                          m_pTbPrmsCpu,
                                                          m_pTbPrmsGpu,
                                                          dynCpuDescrStartAddrs[PUSCH_CRC],
                                                          dynGpuDescrStartAddrs[PUSCH_CRC],
                                                          1,
                                                          &m_crcLaunchCfgs[0],
                                                          &m_crcLaunchCfgs[1],
                                                          m_cuStream);
    }

    cuphyStatus_t setupRssiStatus;
    if(m_chEstSettings.enableRssiMeasurement)
    {
        setupRssiStatus = cuphySetupPuschRxRssi(
            m_rssiHndl,
            m_drvdUeGrpPrmsCpu,
            m_drvdUeGrpPrmsGpu,
            m_cuphyPuschCellGrpDynPrm.nUeGrps,
            m_nMaxPrb,
            enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
            static_cast<void*>(dynCpuDescrStartAddrs[PUSCH_RSSI]),
            static_cast<void*>(dynGpuDescrStartAddrs[PUSCH_RSSI]),
            &m_rssiLaunchCfgs,
            m_cuStream);
    }

    cuphyStatus_t setupRsrpStatus;
    if(m_chEstSettings.enableSinrMeasurement)
    {
        setupRsrpStatus = cuphySetupPuschRxRsrp(
            m_rssiHndl,
            m_drvdUeGrpPrmsCpu,
            m_drvdUeGrpPrmsGpu,
            m_cuphyPuschCellGrpDynPrm.nUeGrps,
            m_nMaxPrb,
            enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
            static_cast<void*>(dynCpuDescrStartAddrs[PUSCH_RSRP]),
            static_cast<void*>(dynGpuDescrStartAddrs[PUSCH_RSRP]),
            &m_rsrpLaunchCfgs,
            m_cuStream);
    }

    //---------------------------------------------------------------------------------
    // Error checking
    if(CUPHY_STATUS_SUCCESS != chEstSetupStatus)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphySetupPuschRxChEst()");
        pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_CHEST_SETUP_ERROR;
        pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
        pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
        return CUPHY_STATUS_INTERNAL_ERROR;
    }

    if(CUPHY_STATUS_SUCCESS != cfoTaEstSetupStatus)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphySetupPuschRxCfoTaEst()");
        pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_CFO_TA_SETUP_ERROR;
        pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
        pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
        return CUPHY_STATUS_INTERNAL_ERROR;
    }

    if(CUPHY_STATUS_SUCCESS != coefComputeSetupStatus)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphySetupPuschRxChEqCoefCompute()");
        pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_CHEQ_COEF_COMP_SETUP_ERROR;
        pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
        pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    
    if(CUPHY_STATUS_SUCCESS != setupSoftDemapEarlyHarqStatus)
    {   
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphySetupPuschRxChEqSoftDemap()");
        pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_SOFT_DEMAP_EARLY_HARQ_SETUP_ERROR;
        pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
        pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
        return CUPHY_STATUS_INTERNAL_ERROR;
    }

    if(CUPHY_STATUS_SUCCESS != setupSoftDemapStatus)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphySetupPuschRxChEqSoftDemap()");
        pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_SOFT_DEMAP_SETUP_ERROR;
        pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
        pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
        return CUPHY_STATUS_INTERNAL_ERROR;
    }

    if(CUPHY_STATUS_SUCCESS != setupSoftDemapBluesteinWorkspaceStatus)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphySetupPuschRxChEqSoftDemapBluesteinWorkspace()");
        pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_BS_WORKSPACE_SETUP_ERROR;
        pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
        pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    
    if(CUPHY_STATUS_SUCCESS != setupSoftDemapIdftEarlyHarqStatus)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphySetupPuschRxChEqSoftDemapIdft()");
        pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_IDFT_EARLY_HARQ_SETUP_ERROR;
        pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
        pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
        return CUPHY_STATUS_INTERNAL_ERROR;
    }

    if(CUPHY_STATUS_SUCCESS != setupSoftDemapAfterDftEarlyHarqStatus)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphySetupPuschRxChEqSoftDemapAfterDft()");
        pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_SOFT_DEMAP_AFTER_DFT_EARLY_HARQ_SETUP_ERROR;
        pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
        pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
        return CUPHY_STATUS_INTERNAL_ERROR;
    }

    if(CUPHY_STATUS_SUCCESS != setupSoftDemapIdftStatus)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphySetupPuschRxChEqSoftDemapIdft()");
        pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_IDFT_SETUP_ERROR;
        pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
        pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
        return CUPHY_STATUS_INTERNAL_ERROR;
    }

    if(CUPHY_STATUS_SUCCESS != setupSoftDemapAfterDftStatus)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphySetupPuschRxChEqSoftDemapAfterDft()");
        pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_SOFT_DEMAP_AFTER_DFT_SETUP_ERROR;
        pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
        pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
        return CUPHY_STATUS_INTERNAL_ERROR;
    }

    if(m_nSchUes > 0)
    {
        if(CUPHY_STATUS_SUCCESS != setupRateMatchStatus)
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphySetupPuschRxRateMatch()");
            pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_RATE_MATCH_SETUP_ERROR;
            pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
            pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
            return CUPHY_STATUS_INTERNAL_ERROR;
        }

        if(CUPHY_STATUS_SUCCESS != setupCrcDecodeStatus)
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphySetupPuschRxCrcDecode()");
            pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_CRC_DECODE_SETUP_ERROR;
            pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
            pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
    }

    if(m_chEstSettings.enableRssiMeasurement)
    {
        if(CUPHY_STATUS_SUCCESS != setupRssiStatus)
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphySetupPuschRxRssi()");
            pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_RSSI_SETUP_ERROR;
            pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
            pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
    }

    if(m_chEstSettings.enableSinrMeasurement || (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE)|| (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC))
    {
        if(CUPHY_STATUS_SUCCESS != noiseIntfEstEarlyHarqSetupStatus)
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphySetupPuschRxNoiseIntfEst()");
            pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_NOISE_INTF_EST_EARLY_HARQ_SETUP_ERROR;
            pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
            pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
        
        if(CUPHY_STATUS_SUCCESS != noiseIntfEstSetupStatus)
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphySetupPuschRxNoiseIntfEst()");
            pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_NOISE_INTF_EST_SETUP_ERROR;
            pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
            pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
    }

    if(m_chEstSettings.enableSinrMeasurement)
    {
        if(CUPHY_STATUS_SUCCESS != setupRsrpStatus)
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "cuphySetupPuschRxRsrp()");
            pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_RSRP_SETUP_ERROR;
            pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
            pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
    }
    
    return CUPHY_STATUS_SUCCESS;
}

//===============================================================================================================================


void PuschRx::addEarlyHarqNodes(std::vector<CUgraphNode>* currNodeDeps, std::vector<CUgraphNode>* nextNodeDeps)
{
    // Use m_ehqPreWaitNode as the root for early HARQ path
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_ehqPreWaitNode, m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode2paramsDriver));

    //==========================================================================================================/
    // add m_ehqChEstNodes, parent node  : m_ehqPreWaitNode                                                     /
    //                      sibling nodes: other m_ehqChEstNodes                                                /
    //==========================================================================================================/
    currNodeDeps[0].emplace_back(m_ehqPreWaitNode);

    for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_CH_EST_N_MAX_HET_CFGS; ++hetCfgIdx)
    {
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_ehqChEstNodes[hetCfgIdx], m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode2paramsDriver));
        nextNodeDeps[0].emplace_back(m_ehqChEstNodes[hetCfgIdx]);
    }

    //==========================================================================================================/
    // add m_ehqNoiseIntfEstNodes, parent node(s) : m_ehqChEstNodes                                             /
    //                             sibling node(s): other m_ehqNoiseIntfEstNodes                                /
    //==========================================================================================================/
    // At minimum equalizer coefficient compute depends on channel estimation. Additionally, if noise-interference estimation
    // is enabled then it depends on that as well
    nextNodeDeps[1] = nextNodeDeps[0];

    if(m_chEstSettings.enableSinrMeasurement  || (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE)|| (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC))
    {
        // Noise-interference estimation depends on all channel estimations
        for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_ehqNoiseIntfEstNodes[hetCfgIdx], m_graph, nextNodeDeps[0].data(), nextNodeDeps[0].size(), &m_emptyNode1paramDriver));
            nextNodeDeps[1].emplace_back(m_ehqNoiseIntfEstNodes[hetCfgIdx]);
        }
    }

    //==========================================================================================================/
    // add m_ehqChEqCoefCompNodes, parent node(s) : m_ehqChEstNodes (and conditionally m_ehqNoiseIntfEstNodes)  /
    //                             sibling node(s): other m_ehqChEqCoefCompNodes                                /
    //==========================================================================================================/
    // Since equalizer coefficient compute depends on channel, and noise-interference when enabled, soft demap
    // only depend on equalizer coefficient compute
    nextNodeDeps[0].clear();

    for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
    {
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_ehqChEqCoefCompNodes[hetCfgIdx], m_graph, nextNodeDeps[1].data(), nextNodeDeps[1].size(), &m_emptyNode2paramsDriver));
        nextNodeDeps[0].emplace_back(m_ehqChEqCoefCompNodes[hetCfgIdx]);
    }

    //==========================================================================================================/
    // add m_ehqChEqSoftDemapNodes, parent node(s) : m_ehqChEqCoefCompNodes                                     /
    //                              sibling node(s): other m_ehqChEqSoftDemapNodes                              /
    //==========================================================================================================/
    currNodeDeps[0] = nextNodeDeps[0];
    nextNodeDeps[0].clear();

    for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
    {
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_ehqChEqSoftDemapNodes[hetCfgIdx], m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode2paramsDriver));
        nextNodeDeps[0].emplace_back(m_ehqChEqSoftDemapNodes[hetCfgIdx]);
    }

    if(m_chEstSettings.enableDftSOfdm == 1)
    {
        //==========================================================================================================/
        // add m_ehqChEqSoftDemapBluesteinWorkspaceNodes, parent node(s) : m_ehqChEqSoftDemapNodes                  /
        //                                         sibling node(s): other m_ehqChEqSoftDemapBluesteinWorkspaceNodes /
        //==========================================================================================================/
        currNodeDeps[0] = nextNodeDeps[0];
        nextNodeDeps[0].clear();
        for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_ehqChEqSoftDemapBluesteinWorkspaceNodes[hetCfgIdx], m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode2paramsDriver));
            nextNodeDeps[0].emplace_back(m_ehqChEqSoftDemapBluesteinWorkspaceNodes[hetCfgIdx]);
        }

        //==========================================================================================================/
        // add m_ehqChEqSoftDemapIdftNodes, parent node(s) : m_ehqChEqSoftDemapIdftNodes                            /
        //                                   sibling node(s): other m_ehqChEqSoftDemapIdftNodes                    /
        //==========================================================================================================/
        currNodeDeps[0] = nextNodeDeps[0];
        nextNodeDeps[0].clear();
        for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_ehqChEqSoftDemapIdftNodes[hetCfgIdx], m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode2paramsDriver));
            nextNodeDeps[0].emplace_back(m_ehqChEqSoftDemapIdftNodes[hetCfgIdx]);
        }

        //==========================================================================================================/
        // add m_ehqChEqSoftDemapAfterDftNodes, parent node(s) : m_ehqChEqSoftDemapIDftNodes                        /
        //                                      sibling node(s): other m_ehqChEqSoftDemapAfterDftNodes              /
        //==========================================================================================================/
        currNodeDeps[0] = nextNodeDeps[0];
        nextNodeDeps[0].clear();
        for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_ehqChEqSoftDemapAfterDftNodes[hetCfgIdx], m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode2paramsDriver));
            nextNodeDeps[0].emplace_back(m_ehqChEqSoftDemapAfterDftNodes[hetCfgIdx]);
        }

    }

    // SCH backend and UCI backend kernels can start after soft demap
    currNodeDeps[0] = nextNodeDeps[0];
    nextNodeDeps[0].clear();
    nextNodeDeps[1].clear();
    currNodeDeps[1].clear();

    //==========================================================================================================/
    // add m_ehqCompCwTreeTypesNode, parent node(s) : m_ehqChEqSoftDemapAfterDftNodes                           /
    //                               sibling node(s): m_ehqUciSegLLRs0Node                                      /
    //==========================================================================================================/
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_ehqCompCwTreeTypesNode, m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode1paramDriver));
    currNodeDeps[1].emplace_back(m_ehqCompCwTreeTypesNode);

    //==========================================================================================================/
    // add m_ehqUciSegLLRs0Node, parent node(s) : m_ehqChEqSoftDemapNodes                                       /
    //                           sibling node(s): m_ehqCompCwTreeTypesNode                                      /
    //==========================================================================================================/
    // Most of these kernels are run optionally, and calling the setup function here during graph creation can cause issues
    // if there's nothing to set up. Instead, we just use the null kernel here and update the nodes during updateGraph().
    //if(m_nUciUes > 0)
    {
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_ehqUciSegLLRs0Node, m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode1paramDriver));
        nextNodeDeps[0].emplace_back(m_ehqUciSegLLRs0Node);

        // We have a dependency on the CSI-P1 control kernel for the next 3 kernels: simplex, RM and polar
        // Simplex, RM and polar run in parallel
        currNodeDeps[0] = nextNodeDeps[0];
    }

    //==========================================================================================================/
    // add m_ehqSimplexDecoderNode, parent node(s) : m_ehqUciSegLLRs0Node                                           /
    //                          sibling node(s): m_ehqRmDecoderNode, m_ehqCompCwTreeTypesNode                       /
    //==========================================================================================================/
    //if(m_nSpxCws_early > 0)
    {
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_ehqSimplexDecoderNode, m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode1paramDriver));
        nextNodeDeps[1].emplace_back(m_ehqSimplexDecoderNode);
    }

    //==========================================================================================================/
    // add m_ehqRmDecoderNode, parent node(s) : m_ehqUciSegLLRs0Node                                                /
    //                     sibling node(s): m_ehqSimplexDecoderNode, m_ehqCompCwTreeTypesNode                       /
    //==========================================================================================================/
    //if(m_nRmCws > 0)
    {
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_ehqRmDecoderNode, m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode1paramDriver));
        nextNodeDeps[1].emplace_back(m_ehqRmDecoderNode);
    }

    //if(m_nPolUciSegs_early > 0)
    {
        //==========================================================================================================/
        // add m_ehqPolSegDeRmDeItlNode, parent node(s) : m_ehqUciSegLLRs0Node, m_ehqCompCwTreeTypesNode            /
        //                               sibling node(s): none                                                      /
        //==========================================================================================================/
        currNodeDeps[1].insert(currNodeDeps[1].end(), currNodeDeps[0].begin(), currNodeDeps[0].end());
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_ehqPolSegDeRmDeItlNode, m_graph, currNodeDeps[1].data(), currNodeDeps[1].size(), &m_emptyNode1paramDriver));
        currNodeDeps[1].clear();
        currNodeDeps[1].emplace_back(m_ehqPolSegDeRmDeItlNode);

        //==========================================================================================================/
        // add m_ehqPolarDecoderNode, parent node(s) : m_ehqPolSegDeRmDeItlNode                                     /
        //                            sibling node(s): none                                                         /
        //==========================================================================================================/
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_ehqPolarDecoderNode, m_graph, currNodeDeps[1].data(), currNodeDeps[1].size(), &m_emptyNode1paramDriver));
        nextNodeDeps[1].emplace_back(m_ehqPolarDecoderNode);
    }

    currNodeDeps[0] = nextNodeDeps[1];
    nextNodeDeps[0].clear();
    nextNodeDeps[1].clear();

    //==========================================================================================================/
    // add m_ehqMemcpyNodes, parent nodes : m_ehqSimplexDecoderNode, m_ehqRmDecoderNode and m_ehqPolarDecoderNode       /
    //                       sibling nodes: m_ehqPostWaitNode, other memcpy nodes                               /
    //==========================================================================================================/
    // Initialize memcopy nodes
    // note all mem-copy nodes are created based on a place-holder for mem-copy parameters. In graph update, these parameters will be updated
    // with correct source/destination pointers and copy-size. Also, cuGraphAddMemcpyNode needs context, hence we use the current context
    CU_CHECK_EXCEPTION(cuCtxGetCurrent(&m_puschCtx));

    //if(m_outputPrms.cpuCopyOn)
    {
        CU_CHECK_EXCEPTION(cuGraphAddMemcpyNode(&m_ehqMemcpyUciPayloadNode, m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_memcpyTemplateD2H.params, m_puschCtx));
        nextNodeDeps[0].emplace_back(m_ehqMemcpyUciPayloadNode);
        CU_CHECK_EXCEPTION(cuGraphAddMemcpyNode(&m_ehqMemcpyUciCrcNode, m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_memcpyTemplateD2H.params, m_puschCtx));
        nextNodeDeps[0].emplace_back(m_ehqMemcpyUciCrcNode);
        CU_CHECK_EXCEPTION(cuGraphAddMemcpyNode(&m_ehqMemcpyDetectionStatNode, m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_memcpyTemplateD2H.params, m_puschCtx));
        nextNodeDeps[0].emplace_back(m_ehqMemcpyDetectionStatNode);
    }

    //==========================================================================================================/
    // add m_ehqEventReadyNode parent nodes : Memcpy node                                                       /
    //                         sibling nodes: none                                                              /
    //==========================================================================================================/
    // add event node to notify cuPHY-CP of the completion of early-HARQ processing
    CU_CHECK_EXCEPTION(cuGraphAddEventRecordNode(&m_ehqEventReadyNode, m_graph, nextNodeDeps[0].data(), nextNodeDeps[0].size(), m_cuphyPuschStatPrms.earlyHarqReadyEvent));

    //==========================================================================================================/
    // add m_ehqPostWaitNode, parent nodes : m_ehqSimplexDecoderNode, m_ehqRmDecoderNode and m_ehqPolarDecoderNode      /
    //                        sibling nodes: Memcpy nodes                                                       /
    //==========================================================================================================/
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_ehqPostWaitNode, m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode2paramsDriver));

    currNodeDeps[0].clear();
    currNodeDeps[1].clear();
    nextNodeDeps[0].clear();
    nextNodeDeps[1].clear();
    currNodeDeps[0].emplace_back(m_ehqPostWaitNode);

}


void PuschRx::createGraph()
{
#if CUDART_VERSION < 11000
        throw cuphy::cuda_driver_exception("Graph mode requires CUDA driver kernel node params which requires CUDA 11.0 or higher");
#endif

    CUDA_CHECK_EXCEPTION(cudaGraphCreate(&m_graph, 0));

    std::vector<CUgraphNode> currNodeDeps[2], nextNodeDeps[2];

    void* arg;
    void* kernelParams[2] = {&arg, &arg};

    // Initialize empty nodes with 0, 1, and 2 input pointer args
    CUPHY_CHECK(cuphySetEmptyKernelNodeParams(&m_emptyNode0paramDriver));
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode1paramDriver, 1, &(kernelParams[0])));
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode2paramsDriver, 2, &(kernelParams[0])));

    // Use empty node as a root
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_emptyRootNode, m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode0paramDriver));

    //==========================================================================================================/
    // add early HARQ nodes                                                                                     /
    //==========================================================================================================/
    currNodeDeps[0].emplace_back(m_emptyRootNode);
    // using PUSCH static api parameter to decide adding early-HARQ nodes in the constructor or not
    if(m_cuphyPuschStatPrms.enableEarlyHarq)
    {
        addEarlyHarqNodes(currNodeDeps, nextNodeDeps);
    }

    //==========================================================================================================/
    // add m_chEstNodes, parent node  : m_ehqPostWaitNode                                                       /
    //                   sibling nodes: other m_chEstNodes                                                      /
    //==========================================================================================================/

    for(int32_t chEstTimeInstIdx = 0; chEstTimeInstIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEstTimeInstIdx)
    {
        for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_CH_EST_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_chEstNodes[chEstTimeInstIdx][hetCfgIdx], m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode2paramsDriver));
            nextNodeDeps[0].emplace_back(m_chEstNodes[chEstTimeInstIdx][hetCfgIdx]);
        }
    }

    //==========================================================================================================/
    // add m_noiseIntfEstNodes, parent node(s) : m_chEstNodes                                                   /
    //                          sibling node(s): m_cfoTaEstNodes, other m_noiseIntfEstNodes                     /
    //==========================================================================================================/
    // At minimum equalizer coefficient compute depends on channel estimation. Additionally, if noise-interference estimation and CFO estimation
    // are enabled then it depends on them as well
    nextNodeDeps[1] = nextNodeDeps[0];

    if(m_chEstSettings.enableSinrMeasurement || (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE)|| (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC))
    {
        // Noise-interference estimation depends on all channel estimations
        for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_noiseIntfEstNodes[hetCfgIdx], m_graph, nextNodeDeps[0].data(), nextNodeDeps[0].size(), &m_emptyNode1paramDriver));
            nextNodeDeps[1].emplace_back(m_noiseIntfEstNodes[hetCfgIdx]);
        }
    }

    //==========================================================================================================/
    // add m_cfoTaEstNodes, parent node(s) : m_chEstNodes                                                       /
    //                      sibling node(s): m_noiseIntfEstNodes, other m_cfoTaEstNodes                         /
    //==========================================================================================================/
    if(m_chEstSettings.enableCfoCorrection || m_chEstSettings.enableToEstimation)
    {
        // CFO calculation depends on all channel estimations
        for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_CFO_EST_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_cfoTaEstNodes[hetCfgIdx], m_graph, nextNodeDeps[0].data(), nextNodeDeps[0].size(), &m_emptyNode2paramsDriver));
            nextNodeDeps[1].emplace_back(m_cfoTaEstNodes[hetCfgIdx]);
        }
    }

    //=================================================================================================================/
    // add m_chEqCoefCompNodes, parent node(s) : m_chEstNodes (and conditionally m_noiseIntfEstNodes, m_cfoTaEstNodes) /
    //                          sibling node(s): other m_chEqCoefCompNodes                                             /
    //=================================================================================================================/
    // Since equalizer coefficient compute depends on channel, noise-interference and CFO estimation when enabled, soft demap need only depend on
    // equalizer coefficient compute
    nextNodeDeps[0].clear();
    int32_t bound = 1;
    if(m_chEstSettings.enablePuschTdi)
    {
        bound = CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ;
    }

    for(int32_t chEqTimeInstIdx = 0; chEqTimeInstIdx < bound; ++chEqTimeInstIdx)
    {
        for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_chEqCoefCompNodes[chEqTimeInstIdx][hetCfgIdx], m_graph, nextNodeDeps[1].data(), nextNodeDeps[1].size(), &m_emptyNode2paramsDriver));
            nextNodeDeps[0].emplace_back(m_chEqCoefCompNodes[chEqTimeInstIdx][hetCfgIdx]);
        }
    }

    //==========================================================================================================/
    // add m_chEqSoftDemapNodes, parent node(s) : m_chEqCoefCompNodes                                           /
    //                           sibling node(s): other m_chEqSoftDemapNodes                                    /
    //==========================================================================================================/
    currNodeDeps[0] = nextNodeDeps[0];
    nextNodeDeps[0].clear();

    for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
    {
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_chEqSoftDemapNodes[hetCfgIdx], m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode2paramsDriver));
        nextNodeDeps[0].emplace_back(m_chEqSoftDemapNodes[hetCfgIdx]);
    }

    if(m_chEstSettings.enableDftSOfdm==1)
    {
        //==========================================================================================================/
        // add m_chEqSoftDemapBluesteinWorkspaceNodes, parent node(s) : m_chEqSoftDemapNodes                        /
        //                                             sibling node(s): other m_chEqSoftDemapBluesteinWorkspaceNodes/
        //==========================================================================================================/
        currNodeDeps[0] = nextNodeDeps[0];
        nextNodeDeps[0].clear();
        for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_chEqSoftDemapBluesteinWorkspaceNodes[hetCfgIdx], m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode2paramsDriver));
            nextNodeDeps[0].emplace_back(m_chEqSoftDemapBluesteinWorkspaceNodes[hetCfgIdx]);
        }

        //==========================================================================================================/
        // add m_chEqSoftDemapIdftNodes, parent node(s) : m_chEqSoftDemapIdftNodes                                  /
        //                               sibling node(s): other m_chEqSoftDemapIdftNodes                           /
        //==========================================================================================================/
        currNodeDeps[0] = nextNodeDeps[0];
        nextNodeDeps[0].clear();
        for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_chEqSoftDemapIdftNodes[hetCfgIdx], m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode2paramsDriver));
            nextNodeDeps[0].emplace_back(m_chEqSoftDemapIdftNodes[hetCfgIdx]);
        }

        //==========================================================================================================/
        // add m_chEqSoftDemapAfterDftNodes, parent node(s) : m_chEqSoftDemapDft2Nodes                                   /
        //                                   sibling node(s): other m_chEqSoftDemapAfterDftNodes                    /
        //==========================================================================================================/
        currNodeDeps[0] = nextNodeDeps[0];
        nextNodeDeps[0].clear();
        for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_chEqSoftDemapAfterDftNodes[hetCfgIdx], m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode2paramsDriver));
            nextNodeDeps[0].emplace_back(m_chEqSoftDemapAfterDftNodes[hetCfgIdx]);
        }

    }

    // SCH backend and UCI backend kernels can start after soft demap
    currNodeDeps[0] = nextNodeDeps[0];
    nextNodeDeps[0].clear();
    nextNodeDeps[1].clear();
    currNodeDeps[1].clear();

    //==========================================================================================================/
    // add m_compCwTreeTypesNode, parent node(s) : m_chEqSoftDemapAfterDftNodes                                 /
    //                            sibling node(s): m_rssiNodes, m_uciSegLLRs0Node, m_rsrpNodes                  /
    //==========================================================================================================/
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_compCwTreeTypesNode, m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode1paramDriver));
    currNodeDeps[1].emplace_back(m_compCwTreeTypesNode);

    //==========================================================================================================/
    // add m_rsrpNodes, parent node(s) : m_chEqSoftDemapNodes                                                   /
    //                  sibling node(s): m_rssiNodes, m_uciSegLLRs0Node, m_compCwTreeTypesNode                  /
    //==========================================================================================================/
    // SINR and RSSI estimation can go in parallel with SCH backend and UCI backend kernels
    if(m_chEstSettings.enableSinrMeasurement)
    {
        for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_RSRP_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_rsrpNodes[hetCfgIdx], m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode1paramDriver));
        }
    }

    //==========================================================================================================/
    // add m_rssiNodes, parent node(s) : m_chEqSoftDemapNodes                                                   /
    //                  sibling node(s): m_rsrpNodes, m_uciSegLLRs0Node, m_compCwTreeTypesNode                  /
    //==========================================================================================================/
    if(m_chEstSettings.enableRssiMeasurement)
    {
        for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_RSSI_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_rssiNodes[hetCfgIdx], m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode1paramDriver));
        }
    }

    //==========================================================================================================/
    // add m_uciSegLLRs0Node, parent node(s) : m_chEqSoftDemapNodes                                             /
    //                        sibling node(s): m_rsrpNodes, m_rssiNodes, m_compCwTreeTypesNode                  /
    //==========================================================================================================/
    // Most of these kernels are run optionally, and calling the setup function here during graph creation can cause issues
    // if there's nothing to set up. Instead, we just use the null kernel here and update the nodes during updateGraph().
    //if(m_nUciUes > 0)
    {
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_uciSegLLRs0Node, m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode1paramDriver));
        nextNodeDeps[0].emplace_back(m_uciSegLLRs0Node);

        // We have a dependency on the CSI-P1 control kernel for the next 3 kernels: simplex, RM and polar
        // Simplex, RM and polar run in parallel
        currNodeDeps[0] = nextNodeDeps[0];
    }

    //==========================================================================================================/
    // add m_simplexDecoderNode, parent node(s) : m_uciSegLLRs0Node                                                 /
    //                       sibling node(s): m_rmDecoderNode, m_compCwTreeTypesNode                                /
    //==========================================================================================================/
    //if(m_nSpxCws > 0)
    {
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_simplexDecoderNode, m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode1paramDriver));
        nextNodeDeps[1].emplace_back(m_simplexDecoderNode);
    }

    //==========================================================================================================/
    // add m_rmDecoderNode, parent node(s) : m_uciSegLLRs0Node                                                      /
    //                  sibling node(s): m_simplexDecoderNode, m_compCwTreeTypesNode                                /
    //==========================================================================================================/
    //if(m_nRmCws > 0)
    {
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_rmDecoderNode, m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode1paramDriver));
        nextNodeDeps[1].emplace_back(m_rmDecoderNode);
    }

    //if(m_nPolUciSegs > 0)
    {
        //==========================================================================================================/
        // add m_polSegDeRmDeItlNode, parent node(s) : m_uciSegLLRs0Node, m_compCwTreeTypesNode                     /
        //                            sibling node(s): none                                                         /
        //==========================================================================================================/
        currNodeDeps[1].insert(currNodeDeps[1].end(), currNodeDeps[0].begin(), currNodeDeps[0].end());
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_polSegDeRmDeItlNode, m_graph, currNodeDeps[1].data(), currNodeDeps[1].size(), &m_emptyNode1paramDriver));
        currNodeDeps[1].clear();
        currNodeDeps[1].emplace_back(m_polSegDeRmDeItlNode);

        //==========================================================================================================/
        // add m_polarDecoderNode, parent node(s) : m_polSegDeRmDeItlNode                                           /
        //                         sibling node(s): none                                                            /
        //==========================================================================================================/
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_polarDecoderNode, m_graph, currNodeDeps[1].data(), currNodeDeps[1].size(), &m_emptyNode1paramDriver));
        nextNodeDeps[1].emplace_back(m_polarDecoderNode);
    }

    //===================================== CSI-P2 =======================================

    //if(m_nCsi2Ues > 0)
    {
        //==========================================================================================================/
        // add m_uciOnPuschCsi2CtrlNode, parent node(s) : m_simplexDecoderNode, m_rmDecoderNode, m_polarDecoderNode         /
        //                               sibling node(s): none                                                      /
        //==========================================================================================================/
        // CSI-P2 control kernel depends on CSI-P1 kernels (simplex, RM and polar) completion
        currNodeDeps[0] = nextNodeDeps[1];
        nextNodeDeps[0].clear();
        nextNodeDeps[1].clear();
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_uciOnPuschCsi2CtrlNode, m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode1paramDriver));
        nextNodeDeps[0].emplace_back(m_uciOnPuschCsi2CtrlNode);

        //==========================================================================================================/
        // add m_uciOnPuschSegLLRs2Node, parent node(s) : m_uciOnPuschCsi2CtrlNode                                  /
        //                               sibling node(s): none                                                      /
        //==========================================================================================================/
        // CSI-P2 control kernel (Csi2Ctrl) calculates SCH rate-matched bits which is used by SegLLRs2 for demux
        currNodeDeps[0] = nextNodeDeps[0];
        nextNodeDeps[0].clear();
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_uciOnPuschSegLLRs2Node, m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode1paramDriver));
        nextNodeDeps[0].emplace_back(m_uciOnPuschSegLLRs2Node);

        //==========================================================================================================/
        // add m_rmDecCsi2Node, parent node(s) : m_uciOnPuschSegLLRs2Node                                           /
        //                      sibling node(s): m_simplexDecoderCsi2Node, m_compCwTreeTypesNode, m_rateMatchNode   /
        //==========================================================================================================/
        currNodeDeps[0] = nextNodeDeps[0];
        nextNodeDeps[0].clear();
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_rmDecCsi2Node, m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode1paramDriver));
        nextNodeDeps[0].emplace_back(m_rmDecCsi2Node);

        //==========================================================================================================/
        // add m_simplexDecoderCsi2Node, parent node(s) : m_uciOnPuschSegLLRs2Node                                  /
        //                               sibling node(s): m_rmDecCsi2Node, m_compCwTreeTypesNode, m_rateMatchNode   /
        //==========================================================================================================/
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_simplexDecoderCsi2Node, m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode1paramDriver));
        nextNodeDeps[0].emplace_back(m_simplexDecoderCsi2Node);
    }

    //if(m_nSchUes > 0) {
    {
        //==========================================================================================================/
        // add m_rateMatchNode, parent node(s) : m_uciOnPuschSegLLRs2Node                                           /
        //                      sibling node(s): m_rmDecCsi2Node, m_simplexDecoderCsi2Node                          /
        //==========================================================================================================/
        // SCH backend (derate-match, LDPC, CRC) of CSI-P2 UEs needs to wait until CSI-P2/SCH demux (SegLLRs2) occurs.
        // @todo: SCH backend of CSI-P1 UEs can run right after PUSCH soft-demap completion (but this is not supported by pipeline yet)
        nextNodeDeps[0].clear();
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_rateMatchNode, m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode1paramDriver));
        nextNodeDeps[0].emplace_back(m_rateMatchNode);

        //==========================================================================================================/
        // add m_ldpcDecoderNodes, parent node(s) : m_rateMatchNode                                                      /
        //                    sibling node(s): other m_ldpcDecNodes                                                 /
        //==========================================================================================================/
        currNodeDeps[0] = nextNodeDeps[0];
        nextNodeDeps[0].clear();

        for(int i = 0; i < N_MAX_LDPC_HET_CFGS; ++i)
        {
            CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_ldpcDecoderNodes[i], m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode2paramsDriver));
            nextNodeDeps[0].emplace_back(m_ldpcDecoderNodes[i]);
        }

        //==========================================================================================================/
        // add m_crcNodes[0], parent node(s) : m_ldpcDecoderNodes                                                        /
        //                    sibling node(s): none                                                                 /
        //==========================================================================================================/
        currNodeDeps[0] = nextNodeDeps[0];
        nextNodeDeps[0].clear();
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_crcNodes[0], m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode1paramDriver));
        nextNodeDeps[0].emplace_back(m_crcNodes[0]);

        //==========================================================================================================/
        // add m_crcNodes[1], parent node(s) : m_crcNodes[0]                                                        /
        //                    sibling node(s): none                                                                 /
        //==========================================================================================================/
        currNodeDeps[0] = nextNodeDeps[0];
        //nextNodeDeps[0].clear();
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_crcNodes[1], m_graph, currNodeDeps[0].data(), currNodeDeps[0].size(), &m_emptyNode1paramDriver));
        //nextNodeDeps[0].emplace_back(m_crcNodes[1]);
    }
}

void PuschRx::disableAllEarlyHarqNodes()
{
    if(m_ehqAllNodesDisabled==1 || m_cuphyPuschStatPrms.enableEarlyHarq==0) return;
    if(m_ehqPreWaitNodeEnabled != 0)
    {
        m_ehqPreWaitNodeEnabled = 0;
#if CUDART_VERSION >= 11060
        CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqPreWaitNode, 0));
#else
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqPreWaitNode, &m_emptyNode1paramDriver));
#endif
    }

    for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_CH_EST_N_MAX_HET_CFGS; ++hetCfgIdx)
    {
        if(m_ehqChEstNodesEnabled[hetCfgIdx] != 0)
        {
            m_ehqChEstNodesEnabled[hetCfgIdx] = 0;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqChEstNodes[hetCfgIdx], 0));
#else
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_chEstNodes[chEstTimeInstIdx][hetCfgIdx], &(m_emptyNode2paramsDriver)));
#endif
        }
    }

    if(m_chEstSettings.enableSinrMeasurement  || (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE)|| (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC))
    {
        for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            if(m_ehqNoiseIntfEstNodesEnabled[hetCfgIdx] != 0)
            {
                m_ehqNoiseIntfEstNodesEnabled[hetCfgIdx] = 0;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqNoiseIntfEstNodes[hetCfgIdx], 0));
#else
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqNoiseIntfEstNodes[hetCfgIdx], &(m_emptyNode1paramDriver)));
#endif
            }
        }
    }

    for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
    {
        if(m_ehqChEqCoefCompNodesEnabled[hetCfgIdx] != 0)
        {
            m_ehqChEqCoefCompNodesEnabled[hetCfgIdx] = 0;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqChEqCoefCompNodes[hetCfgIdx], 0));
#else
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqChEqCoefCompNodes[hetCfgIdx], &(m_emptyNode2paramsDriver)));
#endif
        }
    }

    for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
    {
        if(m_ehqChEqSoftDemapNodesEnabled[hetCfgIdx] != 0)
        {
            m_ehqChEqSoftDemapNodesEnabled[hetCfgIdx] = 0;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqChEqSoftDemapNodes[hetCfgIdx], 0));
#else
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqChEqSoftDemapNodes[hetCfgIdx], &(m_emptyNode2paramsDriver)));
#endif
        }
    }

    if(m_chEstSettings.enableDftSOfdm==1)
    {
        for(int hetCfgIdx = 0; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            if(m_ehqChEqSoftDemapBluesteinWorkspaceNodesEnabled[hetCfgIdx] != 0)
            {
                m_ehqChEqSoftDemapBluesteinWorkspaceNodesEnabled[hetCfgIdx] = 0;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqChEqSoftDemapBluesteinWorkspaceNodes[hetCfgIdx], 0));
#else
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqChEqSoftDemapBluesteinWorkspaceNodes[hetCfgIdx], &(m_emptyNode2paramsDriver)));
#endif
            }

            if(m_ehqChEqSoftDemapIdftNodesEnabled[hetCfgIdx] != 0)
            {
                m_ehqChEqSoftDemapIdftNodesEnabled[hetCfgIdx] = 0;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqChEqSoftDemapIdftNodes[hetCfgIdx], 0));
#else
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqChEqSoftDemapIdftNodes[hetCfgIdx], &(m_emptyNode2paramsDriver)));
#endif
            }

            if(m_ehqChEqSoftDemapAfterDftNodesEnabled[hetCfgIdx] != 0)
            {
                m_ehqChEqSoftDemapAfterDftNodesEnabled[hetCfgIdx] = 0;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqChEqSoftDemapAfterDftNodes[hetCfgIdx], 0));
#else
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqChEqSoftDemapAfterDftNodes[hetCfgIdx], &(m_emptyNode2paramsDriver)));
#endif
            }
        }
    }

    if(m_ehqUciSegLLRs0NodeEnabled != 0)
    {
        m_ehqUciSegLLRs0NodeEnabled = 0;
#if CUDART_VERSION >= 11060
        CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqUciSegLLRs0Node, 0));
#else
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqUciSegLLRs0Node, &m_emptyNode1paramDriver));
#endif
    }

    if(m_ehqSimplexDecoderNodeEnabled != 0)
    {
        m_ehqSimplexDecoderNodeEnabled = 0;
#if CUDART_VERSION >= 11060
        CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqSimplexDecoderNode, 0));
#else
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqSimplexDecoderNode, &m_emptyNode1paramDriver));
#endif
    }

    if(m_ehqRmDecoderNodeEnabled != 0)
    {
        m_ehqRmDecoderNodeEnabled = 0;
#if CUDART_VERSION >= 11060
        CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqRmDecoderNode, 0));
#else
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqRmDecoderNode, &m_emptyNode1paramDriver));
#endif
    }

    if(m_ehqPolarNodeEnabled != 0)
    {
        m_ehqPolarNodeEnabled = 0;
#if CUDART_VERSION >= 11060
        CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqCompCwTreeTypesNode, 0));
        CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqPolSegDeRmDeItlNode, 0));
        CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqPolarDecoderNode, 0));
#else
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqCompCwTreeTypesNode, &m_emptyNode1paramDriver));
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqPolSegDeRmDeItlNode, &m_emptyNode1paramDriver));
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqPolarDecoderNode, &m_emptyNode1paramDriver));
#endif
    }

    if(m_ehqMemcpyUciPayloadNodeEnabled != 0)
    {
        m_ehqMemcpyUciPayloadNodeEnabled = 0;
#if CUDART_VERSION >= 11060
        CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqMemcpyUciPayloadNode, 0));
#else
        //FixMe if we need to support older CUDA versions
        //CU_CHECK_EXCEPTION(cuGraphExecMemcpyNodeSetParams(m_graphExec, m_ehqMemcpyUciPayloadNode, &m_emptyMemcpyD2Hparams));
        // empty memcopy node throws error, find another way to create empty memcpy node to support older versions of CUDA?
        // cuphySetD2HMemcpyNodeParams(&m_emptyMemcpyD2Hparams, nullptr, nullptr, 0); does not work
#endif
    }

    if(m_ehqMemcpyUciCrcNodeEnabled != 0)
    {
        m_ehqMemcpyUciCrcNodeEnabled = 0;
#if CUDART_VERSION >= 11060
        CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqMemcpyUciCrcNode, 0));
#else
        //FixMe if we need to support older CUDA versions
        //CU_CHECK_EXCEPTION(cuGraphExecMemcpyNodeSetParams(m_graphExec, m_ehqMemcpyUciCrcNode, &m_emptyMemcpyD2Hparams));
        // empty memcopy node throws error, find another way to create empty memcpy node to support older versions of CUDA?
        // cuphySetD2HMemcpyNodeParams(&m_emptyMemcpyD2Hparams, nullptr, nullptr, 0); does not work
#endif
    }

    if(m_ehqMemcpyDetectionStatNodeEnabled != 0)
    {
        m_ehqMemcpyDetectionStatNodeEnabled = 0;
#if CUDART_VERSION >= 11060
        CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqMemcpyDetectionStatNode, 0));
#else
        //FixMe if we need to support older CUDA versions
        //CU_CHECK_EXCEPTION(cuGraphExecMemcpyNodeSetParams(m_graphExec, m_ehqMemcpyDetectionStatNode, &m_emptyMemcpyD2Hparams));
        // empty memcopy node throws error, find another way to create empty memcpy node to support older versions of CUDA?
        // cuphySetD2HMemcpyNodeParams(&m_emptyMemcpyD2Hparams, nullptr, nullptr, 0); does not work
#endif
    }

#if 0
    // currently only kernel, memset and memcpy nodes are supported in cuGraphNodeSetEnabled
    if(m_ehqEventReadyNodeEnabled != 0)
    {
        m_ehqEventReadyNodeEnabled = 0;
#if CUDART_VERSION >= 11060
        CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqEventReadyNode, 0));
#else
        //FixMe fix it for older version of CUDA if needed
#endif
    }
#endif

    if(m_ehqPostWaitNodeEnabled != 0)
    {
        m_ehqPostWaitNodeEnabled = 0;
#if CUDART_VERSION >= 11060
        CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqPostWaitNode, 0));
#else
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqPostWaitNode, &m_emptyNode1paramDriver));
#endif
    }
    m_ehqAllNodesDisabled = 1;
}

void PuschRx::updateEarlyHarqNodes(bool enableNodes)
{
    if (enableNodes)
    {
        m_ehqAllNodesDisabled = 0;
        if(m_ehqPreWaitNodeEnabled != 1)
        {
            m_ehqPreWaitNodeEnabled = 1;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqPreWaitNode, 1));
#endif
        }
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqPreWaitNode, &(m_preEarlyHarqWaitCfgs.kernelNodeParamsDriver)));

        for(int hetCfgIdx = 0; hetCfgIdx < m_chEstLaunchCfgs[0].nCfgs; ++hetCfgIdx)
        {
            if(m_ehqChEstNodesEnabled[hetCfgIdx] != 1)
            {
                m_ehqChEstNodesEnabled[hetCfgIdx] = 1;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqChEstNodes[hetCfgIdx], 1));
#endif
            }
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqChEstNodes[hetCfgIdx], &(m_chEstLaunchCfgs[0].cfgs[hetCfgIdx].kernelNodeParamsDriver)));
        }
        for(int hetCfgIdx = m_chEstLaunchCfgs[0].nCfgs; hetCfgIdx < CUPHY_PUSCH_RX_CH_EST_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            if(m_ehqChEstNodesEnabled[hetCfgIdx] != 0)
            {
                m_ehqChEstNodesEnabled[hetCfgIdx] = 0;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqChEstNodes[hetCfgIdx], 0));
#else
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_chEstNodes[chEstTimeInstIdx][hetCfgIdx], &(m_emptyNode2paramsDriver)));
#endif
            }
        }


        for(int hetCfgIdx = 0; hetCfgIdx < m_noiseIntfEstLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH].nCfgs; ++hetCfgIdx)
        {
            if (m_ehqNoiseIntfEstNodesEnabled[hetCfgIdx] != 1)
            {
                m_ehqNoiseIntfEstNodesEnabled[hetCfgIdx] = 1;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqNoiseIntfEstNodes[hetCfgIdx], 1));
#endif
            }
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqNoiseIntfEstNodes[hetCfgIdx], &(m_noiseIntfEstLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH].cfgs[hetCfgIdx].kernelNodeParamsDriver)));
        }
        for(int hetCfgIdx = m_noiseIntfEstLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH].nCfgs; hetCfgIdx < CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            if (m_ehqNoiseIntfEstNodesEnabled[hetCfgIdx] != 0)
            {
                m_ehqNoiseIntfEstNodesEnabled[hetCfgIdx] = 0;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqNoiseIntfEstNodes[hetCfgIdx], 0));
#else
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqNoiseIntfEstNodes[hetCfgIdx], &(m_emptyNode1paramDriver)));
#endif
            }
        }


        for(int hetCfgIdx = 0; hetCfgIdx < m_chEqCoefCompLaunchCfgs[0].nCfgs; ++hetCfgIdx)
        {
            if (m_ehqChEqCoefCompNodesEnabled[hetCfgIdx] != 1)
            {
                m_ehqChEqCoefCompNodesEnabled[hetCfgIdx] = 1;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqChEqCoefCompNodes[hetCfgIdx], 1));
#endif
            }
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqChEqCoefCompNodes[hetCfgIdx], &(m_chEqCoefCompLaunchCfgs[0].cfgs[hetCfgIdx].kernelNodeParamsDriver)));
        }
        for(int hetCfgIdx = m_chEqCoefCompLaunchCfgs[0].nCfgs; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            if (m_ehqChEqCoefCompNodesEnabled[hetCfgIdx] != 0)
            {
                m_ehqChEqCoefCompNodesEnabled[hetCfgIdx] = 0;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqChEqCoefCompNodes[hetCfgIdx], 0));
#else
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqChEqCoefCompNodes[hetCfgIdx], &(m_emptyNode2paramsDriver)));
#endif
            }

        }


        for(int hetCfgIdx = 0; hetCfgIdx < m_chEqSoftDemapLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH].nCfgs; ++hetCfgIdx)
        {
            if (m_ehqChEqSoftDemapNodesEnabled[hetCfgIdx] != 1)
            {
                m_ehqChEqSoftDemapNodesEnabled[hetCfgIdx] = 1;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqChEqSoftDemapNodes[hetCfgIdx], 1));
#endif
            }
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqChEqSoftDemapNodes[hetCfgIdx], &(m_chEqSoftDemapLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH].cfgs[hetCfgIdx].kernelNodeParamsDriver)));
        }
        for(int hetCfgIdx = m_chEqSoftDemapLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH].nCfgs; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            if (m_ehqChEqSoftDemapNodesEnabled[hetCfgIdx] != 0)
            {
                m_ehqChEqSoftDemapNodesEnabled[hetCfgIdx] = 0;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqChEqSoftDemapNodes[hetCfgIdx], 0));
#else
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqChEqSoftDemapNodes[hetCfgIdx], &(m_emptyNode2paramsDriver)));
#endif
            }
        }


        if(m_chEstSettings.enableDftSOfdm==1)
        {
            for(int hetCfgIdx = 0; hetCfgIdx < m_chEqSoftDemapBluesteinWorkspaceLaunchCfgs.nCfgs; ++hetCfgIdx)
            {
                if (m_ehqChEqSoftDemapBluesteinWorkspaceNodesEnabled[hetCfgIdx] != 1)
                {
                    m_ehqChEqSoftDemapBluesteinWorkspaceNodesEnabled[hetCfgIdx] = 1;
#if CUDART_VERSION >= 11060
                    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqChEqSoftDemapBluesteinWorkspaceNodes[hetCfgIdx], 1));
#endif
                }
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqChEqSoftDemapBluesteinWorkspaceNodes[hetCfgIdx], &(m_chEqSoftDemapBluesteinWorkspaceLaunchCfgs.cfgs[hetCfgIdx].kernelNodeParamsDriver)));
            }
            for(int hetCfgIdx = m_chEqSoftDemapBluesteinWorkspaceLaunchCfgs.nCfgs; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
            {
                if (m_ehqChEqSoftDemapBluesteinWorkspaceNodesEnabled[hetCfgIdx] != 0)
                {
                    m_ehqChEqSoftDemapBluesteinWorkspaceNodesEnabled[hetCfgIdx] = 0;
#if CUDART_VERSION >= 11060
                    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqChEqSoftDemapBluesteinWorkspaceNodes[hetCfgIdx], 0));
#else
                    CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqChEqSoftDemapBluesteinWorkspaceNodes[hetCfgIdx], &(m_emptyNode2paramsDriver)));
#endif
                }
            }


            for(int hetCfgIdx = 0; hetCfgIdx < m_chEqSoftDemapIdftLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH].nCfgs; ++hetCfgIdx)
            {
                if (m_ehqChEqSoftDemapIdftNodesEnabled[hetCfgIdx] != 1)
                {
                    m_ehqChEqSoftDemapIdftNodesEnabled[hetCfgIdx] = 1;
#if CUDART_VERSION >= 11060
                    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqChEqSoftDemapIdftNodes[hetCfgIdx], 1));
#endif
                }
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqChEqSoftDemapIdftNodes[hetCfgIdx], &(m_chEqSoftDemapIdftLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH].cfgs[hetCfgIdx].kernelNodeParamsDriver)));
            }
            for(int hetCfgIdx = m_chEqSoftDemapIdftLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH].nCfgs; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
            {
                if (m_ehqChEqSoftDemapIdftNodesEnabled[hetCfgIdx] != 0)
                {
                    m_ehqChEqSoftDemapIdftNodesEnabled[hetCfgIdx] = 0;
#if CUDART_VERSION >= 11060
                    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqChEqSoftDemapIdftNodes[hetCfgIdx], 0));
#else
                    CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqChEqSoftDemapIdftNodes[hetCfgIdx], &(m_emptyNode2paramsDriver)));
#endif
                }
            }

            for(int hetCfgIdx = 0; hetCfgIdx < m_chEqSoftDemapAfterDftLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH].nCfgs; ++hetCfgIdx)
            {
                if (m_ehqChEqSoftDemapAfterDftNodesEnabled[hetCfgIdx] != 1)
                {
                    m_ehqChEqSoftDemapAfterDftNodesEnabled[hetCfgIdx] = 1;
#if CUDART_VERSION >= 11060
                    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqChEqSoftDemapAfterDftNodes[hetCfgIdx], 1));
#endif
                }
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqChEqSoftDemapAfterDftNodes[hetCfgIdx], &(m_chEqSoftDemapAfterDftLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH].cfgs[hetCfgIdx].kernelNodeParamsDriver)));
            }
            for(int hetCfgIdx = m_chEqSoftDemapAfterDftLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH].nCfgs; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
            {
                if (m_ehqChEqSoftDemapAfterDftNodesEnabled[hetCfgIdx] != 0)
                {
                    m_ehqChEqSoftDemapAfterDftNodesEnabled[hetCfgIdx] = 0;
#if CUDART_VERSION >= 11060
                    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqChEqSoftDemapAfterDftNodes[hetCfgIdx], 0));
#else
                    CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqChEqSoftDemapAfterDftNodes[hetCfgIdx], &(m_emptyNode2paramsDriver)));
#endif
                }
            }
        } //ToDo see if it's necessary to add "else" here and go through all those graph nodes and disable them when needed


        if(m_nUciUes > 0)
        {
            if(m_ehqUciSegLLRs0NodeEnabled != 1)
            {
                m_ehqUciSegLLRs0NodeEnabled = 1;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqUciSegLLRs0Node, 1));
#endif
            }
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqUciSegLLRs0Node, &(m_uciOnPuschSegLLRs0LaunchCfg.kernelNodeParamsDriver)));
        }
        else
        {
            if(m_ehqUciSegLLRs0NodeEnabled != 0)
            {
                m_ehqUciSegLLRs0NodeEnabled = 0;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqUciSegLLRs0Node, 0));
#else
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqUciSegLLRs0Node, &m_emptyNode1paramDriver));
#endif
            }
        }


        if(m_nSpxCws_early > 0)
        {
            if (m_ehqSimplexDecoderNodeEnabled != 1) {
                m_ehqSimplexDecoderNodeEnabled = 1;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqSimplexDecoderNode, 1));
#endif
            }
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqSimplexDecoderNode, &(m_simplexDecoderLaunchCfg_early.kernelNodeParamsDriver)));
        }
        else
        {
            if (m_ehqSimplexDecoderNodeEnabled != 0)
            {
                m_ehqSimplexDecoderNodeEnabled = 0;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqSimplexDecoderNode, 0));
#else
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqSimplexDecoderNode, &m_emptyNode1paramDriver));
#endif
            }
        }


        if(m_nRmCws_early > 0)
        {
            if (m_ehqRmDecoderNodeEnabled != 1) {
                m_ehqRmDecoderNodeEnabled = 1;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqRmDecoderNode, 1));
#endif
            }
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqRmDecoderNode, &(m_rmDecoderLaunchCfg_early.kernelNodeParamsDriver)));
        }
        else
        {
            if(m_ehqRmDecoderNodeEnabled != 0)
            {
                m_ehqRmDecoderNodeEnabled = 0;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqRmDecoderNode, 0));
#else
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqRmDecoderNode, &m_emptyNode1paramDriver));
#endif
            }
        }


        if(m_nPolUciSegs_early > 0)
        {
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqCompCwTreeTypesNode, &(m_compCwTreeTypesLaunchCfg_early.kernelNodeParamsDriver)));
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqPolSegDeRmDeItlNode, &(m_polSegDeRmDeItlLaunchCfg_early.kernelNodeParamsDriver)));
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqPolarDecoderNode, &(m_polarDecoderLaunchCfg_early.kernelNodeParamsDriver)));

            if (m_ehqPolarNodeEnabled != 1)
            {
                m_ehqPolarNodeEnabled = 1;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqCompCwTreeTypesNode, 1));
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqPolSegDeRmDeItlNode, 1));
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqPolarDecoderNode, 1));
#endif
            }
        }
        else
        {
            if (m_ehqPolarNodeEnabled != 0)
            {
                m_ehqPolarNodeEnabled = 0;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqCompCwTreeTypesNode, 0));
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqPolSegDeRmDeItlNode, 0));
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqPolarDecoderNode, 0));
#else
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqCompCwTreeTypesNode, &m_emptyNode1paramDriver));
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqPolSegDeRmDeItlNode, &m_emptyNode1paramDriver));
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqPolarDecoderNode, &m_emptyNode1paramDriver));
#endif
            }
        }


        if(m_outputPrms.cpuCopyOn)
        {
            // size of data copied in a memcopy node can't be zero ----------------------------
            if (m_outputPrms.totNumUciPayloadBytes > 0)
            {
                if (m_ehqMemcpyUciPayloadNodeEnabled != 1)
                {
                    m_ehqMemcpyUciPayloadNodeEnabled = 1;
#if CUDART_VERSION >= 11060
                    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqMemcpyUciPayloadNode, 1));
#endif
                }
                // get memcopy parameters
                cuphySetD2HMemcpyNodeParams(&m_ehqUciPayloadMemcpyD2Hparams, m_outputPrms.pUciPayloadsDevice, m_outputPrms.pUciPayloadsHost, m_outputPrms.totNumUciPayloadBytes);
                // update memcopy node
                CU_CHECK_EXCEPTION(cuGraphExecMemcpyNodeSetParams(m_graphExec, m_ehqMemcpyUciPayloadNode, &m_ehqUciPayloadMemcpyD2Hparams, m_puschCtx));
            } else {
                if (m_ehqMemcpyUciPayloadNodeEnabled != 0)
                {
                    m_ehqMemcpyUciPayloadNodeEnabled = 0;
#if CUDART_VERSION >= 11060
                    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqMemcpyUciPayloadNode, 0));
#else
                    //CU_CHECK_EXCEPTION(cuGraphExecMemcpyNodeSetParams(m_graphExec, m_ehqMemcpyUciPayloadNode, &m_emptyMemcpyD2Hparams));
                    // empty memcopy node throws error, find another way to create empty memcpy node to support older versions of CUDA?
                    // cuphySetD2HMemcpyNodeParams(&m_emptyMemcpyD2Hparams, nullptr, nullptr, 0); does not work
#endif
                }
            }

            // size of data copied in a memcopy node can't be zero ------------------------------
            if (m_nPolUciSegs > 0)
            {
                if (m_ehqMemcpyUciCrcNodeEnabled != 1)
                {
                    m_ehqMemcpyUciCrcNodeEnabled = 1;
#if CUDART_VERSION >= 11060
                    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqMemcpyUciCrcNode, 1));
#endif
                }
                // get memcopy parameters
                cuphySetD2HMemcpyNodeParams(&m_ehqUciCrcMemcpyD2Hparams, m_outputPrms.pUciCrcFlagsDevice_early, m_outputPrms.pUciCrcFlagsHost, m_nPolUciSegs);
                // update memcopy node
                CU_CHECK_EXCEPTION(cuGraphExecMemcpyNodeSetParams(m_graphExec, m_ehqMemcpyUciCrcNode, &m_ehqUciCrcMemcpyD2Hparams, m_puschCtx));
            } else {
                if (m_ehqMemcpyUciCrcNodeEnabled != 0)
                {
                    m_ehqMemcpyUciCrcNodeEnabled = 0;
#if CUDART_VERSION >= 11060
                    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqMemcpyUciCrcNode, 0));
#else
                    //CU_CHECK_EXCEPTION(cuGraphExecMemcpyNodeSetParams(m_graphExec, m_ehqMemcpyUciCrcNode, &m_emptyMemcpyD2Hparams));
                    // empty memcopy node throws error, find another way to create empty memcpy node to support older versions of CUDA?
                    // cuphySetD2HMemcpyNodeParams(&m_emptyMemcpyD2Hparams, nullptr, nullptr, 0); does not work
#endif
                }
            }

            // size of data copied in a memcopy node can't be zero ---------------------------------
            if (m_nUes > 0)
            {
                if (m_ehqMemcpyDetectionStatNodeEnabled != 1)
                {
                    m_ehqMemcpyDetectionStatNodeEnabled = 1;
#if CUDART_VERSION >= 11060
                    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqMemcpyDetectionStatNode, 1));
#endif
                }
                // get memcopy parameters
                cuphySetD2HMemcpyNodeParams(&m_ehqDetectionStatMemcpyD2Hparams, m_outputPrms.pHarqDetectionStatusDevice, m_outputPrms.pHarqDetectionStatusHost, m_nUes);
                // update memcopy node
                CU_CHECK_EXCEPTION(cuGraphExecMemcpyNodeSetParams(m_graphExec, m_ehqMemcpyDetectionStatNode, &m_ehqDetectionStatMemcpyD2Hparams, m_puschCtx));
            } else {
                if (m_ehqMemcpyDetectionStatNodeEnabled != 0)
                {
                    m_ehqMemcpyDetectionStatNodeEnabled = 0;
#if CUDART_VERSION >= 11060
                    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqMemcpyDetectionStatNode, 0));
#else
                    //CU_CHECK_EXCEPTION(cuGraphExecMemcpyNodeSetParams(m_graphExec, m_ehqMemcpyDetectionStatNode, &m_emptyMemcpyD2Hparams));
                    // empty memcopy node throws error, find another way to create empty memcpy node to support older versions of CUDA?
                    // cuphySetD2HMemcpyNodeParams(&m_emptyMemcpyD2Hparams, nullptr, nullptr, 0); does not work
#endif
                }
            }
        }
        else
        {
            if (m_ehqMemcpyUciPayloadNodeEnabled != 0)
            {
                m_ehqMemcpyUciPayloadNodeEnabled = 0;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqMemcpyUciPayloadNode, 0));
#else
                //CU_CHECK_EXCEPTION(cuGraphExecMemcpyNodeSetParams(m_graphExec, m_ehqMemcpyUciPayloadNode, &m_emptyMemcpyD2Hparams));
                // empty memcopy node throws error, find another way to create empty memcpy node to support older versions of CUDA?
                // cuphySetD2HMemcpyNodeParams(&m_emptyMemcpyD2Hparams, nullptr, nullptr, 0); does not work
#endif
            }

            if (m_ehqMemcpyUciCrcNodeEnabled != 0)
            {
                m_ehqMemcpyUciCrcNodeEnabled = 0;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqMemcpyUciCrcNode, 0));
#else
                //CU_CHECK_EXCEPTION(cuGraphExecMemcpyNodeSetParams(m_graphExec, m_ehqMemcpyUciCrcNode, &m_emptyMemcpyD2Hparams));
                // empty memcopy node throws error, find another way to create empty memcpy node to support older versions of CUDA?
                // cuphySetD2HMemcpyNodeParams(&m_emptyMemcpyD2Hparams, nullptr, nullptr, 0); does not work
#endif
            }

            if (m_ehqMemcpyDetectionStatNodeEnabled != 0)
            {
                m_ehqMemcpyDetectionStatNodeEnabled = 0;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqMemcpyDetectionStatNode, 0));
#else
                //CU_CHECK_EXCEPTION(cuGraphExecMemcpyNodeSetParams(m_graphExec, m_ehqMemcpyDetectionStatNode, &m_emptyMemcpyD2Hparams));
                // empty memcopy node throws error, find another way to create empty memcpy node to support older versions of CUDA?
                // cuphySetD2HMemcpyNodeParams(&m_emptyMemcpyD2Hparams, nullptr, nullptr, 0); does not work
#endif
            }
        }

#if 0
        // currently only kernel, memset and memcpy nodes are supported in cuGraphNodeSetEnabled
        if(m_ehqEventReadyNodeEnabled != 1)
        {
            m_ehqEventReadyNodeEnabled = 1;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqEventReadyNode, 1));
#endif
        }
#endif

        if(m_ehqPostWaitNodeEnabled != 1)
        {
            m_ehqPostWaitNodeEnabled = 1;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ehqPostWaitNode, 1));
#endif
        }
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ehqPostWaitNode, &(m_postEarlyHarqWaitCfgs.kernelNodeParamsDriver)));

    }
    else
    {
        disableAllEarlyHarqNodes();
    }

}

cuphyStatus_t PuschRx::updateGraph()
{
    //MemtraceDisableScope md; // Disable temporarity GT-7257
#if CUDART_VERSION < 11000
        throw cuphy::cuda_driver_exception("Graph mode requires CUDA driver kernel node params which requires CUDA 11.0 or higher");
#endif

#ifdef MEMTRACE
    memtrace_set_config(MI_MEMTRACE_CONFIG_ENABLE|MI_MEMTRACE_CONFIG_EXIT_AFTER_BACKTRACE);
#endif

    if(m_cuphyPuschStatPrms.enableEarlyHarq)
    {
        updateEarlyHarqNodes(m_earlyHarqModeEnabled);
    }

    for(int32_t chEstTimeInstIdx = 0; chEstTimeInstIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEstTimeInstIdx)
    {
        for(int hetCfgIdx = 0; hetCfgIdx < m_chEstLaunchCfgs[chEstTimeInstIdx].nCfgs; ++hetCfgIdx)
        {
            if (m_chEstNodesEnabled[chEstTimeInstIdx][hetCfgIdx] != 1)
            {
                m_chEstNodesEnabled[chEstTimeInstIdx][hetCfgIdx] = 1;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_chEstNodes[chEstTimeInstIdx][hetCfgIdx], 1));
#endif
            }
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_chEstNodes[chEstTimeInstIdx][hetCfgIdx], &(m_chEstLaunchCfgs[chEstTimeInstIdx].cfgs[hetCfgIdx].kernelNodeParamsDriver)));
        }
        for(int hetCfgIdx = m_chEstLaunchCfgs[chEstTimeInstIdx].nCfgs; hetCfgIdx < CUPHY_PUSCH_RX_CH_EST_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            if (m_chEstNodesEnabled[chEstTimeInstIdx][hetCfgIdx] != 0)
            {
                m_chEstNodesEnabled[chEstTimeInstIdx][hetCfgIdx] = 0;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_chEstNodes[chEstTimeInstIdx][hetCfgIdx], 0));
#else
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_chEstNodes[chEstTimeInstIdx][hetCfgIdx], &(m_emptyNode2paramsDriver)));
#endif
            }
        }
    }

    if(m_chEstSettings.enableSinrMeasurement || (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE) || (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC))
    {
        for(int hetCfgIdx = 0; hetCfgIdx < m_noiseIntfEstLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].nCfgs; ++hetCfgIdx)
        {
            if (m_noiseIntfEstNodesEnabled[hetCfgIdx] != 1)
            {
                m_noiseIntfEstNodesEnabled[hetCfgIdx] = 1;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_noiseIntfEstNodes[hetCfgIdx], 1));
#endif
            }
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_noiseIntfEstNodes[hetCfgIdx], &(m_noiseIntfEstLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].cfgs[hetCfgIdx].kernelNodeParamsDriver)));
        }
        for(int hetCfgIdx = m_noiseIntfEstLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].nCfgs; hetCfgIdx < CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            if (m_noiseIntfEstNodesEnabled[hetCfgIdx] != 0)
            {
                m_noiseIntfEstNodesEnabled[hetCfgIdx] = 0;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_noiseIntfEstNodes[hetCfgIdx], 0));
#else
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_noiseIntfEstNodes[hetCfgIdx], &(m_emptyNode1paramDriver)));
#endif
            }
        }

    }

    if(m_chEstSettings.enableCfoCorrection || m_chEstSettings.enableToEstimation)
    {
        for(int hetCfgIdx = 0; hetCfgIdx < m_cfoTaEstLaunchCfgs.nCfgs; ++hetCfgIdx)
        {
            if (m_cfoTaEstNodesEnabled[hetCfgIdx] != 1)
            {
                m_cfoTaEstNodesEnabled[hetCfgIdx] = 1;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_cfoTaEstNodes[hetCfgIdx], 1));
#endif
            }
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_cfoTaEstNodes[hetCfgIdx], &(m_cfoTaEstLaunchCfgs.cfgs[hetCfgIdx].kernelNodeParamsDriver)));
        }
        for(int hetCfgIdx = m_cfoTaEstLaunchCfgs.nCfgs; hetCfgIdx < CUPHY_PUSCH_RX_CFO_EST_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            if (m_cfoTaEstNodesEnabled[hetCfgIdx] != 0)
            {
                m_cfoTaEstNodesEnabled[hetCfgIdx] = 0;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_cfoTaEstNodes[hetCfgIdx], 0));
#else
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_cfoTaEstNodes[hetCfgIdx], &(m_emptyNode2paramsDriver)));
#endif
            }
        }
    }

    int32_t bound = 1;
    if(m_chEstSettings.enablePuschTdi)
    {
        bound = CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ;
    }

    for(int32_t chEqTimeInstIdx = 0; chEqTimeInstIdx < bound; ++chEqTimeInstIdx)
    {
        for(int hetCfgIdx = 0; hetCfgIdx < m_chEqCoefCompLaunchCfgs[chEqTimeInstIdx].nCfgs; ++hetCfgIdx)
        {
            if (m_chEqCoefCompNodesEnabled[chEqTimeInstIdx][hetCfgIdx] != 1)
            {
                m_chEqCoefCompNodesEnabled[chEqTimeInstIdx][hetCfgIdx] = 1;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_chEqCoefCompNodes[chEqTimeInstIdx][hetCfgIdx], 1));
#endif
            }
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_chEqCoefCompNodes[chEqTimeInstIdx][hetCfgIdx], &(m_chEqCoefCompLaunchCfgs[chEqTimeInstIdx].cfgs[hetCfgIdx].kernelNodeParamsDriver)));
        }
        for(int hetCfgIdx = m_chEqCoefCompLaunchCfgs[chEqTimeInstIdx].nCfgs; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            if (m_chEqCoefCompNodesEnabled[chEqTimeInstIdx][hetCfgIdx] != 0)
            {
                m_chEqCoefCompNodesEnabled[chEqTimeInstIdx][hetCfgIdx] = 0;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_chEqCoefCompNodes[chEqTimeInstIdx][hetCfgIdx], 0));
#else
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_chEqCoefCompNodes[chEqTimeInstIdx][hetCfgIdx], &(m_emptyNode2paramsDriver)));
#endif
            }

        }
    }

    for(int hetCfgIdx = 0; hetCfgIdx < m_chEqSoftDemapLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].nCfgs; ++hetCfgIdx)
    {
        if (m_chEqSoftDemapNodesEnabled[hetCfgIdx] != 1)
        {
            m_chEqSoftDemapNodesEnabled[hetCfgIdx] = 1;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_chEqSoftDemapNodes[hetCfgIdx], 1));
#endif
        }
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_chEqSoftDemapNodes[hetCfgIdx], &(m_chEqSoftDemapLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].cfgs[hetCfgIdx].kernelNodeParamsDriver)));
    }
    for(int hetCfgIdx = m_chEqSoftDemapLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].nCfgs; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
    {
        if (m_chEqSoftDemapNodesEnabled[hetCfgIdx] != 0)
        {
            m_chEqSoftDemapNodesEnabled[hetCfgIdx] = 0;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_chEqSoftDemapNodes[hetCfgIdx], 0));
#else
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_chEqSoftDemapNodes[hetCfgIdx], &(m_emptyNode2paramsDriver)));
#endif
        }
    }
    
    if(m_chEstSettings.enableDftSOfdm==1)
    {
        for(int hetCfgIdx = 0; hetCfgIdx < m_chEqSoftDemapBluesteinWorkspaceLaunchCfgs.nCfgs; ++hetCfgIdx)
        {
            if (m_chEqSoftDemapBluesteinWorkspaceNodesEnabled[hetCfgIdx] != 1)
            {
                m_chEqSoftDemapBluesteinWorkspaceNodesEnabled[hetCfgIdx] = 1;
    #if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_chEqSoftDemapBluesteinWorkspaceNodes[hetCfgIdx], 1));
    #endif
            }
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_chEqSoftDemapBluesteinWorkspaceNodes[hetCfgIdx], &(m_chEqSoftDemapBluesteinWorkspaceLaunchCfgs.cfgs[hetCfgIdx].kernelNodeParamsDriver)));
        }
        for(int hetCfgIdx = m_chEqSoftDemapBluesteinWorkspaceLaunchCfgs.nCfgs; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            if (m_chEqSoftDemapBluesteinWorkspaceNodesEnabled[hetCfgIdx] != 0)
            {
                m_chEqSoftDemapBluesteinWorkspaceNodesEnabled[hetCfgIdx] = 0;
    #if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_chEqSoftDemapBluesteinWorkspaceNodes[hetCfgIdx], 0));
    #else
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_chEqSoftDemapBluesteinWorkspaceNodes[hetCfgIdx], &(m_emptyNode2paramsDriver)));
    #endif
            }
        }  


        for(int hetCfgIdx = 0; hetCfgIdx < m_chEqSoftDemapIdftLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].nCfgs; ++hetCfgIdx)
        {
            if (m_chEqSoftDemapIdftNodesEnabled[hetCfgIdx] != 1)
            {
                m_chEqSoftDemapIdftNodesEnabled[hetCfgIdx] = 1;
    #if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_chEqSoftDemapIdftNodes[hetCfgIdx], 1));
    #endif
            }
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_chEqSoftDemapIdftNodes[hetCfgIdx], &(m_chEqSoftDemapIdftLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].cfgs[hetCfgIdx].kernelNodeParamsDriver)));
        }
        for(int hetCfgIdx = m_chEqSoftDemapIdftLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].nCfgs; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            if (m_chEqSoftDemapIdftNodesEnabled[hetCfgIdx] != 0)
            {
                m_chEqSoftDemapIdftNodesEnabled[hetCfgIdx] = 0;
    #if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_chEqSoftDemapIdftNodes[hetCfgIdx], 0));
    #else
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_chEqSoftDemapIdftNodes[hetCfgIdx], &(m_emptyNode2paramsDriver)));
    #endif
            }
        }  

        for(int hetCfgIdx = 0; hetCfgIdx < m_chEqSoftDemapAfterDftLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].nCfgs; ++hetCfgIdx)
        {
            if (m_chEqSoftDemapAfterDftNodesEnabled[hetCfgIdx] != 1)
            {
                m_chEqSoftDemapAfterDftNodesEnabled[hetCfgIdx] = 1;
    #if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_chEqSoftDemapAfterDftNodes[hetCfgIdx], 1));
    #endif
            }
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_chEqSoftDemapAfterDftNodes[hetCfgIdx], &(m_chEqSoftDemapAfterDftLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].cfgs[hetCfgIdx].kernelNodeParamsDriver)));
        }
        for(int hetCfgIdx = m_chEqSoftDemapAfterDftLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].nCfgs; hetCfgIdx < CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            if (m_chEqSoftDemapAfterDftNodesEnabled[hetCfgIdx] != 0)
            {
                m_chEqSoftDemapAfterDftNodesEnabled[hetCfgIdx] = 0;
    #if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_chEqSoftDemapAfterDftNodes[hetCfgIdx], 0));
    #else
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_chEqSoftDemapAfterDftNodes[hetCfgIdx], &(m_emptyNode2paramsDriver)));
    #endif
            }
        }
    } //ToDo see if it's necessary to add "else" here and go through all those graph nodes and disable them when needed

    if(m_nUciUes > 0)
    {
        if(m_uciSegLLRs0NodeEnabled != 1)
        {
            m_uciSegLLRs0NodeEnabled = 1;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_uciSegLLRs0Node, 1));
#endif
        }
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_uciSegLLRs0Node, &(m_uciOnPuschSegLLRs0LaunchCfg.kernelNodeParamsDriver)));
    }
    else
    {
        if(m_uciSegLLRs0NodeEnabled != 0)
        {
            m_uciSegLLRs0NodeEnabled = 0;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_uciSegLLRs0Node, 0));
#else
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_uciSegLLRs0Node, &m_emptyNode1paramDriver));
#endif
        }
    }

    if(m_nSpxCws > 0)
    {
        if (m_simplexDecoderNodeEnabled != 1) {
            m_simplexDecoderNodeEnabled = 1;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_simplexDecoderNode, 1));
#endif
        }
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_simplexDecoderNode, &(m_simplexDecoderLaunchCfg.kernelNodeParamsDriver)));
    }
    else
    {
        if (m_simplexDecoderNodeEnabled != 0)
        {
            m_simplexDecoderNodeEnabled = 0;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_simplexDecoderNode, 0));
#else
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_simplexDecoderNode, &m_emptyNode1paramDriver));
#endif
        }
    }

    if(m_nRmCws > 0)
    {
        if (m_rmDecoderNodeEnabled != 1) {
            m_rmDecoderNodeEnabled = 1;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_rmDecoderNode, 1));
#endif
        }
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_rmDecoderNode, &(m_rmDecoderLaunchCfg.kernelNodeParamsDriver)));
    }
    else
    {
        if(m_rmDecoderNodeEnabled != 0)
        {
            m_rmDecoderNodeEnabled = 0;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_rmDecoderNode, 0));
#else
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_rmDecoderNode, &m_emptyNode1paramDriver));
#endif
        }
    }


    if(m_nPolUciSegs > 0)
    {
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_compCwTreeTypesNode, &(m_compCwTreeTypesLaunchCfg.kernelNodeParamsDriver)));
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_polSegDeRmDeItlNode, &(m_polSegDeRmDeItlLaunchCfg.kernelNodeParamsDriver)));
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_polarDecoderNode, &(m_polarDecoderLaunchCfg.kernelNodeParamsDriver)));

        if (m_polarNodeEnabled != 1)
        {
            m_polarNodeEnabled = 1;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_compCwTreeTypesNode, 1));
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_polSegDeRmDeItlNode, 1));
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_polarDecoderNode, 1));
#endif
        }
    }
    else
    {
        if (m_polarNodeEnabled != 0)
        {
            m_polarNodeEnabled = 0;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_compCwTreeTypesNode, 0));
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_polSegDeRmDeItlNode, 0));
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_polarDecoderNode, 0));
#else
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_compCwTreeTypesNode, &m_emptyNode1paramDriver));
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_polSegDeRmDeItlNode, &m_emptyNode1paramDriver));
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_polarDecoderNode, &m_emptyNode1paramDriver));
#endif
        }
    }

    if(m_nCsi2Ues > 0)
    {
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_uciOnPuschCsi2CtrlNode, &(m_uciOnPuschCsi2CtrlLaunchCfg.kernelNodeParamsDriver)));
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_uciOnPuschSegLLRs2Node, &(m_uciOnPuschSegLLRs2LaunchCfg.kernelNodeParamsDriver)));
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_rmDecCsi2Node, &(m_rmDecoderLaunchCfg_csi2.kernelNodeParamsDriver)));
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_simplexDecoderCsi2Node, &(m_simplexDecoderLaunchCfg_csi2.kernelNodeParamsDriver)));

        if(m_cuphyPuschStatPrms.pDbg->forcedNumCsi2Bits > 0)
        {
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_uciOnPuschCsi2CompCwTreeTypesNode, &(m_compCwTreeTypesLaunchCfg_csi2.kernelNodeParamsDriver)));
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_uciOnPuschCsi2PolSegDeRmDeItlNode, &(m_polSegDeRmDeItlLaunchCfg_csi2.kernelNodeParamsDriver)));
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_uciOnPuschCsi2PolarDecoderNode, &(m_polarDecoderLaunchCfg_csi2.kernelNodeParamsDriver)));
        }

        if (m_csi2NodeEnabled != 1)
        {
            m_csi2NodeEnabled = 1;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_uciOnPuschCsi2CtrlNode, 1));
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_uciOnPuschSegLLRs2Node, 1));
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_rmDecCsi2Node, 1));
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_simplexDecoderCsi2Node, 1));

            if(m_cuphyPuschStatPrms.pDbg->forcedNumCsi2Bits > 0)
            {
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_uciOnPuschCsi2CompCwTreeTypesNode, 1));
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_uciOnPuschCsi2PolSegDeRmDeItlNode, 1));
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_uciOnPuschCsi2PolarDecoderNode, 1));
            }
#endif
        }
    }
    else
    {
        if (m_csi2NodeEnabled != 0)
        {
            m_csi2NodeEnabled = 0;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_uciOnPuschCsi2CtrlNode, 0));
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_uciOnPuschSegLLRs2Node, 0));
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_rmDecCsi2Node         , 0));
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_simplexDecoderCsi2Node, 0));

            //CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_uciOnPuschCsi2CompCwTreeTypesNode, 0)); //FixMe these nodes have not been added in createGraph()
            //CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_uciOnPuschCsi2PolSegDeRmDeItlNode, 0));
            //CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_uciOnPuschCsi2PolarDecoderNode, 0));

#else
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_uciOnPuschCsi2CtrlNode, &m_emptyNode1paramDriver));
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_uciOnPuschSegLLRs2Node, &m_emptyNode1paramDriver));
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_rmDecCsi2Node         , &m_emptyNode1paramDriver));
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_simplexDecoderCsi2Node, &m_emptyNode1paramDriver));
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_uciOnPuschCsi2CompCwTreeTypesNode, &m_emptyNode1paramDriver)); //FixMe these nodes have not been added in createGraph()
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_uciOnPuschCsi2PolSegDeRmDeItlNode, &m_emptyNode1paramDriver));
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_uciOnPuschCsi2PolarDecoderNode, &m_emptyNode1paramDriver));
#endif
        }
    }

    if(m_nSchUes > 0)
    {
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_rateMatchNode, &(m_rateMatchLaunchCfg.kernelNodeParamsDriver)));
        if (m_rateMatchNodeEnabled != 1)
        {
            m_rateMatchNodeEnabled = 1;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_rateMatchNode, 1));
#endif
        }

        {
            //ToDo due to different signatures of LDPC kernels, it is not straightforward to avoid dyn mem allocation in graph api call
            // as a temporary workaround disable memtrace for m_ldpcDecoderNodes
            MemtraceDisableScope md;
            for(int i = 0; i < m_LDPCDecodeDescSet.count(); ++i)
            {
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ldpcDecoderNodes[i], &m_ldpcLaunchCfgs[i].kernel_node_params_driver));
                if(m_ldpcDecoderNodesEnabled[i] != 1)
                {
                    m_ldpcDecoderNodesEnabled[i] = 1;
#if CUDART_VERSION >= 11060
                    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ldpcDecoderNodes[i], 1));
#endif
                }
            }
            for(int i = m_LDPCDecodeDescSet.count(); i < N_MAX_LDPC_HET_CFGS; ++i)
            {
                if(m_ldpcDecoderNodesEnabled[i] != 0)
                {
                    m_ldpcDecoderNodesEnabled[i] = 0;
#if CUDART_VERSION >= 11060
                    CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ldpcDecoderNodes[i], 0));
#else
                    CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ldpcDecoderNodes[i], &m_emptyNode2paramsDriver));
#endif
                }
            }
        }

        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_crcNodes[0], &(m_crcLaunchCfgs[0].kernelNodeParamsDriver)));
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_crcNodes[1], &(m_crcLaunchCfgs[1].kernelNodeParamsDriver)));
        if (m_crcNodesEnabled != 1)
        {
            m_crcNodesEnabled = 1;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_crcNodes[0], 1));
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_crcNodes[1], 1));
#endif
        }

    } else
    {

        if (m_rateMatchNodeEnabled != 0)
        {
            m_rateMatchNodeEnabled = 0;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_rateMatchNode, 0));
#else
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_rateMatchNode, &m_emptyNode1paramDriver));
#endif
        }

        for(int i = 0; i < N_MAX_LDPC_HET_CFGS; ++i)
        {
            if(m_ldpcDecoderNodesEnabled[i] != 0)
            {
                m_ldpcDecoderNodesEnabled[i] = 0;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_ldpcDecoderNodes[i], 0));
#else
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ldpcDecoderNodes[i], &m_emptyNode2paramsDriver));
#endif
            }
        }

        if (m_crcNodesEnabled != 0)
        {
            m_crcNodesEnabled = 0;
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_crcNodes[0], 0));
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_crcNodes[1], 0));
#else
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_crcNodes[0], &m_emptyNode1paramDriver));
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_crcNodes[1], &m_emptyNode1paramDriver));
#endif
        }

    }

    if(m_chEstSettings.enableRssiMeasurement)
    {
        for(int hetCfgIdx = 0; hetCfgIdx < m_rssiLaunchCfgs.nCfgs; ++hetCfgIdx)
        {
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_rssiNodes[hetCfgIdx], &(m_rssiLaunchCfgs.cfgs[hetCfgIdx].kernelNodeParamsDriver)));
            if(m_rssiNodesEnabled[hetCfgIdx] != 1)
            {
                m_rssiNodesEnabled[hetCfgIdx] = 1;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_rssiNodes[hetCfgIdx], 1));
#endif
            }
        }
        for(int hetCfgIdx = m_rssiLaunchCfgs.nCfgs; hetCfgIdx < CUPHY_PUSCH_RX_RSSI_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_rssiNodes[hetCfgIdx], &(m_emptyNode1paramDriver)));
            if(m_rssiNodesEnabled[hetCfgIdx] != 0)
            {
                m_rssiNodesEnabled[hetCfgIdx] = 0;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_rssiNodes[hetCfgIdx], 0));
#else
                CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_rssiNodes[hetCfgIdx], &m_emptyNode1paramDriver));
#endif
            }
        }
    }

    if(m_chEstSettings.enableSinrMeasurement)
    {
        for(int hetCfgIdx = 0; hetCfgIdx < m_rsrpLaunchCfgs.nCfgs; ++hetCfgIdx)
        {
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_rsrpNodes[hetCfgIdx], &(m_rsrpLaunchCfgs.cfgs[hetCfgIdx].kernelNodeParamsDriver)));
            if(m_rsrpNodesEnabled[hetCfgIdx] != 1)
            {
                m_rsrpNodesEnabled[hetCfgIdx] = 1;
#if CUDART_VERSION >= 11060
                CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_rsrpNodes[hetCfgIdx], 1));
#endif
            }
        }
        for(int hetCfgIdx = m_rsrpLaunchCfgs.nCfgs; hetCfgIdx < CUPHY_PUSCH_RX_RSRP_N_MAX_HET_CFGS; ++hetCfgIdx)
        {
#if CUDART_VERSION >= 11060
            CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_rsrpNodes[hetCfgIdx], 0));
#else
            CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_rsrpNodes[hetCfgIdx], &m_emptyNode1paramDriver));
#endif
            if(m_rsrpNodesEnabled[hetCfgIdx] != 0)
            {
                m_rsrpNodesEnabled[hetCfgIdx] = 0;
            }
        }
    }

#if CUDART_VERSION >= 11010
    {
        MemtraceDisableScope md; // Disable dynamic memory allocation check temporarity GT-7257
        CUDA_CHECK_EXCEPTION(cudaGraphUpload(m_graphExec, m_cuStream));
    }
#endif

#ifdef MEMTRACE
    memtrace_set_config(0);
#endif

    return CUPHY_STATUS_SUCCESS;
}

cuphyStatus_t PuschRx::setup(cuphyPuschDynPrms_t* pDynPrm)
{
    PUSH_RANGE("cuphySetupPuschRx", 1);
    m_cuphyPuschDynPrms = *(pDynPrm);

    //  if ((pDynPrm->setupPhase == PUSCH_SETUP_PHASE_INVALID) || (pDynPrm->setupPhase >= PUSCH_SETUP_MAX_PHASES))
    //  {
    //      throw cuphy::cuphy_fn_exception(CUPHY_STATUS_INVALID_ARGUMENT, "cuphySetupPuschRx()");
    //  }
    if(pDynPrm->pDataOut->h_harqBufferSizeInBytes == nullptr)
    {
        pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_HARQ_BUFFER_SIZE_NULLPTR;
        pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
        pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    m_cuStream = pDynPrm->cuStream;

    bool enableCpuToGpuDescrAsyncCpy = false; // true;// false;

    if(pDynPrm->setupPhase == PUSCH_SETUP_PHASE_1)
    {
        cuphyStatus_t status = setupCmnPhase1(pDynPrm);
        POP_RANGE
        return status;
    }

    setupCmnPhase2(pDynPrm);
    
    m_cudaGraphModeEnabled = (pDynPrm->procModeBmsk & PUSCH_PROC_MODE_FULL_SLOT_GRAPHS) ? true : false;
    
    cuphyStatus_t status = setupComponents(enableCpuToGpuDescrAsyncCpy, pDynPrm);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        return status;
    }
    if(m_cudaGraphModeEnabled)
    {
        cuphyStatus_t status = updateGraph();
        if(CUPHY_STATUS_SUCCESS != status)
        {
            return status;
        }
    }

    //----------------------------------------------------------------------------------
    // Move data from CPU to GPU
    m_h2dBuffer.asyncCpuToGpuCpy(m_cuStream);

    if(!enableCpuToGpuDescrAsyncCpy)
    {
        m_kernelDynDescr.asyncCpuToGpuCpy(m_cuStream);
    }
    POP_RANGE
    
    return CUPHY_STATUS_SUCCESS;
}

PuschRx::~PuschRx()
{
    CUDA_CHECK(cudaGraphDestroy(m_graph));
    CUDA_CHECK(cudaGraphExecDestroy(m_graphExec));
    destroyComponents();
}

void PuschRx::copyOutputToCPU(cudaStream_t cuStrm)
{
    // printf("copyOutputToCPU: totNumTbs %d totNumCbs %d totNumPayloadBytes %d pTbPayloadsHost %p pTbPayloadsDevice %p\n", m_outputPrms.totNumTbs, m_outputPrms.totNumCbs, m_outputPrms.totNumPayloadBytes, m_outputPrms.pTbPayloadsHost, m_outputPrms.pTbPayloadsDevice);

    // @todo: Potential optimization opportunity: carry out all the output transfers by using a  single cudaMemcpyAsync.
    // This would need the following 2 changes:
    // 1. Allocate d_pOutputTbs, d_pCbCrcs, d_pTbCrcs via a single call to linear allocator (one large allocation vs 3 individual allocations).
    // 2. API change to reflect the single large buffer containing the afore-mentioned buffers.
    if(m_outputPrms.pCbCrcsHost)
    {
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pCbCrcsHost, m_outputPrms.pCbCrcsDevice, sizeof(uint32_t) * m_outputPrms.totNumCbs, cudaMemcpyDeviceToHost, cuStrm));
    }
    if(m_outputPrms.pTbCrcsHost)
    {
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pTbCrcsHost, m_outputPrms.pTbCrcsDevice, sizeof(uint32_t) * m_outputPrms.totNumTbs, cudaMemcpyDeviceToHost, cuStrm));
    }
    if(m_outputPrms.pTbPayloadsHost)
    {
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pTbPayloadsHost, m_outputPrms.pTbPayloadsDevice, m_outputPrms.totNumPayloadBytes, cudaMemcpyDeviceToHost, cuStrm));
    }
    if(m_chEstSettings.enableToEstimation && m_outputPrms.pTaEstsHost)
    {
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pTaEstsHost, m_outputPrms.pTaEstsDevice, sizeof(float) * m_cuphyPuschCellGrpDynPrm.nUes, cudaMemcpyDeviceToHost, cuStrm));
    }
    if(m_chEstSettings.enableRssiMeasurement && m_outputPrms.pRssiHost)
    {
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pRssiHost, m_outputPrms.pRssiDevice, sizeof(float) * m_cuphyPuschCellGrpDynPrm.nUeGrps, cudaMemcpyDeviceToHost, cuStrm));
    }
    if(m_chEstSettings.enableSinrMeasurement)
    {
        // Per UE RSRP
        if(m_outputPrms.pRsrpHost)
        {
            CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pRsrpHost, m_outputPrms.pRsrpDevice, sizeof(float) * m_cuphyPuschCellGrpDynPrm.nUes, cudaMemcpyDeviceToHost, cuStrm));
        }
        // Per UE group pre-equalizer noise variance
        if(m_outputPrms.pNoiseVarPreEqHost)
        {
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
            CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pNoiseVarPreEqHost, m_outputPrms.pNoiseVarPreEqDevice, sizeof(float) * m_cuphyPuschCellGrpDynPrm.nUes, cudaMemcpyDeviceToHost, cuStrm));
#else
            CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pNoiseVarPreEqHost, m_outputPrms.pNoiseVarPreEqDevice, sizeof(float) * m_cuphyPuschCellGrpDynPrm.nUeGrps, cudaMemcpyDeviceToHost, cuStrm));
#endif
        }
        // Per UE post-equalizer noise variance
        if(m_outputPrms.pNoiseVarPostEqHost)
        {
            CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pNoiseVarPostEqHost, m_outputPrms.pNoiseVarPostEqDevice, sizeof(float) * m_cuphyPuschCellGrpDynPrm.nUes, cudaMemcpyDeviceToHost, cuStrm));
        }
        // Per UE pre-Eq SINR
        if(m_outputPrms.pSinrPreEqHost)
        {
            CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pSinrPreEqHost, m_outputPrms.pSinrPreEqDevice, sizeof(float) * m_cuphyPuschCellGrpDynPrm.nUes, cudaMemcpyDeviceToHost, cuStrm));
        }
        // Per layer post-Eq SINR
        if(m_outputPrms.pSinrPostEqHost)
        {
            CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pSinrPostEqHost, m_outputPrms.pSinrPostEqDevice, sizeof(float) * m_cuphyPuschCellGrpDynPrm.nUes, cudaMemcpyDeviceToHost, cuStrm));
        }
    }
    if(m_chEstSettings.enableCfoCorrection)
    {
        if(m_outputPrms.pCfoHzHost)
        {
            CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pCfoHzHost, m_outputPrms.pCfoHzDevice, sizeof(float) * m_cuphyPuschCellGrpDynPrm.nUes, cudaMemcpyDeviceToHost, cuStrm));
        }
    }
    if(m_outputPrms.totNumUciPayloadBytes > 0)
    {
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pUciPayloadsHost, m_outputPrms.pUciPayloadsDevice, m_outputPrms.totNumUciPayloadBytes, cudaMemcpyDeviceToHost, cuStrm));
    }
    if(m_nCsi2Ues > 0)
    {
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pNumCsi2BitsHost, m_outputPrms.pNumCsi2BitsDevice, m_nCsi2Ues * sizeof(uint16_t), cudaMemcpyDeviceToHost, cuStrm));
    }
    if(m_nPolUciSegs > 0)
    {
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pUciCrcFlagsHost, m_outputPrms.pUciCrcFlagsDevice, m_nPolUciSegs, cudaMemcpyDeviceToHost, cuStrm));
    }
    if(m_nUciUes > 0)
    {
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pHarqDetectionStatusHost, m_outputPrms.pHarqDetectionStatusDevice, m_nUes, cudaMemcpyDeviceToHost, cuStrm));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pCsiP1DetectionStatusHost, m_outputPrms.pCsiP1DetectionStatusDevice, m_nUes, cudaMemcpyDeviceToHost, cuStrm));
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pCsiP2DetectionStatusHost, m_outputPrms.pCsiP2DetectionStatusDevice, m_nUes, cudaMemcpyDeviceToHost, cuStrm));
    }
}

void PuschRx::writeDbgBufSynch(cudaStream_t cuStream)
{
    //printStaticApiPrms(&m_cuphyPuschStatPrms);
    //printDynApiPrms(&m_cuphyPuschDynPrms);
    if(m_outputPrms.debugOutputFlag)
    {
        uint32_t           nUes       = m_cuphyPuschCellGrpDynPrm.nUes;
        uint32_t           nUeGrps    = m_cuphyPuschCellGrpDynPrm.nUeGrps;
        uint32_t           nCells     = m_cuphyPuschCellGrpDynPrm.nCells;
        PerTbParams*       pPerTbPrms = m_pTbPrmsCpu;
        cuphyPuschUePrm_t* pUePrms    = m_cuphyPuschCellGrpDynPrm.pUePrms;

        uint16_t slotNumPerCell[16]  = {0};

        //UE/Tb Config Parameters
        uint32_t enableTfPrcd[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t puschIdentity[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t groupOrSequenceHopping[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t N_symb_slot[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t N_slot_frame[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t lowPaprGroupNumber[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t lowPaprSequenceNumber[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t scid[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t rnti[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t rv[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t ndi[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t harqProcessId[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t dataScramId[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t nUeLayers[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t userGroupIndex[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t dmrsPortBmsk[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t targetCodeRate[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t qamModOrder[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t tbSizeBytes[MAX_N_TBS_SUPPORTED] = {0};
        //UE/Tb LDPC Config Parameters
        uint32_t bg[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t Zc[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t num_CBs[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t Ncb[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t Ncb_padded[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t K[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t F[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t nZpBitsPerCb[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t encodedSize[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t KbArray[MAX_N_TBS_SUPPORTED] = {0};
        uint32_t parityNodesArray[MAX_N_TBS_SUPPORTED] = {0};

        //UE GRP Config Parameters
        uint16_t slotNumPerUeGrp[MAX_N_USER_GROUPS_SUPPORTED]  = {0};
        uint32_t startPrb[MAX_N_USER_GROUPS_SUPPORTED] = {0};
        uint32_t nPrb[MAX_N_USER_GROUPS_SUPPORTED] = {0};
        uint32_t startSym[MAX_N_USER_GROUPS_SUPPORTED] = {0};
        uint32_t nSym[MAX_N_USER_GROUPS_SUPPORTED] = {0};
        uint32_t dmrsSymLocBmsk[MAX_N_USER_GROUPS_SUPPORTED] = {0};
        uint32_t dmrsAddlnPos[MAX_N_USER_GROUPS_SUPPORTED] = {0};
        uint32_t dmrsMaxLen[MAX_N_USER_GROUPS_SUPPORTED] = {0};
        uint32_t nDmrsCdmGrpsNoData[MAX_N_USER_GROUPS_SUPPORTED] = {0};
        uint32_t dmrsScrmId[MAX_N_USER_GROUPS_SUPPORTED] = {0};
        uint32_t cellPrmStatIdx[MAX_N_USER_GROUPS_SUPPORTED] = {0};
        uint32_t cellPrmDynIdx[MAX_N_USER_GROUPS_SUPPORTED] = {0};

        for(int32_t ueIdx = 0; ueIdx < nUes; ++ueIdx)
        {
            enableTfPrcd[ueIdx]   = pUePrms[ueIdx].enableTfPrcd;
            puschIdentity[ueIdx]  = pUePrms[ueIdx].puschIdentity;
            groupOrSequenceHopping[ueIdx]  = pUePrms[ueIdx].groupOrSequenceHopping;
            N_symb_slot[ueIdx]    = pUePrms[ueIdx].N_symb_slot;
            N_slot_frame[ueIdx]   = pUePrms[ueIdx].N_slot_frame;
            lowPaprGroupNumber[ueIdx]    = pUePrms[ueIdx].lowPaprGroupNumber;
            lowPaprSequenceNumber[ueIdx] = pUePrms[ueIdx].lowPaprSequenceNumber;
            scid[ueIdx]           = pUePrms[ueIdx].scid;
            rv[ueIdx]             = pUePrms[ueIdx].rv;
            ndi[ueIdx]            = pUePrms[ueIdx].ndi;
            harqProcessId[ueIdx]  = pUePrms[ueIdx].harqProcessId;
            dataScramId[ueIdx]    = pUePrms[ueIdx].dataScramId;
            nUeLayers[ueIdx]      = pUePrms[ueIdx].nUeLayers;
            rnti[ueIdx]           = pUePrms[ueIdx].rnti;
            userGroupIndex[ueIdx] = pPerTbPrms[ueIdx].userGroupIndex;
            dmrsPortBmsk[ueIdx]   = pUePrms[ueIdx].dmrsPortBmsk;
            targetCodeRate[ueIdx] = m_cuphyPuschCellGrpDynPrm.pUePrms[ueIdx].targetCodeRate;
            qamModOrder[ueIdx]    = m_cuphyPuschCellGrpDynPrm.pUePrms[ueIdx].qamModOrder;
            tbSizeBytes[ueIdx]    = m_cuphyPuschCellGrpDynPrm.pUePrms[ueIdx].TBSize;

            //LDPC
            bg[ueIdx]               = pPerTbPrms[ueIdx].bg;
            Zc[ueIdx]               = pPerTbPrms[ueIdx].Zc;
            num_CBs[ueIdx]          = pPerTbPrms[ueIdx].num_CBs;
            Ncb[ueIdx]              = pPerTbPrms[ueIdx].Ncb;
            Ncb_padded[ueIdx]       = pPerTbPrms[ueIdx].Ncb_padded;
            K[ueIdx]                = pPerTbPrms[ueIdx].K;
            F[ueIdx]                = pPerTbPrms[ueIdx].F;
            nZpBitsPerCb[ueIdx]     = pPerTbPrms[ueIdx].nZpBitsPerCb;
            encodedSize[ueIdx]      = pPerTbPrms[ueIdx].encodedSize;
            KbArray[ueIdx]          = m_ldpcPrms.KbArray[ueIdx];
            parityNodesArray[ueIdx] = m_ldpcPrms.parityNodesArray[ueIdx];


        }
        for(int32_t ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
        {
            startPrb[ueGrpIdx] = m_cuphyPuschCellGrpDynPrm.pUeGrpPrms[ueGrpIdx].startPrb;
            nPrb[ueGrpIdx]     = m_cuphyPuschCellGrpDynPrm.pUeGrpPrms[ueGrpIdx].nPrb;
            startSym[ueGrpIdx] = m_cuphyPuschCellGrpDynPrm.pUeGrpPrms[ueGrpIdx].puschStartSym;
            nSym[ueGrpIdx]     = m_cuphyPuschCellGrpDynPrm.pUeGrpPrms[ueGrpIdx].nPuschSym;
            dmrsSymLocBmsk[ueGrpIdx] = m_cuphyPuschCellGrpDynPrm.pUeGrpPrms[ueGrpIdx].dmrsSymLocBmsk;
            dmrsAddlnPos[ueGrpIdx] = m_cuphyPuschCellGrpDynPrm.pUeGrpPrms[ueGrpIdx].pDmrsDynPrm->dmrsAddlnPos;
            dmrsMaxLen[ueGrpIdx] = m_cuphyPuschCellGrpDynPrm.pUeGrpPrms[ueGrpIdx].pDmrsDynPrm->dmrsMaxLen;
            nDmrsCdmGrpsNoData[ueGrpIdx] = m_cuphyPuschCellGrpDynPrm.pUeGrpPrms[ueGrpIdx].pDmrsDynPrm->nDmrsCdmGrpsNoData;
            dmrsScrmId[ueGrpIdx] = m_cuphyPuschCellGrpDynPrm.pUeGrpPrms[ueGrpIdx].pDmrsDynPrm->dmrsScrmId;
            cellPrmStatIdx[ueGrpIdx] = m_cuphyPuschCellGrpDynPrm.pUeGrpPrms[ueGrpIdx].pCellPrm->cellPrmStatIdx;
            cellPrmDynIdx[ueGrpIdx] = m_cuphyPuschCellGrpDynPrm.pUeGrpPrms[ueGrpIdx].pCellPrm->cellPrmDynIdx;
            slotNumPerUeGrp[ueGrpIdx] = m_cuphyPuschCellGrpDynPrm.pUeGrpPrms[ueGrpIdx].pCellPrm->slotNum;
        }
        for(int32_t cellIdx = 0; cellIdx < nCells; ++cellIdx)
        {
            slotNumPerCell[cellIdx] = m_cuphyPuschCellGrpDynPrm.pCellPrms[cellIdx].slotNum;
        }

        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "slotNumPerCell", CUPHY_R_16U, nCells, slotNumPerCell);
        
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "enableTfPrcd", CUPHY_R_32U, nUes, enableTfPrcd);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "puschIdentity", CUPHY_R_32U, nUes, puschIdentity);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "groupOrSequenceHopping", CUPHY_R_32U, nUes, groupOrSequenceHopping);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "N_symb_slot", CUPHY_R_32U, nUes, N_symb_slot);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "N_slot_frame", CUPHY_R_32U, nUes, N_slot_frame);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "lowPaprGroupNumber", CUPHY_R_32U, nUes, lowPaprGroupNumber);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "lowPaprSequenceNumber", CUPHY_R_32U, nUes, lowPaprSequenceNumber);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "scid", CUPHY_R_32U, nUes, scid);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "rv", CUPHY_R_32U, nUes, rv);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "ndi", CUPHY_R_32U, nUes, ndi);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "harqProcessId", CUPHY_R_32U, nUes, harqProcessId);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "dataScramId", CUPHY_R_32U, nUes, dataScramId);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "nUeLayers", CUPHY_R_32U, nUes, nUeLayers);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "rnti", CUPHY_R_32U, nUes, rnti);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "userGroupIndex", CUPHY_R_32U, nUes, userGroupIndex);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "dmrsPortBmsk", CUPHY_R_32U, nUes, dmrsPortBmsk);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "targetCodeRate", CUPHY_R_32U, nUes, targetCodeRate);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "qamModOrder", CUPHY_R_32U, nUes, qamModOrder);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "tbSizeBytes", CUPHY_R_32U, nUes, tbSizeBytes);

        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "ldpc_bg", CUPHY_R_32U, nUes, bg);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "ldpc_Zc", CUPHY_R_32U, nUes, Zc);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "ldpc_num_CBs", CUPHY_R_32U, nUes, num_CBs);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "ldpc_Ncb", CUPHY_R_32U, nUes, Ncb);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "ldpc_Ncb_padded", CUPHY_R_32U, nUes, Ncb_padded);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "ldpc_K", CUPHY_R_32U, nUes, K);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "ldpc_F", CUPHY_R_32U, nUes, F);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "ldpc_nZpBitsPerCb", CUPHY_R_32U, nUes, nZpBitsPerCb);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "ldpc_encodedSize", CUPHY_R_32U, nUes, encodedSize);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "ldpc_KbArray", CUPHY_R_32U, nUes, KbArray);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "ldpc_parityNodesArray", CUPHY_R_32U, nUes, parityNodesArray);


        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "startPrb", CUPHY_R_32U, nUeGrps, startPrb);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "nPrb", CUPHY_R_32U, nUeGrps, nPrb);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "startSym", CUPHY_R_32U, nUeGrps, startSym);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "nSym", CUPHY_R_32U, nUeGrps, nSym);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "dmrsSymLocBmsk", CUPHY_R_32U, nUeGrps, dmrsSymLocBmsk);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "dmrsAddlnPos", CUPHY_R_32U, nUeGrps, dmrsAddlnPos);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "dmrsMaxLen", CUPHY_R_32U, nUeGrps, dmrsMaxLen);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "nDmrsCdmGrpsNoData", CUPHY_R_32U, nUeGrps, nDmrsCdmGrpsNoData);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "dmrsScrmId", CUPHY_R_32U, nUeGrps, dmrsScrmId);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "cellPrmStatIdxPerUeGrp", CUPHY_R_32U, nUeGrps, cellPrmStatIdx);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "cellPrmDynIdxPerUeGrp", CUPHY_R_32U, nUeGrps, cellPrmDynIdx);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, "slotNumPerUeGrp", CUPHY_R_32U, nUeGrps, slotNumPerUeGrp);
        //Input Data
        for(int32_t cellIdx = 0; cellIdx < m_cuphyPuschCellGrpDynPrm.nCells; ++cellIdx)
        {
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefDataRx[cellIdx], m_tRefDataRx[cellIdx].desc(), std::string("DataRx" + std::to_string(cellIdx)).c_str(), cuStream);
        }

        // Channel estimation filters/sequences
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_chEstSettings.tWFreq, "WFreq", cuStream);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_chEstSettings.tShiftSeq, "ShiftSeq", cuStream);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_chEstSettings.tUnShiftSeq, "UnShiftSeq", cuStream);
        cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_chEstSettings.tWFreqSmall, "WFreqSmall", cuStream);

        // Front end tensors
        for(int i = 0; i < nUeGrps; i++)
        {
#if 0
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_drvdUeGrpPrmsGpu[i].dataSymLoc, "DataSymLoc", cuStream);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_drvdUeGrpPrmsGpu[i].qam, "QamInfo", cuStream);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_drvdUeGrpPrmsGpu[i].startPrb, "StartPrb", cuStream);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_drvdUeGrpPrmsGpu[i].nPrb, "NumPrb", cuStream);
#endif

            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefDataEqVec[i], m_tRefDataEqVec[i].desc(), std::string("DataEq" + std::to_string(i)).c_str(), cuStream); //Note, will be zero unless #define WRITE_EQ_OUTPUT enabled in channel_eq.cu
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefCoefVec[i], m_tRefCoefVec[i].desc(), std::string("EqCoeff" + std::to_string(i)).c_str(), cuStream);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefReeDiagInvVec[i], m_tRefReeDiagInvVec[i].desc(), std::string("Ree" + std::to_string(i)).c_str(), cuStream);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefHEstVec[i], m_tRefHEstVec[i].desc(), std::string("HEst" + std::to_string(i)).c_str(), cuStream);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefCfoEstVec[i], m_tRefCfoEstVec[i].desc(), std::string("CfoEst" + std::to_string(i)).c_str(), cuStream);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefLLRVec[i], m_tRefLLRVec[i].desc(), std::string("LLR" + std::to_string(i)).c_str(), cuStream);
            
            if(m_chEstSettings.enableDftSOfdm==1)
            {
                cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefDataEqDftVec[i], m_tRefDataEqDftVec[i].desc(), std::string("DataEqDft" + std::to_string(i)).c_str(), cuStream);
                
                cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefDataEqDftIntermediateVec[i], m_tRefDataEqDftIntermediateVec[i].desc(), std::string("DataEqDftIntermediate" + std::to_string(i)).c_str(), cuStream);
                 
                cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefDftBluesteinWorkspaceTimeVec[i], m_tRefDftBluesteinWorkspaceTimeVec[i].desc(), std::string("DftBluesteinWorkspaceTime" + std::to_string(i)).c_str(), cuStream);
                 
                cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefDftBluesteinWorkspaceFreqVec[i], m_tRefDftBluesteinWorkspaceFreqVec[i].desc(), std::string("DftBluesteinWorkspaceFreq" + std::to_string(i)).c_str(), cuStream);
            }

            if(m_chEstSettings.enableSinrMeasurement || (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE)|| (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC))
            {
                cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefLwInvVec[i], m_tRefLwInvVec[i].desc(), std::string("LwInv" + std::to_string(i)).c_str(), cuStream);
            }
        }

        // RSSI Estimates
        if(m_chEstSettings.enableRssiMeasurement)
        {
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefRssiFull, m_tRefRssiFull.desc(), "RssiFull", cuStream);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefRssi, m_tRefRssi.desc(), "Rssi", cuStream);
        }

        if(m_chEstSettings.enableSinrMeasurement ||(m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE)|| (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC))
        {
            // noise variance estimates
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefNoiseVarPreEq, m_tRefNoiseVarPreEq.desc(), "NoiseVarPreEqPerUe", cuStream);
#else
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefNoiseVarPreEq, m_tRefNoiseVarPreEq.desc(), "NoiseVarPreEq", cuStream);
#endif
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefNoiseVarPostEq, m_tRefNoiseVarPostEq.desc(), "NoiseVarPostEq", cuStream);
        }

        if(m_chEstSettings.enableSinrMeasurement)
        {
            // RSRP, SINR estimates
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefRsrp, m_tRefRsrp.desc(), "Rsrp", cuStream);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefSinrPreEq, m_tRefSinrPreEq.desc(), "SinrPreEq", cuStream);
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefSinrPostEq, m_tRefSinrPostEq.desc(), "SinrPostEq", cuStream);
        }

        if(m_chEstSettings.enableCfoCorrection)
        {
            // CFO(Hz) estimates
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefCfoHz, m_tRefCfoHz.desc(), "CfoEstHzPerUe", cuStream);
        }

        // uci segmented LLRs
        for(int i = 0; i < m_nUciUes; ++i)
        {
            uint16_t ueIdx = m_uciUserIdxsVec[i];

            uint32_t  G               = m_pTbPrmsCpu[ueIdx].G;
            __half*  d_schAndCsi2LLRs = m_pTbPrmsCpu[ueIdx].d_schAndCsi2LLRs;
            if(G > 0)
            {
                cuphy::tensor_ref tRefSchLLR;
                tRefSchLLR.desc().set(CUPHY_R_16F, static_cast<int>(G), cuphy::tensor_flags::align_tight);
                tRefSchLLR.set_addr(static_cast<void*>(d_schAndCsi2LLRs));
    
                cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, tRefSchLLR, tRefSchLLR.desc(), std::string("schLLR" + std::to_string(i)).c_str(), cuStream);
            }
        }

        // TA Estimates
        if(m_chEstSettings.enableToEstimation)
        {
            //TO(microsecond) estimates
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, m_tRefTaEst, m_tRefTaEst.desc(), "ToEstMicroSecPerUe", cuStream);
        }

        if(m_nSchUes>0)
        {
            // de-rate matching output
            for(int i = 0; i < m_nSchUes; ++i)
            {
                uint16_t ueIdx               = m_schUserIdxsVec[i];
                uint8_t*          pTbDeRmLLR = static_cast<uint8_t*>(m_pHarqBuffers[ueIdx]);
                cuphy::tensor_ref tRefTbDeRmLLR;
                tRefTbDeRmLLR.desc().set(CUPHY_R_16F, static_cast<int>(m_pTbPrmsCpu[ueIdx].Ncb_padded), static_cast<int>(m_pTbPrmsCpu[ueIdx].num_CBs), cuphy::tensor_flags::align_tight);
                tRefTbDeRmLLR.set_addr(static_cast<void*>(pTbDeRmLLR));

                cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, tRefTbDeRmLLR, tRefTbDeRmLLR.desc(), std::string("deRmLLR" + std::to_string(i)).c_str(), cuStream);
            }

            // LDPC decoder output
            uint8_t*      pTbDecodeOut     = static_cast<uint8_t*>(d_pLDPCOut);
            const int32_t OUT_STRIDE_WORDS = (MAX_DECODED_CODE_BLOCK_BIT_SIZE + 31) / 32;
            const int32_t BITS_PER_WORD    = 32;
            const int32_t BYTES_PER_WORD   = 4;

            size_t oWords = 0; // offset into LDPC output
            for(int i = 0; i < m_nSchUes; ++i)
            {
                uint16_t ueIdx = m_schUserIdxsVec[i];
                cuphy::tensor_ref tRefDecoderOut;
                tRefDecoderOut.desc().set(CUPHY_BIT, static_cast<int>(BITS_PER_WORD * OUT_STRIDE_WORDS), static_cast<int>(m_pTbPrmsCpu[ueIdx].num_CBs), cuphy::tensor_flags::align_tight);
                tRefDecoderOut.set_addr(static_cast<void*>(static_cast<uint32_t*>(d_pLDPCOut) + oWords));
                oWords += OUT_STRIDE_WORDS * m_pTbPrmsCpu[ueIdx].num_CBs;

                cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, tRefDecoderOut, tRefDecoderOut.desc(), std::string("LDPCDecodeOut" + std::to_string(i)).c_str(), cuStream);
                pTbDecodeOut += BYTES_PER_WORD * OUT_STRIDE_WORDS * m_pTbPrmsCpu[ueIdx].num_CBs;
            }

            // TB CRCs
            cuphy::tensor_ref tTbCRC;
            tTbCRC.desc().set(CUPHY_R_32U, m_outputPrms.totNumTbs, cuphy::tensor_flags::align_tight);
            tTbCRC.set_addr(static_cast<void*>(m_outputPrms.pTbCrcsDevice));
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, tTbCRC, tTbCRC.desc(), "tbCrcs", cuStream);

            // CB CRCs
            cuphy::tensor_ref tCbCRC;
            tCbCRC.desc().set(CUPHY_R_32U, m_outputPrms.totNumCbs, cuphy::tensor_flags::align_tight);
            tCbCRC.set_addr(static_cast<void*>(m_outputPrms.pCbCrcsHost));
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, tCbCRC, tCbCRC.desc(), "cbCrcs", cuStream);

            // TB payload
            cuphy::tensor_ref tTbPayload;
            tTbPayload.desc().set(CUPHY_R_8U, m_outputPrms.totNumPayloadBytes, cuphy::tensor_flags::align_tight);
            tTbPayload.set_addr(static_cast<void*>(m_outputPrms.pTbPayloadsDevice));
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, tTbPayload, tTbPayload.desc(), "TbPayload", cuStream);
        }

        // UCI output (first two bytes)
        if(m_outputPrms.totNumPayloadBytes > 0)
        {
            cuphy::tensor_ref tUciPayload2Bytes;
            tUciPayload2Bytes.desc().set(CUPHY_R_8U, 2, cuphy::tensor_flags::align_tight);
            tUciPayload2Bytes.set_addr(static_cast<void*>(m_outputPrms.pUciPayloadsDevice));
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, tUciPayload2Bytes, tUciPayload2Bytes.desc(), "uciPayload2Bytes", cuStream);
        }

        // first CW tree and CW LLR
        if(m_nPolUciSegs > 0)
        {
            cuphyPolarUciSegPrm_t& uciSegPrms    = m_pUciSegPrmsCpu[0];
            uint16_t               N_cw          = uciSegPrms.N_cw;
            uint16_t               nBytes_cwTree = 2 * N_cw;
            uint16_t               nCbEstWords   = div_round_up(static_cast<uint16_t>(uciSegPrms.K_cw - uciSegPrms.nCrcBits), static_cast<uint16_t>(32));

            cuphy::tensor_ref tCwTree;
            tCwTree.desc().set(CUPHY_R_8U, nBytes_cwTree, cuphy::tensor_flags::align_tight);
            tCwTree.set_addr(static_cast<void*>(m_cwTreeTypesAddrVec[0]));
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, tCwTree, tCwTree.desc(), "cwTreeTypes", cuStream);

            cuphy::tensor_ref tCwLLRs;
            tCwLLRs.desc().set(CUPHY_R_16F, N_cw, cuphy::tensor_flags::align_tight);
            tCwLLRs.set_addr(static_cast<void*>(m_cwLLRsAddrVec[0]));
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, tCwLLRs, tCwLLRs.desc(), "cwLLRs", cuStream);

            cuphy::tensor_ref tCbEst;
            tCbEst.desc().set(CUPHY_R_32U, nCbEstWords, cuphy::tensor_flags::align_tight);
            tCbEst.set_addr(static_cast<void*>(m_cbEstAddrVec[0]));
            cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, tCbEst, tCbEst.desc(), "cbEst", cuStream);
        }

        // // first Harq LLRs
        // uint32_t G_harq = m_tbPrmsArrayCpu[0].G_harq;
        // if(G_harq > 0)
        // {
        //     cuphy::tensor_ref tUciHarqLLRs;
        //     tUciHarqLLRs.desc().set(CUPHY_R_16F, G_harq, cuphy::tensor_flags::align_tight);
        //     tUciHarqLLRs.set_addr(static_cast<void*>(m_harqLLRaddrVec[0]));
        //     cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, tUciHarqLLRs, tUciHarqLLRs.desc(), "uciHarqLLRs", cuStream);
        // }

        // uint32_t G_csi1 = m_tbPrmsArrayCpu[0].G_csi1;
        // if(G_csi1 > 0)
        // {
        //     cuphy::tensor_ref tUciCsi1LLRs;
        //     tUciCsi1LLRs.desc().set(CUPHY_R_16F, G_csi1, cuphy::tensor_flags::align_tight);
        //     tUciCsi1LLRs.set_addr(static_cast<void*>(m_csi1LLRaddrVec[0]));
        //     cuphy::write_HDF5_dataset(m_outputPrms.outHdf5File, tUciCsi1LLRs, tUciCsi1LLRs.desc(), "uciCsi1LLRs", cuStream);
        // }
    }
}

void PuschRx::uciEarlyHarqKernelLaunch()
{
    const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_preEarlyHarqWaitCfgs.kernelNodeParamsDriver;
    CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, m_cuStream));
        
    for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_chEstLaunchCfgs[0].nCfgs; ++hetCfgIdx)
    {
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_chEstLaunchCfgs[0].cfgs[hetCfgIdx].kernelNodeParamsDriver;
        CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, m_cuStream));
    }
    
    if(m_chEstSettings.enableSinrMeasurement  || (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE)|| (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC))
    {
        for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_noiseIntfEstLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH].nCfgs; ++hetCfgIdx)
        {
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_noiseIntfEstLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH].cfgs[hetCfgIdx].kernelNodeParamsDriver;

            CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, m_cuStream));
        }
    }
    
    for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_chEqCoefCompLaunchCfgs[0].nCfgs; ++hetCfgIdx)
    {
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_chEqCoefCompLaunchCfgs[0].cfgs[hetCfgIdx].kernelNodeParamsDriver;
        CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, m_cuStream));
    }

    for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_chEqSoftDemapLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH].nCfgs; ++hetCfgIdx)
    {
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_chEqSoftDemapLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH].cfgs[hetCfgIdx].kernelNodeParamsDriver;
        CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, m_cuStream));
    }
    
    if(m_chEstSettings.enableDftSOfdm==1)
    {
        for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_chEqSoftDemapBluesteinWorkspaceLaunchCfgs.nCfgs; ++hetCfgIdx)
        {
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_chEqSoftDemapBluesteinWorkspaceLaunchCfgs.cfgs[hetCfgIdx].kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, m_cuStream));
        }

        for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_chEqSoftDemapIdftLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH].nCfgs; ++hetCfgIdx)
        {
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_chEqSoftDemapIdftLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH].cfgs[hetCfgIdx].kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, m_cuStream));
        }
         
        for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_chEqSoftDemapAfterDftLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH].nCfgs; ++hetCfgIdx)
        {
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_chEqSoftDemapAfterDftLaunchCfgs[CUPHY_PUSCH_SUB_SLOT_PATH].cfgs[hetCfgIdx].kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, m_cuStream));
        }
    }
        
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
    // use stream pool to launch compCwTreeTypes along uciOnPuschSeqLLRs0 on different streams
    m_G1streamPool.fork(m_cuStream);
#endif

    if(m_nEarlyHarqUes > 0)
    {
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
        cudaStream_t stream = m_G1streamPool.current_stream().handle();
#else
        cudaStream_t stream = m_cuStream;
#endif
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_uciOnPuschEarlySegLLRs0LaunchCfg.kernelNodeParamsDriver;
        CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, stream));
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
        m_uciOnPuschSegLLRs0Event[CUPHY_PUSCH_SUB_SLOT_PATH].record(stream);
        m_G1streamPool.advance();
#endif
    }

#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
    if(m_nUciUes > 0)
    {
        // Force the main stream to wait for event from m_uciOnPuschSegLLRs0LaunchCfg
        CUDA_CHECK(cudaStreamWaitEvent(m_cuStream, m_uciOnPuschSegLLRs0Event[CUPHY_PUSCH_SUB_SLOT_PATH].handle(), 0));
    }
    // use stream pool to launch simplex_decoder, rm_decoder_3 and polSegDeRmDeIt/polarDecoder on 3 different streams
    m_G2streamPool.fork(m_cuStream, 3);
#endif

    if(m_nSpxCws_early > 0)
    {
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
        cudaStream_t stream = m_G2streamPool.current_stream().handle();
#else
        cudaStream_t stream = m_cuStream;
#endif
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_simplexDecoderLaunchCfg_early.kernelNodeParamsDriver;
        CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, stream));
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
        m_G2streamPool.advance();
#endif
    }

    if(m_nRmCws_early > 0)
    {
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
        cudaStream_t stream = m_G2streamPool.current_stream().handle();
#else
        cudaStream_t stream = m_cuStream;
#endif
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_rmDecoderLaunchCfg_early.kernelNodeParamsDriver;
        CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, stream));
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
        m_G2streamPool.advance();
#endif
    }

    if(m_nPolUciSegs_early > 0)
    {
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
        cudaStream_t stream = m_G1streamPool.current_stream().handle();
#else
        cudaStream_t stream = m_cuStream;
#endif
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver1 = m_compCwTreeTypesLaunchCfg_early.kernelNodeParamsDriver;
        CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver1, stream));

#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
        m_compCwTreeTypesEvent[CUPHY_PUSCH_SUB_SLOT_PATH].record(stream);
#endif
    }

    if(m_nPolUciSegs_early > 0)
    {
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
        cudaStream_t stream = m_G2streamPool.current_stream().handle();
        CUDA_CHECK(cudaStreamWaitEvent(stream, m_compCwTreeTypesEvent[CUPHY_PUSCH_SUB_SLOT_PATH].handle(), 0));
#else
        cudaStream_t stream = m_cuStream;
#endif
        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver2 = m_polSegDeRmDeItlLaunchCfg_early.kernelNodeParamsDriver;
        CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver2, stream));

        const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver3 = m_polarDecoderLaunchCfg_early.kernelNodeParamsDriver;
        CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver3, stream));
    }

#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
    m_G2streamPool.join(m_cuStream);
#endif

    if(m_outputPrms.cpuCopyOn)
    {      
        if(m_outputPrms.totNumUciPayloadBytes > 0)
            CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pUciPayloadsHost, m_outputPrms.pUciPayloadsDevice, m_outputPrms.totNumUciPayloadBytes, cudaMemcpyDeviceToHost, m_cuStream));
        
        if(m_nPolUciSegs > 0)
            CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pUciCrcFlagsHost, m_outputPrms.pUciCrcFlagsDevice_early, m_nPolUciSegs, cudaMemcpyDeviceToHost, m_cuStream)); 
            
        if(m_nUciUes > 0)   
            CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(m_outputPrms.pHarqDetectionStatusHost, m_outputPrms.pHarqDetectionStatusDevice, m_nUes, cudaMemcpyDeviceToHost, m_cuStream));
    }
    
    // notify cuPHY-CP of the completion of early-HARQ processing
    CUDA_CHECK_EXCEPTION(cudaEventRecord(m_cuphyPuschStatPrms.earlyHarqReadyEvent, m_cuStream));
    
    const CUDA_KERNEL_NODE_PARAMS& kernelNodePostEHQParamsDriver = m_postEarlyHarqWaitCfgs.kernelNodeParamsDriver;
    CU_CHECK_EXCEPTION(launch_kernel(kernelNodePostEHQParamsDriver, m_cuStream));
}

//------------------------------------------------------------------------------------------------------------

void PuschRx::run(cuphyPuschRunPhase_t runPhase)
{
    PUSH_RANGE("cuphyRunPuschRx", 1);
    if((runPhase == PUSCH_RUN_PHASE_1) || (runPhase == PUSCH_RUN_PHASE_3))
    {
    if(m_cudaGraphModeEnabled)
    {
#if CUDART_VERSION < 11000
        throw cuphy::cuda_driver_exception("Graph mode requires CUDA driver kernel node params which requires CUDA 11.0 or higher");
#endif
        MemtraceDisableScope md; // Disable temporarity GT-7257
        CUDA_CHECK_EXCEPTION(cudaGraphLaunch(m_graphExec, m_cuStream));
        // graphPrintGroupInfo(m_graphExec, stdout);
    }
    else
    {
        //*********************** Start of Early HARQ Sub-Slot Processing Task Graph ********************************/
        if(m_earlyHarqModeEnabled)
        {
            uciEarlyHarqKernelLaunch();
        }
        //************************ End of Early HARQ Sub-Slot Processing Task Graph *********************************/

        /////////////////////////////////// full-slot processing ////////////////////////////////////////////
        for(uint32_t chEstInstIdx = 0; chEstInstIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST; ++chEstInstIdx)
        {
            for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_chEstLaunchCfgs[chEstInstIdx].nCfgs; ++hetCfgIdx)
            {
                const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_chEstLaunchCfgs[chEstInstIdx].cfgs[hetCfgIdx].kernelNodeParamsDriver;
                CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, m_cuStream));
            }
        }

        if(m_chEstSettings.enableSinrMeasurement  || (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE)|| (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC))
        {
            for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_noiseIntfEstLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].nCfgs; ++hetCfgIdx)
            {
                const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_noiseIntfEstLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].cfgs[hetCfgIdx].kernelNodeParamsDriver;

                CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, m_cuStream));
            }
        }

        for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_cfoTaEstLaunchCfgs.nCfgs; ++hetCfgIdx)
        {
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriverCfo = m_cfoTaEstLaunchCfgs.cfgs[hetCfgIdx].kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriverCfo, m_cuStream));
        }

        for(uint32_t chEqInstIdx = 0; chEqInstIdx < CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ; ++chEqInstIdx)
        {
            for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_chEqCoefCompLaunchCfgs[chEqInstIdx].nCfgs; ++hetCfgIdx)
            {
                const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_chEqCoefCompLaunchCfgs[chEqInstIdx].cfgs[hetCfgIdx].kernelNodeParamsDriver;
                CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, m_cuStream));
            }
        }
        
        for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_chEqSoftDemapLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].nCfgs; ++hetCfgIdx)
        {
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_chEqSoftDemapLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].cfgs[hetCfgIdx].kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, m_cuStream));
        }
        
        if(m_chEstSettings.enableDftSOfdm==1)
        {
            for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_chEqSoftDemapBluesteinWorkspaceLaunchCfgs.nCfgs; ++hetCfgIdx)
            {
                const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_chEqSoftDemapBluesteinWorkspaceLaunchCfgs.cfgs[hetCfgIdx].kernelNodeParamsDriver;
                CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, m_cuStream));
            }
    
            for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_chEqSoftDemapIdftLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].nCfgs; ++hetCfgIdx)
            {
                const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_chEqSoftDemapIdftLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].cfgs[hetCfgIdx].kernelNodeParamsDriver;
                CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, m_cuStream));
            }
             
            for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_chEqSoftDemapAfterDftLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].nCfgs; ++hetCfgIdx)
            {
                const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_chEqSoftDemapAfterDftLaunchCfgs[CUPHY_PUSCH_FULL_SLOT_PATH].cfgs[hetCfgIdx].kernelNodeParamsDriver;
                CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, m_cuStream));
            }
        }

#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
        // use stream pool to launch uciOnPuschSeqLLRs0 along with compCwTreeTypes on different streams
        m_G1streamPool.fork(m_cuStream);
#endif

        if(m_nUciUes > 0)
        {
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
            cudaStream_t stream = m_G1streamPool.current_stream().handle();
#else
            cudaStream_t stream = m_cuStream;
#endif
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_uciOnPuschSegLLRs0LaunchCfg.kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, stream));
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
            m_uciOnPuschSegLLRs0Event[CUPHY_PUSCH_FULL_SLOT_PATH].record(stream);
            m_G1streamPool.advance();
#endif
        }

        //------------------------------------------------------------------------------------------------------------
        //Backend

#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
        if(m_nUciUes > 0)
        {
            // Force the main stream to wait for event from m_uciOnPuschSegLLRs0LaunchCfg
            CUDA_CHECK(cudaStreamWaitEvent(m_cuStream, m_uciOnPuschSegLLRs0Event[CUPHY_PUSCH_FULL_SLOT_PATH].handle(), 0));
        }
        // use stream pool to launch simplex_decoder, rm_decoder_3 and polSegDeRmDeIt/polarDecoder on 3 different streams
        m_G2streamPool.fork(m_cuStream, 3);
#endif

        if(m_nSpxCws > 0)
        {
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
            cudaStream_t stream = m_G2streamPool.current_stream().handle();
#else
            cudaStream_t stream = m_cuStream;
#endif
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_simplexDecoderLaunchCfg.kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, stream));
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
            m_G2streamPool.advance();
#endif
        }

        if(m_nRmCws > 0)
        {
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
            cudaStream_t stream = m_G2streamPool.current_stream().handle();
#else
            cudaStream_t stream = m_cuStream;
#endif
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_rmDecoderLaunchCfg.kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, stream));
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
            m_G2streamPool.advance();
#endif
        }

        if(m_nPolUciSegs > 0)
        {
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
            cudaStream_t stream = m_G1streamPool.current_stream().handle();
#else
            cudaStream_t stream = m_cuStream;
#endif
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver1 = m_compCwTreeTypesLaunchCfg.kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver1, stream));

#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
            m_compCwTreeTypesEvent[CUPHY_PUSCH_FULL_SLOT_PATH].record(stream);
#endif
        }

        if(m_nPolUciSegs > 0)
        {
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
            cudaStream_t stream = m_G2streamPool.current_stream().handle();
            CUDA_CHECK(cudaStreamWaitEvent(stream, m_compCwTreeTypesEvent[CUPHY_PUSCH_FULL_SLOT_PATH].handle(), 0));
#else
            cudaStream_t stream = m_cuStream;
#endif
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver2 = m_polSegDeRmDeItlLaunchCfg.kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver2, stream));

            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver3 = m_polarDecoderLaunchCfg.kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver3, stream));
        }

#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
        m_G2streamPool.join(m_cuStream);
#endif
//---------------------------------------------------------------------------------------------------------------------------
        if(m_nCsi2Ues > 0)
        {
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver1 = m_uciOnPuschCsi2CtrlLaunchCfg.kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver1, m_cuStream));

            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver2 = m_uciOnPuschSegLLRs2LaunchCfg.kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver2, m_cuStream));

#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
            // use stream pool to launch simplex_decoder, rm_decoder_3, polar decoder and de_rate_matching on different streams
            m_G2streamPool.fork(m_cuStream);
            cudaStream_t stream = m_G2streamPool.current_stream().handle();
#else
            cudaStream_t stream = m_cuStream;
#endif

            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver3 = m_rmDecoderLaunchCfg_csi2.kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver3, stream));
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
            m_G2streamPool.advance();
            stream = m_G2streamPool.current_stream().handle();
#endif

            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver4 = m_simplexDecoderLaunchCfg_csi2.kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver4, stream));

            if(m_cuphyPuschStatPrms.pDbg->forcedNumCsi2Bits > 0)
            {
                #ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
                            m_G2streamPool.advance();
                            stream = m_G2streamPool.current_stream().handle();
                #endif

                const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver5 = m_compCwTreeTypesLaunchCfg_csi2.kernelNodeParamsDriver;
                CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver5, stream));

                const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver6 = m_polSegDeRmDeItlLaunchCfg_csi2.kernelNodeParamsDriver;
                CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver6, stream));
                                
                const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver7 = m_polarDecoderLaunchCfg_csi2.kernelNodeParamsDriver;
                CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver7, stream));
                }

#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
            m_G2streamPool.advance();
#endif
        }

        if(m_nSchUes > 0)
        {
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
            if(m_nCsi2Ues <= 0)
            {
                m_G2streamPool.fork(m_cuStream, 1);
            }
            cudaStream_t stream = m_G2streamPool.current_stream().handle();
#else
            cudaStream_t stream = m_cuStream;
#endif

            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_rateMatchLaunchCfg.kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, stream));
            m_rateMatchEvent.record(stream);
            cudaError_t e = cudaStreamWaitEvent(m_cuStream, m_rateMatchEvent.handle(), 0);

#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
            m_rateMatchEvent.record(stream);
            CUDA_CHECK(cudaStreamWaitEvent(m_cuStream, m_rateMatchEvent.handle(), 0));
#endif

            if(m_LDPCkernelLaunchMode & PUSCH_RX_ENABLE_DRIVER_LDPC_LAUNCH)
            {
                for(int i = 0; i < m_LDPCDecodeDescSet.count(); ++i)
                {
                    const CUDA_KERNEL_NODE_PARAMS& kernel_node_params_driver = m_ldpcLaunchCfgs[i].kernel_node_params_driver;
                    CU_CHECK_EXCEPTION(launch_kernel(kernel_node_params_driver, m_cuStream));
                // return (CUDA_SUCCESS == e) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
                }
            }
            else
            {
                if(m_LDPCkernelLaunchMode & PUSCH_RX_LDPC_STREAM_SEQUENTIAL)
                {
                    launchLDPCStreamsTensor(m_cuStream); // Note: using tensor interface for sequential for now
                }
                else //PUSCH_RX_LDPC_STREAM_POOL or PUSCH_RX_ENABLE_LDPC_DEC_SINGLE_STREAM_OPT
                {
                    launchLDPCStreamsTB(m_cuStream);
                }
            }

            {
                const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_crcLaunchCfgs[0].kernelNodeParamsDriver;
                CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, m_cuStream));
            }
            {
                const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_crcLaunchCfgs[1].kernelNodeParamsDriver;
                CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, m_cuStream));
            }
        }

        //if(m_cmnPrms.rssiSymLocBmsk)//TODO:revisit this when enable RSSI
        /*Note: rssiSymLocBmsk for a given slot could be 0 even though enableRssiMeasurement is set,
         in such a case the launch below of RSSI kernel would fail*/
        if(m_chEstSettings.enableRssiMeasurement)
        {
            for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_rssiLaunchCfgs.nCfgs; ++hetCfgIdx)
            {
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
                cudaStream_t stream = m_G1streamPool.current_stream().handle();
#else
                cudaStream_t stream = m_cuStream;
#endif
                const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_rssiLaunchCfgs.cfgs[hetCfgIdx].kernelNodeParamsDriver;
                CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, stream));
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
                m_G1streamPool.advance();
#endif
            }
        }

        if(m_chEstSettings.enableSinrMeasurement)
        {
            for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_rsrpLaunchCfgs.nCfgs; ++hetCfgIdx)
            {
                const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriver = m_rsrpLaunchCfgs.cfgs[hetCfgIdx].kernelNodeParamsDriver;

#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
                cudaStream_t stream = m_G1streamPool.current_stream().handle();
#else
                cudaStream_t stream = m_cuStream;
#endif
                CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriver, stream));
#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
                m_G1streamPool.advance();
#endif
            }
        }

#ifdef PUSCH_RX_ENABLE_MULTI_STREAM_LAUNCH
        m_G1streamPool.join(m_cuStream);
        m_G2streamPool.join(m_cuStream);
#endif
    }
    }
    
    if((runPhase == PUSCH_RUN_PHASE_2) || (runPhase == PUSCH_RUN_PHASE_3))
    {
        if(m_outputPrms.cpuCopyOn)
        {
            copyOutputToCPU(m_cuStream);
        }
    }

    POP_RANGE
}

////////////////////////////////////////////////////////////////////////
// PuschRx::launchLDPCStreamsTensor()
void PuschRx::launchLDPCStreamsTensor(cudaStream_t strm)
{
    uint32_t oBytes = 0;
    uint32_t cs     = 0;

    uint32_t f = SPLIT_LDPC ? SPLIT_LDPC : 1;

    uint32_t nTb   = m_nSchUes;
    uint32_t curTb = 0;
    for(int i = 0; i < nTb * f; i++)
    {
        uint16_t ueIdx = m_schUserIdxsVec[i % nTb];
        int      dims[2];
        int      strides[2];
        uint32_t ncbs = m_pTbPrmsCpu[ueIdx].num_CBs / f;
        if(i == nTb * f - 1)
            ncbs += m_pTbPrmsCpu[ueIdx].num_CBs % f;
        dims[0]    = m_pTbPrmsCpu[ueIdx].Ncb_padded;
        dims[1]    = ncbs;
        strides[0] = 1;
        strides[1] = m_pTbPrmsCpu[ueIdx].Ncb_padded;

        cuphy::tensor_layout tlTB(2, dims, strides);
        cuphy::tensor_info   tiTB(m_ldpcPrms.useHalf ? CUPHY_R_16F : CUPHY_R_32F, tlTB);
        cuphy::tensor_desc   tdTB(tiTB);

        dims[0]    = MAX_DECODED_CODE_BLOCK_BIT_SIZE;
        dims[1]    = ncbs;
        strides[0] = 1;
        strides[1] = MAX_DECODED_CODE_BLOCK_BIT_SIZE;

        cuphy::tensor_layout otlTB(2, dims, strides);
        cuphy::tensor_info   otiTB(CUPHY_BIT, otlTB);
        cuphy::tensor_desc   otdTB(otiTB);

        cuphy::LDPC_decode_config dec_config(m_ldpcPrms.useHalf ? CUPHY_R_16F : CUPHY_R_32F, // LLR type
                                             m_ldpcPrms.parityNodesArray[ueIdx],           // num parity nodes
                                             m_pTbPrmsCpu[ueIdx].Zc,                   // lifting size
                                             m_ldpcPrms.nIterations,                         // max iterations
                                             m_ldpcPrms.KbArray[ueIdx],                    // num info nodes (Kb)
                                             0.8125f,                                        // normalization
                                             m_ldpcPrms.flags,                               // flags,
                                             m_pTbPrmsCpu[ueIdx].bg,                   // base graph
                                             m_ldpcPrms.algoIndex,                           // algorithm index
                                             nullptr);                                       // workspace

        cuphy::LDPC_decode_tensor_params dec_params(dec_config,
                                                    otdTB.handle(),                             // output descriptor
                                                    static_cast<char*>(d_pLDPCOut) + oBytes,    // output address
                                                    tdTB.handle(),                              // LLR descriptor
                                                    static_cast<char*>(m_pHarqBuffers[ueIdx])); // LLR address

        m_LDPCdecoder.decode(dec_params, strm);

        oBytes += otdTB.get_size_in_bytes();
        cs += m_pTbPrmsCpu[i % nTb].num_CBs;

        if((i % f) == (f - 1))
        {
            curTb++;
        }

        //m_ldpcStreamPool.advance();
    }

    // Subsequent kernel submissions to strm must wait for all of
    // the LDPC kernels to complete
    //m_ldpcStreamPool.join(strm);
}

////////////////////////////////////////////////////////////////////////
// PuschRx::prepareLDPCStreamsTB()
void PuschRx::prepareLDPCStreamsTB()
{
    //------------------------------------------------------------------
    // Reset the count of valid LDPC descriptors in the descriptor set
    m_LDPCDecodeDescSet.reset();
    //------------------------------------------------------------------
    // Collect transport blocks with identical LDPC configurations into
    // descriptors.
    cuphyLDPCParams& LDPC_params      = m_ldpcPrms;
    const int32_t    OUT_STRIDE_WORDS = (MAX_DECODED_CODE_BLOCK_BIT_SIZE + 31) / 32;

    size_t oWords = 0; // offset into LDPC output

    for(int i = 0; i < m_nSchUes; i++)
    {
        uint16_t ueIdx = m_schUserIdxsVec[i];

        const int16_t BG         = m_pTbPrmsCpu[ueIdx].bg;          // base graph
        const int     Z          = m_pTbPrmsCpu[ueIdx].Zc;          // lifting value
        const int     NUM_PARITY = LDPC_params.parityNodesArray[ueIdx]; // num parity nodes
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Find/allocate a descriptor with a matching configuration
        cuphy::LDPC_decode_desc& desc = m_LDPCDecodeDescSet.find(BG, Z, NUM_PARITY);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Populate the descriptor configuration fields for the first
        // transport block in the descriptor. (BG, Z, and NUM_PARITY
        // are already set by the find() function.)
        if(0 == desc.num_tbs)
        {
            desc.config.llr_type       = (LDPC_params.useHalf ? CUPHY_R_16F : CUPHY_R_32F);
            desc.config.max_iterations = LDPC_params.nIterations;

            switch(m_cuphyPuschStatPrms.ldpcMaxNumItrAlgo)
            {
                case LDPC_MAX_NUM_ITR_ALGO_TYPE_FIXED : {
                    desc.config.max_iterations = LDPC_params.nIterations;
                    break;
                    }
                case LDPC_MAX_NUM_ITR_ALGO_TYPE_LUT : {
                    cuphyPuschUePrm_t& uePrms = m_cuphyPuschDynPrms.pCellGrpDynPrm->pUePrms[ueIdx];
                    float spectralEfficency   = static_cast<float>(uePrms.qamModOrder) * static_cast<float>(uePrms.targetCodeRate) / static_cast<float>(10240);
                    if(spectralEfficency > 7.2)
                    {
                        desc.config.max_iterations = 7;
                    }else if(spectralEfficency < 0.4)
                    {
                        desc.config.max_iterations = 20;
                    }else
                    {
                        desc.config.max_iterations = 10;
                    }
                    break;         
                }
                default : {
                    desc.config.max_iterations = 10;
                }
            }

            // TEMPORARY HACK FOR TESTING NUM_ITERATIONS
            if(getenv("LDPC_NUM_ITER"))
            {
                if(1 != sscanf(getenv("LDPC_NUM_ITER"), "%hi", &desc.config.max_iterations))
                {
                    NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Error reading LDPC_NUM_ITER from environment variable {}, using file value {}", __FUNCTION__, getenv("LDPC_NUM_ITER"), static_cast<int>(desc.config.max_iterations));
                }
            }
            desc.config.Kb        = LDPC_params.KbArray[ueIdx];
            desc.config.flags     = LDPC_params.flags;
            desc.config.algo      = LDPC_params.algoIndex;
            desc.config.workspace = nullptr;
            // Set the normalization constant based on the code rate
            m_LDPCdecoder.set_normalization(desc.config);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Set up input and output addresses
        cuphyTransportBlockLLRDesc_t& llr_input = desc.llr_input[desc.num_tbs];
        llr_input.addr                          = static_cast<char*>(m_pHarqBuffers[ueIdx]);
        llr_input.stride_elements               = m_pTbPrmsCpu[ueIdx].Ncb_padded;
        llr_input.num_codewords                 = m_pTbPrmsCpu[ueIdx].num_CBs;

        cuphyTransportBlockDataDesc_t& tb_output = desc.tb_output[desc.num_tbs];
        tb_output.addr                           = static_cast<uint32_t*>(d_pLDPCOut) + oWords;
        tb_output.stride_words                   = OUT_STRIDE_WORDS;
        tb_output.num_codewords                  = m_pTbPrmsCpu[ueIdx].num_CBs;
        oWords += (OUT_STRIDE_WORDS * tb_output.num_codewords);

        //printf("prepareLDPCStreamsTB()[%i]: out: %p, in: %p, num_cw = %u\n",
        //       i,
        //       tb_output.addr,
        //       llr_input.addr,
        //       tb_output.num_codewords);

        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Increment the number of "used" TBs in the descriptor
        ++desc.num_tbs;
    }
}

////////////////////////////////////////////////////////////////////////
// PuschRx::launchLDPCStreamsTB()
void PuschRx::launchLDPCStreamsTB(cudaStream_t strm)
{
    const size_t DESC_COUNT = m_LDPCDecodeDescSet.count();
    if((m_LDPCkernelLaunchMode & PUSCH_RX_ENABLE_LDPC_DEC_SINGLE_STREAM_OPT) && (1 == DESC_COUNT))
    {
        // A single descriptor can be launched directly in the source stream
        m_LDPCdecoder.decode(m_LDPCDecodeDescSet[0], strm);
    }
    else
    {
        // Do LDPC for each TB in a different stream using a "round-robin"
        // distribution.
        m_ldpcStreamPool.fork(strm, std::min(DESC_COUNT, m_ldpcStreamPool.max_size()));
        for(int i = 0; i < DESC_COUNT; ++i)
        {
            m_LDPCdecoder.decode(m_LDPCDecodeDescSet[i],
                                 m_ldpcStreamPool.current_stream().handle());
            m_ldpcStreamPool.advance();
        }
        // Subsequent kernel submissions to strm must wait for all of
        // the LDPC kernels to complete
        m_ldpcStreamPool.join(strm);
    }
}

////////////////////////////////////////////////////////////////////////
// PuschRx::allocateUciBackendBuffers()
//Function allocates and links polar buffers for a single UCI segment

void PuschRx::allocAndLinkPolBuffers(cuphyPolarUciSegPrm_t&     uciSegPrms,
                                    const uint16_t&         polSegIdx,
                                    void*                   pSegLLRs,
                                    uint8_t*                pCrcStatus,
                                    uint32_t*               pUciSegEst,
                                    cuphyPolarCwPrm_t*      pUciCwPrmsCpu,
                                    std::vector<uint8_t*>&  cwTreeTypesAddrVec,
                                    std::vector<__half*>&   uciSegLLRsAddrVec,
                                    std::vector<__half*>&   cwTreeLLRsAddrVec,
                                    std::vector<__half*>&   cwLLRsAddrVec,
                                    std::vector<bool*>&     listPolScratchAddrVec,
                                    std::vector<uint32_t*>& cbEstAddrVec)
{
    uint16_t N_cw          = uciSegPrms.N_cw;
    uint16_t nBytes_cwTree = 2 * N_cw;

    cwTreeTypesAddrVec[polSegIdx] = static_cast<uint8_t*>(m_LinearAlloc.alloc(nBytes_cwTree));
    uciSegLLRsAddrVec[polSegIdx]  = static_cast<__half*>(pSegLLRs);
    uciSegPrms.pUciSegLLRs        = static_cast<__half*>(pSegLLRs);


    uint16_t nDecodedCbWords = div_round_up(static_cast<uint16_t>(uciSegPrms.K_cw - uciSegPrms.nCrcBits), static_cast<uint16_t>(32));

    for(int i = 0; i < uciSegPrms.nCbs; ++i)
    {
        uint8_t cwIdx = uciSegPrms.childCbIdxs[i];

        cwTreeLLRsAddrVec[cwIdx] = static_cast<__half*>(m_LinearAlloc.alloc(sizeof(__half) * (2 * N_cw) * m_polDcdrListSz));
        cwLLRsAddrVec[cwIdx]    = cwTreeLLRsAddrVec[cwIdx] + N_cw;

        pUciCwPrmsCpu[cwIdx].pCwTreeLLRs = cwTreeLLRsAddrVec[cwIdx];
        pUciCwPrmsCpu[cwIdx].pCwLLRs     = cwLLRsAddrVec[cwIdx];

        if (m_polDcdrListSz > 1) {
            listPolScratchAddrVec[cwIdx] = static_cast<bool*>(m_LinearAlloc.alloc(sizeof(bool) * (2 * N_cw * m_polDcdrListSz)));
        }
        pUciCwPrmsCpu[cwIdx].pCwTreeTypes = cwTreeTypesAddrVec[polSegIdx];
        pUciCwPrmsCpu[cwIdx].pCrcStatus   = pCrcStatus; //fix it only support 1Cb per Tb
        pUciCwPrmsCpu[cwIdx].en_CrcStatus = CUPHY_DET_EN;
        if(uciSegPrms.nCbs == 1)
        {
            cbEstAddrVec[cwIdx] = pUciSegEst;
            pUciCwPrmsCpu[cwIdx].pCbEst = pUciSegEst;
        }else
        {
            cbEstAddrVec[cwIdx] = static_cast<uint32_t*>(m_LinearAlloc.alloc(nDecodedCbWords));
            pUciCwPrmsCpu[cwIdx].pCbEst = cbEstAddrVec[cwIdx];
        }
        pUciCwPrmsCpu[cwIdx].pUciSegEst = pUciSegEst;
    }
}


////////////////////////////////////////////////////////////////////////
// PuschRx::allocateDeviceMemory()
//Function linear allocates device memory. Used to store intermediate results between componets, and output.

void PuschRx::allocateDeviceMemory(cuphyPuschDynPrms_t* pDynPrm)
{
    //------------------------------------------------------------------
    // Allocate front-end memory

    // mark the initial address and offset
    size_t initOffset = m_LinearAlloc.offset();
    void*  initAddr   = static_cast<char*>(m_LinearAlloc.address()) + initOffset;

    // Tensor allocations common across UE groups

    // Construct CFO/TA tensor dimensions and linear memory allocation
    m_tRefCfoPhaseRot.desc().set(CUPHY_C_32F, CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST, CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP, MAX_N_USER_GROUPS_SUPPORTED, cuphy::tensor_flags::align_tight);
    m_LinearAlloc.alloc(m_tRefCfoPhaseRot);

    m_tRefTaPhaseRot.desc().set(CUPHY_C_32F, CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP, m_cuphyPuschCellGrpDynPrm.nUeGrps, cuphy::tensor_flags::align_tight);
    m_LinearAlloc.alloc(m_tRefTaPhaseRot);

    m_tRefCfoHz.desc().set(CUPHY_R_32F, m_cuphyPuschCellGrpDynPrm.nUes, cuphy::tensor_flags::align_tight);
    m_LinearAlloc.alloc(m_tRefCfoHz);
    m_outputPrms.pCfoHzDevice = static_cast<float*>(m_tRefCfoHz.addr());

    m_tRefTaEst.desc().set(CUPHY_R_32F, m_cuphyPuschCellGrpDynPrm.nUes, cuphy::tensor_flags::align_tight);
    m_LinearAlloc.alloc(m_tRefTaEst);
    m_outputPrms.pTaEstsDevice = static_cast<float*>(m_tRefTaEst.addr());

    m_tRefCfoTaEstInterCtaSyncCnt.desc().set(CUPHY_R_32U, MAX_N_USER_GROUPS_SUPPORTED, cuphy::tensor_flags::align_tight);
    m_LinearAlloc.alloc(m_tRefCfoTaEstInterCtaSyncCnt);
#if CUPHY_ENABLE_SUB_SLOT_PROCESSING
    m_tRefCfoEstInterCtaSyncCnt.desc().set(CUPHY_R_32U, MAX_N_USER_GROUPS_SUPPORTED, cuphy::tensor_flags::align_tight);
    m_LinearAlloc.alloc(m_tRefCfoEstInterCtaSyncCnt);
    m_tRefTaEstInterCtaSyncCnt.desc().set(CUPHY_R_32U, MAX_N_USER_GROUPS_SUPPORTED, cuphy::tensor_flags::align_tight);
    m_LinearAlloc.alloc(m_tRefTaEstInterCtaSyncCnt);
#endif

    // Construct RSSI measurement tensor dimensions and linear memory allocation
    if(m_chEstSettings.enableRssiMeasurement)
    {
        m_tRefRssiFull.desc().set(CUPHY_R_32F, MAX_ND_SUPPORTED, MAX_N_ANTENNAS_SUPPORTED, m_cuphyPuschCellGrpDynPrm.nUeGrps, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefRssiFull);

        m_tRefRssi.desc().set(CUPHY_R_32F, m_cuphyPuschCellGrpDynPrm.nUeGrps, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefRssi);
        m_outputPrms.pRssiDevice = static_cast<float*>(m_tRefRssi.addr());

        m_tRefRssiInterCtaSyncCnt.desc().set(CUPHY_R_32U, m_cuphyPuschCellGrpDynPrm.nUeGrps, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefRssiInterCtaSyncCnt);
    }

    if(m_chEstSettings.enableSinrMeasurement || (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE)|| (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC))
    {
        // Construct noise variance measurement tensor dimensions and linear memory allocation
        // Note that these buffers are assumed to be zero initialized
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
        m_tRefNoiseVarPreEq.desc().set(CUPHY_R_32F, m_cuphyPuschCellGrpDynPrm.nUes, cuphy::tensor_flags::align_tight);
#else
        m_tRefNoiseVarPreEq.desc().set(CUPHY_R_32F, m_cuphyPuschCellGrpDynPrm.nUeGrps, cuphy::tensor_flags::align_tight);
#endif
        m_LinearAlloc.alloc(m_tRefNoiseVarPreEq);
        m_outputPrms.pNoiseVarPreEqDevice = static_cast<float*>(m_tRefNoiseVarPreEq.addr());

        m_tRefNoiseVarPostEq.desc().set(CUPHY_R_32F, m_cuphyPuschCellGrpDynPrm.nUes, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefNoiseVarPostEq);
        m_outputPrms.pNoiseVarPostEqDevice = static_cast<float*>(m_tRefNoiseVarPostEq.addr());

        m_tRefNoiseIntfEstInterCtaSyncCnt.desc().set(CUPHY_R_32U, m_cuphyPuschCellGrpDynPrm.nUeGrps, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefNoiseIntfEstInterCtaSyncCnt);
    }

    if(m_chEstSettings.enableSinrMeasurement)
    {
        // Construct RSRP and SINR measurement tensor dimensions and linear memory allocation
        m_tRefRsrp.desc().set(CUPHY_R_32F, m_cuphyPuschCellGrpDynPrm.nUes, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefRsrp);
        m_outputPrms.pRsrpDevice = static_cast<float*>(m_tRefRsrp.addr());

        m_tRefRsrpInterCtaSyncCnt.desc().set(CUPHY_R_32U, m_cuphyPuschCellGrpDynPrm.nUeGrps, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefRsrpInterCtaSyncCnt);

        m_tRefSinrPreEq.desc().set(CUPHY_R_32F, m_cuphyPuschCellGrpDynPrm.nUes, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefSinrPreEq);
        m_outputPrms.pSinrPreEqDevice = static_cast<float*>(m_tRefSinrPreEq.addr());

        m_tRefSinrPostEq.desc().set(CUPHY_R_32F, m_cuphyPuschCellGrpDynPrm.nUes, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefSinrPostEq);
        m_outputPrms.pSinrPostEqDevice = static_cast<float*>(m_tRefSinrPostEq.addr());
    }

    // mark the final offset and memset
    size_t finalOffset = m_LinearAlloc.offset();
    CUDA_CHECK(cudaMemsetAsync(initAddr, 0, finalOffset - initOffset, m_cuStream));

    // Per UE group tensor allocations
    int NUM_ANTENNAS, NUM_LAYERS, NF, NUM_DMRS_SYMS, NUM_DATA_SYMS, NH, NUM_PRBS;

    cuphyTensorPrm_t* pTDataRx   = pDynPrm->pDataIn->pTDataRx;
    for(int i = 0; i < m_cuphyPuschCellGrpDynPrm.nUeGrps; i++)
    {
        uint16_t cellPrmDynIdx = (m_cuphyPuschCellGrpDynPrm.pUeGrpPrms[i]).pCellPrm->cellPrmDynIdx;

        cuphyDataType_t cplxTypeCh = CUPHY_C_32F;
        cuphyDataType_t realTypeCh = scalar_type_from_complex_type(cplxTypeCh);

        cuphyPuschRxUeGrpPrms_t& drvdUeGrpPrmsCpu = m_drvdUeGrpPrmsCpu[i];

        NUM_ANTENNAS  = drvdUeGrpPrmsCpu.nRxAnt;
        NUM_LAYERS    = drvdUeGrpPrmsCpu.nLayers;
        NF            = CUPHY_N_TONES_PER_PRB * drvdUeGrpPrmsCpu.nPrb;
        NUM_DMRS_SYMS = drvdUeGrpPrmsCpu.nDmrsSyms;
        NUM_DATA_SYMS = drvdUeGrpPrmsCpu.nDataSym;
        NH            = m_drvdUeGrpPrmsCpu[i].dmrsAddlnPos + 1;
        NUM_PRBS      = drvdUeGrpPrmsCpu.nPrb;

        uint8_t nDmrsCdmGrpsNoData = m_cuphyPuschCellGrpDynPrm.pUeGrpPrms[i].pDmrsDynPrm->nDmrsCdmGrpsNoData;
        if(nDmrsCdmGrpsNoData==1)
        {
            NUM_DATA_SYMS += NUM_DMRS_SYMS;
        }

        // Slot buffer buffer allocated by L1 control
        copyTensorPrm2Info(pTDataRx[cellPrmDynIdx],
                           drvdUeGrpPrmsCpu.tInfoDataRx);

        // Construct Channel Estimation tensor dimensions and linear memory allocation
        m_tRefHEstVec[i].desc().set(cplxTypeCh, NUM_ANTENNAS, NUM_LAYERS, NF, NH, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefHEstVec[i]);
        copyTensorRef2Info(m_tRefHEstVec[i], drvdUeGrpPrmsCpu.tInfoHEst);

        if(m_chEstSettings.enableSinrMeasurement || (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE)|| (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC))
        {
            // Construct Channel Estimation tensor dimensions and linear memory allocation
            m_tRefLwInvVec[i].desc().set(cplxTypeCh, NUM_ANTENNAS, NUM_ANTENNAS, drvdUeGrpPrmsCpu.nPrb, cuphy::tensor_flags::align_tight);
            m_LinearAlloc.alloc(m_tRefLwInvVec[i]);
            copyTensorRef2Info(m_tRefLwInvVec[i], drvdUeGrpPrmsCpu.tInfoLwInv);
        }

        // Channel Estimation debug tensor dimensions and linear memory allocation
        if(m_outputPrms.debugOutputFlag)
        {
            m_tRefChEstDbgVec[i].desc().set(cplxTypeCh, NF / 2, NUM_DMRS_SYMS, 1, 1, cuphy::tensor_flags::align_tight);
            m_LinearAlloc.alloc(m_tRefChEstDbgVec[i]);
            copyTensorRef2Info(m_tRefChEstDbgVec[i], drvdUeGrpPrmsCpu.tInfoChEstDbg);
        }

        // Construct noise variance measurement tensor dimensions and linear memory allocation per PRB in sub-slot processing
#if CUPHY_ENABLE_SUB_SLOT_PROCESSING
        if(m_chEstSettings.enableSinrMeasurement || (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE)|| (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC))
        {
            m_tRefPerPrbNoiseVarVec[i].desc().set(CUPHY_R_32F, NUM_PRBS, cuphy::tensor_flags::align_tight);
            m_LinearAlloc.alloc(m_tRefPerPrbNoiseVarVec[i]);
            copyTensorRef2Info(m_tRefPerPrbNoiseVarVec[i], drvdUeGrpPrmsCpu.tInfoPerPrbNoiseVar);
        }
#endif

        // Construct CfoEst tensor dimensions and linear memory allocation
        m_tRefCfoEstVec[i].desc().set(cplxTypeCh, MAX_ND_SUPPORTED, drvdUeGrpPrmsCpu.nUes, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefCfoEstVec[i]);
        copyTensorRef2Info(m_tRefCfoEstVec[i], drvdUeGrpPrmsCpu.tInfoCfoEst);

        m_tRefCoefVec[i].desc().set(cplxTypeCh, NUM_ANTENNAS, CUPHY_N_TONES_PER_PRB, NUM_LAYERS, NUM_PRBS, NH, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefCoefVec[i]);
        copyTensorRef2Info(m_tRefCoefVec[i], drvdUeGrpPrmsCpu.tInfoEqCoef);

        // Construct ReeDiagInv (inverse of equalizer output error variance) tensor dimensions and linear memory allocation
        uint32_t nTimeChEq = m_cuphyPuschStatPrms.enablePuschTdi ? CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ : 1;
        m_tRefReeDiagInvVec[i].desc().set(realTypeCh, CUPHY_N_TONES_PER_PRB, NUM_LAYERS, NUM_PRBS, nTimeChEq, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefReeDiagInvVec[i]);
        copyTensorRef2Info(m_tRefReeDiagInvVec[i], drvdUeGrpPrmsCpu.tInfoReeDiagInv);

        // Construct Equalizer debug tensor dimensions and linear memory allocation
        if(m_outputPrms.debugOutputFlag)
        {
            m_tRefEqDbgVec[i].desc().set(cplxTypeCh, NUM_LAYERS + NUM_ANTENNAS, NUM_LAYERS, NF, NH, cuphy::tensor_flags::align_tight);
            m_LinearAlloc.alloc(m_tRefEqDbgVec[i]);
            copyTensorRef2Info(m_tRefEqDbgVec[i], drvdUeGrpPrmsCpu.tInfoChEqDbg);
        }

        // Construct estimated data (i.e. transmitted qams) tensor dimensions and linear memory allocation
        m_tRefDataEqVec[i].desc().set(CUPHY_C_16F, NUM_LAYERS, NF, NUM_DATA_SYMS, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefDataEqVec[i]);
        copyTensorRef2Info(m_tRefDataEqVec[i], drvdUeGrpPrmsCpu.tInfoDataEq);
        
        if(m_chEstSettings.enableDftSOfdm==1)
        {   
            m_tRefDataEqDftVec[i].desc().set(CUPHY_C_32F, NF*NUM_DATA_SYMS, cuphy::tensor_flags::align_tight);  //One UE per UEGRP, One layer per UE.
            m_LinearAlloc.alloc(m_tRefDataEqDftVec[i]);
            copyTensorRef2Info(m_tRefDataEqDftVec[i], drvdUeGrpPrmsCpu.tInfoDataEqDft);
        
            m_tRefDataEqDftIntermediateVec[i].desc().set(CUPHY_C_32F, FFT8192*NUM_DATA_SYMS, cuphy::tensor_flags::align_tight);  //One UE per UEGRP, One layer per UE.
            m_LinearAlloc.alloc(m_tRefDataEqDftIntermediateVec[i]);
            copyTensorRef2Info(m_tRefDataEqDftIntermediateVec[i], drvdUeGrpPrmsCpu.tInfoDataEqDftIntermediate);
            
            m_tRefDftBluesteinWorkspaceTimeVec[i].desc().set(CUPHY_C_32F, FFT8192, cuphy::tensor_flags::align_tight);  //One UE per UEGRP, One layer per UE.
            m_LinearAlloc.alloc(m_tRefDftBluesteinWorkspaceTimeVec[i]);
            copyTensorRef2Info(m_tRefDftBluesteinWorkspaceTimeVec[i], drvdUeGrpPrmsCpu.tInfoDftBluesteinWorkspaceTime);
            
            m_tRefDftBluesteinWorkspaceFreqVec[i].desc().set(CUPHY_C_32F, FFT8192, cuphy::tensor_flags::align_tight);  //One UE per UEGRP, One layer per UE.
            m_LinearAlloc.alloc(m_tRefDftBluesteinWorkspaceFreqVec[i]);
            copyTensorRef2Info(m_tRefDftBluesteinWorkspaceFreqVec[i], drvdUeGrpPrmsCpu.tInfoDftBluesteinWorkspaceFreq);
        }

// Construct LLR tensor dimensions and linear memory allocation
#ifdef LLR_FP16
        m_tRefLLRVec[i].desc().set(CUPHY_R_16F, CUPHY_QAM_256, NUM_LAYERS, NF, NUM_DATA_SYMS, cuphy::tensor_flags::align_tight);
        m_tRefLLRCdm1Vec[i].desc().set(CUPHY_R_16F, CUPHY_QAM_256, NUM_LAYERS, NF, NUM_DATA_SYMS, cuphy::tensor_flags::align_tight);
#else
        m_tRefLLRVec[i].desc().set(CUPHY_R_32F, CUPHY_QAM_256, NUM_LAYERS, NF, NUM_DATA_SYMS, cuphy::tensor_flags::align_tight);
        m_tRefLLRCdm1Vec[i].desc().set(CUPHY_R_32F, CUPHY_QAM_256, NUM_LAYERS, NF, NUM_DATA_SYMS, cuphy::tensor_flags::align_tight);
#endif
        m_LinearAlloc.alloc(m_tRefLLRVec[i]);
        copyTensorRef2Info(m_tRefLLRVec[i], drvdUeGrpPrmsCpu.tInfoLLR);

        m_LinearAlloc.alloc(m_tRefLLRCdm1Vec[i]);
        copyTensorRef2Info(m_tRefLLRCdm1Vec[i], drvdUeGrpPrmsCpu.tInfoLLRCdm1);

        // Note: these tensors are common across UE groups and could be placed in a new descriptor type that is common across UE groups

        // CFO/TA Estimation
        copyTensorRef2Info(m_tRefCfoPhaseRot, drvdUeGrpPrmsCpu.tInfoCfoPhaseRot);
        copyTensorRef2Info(m_tRefTaPhaseRot, drvdUeGrpPrmsCpu.tInfoTaPhaseRot);
        copyTensorRef2Info(m_tRefCfoHz, drvdUeGrpPrmsCpu.tInfoCfoHz);
        copyTensorRef2Info(m_tRefTaEst, drvdUeGrpPrmsCpu.tInfoTaEst);
        copyTensorRef2Info(m_tRefCfoTaEstInterCtaSyncCnt, drvdUeGrpPrmsCpu.tInfoCfoTaEstInterCtaSyncCnt);
#if CUPHY_ENABLE_SUB_SLOT_PROCESSING
        copyTensorRef2Info(m_tRefCfoEstInterCtaSyncCnt, drvdUeGrpPrmsCpu.tInfoCfoEstInterCtaSyncCnt);
        copyTensorRef2Info(m_tRefTaEstInterCtaSyncCnt, drvdUeGrpPrmsCpu.tInfoTaEstInterCtaSyncCnt);
#endif

        // RSSI measurement
        if(m_chEstSettings.enableRssiMeasurement)
        {
            copyTensorRef2Info(m_tRefRssiFull, drvdUeGrpPrmsCpu.tInfoRssiFull);
            copyTensorRef2Info(m_tRefRssi, drvdUeGrpPrmsCpu.tInfoRssi);
            copyTensorRef2Info(m_tRefRssiInterCtaSyncCnt, drvdUeGrpPrmsCpu.tInfoRssiInterCtaSyncCnt);
        }

        // Noise variance measurement
        if(m_chEstSettings.enableSinrMeasurement || (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE)|| (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC))
        {
            copyTensorRef2Info(m_tRefNoiseVarPreEq, drvdUeGrpPrmsCpu.tInfoNoiseVarPreEq);
            copyTensorRef2Info(m_tRefNoiseVarPostEq, drvdUeGrpPrmsCpu.tInfoNoiseVarPostEq);

            copyTensorRef2Info(m_tRefNoiseIntfEstInterCtaSyncCnt, drvdUeGrpPrmsCpu.tInfoNoiseIntfEstInterCtaSyncCnt);
        }

        // RSRP, SINR measurement
        if(m_chEstSettings.enableSinrMeasurement)
        {
            copyTensorRef2Info(m_tRefRsrp, drvdUeGrpPrmsCpu.tInfoRsrp);
            copyTensorRef2Info(m_tRefRsrpInterCtaSyncCnt, drvdUeGrpPrmsCpu.tInfoRsrpInterCtaSyncCnt);

            copyTensorRef2Info(m_tRefSinrPreEq, drvdUeGrpPrmsCpu.tInfoSinrPreEq);
            copyTensorRef2Info(m_tRefSinrPostEq, drvdUeGrpPrmsCpu.tInfoSinrPostEq);
        }

        // Convert from c++ to c
        m_tPrmLLRVec[i].desc  = m_tRefLLRVec[i].desc().handle();
        m_tPrmLLRVec[i].pAddr = m_tRefLLRVec[i].addr();

        m_tPrmLLRCdm1Vec[i].desc  = m_tRefLLRCdm1Vec[i].desc().handle();
        m_tPrmLLRCdm1Vec[i].pAddr = m_tRefLLRCdm1Vec[i].addr();

        /* Debug code to print out tensor dimensions
        printf("UeGrp[%d] NUM_ANTENNAS %d NUM_LAYERS %d NF %d NUM_DMRS_SYMS %d NUM_DATA_SYMS %d NH %d NUM_PRBS %d NUM_UE %d\n", i, NUM_ANTENNAS, NUM_LAYERS, NF, NUM_DMRS_SYMS, NUM_DATA_SYMS, NH, NUM_PRBS, drvdUeGrpPrmsCpu.nUes);
        const tensor_desc& tDataRxDesc = static_cast<const tensor_desc&>(*(pTDataRx[cellPrmDynIdx].desc));       
        const cuphy::tensor_info tDataRxInfo(tDataRxDesc.type(), cuphy::tensor_layout{tDataRxDesc.layout().rank(), tDataRxDesc.layout().dimensions.begin(),  tDataRxDesc.layout().strides.begin()});
        printf("UeGrp[%d]: DataRx %s\n", i, tDataRxInfo.to_string(false).c_str());                                

        const tensor_desc& tNoisePwrDesc = static_cast<const tensor_desc&>(*(pTNoisePwr[cellPrmDynIdx].desc));       
        const cuphy::tensor_info tNoisePwrInfo(tNoisePwrDesc.type(), cuphy::tensor_layout{tNoisePwrDesc.layout().rank(), tNoisePwrDesc.layout().dimensions.begin(),  tNoisePwrDesc.layout().strides.begin()});
        printf("UeGrp[%d]: NoisePwr %s\n", i, tNoisePwrInfo.to_string(false).c_str());                                

        printf("UeGrp[%d]: Coef %s\n"      , i, m_tRefCoefVec[i].desc().get_info().to_string(false).c_str());          
        printf("UeGrp[%d]: CfoEst %s\n"    , i, m_tRefCfoEstVec[i].desc().get_info().to_string(false).c_str());          
        printf("UeGrp[%d]: ReeDiagInv %s\n", i, m_tRefReeDiagInvVec[i].desc().get_info().to_string(false).c_str());          
        printf("UeGrp[%d]: DataEq %s\n"    , i, m_tRefDataEqVec[i].desc().get_info().to_string(false).c_str());          
        */
    }

    //-----------------------------------------------------------
    //Allocate output memory

    m_outputPrms.pTbCrcsDevice               = static_cast<uint32_t*>(m_LinearAlloc.alloc(m_outputPrms.totNumTbs * sizeof(uint32_t)));
    m_outputPrms.pCbCrcsDevice               = static_cast<uint32_t*>(m_LinearAlloc.alloc(m_outputPrms.totNumCbs * sizeof(uint32_t)));
    m_outputPrms.pTbPayloadsDevice           = static_cast<uint8_t*>(m_LinearAlloc.alloc(m_outputPrms.totNumPayloadBytes));

    m_outputPrms.pUciCrcFlagsDevice          = static_cast<uint8_t*>(m_LinearAlloc.alloc(m_outputPrms.totNumUciSegs));
    m_outputPrms.pUciCrcFlagsDevice_csi2     = static_cast<uint8_t*>(m_LinearAlloc.alloc(m_outputPrms.totNumUciSegs));
    m_outputPrms.pUciCrcFlagsDevice_early    = static_cast<uint8_t*>(m_LinearAlloc.alloc(m_outputPrms.totNumUciSegs));
    m_pPolCrcFlags                           = m_outputPrms.pUciCrcFlagsDevice;
    m_pPolCrcFlags_csi2                      = m_outputPrms.pUciCrcFlagsDevice_csi2;
    m_pPolCrcFlags_early                     = m_outputPrms.pUciCrcFlagsDevice_early;   

    void* uciPayloadsAddrVoid                = m_LinearAlloc.alloc(m_outputPrms.totNumUciPayloadBytes);
    m_outputPrms.pUciPayloadsDevice          = static_cast<uint8_t*>(uciPayloadsAddrVoid);
    CUDA_CHECK(cudaMemsetAsync(m_outputPrms.pUciPayloadsDevice, 0, m_outputPrms.totNumUciPayloadBytes, m_cuStream));

    m_outputPrms.pNumCsi2BitsDevice          = static_cast<uint16_t*>(m_LinearAlloc.alloc(m_nCsi2Ues * sizeof(uint16_t)));

    //--------------------------------------------------------------------
    // Allocate back-end memory

    uint32_t           nUes                = m_cuphyPuschCellGrpDynPrm.nUes;
    cuphyPuschUePrm_t* pUePrms             = m_cuphyPuschCellGrpDynPrm.pUePrms;
    auto               h2dBufGpuStartAddrs = m_h2dBuffer.getGpuStartAddrs();

    size_t        deRmLLRSize          = 0;
    size_t        NUM_BYTES_PER_LLR    = 2; // fp16 LLRs
    const int32_t OUT_STRIDE_WORDS     = (MAX_DECODED_CODE_BLOCK_BIT_SIZE + 31) / 32;
    const int32_t BYTES_PER_WORD       = 4;
    size_t        LDPCdecodeOutSize    = 0;
    uint16_t      spxCwIdx             = 0;
    uint16_t      rmCwIdx              = 0;
    uint16_t      polSegIdx            = 0;
    uint16_t      spxCwIdx_early       = 0;
    uint16_t      rmCwIdx_early        = 0;
    uint16_t      polSegIdx_early      = 0;
    uint32_t      currentUciWordOffset = 0;
    uint16_t      csi2Idx              = 0;

    m_nUes                           = nUes;
    void* uciDTXsAddrVoid            = m_LinearAlloc.alloc(cuphyUciDtxTypes_t::N_UCI_DTX*m_nUciUes);
    m_outputPrms.pUciDTXsDevice      = static_cast<uint8_t*>(uciDTXsAddrVoid);

    // mark the initial address and offset
    initOffset = m_LinearAlloc.offset();
    initAddr   = static_cast<char*>(m_LinearAlloc.address()) + initOffset;

    void* uciHarqDetectionStatusAddrVoid         = m_LinearAlloc.alloc(m_nUes);
    m_outputPrms.pHarqDetectionStatusDevice      = static_cast<uint8_t*>(uciHarqDetectionStatusAddrVoid);
    //CUDA_CHECK(cudaMemsetAsync(m_outputPrms.pHarqDetectionStatusDevice, 2, m_nUes, m_cuStream)); //no HARQ, default set to CRC Failure, SCF FAPIv10.04, Table 3–125

    void* uciCsiP1DetectionStatusAddrVoid         = m_LinearAlloc.alloc(m_nUes);
    m_outputPrms.pCsiP1DetectionStatusDevice      = static_cast<uint8_t*>(uciCsiP1DetectionStatusAddrVoid);
    //CUDA_CHECK(cudaMemsetAsync(m_outputPrms.pCsiP1DetectionStatusDevice, 2, m_nUes, m_cuStream)); //no CSI part 1, default set to CRC Failure, SCF FAPIv10.04, Table 3–125

    void* uciCsiP2DetectionStatusAddrVoid         = m_LinearAlloc.alloc(m_nUes);
    m_outputPrms.pCsiP2DetectionStatusDevice      = static_cast<uint8_t*>(uciCsiP2DetectionStatusAddrVoid);
    //CUDA_CHECK(cudaMemsetAsync(m_outputPrms.pCsiP2DetectionStatusDevice, 2, m_nUes, m_cuStream)); //no CSI part 2, default set to CRC Failure, SCF FAPIv10.04, Table 3–125

    finalOffset = m_LinearAlloc.offset();
    CUDA_CHECK(cudaMemsetAsync(initAddr, 2, finalOffset - initOffset, m_cuStream)); //default set to CRC Failure, SCF FAPIv10.04, Table 3–125

    uint16_t uciUeIdx = 0;

    for(uint32_t tbIdx = 0; tbIdx < nUes; ++tbIdx)
    {
        size_t cur_deRmLLRSize;
        cur_deRmLLRSize                = NUM_BYTES_PER_LLR * (m_pTbPrmsCpu[tbIdx].Ncb_padded) * m_pTbPrmsCpu[tbIdx].num_CBs;
        m_harqBufferSizeInBytes[tbIdx] = cur_deRmLLRSize;
        deRmLLRSize += cur_deRmLLRSize;
        LDPCdecodeOutSize += BYTES_PER_WORD * OUT_STRIDE_WORDS * (m_pTbPrmsCpu[tbIdx].num_CBs);

        if(m_pTbPrmsCpu[tbIdx].uciOnPuschFlag)
        {
            m_pTbPrmsCpu[tbIdx].d_schAndCsi2LLRs = static_cast<__half*>(m_LinearAlloc.alloc(m_pTbPrmsCpu[tbIdx].G * NUM_BYTES_PER_LLR));

            float DTXthreshold = pUePrms[tbIdx].pUciPrms->DTXthreshold;
            uint32_t nBitsHarq = pUePrms[tbIdx].pUciPrms->nBitsHarq;
            uint32_t G_harq    = m_pTbPrmsCpu[tbIdx].G_harq;
            void*    pHarqLLRs = m_LinearAlloc.alloc(G_harq * NUM_BYTES_PER_LLR);

            m_pTbPrmsCpu[tbIdx].d_harqLLrs = static_cast<__half*>(pHarqLLRs);
            uint8_t* d_HarqDetStatus = static_cast<uint8_t*>(uciHarqDetectionStatusAddrVoid) + m_outputPrms.pUciOnPuschOutOffsets[tbIdx].HarqDetectionStatusOffset;
            if(nBitsHarq > 0)
            {
                if(m_pTbPrmsCpu[tbIdx].isEarlyHarq)
                {
                    if(nBitsHarq <= CUPHY_N_MAX_UCI_BITS_SIMPLEX)
                    {
                        m_pSpxCwPrmsCpu_early[spxCwIdx_early].d_LLRs                                  = static_cast<__half*>(pHarqLLRs);
                        m_pSpxCwPrmsCpu_early[spxCwIdx_early].d_noiseVar                              = &(m_drvdUeGrpPrmsGpu[pUePrms[tbIdx].ueGrpIdx].noiseVarForDtx);//pUePrms[tbIdx].ueGrpIdx;
                        m_outputPrms.pUciOnPuschOutOffsets[tbIdx].harqPayloadByteOffset = currentUciWordOffset * sizeof(uint32_t);
                        m_pSpxCwPrmsCpu_early[spxCwIdx_early].d_cbEst                                 = static_cast<uint32_t*>(uciPayloadsAddrVoid) + currentUciWordOffset;
                        //m_pSpxCwPrmsCpu[spxCwIdx].d_DTXEst                                = static_cast<uint8_t*>(uciDTXsAddrVoid) + (cuphyUciDtxTypes_t::N_UCI_DTX*uciUeIdx + cuphyUciDtxTypes_t::UCI_HARQ_DTX);
                        m_pSpxCwPrmsCpu_early[spxCwIdx_early].d_DTXStatus                             = d_HarqDetStatus;
                        spxCwIdx_early += 1;
                        currentUciWordOffset += 1;
                    }
                    else if(nBitsHarq <= CUPHY_N_MAX_UCI_BITS_RM)
                    {
                        m_pRmCwPrmsCpu_early[rmCwIdx_early].d_LLRs                                   = static_cast<__half*>(pHarqLLRs);
                        m_pRmCwPrmsCpu_early[rmCwIdx_early].d_noiseVar                               = &(m_drvdUeGrpPrmsGpu[pUePrms[tbIdx].ueGrpIdx].noiseVarForDtx);//pUePrms[tbIdx].ueGrpIdx;
                        m_pRmCwPrmsCpu_early[rmCwIdx_early].Qm                                       = m_pTbPrmsCpu[tbIdx].Qm;
                        m_outputPrms.pUciOnPuschOutOffsets[tbIdx].harqPayloadByteOffset = currentUciWordOffset * sizeof(uint32_t);
                        m_pRmCwPrmsCpu_early[rmCwIdx_early].d_cbEst                                  = static_cast<uint32_t*>(uciPayloadsAddrVoid) + currentUciWordOffset;
                        //m_pRmCwPrmsCpu[rmCwIdx].d_DTXEst                                 = static_cast<uint8_t*>(uciDTXsAddrVoid) + (cuphyUciDtxTypes_t::N_UCI_DTX*uciUeIdx + cuphyUciDtxTypes_t::UCI_HARQ_DTX);
                        m_pRmCwPrmsCpu_early[rmCwIdx_early].d_DTXStatus                              = d_HarqDetStatus;
                        rmCwIdx_early += 1;
                        currentUciWordOffset += 1;
                    }
                    else
                    {
                        uint32_t* pUciSegEst                                            = static_cast<uint32_t*>(uciPayloadsAddrVoid) + currentUciWordOffset;
                        m_outputPrms.pUciOnPuschOutOffsets[tbIdx].harqPayloadByteOffset = currentUciWordOffset * sizeof(uint32_t);
                        m_outputPrms.pUciOnPuschOutOffsets[tbIdx].harqCrcFlagOffset     = polSegIdx_early;
                        cuphyPolarUciSegPrm_t& uciSegPrms                               = m_pUciSegPrmsCpu_early[polSegIdx_early];

                        allocAndLinkPolBuffers(uciSegPrms,
                                                polSegIdx_early,
                                                pHarqLLRs,
                                                d_HarqDetStatus,
                                                pUciSegEst,
                                                m_pUciCwPrmsCpu_early,
                                                m_cwTreeTypesAddrVec_early,
                                                m_uciSegLLRsAddrVec_early,
                                                m_cwTreeLLRsAddrVec_early,
                                                m_cwLLRsAddrVec_early,
                                                m_listPolScratchAddrVec_early,
                                                m_cbEstAddrVec_early);

                        uint16_t nUciSegBits   = uciSegPrms.nCbs * (uciSegPrms.K_cw - uciSegPrms.nCrcBits) - uciSegPrms.zeroInsertFlag;
                        uint16_t nUciSegWords  = div_round_up(nUciSegBits, static_cast<uint16_t>(32));
                        currentUciWordOffset  += nUciSegWords;
                        polSegIdx_early       += 1;
                    }
                }else
                {
                    if(nBitsHarq <= CUPHY_N_MAX_UCI_BITS_SIMPLEX)
                    {
                        m_pSpxCwPrmsCpu[spxCwIdx].d_LLRs                                  = static_cast<__half*>(pHarqLLRs);
                        m_pSpxCwPrmsCpu[spxCwIdx].d_noiseVar                              = &(m_drvdUeGrpPrmsGpu[pUePrms[tbIdx].ueGrpIdx].noiseVarForDtx);//pUePrms[tbIdx].ueGrpIdx;
                        m_outputPrms.pUciOnPuschOutOffsets[tbIdx].harqPayloadByteOffset = currentUciWordOffset * sizeof(uint32_t);
                        m_pSpxCwPrmsCpu[spxCwIdx].d_cbEst                                 = static_cast<uint32_t*>(uciPayloadsAddrVoid) + currentUciWordOffset;
                        //m_pSpxCwPrmsCpu[spxCwIdx].d_DTXEst                                = static_cast<uint8_t*>(uciDTXsAddrVoid) + (cuphyUciDtxTypes_t::N_UCI_DTX*uciUeIdx + cuphyUciDtxTypes_t::UCI_HARQ_DTX);
                        m_pSpxCwPrmsCpu[spxCwIdx].d_DTXStatus                             = d_HarqDetStatus;
                        spxCwIdx += 1;
                        currentUciWordOffset += 1;
                    }
                    else if(nBitsHarq <= CUPHY_N_MAX_UCI_BITS_RM)
                    {
                        m_pRmCwPrmsCpu[rmCwIdx].d_LLRs                                   = static_cast<__half*>(pHarqLLRs);
                        m_pRmCwPrmsCpu[rmCwIdx].d_noiseVar                               = &(m_drvdUeGrpPrmsGpu[pUePrms[tbIdx].ueGrpIdx].noiseVarForDtx);//pUePrms[tbIdx].ueGrpIdx;
                        m_pRmCwPrmsCpu[rmCwIdx].Qm                                       = m_pTbPrmsCpu[tbIdx].Qm;
                        m_outputPrms.pUciOnPuschOutOffsets[tbIdx].harqPayloadByteOffset = currentUciWordOffset * sizeof(uint32_t);
                        m_pRmCwPrmsCpu[rmCwIdx].d_cbEst                                  = static_cast<uint32_t*>(uciPayloadsAddrVoid) + currentUciWordOffset;
                        //m_pRmCwPrmsCpu[rmCwIdx].d_DTXEst                                 = static_cast<uint8_t*>(uciDTXsAddrVoid) + (cuphyUciDtxTypes_t::N_UCI_DTX*uciUeIdx + cuphyUciDtxTypes_t::UCI_HARQ_DTX);
                        m_pRmCwPrmsCpu[rmCwIdx].d_DTXStatus                              = d_HarqDetStatus;
                        rmCwIdx += 1;
                        currentUciWordOffset += 1;
                    }
                    else
                    {
                        uint32_t* pUciSegEst                                            = static_cast<uint32_t*>(uciPayloadsAddrVoid) + currentUciWordOffset;
                        m_outputPrms.pUciOnPuschOutOffsets[tbIdx].harqPayloadByteOffset = currentUciWordOffset * sizeof(uint32_t);
                        m_outputPrms.pUciOnPuschOutOffsets[tbIdx].harqCrcFlagOffset     = polSegIdx;
                        cuphyPolarUciSegPrm_t& uciSegPrms                               = m_pUciSegPrmsCpu[polSegIdx];

                        allocAndLinkPolBuffers(uciSegPrms,
                                            polSegIdx,
                                            pHarqLLRs,
                                            d_HarqDetStatus,
                                            pUciSegEst,
                                            m_pUciCwPrmsCpu,
                                            m_cwTreeTypesAddrVec,
                                            m_uciSegLLRsAddrVec,
                                            m_cwTreeLLRsAddrVec,
                                            m_cwLLRsAddrVec,
                                            m_listPolScratchAddrVec,
                                            m_cbEstAddrVec);

                        uint16_t nUciSegBits   = uciSegPrms.nCbs * (uciSegPrms.K_cw - uciSegPrms.nCrcBits) - uciSegPrms.zeroInsertFlag;
                        uint16_t nUciSegWords  = div_round_up(nUciSegBits, static_cast<uint16_t>(32));
                        currentUciWordOffset  += nUciSegWords;
                        polSegIdx             += 1;
                    }
                }
                //m_outputPrms.pUciOnPuschOutOffsets[tbIdx].harqDtxFlagOffset = (cuphyUciDtxTypes_t::N_UCI_DTX*tbIdx+cuphyUciDtxTypes_t::UCI_HARQ_DTX);
            }
            // else
            // {
            //     CUDA_CHECK(cudaMemsetAsync(d_HarqDTXStatus, 2, 1, cuStream)); //no HARQ, default set to CRC Failure, SCF FAPIv10.04, Table 3–125
            // }

            uint32_t nBitsCsi1                 = pUePrms[tbIdx].pUciPrms->nBitsCsi1;
            uint32_t G_csi1                    = m_pTbPrmsCpu[tbIdx].G_csi1;
            void*    pCsi1LLRs                 = m_LinearAlloc.alloc(G_csi1 * NUM_BYTES_PER_LLR);
            m_pTbPrmsCpu[tbIdx].d_csi1LLRs = static_cast<__half*>(pCsi1LLRs);
            uint8_t* d_CsiP1DetStatus = static_cast<uint8_t*>(uciCsiP1DetectionStatusAddrVoid) + m_outputPrms.pUciOnPuschOutOffsets[tbIdx].CsiP1DetectionStatusOffset;
            if(nBitsCsi1 > 0)
            {
                if(nBitsCsi1 <= CUPHY_N_MAX_UCI_BITS_SIMPLEX)
                {
                    m_pSpxCwPrmsCpu[spxCwIdx].d_LLRs                                  = static_cast<__half*>(pCsi1LLRs);
                    m_pSpxCwPrmsCpu[spxCwIdx].d_noiseVar                              = &(m_drvdUeGrpPrmsGpu[pUePrms[tbIdx].ueGrpIdx].noiseVarForDtx);//pUePrms[tbIdx].ueGrpIdx;
                    m_outputPrms.pUciOnPuschOutOffsets[tbIdx].csi1PayloadByteOffset = currentUciWordOffset * sizeof(uint32_t);
                    m_pSpxCwPrmsCpu[spxCwIdx].d_cbEst                                 = static_cast<uint32_t*>(uciPayloadsAddrVoid) + currentUciWordOffset;
                    //m_pSpxCwPrmsCpu[spxCwIdx].d_DTXEst                                = static_cast<uint8_t*>(uciDTXsAddrVoid) + (cuphyUciDtxTypes_t::N_UCI_DTX*uciUeIdx + cuphyUciDtxTypes_t::UCI_CSI1_DTX);
                    m_pSpxCwPrmsCpu[spxCwIdx].d_DTXStatus                             = d_CsiP1DetStatus;
                    spxCwIdx += 1;
                    currentUciWordOffset += 1;
                }
                else if(nBitsCsi1 <= CUPHY_N_MAX_UCI_BITS_RM)
                {
                    m_pRmCwPrmsCpu[rmCwIdx].d_LLRs                                   = static_cast<__half*>(pCsi1LLRs);
                    m_pRmCwPrmsCpu[rmCwIdx].d_noiseVar                               = &(m_drvdUeGrpPrmsGpu[pUePrms[tbIdx].ueGrpIdx].noiseVarForDtx);//pUePrms[tbIdx].ueGrpIdx;
                    m_pRmCwPrmsCpu[rmCwIdx].Qm                                       = m_pTbPrmsCpu[tbIdx].Qm;
                    m_outputPrms.pUciOnPuschOutOffsets[tbIdx].csi1PayloadByteOffset = currentUciWordOffset * sizeof(uint32_t);
                    m_pRmCwPrmsCpu[rmCwIdx].d_cbEst                                  = static_cast<uint32_t*>(uciPayloadsAddrVoid) + currentUciWordOffset;
                    //m_pRmCwPrmsCpu[rmCwIdx].d_DTXEst                                 = static_cast<uint8_t*>(uciDTXsAddrVoid) + (cuphyUciDtxTypes_t::N_UCI_DTX*uciUeIdx + cuphyUciDtxTypes_t::UCI_CSI1_DTX);
                    m_pRmCwPrmsCpu[rmCwIdx].d_DTXStatus                              = d_CsiP1DetStatus;
                    rmCwIdx += 1;
                    currentUciWordOffset += 1;
                }
                else
                {
                    uint32_t* pUciSegEst                                            = static_cast<uint32_t*>(uciPayloadsAddrVoid) + currentUciWordOffset;
                    m_outputPrms.pUciOnPuschOutOffsets[tbIdx].csi1PayloadByteOffset = currentUciWordOffset * sizeof(uint32_t);
                    m_outputPrms.pUciOnPuschOutOffsets[tbIdx].csi1CrcFlagOffset     = polSegIdx;
                    cuphyPolarUciSegPrm_t& uciSegPrms                               = m_pUciSegPrmsCpu[polSegIdx];

                    allocAndLinkPolBuffers(uciSegPrms,
                                           polSegIdx,
                                           pCsi1LLRs,
                                           d_CsiP1DetStatus,
                                           pUciSegEst,
                                           m_pUciCwPrmsCpu,
                                           m_cwTreeTypesAddrVec,
                                           m_uciSegLLRsAddrVec,
                                           m_cwTreeLLRsAddrVec,
                                           m_cwLLRsAddrVec,
                                           m_listPolScratchAddrVec,
                                           m_cbEstAddrVec);

                    uint16_t nUciSegBits   = uciSegPrms.nCbs * (uciSegPrms.K_cw - uciSegPrms.nCrcBits) - uciSegPrms.zeroInsertFlag;
                    uint16_t nUciSegWords  = div_round_up(nUciSegBits, static_cast<uint16_t>(32));
                    currentUciWordOffset  += nUciSegWords;
                    polSegIdx             += 1;
                }
                //m_outputPrms.pUciOnPuschOutOffsets[tbIdx].csi1DtxFlagOffset = (cuphyUciDtxTypes_t::N_UCI_DTX*tbIdx+cuphyUciDtxTypes_t::UCI_CSI1_DTX);
            }
            // else
            // {
            //     CUDA_CHECK(cudaMemsetAsync(d_CsiP1DTXStatus, 2, 1, cuStream)); //no CSI part 1, default set to CRC Failure, SCF FAPIv10.04, Table 3–125
            // }

            uint8_t* d_CsiP2DetStatus = static_cast<uint8_t*>(uciCsiP2DetectionStatusAddrVoid) + m_outputPrms.pUciOnPuschOutOffsets[tbIdx].CsiP2DetectionStatusOffset;
            if((pUePrms[tbIdx].pduBitmap >> 5) & 1) // Check for CSI-P2
            {
                m_outputPrms.pUciOnPuschOutOffsets[tbIdx].csi2PayloadByteOffset = currentUciWordOffset * sizeof(uint32_t);
                m_outputPrms.pUciOnPuschOutOffsets[tbIdx].numCsi2BitsOffset     = csi2Idx;

                m_pRmCwPrmsCpu_csi2[csi2Idx].d_noiseVar   = &(m_drvdUeGrpPrmsGpu[pUePrms[tbIdx].ueGrpIdx].noiseVarForDtx);//pUePrms[tbIdx].ueGrpIdx;
                m_pRmCwPrmsCpu_csi2[csi2Idx].Qm           = m_pTbPrmsCpu[tbIdx].Qm;

                m_pSpxCwPrmsCpu_csi2[csi2Idx].d_cbEst      = static_cast<uint32_t*>(uciPayloadsAddrVoid) + currentUciWordOffset;
                m_pRmCwPrmsCpu_csi2[csi2Idx].d_cbEst      = static_cast<uint32_t*>(uciPayloadsAddrVoid) + currentUciWordOffset;
                //m_pSpxCwPrmsCpu_csi2[csi2Idx].d_DTXEst     = static_cast<uint8_t*>(uciDTXsAddrVoid) + (cuphyUciDtxTypes_t::N_UCI_DTX*uciUeIdx + cuphyUciDtxTypes_t::UCI_CSI2_DTX);
                //m_pRmCwPrmsCpu_csi2[csi2Idx].d_DTXEst     = static_cast<uint8_t*>(uciDTXsAddrVoid) + (cuphyUciDtxTypes_t::N_UCI_DTX*uciUeIdx + cuphyUciDtxTypes_t::UCI_CSI2_DTX);
                m_pSpxCwPrmsCpu_csi2[csi2Idx].d_DTXStatus  = d_CsiP2DetStatus;
                m_pSpxCwPrmsCpu_csi2[csi2Idx].d_noiseVar   = &(m_drvdUeGrpPrmsGpu[pUePrms[tbIdx].ueGrpIdx].noiseVarForDtx);
                m_pRmCwPrmsCpu_csi2[csi2Idx].d_DTXStatus  = d_CsiP2DetStatus;
                m_pSpxCwPrmsCpu_csi2[csi2Idx].en_DTXest = CUPHY_DTX_EN + CUPHY_DET_EN;
                m_pRmCwPrmsCpu_csi2[csi2Idx].en_DTXest = CUPHY_DTX_EN + CUPHY_DET_EN;
                m_pRmCwPrmsCpu_csi2[csi2Idx].DTXthreshold = DTXthreshold;

                if(m_cuphyPuschStatPrms.pDbg->forcedNumCsi2Bits > 0)
                {
                    m_pUciSegPrmsCpu_csi2[csi2Idx].N_cw           = CUPHY_POLAR_DECODER_MAX_BITS;
                    m_pUciSegPrmsCpu_csi2[csi2Idx].K_cw           = CUPHY_POLAR_DECODER_MAX_BITS;
                    m_pUciSegPrmsCpu_csi2[csi2Idx].nCrcBits       = 0;
                    m_pUciSegPrmsCpu_csi2[csi2Idx].nCbs           = 2;                                                                             
                    m_pUciSegPrmsCpu_csi2[csi2Idx].childCbIdxs[0] = 2*csi2Idx;
                    m_pUciSegPrmsCpu_csi2[csi2Idx].childCbIdxs[1] = 2*csi2Idx + 1;
                    m_pUciSegPrmsCpu_csi2[csi2Idx].exitFlag       = 1;

                    m_pUciCwPrmsCpu_csi2[2*csi2Idx].cbIdxWithinUciSeg = 0;
                    m_pUciCwPrmsCpu_csi2[2*csi2Idx].pCrcStatus        = d_CsiP2DetStatus; //fix it only support 1Cb per Tb
                    m_pUciCwPrmsCpu_csi2[2*csi2Idx].en_CrcStatus      = CUPHY_DET_EN;
                    m_pUciCwPrmsCpu_csi2[2*csi2Idx].exitFlag          = 1;

                    m_pUciCwPrmsCpu_csi2[2*csi2Idx + 1].cbIdxWithinUciSeg = 1;
                    m_pUciCwPrmsCpu_csi2[2*csi2Idx + 1].pCrcStatus        = d_CsiP2DetStatus;
                    m_pUciCwPrmsCpu_csi2[2*csi2Idx + 1].en_CrcStatus      = CUPHY_DET_EN;
                    m_pUciCwPrmsCpu_csi2[2*csi2Idx + 1].exitFlag          = 1;

                    uint32_t* pUciSegCsi2Est = static_cast<uint32_t*>(uciPayloadsAddrVoid) + currentUciWordOffset;
                    void*     pUciSegCsi2LLRs = NULL; // pointer will be populated during CSI2 control kernel

                    allocAndLinkPolBuffers( m_pUciSegPrmsCpu_csi2[csi2Idx],
                                            csi2Idx,
                                            pUciSegCsi2LLRs,
                                            d_CsiP2DetStatus,
                                            pUciSegCsi2Est,
                                            m_pUciCwPrmsCpu_csi2,
                                            m_cwTreeTypesAddrVec_csi2,
                                            m_uciSegLLRsAddrVec_csi2,
                                            m_cwTreeLLRsAddrVec_csi2,
                                            m_cwLLRsAddrVec_csi2,
                                            m_listPolScratchAddrVec_csi2,
                                            m_cbEstAddrVec_csi2);
                }
                //m_outputPrms.pUciOnPuschOutOffsets[tbIdx].csi2DtxFlagOffset = (cuphyUciDtxTypes_t::N_UCI_DTX*tbIdx+cuphyUciDtxTypes_t::UCI_CSI2_DTX);

                currentUciWordOffset += CUPHY_MAX_N_CSI2_WORDS;
                csi2Idx += 1;
            }
            // else
            // {
            //     CUDA_CHECK(cudaMemsetAsync(d_CsiP2DTXStatus, 2, 1, cuStream)); ////no CSI part 2, default set to CRC Failure, SCF FAPIv10.04, Table 3–125
            // }
            uciUeIdx += 1;
        }
    }


    if(m_nSpxCws > 0)
    {
        m_pSpxPrmsGpu = reinterpret_cast<cuphySimplexCwPrm_t*>(h2dBufGpuStartAddrs[PUSCH_SPX_PRMS]);
    }

    if(m_nSpxCws_early > 0)
    {
        m_pSpxPrmsGpu_early = reinterpret_cast<cuphySimplexCwPrm_t*>(h2dBufGpuStartAddrs[PUSCH_SPX_EARLY_PRMS]);
    }

    if(m_nRmCws > 0)
    {
        m_pRmCwPrmsGpu = reinterpret_cast<cuphyRmCwPrm_t*>(h2dBufGpuStartAddrs[PUSCH_RM_CW_PRMS]);
    }

    if(m_nRmCws_early > 0)
    {
        m_pRmCwPrmsGpu_early = reinterpret_cast<cuphyRmCwPrm_t*>(h2dBufGpuStartAddrs[PUSCH_RM_CW_EARLY_PRMS]);
    }

    if(m_nPolUciSegs > 0)
    {
        m_pUciSegPrmsGpu = reinterpret_cast<cuphyPolarUciSegPrm_t*>(h2dBufGpuStartAddrs[PUSCH_UCI_SEG_PRMS]);
        m_pUciCwPrmsGpu  = reinterpret_cast<cuphyPolarCwPrm_t*>(h2dBufGpuStartAddrs[PUSCH_UCI_CW_PRMS]);
    }

    if(m_nPolUciSegs_early > 0)
    {
        m_pUciSegPrmsGpu_early = reinterpret_cast<cuphyPolarUciSegPrm_t*>(h2dBufGpuStartAddrs[PUSCH_UCI_SEG_EARLY_PRMS]);
        m_pUciCwPrmsGpu_early  = reinterpret_cast<cuphyPolarCwPrm_t*>(h2dBufGpuStartAddrs[PUSCH_UCI_CW_EARLY_PRMS]);
    }

    if(m_nCsi2Ues > 0)
    {
        m_pSpxPrmsGpu_csi2 = reinterpret_cast<cuphySimplexCwPrm_t*>(h2dBufGpuStartAddrs[PUSCH_SPX_CSI2_PRMS]);
        m_pRmCwPrmsGpu_csi2 = reinterpret_cast<cuphyRmCwPrm_t*>(h2dBufGpuStartAddrs[PUSCH_RM_CW_CSI2_PRMS]);
        m_pUciSegPrmsGpu_csi2 = reinterpret_cast<cuphyPolarUciSegPrm_t*>(h2dBufGpuStartAddrs[PUSCH_UCI_SEG_CSI2_PRMS]);
        m_pUciCwPrmsGpu_csi2 = reinterpret_cast<cuphyPolarCwPrm_t*>(h2dBufGpuStartAddrs[PUSCH_UCI_CW_CSI2_PRMS]);
    }

    m_pTbPrmsGpu = reinterpret_cast<PerTbParams*>(h2dBufGpuStartAddrs[PUSCH_TB_PRMS]);

    d_pLDPCOut = m_LinearAlloc.alloc(LDPCdecodeOutSize);
}

// // ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// // // cuphyCreatePuschRx()

cuphyStatus_t CUPHYWINAPI cuphyCreatePuschRx(cuphyPuschRxHndl_t* pPuschRxHndl, cuphyPuschStatPrms_t const* pStatPrms, cudaStream_t cuStream)
{
    if(!pPuschRxHndl || !pStatPrms || !cuStream)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pPuschRxHndl = nullptr;

    return cuphy::tryCallableAndCatch([&]
    {
        if (pStatPrms->pDbg->enableApiLogging) {
            PuschRx::printStaticApiPrms(pStatPrms);
        }
        PuschRx* p    = new PuschRx(pStatPrms, cuStream);
        *pPuschRxHndl = static_cast<cuphyPuschRxHndl_t>(p);
    });
}

#if 0
const void* cuphyGetMemoryFootprintTrackerPuschRx(cuphyPuschRxHndl_t puschRxHndl)
{
    if(puschRxHndl == nullptr)
    {
        return nullptr;
    }
    PuschRx* pipeline_ptr  = static_cast<PuschRx*>(puschRxHndl);
    return pipeline_ptr->getMemoryTracker();
}
#endif

const void* PuschRx::getMemoryTracker()
{
    return &m_memoryFootprint;
}


////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cuphySetupPuschRx()

cuphyStatus_t CUPHYWINAPI cuphySetupPuschRx(cuphyPuschRxHndl_t puschRxHndl, cuphyPuschDynPrms_t* pDynPrms, cuphyPuschBatchPrmHndl_t const batchPrmHndl)
{
    MemtraceDisableScope md; // Disable temporarity GT-7257
    if(!puschRxHndl || !pDynPrms)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
       if (pDynPrms->pDbg->enableApiLogging) {
        PuschRx::printDynApiPrms(pDynPrms);
       }

       PuschRx* p = static_cast<PuschRx*>(puschRxHndl);
       cuphyStatus_t status = p->setup(pDynPrms);
       if(0)
       {
            PuschRx::printDynApiPrms<fmtlog::ERR>(pDynPrms);
       }
       return status;
    });
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cuphyRunPuschRx()

cuphyStatus_t CUPHYWINAPI cuphyRunPuschRx(cuphyPuschRxHndl_t puschRxHndl, cuphyPuschRunPhase_t runPhase)
{
    MemtraceDisableScope md; // Disable temporarity GT-7257
    if(!puschRxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
        PuschRx* p = static_cast<PuschRx*>(puschRxHndl);
        p->run(runPhase);
    });
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cuphyWriteDbgBufSynch()
cuphyStatus_t CUPHYWINAPI cuphyWriteDbgBufSynch(cuphyPuschRxHndl_t puschRxHndl, cudaStream_t cuStream)
{
    if(!puschRxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
        PuschRx* p = static_cast<PuschRx*>(puschRxHndl);
        p->writeDbgBufSynch(cuStream);
        p->copyOutputToCPU(cuStream);
        //p->printInfo();
    });
}

//////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cuphyDestroyPuschRx()
cuphyStatus_t CUPHYWINAPI cuphyDestroyPuschRx(cuphyPuschRxHndl_t puschRxHndl)
{
    if(!puschRxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    PuschRx* p = static_cast<PuschRx*>(puschRxHndl);
    delete p;
    return CUPHY_STATUS_SUCCESS;
}

///////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// getBufferSize()
size_t PuschRx::getBufferSize(cuphyPuschStatPrms_t const* pStatPrms)
{
    // data type sizes
    static constexpr uint32_t N_BYTES_C16        = sizeof(data_type_traits<CUPHY_C_16F>::type);
    static constexpr uint32_t N_BYTES_R16        = sizeof(data_type_traits<CUPHY_R_16F>::type);
    static constexpr uint32_t N_BYTES_R32        = sizeof(data_type_traits<CUPHY_R_32F>::type);
    static constexpr uint32_t N_BYTES_C32        = sizeof(data_type_traits<CUPHY_C_32F>::type);
    static constexpr uint32_t N_BYTES_PER_UINT32 = 4;
    static constexpr uint32_t MAX_N_UE           = MAX_N_TBS_SUPPORTED; // 1UE per TB for PUSCH

    //Find the max UL BWP and max layers across all cells
    uint32_t max_nPrbUlBwp = pStatPrms->nMaxPrb;
    uint32_t max_N_RX      = pStatPrms->nMaxRx;

    // if max parameters are zero set to max supported
    if(max_nPrbUlBwp == 0)
        max_nPrbUlBwp = MAX_N_PRBS_SUPPORTED;
    if(max_N_RX == 0)
        max_N_RX = MAX_N_ANTENNAS_SUPPORTED;

    if(max_nPrbUlBwp > MAX_N_PRBS_SUPPORTED)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "nMaxPrb provided {} is larger than supported max {}", max_nPrbUlBwp, MAX_N_PRBS_SUPPORTED);
        std::string err("PUSCH: nMaxPrb provided (" + std::to_string(max_nPrbUlBwp) + ") is larger than supported max (" + std::to_string(MAX_N_PRBS_SUPPORTED) + ")");
        throw std::out_of_range(err);
    }

    if(max_N_RX > MAX_N_ANTENNAS_SUPPORTED)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "nMaxRx provided {} is larger than supported max {}", max_N_RX, MAX_N_ANTENNAS_SUPPORTED);
        std::string err("PUSCH: nMaxRx provided (" + std::to_string(max_N_RX) + ") is larger than supported max (" + std::to_string(MAX_N_ANTENNAS_SUPPORTED) + ")");
        throw std::out_of_range(err);
    }

    // compute linear buffer size
    size_t        nBytesBuffer          = 0;
    uint32_t      NF                    = max_nPrbUlBwp * CUPHY_N_TONES_PER_PRB;
    uint32_t      N_RX                  = max_N_RX;
    uint32_t      N_MAX_LAYERS          = N_RX;
    uint32_t      MAX_N_UE_PER_UE_GROUP = N_MAX_LAYERS;
    const int32_t OUT_STRIDE_WORDS      = (MAX_DECODED_CODE_BLOCK_BIT_SIZE + 31) / 32;
    const int32_t BYTES_PER_WORD        = 4;
    const int32_t EXTRA_PADDING         = MAX_N_USER_GROUPS_SUPPORTED * LINEAR_ALLOC_PAD_BYTES; // upper bound for extra memory required per allocation

    // max channel estimation buffer
    uint32_t maxBytesChEst = N_BYTES_C32 * N_RX * N_RX * NF * CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST;
    nBytesBuffer += maxBytesChEst + EXTRA_PADDING;

    // max channel estimation debug buffer
    if(pStatPrms->pDbg != nullptr)
    {
        if(pStatPrms->pDbg->pOutFileName != nullptr)
        {
            uint32_t maxBytesChEstDbg = N_BYTES_C32 * (NF / 2) * MAX_N_DMRSSYMS_SUPPORTED;
            nBytesBuffer += CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST * maxBytesChEstDbg + EXTRA_PADDING;
        }
    }

    uint32_t nTimeChEq = pStatPrms->enablePuschTdi ? CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ : 1;

    // max equalizer coefficent buffer
    uint32_t maxBytesEqualizer = N_BYTES_C32 * N_RX * N_RX * NF * nTimeChEq;
    nBytesBuffer += maxBytesEqualizer + EXTRA_PADDING;

    // max ReeDiagInv buffer (equalizer preceision)
    uint32_t maxBytesPrecesion = N_BYTES_R32 * N_RX * NF * nTimeChEq;
    nBytesBuffer += maxBytesPrecesion + EXTRA_PADDING;

    // max equalizer debug buffer
    if(pStatPrms->pDbg != nullptr)
    {
        if(pStatPrms->pDbg->pOutFileName != nullptr)
        {
            uint32_t maxBytesEqualizerDbg = N_BYTES_C32 * (2 * N_RX) * N_RX * NF * (OFDM_SYMBOLS_PER_SLOT - 1);
            nBytesBuffer += maxBytesEqualizerDbg + EXTRA_PADDING;
        }
    }

    // max CFO/TA est buffer
    uint32_t maxBytesCfoEst = N_BYTES_C32 * MAX_ND_SUPPORTED * CUPHY_PUSCH_RX_MAX_N_LAYERS_PER_UE_GROUP;
    nBytesBuffer += maxBytesCfoEst + EXTRA_PADDING;

    uint32_t maxBytesCfoPhaseRot = N_BYTES_C32 * MAX_N_UE_PER_UE_GROUP * MAX_N_USER_GROUPS_SUPPORTED;
    nBytesBuffer += maxBytesCfoPhaseRot + EXTRA_PADDING;

    uint32_t maxBytesTaEst = N_BYTES_R32 * MAX_N_UE_PER_UE_GROUP * MAX_N_USER_GROUPS_SUPPORTED;
    nBytesBuffer += maxBytesTaEst + EXTRA_PADDING;

    uint32_t maxBytesTaPhaseRot = N_BYTES_C32 * MAX_N_UE_PER_UE_GROUP * MAX_N_USER_GROUPS_SUPPORTED;
    nBytesBuffer += maxBytesTaPhaseRot + EXTRA_PADDING;

    uint32_t maxBytesCfoTaEstInterCtaSyncCnt = N_BYTES_PER_UINT32 * MAX_N_USER_GROUPS_SUPPORTED;
    nBytesBuffer += maxBytesCfoTaEstInterCtaSyncCnt + EXTRA_PADDING;

    nBytesBuffer += N_BYTES_R32 * MAX_N_UE;  //tCfoHz

    // max estiamted data buffer
    uint32_t maxBytesEstimatedData = N_BYTES_C16 * N_RX * (OFDM_SYMBOLS_PER_SLOT - 1);
    nBytesBuffer += maxBytesEstimatedData + EXTRA_PADDING;
    
    // max DFT data buffer
    if(m_chEstSettings.enableDftSOfdm==1)
    {
        nBytesBuffer += (N_BYTES_C32 * MAX_N_USER_GROUPS_SUPPORTED * 3276 * (OFDM_SYMBOLS_PER_SLOT - 1) + EXTRA_PADDING);
        nBytesBuffer += (N_BYTES_C32 * MAX_N_USER_GROUPS_SUPPORTED * FFT8192 * (OFDM_SYMBOLS_PER_SLOT - 1) + EXTRA_PADDING); //for intermediate results in Bluestein's FFT
        nBytesBuffer += (N_BYTES_C32 * MAX_N_USER_GROUPS_SUPPORTED * FFT8192 + EXTRA_PADDING); //for time domain data in Bluestein's FFT Workspace
        nBytesBuffer += (N_BYTES_C32 * MAX_N_USER_GROUPS_SUPPORTED * FFT8192 + EXTRA_PADDING); //for freq domain data in Bluestein's FFT Workspace
    }
    // max equalizer output LLR buffer
    uint32_t maxBitsPerQam     = 8;
    uint32_t maxBytesEqOutLLRs = N_BYTES_R16 * NF * maxBitsPerQam * N_MAX_LAYERS * (OFDM_SYMBOLS_PER_SLOT - 1);
    nBytesBuffer += maxBytesEqOutLLRs + EXTRA_PADDING;
    nBytesBuffer += maxBytesEqOutLLRs + EXTRA_PADDING; // for LLR CDM1

    uint32_t max_N_TBs = pStatPrms->nMaxTbs ? pStatPrms->nMaxTbs : ((pStatPrms->nMaxCellsPerSlot) > 1 ? MAX_N_TBS_PER_CELL_GROUP_SUPPORTED : MAX_N_TBS_SUPPORTED);

    uint32_t max_N_CBs_PER_TB = pStatPrms->nMaxCbsPerTb ? pStatPrms->nMaxCbsPerTb : (pStatPrms->nMaxCellsPerSlot) > 1 ? MAX_N_CBS_PER_TB_PER_CELL_GROUP_SUPPORTED : MAX_N_CBS_PER_TB_SUPPORTED;
    uint32_t max_N_TOT_CBs    = pStatPrms->nMaxTotCbs ? pStatPrms->nMaxTotCbs : max_N_TBs * max_N_CBs_PER_TB;

    // TODO: check against max values per cell-group

    uint32_t MAX_N_TBs_SUPPORTED        = (pStatPrms->nMaxCellsPerSlot) > 1 ? MAX_N_TBS_PER_CELL_GROUP_SUPPORTED : MAX_N_TBS_SUPPORTED;
    uint32_t MAX_N_CBs_PER_TB_SUPPORTED = (pStatPrms->nMaxCellsPerSlot) > 1 ? MAX_N_CBS_PER_TB_PER_CELL_GROUP_SUPPORTED : MAX_N_CBS_PER_TB_SUPPORTED;

    if(max_N_TBs > MAX_N_TBs_SUPPORTED)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "nMaxTbs provided {} is larger than supported max {}", max_N_TBs, MAX_N_TBs_SUPPORTED);
        std::string err("PUSCH: nMaxTbs provided (" + std::to_string(max_N_TBs) + ") is larger than supported max (" + std::to_string(MAX_N_TBs_SUPPORTED) + ")");
        throw std::out_of_range(err);
    }

    if(max_N_CBs_PER_TB > MAX_N_CBs_PER_TB_SUPPORTED)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "nMaxCbsPerTb provided {} is larger than supported max {}", max_N_CBs_PER_TB, MAX_N_CBs_PER_TB_SUPPORTED);
        std::string err("PUSCH: nMaxCbsPerTb provided (" + std::to_string(max_N_CBs_PER_TB) + ") is larger than supported max (" + std::to_string(MAX_N_CBs_PER_TB_SUPPORTED) + ")");
        throw std::out_of_range(err);
    }

    if(max_N_TOT_CBs > max_N_TBs * max_N_CBs_PER_TB)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "nMaxTotCbs provided {} is larger than provided nMaxTbs * nMaxCbsPerTb {}", max_N_TOT_CBs, max_N_TBs * max_N_CBs_PER_TB);
        std::string err("PUSCH: nMaxTotCbs provided (" + std::to_string(max_N_TOT_CBs) + ") is larger than provided nMaxTbs * nMaxCbsPerTb (" + std::to_string(max_N_TBs * max_N_CBs_PER_TB) + ")");
        throw std::out_of_range(err);
    }

    // max LDPC outout buffer
    uint32_t maxBytesLDPCout = BYTES_PER_WORD * OUT_STRIDE_WORDS * max_N_TOT_CBs;
    nBytesBuffer += maxBytesLDPCout + EXTRA_PADDING;

    // max data output buffer
    uint32_t maxBytesDataOut = maxBytesLDPCout;
    nBytesBuffer += maxBytesDataOut + EXTRA_PADDING;

    // max CB crc buffer (each crc stored in uint32_t)
    uint32_t maxBytesCbCrc = N_BYTES_PER_UINT32 * max_N_TOT_CBs;
    nBytesBuffer += maxBytesCbCrc + EXTRA_PADDING;

    // max TB crc buffer (each crc stored in uint32_t)
    uint32_t maxBytesTbCrc = N_BYTES_PER_UINT32 * max_N_TBs;
    nBytesBuffer += maxBytesTbCrc + EXTRA_PADDING;

    // perTBPrms size
    uint32_t perTBPrmsMaxBytes = sizeof(PerTbParams) * max_N_TBs;
    nBytesBuffer += perTBPrmsMaxBytes + EXTRA_PADDING;

    // uci on pusch codeword parameters:
    uint32_t nCtrlChannels  = 3; // HARQ + CSI1 + CSI2
    uint32_t cwPrmsMaxBytes = nCtrlChannels * CUPHY_MAX_N_UCI_ON_PUSCH * (sizeof(cuphySimplexCwPrm_t) + sizeof(cuphyRmCwPrm_t) + sizeof(cuphyPolarUciSegPrm_t) + 2 * sizeof(cuphyPolarCwPrm_t));
    nBytesBuffer += cwPrmsMaxBytes + 8 * EXTRA_PADDING; // 8 = 4 x 2

    // uci on pusch output:
    uint32_t uciOutMaxBytes = nCtrlChannels * CUPHY_MAX_N_UCI_ON_PUSCH * sizeof(uint32_t);
    nBytesBuffer += uciOutMaxBytes + EXTRA_PADDING;

    // polar cwTree:
    uint32_t max_N          = 1024;
    uint32_t cwTreeMaxBytes = (2 * max_N) * (CUPHY_MAX_N_POL_UCI_SEGS + CUPHY_MAX_N_POL_UCI_SEGS_CSI2);
    nBytesBuffer += cwTreeMaxBytes + EXTRA_PADDING;

    // polar cbEst workspace:
    uint32_t cbEstWorkspaceBytes = (1024 / 8) * 2 * (CUPHY_MAX_N_POL_UCI_SEGS + CUPHY_MAX_N_POL_UCI_SEGS_CSI2);
    nBytesBuffer += cbEstWorkspaceBytes + EXTRA_PADDING;

    // polar cw LLRs:
    uint32_t cwLLRsMaxBytes = 2 * max_N * (CUPHY_MAX_N_POL_CWS + CUPHY_MAX_N_POL_CWS_CSI2);
    nBytesBuffer += cwLLRsMaxBytes;

    if(pStatPrms->enableRssiMeasurement)
    {
        uint32_t maxBytesRssiFull            = N_BYTES_R32 * MAX_ND_SUPPORTED * MAX_N_ANTENNAS_SUPPORTED * MAX_N_USER_GROUPS_SUPPORTED + LINEAR_ALLOC_PAD_BYTES;
        uint32_t maxBytesRssi                = N_BYTES_R32 * MAX_N_USER_GROUPS_SUPPORTED + LINEAR_ALLOC_PAD_BYTES;
        uint32_t maxBytesRssiInterCtaSyncCnt = N_BYTES_PER_UINT32 * MAX_N_USER_GROUPS_SUPPORTED + LINEAR_ALLOC_PAD_BYTES;

        nBytesBuffer += (maxBytesRssiFull + maxBytesRssi + maxBytesRssiInterCtaSyncCnt);
    }

    if(m_chEstSettings.enableSinrMeasurement || (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_NOISE_DIAG_MMSE)|| (m_chEstSettings.eqCoeffAlgo == PUSCH_EQ_ALGO_TYPE_MMSE_IRC))
    {
#if USE_PUSCH_PER_UE_PREQ_NOISE_VAR
        uint32_t maxBytesNoiseVarPreEq               = N_BYTES_R32 * MAX_N_UE + LINEAR_ALLOC_PAD_BYTES;
#else
        uint32_t maxBytesNoiseVarPreEq               = N_BYTES_R32 * MAX_N_USER_GROUPS_SUPPORTED + LINEAR_ALLOC_PAD_BYTES;
#endif
        uint32_t maxBytesNoiseVarPostEq              = N_BYTES_R32 * MAX_N_UE + LINEAR_ALLOC_PAD_BYTES;
        uint32_t maxBytesNoiseIntfEstInterCtaSyncCnt = N_BYTES_PER_UINT32 * MAX_N_USER_GROUPS_SUPPORTED + LINEAR_ALLOC_PAD_BYTES;
        uint32_t maxBytesNoiseIntfEstLwInv           = N_BYTES_C32 * N_RX * N_RX * max_nPrbUlBwp * nTimeChEq + (MAX_N_USER_GROUPS_SUPPORTED * LINEAR_ALLOC_PAD_BYTES);
        uint32_t maxBytesNoiseVarPerPrb              = CUPHY_ENABLE_SUB_SLOT_PROCESSING ? N_BYTES_R32 * MAX_N_PRBS_SUPPORTED + LINEAR_ALLOC_PAD_BYTES : 0;
        // 2x PreEq noise buffer (one in dB reported to L1C + one in linear domain used in DTX detection) fix it
        nBytesBuffer += ((2*maxBytesNoiseVarPreEq) + maxBytesNoiseVarPostEq + maxBytesNoiseIntfEstInterCtaSyncCnt + maxBytesNoiseIntfEstLwInv + maxBytesNoiseVarPerPrb);
    }

    // for DTX results
    nBytesBuffer += cuphyUciDtxTypes_t::N_UCI_DTX*MAX_N_UE;

    if(pStatPrms->enableSinrMeasurement)
    {
        uint32_t maxBytesRsrp                = N_BYTES_R32 * MAX_N_UE + LINEAR_ALLOC_PAD_BYTES;
        uint32_t maxBytesRsrpInterCtaSyncCnt = N_BYTES_PER_UINT32 * MAX_N_USER_GROUPS_SUPPORTED + LINEAR_ALLOC_PAD_BYTES;

        // 2 - 1 for pre-Eq SINR + 1 for post-Eq SINR
        uint32_t maxBytesSinr = (2 * N_BYTES_R32 * MAX_N_UE) + LINEAR_ALLOC_PAD_BYTES;

        nBytesBuffer += (maxBytesRsrp + maxBytesRsrpInterCtaSyncCnt + maxBytesSinr);
    }

    // printf("nBytesBuffer %zu nMaxCellsPerSlot %d\n", nBytesBuffer, pStatPrms->nMaxCellsPerSlot);
    return (nBytesBuffer * pStatPrms->nMaxCellsPerSlot);
}

uint32_t PuschRx::expandFrontEndParameters(cuphyPuschDynPrms_t* pDynPrm, cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms, uint8_t enableRssiMeasurement)
{
    uint32_t                  nMaxPrb       = 0;
    cuphyPuschCellGrpDynPrm_t cellGrpDynPrm = *(pDynPrm->pCellGrpDynPrm);
    for(uint32_t iterator = 0; iterator < cellGrpDynPrm.nUeGrps; iterator++)
    {
        cuphyPuschUeGrpPrm_t*   ueGrpPrms   = &cellGrpDynPrm.pUeGrpPrms[iterator];
        cuphyPuschCellDynPrm_t* cellDynPrm  = ueGrpPrms->pCellPrm;
        cuphyPuschUePrm_t*      uePrmsArray = pDynPrm->pCellGrpDynPrm->pUePrms;

        // common parameters
        uint16_t cellPrmStatIdx              = ueGrpPrms->pCellPrm->cellPrmStatIdx;
        pDrvdUeGrpPrms[iterator].statCellIdx = cellPrmStatIdx;
        pDrvdUeGrpPrms[iterator].nRxAnt      = m_cuphyCellStatPrmVecCpu[cellPrmStatIdx].nRxAnt;
        pDrvdUeGrpPrms[iterator].slotNum     = cellDynPrm->slotNum;
        pDrvdUeGrpPrms[iterator].scsKHz      = cuphy::context::getScsKHz(m_cuphyCellStatPrmVecCpu[cellPrmStatIdx].mu);

        // Frequency domain resource allocation
        pDrvdUeGrpPrms[iterator].startPrb = ueGrpPrms->startPrb;
        pDrvdUeGrpPrms[iterator].nPrb     = ueGrpPrms->nPrb;
        if(nMaxPrb < pDrvdUeGrpPrms[iterator].nPrb)
            nMaxPrb = pDrvdUeGrpPrms[iterator].nPrb;

        if(nMaxPrb > m_maxNPrbAlloc)
        {
            pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_NPRBS_OUT_OF_RANGE;
            pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
            pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16;
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Number of Prbs in cell is {} but current max. specified for PUSCH is {}", __FUNCTION__, nMaxPrb, m_maxNPrbAlloc);
            return 0xFFFFFFFF;
        }

        // time allocation
        pDrvdUeGrpPrms[iterator].puschStartSym = ueGrpPrms->puschStartSym;
        pDrvdUeGrpPrms[iterator].nPuschSym     = ueGrpPrms->nPuschSym;

        cuphyPuschDmrsPrm_t* dmrsPrm      = ueGrpPrms->pDmrsDynPrm;
        uint32_t             nDmrsSymbols = static_cast<uint32_t>((1 + dmrsPrm->dmrsAddlnPos) * dmrsPrm->dmrsMaxLen);
        //pDrvdUeGrpPrms[iterator].nDmrsSyms = static_cast<uint32_t>(dmrsPrm->dmrsMaxLen); //does not include additional DMRS
        pDrvdUeGrpPrms[iterator].nDmrsSyms  = nDmrsSymbols;
        pDrvdUeGrpPrms[iterator].dmrsScrmId = dmrsPrm->dmrsScrmId;
        pDrvdUeGrpPrms[iterator].nDmrsCdmGrpsNoData = dmrsPrm->nDmrsCdmGrpsNoData;

        if(0 == ueGrpPrms->dmrsSymLocBmsk) CUPHY_CHECK(CUPHY_STATUS_NOT_SUPPORTED);
        uint32_t nTotalDmrsSyms                 = __builtin_popcount(ueGrpPrms->dmrsSymLocBmsk);
        pDrvdUeGrpPrms[iterator].dmrsSymLocBmsk = ueGrpPrms->dmrsSymLocBmsk;

        // initialization/constants
        pDrvdUeGrpPrms[iterator].nLayers            = 0;
        pDrvdUeGrpPrms[iterator].nUes               = ueGrpPrms->nUes;
        uint16_t                  groupDmrsPortBmsk = 0;
        static constexpr uint16_t GRP_DMRS_PORT_MSK = (1U << 12) - 1; // DMRS port info in bit 0 to bit 11

        uint8_t arrDmrsPorts[MAX_N_LAYERS_PUSCH][2];
        uint8_t layerIdx      = 0;
        uint8_t ueGrpLayerIdx = 0;
        for(int i = 0; i < ueGrpPrms->nUes; ++i)
        {
            uint16_t ueIdx                     = ueGrpPrms->pUePrmIdxs[i];
            pDrvdUeGrpPrms[iterator].ueIdxs[i] = ueIdx;

            // number of layers
            pDrvdUeGrpPrms[iterator].nUeLayers[i] = uePrmsArray[ueIdx].nUeLayers;
            pDrvdUeGrpPrms[iterator].nLayers += static_cast<uint32_t>(uePrmsArray[ueIdx].nUeLayers);

            // derive active dmrs grids
            uint16_t DmrsPortBmsk = uePrmsArray[ueIdx].dmrsPortBmsk & GRP_DMRS_PORT_MSK;

            groupDmrsPortBmsk = groupDmrsPortBmsk | DmrsPortBmsk;

            for(uint16_t k = 0; k < 8; k++)
            {
                if((DmrsPortBmsk >> k) & 0x1)
                {
                    arrDmrsPorts[layerIdx][0]                       = k;
                    arrDmrsPorts[layerIdx][1]                       = k & 0x2;
                    pDrvdUeGrpPrms[iterator].dmrsPortIdxs[layerIdx] = k;
                    // printf("UeGrp[%d] Ue[%d] dmrsPortIdxs[%d] = %d\n", iterator, i, layerIdx, k);
                    layerIdx++;
                }
            }

            for(int j = 0; j < pDrvdUeGrpPrms[iterator].nUeLayers[i]; ++j)
            {
                pDrvdUeGrpPrms[iterator].ueGrpLayerToUeIdx[ueGrpLayerIdx] = i;
                // printf("ueGrpIdx %d ueGrpLayerIdx %d ueGrpLayerToUeIdx %d\n", iterator, ueGrpLayerIdx, pDrvdUeGrpPrms[iterator].ueGrpLayerToUeIdx[ueGrpLayerIdx]);
                ueGrpLayerIdx++;
            }

            // printf("UeGrp[%d] Ue[%d] dmrsPortBmsk 0x%x groupDmrsPortBmsk 0x%x\n", iterator, i, DmrsPortBmsk, groupDmrsPortBmsk);

            // scid
            pDrvdUeGrpPrms[iterator].scid = uePrmsArray[ueIdx].scid;
            
            // DFT-s-OFDM
            pDrvdUeGrpPrms[iterator].enableTfPrcd           = uePrmsArray[ueIdx].enableTfPrcd;
            if(pDrvdUeGrpPrms[iterator].enableTfPrcd==1)
            {
                pDrvdUeGrpPrms[iterator].puschIdentity          = uePrmsArray[ueIdx].puschIdentity;
                pDrvdUeGrpPrms[iterator].groupOrSequenceHopping = uePrmsArray[ueIdx].groupOrSequenceHopping;
                pDrvdUeGrpPrms[iterator].N_symb_slot            = uePrmsArray[ueIdx].N_symb_slot;
                pDrvdUeGrpPrms[iterator].N_slot_frame           = uePrmsArray[ueIdx].N_slot_frame;
                pDrvdUeGrpPrms[iterator].lowPaprGroupNumber     = uePrmsArray[ueIdx].lowPaprGroupNumber;
                pDrvdUeGrpPrms[iterator].lowPaprSequenceNumber  = uePrmsArray[ueIdx].lowPaprSequenceNumber;
                
                if(uePrmsArray[ueIdx].pduBitmap & 0x8)
                {
                    pDrvdUeGrpPrms[iterator].optionalDftSOfdm = 1;
                }
                else
                {
                    pDrvdUeGrpPrms[iterator].optionalDftSOfdm = 0;
                }
            }
        }

        // printf("nMaxCells %d nMaxCellsPerSlot %d nCells %d cellDynIdx[%d]: phyCellId %d nPrbUlBwp %d nRxAnt %d \n", m_cuphyPuschStatPrms.nMaxCells, m_cuphyPuschStatPrms.nMaxCellsPerSlot, m_cuphyPuschCellGrpDynPrm.nCells, cellDynPrm->cellPrmDynIdx, m_cuphyCellStatPrmVecCpu[cellDynPrm->cellPrmStatIdx].phyCellId, m_cuphyCellStatPrmVecCpu[cellDynPrm->cellPrmStatIdx].nPrbUlBwp, m_cuphyCellStatPrmVecCpu[cellDynPrm->cellPrmStatIdx].nRxAnt);

        // finish derivation of active dmrs grids
        uint32_t gridBmsk0 = static_cast<uint32_t>(((0x33 & groupDmrsPortBmsk) != 0) ? 1 : 0);
        uint32_t gridBmsk1 = static_cast<uint32_t>(((0xCC & groupDmrsPortBmsk) != 0) ? 1 : 0);

        pDrvdUeGrpPrms[iterator].nDmrsGridsPerPrb   = 2;
        pDrvdUeGrpPrms[iterator].activeDMRSGridBmsk = gridBmsk0 | (gridBmsk1 << 1);

        // derive active TOCCs and FOCCs

        uint32_t ToccBmsk0_0 = static_cast<uint32_t>(((0x03 & groupDmrsPortBmsk) != 0) ? 1 : 0);
        uint32_t ToccBmsk0_1 = static_cast<uint32_t>(((0x30 & groupDmrsPortBmsk) != 0) ? 1 : 0);
        uint32_t ToccBmsk1_0 = static_cast<uint32_t>(((0x0C & groupDmrsPortBmsk) != 0) ? 1 : 0);
        uint32_t ToccBmsk1_1 = static_cast<uint32_t>(((0xC0 & groupDmrsPortBmsk) != 0) ? 1 : 0);

        uint32_t FoccBmsk0_0 = static_cast<uint32_t>(((0x11 & groupDmrsPortBmsk) != 0) ? 1 : 0);
        uint32_t FoccBmsk0_1 = static_cast<uint32_t>(((0x22 & groupDmrsPortBmsk) != 0) ? 1 : 0);
        uint32_t FoccBmsk1_0 = static_cast<uint32_t>(((0x44 & groupDmrsPortBmsk) != 0) ? 1 : 0);
        uint32_t FoccBmsk1_1 = static_cast<uint32_t>(((0x88 & groupDmrsPortBmsk) != 0) ? 1 : 0);

        pDrvdUeGrpPrms[iterator].activeTOCCBmsk[0] = ToccBmsk0_0 | (ToccBmsk0_1 << 1);
        pDrvdUeGrpPrms[iterator].activeTOCCBmsk[1] = ToccBmsk1_0 | (ToccBmsk1_1 << 1);
        pDrvdUeGrpPrms[iterator].activeFOCCBmsk[0] = FoccBmsk0_0 | (FoccBmsk0_1 << 1);
        pDrvdUeGrpPrms[iterator].activeFOCCBmsk[1] = FoccBmsk1_0 | (FoccBmsk1_1 << 1);

        uint32_t sumToccBmsk0 = ToccBmsk0_0 + ToccBmsk0_1;
        uint32_t sumToccBmsk1 = ToccBmsk1_0 + ToccBmsk1_1;
        uint32_t sumFoccBmsk0 = FoccBmsk0_0 + FoccBmsk0_1;
        uint32_t sumFoccBmsk1 = FoccBmsk1_0 + FoccBmsk1_1;

        for(int i = 0; i < pDrvdUeGrpPrms[iterator].nLayers; i++)
        {
            if(arrDmrsPorts[i][1] == 0)
            { // first grid
                if(sumToccBmsk0 == 1)
                {
                    if(sumFoccBmsk0 == 1)
                    {
                        pDrvdUeGrpPrms[iterator].OCCIdx[i] = 0 + (arrDmrsPorts[i][1] << 1);
                    }
                    else
                    { // sumFoccBmsk0 == 2
                        pDrvdUeGrpPrms[iterator].OCCIdx[i] = (arrDmrsPorts[i][0] & 0x1) + (arrDmrsPorts[i][1] << 1);
                    }
                }
                else
                { // sumToccBmsk0 == 2
                    if(sumFoccBmsk0 == 1)
                    {
                        pDrvdUeGrpPrms[iterator].OCCIdx[i] = (arrDmrsPorts[i][0] < 4 ? 0 : 1) + (arrDmrsPorts[i][1] << 1);
                    }
                    else
                    { // sumFoccBmsk0 == 2
                        pDrvdUeGrpPrms[iterator].OCCIdx[i] = ((arrDmrsPorts[i][0] >> 1) & 0x2) + (arrDmrsPorts[i][0] & 0x1) + (arrDmrsPorts[i][1] << 1);
                    }
                }
            }
            else
            { // second grid
                if(sumToccBmsk1 == 1)
                {
                    if(sumFoccBmsk1 == 1)
                    {
                        pDrvdUeGrpPrms[iterator].OCCIdx[i] = 0 + (arrDmrsPorts[i][1] << 1);
                    }
                    else
                    {
                        pDrvdUeGrpPrms[iterator].OCCIdx[i] = (arrDmrsPorts[i][0] & 0x1) + (arrDmrsPorts[i][1] << 1);
                    }
                }
                else
                {
                    if(sumFoccBmsk1 == 1)
                    {
                        pDrvdUeGrpPrms[iterator].OCCIdx[i] = (arrDmrsPorts[i][0] < 4 ? 0 : 1) + (arrDmrsPorts[i][1] << 1);
                    }
                    else
                    {
                        pDrvdUeGrpPrms[iterator].OCCIdx[i] = ((arrDmrsPorts[i][0] >> 1) & 0x2) + (arrDmrsPorts[i][0] & 0x1) + (arrDmrsPorts[i][1] << 1);
                    }
                }
            }
        }

        // Channel equalization & Soft Demap
        uint8_t dataCnt = 0;
        uint8_t dmrsCnt = 0;
        for(uint8_t i = ueGrpPrms->puschStartSym; i < (ueGrpPrms->puschStartSym + ueGrpPrms->nPuschSym); ++i)
        {
            if(1 & (ueGrpPrms->dmrsSymLocBmsk >> i))
            {
                pDrvdUeGrpPrms[iterator].dmrsSymLoc[dmrsCnt++] = i;
                if(pDrvdUeGrpPrms[iterator].nDmrsCdmGrpsNoData==1)
                {
                    pDrvdUeGrpPrms[iterator].dataSymLoc[dataCnt++] = i;
                }
            }
            else
                pDrvdUeGrpPrms[iterator].dataSymLoc[dataCnt++] = i;
        }

        pDrvdUeGrpPrms[iterator].dmrsCnt      = dmrsCnt;
        pDrvdUeGrpPrms[iterator].dmrsMaxLen   = dmrsPrm->dmrsMaxLen;
        pDrvdUeGrpPrms[iterator].dmrsAddlnPos = dmrsPrm->dmrsAddlnPos;
        pDrvdUeGrpPrms[iterator].dataCnt      = dataCnt;
        pDrvdUeGrpPrms[iterator].nTimeChEsts  = dmrsPrm->dmrsAddlnPos + 1;

        pDrvdUeGrpPrms[iterator].nDataSym            = static_cast<uint32_t>(ueGrpPrms->nPuschSym) - nDmrsSymbols;
        pDrvdUeGrpPrms[iterator].enableCfoCorrection = (m_chEstSettings.enableCfoCorrection && (dmrsPrm->dmrsAddlnPos > 0)) ? 1 : 0;
        pDrvdUeGrpPrms[iterator].enableToEstimation  = (m_chEstSettings.enableToEstimation && (dmrsPrm->dmrsAddlnPos > 0)) ? 1 : 0;
        pDrvdUeGrpPrms[iterator].enablePuschTdi      = m_chEstSettings.enablePuschTdi;
        pDrvdUeGrpPrms[iterator].eqCoeffAlgo         = m_chEstSettings.eqCoeffAlgo;

        // RSSI measurement
        static constexpr uint16_t SLOT_SYMB_BMSK = (static_cast<decltype(ueGrpPrms->rssiSymLocBmsk)>(1) << MAX_ND_SUPPORTED) - 1;
        pDrvdUeGrpPrms[iterator].rssiSymPosBmsk  = enableRssiMeasurement ? (ueGrpPrms->rssiSymLocBmsk & SLOT_SYMB_BMSK) : 0;
    }
    return nMaxPrb;
}

cuphyStatus_t PuschRx::expandBackEndParameters(cuphyPuschDynPrms_t* pDynPrm, cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms, PerTbParams* pPerTbPrms, cuphyLDPCParams& ldpcPrms)
{
    cuphyPuschCellGrpDynPrm_t cellGrpDynPrm = *(pDynPrm->pCellGrpDynPrm);
    cuphyPuschUePrm_t*        uePrmsArray   = pDynPrm->pCellGrpDynPrm->pUePrms;
    double                    codeRateArray[MAX_N_TBS_PER_CELL_GROUP_SUPPORTED];
    uint32_t                  K_cb, B, B_prime, K_prime, crcPolyByteSize;
    //
    // Loop over per-UE items
    //

    // clang-format off
    uint32_t TBS_table[93] = {24, 32, 40, 48, 56, 64, 72, 80, 88, 96, 104, 112, 120, 128, 136, 144, 152,
                              160, 168, 176, 184, 192, 208, 224, 240, 256, 272, 288, 304, 320, 336, 352,
                              368, 384, 408, 432, 456, 480, 504, 528, 552, 576, 608, 640, 672, 704, 736,
                              768, 808, 848, 888, 928, 984, 1032, 1064, 1128, 1160, 1192, 1224, 1256, 1288,
                              1320, 1352, 1416, 1480, 1544, 1608, 1672, 1736, 1800, 1864, 1928, 2024, 2088,
                              2152, 2216, 2280, 2408, 2472, 2536, 2600, 2664, 2728, 2792, 2856, 2976, 3104,
                              3240, 3368, 3496, 3624, 3752, 3824};
    // clang-format on

    uint32_t totBitSize = 0;

    uint32_t layerCount[MAX_N_USER_GROUPS_SUPPORTED] = {0};

    uint32_t totNCBs = 0;
    for(int i = 0; i < cellGrpDynPrm.nUes; i++)
    {
        cuphyPuschUeGrpPrm_t*    pUeGrpPrm     = uePrmsArray[i].pUeGrpPrm;
        cuphyPuschCellDynPrm_t*  pCellDynPrm   = pUeGrpPrm->pCellPrm;
        uint16_t                 ueGrpIdx      = uePrmsArray[i].ueGrpIdx;
        cuphyPuschRxUeGrpPrms_t* drvdUeGrpPrms = &pDrvdUeGrpPrms[ueGrpIdx];
        
        //DFT-s-OFDM
        pPerTbPrms[i].enableTfPrcd             = uePrmsArray[i].enableTfPrcd;

        // HARQ parameters
        // pPerTbPrms[i].ndi                      = uePrmsArray[i].ndi; //ndi will be updated in setupCmnPhase2()
        pPerTbPrms[i].rv                       = uePrmsArray[i].rv;
        pPerTbPrms[i].debug_d_derateCbsIndices = uePrmsArray[i].debug_d_derateCbsIndices;

        // compute cinit seeds for descrambling
        pPerTbPrms[i].cinit = (static_cast<uint32_t>(((uePrmsArray[i].rnti << 15) + uePrmsArray[i].dataScramId)) & (0x7FFFFFFF));

        // find Qm and target code rate
        uint32_t tbSize;

        pPerTbPrms[i].Qm = uePrmsArray[i].qamModOrder;
        codeRateArray[i] = uePrmsArray[i].targetCodeRate / (10.0f * 1024.0f);

        // Derive TB size and number of code blocks C(from derive_TB_size.m)

        // Compute number of REs
        uint32_t nUeLayers        = static_cast<uint32_t>(uePrmsArray[i].nUeLayers);
        uint32_t nPrb             = static_cast<uint32_t>(uePrmsArray[i].pUeGrpPrm->nPrb);
        uint32_t nDataSymbols     = drvdUeGrpPrms->nDataSym;
        uint32_t Ndata            = CUPHY_N_TONES_PER_PRB * nPrb * nDataSymbols;
        uint8_t  nDmrsCdmGrpsNoData = drvdUeGrpPrms->nDmrsCdmGrpsNoData;
        if(nDmrsCdmGrpsNoData==1)
        {
            Ndata += ((CUPHY_N_TONES_PER_PRB>>1) * nPrb * drvdUeGrpPrms->nDmrsSyms);
        }
        uint32_t Nre              = Ndata;//std::min(static_cast<uint32_t>(156), Ndata / nPrb) * nPrb;
        pPerTbPrms[i].encodedSize = Nre * pPerTbPrms[i].Qm * nUeLayers;
        pPerTbPrms[i].nDmrsCdmGrpsNoData = nDmrsCdmGrpsNoData;
        
        if(m_chEstSettings.enableTbSizeCheck==1)
        {
            uint8_t  mcsTableIndex = uePrmsArray[i].mcsTableIndex; 
            uint8_t  mcsIndex      = uePrmsArray[i].mcsIndex;
        
            bool enableTbSizeCheckCondition0 = ((mcsTableIndex==1)||(mcsTableIndex==3)||(mcsTableIndex==4))&&(mcsIndex<28); 
            bool enableTbSizeCheckCondition1 = ((mcsTableIndex==0)||(mcsTableIndex==2))&&(mcsIndex<29); 
            bool enableTbSizeCheckCondition2 = uePrmsArray[i].pduBitmap & 1;  //TODO further investigation why enableTbSizeCheckCondition2 is needed for cuBB test.  
        
            if(enableTbSizeCheckCondition2 && (enableTbSizeCheckCondition0 || enableTbSizeCheckCondition1))
            {
                // Compute number of info bits
                float    Ninfo = Nre * codeRateArray[i] * pPerTbPrms[i].Qm * nUeLayers;
                uint32_t Ninfo_prime;

                if(Ninfo <= 3824)
                {
                    // For "small" sizes, look up TBS in a table. First round the
                    // number of information bits.
                    uint32_t n  = std::max(3, int(floor(log2(Ninfo)) - 6));
                    Ninfo_prime = std::max(24, int(pow(2, n) * floor(Ninfo / pow(2, n))));

                    // Pick smallest TB from TBS_table which is larger than or equal to Ninfo_prime
                    for(int j = 0; j < 93; j++)
                    {
                        if(Ninfo_prime <= TBS_table[j])
                        {
                            tbSize = TBS_table[j];
                            break;
                        }
                    }
                    pPerTbPrms[i].num_CBs = 1;
                }
                else
                {
                    // For "large" sizes, compute TBS. First round the number of
                    // information bits to a power of two.
                    uint32_t n  = floor(log2(static_cast<float>(Ninfo - 24))) - 5;
                    Ninfo_prime = std::max(3840, int(pow(2, n) * round((double(Ninfo - 24.0) / pow(2, n)))));
                    // printf("nre %d n %d, ninfo %d Ninfo_prime %d\n", Nre, n, Ninfo, Ninfo_prime);
                    // Next, compute the number of code words. For large code rates,
                    // use base-graph 1. For small code rate use base-graph 2.

                    if(codeRateArray[i] < 0.25)
                    {
                        uint32_t C = div_round_up((Ninfo_prime + 24), static_cast<uint32_t>(3816));
                        tbSize     = 8 * C * div_round_up((Ninfo_prime + 24), (8 * C)) - 24;
                    }
                    else
                    {
                        if(Ninfo_prime > 8424)
                        {
                            uint32_t C = div_round_up((Ninfo_prime + 24), static_cast<uint32_t>(8424));
                            tbSize     = 8 * C * div_round_up((Ninfo_prime + 24), (8 * C)) - 24;
                        }
                        else
                        {
                            uint32_t C = 1;
                            tbSize     = 8 * C * div_round_up((Ninfo_prime + 24), (8 * C)) - 24;
                        }
                    }
                }
            
                if(uePrmsArray[i].TBSize*8 != tbSize)  //TBSize (in bytes) following FAPI 10.04
                {
                    pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_TBSIZE_MISMATCH;
                    pDynPrm->pStatusOut->ueIdx = i;
                    pDynPrm->pStatusOut->cellPrmStatIdx = pCellDynPrm->cellPrmStatIdx; 
                    NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: L1 calculated tbSize {} is not equal to L2 provided tbSize {} with mcsTableIndex {} and mcsIndex {} for UE {} in Cell {}", __FUNCTION__, tbSize, uePrmsArray[i].TBSize*8, mcsTableIndex, mcsIndex, pDynPrm->pStatusOut->ueIdx, pDynPrm->pStatusOut->cellPrmStatIdx);
                    return CUPHY_STATUS_INVALID_ARGUMENT;
                }
            }
        }
        
        //get tbSize (in bits) from L2 TBSize (in bytes)
        tbSize = uePrmsArray[i].TBSize*8; 
        
        pPerTbPrms[i].tbSize = tbSize;
        // Derive BG (from derive_BGN.m)
        if((tbSize <= 292) || ((tbSize <= 3824) && (codeRateArray[i] <= 0.67)) || (codeRateArray[i] <= 0.25))
            pPerTbPrms[i].bg = 2;
        else
            pPerTbPrms[i].bg = 1;

        // Derive codeblock size and number of filler bits

        // uint32_t polyBitSize = 24;
        // Max number of bits per codeblock
        if(pPerTbPrms[i].bg == 1)
        {
            K_cb = 8448;
        }
        else
        {
            K_cb = 3840;
        }
        // Number of codeblocks
        if(tbSize <= K_cb)
        {
            crcPolyByteSize       = tbSize <= 3824 ? 2 : 3;                     // CRC-16
            B                     = tbSize <= 3824 ? tbSize + 16 : tbSize + 24; // size of TB + TB-CRC
            pPerTbPrms[i].num_CBs = 1;                                          // number of CBs //TODO for tbSize = 0
            B_prime               = B;                                          // size of TB + TB-CRC + CB-CRCs
        }
        else
        {
            crcPolyByteSize       = 3;                              // CRC-24
            B                     = tbSize + 24;                    // size of TB + TB-CRC
            pPerTbPrms[i].num_CBs = div_round_up(B, (K_cb - 24));   // number of CBs
            B_prime               = B + pPerTbPrms[i].num_CBs * 24; // size of TB + TB-CRC + CB-CRCs
        }

        // Bits per code block
        K_prime = B_prime / pPerTbPrms[i].num_CBs;
        
        // check the number of encoded bits per CB
        if(uePrmsArray[i].pduBitmap & 1)
        {
            uint32_t Eh = nUeLayers * pPerTbPrms[i].Qm * ceilf(float(pPerTbPrms[i].encodedSize) / float(nUeLayers * pPerTbPrms[i].Qm * pPerTbPrms[i].num_CBs));
            if (Eh > PUSCH_MAX_ER_PER_CB_BITS)
            {
                pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_UNSUPPORTED_MAX_ER_PER_CB;
                pDynPrm->pStatusOut->ueIdx = i;
                pDynPrm->pStatusOut->cellPrmStatIdx = pCellDynPrm->cellPrmStatIdx; 
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: The number of encoded bits {} for UE {} in Cell {} per CB exceeds PUSCH_MAX_ER_PER_CB_BITS.", __FUNCTION__, Eh, pDynPrm->pStatusOut->ueIdx, pDynPrm->pStatusOut->cellPrmStatIdx);
                return CUPHY_STATUS_INVALID_ARGUMENT;
            }
        }
        /////////////////////////////////////////////////////

        // Derive lifting size
        if(pPerTbPrms[i].bg == 1)
            ldpcPrms.KbArray[i] = 22;
        else if(B > 640)
            ldpcPrms.KbArray[i] = 10;
        else if(B > 560)
            ldpcPrms.KbArray[i] = 9;
        else if(B > 192)
            ldpcPrms.KbArray[i] = 8;
        else
            ldpcPrms.KbArray[i] = 6;
        uint32_t Z[51] = {2, 4, 8, 16, 32, 64, 128, 256, 3, 6, 12, 24, 48, 96, 192, 384, 5, 10, 20, 40, 80, 160, 320, 7, 14, 28, 56, 112, 224, 9, 18, 36, 72, 144, 288, 11, 22, 44, 88, 176, 352, 13, 26, 52, 104, 208, 15, 30, 60, 120, 240};

        // Derive ZcArray (from derive_lifting.m)
        // find smallest Z such that Z*K_b >= K_prime:
        uint32_t tmp1, tmp2 = 1000000;
        for(int j = 0; j < 51; j++)
        {
            tmp1 = Z[j] * ldpcPrms.KbArray[i];

            if((tmp1 >= K_prime) && (tmp1 < tmp2))
            {
                tmp2             = tmp1;
                pPerTbPrms[i].Zc = Z[j];
            }
        }

        // Derive K (codeblock size) and F (number of filler bits)

        if(pPerTbPrms[i].bg == 1)
        {
            pPerTbPrms[i].K = pPerTbPrms[i].Zc * 22;
        }
        else
        {
            pPerTbPrms[i].K = pPerTbPrms[i].Zc * 10;
        }

        pPerTbPrms[i].F = pPerTbPrms[i].K - K_prime;

        // Derive startIdx

        // Derive E_vec - rate-matched code block sizes

        // Fill out output parameter structure TB-specific
        // Back-end

        uint32_t UeGrpNLayers             = drvdUeGrpPrms->nLayers;
        nDataSymbols                      = drvdUeGrpPrms->nDataSym;
        pPerTbPrms[i].firstCodeBlockIndex = (0); //NEEDS FIX, input tbStructs will have to contain symbol-by-symbol processing info
        pPerTbPrms[i].userGroupIndex      = static_cast<uint32_t>(ueGrpIdx);
        pPerTbPrms[i].nBBULayers          = UeGrpNLayers;
        pPerTbPrms[i].Nl                  = nUeLayers;
        pPerTbPrms[i].startLLR            = static_cast<uint32_t>(cellGrpDynPrm.pUeGrpPrms[ueGrpIdx].startPrb * 12 * QAM_STRIDE * pPerTbPrms[i].nBBULayers * nDataSymbols);

        for(int l = 0; l < nUeLayers; l++)
        {
            pPerTbPrms[i].layer_map_array[l] = layerCount[ueGrpIdx];
            layerCount[ueGrpIdx]++;
        }

        uint32_t Kd = pPerTbPrms[i].K - pPerTbPrms[i].F - 2 * pPerTbPrms[i].Zc;

        // Calculate Ncb (circular buffer) and Ncb_padded (circular buffer padded to 8b boundary)
        uint32_t Ncb = pPerTbPrms[i].bg == 1 ? pPerTbPrms[i].Zc * 66 : pPerTbPrms[i].Zc * 50;

        if(uePrmsArray[i].i_lbrm == 1)
        {
            const float R_LBRM      = 2. / 3.;
            const float maxRate     = 948. / 1024.;
            const int   num_symbols = 156 / 12;
            uint32_t    TBS_LBRM, num_CBs_unused;
            get_TB_size_and_num_CBs(
                num_symbols,               // int num_symbols,
                uePrmsArray[i].n_PRB_LBRM, // int num_prbs,
                uePrmsArray[i].maxLayers,  // int num_layers,
                maxRate,                   // float code_rate,
                uePrmsArray[i].maxQm,      // uint32_t Qm,
                num_CBs_unused,            // uint32_t &num_cBS
                TBS_LBRM,                  // uint32_t &tb_size
                0);                        // int num_dmrs_cdmGrpsNoData1_symbols; set to 0 for PUSCH

            uint32_t Nref = floor(TBS_LBRM / (pPerTbPrms[i].num_CBs * R_LBRM));
            if(Nref < Ncb) Ncb = Nref;
        }
        pPerTbPrms[i].Ncb = Ncb;

        if(pPerTbPrms[i].bg == 1)
        {
            uint32_t rv  = pPerTbPrms[i].rv;
            uint32_t Zc  = pPerTbPrms[i].Zc;
            uint32_t Ncb = pPerTbPrms[i].Ncb;
            uint32_t k0;
            if(rv == 0)
            {
                k0 = 0;
            }
            else if(rv == 1)
            {
                k0 = (17 * Ncb / (66 * Zc)) * Zc;
            }
            else if(rv == 2)
            {
                k0 = (33 * Ncb / (66 * Zc)) * Zc;
            }
            else if(rv == 3)
            {
                k0 = (56 * Ncb / (66 * Zc)) * Zc;
            }
            uint32_t Ncb_forparity       = std::min<uint32_t>((pPerTbPrms[i].encodedSize) / pPerTbPrms[i].num_CBs + k0, Ncb);
            uint32_t nPairtyNodes        = (uePrmsArray[i].i_lbrm == 0) ? ((Ncb_forparity - Kd + Zc - 1) / Zc) : ((Ncb_forparity - Kd) / Zc);
            ldpcPrms.parityNodesArray[i] = std::max<uint32_t>(4, std::min<uint32_t>(CUPHY_LDPC_MAX_BG1_PARITY_NODES, nPairtyNodes));
            DEBUG_PRINTF("BG=%u encodedSize=%u Zc=%u Ncb=%u k0=%u num_CBs=%u Kd=%u parityNodes=%u Ncb_forparity=%u\n",pPerTbPrms[i].bg,pPerTbPrms[i].encodedSize,Zc,Ncb,k0,pPerTbPrms[i].num_CBs,Kd,ldpcPrms.parityNodesArray[i],Ncb_forparity);
        }
        else
        {
            uint32_t rv  = pPerTbPrms[i].rv;
            uint32_t Zc  = pPerTbPrms[i].Zc;
            uint32_t Ncb = pPerTbPrms[i].Ncb;
            uint32_t k0;
            if(rv == 0)
            {
                k0 = 0;
            }
            else if(rv == 1)
            {
                k0 = (13 * Ncb / (50 * Zc)) * Zc;
            }
            else if(rv == 2)
            {
                k0 = (25 * Ncb / (50 * Zc)) * Zc;
            }
            else if(rv == 3)
            {
                k0 = (43 * Ncb / (50 * Zc)) * Zc;
            }
            uint32_t Ncb_forparity       = std::min<uint32_t>((pPerTbPrms[i].encodedSize) / pPerTbPrms[i].num_CBs + k0, Ncb);
            uint32_t nPairtyNodes        = (uePrmsArray[i].i_lbrm == 0) ? ((Ncb_forparity - Kd + Zc - 1) / Zc) : ((Ncb_forparity - Kd) / Zc);
            ldpcPrms.parityNodesArray[i] = std::max<uint32_t>(4, std::min<uint32_t>(CUPHY_LDPC_MAX_BG2_PARITY_NODES, nPairtyNodes));
            DEBUG_PRINTF("BG=%u encodedSize=%u Zc=%u Ncb=%u k0=%u num_CBs=%u Kd=%u parityNodes=%u Ncb_forparity=%u\n",pPerTbPrms[i].bg,pPerTbPrms[i].encodedSize,Zc,Ncb,k0,pPerTbPrms[i].num_CBs,Kd,ldpcPrms.parityNodesArray[i],Ncb_forparity);
        }

        uint32_t nZpBitsPerCb = (ldpcPrms.parityNodesArray[i] * pPerTbPrms[i].Zc) + pPerTbPrms[i].K;
        pPerTbPrms[i].nZpBitsPerCb = nZpBitsPerCb;

        // Calculate padding for codeblock LLRs
        uint32_t Ncb_padded = pPerTbPrms[i].Ncb + 2 * pPerTbPrms[i].Zc;

        // Ncb_padded is rounded up to byte (8b) boundary
        Ncb_padded  = (Ncb_padded + 7) / 8;
        Ncb_padded *= 8;
        pPerTbPrms[i].Ncb_padded = Ncb_padded;

        uint32_t codeBlockDataByteSize = (pPerTbPrms[i].K - crcPolyByteSize * 8 - pPerTbPrms[i].F + 8 - 1) / 8;

        DEBUG_PRINTF("TB %d (ueGrpIdx %d nUeGrpLayers %d nUeLayers %d) K %d F %d crcPolyByteSize %d num_CBs %d tbSize %d Zc %d bg %d parityNodes %d K_prime %d Kb %d B_prime %d K_cb %d nZpBitsPerCb %d\n", i, pPerTbPrms[i].userGroupIndex, drvdUeGrpPrms->nLayers, pPerTbPrms[i].Nl, pPerTbPrms[i].K, pPerTbPrms[i].F, crcPolyByteSize, pPerTbPrms[i].num_CBs, tbSize, pPerTbPrms[i].Zc, pPerTbPrms[i].bg, ldpcPrms.parityNodesArray[i], K_prime, ldpcPrms.KbArray[i], B_prime, K_cb, pPerTbPrms[i].nZpBitsPerCb);

        uint32_t decodedTbSize   = codeBlockDataByteSize * pPerTbPrms[i].num_CBs;
        pPerTbPrms[i].nDataBytes = decodedTbSize;

        totBitSize += decodedTbSize * 8;
        rate_match_seq_len(pPerTbPrms[i], uePrmsArray[i], *pCellDynPrm, *pUeGrpPrm, codeRateArray[i], drvdUeGrpPrms->dataCnt, drvdUeGrpPrms->dataSymLoc, drvdUeGrpPrms->dmrsCnt, drvdUeGrpPrms->dmrsSymLoc, drvdUeGrpPrms->nDmrsCdmGrpsNoData);

        // flag if uci on pusch present
        pPerTbPrms[i].uciOnPuschFlag = static_cast<uint8_t>((uePrmsArray[i].pduBitmap >> 1) & 1);
        if(pPerTbPrms[i].num_CBs > m_maxNCbsPerTb)
        {
            pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_NCBS_PERTB_OUT_OF_RANGE;
            pDynPrm->pStatusOut->ueIdx = i;
            pDynPrm->pStatusOut->cellPrmStatIdx = pCellDynPrm->cellPrmStatIdx; 
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Number of CBs per TB in cell group is {} but current max. specified for PUSCH is {}", __FUNCTION__, pPerTbPrms[i].num_CBs, m_maxNCbsPerTb);
            return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
        }
        totNCBs += pPerTbPrms[i].num_CBs; 
    }

    if(totNCBs > m_maxNCbs)
    {
        pDynPrm->pStatusOut->status = cuphyPuschStatusType_t::CUPHY_PUSCH_STATUS_NCBS_PERCELLGROUP_OUT_OF_RANGE;
        pDynPrm->pStatusOut->ueIdx = MAX_UINT16;
        pDynPrm->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "{}: Total number of CBs  in cell group is {} but current max. specified for PUSCH is {}", __FUNCTION__, totNCBs, m_maxNCbs);
        return CUPHY_STATUS_VALUE_OUT_OF_RANGE;
    }
    return CUPHY_STATUS_SUCCESS;
}

template <fmtlog::LogLevel log_level>
void PuschRx::printDynApiPrms(cuphyPuschDynPrms_t* pDynPrm)
{
    NVLOG_FMT(log_level, NVLOG_PUSCH, "===============================================");
    NVLOG_FMT(log_level, NVLOG_PUSCH, "===========print PUSCH DynApiPrms==============");
    NVLOG_FMT(log_level, NVLOG_PUSCH, "setupPhase: {}", pDynPrm->setupPhase);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "procModeBmsk: {:x}", pDynPrm->procModeBmsk);

    const cuphyPuschCellGrpDynPrm_t* pCellGrpDynPrm = pDynPrm->pCellGrpDynPrm;
    NVLOG_FMT(log_level, NVLOG_PUSCH, "nCells: {}", pCellGrpDynPrm->nCells);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "===============================================");
    for (uint16_t i = 0 ; i < pCellGrpDynPrm->nCells; i++)
    {
        NVLOG_FMT(log_level, NVLOG_PUSCH, "-->Cell[{}]", i);
        cuphyPuschCellDynPrm_t* pCellDynPrm = &pCellGrpDynPrm->pCellPrms[i];
        NVLOG_FMT(log_level, NVLOG_PUSCH, "pCellPrms: {:p}", static_cast<void*>(pCellDynPrm));
        NVLOG_FMT(log_level, NVLOG_PUSCH, "cellPrmStatIdx: {}", pCellDynPrm->cellPrmStatIdx);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "cellPrmDynIdx: {}", pCellDynPrm->cellPrmDynIdx);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "slotNum: {}", pCellDynPrm->slotNum);
    }

    NVLOG_FMT(log_level, NVLOG_PUSCH, "===============================================");
    NVLOG_FMT(log_level, NVLOG_PUSCH, "nUeGrps: {}",  pCellGrpDynPrm->nUeGrps);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "===============================================");
    for (uint16_t i=0; i< pCellGrpDynPrm->nUeGrps; i++)
    {
        NVLOG_FMT(log_level, NVLOG_PUSCH, "-->UeGrp[{}]", i);
        cuphyPuschUeGrpPrm_t* pUeGrpDynPrm = &pCellGrpDynPrm->pUeGrpPrms[i];
        NVLOG_FMT(log_level, NVLOG_PUSCH, "pUeGrpPrms: {:p}",  static_cast<void*>(pUeGrpDynPrm));
        NVLOG_FMT(log_level, NVLOG_PUSCH, "cuphyPuschUeGrpPrm->pCellPrms: {:p}",  static_cast<void*>(pUeGrpDynPrm->pCellPrm));
        NVLOG_FMT(log_level, NVLOG_PUSCH, "cuphyPuschUeGrpPrm->pCellPrms->cellPrmStatIdx: {}",  pUeGrpDynPrm->pCellPrm->cellPrmStatIdx);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "cuphyPuschUeGrpPrm->pCellPrms->cellPrmDynIdx: {}",  pUeGrpDynPrm->pCellPrm->cellPrmDynIdx);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "cuphyPuschUeGrpPrm->pCellPrms->slotNum: {}",  pUeGrpDynPrm->pCellPrm->slotNum);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "cuphyPuschUeGrpPrm->pDmrsDynPrm: {:p}",  static_cast<void*>(pUeGrpDynPrm->pDmrsDynPrm));
        NVLOG_FMT(log_level, NVLOG_PUSCH, "startPrb: {}",  pUeGrpDynPrm->startPrb);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "nPrb: {}",  pUeGrpDynPrm->nPrb);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "puschStartSym: {}",  pUeGrpDynPrm->puschStartSym);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "nPuschSym: {}",  pUeGrpDynPrm->nPuschSym);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "dmrsSymLocBmsk: {}",  pUeGrpDynPrm->dmrsSymLocBmsk);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "rssiSymLocBmsk: {}",  pUeGrpDynPrm->rssiSymLocBmsk);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "nUes: {}",  pUeGrpDynPrm->nUes);

        for (uint16_t j = 0; j < pUeGrpDynPrm->nUes; j++)
        {
            NVLOG_FMT(log_level, NVLOG_PUSCH, "pUePrmIdxs: {}",  pUeGrpDynPrm->pUePrmIdxs[j]);
        }

        const cuphyPuschDmrsPrm_t* pDmrsDynPrm = pUeGrpDynPrm->pDmrsDynPrm;
        if(pDmrsDynPrm != nullptr)
        {
            NVLOG_FMT(log_level, NVLOG_PUSCH, "dmrsAddlnPos: {}",  pDmrsDynPrm->dmrsAddlnPos);
            NVLOG_FMT(log_level, NVLOG_PUSCH, "dmrsMaxLen: {}",  pDmrsDynPrm->dmrsMaxLen);
            NVLOG_FMT(log_level, NVLOG_PUSCH, "nDmrsCdmGrpsNoData: {}",  pDmrsDynPrm->nDmrsCdmGrpsNoData);
            NVLOG_FMT(log_level, NVLOG_PUSCH, "dmrsScrmId: {}",  pDmrsDynPrm->dmrsScrmId);
        }
    }
    NVLOG_FMT(log_level, NVLOG_PUSCH, "===============================================");
    NVLOG_FMT(log_level, NVLOG_PUSCH, "nUes: {}",  pCellGrpDynPrm->nUes);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "===============================================");
    for (uint16_t i = 0; i < pCellGrpDynPrm->nUes; i++)
    {
        NVLOG_FMT(log_level, NVLOG_PUSCH, "-->UE[{}]", i);
        cuphyPuschUePrm_t* pUeDynPrm = &pCellGrpDynPrm->pUePrms[i];
        NVLOG_FMT(log_level, NVLOG_PUSCH, "pUePrms: {:p}",  static_cast<void*>(pUeDynPrm));
        NVLOG_FMT(log_level, NVLOG_PUSCH, "cuphyPuschUePrm->pUeGrpPrm: {:p}",  static_cast<void*>(pUeDynPrm->pUeGrpPrm));
        NVLOG_FMT(log_level, NVLOG_PUSCH, "pduBitmap: {}",  pUeDynPrm->pduBitmap);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "ueGrpIdx: {}",  pUeDynPrm->ueGrpIdx);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "enableTfPrcd: {}",  pUeDynPrm->enableTfPrcd);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "puschIdentity: {}",  pUeDynPrm->puschIdentity);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "groupOrSequenceHopping: {}",  pUeDynPrm->groupOrSequenceHopping);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "N_symb_slot: {}",  pUeDynPrm->N_symb_slot);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "N_slot_frame: {}",  pUeDynPrm->N_slot_frame);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "lowPaprGroupNumber: {}",  pUeDynPrm->lowPaprGroupNumber);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "lowPaprSequenceNumber: {}",  pUeDynPrm->lowPaprSequenceNumber);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "scid: {}",  pUeDynPrm->scid);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "dmrsPortBmsk: {}",  pUeDynPrm->dmrsPortBmsk);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "mcsTableIndex: {}",  pUeDynPrm->mcsTableIndex);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "mcsIndex: {}",  pUeDynPrm->mcsIndex);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "targetCodeRate: {}",  pUeDynPrm->targetCodeRate);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "qamModOrder: {}",  pUeDynPrm->qamModOrder);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "tbSizeBytes: {}",  pUeDynPrm->TBSize);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "rv: {}",  pUeDynPrm->rv);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "rnti: {}",  pUeDynPrm->rnti);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "dataScramId: {}",  pUeDynPrm->dataScramId);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "nUeLayers: {}",  pUeDynPrm->nUeLayers);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "ndi: {}",  pUeDynPrm->ndi);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "harqProcessId: {}",  pUeDynPrm->harqProcessId);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "i_lbrm: {}",  pUeDynPrm->i_lbrm);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "maxLayers: {}",  pUeDynPrm->maxLayers);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "maxQm: {}",  pUeDynPrm->maxQm);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "n_PRB_LBRM: {}",  pUeDynPrm->n_PRB_LBRM);

        if(pUeDynPrm->pUciPrms != nullptr)
        {
            cuphyUciOnPuschPrm_t* pUci = pUeDynPrm->pUciPrms;
            NVLOG_FMT(log_level, NVLOG_PUSCH, "pUciPrms: {:p}",  static_cast<void*>(pUci));
            NVLOG_FMT(log_level, NVLOG_PUSCH, "nBitsHarq: {}",  pUci->nBitsHarq);
            NVLOG_FMT(log_level, NVLOG_PUSCH, "nBitsCsi1: {}",  pUci->nBitsCsi1);
            NVLOG_FMT(log_level, NVLOG_PUSCH, "alphaScaling: {}",  pUci->alphaScaling);
            NVLOG_FMT(log_level, NVLOG_PUSCH, "betaOffsetHarqAck: {}",  pUci->betaOffsetHarqAck);
            NVLOG_FMT(log_level, NVLOG_PUSCH, "betaOffsetCsi1: {}",  pUci->betaOffsetCsi1);
            NVLOG_FMT(log_level, NVLOG_PUSCH, "betaOffsetCsi2: {}",  pUci->betaOffsetCsi2);
            NVLOG_FMT(log_level, NVLOG_PUSCH, "rankBitOffset: {}",  pUci->rankBitOffset);
            NVLOG_FMT(log_level, NVLOG_PUSCH, "nRanksBits: {}",  pUci->nRanksBits);
            NVLOG_FMT(log_level, NVLOG_PUSCH, "nCsiReports: {}",  pUci->nCsiReports);
        }
    }
    NVLOG_FMT(log_level, NVLOG_PUSCH, "===============================================");
}

template <fmtlog::LogLevel log_level>
void PuschRx::printStaticApiPrms(cuphyPuschStatPrms_t const* pStaticPrm)
{
    NVLOG_FMT(log_level, NVLOG_PUSCH, "===============================================");
    NVLOG_FMT(log_level, NVLOG_PUSCH, "=======print PUSCH printStaticApiPrms==========");
    NVLOG_FMT(log_level, NVLOG_PUSCH, "enableCfoCorrection: {}", pStaticPrm->enableCfoCorrection);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "enableToEstimation: {}", pStaticPrm->enableToEstimation);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "enablePuschTdi: {}", pStaticPrm->enablePuschTdi);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "stream_priority: {}", pStaticPrm->stream_priority);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "ldpcnIterations: {}", pStaticPrm->ldpcnIterations);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "ldpcEarlyTermination: {}", pStaticPrm->ldpcEarlyTermination);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "ldpcUseHalf: {}", pStaticPrm->ldpcUseHalf);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "ldpcAlgoIndex: {}", pStaticPrm->ldpcAlgoIndex);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "ldpcFlags: {}", pStaticPrm->ldpcFlags);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "ldpcKernelLaunch: {}", pStaticPrm->ldpcKernelLaunch);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "enableRssiMeasurement: {}", pStaticPrm->enableRssiMeasurement);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "enableSinrMeasurement: {}", pStaticPrm->enableSinrMeasurement);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "nMaxCells: {}", pStaticPrm->nMaxCells);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "nMaxCellsPerSlot: {}", pStaticPrm->nMaxCellsPerSlot);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "nMaxTbs: {}", pStaticPrm->nMaxTbs);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "nMaxCbsPerTb: {}", pStaticPrm->nMaxCbsPerTb);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "nMaxTotCbs: {}", pStaticPrm->nMaxTotCbs);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "nMaxRx: {}", pStaticPrm->nMaxRx);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "nMaxPrb: {}", pStaticPrm->nMaxPrb);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "polarDcdrListSz: {}", pStaticPrm->polarDcdrListSz);

    const cuphyCellStatPrm_t* pCellStatPrms = pStaticPrm->pCellStatPrms;
    NVLOG_FMT(log_level, NVLOG_PUSCH, "===============================================");
    NVLOG_FMT(log_level, NVLOG_PUSCH, "cuphyCellStatPrm_t:");
    NVLOG_FMT(log_level, NVLOG_PUSCH, "===============================================");

    NVLOG_FMT(log_level, NVLOG_PUSCH, "phyCellId: {}", pCellStatPrms->phyCellId);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "nRxAnt: {}", pCellStatPrms->nRxAnt);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "nTxAnt: {}", pCellStatPrms->nTxAnt);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "nPrbUlBwp: {}", pCellStatPrms->nPrbUlBwp);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "nPrbDlBwp: {}", pCellStatPrms->nPrbDlBwp);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "mu: {}", pCellStatPrms->mu);

    NVLOG_FMT(log_level, NVLOG_PUSCH, "===============================================");
    NVLOG_FMT(log_level, NVLOG_PUSCH, "nCells: {}", pStaticPrm->nMaxCells);
    NVLOG_FMT(log_level, NVLOG_PUSCH, "===============================================");
    for (uint16_t i = 0 ; i < pStaticPrm->nMaxCells; i++)
    {
        NVLOG_FMT(log_level, NVLOG_PUSCH, "-->Cell[{}]", i);
        const cuphyPuschCellStatPrm_t* pPuschCellStatPrms = &pCellStatPrms->pPuschCellStatPrms[i];
        NVLOG_FMT(log_level, NVLOG_PUSCH, "nCsirsPorts: {}", pPuschCellStatPrms->nCsirsPorts);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "N1: {}", pPuschCellStatPrms->N1);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "N2: {}", pPuschCellStatPrms->N2);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "csiReportingBand: {}", pPuschCellStatPrms->csiReportingBand);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "codebookType: {}", pPuschCellStatPrms->codebookType);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "codebookMode: {}", pPuschCellStatPrms->codebookMode);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "isCqi: {}", pPuschCellStatPrms->isCqi);
        NVLOG_FMT(log_level, NVLOG_PUSCH, "isLi: {}", pPuschCellStatPrms->isLi);
    }

    NVLOG_FMT(log_level, NVLOG_PUSCH, "===============================================");
}
