/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include <string>
#include "cuphy_api.h"
#include "cuphy_internal.h"
#include "cuphy.hpp"
#include "bfw_tx.hpp"

BfwTx::BfwTx(cuphyBfwStatPrms_t const* pStatPrm, cudaStream_t cuStrm)
:   m_kernelStatDescr("BfwTxStatDescr"),
    m_kernelDynDescr("BfwTxDynDescr"),
    m_LinearAlloc(getBufferSize(pStatPrm)),
    m_compressBitwidth(pStatPrm->compressBitwidth)
{   
    m_bfwCoefCompLaunchCfgs.nCfgs = 0;
    allocateDescrs(pStatPrm); 
    createComponents(pStatPrm, cuStrm);
    createGraphExec();

    m_debugOutputFlag = false;
    if(nullptr != pStatPrm->pStatDbg)
    {
        if(pStatPrm->pStatDbg->pOutFileName != nullptr)
        {
            m_debugOutputFlag = true;
            m_outHdf5File     = hdf5hpp::hdf5_file::open(pStatPrm->pStatDbg->pOutFileName);
        }
    }
}

size_t BfwTx::getBufferSize(cuphyBfwStatPrms_t const* pStatPrms)
{
    size_t nBytesBuffer = 0;
    size_t nMaxBytesPerLayer = 0;
#ifdef BFW_BOTH_COMP_FLOAT
    nMaxBytesPerLayer += (pStatPrms->nMaxGnbAnt * pStatPrms->nMaxPrbGrps)*sizeof(float)*2;
#endif
    nMaxBytesPerLayer += ((2 * pStatPrms->compressBitwidth * pStatPrms->nMaxGnbAnt + 7)/8 + 1) * pStatPrms->nMaxPrbGrps;
    nMaxBytesPerLayer += LINEAR_ALLOC_PAD_BYTES; // TODO: is the extra padding for alignment needed here?
    nBytesBuffer += nMaxBytesPerLayer * pStatPrms->nMaxTotalLayers;
    return nBytesBuffer;
}

void BfwTx::allocateDescrs(cuphyBfwStatPrms_t const* pStatPrm)
{
    // zero-initialize
    std::array<size_t, N_BFW_TX_DESCR_TYPES> statDescrSizeBytes{};
    std::array<size_t, N_BFW_TX_DESCR_TYPES> statDescrAlignBytes{};
    std::array<size_t, N_BFW_TX_DESCR_TYPES> dynDescrSizeBytes{};
    std::array<size_t, N_BFW_TX_DESCR_TYPES> dynDescrAlignBytes{};

    size_t* pStatDescrSizeBytes  = statDescrSizeBytes.data();
    size_t* pStatDescrAlignBytes = statDescrAlignBytes.data();
    size_t* pDynDescrSizeBytes   = dynDescrSizeBytes.data();
    size_t* pDynDescrAlignBytes  = dynDescrAlignBytes.data();

    cuphyStatus_t status = cuphyGetDescrInfoBfwCoefComp(pStatPrm->nMaxUeGrps, 
                                                        pStatPrm->nMaxTotalLayers,
                                                        &pStatDescrSizeBytes[BFW_COEF_COMP],
                                                        &pStatDescrAlignBytes[BFW_COEF_COMP],
                                                        &pDynDescrSizeBytes[BFW_COEF_COMP],
                                                        &pDynDescrAlignBytes[BFW_COEF_COMP],                                                        
                                                        &pDynDescrSizeBytes[BFW_COEF_COMP_HET_CFG_UE_GRP_MAP],
                                                        &pDynDescrAlignBytes[BFW_COEF_COMP_HET_CFG_UE_GRP_MAP],
                                                        &pDynDescrSizeBytes[BFW_COEF_COMP_UE_GRP_PRMS],
                                                        &pDynDescrAlignBytes[BFW_COEF_COMP_UE_GRP_PRMS],
                                                        &pDynDescrSizeBytes[BFW_COEF_COMP_LAYER_PRMS],
                                                        &pDynDescrAlignBytes[BFW_COEF_COMP_LAYER_PRMS]);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyGetDescrInfoBfwCoefComp()");
    }

    // Allocate descriptor
    m_kernelStatDescr.alloc(statDescrSizeBytes, statDescrAlignBytes);
    m_kernelDynDescr.alloc(dynDescrSizeBytes, dynDescrAlignBytes);

#ifdef BFW_BOTH_COMP_FLOAT
    m_tRefBfwVec.resize(pStatPrm->nMaxUeGrps);
    m_tPrmBfwVec.reserve(pStatPrm->nMaxUeGrps);
#endif
    m_bfwComppVec.reserve(pStatPrm->nMaxUeGrps);
}

void BfwTx::createComponents(cuphyBfwStatPrms_t const* pStatPrm, cudaStream_t cuStrm)
{
#ifdef ENABLE_DEBUG
    NVLOGD_FMT(NVLOG_BFW, "Begin {}", __FUNCTION__);
#endif

    auto statCpuDescrStartAddrs = m_kernelStatDescr.getCpuStartAddrs();
    auto statGpuDescrStartAddrs = m_kernelStatDescr.getGpuStartAddrs();
    auto dynCpuDescrStartAddrs  = m_kernelDynDescr.getCpuStartAddrs();
    auto dynGpuDescrStartAddrs  = m_kernelDynDescr.getGpuStartAddrs();

    bool enableCpuToGpuDescrAsyncCpy = true;
    cuphyStatus_t status = cuphyCreateBfwCoefComp(&m_bfwCoefCompHndl,
                                                  enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                  pStatPrm->compressBitwidth,
                                                  pStatPrm->nMaxUeGrps, 
                                                  pStatPrm->nMaxTotalLayers, 
                                                  pStatPrm->lambda,
                                                  reinterpret_cast<void*>(statCpuDescrStartAddrs[BFW_COEF_COMP]),
                                                  reinterpret_cast<void*>(statGpuDescrStartAddrs[BFW_COEF_COMP]),
                                                  reinterpret_cast<void*>(dynCpuDescrStartAddrs[BFW_COEF_COMP]),
                                                  reinterpret_cast<void*>(dynGpuDescrStartAddrs[BFW_COEF_COMP]),
                                                  reinterpret_cast<void*>(dynCpuDescrStartAddrs[BFW_COEF_COMP_HET_CFG_UE_GRP_MAP]),
                                                  reinterpret_cast<void*>(dynGpuDescrStartAddrs[BFW_COEF_COMP_HET_CFG_UE_GRP_MAP]),
                                                  reinterpret_cast<void*>(dynCpuDescrStartAddrs[BFW_COEF_COMP_UE_GRP_PRMS]),
                                                  reinterpret_cast<void*>(dynGpuDescrStartAddrs[BFW_COEF_COMP_UE_GRP_PRMS]),
                                                  reinterpret_cast<void*>(dynCpuDescrStartAddrs[BFW_COEF_COMP_LAYER_PRMS]),
                                                  reinterpret_cast<void*>(dynGpuDescrStartAddrs[BFW_COEF_COMP_LAYER_PRMS]),
                                                  cuStrm);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        throw cuphy::cuphy_fn_exception(status, "cuphyCreateBfwCoefComp()");
    }
#ifdef ENABLE_DEBUG
    NVLOGD_FMT(NVLOG_BFW, "Done {}", __FUNCTION__);
#endif
}

void BfwTx::createGraphExec()
{
    //--------------------------------------------------------------------------
    // Create graph
    CUDA_CHECK_EXCEPTION(cudaGraphCreate(&m_graph, 0));

    std::vector<CUgraphNode> bfwCoefCompNodeDeps{};

    // Set empty node as root node
    CUPHY_CHECK(cuphySetEmptyKernelNodeParams(&m_emptyNodePrms));

    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_emptyRootNode, m_graph, bfwCoefCompNodeDeps.data(), bfwCoefCompNodeDeps.size(), &m_emptyNodePrms));
    bfwCoefCompNodeDeps.emplace_back(m_emptyRootNode);

    for(int32_t hetCfgIdx = 0; hetCfgIdx < CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS; ++hetCfgIdx)
    {
        CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_bfwCoefCompNodes[hetCfgIdx], m_graph, bfwCoefCompNodeDeps.data(), bfwCoefCompNodeDeps.size(), &(m_emptyNodePrms)));
    }

    //--------------------------------------------------------------------------
    // Instantiate graph executable
    CUDA_CHECK_EXCEPTION(cudaGraphInstantiate(&m_graphExec, m_graph, NULL, NULL, 0));

    //--------------------------------------------------------------------------
    // Set nodes in disabled state
    for(int32_t hetCfgIdx = 0; hetCfgIdx < CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS; ++hetCfgIdx)
    {
#if CUDART_VERSION >= 11060
        CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_bfwCoefCompNodes[hetCfgIdx], 0));
#else
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_bfwCoefCompNodes[hetCfgIdx], &(m_emptyNodePrms)));
#endif
    }
}

void BfwTx::destroyComponents()
{
#ifdef ENABLE_DEBUG
    NVLOGD_FMT(NVLOG_BFW, "Begin {}", __FUNCTION__);
#endif
    cuphyStatus_t status = cuphyDestroyBfwCoefComp(m_bfwCoefCompHndl);
    if(CUPHY_STATUS_SUCCESS != status)
    {
        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "cuphyDestroyBfwCoefComp() error {}", status);
    }
#ifdef ENABLE_DEBUG
    NVLOGD_FMT(NVLOG_BFW, "Done {}", __FUNCTION__);
#endif
}

BfwTx::~BfwTx()
{
    CUDA_CHECK(cudaGraphDestroy(m_graph));
    CUDA_CHECK(cudaGraphExecDestroy(m_graphExec));
    destroyComponents();
}

void BfwTx::updateGraphExec()
{
    for(int32_t hetCfgIdx = 0; hetCfgIdx < m_bfwCoefCompLaunchCfgs.nCfgs; ++hetCfgIdx)
    {
#if CUDART_VERSION >= 11060
        CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_bfwCoefCompNodes[hetCfgIdx], 1));
#endif
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_bfwCoefCompNodes[hetCfgIdx], &(m_bfwCoefCompLaunchCfgs.cfgs[hetCfgIdx].kernelNodeParamsDriver)));
    }

    for(int32_t hetCfgIdx = m_bfwCoefCompLaunchCfgs.nCfgs; hetCfgIdx < CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS; ++hetCfgIdx)
    {
#if CUDART_VERSION >= 11060
        CU_CHECK_EXCEPTION(cuGraphNodeSetEnabled(m_graphExec, m_bfwCoefCompNodes[hetCfgIdx], 0));
#else
        CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_bfwCoefCompNodes[hetCfgIdx], &(m_emptyNodePrms)));
#endif
    }
}

cuphyStatus_t BfwTx::setup(cuphyBfwDynPrms_t* pDynPrms)
{
    PUSH_RANGE("cuphySetupBfwTx", 1);
    m_pDynPrms = pDynPrms;

    cuphyBfwDynPrm_t const& dynPrm = *(pDynPrms->pDynPrm);
    m_cuStrm = pDynPrms->cuStream;
    
    bool enableCpuToGpuDescrAsyncCpy = false; // true;

    m_LinearAlloc.reset();

#ifdef BFW_BOTH_COMP_FLOAT
    m_tPrmBfwVec.resize(dynPrm.nUeGrps);
#endif
    m_bfwComppVec.resize(dynPrm.nUeGrps);
    for (int i=0; i < dynPrm.nUeGrps; i++)
    {
        cuphyBfwUeGrpPrm_t const& ueGrpPrm = dynPrm.pUeGrpPrms[i];
#ifdef BFW_BOTH_COMP_FLOAT        
        cuphyTensorPrm_t& bfwTensorPrm = m_tPrmBfwVec[ueGrpPrm.coefBufIdx];
        m_tRefBfwVec[i].desc().set(CUPHY_C_32F, ueGrpPrm.nRxAnt, ueGrpPrm.nBfLayers, ueGrpPrm.nPrbGrp, cuphy::tensor_flags::align_tight);
        m_LinearAlloc.alloc(m_tRefBfwVec[i]);
        bfwTensorPrm.desc  = m_tRefBfwVec[i].desc().handle();
        bfwTensorPrm.pAddr = m_tRefBfwVec[i].addr();
#endif
        m_bfwComppVec[i] = static_cast<uint8_t*>(m_LinearAlloc.alloc(((2 * m_compressBitwidth * ueGrpPrm.nRxAnt + 7)/8 + 1) * ueGrpPrm.nPrbGrp * ueGrpPrm.nBfLayers));
    }


    // call component setup
    cuphyStatus_t coefCompSetupStatus = cuphySetupBfwCoefComp(m_bfwCoefCompHndl,
                                                              dynPrm.nUeGrps,
                                                              dynPrm.pUeGrpPrms,
                                                              enableCpuToGpuDescrAsyncCpy ? static_cast<uint8_t>(1) : static_cast<uint8_t>(0),
                                                              pDynPrms->pDataIn->pChEstInfo,
#ifdef BFW_BOTH_COMP_FLOAT
                                                              m_tPrmBfwVec.data(),
#endif
                                                              m_bfwComppVec.data(),
                                                              &m_bfwCoefCompLaunchCfgs,
                                                              m_cuStrm);

    m_enableGraph = (pDynPrms->procModeBmsk & BFW_PROC_MODE_WITH_GRAPH) ? true : false;
    if(m_enableGraph)
    {
        updateGraphExec();
    }

    // Copy descriptor
    if(CUPHY_STATUS_SUCCESS != coefCompSetupStatus)
    {
        NVLOGE_FMT(NVLOG_BFW, AERIAL_CUPHY_EVENT, "cuphySetupBfwCoefComp() error {}", coefCompSetupStatus);
        pDynPrms->pStatusOut->status = cuphyBfwStatusType_t::CUPHY_BFW_STATUS_COEF_COMP_SETUP_ERROR;
        pDynPrms->pStatusOut->ueIdx = MAX_UINT16;
        pDynPrms->pStatusOut->cellPrmStatIdx = MAX_UINT16; 
        POP_RANGE
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    if(!enableCpuToGpuDescrAsyncCpy)
    {
        m_kernelDynDescr.asyncCpuToGpuCpy(m_cuStrm);
    }    
    POP_RANGE
    return CUPHY_STATUS_SUCCESS;
}

void BfwTx::run(uint64_t /*procModeBmsk*/)
{
    PUSH_RANGE("cuphyRunBfwTx", 1);

    if(m_enableGraph)
    {
        CUDA_CHECK_EXCEPTION(cudaGraphLaunch(m_graphExec, m_cuStrm));
    }
    else
    {
        for(uint32_t hetCfgIdx = 0; hetCfgIdx < m_bfwCoefCompLaunchCfgs.nCfgs; ++hetCfgIdx)
        {
            const CUDA_KERNEL_NODE_PARAMS& kernelNodeParamsDriverBfwCoefComp = m_bfwCoefCompLaunchCfgs.cfgs[hetCfgIdx].kernelNodeParamsDriver;
            CU_CHECK_EXCEPTION(launch_kernel(kernelNodeParamsDriverBfwCoefComp, m_cuStrm));
        }        
    }

    for(int i = 0; i < m_bfwComppVec.size(); i++)
    {
        uint8_t* pBfwCoef = m_pDynPrms->pDataOut->pBfwCoef[i];
        cuphyBfwUeGrpPrm_t const& ueGrpPrm = m_pDynPrms->pDynPrm->pUeGrpPrms[i];
        uint32_t nCompressBytes = ((2 * m_compressBitwidth * ueGrpPrm.nRxAnt + 7)/8 + 1) * ueGrpPrm.nPrbGrp * ueGrpPrm.nBfLayers;
#ifdef BFW_BOTH_COMP_FLOAT
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(pBfwCoef, m_tRefBfwVec[i].addr(), m_tRefBfwVec[i].desc().get_size_in_bytes(), cudaMemcpyDeviceToHost, m_cuStrm));
        pBfwCoef = m_pDynPrms->pDataOut->pBfwCompCoef[i];
#endif
        CUDA_CHECK_EXCEPTION(cudaMemcpyAsync(pBfwCoef, m_bfwComppVec[i], nCompressBytes, cudaMemcpyDeviceToHost, m_cuStrm));
    }
    
    POP_RANGE
}

template <fmtlog::LogLevel log_level>
void BfwTx::printStaticApiPrms(cuphyBfwStatPrms_t const* pStaticPrms)
{
    NVLOG_FMT(log_level, NVLOG_BFW,"============={}=============",__FUNCTION__);
    NVLOG_FMT(log_level, NVLOG_BFW,"lambda: {}", pStaticPrms->lambda);
    NVLOG_FMT(log_level, NVLOG_BFW,"nMaxUeGrps: {}", pStaticPrms->nMaxUeGrps);
    NVLOG_FMT(log_level, NVLOG_BFW,"nMaxTotalLayers: {}", pStaticPrms->nMaxTotalLayers);
    NVLOG_FMT(log_level, NVLOG_BFW,"compressBitwidth: {}", pStaticPrms->compressBitwidth);
    NVLOG_FMT(log_level, NVLOG_BFW,"nMaxPrbGrps: {}",pStaticPrms->nMaxPrbGrps);
    NVLOG_FMT(log_level, NVLOG_BFW,"nMaxGnbAnt: {}", pStaticPrms->nMaxGnbAnt);
    NVLOG_FMT(log_level, NVLOG_BFW,"beta: {}", pStaticPrms->beta);

    NVLOG_FMT(log_level, NVLOG_BFW,"===============================================\n");
}

template <fmtlog::LogLevel log_level>
void BfwTx::printDynApiPrms(cuphyBfwDynPrms_t const* pDynPrms)
{
    NVLOG_FMT(log_level, NVLOG_BFW,"==============={}=================",__FUNCTION__);
    NVLOG_FMT(log_level, NVLOG_BFW,"procModeBmsk: {}", pDynPrms->procModeBmsk);

    cuphyBfwUeGrpPrm_t const* pUeGrpPrms = pDynPrms->pDynPrm->pUeGrpPrms;
    uint16_t                  nUeGrps    = pDynPrms->pDynPrm->nUeGrps;
    NVLOG_FMT(log_level, NVLOG_BFW,"nUeGrps: {}", nUeGrps);
    NVLOG_FMT(log_level, NVLOG_BFW,"===============================================");
    for (uint16_t ueGrpIdx = 0 ; ueGrpIdx < nUeGrps; ueGrpIdx++)
    {
        NVLOG_FMT(log_level, NVLOG_BFW,"-->ueGrp[{}]", ueGrpIdx);
        NVLOG_FMT(log_level, NVLOG_BFW,"startPrbGrp: {}", pUeGrpPrms[ueGrpIdx].startPrbGrp);
        NVLOG_FMT(log_level, NVLOG_BFW,"nPrbGrp: {}", pUeGrpPrms[ueGrpIdx].nPrbGrp);
        NVLOG_FMT(log_level, NVLOG_BFW,"nRxAnt: {}", pUeGrpPrms[ueGrpIdx].nRxAnt);
        NVLOG_FMT(log_level, NVLOG_BFW,"nBfLayers: {}", pUeGrpPrms[ueGrpIdx].nBfLayers);
        NVLOG_FMT(log_level, NVLOG_BFW,"coefBufIdx: {}", pUeGrpPrms[ueGrpIdx].coefBufIdx);
        for(int layerIdx = 0; layerIdx < pUeGrpPrms[ueGrpIdx].nBfLayers; ++layerIdx)
        {
            NVLOG_FMT(log_level, NVLOG_BFW,"chEstInfoBufIdx[{}]: {}", layerIdx, pUeGrpPrms[ueGrpIdx].pBfLayerPrm[layerIdx].chEstInfoBufIdx);
            NVLOG_FMT(log_level, NVLOG_BFW,"ueLayerIndex[{}]: {}", layerIdx, pUeGrpPrms[ueGrpIdx].pBfLayerPrm[layerIdx].ueLayerIndex);
        }
    }
    NVLOG_FMT(log_level, NVLOG_BFW,"===============================================");
}

void BfwTx::writeDbgBufSynch(cudaStream_t cuStream)
{
    if(m_debugOutputFlag)
    {
        using cpuCuPhyBufU32_t = cuphy::buffer<uint32_t, cuphy::pinned_alloc>;
        using cpuCuPhyBufInt_t = cuphy::buffer<int, cuphy::pinned_alloc>;

        // init ueGrp paramaters:
        uint16_t nUeGrps = m_pDynPrms->pDynPrm->nUeGrps;
        uint16_t maxNumLayersPerUeGrp = 16;
        cpuCuPhyBufU32_t  startPrbGrp = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufU32_t  nPrbGrp    = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufU32_t  nRxAnt     = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufU32_t  nBfLayers  = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufU32_t  coefBufIdx = std::move(cuphy::buffer<uint32_t, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  chEstInfoBufIdx0 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  chEstInfoBufIdx1 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  chEstInfoBufIdx2 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  chEstInfoBufIdx3 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  chEstInfoBufIdx4 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  chEstInfoBufIdx5 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  chEstInfoBufIdx6 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  chEstInfoBufIdx7 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  chEstInfoBufIdx8 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  chEstInfoBufIdx9 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  chEstInfoBufIdx10 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  chEstInfoBufIdx11 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  chEstInfoBufIdx12 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  chEstInfoBufIdx13 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  chEstInfoBufIdx14 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  chEstInfoBufIdx15 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  ueLayerIndex0 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  ueLayerIndex1 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  ueLayerIndex2 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  ueLayerIndex3 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  ueLayerIndex4 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  ueLayerIndex5 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  ueLayerIndex6 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  ueLayerIndex7 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  ueLayerIndex8 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  ueLayerIndex9 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  ueLayerIndex10 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  ueLayerIndex11 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  ueLayerIndex12 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  ueLayerIndex13 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  ueLayerIndex14 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));
        cpuCuPhyBufInt_t  ueLayerIndex15 = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(nUeGrps));

        // copy ueGrp paramaters:
        cuphyBfwUeGrpPrm_t const* pUeGrpPrms = m_pDynPrms->pDynPrm->pUeGrpPrms;
        for(int ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
        {
            startPrbGrp[ueGrpIdx] = pUeGrpPrms[ueGrpIdx].startPrbGrp;
            nPrbGrp[ueGrpIdx] = pUeGrpPrms[ueGrpIdx].nPrbGrp;
            nRxAnt[ueGrpIdx] = pUeGrpPrms[ueGrpIdx].nRxAnt;
            nBfLayers[ueGrpIdx] = pUeGrpPrms[ueGrpIdx].nBfLayers;
            coefBufIdx[ueGrpIdx] = pUeGrpPrms[ueGrpIdx].coefBufIdx;

            cpuCuPhyBufInt_t ueGrpChEstInfoBufIdxs = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(maxNumLayersPerUeGrp));
            cpuCuPhyBufInt_t ueGrpChEstUeLayerIdxs = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(maxNumLayersPerUeGrp));

            for(int layerIdx = 0; layerIdx < maxNumLayersPerUeGrp; ++layerIdx)
            {
                ueGrpChEstInfoBufIdxs[layerIdx] = -1;
                ueGrpChEstUeLayerIdxs[layerIdx] = -1;
            }
            for(int layerIdx = 0; layerIdx < pUeGrpPrms[ueGrpIdx].nBfLayers; ++layerIdx)
            {
                ueGrpChEstInfoBufIdxs[layerIdx] = pUeGrpPrms[ueGrpIdx].pBfLayerPrm[layerIdx].chEstInfoBufIdx;
                ueGrpChEstUeLayerIdxs[layerIdx] = pUeGrpPrms[ueGrpIdx].pBfLayerPrm[layerIdx].ueLayerIndex;
            }
            chEstInfoBufIdx0[ueGrpIdx] = ueGrpChEstInfoBufIdxs[0];
            chEstInfoBufIdx1[ueGrpIdx] = ueGrpChEstInfoBufIdxs[1];
            chEstInfoBufIdx2[ueGrpIdx] = ueGrpChEstInfoBufIdxs[2];
            chEstInfoBufIdx3[ueGrpIdx] = ueGrpChEstInfoBufIdxs[3];
            chEstInfoBufIdx4[ueGrpIdx] = ueGrpChEstInfoBufIdxs[4];
            chEstInfoBufIdx5[ueGrpIdx] = ueGrpChEstInfoBufIdxs[5];
            chEstInfoBufIdx6[ueGrpIdx] = ueGrpChEstInfoBufIdxs[6];
            chEstInfoBufIdx7[ueGrpIdx] = ueGrpChEstInfoBufIdxs[7];
            chEstInfoBufIdx8[ueGrpIdx] = ueGrpChEstInfoBufIdxs[8];
            chEstInfoBufIdx9[ueGrpIdx] = ueGrpChEstInfoBufIdxs[9];
            chEstInfoBufIdx10[ueGrpIdx] = ueGrpChEstInfoBufIdxs[10];
            chEstInfoBufIdx11[ueGrpIdx] = ueGrpChEstInfoBufIdxs[11];
            chEstInfoBufIdx12[ueGrpIdx] = ueGrpChEstInfoBufIdxs[12];
            chEstInfoBufIdx13[ueGrpIdx] = ueGrpChEstInfoBufIdxs[13];
            chEstInfoBufIdx14[ueGrpIdx] = ueGrpChEstInfoBufIdxs[14];
            chEstInfoBufIdx15[ueGrpIdx] = ueGrpChEstInfoBufIdxs[15];

            ueLayerIndex0[ueGrpIdx] = ueGrpChEstUeLayerIdxs[0];
            ueLayerIndex1[ueGrpIdx] = ueGrpChEstUeLayerIdxs[1];
            ueLayerIndex2[ueGrpIdx] = ueGrpChEstUeLayerIdxs[2];
            ueLayerIndex3[ueGrpIdx] = ueGrpChEstUeLayerIdxs[3];
            ueLayerIndex4[ueGrpIdx] = ueGrpChEstUeLayerIdxs[4];
            ueLayerIndex5[ueGrpIdx] = ueGrpChEstUeLayerIdxs[5];
            ueLayerIndex6[ueGrpIdx] = ueGrpChEstUeLayerIdxs[6];
            ueLayerIndex7[ueGrpIdx] = ueGrpChEstUeLayerIdxs[7];
            ueLayerIndex8[ueGrpIdx] = ueGrpChEstUeLayerIdxs[8];
            ueLayerIndex9[ueGrpIdx] = ueGrpChEstUeLayerIdxs[9];
            ueLayerIndex10[ueGrpIdx] = ueGrpChEstUeLayerIdxs[10];
            ueLayerIndex11[ueGrpIdx] = ueGrpChEstUeLayerIdxs[11];
            ueLayerIndex12[ueGrpIdx] = ueGrpChEstUeLayerIdxs[12];
            ueLayerIndex13[ueGrpIdx] = ueGrpChEstUeLayerIdxs[13];
            ueLayerIndex14[ueGrpIdx] = ueGrpChEstUeLayerIdxs[14];
            ueLayerIndex15[ueGrpIdx] = ueGrpChEstUeLayerIdxs[15];
        }


        // write user grp paramaters to H5:
        cuphy::write_HDF5_dataset(m_outHdf5File, "startPrbGrp", CUPHY_R_32U, nUeGrps, startPrbGrp.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "nPrbGrp", CUPHY_R_32U, nUeGrps, nPrbGrp.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "nRxAnt", CUPHY_R_32U, nUeGrps, nRxAnt.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "nBfLayers", CUPHY_R_32U, nUeGrps, nBfLayers.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "coefBufIdx", CUPHY_R_32U, nUeGrps, coefBufIdx.addr());

        cuphy::write_HDF5_dataset(m_outHdf5File, "chEstInfoBufIdx0", CUPHY_R_32I, nUeGrps,    chEstInfoBufIdx0.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "chEstInfoBufIdx1", CUPHY_R_32I, nUeGrps,    chEstInfoBufIdx1.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "chEstInfoBufIdx2", CUPHY_R_32I, nUeGrps,    chEstInfoBufIdx2.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "chEstInfoBufIdx3", CUPHY_R_32I, nUeGrps,    chEstInfoBufIdx3.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "chEstInfoBufIdx4", CUPHY_R_32I, nUeGrps,    chEstInfoBufIdx4.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "chEstInfoBufIdx5", CUPHY_R_32I, nUeGrps,    chEstInfoBufIdx5.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "chEstInfoBufIdx6", CUPHY_R_32I, nUeGrps,    chEstInfoBufIdx6.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "chEstInfoBufIdx7", CUPHY_R_32I, nUeGrps,    chEstInfoBufIdx7.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "chEstInfoBufIdx8", CUPHY_R_32I, nUeGrps,    chEstInfoBufIdx8.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "chEstInfoBufIdx9", CUPHY_R_32I, nUeGrps,    chEstInfoBufIdx9.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "chEstInfoBufIdx10", CUPHY_R_32I, nUeGrps,    chEstInfoBufIdx10.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "chEstInfoBufIdx11", CUPHY_R_32I, nUeGrps,    chEstInfoBufIdx11.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "chEstInfoBufIdx12", CUPHY_R_32I, nUeGrps,    chEstInfoBufIdx12.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "chEstInfoBufIdx13", CUPHY_R_32I, nUeGrps,    chEstInfoBufIdx13.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "chEstInfoBufIdx14", CUPHY_R_32I, nUeGrps,    chEstInfoBufIdx14.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "chEstInfoBufIdx15", CUPHY_R_32I, nUeGrps,    chEstInfoBufIdx15.addr());

        cuphy::write_HDF5_dataset(m_outHdf5File, "ueLayerIndex0", CUPHY_R_32I, nUeGrps,    ueLayerIndex0.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "ueLayerIndex1", CUPHY_R_32I, nUeGrps,    ueLayerIndex1.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "ueLayerIndex2", CUPHY_R_32I, nUeGrps,    ueLayerIndex2.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "ueLayerIndex3", CUPHY_R_32I, nUeGrps,    ueLayerIndex3.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "ueLayerIndex4", CUPHY_R_32I, nUeGrps,    ueLayerIndex4.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "ueLayerIndex5", CUPHY_R_32I, nUeGrps,    ueLayerIndex5.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "ueLayerIndex6", CUPHY_R_32I, nUeGrps,    ueLayerIndex6.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "ueLayerIndex7", CUPHY_R_32I, nUeGrps,    ueLayerIndex7.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "ueLayerIndex8", CUPHY_R_32I, nUeGrps,    ueLayerIndex8.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "ueLayerIndex9", CUPHY_R_32I, nUeGrps,    ueLayerIndex9.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "ueLayerIndex10", CUPHY_R_32I, nUeGrps,    ueLayerIndex10.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "ueLayerIndex11", CUPHY_R_32I, nUeGrps,    ueLayerIndex11.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "ueLayerIndex12", CUPHY_R_32I, nUeGrps,    ueLayerIndex12.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "ueLayerIndex13", CUPHY_R_32I, nUeGrps,    ueLayerIndex13.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "ueLayerIndex14", CUPHY_R_32I, nUeGrps,    ueLayerIndex14.addr());
        cuphy::write_HDF5_dataset(m_outHdf5File, "ueLayerIndex15", CUPHY_R_32I, nUeGrps,    ueLayerIndex15.addr());

        // Copy beamforming weights (output) to H5:
        for(int ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
        {
#ifdef BFW_BOTH_COMP_FLOAT
            cuphyTensorPrm_t& tPrmBfwCoef = m_tPrmBfwVec[ueGrpIdx];
            cuphy::write_HDF5_dataset(m_outHdf5File, tPrmBfwCoef, std::string("bfw" + std::to_string(ueGrpIdx)).c_str(), cuStream);
#endif
            cuphyBfwUeGrpPrm_t const& ueGrpPrm = pUeGrpPrms[ueGrpIdx];
            cuphy::tensor_ref tRefBfwCompCoef;
            tRefBfwCompCoef.desc().set(CUPHY_R_8U, (2 * m_compressBitwidth * ueGrpPrm.nRxAnt + 7)/8 + 1, ueGrpPrm.nPrbGrp, ueGrpPrm.nBfLayers, cuphy::tensor_flags::align_tight);
            cuphyTensorPrm_t tPrmBfwCompCoef;
            tPrmBfwCompCoef.desc  = tRefBfwCompCoef.desc().handle();
            tPrmBfwCompCoef.pAddr = m_bfwComppVec[ueGrpIdx];
            cuphy::write_HDF5_dataset(m_outHdf5File, tPrmBfwCompCoef, std::string("bfwComp" + std::to_string(ueGrpIdx)).c_str(), cuStream);
        }

        // determine unqiue chEstInfoBufIdxs
        std::vector<uint16_t> uniqueChestInfoBufIdxs;
        uint16_t maxChestInfoBufIdx = 0;
        uint16_t nUniqueChestInfoBufIdxs = 0;
        for(int ueGrpIdx = 0; ueGrpIdx < nUeGrps; ++ueGrpIdx)
        {
            for(int layerIdx = 0; layerIdx < pUeGrpPrms[ueGrpIdx].nBfLayers; ++layerIdx)
            {
                uint16_t chEstInfoBufIdx = pUeGrpPrms[ueGrpIdx].pBfLayerPrm[layerIdx].chEstInfoBufIdx;
                if(chEstInfoBufIdx > maxChestInfoBufIdx)
                {
                    maxChestInfoBufIdx = chEstInfoBufIdx;
                }
                bool isUnique = true;
                for(int i = 0; i < nUniqueChestInfoBufIdxs; ++i)
                {
                    if(chEstInfoBufIdx == uniqueChestInfoBufIdxs[i])
                    {
                        isUnique = false;
                        break;
                    }
                }
                if(isUnique)
                {
                    nUniqueChestInfoBufIdxs += 1;
                    uniqueChestInfoBufIdxs.emplace_back(chEstInfoBufIdx);
                }
            }
        }

        // save called ChEst buffers and start Prbs:
        cpuCuPhyBufInt_t  chEstBuffsStartPrbGrps = std::move(cuphy::buffer<int, cuphy::pinned_alloc>(maxChestInfoBufIdx + 1));
        for(int i = 0; i < maxChestInfoBufIdx; ++i)
        {
            chEstBuffsStartPrbGrps[i] = -1;
        }

        for(int i = 0; i < nUniqueChestInfoBufIdxs; ++i)
        {
            uint16_t chestInfoBufIdx = uniqueChestInfoBufIdxs[i];
            cuphySrsChEstBuffInfo_t chEstInfo = m_pDynPrms->pDataIn->pChEstInfo[chestInfoBufIdx];

            chEstBuffsStartPrbGrps[chestInfoBufIdx] = chEstInfo.startPrbGrp;
            cuphy::write_HDF5_dataset(m_outHdf5File, chEstInfo.tChEstBuffer, std::string("chEstBuff" + std::to_string(chestInfoBufIdx)).c_str(), cuStream);
        }
        cuphy::write_HDF5_dataset(m_outHdf5File, "chEstBuffsStartPrbGrps", CUPHY_R_32I, maxChestInfoBufIdx + 1,  chEstBuffsStartPrbGrps.addr());
    }
}


////////////////////////////////////////////////////////////////////////
// Pipeline API functions

cuphyStatus_t CUPHYWINAPI cuphyCreateBfwTx(cuphyBfwTxHndl_t*         pBfwTxHndl, 
                                           cuphyBfwStatPrms_t const* pStatPrms, 
                                           cudaStream_t              cuStream)
{
    if(!pBfwTxHndl || !pStatPrms || !cuStream)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    *pBfwTxHndl = nullptr;
    return cuphy::tryCallableAndCatch([&]
    {
        if (pStatPrms->pStatDbg->enableApiLogging) { 
            BfwTx::printStaticApiPrms(pStatPrms);
        }
        *pBfwTxHndl = static_cast<cuphyBfwTxHndl_t>(new BfwTx(pStatPrms, cuStream));
    });
}

cuphyStatus_t CUPHYWINAPI cuphySetupBfwTx(cuphyBfwTxHndl_t   bfwTxHndl, 
                                          cuphyBfwDynPrms_t* pDynPrms)
{
    if(!bfwTxHndl || !pDynPrms)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
        if (pDynPrms->pDynDbg->enableApiLogging) { 
            BfwTx::printDynApiPrms(pDynPrms);
        }
        BfwTx* p = static_cast<BfwTx*>(bfwTxHndl);
        return p->setup(pDynPrms);
    });
 
}

cuphyStatus_t CUPHYWINAPI cuphyRunBfwTx(cuphyBfwTxHndl_t bfwTxHndl, 
                                        uint64_t         procModeBmsk)
{
    if(!bfwTxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    
    return cuphy::tryCallableAndCatch([&]
    {
        BfwTx* p = static_cast<BfwTx*>(bfwTxHndl);
        p->run(procModeBmsk);
    });
}

cuphyStatus_t CUPHYWINAPI cuphyDestroyBfwTx(cuphyBfwTxHndl_t bfwTxHndl)
{
    if(!bfwTxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    BfwTx* p = static_cast<BfwTx*>(bfwTxHndl);
    delete p;

    return CUPHY_STATUS_SUCCESS;
}

 //////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
// cuphyWriteDbgBufSynch()

cuphyStatus_t CUPHYWINAPI cuphyWriteDbgBufSynchBfw(cuphyBfwTxHndl_t bfwTxHndl, cudaStream_t cuStream)
{
    if(!bfwTxHndl)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
        BfwTx* p = static_cast<BfwTx*>(bfwTxHndl);
        p->writeDbgBufSynch(cuStream);
        //p->printInfo();
    });
}
