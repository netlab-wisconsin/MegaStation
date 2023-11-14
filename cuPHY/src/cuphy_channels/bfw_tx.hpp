/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#if !defined(BFW_TX_HPP_INCLUDED_)
#define BFW_TX_HPP_INCLUDED_
#include "cuphy_hdf5.hpp"



struct cuphyBfwTx
{};

class BfwTx : public cuphyBfwTx
{
public:
    enum DescriptorTypes
    {
        BFW_COEF_COMP                    = 0,
        BFW_COEF_COMP_HET_CFG_UE_GRP_MAP = BFW_COEF_COMP                    + 1,
        BFW_COEF_COMP_UE_GRP_PRMS        = BFW_COEF_COMP_HET_CFG_UE_GRP_MAP + 1,
        BFW_COEF_COMP_LAYER_PRMS         = BFW_COEF_COMP_UE_GRP_PRMS        + 1,
        N_BFW_TX_DESCR_TYPES             = BFW_COEF_COMP_LAYER_PRMS         + 1
    };

    BfwTx(cuphyBfwStatPrms_t const* pStatPrms, cudaStream_t cuStrm);
    BfwTx(BfwTx const&)            = delete;
    BfwTx& operator=(BfwTx const&) = delete;
    ~BfwTx();

    // void writeDbgBufSynch(cudaStream_t cuStrm);

    cuphyStatus_t setup(cuphyBfwDynPrms_t* pDynPrm);
    void run(uint64_t procModeBmsk);
    void destroyComponents();

    // uint32_t processApi(cuphyBfwDynPrms_t* pDynPrm, cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms, uint8_t enableRssiMeasurement);

    // debug functions:
    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    static void printStaticApiPrms(cuphyBfwStatPrms_t const* pStaticPrms);
    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    static void printDynApiPrms(cuphyBfwDynPrms_t const* pDynPrms);
    void writeDbgBufSynch(cudaStream_t cuStream);

private:
    size_t getBufferSize(cuphyBfwStatPrms_t const* pStatPrms);
    void allocateDescrs(cuphyBfwStatPrms_t const* pStatPrm);
    void createComponents(cuphyBfwStatPrms_t const* pStatPrm, cudaStream_t cuStrm);
    void createGraphExec();
    void updateGraphExec();

    // intermediate/output buffers
    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_LinearAlloc; // linear buffer holding intermediate results
#ifdef BFW_BOTH_COMP_FLOAT    
    std::vector<cuphyTensorPrm_t>             m_tPrmBfwVec;
    std::vector<cuphy::tensor_ref>            m_tRefBfwVec;
#endif
    std::vector<uint8_t*>                     m_bfwComppVec;

    cuphy::kernelDescrs<N_BFW_TX_DESCR_TYPES> m_kernelStatDescr;
    cuphy::kernelDescrs<N_BFW_TX_DESCR_TYPES> m_kernelDynDescr;

    cuphyBfwCoefCompHndl_t m_bfwCoefCompHndl;
    cuphyBfwCoefCompLaunchCfgs_t m_bfwCoefCompLaunchCfgs;

    CUgraphNode m_emptyRootNode;
    CUDA_KERNEL_NODE_PARAMS m_emptyNodePrms;

    // Paramaters: 
    cuphyBfwDynPrms_t* m_pDynPrms;
    uint8_t            m_compressBitwidth;

    // debug paramaters:
    bool               m_debugOutputFlag;
    hdf5hpp::hdf5_file m_outHdf5File;

    // CUDA stream on which are launched
    cudaStream_t m_cuStrm;

    // Graph parameters
    bool m_enableGraph = false;
    cudaGraph_t     m_graph;
    cudaGraphExec_t m_graphExec;

    CUgraphNode m_bfwCoefCompNodes[CUPHY_BFW_COEF_COMP_N_MAX_HET_CFGS];
};

#endif // !defined(BFW_TX_HPP_INCLUDED_)