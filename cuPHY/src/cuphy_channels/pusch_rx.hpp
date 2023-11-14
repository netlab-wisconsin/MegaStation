/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#if !defined(PUSCH_RX_HPP_INCLUDED_)
#define PUSCH_RX_HPP_INCLUDED_

#include "pusch_utils.hpp"
#include <vector>
#include <string>
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "tensor_desc.hpp"

// for memory tracing
#include "memtrace.h"

#define LLR_FP16
#define FRONT_END_DESCR 1

// ldpc multi-stream test for huge TB, temporary hack
#define SPLIT_LDPC 0

static constexpr unsigned int BYTES_PER_WORD               = sizeof(uint32_t) / sizeof(uint8_t);
static constexpr uint32_t     N_MAX_LDPC_HET_CFGS          = 32; // maximum number of allowed heterogenous workload configs for LDPC
static constexpr uint32_t     N_INTERP_DMRS_TONES_PER_GRID = CUPHY_N_TONES_PER_PRB;

struct cuphyPuschRx
{};

class PuschRx : public cuphyPuschRx {
public:
    enum DescriptorTypes
    {
        PUSCH_CH_EST                        = 0,
        PUSCH_NOISE_INTF_EST                = PUSCH_CH_EST + CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST,
        PUSCH_CH_EQ_COEF                    = PUSCH_NOISE_INTF_EST + 1,
        PUSCH_CH_EQ_SOFT_DEMAP              = PUSCH_CH_EQ_COEF + CUPHY_PUSCH_RX_MAX_N_TIME_CH_EQ,
        PUSCH_RATE_MATCH                    = PUSCH_CH_EQ_SOFT_DEMAP + 1,
        PUSCH_LDPC_DEC                      = PUSCH_RATE_MATCH + 1,
        PUSCH_CRC                           = PUSCH_LDPC_DEC + 1,
        PUSCH_CFO_TA_EST                    = PUSCH_CRC + 1,
        PUSCH_RSSI                          = PUSCH_CFO_TA_EST + 1,
        PUSCH_RSRP                          = PUSCH_RSSI + 1,
        PUSCH_FRONT_END_PARAMS              = PUSCH_RSRP + 1,
        PUSCH_SEG_UCI_LLRS0                 = PUSCH_FRONT_END_PARAMS + 1,
        PUSCH_SEG_EARLY_UCI_LLRS0           = PUSCH_SEG_UCI_LLRS0 + 1,
        PUSCH_SEG_UCI_LLRS2                 = PUSCH_SEG_EARLY_UCI_LLRS0 + 1,
        PUSCH_UCI_CSI2_CTRL                 = PUSCH_SEG_UCI_LLRS2 + 1,
        POL_COMP_CW_TREE                    = PUSCH_UCI_CSI2_CTRL + 1,
        POL_COMP_CW_TREE_ADDRS              = POL_COMP_CW_TREE + 1,
        POL_SEG_DERM_DEITL                  = POL_COMP_CW_TREE_ADDRS + 1,
        POL_SEG_DERM_DEITL_CW_ADDRS         = POL_SEG_DERM_DEITL + 1,
        POL_SEG_DERM_DEITL_UCI_ADDRS        = POL_SEG_DERM_DEITL_CW_ADDRS + 1,
        POL_DECODE                          = POL_SEG_DERM_DEITL_UCI_ADDRS + 1,
        POL_DECODE_LLR_ADDRS                = POL_DECODE + 1,
        POL_DECODE_CB_ADDRS                 = POL_DECODE_LLR_ADDRS + 1,
        LIST_POL_DECODE_SCRATCH_ADDRS       = POL_DECODE_CB_ADDRS + 1,
        POL_COMP_CW_TREE_CSI2               = LIST_POL_DECODE_SCRATCH_ADDRS + 1,
        POL_COMP_CW_TREE_ADDRS_CSI2         = POL_COMP_CW_TREE_CSI2 + 1,
        POL_SEG_DERM_DEITL_CSI2             = POL_COMP_CW_TREE_ADDRS_CSI2 + 1,
        POL_SEG_DERM_DEITL_CW_ADDRS_CSI2    = POL_SEG_DERM_DEITL_CSI2 + 1,
        POL_SEG_DERM_DEITL_UCI_ADDRS_CSI2   = POL_SEG_DERM_DEITL_CW_ADDRS_CSI2 + 1,
        POL_DECODE_CSI2                     = POL_SEG_DERM_DEITL_UCI_ADDRS_CSI2 + 1,
        POL_DECODE_LLR_ADDRS_CSI2           = POL_DECODE_CSI2 + 1,
        POL_DECODE_CB_ADDRS_CSI2            = POL_DECODE_LLR_ADDRS_CSI2 + 1,
        LIST_POL_DECODE_SCRATCH_ADDRS_CSI2  = POL_DECODE_CB_ADDRS_CSI2 + 1,
        POL_COMP_CW_TREE_EARLY              = LIST_POL_DECODE_SCRATCH_ADDRS_CSI2 + 1,
        POL_COMP_CW_TREE_ADDRS_EARLY        = POL_COMP_CW_TREE_EARLY + 1,
        POL_SEG_DERM_DEITL_EARLY            = POL_COMP_CW_TREE_ADDRS_EARLY + 1,
        POL_SEG_DERM_DEITL_CW_ADDRS_EARLY   = POL_SEG_DERM_DEITL_EARLY + 1,
        POL_SEG_DERM_DEITL_UCI_ADDRS_EARLY  = POL_SEG_DERM_DEITL_CW_ADDRS_EARLY + 1,
        POL_DECODE_EARLY                    = POL_SEG_DERM_DEITL_UCI_ADDRS_EARLY + 1,
        POL_DECODE_LLR_ADDRS_EARLY          = POL_DECODE_EARLY + 1,
        POL_DECODE_CB_ADDRS_EARLY           = POL_DECODE_LLR_ADDRS_EARLY  + 1,
        LIST_POL_DECODE_SCRATCH_ADDRS_EARLY = POL_DECODE_CB_ADDRS_EARLY + 1,
        SPX_DECODE                          = LIST_POL_DECODE_SCRATCH_ADDRS_EARLY + 1,
        RM_DECODE                           = SPX_DECODE + 1,
        SPX_DECODE_CSI2                     = RM_DECODE + 1,
        RM_DECODE_CSI2                      = SPX_DECODE_CSI2 + 1,
        SPX_DECODE_EARLY                    = RM_DECODE_CSI2 + 1,
        RM_DECODE_EARLY                     = SPX_DECODE_EARLY + 1,
        N_PUSCH_DESCR_TYPES                 = RM_DECODE_EARLY + 1
    };
    struct OutputParams
    {
        // flag to copy outputs to host (cpu)
        bool                         cpuCopyOn;

        // size of outputs
        uint32_t                     totNumTbs;
        uint32_t                     totNumCbs;
        uint32_t                     totNumPayloadBytes;
        uint32_t                     totNumUciSegs;
        uint32_t                     totNumUciPayloadBytes;

        // device (GPU) output addresses
        uint32_t* pCbCrcsDevice;
        uint32_t* pTbCrcsDevice;
        uint8_t*  pTbPayloadsDevice;
        float*    pTaEstsDevice;
        float*    pRssiDevice;
        float*    pRsrpDevice;
        float*    pNoiseVarPreEqDevice;
        float*    pNoiseVarPostEqDevice;
        float*    pSinrPreEqDevice;
        float*    pSinrPostEqDevice;
        float*    pCfoHzDevice;
        uint8_t*  pUciPayloadsDevice;
        uint8_t*  pUciCrcFlagsDevice_csi2;
        uint8_t*  pUciCrcFlagsDevice;
        uint8_t*  pUciCrcFlagsDevice_early;
        uint16_t* pNumCsi2BitsDevice;

        uint8_t*  pHarqDetectionStatusDevice;
        uint8_t*  pCsiP1DetectionStatusDevice;
        uint8_t*  pCsiP2DetectionStatusDevice;
        uint8_t*  pUciDTXsDevice;

        // host (CPU) output addresses
        uint32_t*                    pCbCrcsHost;
        uint32_t*                    pTbCrcsHost;
        uint8_t*                     pTbPayloadsHost;
        float*                       pTaEstsHost;
        float*                       pRssiHost;
        float*                       pRsrpHost;
        float*                       pNoiseVarPreEqHost;
        float*                       pNoiseVarPostEqHost;
        float*                       pSinrPreEqHost;
        float*                       pSinrPostEqHost;
        float*                       pCfoHzHost;
        uint8_t*                     pUciPayloadsHost;
        uint8_t*                     pUciCrcFlagsHost;
        uint16_t*                    pNumCsi2BitsHost;

        uint8_t*                     pHarqDetectionStatusHost;
        uint8_t*                     pCsiP1DetectionStatusHost;
        uint8_t*                     pCsiP2DetectionStatusHost;
        //uint8_t*                     pUciDTXsHost;

        cuphyUciOnPuschOutOffsets_t* pUciOnPuschOutOffsets;

        // output parameters
        bool                         debugOutputFlag;
        hdf5hpp::hdf5_file           outHdf5File;
    };

    enum H2DDataTypes
    {
        // note that PUSCH_TB_PRMS is assumed to be the first item in this list and should not be reordered
        PUSCH_TB_PRMS            = 0,
        PUSCH_SPX_PRMS           = PUSCH_TB_PRMS + 1,
        PUSCH_RM_CW_PRMS         = PUSCH_SPX_PRMS + 1,
        PUSCH_UCI_SEG_PRMS       = PUSCH_RM_CW_PRMS + 1,
        PUSCH_UCI_CW_PRMS        = PUSCH_UCI_SEG_PRMS + 1,
        PUSCH_UCI_SEG_CSI2_PRMS  = PUSCH_UCI_CW_PRMS + 1,
        PUSCH_UCI_CW_CSI2_PRMS   = PUSCH_UCI_SEG_CSI2_PRMS + 1,
        PUSCH_SPX_CSI2_PRMS      = PUSCH_UCI_CW_CSI2_PRMS + 1,
        PUSCH_RM_CW_CSI2_PRMS    = PUSCH_SPX_CSI2_PRMS + 1,
        PUSCH_UCI_SEG_EARLY_PRMS = PUSCH_RM_CW_CSI2_PRMS + 1,
        PUSCH_UCI_CW_EARLY_PRMS  = PUSCH_UCI_SEG_EARLY_PRMS + 1,
        PUSCH_SPX_EARLY_PRMS     = PUSCH_UCI_CW_EARLY_PRMS + 1,
        PUSCH_RM_CW_EARLY_PRMS   = PUSCH_SPX_EARLY_PRMS + 1,
        N_PUSCH_H2D_DATA_TYPES   = PUSCH_RM_CW_EARLY_PRMS + 1
    };
    template<size_t N_DATA> using dataBundle = cuphy::kernelDescrs<N_DATA>;

    PuschRx(cuphyPuschStatPrms_t const* pStatPrms, cudaStream_t cuStream);
    PuschRx(PuschRx const&) = delete;
    PuschRx& operator=(PuschRx const&) = delete;
    ~PuschRx();

    void copyOutputToCPU(cudaStream_t cuStrm);

    // determine HARQ/CSI part 1/CSI part 2 detection status
    // void detStatus();
    
    void writeDbgBufSynch(cudaStream_t cuStrm);

    cuphyStatus_t setup(cuphyPuschDynPrms_t* pDynPrm);
    void run(cuphyPuschRunPhase_t runPhase);

    uint32_t expandFrontEndParameters(cuphyPuschDynPrms_t* pDynPrm, cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms, uint8_t enableRssiMeasurement);
    cuphyStatus_t expandBackEndParameters(cuphyPuschDynPrms_t* pDynPrm, cuphyPuschRxUeGrpPrms_t* pDrvdUeGrpPrms, PerTbParams* pPerTbPrms, cuphyLDPCParams& ldpcPrms);
    const void* getMemoryTracker();

    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    static void printDynApiPrms(cuphyPuschDynPrms_t* pDynPrm);
    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    static void printStaticApiPrms(cuphyPuschStatPrms_t const* pStaticPrm);
    template <typename T>
    void copyTensorRef2Info(cuphy::tensor_ref& tRef, T& tInfo)
    {
        tInfo.pAddr              = tRef.addr();
        const tensor_desc& tDesc = static_cast<const tensor_desc&>(*(tRef.desc().handle()));
        tInfo.elemType           = tDesc.type();
        std::copy_n(tDesc.layout().strides.begin(), std::extent<decltype(tInfo.strides)>::value, tInfo.strides);
    }

private:
    cuphyMemoryFootprint m_memoryFootprint;

    // setup functions
    cuphyStatus_t   setupCmnPhase1(cuphyPuschDynPrms_t* pDynPrm);
    void   setupCmnPhase2(cuphyPuschDynPrms_t* pDynPrm);
    void   allocateDeviceMemory(cuphyPuschDynPrms_t* pDynPrm);
    void   allocateDescr(void);
    void   allocateInputBuf(uint32_t nMaxTbs, uint32_t maxNumTbsSupported);
    void   updateInputBuf(void);
    size_t getBufferSize(cuphyPuschStatPrms_t const* pStatPrms);
    void   expandUciParameters(bool updateOnlyNumInputPrms);
    void   expandUciCodingPrms(uint32_t nInfoBits, uint32_t nRmBits, uint8_t Qm, float DTXthreshold, bool updateOnlyNumInputPrms,
    uint16_t& nRmCws, cuphyRmCwPrm_t* pRmCwPrms, uint16_t& nSpxCws, cuphySimplexCwPrm_t* pSpxCwPrms,
    uint16_t& nPolUciSegs, uint16_t& nPolCbs, cuphyPolarUciSegPrm_t* pUciSegPrms, cuphyPolarCwPrm_t* pUciCwPrms);

    void   allocAndLinkPolBuffers(cuphyPolarUciSegPrm_t&     uciSegPrms,
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
                                     std::vector<uint32_t*>& cbEstAddrVec);

    // component functions (non-LDPC)
    void createComponents(cudaStream_t cuStrm,
                          int          rmFPconfig     = 3,
                          int          descramblingOn = 1);
    cuphyStatus_t setupComponents(bool enableCpuToGpuDescrAsyncCpy, cuphyPuschDynPrms_t* pDynPrm);
    void destroyComponents();

    // LDPC component functions
    void prepareLDPCStreamsTB();
    void launchLDPCStreamsTensor(cudaStream_t strm);
    void launchLDPCStreamsTB(cudaStream_t strm);

    // graph functions
    void          createGraph();
    void          addEarlyHarqNodes(std::vector<CUgraphNode>* currNodeDeps, std::vector<CUgraphNode>* nextNodeDeps);
    cuphyStatus_t updateGraph();
    void          updateEarlyHarqNodes(bool enableNodes);
    void          disableAllEarlyHarqNodes();

    // early-Harq kernel launch
    void uciEarlyHarqKernelLaunch();

    // pipeline inputs
    cuphyTensorPrm_t  m_tPrmDataRx;

    // pipeline parameters
    const cuphyPuschStatPrms_t m_cuphyPuschStatPrms;
    cuphyPuschDynPrms_t        m_cuphyPuschDynPrms;
    cuphyPuschCellGrpDynPrm_t  m_cuphyPuschCellGrpDynPrm;
    cuphyLDPCParams            m_ldpcPrms;
    uint32_t                   m_nMaxPrb;
    uint32_t                   m_maxNPrbAlloc;
    uint32_t                   m_maxNRx;
    cuphyPuschRxUeGrpPrms_t*   m_drvdUeGrpPrmsCpu;
    cuphyPuschRxUeGrpPrms_t*   m_drvdUeGrpPrmsGpu;
    cuphyChEstSettings         m_chEstSettings;
    cuphyChEqParams            m_chEqPrms;
    uint32_t*                  m_harqBufferSizeInBytes;

    PerTbParams*                                                m_pTbPrmsCpu;
    PerTbParams*                                                m_pTbPrmsGpu;
    cuphy::buffer<cuphyPuschCellStatPrm_t, cuphy::device_alloc> m_puschCellStatPrmBufGpu;
    cuphy::buffer<cuphyPuschCellStatPrm_t, cuphy::pinned_alloc> m_puschCellStatPrmCpu;
    const std::vector<cuphyCellStatPrm_t>                       m_cuphyCellStatPrmVecCpu;

    // CPU to GPU data copies
    dataBundle<N_PUSCH_H2D_DATA_TYPES> m_h2dBuffer;
    // zero-initialize
    std::array<size_t, N_PUSCH_H2D_DATA_TYPES> m_inputBufSizeBytes{};
    std::array<size_t, N_PUSCH_H2D_DATA_TYPES> m_inputBufAlignBytes{};

    // intermediate/output buffers
    static constexpr uint32_t LINEAR_ALLOC_PAD_BYTES = 128;
    cuphy::linear_alloc<LINEAR_ALLOC_PAD_BYTES, cuphy::device_alloc> m_LinearAlloc; // linear buffer holding intermediate results

    std::vector<cuphyTensorPrm_t>  m_tPrmLLRVec, m_tPrmLLRCdm1Vec;
    std::vector<cuphy::tensor_ref> m_tRefDataRx, m_tRefLLRVec, m_tRefLLRCdm1Vec, m_tRefHEstVec, m_tRefPerPrbNoiseVarVec, m_tRefLwInvVec, m_tRefChEstDbgVec, m_tRefCfoEstVec, m_tRefReeDiagInvVec, m_tRefDataEqVec, m_tRefDataEqDftVec, m_tRefDataEqDftIntermediateVec, m_tRefDftBluesteinWorkspaceTimeVec, m_tRefDftBluesteinWorkspaceFreqVec, m_tRefCoefVec, m_tRefEqDbgVec;

    cuphy::tensor_ref m_tRefCfoPhaseRot, m_tRefTaPhaseRot, m_tRefTaEst, m_tRefCfoTaEstInterCtaSyncCnt, m_tRefCfoEstInterCtaSyncCnt, m_tRefTaEstInterCtaSyncCnt;
    cuphy::tensor_ref m_tRefRssi, m_tRefRssiFull, m_tRefRssiInterCtaSyncCnt;
    cuphy::tensor_ref m_tRefRsrp, m_tRefRsrpInterCtaSyncCnt, m_tRefNoiseVarPreEq, m_tRefNoiseVarPostEq, m_tRefNoiseIntfEstInterCtaSyncCnt, m_tRefSinrPreEq, m_tRefSinrPostEq;
    cuphy::tensor_ref m_tRefCfoHz;
    void**            m_pHarqBuffers;
    void*             d_pLDPCOut;
    OutputParams      m_outputPrms; // CPU and GPU adreasses of output buffers.

    // Component handles (non-LDPC)
    cuphyPuschRxChEstHndl_t        m_chEstHndl;
    cuphyPuschRxNoiseIntfEstHndl_t m_noiseIntfEstHndl;
    cuphyPuschRxCfoTaEstHndl_t     m_cfoTaEstHndl;
    cuphyPuschRxChEqHndl_t         m_chEqHndl;
    cuphyPuschRxRateMatchHndl_t    m_rateMatchHndl;
    cuphyPuschRxCrcDecodeHndl_t    m_crcDecodeHndl;
    cuphyPuschRxRssiHndl_t         m_rssiHndl;
    cuphyRmDecoderHndl_t           m_rmDecodeHndl;
    cuphySimplexDecoderHndl_t      m_spxDecoderHndl;
    cuphyRmDecoderHndl_t           m_rmDecodeHndl_csi2;
    cuphySimplexDecoderHndl_t      m_spxDecoderHndl_csi2;
    cuphyRmDecoderHndl_t           m_rmDecodeHndl_early;
    cuphySimplexDecoderHndl_t      m_spxDecoderHndl_early;
    cuphyUciOnPuschSegLLRs0Hndl_t  m_uciOnPuschSegLLRs0Hndl;
    cuphyUciOnPuschSegLLRs0Hndl_t  m_uciOnPuschEarlySegLLRs0Hndl;
    cuphyUciOnPuschSegLLRs2Hndl_t  m_uciOnPuschSegLLRs2Hndl;
    cuphyUciOnPuschCsi2CtrlHndl_t  m_uciOnPuschCsi2CtrlHndl;
    cuphyCompCwTreeTypesHndl_t     m_compCwTreeTypesHndl;
    cuphyPolSegDeRmDeItlHndl_t     m_polSegDeRmDeItlHndl;
    cuphyPolarDecoderHndl_t        m_polarDecoderHndl;
    cuphyCompCwTreeTypesHndl_t     m_compCwTreeTypesHndl_csi2;
    cuphyPolSegDeRmDeItlHndl_t     m_polSegDeRmDeItlHndl_csi2;
    cuphyPolarDecoderHndl_t        m_polarDecoderHndl_csi2;
    cuphyCompCwTreeTypesHndl_t     m_compCwTreeTypesHndl_early;
    cuphyPolSegDeRmDeItlHndl_t     m_polSegDeRmDeItlHndl_early;
    cuphyPolarDecoderHndl_t        m_polarDecoderHndl_early;

    uint32_t m_nUes;

    // SCH-on-PUSCH parameters
    uint16_t m_nSchUes;
    std::vector<uint16_t> m_schUserIdxsVec;

    // UCI-on-PUSCH LLR seg parameters.
    std::vector<uint16_t> m_nPrbsVec;
    uint16_t              m_nUciUes, m_nEarlyHarqUes, m_nCsi2Ues;
    std::vector<uint16_t> m_uciUserIdxsVec, m_csi2UeIdxsVec;
    //cudaDataType_t        m_llrDataType = CUDA_R_16F; // if to be uncommented, move initialization to constructor

    // Simplex decoder parameters (HARQ + CSI1)
    uint16_t             m_nSpxCws;
    cuphySimplexCwPrm_t* m_pSpxCwPrmsCpu;
    cuphySimplexCwPrm_t* m_pSpxPrmsGpu;

    // simplex decoder parameters (CSI2)
    uint16_t             m_nSpxCws_csi2;
    cuphySimplexCwPrm_t* m_pSpxCwPrmsCpu_csi2;
    cuphySimplexCwPrm_t* m_pSpxPrmsGpu_csi2;

    // simplex decoder parameters (early)
    uint16_t             m_nSpxCws_early;
    cuphySimplexCwPrm_t* m_pSpxCwPrmsCpu_early;
    cuphySimplexCwPrm_t* m_pSpxPrmsGpu_early;

    // Reed Muller decoder parameters (HARQ + CSI1)
    cuphyRmCwPrm_t* m_pRmCwPrmsCpu;
    cuphyRmCwPrm_t* m_pRmCwPrmsGpu;
    uint16_t        m_nRmCws;

    // Reed Muller decoder parameters (CSI2)
    cuphyRmCwPrm_t* m_pRmCwPrmsCpu_csi2;
    cuphyRmCwPrm_t* m_pRmCwPrmsGpu_csi2;
    uint16_t        m_nRmCws_csi2;

    // Reed Muller decoder parameters (early)
    cuphyRmCwPrm_t* m_pRmCwPrmsCpu_early;
    cuphyRmCwPrm_t* m_pRmCwPrmsGpu_early;
    uint16_t        m_nRmCws_early;

    // List size for Polar Decoder
    uint8_t m_polDcdrListSz;

    // Polar decoder parameters (HARQ + CSI1)
    uint16_t               m_nPolUciSegs;
    uint16_t               m_nPolCbs;
    cuphyPolarUciSegPrm_t* m_pUciSegPrmsCpu;
    cuphyPolarUciSegPrm_t* m_pUciSegPrmsGpu;
    cuphyPolarCwPrm_t*     m_pUciCwPrmsCpu;
    cuphyPolarCwPrm_t*     m_pUciCwPrmsGpu;
    std::vector<uint8_t*>  m_cwTreeTypesAddrVec;
    std::vector<__half*>   m_uciSegLLRsAddrVec;
    std::vector<__half*>   m_cwLLRsAddrVec;
    std::vector<__half*>   m_cwTreeLLRsAddrVec;
    std::vector<uint32_t*> m_cbEstAddrVec;
    std::vector<bool*>     m_listPolScratchAddrVec;
    uint8_t*               m_pPolCrcFlags;
    std::vector<uint32_t*> m_pUciSegEst;

    // Polar decoder parameters (CSI2)
    uint16_t               m_nPolUciSegs_csi2;
    uint16_t               m_nPolCbs_csi2;
    cuphyPolarUciSegPrm_t* m_pUciSegPrmsCpu_csi2;
    cuphyPolarUciSegPrm_t* m_pUciSegPrmsGpu_csi2;
    cuphyPolarCwPrm_t*     m_pUciCwPrmsCpu_csi2;
    cuphyPolarCwPrm_t*     m_pUciCwPrmsGpu_csi2;
    std::vector<uint8_t*>  m_cwTreeTypesAddrVec_csi2;
    std::vector<__half*>   m_uciSegLLRsAddrVec_csi2;
    std::vector<__half*>   m_cwLLRsAddrVec_csi2;
    std::vector<__half*>   m_cwTreeLLRsAddrVec_csi2;
    std::vector<uint32_t*> m_cbEstAddrVec_csi2;
    std::vector<bool*>     m_listPolScratchAddrVec_csi2;
    uint8_t*               m_pPolCrcFlags_csi2;
    std::vector<uint32_t*> m_pUciSegEst_csi2;

    // Polar decoder parameters (early)
    uint16_t               m_nPolUciSegs_early;
    uint16_t               m_nPolCbs_early;
    cuphyPolarUciSegPrm_t* m_pUciSegPrmsCpu_early;
    cuphyPolarUciSegPrm_t* m_pUciSegPrmsGpu_early;
    cuphyPolarCwPrm_t*     m_pUciCwPrmsCpu_early;
    cuphyPolarCwPrm_t*     m_pUciCwPrmsGpu_early;
    std::vector<uint8_t*>  m_cwTreeTypesAddrVec_early;
    std::vector<__half*>   m_uciSegLLRsAddrVec_early;
    std::vector<__half*>   m_cwLLRsAddrVec_early;
    std::vector<__half*>   m_cwTreeLLRsAddrVec_early;
    std::vector<uint32_t*> m_cbEstAddrVec_early;
    std::vector<bool*>     m_listPolScratchAddrVec_early;
    uint8_t*               m_pPolCrcFlags_early;
    std::vector<uint32_t*> m_pUciSegEst_early;

    // LDPC decoder
    size_t                                   m_ldpcWorkspaceSize;
    cuphy::context                           m_ctx;
    cuphy::LDPC_decoder                      m_LDPCdecoder;
    //cuphy::buffer<char, cuphy::device_alloc> m_ldpcWorkspaceBuffer;
    cuphy::stream_pool                       m_ldpcStreamPool; // Use a pool of CUDA streams to launch LDPC kernels (one per transport block)
    cuphyPuschLdpcKernelLaunch_t             m_LDPCkernelLaunchMode;

    // kernel launch configurationsen
    cuphyPuschRxEarlyHarqWaitLaunchCfg_t m_preEarlyHarqWaitCfgs;
    cuphyPuschRxEarlyHarqWaitLaunchCfg_t m_postEarlyHarqWaitCfgs;
    cuphyPuschRxChEstLaunchCfgs_t        m_chEstLaunchCfgs[CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST];
    cuphyPuschRxNoiseIntfEstLaunchCfgs_t m_noiseIntfEstLaunchCfgs[CUPHY_MAX_PUSCH_EXECUTION_PATHS];
    cuphyPuschRxChEqLaunchCfgs_t         m_chEqCoefCompLaunchCfgs[CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST];
    cuphyPuschRxChEqLaunchCfgs_t         m_chEqSoftDemapLaunchCfgs[CUPHY_MAX_PUSCH_EXECUTION_PATHS];
    cuphyPuschRxChEqLaunchCfgs_t         m_chEqSoftDemapBluesteinWorkspaceLaunchCfgs;
    cuphyPuschRxChEqLaunchCfgs_t         m_chEqSoftDemapIdftLaunchCfgs[CUPHY_MAX_PUSCH_EXECUTION_PATHS];
    cuphyPuschRxChEqLaunchCfgs_t         m_chEqSoftDemapAfterDftLaunchCfgs[CUPHY_MAX_PUSCH_EXECUTION_PATHS];
    cuphyPuschRxCfoTaEstLaunchCfgs_t     m_cfoTaEstLaunchCfgs;
    cuphyPuschRxRateMatchLaunchCfg_t     m_rateMatchLaunchCfg;
    cuphyLDPCDecodeLaunchConfig_t        m_ldpcLaunchCfgs[N_MAX_LDPC_HET_CFGS];
    cuphyPuschRxCrcDecodeLaunchCfg_t     m_crcLaunchCfgs[2]; // CB CRC + TB CRC
    cuphyPuschRxRssiLaunchCfgs_t         m_rssiLaunchCfgs;
    cuphyPuschRxRsrpLaunchCfgs_t         m_rsrpLaunchCfgs;
    cuphyUciOnPuschSegLLRs0LaunchCfg_t   m_uciOnPuschSegLLRs0LaunchCfg;
    cuphyUciOnPuschSegLLRs0LaunchCfg_t   m_uciOnPuschEarlySegLLRs0LaunchCfg;
    cuphyUciOnPuschSegLLRs2LaunchCfg_t   m_uciOnPuschSegLLRs2LaunchCfg;
    cuphyUciOnPuschCsi2CtrlLaunchCfg_t   m_uciOnPuschCsi2CtrlLaunchCfg;
    cuphyCompCwTreeTypesLaunchCfg_t      m_compCwTreeTypesLaunchCfg;
    cuphyPolSegDeRmDeItlLaunchCfg_t      m_polSegDeRmDeItlLaunchCfg;
    cuphyPolarDecoderLaunchCfg_t         m_polarDecoderLaunchCfg;
    cuphySimplexDecoderLaunchCfg_t       m_simplexDecoderLaunchCfg;
    cuphyRmDecoderLaunchCfg_t            m_rmDecoderLaunchCfg;
    cuphySimplexDecoderLaunchCfg_t       m_simplexDecoderLaunchCfg_csi2;
    cuphySimplexDecoderLaunchCfg_t       m_simplexDecoderLaunchCfg_early;
    cuphyRmDecoderLaunchCfg_t            m_rmDecoderLaunchCfg_csi2;
    cuphyCompCwTreeTypesLaunchCfg_t      m_compCwTreeTypesLaunchCfg_csi2;
    cuphyPolSegDeRmDeItlLaunchCfg_t      m_polSegDeRmDeItlLaunchCfg_csi2;
    cuphyPolarDecoderLaunchCfg_t         m_polarDecoderLaunchCfg_csi2;
    cuphyRmDecoderLaunchCfg_t            m_rmDecoderLaunchCfg_early;
    cuphyCompCwTreeTypesLaunchCfg_t      m_compCwTreeTypesLaunchCfg_early;
    cuphyPolSegDeRmDeItlLaunchCfg_t      m_polSegDeRmDeItlLaunchCfg_early;
    cuphyPolarDecoderLaunchCfg_t         m_polarDecoderLaunchCfg_early;

    // kernel descriptors
    cuphy::LDPC_decode_desc_set<N_MAX_LDPC_HET_CFGS> m_LDPCDecodeDescSet; // descriptors for LDPC (TB interface only)
    cuphy::kernelDescrs<N_PUSCH_DESCR_TYPES>         m_kernelStatDescr;
    cuphy::kernelDescrs<N_PUSCH_DESCR_TYPES>         m_kernelDynDescr;
    
    //the enabling flag for the early-HARQ mode
    bool m_earlyHarqModeEnabled;

    // graph parameters
    //======================================================================================================
    bool            m_cudaGraphModeEnabled;
    CUgraph         m_graph;
    CUgraphExec     m_graphExec;
    // graph kernel nodes
    CUgraphNode     m_emptyRootNode;
    CUgraphNode     m_chEstNodes[CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST][CUPHY_PUSCH_RX_CH_EST_N_MAX_HET_CFGS];
    CUgraphNode     m_noiseIntfEstNodes[CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS];
    //CUgraphNode     m_noiseIntfEstSspNodes[CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS][N_MAX_SUB_SLOT_STAGES];   // for sub-slot processing
    CUgraphNode     m_cfoTaEstNodes[CUPHY_PUSCH_RX_CFO_EST_N_MAX_HET_CFGS];
    CUgraphNode     m_chEqCoefCompNodes[CUPHY_PUSCH_RX_MAX_N_TIME_CH_EST][CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS];
    CUgraphNode     m_chEqSoftDemapNodes[CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS];
    CUgraphNode     m_chEqSoftDemapBluesteinWorkspaceNodes[CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS];
    CUgraphNode     m_chEqSoftDemapIdftNodes[CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS];
    CUgraphNode     m_chEqSoftDemapAfterDftNodes[CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS];
    CUgraphNode     m_rateMatchNode;
    CUgraphNode     m_ldpcDecoderNodes[N_MAX_LDPC_HET_CFGS];
    CUgraphNode     m_crcNodes[2];  // CB CRC + TB CRC
    CUgraphNode     m_uciSegLLRs0Node;
    CUgraphNode     m_simplexDecoderNode;
    CUgraphNode     m_rmDecoderNode;
    CUgraphNode     m_compCwTreeTypesNode;
    CUgraphNode     m_polSegDeRmDeItlNode;
    CUgraphNode     m_polarDecoderNode;
    // CSI-P2
    CUgraphNode     m_uciOnPuschCsi2CtrlNode;
    CUgraphNode     m_uciOnPuschSegLLRs2Node;
    CUgraphNode     m_rmDecCsi2Node;
    CUgraphNode     m_simplexDecoderCsi2Node;
    CUgraphNode     m_uciOnPuschCsi2CompCwTreeTypesNode;
    CUgraphNode     m_uciOnPuschCsi2PolSegDeRmDeItlNode;
    CUgraphNode     m_uciOnPuschCsi2PolarDecoderNode;
    //
    CUgraphNode     m_rssiNodes[CUPHY_PUSCH_RX_RSSI_N_MAX_HET_CFGS];
    CUgraphNode     m_rsrpNodes[CUPHY_PUSCH_RX_RSRP_N_MAX_HET_CFGS];

    // node states used in updateGraph()
    std::vector<std::vector<uint8_t>>    m_chEstNodesEnabled;
    std::vector<uint8_t>                 m_noiseIntfEstNodesEnabled;
    std::vector<uint8_t>                 m_cfoTaEstNodesEnabled;
    std::vector<std::vector<uint8_t>>    m_chEqCoefCompNodesEnabled;
    std::vector<uint8_t>                 m_chEqSoftDemapNodesEnabled;
    std::vector<uint8_t>                 m_chEqSoftDemapBluesteinWorkspaceNodesEnabled;
    std::vector<uint8_t>                 m_chEqSoftDemapIdftNodesEnabled;
    std::vector<uint8_t>                 m_chEqSoftDemapAfterDftNodesEnabled;
    uint8_t                              m_rateMatchNodeEnabled;
    std::vector<uint8_t>                 m_ldpcDecoderNodesEnabled;
    uint8_t                              m_crcNodesEnabled;
    uint8_t                              m_uciSegLLRs0NodeEnabled;
    uint8_t                              m_simplexDecoderNodeEnabled;
    uint8_t                              m_rmDecoderNodeEnabled;
    uint8_t                              m_polarNodeEnabled;    // used for m_compCwTreeTypesNode, m_polSegDeRmDeItlNode and m_polarDecoderNode
    uint8_t                              m_csi2NodeEnabled;     // used for all csi2 nodes
    std::vector<uint8_t>                 m_rssiNodesEnabled;
    std::vector<uint8_t>                 m_rsrpNodesEnabled;

    // PUSCH context used for mem-copy nodes in graph processing
    CUcontext       m_puschCtx;

    // for early HARQ sub-slot processing using CUDA graphs
    //======================================================================================================
    // early HARQ graph nodes
    CUgraphNode     m_ehqPreWaitNode;
    CUgraphNode     m_ehqChEstNodes[CUPHY_PUSCH_RX_CH_EST_N_MAX_HET_CFGS];
    CUgraphNode     m_ehqNoiseIntfEstNodes[CUPHY_PUSCH_RX_NOISE_INTF_EST_N_MAX_HET_CFGS];
    CUgraphNode     m_ehqChEqCoefCompNodes[CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS];
    CUgraphNode     m_ehqChEqSoftDemapNodes[CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS];
    CUgraphNode     m_ehqChEqSoftDemapBluesteinWorkspaceNodes[CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS];
    CUgraphNode     m_ehqChEqSoftDemapIdftNodes[CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS];
    CUgraphNode     m_ehqChEqSoftDemapAfterDftNodes[CUPHY_PUSCH_RX_CH_EQ_N_MAX_HET_CFGS];
    CUgraphNode     m_ehqUciSegLLRs0Node;
    CUgraphNode     m_ehqSimplexDecoderNode;
    CUgraphNode     m_ehqRmDecoderNode;
    CUgraphNode     m_ehqCompCwTreeTypesNode;
    CUgraphNode     m_ehqPolSegDeRmDeItlNode;
    CUgraphNode     m_ehqPolarDecoderNode;
    CUgraphNode     m_ehqMemcpyUciPayloadNode;
    CUgraphNode     m_ehqMemcpyUciCrcNode;
    CUgraphNode     m_ehqMemcpyDetectionStatNode;
    CUgraphNode     m_ehqEventReadyNode;
    CUgraphNode     m_ehqPostWaitNode;

    // node states used in updateEarlyHarqNodes()
    uint8_t              m_ehqAllNodesDisabled;
    uint8_t              m_ehqPreWaitNodeEnabled;
    std::vector<uint8_t> m_ehqChEstNodesEnabled;
    std::vector<uint8_t> m_ehqNoiseIntfEstNodesEnabled;
    std::vector<uint8_t> m_ehqChEqCoefCompNodesEnabled;
    std::vector<uint8_t> m_ehqChEqSoftDemapNodesEnabled;
    std::vector<uint8_t> m_ehqChEqSoftDemapBluesteinWorkspaceNodesEnabled;
    std::vector<uint8_t> m_ehqChEqSoftDemapIdftNodesEnabled;
    std::vector<uint8_t> m_ehqChEqSoftDemapAfterDftNodesEnabled;
    uint8_t              m_ehqUciSegLLRs0NodeEnabled;
    uint8_t              m_ehqSimplexDecoderNodeEnabled;
    uint8_t              m_ehqRmDecoderNodeEnabled;
    uint8_t              m_ehqPolarNodeEnabled;     // used for m_ehqCompCwTreeTypesNode, m_ehqPolSegDeRmDeItlNode and m_ehqPolarDecoderNode
    uint8_t              m_ehqMemcpyUciPayloadNodeEnabled;
    uint8_t              m_ehqMemcpyUciCrcNodeEnabled;
    uint8_t              m_ehqMemcpyDetectionStatNodeEnabled;
    uint8_t              m_ehqEventReadyNodeEnabled;
    uint8_t              m_ehqPostWaitNodeEnabled;


    // node parameters
    CUDA_KERNEL_NODE_PARAMS m_emptyNode0paramDriver;
    CUDA_KERNEL_NODE_PARAMS m_emptyNode1paramDriver;
    CUDA_KERNEL_NODE_PARAMS m_emptyNode2paramsDriver;

    CUDA_MEMCPY3D           m_ehqUciPayloadMemcpyD2Hparams;
    CUDA_MEMCPY3D           m_ehqUciCrcMemcpyD2Hparams;
    CUDA_MEMCPY3D           m_ehqDetectionStatMemcpyD2Hparams;
    struct cuphyGraphMemCopyD2Hparams
    {
        CUDA_MEMCPY3D params;
        cuphy::buffer<char, cuphy::pinned_alloc> host_mem;
        cuphy::buffer<char, cuphy::device_alloc> device_mem;
        int sz;

        cuphyGraphMemCopyD2Hparams(int copy_sz = 1) :
            sz(copy_sz),
            params{0},
            host_mem(copy_sz),
            device_mem(copy_sz)
        {
            params.WidthInBytes = copy_sz;
            params.Height = 1;
            params.Depth = 1;
            params.dstHost = host_mem.addr();
            params.dstMemoryType = CU_MEMORYTYPE_HOST;
            params.srcDevice = reinterpret_cast<CUdeviceptr>(device_mem.addr());
            params.srcMemoryType = CU_MEMORYTYPE_DEVICE;
        }

    };
    cuphyGraphMemCopyD2Hparams  m_memcpyTemplateD2H;


    uint32_t m_maxNTbs;
    uint32_t m_maxNCbs;
    uint32_t m_maxNCbsPerTb;

    // CUDA stream used for GPU operations
    cudaStream_t m_cuStream;
    cuphy::stream_pool m_G1streamPool; // Use a pool of CUDA streams to concurrently launch some kernels in CSI-P1 and CSI-P2
    cuphy::stream_pool m_G2streamPool; // Use a pool of CUDA streams to concurrently launch some other kernels in CSI-P1 and CSI-P2

    // event for stream syncs
    cuphy::event m_uciOnPuschSegLLRs0Event[CUPHY_MAX_PUSCH_EXECUTION_PATHS];
    cuphy::event m_compCwTreeTypesEvent[CUPHY_MAX_PUSCH_EXECUTION_PATHS];
    cuphy::event m_rateMatchEvent;
};

#endif // !defined(PUSCH_RX_HPP_INCLUDED_)
