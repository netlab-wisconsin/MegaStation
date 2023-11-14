/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "cuphy.h"
#include <vector>
#include <string>
#include "cuphy_hdf5.hpp"
#include "cuphy.hpp"
#include "tensor_desc.hpp"

#if !defined(PUCCH_RX_HPP_INCLUDED_)
#define PUCCH_RX_HPP_INCLUDED_

struct cuphyPucchRx
{

};

class PucchRx : public cuphyPucchRx {
public:
    enum Component
    {
        PUCCH_CELL_INFO              = 0,
        PUCCH_F0_Rx                  = 1,
        PUCCH_F1_Rx                  = 2,
        PUCCH_F2_RX                  = 3,
        PUCCH_F3_RX                  = 4,
        PUCCH_F234_UCI_SEG           = 5,
        POL_COMP_CW_TREE             = 6,
        POL_COMP_CW_TREE_ADDRS       = 7,
        POL_SEG_DERM_DEITL           = 8,
        POL_SEG_DERM_DEITL_CW_ADDRS  = 9,
        POL_SEG_DERM_DEITL_UCI_ADDRS = 10,
        POL_DECODE                   = 11,
        POL_DECODE_LLR_ADDRS         = 12,
        POL_DECODE_CB_ADDRS          = 13,
        LIST_POL_DECODE_SCRATCH_ADDRS= 14,
        RM_DECODE                    = 15,
        N_PUCCH_COMPONENTS           = 16
    };
    typedef struct F234RmSizes
    {
        uint16_t E_seg1;
        uint16_t E_seg2;
    } F234RmSizes_t;
    struct OutputParams
    {
        // flag to copy outputs to CPU after run
        bool                        cpuCopyOn;

        // sizes
        uint16_t                    nF234Ucis;
        uint16_t                    nUciSegs;
        uint32_t                    nUciPayloadBytes;

        // device (GPU) output addresses
        cuphyPucchF0F1UciOut_t* pF0UciOutGpu;
        cuphyPucchF0F1UciOut_t* pF1UciOutGpu;
        uint8_t*                pUciPayloadsGpu;
        uint8_t*                pCrcFlagsGpu;
        uint8_t*                pDtxFlagsGpu;
        float*                  pRssiGpu;
        float*                  pRsrpGpu;
        float*                  pSinrGpu;
        float*                  pInterfGpu;
        float*                  pNoiseVarGpu;
        float*                  pTaEstGpu;
        uint16_t*               pNumCsi2BitsGpu;

        uint8_t*                    pHarqDetectionStatusGpu;
        uint8_t*                    pCsiP1DetectionStatusGpu;
        uint8_t*                    pCsiP2DetectionStatusGpu;

        // host (CPU) output addresses
        cuphyPucchF234OutOffsets_t* pPucchF2OutOffsetsCpu;
        cuphyPucchF234OutOffsets_t* pPucchF3OutOffsetsCpu;
        cuphyPucchF0F1UciOut_t*     pF0UciOutCpu;
        cuphyPucchF0F1UciOut_t*     pF1UciOutCpu;
        uint8_t*                    pUciPayloadsCpu;
        uint8_t*                    pCrcFlagsCpu;
        //uint8_t*                    pDtxFlagsCpu;
        float*                      pRssiCpu;
        float*                      pRsrpCpu;
        float*                      pSinrCpu;
        float*                      pInterfCpu;
        float*                      pNoiseVarCpu;
        float*                      pTaEstCpu;
        //uint16_t*                   pNumCsi2BitsCpu;
        // HARQ/CSI part 1/CSI part 2 detection status. Refer to SCF FAPIv10.04, table 3–125, 126, 127 
        uint8_t*                    pHarqDetectionStatusCpu;
        uint8_t*                    pCsiP1DetectionStatusCpu;
        uint8_t*                    pCsiP2DetectionStatusCpu;

        // debug parameters
        bool                        debugOutputFlag;
        hdf5hpp::hdf5_file          outHdf5File;
    };
    PucchRx(cuphyPucchStatPrms_t const* pStatPrms, cudaStream_t strm);
    PucchRx(PucchRx const&) = delete;
    PucchRx& operator=(PucchRx const&) = delete;
    ~PucchRx();

    cuphyStatus_t setup(cuphyPucchDynPrms_t *pDynPrm);
    void run();
    void writeDbgBufSynch(cudaStream_t cuStream);
    const void* getMemoryTracker();

    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    static void printUciPrms(const cuphyPucchUciPrm_t* uciPrms);
    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    static void printDynApiPrms(cuphyPucchDynPrms_t* pDynPrm);
    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    static void printStaticApiPrms(cuphyPucchStatPrms_t const* pStaticPrm);

private:
    cuphyMemoryFootprint m_memoryFootprint;

    // creation functions
    size_t getBufferSize(cuphyPucchStatPrms_t const* pStatPrms);
    void   allocateDescr(void);
    void   createComponents(void);

    // setup functions
    F234RmSizes_t compRateMatchSizesF2(cuphyPucchUciPrm_t& F2uciPrms);
    F234RmSizes_t compRateMatchSizesF3(cuphyPucchUciPrm_t& F3uciPrms);

    void        expandF234UciParameters(cuphyPucchUciPrm_t& uciPrms, F234RmSizes_t& rmSizes, uint16_t nBitsUciSeg1);
    void        expandUciCodingPrms(cuphyPucchUciPrm_t& uciPrms, uint32_t nInfoBits, uint32_t nRmBits);

    cuphyStatus_t setupCmn(cuphyPucchDynPrms_t *pDynPrm);

    void allocateDeviceMemory(void);
    void allocateBackendBuffers(cuphyPucchUciPrm_t&           uciPrms, 
                                cuphyPucchF234OutOffsets_t&   outOffsets, 
                                F234RmSizes_t&                rmSizes,
                                void*                         pSeg1LLRs,
                                uint16_t                      nBitsUciSeg1,
                                uint16_t&                     F234uciIdx,
                                uint16_t&                     rmCwIdx,
                                uint16_t&                     polSegIdx,
                                uint32_t&                     polarWordOffset);
    
    cuphyStatus_t setupComponents(bool enableCpuToGpuDescrAsyncCpy, cuphyPucchDynPrms_t *pDynPrm);

    // graph functions
    void createGraph();
    void updateGraph();

    // run functions
    void copyOutputToCPU();

    // destroy functions
    void destroyComponents();

    // stream worker
    cudaStream_t  m_cuStream;

    // pipeline parameters
    cuphyPucchCellPrm_t*         m_pCellCmnPrms;
    cuphyPucchStatPrms_t         m_cuphyPucchStatPrms;
    cuphyPucchCellGrpDynPrm_t    m_cuphyPucchCellGrpDynPrm;

    cuphyCellStatPrm_t*&         m_pCellStatPrms = m_cuphyPucchStatPrms.pCellStatPrms;
    cuphyPucchCellDynPrm_t*&     m_pCellDynPrms  = m_cuphyPucchCellGrpDynPrm.pCellPrms;
    uint16_t&                    m_nF0Ucis       = m_cuphyPucchCellGrpDynPrm.nF0Ucis;
    cuphyPucchUciPrm_t*&         m_pF0UciPrms    = m_cuphyPucchCellGrpDynPrm.pF0UciPrms;  
    uint16_t&                    m_nF1Ucis       = m_cuphyPucchCellGrpDynPrm.nF1Ucis;
    cuphyPucchUciPrm_t*&         m_pF1UciPrms    = m_cuphyPucchCellGrpDynPrm.pF1UciPrms;
    uint16_t&                    m_nF2Ucis       = m_cuphyPucchCellGrpDynPrm.nF2Ucis;
    cuphyPucchUciPrm_t*&         m_pF2UciPrms    = m_cuphyPucchCellGrpDynPrm.pF2UciPrms;
    uint16_t&                    m_nF3Ucis       = m_cuphyPucchCellGrpDynPrm.nF3Ucis;
    cuphyPucchUciPrm_t*&         m_pF3UciPrms    = m_cuphyPucchCellGrpDynPrm.pF3UciPrms;
    uint16_t                     m_nF3Csi2Ucis;
    std::vector<uint16_t>        m_F3Csi2UciIdxsVec;

    // input/intermediate/output buffers
    cuphy::buffer<cuphyTensorPrm_t, cuphy::pinned_alloc>  m_tPrmDataRxBufCpu;
    cuphy::linear_alloc<128, cuphy::device_alloc>         m_LinearAlloc;
    OutputParams                                          m_outputPrms;
    std::vector<cuphy::tensor_ref>                        m_tRefDataRxBufCpu;
    cuphy::tensor_ref                                     m_tUciPayload;
    cuphy::tensor_ref                                     m_tDtxFlags;
    cuphy::tensor_ref                                     m_tHarqDetectionStatus;
    cuphy::tensor_ref                                     m_tCsiP1DetectionStatus;
    cuphy::tensor_ref                                     m_tCsiP2DetectionStatus;
    cuphy::tensor_ref                                     m_tSinr;
    cuphy::tensor_ref                                     m_tRssi;
    cuphy::tensor_ref                                     m_tRsrp;
    cuphy::tensor_ref                                     m_tInterf;
    cuphy::tensor_ref                                     m_tNoiseVar;
    cuphy::tensor_ref                                     m_tTaEst;

    // kernel descriptors
    cuphy::kernelDescrs<N_PUCCH_COMPONENTS>   m_kernelStatDescr;
    cuphy::kernelDescrs<N_PUCCH_COMPONENTS>   m_kernelDynDescr;

    // Component handles 
    cuphyPucchF0RxHndl_t         m_pucchF0RxHndl;
    cuphyPucchF1RxHndl_t         m_pucchF1RxHndl;
    cuphyPucchF2RxHndl_t         m_pucchF2RxHndl;
    cuphyPucchF3RxHndl_t         m_pucchF3RxHndl;
    cuphyPucchF234UciSegHndl_t   m_pucchF234UciSegHndl;
    cuphyRmDecoderHndl_t         m_rmDecodeHndl;
    cuphyCompCwTreeTypesHndl_t   m_compCwTreeTypesHndl;
    cuphyPolSegDeRmDeItlHndl_t   m_polSegDeRmDeItlHndl;
    cuphyPolarDecoderHndl_t      m_polarDecoderHndl;

    // kernel launch configurations
    cuphyPucchF0RxLaunchCfg_t        m_pucchF0RxLaunchCfg;
    cuphyPucchF1RxLaunchCfg_t        m_pucchF1RxLaunchCfg;
    cuphyPucchF2RxLaunchCfg_t        m_pucchF2RxLaunchCfg;
    cuphyPucchF3RxLaunchCfg_t        m_pucchF3RxLaunchCfg;
    cuphyPucchF234UciSegLaunchCfg_t  m_pucchF234UciSegLaunchCfg;
    cuphyCompCwTreeTypesLaunchCfg_t  m_compCwTreeTypesLaunchCfg;
    cuphyPolSegDeRmDeItlLaunchCfg_t  m_polSegDeRmDeItlLaunchCfg;
    cuphyPolarDecoderLaunchCfg_t     m_polarDecoderLaunchCfg;
    cuphyRmDecoderLaunchCfg_t        m_rmDecoderLaunchCfg;

    // graph parameters
    bool        m_cudaGraphModeEnabled;
    CUgraph     m_graph;
    CUgraphExec m_graphExec;
    CUgraphNode m_emptyRootNode;
    CUgraphNode m_pucchF0RxKernelNode;
    CUgraphNode m_pucchF1RxKernelNode;
    CUgraphNode m_pucchF2RxKernelNode;
    CUgraphNode m_pucchF3RxKernelNode;
    CUgraphNode m_rmDecoderKernelNode;
    CUgraphNode m_compCwTreeTypesKernelNode;
    CUgraphNode m_polSegDeRmDeItlKernelNode;
    CUgraphNode m_polarDecoderKernelNode;
    CUgraphNode m_pucchF234RxKernelNode;

    CUDA_KERNEL_NODE_PARAMS m_emptyNode0paramDriver, m_emptyNode1paramDriver;


    // F2 buffers
    std::vector<__half*>        m_F2seg1LLRaddrsVec;
    std::vector<F234RmSizes_t>  m_F2RmSizesVec;
    uint8_t*                    m_pF2dtxFlags;
    float*                      m_pF2pSinr;
    float*                      m_pF2pRssi;
    float*                      m_pF2pRsrp;
    float*                      m_pF2pInterf;
    float*                      m_pF2pNoiseVar;
    float*                      m_pF2pTaEst;
    std::vector<uint16_t>       m_F2nBitsUciSeg1;

    // F3 buffers
    std::vector<__half*>        m_F3seg1LLRaddrsVec;
    std::vector<F234RmSizes_t>  m_F3RmSizesVec;
    uint8_t*                    m_pF3dtxFlags;
    float*                      m_pF3pSinr;
    float*                      m_pF3pRssi;
    float*                      m_pF3pRsrp;
    float*                      m_pF3pInterf;
    float*                      m_pF3pNoiseVar;
    float*                      m_pF3pTaEst;
    std::vector<uint16_t>       m_F3nBitsUciSeg1;

    // Reed Muller decoder parameters
    uint16_t                                           m_nRmCws;
    cuphy::buffer<cuphyRmCwPrm_t, cuphy::pinned_alloc> m_rmCwPrmsBufCpu;
    cuphyRmCwPrm_t*                                    m_pRmCwPrmsGpu;

    // Polar decoder parameters
    uint8_t                                                   m_polDcdrListSz;
    uint16_t                                                  m_nPolSegs;
    uint16_t                                                  m_nPolCbs;
    cuphy::buffer<cuphyPolarUciSegPrm_t, cuphy::pinned_alloc> m_polSegPrmsBufCpu;
    cuphyPolarUciSegPrm_t*                                    m_pPolSegPrmsGpu;
    cuphy::buffer<cuphyPolarCwPrm_t, cuphy::pinned_alloc>     m_polCwPrmsBufCpu;
    cuphyPolarCwPrm_t*                                        m_pPolCwPrmsGpu;
    std::vector<uint8_t*>                                     m_polCwTreeTypesAddrVec;
    std::vector<__half*>                                      m_polSegLLRsAddrVec;
    std::vector<__half*>                                      m_polCwLLRsAddrVec;
    std::vector<__half*>                                      m_polCwTreeLLRsAddrVec;
    std::vector<uint32_t*>                                    m_polCbEstAddrVec;
    std::vector<bool*>                                        m_listPolScratchAddrVec;
    uint8_t*                                                  m_pPolCrcFlags;
    std::vector<uint32_t*>                                    m_pUciSegEst;
};

#endif // !defined(PUCCH_RX_HPP_INCLUDED_)




