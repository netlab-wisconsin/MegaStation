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

#if !defined(SRS_RX_HPP_INCLUDED_)
#define SRS_RX_HPP_INCLUDED_

struct cuphySrsRx
{

};

class SrsRx : public cuphySrsRx {
public:
    enum Component
    {
        SRS_CHEST                  = 0,
        N_SRS_COMPONENTS           = 1
    };
    struct OutputParams
    {
        // flag to copy outputs to CPU after run
        bool cpuCopyOn;

        // device (GPU) output addresses
        cuphySrsReport_t*          d_srsReports;        // array containing SRS reports of all users
        float*                     d_rbSnrBuffer;       // buffer containing RB SNRs of all users

        // host (CPU) output addresses
        cuphySrsChEstBuffInfo_t*  h_chEstBuffInfo;
        cuphySrsReport_t*         h_srsReports;        // array containing SRS reports of all users
        float*                    h_rbSnrBuffer;       // buffer containing RB SNRs of all users
        uint32_t*                 h_rbSnrBuffOffsets;  // buffer containing user offsets into pRbSnrBuffer
        cuphySrsChEstToL2_t*      h_srsChEstToL2;      // buffer containing SRS ChEst to L2

        // debug parameters
        bool               debugOutputFlag;
        hdf5hpp::hdf5_file outHdf5File;
    };
    SrsRx(cuphySrsStatPrms_t const* pStatPrms, cudaStream_t cuStream);
    SrsRx(SrsRx const&) = delete;
    SrsRx& operator=(SrsRx const&) = delete;
    ~SrsRx();

    cuphyStatus_t setup(cuphySrsDynPrms_t *pDynPrm);
    void run();
    void copyOutputToCPU(cudaStream_t cuStream);

    // debug functions:
    void writeDbgBufSynch(cudaStream_t cuStream);
    static void printStaticApiPrms(cuphySrsStatPrms_t const* pStaticPrms);
    static void printDynApiPrms(cuphySrsDynPrms_t* pDynPrm);


private:
    // creation functions
    size_t getBufferSize(cuphySrsStatPrms_t const* pStatPrms);
    void   allocateDescr(void);
    void   createComponents(cuphySrsFilterPrms_t* pSrsFilterPrms, cudaStream_t cuStream);

    // setup functions
    void setupCmn(cuphySrsDynPrms_t *pDynPrm);
    void allocateDeviceMemory(void);
    cuphyStatus_t setupComponents(bool enableCpuToGpuDescrAsyncCpy, cuphySrsDynPrms_t *pDynPrm);

    // graph functions
    void createGraph();
    void updateGraph();

    // destroy functions
    void destroyComponents();

    // stream worker:
    cudaStream_t  m_cuStream;

    // pipeline parameters:
    std::vector<cuphySrsCellPrms_t> m_srsCellPrmsVec;
    uint16_t                        m_nSrsUes;
    uint16_t                        m_nCells;
    cuphyCellStatPrm_t*             m_hCellStatPrms;
    cuphyUeSrsPrm_t*                m_hUeSrsPrm;
  
    // input/intermediate/output buffers
    cuphyTensorPrm_t*                             m_hPrmDataRx;
    cuphy::linear_alloc<128, cuphy::device_alloc> m_LinearAlloc;
    OutputParams                                  m_outputPrms;
    std::vector<void*>                            m_gpuAddrsChEstToL2Vec;

    // kernel descriptors
    cuphy::kernelDescrs<N_SRS_COMPONENTS>   m_kernelStatDescr;
    cuphy::kernelDescrs<N_SRS_COMPONENTS>   m_kernelDynDescr;

    // Component handles 
    cuphySrsChEst0Hndl_t  m_srsChEst0Hndl;

    // kernel launch configurations
    cuphySrsChEst0LaunchCfg_t  m_srsChEst0LaunchCfg;

    // graph parameters
    bool        m_cudaGraphModeEnabled;
    CUgraph     m_graph;
    CUgraphExec m_graphExec;
    CUgraphNode m_srsKernelNode;
};

#endif // !defined(SRS_RX_HPP_INCLUDED_)