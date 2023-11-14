/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(PDCCH_TX_HPP_INCLUDED_)
#define PDCCH_TX_HPP_INCLUDED_

#include "cuphy.h"
#include "cuphy_api.h"
#include "cuphy.hpp"
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "util.hpp"
#include <iostream>
#include <iostream>
#include <cstdlib>
#include <string>

struct cuphyPdcchTx
{};

class PdcchTx : public cuphyPdcchTx {

public:
    enum Component
    {
        //PDCCH WORKSPACES
        PDCCH_PARAMS = 0,
        PDCCH_DCI_PARAMS = 1,
        PDCCH_INPUT_W_CRC = 2,
        PDCCH_PMW_PARAMS = 3,
        PDCCH_TM_PARAMS = 4,
        N_PDCCH_COMPONENTS  = 5
    };

    /**
     * @brief: Construct PdcchTx object.
     * @param[in] cfg_static_params: pointer to PDCCH static parameters.
     */
     PdcchTx(const cuphyPdcchStatPrms_t* cfg_static_params);

    /**
     * @brief: PdcchTx cleanup.
     */
    ~PdcchTx();

    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    void printPdcchConfig(const cuphyPdcchDynPrms_t& params);

    cuphyStatus_t expandParameters(cuphyPdcchDynPrms_t* dyn_params,
                          cudaStream_t cuda_strm=0);

    /**
     * @brief: Setup PDCCH
     * @param[in] dyn_params: PDCCH dynamic parameters
     * @return CUPHY_STATUS_SUCCESS or other setup error
     */
    cuphyStatus_t setup(cuphyPdcchDynPrms_t* dyn_params);

    /**
     * @brief: Run PDCCH
     * @param[in] cuda_strm: CUDA stream for kernel launches.
     */
    int run(const cudaStream_t& cuda_strm);

    const cuphyPdcchDynPrms_t*  dynamic_params;
    const cuphyPdcchStatPrms_t* static_params;

    const void* getMemoryTracker();




private:
    cuphyMemoryFootprint memory_footprint;

    static constexpr int BITS_PER_UINT8 = 8;
    int      max_tx_bytes, max_coded_bytes;
    uint32_t max_coresets_per_slot, max_DCIs_per_slot;
    const int max_alignment = std::max(alignof(PdcchParams), alignof(cuphyPdcchDciPrm_t));

    // This method is called in the constructor. It allocates overprovisioned buffers needed.
    size_t getBufferSize(const cuphyPdcchStatPrms_t* pStatParams);
    void   allocateBuffers();
    void   allocateDescr();
    cuphyStatus_t   checkConfig(PdcchParams& params);

    // input/intermediate/output buffers
    cuphy::linear_alloc<128, cuphy::device_alloc> m_LinearAlloc;

    // Needed internally by the polar encoder component. Intermediate buffer; no need for an equivalent host allocation.
    uint8_t* d_x_coded_bytes;
    // Output of polar encoder.
    uint8_t* d_x_tx_bytes;
    // Scrambling sequence buffer
    uint32_t* d_scrambling_seq;

    int num_coresets, num_DCIs;
    uint8_t* h_input_w_crc_bytes, *d_input_w_crc_bytes;

    // kernel args
    void* m_encodeRateMatchMultiDCIsArgs[5];
    void* m_genScramblingSeqArgs[2];
    void* m_genTfSignalArgs[6];

    cuphy::tensor_desc  input_w_crc_desc, h_input_bytes_desc;
    cuphy::kernelDescrs<N_PDCCH_COMPONENTS> m_component_descrs; // workspace descriptors
    bool bulk_desc_async_copy;
    const uint8_t * h_input_bytes_ptr;

    cuphyStatus_t preparePdcch(cudaStream_t cuda_strm);
    PdcchParams* h_coreset_params, * d_coreset_params;
    cuphyPdcchDciPrm_t* h_dci_params, * d_dci_params;
    cuphyPdcchPmWOneLayer_t* h_pmw_params, * d_pmw_params;
    uint8_t* h_dci_tm_info, * d_dci_tm_info;
    bool first_setup = true;

    // Exploring benefit of manual descriptors
    cuphy::unique_pinned_ptr<uint8_t> h_workspace;
    cuphy::unique_device_ptr<uint8_t> d_workspace;

    // Offset of the various host/device buffers, in bytes, from the start of the workspace
    std::array<int, N_PDCCH_COMPONENTS+1> workspace_offsets;

    // kernel launch configurations
    cuphyEncoderRateMatchMultiDCILaunchCfg_t m_encodeRateMatchMultiDCIsLaunchCfg;
    cuphyGenScramblingSeqLaunchCfg_t         m_genScramblingSeqLaunchCfg;
    cuphyGenPdcchTfSgnlLaunchCfg_t           m_genTfSignalLaunchCfg;

    // graph functions
    void createGraph();
    void updateGraph();

    // graph parameters
    bool        m_cudaGraphModeEnabled;
    CUgraph     m_graph;
    CUgraphExec m_graphExec;
    CUgraphNode m_encodeRateMatchMultiDCIsNode;
    CUgraphNode m_genScramblingSeqNode;
    CUgraphNode m_genTfSignalNode;
    CUDA_KERNEL_NODE_PARAMS m_emptyNode6ParamsDriver, m_emptyNode5ParamsDriver, m_emptyNode2ParamsDriver;

    // Update workspace_offsets for all buffers for each new config of coresets and DCIs
    void updateWorkspaceOffsets(int coreset_cnt, int dci_cnt);
    // Update the workspace pointers
    void updateWorkspacePtrs();

};

#endif // !defined(PDCCH_TX_HPP_INCLUDED_)
