/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(CSIRS_TX_HPP_INCLUDED_)
#define CSIRS_TX_HPP_INCLUDED_

#include "cuphy.h"
#include "cuphy_api.h"
#include "cuphy.hpp"
#include "util.hpp"
#include "csirs/csirs.hpp"
#include <iostream>
#include <iostream>
#include <cstdlib>
#include <string>

struct cuphyCsirsTx
{};

class CsirsTx : public cuphyCsirsTx {

public:
    static constexpr int Ng = 273*2*3;

    enum Component
    {
        CSIRS_PARAMS       = 0,
        CSIRS_OFFSETS      = 1,
        CSIRS_TENSOR_ADDR  = 2,
        CSIRS_TENSOR_RES   = 3,
        CSIRS_PMW_PARAMS   = 4,
        N_CSIRS_COMPONENTS = 5
    };

    /**
     * @brief: Construct CsirsTx class.
     * @param[in] pStatPrms: static parameters for CsirsTx
     */
     CsirsTx(cuphyCsirsStatPrms_t const* pStatPrms);

    /**
     * @brief: CsirsTx cleanup.
     */
    ~CsirsTx();

    /**
     * @brief: Convert input dyn_params to CsirsParams struct
     * @param[in] dyn_params: input parameters to CSI-RS.
     * @param[in] cuda_strm: CUDA stream for memcpys.
     * @return CUPHY_STATUS_SUCCESS or relevant error status
     */
    cuphyStatus_t expandParameters(cuphyCsirsDynPrms_t* dyn_params,
                                   cudaStream_t cuda_strm);

    /**
     * @brief: set kernel launch parameters
     */
    void setKernelParams();

    /**
     * @brief: Run CSI-RS
     * @param[in] cuda_strm: CUDA stream for kernel launches.
     */
    int run(const cudaStream_t& cuda_strm);

    const cuphyCsirsDynPrms_t*  dynamic_params;

    /**
     * @brief Print input parameters
     * @param[in] cell_static_params: CSI-RS static input parameters
     * @param[in] dyn_params: CSI-RS dynamic input parameters
     */
    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    void printCsirsConfig(const cuphyCsirsStatPrms_t& cell_static_params, const cuphyCsirsDynPrms_t& dyn_params);

    const void* getMemoryTracker();

    CUgraph *GetGraph() { return &m_graph; }

    const cuphyCsirsStatPrms_t* static_params;

private:

    cuphyMemoryFootprint memory_footprint;

    /**
     * @brief Validate input parameters
     * @param[in] params: Parameter list array
     * @param[in] numParams: number of elements in params
     * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT
     */
    cuphyStatus_t checkConfig(CsirsParams* params, int numParams);

    // Workspace buffers that enables single, not overprovisioned H2D copy, per setup instead of the
    // overprovisioned one with descriptors.
    cuphy::unique_device_ptr<uint8_t> d_workspace;
    cuphy::unique_pinned_ptr<uint8_t> h_workspace;

    // Pointers to host or device buffers within h_workspace  or d_workspace respectively

    // CSI-RS input parameter list
    CsirsParams* h_params;
    CsirsParams* d_params;

    // kernel launch configurations
    cuphyGenScramblingLaunchCfg_t    m_genCsirsScramblingLaunchCfg;
    cuphyGenCsirsTfSignalLaunchCfg_t m_genCsirsTfSignalLaunchCfg;

    // kernel args
    void* m_genScramblingArgs[3];
    void* m_genTfSignalArgs[8];

    // graph functions
    void createGraph();
    void updateGraph();

    // graph parameters
    bool        m_cudaGraphModeEnabled;
    CUgraph     m_graph;
    CUgraphExec m_graphExec;
    CUgraphNode m_genCsirsScramblingNode;
    CUgraphNode m_genCsirsTfSignalNode;
    CUDA_KERNEL_NODE_PARAMS m_emptyNode3ParamsDriver, m_emptyNode8ParamsDriver;

    // gold sequence buffer
    cuphy::unique_device_ptr<uint8_t> d_goldSeq;

    // Offset array used to map starting thread index for each parameter in the batch
    uint32_t* h_offsets;
    uint32_t* d_offsets;

    // Array with output tensor address for each cell in the batch
    __half2** h_cell_tensor_addr;
    __half2** d_cell_tensor_addr;

    // Array with number of REs each cell output tensor in the batch
    uint16_t* h_cell_tensor_REs;
    uint16_t* d_cell_tensor_REs;

    cuphyCsirsPmWOneLayer_t* h_pmw_params;
    cuphyCsirsPmWOneLayer_t* d_pmw_params;

    // Offset of the various host/device buffers, in bytes, from the start of the workspace
    std::array<int, N_CSIRS_COMPONENTS+1> workspace_offsets;

    // Update workspace_offsets for all buffers for each new config of cells and CSI-RS parameters
    void updateWorkspaceOffsets(int cells, int nRrcParams);
    // Update the workspace pointers
    void updateWorkspacePtrs();

    // number of CSI-RS parameters in the batch across all cells
    size_t numParams;

    // number of cells to be processed in this batch
    int    numCells;

    // maximum number of cells to be processed per slot
    int    maxCellsPerSlot;
};

#endif // !defined(CSIRS_TX_HPP_INCLUDED_)
