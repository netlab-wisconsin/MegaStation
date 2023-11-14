/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(SSB_TX_HPP_INCLUDED_)
#define SSB_TX_HPP_INCLUDED_

#include "cuphy.h"
#include "cuphy_api.h"
#include "cuphy.hpp"
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "util.hpp"
#include "ss/ss.hpp"
#include <iostream>
#include <iostream>
#include <cstdlib>
#include <string>

#define Nc          1600 // starting index for PN sequence
#define G_CRC_24_C  0x01B2B117

struct cuphySsbTx
{};

class SsbTx : public cuphySsbTx {

public:
    enum Component
    {
        //SSB WORKSPACES
        SSB_PER_CELL_PARAMS = 0,
        PER_SS_BLOCK_PARAMS = 1,
        SSB_INPUT_W_CRC = 2,
        CELL_OUTPUT_ADDR = 3,
        SSB_PRECODING_MATRIX = 4,
        N_SSB_COMPONENTS  = 5
    };

    /**
     * @brief: Construct SsbTx object.
     * @param[in] cfg_static_params: pointer to SSB static parameters.
     */
     SsbTx(const cuphySsbStatPrms_t* cfg_static_params);

    /**
     * @brief: SsbTx cleanup.
     */
    ~SsbTx();

    cuphyStatus_t expandParameters(cuphySsbDynPrms_t* dyn_params,
                                   cudaStream_t cuda_strm=0);

    /**
     * @brief: Setup SSB
     * @param[in] dyn_params: SSB dynamic parameters
     * @return CUPHY_STATUS_SUCCESS or relevant error status
     */
    cuphyStatus_t setup(cuphySsbDynPrms_t* dyn_params);

    /**
     * @brief: Run SSB
     * @param[in] cuda_strm: CUDA stream for kernel launches.
     */
    int run(const cudaStream_t& cuda_strm);

    template <fmtlog::LogLevel log_level=fmtlog::DBG>
    void printSsbConfig(const cuphySsbDynPrms_t& params);
    const void* getMemoryTracker();

    const cuphySsbDynPrms_t*  dynamic_params;
    const cuphySsbStatPrms_t* static_params;


private:

    cuphyMemoryFootprint memory_footprint;

   /**
    * @brief: Generate PBCH payload and CRC for all num_SSB
    *         SSBs across all num_cells cells.
    * @param[in] h_x_mib: pointer to the input MIB sequence
    * @param[in] h_ssb_params: SSB parameters for all num_SSBs SSBs
    * @param[in] h_per_cell_params: cell specific parameters for all num_cells cells
    * @param[in, out] pEncdRMLaunchCfg: pointer to launch configuration to encode rate match SSBs
    * @param[in, out] pSsbMapperLaunchCfg: pointer to launch configurations for SSB mapper
    * @return CUPHY_STATUS_SUCCESS or CUPHY_STATUS_INVALID_ARGUMENT.
    */
    cuphyStatus_t preparePBCH(uint32_t*                                 h_x_mib,
                              const cuphyPerSsBlockDynPrms_t*           h_ssb_params,
                              const cuphyPerCellSsbDynPrms_t*           h_per_cell_params,
                              cuphyEncoderRateMatchMultiSSBLaunchCfg_t* pEncdRMLaunchCfg,
                              cuphySsbMapperLaunchCfg_t*                pSsbMapperLaunchCfg);

    void allocateBuffers();
    void allocateDescr();

    int max_cells_per_slot;
    int max_SSBs_per_slot;
    int num_cells, num_SSBs;

    // Could potentially allocate a single workspace instead
    cuphy::unique_device_ptr<uint8_t>        d_x_coded;  // output sequence of polar encoder
    cuphy::unique_device_ptr<uint8_t>        d_x_tx;     // output sequence of rate match, 32-bit aligned and multiple of 4 bytes

    // Pointers to PBCH payload with CRC attached.
    uint8_t* d_x_crc;
    uint8_t* h_x_crc;

    cuphy::kernelDescrs<N_SSB_COMPONENTS> m_component_descrs; // workspace descriptors
    bool bulk_desc_async_copy;

    // kernel launch configurations
    cuphyEncoderRateMatchMultiSSBLaunchCfg_t m_encodeRmMultiSsbLaunchCfg;
    cuphySsbMapperLaunchCfg_t m_ssbMapperLaunchCfg;

    // kernel args
    void* m_encdRmSSBArgs[3];
    void* m_ssbMapperArgs[5];

    // graph functions
    void createGraph();
    void updateGraph();

    // graph parameters
    bool        m_cudaGraphModeEnabled;
    CUgraph     m_graph;
    CUgraphExec m_graphExec;
    CUgraphNode m_encdRmMultiSsbNode;
    CUgraphNode m_ssbMapperNode;
    CUDA_KERNEL_NODE_PARAMS m_emptyNode5ParamsDriver, m_emptyNode3ParamsDriver;

};

#endif // !defined(SSB_TX_HPP_INCLUDED_)
