/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "ssb_tx.hpp"
#include "cuphy_internal.h"

using namespace cuphy;

cuphyStatus_t CUPHYWINAPI cuphyCreateSsbTx(cuphySsbTxHndl_t* pSsbTxHndl, cuphySsbStatPrms_t const* pStatPrms)
{
    if((pSsbTxHndl == nullptr) || (pStatPrms == nullptr))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    return cuphy::tryCallableAndCatch([&]
    {
        SsbTx* new_pipeline = new SsbTx(pStatPrms);

        if(new_pipeline == nullptr)
        {
            return CUPHY_STATUS_ALLOC_FAILED;
        }
        *pSsbTxHndl = new_pipeline;

        return CUPHY_STATUS_SUCCESS;
    });
}

#if 0
const void* cuphyGetMemoryFootprintTrackerSsbTx(cuphySsbTxHndl_t ssbTxHndl)
{
    if(ssbTxHndl == nullptr)
    {
        return nullptr;
    }
    SsbTx* pipeline_ptr  = static_cast<SsbTx*>(ssbTxHndl);
    return pipeline_ptr->getMemoryTracker();
}
#endif

const void* SsbTx::getMemoryTracker()
{
    return &memory_footprint;
}


cuphyStatus_t CUPHYWINAPI cuphySetupSsbTx(cuphySsbTxHndl_t ssbTxHndl, cuphySsbDynPrms_t* pDynPrms)
{
    PUSH_RANGE("SSB_SETUP", 1);
    if((pDynPrms == nullptr) || (ssbTxHndl == nullptr))
    {
        POP_RANGE("SSB_SETUP", 1);
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    return cuphy::tryCallableAndCatch([&]
    {
        SsbTx* pipeline_ptr  = static_cast<SsbTx*>(ssbTxHndl);
        cuphyStatus_t status = pipeline_ptr->setup(pDynPrms);
        POP_RANGE("SSB_SETUP", 1);
        return status;
    }, CUPHY_STATUS_INVALID_ARGUMENT);
}

cuphyStatus_t CUPHYWINAPI cuphyRunSsbTx(cuphySsbTxHndl_t ssbTxHndl, uint64_t procModeBmsk /* not used */)
{
    if(ssbTxHndl == nullptr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }
    return cuphy::tryCallableAndCatch([&]
    {
        SsbTx* pipeline_ptr  = static_cast<SsbTx*>(ssbTxHndl);

        int failed = pipeline_ptr->run(pipeline_ptr->dynamic_params->cuStream);
        return (failed == 0) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
    });
}

cuphyStatus_t CUPHYWINAPI cuphyDestroySsbTx(cuphySsbTxHndl_t ssbTxHndl)
{
    if(ssbTxHndl == nullptr)
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    SsbTx* pipeline_ptr  = static_cast<SsbTx*>(ssbTxHndl);
    delete pipeline_ptr;
    return CUPHY_STATUS_SUCCESS;
}


SsbTx::SsbTx(const cuphySsbStatPrms_t* cfg_static_params):
    static_params(cfg_static_params),
    m_component_descrs("SsbDescr"),
    bulk_desc_async_copy(true)
{
    cfg_static_params->pOutInfo->pMemoryFootprint = &memory_footprint; // update  static parameter field that points to the cuphyMemoryFootprintTracker object for this channel
    max_cells_per_slot   = static_params->nMaxCellsPerSlot;
    max_SSBs_per_slot    = max_cells_per_slot * CUPHY_SSB_MAX_SSBS_PER_CELL_PER_SLOT;
    allocateBuffers();
    allocateDescr(); //Allocate Descriptors.

    // the following call to create graph also helps (by calling cudaGetFuncBySymbol) to move CUDA runtime initialization overhead into channel constructor
    createGraph();
#if CUDA_VERSION >= 12000
    CU_CHECK_EXCEPTION(cuGraphInstantiate(&m_graphExec, m_graph, 0));
#else
    CU_CHECK_EXCEPTION(cuGraphInstantiate(&m_graphExec, m_graph, 0, 0, 0));
#endif
    if (PRINT_GPU_MEMORY_CUPHY_CHANNEL == 1) 
    {
        memory_footprint.printMemoryFootprint(this, "SSB");
    }
}

template <fmtlog::LogLevel log_level>
void SsbTx::printSsbConfig(const cuphySsbDynPrms_t& params)
{
    NVLOG_FMT(log_level, NVLOG_SSB, "SSB TX pipeline with {} cells", params.nCells);

    NVLOG_FMT(log_level, NVLOG_SSB, "SSB TX pipeline: per cell SSB parameters");
    for (int i = 0; i < params.nCells; i++) {
        const cuphyPerCellSsbDynPrms_t& per_cell_SSB_params = params.pPerCellSsbDynParams[i];
        NVLOG_FMT(log_level, NVLOG_SSB, "SSB parameters for cell {:5d}: ", i);
        NVLOG_FMT(log_level, NVLOG_SSB, "------------------------------------------------------");
        NVLOG_FMT(log_level, NVLOG_SSB, "NID:            {:5d}", per_cell_SSB_params.NID);
        NVLOG_FMT(log_level, NVLOG_SSB, "nHF:            {:5d}", per_cell_SSB_params.nHF);
        NVLOG_FMT(log_level, NVLOG_SSB, "Lmax:           {:5d}", per_cell_SSB_params.Lmax);
        NVLOG_FMT(log_level, NVLOG_SSB, "SFN:            {:5d}", per_cell_SSB_params.SFN);
        NVLOG_FMT(log_level, NVLOG_SSB, "k_SSB:          {:5d}", per_cell_SSB_params.k_SSB);
        NVLOG_FMT(log_level, NVLOG_SSB, "nF:             {:5d}", per_cell_SSB_params.nF);
        NVLOG_FMT(log_level, NVLOG_SSB, "slotBufferIdx   {:5d}", per_cell_SSB_params.slotBufferIdx);
    }
    NVLOG_FMT(log_level, NVLOG_SSB, "SSB TX pipeline with {} precoded SS blocks", params.nPrecodingMatrices);
    NVLOG_FMT(log_level, NVLOG_SSB, "SSB TX pipeline: SSB parameters across all {} SS blocks: ", params.nSSBlocks);
    
    for (int i = 0; i < params.nSSBlocks; i++) {
        NVLOG_FMT(log_level, NVLOG_SSB, "SSB parameters for SS block {}: ", i);
        NVLOG_FMT(log_level, NVLOG_SSB, "------------------------------------------------------");
        const cuphyPerSsBlockDynPrms_t& per_SS_block_params = params.pPerSsBlockParams[i];
        NVLOG_FMT(log_level, NVLOG_SSB, "f0:            {:5d}", per_SS_block_params.f0);
        NVLOG_FMT(log_level, NVLOG_SSB, "t0:            {:5d}", per_SS_block_params.t0);
        NVLOG_FMT(log_level, NVLOG_SSB, "blockIndex:    {:5d}", per_SS_block_params.blockIndex);
        NVLOG_FMT(log_level, NVLOG_SSB, "beta_pss:      {:f}",  per_SS_block_params.beta_pss);
        NVLOG_FMT(log_level, NVLOG_SSB, "beta_sss:      {:f}",  per_SS_block_params.beta_sss);
        NVLOG_FMT(log_level, NVLOG_SSB, "cell_index:    {:5d}", per_SS_block_params.cell_index);
        NVLOG_FMT(log_level, NVLOG_SSB, "enablePrcdBf:  {:5d}", per_SS_block_params.enablePrcdBf);
        if(per_SS_block_params.enablePrcdBf)
        {
            NVLOG_FMT(log_level, NVLOG_SSB, "pmwPrmIdx:     {:5d}", per_SS_block_params.pmwPrmIdx);
            NVLOG_FMT(log_level, NVLOG_SSB, "nPorts :       {:5d}", params.pPmwParams[per_SS_block_params.pmwPrmIdx].nPorts);
            std::stringstream matrix_row;
            matrix_row.precision(5);
            matrix_row << "Precoding Matrix:      ";
            for(int idx = 0; idx < params.pPmwParams[per_SS_block_params.pmwPrmIdx].nPorts; idx++)
            {
                if (idx != 0) matrix_row << ", ";
                matrix_row << std::fixed << "{" << (float)params.pPmwParams[per_SS_block_params.pmwPrmIdx].matrix[idx].x << ", " <<  (float)params.pPmwParams[per_SS_block_params.pmwPrmIdx].matrix[idx].y << "}";
            }
            NVLOG_FMT(log_level, NVLOG_SSB, "{}", matrix_row.str());
        }
    }

    NVLOG_FMT(log_level, NVLOG_SSB, "SSB TX pipeline: PBCH MIB across all {} SSBs: ", params.nSSBlocks);
    for (int i = 0; i < params.nSSBlocks; i++) {
        // 24 (Nmib) most significant bits are valid; SSBs in the same slot of a cell share the same MIB but it is still separately stored //FIXME
        // cuPHY does not check that this is the case.
        NVLOG_FMT(log_level, NVLOG_SSB, "PBCH MIB for SS block {}: {:#x}", i, params.pDataIn[0].pMibInput[i]);
    }
}

void SsbTx::allocateBuffers()
{
    // allocate device tensors
    d_x_coded    = make_unique_device<uint8_t>(max_SSBs_per_slot * (CUPHY_SSB_N_PBCH_POLAR_ENCODED_BITS/8), &memory_footprint); // output sequence of polar encoder
    d_x_tx       = make_unique_device<uint8_t>(max_SSBs_per_slot * (round_up_to_next(CUPHY_SSB_N_PBCH_SCRAMBLING_SEQ_BITS, 32)/8), &memory_footprint); // output sequence of rate match, 32-bit aligned and multiple of 4 bytes
}

void SsbTx::allocateDescr()
{
    std::array<size_t, N_SSB_COMPONENTS> dynDescrSizeBytes{};
    std::array<size_t, N_SSB_COMPONENTS> dynDescrAlignBytes{};

    size_t* pDynDescrSizeBytes  = dynDescrSizeBytes.data();
    size_t* pDynDescrAlignBytes = dynDescrAlignBytes.data();

    //Allocate workspace memory to take advantage of the bulk async copy of the descriptors.

    // Allocate memory for parameters
    pDynDescrSizeBytes[SSB_PER_CELL_PARAMS]  = sizeof(cuphyPerCellSsbDynPrms_t) * max_cells_per_slot;
    pDynDescrAlignBytes[SSB_PER_CELL_PARAMS] = alignof(cuphyPerCellSsbDynPrms_t);

    pDynDescrSizeBytes[PER_SS_BLOCK_PARAMS]  = max_SSBs_per_slot * sizeof(cuphyPerSsBlockDynPrms_t);
    pDynDescrAlignBytes[PER_SS_BLOCK_PARAMS] = alignof(cuphyPerSsBlockDynPrms_t);

    pDynDescrSizeBytes[SSB_PRECODING_MATRIX]  = max_SSBs_per_slot * sizeof(cuphyPmWOneLayer_t);
    pDynDescrAlignBytes[SSB_PRECODING_MATRIX] = alignof(cuphyPmWOneLayer_t);

    // Allocate memory for input array after CRC attachment. TODO This computation is currently done on the host.
    // FIXME  x_crc per SSB has to be uint32_t aligned so use aligned_x_crc size
    int aligned_x_crc_size = round_up_to_next(CUPHY_SSB_N_PBCH_SEQ_W_CRC_BITS, 32)/ 8;
    pDynDescrSizeBytes[SSB_INPUT_W_CRC]  = max_SSBs_per_slot * aligned_x_crc_size * sizeof(uint8_t);
    pDynDescrAlignBytes[SSB_INPUT_W_CRC] = alignof(uint32_t);

    pDynDescrSizeBytes[CELL_OUTPUT_ADDR]  = max_cells_per_slot * sizeof(__half2*);
    pDynDescrAlignBytes[CELL_OUTPUT_ADDR] = alignof(__half2*);

    m_component_descrs.alloc(dynDescrSizeBytes, dynDescrAlignBytes, &memory_footprint);
    //m_component_descrs.displayDescrSizes();

    // Added for convenience
    h_x_crc = (uint8_t*)m_component_descrs.getCpuStartAddrs()[SSB_INPUT_W_CRC];
    d_x_crc = (uint8_t*)m_component_descrs.getGpuStartAddrs()[SSB_INPUT_W_CRC];

}


// generate gold sequence with given init state and length
void genGoldSeq(uint32_t c_init, uint32_t len, uint32_t *x1, uint32_t *x2, uint32_t *c)
{
    for(int i = 0; i < 32; i++) {
        x1[i] = 0;
        x2[i] = (c_init >> i) & 0x1;
    }
    x1[0] = 1;

    for(int i = 0; i < Nc + len - 31; i++) {
        x1[i + 31] = (x1[i + 3] + x1[i]) % 2;                         // x1(n + 31) = mod(x1(n + 3) + x1(n),2)
        x2[i + 31] = (x2[i + 3] + x2[i + 2] + x2[i + 1] + x2[i]) % 2; // x2(n + 31) = mod(x2(n + 3) + x2(n + 2) + x2(n + 1) + x2(n),2)
    }

    for(int i = 0; i < len; i++) {
        c[i] = (x1[i + Nc] + x2[i + Nc]) % 2; // c(n) = mod(x1(n + Nc) + x2(n + Nc),2)
    }
}

// reverse bits order within each byte
int reverseBitInByte(uint8_t * inputByte, uint8_t * outputByte, uint32_t Nbyte)
{
    for (int idxByte = 0; idxByte < Nbyte; idxByte++) {
        outputByte[idxByte] = 0;
        for (int idxBit = 0; idxBit < 8; idxBit ++) {
            outputByte[idxByte] <<= 1;
            outputByte[idxByte] += (inputByte[idxByte] >> idxBit) & 1;
        }
    }
    return 0;
}


// genereate crc bits for pbch payload
template <typename uintCRC_t, size_t uintCRCBitLength>
uintCRC_t computeCRC(const uint8_t* input,
                     uint32_t       size,
                     uintCRC_t      poly,
                     uintCRC_t      initVal = 0,
                     uint64_t       stride  = 1)
{
    uintCRC_t crc         = initVal;
    uintCRC_t msbMask     = (1 << (uintCRCBitLength - 1));
    uintCRC_t allOnesMask = static_cast<uintCRC_t>(-1);
    if((sizeof(uintCRC_t) * 8 - uintCRCBitLength) > 0)
        allOnesMask >>= (sizeof(uintCRC_t) * 8 - uintCRCBitLength - 1);
    for(int i = 0; i < size * stride; i += stride)
    {
        crc ^= static_cast<uintCRC_t>(input[i] << (uintCRCBitLength - 8));
        for(int b = 0; b < 8; b++)
        {
            uintCRC_t pred = (crc & msbMask) == 0;
            crc <<= 1;
            crc ^= (poly & (pred + allOnesMask));
        }
    }
    return crc;
}


//Assume x_mib is packed bits per uint32_t
void pbchGenPayload_per_SSB(uint32_t* x_mib,
                            uint32_t* x_payload,
                            uint8_t   SSB_block_index,
                            const cuphyPerCellSsbDynPrms_t*   cell_params)
{
   // The CUPHY_SSB_N_PBCH_PAYLOAD_BITS 32-bit wide sequence has always bits 0, 6 and 24 set to 1, whereas bits 2, 3 and 5 are only set if Lmax==64.
   // Can change bit order if more convenient

   const uint32_t no_scrambling_Lmax_ne_64 = 0x01000041;
   const uint32_t no_scrambling_Lmax_eq_64 = (no_scrambling_Lmax_ne_64 | 0x0000001c);
   const uint32_t no_scrambling_val = (cell_params->Lmax == 64) ? no_scrambling_Lmax_eq_64 : no_scrambling_Lmax_ne_64; // Reminder Lmax is per cell/slot

   const int a_from_abar_mapping[32] =
   {28, 0, 31, 30, 7, 29, 25, 27,
     5, 8, 24,  9, 10, 11, 12, 13,
     1, 4,  3, 14, 15, 16, 17, 2,
    26, 18, 19, 20, 21, 22, 6, 23};

   uint32_t a0_3 = cell_params->SFN & 0xf; // cell
   uint32_t a5_7, M;

    if (cell_params->Lmax == 64) {
        a5_7 = (SSB_block_index & 0x3f) >> 3;
        M = 32-6;
    }
    else if ((cell_params->Lmax == 4) || (cell_params->Lmax == 8)) {
        uint32_t msb_k_ssb = cell_params->k_SSB >> 4; // cell
        a5_7 = msb_k_ssb << 2;
        M = 32-3;
    }
    else {
        NVLOGF_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "Invalid Lmax value. L_max can only be 4, 8 or 64.");
    }

    uint32_t a_bar = x_mib[0] << (32 - CUPHY_SSB_N_MIB_BITS); // Assumption that MIB occupies the least significant bits and all  unused MIB bits set to 0; can also mask to be safe
    a_bar |= (a0_3 << 4);
    a_bar |= ((cell_params->nHF & 0x1) << 3);
    a_bar |= (a5_7 & 0x7);
    //all 32-bits of a_bar are used.

    uint32_t v = (a0_3 & 6) >> 1;
    uint32_t len = v*M + 32;

    // FIXME gold seq. gen and x_payload below are one bit per uint32_t element could just pack them and then do bitwise operations
    // Gold seq. generation
    uint32_t x1[Nc + len], x2[Nc + len], c[len];
    genGoldSeq(cell_params->NID, len, x1, x2, c); // populates array c, with one valid bit per uint32_t.

    uint32_t j = 0;
    x_payload[0] = 0;
    for (int i = 0; i < CUPHY_SSB_N_PBCH_PAYLOAD_BITS; i++) {
         uint32_t a = (a_bar >> (31-a_from_abar_mapping[i])) & 0x1;

         if (((no_scrambling_val >> i) & 0x1) == 0) {
             x_payload[0] |= (((a + c[j + v*M]) & 1) << (31 - i));
             j++;
         } else {
             x_payload[0] |= ((a & 1) << (31 - i));
         }
    }
}

cuphyStatus_t SsbTx::preparePBCH(uint32_t*                                 h_x_mib,
                                 const cuphyPerSsBlockDynPrms_t*           h_ssb_params,
                                 const cuphyPerCellSsbDynPrms_t*           h_per_cell_params,
                                 cuphyEncoderRateMatchMultiSSBLaunchCfg_t* pEncdRMLaunchCfg,
                                 cuphySsbMapperLaunchCfg_t*                pSsbMapperLaunchCfg)
{
    if ((h_x_mib == nullptr) || (h_ssb_params == nullptr) || (h_per_cell_params == nullptr) ||
        (num_SSBs == 0) || (num_cells == 0))
    {
        return CUPHY_STATUS_INVALID_ARGUMENT;
    }

    cuphyStatus_t status;

    // Go over all SSBs and populate the h_x_crc host buffer based on the SSB and cell parameters and the SSB MIB input data.
    // The h_x_crc buffer will be copied to the GPU as part of the SSB pipeline before calling Run.
    for (int i = 0; i < num_SSBs; i++) {

        uint32_t* x_mib_for_SSB = h_x_mib + i; // The MIB input buffer contains one uint32_t element per SSB

        // Generate 32-bit payload, x_payload_packed, for this SSB. The following parameter fields are used:
        //  -  SSB specific: blockIndex
        //  -  cell specific: SFN, Nhf, NID, Lmax, k_SSB

        uint32_t x_payload_packed; // packed; old x_payload[0] is at least significant x_payload bit
        uint16_t cell_index = h_ssb_params[i].cell_index;
        if (cell_index > num_cells) {
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
        pbchGenPayload_per_SSB(x_mib_for_SSB, &x_payload_packed, h_ssb_params[i].blockIndex, h_per_cell_params + cell_index);  // Note x_payload is packed 32 bits


        // add CRC to PBCH payload
        uint8_t addCrc[CUPHY_SSB_N_PBCH_SEQ_W_CRC_BITS/8];
        uint32_t totalOutTBByteSize = CUPHY_SSB_N_PBCH_SEQ_W_CRC_BITS/8;
        for (int idxByte = 0; idxByte < 4; idxByte ++) { // 4 = CUPHY_SSB_N_PBCH_PAYLOAD_BITS/8
            addCrc[idxByte] = (x_payload_packed >> (3-idxByte)*8);
        }

        // Compute 24-bit CRC
        uint32_t crc = computeCRC<uint32_t, 24>(addCrc, 4, G_CRC_24_C, 0, 1);

        // Fill in 24-bit CRC into bytes 4-6 of addCrc
        addCrc[4] = (crc >> 16) & 0xFF;
        addCrc[5] = (crc >> 8) & 0xFF;
        addCrc[6] = (crc >> 0) & 0xFF;

        // Reverse bits in byte in addCRC and store output in h_x_crc buffer to match with expected polar encoder input format
        int aligned_x_crc_size = 8; //round_up_to_next(CUPHY_SSB_N_PBCH_SEQ_W_CRC_BITS, 32)/ 8;
        reverseBitInByte(addCrc, h_x_crc + i * aligned_x_crc_size, totalOutTBByteSize);
    }

    // set SSB kernels launch configs
    status = cuphySSBsKernelSelect(pEncdRMLaunchCfg, pSsbMapperLaunchCfg, num_SSBs);

    return status;
}


cuphyStatus_t SsbTx::expandParameters(cuphySsbDynPrms_t* dyn_params,
                             cudaStream_t         cuda_strm)
{
    num_cells = dyn_params->nCells;
    num_SSBs  = dyn_params->nSSBlocks;

    //The extra host to host copy is needed if we want the respective H2D copy to be part of the single bulk async. copy.
    //FIXME can skip and have separate H2D copies instead
    memcpy(m_component_descrs.getCpuStartAddrs()[SSB_PER_CELL_PARAMS],
           dyn_params->pPerCellSsbDynParams,
           sizeof(cuphyPerCellSsbDynPrms_t) * num_cells); // reminder the bulk copy will copy max_cells_per_slot

    memcpy(m_component_descrs.getCpuStartAddrs()[PER_SS_BLOCK_PARAMS],
           dyn_params->pPerSsBlockParams,
           sizeof(cuphyPerSsBlockDynPrms_t) * num_SSBs);

    if(dyn_params->nPrecodingMatrices>0)
    {
        memcpy(m_component_descrs.getCpuStartAddrs()[SSB_PRECODING_MATRIX],
               dyn_params->pPmwParams, sizeof(cuphyPmWOneLayer_t) * dyn_params->nPrecodingMatrices);
    }

    cuphyStatus_t status = preparePBCH(dyn_params->pDataIn->pMibInput,
                                       dyn_params->pPerSsBlockParams,
                                       dyn_params->pPerCellSsbDynParams,
                                       &m_encodeRmMultiSsbLaunchCfg,
                                       &m_ssbMapperLaunchCfg);

    if (status != CUPHY_STATUS_SUCCESS) {
        NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "Error during SSB setup.");
        return status;
    }

    __half2** h_output_addr = (__half2**)m_component_descrs.getCpuStartAddrs()[CELL_OUTPUT_ADDR];
    for (int i = 0; i < num_cells; i++) {
        h_output_addr[i] = (__half2*)dyn_params->pDataOut->pTDataTx[i].pAddr;
    }

    // Bulk async H2D copy for all workspaces
    if(bulk_desc_async_copy)
    {
        m_component_descrs.asyncCpuToGpuCpy(cuda_strm);
    }
    return CUPHY_STATUS_SUCCESS;
}


void SsbTx::createGraph()
{
#if CUDART_VERSION < 11000
    NVLOGF_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "Graph mode requires CUDA driver kernel node params which requires CUDA 11.0 or higher.");
#endif

    CU_CHECK_EXCEPTION(cuGraphCreate(&m_graph, 0));
    // Add node(s). Initially start with some kernel parameters, and at setup do the updating

    // Set empty graph kernel nodes with the appropriate argument count (all pointers) to avoid dynamic
    // memory allocation during graph kernel node update. If the number of kernel parameters changes, the calls below should be updated.
    void* arg;
    void* kernelParams[5] = {&arg, &arg, &arg, &arg, &arg}; // use max. number of kernel args for array size
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode5ParamsDriver, 5, &kernelParams[0]));
    CUPHY_CHECK(cuphySetGenericEmptyKernelNodeParams(&m_emptyNode3ParamsDriver, 3, &kernelParams[0]));

    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_encdRmMultiSsbNode, m_graph, nullptr, 0, &m_emptyNode3ParamsDriver));
    CU_CHECK_EXCEPTION(cuGraphAddKernelNode(&m_ssbMapperNode, m_graph, &m_encdRmMultiSsbNode, 1, &m_emptyNode5ParamsDriver));
}

void SsbTx::updateGraph()
{
#if CUDART_VERSION < 11000
    NVLOGF_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "Graph mode requires CUDA driver kernel node params which requires CUDA 11.0 or higher.");
#endif

    CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_encdRmMultiSsbNode, &(m_encodeRmMultiSsbLaunchCfg.kernelNodeParamsDriver)));
    CU_CHECK_EXCEPTION(cuGraphExecKernelNodeSetParams(m_graphExec, m_ssbMapperNode, &(m_ssbMapperLaunchCfg.kernelNodeParamsDriver)));
}

cuphyStatus_t SsbTx::setup(cuphySsbDynPrms_t* dyn_params)
{
    dynamic_params = dyn_params; // Needed to retrieve CUDA stream etc.
    //printSsbConfig<fmtlog::DBG>(*dynamic_params); //FIXME comment out or add a single check if log level < comp. nvlog level and return immediately to avoid runtime checks overhead
    cuphyStatus_t status = expandParameters(dyn_params, dyn_params->cuStream);
    if (status != CUPHY_STATUS_SUCCESS) {
        return status;
    }
    dyn_params->chan_graph = &m_graph;
    //---------------------------------------------
    // set kernel args for encodeRateMatchMultipleSSBsKernel()
    m_encdRmSSBArgs[0] = d_x_crc;
    m_encdRmSSBArgs[1] = d_x_coded.get();
    m_encdRmSSBArgs[2] = d_x_tx.get();

    m_encodeRmMultiSsbLaunchCfg.kernelArgs[0] = &m_encdRmSSBArgs[0];
    m_encodeRmMultiSsbLaunchCfg.kernelArgs[1] = &m_encdRmSSBArgs[1];
    m_encodeRmMultiSsbLaunchCfg.kernelArgs[2] = &m_encdRmSSBArgs[2];

    m_encodeRmMultiSsbLaunchCfg.kernelNodeParamsDriver.kernelParams = m_encodeRmMultiSsbLaunchCfg.kernelArgs;

    // set kernel args for ssbModTfSigKernel()
    m_ssbMapperArgs[0] = m_component_descrs.getGpuStartAddrs()[CELL_OUTPUT_ADDR];
    m_ssbMapperArgs[1] = d_x_tx.get();
    m_ssbMapperArgs[2] = m_component_descrs.getGpuStartAddrs()[PER_SS_BLOCK_PARAMS];
    m_ssbMapperArgs[3] = m_component_descrs.getGpuStartAddrs()[SSB_PER_CELL_PARAMS];
    m_ssbMapperArgs[4] = m_component_descrs.getGpuStartAddrs()[SSB_PRECODING_MATRIX];

    m_ssbMapperLaunchCfg.kernelArgs[0] = &m_ssbMapperArgs[0];
    m_ssbMapperLaunchCfg.kernelArgs[1] = &m_ssbMapperArgs[1];
    m_ssbMapperLaunchCfg.kernelArgs[2] = &m_ssbMapperArgs[2];
    m_ssbMapperLaunchCfg.kernelArgs[3] = &m_ssbMapperArgs[3];
    m_ssbMapperLaunchCfg.kernelArgs[4] = &m_ssbMapperArgs[4];

    m_ssbMapperLaunchCfg.kernelNodeParamsDriver.kernelParams = m_ssbMapperLaunchCfg.kernelArgs;
    //---------------------------------------------

    //executable graph setup
    m_cudaGraphModeEnabled = (dynamic_params->procModeBmsk & CSIRS_PROC_MODE_GRAPHS) ? true : false;
    if(m_cudaGraphModeEnabled)
    {
        updateGraph();
    }
    return CUPHY_STATUS_SUCCESS;
}


int SsbTx::run(const cudaStream_t& cuda_strm)
{

    if(m_cudaGraphModeEnabled)
    {
        MemtraceDisableScope md; // Disable temporarily
        CUresult e = cuGraphLaunch(m_graphExec, cuda_strm);
        if (e != CUDA_SUCCESS) {
            NVLOGE_FMT(NVLOG_SSB, AERIAL_CUPHY_EVENT, "Invalid graph launch for SSB.");
            return 1;
        }
    }
    else
    {
        // Polar encoder and rate matching kernel processing multiple SSBs.
        // Offsets into d_x_crc and d_x_coded and d_x_tx buffers per SSB are hard-coded internally.
        cuphyStatus_t polar_enc_status = cuphyRunPolarEncRateMatchSSBs(&m_encodeRmMultiSsbLaunchCfg,
                                                                       d_x_crc,
                                                                       d_x_coded.get(),
                                                                       d_x_tx.get(),
                                                                       num_SSBs,
                                                                       cuda_strm);


        // modulate PBCH, DMRS, PSS and SSS sequences and map to time frequency domain subcarriers
        cuphyStatus_t ssb_mapper_status = cuphyRunSsbMapper(d_x_tx.get(),
                                                 (__half2**)m_component_descrs.getGpuStartAddrs()[CELL_OUTPUT_ADDR],
                                                 (const cuphyPerSsBlockDynPrms_t*)m_component_descrs.getGpuStartAddrs()[PER_SS_BLOCK_PARAMS],
                                                 (const cuphyPerCellSsbDynPrms_t*)m_component_descrs.getGpuStartAddrs()[SSB_PER_CELL_PARAMS],
                                                 (const cuphyPmWOneLayer_t*)m_component_descrs.getGpuStartAddrs()[SSB_PRECODING_MATRIX],
                                                 num_SSBs,
                                                 num_cells,
                                                 cuda_strm,
                                                 &m_ssbMapperLaunchCfg);


        if ((ssb_mapper_status != CUPHY_STATUS_SUCCESS) || (polar_enc_status != CUPHY_STATUS_SUCCESS)) {
            return 1;
            //throw std::runtime_error("\nError for cuphySsbTxPipeline\n");
        }
    }

    return 0;
}

SsbTx::~SsbTx()
{
    CUDA_CHECK(cudaGraphDestroy(m_graph));
    CUDA_CHECK(cudaGraphExecDestroy(m_graphExec));
}
