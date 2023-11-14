/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


//#define CUPHY_DEBUG 1

#include "ldpc_ms_mka_flooding.hpp"
#include "ldpc.cuh"
#include "ldpc_c2v_message.cuh"

////////////////////////////////////////////////////////////////////////
// workspace_ms_mka_flooding
template <typename TLLR>
struct workspace_ms_mka_flooding : public LDPC_workspace<workspace_ms_mka_flooding<TLLR>>
{
    typedef TLLR                                            LLR_t;
    typedef c2v_message_t<TLLR>                             message_t;
    typedef LDPC_workspace<workspace_ms_mka_flooding<TLLR>> inherited_t;
    //------------------------------------------------------------------
    // Constructor
    __device__
    workspace_ms_mka_flooding(void* pv, const LDPC_config& config) :
        inherited_t(pv)
    {
    }
    //------------------------------------------------------------------
    // workspace_bytes_per_codeword()
    // For each codeword:
    //     C2V: (mb * Z) instances of c2v_message_t
    //     APP: ((Kb + 4) * Z) * sizeof(TLLR)
    CUDA_BOTH_INLINE
    static size_t workspace_bytes_per_codeword(const LDPC_config& config)
    {
        unsigned int sC2V  = (config.mb * config.Z * sizeof(message_t));
        unsigned int szAPP = ((config.Kb + 4) * config.Z) * sizeof(TLLR);
        // Make sure that the data for each codeword is aligned to at
        // least 8 bytes
        return ((szAPP + sC2V + 7) / 8) * 8;
    }
    __device__
        message_t*
        C2V(const LDPC_config& config, int codewordIndex)
    {
        return inherited_t::template offset_as<message_t>(codeword_base_offset(config, codewordIndex));
    }
    __device__
        LLR_t*
        APP(const LDPC_config& config, int codewordIndex)
    {
        return inherited_t::template offset_as<LLR_t>(codeword_base_offset(config, codewordIndex) + app_offset(config));
    }

private:
    __device__
        size_t
        codeword_base_offset(const LDPC_config& config, int codewordIndex)
    {
        return (workspace_bytes_per_codeword(config) * codewordIndex);
    }
    __device__
        size_t
        app_offset(const LDPC_config& config) const
    {
        // APP buffer is after the C2V buffer in each codeword segment
        return (config.mb * config.Z * sizeof(message_t));
    }
};

template <cuphyDataType_t TLLREnum>
__global__ void ldpc_cp_llr(void*                                workspaceAddress,
                            LDPC_config                          config,
                            const_tensor_ref_contig_2D<TLLREnum> tLLRSrc)
{
    typedef typename data_type_traits<TLLREnum>::type LLR_t;

    const unsigned int               NLLR = config.Z * ((1 == config.BG) ? 26 : 14);
    const unsigned int               NCW  = tLLRSrc.layout().dimensions[1];
    workspace_ms_mka_flooding<LLR_t> workspace(workspaceAddress, config);
    for(int codeWordIndex = blockIdx.y; codeWordIndex < NCW; codeWordIndex += gridDim.y)
    {
        LLR_t* cwAPP = workspace.APP(config, codeWordIndex);
        for(int LLRIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
            LLRIndex < NLLR;
            LLRIndex += (gridDim.x * blockDim.x))
        {
            //KERNEL_PRINT_IF(blockIdx.x == 9, "threadIdx.x = %u, blockIdx.x = %u, LLRIndex = %i, codeWordIndex = %i, NLLR = %i, inc = %i\n", threadIdx.x, blockIdx.x, LLRIndex, codeWordIndex, NLLR, (gridDim.x * blockDim.x));
            cwAPP[LLRIndex] = tLLRSrc({LLRIndex, codeWordIndex});
        }
    }
}

template <cuphyDataType_t TLLREnum>
__global__ void ldpc_hard_decision(LDPC_output_t tOutput,
                                   void*         workspaceAddress,
                                   LDPC_config   config)
{
    typedef typename data_type_traits<TLLREnum>::type LLR_t;
    const int                                         CODEWORD_IDX = blockIdx.x;
    workspace_ms_mka_flooding<LLR_t>                  workspace(workspaceAddress, config);
    LLR_t*                                            dAPP            = workspace.APP(config, CODEWORD_IDX);
    const int                                         K               = config.Kb * config.Z;
    const int                                         WORDS_PER_CW    = (K + 31) / 32;
    const int                                         BIT_BLOCK_COUNT = (K + 1023) / 1024;
    const int                                         WARPS_PER_BLOCK = blockDim.x / 32;
    const int                                         WARP_IDX        = threadIdx.x / 32;
    const int                                         LANE_IDX        = threadIdx.x % 32;
    // Write output bits. Each warp of 32 threads will cooperate to
    // generate up to 32 output "words", and each of those "words" will
    // contain 32 decision bits. (Each warp will read 1024 LLR values,
    // and generate 1024 output bits in 32 uint32_t words.)
    // The maximum codeword size is 8448 bits, which corresponds to
    // 8448 / 32 = 264 32-bit words for output.
    for(int iOutBlock = WARP_IDX; iOutBlock < BIT_BLOCK_COUNT; iOutBlock += WARPS_PER_BLOCK)
    {
        uint32_t thread_output = 0;
        int      start_bit_idx = iOutBlock * 1024;
        for(int i = 0; i < 32; ++i)
        {
            int      idx           = start_bit_idx + (i * 32) + LANE_IDX;
            uint32_t hard_decision = ((idx < K) && is_neg(dAPP[idx])) ? 1 : 0;
            uint32_t warp_bits     = __ballot_sync(0xFFFFFFFF, hard_decision);
            if(i == LANE_IDX)
            {
                thread_output = warp_bits;
            }
        }
        const int OUT_INDEX = (iOutBlock * 32) + LANE_IDX;
        //KERNEL_PRINT("threadidx.x = %u, iOutBlock = %i, OUT_INDEX = %i\n", threadIdx.x, iOutBlock, OUT_INDEX);
        if(OUT_INDEX < WORDS_PER_CW)
        {
            //KERNEL_PRINT_IF(0 == OUT_INDEX, "output[0] = 0x%X\n", thread_output);
            tOutput({OUT_INDEX, CODEWORD_IDX}) = thread_output;
        }
    }
}

template <cuphyDataType_t TLLREnum>
__global__
    __launch_bounds__(384, 2/*5*/) void ldpc_flooding_multi_kernel_atomic(LDPC_config                          config,
                                                                          float                                normalization,
                                                                          const_tensor_ref_contig_2D<TLLREnum> tLLR,
                                                                          void*                                workspaceAddress,
                                                                          int                                  iteration)
{
    typedef typename data_type_traits<TLLREnum>::type LLR_t;
    typedef c2v_message_t<LLR_t>                      message_t;
    const int                                         CODEWORD_INDEX       = blockIdx.y;
    const int                                         CHECK_IDX            = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int                                         NODE_IDX             = CHECK_IDX / config.Z;
    const int                                         NODE_OFFSET          = CHECK_IDX % config.Z;
    const int                                         CHECK_COUNT          = config.mb * config.Z;
    const bool                                        THREAD_HAS_CHECK     = (CHECK_IDX < CHECK_COUNT);
    const int                                         NUM_APP_NODES_STORED = config.Kb + 4;
    workspace_ms_mka_flooding<LLR_t>                  workspace(workspaceAddress, config);
    const LLR_t*                                      channelLLR = &tLLR({0, CODEWORD_INDEX});
    c2v_message_t<LLR_t>*                             C2V        = workspace.C2V(config, CODEWORD_INDEX);
    LLR_t*                                            dAPP       = workspace.APP(config, CODEWORD_INDEX);
    //KERNEL_PRINT_GRID_ONCE("ldpc_flooding_multi_kernel_atomic, CHECK_IDX = %i, CHECK_COUNT = %i\n", CHECK_IDX, CHECK_COUNT);
    //c2vMessage.init();
#if 0
    if(0 == blockIdx.y)
    {
        for(int idx = (blockIdx.x * blockDim.x) + threadIdx.x;
            idx < (NUM_APP_NODES_STORED * config.Z);
            idx += (gridDim.x * blockDim.x))
        {
            KERNEL_PRINT_IF(idx < 32, "iter = %i, APP[%i] = %f\n", iteration, idx, to_float(dAPP[idx]));
        }
    }
#endif
    if(THREAD_HAS_CHECK)
    {
        bg1_CN_row_shift_info_t CNShift(NODE_IDX, config.Z);
        message_t               c2vOut;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        if(0 == iteration)
        {
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // First iteration: use channel LLR as input
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Iterate over received LLR data to generate the first APP
            // increment.
            c2vOut = message_t::create_message(CNShift,     // shift data
                                               NODE_OFFSET, // offset of check variable within node
                                               config.Z,    // lifting factor
                                               channelLLR);
            // Increment APP values based on C2V data
            for(int iVN = 0; iVN < CNShift.row_degree; ++iVN)
            {
                if(CNShift.column_values[iVN] < NUM_APP_NODES_STORED)
                {
                    LLR_t     llrValue = c2vOut.get_value_for_index(iVN, normalization);
                    const int VN_idx   = message_t::get_variable_index(CNShift, iVN, NODE_OFFSET, config.Z);
                    atomicAdd(dAPP + VN_idx, llrValue);
                    // WRONG! Used instead of previous line, only to time impact of atomic
                    //dAPP[VN_idx] = llrValue;
                    //KERNEL_PRINT_IF(32 == VN_idx, "Adding %f to APP[%i] (CHECK_IDX = %i, iVN = %i, threadIdx.x = %u, blockIdx.x = %u)\n",
                    //                to_float(llrValue), VN_idx, CHECK_IDX, iVN, threadIdx.x, blockIdx.x);
                }
            }
        }
        else
        {
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // After the first iteration: read previous C2V message and
            // generate values used as inputs to the next C2V
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   -  -  -
            // Load the C2V message from the previous iteration
            message_t c2vIn;
            load_c2v_message(&c2vIn, C2V + CHECK_IDX);
            //KERNEL_PRINT("CHECK_IDX = %i, min0 = %.4f, min1 = %.4f, sign_index = 0x%X\n", CHECK_IDX, to_float(c2vIn.min0), to_float(c2vIn.min1), c2vIn.sign_index);
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   -  -  -
            // Initialize a new message for the next iteration. We will
            // write this out before the kernel ends.
            c2vOut.init();
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   -  -  -
            // Determine the max row index. We only store APP values for
            // BG1 up to the 26th variable node. For variables beyond that,
            // the C2V message is just the LLR value. We check the last
            // column value (instead of just looking at the check node
            // index), just in case we rearrange rows later.
            int rowIndexEnd = CNShift.row_degree;
            if(CNShift.column_values[CNShift.row_degree - 1] >= NUM_APP_NODES_STORED)
            {
                --rowIndexEnd;
            }
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   -  -  -
            for(int iVN = 0; iVN < rowIndexEnd; ++iVN)
            {
                // Calculate the V2C message from last iteration
                LLR_t     prevValue = c2vIn.get_value_for_index(iVN, normalization);
                const int VN_idx    = message_t::get_variable_index(CNShift, iVN, NODE_OFFSET, config.Z);
                LLR_t     v2c       = dAPP[VN_idx] - prevValue;
                // Use the new V2C value to update the C2V message for the next
                // iteration.
                //KERNEL_PRINT_IF(0 == CHECK_IDX, "CHECK_IDX = %i, iVN = %i, APP[%i] = %.4f, prevValue = %.4f, V2C = %.4f\n",
                //                CHECK_IDX, iVN, VN_idx, to_float(dAPP[VN_idx]), to_float(prevValue), to_float(v2c));
                c2vOut.process(v2c, iVN);
            } // iVN
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   -  -  -
            // If this check node has an extension parity node, process it
            if(rowIndexEnd != CNShift.row_degree)
            {
                const int VN_idx = message_t::get_variable_index(CNShift, rowIndexEnd, NODE_OFFSET, config.Z);
                LLR_t     v2c    = channelLLR[VN_idx];
                // Update the C2V message for the next iteration.
                c2vOut.process(v2c, rowIndexEnd);
            }
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   -  -  -
            // Increment APP values based on C2V data
            for(int iVN = 0; iVN < CNShift.row_degree; ++iVN)
            {
                if(CNShift.column_values[iVN] < NUM_APP_NODES_STORED)
                {
                    const int VN_idx = message_t::get_variable_index(CNShift, iVN, NODE_OFFSET, config.Z);
                    LLR_t     llrNew = c2vOut.get_value_for_index(iVN, normalization);
                    LLR_t     llrOld = c2vIn.get_value_for_index(iVN, normalization);
                    atomicAdd(dAPP + VN_idx, llrNew - llrOld);
                    // WRONG! Used instead of previous line, only to time impact of atomic
                    //dAPP[VN_idx] = llrNew - llrOld;
                    //KERNEL_PRINT_IF(32 == VN_idx, "Adding %f to APP[%i] (delta = %f) (CHECK_IDX = %i, iVN = %i, threadIdx.x = %u, blockIdx.x = %u)\n",
                    //                to_float(llrNew), to_float(llrNew - llrOld), VN_idx, CHECK_IDX, iVN, threadIdx.x, blockIdx.x);
                }
            }
        }
        //--------------------------------------------------------------
        // Write C2V for the next iteration
        //KERNEL_PRINT_IF(CHECK_IDX < 128, "Writing C2V message with min0 = %f, min1 = %f, signs = 0x%X, row_index = %i, CHECK_IDX = %i\n",
        //                                 to_float(c2vOut.min0),
        //                                 to_float(c2vOut.min1),
        //                                 c2vOut.get_signs(),
        //                                 c2vOut.get_row_index(),
        //                                 CHECK_IDX);
        C2V[CHECK_IDX] = c2vOut;
    }
}

////////////////////////////////////////////////////////////////////////
// launch_decode_multi_kernel_atomic()
template <cuphyDataType_t TType>
cuphyStatus_t launch_decode_multi_kernel_atomic(LDPC_output_t&      tOutputWord,
                                                const_tensor_pair&  tLLRInput,
                                                const LDPC_config&  config,
                                                float               normalization,
                                                cuphyLDPCResults_t* results,
                                                void*               workspace,
                                                cudaStream_t        strm)
{
    DEBUG_PRINTF("ldpc::launch_decode_multi_kernel_atomic()\n");
    typedef const_tensor_ref_contig_2D<TType>      const_tensor2f;
    typedef typename data_type_traits<TType>::type LLR_t;
    typedef c2v_message_t<LLR_t>                   message_t;

    // The kernel is only implemented for contiguous, 2D tensors.
    // Attempt to convert to such a tensor descriptor.
    cuphy_optional<const_tensor2f> tOptLLR = tLLRInput.first.get().get_ref_contig_rank<TType, 2>(tLLRInput.second);
    if(!tOptLLR)
    {
        // Layout is not 2D contiguous
        return CUPHY_STATUS_UNSUPPORTED_LAYOUT;
    }
    //const int          SHMEM_SIZE = 0;
    const int NUM_STORED_LLR = (config.Kb + 4) * config.Z;
    //------------------------------------------------------------------
    // Perform initial copy of LLR data so that we can use atomic increment
    {
        dim3 blockDim(1024);
        dim3 gridDim(div_round_up(NUM_STORED_LLR, 1024), config.num_codewords);
        ldpc_cp_llr<TType><<<gridDim, blockDim, 0, strm>>>(workspace, config, tOptLLR.value());
    }

    //------------------------------------------------------------------
    dim3 blkDimCN(config.Z);
    dim3 grdDimCN(config.mb, config.num_codewords);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc_flooding_multi_kernel_atomic<TType>);
    DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc_flooding_multi_kernel_atomic<TType>, blkDimCN, 0);
    for(int iter = 0; iter < config.max_iterations; ++iter)
    {
        ldpc_flooding_multi_kernel_atomic<TType><<<grdDimCN, blkDimCN, 0, strm>>>(config,
                                                                                  normalization,
                                                                                  tOptLLR.value(),
                                                                                  workspace,
                                                                                  iter);
    }
    {
        dim3 blkDimHD(256);
        dim3 grdDimHD(config.num_codewords);
        DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc_hard_decision<TType>);
        DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc_hard_decision<TType>, blkDimHD, 0);
        ldpc_hard_decision<TType><<<grdDimHD, blkDimHD, 0, strm>>>(tOutputWord,
                                                                   workspace,
                                                                   config);
    }
#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif

    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

namespace ldpc
{
////////////////////////////////////////////////////////////////////////
// decode_multi_kernel_atomic()
cuphyStatus_t decode_multi_kernel_atomic(LDPC_output_t&      tDst,
                                         const_tensor_pair&  tLLR,
                                         const LDPC_config&  config,
                                         float               normalization,
                                         cuphyLDPCResults_t* results,
                                         void*               workspace,
                                         cudaStream_t        strm)
{
    DEBUG_PRINTF("ldpc::decode_multi_kernel_atomic()\n");
    switch(tLLR.first.get().type())
    {
    case CUPHY_R_32F:
        return launch_decode_multi_kernel_atomic<CUPHY_R_32F>(tDst,
                                                              tLLR,
                                                              config,
                                                              normalization,
                                                              results,
                                                              workspace,
                                                              strm);
    case CUPHY_R_16F:
        return launch_decode_multi_kernel_atomic<CUPHY_R_16F>(tDst,
                                                              tLLR,
                                                              config,
                                                              normalization,
                                                              results,
                                                              workspace,
                                                              strm);
    default:
        return CUPHY_STATUS_SUCCESS;
    }
}

std::pair<bool, size_t> decode_multi_kernel_atomic_workspace_size(const LDPC_config& config)
{
    switch(config.type)
    {
    case CUPHY_R_32F:
    {
        typedef data_type_traits<CUPHY_R_32F>::type LLR_t;
        typedef workspace_ms_mka_flooding<LLR_t>    workspace_t;
        return std::pair<bool, size_t>(true,
                                       workspace_t::get_workspace_size(config));
    }
    case CUPHY_R_16F:
    {
        typedef data_type_traits<CUPHY_R_16F>::type LLR_t;
        typedef workspace_ms_mka_flooding<LLR_t>    workspace_t;
        return std::pair<bool, size_t>(true,
                                       workspace_t::get_workspace_size(config));
    }
    default:
        return std::pair<bool, size_t>(false, 0);
    }
}

} // namespace ldpc
