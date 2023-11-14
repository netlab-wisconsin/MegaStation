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

#include "ldpc_ms_cta_flooding.hpp"
#include "ldpc.cuh"
#include "ldpc_c2v_message.cuh"

namespace
{
////////////////////////////////////////////////////////////////////////
// workspace_ms_cta_flooding
// Class to represent the workspace required by this LDPC implementation.
// API callers will query the workspace size offline and allocate a
// buffer to provide to the implementation to avoid allocating memory
// in each call.
template <typename TLLR>
class workspace_ms_cta_flooding : public LDPC_workspace<workspace_ms_cta_flooding<TLLR>> {
public:
    typedef TLLR                                            LLR_t;
    typedef c2v_message_t<TLLR>                             message_t;
    typedef LDPC_workspace<workspace_ms_cta_flooding<TLLR>> inherited_t;
    //------------------------------------------------------------------
    // Constructor
    //__device__
    //workspace_ms_cta_flooding(void* pv, const LDPC_config&) :
    //    inherited_t(pv)
    //{
    //}
    //------------------------------------------------------------------
    // For each codeword:
    //     C2V: (mb * Z) instances of c2v_message_t
    // Note: It is possible that some combinations of (Z, mb) will allow
    // C2V data to be stored in shared memory. For now, this class sizes
    // the workspace assuming that it will instead always be stored in
    // global workspace memory.
    CUDA_BOTH_INLINE
    static size_t workspace_bytes_per_codeword(const LDPC_config& config)
    {
        unsigned int sC2V = (config.mb * config.Z * sizeof(message_t));
        // Make sure that the data for each codeword is aligned to at
        // least 8 bytes
        return sC2V;
    }
    //__device__
    //message_t* C2V(const LDPC_config& config, int codewordIndex)
    //{
    //    return inherited_t::template offset_as<message_t>(codeword_base_offset(config, codewordIndex));
    //}
private:
    //__device__
    //size_t codeword_base_offset(const LDPC_config& config, int codewordIndex)
    //{
    //    return (workspace_bytes_per_codeword(config) * codewordIndex);
    //}
};

////////////////////////////////////////////////////////////////////////
// shared_mem_cta_flooding
// Shared memory representation for configurations that store both APP
// and C2V data in shared memory. In practice, this will probably be
// configurations with small mb (number of parity nodes).
template <typename TLLR>
class shared_mem_cta_flooding : public LDPC_shared_mem<shared_mem_cta_flooding<TLLR>> {
public:
    typedef TLLR                                           LLR_t;
    typedef c2v_message_t<TLLR>                            message_t;
    typedef LDPC_shared_mem<shared_mem_cta_flooding<TLLR>> inherited_t;
    CUDA_BOTH_INLINE
    shared_mem_cta_flooding() {}

    static int get_shared_mem_size(const LDPC_config& config)
    {
        // Shared memory requirements:
        // number                type         description
        // NUM_KERNEL_NODES * Z  TLLR         a priori probability (APP) array, 1 per coded bit
        // mb * Z                c2v_message  C2V messages
        // 1                     int          per-block count of failed parity checks

        return ((config.num_kernel_nodes() * config.Z * sizeof(TLLR)) + // APP
                (config.mb * config.Z * sizeof(message_t)) +            // C2V
                sizeof(int));                                           // check cound
    }
    __device__ int& check_fail_count(const LDPC_config& config)
    {
        return *inherited_t::template offset_as<int>((config.num_kernel_nodes() * config.Z * sizeof(TLLR)) + // APP
                                                     (config.mb * config.Z * sizeof(message_t)));
    }
    __device__
        TLLR*
        app_addr() { return inherited_t::template offset_as<TLLR>(0); }
    __device__
        message_t*
        c2v_addr(const LDPC_config& config)
    {
        return inherited_t::template offset_as<message_t>(config.num_kernel_nodes() * config.Z * sizeof(TLLR));
    }
};

} // namespace

template <cuphyDataType_t TLLREnum>
__device__ void write_hard_decision(LDPC_output_t                                    tOutput,
                                    int                                              codeWordIdx,
                                    LDPC_config                                      config,
                                    const typename data_type_traits<TLLREnum>::type* srcAPP)
{
    typedef typename data_type_traits<TLLREnum>::type LLR_t;
    const int                                         K               = config.Kb * config.Z;
    const int                                         WORDS_PER_CW    = (K + 31) / 32;
    const int                                         BIT_BLOCK_COUNT = (K + 1023) / 1024;
    const int                                         BLOCK_SIZE      = blockDim.x * blockDim.y;
    const int                                         WARPS_PER_BLOCK = BLOCK_SIZE / 32;
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
            uint32_t hard_decision = ((idx < K) && is_neg(srcAPP[idx])) ? 1 : 0;
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
            tOutput({OUT_INDEX, codeWordIdx}) = thread_output;
        }
    }
}

template <cuphyDataType_t TLLREnum, int Z, int NODES_PER_CTA, int MIN_BLOCKS>
__global__
__launch_bounds__(Z* NODES_PER_CTA, MIN_BLOCKS) void ldpc_ms_cta_flooding(LDPC_output_t                        tOutput,
                                                                          LDPC_config                          config,
                                                                          float                                normalization,
                                                                          const_tensor_ref_contig_2D<TLLREnum> tLLR,
                                                                          void*                                workspaceAddress)
{
    typedef typename data_type_traits<TLLREnum>::type LLR_t;
    typedef c2v_message_t<LLR_t>                      message_t;
    //workspace_ms_cta_flooding<LLR_t> workspace(workspaceAddress, config);
    shared_mem_cta_flooding<LLR_t> sharedMem;
    const int                      CODEWORD_INDEX   = blockIdx.x;
    const LLR_t*                   channelLLR       = &tLLR({0, CODEWORD_INDEX});
    c2v_message_t<LLR_t>*          shC2V            = sharedMem.c2v_addr(config);
    LLR_t*                         shAPP            = sharedMem.app_addr();
    int&                           shCheckFailCount = sharedMem.check_fail_count(config);
    const int                      NODE_OFFSET      = threadIdx.x;

    //KERNEL_PRINT_GRID_ONCE("ldpc_ms_cta_flooding\n");
    //------------------------------------------------------------------
    // Load LLR data into shared memory APP
    block_copy_sync_2D(shAPP, channelLLR, Z * config.num_kernel_nodes());
    //block_zero_sync_2D(shAPP, Z * config.num_kernel_nodes());
    //print_array_sync("shAPP (init)", shAPP, Z * config.num_kernel_nodes());
    //------------------------------------------------------------------
    // Iteration Loop
    for(int iIter = 0; iIter < config.max_iterations; ++iIter)
    {
        // Parity Node Loop
        for(int checkIdx = (threadIdx.y * Z) + NODE_OFFSET, nodeIdx = threadIdx.y;
            checkIdx < (Z * config.mb);
            checkIdx += (NODES_PER_CTA * blockDim.x), nodeIdx += NODES_PER_CTA)
        {
            //KERNEL_PRINT_IF(threadIdx.x == 0, "threadIdx = (%u, %u), checkIdx = %i, nodeIdx = %i\n",
            //                threadIdx.x, threadIdx.y, checkIdx, nodeIdx);
            bg1_CN_row_shift_info_t CNShift(nodeIdx, config.Z);
            message_t               c2vOut;
            if(0 == iIter)
            {
                // First iteration: use channel LLR as input
                // Iterate over received LLR data to generate the first APP
                // increment.
                // TODO: FIX THIS - create_message() won't work unless ALL
                // channel LLRs are stored in shared memory!
                c2vOut = message_t::create_message(CNShift,     // shift data
                                                   NODE_OFFSET, // offset of check variable within node
                                                   config.Z,    // lifting factor
                                                   channelLLR);
                //KERNEL_PRINT_IF(checkIdx < 4, "checkIdx = %i, min0 = %f, min1 = %f, index = %i\n",
                //                checkIdx, to_float(c2vOut.min0), to_float(c2vOut.min1), c2vOut.get_row_index());
#if 1
                c2v_message_reader<LLR_t> c2vOutReader(c2vOut, type_convert<LLR_t>(normalization));
#endif
                // Increment APP values based on C2V data
                for(int iVN = 0; iVN < CNShift.row_degree; ++iVN)
                {
                    if(CNShift.column_values[iVN] < config.num_kernel_nodes())
                    {
#if 1
                        LLR_t llrValue = c2vOutReader.get_value_for_index_and_advance(iVN);
#else
                        LLR_t llrValue = c2vOut.get_value_for_index(iVN, normalization);
#endif
                        const int VN_idx = message_t::get_variable_index(CNShift, iVN, NODE_OFFSET, config.Z);
                        atomicAdd(shAPP + VN_idx, llrValue);
                        // WRONG! Used instead of previous line, only to time impact of atomic
                        //shAPP[VN_idx] = llrValue;
                        //KERNEL_PRINT_IF(VN_idx < 1, "Adding %f to APP[%i] (CHECK_IDX = %i, iVN = %i, threadIdx.x = %u, blockIdx.x = %u)\n",
                        //                to_float(llrValue), VN_idx, checkIdx, iVN, threadIdx.x, blockIdx.x);
                    }
                }
            }
            else // iterations 1 through N
            {
                // After the first iteration: read previous C2V message and
                // generate values used as inputs to the next C2V
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   -  -  -
                // Load the C2V message from the previous iteration
                message_t c2vIn;
                load_c2v_message(&c2vIn, shC2V + checkIdx);
                //KERNEL_PRINT("CHECK_IDX = %i, min0 = %.4f, min1 = %.4f, sign_index = 0x%X\n", CHECK_IDX, to_float(c2vIn.min0), to_float(c2vIn.min1), c2vIn.sign_index);
#if 1
                c2v_message_reader<LLR_t> c2vInReader(c2vIn, type_convert<LLR_t>(normalization));
#endif
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
                if(CNShift.column_values[CNShift.row_degree - 1] >= config.num_kernel_nodes())
                {
                    --rowIndexEnd;
                }
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   -  -  -
                for(int iVN = 0; iVN < rowIndexEnd; ++iVN)
                {
                    // Calculate the V2C message from last iteration
#if 1
                    LLR_t prevValue = c2vInReader.get_value_for_index_and_advance(iVN);
#else
                    LLR_t prevValue = c2vIn.get_value_for_index(iVN, normalization);
#endif
                    const int VN_idx = message_t::get_variable_index(CNShift, iVN, NODE_OFFSET, config.Z);
                    LLR_t     v2c    = shAPP[VN_idx] - prevValue;
                    // Use the new V2C value to update the C2V message for the next
                    // iteration.
                    //KERNEL_PRINT_IF(checkIdx < 1, "CHECK_IDX = %i, iVN = %i, APP[%i] = %.4f, prevValue = %.4f, V2C = %.4f\n",
                    //                checkIdx, iVN, VN_idx, to_float(shAPP[VN_idx]), to_float(prevValue), to_float(v2c));
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
#if 1
                c2v_message_reader<LLR_t> c2vOutReader(c2vOut, type_convert<LLR_t>(normalization));
#endif
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   -  -  -
                // Increment APP values based on C2V data
                for(int iVN = 0; iVN < CNShift.row_degree; ++iVN)
                {
                    if(CNShift.column_values[iVN] < config.num_kernel_nodes())
                    {
                        const int VN_idx = message_t::get_variable_index(CNShift, iVN, NODE_OFFSET, config.Z);
#if 1
                        LLR_t llrNew = c2vOutReader.get_value_for_index_and_advance(iVN);
                        LLR_t llrOld = c2vInReader.get_value_for_index_and_advance(iVN);
#else
                        LLR_t llrNew = c2vOut.get_value_for_index(iVN, normalization);
                        LLR_t llrOld = c2vIn.get_value_for_index(iVN, normalization);
#endif
                        atomicAdd(shAPP + VN_idx, llrNew - llrOld);
                        // WRONG! Used instead of previous line, only to time impact of atomic
                        //shAPP[VN_idx] = llrNew - llrOld;
                        //KERNEL_PRINT_IF(32 == VN_idx, "Adding %f to APP[%i] (delta = %f) (CHECK_IDX = %i, iVN = %i, threadIdx.x = %u, blockIdx.x = %u)\n",
                        //                to_float(llrNew), to_float(llrNew - llrOld), VN_idx, CHECK_IDX, iVN, threadIdx.x, blockIdx.x);
                    }
                }
            }
            //--------------------------------------------------------------
            // Write C2V for the next iteration
            //KERNEL_PRINT_IF(checkIdx < 4,
            //                "iteration %i: Writing C2V message with min0 = %f, min1 = %f, signs = 0x%X, row_index = %i, checkIdx = %i\n",
            //                iIter,
            //                to_float(c2vOut.min0),
            //                to_float(c2vOut.min1),
            //                c2vOut.get_signs(),
            //                c2vOut.get_row_index(),
            //                checkIdx);
            shC2V[checkIdx] = c2vOut;
            //__syncthreads();
        } // parity node loop
        __syncthreads();
        //print_array_sync("APP", shAPP, Z * config.num_kernel_nodes());
    } // iteration loop
    write_hard_decision<TLLREnum>(tOutput,
                                  CODEWORD_INDEX,
                                  config,
                                  shAPP);
}

////////////////////////////////////////////////////////////////////////
// launch_decode_ms_cta_flooding()
template <cuphyDataType_t TType>
cuphyStatus_t launch_decode_ms_cta_flooding(LDPC_output_t&      tOutputWord,
                                            const_tensor_pair&  tLLRInput,
                                            const LDPC_config&  config,
                                            float               normalization,
                                            cuphyLDPCResults_t* results,
                                            void*               workspace,
                                            cudaStream_t        strm)
{
    DEBUG_PRINTF("ldpc::launch_decode_ms_cta_flooding()\n");
    typedef const_tensor_ref_contig_2D<TType>      const_tensor2f;
    typedef typename data_type_traits<TType>::type LLR_t;
    typedef workspace_ms_cta_flooding<LLR_t>       workspace_t;
    typedef shared_mem_cta_flooding<LLR_t>         shared_mem_t;
    typedef c2v_message_t<LLR_t>                   message_t;

    // The kernel is only implemented for contiguous, 2D tensors.
    // Attempt to convert to such a tensor descriptor.
    cuphy_optional<const_tensor2f> tOptLLR = tLLRInput.first.get().get_ref_contig_rank<TType, 2>(tLLRInput.second);
    if(!tOptLLR)
    {
        // Layout is not 2D contiguous
        return CUPHY_STATUS_UNSUPPORTED_LAYOUT;
    }
    if((384 != config.Z) || (4 != config.mb))
    {
        // TODO: Remove this
        return CUPHY_STATUS_UNSUPPORTED_LAYOUT;
    }

    dim3      blkDim(config.Z, 2);
    dim3      grdDim(config.num_codewords);
    const int SHMEM_SIZE = shared_mem_t::get_shared_mem_size(config);
    //using ldpc_ms_cta_flooding_fcn = ldpc_ms_cta_flooding<TType, 384, 2, 1/*2*/>;
    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc_ms_cta_flooding<TType, 384, 2, 1/*2*/>));
    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc_ms_cta_flooding<TType, 384, 2, 1/*2*/>), blkDim, SHMEM_SIZE);
    DEBUG_PRINTF("grid = (%u, %u, %u), block = (%u, %u, %u), shmem = %i\n",
                 grdDim.x,
                 grdDim.y,
                 grdDim.z,
                 blkDim.x,
                 blkDim.y,
                 blkDim.z,
                 SHMEM_SIZE);
    ldpc_ms_cta_flooding<TType, 384, 2, 1/*2*/><<<grdDim, blkDim, SHMEM_SIZE, strm>>>(tOutputWord,
                                                                                      config,
                                                                                      normalization,
                                                                                      tOptLLR.value(),
                                                                                      workspace);
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
// decode_ms_cta_flooding()
cuphyStatus_t decode_ms_cta_flooding(LDPC_output_t&      tDst,
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
        return launch_decode_ms_cta_flooding<CUPHY_R_32F>(tDst,
                                                          tLLR,
                                                          config,
                                                          normalization,
                                                          results,
                                                          workspace,
                                                          strm);
    case CUPHY_R_16F:
        return launch_decode_ms_cta_flooding<CUPHY_R_16F>(tDst,
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

////////////////////////////////////////////////////////////////////////
// decode_ms_cta_flooding_workspace_size()
std::pair<bool, size_t> decode_ms_cta_flooding_workspace_size(const LDPC_config& cfg)
{
    switch(cfg.type)
    {
    case CUPHY_R_32F:
    {
        typedef data_type_traits<CUPHY_R_32F>::type LLR_t;
        typedef workspace_ms_cta_flooding<LLR_t>    workspace_t;
        return std::pair<bool, size_t>(true,
                                       workspace_t::get_workspace_size(cfg));
    }
    case CUPHY_R_16F:
    {
        typedef data_type_traits<CUPHY_R_16F>::type LLR_t;
        typedef workspace_ms_cta_flooding<LLR_t>    workspace_t;
        return std::pair<bool, size_t>(true,
                                       workspace_t::get_workspace_size(cfg));
    }
    default:
        return std::pair<bool, size_t>(false, 0);
    }
}

} // namespace ldpc
