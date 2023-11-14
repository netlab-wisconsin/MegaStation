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

#include "ldpc_ms_cta_simd_flooding.hpp"
#include "ldpc.cuh"
#include "ldpc_c2v_message.cuh"

namespace
{
////////////////////////////////////////////////////////////////////////
// workspace_ms_cta_simd_flooding
// Class to represent the workspace required by this LDPC implementation.
// API callers will query the workspace size offline and allocate a
// buffer to provide to the implementation to avoid allocating memory
// in each call.
template <typename TLLR>
class workspace_ms_cta_simd_flooding : public LDPC_workspace<workspace_ms_cta_simd_flooding<TLLR>> {
public:
    typedef TLLR                                                 LLR_t;
    typedef c2v_message_t<TLLR>                                  message_t;
    typedef LDPC_workspace<workspace_ms_cta_simd_flooding<TLLR>> inherited_t;
    //------------------------------------------------------------------
    // Constructor
    //__device__
    //workspace_ms_cta_simd_flooding(void* pv, const LDPC_config&) :
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
// shared_mem_cta_simd_flooding
// Shared memory representation for configurations that store all APP
// data in shared memory.
template <typename TLLR>
class shared_mem_cta_simd_flooding : public LDPC_shared_mem<shared_mem_cta_simd_flooding<TLLR>> {
public:
    typedef TLLR                                                LLR_t;
    typedef c2v_message_t<TLLR>                                 message_t;
    typedef LDPC_shared_mem<shared_mem_cta_simd_flooding<TLLR>> inherited_t;
    CUDA_BOTH_INLINE
    shared_mem_cta_simd_flooding() {}

    static int get_shared_mem_size(const LDPC_config& config)
    {
        // Shared memory requirements:
        // number                type         description
        // NUM_KERNEL_NODES * Z  TLLR         a priori probability (APP) array, 1 per coded bit
        // mb * Z                c2v_message  C2V messages
        // 1                     int          per-block count of failed parity checks

        return ((config.Kb + config.mb) * config.Z * sizeof(TLLR)) + // APP
               sizeof(int);                                          // check count
    }
    __device__ int& check_fail_count(const LDPC_config& config)
    {
        return *inherited_t::template offset_as<int>((config.Kb + config.mb) * config.Z * sizeof(TLLR)); // APP
    }
    __device__
        TLLR*
        app_addr() { return inherited_t::template offset_as<TLLR>(0); }
};

} // namespace

template <typename LLR_t, int NCHECK>
struct c2v_message_vec
{
    typedef c2v_message_t<LLR_t> message_t;
    //------------------------------------------------------------------
    // Data
    message_t messages[NCHECK];
    //------------------------------------------------------------------
    // initialize()
    __device__ void initialize()
    {
#pragma unroll
        for(int i = 0; i < NCHECK; ++i)
        {
            messages[i].init();
        }
    }
    //------------------------------------------------------------------
    // initialize()
    __device__ void initialize(const bg1_CN_row_shift_info_t& shiftInfo,       // shift data
                               int                            nodeOffsetStart, // offset of check variable within node
                               int                            Z,               // lifting factor
                               const LLR_t*                   initLLR)                           // channel LLRs
    {
#pragma unroll
        for(int i = 0; i < NCHECK; ++i)
        {
            messages[i] = message_t::create_message(shiftInfo,
                                                    nodeOffsetStart + i,
                                                    Z,
                                                    initLLR);
        }
    }
    //------------------------------------------------------------------
    // update_app()
    __device__ void update_app(const bg1_CN_row_shift_info_t& shiftInfo,       // shift data
                               int                            nodeOffsetStart, // offset of check variable within node
                               LLR_t*                         shAPP,           // channel LLRs
                               LLR_t                          normalization,   // min-sum normalization
                               const LDPC_config&             config)                      // LDPC config
    {
        for(int iVN = 0; iVN < shiftInfo.row_degree; ++iVN)
        {
            // We don't update APP nodes for extension parity nodes.
            // Since there is only 1 check node attached, and the
            // update value will be subtracted, the APP value will
            // remain equal to the channel LLR.
            if(shiftInfo.column_values[iVN] < config.num_kernel_nodes())
            {
#pragma unroll
                for(int i = 0; i < NCHECK; ++i)
                {
                    LLR_t     llrValue = messages[i].get_value_for_index(iVN, normalization);
                    const int VN_idx   = message_t::get_variable_index(shiftInfo, iVN, nodeOffsetStart + i, config.Z);
                    atomicAdd(shAPP + VN_idx, llrValue);
                    //KERNEL_PRINT_IF(VN_idx < 1, "Adding %f to APP[%i] (iVN = %i, threadIdx.x = %u, blockIdx.x = %u)\n",
                    //                to_float(llrValue), VN_idx, iVN, threadIdx.x, blockIdx.x);
                }
            }
        }
    }
    __device__ void update_c2v(const bg1_CN_row_shift_info_t& shiftInfo,       // shift data
                               int                            nodeOffsetStart, // offset of check variable within node
                               const LLR_t*                   shAPP,           // channel LLRs
                               LLR_t                          normalization,   // min-sum normalization
                               const LDPC_config&             config,          // LDPC config
                               const c2v_message_vec&         src,
                               int                            nodeIndex)
    {
        for(int iVN = 0; iVN < shiftInfo.row_degree; ++iVN)
        {
#pragma unroll
            for(int i = 0; i < NCHECK; ++i)
            {
                // Calculate the V2C message from last iteration
                LLR_t     prevValue = src.messages[i].get_value_for_index(iVN, normalization);
                const int VN_idx    = message_t::get_variable_index(shiftInfo, iVN, nodeOffsetStart + i, config.Z);
                LLR_t     v2c       = shAPP[VN_idx] - prevValue;
                // Use the new V2C value to update the C2V message for the next
                // iteration.
                //KERNEL_PRINT_IF((0 == nodeOffsetStart) && (0 == nodeIndex) && (0 == i), "checkIdx = %i, iVN = %i, APP[%i] = %.4f, prevValue = %.4f, V2C = %.4f\n",
                //                (nodeIndex * config.Z) + nodeOffsetStart + i, iVN, VN_idx, to_float(shAPP[VN_idx]), to_float(prevValue), to_float(v2c));
                messages[i].process(v2c, iVN);
            }
        } // iVN
    }
};

////////////////////////////////////////////////////////////////////////
// NUM_PARITY_NODES = blockDim.y
// Threads within a block that have the same threadIdx.y map to the
// same parity node.
// Expected block size: Dim3(Z / CHECKS_PER_THREAD, mb)
// Example: For Z = 384, mb = 4, CHECKS_PER_THREAD = 2:
//          blockDim = (384 / 2, 4) = (192, 4)
//          (768 threads processing 1536 parity checks)
// Example: For Z = 384, mb = 4, CHECKS_PER_THREAD = 4:
//          blockDim = (384 / 4, 4) = (96, 4)
//          (384 threads processing 1536 parity checks)
template <cuphyDataType_t TLLREnum, int NTHREADS, int CHECKS_PER_THREAD, int MIN_BLOCKS>
__global__
    __launch_bounds__(NTHREADS, MIN_BLOCKS) void ldpc_ms_cta_simd_flooding(LDPC_output_t                        tOutput,
                                                                           LDPC_config                          config,
                                                                           float                                normalization,
                                                                           const_tensor_ref_contig_2D<TLLREnum> tLLR,
                                                                           void*                                workspaceAddress)
{
    typedef typename data_type_traits<TLLREnum>::type LLR_t;
    typedef c2v_message_vec<LLR_t, CHECKS_PER_THREAD> c2v_message_vec_t;
    //workspace_ms_cta_flooding<LLR_t>    workspace(workspaceAddress, config);
    shared_mem_cta_simd_flooding<LLR_t> sharedMem;
    const int                           CODEWORD_INDEX = blockIdx.x;
    const LLR_t*                        channelLLR     = &tLLR({0, CODEWORD_INDEX});
    LLR_t*                              shAPP          = sharedMem.app_addr();
    //KERNEL_PRINT_GRID_ONCE("ldpc_ms_cta_simd_flooding\n");
    //------------------------------------------------------------------
    // Load LLR data into shared memory APP
    block_copy_sync_2D(shAPP, channelLLR, config.Z * (config.mb + config.Kb));
    //print_array_sync("shAPP (init)", shAPP, config.Z * config.num_kernel_nodes());
    //------------------------------------------------------------------
    const int               NODE_INDEX        = threadIdx.y;
    const int               NODE_OFFSET_START = threadIdx.x * CHECKS_PER_THREAD;
    bg1_CN_row_shift_info_t CNShift(NODE_INDEX, config.Z);

    c2v_message_vec_t messageVector0, messageVector1;

    for(int iIter = 0; iIter < config.max_iterations; ++iIter)
    {
        //KERNEL_PRINT_IF(threadIdx.x < 4, "threadIdx = (%u, %u), NODE_INDEX = %i, NODE_OFFSET_START = %i\n",
        //                threadIdx.x, threadIdx.y, NODE_INDEX, NODE_OFFSET_START);
        if(0 == iIter)
        {
            // Initialize message vector with values from channel LLR
            messageVector0.initialize(CNShift, NODE_OFFSET_START, config.Z, channelLLR);
            KERNEL_PRINT_IF((threadIdx.x < 2) && (0 == threadIdx.y),
                            "checkIdx = %i, min0 = %f, min1 = %f, index = %i\ncheckIdx = %i, min0 = %f, min1 = %f, index = %i\n",
                            NODE_OFFSET_START,
                            to_float(messageVector0.messages[0].min0),
                            to_float(messageVector0.messages[0].min1),
                            messageVector0.messages[0].get_row_index(),
                            NODE_OFFSET_START + 1,
                            to_float(messageVector0.messages[1].min0),
                            to_float(messageVector0.messages[1].min1),
                            messageVector0.messages[1].get_row_index());
        }
        else
        {
            // Calculate a new set of messages based on the previous message
            messageVector1.initialize();
            messageVector1.update_c2v(CNShift,           // shift data
                                      NODE_OFFSET_START, // offset of check variable within node
                                      shAPP,             // channel LLRs
                                      normalization,     // min-sum normalization
                                      config,            // LDPC config
                                      messageVector0,    // previous message
                                      NODE_INDEX);       // node index (debugging only)

            // Copy to message 0 for APP update
            messageVector0 = messageVector1;
        }
        // Allow all threads to read from shAPP before updating it below
        __syncthreads();

        // Increment APP values based on C2V data
        messageVector0.update_app(CNShift, NODE_OFFSET_START, shAPP, normalization, config);

        //print_array_sync("APP", shAPP, config.Z * config.num_kernel_nodes());
        // All threads in the block should arrive here before starting
        // the next iteration.

        //KERNEL_PRINT_IF((threadIdx.x == 0) && (0 == threadIdx.y),
        //                "iteration %i: Writing C2V message with min0 = %f, min1 = %f, signs = 0x%X, row_index = %i, checkIdx = %i\n"
        //                "iteration %i: Writing C2V message with min0 = %f, min1 = %f, signs = 0x%X, row_index = %i, checkIdx = %i\n",
        //                iIter,
        //                to_float(messageVector0.messages[0].min0),
        //                to_float(messageVector0.messages[0].min1),
        //                messageVector0.messages[0].get_signs(),
        //                messageVector0.messages[0].get_row_index(),
        //                (NODE_INDEX * config.Z) + NODE_OFFSET_START + 0,
        //                iIter,
        //                to_float(messageVector0.messages[1].min0),
        //                to_float(messageVector0.messages[1].min1),
        //                messageVector0.messages[1].get_signs(),
        //                messageVector0.messages[1].get_row_index(),
        //                (NODE_INDEX * config.Z) + NODE_OFFSET_START + 1);
    }
    cta_write_hard_decision<TLLREnum>(tOutput,
                                      CODEWORD_INDEX,
                                      (config.Kb * config.Z),
                                      shAPP);
}

////////////////////////////////////////////////////////////////////////
// launch_decode_ms_cta_simd_flooding()
template <cuphyDataType_t TType>
cuphyStatus_t launch_decode_ms_cta_simd_flooding(LDPC_output_t&      tOutputWord,
                                                 const_tensor_pair&  tLLRInput,
                                                 const LDPC_config&  config,
                                                 float               normalization,
                                                 cuphyLDPCResults_t* results,
                                                 void*               workspace,
                                                 cudaStream_t        strm)
{
    DEBUG_PRINTF("ldpc::launch_decode_ms_cta_simd_flooding()\n");
    typedef const_tensor_ref_contig_2D<TType>      const_tensor2f;
    typedef typename data_type_traits<TType>::type LLR_t;
    typedef workspace_ms_cta_simd_flooding<LLR_t>  workspace_t;
    typedef shared_mem_cta_simd_flooding<LLR_t>    shared_mem_t;
    typedef c2v_message_t<LLR_t>                   message_t;

    // The kernel is only implemented for contiguous, 2D tensors.
    // Attempt to convert to such a tensor descriptor.
    cuphy_optional<const_tensor2f> tOptLLR = tLLRInput.first.get().get_ref_contig_rank<TType, 2>(tLLRInput.second);
    if(!tOptLLR)
    {
        // Layout is not 2D contiguous
        return CUPHY_STATUS_UNSUPPORTED_LAYOUT;
    }
    //------------------------------------------------------------------
    if((384 != config.Z) || (4 != config.mb))
    {
        // TODO: Remove this
        return CUPHY_STATUS_UNSUPPORTED_LAYOUT;
    }

    const int CHECKS_PER_THREAD = 2;
    dim3      blkDim(config.Z / CHECKS_PER_THREAD, config.mb);
    dim3      grdDim(config.num_codewords);
    const int SHMEM_SIZE = shared_mem_t::get_shared_mem_size(config);
    const int MIN_BLOCKS = 1;
    //using ldpc_ms_cta_flooding_fcn = ldpc_ms_cta_flooding<TType, 384 * 2, 2, 1>;
    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc_ms_cta_simd_flooding<TType, 384 * 2, 2, MIN_BLOCKS>));
    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc_ms_cta_simd_flooding<TType, 384 * 2, 2, MIN_BLOCKS>), blkDim, SHMEM_SIZE);
    DEBUG_PRINTF("grid = (%u, %u, %u), block = (%u, %u, %u), shmem = %i\n",
                 grdDim.x,
                 grdDim.y,
                 grdDim.z,
                 blkDim.x,
                 blkDim.y,
                 blkDim.z,
                 SHMEM_SIZE);
    //ldpc_ms_cta_simd_flooding<TType, 384, 2, 2><<<grdDim, blkDim, SHMEM_SIZE>>>(tOutputWord,
    ldpc_ms_cta_simd_flooding<TType, 384 * 2, 2, MIN_BLOCKS><<<grdDim, blkDim, SHMEM_SIZE>>>(tOutputWord,
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
// decode_ms_cta_simd_flooding()
cuphyStatus_t decode_ms_cta_simd_flooding(LDPC_output_t&      tDst,
                                          const_tensor_pair&  tLLR,
                                          const LDPC_config&  config,
                                          float               normalization,
                                          cuphyLDPCResults_t* results,
                                          void*               workspace,
                                          cudaStream_t        strm)
{
    DEBUG_PRINTF("ldpc::decode_ms_cta_simd_flooding()\n");
    switch(tLLR.first.get().type())
    {
    case CUPHY_R_16F:
        return launch_decode_ms_cta_simd_flooding<CUPHY_R_16F>(tDst,
                                                               tLLR,
                                                               config,
                                                               normalization,
                                                               results,
                                                               workspace,
                                                               strm);
    default:
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
}

////////////////////////////////////////////////////////////////////////
// decode_ms_cta_simd_flooding_workspace_size()
std::pair<bool, size_t> decode_ms_cta_simd_flooding_workspace_size(const LDPC_config& cfg)
{
    switch(cfg.type)
    {
    case CUPHY_R_16F:
    {
        typedef data_type_traits<CUPHY_R_16F>::type   LLR_t;
        typedef workspace_ms_cta_simd_flooding<LLR_t> workspace_t;
        return std::pair<bool, size_t>(true,
                                       workspace_t::get_workspace_size(cfg));
    }
    default:
        return std::pair<bool, size_t>(false, 0);
    }
}

} // namespace ldpc
