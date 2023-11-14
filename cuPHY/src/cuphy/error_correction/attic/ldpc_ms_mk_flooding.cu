/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


//#define CUPHY_DEBUG 1

#include "ldpc_ms_mk_flooding.hpp"
#include "ldpc.cuh"
#include "ldpc_c2v_message.cuh"

////////////////////////////////////////////////////////////////////////
// workspace_multi_kernel_t
template <typename TLLR>
struct workspace_multi_kernel_t
{
    typedef TLLR                LLR_t;
    typedef c2v_message_t<TLLR> message_t;
    //------------------------------------------------------------------
    // Constructor
    __device__
    workspace_multi_kernel_t(void* pv, int mb, int Z, int Kb) :
        address_(pv),
        mb_(mb),
        Z_(Z),
        Kb_(Kb)
    {
    }
    //------------------------------------------------------------------
    // get_size()
    // Workspace:
    // For each codeword:
    //     C2V: (mb * Z) instances of c2v_message_t
    //     APP: ((Kb + 4) * Z) * sizeof(TLLR)
    static size_t get_size(int mb, int Kb, int Z, int numCodewords)
    {
        return (bytes_per_codeword(mb, Kb, Z) * numCodewords);
    }
    __device__
        message_t*
        C2V(int codewordIndex) { return offset_as<message_t>(codeword_base_offset(codewordIndex)); }
    __device__
        LLR_t*
        APP(int codewordIndex) { return offset_as<LLR_t>(codeword_base_offset(codewordIndex) + app_offset()); }

private:
    template <typename T>
    __device__
        T*
        offset_as(int offset_bytes)
    {
        return reinterpret_cast<T*>(static_cast<char*>(address_) + offset_bytes);
    }
    __device__
        size_t
        codeword_base_offset(int codewordIndex)
    {
        return (bytes_per_codeword(mb_, Kb_, Z_) * codewordIndex);
    }
    __device__
        size_t
        app_offset() const
    {
        // APP buffer is after the C2V buffer in each codeword segment
        return (mb_ * Z_ * sizeof(message_t));
    }
    CUDA_BOTH_INLINE
    static size_t bytes_per_codeword(int mb, int Kb, int Z)
    {
        unsigned int sC2V  = (mb * Z * sizeof(message_t));
        unsigned int szAPP = ((Kb + 4) * Z) * sizeof(TLLR);
        // Make sure that the data for each codeword is aligned to at
        // least 8 bytes
        return ((szAPP + sC2V + 7) / 8) * 8;
    }
    void* address_;
    int   mb_;
    int   Z_;
    int   Kb_;
};

template <cuphyDataType_t TLLREnum>
__global__ void ldpc_flooding_multi_kernel_cnp_first(int                                  iLS,
                                                     int                                  Z,
                                                     int                                  mb,
                                                     int                                  Kb,
                                                     const_tensor_ref_contig_2D<TLLREnum> tLLR,
                                                     void*                                workspaceAddress,
                                                     int                                  iteration)
{
    typedef typename data_type_traits<TLLREnum>::type LLR_t;
    typedef c2v_message_t<LLR_t>                      message_t;
    const int                                         CODEWORD_INDEX   = blockIdx.y;
    const int                                         CHECK_IDX        = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int                                         NODE_IDX         = CHECK_IDX / Z;
    const int                                         NODE_OFFSET      = CHECK_IDX % Z;
    const int                                         CHECK_COUNT      = mb * Z;
    const bool                                        THREAD_HAS_CHECK = (CHECK_IDX < CHECK_COUNT);
    const LLR_t*                                      channelLLR       = &tLLR({0, CODEWORD_INDEX});

    workspace_multi_kernel_t<LLR_t> workspace(workspaceAddress, mb, Z, Kb);
    c2v_message_t<LLR_t>*           C2V = workspace.C2V(CODEWORD_INDEX);
    //KERNEL_PRINT_GRID_ONCE("ldpc_flooding_multi_kernel_cnp_first, CHECK_IDX = %i, CHECK_COUNT = %i\n", CHECK_IDX, CHECK_COUNT);
    //c2vMessage.init();
    if(THREAD_HAS_CHECK)
    {
        bg1_CN_row_shift_info_t CNShift(NODE_IDX, Z);
#if 1
        message_t c2vMessage = message_t::create_message(CNShift,     // shift data
                                                         NODE_OFFSET, // offset of check variable within node
                                                         Z,           // lifting factor
                                                         channelLLR);
#else
        message_t c2vMessage;
        c2vMessage.init();
        LLR_t    min0     = 10000; // TODO
        LLR_t    min1     = 10000; // TODO
        uint32_t signBits = 0;
        int      minIndex = -1;
        for(int iVN = 0; iVN < CNShift.row_degree; ++iVN)
        {
            const int8_t POS          = CNShift.column_values[iVN];
            int          block_offset = NODE_OFFSET + CNShift.shift_values[iVN];
            if(block_offset >= Z) block_offset -= Z;
            const int VN_idx = (POS * Z) + block_offset;
            LLR_t     Lvc    = channelLLR[VN_idx];
            signBits |= (is_neg(Lvc) ? 1 : 0) << iVN;
            LLR_t Lvcabs = llr_abs(Lvc);
            if(Lvcabs < min0)
            {
                minIndex = iVN;
                min1     = min0;
                min0     = Lvcabs;
            }
            else if(Lvcabs < min1)
            {
                min1 = Lvcabs;
            }
            //KERNEL_PRINT_IF(CHECK_IDX == 6, "CHECK_IDX = %i, iVN = %i, minIndex = %i, minPos = %i, min0 = %.4f, min1 = %.4f, Lvc = %.4f\n",
            //                CHECK_IDX, iVN, minIndex, minPos, to_float(min0), to_float(min1), to_float(Lvc));
        }
        c2vMessage.set_signs(signBits);
        c2vMessage.set_row_index(minIndex);
        c2vMessage.min0 = min0;
        c2vMessage.min1 = min1;
#endif
        //KERNEL_PRINT_IF(CHECK_IDX < 128, "Writing C2V message with min0 = %f, min1 = %f, signs = 0x%X, row_index = %i, CHECK_IDX = %i\n",
        //                                 to_float(c2vMessage.min0),
        //                                 to_float(c2vMessage.min1),
        //                                 c2vMessage.get_signs(),
        //                                 c2vMessage.get_row_index(),
        //                                 CHECK_IDX);
        C2V[CHECK_IDX] = c2vMessage;
    }
}

template <cuphyDataType_t TLLREnum>
__global__ void ldpc_flooding_multi_kernel_cnp(int                                  iLS,
                                               int                                  Z,
                                               int                                  mb,
                                               int                                  Kb,
                                               float                                normalization,
                                               const_tensor_ref_contig_2D<TLLREnum> tLLR,
                                               void*                                workspaceAddress,
                                               int                                  iteration)
{
    typedef typename data_type_traits<TLLREnum>::type LLR_t;
    typedef c2v_message_t<LLR_t>                      message_t;
    const int                                         CODEWORD_INDEX       = blockIdx.y;
    const int                                         CHECK_IDX            = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int                                         NODE_IDX             = CHECK_IDX / Z;
    const int                                         NODE_OFFSET          = CHECK_IDX % Z;
    const int                                         CHECK_COUNT          = mb * Z;
    const bool                                        THREAD_HAS_CHECK     = (CHECK_IDX < CHECK_COUNT);
    const LLR_t*                                      channelLLR           = &tLLR({0, CODEWORD_INDEX});
    const int                                         NUM_APP_NODES_STORED = Kb + 4;
    workspace_multi_kernel_t<LLR_t>                   workspace(workspaceAddress, mb, Z, Kb);
    c2v_message_t<LLR_t>*                             C2V  = workspace.C2V(CODEWORD_INDEX);
    LLR_t*                                            dAPP = workspace.APP(CODEWORD_INDEX);
    //KERNEL_PRINT_GRID_ONCE("ldpc_flooding_multi_kernel_cnp, CHECK_IDX = %i, CHECK_COUNT = %i\n", CHECK_IDX, CHECK_COUNT);
    if(THREAD_HAS_CHECK)
    {
        bg1_CN_row_shift_info_t CNShift(NODE_IDX, Z);
        message_t               c2v0;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Load the C2V message from the previous iteration
        load_c2v_message(&c2v0, C2V + CHECK_IDX);
        //KERNEL_PRINT("CHECK_IDX = %i, min0 = %.4f, min1 = %.4f, sign_index = 0x%X\n", CHECK_IDX, to_float(c2v0.min0), to_float(c2v0.min1), c2v0.sign_index);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Initialize a new message for the next iteration. We will
        // write this out before the kernel ends.
        message_t c2v1;
        c2v1.init();
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
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
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        for(int iVN = 0; iVN < rowIndexEnd; ++iVN)
        {
            // Calculate the V2C message from last iteration
            LLR_t     prevValue = c2v0.get_value_for_index(iVN, normalization);
            const int VN_idx    = message_t::get_variable_index(CNShift, iVN, NODE_OFFSET, Z);
            LLR_t     v2c       = dAPP[VN_idx] - prevValue;
            // Use the new V2C value to update the C2V message for the next
            // iteration.
            //KERNEL_PRINT_IF(0 == CHECK_IDX, "CHECK_IDX = %i, iVN = %i, APP[%i] = %.4f, prevValue = %.4f, V2C = %.4f\n", CHECK_IDX, iVN, VN_idx, to_float(dAPP[VN_idx]), to_float(prevValue), to_float(v2c));
            c2v1.process(v2c, iVN);
        } // iVN
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // If this check node has an extension parity node, process now
        if(rowIndexEnd != CNShift.row_degree)
        {
            const int VN_idx = message_t::get_variable_index(CNShift, rowIndexEnd, NODE_OFFSET, Z);
            LLR_t     v2c    = channelLLR[VN_idx];
            // Update the C2V message for the next iteration.
            c2v1.process(v2c, rowIndexEnd);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Write the value to the output buffer
        C2V[CHECK_IDX] = c2v1;
    }
}

template <cuphyDataType_t TLLREnum>
__global__ void ldpc_flooding_multi_kernel_app(int                                  iLS,
                                               int                                  Z,
                                               int                                  Kb,
                                               int                                  mb,
                                               float                                normalization,
                                               const_tensor_ref_contig_2D<TLLREnum> tLLR,
                                               LDPC_output_t                        tOutput,
                                               void*                                workspaceAddress,
                                               bool                                 lastIteration,
                                               int                                  iteration)
{
    //KERNEL_PRINT_GRID_ONCE("ldpc_flooding_multi_kernel_app, CHECK_IDX = %i, CHECK_COUNT = %i\n", CHECK_IDX, CHECK_COUNT);
    typedef typename data_type_traits<TLLREnum>::type LLR_t;
    typedef c2v_message_t<LLR_t>                      message_t;
    const int                                         CODEWORD_IDX = blockIdx.y;
    const int                                         VAR_IDX      = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int                                         NODE_IDX     = VAR_IDX / Z;
    const int                                         VAR_OFFSET   = VAR_IDX % Z;
    //const int                       VAR_COUNT    = (Kb + 4) * Z; // TODO Fix for BG2
    //const bool                      VAR_IS_VALID = (VAR_IDX < VAR_COUNT);
    const LLR_t*                    channelLLR = &tLLR({0, CODEWORD_IDX});
    workspace_multi_kernel_t<LLR_t> workspace(workspaceAddress, mb, Z, Kb);
    c2v_message_t<LLR_t>*           C2V  = workspace.C2V(CODEWORD_IDX);
    LLR_t*                          dAPP = workspace.APP(CODEWORD_IDX);

    //kernel_shared_mem_multi_kernel_app_t<LLR_t> shmem;
    //message_t*                      shC2V        = shmem.c2v_addr();
    //------------------------------------------------------------------
    // Initialize the channel APP with the channel LLR
    LLR_t APP = channelLLR[VAR_IDX];
    //KERNEL_PRINT("iter = %i, start APP[%i] = %f\n", iteration, VAR_IDX, to_float(APP));
    //------------------------------------------------------------------
    // Iterate over parity nodes
    bg1_VN_col_shift_index_info_t VNShift(NODE_IDX, Z);
    message_t                     msg;
    for(int iCN = 0; iCN < VNShift.col_degree; ++iCN)
    {
        const csc_row_idx_t ROW_IDX = VNShift.row_idx_values[iCN];
        const int8_t        ROW     = ROW_IDX.x;
        if(ROW >= mb)
        {
            break;
        }
        const int8_t  IDX   = ROW_IDX.y;
        const int16_t SHIFT = VNShift.shift_values[iCN];
        // Determine the row that corresponds to this thread's variable
        int blockRowIndex = VAR_OFFSET - SHIFT;
        if(blockRowIndex < 0) blockRowIndex += Z;
        load_c2v_message(&msg, &C2V[(ROW * Z) + blockRowIndex]);
        //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        // Add to the APP value
        APP += msg.get_value_for_index(IDX, normalization);
        //KERNEL_PRINT_IF(VAR_IDX < 32,
        //                "Adding %.4f from block row %i (original row %i) to APP for variable %i, new value: %.4f, iCN = %i, SHIFT = %i, NODE_IDX = %i, ROW = %i, signBits = 0x%X\n",
        //                to_float(msg.get_value_for_index(IDX, normalization)),
        //                blockRowIndex,
        //                (blockRowIndex + (ROW * Z)),
        //                VAR_IDX,
        //                to_float(APP),
        //                iCN,
        //                SHIFT,
        //                NODE_IDX,
        //                ROW,
        //                msg.get_signs());
    }
    //KERNEL_PRINT_IF(VAR_IDX < 32, "iter = %i, end APP[%i] = %.4f\n", iteration, VAR_IDX, to_float(APP));
    if(!lastIteration)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Write APP values to the workspace
        dAPP[VAR_IDX] = APP;
    }
    else
    {
        const int WARP_IDX          = threadIdx.x / 32;
        const int WARP_OUTPUT_WORDS = blockDim.x / 32;
        const int OUT_INDEX         = (blockIdx.x * WARP_OUTPUT_WORDS) + WARP_IDX;
        const int NUM_OUTPUT_WORDS  = (Kb * Z) / 32;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Write output bits. For now, assume a thread block size that is
        // a multiple of 32, so that writes don't need to cross word
        // boundaries.
        // Assuming launch configuration with full warps, even if the number
        // of check nodes is not a multiple of 32. This won't always be
        // true. TODO!
        if(OUT_INDEX < NUM_OUTPUT_WORDS)
        {
            // Lane 0 of each warp will collect a hard decision bit from each
            // thread in the warp to form a 32-bit word
            uint32_t  hard_decision = is_neg(APP) ? 1 : 0;
            uint32_t  warp_bits     = __ballot_sync(0xFFFFFFFF, hard_decision);
            const int LANE_IDX      = threadIdx.x % 32;
            if(0 == LANE_IDX)
            {
                //KERNEL_PRINT_IF(OUT_INDEX == 1, "LDPC_out[%3i] = 0x%08X (block = %2u, thread  = %3u)\n", OUT_INDEX, warp_bits, blockIdx.x, threadIdx.x);
                tOutput({OUT_INDEX, CODEWORD_IDX}) = warp_bits;
            }
        }
    }
}

namespace ldpc
{

////////////////////////////////////////////////////////////////////////
// launch_decode_multi_kernel()
template <cuphyDataType_t TType>
cuphyStatus_t launch_decode_multi_kernel(LDPC_output_t&      tOutputWord,
                                         const_tensor_pair&  tLLRInput,
                                         const LDPC_config&  config,
                                         float               normalization,
                                         cuphyLDPCResults_t* results,
                                         void*               workspace,
                                         cudaStream_t        strm)
{
    DEBUG_PRINTF("ldpc::launch_decode_multi_kernel()\n");
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
    int                iLS = set_from_Z(config.Z);
    dim3               blockDim(config.Z);
    const unsigned int NCW = tLLRInput.first.get().layout().dimensions[1];
    dim3               gridDimCNP(config.mb, NCW);
    dim3               gridDimAPP(config.Kb + 4, NCW); // TODO: Fix for BG2
    const int          SHMEM_SIZE = 0;
    //------------------------------------------------------------------
    DEBUG_PRINTF("NCW = %i, BG = %i, N = %i, K = %i, Kb = %i, mb = %i, Z = %i, M = %i, iLS = %i, R_trans = %.2f\n",
                 NCW,
                 config.BG,
                 (config.Kb + config.mb) * config.Z,
                 config.Kb * config.Z,
                 config.Kb,
                 config.mb,
                 config.Z,
                 config.mb * config.Z,
                 iLS,
                 static_cast<float>(config.Kb) / (config.Kb + config.mb - 2));
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc_flooding_multi_kernel_cnp_first<TType>);
    DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc_flooding_multi_kernel_cnp_first<TType>, blockDim, SHMEM_SIZE);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc_flooding_multi_kernel_cnp<TType>);
    DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc_flooding_multi_kernel_cnp<TType>, blockDim, 0);
    DEBUG_PRINT_FUNC_ATTRIBUTES(ldpc_flooding_multi_kernel_app<TType>);
    DEBUG_PRINT_FUNC_MAX_BLOCKS(ldpc_flooding_multi_kernel_app<TType>, blockDim, SHMEM_SIZE);
    //------------------------------------------------------------------
    for(int iter = 0; iter < config.max_iterations; ++iter)
    {
        if(0 == iter)
        {
            // CNP (first time)
            {
                ldpc_flooding_multi_kernel_cnp_first<TType><<<gridDimCNP, blockDim, 0, strm>>>(iLS,             // set index
                                                                                               config.Z,        // lifting size
                                                                                               config.mb,       // num parity nodes
                                                                                               config.Kb,       // num info nodes
                                                                                               tOptLLR.value(), // LLR tensor
                                                                                               workspace,       // workspace
                                                                                               iter);
            }
        }
        else
        {
            // CNP (after the first iteration)
            ldpc_flooding_multi_kernel_cnp<TType><<<gridDimCNP, blockDim, SHMEM_SIZE, strm>>>(iLS,             // set index
                                                                                              config.Z,        // lifting size
                                                                                              config.mb,       // num parity nodes
                                                                                              config.Kb,       // num info nodes
                                                                                              normalization,   // min-sum normalization
                                                                                              tOptLLR.value(), // LLR tensor
                                                                                              workspace,       // workspace (C2V)
                                                                                              iter);
        }
        // APP
        const bool isLast = ((iter + 1) == config.max_iterations);
        ldpc_flooding_multi_kernel_app<TType><<<gridDimAPP, blockDim, SHMEM_SIZE, strm>>>(iLS,             // set index
                                                                                          config.Z,        // lifting size
                                                                                          config.Kb,       // num info nodes
                                                                                          config.mb,       // num parity nodes
                                                                                          normalization,   // normalization
                                                                                          tOptLLR.value(), // LLR tensor
                                                                                          tOutputWord,     // output decision bits
                                                                                          workspace,       // workspace
                                                                                          isLast,          // last iteration? ( if so, write hard decisions)
                                                                                          iter);
    }

#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

////////////////////////////////////////////////////////////////////////
// decode_multi_kernel()
cuphyStatus_t decode_multi_kernel(LDPC_output_t&      tDst,
                                  const_tensor_pair&  tLLR,
                                  const LDPC_config&  config,
                                  float               normalization,
                                  cuphyLDPCResults_t* results,
                                  void*               workspace,
                                  cudaStream_t        strm)
{
    DEBUG_PRINTF("ldpc::decode_multi_kernel()\n");
    switch(tLLR.first.get().type())
    {
    case CUPHY_R_32F:
        return launch_decode_multi_kernel<CUPHY_R_32F>(tDst,
                                                       tLLR,
                                                       config,
                                                       normalization,
                                                       results,
                                                       workspace,
                                                       strm);
    case CUPHY_R_16F:
        return launch_decode_multi_kernel<CUPHY_R_16F>(tDst,
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
// decode_multi_kernel_workspace_size()
std::pair<bool, size_t> decode_multi_kernel_workspace_size(const LDPC_config& config)
{
    switch(config.type)
    {
    case CUPHY_R_32F:
        return std::pair<bool, size_t>(true,
                                       workspace_multi_kernel_t<data_type_traits<CUPHY_R_32F>::type>::get_size(config.mb,
                                                                                                               config.Kb,
                                                                                                               config.Z,
                                                                                                               config.num_codewords));
    case CUPHY_R_16F:
        return std::pair<bool, size_t>(true,
                                       workspace_multi_kernel_t<data_type_traits<CUPHY_R_16F>::type>::get_size(config.mb,
                                                                                                               config.Kb,
                                                                                                               config.Z,
                                                                                                               config.num_codewords));
    default:
        return std::pair<bool, size_t>(false, 0);
    }
}

} // namespace ldpc
