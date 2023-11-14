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

#include "ldpc_ms_mka_flooding_flat.hpp"
#include "ldpc.cuh"
#include "nrLDPC_flat.cuh"

typedef int32_t            bg1_row_degree_array_t[BG1_M];
typedef ldpc_column_info_t bg1_graph_t[BG1_M][BG1_MAX_ROW_DEG];

////////////////////////////////////////////////////////////////////////
// BG1_graph
// Template to access 3GPP spec column/shift info using a template int
// parameter. Specializations for each Z return a reference to the
// appropriate const data structure.
template <int N>
struct BG1_graph;

template <>
struct BG1_graph<384>
{
    __device__ static constexpr const bg1_graph_t&            graph() { return bg1_384; }
    __device__ static constexpr const bg1_row_degree_array_t& row_degrees() { return bg1_row_degrees; }
};

////////////////////////////////////////////////////////////////////////
// C2V message
// "Compressed" Check to Variable (C2V) message representation
template <typename LLR_t>
struct c2v_msg_t
{
    LLR_t    min0;
    LLR_t    min1;
    uint32_t sign_index; // Low 19 bits are sign bits (a set bit indicates a negative number)
                         // High 13 bits are the index, within the row values, of min0. This number
                         // must be between 0 and (ROW_DEGREE - 1). (max row degree is 19, so value < 19)
    /*CUDA_BOTH_INLINE*/ c2v_msg_t() = default;
    CUDA_BOTH_INLINE c2v_msg_t(LLR_t v0, LLR_t v1) :
        sign_index(0)
    {
        LLR_t    abs0  = llr_abs(v0);
        LLR_t    abs1  = llr_abs(v1);
        uint32_t signs = is_neg(v0) ? 1 : 0;
        if(is_neg(v1)) signs |= 2;
        set_signs(signs);
        if(abs0 < abs1)
        {
            min0 = abs0;
            min1 = abs1;
            set_row_index(0);
        }
        else
        {
            min0 = abs1;
            min0 = abs0;
            set_row_index(1);
        }
    }
    CUDA_BOTH_INLINE c2v_msg_t(LLR_t m0, LLR_t m1, uint32_t sign, uint32_t index) :
        min0(m0),
        min1(m1),
        sign_index(sign | (index << 19)) {}
    CUDA_BOTH_INLINE c2v_msg_t(LLR_t m0, LLR_t m1, uint32_t signIndex) :
        min0(m0),
        min1(m1),
        sign_index(signIndex) {}
    CUDA_BOTH_INLINE void set_signs(uint32_t s) { sign_index = (sign_index | (s & 0x7FFFF)); };
    CUDA_BOTH_INLINE uint32_t get_signs() const { return (sign_index & 0x7FFFF); }
    CUDA_BOTH_INLINE void     set_row_index(int idx) { sign_index = (sign_index | (idx << 19)); }
    CUDA_BOTH_INLINE uint32_t get_row_index() const { return (sign_index >> 19); }
    CUDA_INLINE LLR_t get_value_for_index(int rowIndex, float norm) const
    {
        uint32_t minRowIndex = get_row_index();
        uint32_t signs       = get_signs();
        LLR_t    minAbsLvc   = (rowIndex == minRowIndex) ? min1 : min0;
        LLR_t    signProd    = (0 != (__popc(signs & ~(1 << rowIndex)) & 1)) ? -1.0f : 1.0f; // TODO: get approprate constants for type
        //KERNEL_PRINT("col_index = %u, row_index = %u, signs = 0x%X, minAbsLvc = %.4f, signProd = %.4f, returning %.4f\n",
        //             minColIndex, minRowIndex, signs, to_float(minAbsLvc), to_float(signProd), to_float(type_convert<LLR_t>(norm) * minAbsLvc * signProd));
        return type_convert<LLR_t>(norm) * minAbsLvc * signProd;
    }
    CUDA_INLINE void process(LLR_t value, int row_index)
    {
        sign_index |= (is_neg(value) ? 1 : 0) << row_index;
        LLR_t Lvcabs = llr_abs(value);
        if(Lvcabs < min0)
        {
            set_row_index(row_index);
            min1 = min0;
            min0 = Lvcabs;
        }
        else if(Lvcabs < min1)
        {
            min1 = Lvcabs;
        }
    }
    CUDA_BOTH_INLINE void init()
    {
        min0 = min1 = 10000;
        sign_index  = 0;
    }
    static CUDA_INLINE int get_variable_index(const bg1_graph_t& graph,      // shift data
                                              int                row,        // check variable node index
                                              int                iVN,        // index within row
                                              int                nodeOffset, // offset of check variable within node
                                              int                Z)                         // lifting factor
    {
        const ldpc_column_info_t& COL_INFO    = graph[row][iVN];
        const int16_t             POS         = COL_INFO.index;
        int                       blockOffset = nodeOffset + COL_INFO.shift;
        if(blockOffset >= Z) blockOffset -= Z;
        return (POS * Z) + blockOffset;
    }
    static CUDA_INLINE c2v_msg_t create_message(const bg1_graph_t& graph,      // shift data
                                                int                graphRow,   // check node row
                                                int                nodeOffset, // offset of check variable within node
                                                int                Z,          // lifting factor
                                                const LLR_t*       initLLR)          // initial LLR data
    {
        uint32_t signBits = 0;
        int      minIndex;
        LLR_t    min0, min1;
        // The minimum row degree in BG1 is 3, so we Unroll the first 3
        int   VN0     = get_variable_index(graph, graphRow, 0, nodeOffset, Z);
        int   VN1     = get_variable_index(graph, graphRow, 1, nodeOffset, Z);
        int   VN2     = get_variable_index(graph, graphRow, 2, nodeOffset, Z);
        LLR_t Lvc0    = initLLR[VN0];
        LLR_t Lvc1    = initLLR[VN1];
        LLR_t Lvc2    = initLLR[VN2];
        LLR_t LvcAbs0 = llr_abs(Lvc0);
        LLR_t LvcAbs1 = llr_abs(Lvc1);
        LLR_t LvcAbs2 = llr_abs(Lvc2);
        signBits      = (is_neg(Lvc0) ? 1 : 0);
        signBits |= (is_neg(Lvc1) ? 2 : 0);
        if(LvcAbs0 < LvcAbs1)
        {
            minIndex = 0;
            min0     = LvcAbs0;
            min1     = LvcAbs1;
        }
        else
        {
            minIndex = 1;
            min0     = LvcAbs1;
            min1     = LvcAbs0;
        }
        signBits |= (is_neg(Lvc2) ? 4 : 0);
        if(LvcAbs2 < min0)
        {
            minIndex = 2;
            min1     = min0;
            min0     = LvcAbs2;
        }
        else if(LvcAbs2 < min1)
        {
            min1 = LvcAbs2;
        }
        const int32_t ROW_DEGREE = bg1_row_degrees[graphRow];
        for(int iVN = BG1_MIN_ROW_DEG; iVN < ROW_DEGREE; ++iVN)
        {
            const int VN_idx = get_variable_index(graph, graphRow, iVN, nodeOffset, Z);
            LLR_t     Lvc    = initLLR[VN_idx];
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
        return c2v_msg_t(min0, min1, signBits, minIndex);
    }
};

template <typename LLR_t>
CUDA_INLINE void copy_c2v_msg(c2v_msg_t<LLR_t>* dst, const c2v_msg_t<LLR_t>* src)
{
    *dst = *src;
}

template <>
CUDA_INLINE void copy_c2v_msg(c2v_msg_t<__half>* dst, const c2v_msg_t<__half>* src)
{
    *(reinterpret_cast<uint2*>(dst)) = *(reinterpret_cast<const uint2*>(src));
}

////////////////////////////////////////////////////////////////////////
// workspace_ms_mka_flooding_flat
template <typename TLLR>
struct workspace_ms_mka_flooding_flat : public LDPC_workspace<workspace_ms_mka_flooding_flat<TLLR>>
{
    typedef TLLR                                                 LLR_t;
    typedef c2v_msg_t<TLLR>                                      message_t;
    typedef LDPC_workspace<workspace_ms_mka_flooding_flat<TLLR>> inherited_t;
    //------------------------------------------------------------------
    // Constructor
    __device__
    workspace_ms_mka_flooding_flat(void* pv, const LDPC_config& config) :
        inherited_t(pv)
    {
    }
    //------------------------------------------------------------------
    // workspace_bytes_per_codeword()
    // For each codeword:
    //     C2V: (mb * Z) instances of c2v_msg_t
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

template <typename LLR_t, template <typename> class TWorkspace>
__global__ void ldpc_hard_decision_workspace(LDPC_output_t tOutput,
                                             void*         workspaceAddress,
                                             LDPC_config   config)
{
    //KERNEL_PRINT_GRID_ONCE("ldpc_hard_decision_workspace(), workspaceAddress = %p\n", workspaceAddress);
    const int         CODEWORD_IDX = blockIdx.x;
    TWorkspace<LLR_t> workspace(workspaceAddress, config);
    LLR_t*            dAPP            = workspace.APP(config, CODEWORD_IDX);
    const int         K               = config.Kb * config.Z;
    const int         WORDS_PER_CW    = (K + 31) / 32;
    const int         BIT_BLOCK_COUNT = (K + 1023) / 1024;
    const int         WARPS_PER_BLOCK = blockDim.x / 32;
    const int         WARP_IDX        = threadIdx.x / 32;
    const int         LANE_IDX        = threadIdx.x % 32;
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
            int idx = start_bit_idx + (i * 32) + LANE_IDX;
            //KERNEL_PRINT_IF(idx < 32, "APP[%i] = %f\n", idx, to_float(dAPP[idx]));
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
            //KERNEL_PRINT_IF(OUT_INDEX == 0, "output[%i] = 0x%X\n", OUT_INDEX, thread_output);
            tOutput({OUT_INDEX, CODEWORD_IDX}) = thread_output;
        }
    }
}

template <cuphyDataType_t TLLREnum, template <typename> class TWorkspace>
__global__ void ldpc_cp_llr_workspace(void*                                workspaceAddress,
                                      LDPC_config                          config,
                                      const_tensor_ref_contig_2D<TLLREnum> tLLRSrc)
{
    typedef typename data_type_traits<TLLREnum>::type LLR_t;
    //KERNEL_PRINT_GRID_ONCE("ldpc_cp_llr_workspace()\n");
    const unsigned int NLLR = config.Z * config.num_kernel_nodes();
    const unsigned int NCW  = tLLRSrc.layout().dimensions[1];
    TWorkspace<LLR_t>  workspace(workspaceAddress, config);
    for(int codeWordIndex = blockIdx.y; codeWordIndex < NCW; codeWordIndex += gridDim.y)
    {
        LLR_t* cwAPP = workspace.APP(config, codeWordIndex);
        for(int LLRIndex = (blockIdx.x * blockDim.x) + threadIdx.x;
            LLRIndex < NLLR;
            LLRIndex += (gridDim.x * blockDim.x))
        {
            //KERNEL_PRINT_IF(blockIdx.x == 0, "threadIdx.x = %u, blockIdx.x = %u, LLRIndex = %i, codeWordIndex = %i, NLLR = %i, inc = %i\n", threadIdx.x, blockIdx.x, LLRIndex, codeWordIndex, NLLR, (gridDim.x * blockDim.x));
            //KERNEL_PRINT_IF(LLRIndex < 32, "cp: APP[%i] = %f\n", LLRIndex, to_float(tLLRSrc({LLRIndex, codeWordIndex})));
            cwAPP[LLRIndex] = tLLRSrc({LLRIndex, codeWordIndex});
        }
    }
}

template <cuphyDataType_t TLLREnum,
          int             Z>
__global__
    //__launch_bounds__(384, 5)
    __launch_bounds__(384) void ldpc_flooding_multi_kernel_atomic_flat(LDPC_config                          config,
                                                                       float                                normalization,
                                                                       const_tensor_ref_contig_2D<TLLREnum> tLLR,
                                                                       void*                                workspaceAddress,
                                                                       int                                  iteration)
{
    typedef typename data_type_traits<TLLREnum>::type LLR_t;
    typedef c2v_msg_t<LLR_t>                          message_t;
    const int                                         CODEWORD_INDEX       = blockIdx.y;
    const int                                         CHECK_IDX            = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int                                         NODE_IDX             = CHECK_IDX / config.Z;
    const int                                         NODE_OFFSET          = CHECK_IDX % config.Z;
    //const int                                         CHECK_COUNT          = config.mb * config.Z;
    const int                                         NUM_APP_NODES_STORED = config.Kb + 4;
    workspace_ms_mka_flooding_flat<LLR_t>             workspace(workspaceAddress, config);
    const LLR_t*                                      channelLLR = &tLLR({0, CODEWORD_INDEX});
    c2v_msg_t<LLR_t>*                                 C2V        = workspace.C2V(config, CODEWORD_INDEX);
    LLR_t*                                            dAPP       = workspace.APP(config, CODEWORD_INDEX);
    //------------------------------------------------------------------
    // LDPC graph for Z template parameter:
    const bg1_graph_t&            graph       = BG1_graph<Z>::graph();
    const bg1_row_degree_array_t& row_degrees = bg1_row_degrees;

    //KERNEL_PRINT_GRID_ONCE("ldpc_flooding_multi_kernel_atomic_flat, CHECK_IDX = %i, CHECK_COUNT = %i\n", CHECK_IDX, CHECK_COUNT);
    //c2vMessage.init();
    message_t c2vOut;
    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    if(0 == iteration)
    {
        //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        // First iteration: use channel LLR as input
        //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
        // Iterate over received LLR data to generate the first APP
        // increment.
        c2vOut = message_t::create_message(graph,       // shift data
                                           NODE_IDX,    // check node
                                           NODE_OFFSET, // offset of check variable within node
                                           Z,           // lifting factor
                                           channelLLR);
        // Increment APP values based on C2V data
        for(int iVN = 0; iVN < row_degrees[NODE_IDX]; ++iVN)
        {
            const ldpc_column_info_t& COL_INFO = graph[NODE_IDX][iVN];
            if(COL_INFO.index < NUM_APP_NODES_STORED)
            {
                LLR_t     llrValue = c2vOut.get_value_for_index(iVN, normalization);
                const int VN_idx   = message_t::get_variable_index(graph, NODE_IDX, iVN, NODE_OFFSET, Z);
                atomicAdd(dAPP + VN_idx, llrValue);
                // WRONG! Used instead of previous line, only to time impact of atomic
                //dAPP[VN_idx] = llrValue;
                //KERNEL_PRINT_IF(VN_idx < 32, "Adding %f to APP[%i] (CHECK_IDX = %i, iVN = %i, threadIdx.x = %u, blockIdx.x = %u)\n",
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
        copy_c2v_msg(&c2vIn, C2V + CHECK_IDX);
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
        int rowIndexEnd = row_degrees[NODE_IDX];
        if(graph[NODE_IDX][rowIndexEnd - 1].index >= NUM_APP_NODES_STORED)
        {
            --rowIndexEnd;
        }
        //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   -  -  -
        for(int iVN = 0; iVN < rowIndexEnd; ++iVN)
        {
            // Calculate the V2C message from last iteration
            LLR_t     prevValue = c2vIn.get_value_for_index(iVN, normalization);
            const int VN_idx    = message_t::get_variable_index(graph, NODE_IDX, iVN, NODE_OFFSET, config.Z);
            LLR_t     v2c       = dAPP[VN_idx] - prevValue;
            // Use the new V2C value to update the C2V message for the next
            // iteration.
            //KERNEL_PRINT_IF(0 == CHECK_IDX, "CHECK_IDX = %i, iVN = %i, APP[%i] = %.4f, prevValue = %.4f, V2C = %.4f\n",
            //                CHECK_IDX, iVN, VN_idx, to_float(dAPP[VN_idx]), to_float(prevValue), to_float(v2c));
            c2vOut.process(v2c, iVN);
        } // iVN
        //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   -  -  -
        // If this check node has an extension parity node, process it
        if(rowIndexEnd != row_degrees[NODE_IDX])
        {
            const int VN_idx = message_t::get_variable_index(graph, NODE_IDX, rowIndexEnd, NODE_OFFSET, config.Z);
            LLR_t     v2c    = channelLLR[VN_idx];
            // Update the C2V message for the next iteration.
            c2vOut.process(v2c, rowIndexEnd);
        }
        //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -   -  -  -
        // Increment APP values based on C2V data
        for(int iVN = 0; iVN < row_degrees[NODE_IDX]; ++iVN)
        {
            const ldpc_column_info_t& COL_INFO = graph[NODE_IDX][iVN];
            if(COL_INFO.index < NUM_APP_NODES_STORED)
            {
                const int VN_idx = message_t::get_variable_index(graph, NODE_IDX, iVN, NODE_OFFSET, config.Z);
                LLR_t     llrNew = c2vOut.get_value_for_index(iVN, normalization);
                LLR_t     llrOld = c2vIn.get_value_for_index(iVN, normalization);
                atomicAdd(dAPP + VN_idx, llrNew - llrOld);
                // WRONG! Used instead of previous line, only to time impact of atomic
                //dAPP[VN_idx] = llrNew - llrOld;
                //KERNEL_PRINT_IF(VN_idx < 32, "Adding %f to APP[%i] (delta = %f) (CHECK_IDX = %i, iVN = %i, threadIdx.x = %u, blockIdx.x = %u)\n",
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
    copy_c2v_msg(C2V + CHECK_IDX, &c2vOut);
}

////////////////////////////////////////////////////////////////////////
// launch_decode_multi_kernel_atomic_flat()
template <cuphyDataType_t TType>
cuphyStatus_t launch_decode_multi_kernel_atomic_flat(LDPC_output_t&      tOutputWord,
                                                     const_tensor_pair&  tLLRInput,
                                                     const LDPC_config&  config,
                                                     float               normalization,
                                                     cuphyLDPCResults_t* results,
                                                     void*               workspace,
                                                     cudaStream_t        strm)
{
    DEBUG_PRINTF("ldpc::launch_decode_multi_kernel_atomic_flat()\n");
    typedef const_tensor_ref_contig_2D<TType>      const_tensor2f;
    typedef typename data_type_traits<TType>::type LLR_t;

    // The kernel is only implemented for contiguous, 2D tensors.
    // Attempt to convert to such a tensor descriptor.
    cuphy_optional<const_tensor2f> tOptLLR = tLLRInput.first.get().get_ref_contig_rank<TType, 2>(tLLRInput.second);
    if(!tOptLLR)
    {
        // Layout is not 2D contiguous
        return CUPHY_STATUS_UNSUPPORTED_LAYOUT;
    }
    // For now, only support Z = 384
    if(384 != config.Z)
    {
        return CUPHY_STATUS_UNSUPPORTED_LAYOUT;
    }
    //const int          SHMEM_SIZE = 0;
    const int NUM_STORED_LLR = (config.Kb + 4) * config.Z;
    //------------------------------------------------------------------
    DEBUG_PRINTF("NCW = %i, BG = %i, N = %i, K = %i, Kb = %i, mb = %i, Z = %i, M = %i, iLS = %i, R_trans = %.2f\n",
                 config.num_codewords,
                 config.BG,
                 (config.Kb + config.mb) * config.Z,
                 config.Kb * config.Z,
                 config.Kb,
                 config.mb,
                 config.Z,
                 config.mb * config.Z,
                 set_from_Z(config.Z),
                 static_cast<float>(config.Kb) / (config.Kb + config.mb - 2));
    //------------------------------------------------------------------
    // Perform initial copy of LLR data so that we can use atomic increment
    {
        dim3 blockDim(1024);
        dim3 gridDim(div_round_up(NUM_STORED_LLR, 1024), config.num_codewords);
        ldpc_cp_llr_workspace<TType, workspace_ms_mka_flooding_flat><<<gridDim, blockDim, 0, strm>>>(workspace, config, tOptLLR.value());
    }

    //------------------------------------------------------------------
    dim3 blkDimCN(config.Z);
    dim3 grdDimCN(config.mb, config.num_codewords);
    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc_flooding_multi_kernel_atomic_flat<TType, 384>));
    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc_flooding_multi_kernel_atomic_flat<TType, 384>), blkDimCN, 0);
    for(int iter = 0; iter < config.max_iterations; ++iter)
    {
        ldpc_flooding_multi_kernel_atomic_flat<TType, 384><<<grdDimCN, blkDimCN, 0, strm>>>(config,
                                                                                            normalization,
                                                                                            tOptLLR.value(),
                                                                                            workspace,
                                                                                            iter);
    }
    {
        dim3 blkDimHD(256);
        dim3 grdDimHD(config.num_codewords);
        DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc_hard_decision_workspace<LLR_t, workspace_ms_mka_flooding_flat>));
        DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc_hard_decision_workspace<LLR_t, workspace_ms_mka_flooding_flat>), blkDimHD, 0);
        ldpc_hard_decision_workspace<LLR_t, workspace_ms_mka_flooding_flat><<<grdDimHD, blkDimHD, 0, strm>>>(tOutputWord,
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
// decode_multi_kernel_atomic_flat()
cuphyStatus_t decode_multi_kernel_atomic_flat(LDPC_output_t&      tDst,
                                              const_tensor_pair&  tLLR,
                                              const LDPC_config&  config,
                                              float               normalization,
                                              cuphyLDPCResults_t* results,
                                              void*               workspace,
                                              cudaStream_t        strm)
{
    DEBUG_PRINTF("ldpc::decode_multi_kernel_atomic_flat()\n");
    switch(tLLR.first.get().type())
    {
    case CUPHY_R_32F:
        return launch_decode_multi_kernel_atomic_flat<CUPHY_R_32F>(tDst,
                                                                   tLLR,
                                                                   config,
                                                                   normalization,
                                                                   results,
                                                                   workspace,
                                                                   strm);
    case CUPHY_R_16F:
        return launch_decode_multi_kernel_atomic_flat<CUPHY_R_16F>(tDst,
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

std::pair<bool, size_t> decode_multi_kernel_atomic_flat_workspace_size(const LDPC_config& config)
{
    switch(config.type)
    {
    case CUPHY_R_32F:
    {
        typedef data_type_traits<CUPHY_R_32F>::type   LLR_t;
        typedef workspace_ms_mka_flooding_flat<LLR_t> workspace_t;
        return std::pair<bool, size_t>(true,
                                       workspace_t::get_workspace_size(config));
    }
    case CUPHY_R_16F:
    {
        typedef data_type_traits<CUPHY_R_16F>::type   LLR_t;
        typedef workspace_ms_mka_flooding_flat<LLR_t> workspace_t;
        return std::pair<bool, size_t>(true,
                                       workspace_t::get_workspace_size(config));
    }
    default:
        return std::pair<bool, size_t>(false, 0);
    }
}

} // namespace ldpc
