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

#include "ldpc_ms_cta_shmem_flooding.hpp"
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

////////////////////////////////////////////////////////////////////////
// Fixed length array of compressed Check to Variable (C2V) messages,
// processed by a single thread.
template <typename LLR_t, int NCHECK>
struct c2v_msg_vec_t
{
    typedef c2v_msg_t<LLR_t> message_t;
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
    // Initializes each element in the vector using channel LLR data.
    // (This is typically done on the first iteration, before V2C
    // messages have been created.)
    CUDA_INLINE void initialize(const bg1_graph_t& graph,           // graph shift data
                                int                graphRow,        // check node row
                                int                nodeOffsetStart, // offset of check variable within node
                                int                Z,               // lifting factor
                                const LLR_t*       initLLR)               // channel LLRs
    {
#pragma unroll
        for(int i = 0; i < NCHECK; ++i)
        {
            messages[i] = message_t::create_message(graph,
                                                    graphRow,
                                                    nodeOffsetStart + i,
                                                    Z,
                                                    initLLR);
        }
    }
    //------------------------------------------------------------------
    // update_app_atomic()
    __device__ void update_app_atomic(const bg1_graph_t& graph,           // shift data
                                      int                graphRow,        // check node row
                                      int                nodeOffsetStart, // offset of first check variable (within node)
                                      LLR_t*             shAPP,           // channel LLRs
                                      LLR_t              normalization,   // min-sum normalization
                                      int                Z)                              // lifting value
    {
        const int32_t ROW_DEGREE = bg1_row_degrees[graphRow];
        for(int iVN = 0; iVN < ROW_DEGREE; ++iVN)
        {
            const ldpc_column_info_t& COL_INFO = graph[graphRow][iVN];
            // We don't update APP nodes for extension parity nodes.
            // Since there is only 1 check node attached, and the
            // update value will be subtracted, the APP value will
            // remain equal to the channel LLR.
            if(COL_INFO.index < BG1_NUM_KERNEL_NODES)
            {
#pragma unroll
                for(int i = 0; i < NCHECK; ++i)
                {
                    LLR_t     llrValue = messages[i].get_value_for_index(iVN, normalization);
                    const int VN_idx   = message_t::get_variable_index(graph, graphRow, iVN, nodeOffsetStart + i, Z);
                    atomicAdd(shAPP + VN_idx, llrValue);
                    // Replace line above with this to test the impact of atomics
                    //shAPP[VN_idx] += llrValue;
                    //KERNEL_PRINT_IF(VN_idx < 1, "Adding %f to APP[%i] (iVN = %i, threadIdx.x = %u, blockIdx.x = %u)\n",
                    //                to_float(llrValue), VN_idx, iVN, threadIdx.x, blockIdx.x);
                }
            }
        }
    }
    //------------------------------------------------------------------
    // update_app_atomic()
    __device__ void update_app_atomic(const bg1_graph_t&   graph,           // shift data
                                      const c2v_msg_vec_t& prev,            // previous iteration message
                                      int                  graphRow,        // check node row
                                      int                  nodeOffsetStart, // offset of first check variable (within node)
                                      LLR_t*               shAPP,           // channel LLRs
                                      LLR_t                normalization,   // min-sum normalization
                                      int                  Z)                                // lifting value
    {
        const int32_t ROW_DEGREE = bg1_row_degrees[graphRow];
        for(int iVN = 0; iVN < ROW_DEGREE; ++iVN)
        {
            const ldpc_column_info_t& COL_INFO = graph[graphRow][iVN];
            // We don't update APP nodes for extension parity nodes.
            // Since there is only 1 check node attached, and the
            // update value will be subtracted, the APP value will
            // remain equal to the channel LLR.
            if(COL_INFO.index < BG1_NUM_KERNEL_NODES)
            {
#pragma unroll
                for(int i = 0; i < NCHECK; ++i)
                {
                    LLR_t     llrNew = messages[i].get_value_for_index(iVN, normalization);
                    LLR_t     llrOld = prev.messages[i].get_value_for_index(iVN, normalization);
                    const int VN_idx = message_t::get_variable_index(graph, graphRow, iVN, nodeOffsetStart + i, Z);
                    atomicAdd(shAPP + VN_idx, llrNew - llrOld);
                    // Replace the line above with this to test the impact of atomics
                    //shAPP[VN_idx] += (llrNew - llrOld);
                    //KERNEL_PRINT_IF(VN_idx < 1, "Adding %f to APP[%i] (iVN = %i, threadIdx.x = %u, blockIdx.x = %u)\n",
                    //                to_float(llrValue), VN_idx, iVN, threadIdx.x, blockIdx.x);
                }
            }
        }
    }
    //------------------------------------------------------------------
    // update_c2v()
    __device__ void update_c2v(const bg1_graph_t&   graph,           // graph shift data
                               int                  graphRow,        // check node row
                               int                  nodeOffsetStart, // offset of check variable within node
                               const LLR_t*         shAPP,           // channel LLRs
                               LLR_t                normalization,   // min-sum normalization
                               int                  Z,               // lifting factor
                               const c2v_msg_vec_t& src)             // Source check to variable
    {
        const int32_t ROW_DEGREE = bg1_row_degrees[graphRow];
        for(int iVN = 0; iVN < ROW_DEGREE; ++iVN)

        {
#pragma unroll
            for(int i = 0; i < NCHECK; ++i)
            {
                // Calculate the V2C message from last iteration
                LLR_t     prevValue = src.messages[i].get_value_for_index(iVN, normalization);
                const int VN_idx    = message_t::get_variable_index(graph, graphRow, iVN, nodeOffsetStart + i, Z);
                LLR_t     v2c       = shAPP[VN_idx] - prevValue;
                // Use the new V2C value to update the C2V message for the next
                // iteration.
                //KERNEL_PRINT_IF((0 == nodeOffsetStart) && (0 == nodeIndex) && (0 == i), "checkIdx = %i, iVN = %i, APP[%i] = %.4f, prevValue = %.4f, V2C = %.4f\n",
                //                (graphRow * config.Z) + nodeOffsetStart + i, iVN, VN_idx, to_float(shAPP[VN_idx]), to_float(prevValue), to_float(v2c));
                messages[i].process(v2c, iVN);
            }
        } // iVN
    }
};

////////////////////////////////////////////////////////////////////////
// shmem_C2V_fp16
// Shared memory storage of Check to Variable (C2V) messages in a
// compressed representation.
// Stores min0/min1 values and sign/index values as separate arrays,
// and in thread-major order to avoid/minimize bank conflicts when
// reading and writing.
template <int NTHREADS, int CHECKS_PER_THREAD>
struct shmem_C2V_fp16
{
    //------------------------------------------------------------------
    typedef __half2           min_t;
    typedef uint32_t          sign_index_t;
    typedef c2v_msg_t<__half> message_t;
    //------------------------------------------------------------------
    // Retrieve from shared memory
    __device__
        message_t
        get(unsigned int threadIndex, int perThreadIndex)
    {
        return message_t(min_values[perThreadIndex][threadIndex].x,
                         min_values[perThreadIndex][threadIndex].y,
                         sign_index[perThreadIndex][threadIndex]);
    }
    //------------------------------------------------------------------
    // Write to shared memory
    __device__ void put(unsigned int threadIndex, int perThreadIndex, message_t msg)
    {
        min_values[perThreadIndex][threadIndex] = make_half2(msg.min0, msg.min1);
        sign_index[perThreadIndex][threadIndex] = msg.sign_index;
    }
    //------------------------------------------------------------------
    // Data
    min_t        min_values[CHECKS_PER_THREAD][NTHREADS];
    sign_index_t sign_index[CHECKS_PER_THREAD][NTHREADS];
};

////////////////////////////////////////////////////////////////////////
// CTA Dimensions: x = THREADS_PER_NODE = Z / CHECKS_PER_THREAD
//                 y = mb
// Example: Z = 384, CHECKS_PER_THREAD = 2, mb = 4 (highest code rate)
//          NUM_PARITY_CHECKS = 384 * 4 = 1536
//          blockDim.x = 192, blockDim.y = 4
//          THREADS_PER_BLOCK = 4 * 192 = 768
// Each thread block decodes a single codeword. All threads in a block
// with the same threadIdx.x operate on the same node.
template <cuphyDataType_t TLLREnum,
          int             NTHREADS,
          int             MIN_BLOCKS,
          int             Z,
          int             Kb,
          int             mb>
__global__
    __launch_bounds__(NTHREADS, MIN_BLOCKS) void ldpc_ms_cta_shmem_flooding(LDPC_output_t                        tOutput,
                                                                            int                                  K,
                                                                            float                                normalization,
                                                                            const_tensor_ref_contig_2D<TLLREnum> tLLR,
                                                                            void*                                workspaceAddress,
                                                                            int                                  maxIterations,
                                                                            LDPC_output_diagnostic_t<TLLREnum>   diag)
{
    typedef typename data_type_traits<TLLREnum>::type LLR_t;
    typedef c2v_msg_t<LLR_t>                          message_t;
    //------------------------------------------------------------------
    // LDPC graph for Z template parameter:
    //const bg1_graph_t& graph = BG1_graph<Z>::graph();
    //const bg1_graph_t& graph = bg1_384;


    //if((0 == threadIdx.x) && (0 == threadIdx.y) && (0 == blockIdx.x))
    //{
    //    for(int i = 0; i < BG1_M; ++i)
    //    {
    //        for(int j = 0; j < bg1_row_degrees[i]; ++j)
    //        {
    //            ldpc_column_info_t col = graph[i][j];
    //            printf("(%2i,%3i)", col.index, col.shift);
    //        }
    //        printf("\n");
    //    }
    //}
    //------------------------------------------------------------------

    constexpr int NUM_PARITY_CHECKS       = Z * mb;
    constexpr int NUM_APP                 = Z * (Kb + mb);
    constexpr int CHECKS_PER_THREAD       = NUM_PARITY_CHECKS / NTHREADS;
    const int     CHECK_NODE_INDEX        = threadIdx.y;
    const int     CHECK_NODE_OFFSET_START = threadIdx.x * CHECKS_PER_THREAD;
    const int     CODEWORD_INDEX          = blockIdx.x;

    //__shared__ shmem_C2V_fp16<NTHREADS, CHECKS_PER_THREAD> shC2V0;
    //__shared__ shmem_C2V_fp16<NTHREADS, CHECKS_PER_THREAD> shC2V1;
    __shared__ LLR_t shAPP[NUM_APP];

    //------------------------------------------------------------------
    // Copy LLR data to shared memory
    {
        const LLR_t* channelLLR     = &tLLR({0, CODEWORD_INDEX});
        //const LLR_t* channelLLR     = tLLR + (CODEWORD_INDEX * tLLRStride);
        block_copy_sync_2D(shAPP, channelLLR, NUM_APP);
    }
    //------------------------------------------------------------------
    // Fixed length vectors of compressed C2V messages that will be used
    // by this thread.
    // c2vVec0: "previous" iteration value
    // c2vVec1: "new" value
    c2v_msg_vec_t<LLR_t, CHECKS_PER_THREAD> c2vVec0, c2vVec1;

    //------------------------------------------------------------------
    for(int iIter = 0; iIter < maxIterations; ++iIter)
    {
        //KERNEL_PRINT_IF(threadIdx.x < 4, "threadIdx = (%u, %u), CHECK_NODE_INDEX = %i, NODE_OFFSET_START = %i\n",
        //                threadIdx.x, threadIdx.y, CHECK_NODE_INDEX, CHECK_NODE_OFFSET_START);
        if(0 == iIter)
        {
            c2vVec0.initialize(bg1_384, /*graph,*/
                               CHECK_NODE_INDEX,
                               CHECK_NODE_OFFSET_START,
                               Z,
                               shAPP);
            //KERNEL_PRINT_IF((threadIdx.x < 2) && (0 == threadIdx.y),
            //                "checkIdx = %i, min0 = %f, min1 = %f, index = %i\ncheckIdx = %i, min0 = %f, min1 = %f, index = %i\n",
            //                (CHECK_NODE_INDEX * Z) + CHECK_NODE_OFFSET_START,     to_float(c2vVec0.messages[0].min0), to_float(c2vVec0.messages[0].min1), c2vVec0.messages[0].get_row_index(),
            //                (CHECK_NODE_INDEX * Z) + CHECK_NODE_OFFSET_START + 1, to_float(c2vVec0.messages[1].min0), to_float(c2vVec0.messages[1].min1), c2vVec0.messages[1].get_row_index());
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Increment APP values
            c2vVec0.update_app_atomic(bg1_384, /*graph,*/
                                      CHECK_NODE_INDEX,
                                      CHECK_NODE_OFFSET_START,
                                      shAPP,
                                      normalization,
                                      Z);
        }
        else
        {
            // Calculate a new set of messages based on the previous message
            c2vVec1.initialize();
            c2vVec1.update_c2v(bg1_384, /*graph,*/      // graph shift data
                               CHECK_NODE_INDEX,        // check node row
                               CHECK_NODE_OFFSET_START, // offset of check variable within node
                               shAPP,                   // channel LLRs
                               normalization,           // min-sum normalization
                               Z,                       // lifting factor
                               c2vVec0);                // source check to variable
            c2vVec1.update_app_atomic(bg1_384,          /*graph,*/
                                      c2vVec0,
                                      CHECK_NODE_INDEX,
                                      CHECK_NODE_OFFSET_START,
                                      shAPP,
                                      normalization,
                                      Z);
            // Copy to message 0
            c2vVec0 = c2vVec1;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Allow all threads to read from shAPP (above) before updating
        // it (below)
        //print_array_sync("APP", shAPP, Z * Kb);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Make sure all threads have updated APP before starting the
        // next iteration.
        __syncthreads();

        //KERNEL_PRINT_IF((threadIdx.x == 0) && (0 == threadIdx.y),
        //                "iteration %i: Writing C2V message with min0 = %f, min1 = %f, signs = 0x%X, row_index = %i, checkIdx = %i\n"
        //                "iteration %i: Writing C2V message with min0 = %f, min1 = %f, signs = 0x%X, row_index = %i, checkIdx = %i\n",
        //                iIter,
        //                to_float(messageVector0.messages[0].min0),
        //                to_float(messageVector0.messages[0].min1),
        //                messageVector0.messages[0].get_signs(),
        //                messageVector0.messages[0].get_row_index(),
        //                (CHECK_NODE_INDEX * config.Z) + NODE_OFFSET_START + 0,
        //                iIter,
        //                to_float(messageVector0.messages[1].min0),
        //                to_float(messageVector0.messages[1].min1),
        //                messageVector0.messages[1].get_signs(),
        //                messageVector0.messages[1].get_row_index(),
        //                (CHECK_NODE_INDEX * config.Z) + NODE_OFFSET_START + 1);
        LDPC_DIAGNOSTIC_IF(diag.addr(), block_copy_sync_2D(&diag({0, CODEWORD_INDEX, iIter}), shAPP, Z * Kb));
    }

    //------------------------------------------------------------------
    // Write hard decision values based on APP array
    {
        cta_write_hard_decision<TLLREnum>(tOutput,
                                          CODEWORD_INDEX,
                                          Z * Kb,
                                          shAPP);
    }
}

////////////////////////////////////////////////////////////////////////
// launch_decode_ms_cta_shmem_flooding()
template <cuphyDataType_t TType>
cuphyStatus_t launch_decode_ms_cta_shmem_flooding(LDPC_output_t&         tOutputWord,
                                                  const_tensor_pair&     tLLRInput,
                                                  const LDPC_config&     config,
                                                  float                  normalization,
                                                  cuphyLDPCResults_t*    results,
                                                  void*                  workspace,
                                                  cuphyLDPCDiagnostic_t* diag,
                                                  cudaStream_t           strm)
{
    DEBUG_PRINTF("ldpc::launch_decode_ms_cta_shmem_flooding()\n");
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
    //------------------------------------------------------------------
    if((384 != config.Z) || (4 != config.mb))
    {
        // TODO: Remove this
        return CUPHY_STATUS_UNSUPPORTED_LAYOUT;
    }

    const int DYN_SHMEM_SIZE = 0;
    // TODO: set up templated function calls to different combinations
    const int CHECKS_PER_THREAD = 2;
    const int MIN_BLKS          = 1;
    const int Z                 = 384;
    const int Kb                = 22;
    const int mb                = 4;
    const int BLK_THREADS       = Z * mb / CHECKS_PER_THREAD;

    dim3 blkDim(config.Z / CHECKS_PER_THREAD, config.mb);
    dim3 grdDim(config.num_codewords);
    //using ldpc_ms_cta_flooding_fcn = ldpc_ms_cta_flooding<TType, 384 * 2, 2, 1>;
    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc_ms_cta_shmem_flooding<TType, BLK_THREADS, MIN_BLKS, Z, Kb, mb>));
    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc_ms_cta_shmem_flooding<TType, BLK_THREADS, MIN_BLKS, Z, Kb, mb>), blkDim, DYN_SHMEM_SIZE);
    DEBUG_PRINTF("grid = (%u, %u, %u), block = (%u, %u, %u), shmem = %i\n",
                 grdDim.x,
                 grdDim.y,
                 grdDim.z,
                 blkDim.x,
                 blkDim.y,
                 blkDim.z,
                 DYN_SHMEM_SIZE);

    LDPC_output_diagnostic_t<TType> tDiagnostic = ldpc::template get_diagnostic_tensor_ref<TType>(diag);
    ldpc_ms_cta_shmem_flooding<TType, BLK_THREADS, MIN_BLKS, Z, Kb, mb>
        <<<grdDim, blkDim, DYN_SHMEM_SIZE>>>(tOutputWord,
                                             config.Kb * config.Z,
                                             normalization,
                                             tOptLLR.value(),
                                             workspace,
                                             config.max_iterations,
                                             tDiagnostic);
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
// decode_ms_cta_shmem_flooding()
cuphyStatus_t decode_ms_cta_shmem_flooding(LDPC_output_t&         tDst,
                                           const_tensor_pair&     tLLR,
                                           const LDPC_config&     config,
                                           float                  normalization,
                                           cuphyLDPCResults_t*    results,
                                           void*                  workspace,
                                           cuphyLDPCDiagnostic_t* diag,
                                           cudaStream_t           strm)
{
    DEBUG_PRINTF("ldpc::decode_ms_cta_shmem_flooding()\n");
    switch(tLLR.first.get().type())
    {
    case CUPHY_R_16F:
        return launch_decode_ms_cta_shmem_flooding<CUPHY_R_16F>(tDst,
                                                                tLLR,
                                                                config,
                                                                normalization,
                                                                results,
                                                                workspace,
                                                                diag,
                                                                strm);
    default:
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
}

////////////////////////////////////////////////////////////////////////
// decode_ms_cta_shmem_flooding_workspace_size()
std::pair<bool, size_t> decode_ms_cta_shmem_flooding_workspace_size(const LDPC_config& cfg)
{
    switch(cfg.type)
    {
    case CUPHY_R_16F:
    {
        typedef data_type_traits<CUPHY_R_16F>::type LLR_t;
        return std::pair<bool, size_t>(true,
                                       0);
    }
    default:
        return std::pair<bool, size_t>(false, 0);
    }
}

} // namespace ldpc
