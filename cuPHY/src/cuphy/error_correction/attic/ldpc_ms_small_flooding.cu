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

#include "ldpc_ms_small_flooding.hpp"
#include "ldpc.cuh"
#include "nrLDPC.cuh"
#include <cooperative_groups.h>

namespace cg = cooperative_groups;

template <typename TLLR>
struct kernel_shared_mem_precompute_t
{
    __device__ kernel_shared_mem_precompute_t(int Kb, int Z) :
        Kb_(Kb),
        Z_(Z)
    {
    }
    static int get_shared_mem_size(int Kb, int Z, int M, int MAX_ROW_DEG)
    {
        // Shared memory requirements:
        // number                 type       description
        // (Kb + 4) * Z           TLLR       a priori probability (APP) array, 1 per coded bit (without extensions values)
        // (Kb + 4) * Z           TLLR       log-likelihood ratio (from block input), 1 per coded bit
        // 1                      int        per-block count of failed parity checks
        // M * MAX_ROW_DEG        TLLR       V2C messages
        // TODO: Kb + 4 only valid for BG1
        return ((((Kb + 4) * Z * 2) + (M * MAX_ROW_DEG)) * sizeof(TLLR) + sizeof(int));
    }
    __device__ int& check_fail_count() { return *offset_as<int>(0); }
    __device__
        TLLR*
        app_addr() { return offset_as<TLLR>(sizeof(int)); }
    __device__
        TLLR*
        llr_addr() { return (app_addr() + ((Kb_ + 4) * Z_)); }
    __device__
        TLLR*
        v2c_addr() { return (llr_addr() + ((Kb_ + 4) * Z_)); }

private:
    template <typename T>
    __device__
        T*
        offset_as(int offset_bytes)
    {
        return reinterpret_cast<T*>(shmem_.addr() + offset_bytes);
    }
    shared_mem_t<char> shmem_;
    int                Kb_;
    int                Z_;
};

////////////////////////////////////////////////////////////////////////
// ldpc_flooding_bg1_sm_precompute()
template <cuphyDataType_t TLLREnum>
__global__ void
    __launch_bounds__(1024)
        ldpc_flooding_bg1_sm_precompute(int                                  iLS,
                                        const LDPC_config&                   config,
                                        LDPC_output_t                        tOutput,
                                        const_tensor_ref_contig_2D<TLLREnum> tLLR,
                                        float                                norm,
                                        cuphyLDPCResults_t*                  results)

{
    //------------------------------------------------------------------
    // Data type corresponding to the cuPHYDataType_t value, as stored
    // in the tensor
    typedef typename const_tensor_ref_contig_2D<TLLREnum>::non_const_element_t LLR_t;
    cg::thread_block                                                           block = cg::this_thread_block();
    //------------------------------------------------------------------
    // Shared memory (dynamically allocated at launch)
    kernel_shared_mem_precompute_t<LLR_t> shmem(config.Kb, config.Z);
    LLR_t*                                sAPP = shmem.app_addr();
    LLR_t*                                sLLR = shmem.llr_addr();
    kernel_mat<LLR_t>                     sV2C(shmem.v2c_addr(), config.mb * config.Z);
    int&                                  checkFailCount = shmem.check_fail_count();
    //------------------------------------------------------------------
    // Initialize the parity check failure count
    if(0 == threadIdx.x)
    {
        checkFailCount = 0;
    }
    //------------------------------------------------------------------
    // Extension nodes are used by only one check node and only one
    // variable node. We conserve shared memory by storing these values
    // in the registers of threads that are the only users of those
    // values.
    LLR_t extensionAPP = 0;
    LLR_t extensionLLR = 0;
    //------------------------------------------------------------------
    // Compute the thread location in the parity check matrix
    const int               CHECK_IDX           = threadIdx.x;
    const int               NODE_IDX            = CHECK_IDX / config.Z;
    const int               NODE_OFFSET         = CHECK_IDX % config.Z;
    const int               CHECK_COUNT         = config.mb * config.Z;
    const bool              THREAD_CHECK_ACTIVE = (CHECK_IDX < CHECK_COUNT);
    const int               APP_BUFFER_SIZE     = (config.Kb + 4) * config.Z; // TODO: Valid for BG1 only
    const int               MAX_PARITY_COL      = 25;                         // TODO: Valid for BG1 only
    const int               Z                   = config.Z;
    bg1_CN_row_shift_info_t CNShift(NODE_IDX, Z);
    //------------------------------------------------------------------
    // Load LLR values into shared memory (both sLLR and sAPP)
    const int    CODEWORD_IDX = static_cast<int>(blockIdx.x);
    const LLR_t* channelLLR   = &tLLR({0, CODEWORD_IDX});
    block_copy_pair_sync(sAPP, sLLR, channelLLR, APP_BUFFER_SIZE);
    //print_array_sync("LLR (init)", sLLR, APP_BUFFER_SIZE);
    //print_array_sync("APP (init)", sAPP, N);
    //------------------------------------------------------------------
    //KERNEL_PRINT("CHECK_IDX = %i, NODE_IDX = %i, NODE_OFFSET = %i\n", CHECK_IDX, NODE_IDX, NODE_OFFSET);
    //------------------------------------------------------------------
    // Initialize C2V with input LLRs
    if(THREAD_CHECK_ACTIVE)
    {
        for(int iVN = 0; iVN < CNShift.row_degree; ++iVN)
        {
            const int8_t POS          = CNShift.column_values[iVN];
            int          block_offset = NODE_OFFSET + CNShift.shift_values[iVN];
            if(block_offset >= Z) block_offset -= Z;
            const int VN_idx     = (POS * Z) + block_offset;
            sV2C(CHECK_IDX, iVN) = (POS <= MAX_PARITY_COL) ? sAPP[VN_idx] : channelLLR[VN_idx];
            // Also store APP and LLR value for extension bits locally
            if(POS > MAX_PARITY_COL)
            {
                extensionAPP = extensionLLR = channelLLR[VN_idx];
            }
        }
    }
    //print_kernel_mat("V2C(init)", sV2C, blockDim.x, BG1_MAX_ROW_DEG);
    //------------------------------------------------------------------
    // Iterate 'maxIter' times (unless early termination occurs)
    int iIter = 0;
    while(iIter < config.max_iterations)
    {
        //KERNEL_PRINT_GRID_ONCE("ITER= %i\n", iIter);
        uint32_t signBits = 0;
        LLR_t    min1 = 10000, min2 = 10000; // TODO
        int      min_idx = -1;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Check Node Processing (updates of C2V messages)
        // For each variable node (VN) associated with this check node (CN)
        if(THREAD_CHECK_ACTIVE)
        {
            for(int iVN = 0; iVN < CNShift.row_degree; ++iVN)
            {
                LLR_t Lvc = sV2C(CHECK_IDX, iVN);
                //signBits |= ((__hlt(Lvc, 0)) ? 1 : 0) << iVN;
                //LLR_t Lvcabs = __hlt(Lvc, 0) ? __hneg(Lvc) : Lvc;
                signBits |= (is_neg(Lvc) ? 1 : 0) << iVN;
                LLR_t Lvcabs = llr_abs(Lvc);
                if(Lvcabs < min1)
                {
                    min_idx = iVN;
                    min2    = min1;
                    min1    = Lvcabs;
                }
                else if(Lvcabs < min2)
                {
                    min2 = Lvcabs;
                }
                //KERNEL_PRINT("CHECK_IDX = %i, NODE_IDX = %i, iVN = %i, Lvc = %f, min1 = %f, min2 = %f, min_idx = %i, signBits = 0x%X\n",
                //             CHECK_IDX, NODE_IDX, iVN, to_float(Lvc), to_float(min1), to_float(min2), min_idx, signBits);
            } // iVN
            //KERNEL_PRINT("CHECK_IDX = %i, NODE_IDX = %i, min1 = %f, min2 = %f, min_idx = %i, signBits = 0x%X\n",
            //             CHECK_IDX, NODE_IDX, to_float(min1), to_float(min2), min_idx, signBits);
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Bit (Variable) Node Processing (updates of APP values)
            for(int iVN = 0; iVN < CNShift.row_degree; ++iVN)
            {
                const int8_t POS          = CNShift.column_values[iVN];
                int          block_offset = NODE_OFFSET + CNShift.shift_values[iVN];
                if(block_offset >= Z) block_offset -= Z;
                const int VN_idx    = (POS * Z) + block_offset;
                LLR_t     minAbsLvc = (iVN == min_idx) ? min2 : min1;
                LLR_t     signProd  = (0 != (__popc(signBits & ~(1 << iVN)) & 1)) ? -1.0f : 1.0f; // TODO: get approprate constants for type
                LLR_t     Lcv       = type_convert<LLR_t>(norm) * minAbsLvc * signProd;
                if(POS <= MAX_PARITY_COL)
                {
                    //sAPP[] = sLLR[] + sum(C2V)
                    //KERNEL_PRINT_IF(768 == VN_idx, "CHECK_IDX = %i, adding %f to sAPP[%i], min_idx was %i, minAbsLvc was %.4f, signBits was 0x%X, signProd was %.4f, iVN = %i\n",
                    //                CHECK_IDX, to_float(Lcv), VN_idx, min_idx, to_float(minAbsLvc), signBits, to_float(signProd), iVN);
                    atomicAdd(&sAPP[VN_idx], Lcv);
                }
                else
                {
                    extensionAPP = extensionLLR + Lcv;
                }
            } // iVN
        }     // if(THREAD_CHECK_ACTIVE)
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // All check nodes need updates to sAPP
        block.sync();
        //print_array_sync("APP (after atomicAdd())", sAPP, APP_BUFFER_SIZE);
        //KERNEL_PRINT_BLOCK_ONCE("sAPP[0] = %f\n", sAPP[0]);
        uint32_t CN_bits = 0;
        if(THREAD_CHECK_ACTIVE)
        {
            //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
            // Update V2C Messages and check hard decision bits
            for(int iVN = 0; iVN < CNShift.row_degree; ++iVN)
            {
                const int8_t POS = CNShift.column_values[iVN];
                LLR_t        APP;
                if(POS <= MAX_PARITY_COL)
                {
                    int block_offset = NODE_OFFSET + CNShift.shift_values[iVN];
                    if(block_offset >= Z) block_offset -= Z;
                    const int VN_idx     = (POS * Z) + block_offset;
                    LLR_t     minAbsLvc  = (iVN == min_idx) ? min2 : min1;
                    LLR_t     signProd   = (0 != (__popc(signBits & ~(1 << iVN)) & 1)) ? -1.0f : 1.0f; // TODO: get approprate constants for type
                    LLR_t     Lcv        = type_convert<LLR_t>(norm) * signProd * minAbsLvc;
                    APP                  = sAPP[VN_idx];
                    sV2C(CHECK_IDX, iVN) = APP - Lcv;
                    //KERNEL_PRINT_IF(0 == CHECK_IDX, "CHECK_IDX = %i, iVN = %i, Lcv = %.4f, APP[%i] = %.4f, sV2C = %.4f\n", CHECK_IDX, iVN, to_float(Lcv), VN_idx, to_float(APP), to_float(sV2C(CHECK_IDX, iVN)));
                }
                else
                {
                    APP = extensionAPP;
                    // Store for next C2V loop
                    sV2C(CHECK_IDX, iVN) = extensionLLR;
                }
                //KERNEL_PRINT_IF(CHECK_IDX == 0, "CHECK_IDX = %i, SV2C[%i] = %.4f\n", CHECK_IDX, iVN, to_float(sV2C(CHECK_IDX, iVN)));
                // We don't need to maintain the values - just the sum mod 2,
                // but keep them for now for debugging.
                uint32_t hard_decision = is_neg(APP) ? 1 : 0;
                CN_bits |= (hard_decision << iVN);
            } // iVN
            //print_kernel_mat("V2C(updated)", sV2C, blockDim.x, BG1_MAX_ROW_DEG);
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Early termination check
            bool parity_correct = (0 == (__popc(CN_bits) % 2));
            //KERNEL_PRINT("CHECK_IDX = %i, iter = %i, CN_bits = 0x%X, popcount(CN_bits) = %i CHECK = %s\n", CHECK_IDX, iIter, CN_bits, __popc(CN_bits), parity_correct ? "true" : "false");
            if(!parity_correct)
            {
                atomicAdd(&checkFailCount, 1);
#if 0
                // Print values associated with a parity check error
                for(int iVN = 0; iVN < BG1_MAX_ROW_DEG; ++iVN)
                {
                    const int8_t POS = bg1_pos[NODE_IDX][iVN];
                    if(POS >= 0)
                    {
                        int16_t shift        = shiftmat[NODE_IDX][iVN] % Z;
                        int     block_offset = NODE_OFFSET + shift;
                        if(block_offset >= Z) block_offset -= Z;
                        int VN_idx = (POS * Z) + block_offset;
                        KERNEL_PRINT("CHECK_IDX = %i, NODE_IDX = %i, POS = %i, table_shift = %i, shift = %i, VN_idx = %i, sAPP[VN_idx] = %f\n",
                                     CHECK_IDX, NODE_IDX, (int)POS, (int)(shiftmat[NODE_IDX][iVN]), (int)shift, VN_idx, sAPP[VN_idx]);
                    } // if(POS >= 0)
                } // iVN
#endif
            } // if(!parity_correct)
        }     // if(THREAD_CHECK_ACTIVE)
        // Sync to allow all threads to update the count of failed checks...
        block.sync();
        if(0 == checkFailCount)
        {
            //KERNEL_PRINT_BLOCK_ONCE("NO PARITY ERRORS AFTER %i ITERATIONS\n", iIter + 1);
            break;
        }
        else
        {
            //KERNEL_PRINT_GRID_ONCE("%i PARITY ERROR(S) AFTER %i ITERATIONS\n", checkFailCount, iIter + 1);
            if((iIter + 1) < config.max_iterations)
            {
                if(0 == block.thread_rank())
                {
                    checkFailCount = 0;
                }
                // Update APP values with the input LLRs in preparation for
                // the next iteration
                block_copy_sync(sAPP, sLLR, APP_BUFFER_SIZE);
            }
        }
        ++iIter;
    } // iIter
    //------------------------------------------------------------------
    // Write output bits. Each warp of 32 threads will cooperate to
    // generate up to 32 output "words", and each of those "words" will
    // contain 32 decision bits. (Each warp will read 1024 LLR values,
    // and generate 1024 output bits in 32 uint32_t words.)
    // Assuming launch configuration with full warps, even if the number
    // of check nodes is not a multiple of 32. (See the THREAD_HAS_CHECK
    // conditional above.) We can't just use full warps here - we could
    // have a launch with less than 32 check nodes, and in that case the
    // number of full warps would be zero.
    block.sync();
    //print_array_sync("APP (before write)", sAPP, N);
    cg::thread_block_tile<32> tile32          = cg::tiled_partition<32>(cg::this_thread_block());
    const int                 K               = config.Kb * Z;
    const int                 BIT_BLOCK_COUNT = (K + 1023) / 1024;
    const int                 WARPS_PER_BLOCK = block.size() / 32;
    const int                 WARP_IDX        = block.thread_rank() / 32;
    const int                 OUTPUT_WORDS    = (K + 31) / 32;
    for(int iOutBlock = WARP_IDX; iOutBlock < BIT_BLOCK_COUNT; iOutBlock += WARPS_PER_BLOCK)
    {
        uint32_t thread_output = 0;
        int      start_bit_idx = iOutBlock * 1024;
        for(int i = 0; i < 32; ++i)
        {
            int      idx           = start_bit_idx + (i * 32) + tile32.thread_rank();
            uint32_t hard_decision = ((idx < K) && is_neg(sAPP[idx])) ? 1 : 0;
            uint32_t warp_bits     = tile32.ballot(hard_decision);
            if(i == tile32.thread_rank())
            {
                thread_output = warp_bits;
            }
        }
        const int OUT_INDEX = (iOutBlock * 32) + (int)threadIdx.x;
        if(OUT_INDEX < OUTPUT_WORDS)
        {
            //KERNEL_PRINT_IF(0 == OUT_INDEX, "output[0] = 0x%X\n", thread_output);
            tOutput({OUT_INDEX, CODEWORD_IDX}) = thread_output;
        }
    }
    block.sync();
    //------------------------------------------------------------------
    // Populate the results structure for this block if the user
    // provided an address
    if((0 == threadIdx.x) && results)
    {
        cuphyLDPCResults_t res;
        res.numIterations   = min(iIter + 1, config.max_iterations);
        res.checkErrorCount = checkFailCount;
        // Write to device memory
        results[CODEWORD_IDX] = res;
    }
}

#if __CUDA_ARCH__ < 700
//template <>
//__global__ void
//ldpc_flooding_bg1_sm<CUPHY_R_16F>(int, int, int, int, LDPC_output_t, const_tensor_ref_contig_2D<CUPHY_R_16F>, int, float, cuphyLDPCResults_t*)
//{
//    printf("ldpc_flooding_bg1_sm<> not implemented for __CUDA_ARCH__ << 700\n");
//}
template <>
__global__ void
ldpc_flooding_bg1_sm_precompute<CUPHY_R_16F>(int, const LDPC_config&, LDPC_output_t, const_tensor_ref_contig_2D<CUPHY_R_16F>, float, cuphyLDPCResults_t*)
{
    printf("ldpc_flooding_bg1_sm_precompute<> not implemented for __CUDA_ARCH__ << 700\n");
}
#endif

namespace ldpc
{

////////////////////////////////////////////////////////////////////////
// launch_decode_small_flooding()
template <cuphyDataType_t TType>
cuphyStatus_t launch_decode_small_flooding(LDPC_output_t&      tOutputWord,
                                           const_tensor_pair&  tLLR,
                                           const LDPC_config&  config,
                                           float               normalization,
                                           cuphyLDPCResults_t* results,
                                           void*               workspace,
                                           cudaStream_t        strm)
{
    DEBUG_PRINTF("ldpc::launch_decode_small_flooding()\n");
    const tensor_desc&                        tLLRDesc = tLLR.first.get();
    typedef const_tensor_ref_contig_2D<TType> const_tensor2f;
    //------------------------------------------------------------------
    // The kernel is only implemented for contiguous, 2D tensors.
    // Attempt to convert to such a tensor descriptor.
    cuphy_optional<const_tensor2f> tOptLLR = tLLRDesc.get_ref_contig_rank<TType, 2>(tLLR.second);
    if(!tOptLLR)
    {
        // Layout is not 2D contiguous
        return CUPHY_STATUS_UNSUPPORTED_LAYOUT;
    }
    //------------------------------------------------------------------
    int       iLS        = set_from_Z(config.Z);
    const int SHMEM_SIZE = kernel_shared_mem_precompute_t<float>::get_shared_mem_size(config.Kb,
                                                                                      config.Z,
                                                                                      config.mb * config.Z,
                                                                                      BG1_MAX_ROW_DEG);
    dim3      gridDim(tLLRDesc.layout().dimensions[1]);
    dim3      blockDim(config.mb * config.Z);
    DEBUG_PRINTF("NCW = %i, BG = %i, K = %i, Kb = %i, mb = %i, Z = %i, M = %i, N = %i, iLS = %i, R_trans = %.2f, shared_mem = %i bytes, grid = (%u, %u, %u), block = (%u, %u, %u)\n",
                 gridDim.x,
                 config.BG,
                 config.Kb * config.Z,
                 config.Kb,
                 config.mb,
                 config.Z,
                 config.mb * config.Z,
                 (config.Kb + config.mb) * config.Z,
                 iLS,
                 static_cast<float>(config.Kb) / (config.Kb + config.mb - 2),
                 SHMEM_SIZE,
                 gridDim.x,
                 gridDim.y,
                 gridDim.z,
                 blockDim.x,
                 blockDim.y,
                 blockDim.z);
    ldpc_flooding_bg1_sm_precompute<TType><<<gridDim, blockDim, SHMEM_SIZE, strm>>>(iLS,             // set index
                                                                                    config,          // LDPC configuration
                                                                                    tOutputWord,     // output bits (addressed as words)
                                                                                    tOptLLR.value(), // LLR tensor
                                                                                    normalization,   // normalization
                                                                                    results);        // results
#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

////////////////////////////////////////////////////////////////////////
// decode_small_flooding()
cuphyStatus_t decode_small_flooding(LDPC_output_t&      tDst,
                                    const_tensor_pair&  tLLR,
                                    const LDPC_config&  config,
                                    float               normalization,
                                    cuphyLDPCResults_t* results,
                                    void*               workspace,
                                    cudaStream_t        strm)
{
    DEBUG_PRINTF("ldpc::decode_small_flooding()\n");
    //------------------------------------------------------------------
    // Disabling early termination not supported for this kernel yet
    if(!config.early_termination)
    {
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    //------------------------------------------------------------------
    switch(tLLR.first.get().type())
    {
    case CUPHY_R_32F:
        return launch_decode_small_flooding<CUPHY_R_32F>(tDst, tLLR, config, normalization, results, workspace, strm);
    case CUPHY_R_16F:
        return launch_decode_small_flooding<CUPHY_R_16F>(tDst, tLLR, config, normalization, results, workspace, strm);
    default:
        return CUPHY_STATUS_UNSUPPORTED_TYPE;
    }
}

////////////////////////////////////////////////////////////////////////
// decode_small_flooding_workspace_size()
std::pair<bool, size_t> decode_small_flooding_workspace_size(const LDPC_config& config)
{
    const int   storageElemBytes = get_cuphy_type_storage_element_size(config.type);
    // TODO: Kb + 4 not correct for BG2
    size_t szAPP = ((config.Kb + 4) * config.Z) * storageElemBytes;
    size_t szET  = sizeof(uint32_t);
    return std::pair<bool, size_t>(true, ((szAPP + szET) * config.num_codewords));
}

} // namespace ldpc
