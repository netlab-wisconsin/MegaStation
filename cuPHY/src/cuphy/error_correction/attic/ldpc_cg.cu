/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "nrLDPC.cuh"
#include "ldpc.hpp"
#include "type_convert.hpp"
#include <vector>
#include <cooperative_groups.h>

template <cuphyDataType_t TLLREnum, int THREAD_COUNT>
__global__ void
    __launch_bounds__(THREAD_COUNT, 2)
        ldpc_flooding_bg1_multi_sm_precompute(int                                  iLS,
                                              int                                  Kb,
                                              int                                  Z,
                                              int                                  mb,
                                              int                                  maxIter,
                                              LDPC_output_t                        tOutput,
                                              const_tensor_ref_contig_2D<TLLREnum> tLLR,
                                              void*                                workspace,
                                              float                                norm,
                                              cuphyLDPCResults_t*                  results)
{
    //KERNEL_PRINT_GRID_ONCE("ldpc_flooding_bg1_multi_sm_precompute()\n");
    //------------------------------------------------------------------
    // Data type corresponding to the cuPHYDataType_t value, as stored
    // in the tensor
    typedef typename const_tensor_ref_contig_2D<TLLREnum>::non_const_element_t LLR_t;
    //------------------------------------------------------------------
    // Shared memory (dynamically allocated at launch)
    lg_kernel_shared_mem_t<LLR_t> shmem(Kb, Z);
    LLR_t*                        sAPP = shmem.app_addr();
    kernel_mat<LLR_t>             sV2C(shmem.v2c_addr(), blockDim.x);
    int&                          shCheckFailCount = shmem.check_fail_count();
    //------------------------------------------------------------------
    const int               CW_INDEX         = blockIdx.y;
    const int               CHECK_IDX        = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int               NODE_IDX         = CHECK_IDX / Z;
    const int               NODE_OFFSET      = CHECK_IDX % Z;
    const int               CHECK_COUNT      = mb * Z;
    const bool              THREAD_HAS_CHECK = (CHECK_IDX < CHECK_COUNT);
    cg::thread_block        block            = cg::this_thread_block();
    cg::grid_group          grid             = cg::this_grid();
    const int               MAX_PARITY_COL   = 25;           // TODO: Valid for BG1 only
    const int               APP_BUFFER_SIZE  = (Kb + 4) * Z; // TODO: Valid for BG1 only
    bg1_CN_row_shift_info_t CNShift(NODE_IDX, Z);
    //------------------------------------------------------------------
    // Extension nodes are used by only one check node and only one
    // variable node. We conserve shared memory by storing these values
    // in the registers of threads that are the only users of those
    // values.
    LLR_t extensionAPP = 0;
    LLR_t extensionLLR = 0;
    //------------------------------------------------------------------
    // Workspace layout:
    // parity check failure counts (integers), 1 per codeword (used for early termination)
    // array of APP_BUFFER_SIZE elements, 1 array per codeword
    int&   codeWordCheckFailCount = *(static_cast<int*>(workspace) + CW_INDEX);
    LLR_t* workspaceAPPStart      = reinterpret_cast<LLR_t*>(static_cast<int*>(workspace) + gridDim.y);
    LLR_t* codewordAPP            = workspaceAPPStart + (CW_INDEX * APP_BUFFER_SIZE);
    //------------------------------------------------------------------
    // Initialize the per-block parity check failure count
    if(0 == block.thread_rank())
    {
        shCheckFailCount = 0;
    }
    //------------------------------------------------------------------
    // Copy the initial part of the LLR data to per-block shared memory.
    // (Extension LLRs will not be located in shared memory to save
    // resources.)
    const LLR_t* channelLLR = &tLLR({0, CW_INDEX});
    // Block copy of channelLLR data to shared memory
    block_copy_sync(sAPP, channelLLR, APP_BUFFER_SIZE);
    // Grid copy of channelLLR data to workspace for APP
    grid_copy_x(codewordAPP, channelLLR, APP_BUFFER_SIZE);
    //print_array_sync("APP (init)", sAPP, APP_BUFFER_SIZE);
    //------------------------------------------------------------------
    // Initialize C2V with input LLRs
    if(THREAD_HAS_CHECK)
    {
        for(int iVN = 0; iVN < CNShift.row_degree; ++iVN)
        {
            const int8_t POS          = CNShift.column_values[iVN];
            int          block_offset = NODE_OFFSET + CNShift.shift_values[iVN];
            if(block_offset >= Z) block_offset -= Z;
            const int VN_idx       = (POS * Z) + block_offset;
            sV2C(threadIdx.x, iVN) = (POS <= MAX_PARITY_COL) ? sAPP[VN_idx] : channelLLR[VN_idx];
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
    while(iIter < maxIter)
    {
        //KERNEL_PRINT_GRID_ONCE("ITER %i (block error count = %i, grid error count = %i):\n", iIter, shCheckFailCount, gridCheckFailCount);
        uint32_t signBits = 0;
        LLR_t    min1 = 10000, min2 = 10000; // TODO
        int      min_idx = -1;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Check Node Processing (updates of C2V messages)
        // For each variable node (VN) associated with this check node (CN)
        if(THREAD_HAS_CHECK)
        {
            for(int iVN = 0; iVN < CNShift.row_degree; ++iVN)
            {
                LLR_t Lvc = sV2C(threadIdx.x, iVN);
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
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // We only store increments (without the base LLR) in the shared APP
        // buffer, since contributions from multiple blocks will be
        // accumulated. (We don't want to add the LLR multiple times.)
        block_zero_sync(sAPP, APP_BUFFER_SIZE);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Bit (Variable) Node Processing (updates of block-local APP
        // values)
        if(THREAD_HAS_CHECK)
        {
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
                    //KERNEL_PRINT_IF(0 == VN_idx, "CHECK_IDX = %i, adding %f to sAPP[0]\n", CHECK_IDX, Lcv);
                    atomicAdd(&sAPP[VN_idx], Lcv);
                }
                else
                {
                    extensionAPP = extensionLLR + Lcv;
                }

            } // iVN
        }     // if(THREAD_HAS_CHECK)
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Sync on updates to block-local sAPP before updating global
        block.sync();
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Grid copy of channelLLR data to workspace for APP
        grid_copy_x(codewordAPP, channelLLR, APP_BUFFER_SIZE);
        //print_array_sync("APP (init)", sAPP, APP_BUFFER_SIZE);
        if((0 == threadIdx.x) && (0 == blockIdx.x))
        {
            codeWordCheckFailCount = 0;
        }
        grid.sync();
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Update global memory APP with local block APP data.
        for(int iVN = threadIdx.x; iVN < APP_BUFFER_SIZE; iVN += blockDim.x)
        {
            //KERNEL_PRINT_IF(0 == iVN, "Block %u Adding %f to global workspace APP[0]\n", blockIdx.x, sAPP[iVN]);
            atomicAdd(&codewordAPP[iVN], sAPP[iVN]);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Wait for atomic global memory updates from all blocks
        grid.sync();
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Copy data back to block-local sAPP (each block has own copy)
        block_copy_sync(sAPP, codewordAPP, APP_BUFFER_SIZE);
        //print_array_sync("APP (after atomicAdd())", sAPP, APP_BUFFER_SIZE);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        uint32_t CN_bits = 0;
        if(THREAD_HAS_CHECK)
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
                    const int VN_idx       = (POS * Z) + block_offset;
                    LLR_t     minAbsLvc    = (iVN == min_idx) ? min2 : min1;
                    LLR_t     signProd     = (0 != (__popc(signBits & ~(1 << iVN)) & 1)) ? -1.0f : 1.0f; // TODO: get approprate constants for type
                    LLR_t     Lcv          = type_convert<LLR_t>(norm) * signProd * minAbsLvc;
                    APP                    = sAPP[VN_idx];
                    sV2C(threadIdx.x, iVN) = APP - Lcv;
                }
                else
                {
                    APP = extensionAPP;
                    // Store for next C2V loop
                    sV2C(threadIdx.x, iVN) = extensionLLR;
                }
                // We don't need to maintain the values - just the sum mod 2,
                // but keep them for now for debugging.
                uint32_t hard_decision = is_neg(APP) ? 1 : 0;
                CN_bits |= (hard_decision << iVN);
            } // iVN
        }     // if(THREAD_HAS_CHECK)
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // No sync needed - sV2C not read until after ET sync
        if(THREAD_HAS_CHECK)
        {
            //            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            //            // Tentative Decoding
            //            // TODO: Fuse this with the above identical loop
            //            for(int iVN = 0; iVN < CNShift.row_degree; ++iVN)
            //            {
            //                const int8_t POS = CNShift.column_values[iVN];
            //                LLR_t APPValue;
            //                if(POS <= MAX_PARITY_COL)
            //                {
            //                    int       block_offset = NODE_OFFSET + CNShift.shift_values[iVN];
            //                    if(block_offset >= Z) block_offset -= Z;
            //                    const int VN_idx       = (POS * Z) + block_offset;
            //                    APPValue = sAPP[VN_idx];
            //                    //KERNEL_PRINT("CHECK_IDX = %i, NODE_IDX = %i, POS = %i, table_shift = %i, shift = %i, VN_idx = %i, sAPP[VN_idx] = %f\n",
            //                    //             CHECK_IDX, NODE_IDX, (int)POS, (int)(shiftmat[NODE_IDX][iVN]), (int)shift, VN_idx, sAPP[VN_idx]);
            //                }
            //                else
            //                {
            //                    APPValue = extensionAPP;
            //                    //KERNEL_PRINT("CHECK_IDX = %i, NODE_IDX = %i, POS = %i, VN_idx = %i, APP = %f\n",
            //                    //             CHECK_IDX, NODE_IDX, (int)POS, (POS * Z) + NODE_OFFSET, APPValue);
            //                }
            //                // We don't need to maintain the values - just the sum mod 2,
            //                // but keep them for now for debugging.
            //                uint32_t hard_decision = is_neg(APPValue) ? 1 : 0;
            //                CN_bits |= (hard_decision << iVN);
            //            } // iVN

            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Early termination check
            bool parityCorrect = (0 == (__popc(CN_bits) % 2));
            //KERNEL_PRINT_IF((iIter == (maxIter - 1)) && !parityCorrect, "CHECK_IDX = %i, iter = %i, CN_bits = 0x%X, popcount(CN_bits) = %i CHECK = %s\n", CHECK_IDX, iIter, CN_bits, __popc(CN_bits), parityCorrect ? "true" : "false");
            if(!parityCorrect)
            {
                //KERNEL_PRINT_IF(iIter + 1 == maxIter, "CHECK_IDX = %i, iter = %i, CN_bits = 0x%X, popcount(CN_bits) = %i CHECK = %s\n", CHECK_IDX, iIter, CN_bits, __popc(CN_bits), parityCorrect ? "true" : "false");
                atomicAdd(&shCheckFailCount, 1);
#if 0
                // Print values associated with a parity check error
                if((iIter + 1) == maxIter)
                {
                    for(int iVN = 0; iVN < CNShift.row_degree; ++iVN)
                    {
                        const int8_t POS          = CNShift.column_values[iVN];
                        int          block_offset = NODE_OFFSET + CNShift.shift_values[iVN];
                        if(block_offset >= Z) block_offset -= Z;
                        const int    VN_idx       = (POS * Z) + block_offset;
                        if(POS <= MAX_PARITY_COL)
                        {
                            KERNEL_PRINT("CHECK_IDX = %i, NODE_IDX = %i, POS = %i, table_shift = %i, VN_idx = %i, sAPP[VN_idx] = %f\n",
                                         CHECK_IDX, NODE_IDX, (int)POS, (int)(CNShift.shift_values[iVN]), VN_idx, to_float(sAPP[VN_idx]));
                        }
                        else
                        {
                            KERNEL_PRINT("CHECK_IDX = %i, NODE_IDX = %i, POS = %i, table_shift = %i, VN_idx = %i, sAPP[VN_idx]* = %f\n",
                                         CHECK_IDX, NODE_IDX, (int)POS, (int)(CNShift.shift_values[iVN]), VN_idx, to_float(extensionAPP));
                        }

                    } // iVN
                }
#endif
            }
            else
            {
                //KERNEL_PRINT("CHECK_IDX = %i: parity correct\n", CHECK_IDX);
            }
            //
        } // if(THREAD_HAS_CHECK)
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Sync block for all block-local parity checks
        block.sync();
        //KERNEL_PRINT_BLOCK_ONCE("block %u: shCheckFailCount = %i\n", blockIdx.x, shCheckFailCount);
        // Write block count to global count
        if((shCheckFailCount > 0) && (0 == block.thread_rank()))
        {
            atomicAdd(&codeWordCheckFailCount, shCheckFailCount);
            // Reset block-local count
            shCheckFailCount = 0;
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Grid sync for parity counts from all blocks
        grid.sync();
        //__threadfence();
        if(0 == codeWordCheckFailCount)
        {
            //KERNEL_PRINT_GRID_ONCE("NO PARITY ERRORS AFTER %i ITERATIONS\n", iIter);
            break;
        }
        else
        {
            //KERNEL_PRINT_GRID_ONCE("%i PARITY ERROR(S) AFTER %i ITERATIONS\n", codeWordCheckFailCount, iIter + 1);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        //KERNEL_PRINT_GRID_ONCE("END ITER %i\n", iIter);
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
    cg::thread_block_tile<32> tile32          = cg::tiled_partition<32>(block);
    const int                 K               = Kb * Z;
    const int                 BIT_BLOCK_COUNT = (K + 1023) / 1024; // 1 bit block per warp
    const int                 WARPS_PER_BLOCK = block.size() / 32;
    const int                 WARP_IDX        = (WARPS_PER_BLOCK * blockIdx.x) + block.thread_rank() / 32;
    const int                 OUTPUT_WORDS    = (K + 31) / 32;
    //KERNEL_PRINT_BLOCK_ONCE("blockIdx.x = %u, BIT_BLOCK_COUNT = %i, WARP_IDX = %i, WARPS_PER_BLOCK = %i\n", blockIdx.x, BIT_BLOCK_COUNT, WARP_IDX, WARPS_PER_BLOCK);
    for(int iOutBlock = WARP_IDX; iOutBlock < BIT_BLOCK_COUNT; iOutBlock += (WARPS_PER_BLOCK * gridDim.x))
    {
        uint32_t threadOutput  = 0;
        int      startBitIndex = iOutBlock * 1024;
        //KERNEL_PRINT("threadIdx.x = %u, blockIdx.x = %u, iOutBlock = %i, startBitIndex = %i\n", threadIdx.x, blockIdx.x, iOutBlock, startBitIndex);
        for(int i = 0; i < 32; ++i)
        {
            int idx = startBitIndex + (i * 32) + tile32.thread_rank();
            // All blocks have a copy of sAPP in shared mem
            //KERNEL_PRINT_IF(i == 0, "threadIdx.x = %u, idx = %i, K = %i, sAPP[idx] = %f\n", threadIdx.x, idx, K, (idx < K) ? sAPP[idx] : -100.0f);
            uint32_t hardDecision = ((idx < K) && is_neg(sAPP[idx])) ? 1 : 0;
            uint32_t warpBits     = tile32.ballot(hardDecision);
            if(i == tile32.thread_rank())
            {
                threadOutput = warpBits;
            }
        }
        const int OUT_INDEX = (iOutBlock * 32) + (int)tile32.thread_rank();
        //KERNEL_PRINT("threadIdx.x = %u, index = (%i, %i), value = 0x%X\n", threadIdx.x, OUT_INDEX, CW_INDEX, threadOutput);
        if(OUT_INDEX < OUTPUT_WORDS)
        {
            tOutput({OUT_INDEX, CW_INDEX}) = threadOutput;
        }
    }
    //------------------------------------------------------------------
    // Populate the results structure for this block if the user
    // provided an address
    if((0 == threadIdx.x) && (0 == blockIdx.x) && results)
    {
        cuphyLDPCResults_t res;
        res.numIterations   = min(iIter + 1, maxIter);
        res.checkErrorCount = codeWordCheckFailCount;
        // Write to device memory
        results[CW_INDEX] = res;
    }
    //KERNEL_PRINT_GRID_ONCE("KERNEL EXIT\n");
}

template <cuphyDataType_t TLLREnum, int THREAD_COUNT>
__global__ void
    __launch_bounds__(THREAD_COUNT, 2)
        ldpc_flooding_bg1_multi_sm_precompute_no_et(int                                  iLS,
                                                    int                                  Kb,
                                                    int                                  Z,
                                                    int                                  mb,
                                                    int                                  maxIter,
                                                    LDPC_output_t                        tOutput,
                                                    const_tensor_ref_contig_2D<TLLREnum> tLLR,
                                                    void*                                workspace,
                                                    float                                norm,
                                                    cuphyLDPCResults_t*                  results)
{
    //KERNEL_PRINT_GRID_ONCE("ldpc_flooding_bg1_multi_sm_precompute_no_et()\n");
    //------------------------------------------------------------------
    // Data type corresponding to the cuPHYDataType_t value, as stored
    // in the tensor
    typedef typename const_tensor_ref_contig_2D<TLLREnum>::non_const_element_t LLR_t;
    //------------------------------------------------------------------
    // Shared memory (dynamically allocated at launch)
    lg_kernel_no_et_shared_mem_t<LLR_t> shmem(blockDim.x, Kb, Z);
    LLR_t*                              sAPP = shmem.app_addr();
    kernel_mat<LLR_t>                   sV2C(shmem.v2c_addr(), blockDim.x);
    //------------------------------------------------------------------
    const int               CW_INDEX         = blockIdx.y;
    const int               CHECK_IDX        = (blockIdx.x * blockDim.x) + threadIdx.x;
    const int               NODE_IDX         = CHECK_IDX / Z;
    const int               NODE_OFFSET      = CHECK_IDX % Z;
    const int               CHECK_COUNT      = mb * Z;
    const bool              THREAD_HAS_CHECK = (CHECK_IDX < CHECK_COUNT);
    cg::thread_block        block            = cg::this_thread_block();
    cg::grid_group          grid             = cg::this_grid();
    const int               MAX_PARITY_COL   = 25;           // TODO: Valid for BG1 only
    const int               APP_BUFFER_SIZE  = (Kb + 4) * Z; // TODO: Valid for BG1 only
    bg1_CN_row_shift_info_t CNShift(NODE_IDX, Z);
    //------------------------------------------------------------------
    // Extension nodes are used by only one check node and only one
    // variable node. We conserve shared memory by storing these values
    // in the registers of threads that are the only users of those
    // values.
    LLR_t extensionAPP = 0;
    LLR_t extensionLLR = 0;
    //------------------------------------------------------------------
    // Workspace layout:
    // parity check failure counts (integers), 1 per codeword (used for early termination)
    // array of APP_BUFFER_SIZE elements, 1 array per codeword
    LLR_t* workspaceAPPStart = reinterpret_cast<LLR_t*>(static_cast<int*>(workspace) + gridDim.y);
    LLR_t* codewordAPP       = workspaceAPPStart + (CW_INDEX * APP_BUFFER_SIZE);
    //------------------------------------------------------------------
    // Copy the initial part of the LLR data to per-block shared memory.
    // (Extension LLRs will not be located in shared memory to save
    // resources.)
    const LLR_t* channelLLR = &tLLR({0, CW_INDEX});
    // Block copy of channelLLR data to shared memory
    block_copy_sync(sAPP, channelLLR, APP_BUFFER_SIZE);
    // Grid copy of channelLLR data to workspace for APP
    grid_copy_x(codewordAPP, channelLLR, APP_BUFFER_SIZE);
    //print_array_sync("APP (init)", sAPP, APP_BUFFER_SIZE);
    //------------------------------------------------------------------
    // Initialize C2V with input LLRs
    if(THREAD_HAS_CHECK)
    {
        for(int iVN = 0; iVN < CNShift.row_degree; ++iVN)
        {
            const int8_t POS          = CNShift.column_values[iVN];
            int          block_offset = NODE_OFFSET + CNShift.shift_values[iVN];
            if(block_offset >= Z) block_offset -= Z;
            const int VN_idx       = (POS * Z) + block_offset;
            sV2C(threadIdx.x, iVN) = (POS <= MAX_PARITY_COL) ? sAPP[VN_idx] : channelLLR[VN_idx];
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
    while(iIter < maxIter)
    {
        uint32_t signBits = 0;
        LLR_t    min1 = 10000, min2 = 10000; // TODO
        int      min_idx = -1;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Check Node Processing (updates of C2V messages)
        // For each variable node (VN) associated with this check node (CN)
        if(THREAD_HAS_CHECK)
        {
            for(int iVN = 0; iVN < CNShift.row_degree; ++iVN)
            {
                LLR_t Lvc = sV2C(threadIdx.x, iVN);
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
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // We only store increments (without the base LLR) in the shared APP
        // buffer, since contributions from multiple blocks will be
        // accumulated. (We don't want to add the LLR multiple times.)
        block_zero_sync(sAPP, APP_BUFFER_SIZE);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Bit (Variable) Node Processing (updates of block-local APP
        // values)
        if(THREAD_HAS_CHECK)
        {
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
                    //KERNEL_PRINT_IF(0 == VN_idx, "CHECK_IDX = %i, adding %f to sAPP[0]\n", CHECK_IDX, Lcv);
                    atomicAdd(&sAPP[VN_idx], Lcv);
                }
                else
                {
                    extensionAPP = extensionLLR + Lcv;
                }

            } // iVN
        }     // if(THREAD_HAS_CHECK)
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Sync on updates to block-local sAPP before updating global
        block.sync();
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Grid copy of channelLLR data to workspace for APP
        grid_copy_x(codewordAPP, channelLLR, APP_BUFFER_SIZE);
        //print_array_sync("APP (init)", sAPP, APP_BUFFER_SIZE);
        //if((0 == threadIdx.x) && (0 == blockIdx.x))
        //{
        //    codeWordCheckFailCount = 0;
        //}
#if 1
        grid.sync();
#endif
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Update global memory APP with local block APP data.
        for(int iVN = threadIdx.x; iVN < APP_BUFFER_SIZE; iVN += blockDim.x)
        {
            //KERNEL_PRINT_IF(0 == iVN, "Block %u Adding %f to global workspace APP[0]\n", blockIdx.x, sAPP[iVN]);
            atomicAdd(&codewordAPP[iVN], sAPP[iVN]);
        }
#if 1
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Wait for atomic global memory updates from all blocks
        grid.sync();
#endif
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Copy data back to block-local sAPP (each block has own copy)
        block_copy_sync(sAPP, codewordAPP, APP_BUFFER_SIZE);
        //print_array_sync("APP (after atomicAdd())", sAPP, APP_BUFFER_SIZE);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        //uint32_t CN_bits = 0;
        if(THREAD_HAS_CHECK)
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
                    const int VN_idx       = (POS * Z) + block_offset;
                    LLR_t     minAbsLvc    = (iVN == min_idx) ? min2 : min1;
                    LLR_t     signProd     = (0 != (__popc(signBits & ~(1 << iVN)) & 1)) ? -1.0f : 1.0f; // TODO: get approprate constants for type
                    LLR_t     Lcv          = type_convert<LLR_t>(norm) * signProd * minAbsLvc;
                    APP                    = sAPP[VN_idx];
                    sV2C(threadIdx.x, iVN) = APP - Lcv;
                }
                else
                {
                    APP = extensionAPP;
                    // Store for next C2V loop
                    sV2C(threadIdx.x, iVN) = extensionLLR;
                }
                // We don't need to maintain the values - just the sum mod 2,
                // but keep them for now for debugging.
                //uint32_t hard_decision = is_neg(APP) ? 1 : 0;
                //CN_bits |= (hard_decision << iVN);
            } // iVN
        }     // if(THREAD_HAS_CHECK)
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        //KERNEL_PRINT_GRID_ONCE("END ITER %i\n", iIter);
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
    cg::thread_block_tile<32> tile32          = cg::tiled_partition<32>(block);
    const int                 K               = Kb * Z;
    const int                 BIT_BLOCK_COUNT = (K + 1023) / 1024; // 1 bit block per warp
    const int                 WARPS_PER_BLOCK = block.size() / 32;
    const int                 WARP_IDX        = (WARPS_PER_BLOCK * blockIdx.x) + block.thread_rank() / 32;
    const int                 OUTPUT_WORDS    = (K + 31) / 32;
    //KERNEL_PRINT_BLOCK_ONCE("blockIdx.x = %u, BIT_BLOCK_COUNT = %i, WARP_IDX = %i, WARPS_PER_BLOCK = %i\n", blockIdx.x, BIT_BLOCK_COUNT, WARP_IDX, WARPS_PER_BLOCK);
    for(int iOutBlock = WARP_IDX; iOutBlock < BIT_BLOCK_COUNT; iOutBlock += (WARPS_PER_BLOCK * gridDim.x))
    {
        uint32_t threadOutput  = 0;
        int      startBitIndex = iOutBlock * 1024;
        //KERNEL_PRINT("threadIdx.x = %u, blockIdx.x = %u, iOutBlock = %i, startBitIndex = %i\n", threadIdx.x, blockIdx.x, iOutBlock, startBitIndex);
        for(int i = 0; i < 32; ++i)
        {
            int idx = startBitIndex + (i * 32) + tile32.thread_rank();
            // All blocks have a copy of sAPP in shared mem
            //KERNEL_PRINT_IF(i == 0, "threadIdx.x = %u, idx = %i, K = %i, sAPP[idx] = %f\n", threadIdx.x, idx, K, (idx < K) ? sAPP[idx] : -100.0f);
            uint32_t hardDecision = ((idx < K) && is_neg(sAPP[idx])) ? 1 : 0;
            uint32_t warpBits     = tile32.ballot(hardDecision);
            if(i == tile32.thread_rank())
            {
                threadOutput = warpBits;
            }
        }
        const int OUT_INDEX = (iOutBlock * 32) + (int)tile32.thread_rank();
        //KERNEL_PRINT("threadIdx.x = %u, index = (%i, %i), value = 0x%X\n", threadIdx.x, OUT_INDEX, CW_INDEX, threadOutput);
        if(OUT_INDEX < OUTPUT_WORDS)
        {
            tOutput({OUT_INDEX, CW_INDEX}) = threadOutput;
        }
    }
    //------------------------------------------------------------------
    // Populate the results structure for this block if the user
    // provided an address
    if((0 == threadIdx.x) && (0 == blockIdx.x) && results)
    {
        cuphyLDPCResults_t res;
        res.numIterations   = min(iIter + 1, maxIter);
        res.checkErrorCount = 255;
        // Write to device memory
        results[CW_INDEX] = res;
    }
    //KERNEL_PRINT_GRID_ONCE("KERNEL EXIT\n");
}

#if __CUDA_ARCH__ < 700
template <>
__global__ void
ldpc_flooding_bg1_multi_sm_precompute<CUPHY_R_16F, 640>(int, int, int, int, int, LDPC_output_t, const_tensor_ref_contig_2D<CUPHY_R_16F>, void* workspace, float, cuphyLDPCResults_t*)
{
    printf("ldpc_flooding_bg1_multi_sm<> not implemented for __CUDA_ARCH__ < 700\n");
}

#endif // __CUDA_ARCH__ < 700

////////////////////////////////////////////////////////////////////////
// launch_decode_multi_sm_precompute()
template <cuphyDataType_t TType, unsigned int NTHREADS>
cuphyStatus_t launch_decode_multi_sm_precompute(LDPC_output_t&      tOutputWord,
                                                const_tensor_pair&  tLLRInput,
                                                int                 BG,
                                                int                 Z,
                                                int                 mb,
                                                int                 maxNumIterations,
                                                float               normalization,
                                                cuphyLDPCResults_t* results,
                                                void*               workspace,
                                                cudaStream_t        strm)
{
    DEBUG_PRINTF("ldpc::launch_decode_multi_sm_precompute()\n");
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
    int       Kb                     = 22;
    const int NODES_PER_THREAD_BLOCK = 2;
    const int Nthreads               = NODES_PER_THREAD_BLOCK * Z;
    int       iLS                    = set_from_Z(Z);
    const int SHMEM_SIZE             = lg_kernel_shared_mem_t<LLR_t>::get_shared_mem_size(Nthreads,
                                                                              Kb,
                                                                              Z,
                                                                              BG1_MAX_ROW_DEG);
    dim3      gridDim((mb + (NODES_PER_THREAD_BLOCK - 1)) / NODES_PER_THREAD_BLOCK,
                 tLLRInput.first.get().layout().dimensions[1]);
    dim3      blockDim(Z * NODES_PER_THREAD_BLOCK);
#if CUPHY_DEBUG
    {
        int                maxBlocks = 0;
        cudaFuncAttributes funcAttr;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks,
                                                                 ldpc_flooding_bg1_multi_sm_precompute<TType, NTHREADS>,
                                                                 blockDim.x,
                                                                 SHMEM_SIZE));
        CUDA_CHECK(cudaFuncGetAttributes(&funcAttr,
                                         ldpc_flooding_bg1_multi_sm_precompute<TType, NTHREADS>));
        DEBUG_PRINTF("NCW = %i, BG = %i, N = %i, K = %i, Kb = %i, mb = %i, Z = %i, M = %i, iLS = %i, R_trans = %.2f\n",
                     gridDim.y,
                     BG,
                     (Kb + mb) * Z,
                     Kb * Z,
                     Kb,
                     mb,
                     Z,
                     mb * Z,
                     iLS,
                     static_cast<float>(Kb) / (Kb + mb - 2));
        DEBUG_PRINTF("shmem = %i bytes, grid = (%u,%u,%u), block = (%u,%u,%u), max_blocks_per_SM = %i, max_shmem_per_block = %i, shmem_carveout_pct = %i\n",
                     SHMEM_SIZE,
                     gridDim.x,
                     gridDim.y,
                     gridDim.z,
                     blockDim.x,
                     blockDim.y,
                     blockDim.z,
                     maxBlocks,
                     funcAttr.maxDynamicSharedSizeBytes,
                     funcAttr.preferredShmemCarveout);
    }
#endif

    const_tensor_ref_contig_2D<TType> tLLR = tOptLLR.value();
    std::array<void*, 11>             args;
    args[0]       = &iLS;
    args[1]       = &Kb;
    args[2]       = &Z;
    args[3]       = &mb;
    args[4]       = &maxNumIterations;
    args[5]       = &tOutputWord;
    args[6]       = &tLLR;
    args[7]       = &workspace;
    args[8]       = &normalization;
    args[9]       = &results;
    args.back()   = nullptr;
    cudaError_t e = cudaLaunchCooperativeKernel((void*)ldpc_flooding_bg1_multi_sm_precompute<TType, NTHREADS>,
                                                gridDim,
                                                blockDim,
                                                args.data(),
                                                SHMEM_SIZE,
                                                strm);
    if(cudaSuccess != e)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

////////////////////////////////////////////////////////////////////////
// launch_decode_multi_sm_precompute()
template <cuphyDataType_t TType, unsigned int NTHREADS>
cuphyStatus_t launch_decode_multi_sm_precompute_no_et(LDPC_output_t&      tOutputWord,
                                                      const_tensor_pair&  tLLRInput,
                                                      int                 BG,
                                                      int                 Z,
                                                      int                 mb,
                                                      int                 maxNumIterations,
                                                      float               normalization,
                                                      cuphyLDPCResults_t* results,
                                                      void*               workspace,
                                                      cudaStream_t        strm)
{
    DEBUG_PRINTF("ldpc::launch_decode_multi_sm_precompute_no_et()\n");
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
    int       Kb                     = 22;
    const int NODES_PER_THREAD_BLOCK = 2;
    const int Nthreads               = NODES_PER_THREAD_BLOCK * Z;
    int       iLS                    = set_from_Z(Z);
    const int SHMEM_SIZE             = lg_kernel_no_et_shared_mem_t<LLR_t>::get_shared_mem_size(Nthreads,
                                                                                    Kb,
                                                                                    Z,
                                                                                    BG1_MAX_ROW_DEG);
    dim3      gridDim((mb + (NODES_PER_THREAD_BLOCK - 1)) / NODES_PER_THREAD_BLOCK,
                 tLLRInput.first.get().layout().dimensions[1]);
    dim3      blockDim(Z * NODES_PER_THREAD_BLOCK);
#if CUPHY_DEBUG
    {
        int                maxBlocks = 0;
        cudaFuncAttributes funcAttr;
        CUDA_CHECK(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&maxBlocks,
                                                                 ldpc_flooding_bg1_multi_sm_precompute_no_et<TType, NTHREADS>,
                                                                 blockDim.x,
                                                                 SHMEM_SIZE));
        CUDA_CHECK(cudaFuncGetAttributes(&funcAttr, ldpc_flooding_bg1_multi_sm_precompute_no_et<TType, NTHREADS>));
        DEBUG_PRINTF("NCW = %i, BG = %i, N = %i, K = %i, Kb = %i, mb = %i, Z = %i, M = %i, iLS = %i, R_trans = %.2f\n",
                     gridDim.y,
                     BG,
                     (Kb + mb) * Z,
                     Kb * Z,
                     Kb,
                     mb,
                     Z,
                     mb * Z,
                     iLS,
                     static_cast<float>(Kb) / (Kb + mb - 2));
        DEBUG_PRINTF("shmem = %i bytes, grid = (%u,%u,%u), block = (%u,%u,%u), max_blocks_per_SM = %i, max_shmem_per_block = %i, shmem_carveout_pct = %i\n",
                     SHMEM_SIZE,
                     gridDim.x,
                     gridDim.y,
                     gridDim.z,
                     blockDim.x,
                     blockDim.y,
                     blockDim.z,
                     maxBlocks,
                     funcAttr.maxDynamicSharedSizeBytes,
                     funcAttr.preferredShmemCarveout);
    }
#endif

    const_tensor_ref_contig_2D<TType> tLLR = tOptLLR.value();
    std::array<void*, 11>             args;
    args[0]       = &iLS;
    args[1]       = &Kb;
    args[2]       = &Z;
    args[3]       = &mb;
    args[4]       = &maxNumIterations;
    args[5]       = &tOutputWord;
    args[6]       = &tLLR;
    args[7]       = &workspace;
    args[8]       = &normalization;
    args[9]       = &results;
    args.back()   = nullptr;
    cudaError_t e = cudaLaunchCooperativeKernel((void*)ldpc_flooding_bg1_multi_sm_precompute_no_et<TType, NTHREADS>,
                                                gridDim,
                                                blockDim,
                                                args.data(),
                                                SHMEM_SIZE,
                                                strm);
    if(cudaSuccess != e)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

////////////////////////////////////////////////////////////////////////
// decode_multi_sm_precompute()
cuphyStatus_t decode_multi_sm_precompute(LDPC_output_t&      tDst,
                                         const_tensor_pair&  tLLR,
                                         int                 BG,
                                         int                 Z,
                                         int                 mb,
                                         int                 maxNumIterations,
                                         float               normalization,
                                         bool                earlyTermination,
                                         cuphyLDPCResults_t* results,
                                         void*               workspace,
                                         cudaStream_t        strm)
{
    DEBUG_PRINTF("ldpc::decode_multi_sm_precompute()\n");
    //------------------------------------------------------------------
    switch(tLLR.first.get().type())
    {
    case CUPHY_R_32F:
        switch(Z)
        {
        case 320:
            return launch_decode_multi_sm_precompute<CUPHY_R_32F, 640>(tDst, tLLR, BG, Z, mb, maxNumIterations, normalization, results, workspace, strm);
        case 384:
            return launch_decode_multi_sm_precompute_no_et<CUPHY_R_32F, 768>(tDst, tLLR, BG, Z, mb, maxNumIterations, normalization, results, workspace, strm);
        default:
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
    case CUPHY_R_16F:
        switch(Z)
        {
        case 320:
            if(earlyTermination)
            {
                return launch_decode_multi_sm_precompute<CUPHY_R_16F, 640>(tDst,
                                                                           tLLR,
                                                                           BG,
                                                                           Z,
                                                                           mb,
                                                                           maxNumIterations,
                                                                           normalization,
                                                                           results,
                                                                           workspace,
                                                                           strm);
            }
            else
            {
                return launch_decode_multi_sm_precompute_no_et<CUPHY_R_16F, 640>(tDst,
                                                                                 tLLR,
                                                                                 BG,
                                                                                 Z,
                                                                                 mb,
                                                                                 maxNumIterations,
                                                                                 normalization,
                                                                                 results,
                                                                                 workspace,
                                                                                 strm);
            }

        case 384:
            if(earlyTermination)
            {
                return CUPHY_STATUS_INTERNAL_ERROR;
            }
            else
            {
                return launch_decode_multi_sm_precompute_no_et<CUPHY_R_16F, 768>(tDst,
                                                                                 tLLR,
                                                                                 BG,
                                                                                 Z,
                                                                                 mb,
                                                                                 maxNumIterations,
                                                                                 normalization,
                                                                                 results,
                                                                                 workspace,
                                                                                 strm);
            }
        default:
            return CUPHY_STATUS_INTERNAL_ERROR;
        }
    default:
        return CUPHY_STATUS_UNSUPPORTED_TYPE;
    }
}
