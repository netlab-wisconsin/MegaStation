/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_REG_CUH_INCLUDED_)
#define LDPC2_REG_CUH_INCLUDED_

#include "ldpc2_c2v_cache_register.cuh"
#include "ldpc2_schedule_fixed.cuh"
#include "ldpc2_schedule_cluster.cuh"
#include "ldpc2_app_address.cuh"
#include "ldpc2_kernel.cuh"

namespace
{

template <typename                           T,
          int                                BG,
          int                                Kb,
          int                                Z,
          int                                NUM_PARITY,
          class                              TC2V,
          template<typename, int, int> class TAPPLoc,
          int                                BLOCKS_PER_SM>
cuphyStatus_t launch_register_kernel(ldpc::decoder&                   dec,
                                     const ldpc2::LDPC_kernel_params& params,
                                     const dim3&                      grdDim,
                                     const dim3&                      blkDim,
                                     cudaStream_t                     strm)
{
    // C2V message cache (register memory here)
    typedef ldpc2::c2v_cache_register<BG, NUM_PARITY, TC2V, ldpc2::LDPC_kernel_params> c2v_cache_t;

    //------------------------------------------------------------------
    // APP "location" manager - calculates location of APP values for
    // threads based on base graph shift values
    typedef TAPPLoc<T, BG, Z> app_loc_t;

    // Note: NUM_SMEM_CHECK_NODES is the number of check nodes for which
    // the corresponding APP data can be found in shared memory. When
    // using the register C2V cache, we assume here that the number of
    // check nodes is relatively small, and that APP data for ALL check
    // nodes is found in shared memory.
    typedef ldpc2::ldpc_schedule_fixed<BG,                   // base graph
                                       NUM_PARITY,           // NUM_CHECK_NODES
                                       app_loc_t,            // APP location/address calc
                                       c2v_cache_t> sched_t; // C2V cache

    // LLR loader, used to load LLR data from global to shared memory
    typedef ldpc2::llr_loader_fixed<T, Z, ldpc2::max_info_nodes<BG>::value + NUM_PARITY> llr_loader_t;
    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory
    const uint32_t SHMEM_SIZE = ldpc2::shmem_llr_buffer_size(ldpc2::max_info_nodes<BG>::value + NUM_PARITY, // num shared memory nodes
                                                             Z,                                             // lifting size
                                                             sizeof(T));                                    // element size
    cudaError_t e = cudaFuncSetAttribute(ldpc2::ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SHMEM_SIZE);
    if(cudaSuccess != e)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    //------------------------------------------------------------------
    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc2::ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>));
    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc2::ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>), blkDim, SHMEM_SIZE);
    ldpc2::ldpc2_kernel<T,               // LLR data type
                        BG,              // base graph
                        Kb,              // num info nodes
                        Z,               // lifting size
                        sched_t,         // schedule type
                        llr_loader_t,    // LLR loader type
                        BLOCKS_PER_SM>   // launch bounds
                        <<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params);
    return CUPHY_STATUS_SUCCESS;
}

template <typename                           T,
          int                                BG,
          int                                Kb,
          int                                Z,
          int                                NUM_PARITY,
          class                              TC2V,
          template<typename, int, int> class TAPPLoc,
          int                                BLOCKS_PER_SM>
cuphyStatus_t launch_register_kernel_cluster(ldpc::decoder&                   dec,
                                             const ldpc2::LDPC_kernel_params& params,
                                             const dim3&                      grdDim,
                                             const dim3&                      blkDim,
                                             cudaStream_t                     strm)
{
    // C2V message cache (register memory here)
    typedef ldpc2::c2v_cache_register<BG, NUM_PARITY, TC2V, ldpc2::LDPC_kernel_params> c2v_cache_t;

    //------------------------------------------------------------------
    // APP "location" manager - calculates location of APP values for
    // threads based on base graph shift values
    typedef TAPPLoc<T, BG, Z> app_loc_t;

    typedef ldpc2::ldpc_schedule_cluster<BG,                  // base graph
                                         app_loc_t,           // APP location/address calc
                                         c2v_cache_t,         // C2V cache
                                         NUM_PARITY> sched_t; // Number of parity nodes
                                
    // LLR loader, used to load LLR data from global to shared memory
    typedef ldpc2::llr_loader_fixed<T, Z, ldpc2::max_info_nodes<BG>::value + NUM_PARITY> llr_loader_t;
    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory
    const uint32_t SHMEM_SIZE = ldpc2::shmem_llr_buffer_size(ldpc2::max_info_nodes<BG>::value + NUM_PARITY, // num shared memory nodes
                                                             Z,                                             // lifting size
                                                             sizeof(T));                                    // element size
    cudaError_t e = cudaFuncSetAttribute(ldpc2::ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SHMEM_SIZE);
    if(cudaSuccess != e)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    //------------------------------------------------------------------
    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc2::ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>));
    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc2::ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>), blkDim, SHMEM_SIZE);
    ldpc2::ldpc2_kernel<T,               // LLR data type
                        BG,              // base graph
                        Kb,              // num info nodes
                        Z,               // lifting size
                        sched_t,         // schedule type
                        llr_loader_t,    // LLR loader type
                        BLOCKS_PER_SM>   // launch bounds
                        <<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params);
    return CUPHY_STATUS_SUCCESS;
}

} // namespace

#endif // !defined(LDPC2_SHARED_CUH_INCLUDED_)
