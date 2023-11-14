/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_SPLIT_DYNAMIC_CUH_INCLUDED_)
#define LDPC2_SPLIT_DYNAMIC_CUH_INCLUDED_

#include "ldpc2_c2v_cache_split.cuh"
#include "ldpc2_schedule_dynamic.cuh"
#include "ldpc2_app_address.cuh"
#include "ldpc2_kernel.cuh"
#include <float.h>

using namespace ldpc2;

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// launch_split_dynamic()
// Launch the LDPC kernel that uses a "split" C2V message cache (part
// global and part shared memory), with ALL APP values stored in shared
// memory, and DYNAMIC (i.e. at runtime) checks for the number of parity
// nodes.
template <typename                           T,
          int                                BG,
          int                                Kb,
          int                                Z,
          int                                NUM_C2V_SMEM,
          class                              TC2V,
          template<typename, int, int> class TAPPLoc,
          int                                BLOCKS_PER_SM>
cuphyStatus_t launch_split_dynamic(ldpc::decoder&            dec,
                                   const LDPC_kernel_params& params,
                                   const dim3&               grdDim,
                                   const dim3&               blkDim,
                                   cudaStream_t              strm)
{
    //------------------------------------------------------------------
    // C2V message cache (shared memory here)
    typedef c2v_cache_split<BG, Z, NUM_C2V_SMEM, TC2V> c2v_cache_t;

    //------------------------------------------------------------------
    // APP "location" manager - calculates location of APP values for
    // threads based on base graph shift values
    typedef TAPPLoc<T, BG, Z> app_loc_t;

    //------------------------------------------------------------------
    // LDPC schedule (variable number of check nodes)
    typedef ldpc_schedule_dynamic<BG,                   // base graph
                                  app_loc_t,            // APP location/address calc
                                  c2v_cache_t> sched_t; // C2V cache
    //------------------------------------------------------------------
    // LLR loader, used to load LLR data from global to shared memory
    typedef llr_loader_variable<T, Z, ldpc2::max_variable_nodes<BG>::value> llr_loader_t;

    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory
    uint32_t shmem_size = 0;
    //uint32_t       c2v_size = ldpc2::get_c2v_shared_mem_size(NUM_C2V_SMEM, Z, sizeof(T));
    uint32_t       c2v_size = c2v_cache_t::get_c2v_size_bytes(NUM_C2V_SMEM);
    const uint32_t app_size = shmem_llr_buffer_size(params.num_var_nodes, // num shared memory nodes
                                                    Z,                    // lifting size
                                                    sizeof(T));           // element size
    shmem_size = c2v_size + app_size;
    int32_t device_shmem_max = dec.max_shmem_per_block_optin();
    if(device_shmem_max <= 0)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    if(shmem_size > device_shmem_max)
    {
        printf("c2v_size = %u, app_size = %u, shmem_size = %u, device_shmem_max = %u\n", c2v_size, app_size, shmem_size, device_shmem_max);
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    //printf("c2v_size = %u, app_size = %u, shmem_size = %u\n", c2v_size, app_size, shmem_size);
    cudaFuncSetAttribute(ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         shmem_size);
    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>));
    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>), blkDim, shmem_size);
    ldpc2_kernel<T,                // LLR data type
                 BG,               // base graph
                 Kb,               // num info nodes
                 Z,                // lifting size
                 sched_t,          // schedule type
                 llr_loader_t,     // LLR loader type
                 BLOCKS_PER_SM>    // launch bounds
                 <<<grdDim, blkDim, shmem_size, strm>>>(params);
    return CUPHY_STATUS_SUCCESS;
}

} // namespace ldpc2

#endif // !defined(LDPC2_SPLIT_DYNAMIC_CUH_INCLUDED_)
