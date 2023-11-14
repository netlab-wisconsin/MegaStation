/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_SHARED_CUH_INCLUDED_)
#define LDPC2_SHARED_CUH_INCLUDED_

//#define CUPHY_DEBUG 1

#include "ldpc2_c2v_cache_shared.cuh"
#include "ldpc2_c2v_cache_shared_strided.cuh"
#include "ldpc2_schedule_fixed.cuh"
#include "ldpc2_app_address.cuh"
#include "ldpc2_kernel.cuh"
#include <float.h>
#include "device.hpp"

using namespace ldpc2;

namespace
{

////////////////////////////////////////////////////////////////////////
// launch_all_shared()
// Launch the LDPC kernel that uses an all-shared memory C2V message cache,
// with ALL APP values stored in shared memory.
template <typename                           T,
          int                                BG,
          int                                Kb,
          int                                Z,
          int                                NUM_PARITY,
          class                              TC2V,
          template<typename, int, int> class TAPPLoc,
          int                                BLOCKS_PER_SM>
cuphyStatus_t launch_all_shared(ldpc::decoder&            dec,
                                const LDPC_kernel_params& params,
                                const dim3&               grdDim,
                                const dim3&               blkDim,
                                cudaStream_t              strm)
{
    //------------------------------------------------------------------
    // C2V message cache (shared memory here)
    typedef c2v_cache_shared<BG, Z, max_info_nodes<BG>::value + NUM_PARITY, TC2V> c2v_cache_t;

    //------------------------------------------------------------------
    // APP "location" manager - calculates location of APP values for
    // threads based on base graph shift values
    typedef TAPPLoc<T, BG, Z> app_loc_t;

    //------------------------------------------------------------------
    // LDPC schedule (fixed number of check nodes)
    typedef ldpc_schedule_fixed<BG,                   // base graph
                                NUM_PARITY,           // NUM_CHECK_NODES
                                app_loc_t,            // APP location/address calc
                                c2v_cache_t> sched_t; // C2V cache
    //------------------------------------------------------------------
    // LLR loader, used to load LLR data from global to shared memory
    typedef llr_loader_fixed<T, Z, max_info_nodes<BG>::value + NUM_PARITY> llr_loader_t;

    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory
    uint32_t shmem_size = 0;
    // Assuming that this will only be used when we can fit ALL
    // data in shared memory. A "split" shared/global approach will
    // be in a separate implementation file.
    //uint32_t       c2v_size = ldpc2::get_c2v_shared_mem_size(NUM_PARITY, Z, sizeof(T));
    uint32_t       c2v_size = c2v_cache_t::get_c2v_size_bytes(NUM_PARITY);
    const uint32_t app_size = shmem_llr_buffer_size(max_info_nodes<BG>::value + NUM_PARITY, // num shared memory nodes
                                                    Z,                                      // lifting size
                                                    sizeof(T));                             // element size
    shmem_size = c2v_size + app_size;
    int32_t device_shmem_max = dec.max_shmem_per_block_optin();
    if(device_shmem_max <= 0)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    if(shmem_size > device_shmem_max)
    {
        //printf("shmem_size: %i    device_max: %i\n", shmem_size, device_shmem_max);
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    //printf("c2v_size = %u, app_size = %u, shmem_size = %u\n", c2v_size, app_size, shmem_size);
    cudaError_t e = cudaFuncSetAttribute(ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shmem_size);
    if(e != cudaSuccess)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    
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

////////////////////////////////////////////////////////////////////////
// launch_all_shared_strided()
// Launch the LDPC kernel that uses an all-shared memory C2V message
// cache with ALL APP values stored in shared memory, and using
// "strided" shared memory storage to avoid bank conflicts.
template <typename                           T,
          int                                BG,
          int                                Kb,
          int                                Z,
          int                                NUM_PARITY,
          class                              TC2V,
          template<typename, int, int> class TAPPLoc,
          int                                BLOCKS_PER_SM>
cuphyStatus_t launch_all_shared_strided(ldpc::decoder&            dec,
                                        const LDPC_kernel_params& params,
                                        const dim3&               grdDim,
                                        const dim3&               blkDim,
                                        cudaStream_t              strm)
{
    //------------------------------------------------------------------
    // C2V message cache (shared memory here)
    typedef c2v_cache_shared_strided<BG, Z, max_info_nodes<BG>::value + NUM_PARITY, TC2V> c2v_cache_t;

    //------------------------------------------------------------------
    // APP "location" manager - calculates location of APP values for
    // threads based on base graph shift values
    typedef TAPPLoc<T, BG, Z> app_loc_t;

    //------------------------------------------------------------------
    // LDPC schedule (fixed number of check nodes)
    typedef ldpc_schedule_fixed<BG,                   // base graph
                                NUM_PARITY,           // NUM_CHECK_NODES
                                app_loc_t,            // APP location/address calc
                                c2v_cache_t> sched_t; // C2V cache
    //------------------------------------------------------------------
    // LLR loader, used to load LLR data from global to shared memory
    typedef llr_loader_fixed<T, Z, max_info_nodes<BG>::value + NUM_PARITY> llr_loader_t;

    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory
    uint32_t shmem_size = 0;
    // Assuming that this will only be used when we can fit ALL
    // data in shared memory. A "split" shared/global approach will
    // be in a separate implementation file.
    // TODO: Make the calculation more accurate here.
    //uint32_t       c2v_size = ldpc2::get_c2v_shared_mem_size(NUM_PARITY, Z, sizeof(T));
    uint32_t       c2v_size = c2v_cache_t::get_c2v_size_bytes(NUM_PARITY);
    const uint32_t app_size = shmem_llr_buffer_size(max_info_nodes<BG>::value + NUM_PARITY, // num shared memory nodes
                                                    Z,                                      // lifting size
                                                    sizeof(T));                             // element size
    shmem_size = c2v_size + app_size;
    int32_t device_shmem_max = dec.max_shmem_per_block_optin();
    if(device_shmem_max <= 0)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    if(shmem_size > device_shmem_max)
    {
        //printf("shmem_size: %i    device_max: %i\n", shmem_size, device_shmem_max);
        return CUPHY_STATUS_UNSUPPORTED_CONFIG;
    }
    //printf("c2v_size = %u, app_size = %u, shmem_size = %u\n", c2v_size, app_size, shmem_size);
    cudaError_t e = cudaFuncSetAttribute(ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         shmem_size);
    if(e != cudaSuccess)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    
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

} // namespace

#endif // !defined(LDPC2_SHARED_CUH_INCLUDED_)
