/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_SPLIT_CUH_INCLUDED_)
#define LDPC2_SPLIT_CUH_INCLUDED_

#include "ldpc2_c2v_cache_split.cuh"
#include "ldpc2_schedule_fixed.cuh"
#include "ldpc2_app_address.cuh"
#include "ldpc2_kernel.cuh"
#include <float.h>

using namespace ldpc2;

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// max_num_c2v_shmem
// Template structure to calculate the number of compressed C2V
// structures that will reside in shared memory for a "split" (global/
// shared) compressed C2V layout.
// If we assume that all APP values will reside in shared memory (as
// access to those values is more latency sensitive), then the number
// of compressed C2V values that can reside in shared memory is given
// by:
//
// MAX_NUM_C2V_SHMEM = MAX_SHMEM - APP_SHMEM / COMPRESSED_C2V_SIZE
//
template <int MAX_SHMEM, typename T, int BG, int Z, int NUM_PARITY>
struct max_num_c2v_shmem
{
    static const int num_app_nodes  = ldpc2::max_info_nodes<BG>::value + NUM_PARITY;
    static const int num_app_values = num_app_nodes * Z;
    static const int num_app_bytes  = num_app_values * sizeof(T);
    static const int c2v_bytes      = sizeof(cC2V_storage_t<T>);
    static const int value_max      = (MAX_SHMEM - num_app_bytes) / (c2v_bytes * Z);
    static const int value          = (value_max > NUM_PARITY) ? NUM_PARITY : value_max;
    static const int shmem_size     = (value * Z * c2v_bytes) + num_app_bytes;
};

////////////////////////////////////////////////////////////////////////
// launch_split_fixed()
// Launch the LDPC kernel that uses a "split" C2V message cache (part
// global and part shared memory), with ALL APP values stored in shared
// memory, and a FIXED number of parity nodes.
template <int                                MAX_SHMEM,
          typename                           T,
          int                                BG,
          int                                Kb,
          int                                Z,
          int                                NUM_PARITY,
          class                              TC2V,
          template<typename, int, int> class TAPPLoc,
          int                                BLOCKS_PER_SM>
cuphyStatus_t launch_split_fixed(const LDPC_kernel_params& params,
                                 const dim3&               grdDim,
                                 const dim3&               blkDim,
                                 cudaStream_t              strm)
{
    typedef max_num_c2v_shmem<MAX_SHMEM, T, BG, Z, NUM_PARITY> max_num_c2v_shmem_t;
    
    constexpr int NUM_C2V_SMEM = max_num_c2v_shmem_t::value;
    constexpr int SHMEM_SIZE   = max_num_c2v_shmem_t::shmem_size;
    //------------------------------------------------------------------
    // C2V message cache (shared memory here)
    typedef c2v_cache_split<BG, Z, NUM_C2V_SMEM, TC2V> c2v_cache_t;

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

    //printf("NUM_C2V_SMEM = %i, c2v_size = %u, app_size = %u, shmem_size = %u\n",
    //       NUM_C2V_SMEM,
    //       max_num_c2v_shmem_t::c2v_bytes * NUM_C2V_SMEM * Z,
    //       max_num_c2v_shmem_t::num_app_bytes,
    //       SHMEM_SIZE);
    cudaError_t e = cudaFuncSetAttribute(ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SHMEM_SIZE);
    if(e != cudaSuccess)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>));
    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>), blkDim, SHMEM_SIZE);
    ldpc2_kernel<T,                // LLR data type
                 BG,               // base graph
                 Kb,               // num info nodes
                 Z,                // lifting size
                 sched_t,          // schedule type
                 llr_loader_t,     // LLR loader type
                 BLOCKS_PER_SM>    // launch bounds
                 <<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params);
    return CUPHY_STATUS_SUCCESS;
}

} // namespace ldpc2

#endif // !defined(LDPC2_SPLIT_CUH_INCLUDED_)
