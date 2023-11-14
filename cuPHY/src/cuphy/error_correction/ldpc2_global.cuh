/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_GLOBAL_CUH_INCLUDED_)
#define LDPC2_GLOBAL_CUH_INCLUDED_

#include "ldpc2_c2v_cache_global.cuh"
#include "ldpc2_schedule_fixed.cuh"
#include "ldpc2_schedule_dynamic.cuh"
#include "ldpc2_app_address.cuh"
#include "ldpc2_kernel.cuh"

using namespace ldpc2;

namespace
{
////////////////////////////////////////////////////////////////////////
// launch_glob_all_shared()
// Launch the LDPC kernel that uses a global memory C2V message cache,
// with ALL APP values stored in shared memory
template <typename                           T,
          int                                BG,
          int                                Kb,
          int                                Z,
          int                                NUM_PARITY,
          class                              TC2V,
          template<typename, int, int> class TAPPLoc,
          int                                BLOCKS_PER_SM>
cuphyStatus_t launch_glob_all_shared(ldpc::decoder&            dec,
                                     const LDPC_kernel_params& params,
                                     const dim3&               grdDim,
                                     const dim3&               blkDim,
                                     cudaStream_t              strm)
{
    //------------------------------------------------------------------
    // C2V message cache (global memory here), with all APP values in
    // shared memory
    typedef c2v_cache_global<BG, NUM_PARITY, TC2V> c2v_cache_t;

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
    const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(max_info_nodes<BG>::value + NUM_PARITY, // num shared memory nodes
                                                      Z,                                      // lifting size
                                                      sizeof(T));                             // element size
    cudaError_t e = cudaFuncSetAttribute(ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SHMEM_SIZE);
    if(cudaSuccess != e)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    //------------------------------------------------------------------
    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>));
    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>), blkDim, SHMEM_SIZE);
    ldpc2_kernel<T,                // LLR data type
                 BG,               // base graph
                 Kb,               // num info nodes
                 Z,                // lifting size
                 sched_t,          // schedule type
                 llr_loader_t,     // LLR loader
                 BLOCKS_PER_SM>    // launch bounds
                 <<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params);
#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

////////////////////////////////////////////////////////////////////////
// launch_glob_all_shared_dynamic()
// Launch the LDPC kernel that uses a global memory C2V message cache,
// with ALL APP values stored in shared memory, and a "dynamic" parity
// node schedule that handles multiple numbers of parity nodes and
// checks after each node.
template <typename                           T,
          int                                BG,
          int                                Kb,
          int                                Z,
          class                              TC2V,
          template<typename, int, int> class TAPPLoc,
          int                                BLOCKS_PER_SM>
cuphyStatus_t launch_glob_all_shared_dynamic(ldpc::decoder&            dec,
                                             const LDPC_kernel_params& params,
                                             const dim3&               grdDim,
                                             const dim3&               blkDim,
                                             cudaStream_t              strm)
{
    //------------------------------------------------------------------
    // C2V message cache (global memory here). Assuming all APP values
    // are stored in shared memory, so we use the maximum number of
    // parity nodes for the c2v_cache_global parameter. (This is OK,
    // even if the actual number of parity nodes is fewer.)
    typedef c2v_cache_global<BG, max_parity_nodes<BG>::value, TC2V> c2v_cache_t;

    //------------------------------------------------------------------
    // Input LLR loader (variable here because the number of nodes is
    // unknown at compile time)
    typedef llr_loader_variable<T, Z, max_variable_nodes<BG>::value> llr_loader_t;

    //------------------------------------------------------------------
    // APP "location" manager - calculates location of APP values for
    // threads based on base graph shift values
    typedef TAPPLoc<T, BG, Z> app_loc_t;

    //------------------------------------------------------------------
    // LDPC schedule (kernel handles different numbers of check nodes)
    typedef ldpc_schedule_dynamic<BG,                                        // base graph
                                  app_loc_t,                                 // APP location/address calc
                                  c2v_cache_t> sched_t;                      // C2V cache
    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory
    const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(max_info_nodes<BG>::value + params.num_parity_nodes, // num shared memory nodes
                                                      Z,                                                   // lifting size
                                                      sizeof(T));                                          // element size
    cudaError_t e = cudaFuncSetAttribute(ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SHMEM_SIZE);
    if(cudaSuccess != e)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    //------------------------------------------------------------------
    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>));
    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>), blkDim, SHMEM_SIZE);
    ldpc2_kernel<T,                // LLR data type
                 BG,               // base graph
                 Kb,               // num info nodes
                 Z,                // lifting size
                 sched_t,          // schedule type
                 llr_loader_t,     // input loader type
                 BLOCKS_PER_SM>    // launch bounds
                 <<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params);
#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

////////////////////////////////////////////////////////////////////////
// launch_glob_part_shared()
// Launch the LDPC kernel that uses a global memory C2V message cache,
// with only SOME of the APP values stored in shared memory. (For the
// lowest code rates, it may not be possible or desirable to store all
// APP values in shared memory, depending on the LLR type size, the
// target architecture, and the shared memory carveout.
template <typename                           T,
          int                                BG,
          int                                Kb,
          int                                Z,
          int                                NUM_PARITY,
          int                                NUM_SHMEM_APP,
          class                              TC2V,
          template<typename, int, int> class TAPPLoc,
          int                                BLOCKS_PER_SM>
cuphyStatus_t launch_glob_part_shared(ldpc::decoder&            dec,
                                      const LDPC_kernel_params& params,
                                      const dim3&               grdDim,
                                      const dim3&               blkDim,
                                      cudaStream_t              strm)
{
    //------------------------------------------------------------------
    // C2V message cache (global memory here)
    typedef c2v_cache_global<BG,
                             NUM_SHMEM_APP - max_info_nodes<BG>::value, // Num check nodes with APP in shared
                             TC2V> c2v_cache_t;

    //------------------------------------------------------------------
    // APP "location" manager - calculates location of APP values for
    // threads based on base graph shift values
    typedef TAPPLoc<T, BG, Z> app_loc_t;

    //------------------------------------------------------------------
    // LDPC schedule (fixed number of check nodes)
    typedef ldpc_schedule_fixed<BG,                                        // base graph
                                NUM_PARITY,                                // NUM_CHECK_NODES
                                app_loc_t,                                 // APP location/address calc
                                c2v_cache_t> sched_t;                      // C2V cache

    //------------------------------------------------------------------
    // LLR loader, used to load LLR data from global to shared memory
    typedef llr_loader_fixed<T, Z, NUM_SHMEM_APP> llr_loader_t;

    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory
    const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(NUM_SHMEM_APP, // num shared memory nodes
                                                      Z,             // lifting size
                                                      sizeof(T));    // element size
    cudaError_t e = cudaFuncSetAttribute(ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SHMEM_SIZE);
    if(cudaSuccess != e)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    //------------------------------------------------------------------
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
#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

////////////////////////////////////////////////////////////////////////
// launch_glob_part_shared_dynamic()
// Launch the LDPC kernel that uses a global memory C2V message cache,
// with only SOME of the APP values stored in shared memory, and a
// dynamic check node schedule. (For the lowest code rates, it may not
// be possible or desirable to store all APP values in shared memory,
// depending on the LLR type size, the target architecture, and the
// shared memory carveout.
// The dynamic schedule checks the number of parity nodes after each
// row, and allows a single kernel to handle multiple code rates.
template <typename                           T,
          int                                BG,
          int                                Kb,
          int                                Z,
          int                                NUM_SHMEM_APP,
          class                              TC2V,
          template<typename, int, int> class TAPPLoc,
          int                                BLOCKS_PER_SM>
cuphyStatus_t launch_glob_part_shared_dynamic(ldpc::decoder&            dec,
                                              const LDPC_kernel_params& params,
                                              const dim3&               grdDim,
                                              const dim3&               blkDim,
                                              cudaStream_t              strm)
{
    //------------------------------------------------------------------
    // C2V message cache (global memory here)
    typedef c2v_cache_global<BG,
                             NUM_SHMEM_APP - max_info_nodes<BG>::value, // Num check nodes with APP in shared
                             TC2V> c2v_cache_t;

    //------------------------------------------------------------------
    // APP "location" manager - calculates location of APP values for
    // threads based on base graph shift values
    typedef TAPPLoc<T, BG, Z> app_loc_t;

    //------------------------------------------------------------------
    // LDPC schedule (kernel handles different numbers of check nodes)
    typedef ldpc_schedule_dynamic<BG,                                        // base graph
                                  app_loc_t,                                 // APP location/address calc
                                  c2v_cache_t> sched_t;                      // C2V cache
    //------------------------------------------------------------------
    // LLR loader, used to load LLR data from global to shared memory
    typedef llr_loader_fixed<T, Z, NUM_SHMEM_APP> llr_loader_t;
    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory
    const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(NUM_SHMEM_APP, // num shared memory nodes
                                                      Z,             // lifting size
                                                      sizeof(T));    // element size
    cudaError_t e = cudaFuncSetAttribute(ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SHMEM_SIZE);
    if(e != cudaSuccess)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    //------------------------------------------------------------------
    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>));
    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc2_kernel<T, BG, Kb, Z, sched_t, llr_loader_t, BLOCKS_PER_SM>), blkDim, SHMEM_SIZE);
    ldpc2_kernel<T,             // LLR data type
                 BG,            // base graph
                 Kb,            // num info nodes
                 Z,             // lifting size
                 sched_t,       // schedule type
                 llr_loader_t,  // LLR loader type
                 BLOCKS_PER_SM> // launch bounds
                 <<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params);
#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

} // namespace

#endif // !defined(LDPC2_SHARED_CUH_INCLUDED_)
