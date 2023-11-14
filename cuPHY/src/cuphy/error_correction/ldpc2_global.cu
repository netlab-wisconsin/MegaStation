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

#include "ldpc2.hpp"
#include "ldpc2.cuh"
#include <float.h>

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
          template<typename, int, int> class TAPPLoc>
cuphyStatus_t launch_glob_all_shared(const LDPC_kernel_params& params,
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
    // LDPC iterator (fixed number of check nodes)
    typedef ldpc_iterator_fixed<BG,                   // base graph
                                NUM_PARITY,           // NUM_CHECK_NODES
                                app_loc_t,            // APP location/address calc
                                c2v_cache_t> iter_t;  // C2V cache

    //------------------------------------------------------------------
    // LLR loader, used to load LLR data from global to shared memory
    typedef llr_loader_fixed<T, Z, Kb + NUM_PARITY> llr_loader_t;

    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory
    const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(Kb + NUM_PARITY, // num shared memory nodes
                                                      Z,               // lifting size
                                                      sizeof(T));      // element size
    cudaError_t e = cudaFuncSetAttribute(ldpc2_kernel<T, BG, Kb, Z, iter_t, llr_loader_t>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SHMEM_SIZE);
    if(cudaSuccess != e)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    //------------------------------------------------------------------
    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc2_kernel<T, BG, Kb, Z, iter_t, llr_loader_t>));
    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc2_kernel<T, BG, Kb, Z, iter_t, llr_loader_t>), blkDim, SHMEM_SIZE);
    ldpc2_kernel<T,                // LLR data type
                 BG,               // base graph
                 Kb,               // num info nodes
                 Z,                // lifting size
                 iter_t,           // iterator type
                 llr_loader_t>     // LLR loader
                 <<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params);
    return CUPHY_STATUS_SUCCESS;
}

#if 0
////////////////////////////////////////////////////////////////////////
// launch_glob_all_shared_dynamic()
// Launch the LDPC kernel that uses a global memory C2V message cache,
// with ALL APP values stored in shared memory, and a "dynamic" parity
// node iterator that handles multiple numbers of parity nodes and
// checks after each node.
template <typename                           T,
          int                                BG,
          int                                Kb,
          int                                Z,
          int                                NUM_PARITY,
          class                              TC2V,
          template<typename, int, int> class TAPPLoc>
void launch_glob_all_shared_dynamic(const LDPC_kernel_params& params,
                                    const dim3&               grdDim,
                                    const dim3&               blkDim,
                                    cudaStream_t              strm)
{
    //------------------------------------------------------------------
    // C2V message cache (global memory here). Assuming all APP values
    // are stored in shared memory.
    typedef c2v_cache_global<BG, NUM_PARITY, TC2V> c2v_cache_t;

    //------------------------------------------------------------------
    // APP "location" manager - calculates location of APP values for
    // threads based on base graph shift values
    typedef TAPPLoc<T, BG, Z> app_loc_t;

    //------------------------------------------------------------------
    // LDPC iterator (kernel handles different numbers of check nodes)
    typedef ldpc_iterator_dynamic<BG,                                        // base graph
                                  NUM_SHMEM_APP - max_info_nodes<BG>::value, // NUM_SMEM_CHECK_NODES
                                  app_loc_t,                                 // APP location/address calc
                                  c2v_cache_t> iter_t;                       // C2V cache
    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory
    const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(Kb + NUM_PARITY, // num shared memory nodes
                                                      Z,               // lifting size
                                                      sizeof(T));      // element size
    cudaFuncSetAttribute(ldpc2_kernel<T, BG, Kb, Z, iter_t>,
                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                         SHMEM_SIZE);
    //------------------------------------------------------------------
    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc2_kernel<T, BG, Kb, Z, iter_t>));
    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc2_kernel<T, BG, Kb, Z, iter_t>), blkDim, SHMEM_SIZE);
    ldpc2_kernel<T,                // LLR data type
                 BG,               // base graph
                 Kb,               // num info nodes
                 Z,                // lifting size
                 iter_t>           // iterator type
                 <<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params);
}
#endif

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
          template<typename, int, int> class TAPPLoc>
cuphyStatus_t launch_glob_part_shared(const LDPC_kernel_params& params,
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
    // LDPC iterator (fixed number of check nodes)
    typedef ldpc_iterator_fixed<BG,                                        // base graph
                                NUM_PARITY,                                // NUM_CHECK_NODES
                                app_loc_t,                                 // APP location/address calc
                                c2v_cache_t> iter_t;                       // C2V cache

    //------------------------------------------------------------------
    // LLR loader, used to load LLR data from global to shared memory
    typedef llr_loader_fixed<T, Z, NUM_SHMEM_APP> llr_loader_t;

    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory
    const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(NUM_SHMEM_APP, // num shared memory nodes
                                                      Z,             // lifting size
                                                      sizeof(T));    // element size
    cudaError_t e = cudaFuncSetAttribute(ldpc2_kernel<T, BG, Kb, Z, iter_t, llr_loader_t>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SHMEM_SIZE);
    if(cudaSuccess != e)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    //------------------------------------------------------------------
    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc2_kernel<T, BG, Kb, Z, iter_t, llr_loader_t>));
    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc2_kernel<T, BG, Kb, Z, iter_t, llr_loader_t>), blkDim, SHMEM_SIZE);
    ldpc2_kernel<T,                // LLR data type
                 BG,               // base graph
                 Kb,               // num info nodes
                 Z,                // lifting size
                 iter_t,           // iterator type
                 llr_loader_t>     // LLR loader type
                 <<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params);
    return CUPHY_STATUS_SUCCESS;
}

////////////////////////////////////////////////////////////////////////
// launch_glob_part_shared_dynamic()
// Launch the LDPC kernel that uses a global memory C2V message cache,
// with only SOME of the APP values stored in shared memory, and a
// dynamic check node iterator. (For the lowest code rates, it may not
// be possible or desirable to store all APP values in shared memory,
// depending on the LLR type size, the target architecture, and the
// shared memory carveout.
// The dynamic iterator checks the number of parity nodes after each
// iteration, and allows a single kernel to handle multiple code rates.
template <typename                           T,
          int                                BG,
          int                                Kb,
          int                                Z,
          int                                NUM_SHMEM_APP,
          class                              TC2V,
          template<typename, int, int> class TAPPLoc>
cuphyStatus_t launch_glob_part_shared_dynamic(const LDPC_kernel_params& params,
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
    // LDPC iterator (kernel handles different numbers of check nodes)
    typedef ldpc_iterator_dynamic<BG,                                        // base graph
                                  app_loc_t,                                 // APP location/address calc
                                  c2v_cache_t> iter_t;                       // C2V cache
    //------------------------------------------------------------------
    // LLR loader, used to load LLR data from global to shared memory
    typedef llr_loader_fixed<T, Z, NUM_SHMEM_APP> llr_loader_t;
    //------------------------------------------------------------------
    // Determine the dynamic amount of shared memory
    const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(NUM_SHMEM_APP, // num shared memory nodes
                                                      Z,             // lifting size
                                                      sizeof(T));    // element size
    cudaError_t e = cudaFuncSetAttribute(ldpc2_kernel<T, BG, Kb, Z, iter_t, llr_loader_t>,
                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
                                         SHMEM_SIZE);
    if(e != cudaSuccess)
    {
        return CUPHY_STATUS_INTERNAL_ERROR;
    }
    //------------------------------------------------------------------
    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc2_kernel<T, BG, Kb, Z, iter_t, llr_loader_t>));
    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc2_kernel<T, BG, Kb, Z, iter_t, llr_loader_t>), blkDim, SHMEM_SIZE);
    ldpc2_kernel<T,             // LLR data type
                 BG,            // base graph
                 Kb,            // num info nodes
                 Z,             // lifting size
                 iter_t,        // iterator type
                 llr_loader_t>  // LLR loader type
                 <<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params);
    return CUPHY_STATUS_SUCCESS;
}

} // namespace

namespace ldpc
{
////////////////////////////////////////////////////////////////////////
// decode_ldpc2()
cuphyStatus_t decode_ldpc2_global_address(LDPC_output_t&         tDst,
                                          const_tensor_pair&     tLLR,
                                          const LDPC_config&     config,
                                          float                  normalization,
                                          cuphyLDPCResults_t*    results,
                                          void*                  workspace,
                                          cuphyLDPCDiagnostic_t* diag,
                                          cudaStream_t           strm)
{
    DEBUG_PRINTF("ldpc::decode_ldpc2_global()\n");
    //------------------------------------------------------------------
    cuphyDataType_t llrType = tLLR.first.get().type();
    // TODO: Examine expanding to other configurations. Unsupported
    // numbers of parity nodes will be handled below.
    if(llrType != CUPHY_R_32F)
    {
        return CUPHY_STATUS_NOT_SUPPORTED;
    }
    //------------------------------------------------------------------
    dim3 grdDim(config.num_codewords); // 1-D, NUM_CW for fp32, NUM_CW / 2 for fp16
    dim3 blkDim(config.Z);

    //------------------------------------------------------------------
    // Initialize the kernel params struct
    LDPC_kernel_params params(config, tLLR, tDst, normalization, workspace);

    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;

    if(config.BG == 1)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - -
        /// Check to Variable (C2V) message type
        // USE THIS for min1 delta
        typedef cC2V_address<float, 1, sign_store_policy_dst, min1_policy_delta, sign_mgr_be> cC2V_t;
    
        //typedef cC2V_address<float, sign_store_policy_src, min1_policy_delta, sign_mgr_be> cC2V_t;
        //typedef cC2V_address<float, sign_store_policy_src, min1_policy_delta, sign_mgr_le> cC2V_t;
        //typedef cC2V_address<float, sign_store_policy_dst, min1_policy_delta, sign_mgr_le> cC2V_t;
    
        // USE THIS for min1 default
        //typedef cC2V_address<float, sign_store_policy_dst, min1_policy_default, sign_mgr_be> cC2V_t;
    
        //typedef cC2V_address<float, sign_store_policy_src, min1_policy_default, sign_mgr_be> cC2V_t;
        switch(config.Z)
        {
        case 256:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 1, 22, 256, 4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 1, 22, 256, 5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 1, 22, 256, 6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 1, 22, 256, 7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 1, 22, 256, 8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:                                                                                                          break;
            }
            break;
        case 288:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 1, 22, 288, 4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 1, 22, 288, 5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 1, 22, 288, 6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 1, 22, 288, 7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 1, 22, 288, 8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:                                                                                                          break;
            }
            break;
        case 320:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 1, 22, 320, 4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 1, 22, 320, 5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 1, 22, 320, 6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 1, 22, 320, 7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 1, 22, 320, 8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:                                                                                                          break;
            }
            break;
        case 352:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 1, 22, 352, 4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 1, 22, 352, 5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 1, 22, 352, 6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 1, 22, 352, 7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 1, 22, 352, 8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:                                                                                                          break;
            }
            break;
        case 384:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 1, 22, 384, 4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 1, 22, 384, 5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 1, 22, 384, 6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 1, 22, 384, 7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 1, 22, 384, 8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:                                                                                                          break;
            }
            break;
        default:
            return CUPHY_STATUS_NOT_SUPPORTED;
        }
    }
    else
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - -
        /// Check to Variable (C2V) message type
        // USE THIS for min1 delta
        typedef cC2V_address<float, 2, sign_store_policy_dst, min1_policy_delta, sign_mgr_be> cC2V_t;
    
        //typedef cC2V_address<float, sign_store_policy_src, min1_policy_delta, sign_mgr_be> cC2V_t;
        //typedef cC2V_address<float, sign_store_policy_src, min1_policy_delta, sign_mgr_le> cC2V_t;
        //typedef cC2V_address<float, sign_store_policy_dst, min1_policy_delta, sign_mgr_le> cC2V_t;
    
        // USE THIS for min1 default
        //typedef cC2V_address<float, sign_store_policy_dst, min1_policy_default, sign_mgr_be> cC2V_t;
    
        //typedef cC2V_address<float, sign_store_policy_src, min1_policy_default, sign_mgr_be> cC2V_t;
        switch(config.Z)
        {
        case 256:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 2, 10, 256, 4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 2, 10, 256, 5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 2, 10, 256, 6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 2, 10, 256, 7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 2, 10, 256, 8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:                                                                                                          break;
            }
            break;
        case 288:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 2, 10, 288, 4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 2, 10, 288, 5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 2, 10, 288, 6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 2, 10, 288, 7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 2, 10, 288, 8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:                                                                                                          break;
            }
            break;
        case 320:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 2, 10, 320, 4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 2, 10, 320, 5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 2, 10, 320, 6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 2, 10, 320, 7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 2, 10, 320, 8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:                                                                                                          break;
            }
            break;
        case 352:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 2, 10, 352, 4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 2, 10, 352, 5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 2, 10, 352, 6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 2, 10, 352, 7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 2, 10, 352, 8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:                                                                                                          break;
            }
            break;
        case 384:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 2, 10, 384, 4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 2, 10, 384, 5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 2, 10, 384, 6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 2, 10, 384, 7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 2, 10, 384, 8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:                                                                                                          break;
            }
            break;
        default:
            break;
        }
    }
    if(CUPHY_STATUS_SUCCESS != s)
    {
        return s;
    }

#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_global_index_half()
cuphyStatus_t decode_ldpc2_global_index_half(const LDPC_config&        config,
                                             const LDPC_kernel_params& params,
                                             const dim3&               grdDim,
                                             const dim3&               blkDim,
                                             cudaStream_t              strm)
{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    if(config.BG == 1)
    {
        //typedef cC2V_index<__half, 1, sign_store_policy_dst, sign_store_policy_split_src> cC2V_t;
        typedef cC2V_index<__half, 1, sign_store_policy_src, sign_store_policy_split_src> cC2V_t;
        switch(config.Z)
        {
        case 384:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared         <__half, 1, 22, 384,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared         <__half, 1, 22, 384,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared         <__half, 1, 22, 384,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared         <__half, 1, 22, 384,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared         <__half, 1, 22, 384,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 9:  s = launch_glob_all_shared         <__half, 1, 22, 384,  9, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 10: s = launch_glob_all_shared         <__half, 1, 22, 384, 10, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 11: s = launch_glob_all_shared         <__half, 1, 22, 384, 11, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 12: s = launch_glob_all_shared         <__half, 1, 22, 384, 12, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
#if 0
            default: s = launch_glob_part_shared_dynamic<__half, 1, 22, 384, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
#else
            //case 46: launch_glob_all_shared         <__half, 1, 22, 384, 46, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 46: s = launch_glob_part_shared_dynamic<__half, 1, 22, 384, 46, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:                                                                                                                     break;
#endif
            }
            break;
        default:
            break;
        }
    }
    else if(config.BG == 2)
    {
        // TODO
    }
    return s;
}

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_global_index_float()
cuphyStatus_t decode_ldpc2_global_index_float(const LDPC_config&        config,
                                              const LDPC_kernel_params& params,
                                              const dim3&               grdDim,
                                              const dim3&               blkDim,
                                              cudaStream_t              strm)
{
    cuphyStatus_t s = CUPHY_STATUS_NOT_SUPPORTED;
    if(config.BG == 1)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Check to Variable (C2V) message type
        typedef cC2V_index<float, 1, sign_store_policy_dst, sign_mgr_be> cC2V_t;
        //typedef cC2V_index<float, sign_store_policy_src, sign_mgr_be> cC2V_t;

        switch(config.Z)
        {
        case 64:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 1, 22, 64,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 1, 22, 64,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 1, 22, 64,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 1, 22, 64,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 1, 22, 64,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 1, 22, 64, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
                }
                break;
            }
            break;
        case 96:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 1, 22, 96,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 1, 22, 96,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 1, 22, 96,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 1, 22, 96,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 1, 22, 96,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 1, 22, 96, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
                }
                break;
            }
            break;
        case 128:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 1, 22, 128,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 1, 22, 128,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 1, 22, 128,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 1, 22, 128,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 1, 22, 128,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 1, 22, 128, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
                }
                break;
            }
            break;
        case 160:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 1, 22, 160,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 1, 22, 160,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 1, 22, 160,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 1, 22, 160,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 1, 22, 160,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 1, 22, 160, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
                }
                break;
            }
            break;
        case 192:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 1, 22, 192,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 1, 22, 192,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 1, 22, 192,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 1, 22, 192,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 1, 22, 192,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 1, 22, 192, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
                }
                break;
            }
            break;
        case 224:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 1, 22, 224,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 1, 22, 224,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 1, 22, 224,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 1, 22, 224,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 1, 22, 224,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 1, 22, 224, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);;
                }
                break;
            }
            break;
        case 256:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 1, 22, 256,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 1, 22, 256,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 1, 22, 256,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 1, 22, 256,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 1, 22, 256,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 1, 22, 256, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
                }
                break;
            }
            break;
        case 288:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 1, 22, 288,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 1, 22, 288,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 1, 22, 288,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 1, 22, 288,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 1, 22, 288,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 1, 22, 288, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
                }
                break;
            }
            break;
        case 320:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 1, 22, 320,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 1, 22, 320,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 1, 22, 320,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 1, 22, 320,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 1, 22, 320,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 1, 22, 320, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
                }
                break;
            }
            break;
        case 352:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 1, 22, 352,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 1, 22, 352,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 1, 22, 352,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 1, 22, 352,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 1, 22, 352,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 1, 22, 352, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
                }
                break;
            }
            break;
        case 384:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 1, 22, 384,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 1, 22, 384,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 1, 22, 384,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 1, 22, 384,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 1, 22, 384,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            
            //case 8:  s = launch_glob_part_shared<float, 1, 22, 384,  8, 26, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            
            //case 9:  s = launch_glob_all_shared<float, 1, 22, 384,  9, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 10: s = launch_glob_all_shared<float, 1, 22, 384, 10, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 11: s = launch_glob_part_shared<float, 1, 22, 384, 11, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 12: s = launch_glob_part_shared<float, 1, 22, 384, 12, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 13: s = launch_glob_part_shared<float, 1, 22, 384, 13, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 14: s = launch_glob_part_shared<float, 1, 22, 384, 14, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 15: s = launch_glob_part_shared<float, 1, 22, 384, 15, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 16: s = launch_glob_part_shared<float, 1, 22, 384, 16, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 17: s = launch_glob_part_shared<float, 1, 22, 384, 17, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 18: s = launch_glob_part_shared<float, 1, 22, 384, 18, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 19: s = launch_glob_part_shared<float, 1, 22, 384, 19, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 20: s = launch_glob_part_shared<float, 1, 22, 384, 20, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 21: s = launch_glob_part_shared<float, 1, 22, 384, 21, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 22: s = launch_glob_part_shared<float, 1, 22, 384, 22, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 23: s = launch_glob_part_shared<float, 1, 22, 384, 23, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 24: s = launch_glob_part_shared<float, 1, 22, 384, 24, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 25: s = launch_glob_part_shared<float, 1, 22, 384, 25, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 26: s = launch_glob_part_shared<float, 1, 22, 384, 26, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 27: s = launch_glob_part_shared<float, 1, 22, 384, 27, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 28: s = launch_glob_part_shared<float, 1, 22, 384, 28, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 29: s = launch_glob_part_shared<float, 1, 22, 384, 29, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 30: s = launch_glob_part_shared<float, 1, 22, 384, 30, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 31: s = launch_glob_part_shared<float, 1, 22, 384, 31, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 32: s = launch_glob_part_shared<float, 1, 22, 384, 32, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 33: s = launch_glob_part_shared<float, 1, 22, 384, 33, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 34: s = launch_glob_part_shared<float, 1, 22, 384, 34, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 35: s = launch_glob_part_shared<float, 1, 22, 384, 35, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 36: s = launch_glob_part_shared<float, 1, 22, 384, 36, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 37: s = launch_glob_part_shared<float, 1, 22, 384, 37, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 38: s = launch_glob_part_shared<float, 1, 22, 384, 38, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 39: s = launch_glob_part_shared<float, 1, 22, 384, 39, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 40: s = launch_glob_part_shared<float, 1, 22, 384, 40, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 41: s = launch_glob_part_shared<float, 1, 22, 384, 41, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 42: s = launch_glob_part_shared<float, 1, 22, 384, 42, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 43: s = launch_glob_part_shared<float, 1, 22, 384, 43, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 44: s = launch_glob_part_shared<float, 1, 22, 384, 44, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 45: s = launch_glob_part_shared<float, 1, 22, 384, 45, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            //case 46: s = launch_glob_part_shared<float, 1, 22, 384, 46, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:
#if 1
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 1, 22, 384, 32, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
                }
#endif
                break;
            }
            break;
        default:
            break;
        }
    }
    else
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Check to Variable (C2V) message type
        typedef cC2V_index<float, 2, sign_store_policy_dst, sign_mgr_be> cC2V_t;
        //typedef cC2V_index<float, sign_store_policy_src, sign_mgr_be> cC2V_t;

        switch(config.Z)
        {
        case 64:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 2, 10, 64,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 2, 10, 64,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 2, 10, 64,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 2, 10, 64,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 2, 10, 64,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 2, 10, 64, 20, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
                }
                break;
            }
            break;
        case 96:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 2, 10, 96,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 2, 10, 96,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 2, 10, 96,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 2, 10, 96,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 2, 10, 96,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 2, 10, 96, 20, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
                }
                break;
            }
            break;
        case 128:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 2, 10, 128,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 2, 10, 128,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 2, 10, 128,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 2, 10, 128,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 2, 10, 128,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 2, 10, 128, 20, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
                }
                break;
            }
            break;
        case 160:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 2, 10, 160,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 2, 10, 160,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 2, 10, 160,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 2, 10, 160,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 2, 10, 160,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 2, 10, 160, 20, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
                }
                break;
            }
            break;
        case 192:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 2, 10, 192,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 2, 10, 192,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 2, 10, 192,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 2, 10, 192,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 2, 10, 192,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 2, 10, 192, 20, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
                }
                break;
            }
            break;
        case 224:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 2, 10, 224,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 2, 10, 224,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 2, 10, 224,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 2, 10, 224,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 2, 10, 224,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 2, 10, 224, 20, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
                }
                break;
            }
            break;
        case 256:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 2, 10, 256,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 2, 10, 256,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 2, 10, 256,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 2, 10, 256,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 2, 10, 256,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 2, 10, 256, 20, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
                }
                break;
            }
            break;
        case 288:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 2, 10, 288,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 2, 10, 288,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 2, 10, 288,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 2, 10, 288,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 2, 10, 288,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 2, 10, 288, 20, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
                }
                break;
            }
            break;
        case 320:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 2, 10, 320,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 2, 10, 320,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 2, 10, 320,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 2, 10, 320,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 2, 10, 320,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 2, 10, 320, 20, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
                }
                break;
            }
            break;
        case 352:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 2, 10, 352,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 2, 10, 352,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 2, 10, 352,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 2, 10, 352,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 2, 10, 352,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            default:
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 2, 10, 352, 20, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
                }
                break;
            }
            break;
        case 384:
            switch(config.mb)
            {
            case 4:  s = launch_glob_all_shared<float, 2, 10, 384,  4, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 5:  s = launch_glob_all_shared<float, 2, 10, 384,  5, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 6:  s = launch_glob_all_shared<float, 2, 10, 384,  6, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 7:  s = launch_glob_all_shared<float, 2, 10, 384,  7, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            case 8:  s = launch_glob_all_shared<float, 2, 10, 384,  8, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm); break;
            
            default:
                if(config.mb <= 46)
                {
                    s = launch_glob_part_shared_dynamic<float, 2, 10, 384, 20, cC2V_t, app_loc_address>(params, grdDim, blkDim, strm);
                }
                break;
            }
            break;
        default:
            break;
        }
    }
    return s;
}

////////////////////////////////////////////////////////////////////////
// decode_ldpc2_global_index()
cuphyStatus_t decode_ldpc2_global_index(LDPC_output_t&         tDst,
                                        const_tensor_pair&     tLLR,
                                        const LDPC_config&     config,
                                        float                  normalization,
                                        cuphyLDPCResults_t*    results,
                                        void*                  workspace,
                                        cuphyLDPCDiagnostic_t* diag,
                                        cudaStream_t           strm)
{
    DEBUG_PRINTF("ldpc::decode_ldpc2_global_index()\n");
    //------------------------------------------------------------------
    cuphyDataType_t llrType = tLLR.first.get().type();
    //------------------------------------------------------------------
    dim3 grdDim(config.num_codewords); // 1-D, NUM_CW for fp32, NUM_CW / 2 for fp16
    dim3 blkDim(config.Z);

    //------------------------------------------------------------------
    // Initialize the kernel params struct
    LDPC_kernel_params params(config, tLLR, tDst, normalization, workspace);

    cuphyStatus_t s;
    
    if(llrType == CUPHY_R_32F)
    {
        s = decode_ldpc2_global_index_float(config,
                                            params,
                                            grdDim,
                                            blkDim,
                                            strm);
    }
    else if(llrType == CUPHY_R_16F)
    {
        // Convert the normalization value to __half2
        params.norm.f16x2 = __float2half2_rn(params.norm.f32);
        s = decode_ldpc2_global_index_half(config,
                                           params,
                                           grdDim,
                                           blkDim,
                                           strm);
    }
    else
    {
        return CUPHY_STATUS_NOT_SUPPORTED;
    }

    if(CUPHY_STATUS_SUCCESS != s)
    {
        return s;
    }

#if CUPHY_DEBUG
    cudaDeviceSynchronize();
#endif
    cudaError_t e = cudaGetLastError();
    DEBUG_PRINTF("CUDA STATUS (%s:%i): %s\n", __FILE__, __LINE__, cudaGetErrorString(e));
    return (e == cudaSuccess) ? CUPHY_STATUS_SUCCESS : CUPHY_STATUS_INTERNAL_ERROR;
}

//----------------------------------------------------------------------
// decode_ldpc2_global_address_workspace_size()
std::pair<bool, size_t> decode_ldpc2_global_address_workspace_size(const LDPC_config& cfg)
{
    // TODO: Possibly adjust size of per-thread element
    return std::pair<bool, size_t>(true, cfg.num_codewords * cfg.mb * cfg.Z * sizeof(int4));
}

//----------------------------------------------------------------------
// decode_ldpc2_global_index_workspace_size()
std::pair<bool, size_t> decode_ldpc2_global_index_workspace_size(const LDPC_config& cfg)
{
    if(CUPHY_R_32F == cfg.type)
    {
        return std::pair<bool, size_t>(true, cfg.num_codewords * cfg.mb * cfg.Z * sizeof(int4));
    }
    else if(CUPHY_R_16F == cfg.type)
    {
        // Assumes all of workspace is used for cC2V messages (i.e. no APP values)
        return std::pair<bool, size_t>(true, cfg.num_codewords * cfg.mb * cfg.Z * sizeof(int2));
    }
    else
    {
        return std::pair<bool, size_t>(false, 0);
    }
}

} // namespace ldpc
