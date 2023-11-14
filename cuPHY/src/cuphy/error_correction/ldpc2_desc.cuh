/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_DESC_CUH_INCLUDED_)
#define LDPC2_DESC_CUH_INCLUDED_

// Functions and kernels supporting an LDPC decoder that uses a descriptor
// structure to provide base graph information. (Other kernels may embed
// base graph info directly into instructions as immediates, resulting in
// a separate kernel for each LDPC configuration.)

#include "ldpc2_sign_split.cuh"
#include "nrLDPC_templates.cuh"
#include "ldpc2_llr_loader.cuh"
#include "ldpc2_dec_output.cuh"

////////////////////////////////////////////////////////////////////////
// Base graph descriptor LDPC decoder kernels
// These kernels use a base graph descriptor, passed as a kernel
// argument, to determine APP addresses. (Alternative approaches are
// to use a constant memory buffer, or to embed the base graph data
// in instruction immediates, with a different kernel for each base
// graph configuration.)
//
//namespace ldpc2
//{
//
////////////////////////////////////////////////////////////////////////
// ldpc2_bg_desc_kernel()
// Kernel for LDPC decoding, using a "base graph descriptor" kernel
// argument to perform APP address calculations.
// Template parameters:
// T:               Data type (__half)
// BG:              Base graph (1 or 2)
// TLoader:         LLR variable loader
// TSched:          Scheduler class
// THREADS_PER_CTA: Number of threads per CTA (used for launch bounds only)
// BLOCKS_PER_SM:   Number of blocks per SM (used for launch bounds only)
//template <typename             T,
//          int                  BG,
//          class                TLoader,
//          class                TSched,
//          template <int> class BGDesc,
//          int                  THREADS_PER_CTA,
//          int                  BLOCKS_PER_SM>
//__global__ __launch_bounds__(THREADS_PER_CTA, BLOCKS_PER_SM)
//void ldpc2_bg_desc_kernel(LDPC_kernel_params params, BGDesc<BG> bgdesc)
//{
//    //------------------------------------------------------------------
//    // LLR loader (used to load input from global to shared memory)
//    typedef TLoader                           llr_loader_t;
//    typedef typename llr_loader_t::app_buf_t  app_buf_t;
//    typedef ldpc_dec_output_params<app_buf_t> output_params_t;
//    //------------------------------------------------------------------
//    // Shared memory
//    extern __shared__ char smem[];
//    app_buf_t*             app_smem = reinterpret_cast<app_buf_t*>(smem);
//    //------------------------------------------------------------------
//    // Load LLR data from global to shared memory
//    llr_loader_t::load_sync(smem, params);
//
//    //------------------------------------------------------------------
//    TSched sched(params, bgdesc);
//
//    for(int iter = 0; iter < params.max_iterations; ++iter)
//    {
//        sched.do_iteration();
//    }
//#if 0
//    print_array_sync("APP", app_smem, params.num_var_nodes * params.Z);
//#endif
//    //------------------------------------------------------------------
//    // Write output based on APP values
//    //ldpc_dec_output_fixed<Kb, Z>(params, app_smem, threadIdx.x);
//    ldpc_dec_output_variable(output_params_t(params), app_smem);
//}

////////////////////////////////////////////////////////////////////////
// ldpc2_bg_desc_device_func()
// Device function for LDPC decoding, using a "base graph descriptor"
// argument to perform APP address calculations.
// Template parameters:
// T:               Data type (__half)
// BG:              Base graph (1 or 2)
// TLoader:         LLR variable loader
// TSched:          Scheduler class
// THREADS_PER_CTA: Number of threads per CTA (used for launch bounds only)
// BLOCKS_PER_SM:   Number of blocks per SM (used for launch bounds only)
//template <typename             T,
//          int                  BG,
//          class                TLoader,
//          class                TSched,
//          template <int> class BGDesc>
//__forceinline__ __device__
//void ldpc2_bg_desc_device_func(const LDPC_kernel_params& params, const BGDesc<BG>& bgdesc)
//{
//    //------------------------------------------------------------------
//    // LLR loader (used to load input from global to shared memory)
//    typedef TLoader                           llr_loader_t;
//    typedef typename llr_loader_t::app_buf_t  app_buf_t;
//    typedef ldpc_dec_output_params<app_buf_t> output_params_t;
//    //------------------------------------------------------------------
//    // Shared memory
//    extern __shared__ char smem[];
//    app_buf_t*             app_smem = reinterpret_cast<app_buf_t*>(smem);
//    //------------------------------------------------------------------
//    // Load LLR data from global to shared memory
//    llr_loader_t::load_sync(smem, params);
//
//    //------------------------------------------------------------------
//    TSched sched(params, bgdesc);
//
//    for(int iter = 0; iter < params.max_iterations; ++iter)
//    {
//        sched.do_iteration();
//    }
//#if 0
//    print_array_sync("APP", app_smem, params.num_var_nodes * params.Z);
//#endif
//    //------------------------------------------------------------------
//    // Write output based on APP values
//    //ldpc_dec_output_fixed<Kb, Z>(params, app_smem, threadIdx.x);
//    ldpc_dec_output_variable(output_params_t(params), app_smem);
//}

////////////////////////////////////////////////////////////////////////
// ldpc2_desc_tb_device_func()
// Device function for performing LDPC decode using the "transport
// block" interface, and a base graph descriptor argument.
// Template parameters:
// T:               Data type (__half)
// BG:              Base graph (1 or 2)
// TLoader:         LLR variable loader
// TSched:          Scheduler class
// THREADS_PER_CTA: Number of threads per CTA (used for launch bounds only)
// BLOCKS_PER_SM:   Number of blocks per SM (used for launch bounds only)
//template <typename             T,
//          int                  BG,
//          class                TLoader,
//          class                TSched,
//          template <int> class BGDesc>
//__forceinline__ __device__
//void ldpc2_desc_tb_device_func(const cuphyLDPCDecodeDesc_t& decodeDesc, const BGDesc<BG>& bgdesc)
//{
//    //------------------------------------------------------------------
//    // LLR loader (used to load input from global to shared memory)
//    typedef TLoader                           llr_loader_t;
//    typedef typename llr_loader_t::app_buf_t  app_buf_t;
//    typedef ldpc_dec_output_params<app_buf_t> output_params_t;
//    //------------------------------------------------------------------
//    // Shared memory
//    extern __shared__ char smem[];
//    app_buf_t*             app_smem   = reinterpret_cast<app_buf_t*>(smem);
//    //------------------------------------------------------------------
//    // Load LLR data from global to shared memory
//    llr_loader_t::load_sync(smem, decodeDesc);
//    //------------------------------------------------------------------
//    TSched sched(decodeDesc.config, bgdesc);
//    for(int iter = 0; iter < decodeDesc.config.max_iterations; ++iter)
//    {
//        sched.do_iteration();
//    }
//    //------------------------------------------------------------------
//    // Write output based on APP values
//    ldpc_dec_output_variable(output_params_t(decodeDesc), app_smem);
//}

////////////////////////////////////////////////////////////////////////
// ldpc2_desc_tb_kernel()
// CUDA kernel for performing LDPC decode using the "transport block"
// interface, and a base graph descriptor kernel argument.
// Template parameters:
// T:               Data type (__half)
// BG:              Base graph (1 or 2)
// TLoader:         LLR variable loader
// TSched:          Scheduler class
// THREADS_PER_CTA: Number of threads per CTA (used for launch bounds only)
// BLOCKS_PER_SM:   Number of blocks per SM (used for launch bounds only)
//template <typename             T,
//          int                  BG,
//          class                TLoader,
//          class                TSched,
//          template <int> class BGDesc,
//          int                  THREADS_PER_CTA,
//          int                  BLOCKS_PER_SM>
//__global__ __launch_bounds__(THREADS_PER_CTA, BLOCKS_PER_SM)
//void ldpc2_desc_tb_kernel(cuphyLDPCDecodeDesc_t decodeDesc, BGDesc<BG> bgdesc)
//{
//    //------------------------------------------------------------------
//    // LLR loader (used to load input from global to shared memory)
//    typedef TLoader                           llr_loader_t;
//    typedef typename llr_loader_t::app_buf_t  app_buf_t;
//    typedef ldpc_dec_output_params<app_buf_t> output_params_t;
//    //------------------------------------------------------------------
//    // Shared memory
//    extern __shared__ char smem[];
//    app_buf_t*             app_smem   = reinterpret_cast<app_buf_t*>(smem);
//    //------------------------------------------------------------------
//    // Load LLR data from global to shared memory
//    llr_loader_t::load_sync(smem, decodeDesc);
//    //------------------------------------------------------------------
//    TSched sched(decodeDesc.config, bgdesc);
//    for(int iter = 0; iter < decodeDesc.config.max_iterations; ++iter)
//    {
//        sched.do_iteration();
//    }
//    //------------------------------------------------------------------
//    // Write output based on APP values
//    ldpc_dec_output_variable(output_params_t(decodeDesc), app_smem);
//}

////////////////////////////////////////////////////////////////////////
// launch_reg_bg_desc_kernel_dynamic()
// Kernel launch wrapper function for the LDPC decoder kernel using:
// a.) register storage for cC2V data
// b.) base graph descriptor kernel argument (to allow handling multiple
//     lifting sizes with the same kernel)
// c.) "dynamic" row schedule, i.e. the number of parity check rows is
//     unknown at compile time. The kernel checks the input parameters
//     for the number of parity check rows at the end of each row.
// d.) the cuPHY tensor-based interface (as opposed to the cuPHY
//     transport block interface)
//template <typename             T,
//          int                  BG,
//          class                TC2V,
//          template <int> class BGDesc,
//          class                TAPPLoc,
//          int                  MIN_PARITY_ROWS,
//          int                  MAX_PARITY_ROWS,
//          int                  THREADS_PER_CTA,
//          int                  BLOCKS_PER_SM>
//cuphyStatus_t launch_reg_bg_desc_kernel_dynamic(ldpc::decoder&            dec,
//                                                const LDPC_kernel_params& params,
//                                                const BGDesc<BG>&         bg_desc,
//                                                const dim3&               grdDim,
//                                                const dim3&               blkDim,
//                                                cudaStream_t              strm)
//{
//    // C2V message cache (register memory here)
//    // Maximum size for worst-case (lowest code rate) scenario
//    //typedef c2v_cache_register<BG, max_parity_nodes<BG>::value, TC2V> c2v_cache_t;
//    typedef c2v_cache_register<BG, MAX_PARITY_ROWS, TC2V, LDPC_kernel_params> c2v_cache_t;
//
//    //------------------------------------------------------------------
//    // APP "location" manager - calculates location of APP values for
//    // threads based on base graph shift values
//    //typedef TAPPLoc<T, BG, Z> app_loc_t;
//    //typedef app_loc_address_fp_desc<T, BG> app_loc_t;
//    typedef TAPPLoc app_loc_t;
//    
//    //------------------------------------------------------------------
//    // "Dynamic" schedule, with the number of parity rows not known until
//    // runtime.
//    typedef ldpc_schedule_dynamic_desc<BG,
//                                       app_loc_t,
//                                       c2v_cache_t,
//                                       LDPC_kernel_params,
//                                       BGDesc,
//                                       MIN_PARITY_ROWS,
//                                       MAX_PARITY_ROWS> sched_t;
//    
//    //------------------------------------------------------------------
//    // LLR loader, used to load LLR data from global to shared memory
//    //typedef llr_loader_variable_batch_fixed_cta<T, THREADS_PER_CTA, 4> llr_loader_t;
//    typedef llr_loader_variable_batch<T, 4, llr_op_clamp> llr_loader_t;
//
//    //------------------------------------------------------------------
//    // Determine the dynamic amount of shared memory
//    const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(params.num_var_nodes, // num shared memory nodes
//                                                      params.Z,             // lifting size
//                                                      sizeof(T));           // element size
//    cudaError_t e = cudaFuncSetAttribute(ldpc2_bg_desc_kernel<T, BG, llr_loader_t, sched_t, BGDesc, THREADS_PER_CTA, BLOCKS_PER_SM>,
//                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
//                                         SHMEM_SIZE);
//    if(cudaSuccess != e)
//    {
//        return CUPHY_STATUS_INTERNAL_ERROR;
//    }
//    //------------------------------------------------------------------
//    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc2_bg_desc_kernel<T, BG, llr_loader_t, sched_t, BGDesc, THREADS_PER_CTA, BLOCKS_PER_SM>));
//    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc2_bg_desc_kernel<T, BG, llr_loader_t, sched_t, BGDesc, THREADS_PER_CTA, BLOCKS_PER_SM>), blkDim, SHMEM_SIZE);
//    ldpc2_bg_desc_kernel<T,               // LLR data type
//                         BG,              // base graph
//                         llr_loader_t,    // LLR loader type
//                         sched_t,         // row scheduler
//                         BGDesc,          // base graph descriptor
//                         THREADS_PER_CTA, // threads per block
//                         BLOCKS_PER_SM>   // launch bounds
//                         <<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, bg_desc);
//    return CUPHY_STATUS_SUCCESS;
//}

////////////////////////////////////////////////////////////////////////
// launch_reg_bg_desc_kernel_dynamic_convert()
// Kernel launch wrapper function for the LDPC decoder kernel using:
// a.) register storage for cC2V data
// b.) base graph descriptor kernel argument (to allow handling multiple
//     lifting sizes with the same kernel)
// c.) "dynamic" row schedule, i.e. the number of parity check rows is
//     unknown at compile time. The kernel checks the input parameters
//     for the number of parity check rows at the end of each row.
// d.) the cuPHY tensor-based interface (as opposed to the cuPHY
//     transport block interface)
// e.) conversion of the source LLR type to a different APP type upon
//     loading
//template <typename             T,
//          int                  BG,
//          class                TC2V,
//          template <int> class BGDesc,
//          class                TAPPLoc,
//          int                  MIN_PARITY_ROWS,
//          int                  MAX_PARITY_ROWS,
//          int                  THREADS_PER_CTA,
//          int                  BLOCKS_PER_SM>
//cuphyStatus_t launch_reg_bg_desc_kernel_dynamic_convert(ldpc::decoder&            dec,
//                                                        const LDPC_kernel_params& params,
//                                                        const BGDesc<BG>&         bg_desc,
//                                                        const dim3&               grdDim,
//                                                        const dim3&               blkDim,
//                                                        cudaStream_t              strm)
//{
//    // C2V message cache (register memory here)
//    // Maximum size for worst-case (lowest code rate) scenario
//    //typedef c2v_cache_register<BG, max_parity_nodes<BG>::value, TC2V> c2v_cache_t;
//    typedef c2v_cache_register<BG, MAX_PARITY_ROWS, TC2V, LDPC_kernel_params> c2v_cache_t;
//
//    //------------------------------------------------------------------
//    // APP "location" manager - calculates location of APP values for
//    // threads based on base graph shift values
//    //typedef TAPPLoc<T, BG, Z> app_loc_t;
//    //typedef app_loc_address_fp_desc<T, BG> app_loc_t;
//    typedef TAPPLoc app_loc_t;
//    
//    //------------------------------------------------------------------
//    // "Dynamic" schedule, with the number of parity rows not known until
//    // runtime.
//    typedef ldpc_schedule_dynamic_desc<BG,
//                                       app_loc_t,
//                                       c2v_cache_t,
//                                       LDPC_kernel_params,
//                                       BGDesc,
//                                       MIN_PARITY_ROWS,
//                                       MAX_PARITY_ROWS> sched_t;
//    
//    //------------------------------------------------------------------
//    // LLR loader, used to load LLR data from global to shared memory
//    typedef llr_loader_variable_batch_convert<T, 4, llr_op_clamp> llr_loader_t;
//    typedef typename llr_loader_t::app_buf_t                      app_buf_t;
//    //------------------------------------------------------------------
//    // Determine the dynamic amount of shared memory
//    const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(params.num_var_nodes, // num shared memory nodes
//                                                      params.Z,             // lifting size
//                                                      sizeof(app_buf_t));   // element size
//    cudaError_t e = cudaFuncSetAttribute(ldpc2_bg_desc_kernel<T, BG, llr_loader_t, sched_t, BGDesc, THREADS_PER_CTA, BLOCKS_PER_SM>,
//                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
//                                         SHMEM_SIZE);
//    if(cudaSuccess != e)
//    {
//        return CUPHY_STATUS_INTERNAL_ERROR;
//    }
//    //------------------------------------------------------------------
//    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc2_bg_desc_kernel<T, BG, llr_loader_t, sched_t, BGDesc, THREADS_PER_CTA, BLOCKS_PER_SM>));
//    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc2_bg_desc_kernel<T, BG, llr_loader_t, sched_t, BGDesc, THREADS_PER_CTA, BLOCKS_PER_SM>), blkDim, SHMEM_SIZE);
//    ldpc2_bg_desc_kernel<T,               // LLR data type
//                         BG,              // base graph
//                         llr_loader_t,    // LLR loader type
//                         sched_t,         // row scheduler
//                         BGDesc,          // base graph descriptor
//                         THREADS_PER_CTA, // threads per block
//                         BLOCKS_PER_SM>   // launch bounds
//                         <<<grdDim, blkDim, SHMEM_SIZE, strm>>>(params, bg_desc);
//    return CUPHY_STATUS_SUCCESS;
//}

////////////////////////////////////////////////////////////////////////
// launch_reg_bg_desc_tb_kernel_dynamic()
// Kernel launch wrapper function for the LDPC decoder kernel using:
// a.) register storage for cC2V data
// b.) base graph descriptor kernel argument (to allow handling multiple
//     lifting sizes with the same kernel)
// c.) "dynamic" row schedule, i.e. the number of parity check rows is
//     unknown at compile time. The kernel checks the input parameters
//     for the number of parity check rows at the end of each row.
// d.) the cuPHY transport block interface (as opposed to the cuPHY
//     tensor interface). The transport block interface allows the
//     caller to provide multiple (identical) transport blocks with a
//     single kernel launch. The tensor interface accepts a single
//     tensor that prescribes a 2-D tensor layou, and as such requires
//     a separate kernel launch for each transport block (unless
//     multiple tranport blocks are "fused" to become representable by
//     a single tensor).
//template <typename             T,
//          int                  BG,
//          class                TC2V,
//          template <int> class BGDesc,
//          class                TAPPLoc,
//          int                  MIN_PARITY_ROWS,
//          int                  MAX_PARITY_ROWS,
//          int                  THREADS_PER_CTA,
//          int                  BLOCKS_PER_SM>
//cuphyStatus_t launch_reg_bg_desc_tb_kernel_dynamic(ldpc::decoder&               dec,
//                                                   const cuphyLDPCDecodeDesc_t& dec_desc,
//                                                   const BGDesc<BG>&            bg_desc,
//                                                   const dim3&                  grdDim,
//                                                   const dim3&                  blkDim,
//                                                   cudaStream_t                 strm)
//{
//    // C2V message cache (register memory here)
//    // Maximum size for worst-case (lowest code rate) scenario
//    //typedef c2v_cache_register<BG, max_parity_nodes<BG>::value, TC2V> c2v_cache_t;
//    typedef c2v_cache_register<BG, MAX_PARITY_ROWS, TC2V, cuphyLDPCDecodeConfigDesc_t> c2v_cache_t;
//
//    //------------------------------------------------------------------
//    // APP "location" manager - calculates location of APP values for
//    // threads based on base graph shift values
//    //typedef TAPPLoc<T, BG, Z> app_loc_t;
//    //typedef app_loc_address_fp_desc<T, BG> app_loc_t;
//    typedef TAPPLoc app_loc_t;
//    
//    //------------------------------------------------------------------
//    // "Dynamic" schedule, with the number of parity rows not known until
//    // runtime.
//    typedef ldpc_schedule_dynamic_desc<BG,
//                                       app_loc_t,
//                                       c2v_cache_t,
//                                       cuphyLDPCDecodeConfigDesc_t,
//                                       BGDesc,
//                                       MIN_PARITY_ROWS,
//                                       MAX_PARITY_ROWS> sched_t;
//    
//    //------------------------------------------------------------------
//    // LLR loader, used to load LLR data from global to shared memory
//    //typedef llr_loader_variable_batch_fixed_cta<T, THREADS_PER_CTA, 4> llr_loader_t;
//    typedef llr_loader_variable_batch<T, 4, llr_op_clamp> llr_loader_t;
//
//    //------------------------------------------------------------------
//    // Determine the dynamic amount of shared memory
//    const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(dec_desc.config.num_parity_nodes + max_info_nodes<BG>::value, // num shared memory nodes
//                                                      dec_desc.config.Z,                                            // lifting size
//                                                      sizeof(T));                                                   // element size
//    cudaError_t e = cudaFuncSetAttribute(ldpc2_desc_tb_kernel<T, BG, llr_loader_t, sched_t, BGDesc, THREADS_PER_CTA, BLOCKS_PER_SM>,
//                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
//                                         SHMEM_SIZE);
//    if(cudaSuccess != e)
//    {
//        return CUPHY_STATUS_INTERNAL_ERROR;
//    }
//    //------------------------------------------------------------------
//    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc2_desc_tb_kernel<T, BG, llr_loader_t, sched_t, BGDesc, THREADS_PER_CTA, BLOCKS_PER_SM>));
//    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc2_desc_tb_kernel<T, BG, llr_loader_t, sched_t, BGDesc, THREADS_PER_CTA, BLOCKS_PER_SM>), blkDim, SHMEM_SIZE);
//    ldpc2_desc_tb_kernel<T,               // LLR data type
//                         BG,              // base graph
//                         llr_loader_t,    // LLR loader type
//                         sched_t,         // row scheduler
//                         BGDesc,          // base graph descriptor
//                         THREADS_PER_CTA, // threads per block
//                         BLOCKS_PER_SM>   // launch bounds
//                         <<<grdDim, blkDim, SHMEM_SIZE, strm>>>(dec_desc, bg_desc);
//    return CUPHY_STATUS_SUCCESS;
//}

////////////////////////////////////////////////////////////////////////
// launch_reg_bg_desc_tb_kernel_dynamic()
// Kernel launch wrapper function for the LDPC decoder kernel using:
// a.) register storage for cC2V data
// b.) base graph descriptor kernel argument (to allow handling multiple
//     lifting sizes with the same kernel)
// c.) "dynamic" row schedule, i.e. the number of parity check rows is
//     unknown at compile time. The kernel checks the input parameters
//     for the number of parity check rows at the end of each row.
// d.) the cuPHY transport block interface (as opposed to the cuPHY
//     tensor interface). The transport block interface allows the
//     caller to provide multiple (identical) transport blocks with a
//     single kernel launch. The tensor interface accepts a single
//     tensor that prescribes a 2-D tensor layou, and as such requires
//     a separate kernel launch for each transport block (unless
//     multiple tranport blocks are "fused" to become representable by
//     a single tensor).
// 3.) convert from the source LLR type to a different APP buffer type
//template <typename             T,
//          int                  BG,
//          class                TC2V,
//          template <int> class BGDesc,
//          class                TAPPLoc,
//          int                  MIN_PARITY_ROWS,
//          int                  MAX_PARITY_ROWS,
//          int                  THREADS_PER_CTA,
//          int                  BLOCKS_PER_SM>
//cuphyStatus_t launch_reg_bg_desc_tb_kernel_dynamic_convert(ldpc::decoder&               dec,
//                                                           const cuphyLDPCDecodeDesc_t& dec_desc,
//                                                           const BGDesc<BG>&            bg_desc,
//                                                           const dim3&                  grdDim,
//                                                           const dim3&                  blkDim,
//                                                           cudaStream_t                 strm)
//{
//    // C2V message cache (register memory here)
//    // Maximum size for worst-case (lowest code rate) scenario
//    //typedef c2v_cache_register<BG, max_parity_nodes<BG>::value, TC2V> c2v_cache_t;
//    typedef c2v_cache_register<BG, MAX_PARITY_ROWS, TC2V, cuphyLDPCDecodeConfigDesc_t> c2v_cache_t;
//
//    //------------------------------------------------------------------
//    // APP "location" manager - calculates location of APP values for
//    // threads based on base graph shift values
//    //typedef TAPPLoc<T, BG, Z> app_loc_t;
//    //typedef app_loc_address_fp_desc<T, BG> app_loc_t;
//    typedef TAPPLoc app_loc_t;
//    
//    //------------------------------------------------------------------
//    // "Dynamic" schedule, with the number of parity rows not known until
//    // runtime.
//    typedef ldpc_schedule_dynamic_desc<BG,
//                                       app_loc_t,
//                                       c2v_cache_t,
//                                       cuphyLDPCDecodeConfigDesc_t,
//                                       BGDesc,
//                                       MIN_PARITY_ROWS,
//                                       MAX_PARITY_ROWS> sched_t;
//    
//    //------------------------------------------------------------------
//    // LLR loader, used to load LLR data from global to shared memory
//    //typedef llr_loader_variable_batch_fixed_cta<T, THREADS_PER_CTA, 4> llr_loader_t;
//    typedef llr_loader_variable_batch_convert<T, 4, llr_op_clamp> llr_loader_t;
//    typedef typename llr_loader_t::app_buf_t                      app_buf_t;
//    
//    //------------------------------------------------------------------
//    // Determine the dynamic amount of shared memory
//    const uint32_t SHMEM_SIZE = shmem_llr_buffer_size(dec_desc.config.num_parity_nodes + max_info_nodes<BG>::value, // num shared memory nodes
//                                                      dec_desc.config.Z,                                            // lifting size
//                                                      sizeof(app_buf_t));                                           // element size
//    cudaError_t e = cudaFuncSetAttribute(ldpc2_desc_tb_kernel<T, BG, llr_loader_t, sched_t, BGDesc, THREADS_PER_CTA, BLOCKS_PER_SM>,
//                                         cudaFuncAttributeMaxDynamicSharedMemorySize,
//                                         SHMEM_SIZE);
//    if(cudaSuccess != e)
//    {
//        return CUPHY_STATUS_INTERNAL_ERROR;
//    }
//    //------------------------------------------------------------------
//    DEBUG_PRINT_FUNC_ATTRIBUTES((ldpc2_desc_tb_kernel<T, BG, llr_loader_t, sched_t, BGDesc, THREADS_PER_CTA, BLOCKS_PER_SM>));
//    DEBUG_PRINT_FUNC_MAX_BLOCKS((ldpc2_desc_tb_kernel<T, BG, llr_loader_t, sched_t, BGDesc, THREADS_PER_CTA, BLOCKS_PER_SM>), blkDim, SHMEM_SIZE);
//    ldpc2_desc_tb_kernel<T,               // LLR data type
//                         BG,              // base graph
//                         llr_loader_t,    // LLR loader type
//                         sched_t,         // row scheduler
//                         BGDesc,          // base graph descriptor
//                         THREADS_PER_CTA, // threads per block
//                         BLOCKS_PER_SM>   // launch bounds
//                         <<<grdDim, blkDim, SHMEM_SIZE, strm>>>(dec_desc, bg_desc);
//    return CUPHY_STATUS_SUCCESS;
//}
//
//
//} // namespace ldpc2

#endif // !defined(LDPC2_DESC_CUH_INCLUDED_)
