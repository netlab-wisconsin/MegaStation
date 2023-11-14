/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_KERNEL_CUH_INCLUDED_)
#define LDPC2_KERNEL_CUH_INCLUDED_

#include "ldpc2_dec_output.cuh"
#include "ldpc2_llr_loader.cuh"

//namespace ldpc2
//{
//
//template <typename T,        // LLR data type (float, __half, ...)
//          int BG,            // Base graph (1 or 2>
//          int Kb,            // Number of information nodes (22 for BG1, 6/8/9/10 for BG2)
//          int Z,             // Lifting size
//          class TSched,      // Schedule
//          class TLoader,     // LLR loader
//          int BLOCKS_PER_SM>
//__global__ __launch_bounds__(round_up_t<Z, 32>::value, BLOCKS_PER_SM)
//void ldpc2_kernel(LDPC_kernel_params params)
//{
//    static_assert(Kb <= max_info_nodes<BG>::value,
//                  "Number of info nodes greater than maximum");
//    //static_assert(V_SHARED <= (C + Kb),
//    //              "Number of APP variables nodes greater than expected");
//    //------------------------------------------------------------------
//    // LLR loader (used to load input from global to shared memory)
//    typedef TLoader  llr_loader_t;
//    typedef typename llr_loader_t::app_buf_t  app_buf_t;
//    typedef typename llr_loader_t::app_elem_t app_elem_t;
//    //------------------------------------------------------------------
//    // Shared memory
//    extern __shared__ char smem[];
//    //app_buf_t*             app_smem = reinterpret_cast<app_buf_t*>(smem);
//    const T*                 app_smem = reinterpret_cast<const T*>(smem);
//    //------------------------------------------------------------------
//    // Load LLR data from global to shared memory
//    llr_loader_t::load_sync(smem, params);
//    //------------------------------------------------------------------
//    // First iteration (no previous C2V values)
//    TSched ldpc_sched(params);
//    ldpc_sched.do_first_iteration();
//    //------------------------------------------------------------------
//    // Iterations 1 through (N-1)
//    for(int iter = 1; iter < params.max_iterations; ++iter)
//    {
//        ldpc_sched.do_iteration();
//    }
//
//#if 0
//    print_array_sync("APP", app_smem, params.num_var_nodes * Z);
//#endif
//    //------------------------------------------------------------------
//    // Write output based on APP values
//    ldpc_dec_output_variable(params, app_smem);
//}
//
//} // namespace ldpc2

#endif // !defined(LDPC2_KERNEL_CUH_INCLUDED_)

