/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_C2V_CACHE_GLOBAL_CUH_INCLUDED_)
#define LDPC2_C2V_CACHE_GLOBAL_CUH_INCLUDED_

#include "ldpc2.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// c2v_cache_global
// Check to variable (C2V) messages are stored in global memory
//
// BG: Base graph (1 or 2)
// NUM_SMEM_APP_CHECK_NODES: Number of check nodes with APP data in SHMEM
// TC2V: C2V type
//
// NUM_SMEM_APP_CHECK_NODES is used by the C2V class to determine whether
// to load APP data from shared or global memory. The value can be larger
// than the actual number of parity check nodes being used.
template <int   BG_,
          int   NUM_SMEM_APP_CHECK_NODES,
          class TC2V>
struct c2v_cache_global
{
    //------------------------------------------------------------------
    // C2V message type
    typedef TC2V                  c2v_t;
    typedef typename c2v_t::app_t app_t;
    static const int BG = BG_;
    //------------------------------------------------------------------
    template <int CHECK_IDX>
    __device__
    void process_row_init(const LDPC_kernel_params& params,
                          word_t                    (&app)[app_num_words<app_t, BG, CHECK_IDX>::value],
                          int                       (&app_addr)[row_degree<BG, CHECK_IDX>::value])

    {
        c2v_.process_row_init<CHECK_IDX, NUM_SMEM_APP_CHECK_NODES>(params, app, app_addr);
        
        //c2v_.store_global<CHECK_IDX>(params);
        c2v_.store_global(params, CHECK_IDX);
    }
    //------------------------------------------------------------------
    template <int CHECK_IDX>
    __device__
    void process_row(const LDPC_kernel_params& params,
                     word_t                    (&app)[app_num_words<app_t, BG_, CHECK_IDX>::value],
                     int                       (&app_addr)[row_degree<BG_, CHECK_IDX>::value])
    {
        //c2v_.load_global<CHECK_IDX>(params);
        c2v_.load_global(params, CHECK_IDX);
        
        c2v_.process_row<CHECK_IDX, NUM_SMEM_APP_CHECK_NODES>(params, app, app_addr);
        
        //c2v_.store_global<CHECK_IDX>(params);
        c2v_.store_global(params, CHECK_IDX);
    }
    //------------------------------------------------------------------
    // Data
    c2v_t c2v_;
};

} // namespace ldpc2

#endif // !defined(LDPC2_C2V_CACHE_GLOBAL_CUH_INCLUDED_)