/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_SCHEDULE_FIXED_CUH_INCLUDED_)
#define LDPC2_SCHEDULE_FIXED_CUH_INCLUDED_

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// check_node_unroll
template <int BG,
          class TAPPLoc,
          class TC2VCache,
          int COUNT,
          int MAX_COUNT> struct check_node_unroll;

// Specialization for the first check node
template <int BG, class TAPPLoc, class TC2VCache, int MAX_COUNT>
struct check_node_unroll<BG, TAPPLoc, TC2VCache, 1, MAX_COUNT>
{
    typedef typename TC2VCache::app_t app_t;
    __device__
    static void process(const LDPC_kernel_params& params,
                        TAPPLoc&                  app_loc,
                        TC2VCache&                c2v_cache)
    {
        if(thread_is_active<TAPPLoc::Z>::value())
        {
            int    app_addr[row_degree<BG, 0>::value];      // shared memory (byte) addresses
            word_t app[app_num_words<app_t, BG, 0>::value]; // APP values

            // Generate APP locations/address
            app_loc.template generate<0>(app_addr);
            // Process the C2V message for this row
            c2v_cache.process_row<0>(params, app, app_addr);
        }
        __syncthreads();
    }
};

template <int BG, class TAPPLoc, class TC2VCache, int COUNT, int MAX_COUNT>
struct check_node_unroll
{
    typedef typename TC2VCache::app_t app_t;
    __device__
    static void process(const LDPC_kernel_params& params,
                        TAPPLoc&                  app_loc,
                        TC2VCache&                c2v_cache)
    {
        // Call the process version for the previous check node
        check_node_unroll<BG, TAPPLoc, TC2VCache, COUNT-1, MAX_COUNT>::process(params,
                                                                               app_loc,
                                                                               c2v_cache);
        if(thread_is_active<TAPPLoc::Z>::value())
        {
            int    app_addr[row_degree<BG, COUNT-1>::value];      // shared memory (byte) addresses
            word_t app[app_num_words<app_t, BG, COUNT-1>::value]; // APP values

            // Generate APP locations/address
            app_loc.template generate<COUNT-1>(app_addr);
            // Process the C2V message for this row
            c2v_cache.process_row<COUNT-1>(params, app, app_addr);
        }
        // Synchronize on shared memory writes
        __syncthreads();
    }
};

////////////////////////////////////////////////////////////////////////
// check_node_unroll_init
template <int BG,
          class TAPPLoc,
          class TC2VCache,
          int COUNT,
          int MAX_COUNT> struct check_node_unroll_init;

// Specialization for the first check node
template <int BG, class TAPPLoc, class TC2VCache, int MAX_COUNT>
struct check_node_unroll_init<BG, TAPPLoc, TC2VCache, 1, MAX_COUNT>
{
    typedef typename TC2VCache::app_t app_t;
    __device__
    static void process(const LDPC_kernel_params& params,
                        TAPPLoc&                  app_loc,
                        TC2VCache&                c2v_cache)
    {
        if(thread_is_active<TAPPLoc::Z>::value())
        {
            int    app_addr[row_degree<BG, 0>::value];      // shared memory (byte) addresses
            word_t app[app_num_words<app_t, BG, 0>::value]; // APP values

            // Generate APP locations/address
            app_loc.template generate<0>(app_addr);
            // Process the C2V message for this row
            c2v_cache.process_row_init<0>(params, app, app_addr);
        }
        __syncthreads();
    }
};

template <int   BG,
          class TAPPLoc,
          class TC2VCache,
          int   COUNT,
          int   MAX_COUNT>
struct check_node_unroll_init
{
    typedef typename TC2VCache::app_t app_t;
    __device__
    static void process(const LDPC_kernel_params& params,
                        TAPPLoc&                  app_loc,
                        TC2VCache&                c2v_cache)
    {
        // Call the process version for the previous check node
        check_node_unroll_init<BG,
                               TAPPLoc,
                               TC2VCache,
                               COUNT-1,
                               MAX_COUNT>::process(params, app_loc, c2v_cache);
        if(thread_is_active<TAPPLoc::Z>::value())
        {
            int    app_addr[row_degree<BG, COUNT-1>::value];      // shared memory (byte) addresses
            word_t app[app_num_words<app_t, BG, COUNT-1>::value]; // APP values
            
            // Generate APP locations/address
            app_loc.template generate<COUNT-1>(app_addr);
            // Process the C2V message for this row
            c2v_cache.process_row_init<COUNT-1>(params, app, app_addr);
        }
        // Synchronize on shared memory writes
        __syncthreads();
    }
};

////////////////////////////////////////////////////////////////////////
// ldpc_schedule_fixed
// Schedule class to perform processing for a FIXED (at compile time)
// number of check nodes.
// BG: Base Graph (1 or 2>
// NUM_CHECK_NODES : Number of parity check nodes (4 to 46 for BG1, ...)
// TAPPLoc: APP location evaluator, determines the offset in APP memory
//          for value used by a given check node, based on the 3GPP spec
//          base graph descriptions
// TC2VCache: Class to manage storage of C2V messages between iterations
template <int BG,
          int NUM_CHECK_NODES,
          class TAPPLoc,
          class TC2VCache>
struct ldpc_schedule_fixed
{
    typedef typename TC2VCache::app_t app_t;
    __device__
    ldpc_schedule_fixed(const LDPC_kernel_params& p) : params(p) {}
    //------------------------------------------------------------------
    // do_first_iteration()
    __device__
    void do_first_iteration()
    {
        //--------------------------------------------------------------
        // Define a type to unroll the loop for the given number of check nodes
        typedef check_node_unroll_init<BG,
                                       TAPPLoc,
                                       TC2VCache,
                                       NUM_CHECK_NODES,
                                       NUM_CHECK_NODES> check_unroll_t;
        //--------------------------------------------------------------
        // Process all check nodes
        check_unroll_t::process(params,
                                app_loc,
                                c2v_cache);

    }
    //------------------------------------------------------------------
    // do_iteration()
    __device__
    void do_iteration()
    {
        //--------------------------------------------------------------
        // Define a type to unroll the loop for the given number of check nodes
        typedef check_node_unroll<BG,
                                  TAPPLoc,
                                  TC2VCache,
                                  NUM_CHECK_NODES,
                                  NUM_CHECK_NODES> check_unroll_t;
        //--------------------------------------------------------------
        // Process all check nodes
        check_unroll_t::process(params,
                                app_loc,
                                c2v_cache);
    }
    //------------------------------------------------------------------
    // Data
    TAPPLoc                   app_loc;
    TC2VCache                 c2v_cache;
    const LDPC_kernel_params& params;
};

} // namespace ldpc2

#endif // !defined(LDPC2_SCHEDULE_FIXED_CUH_INCLUDED_)
