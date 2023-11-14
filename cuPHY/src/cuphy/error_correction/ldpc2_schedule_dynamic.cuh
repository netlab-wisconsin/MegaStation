/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_SCHEDULE_DYNAMIC_CUH_INCLUDED_)
#define LDPC2_SCHEDULE_DYNAMIC_CUH_INCLUDED_

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// ldpc_schedule_dynamic_base
// Schedule class for kernel invocations that will work for more than
// one code rate (i.e. more than one number of parity nodes).
// Derived class, specialized on the base graph, will perform the loop
// structure, whereas this class provides the invocation of the address
// and C2V processing.
template <class TAPPLoc,
          class TC2VCache>
struct ldpc_schedule_dynamic_base
{
    typedef TAPPLoc                   app_loc_t;
    typedef TC2VCache                 c2v_cache_t;
    typedef typename TC2VCache::app_t app_t;
    
    static const int BG = c2v_cache_t::BG;
    //------------------------------------------------------------------
    __device__
    ldpc_schedule_dynamic_base(const LDPC_kernel_params& p) : params(p) {}
    //------------------------------------------------------------------
    // process_first_iter()
    template <int CHECK_IDX>
    __device__
    void process_first_iter()
    {
        if(thread_is_active<TAPPLoc::Z>::value())
        {
            int    app_addr[row_degree<BG, CHECK_IDX>::value];      // shared memory (byte) addresses
            word_t app[app_num_words<app_t, BG, CHECK_IDX>::value]; // APP values

            // Generate APP locations/address
            app_loc.template generate<CHECK_IDX>(app_addr);
            // Process the C2V message for this row
            c2v_cache.process_row_init<CHECK_IDX>(params, app, app_addr);
        }
        __syncthreads();
    }
    //------------------------------------------------------------------
    // process()
    template <int CHECK_IDX>
    __device__
    void process()
    {
        if(thread_is_active<TAPPLoc::Z>::value())
        {
            int    app_addr[row_degree<BG, CHECK_IDX>::value];      // shared memory (byte) addresses
            word_t app[app_num_words<app_t, BG, CHECK_IDX>::value]; // APP values

            // Generate APP locations/address
            app_loc.template generate<CHECK_IDX>(app_addr);
            // Process the C2V message for this row
            c2v_cache.process_row<CHECK_IDX>(params, app, app_addr);
        }
        __syncthreads();
    }
    //------------------------------------------------------------------
    // Data
    app_loc_t                 app_loc;
    c2v_cache_t               c2v_cache;
    const LDPC_kernel_params& params;
};

////////////////////////////////////////////////////////////////////////
// ldpc_schedule_dynamic
template <int BG,
          class TAPPLoc,
          class TC2VCache>
struct ldpc_schedule_dynamic;

// ldpc_schedule_dynamic specialization for base graph 1
template <class TAPPLoc,
          class TC2VCache>
struct ldpc_schedule_dynamic<1,
                             TAPPLoc,
                             TC2VCache> :
    public ldpc_schedule_dynamic_base<TAPPLoc, TC2VCache>
{
    typedef ldpc_schedule_dynamic_base<TAPPLoc, TC2VCache> inherited;
    typedef typename TC2VCache::app_t app_t;
    //------------------------------------------------------------------
    // ldpc_schedule_dynamic()
    __device__
    ldpc_schedule_dynamic(const LDPC_kernel_params& p) : inherited(p) {}
    //------------------------------------------------------------------
    // do_first_iteration()
    __device__
    void do_first_iteration()
    {
        (*this).template process_first_iter<0> ();
        (*this).template process_first_iter<1> ();
        (*this).template process_first_iter<2> ();
        (*this).template process_first_iter<3> (); if(4  == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<4> (); if(5  == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<5> (); if(6  == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<6> (); if(7  == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<7> (); if(8  == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<8> (); if(9  == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<9> (); if(10 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<10>(); if(11 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<11>(); if(12 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<12>(); if(13 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<13>(); if(14 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<14>(); if(15 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<15>(); if(16 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<16>(); if(17 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<17>(); if(18 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<18>(); if(19 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<19>(); if(20 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<20>(); if(21 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<21>(); if(22 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<22>(); if(23 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<23>(); if(24 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<24>(); if(25 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<25>(); if(26 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<26>(); if(27 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<27>(); if(28 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<28>(); if(29 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<29>(); if(30 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<30>(); if(31 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<31>(); if(32 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<32>(); if(33 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<33>(); if(34 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<34>(); if(35 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<35>(); if(36 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<36>(); if(37 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<37>(); if(38 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<38>(); if(39 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<39>(); if(40 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<40>(); if(41 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<41>(); if(42 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<42>(); if(43 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<43>(); if(44 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<44>(); if(45 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<45>();
    }
    //------------------------------------------------------------------
    // do_iteration()
    __device__
    void do_iteration()
    {
        (*this).template process<0> ();
        (*this).template process<1> ();
        (*this).template process<2> ();
        (*this).template process<3> (); if(4  == inherited::params.num_parity_nodes) return;
        (*this).template process<4> (); if(5  == inherited::params.num_parity_nodes) return;
        (*this).template process<5> (); if(6  == inherited::params.num_parity_nodes) return;
        (*this).template process<6> (); if(7  == inherited::params.num_parity_nodes) return;
        (*this).template process<7> (); if(8  == inherited::params.num_parity_nodes) return;
        (*this).template process<8> (); if(9  == inherited::params.num_parity_nodes) return;
        (*this).template process<9> (); if(10 == inherited::params.num_parity_nodes) return;
        (*this).template process<10>(); if(11 == inherited::params.num_parity_nodes) return;
        (*this).template process<11>(); if(12 == inherited::params.num_parity_nodes) return;
        (*this).template process<12>(); if(13 == inherited::params.num_parity_nodes) return;
        (*this).template process<13>(); if(14 == inherited::params.num_parity_nodes) return;
        (*this).template process<14>(); if(15 == inherited::params.num_parity_nodes) return;
        (*this).template process<15>(); if(16 == inherited::params.num_parity_nodes) return;
        (*this).template process<16>(); if(17 == inherited::params.num_parity_nodes) return;
        (*this).template process<17>(); if(18 == inherited::params.num_parity_nodes) return;
        (*this).template process<18>(); if(19 == inherited::params.num_parity_nodes) return;
        (*this).template process<19>(); if(20 == inherited::params.num_parity_nodes) return;
        (*this).template process<20>(); if(21 == inherited::params.num_parity_nodes) return;
        (*this).template process<21>(); if(22 == inherited::params.num_parity_nodes) return;
        (*this).template process<22>(); if(23 == inherited::params.num_parity_nodes) return;
        (*this).template process<23>(); if(24 == inherited::params.num_parity_nodes) return;
        (*this).template process<24>(); if(25 == inherited::params.num_parity_nodes) return;
        (*this).template process<25>(); if(26 == inherited::params.num_parity_nodes) return;
        (*this).template process<26>(); if(27 == inherited::params.num_parity_nodes) return;
        (*this).template process<27>(); if(28 == inherited::params.num_parity_nodes) return;
        (*this).template process<28>(); if(29 == inherited::params.num_parity_nodes) return;
        (*this).template process<29>(); if(30 == inherited::params.num_parity_nodes) return;
        (*this).template process<30>(); if(31 == inherited::params.num_parity_nodes) return;
        (*this).template process<31>(); if(32 == inherited::params.num_parity_nodes) return;
        (*this).template process<32>(); if(33 == inherited::params.num_parity_nodes) return;
        (*this).template process<33>(); if(34 == inherited::params.num_parity_nodes) return;
        (*this).template process<34>(); if(35 == inherited::params.num_parity_nodes) return;
        (*this).template process<35>(); if(36 == inherited::params.num_parity_nodes) return;
        (*this).template process<36>(); if(37 == inherited::params.num_parity_nodes) return;
        (*this).template process<37>(); if(38 == inherited::params.num_parity_nodes) return;
        (*this).template process<38>(); if(39 == inherited::params.num_parity_nodes) return;
        (*this).template process<39>(); if(40 == inherited::params.num_parity_nodes) return;
        (*this).template process<40>(); if(41 == inherited::params.num_parity_nodes) return;
        (*this).template process<41>(); if(42 == inherited::params.num_parity_nodes) return;
        (*this).template process<42>(); if(43 == inherited::params.num_parity_nodes) return;
        (*this).template process<43>(); if(44 == inherited::params.num_parity_nodes) return;
        (*this).template process<44>(); if(45 == inherited::params.num_parity_nodes) return;
        (*this).template process<45>();
    }
};

// ldpc_schedule_dynamic specialization for base graph 2
template <class TAPPLoc,
          class TC2VCache>
struct ldpc_schedule_dynamic<2,
                             TAPPLoc,
                             TC2VCache> :
    public ldpc_schedule_dynamic_base<TAPPLoc, TC2VCache>
{
    typedef ldpc_schedule_dynamic_base<TAPPLoc, TC2VCache> inherited;
    typedef typename TC2VCache::app_t app_t;
    //------------------------------------------------------------------
    // ldpc_schedule_dynamic()
    __device__
    ldpc_schedule_dynamic(const LDPC_kernel_params& p) : inherited(p) {}
    //------------------------------------------------------------------
    // do_first_iteration()
    __device__
    void do_first_iteration()
    {
        (*this).template process_first_iter<0> ();
        (*this).template process_first_iter<1> ();
        (*this).template process_first_iter<2> ();
        (*this).template process_first_iter<3> (); if(4  == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<4> (); if(5  == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<5> (); if(6  == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<6> (); if(7  == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<7> (); if(8  == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<8> (); if(9  == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<9> (); if(10 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<10>(); if(11 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<11>(); if(12 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<12>(); if(13 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<13>(); if(14 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<14>(); if(15 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<15>(); if(16 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<16>(); if(17 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<17>(); if(18 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<18>(); if(19 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<19>(); if(20 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<20>(); if(21 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<21>(); if(22 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<22>(); if(23 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<23>(); if(24 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<24>(); if(25 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<25>(); if(26 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<26>(); if(27 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<27>(); if(28 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<28>(); if(29 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<29>(); if(30 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<30>(); if(31 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<31>(); if(32 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<32>(); if(33 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<33>(); if(34 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<34>(); if(35 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<35>(); if(36 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<36>(); if(37 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<37>(); if(38 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<38>(); if(39 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<39>(); if(40 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<40>(); if(41 == inherited::params.num_parity_nodes) return;
        (*this).template process_first_iter<41>();
    }
    //------------------------------------------------------------------
    // do_iteration()
    __device__
    void do_iteration()
    {
        (*this).template process<0> ();
        (*this).template process<1> ();
        (*this).template process<2> ();
        (*this).template process<3> (); if(4  == inherited::params.num_parity_nodes) return;
        (*this).template process<4> (); if(5  == inherited::params.num_parity_nodes) return;
        (*this).template process<5> (); if(6  == inherited::params.num_parity_nodes) return;
        (*this).template process<6> (); if(7  == inherited::params.num_parity_nodes) return;
        (*this).template process<7> (); if(8  == inherited::params.num_parity_nodes) return;
        (*this).template process<8> (); if(9  == inherited::params.num_parity_nodes) return;
        (*this).template process<9> (); if(10 == inherited::params.num_parity_nodes) return;
        (*this).template process<10>(); if(11 == inherited::params.num_parity_nodes) return;
        (*this).template process<11>(); if(12 == inherited::params.num_parity_nodes) return;
        (*this).template process<12>(); if(13 == inherited::params.num_parity_nodes) return;
        (*this).template process<13>(); if(14 == inherited::params.num_parity_nodes) return;
        (*this).template process<14>(); if(15 == inherited::params.num_parity_nodes) return;
        (*this).template process<15>(); if(16 == inherited::params.num_parity_nodes) return;
        (*this).template process<16>(); if(17 == inherited::params.num_parity_nodes) return;
        (*this).template process<17>(); if(18 == inherited::params.num_parity_nodes) return;
        (*this).template process<18>(); if(19 == inherited::params.num_parity_nodes) return;
        (*this).template process<19>(); if(20 == inherited::params.num_parity_nodes) return;
        (*this).template process<20>(); if(21 == inherited::params.num_parity_nodes) return;
        (*this).template process<21>(); if(22 == inherited::params.num_parity_nodes) return;
        (*this).template process<22>(); if(23 == inherited::params.num_parity_nodes) return;
        (*this).template process<23>(); if(24 == inherited::params.num_parity_nodes) return;
        (*this).template process<24>(); if(25 == inherited::params.num_parity_nodes) return;
        (*this).template process<25>(); if(26 == inherited::params.num_parity_nodes) return;
        (*this).template process<26>(); if(27 == inherited::params.num_parity_nodes) return;
        (*this).template process<27>(); if(28 == inherited::params.num_parity_nodes) return;
        (*this).template process<28>(); if(29 == inherited::params.num_parity_nodes) return;
        (*this).template process<29>(); if(30 == inherited::params.num_parity_nodes) return;
        (*this).template process<30>(); if(31 == inherited::params.num_parity_nodes) return;
        (*this).template process<31>(); if(32 == inherited::params.num_parity_nodes) return;
        (*this).template process<32>(); if(33 == inherited::params.num_parity_nodes) return;
        (*this).template process<33>(); if(34 == inherited::params.num_parity_nodes) return;
        (*this).template process<34>(); if(35 == inherited::params.num_parity_nodes) return;
        (*this).template process<35>(); if(36 == inherited::params.num_parity_nodes) return;
        (*this).template process<36>(); if(37 == inherited::params.num_parity_nodes) return;
        (*this).template process<37>(); if(38 == inherited::params.num_parity_nodes) return;
        (*this).template process<38>(); if(39 == inherited::params.num_parity_nodes) return;
        (*this).template process<39>(); if(40 == inherited::params.num_parity_nodes) return;
        (*this).template process<40>(); if(41 == inherited::params.num_parity_nodes) return;
        (*this).template process<41>();
    }
};


} // namespace ldpc2

#endif // !defined(LDPC2_SCHEDULE_DYNAMIC_CUH_INCLUDED_)
