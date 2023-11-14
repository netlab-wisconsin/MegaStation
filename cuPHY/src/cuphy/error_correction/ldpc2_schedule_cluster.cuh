/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_SCHEDULE_CLUSTER_CUH_INCLUDED_)
#define LDPC2_SCHEDULE_CLUSTER_CUH_INCLUDED_

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// ldpc_schedule_cluster_base
// Schedule class that processes more than one row (between calls to
// __syncthreads()) when those rows do not have overlapping variable
// node outputs.
// Derived class, specialized on the base graph, will perform the loop
// structure, whereas this class provides the invocation of the address
// and C2V processing.
template <class TAPPLoc,
          class TC2VCache>
struct ldpc_schedule_cluster_base
{
    typedef TAPPLoc                   app_loc_t;
    typedef TC2VCache                 c2v_cache_t;
    typedef typename TC2VCache::app_t app_t;
    static const int BG = c2v_cache_t::BG;
    //------------------------------------------------------------------
    // ldpc_schedule_cluster_base()
    __device__
    ldpc_schedule_cluster_base(const LDPC_kernel_params& p) : params(p) {}
    //------------------------------------------------------------------
    // process_row()
    // Call the appropriate function for the C2V cache member (either
    // process_row_init() or process_row(), depending on whether this is
    // the first iteration or not).
    template <int CHECK_IDX, bool IS_FIRST_ITER>
    __device__ __forceinline__
    void process_row()
    {
        if(thread_is_active<TAPPLoc::Z>::value())
        {
            int    app_addr[row_degree<BG, CHECK_IDX>::value];      // shared memory (byte) addresses
            word_t app[app_num_words<app_t, BG, CHECK_IDX>::value]; // APP values

            // Generate APP locations/address
            app_loc.template generate<CHECK_IDX>(app_addr);
            // Process the C2V message for this row
            if(IS_FIRST_ITER)
            {
                c2v_cache.process_row_init<CHECK_IDX>(params, app, app_addr);
            }
            else
            {
                c2v_cache.process_row<CHECK_IDX>(params, app, app_addr);
            }
        }
        //__syncthreads();
    }
    //------------------------------------------------------------------
    // Data
    app_loc_t                 app_loc;
    c2v_cache_t               c2v_cache;
    const LDPC_kernel_params& params;
};

#define LDPC2_SYNC(row)        __syncthreads()
#define LDPC2_SYNC_CHECK(row)  __syncthreads(); if((row + 1) == params.num_parity_nodes) return
#define LDPC2_REG_BARRIER(row) if((row + 1) == params.num_parity_nodes) return
#define LDPC2_NO_SYNC(row)

////////////////////////////////////////////////////////////////////////
// row_schedule
// Templated class to provide the interface for the row schedule for a
// fixed base graph/num parity nodes combination.
template <int BG, int NUM_PARITY_NODES> struct row_schedule
{
    template <class TRowProcessor, class TIsFirst>
    __device__ static void process_rows(TRowProcessor&            proc,
                                        const TIsFirst&           isFirst);
};

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 10 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 10>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0, TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1, TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2, TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3, TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4, TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4); // avoid reg spill
    proc.template process_row<5, TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6, TIsFirst::value>(); LDPC2_SYNC(6);
    proc.template process_row<7, TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8, TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9, TIsFirst::value>(); LDPC2_SYNC(9);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 11 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 11>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4); // avoid reg spill
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 12 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 12>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4); // avoid reg spill
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 13 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 13>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4); // avoid reg spill
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 14 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 14>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4); // avoid reg spill
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 15 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 15>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4); // avoid reg spill
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_SYNC(14);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 16 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 16>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4); // avoid reg spill
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_SYNC(14);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 17 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 17>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4); // avoid reg spill
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 18 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 18>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_SYNC(14);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_NO_SYNC(16);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_SYNC(17);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 19 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 19>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 20 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 20>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 21 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 21>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_SYNC(20);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 22 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 22>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 23 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 23>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_SYNC(22);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 24 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 24>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 25 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 25>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_SYNC(24);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 26 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 26>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_NO_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 27 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 27>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_NO_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
    proc.template process_row<26, TIsFirst::value>(); LDPC2_SYNC(26);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 28 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 28>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_NO_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
    proc.template process_row<26, TIsFirst::value>(); LDPC2_NO_SYNC(26);
    proc.template process_row<27, TIsFirst::value>(); LDPC2_SYNC(27);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 29 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 29>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_NO_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
    proc.template process_row<26, TIsFirst::value>(); LDPC2_NO_SYNC(26);
    proc.template process_row<27, TIsFirst::value>(); LDPC2_SYNC(27);
    proc.template process_row<28, TIsFirst::value>(); LDPC2_SYNC(28);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 30 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 30>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_NO_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
    proc.template process_row<26, TIsFirst::value>(); LDPC2_NO_SYNC(26);
    proc.template process_row<27, TIsFirst::value>(); LDPC2_SYNC(27);
    proc.template process_row<28, TIsFirst::value>(); LDPC2_NO_SYNC(28);
    proc.template process_row<29, TIsFirst::value>(); LDPC2_SYNC(29);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 31 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 31>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_NO_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
    proc.template process_row<26, TIsFirst::value>(); LDPC2_NO_SYNC(26);
    proc.template process_row<27, TIsFirst::value>(); LDPC2_SYNC(27);
    proc.template process_row<28, TIsFirst::value>(); LDPC2_NO_SYNC(28);
    proc.template process_row<29, TIsFirst::value>(); LDPC2_SYNC(29);
    proc.template process_row<30, TIsFirst::value>(); LDPC2_SYNC(30);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 32 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 32>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_NO_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
    proc.template process_row<26, TIsFirst::value>(); LDPC2_NO_SYNC(26);
    proc.template process_row<27, TIsFirst::value>(); LDPC2_SYNC(27);
    proc.template process_row<28, TIsFirst::value>(); LDPC2_NO_SYNC(28);
    proc.template process_row<29, TIsFirst::value>(); LDPC2_SYNC(29);
    proc.template process_row<30, TIsFirst::value>(); LDPC2_NO_SYNC(30);
    proc.template process_row<31, TIsFirst::value>(); LDPC2_SYNC(31);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 33 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 33>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_NO_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
    proc.template process_row<26, TIsFirst::value>(); LDPC2_NO_SYNC(26);
    proc.template process_row<27, TIsFirst::value>(); LDPC2_SYNC(27);
    proc.template process_row<28, TIsFirst::value>(); LDPC2_NO_SYNC(28);
    proc.template process_row<29, TIsFirst::value>(); LDPC2_SYNC(29);
    proc.template process_row<30, TIsFirst::value>(); LDPC2_NO_SYNC(30);
    proc.template process_row<31, TIsFirst::value>(); LDPC2_SYNC(31);
    proc.template process_row<32, TIsFirst::value>(); LDPC2_SYNC(32);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 34 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 34>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_NO_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
    proc.template process_row<26, TIsFirst::value>(); LDPC2_NO_SYNC(26);
    proc.template process_row<27, TIsFirst::value>(); LDPC2_SYNC(27);
    proc.template process_row<28, TIsFirst::value>(); LDPC2_NO_SYNC(28);
    proc.template process_row<29, TIsFirst::value>(); LDPC2_SYNC(29);
    proc.template process_row<30, TIsFirst::value>(); LDPC2_NO_SYNC(30);
    proc.template process_row<31, TIsFirst::value>(); LDPC2_SYNC(31);
    proc.template process_row<32, TIsFirst::value>(); LDPC2_NO_SYNC(32);
    proc.template process_row<33, TIsFirst::value>(); LDPC2_SYNC(33);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 35 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 35>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_NO_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
    proc.template process_row<26, TIsFirst::value>(); LDPC2_NO_SYNC(26);
    proc.template process_row<27, TIsFirst::value>(); LDPC2_SYNC(27);
    proc.template process_row<28, TIsFirst::value>(); LDPC2_NO_SYNC(28);
    proc.template process_row<29, TIsFirst::value>(); LDPC2_SYNC(29);
    proc.template process_row<30, TIsFirst::value>(); LDPC2_NO_SYNC(30);
    proc.template process_row<31, TIsFirst::value>(); LDPC2_SYNC(31);
    proc.template process_row<32, TIsFirst::value>(); LDPC2_NO_SYNC(32);
    proc.template process_row<33, TIsFirst::value>(); LDPC2_SYNC(33);
    proc.template process_row<34, TIsFirst::value>(); LDPC2_SYNC(34);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 36 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 36>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_NO_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
    proc.template process_row<26, TIsFirst::value>(); LDPC2_NO_SYNC(26);
    proc.template process_row<27, TIsFirst::value>(); LDPC2_SYNC(27);
    proc.template process_row<28, TIsFirst::value>(); LDPC2_NO_SYNC(28);
    proc.template process_row<29, TIsFirst::value>(); LDPC2_SYNC(29);
    proc.template process_row<30, TIsFirst::value>(); LDPC2_NO_SYNC(30);
    proc.template process_row<31, TIsFirst::value>(); LDPC2_SYNC(31);
    proc.template process_row<32, TIsFirst::value>(); LDPC2_NO_SYNC(32);
    proc.template process_row<33, TIsFirst::value>(); LDPC2_SYNC(33);
    proc.template process_row<34, TIsFirst::value>(); LDPC2_NO_SYNC(34);
    proc.template process_row<35, TIsFirst::value>(); LDPC2_SYNC(35);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 37 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 37>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_NO_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
    proc.template process_row<26, TIsFirst::value>(); LDPC2_NO_SYNC(26);
    proc.template process_row<27, TIsFirst::value>(); LDPC2_SYNC(27);
    proc.template process_row<28, TIsFirst::value>(); LDPC2_NO_SYNC(28);
    proc.template process_row<29, TIsFirst::value>(); LDPC2_SYNC(29);
    proc.template process_row<30, TIsFirst::value>(); LDPC2_NO_SYNC(30);
    proc.template process_row<31, TIsFirst::value>(); LDPC2_SYNC(31);
    proc.template process_row<32, TIsFirst::value>(); LDPC2_NO_SYNC(32);
    proc.template process_row<33, TIsFirst::value>(); LDPC2_SYNC(33);
    proc.template process_row<34, TIsFirst::value>(); LDPC2_NO_SYNC(34);
    proc.template process_row<35, TIsFirst::value>(); LDPC2_SYNC(35);
    proc.template process_row<36, TIsFirst::value>(); LDPC2_SYNC(36);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 38 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 38>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_NO_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
    proc.template process_row<26, TIsFirst::value>(); LDPC2_NO_SYNC(26);
    proc.template process_row<27, TIsFirst::value>(); LDPC2_SYNC(27);
    proc.template process_row<28, TIsFirst::value>(); LDPC2_NO_SYNC(28);
    proc.template process_row<29, TIsFirst::value>(); LDPC2_SYNC(29);
    proc.template process_row<30, TIsFirst::value>(); LDPC2_NO_SYNC(30);
    proc.template process_row<31, TIsFirst::value>(); LDPC2_SYNC(31);
    proc.template process_row<32, TIsFirst::value>(); LDPC2_NO_SYNC(32);
    proc.template process_row<33, TIsFirst::value>(); LDPC2_SYNC(33);
    proc.template process_row<34, TIsFirst::value>(); LDPC2_NO_SYNC(34);
    proc.template process_row<35, TIsFirst::value>(); LDPC2_SYNC(35);
    proc.template process_row<36, TIsFirst::value>(); LDPC2_NO_SYNC(36);
    proc.template process_row<37, TIsFirst::value>(); LDPC2_SYNC(37);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 39 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 39>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_NO_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
    proc.template process_row<26, TIsFirst::value>(); LDPC2_NO_SYNC(26);
    proc.template process_row<27, TIsFirst::value>(); LDPC2_SYNC(27);
    proc.template process_row<28, TIsFirst::value>(); LDPC2_NO_SYNC(28);
    proc.template process_row<29, TIsFirst::value>(); LDPC2_SYNC(29);
    proc.template process_row<30, TIsFirst::value>(); LDPC2_NO_SYNC(30);
    proc.template process_row<31, TIsFirst::value>(); LDPC2_SYNC(31);
    proc.template process_row<32, TIsFirst::value>(); LDPC2_NO_SYNC(32);
    proc.template process_row<33, TIsFirst::value>(); LDPC2_SYNC(33);
    proc.template process_row<34, TIsFirst::value>(); LDPC2_NO_SYNC(34);
    proc.template process_row<35, TIsFirst::value>(); LDPC2_SYNC(35);
    proc.template process_row<36, TIsFirst::value>(); LDPC2_NO_SYNC(36);
    proc.template process_row<37, TIsFirst::value>(); LDPC2_SYNC(37);
    proc.template process_row<38, TIsFirst::value>(); LDPC2_SYNC(38);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 40 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 40>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_NO_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
    proc.template process_row<26, TIsFirst::value>(); LDPC2_NO_SYNC(26);
    proc.template process_row<27, TIsFirst::value>(); LDPC2_SYNC(27);
    proc.template process_row<28, TIsFirst::value>(); LDPC2_NO_SYNC(28);
    proc.template process_row<29, TIsFirst::value>(); LDPC2_SYNC(29);
    proc.template process_row<30, TIsFirst::value>(); LDPC2_NO_SYNC(30);
    proc.template process_row<31, TIsFirst::value>(); LDPC2_SYNC(31);
    proc.template process_row<32, TIsFirst::value>(); LDPC2_NO_SYNC(32);
    proc.template process_row<33, TIsFirst::value>(); LDPC2_SYNC(33);
    proc.template process_row<34, TIsFirst::value>(); LDPC2_NO_SYNC(34);
    proc.template process_row<35, TIsFirst::value>(); LDPC2_SYNC(35);
    proc.template process_row<36, TIsFirst::value>(); LDPC2_NO_SYNC(36);
    proc.template process_row<37, TIsFirst::value>(); LDPC2_SYNC(37);
    proc.template process_row<38, TIsFirst::value>(); LDPC2_NO_SYNC(38);
    proc.template process_row<39, TIsFirst::value>(); LDPC2_SYNC(39);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 41 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 41>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_NO_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
    proc.template process_row<26, TIsFirst::value>(); LDPC2_NO_SYNC(26);
    proc.template process_row<27, TIsFirst::value>(); LDPC2_SYNC(27);
    proc.template process_row<28, TIsFirst::value>(); LDPC2_NO_SYNC(28);
    proc.template process_row<29, TIsFirst::value>(); LDPC2_SYNC(29);
    proc.template process_row<30, TIsFirst::value>(); LDPC2_NO_SYNC(30);
    proc.template process_row<31, TIsFirst::value>(); LDPC2_SYNC(31);
    proc.template process_row<32, TIsFirst::value>(); LDPC2_NO_SYNC(32);
    proc.template process_row<33, TIsFirst::value>(); LDPC2_SYNC(33);
    proc.template process_row<34, TIsFirst::value>(); LDPC2_NO_SYNC(34);
    proc.template process_row<35, TIsFirst::value>(); LDPC2_SYNC(35);
    proc.template process_row<36, TIsFirst::value>(); LDPC2_NO_SYNC(36);
    proc.template process_row<37, TIsFirst::value>(); LDPC2_SYNC(37);
    proc.template process_row<38, TIsFirst::value>(); LDPC2_NO_SYNC(38);
    proc.template process_row<39, TIsFirst::value>(); LDPC2_SYNC(39);
    proc.template process_row<40, TIsFirst::value>(); LDPC2_SYNC(40);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 42 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 42>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_NO_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
    proc.template process_row<26, TIsFirst::value>(); LDPC2_NO_SYNC(26);
    proc.template process_row<27, TIsFirst::value>(); LDPC2_SYNC(27);
    proc.template process_row<28, TIsFirst::value>(); LDPC2_NO_SYNC(28);
    proc.template process_row<29, TIsFirst::value>(); LDPC2_SYNC(29);
    proc.template process_row<30, TIsFirst::value>(); LDPC2_NO_SYNC(30);
    proc.template process_row<31, TIsFirst::value>(); LDPC2_SYNC(31);
    proc.template process_row<32, TIsFirst::value>(); LDPC2_NO_SYNC(32);
    proc.template process_row<33, TIsFirst::value>(); LDPC2_SYNC(33);
    proc.template process_row<34, TIsFirst::value>(); LDPC2_NO_SYNC(34);
    proc.template process_row<35, TIsFirst::value>(); LDPC2_SYNC(35);
    proc.template process_row<36, TIsFirst::value>(); LDPC2_NO_SYNC(36);
    proc.template process_row<37, TIsFirst::value>(); LDPC2_SYNC(37);
    proc.template process_row<38, TIsFirst::value>(); LDPC2_NO_SYNC(38);
    proc.template process_row<39, TIsFirst::value>(); LDPC2_SYNC(39);
    proc.template process_row<40, TIsFirst::value>(); LDPC2_NO_SYNC(40);
    proc.template process_row<41, TIsFirst::value>(); LDPC2_SYNC(41);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 43 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 43>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_NO_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
    proc.template process_row<26, TIsFirst::value>(); LDPC2_NO_SYNC(26);
    proc.template process_row<27, TIsFirst::value>(); LDPC2_SYNC(27);
    proc.template process_row<28, TIsFirst::value>(); LDPC2_NO_SYNC(28);
    proc.template process_row<29, TIsFirst::value>(); LDPC2_SYNC(29);
    proc.template process_row<30, TIsFirst::value>(); LDPC2_NO_SYNC(30);
    proc.template process_row<31, TIsFirst::value>(); LDPC2_SYNC(31);
    proc.template process_row<32, TIsFirst::value>(); LDPC2_NO_SYNC(32);
    proc.template process_row<33, TIsFirst::value>(); LDPC2_SYNC(33);
    proc.template process_row<34, TIsFirst::value>(); LDPC2_NO_SYNC(34);
    proc.template process_row<35, TIsFirst::value>(); LDPC2_SYNC(35);
    proc.template process_row<36, TIsFirst::value>(); LDPC2_NO_SYNC(36);
    proc.template process_row<37, TIsFirst::value>(); LDPC2_SYNC(37);
    proc.template process_row<38, TIsFirst::value>(); LDPC2_NO_SYNC(38);
    proc.template process_row<39, TIsFirst::value>(); LDPC2_SYNC(39);
    proc.template process_row<40, TIsFirst::value>(); LDPC2_NO_SYNC(40);
    proc.template process_row<41, TIsFirst::value>(); LDPC2_SYNC(41);
    proc.template process_row<42, TIsFirst::value>(); LDPC2_SYNC(42);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 44 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 44>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_NO_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
    proc.template process_row<26, TIsFirst::value>(); LDPC2_NO_SYNC(26);
    proc.template process_row<27, TIsFirst::value>(); LDPC2_SYNC(27);
    proc.template process_row<28, TIsFirst::value>(); LDPC2_NO_SYNC(28);
    proc.template process_row<29, TIsFirst::value>(); LDPC2_SYNC(29);
    proc.template process_row<30, TIsFirst::value>(); LDPC2_NO_SYNC(30);
    proc.template process_row<31, TIsFirst::value>(); LDPC2_SYNC(31);
    proc.template process_row<32, TIsFirst::value>(); LDPC2_NO_SYNC(32);
    proc.template process_row<33, TIsFirst::value>(); LDPC2_SYNC(33);
    proc.template process_row<34, TIsFirst::value>(); LDPC2_NO_SYNC(34);
    proc.template process_row<35, TIsFirst::value>(); LDPC2_SYNC(35);
    proc.template process_row<36, TIsFirst::value>(); LDPC2_NO_SYNC(36);
    proc.template process_row<37, TIsFirst::value>(); LDPC2_SYNC(37);
    proc.template process_row<38, TIsFirst::value>(); LDPC2_NO_SYNC(38);
    proc.template process_row<39, TIsFirst::value>(); LDPC2_SYNC(39);
    proc.template process_row<40, TIsFirst::value>(); LDPC2_NO_SYNC(40);
    proc.template process_row<41, TIsFirst::value>(); LDPC2_SYNC(41);
    proc.template process_row<42, TIsFirst::value>(); LDPC2_NO_SYNC(42);
    proc.template process_row<43, TIsFirst::value>(); LDPC2_SYNC(43);
}

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 45 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__
void row_schedule<1, 45>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_NO_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
    proc.template process_row<26, TIsFirst::value>(); LDPC2_NO_SYNC(26);
    proc.template process_row<27, TIsFirst::value>(); LDPC2_SYNC(27);
    proc.template process_row<28, TIsFirst::value>(); LDPC2_NO_SYNC(28);
    proc.template process_row<29, TIsFirst::value>(); LDPC2_SYNC(29);
    proc.template process_row<30, TIsFirst::value>(); LDPC2_NO_SYNC(30);
    proc.template process_row<31, TIsFirst::value>(); LDPC2_SYNC(31);
    proc.template process_row<32, TIsFirst::value>(); LDPC2_NO_SYNC(32);
    proc.template process_row<33, TIsFirst::value>(); LDPC2_SYNC(33);
    proc.template process_row<34, TIsFirst::value>(); LDPC2_NO_SYNC(34);
    proc.template process_row<35, TIsFirst::value>(); LDPC2_SYNC(35);
    proc.template process_row<36, TIsFirst::value>(); LDPC2_NO_SYNC(36);
    proc.template process_row<37, TIsFirst::value>(); LDPC2_SYNC(37);
    proc.template process_row<38, TIsFirst::value>(); LDPC2_NO_SYNC(38);
    proc.template process_row<39, TIsFirst::value>(); LDPC2_SYNC(39);
    proc.template process_row<40, TIsFirst::value>(); LDPC2_NO_SYNC(40);
    proc.template process_row<41, TIsFirst::value>(); LDPC2_SYNC(41);
    proc.template process_row<42, TIsFirst::value>(); LDPC2_NO_SYNC(42);
    proc.template process_row<43, TIsFirst::value>(); LDPC2_SYNC(43);
    proc.template process_row<44, TIsFirst::value>(); LDPC2_SYNC(44);
}

#if 1

////////////////////////////////////////////////////////////////////////
// row_schedule specialization for BG 1 with 46 parity nodes
template <>
template<class TRowProcessor, class TIsFirst>
__device__ __forceinline__
void row_schedule<1, 46>::process_rows(TRowProcessor&            proc,
                                       const TIsFirst&           )
{
#if 1
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4); // LDPC2_SYNC_CHECK(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_NO_SYNC(6);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_NO_SYNC(14);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_NO_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_NO_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_NO_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_NO_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
    proc.template process_row<26, TIsFirst::value>(); LDPC2_NO_SYNC(26);
    proc.template process_row<27, TIsFirst::value>(); LDPC2_SYNC(27);
    proc.template process_row<28, TIsFirst::value>(); LDPC2_NO_SYNC(28);
    proc.template process_row<29, TIsFirst::value>(); LDPC2_SYNC(29);
    proc.template process_row<30, TIsFirst::value>(); LDPC2_NO_SYNC(30);
    proc.template process_row<31, TIsFirst::value>(); LDPC2_SYNC(31);
    proc.template process_row<32, TIsFirst::value>(); LDPC2_NO_SYNC(32);
    proc.template process_row<33, TIsFirst::value>(); LDPC2_SYNC(33);
    proc.template process_row<34, TIsFirst::value>(); LDPC2_NO_SYNC(34);
    proc.template process_row<35, TIsFirst::value>(); LDPC2_SYNC(35);
    proc.template process_row<36, TIsFirst::value>(); LDPC2_NO_SYNC(36);
    proc.template process_row<37, TIsFirst::value>(); LDPC2_SYNC(37);
    proc.template process_row<38, TIsFirst::value>(); LDPC2_NO_SYNC(38);
    proc.template process_row<39, TIsFirst::value>(); LDPC2_SYNC(39);
    proc.template process_row<40, TIsFirst::value>(); LDPC2_NO_SYNC(40);
    proc.template process_row<41, TIsFirst::value>(); LDPC2_SYNC(41);
    proc.template process_row<42, TIsFirst::value>(); LDPC2_NO_SYNC(42);
    proc.template process_row<43, TIsFirst::value>(); LDPC2_SYNC(43);
    proc.template process_row<44, TIsFirst::value>(); LDPC2_NO_SYNC(44);
    proc.template process_row<45, TIsFirst::value>(); LDPC2_SYNC(45);
#else
    proc.template process_row<0,  TIsFirst::value>(); LDPC2_SYNC(0);
    proc.template process_row<1,  TIsFirst::value>(); LDPC2_SYNC(1);
    proc.template process_row<2,  TIsFirst::value>(); LDPC2_SYNC(2);
    proc.template process_row<3,  TIsFirst::value>(); LDPC2_SYNC(3);
    proc.template process_row<4,  TIsFirst::value>(); LDPC2_SYNC(4);
    proc.template process_row<5,  TIsFirst::value>(); LDPC2_SYNC(5);
    proc.template process_row<6,  TIsFirst::value>(); LDPC2_SYNC(6);
    proc.template process_row<7,  TIsFirst::value>(); LDPC2_SYNC(7);
    proc.template process_row<8,  TIsFirst::value>(); LDPC2_SYNC(8);
    proc.template process_row<9,  TIsFirst::value>(); LDPC2_SYNC(9);
    proc.template process_row<10, TIsFirst::value>(); LDPC2_SYNC(10);
    proc.template process_row<11, TIsFirst::value>(); LDPC2_SYNC(11);
    proc.template process_row<12, TIsFirst::value>(); LDPC2_SYNC(12);
    proc.template process_row<13, TIsFirst::value>(); LDPC2_SYNC(13);
    proc.template process_row<14, TIsFirst::value>(); LDPC2_SYNC(14);
    proc.template process_row<15, TIsFirst::value>(); LDPC2_SYNC(15);
    proc.template process_row<16, TIsFirst::value>(); LDPC2_SYNC(16);
    proc.template process_row<17, TIsFirst::value>(); LDPC2_SYNC(17);
    proc.template process_row<18, TIsFirst::value>(); LDPC2_SYNC(18);
    proc.template process_row<19, TIsFirst::value>(); LDPC2_SYNC(19);
    proc.template process_row<20, TIsFirst::value>(); LDPC2_SYNC(20);
    proc.template process_row<21, TIsFirst::value>(); LDPC2_SYNC(21);
    proc.template process_row<22, TIsFirst::value>(); LDPC2_SYNC(22);
    proc.template process_row<23, TIsFirst::value>(); LDPC2_SYNC(23);
    proc.template process_row<24, TIsFirst::value>(); LDPC2_SYNC(24);
    proc.template process_row<25, TIsFirst::value>(); LDPC2_SYNC(25);
    proc.template process_row<26, TIsFirst::value>(); LDPC2_SYNC(26);
    proc.template process_row<27, TIsFirst::value>(); LDPC2_SYNC(27);
    proc.template process_row<28, TIsFirst::value>(); LDPC2_SYNC(28);
    proc.template process_row<29, TIsFirst::value>(); LDPC2_SYNC(29);
    proc.template process_row<30, TIsFirst::value>(); LDPC2_SYNC(30);
    proc.template process_row<31, TIsFirst::value>(); LDPC2_SYNC(31);
    proc.template process_row<32, TIsFirst::value>(); LDPC2_SYNC(32);
    proc.template process_row<33, TIsFirst::value>(); LDPC2_SYNC(33);
    proc.template process_row<34, TIsFirst::value>(); LDPC2_SYNC(34);
    proc.template process_row<35, TIsFirst::value>(); LDPC2_SYNC(35);
    proc.template process_row<36, TIsFirst::value>(); LDPC2_SYNC(36);
    proc.template process_row<37, TIsFirst::value>(); LDPC2_SYNC(37);
    proc.template process_row<38, TIsFirst::value>(); LDPC2_SYNC(38);
    proc.template process_row<39, TIsFirst::value>(); LDPC2_SYNC(39);
    proc.template process_row<40, TIsFirst::value>(); LDPC2_SYNC(40);
    proc.template process_row<41, TIsFirst::value>(); LDPC2_SYNC(41);
    proc.template process_row<42, TIsFirst::value>(); LDPC2_SYNC(42);
    proc.template process_row<43, TIsFirst::value>(); LDPC2_SYNC(43);
    proc.template process_row<44, TIsFirst::value>(); LDPC2_SYNC(44);
    proc.template process_row<45, TIsFirst::value>(); LDPC2_SYNC(45);
#endif
}

#else

#include "ldpc2_cluster_schedule_gen.cuh"

#endif

#include "ldpc2_cluster_schedule_gen_bg2.cuh"

////////////////////////////////////////////////////////////////////////
// ldpc_schedule_cluster
template <int BG,
          class TAPPLoc,
          class TC2VCache,
          int NUM_PARITY>
struct ldpc_schedule_cluster : ldpc_schedule_cluster_base<TAPPLoc, TC2VCache>
{
    typedef ldpc_schedule_cluster_base<TAPPLoc, TC2VCache> inherited;
    typedef typename TC2VCache::app_t                      app_t;
    typedef row_schedule<BG, NUM_PARITY>                   row_sched_t;
    //------------------------------------------------------------------
    __device__
    ldpc_schedule_cluster(const LDPC_kernel_params& p) : inherited(p) {}
    //------------------------------------------------------------------
    // do_first_iteration()
    __device__
    void do_first_iteration()
    {
        row_sched_t::process_rows(*this, std::true_type());
    }
    //------------------------------------------------------------------
    // do_iteration()
    __device__
    void do_iteration()
    {
        row_sched_t::process_rows(*this, std::false_type());
    }
};

} // namespace ldpc2

#endif // !defined(LDPC2_SCHEDULE_CLUSTER_CUH_INCLUDED_)
