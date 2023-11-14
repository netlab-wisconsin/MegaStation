/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_C2V_CACHE_SHARED_STRIDED_CUH_INCLUDED_)
#define LDPC2_C2V_CACHE_SHARED_STRIDED_CUH_INCLUDED_

#include "ldpc2.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// c2v_cache_shared_strided
// Check to variable (C2V) messages are stored in shared memory.
// Assumes that all APP variables are in shared memory. Cached C2V data
// will be stored "after" the APP data.
// BG:              Base graph (1 or 2)
// NUM_SMEM_VNODES: Number of APP variable nodes in shared memory
// TC2V:            Check to variable node class
//
// This storage approach spreads the words of a c2V storage structure
// over multiple columns (assuming a column-major interpretation of the
// data).
//                    |<---  C2V data spread over multiple columns -->|
//          c2V "word":     0          1            2     ...
// ---------------------------------------------------------------------
// parity node 0      |     0    |    mb*Z    |    <-- offset from start
//      0      1      |     1    |  mb*Z + 1  |
//             2      |     2    |  mb*Z + 2  |
//             ...    |     |    |            |
//             Z-2    |     |    |            |
//             Z-1    |     v    |
// - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
// parity node 0      |     Z    | (mb+1)*Z   |
//      1      1      |   Z + 1  |(mb+1)*Z +1 |
//             2      |   Z + 2  |    ...     |
//             ...    |          |            |
//             Z-2    |          |            |
//             Z-1    |          |            |
//
// The distance (in words) between two words in the same row is given
// by (num_parity_nodes * Z)
// This layout avoid bank conflicts when loading c2v data. For BG1, it
// can also be used to slightly reduce shared memory requirements. The
// row degree of each of the first 4 nodes in BG1 is 19. The maximum
// row degree for the remaining rows is 10. If the c2v storage structure
// separates the sign values for columns after the 10th column (per row),
// then we can avoid storing (and loading) those values for parity nodes
// after the first 4.
//
//                                         sign bits for parity
//                                         nodes with row_degree > 10
//   0    [ word 0 ] [ word 1] [ word 2 ] [ word 3 ]
//   1    [ word 0 ] [ word 1] [ word 2 ] [ word 3 ]
//   2    [ word 0 ] [ word 1] [ word 2 ] [ word 3 ]
//   3    [ word 0 ] [ word 1] [ word 2 ] [ word 3 ]
//   4    [ word 0 ] [ word 1] [ word 2 ]
//   5    [ word 0 ] [ word 1] [ word 2 ]
//   6    [ word 0 ] [ word 1] [ word 2 ]
//   7    [ word 0 ] [ word 1] [ word 2 ]
//   8    [ word 0 ] [ word 1] [ word 2 ]
// ...

template <int BG,
          int Z,
          int NUM_SMEM_VNODES,
          class TC2V>
struct c2v_cache_shared_strided
{
    //------------------------------------------------------------------
    // C2V message type
    typedef TC2V                  c2v_t;
    typedef typename c2v_t::app_t app_t;
    //------------------------------------------------------------------
    // shared_byte_offset()
    __device__
    constexpr int shared_byte_offset()
    {
        return (NUM_SMEM_VNODES * Z * sizeof(app_t));
    }
    //------------------------------------------------------------------
    // shared_byte_offset()
    __device__
    constexpr int shared_byte_stride()
    {
        // the distance (in words) between two words is
        // (num_parity_nodes * Z).
        return ((NUM_SMEM_VNODES - max_info_nodes<BG>::value) * Z * sizeof(word_t));
    }
    //------------------------------------------------------------------
    template <int CHECK_IDX>
    __device__
    void process_row_init(const LDPC_kernel_params& params,
                          word_t                    (&app)[app_num_words<app_t, BG, CHECK_IDX>::value],
                          int                       (&app_addr)[row_degree<BG, CHECK_IDX>::value])
    {
        // Assuming all APP data in shared memory, so use max_parity_nodes<> here
        c2v_.process_row_init<CHECK_IDX, max_parity_nodes<BG>::value>(params, app, app_addr);
        c2v_.store_shared_strided(params,
                                  CHECK_IDX,
                                  shared_byte_offset() + (CHECK_IDX * Z * sizeof(word_t)) + (threadIdx.x * sizeof(word_t)),
                                  shared_byte_stride());
    }
    //------------------------------------------------------------------
    template <int CHECK_IDX>
    __device__
    void process_row(const LDPC_kernel_params& params,
                     word_t                   (&app)[app_num_words<app_t, BG, CHECK_IDX>::value],
                     int                      (&app_addr)[row_degree<BG, CHECK_IDX>::value])
    {
        c2v_.load_shared_strided(params,
                                 CHECK_IDX,
                                 shared_byte_offset() + (CHECK_IDX * Z * sizeof(word_t)) + (threadIdx.x * sizeof(word_t)),
                                 shared_byte_stride());
        // Assuming all APP data in shared memory, so use max_parity_nodes<> here
        c2v_.process_row<CHECK_IDX, max_parity_nodes<BG>::value>(params, app, app_addr);
        c2v_.store_shared_strided(params,
                                  CHECK_IDX,
                                  shared_byte_offset() + (CHECK_IDX * Z * sizeof(word_t)) + (threadIdx.x * sizeof(word_t)),
                                  shared_byte_stride());
    }
    //------------------------------------------------------------------
    // get_c2v_size_bytes()
    // Return the size, in bytes, required to store the given number of
    // parity nodes.
    static
    uint32_t get_c2v_size_bytes(int num_parity_nodes)
    {
        // The first 4 parity nodes require additional storage because
        // the row degree is much larger.
        return (Z * ( (                    4  * sizeof(typename c2v_t::c2v_storage_t::c2v_large_t)) +
                      ((num_parity_nodes - 4) * sizeof(typename c2v_t::c2v_storage_t::c2v_small_t))
                    )
               );
    }
    //------------------------------------------------------------------
    // Data
    c2v_t c2v_;
};

} // namespace ldpc2

#endif // !defined(LDPC2_C2V_CACHE_SHARED_STRIDED_CUH_INCLUDED_)