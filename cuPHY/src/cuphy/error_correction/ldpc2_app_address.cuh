/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_APP_ADDRESS_CUH_INCLUDED_)
#define LDPC2_APP_ADDRESS_CUH_INCLUDED_

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// app_address()
// Calculate the BYTE ADDRESS (in shared memory)  of the APP value that
// maps to a thread.
// --base offset---->|
//     <---shift---->|
//     | - - - - - - - - - - - - - - - - - - - - |
//  0  |             x                           |
//  1  |               x                         |
//  2  |                 x                       |
//  3  |                   x                     |
//  4  |                     x                   |
//  5  |                       x                 |
//  6  |                         x               |
//  7  |                           x             |
//  8  |                             x           |
//  9  |                               x         |
// 10  |                                 x       |
// 11  |                                   x     |
// 12  |                                     x   |
// 13  |                                       x |
// 14  | x.......................................|<-- wrap index
// 15  |   x                                     |
// 16  |     x                                   |
// 17  |       x                                 |
// 18  |         x                               |
// 19  |           x                             |
//     | - - - - - - - - - - - - - - - - - - - - |
//
template <typename T, int Z, int COL_INDEX, int SHIFT>
inline __device__ int app_address(const LDPC_kernel_params& params)
{
    constexpr int BASE_OFFSET = ((COL_INDEX * Z) + (SHIFT % Z)) * sizeof(T);
    constexpr int WRAP_INDEX  = (Z - (SHIFT % Z));
    int idx                   =  BASE_OFFSET + (threadIdx.x * sizeof(T));
    if(threadIdx.x >= WRAP_INDEX)
    {
        idx -= (Z * sizeof(T));
        //idx -= params.z4;
    }
    return idx;
}

template <typename T, int Z, int COL_INDEX, int SHIFT>
inline __device__ int app_address()
{
    constexpr int BASE_OFFSET = ((COL_INDEX * Z) + (SHIFT % Z)) * sizeof(T);
    constexpr int WRAP_INDEX  = (Z - (SHIFT % Z));
    int idx                   =  BASE_OFFSET + (threadIdx.x * sizeof(T));
    if(threadIdx.x >= WRAP_INDEX)
    {
        idx -= (Z * sizeof(T));
        //idx -= params.z4;
    }
    return idx;
}

////////////////////////////////////////////////////////////////////////
// app_address_unroll
template <typename T, int BG, int Z, int ILS, int CHECK_INDEX, int COUNT> struct app_address_unroll;
template <typename T, int BG, int Z, int ILS, int CHECK_INDEX>
struct app_address_unroll<T, BG, Z, ILS, CHECK_INDEX, 1>
{
    __device__
    static void generate(int (&app_addr)[row_degree<BG, CHECK_INDEX>::value])
    {
        app_addr[0]  = app_address<T, Z, vnode_index<BG, CHECK_INDEX, 0>::value,  vnode_shift<BG, ILS, CHECK_INDEX,  0>::value>();
    }
};

template <typename T, int BG, int Z, int ILS, int CHECK_INDEX, int COUNT>
struct app_address_unroll
{
    __device__
    static void generate(int (&app_addr)[row_degree<BG, CHECK_INDEX>::value])
    {
        static_assert(row_degree<BG, CHECK_INDEX>::value >=COUNT, "Loop unroll exceeds the maximum row degree");
        app_address_unroll<T, BG, Z, ILS, CHECK_INDEX, COUNT - 1>::generate(app_addr);
        app_addr[COUNT-1] = app_address<T,
                                        Z,
                                        vnode_index<BG, CHECK_INDEX, COUNT-1>::value,
                                        vnode_shift<BG, ILS, CHECK_INDEX, COUNT-1>::value>();
    }
};

////////////////////////////////////////////////////////////////////////
// app_address_generator
template <typename T, int BG, int Z, int CHECK_INDEX> struct app_address_generator
{
    __device__
    static void generate(int (&app_addr)[row_degree<BG, CHECK_INDEX>::value])
    {
        //--------------------------------------------------------------
        // Define a type for a generator with the number of elements in the row
        typedef app_address_unroll<T, BG, Z, set_index<Z>::value, CHECK_INDEX, row_degree<BG, CHECK_INDEX>::value> generator_t;
        //--------------------------------------------------------------
        // Generate address values
        generator_t::generate(app_addr);
    }
};

////////////////////////////////////////////////////////////////////////
// app_loc_address
// Manager for calculation and storage of APP locations (in shared memory)
// This implementation stores shared memory addresses (as opposed to
// storing INDICES into the APP array). The addresses are stored in
// registers (to avoid recalculation), so use will result in increased
// register pressure.
template <typename T, int BG, int Z_>
struct app_loc_address
{
    static constexpr int Z = Z_;
    //------------------------------------------------------------------
    template <int CHECK_IDX>
    __device__
    void generate(int (&app_addr)[row_degree<BG, CHECK_IDX>::value])
    {
        app_address_generator<T, BG, Z_, CHECK_IDX>::generate(app_addr);
    }
};

} // namespace ldpc2

#endif // !defined(LDPC2_APP_ADDRESS_CUH_INCLUDED_)
