/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC_LOAD_STORE_CUH_INCLUDED_)
#define LDPC_LOAD_STORE_CUH_INCLUDED_

#include "ldpc.cuh"
#include "nrLDPC.cuh"

#include <cooperative_groups.h>

#if(__CUDACC_VER_MAJOR__ >= 11 && (__CUDACC_VER_MINOR__ >= 1))
#include <cooperative_groups/memcpy_async.h>
#endif

namespace cg = cooperative_groups;

namespace ldpc
{
// Split 32-bit data into (n_lo_bits)-bit data and (32-n_lo_bits) data.
// For (n_lo_bits)-bit data, lo_shift decides the bit index it is stored.
// carry_shift and succeeding parameters decide if one of the two data
// must be stored to dst_vector[carry_shift] at carry_shift bit index.
// Bit carrying is introduced to help circular shift in LDPC encoder.
// * dst_vector is assumed to reside in SMEM.
inline __device__ void atomic_bits_scatter2(uint32_t  src_bits,
                                            uint32_t* dst_vector,
                                            int       n_lo_bits,
                                            int       st_index,
                                            int       lo_shift,
                                            int       carry_shift  = 0,
                                            bool      lo_carried   = true,
                                            int       carry_offset = 0)
{
    int      lo_mask = -1 + (1 << n_lo_bits);
    uint32_t lo      = src_bits & lo_mask;
    uint32_t hi      = (src_bits >> n_lo_bits);
    atomicAdd(&dst_vector[st_index], lo << lo_shift);
    atomicAdd(&dst_vector[st_index + 1], hi);
    if(carry_shift > 0)
    {
        int carried_bits = lo_carried ? lo : hi;
        atomicAdd(&dst_vector[st_index + carry_offset], carried_bits << carry_shift);
    }
}

// Merge (n_lo_bits)-bit data and (32-n_lo_bits) data into 32-bit data.
inline __device__ uint32_t bits_gather2(uint32_t* src_vector,
                                        int       n_lo_bits,
                                        int       st_index,
                                        int       lo_shift,
                                        bool      omit_hi = false)
{
    int      lo_mask = (1 << n_lo_bits) - 1;
    uint32_t lo      = (src_vector[st_index] >> lo_shift) & lo_mask;
    uint32_t hi      = omit_hi ? 0 : (src_vector[st_index + 1] << n_lo_bits);
    return hi | lo;
}

// Merge (n_lo_bits)-bit, (n_mi_bits)-bit and remaining bits into 32-bit data.
inline __device__ uint32_t bits_gather3(uint32_t* src_vector,
                                        int       n_lo_bits,
                                        int       n_mi_bits,
                                        int       st_index,
                                        int       lo_shift,
                                        bool      omit_hi = false)
{
    int      lo_mask = (1 << n_lo_bits) - 1;
    int      mi_mask = ((1 << n_mi_bits) - 1) << n_lo_bits;
    uint32_t lo      = (src_vector[st_index] >> lo_shift) & lo_mask;
    uint32_t mi      = (src_vector[st_index + 1] << n_lo_bits) & mi_mask;
    uint32_t hi      = omit_hi ? 0 : (src_vector[st_index + 2] << (n_mi_bits + n_lo_bits));
    return hi | mi | lo;
}

// Split 32-bit data into the three subdata: hi, mi, lo
// For (n_lo_bits)-bit data, lo_shift decides the bit index it is stored.
// primary_carry_shift and succeeding parameters decide if lo or hi bits should be carried.
// must be stored to dst_vector[primary_carry_shift] at primary_carry_shift bit index.
// secondary_carry_shift decides if lo bits must be carried additionally.
// Bit carrying is introduced to help circular shift in LDPC encoder.
// * dst_vector is assumed to reside in SMEM.
inline __device__ void atomic_bits_scatter3(uint32_t  src_bits,
                                            uint32_t* dst_vector,
                                            int       n_lo_bits,
                                            int       n_mi_bits,
                                            int       st_index,
                                            int       lo_shift,
                                            int       primary_carry_shift    = 0,
                                            bool      lo_is_primary_carried  = true,
                                            int       primary_carry_offset   = 0,
                                            int       secondary_carry_shift  = 0,
                                            int       secondary_carry_offset = 0)
{
    int      lo_mask = (1 << n_lo_bits) - 1;
    int      mi_mask = (1 << n_mi_bits) - 1;
    uint32_t lo      = src_bits & lo_mask;
    uint32_t mi      = (src_bits >> n_lo_bits) & mi_mask;
    uint32_t hi      = (src_bits >> (n_mi_bits + n_lo_bits));

    atomicAdd(&dst_vector[st_index], lo << lo_shift);
    atomicAdd(&dst_vector[st_index + 1], mi);
    atomicAdd(&dst_vector[st_index + 2], hi);
    if(primary_carry_shift > 0)
    {
        int carried_bits = lo_is_primary_carried ? lo : hi;
        atomicAdd(&dst_vector[st_index + primary_carry_offset], carried_bits << primary_carry_shift);
    }
    if(secondary_carry_shift > 0)
    {
        atomicAdd(&dst_vector[st_index + secondary_carry_offset], lo << secondary_carry_shift);
    }
}

// TODO: Below are the template specializations for handling load/store between GMEM and SMEM.
// They were introduced to handle lifting sizes whic are not multiples of 32-bit
// They must be refactored to remove a lot of duplicate codes based on the template and structure.

// The default version of load to handle major lifting sizes such as 384, 320, 256 and etc.
// remainder means Z % 32.
template <typename DType, int remainder>
__device__ void load_from_gmem_to_smem(const int         Z,
                                       LDPC_output_t     input,
                                       uint32_t*         info_vec,
                                       int               cb_id,
                                       int               K_in_word,
                                       cg::thread_block& block)
{
#if(__CUDACC_VER_MAJOR__ >= 11 && (__CUDACC_VER_MINOR__ >= 1))
    cg::memcpy_async(block, info_vec, (uint32_t*)(&input({0, cb_id})), sizeof(uint32_t) * K_in_word);
    cg::wait(block);

#else
    for(int i = threadIdx.x; i < K_in_word; i += blockDim.x)
    {
        uint32_t bits = input({i, cb_id});
        info_vec[i]   = bits;
    }
    block.sync();
#endif
}

// The other implementations loads global data in a 32-bit aligned manner.
// Howerver, it splits the data and store them into SMEM including the padding,
// based on atomic_bits_scatter[23] defined above.
// For instance, if Z is 80, the first two 32-bit input data are stored into
// info_vec[0] and info_vec[1]. However, the third input data must be scattered
// info_vec[2] and info_vec[3].
template <>
__device__ void load_from_gmem_to_smem<uint32_t, 4>(const int         Z,
                                                    LDPC_output_t     input,
                                                    uint32_t*         info_vec,
                                                    int               cb_id,
                                                    int               K_in_word,
                                                    cg::thread_block& block)
{
    // only for Z = 36
    const int num_grouped_words = 8 * Z / 32;
    const int num_padded_words  = num_grouped_words + 7;
    const int padded_K_in_word  = (K_in_word * 32 + 16) / num_grouped_words * num_padded_words / 32;

    for(int gi = threadIdx.x; gi < padded_K_in_word; gi += blockDim.x)
    {
        info_vec[gi] = 0;
    }
    block.sync();

    for(int gi = threadIdx.x; gi < K_in_word; gi += blockDim.x)
    {
        uint32_t  bits     = input({gi, cb_id});
        const int local_gi = gi % num_grouped_words;
        const int local_si = local_gi + ((local_gi >= 2) ? (local_gi - 2) : 0);
        const int si       = (gi / num_grouped_words) * num_padded_words + local_si;

        if(local_gi == 0)
        {
            info_vec[si] = bits;
            atomicAdd(&info_vec[si + 1], bits << 4);
        }
        else if(local_gi == 1)
        {
            atomic_bits_scatter2(bits,
                                 info_vec,
                                 4,
                                 si,
                                 0,
                                 4,
                                 false,
                                 2);
        }
        else if(local_gi < 8)
        {
            int n_lo_bits             = (local_gi - 1) * 4;
            int lo_shift              = 32 - n_lo_bits;
            int secondary_carry_shift = lo_shift + 4;
            atomic_bits_scatter3(bits,
                                 info_vec,
                                 n_lo_bits,
                                 4,
                                 si,
                                 lo_shift,
                                 4,
                                 false,
                                 3,
                                 secondary_carry_shift,
                                 1);
        }
        else
        {
            atomic_bits_scatter2(bits,
                                 info_vec,
                                 28,
                                 si,
                                 4,
                                 8,
                                 true,
                                 1);
        }
    }
    block.sync();
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, 8>(const int         Z,
                                                    LDPC_output_t     input,
                                                    uint32_t*         info_vec,
                                                    int               cb_id,
                                                    int               K_in_word,
                                                    cg::thread_block& block)
{
    const int div_point         = Z / 32;
    const int num_grouped_words = Z / 8;
    const int num_padded_words  = num_grouped_words + 3;
    const int padded_K_in_word  = (K_in_word * 32 + 16) / num_grouped_words * num_padded_words / 32;

    for(int gi = threadIdx.x; gi < padded_K_in_word; gi += blockDim.x)
    {
        info_vec[gi] = 0;
    }
    block.sync();

    for(int gi = threadIdx.x; gi < K_in_word; gi += blockDim.x)
    {
        uint32_t  bits     = input({gi, cb_id});
        const int local_gi = gi % num_grouped_words;
        const int local_si = local_gi + (local_gi > div_point * 2) * (local_gi / div_point - 1 - (local_gi % div_point == 0));
        const int si       = (gi / num_grouped_words) * num_padded_words + local_si;
        if(local_gi < div_point)
        {
            info_vec[si] = bits;
            if(local_gi == 0)
            {
                atomicAdd(&info_vec[si + div_point], bits << 8);
            }
        }
        else if(local_gi < div_point * 2)
        {
            atomic_bits_scatter2(bits,
                                 info_vec,
                                 8,
                                 si,
                                 (local_gi == div_point) ? 0 : 24,
                                 (local_gi == div_point) ? 8 : 0,
                                 false,
                                 div_point + 1);
        }
        else if(local_gi == div_point * 2)
        {
            atomic_bits_scatter3(bits,
                                 info_vec,
                                 8,
                                 8,
                                 si,
                                 24,
                                 8,
                                 false,
                                 div_point + 2);
        }
        else if(local_gi < div_point * 3)
        {
            atomic_bits_scatter2(bits,
                                 info_vec,
                                 16,
                                 si,
                                 16,
                                 (local_gi == div_point * 2 + 1) ? 24 : 0,
                                 true,
                                 div_point);
        }
        else if(local_gi == div_point * 3)
        {
            atomic_bits_scatter3(bits,
                                 info_vec,
                                 16,
                                 8,
                                 si,
                                 16,
                                 8,
                                 false,
                                 div_point + 2,
                                 (div_point * 3 == div_point * 2 + 1) ? 24 : 0,
                                 div_point);
        }
        else
        {
            atomic_bits_scatter2(bits,
                                 info_vec,
                                 24,
                                 si,
                                 8,
                                 (local_gi == div_point * 3 + 1) ? 16 : 0,
                                 true,
                                 div_point);
        }
    }
    block.sync();
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, 12>(const int         Z,
                                                     LDPC_output_t     input,
                                                     uint32_t*         info_vec,
                                                     int               cb_id,
                                                     int               K_in_word,
                                                     cg::thread_block& block)
{
    // only for Z = 44
    const int num_grouped_words = 8 * Z / 32;
    const int num_padded_words  = num_grouped_words + 5;
    const int padded_K_in_word  = (K_in_word * 32 + 10) / num_grouped_words * num_padded_words / 32;

    for(int gi = threadIdx.x; gi < padded_K_in_word; gi += blockDim.x)
    {
        info_vec[gi] = 0;
    }
    block.sync();

    for(int gi = threadIdx.x; gi < K_in_word; gi += blockDim.x)
    {
        uint32_t  bits     = input({gi, cb_id});
        const int local_gi = gi % num_grouped_words;
        const int si_base  = (gi / num_grouped_words) * num_padded_words;

        if(local_gi == 0)
        {
            const int si = si_base;
            info_vec[si] = bits;
            atomicAdd(&info_vec[si + 1], bits << 12);
        }
        else if(local_gi == 1 || local_gi == 4 || local_gi == 8)
        {
            const int ii           = (local_gi == 1) ? 0 : ((local_gi == 4) ? 1 : 2);
            const int local_si[3]  = {1, 5, 11};
            const int si           = si_base + local_si[ii];
            const int n_lo_bits[3] = {12, 4, 8};
            const int lo_shift[3]  = {0, 8, 4};

            atomic_bits_scatter2(bits,
                                 info_vec,
                                 n_lo_bits[ii],
                                 si,
                                 lo_shift[ii],
                                 12,
                                 false,
                                 2);
        }
        else if(local_gi == 2 || local_gi == 5 || local_gi == 6 || local_gi == 9)
        {
            const int ii                    = (local_gi == 2) ? 0 : ((local_gi == 5) ? 1 : ((local_gi == 6) ? 2 : 3));
            const int local_si[4]           = {2, 6, 8, 12};
            const int si                    = si_base + local_si[ii];
            const int n_lo_bits[4]          = {12, 4, 16, 8};
            const int secondary_carry_shift = 44 - n_lo_bits[ii];
            atomic_bits_scatter3(bits,
                                 info_vec,
                                 n_lo_bits[ii],
                                 12,
                                 si,
                                 32 - n_lo_bits[ii],
                                 12,
                                 false,
                                 3,
                                 secondary_carry_shift < 32 ? secondary_carry_shift : 0,
                                 1);
        }
        else
        {
            const int ii           = (local_gi == 3) ? 0 : ((local_gi == 7) ? 1 : 2);
            const int local_si[3]  = {4, 10, 14};
            const int si           = si_base + local_si[ii];
            const int n_lo_bits[4] = {24, 28, 20};

            atomic_bits_scatter2(bits,
                                 info_vec,
                                 n_lo_bits[ii],
                                 si,
                                 32 - n_lo_bits[ii],
                                 12 + 32 - n_lo_bits[ii],
                                 true,
                                 1);
        }
    }
    block.sync();
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, 16>(const int         Z,
                                                     LDPC_output_t     input,
                                                     uint32_t*         info_vec,
                                                     int               cb_id,
                                                     int               K_in_word,
                                                     cg::thread_block& block)
{
    const int div_point         = Z / 32;
    const int num_grouped_words = Z / 16;
    const int num_padded_words  = num_grouped_words + 1;
    const int padded_K_in_word  = (K_in_word / num_grouped_words) * num_padded_words;

    for(int gi = threadIdx.x; gi < padded_K_in_word; gi += blockDim.x)
    {
        info_vec[gi] = 0;
    }
    block.sync();

    for(int gi = threadIdx.x; gi < K_in_word; gi += blockDim.x)
    {
        uint32_t  bits     = input({gi, cb_id});
        const int local_si = gi % num_grouped_words;
        const int si       = (gi / num_grouped_words) * num_padded_words + local_si;

        if(local_si < div_point)
        {
            info_vec[si] = bits;
            atomicAdd(&info_vec[si + div_point], (local_si == 0) ? bits << 16 : 0);
        }
        else
        {
            atomic_bits_scatter2(bits,
                                 info_vec,
                                 16,
                                 si,
                                 (local_si == div_point) ? 0 : 16,
                                 (local_si == div_point) ? 16 : 0,
                                 false,
                                 div_point + 1);
        }
    }
    block.sync();
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, 20>(const int         Z,
                                                     LDPC_output_t     input,
                                                     uint32_t*         info_vec,
                                                     int               cb_id,
                                                     int               K_in_word,
                                                     cg::thread_block& block)
{
    // only for Z = 52
    const int num_grouped_words = 8 * Z / 32;
    const int num_padded_words  = num_grouped_words + 3;
    const int padded_K_in_word  = (K_in_word * 32 + 8) / num_grouped_words * num_padded_words / 32;

    for(int gi = threadIdx.x; gi < padded_K_in_word; gi += blockDim.x)
    {
        info_vec[gi] = 0;
    }
    block.sync();

    for(int gi = threadIdx.x; gi < K_in_word; gi += blockDim.x)
    {
        uint32_t  bits     = input({gi, cb_id});
        const int local_gi = gi % num_grouped_words;
        const int si_base  = (gi / num_grouped_words) * num_padded_words;

        if(local_gi == 0)
        {
            const int si = si_base;
            info_vec[si] = bits;
            atomicAdd(&info_vec[si + 1], bits << 20);
        }
        else if(local_gi == 1 || local_gi == 3 || local_gi == 6 || local_gi == 8 || local_gi == 11)
        {
            const int ii           = (local_gi == 1) ? 0 : ((local_gi == 3) ? 1 : ((local_gi == 6) ? 2 : (local_gi == 8) ? 3 : 4));
            const int local_si[5]  = {1, 3, 7, 9, 13};
            const int si           = si_base + local_si[ii];
            const int n_lo_bits[5] = {20, 8, 16, 4, 12};
            const int lo_shift[5]  = {0, 12, 4, 16, 8};

            atomic_bits_scatter2(bits,
                                 info_vec,
                                 n_lo_bits[ii],
                                 si,
                                 lo_shift[ii],
                                 20,
                                 false,
                                 2);
        }
        else if(local_gi == 2 || local_gi == 5 || local_gi == 7 || local_gi == 10 || local_gi == 12)
        {
            const int ii             = (local_gi == 2) ? 0 : ((local_gi == 5) ? 1 : ((local_gi == 7) ? 2 : (local_gi == 10) ? 3 : 4));
            const int local_si[5]    = {2, 6, 8, 12, 14};
            const int si             = si_base + local_si[ii];
            const int n_lo_bits[5]   = {20, 28, 16, 24, 12};
            const int carry_shift[5] = {0, 24, 0, 28, 0};

            atomic_bits_scatter2(bits,
                                 info_vec,
                                 n_lo_bits[ii],
                                 si,
                                 32 - n_lo_bits[ii],
                                 carry_shift[ii],
                                 true,
                                 1);
        }
        else
        {
            const int ii           = (local_gi == 4) ? 0 : 1;
            const int local_si[2]  = {4, 10};
            const int si           = si_base + local_si[ii];
            const int n_lo_bits[2] = {8, 4};

            atomic_bits_scatter3(bits,
                                 info_vec,
                                 n_lo_bits[ii],
                                 20,
                                 si,
                                 32 - n_lo_bits[ii],
                                 20,
                                 false,
                                 3);
        }
    }
    block.sync();
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, 28>(const int         Z,
                                                     LDPC_output_t     input,
                                                     uint32_t*         info_vec,
                                                     int               cb_id,
                                                     int               K_in_word,
                                                     cg::thread_block& block)
{
    const int num_grouped_words = 15;
    const int num_padded_words  = 16;
    const int padded_K_in_word  = (K_in_word * 32) / num_grouped_words * num_padded_words / 32;

    for(int gi = threadIdx.x; gi < padded_K_in_word; gi += blockDim.x)
    {
        info_vec[gi] = 0;
    }
    block.sync();

    for(int gi = threadIdx.x; gi < K_in_word; gi += blockDim.x)
    {
        uint32_t  bits     = input({gi, cb_id});
        const int local_gi = gi % num_grouped_words;
        const int si       = (gi / num_grouped_words) * num_padded_words + local_gi;
        if(local_gi == 0)
        {
            info_vec[si] = bits;
            atomicAdd(&info_vec[si + 1], bits << 28);
        }
        else
        {
            int n_lo_bits = 28 - 4 * ((local_gi - 1) / 2);
            int lo_shift  = 4 * ((local_gi) / 2);
            //int carried_bits = 4;
            int carry_shift  = (local_gi % 2) ? 28 : 0;
            int lo_carried   = false;
            int carry_offset = 2;
            atomic_bits_scatter2(bits,
                                 info_vec,
                                 n_lo_bits,
                                 si,
                                 lo_shift,
                                 carry_shift,
                                 lo_carried,
                                 carry_offset);
        }
    }
    block.sync();
}
template <>
__device__ void load_from_gmem_to_smem<uint32_t, 24>(const int         Z,
                                                     LDPC_output_t     input,
                                                     uint32_t*         info_vec,
                                                     int               cb_id,
                                                     int               K_in_word,
                                                     cg::thread_block& block)
{
    const int div_point1 = Z / 32;
    const int div_point2 = div_point1 + div_point1 + 1;
    const int div_point3 = div_point2 + div_point1 + 1;

    const int num_grouped_words = Z / 8;
    const int num_padded_words  = num_grouped_words + 1;
    const int padded_K_in_word  = (K_in_word * 32 - 16) / num_grouped_words * num_padded_words / 32;

    for(int gi = threadIdx.x; gi < padded_K_in_word; gi += blockDim.x)
    {
        info_vec[gi] = 0;
    }
    block.sync();

    for(int gi = threadIdx.x; gi < K_in_word; gi += blockDim.x)
    {
        uint32_t  bits     = input({gi, cb_id});
        const int local_si = gi % num_grouped_words;
        const int si       = (gi / num_grouped_words) * num_padded_words + local_si;

        if(local_si < div_point1)
        {
            info_vec[si] = bits;
            if(local_si == 0)
            {
                atomicAdd(&info_vec[si + div_point1], bits << 24); // 8 bits
            }
        }
        else if(local_si < div_point2)
        {
            atomic_bits_scatter2(bits,
                                 info_vec,
                                 24,
                                 si,
                                 (local_si == div_point1) ? 0 : 8,
                                 (local_si == div_point1) ? 24 : 0,
                                 false,
                                 div_point1 + 1);
        }
        else if(local_si < div_point3)
        {
            atomic_bits_scatter2(bits,
                                 info_vec,
                                 16,
                                 si,
                                 (local_si == div_point2) ? 8 : 16,
                                 (local_si == div_point2) ? 24 : 0,
                                 false,
                                 div_point1 + 1);
        }
        else
        {
            atomic_bits_scatter2(bits,
                                 info_vec,
                                 8,
                                 si,
                                 (local_si == div_point3) ? 16 : 24,
                                 (local_si == div_point3) ? 24 : 0,
                                 false,
                                 div_point1 + 1);
        }
    }
    block.sync();
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -30>(const int         Z,
                                                      LDPC_output_t     input,
                                                      uint32_t*         info_vec,
                                                      int               cb_id,
                                                      int               K_in_word,
                                                      cg::thread_block& block)
{
    const int num_grouped_words = 15;
    const int num_padded_words  = 16;
    const int padded_K_in_word  = (K_in_word * 32 + 12) / num_grouped_words * num_padded_words / 32;

    for(int gi = threadIdx.x; gi < padded_K_in_word; gi += blockDim.x)
    {
        info_vec[gi] = 0;
    }
    block.sync();

    for(int gi = threadIdx.x; gi < K_in_word; gi += blockDim.x)
    {
        uint32_t  bits     = input({gi, cb_id});
        const int local_si = gi % num_grouped_words;
        const int si       = (gi / num_grouped_words) * num_padded_words + local_si;

        int n_lo_bits = Z - local_si * 2;

        atomic_bits_scatter2(bits,
                             info_vec,
                             n_lo_bits,
                             si,
                             Z - n_lo_bits,
                             Z,
                             false,
                             1);
        if(local_si == 0)
        {
            atomicAdd(&info_vec[si], bits << Z);
        }
    }
    block.sync();
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -28>(const int         Z,
                                                      LDPC_output_t     input,
                                                      uint32_t*         info_vec,
                                                      int               cb_id,
                                                      int               K_in_word,
                                                      cg::thread_block& block)
{
    const int num_grouped_words = 7;
    const int num_padded_words  = 8;
    const int padded_K_in_word  = (K_in_word * 32 + 24) / num_grouped_words * num_padded_words / 32;

    for(int gi = threadIdx.x; gi < padded_K_in_word; gi += blockDim.x)
    {
        info_vec[gi] = 0;
    }
    block.sync();

    for(int gi = threadIdx.x; gi < K_in_word; gi += blockDim.x)
    {
        uint32_t  bits     = input({gi, cb_id});
        const int local_si = gi % num_grouped_words;
        const int si       = (gi / num_grouped_words) * num_padded_words + local_si;

        int n_lo_bits = Z - local_si * 4;

        atomic_bits_scatter2(bits,
                             info_vec,
                             n_lo_bits,
                             si,
                             Z - n_lo_bits,
                             Z,
                             false,
                             1);
        if(local_si == 0)
        {
            atomicAdd(&info_vec[si], bits << Z);
        }
    }
    block.sync();
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -26>(const int         Z,
                                                      LDPC_output_t     input,
                                                      uint32_t*         info_vec,
                                                      int               cb_id,
                                                      int               K_in_word,
                                                      cg::thread_block& block)
{
    const int num_grouped_words = 13;
    const int num_padded_words  = 16;
    const int padded_K_in_word  = (K_in_word * 32 + 4) / num_grouped_words * num_padded_words / 32;

    for(int gi = threadIdx.x; gi < padded_K_in_word; gi += blockDim.x)
    {
        info_vec[gi] = 0;
    }
    block.sync();

    for(int gi = threadIdx.x; gi < K_in_word; gi += blockDim.x)
    {
        uint32_t  bits     = input({gi, cb_id});
        const int local_gi = gi % num_grouped_words;
        const int local_si = local_gi +
                             ((local_gi <= 4) ? 0 : ((local_gi <= 8) ? 1 : 2));
        const int si = (gi / num_grouped_words) * num_padded_words + local_si;

        if(local_gi < 4)
        {
            int n_lo_bits = Z - local_gi * 6;

            atomic_bits_scatter2(bits,
                                 info_vec,
                                 n_lo_bits,
                                 si,
                                 Z - n_lo_bits,
                                 Z,
                                 false,
                                 1);
            if(local_gi == 0)
            {
                atomicAdd(&info_vec[si], bits << Z);
            }
        }
        else if(local_gi == 4 || local_gi == 8)
        {
            int n_lo_bits = (local_gi == 4) ? 2 : 4;
            int n_mi_bits = 26;
            int lo_shift  = (local_gi == 4) ? 24 : 22;
            atomic_bits_scatter3(bits,
                                 info_vec,
                                 n_lo_bits,
                                 n_mi_bits,
                                 si,
                                 lo_shift,
                                 Z,
                                 false,
                                 2);
        }
        else if(local_gi < 8)
        {
            int n_lo_bits = Z - 4 - (local_gi - 5) * 6;

            atomic_bits_scatter2(bits,
                                 info_vec,
                                 n_lo_bits,
                                 si,
                                 Z - n_lo_bits,
                                 Z,
                                 false,
                                 1);
            if(local_gi == 5)
            {
                atomicAdd(&info_vec[si], bits << (Z + 4));
            }
        }
        else
        {
            int n_lo_bits = Z - 2 - (local_gi - 9) * 6;

            atomic_bits_scatter2(bits,
                                 info_vec,
                                 n_lo_bits,
                                 si,
                                 Z - n_lo_bits,
                                 Z,
                                 false,
                                 1);
            if(local_gi == 9)
            {
                atomicAdd(&info_vec[si], bits << (Z + 2));
            }
        }
    }
    block.sync();
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -24>(const int         Z,
                                                      LDPC_output_t     input,
                                                      uint32_t*         info_vec,
                                                      int               cb_id,
                                                      int               K_in_word,
                                                      cg::thread_block& block)
{
    const int num_grouped_words = 3;
    const int num_padded_words  = 4;
    const int padded_K_in_word  = (K_in_word * 32 + 16) / num_grouped_words * num_padded_words / 32;

    for(int gi = threadIdx.x; gi < padded_K_in_word; gi += blockDim.x)
    {
        info_vec[gi] = 0;
    }
    block.sync();

    for(int gi = threadIdx.x; gi < K_in_word; gi += blockDim.x)
    {
        uint32_t  bits     = input({gi, cb_id});
        const int local_si = gi % num_grouped_words;
        const int si       = (gi / num_grouped_words) * num_padded_words + local_si;

        int n_lo_bits = Z - local_si * 8;

        atomic_bits_scatter2(bits,
                             info_vec,
                             n_lo_bits,
                             si,
                             Z - n_lo_bits,
                             Z,
                             false,
                             1);
        if(local_si == 0)
        {
            atomicAdd(&info_vec[si], bits << Z);
        }
    }
    block.sync();
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -22>(const int         Z,
                                                      LDPC_output_t     input,
                                                      uint32_t*         info_vec,
                                                      int               cb_id,
                                                      int               K_in_word,
                                                      cg::thread_block& block)
{
    const int num_grouped_words = 11;
    const int num_padded_words  = 16;
    const int padded_K_in_word  = (K_in_word * 32 + 28) / num_grouped_words * num_padded_words / 32;

    for(int gi = threadIdx.x; gi < padded_K_in_word; gi += blockDim.x)
    {
        info_vec[gi] = 0;
    }
    block.sync();

    for(int gi = threadIdx.x; gi < K_in_word; gi += blockDim.x)
    {
        uint32_t  bits     = input({gi, cb_id});
        const int local_gi = gi % num_grouped_words;
        const int local_si = local_gi +
                             ((local_gi <= 2) ? 0 : ((local_gi <= 4) ? 1 : ((local_gi <= 6) ? 2 : ((local_gi <= 8) ? 3 : 4))));
        const int si = (gi / num_grouped_words) * num_padded_words + local_si;

        if(local_gi == 0)
        {
            int n_lo_bits = Z;

            atomic_bits_scatter2(bits,
                                 info_vec,
                                 n_lo_bits,
                                 si,
                                 0,
                                 Z,
                                 false,
                                 1);
            if(local_gi == 0)
            {
                atomicAdd(&info_vec[si], bits << Z);
            }
        }
        else if(local_gi % 2 == 1 || local_gi == 10)
        {
            int n_lo_bits = (local_gi != 10) ? local_gi + 11 : 10;

            atomic_bits_scatter2(bits,
                                 info_vec,
                                 n_lo_bits,
                                 si,
                                 Z - n_lo_bits,
                                 Z,
                                 false,
                                 1);
        }
        else
        {
            int n_lo_bits = local_gi;
            int n_mi_bits = 22;
            int lo_shift  = Z - n_lo_bits;
            atomic_bits_scatter3(bits,
                                 info_vec,
                                 n_lo_bits,
                                 n_mi_bits,
                                 si,
                                 lo_shift,
                                 Z,
                                 false,
                                 2);
        }
    }
    block.sync();
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -20>(const int         Z,
                                                      LDPC_output_t     input,
                                                      uint32_t*         info_vec,
                                                      int               cb_id,
                                                      int               K_in_word,
                                                      cg::thread_block& block)
{
    const int num_grouped_words = 5;
    const int num_padded_words  = 8;
    const int padded_K_in_word  = (K_in_word * 32 + 8) / num_grouped_words * num_padded_words / 32;

    for(int gi = threadIdx.x; gi < padded_K_in_word; gi += blockDim.x)
    {
        info_vec[gi] = 0;
    }
    block.sync();

    for(int gi = threadIdx.x; gi < K_in_word; gi += blockDim.x)
    {
        uint32_t  bits     = input({gi, cb_id});
        const int local_gi = gi % num_grouped_words;
        const int local_si = local_gi +
                             ((local_gi <= 1) ? 0 : ((local_gi <= 3) ? 1 : 2));
        const int si = (gi / num_grouped_words) * num_padded_words + local_si;

        if(local_gi % 2 == 0)
        {
            int n_lo_bits = Z - 2 * local_gi;

            atomic_bits_scatter2(bits,
                                 info_vec,
                                 n_lo_bits,
                                 si,
                                 Z - n_lo_bits,
                                 Z,
                                 false,
                                 1);
            if(local_gi == 0)
            {
                atomicAdd(&info_vec[si], bits << Z);
            }
        }
        else
        {
            int n_lo_bits = (local_gi == 1) ? 8 : 4;
            int n_mi_bits = 20;
            int lo_shift  = Z - n_lo_bits;
            atomic_bits_scatter3(bits,
                                 info_vec,
                                 n_lo_bits,
                                 n_mi_bits,
                                 si,
                                 lo_shift,
                                 Z,
                                 false,
                                 2);
        }
    }
    block.sync();
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -18>(const int         Z,
                                                      LDPC_output_t     input,
                                                      uint32_t*         info_vec,
                                                      int               cb_id,
                                                      int               K_in_word,
                                                      cg::thread_block& block)
{
    const int num_grouped_words = 9;
    const int num_padded_words  = 16;
    const int padded_K_in_word  = (K_in_word * 32 + 20) / num_grouped_words * num_padded_words / 32;

    for(int gi = threadIdx.x; gi < padded_K_in_word; gi += blockDim.x)
    {
        info_vec[gi] = 0;
    }
    block.sync();

    for(int gi = threadIdx.x; gi < K_in_word; gi += blockDim.x)
    {
        uint32_t  bits     = input({gi, cb_id});
        const int local_gi = gi % num_grouped_words;
        const int local_si = local_gi +
                             ((local_gi <= 1) ? 0 : ((local_gi <= 4) ? local_gi - 1 : local_gi - 2));
        const int si = (gi / num_grouped_words) * num_padded_words + local_si;

        if(local_gi % 4 == 0)
        {
            int n_lo_bits = Z - local_gi / 2;

            atomic_bits_scatter2(bits,
                                 info_vec,
                                 n_lo_bits,
                                 si,
                                 Z - n_lo_bits,
                                 Z,
                                 false,
                                 1);
            if(local_gi == 0)
            {
                atomicAdd(&info_vec[si], bits << Z);
            }
        }
        else
        {
            int n_lo_bits = (local_gi < 4) ? local_gi * 4 : 2 + (local_gi - 5) * 4;
            int n_mi_bits = 18;
            int lo_shift  = Z - n_lo_bits;
            atomic_bits_scatter3(bits,
                                 info_vec,
                                 n_lo_bits,
                                 n_mi_bits,
                                 si,
                                 lo_shift,
                                 Z,
                                 false,
                                 2);
        }
    }
    block.sync();
}

template <int Z>
__device__ void small_load_from_gemm_to_smem(LDPC_output_t     input,
                                             uint32_t*         info_vec,
                                             int               cb_id,
                                             int               K_in_word,
                                             cg::thread_block& block)
{
    if(threadIdx.x == 0)
    {
        int len = (K_in_word * 32 / Z >= 22) ? 22 : 10;
        for(int gi = 0; gi < len; gi++)
        {
            info_vec[gi] = 0;
        }
        int si        = 0;
        int start_pos = 0;
        for(int gi = 0; gi < K_in_word; gi++)
        {
            uint32_t bits    = input({gi, cb_id});
            int      bit_pos = 0;
            while(bit_pos < 32)
            {
                int cur_n_bits = (bit_pos + Z > 32) ? (32 - bit_pos) : (Z - start_pos);
                int mask       = (1 << cur_n_bits) - 1;
                info_vec[si] += ((bits >> bit_pos) & mask) << start_pos;

                for(int offset = Z + start_pos; offset < 32; offset += Z)
                {
                    info_vec[si] += ((bits >> bit_pos) & mask) << offset;
                }

                bit_pos += cur_n_bits;

                if(bit_pos >= 32 && cur_n_bits < Z)
                {
                    start_pos = cur_n_bits;
                }
                else
                {
                    start_pos = 0;
                    si++;
                }
            }
        }
    }
    block.sync();
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -16>(const int         Z,
                                                      LDPC_output_t     input,
                                                      uint32_t*         info_vec,
                                                      int               cb_id,
                                                      int               K_in_word,
                                                      cg::thread_block& block)
{
    small_load_from_gemm_to_smem<16>(input,
                                     info_vec,
                                     cb_id,
                                     K_in_word,
                                     block);
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -15>(const int         Z,
                                                      LDPC_output_t     input,
                                                      uint32_t*         info_vec,
                                                      int               cb_id,
                                                      int               K_in_word,
                                                      cg::thread_block& block)
{
    small_load_from_gemm_to_smem<15>(input,
                                     info_vec,
                                     cb_id,
                                     K_in_word,
                                     block);
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -14>(const int         Z,
                                                      LDPC_output_t     input,
                                                      uint32_t*         info_vec,
                                                      int               cb_id,
                                                      int               K_in_word,
                                                      cg::thread_block& block)
{
    small_load_from_gemm_to_smem<14>(input,
                                     info_vec,
                                     cb_id,
                                     K_in_word,
                                     block);
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -13>(const int         Z,
                                                      LDPC_output_t     input,
                                                      uint32_t*         info_vec,
                                                      int               cb_id,
                                                      int               K_in_word,
                                                      cg::thread_block& block)
{
    small_load_from_gemm_to_smem<13>(input,
                                     info_vec,
                                     cb_id,
                                     K_in_word,
                                     block);
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -12>(const int         Z,
                                                      LDPC_output_t     input,
                                                      uint32_t*         info_vec,
                                                      int               cb_id,
                                                      int               K_in_word,
                                                      cg::thread_block& block)
{
    small_load_from_gemm_to_smem<12>(input,
                                     info_vec,
                                     cb_id,
                                     K_in_word,
                                     block);
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -11>(const int         Z,
                                                      LDPC_output_t     input,
                                                      uint32_t*         info_vec,
                                                      int               cb_id,
                                                      int               K_in_word,
                                                      cg::thread_block& block)
{
    small_load_from_gemm_to_smem<11>(input,
                                     info_vec,
                                     cb_id,
                                     K_in_word,
                                     block);
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -10>(const int         Z,
                                                      LDPC_output_t     input,
                                                      uint32_t*         info_vec,
                                                      int               cb_id,
                                                      int               K_in_word,
                                                      cg::thread_block& block)
{
    small_load_from_gemm_to_smem<10>(input,
                                     info_vec,
                                     cb_id,
                                     K_in_word,
                                     block);
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -9>(const int         Z,
                                                     LDPC_output_t     input,
                                                     uint32_t*         info_vec,
                                                     int               cb_id,
                                                     int               K_in_word,
                                                     cg::thread_block& block)
{
    small_load_from_gemm_to_smem<9>(input,
                                    info_vec,
                                    cb_id,
                                    K_in_word,
                                    block);
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -8>(const int         Z,
                                                     LDPC_output_t     input,
                                                     uint32_t*         info_vec,
                                                     int               cb_id,
                                                     int               K_in_word,
                                                     cg::thread_block& block)
{
    small_load_from_gemm_to_smem<8>(input,
                                    info_vec,
                                    cb_id,
                                    K_in_word,
                                    block);
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -7>(const int         Z,
                                                     LDPC_output_t     input,
                                                     uint32_t*         info_vec,
                                                     int               cb_id,
                                                     int               K_in_word,
                                                     cg::thread_block& block)
{
    small_load_from_gemm_to_smem<7>(input,
                                    info_vec,
                                    cb_id,
                                    K_in_word,
                                    block);
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -6>(const int         Z,
                                                     LDPC_output_t     input,
                                                     uint32_t*         info_vec,
                                                     int               cb_id,
                                                     int               K_in_word,
                                                     cg::thread_block& block)
{
    small_load_from_gemm_to_smem<6>(input,
                                    info_vec,
                                    cb_id,
                                    K_in_word,
                                    block);
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -5>(const int         Z,
                                                     LDPC_output_t     input,
                                                     uint32_t*         info_vec,
                                                     int               cb_id,
                                                     int               K_in_word,
                                                     cg::thread_block& block)
{
    small_load_from_gemm_to_smem<5>(input,
                                    info_vec,
                                    cb_id,
                                    K_in_word,
                                    block);
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -4>(const int         Z,
                                                     LDPC_output_t     input,
                                                     uint32_t*         info_vec,
                                                     int               cb_id,
                                                     int               K_in_word,
                                                     cg::thread_block& block)
{
    small_load_from_gemm_to_smem<4>(input,
                                    info_vec,
                                    cb_id,
                                    K_in_word,
                                    block);
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -3>(const int         Z,
                                                     LDPC_output_t     input,
                                                     uint32_t*         info_vec,
                                                     int               cb_id,
                                                     int               K_in_word,
                                                     cg::thread_block& block)
{
    small_load_from_gemm_to_smem<3>(input,
                                    info_vec,
                                    cb_id,
                                    K_in_word,
                                    block);
}

template <>
__device__ void load_from_gmem_to_smem<uint32_t, -2>(const int         Z,
                                                     LDPC_output_t     input,
                                                     uint32_t*         info_vec,
                                                     int               cb_id,
                                                     int               K_in_word,
                                                     cg::thread_block& block)
{
    small_load_from_gemm_to_smem<2>(input,
                                    info_vec,
                                    cb_id,
                                    K_in_word,
                                    block);
}

template <typename DType, int remainder>
__device__ void store_from_smem_to_gmem(const int         Z,
                                        const int         num_tdbv,
                                        LDPC_output_t     output,
                                        uint32_t*         sbuf,
                                        int               cb_id,
                                        int               N_in_word,
                                        bool              puncture,
                                        cg::thread_block& block);

// The default version of store to handle major lifting sizes such as 384, 320, 256 and etc.
// remainder means Z % 32.
template <>
__device__ void store_from_smem_to_gmem<uint32_t, 0>(const int         Z,
                                                     const int         num_tdbv,
                                                     LDPC_output_t     output,
                                                     uint32_t*         sbuf,
                                                     int               cb_id,
                                                     int               N_in_word,
                                                     bool              puncture,
                                                     cg::thread_block& block)
{
    const int pun_base = puncture ? num_tdbv * 2 : 0;
    for(int i = threadIdx.x; i < N_in_word; i += blockDim.x)
    {
        uint32_t bits      = sbuf[i + pun_base];
        output({i, cb_id}) = bits;
    }
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, 4>(const int         Z,
                                                     const int         num_tdbv,
                                                     LDPC_output_t     output,
                                                     uint32_t*         sbuf,
                                                     int               cb_id,
                                                     int               N_in_word,
                                                     bool              puncture,
                                                     cg::thread_block& block)
{
    const int num_grouped_words = 8 * Z / 32;
    const int num_padded_words  = num_grouped_words + 7;
    const int pun_base          = puncture ? num_tdbv * 2 : 0;
    for(int gi = threadIdx.x; gi < N_in_word; gi += blockDim.x)
    {
        const int local_gi = gi % num_grouped_words;
        const int local_si = local_gi + ((local_gi >= 2) ? (local_gi - 2) : 0);
        const int si       = (gi / num_grouped_words) * num_padded_words + local_si;
        uint32_t  bits     = 0;

        if(local_gi == 0)
        {
            bits = sbuf[si + pun_base];
        }
        else if(local_gi == 1)
        {
            bits = bits_gather2(sbuf, 4, si + pun_base, 0);
        }
        else if(local_gi < 8)
        {
            int n_lo_bits = (local_gi - 1) * 4;
            int lo_shift  = 32 - n_lo_bits;
            bits          = bits_gather3(sbuf, n_lo_bits, 4, si + pun_base, lo_shift, gi == N_in_word - 1);
        }
        else
        {
            bits = bits_gather2(sbuf, 28, si + pun_base, 4);
        }

        output({gi, cb_id}) = bits;
    }
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, 8>(const int         Z,
                                                     const int         num_tdbv,
                                                     LDPC_output_t     output,
                                                     uint32_t*         sbuf,
                                                     int               cb_id,
                                                     int               N_in_word,
                                                     bool              puncture,
                                                     cg::thread_block& block)
{
    const int div_point         = Z / 32;
    const int num_grouped_words = Z / 8;
    const int num_padded_words  = num_grouped_words + 3;
    const int pun_base          = puncture ? num_tdbv * 2 : 0;
    for(int gi = threadIdx.x; gi < N_in_word; gi += blockDim.x)
    {
        const int local_gi = gi % num_grouped_words;
        const int local_si = local_gi + (local_gi > div_point * 2) * (local_gi / div_point - 1 - (local_gi % div_point == 0));
        const int si       = (gi / num_grouped_words) * num_padded_words + local_si;
        uint32_t  bits     = 0;
        if(local_gi < div_point)
        {
            bits = sbuf[si + pun_base];
        }
        else if(local_gi < div_point * 2)
        {
            bits = bits_gather2(sbuf, 8, si + pun_base, (local_gi == div_point) ? 0 : 24);
        }
        else if(local_gi == div_point * 2)
        {
            bits = bits_gather3(sbuf, 8, 8, si + pun_base, 24);
        }
        else if(local_gi < div_point * 3)
        {
            bits = bits_gather2(sbuf, 16, si + pun_base, 16);
        }
        else if(local_gi == div_point * 3)
        {
            bits = bits_gather3(sbuf, 16, 8, si + pun_base, 16);
        }
        else
        {
            bits = bits_gather2(sbuf, 24, si + pun_base, 8);
        }
        output({gi, cb_id}) = bits;
    }
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, 12>(const int         Z,
                                                      const int         num_tdbv,
                                                      LDPC_output_t     output,
                                                      uint32_t*         sbuf,
                                                      int               cb_id,
                                                      int               N_in_word,
                                                      bool              puncture,
                                                      cg::thread_block& block)
{
    // only for Z = 44
    const int num_grouped_words = 8 * Z / 32;
    const int num_padded_words  = num_grouped_words + 5;

    const int pun_base = puncture ? num_tdbv * 2 : 0;

    for(int gi = threadIdx.x; gi < N_in_word; gi += blockDim.x)
    {
        const int local_gi = gi % num_grouped_words;
        const int si_base  = (gi / num_grouped_words) * num_padded_words;

        uint32_t bits = 0;

        if(local_gi == 0)
        {
            const int si = si_base;
            bits         = sbuf[si + pun_base];
        }
        else if(local_gi == 1 || local_gi == 4 || local_gi == 8)
        {
            const int ii           = (local_gi == 1) ? 0 : ((local_gi == 4) ? 1 : 2);
            const int local_si[3]  = {1, 5, 11};
            const int si           = si_base + local_si[ii];
            const int n_lo_bits[4] = {12, 4, 8};
            const int lo_shift[4]  = {0, 8, 4};

            bits = bits_gather2(sbuf,
                                n_lo_bits[ii],
                                si + pun_base,
                                lo_shift[ii]);
        }
        else if(local_gi == 2 || local_gi == 5 || local_gi == 6 || local_gi == 9)
        {
            const int ii           = (local_gi == 2) ? 0 : ((local_gi == 5) ? 1 : ((local_gi == 6) ? 2 : 3));
            const int local_si[4]  = {2, 6, 8, 12};
            const int si           = si_base + local_si[ii];
            const int n_lo_bits[4] = {12, 4, 16, 8};
            bits                   = bits_gather3(sbuf,
                                n_lo_bits[ii],
                                12,
                                si + pun_base,
                                32 - n_lo_bits[ii],
                                gi == N_in_word - 1);
        }
        else
        {
            const int ii           = (local_gi == 3) ? 0 : ((local_gi == 7) ? 1 : 2);
            const int local_si[3]  = {4, 10, 14};
            const int si           = si_base + local_si[ii];
            const int n_lo_bits[4] = {24, 28, 20};

            bits = bits_gather2(sbuf,
                                n_lo_bits[ii],
                                si + pun_base,
                                32 - n_lo_bits[ii]);
        }
        output({gi, cb_id}) = bits;
    }
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, 16>(const int         Z,
                                                      const int         num_tdbv,
                                                      LDPC_output_t     output,
                                                      uint32_t*         sbuf,
                                                      int               cb_id,
                                                      int               N_in_word,
                                                      bool              puncture,
                                                      cg::thread_block& block)
{
    const int div_point         = Z / 32;
    const int num_grouped_words = Z / 16; // 2 * Z / 32;
    const int num_padded_words  = num_grouped_words + 1;
    const int pun_base          = puncture ? num_tdbv * 2 : 0;
    for(int gi = threadIdx.x; gi < N_in_word; gi += blockDim.x)
    {
        const int local_si = gi % num_grouped_words;
        const int si       = (gi / num_grouped_words) * num_padded_words + local_si;
        uint32_t  bits     = 0;
        if(local_si < div_point)
        {
            bits = sbuf[si + pun_base];
        }
        else
        {
            bits = bits_gather2(sbuf, 16, si + pun_base, (local_si == div_point) ? 0 : 16);
        }
        output({gi, cb_id}) = bits;
    }
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, 20>(const int         Z,
                                                      const int         num_tdbv,
                                                      LDPC_output_t     output,
                                                      uint32_t*         sbuf,
                                                      int               cb_id,
                                                      int               N_in_word,
                                                      bool              puncture,
                                                      cg::thread_block& block)
{
    // only for Z = 52
    const int num_grouped_words = 8 * Z / 32;
    const int num_padded_words  = num_grouped_words + 3;

    const int pun_base = puncture ? num_tdbv * 2 : 0;

    for(int gi = threadIdx.x; gi < N_in_word; gi += blockDim.x)
    {
        const int local_gi = gi % num_grouped_words;
        const int si_base  = (gi / num_grouped_words) * num_padded_words;

        uint32_t bits = 0;

        if(local_gi == 0)
        {
            const int si = si_base;
            bits         = sbuf[si + pun_base];
        }
        else if(local_gi == 1 || local_gi == 3 || local_gi == 6 || local_gi == 8 || local_gi == 11)
        {
            const int ii           = (local_gi == 1) ? 0 : ((local_gi == 3) ? 1 : ((local_gi == 6) ? 2 : (local_gi == 8) ? 3 : 4));
            const int local_si[5]  = {1, 3, 7, 9, 13};
            const int si           = si_base + local_si[ii];
            const int n_lo_bits[5] = {20, 8, 16, 4, 12};
            const int lo_shift[5]  = {0, 12, 4, 16, 8};

            bits = bits_gather2(sbuf,
                                n_lo_bits[ii],
                                si + pun_base,
                                lo_shift[ii],
                                gi == N_in_word - 1);
        }
        else if(local_gi == 2 || local_gi == 5 || local_gi == 7 || local_gi == 10 || local_gi == 12)
        {
            const int ii           = (local_gi == 2) ? 0 : ((local_gi == 5) ? 1 : ((local_gi == 7) ? 2 : (local_gi == 10) ? 3 : 4));
            const int local_si[5]  = {2, 6, 8, 12, 14};
            const int si           = si_base + local_si[ii];
            const int n_lo_bits[5] = {20, 28, 16, 24, 12};

            bits = bits_gather2(sbuf,
                                n_lo_bits[ii],
                                si + pun_base,
                                32 - n_lo_bits[ii]);
        }
        else
        {
            const int ii           = (local_gi == 4) ? 0 : 1;
            const int local_si[2]  = {4, 10};
            const int si           = si_base + local_si[ii];
            const int n_lo_bits[2] = {8, 4};

            bits = bits_gather3(sbuf,
                                n_lo_bits[ii],
                                20,
                                si + pun_base,
                                32 - n_lo_bits[ii]);
        }
        output({gi, cb_id}) = bits;
    }
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, 24>(const int         Z,
                                                      const int         num_tdbv,
                                                      LDPC_output_t     output,
                                                      uint32_t*         sbuf,
                                                      int               cb_id,
                                                      int               N_in_word,
                                                      bool              puncture,
                                                      cg::thread_block& block)
{
    const int div_point1 = Z / 32;
    const int div_point2 = div_point1 + div_point1 + 1;
    const int div_point3 = div_point2 + div_point1 + 1;

    const int num_grouped_words = Z / 8;
    const int num_padded_words  = num_grouped_words + 1;

    const int pun_base = puncture ? num_tdbv * 2 : 0;

    for(int gi = threadIdx.x; gi < N_in_word; gi += blockDim.x)
    {
        const int local_si = gi % num_grouped_words;
        const int si       = (gi / num_grouped_words) * num_padded_words + local_si;

        uint32_t bits = 0;
        if(local_si < div_point1)
        {
            bits = sbuf[si + pun_base];
        }
        else if(local_si < div_point2)
        {
            bits = bits_gather2(sbuf, 24, si + pun_base, (local_si == div_point1) ? 0 : 8);
        }
        else if(local_si < div_point3)
        {
            bits = bits_gather2(sbuf, 16, si + pun_base, (local_si == div_point2) ? 8 : 16);
        }
        else
        {
            bits = bits_gather2(sbuf, 8, si + pun_base, (local_si == div_point3) ? 16 : 24);
        }
        output({gi, cb_id}) = bits;
    }
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, 28>(const int         Z,
                                                      const int         num_tdbv,
                                                      LDPC_output_t     output,
                                                      uint32_t*         sbuf,
                                                      int               cb_id,
                                                      int               N_in_word,
                                                      bool              puncture,
                                                      cg::thread_block& block)
{
    const int num_grouped_words = 15;
    const int num_padded_words  = 16;
    const int pun_base          = puncture ? num_tdbv * 2 : 0;
    for(int gi = threadIdx.x; gi < N_in_word; gi += blockDim.x)
    {
        const int local_si  = gi % num_grouped_words;
        const int si        = (gi / num_grouped_words) * num_padded_words + local_si;
        uint32_t  bits      = 0;
        int       n_lo_bits = 28 - 4 * ((local_si - 1) / 2);
        int       lo_shift  = 4 * ((local_si) / 2);

        if(local_si == 0)
        {
            bits = sbuf[si + pun_base];
        }
        else
        {
            bits = bits_gather2(sbuf, n_lo_bits, si + pun_base, lo_shift);
        }
        output({gi, cb_id}) = bits;
    }
}
template <>
__device__ void store_from_smem_to_gmem<uint32_t, -30>(const int         Z,
                                                       const int         num_tdbv,
                                                       LDPC_output_t     output,
                                                       uint32_t*         sbuf,
                                                       int               cb_id,
                                                       int               N_in_word,
                                                       bool              puncture,
                                                       cg::thread_block& block)
{
    const int num_grouped_words = 15;
    const int num_padded_words  = 16;

    const int pun_base = puncture ? num_tdbv * 2 : 0;

    for(int gi = threadIdx.x; gi < N_in_word; gi += blockDim.x)
    {
        const int local_si = gi % num_grouped_words;
        const int si       = (gi / num_grouped_words) * num_padded_words + local_si;

        int n_lo_bits = Z - local_si * 2;

        uint32_t bits = 0;
        bits          = bits_gather2(sbuf,
                            n_lo_bits,
                            si + pun_base,
                            Z - n_lo_bits,
                            gi == N_in_word - 1);

        output({gi, cb_id}) = bits;
    }
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, -28>(const int         Z,
                                                       const int         num_tdbv,
                                                       LDPC_output_t     output,
                                                       uint32_t*         sbuf,
                                                       int               cb_id,
                                                       int               N_in_word,
                                                       bool              puncture,
                                                       cg::thread_block& block)
{
    const int num_grouped_words = 7;
    const int num_padded_words  = 8;

    const int pun_base = puncture ? num_tdbv * 2 : 0;

    for(int gi = threadIdx.x; gi < N_in_word; gi += blockDim.x)
    {
        const int local_si = gi % num_grouped_words;
        const int si       = (gi / num_grouped_words) * num_padded_words + local_si;

        int n_lo_bits = Z - local_si * 4;

        uint32_t bits = 0;
        bits          = bits_gather2(sbuf,
                            n_lo_bits,
                            si + pun_base,
                            Z - n_lo_bits,
                            gi == N_in_word - 1);

        output({gi, cb_id}) = bits;
    }
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, -26>(const int         Z,
                                                       const int         num_tdbv,
                                                       LDPC_output_t     output,
                                                       uint32_t*         sbuf,
                                                       int               cb_id,
                                                       int               N_in_word,
                                                       bool              puncture,
                                                       cg::thread_block& block)
{
    const int num_grouped_words = 13;
    const int num_padded_words  = 16;

    const int pun_base = puncture ? num_tdbv * 2 : 0;

    for(int gi = threadIdx.x; gi < N_in_word; gi += blockDim.x)
    {
        uint32_t  bits     = 0;
        const int local_gi = gi % num_grouped_words;
        const int local_si = local_gi +
                             ((local_gi <= 4) ? 0 : ((local_gi <= 8) ? 1 : 2));
        const int si = (gi / num_grouped_words) * num_padded_words + local_si;

        if(local_gi < 4)
        {
            int n_lo_bits = Z - local_gi * 6;
            bits          = bits_gather2(sbuf,
                                n_lo_bits,
                                si + pun_base,
                                Z - n_lo_bits);
        }
        else if(local_gi == 4 || local_gi == 8)
        {
            int n_lo_bits = (local_gi == 4) ? 2 : 4;
            int n_mi_bits = 26;
            int lo_shift  = (local_gi == 4) ? 24 : 22;
            bits          = bits_gather3(sbuf,
                                n_lo_bits,
                                n_mi_bits,
                                si + pun_base,
                                lo_shift,
                                gi == N_in_word - 1);
        }
        else if(local_gi < 8)
        {
            int n_lo_bits = Z - 4 - (local_gi - 5) * 6;
            bits          = bits_gather2(sbuf,
                                n_lo_bits,
                                si + pun_base,
                                Z - n_lo_bits);
        }
        else
        {
            int n_lo_bits = Z - 2 - (local_gi - 9) * 6;
            bits          = bits_gather2(sbuf,
                                n_lo_bits,
                                si + pun_base,
                                Z - n_lo_bits);
        }
        output({gi, cb_id}) = bits;
    }
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, -24>(const int         Z,
                                                       const int         num_tdbv,
                                                       LDPC_output_t     output,
                                                       uint32_t*         sbuf,
                                                       int               cb_id,
                                                       int               N_in_word,
                                                       bool              puncture,
                                                       cg::thread_block& block)
{
    const int num_grouped_words = 3;
    const int num_padded_words  = 4;

    const int pun_base = puncture ? num_tdbv * 2 : 0;

    for(int gi = threadIdx.x; gi < N_in_word; gi += blockDim.x)
    {
        const int local_si = gi % num_grouped_words;
        const int si       = (gi / num_grouped_words) * num_padded_words + local_si;

        int n_lo_bits = Z - local_si * 8;

        uint32_t bits = 0;
        bits          = bits_gather2(sbuf,
                            n_lo_bits,
                            si + pun_base,
                            Z - n_lo_bits);

        output({gi, cb_id}) = bits;
    }
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, -22>(const int         Z,
                                                       const int         num_tdbv,
                                                       LDPC_output_t     output,
                                                       uint32_t*         sbuf,
                                                       int               cb_id,
                                                       int               N_in_word,
                                                       bool              puncture,
                                                       cg::thread_block& block)
{
    const int num_grouped_words = 11;
    const int num_padded_words  = 16;

    const int pun_base = puncture ? num_tdbv * 2 : 0;

    for(int gi = threadIdx.x; gi < N_in_word; gi += blockDim.x)
    {
        uint32_t  bits     = 0;
        const int local_gi = gi % num_grouped_words;
        const int local_si = local_gi +
                             ((local_gi <= 2) ? 0 : ((local_gi <= 4) ? 1 : ((local_gi <= 6) ? 2 : ((local_gi <= 8) ? 3 : 4))));
        const int si = (gi / num_grouped_words) * num_padded_words + local_si;

        if(local_gi == 0)
        {
            int n_lo_bits = Z;

            bits = bits_gather2(sbuf,
                                n_lo_bits,
                                si + pun_base,
                                0);
        }
        else if(local_gi % 2 == 1 || local_gi == 10)
        {
            int n_lo_bits = (local_gi != 10) ? local_gi + 11 : 10;

            bits = bits_gather2(sbuf,
                                n_lo_bits,
                                si + pun_base,
                                Z - n_lo_bits,
                                gi == N_in_word - 1);
        }
        else
        {
            int n_lo_bits = local_gi;
            int n_mi_bits = 22;
            int lo_shift  = Z - n_lo_bits;

            bits = bits_gather3(sbuf,
                                n_lo_bits,
                                n_mi_bits,
                                si + pun_base,
                                lo_shift);
        }
        output({gi, cb_id}) = bits;
    }
    block.sync();
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, -20>(const int         Z,
                                                       const int         num_tdbv,
                                                       LDPC_output_t     output,
                                                       uint32_t*         sbuf,
                                                       int               cb_id,
                                                       int               N_in_word,
                                                       bool              puncture,
                                                       cg::thread_block& block)
{
    const int num_grouped_words = 5;
    const int num_padded_words  = 8;

    const int pun_base = puncture ? num_tdbv * 2 : 0;

    for(int gi = threadIdx.x; gi < N_in_word; gi += blockDim.x)
    {
        uint32_t  bits     = 0;
        const int local_gi = gi % num_grouped_words;
        const int local_si = local_gi +
                             ((local_gi <= 1) ? 0 : ((local_gi <= 3) ? 1 : 2));
        const int si = (gi / num_grouped_words) * num_padded_words + local_si;

        if(local_gi % 2 == 0)
        {
            int n_lo_bits = Z - 2 * local_gi;

            bits = bits_gather2(sbuf,
                                n_lo_bits,
                                si + pun_base,
                                Z - n_lo_bits,
                                gi == N_in_word - 1);
        }
        else
        {
            int n_lo_bits = (local_gi == 1) ? 8 : 4;
            int n_mi_bits = 20;
            int lo_shift  = Z - n_lo_bits;

            bits = bits_gather3(sbuf,
                                n_lo_bits,
                                n_mi_bits,
                                si + pun_base,
                                lo_shift);
        }
        output({gi, cb_id}) = bits;
    }
    block.sync();
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, -18>(const int         Z,
                                                       const int         num_tdbv,
                                                       LDPC_output_t     output,
                                                       uint32_t*         sbuf,
                                                       int               cb_id,
                                                       int               N_in_word,
                                                       bool              puncture,
                                                       cg::thread_block& block)
{
    const int num_grouped_words = 9;
    const int num_padded_words  = 16;

    const int pun_base = puncture ? num_tdbv * 2 : 0;

    for(int gi = threadIdx.x; gi < N_in_word; gi += blockDim.x)
    {
        uint32_t  bits     = 0;
        const int local_gi = gi % num_grouped_words;
        const int local_si = local_gi +
                             ((local_gi <= 1) ? 0 : ((local_gi <= 4) ? local_gi - 1 : local_gi - 2));
        const int si = (gi / num_grouped_words) * num_padded_words + local_si;

        if(local_gi % 4 == 0)
        {
            int n_lo_bits = Z - local_gi / 2;

            bits = bits_gather2(sbuf,
                                n_lo_bits,
                                si + pun_base,
                                Z - n_lo_bits,
                                gi == N_in_word - 1);
        }
        else
        {
            int n_lo_bits = (local_gi < 4) ? local_gi * 4 : 2 + (local_gi - 5) * 4;
            int n_mi_bits = 18;
            int lo_shift  = Z - n_lo_bits;

            bits = bits_gather3(sbuf,
                                n_lo_bits,
                                n_mi_bits,
                                si + pun_base,
                                lo_shift);
        }
        output({gi, cb_id}) = bits;
    }
    block.sync();
}

template <int Z>
__device__ void small_store_from_smem_to_gmem(const int         num_tdbv,
                                              LDPC_output_t     output,
                                              uint32_t*         sbuf,
                                              int               cb_id,
                                              int               N_in_word,
                                              bool              puncture,
                                              cg::thread_block& block)
{
    if(threadIdx.x == 0)
    {
        const int pun_base  = puncture ? num_tdbv * 2 : 0;
        int       si        = 0;
        int       start_pos = 0;
        for(int gi = 0; gi < N_in_word; gi++)
        {
            uint32_t bits    = 0;
            int      bit_pos = 0;
            while(bit_pos < 32)
            {
                int cur_n_bits = (bit_pos + Z > 32) ? (32 - bit_pos) : (Z - start_pos);
                int mask       = (1 << cur_n_bits) - 1;
                bits += ((sbuf[pun_base + si] >> start_pos) & mask) << bit_pos;
                bit_pos += cur_n_bits;

                if(bit_pos >= 32 && cur_n_bits < Z)
                {
                    start_pos = cur_n_bits;
                }
                else
                {
                    start_pos = 0;
                    si++;
                }
            }
            output({gi, cb_id}) = bits;
        }
    }
    block.sync();
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, -16>(const int         Z,
                                                       const int         num_tdbv,
                                                       LDPC_output_t     output,
                                                       uint32_t*         sbuf,
                                                       int               cb_id,
                                                       int               N_in_word,
                                                       bool              puncture,
                                                       cg::thread_block& block)
{
    small_store_from_smem_to_gmem<16>(num_tdbv,
                                      output,
                                      sbuf,
                                      cb_id,
                                      N_in_word,
                                      puncture,
                                      block);
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, -15>(const int         Z,
                                                       const int         num_tdbv,
                                                       LDPC_output_t     output,
                                                       uint32_t*         sbuf,
                                                       int               cb_id,
                                                       int               N_in_word,
                                                       bool              puncture,
                                                       cg::thread_block& block)
{
    small_store_from_smem_to_gmem<15>(num_tdbv,
                                      output,
                                      sbuf,
                                      cb_id,
                                      N_in_word,
                                      puncture,
                                      block);
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, -14>(const int         Z,
                                                       const int         num_tdbv,
                                                       LDPC_output_t     output,
                                                       uint32_t*         sbuf,
                                                       int               cb_id,
                                                       int               N_in_word,
                                                       bool              puncture,
                                                       cg::thread_block& block)
{
    small_store_from_smem_to_gmem<14>(num_tdbv,
                                      output,
                                      sbuf,
                                      cb_id,
                                      N_in_word,
                                      puncture,
                                      block);
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, -13>(const int         Z,
                                                       const int         num_tdbv,
                                                       LDPC_output_t     output,
                                                       uint32_t*         sbuf,
                                                       int               cb_id,
                                                       int               N_in_word,
                                                       bool              puncture,
                                                       cg::thread_block& block)
{
    small_store_from_smem_to_gmem<13>(num_tdbv,
                                      output,
                                      sbuf,
                                      cb_id,
                                      N_in_word,
                                      puncture,
                                      block);
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, -12>(const int         Z,
                                                       const int         num_tdbv,
                                                       LDPC_output_t     output,
                                                       uint32_t*         sbuf,
                                                       int               cb_id,
                                                       int               N_in_word,
                                                       bool              puncture,
                                                       cg::thread_block& block)
{
    small_store_from_smem_to_gmem<12>(num_tdbv,
                                      output,
                                      sbuf,
                                      cb_id,
                                      N_in_word,
                                      puncture,
                                      block);
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, -11>(const int         Z,
                                                       const int         num_tdbv,
                                                       LDPC_output_t     output,
                                                       uint32_t*         sbuf,
                                                       int               cb_id,
                                                       int               N_in_word,
                                                       bool              puncture,
                                                       cg::thread_block& block)
{
    small_store_from_smem_to_gmem<11>(num_tdbv,
                                      output,
                                      sbuf,
                                      cb_id,
                                      N_in_word,
                                      puncture,
                                      block);
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, -10>(const int         Z,
                                                       const int         num_tdbv,
                                                       LDPC_output_t     output,
                                                       uint32_t*         sbuf,
                                                       int               cb_id,
                                                       int               N_in_word,
                                                       bool              puncture,
                                                       cg::thread_block& block)
{
    small_store_from_smem_to_gmem<10>(num_tdbv,
                                      output,
                                      sbuf,
                                      cb_id,
                                      N_in_word,
                                      puncture,
                                      block);
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, -9>(const int         Z,
                                                      const int         num_tdbv,
                                                      LDPC_output_t     output,
                                                      uint32_t*         sbuf,
                                                      int               cb_id,
                                                      int               N_in_word,
                                                      bool              puncture,
                                                      cg::thread_block& block)
{
    small_store_from_smem_to_gmem<9>(num_tdbv,
                                     output,
                                     sbuf,
                                     cb_id,
                                     N_in_word,
                                     puncture,
                                     block);
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, -8>(const int         Z,
                                                      const int         num_tdbv,
                                                      LDPC_output_t     output,
                                                      uint32_t*         sbuf,
                                                      int               cb_id,
                                                      int               N_in_word,
                                                      bool              puncture,
                                                      cg::thread_block& block)
{
    small_store_from_smem_to_gmem<8>(num_tdbv,
                                     output,
                                     sbuf,
                                     cb_id,
                                     N_in_word,
                                     puncture,
                                     block);
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, -7>(const int         Z,
                                                      const int         num_tdbv,
                                                      LDPC_output_t     output,
                                                      uint32_t*         sbuf,
                                                      int               cb_id,
                                                      int               N_in_word,
                                                      bool              puncture,
                                                      cg::thread_block& block)
{
    small_store_from_smem_to_gmem<7>(num_tdbv,
                                     output,
                                     sbuf,
                                     cb_id,
                                     N_in_word,
                                     puncture,
                                     block);
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, -6>(const int         Z,
                                                      const int         num_tdbv,
                                                      LDPC_output_t     output,
                                                      uint32_t*         sbuf,
                                                      int               cb_id,
                                                      int               N_in_word,
                                                      bool              puncture,
                                                      cg::thread_block& block)
{
    small_store_from_smem_to_gmem<6>(num_tdbv,
                                     output,
                                     sbuf,
                                     cb_id,
                                     N_in_word,
                                     puncture,
                                     block);
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, -5>(const int         Z,
                                                      const int         num_tdbv,
                                                      LDPC_output_t     output,
                                                      uint32_t*         sbuf,
                                                      int               cb_id,
                                                      int               N_in_word,
                                                      bool              puncture,
                                                      cg::thread_block& block)
{
    small_store_from_smem_to_gmem<5>(num_tdbv,
                                     output,
                                     sbuf,
                                     cb_id,
                                     N_in_word,
                                     puncture,
                                     block);
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, -4>(const int         Z,
                                                      const int         num_tdbv,
                                                      LDPC_output_t     output,
                                                      uint32_t*         sbuf,
                                                      int               cb_id,
                                                      int               N_in_word,
                                                      bool              puncture,
                                                      cg::thread_block& block)
{
    small_store_from_smem_to_gmem<4>(num_tdbv,
                                     output,
                                     sbuf,
                                     cb_id,
                                     N_in_word,
                                     puncture,
                                     block);
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, -3>(const int         Z,
                                                      const int         num_tdbv,
                                                      LDPC_output_t     output,
                                                      uint32_t*         sbuf,
                                                      int               cb_id,
                                                      int               N_in_word,
                                                      bool              puncture,
                                                      cg::thread_block& block)
{
    small_store_from_smem_to_gmem<3>(num_tdbv,
                                     output,
                                     sbuf,
                                     cb_id,
                                     N_in_word,
                                     puncture,
                                     block);
}

template <>
__device__ void store_from_smem_to_gmem<uint32_t, -2>(const int         Z,
                                                      const int         num_tdbv,
                                                      LDPC_output_t     output,
                                                      uint32_t*         sbuf,
                                                      int               cb_id,
                                                      int               N_in_word,
                                                      bool              puncture,
                                                      cg::thread_block& block)
{
    small_store_from_smem_to_gmem<2>(num_tdbv,
                                     output,
                                     sbuf,
                                     cb_id,
                                     N_in_word,
                                     puncture,
                                     block);
}

} // namespace ldpc

#endif // !defined(LDPC_LOAD_STORE_CUH_INCLUDED_)
