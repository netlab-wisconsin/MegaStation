/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC_CUH_INCLUDED_)
#define LDPC_CUH_INCLUDED_

#include "cuphy_kernel_util.cuh"
#include "type_convert.hpp"
#include "cuphy.hpp"

namespace
{
// Z = a * 2^j
// Set 0: a = 2,  j = 0..7
// Set 1: a = 3,  j = 0..7
// Set 2: a = 5,  j = 0..6
// Set 3: a = 7,  j = 0..5
// Set 4: a = 9,  j = 0..5
// Set 5: a = 11, j = 0..5
// Set 6: a = 13, j = 0..4
// Set 7: a = 15, j = 0..4
//                         j
//       --------------------------------------
//  a  |   0    1    2    3    4    5    6    7
//  2  |   2    4    8   16   32   64  128  256
//  3  |   3    6   12   24   48   96  192  384
//  5  |   5   10   20   40   80  160  320   -
//  7  |   7   14   28   56  112  224   -    -
//  9  |   9   18   36   72  144  288   -    -
//  11 |  11   22   44   88  176  352   -    -
//  13 |  13   26   52  104  208   -    -    -
//  15 |  15   30   60  120  240   -    -    -
//
// TODO: Currently only works for valid Z values. Change
// return statement to check if a*2j == Z and return -1 if
// not equal. Also, add invalid Z values to test values to confirm.
int set_from_Z(int Z)
{
    if((Z < 2) || (Z > 384))
    {
        return -1;
    }
    //------------------------------------------------------------------
    // All powers of 2 are in set 0
    if(0 == (Z & (Z - 1)))
    {
        return 0;
    }
    //------------------------------------------------------------------
    while(Z)
    {
        if((Z < 16) && (1 == (Z % 2)))
            return (Z / 2);
        Z >>= 1;
    }
    return -1;
}

#if 0
void test_set_from_Z()
{
    struct Z_set
    {
        int Z;
        int iLS;
    };
    const Z_set test_values[] = { {  2, 0}, {  3, 1}, {  4, 0}, {  5, 2}, {  6, 1}, {  7, 3}, {  8, 0}, {  9, 4}, { 10, 2}, { 11, 5},
                                  { 12, 1}, { 13, 6}, { 14, 3}, { 15, 7}, { 16, 0}, { 18, 4}, { 20, 2}, { 22, 5}, { 24, 1}, { 26, 6},
                                  { 28, 3}, { 30, 7}, { 32, 0}, { 36, 4}, { 40, 2}, { 44, 5}, { 48, 1}, { 52, 6}, { 56, 3}, { 60, 7},
                                  { 64, 0}, { 72, 4}, { 80, 2}, { 88, 5}, { 96, 1}, {104, 6}, {112, 3}, {120, 7}, {128, 0}, {144, 4},
                                  {160, 2}, {176, 5}, {192, 1}, {208, 6}, {224, 3}, {240, 7}, {256, 0}, {288, 4}, {320, 2}, {352, 5},
                                  {384, 1}};
    int correct_count = 0;
    for(int i = 0; i <= sizeof(test_values) / sizeof(test_values[0]); ++i)
    {
        const Z_set& s = test_values[i];
        int iLS = set_from_Z(s.Z);
        if(iLS != s.iLS)
        {
            NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT, "Z to set mismatch: Z = {}, actual = {}, returned = {}", s.Z, s.iLS, iLS);
        }
        else
        {
            ++correct_count;
        }
    }
    printf("test_set_from_Z(): %i out of %lu correct\n",
           correct_count,
           sizeof(test_values) / sizeof(test_values[0]));
}
#endif
} // namespace

// clang-format off
template <typename TLLR> CUDA_INLINE bool is_neg        (const TLLR& llr)   { return (llr < static_cast<TLLR>(0)); }
template <>              CUDA_INLINE bool is_neg<__half>(const __half& llr) { return __hlt(llr, 0);                }

template <typename TLLR> CUDA_INLINE TLLR   negate        (const TLLR& llr)   { return (-llr); }
template <>              CUDA_INLINE __half negate<__half>(const __half& llr) { return __hneg(llr);                }

template <typename TLLR> CUDA_INLINE float to_float     (const TLLR&  llr);
template <>              CUDA_INLINE float to_float     (const float& llr)  { return llr; }
template <>              CUDA_INLINE float to_float     (const __half& llr) { return __half2float(llr); }

template <typename TLLR> CUDA_INLINE TLLR   llr_abs        (const TLLR& llr)   { return fabsf(llr);                       }
template <> CUDA_INLINE              __half llr_abs<__half>(const __half& llr) { return __hlt(llr, 0) ? __hneg(llr): llr; }
// clang-format on

namespace ldpc
{

CUDA_INLINE
int device_popc(unsigned int a) { return __popc(a); }

} // namespace ldpc

template <cuphyDataType_t TLLREnum>
__device__ void cta_write_hard_decision(LDPC_output_t                                    tOutput,
                                        int                                              codeWordIdx,
                                        int                                              K,
                                        const typename data_type_traits<TLLREnum>::type* srcAPP)
{
    typedef typename data_type_traits<TLLREnum>::type LLR_t;
    const int                                         WORDS_PER_CW      = (K + 31) / 32;
    const int                                         BIT_BLOCK_COUNT   = (K + 1023) / 1024;
    const int                                         THREAD_BLOCK_SIZE = blockDim.x * blockDim.y;
    const int                                         WARPS_PER_BLOCK   = THREAD_BLOCK_SIZE / 32;
    const int                                         THREAD_RANK       = (threadIdx.y * blockDim.x) + threadIdx.x;
    const int                                         WARP_IDX          = THREAD_RANK / 32;
    const int                                         LANE_IDX          = THREAD_RANK % 32;
    // Write output bits. Each warp of 32 threads will cooperate to
    // generate up to 32 output "words", and each of those "words" will
    // contain 32 decision bits. (Each warp will read 1024 LLR values,
    // and generate 1024 output bits in 32 uint32_t words.)
    // The maximum codeword size is 8448 bits, which corresponds to
    // 8448 / 32 = 264 32-bit words for output.
    for(int iOutBlock = WARP_IDX; iOutBlock < BIT_BLOCK_COUNT; iOutBlock += WARPS_PER_BLOCK)
    {
        uint32_t thread_output = 0;
        int      start_bit_idx = iOutBlock * 1024;
        for(int i = 0; i < 32; ++i)
        {
            int      idx           = start_bit_idx + (i * 32) + LANE_IDX;
            uint32_t hard_decision = ((idx < K) && is_neg(srcAPP[idx])) ? 1 : 0;
            uint32_t warp_bits     = __ballot_sync(0xFFFFFFFF, hard_decision);
            if(i == LANE_IDX)
            {
                thread_output = warp_bits;
            }
        }
        const int OUT_INDEX = (iOutBlock * 32) + LANE_IDX;
        //KERNEL_PRINT("THREAD_RANK = %i, iOutBlock = %i, OUT_INDEX = %i\n", THREAD_RANK, iOutBlock, OUT_INDEX);
        if(OUT_INDEX < WORDS_PER_CW)
        {
            //KERNEL_PRINT_IF(0 == OUT_INDEX, "output[0] = 0x%X\n", thread_output);
            tOutput({OUT_INDEX, codeWordIdx}) = thread_output;
        }
    }
}

#endif // !defined(LDPC_CUH_INCLUDED_)
