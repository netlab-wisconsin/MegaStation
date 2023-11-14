/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_MIN_SUM_UPDATE_HALF_0_CUH_INCLUDED_)
#define LDPC2_MIN_SUM_UPDATE_HALF_0_CUH_INCLUDED_

#include "ldpc2.cuh"


namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// half2_min_sum_update_low()
// Update the min-sum representation using the low fp16 value in an
// input pair.
__device__ __inline__
void half2_min_sum_update_low(word_t& min1_min0, int& min0_index, word_t new_value, int idx)
{
    // Duplicate the low word (expecting instruction swizzle to avoid an explicit instruction)
    word_t v = h0_h0(new_value);
    
#if ((__CUDACC_VER_MAJOR__ == 10) && (__CUDACC_VER_MINOR__ >= 2)) || (__CUDACC_VER_MAJOR__ >= 11)
    LDPC2_ASM("\t"
              "{\n\t\t"
              ".reg .pred ltmin0, ltmin1;\n\t\t"
              ".reg .f16x2 abs_min1_min0, abs_v;\n\t\t"
              "abs.f16x2 abs_min1_min0, %0;\n\t\t"                        // abs_min1_min0 = [abs(min1)  abs(min0) ] (expect |value| modifier in setp)
              "abs.f16x2 abs_v, %2;\n\t\t"                                // abs_v         = [abs(v)     abs(v)    ] (expect |value| modifier in setp)
              "setp.ltu.f16x2 ltmin0|ltmin1, abs_v, abs_min1_min0;\n\t\t" // ltmin0 = abs(v) < abs(min0)
                                                                          // ltmin1 = abs(v) < abs(min1)
              "selp.s32 %1, %3, %1, ltmin0;\n\t\t"                        // if abs(v) < min0, replace min0_idx
              "@ltmin1  prmt.b32 %0, %0, %4, 0x5410;\n\t"                 // if abs(v) < min1, select to have [ new_value  original_min0 ]
              "@ltmin0  prmt.b32 %0, %0, %0, 0x1032;\n\t"                 // if abs(v) < min0, select to have [ original_min0  new_value]
              "}\n"
              : "+r"(min1_min0.u32),
                "+r"(min0_index)
              : "r"(v.u32),
                "r"(idx),
                "r"(new_value.u32));
#else
    // abs.f16x2 not in PTX until v6.5 (CUDA 10.2)
    word_t abs_min1_min0, abs_v;
    abs_min1_min0 = word_pair_abs(min1_min0);
    abs_v         = word_pair_abs(v);
    LDPC2_ASM("\t"
              "{\n\t\t"
              ".reg .pred ltmin0, ltmin1;\n\t\t"
              "setp.ltu.f16x2 ltmin0|ltmin1, %2, %3;\n\t\t" // ltmin0 = abs(v) < abs(min0)
                                                            // ltmin1 = abs(v) < abs(min1)
              "selp.s32 %1, %5, %1, ltmin0;\n\t\t"              // if abs(v) < min0, replace min0_idx
              "@ltmin1  prmt.b32 %0, %0, %4, 0x5410;\n\t"   // if abs(v) < min1, permute to get [ new_value  original_min0 ]
              "@ltmin0  prmt.b32 %0, %0, %0, 0x1032;\n\t"   // if abs(v) < min0, permute to get [ original_min0  new_value]
              "}\n"
              : "+r"(min1_min0.u32),
                "+r"(min0_index)
              : "r"(abs_v.u32),
                "r"(abs_min1_min0.u32),
                "r"(new_value.u32),
                "r"(idx));
#endif
}

////////////////////////////////////////////////////////////////////////
// half2_min_sum_update_high()
// Update the min-sum representation using the high fp16 value in an
// input pair.
__device__ __inline__
void half2_min_sum_update_high(word_t& min1_min0, int& min0_index, word_t new_value, int idx)
{
    // Duplicate the high word (expecting instruction swizzle to avoid an explicit instruction)
    word_t v = h1_h1(new_value);
    
#if ((__CUDACC_VER_MAJOR__ == 10) && (__CUDACC_VER_MINOR__ >= 2)) || (__CUDACC_VER_MAJOR__ >= 11)
    LDPC2_ASM("\t"
              "{\n\t\t"
              ".reg .pred ltmin0, ltmin1;\n\t\t"
              ".reg .f16x2 abs_min1_min0, abs_v;\n\t\t"
              "abs.f16x2 abs_min1_min0, %0;\n\t\t"                        // abs_min1_min0 = [abs(min1)  abs(min0) ] (expect |value| modifier in setp)
              "abs.f16x2 abs_v, %2;\n\t\t"                                // abs_v         = [abs(v)     abs(v)    ] (expect |value| modifier in setp)
              "setp.ltu.f16x2 ltmin0|ltmin1, abs_v, abs_min1_min0;\n\t\t" // ltmin0 = abs(v) < abs(min0)
                                                                          // ltmin1 = abs(v) < abs(min1)
              "selp.s32 %1, %3, %1, ltmin0;\n\t\t"                        // if abs(v) < min0, replace min0_idx
              "@ltmin1  prmt.b32 %0, %0, %4, 0x7610;\n\t"                 // if abs(v) < min1, select to have [ new_value  original_min0 ]
              "@ltmin0  prmt.b32 %0, %0, %0, 0x1032;\n\t"                 // if abs(v) < min0, select to have [ original_min0  new_value]
              "}\n"
              : "+r"(min1_min0.u32),
                "+r"(min0_index)
              : "r"(v.u32),
                "r"(idx),
                "r"(new_value.u32));
#else
    // abs.f16x2 not in PTX until v6.5 (CUDA 10.2)
    word_t abs_min1_min0, abs_v;
    abs_min1_min0 = word_pair_abs(min1_min0);
    abs_v         = word_pair_abs(v);
    LDPC2_ASM("\t"
              "{\n\t\t"
              ".reg .pred ltmin0, ltmin1;\n\t\t"
              "setp.ltu.f16x2 ltmin0|ltmin1, %2, %3;\n\t\t" // ltmin0 = abs(v) < abs(min0)
                                                            // ltmin1 = abs(v) < abs(min1)
              "selp.s32 %1, %5, %1, ltmin0;\n\t\t"              // if abs(v) < min0, replace min0_idx
              "@ltmin1  prmt.b32 %0, %0, %4, 0x7610;\n\t"   // if abs(v) < min1, permute to get [ new_value  original_min0 ]
              "@ltmin0  prmt.b32 %0, %0, %0, 0x1032;\n\t"   // if abs(v) < min0, permute to get [ original_min0  new_value]
              "}\n"
              : "+r"(min1_min0.u32),
                "+r"(min0_index)
              : "r"(abs_v.u32),
                "r"(abs_min1_min0.u32),
                "r"(new_value.u32),
                "r"(idx));
#endif
}

////////////////////////////////////////////////////////////////////////
// min_sum_update_half_0
struct min_sum_update_half_0
{
    //------------------------------------------------------------------
    // update_pair()
    // Update the internal representation with a new pair of values
    static __device__ __inline__
    void update_pair(word_t&         min1_min0,
                     int&            min0_index,
                     word_t          v1_v0,
                     int             pair_idx)
    {
        half2_min_sum_update_low (min1_min0, min0_index, v1_v0, pair_idx * 2);
        half2_min_sum_update_high(min1_min0, min0_index, v1_v0, pair_idx * 2 + 1);
    }
    //------------------------------------------------------------------
    // update_low()
    // Update the internal representation with a single value
    static __device__ __inline__
    void update_low(word_t& min1_min0,
                    int&    min0_index,
                    word_t  vx_v0,
                    int     pair_idx)
    {
        half2_min_sum_update_low(min1_min0, min0_index, vx_v0, pair_idx * 2);
    }
};

} // namespace ldpc2

#endif // !defined(LDPC2_MIN_SUM_UPDATE_HALF_0_CUH_INCLUDED_)