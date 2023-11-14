/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_SIGN_CUH_INCLUDED_)
#define LDPC2_SIGN_CUH_INCLUDED_

#include "ldpc2.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// sign_order_base
template <typename T> struct sign_order_base
{
    //------------------------------------------------------------------
    // init()
    __device__
    static void init(uint32_t& s)
    {
        s = 0;
    }
    //------------------------------------------------------------------
    // pop_count()
    // Returns the number of negative sign bits
    __device__
    static int pop_count(const uint32_t& s)
    {
        return __popc(s);
    }
    //------------------------------------------------------------------
    // sign_product()
    // Returns 0 or 1, based on whether the product of all signs is
    // negative(1) or positive (0).
    __device__
    static uint32_t sign_product(const uint32_t& s)
    {
        return (uint32_t)(pop_count(s) & 0x1);
    }
    //------------------------------------------------------------------
    // sign_product_mask()
    // Returns 0x80000000 or 0x00000000, based on whether the product
    // of all signs is negative(0x80000000) or positive (0x00000000).
    __device__
    static uint32_t sign_product_mask(const uint32_t& s)
    {
        return (((uint32_t)pop_count(s)) << 31);
    }
};

template <typename T> struct sign_order_le;

////////////////////////////////////////////////////////////////////////
// sign_order_le<float>
// Little endian sign manager: Bit 0 indicates the sign of index 0
template <>
struct sign_order_le<float> : public sign_order_base<float>
{
    //------------------------------------------------------------------
    // init_row()
    __device__
    static void init_row(uint32_t& s, word_t v0, word_t v1)
    {
        word_t smask0 = word_sign_mask(v0);
        word_t smask1 = word_sign_mask(v1);
        s             = (smask0.u32 >> 31) | (smask1.u32 >> 30);
    }
    //------------------------------------------------------------------
    // sign_mask()
    // Returns 0x80000000 or 0x00000000, based on the sign bit for the
    // given index.
    __device__
    static uint32_t sign_mask(const uint32_t& s, int index)
    {
        const uint32_t smask = s << (31 - index);
        return (smask & 0x80000000);
    }
    //------------------------------------------------------------------
    // update()
    __device__
    static void update(uint32_t& s, word_t v, int index)
    {
        word_t smask = word_sign_mask(v);
        s |= (smask.u32 >> (31 - index));
    }
    //------------------------------------------------------------------
    // signs_xor()
    // xor() sign bits in s with `signs_prod`, which should be either 0 or 1.
    // `count` indicates how many bits are valid in `s`.
    __device__
    static uint32_t signs_xor(uint32_t s_in, uint32_t signs_prod, int count)
    {
        uint32_t even_odd_mask = 0;
        if(signs_prod != 0)
        {
            even_odd_mask = (1 << count) - 1;
        }
        return (s_in ^ even_odd_mask);
    }
};

template <typename T> struct sign_order_be;

////////////////////////////////////////////////////////////////////////
// sign_order_be<float>
// Big endian sign manager: Bit 31 indicates the sign of index 0
template <>
struct sign_order_be<float> : public sign_order_base<float>
{
    //------------------------------------------------------------------
    // init_row()
    __device__
    static void init_row(uint32_t& s, word_t v0, word_t v1)
    {
        word_t smask0 = word_sign_mask(v0);
        word_t smask1 = word_sign_mask(v1);
        s             = (smask0.u32) | (smask1.u32 >> 1);
    }
    //------------------------------------------------------------------
    // sign_mask()
    // Returns 0x80000000 or 0x00000000, based on the sign bit for the
    // given index.
    __device__
    static uint32_t sign_mask(const uint32_t& s, int index)
    {
        const uint32_t smask = s << index;
        return (smask & 0x80000000);
    }
    //------------------------------------------------------------------
    // update()
    __device__
    static void update(uint32_t& s, word_t v, int index)
    {
        word_t smask = word_sign_mask(v);
        s |= (smask.u32 >> index);
    }
    //------------------------------------------------------------------
    // signs_xor()
    // xor() sign bits in s with `signs_prod`, which should be either 0 or 1.
    // `count` indicates how many bits are valid in `s`.
    __device__
    static uint32_t signs_xor(uint32_t s_in, uint32_t signs_prod, int count)
    {
        uint32_t xor_mask = 0;
        if(signs_prod != 0)
        {
            xor_mask = ((1 << count) - 1) << (32 - count);
        }
        return (s_in ^ xor_mask);
    }
};

////////////////////////////////////////////////////////////////////////
// sign_store_policy_src
// This policy stores the "source" sign of the input values in the
// signs member of the cC2V struct. (In contrast, it is possible to
// modify the signs that were encountered, based on the product of all
// signs - see sign_policy_store_dst.)
// Usage: 
//     - Store min0 with a sign the same as the product of all signs.
//     - In the apply_sign() function, since the min0 value will have a
//       sign bit equal to the product of the signs, set the sign of
//       the result using the bitwise xor operator (^).
template <typename T, class TSignOrder> struct sign_store_policy_src;

template <class TSignOrder>
struct sign_store_policy_src<float, TSignOrder> : public TSignOrder
{
    typedef TSignOrder inherited_t;
    //------------------------------------------------------------------
    // finalize()
    static __device__
    void finalize(word_t&   min0,
                  word_t&   min1_or_delta,
                  uint32_t& signs,
                  int       /*unused_row_count*/,
                  bool      set_min1)
    {
        uint32_t sign_prod = inherited_t::sign_product(signs);
        // We will store the product of all signs in the sign bit of
        // min0 and min1. When retrieving the value, we can then take
        // the xor of the desired input sign bit (retrieved from the
        // signs field) with the min0 sign to get the desired output sign.
        min0.u32          = (min0.u32          & 0x7FFFFFFF) | (sign_prod << 31);
        if(set_min1)
        {
            min1_or_delta.u32 = (min1_or_delta.u32 & 0x7FFFFFFF) | (sign_prod << 31);
        }
    }
    //------------------------------------------------------------------
    // apply_sign()
    static __device__
    word_t apply_sign(uint32_t signs, int index, word_t value)
    {
        word_t         out;
        out.u32 = inherited_t::sign_mask(signs, index) ^ value.u32;
        return out;
    }
};


////////////////////////////////////////////////////////////////////////
// sign_store_policy_dst
// This policy stores the "destination" sign of the input values in the
// signs member of the cC2V struct. (In contrast, it is possible to
// store the original signs that were encountered - see
// sign_store_policy_src.). The destination sign takes into account the
// product of all signs.
// Usage:
//     - Store min0 as a positive value.
//     - In the finalize function, change bits to reflect the desired
//       sign of the output.
//     - In the apply_sign() function, since the min0 value will be
//       stored as a positive value (using the absolute value), set the
//       sign using the bitwise or operator (|).
template <typename T, class TSignOrder> struct sign_store_policy_dst;

template <class TSignOrder>
struct sign_store_policy_dst<float, TSignOrder> : public TSignOrder
{
    typedef TSignOrder inherited_t;
    //------------------------------------------------------------------
    // finalize()
    static __device__ __inline__
    void finalize(word_t&   min0,
                  word_t&   min1_or_delta,
                  uint32_t& signs,
                  uint32_t  sign_prod,
                  int       row_count,
                  bool      set_min1)
    {
        // The signs member has the current signs of each index
        // in the row. When we expand, we want the product of
        // signs, but without a specific index. We convert here
        // to what the actual sign will be, to simplify that
        // operation, by taking the XOR with a bitmask that has
        // each bit set to the overall parity (1s if odd, 0s if even).
        // For example, consider a row with 19 elements and the
        // following signs:
        //  18  16        12         8         4         0
        // x 0 0 1 | 1 0 1 0 | 1 0 1 0 | 0 0 0 0 | 1 1 1 1
        // The number of 1s is odd, so popc(signs) & 0x1 = 1,
        // and after shifting and subtracting we get:
        //  18  16        12         8         4         0
        // x 1 1 1 | 1 1 1 1 | 1 1 1 1 | 1 1 1 1 | 1 1 1 1
        // Taking the XOR with the original signs provides the
        // product of signs that will occur WITHOUT that specific
        // value:
        //  18  16        12         8         4         0
        // x 1 1 0 | 0 1 0 1 | 0 1 0 1 | 1 1 1 1 | 0 0 0 0
        signs = inherited_t::signs_xor(signs, sign_prod, row_count);

        // Store the absolute value of min0 and min1
        min0.f32 = fabsf(min0.f32);
        if(set_min1)
        {
            min1_or_delta.f32 = fabsf(min1_or_delta.f32);
        }
    }
    static __device__ __inline__
    word_t apply_sign(uint32_t signs, int index, word_t value)
    {
        word_t         out;
        out.u32 = inherited_t::sign_mask(signs, index) | value.u32;
        return out;
    }
};

//template <>
//struct sign_store_policy_dst<__half>
//{
//    //------------------------------------------------------------------
//    // finalize()
//    template <class TC2V>
//    static __device__ __inline__
//    void finalize(TC2V& c2v, int row_count, uint32_t sign_prod, bool set_min1)
//    {
//        // The signs member has the current signs of each index
//        // in the row. When we expand, we want the product of
//        // signs, but without a specific index. We convert here
//        // to what the actual sign will be, to simplify that
//        // operation, by taking the XOR with a bitmask that has
//        // each bit set to the overall parity (1s if odd, 0s if even).
//        // For example, consider a row with 19 elements and the
//        // following signs:
//        //  18  16        12         8         4         0
//        // x 0 0 1 | 1 0 1 0 | 1 0 1 0 | 0 0 0 0 | 1 1 1 1
//        // The number of 1s is odd, so popc(signs) & 0x1 = 1,
//        // and after shifting and subtracting we get:
//        //  18  16        12         8         4         0
//        // x 1 1 1 | 1 1 1 1 | 1 1 1 1 | 1 1 1 1 | 1 1 1 1
//        // Taking the XOR with the original signs provides the
//        // product of signs that will occur WITHOUT that specific
//        // value:
//        //  18  16        12         8         4         0
//        // x 1 1 0 | 0 1 0 1 | 0 1 0 1 | 1 1 1 1 | 0 0 0 0
//        c2v.signs_xor(sign_prod, row_count);
//
//        if(set_min1)
//        {
//            // Clear sign bits for both values
//            c2v.min0_min1.u32 |= 0x7FFF7FFF;
//        }
//        else
//        {
//            // Clear the sign bit for min0
//            c2v.min0_min1.u32 |= 0xFFFF7FFF;
//        }
//    }
//    template <class TC2V>
//    static __device__ __inline__
//    half_word_t apply_sign(const TC2V& c2v, int index, half_word_t value)
//    {
//        half_word_t         out;
//        out.u16 = c2v.sign_mask(index) | value.u16;
//        return out;
//    }
//};


} // namespace ldpc2

#endif // !defined(LDPC2_SIGN_CUH_INCLUDED_)
