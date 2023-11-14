/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_SIGN_SPLIT_CUH_INCLUDED_)
#define LDPC2_SIGN_SPLIT_CUH_INCLUDED_

#include "ldpc2.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// split_sign_update_bit_ops
// Class to perform "updates" of a 32-bit word that stores signs, using
// bitwise operations.
struct split_sign_update_bit_ops
{
    //------------------------------------------------------------------
    // init_row()
    __device__
    static void init_row(uint32_t& s, word_t v0_v1)
    {
        word_t smask = word_pair_sign_mask(v0_v1);
        s            = (smask.u32 >> 15);
    }
    //------------------------------------------------------------------
    // update()
    // Updates the given signs storage word with signs from the given
    // pair of values.
    // (Index pair 0 represents values 0 and 1, pair 1 represents values
    // 2 and 3, etc.)
    __device__
    static void update(uint32_t& s, word_t value_pair, int pair_index)
    {
        word_t smask = word_pair_sign_mask(value_pair);
        s |= (smask.u32 >> (15 - pair_index));
    }
    //------------------------------------------------------------------
    // update_low()
    // Updates the given signs storage word with signs from low half
    // word from the input value.
    // (Index pair 0 represents values 0 and 1, pair 1 represents values
    // 2 and 3, etc.)
    __device__
    static void update_low(uint32_t& s, word_t value, int pair_index)
    {
        word_t smask = half_word_sign_mask(value);
        s |= (smask.u32 >> (15 - pair_index));
    }
};

////////////////////////////////////////////////////////////////////////
// split_sign_update_fp
// Uses floating point (fp16) instructions to update stored signs that
// are stored in "split" modes (i.e. signs from 0, 2, 4, 6, ... in the
// low half word and signs from 1, 3, 5, 7, ... in the high half word.)
struct split_sign_update_fp
{
    //------------------------------------------------------------------
    // init_row()
    __device__
    static void init_row(uint32_t& s, word_t v0_v1)
    {
        s = update_signs_fp_pair(0, v0_v1, 0);
    }
    //------------------------------------------------------------------
    // update()
    // Updates the given signs storage word with signs from the given
    // pair of values.
    // (Index pair 0 represents values 0 and 1, pair 1 represents values
    // 2 and 3, etc.)
    __device__
    static void update(uint32_t& s, word_t value_pair, int pair_index)
    {
        s = update_signs_fp_pair(s, value_pair, pair_index);
    }
    //------------------------------------------------------------------
    // update_low()
    // Updates the given signs storage word with signs from low half
    // word from the input value.
    // (Index pair 0 represents values 0 and 1, pair 1 represents values
    // 2 and 3, etc.)
    __device__
    static void update_low(uint32_t& s, word_t value, int pair_index)
    {
        s = update_signs_fp_low(s, value, pair_index);
    }
};

template <typename T, class TSignUpdate> struct sign_store_policy_split_base;

////////////////////////////////////////////////////////////////////////
// sign_store_policy_split_base<__half>
// Sign manager for fp16. Signs are maintained "pairwise" - they are
// extracted from a pair of fp16 values and shifted together.
//
//  31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16 15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
//   s  s  s  s  s    17 16 15 13 11  9  7  5  3  1                   18 16 14 12 10  8  6  4  2  0
//
// Since the maximum row degree is 19 for BG1 and BG2, we will assume
// below that the most significant 5 bits are used for storing the
// index when the signs and indices are stored in a 32-bit word.
template <class TSignUpdate> struct sign_store_policy_split_base<__half, TSignUpdate>
{
    //------------------------------------------------------------------
    // from_packed_word()
    __device__
    static uint32_t from_packed_word(uint32_t s)
    {
        return (s & 0x07FFFFFF);
    }
    //------------------------------------------------------------------
    // to_packed_word()
    __device__
    static uint32_t to_packed_word(uint32_t s, int idx)
    {
        return (s & 0x07FFFFFF) | ((uint32_t)idx << 27);
    }
    //------------------------------------------------------------------
    // non_sign_bits()
    __device__
    static uint32_t non_sign_bits(uint32_t s)
    {
        return (s >> 27);
    }

    //------------------------------------------------------------------
    // init_row()
    template <int ROW_DEGREE>
    __device__
    static void init_row(uint32_t& s, word_t v0_v1)
    {
        TSignUpdate::init_row(s, v0_v1);
    }
    //------------------------------------------------------------------
    // pop_count()
    // Returns the number of negative sign bits. Assumes that most
    // significant bits (typically used for min0_index) are zero.
    __device__
    static int pop_count(const uint32_t& s)
    {
        return __popc(s);
    }
    //------------------------------------------------------------------
    // sign_mask()
    // Returns a mask with the appropriate sign bits set at bits 15 and
    // 31, based on the stored sign bits for the given index pair.
    // (Index pair 0 represents values 0 and 1, pair 1 represents values
    // 2 and 3, etc.)
    __device__
    static uint32_t sign_mask(const uint32_t& s, int pair_index)
    {
        const uint32_t smask = (s << (15 - pair_index));
        return (smask & 0x80008000);
    }
    //------------------------------------------------------------------
    // sign_product_mask()
    // Returns 0x80008000 or 0x00000000, based on whether the product
    // of all signs is negative(0x80008000) or positive (0x00000000).
    // Assumes that the most significant bits (sometimes used for
    // storage of an index value) are 0.
    __device__
    static uint32_t sign_product_mask(const uint32_t& s)
    {
        uint32_t sprod = (uint32_t)pop_count(s) & 0x1;
        return ((sprod << 15) | (sprod << 31));
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
    // update()
    // Updates the given signs storage word with signs from the given
    // pair of values.
    // (Index pair 0 represents values 0 and 1, pair 1 represents values
    // 2 and 3, etc.)
    __device__
    static void update(uint32_t& s, word_t value_pair, int pair_index)
    {
        TSignUpdate::update(s, value_pair, pair_index);
    }
    //------------------------------------------------------------------
    // update()
    // Updates the given signs storage word with signs from the given
    // pair of values.
    // (Index pair 0 represents values 0 and 1, pair 1 represents values
    // 2 and 3, etc.)
    __device__
    static void update_low(uint32_t& s, word_t value, int pair_index)
    {
        TSignUpdate::update_low(s, value, pair_index);
    }
    //------------------------------------------------------------------
    // signs_xor()
    // xor() sign bits in s with the product of the signs.
    // `pair_count` indicates how many value pairs have sign bits set in
    // `s`.
    // This can be used to store the "destination" sign bits: what the
    // output will be when the compressed C2V is extracted.
    __device__
    static void signs_xor(uint32_t& s, int pair_count)
    {
        uint32_t even_odd_mask = 0;
        if(0 != ((unsigned int)pop_count(s) & 0x1))
        {
            // This way maintains zeros in the gap between the valid
            // sign regions. We may not need to do so, but popc()
            // will be wrong with 1s in those positions, so we make
            // sure they are 0 here.
            even_odd_mask = (1 << pair_count) - 1;
            even_odd_mask |= (even_odd_mask << 16);
        }
        s ^= even_odd_mask;
    }
};

////////////////////////////////////////////////////////////////////////
// sign_store_policy_split_src
// T: APP type (__half, __half2, ...)
// TSignUpdate: sign update class, responsible for the selection of the
//              instruction sequence used to update the sign word
// TClamp: Boolean value to indicate whether the min0 and min1 values
//         are clamped in the finalize() function, to protect against
//         NaN values that can occur when APP values become infinite.
//         If TClamp is false, no clamping is performed, and it is
//         assumed that input values are constrained such that infinite
//         values will not occur (e.g., by clamping values at input).
template <typename T, class TSignUpdate, bool TClamp> struct sign_store_policy_split_src;

////////////////////////////////////////////////////////////////////////
// sign_store_policy_split_src<__half>
// This class assumes that the signs word will, after the finalize()
// function is called, store the "source" signs (e.g. the sign of the
// input value to the compressed C2V update function). In general,
// this requires that the product of the signs be stored in the min0
// and min1 values, and the output sign can be obtained via the xor
// of the "src" sign and the sign product.
template <class TSignUpdate, bool TClamp>
struct sign_store_policy_split_src<__half, TSignUpdate, TClamp> : public sign_store_policy_split_base<__half, TSignUpdate>
{
    typedef sign_store_policy_split_base<__half, TSignUpdate> inherited_t;
    //------------------------------------------------------------------
    // finalize()
    __device__
    static void finalize(word_t& min1_min0, uint32_t& signs, int /*pair_count_unused*/)
    {
        // We will store the product of all signs in the sign bits of
        // min0 and min1. When retrieving the value, we can then take
        // the xor of the desired input sign bit (retrieved from the
        // signs field) with the min0 sign to get the desired output sign.
        uint32_t min1_min0_mask = inherited_t::sign_product_mask(signs);
        min1_min0.u32 = (min1_min0.u32 & 0x7FFF7FFF) | min1_min0_mask;

        if(TClamp)
        {
            // Clamp to +/- FP16_max to avoid subtracting +/-Inf during
            // the next iteration. (-Inf - (-Inf) = NaN, Inf - Inf = NaN.)
            // In this case, min1 and min0 are signed values, so we must
            // clamp to +/- FP16_max.
            min1_min0 = clamp_to_half_max(min1_min0);
        }
    }
    //------------------------------------------------------------------
    // apply_sign()
    __device__
    static word_t apply_sign(word_t v1_v0, uint32_t signs, int pair_index)
    {
        word_t out;
        out.u32 = inherited_t::sign_mask(signs, pair_index) ^ v1_v0.u32;
        return out;
    }
};

////////////////////////////////////////////////////////////////////////
// sign_store_policy_split_dst
// T: APP type (__half, __half2, ...)
// TSignUpdate: sign update class, responsible for the selection of the
//              instruction sequence used to update the sign word
// TClamp: Boolean value to indicate whether the min0 and min1 values
//         are clamped in the finalize() function, to protect against
//         NaN values that can occur when APP values become infinite.
//         If TClamp is false, no clamping is performed, and it is
//         assumed that input values are constrained such that infinite
//         values will not occur (e.g., by clamping values at input).
template <typename T, class TSignUpdate, bool TClamp> struct sign_store_policy_split_dst;

////////////////////////////////////////////////////////////////////////
// sign_store_policy_split_dst<__half>
// This class assumes that the signs word will, after the finalize()
// function is called, store the "destination" signs (e.g. the sign of
// the output value from the extract() function). In general,
// this requires that the source signs be optionally modified using
// the product of the signs, and the min0/min1 values are stored in
// absolute value form (e.g. the positive values are stored).
template <class TSignUpdate, bool TClamp>
struct sign_store_policy_split_dst<__half, TSignUpdate, TClamp> : public sign_store_policy_split_base<__half, TSignUpdate>
{
    typedef sign_store_policy_split_base<__half, TSignUpdate> inherited_t;
    //------------------------------------------------------------------
    // finalize()
    __device__
    static void finalize(word_t& min1_min0, uint32_t& signs, int pair_count)
    {
        // Before this function is called, the signs value has the
        // current signs of each input in the row. When we expand,
        // we want the product of signs, but without the sign
        // contribution of a specific index. We convert here
        // to what the actual output sign will be, to simplify that
        // operation, by taking the XOR with a bitmask that has
        // each bit set to the overall parity (1s if odd, 0s if even).
        inherited_t::signs_xor(signs, pair_count);

        // Store the absolute value of min0 and min1
        min1_min0.u32 = (min1_min0.u32 & 0x7FFF7FFF);

        if(TClamp)
        {
            // Clamp to +FP16_max to avoid subtracting Inf during the next
            // iteration. (Inf - Inf = NaN.)
            // In this case, min1 and min0 are stored as positive values,
            // so we only need to clamp to +FP16_max.
            min1_min0 = clamp_pos_to_half_max(min1_min0);
        }
    }
    //------------------------------------------------------------------
    // apply_sign()
    __device__
    static word_t apply_sign(word_t v1_v0, uint32_t signs, int pair_index)
    {
        word_t out;
        out.u32 = inherited_t::sign_mask(signs, pair_index) | v1_v0.u32;
        return out;
    }
};

} // namespace ldpc2

#endif // !defined(LDPC2_SIGN_SPLIT_CUH_INCLUDED_)
