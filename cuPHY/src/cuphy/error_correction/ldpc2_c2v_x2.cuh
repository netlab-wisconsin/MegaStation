/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_C2V_X2_CUH_INCLUDED_)
#define LDPC2_C2V_X2_CUH_INCLUDED_

// Classes/structs for LDPC min-sum decode kernels that work with a pair
// of codewords per CTA.

#include "ldpc2.cuh"
#include "ldpc2_c2v.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// signs_pair_high_degree_split
// Sign bit storage and management for "2 codewords at a time", for high
// degree (greater than 10) rows. Internally, signs for each codeword
// are "split" (i.e. stored in two different words). This helps when
// using the same data structure for all rows - the compiler can
// (hopefully) determine that the second word is not used for lower
// degree rows.
//
//  Bit
//  31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16   15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
//
// The first 10 bits of the stored signs will be in the first signs word,
// ordered as follows:
//
//  Word 1 Sign Bits                                  Word 0 Sign Bits
//  ------------------------------------------------|------------------------------------------------
//   9  8  7  6  5  4  3  2  1  0  .  .  .  .  .  . |  9  8  7  6  5  4  3  2  1  0  .  .  .  .  .  .
//                              ^                                                 ^
//                   Row Index  |                                      Row Index  |
//
//  Word 1 Sign Bits                                  Word 0 Sign Bits
//  ------------------------------------------------|------------------------------------------------
//  18 17 16 15 14 13 12 11 10  .  .  .  .  .  .  . | 18 17 16 15 14 13 12 11 10  .  .  .  .  .  .  .
//        ^                                                  ^
//        | Row Index                                        |  Row Index
struct signs_pair_high_degree_split
{
    //------------------------------------------------------------------
    // Data
    uint32_t signs_0_9;
    uint32_t signs_10_;
    //------------------------------------------------------------------
    // signs_pair_high_degree_split()
    signs_pair_high_degree_split() = default;
    //------------------------------------------------------------------
    // signs_pair_high_degree_split()
    __device__
    signs_pair_high_degree_split(uint32_t s0_9, uint32_t s10_) :
        signs_0_9(s0_9),
        signs_10_(s10_)
    {
    }
    //------------------------------------------------------------------
    // init_row()
    // Initialize the internal sign bit representation with a pair of
    // words.
    __device__
    void init_row(word_t v0, word_t v1)
    {
        word_t s0 = fp16x2_sign_mask(v0);
        word_t s1 = fp16x2_sign_mask(v1);
        signs_0_9 = (s0.u32 >> 9) | (s1.u32 >> 8);
        signs_10_ = 0;
    }
    //------------------------------------------------------------------
    // update()
    // Update the internal sign bit representation for two individual
    // codewords with the contents of the high and low halves of 'v'.
    __device__
    void update(word_t v, int idx)
    {
        word_t sv = fp16x2_sign_mask(v);
        if(idx < 10) // compile time branch with unrolled loops...
        {
            signs_0_9 |= (sv.u32 >> (9-idx));
        }
        else
        {
            signs_10_ |= (sv.u32 >> (18-idx));
        }
    }
    //------------------------------------------------------------------
    // sign_product_mask()
    // Returns a value with bits 15 and 31 set or cleared, based on the
    // product of the internal stored signs for each of the two
    // codewords represented. All other bits will be zero.
    __device__
    uint32_t sign_product_mask() const
    {
        // Gather signs from 0-9 and 10+
        const uint32_t CW0_signs = (signs_0_9 & 0x0000FFC0) | (signs_10_ << 16);
        const uint32_t CW1_signs = (signs_0_9 >> 16)        | (signs_10_ & 0xFFC00000);
        const uint32_t CW0_sign_prod = (unsigned int)__popc(CW0_signs) & 0x1;
        const uint32_t CW1_sign_prod = (unsigned int)__popc(CW1_signs) & 0x1;
        
        return (CW0_sign_prod << 15) | (CW1_sign_prod << 31);
    }
    //------------------------------------------------------------------
    // sign_mask()
    // Returns a mask with the appropriate sign bits set at bits 15 and
    // 31, based on the stored sign bits for the given index pair.
    __device__
    uint32_t sign_mask(int idx) const
    {
        const uint32_t smask = (idx < 10) ? (signs_0_9 << (9-idx)) : (signs_10_ << (18-idx)); 
        return (smask & 0x80008000);
    }
};

////////////////////////////////////////////////////////////////////////
// signs_pair_low_degree
// Sign bit storage and management for "2 codewords at a time", for high
// degree (greater than 10) rows. Internally, signs are stored in a
// single word.
//
//  Bit
//  31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16   15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
//
// The first 10 bits of the stored signs will be in the first signs word,
// ordered as follows:
//
//  Word 1 Sign Bits                                  Word 0 Sign Bits
//  ------------------------------------------------|------------------------------------------------
//   9  8  7  6  5  4  3  2  1  0  .  .  .  .  .  . |  9  8  7  6  5  4  3  2  1  0  .  .  .  .  .  .
//                              ^                                                 ^
//                   Row Index  |                                      Row Index  |
struct signs_pair_low_degree
{
    //------------------------------------------------------------------
    // Data
    uint32_t signs_0_9;
    //------------------------------------------------------------------
    // signs_pair_low_degree()
    signs_pair_low_degree() = default;
    //------------------------------------------------------------------
    // signs_pair_low_degree()
    __device__
    signs_pair_low_degree(uint32_t u) : signs_0_9(u) {}
    //------------------------------------------------------------------
    // init_row()
    // Initialize the internal sign bit representation with a pair of
    // words.
    __device__
    void init_row(word_t v0, word_t v1)
    {
        word_t s0 = fp16x2_sign_mask(v0);
        word_t s1 = fp16x2_sign_mask(v1);
        signs_0_9 = (s0.u32 >> 9) | (s1.u32 >> 8);
    }
    //------------------------------------------------------------------
    // update()
    // Update the internal sign bit representation for two individual
    // codewords with the contents of the high and low halves of 'v'.
    __device__
    void update(word_t v, int idx)
    {
        word_t sv = fp16x2_sign_mask(v);
        signs_0_9 |= (sv.u32 >> (9-idx));
    }
    //------------------------------------------------------------------
    // sign_product_mask()
    // Returns a value with bits 15 and 31 set or cleared, based on the
    // product of the internal stored signs for each of the two
    // codewords represented. All other bits will be zero.
    __device__
    uint32_t sign_product_mask() const
    {
        const uint32_t CW0_sign_prod = (unsigned int)__popc(signs_0_9 & 0x0000FFC0) & 0x1;
        const uint32_t CW1_sign_prod = (unsigned int)__popc(signs_0_9 & 0xFFC00000) & 0x1;
        return (CW0_sign_prod << 15) | (CW1_sign_prod << 31);
    }
    //------------------------------------------------------------------
    // sign_mask()
    // Returns a mask with the appropriate sign bits set at bits 15 and
    // 31, based on the stored sign bits for the given index pair.
    __device__
    uint32_t sign_mask(int idx) const
    {
        const uint32_t smask = (signs_0_9 << (9-idx));
        return (smask & 0x80008000);
    }
};

////////////////////////////////////////////////////////////////////////
// sign_mgr_pair_src
// This class assumes that the signs word will, after the finalize()
// function is called, store the "source" signs (e.g. the sign of the
// input value to the compressed C2V update function). In general,
// this requires that the product of the signs be stored in the min0
// and min1 values, and the output sign can be obtained via the xor
// of the "src" sign and the sign product.
// T: APP type (__half, __half2, ...)
// TClamp: Boolean value to indicate whether the min0 and min1 values
//         are clamped in the finalize() function, to protect against
//         NaN values that can occur when APP values become infinite.
//         If TClamp is false, no clamping is performed, and it is
//         assumed that input values are constrained such that infinite
//         values will not occur (e.g., by clamping values at input).
template <bool TClamp>
struct sign_mgr_pair_src
{
    //------------------------------------------------------------------
    // finalize()
    template <class TSignsPair>
    static
    __device__
    void finalize(const TSignsPair& s_pair,
                  word_t&           min0,
                  word_t&           min1)
    {
        const uint32_t sign_prod_mask = s_pair.sign_product_mask();
        
        // We will store the product of all signs in the sign bits of
        // min0 and min1. When retrieving the value, we can then take
        // the xor of the desired input sign bit (retrieved from the
        // signs field) with the min0 sign to get the desired output sign.
        min0.u32 = (min0.u32 & 0x7FFF7FFF) | sign_prod_mask;
        min1.u32 = (min1.u32 & 0x7FFF7FFF) | sign_prod_mask;

        if(TClamp)
        {
            // Clamp to +/- FP16_max to avoid subtracting +/-Inf during
            // the next iteration. (-Inf - (-Inf) = NaN, Inf - Inf = NaN.)
            // In this case, min1 and min0 are signed values, so we must
            // clamp to +/- FP16_max.
            min0 = clamp_to_half_max(min0);
            min1 = clamp_to_half_max(min1);
        }
    }
    //------------------------------------------------------------------
    // apply_sign()
    template <class TSignsPair>
    static
    __device__
    word_t apply_sign(const TSignsPair& s_pair,
                      word_t            v,
                      int               index)
    {
        word_t out;
        out.u32 = s_pair.sign_mask(index) ^ v.u32;
        return out;
    }
};

////////////////////////////////////////////////////////////////////////
// cC2V_storage_x2_low_degree
// Compressed check-to-variable data storage for 2 codewords at a time
// implementations (rows with low degree only). In this case, "low
// degree" refers to 10 or less.
//
//  Bit
//  31 30 29 28 27 26 25 24 23 22 21 20 19 18 17 16   15 14 13 12 11 10  9  8  7  6  5  4  3  2  1  0
//
// The first 10 bits of the stored signs will be in the first signs word,
// ordered as follows:
//
//  Word 1 Sign Bits                                  Word 0 Sign Bits
//  ------------------------------------------------|------------------------------------------------
//   9  8  7  6  5  4  3  2  1  0  .  .  .  .  .  . |  9  8  7  6  5  4  3  2  1  0  .  .  .  .  .  .
//                              ^                                                 ^
//                   Row Index  |                                      Row Index  |
//
// Index of min0 will be 5 bits or less (max row degree is 19). The min0_index value for each word will
// be stored in the low bits of each half word.
//
//  Word 1 Min0 Index                                 Word 0 Min0 Index
//  ------------------------------------------------|------------------------------------------------
//   .  .  .  .  .  .  .  .  .  .  .  x  x  x  x  x |  .  .  .  .  .  .  .  .  .  .  .  x  x  x  x  x
struct cC2V_storage_x2_low_degree
{
    word_t   min0;
    word_t   min1;
    uint32_t signs_0_9_min0_index;
    //------------------------------------------------------------------
    // Typedef for struct to hold sign bits
    typedef signs_pair_low_degree signs_pair_t;
    //------------------------------------------------------------------
    // cC2V_storage_x2_low_degree()
    cC2V_storage_x2_low_degree() = default;
    //------------------------------------------------------------------
    // cC2V_storage_x2_low_degree()
    __device__
    cC2V_storage_x2_low_degree(word_t m0,
                               word_t m1,
                               word_t m0_index,
                               const signs_pair_t& s_pair) :
        min0(m0),
        min1(m1),
        signs_0_9_min0_index(s_pair.signs_0_9 | m0_index.u32)
    {
    }
    //------------------------------------------------------------------
    // init()
    __device__
    void init()
    {
        min0.u32 = 0; min1.u32 = 0; signs_0_9_min0_index = 0;
    }
    //------------------------------------------------------------------
    // get_signs_pair()
    __device__
    signs_pair_low_degree get_signs_pair() const
    {
        return signs_pair_low_degree(signs_0_9_min0_index & 0xFFC0FFC0);
    }
    //------------------------------------------------------------------
    // get_min0_index_pair()
    __device__
    word_t get_min0_index_pair() const
    {
        word_t w;
        w.u32 = (signs_0_9_min0_index & 0x001F001F);
        return w;
    }
#if 0
    //------------------------------------------------------------------
    // print()
    __device__
    void print() const
    {
        printf("min1 = [%f %f], min0 = [%f %f], index = [%i %i], signs = [0x%X 0x%X]\n",
               __high2float(min1.f16x2),
               __low2float(min1.f16x2),
               __high2float(min0.f16x2),
               __low2float(min0.f16x2),
               (signs_0_9_min0_index >> 16) & 0x1F,
               (signs_0_9_min0_index >> 0)  & 0x1F,
               (signs_0_9_min0_index >> 22),
               (signs_0_9_min0_index >> 6)  & 0x3FF);
    }
#endif
};

////////////////////////////////////////////////////////////////////////
// cC2V_storage_x2_high_degree_split
// Compressed check-to-variable data storage for 2 codewords at a time
// implementations (rows with high degree only). In this case, "high
// degree" refers to greater than 10 (i.e. the first 3 rows of BG1 only).
//
// "Split" refers to the fact that the sign bits for a single codeword
// are stored in two separate machine words.
//
// Storage of the first 10 sign bits, and the min0 index will be
// identical to the cC2V_storage_x2_low_degree case.
//
// The remainder of the stored signs will be in the second signs word,
// ordered as follows:
//
//  Word 1 Sign Bits                                  Word 0 Sign Bits
//  ------------------------------------------------|------------------------------------------------
//  18 17 16 15 14 13 12 11 10  .  .  .  .  .  .  . | 18 17 16 15 14 13 12 11 10  .  .  .  .  .  .  .
//        ^                                                  ^
//        | Row Index                                        |  Row Index
struct cC2V_storage_x2_high_degree_split
{
    word_t    min0;
    word_t    min1;
    uint32_t  signs_0_9_min0_index;
    uint32_t  signs_10_;
    //------------------------------------------------------------------
    // Typedef for struct to hold sign bits
    typedef signs_pair_high_degree_split signs_pair_t;
    //------------------------------------------------------------------
    // cC2V_storage_x2_high_degree_split()
    cC2V_storage_x2_high_degree_split() = default;
    //------------------------------------------------------------------
    // cC2V_storage_x2_high_degree_split()
    __device__
    cC2V_storage_x2_high_degree_split(word_t m0,
                                      word_t m1,
                                      word_t m0_index,
                                      const signs_pair_t& s_pair) :
        min0(m0),
        min1(m1),
        signs_0_9_min0_index(s_pair.signs_0_9 | m0_index.u32),
        signs_10_(s_pair.signs_10_)
    {
    }
    //------------------------------------------------------------------
    // init()
    __device__
    void init()
    {
        min0.u32 = 0; min1.u32 = 0; signs_0_9_min0_index = 0; signs_10_ = 0;
    }
    //------------------------------------------------------------------
    // get_signs_pair()
    __device__
    signs_pair_high_degree_split get_signs_pair() const
    {
        return signs_pair_high_degree_split(signs_0_9_min0_index & 0xFFC0FFC0,
                                            signs_10_);
    }
    //------------------------------------------------------------------
    // get_min0_index_pair()
    __device__
    word_t get_min0_index_pair() const
    {
        word_t w;
        w.u32 = (signs_0_9_min0_index & 0x001F001F);
        return w;
    }
#if 0
    //------------------------------------------------------------------
    // print()
    __device__
    void print() const
    {
        printf("min1 = [%f %f], min0 = [%f %f], index = [%i %i], signs = [0x%X 0x%X]\n",
               __high2float(min1.f16x2),
               __low2float(min1.f16x2),
               __high2float(min0.f16x2),
               __low2float(min0.f16x2),
               (signs_0_9_min0_index >> 16) & 0x1F,
               (signs_0_9_min0_index >> 0)  & 0x1F,
               (signs_0_9_min0_index >> 22) | ((signs_10_ >> 23) << 10),
               ((signs_0_9_min0_index >> 6) & 0x3FF) | ((signs_10_ << 16) >> 22));
    }
#endif
};

//------------------------------------------------------------------
// Template structure to provide type TA if the TIndex is less than
// TAIfLessThan, and TB otherwise.
template <int   TIndex,
          int   TAIfLessThan,
          class TA,
          class TB> struct if_less_than
{
    typedef typename std::conditional<(TIndex < TAIfLessThan), TA, TB>::type type;
};

////////////////////////////////////////////////////////////////////////
// Structure with expanded, in-register C2V representation (as opposed
// to one compressed for storage to global/shared memory), for temporary
// use in processing a single parity node.
template <class TSignMgr, class TMinSumUpdate, class TStorage>
struct cC2V_row_context<__half2, TSignMgr, TMinSumUpdate, TStorage>
{
    //------------------------------------------------------------------
    typedef TSignMgr                         sign_mgr_t;
    typedef TStorage                         storage_t;
    // Infer the type for storage of sign bits from the compressed
    // storage type. (High degree rows may require 2 words, whereas low
    // degree rows may only need one.)
    typedef typename storage_t::signs_pair_t signs_pair_t;
    //------------------------------------------------------------------
    __device__
    cC2V_row_context() {}
    //------------------------------------------------------------------
    // Constructor
    // Initialize a row context from the storage structure (which is
    // typically as small as possible to conserve space and bandwidth)
    template <int ROW_DEGREE>
    __device__
    cC2V_row_context(const storage_t& s,
                     std::integral_constant<int, ROW_DEGREE>) :
        min0(s.min0),
        min1(s.min1),
        min0_index(s.get_min0_index_pair()),
        signs_pair(s.get_signs_pair())
    {
    }
    //------------------------------------------------------------------
    // Constructor
    // Initialize a row context from a sequence of APP values,
    // proceeding through the sequence and updating the min-sum
    // representation.
    template <int ROW_DEGREE, int MAX_WORDS>
    __device__
    explicit cC2V_row_context(const __half2&                          norm,
                              word_t                                  (&app)[MAX_WORDS],
                              std::integral_constant<int, ROW_DEGREE>)
    {
        // Initialize with first two values
        init_row(app[0], app[1]);
        // Update min-sum with the rest of the values
        #pragma unroll
        for(int i = 2; i < ROW_DEGREE; ++i)
        {
            update(app[i], i);
        }
        // Post-process row context to prepare for extraction
        finalize(norm, ROW_DEGREE);
    }
    //------------------------------------------------------------------
    // init_row()
    // Initialize row context min-sum fields with the first pair of
    // sequence values
    __device__
    void init_row(word_t v0, word_t v1)
    {
        word_t cZeros, cOnes;
        cZeros.u16x2 = ushort2{0, 0};
        cOnes.u16x2 = ushort2{1, 1};
        
        word_t absv0_lt_absv1 = hset2_bm_lt(fp16x2_abs(v0), fp16x2_abs(v1));
        min0                  = select_from_mask(absv0_lt_absv1, v0, v1);
        min1                  = select_from_mask(absv0_lt_absv1, v1, v0);
        min0_index            = select_from_mask(absv0_lt_absv1, cZeros, cOnes);
        signs_pair.init_row(v0, v1);
    }
    //------------------------------------------------------------------
    // update()
    // Update the internal representation with a word containing a new
    // value for each codeword.
    __device__
    void update(word_t v, int idx)
    {
        word_t maybe_min1, idx_pair;

        idx_pair.u16x2           = ushort2{static_cast<unsigned short>(idx),
                                           static_cast<unsigned short>(idx)};
        word_t absv_lt_absmin0   = hset2_bm_lt(fp16x2_abs(v), fp16x2_abs(min0));            // v < min0?
        min0_index               = select_from_mask(absv_lt_absmin0, idx_pair, min0_index); // min0_index   = (v < min0) ? idx ? min0_index
        maybe_min1               = select_from_mask(absv_lt_absmin0, min0, v);              // maybe_min1 = (v < min0) ? min0 : v
        min0                     = select_from_mask(absv_lt_absmin0, v, min0);              // min0       = (v < min0) ? v : min0
#if CUDART_VERSION >= 11000 && __CUDA_ARCH__ >= 800
        // We need to be careful here - assuming that stripping the sign
        // from min1 is OK with the sign manager.
        min1.f16x2               = __hmin2(__habs2(maybe_min1.f16x2), __habs2(min1.f16x2));
#else
        word_t abstmp_lt_absmin1 = hset2_bm_lt(fp16x2_abs(maybe_min1), fp16x2_abs(min1));   // maybe_min1 < min1?
        min1                     = select_from_mask(abstmp_lt_absmin1, maybe_min1, min1);   // min1       = (maybe_min1 < min1) ? maybe_min1 : min1
#endif
        signs_pair.update(v, idx);
    }
    //------------------------------------------------------------------
    // finalize()
    __device__
    void finalize(const __half2& norm, int row_degree)
    {
        // Apply normalization to both values (min1 and min0)
        // TODO: fuse with add of update to APP value
        min0.f16x2 = __hmul2(min0.f16x2, norm);
        min1.f16x2 = __hmul2(min1.f16x2, norm);

        // Adjust signs of min0, min1 values (optionally)
        sign_mgr_t::finalize(signs_pair, min0, min1);
    }
    //------------------------------------------------------------------
    // get_storage()
    // Initialize a storage structure from the internal min-sum
    // representation
    __device__
    storage_t get_storage() const
    {
        return storage_t(min0, min1, min0_index, signs_pair);
    }
    //------------------------------------------------------------------
    // extract_pair()
    // Extract a pair of values using the internal representation. The
    // returned word will contain two values, with the low word having
    // the sequence value with the lower index.
    __device__
    word_t extract_pair(int idx) const
    {
        word_t idx_pair;
        // Create a fp16x2 word with the index value replicated, and
        // use denormalized float comparisons to compare the index to
        // the min0_index for each codeword
        idx_pair.u16x2       = ushort2{static_cast<unsigned short>(idx),
                                       static_cast<unsigned short>(idx)};
        word_t idx_is_min0   = hequ_bm(idx_pair, min0_index);             // idx == min0_index?
        word_t ex_value      = select_from_mask(idx_is_min0, min1, min0); // ex_value = (idx_is_min0) ? min1 : min0
        return sign_mgr_t::apply_sign(signs_pair, ex_value, idx);
    }
    //------------------------------------------------------------------
    // Data
    word_t       min0;        // (fp16x2 for 2 codewords)
    word_t       min1;        // (fp16x2 for 2 codewords)
    word_t       min0_index;  // index of min 0 for each of 2 codewords
    signs_pair_t signs_pair;  // 1 or 2 words of sign bits, depending on
                              // row degree
};

template <class TRowContext>
class cC2V_row_proc<__half2, TRowContext>
{
private:
    //------------------------------------------------------------------
    typedef TRowContext row_context_t;
    typedef __half2     app_t;
public:
    //------------------------------------------------------------------
    // process_row()
    template <int ROW_DEGREE, int UPDATE_ROW_DEGREE, class TStorage>
    __device__
    static void process_row(word_t          (&app)[ROW_DEGREE],
                            TStorage&       row_storage,
                            const __half2&  norm)
    {
        app_sub_prev_iter<ROW_DEGREE, UPDATE_ROW_DEGREE>(app, row_storage);
        app_update<ROW_DEGREE, UPDATE_ROW_DEGREE>(row_storage, norm, app);
    }
private:
    //------------------------------------------------------------------
    // app_sub_prev_iter()
    template <int ROW_DEGREE, int UPDATE_ROW_DEGREE, class TStorage>
    __device__
    static void app_sub_prev_iter(word_t          (&app)[ROW_DEGREE],
                                  const TStorage& row_storage)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Initialize a row processing context using data from the
        // previous iteration (which may be stored in registers or
        // global memory).
        row_context_t rc(row_storage, std::integral_constant<int, ROW_DEGREE>{});
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Update the APP values with the decrement from the previous
        // iteration. We don't need to update APP values for extension
        // nodes.
        #pragma unroll
        for(int i = 0; i < UPDATE_ROW_DEGREE; ++i)
        {
            // Operate on a pair of values at a time
            word_t dec = rc.extract_pair(i);
            app[i].f16x2 = __hsub2(app[i].f16x2, dec.f16x2);
        }
    }
    //------------------------------------------------------------------
    // app_update()
    template <int ROW_DEGREE, int UPDATE_ROW_DEGREE, class TStorage>
    __device__
    static void app_update(TStorage&            row_storage,
                           const __half2&       norm,
                           word_t               (&app)[ROW_DEGREE])
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Construct a row context using the updated APP values (with
        // values from the previous iteration subtracted). This will
        // create a min-sum representation of the APP values.
        row_context_t rc(norm,
                         app,
                         std::integral_constant<int, ROW_DEGREE>{});
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Extract values from the new row context and update the APP
        // values.
        #pragma unroll
        for(int i = 0; i < UPDATE_ROW_DEGREE; ++i)
        {
            word_t inc = rc.extract_pair(i);
            // TODO: Fuse multiply-add of normalization in finalize()
            // here
            app[i].f16x2 = __hadd2(app[i].f16x2, inc.f16x2);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Store the compressed representation in row storage.
        row_storage = rc.get_storage();
    }
};

////////////////////////////////////////////////////////////////////////
// cC2V_index
// Specialization for two codewords at a time as independent "high" and
// "low" fp16 values in a __half2 APP data type.
template <int                                 BG,
          class                               TSignManager,
          class                               TMinSumUpdate,
          template <typename, int> class      TAPPLoader,
          template <typename, int, int> class TAPPWriter>
struct cC2V_index<__half2, BG, TSignManager, TMinSumUpdate, TAPPLoader, TAPPWriter>
{
    //typedef C2V_storage_t<__half2, 4>                                             c2v_storage_t;
    //typedef cC2V_row_context<__half2, TSignManager, TMinSumUpdate, c2v_storage_t> row_context_t;
    typedef __half2                                                               app_t;
    //------------------------------------------------------------------
    // app_sub_prev_iter()
    template <int CHECK_IDX, class TRowContext>
    __device__
    void app_sub_prev_iter(word_t             (&app)[app_num_words<__half2, BG, CHECK_IDX>::value],
                           const TRowContext& rc)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Update the APP values with the decrement from the previous
        // iteration. We don't need to update APP values for extension
        // nodes.
        #pragma unroll
        for(int i = 0; i < update_row_degree<BG, CHECK_IDX>::value; ++i)
        {
            // Operate on a pair of values at a time
            word_t dec = rc.extract_pair(i);

            // TODO: Fuse normalization from prev iteration?
            app[i].f16x2 = __hsub2(app[i].f16x2, dec.f16x2);
        }
    }
    //------------------------------------------------------------------
    // app_update()
    template <int CHECK_IDX, class TRowContext>
    __device__
    void app_update(word_t             (&app)[app_num_words<__half2, BG, CHECK_IDX>::value],
                    const TRowContext& rc)
    {
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Extract values from the new row context and update the APP
        // values.
        #pragma unroll
        for(int i = 0; i < update_row_degree<BG, CHECK_IDX>::value; ++i)
        {
            word_t inc = rc.extract_pair(i);
            // TODO: Fuse multiply-add of normalization in finalize()
            // here
            app[i].f16x2 = __hadd2(app[i].f16x2, inc.f16x2);
        }
    }
    //------------------------------------------------------------------
    // process_row()
    template <int CHECK_IDX, class TKernelParams, class TC2VStorage>
    __device__
    void process_row(const TKernelParams& params,
                     word_t               (&app)[app_num_words<__half2, BG, CHECK_IDX>::value],
                     int                  (&app_addr)[row_degree<BG, CHECK_IDX>::value],
                     TC2VStorage&         c2v_storage,
                     int                  smem_offset)
    {
        typedef cC2V_row_context<__half2,
                                 TSignManager,
                                 TMinSumUpdate,
                                 TC2VStorage> row_context_t;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Load APP values into registers
        typedef TAPPLoader<__half2, row_degree<BG, CHECK_IDX>::value> app_loader_t;
        app_loader_t::load(app, app_addr, smem_offset);
        {
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Initialize a row processing context using data from the
            // previous iteration (which may be stored in registers or
            // global memory).
            row_context_t rc(c2v_storage,
                             std::integral_constant<int, row_degree<BG, CHECK_IDX>::value>{});

            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Update the APP values with the decrement from the previous
            // iteration.
            app_sub_prev_iter<CHECK_IDX>(app, rc);
        }
        {
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Construct a row context using the updated APP values (with
            // values from the previous iteration subtracted). This will
            // create a min-sum representation of the APP values.
            row_context_t rcNew(params.norm.f16x2,
                                app,
                                std::integral_constant<int, row_degree<BG, CHECK_IDX>::value>{});

            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Store compressed representation in member variable. It may
            // (or may not) separately be saved to global memory.
            c2v_storage = rcNew.get_storage();

            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Extract values from the new row context and update the APP
            // values.
            app_update<CHECK_IDX>(app, rcNew);
        }
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Write output values to shared_memory
        typedef TAPPWriter<__half2,
                           row_degree       <BG, CHECK_IDX>::value,
                           update_row_degree<BG, CHECK_IDX>::value> app_writer_t;
        app_writer_t::write_non_ext(app, app_addr, smem_offset);
    }
    //------------------------------------------------------------------
    // load_shared_strided()
    //__device__
    //void load_shared_strided(const LDPC_kernel_params& params,
    //                         int                       checkIdx,
    //                         int                       threadByteOffset,
    //                         int                       strideBytes)
    //{
    //    if((BG != 1) || checkIdx >= 4)
    //    {
    //        #pragma unroll
    //        for(int i = 0; i < c2v_storage_t::NUM_WORDS_SMALL; ++i)
    //        {
    //            c2v_storage.v.words[i] = smem_address_as<word_t>(threadByteOffset + (i * strideBytes));
    //        }
    //    }
    //    else
    //    {
    //        #pragma unroll
    //        for(int i = 0; i < c2v_storage_t::NUM_WORDS; ++i)
    //        {
    //            c2v_storage.v.words[i] = smem_address_as<word_t>(threadByteOffset + (i * strideBytes));
    //        }
    //    }
    //}
    //------------------------------------------------------------------
    // store_shared_strided()
    //__device__
    //void store_shared_strided(const LDPC_kernel_params& params,
    //                          int                       checkIdx,
    //                          int                       threadByteOffset,
    //                          int                       strideBytes)
    //{
    //    if((BG != 1) || checkIdx >= 4)
    //    {
    //        #pragma unroll
    //        for(int i = 0; i < c2v_storage_t::NUM_WORDS_SMALL; ++i)
    //        {
    //            write_shared_word(c2v_storage.v.words[i], threadByteOffset + (i * strideBytes));
    //        }
    //    }
    //    else
    //    {
    //        #pragma unroll
    //        for(int i = 0; i < c2v_storage_t::NUM_WORDS; ++i)
    //        {
    //            write_shared_word(c2v_storage.v.words[i], threadByteOffset + (i * strideBytes));
    //        }
    //    }
    //}
};

//------------------------------------------------------------------
// "Core" rows in the 5G base graphs are the first 4 rows, and for
// each base graph these have the highest row degree (19 for BG1,
// 10 for BG2).
template <int BG> struct core_storage_x2;
template <> struct core_storage_x2<1>
{
    typedef cC2V_storage_x2_high_degree_split type;
};
template <> struct core_storage_x2<2>
{
    typedef cC2V_storage_x2_low_degree type;
};

////////////////////////////////////////////////////////////////////////
// box_plus_row_proc_x2
// Row processor for a box plus implementation that processes two
// codewords at a time. The low fp16 values of each word in the APP
// array have the values for one codeword, and the high fp16 values
// correspond to the second codeword.
template <class TBoxPlusOp>
class box_plus_row_proc_x2
{
public:
    //typedef TC2VStorage storage_t;
    //------------------------------------------------------------------
    // init()
    //__device__ static void init(storage_t& s) { s.init(); }
    //------------------------------------------------------------------
    // process_row()
    template <int ROW_DEGREE, int UPDATE_ROW_DEGREE, class TStorage>
    __device__
    static void process_row(word_t          (&app)[row_num_words<__half, ROW_DEGREE>::value],
                            TStorage&       row_storage,
                            const __half2&  norm)
    {
        app_sub_prev_iter<ROW_DEGREE, UPDATE_ROW_DEGREE>(app, row_storage);
        app_update<ROW_DEGREE, UPDATE_ROW_DEGREE>(row_storage, norm, app);
    }
private:
    //------------------------------------------------------------------
    // app_sub_prev_iter()
    template <int ROW_DEGREE, int UPDATE_ROW_DEGREE, class TStorage>
    __device__
    static void app_sub_prev_iter(word_t          (&app)[ROW_DEGREE],
                                  const TStorage& row_storage)
    {
        #pragma unroll
        for(int i = 0; i < UPDATE_ROW_DEGREE; ++i)
        {
            app[i].f16x2 = __hsub2(app[i].f16x2, row_storage.v.w[i].f16x2);
        }
    }
    //------------------------------------------------------------------
    // app_update()
    template <int ROW_DEGREE, int UPDATE_ROW_DEGREE, class TStorage>
    __device__
    static void app_update(TStorage&            row_storage,
                           const __half2&       norm,
                           word_t               (&app)[ROW_DEGREE])
    {
        word_t bp_update_seq[UPDATE_ROW_DEGREE];  // temporary storage

        //typedef box_plus_seq_gen<TBoxPlusOp, ROW_DEGREE, UPDATE_ROW_DEGREE> box_plus_seq_gen_t;
        
        //box_plus_seq_gen_t::generate(bp_update_seq, app);
        //#pragma unroll
        //for(int i = 0; i < UPDATE_ROW_DEGREE; ++i)
        //{
        //    // Apply normalization and store for next iteration
        //    row_storage.v.w[i].f16x2 = __hmul2(bp_update_seq[i].f16x2, norm);
        //    // Update APP
        //    app[i].f16x2 = __hadd2(row_storage.v.w[i].f16x2, app[i].f16x2);
        //}
    }
};

} // namespace ldpc2

#endif // !defined(LDPC2_C2V_X2_CUH_INCLUDED_)
