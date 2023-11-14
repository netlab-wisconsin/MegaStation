/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_APP_ADDRESS_FP_CUH_INCLUDED_)
#define LDPC2_APP_ADDRESS_FP_CUH_INCLUDED_

#include <limits>

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// get_high()
__device__ inline
word_t get_high(word_t win)
{
    word_t out;
    LDPC2_ASM("prmt.b32 %0, %1, 0, 0x5432;\n"
              : "=r"(out.u32)
              : "r"(win.u32));
    return out;
}

////////////////////////////////////////////////////////////////////////
// app_address_pair()
//
// The APP address (where address = index * sizeof(T)) is given by:
//
// addr_app = [(col_app * Z) + (shift + threadIdx)    ] * sizeof(T)    threadIdx < (Z-shift)
// addr_app = [(col_app * Z) + (shift + threadIdx - Z)] * sizeof(T)    threadIdx >= (Z-shift)
//
// Use fp16x2 instructions and denormalized values (to avoid the
// requirement for int/float conversions). (Note that fp16 has only
// 10 stored mantissa bits, the maximum Z value is 384, and
// sizeof(__half) = 2.)
// RA = (shift * sizeof(T)) + (threadIdx * sizeof(T))
// RB = (threadIdx * sizeof(T)) >= ((Z - shift) * sizeof(T)) ? 1.0 : 0.0
// RC = RA - RB * Z * sizeof(T)
// addr_app = RC + (col_app * Z * sizeof(T))
//
//
// Expected SASS (V100):
// 4 instructions to calculate 2 addresses
// 2 instructions to extract LO, HI values
// Example:
// (Setup instructions, used by all addresses)
// S2R R40, SR_TID.X ;
// IMAD.SHL.U32 R145, R40, 0x2, RZ ;
//        
// HADD2 R41, R145.reuse.H0_H0, 4.398822784423828125e-05, 5.9604644775390625e-06 ;
// HSET2.BF.GE.AND R42, R145.H0_H0, 1.78813934326171875e-06, 3.98159027099609375e-05, PT ;
// HFMA2 R41, R42, -4.57763671875e-05, -4.57763671875e-05, R41 ;
// IADD3 R41, R41, 0x9000600, RZ ;
//
// (Extract high and low)
// PRMT R79, R41.reuse, 0x5410, RZ ;
// PRMT R78, R41, 0x5432, RZ ;
//
// (Load values)
// LDS.U.U16 R11, [R79] ;
// LDS.U.U16 R12, [R78] ;

template <typename T, int BG, int Z, int CHECK_INDEX, int PAIR_INDEX>
inline __device__
void app_address_pair(int (&app_addr)[row_degree<BG, CHECK_INDEX>::value])
{
    //static_assert(sizeof(T) == sizeof(__half), "half assumed below");
    //------------------------------------------------------------------
    // Create a "wrap index" pair with values that indicate whether we
    // need to subtract due to wraparound.
    word_t wrapIndexPair;
    wrapIndexPair.u32 = wrap_index_pair<BG, Z, CHECK_INDEX, PAIR_INDEX>::value * sizeof(T);
    //------------------------------------------------------------------
    // Create a pair of values, where each represents the value of
    // (col_index[i] * Z) + shift_mod[i]. We will add the thread index
    // to this value, and subtract a multiple of Z to handle wraparound
    // for the appropriate threads.
    word_t vnodeShiftPair;
    vnodeShiftPair.u32 = vnode_shift_mod_pair<BG, Z, CHECK_INDEX, PAIR_INDEX>::value * sizeof(T);
    //------------------------------------------------------------------
    // Calculate a pair of "base" variable node offsets (i.e. the index
    // of the first APP element for that variable node).
    word_t vnodeBaseOffsetPair;
    vnodeBaseOffsetPair.u32 = vnode_base_offset_pair<BG, Z, CHECK_INDEX, PAIR_INDEX>::value * sizeof(T);
    //------------------------------------------------------------------
    word_t wThreadIdx;
    wThreadIdx.u32 = threadIdx.x * sizeof(T);
    //------------------------------------------------------------------
    // Add the thread index to the variable node offset. (Use
    // denormalized floats to use the floating point unit.)
    vnodeShiftPair = hadd2(vnodeShiftPair, h0_h0(wThreadIdx));
    //------------------------------------------------------------------
    // Compare the (scaled) thread index to the per-node wrap index, to
    // determine if we need to subtract Z from the address for this
    // thread:
    // isWrap = [(threadIdx*2) < wrapIndexPair[1]   (threadIdx*2) < wrapIndexPair[0] ]
    word_t isWrap, blockOffset, wZ;
    isWrap = hset2_bf_ge(h0_h0(wThreadIdx), wrapIndexPair);
    //------------------------------------------------------------------
    // Use a fused multiply add with 1.0/0.0 values to optionally
    // subtract the Z value.
    wZ.u32 = Z * sizeof(T);
    wZ = hneg2(h0_h0(wZ));
    blockOffset = hfma2(isWrap, wZ, vnodeShiftPair);
    //------------------------------------------------------------------
    // Use regular unsigned addition for the two 16-bit values. Since
    // the maximum address will be (384*68*sizeof(half)) - 1 = 52,224
    // we won't have any overflow from the low word to the high word.
    // But this is only valid for __half addresses.
    word_t outAddress;
    outAddress.u32 = blockOffset.u32 + vnodeBaseOffsetPair.u32;
    //------------------------------------------------------------------
    // Write output values to the array argument
#if 0
    app_addr[(PAIR_INDEX * 2)]     = outAddress.u32 & 0xFFFF;    
    app_addr[(PAIR_INDEX * 2) + 1] = outAddress.u32 >> 16;
#else
    word_t addr0, addr1;
    addr0 = set_high_zero(outAddress);
    addr1 = get_high(outAddress);
    app_addr[(PAIR_INDEX * 2)]     = addr0.i32;
    if(((PAIR_INDEX * 2) + 1) < row_degree<BG, CHECK_INDEX>::value)
    {
        app_addr[(PAIR_INDEX * 2) + 1] = addr1.i32;
    }
#endif
#if 0
    if(253 == threadIdx.x)
    {
        float2 isWrapfp32 = __half22float2(isWrap.f16x2);
        printf("app_address_fp_unroll<%i>, threadIdx.x = %u, wrap_lo = %u, wrap_hi = %u, wThreadIdx = 0x%X, wZ = 0x%X, blockOffset = 0x%X, isWrap[0] = %f, isWrap[1] = %f, blockOffset[0] = %u, blockOffset[1] = %u, totalOffset[0] = %lu, totalOffset[1] = %lu, vnodeBaseOffset = 0x%X, outAddress[0] = %u, outAddress[1] = %u\n",
                PAIR_INDEX,
                threadIdx.x,
                wrap_index_if<BG, Z, CHECK_INDEX, (PAIR_INDEX * 2)>::value,
                wrap_index_if<BG, Z, CHECK_INDEX, (PAIR_INDEX * 2) + 1>::value,
                h0_h0(wThreadIdx).u32,
                wZ.u32,
                blockOffset.u32,
                isWrapfp32.x,
                isWrapfp32.y,
                blockOffset.f16x2.x,
                blockOffset.f16x2.y,
                (Z * vnode_index_if<BG, CHECK_INDEX, PAIR_INDEX*2>::value * sizeof(T))    + blockOffset.f16x2.x,
                (Z * vnode_index_if<BG, CHECK_INDEX, PAIR_INDEX*2 + 1>::value* sizeof(T)) + blockOffset.f16x2.y,
                vnodeBaseOffsetPair.u32,
                outAddress.f16x2.x,
                outAddress.f16x2.y);
    }
#endif
}

////////////////////////////////////////////////////////////////////////
// app_address_fp_unroll
template <typename T, int BG, int Z, int ILS, int CHECK_INDEX, int PAIR_COUNT> struct app_address_fp_unroll;
template <typename T, int BG, int Z, int ILS, int CHECK_INDEX>
struct app_address_fp_unroll<T, BG, Z, ILS, CHECK_INDEX, 1>
{
    __device__
    static void generate(int (&app_addr)[row_degree<BG, CHECK_INDEX>::value])
    {
        app_address_pair<T, BG, Z, CHECK_INDEX, 0>(app_addr);
    }
};

template <typename T, int BG, int Z, int ILS, int CHECK_INDEX, int PAIR_COUNT>
struct app_address_fp_unroll
{
    __device__
    static void generate(int (&app_addr)[row_degree<BG, CHECK_INDEX>::value])
    {
        app_address_fp_unroll<T, BG, Z, ILS, CHECK_INDEX, PAIR_COUNT - 1>::generate(app_addr);
        app_address_pair<T, BG, Z, CHECK_INDEX, PAIR_COUNT-1>(app_addr);
    }
};

////////////////////////////////////////////////////////////////////////
// app_address_fp_generator
template <typename T, int BG, int Z, int CHECK_INDEX> struct app_address_fp_generator
{
    __device__
    static void generate(int (&app_addr)[row_degree<BG, CHECK_INDEX>::value])
    {
        //--------------------------------------------------------------
        // Define a type for a generator with the number of elements in the row
        typedef app_address_fp_unroll<T,                    // APP type
                                      BG,                   // base graph
                                      Z,                    // lifting size
                                      set_index<Z>::value,  // base graph set index
                                      CHECK_INDEX,          // parity node row
                                      div_round_up_t<row_degree<BG, CHECK_INDEX>::value, 2>::value> generator_t;
        //--------------------------------------------------------------
        // Generate address values
        generator_t::generate(app_addr);
    }
};

////////////////////////////////////////////////////////////////////////
// app_loc_address_fp
// Manager for calculation and storage of APP locations (in shared memory)
// This implementation stores shared memory addresses (as opposed to
// storing INDICES into the APP array). The addresses are stored in
// registers (to avoid recalculation), so use will result in increased
// register pressure.
template <typename T, int BG, int Z_>
struct app_loc_address_fp
{
    static constexpr int Z = Z_;
    //------------------------------------------------------------------
    template <int CHECK_IDX>
    __device__
    void generate(int (&app_addr)[row_degree<BG, CHECK_IDX>::value])
    {
        app_address_fp_generator<T, BG, Z_, CHECK_IDX>::generate(app_addr);
    }
};

////////////////////////////////////////////////////////////////////////
// app_address_pair_imad()
// Similar to the app_address_pair() function above. Instead of
// incorporating the sizeof(T) value in the constants, this version
// multiplies by sizeof(T) in the last instruction. (This is
// advantageous for types T that are larger than __half, because the
// use of denormalized floats can only use 10 bits of mantissa.)
// The last instruction be IMAD in this formulation, as opposed to the
// use of IADD in app_address_pair()
//
// The APP address (where address = index * sizeof(T)) is given by:
//
// addr_app = [(col_app * Z) + (shift + threadIdx)    ] * sizeof(T)    threadIdx < (Z-shift)
// addr_app = [(col_app * Z) + (shift + threadIdx - Z)] * sizeof(T)    threadIdx >= (Z-shift)
//
// Use fp16x2 values and instructions and denormalized values (to avoid
// the requirement for int/float conversions). (Note that fp16 has only
// 10 stored mantissa bits, the maximum Z value is 384, and
// sizeof(__half) = 2.)
//
// Use fp16x2 instructions and denormalized values (to avoid the
// requirement for int/float conversions). (Note that fp16 has only
// 10 stored mantissa bits, the maximum Z value is 384, and
// sizeof(__half) = 2.)
// RA = shift + threadIdx
// RB = (threadIdx  >= (Z - shift)) ? 1.0 : 0.0
// RC = RA - RB * Z
// addr_app = (RC * sizeof(T)) + (col_app * Z * sizeof(T))
//
// NOTE: If (col_app * Z * sizeof(T)) is too large for 16 bits, we
// cannot use this approach.
//
template <typename T, int BG, int Z, int CHECK_INDEX, int PAIR_INDEX>
inline __device__
void app_address_pair_imad(int (&app_addr)[row_degree<BG, CHECK_INDEX>::value])
{
    //static_assert(sizeof(T) == sizeof(__half), "half assumed below");
    //------------------------------------------------------------------
    // Create a "wrap index" pair with values that indicate whether we
    // need to subtract due to wraparound.
    word_t wrapIndexPair;
    wrapIndexPair.u32 = wrap_index_pair<BG, Z, CHECK_INDEX, PAIR_INDEX>::value;
    //------------------------------------------------------------------
    // Create a pair of values, where each represents the value of
    // (col_index[i] * Z) + shift_mod[i]. We will add the thread index
    // to this value, and subtract a multiple of Z to handle wraparound
    // for the appropriate threads.
    word_t vnodeShiftPair;
    vnodeShiftPair.u32 = vnode_shift_mod_pair<BG, Z, CHECK_INDEX, PAIR_INDEX>::value;
    //------------------------------------------------------------------
    // Calculate a pair of "base" variable node offsets (i.e. the index
    // of the first APP element for that variable node). Note that for
    // some types and Z values, the offset is too large to be stored
    // with 16 bits.
    static_assert((vnode_base_offset_if<BG, Z, CHECK_INDEX, (PAIR_INDEX*2)  >::value * sizeof(T)) <= USHRT_MAX,
                  "Base offset (low) cannot be represented with 16 bits");
    static_assert((vnode_base_offset_if<BG, Z, CHECK_INDEX, (PAIR_INDEX*2)+1>::value * sizeof(T)) <= USHRT_MAX,
                  "Base offset (high) cannot be represented with 16 bits");
    word_t vnodeBaseOffsetPair;
    vnodeBaseOffsetPair.u32 = vnode_base_offset_pair<BG, Z, CHECK_INDEX, PAIR_INDEX>::value * sizeof(T);
    //------------------------------------------------------------------
    word_t wThreadIdx;
    wThreadIdx.u32 = threadIdx.x;
    //------------------------------------------------------------------
    // Add the thread index to the variable node offset. (Use
    // denormalized floats to use the floating point unit.)
    vnodeShiftPair = hadd2(vnodeShiftPair, h0_h0(wThreadIdx));
    //------------------------------------------------------------------
    // Compare the (scaled) thread index to the per-node wrap index, to
    // determine if we need to subtract Z from the address for this
    // thread:
    // isWrap = [ threadIdx < wrapIndexPair[1]   threadIdx < wrapIndexPair[0] ]
    word_t isWrap, blockOffset, wZ;
    isWrap = hset2_bf_ge(h0_h0(wThreadIdx), wrapIndexPair);
    //------------------------------------------------------------------
    // Use a fused multiply add with 1.0/0.0 values to optionally
    // subtract the Z value.
    wZ.u32 = Z;
    wZ = hneg2(h0_h0(wZ));
    blockOffset = hfma2(isWrap, wZ, vnodeShiftPair);
    //------------------------------------------------------------------
    // Use regular unsigned mul-add for the two 16-bit values. For some
    // values of sizeof(T) and Z, the value may exceed 16 bits, and this
    // function/approach should not be used!
    word_t outAddress;
    outAddress.u32 = (blockOffset.u32 * sizeof(T)) + vnodeBaseOffsetPair.u32;
    //------------------------------------------------------------------
    // Write output values to the array argument
#if 0
    app_addr[(PAIR_INDEX * 2)]     = outAddress.u32 & 0xFFFF;    
    app_addr[(PAIR_INDEX * 2) + 1] = outAddress.u32 >> 16;
#else
    word_t addr0, addr1;
    addr0 = set_high_zero(outAddress);
    addr1 = get_high(outAddress);
    app_addr[(PAIR_INDEX * 2)]     = addr0.i32;
    if(((PAIR_INDEX * 2) + 1) < row_degree<BG, CHECK_INDEX>::value)
    {
        app_addr[(PAIR_INDEX * 2) + 1] = addr1.i32;
    }
#endif
}

////////////////////////////////////////////////////////////////////////
// app_address_fp_imad_unroll
template <typename T, int BG, int Z, int ILS, int CHECK_INDEX, int PAIR_COUNT> struct app_address_fp_imad_unroll;
template <typename T, int BG, int Z, int ILS, int CHECK_INDEX>
struct app_address_fp_imad_unroll<T, BG, Z, ILS, CHECK_INDEX, 1>
{
    __device__
    static void generate(int (&app_addr)[row_degree<BG, CHECK_INDEX>::value])
    {
        app_address_pair_imad<T, BG, Z, CHECK_INDEX, 0>(app_addr);
    }
};

template <typename T, int BG, int Z, int ILS, int CHECK_INDEX, int PAIR_COUNT>
struct app_address_fp_imad_unroll
{
    __device__
    static void generate(int (&app_addr)[row_degree<BG, CHECK_INDEX>::value])
    {
        app_address_fp_imad_unroll<T, BG, Z, ILS, CHECK_INDEX, PAIR_COUNT - 1>::generate(app_addr);
        app_address_pair_imad<T, BG, Z, CHECK_INDEX, PAIR_COUNT-1>(app_addr);
    }
};

////////////////////////////////////////////////////////////////////////
// app_address_fp_imad_generator
template <typename T, int BG, int Z, int CHECK_INDEX> struct app_address_fp_imad_generator
{
    __device__
    static void generate(int (&app_addr)[row_degree<BG, CHECK_INDEX>::value])
    {
        //--------------------------------------------------------------
        // Define a type for a generator with the number of elements in the row
        typedef app_address_fp_imad_unroll<T,                    // APP type
                                           BG,                   // base graph
                                           Z,                    // lifting size
                                           set_index<Z>::value,  // base graph set index
                                           CHECK_INDEX,          // parity node row
                                           div_round_up_t<row_degree<BG, CHECK_INDEX>::value, 2>::value> generator_t;
        //--------------------------------------------------------------
        // Generate address values
        generator_t::generate(app_addr);
    }
};

////////////////////////////////////////////////////////////////////////
// app_loc_address_fp_imad
// Manager for calculation and storage of APP locations (in shared memory)
// This implementation stores shared memory addresses (as opposed to
// storing INDICES into the APP array). The addresses are stored in
// registers (to avoid recalculation), so use will result in increased
// register pressure.
template <typename T, int BG, int Z_>
struct app_loc_address_fp_imad
{
    static constexpr int Z = Z_;
    //------------------------------------------------------------------
    template <int CHECK_IDX>
    __device__
    void generate(int (&app_addr)[row_degree<BG, CHECK_IDX>::value])
    {
        app_address_fp_imad_generator<T, BG, Z_, CHECK_IDX>::generate(app_addr);
    }
};

////////////////////////////////////////////////////////////////////////
// app_address_pair_imad_lg()
// Similar to the app_address_pair_imad() function above. Instead of
// incorporating the sizeof(T) value in the constants, this version
// multiplies by sizeof(T) after extracting the offset values. This is
// necessary when sizeof(T) is large (.e.g __half2 or float) and/or the
// number of parity nodes is large because the output address may not
// be represented
// use of denormalized floats can only use 10 bits of mantissa.)
// The last instruction be IMAD in this formulation, as opposed to the
// use of IADD in app_address_pair()
//
// The APP address (where address = index * sizeof(T)) is given by:
//
// addr_app = [(col_app * Z) + (shift + threadIdx)    ] * sizeof(T)    threadIdx < (Z-shift)
// addr_app = [(col_app * Z) + (shift + threadIdx - Z)] * sizeof(T)    threadIdx >= (Z-shift)
//
// Use fp16x2 values and instructions and denormalized values (to avoid
// the requirement for int/float conversions). (Note that fp16 has only
// 10 stored mantissa bits, the maximum Z value is 384, and
// sizeof(__half) = 2.)
//
// Use fp16x2 instructions and denormalized values (to avoid the
// requirement for int/float conversions). (Note that fp16 has only
// 10 stored mantissa bits, the maximum Z value is 384, and
// sizeof(__half) = 2.)
// RA = shift + threadIdx
// RB = (threadIdx  >= (Z - shift)) ? 1.0 : 0.0
// RC = RA - RB * Z
// addr_app = (RC * sizeof(T)) + (col_app * Z * sizeof(T))
//
// NOTE: If (col_app * Z * sizeof(T)) is too large for 16 bits, we
// cannot use this approach.
//
template <typename T, int BG, int Z, int CHECK_INDEX, int PAIR_INDEX>
inline __device__
void app_address_pair_imad_lg(int (&app_addr)[row_degree<BG, CHECK_INDEX>::value])
{
    //------------------------------------------------------------------
    // Create a "wrap index" pair with values that indicate whether we
    // need to subtract due to wraparound.
    word_t wrapIndexPair;
    wrapIndexPair.u32 = wrap_index_pair<BG, Z, CHECK_INDEX, PAIR_INDEX>::value;
    //------------------------------------------------------------------
    // Create a pair of values, where each represents the value of
    // (col_index[i] * Z) + shift_mod[i]. We will add the thread index
    // to this value, and subtract a multiple of Z to handle wraparound
    // for the appropriate threads.
    word_t vnodeShiftPair;
    vnodeShiftPair.u32 = vnode_shift_mod_pair<BG, Z, CHECK_INDEX, PAIR_INDEX>::value;
    //------------------------------------------------------------------
    word_t wThreadIdx;
    wThreadIdx.u32 = threadIdx.x;
    //------------------------------------------------------------------
    // Compare the (scaled) thread index to the per-node wrap index, to
    // determine if we need to subtract Z from the address for this
    // thread:
    // isWrap = [ threadIdx < wrapIndexPair[1]   threadIdx < wrapIndexPair[0] ]
    word_t isWrap, blockOffset, wZ;
    isWrap = hset2_bf_ge(h0_h0(wThreadIdx), wrapIndexPair);
    //------------------------------------------------------------------
    // Add the thread index to the variable node offset. (Use
    // denormalized floats to use the floating point unit.)
    vnodeShiftPair = hadd2(vnodeShiftPair, h0_h0(wThreadIdx));
    //------------------------------------------------------------------
    // Use a fused multiply add with 1.0/0.0 values to optionally
    // subtract the Z value.
    wZ.u32 = Z;
    wZ = hneg2(h0_h0(wZ));
    blockOffset = hfma2(isWrap, wZ, vnodeShiftPair);
    //------------------------------------------------------------------
    word_t vnodeBaseOffsetPair;
    word_t outAddress;
    if(max_pair_app_address<T, BG, Z, CHECK_INDEX, PAIR_INDEX>::value <= USHRT_MAX)
    {
        // Calculate a pair of "base" variable node offsets (i.e. the index
        // of the first APP element for that variable node). Note that for
        // some types and Z values, the offset is too large to be stored
        // with 16 bits.
        //static_assert((vnode_base_offset_if<BG, Z, CHECK_INDEX, (PAIR_INDEX*2)  >::value * sizeof(T)) <= USHRT_MAX,
        //              "Base offset (low) cannot be represented with 16 bits");
        //static_assert((vnode_base_offset_if<BG, Z, CHECK_INDEX, (PAIR_INDEX*2)+1>::value * sizeof(T)) <= USHRT_MAX,
        //              "Base offset (high) cannot be represented with 16 bits");
        // Use static_cast here to avoid warnings for:
        // integer conversion resulted in truncation
        // Conditional above should (!) avoid execution when there would
        // actually be truncation.
        vnodeBaseOffsetPair.u32 = static_cast<uint32_t>(vnode_base_offset_pair<BG, Z, CHECK_INDEX, PAIR_INDEX>::value * sizeof(T));
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - - 
        // Use regular unsigned mul-add for the two 16-bit values. For some
        // values of sizeof(T) and Z, the value may exceed 16 bits, and this
        // function/approach should not be used!
        outAddress.u32 = (blockOffset.u32 * sizeof(T)) + vnodeBaseOffsetPair.u32;
#if 0
        app_addr[(PAIR_INDEX * 2)]     = outAddress.u32 & 0xFFFF;    
        app_addr[(PAIR_INDEX * 2) + 1] = outAddress.u32 >> 16;
#else
        word_t addr0, addr1;
        addr0 = set_high_zero(outAddress);
        addr1 = get_high(outAddress);
        app_addr[(PAIR_INDEX * 2)]     = addr0.u32;
        if(((PAIR_INDEX * 2) + 1) < row_degree<BG, CHECK_INDEX>::value)
        {
            app_addr[(PAIR_INDEX * 2) + 1] = addr1.u32;
        }
#endif
    }
    else
    {
        //------------------------------------------------------------------
        // Calculate a pair of "base" variable node offsets (i.e. the index
        // of the first APP element for that variable node). Note that for
        // some types and Z values, the offset is too large to be stored
        // with 16 bits.
        vnodeBaseOffsetPair.u32 = vnode_base_offset_pair<BG, Z, CHECK_INDEX, PAIR_INDEX>::value;
        //------------------------------------------------------------------
        // Use regular unsigned add for the two 16-bit values.
        outAddress.u32 = blockOffset.u32 + vnodeBaseOffsetPair.u32;
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Write output values to the array argument
#if 0
        app_addr[(PAIR_INDEX * 2)]     = outAddress.u32 & 0xFFFF;    
        app_addr[(PAIR_INDEX * 2) + 1] = outAddress.u32 >> 16;
#else
        word_t addr0, addr1;
        addr0 = set_high_zero(outAddress);
        addr1 = get_high(outAddress);
        app_addr[(PAIR_INDEX * 2)]     = static_cast<int>(addr0.u32 * sizeof(T));
        if(((PAIR_INDEX * 2) + 1) < row_degree<BG, CHECK_INDEX>::value)
        {
            app_addr[(PAIR_INDEX * 2) + 1] = static_cast<int>(addr1.u32 * sizeof(T));
        }
#endif
    }
    //------------------------------------------------------------------
}

////////////////////////////////////////////////////////////////////////
// app_address_fp_imad_lg_unroll
template <typename T, int BG, int Z, int ILS, int CHECK_INDEX, int PAIR_COUNT> struct app_address_fp_imad_lg_unroll;
template <typename T, int BG, int Z, int ILS, int CHECK_INDEX>
struct app_address_fp_imad_lg_unroll<T, BG, Z, ILS, CHECK_INDEX, 1>
{
    __device__
    static void generate(int (&app_addr)[row_degree<BG, CHECK_INDEX>::value])
    {
        app_address_pair_imad_lg<T, BG, Z, CHECK_INDEX, 0>(app_addr);
    }
};

template <typename T, int BG, int Z, int ILS, int CHECK_INDEX, int PAIR_COUNT>
struct app_address_fp_imad_lg_unroll
{
    __device__
    static void generate(int (&app_addr)[row_degree<BG, CHECK_INDEX>::value])
    {
        app_address_fp_imad_lg_unroll<T, BG, Z, ILS, CHECK_INDEX, PAIR_COUNT - 1>::generate(app_addr);
        app_address_pair_imad_lg<T, BG, Z, CHECK_INDEX, PAIR_COUNT-1>(app_addr);
    }
};

////////////////////////////////////////////////////////////////////////
// app_address_fp_imad_lg_generator
template <typename T, int BG, int Z, int CHECK_INDEX> struct app_address_fp_imad_lg_generator
{
    __device__
    static void generate(int (&app_addr)[row_degree<BG, CHECK_INDEX>::value])
    {
        //--------------------------------------------------------------
        // Define a type for a generator with the number of elements in the row
        typedef app_address_fp_imad_lg_unroll<T,                    // APP type
                                              BG,                   // base graph
                                              Z,                    // lifting size
                                              set_index<Z>::value,  // base graph set index
                                              CHECK_INDEX,          // parity node row
                                              div_round_up_t<row_degree<BG, CHECK_INDEX>::value, 2>::value> generator_t;
        //--------------------------------------------------------------
        // Generate address values
        generator_t::generate(app_addr);
    }
};

////////////////////////////////////////////////////////////////////////
// app_loc_address_fp_imad_lg
// Manager for calculation and storage of APP locations (in shared memory)
// This implementation stores shared memory addresses (as opposed to
// storing INDICES into the APP array). The addresses are stored in
// registers (to avoid recalculation), so use will result in increased
// register pressure.
template <typename T, int BG, int Z_>
struct app_loc_address_fp_imad_lg
{
    static constexpr int Z = Z_;
    //------------------------------------------------------------------
    template <int CHECK_IDX>
    __device__
    void generate(int (&app_addr)[row_degree<BG, CHECK_IDX>::value])
    {
        app_address_fp_imad_lg_generator<T, BG, Z_, CHECK_IDX>::generate(app_addr);
    }
};

} // namespace ldpc2

#endif // !defined(LDPC2_APP_ADDRESS_FP_CUH_INCLUDED_)
