/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC_C2V_MESSAGE_CUH_INCLUDED_)
#define LDPC_C2V_MESSAGE_CUH_INCLUDED_

#include "nrLDPC.cuh"

template <typename LLR_t>
struct c2v_message_t
{
    LLR_t    min0;
    LLR_t    min1;
    uint32_t sign_index; // Low 19 bits are sign bits (a set bit indicates a negative number)
    // High 13 bits are the index, within the row values, of min0. This number
    // must be between 0 and (ROW_DEGREE - 1). (max row degree is 19, so value < 19)
    /*CUDA_BOTH_INLINE*/ c2v_message_t() = default;
    CUDA_BOTH_INLINE c2v_message_t(LLR_t v0, LLR_t v1) :
        sign_index(0)
    {
        LLR_t    abs0  = llr_abs(v0);
        LLR_t    abs1  = llr_abs(v1);
        uint32_t signs = is_neg(v0) ? 1 : 0;
        if(is_neg(v1)) signs |= 2;
        set_signs(signs);
        if(abs0 < abs1)
        {
            min0 = abs0;
            min1 = abs1;
            set_row_index(0);
        }
        else
        {
            min0 = abs1;
            min0 = abs0;
            set_row_index(1);
        }
    }
    CUDA_BOTH_INLINE c2v_message_t(LLR_t m0, LLR_t m1, uint32_t sign, uint32_t index) :
        min0(m0),
        min1(m1),
        sign_index(sign | (index << 19)) {}
    CUDA_BOTH_INLINE c2v_message_t(LLR_t m0, LLR_t m1, uint32_t signIndex) :
        min0(m0),
        min1(m1),
        sign_index(signIndex) {}
    CUDA_BOTH_INLINE void set_signs(uint32_t s) { sign_index = (sign_index | (s & 0x7FFFF)); };
    CUDA_BOTH_INLINE uint32_t get_signs() const { return (sign_index & 0x7FFFF); }
    CUDA_BOTH_INLINE void     set_row_index(int idx) { sign_index = (sign_index | (idx << 19)); }
    CUDA_BOTH_INLINE uint32_t get_row_index() const { return (sign_index >> 19); }
    CUDA_INLINE LLR_t get_value_for_index(int rowIndex, float norm) const
    {
        uint32_t minRowIndex = get_row_index();
        uint32_t signs       = get_signs();
        LLR_t    minAbsLvc   = (rowIndex == minRowIndex) ? min1 : min0;
        LLR_t    signProd    = (0 != (__popc(signs & ~(1 << rowIndex)) & 1)) ? -1.0f : 1.0f; // TODO: get approprate constants for type
        //KERNEL_PRINT("col_index = %u, row_index = %u, signs = 0x%X, minAbsLvc = %.4f, signProd = %.4f, returning %.4f\n",
        //             minColIndex, minRowIndex, signs, to_float(minAbsLvc), to_float(signProd), to_float(type_convert<LLR_t>(norm) * minAbsLvc * signProd));
        return type_convert<LLR_t>(norm) * minAbsLvc * signProd;
    }
    CUDA_INLINE void process(LLR_t value, int row_index)
    {
        sign_index |= (is_neg(value) ? 1 : 0) << row_index;
        LLR_t Lvcabs = llr_abs(value);
        if(Lvcabs < min0)
        {
            set_row_index(row_index);
            min1 = min0;
            min0 = Lvcabs;
        }
        else if(Lvcabs < min1)
        {
            min1 = Lvcabs;
        }
    }
    CUDA_BOTH_INLINE void init()
    {
        min0 = min1 = 10000;
        sign_index  = 0;
    }
    static CUDA_INLINE int get_variable_index(const bg1_CN_row_shift_info_t& shiftInfo,  // shift data
                                              int                            iVN,        // index within row
                                              int                            nodeOffset, // offset of check variable within node
                                              int                            Z)                                     // lifting factor
    {
        const int8_t POS          = shiftInfo.column_values[iVN];
        int          block_offset = nodeOffset + shiftInfo.shift_values[iVN];
        if(block_offset >= Z) block_offset -= Z;
        return (POS * Z) + block_offset;
    }
    static CUDA_INLINE c2v_message_t create_message(const bg1_CN_row_shift_info_t& shiftInfo,  // shift data
                                                    int                            nodeOffset, // offset of check variable within node
                                                    int                            Z,          // lifting factor
                                                    const LLR_t*                   initLLR)                      // initial LLR data
    {
        uint32_t signBits = 0;
        int      minIndex;
        LLR_t    min0, min1;
        // The minimum row degree in BG1 is 3, so we Unroll the first 3
        int   VN0     = get_variable_index(shiftInfo, 0, nodeOffset, Z);
        int   VN1     = get_variable_index(shiftInfo, 1, nodeOffset, Z);
        int   VN2     = get_variable_index(shiftInfo, 2, nodeOffset, Z);
        LLR_t Lvc0    = initLLR[VN0];
        LLR_t Lvc1    = initLLR[VN1];
        LLR_t Lvc2    = initLLR[VN2];
        LLR_t LvcAbs0 = llr_abs(Lvc0);
        LLR_t LvcAbs1 = llr_abs(Lvc1);
        LLR_t LvcAbs2 = llr_abs(Lvc2);
        signBits      = (is_neg(Lvc0) ? 1 : 0);
        signBits |= (is_neg(Lvc1) ? 2 : 0);
        if(LvcAbs0 < LvcAbs1)
        {
            minIndex = 0;
            min0     = LvcAbs0;
            min1     = LvcAbs1;
        }
        else
        {
            minIndex = 1;
            min0     = LvcAbs1;
            min1     = LvcAbs0;
        }
        signBits |= (is_neg(Lvc2) ? 4 : 0);
        if(LvcAbs2 < min0)
        {
            minIndex = 2;
            min1     = min0;
            min0     = LvcAbs2;
        }
        else if(LvcAbs2 < min1)
        {
            min1 = LvcAbs2;
        }
        for(int iVN = BG1_MIN_ROW_DEG; iVN < shiftInfo.row_degree; ++iVN)
        {
            const int VN_idx = get_variable_index(shiftInfo, iVN, nodeOffset, Z);
            LLR_t     Lvc    = initLLR[VN_idx];
            signBits |= (is_neg(Lvc) ? 1 : 0) << iVN;
            LLR_t Lvcabs = llr_abs(Lvc);
            if(Lvcabs < min0)
            {
                minIndex = iVN;
                min1     = min0;
                min0     = Lvcabs;
            }
            else if(Lvcabs < min1)
            {
                min1 = Lvcabs;
            }
            //KERNEL_PRINT_IF(CHECK_IDX == 6, "CHECK_IDX = %i, iVN = %i, minIndex = %i, minPos = %i, min0 = %.4f, min1 = %.4f, Lvc = %.4f\n",
            //                CHECK_IDX, iVN, minIndex, minPos, to_float(min0), to_float(min1), to_float(Lvc));
        }
        return c2v_message_t(min0, min1, signBits, minIndex);
    }
};

template <typename LLR_t>
class c2v_message_reader {
public:
    typedef c2v_message_t<LLR_t> c2v_message;
    CUDA_INLINE
    c2v_message_reader(const c2v_message& msg, LLR_t norm) :
        norm_min0(norm * msg.min0),
        norm_min1(norm * msg.min1),
        sign_bits(msg.get_signs()),
        min_index(msg.get_row_index()),
        sign_prod(ldpc::device_popc(sign_bits) & 0x1) // 0 if count is even, 1 if odd
    {
    }
    CUDA_INLINE LLR_t get_value_for_index_and_advance(int rowIndex)
    {
        LLR_t result = (rowIndex == min_index) ? norm_min1 : norm_min0;
        // The least significant bit of sign_bits represents the sign
        // of the "current" value. sign_prod represents product of the
        // sign of all values. If we remove the "current" value, what
        // is the product of the remaining signs?
        //               product of all
        //  current        sign values
        // value sign     0       1
        //     0          0       1
        //     1          1       0
        // This is an XOR operation.
        if(0 != ((sign_bits & 0x1) ^ sign_prod))
        {
            result = negate(result);
        }
        //KERNEL_PRINT("rowIndex = %u, sign_bits = 0x%X, sign_prod = %u, returning %.4f\n",
        //             rowIndex, sign_bits, sign_prod, to_float(result));
        // Prepare for the next iteration
        sign_bits >>= 1;

        return result;
    }

private:
    const LLR_t norm_min0;
    const LLR_t norm_min1;
    uint32_t    sign_bits; // Low 19 bits are valid
    uint32_t    min_index; // Low 13 bits are valid
    uint32_t    sign_prod;
};

template <cuphyDataType_t TType>
using c2v_message_type_t = c2v_message_t<typename data_type_traits<TType>::type>;

template <typename LLR_t>
CUDA_INLINE void load_c2v_message(c2v_message_t<LLR_t>* dst, const c2v_message_t<LLR_t>* src)
{
    *dst = *src;
}

template <>
CUDA_INLINE void load_c2v_message(c2v_message_t<__half>* dst, const c2v_message_t<__half>* src)
{
    *(reinterpret_cast<uint2*>(dst)) = *(reinterpret_cast<const uint2*>(src));
}

#endif // !defined(LDPC_C2V_MESSAGE_CUH_INCLUDED_)
