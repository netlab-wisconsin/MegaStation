/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "ldpc_ms_cta_layered.hpp"
#include "ldpc.cuh"
#include "ldpc_c2v_message.cuh"

namespace
{
template <typename TLLR>
class workspace_ms_cta_layered : public LDPC_workspace<workspace_ms_cta_layered<TLLR>> {
public:
    typedef TLLR                                           LLR_t;
    typedef c2v_message_t<TLLR>                            message_t;
    typedef LDPC_workspace<workspace_ms_cta_layered<TLLR>> inherited_t;
    //------------------------------------------------------------------
    // Constructor
    //__device__
    //workspace_ms_cta_layered(void* pv) : inherited_t(pv) {}
    //------------------------------------------------------------------
    // For each codeword:
    //     C2V: (mb * Z) instances of c2v_message_t
    //     APP: ((Kb + 4) * Z) * sizeof(TLLR)
    CUDA_BOTH_INLINE
    static size_t workspace_bytes_per_codeword(const LDPC_config& config)
    {
        unsigned int sC2V  = (config.mb * config.Z * sizeof(message_t));
        unsigned int szAPP = ((config.Kb + 4) * config.Z) * sizeof(TLLR);
        // Make sure that the data for each codeword is aligned to at
        // least 8 bytes
        return ((szAPP + sC2V + 7) / 8) * 8;
    }
    //__device__
    //message_t* C2V(const LDPC_config& config, int codewordIndex)
    //{
    //    return inherited_t::template offset_as<message_t>(codeword_base_offset(config, codewordIndex));
    //}
    //__device__
    //LLR_t*     APP(const LDPC_config& config, int codewordIndex)
    //{
    //    return inherited_t::template offset_as<LLR_t>(codeword_base_offset(codewordIndex) + app_offset(config));
    //}
private:
    //__device__
    //size_t codeword_base_offset(const LDPC_config& config, int codewordIndex)
    //{
    //    return (bytes_per_codeword(config) * codewordIndex);
    //}
    //__device__
    //size_t app_offset(const LDPC_config& config) const
    //{
    //    // APP buffer is after the C2V buffer in each codeword segment
    //    return (config.mb * config.Z * sizeof(message_t));
    //}
};

} // namespace

namespace ldpc
{
////////////////////////////////////////////////////////////////////////
// decode_ms_cta_layered()
cuphyStatus_t decode_ms_cta_layered(LDPC_output_t&      tDst,
                                    const_tensor_pair&  tLLR,
                                    const LDPC_config&  config,
                                    int                 maxNumIterations,
                                    float               normalization,
                                    bool                earlyTermination,
                                    cuphyLDPCResults_t* results,
                                    void*               workspace,
                                    cudaStream_t        strm)
{
    return CUPHY_STATUS_NOT_SUPPORTED;
}

//----------------------------------------------------------------------
// decode_multi_kernel_atomic_workspace_size()
std::pair<bool, size_t> decode_ms_cta_layered_workspace_size(const LDPC_config& cfg)
{
    switch(cfg.type)
    {
    case CUPHY_R_32F:
    {
        typedef data_type_traits<CUPHY_R_32F>::type LLR_t;
        typedef workspace_ms_cta_layered<LLR_t>     workspace_t;
        return std::pair<bool, size_t>(true,
                                       workspace_t::get_workspace_size(cfg));
    }
    case CUPHY_R_16F:
    {
        typedef data_type_traits<CUPHY_R_16F>::type LLR_t;
        typedef workspace_ms_cta_layered<LLR_t>     workspace_t;
        return std::pair<bool, size_t>(true,
                                       workspace_t::get_workspace_size(cfg));
    }
    default:
        return std::pair<bool, size_t>(false, 0);
    }
}

} // namespace ldpc
