/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_REG_INDEX_FP_DESC_DYN_ROW_DEP_HPP_INCLUDED_)
#define LDPC2_REG_INDEX_FP_DESC_DYN_ROW_DEP_HPP_INCLUDED_

#include "ldpc.hpp"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// reg_index_fp_desc_dyn_row_dep
class reg_index_fp_desc_dyn_row_dep : public ldpc::decode_algo
{
public:
    //------------------------------------------------------------------
    // Constructor
    reg_index_fp_desc_dyn_row_dep(ldpc::decoder& desc);
    //------------------------------------------------------------------
    // decode()
    virtual cuphyStatus_t decode(ldpc::decoder&                     dec,
                                 LDPC_output_t&                     tDst,
                                 const_tensor_pair&                 tLLR,
                                 const cuphyLDPCDecodeConfigDesc_t& config,
                                 cudaStream_t                       strm) override;
    //------------------------------------------------------------------
    // decode_tb()
    virtual cuphyStatus_t decode_tb(ldpc::decoder&               dec,
                                    const cuphyLDPCDecodeDesc_t& decodeDesc,
                                    cudaStream_t                 strm) override;
    //------------------------------------------------------------------
    // get_workspace_size()
    virtual std::pair<bool, size_t> get_workspace_size(const ldpc::decoder&               dec,
                                                       const cuphyLDPCDecodeConfigDesc_t& config,
                                                       int                                num_cw) override;
    //------------------------------------------------------------------
    // get_launch_config()
    virtual cuphyStatus_t get_launch_config(const ldpc::decoder&           dec,
                                            cuphyLDPCDecodeLaunchConfig_t& launchConfig) override;
};
    
} // namespace ldpc2

#endif // !defined(LDPC2_REG_INDEX_FP_DESC_DYN_ROW_DEP_HPP_INCLUDED_)
