/*
 * Copyright (c) 2020-2022, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_SPLIT_INDEX_FP_X2_DESC_DYN_SM86_HPP_INCLUDED_)
#define LDPC2_SPLIT_INDEX_FP_X2_DESC_DYN_SM86_HPP_INCLUDED_

#include "ldpc.hpp"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// split_index_fp_x2_desc_dyn_sm86
class split_index_fp_x2_desc_dyn_sm86 : public ldpc::decode_algo
{
public:
    //------------------------------------------------------------------
    // Constructor
    split_index_fp_x2_desc_dyn_sm86(ldpc::decoder& desc);
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
    // can_decode_config()
    virtual bool can_decode_config(const ldpc::decoder&               dec,
                                   const cuphyLDPCDecodeConfigDesc_t& config) override;
    //------------------------------------------------------------------
    // get_launch_config()
    virtual cuphyStatus_t get_launch_config(const ldpc::decoder&           dec,
                                            cuphyLDPCDecodeLaunchConfig_t& launchConfig) override;
private:
    // CUDA driver API module for SM80 .cubin binaries with internal-only
    // instructions.
    // TODO: Remove and use CUDA runtime API when instructions are made
    // public:
    cuphy_i::cu_module sm86_module_;
    CUfunction         sm86_BG1_kernel_;
    CUfunction         sm86_BG2_kernel_;
    CUfunction         sm86_BG1_tb_kernel_;
    CUfunction         sm86_BG2_tb_kernel_;
};

} // namespace ldpc2

#endif // !defined(LDPC2_SPLIT_INDEX_FP_X2_DESC_DYN_SM86_HPP_INCLUDED_)
