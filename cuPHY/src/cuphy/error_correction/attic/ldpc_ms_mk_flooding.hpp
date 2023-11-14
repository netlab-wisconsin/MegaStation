/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC_MS_MK_FLOODING_HPP_INCLUDED_)
#define LDPC_MS_MK_FLOODING_HPP_INCLUDED_

// Min-sum, Multi-kernel, Flooding LDPC Implementation

#include "ldpc.hpp"

////////////////////////////////////////////////////////////////////////
// ldpc
namespace ldpc
{
////////////////////////////////////////////////////////////////////////
// decode_multi_kernel()
cuphyStatus_t decode_multi_kernel(LDPC_output_t&      tDst,
                                  const_tensor_pair&  tLLR,
                                  const LDPC_config&  config,
                                  float               normalization,
                                  cuphyLDPCResults_t* results,
                                  void*               workspace,
                                  cudaStream_t        strm);

////////////////////////////////////////////////////////////////////////
// decode_multi_kernel_workspace_size()
std::pair<bool, size_t> decode_multi_kernel_workspace_size(const LDPC_config& config);

} // namespace ldpc

#endif // !defined(LDPC_MS_MK_FLOODING_HPP_INCLUDED_)
