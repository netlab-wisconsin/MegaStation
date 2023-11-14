/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC_MS_CTA_SHMEM_FLOODING_HPP_INCLUDED_)
#define LDPC_MS_CTA_SHMEM_FLOODING_HPP_INCLUDED_

// Min-sum, Single Cooperative Thread Array, Shared Memory, Flooding LDPC Implementation

#include "ldpc.hpp"

////////////////////////////////////////////////////////////////////////
// ldpc
namespace ldpc
{
////////////////////////////////////////////////////////////////////////
// decode_ms_cta_shmem_flooding()
cuphyStatus_t decode_ms_cta_shmem_flooding(LDPC_output_t&         tDst,
                                           const_tensor_pair&     tLLR,
                                           const LDPC_config&     config,
                                           float                  normalization,
                                           cuphyLDPCResults_t*    results,
                                           void*                  workspace,
                                           cuphyLDPCDiagnostic_t* diag,
                                           cudaStream_t           strm);

////////////////////////////////////////////////////////////////////////
// decode_ms_cta_simd_flooding_workspace_size()
std::pair<bool, size_t> decode_ms_cta_shmem_flooding_workspace_size(const LDPC_config& cfg);

} // namespace ldpc

#endif // !defined(LDPC_MS_CTA_LAYERED_HPP_INCLUDED_)
