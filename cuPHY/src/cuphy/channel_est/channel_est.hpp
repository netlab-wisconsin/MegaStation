/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(CHANNEL_EST_HPP_INCLUDED_)
#define CHANNEL_EST_HPP_INCLUDED_

#include "tensor_desc.hpp"

////////////////////////////////////////////////////////////////////////
// channel_est
namespace channel_est
{
//----------------------------------------------------------------------
// mmse_1D_time_frequency()
void mmse_1D_time_frequency(tensor_pair&       tDst,
                            const_tensor_pair& tSymbols,
                            const_tensor_pair& tFreqFilters,
                            const_tensor_pair& tTimeFilters,
                            const_tensor_pair& tFreqIndices,
                            const_tensor_pair& tTimeIndices,
                            cudaStream_t       strm);

} // namespace channel_est

#endif // !defined(CHANNEL_EST_HPP_INCLUDED_)
