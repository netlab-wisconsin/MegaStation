/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(CUPHY_TENSOR_TILE_HPP_INCLUDED_)
#define CUPHY_TENSOR_TILE_HPP_INCLUDED_

#include "cuphy.h"
#include "tensor_desc.hpp"

namespace cuphy_i
{

////////////////////////////////////////////////////////////////////////
// tensor_tile()
cuphyStatus_t tensor_tile(const tensor_desc&    tDst,
                          void*                 dstAddr,
                          const tensor_desc&    tSrc,
                          const void*           srcAddr,
                          int                   tileRank,
                          const int*            tileExtents,
                          cudaStream_t          strm);


} // namespace cuphy_i

#endif // !defined(CUPHY_TENSOR_TILE_HPP_INCLUDED_)
