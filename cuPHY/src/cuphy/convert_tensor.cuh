/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(CONVERT_TENSOR_CUH_INCLUDED_)
#define CONVERT_TENSOR_CUH_INCLUDED_

#include "tensor_desc.hpp"

cuphyStatus_t convert_tensor_layout(const tensor_desc& dstTensorDesc,
                                    void*              dstAddr,
                                    const tensor_desc& srcTensorDesc,
                                    const void*        srcAddr,
                                    cudaStream_t       strm);

#endif // !defined(CONVERT_TENSOR_CUH_INCLUDED_)
