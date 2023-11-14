/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "cuphy_context.hpp"
#include "cuphy_internal.h"

namespace cuphy_i
{
    
////////////////////////////////////////////////////////////////////////
// cuphy_i::context::context()
context::context()
{
    //------------------------------------------------------------------
    // Retrieve the device that will be associated with this context
    cudaError_t e = cudaGetDevice(&deviceIndex_);
    if(cudaSuccess != e)
    {
        throw cuda_exception(e);
    }
    //------------------------------------------------------------------
    // Retrieve device properties
    cudaDeviceProp devProp;
    e = cudaGetDeviceProperties(&devProp, deviceIndex_);
    if(cudaSuccess != e)
    {
        throw cuda_exception(e);
    }
    cc_                     = make_cc_uint64(devProp.major, devProp.minor);
    sharedMemPerBlockOptin_ = devProp.sharedMemPerBlockOptin;
    multiProcessorCount_    = devProp.multiProcessorCount;
    //------------------------------------------------------------------
    softDemapperContext_.reset(new soft_demapper_context);
};

} // namespace cuphy_i
