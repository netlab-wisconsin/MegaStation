/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "device.hpp"

////////////////////////////////////////////////////////////////////////
// cuphy_i
namespace cuphy_i // cuphy internal
{
//----------------------------------------------------------------------
// device::device()
device::device(int device_index) :
    index_(device_index)
{
    if(device_index >= get_count())
    {
        throw cuda_exception(cudaErrorInvalidDevice);
    }
    cudaError_t res = cudaGetDeviceProperties(&properties_, device_index);
    if(cudaSuccess != res)
    {
        throw cuda_exception(res);
    }
}

//----------------------------------------------------------------------
// device::get_count()
int device::get_count()
{
    int         count = 0;
    cudaError_t res   = cudaGetDeviceCount(&count);
    if(cudaSuccess != res)
    {
        throw cuda_exception(res);
    }
    return count;
}

//----------------------------------------------------------------------
// device::get_current()
int device::get_current()
{
    int         device_idx = 0;
    cudaError_t res        = cudaGetDevice(&device_idx);
    if(cudaSuccess != res)
    {
        throw cuda_exception(res);
    }
    return device_idx;
}

} // namespace cuphy_i
