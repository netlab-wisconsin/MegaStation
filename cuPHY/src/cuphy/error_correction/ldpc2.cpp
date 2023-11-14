/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "ldpc2.hpp"

namespace
{
    
////////////////////////////////////////////////////////////////////////
// Cache of CUDA device properties
// (Calls to cudaGetDeviceProperties() are too expensive when we are
// concerned about latencies at the microsecond level.
// TODO: Make this an API level "context" construct that will be passed
// to all functions.
const int32_t SHMEM_OPTIN_CACHE_NUM_DEVICES = 32;
int32_t g_sharedMemPerBlockOptinCache[SHMEM_OPTIN_CACHE_NUM_DEVICES] =
{
    -1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1,
    -1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1,  -1, -1, -1, -1
};

}

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// ldpc::get_device_max_shmem_per_block_optin()
int32_t get_device_max_shmem_per_block_optin()
{
    int deviceIndex = -1;
    cudaError_t e = cudaGetDevice(&deviceIndex);
    if((cudaSuccess != e) || (deviceIndex >= SHMEM_OPTIN_CACHE_NUM_DEVICES))
    {
        return -1;
    }
    int32_t shmem_value = g_sharedMemPerBlockOptinCache[deviceIndex];
    if(shmem_value <= 0)
    {
        cudaDeviceProp prop;
        e = cudaGetDeviceProperties(&prop, deviceIndex);
        if(cudaSuccess != e)
        {
            return -1;
        }
        shmem_value = static_cast<int32_t>(prop.sharedMemPerBlockOptin);
        g_sharedMemPerBlockOptinCache[deviceIndex] = shmem_value;
    }
    return shmem_value;
}

// No longer valid: let c2v cache classes provide this information
////////////////////////////////////////////////////////////////////////
// get_c2v_shared_mem_size()
// Returns the size of shared memory required to hold ALL cC2V messages.
// TODO: Explore making the data storage adapt to the per-row
// requirements to store more rows in the finite amount of shared
// memory.
//uint32_t get_c2v_shared_mem_size(int numParity, int Z, int elem_size)
//{
//    if(2 == elem_size)
//    {
//        // Assume 16 bits min0 and min1, 32 bits of sign/index (for now).
//        return (numParity * Z * 8);
//    }
//    else
//    {
//        // Not used right now, but set a default that assumes 32-bit
//        // min0 and min1, plus 32 bits of sign/index
//        return (numParity * Z * 12);
//    }
//}

// No longer valid: let c2v cache classes provide this information
////////////////////////////////////////////////////////////////////////
// get_shmem_max_c2v_nodes()
//uint32_t get_shmem_max_c2v_nodes(int BG, int numParity, int Z, int elem_size)
//{
//    int32_t max_shmem_optin = get_device_max_shmem_per_block_optin();
//    if(max_shmem_optin <= 0)
//    {
//        return 0;
//    }
//    int32_t app_size = (1 == BG)                          ?
//                       ((22 + numParity) * Z * elem_size) :
//                       ((10 + numParity) * Z * elem_size);
//    if(app_size > max_shmem_optin)
//    {
//        return 0;
//    }
//    return (max_shmem_optin - app_size) / (Z * ((2 == elem_size) ? 8 : 12));
//}

} // namespace ldpc2
