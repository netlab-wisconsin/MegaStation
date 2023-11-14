/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(CUPHY_CONTEXT_HPP_INCLUDED_)
#define CUPHY_CONTEXT_HPP_INCLUDED_

#include <cuda_runtime.h>
#include <stdint.h>
#include <memory>
#include "soft_demapper/soft_demapper.hpp"

////////////////////////////////////////////////////////////////////////
// cuphyContext
// Empty base class for internal context class, used by forward
// declaration in public-facing cuphy.h.
struct cuphyContext
{
};

namespace cuphy_i // cuphy internal
{

constexpr uint64_t make_cc_uint64(int major, int minor)
{
    return (static_cast<uint64_t>(major) << 32) + static_cast<uint32_t>(minor);
}

   
////////////////////////////////////////////////////////////////////////
// cuphy_i::context
// cuPHY "context" object, used to (perhaps among other things) cache
// device properties.
class context : public cuphyContext
{
public:
    //------------------------------------------------------------------
    // Constructor
    context();
    //------------------------------------------------------------------
    // device index
    int index() const { return deviceIndex_; }
    //------------------------------------------------------------------
    // compute capability
    uint64_t compute_cap() const { return cc_; }
    //------------------------------------------------------------------
    // maximum shared mem per block (optin)
    int max_shmem_per_block_optin() const { return sharedMemPerBlockOptin_; }
    //------------------------------------------------------------------
    // SM count
    int sm_count() const { return multiProcessorCount_; }
    //------------------------------------------------------------------
    // soft demapper context (initialized with cuphy context)
    const soft_demapper_context& soft_demapper_ctx() const { return *softDemapperContext_; }
private:
    typedef std::unique_ptr<soft_demapper_context> demapper_ctx_ptr_t;
    //------------------------------------------------------------------
    // Data
    int                deviceIndex_;            // index of device associated with context
    uint64_t           cc_;                     // compute capability (major << 32) | minor
    int                sharedMemPerBlockOptin_; // maximum shared memory per block usable by option
    int                multiProcessorCount_;    // number of multiprocessors on device
    demapper_ctx_ptr_t softDemapperContext_;
};

}

#endif // !defined(CUPHY_CONTEXT_HPP_INCLUDED_)
