/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(CUPHY_RNG_HPP_INCLUDED_)
#define CUPHY_RNG_HPP_INCLUDED_

#include "cuphy.h"
#include "cuphy_internal.h"
#include "tensor_desc.hpp"
#include <curand_kernel.h>

////////////////////////////////////////////////////////////////////////
// cuphyRNG
// Empty base class for internal random number generator class, used by
// forward declaration in public-facing cuphy.h.
struct cuphyRNG
{
};


namespace cuphy_i // cuphy internal
{

////////////////////////////////////////////////////////////////////////
// rng
class rng : public cuphyRNG
{
public:
    //------------------------------------------------------------------
    // rng()
    // Constructor
    rng(unsigned long long seed, cudaStream_t s);
    //------------------------------------------------------------------
    // normal()
    // Populate the given tensor with random values from a Gaussian
    // (normal) distribution
    cuphyStatus_t normal(const tensor_desc&    t,
                         void*                 p,
                         const cuphyVariant_t& mean,
                         const cuphyVariant_t& stddev,
                         cudaStream_t          strm);
    //------------------------------------------------------------------
    // uniform()
    // Populate the given tensor with random values from a uniform
    // distribution.
    cuphyStatus_t uniform(const tensor_desc&    t,
                          void*                 p,
                          const cuphyVariant_t& min_v,
                          const cuphyVariant_t& max_v,
                          cudaStream_t          strm);

private:
    cuphy_i::unique_device_ptr<curandState> randStates_;
};


} // namespace cuphy_i

#endif // !defined(CUPHY_RNG_HPP_INCLUDED_)
