/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(DEVICE_HPP_INCLUDED_)
#define DEVICE_HPP_INCLUDED_

#include "cuphy_internal.h"

////////////////////////////////////////////////////////////////////////
// cuphy_i
namespace cuphy_i // cuphy internal
{
//----------------------------------------------------------------------
// device
class device //
{
public:
    device(int device_index = 0);
    int        multiProcessorCount()    const { return properties_.multiProcessorCount; }
    size_t     sharedMemPerBlock()      const { return properties_.sharedMemPerBlock; }
    size_t     sharedMemPerBlockOptin() const { return properties_.sharedMemPerBlockOptin; }
    static int get_count();
    static int get_current();

private:
    int            index_;
    cudaDeviceProp properties_;
};

} // namespace cuphy_i

#endif // !defined(DEVICE_HPP_INCLUDED_)
