/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(CUPHY_SOFT_DEMAPPER_HPP_INCLUDED_)
#define CUPHY_SOFT_DEMAPPER_HPP_INCLUDED_

#include "cuphy.h"
#include "cuphy_internal.h"
#include "tensor_desc.hpp"

namespace cuphy_i // cuphy internal
{

class context;
////////////////////////////////////////////////////////////////////////
// soft_demapper_context
// Storage for read-only resources required for some implementations of
// soft demapping
class soft_demapper_context
{
public:
    //------------------------------------------------------------------
    // soft_demapper_context()
    soft_demapper_context();
    //------------------------------------------------------------------
    // QAM_tex()
    const mipmapped_texture& QAM_tex() const { return QAMtex_; }
private:
    mipmapped_texture QAMtex_;
};

////////////////////////////////////////////////////////////////////////
// soft_demap()
cuphyStatus_t soft_demap(context&     ctx,
                         tensor_desc& tLLR,
                         void*        pLLR,
                         tensor_desc& tSym,
                         const void*  pSym,
                         int          log2_QAM,
                         float        noiseVariance,
                         cudaStream_t strm);

} // namespace cuphy_i

#endif // !defined(CUPHY_SOFT_DEMAPPER_HPP_INCLUDED_)
