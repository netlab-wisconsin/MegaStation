/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC2_C2V_ROW_DEP_CUH_INCLUDED_)
#define LDPC2_C2V_ROW_DEP_CUH_INCLUDED_

#include "ldpc2.cuh"

namespace ldpc2
{

////////////////////////////////////////////////////////////////////////
// C2V_row_dep
// Check-to-variable processor, with row-dependent processing selection
template <typename T,
          int      BG>
class C2V_row_dep
{
public:
    //------------------------------------------------------------------
    // init()
    __device__
    void init()
    {
        c2v_storage.init();
    }
private:
    //typedef __half            app_t;
    
    typedef cC2V_storage_t<T> c2v_storage_t;
    //------------------------------------------------------------------
    // Data
    c2v_storage_t c2v_storage;

};
          


} // namespace ldpc2

#endif // !defined(LDPC2_C2V_ROW_DEP_CUH_INCLUDED_)

