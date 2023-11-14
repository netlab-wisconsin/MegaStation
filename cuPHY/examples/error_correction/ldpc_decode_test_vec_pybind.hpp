/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#if !defined(LDPC_DECODE_TEST_VEC_PYBIND_HPP_INCLUDED_)
#define LDPC_DECODE_TEST_VEC_PYBIND_HPP_INCLUDED_

#include <string>
#include "ldpc_decode_test_vec.hpp"

struct test_vec_pybind_params
{
    test_vec_pybind_params(cuphy::tensor_device ip_llr,
                            cuphyDataType_t     llr_type,
                            uint16_t            b,
                            uint16_t            bg,
                            uint16_t            num_cw_lim) :
        inputLLR(ip_llr),
        LLRtype(llr_type),
        B(b),
        BG(bg),
        num_cw_limit(num_cw_lim)
        {

        }

        cuphy::tensor_device    inputLLR;
        cuphyDataType_t         LLRtype;
        uint16_t                B;
        uint16_t                BG;
        uint16_t                num_cw_limit;  
};

///////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_pybind
class ldpc_decode_test_vec_pybind : public ldpc_decode_test_vec
{
    public:
        //------------------------------------------------------------------
        // ldpc_decode_test_vec_pybind()
        ldpc_decode_test_vec_pybind(const test_vec_pybind_params& pyparams);
        //------------------------------------------------------------------
        // ~ldpc_decode_test_vec_pybind()
        virtual ~ldpc_decode_test_vec_pybind() {}
        //------------------------------------------------------------------
        // desc()
        // Descriptive string
        virtual const char* desc() const override;
        //------------------------------------------------------------------
        // Prepare a test vector
        virtual void generate() override;
        
    private:
        //------------------------------------------------------------------
        // populate_config()
        void populate_config(const test_vec_pybind_params& pyparams);
        //------------------------------------------------------------------
        // Data
        cuphy::tensor_device     inputLLR_; // Input LLR Tensor 
        uint16_t                 num_cw_limit_; // Restrict processing to partiular size
};

#endif //!defined(LDPC_DECODE_TEST_VEC_PYBIND_HPP_INCLUDED_)