/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC_DECODE_TEST_VEC_FILE_HPP_INCLUDED_)
#define LDPC_DECODE_TEST_VEC_FILE_HPP_INCLUDED_

#include <string>
#include "ldpc_decode_test_vec.hpp"


struct test_vec_file_params
{
    test_vec_file_params(const char*     fname,
                         cuphyDataType_t llr_type,
                         int             bg,
                         int             nparity,
                         int             num_cw_lim) :
      filename(fname),
      LLRtype(llr_type),
      BG(bg),
      num_parity(nparity),
      num_cw_limit(num_cw_lim)
    {
    }
    const char*     filename;
    cuphyDataType_t LLRtype;
    int             BG;
    int             num_parity;
    int             num_cw_limit;
};

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_file
class ldpc_decode_test_vec_file : public ldpc_decode_test_vec
{
public:
    //------------------------------------------------------------------
    // ldpc_decode_test_vec_file()
    ldpc_decode_test_vec_file(const test_vec_file_params& fparams);
    //------------------------------------------------------------------
    // ~ldpc_decode_test_vec_file()
    virtual ~ldpc_decode_test_vec_file() {}
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
    void populate_config(const test_vec_file_params& fparams);
    //------------------------------------------------------------------
    // Data
    std::string        filename_;     // Source input filename
    int                num_cw_limit_; // Used to restrict processing to
                                      // a subset of the file contents
    cuphy::tensor_desc limit_desc_;   // Used to represent a subset of the
                                      // file tensor, for when the user
                                      // does not want to process all file
                                      // codewords.
};

#endif // !defined(LDPC_DECODE_TEST_VEC_FILE_HPP_INCLUDED_)
