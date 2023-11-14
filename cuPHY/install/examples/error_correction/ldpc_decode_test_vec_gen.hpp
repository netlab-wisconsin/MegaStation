/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#if !defined(LDPC_DECODE_TEST_VEC_GEN_HPP_INCLUDED_)
#define LDPC_DECODE_TEST_VEC_GEN_HPP_INCLUDED_

#include <string>
#include "ldpc_decode_test_vec.hpp"


struct test_vec_gen_params
{
    test_vec_gen_params(cuphyDataType_t llr_type,
                        int             bg,
                        int             Z,
                        int             nparity,
                        int             num_cw_in,
                        int             blockSize,
                        float           codeRate,
                        int             modBits,
                        int             log2QAM,
                        float           SNR_in,
                        bool            puncture_in) :
      LLRtype(llr_type),
      BG(bg),
      lifting_size(Z),
      num_parity(nparity),
      num_cw(num_cw_in),
      block_size(blockSize),
      code_rate(codeRate),
      num_modulated_bits(modBits),
      log2_QAM(log2QAM),
      SNR(SNR_in),
      puncture(puncture_in)
    {
    }
    cuphyDataType_t LLRtype;
    int             BG;
    int             num_parity;
    int             lifting_size;
    int             num_cw;
    int             block_size;
    float           code_rate;
    int             num_modulated_bits;
    int             log2_QAM;
    float           SNR;
    bool            puncture;
};

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_gen
class ldpc_decode_test_vec_gen : public ldpc_decode_test_vec
{
public:
    //------------------------------------------------------------------
    // ldpc_decode_test_vec_gen()
    ldpc_decode_test_vec_gen(cuphy::context&            ctx,
                             cuphy::rng&                rand_gen,
                             const test_vec_gen_params& fparams);
    //------------------------------------------------------------------
    // ~ldpc_decode_test_vec_gen()
    virtual ~ldpc_decode_test_vec_gen() {}
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
    void populate_config(const test_vec_gen_params& fparams);
    //------------------------------------------------------------------
    // Data
    cuphy::context&      ctx_;
    cuphy::rng&          rng_gen_;
    int                  num_cw_;
    int                  log2_QAM_;
    cuphy::tensor_device tSymbols_;
    cuphy::tensor_device tEncoded_;
    float                SNR_;
};

#endif // !defined(LDPC_DECODE_TEST_VEC_GEN_HPP_INCLUDED_)
