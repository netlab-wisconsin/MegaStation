/*
 * Copyright (c) 2021-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <stdio.h>
#include <limits>
#include "hdf5hpp.hpp"
#include "cuphy_hdf5.hpp"
#include "ldpc_decode_test_vec_gen.hpp"

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_gen::ldpc_decode_test_vec_gen()
ldpc_decode_test_vec_gen::ldpc_decode_test_vec_gen(cuphy::context&            ctx,
                                                   cuphy::rng&                rng_gen,
                                                   const test_vec_gen_params& gparams) :
    ldpc_decode_test_vec(gparams.LLRtype),
    ctx_(ctx),
    rng_gen_(rng_gen),
    num_cw_(std::max(gparams.num_cw, 1)),
    log2_QAM_(gparams.log2_QAM),
    SNR_(gparams.SNR)
{
    //------------------------------------------------------------------
    // Populate LDPC "configuration" data using input parameters
    populate_config(gparams);
    //------------------------------------------------------------------
    // Generate random data for input to the encoder. We will work with
    // scalar values instead of bits so that we can leverage the cuPHY
    // fill function for filler bits.
    cuphy::tensor_device tSrcWithFiller(CUPHY_R_8U, config_.K);
    rng_gen.uniform(tSrcWithFiller, 0, 1, 0);
    // Set filler bits to 0 before encoding
    if(config_.F > 0)
    {
        cuphy::index_group grp(cuphy::index_range(config_.K - config_.F,
                                                  config_.K));
        cuphy::tensor_ref  tSrcF = tSrcWithFiller.subset(grp);
        cuphy::tensor_fill(tSrcF, 0);
    }
    //------------------------------------------------------------------
    // Populate the source data tensor with only the source bits
    tSrcData_ = cuphy::tensor_device(CUPHY_BIT, config_.B);
    {
        cuphy::index_group grp(cuphy::index_range(0, config_.B));
        cuphy::tensor_ref  tSrcB = tSrcWithFiller.subset(grp);
        cuphy::tensor_convert(tSrcData_, tSrcB);
    }
    //------------------------------------------------------------------
    // Convert source data to type CUPHY_BIT, as required by the encoder
    cuphy::tensor_device tSrcWithFillerBits(CUPHY_BIT, config_.K);
    cuphy::tensor_convert(tSrcWithFillerBits, tSrcWithFiller);
    //------------------------------------------------------------------
    // Encode
    const int            MAXV = (1 == config_.BG)            ?
                                CUPHY_LDPC_MAX_BG1_VAR_NODES :
                                CUPHY_LDPC_MAX_BG2_VAR_NODES;
    cuphy::tensor_device tEncode(CUPHY_BIT, config_.Z * MAXV);
    cuphy::ldpc_encode(tEncode,
                       tSrcWithFillerBits,
                       config_.BG,
                       config_.Z);
    //------------------------------------------------------------------
    // Modulate (bits to complex values)
    const int            V               = config_.mb + ((1 == config_.BG) ? 22 : 10);
    const int            NUM_SYMBOLS     = (config_.Z * V + gparams.log2_QAM - 1) / gparams.log2_QAM;
    cuphy::index_group   grp(cuphy::index_range(0, config_.Z * V));
    cuphy::tensor_ref    tEncodedPartial = tEncode.subset(grp);
    tEncoded_ = cuphy::tensor_device(CUPHY_BIT, config_.Z * V);
    cuphy::tensor_convert(tEncoded_, tEncodedPartial);
    tSymbols_ = cuphy::tensor_device(CUPHY_C_16F, NUM_SYMBOLS);
    cuphy::modulate_symbol(tSymbols_, tEncodedPartial, gparams.log2_QAM);
    //------------------------------------------------------------------
    // Allocate a tensor to hold LLR data
    const int            NUM_LLR = NUM_SYMBOLS * log2_QAM_;
    tLLR_ = cuphy::tensor_device(gparams.LLRtype, NUM_LLR, num_cw_, cuphy::tensor_flags::align_coalesce);
    //------------------------------------------------------------------
    // Set the base class LLR descriptor
    set_LLR_desc(tLLR_.desc());
    //------------------------------------------------------------------
    cudaStreamSynchronize(0);
}

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_gen::desc()
const char* ldpc_decode_test_vec_gen::desc() const
{
    return "(generated at runtime)";
}

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_gen::populate_config()
void ldpc_decode_test_vec_gen::populate_config(const test_vec_gen_params& gparams)
{

    config_.BG       = gparams.BG;
    config_.num_cw   = num_cw_;
    if(gparams.block_size > 0)
    {
        config_.B  = gparams.block_size;
        config_.Z  = find_lifting_size(config_.BG, config_.B);
        config_.Kb = get_num_info_nodes(config_.BG, config_.B);
        config_.F  = (config_.Z * ((1 == config_.BG) ? 22 : 10)) - config_.B;
        if(gparams.num_modulated_bits > 0)
        {
            config_.N = gparams.num_modulated_bits;
            config_.R = static_cast<float>(config_.B) / static_cast<float>(config_.N);
        }
        else if(gparams.code_rate > 0.0f)
        {
            config_.R = gparams.code_rate;
            config_.N = std::lroundf(config_.B / gparams.code_rate);
        }
        else
        {
            throw std::runtime_error(std::string("Code rate or modulated bits must be "
                                                 "provided with input block size"));
        }
        // N: Number of modulated bits
        // B: Input block size
        // mb: number of parity nodes
        // P: punctured parity bits (0 <= P < Z)
        //
        // N = B + Z(mb - 2) - P
        // N - B             P
        // ----- + 2 = mb - ---
        //   Z               Z
        //
        //
        // 0 <= P/Z < 1
        //
        // mb = ceil(2 + (N - B)/Z) = ceil((2Z + N - B) / Z)
        config_.mb = static_cast<int>(std::ceil((config_.N - config_.B + (2 * config_.Z)) / static_cast<float>(config_.Z)));
        config_.P  = config_.B + (config_.Z * (config_.mb - 2)) - config_.N;
    }
    else
    {
        // Using values for Z, mb. Assuming no filler bits (except
        // for BG2 cases for Kb < 10), no punctured parity bits.
        config_.Z  = gparams.lifting_size;
        config_.mb = gparams.num_parity;
        config_.Kb = get_num_info_nodes_from_Z(config_.BG, config_.Z);
        config_.B  = config_.Z * config_.Kb;
        if(1 == config_.BG)
        {
            config_.F  = 0;
        }
        else
        {
            config_.F = (10 - config_.Kb) * config_.Z;
        }
        config_.P = 0;
        // Use the base class function to calculate the number of modulated
        // bits
        update_config_modulated_bits();
        config_.R = static_cast<float>(config_.B) / static_cast<float>(config_.N);
    }
    config_.K    = config_.B + config_.F;
    config_.QAM  = get_QAM_desc(gparams.log2_QAM);
    config_.punc = gparams.puncture ? LLR_PUNCTURE_STATUS_ENABLED :  LLR_PUNCTURE_STATUS_DISABLED;
}

////////////////////////////////////////////////////////////////////////
// ldpc_decode_test_vec_gen::generate()
void ldpc_decode_test_vec_gen::generate()
{
    //------------------------------------------------------------------
    // Generate random noise. Assuming half of noise power is real and
    // half is complex, and that the input SNR is the total (real +
    // complex).
    const float NOISE_VAR              = std::pow(10.0f, SNR_ / (-10.0f));
    const float NOISE_VAR_HALF         = NOISE_VAR / 2.0f;
    const float NOISE_COMPONENT_STDDEV = std::sqrt(NOISE_VAR_HALF);
    const cuComplex mean               = make_cuFloatComplex(0.0f, 0.0f);
    const cuComplex stddev             = make_cuFloatComplex(NOISE_COMPONENT_STDDEV,
                                                             NOISE_COMPONENT_STDDEV);
    //printf("NOISE_STDDEV = %f\n", NOISE_STDDEV);
    const int NUM_SYMBOLS    = tSymbols_.dimensions()[0];
    cuphy::tensor_device tNoise(CUPHY_C_16F, NUM_SYMBOLS, num_cw_);
    rng_gen_.normal(tNoise, mean, stddev);
    //------------------------------------------------------------------
    // Add noise to modulated symbols
    cuphy::tensor_device tSymbolsPlusNoise(CUPHY_C_16F, NUM_SYMBOLS, num_cw_);
    // Using broadcast semantics here - the symbols "column" vector is
    // automatically replicated for each column of the noise.
    cuphy::tensor_sum(tSymbolsPlusNoise,
                      tSymbols_,
                      tNoise);
    //------------------------------------------------------------------
    // Demodulate symbols (complex values to LLRs)
    ctx_.demodulate_symbol(tLLR_,
                           tSymbolsPlusNoise,
                           log2_QAM_,
                           NOISE_VAR);    
    //------------------------------------------------------------------
    // Set LLRs for punctured bits to 0
    // tLLR(0:2Z,:) = 0
    if(LLR_PUNCTURE_STATUS_ENABLED == config_.punc)
    {
        cuphy::index_group p_grp(cuphy::index_range(0, 2 * config_.Z),
                                 cuphy::dim_all());
        cuphy::tensor_ref tLLR_p = tLLR_.subset(p_grp);
        cuphy::tensor_fill(tLLR_p, 0);
    }
    //------------------------------------------------------------------
    // Set LLRs for filler bits to Inf
    // tLLR(K-F:K,:) = Inf
    if(config_.F > 0)
    {
        cuphy::index_group f_grp(cuphy::index_range(config_.K - config_.F,
                                                    config_.K),
                                 cuphy::dim_all());
        cuphy::tensor_ref  tLLR_f = tLLR_.subset(f_grp);
        cuphy::tensor_fill(tLLR_f, std::numeric_limits<float>::infinity());
    }
#if 0
    //------------------------------------------------------------------
    {
        cuphy::tensor_device tEncodeDebug(CUPHY_R_8U, config_.Z * MAXV);
        cuphy::tensor_convert(tEncodeDebug, tEncode);

        cuphy::tensor_device tSymbolsDebug(CUPHY_C_32F, NUM_SYMBOLS);
        cuphy::tensor_convert(tSymbolsDebug, tSymbols);

        cuphy::tensor_device tNoiseDebug(CUPHY_C_32F, NUM_SYMBOLS, config_.num_cw);
        cuphy::tensor_convert(tNoiseDebug, tNoise);

        cuphy::tensor_device tSymbolsPlusNoiseDebug(CUPHY_C_32F, NUM_SYMBOLS, config_.num_cw);
        cuphy::tensor_convert(tSymbolsPlusNoiseDebug, tSymbolsPlusNoise);

        cuphy::tensor_device tLLRDebug(CUPHY_R_32F, NUM_LLR, config_.num_cw);
        cuphy::tensor_convert(tLLRDebug, tLLR_);

        cudaStreamSynchronize(0);
        hdf5hpp::hdf5_file f = hdf5hpp::hdf5_file::create("debug_gen.h5");
        cuphy::write_HDF5_dataset(f, tSrcWithFiller,         "srcWithFiller",     0);
        cuphy::write_HDF5_dataset(f, tEncodeDebug,           "tEncode",           0);
        cuphy::write_HDF5_dataset(f, tSymbolsDebug,          "tSymbols",          0);
        cuphy::write_HDF5_dataset(f, tNoiseDebug,            "tNoise",            0);
        cuphy::write_HDF5_dataset(f, tSymbolsPlusNoiseDebug, "tSymbolsPlusNoise", 0);
        cuphy::write_HDF5_dataset(f, tLLRDebug,              "tLLR",              0);
        cudaStreamSynchronize(0);
    }
#endif
    cudaStreamSynchronize(0);
}
