/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include "cuphy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include "cuphy.hpp"
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include "ldpc_decode_test_vec_file.hpp"
#include "ldpc_decode_test_vec_gen.hpp"

using namespace cuphy;

////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("cuphy_ex_ldpc [options]\n");
    printf("  Options:\n");
    printf("    -h                     Display usage information\n");
    printf("    Execution (Common) Options:\n");
    printf("    ---------------------------\n");
    printf("        -a algo_index          Use specific implementation (1 or 2) (default: 0 - let library decide)\n");
    printf("        -b                     Use the transport block LDPC interface (instead of the tensor interface)\n");
    printf("        -c max_block_count     Terminate the data loop when the given number of blocks has been\n");
    printf("                               decoded. (When the '-e' error count option is provided, this option can\n");
    printf("                               be used to avoid an infinite loop, as the provided SNR may not generate\n");
    printf("                               any bit or block errors.)\n");
    printf("        -d                     When using the transport block interface, 'spread' the data over mulitple\n");
    printf("                               transport blocks (%i), instead of one large transport block.\n",
           CUPHY_LDPC_DECODE_DESC_MAX_TB);
    printf("        -e block_err_count     Generate data and accumulate error statistics until 'block_err_count'\n");
    printf("                               blocks containing an error have occurred. (Not used for file-based input.)\n");
    printf("        -f                     Use half precision instead of single precision (Volta and later only)\n");
    printf("        -k                     Skip 'warmup' run before timing loop\n");
    printf("        -m normalization       Normalization factor for min-sum. If no value is provided, the library\n");
    printf("                               will choose an appropriate value, based on the LDPC configuration.\n");
    printf("        -n num_iter            Maximum number of LDPC iterations (default: 1)\n");
    printf("        -r num_runs            Number of times to perform batch decoding (default: 1)\n");
    printf("        -s                     Skip comparison of decoder output to input data\n");
    printf("        -t                     Instruct the library algorithm chooser to choose a kernel optimized for\n");
    printf("                               throughput (instead of latency) when a high throughput kernel is available.\n");
    printf("                               (Only valid when algo_index is 0 or is not specified on the command line.)\n");
    printf("\n");
    printf("    Input Data Options:\n");
    printf("    -------------------\n");
    printf("        File Based Input:\n");
    printf("        - - - - - - - - -\n");
    printf("            When using file based input, no additional puncturing or shortening is performed. BER/BLER\n");
    printf("            will reflect the puncturing/shortening conditions used to generate the input data.\n");
    printf("            The number of input information bits is determined from the 'sourceData' data set in\n");
    printf("            the input file, and the lifting size Z is derived appropriately.\n");
    printf("            -i input_filename      Input HDF5 file name, which must contain the following datasets:\n");
    printf("                                       sourceData:    uint8 data set with source information bits\n");
    printf("                                       inputLLR:      Log-likelihood ratios for coded, modulated symbols\n");
    printf("                                       inputCodeWord: uint8 data set with encoded bits (optional)\n");
    printf("                                                     (Initial bits are sourceData. No puncturing assumed.)\n");
    printf("            -g base_graph          Base graph assumed for input data (default: 1)\n");
    printf("            -p mb                  Number of parity nodes mb (must be between 4 and 46 for BG1, and between\n");
    printf("                                   4 and 42 for BG2) (default: 8)\n");
    printf("            -w numCBLimit          Decode numCBLimit code blocks (instead of the total number contained in the\n");
    printf("                                   input file). Must be less than or equal to the number of codewords in the\n");
    printf("                                   input file, or an error will be generated.\n");
    printf("\n");
    printf("        Generating Input Data:\n");
    printf("        - - - - - - - - - - -\n");
    printf("            -g base_graph          Base graph used to generate input data (default: 1)\n");
    printf("            -w num_codewords       Number of codewords to generate (default: 80)\n");
    printf("            -B block_sz            Input data block size (before LDPC encoding). Uses the base graph selection\n");
    printf("                                   to determine the lifting size Z.\n");
    printf("            -M modulation          Modulation used before adding noise. Valid values are 'BPSK', 'QPSK', \n");
    printf("                                   'QAM16', 'QAM64', or 'QAM256'. (default: 'QPSK')\n");
    printf("            -N mod_bits            Number of modulated bits (info + parity) in each codeword.\n");
    printf("            -p mb                  Number of parity nodes mb (must be between 4 and 46 for BG1, and between\n");
    printf("                                   4 and 42 for BG2). This value is not used if the code rate 'R' is specified,\n");
    printf("                                   or if the number of modulated bits 'N' is specified. (default: 8)\n");
    printf("            -P                     Puncture the generated test vector data (default: false)\n");
    printf("            -R code_rate           Code rate. Used in conjunction with the block size parameter 'B' to determine\n");
    printf("                                   the number of parity nodes as well as the number of punctured parity bits. The\n");
    printf("                                   code rate value is ignored if the '-N' option is used to explicitly provide\n");
    printf("                                   the number of modulated bits, and the code rate is instead derived from that\n");
    printf("                                   value\n");
    printf("            -S SNR                 SNR (in dB) for generated noise. The (complex) noise variance is given by \n");
    printf("                                   10^(-SNR_dB/10). The variance of the real and imaginary components are assumed\n");
    printf("                                   to be equal, and in this case each is equal to half of the complex variance.\n");
    printf("                                   (default SNR: 10)\n");
    printf("            -Z size                Lifting size for generated data. This option is only used if the data block\n");
    printf("                                   size is NOT specified. If this option is specified, the number\n");
    printf("                                   of filler bits is zero, and no parity bits are punctured.\n");
    printf("\n");
    printf("            Ways to specify randomly generated input data:\n");
    printf("            B, N   (input block size and number of modulated bits)\n");
    printf("            B, R   (input block size and code rate)\n");
    printf("            p, Z   (num parity nodes and lifting size)\n");
}

////////////////////////////////////////////////////////////////////////
// get_QAM_log2()
int get_QAM_log2(const char* str)
{
    if(0 == strcmp(str, "QAM256"))
    {
        return CUPHY_QAM_256;
    }
    else if(0 == strcmp(str, "QAM64"))
    {
        return CUPHY_QAM_64;
    }
    else if(0 == strcmp(str, "QAM16"))
    {
        return CUPHY_QAM_16;
    }
    else if(0 == strcmp(str, "QPSK"))
    {
        return CUPHY_QAM_4;
    }
    else if(0 == strcmp(str, "BPSK"))
    {
        return CUPHY_QAM_2;
    }
    else
    {
        std::runtime_error(std::string("Invalid modulation: ") + str);
        return 0;
    }
}

////////////////////////////////////////////////////////////////////////
// LDPC_decode_error_stats
class LDPC_decode_error_stats
{
public:
    typedef cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> err_count_tensor_t;

    LDPC_decode_error_stats() :
        bit_error_count_(0),
        bit_count_(0),
        block_error_count_(0),
        block_count_(0)
    {
    }
    //------------------------------------------------------------------
    // update()
    template <class TSrc, class TDecoded>
    void update(TSrc& src, TDecoded& decoded)
    {
        typedef cuphy::typed_tensor<CUPHY_R_32U, cuphy::pinned_alloc> tensor_uint32_p_t;
        
        const int             B      = src.dimensions()[0];
        const int             NUM_CW = decoded.dimensions()[1];
        cuphy::tensor_device  xor_results(CUPHY_BIT, B, NUM_CW);
        tensor_uint32_p_t     err_count(1, NUM_CW);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Generate a reference to ONLY the information bits of the
        // decoder output, without any possible filler bits.
        // tDecodeB = tDecode(0:B-1, :)
        cuphy::tensor_ref tDecodeB = decoded.subset(cuphy::index_group(cuphy::index_range(0, B),
                                                                       cuphy::dim_all()));
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // XOR decoder output with source bits. Each set bit in the
        // xor_results output indicates a bit error.
        // xor_results = tDecodeB ^ src_bits
        cuphy::tensor_xor(xor_results,
                          tDecodeB,
                          src);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Count the number of set bits in each column (codeword)
        cuphy::tensor_reduction_sum(err_count, xor_results, 0);
        cudaStreamSynchronize(0);
        //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
        // Update error statistics
        update_statistics(err_count, B);
    }
    void update_statistics(err_count_tensor_t& tErrorCount,
                           int                 bitsPerCodeword)
    {
        // Input is uint32_t tensor with dimensions (1, NUM_CW)
        int NUM_CW = tErrorCount.dimensions()[1];
        
        for(int i = 0; i < NUM_CW; ++i)
        {
            uint32_t cwBitErrors = tErrorCount(0, i);
            //printf("%i: %u\n", i, err_count(0, i));
            bit_error_count_ += cwBitErrors;
            if(cwBitErrors > 0)
            {
                ++block_error_count_;
            }
        }
        bit_count_   += (bitsPerCodeword * NUM_CW);
        block_count_ += NUM_CW;
    }
    uint64_t bit_error_count()   const { return bit_error_count_;   }
    uint64_t bit_count()         const { return bit_count_;         }
    uint32_t block_error_count() const { return block_error_count_; }
    uint32_t block_count()       const { return block_count_;       }
    float    BER()               const { return static_cast<float>(bit_error_count_)   / bit_count_;   }
    float    BLER()              const { return static_cast<float>(block_error_count_) / block_count_; }
private:
    uint64_t bit_error_count_;
    uint64_t bit_count_;
    uint32_t block_error_count_;
    uint32_t block_count_;
};

////////////////////////////////////////////////////////////////////////
// LDPC_decode_timing_stats
class LDPC_decode_timing_stats
{
public:
    LDPC_decode_timing_stats() :
        run_count_(0),
        total_time_milliseconds_(0.0),
        total_bits_(0.0)
    {
    }
    //------------------------------------------------------------------
    // update()
    void update(float t_milliseconds, int nruns, int64_t bits_per_run)
    {
        run_count_               += nruns;
        total_time_milliseconds_ += t_milliseconds;
        total_bits_              += (bits_per_run * nruns);
        //printf("Average (%u runs) elapsed time in usec = %.1f, throughput = %.2f Gbps\n",
        //       nruns,
        //       t_milliseconds * 1000 / nruns,
        //       (bits_per_run * nruns) / (t_milliseconds / 1000.0f) / 1.0e9);
    }
    int64_t num_runs()          const { return run_count_; }
    float   average_time_usec() const { return (total_time_milliseconds_ * 1000) / run_count_; }
    float   throughput()        const { return (total_bits_ * 1.0e-9) / (total_time_milliseconds_ / 1000.0); }
private:
    int64_t run_count_;
    double  total_time_milliseconds_;
    double  total_bits_;
};

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    
    cuphyNvlogFmtHelper nvlog_fmt("ldpc_decoder.log");
    
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int          iArg = 1;
        std::string  inputFilename;
        int          numIterations        = 1;
        bool         useHalf              = false;
        int          parityNodes          = 46;
        int          algoIndex            = 0;
        bool         compareDecodeOutput  = true;
        unsigned int numRuns              = 1;
        int          numCBLimit           = -1;
        int          doWarmup             = true;
        float        minSumNorm           = 0.0f;
        int          BG                   = 1;
        bool         puncture             = true;
        int          Zi                   = 64;
        float        SNR                  = 10.0f;
        bool         useTBInterface       = false;
        int          blockSize            = -1;
        float        codeRate             = 0.0f;
        int          modulatedBits        = -1;
        int          log2QAM              = CUPHY_QAM_64;
        int          min_block_err_cnt    = 0;
        int          max_block_cnt        = 1000000;
        bool         chooseHighThroughput = false;
        bool         spreadTB             = false;
        while(iArg < argc)
        {
            if('-' == argv[iArg][0])
            {
                switch(argv[iArg][1])
                {
                case 'a':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &algoIndex)) ||
                       (algoIndex < 0))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid algorithm index: {}", argv[iArg]);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'b':
                    useTBInterface = true;
                    ++iArg;
                    break;
                case 'c':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &max_block_cnt)) ||
                       (max_block_cnt <= 0))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid max block count: {}", argv[iArg]);
                        exit(1);
                    }                    
                    ++iArg;
                    break;
                case 'd':
                    spreadTB = true;
                    ++iArg;
                    break;
                case 'B':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &blockSize)) ||
                       (blockSize <= 0))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid block size: {}", argv[iArg]);
                        exit(1);
                    }                    
                    ++iArg;
                    break;
                case 'e':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &min_block_err_cnt)))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid minimum block error count");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'f':
                    useHalf = true;
                    ++iArg;
                    break;
                case 'g':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &BG)) ||
                       (BG < 1)                             ||
                       (BG > 2))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid base graph");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'h':
                    usage();
                    exit(0);
                    break;
                case 'i':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: No input file name specified");
                        exit(1);
                    }
                    inputFilename.assign(argv[iArg++]);
                    break;
                case 'k':
                    doWarmup = false;
                    ++iArg;
                    break;
                case 'm':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%f", &minSumNorm)))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid normalization");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'M':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: No modulation provided");
                        exit(1);
                    }
                    log2QAM = get_QAM_log2(argv[iArg]);
                    ++iArg;
                    break;
                case 'n':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &numIterations)) ||
                       (numIterations < 0))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of iterations");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'N':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &modulatedBits)) ||
                       (modulatedBits <= 0))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of modulated bits: {}", argv[iArg]);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'p':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &parityNodes)) ||
                       (parityNodes <= 3) ||
                       (parityNodes > 46))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of parity nodes: {}", argv[iArg]);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'r':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%u", &numRuns)) ||
                       (numRuns < 1))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of runs: {}", argv[iArg]);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'R':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%f", &codeRate)) ||
                       (codeRate <= 0.0f))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid code rate: {}", argv[iArg]);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 's':
                    compareDecodeOutput = false;
                    ++iArg;
                    break;
                case 'w':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &numCBLimit)) ||
                       (numCBLimit < 1))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of codewords: {}", argv[iArg]);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'P':
                    puncture = true;
                    ++iArg;
                    break;
                case 'Z':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%u", &Zi)) ||
                       (Zi < 1))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid Z: {}", argv[iArg]);
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'S':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%f", &SNR)))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid SNR value");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 't':
                    chooseHighThroughput = true;
                    ++iArg;
                    break;
                default:
                    NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                    usage();
                    exit(1);
                    break;
                }
            }
            else
            {
                NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
                exit(1);
            }
        }
        //--------------------------------------------------------------
        // Display device (GPU) info
        printf("*********************************************************************\n");
        cuphy::device gpuDevice;
        printf("%s\n", gpuDevice.desc().c_str());
        //--------------------------------------------------------------
        // Create a cuPHY context
        cuphy::context ctx;
        //--------------------------------------------------------------
        // Create a random number generator, in case we need it to
        // generate source input data
        cuphy::rng rng_gen;
        //--------------------------------------------------------------
        // Initialize test data and the LDPC configuration using command
        // line arguments.
        cuphyDataType_t                       LLR_type = useHalf ? CUPHY_R_16F : CUPHY_R_32F;
        std::unique_ptr<ldpc_decode_test_vec> ptv;
        if(!inputFilename.empty())
        {
            // Load a test vector from an input file
            ptv.reset(new ldpc_decode_test_vec_file(test_vec_file_params(inputFilename.c_str(), // input file name
                                                                         LLR_type,              // LLR data type
                                                                         BG,                    // base graph
                                                                         parityNodes,           // num parity nodes
                                                                         numCBLimit)));         // limit num CWs
        }
        else
        {
            // Generate test vector data randomly
            ptv.reset(new ldpc_decode_test_vec_gen(ctx,                                  // cuPHY context
                                                   rng_gen,                              // random number generator
                                                   test_vec_gen_params(LLR_type,         // LLR data type
                                                                       BG,               // base graph
                                                                       Zi,               // lifting size
                                                                       parityNodes,      // num parity nodes
                                                                       numCBLimit,       // number of codewords
                                                                       blockSize,        // number of input bits
                                                                       codeRate,         // code rate
                                                                       modulatedBits,    // number of modulated bits
                                                                       log2QAM,          // modulation
                                                                       SNR,              // signal-to-noise ratio
                                                                       puncture)));      // puncture first 2Z LLRs
        }
        //------------------------------------------------------------------
        // Display LDPC test vector configuration info
        ldpc_decode_test_vec&              tv  = *ptv;
        const ldpc_decode_test_vec_config& tv_cfg = tv.config();
        tv.print_config();

        //------------------------------------------------------------------
        // Allocate an output buffer for decoded bits
        tensor_device tDecode(CUPHY_BIT,
                              tv.config().K, //MAX_DECODED_CODE_BLOCK_BIT_SIZE,
                              tv.config().num_cw,
                              cuphy::tensor_flags::align_coalesce);

        //printf("Decode: addr: %p, %s, size: %.1f kB\n\n",
        //       tDecode.addr(),
        //       tDecode.desc().get_info().to_string().c_str(),
        //       tDecode.desc().get_size_in_bytes() / 1024.0);
        //--------------------------------------------------------------
        // Create an LDPC decoder instance
        cuphy::LDPC_decoder dec(ctx);
        //--------------------------------------------------------------
        // Initialize an LDPC decode configuration. This is used for
        // both the tensor and transport block interfaces.
        uint32_t decode_flags = chooseHighThroughput ? CUPHY_LDPC_DECODE_CHOOSE_THROUGHPUT : 0;
        cuphy::LDPC_decode_config dec_cfg(LLR_type,      // LLR type (fp16 or fp32)
                                          tv_cfg.mb,     // num parity nodes
                                          tv_cfg.Z,      // lifting size
                                          numIterations, // max num iterations
                                          tv_cfg.Kb,     // info nodes
                                          minSumNorm,    // normalization value
                                          decode_flags,  // flags
                                          tv_cfg.BG,     // base graph
                                          algoIndex,     // algorithm index
                                          nullptr);      // workspace address
        //--------------------------------------------------------------
        // If no normalization value was provided, query the library for
        // an appropriate value.
        if(minSumNorm <= 0.0f)
        {
            dec.set_normalization(dec_cfg);
        }
        printf("Normalization                    = %f\n", dec_cfg.get_norm());
        printf("Number of iterations             = %i\n", numIterations);
        printf("\n");
        //--------------------------------------------------------------
        // Initialize an LDPC decode descriptor structure. (This is only
        // used when the transport block interface is selected.)
        LDPC_decode_desc dec_desc(dec_cfg);
        if(useTBInterface)
        {
            if(spreadTB)
            {
                // Spread the codewords out into multiple transport blocks,
                // with addresses that point back to the original input
                // tensor.
                const int         CW_PER_TB = (tv_cfg.num_cw + (CUPHY_LDPC_DECODE_DESC_MAX_TB - 1)) /
                                              CUPHY_LDPC_DECODE_DESC_MAX_TB;
                cuphy::tensor_ref tLLR(tv.LLR_desc(), tv.LLR_addr());
                for(int iCW = 0; iCW < tv_cfg.num_cw; iCW += CW_PER_TB)
                {
                    cuphy::index_group slice(cuphy::dim_all(),
                                             cuphy::index_range(iCW, std::min(iCW + CW_PER_TB, tv_cfg.num_cw)));
                    cuphy::tensor_ref  sLLR    = tLLR.subset(slice);
                    cuphy::tensor_ref  sDecode = tDecode.subset(slice);
                    //printf("start = %i, end = %i\n", slice.ranges()[1].start(), slice.ranges()[1].end());
                    dec_desc.add_tensor_as_tb(sLLR.desc(),    sLLR.addr(),
                                              sDecode.desc(), sDecode.addr());
                }
            }
            else
            {
                dec_desc.add_tensor_as_tb(tv.LLR_desc(),
                                          tv.LLR_addr(),
                                          tDecode.desc(),
                                          tDecode.addr());
            }
        }
        //--------------------------------------------------------------
        // Initialize an LDPC decode tensor params structure. (This is
        // only used when the tensor-based decoder interface is selected.)
        LDPC_decode_tensor_params dec_tensor(dec_cfg,                 // LDPC configuration
                                             tDecode.desc().handle(), // output descriptor
                                             tDecode.addr(),          // output address
                                             tv.LLR_desc().handle(),  // LLR descriptor
                                             tv.LLR_addr());          // LLR address
        //--------------------------------------------------------------
        // Decoder execution loop
        LDPC_decode_error_stats  error_stats;
        LDPC_decode_timing_stats timing_stats;
        do
        {
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Generate test vector data
            tv.generate();
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Warmup run
            if(doWarmup)
            {
                if(useTBInterface)
                {
                    dec.decode(dec_desc);
                }
                else
                {
                    dec.decode(dec_tensor);
                }
            }
            cudaDeviceSynchronize();
            //char b;
            //cudaMemcpy(&b, tDecode.addr(), 1, cudaMemcpyDeviceToHost);
            //printf("Decode: %i\n", b);
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Timed run
            cuphy::event_timer tmr;

            tmr.record_begin();
            for(unsigned int uRun = 0; uRun < numRuns; ++uRun)
            {
                //-  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -  -
                // Decode
                if(useTBInterface)
                {
                    dec.decode(dec_desc);
                }
                else
                {
                    dec.decode(dec_tensor);
                }
            }
            tmr.record_end();
            tmr.synchronize();
            timing_stats.update(tmr.elapsed_time_ms(), numRuns, tv_cfg.B * tv_cfg.num_cw);
            //- - - - - - - - - - - - - - - - - - - - - - - - - - - - -
            // Compare decoder output to source bits
            if(compareDecodeOutput)
            {
                error_stats.update(tv.src_bits(), tDecode);
            }
            
            // Optional: export to HDF5
            //hdf5hpp::hdf5_file f = hdf5hpp::hdf5_file::create("ldpc_decode_output.h5");
            //tv.export_hdf5(f);
            
        } while((error_stats.block_count()       < max_block_cnt)     &&
                (error_stats.block_error_count() < min_block_err_cnt));
        //--------------------------------------------------------------
        // Display aggregated timing and error statistics
        printf("Average (%li runs) elapsed time in usec = %.1f, throughput = %.2f Gbps\n",
               timing_stats.num_runs(),
               timing_stats.average_time_usec(),
               timing_stats.throughput());

        if(compareDecodeOutput)
        {
            printf("bit error count = %lu, bit error rate (BER) = (%lu / %lu) = %.5e, block error rate (BLER) = (%u / %u) = %.5e\n",
                   error_stats.bit_error_count(),
                   error_stats.bit_error_count(),
                   error_stats.bit_count(),
                   error_stats.BER(),
                   error_stats.block_error_count(),
                   error_stats.block_count(),
                   error_stats.BLER());
        }
    }

    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    return returnValue;
}
