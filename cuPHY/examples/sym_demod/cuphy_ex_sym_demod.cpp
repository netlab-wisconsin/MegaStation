/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
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
#include <random>
#include "cuphy.hpp"
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"

using namespace cuphy;

////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("cuphy_ex_sym_demod [options]\n");
    printf("  Options:\n");
    printf("    -h                     Display usage information\n");
    printf("    -k                     Skip 'warmup' run before timing loop\n");
    printf("    -n noise_var           Complex noise variance (default: 1.0)\n");
    printf("    -s num_symbols         Number of symbols to demodulate\n");
    printf("    -q QAM                 QAM (2, 4, 16, 64, or 256) (default: 256)\n");
    printf("    -r num_runs            Number of times to perform batch decoding (default: 1)\n");
}

struct QAM_info
{
    int    num_bits; // number of bits
    double A;        // modulation normalization factor
    double range;    // constellation bounds
    QAM_info(int nb, double A_, double r) : num_bits(nb), A(A_), range(r)
    {}
};

////////////////////////////////////////////////////////////////////////
// get_QAM_info()
// Returns an initialized QAM_info structure, or throws an exception
// (for unsupported QAMs).
QAM_info get_QAM_info(int QAM)
{
    switch(QAM)
    {
    case 2:
        return QAM_info(1, 1 / sqrt(2.0),    2 / sqrt(2.0));
    case 4:
        return QAM_info(2, 1 / sqrt(2.0),    2 / sqrt(2.0));
    case 16:
        return QAM_info(4, 1 / sqrt(10.0),   4 / sqrt(10.0));
    case 64:
        return QAM_info(6, 1 / sqrt(42.0),   8 / sqrt(42.0));
    case 256:
        return QAM_info(8, 1 / sqrt(170.0), 16 / sqrt(170.0));
    default:
        throw std::runtime_error(std::string("Invalid QAM: ") + std::to_string(QAM));
    }
}

////////////////////////////////////////////////////////////////////////
// main()
int main(int argc, char* argv[])
{
    int returnValue = 0;
    cuphyNvlogFmtHelper nvlog_fmt("sym_demod.log");
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int          iArg = 1;
        std::string  inputFilename;
        bool         useHalf        = false;
        unsigned int numRuns        = 1;
        int          doWarmup       = true;
        int          numSymbols     = 1024;
        int          QAM            = 256;
        float        noiseVar       = 1.0f;
        while(iArg < argc)
        {
            if('-' == argv[iArg][0])
            {
                switch(argv[iArg][1])
                {
                case 'f':
                    useHalf = true;
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
                case 'n':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%f", &noiseVar)))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid noise variance");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case 'q':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%u", &QAM)) ||
                       (QAM < 2)                             ||
                       (QAM > 256))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid QAM value");
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
                case 's':
                    if((++iArg >= argc) ||
                       (1 != sscanf(argv[iArg], "%i", &numSymbols)) ||
                       (numSymbols < 1))
                    {
                        NVLOGE_FMT(NVLOG_PUSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of symbols: {}", argv[iArg]);
                        exit(1);
                    }
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
        QAM_info qInfo = get_QAM_info(QAM);
        printf("*********************************************************************\n");
        printf("Symbol demodulation configuration:\n");
        printf("*********************************************************************\n");
        //printf("HDF5 input file     = %s\n", inputFilename.c_str());
        printf("Number of symbols     = %i\n", numSymbols);
        printf("Number of runs        = %i\n", numRuns);
        printf("QAM                   = %i\n", QAM);
        printf("num_bits              = %i\n", qInfo.num_bits);
        printf("A                     = %f (1 / %.0f)\n", qInfo.A, (1 / (qInfo.A * qInfo.A)));
        printf("range                 = %f (%.0f A)\n", qInfo.range, round(qInfo.range / qInfo.A));
        printf("Data type             = %s\n", useHalf ? "fp16" : "fp32");
        printf("Noise variance        = %f\n", noiseVar);

        //--------------------------------------------------------------
        // Allocate device tensors
        tensor_device tSym(useHalf ? CUPHY_C_16F : CUPHY_C_32F,
                           {numSymbols});
        tensor_device tLLR(useHalf ? CUPHY_R_16F : CUPHY_R_32F,
                           {numSymbols * qInfo.num_bits});
        //--------------------------------------------------------------
        // Initialize random data
        typed_tensor<CUPHY_C_32F, pinned_alloc> symbols(numSymbols);
        std::mt19937                            e2;
        std::uniform_real_distribution<>        dist(-qInfo.range, qInfo.range);
        for(size_t i = 0; i < numSymbols; ++i)
        {
            symbols(i) = make_cuFloatComplex(dist(e2), dist(e2));
            //printf("%lu: (%f, %f)\n", i, symbols(i).x, symbols(i).y);
        }
        //--------------------------------------------------------------
        // Copy to device tensor
        tSym.copy(symbols);
        //--------------------------------------------------------------
        // Initialize a cuPHY context
        cuphy::context ctx;
        //--------------------------------------------------------------
        // Execute the API call in a timed loop
        cuphy::event_timer tmr;

        tmr.record_begin();
        for(unsigned int uRun = 0; uRun < numRuns; ++uRun)
        {
            ctx.demodulate_symbol(tLLR, tSym, qInfo.num_bits, noiseVar);
        }
        tmr.record_end();
        tmr.synchronize();
        //--------------------------------------------------------------
        // Copy output to host
        typed_tensor<CUPHY_R_32F, pinned_alloc> LLR_h(numSymbols * qInfo.num_bits);
        
        LLR_h.copy(tLLR);
        cudaStreamSynchronize(0); // Synchronize async copy to pinned mem

        //for(int i = 0; i < (numSymbols * qInfo.num_bits); ++i)
        //{
        //    printf("%i: %f\n", i, LLR_h(i));
        //}

        float avg_time_sec = tmr.elapsed_time_ms() / ((1000.0f) * numRuns);
        printf("Average (%u runs) elapsed time in usec = %.1f, throughput = %.2f \n",
               numRuns,
               tmr.elapsed_time_ms() * 1000 / numRuns,
               0 / avg_time_sec);
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
