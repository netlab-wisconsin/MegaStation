/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include "cuda_profiler_api.h"
#include "cuphy.h"
#include "cuphy_internal.h"
#include "cuphy.hpp"
#include "util.hpp"
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"
#include <chrono>
using Clock     = std::chrono::steady_clock;
using TimePoint = std::chrono::time_point<Clock>;
template <typename T, typename unit>
using duration = std::chrono::duration<T, unit>;
template <typename T>
using ms = std::chrono::milliseconds;
template <typename T>
using us = std::chrono::microseconds;

using namespace cuphy;

////////////////////////////////////////////////////////////////////////
// usage()
void usage()
{
    printf("cuphy_ex_polar_encoder [options]\n");
    printf("  Options:\n");
    printf("    -h                  Display usage information\n");
    printf("    -i  input_filename  Input HDF5 filename, which must contain the following datasets:\n");
    printf("    -o  output_filename Write pipeline tensors to an HDF5 output file.\n");
    printf("                        (Not recommended for use during timing runs.)\n");
    printf("    -r  # of iterations Number of iterations to run\n");
    // printf("    --I                 Number of info bits\n");
    // printf("    --T                 Number of transmit bits\n");
    printf("    --V                 Verbose logging (default 1)\n");
    printf("                        0 - disable (No verbose logging)\n");
    printf("                        1 - verbose errors only\n");
    printf("                        2 - verbose errors and display coded/transmit bits\n");
}

int main(int argc, char* argv[])
{
    int returnValue = 0;
    cuphyNvlogFmtHelper nvlog_fmt("polar_encoder.log");
    try
    {
        //------------------------------------------------------------------
        // Parse command line arguments
        int         iArg = 1;
        std::string inputFilename;
        std::string outputFilename;
        uint32_t    nInfoBits   = 0;
        uint32_t    nTxBits     = 0;
        bool        enNvprof    = false;
        uint32_t    verboseMode = 1;
        uint32_t    nIter       = 1000;

        cudaStream_t cuStream;
        cudaStreamCreateWithFlags(&cuStream, cudaStreamNonBlocking);

        while(iArg < argc)
        {
            if('-' == argv[iArg][0])
            {
                switch(argv[iArg][1])
                {
                case 'i':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: No filename provided");
                    }
                    inputFilename.assign(argv[iArg++]);
                    break;
                case 'h':
                    usage();
                    exit(0);
                    break;
                case 'o':
                    if(++iArg >= argc)
                    {
                        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: No output file name given");
                    }
                    outputFilename.assign(argv[iArg++]);
                    break;
                case 'r':
                    if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nIter)) || ((nIter <= 0)))
                    {
                        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid number of run iterations");
                        exit(1);
                    }
                    ++iArg;
                    break;
                case '-':
                    switch(argv[iArg][2])
                    {
                    case 'I':
                        if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nInfoBits)) || ((nInfoBits < 1) || (nInfoBits > CUPHY_POLAR_ENC_MAX_INFO_BITS)))
                        {
                            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: # of information bits invalid {}", nInfoBits);
                            exit(1);
                        }
                        ++iArg;
                        break;
                    case 'T':
                        if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &nTxBits)) || ((nTxBits < 1) || (nTxBits > CUPHY_POLAR_ENC_MAX_TX_BITS)))
                        {
                            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: # of transmit bits invalid {}", nTxBits);
                            exit(1);
                        }
                        ++iArg;
                        break;
                    case 'V':
                        if((++iArg >= argc) || (1 != sscanf(argv[iArg], "%i", &verboseMode)) || ((verboseMode < 0) || (verboseMode > 2)))
                        {
                            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid verbose mode {}", verboseMode);
                            exit(1);
                        }
                        ++iArg;
                        break;
                    case 'P':
                        enNvprof = true;
                        nIter    = 1;
                        ++iArg;
                        break;
                    default:
                        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                        usage();
                        exit(1);
                        break;
                    }
                    break;
                default:
                    NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Unknown option: {}", argv[iArg]);
                    usage();
                    exit(1);
                    break;
                }
            }
            else
            {
                NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: Invalid command line argument: {}", argv[iArg]);
                exit(1);
            }
        }
        if(inputFilename.empty())
        {
            usage();
            exit(1);
        }

        bool verboseErrsOnly = false;
        bool verbose         = false;
        if(1 == verboseMode) verboseErrsOnly = true;
        if(2 == verboseMode) { verboseErrsOnly = true; verbose = true; } ;

        uint8_t procModeBmsk = 0; // Downlink

        cudaEvent_t eStart, eStop;
        CUDA_CHECK(cudaEventCreateWithFlags(&eStart, cudaEventBlockingSync));
        CUDA_CHECK(cudaEventCreateWithFlags(&eStop, cudaEventBlockingSync));

        //------------------------------------------------------------------
        // Open the input file
        hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFilename.c_str());
        using tensor_pinned_R_8U  = typed_tensor<CUPHY_R_8U, pinned_alloc>;

        cuphy::tensor_device tGpuInfoBits  = cuphy::tensor_from_dataset(fInput.open_dataset("InfoBits"), CUPHY_R_8U, cuphy::tensor_flags::align_tight, cuStream);
        tensor_pinned_R_8U   tCpuCodedBits = typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(fInput.open_dataset("CodedBits"), cuphy::tensor_flags::align_tight, cuStream);
        tensor_pinned_R_8U   tCpuTxBits    = typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(fInput.open_dataset("TxBits"), cuphy::tensor_flags::align_tight, cuStream);

        uint32_t nExpectedCodedBits = 0;

        cuphy::disable_hdf5_error_print(); // Disable HDF5 stderr printing
        try
        {
            cuphy::cuphyHDF5_struct encPrms = cuphy::get_HDF5_struct(fInput, "encPrms");
            nInfoBits                       = encPrms.get_value_as<uint32_t>("nInfoBits");
            nExpectedCodedBits              = encPrms.get_value_as<uint32_t>("nCodedBits");
            nTxBits                         = encPrms.get_value_as<uint32_t>("nTxBits");
        }
        catch(const std::exception& exc)
        {
            printf("%s\n", exc.what());
            throw exc;
            // Continue using command line arguments if the input file does not
            // have an encPrms struct.

            nExpectedCodedBits = round_up_to_next(nInfoBits, 32U);
        }
        cuphy::enable_hdf5_error_print(); // Re-enable HDF5 stderr printing

        // Allocate output tensors
        // For coded bits provide the worst case storage
        cuphy::tensor_device tGpuCodedBits(CUPHY_R_8U,
                                           div_round_up(CUPHY_POLAR_ENC_MAX_CODED_BITS, 8),
                                           cuphy::tensor_flags::align_tight);
        cuphy::tensor_device tGpuTxBits(CUPHY_R_8U,
                                        round_up_to_next(CUPHY_POLAR_ENC_MAX_TX_BITS, 32) / 8, // roundup to nearest 32b boundary (multiple of words)
                                        cuphy::tensor_flags::align_tight);

        cudaStreamSynchronize(cuStream);
        cudaDeviceSynchronize(); // Needed because typed_tensor does not support non-default streams

        //------------------------------------------------------------------
        // Run the test
        if(enNvprof) cudaProfilerStart();

        TimePoint startTime = Clock::now();
        CUDA_CHECK(cudaEventRecord(eStart, cuStream));
        uint32_t nCodedBits = 0;

        for(uint32_t i = 0; i < nIter; ++i)
        {
            cuphyStatus_t polarEncStat = cuphyPolarEncRateMatch(nInfoBits,
                                                                nTxBits,
                                                                static_cast<uint8_t const*>(tGpuInfoBits.addr()),
                                                                &nCodedBits,
                                                                static_cast<uint8_t*>(tGpuCodedBits.addr()),
                                                                static_cast<uint8_t*>(tGpuTxBits.addr()),
                                                                procModeBmsk,
                                                                cuStream);
            if(CUPHY_STATUS_SUCCESS != polarEncStat) throw cuphy::cuphy_exception(polarEncStat);
        }

        CUDA_CHECK(cudaEventRecord(eStop, cuStream));
        CUDA_CHECK(cudaEventSynchronize(eStop));

        cudaStreamSynchronize(cuStream);

        TimePoint stopTime = Clock::now();

        if(enNvprof) cudaProfilerStop();

        //------------------------------------------------------------------
        // Display execution times
        float elapsedMs = 0.0f;
        cudaEventElapsedTime(&elapsedMs, eStart, eStop);

        printf("Execution time: Polar encoding + Rate matching \n");
        printf("---------------------------------------------------------------\n");
        printf("Average (over %d runs) elapsed time in usec (CUDA event) = %.0f\n",
               nIter,
               elapsedMs * 1000 / nIter);

        duration<float, std::milli> diff = stopTime - startTime;
        printf("Average (over %d runs) elapsed time in usec (wall clock) w/ 1s delay kernel = %.0f\n",
               nIter,
               diff.count() * 1000 / nIter);

        //------------------------------------------------------------------
        // Verify results
        // Coded bits are always a multiple of 32
        tensor_pinned_R_8U tCpuCpyCodedBits(tGpuCodedBits.layout(), cuphy::tensor_flags::align_tight);
        // typed_tensor<CUPHY_BIT, pinned_alloc> tCpuCpyCodedBits(tGpuCodedBits.layout(), cuphy::tensor_flags::align_tight);
#if 1     
        tCpuCpyCodedBits = tGpuCodedBits;
#endif
        tensor_pinned_R_8U tCpuCpyTxBits(tGpuTxBits.layout(), cuphy::tensor_flags::align_tight);
        // typed_tensor<CUPHY_BIT, pinned_alloc> tCpuCpyInfoBits(tGpuInfoBits.layout(), cuphy::tensor_flags::align_tight);
#if 1    
        tCpuCpyTxBits = tGpuTxBits;
#endif
        // Wait for copy to complete
        cudaStreamSynchronize(cuStream);
        cudaDeviceSynchronize(); // Needed becase typed_tensor does not support non-default streams

        // Compare expected vs observed
        printf("nInfoBits: %d nExpectedCodedBits: %d nComputedCodedBits: %d nTxBits: %d\n", nInfoBits, nExpectedCodedBits, nCodedBits, nTxBits);
        // Coded bits
        printf("---------------------------------------------------------------\n");
        printf("Comparing coded bits\n");
        uint32_t nCodedByteErrs = 0;
        uint32_t nCodedBytes    = nExpectedCodedBits / 8; // nExpectedCodedBits is a multiple of 32
        for(int n = 0; n < nCodedBytes; ++n)
        {
            uint32_t expectedCodedByte = tCpuCodedBits({n});
            uint32_t observedCodedByte = tCpuCpyCodedBits({n});
            if(expectedCodedByte != observedCodedByte)
            {
                if(verboseErrsOnly) printf("Error: Byte[%03d] Expected 0x%02x Observed 0x%02x\n", n, tCpuCodedBits({n}), tCpuCpyCodedBits({n}));
                nCodedByteErrs++;
            }
        }

        if(0 == nCodedByteErrs)
        {
            printf("No errors detected in coded bits\n");
        }
        else
        {
            if(! verboseErrsOnly) printf("Errors detected in coded bits\n");
        }        

        // Transmit bits
        printf("---------------------------------------------------------------\n");
        printf("Comparing transmit bits\n");
        uint32_t nTxByteErrs = 0;
        uint32_t nTxBytes    = (nTxBits + 7) / 8; // nTxBits is not a multiple of 8, needs rounding
        for(int n = 0; n < nTxBytes; ++n)
        {
            uint32_t expectedTxByte = tCpuTxBits({n});
            uint32_t observedTxByte = tCpuCpyTxBits({n});
            if(expectedTxByte != observedTxByte)
            {
                if(verboseErrsOnly) printf("Error: Byte[%03d] Expected 0x%02x Observed 0x%02x\n", n, tCpuTxBits({n}), tCpuCpyTxBits({n}));
                nTxByteErrs++;
            }
        }

        if(0 == nTxByteErrs)
        {
            printf("No errors detected in transmit bits\n");
        }
        else
        {
            if(! verboseErrsOnly) printf("Errors detected in transmit bits\n");
        }

        if(verbose)
        {
            // Coded bits
            printf("---------------------------------------------------------------\n");
            printf("Dumping coded bits (formatted as %d bytes)\n", nCodedBytes);
            for(int n = 0; n < nCodedBytes; ++n)
            {
                uint32_t expectedCodedByte = tCpuCodedBits({n});
                uint32_t observedCodedByte = tCpuCpyCodedBits({n});
                printf("Byte[%03d] Expected 0x%02x Observed 0x%02x\n", n, tCpuCodedBits({n}), tCpuCpyCodedBits({n}));
            }

            // Transmit bits
            printf("---------------------------------------------------------------\n");
            printf("Dumping transmit bits (formatted as %d bytes)\n", nTxBytes);
            for(int n = 0; n < nTxBytes; ++n)
            {
                uint32_t expectedTxByte = tCpuTxBits({n});
                uint32_t observedTxByte = tCpuCpyTxBits({n});
                printf("Byte[%03d] Expected 0x%02x Observed 0x%02x\n", n, tCpuTxBits({n}), tCpuCpyTxBits({n}));
            }
        }

        //------------------------------------------------------------------
        // Cleanup
        CUDA_CHECK(cudaEventDestroy(eStart));
        CUDA_CHECK(cudaEventDestroy(eStop));

        cudaStreamSynchronize(cuStream);

        cudaDeviceSynchronize();
        cudaStreamDestroy(cuStream);
        
    }
    catch(std::exception& e)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "EXCEPTION: {}", e.what());
        returnValue = 1;
    }
    catch(...)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "UNKNOWN EXCEPTION");
        returnValue = 2;
    }
    return returnValue;
}

