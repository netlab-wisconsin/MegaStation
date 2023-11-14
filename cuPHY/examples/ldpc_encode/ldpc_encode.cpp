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
#include <iostream>
#include <vector>
#include <getopt.h>
#include "cuphy.hpp"
#include "cuphy_hdf5.hpp"
#include "hdf5hpp.hpp"

using namespace cuphy;

uint32_t N_ITER = 200;

void usage(char* prog)
{
    printf("%s [options] [test-vector-file]\n", prog);
    printf("  Options:\n");
    printf("    -g base_graph          Base graph (default: 1)\n");
    printf("    -p                     enable puncuture (default: no)\n");
}

int main(int argc, char* argv[])
{
    std::string  inputFilename = "mat_gen_data.hdf5";
    extern char* optarg;
    extern int   optind;
    int          option, bg = 1;
    bool         puncture = false;
    std::string  outFileName;
    bool         debugOut = false;
    
    cuphyNvlogFmtHelper nvlog_fmt("ldpc_encode.log");
    
    while((option = getopt(argc, argv, "pgo:")) != -1)
    {
        switch(option)
        {
        case 'p':
            puncture = true;
            break;
        case 'g':
            bg = atoi(optarg);
            if(bg < 1 || bg > 2)
            {
                NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "ERROR: Invalid basegraph number {}", bg);
                usage(argv[0]);
                return (EINVAL);
            }
            break;
        case 'o':
            outFileName = optarg;
            debugOut    = true;
            break;

        default:
            NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "ERROR: Invalid option: {}", argv[optind]);
            usage(argv[0]);
            return (EINVAL);
        }
    }

    if(optind < argc)
    {
        inputFilename.assign(argv[optind]);
    }

    hdf5hpp::hdf5_file fInput = hdf5hpp::hdf5_file::open(inputFilename.c_str());

    using tensor_pinned_R_8U  = typed_tensor<CUPHY_R_8U, pinned_alloc>;
    using tensor_pinned_R_32U = typed_tensor<CUPHY_R_32U, pinned_alloc>;

    tensor_pinned_R_8U  tSourceData = typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(fInput.open_dataset("TbCbsUncoded"));
    tensor_pinned_R_8U  tBG         = typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(fInput.open_dataset("BG"));
    tensor_pinned_R_32U tK          = typed_tensor_from_dataset<CUPHY_R_32U, pinned_alloc>(fInput.open_dataset("K"));
    //   tensor_pinned_R_32U tKb            = typed_tensor_from_dataset<CUPHY_R_32U, pinned_alloc>(fInput.open_dataset("K_b"));
    tensor_pinned_R_8U maxParityNodes = typed_tensor_from_dataset<CUPHY_R_8U, pinned_alloc>(fInput.open_dataset("nV_parity"));
    tensor_device      d_EncodedData;
    try
    {
        cuphy::disable_hdf5_error_print();
        d_EncodedData = tensor_from_dataset(fInput.open_dataset("TbCbsCoded"));
    }
    catch(std::exception& e)
    {
        // test vectors may be using 'inputCodeWord'
        d_EncodedData = tensor_from_dataset(fInput.open_dataset("inputCodeWord"));
    }
    tensor_pinned_R_8U tEncodedData(d_EncodedData.layout());
    cuphyStatus_t      s = cuphyConvertTensor(tEncodedData.desc().handle(),
                                         tEncodedData.addr(),
                                         d_EncodedData.desc().handle(),
                                         d_EncodedData.addr(),
                                         0);

    const int K  = tK(0, 0);
    const int C  = tSourceData.dimensions()[1];
    int       BG = tBG(0, 0);
    int       ncwnodes, Kb, Z;

    Kb = CUPHY_LDPC_BG1_INFO_NODES;

    if(BG == 1)
    {
        ncwnodes = CUPHY_LDPC_MAX_BG1_UNPUNCTURED_VAR_NODES;
        Z        = K / Kb;
    }
    else
    {
        Kb       = CUPHY_LDPC_MAX_BG2_INFO_NODES;
        Z        = K / 10;
        ncwnodes = CUPHY_LDPC_MAX_BG2_UNPUNCTURED_VAR_NODES;
    }

    const int N = Z * (ncwnodes + (puncture ? 0 : 2));
    NVLOGI_FMT(NVLOG_PDSCH, "{} --- Kb : {} K: {} C: {} BG: {} Z: {} N: {}", inputFilename.c_str(), Kb, K, C, BG, Z, N);

    int N_padding = (N % 32 != 0) * (32 - (N % 32));
    if(N != tEncodedData.dimensions()[0])
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR: the wrongly structured reference output: {} vs. {}", N, tEncodedData.dimensions()[0]);
        exit(1);
    }

    int           K_padding = (K % 32 != 0) * (32 - (K % 32));
    tensor_device d_in_tensor(CUPHY_BIT, K + K_padding, C);

    buffer<uint32_t, pinned_alloc> h_in_tensor(d_in_tensor.desc().get_size_in_bytes());

    for(int c = 0; c < C; c++)
    {
        for(int k = 0; k < K; k += 32)
        {
            uint32_t bits = 0;
            for(int o = 0; o < 32; o++)
            {
                if(k + o < K)
                {
                    uint32_t bit = tSourceData(k + o, c) & 0x1;
                    bits |= (bit << o);
                }
            }
            uint32_t* word = h_in_tensor.addr() + (k / 32) + ((K + K_padding) / 32) * c;
            *word          = bits;
        }
    }
    CUDA_CHECK(cudaMemcpy(d_in_tensor.addr(), h_in_tensor.addr(), d_in_tensor.desc().get_size_in_bytes(), cudaMemcpyHostToDevice));

    int           rv               = 0;                    //redundancy version
    int           max_parity_nodes = maxParityNodes(0, 0); // treated as unknown when 0
    tensor_device d_out_tensor(CUPHY_BIT, N + N_padding, C);

    // Allocate launch config struct.
    std::unique_ptr<cuphyLDPCEncodeLaunchConfig> ldpc_hndl = std::make_unique<cuphyLDPCEncodeLaunchConfig>();

    // Allocate descriptors and setup LDPC encoder
    uint8_t desc_async_copy = 1; // Copy descriptor to the GPU during setup
    size_t  workspace_size = 0;
    int     max_UEs    = PDSCH_MAX_UES_PER_CELL_GROUP;

    size_t        desc_size = 0, alloc_size = 0;
    cuphyStatus_t status = cuphyLDPCEncodeGetDescrInfo(&desc_size, &alloc_size, max_UEs, &workspace_size);
    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "cuphyLDPCEncodeGetDescrInfo error {}", status);
    }
    cuphy::unique_device_ptr<uint8_t> d_ldpc_desc = make_unique_device<uint8_t>(desc_size);
    cuphy::unique_pinned_ptr<uint8_t> h_ldpc_desc = make_unique_pinned<uint8_t>(desc_size);

    cuphy::unique_device_ptr<uint8_t> d_workspace = make_unique_device<uint8_t>(workspace_size);
    cuphy::unique_pinned_ptr<uint8_t> h_workspace = make_unique_pinned<uint8_t>(workspace_size);

    cudaStream_t cuda_strm = 0;
    status                 = cuphySetupLDPCEncode(ldpc_hndl.get(),
                                  d_in_tensor.desc().handle(),
                                  d_in_tensor.addr(),
                                  d_out_tensor.desc().handle(),
                                  d_out_tensor.addr(),
                                  BG,
                                  Z,
                                  puncture,
                                  max_parity_nodes,
                                  rv,
                                  0, 1, nullptr, nullptr,
                                  h_workspace.get(),
                                  d_workspace.get(),
                                  h_ldpc_desc.get(),
                                  d_ldpc_desc.get(),
                                  desc_async_copy,
                                  cuda_strm);

    if(status != CUPHY_STATUS_SUCCESS)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Invalid argument(s) for cuphySetupLDPCEncode");
        exit(1);
    }

    printf("LDPC Encoder\n");
    CUresult r = launch_kernel(ldpc_hndl.get()->m_kernelNodeParams, cuda_strm);

    if(r != CUDA_SUCCESS)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "Invalid argument for LDPC kernel launch");
        exit(1);
    }
    CUDA_CHECK(cudaDeviceSynchronize());

    typed_tensor<CUPHY_BIT, pinned_alloc> h_out_tensor(d_out_tensor.layout());
    h_out_tensor = d_out_tensor;

    int  k_error_word_count = 0;
    int  n_error_word_count = 0;
    bool fail               = false;

    for(int c = 0; c < C; c++)
    {
        for(int n = 0; n < N; n += 32)
        {
            uint32_t bits = 0;
            for(int o = 0; o < 32; o++)
            {
                if(n + o < N)
                {
                    uint32_t bit = tEncodedData(n + o, c) & 0x1;
                    bits |= (bit << o);
                }
            }
            uint32_t val = h_out_tensor(n / 32, c);

            if(val != bits)
            {
                if(!fail)
                    NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR");
                fail = true;
                NVLOGI_FMT(NVLOG_PDSCH, "MISMATCH: ({}) TV:{} vs. GPU:{} in ({}, {})", bits == val, bits, val, c, n);
                if(n < K)
                    k_error_word_count++;
                else
                    n_error_word_count++;
            }
        }
    }
    if(!fail)
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT,  "ERROR");
    NVLOGI_FMT(NVLOG_PDSCH, "k_error_word_count: {} n_error_word_count: {}", k_error_word_count, n_error_word_count);

    if(debugOut)

    {
        hdf5hpp::hdf5_file fOutput = hdf5hpp::hdf5_file::create(outFileName.c_str());
        cuphy::write_HDF5_dataset(fOutput, h_out_tensor, "EncoderOut", 0);
    }

    if(k_error_word_count || n_error_word_count)
    {
        exit(1);
    }

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

    float time = 0.0f;
    cudaEventRecord(start);

    for(int i = 0; i < N_ITER; i++)
    {
        launch_kernel(ldpc_hndl.get()->m_kernelNodeParams, 0);
    }

    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    time /= N_ITER;
    NVLOGI_FMT(NVLOG_PDSCH, "LDPC Encoder: {} us", time * 1000);

    cudaError_t err = cudaGetLastError();
    if(err != cudaSuccess)
    {
        NVLOGE_FMT(NVLOG_PDSCH, AERIAL_CUPHY_EVENT, "ERROR: {}", cudaGetErrorString(err));
        exit(1);
    }
    NVLOGI_FMT(NVLOG_PDSCH, "N: {} outTensor dims: {} {}", N, d_out_tensor.dimensions()[0], d_out_tensor.dimensions()[1]);

    return 0;
}
