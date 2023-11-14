/*
 * Copyright (c) 2020-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */


#include <gtest/gtest.h>
#include <iostream>
#include <vector>
#include <cuda_fp16.h>
#include "cuphy.h"
#include "cuphy.hpp"
#include "rm_decoder.hpp"
#include "rm_codebook.h"

// struct TestParams {
//     int E;
//     int K;
//     int NUM_CW;
//     int CW_PER_BLOCK;
// };

// const std::vector<TestParams> CONFIGS = {
//     {32, 11, 64, 1},
//     {32, 10, 64, 1},
//     {32, 4, 64, 1},
//     {32, 3, 64, 1},

//     {24, 11, 64, 1},
//     {24, 3, 64, 1},

//     {20, 11, 64, 1},
//     {20, 3, 64, 1},

//     {32, 11, 128, 2},

//     {64, 11, 128, 2}
// };

// class RmDecTest : public ::testing::TestWithParam<TestParams>
// {
// public:
//     TestParams params = ::testing::TestWithParam<TestParams>::GetParam();
//     static const int TRIALS = 2048;
//     static const int MAX_NUM_CW = 1024;
//     static const int MAX_E = 1536;
//     static const int MAX_N = PUCCH_F2_RM_MAX_N;
//     static const int MAX_TWO_TO_K = 2048;
//     int h_E[MAX_NUM_CW];
//     int h_TWO_TO_K[MAX_NUM_CW];
//     int NUM_CW;
//     int CW_PER_BLOCK;
//     uint32_t h_x_out[MAX_NUM_CW];
//     float h_r_in[MAX_NUM_CW][MAX_E];
//     __half h_r_in_half[MAX_NUM_CW][MAX_E];

//     uint32_t *d_x_out;
//     float **d_r_in;
//     __half **d_r_in_half;
//     float *d_codebook;
//     uint32_t *d_codebook_u32;
//     int *d_E;
//     int *d_TWO_TO_K;

//     cudaStream_t stream;

//     cuphyContext_t ctx;
//     cuphyPucchRmDecoderHndl_t hndl;

//     void SetUp() override
//     {
//         NUM_CW = params.NUM_CW;
//         CW_PER_BLOCK = params.CW_PER_BLOCK;

//         cuphyCreateContext(&ctx,0);

//         CUDA_CHECK(cudaHostAlloc(&d_r_in, sizeof(float*)*NUM_CW, cudaHostAllocPortable | cudaHostAllocMapped));
//         CUDA_CHECK(cudaHostAlloc(&d_r_in_half, sizeof(__half*)*NUM_CW, cudaHostAllocPortable | cudaHostAllocMapped));
//         for (int k=0; k<NUM_CW; k++)
//         {
//             h_E[k] = params.E; // TODO add a test with a non-uniform distribution of E and K
//             h_TWO_TO_K[k] = 1 << params.K;
//             CUDA_CHECK(cudaMalloc(&d_r_in[k], sizeof(float)*params.E));
//             CUDA_CHECK(cudaMalloc(&d_r_in_half[k], sizeof(__half)*params.E));
//         }

//         CUDA_CHECK(cudaMalloc(&d_x_out, sizeof(uint32_t)*NUM_CW));
//         CUDA_CHECK(cudaMalloc(&d_codebook, sizeof(float)*MAX_TWO_TO_K*MAX_N));
//         CUDA_CHECK(cudaMalloc(&d_codebook_u32, sizeof(uint32_t)*MAX_TWO_TO_K));
//         CUDA_CHECK(cudaMalloc(&d_E, sizeof(int)*NUM_CW));
//         CUDA_CHECK(cudaMalloc(&d_TWO_TO_K, sizeof(int)*NUM_CW));
//         CUDA_CHECK(cudaStreamCreate(&stream));

//         CUDA_CHECK(cudaMemcpy(d_E, h_E, sizeof(int)*NUM_CW, cudaMemcpyHostToDevice));
//         CUDA_CHECK(cudaMemcpy(d_TWO_TO_K, h_TWO_TO_K, sizeof(int)*NUM_CW, cudaMemcpyHostToDevice));
//     }

//     void TearDown() override
//     {
//         cudaStreamDestroy(stream);
//         cudaFree(d_TWO_TO_K);
//         cudaFree(d_E);
//         cudaFree(d_codebook_u32);
//         cudaFree(d_codebook);
//         cudaFree(d_x_out);

//         for (int k=0; k<NUM_CW; k++)
//         {
//             cudaFree(d_r_in_half[k]);
//             cudaFree(d_r_in[k]);
//         }
//         cudaFreeHost(d_r_in_half);
//         cudaFreeHost(d_r_in);
//     }
// };


// TEST_P(RmDecTest, Run_0)
// {
//     cuphyCreatePucchRmDecoder(ctx,&hndl,0,0);
//     CUDA_CHECK(cudaMemcpy(d_codebook, rm_codebook_0, sizeof(float)*MAX_TWO_TO_K*MAX_N, cudaMemcpyHostToDevice));
//     for (int k=0; k<TRIALS; k++)
//     {
//         for (int e=0; e<params.E; e++)
//         {
//             for (int cw=0; cw<NUM_CW; cw++)
//             {
//                 int k_choice  = (k + cw) % h_TWO_TO_K[cw];
//                 h_r_in[cw][e] = rm_codebook_0[k_choice*MAX_N + (e % MAX_N)];
//             }
//         }

//         for (int cw=0; cw<NUM_CW; cw++)
//         {
//             CUDA_CHECK(cudaMemcpyAsync(d_r_in[cw], h_r_in[cw], sizeof(float)*params.E, cudaMemcpyHostToDevice, stream));
//         }

//         CUPHY_CHECK(cuphySetupPucchRmDecoder(hndl, d_x_out, CUPHY_R_32U, reinterpret_cast<const void* const*>(d_r_in), CUPHY_R_32F, NUM_CW, CW_PER_BLOCK, d_E, d_TWO_TO_K, stream, nullptr));
//         CUPHY_CHECK(cuphyRunPucchRmDecoder(hndl, stream));

//         CUDA_CHECK(cudaMemcpyAsync(h_x_out,d_x_out, sizeof(uint32_t)*NUM_CW, cudaMemcpyDeviceToHost, stream));
//         CUDA_CHECK(cudaStreamSynchronize(stream));
//         //printf("Decoded %d to %d\n",k,h_x_out[0]);
//         for (int cw=0; cw<NUM_CW; cw++)
//         {
//             int k_choice = (k + cw) % h_TWO_TO_K[cw];
//             ASSERT_EQ(k_choice, h_x_out[cw]) << "for codeword " << cw;
//         }
//     }
// }

// TEST_P(RmDecTest, Run_1)
// {
//     cuphyCreatePucchRmDecoder(ctx,&hndl,1,0);
//     CUDA_CHECK(cudaMemcpy(d_codebook, rm_codebook_1, sizeof(float)*MAX_TWO_TO_K*MAX_N, cudaMemcpyHostToDevice));
//     for (int k=0; k<TRIALS; k++)
//     {
//         for (int e=0; e<params.E; e++)
//         {
//             for (int cw=0; cw<NUM_CW; cw++)
//             {
//                 int k_choice = (k + cw) % h_TWO_TO_K[cw];
//                 h_r_in[cw][e] = rm_codebook_1[(e % MAX_N)*MAX_TWO_TO_K + k_choice];
//             }
//         }

//         for (int cw=0; cw<NUM_CW; cw++)
//         {
//             CUDA_CHECK(cudaMemcpyAsync(d_r_in[cw], h_r_in[cw], sizeof(float)*params.E, cudaMemcpyHostToDevice, stream));
//         }

//         CUPHY_CHECK(cuphySetupPucchRmDecoder(hndl, d_x_out, CUPHY_R_32U, reinterpret_cast<const void* const*>(d_r_in), CUPHY_R_32F, NUM_CW, CW_PER_BLOCK, d_E, d_TWO_TO_K, stream, nullptr));
//         CUPHY_CHECK(cuphyRunPucchRmDecoder(hndl, stream));
//         CUDA_CHECK(cudaMemcpyAsync(h_x_out,d_x_out, sizeof(uint32_t)*NUM_CW, cudaMemcpyDeviceToHost, stream));
//         CUDA_CHECK(cudaStreamSynchronize(stream));
//         //printf("Decoded %d to %d\n",k,h_x_out[0]);
//         for (int cw=0; cw<NUM_CW; cw++)
//         {
//             int k_choice = (k + cw) % h_TWO_TO_K[cw];
//             ASSERT_EQ(k_choice, h_x_out[cw]) << "for codeword " << cw;
//         }
//     }
// }

// TEST_P(RmDecTest, Run_2)
// {
//     cuphyCreatePucchRmDecoder(ctx,&hndl,2,0);
//     CUDA_CHECK(cudaMemcpy(d_codebook, rm_codebook_1, sizeof(float)*MAX_TWO_TO_K*MAX_N, cudaMemcpyHostToDevice));
//     for (int k=0; k<TRIALS; k++)
//     {
//         for (int e=0; e<params.E; e++)
//         {
//             for (int cw=0; cw<NUM_CW; cw++)
//             {
//                 int k_choice = (k + cw) % h_TWO_TO_K[cw];
//                 h_r_in[cw][e] = rm_codebook_1[(e % MAX_N)*MAX_TWO_TO_K + k_choice];
//             }
//         }

//         for (int cw=0; cw<NUM_CW; cw++)
//         {
//             CUDA_CHECK(cudaMemcpyAsync(d_r_in[cw], h_r_in[cw], sizeof(float)*params.E, cudaMemcpyHostToDevice, stream));
//         }

//         CUPHY_CHECK(cuphySetupPucchRmDecoder(hndl, d_x_out, CUPHY_R_32U, reinterpret_cast<const void* const*>(d_r_in), CUPHY_R_32F, NUM_CW, CW_PER_BLOCK, d_E, d_TWO_TO_K, stream, nullptr));
//         CUPHY_CHECK(cuphyRunPucchRmDecoder(hndl, stream));
//         CUDA_CHECK(cudaMemcpyAsync(h_x_out,d_x_out, sizeof(uint32_t)*NUM_CW, cudaMemcpyDeviceToHost, stream));
//         CUDA_CHECK(cudaStreamSynchronize(stream));
//         //printf("Decoded %d to %d\n",k,h_x_out[0]);
//         for (int cw=0; cw<NUM_CW; cw++)
//         {
//             int k_choice = (k + cw) % h_TWO_TO_K[cw];
//             ASSERT_EQ(k_choice, h_x_out[cw]) << "for codeword " << cw;
//         }
//     }
// }

// TEST_P(RmDecTest, Run_3float)
// {
//     cuphyCreatePucchRmDecoder(ctx,&hndl,3,0);
//     CUDA_CHECK(cudaMemcpy(d_codebook_u32, rm_codebook_3, sizeof(uint32_t)*MAX_TWO_TO_K, cudaMemcpyHostToDevice));
//     for (int k=0; k<TRIALS; k++)
//     {
//         for (int e=0; e<params.E; e++)
//         {
//             for (int cw=0; cw<NUM_CW; cw++)
//             {
//                 int k_choice = (k + cw) % h_TWO_TO_K[cw];
//                 h_r_in[cw][e] = rm_codebook_1[(e % MAX_N)*MAX_TWO_TO_K + k_choice];
//             }
//         }

//         for (int cw=0; cw<NUM_CW; cw++)
//         {
//             CUDA_CHECK(cudaMemcpyAsync(d_r_in[cw], h_r_in[cw], sizeof(float)*params.E, cudaMemcpyHostToDevice, stream));
//         }

//         CUPHY_CHECK(cuphySetupPucchRmDecoder(hndl, d_x_out, CUPHY_R_32U, reinterpret_cast<const void* const*>(d_r_in), CUPHY_R_32F, NUM_CW, CW_PER_BLOCK, d_E, d_TWO_TO_K, stream, nullptr));
//         CUPHY_CHECK(cuphyRunPucchRmDecoder(hndl, stream));
//         CUDA_CHECK(cudaMemcpyAsync(h_x_out,d_x_out, sizeof(uint32_t)*NUM_CW, cudaMemcpyDeviceToHost, stream));
//         CUDA_CHECK(cudaStreamSynchronize(stream));
//         //printf("Decoded %d to %d\n",k,h_x_out[0]);
//         for (int cw=0; cw<NUM_CW; cw++)
//         {
//             int k_choice = (k + cw) % h_TWO_TO_K[cw];
//             ASSERT_EQ(k_choice, h_x_out[cw]) << "for codeword " << cw;
//         }
//     }
// }


// TEST_P(RmDecTest, Run_3half)
// {
//     cuphyCreatePucchRmDecoder(ctx,&hndl,3,0);
//     CUDA_CHECK(cudaMemcpy(d_codebook_u32, rm_codebook_3, sizeof(uint32_t)*MAX_TWO_TO_K, cudaMemcpyHostToDevice));

//     for (int k=0; k<TRIALS; k++)
//     {
//         for (int e=0; e<params.E; e++)
//         {
//             for (int cw=0; cw<NUM_CW; cw++)
//             {
//                 int k_choice = (k + cw) % h_TWO_TO_K[cw];
//                 if ((params.E > MAX_N) && (e < MAX_N))
//                 {
//                     // For E > N, check the de-rate-matching is doing something by zeroing the first N samples
//                     h_r_in_half[cw][e] = __float2half(0.);
//                 }
//                 else
//                 {
//                     h_r_in_half[cw][e] = __float2half(rm_codebook_1[(e % MAX_N)*MAX_TWO_TO_K + k_choice]);
//                 }
//             }
//         }

//         for (int cw=0; cw<NUM_CW; cw++)
//         {
//             CUDA_CHECK(cudaMemcpyAsync(d_r_in_half[cw], h_r_in_half[cw], sizeof(__half)*params.E, cudaMemcpyHostToDevice, stream));
//         }

//         CUPHY_CHECK(cuphySetupPucchRmDecoder(hndl, d_x_out, CUPHY_R_32U, reinterpret_cast<const void* const*>(d_r_in_half), CUPHY_R_16F, NUM_CW, CW_PER_BLOCK, d_E, d_TWO_TO_K, stream, nullptr));
//         CUPHY_CHECK(cuphyRunPucchRmDecoder(hndl, stream));
//         CUDA_CHECK(cudaMemcpyAsync(h_x_out,d_x_out, sizeof(uint32_t)*NUM_CW, cudaMemcpyDeviceToHost, stream));
//         CUDA_CHECK(cudaStreamSynchronize(stream));
//         //printf("Decoded %d to %d\n",k,h_x_out[0]);
//         for (int cw=0; cw<NUM_CW; cw++)
//         {
//             int k_choice = (k + cw) % h_TWO_TO_K[cw];
//             ASSERT_EQ(k_choice, h_x_out[cw]) << "for codeword " << cw << " in trial " << k;
//         }
//     }
// }

// INSTANTIATE_TEST_CASE_P(RmDecTests, RmDecTest,::testing::ValuesIn(CONFIGS));

// int main(int argc, char** argv) {
//     testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
