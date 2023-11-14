/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

// /*
//  * Copyright (c) 2020, NVIDIA CORPORATION.  All rights reserved.
//  *
//  * NVIDIA CORPORATION and its licensors retain all intellectual property
//  * and proprietary rights in and to this software, related documentation
//  * and any modifications thereto.  Any use, reproduction, disclosure or
//  * distribution of this software and related documentation without an express
//  * license agreement from NVIDIA CORPORATION is strictly prohibited.
//  */

// #include <gtest/gtest.h>
// #include <algorithm>
// #include <cstdlib>
// #include <iostream>
// #include <vector>
// #include "cuphy.h"
// #include "cuphy_internal.h"
// #include "descrambling.hpp"
// #include "pusch_rx.hpp"
// #include "cuphy.hpp"

// uint32_t       N_ITER    = 1000;
// const uint32_t Zlist[51] = {2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 18, 20, 22, 24, 26, 28, 30, 32, 36, 40, 44, 48, 52, 56, 60, 64, 72, 80, 88, 96, 104, 112, 120, 128, 144, 160, 176, 192, 208, 224, 240, 256, 288, 320, 352, 384};

// using namespace cuphy_i;
// using namespace descrambling;

// uint32_t factor = 58;
// void     de_rate_matching(float llr_vec_in[], float out[], int inputLen, int outputLen, int nTb, int nBBULayers, const std::vector<uint32_t>& tbSizeArray, const std::vector<uint32_t>& bgArray, const std::vector<uint32_t>& codeBlockSizeArray, const std::vector<uint32_t>& nCodeBlocksArray, const std::vector<uint32_t>& ZcArray, const std::vector<uint32_t>& rvArray, const std::vector<uint32_t>& KArray, const std::vector<uint32_t>& KdArray, const std::vector<uint32_t>& FArray, const std::vector<uint32_t>& nUserLayersArray, const std::vector<uint32_t>& userLayerMapArray, const std::vector<uint32_t>& QmArray, const std::vector<uint32_t>& CArray)
// {
//     std::vector<uint32_t> k0Array(nTb);
//     std::vector<uint32_t> QmSumArray(nBBULayers + 1);
//     std::vector<uint32_t> tbStartArray(nTb + 1);
//     std::vector<uint32_t> BBULayerMap(nBBULayers);

//     for(int i = 0; i < nTb; i++)
//     {
//         for(int j = 0; j < nBBULayers; j++)
//         {
//             if(userLayerMapArray[i * nBBULayers + j] == j)
//             {
//                 BBULayerMap[j] = i;
//             }
//         }
//     }

//     // Compute max number of code blocks in transport block
//     uint32_t CMax = 0;

//     for(int i = 0; i < nTb; i++)
//         CMax = CMax > nCodeBlocksArray[i] ? CMax : nCodeBlocksArray[i];

//     std::vector<uint32_t> codeBlockQAMStartIndex(CMax * nTb);
//     std::vector<uint32_t> E_vec(CMax * nTb);

//     QmSumArray[0]   = 0;
//     tbStartArray[0] = 0;
//     for(int i = 1; i < nTb + 1; i++)
//     {
//         tbStartArray[i] = tbStartArray[i - 1] + codeBlockSizeArray[i - 1] * nCodeBlocksArray[i - 1];
//     }

//     for(int i = 1; i < nBBULayers + 1; i++)
//     {
//         QmSumArray[i] = QmSumArray[i - 1] + QmArray[BBULayerMap[i - 1]];
//     }

//     uint32_t EMax = 0;

//     // Determine rate matched block size E and start index codeBlockQAMStartIndex
//     for(uint32_t i = 0; i < nTb; i++)
//     {
//         uint32_t E;
//         codeBlockQAMStartIndex[i * CMax] = 0;
//         for(uint32_t r = 0; r < nCodeBlocksArray[i]; r++)
//         {
//             uint32_t C = nCodeBlocksArray[i];
//             // Determine block size E
//             if(r <= C - (tbSizeArray[i] / (nUserLayersArray[i] * QmArray[i])) % C - 1)
//             {
//                 E = nUserLayersArray[i] * QmArray[i] * floorf(float(tbSizeArray[i]) / float(nUserLayersArray[i] * QmArray[i] * C));
//             }
//             else
//             {
//                 E = nUserLayersArray[i] * QmArray[i] * ceilf(float(tbSizeArray[i]) / float(nUserLayersArray[i] * QmArray[i] * C));
//             }
//             if(r < C - 1) codeBlockQAMStartIndex[i * CMax + r + 1] = codeBlockQAMStartIndex[i * CMax + r] + E;
//             E_vec[i * CMax + r] = E;
//             if(E > EMax)
//                 EMax = E;
//         }
//     }
//     for(int i = 0; i < nTb; i++)
//     {
//         // Determine k0[i] based on rv and bg
//         if(bgArray[i] == 1)
//         {
//             if(rvArray[i] == 0)
//             {
//                 k0Array[i] = 0;
//             }
//             else if(rvArray[i] == 1)
//             {
//                 k0Array[i] = floorf(17 * codeBlockSizeArray[i] / (66 * ZcArray[i])) * ZcArray[i];
//             }
//             else if(rvArray[i] == 2)
//             {
//                 k0Array[i] = floorf(33 * codeBlockSizeArray[i] / (66 * ZcArray[i])) * ZcArray[i];
//             }
//             else if(rvArray[i] == 3)
//             {
//                 k0Array[i] = floorf(56 * codeBlockSizeArray[i] / (66 * ZcArray[i])) * ZcArray[i];
//             }
//         }
//         else if(bgArray[i] == 2)
//         {
//             if(rvArray[i] == 0)
//             {
//                 k0Array[i] = 0;
//             }
//             else if(rvArray[i] == 1)
//             {
//                 k0Array[i] = floorf(13 * codeBlockSizeArray[i] / (50 * ZcArray[i])) * ZcArray[i];
//             }
//             else if(rvArray[i] == 2)
//             {
//                 k0Array[i] = floorf(25 * codeBlockSizeArray[i] / (50 * ZcArray[i])) * ZcArray[i];
//             }
//             else if(rvArray[i] == 3)
//             {
//                 k0Array[i] = floorf(43 * codeBlockSizeArray[i] / (50 * ZcArray[i])) * ZcArray[i];
//             }
//         }
//     }

//     // Assuming no limit soft buffer size
//     int Ncb = codeBlockSizeArray[0];

//     int      startIdx = codeBlockQAMStartIndex[0];
//     int      C        = nCodeBlocksArray[0];
//     int      Qm       = QmArray[0];
//     int      Nl       = nUserLayersArray[0];
//     uint32_t E        = 0;
//     uint32_t k0       = k0Array[0];
//     uint32_t Zc       = ZcArray[0];
//     uint32_t Kd       = KdArray[0];
//     uint32_t K        = KArray[0] - 2 * Zc;
//     for(int r = 0; r < C; r++)
//     {
//         // Determine starting point
//         startIdx += E;
//         //std::cout << "\n" << "startidx: " << startIdx << "\n";

//         // Determine E
//         if(r <= C - (inputLen / (Nl * Qm)) % C - 1)
//         {
//             E = Nl * Qm * floorf(float(inputLen) / float(Nl * Qm * C));
//         }
//         else
//         {
//             E = Nl * Qm * ceilf(float(inputLen) / float(Nl * Qm * C));
//         }
//         //printf("E is %d %d", E, r);
//         float in[25344];

//         // De-interleave
//         for(int kk = 0; kk < Qm; kk++)
//         {
//             for(int k = 0; k < E / Qm; k++)
//             {
//                 in[k + kk * E / Qm] = llr_vec_in[startIdx + kk + (k * Qm)];
//             }
//         }
//         //for (int i=0;i<10;i++){
//         //      std::cout << in[i] << ", ";
//         //}

//         // Expand code block
//         int k = 0;
//         int j = 0;
//         int idx;
//         int indices[25344];

//         // Find indices for bits
//         while(k < E)
//         {
//             idx = k0 + j % Ncb;
//             if(!((idx >= Kd) && (idx < K)))
//             {
//                 indices[k] = idx;
//                 k          = k + 1;
//             }
//             j = j + 1;
//         }

//         // Write output
//         for(int n = 0; n < E; n++)
//         {
//             out[2 * Zc + r * Ncb + indices[n]] = in[n];
//         }

//         // Write filler bits
//         // Write output
//         // Write filler bits
//         for(int n = Kd; n < K; n++)
//         {
//             out[2 * Zc + r * Ncb + n] = 10000.0;
//         }
//     }
// }

// uint32_t reverse(uint32_t x, int bits)
// {
//     x = ((x & 0x55555555) << 1) | ((x & 0xAAAAAAAA) >> 1);   // Swap _<>_
//     x = ((x & 0x33333333) << 2) | ((x & 0xCCCCCCCC) >> 2);   // Swap __<>__
//     x = ((x & 0x0F0F0F0F) << 4) | ((x & 0xF0F0F0F0) >> 4);   // Swap ____<>____
//     x = ((x & 0x00FF00FF) << 8) | ((x & 0xFF00FF00) >> 8);   // Swap ...
//     x = ((x & 0x0000FFFF) << 16) | ((x & 0xFFFF0000) >> 16); // Swap ...
//     return x >> (32 - bits);
// }

// int testRateMatching()
// {
//     int                            res                  = 1;
//     uint32_t                       nTb                  = 1;
//     uint32_t                       nBBULayers           = 1;
//     uint32_t                       outputSize           = 25344 * 13 * factor * nTb;
//     uint32_t                       inputSize            = 117504 * factor * nTb;
//     uint32_t                       paddedInputSize      = (117504 / 6) * QAM_STRIDE * factor * nTb;
//     uint32_t                       maxScramblingSeqSize = inputSize;
//     bool                           descramblingOn       = false;
//     unique_device_ptr<float>       d_input;
//     unique_device_ptr<PerTbParams> d_tbPrmsArray;
// #ifdef _FP_16_
//     unique_device_ptr<__half> d_output;
// #else
//     unique_device_ptr<float> d_output;
// #endif
//     std::vector<float>    input(inputSize);
//     std::vector<float>    paddedInput(paddedInputSize, 0);
//     std::vector<float>    gpuOutput(outputSize);
//     std::vector<float>    cpuOutput(outputSize);
//     std::vector<uint32_t> cinitArray(nTb);
//     std::vector<uint32_t> cpuSeq(maxScramblingSeqSize);
//     std::vector<uint32_t> tbBoundaryArray(nTb + 1);
//     std::vector<uint32_t> qamArray(nTb);
//     std::vector<uint32_t> bgArray(nTb);                // Base graph type
//     std::vector<uint32_t> inputLenLayerArray(nTb);     // Number of soft bits De-rate-matching input
//     std::vector<uint32_t> inputTBLenArray(nTb);        // Number of soft bits per transport block
//     std::vector<uint32_t> codeBlockSizeArray(nTb);     // Coded code block size
//     std::vector<uint32_t> firstCodeBlockIdxArray(nTb); // Index of first code block to be processed for each transport block
//     std::vector<uint32_t> nCodeBlocksArray(nTb);       // Number of code blocks for each transport block starting from (and including) first
//     std::vector<uint32_t> ZcArray(nTb);
//     std::vector<uint32_t> rvArray(nTb);                        // Redundancy version
//     std::vector<uint32_t> nUserLayersArray(nTb);               // Number of layers per user
//     std::vector<uint32_t> userLayerMapArray(nTb * nBBULayers); // Layer map
//     std::vector<uint32_t> KArray(nTb);                         // Number of non-punctured systematic bits
//     std::vector<uint32_t> FArray(nTb);                         // Number of filler bits
//     std::vector<uint32_t> KdArray(nTb);

//     std::vector<PerTbParams> tbPrmsArray(nTb);

//     for(int i = 0; i < inputSize; i++)
//         input[i] = i + 1;

//     for(int i = 0; i < nTb; i++)
//     {
//         inputLenLayerArray[i]              = 117504 * factor;
//         tbPrmsArray[i].encodedSize         = inputLenLayerArray[i];
//         codeBlockSizeArray[i]              = 25344;
//         tbPrmsArray[i].Ncb                 = codeBlockSizeArray[i];
//         firstCodeBlockIdxArray[i]          = 0;
//         tbPrmsArray[i].firstCodeBlockIndex = firstCodeBlockIdxArray[i];
//         nCodeBlocksArray[i]                = 13 * factor;
//         tbPrmsArray[i].num_CBs             = nCodeBlocksArray[i];
//         tbPrmsArray[i].nBBULayers          = nBBULayers;
//         qamArray[i]                        = 6;
//         tbPrmsArray[i].Qm                  = qamArray[i];
//         ZcArray[i]                         = 384;
//         tbPrmsArray[i].Zc                  = ZcArray[i];
//         FArray[i]                          = 72;
//         tbPrmsArray[i].F                   = FArray[i];
//         rvArray[i]                         = 0;
//         tbPrmsArray[i].rv                  = rvArray[i];
//         bgArray[i]                         = 1;
//         tbPrmsArray[i].bg                  = bgArray[i];
//         nUserLayersArray[i]                = 1;
//         tbPrmsArray[i].Nl                  = nUserLayersArray[i];
//         userLayerMapArray[i]               = i;
//         tbPrmsArray[i].layer_map_array[0]  = i;
//         KArray[i]                          = 8448;
//         tbPrmsArray[i].K                   = KArray[i];
//         KdArray[i]                         = KArray[i] - 2 * ZcArray[i] - FArray[i];
//         inputTBLenArray[i]                 = inputLenLayerArray[i] * nUserLayersArray[i];
//         tbPrmsArray[i].encodedSize         = inputTBLenArray[i];
//     }

//     tbBoundaryArray[0] = 0;
//     for(int i = 1; i <= nTb; i++)
//         tbBoundaryArray[i] = tbBoundaryArray[i - 1] + inputTBLenArray[i - 1];

//     for(int i = 0; i < nTb; i++) cinitArray[i] = 1507348;

//     // generate CPU scrambling sequence

//     std::vector<uint32_t> x1(NC + maxScramblingSeqSize + 31);
//     std::vector<uint32_t> x2(NC + maxScramblingSeqSize + 31);

//     for(int i = 0; i < nTb; i++)
//     {
//         for(int n = 0; n < 31; n++)
//         {
//             x2[n] = (cinitArray[i] >> n) & 0x1;
//         }

//         x1[0] = 1;

//         for(int j = 0; j < NC + tbBoundaryArray[i + 1] - tbBoundaryArray[i]; j++)
//         {
//             x1[j + 31] = (x1[j + 3] + x1[j]) & 0x1;

//             x2[j + 31] = (x2[j + 3] + x2[j + 2] + +x2[j + 1] + x2[j]) & 0x1;
//         }
//         /*
//      for (n = 0; n < len; n++) printf("%d, ", x1[n]);
//     */
//         for(int j = 0; j < tbBoundaryArray[i + 1] - tbBoundaryArray[i]; j++)
//             cpuSeq[tbBoundaryArray[i] + j] = (x1[j + NC] + x2[j + NC]) & 0x1;
//     }

//     int p = 0;
//     for(int j = 0; j < inputSize; j++)
//     {
//         paddedInput[p] = input[j];
//         p++;
//         if((((j + 1) % qamArray[0]) == 0))
//             p += QAM_STRIDE - qamArray[0];
//     }

//     d_input       = make_unique_device<float>(paddedInputSize);
//     d_output      = make_unique_device<float>(outputSize);
//     d_tbPrmsArray = make_unique_device<PerTbParams>(nTb);

//     cudaMemcpy(d_input.get(), paddedInput.data(), paddedInputSize * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_tbPrmsArray.get(), tbPrmsArray.data(), nTb * sizeof(PerTbParams), cudaMemcpyHostToDevice);

//     cuphy::unique_device_ptr<const void*> d_inputs = cuphy::make_unique_device<const void*>(1);

//     const void* inputPtrs[1];
//     inputPtrs[0] = (const void*)d_input.get();
//     cudaMemcpy(d_inputs.get(), inputPtrs, sizeof(void*), cudaMemcpyHostToDevice);

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);

//     uint32_t EMax = nUserLayersArray[0] * qamArray[0] * ceilf(float(inputTBLenArray[0]) / float(nUserLayersArray[0] * qamArray[0] * nCodeBlocksArray[0]));

//     rate_matching(
//         nCodeBlocksArray[0],
//         EMax,
//         nTb,
//         d_tbPrmsArray.get(),
//         d_inputs.get(),
//         d_output.get(),
//         0,
//         descramblingOn,
//         0);

//     float time1 = 0.0;
//     cudaEventRecord(start);

//     for(int i = 0; i < N_ITER; i++)
//     {
// #ifdef _FP_16_
//         rate_matching(
//             nCodeBlocksArray[0],
//             EMax,
//             nTb,
//             d_tbPrmsArray.get(),
//             d_inputs.get(),
//             d_output.get(),
//             2,
//             descramblingOn,
//             0);
// #else
//         rate_matching(
//             nCodeBlocksArray[0],
//             EMax,
//             nTb,
//             d_tbPrmsArray.get(),
//             d_inputs.get(),
//             d_output.get(),
//             0,
//             descramblingOn,
//             0);
// #endif
//     }

//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&time1, start, stop);

//     time1 /= N_ITER;

//     printf(
//         "Rate Matching Kernel"
//         "\n %.2f us\n",
//         time1 * 1000);

//     cudaMemcpy(gpuOutput.data(), d_output.get(), outputSize * sizeof(float), cudaMemcpyDeviceToHost);

//     // CPU computation
//     de_rate_matching(input.data(), cpuOutput.data(), inputSize, outputSize, nTb, nBBULayers, inputTBLenArray, bgArray, codeBlockSizeArray, nCodeBlocksArray, ZcArray, rvArray, KArray, KdArray, FArray, nUserLayersArray, userLayerMapArray, qamArray, nCodeBlocksArray);

//     if(descramblingOn)
//     {
//         for(int i = 0; i < nTb; i++)
//         {
//             for(int j = 0; j < tbBoundaryArray[i + 1] - tbBoundaryArray[i]; j++)
//             {
//                 if(cpuSeq[tbBoundaryArray[i] + j])
//                 {
//                     cpuOutput[tbBoundaryArray[i] + j] = -input[tbBoundaryArray[i] + j];
//                 }
//             }
//         }
//     }
//     res = 1;
//     for(int i = 0; i < outputSize; i++)
//     {
//         if(gpuOutput[i] != cpuOutput[i])
//         {
//             std::cout << "Error: not equal at " << i << " GPU: " << gpuOutput[i] << " CPU: " << cpuOutput[i] << "\n";
//             res = 0;
//         }
//     }
//     if(!res)
//     {
//         for(int i = 0; i < inputSize; i++)
//         {
//             std::cout << "Input at " << i << ": " << input[i] << "\n";
//         }
//     }
//     return res;
// }

// int RATE_MATCHING_TEST()
// {
//     int res = testRateMatching();

//     return res;
// }

// TEST(RATE_MATCHING, SU_SISO_1) { EXPECT_EQ(RATE_MATCHING_TEST(), 1); }
// int main(int argc, char** argv)
// {
//     if(argc > 1)
//         cudaSetDevice(std::stoi(argv[1]));
//     if(argc > 2)
//         factor = std::stoi(argv[2]);
//     if(argc > 3)
//         N_ITER = std::stoi(argv[3]);

//     testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
