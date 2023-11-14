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

// using namespace cuphy_i;
// using namespace descrambling;

// unsigned int N_ITER = 10000;

// int rmTest(const std::string& tv)
// {
//     // read test vectors

//     std::vector<tb_pars>  tbPrmsArray;
//     gnb_pars              gnbPrms;
//     std::vector<float>    inputLLRs;
//     std::vector<float>    outputLLRs;
//     std::vector<float>    outputGPULLRs;
//     PuschRx::ConfigParams tvPrms;
//     try
//     {
//         hdf5hpp::hdf5_file fInput(hdf5hpp::hdf5_file::open(tv.c_str()));

//         hdf5hpp::hdf5_dataset iLLRs = fInput.open_dataset(std::string("reference_LLR_demap").c_str());
//         inputLLRs.resize(iLLRs.get_buffer_size_bytes() / sizeof(float));
//         iLLRs.read(static_cast<void*>(inputLLRs.data()));

//         hdf5hpp::hdf5_dataset oLLRs = fInput.open_dataset(std::string("reference_derateCbs").c_str());
//         outputLLRs.resize(oLLRs.get_buffer_size_bytes() / sizeof(float));
//         outputGPULLRs.resize(outputLLRs.size());
//         oLLRs.read(static_cast<void*>(outputLLRs.data()));

//         cuphy::cuphyHDF5_struct gnbConfig = cuphy::get_HDF5_struct(fInput, std::string("gnb_pars").c_str());

//         int slotNumber               = gnbConfig.get_value_as<uint32_t>("slotNumber");
//         gnbPrms.fc                   = gnbConfig.get_value_as<uint32_t>("fc");
//         gnbPrms.mu                   = gnbConfig.get_value_as<uint32_t>("mu");
//         gnbPrms.nRx                  = gnbConfig.get_value_as<uint32_t>("nRx");
//         gnbPrms.nPrb                 = gnbConfig.get_value_as<uint32_t>("nPrb");
//         gnbPrms.cellId               = gnbConfig.get_value_as<uint32_t>("cellId");
//         gnbPrms.slotNumber           = gnbConfig.get_value_as<uint32_t>("slotNumber");
//         gnbPrms.Nf                   = gnbConfig.get_value_as<uint32_t>("Nf");
//         gnbPrms.Nt                   = gnbConfig.get_value_as<uint32_t>("Nt");
//         gnbPrms.df                   = gnbConfig.get_value_as<uint32_t>("df");
//         gnbPrms.dt                   = gnbConfig.get_value_as<uint32_t>("dt");
//         gnbPrms.numBsAnt             = gnbConfig.get_value_as<uint32_t>("numBsAnt");
//         gnbPrms.numBbuLayers         = gnbConfig.get_value_as<uint32_t>("numBbuLayers");
//         gnbPrms.numTb                = gnbConfig.get_value_as<uint32_t>("numTb");
//         gnbPrms.ldpcnIterations      = gnbConfig.get_value_as<uint32_t>("ldpcnIterations");
//         gnbPrms.ldpcEarlyTermination = gnbConfig.get_value_as<uint32_t>("ldpcEarlyTermination");
//         gnbPrms.ldpcAlgoIndex        = gnbConfig.get_value_as<uint32_t>("ldpcAlgoIndex");
//         gnbPrms.ldpcFlags            = gnbConfig.get_value_as<uint32_t>("ldpcFlags");
//         gnbPrms.ldpcUseHalf          = gnbConfig.get_value_as<uint32_t>("ldpcUseHalf");
//         try
//         {
//             gnbPrms.nUserGroups = gnbConfig.get_value_as<uint32_t>("nUserGroups");
//         }
//         catch(...)
//         {
//             gnbPrms.nUserGroups = 1;
//         }

//         tbPrmsArray.resize(gnbPrms.numTb);

//         // parse array of tb_pars structs

//         hdf5hpp::hdf5_dataset tbpDset = fInput.open_dataset(std::string("tb_pars").c_str());
//         for(int j = 0; j < gnbPrms.numTb; j++)
//         {
//             cuphy::cuphyHDF5_struct tbConfig = cuphy::get_HDF5_struct_index(tbpDset, j);
//             tbPrmsArray[j].numLayers         = tbConfig.get_value_as<uint32_t>("numLayers");
//             tbPrmsArray[j].layerMap          = tbConfig.get_value_as<uint32_t>("layerMap");
//             tbPrmsArray[j].startPrb          = tbConfig.get_value_as<uint32_t>("startPrb");
//             tbPrmsArray[j].numPrb            = tbConfig.get_value_as<uint32_t>("numPRb");
//             tbPrmsArray[j].startSym          = tbConfig.get_value_as<uint32_t>("startSym");
//             tbPrmsArray[j].numSym            = tbConfig.get_value_as<uint32_t>("numSym");
//             tbPrmsArray[j].dmrsMaxLength     = tbConfig.get_value_as<uint32_t>("dmrsMaxLength");
//             tbPrmsArray[j].dataScramId       = tbConfig.get_value_as<uint32_t>("dataScramId");
//             tbPrmsArray[j].mcsTableIndex     = tbConfig.get_value_as<uint32_t>("mcsTableIndex");
//             tbPrmsArray[j].mcsIndex          = tbConfig.get_value_as<uint32_t>("mcsIndex");
//             tbPrmsArray[j].rv                = tbConfig.get_value_as<uint32_t>("rv");
//             tbPrmsArray[j].dmrsType          = tbConfig.get_value_as<uint32_t>("dmrsType");
//             tbPrmsArray[j].dmrsAddlPosition  = tbConfig.get_value_as<uint32_t>("dmrsAddlPosition");
//             tbPrmsArray[j].dmrsMaxLength     = tbConfig.get_value_as<uint32_t>("dmrsMaxLength");
//             tbPrmsArray[j].dmrsScramId       = tbConfig.get_value_as<uint32_t>("dmrsScramId");
//             tbPrmsArray[j].dmrsEnergy        = tbConfig.get_value_as<uint32_t>("dmrsEnergy");
//             tbPrmsArray[j].nRnti             = tbConfig.get_value_as<uint32_t>("nRnti");
//             tbPrmsArray[j].dmrsCfg           = tbConfig.get_value_as<uint32_t>("dmrsCfg");

//             tbPrmsArray.resize(gnbPrms.numTb);
//         }

//         PuschRx::expandParameters(tvPrms, tbPrmsArray, gnbPrms);

//         for(int j = 0; j < gnbPrms.numTb; j++)
//         {
//             cuphy::cuphyHDF5_struct tbConfig = cuphy::get_HDF5_struct_index(tbpDset, j);
//             try
//             {
//                 tvPrms.cmnPrms.tbPrmsArray[j].nBBULayers = tbConfig.get_value_as<uint32_t>("nBBULayers");
//             }
//             catch(...)
//             {
//                 tvPrms.cmnPrms.tbPrmsArray[j].nBBULayers = gnbPrms.numBbuLayers;
//             }
//             try
//             {
//                 tvPrms.cmnPrms.tbPrmsArray[j].userGroupIndex = tbConfig.get_value_as<uint32_t>("userGroupIndex");
//             }
//             catch(...)
//             {
//                 tvPrms.cmnPrms.tbPrmsArray[j].userGroupIndex = 0;
//             }
//         }
//     }
//     catch(const std::exception& exc)
//     {
//         printf("%s\n", exc.what());
//         throw exc;
//         // Continue using command line arguments if the input file does not
//         // have a config struct.
//     }

//     cuphy::unique_device_ptr<float>       d_input       = cuphy::make_unique_device<float>(inputLLRs.size());
//     cuphy::unique_device_ptr<const void*> d_inputs      = cuphy::make_unique_device<const void*>(gnbPrms.nUserGroups);
//     cuphy::unique_device_ptr<float>       d_output      = cuphy::make_unique_device<float>(outputLLRs.size());
//     cuphy::unique_device_ptr<PerTbParams> d_tbPrmsArray = cuphy::make_unique_device<PerTbParams>(gnbPrms.numTb);

//     std::vector<const void*> inputPtrs(gnbPrms.nUserGroups, (const void*)d_input.get());
//     for(int i = 1; i < gnbPrms.nUserGroups; i++)
//     {
//         for(int j = 0; j < gnbPrms.numTb; j++)
//         {
//             if(tvPrms.cmnPrms.tbPrmsArray[j].userGroupIndex < i)
//                 inputPtrs[i] = (void*)((float*)inputPtrs[i] + tbPrmsArray[j].numPrb * 12 * QAM_STRIDE * tvPrms.cmnPrms.tbPrmsArray[j].Nl * tvPrms.fePrms.Nd);
//         }
//     }

//     //  for(int j = 0; j < gnbPrms.numTb; j++)
//     //  {
//     //      if(tvPrms.cmnPrms.tbPrmsArray[j].userGroupIndex >= 1)
//     //         tvPrms.cmnPrms.tbPrmsArray[j].startLLR -= (uint32_t)((float*)inputPtrs[tvPrms.cmnPrms.tbPrmsArray[j].userGroupIndex] - (float*)inputPtrs[0]);

//     //    }
//     cudaMemcpy(d_inputs.get(), inputPtrs.data(), sizeof(void*) * inputPtrs.size(), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_input.get(), inputLLRs.data(), inputLLRs.size() * sizeof(float), cudaMemcpyHostToDevice);
//     cudaMemcpy(d_tbPrmsArray.get(), tvPrms.cmnPrms.tbPrmsArray.addr(), tvPrms.bePrms.nTb * sizeof(PerTbParams), cudaMemcpyHostToDevice);

//     cudaEvent_t start, stop;
//     cudaEventCreate(&start);
//     cudaEventCreate(&stop);
//     float time1 = 0.0;
//     cudaEventRecord(start);

//     for(int i = 0; i < N_ITER; i++)
//     {
//         rate_matching(
//             tvPrms.bePrms.CMax,
//             tvPrms.bePrms.EMax,
//             tvPrms.bePrms.nTb,
//             d_tbPrmsArray.get(),
//             d_inputs.get(),
//             d_output.get(),
//             0,
//             1,
//             0);
//     }

//     cudaEventRecord(stop);
//     cudaEventSynchronize(stop);
//     cudaEventElapsedTime(&time1, start, stop);

//     time1 /= N_ITER;

//     //printf(
//     //     "Rate Matching Kernel"
//     //     "\n %.2f us\n",
//     //     time1 * 1000);

//     cudaMemcpy(outputGPULLRs.data(), d_output.get(), outputLLRs.size() * sizeof(float), cudaMemcpyDeviceToHost);
//     bool res = 1;
//     for(int i = 0; i < outputGPULLRs.size(); i++)
//     {
//         if(outputGPULLRs[i] != outputLLRs[i])
//         {
//             std::cout << i << " GPU: " << outputGPULLRs[i] << " CPU: " << outputLLRs[i] << ": ERROR, not equal!"
//                       << "\n";
//             res = 0;
//         }
//     }
//     if(!res)
//     {
//         for(int i = 0; i < inputLLRs.size(); i++)
//         {
//             std::cout << "Input at " << i << ": " << inputLLRs[i] << "\n";
//         }
//     }
//     return res;
// }

// int RATE_MATCHING_TEST_1()
// {
//     std::string tv("TV_cuphy_perf-pusch-TC281_snrdb40.00_MIMO8x16_PRB272_DataSyms10_qam256.h5");
//     int         res = rmTest(tv);

//     return res;
// }

// int RATE_MATCHING_TEST_2()
// {
//     std::string tv("TV_cuphy_perf-pusch-TC231_snrdb40.00_MIMO1x4_PRB104_DataSyms13_qam256.h5");
//     int         res = rmTest(tv);

//     return res;
// }
// int RATE_MATCHING_TEST_3()
// {
//     std::string tv("TV_cuphy_perf-pusch-TC-FDM23_snrdb40.00_MIMO1x4_PRB24_DataSyms11_qam256.h5");
//     int         res = rmTest(tv);

//     return res;
// }
// int RATE_MATCHING_TEST_4()
// {
//     std::string tv("TV_cuphy_perf-pusch-TC-FDM28_snrdb40.00_MIMO8x16_PRB64_DataSyms10_qam256.h5");
//     int         res = rmTest(tv);

//     return res;
// }
// TEST(RATE_MATCHING, MU_MIMO_1) { EXPECT_EQ(RATE_MATCHING_TEST_1(), 1); }
// TEST(RATE_MATCHING, MU_MIMO_2) { EXPECT_EQ(RATE_MATCHING_TEST_2(), 1); }
// TEST(RATE_MATCHING, FDM_1) { EXPECT_EQ(RATE_MATCHING_TEST_3(), 1); }
// TEST(RATE_MATCHING, FDM_2) { EXPECT_EQ(RATE_MATCHING_TEST_4(), 1); }

// int main(int argc, char** argv)
// {
//     if(argc > 1)
//         cudaSetDevice(std::stoi(argv[1]));
//     if(argc > 2)
//         N_ITER = (std::stoi(argv[2]));
//     testing::InitGoogleTest(&argc, argv);
//     return RUN_ALL_TESTS();
// }
