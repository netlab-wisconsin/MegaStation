/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

 #if !defined(GEN_PRACH_PERF_CURVE_INCLUDED_)
#define GEN_PRACH_PERF_CURVE_INCLUDED_

void gen_prach_perf_curve(std::string prachInputFilename, std::string perfOutFilename, std::vector<float> snrVec, uint32_t nItrPerSnr, uint8_t seed);

#endif // GEN_PRACH_PERF_CURVE_INCLUDED_