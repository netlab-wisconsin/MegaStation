/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#if !defined(GEN_PUCCH_PERF_CURVE_INCLUDED_)
#define GEN_PUCCH_PERF_CURVE_INCLUDED_

void gen_pucch_perf_curve(std::string pucchInputFilename, std::string pucchDbgFilename, std::string perfOutFilename, uint8_t  formatType, std::vector<float> snrVec, uint32_t nItrPerSnr, uint8_t mode, uint8_t seed);

#endif // GEN_PUCCH_PERF_CURVE_INCLUDED_