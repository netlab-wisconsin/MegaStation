/*
 * Copyright (c) 2019-2023, NVIDIA CORPORATION & AFFILIATES.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#if !defined(GEN_PUSCH_BLER_CURVE_INCLUDED_)
#define GEN_PUSCH_BLER_CURVE_INCLUDED_


void gen_pusch_bler_curve(std::string puschInputFilename, std::string puschDbgFilename, std::string tberFilename, std::vector<float> snrVec, uint32_t nItrPerSnr, bool quickBlerFlag, uint32_t quickBlerNumTbErrs, uint8_t seed);


#endif // GEN_PUSCH_BLER_CURVE_INCLUDED_