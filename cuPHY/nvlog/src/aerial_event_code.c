/*
 * Copyright (c) 2021, NVIDIA CORPORATION.  All rights reserved.
 *
 * NVIDIA CORPORATION and its licensors retain all intellectual property
 * and proprietary rights in and to this software, related documentation
 * and any modifications thereto.  Any use, reproduction, disclosure or
 * distribution of this software and related documentation without an express
 * license agreement from NVIDIA CORPORATION is strictly prohibited.
 */

#include "aerial_event_code.h"

char* nvlog_strerror(int code)
{
    switch(code)
    {
    case AERIAL_SUCCESS: return "AERIAL_SUCCESS";
    // 1 ~ 5
    case AERIAL_INVALID_PARAM_EVENT: return "AERIAL_INVALID_PARAM_EVENT";
    case AERIAL_INTERNAL_EVENT: return "AERIAL_INTERNAL_EVENT";
    case AERIAL_CUDA_API_EVENT: return "AERIAL_CUDA_API_EVENT";
    case AERIAL_DPDK_API_EVENT: return "AERIAL_DPDK_API_EVENT";
    case AERIAL_THREAD_API_EVENT: return "AERIAL_THREAD_API_EVENT";
    // 6 ~ 10
    case AERIAL_CLOCK_API_EVENT: return "AERIAL_CLOCK_API_EVENT";
    case AERIAL_NVIPC_API_EVENT: return "AERIAL_NVIPC_API_EVENT";
    case AERIAL_ORAN_FH_EVENT: return "AERIAL_ORAN_FH_EVENT";
    case AERIAL_CUPHYDRV_API_EVENT: return "AERIAL_CUPHYDRV_API_EVENT";
    case AERIAL_INPUT_OUTPUT_EVENT: return "AERIAL_INPUT_OUTPUT_EVENT";
    // 11 ~ 15
    case AERIAL_MEMORY_EVENT: return "AERIAL_MEMORY_EVENT";
    case AERIAL_YAML_PARSER_EVENT: return "AERIAL_YAML_PARSER_EVENT";
    case AERIAL_NVLOG_EVENT: return "AERIAL_NVLOG_EVENT";
    case AERIAL_CONFIG_EVENT: return "AERIAL_CONFIG_EVENT";
    case AERIAL_FAPI_EVENT: return "AERIAL_FAPI_EVENT";
    // 16 ~ 20
    case AERIAL_NO_SUPPORT_EVENT: return "AERIAL_NO_SUPPORT_EVENT";
    case AERIAL_SYSTEM_API_EVENT: return "AERIAL_SYSTEM_API_EVENT";
    case AERIAL_L2ADAPTER_EVENT: return "AERIAL_L2ADAPTER_EVENT";
    case AERIAL_RU_EMULATOR_EVENT: return "AERIAL_RU_EMULATOR_EVENT";
    case AERIAL_CUDA_KERNEL_EVENT: return "AERIAL_CUDA_KERNEL_EVENT";
    // 21 ~
    case AERIAL_CUPHY_API_EVENT: return "AERIAL_CUPHY_API_EVENT";
    case AERIAL_DOCA_API_EVENT: return "AERIAL_DOCA_API_EVENT";
    case AERIAL_CUPHY_EVENT: return "AERIAL_CUPHY_EVENT";
    // Other: unknown
    default: return "AERIAL_UNKNOWN";
    }
}
